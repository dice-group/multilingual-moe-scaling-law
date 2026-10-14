from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import data
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as m
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import replay
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.campaign import registered_tasks
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.campaign import render_array
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.campaign import accounted_reservations
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.protocol import INNER
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.protocol import PROTOCOL
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.protocol import check_budget
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.protocol import choose_procedure
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.reporting import checked_score
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.reporting import coefficients_predict
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.worker import best_attempt
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.worker import geometry
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.reproduction import shared_equation
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.driver import stage_state
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.driver import final_accounting
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import campaign
from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import recovery
from tests.test_stage1_repetition import fixture


def source_fixture():
    p,u,x = fixture()
    x = np.r_[x,.4]
    D = np.column_stack((p.D,p.D_family,p.D_script,p.D_other))
    U = np.column_stack((u,u*.2,u*.4,u*2))
    family,script = np.zeros((4,4)),np.zeros((4,4))
    family[0,1],script[0,2] = 1.,1.
    sources = m.Sources(D,U,p.N,p.N*8,np.arange(len(p.y)),np.zeros(len(p.y),int),family,script)
    return p,u,x,sources


class LawSearchTest(unittest.TestCase):
    def test_all_candidate_derivatives(self):
        p,u,x,sources=source_fixture()
        for name in m.CANDIDATES:
            cap,term,cj,dj=m.candidate_terms(name,x,p,u,sources)
            for i in range(17):
                step=np.eye(17)[i]*1e-6
                a,b=m.candidate_terms(name,x+step,p,u,sources),m.candidate_terms(name,x-step,p,u,sources)
                np.testing.assert_allclose((a[0]-b[0])/2e-6,cj[:,i],rtol=3e-6,atol=1e-8)
                np.testing.assert_allclose((a[1]-b[1])/2e-6,dj[:,i],rtol=3e-6,atol=1e-8)

    def test_source_sums_and_exact_limits(self):
        p,u,x,sources=source_fixture()
        null=x.copy(); null[15]=0.
        effective,*_=sources.effective(null)
        np.testing.assert_allclose(effective,p.D+x[11]*p.D_family+x[12]*p.D_script,rtol=1e-14)
        no_composition=x.copy();no_composition[11:13]=0.
        a=m.candidate_terms("RN",no_composition,p,u,sources)
        b=m.candidate_terms("SOURCE-RN",no_composition,p,u,sources)
        for i in (0,1): np.testing.assert_allclose(a[i],b[i],rtol=1e-14)
        zero=x.copy();zero[16]=0.
        a=m.candidate_terms("SOURCE-RN",zero,p,u,sources)
        b=m.candidate_terms("SOURCE-RP",zero,p,u,sources)
        for i in (0,1): np.testing.assert_array_equal(a[i],b[i])
        # Repeating donors identify lambda even when every target is below U.
        target_only_u=p.D*2
        donor_sources=replace(sources,U=np.column_stack((target_only_u,sources.U[:,1:])))
        bounds=m.candidate_bounds("SOURCE-RN",p,target_only_u,donor_sources)
        self.assertLess(bounds[15].lower,bounds[15].upper)
        self.assertEqual(m.candidate_bounds("RN",p,target_only_u)[15].upper,0.)

    def test_independent_numpy_candidate_replay(self):
        p,u,x,sources=source_fixture()
        rows=pd.DataFrame(dict(N_act=p.N,E=p.E,K=p.K,D_t=p.D,U_t=u,D_family=p.D_family,D_script=p.D_script,D_other=p.D_other))
        for name in m.CANDIDATES:
            actual=m.candidate_terms(name,x,p,u,sources)
            independent=replay.candidate(name,x,rows,sources)
            for i in (0,1): np.testing.assert_allclose(actual[i],independent[i],rtol=0,atol=1e-12)

    def test_rho_penalty_derivative_and_nulls(self):
        p,u,x,sources=source_fixture()
        candidate=m.Model("RN",p,u,sources,tau=.1)
        value,gradient=candidate.penalty(x)
        for i in range(17):
            step=np.eye(17)[i]*1e-6
            self.assertAlmostEqual((candidate.penalty(x+step)[0]-candidate.penalty(x-step)[0])/2e-6,gradient[i],places=9)
        p=replace(p,K=np.full(len(p.y),12.))
        self.assertEqual(m.Model("RN",p,u,sources).bounds[10].upper,0.)
        self.assertEqual(sum(b.upper>b.lower for b in m.Model("RN-no-composition",p,u,sources).bounds),13)

    def test_stationarity_never_treats_infinity_as_active_bound(self):
        p,u,x,sources=source_fixture()
        candidate=m.Model("C0",p,u,sources)
        bounds=[m.repetition.Bound("x",-np.inf,np.inf,"signed")]
        self.assertEqual(m.stationarity([0.],[1.],bounds,candidate),1.)
        bounds=[m.repetition.Bound("x",0.,1.,"nonnegative")]
        self.assertAlmostEqual(m.stationarity([0.],[1.],bounds,candidate),0.)
        self.assertEqual(m.stationarity([0.],[-1.],bounds,candidate),1.)

    def test_synthetic_source_projected_direct_fit(self):
        p,u,x,sources=source_fixture()
        cap,term,_,_=m.candidate_terms("SOURCE-RN",x,p,u,sources)
        p=replace(p,y=-5.+.3*cap+.4*term,regularization=fit.M5M6Regularization(0.,.001))
        candidate=m.Model("SOURCE-RN",p,u,sources)
        candidate.bounds=[replace(b,lower=float(v),upper=float(v)) for b,v in zip(candidate.bounds,x)]
        candidate.bounds[15]=replace(candidate.bounds[15],lower=.01,upper=1.)
        candidate.bounds[16]=replace(candidate.bounds[16],lower=-1.,upper=1.)
        start=x.copy();start[15:]=[.15,.2]
        profile=m.Profile(candidate,.15)
        selected=profile.run(start)
        self.assertTrue(selected["success"],selected)
        direct=m.Direct(profile,selected).run()
        self.assertTrue(direct["success"],direct)
        np.testing.assert_allclose(m.predict(candidate,selected),m.predict(candidate,direct),atol=1e-5,rtol=0)
        np.testing.assert_allclose(m.predict(candidate,selected),p.y,atol=1e-5,rtol=0)
        self.assertIn("nuisance_adjusted_rank",geometry(profile,selected))

    def test_selector_complete_and_simplicity_guard(self):
        records=[dict(model=n,tau=t,fold=f,certified=True,complete=True,
                      rmse=1. if n=="RN" else .995 if n=="RN-no-composition" else 1.5)
                 for n in m.CANDIDATES for t in m.TAUS for f in INNER]
        self.assertEqual(choose_procedure(records)["chosen"]["model"],"RN-no-composition")
        with self.assertRaises(ValueError):choose_procedure(records[:-1])
        with self.assertRaises(ValueError):choose_procedure([records[0]|{"certified":False},*records[1:]])
        with self.assertRaises(ValueError):choose_procedure([records[0],*records])
        for r in records:
            if r["model"]=="RN-no-composition":r["rmse"]=1.03 if r["fold"]==INNER[0] else .98
        self.assertEqual(choose_procedure(records)["chosen"]["model"],"RN")

    def test_low_uncertified_objective_blocks_certification(self):
        good=dict(index=0,success=True,objective=1.)
        self.assertIsNone(best_attempt([good,dict(index=1,success=False,objective=.9)]))
        self.assertEqual(best_attempt([good])["index"],0)

    def test_score_coverage_and_configuration_weights(self):
        rows=pd.DataFrame(dict(trajectory_id=["a","a","b"],readout_id=["1","2","3"],language=["x"]*3,adjusted_loss=[0.]*3))
        predicted=rows.assign(prediction=[1.,1.,3.])
        self.assertAlmostEqual(checked_score(predicted,rows)["rmse"],np.sqrt(5.))
        self.assertFalse(checked_score(predicted.iloc[:2],rows)["complete"])
        self.assertFalse(checked_score(predicted.assign(prediction=[1.,1.,np.nan]),rows)["complete"])
        with self.assertRaises(ValueError):checked_score(pd.concat([predicted,predicted.iloc[:1]]),rows)

    def test_split_membership_isolation(self):
        frame=pd.DataFrame(dict(trajectory_id=["a","a","b"],readout_id=["1","2","3"],language=["x"]*3,
                                adjusted_loss=[-1.,-2.,-3.]))
        member=data.membership(frame.iloc[:1],frame.iloc[1:])
        altered=frame.copy();altered.loc[1:,"adjusted_loss"]=999.
        pd.testing.assert_frame_equal(data.subset(frame,member,"train"),data.subset(altered,member,"train"))
        unrestricted=member | {"train_readout_ids":[]}
        self.assertEqual(len(data.subset(frame,unrestricted,"train")),2)
        with self.assertRaises(ValueError):data.membership(frame,frame.iloc[:1])
        with self.assertRaises(ValueError):data.source_arrays(frame,frame,{}, {})

    def test_budget_and_cpu_only_array(self):
        tasks=registered_tasks([dict(name=f"native-{i}") for i in range(42)])
        amounts=check_budget(tasks)
        self.assertLessEqual(sum(amounts.values()),512.)
        with self.assertRaises(ValueError):check_budget(tasks+[dict(phase="selection",cores=1,minutes=600)])
        with self.assertRaises(ValueError):check_budget([dict(phase="pilot",cores=8,minutes=1)])
        plan={"profile":dict(python_executable="/env/bin/python",account="fixture",partition="cpu",memory="4G")}
        selected=[t for t in tasks if t["stage"]=="inner"]
        script=render_array(Path("/tmp/law search/plan.json"),plan,selected,"test")
        self.assertIn("%32",script)
        self.assertNotIn("--gres",script)
        self.assertNotIn("sbatch ",script)
        subprocess.run(["bash","-n"],input=script,text=True,check=True)

    def test_all_native_equations_replay(self):
        packet=dict(D=[[4e9,1e9,2e9,3e9,6e9],[5e9,1e9,2e9,4e9,7e9]],
            U=[[1e9]*5,[1e9]*5],N=[2e8,5e8],N_total=[3e8,2e9],E=[4.,32.],K=[50.,200.],
            family_share=[.3,.5],training_minimum=2.)
        for spec in m.native.panel():
            lo,hi=m.native.bounds(spec)
            x=np.where(np.isfinite(hi),(lo+hi)/2,.5)
            for i,n in enumerate(m.native.names(spec)):
                if n.startswith("tau_"):x[i]=.1
            actual=m.native.predict(spec,x,packet)[0]
            np.testing.assert_allclose(actual,replay.native(spec.payload(),x,packet),rtol=0,atol=1e-10)

    def test_accounting_rejects_missing_or_larger_allocation(self):
        plan={"tasks":[dict(id="fit",phase="pilot",minutes=60)]}
        receipts=[dict(job_id="123",task_ids=["fit"])]
        with self.assertRaises(ValueError):accounted_reservations(plan,receipts,[])
        row=dict(job_id="123_0",cores=1,seconds=60)
        self.assertEqual(accounted_reservations(plan,receipts,[row])[0]["allocated_core_hours"],1.)
        with self.assertRaises(ValueError):accounted_reservations(plan,receipts,[row|{"cores":2}])

    def test_extreme_native_replay_keeps_absolute_tolerance(self):
        spec=m.native.Spec("extreme","full",signed=True,tau_cap=10000.,lambda_cap=100.)
        packet=dict(D=[[1e-4,1.,1.,1.,1.]],U=[[1e4]*5],N=[2e8],N_total=[2e9],E=[32.],K=[200.],training_minimum=2.)
        x=np.array([.88,.518,7.185,.574,.389,8.218,0.,0.,0.,0.])
        a=m.native.predict(spec,x,packet)[0]
        self.assertGreater(a[0],1e5)
        np.testing.assert_allclose(a,replay.native(spec.payload(),x,packet),rtol=0,atol=1e-10)

    def test_independent_shared_baseline_equations(self):
        from dataclasses import asdict
        from multilingual_megatron.experiments.scaling.stage1_fit.law_search.protocol import SHARED
        p,u,x,sources=source_fixture()
        rows=pd.DataFrame(dict(N_act=p.N,E=p.E,K=p.K,D_t=p.D,D_family=p.D_family,
                               D_script=p.D_script,D_other=p.D_other,language=p.language_ids))
        for name in SHARED:
            candidate=m.Model(name,p,u,sources)
            coords=m.starts(candidate)[0]
            long=name in ("M6","M6-ND","M5-ND","J0-ND")
            parameters=fit.shared_from_outer("M6" if long else name,coords[:14] if long else coords)
            record=dict(shared_parameters=asdict(parameters),zeta=float(coords[14]) if long else 0.,
                        language_coefficients=[dict(language="x",L_inf=-5.,A=.3,B=.4)])
            cap,term,_,_=candidate.terms(coords)
            np.testing.assert_allclose(shared_equation(record,rows),-5.+.3*cap+.4*term,rtol=0,atol=1e-10)

    def test_driver_distinguishes_missing_uncertified_and_foreign_receipts(self):
        plan=dict(sha256="registered",tasks=[dict(id="fit",stage="inner")])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.assertFalse(stage_state(plan,root,"inner")["all_finished"])
            io.write_json_once(root/"results/fit.json",dict(plan_sha256="registered",task_id="fit",certified=False))
            state=stage_state(plan,root,"inner")
            self.assertTrue(state["all_finished"])
            self.assertEqual(state["failed"],["fit"])
            with self.assertRaises(ValueError):stage_state(plan|{"sha256":"foreign"},root,"inner")

    def test_new_shared_and_pooled_fits_require_independent_equation_replay(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.law_search.protocol import SHARED
        p,u,x,sources=source_fixture()
        rows=pd.DataFrame(dict(N_act=p.N,E=p.E,K=p.K,D_t=p.D,D_family=p.D_family,
            D_script=p.D_script,D_other=p.D_other,language=p.language_ids,U_t=u,
            adjusted_loss=p.y,trajectory_id=[str(i) for i in range(len(p.y))],unigram_nll=6.))
        packet=dict(D=np.column_stack((sources.D,p.D)).tolist(),U=np.column_stack((sources.U,u)).tolist(),
            N=p.N.tolist(),N_total=(p.N*8).tolist(),E=p.E.tolist(),K=p.K.tolist(),training_minimum=2.)
        packets=[dict(indices=list(range(len(rows))),train=packet)]
        for name in (*SHARED,*(s.name for s in m.POOLED)):
            candidate=data.make_model(name,rows,sources,packets,p.regularization.as_dict(),0.,.1)
            coords=m.starts(candidate)[0]
            result=dict(model=name,regularization=p.regularization.as_dict(),tau=0.,delta=.1,
                selected=dict(coordinates=coords,language_coefficients=[dict(language="x",L_inf=-5.,A=.3,B=.4)]))
            expected=m.predict(candidate,result["selected"])
            np.testing.assert_array_equal(coefficients_predict(result,rows,sources,packets),expected)
            with patch.object(m,"predict",return_value=expected+1e-4):
                with self.assertRaises(AssertionError):coefficients_predict(result,rows,sources,packets)

    def test_native_box_repair_is_limited_to_eight_ulps(self):
        spec=m.native.Spec("fixture","other")
        lo,hi=m.native.bounds(spec)
        x=np.array([lo[0],.5,.5,.3,.3,.1,.1])
        self.assertIsNone(recovery.canonical_box(spec,x))
        x[0]=np.nextafter(lo[0],0.)
        corrected=recovery.canonical_box(spec,x)
        self.assertEqual(corrected[0],lo[0])
        np.testing.assert_array_equal(corrected[1:],x[1:])
        x[0]=lo[0]-16*np.spacing(lo[0])
        self.assertIsNone(recovery.canonical_box(spec,x))
        with patch.object(recovery.n,"direct",side_effect=AssertionError("not a roundoff repair")):
            result=recovery.repair_group(dict(certified=False,spec=spec.payload(),attempts=[
                dict(optimizer_success=True,coordinates=x.tolist())]),{})
        self.assertFalse(result["certified"])

    def test_roundoff_repair_keeps_independent_audit_gate(self):
        spec=m.native.Spec("fixture","other")
        lo,_=m.native.bounds(spec)
        x=np.array([np.nextafter(lo[0],0.),.5,.5,.3,.3,.1,.1])
        group=dict(certified=False,spec=spec.payload(),attempts=[
            dict(optimizer_success=True,coordinates=x.tolist(),index=0)])
        record=dict(success=True,index=0,coordinates=np.maximum(x,lo).tolist(),objective=.01)
        with patch.object(recovery.n,"record",return_value=record), \
             patch.object(recovery.n,"direct",return_value=dict(success=False)) as audit:
            result=recovery.repair_group(group,{})
        self.assertFalse(result["certified"])
        audit.assert_called_once()
        self.assertEqual(result["box_recovery"]["status"],"independent_audit_failed")

    def test_native_physical_certificate_rejects_then_repairs_one_ulp(self):
        spec=m.native.Spec("fixture","other")
        lo,_=m.native.bounds(spec)
        x=np.array([lo[0],.5,.5,.3,.3,.1,.1])
        packet=dict(D=[[4e9,1e9,2e9,3e9,6e9],[5e9,1e9,2e9,4e9,7e9]],
            U=[[1e9]*5,[1e9]*5],N=[2e8,5e8],N_total=[3e8,2e9],E=[4.,32.],K=[50.,200.],
            family_share=[.3,.5],training_minimum=2.,weights=[.5,.5],delta=.1)
        packet["y"]=m.native.predict(spec,x,packet)[0].tolist()
        outside=x.copy();outside[0]=np.nextafter(lo[0],0.)
        kwargs=dict(kind="fixture",index=0,flag=True,message="synthetic physical audit",runtime=0.)
        original=recovery.n.record(spec,outside,packet,packet,**kwargs)
        corrected=recovery.canonical_box(spec,outside)
        repaired=recovery.n.record(spec,corrected,packet,packet,**kwargs)
        self.assertFalse(original["success"])
        self.assertLessEqual(original["kkt"],1e-8)
        self.assertTrue(repaired["success"])
        self.assertEqual(original["objective"],repaired["objective"])
        np.testing.assert_array_equal(m.native.predict(spec,outside,packet)[0],
                                      m.native.predict(spec,corrected,packet)[0])

    def test_roundoff_recovery_fits_original_phase_budget(self):
        tasks=registered_tasks([dict(name=f"native-{i}") for i in range(42)])
        extra=[dict(phase="baselines",cores=1,minutes=15) for _ in range(42)]
        self.assertEqual(sum(check_budget(tasks+extra).values()),461.5)
        self.assertLessEqual(check_budget(tasks+extra)["baselines"],200.)

    def test_final_accounting_includes_last_wave_and_is_resumable(self):
        plan=dict(sha256="registered",tasks=[dict(id="fit",phase="stability",minutes=60)])
        row=dict(job_id="123_0",state="COMPLETED",cores=1,seconds=60)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            io.write_json_once(root/"submissions/000.json",dict(job_id="123",task_ids=["fit"]))
            with patch.object(campaign,"scheduler_records",return_value=[row]):
                result=final_accounting(plan,root,campaign)
                self.assertEqual(result["charged_core_hours"],1.)
                self.assertAlmostEqual(result["measured_allocated_core_hours"],1./60.)
                self.assertEqual(final_accounting(plan,root,campaign),result)
            with patch.object(campaign,"scheduler_records",return_value=[row|{"state":"RUNNING"}]):
                with self.assertRaises(ValueError):final_accounting(plan,root,campaign)
            with patch.object(campaign,"scheduler_records",return_value=[row|{"seconds":120}]):
                with self.assertRaises(ValueError):final_accounting(plan,root,campaign)


if __name__ == "__main__":
    unittest.main()
