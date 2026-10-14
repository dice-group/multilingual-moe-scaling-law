"""Equation, numerical, and leakage contracts for the frozen ATLAS panel."""
import copy
from dataclasses import replace
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import atlas_full_model as old
from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import model as m, numerics as n
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.campaign import packets_for


def fixture(count=40):
    rng = np.random.default_rng(20260906)
    return dict(D=np.exp(rng.uniform(np.log(1e7), np.log(1e10), (count, 5))),
        U=np.full((count, 5), 1e9), N=np.exp(rng.uniform(np.log(1e8), np.log(1e9), count)),
        N_total=np.exp(rng.uniform(np.log(1e8), np.log(1e10), count)),
        E=rng.choice([1, 4, 16, 32], count), K=rng.choice([12, 50, 200], count),
        family_share=rng.uniform(.01, .9, count), training_minimum=3.,
        weights=np.ones(count), delta=.1)


class AtlasComparisonTests(unittest.TestCase):
    def test_extended_precision_equation_gradient_and_hessian_parity(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import precision_numerics as p, recovery_numerics as r
        packet=fixture(12); packet['y']=np.linspace(2.,4.,12); spec=m.Spec('test','full',signed=True)
        for rate in (0.,.0001,.1,10.):
            x=np.array([.3,.7,.4,.3,.4,rate,.2,-.001,.3,.2])
            extended=p.Evaluator(spec,packet).evaluate(x,hessian=True); original=n.objective(spec,x,packet)
            # The original exp-based saturation derivative loses digits near
            # its small-rate branch. This is a parity bound, not a relaxation
            # of the unchanged 1e-8 production KKT acceptance threshold.
            for a,b in zip(extended[:4],original): np.testing.assert_allclose(np.asarray(a,float),b,rtol=1e-8,atol=1e-9)
            np.testing.assert_allclose(np.asarray(extended[4],float),r.analytic_hessian(spec,x,packet),rtol=1e-8,atol=1e-10)
            high=p.mp_objective(spec,x,packet,60)
            self.assertAlmostEqual(high[0],float(extended[0]),places=13)
            np.testing.assert_allclose(high[1],np.asarray(extended[1],float),rtol=1e-11,atol=1e-12)

    def test_whitening_is_invertible_without_discarding_weak_directions(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import precision_numerics as p
        rng=np.random.default_rng(7); jac=rng.normal(size=(30,5)); jac[:,4]=jac[:,3]+1e-6*jac[:,4]
        transform=p.whitening(jac,np.ones(30)/30,np.arange(5))
        self.assertEqual(np.linalg.matrix_rank(transform),5)
        normalized=jac@transform/np.sqrt(30)
        np.testing.assert_allclose(normalized.T@normalized,np.eye(5),atol=1e-8)

    def test_precision_newton_preserves_a_known_training_solution(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import precision_numerics as p
        packet=fixture(); spec=m.Spec('test','full',signed=True)
        x=np.array([.3,.7,.4,.3,.4,.1,.2,.1,.3,.2]); packet['y']=m.predict(spec,x,packet)[0]
        result=p.newton(spec,x,packet,packet)
        self.assertTrue(result['success']); self.assertLessEqual(result['kkt'],1e-8)
        np.testing.assert_allclose(result['coordinates'],x,rtol=0,atol=1e-12)

    def test_recovery_requires_distinct_solvers_and_no_hidden_better_point(self):
        from unittest.mock import patch
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import recovery_numerics as r
        spec=m.Spec("test","target")
        a=dict(success=True,objective=.001,kind="physical_newton_polish")
        b=dict(success=True,objective=.001,kind="joint_scaled_sqp")
        with patch.object(n,"agreement",return_value=True):
            self.assertIsNotNone(r.pair(spec,[a,b],{},.001)[1])
            self.assertIsNone(r.pair(spec,[a,b],{},.0009)[1])
            self.assertIsNone(r.pair(spec,[a,a.copy()],{},.001)[1])
            self.assertIsNone(r.pair(spec,[a,b|{"kind":"start"}],{},.001)[1])
            self.assertIsNone(r.pair(spec,[a,b|{"success":False}],{},.001)[1])
        self.assertNotEqual(r.family(dict(kind="direct",solver="dogbox")),r.family(dict(kind="direct")))

    def test_recovery_polish_keeps_original_equation_and_physical_gradient(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import recovery_numerics as r
        p=fixture(); spec=m.Spec("test","target")
        x=np.array([.3,.7,.4,.3,.4]); p["y"]=m.predict(spec,x,p)[0]
        result=r.polish(spec,x+np.array([1e-7,2e-7,-1e-7,1e-7,1e-7]),p,p)
        self.assertTrue(result["success"])
        self.assertLessEqual(result["kkt"],1e-8)
        self.assertLess(result["objective"],1e-13)
        np.testing.assert_allclose(m.predict(spec,result["coordinates"],p)[0],p["y"],atol=1e-6)

    def test_recovery_seeds_ignore_outcome_scores_and_filter_invalid_records(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import recovery_numerics as r
        rows=[dict(coordinates=[i],objective=float(i)) for i in (3,1,2)]
        value=dict(selected=rows[0],audit=rows[1],attempts=[rows[1],rows[2],{"error":"bad"}])
        self.assertEqual([v["objective"] for v in r.seed_points(value)], [1.,2.])
        self.assertEqual(r.seed_points(value),r.seed_points(value|{"heldout_rmse":-1e99}))

    def test_recovery_end_to_end_on_synthetic_training_packet(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import recovery_numerics as r
        p=fixture(); spec=m.Spec("test","target"); x=np.array([.3,.7,.4,.3,.4])
        p["y"]=m.predict(spec,x,p)[0]
        old=n.record(spec,x,p,p,kind="start",index=0,flag=True,message="synthetic",runtime=0.)
        previous=dict(selected=old,audit={"success":False},attempts=[])
        invalid=dict(spec=spec.payload(),point=old|{"coordinates":[-1.]*len(x),"objective":-1e99})
        value=r.recover(spec,p,p,previous,[invalid])
        self.assertTrue(value["certified"])
        self.assertFalse(value["heldout_scores_computed"])
        self.assertTrue(any(f.startswith("joint_") for f in value["recovery"]["independent_families"]))
        self.assertGreaterEqual(value["recovery"]["minimum_feasible_objective"],0.)

    def test_recovery_budget_and_scientific_contract_unchanged(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import campaign as c, recovery_campaign as r
        amended=r.protocol(c.PROTOCOL)
        for key in ("panel","folds","numerics","signed_domain","data","loss","family","selection","evidence","bootstrap_seed","bootstrap_replicates"):
            self.assertEqual(amended[key],c.PROTOCOL[key])
        self.assertLess(r.check_budget(83.07),250.)
        with self.assertRaises(ValueError): r.check_budget(120.)
        with self.assertRaises(ValueError): r.check_budget(float("nan"))
        self.assertEqual(r.failure_class({"selected":None}),"no_selected")

    def test_full_atlas_analytic_hessian_matches_original_gradient(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import recovery_numerics as r
        p=fixture(); p['y']=np.linspace(2.,4.,len(p['N'])); spec=m.Spec('test','full',signed=True)
        for rate in (0.,.0001,.1,3.):
            x=np.array([.3,.7,.4,.3,.4,rate,.2,.1,.3,.2])
            h=r.analytic_hessian(spec,x,p)
            numerical=np.empty_like(h)
            for j in range(len(x)):
                a=x.copy(); b=x.copy(); step=1e-6; a[j]+=step; b[j]-=step
                if j==5 and rate==0: b[j]=0.
                numerical[:,j]=(n.objective(spec,a,p)[1]-n.objective(spec,b,p)[1])/(a[j]-b[j])
            np.testing.assert_allclose(h,numerical,rtol=2e-4,atol=3e-6)

    def test_consolidation_rechecks_all_feasible_points_and_does_not_fit(self):
        from unittest.mock import patch
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import recovery_merge as r
        p=fixture(); spec=m.Spec('test','target'); x=np.array([.3,.7,.4,.3,.4]); p['y']=m.predict(spec,x,p)[0]
        best=n.record(spec,x,p,p,kind='start',index=0,flag=True,message='fixture',runtime=0.)
        partner=best|dict(kind='direct',solver='dogbox')
        old=dict(selected=best,audit={'success':False},attempts=[best],probes=[])
        with patch.object(n,'direct',side_effect=AssertionError('must not fit')):
            merged=r.consolidate(spec,dict(train=p,support=p),old,[partner])
        self.assertTrue(merged['certified']); self.assertFalse(merged['recovery']['additional_fitting'])
        self.assertEqual(merged['recovery']['minimum_feasible_objective'],0.)

    def test_panel_unique_and_complete(self):
        panel = m.panel()
        self.assertEqual(len(panel), 65)
        self.assertEqual(len(set(s.name for s in panel)), 65)
        self.assertEqual({s.tau_cap for s in panel}, {100., 1000., 10000.})
        self.assertEqual({s.lambda_cap for s in panel}, {10., 100.})

    def test_native_panel_only_defers_constructed_extension(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import campaign as c
        self.assertEqual(len(c.PANEL), 53)
        self.assertEqual(len(c.DEFERRED), 12)
        self.assertTrue(all(s.form == "moe" for s in c.DEFERRED))
        self.assertEqual(set(c.PANEL) | set(c.DEFERRED), set(m.panel()))
        self.assertEqual(c.PROTOCOL["pilot_allocated_core_hours"], 0.)
        self.assertEqual(c.PROTOCOL["main_allocated_core_hours"], 1024*13/60)
        self.assertEqual(c.PROTOCOL["max_concurrent_cores"], 1024)
        self.assertEqual(c.PROTOCOL["maximum_total_core_hours"], 250.)

    def test_total_budget_counts_prior_runs_and_final_audit(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import campaign as c
        c.check_budget(4.555833333333333)
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import array_execution as arrays
        allocated = sum(arrays.allocated_core_hours(a) for a in c.PROTOCOL["allocations"].values())
        self.assertAlmostEqual(allocated, 1024*13/60+2)
        self.assertEqual(arrays.concurrent_cores(c.PROTOCOL["allocations"]["main"]), 1024)
        self.assertNotIn("pilot", c.PROTOCOL["allocations"])
        with self.assertRaises(ValueError): c.check_budget(250-allocated)
        with self.assertRaises(ValueError): c.check_budget(-1.)
        with self.assertRaises(ValueError): c.check_budget(float("nan"))

    def test_budget_amendment_rejects_scientific_changes(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import campaign as c, reuse
        before = copy.deepcopy(c.PROTOCOL)
        before.update(version="atlas_comparison/v6_native", main_allocated_core_hours=44., maximum_total_core_hours=50.)
        reuse.budget_protocol_parity(before, c.PROTOCOL)
        before["numerics"]["gradient_tolerance"] = 1e-7
        with self.assertRaises(ValueError): reuse.budget_protocol_parity(before, c.PROTOCOL)

    def test_budget_inheritance_preserves_uncertified_receipts(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import reuse
        value = dict(sha256="old receipt", plan_sha256="old plan", task=16,
            packet_sha256="packet", spec={"name": "signed"}, heldout_scores_computed=False,
            certified=False, selected={"objective": .001}, runtime_seconds=74.)
        args = (value, {"sha256": "old plan"}, 16, {"sha256": "packet"}, {"name": "signed"}, {"path": "receipt"})
        result = reuse.budget_receipt(*args, certified=False)
        self.assertFalse(result["certified"])
        self.assertFalse(result["inherited_from"]["certified"])
        self.assertEqual(result["selected"], value["selected"])
        self.assertEqual(result["runtime_seconds"], 74.)
        self.assertNotIn("sha256", result)
        self.assertNotIn("plan_sha256", result)
        with self.assertRaises(ValueError): reuse.budget_receipt(*args, certified=True)
        value["heldout_scores_computed"] = True
        with self.assertRaises(ValueError): reuse.budget_receipt(*args, certified=False)

    def test_array_shards_are_disjoint_complete_and_do_not_rerun_failures(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import array_execution as arrays
        panel = [m.Spec("fast", "target").payload(), m.Spec("slow", "full", signed=True).payload()]
        inherited = [dict(task=0, spec=panel[0], runtime_seconds=1., certified=True),
                     dict(task=1, spec=panel[1], runtime_seconds=100., certified=False)]
        result = arrays.partition(100, panel, inherited, 8)
        tasks = [task for shard in result["tasks"] for task in shard]
        self.assertEqual(sorted(tasks), list(range(2, 100)))
        self.assertEqual(len(tasks), len(set(tasks)))
        self.assertEqual(result, arrays.partition(100, panel, inherited, 8))
        self.assertLessEqual(max(result["estimated_cpu_seconds"])-min(result["estimated_cpu_seconds"]), 100.)
        with self.assertRaises(ValueError): arrays.partition(100, panel, inherited*2, 8)
        with self.assertRaises(ValueError): arrays.partition(99, panel, inherited, 8)

    def test_array_requires_exact_runtime_dimensions_and_unique_index(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import array_execution as arrays
        allocation = dict(cores=8, jobs=128, parallel_jobs=128, minutes=13)
        environ = dict(SLURM_ARRAY_JOB_ID="123", SLURM_ARRAY_TASK_COUNT="128",
            SLURM_ARRAY_TASK_MIN="0", SLURM_ARRAY_TASK_MAX="127", SLURM_ARRAY_TASK_ID="9")
        self.assertEqual(arrays.array_index(allocation, environ), 9)
        with self.assertRaises(ValueError): arrays.array_index(allocation, {})
        with self.assertRaises(ValueError): arrays.array_index(allocation, environ | {"SLURM_ARRAY_TASK_ID": "128"})
        with self.assertRaises(ValueError): arrays.array_index(allocation, environ | {"SLURM_ARRAY_TASK_COUNT": "127"})
        self.assertIsNone(arrays.array_index({"cores": 1}, {}))

    def test_pilot_remapping_keeps_target_and_equation(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import campaign as c, reuse
        old, new = [s.payload() for s in m.panel()], [s.payload() for s in c.PANEL]
        for packet in (0, 200, 449):
            for index, spec in enumerate(old):
                task = reuse.remap_task(packet*65+index, old, new)
                if spec["form"] == "moe":
                    self.assertIsNone(task)
                else:
                    self.assertEqual(divmod(task, 53)[0], packet)
                    self.assertEqual(new[task % 53], spec)
        corrupted = copy.deepcopy(new); corrupted[0]["parameter_count"] = "total"
        with self.assertRaises(ValueError): reuse.remap_task(0, old, corrupted)
        with self.assertRaises(ValueError): reuse.remap_task(0, old, new[1:])

    def test_pilot_packet_parity_rejects_byte_or_population_changes(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import reuse
        old = [{"file_sha256": "a"}, {"file_sha256": "b"}]
        reuse.packet_parity(old, copy.deepcopy(old))
        with self.assertRaises(ValueError): reuse.packet_parity(old, old[:1])
        with self.assertRaises(ValueError): reuse.packet_parity(old, old[::-1])

    def test_reduced_panel_claim_explicitly_defers_moe(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.reporting import manuscript
        text = manuscript({}, {"panel_complete": True, "robust_advantage": True}, 0)
        self.assertIn("defers the 12", text)
        self.assertIn("no comparison against them is claimed", text)

    def test_snapshot_survives_live_edit_but_rejects_frozen_drift(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import snapshot
        from multilingual_megatron.experiments.scaling.stage1_fit import atlas_full_pilot as io
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary); source = repo/"src/example.py"
            source.parent.mkdir(); source.write_text("value = 1\n")
            output = repo/"campaign"
            runtime, bindings, archive = snapshot.freeze(output, [source], repo)
            source.write_text("value = 2\n")
            io.verify(bindings[0]); io.verify(archive)
            with tarfile.open(archive["path"]) as tar:
                self.assertEqual(tar.extractfile("src/example.py").read(), b"value = 1\n")
            Path(bindings[0]["path"]).write_text("value = 3\n")
            with self.assertRaises(ValueError): io.verify(bindings[0])
            with self.assertRaises(ValueError): snapshot.require_frozen_module({"runtime_root": str(runtime)}, source)

    def test_budget_snapshot_reads_prior_private_scientific_sources(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import snapshot
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary); source = repo/"science.py"; private = repo/"prior.py"
            source.write_text("unrelated newer edit\n"); private.write_text("frozen science\n")
            _, bindings, _ = snapshot.freeze(repo/"new", [source], repo, content_sources={source: private})
            self.assertEqual(Path(bindings[0]["path"]).read_text(), "frozen science\n")
            self.assertEqual(bindings[0]["original_path"], str(source))

    def test_all_equation_gradients(self):
        p = fixture()
        for form in ("target", "chinchilla", "joint", "capacity", "he", "full", "other", "moe"):
            spec = m.Spec(form, form, signed=True)
            x = n.starts(spec)[0]
            x[:3] = [.5, .3, .8]
            for j, key in enumerate(m.names(spec)):
                if key == "lambda": x[j] = .2
                if key.startswith("tau_"): x[j] = .3
                if key == "gamma": x[j] = .2
            _, jac = m.predict(spec, x, p)
            for j in range(len(x)):
                h = np.eye(len(x))[j]*1e-6
                numeric = (m.predict(spec, x+h, p)[0]-m.predict(spec, x-h, p)[0])/2e-6
                np.testing.assert_allclose(jac[:, j], numeric, rtol=2e-5, atol=1e-7, err_msg=f"{form}/{j}")

    def test_original_atlas_raw_adjusted_equivalence(self):
        spec = m.Spec("atlas", "full")
        p = fixture()
        x = np.array([.5, .3, .8, .4, .3, .2, .1, .2, .3, .4])
        previous = p | {"physical_floor": -5., "training_minimum": -2.}
        raw, jac = m.predict(spec, x, p)
        adj, old_jac = old.predict(x, previous)
        np.testing.assert_allclose(raw-5., adj, rtol=0, atol=1e-14)
        np.testing.assert_allclose(jac, old_jac, rtol=1e-14, atol=1e-14)

    def test_moe_atlas_exact_null_and_joint_limit(self):
        p = fixture()
        atlas, moe, joint = m.Spec("a", "full"), m.Spec("m", "moe"), m.Spec("j", "joint")
        a = np.array([.5, .3, .8, .4, .3, .2, .1, .2, .3, .4])
        x = np.r_[a[:3], .4, .4, .3, .3, 0., 0., a[5:]]
        np.testing.assert_allclose(m.predict(atlas, a, p)[0], m.predict(moe, x, p)[0], atol=1e-14)
        x[9:] = [0., 1., 1., 1., 1.]
        np.testing.assert_allclose(m.predict(joint, x[:9], p)[0], m.predict(moe, x, p)[0], atol=1e-14)

    def test_jh_coordinate_parity(self):
        p = fixture(); spec = m.Spec("j", "joint")
        x = np.array([.5, .3, .8, .3, .4, .2, .25, -.1, .02])
        shared = fit.shared_from_outer("JH", x[3:])
        expected = fit.predict("JH", shared, p["N"], p["D"].sum(axis=1), p["E"], p["K"], 1.5, .3, .8)
        np.testing.assert_allclose(m.predict(spec, x, p)[0], expected, atol=1e-14)

    def test_he_multiplies_raw_floor_too(self):
        p = fixture(); spec = m.Spec("he", "he")
        x = np.array([.5, .3, .8, .4, .3, .2])
        base = m.predict(m.Spec("c", "chinchilla"), x[:5], p)[0]
        np.testing.assert_allclose(m.predict(spec, x, p)[0], base*p["family_share"]**(-.2))

    def test_ablation_aggregates_before_saturation(self):
        p = fixture(); spec = m.Spec("other", "other")
        D, U = m.components(spec, p)
        np.testing.assert_array_equal(D[:, 1], p["D"][:, 1:].sum(axis=1))
        np.testing.assert_array_equal(U[:, 1], p["U"][:, 1:].sum(axis=1))

    def test_signed_domain_not_silently_clipped(self):
        p = fixture(); spec = m.Spec("s", "full", signed=True)
        x = n.starts(spec)[0]; x[-4:] = -100.
        with self.assertRaises(ValueError): m.predict(spec, x, p)
        self.assertLess(m.domain(spec, x, p)[0].min(), 0)

    def test_domain_derivative_and_count_sensitivity(self):
        p = fixture(); spec = m.Spec("s", "full", signed=True)
        x = n.starts(spec)[0]
        _, jac = m.domain(spec, x, p)
        for j in range(5, len(x)):
            h = np.eye(len(x))[j]*1e-6
            numeric = (m.domain(spec, x+h, p)[0]-m.domain(spec, x-h, p)[0])/2e-6
            np.testing.assert_allclose(jac[:, j], numeric, rtol=1e-5, atol=1e-7)
        self.assertFalse(np.allclose(m.predict(spec, x, p)[0], m.predict(replace(spec, parameter_count="total"), x, p)[0]))

    def test_encode_roundtrip_and_caps(self):
        for spec in m.panel():
            for x in n.starts(spec)[:5]:
                np.testing.assert_allclose(n.decode(spec, n.encode(spec, x))[0], x, atol=1e-10)
                lo, hi = m.bounds(spec)
                self.assertTrue(np.all(x >= lo) and np.all(x <= hi))

    def test_projected_and_direct_synthetic_fit(self):
        spec = m.Spec("target", "target"); p = fixture(80)
        x = np.array([.5, .3, .8, .4, .3])
        p["y"] = m.predict(spec, x, p)[0]
        result = n.run_start(spec, n.starts(spec)[0], p, p, 0)
        self.assertTrue(result["success"], result)
        audit = n.direct(spec, result, p, p)
        self.assertTrue(audit["success"], audit)
        self.assertLess(audit["objective"], 1e-12)

    def test_packet_heldout_loss_poisoning(self):
        rows = []
        for split in ("train", "test"):
            for k in (1, 4):
                for language in list("abcd")[:k]:
                    rows.append(dict(trajectory_id=split, readout_id=f"{split}{k}", language=language,
                        K=k, E=1, D_t=400./k, D_total=400., N_act=1e8, N_total=1e8,
                        raw_nll=3., adjusted_loss=-2., unigram_nll=5.))
        f = pd.DataFrame(rows)
        member = dict(train_trajectory_ids=["train"], test_trajectory_ids=["test"],
                      train_readout_ids=[], test_readout_ids=[], score_languages=[])
        meta = {key: {"family": "fam" if key != "d" else None, "script": "Latn"} for key in "abcd"}
        before, _ = packets_for(f, member, dict.fromkeys("abcd", 200), meta, .1)
        f.loc[f.trajectory_id.eq("test"), ["raw_nll", "adjusted_loss"]] = 1e20
        after, _ = packets_for(f, member, dict.fromkeys("abcd", 200), meta, .1)
        self.assertEqual(before, after)
        self.assertNotIn("y", before[0]["test"])
        self.assertNotIn("y", before[0]["support"])
        self.assertTrue(before[-1]["missing_family_singleton"])
        self.assertEqual(before[-1]["train"]["family_share"], [.25])

    def test_membership_overlap_rejected(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.campaign import select
        with self.assertRaises(ValueError):
            select(pd.DataFrame(), dict(train_trajectory_ids=["a"], test_trajectory_ids=["a"]), "train")

    def test_weights_immutable_and_constrained_stationarity(self):
        p = fixture(); spec = m.Spec("c", "target")
        x = n.starts(spec)[0]; p["y"] = m.predict(spec, x, p)[0]
        before = p["weights"].copy()
        n.inner(spec, x, p)
        n.objective(spec, x, p)
        np.testing.assert_array_equal(p["weights"], before)
        lo, _ = m.bounds(spec); x[0] = lo[0]
        gradient = np.zeros(len(x)); gradient[0] = 1.
        self.assertLess(n.stationarity(spec, x, gradient, p), 1e-12)
        gradient[0] = -1.
        self.assertGreater(n.stationarity(spec, x, gradient, p), .9)

    def test_kkt_multiplier_solver_has_no_artificial_small_gradient_floor(self):
        p = fixture(); spec = m.Spec("c", "target")
        x = n.starts(spec)[0]; lo, _ = m.bounds(spec)
        x[:3] = lo[:3]
        gradient = np.array([3e-7, -1e-11, 4e-10, 2e-11, -3e-11])
        # Bound coefficients 0 and 2 have legitimate positive multipliers;
        # the small negative coefficient-1 gradient must not induce 1e-7 noise.
        self.assertLess(n.stationarity(spec, x, gradient, p), 4e-11)
        gradient[1] = -2e-8
        self.assertGreater(n.stationarity(spec, x, gradient, p), 1e-8)

    def test_physical_amplitude_polish_preserves_the_equation(self):
        p = fixture(80); spec = m.Spec("he", "he")
        p["family_share"] = np.linspace(.0013, .00131, 80)
        x = np.array([1e-6, 2e-6, 3e-6, .4, .3, 2.])
        p["y"] = m.predict(spec, x, p)[0]+np.random.default_rng(4).normal(0,.04,80)
        before = x.copy(); before[:3] = n.inner(spec, x, p)
        after = x.copy(); after[:3] = n.inner(spec, x, p, physical_kkt=1e-8)
        old_obj = n.objective(spec, before, p)[0]
        new_obj, gradient, _, _ = n.objective(spec, after, p)
        self.assertLessEqual(new_obj, old_obj+1e-14)
        self.assertLess(max(abs(gradient[:3])), 1e-8)
        np.testing.assert_array_equal(after[3:], x[3:])

    def test_conditioned_inner_active_bound_is_physically_feasible(self):
        # The optimum has a negative unconstrained intercept, so the physical
        # floor must stay exactly inside its box after undoing conditioning.
        p = fixture(80); spec = m.Spec("total", "full", parameter_count="total")
        x = n.starts(spec)[0]; x[:3] = [1e-10, 1.2, 5.6]
        p["y"] = m.predict(spec, x, p)[0]-.01
        x[:3] = n.inner(spec, x, p)
        lo, hi = m.bounds(spec)
        self.assertTrue(np.all(x >= lo) and np.all(x <= hi))
        value, gradient, _, _ = n.objective(spec, x, p)
        self.assertTrue(np.isfinite(value))
        projected = x[:3]-np.clip(x[:3]-gradient[:3], lo[:3], hi[:3])
        self.assertLessEqual(np.max(abs(projected)), 1e-8)

    def test_signed_independent_audit_on_feasible_negative_transfer(self):
        p = fixture(60); spec = m.Spec("signed", "full", signed=True)
        p["D"][:, 1:] = p["D"][:, :1]*np.array([.5, 1., 2., 3.])[None, :]
        x = np.array([.5, .3, .8, .4, .3, .2, -.01, .2, .3, .4])
        p["y"] = m.predict(spec, x, p)[0]
        result = n.direct(spec, {"coordinates": x.tolist()}, p, p)
        self.assertTrue(result["success"], result)
        self.assertLess(result["objective"], 1e-15)

    def test_incomplete_scoring_population_rejected(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.reporting import assert_same_rows
        reference = pd.DataFrame(dict(fold=["f", "f"], trajectory_id=["a", "b"], readout_id=["a1", "b1"], language=["en", "en"]))
        with self.assertRaises(ValueError): assert_same_rows(reference, reference.iloc[:1])
        with self.assertRaises(ValueError): assert_same_rows(reference, pd.concat((reference, reference)))

    def test_failed_baseline_never_generates_win_prose(self):
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.reporting import manuscript
        text = manuscript({}, {"panel_complete": False, "robust_advantage": False}, 1)
        self.assertIn("make no claim", text)
        self.assertIn("previously inspected", text)

    def test_duplicate_constraints_preserve_domain_and_penalty_exactly(self):
        p = fixture(12)
        for key in ("D", "U", "N", "N_total", "E", "K", "family_share"):
            p[key] = np.concatenate((p[key], p[key], p[key][:4]))
        reduced, multiplicity, inverse = n.compact_domain(p)
        self.assertEqual(len(reduced["N"]), 12)
        for form in ("full", "other", "moe"):
            spec = m.Spec(form, form, signed=True)
            x = n.starts(spec)[0]; x[-1] = -10.
            a, aj = m.domain(spec, x, p)
            b, bj = m.domain(spec, x, reduced)
            np.testing.assert_array_equal(a, b[inverse])
            np.testing.assert_array_equal(aj, bj[inverse])
            self.assertAlmostEqual(float(np.minimum(a, 0.)@np.minimum(a, 0.)),
                float(np.sum(multiplicity*np.minimum(b, 0.)**2)), places=9)


if __name__ == "__main__":
    unittest.main()
