"""Equation, fitting, selection and scheduler safety fixtures; no cluster jobs."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as legacy
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import campaign, contracts as c, numerics, replay, reporting
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.model import Model, Reduction, starts, validate
from tests.test_stage1_law_search import source_fixture


def fixture(name):
    problem, inventory, x, _ = source_fixture()
    problem = replace(problem, K=np.tile([1., 12., 50., 200.], 3))
    model = Model(name, problem, inventory)
    vector = np.r_[x, .4, .7]
    reduction = Reduction(model)
    vector = reduction.expand(reduction.pack(vector))
    rows = pd.DataFrame(dict(language=problem.language_ids, N_act=problem.N, E=problem.E, K=problem.K,
                        D_t=problem.D, U_t=inventory, D_family=problem.D_family,
                        D_script=problem.D_script, D_other=problem.D_other))
    return model, vector, rows


def profile():
    return dict(workspace_root="/tmp/coverage fixture", python_executable="/env/bin/python",
                account="fixture", partition="cpu", memory="4G", concurrency=256,
                start_minutes=240, certify_minutes=240, analysis_minutes=60, max_reserved_core_hours=2500)


def scores():
    return [dict(form=name, fold=fold, certified=True, complete=True, rmse=1.)
            for name in c.FORMS for fold in c.INNER]


class CoverageLawsTest(unittest.TestCase):
    def test_ablations_import_without_retired_campaigns(self):
        workspace = Path(__file__).resolve().parents[1]
        program = """
import importlib
from pathlib import Path
import sys
sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / "src")]
prefix = "multilingual_megatron.experiments.scaling.stage1_fit."
sys.modules[prefix + "law_simplification"] = None
for name in ("campaign", "numerics", "replay", "reporting", "inference", "diagnostics"):
    sys.modules[prefix + "nd_coupling." + name] = None
for name in ("campaign", "reporting", "diagnostics", "recovery"):
    importlib.import_module(prefix + "coverage_laws." + name)
"""
        subprocess.run([sys.executable, "-I", "-B", "-c", program, str(workspace)],
                       check=True, capture_output=True, text=True)

    def test_task_graph_complete_and_bounded(self):
        p = profile()
        c.validate_profile(p)
        tasks = c.tasks(p)
        self.assertEqual(len(tasks), 624)
        self.assertEqual(len(c.cells()), 48)
        self.assertEqual(c.reservation(p), 2498.)
        self.assertEqual(len({(t['stage'], t['id']) for t in tasks}), len(tasks))
        for cell in c.cells():
            indices = [i for t in tasks if t['stage'] == 'starts' and t['cell'] == cell['id'] for i in t['start_indices']]
            self.assertEqual(indices, list(range(48)))
        for change in ({'concurrency':257}, {'start_minutes':0}, {'account':'bad\n#SBATCH --gres=gpu:1'},
                       {'max_reserved_core_hours':1}):
            with self.assertRaises(ValueError): c.validate_profile(p | change)

    def test_counts_and_independent_replay_all_forms(self):
        for name, count in c.FORMS.items():
            model, x, rows = fixture(name)
            validate(model, {'coordinates':x})
            self.assertEqual(len(Reduction(model).groups), count)
            record = dict(coordinates=x, language_coefficients=[dict(language='x', L_inf=-5., A=.3, B=.4)])
            np.testing.assert_allclose(legacy.predict(model, record), replay.predict(name, record, rows), atol=1e-12, rtol=0)

    def test_exact_rn_restrictions_and_pulled_back_penalty(self):
        for name in ('RN12-common-K', 'RN13-capacity-tied', 'RN14-matched'):
            model, x, _ = fixture(name)
            old = legacy.Model('RN-no-composition', model.problem, model.inventory)
            for i in (0,1): np.testing.assert_allclose(model.terms(x)[i], old.terms(x[:17])[i], atol=0, rtol=0)
            self.assertEqual(model.penalty(x)[0],old.penalty(x[:17])[0])
            np.testing.assert_array_equal(model.penalty(x)[1][:17],old.penalty(x[:17])[1])

    def test_all_analytic_derivatives_and_reduced_gradients(self):
        for name in c.FORMS:
            model, x, _ = fixture(name)
            _, _, cj, dj = model.terms(x)
            reduction = Reduction(model)
            for group in reduction.groups:
                step = np.zeros(19); step[group] = 1e-6
                plus, minus = model.terms(x+step), model.terms(x-step)
                np.testing.assert_allclose((plus[0]-minus[0])/2e-6, cj[:,group].sum(axis=1), atol=1e-8, rtol=4e-6)
                np.testing.assert_allclose((plus[1]-minus[1])/2e-6, dj[:,group].sum(axis=1), atol=1e-8, rtol=4e-6)
                derivative = (model.penalty(x+step)[0]-model.penalty(x-step)[0])/2e-6
                self.assertAlmostEqual(derivative,model.penalty(x)[1][group].sum(),places=9)

    def test_shared_load_nulls_and_monolingual_anchor(self):
        model, x, _ = fixture('Shared-load-13')
        tied, z, _ = fixture('RN12-common-K')
        z[:6]=x[:6]; z[6:8]=x[6]; z[8:10]=x[8]; z[10]=0.; z[14:17]=x[14:17]
        x[17:]=[1.,0.]
        for a,b in zip(model.terms(x)[:2],tied.terms(z)[:2]): np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
        before = model.terms(x)
        x[17:] = [5.,1.2]
        after = model.terms(x)
        mask = model.problem.K == 1
        for i in (0,1): np.testing.assert_array_equal(before[i][mask],after[i][mask])
        x[17]=0.
        self.assertTrue(np.isfinite(model.terms(x)[2]).all())
        np.testing.assert_array_equal(model.terms(x)[2][:,18],0.)

    def test_deterministic_training_only_starts_and_domains(self):
        _, _, anchor, _ = source_fixture()
        anchor=anchor[:15]
        for name in c.FORMS:
            model, _, _ = fixture(name)
            a,b=starts(model,anchor),starts(model,anchor)
            self.assertEqual(a.shape,(48,19));np.testing.assert_array_equal(a,b)
            for x in a: validate(model,{'coordinates':x})
        with self.assertRaises(ValueError): starts(model,anchor[:5])

    def test_real_synthetic_profile_direct_certification(self):
        model,x,rows=fixture('Shared-load-13')
        cap,term,*_=model.terms(x)
        model.problem=replace(model.problem,y=-5.+.3*cap+.4*term,
                              regularization=legacy.fit.M5M6Regularization(0.,.001))
        model.bounds=[replace(b,lower=float(v),upper=float(v)) for b,v in zip(model.bounds,x)]
        for i in (17,18):model.bounds[i]=replace(model.bounds[i],lower=.01,upper=1.5)
        initial=x.copy();initial[17:]=[.5,.6]
        projected=legacy.Profile(model,.15)
        result=projected.run(initial)
        self.assertTrue(result['success'],result)
        direct=legacy.Direct(projected,result).run()
        self.assertTrue(direct['success'],direct)
        np.testing.assert_allclose(legacy.predict(model,result),legacy.predict(model,direct),atol=1e-4,rtol=0)
        np.testing.assert_allclose(legacy.predict(model,result),model.problem.y,atol=1e-5,rtol=0)
        self.assertEqual(numerics.geometry(projected,result)['free_shared'],2)

    def test_restart_coverage_and_failed_lower_objective_block(self):
        shard_tasks=[t for t in c.tasks(profile()) if t['stage']=='starts' and t['cell']==c.cells()[0]['id']]
        shards=[(t,dict(complete=True,attempts=[dict(index=i,success=True,objective=1.,kkt=0.)
                 for i in t['start_indices']])) for t in shard_tasks]
        self.assertEqual(len(numerics.collect_attempts(shards)),48)
        for bad in (shards[:-1],shards+[shards[0]],[(shards[0][0],dict(complete=False)),*shards[1:]]):
            with self.assertRaises(ValueError):numerics.collect_attempts(bad)
        self.assertIsNone(numerics.best_attempt([dict(index=0,success=True,objective=1.),
                                                dict(index=1,success=False,objective=.9)]))

    def test_selector_excludes_terminal_and_auxiliary_controls(self):
        rows=scores()
        decision=c.choose(rows)
        self.assertEqual(decision['chosen']['form'],'RN12-common-K')
        for row in rows:
            if row['fold']=='inner_terminal' or row['form'] in ('RN14-no-ridge','Shared-load-chi0'):
                row['rmse']=1000. if row['form']=='RN12-common-K' else 1e-12
        self.assertEqual(decision,c.choose(rows))
        self.assertEqual(decision,c.choose(list(reversed(rows))))
        for row in rows:
            if row['form']=='RN12-common-K' and row['fold']==c.SCREEN[0]:row['rmse']=1.03
        self.assertNotEqual(c.choose(rows)['chosen']['form'],'RN12-common-K')
        with self.assertRaises(ValueError):c.choose(rows+[rows[0]])
        rows[0]['certified']=False
        with self.assertRaises(ValueError):c.choose(rows)

    def test_prepared_scripts_are_cpu_only_and_highly_parallel(self):
        plan=dict(profile=profile(),tasks=c.tasks(profile()),sha256='fixture',reserved_core_hours=2498.)
        for stage in c.STAGES:
            script=campaign.script(Path('/tmp/coverage fixture/plan.json'),plan,stage)
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertNotIn('--gres',script);self.assertNotIn('sbatch ',script)
            self.assertIn('--no-requeue',script);self.assertIn('PYTHONSAFEPATH=1',script)
            self.assertIn('--cpus-per-task=1',script)
            if stage=='starts':self.assertIn('--array=0-575%256',script)

    def test_mocked_submission_dependencies_and_duplicate_guard(self):
        plan=dict(profile=profile(),tasks=c.tasks(profile()),sha256='fixture',reserved_core_hours=2498.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'submissions').mkdir();(root/'results').mkdir()
            for stage in c.STAGES:
                io.write_text_once(root/'submissions'/f'{stage}.sbatch',campaign.script(root/'plan.json',plan,stage))
            with patch.object(campaign,'load',return_value=plan),patch.dict(campaign.os.environ,{},clear=True),\
                    patch.object(campaign.subprocess,'check_output',side_effect=['10','11','12','13']) as submit:
                campaign.submit(root/'plan.json')
                calls=[v.args[0] for v in submit.call_args_list]
                self.assertIn('--dependency=afterany:10',calls[1])
                self.assertIn('--dependency=afterany:12',calls[3])
                with self.assertRaises(FileExistsError):campaign.submit(root/'plan.json')
            with patch.dict(campaign.os.environ,{'SLURM_JOB_ID':'1'}):
                with self.assertRaisesRegex(ValueError,'nested'):campaign.submit(root/'plan.json')

    def test_selection_receipt_tamper_and_failed_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            payload=c.choose(scores())|dict(passed=True,plan_sha256='fixture',inputs=[],inner_metrics=scores(),
                        frozen_before_outer_scoring=True,outer_or_endpoint_responses_read=False)
            campaign.seal(root/'selection.json',payload)
            self.assertEqual(reporting.selection_receipt(root,{'sha256':'fixture'})['chosen'],payload['chosen'])
            with self.assertRaises(ValueError):reporting.selection_receipt(root,{'sha256':'foreign'})

    def test_end_to_end_synthetic_selection_and_report(self):
        """Exercise real scoring/replay/files without study fits or scheduler access."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'results').mkdir()
            base,_,rows=fixture('RN14-matched')
            rows['trajectory_id']=[f't{i}' for i in range(len(rows))]
            rows['readout_id']=[f'r{i}' for i in range(len(rows))]
            rows['adjusted_loss']=base.problem.y
            rows['unigram_nll']=-base.problem.physical_floors
            rows['D_total']=rows.D_t+rows.D_family+rows.D_script+rows.D_other
            rows.to_parquet(root/'rows.parquet',index=False)
            row_entry=campaign.binding(root/'rows.parquet')
            baseline=pd.DataFrame([dict(model='published',category='published_form_adaptation',role='baseline',
                       fold=fold,case=case,certified=True,complete=True,rmse=1.)
                       for fold in (*c.PRIMARY,'all_grid') for case in
                       (['L26'] if fold=='all_grid' else ['grid','L26'])])
            baseline.to_csv(root/'baseline.csv',index=False)
            members={f:dict(train_trajectory_ids=[],test_trajectory_ids=rows.trajectory_id.tolist(),
                     train_readout_ids=[],test_readout_ids=[],score_languages=['x']) for f in c.PRIMARY}
            plan=dict(sha256='fixture',profile=profile(),tasks=c.tasks(profile()),memberships=members,
                     inner_scoring={f:[row_entry] for f in c.INNER},
                     scoring={'grid.parquet':row_entry,'endpoint.parquet':row_entry},
                     baseline_metrics=campaign.binding(root/'baseline.csv'),reserved_core_hours=2498.)
            for task in [t for t in plan['tasks'] if t['stage']=='certify']:
                model,x,_=fixture(task['form'])
                result=dict(certified=True,complete=True,plan_sha256='fixture',
                       task_sha256=campaign.canonical_json_sha256(task),dependencies=[],
                       selected=dict(coordinates=x.tolist(),language_coefficients=[dict(language='x',L_inf=-5.,A=.3,B=.4)]),
                       bounds=[asdict(b) for b in model.bounds],equalities=list(model.equalities),
                       regularization=model.problem.regularization.as_dict(),geometry={})
                io.write_json_once(campaign.result_path(root,task),result)
            with patch.object(reporting,'load',return_value=plan),patch.object(reporting,'cpu_guard'):
                self.assertTrue(reporting.select(root/'plan.json'))
                self.assertTrue(reporting.report(root/'plan.json'))
                with self.assertRaises(ValueError):reporting.select(root/'plan.json')
            summary=pd.read_csv(root/'report/comparison.csv')
            self.assertEqual(set(summary.model),set(c.FORMS)|{'published'})
            self.assertTrue(summary.mean_primary_rmse.notna().all())
            receipt=io.read_json(root/'report/report.json')
            self.assertEqual(receipt['status'],'complete')
            self.assertFalse(receipt['new_significance_test'])

    def test_failed_selector_produces_incomplete_report_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=dict(sha256='fixture')
            campaign.seal(root/'selection.json',dict(passed=False,plan_sha256='fixture',inputs=[]))
            with patch.object(reporting,'load',return_value=plan),patch.object(reporting,'cpu_guard'):
                self.assertFalse(reporting.report(root/'plan.json'))
            receipt=io.read_json(root/'report/report.json')
            self.assertFalse(receipt['outer_scores_computed'])
            self.assertFalse((root/'report/metrics.csv').exists())

    def test_missing_baseline_cases_are_unresolved_not_defeated(self):
        existing=pd.DataFrame([dict(model='later-ATLAS',fold='all_grid',case='L26',
                    category='published_form_adaptation',certified=True,complete=True,rmse=.2)])
        expanded=reporting.complete_baseline_panel(existing)
        self.assertEqual(len(expanded),7)
        self.assertEqual(expanded.certified.sum(),1)
        self.assertEqual(expanded.missing_reason.notna().sum(),6)
        means=reporting.mean_metrics(expanded)
        self.assertFalse(means[0]['complete'])
        panel=expanded[expanded.case.eq('grid')&expanded.fold.eq(c.PRIMARY[0])]
        gate=reporting.comparison_gate(dict(certified=True,complete=True,rmse=.1),panel)
        self.assertFalse(gate['strict_all_baseline_win'])
        self.assertEqual(gate['unresolved'],['later-ATLAS'])


if __name__=='__main__':unittest.main()
