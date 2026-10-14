"""Coverage exponent ablation: nested forms, full derivatives and workflow."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as legacy
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import campaign, contracts as c, numerics, replay, reporting
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.model import Model, Reduction, starts, validate
from tests.test_stage1_law_improvement_round2 import fixture
from tests.test_stage1_law_search import source_fixture


def profile():
    return json.loads(Path('configs/stage1_fit/coverage_exponents_v1.json').read_text())


class CoverageExponentsTest(unittest.TestCase):
    def test_audit_recovery_only_changes_optimizer_stop(self):
        model, x, _ = fixture('RN15-stored-separate-exponents')
        with patch.object(legacy, 'minimize') as minimize:
            for tolerance in (1e-12, 1e-14, 1e-16):
                kwargs = {} if tolerance == 1e-12 else {'ftol': tolerance}
                legacy.optimize(lambda z: (0., np.zeros(len(z))), x, model.bounds, model, **kwargs)
                call = minimize.call_args.kwargs
                self.assertEqual(call['options'], {'ftol': tolerance, 'maxiter': 4000})
                self.assertEqual(call['bounds'], [(b.lower, b.upper) for b in model.bounds])
                self.assertEqual(call['method'], 'SLSQP')

    def test_counts_nulls_and_literal_replay(self):
        for name, count in c.DATA_EXPONENT_FORMS.items():
            model, x, rows = fixture(name)
            self.assertEqual(len(Reduction(model).groups), count)
            validate(model, {'coordinates': x})
            self.assertEqual(model.penalty(x)[0], 0.)
            record = dict(coordinates=x, language_coefficients=[dict(language='x', L_inf=-5., A=.3, B=.4)])
            np.testing.assert_allclose(legacy.predict(model, record), replay.predict(name, record, rows), atol=1e-12, rtol=0)
            base, _, _ = fixture('RN14-stored-reuse')
            null = x.copy(); null[17] = 0.
            for a, b in zip(model.terms(null)[:2], base.terms(null)[:2]):
                np.testing.assert_array_equal(a, b)
            separate, _, _ = fixture('RN15-stored-separate-exponents')
            for a, b in zip(model.terms(x)[:2], separate.terms(x)[:2]):
                np.testing.assert_array_equal(a, b)

    def test_all_reduced_derivatives_and_zero_rate(self):
        for name in c.DATA_EXPONENT_FORMS:
            model, x, _ = fixture(name)
            for rate in (.27, 0.):
                x[15] = rate
                reduction = Reduction(model); z = reduction.pack(x)
                cap, term, cj, dj = model.terms(x)
                for j, group in enumerate(reduction.groups):
                    with self.subTest(name=name, rate=rate, group=group):
                        step = np.eye(len(z))[j]*1e-6
                        plus = model.terms(reduction.expand(z+step))
                        if rate == 0 and 15 in group:
                            numeric = (plus[1]-term)/1e-6
                            np.testing.assert_allclose(numeric, dj[:, group].sum(axis=1), atol=3e-5, rtol=2e-4)
                            continue
                        minus = model.terms(reduction.expand(z-step))
                        for i, jac in ((0, cj), (1, dj)):
                            np.testing.assert_allclose((plus[i]-minus[i])/2e-6, jac[:, group].sum(axis=1), atol=1e-7, rtol=2e-5)

    def test_positivity_both_exponents_and_support(self):
        for name in c.DATA_EXPONENT_FORMS:
            model, x, _ = fixture(name)
            bad = x.copy(); bad[2:4] = .01; bad[17] = -.1
            if model.equalities: bad[10] = -.1
            with self.assertRaisesRegex(ValueError, 'positivity'): validate(model, {'coordinates': bad})
            small = replace(model.problem, K=np.full_like(model.problem.K, 12.))
            restricted = Model(name, small, model.inventory, stored_parameters=model.stored_parameters)
            self.assertEqual(restricted.bounds[10].upper, 0.)
            self.assertEqual(restricted.bounds[17].upper, 0.)
            np.testing.assert_array_equal(model.terms(x)[3][model.problem.K <= 12, 17], 0.)

    def test_start_coverage_matching_and_budget(self):
        anchor = source_fixture()[2][:15]
        base, _, _ = fixture('RN14-stored-reuse')
        reference = starts(base, anchor)
        for name in c.DATA_EXPONENT_FORMS:
            model, _, _ = fixture(name)
            initial = starts(model, anchor)
            np.testing.assert_array_equal(initial, starts(model, anchor))
            np.testing.assert_array_equal(initial[:, :10], reference[:, :10])
            np.testing.assert_array_equal(initial[:, 11:17], reference[:, 11:17])
            for row in initial: validate(model, {'coordinates': row})
            if name == 'RN15-stored-separate-exponents':
                np.testing.assert_array_equal(initial[:12], reference[:12])
        p=profile(); c.validate_profile(p)
        self.assertEqual(len(c.cells(c.rules_for(p))), 32)
        self.assertEqual(len(c.tasks(p)), 1568)
        self.assertEqual(c.reservation(p), 12548.)

    def test_synthetic_profile_and_direct_optimization(self):
        model, truth, rows = fixture('RN15-stored-separate-exponents')
        truth[17] = -.03
        cap, term = model.terms(truth)[:2]
        model.problem = replace(model.problem, y=-5.+.3*cap+.4*term)
        for i in range(19):
            if i not in (10, 17): model.bounds[i] = replace(model.bounds[i], lower=truth[i], upper=truth[i])
        initial = truth.copy(); initial[10] += .01; initial[17] = .01
        fitted = legacy.Profile(model, None, time.monotonic()+60)
        result = fitted.run(initial)
        audit = legacy.Direct(fitted, result).run()
        self.assertTrue(result['success']); self.assertTrue(audit['success'])
        np.testing.assert_allclose(legacy.predict(model, result), replay.predict(model.name, result, rows), atol=1e-12)
        geometry = numerics.geometry(fitted, result)
        self.assertIn('coverage_exponents', geometry)

    def test_synthetic_full_report_and_shared_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'results').mkdir()
            base, _, rows=fixture('RN14-stored-reuse')
            rows['trajectory_id']=[f't{i}' for i in range(len(rows))]
            rows['readout_id']=[f'r{i}' for i in range(len(rows))]
            rows['adjusted_loss']=base.problem.y
            rows['unigram_nll']=-base.problem.physical_floors
            rows['D_total']=rows.D_t+rows.D_family+rows.D_script+rows.D_other
            rows.to_parquet(root/'rows.parquet',index=False)
            entry=campaign.binding(root/'rows.parquet')
            pd.DataFrame([dict(model='published',category='published_form_adaptation',fold=f,case=k,
                certified=True,complete=True,rmse=1.) for f in (*c.PRIMARY,'all_grid')
                for k in (['L26'] if f=='all_grid' else ['grid','L26'])]).to_csv(root/'baseline.csv',index=False)
            members={f:dict(train_trajectory_ids=[],test_trajectory_ids=rows.trajectory_id.tolist(),
                train_readout_ids=[],test_readout_ids=[],score_languages=['x']) for f in c.PRIMARY}
            plan=dict(sha256='fixture',profile=profile(),tasks=c.tasks(profile()),memberships=members,
                inner_scoring={f:[entry] for f in c.INNER},scoring={'grid.parquet':entry,'endpoint.parquet':entry},
                baseline_metrics=campaign.binding(root/'baseline.csv'),reserved_core_hours=12548.)
            for task in (t for t in plan['tasks'] if t['stage']=='certify'):
                model,x,_=fixture(task['form']); x[10]=0.; x[17]=0.
                value=dict(certified=True,complete=True,plan_sha256='fixture',dependencies=[],
                    task_sha256=campaign.canonical_json_sha256(task),
                    selected=dict(coordinates=x.tolist(),language_coefficients=[dict(language='x',L_inf=-5.,A=.3,B=.4)]),
                    bounds=[asdict(b) for b in model.bounds],equalities=list(model.equalities),
                    regularization=model.problem.regularization.as_dict(),geometry={})
                io.write_json_once(campaign.result_path(root,task),value)
            with patch.object(reporting,'load',return_value=plan),patch.object(reporting,'cpu_guard'):
                self.assertTrue(reporting.select(root/'plan.json'))
                self.assertTrue(reporting.report(root/'plan.json'))
            pairs=pd.read_csv(root/'report/paired_ablation.csv')
            self.assertEqual(len(pairs),36)
            np.testing.assert_allclose(pairs.delta_rmse,0.,atol=1e-12)
            self.assertEqual(io.read_json(root/'selection.json')['chosen']['form'],'RN14-stored-reuse')


if __name__=='__main__': unittest.main()
