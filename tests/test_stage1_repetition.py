from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import fit, repetition_model as m
from multilingual_megatron.experiments.scaling.stage1_fit.repetition_campaign import inventories, partition, selected_attempt, render_job, validate_output, binding, verify
from multilingual_megatron.experiments.scaling.stage1_fit.repetition_reporting import score


def fixture():
    n = 12
    p = fit.ProjectedProblem(y=np.linspace(-2.8, -4., n), weights=np.ones(n)/n,
        N=np.geomspace(1e8, 8e8, n), D=np.geomspace(1e8, 2e10, n),
        E=np.tile([1., 4., 16., 32.], 3), K=np.full(n, 200.), language_ids=("x",)*n,
        physical_floors=np.full(n, -6.), D_family=np.full(n, 2e9),
        D_script=np.full(n, 3e9), D_other=np.full(n, 4e9),
        regularization=fit.M5M6Regularization(.01, .001))
    x = np.array([.4, .5, .3, .35, -.04, .02, .05, .06, -.03, -.02, -.01, .1, .03, .01, .15, .2])
    return p, np.full(n, 1e9), x


class RepetitionTest(unittest.TestCase):
    def test_output_cannot_overlap_upstream_or_escape_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, upstream = root/'workspace', root/'upstream'
            workspace.mkdir(); upstream.mkdir()
            profile = {"workspace_root": str(workspace), "read_only_root": str(upstream),
                       "output_root": str(workspace/'results')}
            self.assertEqual(validate_output(profile), workspace/'results')
            for output in (workspace, upstream/'results', root/'elsewhere'):
                with self.assertRaises(ValueError):
                    validate_output(profile | {"output_root": str(output)})
            (workspace/'link').symlink_to(upstream, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output(profile | {"output_root": str(workspace/'link'/'results')})

    def test_input_binding_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'input.json'
            path.write_text('{}')
            entry = binding(path)
            self.assertEqual(verify(entry), path)
            path.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "changed bound input"):
                verify(entry)

    def test_synthetic_optimization_and_direct_replay(self):
        p, u, x = fixture()
        x = np.r_[x, .4]
        cap, data, _, _ = m.terms(x, p, u)
        p = replace(p, y=-5.+.3*cap+.4*data,
                    regularization=fit.M5M6Regularization(0., .001))
        # Fix shape coordinates and exercise real optimization of the new rate
        # and exponent plus profiled language coefficients, using no study data.
        bounds = [m.Bound(str(i), float(v), float(v), "signed") for i, v in enumerate(x)]
        bounds[15] = m.Bound('lambda_repetition', .01, 1., 'nonnegative')
        bounds[16] = m.Bound('rho', -1., 1., 'signed')
        initial = x.copy(); initial[15:] = [.15, .2]
        profile = m.Profile(p, .15, u, constrained=True)
        selected = profile.run(initial, bounds)
        self.assertTrue(selected['success'], selected['message'])
        np.testing.assert_allclose(m.predict(p, u, selected), p.y, atol=1e-5, rtol=0)
        direct = m.Direct(profile, selected, bounds).run()
        self.assertTrue(direct['success'], direct['message'])
        np.testing.assert_allclose(m.predict(p, u, direct), m.predict(p, u, selected), atol=1e-5, rtol=0)

    def test_fit_and_audit_logs_are_scoped_without_moving_results(self):
        plan_path = Path("/tmp/repetition with spaces/plan.json")
        plan = {"sha256": "a"*64, "profile": {"python_executable": "/env/bin/python", "account": "fixture",
                "partition": "cpu", "memory": "4G", "concurrency": 8}, "protocol": {"time_limit": "00:30:00"}}
        for action, count in (("fit", 18), ("audit", 6)):
            script, logs = render_job(plan_path, plan, action, count)
            self.assertEqual(logs, plan_path.parent/"logs/attempts"/f"{action}-{'a'*12}")
            self.assertIn(f"#SBATCH --array=0-{count-1}%8", script)
            self.assertIn(str(logs), script)
            self.assertIn(str(plan_path), script)
            self.assertIn(str(plan_path.parent/"source"), script)
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        with self.assertRaisesRegex(ValueError, "invalid repetition"):
            render_job(plan_path, plan, "retry", 18)

    def test_zero_rate_exactly_nests_parent(self):
        p, u, x = fixture()
        x[-1] = 0.
        old, new = m.parent.terms(x[:15], p), m.terms(x, p, u)
        for a, b in zip(old[:2], new[:2]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(old[2:], new[2:]):
            np.testing.assert_allclose(a, b[:, :15], atol=1e-14)

    def test_complete_analytic_term_derivative(self):
        p, u, x = fixture()
        values = m.terms(x, p, u)
        for i in range(16):
            h = np.eye(16)[i]*1e-6
            plus, minus = m.terms(x+h, p, u), m.terms(x-h, p, u)
            for term in range(2):
                np.testing.assert_allclose((plus[term]-minus[term])/2e-6, values[term+2][:, i], rtol=2e-6, atol=1e-8)

    def test_size_dependent_rate_nesting_and_derivatives(self):
        p, u, x = fixture()
        nested = m.terms(np.r_[x, 0.], p, u)
        base = m.terms(x, p, u)
        for a, b in zip(base[:2], nested[:2]):
            np.testing.assert_array_equal(a, b)
        x = np.r_[x, .4]
        terms = m.terms(x, p, u)
        for i in range(17):
            h = np.eye(17)[i]*1e-6
            plus, minus = m.terms(x+h, p, u), m.terms(x-h, p, u)
            np.testing.assert_allclose((plus[1]-minus[1])/2e-6, terms[3][:, i], rtol=2e-6, atol=1e-8)
        profile = m.Profile(p, .15, u)
        selected = profile.evaluate(x)[0]
        direct = m.Direct(profile, selected, [m.Bound(str(i), -2., 2., "signed") for i in range(17)])
        z = np.array(direct.initial)
        gradient = direct.evaluate(z)[1]
        for i in range(len(z)):
            h = np.eye(len(z))[i]*1e-6
            self.assertAlmostEqual((direct.evaluate(z+h)[0]-direct.evaluate(z-h)[0])/2e-6, gradient[i], places=7)

    def test_initializer_fallback_is_narrow_and_keeps_solver(self):
        p, u, x = fixture()
        with patch.object(m, "original_solve", side_effect=ValueError("Initial guess is outside of provided bounds")):
            result = m.Profile(p, .15, u).evaluate(x)[0]
            self.assertTrue(np.isfinite(result["objective"]))
        with patch.object(m, "original_solve", side_effect=ValueError("different bug")):
            with self.assertRaisesRegex(ValueError, "different bug"):
                m.Profile(p, .15, u).evaluate(x)

    def test_saturation_is_not_an_overtraining_penalty(self):
        p, u, x = fixture()
        q = replace(p, N=np.full(12, 2e8), E=np.full(12, 4.), D_family=np.zeros(12),
                    D_script=np.zeros(12), D_other=np.zeros(12))
        cap, data, _, _ = m.terms(x, q, u)
        self.assertTrue(np.all(np.diff(data) <= 0))
        np.testing.assert_array_equal(cap, np.full(12, cap[0]))
        zero = x.copy(); zero[-1] = 0.
        self.assertTrue(np.all(data >= m.terms(zero, q, u)[1]))
        with self.assertRaises(ValueError):
            m.terms(x, p, np.zeros(12))
        with self.assertRaises(ValueError):
            m.terms(x, p, np.ones(2))

    def test_direct_objective_gradient(self):
        p, u, x = fixture()
        profile = m.Profile(p, .15, u)
        selected = profile.evaluate(x)[0]
        bounds = [m.Bound(str(i), -2., 2., "signed") for i in range(16)]
        direct = m.Direct(profile, selected, bounds)
        z = np.asarray(direct.initial)
        value, gradient, prediction, _ = direct.evaluate(z)
        np.testing.assert_allclose(prediction, m.predict(p, u, selected), atol=1e-9)
        self.assertAlmostEqual(value, selected["objective"], places=10)
        for i in range(len(z)):
            h = np.eye(len(z))[i]*1e-6
            difference = (direct.evaluate(z+h)[0]-direct.evaluate(z-h)[0])/2e-6
            self.assertAlmostEqual(difference, gradient[i], places=7)

    def test_split_and_inventory_fail_closed(self):
        rows = pd.DataFrame({"trajectory_id": ["a", "b"], "language": ["x", "x"], "D_total": [1e9, 69e9]})
        member = {"train_trajectory_ids": ["a"], "test_trajectory_ids": ["b"], "score_languages": ["x"]}
        self.assertEqual(partition(rows, member, "train").trajectory_id.tolist(), ["a"])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            partition(rows, dict(member, train_trajectory_ids=["a", "b"]), "train")
        with self.assertRaisesRegex(ValueError, "entered training"):
            partition(rows, dict(member, train_trajectory_ids=["b"], test_trajectory_ids=["a"]), "train")
        packet = {"language": "x", "support": {"K": [200, 200], "U": [[1], [2]]}}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            inventories([packet])

    def test_scoring_does_not_drop_languages_or_center_bias(self):
        rows = pd.DataFrame({"trajectory_id": ["a", "a", "b"], "readout_id": ["1", "1", "2"],
                             "language": ["x", "y", "x"], "adjusted_loss": [0., 0., 0.]})
        prediction = rows.assign(prediction=[1., 1., 3.])
        self.assertAlmostEqual(score(prediction, rows)["rmse"], np.sqrt(5.))
        self.assertAlmostEqual(score(prediction, rows, endpoint=True)["rmse"], np.sqrt(11/3))
        self.assertFalse(score(prediction.iloc[:2], rows)["complete"])
        self.assertFalse(score(prediction.assign(prediction=[1., np.nan, 3.]), rows)["complete"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score(pd.concat([prediction, prediction]), rows)

    def test_failed_better_objective_cannot_be_ignored(self):
        good = {"success": True, "objective": 1., "task_id": 1}
        self.assertEqual(selected_attempt([good]), good)
        with self.assertRaisesRegex(ValueError, "better objective"):
            selected_attempt([good, {"success": False, "objective": .9}])


if __name__ == "__main__":
    unittest.main()
