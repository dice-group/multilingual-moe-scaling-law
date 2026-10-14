from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow import recovery_audit as a
from multilingual_megatron.experiments.scaling.stage1_fit.execution import write_json_once
from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow import recovery_numerics as n
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.recovery_diagnostics import summary
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow import recovery, refit_audit


class BaselineRecoveryAuditTest(unittest.TestCase):
    def target(self):
        manifest = {"manifest_sha256": "grid", "fit": {
            "model": "J0", "fold_id": "largest_N_act", "objective": "huber",
            "delta": .1, "bounds": [], "bound_version": 0},
            "input": {"source": "fixed"}, "membership": {"train": "fixed"},
            "review_only": False, "selection_report": None}
        attempts = [{"start_id": f"grid:{i}", "success": True,
                     "objective": float(i + 1), "coordinates": [float(i)]} for i in range(3)]
        result = {"result_sha256": "result", "selected": {"boundary_contacts": []}, "attempts": attempts}
        audit = {"status": "failed", "reason": "best basin unresolved",
                 "manifest_sha256": "grid", "result_sha256": "result"}
        basin = copy.deepcopy(manifest)
        basin["fit"]["explicit_starts"] = [
            {"start_id": v["start_id"], "values": v["coordinates"], "kind": "basin"} for v in attempts]
        return manifest, result, audit, basin

    def test_accepts_only_exact_original_three_starts(self):
        values = self.target()
        self.assertEqual(len(a.validate_target(*values)), 3)
        values[-1]["fit"]["explicit_starts"][1]["values"] = [99.]
        with self.assertRaisesRegex(a.BaselineRefitError, "exact original three"):
            a.validate_target(*values)

    def test_other_models_and_failure_reasons_are_rejected(self):
        values = self.target()
        values[0]["fit"]["model"] = "M2"
        with self.assertRaisesRegex(a.BaselineRefitError, "limited to J0"):
            a.validate_target(*values)
        values = self.target()
        values[2]["reason"] = "nested warm start disagrees"
        with self.assertRaisesRegex(a.BaselineRefitError, "original failed basin"):
            a.validate_target(*values)

    def test_changed_membership_and_boundary_cannot_be_amended(self):
        values = self.target()
        values[3]["membership"] = {"train": "other"}
        with self.assertRaisesRegex(a.BaselineRefitError, "membership"):
            a.validate_target(*values)
        values = self.target()
        values[1]["selected"]["boundary_contacts"] = ["upper"]
        with self.assertRaisesRegex(a.BaselineRefitError, "boundary failure"):
            a.validate_target(*values)

    def test_tolerances_are_not_relaxed(self):
        retained = {"objective": 1.0}
        candidate = {"objective": 1.0, "success": True, "boundary_contacts": []}
        with patch.object(a.campaign, "_record_prediction", side_effect=[np.array([0.]), np.array([5.7e-4])]):
            self.assertFalse(a.check_record(retained, candidate, {})["resolved"])
        with patch.object(a.campaign, "_record_prediction", side_effect=[np.array([0.]), np.array([1e-5])]):
            self.assertTrue(a.check_record(retained, candidate, {})["resolved"])

    def test_failed_optimizer_cannot_pass_equivalence(self):
        self.assertFalse(a.check_record({}, {"success": False}, {})["resolved"])

    def test_source_binding_detects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.json"
            write_json_once(path, a.seal({"value": 1}, "test_sha256"))
            value = a.binding(path, "test_sha256")
            self.assertEqual(a.bound(value, "test_sha256")[1]["value"], 1)
            value["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(a.BaselineRefitError, "input changed"):
                a.bound(value, "test_sha256")


class ConditionedProfileTest(unittest.TestCase):
    def problem(self, weight_scale=1):
        rng = np.random.default_rng(193)
        cap = np.exp(rng.normal(size=40))
        data = np.exp(rng.normal(size=40))
        y = -2 + .3 * cap + .7 * data + rng.normal(scale=.02, size=40)
        y[-1] += 1
        weights = np.linspace(.001, .01, 40) * weight_scale
        return fit.LanguageProblem(y, cap, data, weights, -8, float(y.min()))

    def test_weighted_huber_is_exact_original_objective(self):
        problem = self.problem()
        residual = np.linspace(-1, 1, 40)
        delta = .08
        actual = .5 * delta ** 2 * n.weighted_huber_loss(problem.weights)((residual / delta) ** 2)[0].sum()
        self.assertAlmostEqual(actual, fit.weighted_objective(residual, problem.weights, delta), places=14)

    def test_inner_is_invariant_to_small_weight_scale(self):
        for delta in (None, .08):
            one = n.solve_language(self.problem(), delta)
            small = n.solve_language(self.problem(1e-8), delta)
            np.testing.assert_allclose(one.prediction, small.prediction, rtol=0, atol=1e-8)
            self.assertAlmostEqual(one.objective * 1e-8, small.objective, places=18)

    def test_inner_improves_or_matches_original_solver(self):
        problem = self.problem()
        conditioned = n.solve_language(problem, .08)
        original = fit.solve_language(problem, .08)
        self.assertLessEqual(conditioned.objective, original.objective + 1e-14)
        self.assertLess(conditioned.gradient_infinity_norm, 1e-9)

    def test_active_floor_retains_original_constraint(self):
        problem = self.problem()
        cap, data = problem.capacity + 10, problem.data + 10
        y = -2 + .3 * cap + .7 * data
        bounded = fit.LanguageProblem(y, cap, data, problem.weights, 0, float(y.min()))
        solution = n.solve_language(bounded, .08)
        self.assertGreater(solution.L_inf, 0)
        self.assertLess(solution.L_inf, 1e-6)
        self.assertTrue(solution.floor_saturated)
        self.assertGreater(solution.A, 0)
        self.assertGreater(solution.B, 0)

    def test_profile_analytic_gradient_matches_difference(self):
        from scipy.optimize._numdiff import approx_derivative
        rng = np.random.default_rng(12)
        count = 90
        N = np.exp(rng.uniform(np.log(1e8), np.log(8e8), count))
        D = np.exp(rng.uniform(np.log(1e8), np.log(2e9), count))
        E = rng.choice([1, 4, 16, 32], count)
        K = np.ones(count) * 50
        coordinates = np.asarray([.35, .54, .26, .25, -.13, .04])
        cap, data, _, _ = fit.predictor_terms_and_jacobian("J0", coordinates, N, D, E, K)
        y = -3 + .3 * cap + .7 * data + rng.normal(scale=.01, size=count)
        problem = fit.ProjectedProblem(y, np.ones(count) / count, N, D, E, K,
            tuple(["language"] * count), np.ones(count) * -8)
        profiler = n.J0Profile(problem, .02)
        _, gradient = profiler.evaluate(coordinates)
        expected = approx_derivative(lambda c: profiler.evaluate(c)[0].objective,
            coordinates, method="3-point", abs_step=1e-6)
        np.testing.assert_allclose(gradient, expected, rtol=1e-3, atol=1e-8)


class RecoveryResidualTest(unittest.TestCase):
    def test_summary_weights_trajectories_not_rows(self):
        rows = pd.DataFrame({"trajectory_id": ["a", "a", "b"],
            "residual": [1., 1., 3.], "squared_error": [1., 1., 9.]})
        actual = summary(rows)
        self.assertAlmostEqual(actual["rmse"], np.sqrt(5))
        self.assertAlmostEqual(actual["mean_prediction_minus_observed"], 2)
        self.assertEqual(actual["trajectories"], 2)


class RecoveryResultLoaderTest(unittest.TestCase):
    def test_only_new_refined_result_gets_explicit_amendment(self):
        evidence = {"refined_result": {"path": "/tmp/refined.json"},
                    "original_result": {"path": "/tmp/original.json"}}
        with patch.object(refit_audit, "validate_acceptance", return_value=evidence), \
                patch.object(a, "bound", return_value=(Path("/tmp/refined.json"), {"result_sha256": "new"})), \
                patch.object(recovery.campaign, "_completion_receipt"), \
                patch.object(recovery.campaign, "scientific_result", return_value={"ordinary": True}) as ordinary:
            loader = recovery.amended_result_loader({"audit_plan": {"path": "/tmp/plan.json"}})
            self.assertEqual(loader(Path("/tmp/refined.json"))["audit_amendment"], evidence)
            self.assertEqual(loader(Path("/tmp/original.json")), {"ordinary": True})
            self.assertEqual(loader(Path("/tmp/another-cell.json")), {"ordinary": True})
            self.assertEqual(ordinary.call_count, 2)

    def test_failed_refinement_cannot_produce_loader(self):
        with patch.object(refit_audit, "validate_acceptance", side_effect=a.BaselineRefitError("not accepted")):
            with self.assertRaisesRegex(a.BaselineRefitError, "not accepted"):
                recovery.amended_result_loader({"audit_plan": {"path": "/tmp/plan.json"}})


if __name__ == "__main__":
    unittest.main()
