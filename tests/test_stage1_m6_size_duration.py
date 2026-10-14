from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as m, planning as p, execution as e, reporting as r
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import numerics


def fixture():
    rng = np.random.default_rng(257)
    count = 120
    x = np.array([.35, .5, .27, .3, -.1, .04, .03, .01, -.02, -.01, .02, .1, .05, .03, .1])
    N = np.exp(rng.uniform(np.log(5e7), np.log(8e8), count))
    D = np.exp(rng.uniform(np.log(1e7), np.log(2e9), count))
    E = rng.choice([1, 4, 16, 32], count)
    K = rng.choice([1, 12, 50, 200], count)
    fractions = rng.dirichlet([2, 2, 2], count)
    other = (K-1)*D
    problem = fit.ProjectedProblem.from_arrays(y=np.ones(count), weights=np.ones(count)/count,
        N=N, D=D, E=E, K=K, language_ids=["a"]*60+["b"]*60, physical_floors=np.full(count, -8.),
        D_family=other*fractions[:, 0], D_script=other*fractions[:, 1], D_other=other*fractions[:, 2],
        regularization=fit.M5M6Regularization(strength=.01, m2_reference_objective=.0005))
    cap, data, _, _ = m.terms(x, problem)
    y = -3+.3*cap+.7*data+rng.normal(scale=.002, size=count)
    y[-1] += .2
    problem = replace(problem, y=y)
    bounds = (*fit.initial_bounds("M6"), fit.OuterBound("zeta", 0., .5, "nonnegative"))
    return problem, x, bounds


class ModelTest(unittest.TestCase):
    def test_scaled_optimizer_stopping_remains_stricter_than_acceptance(self):
        self.assertEqual(m.OUTER_SETTINGS["gtol"]/m.OBJECTIVE_SCALE, m.OPTIMIZER_UNSCALED_GTOL)
        self.assertLess(m.OPTIMIZER_UNSCALED_GTOL, p.PROTOCOL["gradient_tolerance"])
        self.assertEqual(p.PROTOCOL["gradient_tolerance"], 1e-8)
        self.assertEqual(p.PROTOCOL["objective_tolerance"], 1e-6)
        self.assertEqual(p.PROTOCOL["prediction_tolerance"], 1e-4)

    def test_inner_refinement_preserves_huber_and_kkt_contract(self):
        problem, x, _ = fixture()
        profile = m.Profile(problem, .05)
        cap, data, _, _ = m.terms(x, problem)
        for name, indices in profile.groups:
            inner = profile.inner_problem(indices, cap, data)
            original = numerics.original_solve(inner, .05, language=name)
            refined = numerics.refine_language(inner, .05, language=name)
            self.assertLessEqual(refined.objective, original.objective+1e-13)
            np.testing.assert_allclose(refined.prediction, original.prediction, atol=1e-8, rtol=0)
            with patch.object(numerics, "original_solve", side_effect=fit.InnerSolveError("KKT")):
                recovered = numerics.solve_language(inner, .05, language=name)
            np.testing.assert_array_equal(recovered.prediction, refined.prediction)
            weights = inner.weights/inner.weights.sum()
            scales = np.sqrt(np.sum(weights[:, None]*inner.design**2, axis=0))
            values = np.array([refined.L_inf, refined.A, refined.B])
            gradient = fit.language_objective_and_gradient(values, inner, .05)[1]/(scales*inner.weights.sum())
            lower, upper = (b*scales for b in inner.strict_bounds)
            self.assertLessEqual(np.max(np.abs(values*scales-np.clip(values*scales-gradient, lower, upper))), 1e-10)

    def test_exact_null_terms_jacobian_and_penalty(self):
        problem, x, _ = fixture()
        x[-1] = 0
        a = m.terms(x, problem)
        b = fit.predictor_terms_and_jacobian("M6", x[:14], problem.N, problem.D, problem.E, problem.K,
            D_family=problem.D_family, D_script=problem.D_script, D_other=problem.D_other)
        for i in range(4):
            np.testing.assert_array_equal(a[i] if i < 2 else a[i][:, :14], b[i])
        value, gradient = m.penalty(x, problem)
        old, old_gradient = fit.regularization_objective_and_gradient("M6", x[:14], problem.regularization)
        self.assertEqual(value, old)
        np.testing.assert_array_equal(gradient[:14], old_gradient)
        self.assertEqual(gradient[-1], 0)

    def test_all_basis_gradients(self):
        problem, x, _ = fixture()
        cap, data, cj, dj = m.terms(x, problem)
        for term, jac in ((0, cj), (1, dj)):
            numeric = approx_derivative(lambda v: m.terms(v, problem)[term], x, method="3-point", abs_step=1e-6)
            np.testing.assert_allclose(jac, numeric, atol=1e-7, rtol=1e-6)

    def test_profile_gradient_and_direct_objective_gradient(self):
        problem, x, bounds = fixture()
        profile = m.Profile(problem, .05)
        selected, gradient, prediction, _ = profile.evaluate(x)
        numeric = approx_derivative(lambda v: profile.evaluate(v)[0]["objective"], x, abs_step=1e-6).ravel()
        np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-5)
        direct = m.Direct(profile, selected, bounds)
        objective, joint_gradient, joint_prediction, _ = direct.evaluate(direct.initial)
        self.assertAlmostEqual(objective, selected["objective"], places=13)
        np.testing.assert_allclose(prediction, joint_prediction, atol=1e-12, rtol=0)
        numeric = approx_derivative(lambda v: direct.evaluate(v)[0], np.array(direct.initial), abs_step=1e-6).ravel()
        np.testing.assert_allclose(joint_gradient, numeric, atol=1e-8, rtol=1e-5)

    def test_synthetic_optimizer_and_direct_smoke(self):
        problem, x, bounds = fixture()
        # Optimize a one-coordinate slice by tight but nonzero bounds on others.
        local = tuple(fit.OuterBound(b.name, float(v)-1e-10, float(v)+1e-10, b.kind)
            if b.name != "zeta" else b for b, v in zip(bounds, x))
        profile = m.Profile(problem, .05)
        initial = profile.evaluate(x)[0]["objective"]
        selected = profile.run(x, local)
        self.assertTrue(selected["success"], selected)
        self.assertLessEqual(selected["objective"], initial+1e-12)
        joint = m.Direct(profile, selected, local).run()
        self.assertTrue(joint["success"], joint)
        self.assertTrue(m.equivalence(selected, joint, problem)["equivalent"])

    def test_invalid_coordinates_and_equivalence_tolerances(self):
        problem, x, _ = fixture()
        for value in (x[:14], np.full(15, np.nan)):
            with self.assertRaises(ValueError):
                m.terms(value, problem)
        selected = m.Profile(problem, .05).evaluate(x)[0] | {"success": True}
        bad = copy.deepcopy(selected)
        bad["language_coefficients"][0]["L_inf"] += .001
        self.assertFalse(m.equivalence(selected, bad, problem)["equivalent"])
        bad = selected | {"objective": selected["objective"]*1.000002}
        self.assertFalse(m.equivalence(selected, bad, problem)["equivalent"])
        self.assertFalse(m.equivalence(selected, selected | {"success": False}, problem)["equivalent"])

    def test_positive_zeta_targets_size_not_a_constant_offset(self):
        problem, x, _ = fixture()
        x[-1] = 0
        null = m.terms(x, problem)[1]
        x[-1] = .1
        ratio = m.terms(x, problem)[1]/null
        np.testing.assert_allclose(ratio, (problem.N/1e8)**.1)


class WorkflowTest(unittest.TestCase):
    def plan(self):
        anchors = {f: [{"start_id": f"old:{i}", "coordinates": [float(i)]*14} for i in range(8)] for f in p.FOLDS}
        return {"sha256": "sealed", "tasks": p.tasks_for(anchors)}

    def test_exact_grid_and_pilot_reuse(self):
        plan = self.plan()
        p.validate_tasks(plan["tasks"])
        self.assertEqual(len(plan["tasks"]), 72)
        pilot = [s for t in plan["tasks"][:3] for s in t["starts"]]
        self.assertEqual([s["coordinates"][-1] for s in pilot], list(m.ZETA_STARTS))
        self.assertEqual({s["fold"] for s in pilot}, {"largest_N_act"})
        self.assertEqual(len({s["start_id"] for t in plan["tasks"] for s in t["starts"]}), 144)
        plan["tasks"][1] = plan["tasks"][0]
        with self.assertRaises(ValueError):
            p.validate_tasks(plan["tasks"])

    def test_chunk_identity_and_missing_failed_records(self):
        plan = self.plan()
        starts = plan["tasks"][0]["starts"]
        records = [dict(s, initial_coordinates=s["coordinates"], success=False) for s in starts]
        chunk = {"plan_sha256": "sealed", "task_id": 0, "attempts": records}
        self.assertEqual(len(e.validate_chunk(plan, 0, chunk)), 2)
        records.pop()
        with self.assertRaises(ValueError):
            e.validate_chunk(plan, 0, chunk)

    def test_pilot_is_only_runtime_and_training_success(self):
        rows = [{"success": True, "runtime_seconds": 100.} for _ in range(6)]
        self.assertEqual(e.pilot_decision(rows)["status"], "passed")
        rows[0]["success"] = False
        self.assertEqual(e.pilot_decision(rows)["status"], "blocked")
        rows[0] = {"success": True, "runtime_seconds": 1000.}
        self.assertEqual(e.pilot_decision(rows)["status"], "blocked")

    def test_audit_selection_uses_training_only_and_deterministic_tie(self):
        rows = [{"success": True, "objective": 1., "start_id": s, "test_rmse": v} for s, v in (("b", .1), ("a", 2.), ("c", .01))]
        selected, audits = e.select_audits(rows)
        self.assertEqual(selected["start_id"], "a")
        self.assertEqual([v["start_id"] for v in audits], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            e.select_audits(rows[:2])

    def test_seals_are_create_only_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"x.json"
            result = p.write_sealed(path, {"value": 1})
            self.assertEqual(p.read_sealed(path), result)
            with self.assertRaises(RuntimeError):
                p.write_sealed(path, {"value": 2})
            with patch.object(p.execution, "read_json", return_value=dict(result, value=2)):
                with self.assertRaises(RuntimeError):
                    p.read_sealed(path)

    def test_worker_requires_allocation_and_single_threads(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                e.runtime_context()
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test", "OPENBLAS_NUM_THREADS": "2"}, clear=True):
            with self.assertRaises(ValueError):
                e.runtime_context()

    def test_script_preparation_does_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = {"python_executable": "/usr/bin/python", "partition": "normal", "account": "test", "memory": "4G", "time_limit": "01:00:00"}
            with patch.object(p.subprocess, "run") as called:
                scripts = p.scripts(Path(directory), profile)
            self.assertEqual(len(scripts), 6)
            self.assertTrue(all(c.args[0][:2] == ["bash", "-n"] for c in called.call_args_list))
            self.assertIn("3-71%12", (Path(directory)/"slurm/grid.sbatch").read_text())

    def test_paired_bootstrap_identical_predictions_have_zero_difference(self):
        records = []
        for model in ("M6", m.MODEL):
            for fold in p.FOLDS:
                for t in range(5):
                    records.append(dict(model=model, fold=fold, trajectory_id=f"t{t}", readout_id=f"r{t}", language="a", split="test", squared_error=.01*(t+1)))
        frame = pd.DataFrame(records)
        result = r.metrics_and_bootstrap(frame, replicates=100)
        self.assertEqual(result["paired_intervals"][0]["p025"], 0.)
        self.assertEqual(result["paired_intervals"][0]["p975"], 0.)
        with self.assertRaises(ValueError):
            r.metrics_and_bootstrap(frame.iloc[:-1], replicates=100)

    def test_pressure_is_not_prediction_residual_and_does_not_extrapolate(self):
        records = []
        for name in ("M6", m.MODEL):
            for k in (12, 200):
                for i, dt in enumerate((1., 2., 4.) if k == 12 else (1.5, 3., 8.)):
                    loss = 2-np.log(dt)+(.05 if k == 200 else 0.)
                    records.append(dict(model=name, fold="largest_N_act", ladder="D", N_act=5e8, E=32,
                        K=k, language="a", trajectory_id=f"k{k}", readout_id=f"k{k}r{i}", D_t=dt,
                        D_total=k*dt, adjusted_loss=loss, raw_nll=loss+5, unigram_nll=5,
                        prediction=loss+.7))
        pressure = r.pressure_curves(pd.DataFrame(records))
        self.assertEqual(len(pressure), 4)
        np.testing.assert_allclose(pressure.pressure, .05, atol=1e-12)
        np.testing.assert_allclose(pressure.pressure_error, 0., atol=1e-12)
        self.assertFalse(pressure.D_t.eq(8).any())

    def test_frozen_gates_reject_tail_regression_and_null_zeta(self):
        rows = []
        for name, error in (("M6", .1), (m.MODEL, .08)):
            for k in (12, 50):
                rows.append(dict(model=name, fold="largest_N_act", split="test", E=32, K=k,
                    D_total=p.PROTOCOL["long_total_tokens"], trajectory_id=f"k{k}", residual=error,
                    squared_error=error**2, huber_influence_ratio=1.))
        frame = pd.DataFrame(rows)
        sizes = pd.DataFrame([dict(model=name, left_ladder="C", right_ladder="D",
            D_total=p.PROTOCOL["short_total_tokens"], predicted_size_gain=gain, observed_size_gain=.04,
            target_and_component_exposure_matched=True) for name, gain in (("M6", .09), (m.MODEL, .05))])
        reduction = {"folds": {f: {"selected": {"coordinates": [0.]*14+[.1]}} for f in p.FOLDS}}
        statistics = {"metrics": {m.MODEL: dict(zip([*p.FOLDS, "mean"], [.04, .03, .04, .0367]))}}
        self.assertTrue(r.gates(statistics, frame, sizes, reduction)["passed"])
        frame.loc[frame.model.eq(m.MODEL), "squared_error"] = .5
        reduction["folds"][p.FOLDS[0]]["selected"]["coordinates"][-1] = 0.
        decision = r.gates(statistics, frame, sizes, reduction)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["checks"]["positive_zeta_all_folds"])
        self.assertFalse(decision["checks"]["largest_32B_E32_K12"])


if __name__ == "__main__":
    unittest.main()
