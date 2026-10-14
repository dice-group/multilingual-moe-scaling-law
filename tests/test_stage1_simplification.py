"""Inference-only checks of compact algebra and coordinate chain rules.

No optimizer runs here. The profiled wrapper uses a fixed-coefficient inner
double; actual profile/direct optimizer agreement belongs to Slurm audits.
"""
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.simplification import model as s
from multilingual_megatron.experiments.scaling.stage1_fit.simplification import campaign
from multilingual_megatron.experiments.scaling.stage1_fit.simplification import reporting
from multilingual_megatron.experiments.scaling.stage1_fit.data import canonical_json_sha256
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as parent


def fixture():
    grid = np.asarray(list(product((.9, 2., 6.), (.2, 1., 8.), (1., 8., 32.), (12., 50., 200.))))
    n, d, e, k = grid.T
    p = SimpleNamespace(N=n*1e8, D=d*1e9, E=e, K=k,
        D_family=d*2e9, D_script=d*3e9, D_other=d*4e9,
        language_ids=tuple("a" if i % 2 else "b" for i in range(len(n))),
        physical_floors=np.full(len(n), -6.), y=-2.+.03*np.sin(np.arange(len(n))),
        weights=np.arange(1., len(n)+1))
    p.weights /= p.weights.sum()
    p.regularization = fit.M5M6Regularization(.01, .002)
    x = np.array([.35, .32, -.08, -.02, .04, .09, .025, .15])
    box = tuple(s.Bound(name, .001 if kind == "positive" else -1., 1., kind)
                for name, kind in zip((*fit.outer_parameter_names("M6"), "zeta"),
                    (*fit.outer_parameter_kinds("M6"), "nonnegative")))
    return p, x, box


def fixed_inner(problem, delta, *, language):
    floor, a, b = (-3.5, .3, .7) if language == "a" else (-3., .5, .4)
    prediction = floor+a*problem.capacity+b*problem.data
    low, high = problem.strict_bounds
    return SimpleNamespace(language=language, L_inf=floor, A=a, B=b,
        prediction=prediction, floor_fraction=(floor-low[0])/(high[0]-low[0]),
        floor_saturated=False, A_zero_contact=False, B_zero_contact=False,
        message="fixed synthetic coefficients; no optimization")


class SimplificationTest(unittest.TestCase):
    def test_explicit_eight_parameter_equation(self):
        p, x, _ = fixture()
        alpha, beta, delta, omega, phi, phi32, psi, zeta = x
        theta = (phi32-phi)/np.log(32.)
        expected_cap = p.E**delta*p.K**(phi+theta*np.log(p.E))*(p.N/1e8)**(-alpha)
        expected_data = p.E**omega*p.K**psi*(p.N/1e8)**zeta*(p.D/1e9)**(-beta)
        cap, data, _, _ = s.terms(x, p, "S8")
        np.testing.assert_allclose(cap, expected_cap, rtol=2e-14)
        np.testing.assert_allclose(data, expected_data, rtol=2e-14)

    def test_null_variants_and_exact_reduced_roundtrip(self):
        p, x, _ = fixture()
        for name, count in zip(s.MODELS, (8, 7, 7)):
            matrix, names = s.mapping(name)
            self.assertEqual(matrix.shape, (15, count))
            self.assertEqual(len(names), count)
            reduced = s.restrict(s.mapping("S8")[0] @ x, name)
            full = matrix @ reduced
            np.testing.assert_allclose(s.restrict(full, name), reduced, atol=1e-15)
            self.assertTrue(np.all(full[10:14] == 0.))
            self.assertEqual(full[0], full[1])
            self.assertEqual(full[2], full[3])
            self.assertEqual(full[8], full[9])
            if name == "S7-no-ND":
                self.assertEqual(full[14], 0.)
            if name == "S7-no-EK":
                self.assertEqual(full[6], full[7])
            for actual, expected in zip(s.terms(reduced, p, name)[:2], parent.terms(full, p)[:2]):
                np.testing.assert_array_equal(actual, expected)

    def test_basis_chain_rule_all_variants(self):
        p, x, _ = fixture()
        full = s.mapping("S8")[0] @ x
        for name in s.MODELS:
            reduced = s.restrict(full, name)
            result = s.terms(reduced, p, name)
            for term in (0, 1):
                numeric = approx_derivative(lambda y: s.terms(y, p, name)[term], reduced, abs_step=1e-6)
                np.testing.assert_allclose(numeric, result[term+2], atol=1e-8, rtol=1e-6)

    def test_inherited_ridge_and_gradient(self):
        p, x, _ = fixture()
        for name in s.MODELS:
            matrix, _ = s.mapping(name)
            reduced = s.restrict(s.mapping("S8")[0] @ x, name)
            full = matrix @ reduced
            value, grad = parent.penalty(full, p)
            reg = p.regularization
            expected = .5*reg.strength*reg.m2_reference_objective*((full[7]-full[6])/reg.phi_endpoint_scale)**2
            self.assertAlmostEqual(value, expected, places=16)
            numeric = approx_derivative(lambda y: parent.penalty(matrix@y, p)[0], reduced, abs_step=1e-6)
            np.testing.assert_allclose(numeric.ravel(), matrix.T@grad, atol=1e-12, rtol=1e-7)
            if name == "S7-no-EK":
                self.assertEqual(value, 0.)

    def test_profile_wrapper_gradient_without_inner_optimizer(self):
        p, x, _ = fixture()
        for name in s.MODELS:
            reduced = s.restrict(s.mapping("S8")[0] @ x, name)
            profile = s.Profile(p, .05, name)
            with patch.object(parent, "solve_language", side_effect=fixed_inner):
                result, gradient, _, jac = profile.evaluate(reduced)
                numeric = approx_derivative(lambda y: profile.evaluate(y)[0]["objective"], reduced, abs_step=1e-6)
            self.assertEqual(result["coordinates"], reduced.tolist())
            self.assertEqual(jac.shape, (len(p.y), len(reduced)))
            np.testing.assert_allclose(numeric.ravel(), gradient, atol=1e-9, rtol=1e-6)

    def test_direct_shared_and_nuisance_gradients_without_optimizer(self):
        p, x, box = fixture()
        for name in s.MODELS:
            reduced = s.restrict(s.mapping("S8")[0] @ x, name)
            profile = s.Profile(p, .05, name)
            with patch.object(parent, "solve_language", side_effect=fixed_inner):
                selected, _, prediction, _ = profile.evaluate(reduced)
            direct = s.Direct(profile, selected, s.bounds(box, name))
            value, gradient, actual, _ = direct.evaluate(direct.initial)
            self.assertAlmostEqual(value, selected["objective"], places=14)
            np.testing.assert_allclose(actual, prediction, atol=1e-14)
            numeric = approx_derivative(lambda y: direct.evaluate(y)[0], direct.initial, abs_step=1e-6)
            np.testing.assert_allclose(numeric.ravel(), gradient, atol=1e-9, rtol=1e-6)
            np.testing.assert_allclose(s.predict(p, selected, name), prediction, atol=1e-14)

    def test_tied_bounds_are_intersections(self):
        _, _, box = fixture()
        box = list(box)
        box[0] = s.Bound("alpha_e1", .1, .8, "positive")
        box[1] = s.Bound("alpha_e32", .2, .9, "positive")
        compact = s.bounds(box, "S8")
        self.assertEqual((compact[0].lower, compact[0].upper), (.2, .8))
        box[1] = s.Bound("alpha_e32", .8, .9, "positive")
        with self.assertRaisesRegex(ValueError, "empty tied-coordinate"):
            s.bounds(box, "S8")

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown compact law"):
            s.mapping("S9")

    def test_training_partition_retains_languages_excluded_from_scoring(self):
        rows = pd.DataFrame({"trajectory_id": ["train", "train", "test", "test"],
                             "language": ["a", "b", "a", "b"], "D_total": [1e9]*4})
        member = {"train_trajectory_ids": ["train"], "test_trajectory_ids": ["test"],
                  "score_languages": ["a"]}
        training = campaign.reuse.partition(rows, member, "train")
        testing = campaign.reuse.partition(rows, member, "test")
        self.assertEqual(set(training.language), {"a", "b"})
        self.assertEqual(set(testing.language), {"a"})
        member["test_trajectory_ids"] = ["train"]
        with self.assertRaisesRegex(ValueError, "split overlaps"):
            campaign.reuse.partition(rows, member, "train")

    def test_score_alignment_and_trajectory_weighting(self):
        reference = pd.DataFrame({"trajectory_id": ["small", "large", "large", "large"],
            "readout_id": ["r0", "r1", "r2", "r3"], "language": ["a"]*4,
            "adjusted_loss": [0.]*4})
        prediction = reference.assign(prediction=[2., 0., 0., 0.]).iloc[::-1]
        score = reporting.score(prediction, reference)
        self.assertTrue(score["complete"])
        self.assertAlmostEqual(score["rmse"], np.sqrt(2.))
        self.assertAlmostEqual(score["bias"], 1.)
        self.assertAlmostEqual(reporting.score(prediction, reference, endpoint=True)["rmse"], 1.)
        self.assertFalse(reporting.score(prediction.iloc[:-1], reference)["complete"])
        with self.assertRaisesRegex(ValueError, "duplicate score identity"):
            reporting.score(pd.concat([prediction, prediction.iloc[:1]]), reference)
        wrong = prediction.copy()
        wrong.loc[0, "adjusted_loss"] = .1
        with self.assertRaises(AssertionError):
            reporting.score(wrong, reference)

    def test_worker_load_verifies_only_own_training_targets(self):
        train = {"path": "/tmp/screen/packets/E32-train.parquet"}
        test = {"path": "/tmp/screen/packets/E32-test.parquet"}
        other = {"path": "/tmp/screen/packets/largest_N_act-train.parquet"}
        source = {"path": "/tmp/screen/source/model.py"}
        script = {"path": "/tmp/screen/run.sbatch"}
        plan = {"schema": campaign.SCHEMA, "protocol": campaign.PROTOCOL,
            "cells": [{"fold": "E32"}], "frozen_inputs": [train, test, other],
            "source": [source], "script": script, "environment": {"test": True}}
        plan["sha256"] = canonical_json_sha256(plan)
        with patch.object(campaign.core, "read_json", return_value=plan), \
             patch.object(campaign.reuse, "environment", return_value={"test": True}), \
             patch.object(campaign.reuse, "verify") as verify:
            campaign.load("/tmp/screen/plan.json", training_cell=0)
        self.assertEqual([call.args[0] for call in verify.call_args_list], [train, source, script])

    def test_start_selection_uses_training_objective_and_rejects_failed_better_fit(self):
        attempts = [{"task_id": 0, "objective": .3, "success": True},
                    {"task_id": 1, "objective": .2, "success": True}]
        self.assertEqual(campaign.reuse.selected_attempt(attempts)["task_id"], 1)
        attempts.append({"task_id": 2, "objective": .1, "success": False})
        with self.assertRaisesRegex(ValueError, "nonstationary attempt"):
            campaign.reuse.selected_attempt(attempts)


if __name__ == "__main__":
    unittest.main()
