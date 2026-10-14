from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multilingual_megatron.experiments.scaling.stage1_fit.m5_m6_workflow import (
    campaign as m5_m6_campaign,
)
from multilingual_megatron.experiments.scaling.stage1_fit.fit import (
    EXPLORATORY_COMPOSITION_MODELS,
    M5M6Regularization,
    PRIMARY_MODELS,
    initial_bounds,
    outer_parameter_names,
    predictor_terms,
    predictor_terms_and_jacobian,
    regularization_objective_and_gradient,
    shared_from_outer,
)
from multilingual_megatron.experiments.scaling.stage1_fit.m5_m6 import (
    ANCHOR_COUNT,
    CHUNK_SIZE,
    EQUAL_FOLD_M2_GATE,
    K200_M1_GATE,
    M6_SEEDS,
    RIDGE_LAMBDAS,
    _bound_scaled_rank,
    _reduced_current_exploratory,
    _regularization,
    composition_starts,
    parser,
)


def _anchor_result() -> dict[str, object]:
    coordinates = np.linspace(-0.4, 0.6, len(outer_parameter_names("M5")))
    attempts = []
    for index in range(10):
        attempts.append({
            "success": True,
            "objective": 0.01 + index * 0.001,
            "start_id": f"historical:{index:02d}",
            "coordinates": (coordinates + index * 0.01).tolist(),
        })
    return {"attempts": attempts}


class Stage1M5M6CampaignTest(unittest.TestCase):
    def test_models_remain_exploratory_and_parameter_counts_are_locked(self) -> None:
        self.assertEqual(PRIMARY_MODELS, ("M0", "M1", "M2"))
        self.assertEqual(EXPLORATORY_COMPOSITION_MODELS, ("M5", "M6"))
        self.assertEqual(len(outer_parameter_names("M5")), 11)
        self.assertEqual(len(outer_parameter_names("M6")), 14)

    def test_screen_grid_and_gates_are_locked(self) -> None:
        self.assertEqual(ANCHOR_COUNT, 8)
        self.assertEqual(CHUNK_SIZE, 2)
        self.assertEqual(RIDGE_LAMBDAS, (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0))
        self.assertEqual(M6_SEEDS, (0.0, 0.03, 0.1))
        self.assertEqual(EQUAL_FOLD_M2_GATE, 0.05066808)
        self.assertEqual(K200_M1_GATE, 0.05441795)
        starts = 3 * len(RIDGE_LAMBDAS) * (8 + 8 * 27)
        tasks = 3 * len(RIDGE_LAMBDAS) * (8 // CHUNK_SIZE + (8 * 27) // CHUNK_SIZE)
        self.assertEqual((starts, tasks), (4704, 2352))
        audit_manifests = 2 * 3 * len(RIDGE_LAMBDAS) * 2
        audit_tasks = 2 * 3 * len(RIDGE_LAMBDAS) * (2 + 1)
        self.assertEqual((audit_manifests, audit_tasks), (84, 126))

    def test_m5_uses_eight_exact_anchors_and_m6_crosses_27_extensions(self) -> None:
        anchor = _anchor_result()
        m5 = composition_starts("M5", anchor, namespace="test")
        m6 = composition_starts("M6", anchor, namespace="test")
        self.assertEqual(len(m5), 8)
        self.assertEqual(len(m6), 216)
        self.assertTrue(all(item.values.shape == (11,) for item in m5))
        self.assertTrue(all(item.values.shape == (14,) for item in m6))
        np.testing.assert_array_equal(m5[0].values, m6[0].values[:11])
        extensions = {tuple(item.values[-3:].tolist()) for item in m6[:27]}
        self.assertEqual(extensions, {
            (family, script, other)
            for family in M6_SEEDS for script in M6_SEEDS for other in M6_SEEDS
        })

    def test_regularization_mapping_has_exact_manifest_contract(self) -> None:
        value = _regularization(0.3, m2_reference_objective=0.0125)
        self.assertEqual(value, {
            "schema_version": "stage1_m5_m6_ridge/v1",
            "lambda": 0.3,
            "m2_reference_objective": 0.0125,
            "phi_endpoint_scale": 0.05,
            "psi_endpoint_scale": 0.05,
            "gamma_n_scale": 0.1,
        })

    def test_m5_and_m6_nesting_is_exact(self) -> None:
        outer = np.asarray([
            0.28, 0.36, 0.22, 0.19, -0.10, 0.04,
            0.08, -0.02, -0.04, 0.03, 0.02,
        ], dtype=np.float64)
        predictors = (
            np.asarray([1.0e8, 3.0e8, 7.0e8]),
            np.asarray([2.0e8, 1.0e9, 5.0e9]),
            np.asarray([1.0, 16.0, 32.0]),
            np.asarray([1.0, 50.0, 200.0]),
        )
        m2_kn = predictor_terms("M2-KN", shared_from_outer("M2-KN", outer),
            *predictors)
        m5 = predictor_terms("M5", shared_from_outer("M5", outer), *predictors)
        for expected, actual in zip(m2_kn, m5):
            np.testing.assert_array_equal(actual, expected)

        family = np.asarray([0.0, 8.0e8, 3.0e9])
        script = np.asarray([1.0e8, 2.0e8, 7.0e8])
        other = np.asarray([7.0e8, 4.0e9, 1.0e10])
        m6 = predictor_terms("M6", shared_from_outer("M6", np.r_[outer, 0.0, 0.0, 0.0]),
            *predictors, D_family=family, D_script=script, D_other=other)
        for expected, actual in zip(m5, m6):
            np.testing.assert_array_equal(actual, expected)

    def test_m6_predictor_jacobian_matches_finite_differences(self) -> None:
        outer = np.asarray([
            0.28, 0.36, 0.22, 0.19, -0.10, 0.04,
            0.08, -0.02, -0.04, 0.03, 0.02,
            0.20, 0.10, 0.08,
        ], dtype=np.float64)
        predictors = (
            np.asarray([1.0e8, 3.0e8, 7.0e8]),
            np.asarray([2.0e8, 1.0e9, 5.0e9]),
            np.asarray([1.0, 16.0, 32.0]),
            np.asarray([1.0, 50.0, 200.0]),
        )
        components = {
            "D_family": np.asarray([0.0, 8.0e8, 3.0e9]),
            "D_script": np.asarray([1.0e8, 2.0e8, 7.0e8]),
            "D_other": np.asarray([7.0e8, 4.0e9, 1.0e10]),
        }
        _, _, capacity_jacobian, data_jacobian = predictor_terms_and_jacobian(
            "M6", outer, *predictors, **components)
        epsilon = 1e-6
        for column in range(outer.size):
            plus, minus = outer.copy(), outer.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            capacity_plus, data_plus = predictor_terms(
                "M6", shared_from_outer("M6", plus), *predictors, **components)
            capacity_minus, data_minus = predictor_terms(
                "M6", shared_from_outer("M6", minus), *predictors, **components)
            np.testing.assert_allclose(capacity_jacobian[:, column],
                (capacity_plus - capacity_minus) / (2.0 * epsilon),
                rtol=3e-7, atol=3e-10)
            np.testing.assert_allclose(data_jacobian[:, column],
                (data_plus - data_minus) / (2.0 * epsilon),
                rtol=3e-7, atol=3e-10)

    def test_ridge_gradient_matches_finite_differences(self) -> None:
        outer = np.asarray([
            0.28, 0.36, 0.22, 0.19, -0.10, 0.04,
            0.08, -0.02, -0.04, 0.03, 0.02,
        ], dtype=np.float64)
        regularization = M5M6Regularization(
            strength=0.3, m2_reference_objective=0.0125)
        objective, gradient = regularization_objective_and_gradient(
            "M5", outer, regularization)
        self.assertGreater(objective, 0.0)
        epsilon = 1e-7
        numerical = np.empty_like(outer)
        for column in range(outer.size):
            plus, minus = outer.copy(), outer.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            plus_value = regularization_objective_and_gradient(
                "M5", plus, regularization)[0]
            minus_value = regularization_objective_and_gradient(
                "M5", minus, regularization)[0]
            numerical[column] = (plus_value - minus_value) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-8, atol=2e-10)

    def test_bound_scaled_prediction_jacobian_rank_uses_relative_svd_gate(self) -> None:
        bounds = initial_bounds("M5")
        widths = np.asarray([item.upper - item.lower for item in bounds])
        full = np.diag(1.0 / widths)
        result = _bound_scaled_rank(full, bounds)
        self.assertEqual(result["rank"], 11)
        self.assertTrue(result["full_rank"])
        self.assertEqual(result["relative_svd_threshold"], 1e-8)
        deficient = full.copy()
        deficient[:, -1] = 0.0
        result = _bound_scaled_rank(deficient, bounds)
        self.assertEqual(result["rank"], 10)
        self.assertFalse(result["full_rank"])

    def test_campaign_result_reader_requires_exact_current_provenance(self) -> None:
        manifest_path = Path("/tmp/current-m5-m6-manifest.json")
        result = {
            "schema_version": "stage1_fit_reduced/v2",
            "review_only": True,
            "receipt_path": "/tmp/current-m5-m6-receipt.json",
            "manifest_path": str(manifest_path),
            "manifest_sha256": "manifest",
            "membership": {"membership_sha256": "membership"},
            "status": "reduced",
            "selected": {"objective": 0.1},
        }
        receipt = {
            "schema_version": "stage1_fit_complete/v2",
            "manifest_sha256": "manifest",
        }
        manifest = {
            "review_only": True,
            "manifest_sha256": "manifest",
            "membership": {"membership_sha256": "membership"},
        }
        with patch.object(m5_m6_campaign, "read_json", side_effect=(result, receipt)), \
                patch.object(m5_m6_campaign, "_sealed", side_effect=lambda value, _: value), \
                patch.object(m5_m6_campaign, "load_manifest", return_value=manifest) as loader:
            loaded_result, loaded_manifest = _reduced_current_exploratory(
                Path("/tmp/current-m5-m6-result.json"))
        self.assertIs(loaded_result, result)
        self.assertIs(loaded_manifest, manifest)
        loader.assert_called_once_with(manifest_path)

    def test_cli_emits_but_cannot_submit_or_skip_to_nested_confirmation(self) -> None:
        subparsers = next(action for action in parser()._actions
            if action.__class__.__name__ == "_SubParsersAction")
        self.assertIn("emit-slurm", subparsers.choices)
        self.assertIn("plan-audits", subparsers.choices)
        self.assertNotIn("submit", subparsers.choices)
        self.assertNotIn("record-submission", subparsers.choices)
        self.assertNotIn("plan-inner", subparsers.choices)
        self.assertNotIn("plan-outer", subparsers.choices)


if __name__ == "__main__":
    unittest.main()
