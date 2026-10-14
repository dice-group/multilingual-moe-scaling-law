from __future__ import annotations

import unittest

import numpy as np

from multilingual_megatron.experiments.scaling.stage1_fit.fit import (
    EXPLORATORY_INTERACTION_MODELS,
    PRIMARY_MODELS,
    SharedParameters,
    effective_exponents,
    multilingual_k_coordinate,
    predictor_terms,
    predictor_terms_and_jacobian,
    shared_from_outer,
)
from multilingual_megatron.experiments.scaling.stage1_fit.interaction import interaction_starts, parser


M2_COORDINATES = np.asarray([
    0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.02, -0.04, 0.03,
], dtype=np.float64)


def _m2_result() -> dict[str, object]:
    attempts = []
    for index in range(12):
        coordinates = M2_COORDINATES.copy()
        coordinates[index % coordinates.size] += (index + 1) * 1e-4
        attempts.append({"success": True, "objective": 1.0 + index * 1e-8,
            "start_id": f"grid:{index:06d}", "coordinates": coordinates.tolist()})
    return {
        "manifest_path": "unused-by-patched-test",
        "selected": {"coordinates": M2_COORDINATES.tolist()},
        "attempts": attempts,
    }


class Stage1FitInteractionModelTest(unittest.TestCase):
    def test_models_are_executable_but_not_registered_candidates(self) -> None:
        self.assertEqual(PRIMARY_MODELS, ("M0", "M1", "M2"))
        self.assertEqual(EXPLORATORY_INTERACTION_MODELS, ("M2-KN", "M2-KD", "M2-KND"))

    def test_multilingual_coordinate_anchors_k1_and_k12(self) -> None:
        actual = multilingual_k_coordinate(np.asarray([1.0, 12.0, 50.0, 200.0]))
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(actual[:2], [0.0, 0.0], rtol=0.0, atol=0.0)
        self.assertGreater(actual[2], 0.0)
        self.assertLess(actual[2], 1.0)
        self.assertAlmostEqual(float(actual[3]), 1.0)

    def test_zero_interactions_exactly_nest_m2(self) -> None:
        predictors = (np.asarray([1e8, 2e8, 4e8, 8e8]), np.asarray([1e8, 5e8, 2e9, 8e9]),
            np.asarray([1.0, 4.0, 16.0, 32.0]), np.asarray([1.0, 12.0, 50.0, 200.0]))
        m2 = shared_from_outer("M2", M2_COORDINATES)
        for model in EXPLORATORY_INTERACTION_MODELS:
            suffix = [0.0, 0.0] if model == "M2-KND" else [0.0]
            candidate = shared_from_outer(model, np.asarray([*M2_COORDINATES, *suffix]))
            for expected, actual in zip(predictor_terms("M2", m2, *predictors),
                    predictor_terms(model, candidate, *predictors)):
                np.testing.assert_array_equal(actual, expected)

    def test_interaction_jacobians_match_finite_difference(self) -> None:
        predictors = (np.asarray([1e8, 3e8, 7e8]), np.asarray([2e8, 1e9, 5e9]),
            np.asarray([1.0, 16.0, 32.0]), np.asarray([1.0, 50.0, 200.0]))
        cases = {
            "M2-KN": np.asarray([*M2_COORDINATES, 0.03]),
            "M2-KD": np.asarray([*M2_COORDINATES, -0.02]),
            "M2-KND": np.asarray([*M2_COORDINATES, 0.03, -0.02]),
        }
        epsilon = 1e-6
        for model, outer in cases.items():
            with self.subTest(model=model):
                _, _, capacity_jac, data_jac = predictor_terms_and_jacobian(model, outer, *predictors)
                for column in range(outer.size):
                    plus, minus = outer.copy(), outer.copy()
                    plus[column] += epsilon; minus[column] -= epsilon
                    cap_plus, data_plus = predictor_terms(model, shared_from_outer(model, plus), *predictors)
                    cap_minus, data_minus = predictor_terms(model, shared_from_outer(model, minus), *predictors)
                    np.testing.assert_allclose(capacity_jac[:, column], (cap_plus-cap_minus)/(2*epsilon), rtol=4e-8, atol=4e-10)
                    np.testing.assert_allclose(data_jac[:, column], (data_plus-data_minus)/(2*epsilon), rtol=4e-8, atol=4e-10)

    def test_nonpositive_effective_exponent_is_rejected(self) -> None:
        parameters = SharedParameters(alpha_0=.1, alpha_1=0.0, beta_0=.2, beta_1=0.0, gamma_n=-.2)
        with self.assertRaisesRegex(FloatingPointError, "non-positive"):
            effective_exponents(parameters, [1.0], [200.0], model="M2-KN")

    def test_start_sets_have_frozen_counts(self) -> None:
        from unittest.mock import patch
        fake_manifest = {"fit": {"model": "M2"}}
        with patch("multilingual_megatron.experiments.scaling.stage1_fit.interaction.load_manifest", return_value=fake_manifest):
            for wave, expected in (("smoke", {"M2-KN": 3, "M2-KD": 3, "M2-KND": 9}),
                    ("screen", {"M2-KN": 24, "M2-KD": 24, "M2-KND": 72})):
                for model, count in expected.items():
                    first = interaction_starts(model, _m2_result(), wave=wave)
                    second = interaction_starts(model, _m2_result(), wave=wave)
                    self.assertEqual(len(first), count)
                    self.assertEqual([item.start_id for item in first], [item.start_id for item in second])
                    for left, right in zip(first, second):
                        np.testing.assert_array_equal(left.values, right.values)

    def test_cli_exposes_only_bounded_waves(self) -> None:
        parsed = parser().parse_args(["plan-wave", "--wave", "smoke", "--profile", "p",
            "--input-receipt", "r", "--holdouts", "h", "--candidate-report", "c", "--dry-run"])
        self.assertEqual(parsed.wave, "smoke")
        self.assertTrue(parsed.dry_run)


if __name__ == "__main__":
    unittest.main()
