from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from multilingual_megatron.experiments.scaling.stage1_fit.fit import (
    EXPLORATORY_M1_ABLATION_MODELS,
    PRIMARY_MODELS,
    outer_from_shared,
    outer_parameter_names,
    predictor_terms,
    predictor_terms_and_jacobian,
    shared_from_outer,
)
from multilingual_megatron.experiments.scaling.stage1_fit.interaction_ablation import (
    ablation_starts,
    parser,
)


M1_COORDINATES = np.asarray([
    0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.04,
], dtype=np.float64)
M2_COORDINATES = np.asarray([
    0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.02, -0.04, 0.03,
], dtype=np.float64)


def _result(model: str, coordinates: np.ndarray) -> dict[str, object]:
    attempts = []
    for index in range(12):
        values = coordinates.copy()
        values[index % values.size] += (index + 1) * 1e-4
        attempts.append({"success": True, "objective": 1.0 + index * 1e-8,
            "start_id": f"grid:{index:06d}", "coordinates": values.tolist()})
    return {"manifest_path": f"{model}.json", "selected": {"coordinates": coordinates.tolist()},
        "attempts": attempts}


class Stage1FitInteractionAblationTest(unittest.TestCase):
    def test_models_are_exploratory_and_have_nine_parameters(self) -> None:
        self.assertEqual(PRIMARY_MODELS, ("M0", "M1", "M2"))
        self.assertEqual(EXPLORATORY_M1_ABLATION_MODELS, ("M1-PhiE", "M1-PsiE"))
        for model in EXPLORATORY_M1_ABLATION_MODELS:
            self.assertEqual(len(outer_parameter_names(model)), 9)

    def test_models_exactly_nest_between_m1_and_m2(self) -> None:
        predictors = (np.asarray([1e8, 2e8, 4e8, 8e8]), np.asarray([1e8, 5e8, 2e9, 8e9]),
            np.asarray([1.0, 4.0, 16.0, 32.0]), np.asarray([1.0, 12.0, 50.0, 200.0]))
        m1 = shared_from_outer("M1", M1_COORDINATES)
        for model in EXPLORATORY_M1_ABLATION_MODELS:
            intermediate = shared_from_outer(model, outer_from_shared(model, m1))
            for expected, actual in zip(predictor_terms("M1", m1, *predictors),
                    predictor_terms(model, intermediate, *predictors)):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
            m2 = shared_from_outer("M2", outer_from_shared("M2", intermediate))
            for expected, actual in zip(predictor_terms(model, intermediate, *predictors),
                    predictor_terms("M2", m2, *predictors)):
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_jacobians_match_finite_difference(self) -> None:
        predictors = (np.asarray([1e8, 3e8, 7e8]), np.asarray([2e8, 1e9, 5e9]),
            np.asarray([1.0, 16.0, 32.0]), np.asarray([1.0, 50.0, 200.0]))
        cases = {
            "M1-PhiE": np.asarray([0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.02, -0.04]),
            "M1-PsiE": np.asarray([0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.04, 0.03]),
        }
        epsilon = 1e-6
        for model, outer in cases.items():
            _, _, capacity_jac, data_jac = predictor_terms_and_jacobian(model, outer, *predictors)
            for column in range(outer.size):
                plus, minus = outer.copy(), outer.copy(); plus[column] += epsilon; minus[column] -= epsilon
                cap_plus, data_plus = predictor_terms(model, shared_from_outer(model, plus), *predictors)
                cap_minus, data_minus = predictor_terms(model, shared_from_outer(model, minus), *predictors)
                np.testing.assert_allclose(capacity_jac[:, column], (cap_plus-cap_minus)/(2*epsilon), rtol=4e-8, atol=4e-10)
                np.testing.assert_allclose(data_jac[:, column], (data_plus-data_minus)/(2*epsilon), rtol=4e-8, atol=4e-10)

    def test_start_sets_span_m1_and_m2_basins(self) -> None:
        manifests = {"M1.json": {"fit": {"model": "M1"}}, "M2.json": {"fit": {"model": "M2"}}}
        with patch("multilingual_megatron.experiments.scaling.stage1_fit.interaction_ablation.load_manifest",
                side_effect=lambda path, **_: manifests[str(path)]):
            for model in EXPLORATORY_M1_ABLATION_MODELS:
                first = ablation_starts(model, _result("M1", M1_COORDINATES), _result("M2", M2_COORDINATES))
                second = ablation_starts(model, _result("M1", M1_COORDINATES), _result("M2", M2_COORDINATES))
                self.assertEqual(len(first), 24)
                self.assertEqual([item.start_id for item in first], [item.start_id for item in second])
                self.assertTrue(all(item.values.shape == (9,) for item in first))
                for left, right in zip(first, second):
                    np.testing.assert_array_equal(left.values, right.values)

    def test_cli_supports_dry_run(self) -> None:
        parsed = parser().parse_args(["plan", "--profile", "p", "--input-receipt", "i",
            "--holdouts", "h", "--candidate-report", "c", "--dry-run"])
        self.assertEqual(parsed.command, "plan"); self.assertTrue(parsed.dry_run)


if __name__ == "__main__":
    unittest.main()
