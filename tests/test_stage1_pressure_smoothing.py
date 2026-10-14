"""Scientific contracts for curve smoothing, censoring, and policy regret."""

import unittest
import numpy as np
from scipy.interpolate import PPoly

from result_analysis.learning_curves import (
    LearningCurve, allocation_metrics, invert_monotone, first_feasible_piecewise,
)


class SmoothingTests(unittest.TestCase):
    def test_preserves_affine_log_curve_and_inverts_known_target(self):
        x = np.geomspace(1, 32, 6)
        curve = LearningCurve.fit(x, 3-.2*np.log(x), .01)
        grid = np.geomspace(1, 32, 101)
        np.testing.assert_allclose(curve(grid), 3-.2*np.log(grid), atol=1e-12)
        root, status = invert_monotone(curve, 3-.2*np.log(7), 1, 32)
        self.assertEqual(status, "bracketed")
        self.assertAlmostEqual(root, 7)
        with self.assertRaisesRegex(ValueError, "outside observed"):
            curve(33)

    def test_censoring_is_not_an_exact_compute_target(self):
        function = lambda x: 2-np.log(x)
        self.assertEqual(invert_monotone(function, 3, 1, 10), (1., "left_censored"))
        self.assertEqual(invert_monotone(function, -2, 1, 10), (None, "unreached"))

    def test_nonmonotone_inversion_retains_early_feasible_interval(self):
        # Values 1,-1,1 at 0,1,2: endpoint-only inversion would miss feasibility.
        polynomial = PPoly(np.array([[[-2.], [2.]], [[1.], [-1.]]]), [0., 1., 2.])
        point, status = first_feasible_piecewise(polynomial)
        self.assertEqual(status, "bracketed")
        self.assertAlmostEqual(point, .5)

    def test_individual_protection_requires_joint_feasibility(self):
        # Opposite constraints leave only x=1 feasible for both components.
        polynomial = PPoly(np.array([[[-1., 1.]], [[1., -1.]]]), [0., 2.])
        point, status = first_feasible_piecewise(polynomial)
        self.assertEqual(status, "bracketed")
        self.assertAlmostEqual(point, 1.)

    def test_regret_ignores_common_prediction_bias_but_detects_wrong_decision(self):
        observed = np.array([[1., 1.03], [1.04, 1.]])
        good = allocation_metrics(observed, observed+10)
        np.testing.assert_allclose(good["regret"], 0)
        self.assertAlmostEqual(good["pairwise_margin_rmse"], 0)
        wrong = allocation_metrics(observed, observed[::-1])
        np.testing.assert_allclose(wrong["regret"], [.04, .03])
        self.assertEqual(wrong["resolved_pairwise_accuracy"], 0)

    def test_mean_and_individual_protection_can_choose_different_recipes(self):
        from result_analysis.language_preservation.sensitivity import restore

        class Recipe:
            fmin, fmax, n, stored, e = 1., 30., 100_000_000, 200_000_000, 4

            def __init__(self, name, intercepts):
                self.name, self.intercepts = name, np.asarray(intercepts)

            def values(self, compute, cohort, method):
                return self.intercepts-np.log(compute)

        recipes = [Recipe("A", [1., 3.]), Recipe("B", [2.1, 2.1])]
        mean = restore(recipes, np.zeros(2), ["x", "y"], "smooth", "mean")
        individual = restore(recipes, np.zeros(2), ["x", "y"], "smooth", "all")
        self.assertEqual(mean["config"], "A")
        self.assertAlmostEqual(mean["F"], np.exp(2))
        self.assertAlmostEqual(mean["protected_fraction"], .5)
        self.assertEqual(individual["config"], "B")
        self.assertAlmostEqual(individual["F"], np.exp(2.1))
        self.assertAlmostEqual(individual["protected_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
