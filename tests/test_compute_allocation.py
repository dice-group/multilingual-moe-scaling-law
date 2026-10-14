"""Support, matched decision sets, and fixed-law allocation checks."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.compute_allocation.analysis import analyze
from result_analysis.compute_allocation.fitted import optimum
from result_analysis.compute_allocation.observed import policy_scores
from result_analysis.learning_curves import LearningCurve


class ComputeAllocationTest(unittest.TestCase):
    def test_supported_curve_and_no_switch(self):
        tokens = np.geomspace(1e8, 1e10, 8)
        loss = 5 - np.log(tokens / 1e8)
        curve = LearningCurve.fit(tokens, loss, 1e-6)
        np.testing.assert_allclose(curve(tokens), loss, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "outside observed"):
            curve(2e10)
        rows = pd.DataFrame(dict(ladder="A", E=2, K=12, language="x",
                                 N_act=1e8, D_total=tokens, adjusted_loss=loss))
        geometry = pd.DataFrame([dict(ladder="A", E=2, N_act=1e8, N_total=2e8)])
        settings = dict(languages=("x",), coverages=(12,), cap=3e8,
                        penalty=1e-6, methods=("linear_log",), grid_points=9)
        result = analyze(rows, geometry, **settings)
        self.assertTrue(result["switches"].empty)
        self.assertEqual(len(result["loss_curves"]), 9)
        with self.assertRaisesRegex(ValueError, "geometry"):
            analyze(rows, geometry.assign(N_act=2e8), **settings)

    def test_fixed_law_optimum_respects_compute_storage_and_minimum(self):
        p = pd.Series(dict(base=1., C=.8, B=.4, alpha=.3, beta=.4, zeta=.1))
        geometry = pd.DataFrame(dict(N_act=[1e8, 2e8, 4e8, 6e8],
                                     N_total=[2e8, 4e8, 8e8, 12e8]))
        budgets = np.geomspace(1e18, 1e21, 8)
        answer = optimum(p, geometry, 5e8, budgets)
        np.testing.assert_allclose(6*answer["N_act"]*answer["D_total"], budgets)
        self.assertTrue(np.all(answer["N_total"] <= 5e8))
        self.assertTrue(np.all(answer["D_total"] >= .5e9))
        for budget, value in zip(budgets, answer["loss"]):
            n = np.geomspace(1., min(2.5, budget/(6*.5e9)/1e8), 1001)
            losses = p.base+p.C*n**(-p.alpha)+p.B*(budget/.6e18)**(-p.beta)*n**(p.beta+p.zeta)
            self.assertLessEqual(value, losses.min()+1e-12)

    def test_policy_regret_requires_same_candidates(self):
        predicted = pd.DataFrame(dict(model=["law"]*2, config=["A", "B"],
            language=["x"]*2, FLOPs=[1e18]*2, N_total=[2e8]*2, prediction=[2., 1.]))
        observed = pd.DataFrame(dict(reference=["measured"]*2, config=["A", "B"],
            language=["x"]*2, FLOPs=[1e18]*2, observed=[1., 1.2]))
        decisions, _ = policy_scores(predicted, observed, {"common": ["x"]}, caps=(3e8,))
        self.assertAlmostEqual(decisions.regret.iloc[0], .2)
        for prediction, truth in ((predicted.iloc[:1], observed), (predicted, observed.iloc[:1])):
            with self.assertRaisesRegex(ValueError, "candidate populations"):
                policy_scores(prediction, truth, {"common": ["x"]}, caps=(3e8,))


if __name__ == "__main__":
    unittest.main()
