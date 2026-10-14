"""Current-paper module contracts without dependence on HPC result files."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from result_analysis.checks import Checks
from result_analysis.scaling_law_fit.analysis import manuscript_scores, endpoint_scores, grid_scores
from result_analysis.scaling_law_fit.composition import arm_contrasts
from result_analysis.compute_allocation.observed import policy_scores
from result_analysis.compute_allocation.fitted import optimum
from result_analysis.downstream_evaluations import contrasts
from result_analysis.downstream_evaluations.analysis import compare


class CurrentEvidence(unittest.TestCase):
    def test_profile_preserves_current_and_historical_roles(self):
        root = Path(__file__).resolve().parents[1]
        profile = json.loads((root / "configs/result_analysis/stage1_paper_v2.json").read_text())
        self.assertEqual(profile["analysis"]["main_law"], "RN14-stored-reuse")
        self.assertEqual(profile["analysis"]["historical_inner_selected"], "RN13-psi-tied")
        self.assertEqual(profile["analysis"]["allocation_predictor"], "M6-ND")
        self.assertEqual(len(profile["analysis"]["main_table"]), 8)

    def test_main_table_uses_source_model_not_display_label(self):
        table = manuscript_scores("% Source model: RN14-stored-reuse\nAny label & 14 & \\textbf{0.02912} & 0.02587 & 0.04496 & 0.03332 & 0.17280 \\\\\n")
        self.assertAlmostEqual(table.loc["RN14-stored-reuse", "L26_all_grid"], .1728)
        with self.assertRaisesRegex(ValueError, "missing"):
            manuscript_scores("Any label & 0.01 & 0.02 & 0.03 & 0.04 & 0.05")


class CurrentScience(unittest.TestCase):
    def test_fixed_law_allocation_is_bounded_and_minimal(self):
        p = pd.Series(dict(base=1., C=.8, B=.4, alpha=.3, beta=.4, zeta=.1))
        geometry = pd.DataFrame(dict(N_act=[1e8, 2e8, 4e8, 6e8], N_total=[2e8, 4e8, 8e8, 12e8]))
        budgets = np.geomspace(1e18, 1e21, 8)
        answer = optimum(p, geometry, 5e8, budgets)
        np.testing.assert_allclose(6*answer["N_act"]*answer["D_total"], budgets, rtol=1e-14)
        self.assertTrue(np.all(answer["N_total"] <= 5e8))
        self.assertTrue(np.all(answer["D_total"] >= .5e9))
        for f, value in zip(budgets, answer["loss"]):
            n = np.geomspace(1., min(2.5, f/(6*.5e9)/1e8), 1001)
            values = p.base+p.C*n**(-p.alpha)+p.B*(f/.6e18)**(-p.beta)*n**(p.beta+p.zeta)
            self.assertLessEqual(value, values.min()+1e-12)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            optimum(p, geometry, 1e8, budgets)

    def test_endpoint_language_balance_and_metric_orientation(self):
        rows = []
        for k in (12, 200):
            for e in (1, 32):
                for variety, language, baseline, gain in (("en1", "eng_Latn", .1, .1),
                                                        ("en2", "eng_Latn", .3, .1),
                                                        ("x", "xxx_Latn", .8, .2)):
                    rows.append(dict(model_id=f"{e}/{k}", ladder="B", E=e, K=k, seed=1, benchmark="test",
                        language=variety, corpus_language=language, panel="fixture", primary=baseline+(gain if e == 32 else 0),
                        validation_nll=2.))
        tables = contrasts.aggregate(pd.DataFrame(rows), ["eng_Latn", "xxx_Latn"])
        metrics = tables["cohort_metrics"].query("E == 1 and K == 12 and cohort == 'common12'")
        self.assertAlmostEqual(metrics.primary.item(), .5)  # not (.1+.3+.8)/3
        gains = tables["expert_gains"].query("cohort == 'common12' and K == 12")
        self.assertAlmostEqual(gains.gain.item(), 15.)
        self.assertEqual(contrasts.orientation("flores_mono", "primary"), -1)
        self.assertEqual(contrasts.orientation("test", "primary"), 1)
        self.assertEqual(contrasts.scale("flores_mono", "external_nll"), 1)
        broken = pd.DataFrame(rows)
        broken.loc[(broken.E == 32) & (broken.language == "x"), "primary"] = np.nan
        with self.assertRaisesRegex(ValueError, "unpaired"):
            contrasts.aggregate(broken, ["eng_Latn", "xxx_Latn"])

    def test_report_comparison_rejects_population_and_missingness_drift(self):
        actual = pd.DataFrame(dict(cell=["a", "b"], value=[1., np.nan]))
        compare(actual, actual.copy(), ["cell"], Checks(), "fixture")
        with self.assertRaisesRegex(ValueError, "membership"):
            compare(actual, actual.iloc[:1], ["cell"], Checks(), "fixture")
        with self.assertRaisesRegex(ValueError, "missingness"):
            compare(actual, actual.fillna(0), ["cell"], Checks(), "fixture")

    def test_grid_rmse_weights_trajectories_not_rows(self):
        frame = pd.DataFrame(dict(model=["law"]*4, fold=["E32"]*4,
            trajectory_id=["a", "a", "a", "b"], readout_id=["a1", "a2", "a3", "b1"],
            language=["x"]*4, adjusted_loss=[0.]*4, prediction=[0., 0., 0., 2.]))
        self.assertAlmostEqual(grid_scores(frame).loc["law", "E32"], np.sqrt(2))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            grid_scores(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        frame.loc[0, "prediction"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            grid_scores(frame)

    def test_partial_endpoint_is_not_a_complete_200_language_result(self):
        frame = pd.DataFrame(dict(model=["law"]*200, language=[f"l{i}" for i in range(200)],
                                  adjusted_loss=[0.]*200, prediction=[.2]*200))
        self.assertAlmostEqual(endpoint_scores(frame).loc["law"], .2)
        self.assertTrue(endpoint_scores(frame.iloc[:-1]).empty)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            endpoint_scores(pd.concat([frame, frame.iloc[:1]]))

    def test_composition_is_paired_and_not_absolute_loss(self):
        rows = pd.DataFrame(dict(arm=["compact", "diverse"], seed=[1, 1], E=[1, 1],
            horizon=[10, 10], language=["x", "x"], prediction=[2., 2.2], adjusted_loss=[3., 3.3]))
        contrast = arm_contrasts(rows, ["prediction", "adjusted_loss"])
        self.assertAlmostEqual(contrast.prediction.iloc[0], .2)
        self.assertAlmostEqual(contrast.adjusted_loss.iloc[0], .3)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            arm_contrasts(rows.iloc[:1], ["prediction", "adjusted_loss"])

    def test_allocation_selects_on_predictions_and_scores_observations(self):
        pred = pd.DataFrame(dict(model=["law", "law"], config=["a", "b"], language=["x", "x"],
            FLOPs=[1., 1.], N_total=[1e8, 2e8], prediction=[3., 2.]))
        observed = pd.DataFrame(dict(reference=["smooth", "smooth"], config=["a", "b"],
            language=["x", "x"], FLOPs=[1., 1.], observed=[1., 1.1]))
        decisions, metrics = policy_scores(pred, observed, {"test": ["x"]})
        self.assertTrue(decisions.chosen.eq("b").all())
        np.testing.assert_allclose(metrics.max_regret, .1)
        self.assertTrue(metrics.correct_choices.eq(0).all())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            policy_scores(pred, observed, {"test": ["x", "missing"]})


if __name__ == "__main__":
    unittest.main()
