"""Preservation objectives, measured support, and incumbent-language checks."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.language_preservation.analysis import analyze


class LanguagePreservationTest(unittest.TestCase):
    def panel(self, gaps=(.2, .8)):
        rows = []
        for coverage in (2, 5):
            for language, gap in zip(("x", "y"), gaps):
                for tokens in np.geomspace(1e8, 1e10, 8):
                    rows.append(dict(ladder="A", E=2, K=coverage, language=language,
                        N_act=1e8, D_total=tokens,
                        adjusted_loss=5-np.log(tokens/1e8)+(gap if coverage == 5 else 0)))
        return pd.DataFrame(rows)

    def analyze(self, rows):
        geometry = pd.DataFrame([dict(ladder="A", E=2, N_act=1e8, N_total=2e8)])
        return analyze(rows, geometry, languages=("x", "y"), base_k=2,
                       target_k=5, cap=3e8, baseline_f=.6, penalty=1e-6)

    def test_mean_and_all_language_objectives_differ(self):
        result = self.analyze(self.panel())
        summary = result["summary"]
        self.assertAlmostEqual(summary.loc["mean", "compute_multiplier"], np.exp(.5), places=7)
        self.assertAlmostEqual(summary.loc["all", "compute_multiplier"], np.exp(.8), places=7)
        self.assertEqual(summary.loc["mean", "protected_languages"], 1)
        self.assertEqual(summary.loc["all", "protected_languages"], 2)

    def test_unreached_is_not_extrapolated(self):
        result = self.analyze(self.panel((100., 100.)))
        self.assertTrue(result["policies"].empty)
        self.assertTrue(result["language_deltas"].empty)
        self.assertTrue(result["candidates"].status.eq("unreached").all())

    def test_missing_language_and_duplicate_observations_rejected(self):
        rows = self.panel()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.analyze(rows[~(rows.K.eq(5) & rows.language.eq("y"))])
        with self.assertRaises(ValueError):
            self.analyze(pd.concat([rows, rows.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
