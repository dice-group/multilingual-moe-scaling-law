"""Language weighting, metric orientation, and downstream evidence validation."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.checks import Checks
from result_analysis.downstream_evaluations import accounting, analysis, contrasts
from result_analysis.downstream_evaluations.reproduce import compare


class DownstreamEvaluationsTest(unittest.TestCase):
    def test_corpus_languages_have_equal_weight(self):
        rows = []
        for k in (12, 200):
            for e in (1, 32):
                for variety, language, baseline, gain in (
                        ("en1", "eng_Latn", .1, .1), ("en2", "eng_Latn", .3, .1),
                        ("x", "xxx_Latn", .8, .2)):
                    rows.append(dict(model_id=f"{e}/{k}", ladder="B", E=e, K=k,
                        seed=1, benchmark="test", language=variety, corpus_language=language,
                        panel="fixture", primary=baseline+(gain if e == 32 else 0), validation_nll=2.))
        tables = contrasts.aggregate(pd.DataFrame(rows), ["eng_Latn", "xxx_Latn"])
        score = tables["cohort_metrics"].query("E == 1 and K == 12 and cohort == 'common12'")
        self.assertAlmostEqual(score.primary.item(), .5)
        gains = tables["expert_gains"].query("cohort == 'common12' and K == 12")
        self.assertAlmostEqual(gains.gain.item(), 15.)
        self.assertEqual(contrasts.orientation("flores_mono", "primary"), -1)
        self.assertEqual(contrasts.scale("flores_mono", "external_nll"), 1)

    def test_custom_scores_and_missing_pairs(self):
        frame = pd.DataFrame([dict(model_id=f"{k}/{e}", ladder="A", E=e, K=k,
            seed=7, benchmark="custom", language=l, corpus_language=l, panel="test",
            score=(e/8)*(k/5)+(1 if l == "y" else 0))
            for k in (2, 5) for e in (2, 8) for l in ("x", "y")])
        settings = dict(languages=("x", "y"), coverages=(2, 5), coverage_pair=(2, 5),
            expert_pairs=((2, 8),), metric_directions={("custom", "score"): 1})
        result = analysis.analyze(frame, **settings)
        self.assertAlmostEqual(result["coverage_interactions"].interaction.iloc[0], .45)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            analysis.analyze(frame.iloc[1:], **settings)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            analysis.analyze(frame.assign(score=np.inf), **settings)

    def test_table_comparison_rejects_drift(self):
        frame = pd.DataFrame(dict(cell=["a", "b"], value=[1., np.nan]))
        compare(frame, frame.copy(), ["cell"], Checks(), "fixture")
        for bad in (frame.iloc[:1], frame.fillna(0), frame.assign(value=[np.inf, np.nan])):
            with self.assertRaises(ValueError):
                compare(bad, frame, ["cell"], Checks(), "fixture")

    def test_unmapped_answer_language_is_not_silently_dropped(self):
        languages = pd.DataFrame(dict(benchmark=["task"], language=["x"], corpus_language=["x"]))
        flips = pd.DataFrame(dict(benchmark=["task"], language=["missing"]))
        with self.assertRaisesRegex(ValueError, "mapped"):
            accounting.aggregate(languages, flips, Checks())


if __name__ == "__main__":
    unittest.main()
