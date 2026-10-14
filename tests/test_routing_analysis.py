"""Routing count normalization and fixed-cohort validation."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.routing_analysis.analysis import analyze, from_counts, summaries_from_counts


class RoutingAnalysisTest(unittest.TestCase):
    def test_parquet_count_summary_recomputes_metrics_and_matches_loss_targets(self):
        counts, losses, dense = [], [], []
        for language, values in (("x", [10, 0]), ("y", [0, 10])):
            counts.append(dict(id="moe", ladder="A", experts=2, languages=2, seed=1,
                tokens_seen=100, language=language, layer_index=0, expert_token_counts=values))
            losses.append(dict(id="moe", language=language, token_count=10, nll_sum=20.,
                               target_token_ids_sha256=language))
            dense.append(dict(E=1, readout_id="dense", ladder="A", K=2, model_seed=1,
                D_total=100, language=language, token_count=10, nll_sum=30.,
                target_token_ids_sha256=language))
        frames = [pd.DataFrame(v) for v in (counts, losses, dense)]
        checkpoint, languages = summaries_from_counts(*frames, cohorts={"pair": ["x", "y"]})
        self.assertAlmostEqual(checkpoint.mi_nats.item(), np.log(2))
        self.assertAlmostEqual(checkpoint.effective_expert_fraction.item(), .5)
        np.testing.assert_allclose(languages.mean_pair_js_nats, np.log(2))
        np.testing.assert_array_equal(languages.moe_gain_nats, [1., 1.])
        frames[2].loc[0, "target_token_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "targets differ"):
            summaries_from_counts(*frames, cohorts={"pair": ["x", "y"]})

    def test_uniform_and_language_specific_routing(self):
        uniform = from_counts(np.ones((2, 3, 4)))["summary"]
        self.assertAlmostEqual(uniform["mi_nats"], 0.)
        self.assertAlmostEqual(uniform["effective_expert_fraction"], 1.)
        specific = from_counts(np.eye(2)[None, :, :])["summary"]
        self.assertAlmostEqual(specific["mi_nats"], np.log(2))
        self.assertAlmostEqual(specific["effective_expert_fraction"], .5)

    def test_bad_counts_rejected(self):
        for counts in (np.zeros((1, 2, 2)), -np.ones((1, 2, 2)), np.ones((2, 2)),
                       np.full((1, 2, 2), np.nan)):
            with self.assertRaises(ValueError):
                from_counts(counts)

    def panel(self):
        summaries, languages = [], []
        for k in (3, 5):
            row = dict(id=f"k{k}", ladder="A", experts=4, languages=k, seed=1,
                tokens_seen=100, cohort="fixed", cohort_size=3, mi_nats=k*.1,
                effective_expert_fraction=.5, pooled_expert_fraction=.8)
            summaries.append(row)
            for i, language in enumerate(("x", "y", "z")):
                languages.append(dict(row, language=language, mean_pair_js_nats=.1*i,
                                      moe_gain_nats=.2*i))
        return pd.DataFrame(summaries), pd.DataFrame(languages)

    def test_contrasts_and_associations(self):
        summary, languages = self.panel()
        tables = analyze(summary, languages, cohorts=("fixed",))
        self.assertAlmostEqual(tables["coverage_contrasts"].query("metric == 'mi_nats'").delta.item(), .2)
        associations = tables["loss_associations"]
        self.assertTrue(associations.query("predictor == 'effective_expert_fraction'").spearman_rho.isna().all())
        np.testing.assert_allclose(associations.query("predictor == 'mean_pair_js_nats'").spearman_rho, 1.)
        with self.assertRaises(ValueError):
            analyze(summary, languages.iloc[1:], cohorts=("fixed",))
        with self.assertRaises(ValueError):
            analyze(summary, pd.concat([languages, languages.iloc[:1]]), cohorts=("fixed",))
        with self.assertRaises(ValueError):
            analyze(summary, languages, cohorts=("trained",))


if __name__ == '__main__':
    unittest.main()
