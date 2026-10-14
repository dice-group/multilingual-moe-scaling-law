"""Scientific estimators and immutable reporting contracts, CPU only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_routing_repair.analysis import bootstrap_probabilities
from multilingual_megatron.experiments.scaling.stage1_routing_repair.campaign import identity, sha, write
from multilingual_megatron.experiments.scaling.stage1_routing_repair.reporting import (
    InputFiles, METRICS, _pair_key, matched_gain, real_block_layout,
    report, require_stream_match, routing_metrics, spearman, summarize_layers,
    verify_report,
)


def language(nll=20.):
    return {"language": "eng_Latn", "token_count": 10, "nll_sum": nll,
            "atlas_tao_unigram_adjusted_loss": nll / 10 - 4,
            "target_token_ids_sha256": "targets", "unigram_counts_sha256": "unigram",
            "unigram_log_probability_sum": -40., "special_token_ids": [0, 1], "vocab_size": 8,
            "input_sha256": f"includes-nll-{nll}",
            "backend": {"sequence_length": 4, "stream_tokens_including_special": 12,
                        "validation_prefixes": ["sealed-language"], "loss_statistic": "masked NLL"}}


class RoutingEstimatorsTest(unittest.TestCase):
    def test_uniform_and_language_exclusive_known_values(self):
        uniform = routing_metrics(np.full((3, 4), .25))
        self.assertAlmostEqual(uniform["mi_nats"], 0)
        self.assertAlmostEqual(uniform["effective_expert_fraction"], 1)
        self.assertAlmostEqual(uniform["pooled_expert_fraction"], 1)
        exclusive = routing_metrics(np.eye(4))
        self.assertAlmostEqual(exclusive["mi_nats"], np.log(4))
        self.assertAlmostEqual(exclusive["effective_expert_fraction"], .25)
        self.assertAlmostEqual(exclusive["pooled_expert_fraction"], 1)

    def test_language_weights_and_arithmetic_effective_experts(self):
        counts = np.array([[9., 0], [50., 50.]])
        metrics = routing_metrics(counts / counts.sum(-1, keepdims=True))
        self.assertAlmostEqual(metrics["effective_expert_fraction"], .75)
        self.assertNotAlmostEqual(metrics["effective_expert_fraction"], np.sqrt(2) / 2)
        counts[0] *= 1000
        other = routing_metrics(counts / counts.sum(-1, keepdims=True))
        for key in METRICS:
            self.assertEqual(metrics[key], other[key])

    def test_expert_and_language_permutation_invariance(self):
        p = np.array([[.1, .2, .7], [.5, .4, .1]])
        left, right = routing_metrics(p), routing_metrics(p[::-1, ::-1])
        for key in METRICS:
            self.assertAlmostEqual(left[key], right[key])

    def test_single_language_mi_is_undefined(self):
        result = routing_metrics(np.array([[.2, .8]]))
        self.assertTrue(np.isnan(result["mi_nats"]))
        self.assertTrue(np.isfinite(result["effective_expert_fraction"]))

    def test_invalid_probabilities_rejected(self):
        for p in ([[.5, .6]], [[-.1, 1.1]], [[np.nan, .2]]):
            with self.assertRaises(ValueError):
                routing_metrics(p)

    def test_bootstrap_median_preserves_correlated_layers(self):
        layer_metrics = [{key: 2. for key in METRICS} for _ in range(3)]
        sampled = [{key: np.array(draws) for key in METRICS}
                   for draws in ([0., 0., 10., 10.], [10., 10., 0., 0.], [2., 2., 2., 2.])]
        summary, draws = summarize_layers(layer_metrics, sampled)
        for key in METRICS:
            np.testing.assert_array_equal(draws[key], [2., 2., 2., 2.])
            self.assertEqual(summary[key + "_ci_low"], 2)
            self.assertEqual(summary[key + "_ci_high"], 2)

    def test_bootstrap_pairing_and_empty_padding_blocks(self):
        a = {"blocks": {"1": [2, 6], "0": [7, 1]}}
        b = {"blocks": {"0": [7, 1], "1": [2, 6], "2": [0, 0]}}
        self.assertEqual(real_block_layout(a), real_block_layout(b))
        p = bootstrap_probabilities(a, "eng_Latn", 1000, 20260907)
        q = bootstrap_probabilities(b, "eng_Latn", 1000, 20260907)
        np.testing.assert_array_equal(p, q)
        shifted = {"blocks": {"1": [7, 1], "2": [2, 6]}}
        self.assertNotEqual(real_block_layout(a), real_block_layout(shifted))

    def test_matched_gain_ignores_loss_dependent_input_hash(self):
        sparse, dense = language(20.), language(25.)
        self.assertNotEqual(sparse["input_sha256"], dense["input_sha256"])
        self.assertEqual(matched_gain(sparse, dense), .5)

    def test_wrong_target_scoring_or_layout_rejected(self):
        left = language()
        for key, value in (("target_token_ids_sha256", "wrong"), ("token_count", 9),
                           ("unigram_counts_sha256", "other"), ("special_token_ids", [0]), ("vocab_size", 9)):
            right = copy.deepcopy(left)
            right[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_stream_match(left, right)
        right = copy.deepcopy(left)
        right["backend"]["sequence_length"] = 8
        with self.assertRaises(ValueError):
            require_stream_match(left, right)

    def test_dense_key_separates_seed_horizon_and_shuffle(self):
        base = {"ladder": "B", "languages": 200, "seed": 1234, "tokens_seen": 16000000000, "dataset_shuffle_seed": 1234}
        for key in ("seed", "tokens_seen", "dataset_shuffle_seed"):
            other = dict(base)
            other[key] += 1
            self.assertNotEqual(_pair_key(base), _pair_key(other))

    def test_raw_and_adjusted_gain_must_agree(self):
        dense = language(25.)
        dense["atlas_tao_unigram_adjusted_loss"] += 1
        with self.assertRaises(ValueError):
            matched_gain(language(), dense)

    def test_spearman_ties_negative_and_constant(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [4, 4, 5, 6])[0], 1)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1])[0], -1)
        self.assertEqual(spearman([1, 1, 1], [1, 2, 3]), (None, "constant_vector"))
        with self.assertRaises(ValueError):
            spearman([1, np.nan, 3], [1, 2, 3])


class ReportIntegrityTest(unittest.TestCase):
    def test_source_hash_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text('{"value": 1}')
            inputs = InputFiles()
            inputs.json(identity(path))
            path.write_text('{"value": 2}')
            with self.assertRaisesRegex(ValueError, "sealed artifact changed"):
                inputs.check()

    def test_existing_or_sealed_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = {"root": str(root / "campaign")}
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "keep"
            sentinel.write_text("original")
            with self.assertRaises(FileExistsError):
                report(campaign, existing)
            with self.assertRaisesRegex(ValueError, "separate"):
                report(campaign, root / "campaign/analysis/new")
            self.assertEqual(sentinel.read_text(), "original")

    def test_verify_and_replay_are_read_only_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "campaign/campaign.json", {"test": "campaign"})
            campaign = {"root": str(root / "campaign")}
            output = root / "report"
            tables = {"checkpoint_summary": pd.DataFrame({"id": ["fixture"], "mi_nats": [.1]})}
            audit = {"checkpoints": 1}
            inputs = InputFiles()
            inputs.add(root / "campaign/campaign.json")
            with patch("multilingual_megatron.experiments.scaling.stage1_routing_repair.reporting.compute_report", return_value=(tables, audit, inputs)), \
                 patch("multilingual_megatron.experiments.scaling.stage1_routing_repair.reporting._validate_tables"), \
                 patch("multilingual_megatron.experiments.scaling.stage1_routing_repair.report_figures.render_figures"), \
                 patch("multilingual_megatron.experiments.scaling.stage1_routing_repair.report_figures.write_findings"):
                result = report(campaign, output)
                self.assertEqual(result["status"], "verified")
                before = {str(p): (sha(p), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
                self.assertTrue(verify_report(campaign, output, replay=True)["numerical_replay"])
                after = {str(p): (sha(p), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
                self.assertEqual(before, after)
                (output / "analysis_spec.json").write_text("{}")
                with self.assertRaises(ValueError):
                    verify_report(campaign, output)


if __name__ == "__main__":
    unittest.main()
