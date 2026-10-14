"""Only the parity check's scientific identity and resource-cap contracts."""
import copy
from pathlib import Path
import unittest

from multilingual_megatron.experiments.scaling.stage1_l26_e32.parity import budget_hours
from multilingual_megatron.experiments.scaling.stage1_l26_e32.parity import loss_comparison
from multilingual_megatron.experiments.scaling.stage1_l26_e32.parity import mode_spec


class ParityTest(unittest.TestCase):
    def test_approved_resource_cap(self):
        cfg = {"nodes": 2, "gpus_per_node": 4, "time_limit": "00:29:00", "gpu_hours_cap": 4}
        self.assertAlmostEqual(budget_hours(cfg), 8 * 29 / 60)
        with self.assertRaises(ValueError):
            budget_hours({**cfg, "consumed_gpu_hours": 0.38})
        self.assertAlmostEqual(budget_hours({**cfg, "time_limit": "00:26:00", "consumed_gpu_hours": 0.38}), 8*26/60)
        for change in ({"time_limit": "00:31:00"}, {"gpu_hours_cap": 5},
                       {"nodes": 4}, {"time_limit": "00:00:00"},
                       {"time_limit": "00:00:00", "consumed_gpu_hours": 0.38},
                       {"consumed_gpu_hours": float("nan")}):
            with self.assertRaises(ValueError):
                budget_hours({**cfg, **change})

    def test_only_topology_changes_and_true_training_scope_is_preserved(self):
        original = {"trained_languages": ["eng_Latn", "nno_Latn", "deu_Latn"],
            "required_languages": ["eng_Latn", "nno_Latn", "deu_Latn"],
            "sequence_length": 2048, "eval_micro_batch_size": 8,
            "model_recipe": {"expected_model": {"pipeline_model_parallel_size": 4, "num_moe_experts": 32},
                "overrides": ["model.pipeline_model_parallel_size=4", "model.pipeline_model_parallel_layout=old",
                              "model.expert_model_parallel_size=2"]}}
        before = copy.deepcopy(original)
        pp4 = mode_spec(original, Path("pp4"), "pp4", ["eng_Latn", "nno_Latn"])
        pp1 = mode_spec(original, Path("pp1"), "pp1", ["eng_Latn", "nno_Latn"])
        self.assertEqual(original, before)
        self.assertEqual(pp4["model_recipe"], original["model_recipe"])
        self.assertEqual(pp1["model_recipe"]["overrides"], ["model.pipeline_model_parallel_size=1",
            "model.pipeline_model_parallel_layout=null", "model.expert_model_parallel_size=2"])
        for field in ("trained_languages", "required_languages", "sequence_length", "eval_micro_batch_size"):
            self.assertEqual(pp1[field], original[field])
        with self.assertRaisesRegex(ValueError, "approved"):
            mode_spec(original, Path("bad"), "pp1", ["eng_Latn"])

    def test_matching_losses_do_not_excuse_changed_tokens_or_invalid_sums(self):
        loss = {"language": "eng_Latn", "token_count": 10, "target_token_ids_sha256": "tokens",
            "unigram_counts_sha256": "unigram", "unigram_log_probability_sum": -70,
            "vocab_size": 64000, "special_token_ids": [0, 1, 2, 3], "unseen_target_token_count": 0,
            "raw_nll": 3, "nll_sum": 30, "atlas_tao_unigram_adjusted_loss": -4,
            "backend": {"backend_token_count": 10, "checkpoint_step": 269634, "sequence_length": 2048,
                "eval_micro_batch_size": 8, "validation_prefixes": ["fixed"],
                "stream_tokens_including_special": 12, "loss_statistic": "sum_and_count"}}
        self.assertTrue(loss_comparison(loss, loss, 1e-4)["passed"])
        different_loss = {**loss, "raw_nll": 3.001, "nll_sum": 30.01, "atlas_tao_unigram_adjusted_loss": -3.999}
        self.assertFalse(loss_comparison(loss, different_loss, 1e-4)["passed"])
        for change in ({"target_token_ids_sha256": "changed"}, {"nll_sum": 0},
                       {"raw_nll": float("nan")}, {"atlas_tao_unigram_adjusted_loss": 0},
                       {"backend": {**loss["backend"], "backend_token_count": 11}},
                       {"backend": {**loss["backend"], "checkpoint_step": 1}}):
            with self.assertRaises(ValueError):
                loss_comparison(loss, {**loss, **change}, 1e-4)


if __name__ == "__main__":
    unittest.main()
