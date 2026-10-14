from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class CooldownReadoutStageTest(unittest.TestCase):
    def test_segmented_training_readout_uses_segment_checkpoint_and_batch(self) -> None:
        from multilingual_megatron.pipeline.stages.cooldown_readout.cooldown_readout_stage import (
            CooldownReadoutConfig,
            _build_readout_rows,
            _readout_run_cfg,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_root = root / "runs"
            run_id = "scaling_switch_moe_scale_scaling_a_g1_a_e32_locked"
            train_dir = train_root / run_id
            seg0_dir = train_dir / "segments" / "00_gbs6"
            seg1_dir = train_dir / "segments" / "01_gbs8"
            (seg1_dir / "checkpoints" / "iter_0000003").mkdir(parents=True)
            (train_dir / ".training_complete").parent.mkdir(parents=True, exist_ok=True)
            (train_dir / ".training_complete").write_text("ok\n", encoding="utf-8")
            (train_dir / "training_run_manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "segmented_training": True,
                            "global_batch_size": 8,
                            "micro_batch_size": 4,
                            "seq_length": 10,
                            "train_iters": 4,
                            "final_checkpoint_step": 4,
                            "final_checkpoint_root": str(seg1_dir / "checkpoints"),
                            "segments": [
                                {
                                    "segment_index": 0,
                                    "target_start_tokens": 0,
                                    "target_end_tokens": 100,
                                    "actual_start_tokens": 0,
                                    "actual_end_tokens": 120,
                                    "global_batch_size": 6,
                                    "micro_batch_size": 3,
                                    "seq_length": 10,
                                    "tokens_per_step": 60,
                                    "cumulative_start_step": 0,
                                    "cumulative_end_step": 2,
                                    "output_dir": str(seg0_dir),
                                },
                                {
                                    "segment_index": 1,
                                    "target_start_tokens": 100,
                                    "target_end_tokens": 200,
                                    "actual_start_tokens": 120,
                                    "actual_end_tokens": 280,
                                    "global_batch_size": 8,
                                    "micro_batch_size": 4,
                                    "seq_length": 10,
                                    "tokens_per_step": 80,
                                    "cumulative_start_step": 2,
                                    "cumulative_end_step": 4,
                                    "output_dir": str(seg1_dir),
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            ctx = SimpleNamespace(
                train_root=train_root,
                logs_root=root / "logs",
                args=SimpleNamespace(dry_run=True),
            )
            cfg = CooldownReadoutConfig(
                enabled=True,
                output_dir=root / "cooldown",
                horizons_tokens_by_scale={"scaling_a": [150]},
                cooldown_fraction=0.2,
                cooldown_max_tokens=256_000_000,
                eval_iters=3,
                require_training_complete=True,
                require_checkpoint=True,
                run_forks=True,
                strict=True,
            )

            rows = _build_readout_rows(
                ctx=ctx,
                cfg=cfg,
                common={"seq_length": 10, "train": {"global_batch_size": 8, "micro_batch_size": 4}},
                runs=[
                    {
                        "id": run_id,
                        "recipe": "scaling_switch_moe_scale_scaling_a_pretrain_config",
                        "scaling": {
                            "scale_label": "scaling_a",
                            "family": "scaling_switch_moe",
                            "variant_label": "a_e32_locked",
                            "num_experts": 32,
                            "num_gpus": 1,
                        },
                    }
                ],
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["global_batch_size"], 8)
        self.assertEqual(row["micro_batch_size"], 4)
        self.assertEqual(row["segment_index"], 1)
        self.assertEqual(row["batch_policy"], "segmented")
        self.assertEqual(row["checkpoint_step"], 3)
        self.assertEqual(row["checkpoint_tokens"], 200)
        self.assertEqual(row["checkpoint_token_delta"], 50)
        self.assertIn("segments/01_gbs8/checkpoints/iter_0000003", row["checkpoint_path"])
        readout_cfg = _readout_run_cfg(
            base_run_cfg={
                "id": run_id,
                "train": {"batch_policy": {"enabled": True}},
                "overrides": ["validation.eval_iters=0", "model.tensor_model_parallel_size=1"],
            },
            row={**row, "eval_iters": 3},
        )
        self.assertEqual(readout_cfg["train"]["batch_policy"], {"enabled": False})
        self.assertNotIn("validation.eval_iters=0", readout_cfg["overrides"])
        self.assertEqual(readout_cfg["overrides"][-2:], ["validation.eval_interval=1", "validation.eval_iters=3"])

    def test_dry_run_writes_readout_plan_and_finetune_checkpoint_overrides(self) -> None:
        from multilingual_megatron.pipeline.context import PipelineContext
        from multilingual_megatron.pipeline.executor import LocalExecutor
        from multilingual_megatron.pipeline.stages.cooldown_readout.cooldown_readout_stage import (
            FIT_OBSERVATIONS_CSV,
            stage_cooldown_readout,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized = root / "tokenized"
            tokenized.mkdir()
            data_manifest = tokenized / "megatron_data_manifest.json"
            data_manifest.write_text(json.dumps({"prefixes": [str(tokenized / "dummy")], "weights": [1.0]}), encoding="utf-8")
            (tokenized / "tokenized_manifests.json").write_text(
                json.dumps(
                    {
                        "tokenizers": {
                            "scaling_fw2_k200_64k": {
                                "tokenizer_id": "scaling_fw2_k200_64k",
                                "tokenizer_type": "SentencePieceTokenizer",
                                "tokenizer_model": str(root / "tokenizer.model"),
                                "vocab_size": 64000,
                                "data_manifest": str(data_manifest),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            train_root = root / "runs"
            run_id = "scaling_dense_scale_scaling_a_g1_a_e1_dense_g1_locked"
            train_dir = train_root / run_id
            (train_dir / "checkpoints" / "iter_0062500").mkdir(parents=True)
            (train_dir / ".training_complete").write_text("", encoding="utf-8")

            train_cfg = {
                "backend": "megatron_bridge",
                "bridge_root": "external/Megatron-Bridge",
                "output_dir": str(train_root),
                "seq_length": 10,
                "nproc_per_node": 1,
                "nnodes": 1,
                "train": {"train_iters": 100, "global_batch_size": 8, "micro_batch_size": 2, "log_interval": 1},
                "eval": {"eval_interval": 100, "eval_iters": 0},
                "overrides": ["validation.eval_iters=0"],
                "runs": [
                    {
                        "id": run_id,
                        "backend": "megatron_bridge",
                        "kind": "megatron_bridge",
                        "recipe": "scaling_dense_scale_scaling_a_pretrain_config",
                        "tokenizer_id": "scaling_fw2_k200_64k",
                        "scaling": {
                            "family": "scaling_dense",
                            "scale_label": "scaling_a",
                            "variant_label": "a_e1_dense_g1_locked",
                            "num_experts": 1,
                            "num_gpus": 1,
                        },
                    }
                ],
            }
            profile = {
                "language_map_path": "configs/language_groupings/scaling_fw2_k001.json",
                "output_root": str(root),
                "cooldown_readout": {
                    "enabled": True,
                    "output_dir": str(root / "cooldown"),
                    "horizons_tokens_by_scale": {"scaling_a": [5_000_000]},
                    "cooldown_fraction": 0.05,
                    "cooldown_max_tokens": 256_000_000,
                    "eval_iters": 3,
                    "require_training_complete": True,
                    "require_checkpoint": True,
                    "run_forks": True,
                },
                "training": train_cfg,
            }
            ctx = PipelineContext(
                args=argparse.Namespace(dry_run=True, run_id=[]),
                repo_root=REPO_ROOT,
                profile=profile,
                force_stages=set(),
                active_stages=["cooldown_readout"],
                output_root=root,
                state_path=root / ".pipeline_state.json",
                language_map_path=root / "language_map.json",
                sample_cfg={},
                tokenizer_cfg={},
                tokenize_cfg={},
                audit_cfg={},
                train_cfg=train_cfg,
                summary_cfg={},
                benchmark_cfg={},
                baseline_eval_cfg={},
                wandb_cfg={},
                env={},
                logger=logging.getLogger("test"),
                executor=LocalExecutor(),
                sampled_dir=root / "sampled",
                sampled_manifest=root / "sampled" / "sampling_manifest.json",
                shard_subdir="shards",
                sample_shard_dir=root / "sampled" / "shards",
                tokenizers_dir=root / "tokenizers",
                tokenized_dir=tokenized,
                train_root=train_root,
                summary_path=root / "summary.csv",
                gate_path=root / "gate.json",
                benchmark_root=root / "benchmarks",
                baseline_cache_root=root / "baseline_cache",
                lineage_manifest_path=root / "lineage.json",
                logs_root=root / "logs",
                run_naming_cfg={},
                run_timestamp="20260605",
                naming_languages=[],
                lang_count=1,
                model_slug="test",
                exp_tag="test",
                base_run_label="test",
            )

            stage_cooldown_readout(ctx)

            plan = json.loads((root / "cooldown" / "checkpoint_readout_plan.json").read_text(encoding="utf-8"))
            row = plan["rows"][0]
            self.assertEqual(row["checkpoint_step"], 62500)
            self.assertEqual(row["cooldown_tokens"], 250000)
            self.assertEqual(row["cooldown_iters"], 3125)
            self.assertEqual(row["status"], "dry_run")
            command_plan = json.loads(Path(row["command_plan"]).read_text(encoding="utf-8"))
            cmd_text = "\n".join(command_plan["cmd"])
            self.assertIn("checkpoint.finetune=true", cmd_text)
            self.assertIn("checkpoint.load_optim=false", cmd_text)
            self.assertIn("checkpoint.ckpt_step=62500", cmd_text)
            extra_overrides = [
                command_plan["cmd"][index + 1]
                for index, value in enumerate(command_plan["cmd"])
                if value == "--extra-override"
            ]
            self.assertEqual(
                [value for value in extra_overrides if value.startswith("validation.eval_iters=")][-1],
                "validation.eval_iters=3",
            )
            self.assertTrue((root / "cooldown" / FIT_OBSERVATIONS_CSV).exists())
            with (root / "cooldown" / FIT_OBSERVATIONS_CSV).open(newline="", encoding="utf-8") as handle:
                observations = list(csv.DictReader(handle))
            self.assertEqual(observations[0]["K"], "1")
            self.assertEqual(observations[0]["D_checkpoint"], "5000000")
            self.assertFalse((root / ".pipeline_state.json").exists())

    def test_save_interval_checkpoint_resolution_uses_saved_step_not_nominal_step(self) -> None:
        from multilingual_megatron.pipeline.context import PipelineContext
        from multilingual_megatron.pipeline.executor import LocalExecutor
        from multilingual_megatron.pipeline.stages.cooldown_readout.cooldown_readout_stage import (
            stage_cooldown_readout,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized = root / "tokenized"
            tokenized.mkdir()
            data_manifest = tokenized / "megatron_data_manifest.json"
            data_manifest.write_text(json.dumps({"prefixes": [str(tokenized / "dummy")], "weights": [1.0]}), encoding="utf-8")
            (tokenized / "tokenized_manifests.json").write_text(
                json.dumps(
                    {
                        "tokenizers": {
                            "scaling_fw2_k200_64k": {
                                "tokenizer_id": "scaling_fw2_k200_64k",
                                "tokenizer_type": "SentencePieceTokenizer",
                                "tokenizer_model": str(root / "tokenizer.model"),
                                "vocab_size": 64000,
                                "data_manifest": str(data_manifest),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            train_root = root / "runs"
            run_id = "scaling_switch_moe_scale_scaling_b_g2_b_e16_locked"
            train_dir = train_root / run_id
            (train_dir / "checkpoints" / "iter_0000004").mkdir(parents=True)
            (train_dir / ".training_complete").write_text("", encoding="utf-8")

            train_cfg = {
                "backend": "megatron_bridge",
                "bridge_root": "external/Megatron-Bridge",
                "output_dir": str(train_root),
                "seq_length": 7,
                "nproc_per_node": 1,
                "nnodes": 1,
                "train": {
                    "train_iters": 4,
                    "global_batch_size": 1,
                    "micro_batch_size": 1,
                    "save_interval": 2,
                    "log_interval": 1,
                },
                "eval": {"eval_interval": 4, "eval_iters": 0},
                "runs": [
                    {
                        "id": run_id,
                        "backend": "megatron_bridge",
                        "kind": "megatron_bridge",
                        "recipe": "scaling_switch_moe_scale_scaling_b_pretrain_config",
                        "tokenizer_id": "scaling_fw2_k200_64k",
                        "scaling": {
                            "family": "scaling_switch_moe",
                            "scale_label": "scaling_b",
                            "variant_label": "b_e16_locked",
                            "num_experts": 16,
                            "num_gpus": 2,
                            "save_interval": 2,
                        },
                    }
                ],
            }
            profile = {
                "language_map_path": "configs/language_groupings/scaling_fw2_v2_k012.json",
                "output_root": str(root),
                "cooldown_readout": {
                    "enabled": True,
                    "output_dir": str(root / "cooldown"),
                    "horizons_tokens_by_scale": {"scaling_b": [10, 20, 30]},
                    "cooldown_fraction": 0.05,
                    "cooldown_max_tokens": 256_000_000,
                    "eval_iters": 3,
                    "require_training_complete": True,
                    "require_checkpoint": True,
                    "run_forks": True,
                    "strict": True,
                },
                "training": train_cfg,
            }
            ctx = PipelineContext(
                args=argparse.Namespace(dry_run=True, run_id=[]),
                repo_root=REPO_ROOT,
                profile=profile,
                force_stages=set(),
                active_stages=["cooldown_readout"],
                output_root=root,
                state_path=root / ".pipeline_state.json",
                language_map_path=root / "language_map.json",
                sample_cfg={},
                tokenizer_cfg={},
                tokenize_cfg={},
                audit_cfg={},
                train_cfg=train_cfg,
                summary_cfg={},
                benchmark_cfg={},
                baseline_eval_cfg={},
                wandb_cfg={},
                env={},
                logger=logging.getLogger("test"),
                executor=LocalExecutor(),
                sampled_dir=root / "sampled",
                sampled_manifest=root / "sampled" / "sampling_manifest.json",
                shard_subdir="shards",
                sample_shard_dir=root / "sampled" / "shards",
                tokenizers_dir=root / "tokenizers",
                tokenized_dir=tokenized,
                train_root=train_root,
                summary_path=root / "summary.csv",
                gate_path=root / "gate.json",
                benchmark_root=root / "benchmarks",
                baseline_cache_root=root / "baseline_cache",
                lineage_manifest_path=root / "lineage.json",
                logs_root=root / "logs",
                run_naming_cfg={},
                run_timestamp="20260605",
                naming_languages=[],
                lang_count=16,
                model_slug="test",
                exp_tag="test",
                base_run_label="test",
            )

            stage_cooldown_readout(ctx)

            plan = json.loads((root / "cooldown" / "checkpoint_readout_plan.json").read_text(encoding="utf-8"))
            rows = {row["target_checkpoint_tokens"]: row for row in plan["rows"]}
            row = rows[20]

        self.assertEqual(row["nominal_checkpoint_step"], 3)
        self.assertEqual(row["checkpoint_step"], 4)
        self.assertEqual(row["checkpoint_resolution"], "save_interval")
        self.assertEqual(row["checkpoint_tokens"], 28)
        self.assertEqual(row["checkpoint_token_delta"], 8)

    def test_readout_metric_falls_back_to_latest_log_lm_loss(self) -> None:
        from multilingual_megatron.pipeline.stages.cooldown_readout.cooldown_readout_stage import (
            _metric_from_readout_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "readout"
            output_dir.mkdir()
            (output_dir / "train_results.json").write_text(json.dumps({"train_loss": None}), encoding="utf-8")
            log_path = root / "readout.log"
            log_path.write_text(
                "\n".join(
                    [
                        "iteration       5/      10 | consumed samples: 10 | lm loss: 3.50E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |",
                        "iteration      10/      10 | consumed samples: 20 | lm loss: 3.25E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |",
                    ]
                ),
                encoding="utf-8",
            )

            metric = _metric_from_readout_artifacts(output_dir, log_path)

        self.assertAlmostEqual(metric["loss"], 3.25)
        self.assertEqual(metric["source"], "log_lm_loss")
        self.assertEqual(metric["global_step"], 10)
        self.assertEqual(metric["skipped_iterations"], 0)
        self.assertEqual(metric["nan_iterations"], 0)

    def test_strict_helpers_reject_dry_run_cache_and_nearest_existing_checkpoint(self) -> None:
        from multilingual_megatron.pipeline.stages.cooldown_readout.cooldown_readout_stage import (
            _resolve_checkpoint_for_horizon,
            _strict_failure_rows,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp) / "checkpoints"
            (checkpoint_root / "iter_0000001").mkdir(parents=True)

            resolved = _resolve_checkpoint_for_horizon(
                checkpoint_root=checkpoint_root,
                horizon_tokens=20,
                global_batch_size=1,
                seq_length=7,
                save_interval=2,
                base_interval_tokens=10,
                base_train_iters=6,
                final_horizon=False,
            )

        self.assertEqual(resolved["nominal_checkpoint_step"], 3)
        self.assertEqual(resolved["checkpoint_step"], 1)
        self.assertEqual(resolved["checkpoint_resolution"], "nearest_existing")
        self.assertEqual(_strict_failure_rows([{"status": "dry_run"}], allow_dry_run=True), [])
        self.assertEqual(_strict_failure_rows([{"status": "dry_run"}], allow_dry_run=False), [{"status": "dry_run"}])


if __name__ == "__main__":
    unittest.main()
