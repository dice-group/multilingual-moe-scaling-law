from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.training.checkpoint_state import (  # noqa: E402
    inspect_full_state_checkpoint,
)
from multilingual_megatron.pipeline.stages.training.segment_runner import (  # noqa: E402
    _replace_extra_override,
    _run_segment,
)


class CheckpointStateTest(unittest.TestCase):
    def _checkpoint(
        self,
        root: Path,
        *,
        step: int = 12,
        samples: int = 384,
        scheduler_samples: int | None = None,
    ) -> Path:
        import torch

        checkpoint_root = root / "checkpoints"
        checkpoint = checkpoint_root / f"iter_{step:07d}"
        checkpoint.mkdir(parents=True)
        (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
        (checkpoint / ".metadata").write_bytes(b"fixture")
        torch.save(
            {"step": torch.tensor(step), "consumed_train_samples": torch.tensor(samples)},
            checkpoint / "train_state.pt",
        )
        torch.save(
            {
                "iteration": step,
                "optimizer": {"optimizer": {}},
                "opt_param_scheduler": {
                    "num_steps": samples if scheduler_samples is None else scheduler_samples,
                    "lr_decay_style": "WSD",
                },
            },
            checkpoint / "common.pt",
        )
        return checkpoint_root

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={
            "embedding.word_embeddings.weight",
            "optimizer.state.exp_avg.embedding.word_embeddings.weight",
            "rng_state/shard_0.0_1.1",
        },
    )
    def test_exact_full_state_checkpoint_passes(self, _metadata) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = self._checkpoint(Path(tmp))
            result = inspect_full_state_checkpoint(
                checkpoint_root=checkpoint_root,
                expected_step=12,
                expected_consumed_samples=384,
            )
        self.assertTrue(result.valid_full_state)
        self.assertEqual(result.errors, ())
        self.assertEqual(result.scheduler_num_steps, 384)
        self.assertTrue(result.has_optimizer_state)
        self.assertTrue(result.has_rng_state)

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={
            "embedding.word_embeddings.weight",
            "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.exp_avg",
            "rng_state/shard_0.0_1.1",
        },
    )
    def test_distributed_optimizer_checkpoint_passes(self, _metadata) -> None:
        """Megatron sharded optimizer tensors are a full resume state."""

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = self._checkpoint(Path(tmp))
            result = inspect_full_state_checkpoint(
                checkpoint_root=checkpoint_root,
                expected_step=12,
                expected_consumed_samples=384,
            )
        self.assertTrue(result.valid_full_state)
        self.assertTrue(result.has_optimizer_state)

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={
            "embedding.word_embeddings.weight",
            "optimizer.state.exp_avg.embedding.word_embeddings.weight",
            "rng_state/shard_0.0_1.1",
        },
    )
    def test_scheduler_progress_must_match_expected_consumed_samples(self, _metadata) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = self._checkpoint(Path(tmp), scheduler_samples=320)
            result = inspect_full_state_checkpoint(
                checkpoint_root=checkpoint_root,
                expected_step=12,
                expected_consumed_samples=384,
            )
        self.assertFalse(result.valid_full_state)
        self.assertEqual(result.scheduler_num_steps, 320)
        self.assertIn("scheduler_num_steps_mismatch:320!=384", result.errors)

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={
            "embedding.word_embeddings.weight",
            "optimizer.state.exp_avg.embedding.word_embeddings.weight",
            "rng_state/shard_0.0_1.1",
        },
    )
    def test_scheduler_progress_is_required(self, _metadata) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = self._checkpoint(Path(tmp))
            checkpoint = checkpoint_root / "iter_0000012"
            common = torch.load(checkpoint / "common.pt", map_location="cpu", weights_only=False)
            common["opt_param_scheduler"].pop("num_steps")
            torch.save(common, checkpoint / "common.pt")
            result = inspect_full_state_checkpoint(
                checkpoint_root=checkpoint_root,
                expected_step=12,
                expected_consumed_samples=384,
            )
        self.assertFalse(result.valid_full_state)
        self.assertIsNone(result.scheduler_num_steps)
        self.assertIn("scheduler_num_steps_missing", result.errors)

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={"embedding.word_embeddings.weight"},
    )
    def test_wrong_endpoint_and_missing_resume_state_fail_closed(self, _metadata) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = self._checkpoint(Path(tmp))
            result = inspect_full_state_checkpoint(
                checkpoint_root=checkpoint_root,
                expected_step=13,
                expected_consumed_samples=512,
            )
        self.assertFalse(result.valid_full_state)
        self.assertIn("optimizer_state_missing", result.errors)
        self.assertIn("rng_state_missing", result.errors)
        self.assertTrue(any(error.startswith("latest_step_mismatch") for error in result.errors))
        self.assertTrue(any(error.startswith("consumed_train_samples_mismatch") for error in result.errors))

    def test_resume_override_replaces_stale_parent_checkpoint(self) -> None:
        cmd = [
            "python",
            "train.py",
            "--extra-override",
            "checkpoint.load=/old",
            "--extra-override",
            "checkpoint.load_rng=true",
        ]
        updated = _replace_extra_override(cmd, "checkpoint.load", "/new")
        self.assertNotIn("checkpoint.load=/old", updated)
        self.assertIn("checkpoint.load=/new", updated)
        self.assertIn("checkpoint.load_rng=true", updated)

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={
            "embedding.word_embeddings.weight",
            "optimizer.state.exp_avg.embedding.word_embeddings.weight",
            "rng_state/shard_0.0_1.1",
        },
    )
    def test_segment_runner_skips_only_an_exact_verified_checkpoint(self, _metadata) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_root = self._checkpoint(root, step=12, samples=384)
            output_dir = checkpoint_root.parent
            segment = {
                "segment_index": 0,
                "cmd": ["this-command-must-not-run"],
                "log_path": str(root / "segment.log"),
                "output_dir": str(output_dir),
                "cumulative_start_step": 0,
                "cumulative_end_step": 12,
                "expected_end_step": 12,
                "expected_consumed_samples": 384,
                "fixed_dataset_sample_horizon": 1000,
                "actual_end_tokens": 384 * 2048,
            }
            result = _run_segment(segment, env={}, cwd=None)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["execution"], "verified_checkpoint_skip")
        self.assertTrue(result["training_complete"])

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={"embedding.weight", "optimizer.state.exp_avg.embedding.weight", "rng_state/shard_0"},
    )
    def test_branch_execution_is_bound_to_the_audited_parent(self, _metadata) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._checkpoint(root / "parent", step=12, samples=384)
            segment = {
                "segment_index": 0,
                "cmd": ["python", "wrapper.py", "--extra-override", "checkpoint.load=relative/parent"],
                "log_path": str(root / "branch.log"),
                "output_dir": str(root / "branch"),
                "initial_checkpoint": {
                    "checkpoint_root": str(parent), "expected_step": 12,
                    "expected_consumed_samples": 384,
                },
            }
            with patch(
                "multilingual_megatron.pipeline.stages.training.segment_runner.subprocess.run",
                side_effect=RuntimeError("captured engine launch"),
            ) as launch:
                with self.assertRaisesRegex(RuntimeError, "captured engine launch"):
                    _run_segment(segment, env={}, cwd=str(root))
            cmd = launch.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--resume-checkpoint-root") + 1], str(parent.resolve()))
            self.assertEqual(cmd[cmd.index("--resume-checkpoint-step") + 1], "12")
            # Preserve ramp-aware samples; 12 * current GBS is not authoritative.
            self.assertEqual(cmd[cmd.index("--resume-checkpoint-consumed-samples") + 1], "384")

    def test_restored_resume_guard_rejects_reset_counters_and_weights_only_loads(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.resume_guard import verify_restored_resume

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"RANK": "0"}):
            root = Path(tmp)
            context = SimpleNamespace(
                state=SimpleNamespace(
                    train_state=SimpleNamespace(step=12, consumed_train_samples=384),
                    cfg=SimpleNamespace(checkpoint=SimpleNamespace(
                        load=str(root), load_optim=True, load_rng=True, finetune=False,
                    )),
                ),
                scheduler=SimpleNamespace(num_steps=384), optimizer=object(),
            )
            receipt = root / "restored.json"
            verify_restored_resume(context, expected_step=12, expected_samples=384, receipt_path=receipt)
            saved = json.loads((root / "restored.rank0.json").read_text())
            self.assertEqual(saved["status"], "verified_before_first_update")
            for target, attr, bad_value in (
                (context.state.train_state, "step", 0),
                (context.state.train_state, "consumed_train_samples", 0),
                (context.scheduler, "num_steps", 0),
                (context.state.cfg.checkpoint, "load_optim", False),
                (context.state.cfg.checkpoint, "load_rng", False),
                (context.state.cfg.checkpoint, "finetune", True),
                (context.state.cfg.checkpoint, "load", "relative/parent"),
            ):
                with self.subTest(attr=attr), patch.object(target, attr, bad_value):
                    with self.assertRaises(RuntimeError):
                        verify_restored_resume(context, expected_step=12, expected_samples=384, receipt_path=receipt)

    @patch(
        "multilingual_megatron.pipeline.stages.training.checkpoint_state._distributed_metadata_keys",
        return_value={"embedding.weight", "optimizer.state.exp_avg.embedding.weight", "rng_state/shard_0"},
    )
    def test_wrapper_pins_parent_and_passes_pre_update_guard(self, _metadata) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge import run_megatron_bridge_training as wrapper

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._checkpoint(root / "parent", step=12, samples=384)
            bridge = root / "external" / "Megatron-Bridge"
            bridge.mkdir(parents=True)
            argv = [
                "wrapper", "--bridge-root", str(bridge), "--output-dir", str(root / "run"),
                "--recipe", "fixture", "--resume-checkpoint-root", str(parent),
                "--resume-checkpoint-step", "12", "--resume-checkpoint-consumed-samples", "384",
                "--extra-override", "checkpoint.load=relative/parent",
            ]
            with patch.object(sys, "argv", argv), patch.object(wrapper, "_resolve_master_port", return_value=29500), patch.object(
                wrapper, "_prepare_pybind11_shim", return_value=None,
            ), patch.object(wrapper.subprocess, "run", side_effect=RuntimeError("captured engine launch")) as launch:
                with self.assertRaisesRegex(RuntimeError, "captured engine launch"):
                    wrapper.main()
            cmd = launch.call_args.args[0]
            self.assertEqual(launch.call_args.kwargs["cwd"], str(bridge))
            self.assertIn(f"checkpoint.load={parent.resolve() / 'iter_0000012'}", cmd)
            self.assertNotIn("checkpoint.load=relative/parent", cmd)
            self.assertIn("checkpoint.ckpt_step=null", cmd)
            self.assertIn("checkpoint.load_optim=true", cmd)
            self.assertIn("checkpoint.load_rng=true", cmd)
            self.assertEqual(cmd[cmd.index("--expected-resume-step") + 1], "12")
            self.assertEqual(cmd[cmd.index("--expected-resume-consumed-samples") + 1], "384")
            self.assertEqual(cmd[cmd.index("--resume-state-receipt") + 1], str(root / "run/restored_resume_state.json"))


if __name__ == "__main__":
    unittest.main()
