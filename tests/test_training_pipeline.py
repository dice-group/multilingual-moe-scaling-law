from __future__ import annotations

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


class RecordingExecutor:
    def __init__(self, *, complete: bool = False, metrics_by_mbs: dict[int, dict[str, float]] | None = None) -> None:
        self.complete = complete
        self.metrics_by_mbs = metrics_by_mbs or {}
        self.calls: list[tuple[list[str], dict[str, str], bool, Path]] = []
        self.batch_calls: list[list[tuple[list[str], dict[str, str], Path]]] = []

    def run(self, cmd: list[str], env: dict[str, str], dry_run: bool, log_path: Path) -> None:
        self.calls.append((list(cmd), dict(env), dry_run, log_path))
        if self.complete and not dry_run:
            self._complete_from_cmd(cmd, log_path)

    def run_batch(self, jobs, dry_run: bool) -> None:
        jobs = [(list(cmd), dict(env), Path(log_path)) for cmd, env, log_path in jobs]
        self.batch_calls.append(jobs)
        for cmd, env, log_path in jobs:
            self.run(cmd, env, dry_run, log_path)

    def _complete_from_cmd(self, cmd: list[str], log_path: Path) -> None:
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        train_iters = int(cmd[cmd.index("--train-iters") + 1])
        global_batch = int(cmd[cmd.index("--global-batch-size") + 1])
        micro_batch = int(cmd[cmd.index("--micro-batch-size") + 1])
        seq_length = int(cmd[cmd.index("--seq-length") + 1])
        (output_dir / "train_results.json").write_text(
            json.dumps(
                {
                    "backend": "megatron_bridge",
                    "dataset": "llm-pretrain",
                    "recipe": "test_recipe",
                    "status": "complete",
                    "train_runtime": 1.0,
                    "train_loss": None,
                    "num_input_tokens_seen": train_iters * global_batch * seq_length,
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "trainer_state.json").write_text(
            json.dumps({"backend": "megatron_bridge", "status": "complete", "global_step": train_iters}),
            encoding="utf-8",
        )
        (output_dir / ".training_complete").write_text("ok\n", encoding="utf-8")
        metrics = self.metrics_by_mbs.get(micro_batch, {})
        if metrics:
            tokens_per_second = float(metrics.get("tokens_per_second", 1000.0))
            model_tflops = float(metrics.get("model_tflops", 0.0))
            allocated = float(metrics.get("allocated_gb", 10.0))
            reserved = float(metrics.get("reserved_gb", allocated))
            step_ms = 1000.0 * global_batch * seq_length / tokens_per_second
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for step in range(1, train_iters + 1):
                lines.append(
                    f"iteration {step}/ {train_iters} | elapsed time per iteration (ms): {step_ms:.3f} | "
                    f"global batch size: {global_batch}"
                )
                lines.append(
                    f"mem-max-allocated-gigabytes: {allocated:.3f} | "
                    f"mem-max-reserved-gigabytes: {reserved:.3f}"
                )
                if model_tflops > 0:
                    lines.append(f"GPU utilization: {model_tflops:.2f}MODEL_TFLOP/s/GPU")
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tokenized_index(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "stage": "tokenize",
        "backend": "megatron_gpt",
        "tokenizers": {
            "tok_a": {
                "tokenizer_id": "tok_a",
                "tokenizer_type": "SentencePieceTokenizer",
                "tokenizer_model": str(root / "tok_a.model"),
                "trust_remote_code": False,
                "vocab_size": 64000,
                "data_manifest": str(root / "tok_a" / "megatron_data_manifest.json"),
            },
            "tok_b": {
                "tokenizer_id": "tok_b",
                "tokenizer_type": "HuggingFaceTokenizer",
                "tokenizer_model": str(root / "tok_b"),
                "trust_remote_code": True,
                "vocab_size": 32000,
                "data_manifest": str(root / "tok_b" / "megatron_data_manifest.json"),
            },
        },
    }
    (root / "tokenized_manifests.json").write_text(json.dumps(payload), encoding="utf-8")


def _ctx(root: Path, *, runs: list[dict], dry_run: bool, executor: RecordingExecutor) -> SimpleNamespace:
    tokenized_dir = root / "tokenized"
    _write_tokenized_index(tokenized_dir)
    return SimpleNamespace(
        active_stages=["train"],
        args=SimpleNamespace(dry_run=dry_run),
        base_run_label="study",
        env={},
        executor=executor,
        force_stages=set(),
        language_map_path=root / "language_map.json",
        logger=logging.getLogger("test"),
        logs_root=root / "logs",
        repo_root=REPO_ROOT,
        state_path=root / "state" / "pipeline_state.json",
        tokenized_dir=tokenized_dir,
        train_root=root / "runs",
        train_cfg={
            "enabled": True,
            "backend": "megatron_bridge",
            "bridge_root": "external/Megatron-Bridge",
            "dataset": "llm-pretrain",
            "recipe": "test_recipe",
            "seq_length": 128,
            "train": {"global_batch_size": 8, "micro_batch_size": 2, "train_iters": 3},
            "eval": {"eval_interval": 3, "eval_iters": 0},
            "runs": runs,
        },
    )


class TrainingPipelineTest(unittest.TestCase):
    def test_calibration_candidate_generation_filters_illegal_mbs(self) -> None:
        from multilingual_megatron.pipeline.stages.training.calibration import (
            TrainingCalibrationSettings,
            candidate_batch_shapes,
            candidate_micro_batch_sizes,
        )

        candidates = candidate_micro_batch_sizes(
            settings=TrainingCalibrationSettings(candidate_micro_batch_sizes=(1, 2, 3, 4, 8, 16)),
            base_micro_batch_size=1,
            global_batch_size=8,
        )

        self.assertEqual(candidates, [1, 2, 4, 8])

        shapes = candidate_batch_shapes(
            settings=TrainingCalibrationSettings(
                candidate_batch_pairs=((176, 22), (160, 20), (129, 16)),
            ),
            base_micro_batch_size=1,
            global_batch_size=128,
        )

        self.assertEqual(shapes, [(176, 22), (160, 20)])

    def test_calibration_selector_prefers_largest_when_smaller_not_clearly_better(self) -> None:
        from multilingual_megatron.pipeline.stages.training.calibration import select_calibration_candidate

        selected = select_calibration_candidate(
            [
                {"selectable": True, "micro_batch_size": 2, "median_tokens_per_second": 1000.0},
                {"selectable": True, "micro_batch_size": 4, "median_tokens_per_second": 980.0},
            ],
            throughput_drop_threshold=0.03,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["micro_batch_size"], 4)
        self.assertEqual(selected["selection_rationale"], "largest_stable")

    def test_calibration_selector_uses_smaller_when_clearly_faster(self) -> None:
        from multilingual_megatron.pipeline.stages.training.calibration import select_calibration_candidate

        selected = select_calibration_candidate(
            [
                {"selectable": True, "micro_batch_size": 2, "median_tokens_per_second": 1050.0},
                {"selectable": True, "micro_batch_size": 4, "median_tokens_per_second": 1000.0},
            ],
            throughput_drop_threshold=0.03,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["micro_batch_size"], 2)
        self.assertEqual(selected["selection_rationale"], "smaller_faster_by_threshold")

    def test_calibration_oom_detector_matches_torch_out_of_memory_error(self) -> None:
        from multilingual_megatron.pipeline.stages.training.calibration import _is_oom

        self.assertTrue(_is_oom("torch.OutOfMemoryError: CUDA allocation failed"))

    def test_command_builder_resolves_tokenized_manifest_entry(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            cmd, env = build_training_cmd(
                repo_root=REPO_ROOT,
                env={"BASE": "1", "PYTHON_EXECUTABLE": "/opt/test-python"},
                output_dir=root / "run",
                run_cfg={"id": "run_a", "tokenizer_id": "tok_a"},
                common={
                    "backend": "megatron_bridge",
                    "bridge_root": "external/Megatron-Bridge",
                    "dataset": "llm-pretrain",
                    "recipe": "test_recipe",
                    "seq_length": 128,
                    "train": {"global_batch_size": 8, "micro_batch_size": 2, "train_iters": 3},
                },
                language_map_path=root / "language_map.json",
                tokenized_path=tokenized_dir,
                run_name="run-a",
            )

        self.assertEqual(env["BASE"], "1")
        self.assertEqual(cmd[0], "/opt/test-python")
        self.assertEqual(cmd[cmd.index("--python-executable") + 1], "/opt/test-python")
        joined = " ".join(cmd)
        self.assertIn("--data-manifest", cmd)
        self.assertIn("tok_a/megatron_data_manifest.json", joined)
        self.assertIn("tokenizer.tokenizer_type=SentencePieceTokenizer", cmd)
        self.assertIn("model.vocab_size=64000", cmd)

    def test_command_builder_passes_explicit_nsys_binary(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            cmd, _env = build_training_cmd(
                repo_root=REPO_ROOT,
                env={"PYTHON_EXECUTABLE": "/opt/test-python"},
                output_dir=root / "run",
                run_cfg={"id": "run_a", "tokenizer_id": "tok_a"},
                common={
                    "backend": "megatron_bridge",
                    "bridge_root": "external/Megatron-Bridge",
                    "dataset": "llm-pretrain",
                    "recipe": "test_recipe",
                    "seq_length": 128,
                    "train": {"global_batch_size": 8, "micro_batch_size": 2, "train_iters": 3},
                    "launch": {"nsys_profile": True, "nsys_bin": "/opt/nsight/bin/nsys"},
                },
                language_map_path=root / "language_map.json",
                tokenized_path=tokenized_dir,
                run_name="run-a",
            )

        self.assertIn("--nsys-profile", cmd)
        self.assertEqual(cmd[cmd.index("--nsys-bin") + 1], "/opt/nsight/bin/nsys")

    def test_nsys_wrapper_fails_fast_when_binary_missing(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import _with_nsys_profile

        args = SimpleNamespace(
            nsys_profile=True,
            nsys_bin="/definitely/missing/nsys",
            nsys_output_dir="",
            nsys_trace="cuda,nvtx",
            nsys_extra_arg=[],
        )

        with self.assertRaises(FileNotFoundError):
            _with_nsys_profile(["python", "-c", "pass"], args, Path(tempfile.gettempdir()))

    def test_training_calibration_probe_disables_profile_by_default(self) -> None:
        from multilingual_megatron.pipeline.stages.training.calibration import (
            TrainingCalibrationSettings,
            build_calibration_probe_plan,
        )
        from multilingual_megatron.pipeline.stages.training.planner import build_training_run_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "run_a", "tokenizer_id": "tok_a"}
            executor = RecordingExecutor()
            ctx = _ctx(root, runs=[run], dry_run=True, executor=executor)
            common = dict(ctx.train_cfg)
            common["launch"] = {"nsys_profile": True, "nsys_bin": "/opt/nsight/bin/nsys"}
            common["overrides"] = ["profiling.use_nsys_profiler=true"]
            base_plan = build_training_run_plan(ctx, run=run, common=common)
            probe = build_calibration_probe_plan(
                ctx,
                run_cfg=run,
                common=common,
                base_plan=base_plan,
                settings=TrainingCalibrationSettings(probe_train_iters=8, profile_probes=False),
                global_batch_size=8,
                micro_batch_size=2,
            )

        self.assertNotIn("--nsys-profile", probe.cmd)
        self.assertIn("profiling.use_nsys_profiler=false", probe.cmd)
        self.assertIn("profiling.use_pytorch_profiler=false", probe.cmd)

    def test_batch_policy_training_plan_materializes_segment_runner(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import (
            SEGMENT_PLAN_NAME,
            build_training_run_plan,
            write_training_command_plan,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "run_a", "tokenizer_id": "tok_a"}
            ctx = _ctx(root, runs=[run], dry_run=True, executor=RecordingExecutor())
            common = dict(ctx.train_cfg)
            common["seq_length"] = 16
            common["overrides"] = ["checkpoint.save=null", "checkpoint.save_interval=0"]
            common["train"] = {
                "global_batch_size": 128,
                "micro_batch_size": 64,
                "token_budget_tokens": 1600,
                "batch_policy": {
                    "enabled": True,
                    "target_global_batch_size": 128,
                    "schedule": [
                        {"start_tokens": 0, "end_tokens": 512, "global_batch_size": 32},
                        {"start_tokens": 512, "end_tokens": 1024, "global_batch_size": 64},
                        {"start_tokens": 1024, "end_tokens": 0, "global_batch_size": 128},
                    ],
                },
            }

            plan = build_training_run_plan(ctx, run=run, common=common)
            write_training_command_plan(plan)
            segment_plan = json.loads((plan.output_dir / SEGMENT_PLAN_NAME).read_text(encoding="utf-8"))
            command_plan = json.loads((plan.output_dir / "training_command_plan.json").read_text(encoding="utf-8"))

        self.assertIn("multilingual_megatron.pipeline.stages.training.segment_runner", plan.cmd)
        self.assertTrue(plan.metadata["segmented_training"])
        self.assertEqual([segment["global_batch_size"] for segment in segment_plan["segments"]], [32, 64, 128])
        self.assertEqual([segment["micro_batch_size"] for segment in segment_plan["segments"]], [32, 64, 64])
        first_cmd = segment_plan["segments"][0]["cmd"]
        second_cmd = segment_plan["segments"][1]["cmd"]
        self.assertEqual(first_cmd[first_cmd.index("--global-batch-size") + 1], "32")
        self.assertEqual(first_cmd[first_cmd.index("--micro-batch-size") + 1], "32")
        self.assertEqual(first_cmd[first_cmd.index("--train-iters") + 1], "1")
        self.assertEqual(first_cmd[first_cmd.index("--save-interval") + 1], "1")
        self.assertEqual(second_cmd[second_cmd.index("--save-interval") + 1], "2")
        first_overrides = [
            first_cmd[index + 1]
            for index, value in enumerate(first_cmd)
            if value == "--extra-override"
        ]
        save_overrides = [value for value in first_overrides if value.startswith("checkpoint.save=")]
        save_interval_overrides = [value for value in first_overrides if value.startswith("checkpoint.save_interval=")]
        self.assertEqual(save_overrides[-1], "checkpoint.save={}".format(Path(segment_plan["segments"][0]["output_dir"]) / "checkpoints"))
        self.assertEqual(save_interval_overrides[-1], "checkpoint.save_interval=1")
        self.assertFalse(any(value.startswith("train.exit_interval=") for value in first_overrides))
        self.assertIn("checkpoint.load=", "\n".join(second_cmd))
        self.assertNotIn("segment_plan", command_plan["metadata"])

    def test_batch_policy_can_require_audited_full_state_checkpoint_resume(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_run_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "checkpoint_resume", "tokenizer_id": "tok_a"}
            ctx = _ctx(root, runs=[run], dry_run=True, executor=RecordingExecutor())
            common = dict(ctx.train_cfg)
            common["seq_length"] = 16
            common["train"] = {
                "global_batch_size": 8,
                "micro_batch_size": 2,
                "token_budget_tokens": 256,
                "batch_policy": {
                    "enabled": True,
                    "target_global_batch_size": 8,
                    "data_parallel_size": 2,
                    "require_full_state_checkpoint": True,
                    "schedule": [
                        {"start_tokens": 0, "end_tokens": 128, "global_batch_size": 8},
                        {"start_tokens": 128, "end_tokens": 0, "global_batch_size": 8},
                    ],
                },
            }
            plan = build_training_run_plan(ctx, run=run, common=common)
            segments = plan.metadata["segment_plan"]["segments"]

        self.assertEqual([segment["expected_end_step"] for segment in segments], [1, 2])
        self.assertTrue(all(segment["require_full_state_checkpoint"] for segment in segments))
        for segment in segments:
            self.assertIn("--require-full-state-checkpoint", segment["cmd"])
            overrides = [
                segment["cmd"][index + 1]
                for index, value in enumerate(segment["cmd"])
                if value == "--extra-override"
            ]
            self.assertIn("checkpoint.async_save=false", overrides)
            self.assertNotIn("checkpoint.async_save=true", overrides)
        parent = segments[1]["initial_checkpoint"]
        self.assertEqual(parent["expected_step"], 1)
        self.assertEqual(parent["expected_consumed_samples"], 8)

    def test_joint_moe_v2_trunk_uses_one_fixed_dataset_horizon_and_wsd_samples(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import JOINT_MOE_FULL_STATE_WSD_V2
        from multilingual_megatron.pipeline.stages.training.planner import build_training_run_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "run_a", "tokenizer_id": "tok_a"}
            ctx = _ctx(root, runs=[run], dry_run=True, executor=RecordingExecutor())
            common = dict(ctx.train_cfg)
            common["seq_length"] = 2048
            common["train"] = {
                "global_batch_size": 128,
                "micro_batch_size": 64,
                "token_budget_tokens": 6_400_000_000,
                "batch_policy": {
                    "enabled": True,
                    "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                    "plan_role": "trunk",
                    "dataset_horizon_tokens": 8_000_000_000,
                    "checkpoint_horizons_tokens": [
                        500_000_000,
                        1_000_000_000,
                        2_000_000_000,
                        4_000_000_000,
                        8_000_000_000,
                    ],
                },
            }

            plan = build_training_run_plan(ctx, run=run, common=common)
            segment_plan = plan.metadata["segment_plan"]

        self.assertEqual(segment_plan["schema_version"], 2)
        self.assertEqual(segment_plan["planning_method"], JOINT_MOE_FULL_STATE_WSD_V2)
        self.assertEqual(segment_plan["plan_role"], "trunk")
        self.assertEqual(segment_plan["fixed_dataset_sample_horizon"], 4_883_072)
        self.assertEqual(segment_plan["expected_end_step"], 32_045)
        self.assertEqual(segment_plan["expected_consumed_samples"], 3_125_120)
        self.assertEqual(
            [segment["target_end_tokens"] for segment in segment_plan["segments"]],
            [400_000_000, 500_000_000, 800_000_000, 1_000_000_000, 1_600_000_000, 3_200_000_000, 6_400_000_000],
        )
        for segment in segment_plan["segments"]:
            cmd = segment["cmd"]
            gbs = int(cmd[cmd.index("--global-batch-size") + 1])
            train_iters = int(cmd[cmd.index("--train-iters") + 1])
            self.assertEqual(train_iters * gbs, 4_883_072)
            self.assertIn("--require-full-state-checkpoint", cmd)
            self.assertEqual(
                int(cmd[cmd.index("--expected-end-step") + 1]),
                segment["expected_end_step"],
            )
            self.assertEqual(
                int(cmd[cmd.index("--expected-consumed-samples") + 1]),
                segment["expected_consumed_samples"],
            )
            overrides = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--extra-override"]
            self.assertIn("checkpoint.async_save=false", overrides)
            self.assertNotIn("checkpoint.async_save=true", overrides)
            self.assertIn(f"train.exit_interval={segment['expected_end_step']}", overrides)
            decay_iters = int(
                next(value.split("=", 1)[1] for value in overrides if value.startswith("scheduler.lr_decay_iters="))
            )
            wsd_iters = int(
                next(value.split("=", 1)[1] for value in overrides if value.startswith("scheduler.lr_wsd_decay_iters="))
            )
            warmup_iters = int(
                next(value.split("=", 1)[1] for value in overrides if value.startswith("scheduler.lr_warmup_iters="))
            )
            self.assertEqual(decay_iters * gbs, 3_906_432)
            self.assertEqual(wsd_iters * gbs, 781_312)
            self.assertEqual(warmup_iters * gbs, 63_488)
            self.assertIn("scheduler.lr_decay_style=WSD", overrides)
            self.assertIn("scheduler.lr_wsd_decay_style=linear", overrides)
            self.assertIn("optimizer.min_lr=0.0", overrides)

    def test_joint_moe_v2_branch_plans_from_0_8d_to_d_with_same_dataset_horizon(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import JOINT_MOE_FULL_STATE_WSD_V2
        from multilingual_megatron.pipeline.stages.training.planner import build_training_run_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "run_a_branch", "tokenizer_id": "tok_a"}
            ctx = _ctx(root, runs=[run], dry_run=True, executor=RecordingExecutor())
            common = dict(ctx.train_cfg)
            common["seq_length"] = 2048
            common["train"] = {
                "global_batch_size": 128,
                "micro_batch_size": 64,
                "token_budget_tokens": 2_000_000_000,
                "batch_policy": {
                    "enabled": True,
                    "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                    "plan_role": "branch",
                    "branch_target_tokens": 2_000_000_000,
                    "branch_parent_checkpoint_root": str(root / "trunk" / "checkpoints"),
                    "dataset_horizon_tokens": 8_000_000_000,
                    "checkpoint_horizons_tokens": [
                        500_000_000,
                        1_000_000_000,
                        2_000_000_000,
                        4_000_000_000,
                        8_000_000_000,
                    ],
                },
            }

            plan = build_training_run_plan(ctx, run=run, common=common)
            segment_plan = plan.metadata["segment_plan"]

        self.assertEqual(segment_plan["plan_role"], "branch")
        self.assertEqual(segment_plan["branch_target_tokens"], 2_000_000_000)
        self.assertEqual(segment_plan["fixed_dataset_sample_horizon"], 4_883_072)
        self.assertEqual(len(segment_plan["segments"]), 1)
        segment = segment_plan["segments"][0]
        self.assertEqual(segment["target_start_tokens"], 1_600_000_000)
        self.assertEqual(segment["target_end_tokens"], 2_000_000_000)
        self.assertEqual(segment["cumulative_start_step"], 13_734)
        self.assertEqual(segment["expected_end_step"], 15_260)
        self.assertEqual(segment["expected_consumed_samples"], 976_640)
        self.assertEqual(
            segment["initial_checkpoint"],
            {
                "checkpoint_root": str(root / "trunk" / "checkpoints"),
                "expected_step": 13_734,
                "expected_consumed_samples": 781_312,
            },
        )
        cmd = segment["cmd"]
        self.assertEqual(cmd[cmd.index("--train-iters") + 1], "38149")
        overrides = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--extra-override"]
        self.assertIn("train.exit_interval=15260", overrides)
        self.assertIn("scheduler.lr_decay_iters=7630", overrides)
        self.assertIn("scheduler.lr_wsd_decay_iters=1526", overrides)
        self.assertIn("scheduler.lr_warmup_iters=496", overrides)
        self.assertIn("scheduler.lr_decay_style=WSD", overrides)
        self.assertIn("scheduler.lr_wsd_decay_style=linear", overrides)
        self.assertIn("optimizer.min_lr=0.0", overrides)
        self.assertIn("checkpoint.async_save=false", overrides)
        self.assertNotIn("checkpoint.async_save=true", overrides)
        self.assertEqual(7630 * 128, 976_640)
        self.assertEqual(1526 * 128, 195_328)
        self.assertEqual(496 * 128, 63_488)
        self.assertIn(f"checkpoint.load={root / 'trunk' / 'checkpoints'}", overrides)
        self.assertIn("checkpoint.ckpt_step=13734", overrides)
        self.assertIn("checkpoint.load_optim=true", overrides)
        self.assertIn("checkpoint.load_rng=true", overrides)

    def test_joint_moe_v2_branch_saves_readout_without_creating_another_branch(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import JOINT_MOE_FULL_STATE_WSD_V2
        from multilingual_megatron.pipeline.stages.training.planner import build_training_run_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "run_final_branch", "tokenizer_id": "tok_a"}
            ctx = _ctx(root, runs=[run], dry_run=True, executor=RecordingExecutor())
            common = dict(ctx.train_cfg)
            common["seq_length"] = 2048
            common["train"] = {
                "global_batch_size": 128,
                "micro_batch_size": 64,
                "token_budget_tokens": 8_000_000_000,
                "batch_policy": {
                    "enabled": True,
                    "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                    "plan_role": "branch",
                    "branch_target_tokens": 8_000_000_000,
                    "branch_parent_checkpoint_root": str(root / "trunk" / "checkpoints"),
                    "dataset_horizon_tokens": 8_000_000_000,
                    "checkpoint_horizons_tokens": [8_000_000_000],
                    "readout_horizons_tokens": [7_000_000_000],
                },
            }

            plan = build_training_run_plan(ctx, run=run, common=common)
            segment_plan = plan.metadata["segment_plan"]

        self.assertEqual(segment_plan["checkpoint_horizons_tokens"], [8_000_000_000])
        self.assertEqual(segment_plan["readout_horizons_tokens"], [7_000_000_000])
        self.assertEqual(len(segment_plan["segments"]), 2)
        self.assertEqual(
            [segment["target_end_tokens"] for segment in segment_plan["segments"]],
            [7_000_000_000, 8_000_000_000],
        )
        self.assertEqual(segment_plan["readout_endpoints"][0]["plan_role"], "branch")
        first, second = segment_plan["segments"]
        self.assertIn("checkpoint.save=", "\n".join(first["cmd"]))
        self.assertIn(f"checkpoint.ckpt_step={first['expected_end_step']}", "\n".join(second["cmd"]))

    def test_batch_policy_calibration_uses_plateau_gbs_without_segment_runner(self) -> None:
        from multilingual_megatron.pipeline.stages.training.calibration import run_training_calibration
        from multilingual_megatron.pipeline.stages.training.planner import build_training_run_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {"id": "run_a", "tokenizer_id": "tok_a"}
            ctx = _ctx(root, runs=[run], dry_run=True, executor=RecordingExecutor())
            common = dict(ctx.train_cfg)
            common["seq_length"] = 16
            common["train"] = {
                "global_batch_size": 32,
                "micro_batch_size": 8,
                "token_budget_tokens": 1600,
                "batch_policy": {
                    "enabled": True,
                    "target_global_batch_size": 128,
                    "schedule": [
                        {"start_tokens": 0, "end_tokens": 512, "global_batch_size": 32},
                        {"start_tokens": 512, "end_tokens": 1024, "global_batch_size": 64},
                        {"start_tokens": 1024, "end_tokens": 0, "global_batch_size": 128},
                    ],
                },
            }
            common["calibration"] = {
                "enabled": True,
                "candidate_micro_batch_sizes": [16, 32, 64],
                "probe_train_iters": 4,
                "warmup_iters": 1,
            }
            base_plan = build_training_run_plan(ctx, run=run, common=common)

            result = run_training_calibration(ctx, run_cfg=run, common=common, base_plan=base_plan)
            probe_plan = json.loads(
                (base_plan.output_dir / "calibration" / "gbs128_mbs16" / "training_command_plan.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual([row["global_batch_size"] for row in result.candidates], [128, 128, 128])
        self.assertNotIn("multilingual_megatron.pipeline.stages.training.segment_runner", probe_plan["cmd"])
        self.assertEqual(probe_plan["cmd"][probe_plan["cmd"].index("--global-batch-size") + 1], "128")
        self.assertEqual(probe_plan["cmd"][probe_plan["cmd"].index("--micro-batch-size") + 1], "16")

    def test_multiple_tokenizers_require_run_tokenizer_id(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            with self.assertRaises(SystemExit):
                build_training_cmd(
                    repo_root=REPO_ROOT,
                    env={},
                    output_dir=root / "run",
                    run_cfg={"id": "run_a"},
                    common={"backend": "megatron_bridge", "dataset": "llm-pretrain"},
                    language_map_path=root / "language_map.json",
                    tokenized_path=tokenized_dir,
                    run_name="run-a",
                )

    def test_run_data_manifest_override_wins_for_smoke_subset(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            smoke_manifest = root / "smoke_megatron_data_manifest.json"
            smoke_manifest.write_text("{}", encoding="utf-8")
            cmd, _env = build_training_cmd(
                repo_root=REPO_ROOT,
                env={},
                output_dir=root / "run",
                run_cfg={"id": "run_a", "tokenizer_id": "tok_a", "data_manifest": str(smoke_manifest)},
                common={"backend": "megatron_bridge", "dataset": "llm-pretrain"},
                language_map_path=root / "language_map.json",
                tokenized_path=tokenized_dir,
                run_name="run-a",
            )

        data_manifest = cmd[cmd.index("--data-manifest") + 1]
        self.assertEqual(data_manifest, str(smoke_manifest))

    def test_data_manifest_max_prefixes_writes_training_subset(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            manifest_path = tokenized_dir / "tok_a" / "megatron_data_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "megatron_gpt",
                        "prefixes": ["a", "b", "c"],
                        "weights": [1.0, 2.0, 3.0],
                        "prefixes_by_language": {"lang_a": ["a", "b"], "lang_b": ["c"]},
                        "documents_per_language": {"lang_a": 2, "lang_b": 1},
                        "splits": {
                            "train": {
                                "prefixes": ["a", "b", "c"],
                                "weights": [1.0, 2.0, 3.0],
                                "prefixes_by_language": {"lang_a": ["a", "b"], "lang_b": ["c"]},
                                "documents_per_language": {"lang_a": 2, "lang_b": 1},
                            },
                            "validation": {
                                "prefixes": ["va", "vb"],
                                "weights": [1.0, 2.0],
                                "prefixes_by_language": {"lang_a": ["va"], "lang_b": ["vb"]},
                                "documents_per_language": {"lang_a": 1, "lang_b": 1},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "run"
            cmd, _env = build_training_cmd(
                repo_root=REPO_ROOT,
                env={},
                output_dir=output_dir,
                run_cfg={"id": "run_a", "tokenizer_id": "tok_a", "data_manifest_max_prefixes": 1},
                common={"backend": "megatron_bridge", "dataset": "llm-pretrain"},
                language_map_path=root / "language_map.json",
                tokenized_path=tokenized_dir,
                run_name="run-a",
            )
            subset_path = Path(cmd[cmd.index("--data-manifest") + 1])
            subset = json.loads(subset_path.read_text())
            self.assertEqual(subset_path, output_dir / "megatron_data_manifest.training_subset.json")
            self.assertEqual(subset["prefixes"], ["a"])
            self.assertIsNone(subset["weights"])
            self.assertEqual(subset["languages"], ["lang_a"])
            self.assertEqual(subset["bounded_max_prefixes"], 1)
            self.assertEqual(subset["splits"]["train"]["prefixes"], ["a"])
            self.assertEqual(subset["splits"]["train"]["languages"], ["lang_a"])
            self.assertIsNone(subset["splits"]["train"]["weights"])
            self.assertEqual(subset["splits"]["validation"]["prefixes"], ["va"])
            self.assertEqual(subset["splits"]["validation"]["languages"], ["lang_a"])
            self.assertIsNone(subset["splits"]["validation"]["weights"])

    def test_ab_v2_presplit_manifest_forces_full_train_split(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            manifest_path = tokenized_dir / "tok_a" / "megatron_data_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            train = {
                "prefixes": ["train-prefix"],
                "weights": [1.0],
            }
            purposes = (
                "nested_K_training_with_global_validation",
                "language_composition_K12_training_with_global_K196_validation",
            )
            for purpose in purposes:
                with self.subTest(purpose=purpose):
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "schema": "fineweb_ab_v2_megatron_data/v1",
                                "purpose": purpose,
                                "prefixes": train["prefixes"],
                                "weights": train["weights"],
                                "splits": {
                                    "train": train,
                                    "validation": {"prefixes": ["validation-prefix"]},
                                },
                            }
                        ),
                        encoding="utf-8",
                    )

                    cmd, _env = build_training_cmd(
                        repo_root=REPO_ROOT,
                        env={},
                        output_dir=root / f"run-{purpose}",
                        run_cfg={"id": "run_a", "tokenizer_id": "tok_a"},
                        common={"backend": "megatron_bridge", "dataset": "llm-pretrain"},
                        language_map_path=root / "language_map.json",
                        tokenized_path=tokenized_dir,
                        run_name="run-a",
                    )
                    overrides = [
                        cmd[index + 1]
                        for index, value in enumerate(cmd)
                        if value == "--extra-override"
                    ]
                    self.assertIn("dataset.split=100,0,0", overrides)

            with self.assertRaisesRegex(SystemExit, "forbids an internal Bridge re-split"):
                build_training_cmd(
                    repo_root=REPO_ROOT,
                    env={},
                    output_dir=root / "bad-run",
                    run_cfg={
                        "id": "run_bad",
                        "tokenizer_id": "tok_a",
                        "overrides": ["dataset.split=9999,8,2"],
                    },
                    common={"backend": "megatron_bridge", "dataset": "llm-pretrain"},
                    language_map_path=root / "language_map.json",
                    tokenized_path=tokenized_dir,
                    run_name="run-bad",
                )
    def test_data_manifest_single_prefix_prefers_combined_partition_prefix(self) -> None:
        from multilingual_megatron.pipeline.stages.training.planner import build_training_cmd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenized_dir = root / "tokenized"
            _write_tokenized_index(tokenized_dir)
            manifest_path = tokenized_dir / "tok_a" / "megatron_data_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            partition_prefix = str(manifest_path.parent / "lang_a_0_text_document")
            combined_prefix = str(manifest_path.parent / "lang_a_text_document")
            Path(f"{combined_prefix}.bin").write_text("", encoding="utf-8")
            Path(f"{combined_prefix}.idx").write_text("", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "megatron_gpt",
                        "prefixes": [partition_prefix],
                        "weights": [1.0],
                        "prefixes_by_language": {"lang_a": [partition_prefix]},
                        "documents_per_language": {"lang_a": 2},
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "run"
            cmd, _env = build_training_cmd(
                repo_root=REPO_ROOT,
                env={},
                output_dir=output_dir,
                run_cfg={"id": "run_a", "tokenizer_id": "tok_a", "data_manifest_max_prefixes": 1},
                common={"backend": "megatron_bridge", "dataset": "llm-pretrain"},
                language_map_path=root / "language_map.json",
                tokenized_path=tokenized_dir,
                run_name="run-a",
            )
            subset = json.loads(Path(cmd[cmd.index("--data-manifest") + 1]).read_text())

            self.assertEqual(subset["prefixes"], [combined_prefix])
            self.assertEqual(subset["prefixes_by_language"], {"lang_a": [combined_prefix]})
            self.assertIsNone(subset["weights"])

    def test_stage_dry_run_batches_multiple_runs_and_writes_command_plans(self) -> None:
        from multilingual_megatron.pipeline.stages.training.train_stage import stage_train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = RecordingExecutor()
            ctx = _ctx(
                root,
                runs=[
                    {"id": "run_a", "tokenizer_id": "tok_a"},
                    {"id": "run_b", "tokenizer_id": "tok_b"},
                ],
                dry_run=True,
                executor=executor,
            )
            stage_train(ctx)

            self.assertEqual(len(executor.batch_calls), 1)
            self.assertEqual(len(executor.batch_calls[0]), 2)
            self.assertTrue((root / "runs" / "run_a" / "training_command_plan.json").exists())
            self.assertTrue((root / "runs" / "run_b" / "training_command_plan.json").exists())

    def test_stage_writes_run_and_aggregate_manifests_after_success(self) -> None:
        from multilingual_megatron.pipeline.stages.training.train_stage import stage_train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = RecordingExecutor(complete=True)
            ctx = _ctx(
                root,
                runs=[{"id": "run_a", "tokenizer_id": "tok_a"}],
                dry_run=False,
                executor=executor,
            )
            stage_train(ctx)

            run_manifest = json.loads((root / "runs" / "run_a" / "training_run_manifest.json").read_text())
            aggregate = json.loads((root / "runs" / "training_manifest.json").read_text())
            self.assertEqual(run_manifest["status"], "complete")
            self.assertEqual(run_manifest["metadata"]["expected_tokens"], 3 * 8 * 128)
            self.assertEqual(aggregate["runs"]["run_a"]["status"], "complete")
            self.assertTrue((root / "runs" / "_check.json").exists())

    def test_stage_calibrates_then_trains_with_selected_micro_batch(self) -> None:
        from multilingual_megatron.pipeline.stages.training.train_stage import stage_train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = RecordingExecutor(
                complete=True,
                metrics_by_mbs={
                    2: {"tokens_per_second": 1000.0, "allocated_gb": 20.0, "reserved_gb": 24.0},
                    4: {"tokens_per_second": 1100.0, "allocated_gb": 28.0, "reserved_gb": 32.0},
                },
            )
            ctx = _ctx(
                root,
                runs=[{"id": "run_a", "tokenizer_id": "tok_a"}],
                dry_run=False,
                executor=executor,
            )
            ctx.train_cfg["calibration"] = {
                "enabled": True,
                "mode": "calibrate_then_train",
                "candidate_batch_pairs": [
                    {"global_batch_size": 8, "micro_batch_size": 2},
                    {"global_batch_size": 12, "micro_batch_size": 4},
                ],
                "probe_train_iters": 4,
                "warmup_iters": 1,
                "gpu_memory_total_gb": 80,
            }
            stage_train(ctx)

            final_calls = [
                cmd
                for cmd, _env, _dry_run, _log_path in executor.calls
                if Path(cmd[cmd.index("--output-dir") + 1]) == root / "runs" / "run_a"
            ]
            self.assertEqual(len(final_calls), 1)
            self.assertEqual(final_calls[0][final_calls[0].index("--micro-batch-size") + 1], "4")
            self.assertEqual(final_calls[0][final_calls[0].index("--global-batch-size") + 1], "12")

            calibration = json.loads((root / "runs" / "run_a" / "training_calibration.json").read_text())
            run_manifest = json.loads((root / "runs" / "run_a" / "training_run_manifest.json").read_text())
            aggregate = json.loads((root / "runs" / "training_calibrations.json").read_text())
            self.assertEqual(calibration["selected_candidate"]["micro_batch_size"], 4)
            self.assertEqual(calibration["selected_candidate"]["global_batch_size"], 12)
            self.assertEqual(run_manifest["metadata"]["micro_batch_size"], 4)
            self.assertEqual(run_manifest["metadata"]["global_batch_size"], 12)
            self.assertEqual(aggregate["runs"]["run_a"]["selected_candidate"]["micro_batch_size"], 4)

    def test_bridge_wrapper_failure_artifacts_do_not_mark_complete(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _write_wrapper_artifacts,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                recipe="test_recipe",
                dataset="llm-pretrain",
                train_iters=2,
                global_batch_size=4,
                seq_length=128,
            )
            _write_wrapper_artifacts(
                output_dir=root,
                args=args,
                status="failed",
                runtime=0.5,
                error="boom",
                return_code=7,
            )
            results = json.loads((root / "train_results.json").read_text())

            self.assertEqual(results["status"], "failed")
            self.assertEqual(results["return_code"], 7)
            self.assertEqual(results["num_input_tokens_seen"], 2 * 4 * 128)
            self.assertFalse((root / ".training_complete").exists())

    def test_bridge_pybind11_shim_adds_python3_config(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _prepare_pybind11_shim,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {"PROOF_ENABLE_PYBIND11_SHIM": "1", "PATH": "/usr/bin"}
            shim_root = _prepare_pybind11_shim(root, env)

            self.assertEqual(shim_root, root / "runtime_shims")
            self.assertTrue((root / "runtime_shims" / "pybind11" / "__main__.py").exists())
            python3 = root / "runtime_shims" / "bin" / "python3"
            self.assertTrue(python3.exists())
            self.assertTrue(python3.stat().st_mode & 0o111)
            python_config = root / "runtime_shims" / "bin" / "python3-config"
            self.assertTrue(python_config.exists())
            self.assertTrue(env["PATH"].startswith(str(root / "runtime_shims" / "bin")))

    def test_bridge_blend_value_uses_single_prefix_fast_path(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _blend_value,
        )

        self.assertEqual(_blend_value({"prefixes": ["a"], "weights": [1.0]}), [["a"], None])

    def test_bridge_physical_split_blend_skips_zero_validation_language(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _dataset_overrides_for_manifest,
            _manifest_blend_per_split,
            _write_validation_coverage,
        )

        manifest = {
            "format": "megatron_gpt",
            "languages": ["eng_Latn", "zsm_Arab"],
            "prefixes": ["train/eng_0", "train/zsm_0"],
            "weights": [100.0, 3.0],
            "documents_per_language": {"eng_Latn": 100, "zsm_Arab": 3},
            "prefixes_by_language": {
                "eng_Latn": ["train/eng_0"],
                "zsm_Arab": ["train/zsm_0"],
            },
            "splits": {
                "train": {
                    "prefixes": ["train/eng_0", "train/zsm_0"],
                    "weights": [100.0, 3.0],
                    "documents_per_language": {"eng_Latn": 100, "zsm_Arab": 3},
                    "prefixes_by_language": {
                        "eng_Latn": ["train/eng_0"],
                        "zsm_Arab": ["train/zsm_0"],
                    },
                },
                "validation": {
                    "prefixes": ["validation/eng_0"],
                    "weights": [10.0],
                    "documents_per_language": {"eng_Latn": 10},
                    "prefixes_by_language": {"eng_Latn": ["validation/eng_0"]},
                },
            },
        }

        blend_per_split = _manifest_blend_per_split(
            manifest,
            include_validation=True,
            include_test=False,
        )

        self.assertIsNotNone(blend_per_split)
        assert blend_per_split is not None
        self.assertEqual(blend_per_split[0], [["train/eng_0", "train/zsm_0"], [100.0, 3.0]])
        self.assertEqual(blend_per_split[1], [["validation/eng_0"], None])
        self.assertIsNone(blend_per_split[2])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coverage_path = _write_validation_coverage(
                root,
                root / "megatron_data_manifest.json",
                manifest,
            )
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

        overrides, uses_physical_splits = _dataset_overrides_for_manifest(
            Path("megatron_data_manifest.json"),
            manifest,
            Path("/tmp/run"),
            eval_iters=1,
        )

        self.assertEqual(coverage["languages"]["zsm_Arab"]["train_docs"], 3)
        self.assertEqual(coverage["languages"]["zsm_Arab"]["validation_docs"], 0)
        self.assertEqual(coverage["skipped_validation_languages"], ["zsm_Arab"])
        self.assertEqual(coverage["train_prefixes"], 2)
        self.assertEqual(coverage["validation_prefixes"], 1)
        self.assertTrue(uses_physical_splits)
        self.assertIsNone(overrides["dataset.blend"])
        self.assertEqual(overrides["dataset.blend_per_split"], blend_per_split)
        self.assertIsNone(overrides["dataset.split"])
        self.assertEqual(overrides["dataset.path_to_cache"], "/tmp/run/dataset_cache")


if __name__ == "__main__":
    unittest.main()
