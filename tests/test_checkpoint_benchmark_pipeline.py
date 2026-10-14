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


def _write_completed_training(train_root: Path, run_id: str) -> None:
    run_dir = train_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_results.json").write_text(
        json.dumps(
            {
                "backend": "megatron_bridge",
                "recipe": "test_recipe",
                "dataset": "llm-pretrain",
                "status": "complete",
                "train_runtime": 1.0,
                "train_loss": None,
                "num_input_tokens_seen": 1024,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"backend": "megatron_bridge", "status": "complete", "global_step": 10}),
        encoding="utf-8",
    )
    (run_dir / ".training_complete").write_text("ok\n", encoding="utf-8")
    (run_dir / "training_run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "complete", "run_id": run_id}),
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoints" / "iter_0000010"
    checkpoint.mkdir(parents=True, exist_ok=True)


def _ctx(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        active_stages=["checkpoint_conversion", "benchmark"],
        args=SimpleNamespace(dry_run=False, run_id=[]),
        benchmark_cfg={"enabled": True, "backend": "bridge_artifacts"},
        benchmark_root=root / "benchmarks",
        env={},
        executor=None,
        force_stages=set(),
        logger=logging.getLogger("test"),
        logs_root=root / "logs",
        output_root=root / "out",
        profile={
            "checkpoint_conversion": {
                "enabled": True,
                "output_dir": str(root / "exports"),
                "mode": "manifest",
            }
        },
        repo_root=REPO_ROOT,
        sampled_dir=root / "sampled",
        sampled_manifest=root / "sampled" / "sample_manifest.json",
        sample_cfg={},
        state_path=root / "state" / "pipeline_state.json",
        tokenize_cfg={},
        tokenized_dir=root / "tokenized",
        train_cfg={
            "runs": [
                {"id": "run_a", "kind": "megatron_bridge", "backend": "megatron_bridge", "recipe": "test_recipe"}
            ]
        },
        train_root=root / "runs",
    )


class RecordingExecutor:
    def __init__(self, *, write_hf_export: bool = False, write_lm_eval_metrics: bool = False) -> None:
        self.write_hf_export = write_hf_export
        self.write_lm_eval_metrics = write_lm_eval_metrics
        self.calls: list[tuple[list[str], dict[str, str], bool, Path]] = []

    def run(self, cmd: list[str], env: dict[str, str], dry_run: bool, log_path: Path) -> None:
        self.calls.append((list(cmd), dict(env), dry_run, Path(log_path)))
        if dry_run:
            return
        if self.write_hf_export and "--hf-path" in cmd:
            hf_path = Path(cmd[cmd.index("--hf-path") + 1])
            hf_path.mkdir(parents=True, exist_ok=True)
            (hf_path / "config.json").write_text("{}", encoding="utf-8")
        if self.write_lm_eval_metrics and "--output-dir" in cmd:
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "backend": "bridge_hf_lm_eval",
                        "run_id": "run_a",
                        "num_tasks_expected": 1,
                        "num_tasks_seen": 1,
                        "missing_tasks": [],
                        "metrics": [{"task": "belebele_arb_Arab", "metric": "acc,none", "value": 0.0}],
                    }
                ),
                encoding="utf-8",
            )

    def run_batch(self, jobs, dry_run: bool) -> None:
        for cmd, env, log_path in jobs:
            self.run(cmd, env, dry_run, log_path)


class CheckpointBenchmarkPipelineTest(unittest.TestCase):
    def test_checkpoint_conversion_manifest_mode_writes_export_index(self) -> None:
        from multilingual_megatron.pipeline.stages.checkpoint_conversion.checkpoint_conversion_stage import stage_checkpoint_conversion

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_completed_training(root / "runs", "run_a")
            ctx = _ctx(root)

            stage_checkpoint_conversion(ctx)

            run_manifest = json.loads((root / "exports" / "run_a" / "checkpoint_export_manifest.json").read_text())
            index = json.loads((root / "exports" / "checkpoint_exports.json").read_text())
            self.assertEqual(run_manifest["status"], "checkpoint_exported")
            self.assertTrue(run_manifest["latest_checkpoint_exists"])
            self.assertTrue(run_manifest["latest_checkpoint"].endswith("iter_0000010"))
            self.assertEqual(index["runs"]["run_a"]["status"], "checkpoint_exported")

    def test_checkpoint_conversion_manifest_mode_handles_missing_checkpoint_dir(self) -> None:
        from multilingual_megatron.pipeline.stages.checkpoint_conversion.checkpoint_conversion_stage import stage_checkpoint_conversion

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_completed_training(root / "runs", "run_a")
            for path in (root / "runs" / "run_a" / "checkpoints").rglob("*"):
                path.rmdir()
            (root / "runs" / "run_a" / "checkpoints").rmdir()
            ctx = _ctx(root)

            stage_checkpoint_conversion(ctx)

            run_manifest = json.loads((root / "exports" / "run_a" / "checkpoint_export_manifest.json").read_text())
            index = json.loads((root / "exports" / "checkpoint_exports.json").read_text())
            self.assertEqual(run_manifest["status"], "metadata_exported")
            self.assertFalse(run_manifest["checkpoint_dir_exists"])
            self.assertIsNone(run_manifest["latest_checkpoint"])
            self.assertFalse(index["runs"]["run_a"]["latest_checkpoint_exists"])

    def test_checkpoint_conversion_bridge_hf_export_runs_conversion_and_records_hf_dir(self) -> None:
        from multilingual_megatron.pipeline.stages.checkpoint_conversion.checkpoint_conversion_stage import stage_checkpoint_conversion

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_completed_training(root / "runs", "run_a")
            executor = RecordingExecutor(write_hf_export=True)
            ctx = _ctx(root)
            ctx.executor = executor
            ctx.profile["checkpoint_conversion"] = {
                "enabled": True,
                "output_dir": str(root / "exports"),
                "mode": "bridge_hf_export",
                "hf_reference_model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "strict": False,
            }

            stage_checkpoint_conversion(ctx)

            self.assertEqual(len(executor.calls), 1)
            cmd = executor.calls[0][0]
            self.assertIn("export", cmd)
            self.assertIn("--not-strict", cmd)
            run_manifest = json.loads((root / "exports" / "run_a" / "checkpoint_export_manifest.json").read_text())
            index = json.loads((root / "exports" / "checkpoint_exports.json").read_text())
            self.assertEqual(run_manifest["status"], "hf_exported")
            self.assertTrue(run_manifest["hf_export_exists"])
            self.assertEqual(index["runs"]["run_a"]["status"], "hf_exported")

    def test_benchmark_bridge_artifact_backend_writes_metrics_and_summary(self) -> None:
        from multilingual_megatron.pipeline.stages.benchmarking.benchmark_stage import stage_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_completed_training(root / "runs", "run_a")
            ctx = _ctx(root)
            ctx.active_stages = ["benchmark"]

            stage_benchmark(ctx)

            metrics = json.loads((root / "benchmarks" / "run_a" / "metrics.json").read_text())
            summary = json.loads((root / "benchmarks" / "benchmark_summary.json").read_text())
            self.assertTrue(metrics["training_complete"])
            self.assertEqual(metrics["global_step"], 10)
            self.assertEqual(summary["backend"], "bridge_artifacts")
            self.assertEqual(summary["runs"][0]["num_tasks_seen"], 1)

    def test_benchmark_bridge_hf_lm_eval_backend_consumes_export_manifest(self) -> None:
        from multilingual_megatron.pipeline.stages.benchmarking.benchmark_stage import stage_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_completed_training(root / "runs", "run_a")
            hf_export = root / "exports" / "run_a" / "hf"
            hf_export.mkdir(parents=True, exist_ok=True)
            (hf_export / "config.json").write_text("{}", encoding="utf-8")
            run_manifest = root / "exports" / "run_a" / "checkpoint_export_manifest.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "checkpoint_conversion",
                        "run_id": "run_a",
                        "status": "hf_exported",
                        "hf_export_dir": str(hf_export),
                    }
                ),
                encoding="utf-8",
            )
            (root / "exports" / "checkpoint_exports.json").write_text(
                json.dumps({"runs": {"run_a": {"export_manifest": str(run_manifest), "status": "hf_exported"}}}),
                encoding="utf-8",
            )
            tasks_file = root / "tasks.txt"
            tasks_file.write_text("belebele_arb_Arab\n", encoding="utf-8")
            ctx = _ctx(root)
            ctx.active_stages = ["benchmark"]
            ctx.executor = RecordingExecutor(write_lm_eval_metrics=True)
            ctx.profile["checkpoint_conversion"] = {"output_dir": str(root / "exports")}
            ctx.benchmark_cfg = {
                "enabled": True,
                "backend": "bridge_hf_lm_eval",
                "tasks_file": str(tasks_file),
                "tasks": ["belebele_arb_Arab"],
                "output_dir": str(root / "benchmarks"),
                "eval_limit": "1",
            }

            stage_benchmark(ctx)

            summary = json.loads((root / "benchmarks" / "benchmark_summary.json").read_text())
            self.assertEqual(summary["backend"], "bridge_hf_lm_eval")
            self.assertEqual(summary["runs"][0]["num_tasks_seen"], 1)
            self.assertTrue((root / "benchmarks" / "selected_tasks.txt").exists())


if __name__ == "__main__":
    unittest.main()
