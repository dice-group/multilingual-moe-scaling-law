from __future__ import annotations

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


def _write_training_artifacts(root: Path, run_id: str) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_results.json").write_text(
        json.dumps(
            {
                "backend": "megatron_bridge",
                "dataset": "llm-pretrain",
                "recipe": "test_recipe",
                "status": "complete",
                "train_runtime": 2.0,
                "train_loss": None,
                "num_input_tokens_seen": 1024,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"backend": "megatron_bridge", "status": "complete", "global_step": 2}),
        encoding="utf-8",
    )
    (run_dir / ".training_complete").write_text("ok\n", encoding="utf-8")
    return run_dir


def _ctx(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        active_stages=["final_loss", "summary"],
        args=SimpleNamespace(dry_run=False, run_id=[], profile="test_profile"),
        baseline_cache_root=root / "baselines",
        benchmark_cfg={"enabled": False},
        benchmark_root=root / "benchmarks",
        env={},
        executor=None,
        force_stages=set(),
        gate_path=root / "gate.json",
        lineage_manifest_path=root / "lineage" / "run_lineage.json",
        logger=logging.getLogger("test"),
        logs_root=root / "logs",
        output_root=root / "out",
        profile={
            "final_loss": {
                "enabled": True,
                "output_dir": str(root / "final_loss"),
            }
        },
        repo_root=REPO_ROOT,
        state_path=root / "state" / "pipeline_state.json",
        summary_cfg={
            "summary_csv": str(root / "summary.csv"),
            "gate_json": str(root / "gate.json"),
            "baseline_id": "run_a",
            "primary_metric": "final_loss",
            "min_relative_improvement": 0.0,
        },
        summary_path=root / "summary.csv",
        tokenize_cfg={},
        tokenized_dir=root / "tokenized",
        train_cfg={"runs": [{"id": "run_a", "kind": "megatron_bridge"}]},
        train_root=root / "runs",
    )


class FinalLossSummaryPipelineTest(unittest.TestCase):
    def test_final_loss_extracts_latest_megatron_log_metric_and_injects_train_results(self) -> None:
        from multilingual_megatron.pipeline.stages.final_loss.final_loss_stage import stage_final_loss

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_training_artifacts(root, "run_a")
            log_dir = root / "logs" / "train_runs" / "run_a"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "run_a.train.log").write_text(
                "\n".join(
                    [
                        "iteration        1/       2 | consumed samples:            4 | lm loss: 1.039547E+01 | number of skipped iterations:   0 | number of nan iterations:   0 |",
                        "iteration        2/       2 | consumed samples:            8 | lm loss: 1.042089E+01 | number of skipped iterations:   0 | number of nan iterations:   0 |",
                    ]
                ),
                encoding="utf-8",
            )
            ctx = _ctx(root)

            stage_final_loss(ctx)

            manifest = json.loads((root / "final_loss" / "run_a" / "final_loss.json").read_text())
            train_results = json.loads((root / "runs" / "run_a" / "train_results.json").read_text())
            self.assertAlmostEqual(manifest["final_loss"], 10.42089)
            self.assertEqual(manifest["global_step"], 2)
            self.assertEqual(manifest["skipped_iterations"], 0)
            self.assertAlmostEqual(train_results["train_loss"], 10.42089)

    def test_summary_uses_final_loss_artifact_and_writes_gate(self) -> None:
        from multilingual_megatron.pipeline.stages.final_loss.final_loss_stage import stage_final_loss
        from multilingual_megatron.pipeline.stages.analysis_report.summary_stage import stage_summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_training_artifacts(root, "run_a")
            log_dir = root / "logs" / "train_runs" / "run_a"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "run_a.train.log").write_text(
                "iteration        2/       2 | consumed samples:            8 | lm loss: 9.500000E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |\n",
                encoding="utf-8",
            )
            ctx = _ctx(root)

            stage_final_loss(ctx)
            stage_summary(ctx)

            with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            gate = json.loads((root / "gate.json").read_text())
            self.assertEqual(rows[0]["run_id"], "run_a")
            self.assertEqual(rows[0]["final_loss"], "9.5")
            self.assertEqual(gate["primary_metric"], "final_loss")
            self.assertEqual(gate["baseline_value"], 9.5)

    def test_final_loss_finds_training_logs_from_previous_attempt(self) -> None:
        from multilingual_megatron.pipeline.stages.final_loss.final_loss_stage import stage_final_loss

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_training_artifacts(root, "run_a")
            logs_base = root / "logs"
            previous_log_dir = logs_base / "attempts" / "attempt-train" / "stages" / "train" / "run_a"
            previous_log_dir.mkdir(parents=True, exist_ok=True)
            (previous_log_dir / "train.log").write_text(
                "iteration        2/       2 | consumed samples:            8 | lm loss: 8.750000E+00 | number of skipped iterations:   0 | number of nan iterations:   0 |\n",
                encoding="utf-8",
            )
            ctx = _ctx(root)
            ctx.logs_root = logs_base / "attempts" / "attempt-final-loss"
            ctx.env = {
                "PROOF_PIPELINE_STRUCTURED_LOGS": "1",
                "PROOF_PIPELINE_LOGS_BASE_ROOT": str(logs_base),
                "PROOF_PIPELINE_ATTEMPT_LOGS_ROOT": str(ctx.logs_root),
            }

            stage_final_loss(ctx)

            manifest = json.loads((root / "final_loss" / "run_a" / "final_loss.json").read_text())
            self.assertAlmostEqual(manifest["final_loss"], 8.75)
            self.assertEqual(manifest["source"], "log")


if __name__ == "__main__":
    unittest.main()
