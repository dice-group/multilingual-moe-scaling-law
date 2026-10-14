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
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str], bool, Path]] = []

    def run(self, cmd: list[str], env: dict[str, str], dry_run: bool, log_path: Path) -> None:
        self.calls.append((list(cmd), dict(env), dry_run, Path(log_path)))
        if not dry_run:
            output_root = Path(cmd[cmd.index("--output-root") + 1])
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "manifest.json").write_text(json.dumps({"rows": []}), encoding="utf-8")

    def run_batch(self, jobs, dry_run: bool) -> None:
        for cmd, env, log_path in jobs:
            self.run(cmd, env, dry_run, log_path)


class BaselineEvalPipelineTest(unittest.TestCase):
    def test_baseline_eval_builds_cache_command_and_records_stage_state(self) -> None:
        from multilingual_megatron.pipeline.stages.benchmarking.baseline_eval_stage import stage_baseline_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_file = root / "tasks.txt"
            tasks_file.write_text("belebele_arb_Arab\n", encoding="utf-8")
            executor = RecordingExecutor()
            ctx = SimpleNamespace(
                active_stages=["baseline_eval"],
                args=SimpleNamespace(dry_run=False),
                baseline_cache_root=root / "baselines",
                baseline_eval_cfg={
                    "enabled": True,
                    "models": ["hf-test/model-a"],
                    "tasks_file": str(tasks_file),
                    "batch_size": "1",
                    "lm_eval_bin": "lm_eval",
                    "seed": 7,
                    "task_prefixes": ["belebele_"],
                },
                env={},
                executor=executor,
                force_stages=set(),
                logger=logging.getLogger("test"),
                logs_root=root / "logs",
                repo_root=REPO_ROOT,
                state_path=root / "state" / "pipeline_state.json",
            )

            stage_baseline_eval(ctx)

            self.assertEqual(len(executor.calls), 1)
            cmd = executor.calls[0][0]
            self.assertIn("--models", cmd)
            self.assertIn("hf-test/model-a", cmd)
            self.assertIn("--task-prefix", cmd)
            self.assertTrue((root / "baselines" / "manifest.json").exists())
            self.assertTrue((root / "state" / "pipeline_state.json").exists())


if __name__ == "__main__":
    unittest.main()
