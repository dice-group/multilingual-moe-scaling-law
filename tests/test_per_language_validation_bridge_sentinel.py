"""Opt-in real-GPU sentinel for the production per-language Bridge path.

Run inside a one-H100 Slurm allocation after producing a tiny one-language
checkpoint/spec fixture:

    AB_V2_BRIDGE_SENTINEL_SPEC=/abs/eval_spec.json \
    PYTHONPATH=src:. python -m unittest \
      tests.test_per_language_validation_bridge_sentinel

The normal CPU suite skips this test; the A/B-v2 launch gate schedules it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = os.environ.get("AB_V2_BRIDGE_SENTINEL_SPEC", "")
RECIPE = os.environ.get("AB_V2_BRIDGE_SENTINEL_RECIPE", "")
BRIDGE_ROOT = os.environ.get(
    "AB_V2_BRIDGE_SENTINEL_BRIDGE_ROOT",
    str(REPO_ROOT / "external" / "Megatron-Bridge"),
)


@unittest.skipUnless(SPEC, "real Bridge sentinel fixture was not requested")
class PerLanguageValidationBridgeSentinelTest(unittest.TestCase):
    def test_real_checkpoint_load_forward_metrics_and_routing_artifacts(self) -> None:
        command = [
            sys.executable,
            "-m",
            "multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_per_language_validation",
            "--bridge-root",
            BRIDGE_ROOT,
            "--spec",
            SPEC,
            "--python-executable",
            sys.executable,
            "--preflight-cuda",
        ]
        if RECIPE:
            command.extend(["--recipe", RECIPE])
        subprocess.run(command, cwd=REPO_ROOT, check=True)

        spec = json.loads(Path(SPEC).read_text(encoding="utf-8"))
        output = Path(spec["output_dir"])
        if not output.is_absolute():
            output = Path(SPEC).resolve().parent / output
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["coverage"]["missing_required_languages"], [])
        for language in manifest["coverage"]["required_languages"]:
            artifact = json.loads(Path(manifest["languages"][language]["path"]).read_text(encoding="utf-8"))
            self.assertGreater(artifact["token_count"], 0)
            if manifest["moe_routing_diagnostics"]["model_is_moe"]:
                self.assertTrue(artifact["moe_routing"]["evidence_available"])
                self.assertIn("dropped_assignments", artifact["moe_routing"]["totals"])


if __name__ == "__main__":
    unittest.main()
