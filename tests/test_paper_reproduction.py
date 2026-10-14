"""Reviewer workflow checks: immutable inputs, attribution and CPU-only execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from result_analysis.evidence import Evidence, sha256
from result_analysis.reproduce import validate_inputs
from multilingual_megatron.devtools.anonymous_provenance import anonymous_source_id


class PaperReproductionTest(unittest.TestCase):
    def test_verification_rejects_corruption_and_wrong_law_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.csv").write_text("language,loss\nx,1.5\n")
            analysis = {"version": "stage1-paper-v2", "main_law": "RN14-stored-reuse",
                        "allocation_predictor": "M6-ND"}
            revision = "a" * 40
            manifest = {"schema": "stage1-paper-inputs/v1", "paper_commit": revision,
                        "analysis": analysis, "files": {"data.csv": {"sha256": sha256(root / "data.csv")}}}
            contract = {"schema": "stage1-paper-reproduction/v1", "paper_commit": anonymous_source_id(revision),
                        "analysis": dict(analysis), "input_files": 1}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(validate_inputs(Evidence(root), contract)["verified_files"], 1)
            with self.assertRaisesRegex(ValueError, "revision differs"):
                validate_inputs(Evidence(root), {**contract, "paper_commit": "b" * 40})
            manifest["analysis"]["allocation_predictor"] = "RN14-stored-reuse"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "attribution"):
                validate_inputs(Evidence(root), contract)
            manifest["analysis"] = contract["analysis"]
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "data.csv").write_text("language,loss\ny,1.5\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_inputs(Evidence(root), contract)

    def test_all_replay_imports_work_without_the_training_stack(self):
        script = """
import importlib, sys
for name in ('torch', 'transformers', 'megatron', 'megatron.bridge'):
    sys.modules[name] = None
for topic in ('capacity_response', 'scaling_law_fit', 'compute_allocation',
              'language_preservation', 'downstream_evaluations', 'routing_analysis'):
    importlib.import_module('result_analysis.' + topic + '.reproduce')
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)

    @unittest.skipUnless(os.environ.get("MMS_PAPER_INPUTS"), "requires the released paper input bundle")
    def test_real_input_smoke_and_create_only_outputs(self):
        from result_analysis.reproduce import run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            summary = run(Path(os.environ["MMS_PAPER_INPUTS"]), output)
            self.assertEqual(summary["checks_passed"], 30)
            self.assertEqual(summary["analysis"]["main_law"], "RN14-stored-reuse")
            self.assertEqual(summary["analysis"]["allocation_predictor"], "M6-ND")
            with self.assertRaises(FileExistsError):
                run(Path(os.environ["MMS_PAPER_INPUTS"]), output)


if __name__ == "__main__":
    unittest.main()
