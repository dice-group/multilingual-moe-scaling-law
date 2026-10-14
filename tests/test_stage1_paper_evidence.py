"""Standalone immutable evidence and numerical-check contracts for paper analysis."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np

from result_analysis.checks import Checks
from result_analysis.evidence import (Evidence, finish, inside, output_directory, prepare, read_result, sha256)


class EvidenceTests(unittest.TestCase):
    def test_portable_copy_hash_verification_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "observation.csv").write_text("x\n1\n")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"schema": "stage1-paper-inputs/v1", "paper_commit": "fixture",
                "roots": {"observed": str(source)}, "files": {"data.csv": {
                "root": "observed", "path": "observation.csv", "sha256": sha256(source / "observation.csv")}}}))
            bundle = root / "bundle"
            prepare(profile, bundle)
            with self.assertRaises(FileExistsError):
                prepare(profile, bundle)
            moved = root / "relocated"
            bundle.rename(moved)
            reader = Evidence(moved)
            self.assertEqual(reader.table("data.csv").x.tolist(), [1])
            self.assertEqual(reader.verify()["verified_files"], 1)
            with self.assertRaises(ValueError):
                output_directory(moved / "outputs", reader)
            (moved / "data.csv").write_text("x\n2\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reader.table("data.csv")

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../escape", "/tmp/absolute", "."):
                with self.assertRaises(ValueError):
                    inside(Path(directory), name)

    def test_bad_source_does_not_create_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x.csv").write_text("x\n1\n")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"schema": "stage1-paper-inputs/v1", "paper_commit": "fixture",
                "roots": {"root": str(root)}, "files": {"x.csv": {
                "root": "root", "path": "x.csv", "sha256": "wrong"}}}))
            with self.assertRaisesRegex(ValueError, "source hash"):
                prepare(profile, root / "output")
            self.assertFalse((root / "output").exists())

    def test_evidence_cli_has_no_combined_analysis_runner(self):
        result = subprocess.run([sys.executable, "-m", "result_analysis.evidence", "--help"],
                                check=True, text=True, capture_output=True)
        self.assertIn("{prepare,verify}", result.stdout)
        self.assertNotIn("run}", result.stdout)


    def test_portable_contract_and_output_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text("value\n1\n")
            contract = {"version": "stage1-paper-v2", "main_law": "RN14-stored-reuse"}
            profile = root / "profile.json"
            profile.write_text(json.dumps({"schema": "stage1-paper-inputs/v1", "paper_commit": "fixture",
                "analysis": contract, "roots": {"test": str(root)}, "files": {"data.csv": {
                    "root": "test", "path": "source.csv", "sha256": sha256(source)}}}))
            prepare(profile, root / "inputs")
            evidence = Evidence(root / "inputs")
            self.assertEqual(evidence.require_current(), contract)
            evidence.table("data.csv")
            output = root / "result"
            output.mkdir()
            finish(output, evidence, "fixture", {"value": 1}, [Path(__file__)])
            self.assertEqual(read_result(output, evidence), {"value": 1})
            (output / "claims.json").write_text('{"value":2}')
            with self.assertRaisesRegex(ValueError, "changed derived output"):
                read_result(output, evidence)


    def test_named_checks_fail_closed(self):
        checks = Checks()
        checks.close("rounded", .12345, .123451, atol=5e-6)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            checks.close("different", 1., 2.)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            checks.close("nan", np.nan, np.nan)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            checks.close("shape", [1, 2], [1])
        with self.assertRaisesRegex(ValueError, "population"):
            checks.require("members", False)


if __name__ == "__main__":
    unittest.main()
