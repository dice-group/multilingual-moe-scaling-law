from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.slurm import discover_slurm_attempts


class NestedReceiptDiscoveryTest(unittest.TestCase):
    def test_discovers_attempts_nested_in_lists_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            receipts = root / "control" / "receipts"
            receipts.mkdir(parents=True)
            payload = {
                "jobs": {
                    "prod_branch_d_e32_k200_d32b": {
                        "attempts": [
                            {"slurm_job_id": "301", "state": "FAILED"},
                            {
                                "nested": [
                                    {
                                        "slurm_job_id": "302",
                                        "state": "COMPLETED",
                                    }
                                ]
                            },
                        ]
                    }
                }
            }
            (receipts / "nested.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            attempts = discover_slurm_attempts([root])

        self.assertEqual(
            {(row["logical_job_id"], row["slurm_job_id"]) for row in attempts},
            {
                ("prod_branch_d_e32_k200_d32b", "301"),
                ("prod_branch_d_e32_k200_d32b", "302"),
            },
        )
        details = {
            source["detail"] for row in attempts for source in row["discovery_sources"]
        }
        self.assertIn(
            "/jobs/prod_branch_d_e32_k200_d32b/attempts/0/slurm_job_id",
            details,
        )
        self.assertIn(
            ("/jobs/prod_branch_d_e32_k200_d32b/attempts/1/nested/0/slurm_job_id"),
            details,
        )

    def test_recovery_receipt_prefers_explicit_canonical_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            receipts = root / "control" / "recovery"
            receipts.mkdir(parents=True)
            payload = {
                "failure": {
                    "canonical_job_id": "prod_trunk_d_e16_k200",
                    "slurm_job_id": "2054109",
                },
                "retry": {
                    "canonical_job_id": "prod_trunk_d_e16_k200",
                    "slurm_job_id": "2064449",
                },
            }
            (receipts / "recovery_receipt.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            attempts = discover_slurm_attempts([root])

        self.assertEqual(
            {(row["logical_job_id"], row["slurm_job_id"]) for row in attempts},
            {
                ("prod_trunk_d_e16_k200", "2054109"),
                ("prod_trunk_d_e16_k200", "2064449"),
            },
        )


if __name__ == "__main__":
    unittest.main()
