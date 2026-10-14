from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.compute import build_physical_compute_category_summary
from multilingual_megatron.experiments.scaling.stage1_release.compute import classify_compute_records


class ReleaseComputeTest(unittest.TestCase):
    def test_classifies_and_deduplicates_physical_records(self) -> None:
        root = "/study/outputs/pipeline_runs/primary"
        recovery = "/study/outputs/pipeline_runs/recovery"
        attempts = [
            {
                "attempt_uid": "a1",
                "logical_job_id": "prod_branch_a_e1_k001_d1",
                "run_root": root,
                "accounting_status": "observed",
            },
            {
                "attempt_uid": "a2",
                "logical_job_id": "prod_eval_a_e1_k001_d1",
                "run_root": root,
                "accounting_status": "observed",
            },
            {
                "attempt_uid": "a3",
                "logical_job_id": "prod_eval_a_e1_k001_d1",
                "run_root": recovery,
                "accounting_status": "missing",
            },
            {
                "attempt_uid": "a4",
                "logical_job_id": "targets",
                "run_root": recovery,
                "slurm_job_id": "1",
                "accounting_status": "observed",
            },
        ]
        allocations = [
            {
                "allocation_uid": "a1:1",
                "attempt_uid": "a1",
                "slurm_record_id": "1",
                "slurm_job_id": "1",
                "array_task_id": None,
                "state": "COMPLETED",
                "elapsed_seconds": 3600,
                "allocated_cpus": 8,
                "allocated_gpus": 1,
                "allocated_gpu_hours": 1.0,
                "allocated_gpu_type_counts": {"h100": 1},
            },
            {
                "allocation_uid": "a2:2",
                "attempt_uid": "a2",
                "slurm_record_id": "2",
                "slurm_job_id": "2",
                "array_task_id": None,
                "state": "FAILED",
                "elapsed_seconds": 1800,
                "allocated_cpus": 4,
                "allocated_gpus": 1,
                "allocated_gpu_hours": 0.5,
                "allocated_gpu_type_counts": {"h100": 1},
            },
        ]
        candidates = [
            {
                "readout_id": "prod_eval_a_e1_k001_d1",
                "selected": True,
                "candidate_manifest_path": (
                    root + "/jobs/prod_eval_a_e1_k001_d1/evaluation_manifest.json"
                ),
            }
        ]

        classified_attempts, classified_allocations = classify_compute_records(
            attempts=attempts,
            allocations=allocations,
            validation_candidates=candidates,
        )
        summary = build_physical_compute_category_summary(
            attempts=classified_attempts,
            allocations=[
                *classified_allocations,
                {
                    **classified_allocations[0],
                    "attempt_uid": "a4",
                    "allocation_uid": "a4:1",
                    "workload_category": "other",
                },
            ],
        )

        categories = {
            row["attempt_uid"]: row["workload_category"] for row in classified_attempts
        }
        self.assertEqual(categories["a1"], "core_training")
        self.assertEqual(categories["a2"], "selected_validation")
        self.assertEqual(categories["a3"], "duplicate_or_recovery_validation")
        self.assertEqual(categories["a4"], "other")
        total = next(
            row for row in summary if row["workload_category"] == "all_physical_records"
        )
        self.assertEqual(total["unique_physical_allocation_records"], 2)
        self.assertEqual(total["duplicate_allocation_evidence_mappings"], 1)
        self.assertEqual(total["allocated_gpu_hours_all_states"], 1.5)
        self.assertEqual(total["allocated_gpu_hours_completed"], 1.0)
        self.assertEqual(total["allocated_gpu_hours_noncompleted"], 0.5)


if __name__ == "__main__":
    unittest.main()
