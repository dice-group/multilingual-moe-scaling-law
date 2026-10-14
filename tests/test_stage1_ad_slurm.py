from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.slurm import SlurmAccountingError
from multilingual_megatron.experiments.scaling.stage1_release.slurm import collect_slurm_accounting
from multilingual_megatron.experiments.scaling.stage1_release.slurm import discover_slurm_attempts
from multilingual_megatron.experiments.scaling.stage1_release.slurm import normalize_sacct_record
from multilingual_megatron.experiments.scaling.stage1_release.slurm import query_sacct_json


def _sacct_job(
    job_id: int,
    *,
    submit: int = 100,
    eligible: int = 130,
    start: int = 190,
    end: int = 550,
    elapsed: int = 360,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "name": "cv1p1_prod_branch_c_e16_k001_d1b",
        "user": "researcher",
        "account": "study",
        "cluster": "h100",
        "partition": "gpu_h100",
        "qos": "lowcont",
        "qosreq": "lowcont",
        "allocation_nodes": 1,
        "nodes": "gpu1012",
        "constraints": "",
        "working_directory": "/study",
        "stdout_expanded": f"/study/job.{job_id}.log",
        "array": {
            "job_id": 0,
            "task_id": {"set": False, "infinite": False, "number": 0},
        },
        "time": {
            "submission": submit,
            "eligible": eligible,
            "start": start,
            "end": end,
            "elapsed": elapsed,
            "planned": {"set": True, "infinite": False, "number": 60},
        },
        "state": {"current": ["COMPLETED"], "reason": "Dependency"},
        "exit_code": {
            "status": ["SUCCESS"],
            "return_code": {"set": True, "infinite": False, "number": 0},
            "signal": {
                "id": {"set": False, "infinite": False, "number": 0},
                "name": "",
            },
        },
        "derived_exit_code": {
            "status": ["SUCCESS"],
            "return_code": {"set": True, "infinite": False, "number": 0},
            "signal": {
                "id": {"set": False, "infinite": False, "number": 0},
                "name": "",
            },
        },
        "reservation": {
            "id": 81,
            "name": "study_phase1",
            "requested": "study_phase1",
        },
        "submit_line": "sbatch --dependency=afterok:41 --partition=gpu_h100 job.sh",
        "tres": {
            "allocated": [
                {"type": "cpu", "name": "", "count": 32},
                {"type": "mem", "name": "", "count": 131072},
                {"type": "node", "name": "", "count": 1},
                {"type": "gres", "name": "gpu", "count": 2},
                {"type": "gres", "name": "gpu:h100", "count": 2},
            ],
            "requested": [
                {"type": "cpu", "name": "", "count": 32},
                {"type": "mem", "name": "", "count": 131072},
                {"type": "gres", "name": "gpu", "count": 2},
                {"type": "gres", "name": "gpu:h100", "count": 2},
            ],
        },
    }


class Stage1ADSlurmTest(unittest.TestCase):
    def test_discovery_merges_state_history_receipt_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            control = root / "control"
            receipts = control / "receipts"
            logs = root / "logs"
            receipts.mkdir(parents=True)
            logs.mkdir()
            state = {
                "jobs": {
                    "prod_train_c_e4_k012": {
                        "slurm_job_id": "102",
                        "submission_history": [
                            {"slurm_job_id": "101", "state": "FAILED"}
                        ],
                    }
                }
            }
            (control / "workflow_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
            (receipts / "prod_train_c_e4_k012.json").write_text(
                json.dumps(
                    {
                        "job_id": "prod_train_c_e4_k012",
                        "slurm_job_id": "102",
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            (logs / "prod_train_c_e4_k012.101.log").write_text(
                "failed",
                encoding="utf-8",
            )
            (logs / "prod_train_c_e4_k012.102.log").write_text(
                "complete",
                encoding="utf-8",
            )
            (logs / "eval_array.200_7.log").write_text(
                "complete",
                encoding="utf-8",
            )

            attempts = discover_slurm_attempts([root])

        by_id = {(row["logical_job_id"], row["slurm_job_id"]): row for row in attempts}
        self.assertEqual(
            set(by_id),
            {
                ("prod_train_c_e4_k012", "101"),
                ("prod_train_c_e4_k012", "102"),
                ("eval_array", "200"),
            },
        )
        self.assertEqual(by_id[("prod_train_c_e4_k012", "101")]["attempt_index"], 1)
        self.assertEqual(by_id[("prod_train_c_e4_k012", "102")]["attempt_index"], 2)
        self.assertTrue(by_id[("prod_train_c_e4_k012", "102")]["is_current_attempt"])
        kinds = {
            source["kind"]
            for source in by_id[("prod_train_c_e4_k012", "102")]["discovery_sources"]
        }
        self.assertEqual(
            kinds,
            {
                "workflow_state_current",
                "execution_receipt",
                "resolved_slurm_log",
            },
        )

    def test_normalizer_preserves_resources_times_waits_and_raw_record(self) -> None:
        raw = _sacct_job(42)

        row = normalize_sacct_record(raw)

        self.assertEqual(row["slurm_job_id"], "42")
        self.assertEqual(row["state"], "COMPLETED")
        self.assertEqual(row["exit_code"], "0:0")
        self.assertEqual(row["partition"], "gpu_h100")
        self.assertEqual(row["reservation_name"], "study_phase1")
        self.assertEqual(row["node_list"], "gpu1012")
        self.assertEqual(row["allocated_cpus"], 32)
        self.assertEqual(row["allocated_memory_mib"], 131072)
        self.assertEqual(row["allocated_gpus"], 2)
        self.assertEqual(row["allocated_gpu_type_counts"], {"h100": 2})
        self.assertAlmostEqual(row["allocated_gpu_hours"], 0.2)
        self.assertEqual(row["pre_eligibility_wait_seconds"], 30)
        self.assertEqual(row["dependency_wait_seconds"], 30)
        self.assertEqual(row["scheduler_queue_wait_seconds"], 60)
        self.assertEqual(row["total_wait_seconds"], 90)
        self.assertEqual(row["wall_clock_runtime_seconds"], 360)
        self.assertEqual(row["submit_time_utc"], "1970-01-01T00:01:40Z")
        self.assertEqual(row["dependency_job_ids"], ["41"])
        self.assertEqual(row["raw_sacct_record"], raw)
        self.assertEqual(row["field_provenance"]["allocated_gpus"], "observed")
        self.assertEqual(row["field_provenance"]["requested_gpus"], "configured")
        self.assertEqual(
            row["field_provenance"]["scheduler_queue_wait_seconds"],
            "derived",
        )

    def test_normalizer_accepts_cpu_only_allocations(self) -> None:
        raw = _sacct_job(43)
        raw["tres"]["allocated"] = [
            {"type": "cpu", "name": "", "count": 4},
            {"type": "mem", "name": "", "count": 8192},
        ]

        row = normalize_sacct_record(raw)

        self.assertEqual(row["allocated_gpus"], 0)
        self.assertEqual(row["allocated_gpu_type_counts"], {})
        self.assertEqual(row["allocated_gpu_hours"], 0.0)

    def test_query_batches_and_retains_payloads(self) -> None:
        commands: list[list[str]] = []

        def runner(command: object) -> SimpleNamespace:
            rendered = list(command)  # type: ignore[arg-type]
            commands.append(rendered)
            ids = rendered[-1].split(",")
            payload = {"jobs": [_sacct_job(int(value)) for value in ids]}
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )

        payloads = query_sacct_json(["3", "1", "2"], batch_size=2, runner=runner)

        self.assertEqual(len(payloads), 2)
        self.assertEqual(commands[0], ["sacct", "--json", "-X", "--jobs", "1,2"])
        self.assertEqual(commands[1], ["sacct", "--json", "-X", "--jobs", "3"])
        self.assertEqual([row["job_id"] for row in payloads[0]["jobs"]], [1, 2])

    def test_collection_can_flag_or_fail_on_missing_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            (root / "control").mkdir(parents=True)
            (root / "control" / "workflow_state.json").write_text(
                json.dumps(
                    {
                        "jobs": {
                            "present": {
                                "slurm_job_id": "10",
                                "submission_history": [],
                            },
                            "missing": {
                                "slurm_job_id": "11",
                                "submission_history": [],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            payload = {"jobs": [_sacct_job(10)], "meta": {"cluster": "h100"}}

            report = collect_slurm_accounting(
                [root],
                injected_sacct_payloads=[payload],
                missing_policy="flag",
            )
            with self.assertRaises(SlurmAccountingError) as captured:
                collect_slurm_accounting(
                    [root],
                    injected_sacct_payloads=[payload],
                    missing_policy="error",
                )

        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["attempts"], 2)
        self.assertEqual(report["counts"]["accounted_attempts"], 1)
        self.assertEqual(report["counts"]["missing_attempts"], 1)
        self.assertEqual(report["raw_sacct_batches"], [payload])
        self.assertEqual(report["missing_accounting"][0]["slurm_job_id"], "11")
        self.assertIsNotNone(captured.exception.report)
        self.assertEqual(captured.exception.report["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
