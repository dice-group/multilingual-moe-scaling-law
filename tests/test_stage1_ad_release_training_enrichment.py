from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import DESIGN_TOTAL_PARAMETER_FORMULA
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import H100_EFFICIENCY_DOCUMENT_SHA256
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import Stage1ADReleaseTrainingEnrichmentError
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import build_readout_compute_attribution
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import enrich_model_configs
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import enrich_training_summaries
from multilingual_megatron.experiments.scaling.stage1_release.training_enrichment import validate_production_graph


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProductionGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.jobs: list[dict[str, object]] = []
        cls.readouts: list[dict[str, object]] = []
        endpoints_by_ladder = {
            "A": (500, 1_000, 2_000, 4_000, 8_000),
            "B": (500, 1_000, 2_000, 4_000, 8_000, 16_000),
            "C": (500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000),
            "D": (500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000),
        }
        for ladder in ("A", "B", "C", "D"):
            for experts in (1, 4, 16, 32):
                for language_count in (1, 12, 50, 200):
                    trajectory = f"{ladder}:E{experts}:K{language_count}"
                    trunk_id = f"trunk-{ladder}-{experts}-{language_count}"
                    trunk_job = cls._write_job(
                        root,
                        job_id=trunk_id,
                        trajectory=trajectory,
                        role="trunk",
                        parent_id=None,
                        target_tokens=32_000,
                        actual_tokens=25_600,
                        step=256,
                    )
                    cls.jobs.append(trunk_job)
                    for nominal_tokens in endpoints_by_ladder[ladder]:
                        branch_id = (
                            f"branch-{ladder}-{experts}-{language_count}-"
                            f"{nominal_tokens}"
                        )
                        branch_job = cls._write_job(
                            root,
                            job_id=branch_id,
                            trajectory=trajectory,
                            role="branch",
                            parent_id=trunk_id,
                            target_tokens=nominal_tokens,
                            actual_tokens=nominal_tokens,
                            step=nominal_tokens // 10,
                        )
                        cls.jobs.append(branch_job)
                        checkpoint_root = Path(str(branch_job["final_checkpoint_root"]))
                        checkpoint_step = int(branch_job["final_checkpoint_step"])
                        iteration = checkpoint_root / f"iter_{checkpoint_step:07d}"
                        iteration.mkdir(parents=True)
                        cls.readouts.append(
                            {
                                "readout_id": f"readout-{branch_id}",
                                "trajectory_id": trajectory,
                                "ladder": ladder,
                                "experts": experts,
                                "language_count": language_count,
                                "nominal_tokens": nominal_tokens,
                                "actual_tokens": nominal_tokens,
                                "expected_consumed_samples": nominal_tokens // 10,
                                "training_job_id": branch_id,
                                "checkpoint_root": str(checkpoint_root),
                                "checkpoint_iteration_path": str(iteration),
                                "checkpoint_step": checkpoint_step,
                                "training_manifest_path": branch_job[
                                    "training_manifest_path"
                                ],
                                "training_manifest_sha256": branch_job[
                                    "training_manifest_sha256"
                                ],
                            }
                        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _write_job(
        cls,
        root: Path,
        *,
        job_id: str,
        trajectory: str,
        role: str,
        parent_id: str | None,
        target_tokens: int,
        actual_tokens: int,
        step: int,
    ) -> dict[str, object]:
        job_root = root / "jobs" / job_id
        checkpoint_root = job_root / "checkpoints"
        checkpoint_root.mkdir(parents=True)
        manifest = job_root / "training_manifest.json"
        manifest.write_text(
            json.dumps({"metadata": {"plan_role": role}}),
            encoding="utf-8",
        )
        first_segment: dict[str, object] = {
            "actual_start_tokens": (
                int(target_tokens * 0.8) if role == "branch" else 0
            ),
            "actual_end_tokens": actual_tokens,
        }
        if role == "branch":
            first_segment["initial_checkpoint"] = {
                "checkpoint_root": str(
                    root / "jobs" / str(parent_id) / "checkpoint-prefix"
                )
            }
        plan = job_root / "segment_plan.json"
        plan.write_text(
            json.dumps(
                {
                    "expected_tokens": actual_tokens,
                    "segments": [first_segment],
                }
            ),
            encoding="utf-8",
        )
        result = job_root / "result.json"
        result_payload = {
            "status": "complete",
            "role": role,
            "actual_end_tokens": actual_tokens,
            "expected_consumed_samples": actual_tokens // 10,
            "final_checkpoint_root": str(checkpoint_root),
            "final_checkpoint_step": step,
            "training_manifest": str(manifest),
            "training_manifest_sha256": _sha256(manifest),
            "segment_plan": str(plan),
            "segment_plan_sha256": _sha256(plan),
        }
        result.write_text(json.dumps(result_payload), encoding="utf-8")
        return {
            "training_job_id": job_id,
            "trajectory_id": trajectory,
            "job_role": role,
            "parent_training_job_id": parent_id,
            "target_tokens": target_tokens,
            "actual_end_tokens": actual_tokens,
            "expected_consumed_samples": actual_tokens // 10,
            "status": "complete",
            "result_path": str(result),
            "training_manifest_path": str(manifest),
            "training_manifest_sha256": _sha256(manifest),
            "final_checkpoint_root": str(checkpoint_root),
            "final_checkpoint_step": step,
        }

    def test_validates_exact_production_graph_and_artifact_hashes(self) -> None:
        report = validate_production_graph(self.jobs, self.readouts)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["training_jobs"], 464)
        self.assertEqual(report["summary"]["trunks"], 64)
        self.assertEqual(report["summary"]["branches"], 400)
        self.assertEqual(report["summary"]["recomputed_artifact_hashes"], 928)
        self.assertTrue(
            all(row["lineage_status"] == "pass" for row in report["readout_bindings"])
        )
        self.assertTrue(
            all(
                row["checkpoint_preservation_status"] == "present"
                for row in report["readout_bindings"]
            )
        )

    def test_missing_checkpoint_is_disclosed_without_invalidating_lineage(self) -> None:
        iteration = Path(str(self.readouts[0]["checkpoint_iteration_path"]))
        iteration.rmdir()
        try:
            report = validate_production_graph(self.jobs, self.readouts)
        finally:
            iteration.mkdir()

        binding = next(
            row
            for row in report["readout_bindings"]
            if row["readout_id"] == self.readouts[0]["readout_id"]
        )
        self.assertEqual(binding["lineage_status"], "pass")
        self.assertEqual(
            binding["checkpoint_preservation_status"],
            "absent_now",
        )
        self.assertEqual(
            binding["status"],
            "lineage_valid_checkpoint_absent_now",
        )
        self.assertEqual(report["summary"]["checkpoint_directories_absent_now"], 1)

    def test_rejects_changed_manifest_hash(self) -> None:
        jobs = copy.deepcopy(self.jobs)
        jobs[0]["training_manifest_sha256"] = "0" * 64

        with self.assertRaises(Stage1ADReleaseTrainingEnrichmentError) as caught:
            validate_production_graph(jobs, self.readouts)

        self.assertEqual(caught.exception.report["status"], "fail")
        self.assertIn("training_manifest_sha256", str(caught.exception))

    def test_rejects_cross_trajectory_parent(self) -> None:
        jobs = copy.deepcopy(self.jobs)
        branch = next(row for row in jobs if row["job_role"] == "branch")
        wrong_parent = next(
            row
            for row in jobs
            if row["job_role"] == "trunk"
            and row["trajectory_id"] != branch["trajectory_id"]
        )
        branch["parent_training_job_id"] = wrong_parent["training_job_id"]

        with self.assertRaises(Stage1ADReleaseTrainingEnrichmentError) as caught:
            validate_production_graph(jobs, self.readouts)

        errors = caught.exception.report["errors"]
        self.assertTrue(any("cross-trajectory" in value for value in errors))


class TrainingSummaryEnrichmentTest(unittest.TestCase):
    def test_enriches_gpu_count_mfu_throughput_and_coverage(self) -> None:
        jobs = [
            {"training_job_id": "one", "resource": "h100"},
            {"training_job_id": "four", "resource": "h100_4"},
        ]
        summaries = [
            {
                "training_job_id": "one",
                "mean_tokens_per_second": 1000.0,
                "mean_model_tflops_per_gpu": 494.75,
                "walltime_seconds": 100.0,
                "board_energy_observed_gpu_seconds": 90.0,
                "telemetry_nominal_interval_seconds": 10.0,
                "valid_gpu_telemetry_rows": 10,
                "telemetry_gpu_uuids_json": '["GPU-a"]',
            },
            {
                "training_job_id": "four",
                "mean_tokens_per_second": 4000.0,
                "mean_model_tflops_per_gpu": 247.375,
                "walltime_seconds": 100.0,
                "board_energy_observed_gpu_seconds": 360.0,
                "telemetry_nominal_interval_seconds": 10.0,
                "valid_gpu_telemetry_rows": 40,
                "telemetry_gpu_uuids_json": ('["GPU-a","GPU-b","GPU-c","GPU-d"]'),
            },
        ]
        original = copy.deepcopy(summaries)

        rows, hardware = enrich_training_summaries(
            jobs,
            summaries,
            REPO_ROOT,
        )

        self.assertEqual(summaries, original)
        by_id = {row["training_job_id"]: row for row in rows}
        self.assertEqual(by_id["one"]["configured_gpu_count"], 1)
        self.assertEqual(by_id["four"]["configured_gpu_count"], 4)
        self.assertEqual(by_id["one"]["mean_tokens_per_second_per_gpu"], 1000)
        self.assertEqual(by_id["four"]["mean_tokens_per_second_per_gpu"], 1000)
        self.assertEqual(by_id["one"]["mean_mfu_pct_of_configured_peak"], 50)
        self.assertEqual(by_id["four"]["mean_mfu_pct_of_configured_peak"], 25)
        self.assertEqual(by_id["one"]["telemetry_temporal_coverage_fraction"], 0.9)
        self.assertEqual(by_id["four"]["telemetry_temporal_coverage_fraction"], 0.9)
        self.assertEqual(by_id["one"]["gpu_sample_count_coverage_status"], "partial")
        self.assertEqual(by_id["four"]["gpu_sample_count_coverage_status"], "partial")
        self.assertEqual(by_id["one"]["board_energy_coverage_status"], "partial")
        self.assertEqual(by_id["four"]["board_energy_coverage_status"], "partial")
        self.assertEqual(
            hardware[0]["source_document_sha256"],
            H100_EFFICIENCY_DOCUMENT_SHA256,
        )
        self.assertEqual(
            hardware[0]["denominator_kind"],
            "configured_theoretical_peak",
        )

    def test_rejects_ambiguous_resource(self) -> None:
        with self.assertRaisesRegex(
            Stage1ADReleaseTrainingEnrichmentError,
            "exactly one of",
        ):
            enrich_training_summaries(
                [{"training_job_id": "bad", "resource": "h100x2"}],
                [{"training_job_id": "bad"}],
                REPO_ROOT,
            )


class ModelConfigEnrichmentTest(unittest.TestCase):
    def test_keeps_parameter_scopes_distinct(self) -> None:
        models = [
            {
                "model_config_id": "A:E4",
                "ladder": "A",
                "experts": 4,
                "num_layers": 12,
                "hidden_size": 768,
                "vocab_size": 64_000,
                "declared_active_parameters": 200_000_000,
                "declared_active_nonembedding_parameters": 100_000_000,
            }
        ]
        runtime = [
            {
                "training_job_id": "job",
                "trajectory_id": "A:E4:K1",
                "ladder": "A",
                "experts": 4,
                "segment_id": "s0",
                "parameter_scope": "model_parallel_rank_partition",
                "parameter_count": 400_123_456,
                "tensor_model_parallel_rank": 0,
                "pipeline_model_parallel_rank": 0,
            },
            {
                "training_job_id": "job",
                "trajectory_id": "A:E4:K1",
                "ladder": "A",
                "experts": 4,
                "segment_id": "s0",
                "parameter_scope": "reported_total_rounded_billions",
                "parameter_count": 400_000_000,
                "parameter_count_billions_reported": 0.4,
            },
        ]
        original = copy.deepcopy(models)

        rows = enrich_model_configs(models, runtime)

        self.assertEqual(models, original)
        expected = 2 * 768 * 64_000 + (4 + 9 * 4) * 12 * 768 * 768
        self.assertEqual(rows[0]["configured_design_total_parameters"], expected)
        self.assertEqual(
            rows[0]["configured_design_total_parameters_formula"],
            DESIGN_TOTAL_PARAMETER_FORMULA,
        )
        self.assertEqual(rows[0]["declared_active_parameters"], 200_000_000)
        self.assertEqual(
            rows[0]["declared_active_nonembedding_parameters"],
            100_000_000,
        )
        self.assertEqual(
            rows[0]["runtime_rank_local_parameter_count_values"],
            [400_123_456],
        )
        self.assertEqual(
            rows[0]["runtime_rounded_global_parameter_count_values"],
            [400_000_000],
        )
        self.assertEqual(
            rows[0]["runtime_parameter_observation_status"],
            "rank_local_and_rounded_global_observed",
        )


class ComputeAttributionTest(unittest.TestCase):
    def test_separates_logical_attribution_from_unique_physical_totals(
        self,
    ) -> None:
        jobs = [
            {
                "training_job_id": "trunk",
                "trajectory_id": "A:E1:K1",
                "job_role": "trunk",
                "parent_training_job_id": None,
                "resource": "h100",
            },
            {
                "training_job_id": "branch",
                "trajectory_id": "A:E1:K1",
                "job_role": "branch",
                "parent_training_job_id": "trunk",
                "resource": "h100_2",
            },
        ]
        readouts = [
            {
                "readout_id": "readout",
                "training_job_id": "branch",
                "trajectory_id": "A:E1:K1",
                "ladder": "A",
                "experts": 1,
                "language_count": 1,
                "nominal_tokens": 300,
            }
        ]
        segments = [
            {
                "segment_id": "trunk-s0",
                "training_job_id": "trunk",
                "actual_start_tokens": 0,
                "actual_end_tokens": 100,
                "segment_train_iters": 10,
            },
            {
                "segment_id": "trunk-s1",
                "training_job_id": "trunk",
                "actual_start_tokens": 100,
                "actual_end_tokens": 200,
                "segment_train_iters": 10,
            },
            {
                "segment_id": "branch-s0",
                "training_job_id": "branch",
                "actual_start_tokens": 150,
                "actual_end_tokens": 300,
                "segment_train_iters": 10,
            },
        ]
        metrics = [
            {
                "event_type": "training_iteration",
                "segment_id": "trunk-s0",
                "elapsed_time_ms": 1000,
            },
            {
                "event_type": "training_iteration",
                "segment_id": "trunk-s1",
                "elapsed_time_ms": 2000,
            },
            {
                "event_type": "training_iteration",
                "segment_id": "branch-s0",
                "elapsed_time_ms": 1500,
            },
        ]
        compute = [
            {
                "training_job_id": "trunk",
                "allocated_gpu_hours_including_retries": 1.0,
                "board_energy_estimate_kwh": 0.1,
                "board_energy_coverage_fraction": 1.0,
            },
            {
                "training_job_id": "branch",
                "allocated_gpu_hours_including_retries": 2.0,
                "board_energy_estimate_kwh": 0.2,
                "board_energy_coverage_fraction": 1.0,
            },
        ]

        result = build_readout_compute_attribution(
            readouts,
            jobs,
            segments,
            metrics,
            compute,
        )

        logical = result["readout_compute_attribution"][0]
        self.assertAlmostEqual(
            logical["attributed_parent_trunk_prefix_productive_gpu_hours"],
            20 / 3600,
        )
        self.assertAlmostEqual(
            logical["attributed_branch_productive_gpu_hours"],
            30 / 3600,
        )
        self.assertAlmostEqual(
            logical["logical_readout_attributed_productive_gpu_hours"],
            50 / 3600,
        )
        self.assertFalse(logical["physical_aggregation_allowed"])
        physical = result["unique_physical_study_compute"][0]
        self.assertEqual(
            physical["observed_allocated_gpu_hours_including_retries_unique_jobs"],
            3.0,
        )
        self.assertAlmostEqual(
            physical["observed_board_energy_estimate_kwh_unique_jobs"],
            0.3,
        )
        self.assertIsNone(physical["logical_readout_attribution_sum"])
        self.assertIn(
            "double_counted",
            physical["logical_attribution_sum_status"],
        )


if __name__ == "__main__":
    unittest.main()
