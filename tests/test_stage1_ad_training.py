from __future__ import annotations

import io
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.training import aggregate_job_training
from multilingual_megatron.experiments.scaling.stage1_release.training import parse_megatron_training_line
from multilingual_megatron.experiments.scaling.stage1_release.training import parse_megatron_training_log
from multilingual_megatron.experiments.scaling.stage1_release.training import parse_megatron_training_lines
from multilingual_megatron.experiments.scaling.stage1_release.training import parse_nvidia_smi_csv


MOE_LINE = (
    " [2026-07-18 10:08:21] iteration     6110/  152600 | "
    "consumed samples:       195520 | elapsed time per iteration (ms): 1976.6 | "
    "throughput per GPU (TFLOP/s/GPU): 13.6 | learning rate: 1.992136E-04 | "
    "global batch size:    32 | lm loss: 3.455902E+00 | "
    "load_balancing_loss: 9.928218E-01 | z_loss: 1.573237E-01 | "
    "loss scale: 1.0 | grad norm: 0.540 | number of skipped iterations: 0 | "
    "number of nan iterations: 0 |"
)


class Stage1ADTrainingTest(unittest.TestCase):
    def test_parses_real_megatron_iteration_and_derives_tokens(self) -> None:
        row = parse_megatron_training_line(
            MOE_LINE,
            sequence_length=2048,
            lineage={
                "trajectory_id": "C:E32:K1",
                "job_id": "prod_trunk_c_e32_k001",
                "segment_id": "00_gbs32",
                "segment_kind": "trunk",
            },
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["event_type"], "training_iteration")
        self.assertEqual(row["timestamp"], "2026-07-18T10:08:21")
        self.assertEqual(row["timestamp_timezone"], "unspecified")
        self.assertEqual(row["iteration"], 6110)
        self.assertEqual(row["iteration_target"], 152600)
        self.assertEqual(row["consumed_samples"], 195520)
        self.assertEqual(row["consumed_tokens"], 195520 * 2048)
        self.assertAlmostEqual(row["model_tflops_per_gpu"], 13.6)
        self.assertAlmostEqual(row["learning_rate"], 1.992136e-4)
        self.assertEqual(row["global_batch_size"], 32)
        self.assertAlmostEqual(row["lm_loss"], 3.455902)
        self.assertAlmostEqual(row["load_balancing_loss"], 0.9928218)
        self.assertAlmostEqual(row["z_loss"], 0.1573237)
        self.assertAlmostEqual(row["grad_norm"], 0.540)
        self.assertEqual(row["skipped_iterations"], 0)
        self.assertEqual(row["nan_iterations"], 0)
        self.assertAlmostEqual(row["samples_per_second"], 32 / 1.9766)
        self.assertAlmostEqual(row["tokens_per_second"], 32 * 2048 / 1.9766)
        self.assertEqual(row["segment_kind"], "trunk")

    def test_dense_line_retains_unavailable_moe_metrics_as_null(self) -> None:
        dense = MOE_LINE.replace(
            " | load_balancing_loss: 9.928218E-01 | z_loss: 1.573237E-01",
            "",
        )
        row = parse_megatron_training_line(dense)

        assert row is not None
        self.assertIsNone(row["load_balancing_loss"])
        self.assertIsNone(row["z_loss"])
        self.assertIsNone(row["consumed_tokens"])
        self.assertIsNone(row["tokens_per_second"])

    def test_parameter_memory_and_source_line_events(self) -> None:
        rows = parse_megatron_training_lines(
            [
                " > number of parameters on (tensor, pipeline) model parallel "
                "rank (0, 0): 678044160\n",
                "Total number of parameters in billions: 0.68\n",
                "[Rank 0] memory (GB) | mem-max-allocated-gigabytes: 42.5 | "
                "mem-max-reserved-gigabytes: 48.0 |\n",
                "irrelevant\n",
            ],
            lineage={"job_id": "job-a"},
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["parameter_count"], 678044160)
        self.assertEqual(rows[0]["parameter_scope"], "model_parallel_rank_partition")
        self.assertEqual(rows[0]["source_line_number"], 1)
        self.assertEqual(rows[1]["parameter_count"], 680_000_000)
        self.assertEqual(rows[1]["parameter_scope"], "reported_total_rounded_billions")
        self.assertEqual(rows[2]["event_type"], "training_memory")
        self.assertEqual(rows[2]["max_allocated_gib"], 42.5)
        self.assertEqual(rows[2]["max_reserved_gib"], 48.0)

    def test_log_path_inference_preserves_trunk_segment_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = (
                Path(temporary)
                / "jobs"
                / "prod_trunk_c_e16_k050"
                / "runs"
                / "cv1_trunk_c_e16_k050_d32000000000"
                / "segments"
                / "04_gbs128.train.log"
            )
            path.parent.mkdir(parents=True)
            path.write_text(MOE_LINE + "\n", encoding="utf-8")

            rows = parse_megatron_training_log(
                path,
                sequence_length=2048,
                lineage={"parent_segment_id": "03_gbs64"},
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["training_job_id"], "prod_trunk_c_e16_k050")
        self.assertEqual(rows[0]["run_id"], "cv1_trunk_c_e16_k050_d32000000000")
        self.assertEqual(rows[0]["segment_id"], "04_gbs128")
        self.assertEqual(rows[0]["segment_index"], 4)
        self.assertEqual(rows[0]["segment_kind"], "trunk")
        self.assertEqual(rows[0]["parent_segment_id"], "03_gbs64")

    def test_parses_nvidia_csv_and_marks_bad_rows(self) -> None:
        rows = parse_nvidia_smi_csv(
            io.StringIO(
                "\n".join(
                    [
                        "timestamp,index,uuid,utilization_gpu_pct,memory_used_mib,"
                        "memory_total_mib,power_draw_w",
                        "2026/07/18 04:09:02.044, 0, GPU-a, 27, 47418, 95830, 303.64",
                        "2026/07/18 04:09:12.044, 0, GPU-a, 93, 49350, 95830, 585.40",
                        "bad, 1, GPU-b, bad, 1, 2, bad",
                    ]
                )
            ),
            lineage={"job_id": "job-a", "attempt_id": "attempt-1"},
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["timestamp"], "2026-07-18T04:09:02.044")
        self.assertTrue(rows[0]["sample_valid"])
        self.assertAlmostEqual(rows[0]["memory_used_fraction"], 47418 / 95830)
        self.assertEqual(rows[1]["seconds_since_previous_sample"], 10.0)
        self.assertFalse(rows[2]["sample_valid"])
        self.assertEqual(rows[2]["job_id"], "job-a")

    def test_aggregate_reports_mfu_throughput_coverage_and_board_energy(self) -> None:
        training = [
            parse_megatron_training_line(
                (
                    "[2026-07-18 10:08:21] iteration 1/2 | consumed samples: 8 | "
                    "elapsed time per iteration (ms): 1000 | "
                    "throughput per GPU (TFLOP/s/GPU): 400 | "
                    "global batch size: 8 | lm loss: 3 | "
                    "number of skipped iterations: 0 | number of nan iterations: 0 |"
                ),
                sequence_length=100,
                lineage={
                    "job_id": "branch-a",
                    "segment_id": "segment-1",
                    "segment_kind": "branch",
                    "parent_segment_id": "trunk-0",
                },
            ),
            parse_megatron_training_line(
                (
                    "[2026-07-18 10:08:22] iteration 2/2 | consumed samples: 16 | "
                    "elapsed time per iteration (ms): 2000 | "
                    "throughput per GPU (TFLOP/s/GPU): 500 | "
                    "global batch size: 8 | lm loss: 2 | "
                    "number of skipped iterations: 0 | number of nan iterations: 0 |"
                ),
                sequence_length=100,
                lineage={
                    "job_id": "branch-a",
                    "segment_id": "segment-1",
                    "segment_kind": "branch",
                    "parent_segment_id": "trunk-0",
                },
            ),
        ]
        assert all(item is not None for item in training)
        telemetry = parse_nvidia_smi_csv(
            io.StringIO(
                "\n".join(
                    [
                        "timestamp,index,uuid,utilization_gpu_pct,memory_used_mib,"
                        "memory_total_mib,power_draw_w",
                        "2026/07/18 10:08:20,0,GPU-a,50,1000,2000,100",
                        "2026/07/18 10:08:30,0,GPU-a,90,1500,2000,200",
                        "2026/07/18 10:08:20,1,GPU-b,60,1100,2000,100",
                        "2026/07/18 10:08:30,1,GPU-b,80,1400,2000,200",
                    ]
                )
            ),
            lineage={"job_id": "branch-a", "attempt_id": "attempt-1"},
        )

        summary = aggregate_job_training(
            [item for item in training if item is not None],
            telemetry,
            gpu_count=2,
            peak_tflops_per_gpu=1000,
            walltime_seconds=10,
            expected_logged_metric_rows=4,
            expected_gpu_sample_interval_seconds=10,
            lineage={"job_id": "branch-a"},
        )

        self.assertEqual(summary["parsed_training_metric_rows"], 2)
        self.assertEqual(summary["training_metric_coverage_fraction"], 0.5)
        self.assertEqual(summary["training_metric_coverage_status"], "partial")
        self.assertEqual(summary["mean_step_ms"], 1500)
        self.assertEqual(summary["mean_tokens_per_second"], 600)
        self.assertEqual(summary["mean_tokens_per_second_per_gpu"], 300)
        self.assertEqual(summary["mean_model_tflops_per_gpu"], 450)
        self.assertEqual(summary["mean_mfu_pct_of_configured_peak"], 45)
        self.assertEqual(summary["mean_gpu_utilization_pct"], 70)
        self.assertEqual(summary["max_memory_used_fraction"], 0.75)
        self.assertEqual(summary["mean_board_power_draw_w"], 150)
        self.assertAlmostEqual(
            summary["board_energy_estimate_kwh"],
            3000 / 3_600_000,
        )
        self.assertEqual(summary["board_energy_coverage_fraction"], 1.0)
        self.assertEqual(summary["board_energy_coverage_status"], "complete")
        self.assertEqual(summary["gpu_sample_count_coverage_fraction"], 1.0)
        self.assertEqual(summary["lineage_records"][0]["segment_kind"], "branch")
        self.assertIn("GPU board energy only", summary["board_energy_estimate_method"])

    def test_energy_excludes_large_gaps_and_unknown_expectations_stay_unknown(
        self,
    ) -> None:
        telemetry = parse_nvidia_smi_csv(
            io.StringIO(
                "\n".join(
                    [
                        "timestamp,index,uuid,utilization_gpu_pct,memory_used_mib,"
                        "memory_total_mib,power_draw_w",
                        "2026/07/18 10:00:00,0,GPU-a,50,1000,2000,100",
                        "2026/07/18 10:00:10,0,GPU-a,50,1000,2000,100",
                        "2026/07/18 10:10:00,0,GPU-a,50,1000,2000,100",
                    ]
                )
            )
        )
        summary = aggregate_job_training(
            [],
            telemetry,
            expected_gpu_sample_interval_seconds=10,
        )

        self.assertAlmostEqual(
            summary["board_energy_estimate_kwh"],
            1000 / 3_600_000,
        )
        self.assertEqual(summary["board_energy_excluded_gap_count"], 1)
        self.assertIsNone(summary["board_energy_coverage_fraction"])
        self.assertEqual(summary["board_energy_coverage_status"], "not_computable")
        self.assertIsNone(summary["training_metric_coverage_fraction"])
        self.assertEqual(
            summary["training_metric_coverage_status"],
            "not_computable",
        )
        self.assertTrue(
            math.isclose(summary["board_energy_observed_gpu_seconds"], 10)
        )

    def test_rejects_invalid_aggregation_denominators(self) -> None:
        with self.assertRaisesRegex(ValueError, "gpu_count"):
            aggregate_job_training([], [], gpu_count=0)
        with self.assertRaisesRegex(ValueError, "peak_tflops"):
            aggregate_job_training([], [], peak_tflops_per_gpu=0)
        with self.assertRaisesRegex(ValueError, "walltime"):
            aggregate_job_training([], [], walltime_seconds=0)


if __name__ == "__main__":
    unittest.main()
