from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.relations import _segment_uses_exact_step_rounding
from multilingual_megatron.experiments.scaling.stage1_release.relations import _training_metric_diagnostics


class Stage1ADReleaseRelationsTest(unittest.TestCase):
    def test_segment_rounding_uses_cumulative_target_endpoints(self) -> None:
        config = {"sequence_length": 2048}
        segment = {
            "target_start_tokens": 3_200_000_000,
            "target_end_tokens": 6_400_000_000,
            "global_batch_size": 128,
            "segment_kind": "trunk",
        }
        step = config["sequence_length"] * segment["global_batch_size"]
        segment["actual_start_tokens"] = (
            math.ceil(segment["target_start_tokens"] / step) * step
        )
        segment["actual_end_tokens"] = (
            math.ceil(segment["target_end_tokens"] / step) * step
        )

        self.assertTrue(_segment_uses_exact_step_rounding(segment, config))
        segment["actual_end_tokens"] += 1
        self.assertFalse(_segment_uses_exact_step_rounding(segment, config))

    def test_branch_rounding_uses_parent_endpoint_plus_incremental_span(self) -> None:
        config = {"sequence_length": 2048}
        segment = {
            "target_start_tokens": 3_200_000_000,
            "target_end_tokens": 4_000_000_000,
            "actual_start_tokens": 3_200_253_952,
            "global_batch_size": 128,
            "segment_kind": "branch",
        }
        step = config["sequence_length"] * segment["global_batch_size"]
        target_span = segment["target_end_tokens"] - segment["target_start_tokens"]
        segment["actual_end_tokens"] = (
            segment["actual_start_tokens"] + math.ceil(target_span / step) * step
        )

        self.assertTrue(_segment_uses_exact_step_rounding(segment, config))
        segment["actual_start_tokens"] = segment["target_start_tokens"] - 1
        self.assertFalse(_segment_uses_exact_step_rounding(segment, config))

    def test_retry_iteration_reset_is_disclosed_not_reordered(self) -> None:
        rows = [
            {
                "event_type": "training_iteration",
                "training_job_id": "job",
                "segment_id": "segment",
                "source_path": "/logs/job.log",
                "source_line_number": 10,
                "iteration": 4600,
                "lm_loss": 3.0,
                "elapsed_time_ms": 500.0,
            },
            {
                "event_type": "training_iteration",
                "training_job_id": "job",
                "segment_id": "segment",
                "source_path": "/logs/job.log",
                "source_line_number": 20,
                "iteration": 10,
                "lm_loss": 3.2,
                "elapsed_time_ms": 510.0,
            },
        ]

        errors, resets = _training_metric_diagnostics(rows)

        self.assertEqual(errors, [])
        self.assertEqual(resets, 1)

    def test_nonfinite_metric_or_reordered_source_line_fails(self) -> None:
        rows = [
            {
                "event_type": "training_iteration",
                "training_job_id": "job",
                "segment_id": "segment",
                "source_path": "/logs/job.log",
                "source_line_number": 20,
                "iteration": 10,
                "lm_loss": 3.0,
                "elapsed_time_ms": 500.0,
            },
            {
                "event_type": "training_iteration",
                "training_job_id": "job",
                "segment_id": "segment",
                "source_path": "/logs/job.log",
                "source_line_number": 19,
                "iteration": 20,
                "lm_loss": float("nan"),
                "elapsed_time_ms": 0.0,
            },
        ]

        errors, resets = _training_metric_diagnostics(rows)

        self.assertEqual(resets, 0)
        self.assertTrue(any("source_line_order" in value for value in errors))
        self.assertTrue(any("nonfinite_lm_loss" in value for value in errors))
        self.assertTrue(any("nonpositive_elapsed_time" in value for value in errors))


if __name__ == "__main__":
    unittest.main()
