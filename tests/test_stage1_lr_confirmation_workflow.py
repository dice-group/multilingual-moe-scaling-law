from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from multilingual_megatron.experiments.scaling.ab_v2.common import PRIMARY_METRIC
from multilingual_megatron.experiments.scaling.ab_v2.gate_lr_diagnostics import (
    _analyze_lr_confirmation_rows,
    _segment_plan_matches_nominal_endpoint,
)
from multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation import DEFAULT_RUNTIME
from multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation import DEFAULT_SPEC
from multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation import LR_CONFIRMATION_VARIANT
from multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation import build_manifest
from multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation import validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


class Stage1LrConfirmationManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.output_root = Path(cls.tempdir.name) / "confirmation"
        cls.manifest = build_manifest(
            spec_path=REPO_ROOT / DEFAULT_SPEC,
            runtime_path=REPO_ROOT / DEFAULT_RUNTIME,
            output_root=cls.output_root,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_manifest_is_confirmation_only_and_has_exact_cardinality(self) -> None:
        result = validate_manifest(self.manifest)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            self.manifest["workflow_variant"],
            LR_CONFIRMATION_VARIANT.manifest_descriptor(),
        )
        self.assertEqual(
            self.manifest["role_counts"],
            {
                "prescribed_lr_schedule_probes": 3,
                "lr_confirmation_trajectories": 54,
                "lr_confirmation_training_jobs": 108,
                "lr_confirmation_evaluations": 54,
                "production_trunks": 0,
                "production_branches": 0,
                "production_evaluations": 0,
            },
        )
        self.assertFalse(
            any(row["stage"] == "production" for row in self.manifest["jobs"])
        )
        self.assertEqual(
            self.manifest["staged_submission"]["phase_submit_slots"][
                "lr_diagnostics"
            ],
            217,
        )
        self.assertFalse(
            self.manifest["staged_submission"]["prior_phase_completion_required"]
        )

    def test_exact_matrix_lrs_and_dependency_chains(self) -> None:
        jobs = {row["job_id"]: row for row in self.manifest["jobs"]}
        trunks = [
            row for row in jobs.values() if row["job_id"].startswith("lrconfirm_trunk_")
        ]
        self.assertEqual(len(trunks), 54)
        expected_base = {
            1: 0.0015596049397046934,
            16: 0.0007798024698523467,
            32: 0.0006557331015048314,
        }
        identities: set[tuple[int, int, str]] = set()
        for trunk in trunks:
            args = trunk["arguments"]
            identity = (args["experts"], args["K"], args["lr_candidate_id"])
            identities.add(identity)
            self.assertEqual(args["D"], 2_000_000_000)
            self.assertEqual(trunk["gpu_tokens"], 1_600_000_000)
            self.assertEqual(args["model_seed"], 1234)
            self.assertEqual(args["dataset_random_seed"], 1234)
            self.assertEqual(args["base_peak_lr"], expected_base[args["experts"]])
            candidate = args["lr_candidate_id"]
            if candidate == "fixed2e4":
                self.assertEqual(args["peak_lr"], 2e-4)
            else:
                factor = {"x025": 0.25, "x050": 0.5, "x100": 1.0, "x200": 2.0, "x400": 4.0}[candidate]
                self.assertEqual(args["peak_lr"], expected_base[args["experts"]] * factor)
            suffix = trunk["job_id"].removeprefix("lrconfirm_trunk_")
            branch = jobs[f"lrconfirm_branch_{suffix}"]
            evaluation = jobs[f"lrconfirm_eval_{suffix}"]
            cleanup = jobs[f"retention_prune_lrconfirm_{suffix}"]
            self.assertEqual(branch["dependencies"], [trunk["job_id"]])
            self.assertEqual(branch["gpu_tokens"], 400_000_000)
            self.assertEqual(evaluation["dependencies"], [branch["job_id"]])
            self.assertEqual(cleanup["dependencies"], [evaluation["job_id"]])
            self.assertEqual(cleanup["arguments"]["source_jobs"], [trunk["job_id"]])
            self.assertEqual(cleanup["arguments"]["retained_jobs"], [branch["job_id"]])
        self.assertEqual(len(identities), 54)

    def test_forecasts_are_informational_and_storage_is_explicit(self) -> None:
        estimate = self.manifest["gpu_hour_estimate"]
        self.assertEqual(estimate["budget_policy"], "informational_no_hard_cap")
        self.assertIsNone(estimate["gpu_hour_limit"])
        self.assertTrue(
            math.isclose(
                estimate["base_estimated_gpu_hours"],
                163.72065663832612,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        walltime = self.manifest["walltime_contract"]
        self.assertEqual(walltime["mode"], "ordinary_queue_forecast")
        self.assertEqual(walltime["assigned_gpu_jobs"], 177)
        self.assertFalse(walltime["automatic_retries"])
        storage = self.manifest["storage_estimate"]
        self.assertEqual(storage["checkpoint_peak_bytes"], 7_564_034_949_120)
        self.assertEqual(
            storage["retained_lr_confirmation_final_checkpoint_bytes"],
            1_891_008_737_280,
        )


def _confirmation_rows(*, mode: str) -> list[dict[str, object]]:
    candidates = ("x025", "x050", "x100", "x200", "x400", "fixed2e4")
    rows: list[dict[str, object]] = []
    cell_index = 0
    for experts in (1, 16, 32):
        for k in (1, 12, 200):
            losses = {
                "x025": -0.96,
                "x050": -0.99,
                "x100": -1.0,
                "x200": -0.97,
                "x400": -0.95,
                "fixed2e4": -0.94,
            }
            if mode == "confirmed" and cell_index >= 6:
                losses["x050"] = -1.0
                losses["x100"] = -0.999
            elif mode == "not_confirmed":
                losses["x050"] = -1.01
            elif mode == "boundary" and cell_index == 0:
                losses["x025"] = -1.02
            for candidate in candidates:
                rows.append(
                    {
                        "diagnostic_protocol": "stage1_jointmoe_lr_confirmation_v1",
                        "ladder": "B",
                        "experts": experts,
                        "K": k,
                        "model_seed": 1234,
                        "lr_candidate_id": candidate,
                        PRIMARY_METRIC: losses[candidate],
                        "raw_nll": 4.0 + losses[candidate],
                    }
                )
            cell_index += 1
    return rows


class Stage1LrConfirmationDecisionTest(unittest.TestCase):
    def analyze(self, mode: str):
        return _analyze_lr_confirmation_rows(
            _confirmation_rows(mode=mode),
            tie_absolute_tolerance=1e-12,
            minimum_x1_cell_wins_or_ties=5,
        )

    def test_negative_adjusted_loss_and_robust_x1_confirmation(self) -> None:
        result = self.analyze("confirmed")
        self.assertEqual(result["decision"], "confirmed")
        self.assertTrue(result["negative_primary_values_are_valid"])
        self.assertEqual(result["candidate_aggregates"]["x100"]["wins_or_ties"], 6)
        self.assertTrue(all(result["checks"].values()))

    def test_better_neighbor_does_not_confirm_x1(self) -> None:
        result = self.analyze("not_confirmed")
        self.assertEqual(result["decision"], "not_confirmed")
        self.assertFalse(
            result["checks"]["x1_strictly_lowest_equal_cell_mean_regret"]
        )

    def test_outer_boundary_optimum_is_inconclusive(self) -> None:
        result = self.analyze("boundary")
        self.assertEqual(result["decision"], "inconclusive_boundary")
        self.assertEqual(result["boundary_cells"], ["B/E1/K1"])


class Stage1LrConfirmationSegmentPlanTest(unittest.TestCase):
    endpoint = {
        "actual_end_tokens": 2_000_158_720,
        "expected_consumed_samples": 976_640,
        "expected_end_step": 15_260,
    }

    def test_branch_execution_plan_matches_nominal_endpoint(self) -> None:
        plan = {
            "plan_role": "branch",
            "target_token_budget_tokens": 2_000_000_000,
            "branch_target_tokens": 2_000_000_000,
            "expected_tokens": 2_000_158_720,
            "expected_consumed_samples": 976_640,
            "expected_end_step": 15_260,
            "final_cumulative_step": 15_260,
            "segments": [
                {
                    "target_end_tokens": 2_000_000_000,
                    "actual_end_tokens": 2_000_158_720,
                    "expected_consumed_samples": 976_640,
                    "expected_end_step": 15_260,
                }
            ],
        }
        self.assertTrue(
            _segment_plan_matches_nominal_endpoint(
                plan,
                nominal_horizon=2_000_000_000,
                expected_endpoint=self.endpoint,
            )
        )
        plan["segments"][0]["actual_end_tokens"] = 2_000_000_000
        self.assertFalse(
            _segment_plan_matches_nominal_endpoint(
                plan,
                nominal_horizon=2_000_000_000,
                expected_endpoint=self.endpoint,
            )
        )

    def test_legacy_combined_plan_matches_nominal_endpoint(self) -> None:
        plan = {
            "dataset_horizon_tokens": 2_000_000_000,
            "branches": [
                {
                    "target_tokens": 2_000_000_000,
                    "actual_end_tokens": 2_000_158_720,
                    "expected_consumed_samples": 976_640,
                    "expected_end_step": 15_260,
                }
            ],
        }
        self.assertTrue(
            _segment_plan_matches_nominal_endpoint(
                plan,
                nominal_horizon=2_000_000_000,
                expected_endpoint=self.endpoint,
            )
        )


if __name__ == "__main__":
    unittest.main()
