from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multilingual_megatron.experiments.scaling.ab_v2.common import _load_object
from multilingual_megatron.experiments.scaling.ab_v2.training_profile import (
    _training_overrides,
)
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import BASELINE_MANIFEST
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import COMPLETE_SEEDS
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import DEFAULT_RUNTIME
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import DEFAULT_SPEC
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import HORIZONS
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import NEW_SEEDS
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import SELECTED_EXPERTS
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import SELECTED_LANGUAGE_COUNTS
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import build_manifest
from multilingual_megatron.experiments.scaling.stage1_training.seed_confirmation import validate_manifest
from multilingual_megatron.experiments.scaling.ab_v2.job_validation import _topological


REPO_ROOT = Path(__file__).resolve().parents[1]


class Stage1BSeedConfirmationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.output_root = Path(cls.tempdir.name) / "seed-confirmation"
        cls.manifest = build_manifest(
            spec_path=REPO_ROOT / DEFAULT_SPEC,
            runtime_path=REPO_ROOT / DEFAULT_RUNTIME,
            output_root=cls.output_root,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_exact_panel_cardinality_and_scope(self) -> None:
        report = validate_manifest(self.manifest, verify_inputs=False)
        self.assertEqual(report["jobs"], 235)
        self.assertEqual(report["new_trajectories"], 18)
        self.assertEqual(report["new_gpu_jobs"], 234)
        self.assertEqual(report["complete_seed_panel_trajectories"], 27)
        self.assertEqual(
            self.manifest["role_counts"],
            {
                "seed_confirmation_trunks": 18,
                "seed_confirmation_branches": 108,
                "seed_confirmation_evaluations": 108,
                "seed_confirmation_summaries": 1,
            },
        )

    def test_only_two_new_seeds_and_exact_factorial_cells_are_trained(self) -> None:
        trunks = [
            row
            for row in self.manifest["jobs"]
            if str(row["job_id"]).startswith("seed_trunk_")
        ]
        identities = {
            (
                int(row["arguments"]["experts"]),
                int(row["arguments"]["K"]),
                int(row["arguments"]["model_seed"]),
            )
            for row in trunks
        }
        self.assertEqual(
            identities,
            {
                (experts, k, seed)
                for experts in SELECTED_EXPERTS
                for k in SELECTED_LANGUAGE_COUNTS
                for seed in NEW_SEEDS
            },
        )
        self.assertNotIn(1234, {identity[2] for identity in identities})
        self.assertEqual(tuple(self.manifest["seed_confirmation_campaign"]["seeds"]["complete_panel"]), COMPLETE_SEEDS)

    def test_jointmoe_x1_and_seed_contract_are_exact(self) -> None:
        training = [
            row
            for row in self.manifest["jobs"]
            if row["action"] == "production_train"
        ]
        self.assertTrue(training)
        self.assertEqual({row["arguments"]["lr_multiplier"] for row in training}, {1.0})
        self.assertEqual(
            {row["arguments"]["learning_rate_policy"] for row in training},
            {"jointmoe_paper_rounded_coefficients_k_invariant"},
        )
        for row in training:
            args = row["arguments"]
            self.assertEqual(args["model_seed"], args["dataset_random_seed"])
            self.assertIn(args["model_seed"], NEW_SEEDS)

        rates: dict[tuple[int, int], float] = {}
        for row in training:
            if row["arguments"]["role"] != "trunk":
                continue
            args = row["arguments"]
            rates[(int(args["experts"]), int(args["K"]))] = float(args["peak_lr"])
        for experts in SELECTED_EXPERTS:
            self.assertEqual(
                {rates[(experts, k)] for k in SELECTED_LANGUAGE_COUNTS},
                {rates[(experts, 1)]},
            )

    def test_full_six_readout_dependency_dag_is_seed_isolated(self) -> None:
        by_id = {str(row["job_id"]): row for row in self.manifest["jobs"]}
        for experts in SELECTED_EXPERTS:
            for k in SELECTED_LANGUAGE_COUNTS:
                for seed in NEW_SEEDS:
                    trunk = f"seed_trunk_b_e{experts}_k{k:03d}_s{seed}"
                    self.assertEqual(by_id[trunk]["dependencies"], [])
                    for horizon in HORIZONS:
                        branch = f"seed_branch_b_e{experts}_k{k:03d}_d{horizon}_s{seed}"
                        evaluation = f"seed_eval_b_e{experts}_k{k:03d}_d{horizon}_s{seed}"
                        self.assertEqual(by_id[branch]["dependencies"], [trunk])
                        self.assertEqual(by_id[branch]["arguments"]["parent_job_id"], trunk)
                        self.assertEqual(by_id[evaluation]["dependencies"], [branch])
                        self.assertEqual(
                            by_id[evaluation]["arguments"]["training_job_id"], branch
                        )

    def test_primary_seed_is_hash_bound_for_reuse_not_recomputed(self) -> None:
        reference = self.manifest["primary_seed_baseline_reference"]
        self.assertEqual(reference["primary_seed"], 1234)
        self.assertEqual(reference["selected_trajectories"], 9)
        self.assertEqual(
            Path(reference["launch_manifest_path"]),
            (REPO_ROOT / BASELINE_MANIFEST).resolve(),
        )
        self.assertFalse(
            any(
                row.get("arguments", {}).get("model_seed") == 1234
                for row in self.manifest["jobs"]
                if row["action"] == "production_train"
            )
        )

    def test_submission_order_releases_heavy_trunks_first(self) -> None:
        ordered = _topological(self.manifest["jobs"])
        first = ordered[:18]
        self.assertTrue(all(str(row["job_id"]).startswith("seed_trunk_") for row in first))
        expert_order = [int(row["arguments"]["experts"]) for row in first]
        self.assertEqual(expert_order[:6], [32] * 6)
        self.assertEqual(expert_order[6:12], [16] * 6)
        self.assertEqual(expert_order[12:], [1] * 6)

    def test_runtime_differs_only_in_identity_and_advisory_storage_policy(self) -> None:
        runtime = _load_object(REPO_ROOT / DEFAULT_RUNTIME)
        parent = _load_object(
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime_jointmoe_x1_production_v1.json"
        )
        for field in ("data", "seeds", "resources", "training", "performance_budget"):
            self.assertEqual(runtime[field], parent[field])
        self.assertEqual(runtime["scheduling"]["predicted_time_multiplier"], 1.35)
        self.assertEqual(runtime["scheduling"]["maximum_job_hours"], 24.0)
        self.assertEqual(
            self.manifest["storage_estimate"]["capacity_enforcement"],
            "advisory_only",
        )
        self.assertEqual(
            self.manifest["storage_estimate"]["go_no_go"],
            "informational_advisory_never_blocks_submission",
        )

    def test_generated_training_overrides_carry_both_seed_controls(self) -> None:
        overrides = _training_overrides(
            experts=16,
            peak_lr=0.0007798024698523467,
            model_seed=5678,
            dataset_random_seed=5678,
            tokenizer_model=Path("/tmp/tokenizer.model"),
        )
        self.assertIn("rng.seed=5678", overrides)
        self.assertIn("dataset.random_seed=5678", overrides)


if __name__ == "__main__":
    unittest.main()
