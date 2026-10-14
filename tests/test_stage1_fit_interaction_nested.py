from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from multilingual_megatron.experiments.scaling.stage1_fit.interaction_nested import (
    INNER_AXES,
    LADDERS,
    _anchored_starts,
    _membership_payload,
    _paired_outer_comparison,
    parser,
)


M2_COORDINATES = np.asarray([
    0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.02, -0.04, 0.03,
], dtype=np.float64)


def _m2_result() -> dict[str, object]:
    attempts = []
    for index in range(12):
        coordinates = M2_COORDINATES.copy()
        coordinates[index % coordinates.size] += (index + 1) * 1e-4
        attempts.append({"success": True, "objective": 1.0 + index * 1e-8,
            "start_id": f"grid:{index:06d}", "coordinates": coordinates.tolist()})
    return {"manifest_path": "patched", "selected": {"coordinates": M2_COORDINATES.tolist()},
        "attempts": attempts}


def _holdouts() -> dict[str, object]:
    trajectories = sorted(
        f"{ladder}:E{experts}:K{languages}"
        for ladder in LADDERS for experts in (1, 4, 16, 32) for languages in (1, 12, 50, 200)
    )
    all_languages = ["eng_Latn", "deu_Latn", "fra_Latn"]
    common = ["eng_Latn", "deu_Latn"]
    primary = []
    for fold_id, test, score in (
        ("largest_N_act", [item for item in trajectories if item.startswith("D:")], all_languages),
        ("E32", [item for item in trajectories if ":E32:" in item], all_languages),
        ("K200_common_languages", [item for item in trajectories if item.endswith(":K200")], common),
    ):
        primary.append({"fold_id": fold_id, "train_trajectory_ids": sorted(set(trajectories) - set(test)),
            "test_trajectory_ids": sorted(test), "score_languages": score})
    return {"manifest_sha256": "source", "primary_folds": primary}


class Stage1NestedInteractionTest(unittest.TestCase):
    def test_memberships_are_four_outer_by_three_inner_partitions(self) -> None:
        payload = _membership_payload(_holdouts(), input_sha256="input")
        ids = [key for key in payload if key.startswith("outer_")]
        self.assertEqual(len(ids), 16)
        for outer in LADDERS:
            outer_item = payload[f"outer_{outer}"]
            self.assertEqual(len(outer_item["train_trajectory_ids"]), 48)
            self.assertEqual(len(outer_item["test_trajectory_ids"]), 16)
            self.assertTrue(all(not item.startswith(f"{outer}:") for item in outer_item["train_trajectory_ids"]))
            for axis in INNER_AXES:
                inner = payload[f"outer_{outer}__{axis}"]
                self.assertFalse(set(inner["train_trajectory_ids"]) & set(inner["test_trajectory_ids"]))
                self.assertTrue(all(not item.startswith(f"{outer}:")
                    for item in inner["train_trajectory_ids"] + inner["test_trajectory_ids"]))
                expected_test = 16 if axis == "largest_remaining" else 12
                self.assertEqual(len(inner["test_trajectory_ids"]), expected_test)
                self.assertEqual(len(inner["train_trajectory_ids"]), 48 - expected_test)

    def test_retained_start_counts_are_frozen_and_nested(self) -> None:
        with patch("multilingual_megatron.experiments.scaling.stage1_fit.interaction.load_manifest",
                return_value={"fit": {"model": "M2"}}):
            expected = {"M2": 8, "M2-KN": 24, "M2-KD": 24, "M2-KND": 72}
            for model, count in expected.items():
                starts = _anchored_starts(model, _m2_result(), namespace="test")
                self.assertEqual(len(starts), count)
                self.assertEqual(len({item.start_id for item in starts}), count)
                expected_width = 10 + int(model in {"M2-KN", "M2-KND"}) \
                    + int(model in {"M2-KD", "M2-KND"})
                self.assertTrue(all(item.values.shape == (expected_width,) for item in starts))

    def test_paired_outer_comparison_uses_shared_resamples(self) -> None:
        baseline = {f"t{index}": 1.0 + index / 100.0 for index in range(16)}
        improved = {key: value * 0.81 for key, value in baseline.items()}
        first = _paired_outer_comparison(baseline, improved, replicates=1000, seed=7)
        second = _paired_outer_comparison(baseline, improved, replicates=1000, seed=7)
        self.assertEqual(first, second)
        self.assertLess(first["difference_procedure_minus_m2"], 0.0)
        self.assertEqual(first["probability_procedure_lower"], 1.0)

    def test_cli_exposes_bounded_nested_waves(self) -> None:
        parsed = parser().parse_args(["plan-scales", "--screen-index", "i", "--screen-report", "r", "--dry-run"])
        self.assertEqual(parsed.command, "plan-scales")
        self.assertTrue(parsed.dry_run)


if __name__ == "__main__":
    unittest.main()
