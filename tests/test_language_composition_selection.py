"""Focused scientific and lifecycle tests for K12 language composition."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

from multilingual_megatron.experiments.scaling.language_composition.common import (
    PRIMARY_AXES,
    SelectionError,
    language_set_sha256,
    mixture_stats,
    resource_violations,
    seal_receipt,
    verify_receipt,
    write_immutable,
)
from multilingual_megatron.experiments.scaling.language_composition.distances import (
    DistanceBundle,
    angular_distance_matrix,
    equal_weight_composite,
    mean_pairwise_distance,
    normalize_distance_matrix,
)
from multilingual_megatron.experiments.scaling.language_composition.optimization import (
    SolverSettings,
    select_language_sets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_spec(
    compact: list[str],
    diverse: list[str],
) -> dict[str, object]:
    return {
        "selection_policy": {
            "set_size": 2,
            "required_language": "eng_Latn",
            "alpha": 0.3,
            "total_tokens": 2,
            "english_probability_absolute_tolerance": 1e-9,
            "effective_language_count_absolute_tolerance": 1e-9,
        },
        "promotion_policy": {
            "expected_language_set_sha256": {
                "compact": language_set_sha256(compact),
                "diverse": language_set_sha256(diverse),
            }
        },
    }


def _bundle(languages: list[str], matrices: dict[str, np.ndarray]) -> DistanceBundle:
    composite = equal_weight_composite(matrices)
    return DistanceBundle(
        languages=tuple(languages),
        matrices=matrices,
        composite=composite,
        raw_maxima={axis: 1.0 for axis in PRIMARY_AXES},
        matrix_sha256={axis: "fixture" for axis in (*PRIMARY_AXES, "composite")},
    )


class CompositeDistanceTests(unittest.TestCase):
    def test_angular_distance_is_normalized(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        matrix = angular_distance_matrix(vectors)

        self.assertAlmostEqual(matrix[0, 1], 0.5)
        self.assertAlmostEqual(matrix[0, 2], 1.0)
        np.testing.assert_allclose(np.diag(matrix), 0.0)

    def test_composite_is_the_equal_mean_of_exactly_four_axes(self) -> None:
        matrices = {
            axis: np.asarray([[0.0, value], [value, 0.0]])
            for axis, value in zip(
                PRIMARY_AXES, (0.0, 0.2, 0.6, 1.0), strict=True
            )
        }
        composite = equal_weight_composite(matrices)

        self.assertAlmostEqual(composite[0, 1], 0.45)
        self.assertAlmostEqual(mean_pairwise_distance(composite, [0, 1]), 0.45)

    def test_candidate_frame_normalization_uses_each_axis_maximum(self) -> None:
        matrix = np.asarray(
            [[0.0, 0.1, 0.2], [0.1, 0.0, 0.4], [0.2, 0.4, 0.0]]
        )
        normalized, maximum = normalize_distance_matrix(matrix, name="fixture")

        self.assertEqual(maximum, 0.4)
        self.assertAlmostEqual(normalized.max(), 1.0)
        self.assertAlmostEqual(normalized[0, 1], 0.25)

    def test_composite_rejects_a_missing_modality(self) -> None:
        matrices = {
            axis: np.zeros((2, 2))
            for axis in PRIMARY_AXES
            if axis != "geography"
        }
        with self.assertRaisesRegex(SelectionError, "requires exactly"):
            equal_weight_composite(matrices)


class ExactSelectionTests(unittest.TestCase):
    languages = ["eng_Latn", "aaa_Latn", "bbb_Cyrl", "ccc_Hani"]
    reference = ["eng_Latn", "bbb_Cyrl"]
    compact = ["eng_Latn", "aaa_Latn"]
    diverse = ["eng_Latn", "ccc_Hani"]
    availability = {language: 100 for language in languages}

    def _select(self, matrices: dict[str, np.ndarray]) -> dict[str, object]:
        return select_language_sets(
            spec=_fixture_spec(self.compact, self.diverse),
            languages=self.languages,
            reference_languages=self.reference,
            availability=self.availability,
            bundle=_bundle(self.languages, matrices),
            settings=SolverSettings(),
        )

    def test_independent_exact_solves_select_both_extrema(self) -> None:
        coordinates = np.asarray([0.0, 0.1, 0.6, 1.0])
        base = np.abs(coordinates[:, None] - coordinates[None, :])
        result = self._select({axis: base.copy() for axis in PRIMARY_AXES})

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["compact"]["languages"], self.compact)
        self.assertEqual(result["diverse"]["languages"], self.diverse)
        self.assertTrue(result["gates"]["all_directional_gates_passed"])
        self.assertEqual(
            set(result["gates"]["directional"]),
            {"composite", *PRIMARY_AXES},
        )
        self.assertTrue(
            all(trace["mip_gap"] == 0.0 for trace in result["solver_trace"])
        )

    def test_one_failed_axis_blocks_composite_promotion(self) -> None:
        coordinates = np.asarray([0.0, 0.1, 0.6, 1.0])
        base = np.abs(coordinates[:, None] - coordinates[None, :])
        matrices = {axis: base.copy() for axis in PRIMARY_AXES}
        matrices["geography"][0, 2] = matrices["geography"][2, 0] = 0.8
        matrices["geography"][0, 3] = matrices["geography"][3, 0] = 0.7

        result = self._select(matrices)

        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["gates"]["directional"]["composite"]["passed"])
        self.assertFalse(result["gates"]["directional"]["geography"]["passed"])
        self.assertFalse(result["gates"]["all_directional_gates_passed"])

    def test_membership_lock_is_a_separate_promotion_gate(self) -> None:
        coordinates = np.asarray([0.0, 0.1, 0.6, 1.0])
        base = np.abs(coordinates[:, None] - coordinates[None, :])
        spec = _fixture_spec(self.compact, self.diverse)
        spec["promotion_policy"]["expected_language_set_sha256"]["compact"] = "0" * 64

        result = select_language_sets(
            spec=spec,
            languages=self.languages,
            reference_languages=self.reference,
            availability=self.availability,
            bundle=_bundle(
                self.languages,
                {axis: base.copy() for axis in PRIMARY_AXES},
            ),
        )

        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["gates"]["all_directional_gates_passed"])
        self.assertFalse(result["gates"]["expected_memberships_passed"])

    def test_milp_resource_constraints_match_brute_force_enumeration(self) -> None:
        languages = [
            "eng_Latn",
            "aaa_Latn",
            "bbb_Latn",
            "ccc_Latn",
            "ddd_Latn",
            "eee_Latn",
        ]
        reference = ["eng_Latn", "aaa_Latn", "ccc_Latn"]
        availability = dict(
            zip(languages, (1000, 400, 200, 100, 40, 20), strict=True)
        )
        policy = {
            "set_size": 3,
            "required_language": "eng_Latn",
            "alpha": 0.3,
            "total_tokens": 200,
            "english_probability_absolute_tolerance": 0.08,
            "effective_language_count_absolute_tolerance": 0.25,
        }
        coordinates = np.asarray([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
        base = np.abs(coordinates[:, None] - coordinates[None, :])
        bundle = _bundle(
            languages, {axis: base.copy() for axis in PRIMARY_AXES}
        )
        reference_stats = mixture_stats(
            reference,
            availability,
            alpha=policy["alpha"],
            total_tokens=policy["total_tokens"],
            required_language=policy["required_language"],
        )
        feasible: list[tuple[list[str], float]] = []
        for others in combinations(languages[1:], 2):
            members = [languages[0], *others]
            stats = mixture_stats(
                members,
                availability,
                alpha=policy["alpha"],
                total_tokens=policy["total_tokens"],
                required_language=policy["required_language"],
            )
            if not resource_violations(
                stats,
                reference=reference_stats,
                english_tolerance=policy[
                    "english_probability_absolute_tolerance"
                ],
                effective_language_count_tolerance=policy[
                    "effective_language_count_absolute_tolerance"
                ],
            ):
                indices = [languages.index(language) for language in members]
                feasible.append(
                    (members, mean_pairwise_distance(bundle.composite, indices))
                )
        self.assertGreater(len(feasible), 1)
        compact = min(feasible, key=lambda item: item[1])[0]
        diverse = max(feasible, key=lambda item: item[1])[0]
        self.assertNotEqual(compact, diverse)
        spec = {
            "selection_policy": policy,
            "promotion_policy": {
                "expected_language_set_sha256": {
                    "compact": language_set_sha256(compact),
                    "diverse": language_set_sha256(diverse),
                }
            },
        }

        result = select_language_sets(
            spec=spec,
            languages=languages,
            reference_languages=reference,
            availability=availability,
            bundle=bundle,
        )

        self.assertEqual(result["compact"]["languages"], compact)
        self.assertEqual(result["diverse"]["languages"], diverse)
        self.assertTrue(result["gates"]["resource_constraints_passed"])


class ReceiptLifecycleTests(unittest.TestCase):
    def test_receipt_self_hash_detects_mutation(self) -> None:
        receipt = seal_receipt(
            {
                "schema": "language_composition_selection_receipt/v2",
                "status": "passed",
            }
        )
        verify_receipt(receipt)
        receipt["status"] = "blocked"
        with self.assertRaisesRegex(SelectionError, "self-hash"):
            verify_receipt(receipt)

    def test_immutable_writer_refuses_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            write_immutable(path, b"first\n")
            write_immutable(path, b"first\n")
            with self.assertRaisesRegex(SelectionError, "Refusing to overwrite"):
                write_immutable(path, b"second\n")


class CheckedInContractTests(unittest.TestCase):
    def test_main_spec_and_groupings_encode_only_the_strict_v2_design(self) -> None:
        spec_path = (
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/"
            "language_composition_selection_v2.json"
        )
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        self.assertEqual(spec["schema"], "language_composition_selection/v2")
        self.assertEqual(spec["metric_policy"]["primary_axes"], list(PRIMARY_AXES))
        self.assertEqual(
            spec["selection_policy"]["english_probability_absolute_tolerance"],
            0.005,
        )
        self.assertEqual(
            spec["selection_policy"][
                "effective_language_count_absolute_tolerance"
            ],
            0.25,
        )
        self.assertEqual(spec["selection_policy"]["alpha"], 0.3)
        self.assertEqual(spec["selection_policy"]["total_tokens"], 32_000_000_000)
        self.assertTrue(
            spec["selection_policy"]["no_additional_corpus_cycling"]
        )
        self.assertEqual(
            spec["promotion_policy"]["directional_gates"],
            ["composite", *PRIMARY_AXES],
        )
        projection_definition = spec["availability_projection"]
        projection_path = REPO_ROOT / projection_definition["path"]
        self.assertEqual(
            hashlib.sha256(projection_path.read_bytes()).hexdigest(),
            projection_definition["file_sha256"],
        )
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        source_grouping = json.loads(
            (REPO_ROOT / spec["source_frame"]["path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(projection["schema"], "language_composition_availability/v1")
        self.assertEqual(len(projection["availability_tokens"]), 200)
        self.assertEqual(
            set(projection["availability_tokens"]),
            set(source_grouping["0"]["languages"]),
        )
        self.assertEqual(
            projection["source_sampling_plan"]["sha256"],
            "5ebe0760ee2868828c339760d7f14144a1ad036a0b992ee30775e4510f81b0dc",
        )
        expected = spec["promotion_policy"]["expected_language_set_sha256"]
        for arm in ("compact", "diverse"):
            grouping = json.loads(
                (
                    REPO_ROOT
                    / f"configs/language_groupings/scaling_fw2_v2_k012_{arm}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(grouping["language_set_sha256"], expected[arm])
            self.assertEqual(
                language_set_sha256(grouping["0"]["languages"]), expected[arm]
            )
            self.assertEqual(
                grouping["0"]["selection_policy"]["method_id"],
                "urielplus_k196_four_axis_minsum_maxsum_v2",
            )

        config_dir = spec_path.parent
        self.assertEqual(
            [path.name for path in config_dir.glob("language_composition_selection*.json")],
            ["language_composition_selection_v2.json"],
        )
        self.assertFalse(
            (
                REPO_ROOT
                / "src/multilingual_megatron/experiments/scaling/"
                "select_language_composition_v2.py"
            ).exists()
        )

    def test_checked_in_receipt_is_sealed_and_points_to_both_groupings(self) -> None:
        manifest_path = (
            REPO_ROOT
            / "result_analysis/language_composition_selection_v2/"
            "selection_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_receipt(manifest)

        self.assertEqual(manifest["status"], "passed")
        self.assertTrue(
            manifest["selection"]["gates"]["all_directional_gates_passed"]
        )
        self.assertTrue(
            all(
                trace["mip_gap"] == 0.0
                for trace in manifest["solver_certificate"]["trace"]
            )
        )
        for record in manifest["files"].values():
            path = REPO_ROOT / record["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"]
            )
            grouping = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                grouping["0"]["selection_policy"]["selection_identity_sha256"],
                manifest["selection_identity_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
