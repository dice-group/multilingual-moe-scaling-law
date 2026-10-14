from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.ab_v2.spec import (  # noqa: E402
    SpecValidationError,
    language_set_sha256,
    validate_ab_v2_spec,
)
from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (  # noqa: E402
    K1_CODES,
    K12_CODES,
    K50_CODES,
    SHORT_CODE_TO_FINEWEB2_SUBSET,
)


EXPECTED_HASHES = {
    1: "301b2b693fa2af539ef72f4f59710b2a21928b584b83bc6cb65d46e86a933272",
    12: "bbff51cc58457e0e22b228df8bfdc34aa05c91e17acebcd4bd64de981c35839d",
    50: "c4afc84137c08ea89255f05008ab9970d9aa134aafb0cdad651c8dfaad7fdd64",
    200: "5e142af3e289e53120f458a051bcbe324905ff1454ceb5e3ee8376608d15df96",
}


def _load_languages(k: int) -> list[str]:
    path = REPO_ROOT / "configs" / "language_groupings" / f"scaling_fw2_v2_k{k:03d}.json"
    return json.loads(path.read_text(encoding="utf-8"))["0"]["languages"]


class ABV2SpecTest(unittest.TestCase):
    def test_full_frozen_spec_validates(self) -> None:
        result = validate_ab_v2_spec(repo_root=REPO_ROOT)

        self.assertEqual(result["language_counts"], {"K1": 1, "K12": 12, "K50": 50, "K200": 200})
        self.assertEqual(result["source_languages_verified"], 200)
        self.assertEqual(result["language_set_sha256"], {f"K{k}": value for k, value in EXPECTED_HASHES.items()})

    def test_tokenizer_acquisition_uses_exact_train_slot_gate(self) -> None:
        spec = json.loads(
            (REPO_ROOT / "configs/scaling_law/multilingual_moe_ab_v2/study_spec.json").read_text(
                encoding="utf-8"
            )
        )
        policy = spec["tokenizer_policy"]

        self.assertIn("1.10 document/byte planning floor", policy["acquisition_protocol"])
        self.assertIn("exact train-side weighted sentence-slot stopping", policy["acquisition_protocol"])
        self.assertEqual(
            policy["sentence_slot_counter"],
            "sum over train documents of deterministic sentence count times the pinned within-language rehydration multiplier",
        )
        self.assertEqual(
            policy["non_exhausted_stop_rule"],
            "document, byte, calibration, and required weighted sentence-slot minima must all be met",
        )

    def test_atlas_membership_and_top150_suffix_are_exact(self) -> None:
        source_lock = json.loads(
            (REPO_ROOT / "configs/data_sources/fineweb2_v2_1_1_source_lock.json").read_text(encoding="utf-8")
        )
        resolutions = source_lock["atlas_mapping_resolutions"]

        def mapped(codes: tuple[str, ...]) -> list[str]:
            return [
                SHORT_CODE_TO_FINEWEB2_SUBSET[code] or resolutions[code]["subset"]
                for code in codes
            ]

        k1 = _load_languages(1)
        k12 = _load_languages(12)
        k50 = _load_languages(50)
        k200 = _load_languages(200)
        audit_path = (
            REPO_ROOT
            / "configs/data_sources/atlas50_fineweb2_top150_by_documents.csv"
        )
        with audit_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            top150 = [row["subset"] for row in csv.DictReader(handle)]

        self.assertEqual(k1, mapped(K1_CODES))
        self.assertEqual(k12, mapped(K12_CODES))
        self.assertEqual(k50, mapped(K50_CODES))
        self.assertEqual(k200, k50 + top150)

    def test_sets_are_nested_and_have_one_script_per_iso_language(self) -> None:
        sets = {k: _load_languages(k) for k in EXPECTED_HASHES}
        for lower, upper in zip((1, 12, 50), (12, 50, 200)):
            self.assertLessEqual(set(sets[lower]), set(sets[upper]))
        for k, languages in sets.items():
            iso_codes = [language.split("_", 1)[0] for language in languages]
            self.assertEqual(len(languages), k)
            self.assertEqual(len(set(languages)), k)
            self.assertEqual(len(set(iso_codes)), k)

    def test_ordered_language_hashes_are_frozen(self) -> None:
        for k, expected in EXPECTED_HASHES.items():
            self.assertEqual(language_set_sha256(_load_languages(k)), expected)

    def test_missing_source_mapping_fails_instead_of_substituting(self) -> None:
        available = set(_load_languages(200))
        available.remove("hau_Latn")

        with self.assertRaisesRegex(SpecValidationError, "hau_Latn.*refusing silent substitution"):
            validate_ab_v2_spec(repo_root=REPO_ROOT, available_source_subsets=available)

    def test_source_lock_pins_corpora_inventory_and_rehydration(self) -> None:
        source_lock = json.loads(
            (REPO_ROOT / "configs/data_sources/fineweb2_v2_1_1_source_lock.json").read_text(encoding="utf-8")
        )
        non_english = source_lock["non_english_corpus"]
        english = source_lock["english_corpus"]

        self.assertEqual(non_english["dataset_revision"], "af9c13333eb981300149d5ca60a8e9d659b276b9")
        self.assertEqual(
            non_english["language_inventory"]["sha256"],
            "0c6e6fbcef62cb034aff493187dee8d327c6bb2c08bb14a81a9c7221b091e0a5",
        )
        self.assertEqual(
            non_english["rehydration"]["weights_sha256"],
            "bf4db6767b9766ef32a27b5888dd9c287fc03d06e69d64c21c8410d002ab0c55",
        )
        self.assertEqual(
            non_english["rehydration"]["missing_weights_policy"],
            "fail_closed_except_explicit_identity_overrides",
        )
        self.assertEqual(
            non_english["rehydration"]["identity_overrides"]["hau_Latn"]["thresholds"],
            {"1": 1},
        )
        self.assertFalse(
            non_english["rehydration"]["identity_overrides"]["hau_Latn"][
                "upstream_distribution_present"
            ]
        )
        self.assertEqual(english["dataset_revision"], "9bb295ddab0e05d785b879661af7260fed5140fc")
        self.assertFalse(english["rehydration"]["enabled"])
        self.assertEqual(source_lock["atlas_mapping_resolutions"]["ha"]["subset"], "hau_Latn")

    def test_scientific_contract_drift_fails_closed(self) -> None:
        spec = json.loads(
            (REPO_ROOT / "configs/scaling_law/multilingual_moe_ab_v2/study_spec.json").read_text(
                encoding="utf-8"
            )
        )
        spec["data_policy"]["language_sampling"]["alpha"] = 0.7
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "drifted-study-spec.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(
                SpecValidationError,
                "scientific contract drift:",
            ):
                validate_ab_v2_spec(repo_root=REPO_ROOT, spec_path=path)


if __name__ == "__main__":
    unittest.main()
