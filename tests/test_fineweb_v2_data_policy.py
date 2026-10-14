from __future__ import annotations

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


from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    LanguageCalibration,
    alpha_probabilities,
    assign_validation_reserve,
    build_nested_sampling_plan,
    largest_remainder_quotas,
    rehydration_multiplier,
    stable_document_id,
    verify_plan_hash,
)
from multilingual_megatron.pipeline.stages.sampling.tools.materialize_fineweb_v2_source_lock import (  # noqa: E402
    materialize_source_lock,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FineWebV2DataPolicyTest(unittest.TestCase):
    def test_document_ids_and_validation_assignment_are_deterministic(self) -> None:
        document_id = stable_document_id(
            source_dataset="HuggingFaceFW/fineweb-2",
            source_revision="abc123",
            subset="deu_Latn",
            source_id="doc-7",
            url="https://example.test/7",
            text="Hallo Welt.",
        )
        same = stable_document_id(
            source_dataset="HuggingFaceFW/fineweb-2",
            source_revision="abc123",
            subset="deu_Latn",
            source_id="doc-7",
            url="https://example.test/7",
            text="Hallo Welt.",
        )
        changed = stable_document_id(
            source_dataset="HuggingFaceFW/fineweb-2",
            source_revision="abc123",
            subset="deu_Latn",
            source_id="doc-7",
            url="https://example.test/7",
            text="Andere Welt.",
        )
        self.assertEqual(document_id, same)
        self.assertNotEqual(document_id, changed)
        self.assertEqual(
            assign_validation_reserve(document_id=document_id, seed=42),
            assign_validation_reserve(document_id=document_id, seed=42),
        )

    def test_rehydration_uses_last_threshold_not_exceeding_cluster_size(self) -> None:
        weights = {"1": 1, "2": 3, "5": 7, "10": 2}
        self.assertEqual(rehydration_multiplier(1, weights), 1)
        self.assertEqual(rehydration_multiplier(2, weights), 3)
        self.assertEqual(rehydration_multiplier(4, weights), 3)
        self.assertEqual(rehydration_multiplier(9, weights), 7)
        self.assertEqual(rehydration_multiplier(1000, weights), 2)

    def test_alpha_probabilities_and_largest_remainder_preserve_exact_total(self) -> None:
        probabilities = alpha_probabilities({"a": 1, "b": 1000}, alpha=0.3)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["b"], probabilities["a"])
        self.assertLess(probabilities["b"], 0.9)
        quotas = largest_remainder_quotas(101, probabilities)
        self.assertEqual(sum(quotas.values()), 101)

    def test_nested_plan_uses_maximum_tier_quota_and_reports_repetition(self) -> None:
        calibrations = {
            "eng_Latn": LanguageCalibration(
                language="eng_Latn",
                available_utf8_bytes=10_000,
                available_documents=100,
                calibration_utf8_bytes=100,
                calibration_tokens=50,
                calibration_documents=10,
                calibration_rehydrated_tokens=50,
            ),
            "deu_Latn": LanguageCalibration(
                language="deu_Latn",
                available_utf8_bytes=1_000,
                available_documents=20,
                calibration_utf8_bytes=100,
                calibration_tokens=50,
                calibration_documents=10,
                calibration_rehydrated_tokens=250,
            ),
        }
        plan = build_nested_sampling_plan(
            calibrations=calibrations,
            language_sets={1: ["eng_Latn"], 2: ["eng_Latn", "deu_Latn"]},
            train_tokens_per_set=10_000,
            validation_cap_tokens=100,
        )
        self.assertTrue(verify_plan_hash(plan))
        self.assertEqual(sum(plan["tiers"]["K1"]["target_exposure_tokens"].values()), 10_000)
        self.assertEqual(sum(plan["tiers"]["K2"]["target_exposure_tokens"].values()), 10_000)
        english = plan["physical_pool"]["eng_Latn"]
        self.assertEqual(english["requested_max_tier_tokens"], 10_000)
        self.assertTrue(english["training_repeat_required"])
        self.assertLessEqual(english["planned_source_documents"], 100)

    def test_alpha_and_physical_pool_use_rehydrated_train_availability(self) -> None:
        calibrations = {
            "a_Latn": LanguageCalibration(
                language="a_Latn",
                available_utf8_bytes=10_000,
                available_documents=100,
                calibration_utf8_bytes=100,
                calibration_tokens=100,
                calibration_documents=10,
                calibration_rehydrated_tokens=100,
            ),
            "b_Latn": LanguageCalibration(
                language="b_Latn",
                available_utf8_bytes=10_000,
                available_documents=100,
                calibration_utf8_bytes=100,
                calibration_tokens=100,
                calibration_documents=10,
                calibration_rehydrated_tokens=1_000,
            ),
        }

        plan = build_nested_sampling_plan(
            calibrations=calibrations,
            language_sets={2: ["a_Latn", "b_Latn"]},
            train_tokens_per_set=1_000_000,
            validation_cap_tokens=100,
        )

        tier = plan["tiers"]["K2"]
        self.assertEqual(
            tier["availability_basis"],
            "estimated_train_tokens_after_fineweb2_rehydration",
        )
        self.assertEqual(tier["availability_tokens"]["b_Latn"], 10 * tier["availability_tokens"]["a_Latn"])
        self.assertGreater(tier["probabilities"]["b_Latn"], tier["probabilities"]["a_Latn"])
        self.assertEqual(
            plan["physical_pool"]["b_Latn"]["estimated_train_rehydrated_tokens"],
            tier["availability_tokens"]["b_Latn"],
        )

    def test_train_availability_uses_the_capped_validation_rule(self) -> None:
        low = LanguageCalibration(
            language="low_Latn",
            available_utf8_bytes=400,
            available_documents=100,
            calibration_utf8_bytes=100,
            calibration_tokens=100,
            calibration_documents=10,
            calibration_rehydrated_tokens=100,
        )
        high = LanguageCalibration(
            language="high_Latn",
            available_utf8_bytes=2_000,
            available_documents=100,
            calibration_utf8_bytes=100,
            calibration_tokens=100,
            calibration_documents=10,
            calibration_rehydrated_tokens=100,
        )
        plan = build_nested_sampling_plan(
            calibrations={"low_Latn": low, "high_Latn": high},
            language_sets={2: ["low_Latn", "high_Latn"]},
            train_tokens_per_set=10_000,
            validation_cap_tokens=100,
        )

        low_pool = plan["physical_pool"]["low_Latn"]
        high_pool = plan["physical_pool"]["high_Latn"]
        self.assertEqual(low_pool["estimated_validation_unique_tokens"], 80)
        self.assertEqual(low_pool["estimated_train_unique_tokens"], 320)
        self.assertEqual(low_pool["estimated_train_fraction"], 0.8)
        self.assertEqual(high_pool["estimated_validation_unique_tokens"], 100)
        self.assertEqual(high_pool["estimated_train_unique_tokens"], 1_900)
        self.assertEqual(high_pool["estimated_train_fraction"], 0.95)
        self.assertEqual(
            high_pool["estimated_train_unique_tokens"]
            + high_pool["estimated_validation_unique_tokens"],
            high_pool["estimated_unique_tokens"],
        )
        self.assertEqual(
            plan["availability_contract"]["train_unique_rule"],
            "estimated_unique_tokens - validation_target_tokens",
        )

    def test_exact_exhausted_train_inventory_overrides_rate_estimate(self) -> None:
        calibration = LanguageCalibration(
            language="low_Latn",
            available_utf8_bytes=400,
            available_documents=100,
            calibration_utf8_bytes=400,
            calibration_tokens=400,
            calibration_documents=100,
            calibration_rehydrated_tokens=800,
            exact_train_rehydrated_tokens=617,
            exact_train_rehydrated_validation_cap_tokens=100,
        )
        self.assertEqual(
            calibration.estimated_train_rehydrated_tokens_for_cap(100), 617
        )
        self.assertNotEqual(
            calibration.estimated_train_rehydrated_tokens_for_cap(50), 617
        )

    def test_source_materialization_is_hash_verified_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory_bytes = b"subset,split,documents\nhau_Latn,train,1\n"
            weights_bytes = b'{"hau_Latn": {"1": 1}}\n'
            inventory_source = root / "inventory-source.csv"
            weights_source = root / "weights-source.json"
            inventory_source.write_bytes(inventory_bytes)
            weights_source.write_bytes(weights_bytes)
            lock = {
                "source_lock_id": "test",
                "non_english_corpus": {
                    "dataset_revision": "fineweb2-rev",
                    "language_inventory": {
                        "url": inventory_source.as_uri(),
                        "sha256": _sha256(inventory_bytes),
                    },
                    "rehydration": {
                        "weights_url": weights_source.as_uri(),
                        "weights_sha256": _sha256(weights_bytes),
                    },
                },
                "english_corpus": {"dataset_revision": "fineweb-rev"},
            }
            lock_path = root / "source-lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            output = root / "materialized"

            first = materialize_source_lock(source_lock_path=lock_path, output_dir=output)
            second = materialize_source_lock(source_lock_path=lock_path, output_dir=output)

            self.assertEqual(first, second)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(
                (output / "fineweb2-language-distribution.csv").read_bytes(),
                inventory_bytes,
            )


if __name__ == "__main__":
    unittest.main()
