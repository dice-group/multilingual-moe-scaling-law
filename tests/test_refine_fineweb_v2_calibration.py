from __future__ import annotations

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
    canonical_json_sha256,
)
from multilingual_megatron.pipeline.stages.sampling.tools.refine_fineweb_v2_calibration import (  # noqa: E402
    REFINEMENT_ID,
    refine_calibration,
)


class RefineFineWebV2CalibrationTest(unittest.TestCase):
    @staticmethod
    def _write(path: Path, payload: dict, hash_key: str) -> None:
        payload[hash_key] = canonical_json_sha256(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def test_refines_token_and_rehydration_rates_from_production_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = {
                "schema_version": 1,
                "tool_id": "fineweb_ab_v2_token_calibration_v1",
                "status": "complete",
                "audit": {
                    "effective_calibration_documents_total": 1000,
                    "calibration_tokens_total": 2000,
                    "calibration_rehydrated_tokens_total": 6000,
                },
                "provenance": {"tokenizer_model_sha256": "a" * 64},
                "languages": {
                    "deu_Latn": {
                        "available_documents": 10_000,
                        "available_utf8_bytes": 100_000,
                        "calibration_documents": 1000,
                        "calibration_utf8_bytes": 10_000,
                        "calibration_tokens": 2000,
                        "calibration_rehydrated_tokens": 6000,
                        "validation_fraction": 0.2,
                    }
                },
                "language_audits": {"deu_Latn": {}},
            }
            calibration_path = root / "calibration.json"
            self._write(calibration_path, calibration, "artifact_sha256")
            bootstrap = {
                "schema_version": 1,
                "languages": {
                    "deu_Latn": {
                        "acquisition": {
                            "unique_source_documents": 100,
                            "unique_source_utf8_bytes": 1500,
                            "acquisition_status": "targets_reached",
                            "complete_source_scan": False,
                        }
                    }
                },
            }
            bootstrap_path = root / "bootstrap.json"
            self._write(bootstrap_path, bootstrap, "manifest_sha256")
            manifest = {
                "schema_version": 1,
                "languages": {
                    "deu_Latn": {
                        "bootstrap": {
                            "train": {
                                "documents": 70,
                                "unique_tokens_including_eod": 700,
                                "physical_tokens_including_rehydration_and_eod": 1400,
                            },
                            "validation_reserve": {
                                "documents": 30,
                                "unique_tokens_including_eod": 300,
                                "physical_tokens_including_rehydration_and_eod": 300,
                            },
                        },
                        "validation_candidate_reserve": {
                            "candidate_train_rehydrated_tokens_if_all_reclaimed": 900
                        },
                    }
                },
            }
            manifest_path = root / "finalizers/rank_000/release/corpus_manifest.json"
            self._write(manifest_path, manifest, "manifest_sha256")

            refined = refine_calibration(
                calibration_path=calibration_path,
                bootstrap_manifest_path=bootstrap_path,
                finalizer_root=root / "finalizers",
            )

            language = refined["languages"]["deu_Latn"]
            self.assertEqual(language["calibration_documents"], 100)
            self.assertEqual(language["calibration_utf8_bytes"], 1500)
            self.assertEqual(language["calibration_tokens"], 1000)
            self.assertEqual(language["calibration_rehydrated_tokens"], 2300)
            self.assertEqual(
                refined["provenance"]["production_refinement"]["refinement_id"],
                REFINEMENT_ID,
            )
            expected = refined.pop("artifact_sha256")
            self.assertEqual(expected, canonical_json_sha256(refined))

    def test_missing_finalizer_language_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = {
                "languages": {"deu_Latn": {}},
                "language_audits": {},
                "audit": {},
                "provenance": {},
            }
            calibration_path = root / "calibration.json"
            self._write(calibration_path, calibration, "artifact_sha256")
            bootstrap = {"languages": {"deu_Latn": {}}}
            bootstrap_path = root / "bootstrap.json"
            self._write(bootstrap_path, bootstrap, "manifest_sha256")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                refine_calibration(
                    calibration_path=calibration_path,
                    bootstrap_manifest_path=bootstrap_path,
                    finalizer_root=root / "empty",
                )


if __name__ == "__main__":
    unittest.main()
