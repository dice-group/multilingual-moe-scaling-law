from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
    verify_plan_hash,
)
from multilingual_megatron.pipeline.stages.sampling.tools.build_fineweb_v2_sampling_plan import (
    build_plan_artifact,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class FineWebV2SamplingPlanArtifactTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, list[str], Path, Path]:
        tokenizer = root / "tokenizer.model"
        tokenizer.write_bytes(b"frozen-tokenizer")
        inventory = root / "inventory.csv"
        inventory.write_text("subset,split\n", encoding="utf-8")
        weights = root / "weights.json"
        weights.write_text("{}\n", encoding="utf-8")
        source_lock_hash = "a" * 64
        source_manifest = {
            "schema_version": 1,
            "source_lock_sha256": source_lock_hash,
            "artifacts": {
                "inventory": {"path": str(inventory), "sha256": sha256_file(inventory)},
                "rehydration_weights": {
                    "path": str(weights),
                    "sha256": sha256_file(weights),
                },
            },
        }
        source_manifest["manifest_sha256"] = canonical_json_sha256(source_manifest)
        source_manifest_path = root / "source-manifest.json"
        _write(source_manifest_path, source_manifest)

        compact = {
            "available_utf8_bytes": 10_000,
            "available_documents": 100,
            "calibration_utf8_bytes": 1_000,
            "calibration_tokens": 500,
            "calibration_documents": 10,
            "calibration_rehydrated_tokens": 750,
            "validation_fraction": 0.2,
        }
        calibration = {
            "schema_version": 1,
            "tool_id": "fineweb_ab_v2_token_calibration_v1",
            "status": "complete",
            "languages": {"eng_Latn": compact, "deu_Latn": compact},
            "language_audits": {},
            "provenance": {
                "tokenizer_model_sha256": sha256_file(tokenizer),
                "source_lock_sha256": source_lock_hash,
                "fineweb2_inventory_sha256": sha256_file(inventory),
            },
            "audit": {},
        }
        calibration["artifact_sha256"] = canonical_json_sha256(calibration)
        calibration_path = root / "calibration.json"
        _write(calibration_path, calibration)

        maps: list[str] = []
        for k, languages in ((1, ["eng_Latn"]), (2, ["eng_Latn", "deu_Latn"])):
            path = root / f"k{k}.json"
            _write(path, {"0": {"languages": languages}})
            maps.append(f"K{k}={path}")
        return calibration_path, maps, source_manifest_path, tokenizer

    def test_plan_requires_complete_cross_linked_hash_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration, maps, source_manifest, tokenizer = self._fixture(root)
            plan = build_plan_artifact(
                calibration_path=calibration,
                language_map_args=maps,
                source_manifest_path=source_manifest,
                tokenizer_model_path=tokenizer,
                target_train_tokens=1_000,
                alpha=0.3,
                validation_cap_tokens=20,
            )
            self.assertTrue(verify_plan_hash(plan))
            self.assertEqual(
                plan["provenance"]["calibration_payload_sha256"],
                json.loads(calibration.read_text(encoding="utf-8"))["artifact_sha256"],
            )

            corrupted = json.loads(calibration.read_text(encoding="utf-8"))
            corrupted["languages"]["eng_Latn"]["calibration_tokens"] += 1
            _write(calibration, corrupted)
            with self.assertRaisesRegex(ValueError, "Calibration artifact hash"):
                build_plan_artifact(
                    calibration_path=calibration,
                    language_map_args=maps,
                    source_manifest_path=source_manifest,
                    tokenizer_model_path=tokenizer,
                    target_train_tokens=1_000,
                    alpha=0.3,
                    validation_cap_tokens=20,
                )


if __name__ == "__main__":
    unittest.main()
