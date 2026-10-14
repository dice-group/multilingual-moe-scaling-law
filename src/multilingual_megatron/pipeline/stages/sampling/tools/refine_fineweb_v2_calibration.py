#!/usr/bin/env python3
"""Refine FineWeb token availability from exact production finalizer audits."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)


SCHEMA_VERSION = 1
REFINEMENT_ID = "fineweb_ab_v2_production_refinement_v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_payload(payload: Mapping[str, Any], key: str, label: str) -> str:
    candidate = dict(payload)
    expected = candidate.pop(key, None)
    if not isinstance(expected, str) or expected != canonical_json_sha256(candidate):
        raise ValueError(f"{label} payload hash mismatch")
    return expected


def _write_frozen(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"refusing to replace different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collect_language_audits(
    finalizer_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    languages: dict[str, dict[str, Any]] = {}
    lineage: dict[str, dict[str, str]] = {}
    for manifest_path in sorted(finalizer_root.glob("rank_*/release/corpus_manifest.json")):
        manifest = _load(manifest_path)
        manifest_hash = _verify_payload(
            manifest, "manifest_sha256", f"finalizer manifest {manifest_path}"
        )
        for language, raw in manifest.get("languages", {}).items():
            if language in languages:
                raise ValueError(f"duplicate finalizer language: {language}")
            if not isinstance(raw, dict):
                raise ValueError(f"invalid finalizer language audit: {language}")
            languages[str(language)] = raw
            lineage[str(language)] = {
                "manifest_path": str(manifest_path.resolve()),
                "manifest_file_sha256": sha256_file(manifest_path),
                "manifest_payload_sha256": manifest_hash,
            }
    return languages, lineage


def refine_calibration(
    *,
    calibration_path: Path,
    bootstrap_manifest_path: Path,
    finalizer_root: Path,
) -> dict[str, Any]:
    calibration_path = calibration_path.resolve()
    bootstrap_manifest_path = bootstrap_manifest_path.resolve()
    finalizer_root = finalizer_root.resolve()
    original = _load(calibration_path)
    original_hash = _verify_payload(original, "artifact_sha256", "token calibration")
    bootstrap = _load(bootstrap_manifest_path)
    bootstrap_hash = _verify_payload(bootstrap, "manifest_sha256", "bootstrap manifest")
    audits, manifest_lineage = _collect_language_audits(finalizer_root)
    original_languages = original.get("languages")
    bootstrap_languages = bootstrap.get("languages")
    if not isinstance(original_languages, dict) or not isinstance(bootstrap_languages, dict):
        raise ValueError("calibration/bootstrap language objects are missing")
    expected = set(map(str, original_languages))
    if set(map(str, bootstrap_languages)) != expected:
        raise ValueError("calibration and bootstrap language coverage differ")
    missing = sorted(expected - set(audits))
    extra = sorted(set(audits) - expected)
    if missing or extra:
        raise ValueError(f"finalizer language coverage mismatch: missing={missing}, extra={extra}")

    refined_languages: dict[str, Any] = {}
    refined_audits: dict[str, Any] = json.loads(
        json.dumps(original.get("language_audits", {}))
    )
    total_documents = 0
    total_tokens = 0
    total_rehydrated = 0
    for language in sorted(expected):
        prior = original_languages[language]
        raw = audits[language]
        split_audit = raw.get("bootstrap")
        candidate = raw.get("validation_candidate_reserve")
        coverage = bootstrap_languages[language].get("acquisition")
        if (
            not isinstance(prior, dict)
            or not isinstance(split_audit, dict)
            or not isinstance(candidate, dict)
            or not isinstance(coverage, dict)
        ):
            raise ValueError(f"incomplete refinement evidence: {language}")
        train = split_audit.get("train")
        reserve = split_audit.get("validation_reserve")
        if not isinstance(train, dict) or not isinstance(reserve, dict):
            raise ValueError(f"incomplete bootstrap token audit: {language}")
        documents = int(train.get("documents", 0)) + int(reserve.get("documents", 0))
        utf8_bytes = int(coverage.get("unique_source_utf8_bytes", 0))
        unique_tokens = int(train.get("unique_tokens_including_eod", 0)) + int(
            reserve.get("unique_tokens_including_eod", 0)
        )
        rehydrated_tokens = int(
            train.get("physical_tokens_including_rehydration_and_eod", 0)
        ) + int(candidate.get("candidate_train_rehydrated_tokens_if_all_reclaimed", 0))
        if min(documents, utf8_bytes, unique_tokens, rehydrated_tokens) <= 0:
            raise ValueError(f"non-positive production refinement measurement: {language}")
        if documents != int(coverage.get("unique_source_documents", -1)):
            raise ValueError(f"bootstrap/finalizer document mismatch: {language}")
        refined_languages[language] = {
            "available_documents": int(prior["available_documents"]),
            "available_utf8_bytes": int(prior["available_utf8_bytes"]),
            "calibration_documents": documents,
            "calibration_utf8_bytes": utf8_bytes,
            "calibration_tokens": unique_tokens,
            "calibration_rehydrated_tokens": rehydrated_tokens,
            "validation_fraction": float(prior["validation_fraction"]),
        }
        if coverage.get("acquisition_status") in {
            "pinned_inventory_exhausted",
            "source_exhausted",
        } or bool(coverage.get("complete_source_scan")):
            exact_train = int(raw.get("train", {}).get("available_tokens", 0))
            if exact_train <= 0:
                raise ValueError(f"missing exact exhausted train inventory: {language}")
            refined_languages[language].update(
                {
                    "exact_train_rehydrated_tokens": exact_train,
                    "exact_train_rehydrated_validation_cap_tokens": 20_000_000,
                }
            )
        audit = refined_audits.setdefault(language, {})
        audit["production_refinement"] = {
            "refinement_id": REFINEMENT_ID,
            "bootstrap_acquisition_status": coverage.get("acquisition_status"),
            "complete_source_scan": bool(coverage.get("complete_source_scan")),
            "measurements": {
                "documents": documents,
                "utf8_bytes": utf8_bytes,
                "unique_tokens_including_eod": unique_tokens,
                "train_rehydrated_tokens_before_final_validation": rehydrated_tokens,
            },
            **manifest_lineage[language],
        }
        total_documents += documents
        total_tokens += unique_tokens
        total_rehydrated += rehydrated_tokens

    artifact = json.loads(json.dumps(original))
    artifact.pop("artifact_sha256", None)
    artifact["languages"] = refined_languages
    artifact["language_audits"] = refined_audits
    artifact["audit"].update(
        {
            "calibration_basis": "exact_large_production_sample",
            "effective_calibration_documents_total": total_documents,
            "calibration_tokens_total": total_tokens,
            "calibration_rehydrated_tokens_total": total_rehydrated,
        }
    )
    artifact["provenance"]["production_refinement"] = {
        "refinement_id": REFINEMENT_ID,
        "original_calibration_path": str(calibration_path),
        "original_calibration_file_sha256": sha256_file(calibration_path),
        "original_calibration_payload_sha256": original_hash,
        "bootstrap_manifest_path": str(bootstrap_manifest_path),
        "bootstrap_manifest_file_sha256": sha256_file(bootstrap_manifest_path),
        "bootstrap_manifest_payload_sha256": bootstrap_hash,
        "finalizer_root": str(finalizer_root),
        "language_count": len(refined_languages),
        "measurement_contract": (
            "exact_sentencepiece_tokens_on_production_sample_with_preserved_"
            "fineweb2_train_rehydration_multipliers"
        ),
    }
    artifact["artifact_sha256"] = canonical_json_sha256(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--finalizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = refine_calibration(
        calibration_path=args.calibration,
        bootstrap_manifest_path=args.bootstrap_manifest,
        finalizer_root=args.finalizer_root,
    )
    _write_frozen(args.output.resolve(), artifact)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": artifact["artifact_sha256"],
                "language_count": len(artifact["languages"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
