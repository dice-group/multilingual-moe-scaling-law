#!/usr/bin/env python3
"""Build the frozen nested-K A/B-v2 token and document sampling plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    LanguageCalibration,
    build_nested_sampling_plan,
    canonical_json_sha256,
    sha256_file,
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_language_map(path: Path) -> list[str]:
    payload = _load_object(path)
    group = payload.get("0")
    if not isinstance(group, dict) or not isinstance(group.get("languages"), list):
        raise ValueError(f"Language map must contain 0.languages: {path}")
    languages = [str(value) for value in group["languages"]]
    if not languages or len(languages) != len(set(languages)):
        raise ValueError(f"Language map is empty or contains duplicates: {path}")
    return languages


def _parse_language_maps(values: list[str]) -> tuple[dict[int, list[str]], dict[str, str]]:
    language_sets: dict[int, list[str]] = {}
    hashes: dict[str, str] = {}
    for raw in values:
        key, separator, path_text = raw.partition("=")
        if not separator:
            raise ValueError("--language-map values must have the form K=path")
        k = int(key.removeprefix("K").removeprefix("k"))
        path = Path(path_text).resolve()
        languages = _load_language_map(path)
        if len(languages) != k:
            raise ValueError(f"K{k} language map contains {len(languages)} languages")
        if k in language_sets:
            raise ValueError(f"Duplicate language map for K{k}")
        language_sets[k] = languages
        hashes[f"K{k}"] = sha256_file(path)
    if not language_sets:
        raise ValueError("At least one --language-map is required")
    return language_sets, hashes


def build_plan_artifact(
    *,
    calibration_path: Path,
    language_map_args: list[str],
    source_manifest_path: Path,
    tokenizer_model_path: Path,
    target_train_tokens: int,
    alpha: float,
    validation_cap_tokens: int,
    sampling_seed: int = 42,
) -> dict[str, Any]:
    calibration_payload = _load_object(calibration_path)
    calibration_hash = calibration_payload.get("artifact_sha256")
    calibration_without_hash = dict(calibration_payload)
    calibration_without_hash.pop("artifact_sha256", None)
    if (
        calibration_payload.get("schema_version") != 1
        or calibration_payload.get("tool_id") != "fineweb_ab_v2_token_calibration_v1"
        or calibration_payload.get("status") != "complete"
        or calibration_hash != canonical_json_sha256(calibration_without_hash)
    ):
        raise ValueError("Calibration artifact hash/schema/status is invalid")
    raw_languages = calibration_payload.get("languages")
    if not isinstance(raw_languages, dict):
        raise ValueError("Calibration artifact must contain a languages object")
    calibrations: dict[str, LanguageCalibration] = {}
    for language, raw in raw_languages.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid calibration for {language}")
        calibration = LanguageCalibration(language=str(language), **raw)
        calibration.validate()
        calibrations[str(language)] = calibration

    language_sets, language_map_hashes = _parse_language_maps(language_map_args)
    maximum_languages = set(language_sets[max(language_sets)])
    if set(calibrations) != maximum_languages:
        raise ValueError(
            "Calibration language coverage must exactly equal the maximum-K language map"
        )
    source_manifest = _load_object(source_manifest_path)
    expected_source_manifest_hash = source_manifest.get("manifest_sha256")
    source_manifest_without_hash = dict(source_manifest)
    source_manifest_without_hash.pop("manifest_sha256", None)
    if expected_source_manifest_hash != canonical_json_sha256(source_manifest_without_hash):
        raise ValueError("Source materialization manifest hash mismatch")
    if not tokenizer_model_path.is_file():
        raise ValueError(f"Tokenizer model does not exist: {tokenizer_model_path}")
    tokenizer_hash = sha256_file(tokenizer_model_path)
    calibration_provenance = calibration_payload.get("provenance")
    if not isinstance(calibration_provenance, dict):
        raise ValueError("Calibration artifact lacks provenance")
    if calibration_provenance.get("tokenizer_model_sha256") != tokenizer_hash:
        raise ValueError("Calibration artifact was measured with a different tokenizer")
    if (
        source_manifest.get("source_lock_sha256")
        != calibration_provenance.get("source_lock_sha256")
    ):
        raise ValueError("Calibration and source materialization use different source locks")
    source_artifacts = source_manifest.get("artifacts")
    inventory_artifact = (
        source_artifacts.get("inventory") if isinstance(source_artifacts, dict) else None
    )
    if (
        not isinstance(inventory_artifact, dict)
        or inventory_artifact.get("sha256")
        != calibration_provenance.get("fineweb2_inventory_sha256")
    ):
        raise ValueError("Calibration and source materialization use different inventories")
    for name, raw_artifact in source_artifacts.items():
        if not isinstance(raw_artifact, dict):
            raise ValueError(f"Source materialization artifact is invalid: {name}")
        artifact_path = Path(str(raw_artifact.get("path", ""))).resolve()
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != raw_artifact.get("sha256")
        ):
            raise ValueError(f"Source materialization artifact changed: {name}")

    plan = build_nested_sampling_plan(
        calibrations=calibrations,
        language_sets=language_sets,
        train_tokens_per_set=int(target_train_tokens),
        alpha=float(alpha),
        validation_cap_tokens=int(validation_cap_tokens),
        sampling_seed=int(sampling_seed),
    )
    provenance = {
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_payload_sha256": calibration_hash,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_manifest_payload_sha256": expected_source_manifest_hash,
        "tokenizer_model_path": str(tokenizer_model_path.resolve()),
        "tokenizer_model_sha256": tokenizer_hash,
        "language_map_file_sha256": language_map_hashes,
    }
    artifact = {**plan, "provenance": provenance}
    artifact.pop("plan_sha256", None)
    artifact["plan_sha256"] = canonical_json_sha256(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--language-map", action="append", default=[], help="K=path; repeat for each tier")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--target-train-tokens", type=int, default=32_000_000_000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--validation-cap-tokens", type=int, default=20_000_000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact = build_plan_artifact(
        calibration_path=Path(args.calibration).resolve(),
        language_map_args=list(args.language_map),
        source_manifest_path=Path(args.source_manifest).resolve(),
        tokenizer_model_path=Path(args.tokenizer_model).resolve(),
        target_train_tokens=int(args.target_train_tokens),
        alpha=float(args.alpha),
        validation_cap_tokens=int(args.validation_cap_tokens),
        sampling_seed=int(args.sampling_seed),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"Refusing to replace a different frozen sampling plan: {output}")
    else:
        output.write_text(rendered, encoding="utf-8")
    print(f"[OK] sampling_plan={output} sha256={artifact['plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
