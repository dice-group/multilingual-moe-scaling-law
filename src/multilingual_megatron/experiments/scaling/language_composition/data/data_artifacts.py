"""Derive arm-specific training and K196 evaluation manifests after tokenization."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.k12_migration.common import (
    _write_frozen_json,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
    verify_plan_hash,
)

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    ARM_GROUPINGS,
    EXCLUDED_LANGUAGES,
    K200_GROUPING,
    REPO_ROOT,
    REFERENCE_GROUPING,
    CompositionStudyError,
    grouping_languages,
    language_set_sha256,
    read_object,
    verify_selection_authority,
)


ARM_MANIFEST_RELATIVE = {
    "compact": Path("composition_arms/compact/megatron_data_manifest.json"),
    "diverse": Path("composition_arms/diverse/megatron_data_manifest.json"),
}
K196_VALIDATION_RELATIVE = Path(
    "composition_validation/k196_megatron_data_manifest.json"
)
COHORTS_RELATIVE = Path("composition_validation/composition_cohorts.json")
AUDIT_RELATIVE = Path("composition_validation/composition_manifest_audit.json")


def _filtered_mapping(value: Any, languages: Sequence[str]) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        language: copy.deepcopy(value[language])
        for language in languages
        if language in value
    }


def _prefixes_for_languages(
    prefixes_by_language: Mapping[str, Any], languages: Sequence[str]
) -> list[str]:
    result: list[str] = []
    for language in languages:
        values = prefixes_by_language.get(language)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise CompositionStudyError(f"invalid tokenized prefixes for {language}")
        result.extend(values)
    return result


def _base_filtered_manifest(
    source: Mapping[str, Any], languages: Sequence[str]
) -> dict[str, Any]:
    source_languages = source.get("languages")
    prefixes_by_language = source.get("prefixes_by_language")
    if not isinstance(source_languages, list) or not isinstance(prefixes_by_language, dict):
        raise CompositionStudyError("tokenized source manifest lacks language prefixes")
    if not set(languages).issubset(source_languages):
        raise CompositionStudyError("requested manifest languages escape tokenized coverage")
    result = copy.deepcopy(dict(source))
    result.pop("manifest_payload_sha256", None)
    result["languages"] = list(languages)
    selected_prefixes_by_language = {
        language: copy.deepcopy(prefixes_by_language[language])
        for language in languages
        if language in prefixes_by_language
    }
    result["prefixes_by_language"] = selected_prefixes_by_language
    result["prefixes"] = (
        _prefixes_for_languages(prefixes_by_language, languages)
        if selected_prefixes_by_language
        else []
    )
    source_weights = source.get("weights")
    source_prefixes = source.get("prefixes")
    if isinstance(source_weights, list) and isinstance(source_prefixes, list):
        if len(source_weights) != len(source_prefixes):
            raise CompositionStudyError("source prefix/weight cardinality mismatch")
        weights_by_prefix = dict(zip(source_prefixes, source_weights, strict=True))
        result["weights"] = [weights_by_prefix[prefix] for prefix in result["prefixes"]]
    for field in ("documents_per_language", "tokens_per_language"):
        result[field] = _filtered_mapping(source.get(field), languages)
    # Tokenization uses one partition per represented language in this frozen
    # corpus.  A validation-only manifest legitimately has empty root prefixes;
    # its materialized prefixes live under splits.validation.
    if result["prefixes"] and len(result["prefixes"]) != len(languages):
        raise CompositionStudyError("composition manifests require one prefix per language")
    splits = result.get("splits")
    if isinstance(splits, dict):
        for split_name, raw_split in list(splits.items()):
            if not isinstance(raw_split, dict):
                continue
            split = copy.deepcopy(raw_split)
            by_language = split.get("prefixes_by_language")
            artifacts_by_language = split.get("prefix_artifacts_by_language")
            if isinstance(by_language, dict):
                selected_by_language = {
                    language: copy.deepcopy(by_language[language])
                    for language in languages
                    if language in by_language
                }
                split["prefixes_by_language"] = selected_by_language
                split["prefixes"] = [
                    prefix
                    for language in languages
                    for prefix in selected_by_language.get(language, [])
                ]
                raw_weights = raw_split.get("weights")
                raw_prefixes = raw_split.get("prefixes")
                if not isinstance(raw_weights, list) or not isinstance(raw_prefixes, list):
                    raise CompositionStudyError(
                        f"{split_name} split lacks prefix weights"
                    )
                if len(raw_weights) != len(raw_prefixes):
                    raise CompositionStudyError(
                        f"{split_name} split prefix/weight cardinality mismatch"
                    )
                weights_by_prefix = dict(zip(raw_prefixes, raw_weights, strict=True))
                split["weights"] = [
                    weights_by_prefix[prefix] for prefix in split["prefixes"]
                ]
            if isinstance(artifacts_by_language, dict):
                split["prefix_artifacts_by_language"] = {
                    language: copy.deepcopy(artifacts_by_language[language])
                    for language in languages
                    if language in artifacts_by_language
                }
            for field in ("documents_per_language", "tokens_per_language"):
                split[field] = _filtered_mapping(raw_split.get(field), languages)
            documents = split.get("documents_per_language")
            tokens = split.get("tokens_per_language")
            if isinstance(documents, dict):
                split["total_documents"] = sum(int(value) for value in documents.values())
            if isinstance(tokens, dict):
                split["total_tokens_including_eod"] = sum(
                    int(value) for value in tokens.values()
                )
            splits[split_name] = split
    return result


def _arm_manifest(
    source: Mapping[str, Any], *, arm: str, arm_plan: Mapping[str, Any]
) -> dict[str, Any]:
    languages = tuple(str(value) for value in arm_plan["languages"])
    result = _base_filtered_manifest(source, languages)
    probabilities = {
        language: float(arm_plan["probabilities"][language]) for language in languages
    }
    result.update(
        {
            "tier": f"K12_{arm}",
            "purpose": "language_composition_K12_training_with_global_K196_validation",
            "alpha": float(arm_plan["alpha"]),
            "language_probabilities": probabilities,
            "target_train_tokens": int(arm_plan["target_train_tokens"]),
            "target_exposure_tokens": {
                language: int(arm_plan["target_exposure_tokens"][language])
                for language in languages
            },
            "weights": [probabilities[language] for language in languages],
        }
    )
    splits = result.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    train_by_language = (
        train.get("prefixes_by_language") if isinstance(train, dict) else None
    )
    if (
        not isinstance(train_by_language, dict)
        or set(train_by_language) != set(languages)
        or any(
            not isinstance(train_by_language[language], list)
            or len(train_by_language[language]) != 1
            for language in languages
        )
    ):
        raise CompositionStudyError(
            "composition training requires one train prefix per selected language"
        )
    expected_train_prefixes = [
        train_by_language[language][0] for language in languages
    ]
    if train.get("prefixes") != expected_train_prefixes:
        raise CompositionStudyError("composition train-prefix order drifted")
    # Megatron consumes splits.train before the root fallback.  Freeze the
    # alpha-sampling probabilities on the actually consumed split as well as
    # on the manifest root.
    train["weights"] = [probabilities[language] for language in languages]
    train["language_probabilities"] = probabilities
    result["language_probabilities"] = probabilities
    result.setdefault("lineage", {})["language_composition_arm"] = {
        "arm": arm,
        "language_set_sha256": arm_plan["language_set_sha256"],
        "sampling_plan_payload_sha256": source["lineage"][
            "sampling_plan_payload_sha256"
        ],
    }
    result["manifest_payload_sha256"] = canonical_json_sha256(result)
    return result


def _validation_manifest(
    source: Mapping[str, Any], languages: Sequence[str]
) -> dict[str, Any]:
    result = _base_filtered_manifest(source, languages)
    result["purpose"] = "language_composition_primary_K196_validation"
    result["lineage"]["language_composition_validation"] = {
        "frame": "frozen K196 eligibility frame",
        "language_set_sha256": language_set_sha256(languages),
        "excluded_languages": list(EXCLUDED_LANGUAGES),
    }
    result["manifest_payload_sha256"] = canonical_json_sha256(result)
    return result


def build_cohorts() -> dict[str, Any]:
    k200 = grouping_languages((REPO_ROOT / K200_GROUPING).resolve(), expected_count=200)
    k196 = tuple(language for language in k200 if language not in EXCLUDED_LANGUAGES)
    reference = set(
        grouping_languages((REPO_ROOT / REFERENCE_GROUPING).resolve(), expected_count=12)
    )
    compact = set(
        grouping_languages(
            (REPO_ROOT / ARM_GROUPINGS["compact"]).resolve(), expected_count=12
        )
    )
    diverse = set(
        grouping_languages(
            (REPO_ROOT / ARM_GROUPINGS["diverse"]).resolve(), expected_count=12
        )
    )
    arm_sets = {"reference": reference, "compact": compact, "diverse": diverse}
    cohorts: dict[str, list[str]] = {
        "k196": list(k196),
        "unseen_177": [
            language for language in k196 if not any(language in values for values in arm_sets.values())
        ],
    }
    patterns = {
        "reference_only": frozenset({"reference"}),
        "compact_only": frozenset({"compact"}),
        "diverse_only": frozenset({"diverse"}),
        "reference_compact": frozenset({"reference", "compact"}),
        "reference_diverse": frozenset({"reference", "diverse"}),
        "compact_diverse": frozenset({"compact", "diverse"}),
        "all_three": frozenset(arm_sets),
    }
    for name, expected_membership in patterns.items():
        cohorts[name] = [
            language
            for language in k196
            if frozenset(
                arm for arm, values in arm_sets.items() if language in values
            )
            == expected_membership
        ]
    if len(k196) != 196 or len(cohorts["unseen_177"]) != 177:
        raise CompositionStudyError("K196/common-unseen cohort cardinality drifted")
    expected_counts = {
        "reference_only": 2,
        "compact_only": 4,
        "diverse_only": 3,
        "reference_compact": 1,
        "reference_diverse": 2,
        "compact_diverse": 0,
        "all_three": 7,
    }
    if {name: len(cohorts[name]) for name in expected_counts} != expected_counts:
        raise CompositionStudyError("three-arm membership cohorts drifted")
    payload: dict[str, Any] = {
        "schema": "stage1_language_composition_cohorts/v1",
        "status": "frozen",
        "primary": "unseen_177",
        "selection": verify_selection_authority(),
        "cohorts": {
            name: {
                "languages": values,
                "language_count": len(values),
                "language_set_sha256": language_set_sha256(values),
            }
            for name, values in cohorts.items()
        },
        "claim_boundary": (
            "unseen_177 is the common held-out language frame; all other cohorts are "
            "predeclared descriptive secondary readouts"
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def build_composition_manifests(
    *, tokenized_root: Path, sampling_plan_path: Path
) -> dict[str, Any]:
    tokenized_root = tokenized_root.resolve()
    plan = read_object(sampling_plan_path.resolve(), label="composition sampling plan")
    if not verify_plan_hash(plan) or set(plan.get("composition_arms", {})) != {
        "compact",
        "diverse",
    }:
        raise CompositionStudyError("sampling plan lacks two hash-valid composition arms")
    source_train = read_object(
        tokenized_root / "megatron_data_manifest.json",
        label="full training manifest",
    )
    source_validation = read_object(
        tokenized_root / "validation_megatron_data_manifest.json",
        label="full validation manifest",
    )
    if source_train.get("lineage", {}).get("sampling_plan_payload_sha256") != plan["plan_sha256"]:
        raise CompositionStudyError("tokenized training manifest belongs to another plan")
    k200 = grouping_languages((REPO_ROOT / K200_GROUPING).resolve(), expected_count=200)
    k196 = tuple(language for language in k200 if language not in EXCLUDED_LANGUAGES)
    artifacts: dict[Path, dict[str, Any]] = {}
    for arm, relative in ARM_MANIFEST_RELATIVE.items():
        artifacts[tokenized_root / relative] = _arm_manifest(
            source_train, arm=arm, arm_plan=plan["composition_arms"][arm]
        )
    artifacts[tokenized_root / K196_VALIDATION_RELATIVE] = _validation_manifest(
        source_validation, k196
    )
    cohorts = build_cohorts()
    artifacts[tokenized_root / COHORTS_RELATIVE] = cohorts
    for path, payload in artifacts.items():
        _write_frozen_json(path, payload)
    audit: dict[str, Any] = {
        "schema": "stage1_language_composition_manifest_audit/v1",
        "status": "passed",
        "sampling_plan": {
            "path": str(sampling_plan_path.resolve()),
            "file_sha256": sha256_file(sampling_plan_path.resolve()),
            "payload_sha256": plan["plan_sha256"],
        },
        "artifacts": {
            str(path.relative_to(tokenized_root)): {
                "file_sha256": sha256_file(path),
                "payload_sha256": payload.get(
                    "manifest_payload_sha256", payload.get("payload_sha256")
                ),
            }
            for path, payload in artifacts.items()
        },
        "counts": {
            "training_arms": 2,
            "training_languages_per_arm": 12,
            "validation_languages": 196,
            "primary_unseen_languages": 177,
        },
    }
    audit["payload_sha256"] = canonical_json_sha256(audit)
    audit_path = tokenized_root / AUDIT_RELATIVE
    _write_frozen_json(audit_path, audit)
    return audit
