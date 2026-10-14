"""Build the common physical corpus capacity required by both K12 arms."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.k12_migration.adoption import (
    _verify_completed_adoption,
)
from multilingual_megatron.experiments.scaling.k12_migration.common import (
    _receipt,
    _reject_output_inside_inputs,
    _verify_bootstrap,
    _verify_plan,
    _verify_release,
    _write_frozen_json,
)
from multilingual_megatron.experiments.scaling.k12_migration.constants import (
    LANGUAGE_MAP_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TOOL_ID,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    LanguageCalibration,
    alpha_probabilities,
    canonical_json_sha256,
    largest_remainder_quotas,
    sha256_file,
)

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    ARM_GROUPINGS,
    REPO_ROOT,
    CompositionStudyError,
    grouping_languages,
    verify_selection_authority,
)


EXPECTED_INCREASED = frozenset(
    {
        "arb_Arab",
        "cmn_Hani",
        "dan_Latn",
        "deu_Latn",
        "fra_Latn",
        "hin_Deva",
        "ita_Latn",
        "jpn_Jpan",
        "kor_Hang",
        "nld_Latn",
        "por_Latn",
        "ron_Latn",
        "rus_Cyrl",
        "spa_Latn",
        "swe_Latn",
        "tha_Thai",
    }
)
EXPECTED_ACQUISITION = frozenset(
    {
        "arb_Arab",
        "dan_Latn",
        "hin_Deva",
        "kor_Hang",
        "nld_Latn",
        "ron_Latn",
        "swe_Latn",
        "tha_Thai",
    }
)
ACQUISITION_SOURCE_DOCUMENT_MARGIN = 1.10


def _arm_tier(
    predecessor: Mapping[str, Any], languages: Sequence[str]
) -> dict[str, Any]:
    availability = {
        language: int(
            predecessor["physical_pool"][language][
                "estimated_train_rehydrated_tokens"
            ]
        )
        for language in languages
    }
    probabilities = alpha_probabilities(
        availability, alpha=float(predecessor["alpha"])
    )
    quotas = largest_remainder_quotas(
        int(predecessor["train_tokens_per_set"]), probabilities
    )
    from multilingual_megatron.experiments.scaling.language_composition.study_contracts import language_set_sha256

    return {
        "languages": list(languages),
        "language_set_sha256": language_set_sha256(languages),
        "alpha": float(predecessor["alpha"]),
        "availability_basis": "estimated_train_tokens_after_fineweb2_rehydration",
        "availability_tokens": availability,
        "target_train_tokens": int(predecessor["train_tokens_per_set"]),
        "probabilities": probabilities,
        "target_exposure_tokens": quotas,
    }


def _physical_with_requested(
    row: Mapping[str, Any],
    requested_tokens: int,
    *,
    source_document_margin: float = 1.0,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    calibration_raw = result.get("calibration")
    if not isinstance(calibration_raw, dict):
        raise CompositionStudyError("physical-pool row lacks tokenizer calibration")
    calibration = LanguageCalibration(**calibration_raw)
    calibration.validate()
    if not math.isfinite(source_document_margin) or source_document_margin < 1.0:
        raise CompositionStudyError(
            "source-document planning margin must be finite and at least one"
        )
    available = int(result["estimated_train_rehydrated_tokens"])
    physical_target = min(int(requested_tokens), available)
    planned_source_documents = min(
        int(result["estimated_train_documents"]),
        max(
            1,
            math.ceil(
                physical_target
                / calibration.rehydrated_tokens_per_source_document
                * source_document_margin
            ),
        ),
    )
    result.update(
        {
            "requested_max_tier_tokens": int(requested_tokens),
            "physical_target_tokens": physical_target,
            "planned_source_documents": planned_source_documents,
            "training_repeat_required": int(requested_tokens) > physical_target,
            "estimated_passes_at_max_quota": int(requested_tokens)
            / physical_target,
        }
    )
    if source_document_margin > 1.0:
        result["acquisition_source_document_margin"] = source_document_margin
    return result


def _rebase_adopted_provenance_paths(
    plan: dict[str, Any],
    *,
    calibration_path: Path,
    source_manifest_path: Path,
    tokenizer_model_path: Path,
) -> None:
    """Pin adopted inputs to their canonical paths in the successor root."""

    provenance = plan.get("provenance")
    if not isinstance(provenance, dict):
        raise CompositionStudyError("successor plan lacks provenance")
    provenance.update(
        {
            "calibration_path": str(calibration_path.resolve()),
            "source_manifest_path": str(source_manifest_path.resolve()),
            "tokenizer_model_path": str(tokenizer_model_path.resolve()),
        }
    )


def build_successor_plan(
    predecessor: Mapping[str, Any],
    *,
    calibration_path: Path,
    source_manifest_path: Path,
    tokenizer_model_path: Path,
) -> dict[str, Any]:
    """Keep canonical nested tiers fixed and add two capacity-only K12 arms."""

    selection = verify_selection_authority()
    arms = {
        arm: _arm_tier(
            predecessor,
            grouping_languages((REPO_ROOT / path).resolve(), expected_count=12),
        )
        for arm, path in ARM_GROUPINGS.items()
    }
    successor = copy.deepcopy(dict(predecessor))
    successor.pop("plan_sha256", None)
    _rebase_adopted_provenance_paths(
        successor,
        calibration_path=calibration_path,
        source_manifest_path=source_manifest_path,
        tokenizer_model_path=tokenizer_model_path,
    )
    successor["composition_arms"] = arms
    for language, old_row in predecessor["physical_pool"].items():
        requested = int(old_row["requested_max_tier_tokens"])
        for arm in arms.values():
            requested = max(
                requested, int(arm["target_exposure_tokens"].get(language, 0))
            )
        successor["physical_pool"][language] = _physical_with_requested(
            old_row,
            requested,
            source_document_margin=(
                ACQUISITION_SOURCE_DOCUMENT_MARGIN
                if language in EXPECTED_ACQUISITION
                else 1.0
            ),
        )
    successor.setdefault("provenance", {})["language_composition"] = {
        **selection,
        "grouping_file_sha256": {
            arm: sha256_file((REPO_ROOT / path).resolve())
            for arm, path in ARM_GROUPINGS.items()
        },
        "capacity_rule": (
            "elementwise maximum physical target across the unchanged nested tiers "
            "and the compact/diverse K12 arm exposures"
        ),
        "acquisition_source_document_margin": (
            ACQUISITION_SOURCE_DOCUMENT_MARGIN
        ),
    }
    successor["plan_sha256"] = canonical_json_sha256(successor)
    return successor


def build_transition(
    *,
    predecessor: Mapping[str, Any],
    successor: Mapping[str, Any],
    predecessor_release: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("tiers") != successor.get("tiers"):
        raise CompositionStudyError("canonical nested tier payloads changed")
    old_pool = predecessor.get("physical_pool")
    new_pool = successor.get("physical_pool")
    release_languages = predecessor_release.get("languages")
    if (
        not isinstance(old_pool, dict)
        or not isinstance(new_pool, dict)
        or not isinstance(release_languages, dict)
        or set(old_pool) != set(new_pool)
        or set(new_pool) != set(release_languages)
        or len(new_pool) != 200
    ):
        raise CompositionStudyError("predecessor/successor/release coverage differs")
    changed = sorted(language for language in new_pool if old_pool[language] != new_pool[language])
    increased = sorted(
        language
        for language in new_pool
        if int(new_pool[language]["physical_target_tokens"])
        > int(old_pool[language]["physical_target_tokens"])
    )
    decreased = sorted(
        language
        for language in new_pool
        if int(new_pool[language]["physical_target_tokens"])
        < int(old_pool[language]["physical_target_tokens"])
    )
    acquisition = sorted(
        language
        for language in new_pool
        if int(new_pool[language]["physical_target_tokens"])
        > int(release_languages[language]["train"].get("available_tokens", 0))
    )
    arm_union = sorted(
        set().union(
            *(
                set(payload["languages"])
                for payload in successor["composition_arms"].values()
            )
        )
    )
    if set(changed) != set(increased) or set(increased) != EXPECTED_INCREASED:
        raise CompositionStudyError("capacity expansion differs from the frozen 16-language delta")
    if decreased:
        raise CompositionStudyError("composition capacity migration may not shrink a pool")
    if set(acquisition) != EXPECTED_ACQUISITION:
        raise CompositionStudyError("incremental acquisition differs from the frozen eight languages")
    reused = sorted(set(new_pool) - set(changed))
    metadata_only = sorted(set(arm_union) - set(changed))
    counts = {
        "changed_physical": 16,
        "increased_physical": 16,
        "decreased_physical": 0,
        "acquisition": 8,
        "delta_finalize": 16,
        "reused_physical": 184,
        # The compact/diverse union has 17 languages.  The full three-arm
        # study union is 19 only after adding the unchanged reference arm.
        "exposure_changed": 17,
    }
    return {
        "changed_physical_languages": changed,
        "increased_physical_languages": increased,
        "decreased_physical_languages": decreased,
        "acquisition_languages": acquisition,
        "delta_finalize_languages": changed,
        "reused_physical_languages": reused,
        "exposure_changed_languages": arm_union,
        "metadata_only_refresh_languages": metadata_only,
        "counts": counts,
    }


def _subset_plan(plan: Mapping[str, Any], languages: Sequence[str]) -> dict[str, Any]:
    selected = set(languages)
    result = copy.deepcopy(dict(plan))
    result.pop("plan_sha256", None)
    result["physical_pool"] = {
        language: copy.deepcopy(plan["physical_pool"][language])
        for language in sorted(selected)
    }
    for collection_name in ("tiers", "composition_arms"):
        for payload in result.get(collection_name, {}).values():
            payload["languages"] = [
                language for language in payload["languages"] if language in selected
            ]
            for field in (
                "availability_tokens",
                "probabilities",
                "target_exposure_tokens",
            ):
                values = payload.get(field)
                if isinstance(values, dict):
                    payload[field] = {
                        language: values[language]
                        for language in sorted(selected)
                        if language in values
                    }
    result.setdefault("provenance", {})["migration_delta"] = {
        "tool_id": TOOL_ID,
        "full_successor_plan_sha256": plan["plan_sha256"],
        "languages": sorted(selected),
    }
    result["plan_sha256"] = canonical_json_sha256(result)
    return result


def prepare_capacity_migration(
    *,
    predecessor_plan_path: Path,
    predecessor_release_dir: Path,
    adoption_manifest_path: Path,
    successor_plan_path: Path,
    delta_plan_path: Path,
    acquisition_language_map_path: Path,
    migration_plan_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Freeze artifacts consumed by the proven incremental migration engine."""

    predecessor_plan_path = predecessor_plan_path.resolve()
    predecessor_release_dir = predecessor_release_dir.resolve()
    adoption_manifest_path = adoption_manifest_path.resolve()
    adoption = _verify_completed_adoption(adoption_manifest_path)
    predecessor_root = Path(str(adoption["predecessor_root"])).resolve()
    successor_root = Path(str(adoption["successor_root"])).resolve()
    canonical_adoption = successor_root / "corpus/k12_migration/adoption_manifest.json"
    if adoption_manifest_path != canonical_adoption.resolve():
        raise CompositionStudyError("adoption manifest is outside the migration engine path")
    for source in (predecessor_plan_path, predecessor_release_dir):
        try:
            source.relative_to(predecessor_root)
        except ValueError as exc:
            raise CompositionStudyError("predecessor artifact escapes the sealed root") from exc
    predecessor = _verify_plan(predecessor_plan_path, label="predecessor sampling")
    predecessor_release, _ = _verify_release(
        predecessor_release_dir, label="predecessor release"
    )
    bootstrap_meta = predecessor_release.get("inputs", {}).get("bootstrap_manifest")
    if not isinstance(bootstrap_meta, dict):
        raise CompositionStudyError("predecessor release lacks bootstrap lineage")
    predecessor_bootstrap_manifest = Path(str(bootstrap_meta.get("path", ""))).resolve()
    predecessor_bootstrap = _verify_bootstrap(
        predecessor_bootstrap_manifest.parent, label="predecessor bootstrap"
    )
    if (
        bootstrap_meta.get("sha256") != sha256_file(predecessor_bootstrap_manifest)
        or predecessor_bootstrap.get("manifest_sha256")
        != bootstrap_meta.get("payload_sha256", predecessor_bootstrap.get("manifest_sha256"))
    ):
        raise CompositionStudyError("predecessor release/bootstrap lineage mismatch")
    provenance = predecessor.get("provenance")
    if not isinstance(provenance, dict):
        raise CompositionStudyError("predecessor plan lacks provenance")
    pinned = {
        "calibration": successor_root / "corpus/token_calibration.json",
        "source_manifest": successor_root
        / "corpus/source_lock/source_materialization_manifest.json",
        "tokenizer": successor_root
        / "corpus/tokenizer/sp_unigram_64k/tokenizer.model",
    }
    expected_hashes = {
        "calibration": provenance["calibration_sha256"],
        "source_manifest": provenance["source_manifest_file_sha256"],
        "tokenizer": provenance["tokenizer_model_sha256"],
    }
    for label, path in pinned.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hashes[label]:
            raise CompositionStudyError(f"adopted predecessor-pinned {label} changed")

    successor = build_successor_plan(
        predecessor,
        calibration_path=pinned["calibration"],
        source_manifest_path=pinned["source_manifest"],
        tokenizer_model_path=pinned["tokenizer"],
    )
    transition = build_transition(
        predecessor=predecessor,
        successor=successor,
        predecessor_release=predecessor_release,
    )
    delta = _subset_plan(successor, transition["delta_finalize_languages"])
    outputs = (
        successor_plan_path.resolve(),
        delta_plan_path.resolve(),
        acquisition_language_map_path.resolve(),
        migration_plan_path.resolve(),
        receipt_path.resolve(),
    )
    for output in outputs:
        try:
            output.relative_to(successor_root)
        except ValueError as exc:
            raise CompositionStudyError(f"successor artifact escapes {successor_root}") from exc
    _reject_output_inside_inputs(migration_plan_path.parent, [predecessor_release_dir])
    acquisition_map = {
        "schema_version": LANGUAGE_MAP_SCHEMA_VERSION,
        "language_set_id": "stage1_language_composition_incremental_acquisition",
        "0": {"languages": transition["acquisition_languages"]},
        "provenance": {
            "tool_id": TOOL_ID,
            "successor_sampling_plan_sha256": successor["plan_sha256"],
            "selection_rule": "new physical target exceeds predecessor acquired train pool",
        },
    }
    migration: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        # The modular bootstrap/release engine intentionally retains its stable
        # tool ID; the composition-specific contract is separately explicit.
        "tool_id": TOOL_ID,
        "status": "prepared",
        "predecessor": {
            "sampling_plan": {
                "path": str(predecessor_plan_path),
                "file_sha256": sha256_file(predecessor_plan_path),
                "payload_sha256": predecessor["plan_sha256"],
            },
            "release": {
                "path": str(predecessor_release_dir),
                "corpus_manifest_sha256": sha256_file(
                    predecessor_release_dir / "corpus_manifest.json"
                ),
                "corpus_manifest_payload_sha256": predecessor_release["manifest_sha256"],
            },
            "production_bootstrap": {
                "path": str(predecessor_bootstrap_manifest.parent),
                "manifest_file_sha256": sha256_file(predecessor_bootstrap_manifest),
                "manifest_payload_sha256": predecessor_bootstrap["manifest_sha256"],
            },
        },
        "successor": {
            "sampling_plan": {
                "path": str(successor_plan_path.resolve()),
                "payload_sha256": successor["plan_sha256"],
            },
            "delta_sampling_plan": {
                "path": str(delta_plan_path.resolve()),
                "payload_sha256": delta["plan_sha256"],
            },
            "acquisition_language_map": {
                "path": str(acquisition_language_map_path.resolve())
            },
        },
        "pinned_inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in pinned.items()
        },
        "upstream_adoption": {
            "path": str(adoption_manifest_path),
            "file_sha256": sha256_file(adoption_manifest_path),
            "payload_sha256": adoption["adoption_manifest_sha256"],
            "predecessor_root": str(predecessor_root),
            "successor_root": str(successor_root),
        },
        "transition": transition,
        "composition_contract": successor["provenance"]["language_composition"],
        "canonical_audit_path_policy": (
            "all adopted absolute paths remain under the final pipeline release root"
        ),
        "execution_contract": {
            "acquisition_worker_count": 8,
            "acquisition_array": "0-7",
            "acquisition_language_order": transition["acquisition_languages"],
            "worker_interface": "stable K12 migration bootstrap worker interface",
            "merge_interface": "stable K12 migration bootstrap merge interface",
            "required_result": "exactly-eight-language hash-valid bootstrap manifest",
        },
    }
    _write_frozen_json(successor_plan_path.resolve(), successor)
    _write_frozen_json(delta_plan_path.resolve(), delta)
    _write_frozen_json(acquisition_language_map_path.resolve(), acquisition_map)
    migration["successor"]["sampling_plan"]["file_sha256"] = sha256_file(
        successor_plan_path.resolve()
    )
    migration["successor"]["delta_sampling_plan"]["file_sha256"] = sha256_file(
        delta_plan_path.resolve()
    )
    migration["successor"]["acquisition_language_map"]["file_sha256"] = sha256_file(
        acquisition_language_map_path.resolve()
    )
    migration["migration_plan_sha256"] = canonical_json_sha256(migration)
    _write_frozen_json(migration_plan_path.resolve(), migration)
    receipt = _receipt(
        "prepare",
        {
            "migration_plan": {
                "path": str(migration_plan_path.resolve()),
                "sha256": sha256_file(migration_plan_path.resolve()),
                "payload_sha256": migration["migration_plan_sha256"],
            },
            "successor_sampling_plan": {
                "path": str(successor_plan_path.resolve()),
                "sha256": sha256_file(successor_plan_path.resolve()),
                "payload_sha256": successor["plan_sha256"],
            },
            "delta_sampling_plan": {
                "path": str(delta_plan_path.resolve()),
                "sha256": sha256_file(delta_plan_path.resolve()),
                "payload_sha256": delta["plan_sha256"],
            },
            "acquisition_language_map": {
                "path": str(acquisition_language_map_path.resolve()),
                "sha256": sha256_file(acquisition_language_map_path.resolve()),
                "languages": transition["acquisition_languages"],
            },
            "transition": transition,
        },
    )
    _write_frozen_json(receipt_path.resolve(), receipt)
    return receipt
