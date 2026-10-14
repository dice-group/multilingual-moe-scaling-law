"""Contract construction, preparation, and loading for K12 migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.k12_migration.common import (
    K12MigrationError,
    _load_object,
    _physical_entry,
    _receipt,
    _reject_output_inside_inputs,
    _subset_plan,
    _tier_exposure,
    _validate_language_against_plan,
    _validate_transition_shape,
    _verify_bootstrap,
    _verify_payload,
    _verify_plan,
    _verify_release,
    _write_frozen_json,
)
from multilingual_megatron.experiments.scaling.k12_migration.constants import (
    EXPECTED_ACQUISITION,
    EXPECTED_INCREASED,
    LANGUAGE_MAP_SCHEMA_VERSION,
    NEW_TIERS,
    OLD_TIERS,
    REMOVED_FROM_K16,
    SCHEMA_VERSION,
    TOOL_ID,
)
from multilingual_megatron.experiments.scaling.k12_migration.adoption import (
    _verify_completed_adoption,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.pipeline.stages.sampling.tools.build_fineweb_v2_sampling_plan import (
    build_plan_artifact,
)


def build_migration_contract(
    *,
    predecessor_plan: Mapping[str, Any],
    successor_plan: Mapping[str, Any],
    predecessor_release: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare plans and freeze physical, metadata, and acquisition deltas."""

    _validate_transition_shape(predecessor_plan, successor_plan)
    release_languages = predecessor_release.get("languages")
    if not isinstance(release_languages, dict) or set(release_languages) != set(
        successor_plan["physical_pool"]
    ):
        raise K12MigrationError("predecessor release coverage differs from the plans")
    changed_physical: list[str] = []
    increased: list[str] = []
    decreased: list[str] = []
    acquisition: list[str] = []
    exposure_changed: list[str] = []
    for language in sorted(successor_plan["physical_pool"]):
        old_physical = _physical_entry(predecessor_plan, language)
        new_physical = _physical_entry(successor_plan, language)
        _validate_language_against_plan(
            language,
            release_languages[language],
            predecessor_plan,
            require_exposure=True,
        )
        if old_physical != new_physical:
            changed_physical.append(language)
        old_target = int(old_physical["physical_target_tokens"])
        new_target = int(new_physical["physical_target_tokens"])
        if new_target > old_target:
            increased.append(language)
        elif new_target < old_target:
            decreased.append(language)
        available = int(release_languages[language]["train"].get("available_tokens", 0))
        if new_target > available:
            acquisition.append(language)
        old_exposure = {
            tier: _tier_exposure(predecessor_plan, tier, language)
            for tier in OLD_TIERS
            if _tier_exposure(predecessor_plan, tier, language) is not None
        }
        new_exposure = {
            tier: _tier_exposure(successor_plan, tier, language)
            for tier in NEW_TIERS
            if _tier_exposure(successor_plan, tier, language) is not None
        }
        if old_exposure != new_exposure:
            exposure_changed.append(language)
    if not set(acquisition).issubset(increased):
        raise K12MigrationError("acquisition languages must be a subset of increased pools")
    reused = sorted(set(successor_plan["physical_pool"]) - set(changed_physical))
    expected_changed = EXPECTED_INCREASED | REMOVED_FROM_K16
    expected_exposure_changed = expected_changed | {"eng_Latn"}
    exact_drift = {
        "changed_physical_languages": (set(changed_physical), expected_changed),
        "increased_physical_languages": (set(increased), EXPECTED_INCREASED),
        "decreased_physical_languages": (set(decreased), REMOVED_FROM_K16),
        "acquisition_languages": (set(acquisition), EXPECTED_ACQUISITION),
        "exposure_changed_languages": (
            set(exposure_changed),
            expected_exposure_changed,
        ),
    }
    drift = [name for name, (observed, expected) in exact_drift.items() if observed != expected]
    if drift:
        raise K12MigrationError(
            "K16-to-K12 migration differs from the frozen predecessor delta: "
            + ", ".join(drift)
        )
    return {
        "changed_physical_languages": changed_physical,
        "increased_physical_languages": increased,
        "decreased_physical_languages": decreased,
        "acquisition_languages": acquisition,
        "delta_finalize_languages": changed_physical,
        "reused_physical_languages": reused,
        "exposure_changed_languages": exposure_changed,
        "metadata_only_refresh_languages": sorted(set(exposure_changed) - set(changed_physical)),
        "counts": {
            "changed_physical": len(changed_physical),
            "increased_physical": len(increased),
            "decreased_physical": len(decreased),
            "acquisition": len(acquisition),
            "delta_finalize": len(changed_physical),
            "reused_physical": len(reused),
            "exposure_changed": len(exposure_changed),
        },
    }


def prepare_migration(
    *,
    predecessor_plan_path: Path,
    predecessor_release_dir: Path,
    adoption_manifest_path: Path,
    language_map_args: Sequence[str],
    successor_plan_path: Path | None = None,
    delta_plan_path: Path | None = None,
    acquisition_language_map_path: Path | None = None,
    migration_plan_path: Path | None = None,
    receipt_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the K12 plan from predecessor-pinned inputs and freeze its delta."""

    predecessor_plan_path = predecessor_plan_path.resolve()
    predecessor_release_dir = predecessor_release_dir.resolve()
    adoption_manifest_path = adoption_manifest_path.resolve()
    adoption = _verify_completed_adoption(adoption_manifest_path)
    predecessor_root = Path(str(adoption["predecessor_root"])).resolve()
    successor_root = Path(str(adoption["successor_root"])).resolve()
    expected_adoption_path = (
        successor_root / "corpus/k12_migration/adoption_manifest.json"
    ).resolve()
    if adoption_manifest_path != expected_adoption_path:
        raise K12MigrationError(
            f"adoption manifest must use the canonical successor path: {expected_adoption_path}"
        )
    try:
        predecessor_plan_path.relative_to(predecessor_root)
        predecessor_release_dir.relative_to(predecessor_root)
    except ValueError as error:
        raise K12MigrationError(
            "predecessor plan/release are not under the adoption predecessor root"
        ) from error
    predecessor = _verify_plan(predecessor_plan_path, label="predecessor sampling")
    predecessor_release, _ = _verify_release(
        predecessor_release_dir, label="predecessor release"
    )
    predecessor_bootstrap_meta = predecessor_release.get("inputs", {}).get(
        "bootstrap_manifest"
    )
    if not isinstance(predecessor_bootstrap_meta, dict):
        raise K12MigrationError("predecessor release lacks bootstrap-manifest lineage")
    predecessor_bootstrap_manifest = Path(
        str(predecessor_bootstrap_meta.get("path", ""))
    ).resolve()
    predecessor_bootstrap = _verify_bootstrap(
        predecessor_bootstrap_manifest.parent, label="predecessor bootstrap"
    )
    if (
        predecessor_bootstrap_meta.get("sha256")
        != sha256_file(predecessor_bootstrap_manifest)
        or predecessor_bootstrap.get("manifest_sha256")
        != predecessor_bootstrap_meta.get(
            "payload_sha256", predecessor_bootstrap["manifest_sha256"]
        )
    ):
        raise K12MigrationError("predecessor release/bootstrap lineage mismatch")
    provenance = predecessor.get("provenance")
    if not isinstance(provenance, dict):
        raise K12MigrationError("predecessor plan lacks provenance")
    required_paths = {
        "calibration": (
            successor_root / "corpus/token_calibration.json",
            "calibration_sha256",
        ),
        "source manifest": (
            successor_root / "corpus/source_lock/source_materialization_manifest.json",
            "source_manifest_file_sha256",
        ),
        "tokenizer": (
            successor_root / "corpus/tokenizer/sp_unigram_64k/tokenizer.model",
            "tokenizer_model_sha256",
        ),
    }
    pinned: dict[str, Path] = {}
    for label, (path, hash_key) in required_paths.items():
        path = path.resolve()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != provenance.get(hash_key):
            raise K12MigrationError(f"adopted predecessor-pinned {label} changed: {path}")
        pinned[label] = path
    successor = build_plan_artifact(
        calibration_path=pinned["calibration"],
        language_map_args=list(language_map_args),
        source_manifest_path=pinned["source manifest"],
        tokenizer_model_path=pinned["tokenizer"],
        target_train_tokens=int(predecessor["train_tokens_per_set"]),
        alpha=float(predecessor["alpha"]),
        validation_cap_tokens=int(predecessor["validation_cap_tokens"]),
        sampling_seed=int(predecessor["sampling_seed"]),
    )
    transition = build_migration_contract(
        predecessor_plan=predecessor,
        successor_plan=successor,
        predecessor_release=predecessor_release,
    )
    delta = _subset_plan(successor, transition["delta_finalize_languages"])
    successor_output = successor_plan_path.resolve() if successor_plan_path else None
    delta_output = delta_plan_path.resolve() if delta_plan_path else None
    acquisition_map_output = (
        acquisition_language_map_path.resolve()
        if acquisition_language_map_path is not None
        else None
    )
    migration_output = migration_plan_path.resolve() if migration_plan_path else None
    migration: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "status": "planned" if dry_run else "prepared",
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
                "path": str(successor_output) if successor_output else None,
                "payload_sha256": successor["plan_sha256"],
            },
            "delta_sampling_plan": {
                "path": str(delta_output) if delta_output else None,
                "payload_sha256": delta["plan_sha256"],
            },
            "acquisition_language_map": {
                "path": str(acquisition_map_output) if acquisition_map_output else None,
            },
        },
        "pinned_inputs": {
            label.replace(" ", "_"): {"path": str(path), "sha256": sha256_file(path)}
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
        "canonical_audit_path_policy": (
            "all adopted absolute paths must remain under the final pipeline release root; "
            "otherwise re-home and re-hash the full pinned artifact graph before audit"
        ),
    }
    acquisition_map = {
        # This artifact is consumed by bootstrap_fineweb_v2, whose language-map
        # contract is independent of the migration-plan schema and requires v2.
        "schema_version": LANGUAGE_MAP_SCHEMA_VERSION,
        "language_set_id": "multilingual_moe_ab_v2_K12_incremental_acquisition",
        "0": {"languages": transition["acquisition_languages"]},
        "provenance": {
            "tool_id": TOOL_ID,
            "successor_sampling_plan_sha256": successor["plan_sha256"],
            "selection_rule": (
                "new physical target exceeds predecessor acquired train pool"
            ),
        },
    }
    migration["execution_contract"] = {
        "acquisition_worker_count": len(transition["acquisition_languages"]),
        "acquisition_array": f"0-{len(transition['acquisition_languages']) - 1}",
        "acquisition_language_order": transition["acquisition_languages"],
        "worker_interface": (
            "bootstrap_fineweb_v2 --language-map K200_IDENTITY_MAP "
            "--language-selection-map ACQUISITION_MAP --sampling-plan SUCCESSOR_PLAN "
            "--incremental-from PREDECESSOR_BOOTSTRAP --language-index"
        ),
        "merge_interface": (
            "bootstrap_fineweb_v2 --language-map K200_IDENTITY_MAP "
            "--language-selection-map ACQUISITION_MAP --sampling-plan SUCCESSOR_PLAN "
            "--incremental-from PREDECESSOR_BOOTSTRAP --merge-summary"
        ),
        "required_result": "exactly-nine-language hash-valid bootstrap manifest",
    }
    migration["migration_plan_sha256"] = canonical_json_sha256(migration)
    if dry_run:
        return {
            "migration": migration,
            "successor_plan": successor,
            "delta_plan": delta,
            "acquisition_language_map": acquisition_map,
        }
    if not all(
        (
            successor_output,
            delta_output,
            acquisition_map_output,
            migration_output,
            receipt_path,
        )
    ):
        raise K12MigrationError(
            "non-dry-run prepare requires successor, delta, acquisition-map, "
            "migration, and receipt paths"
        )
    assert (
        successor_output is not None
        and delta_output is not None
        and acquisition_map_output is not None
        and migration_output is not None
    )
    for output in (
        successor_output,
        delta_output,
        acquisition_map_output,
        migration_output,
        receipt_path.resolve(),
    ):
        try:
            output.relative_to(successor_root)
        except ValueError as error:
            raise K12MigrationError(
                f"K12 successor artifact must be under {successor_root}: {output}"
            ) from error
    _reject_output_inside_inputs(
        migration_output.parent, [predecessor_release_dir]
    )
    _write_frozen_json(successor_output, successor)
    _write_frozen_json(delta_output, delta)
    _write_frozen_json(acquisition_map_output, acquisition_map)
    migration["successor"]["sampling_plan"]["file_sha256"] = sha256_file(successor_output)
    migration["successor"]["delta_sampling_plan"]["file_sha256"] = sha256_file(delta_output)
    migration["successor"]["acquisition_language_map"]["file_sha256"] = sha256_file(
        acquisition_map_output
    )
    migration.pop("migration_plan_sha256", None)
    migration["migration_plan_sha256"] = canonical_json_sha256(migration)
    _write_frozen_json(migration_output, migration)
    receipt = _receipt(
        "prepare",
        {
            "migration_plan": {
                "path": str(migration_output),
                "sha256": sha256_file(migration_output),
                "payload_sha256": migration["migration_plan_sha256"],
            },
            "successor_sampling_plan": {
                "path": str(successor_output),
                "sha256": sha256_file(successor_output),
                "payload_sha256": successor["plan_sha256"],
            },
            "delta_sampling_plan": {
                "path": str(delta_output),
                "sha256": sha256_file(delta_output),
                "payload_sha256": delta["plan_sha256"],
            },
            "acquisition_language_map": {
                "path": str(acquisition_map_output),
                "sha256": sha256_file(acquisition_map_output),
                "languages": transition["acquisition_languages"],
            },
            "transition": transition,
        },
    )
    _write_frozen_json(receipt_path.resolve(), receipt)
    return receipt


def _load_migration(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    migration = _load_object(path.resolve(), label="K12 migration plan")
    _verify_payload(migration, "migration_plan_sha256", label="K12 migration plan")
    if migration.get("tool_id") != TOOL_ID or migration.get("status") != "prepared":
        raise K12MigrationError("K12 migration plan is not prepared")
    adoption_meta = migration.get("upstream_adoption")
    if not isinstance(adoption_meta, dict):
        raise K12MigrationError("K12 migration plan lacks upstream-adoption binding")
    adoption_path = Path(str(adoption_meta.get("path", ""))).resolve()
    adoption = _load_object(adoption_path, label="upstream adoption manifest")
    adoption_hash = _verify_payload(
        adoption, "adoption_manifest_sha256", label="upstream adoption manifest"
    )
    predecessor_root = Path(str(adoption_meta.get("predecessor_root", ""))).resolve()
    successor_root = Path(str(adoption_meta.get("successor_root", ""))).resolve()
    if (
        predecessor_root == successor_root
        or adoption_meta.get("file_sha256") != sha256_file(adoption_path)
        or adoption_meta.get("payload_sha256") != adoption_hash
        or Path(str(adoption.get("predecessor_root", ""))).resolve()
        != predecessor_root
        or Path(str(adoption.get("successor_root", ""))).resolve() != successor_root
    ):
        raise K12MigrationError("K12 upstream-adoption binding changed")
    successor_meta = migration.get("successor")
    if not isinstance(successor_meta, dict):
        raise K12MigrationError("K12 migration plan lacks successor metadata")
    full_meta = successor_meta.get("sampling_plan")
    delta_meta = successor_meta.get("delta_sampling_plan")
    if not isinstance(full_meta, dict) or not isinstance(delta_meta, dict):
        raise K12MigrationError("K12 migration plan lacks plan artifacts")
    full_path = Path(str(full_meta.get("path", ""))).resolve()
    delta_path = Path(str(delta_meta.get("path", ""))).resolve()
    full = _verify_plan(full_path, label="successor sampling")
    delta = _verify_plan(delta_path, label="delta sampling")
    for meta, candidate, payload in ((full_meta, full_path, full), (delta_meta, delta_path, delta)):
        if meta.get("file_sha256") != sha256_file(candidate) or meta.get(
            "payload_sha256"
        ) != payload.get("plan_sha256"):
            raise K12MigrationError(f"migration-pinned plan changed: {candidate}")
    acquisition_meta = successor_meta.get("acquisition_language_map")
    if not isinstance(acquisition_meta, dict):
        raise K12MigrationError("K12 migration plan lacks acquisition language map")
    acquisition_path = Path(str(acquisition_meta.get("path", ""))).resolve()
    acquisition_map = _load_object(acquisition_path, label="acquisition language map")
    acquisition_languages = acquisition_map.get("0", {}).get("languages")
    if (
        int(acquisition_map.get("schema_version", 0)) < LANGUAGE_MAP_SCHEMA_VERSION
        or acquisition_meta.get("file_sha256") != sha256_file(acquisition_path)
        or acquisition_languages != migration["transition"]["acquisition_languages"]
    ):
        raise K12MigrationError("migration-pinned acquisition language map changed")
    for candidate in (full_path, delta_path, acquisition_path, path.resolve()):
        try:
            candidate.relative_to(successor_root)
        except ValueError as error:
            raise K12MigrationError(
                f"successor migration artifact escapes {successor_root}: {candidate}"
            ) from error
    return migration, full, delta
