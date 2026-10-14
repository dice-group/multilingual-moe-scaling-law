"""Bootstrap merge workers and publication for K12 migration."""

from __future__ import annotations

import copy
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.k12_migration.common import (
    K12MigrationError,
    _install_or_verify,
    _load_bootstrap_manifest_only,
    _load_object,
    _load_worker_receipt,
    _migration_roots,
    _publish_verified_worker_file,
    _receipt,
    _reject_output_inside_inputs,
    _require_under,
    _safe_artifact,
    _tier_languages,
    _verify_bootstrap_entry,
    _write_frozen_json,
)
from multilingual_megatron.experiments.scaling.k12_migration.constants import (
    SCHEMA_VERSION,
    TOOL_ID,
)
from multilingual_megatron.experiments.scaling.k12_migration.planning import (
    _load_migration,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)


def _bootstrap_entry_source(
    root: Path, manifest: Mapping[str, Any], language: str
) -> tuple[dict[str, Any], Path]:
    raw = manifest["languages"].get(language)
    if not isinstance(raw, dict):
        raise K12MigrationError(f"bootstrap source lacks {language}")
    report = raw.get("acquisition", {}).get("report_path")
    report_path = _safe_artifact(root, report, label=f"bootstrap/{language}/report")
    return raw, report_path.parent


def bootstrap_merge_worker(
    *,
    migration_plan_path: Path,
    predecessor_bootstrap_dir: Path,
    incremental_bootstrap_dir: Path,
    work_root: Path,
    language_index: int,
) -> dict[str, Any]:
    """Materialize one independently owned bootstrap language for a Slurm array."""

    migration, full_plan, _ = _load_migration(migration_plan_path)
    predecessor_root, successor_root = _migration_roots(migration)
    predecessor_bootstrap_dir = predecessor_bootstrap_dir.resolve()
    incremental_bootstrap_dir = incremental_bootstrap_dir.resolve()
    work_root = work_root.resolve()
    _require_under(
        predecessor_bootstrap_dir, predecessor_root, label="predecessor bootstrap"
    )
    _require_under(
        incremental_bootstrap_dir, successor_root, label="incremental bootstrap"
    )
    _require_under(work_root, successor_root, label="bootstrap merge work root")
    languages = _tier_languages(full_plan, "K200")
    if not 0 <= int(language_index) < len(languages):
        raise K12MigrationError(f"bootstrap language index is out of range: {language_index}")
    language = languages[int(language_index)]
    predecessor = _load_bootstrap_manifest_only(
        predecessor_bootstrap_dir, label="predecessor bootstrap"
    )
    incremental = _load_bootstrap_manifest_only(
        incremental_bootstrap_dir, label="incremental bootstrap"
    )
    acquisition = set(migration["transition"]["acquisition_languages"])
    if set(predecessor["languages"]) != set(languages) or set(
        incremental["languages"]
    ) != acquisition:
        raise K12MigrationError("bootstrap worker input coverage differs from migration")
    pinned = migration["predecessor"]["production_bootstrap"]
    if (
        Path(str(pinned["path"])).resolve() != predecessor_bootstrap_dir
        or pinned["manifest_file_sha256"]
        != sha256_file(predecessor_bootstrap_dir / "bootstrap_manifest.json")
        or pinned["manifest_payload_sha256"] != predecessor["manifest_sha256"]
    ):
        raise K12MigrationError("bootstrap worker predecessor is not migration-pinned")
    source_root = (
        incremental_bootstrap_dir if language in acquisition else predecessor_bootstrap_dir
    )
    source_manifest = incremental if language in acquisition else predecessor
    entry, source_language_dir = _verify_bootstrap_entry(
        source_root, source_manifest, language, label="bootstrap worker source"
    )
    report = _load_object(
        source_language_dir / "bootstrap_report.json", label=f"{language} report"
    )
    if language in acquisition:
        production = report.get("run_config", {}).get("production_target")
        extension = report.get("run_config", {}).get("incremental_extension")
        predecessor_hash = predecessor["languages"][language]["acquisition"][
            "report_payload_sha256"
        ]
        if (
            not isinstance(production, dict)
            or production.get("sampling_plan_sha256") != full_plan["plan_sha256"]
            or not isinstance(extension, dict)
            or extension.get("mode") != "verified_full_deterministic_rebuild"
            or extension.get("predecessor_report_payload_sha256") != predecessor_hash
        ):
            raise K12MigrationError(f"{language}: incremental lineage is invalid")
    target_dir = work_root / "bootstrap_languages" / language
    installed: dict[str, Any] = {}
    for source in sorted(source_language_dir.iterdir()):
        if source.is_symlink():
            raise K12MigrationError(f"bootstrap source contains symlink: {source}")
        if not source.is_file():
            continue
        target = target_dir / source.name
        method = _install_or_verify(
            source, target, force_copy=language not in acquisition
        )
        installed[source.name] = {
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
            "method": method,
            "mode": target.stat().st_mode & 0o777,
        }
    receipt = _receipt(
        "bootstrap-merge-worker",
        {
            "migration_plan_sha256": migration["migration_plan_sha256"],
            "language_index": int(language_index),
            "language": language,
            "source": "incremental" if language in acquisition else "predecessor",
            "source_bootstrap_manifest_sha256": source_manifest["manifest_sha256"],
            "entry": entry,
            "installed_artifacts": installed,
            "work_language_dir": str(target_dir),
            "predecessor_inode_sharing": False,
        },
    )
    receipt_path = work_root / "receipts" / f"bootstrap.{int(language_index):03d}.json"
    _write_frozen_json(receipt_path, receipt)
    return receipt


def finalize_bootstrap_merge(
    *,
    migration_plan_path: Path,
    predecessor_bootstrap_dir: Path,
    incremental_bootstrap_dir: Path,
    work_root: Path,
    output_dir: Path,
    receipt_path: Path,
    execution_mode: str = "parallel",
) -> dict[str, Any]:
    """Verify all bootstrap workers and atomically publish shared metadata."""

    migration, full_plan, _ = _load_migration(migration_plan_path)
    predecessor_root, successor_root = _migration_roots(migration)
    predecessor_bootstrap_dir = predecessor_bootstrap_dir.resolve()
    incremental_bootstrap_dir = incremental_bootstrap_dir.resolve()
    work_root = work_root.resolve()
    output_dir = output_dir.resolve()
    receipt_path = receipt_path.resolve()
    if execution_mode not in {"parallel", "serial"}:
        raise K12MigrationError(
            f"unsupported bootstrap merge execution mode: {execution_mode}"
        )
    _require_under(
        predecessor_bootstrap_dir, predecessor_root, label="predecessor bootstrap"
    )
    _require_under(
        incremental_bootstrap_dir, successor_root, label="incremental bootstrap"
    )
    _require_under(work_root, successor_root, label="bootstrap merge work root")
    _require_under(output_dir, successor_root, label="merged bootstrap")
    _require_under(receipt_path, successor_root, label="bootstrap receipt")
    _reject_output_inside_inputs(
        output_dir,
        [predecessor_bootstrap_dir, incremental_bootstrap_dir, work_root],
    )
    _reject_output_inside_inputs(
        receipt_path,
        [predecessor_bootstrap_dir, incremental_bootstrap_dir, output_dir, work_root],
    )
    predecessor = _load_bootstrap_manifest_only(
        predecessor_bootstrap_dir, label="predecessor bootstrap"
    )
    incremental = _load_bootstrap_manifest_only(
        incremental_bootstrap_dir, label="incremental bootstrap"
    )
    languages = _tier_languages(full_plan, "K200")
    acquisition = set(migration["transition"]["acquisition_languages"])
    if set(predecessor["languages"]) != set(languages) or set(
        incremental["languages"]
    ) != acquisition:
        raise K12MigrationError("bootstrap summary input coverage differs from migration")
    pinned = migration["predecessor"]["production_bootstrap"]
    if (
        Path(str(pinned["path"])).resolve() != predecessor_bootstrap_dir
        or pinned["manifest_file_sha256"]
        != sha256_file(predecessor_bootstrap_dir / "bootstrap_manifest.json")
        or pinned["manifest_payload_sha256"] != predecessor["manifest_sha256"]
    ):
        raise K12MigrationError("bootstrap summary predecessor is not migration-pinned")
    if output_dir.exists() or output_dir.is_symlink():
        raise K12MigrationError(f"bootstrap migration output already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    entries: dict[str, Any] = {}
    sources: dict[str, str] = {}
    worker_receipts: dict[str, str] = {}
    install_methods: dict[str, str] = {}
    try:
        for index, language in enumerate(languages):
            worker_path = work_root / "receipts" / f"bootstrap.{index:03d}.json"
            worker = _load_worker_receipt(
                worker_path, command="bootstrap-merge-worker"
            )
            expected_source = "incremental" if language in acquisition else "predecessor"
            expected_manifest = (
                incremental if expected_source == "incremental" else predecessor
            )
            if (
                worker.get("migration_plan_sha256")
                != migration["migration_plan_sha256"]
                or worker.get("language_index") != index
                or worker.get("language") != language
                or worker.get("source") != expected_source
                or worker.get("source_bootstrap_manifest_sha256")
                != expected_manifest["manifest_sha256"]
                or worker.get("predecessor_inode_sharing") is not False
            ):
                raise K12MigrationError(f"bootstrap worker receipt mismatch: {language}")
            source_dir = Path(str(worker["work_language_dir"])).resolve()
            _require_under(source_dir, work_root, label=f"{language} worker directory")
            installed = worker.get("installed_artifacts")
            if not isinstance(installed, dict) or not installed:
                raise K12MigrationError(f"bootstrap worker has no artifacts: {language}")
            predecessor_language_dir: Path | None = None
            if expected_source == "predecessor":
                _, predecessor_language_dir = _bootstrap_entry_source(
                    predecessor_bootstrap_dir, predecessor, language
                )
            for name, raw in installed.items():
                if Path(name).name != name:
                    raise K12MigrationError(
                        f"bootstrap worker artifact name is invalid: {language}/{name}"
                    )
                source = _safe_artifact(source_dir, name, label=f"{language} worker artifact")
                if (
                    not isinstance(raw, dict)
                    or sha256_file(source) != raw.get("sha256")
                    or source.stat().st_mode & 0o222
                ):
                    raise K12MigrationError(f"bootstrap worker artifact changed: {source}")
                if predecessor_language_dir is not None:
                    predecessor_source = _safe_artifact(
                        predecessor_language_dir,
                        name,
                        label=f"{language} predecessor bootstrap artifact",
                    )
                    if (
                        raw.get("method")
                        != "copy_predecessor_then_make_read_only"
                        or (source.stat().st_dev, source.stat().st_ino)
                        == (
                            predecessor_source.stat().st_dev,
                            predecessor_source.stat().st_ino,
                        )
                    ):
                        raise K12MigrationError(
                            f"bootstrap worker shares predecessor inode: {language}/{name}"
                        )
                relative = f"{language}/{name}"
                install_methods[relative] = _publish_verified_worker_file(
                    source, staging / relative
                )
            entry = copy.deepcopy(worker.get("entry"))
            if not isinstance(entry, dict):
                raise K12MigrationError(f"bootstrap worker entry is invalid: {language}")
            entry["acquisition"]["report_path"] = f"{language}/bootstrap_report.json"
            entry["train"]["path"] = f"{language}/train.jsonl"
            entry["validation_reserve"]["path"] = f"{language}/validation_reserve.jsonl"
            entries[language] = entry
            sources[language] = expected_source
            worker_receipts[language] = sha256_file(worker_path)
        provenance = copy.deepcopy(predecessor.get("provenance", {}))
        provenance.update(
            {
                "tool_id": "fineweb_ab_v2_bootstrap_v1",
                "sampling_plan_file_sha256": migration["successor"]["sampling_plan"][
                    "file_sha256"
                ],
                "sampling_plan_payload_sha256": full_plan["plan_sha256"],
                "tokenizer_acquisition_plan_file_sha256": None,
                "tokenizer_acquisition_plan_payload_sha256": None,
                "incremental_mode": "verified_full_deterministic_rebuild",
                "k12_migration": {
                    "tool_id": TOOL_ID,
                    "migration_plan_sha256": migration["migration_plan_sha256"],
                    "predecessor_bootstrap_manifest_sha256": predecessor[
                        "manifest_sha256"
                    ],
                    "incremental_bootstrap_manifest_sha256": incremental[
                        "manifest_sha256"
                    ],
                    "entry_sources": sources,
                },
            }
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "provenance": provenance,
            "languages": entries,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        _write_frozen_json(staging / "bootstrap_manifest.json", manifest)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "tool_id": "fineweb_ab_v2_bootstrap_v1",
            "languages": languages,
            "report_payload_sha256": {
                language: entries[language]["acquisition"]["report_payload_sha256"]
                for language in languages
            },
            "bootstrap_manifest_payload_sha256": manifest["manifest_sha256"],
        }
        summary["summary_sha256"] = canonical_json_sha256(summary)
        _write_frozen_json(staging / "bootstrap_summary.json", summary)
        delta_manifest = copy.deepcopy(manifest)
        delta_manifest["provenance"]["view"] = "changed_physical_languages_only"
        delta_manifest["languages"] = {
            language: delta_manifest["languages"][language]
            for language in migration["transition"]["delta_finalize_languages"]
        }
        delta_manifest.pop("manifest_sha256", None)
        delta_manifest["manifest_sha256"] = canonical_json_sha256(delta_manifest)
        _write_frozen_json(staging / "bootstrap_delta_manifest.json", delta_manifest)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = _receipt(
        "bootstrap-merge",
        {
            "migration_plan_sha256": migration["migration_plan_sha256"],
            "bootstrap_manifest": {
                "sha256": sha256_file(output_dir / "bootstrap_manifest.json"),
                "payload_sha256": manifest["manifest_sha256"],
            },
            "bootstrap_summary": {
                "sha256": sha256_file(output_dir / "bootstrap_summary.json"),
                "payload_sha256": summary["summary_sha256"],
            },
            "delta_bootstrap_manifest": {
                "sha256": sha256_file(output_dir / "bootstrap_delta_manifest.json"),
                "payload_sha256": delta_manifest["manifest_sha256"],
            },
            "entry_sources": sources,
            "predecessor_inode_sharing": False,
            "execution_metadata": {
                "mode": execution_mode,
                "output_dir": str(output_dir),
                "work_root": str(work_root),
                "worker_receipts": worker_receipts,
                "publication_methods": install_methods,
            },
        },
    )
    _write_frozen_json(receipt_path, receipt)
    return receipt


def merge_bootstraps(
    *,
    migration_plan_path: Path,
    predecessor_bootstrap_dir: Path,
    incremental_bootstrap_dir: Path,
    output_dir: Path,
    receipt_path: Path,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Run all language workers locally, then use the shared finalizer."""

    migration, full_plan, _ = _load_migration(migration_plan_path)
    predecessor_root, successor_root = _migration_roots(migration)
    resolved_predecessor = predecessor_bootstrap_dir.resolve()
    resolved_incremental = incremental_bootstrap_dir.resolve()
    resolved_output = output_dir.resolve()
    resolved_receipt = receipt_path.resolve()
    resolved_work_root = (
        work_root.resolve()
        if work_root is not None
        else resolved_output.with_name(f".{resolved_output.name}.merge-work")
    )
    _require_under(
        resolved_predecessor, predecessor_root, label="predecessor bootstrap"
    )
    for candidate, label in (
        (resolved_incremental, "incremental bootstrap"),
        (resolved_work_root, "bootstrap merge work root"),
        (resolved_output, "merged bootstrap"),
        (resolved_receipt, "bootstrap receipt"),
    ):
        _require_under(candidate, successor_root, label=label)
    _reject_output_inside_inputs(
        resolved_output,
        [resolved_predecessor, resolved_incremental, resolved_work_root],
    )
    _reject_output_inside_inputs(
        resolved_receipt,
        [
            resolved_predecessor,
            resolved_incremental,
            resolved_output,
            resolved_work_root,
        ],
    )
    if resolved_output.exists() or resolved_output.is_symlink():
        raise K12MigrationError(
            f"bootstrap migration output already exists: {resolved_output}"
        )
    for language_index, _ in enumerate(_tier_languages(full_plan, "K200")):
        bootstrap_merge_worker(
            migration_plan_path=migration_plan_path,
            predecessor_bootstrap_dir=resolved_predecessor,
            incremental_bootstrap_dir=resolved_incremental,
            work_root=resolved_work_root,
            language_index=language_index,
        )
    return finalize_bootstrap_merge(
        migration_plan_path=migration_plan_path,
        predecessor_bootstrap_dir=resolved_predecessor,
        incremental_bootstrap_dir=resolved_incremental,
        work_root=resolved_work_root,
        output_dir=resolved_output,
        receipt_path=resolved_receipt,
        execution_mode="serial",
    )
