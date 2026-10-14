"""Release merge workers and publication for K12 migration."""

from __future__ import annotations

import copy
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.k12_migration.common import (
    K12MigrationError,
    _expected_exposure_audit,
    _install_file,
    _install_or_verify,
    _load_object,
    _load_release_manifests_only,
    _load_worker_receipt,
    _migration_roots,
    _physical_entry,
    _publish_verified_worker_file,
    _receipt,
    _reject_output_inside_inputs,
    _require_under,
    _safe_artifact,
    _tier_languages,
    _validate_language_against_plan,
    _verify_payload,
    _verify_plan,
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


def _source_prefix(manifest: Mapping[str, Any], root: Path) -> dict[str, Any]:
    raw = manifest.get("validation_prefix_inputs")
    if not isinstance(raw, dict):
        raise K12MigrationError("release lacks validation_prefix_inputs metadata")
    path = _safe_artifact(root, raw.get("path"), label="validation prefix inputs")
    if sha256_file(path) != raw.get("sha256"):
        raise K12MigrationError("validation prefix inputs hash mismatch")
    payload = _load_object(path, label="validation prefix inputs")
    inputs = payload.get("inputs_by_language")
    if not isinstance(inputs, dict):
        raise K12MigrationError("validation prefix inputs lacks inputs_by_language")
    return inputs


def _link_release_language(
    *,
    source_root: Path,
    destination_root: Path,
    language: str,
    payload: Mapping[str, Any],
    force_copy: bool = False,
) -> dict[str, str]:
    train = payload["train"]
    validation = payload["validation"]
    relatives = (
        train["output"]["path"],
        validation["output"]["path"],
        train["unigram_counts"]["path"],
    )
    methods: dict[str, str] = {}
    for relative in relatives:
        source = _safe_artifact(source_root, relative, label=f"{language} release artifact")
        methods[str(relative)] = _install_file(
            source, destination_root / str(relative), force_copy=force_copy
        )
    return methods


def _verify_delta_bootstrap_identity(
    *,
    delta_release: Mapping[str, Any],
    delta_bootstrap_path: Path,
    delta_bootstrap_payload_sha256: str,
) -> None:
    """Verify finalizer lineage across old and current bootstrap schemas.

    The parallel finalizer records the exact bootstrap file hash but older
    manifests do not duplicate its canonical payload hash.  The file hash plus
    an independent payload verification binds the same identity without making
    the optional duplicate field authoritative.
    """

    metadata = delta_release.get("inputs", {}).get("bootstrap_manifest")
    if not isinstance(metadata, Mapping):
        raise K12MigrationError("delta release lacks bootstrap lineage")
    recorded_path = Path(str(metadata.get("path", "")))
    if (
        not recorded_path.is_absolute()
        or recorded_path != delta_bootstrap_path
        or delta_bootstrap_path.is_symlink()
        or not delta_bootstrap_path.is_file()
        or metadata.get("sha256") != sha256_file(delta_bootstrap_path)
    ):
        raise K12MigrationError("delta release belongs to another bootstrap delta")
    declared_payload = metadata.get("payload_sha256")
    if declared_payload is not None and declared_payload != delta_bootstrap_payload_sha256:
        raise K12MigrationError("delta release belongs to another bootstrap delta")


def _refresh_exposure_pass_audit(
    payload: dict[str, Any], *, plan: Mapping[str, Any], language: str
) -> None:
    """Bind release metadata to every exposure group in the full plan."""

    train = payload.get("train")
    if not isinstance(train, dict):
        raise K12MigrationError(f"{language}: release payload lacks train metadata")
    payload["exposure_pass_audit"] = _expected_exposure_audit(
        plan, language, int(train.get("selected_tokens", 0))
    )


def _verify_cross_root_tokenizer_contract(
    *,
    predecessor_tokenizer: Any,
    delta_tokenizer: Any,
    predecessor_root: Path,
    successor_root: Path,
    successor_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind equal tokenizer identity to each root's canonical operational path."""

    if not isinstance(predecessor_tokenizer, dict) or not isinstance(delta_tokenizer, dict):
        raise K12MigrationError("predecessor and delta tokenizer contracts differ")
    predecessor_path = Path(str(predecessor_tokenizer.get("path", "")))
    successor_path = Path(str(delta_tokenizer.get("path", "")))
    expected_predecessor_path = (
        predecessor_root / "corpus/tokenizer/sp_unigram_64k/tokenizer.model"
    )
    expected_successor_path = (
        successor_root / "corpus/tokenizer/sp_unigram_64k/tokenizer.model"
    )
    if (
        not predecessor_path.is_absolute()
        or not successor_path.is_absolute()
        or predecessor_path != expected_predecessor_path
        or successor_path != expected_successor_path
    ):
        raise K12MigrationError("predecessor and delta tokenizer paths are not root-pinned")
    predecessor_identity = {
        key: value for key, value in predecessor_tokenizer.items() if key != "path"
    }
    successor_identity = {
        key: value for key, value in delta_tokenizer.items() if key != "path"
    }
    tokenizer_sha256 = successor_identity.get("sha256")
    if (
        predecessor_identity != successor_identity
        or not isinstance(tokenizer_sha256, str)
        or len(tokenizer_sha256) != 64
    ):
        raise K12MigrationError("predecessor and delta tokenizer contracts differ")
    for path, label in (
        (predecessor_path, "predecessor tokenizer"),
        (successor_path, "successor tokenizer"),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != tokenizer_sha256:
            raise K12MigrationError(f"{label} changed after it was pinned")
    if (
        predecessor_path.stat().st_dev,
        predecessor_path.stat().st_ino,
    ) == (
        successor_path.stat().st_dev,
        successor_path.stat().st_ino,
    ):
        raise K12MigrationError("successor tokenizer shares its sealed predecessor inode")
    plan_provenance = successor_plan.get("provenance")
    if (
        not isinstance(plan_provenance, Mapping)
        or Path(str(plan_provenance.get("tokenizer_model_path", ""))).resolve()
        != successor_path
        or plan_provenance.get("tokenizer_model_sha256") != tokenizer_sha256
    ):
        raise K12MigrationError("successor sampling plan uses another tokenizer")
    return copy.deepcopy(delta_tokenizer)


def _load_parallel_release_context(
    *,
    migration_plan_path: Path,
    predecessor_release_dir: Path,
    delta_release_dir: Path,
    bootstrap_manifest_path: Path,
) -> dict[str, Any]:
    """Load the small, hash-addressed metadata shared by release workers."""

    migration, full_plan, delta_plan = _load_migration(migration_plan_path)
    predecessor_root, successor_root = _migration_roots(migration)
    predecessor_release_dir = predecessor_release_dir.resolve()
    delta_release_dir = delta_release_dir.resolve()
    bootstrap_manifest_path = bootstrap_manifest_path.resolve()
    _require_under(
        predecessor_release_dir, predecessor_root, label="predecessor release"
    )
    _require_under(delta_release_dir, successor_root, label="delta release")
    _require_under(
        bootstrap_manifest_path, successor_root, label="merged bootstrap manifest"
    )
    predecessor, predecessor_hashes = _load_release_manifests_only(
        predecessor_release_dir, label="predecessor release"
    )
    delta, delta_hashes = _load_release_manifests_only(
        delta_release_dir, label="delta release"
    )
    transition = migration["transition"]
    changed = set(transition["delta_finalize_languages"])
    reused = set(transition["reused_physical_languages"])
    if set(delta.get("languages", {})) != changed or set(
        delta_plan["physical_pool"]
    ) != changed:
        raise K12MigrationError(
            "delta release coverage differs from the frozen physical delta"
        )
    if set(predecessor.get("languages", {})) != changed | reused:
        raise K12MigrationError(
            "predecessor release coverage differs from migration plan"
        )
    if predecessor.get("selection_contract") != delta.get("selection_contract"):
        raise K12MigrationError("predecessor and delta selection contracts differ")
    pinned_release = migration.get("predecessor", {}).get("release")
    predecessor_manifest_path = predecessor_release_dir / "corpus_manifest.json"
    if (
        not isinstance(pinned_release, dict)
        or Path(str(pinned_release.get("path", ""))).resolve()
        != predecessor_release_dir
        or pinned_release.get("corpus_manifest_sha256")
        != sha256_file(predecessor_manifest_path)
        or pinned_release.get("corpus_manifest_payload_sha256")
        != predecessor.get("manifest_sha256")
    ):
        raise K12MigrationError("predecessor release is not migration-pinned")
    old_plan_meta = predecessor.get("inputs", {}).get("sampling_plan", {})
    if old_plan_meta.get("payload_sha256") != migration["predecessor"][
        "sampling_plan"
    ]["payload_sha256"]:
        raise K12MigrationError("predecessor release belongs to another sampling plan")
    delta_plan_meta = delta.get("inputs", {}).get("sampling_plan", {})
    if delta_plan_meta.get("payload_sha256") != delta_plan["plan_sha256"]:
        raise K12MigrationError("delta release belongs to another delta plan")
    delta_bootstrap_path = bootstrap_manifest_path.with_name(
        "bootstrap_delta_manifest.json"
    )
    delta_bootstrap = _load_object(
        delta_bootstrap_path, label="merged delta bootstrap manifest"
    )
    delta_bootstrap_hash = _verify_payload(
        delta_bootstrap,
        "manifest_sha256",
        label="merged delta bootstrap manifest",
    )
    _verify_delta_bootstrap_identity(
        delta_release=delta,
        delta_bootstrap_path=delta_bootstrap_path,
        delta_bootstrap_payload_sha256=delta_bootstrap_hash,
    )
    predecessor_tokenizer = predecessor.get("inputs", {}).get("tokenizer_model")
    delta_tokenizer = delta.get("inputs", {}).get("tokenizer_model")
    successor_tokenizer = _verify_cross_root_tokenizer_contract(
        predecessor_tokenizer=predecessor_tokenizer,
        delta_tokenizer=delta_tokenizer,
        predecessor_root=predecessor_root,
        successor_root=successor_root,
        successor_plan=full_plan,
    )
    bootstrap = _load_object(
        bootstrap_manifest_path, label="merged bootstrap manifest"
    )
    bootstrap_hash = _verify_payload(
        bootstrap, "manifest_sha256", label="merged bootstrap manifest"
    )
    if set(bootstrap.get("languages", {})) != set(full_plan["physical_pool"]):
        raise K12MigrationError(
            "merged bootstrap coverage differs from full successor plan"
        )
    predecessor_plan_path = Path(
        str(migration["predecessor"]["sampling_plan"]["path"])
    ).resolve()
    predecessor_plan = _verify_plan(
        predecessor_plan_path, label="predecessor sampling"
    )
    predecessor_plan_meta = migration["predecessor"]["sampling_plan"]
    if (
        predecessor_plan_meta.get("file_sha256")
        != sha256_file(predecessor_plan_path)
        or predecessor_plan_meta.get("payload_sha256")
        != predecessor_plan["plan_sha256"]
    ):
        raise K12MigrationError("predecessor sampling plan is not migration-pinned")
    return {
        "migration": migration,
        "full_plan": full_plan,
        "predecessor_plan": predecessor_plan,
        "predecessor": predecessor,
        "predecessor_hashes": predecessor_hashes,
        "delta": delta,
        "delta_hashes": delta_hashes,
        "bootstrap": bootstrap,
        "bootstrap_hash": bootstrap_hash,
        "predecessor_tokenizer": predecessor_tokenizer,
        "successor_tokenizer": successor_tokenizer,
        "predecessor_release_dir": predecessor_release_dir,
        "delta_release_dir": delta_release_dir,
        "bootstrap_manifest_path": bootstrap_manifest_path,
        "successor_root": successor_root,
    }


def _release_artifact_metadata(
    language: str, payload: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    try:
        artifacts = {
            "train": payload["train"]["output"],
            "validation": payload["validation"]["output"],
            "unigram_counts": payload["train"]["unigram_counts"],
        }
    except (KeyError, TypeError) as error:
        raise K12MigrationError(
            f"{language}: release artifact metadata is incomplete"
        ) from error
    if not all(isinstance(raw, Mapping) for raw in artifacts.values()):
        raise K12MigrationError(f"{language}: release artifact metadata is invalid")
    return artifacts


def _verify_release_language_artifacts(
    *,
    source_root: Path,
    hash_manifest: Mapping[str, Any],
    language: str,
    payload: Mapping[str, Any],
) -> list[tuple[str, Path, str]]:
    files = hash_manifest.get("files")
    if not isinstance(files, dict):
        raise K12MigrationError("release hash manifest has no file map")
    verified: list[tuple[str, Path, str]] = []
    for name, raw in _release_artifact_metadata(language, payload).items():
        relative = str(raw.get("path", ""))
        expected = raw.get("sha256")
        source = _safe_artifact(
            source_root, relative, label=f"{language}/{name} release artifact"
        )
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or files.get(relative) != expected
            or sha256_file(source) != expected
        ):
            raise K12MigrationError(
                f"{language}/{name} differs from its release hash manifest"
            )
        verified.append((relative, source, expected))
    if len({relative for relative, _, _ in verified}) != len(verified):
        raise K12MigrationError(f"{language}: release artifact paths overlap")
    return verified


def release_merge_worker(
    *,
    migration_plan_path: Path,
    predecessor_release_dir: Path,
    delta_release_dir: Path,
    bootstrap_manifest_path: Path,
    work_root: Path,
    language_index: int,
) -> dict[str, Any]:
    """Materialize one release language without scanning either full release."""

    context = _load_parallel_release_context(
        migration_plan_path=migration_plan_path,
        predecessor_release_dir=predecessor_release_dir,
        delta_release_dir=delta_release_dir,
        bootstrap_manifest_path=bootstrap_manifest_path,
    )
    migration = context["migration"]
    full_plan = context["full_plan"]
    work_root = work_root.resolve()
    _require_under(work_root, context["successor_root"], label="release merge work root")
    languages = _tier_languages(full_plan, "K200")
    if not 0 <= int(language_index) < len(languages):
        raise K12MigrationError(f"release language index is out of range: {language_index}")
    language = languages[int(language_index)]
    changed = set(migration["transition"]["delta_finalize_languages"])
    if language in changed:
        source_name = "delta"
        source_root = context["delta_release_dir"]
        source_manifest = context["delta"]
        source_hashes = context["delta_hashes"]
        payload = copy.deepcopy(source_manifest["languages"][language])
    else:
        source_name = "predecessor"
        source_root = context["predecessor_release_dir"]
        source_manifest = context["predecessor"]
        source_hashes = context["predecessor_hashes"]
        payload = copy.deepcopy(source_manifest["languages"][language])
        if _physical_entry(context["predecessor_plan"], language) != _physical_entry(
            full_plan, language
        ):
            raise K12MigrationError(
                f"{language}: reused language physical contract changed"
            )
    _refresh_exposure_pass_audit(payload, plan=full_plan, language=language)
    _validate_language_against_plan(
        language, payload, full_plan, require_exposure=True
    )
    prefix_meta = source_manifest.get("validation_prefix_inputs")
    if not isinstance(prefix_meta, dict):
        raise K12MigrationError(f"{source_name} release lacks validation-prefix metadata")
    prefix_relative = str(prefix_meta.get("path", ""))
    if source_hashes["files"].get(prefix_relative) != prefix_meta.get("sha256"):
        raise K12MigrationError(
            f"{source_name} validation-prefix hash manifest differs"
        )
    prefixes = _source_prefix(source_manifest, source_root)
    prefix = copy.deepcopy(prefixes.get(language))
    validation_output = payload["validation"]["output"]
    if (
        not isinstance(prefix, dict)
        or prefix.get("jsonl") != validation_output.get("path")
        or prefix.get("sha256") != validation_output.get("sha256")
    ):
        raise K12MigrationError(f"{language}: validation-prefix entry is invalid")
    target_dir = work_root / "release_languages" / language
    installed: dict[str, Any] = {}
    for relative, source, expected in _verify_release_language_artifacts(
        source_root=source_root,
        hash_manifest=source_hashes,
        language=language,
        payload=payload,
    ):
        target = target_dir / relative
        method = _install_or_verify(
            source, target, force_copy=source_name == "predecessor"
        )
        if sha256_file(target) != expected or target.stat().st_mode & 0o222:
            raise K12MigrationError(f"release worker target verification failed: {target}")
        installed[relative] = {
            "sha256": expected,
            "size_bytes": target.stat().st_size,
            "method": method,
            "mode": target.stat().st_mode & 0o777,
        }
    receipt = _receipt(
        "release-merge-worker",
        {
            "migration_plan_sha256": migration["migration_plan_sha256"],
            "language_index": int(language_index),
            "language": language,
            "source": source_name,
            "source_corpus_manifest_sha256": source_manifest["manifest_sha256"],
            "source_hash_manifest_sha256": source_hashes[
                "hash_manifest_payload_sha256"
            ],
            "language_payload": payload,
            "validation_prefix_input": prefix,
            "installed_artifacts": installed,
            "work_language_dir": str(target_dir),
            "predecessor_inode_sharing": False,
        },
    )
    receipt_path = work_root / "receipts" / f"release.{int(language_index):03d}.json"
    _write_frozen_json(receipt_path, receipt)
    return receipt


def finalize_release_merge(
    *,
    migration_plan_path: Path,
    predecessor_release_dir: Path,
    delta_release_dir: Path,
    bootstrap_manifest_path: Path,
    work_root: Path,
    output_dir: Path,
    receipt_path: Path,
    execution_mode: str = "parallel",
) -> dict[str, Any]:
    """Verify all 200 release receipts and atomically publish the K12 release."""

    context = _load_parallel_release_context(
        migration_plan_path=migration_plan_path,
        predecessor_release_dir=predecessor_release_dir,
        delta_release_dir=delta_release_dir,
        bootstrap_manifest_path=bootstrap_manifest_path,
    )
    migration = context["migration"]
    full_plan = context["full_plan"]
    work_root = work_root.resolve()
    output_dir = output_dir.resolve()
    receipt_path = receipt_path.resolve()
    if execution_mode not in {"parallel", "serial"}:
        raise K12MigrationError(
            f"unsupported release merge execution mode: {execution_mode}"
        )
    for candidate, label in (
        (work_root, "release merge work root"),
        (output_dir, "merged release"),
        (receipt_path, "release receipt"),
    ):
        _require_under(candidate, context["successor_root"], label=label)
    _reject_output_inside_inputs(
        output_dir,
        [context["predecessor_release_dir"], context["delta_release_dir"], work_root],
    )
    _reject_output_inside_inputs(
        receipt_path,
        [
            context["predecessor_release_dir"],
            context["delta_release_dir"],
            output_dir,
            work_root,
        ],
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise K12MigrationError(f"release migration output already exists: {output_dir}")
    languages_order = _tier_languages(full_plan, "K200")
    changed = set(migration["transition"]["delta_finalize_languages"])
    staging = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    languages: dict[str, Any] = {}
    prefixes: dict[str, Any] = {}
    sources: dict[str, str] = {}
    worker_receipts: dict[str, str] = {}
    install_methods: dict[str, str] = {}
    publication_methods: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    try:
        for index, language in enumerate(languages_order):
            worker_path = work_root / "receipts" / f"release.{index:03d}.json"
            worker = _load_worker_receipt(
                worker_path, command="release-merge-worker"
            )
            expected_source = "delta" if language in changed else "predecessor"
            if (
                worker.get("migration_plan_sha256")
                != migration["migration_plan_sha256"]
                or worker.get("language_index") != index
                or worker.get("language") != language
                or worker.get("source") != expected_source
            ):
                raise K12MigrationError(f"release worker receipt mismatch: {language}")
            source_manifest = (
                context["delta"]
                if expected_source == "delta"
                else context["predecessor"]
            )
            source_hashes = (
                context["delta_hashes"]
                if expected_source == "delta"
                else context["predecessor_hashes"]
            )
            if (
                worker.get("source_corpus_manifest_sha256")
                != source_manifest["manifest_sha256"]
                or worker.get("source_hash_manifest_sha256")
                != source_hashes["hash_manifest_payload_sha256"]
            ):
                raise K12MigrationError(f"release worker source changed: {language}")
            payload = copy.deepcopy(worker.get("language_payload"))
            prefix = copy.deepcopy(worker.get("validation_prefix_input"))
            if not isinstance(payload, dict) or not isinstance(prefix, dict):
                raise K12MigrationError(f"release worker metadata is invalid: {language}")
            _validate_language_against_plan(
                language, payload, full_plan, require_exposure=True
            )
            validation_output = payload["validation"]["output"]
            if (
                prefix.get("jsonl") != validation_output.get("path")
                or prefix.get("sha256") != validation_output.get("sha256")
                or worker.get("predecessor_inode_sharing") is not False
            ):
                raise K12MigrationError(
                    f"release worker validation/provenance is invalid: {language}"
                )
            expected_artifacts = _release_artifact_metadata(language, payload)
            expected_relatives = {str(raw["path"]) for raw in expected_artifacts.values()}
            installed = worker.get("installed_artifacts")
            if not isinstance(installed, dict) or set(installed) != expected_relatives:
                raise K12MigrationError(
                    f"release worker artifact coverage is invalid: {language}"
                )
            source_dir = Path(str(worker.get("work_language_dir", ""))).resolve()
            _require_under(source_dir, work_root, label=f"{language} worker directory")
            for relative, raw in installed.items():
                source = _safe_artifact(
                    source_dir, relative, label=f"{language} worker artifact"
                )
                if (
                    not isinstance(raw, dict)
                    or sha256_file(source) != raw.get("sha256")
                    or source.stat().st_mode & 0o222
                ):
                    raise K12MigrationError(
                        f"release worker artifact changed: {source}"
                    )
                if expected_source == "predecessor":
                    predecessor_source = _safe_artifact(
                        context["predecessor_release_dir"],
                        relative,
                        label=f"{language} predecessor release artifact",
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
                            f"release worker shares predecessor inode: {language}/{relative}"
                        )
                publication_methods[relative] = _publish_verified_worker_file(
                    source, staging / relative
                )
                artifact_hashes[relative] = str(raw["sha256"])
                install_methods[relative] = str(raw["method"])
            languages[language] = payload
            prefixes[language] = prefix
            sources[language] = expected_source
            worker_receipts[language] = sha256_file(worker_path)
        successor_tokenizer = context["successor_tokenizer"]
        prefix_path = staging / "validation_prefix_inputs.json"
        prefix_payload = {
            "schema_version": SCHEMA_VERSION,
            "tokenizer_model_sha256": successor_tokenizer["sha256"],
            "append_eod": True,
            "inputs_by_language": dict(sorted(prefixes.items())),
        }
        _write_frozen_json(prefix_path, prefix_payload)
        bootstrap_manifest_path = context["bootstrap_manifest_path"]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "release_id": "fineweb_ab_v2_finalized",
            "release_status": "ready",
            "ready": True,
            "blocking_languages": {
                "needs_incremental_source_docs": [],
                "needs_incremental_validation_inventory": [],
                "needs_incremental_validation_docs": [],
            },
            "selection_contract": context["predecessor"]["selection_contract"],
            "inputs": {
                "bootstrap_manifest": {
                    "path": str(bootstrap_manifest_path),
                    "sha256": sha256_file(bootstrap_manifest_path),
                    "payload_sha256": context["bootstrap_hash"],
                    "provenance": context["bootstrap"].get("provenance", {}),
                },
                "sampling_plan": {
                    "path": str(
                        Path(migration["successor"]["sampling_plan"]["path"])
                    ),
                    "file_sha256": migration["successor"]["sampling_plan"][
                        "file_sha256"
                    ],
                    "payload_sha256": full_plan["plan_sha256"],
                },
                "tokenizer_model": successor_tokenizer,
            },
            "validation_prefix_inputs": {
                "path": "validation_prefix_inputs.json",
                "sha256": sha256_file(prefix_path),
                "size_bytes": prefix_path.stat().st_size,
            },
            "languages": dict(sorted(languages.items())),
            "k12_migration": {
                "tool_id": TOOL_ID,
                "migration_plan_sha256": migration["migration_plan_sha256"],
                "language_sources": sources,
                "predecessor_inode_sharing": False,
                "metadata_only_refresh_languages": migration["transition"][
                    "metadata_only_refresh_languages"
                ],
            },
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        manifest_path = staging / "corpus_manifest.json"
        _write_frozen_json(manifest_path, manifest)
        files = {
            **dict(sorted(artifact_hashes.items())),
            "corpus_manifest.json": sha256_file(manifest_path),
            "validation_prefix_inputs.json": sha256_file(prefix_path),
        }
        hash_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "sha256",
            "corpus_manifest_payload_sha256": manifest["manifest_sha256"],
            "files": dict(sorted(files.items())),
        }
        hash_manifest["hash_manifest_payload_sha256"] = canonical_json_sha256(
            hash_manifest
        )
        _write_frozen_json(staging / "hash_manifest.json", hash_manifest)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = _receipt(
        "release-merge",
        {
            "migration_plan_sha256": migration["migration_plan_sha256"],
            "corpus_manifest": {
                "sha256": sha256_file(output_dir / "corpus_manifest.json"),
                "payload_sha256": manifest["manifest_sha256"],
            },
            "hash_manifest": {
                "sha256": sha256_file(output_dir / "hash_manifest.json"),
                "payload_sha256": hash_manifest[
                    "hash_manifest_payload_sha256"
                ],
            },
            "language_sources": sources,
            "predecessor_inode_sharing": False,
            "execution_metadata": {
                "mode": execution_mode,
                "output_dir": str(output_dir),
                "work_root": str(work_root),
                "worker_receipts": worker_receipts,
                "worker_install_methods": install_methods,
                "publication_methods": publication_methods,
            },
        },
    )
    _write_frozen_json(receipt_path, receipt)
    return receipt


def merge_releases(
    *,
    migration_plan_path: Path,
    predecessor_release_dir: Path,
    delta_release_dir: Path,
    bootstrap_manifest_path: Path,
    output_dir: Path,
    receipt_path: Path,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Run all language workers locally, then use the shared finalizer."""

    migration, full_plan, _ = _load_migration(migration_plan_path)
    predecessor_root, successor_root = _migration_roots(migration)
    resolved_predecessor = predecessor_release_dir.resolve()
    resolved_delta = delta_release_dir.resolve()
    resolved_bootstrap = bootstrap_manifest_path.resolve()
    resolved_output = output_dir.resolve()
    resolved_receipt = receipt_path.resolve()
    resolved_work_root = (
        work_root.resolve()
        if work_root is not None
        else resolved_output.with_name(f".{resolved_output.name}.merge-work")
    )
    _require_under(
        resolved_predecessor, predecessor_root, label="predecessor release"
    )
    for candidate, label in (
        (resolved_delta, "delta release"),
        (resolved_bootstrap, "merged bootstrap manifest"),
        (resolved_work_root, "release merge work root"),
        (resolved_output, "merged release"),
        (resolved_receipt, "release receipt"),
    ):
        _require_under(candidate, successor_root, label=label)
    _reject_output_inside_inputs(
        resolved_output, [resolved_predecessor, resolved_delta, resolved_work_root]
    )
    _reject_output_inside_inputs(
        resolved_receipt,
        [resolved_predecessor, resolved_delta, resolved_output, resolved_work_root],
    )
    if resolved_output.exists() or resolved_output.is_symlink():
        raise K12MigrationError(
            f"release migration output already exists: {resolved_output}"
        )
    for language_index, _ in enumerate(_tier_languages(full_plan, "K200")):
        release_merge_worker(
            migration_plan_path=migration_plan_path,
            predecessor_release_dir=resolved_predecessor,
            delta_release_dir=resolved_delta,
            bootstrap_manifest_path=resolved_bootstrap,
            work_root=resolved_work_root,
            language_index=language_index,
        )
    return finalize_release_merge(
        migration_plan_path=migration_plan_path,
        predecessor_release_dir=resolved_predecessor,
        delta_release_dir=resolved_delta,
        bootstrap_manifest_path=resolved_bootstrap,
        work_root=resolved_work_root,
        output_dir=resolved_output,
        receipt_path=resolved_receipt,
        execution_mode="serial",
    )
