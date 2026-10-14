"""Own sealed multilingual corpus evidence and data-stage checks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import (
    DATA_ARTIFACT_SEAL_NAME,
    DEFAULT_SPEC,
    MANIFEST_NAME,
    Paths,
    REPO_ROOT,
    WorkflowError,
    _abs,
    _atomic_json,
    _file_sha256,
    _language_map,
    _load_object,
    _payload_sha256,
    _run,
)


def _guard_sealed_corpus_retry(job: Mapping[str, Any], paths: Paths) -> None:
    """Refuse destructive recovery of a data producer after corpus sealing."""

    if (
        os.getenv("PROOF_AB_V2_RETRY_FAILED") == "1"
        and job.get("stage") == "data"
        and job.get("action") != "megatron_data_smoke"
        and (paths.control / DATA_ARTIFACT_SEAL_NAME).exists()
    ):
        raise WorkflowError(
            "non-destructive recovery required: the corpus is already sealed, so a "
            f"missing/invalid receipt for pre-seal data job {job['job_id']} cannot be "
            "recreated by moving, deleting, unsealing, or overwriting corpus artifacts"
        )


def _languages_from_map(path: Path) -> list[str]:
    payload = _load_object(path)
    group = payload.get("0")
    if not isinstance(group, dict) or not isinstance(group.get("languages"), list):
        raise WorkflowError(f"invalid A/B-v2 language map: {path}")
    return [str(value) for value in group["languages"]]


def _stat_identity(path: Path) -> dict[str, int]:
    stat_result = path.lstat()
    return {
        "size": int(stat_result.st_size),
        "inode": int(stat_result.st_ino),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
        "mode": int(stat_result.st_mode & 0o7777),
    }
def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _walk_prefix_descriptors(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, dict):
        if {
            "prefix",
            "bin_sha256",
            "idx_sha256",
            "bin_size_bytes",
            "idx_size_bytes",
        }.issubset(value):
            yield value
        for child in value.values():
            yield from _walk_prefix_descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_prefix_descriptors(child)


def _write_data_artifact_seal(
    manifest: Mapping[str, Any], paths: Paths
) -> dict[str, Any]:
    """Make audited H100 data inputs read-only and freeze cheap stat identities."""

    audit_path = paths.control / "data_audit.json"
    audit = _load_object(audit_path)
    audit_without_hash = dict(audit)
    audit_payload_hash = audit_without_hash.pop("report_payload_sha256", None)
    if (
        audit.get("pass") is not True
        or audit.get("status") != "passed"
        or audit_payload_hash != _payload_sha256(audit_without_hash)
    ):
        raise WorkflowError("cannot seal data without a hash-valid passing audit")

    candidates: dict[Path, dict[str, Any]] = {}

    def add_small(path: Path, *, rehash_on_verify: bool = True) -> None:
        resolved = path.resolve()
        if not resolved.is_file() or path.is_symlink() or not _under_root(resolved, paths.root):
            raise WorkflowError(f"invalid data-seal file: {path}")
        candidates[resolved] = {
            "sha256": _file_sha256(resolved),
            "rehash_on_verify": bool(rehash_on_verify),
        }

    # These compact files define the exact mixture, tokenizer, and validation
    # statistics consumed by H100 jobs.
    add_small(paths.tokenizer / "tokenizer.model")
    add_small(paths.sampling_plan)
    add_small(paths.release / "corpus_manifest.json")
    for file_path in sorted(paths.release.glob("unigram_counts/*.json")):
        add_small(file_path, rehash_on_verify=False)
    tokenized_json = sorted(paths.tokenized.rglob("*.json"))
    if not tokenized_json:
        raise WorkflowError("data seal found no tokenized manifests")
    for file_path in tokenized_json:
        add_small(file_path)

    # The release audit immediately preceding this seal cryptographically
    # verified every declared bin/idx hash. Rehash each unique large artifact
    # once here to close the audit-to-seal window; consumers then compare the
    # immutable stat identities instead of rehashing the complete corpus for
    # every H100 job.
    for manifest_path in tokenized_json:
        payload = _load_object(manifest_path)
        for raw in _walk_prefix_descriptors(payload):
            prefix = Path(str(raw["prefix"])).resolve()
            for suffix in ("bin", "idx"):
                artifact = Path(f"{prefix}.{suffix}")
                if (
                    not artifact.is_file()
                    or artifact.is_symlink()
                    or not _under_root(artifact, paths.root)
                ):
                    raise WorkflowError(f"invalid data-seal prefix artifact: {artifact}")
                expected_size = int(raw[f"{suffix}_size_bytes"])
                if artifact.stat().st_size != expected_size:
                    raise WorkflowError(f"data-seal prefix size drift: {artifact}")
                expected_hash = str(raw[f"{suffix}_sha256"])
                if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                    raise WorkflowError(f"data-seal prefix lacks SHA-256: {artifact}")
                existing = candidates.get(artifact)
                descriptor = {
                    "sha256": expected_hash,
                    "rehash_on_verify": False,
                }
                if existing is not None and existing != descriptor:
                    raise WorkflowError(f"conflicting data-seal identity: {artifact}")
                if existing is None and _file_sha256(artifact) != expected_hash:
                    raise WorkflowError(
                        f"data-seal prefix changed after the release audit: {artifact}"
                    )
                candidates[artifact] = descriptor

    if not any(path.suffix == ".bin" for path in candidates):
        raise WorkflowError("data seal found no Megatron bin/idx artifacts")

    # Seal files first, then all containing data directories. Directory ctime
    # and inode checks detect later unlink/rename/replacement as well as chmod.
    file_entries: dict[str, Any] = {}
    directories: set[Path] = set()
    corpus_root = (paths.root / "corpus").resolve()
    for artifact, descriptor in sorted(candidates.items(), key=lambda row: str(row[0])):
        artifact.chmod(artifact.stat().st_mode & ~0o222)
        relative = artifact.relative_to(paths.root.resolve()).as_posix()
        file_entries[relative] = {**descriptor, "stat": _stat_identity(artifact)}
        parent = artifact.parent
        while _under_root(parent, corpus_root):
            directories.add(parent)
            if parent == corpus_root:
                break
            parent = parent.parent

    directory_entries: dict[str, Any] = {}
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.chmod(directory.stat().st_mode & ~0o222)
        relative = directory.relative_to(paths.root.resolve()).as_posix()
        directory_entries[relative] = {"stat": _stat_identity(directory)}

    seal: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed",
        "workflow_manifest_sha256": str(manifest["manifest_sha256"]),
        "release_root": str(paths.root.resolve()),
        "audit": {
            "path": str(audit_path.resolve()),
            "file_sha256": _file_sha256(audit_path),
            "payload_sha256": audit_payload_hash,
        },
        "contract": {
            "large_artifact_consumer_check": "stat_identity_no_rehash",
            "stat_fields": [
                "size",
                "inode",
                "mtime_ns",
                "ctime_ns",
                "mode",
            ],
            "write_bits_removed": True,
            "modification_chmod_and_replacement_detected": True,
        },
        "files": file_entries,
        "directories": directory_entries,
    }
    seal["seal_payload_sha256"] = _payload_sha256(seal)
    _atomic_json(paths.control / DATA_ARTIFACT_SEAL_NAME, seal)
    return seal


def _verify_data_artifact_seal(
    manifest: Mapping[str, Any], paths: Paths
) -> dict[str, Any]:
    seal_path = paths.control / DATA_ARTIFACT_SEAL_NAME
    seal = _load_object(seal_path)
    expected = seal.get("seal_payload_sha256")
    candidate = dict(seal)
    candidate.pop("seal_payload_sha256", None)
    errors: list[str] = []
    if expected != _payload_sha256(candidate):
        errors.append("seal payload hash mismatch")
    if seal.get("workflow_manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append("seal belongs to another workflow manifest")
    if seal.get("release_root") != str(paths.root.resolve()):
        errors.append("seal belongs to another release root")
    if seal.get("status") != "sealed":
        errors.append("seal status is not sealed")
    audit = seal.get("audit")
    audit_path = paths.control / "data_audit.json"
    if (
        not isinstance(audit, dict)
        or not audit_path.is_file()
        or _file_sha256(audit_path) != audit.get("file_sha256")
    ):
        errors.append("data-audit file identity mismatch")
    else:
        audit_payload = _load_object(audit_path)
        audit_candidate = dict(audit_payload)
        embedded_audit_hash = audit_candidate.pop("report_payload_sha256", None)
        if (
            audit_payload.get("pass") is not True
            or audit_payload.get("status") != "passed"
            or embedded_audit_hash != audit.get("payload_sha256")
            or embedded_audit_hash != _payload_sha256(audit_candidate)
        ):
            errors.append("data-audit status/payload mismatch")

    for kind, entries in (("file", seal.get("files")), ("directory", seal.get("directories"))):
        if not isinstance(entries, dict):
            errors.append(f"seal lacks {kind} entries")
            continue
        for relative, raw in entries.items():
            relative_path = Path(str(relative))
            path = paths.root / relative_path
            if relative_path.is_absolute() or not _under_root(path, paths.root):
                errors.append(f"sealed {kind} escapes the release root: {relative}")
                continue
            if not isinstance(raw, dict) or path.is_symlink():
                errors.append(f"invalid sealed {kind}: {relative}")
                continue
            exists = path.is_file() if kind == "file" else path.is_dir()
            if not exists or _stat_identity(path) != raw.get("stat"):
                errors.append(f"sealed {kind} stat drift: {relative}")
                continue
            if int(raw["stat"]["mode"]) & 0o222:
                errors.append(f"sealed {kind} became writable: {relative}")
            if kind == "file" and raw.get("rehash_on_verify") is True:
                if _file_sha256(path) != raw.get("sha256"):
                    errors.append(f"sealed compact file hash drift: {relative}")
    if errors:
        raise WorkflowError("data artifact seal verification failed: " + "; ".join(errors[:8]))
    return {
        "path": str(seal_path.resolve()),
        "sha256": _file_sha256(seal_path),
        "seal_payload_sha256": expected,
        "file_count": len(seal["files"]),
    }

def _sealed_corpus_reuse_evidence(
    manifest: Mapping[str, Any], paths: Paths
) -> dict[str, Any]:
    """Verify the source seal that a production relaunch is allowed to read.

    The source corpus remains owned by its completed K12 workflow.  This check
    binds the new workflow to that old workflow's manifest, audit, seal, and
    immutable-stat identities; it deliberately never creates a second seal or
    changes permissions below the source root.
    """

    materialization = manifest.get("data_materialization")
    if not isinstance(materialization, Mapping) or materialization.get("mode") != "reuse_sealed_k12":
        raise WorkflowError("sealed-corpus verification requested for another data mode")
    sealed = materialization.get("sealed_corpus")
    if not isinstance(sealed, Mapping):
        raise WorkflowError("sealed-corpus reuse metadata is missing")
    source_root = paths.corpus_root.resolve()
    if source_root == paths.root.resolve():
        raise WorkflowError("sealed-corpus source unexpectedly aliases workflow outputs")
    if str(source_root) != sealed.get("root"):
        raise WorkflowError("sealed-corpus source path differs from the frozen manifest")
    source_manifest_path = source_root / "control" / MANIFEST_NAME
    if not source_manifest_path.is_file() or _file_sha256(source_manifest_path) != sealed.get(
        "launch_manifest_sha256"
    ):
        raise WorkflowError("sealed-corpus source launch manifest identity changed")
    source_manifest = _load_object(source_manifest_path)
    if source_manifest.get("manifest_sha256") != sealed.get("workflow_manifest_sha256"):
        raise WorkflowError("sealed-corpus source workflow manifest payload changed")
    if str(source_manifest.get("output_root", "")) != str(source_root):
        raise WorkflowError("sealed-corpus source manifest points at another root")
    source_paths = Paths(source_root)
    seal = _verify_data_artifact_seal(source_manifest, source_paths)
    if seal["sha256"] != sealed.get("data_seal_sha256"):
        raise WorkflowError("sealed-corpus data seal file identity changed")
    if seal["seal_payload_sha256"] != sealed.get("data_seal_payload_sha256"):
        raise WorkflowError("sealed-corpus data seal payload changed")
    return {
        "source_root": str(source_root),
        "source_launch_manifest": str(source_manifest_path),
        "source_launch_manifest_sha256": sealed["launch_manifest_sha256"],
        "source_workflow_manifest_sha256": source_manifest["manifest_sha256"],
        "source_data_seal": seal,
    }


def _verify_workflow_data(manifest: Mapping[str, Any], paths: Paths) -> dict[str, Any]:
    materialization = manifest.get("data_materialization")
    if isinstance(materialization, Mapping) and materialization.get("mode") == "reuse_sealed_k12":
        return _sealed_corpus_reuse_evidence(manifest, paths)
    return _verify_data_artifact_seal(manifest, paths)


def _run_reuse_sealed_data_verify(
    manifest: Mapping[str, Any], paths: Paths
) -> None:
    evidence = {
        "schema_version": 1,
        "status": "complete",
        "workflow_manifest_sha256": manifest["manifest_sha256"],
        "reused_corpus": _sealed_corpus_reuse_evidence(manifest, paths),
    }
    evidence["evidence_sha256"] = _payload_sha256(evidence)
    _atomic_json(paths.control / "data_reuse_evidence.json", evidence)


def _run_data_audit(
    manifest: Mapping[str, Any],
    paths: Paths,
    *,
    study_spec_path: Path = DEFAULT_SPEC,
) -> None:
    from multilingual_megatron.pipeline.stages.audit.fineweb_v2_release import (
        write_release_audit,
    )

    report = write_release_audit(
        repo_root=REPO_ROOT,
        study_spec_path=_abs(study_spec_path),
        release_root=paths.root,
        output_path=paths.control / "data_audit.json",
    )
    if report.get("pass") is not True:
        raise WorkflowError(
            "data audit failed: "
            + "; ".join(str(value) for value in report.get("errors", []))
        )
    _write_data_artifact_seal(manifest, paths)


def _run_megatron_data_smoke(
    manifest: Mapping[str, Any], paths: Paths, runtime: Mapping[str, Any]
) -> None:
    """Open every K tier and validation prefix with Megatron's real reader."""

    _verify_workflow_data(manifest, paths)
    megatron_lm = (
        REPO_ROOT / "external/Megatron-Bridge/3rdparty/Megatron-LM"
    ).resolve()
    indexed_reader = megatron_lm / "megatron/core/datasets/indexed_dataset.py"
    if not indexed_reader.is_file():
        raise WorkflowError(
            f"Megatron indexed-dataset reader is missing: {indexed_reader}"
        )
    environment = dict(os.environ)
    megatron_lm_path = str(megatron_lm)
    existing_pythonpath = [
        value
        for value in environment.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            megatron_lm_path,
            *[value for value in existing_pythonpath if value != megatron_lm_path],
        ]
    )
    output = paths.control / "megatron_data_smoke.json"
    _run(
        [
            str(runtime["bridge_python_executable"]),
            "-m",
            "multilingual_megatron.experiments.scaling.k12_migration.data_smoke",
            "--tokenized-root",
            str(paths.tokenized),
            "--output",
            str(output),
        ],
        env=environment,
    )
    report = _load_object(output)
    expected = report.get("report_payload_sha256")
    candidate = dict(report)
    candidate.pop("report_payload_sha256", None)
    if (
        report.get("pass") is not True
        or report.get("status") != "passed"
        or expected != _payload_sha256(candidate)
    ):
        raise WorkflowError("Megatron indexed-dataset reader smoke did not pass")
    # Reuse mode deliberately keeps the seal in the immutable source corpus,
    # not in this workflow-output root.  Recheck the complete source contract
    # after opening the actual indexed prefixes.
    _verify_workflow_data(manifest, paths)


def _verify_k12_migration_completion(
    paths: Paths, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the complete, receipt-backed CPU migration before A/B-v2 starts."""

    from multilingual_megatron.experiments.scaling.k12_migration import workflow as k12_migration_workflow

    control = paths.root / "control" / "k12_migration"
    launch_path = control / k12_migration_workflow.MANIFEST_NAME
    completion_path = control / "migration_complete.json"
    # The completed migration has receipt-backed output hashes.  Its historic
    # fixed-input snapshot includes this downstream orchestration module only
    # because the old preflight imported its seal helper; revalidating that
    # incidental source file would make a later A/B-only repair invalidate
    # already materialized corpus evidence.  This new workflow still pins its
    # own complete source closure and performs a fresh canonical data audit.
    migration_status = k12_migration_workflow.status(
        launch_path, verify_fixed_inputs=False
    )
    if migration_status.get("complete") is not True:
        raise WorkflowError("K12 CPU migration lacks complete receipt evidence")

    migration_manifest = _load_object(launch_path)
    completion = _load_object(completion_path)
    completion_hash = completion.get("payload_sha256")
    candidate = dict(completion)
    candidate.pop("payload_sha256", None)
    predecessor = _abs(str(runtime["data"]["adoption_predecessor_root"]))
    if (
        completion_hash != _payload_sha256(candidate)
        or completion.get("status") != "complete"
        or completion.get("workflow_id") != k12_migration_workflow.WORKFLOW_ID
        or completion.get("manifest_sha256")
        != migration_manifest.get("manifest_sha256")
        or migration_status.get("manifest_sha256")
        != migration_manifest.get("manifest_sha256")
        or Path(str(migration_manifest.get("output_root", ""))).resolve()
        != paths.root.resolve()
        or Path(str(migration_manifest.get("predecessor_root", ""))).resolve()
        != predecessor.resolve()
    ):
        raise WorkflowError("K12 migration completion/root/hash binding is invalid")
    expected_counts = {
        "changed_physical": 15,
        "increased_physical": 11,
        "decreased_physical": 4,
        "acquisition": 9,
        "delta_finalize": 15,
        "reused_physical": 185,
        "exposure_changed": 16,
    }
    if completion.get("transition_counts") != expected_counts:
        raise WorkflowError("K12 migration completion has unexpected transition counts")
    return {
        "launch_manifest": str(launch_path),
        "launch_manifest_sha256": migration_manifest["manifest_sha256"],
        "completion": str(completion_path),
        "completion_payload_sha256": completion_hash,
        "all_receipts_complete": True,
    }
