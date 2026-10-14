"""Own A/B-v2 receipts, corpus seals, and storage checks."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import (
    DATA_ARTIFACT_SEAL_NAME,
    DEFAULT_SPEC,
    MANIFEST_NAME,
    Paths,
    REPO_ROOT,
    WorkflowError,
    _abs,
    _array_indices,
    _atomic_json,
    _file_sha256,
    _git_provenance,
    _job,
    _language_map,
    _load_object,
    _payload_sha256,
    _recovery_receipt_identity,
    _rel_or_abs,
    _run,
)


def _record_source_drift_advisory(
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    job_id: str,
) -> Path | None:
    """Persist code drift as provenance without making it an execution gate."""

    raw_rows = validation.get("source_drift_advisories", [])
    if not isinstance(raw_rows, list) or not raw_rows:
        return None
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    if not rows:
        return None
    try:
        execution_git_provenance: Mapping[str, Any] = _git_provenance()
    except (OSError, WorkflowError) as exc:
        execution_git_provenance = {
            "status": "unavailable_advisory_only",
            "error": str(exc),
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "source_drift_advisory_v1",
        "policy": "record_and_continue_never_blocks_execution",
        "manifest_sha256": manifest.get("manifest_sha256"),
        "launch_git_provenance": manifest.get("git_provenance"),
        "execution_git_provenance": execution_git_provenance,
        "source_drift": rows,
    }
    payload["receipt_sha256"] = _payload_sha256(payload)
    path = (
        Path(str(manifest["output_root"]))
        / "control"
        / "source_drift"
        / f"{payload['receipt_sha256']}.json"
    )
    try:
        _atomic_json(path, payload)
        receipt = str(path)
    except OSError as exc:
        # Provenance documentation must never strand production compute. The
        # controller log still records every hash pair and the write failure.
        receipt = f"write_failed:{exc}"
        path = None
    print(
        "[SOURCE_DRIFT_ADVISORY] "
        f"job={job_id} policy=record_and_continue count={len(rows)} "
        f"receipt={receipt} rows={rows}"
    )
    return path


def _receipt_path(
    manifest: Mapping[str, Any], job_id: str, array_task: str | None = None
) -> Path:
    suffix = f".{array_task}" if array_task is not None else ""
    return (
        Path(str(manifest["output_root"]))
        / "control"
        / "receipts"
        / f"{job_id}{suffix}.json"
    )


def _has_completed_job_evidence(
    manifest: Mapping[str, Any], state: Mapping[str, Any], job: Mapping[str, Any]
) -> bool:
    job_id = str(job["job_id"])
    if job.get("array"):
        paths = [
            (str(task_id), _receipt_path(manifest, job_id, str(task_id)))
            for task_id in _array_indices(str(job["array"]))
        ]
        if not all(path.is_file() for _task, path in paths):
            return False
        for task, _path in paths:
            _verified_receipt_evidence(manifest, job_id, array_task=task)
        return True
    path = _receipt_path(manifest, job_id)
    if not path.is_file():
        return False
    _verified_receipt_evidence(manifest, job_id)
    return True


def _quarantine_job_receipts(
    manifest: Mapping[str, Any], job: Mapping[str, Any]
) -> list[str]:
    job_id = str(job["job_id"])
    candidates = (
        [
            _receipt_path(manifest, job_id, str(task_id))
            for task_id in _array_indices(str(job["array"]))
        ]
        if job.get("array")
        else [_receipt_path(manifest, job_id)]
    )
    quarantine = Path(str(manifest["output_root"])) / "control" / "invalid_receipts"
    moved: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        task = path.stem.split(".", 1)[1] if job.get("array") and "." in path.stem else None
        try:
            _verified_receipt_evidence(manifest, job_id, array_task=task)
        except (WorkflowError, OSError, ValueError):
            pass
        else:
            # A retried Slurm array resubmits every index. Preserve valid task
            # receipts; execute-job will no-op those exact completed tasks.
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        digest = _file_sha256(path)[:16]
        target = quarantine / f"{path.name}.{digest}.{time.time_ns()}.invalid"
        os.replace(path, target)
        moved.append(str(target))
    return moved


def _retry_owned_output_candidates(
    paths: Paths, job: Mapping[str, Any], task: str | None
) -> list[Path]:
    action = str(job.get("action", ""))
    candidates: list[Path] = []
    if action == "tokenizer_sentences":
        candidates.append(paths.sentence_root)
    elif action == "tokenizer_train":
        candidates.append(paths.tokenizer)
    elif action == "finalize_corpus":
        candidates.append(paths.release)
        candidates.extend(paths.release.parent.glob(f".{paths.release.name}.partial-*"))
    elif action == "presplit_tokenize_plan":
        candidates.append(paths.tokenization_work / "plan")
    elif action == "presplit_tokenize_worker":
        if task is None or not task.isdigit():
            raise WorkflowError("retried tokenization array task lacks its numeric task ID")
        candidates.append(paths.tokenization_work / "ranks" / f"rank_{int(task):05d}")
    elif action in {"bootstrap_worker", "tokenizer_bootstrap_worker", "production_worker"}:
        if task is None or not task.isdigit():
            raise WorkflowError("retried bootstrap array task lacks its numeric task ID")
        languages = _languages_from_map(_abs(_language_map(200)))
        index = int(task)
        if not 0 <= index < len(languages):
            raise WorkflowError(f"bootstrap array task {index} is outside the K200 map")
        root = {
            "bootstrap_worker": paths.bootstrap,
            "tokenizer_bootstrap_worker": paths.tokenizer_bootstrap,
            "production_worker": paths.production_bootstrap,
        }[action]
        candidates.append(root / languages[index])
    elif action == "presplit_tokenize_merge":
        candidates.append(paths.tokenized)
        candidates.extend(paths.tokenized.parent.glob(f".{paths.tokenized.name}.partial-*"))
    return [candidate for candidate in candidates if candidate.exists()]


def _quarantine_retry_owned_outputs(
    manifest: Mapping[str, Any], job: Mapping[str, Any]
) -> list[str]:
    task = os.getenv("SLURM_ARRAY_TASK_ID") if job.get("array") else None
    paths = Paths(Path(str(manifest["output_root"])))
    candidates = _retry_owned_output_candidates(paths, job, task)
    if not candidates:
        return []
    quarantine = (
        paths.control
        / "invalid_outputs"
        / str(job["job_id"])
        / (f"task_{task}" if task is not None else "scalar")
        / str(time.time_ns())
    )
    quarantine.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for index, source in enumerate(candidates):
        target = quarantine / f"{index:02d}_{source.name}"
        os.replace(source, target)
        moved.append(str(target.resolve()))
    _atomic_json(
        quarantine / "quarantine.json",
        {
            "schema_version": 1,
            "workflow_manifest_sha256": manifest["manifest_sha256"],
            "job_id": job["job_id"],
            "array_task_id": task,
            "reason": "retry_failed_job_owned_output",
            "moved_outputs": moved,
        },
    )
    return moved


def _retry_history_entry(
    *,
    prior: Mapping[str, Any],
    job: Mapping[str, Any],
    quarantined_receipts: Sequence[str],
    reason: str,
    output_root: Path,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    return {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": prior.get("slurm_job_id"),
        "state": prior.get("state"),
        "exit_code": prior.get("exit_code"),
        "slurm_accounting": prior.get("slurm_accounting", {}),
        "array": prior.get("array", job.get("array")),
        "retry_reason": reason,
        "receipt_quarantine": list(quarantined_receipts),
        "controller_log": str(output_root / "logs" / f"{job_id}.controller.log"),
        "slurm_log_pattern": str(
            output_root
            / "logs"
            / (f"{job_id}.%A_%a.log" if job.get("array") else f"{job_id}.%j.log")
        ),
    }


def _write_receipt(
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    status: str,
    output_hashes: Mapping[str, str],
    details: Mapping[str, Any] | None = None,
) -> None:
    task = os.getenv("SLURM_ARRAY_TASK_ID") if job.get("array") else None
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "job_id": job["job_id"],
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "array_task_id": task,
        "status": status,
        "output_sha256": dict(output_hashes),
        "details": dict(details or {}),
    }
    recovery = _recovery_receipt_identity(manifest)
    if recovery is not None:
        payload["recovery_authorization"] = recovery
    payload["receipt_sha256"] = _payload_sha256(payload)
    _atomic_json(_receipt_path(manifest, str(job["job_id"]), task), payload)


def _declared_output_hashes(
    manifest: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, str]:
    launch_repo_root = Path(str(manifest.get("repo_root", REPO_ROOT))).resolve()
    output_root = Path(str(manifest["output_root"])).resolve()
    hashes: dict[str, str] = {}
    for raw in job.get("outputs", []):
        value = Path(str(raw))
        path = (
            value.resolve()
            if value.is_absolute()
            else (launch_repo_root / value).resolve()
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or not _under_root(path, output_root)
        ):
            raise WorkflowError(
                f"job {job['job_id']} did not produce declared output: {path}"
            )
        try:
            output_key = str(path.relative_to(launch_repo_root))
        except ValueError:
            output_key = str(path)
        hashes[output_key] = _file_sha256(path)
    return hashes


def _array_task_output_hashes(
    job: Mapping[str, Any], paths: Paths, task: str | None
) -> dict[str, str]:
    if not job.get("array"):
        return {}
    output = _array_task_output_path(job, paths, task)
    if not output.is_file():
        raise WorkflowError(f"array task did not produce its completion artifact: {output}")
    return {_rel_or_abs(output): _file_sha256(output)}


def _array_task_output_path(
    job: Mapping[str, Any], paths: Paths, task: str | None
) -> Path:
    """Return the one task-owned artifact that binds an array receipt."""

    if task is None or not task.isdigit():
        raise WorkflowError(f"array job {job['job_id']} lacks a numeric task ID")
    index = int(task)
    if index not in _array_indices(str(job["array"])):
        raise WorkflowError(
            f"array task {index} is outside the frozen array for {job['job_id']}"
        )
    action = str(job.get("action", ""))
    output: Path | None = None
    if action == "presplit_tokenize_worker":
        output = paths.tokenization_work / "ranks" / f"rank_{index:05d}" / "result.json"
    elif action in {"bootstrap_worker", "tokenizer_bootstrap_worker", "production_worker"}:
        languages = _languages_from_map(_abs(_language_map(200)))
        if not 0 <= index < len(languages):
            raise WorkflowError(f"array task {index} is outside the K200 language map")
        root = {
            "bootstrap_worker": paths.bootstrap,
            "tokenizer_bootstrap_worker": paths.tokenizer_bootstrap,
            "production_worker": paths.production_bootstrap,
        }[action]
        output = root / languages[index] / "bootstrap_report.json"
    if output is None:
        raise WorkflowError(
            f"array action {action!r} has no task-owned receipt artifact contract"
        )
    return output


from .artifact_data import (
    _guard_sealed_corpus_retry,
    _languages_from_map,
    _stat_identity,
    _under_root,
    _walk_prefix_descriptors,
    _write_data_artifact_seal,
    _verify_data_artifact_seal,
    _sealed_corpus_reuse_evidence,
    _verify_workflow_data,
    _run_reuse_sealed_data_verify,
    _run_data_audit,
    _run_megatron_data_smoke,
    _verify_k12_migration_completion,
)
def _run_storage_preflight(
    manifest: Mapping[str, Any], paths: Paths, runtime: Mapping[str, Any]
) -> None:
    estimate = manifest.get("storage_estimate")
    if not isinstance(estimate, dict):
        raise WorkflowError("manifest lacks storage estimate")
    paths.root.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(paths.root)
    available = int(stats.f_bavail * stats.f_frsize)
    total = int(stats.f_blocks * stats.f_frsize)
    required = int(estimate["required_available_bytes_at_preflight"])
    minimum_data_reserve = int(runtime["storage"]["non_checkpoint_peak_reserve_bytes"])
    errors: list[str] = []
    warnings: list[str] = []
    soft_limit = runtime["storage"].get("enforcement") == "informational_soft_limit"
    fresh_data_paths = (
        paths.bootstrap,
        paths.sentence_root,
        paths.tokenizer_acquisition_plan,
        paths.tokenizer_bootstrap,
        paths.tokenizer,
        paths.calibration,
        paths.sampling_plan,
        paths.production_bootstrap,
        paths.release,
        paths.tokenization_work,
        paths.tokenized,
    )
    data = runtime.get("data")
    if not isinstance(data, dict):
        raise WorkflowError("runtime data configuration is missing")
    data_mode = str(data.get("mode", "fresh"))
    stale_or_partial: list[str] = []
    missing_adopted: list[str] = []
    invalid_adopted: list[str] = []
    adoption_manifest = paths.corpus_root / "corpus" / "k12_migration" / "adoption_manifest.json"
    migration_manifest = paths.corpus_root / "control" / "k12_migration" / "launch_manifest.json"
    migration_complete = paths.corpus_root / "control" / "k12_migration" / "migration_complete.json"
    migration_evidence: dict[str, Any] | None = None
    if data_mode == "fresh":
        stale_or_partial = [str(path) for path in fresh_data_paths if path.exists()]
        if stale_or_partial:
            errors.append(
                "fresh v2 resampling requires absent sampled/tokenized outputs; found: "
                + ", ".join(stale_or_partial[:8])
            )
    elif data_mode == "adopt_k12_migration":
        required_adopted = {
            paths.bootstrap: "directory",
            paths.sentence_root: "directory",
            paths.tokenizer_acquisition_plan: "file",
            paths.tokenizer_bootstrap: "directory",
            paths.tokenizer: "directory",
            paths.calibration: "file",
            paths.sampling_plan: "file",
            paths.production_bootstrap: "directory",
            paths.release: "directory",
            paths.tokenization_work: "directory",
            paths.tokenized: "directory",
            adoption_manifest: "file",
            migration_manifest: "file",
            migration_complete: "file",
        }
        missing_adopted = [
            str(path) for path in required_adopted if not path.exists()
        ]
        invalid_adopted = [
            str(path)
            for path, kind in required_adopted.items()
            if path.exists()
            and (
                (kind == "file" and not path.is_file())
                or (kind == "directory" and not path.is_dir())
            )
        ]
        if missing_adopted:
            errors.append(
                "K12 adoption requires a fully materialized successor corpus; missing: "
                + ", ".join(missing_adopted[:8])
            )
        if invalid_adopted:
            errors.append(
                "K12 adoption contains invalid artifact types: "
                + ", ".join(invalid_adopted[:8])
            )
        declared = manifest.get("data_materialization")
        if not isinstance(declared, dict) or declared.get("mode") != data_mode:
            errors.append("manifest/runtime K12 data-materialization mode mismatch")
        predecessor = data.get("adoption_predecessor_root")
        if not isinstance(predecessor, str) or not _abs(predecessor).is_dir():
            errors.append("K12 adoption predecessor root is unavailable")
        if not missing_adopted and not invalid_adopted:
            try:
                migration_evidence = _verify_k12_migration_completion(paths, runtime)
            except Exception as exc:
                errors.append(f"K12 migration completion verification failed: {exc}")
    elif data_mode == "reuse_sealed_k12":
        declared = manifest.get("data_materialization")
        if not isinstance(declared, dict) or declared.get("mode") != data_mode:
            errors.append("manifest/runtime sealed-corpus reuse mode mismatch")
        try:
            migration_evidence = _sealed_corpus_reuse_evidence(manifest, paths)
        except Exception as exc:
            errors.append(f"sealed-corpus source verification failed: {exc}")
    else:
        errors.append(f"unsupported data materialization mode: {data_mode}")
    if minimum_data_reserve < 3 * 1024**4:
        errors.append("retained data reserve is below 3 TiB")
    if available < required:
        message = f"available bytes {available} are below required {required}"
        (warnings if soft_limit else errors).append(message)
    report = {
        "schema_version": 1,
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "storage_enforcement": (
            "informational_soft_limit" if soft_limit else "hard_available_bytes_gate"
        ),
        "filesystem_path": str(paths.root.resolve()),
        "filesystem_total_bytes": total,
        "filesystem_available_bytes": available,
        "required_available_bytes": required,
        "projected_remaining_bytes": available - required,
        "storage_estimate_sha256": _payload_sha256(estimate),
        "storage_estimate": estimate,
        "data_materialization_preflight": {
            "mode": data_mode,
            "required_absent_paths": [str(path) for path in fresh_data_paths],
            "existing_paths": stale_or_partial,
            "adoption_manifest": str(adoption_manifest),
            "migration_manifest": str(migration_manifest),
            "migration_complete": str(migration_complete),
            "migration_completion_evidence": migration_evidence,
            "sealed_corpus_root": str(paths.corpus_root),
            "missing_adopted_paths": missing_adopted,
            "invalid_adopted_paths": invalid_adopted,
            "source_lock_metadata_reuse": "allowed_only_after_pinned_hash_revalidation",
        },
    }
    _atomic_json(paths.control / "storage_preflight.json", report)
    if errors:
        raise WorkflowError("storage preflight failed: " + "; ".join(errors))


def _verified_receipt_evidence(
    manifest: Mapping[str, Any], job_id: str, *, array_task: str | None = None
) -> dict[str, Any]:
    path = _receipt_path(manifest, job_id, array_task)
    payload = _load_object(path)
    expected = payload.get("receipt_sha256")
    candidate = dict(payload)
    candidate.pop("receipt_sha256", None)
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("receipt schema version mismatch")
    if expected != _payload_sha256(candidate):
        errors.append("receipt payload hash mismatch")
    if payload.get("manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append("receipt belongs to another manifest")
    if payload.get("job_id") != job_id:
        errors.append("receipt job ID mismatch")
    if payload.get("array_task_id") != array_task:
        errors.append("receipt array task ID mismatch")
    if payload.get("status") not in {"complete", "not_required"}:
        errors.append(f"receipt status is {payload.get('status')!r}")

    job = _job(manifest, job_id)
    if payload.get("status") == "not_required" and job.get("arguments", {}).get(
        "conditional"
    ) is not True:
        errors.append("not_required receipt belongs to a non-conditional job")
    is_array = bool(job.get("array"))
    if is_array != (array_task is not None):
        errors.append("receipt scalar/array binding mismatch")
    output_root = Path(str(manifest["output_root"])).resolve()
    launch_repo_root = Path(str(manifest.get("repo_root", REPO_ROOT))).resolve()

    def receipt_output_key(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(launch_repo_root))
        except ValueError:
            return str(path.resolve())

    expected_paths: dict[str, Path] = {}
    for raw in job.get("outputs", []):
        expected_path = Path(str(raw))
        expected_path = (
            expected_path.resolve()
            if expected_path.is_absolute()
            else (launch_repo_root / expected_path).resolve()
        )
        expected_paths[receipt_output_key(expected_path)] = expected_path
    if is_array:
        try:
            task_output = _array_task_output_path(
                job, Paths(output_root), array_task
            )
        except WorkflowError as exc:
            errors.append(str(exc))
        else:
            expected_paths[receipt_output_key(task_output)] = task_output
    for expected_path in expected_paths.values():
        if not _under_root(expected_path, output_root):
            errors.append(f"receipt expected output escapes workflow root: {expected_path}")

    output_hashes = payload.get("output_sha256")
    if not isinstance(output_hashes, dict):
        errors.append("receipt lacks output hashes")
    else:
        observed_keys = set(output_hashes)
        expected_keys = set(expected_paths)
        if observed_keys != expected_keys:
            errors.append(
                "receipt output keys differ from the frozen job contract: "
                f"missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )
        for output_key, output in expected_paths.items():
            digest = output_hashes.get(output_key)
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not output.is_file()
                or output.is_symlink()
                or not _under_root(output, output_root)
                or _file_sha256(output) != digest
            ):
                errors.append(f"receipt output identity mismatch: {output}")
    if errors:
        raise WorkflowError(f"invalid dependency evidence for {job_id}: {errors}")
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "receipt_sha256": expected,
        "status": payload["status"],
        "array_task_id": array_task,
    }


def _tree_usage(path: Path) -> tuple[int, int]:
    bytes_total = 0
    files = 0
    if not path.exists():
        return 0, 0
    for root, _directories, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            candidate = Path(root) / name
            if candidate.is_symlink():
                continue
            try:
                bytes_total += candidate.stat().st_size
                files += 1
            except FileNotFoundError:
                continue
    return bytes_total, files
