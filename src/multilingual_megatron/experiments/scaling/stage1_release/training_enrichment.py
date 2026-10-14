"""Fail-closed graph and compute enrichment for the Stage 1 A--D release.

This module keeps four concepts separate:

* the physical trunk/branch execution graph;
* configured hardware and theoretical MFU denominator;
* configured, active, and runtime-observed parameter counts; and
* logical per-readout cost attribution versus unique physical compute.

None of the public helpers mutates its input rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_TRAINING_JOBS = 464
EXPECTED_TRAJECTORIES = 64
EXPECTED_TRUNKS = 64
EXPECTED_BRANCHES = 400
EXPECTED_READOUTS = 400

H100_PEAK_TFLOPS_PER_GPU = 989.5
H100_EFFICIENCY_DOCUMENT = Path(
    "external/research/research-vault/100 Projects/multilingual-megatron-study/"
    "Scaling Archive/02 Systems Experiments/h100_k200_ab_efficiency_status_2026-06-18.md"
)
H100_EFFICIENCY_DOCUMENT_SHA256 = (
    "6717002cdde4c7bb8e9f58efef06d2744d64ae2f5b241a259894f16d5f62546f"
)
RESOURCE_GPU_COUNT = {"h100": 1, "h100_2": 2, "h100_4": 4}
DESIGN_TOTAL_PARAMETER_FORMULA = "2*d_model*d_vocab + (4+9E)*layers*d_model^2"


class Stage1ADReleaseTrainingEnrichmentError(RuntimeError):
    """Raised when an enrichment cannot be produced without ambiguity."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"Cannot read JSON {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Stage1ADReleaseTrainingEnrichmentError(f"Expected JSON object: {path}")
    return payload


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coverage_status(value: float | None) -> str:
    if value is None:
        return "not_computable"
    if value >= 0.99:
        return "complete"
    if value > 0:
        return "partial"
    return "missing"


def _positive_int(value: Any, *, label: str) -> int:
    parsed = _finite(value)
    if parsed is None or not parsed.is_integer() or parsed <= 0:
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"{label} must be a positive integer, got {value!r}"
        )
    return int(parsed)


def _resolved(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _resource_gpu_count(resource: Any) -> int:
    if not isinstance(resource, str) or resource not in RESOURCE_GPU_COUNT:
        raise Stage1ADReleaseTrainingEnrichmentError(
            "Configured production resource must be exactly one of "
            f"{sorted(RESOURCE_GPU_COUNT)}, got {resource!r}"
        )
    return RESOURCE_GPU_COUNT[resource]


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(field, ""))
        if not value:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"{label} has an empty {field}"
            )
        if value in result:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Duplicate {label} {field}: {value}"
            )
        result[value] = row
    return result


def _add_check(
    report: dict[str, Any],
    check_id: str,
    *,
    observed: Any,
    expected: Any,
    passed: bool,
    details: str = "",
) -> None:
    report["checks"].append(
        {
            "check_id": check_id,
            "status": "pass" if passed else "fail",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )
    if not passed:
        report["errors"].append(
            f"{check_id}: observed {observed!r}, expected {expected!r}"
            + (f" ({details})" if details else "")
        )


def _job_result(job: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = job.get("result_path")
    if not raw_path:
        return {}
    result = _load_json(_resolved(raw_path))
    if result.get("status") != "complete":
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"Training result is not complete: {raw_path}"
        )
    return result


def _artifact_reference(
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    row_path_field: str,
    row_hash_field: str,
    result_path_field: str,
    result_hash_field: str,
) -> tuple[Path, str]:
    row_path = job.get(row_path_field)
    result_path = result.get(result_path_field)
    if row_path and result_path and _resolved(row_path) != _resolved(result_path):
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"{job.get('training_job_id')} disagrees on {row_path_field}: "
            f"{row_path!r} versus result {result_path!r}"
        )
    raw_path = row_path or result_path
    if not raw_path:
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"{job.get('training_job_id')} lacks {row_path_field}"
        )
    row_hash = job.get(row_hash_field)
    result_hash = result.get(result_hash_field)
    if row_hash and result_hash and str(row_hash) != str(result_hash):
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"{job.get('training_job_id')} disagrees on {row_hash_field}"
        )
    expected_hash = str(row_hash or result_hash or "")
    if len(expected_hash) != 64:
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"{job.get('training_job_id')} lacks a valid {row_hash_field}"
        )
    return _resolved(raw_path), expected_hash


def _verify_job_artifacts(
    job: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    job_id = str(job.get("training_job_id", ""))
    result = _job_result(job)
    manifest_path, manifest_expected = _artifact_reference(
        job,
        result,
        row_path_field="training_manifest_path",
        row_hash_field="training_manifest_sha256",
        result_path_field="training_manifest",
        result_hash_field="training_manifest_sha256",
    )
    plan_path, plan_expected = _artifact_reference(
        job,
        result,
        row_path_field="segment_plan_path",
        row_hash_field="segment_plan_sha256",
        result_path_field="segment_plan",
        result_hash_field="segment_plan_sha256",
    )
    verifications: list[dict[str, Any]] = []
    for artifact_type, path, expected in (
        ("training_manifest", manifest_path, manifest_expected),
        ("segment_plan", plan_path, plan_expected),
    ):
        if not path.is_file():
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Missing {artifact_type} for {job_id}: {path}"
            )
        recomputed = _sha256(path)
        if recomputed != expected:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"{artifact_type} hash mismatch for {job_id}: "
                f"{recomputed} != {expected}"
            )
        verifications.append(
            {
                "training_job_id": job_id,
                "artifact_type": artifact_type,
                "path": str(path),
                "expected_sha256": expected,
                "recomputed_sha256": recomputed,
                "status": "pass",
            }
        )
    return verifications, _load_json(manifest_path), _load_json(plan_path), result


def validate_production_graph(
    training_jobs: Sequence[Mapping[str, Any]],
    readouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the exact 64-trunk/400-branch production graph.

    The function recomputes every training-manifest and segment-plan hash and
    verifies that every readout binds one complete branch endpoint.  On any
    failure, the raised exception carries the partial structured report as
    ``exc.report``.
    """

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "checking",
        "checks": [],
        "errors": [],
        "artifact_verifications": [],
        "readout_bindings": [],
    }
    try:
        _add_check(
            report,
            "training_job_count",
            observed=len(training_jobs),
            expected=EXPECTED_TRAINING_JOBS,
            passed=len(training_jobs) == EXPECTED_TRAINING_JOBS,
        )
        _add_check(
            report,
            "readout_count",
            observed=len(readouts),
            expected=EXPECTED_READOUTS,
            passed=len(readouts) == EXPECTED_READOUTS,
        )
        jobs = _unique_index(training_jobs, "training_job_id", label="training job")
        readout_index = _unique_index(readouts, "readout_id", label="readout")
        _add_check(
            report,
            "unique_training_job_ids",
            observed=len(jobs),
            expected=EXPECTED_TRAINING_JOBS,
            passed=len(jobs) == EXPECTED_TRAINING_JOBS,
        )
        _add_check(
            report,
            "unique_readout_ids",
            observed=len(readout_index),
            expected=EXPECTED_READOUTS,
            passed=len(readout_index) == EXPECTED_READOUTS,
        )

        roles = Counter(str(row.get("job_role", "")) for row in training_jobs)
        _add_check(
            report,
            "job_role_counts",
            observed=dict(sorted(roles.items())),
            expected={"branch": EXPECTED_BRANCHES, "trunk": EXPECTED_TRUNKS},
            passed=roles
            == Counter({"branch": EXPECTED_BRANCHES, "trunk": EXPECTED_TRUNKS}),
        )
        incomplete = sorted(
            str(row.get("training_job_id"))
            for row in training_jobs
            if row.get("status") != "complete"
        )
        _add_check(
            report,
            "complete_training_jobs",
            observed=len(training_jobs) - len(incomplete),
            expected=EXPECTED_TRAINING_JOBS,
            passed=not incomplete,
            details=f"incomplete={incomplete[:8]}",
        )

        trajectories = {str(row.get("trajectory_id", "")) for row in training_jobs}
        _add_check(
            report,
            "trajectory_count",
            observed=len(trajectories),
            expected=EXPECTED_TRAJECTORIES,
            passed=(
                len(trajectories) == EXPECTED_TRAJECTORIES and "" not in trajectories
            ),
        )
        trunks_by_trajectory: dict[str, list[str]] = defaultdict(list)
        branches: dict[str, Mapping[str, Any]] = {}
        parent_errors: list[str] = []
        for job_id, job in jobs.items():
            role = str(job.get("job_role", ""))
            trajectory_id = str(job.get("trajectory_id", ""))
            parent_id = job.get("parent_training_job_id")
            if role == "trunk":
                trunks_by_trajectory[trajectory_id].append(job_id)
                if parent_id not in (None, ""):
                    parent_errors.append(f"{job_id}: trunk has parent {parent_id}")
            elif role == "branch":
                branches[job_id] = job
                parent = jobs.get(str(parent_id))
                if parent is None:
                    parent_errors.append(f"{job_id}: missing parent {parent_id}")
                elif parent.get("job_role") != "trunk":
                    parent_errors.append(f"{job_id}: parent is not trunk")
                elif parent.get("trajectory_id") != trajectory_id:
                    parent_errors.append(f"{job_id}: cross-trajectory parent")
        one_trunk = all(
            len(trunks_by_trajectory.get(trajectory, [])) == 1
            for trajectory in trajectories
        )
        _add_check(
            report,
            "one_trunk_per_trajectory",
            observed={
                trajectory: len(values)
                for trajectory, values in sorted(trunks_by_trajectory.items())
            },
            expected="exactly one for each of 64 trajectories",
            passed=one_trunk and len(trunks_by_trajectory) == EXPECTED_TRAJECTORIES,
        )
        _add_check(
            report,
            "same_trajectory_parent_links",
            observed=len(parent_errors),
            expected=0,
            passed=not parent_errors,
            details="; ".join(parent_errors[:8]),
        )

        artifact_payloads: dict[
            str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        ] = {}
        for job_id, job in sorted(jobs.items()):
            verifications, manifest, plan, result = _verify_job_artifacts(job)
            report["artifact_verifications"].extend(verifications)
            artifact_payloads[job_id] = (manifest, plan, result)
            role = str(job["job_role"])
            manifest_role = manifest.get("metadata", {}).get("plan_role")
            if manifest_role not in (None, role):
                report["errors"].append(
                    f"{job_id}: manifest plan_role {manifest_role!r} != {role!r}"
                )
            segments = plan.get("segments")
            if not isinstance(segments, list) or not segments:
                report["errors"].append(f"{job_id}: empty segment plan")
                continue
            plan_end = segments[-1].get("actual_end_tokens")
            if int(plan_end or -1) != int(job.get("actual_end_tokens", -2)):
                report["errors"].append(
                    f"{job_id}: segment-plan endpoint does not match result row"
                )
            if result:
                for field, expected in (
                    ("role", role),
                    ("actual_end_tokens", job.get("actual_end_tokens")),
                    ("final_checkpoint_step", job.get("final_checkpoint_step")),
                ):
                    if result.get(field) != expected:
                        report["errors"].append(
                            f"{job_id}: result {field} does not match job row"
                        )
            if role == "branch":
                initial = segments[0].get("initial_checkpoint")
                parent_id = str(job.get("parent_training_job_id", ""))
                initial_root = (
                    initial.get("checkpoint_root")
                    if isinstance(initial, Mapping)
                    else None
                )
                if not initial_root or parent_id not in _resolved(initial_root).parts:
                    report["errors"].append(
                        f"{job_id}: branch plan is not rooted in parent {parent_id}"
                    )
        _add_check(
            report,
            "artifact_hashes_and_plan_endpoints",
            observed=len(report["artifact_verifications"]),
            expected=2 * EXPECTED_TRAINING_JOBS,
            passed=(
                len(report["artifact_verifications"]) == 2 * EXPECTED_TRAINING_JOBS
                and not report["errors"]
            ),
        )

        readout_training_ids = [
            str(readout.get("training_job_id", "")) for readout in readouts
        ]
        _add_check(
            report,
            "readout_branch_bijection",
            observed=len(set(readout_training_ids)),
            expected=EXPECTED_BRANCHES,
            passed=(
                set(readout_training_ids) == set(branches)
                and len(set(readout_training_ids)) == len(readouts)
            ),
        )
        binding_errors: list[str] = []
        for readout in readouts:
            readout_id = str(readout["readout_id"])
            job_id = str(readout.get("training_job_id", ""))
            job = branches.get(job_id)
            errors: list[str] = []
            if job is None:
                errors.append("training job is not a production branch")
            else:
                scalar_pairs = (
                    ("trajectory_id", "trajectory_id"),
                    ("actual_tokens", "actual_end_tokens"),
                    ("expected_consumed_samples", "expected_consumed_samples"),
                    ("nominal_tokens", "target_tokens"),
                    ("checkpoint_step", "final_checkpoint_step"),
                )
                for readout_field, job_field in scalar_pairs:
                    if readout.get(readout_field) != job.get(job_field):
                        errors.append(f"{readout_field} != branch {job_field}")
                for readout_field, job_field in (
                    ("checkpoint_root", "final_checkpoint_root"),
                    ("training_manifest_path", "training_manifest_path"),
                ):
                    if _resolved(readout.get(readout_field)) != _resolved(
                        job.get(job_field)
                    ):
                        errors.append(f"{readout_field} != branch {job_field}")
                if readout.get("training_manifest_sha256") != job.get(
                    "training_manifest_sha256"
                ):
                    errors.append("training_manifest_sha256 differs")
                checkpoint_root = _resolved(readout.get("checkpoint_root"))
                checkpoint_step = int(readout.get("checkpoint_step", -1))
                expected_iteration = (
                    checkpoint_root / f"iter_{checkpoint_step:07d}"
                ).resolve()
                iteration_path = _resolved(readout.get("checkpoint_iteration_path"))
                if iteration_path != expected_iteration:
                    errors.append("checkpoint_iteration_path is not root/iter_step")
            binding_errors.extend(f"{readout_id}: {value}" for value in errors)
            checkpoint_exists = iteration_path.is_dir() if job is not None else None
            lineage_status = "fail" if errors else "pass"
            preservation_status = (
                "present"
                if checkpoint_exists is True
                else "absent_now"
                if checkpoint_exists is False
                else "not_applicable"
            )
            report["readout_bindings"].append(
                {
                    "readout_id": readout_id,
                    "training_job_id": job_id,
                    "trajectory_id": readout.get("trajectory_id"),
                    "ladder": readout.get("ladder"),
                    "experts": readout.get("experts"),
                    "language_count": readout.get("language_count"),
                    "checkpoint_iteration_path": str(iteration_path)
                    if job is not None
                    else None,
                    "checkpoint_iteration_exists_now": checkpoint_exists,
                    "lineage_status": lineage_status,
                    "checkpoint_preservation_status": preservation_status,
                    "status": (
                        "fail"
                        if errors
                        else f"lineage_valid_checkpoint_{preservation_status}"
                    ),
                    "errors": errors,
                }
            )
        _add_check(
            report,
            "readout_endpoint_checkpoint_bindings",
            observed=len(readouts)
            - len({value.split(":", 1)[0] for value in binding_errors}),
            expected=EXPECTED_READOUTS,
            passed=not binding_errors,
            details="; ".join(binding_errors[:8]),
        )
    except Stage1ADReleaseTrainingEnrichmentError as exc:
        report["errors"].append(str(exc))

    failed_checks = [row for row in report["checks"] if row.get("status") != "pass"]
    if report["errors"] or failed_checks:
        report["status"] = "fail"
        raise Stage1ADReleaseTrainingEnrichmentError(
            "Production graph validation failed: "
            + "; ".join(str(value) for value in report["errors"][:8]),
            report=report,
        )
    report["status"] = "pass"
    report["summary"] = {
        "training_jobs": len(training_jobs),
        "trunks": EXPECTED_TRUNKS,
        "branches": EXPECTED_BRANCHES,
        "readouts": len(readouts),
        "trajectories": EXPECTED_TRAJECTORIES,
        "recomputed_artifact_hashes": len(report["artifact_verifications"]),
        "checkpoint_directories_present_now": sum(
            row.get("checkpoint_iteration_exists_now") is True
            for row in report["readout_bindings"]
        ),
        "checkpoint_directories_absent_now": sum(
            row.get("checkpoint_iteration_exists_now") is False
            for row in report["readout_bindings"]
        ),
        "checkpoint_presence_semantics": (
            "Current preservation status is separate from execution-bound lineage "
            "validity and the retained normalized result artifacts."
        ),
    }
    return report


def _hardware_contract(repo_root: Path) -> list[dict[str, Any]]:
    path = (repo_root / H100_EFFICIENCY_DOCUMENT).resolve()
    if not path.is_file():
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"Missing frozen H100 efficiency document: {path}"
        )
    recomputed = _sha256(path)
    if recomputed != H100_EFFICIENCY_DOCUMENT_SHA256:
        raise Stage1ADReleaseTrainingEnrichmentError(
            "Frozen H100 efficiency document hash changed: "
            f"{recomputed} != {H100_EFFICIENCY_DOCUMENT_SHA256}"
        )
    text = path.read_text(encoding="utf-8")
    if "H100 SXM BF16 denominator: 989.5 TFLOP/s/GPU" not in text:
        raise Stage1ADReleaseTrainingEnrichmentError(
            "Frozen H100 document no longer states the configured denominator"
        )
    return [
        {
            "hardware_contract_id": "h100_sxm_bf16_989_5_v1",
            "accelerator_family": "NVIDIA H100 SXM",
            "precision": "BF16",
            "peak_tflops_per_gpu": H100_PEAK_TFLOPS_PER_GPU,
            "denominator_kind": "configured_theoretical_peak",
            "source_document_path": str(path),
            "source_document_sha256": recomputed,
            "source_statement": ("H100 SXM BF16 denominator: 989.5 TFLOP/s/GPU"),
            "mfu_formula": (
                "100 * observed Megatron model TFLOP/s/GPU / "
                "989.5 configured peak TFLOP/s/GPU"
            ),
            "scope_disclosure": (
                "This denominator is not achieved throughput and does not "
                "measure end-to-end hardware or facility efficiency."
            ),
            "provenance_class": "configured",
        }
    ]


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def enrich_training_summaries(
    training_jobs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach strict GPU-count, per-GPU throughput, MFU, and coverage fields."""

    jobs = _unique_index(training_jobs, "training_job_id", label="training job")
    summary_index = _unique_index(
        summaries, "training_job_id", label="training summary"
    )
    if set(jobs) != set(summary_index):
        raise Stage1ADReleaseTrainingEnrichmentError(
            "Training summary IDs do not exactly match training job IDs"
        )
    hardware = _hardware_contract(repo_root.resolve())
    enriched: list[dict[str, Any]] = []
    for job_id in sorted(jobs):
        job = jobs[job_id]
        source = summary_index[job_id]
        row = dict(source)
        gpu_count = _resource_gpu_count(job.get("resource"))
        throughput = _finite(source.get("mean_tokens_per_second"))
        tflops = _finite(source.get("mean_model_tflops_per_gpu"))
        walltime = _finite(
            source.get("walltime_seconds", job.get("end_to_end_seconds"))
        )
        observed_gpu_seconds = _finite(source.get("board_energy_observed_gpu_seconds"))
        telemetry_interval = _finite(source.get("telemetry_nominal_interval_seconds"))
        valid_samples = _finite(source.get("valid_gpu_telemetry_rows"))
        expected_gpu_seconds = (
            walltime * gpu_count if walltime is not None and walltime > 0 else None
        )
        temporal_coverage = (
            min(1.0, observed_gpu_seconds / expected_gpu_seconds)
            if observed_gpu_seconds is not None and expected_gpu_seconds
            else None
        )
        expected_samples = (
            (math.floor(walltime / telemetry_interval) + 1) * gpu_count
            if walltime is not None
            and walltime > 0
            and telemetry_interval is not None
            and telemetry_interval > 0
            else None
        )
        sample_coverage = (
            min(1.0, valid_samples / expected_samples)
            if valid_samples is not None and expected_samples
            else None
        )
        observed_uuids = _json_list(source.get("telemetry_gpu_uuids_json"))
        row.update(
            {
                "configured_resource": job["resource"],
                "gpu_count": gpu_count,
                "configured_gpu_count": gpu_count,
                "gpu_count_provenance_class": "configured",
                "hardware_contract_id": hardware[0]["hardware_contract_id"],
                "mean_tokens_per_second_per_gpu": (
                    throughput / gpu_count if throughput is not None else None
                ),
                "peak_tflops_per_gpu": H100_PEAK_TFLOPS_PER_GPU,
                "mean_mfu_pct_of_configured_peak": (
                    100.0 * tflops / H100_PEAK_TFLOPS_PER_GPU
                    if tflops is not None
                    else None
                ),
                "expected_gpu_seconds": expected_gpu_seconds,
                "telemetry_temporal_coverage_fraction": temporal_coverage,
                "board_energy_coverage_fraction": temporal_coverage,
                "expected_gpu_telemetry_rows": expected_samples,
                "gpu_sample_count_coverage_fraction": sample_coverage,
                "gpu_sample_count_coverage_status": _coverage_status(sample_coverage),
                "board_energy_coverage_status": _coverage_status(temporal_coverage),
                "telemetry_observed_gpu_count": (
                    len(set(str(value) for value in observed_uuids))
                    if observed_uuids
                    else None
                ),
                "telemetry_gpu_count_status": (
                    "match"
                    if observed_uuids
                    and len(set(str(value) for value in observed_uuids)) == gpu_count
                    else "mismatch"
                    if observed_uuids
                    else "not_observed"
                ),
                "mfu_enrichment_method": hardware[0]["mfu_formula"],
                "telemetry_coverage_method": (
                    "Integrated valid per-GPU telemetry seconds divided by "
                    "result walltime times configured GPU count. Sample-count "
                    "coverage uses the configured resource and observed monitor "
                    "interval; both are capped at one."
                ),
                "provenance_class": "derived",
            }
        )
        enriched.append(row)
    return enriched, hardware


def _parameter_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in (
            "training_job_id",
            "trajectory_id",
            "ladder",
            "experts",
            "language_count",
            "segment_id",
            "segment_kind",
            "parameter_scope",
            "parameter_count",
            "parameter_count_billions_reported",
            "tensor_model_parallel_rank",
            "pipeline_model_parallel_rank",
            "source_path",
            "source_line_number",
        )
    }


def _deduplicated_observations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        observation = _parameter_observation(raw)
        canonical = json.dumps(
            observation,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical not in seen:
            seen.add(canonical)
            result.append(observation)
    result.sort(
        key=lambda item: (
            str(item.get("training_job_id")),
            str(item.get("segment_id")),
            str(item.get("source_path")),
            int(item.get("source_line_number") or -1),
        )
    )
    return result


def enrich_model_configs(
    model_configs: Sequence[Mapping[str, Any]],
    runtime_parameters: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add configured design totals and scope-separated runtime observations."""

    model_index = _unique_index(model_configs, "model_config_id", label="model config")
    runtime_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for runtime in runtime_parameters:
        scope = str(runtime.get("parameter_scope", ""))
        if scope not in {
            "model_parallel_rank_partition",
            "reported_total_rounded_billions",
        }:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Unexpected runtime parameter scope: {scope!r}"
            )
        model_id = f"{runtime.get('ladder')}:E{int(runtime.get('experts', 0))}"
        if model_id not in model_index:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Runtime parameter row has no model config: {model_id}"
            )
        runtime_by_model[model_id].append(runtime)

    enriched: list[dict[str, Any]] = []
    for model_id, source in model_index.items():
        row = dict(source)
        layers = _positive_int(source.get("num_layers"), label="num_layers")
        d_model = _positive_int(source.get("hidden_size"), label="hidden_size")
        d_vocab = _positive_int(source.get("vocab_size"), label="vocab_size")
        experts = _positive_int(source.get("experts"), label="experts")
        configured_total = (
            2 * d_model * d_vocab + (4 + 9 * experts) * layers * d_model * d_model
        )
        observations = runtime_by_model.get(model_id, [])
        rank_local = _deduplicated_observations(
            [
                value
                for value in observations
                if value.get("parameter_scope") == "model_parallel_rank_partition"
            ]
        )
        rounded_global = _deduplicated_observations(
            [
                value
                for value in observations
                if value.get("parameter_scope") == "reported_total_rounded_billions"
            ]
        )
        row.update(
            {
                "configured_design_total_parameters": configured_total,
                "configured_design_total_parameters_formula": (
                    DESIGN_TOTAL_PARAMETER_FORMULA
                ),
                "configured_design_total_parameters_inputs": {
                    "d_model": d_model,
                    "d_vocab": d_vocab,
                    "layers": layers,
                    "E": experts,
                },
                "configured_design_total_parameters_provenance": "derived",
                "runtime_rank_local_parameter_observations": rank_local,
                "runtime_rank_local_parameter_observation_count": len(rank_local),
                "runtime_rank_local_parameter_count_values": sorted(
                    {
                        int(value["parameter_count"])
                        for value in rank_local
                        if value.get("parameter_count") is not None
                    }
                ),
                "runtime_rounded_global_parameter_observations": rounded_global,
                "runtime_rounded_global_parameter_observation_count": len(
                    rounded_global
                ),
                "runtime_rounded_global_parameter_count_values": sorted(
                    {
                        int(value["parameter_count"])
                        for value in rounded_global
                        if value.get("parameter_count") is not None
                    }
                ),
                "runtime_parameter_observation_status": (
                    "rank_local_and_rounded_global_observed"
                    if rank_local and rounded_global
                    else "partial"
                    if rank_local or rounded_global
                    else "missing"
                ),
                "parameter_scope_disclosure": (
                    "configured_design_total_parameters, declared active "
                    "parameters, declared active non-embedding parameters, "
                    "rank-local runtime counts, and rounded global runtime "
                    "prints are distinct quantities and are not substituted "
                    "for one another."
                ),
            }
        )
        enriched.append(row)
    return enriched


def _segment_compute(
    training_jobs: Sequence[Mapping[str, Any]],
    training_segments: Sequence[Mapping[str, Any]],
    training_metrics: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, float],
]:
    jobs = _unique_index(training_jobs, "training_job_id", label="training job")
    segment_index = _unique_index(
        training_segments, "segment_id", label="training segment"
    )
    elapsed_by_segment: dict[str, list[float]] = defaultdict(list)
    for metric in training_metrics:
        if metric.get("event_type") != "training_iteration":
            continue
        segment_id = str(metric.get("segment_id", ""))
        elapsed = _finite(metric.get("elapsed_time_ms"))
        if segment_id not in segment_index:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Training metric references unknown segment: {segment_id}"
            )
        if elapsed is not None and elapsed > 0:
            elapsed_by_segment[segment_id].append(elapsed)

    by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    job_totals: dict[str, float] = {}
    for segment_id, segment in segment_index.items():
        job_id = str(segment.get("training_job_id", ""))
        job = jobs.get(job_id)
        if job is None:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Segment references unknown training job: {segment_id}"
            )
        observations = elapsed_by_segment.get(segment_id, [])
        if not observations:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"No positive iteration-time observations: {segment_id}"
            )
        iterations = _positive_int(
            segment.get("segment_train_iters"),
            label=f"{segment_id}.segment_train_iters",
        )
        start = int(segment.get("actual_start_tokens", -1))
        end = int(segment.get("actual_end_tokens", -1))
        if start < 0 or end <= start:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Invalid token interval for {segment_id}: {start}..{end}"
            )
        mean_step_ms = statistics.mean(observations)
        gpu_count = _resource_gpu_count(job.get("resource"))
        estimated_gpu_hours = mean_step_ms / 1000.0 * iterations * gpu_count / 3600.0
        by_job[job_id].append(
            {
                "segment_id": segment_id,
                "actual_start_tokens": start,
                "actual_end_tokens": end,
                "segment_train_iters": iterations,
                "iteration_time_observations": len(observations),
                "mean_iteration_time_ms": mean_step_ms,
                "configured_gpu_count": gpu_count,
                "estimated_productive_gpu_hours": estimated_gpu_hours,
            }
        )
    for job_id in jobs:
        segments = sorted(
            by_job.get(job_id, []),
            key=lambda value: (
                value["actual_start_tokens"],
                value["actual_end_tokens"],
            ),
        )
        if not segments:
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Training job has no segments: {job_id}"
            )
        previous_end: int | None = None
        for segment in segments:
            if (
                previous_end is not None
                and segment["actual_start_tokens"] != previous_end
            ):
                raise Stage1ADReleaseTrainingEnrichmentError(
                    f"Non-contiguous token intervals for {job_id}: "
                    f"{previous_end} -> {segment['actual_start_tokens']}"
                )
            previous_end = int(segment["actual_end_tokens"])
        by_job[job_id] = segments
        job_totals[job_id] = sum(
            float(value["estimated_productive_gpu_hours"]) for value in segments
        )
    return dict(by_job), job_totals


def _prefix_gpu_hours(
    segments: Sequence[Mapping[str, Any]],
    prefix_tokens: int,
) -> float:
    if prefix_tokens < int(segments[0]["actual_start_tokens"]):
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"Prefix {prefix_tokens} precedes trunk segment coverage"
        )
    if prefix_tokens > int(segments[-1]["actual_end_tokens"]):
        raise Stage1ADReleaseTrainingEnrichmentError(
            f"Prefix {prefix_tokens} exceeds trunk segment coverage"
        )
    total = 0.0
    for segment in segments:
        start = int(segment["actual_start_tokens"])
        end = int(segment["actual_end_tokens"])
        hours = float(segment["estimated_productive_gpu_hours"])
        if prefix_tokens >= end:
            total += hours
        elif prefix_tokens > start:
            total += hours * (prefix_tokens - start) / (end - start)
            break
        else:
            break
    return total


def _covered_sum(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[float | None, float, float | None]:
    values = [
        value
        for row in rows
        if (value := _finite(row.get(field))) is not None and value >= 0
    ]
    coverage = len(values) / len(rows) if rows else 0.0
    observed_sum = sum(values) if values else None
    complete_sum = observed_sum if len(values) == len(rows) else None
    return observed_sum, coverage, complete_sum


def build_readout_compute_attribution(
    readouts: Sequence[Mapping[str, Any]],
    training_jobs: Sequence[Mapping[str, Any]],
    training_segments: Sequence[Mapping[str, Any]],
    training_metrics: Sequence[Mapping[str, Any]],
    compute_summary: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build logical per-readout attribution and deduplicated physical totals.

    Logical rows include the parent trunk prefix plus the complete branch.
    They are counterfactual per-readout costs and must never be summed as
    physical study compute.  Physical totals include every training job once.
    """

    jobs = _unique_index(training_jobs, "training_job_id", label="training job")
    summaries = _unique_index(
        compute_summary, "training_job_id", label="compute summary"
    )
    if set(jobs) != set(summaries):
        raise Stage1ADReleaseTrainingEnrichmentError(
            "Compute summary IDs do not exactly match training job IDs"
        )
    _unique_index(readouts, "readout_id", label="readout")
    segments_by_job, productive_by_job = _segment_compute(
        training_jobs,
        training_segments,
        training_metrics,
    )

    attribution: list[dict[str, Any]] = []
    for readout in sorted(readouts, key=lambda value: str(value["readout_id"])):
        branch_id = str(readout.get("training_job_id", ""))
        branch = jobs.get(branch_id)
        if branch is None or branch.get("job_role") != "branch":
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Readout is not bound to a branch: {readout.get('readout_id')}"
            )
        parent_id = str(branch.get("parent_training_job_id", ""))
        parent = jobs.get(parent_id)
        if parent is None or parent.get("job_role") != "trunk":
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Branch lacks a trunk parent: {branch_id}"
            )
        if parent.get("trajectory_id") != branch.get("trajectory_id"):
            raise Stage1ADReleaseTrainingEnrichmentError(
                f"Cross-trajectory parent for {branch_id}"
            )
        branch_segments = segments_by_job[branch_id]
        prefix_tokens = min(
            int(value["actual_start_tokens"]) for value in branch_segments
        )
        parent_prefix_hours = _prefix_gpu_hours(
            segments_by_job[parent_id],
            prefix_tokens,
        )
        branch_hours = productive_by_job[branch_id]
        attribution.append(
            {
                "readout_id": readout["readout_id"],
                "trajectory_id": readout["trajectory_id"],
                "ladder": readout.get("ladder"),
                "experts": readout.get("experts"),
                "language_count": readout.get("language_count"),
                "nominal_tokens": readout.get("nominal_tokens"),
                "branch_training_job_id": branch_id,
                "parent_trunk_training_job_id": parent_id,
                "parent_trunk_prefix_tokens": prefix_tokens,
                "attributed_parent_trunk_prefix_productive_gpu_hours": (
                    parent_prefix_hours
                ),
                "attributed_branch_productive_gpu_hours": branch_hours,
                "logical_readout_attributed_productive_gpu_hours": (
                    parent_prefix_hours + branch_hours
                ),
                "estimation_method": (
                    "For each segment: mean observed Megatron iteration time "
                    "times configured segment_train_iters times configured GPU "
                    "count. Parent trunk segments are included through the "
                    "branch initial-token prefix, with a linear within-segment "
                    "token fraction only when the prefix cuts a segment."
                ),
                "physical_aggregation_allowed": False,
                "aggregation_warning": (
                    "This is a logical per-readout attribution. Shared trunk "
                    "prefixes recur across readouts, so summing this column "
                    "would double-count physical training compute."
                ),
                "provenance_class": "derived",
            }
        )

    physical_jobs: list[dict[str, Any]] = []
    for job_id in sorted(jobs):
        job = jobs[job_id]
        summary = summaries[job_id]
        physical_jobs.append(
            {
                "physical_compute_unit_id": f"training_job:{job_id}",
                "training_job_id": job_id,
                "trajectory_id": job.get("trajectory_id"),
                "job_role": job.get("job_role"),
                "resource": job.get("resource"),
                "configured_gpu_count": _resource_gpu_count(job.get("resource")),
                "estimated_productive_gpu_hours": productive_by_job[job_id],
                "allocated_gpu_hours_including_retries": summary.get(
                    "allocated_gpu_hours_including_retries"
                ),
                "board_energy_estimate_kwh": summary.get("board_energy_estimate_kwh"),
                "board_energy_coverage_fraction": summary.get(
                    "board_energy_coverage_fraction"
                ),
                "physical_deduplication_key": job_id,
                "physical_job_count_weight": 1,
                "physical_aggregation_allowed": True,
                "scope_disclosure": (
                    "Each logical training job appears exactly once. Allocated "
                    "GPU-hours may include retries; productive GPU-hours are a "
                    "log-derived estimate; board energy excludes host and "
                    "facility overhead."
                ),
                "provenance_class": "derived",
            }
        )

    allocated_observed, allocated_coverage, allocated_complete = _covered_sum(
        physical_jobs,
        "allocated_gpu_hours_including_retries",
    )
    energy_observed, energy_coverage, energy_complete = _covered_sum(
        physical_jobs,
        "board_energy_estimate_kwh",
    )
    physical_study = [
        {
            "physical_compute_scope_id": "stage1_ad_unique_training_jobs",
            "unique_physical_training_job_count": len(physical_jobs),
            "unique_physical_trunk_count": sum(
                row["job_role"] == "trunk" for row in physical_jobs
            ),
            "unique_physical_branch_count": sum(
                row["job_role"] == "branch" for row in physical_jobs
            ),
            "estimated_productive_gpu_hours_unique_jobs": sum(
                float(row["estimated_productive_gpu_hours"]) for row in physical_jobs
            ),
            "observed_allocated_gpu_hours_including_retries_unique_jobs": (
                allocated_observed
            ),
            "allocated_gpu_hours_job_coverage_fraction": allocated_coverage,
            "complete_allocated_gpu_hours_including_retries_unique_jobs": (
                allocated_complete
            ),
            "observed_board_energy_estimate_kwh_unique_jobs": energy_observed,
            "board_energy_job_coverage_fraction": energy_coverage,
            "complete_board_energy_estimate_kwh_unique_jobs": energy_complete,
            "aggregation_method": (
                "Sum each unique training_job_id once. No logical per-readout "
                "attribution is included in physical totals."
            ),
            "logical_readout_attribution_sum": None,
            "logical_attribution_sum_status": (
                "intentionally_not_computed_because_shared_trunks_would_be_"
                "double_counted"
            ),
            "provenance_class": "derived",
        }
    ]
    return {
        "readout_compute_attribution": attribution,
        "unique_physical_job_compute": physical_jobs,
        "unique_physical_study_compute": physical_study,
    }


__all__ = [
    "DESIGN_TOTAL_PARAMETER_FORMULA",
    "H100_EFFICIENCY_DOCUMENT_SHA256",
    "H100_PEAK_TFLOPS_PER_GPU",
    "Stage1ADReleaseTrainingEnrichmentError",
    "build_readout_compute_attribution",
    "enrich_model_configs",
    "enrich_training_summaries",
    "validate_production_graph",
]
