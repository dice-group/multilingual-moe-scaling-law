"""Cross-table integrity checks for the Stage 1 A--D release."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    Stage1ReleaseError,
    _json_cell,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _segment_uses_exact_step_rounding(
    segment: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    step_tokens = int(config["sequence_length"]) * int(segment["global_batch_size"])
    target_start = int(segment["target_start_tokens"])
    target_end = int(segment["target_end_tokens"])
    actual_start = int(segment["actual_start_tokens"])
    actual_end = int(segment["actual_end_tokens"])
    if step_tokens <= 0 or target_start < 0 or target_end <= target_start:
        return False
    segment_kind = str(segment.get("segment_kind") or "")
    if segment_kind == "trunk":
        return (
            actual_start == math.ceil(target_start / step_tokens) * step_tokens
            and actual_end == math.ceil(target_end / step_tokens) * step_tokens
        )
    if segment_kind == "branch":
        target_span = target_end - target_start
        actual_span = actual_end - actual_start
        return (
            actual_start >= target_start
            and actual_span == math.ceil(target_span / step_tokens) * step_tokens
        )
    return False


def _training_metric_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], int]:
    errors: list[str] = []
    iteration_resets = 0
    previous_iteration: dict[tuple[str, str, str], int] = {}
    previous_source_line: dict[str, int] = {}
    for row in rows:
        if row.get("event_type") != "training_iteration":
            continue
        source_path = str(row.get("source_path") or "")
        source_line = row.get("source_line_number")
        identity = f"{row.get('training_job_id')}:{row.get('segment_id')}"
        if not source_path or source_line is None:
            errors.append(identity + ":missing_source_order")
        else:
            line = int(source_line)
            if (
                source_path in previous_source_line
                and line <= previous_source_line[source_path]
            ):
                errors.append(identity + ":source_line_order")
            previous_source_line[source_path] = line
        iteration = row.get("iteration")
        if iteration is None or int(iteration) <= 0:
            errors.append(identity + ":invalid_iteration")
            continue
        key = (str(row.get("training_job_id")), str(row.get("segment_id")), source_path)
        value = int(iteration)
        if key in previous_iteration and value <= previous_iteration[key]:
            iteration_resets += 1
        previous_iteration[key] = value
        for field in ("lm_loss", "elapsed_time_ms"):
            item = row.get(field)
            if item is not None and not math.isfinite(float(item)):
                errors.append(identity + f":nonfinite_{field}")
        elapsed = row.get("elapsed_time_ms")
        if elapsed is not None and float(elapsed) <= 0:
            errors.append(identity + ":nonpositive_elapsed_time")
    return errors, iteration_resets


def validate_release_relations(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_validation_candidate_count: int = 435,
    expected_duplicate_candidate_count: int = 35,
) -> list[dict[str, Any]]:
    """Fail closed on primary-key, foreign-key, endpoint, and lineage errors."""

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(
        check_id: str,
        *,
        observed: Any,
        expected: Any,
        passed: bool,
        details: str,
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "observed": observed,
                "expected": expected,
                "status": "pass" if passed else "fail",
                "severity": "error",
                "details": details,
                "checked_at_utc": _now(),
                "provenance_class": "derived",
            }
        )
        if not passed:
            failures.append(check_id)

    primary_keys = {
        "source_roots": ("source_root_id",),
        "model_configs": ("model_config_id",),
        "trajectories": ("trajectory_id",),
        "readouts": ("readout_id",),
        "validation_language": ("readout_id", "language"),
        "validation_candidates": (
            "readout_id",
            "candidate_manifest_path",
        ),
        "validation_artifacts": ("artifact_id",),
        "validation_provenance": ("provenance_id",),
        "training_jobs": ("training_job_id",),
        "training_configs": ("training_job_id",),
        "training_segments": ("segment_id",),
        "training_summary": ("training_job_id",),
        "training_artifacts": ("artifact_type", "object_id", "path"),
        "languages": ("language",),
        "language_tier_membership": ("language_count", "language"),
        "data_exposure": ("readout_id", "language"),
        "routing_layers": ("readout_id", "language", "layer_index"),
        "job_attempts": ("attempt_uid",),
        "slurm_allocations": ("allocation_uid",),
        "compute_summary": ("training_job_id",),
        "readout_compute_attribution": ("readout_id",),
    }
    for table_name, columns in primary_keys.items():
        rows = tables.get(table_name, ())
        keys = [_key(row, columns) for row in rows]
        unique = len(set(keys))
        valid = unique == len(keys) and all(
            all(value is not None for value in key) for key in keys
        )
        record(
            f"{table_name}_primary_key",
            observed=f"{unique}/{len(keys)} unique",
            expected=f"{len(keys)}/{len(keys)} unique and non-null",
            passed=valid,
            details=f"Primary key columns: {', '.join(columns)}.",
        )

    model_ids = {str(row["model_config_id"]) for row in tables["model_configs"]}
    trajectory_rows = list(tables["trajectories"])
    trajectory_ids = {str(row["trajectory_id"]) for row in trajectory_rows}
    readout_rows = list(tables["readouts"])
    readout_ids = {str(row["readout_id"]) for row in readout_rows}
    training_rows = list(tables["training_jobs"])
    training_ids = {str(row["training_job_id"]) for row in training_rows}
    language_ids = {str(row["language"]) for row in tables["languages"]}
    attempt_ids = {str(row["attempt_uid"]) for row in tables["job_attempts"]}

    relation_checks = {
        "trajectory_to_model": (
            {str(row["model_config_id"]) for row in trajectory_rows},
            model_ids,
        ),
        "readout_to_trajectory": (
            {str(row["trajectory_id"]) for row in readout_rows},
            trajectory_ids,
        ),
        "readout_to_training_job": (
            {str(row["training_job_id"]) for row in readout_rows},
            training_ids,
        ),
        "training_job_to_trajectory": (
            {str(row["trajectory_id"]) for row in training_rows},
            trajectory_ids,
        ),
        "validation_to_readout": (
            {str(row["readout_id"]) for row in tables["validation_language"]},
            readout_ids,
        ),
        "exposure_to_readout": (
            {str(row["readout_id"]) for row in tables["data_exposure"]},
            readout_ids,
        ),
        "exposure_to_language": (
            {str(row["language"]) for row in tables["data_exposure"]},
            language_ids,
        ),
        "routing_to_readout": (
            {str(row["readout_id"]) for row in tables["routing_layers"]},
            readout_ids,
        ),
        "routing_to_language": (
            {str(row["language"]) for row in tables["routing_layers"]},
            language_ids,
        ),
        "allocation_to_attempt": (
            {str(row["attempt_uid"]) for row in tables["slurm_allocations"]},
            attempt_ids,
        ),
    }
    for check_id, (children, parents) in relation_checks.items():
        missing = sorted(children - parents)
        record(
            check_id,
            observed=len(missing),
            expected=0,
            passed=not missing,
            details=(
                "Full foreign-key closure."
                if not missing
                else f"Missing parent keys: {missing[:8]}"
            ),
        )

    validation_trained = {
        (str(row["readout_id"]), str(row["language"]))
        for row in tables["validation_language"]
        if row.get("metric_role") == "trained"
    }
    exposure_keys = {
        (str(row["readout_id"]), str(row["language"]))
        for row in tables["data_exposure"]
    }
    record(
        "exposure_exactly_matches_trained_validation_roles",
        observed=len(exposure_keys ^ validation_trained),
        expected=0,
        passed=exposure_keys == validation_trained,
        details=(
            "Exposure rows must exist for every and only trained-language "
            "validation rows."
        ),
    )
    unsafe_exposure = [
        row
        for row in tables["data_exposure"]
        if row.get("exact_unique_language_tokens_including_eod") is None
        and row.get("effective_exposure_passes") is not None
    ]
    record(
        "censored_exposure_denominators_not_used_as_exact",
        observed=len(unsafe_exposure),
        expected=0,
        passed=not unsafe_exposure,
        details=(
            "A lower-bound unique-token inventory may be retained but cannot "
            "produce an exact effective-pass value."
        ),
    )

    jobs_by_id = {str(row["training_job_id"]): row for row in training_rows}
    configs_by_id = {
        str(row["training_job_id"]): row for row in tables["training_configs"]
    }
    endpoint_errors: list[str] = []
    for readout in readout_rows:
        job_id = str(readout["training_job_id"])
        job = jobs_by_id[job_id]
        config = configs_by_id[job_id]
        actual = int(readout["actual_tokens"])
        sequence_length = int(config["sequence_length"])
        if actual != int(job["expected_consumed_samples"]) * sequence_length:
            endpoint_errors.append(str(readout["readout_id"]))
        if (
            str(readout["checkpoint_root"]) != str(job["final_checkpoint_root"])
            or int(readout["checkpoint_step"]) != int(job["final_checkpoint_step"])
            or actual != int(job["actual_end_tokens"])
        ):
            endpoint_errors.append(str(readout["readout_id"]))
    record(
        "readout_training_checkpoint_endpoint_binding",
        observed=len(set(endpoint_errors)),
        expected=0,
        passed=not endpoint_errors,
        details=(
            "Readout actual tokens, consumed samples, final checkpoint root, "
            "and step agree with the admitted production result."
        ),
    )

    segments_by_job: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in tables["training_segments"]:
        segments_by_job[str(row["training_job_id"])].append(row)
    segment_errors: list[str] = []
    segment_rounding_errors: list[str] = []
    for job_id, job in jobs_by_id.items():
        segments = sorted(
            segments_by_job.get(job_id, []),
            key=lambda row: int(row["segment_index"]),
        )
        if not segments:
            segment_errors.append(job_id)
            continue
        indices = [int(row["segment_index"]) for row in segments]
        if indices != list(range(len(indices))):
            segment_errors.append(job_id)
        for segment in segments:
            if not _segment_uses_exact_step_rounding(segment, configs_by_id[job_id]):
                segment_rounding_errors.append(str(segment["segment_id"]))
        for previous, current in zip(segments, segments[1:], strict=False):
            if int(previous["actual_end_tokens"]) != int(
                current["actual_start_tokens"]
            ):
                segment_errors.append(job_id)
        if int(segments[-1]["actual_end_tokens"]) != int(job["actual_end_tokens"]):
            segment_errors.append(job_id)
    record(
        "training_segment_continuity_and_endpoints",
        observed=len(set(segment_errors)),
        expected=0,
        passed=not segment_errors,
        details=(
            "Segment indices are contiguous, token boundaries meet, and the "
            "last segment equals the production result endpoint."
        ),
    )

    record(
        "training_segment_exact_optimizer_step_rounding",
        observed=len(set(segment_rounding_errors)),
        expected=0,
        passed=not segment_rounding_errors,
        details=(
            "Trunk segments round cumulative start and end targets in their "
            "configured optimizer-step quantum. Branches begin at the sealed "
            "parent checkpoint and round their incremental target span. These "
            "are the endpoint contracts recorded by the training manifests."
        ),
    )

    metric_errors, iteration_resets = _training_metric_diagnostics(
        tables["training_metrics"]
    )
    record(
        "training_metric_source_order_and_finiteness",
        observed=_json_cell(
            {
                "invalid_metric_streams": len(set(metric_errors)),
                "observed_iteration_resets": iteration_resets,
            }
        ),
        expected=_json_cell(
            {
                "invalid_metric_streams": 0,
                "iteration_resets": "observed and disclosed",
            }
        ),
        passed=not metric_errors,
        details=(
            "Source line order and finite positive timing are required. Iteration "
            "resets are retained as observed retry/resume boundaries in "
            "concatenated logs; they are not silently sorted or treated as new "
            "independent trajectories."
        ),
    )

    summary_by_id = {
        str(row["training_job_id"]): row for row in tables["training_summary"]
    }
    gpu_ids_by_job: dict[str, set[str]] = defaultdict(set)
    time_by_job_gpu: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in tables["gpu_samples"]:
        job_id = str(row["training_job_id"])
        gpu_id = str(row.get("gpu_uuid") or f"index:{row.get('gpu_index')}")
        gpu_ids_by_job[job_id].add(gpu_id)
        timestamp = row.get("timestamp_utc") or row.get("timestamp")
        if timestamp:
            time_by_job_gpu[(job_id, gpu_id)].append(str(timestamp))
    telemetry_errors: list[str] = []
    for job_id, summary in summary_by_id.items():
        expected_gpu_count = summary.get("gpu_count")
        if expected_gpu_count is not None and len(
            gpu_ids_by_job.get(job_id, set())
        ) != int(expected_gpu_count):
            telemetry_errors.append(job_id)
    for (job_id, _), timestamps in time_by_job_gpu.items():
        if timestamps != sorted(timestamps):
            telemetry_errors.append(job_id)
    record(
        "gpu_telemetry_identity_and_timestamp_consistency",
        observed=len(set(telemetry_errors)),
        expected=0,
        passed=not telemetry_errors,
        details=(
            "Telemetry has the configured GPU identity count per training job "
            "and nondecreasing timestamps within each GPU stream."
        ),
    )

    candidate_count = len(tables["validation_candidates"])
    duplicate_count = candidate_count - len(readout_rows)
    record(
        "frozen_candidate_inventory",
        observed=_json_cell(
            {
                "candidates": candidate_count,
                "duplicates_beyond_one_per_readout": duplicate_count,
            }
        ),
        expected=_json_cell(
            {
                "candidates": expected_validation_candidate_count,
                "duplicates_beyond_one_per_readout": expected_duplicate_candidate_count,
            }
        ),
        passed=(
            candidate_count == expected_validation_candidate_count
            and duplicate_count == expected_duplicate_candidate_count
        ),
        details=(
            "Versioned release contract for selected and alternative validation "
            "candidates."
        ),
    )

    if failures:
        raise Stage1ReleaseError(
            "Cross-table release integrity failed: " + ", ".join(failures)
        )
    return checks


__all__ = ["validate_release_relations"]
