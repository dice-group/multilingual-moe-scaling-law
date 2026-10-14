"""Normalized Stage 1 A--D training and GPU-telemetry ingestion.

The helpers in this module deliberately return plain dictionaries.  They are
intended to feed the central Stage 1 tables without making pandas a runtime
dependency of the experiment package.

Two provenance rules are important:

* callers may attach scientific and execution lineage to every parsed row;
* energy is reported only as an estimate from ``nvidia-smi power.draw``.  It is
  board energy over observed intervals, not whole-node or facility energy.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SCHEMA_VERSION = 1

_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|nan|inf)"
_TIMESTAMP_RE = re.compile(r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} [^\]]+)\]")
_ITERATION_RE = re.compile(
    r"\biteration\s+(?P<iteration>\d+)\s*/\s*(?P<iteration_target>\d+)",
    re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"number of parameters on \(tensor,\s*pipeline\) model parallel rank "
    r"\((?P<tensor_rank>\d+),\s*(?P<pipeline_rank>\d+)\):\s*"
    r"(?P<parameter_count>\d+)",
    re.IGNORECASE,
)
_TOTAL_PARAMETER_RE = re.compile(
    rf"Total number of parameters in billions:\s*(?P<billions>{_NUMBER})",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    rf"mem-max-allocated-gigabytes:\s*(?P<allocated>{_NUMBER}).*?"
    rf"mem-max-reserved-gigabytes:\s*(?P<reserved>{_NUMBER})",
    re.IGNORECASE,
)
_STANDALONE_TFLOPS_RE = re.compile(
    rf"GPU utilization:\s*(?P<tflops>{_NUMBER})\s*MODEL_TFLOP/s/GPU",
    re.IGNORECASE,
)

_ITERATION_FIELDS: tuple[tuple[str, str, type], ...] = (
    ("consumed_samples", r"consumed samples", int),
    ("elapsed_time_ms", r"elapsed time per iteration \(ms\)", float),
    ("model_tflops_per_gpu", r"throughput per GPU \(TFLOP/s/GPU\)", float),
    ("learning_rate", r"learning rate", float),
    ("global_batch_size", r"global batch size", int),
    ("lm_loss", r"lm loss", float),
    ("load_balancing_loss", r"load[_ ]balancing[_ ]loss", float),
    ("z_loss", r"z[_ ]loss", float),
    ("grad_norm", r"grad norm", float),
    ("skipped_iterations", r"number of skipped iterations", int),
    ("nan_iterations", r"number of nan iterations", int),
)

_LINEAGE_KEYS = (
    "trajectory_id",
    "ladder",
    "experts",
    "language_count",
    "training_job_id",
    "job_id",
    "slurm_job_id",
    "attempt_id",
    "run_id",
    "segment_id",
    "segment_index",
    "segment_kind",
    "parent_segment_id",
    "trunk_job_id",
    "branch_parent_job_id",
    "source_path",
)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _field(line: str, label: str, converter: type) -> int | float | None:
    match = re.search(
        rf"\b{label}\s*:\s*(?P<value>{_NUMBER})",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    if converter is int:
        return _finite_int(match.group("value"))
    return _finite_float(match.group("value"))


def _normalized_timestamp(
    raw: str,
    *,
    timezone_name: str | None,
) -> tuple[str | None, str]:
    """Return an ISO timestamp and explicit timezone provenance.

    Megatron and ``nvidia-smi`` timestamps do not contain an offset.  They stay
    naive unless the caller supplies the known scheduler-node timezone.
    """

    value = raw.strip()
    parsed: datetime | None = None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None, timezone_name or "unspecified"
    if timezone_name:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {timezone_name}") from exc
    timespec = "milliseconds" if parsed.microsecond else "seconds"
    return parsed.isoformat(timespec=timespec), timezone_name or "unspecified"


def _base_row(lineage: Mapping[str, Any] | None) -> dict[str, Any]:
    row = dict(lineage or {})
    row["schema_version"] = SCHEMA_VERSION
    return row


def infer_training_log_lineage(path: Path) -> dict[str, Any]:
    """Infer only lineage that is encoded unambiguously in a training path."""

    resolved = path.resolve()
    parts = resolved.parts
    lineage: dict[str, Any] = {"source_path": str(resolved)}
    if "jobs" in parts:
        index = parts.index("jobs")
        if index + 1 < len(parts):
            lineage["training_job_id"] = parts[index + 1]
            lineage["job_id"] = parts[index + 1]
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            lineage["run_id"] = parts[index + 1]
    if "segments" in parts:
        segment_name = path.name
        for suffix in (".train.log", ".log"):
            if segment_name.endswith(suffix):
                segment_name = segment_name[: -len(suffix)]
                break
        lineage["segment_id"] = segment_name
        segment_match = re.match(r"(?P<index>\d+)_", segment_name)
        if segment_match:
            lineage["segment_index"] = int(segment_match.group("index"))
    identity = " ".join(
        str(lineage.get(key, "")) for key in ("training_job_id", "run_id", "segment_id")
    ).lower()
    if "trunk" in identity:
        lineage["segment_kind"] = "trunk"
    elif "branch" in identity:
        lineage["segment_kind"] = "branch"
    else:
        lineage["segment_kind"] = "unclassified"
    return lineage


def parse_megatron_training_line(
    line: str,
    *,
    sequence_length: int | None = None,
    lineage: Mapping[str, Any] | None = None,
    timestamp_timezone: str | None = None,
) -> dict[str, Any] | None:
    """Parse one Megatron metric, memory, TFLOPS, or parameter-count line."""

    timestamp_match = _TIMESTAMP_RE.search(line)
    timestamp: str | None = None
    timezone_provenance = timestamp_timezone or "unspecified"
    if timestamp_match:
        timestamp, timezone_provenance = _normalized_timestamp(
            timestamp_match.group("timestamp"),
            timezone_name=timestamp_timezone,
        )

    iteration_match = _ITERATION_RE.search(line)
    if iteration_match:
        row = _base_row(lineage)
        row.update(
            {
                "event_type": "training_iteration",
                "timestamp": timestamp,
                "timestamp_timezone": timezone_provenance,
                "iteration": int(iteration_match.group("iteration")),
                "iteration_target": int(iteration_match.group("iteration_target")),
                "sequence_length": int(sequence_length)
                if sequence_length is not None
                else None,
            }
        )
        for output_name, label, converter in _ITERATION_FIELDS:
            row[output_name] = _field(line, label, converter)
        consumed_samples = row["consumed_samples"]
        global_batch_size = row["global_batch_size"]
        elapsed_ms = row["elapsed_time_ms"]
        if sequence_length is not None and consumed_samples is not None:
            row["consumed_tokens"] = int(consumed_samples) * int(sequence_length)
        else:
            row["consumed_tokens"] = None
        if elapsed_ms is not None and elapsed_ms > 0 and global_batch_size is not None:
            row["samples_per_second"] = global_batch_size / (elapsed_ms / 1000.0)
            row["tokens_per_second"] = (
                row["samples_per_second"] * sequence_length
                if sequence_length is not None
                else None
            )
        else:
            row["samples_per_second"] = None
            row["tokens_per_second"] = None
        return row

    parameter_match = _PARAMETER_RE.search(line)
    if parameter_match:
        row = _base_row(lineage)
        row.update(
            {
                "event_type": "parameter_count",
                "timestamp": timestamp,
                "timestamp_timezone": timezone_provenance,
                "parameter_scope": "model_parallel_rank_partition",
                "parameter_count": int(parameter_match.group("parameter_count")),
                "tensor_model_parallel_rank": int(parameter_match.group("tensor_rank")),
                "pipeline_model_parallel_rank": int(
                    parameter_match.group("pipeline_rank")
                ),
            }
        )
        return row

    total_parameter_match = _TOTAL_PARAMETER_RE.search(line)
    if total_parameter_match:
        billions = _finite_float(total_parameter_match.group("billions"))
        row = _base_row(lineage)
        row.update(
            {
                "event_type": "parameter_count",
                "timestamp": timestamp,
                "timestamp_timezone": timezone_provenance,
                "parameter_scope": "reported_total_rounded_billions",
                "parameter_count": int(round(billions * 1_000_000_000))
                if billions is not None
                else None,
                "parameter_count_billions_reported": billions,
                "tensor_model_parallel_rank": None,
                "pipeline_model_parallel_rank": None,
            }
        )
        return row

    memory_match = _MEMORY_RE.search(line)
    if memory_match:
        row = _base_row(lineage)
        row.update(
            {
                "event_type": "training_memory",
                "timestamp": timestamp,
                "timestamp_timezone": timezone_provenance,
                "max_allocated_gib": _finite_float(memory_match.group("allocated")),
                "max_reserved_gib": _finite_float(memory_match.group("reserved")),
            }
        )
        return row

    tflops_match = _STANDALONE_TFLOPS_RE.search(line)
    if tflops_match:
        row = _base_row(lineage)
        row.update(
            {
                "event_type": "model_tflops",
                "timestamp": timestamp,
                "timestamp_timezone": timezone_provenance,
                "model_tflops_per_gpu": _finite_float(tflops_match.group("tflops")),
            }
        )
        return row
    return None


def parse_megatron_training_lines(
    lines: Iterable[str],
    *,
    sequence_length: int | None = None,
    lineage: Mapping[str, Any] | None = None,
    timestamp_timezone: str | None = None,
) -> list[dict[str, Any]]:
    """Parse normalized events from an iterable and retain source line numbers."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        row = parse_megatron_training_line(
            line,
            sequence_length=sequence_length,
            lineage=lineage,
            timestamp_timezone=timestamp_timezone,
        )
        if row is not None:
            row["source_line_number"] = line_number
            rows.append(row)
    return rows


def parse_megatron_training_log(
    path: Path,
    *,
    sequence_length: int | None = None,
    lineage: Mapping[str, Any] | None = None,
    timestamp_timezone: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a training log, combining path-inferred and supplied lineage."""

    combined_lineage = infer_training_log_lineage(path)
    combined_lineage.update(dict(lineage or {}))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return parse_megatron_training_lines(
            handle,
            sequence_length=sequence_length,
            lineage=combined_lineage,
            timestamp_timezone=timestamp_timezone,
        )


def _csv_source(
    source: Path | TextIO | Iterable[str],
) -> tuple[Iterable[str], TextIO | None]:
    if isinstance(source, Path):
        handle = source.open("r", encoding="utf-8", errors="replace", newline="")
        return handle, handle
    return source, None


def parse_nvidia_smi_csv(
    source: Path | TextIO | Iterable[str],
    *,
    lineage: Mapping[str, Any] | None = None,
    timestamp_timezone: str | None = None,
) -> list[dict[str, Any]]:
    """Parse executor ``nvidia-smi`` CSV samples into normalized rows.

    Malformed samples are retained with ``sample_valid=False`` so coverage
    accounting cannot silently mistake them for missing time.
    """

    rows: list[dict[str, Any]] = []
    csv_lines, handle = _csv_source(source)
    try:
        for source_row_number, raw_row in enumerate(csv.DictReader(csv_lines), start=2):
            stripped = {
                str(key).strip(): str(value).strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            timestamp_raw = stripped.get("timestamp", "")
            timestamp, timezone_provenance = _normalized_timestamp(
                timestamp_raw,
                timezone_name=timestamp_timezone,
            )
            gpu_index = _finite_int(stripped.get("index"))
            utilization = _finite_float(stripped.get("utilization_gpu_pct"))
            memory_used = _finite_float(stripped.get("memory_used_mib"))
            memory_total = _finite_float(stripped.get("memory_total_mib"))
            power = _finite_float(stripped.get("power_draw_w"))
            sample_valid = (
                timestamp is not None
                and gpu_index is not None
                and utilization is not None
                and memory_used is not None
                and memory_total is not None
                and memory_total > 0
                and power is not None
            )
            row = _base_row(lineage)
            row.update(
                {
                    "event_type": "gpu_telemetry",
                    "source_row_number": source_row_number,
                    "timestamp_raw": timestamp_raw,
                    "timestamp": timestamp,
                    "timestamp_timezone": timezone_provenance,
                    "gpu_index": gpu_index,
                    "gpu_uuid": stripped.get("uuid") or None,
                    "gpu_utilization_pct": utilization,
                    "memory_used_mib": memory_used,
                    "memory_total_mib": memory_total,
                    "memory_used_fraction": (
                        memory_used / memory_total
                        if memory_used is not None
                        and memory_total is not None
                        and memory_total > 0
                        else None
                    ),
                    "board_power_draw_w": power,
                    "sample_valid": sample_valid,
                }
            )
            rows.append(row)
    finally:
        if handle is not None:
            handle.close()

    previous_by_gpu: dict[str, datetime] = {}
    for row in rows:
        timestamp = _as_datetime(row.get("timestamp"))
        gpu_identity = _gpu_identity(row)
        previous = previous_by_gpu.get(gpu_identity)
        row["seconds_since_previous_sample"] = (
            (timestamp - previous).total_seconds()
            if timestamp is not None and previous is not None and timestamp > previous
            else None
        )
        if timestamp is not None:
            previous_by_gpu[gpu_identity] = timestamp
    return rows


def _as_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _gpu_identity(row: Mapping[str, Any]) -> str:
    uuid = row.get("gpu_uuid")
    return str(uuid) if uuid else f"index:{row.get('gpu_index')}"


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return statistics.mean(finite) if finite else None


def _median(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return statistics.median(finite) if finite else None


def _maximum(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return max(finite) if finite else None


def _coverage_status(value: float | None) -> str:
    if value is None:
        return "not_computable"
    if value >= 0.99:
        return "complete"
    if value > 0:
        return "partial"
    return "missing"


def _lineage_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            key: row[key]
            for key in _LINEAGE_KEYS
            if key in row and row[key] is not None
        }
        signature = tuple(sorted((key, str(value)) for key, value in record.items()))
        if record and signature not in seen:
            seen.add(signature)
            records.append(record)
    return records


def _energy_estimate(
    gpu_rows: Iterable[Mapping[str, Any]],
    *,
    expected_sample_interval_seconds: float | None,
) -> dict[str, Any]:
    by_gpu: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in gpu_rows:
        timestamp = _as_datetime(row.get("timestamp"))
        power = _finite_float(row.get("board_power_draw_w"))
        if timestamp is not None and power is not None:
            by_gpu[_gpu_identity(row)].append((timestamp, power))

    positive_gaps = [
        (current[0] - previous[0]).total_seconds()
        for samples in by_gpu.values()
        for previous, current in zip(
            sorted(samples),
            sorted(samples)[1:],
            strict=False,
        )
        if current[0] > previous[0]
    ]
    if expected_sample_interval_seconds is not None:
        if expected_sample_interval_seconds <= 0:
            raise ValueError("expected_sample_interval_seconds must be positive")
        nominal_interval = float(expected_sample_interval_seconds)
    else:
        nominal_interval = statistics.median(positive_gaps) if positive_gaps else None
    maximum_gap = 2.5 * nominal_interval if nominal_interval is not None else None

    watt_seconds = 0.0
    observed_gpu_seconds = 0.0
    integrated_intervals = 0
    excluded_gap_count = 0
    for samples in by_gpu.values():
        ordered = sorted(samples)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            seconds = (current[0] - previous[0]).total_seconds()
            if seconds <= 0:
                continue
            if maximum_gap is not None and seconds > maximum_gap:
                excluded_gap_count += 1
                continue
            watt_seconds += 0.5 * (previous[1] + current[1]) * seconds
            observed_gpu_seconds += seconds
            integrated_intervals += 1
    return {
        "board_energy_estimate_kwh": (
            watt_seconds / 3_600_000.0 if integrated_intervals else None
        ),
        "board_energy_observed_gpu_seconds": observed_gpu_seconds,
        "board_energy_integrated_intervals": integrated_intervals,
        "board_energy_excluded_gap_count": excluded_gap_count,
        "telemetry_nominal_interval_seconds": nominal_interval,
        "board_energy_max_integrated_gap_seconds": maximum_gap,
        "board_energy_estimate_method": (
            "Trapezoidal integration of nvidia-smi power.draw for each GPU over "
            "observed intervals; intervals above 2.5x the nominal sampling "
            "interval are excluded. This estimates GPU board energy only and "
            "excludes host, network, storage, cooling, and facility overhead."
        ),
    }


def aggregate_job_training(
    training_rows: Iterable[Mapping[str, Any]],
    gpu_rows: Iterable[Mapping[str, Any]],
    *,
    gpu_count: int | None = None,
    peak_tflops_per_gpu: float | None = None,
    walltime_seconds: float | None = None,
    expected_logged_metric_rows: int | None = None,
    expected_gpu_sample_interval_seconds: float | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate one execution job without erasing trunk/branch lineage.

    ``tokens_per_second`` is job-level throughput derived from global batch,
    sequence length, and logged step time.  MFU is available only when the
    caller supplies the hardware-specific configured peak.  Coverage fields
    remain ``None`` instead of implying completeness when expectations are
    unavailable.
    """

    training = [dict(row) for row in training_rows]
    telemetry = [dict(row) for row in gpu_rows]
    iterations = [
        row for row in training if row.get("event_type") == "training_iteration"
    ]
    valid_gpu_samples = [row for row in telemetry if row.get("sample_valid") is True]
    if gpu_count is not None and gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if peak_tflops_per_gpu is not None and peak_tflops_per_gpu <= 0:
        raise ValueError("peak_tflops_per_gpu must be positive")
    if walltime_seconds is not None and walltime_seconds <= 0:
        raise ValueError("walltime_seconds must be positive")
    if expected_logged_metric_rows is not None and expected_logged_metric_rows <= 0:
        raise ValueError("expected_logged_metric_rows must be positive")

    step_ms = [row.get("elapsed_time_ms") for row in iterations]
    tokens_per_second = [row.get("tokens_per_second") for row in iterations]
    samples_per_second = [row.get("samples_per_second") for row in iterations]
    tflops = [row.get("model_tflops_per_gpu") for row in iterations]
    mean_tflops = _mean(tflops)
    mfu = (
        100.0 * mean_tflops / peak_tflops_per_gpu
        if mean_tflops is not None and peak_tflops_per_gpu is not None
        else None
    )
    metric_coverage = (
        min(1.0, len(iterations) / expected_logged_metric_rows)
        if expected_logged_metric_rows is not None
        else None
    )

    energy = _energy_estimate(
        valid_gpu_samples,
        expected_sample_interval_seconds=expected_gpu_sample_interval_seconds,
    )
    expected_gpu_seconds = (
        walltime_seconds * gpu_count
        if walltime_seconds is not None and gpu_count is not None
        else None
    )
    energy_coverage = (
        min(
            1.0,
            energy["board_energy_observed_gpu_seconds"] / expected_gpu_seconds,
        )
        if expected_gpu_seconds
        else None
    )
    expected_samples = (
        walltime_seconds / expected_gpu_sample_interval_seconds * gpu_count
        if walltime_seconds is not None
        and expected_gpu_sample_interval_seconds is not None
        and gpu_count is not None
        else None
    )
    sample_count_coverage = (
        min(1.0, len(valid_gpu_samples) / expected_samples)
        if expected_samples
        else None
    )

    memory_events = [
        row for row in training if row.get("event_type") == "training_memory"
    ]
    parameter_events = [
        row for row in training if row.get("event_type") == "parameter_count"
    ]
    row = _base_row(lineage)
    row.update(
        {
            "event_type": "training_job_summary",
            "parsed_training_metric_rows": len(iterations),
            "expected_logged_metric_rows": expected_logged_metric_rows,
            "training_metric_coverage_fraction": metric_coverage,
            "training_metric_coverage_status": _coverage_status(metric_coverage),
            "first_iteration": min(
                (int(item["iteration"]) for item in iterations),
                default=None,
            ),
            "last_iteration": max(
                (int(item["iteration"]) for item in iterations),
                default=None,
            ),
            "mean_step_ms": _mean(step_ms),
            "median_step_ms": _median(step_ms),
            "mean_samples_per_second": _mean(samples_per_second),
            "mean_tokens_per_second": _mean(tokens_per_second),
            "median_tokens_per_second": _median(tokens_per_second),
            "throughput_semantics": (
                "Job-level observed throughput derived per logged iteration as "
                "global_batch_size * sequence_length / elapsed_time."
            ),
            "gpu_count": gpu_count,
            "mean_tokens_per_second_per_gpu": (
                _mean(tokens_per_second) / gpu_count
                if _mean(tokens_per_second) is not None and gpu_count is not None
                else None
            ),
            "mean_model_tflops_per_gpu": mean_tflops,
            "peak_tflops_per_gpu": peak_tflops_per_gpu,
            "mean_mfu_pct_of_configured_peak": mfu,
            "mfu_semantics": (
                "Mean Megatron model TFLOP/s/GPU divided by the caller-supplied "
                "hardware peak; unavailable when that configured denominator "
                "is absent."
            ),
            "max_training_allocated_gib": _maximum(
                item.get("max_allocated_gib") for item in memory_events
            ),
            "max_training_reserved_gib": _maximum(
                item.get("max_reserved_gib") for item in memory_events
            ),
            "parameter_count_observations": len(parameter_events),
            "parameter_counts": sorted(
                {
                    value
                    for item in parameter_events
                    if (value := _finite_int(item.get("parameter_count"))) is not None
                }
            ),
            "gpu_telemetry_rows": len(telemetry),
            "valid_gpu_telemetry_rows": len(valid_gpu_samples),
            "telemetry_gpu_uuids": sorted(
                {
                    str(item["gpu_uuid"])
                    for item in valid_gpu_samples
                    if item.get("gpu_uuid")
                }
            ),
            "mean_gpu_utilization_pct": _mean(
                item.get("gpu_utilization_pct") for item in valid_gpu_samples
            ),
            "max_gpu_utilization_pct": _maximum(
                item.get("gpu_utilization_pct") for item in valid_gpu_samples
            ),
            "mean_memory_used_mib": _mean(
                item.get("memory_used_mib") for item in valid_gpu_samples
            ),
            "max_memory_used_mib": _maximum(
                item.get("memory_used_mib") for item in valid_gpu_samples
            ),
            "max_memory_used_fraction": _maximum(
                item.get("memory_used_fraction") for item in valid_gpu_samples
            ),
            "mean_board_power_draw_w": _mean(
                item.get("board_power_draw_w") for item in valid_gpu_samples
            ),
            "max_board_power_draw_w": _maximum(
                item.get("board_power_draw_w") for item in valid_gpu_samples
            ),
            "walltime_seconds": walltime_seconds,
            "expected_gpu_seconds": expected_gpu_seconds,
            "gpu_sample_count_coverage_fraction": sample_count_coverage,
            "gpu_sample_count_coverage_status": _coverage_status(sample_count_coverage),
            "board_energy_coverage_fraction": energy_coverage,
            "board_energy_coverage_status": _coverage_status(energy_coverage),
            "lineage_records": _lineage_records([*training, *telemetry]),
        }
    )
    row.update(energy)
    return row


__all__ = [
    "SCHEMA_VERSION",
    "aggregate_job_training",
    "infer_training_log_lineage",
    "parse_megatron_training_line",
    "parse_megatron_training_lines",
    "parse_megatron_training_log",
    "parse_nvidia_smi_csv",
]
