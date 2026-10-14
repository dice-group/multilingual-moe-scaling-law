"""Parse fail-closed training health and bind it to segment-plan evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .common import (
    GRAD_NORM_RE,
    H100_PEAK_TFLOPS_BF16,
    LEARNING_RATE_RE,
    LOSS_RE,
    NAN_ITER_RE,
    OPTIONAL_DEPENDENCY_PREFIX,
    SKIPPED_RE,
    WorkflowError,
    _file_sha256,
    _load_object,
)


def _resolve_segment_plan_path(
    *, training_manifest_path: Path, metadata: Mapping[str, Any]
) -> Path:
    """Resolve the segment plan even when the summarized manifest omitted it.

    ``segment_runner`` writes ``segment_plan.json`` next to
    ``training_run_manifest.json``.  Older summary metadata dropped the path,
    so branch planning deliberately verifies this adjacent immutable artifact
    rather than trusting an empty metadata field.
    """

    raw = str(metadata.get("segment_plan_path", "")).strip()
    path = (
        Path(raw)
        if raw
        else training_manifest_path.resolve().parent / "segment_plan.json"
    )
    if not path.is_absolute():
        path = training_manifest_path.resolve().parent / path
    return path.resolve()


def _parse_sentinel_health(
    log_path: Path,
    *,
    require_moe: bool = False,
    expected_learning_rate: float | None = None,
    allow_linear_warmup: bool = False,
    throughput_window_size: int = 12,
) -> dict[str, Any]:
    """Parse the real Megatron log and return a fail-closed health record."""

    from multilingual_megatron.pipeline.stages.training.metrics import (
        parse_training_log_metrics,
    )

    errors: list[str] = []
    if not log_path.is_file():
        return {
            "schema_version": 1,
            "status": "failed",
            "errors": [f"missing sentinel training log: {log_path}"],
            "log_path": str(log_path),
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    losses: list[float] = []
    for raw in LOSS_RE.findall(text):
        try:
            losses.append(float(raw))
        except ValueError:
            errors.append(f"unparseable lm loss {raw!r}")
    if not losses:
        errors.append("no lm loss observations")
    elif any(not math.isfinite(value) for value in losses):
        errors.append("non-finite lm loss observation")

    gradient_norms: list[float] = []
    for raw in GRAD_NORM_RE.findall(text):
        try:
            gradient_norms.append(float(raw))
        except ValueError:
            errors.append(f"unparseable gradient norm {raw!r}")
    if not gradient_norms:
        errors.append("no gradient-norm observations")
    elif any(not math.isfinite(value) for value in gradient_norms):
        errors.append("non-finite gradient-norm observation")

    learning_rates: list[float] = []
    for raw in LEARNING_RATE_RE.findall(text):
        try:
            learning_rates.append(float(raw))
        except ValueError:
            errors.append(f"unparseable learning rate {raw!r}")
    if not learning_rates:
        errors.append("no learning-rate observations")
    elif any(not math.isfinite(value) for value in learning_rates):
        errors.append("non-finite learning-rate observation")
    if expected_learning_rate is not None and learning_rates:
        if not math.isfinite(expected_learning_rate) or expected_learning_rate <= 0:
            errors.append("expected constant learning rate is invalid")
        elif allow_linear_warmup:
            if any(
                value <= 0
                or (
                    value > expected_learning_rate
                    and not math.isclose(
                        value,
                        expected_learning_rate,
                        rel_tol=5e-6,
                        abs_tol=1e-12,
                    )
                )
                for value in learning_rates
            ):
                errors.append("logged warmup learning rates exceed the fixed peak")
            if any(
                current > following
                and not math.isclose(
                    current,
                    following,
                    rel_tol=5e-6,
                    abs_tol=1e-12,
                )
                for current, following in zip(
                    learning_rates, learning_rates[1:], strict=False
                )
            ):
                errors.append("logged fixed-LR warmup is not monotonic")
            if not math.isclose(
                learning_rates[-1],
                expected_learning_rate,
                rel_tol=5e-6,
                abs_tol=1e-12,
            ):
                errors.append(
                    "fixed peak LR was not reached before the stable window ended"
                )
            plateau = learning_rates[-3:]
            if len(plateau) < 3 or any(
                not math.isclose(
                    value,
                    expected_learning_rate,
                    rel_tol=5e-6,
                    abs_tol=1e-12,
                )
                for value in plateau
            ):
                errors.append(
                    "fixed peak LR was not stable for three logged observations"
                )
        elif any(
            not math.isclose(
                value,
                expected_learning_rate,
                rel_tol=5e-6,
                abs_tol=1e-12,
            )
            for value in learning_rates
        ):
            errors.append(
                "logged learning rate differs from the fixed constant peak: "
                f"expected {expected_learning_rate}, observed {learning_rates}"
            )

    dependency_records: list[dict[str, Any]] = []
    for line in text.splitlines():
        marker = line.find(OPTIONAL_DEPENDENCY_PREFIX)
        if marker < 0:
            continue
        raw = line[marker + len(OPTIONAL_DEPENDENCY_PREFIX) :].strip()
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid optional-dependency provenance JSON: {exc}")
            continue
        if isinstance(record, dict):
            dependency_records.append(record)
        else:
            errors.append("optional-dependency provenance is not an object")
    if not dependency_records:
        errors.append("no bridge optional-dependency provenance observation")
    for record in dependency_records:
        modelopt_file = Path(str(record.get("modelopt_file", "")))
        modelopt_mode = str(record.get("modelopt_mode", ""))
        nvrx_mode = str(record.get("nvrx_mode", ""))
        if record.get("compatibility_mode_enabled") is not True:
            errors.append("Bridge compatibility mode is not enabled")
        if record.get("real_modelopt_required") is not True:
            errors.append("real ModelOpt is not fail-closed required")
        if not modelopt_mode.startswith("real_"):
            errors.append(f"ModelOpt is not real: {modelopt_mode!r}")
        if not modelopt_file.is_absolute() or not modelopt_file.is_file():
            errors.append(f"ModelOpt file evidence is invalid: {modelopt_file}")
        real_inprocess = record.get("nvrx_inprocess_available") is True
        incompatible_sync = nvrx_mode.startswith(
            "real_inprocess_async_disabled_incompatible_"
        )
        real_new_enough = nvrx_mode.startswith("real_") and not incompatible_sync
        if not real_inprocess or not (incompatible_sync or real_new_enough):
            errors.append(f"NVRx real inprocess evidence is invalid: {nvrx_mode!r}")
        if (
            incompatible_sync
            and record.get("nvrx_async_checkpoint_available") is not False
        ):
            errors.append("incompatible NVRx unexpectedly exposes async checkpointing")
        for key in (
            "checkpoint_async_save_required",
            "nvrx_inprocess_restart_enabled",
            "nvrx_straggler_enabled",
        ):
            if record.get(key) is not False:
                errors.append(f"optional dependency runtime flag must be false: {key}")

    skipped = [int(value) for value in SKIPPED_RE.findall(text)]
    nan_iterations = [int(value) for value in NAN_ITER_RE.findall(text)]
    if not skipped:
        errors.append("no skipped-iteration counters")
    elif max(skipped) != 0:
        errors.append(f"skipped iterations observed: {max(skipped)}")
    if not nan_iterations:
        errors.append("no nan-iteration counters")
    elif max(nan_iterations) != 0:
        errors.append(f"nan iterations observed: {max(nan_iterations)}")

    metrics = parse_training_log_metrics(
        log_path,
        seq_length=2048,
        warmup_iters=0,
        peak_tflops_per_gpu=H100_PEAK_TFLOPS_BF16,
        window_size=throughput_window_size,
    )
    for key in (
        "median_tokens_per_second",
        "mean_model_tflops_per_gpu",
        "mean_mfu_pct_of_peak",
    ):
        try:
            value = float(metrics.get(key, ""))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            errors.append(f"missing or non-positive measured {key}")
    if int(metrics.get("steps_counted", 0) or 0) <= 0:
        errors.append("no timed training iterations")
    if require_moe:
        if int(metrics.get("drop_metric_lines", 0) or 0) <= 0:
            errors.append("no MoE dropped-assignment health observations")
        try:
            dropped = float(metrics.get("max_dropped_assignment_rate", math.nan))
            zero_kept = float(
                metrics.get("max_tokens_with_zero_kept_experts", math.nan)
            )
        except (TypeError, ValueError):
            dropped = math.nan
            zero_kept = math.nan
        if not math.isfinite(dropped) or dropped != 0.0:
            errors.append(f"MoE dropped assignment rate is not exact zero: {dropped!r}")
        if not math.isfinite(zero_kept) or zero_kept != 0.0:
            errors.append(f"MoE zero-kept token count is not exact zero: {zero_kept!r}")
    try:
        tflops = float(metrics.get("mean_model_tflops_per_gpu", ""))
        mfu = float(metrics.get("mean_mfu_pct_of_peak", ""))
        expected_mfu = 100.0 * tflops / H100_PEAK_TFLOPS_BF16
        if not math.isclose(mfu, expected_mfu, rel_tol=0.0, abs_tol=0.02):
            errors.append("MFU is inconsistent with measured model TFLOP/s")
    except (TypeError, ValueError):
        pass
    return {
        "schema_version": 1,
        "status": "failed" if errors else "ok",
        "errors": errors,
        "log_path": str(log_path.resolve()),
        "log_sha256": _file_sha256(log_path),
        "loss_observations": len(losses),
        "last_finite_lm_loss": losses[-1]
        if losses and math.isfinite(losses[-1])
        else None,
        "gradient_norm_observations": len(gradient_norms),
        "last_finite_gradient_norm": (
            gradient_norms[-1]
            if gradient_norms and math.isfinite(gradient_norms[-1])
            else None
        ),
        "learning_rate_observations": len(learning_rates),
        "learning_rate_min": min(learning_rates) if learning_rates else None,
        "learning_rate_max": max(learning_rates) if learning_rates else None,
        "expected_learning_rate": expected_learning_rate,
        "optional_dependencies": dependency_records[-1] if dependency_records else {},
        "max_skipped_iterations": max(skipped) if skipped else None,
        "max_nan_iterations": max(nan_iterations) if nan_iterations else None,
        "metrics": metrics,
        "moe_health_required": bool(require_moe),
    }


def _training_health_for_segment_plan(
    segment_plan_path: Path, *, require_moe: bool
) -> dict[str, Any]:
    plan = _load_object(segment_plan_path)
    raw_segments = plan.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise WorkflowError(
            f"segment plan has no segments for health validation: {segment_plan_path}"
        )
    segments: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            errors.append(f"segment {index} is not an object")
            continue
        health = _parse_sentinel_health(
            Path(str(segment.get("log_path", ""))),
            require_moe=require_moe,
        )
        health["segment_index"] = int(segment.get("segment_index", index))
        segments.append(health)
        if health.get("status") != "ok":
            errors.append(f"segment {index}: {health.get('errors')}")
    result = {
        "schema_version": 1,
        "status": "failed" if errors else "ok",
        "errors": errors,
        "segment_plan": str(segment_plan_path.resolve()),
        "segment_plan_sha256": _file_sha256(segment_plan_path),
        "moe_health_required": bool(require_moe),
        "segments": segments,
    }
    if errors:
        raise WorkflowError(
            f"training health validation failed for {segment_plan_path}: "
            + "; ".join(errors)
        )
    return result
