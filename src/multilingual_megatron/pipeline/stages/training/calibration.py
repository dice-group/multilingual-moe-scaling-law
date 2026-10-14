from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...runtime import _deep_merge, _dump_json, _log_path_component
from .metrics import parse_training_log_metrics
from .planner import (
    TRAINING_CALIBRATION_NAME,
    TRAINING_CALIBRATION_TSV_NAME,
    TRAINING_CALIBRATIONS_NAME,
    TrainingRunPlan,
    build_training_run_plan,
    write_training_command_plan,
)
from .batch_policy import batch_policy_target_global_batch_size, resolve_batch_policy


CALIBRATION_MODES = {"calibrate_then_train", "calibrate_only"}
OOM_MARKERS = (
    "out of memory",
    "out_of_memory",
    "cuda oom",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cuda out of memory",
    "outofmemoryerror",
)


@dataclass(frozen=True, slots=True)
class TrainingCalibrationSettings:
    enabled: bool = False
    mode: str = "calibrate_then_train"
    candidate_batch_pairs: tuple[tuple[int, int], ...] = ()
    candidate_micro_batch_sizes: tuple[int, ...] = ()
    max_micro_batch_size: int = 16
    probe_train_iters: int = 50
    warmup_iters: int = 10
    memory_headroom_fraction: float = 0.10
    throughput_drop_threshold: float = 0.03
    gpu_memory_total_gb: float = 0.0
    peak_tflops_per_gpu: float = 0.0
    require_global_batch_divisibility: bool = True
    stop_after_first_oom: bool = True
    gpu_monitor_interval_seconds: int = 1
    profile_probes: bool = False


@dataclass(frozen=True, slots=True)
class TrainingCalibrationResult:
    run_id: str
    settings: TrainingCalibrationSettings
    candidates: list[dict[str, Any]]
    selected_candidate: dict[str, Any] | None
    selected_run_cfg: dict[str, Any] | None
    manifest_path: Path
    tsv_path: Path


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    out: list[int] = []
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    return tuple(out)


def _positive_int_pair_tuple(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        return ()
    out: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, dict):
            gbs = item.get("global_batch_size", item.get("gbs"))
            mbs = item.get("micro_batch_size", item.get("mbs"))
        elif isinstance(item, list) and len(item) >= 2:
            gbs, mbs = item[0], item[1]
        else:
            continue
        try:
            pair = (int(gbs), int(mbs))
        except (TypeError, ValueError):
            continue
        if pair[0] > 0 and pair[1] > 0 and pair not in out:
            out.append(pair)
    return tuple(out)


def resolve_training_calibration_settings(
    *,
    common: dict[str, Any],
    run_cfg: dict[str, Any],
) -> TrainingCalibrationSettings:
    raw = _deep_merge(common.get("calibration", {}), run_cfg.get("calibration", {}))
    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode", "calibrate_then_train")).strip() or "calibrate_then_train"
    if mode not in CALIBRATION_MODES:
        raise SystemExit(f"training.calibration.mode must be one of {sorted(CALIBRATION_MODES)}")
    probe_train_iters = _as_positive_int(raw.get("probe_train_iters", 50), 50)
    warmup_iters = _as_nonnegative_int(raw.get("warmup_iters", 10), 10)
    if warmup_iters >= probe_train_iters:
        warmup_iters = max(0, probe_train_iters - 1)
    return TrainingCalibrationSettings(
        enabled=_as_bool(raw.get("enabled", False), False),
        mode=mode,
        candidate_batch_pairs=_positive_int_pair_tuple(raw.get("candidate_batch_pairs", [])),
        candidate_micro_batch_sizes=_positive_int_tuple(raw.get("candidate_micro_batch_sizes", [])),
        max_micro_batch_size=_as_positive_int(raw.get("max_micro_batch_size", 16), 16),
        probe_train_iters=probe_train_iters,
        warmup_iters=warmup_iters,
        memory_headroom_fraction=max(0.0, min(0.95, _as_float(raw.get("memory_headroom_fraction", 0.10), 0.10))),
        throughput_drop_threshold=max(0.0, min(0.95, _as_float(raw.get("throughput_drop_threshold", 0.03), 0.03))),
        gpu_memory_total_gb=max(0.0, _as_float(raw.get("gpu_memory_total_gb", 0.0), 0.0)),
        peak_tflops_per_gpu=max(0.0, _as_float(raw.get("peak_tflops_per_gpu", 0.0), 0.0)),
        require_global_batch_divisibility=_as_bool(raw.get("require_global_batch_divisibility", True), True),
        stop_after_first_oom=_as_bool(raw.get("stop_after_first_oom", True), True),
        gpu_monitor_interval_seconds=_as_positive_int(raw.get("gpu_monitor_interval_seconds", 1), 1),
        profile_probes=_as_bool(raw.get("profile_probes", False), False),
    )


def calibration_enabled_for_any_run(common: dict[str, Any], runs: list[dict[str, Any]]) -> bool:
    return any(resolve_training_calibration_settings(common=common, run_cfg=run).enabled for run in runs)


def _merged_train_cfg(common: dict[str, Any], run_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = _deep_merge(common.get("train", {}), run_cfg.get("train", {}))
    return merged if isinstance(merged, dict) else {}


def _global_batch_size(train: dict[str, Any]) -> int:
    return _as_positive_int(train.get("global_batch_size", train.get("global_batch", 4)), 4)


def _micro_batch_size(train: dict[str, Any]) -> int:
    return _as_positive_int(train.get("micro_batch_size", train.get("micro_batch", 1)), 1)


def candidate_micro_batch_sizes(
    *,
    settings: TrainingCalibrationSettings,
    base_micro_batch_size: int,
    global_batch_size: int,
) -> list[int]:
    if settings.candidate_micro_batch_sizes:
        values = list(settings.candidate_micro_batch_sizes)
    else:
        values = []
        current = max(1, int(base_micro_batch_size))
        maximum = max(current, int(settings.max_micro_batch_size))
        while current <= maximum:
            values.append(current)
            current *= 2

    filtered: list[int] = []
    for value in sorted(set(values)):
        if value <= 0:
            continue
        if settings.require_global_batch_divisibility and global_batch_size > 0:
            if value > global_batch_size or global_batch_size % value != 0:
                continue
        filtered.append(value)
    if not filtered:
        raise SystemExit(
            "training.calibration produced no legal micro batch candidates; "
            "check candidate_micro_batch_sizes and global_batch_size"
        )
    return filtered


def candidate_batch_shapes(
    *,
    settings: TrainingCalibrationSettings,
    base_micro_batch_size: int,
    global_batch_size: int,
) -> list[tuple[int, int]]:
    if settings.candidate_batch_pairs:
        shapes = list(settings.candidate_batch_pairs)
        if settings.require_global_batch_divisibility:
            shapes = [(gbs, mbs) for gbs, mbs in shapes if mbs <= gbs and gbs % mbs == 0]
        if not shapes:
            raise SystemExit(
                "training.calibration produced no legal batch-pair candidates; "
                "check candidate_batch_pairs divisibility"
            )
        return shapes
    return [
        (global_batch_size, micro_batch_size)
        for micro_batch_size in candidate_micro_batch_sizes(
            settings=settings,
            base_micro_batch_size=base_micro_batch_size,
            global_batch_size=global_batch_size,
        )
    ]


def _candidate_id(global_batch_size: int, micro_batch_size: int) -> str:
    return f"gbs{int(global_batch_size)}_mbs{int(micro_batch_size)}"


def _probe_run_cfg(
    *,
    run_cfg: dict[str, Any],
    global_batch_size: int,
    micro_batch_size: int,
    settings: TrainingCalibrationSettings,
) -> dict[str, Any]:
    probe = dict(run_cfg)
    train = _deep_merge(probe.get("train", {}), {})
    train["global_batch_size"] = int(global_batch_size)
    train["global_batch"] = int(global_batch_size)
    train["micro_batch_size"] = int(micro_batch_size)
    train["micro_batch"] = int(micro_batch_size)
    train["train_iters"] = int(settings.probe_train_iters)
    train["max_steps"] = int(settings.probe_train_iters)
    train["token_budget_tokens"] = ""
    train["token_budget"] = ""
    train["batch_policy"] = {"enabled": False}
    train["save_interval"] = int(settings.probe_train_iters) + 1
    train["save_interval_iters"] = int(settings.probe_train_iters) + 1
    probe["train"] = train

    if not settings.profile_probes:
        launch = _deep_merge(probe.get("launch", {}), {})
        launch["nsys_profile"] = False
        probe["launch"] = launch
        overrides = list(probe.get("overrides", [])) if isinstance(probe.get("overrides", []), list) else []
        overrides.extend(
            [
                "profiling.use_nsys_profiler=false",
                "profiling.use_pytorch_profiler=false",
            ]
        )
        probe["overrides"] = overrides

    eval_cfg = _deep_merge(probe.get("eval", {}), {})
    eval_cfg["eval_interval"] = int(settings.probe_train_iters)
    eval_cfg["eval_iters"] = 0
    probe["eval"] = eval_cfg
    return probe


def build_calibration_probe_plan(
    ctx: Any,
    *,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    base_plan: TrainingRunPlan,
    settings: TrainingCalibrationSettings,
    global_batch_size: int,
    micro_batch_size: int,
) -> TrainingRunPlan:
    candidate_id = _candidate_id(global_batch_size, micro_batch_size)
    output_dir = base_plan.output_dir / "calibration" / candidate_id
    log_dir = ctx.logs_root / "train_runs" / _log_path_component(base_plan.run_id)
    log_path = log_dir / f"{base_plan.run_name}.calibration.{candidate_id}.train.log"
    probe_plan = build_training_run_plan(
        ctx,
        run=_probe_run_cfg(
            run_cfg=run_cfg,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            settings=settings,
        ),
        common=common,
        output_dir=output_dir,
        log_path=log_path,
    )
    env = dict(probe_plan.env)
    env.setdefault("BRIDGE_LOG_THROUGHPUT", "1")
    env.setdefault("PROOF_GPU_MONITOR_INTERVAL_SECONDS", str(settings.gpu_monitor_interval_seconds))
    return replace(probe_plan, env=env)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path, limit_bytes: int = 250_000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > limit_bytes:
        return text[-limit_bytes:]
    return text


def _is_oom(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _memory_headroom_fraction(metrics: dict[str, Any], settings: TrainingCalibrationSettings) -> float | None:
    max_vram_pct = _float_or_none(metrics.get("max_vram_used_pct"))
    if max_vram_pct is not None:
        return max(0.0, 1.0 - (max_vram_pct / 100.0))
    if settings.gpu_memory_total_gb > 0:
        used_gb = _float_or_none(metrics.get("max_reserved_gb")) or _float_or_none(metrics.get("max_allocated_gb"))
        if used_gb is not None:
            return max(0.0, 1.0 - (used_gb / float(settings.gpu_memory_total_gb)))
    return None


def _candidate_record(
    *,
    plan: TrainingRunPlan,
    micro_batch_size: int,
    global_batch_size: int,
    settings: TrainingCalibrationSettings,
    error: str,
    dry_run: bool,
) -> dict[str, Any]:
    metrics = parse_training_log_metrics(
        plan.log_path,
        seq_length=int(plan.metadata.get("seq_length", 0) or 0),
        warmup_iters=settings.warmup_iters,
        peak_tflops_per_gpu=settings.peak_tflops_per_gpu,
    )
    results = _read_json(plan.output_dir / "train_results.json")
    log_text = _read_text(plan.log_path)
    error_text = "\n".join(str(item) for item in (error, results.get("error", ""), log_text) if str(item).strip())
    complete = (
        str(results.get("status", "")).strip().lower() in {"complete", "ok"}
        and (plan.output_dir / ".training_complete").exists()
    )
    oom = _is_oom(error_text)
    headroom = _memory_headroom_fraction(metrics, settings)
    metric_value = _float_or_none(metrics.get("mean_mfu_pct_of_peak"))
    metric_name = "mean_mfu_pct_of_peak" if metric_value is not None else "median_tokens_per_second"
    if metric_value is None:
        metric_value = _float_or_none(metrics.get("median_tokens_per_second"))

    if dry_run:
        candidate_status = "dry_run"
    elif oom:
        candidate_status = "oom"
    elif not complete:
        candidate_status = "failed"
    elif metric_value is None:
        candidate_status = "metrics_missing"
    elif headroom is None:
        candidate_status = "memory_unknown"
    elif headroom < settings.memory_headroom_fraction:
        candidate_status = "insufficient_headroom"
    else:
        candidate_status = "selectable"

    record: dict[str, Any] = {
        "candidate_id": _candidate_id(global_batch_size, micro_batch_size),
        "run_id": plan.run_id,
        "status": candidate_status,
        "selectable": candidate_status == "selectable",
        "micro_batch_size": int(micro_batch_size),
        "global_batch_size": int(global_batch_size),
        "output_dir": str(plan.output_dir.resolve()),
        "log_path": str(plan.log_path.resolve()),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "memory_headroom_fraction": headroom,
        "required_memory_headroom_fraction": settings.memory_headroom_fraction,
        "error": error,
    }
    record.update(metrics)
    return record


def select_calibration_candidate(
    candidates: list[dict[str, Any]],
    *,
    throughput_drop_threshold: float,
) -> dict[str, Any] | None:
    selectable = [row for row in candidates if row.get("selectable")]
    if not selectable:
        return None
    metric_name = (
        "mean_mfu_pct_of_peak"
        if all(_float_or_none(row.get("mean_mfu_pct_of_peak")) is not None for row in selectable)
        else "median_tokens_per_second"
    )
    for row in selectable:
        row["selection_metric_name"] = metric_name
        row["selection_metric_value"] = _float_or_none(row.get(metric_name)) or 0.0

    largest = max(selectable, key=lambda row: int(row.get("micro_batch_size", 0) or 0))
    best = max(selectable, key=lambda row: float(row.get("selection_metric_value", 0.0) or 0.0))
    largest_metric = float(largest.get("selection_metric_value", 0.0) or 0.0)
    best_metric = float(best.get("selection_metric_value", 0.0) or 0.0)
    selected = largest
    rationale = "largest_stable"
    if (
        int(best.get("micro_batch_size", 0) or 0) < int(largest.get("micro_batch_size", 0) or 0)
        and largest_metric > 0
        and best_metric >= largest_metric * (1.0 + float(throughput_drop_threshold))
    ):
        selected = best
        rationale = "smaller_faster_by_threshold"
    out = dict(selected)
    out["selection_rationale"] = rationale
    return out


def _selected_run_cfg(
    *,
    run_cfg: dict[str, Any],
    selected: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    out = dict(run_cfg)
    train = _deep_merge(out.get("train", {}), {})
    train["global_batch_size"] = int(selected["global_batch_size"])
    train["global_batch"] = int(selected["global_batch_size"])
    train["micro_batch_size"] = int(selected["micro_batch_size"])
    train["micro_batch"] = int(selected["micro_batch_size"])
    out["train"] = train
    out["calibration_selection"] = {
        "manifest": str(manifest_path.resolve()),
        "candidate_id": selected.get("candidate_id"),
        "micro_batch_size": int(selected["micro_batch_size"]),
        "global_batch_size": int(selected["global_batch_size"]),
        "metric_name": selected.get("selection_metric_name", selected.get("metric_name")),
        "metric_value": selected.get("selection_metric_value", selected.get("metric_value")),
        "memory_headroom_fraction": selected.get("memory_headroom_fraction"),
        "selection_rationale": selected.get("selection_rationale"),
    }
    batch_policy = resolve_batch_policy(train, seq_length=int(train.get("seq_length", 2048) or 2048))
    if batch_policy is not None:
        out["calibration_selection"]["batch_policy"] = {
            "mode": batch_policy["mode"],
            "target_global_batch_size": int(batch_policy["target_global_batch_size"]),
            "mbs_policy": batch_policy["mbs_policy"],
        }
    return out


def _write_calibration_tsv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "status",
        "selectable",
        "micro_batch_size",
        "global_batch_size",
        "selection_metric_name",
        "selection_metric_value",
        "median_tokens_per_second",
        "mean_mfu_pct_of_peak",
        "max_allocated_gb",
        "max_reserved_gb",
        "max_vram_used_pct",
        "memory_headroom_fraction",
        "required_memory_headroom_fraction",
        "log_path",
        "output_dir",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_run_calibration_manifest(
    *,
    base_plan: TrainingRunPlan,
    settings: TrainingCalibrationSettings,
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> tuple[Path, Path]:
    manifest_path = base_plan.output_dir / TRAINING_CALIBRATION_NAME
    tsv_path = base_plan.output_dir / TRAINING_CALIBRATION_TSV_NAME
    _dump_json(
        manifest_path,
        {
            "schema_version": 1,
            "stage": "train",
            "kind": "training_calibration",
            "run_id": base_plan.run_id,
            "settings": {
                "mode": settings.mode,
                "candidate_batch_pairs": [
                    {"global_batch_size": gbs, "micro_batch_size": mbs}
                    for gbs, mbs in settings.candidate_batch_pairs
                ],
                "probe_train_iters": settings.probe_train_iters,
                "warmup_iters": settings.warmup_iters,
                "memory_headroom_fraction": settings.memory_headroom_fraction,
                "throughput_drop_threshold": settings.throughput_drop_threshold,
                "gpu_memory_total_gb": settings.gpu_memory_total_gb,
                "peak_tflops_per_gpu": settings.peak_tflops_per_gpu,
                "profile_probes": settings.profile_probes,
            },
            "candidates": candidates,
            "selected_candidate": selected,
        },
    )
    _write_calibration_tsv(tsv_path, candidates)
    return manifest_path, tsv_path


def run_training_calibration(
    ctx: Any,
    *,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    base_plan: TrainingRunPlan,
) -> TrainingCalibrationResult:
    settings = resolve_training_calibration_settings(common=common, run_cfg=run_cfg)
    train = _merged_train_cfg(common, run_cfg)
    seq_length = int(run_cfg.get("seq_length", common.get("seq_length", train.get("seq_length", 2048))) or 2048)
    batch_policy = resolve_batch_policy(train, seq_length=seq_length)
    global_batch_size = batch_policy_target_global_batch_size(batch_policy, _global_batch_size(train))
    if batch_policy is not None and settings.candidate_batch_pairs:
        illegal_pairs = [(gbs, mbs) for gbs, mbs in settings.candidate_batch_pairs if int(gbs) != int(global_batch_size)]
        if illegal_pairs:
            raise SystemExit(
                "training.calibration.candidate_batch_pairs cannot vary global_batch_size when "
                "training.train.batch_policy.enabled=true; use candidate_micro_batch_sizes or pairs with "
                f"global_batch_size={global_batch_size}"
            )
    base_micro_batch_size = _micro_batch_size(train)
    candidates = candidate_batch_shapes(
        settings=settings,
        base_micro_batch_size=base_micro_batch_size,
        global_batch_size=global_batch_size,
    )

    records: list[dict[str, Any]] = []
    for candidate_global_batch_size, micro_batch_size in candidates:
        probe_plan = build_calibration_probe_plan(
            ctx,
            run_cfg=run_cfg,
            common=common,
            base_plan=base_plan,
            settings=settings,
            global_batch_size=candidate_global_batch_size,
            micro_batch_size=micro_batch_size,
        )
        probe_plan.output_dir.mkdir(parents=True, exist_ok=True)
        write_training_command_plan(probe_plan)
        error = ""
        if not ctx.args.dry_run:
            try:
                ctx.executor.run(probe_plan.cmd, probe_plan.env, False, probe_plan.log_path)
            except Exception as exc:
                error = repr(exc)
        record = _candidate_record(
            plan=probe_plan,
            micro_batch_size=micro_batch_size,
            global_batch_size=candidate_global_batch_size,
            settings=settings,
            error=error,
            dry_run=bool(ctx.args.dry_run),
        )
        records.append(record)
        if settings.stop_after_first_oom and record["status"] == "oom":
            break

    selected = None if ctx.args.dry_run else select_calibration_candidate(
        records,
        throughput_drop_threshold=settings.throughput_drop_threshold,
    )
    manifest_path, tsv_path = _write_run_calibration_manifest(
        base_plan=base_plan,
        settings=settings,
        candidates=records,
        selected=selected,
    )
    selected_run_cfg = None
    if selected is not None and settings.mode == "calibrate_then_train":
        selected_run_cfg = _selected_run_cfg(run_cfg=run_cfg, selected=selected, manifest_path=manifest_path)
    return TrainingCalibrationResult(
        run_id=base_plan.run_id,
        settings=settings,
        candidates=records,
        selected_candidate=selected,
        selected_run_cfg=selected_run_cfg,
        manifest_path=manifest_path,
        tsv_path=tsv_path,
    )


def write_training_calibrations_manifest(*, train_root: Path, results: list[TrainingCalibrationResult]) -> Path:
    path = train_root / TRAINING_CALIBRATIONS_NAME
    _dump_json(
        path,
        {
            "schema_version": 1,
            "stage": "train",
            "kind": "training_calibrations",
            "train_root": str(train_root.resolve()),
            "runs": {
                result.run_id: {
                    "mode": result.settings.mode,
                    "manifest": str(result.manifest_path.resolve()),
                    "tsv": str(result.tsv_path.resolve()),
                    "selected_candidate": result.selected_candidate,
                }
                for result in results
            },
        },
    )
    return path
