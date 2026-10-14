from __future__ import annotations

import csv
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any


ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+).*?"
    r"elapsed time per iteration \(ms\): (?P<ms>[0-9.]+).*?"
    r"global batch size:\s*(?P<gbs>\d+)"
)
MEM_RE = re.compile(
    r"mem-max-allocated-gigabytes:\s*(?P<allocated>[0-9.]+).*?"
    r"mem-max-reserved-gigabytes:\s*(?P<reserved>[0-9.]+)"
)
MFU_RE = re.compile(r"GPU utilization:\s*(?P<mfu>[0-9.]+)MODEL_TFLOP/s/GPU")
ITER_MFU_RE = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<mfu>[0-9.]+)")
DROP_LINE_RE = re.compile(
    r"\[moe_token_drop\].*?"
    r"dropped_assignment_rate:\s*(?P<drop_rate>[0-9.]+).*?"
    r"tokens_with_zero_kept_experts:\s*(?P<zero_kept>[0-9.]+)",
    re.IGNORECASE,
)
CONSUMED_SAMPLES_RE = re.compile(r"consumed samples:\s*(?P<value>\d+)")
LM_LOSS_RE = re.compile(r"lm loss:\s*(?P<value>[^|]+)")
GRAD_NORM_RE = re.compile(r"grad norm:\s*(?P<value>[^|]+)")
SKIPPED_RE = re.compile(r"number of skipped iterations:\s*(?P<value>\d+)")
NAN_ITER_RE = re.compile(r"number of nan iterations:\s*(?P<value>\d+)")


def gpu_csv_for_log(path: Path) -> Path:
    if path.name.endswith(".train.slurm.log"):
        return path.with_name(path.name.removesuffix(".train.slurm.log") + ".train.gpu.csv")
    if path.name.endswith(".slurm.log"):
        return path.with_name(path.name.removesuffix(".slurm.log") + ".gpu.csv")
    return path.with_suffix(".gpu.csv")


def gpu_metrics_for_log(path: Path) -> dict[str, float | str]:
    gpu_csv = gpu_csv_for_log(path)
    if not gpu_csv.exists():
        controller_csv = os.environ.get("PROOF_GPU_MONITOR_CSV", "").strip()
        if controller_csv:
            gpu_csv = Path(controller_csv)
    if not gpu_csv.exists():
        return {}

    util_values: list[float] = []
    vram_values: list[float] = []
    power_values: list[float] = []
    with gpu_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                util = float(str(row.get("utilization_gpu_pct", "")).strip())
                memory_used = float(str(row.get("memory_used_mib", "")).strip())
                memory_total = float(str(row.get("memory_total_mib", "")).strip())
            except ValueError:
                continue
            util_values.append(util)
            if memory_total > 0:
                vram_values.append(100.0 * memory_used / memory_total)
            try:
                power_values.append(float(str(row.get("power_draw_w", "")).strip()))
            except ValueError:
                pass

    metrics: dict[str, float | str] = {"gpu_monitor_csv": str(gpu_csv)}
    if util_values:
        metrics["mean_gpu_util_pct"] = round(statistics.mean(util_values), 2)
        metrics["max_gpu_util_pct"] = round(max(util_values), 2)
    if vram_values:
        metrics["mean_vram_used_pct"] = round(statistics.mean(vram_values), 2)
        metrics["max_vram_used_pct"] = round(max(vram_values), 2)
    if power_values:
        metrics["mean_power_w"] = round(statistics.mean(power_values), 2)
        metrics["max_power_w"] = round(max(power_values), 2)
    return metrics


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_training_log_metrics(
    path: Path,
    *,
    seq_length: int,
    warmup_iters: int,
    peak_tflops_per_gpu: float = 0.0,
    window_size: int = 0,
) -> dict[str, Any]:
    values: list[float] = []
    batch_sizes: list[int] = []
    max_allocated_gb: float | None = None
    max_reserved_gb: float | None = None
    mfu_values: list[float] = []
    iteration = 0
    pending_mfu: float | None = None
    max_drop_rate = 0.0
    max_zero_kept = 0.0
    drop_line_count = 0
    iteration_lines = 0
    last_iteration = 0
    reported_total_iterations = 0
    last_consumed_samples = 0
    lm_loss_lines = 0
    grad_norm_lines = 0
    nonfinite_lm_loss_count = 0
    nonfinite_grad_norm_count = 0
    max_skipped_iterations = 0
    max_nan_iterations = 0

    if not path.exists():
        return {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        drop_match = DROP_LINE_RE.search(line)
        if drop_match:
            drop_line_count += 1
            max_drop_rate = max(max_drop_rate, float(drop_match.group("drop_rate")))
            max_zero_kept = max(max_zero_kept, float(drop_match.group("zero_kept")))
        mfu_match = MFU_RE.search(line)
        if mfu_match:
            pending_mfu = float(mfu_match.group("mfu"))
        match = ITER_RE.search(line)
        if match:
            iteration = int(match.group("iter"))
            iteration_lines += 1
            reported_total_iterations = max(reported_total_iterations, int(match.group("total")))
            consumed_match = CONSUMED_SAMPLES_RE.search(line)
            if iteration >= last_iteration:
                last_iteration = iteration
                if consumed_match:
                    last_consumed_samples = int(consumed_match.group("value"))
            loss_match = LM_LOSS_RE.search(line)
            if loss_match:
                lm_loss_lines += 1
                try:
                    if not math.isfinite(float(loss_match.group("value").strip())):
                        nonfinite_lm_loss_count += 1
                except ValueError:
                    nonfinite_lm_loss_count += 1
            grad_match = GRAD_NORM_RE.search(line)
            if grad_match:
                grad_norm_lines += 1
                try:
                    if not math.isfinite(float(grad_match.group("value").strip())):
                        nonfinite_grad_norm_count += 1
                except ValueError:
                    nonfinite_grad_norm_count += 1
            skipped_match = SKIPPED_RE.search(line)
            if skipped_match:
                max_skipped_iterations = max(max_skipped_iterations, int(skipped_match.group("value")))
            nan_match = NAN_ITER_RE.search(line)
            if nan_match:
                max_nan_iterations = max(max_nan_iterations, int(nan_match.group("value")))
            if iteration > warmup_iters:
                values.append(float(match.group("ms")))
                batch_sizes.append(int(match.group("gbs")))
                iter_mfu_match = ITER_MFU_RE.search(line)
                if iter_mfu_match:
                    mfu_values.append(float(iter_mfu_match.group("mfu")))
                elif pending_mfu is not None:
                    mfu_values.append(pending_mfu)
            pending_mfu = None
        mem_match = MEM_RE.search(line)
        if mem_match:
            max_allocated_gb = max(
                max_allocated_gb or 0.0,
                float(mem_match.group("allocated")),
            )
            max_reserved_gb = max(
                max_reserved_gb or 0.0,
                float(mem_match.group("reserved")),
            )

    metrics: dict[str, Any] = {}
    if window_size > 0 and len(values) > window_size:
        values = values[-window_size:]
        batch_sizes = batch_sizes[-window_size:]
        if mfu_values:
            mfu_values = mfu_values[-window_size:]
    if values:
        mean_ms = statistics.mean(values)
        median_ms = statistics.median(values)
        p90_ms = _percentile(values, 0.90)
        p99_ms = _percentile(values, 0.99)
        step_std_ms = statistics.pstdev(values) if len(values) > 1 else 0.0
        step_cv_pct = (100.0 * step_std_ms / mean_ms) if mean_ms > 0 else 0.0
        global_batch_size = batch_sizes[-1]
        mean_tokens_per_second = global_batch_size * seq_length / (mean_ms / 1000.0)
        median_tokens_per_second = global_batch_size * seq_length / (median_ms / 1000.0)
        mean_mfu = statistics.mean(mfu_values) if mfu_values else None
        mean_mfu_pct = (
            (100.0 * mean_mfu / peak_tflops_per_gpu)
            if mean_mfu is not None and peak_tflops_per_gpu > 0
            else None
        )
        metrics.update(
            {
                "steps_counted": len(values),
                "seq_length": int(seq_length),
                "global_batch_size": int(global_batch_size),
                "mean_step_ms": round(mean_ms, 3),
                "median_step_ms": round(median_ms, 3),
                "p90_step_ms": round(p90_ms, 3),
                "p99_step_ms": round(p99_ms, 3),
                "step_std_ms": round(step_std_ms, 3),
                "step_time_cv_pct": round(step_cv_pct, 3),
                "p99_over_median_step_time": round(p99_ms / median_ms, 6) if median_ms > 0 else "",
                "mean_tokens_per_second": round(mean_tokens_per_second, 2),
                "median_tokens_per_second": round(median_tokens_per_second, 2),
                "mean_model_tflops_per_gpu": round(mean_mfu, 2) if mean_mfu is not None else "",
                "mean_mfu_pct_of_peak": round(mean_mfu_pct, 2) if mean_mfu_pct is not None else "",
            }
        )
    if iteration_lines:
        metrics.update(
            {
                "iteration_lines": iteration_lines,
                "last_iteration": last_iteration,
                "reported_total_iterations": reported_total_iterations,
                "last_consumed_samples": last_consumed_samples,
                "observed_tokens": last_consumed_samples * int(seq_length),
                "lm_loss_lines": lm_loss_lines,
                "grad_norm_lines": grad_norm_lines,
                "nonfinite_lm_loss_count": nonfinite_lm_loss_count,
                "nonfinite_grad_norm_count": nonfinite_grad_norm_count,
                "max_skipped_iterations": max_skipped_iterations,
                "max_nan_iterations": max_nan_iterations,
                "finite_loss_and_gradients": (
                    lm_loss_lines > 0
                    and grad_norm_lines > 0
                    and nonfinite_lm_loss_count == 0
                    and nonfinite_grad_norm_count == 0
                ),
            }
        )
    if max_allocated_gb is not None:
        metrics["max_allocated_gb"] = round(max_allocated_gb, 3)
    if max_reserved_gb is not None:
        metrics["max_reserved_gb"] = round(max_reserved_gb, 3)
    if drop_line_count:
        metrics["drop_metric_lines"] = drop_line_count
        metrics["max_dropped_assignment_rate"] = round(max_drop_rate, 6)
        metrics["max_tokens_with_zero_kept_experts"] = round(max_zero_kept, 3)
    metrics.update(gpu_metrics_for_log(path))
    return metrics
