#!/usr/bin/env python3
"""Summarize Nsight Systems CUDA kernel CSVs for 600M Switch-MoE profiling.

Expected input is a CSV exported by:

    nsys stats --report cuda_gpu_kern_sum --format csv --output kernels report.nsys-rep

The parser is intentionally tolerant of Nsight column-name drift. It classifies
kernels into coarse training phases so the MFU sweep can explain where time is
going, not only how fast the full step is.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "te_gemm",
        (
            "nvjet_",
            "cublas",
            "gemm",
            "matmul",
        ),
    ),
    (
        "attention",
        (
            "attention",
            "flash",
            "fmha",
            "fused_attn",
            "scaled_dot",
            "softmax",
            "qkv",
        ),
    ),
    (
        "router",
        (
            "router",
            "topk",
            "top_k",
            "moe_aux",
            "z_loss",
            "load_balanc",
        ),
    ),
    (
        "dispatch_permute",
        (
            "permute",
            "unpermute",
            "scatter",
            "gather",
            "index_select",
            "index_add",
            "sort",
            "histogram",
            "cumsum",
            "prefix",
            "alltoall",
            "allgather",
        ),
    ),
    (
        "expert_gemm",
        (
            "grouped_gemm",
            "moe_gemm",
            "expert",
            "gemm_grouped",
        ),
    ),
    (
        "dense_gemm",
        (
            "linear",
        ),
    ),
    (
        "fp8_cast_transpose",
        (
            "cast_transpose",
            "fp8",
            "cast_from_fp8",
            "cast_to_fp8",
        ),
    ),
    (
        "torch_elementwise",
        (
            "elementwise_kernel",
            "vectorized_elementwise",
            "unrolled_elementwise",
            "reduce_kernel",
            "fillfunctor",
            "copy_kernel_cuda",
            "catarraybatchedcopy",
            "triton_poi",
            "triton_red",
        ),
    ),
    (
        "loss_vocab",
        (
            "cross_entropy",
            "xentropy",
            "nll_loss",
            "log_softmax",
            "vocab",
        ),
    ),
    (
        "optimizer",
        (
            "adam",
            "optimizer",
            "multi_tensor",
            "amp_",
            "scale",
        ),
    ),
    (
        "communication",
        (
            "nccl",
            "all_reduce",
            "reduce_scatter",
            "broadcast",
        ),
    ),
    (
        "normalization_activation",
        (
            "layernorm",
            "rmsnorm",
            "gelu",
            "silu",
            "swiglu",
        ),
    ),
]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_norm(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for name in fieldnames:
        lname = _norm(name)
        if any(candidate in lname for candidate in candidates):
            return name
    return None


def _float(raw: Any) -> float:
    text = str(raw or "").strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _classify_kernel(name: str) -> str:
    lname = name.lower()
    for phase, patterns in PHASE_PATTERNS:
        if any(pattern in lname for pattern in patterns):
            return phase
    return "other"


def _read_kernel_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            return []
        name_col = _pick_column(fieldnames, ("name", "kernel_name"))
        time_col = _pick_column(
            fieldnames,
            (
                "total_time_ns",
                "total_time",
                "total",
                "time_ns",
                "duration_ns",
            ),
        )
        pct_col = _pick_column(fieldnames, ("time", "time_%", "percentage", "percent"))
        count_col = _pick_column(fieldnames, ("instances", "calls", "count", "num_calls"))
        avg_col = _pick_column(fieldnames, ("avg_ns", "average_ns", "avg", "average"))
        if not name_col:
            raise SystemExit(f"Could not find a kernel name column in {path}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            name = str(row.get(name_col, "")).strip()
            if not name:
                continue
            rows.append(
                {
                    "name": name,
                    "phase": _classify_kernel(name),
                    "total_time": _float(row.get(time_col)) if time_col else 0.0,
                    "time_pct": _float(row.get(pct_col)) if pct_col else 0.0,
                    "calls": int(_float(row.get(count_col))) if count_col else 0,
                    "avg_time": _float(row.get(avg_col)) if avg_col else 0.0,
                }
            )
        return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _phase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_phase: dict[str, dict[str, float]] = defaultdict(lambda: {"total_time": 0.0, "calls": 0.0})
    total_time = sum(float(row["total_time"]) for row in rows)
    total_pct = sum(float(row["time_pct"]) for row in rows)
    for row in rows:
        phase = str(row["phase"])
        by_phase[phase]["total_time"] += float(row["total_time"])
        by_phase[phase]["calls"] += float(row["calls"])
    output = []
    for phase, values in sorted(by_phase.items(), key=lambda item: item[1]["total_time"], reverse=True):
        if total_time > 0:
            pct = 100.0 * values["total_time"] / total_time
        elif total_pct > 0:
            pct = sum(float(row["time_pct"]) for row in rows if row["phase"] == phase)
        else:
            pct = 0.0
        output.append(
            {
                "phase": phase,
                "time_pct": round(pct, 2),
                "total_time_raw": round(values["total_time"], 3),
                "calls": int(values["calls"]),
            }
        )
    return output


def _recommendations(phase_rows: list[dict[str, Any]]) -> list[str]:
    phase_pct = {str(row["phase"]): float(row["time_pct"]) for row in phase_rows}
    notes: list[str] = []
    if phase_pct.get("dispatch_permute", 0.0) >= 15.0:
        notes.append("Dispatch/permute is high: prioritize moe_permute_fusion, grouped GEMM, and dispatcher comparisons.")
    if phase_pct.get("expert_gemm", 0.0) < 20.0 and phase_pct.get("dispatch_permute", 0.0) >= 10.0:
        notes.append("Expert GEMM is not dominating: routing/dispatch overhead is likely limiting MFU.")
    if phase_pct.get("attention", 0.0) >= 25.0:
        notes.append("Attention is a major phase: compare auto/fused/flash backends before changing MoE structure.")
    if phase_pct.get("loss_vocab", 0.0) >= 15.0:
        notes.append("Loss/vocab projection is high: check fused cross entropy and tokenizer/vocab-size pressure.")
    if phase_pct.get("optimizer", 0.0) >= 15.0:
        notes.append("Optimizer is high: test gradient accumulation fusion and larger global batch for overhead amortization.")
    if phase_pct.get("communication", 0.0) >= 5.0:
        notes.append("Communication appeared in a single-GPU profile: inspect launcher/parallelism settings before multi-GPU sweeps.")
    if not notes:
        notes.append("No single non-GEMM phase dominates; continue with batch-size and fused-kernel MFU sweeps.")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-kernels", type=int, default=40)
    args = parser.parse_args()

    rows = _read_kernel_rows(args.kernel_csv)
    rows.sort(key=lambda row: float(row["total_time"]), reverse=True)
    top_rows = rows[: max(1, int(args.top_kernels))]
    phase_rows = _phase_rows(rows)
    recommendations = _recommendations(phase_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "kernel_summary.csv", top_rows)
    _write_csv(args.output_dir / "phase_summary.csv", phase_rows)
    (args.output_dir / "recommendations.md").write_text(
        "\n".join(
            [
                "# 600M MoE Profiling Recommendations",
                "",
                "## Phase Summary",
                "",
                *[f"- `{row['phase']}`: {row['time_pct']}%" for row in phase_rows],
                "",
                "## Recommendations",
                "",
                *[f"- {note}" for note in recommendations],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (args.output_dir / "phase_summary.json").write_text(
        json.dumps({"phases": phase_rows, "recommendations": recommendations}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'phase_summary.csv'}")
    print(f"Wrote {args.output_dir / 'kernel_summary.csv'}")
    print(f"Wrote {args.output_dir / 'recommendations.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
