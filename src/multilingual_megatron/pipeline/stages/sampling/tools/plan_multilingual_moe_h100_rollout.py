#!/usr/bin/env python3
"""Build staged H100 rollout plans from multilingual MoE run configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (
    DEFAULT_STUDY_DIR,
    H100_ASSUMED_MFU,
    H100_EFFECTIVE_FLOPS_PER_GPU_HOUR,
    H100_PEAK_TFLOPS_PER_GPU,
    ROUTING_GATE,
    STUDY_ID,
    estimate_gpu_hours,
    write_csv,
    write_json,
)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row(run: dict, *, stage: str, target_tokens: int, purpose: str) -> dict[str, object]:
    rollout = run["h100_rollout"]
    n_act = int(run["counts"]["N_act"])
    experts = int(run["E"])
    hours = 0.0
    if target_tokens > 0:
        hours = estimate_gpu_hours(
            n_act=n_act,
            tokens=target_tokens,
            experts=experts,
            effective_flops_per_gpu_hour=H100_EFFECTIVE_FLOPS_PER_GPU_HOUR,
        )
    return {
        "stage": stage,
        "run_id": run["run_id"],
        "purpose": purpose,
        "ladder": run["ladder"],
        "E": experts,
        "K": int(run["K"]),
        "target_tokens": target_tokens,
        "target_checkpoint": target_tokens if target_tokens > 0 else "",
        "h100_gpus": int(rollout["gpus"]),
        "tensor_parallel": int(rollout["tensor_parallel"]),
        "expert_parallel": int(rollout["expert_parallel"]),
        "estimated_h100_gpu_hours": f"{hours:.3f}",
        "max_dropped_assignment_rate": ROUTING_GATE["max_dropped_assignment_rate"],
        "require_zero_kept_tokens": ROUTING_GATE["require_zero_kept_tokens"],
        "min_capacity_factor": ROUTING_GATE["min_capacity_factor"],
        "notes": rollout["notes"],
    }


def build_rollout_rows(runs: list[dict]) -> list[dict[str, object]]:
    by_id = {run["run_id"]: run for run in runs}
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "stage": "metadata",
            "run_id": "all_64",
            "purpose": "generate and validate all H100 study configs without GPU",
            "ladder": "all",
            "E": "all",
            "K": "all",
            "target_tokens": 0,
            "target_checkpoint": "",
            "h100_gpus": 0,
            "tensor_parallel": 0,
            "expert_parallel": 0,
            "estimated_h100_gpu_hours": "0.000",
            "max_dropped_assignment_rate": ROUTING_GATE["max_dropped_assignment_rate"],
            "require_zero_kept_tokens": ROUTING_GATE["require_zero_kept_tokens"],
            "min_capacity_factor": ROUTING_GATE["min_capacity_factor"],
            "notes": "python generator and validator only",
        }
    )

    for run_id in (
        f"{STUDY_ID}_LA_E1_K1",
        f"{STUDY_ID}_LA_E32_K200",
        f"{STUDY_ID}_LD_E32_K200",
    ):
        rows.append(_row(by_id[run_id], stage="dryrun", target_tokens=0, purpose="render Bridge command and metadata"))

    for ladder in ("A", "B", "C", "D"):
        for experts in (1, 4, 16, 32):
            rows.append(
                _row(
                    by_id[f"{STUDY_ID}_L{ladder}_E{experts}_K1"],
                    stage="h100_shape_smoke",
                    target_tokens=0,
                    purpose="probe the one systems shape reused by all K variants",
                )
            )

    for ladder in ("A", "B", "C", "D"):
        for experts in (4, 16, 32):
            rows.append(
                _row(
                    by_id[f"{STUDY_ID}_L{ladder}_E{experts}_K200"],
                    stage="h100_routing_gate",
                    target_tokens=500_000_000,
                    purpose="confirm small-drop gate on the largest language set",
                )
            )

    for experts in (1, 4, 16, 32):
        for k in (1, 16, 50, 200):
            rows.append(
                _row(
                    by_id[f"{STUDY_ID}_LA_E{experts}_K{k}"],
                    stage="h100_pilot_a_all_k",
                    target_tokens=500_000_000,
                    purpose="first real checkpoint across all K on the smallest scale",
                )
            )

    for experts in (4, 16, 32):
        for k in (1, 200):
            rows.append(
                _row(
                    by_id[f"{STUDY_ID}_LD_E{experts}_K{k}"],
                    stage="h100_pilot_d_endpoints",
                    target_tokens=500_000_000,
                    purpose="largest active scale endpoint check before full trajectories",
                )
            )
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_stage: dict[str, dict[str, float]] = {}
    for row in rows:
        stage = str(row["stage"])
        item = by_stage.setdefault(stage, {"rows": 0, "estimated_h100_gpu_hours": 0.0})
        item["rows"] += 1
        item["estimated_h100_gpu_hours"] += float(row["estimated_h100_gpu_hours"])
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "h100_peak_tflops_per_gpu": H100_PEAK_TFLOPS_PER_GPU,
        "assumed_h100_mfu": H100_ASSUMED_MFU,
        "effective_h100_flops_per_gpu_hour": H100_EFFECTIVE_FLOPS_PER_GPU_HOUR,
        "routing_gate": dict(ROUTING_GATE),
        "partition_assumptions": {
            "partition": "gpu_h100",
            "gpus_per_run": "1-4",
            "target_hardware": "H100 only",
        },
        "stage_summary": by_stage,
        "notes": [
            "H100 estimates are planning estimates, not measured MFU.",
            "Optimize 16 shape cells first: one per ladder and expert count.",
            "K variants reuse the locked shape unless input pipeline throughput shifts enough to invalidate MFU.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", default=str(DEFAULT_STUDY_DIR))
    parser.add_argument("--output-csv", default=str(DEFAULT_STUDY_DIR / "h100_rollout_plan.csv"))
    parser.add_argument("--output-json", default=str(DEFAULT_STUDY_DIR / "h100_rollout_summary.json"))
    args = parser.parse_args()

    runs = _load_jsonl(Path(args.study_dir) / "run_configs.jsonl")
    rows = build_rollout_rows(runs)
    write_csv(Path(args.output_csv), rows)
    write_json(Path(args.output_json), _summary(rows))
    print(f"Wrote H100 rollout plan to {args.output_csv}")
    print(f"Wrote H100 rollout summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
