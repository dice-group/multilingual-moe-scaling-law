"""Freeze the physical 500-H100-hour checkpoint after the required calibration.

The law-selected endpoint is a token/optimizer-step contract.  The 500-H100-hour
checkpoint is different: it can only be placed after measuring the GBS 32/64
ramp and a production-shaped synchronous checkpoint cycle.  This module turns
that receipt into the *last complete optimizer update plus checkpoint* that
does not exceed the metered allocation budget.  It never submits or mutates a
production profile; the emitted JSON is the auditable input to the small
follow-up profile amendment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from multilingual_megatron.pipeline.stages.training.batch_policy import (
    build_joint_moe_full_state_wsd_v2_plan,
    resolve_batch_policy,
)


SCHEMA = "stage1_l26_e32_500h_checkpoint_selection/v1"
REQUIRED_GBS = (32, 64, 128)


class PhysicalCheckpointSelectionError(ValueError):
    """Raised for an incomplete or internally inconsistent calibration receipt."""


def _positive_float(value: Any, *, field: str, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PhysicalCheckpointSelectionError(f"{field} must be a number") from exc
    if parsed < 0.0 or (not allow_zero and parsed == 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise PhysicalCheckpointSelectionError(f"{field} must be {qualifier}")
    return parsed


def _rates_from_receipt(receipt: dict[str, Any]) -> dict[int, float]:
    raw = receipt.get("aggregate_tokens_per_second_by_gbs")
    if not isinstance(raw, dict):
        raise PhysicalCheckpointSelectionError(
            "calibration receipt requires aggregate_tokens_per_second_by_gbs"
        )
    rates: dict[int, float] = {}
    for gbs in REQUIRED_GBS:
        value = raw.get(str(gbs), raw.get(gbs))
        rates[gbs] = _positive_float(value, field=f"aggregate_tokens_per_second_by_gbs[{gbs}]")
    return rates


def _checkpoint_steps(plan: dict[str, Any]) -> set[int]:
    return {
        int(endpoint["expected_step"])
        for endpoint in plan.get("readout_endpoints", [])
    }


def _candidate_rows(plan: dict[str, Any]) -> list[dict[str, int]]:
    """Return every full optimizer-update endpoint in the production trunk."""

    rows: list[dict[str, int]] = []
    phase_starts: dict[int, tuple[int, int]] = {}
    for segment in plan["trunk"]["segments"]:
        start_step = int(segment["cumulative_start_step"])
        end_step = int(segment["cumulative_end_step"])
        start_tokens = int(segment["actual_start_tokens"])
        tokens_per_step = int(segment["tokens_per_step"])
        schedule_index = int(segment["schedule_index"])
        phase_starts.setdefault(
            schedule_index,
            (int(segment["target_start_tokens"]), start_step),
        )
        schedule_start_tokens, phase_start_step = phase_starts[schedule_index]
        for step in range(start_step + 1, end_step + 1):
            phase_iterations = step - phase_start_step
            # ``materialize_batch_segments`` measures a fixed-GBS phase from
            # its *requested* schedule start, then rounds its cumulative
            # updates up.  This is deliberately not necessarily the same as
            # the already-consumed token count at the phase start after the
            # preceding GBS ramp.  Retain both quantities so the emitted
            # readout target maps back to this exact completed update.
            rows.append(
                {
                    "step": step,
                    "actual_tokens": start_tokens + (step - start_step) * tokens_per_step,
                    "global_batch_size": int(segment["global_batch_size"]),
                    "tokens_per_step": tokens_per_step,
                    "schedule_start_tokens": schedule_start_tokens,
                    "phase_iterations": phase_iterations,
                }
            )
    if not rows:
        raise PhysicalCheckpointSelectionError("production trunk contains no optimizer updates")
    return rows


def select_500_h100_hour_checkpoint(
    *,
    endpoint_plan: dict[str, Any],
    calibration_receipt: dict[str, Any],
    gpu_count: int = 8,
    target_h100_gpu_hours: float = 500.0,
) -> dict[str, Any]:
    """Select the latest legal trunk update whose full save completes by 500 H100 h.

    ``aggregate_tokens_per_second_by_gbs`` is the observed aggregate rate across
    all ``gpu_count`` H100s.  The receipt's startup and full-checkpoint values
    are metered *H100 GPU-seconds*, so the calculation includes their exact
    allocation accounting rather than treating a wall-clock duration as one GPU.
    The physical milestone's checkpoint cost is included.  At every scientific
    readout that has already completed, the same full-checkpoint cost is also
    included once.
    """

    if int(gpu_count) <= 0:
        raise PhysicalCheckpointSelectionError("gpu_count must be positive")
    target_gpu_seconds = _positive_float(
        target_h100_gpu_hours, field="target_h100_gpu_hours"
    ) * 3600.0
    rates = _rates_from_receipt(calibration_receipt)
    startup_gpu_seconds = _positive_float(
        calibration_receipt.get("startup_h100_gpu_seconds", 0.0),
        field="startup_h100_gpu_seconds",
        allow_zero=True,
    )
    full_checkpoint_gpu_seconds = _positive_float(
        calibration_receipt.get("full_checkpoint_h100_gpu_seconds"),
        field="full_checkpoint_h100_gpu_seconds",
        allow_zero=True,
    )

    elapsed_gpu_seconds = startup_gpu_seconds
    scientific_checkpoint_steps = _checkpoint_steps(endpoint_plan)
    selected: dict[str, Any] | None = None
    next_candidate: dict[str, Any] | None = None
    for row in _candidate_rows(endpoint_plan):
        gbs = int(row["global_batch_size"])
        elapsed_gpu_seconds += (
            int(row["tokens_per_step"]) / rates[gbs] * int(gpu_count)
        )
        if row["step"] in scientific_checkpoint_steps:
            elapsed_gpu_seconds += full_checkpoint_gpu_seconds
        # A new physical checkpoint is saved at every candidate except when it
        # already coincides with a scientific readout that just saved one.
        completed_checkpoint_gpu_seconds = elapsed_gpu_seconds
        if row["step"] not in scientific_checkpoint_steps:
            completed_checkpoint_gpu_seconds += full_checkpoint_gpu_seconds
        candidate = {
            **row,
            "completed_checkpoint_h100_gpu_seconds": completed_checkpoint_gpu_seconds,
            "completed_checkpoint_h100_gpu_hours": completed_checkpoint_gpu_seconds / 3600.0,
        }
        if completed_checkpoint_gpu_seconds <= target_gpu_seconds:
            selected = candidate
            continue
        next_candidate = candidate
        break

    if selected is None:
        raise PhysicalCheckpointSelectionError(
            "No complete optimizer update plus full checkpoint fits inside the physical budget"
        )
    if next_candidate is None:
        raise PhysicalCheckpointSelectionError(
            "The 500-H100-hour target is not inside the trunk; check the endpoint plan or budget"
        )
    if int(selected["step"]) in scientific_checkpoint_steps:
        checkpoint_kind = "existing_scientific_readout"
    else:
        checkpoint_kind = "additional_physical_readout"
    # Any requested boundary in [start + (k-1)Q + 1, start + kQ] materializes
    # to phase update k.  Emit the lower legal target.  Adding the *actual*
    # consumed-token value would normally round once more because the first
    # two ramp phases already have a small legal overshoot.
    readout_target = (
        int(selected["schedule_start_tokens"])
        + (int(selected["phase_iterations"]) - 1) * int(selected["tokens_per_step"])
        + 1
    )

    return {
        "schema_version": 1,
        "schema": SCHEMA,
        "status": "selected_from_complete_calibration_receipt",
        "meter": (
            "allocated H100 GPU-seconds while the production training allocation is active, "
            "excluding queue time; includes calibrated startup and synchronous full-checkpoint cost"
        ),
        "target_h100_gpu_hours": float(target_h100_gpu_hours),
        "gpu_count": int(gpu_count),
        "calibration": {
            "aggregate_tokens_per_second_by_gbs": {str(key): rates[key] for key in REQUIRED_GBS},
            "startup_h100_gpu_seconds": startup_gpu_seconds,
            "full_checkpoint_h100_gpu_seconds": full_checkpoint_gpu_seconds,
        },
        "selected_checkpoint": {
            **selected,
            "checkpoint_kind": checkpoint_kind,
            "profile_readout_horizon_tokens_to_add": readout_target,
            "remaining_h100_gpu_seconds": target_gpu_seconds
            - float(selected["completed_checkpoint_h100_gpu_seconds"]),
        },
        "next_full_update_plus_checkpoint": next_candidate,
        "amendment_instruction": (
            "Insert profile_readout_horizon_tokens_to_add into the trunk profile's "
            "readout_horizons_tokens in increasing order, regenerate the dry-run segment plan, "
            "and require the selected expected step and consumed tokens to match this receipt."
        ),
    }


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicalCheckpointSelectionError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PhysicalCheckpointSelectionError(f"{label} must be a JSON object: {path}")
    return payload


def load_production_profile(path: Path, *, profile_name: str) -> dict[str, Any]:
    """Load one materialized profile from a bundle directory or legacy JSON file."""

    if path.is_dir():
        from multilingual_megatron.pipeline.config_loader import (
            load_pipeline_config_bundle,
        )
        from multilingual_megatron.pipeline.runtime import _resolve_profile

        return _resolve_profile(load_pipeline_config_bundle(path), profile_name)

    profiles = _load_object(path, label="profiles")
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise PhysicalCheckpointSelectionError(f"Unknown profile {profile_name!r}")
    if str(profile.get("extends", "")).strip():
        raise PhysicalCheckpointSelectionError(
            "An inherited profile must be loaded from the pipeline config bundle directory "
            "so its parent profile is resolved"
        )
    return profile


def build_plan_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    train = profile.get("training", {}).get("train")
    if not isinstance(train, dict):
        raise PhysicalCheckpointSelectionError("profile lacks training.train")
    policy = resolve_batch_policy(train, seq_length=int(profile["training"].get("seq_length", 2048)))
    if policy is None:
        raise PhysicalCheckpointSelectionError("profile batch policy is not enabled")
    return build_joint_moe_full_state_wsd_v2_plan(
        policy=policy,
        plateau_micro_batch_size=int(train["micro_batch_size"]),
        data_parallel_size=int(policy["data_parallel_size"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        required=True,
        help="pipeline config bundle directory or standalone production profiles JSON",
    )
    parser.add_argument("--profile", required=True, help="trunk profile name")
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--target-h100-gpu-hours", type=float, default=500.0)
    args = parser.parse_args(argv)

    profile = load_production_profile(args.profiles, profile_name=args.profile)
    result = select_500_h100_hour_checkpoint(
        endpoint_plan=build_plan_from_profile(profile),
        calibration_receipt=_load_object(args.calibration_receipt, label="calibration receipt"),
        gpu_count=args.gpu_count,
        target_h100_gpu_hours=args.target_h100_gpu_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
