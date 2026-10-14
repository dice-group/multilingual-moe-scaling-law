from __future__ import annotations

import math
from typing import Any


DEFAULT_STAGE1_BATCH_SCHEDULE = (
    {"start_tokens": 0, "end_tokens": 500_000_000, "global_batch_size": 32},
    {"start_tokens": 500_000_000, "end_tokens": 1_000_000_000, "global_batch_size": 64},
    {"start_tokens": 1_000_000_000, "end_tokens": 0, "global_batch_size": 128},
)

JOINT_MOE_FULL_STATE_WSD_V2 = "joint_moe_full_state_wsd_v2"
JOINT_MOE_WSD_V2_WARMUP_TOKENS = 130_000_000
JOINT_MOE_WSD_V2_STABLE_NUMERATOR = 4
JOINT_MOE_WSD_V2_HORIZON_DENOMINATOR = 5


def _as_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_positive_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return parsed if parsed > 0 else default


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _positive_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SystemExit(f"{field} must be a non-empty list of positive integers")
    result = [int(item) for item in value]
    if any(item <= 0 for item in result):
        raise SystemExit(f"{field} must contain only positive integers")
    if result != sorted(set(result)):
        raise SystemExit(f"{field} must be strictly increasing with no duplicates")
    return result


def is_joint_moe_full_state_wsd_v2(policy: dict[str, Any] | None) -> bool:
    return bool(policy) and str(policy.get("planning_method", "")) == JOINT_MOE_FULL_STATE_WSD_V2


def _validate_joint_moe_full_state_wsd_v2(
    *,
    raw: dict[str, Any],
    schedule: list[dict[str, int]],
    target_gbs: int,
    seq_length: int,
    warmup_tokens: int,
) -> dict[str, Any]:
    expected_schedule = [dict(item) for item in DEFAULT_STAGE1_BATCH_SCHEDULE]
    if schedule != expected_schedule:
        raise SystemExit(
            "joint_moe_full_state_wsd_v2 requires the exact batch ramp "
            "GBS 32 through 0.5B tokens, GBS 64 through 1B, then GBS 128"
        )
    if target_gbs != 128:
        raise SystemExit("joint_moe_full_state_wsd_v2 requires target_global_batch_size=128")
    if int(seq_length) != 2048:
        raise SystemExit("joint_moe_full_state_wsd_v2 requires seq_length=2048")
    if int(warmup_tokens) != JOINT_MOE_WSD_V2_WARMUP_TOKENS:
        raise SystemExit("joint_moe_full_state_wsd_v2 requires warmup_tokens=130000000")

    decay_fraction = float(raw.get("decay_fraction", 0.2))
    if not math.isclose(decay_fraction, 0.2, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("joint_moe_full_state_wsd_v2 requires decay_fraction=0.2")

    checkpoint_horizons = _positive_int_list(
        raw.get("checkpoint_horizons_tokens"),
        field="training.train.batch_policy.checkpoint_horizons_tokens",
    )
    readout_horizons = (
        _positive_int_list(
            raw.get("readout_horizons_tokens"),
            field="training.train.batch_policy.readout_horizons_tokens",
        )
        if raw.get("readout_horizons_tokens") not in (None, [])
        else []
    )
    dataset_horizon = _as_positive_int(raw.get("dataset_horizon_tokens"), 0)
    if dataset_horizon <= 0:
        raise SystemExit("joint_moe_full_state_wsd_v2 requires batch_policy.dataset_horizon_tokens")
    if checkpoint_horizons[-1] != dataset_horizon:
        raise SystemExit(
            "joint_moe_full_state_wsd_v2 requires dataset_horizon_tokens to equal the largest "
            "checkpoint_horizons_tokens value"
        )
    for horizon in checkpoint_horizons:
        if (horizon * JOINT_MOE_WSD_V2_STABLE_NUMERATOR) % JOINT_MOE_WSD_V2_HORIZON_DENOMINATOR:
            raise SystemExit(f"checkpoint horizon {horizon} does not have an exact integer 0.8D endpoint")
    if any(horizon >= dataset_horizon for horizon in readout_horizons):
        raise SystemExit("readout_horizons_tokens must be strictly below dataset_horizon_tokens")

    plan_role = str(raw.get("plan_role", "trunk")).strip().lower() or "trunk"
    if plan_role not in {"trunk", "branch"}:
        raise SystemExit("joint_moe_full_state_wsd_v2 batch_policy.plan_role must be 'trunk' or 'branch'")
    branch_target_tokens = _as_positive_int(raw.get("branch_target_tokens"), 0)
    if plan_role == "branch" and branch_target_tokens not in checkpoint_horizons:
        raise SystemExit(
            "joint_moe_full_state_wsd_v2 branch_target_tokens must be one of checkpoint_horizons_tokens"
        )
    branch_parent_checkpoint_root = str(raw.get("branch_parent_checkpoint_root", "")).strip()
    if plan_role == "branch" and not branch_parent_checkpoint_root:
        raise SystemExit(
            "joint_moe_full_state_wsd_v2 branch plans require "
            "batch_policy.branch_parent_checkpoint_root"
        )
    if plan_role == "trunk" and branch_target_tokens:
        raise SystemExit("joint_moe_full_state_wsd_v2 trunk plans must not set branch_target_tokens")
    if plan_role == "trunk" and branch_parent_checkpoint_root:
        raise SystemExit(
            "joint_moe_full_state_wsd_v2 trunk plans must not set "
            "branch_parent_checkpoint_root"
        )

    return {
        "dataset_horizon_tokens": dataset_horizon,
        "checkpoint_horizons_tokens": checkpoint_horizons,
        "readout_horizons_tokens": readout_horizons,
        "plan_role": plan_role,
        "branch_target_tokens": branch_target_tokens,
        "branch_parent_checkpoint_root": branch_parent_checkpoint_root,
        "decay_fraction": decay_fraction,
    }


def resolve_batch_policy(train: dict[str, Any], *, seq_length: int) -> dict[str, Any] | None:
    raw = train.get("batch_policy", {})
    if not isinstance(raw, dict) or not _as_bool(raw.get("enabled", False), False):
        return None

    mode = str(raw.get("mode", "token_ramp")).strip().lower() or "token_ramp"
    if mode != "token_ramp":
        raise SystemExit(f"training.train.batch_policy.mode must be 'token_ramp' (got {mode!r})")

    schedule_raw = raw.get("schedule", DEFAULT_STAGE1_BATCH_SCHEDULE)
    if not isinstance(schedule_raw, (list, tuple)) or not schedule_raw:
        raise SystemExit("training.train.batch_policy.schedule must contain at least one segment")

    schedule: list[dict[str, int]] = []
    for index, item in enumerate(schedule_raw):
        if not isinstance(item, dict):
            raise SystemExit(f"training.train.batch_policy.schedule[{index}] must be an object")
        start_tokens = int(item.get("start_tokens", 0) or 0)
        end_tokens = int(item.get("end_tokens", 0) or 0)
        global_batch_size = _as_positive_int(
            item.get("global_batch_size", item.get("global_batch_sequences")),
            0,
        )
        if global_batch_size <= 0:
            raise SystemExit(f"training.train.batch_policy.schedule[{index}].global_batch_size must be positive")
        if start_tokens < 0 or end_tokens < 0:
            raise SystemExit(f"training.train.batch_policy.schedule[{index}] token bounds must be non-negative")
        schedule.append(
            {
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
                "global_batch_size": global_batch_size,
            }
        )

    schedule.sort(key=lambda item: int(item["start_tokens"]))
    if schedule[0]["start_tokens"] != 0:
        raise SystemExit("training.train.batch_policy.schedule must start at token position 0")
    for previous, current in zip(schedule, schedule[1:]):
        if int(previous["end_tokens"]) != int(current["start_tokens"]):
            raise SystemExit("training.train.batch_policy.schedule must be contiguous")
        if int(previous["end_tokens"]) <= int(previous["start_tokens"]):
            raise SystemExit("non-final batch-policy segments must have end_tokens > start_tokens")
    if int(schedule[-1]["end_tokens"]) not in {0} and int(schedule[-1]["end_tokens"]) <= int(schedule[-1]["start_tokens"]):
        raise SystemExit("final batch-policy segment must be open-ended or have end_tokens > start_tokens")

    target_gbs = _as_positive_int(raw.get("target_global_batch_size"), int(schedule[-1]["global_batch_size"]))
    if target_gbs != int(schedule[-1]["global_batch_size"]):
        raise SystemExit("training.train.batch_policy.target_global_batch_size must match the final schedule GBS")

    planning_method = str(raw.get("planning_method", "")).strip()
    warmup_tokens = _as_positive_int(raw.get("warmup_tokens"), _as_positive_int(train.get("warmup_tokens"), 0))
    v2: dict[str, Any] = {}
    if planning_method == JOINT_MOE_FULL_STATE_WSD_V2:
        configured_warmup = raw.get("warmup_tokens", train.get("warmup_tokens"))
        warmup_tokens = (
            JOINT_MOE_WSD_V2_WARMUP_TOKENS
            if configured_warmup in (None, "")
            else int(configured_warmup)
        )
        v2 = _validate_joint_moe_full_state_wsd_v2(
            raw=raw,
            schedule=schedule,
            target_gbs=target_gbs,
            seq_length=seq_length,
            warmup_tokens=warmup_tokens,
        )

    return {
        "enabled": True,
        "mode": mode,
        "planning_method": planning_method,
        "target_global_batch_size": target_gbs,
        "seq_length": int(seq_length),
        "mbs_policy": str(raw.get("mbs_policy", "cap_per_segment")).strip() or "cap_per_segment",
        "data_parallel_size": _as_positive_int(raw.get("data_parallel_size"), 1),
        "require_full_state_checkpoint": _as_bool(raw.get("require_full_state_checkpoint", False)),
        "warmup_tokens": warmup_tokens,
        "schedule": schedule,
        **v2,
    }


def batch_policy_target_global_batch_size(policy: dict[str, Any] | None, fallback: int) -> int:
    if not policy:
        return int(fallback)
    return int(policy.get("target_global_batch_size", fallback) or fallback)


def derive_segment_micro_batch_size(
    *,
    plateau_micro_batch_size: int,
    global_batch_size: int,
    data_parallel_size: int = 1,
) -> int:
    maximum = min(int(plateau_micro_batch_size), max(1, int(global_batch_size) // max(1, int(data_parallel_size))))
    divisor = max(1, int(data_parallel_size))
    for candidate in range(maximum, 0, -1):
        if int(global_batch_size) % (candidate * divisor) == 0:
            return candidate
    return 1


def materialize_batch_segments(
    *,
    policy: dict[str, Any],
    total_tokens: int,
    plateau_micro_batch_size: int,
    data_parallel_size: int | None = None,
    split_tokens: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    seq_length = int(policy["seq_length"])
    dp_size = int(data_parallel_size or policy.get("data_parallel_size", 1) or 1)
    segments: list[dict[str, Any]] = []
    cumulative_start_step = 0
    actual_start_samples = 0
    actual_start_tokens = 0
    requested_splits = sorted(
        {
            int(value)
            for value in (split_tokens or [])
            if 0 < int(value) < int(total_tokens)
        }
    )
    for index, item in enumerate(policy["schedule"]):
        schedule_start = int(item["start_tokens"])
        if int(total_tokens) <= schedule_start:
            continue
        raw_end = int(item["end_tokens"])
        target_end = int(total_tokens) if raw_end == 0 else min(int(total_tokens), raw_end)
        if target_end <= schedule_start:
            continue
        gbs = int(item["global_batch_size"])
        mbs = derive_segment_micro_batch_size(
            plateau_micro_batch_size=plateau_micro_batch_size,
            global_batch_size=gbs,
            data_parallel_size=dp_size,
        )
        tokens_per_step = gbs * seq_length
        boundaries = [value for value in requested_splits if schedule_start < value < target_end]
        boundaries.append(target_end)
        target_start = schedule_start
        completed_schedule_iters = 0
        for boundary in boundaries:
            # Splitting one fixed-GBS schedule phase for a checkpoint must not
            # create extra optimizer updates.  Compute the cumulative number
            # of updates required from the phase start, then take the delta
            # from the previous checkpoint boundary.  Computing ``ceil`` on
            # every split independently would otherwise make checkpoint
            # placement change the final token count.
            cumulative_schedule_iters = _ceil_div(boundary - schedule_start, tokens_per_step)
            segment_iters = cumulative_schedule_iters - completed_schedule_iters
            cumulative_end_step = cumulative_start_step + segment_iters
            actual_end_samples = actual_start_samples + segment_iters * gbs
            actual_end_tokens = actual_start_tokens + segment_iters * tokens_per_step
            segments.append(
                {
                    "segment_index": len(segments),
                    "schedule_index": index,
                    "target_start_tokens": target_start,
                    "target_end_tokens": boundary,
                    "segment_target_tokens": boundary - target_start,
                    "actual_start_samples": actual_start_samples,
                    "actual_end_samples": actual_end_samples,
                    "actual_start_tokens": actual_start_tokens,
                    "actual_end_tokens": actual_end_tokens,
                    "global_batch_size": gbs,
                    "micro_batch_size": mbs,
                    "data_parallel_size": dp_size,
                    "seq_length": seq_length,
                    "tokens_per_step": tokens_per_step,
                    "segment_train_iters": segment_iters,
                    "cumulative_start_step": cumulative_start_step,
                    "cumulative_end_step": cumulative_end_step,
                    "expected_end_step": cumulative_end_step,
                    "expected_consumed_samples": actual_end_samples,
                }
            )
            target_start = boundary
            completed_schedule_iters = cumulative_schedule_iters
            cumulative_start_step = cumulative_end_step
            actual_start_samples = actual_end_samples
            actual_start_tokens = actual_end_tokens
        if target_end >= int(total_tokens):
            break
    if not segments:
        raise SystemExit("training.train.batch_policy produced no training segments")
    return segments


def _endpoint_state(segments: list[dict[str, Any]], target_tokens: int) -> dict[str, int]:
    for segment in segments:
        if int(segment["target_end_tokens"]) == int(target_tokens):
            return {
                "step": int(segment["cumulative_end_step"]),
                "consumed_samples": int(segment["actual_end_samples"]),
                "actual_tokens": int(segment["actual_end_tokens"]),
            }
    raise SystemExit(f"No materialized batch-policy endpoint exists at {target_tokens} tokens")


def _materialize_continuation_segments(
    *,
    policy: dict[str, Any],
    start_tokens: int,
    end_tokens: int,
    initial_step: int,
    initial_consumed_samples: int,
    plateau_micro_batch_size: int,
    data_parallel_size: int,
    split_tokens: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    seq_length = int(policy["seq_length"])
    cumulative_start_step = int(initial_step)
    actual_start_samples = int(initial_consumed_samples)
    actual_start_tokens = actual_start_samples * seq_length
    segments: list[dict[str, Any]] = []
    requested_splits = sorted(
        {
            int(value)
            for value in (split_tokens or [])
            if int(start_tokens) < int(value) < int(end_tokens)
        }
    )
    for schedule_index, item in enumerate(policy["schedule"]):
        schedule_start = int(item["start_tokens"])
        raw_end = int(item["end_tokens"])
        schedule_end = int(end_tokens) if raw_end == 0 else raw_end
        target_start = max(int(start_tokens), schedule_start)
        target_end = min(int(end_tokens), schedule_end)
        if target_end <= target_start:
            continue
        gbs = int(item["global_batch_size"])
        mbs = derive_segment_micro_batch_size(
            plateau_micro_batch_size=plateau_micro_batch_size,
            global_batch_size=gbs,
            data_parallel_size=data_parallel_size,
        )
        tokens_per_step = gbs * seq_length
        boundaries = [value for value in requested_splits if target_start < value < target_end]
        boundaries.append(target_end)
        completed_schedule_iters = 0
        continuation_start = target_start
        for boundary in boundaries:
            # As in trunk planning, readout/checkpoint splits are bookkeeping
            # boundaries, not additional work.  Anchor their cumulative step
            # count at the continuation start so an inserted checkpoint cannot
            # change the branch's final endpoint.
            cumulative_schedule_iters = _ceil_div(boundary - continuation_start, tokens_per_step)
            segment_iters = cumulative_schedule_iters - completed_schedule_iters
            cumulative_end_step = cumulative_start_step + segment_iters
            actual_end_samples = actual_start_samples + segment_iters * gbs
            actual_end_tokens = actual_end_samples * seq_length
            segments.append(
                {
                    "segment_index": len(segments),
                    "schedule_index": schedule_index,
                    "target_start_tokens": target_start,
                    "target_end_tokens": boundary,
                    "segment_target_tokens": boundary - target_start,
                    "actual_start_samples": actual_start_samples,
                    "actual_end_samples": actual_end_samples,
                    "actual_start_tokens": actual_start_tokens,
                    "actual_end_tokens": actual_end_tokens,
                    "global_batch_size": gbs,
                    "micro_batch_size": mbs,
                    "data_parallel_size": data_parallel_size,
                    "seq_length": seq_length,
                    "tokens_per_step": tokens_per_step,
                    "segment_train_iters": segment_iters,
                    "cumulative_start_step": cumulative_start_step,
                    "cumulative_end_step": cumulative_end_step,
                    "expected_end_step": cumulative_end_step,
                    "expected_consumed_samples": actual_end_samples,
                }
            )
            target_start = boundary
            completed_schedule_iters = cumulative_schedule_iters
            cumulative_start_step = cumulative_end_step
            actual_start_samples = actual_end_samples
            actual_start_tokens = actual_end_tokens
    if not segments or int(segments[-1]["target_end_tokens"]) != int(end_tokens):
        raise SystemExit(f"Batch policy does not cover continuation range {start_tokens}..{end_tokens}")
    return segments


def build_joint_moe_full_state_wsd_v2_plan(
    *,
    policy: dict[str, Any],
    plateau_micro_batch_size: int,
    data_parallel_size: int | None = None,
) -> dict[str, Any]:
    if not is_joint_moe_full_state_wsd_v2(policy):
        raise SystemExit("joint_moe_full_state_wsd_v2 endpoint planning requires the canonical planning_method")

    dp_size = int(data_parallel_size or policy.get("data_parallel_size", 1) or 1)
    horizons = [int(value) for value in policy["checkpoint_horizons_tokens"]]
    readout_horizons = [int(value) for value in policy.get("readout_horizons_tokens", [])]
    stable_endpoints = [
        horizon * JOINT_MOE_WSD_V2_STABLE_NUMERATOR // JOINT_MOE_WSD_V2_HORIZON_DENOMINATOR
        for horizon in horizons
    ]
    trunk_segments = materialize_batch_segments(
        policy=policy,
        total_tokens=stable_endpoints[-1],
        plateau_micro_batch_size=plateau_micro_batch_size,
        data_parallel_size=dp_size,
        split_tokens=sorted(set(stable_endpoints + [value for value in readout_horizons if value <= stable_endpoints[-1]])),
    )

    branches: list[dict[str, Any]] = []
    for horizon, stable_endpoint in zip(horizons, stable_endpoints):
        start_state = _endpoint_state(trunk_segments, stable_endpoint)
        branch_segments = _materialize_continuation_segments(
            policy=policy,
            start_tokens=stable_endpoint,
            end_tokens=horizon,
            initial_step=start_state["step"],
            initial_consumed_samples=start_state["consumed_samples"],
            plateau_micro_batch_size=plateau_micro_batch_size,
            data_parallel_size=dp_size,
            split_tokens=[value for value in readout_horizons if stable_endpoint < value < horizon],
        )
        end_segment = branch_segments[-1]
        branches.append(
            {
                "target_tokens": horizon,
                "start_tokens": stable_endpoint,
                "expected_start_step": start_state["step"],
                "expected_start_consumed_samples": start_state["consumed_samples"],
                "expected_end_step": int(end_segment["cumulative_end_step"]),
                "expected_consumed_samples": int(end_segment["actual_end_samples"]),
                "actual_end_tokens": int(end_segment["actual_end_tokens"]),
                "segments": branch_segments,
            }
        )

    final_branch = branches[-1]
    readout_endpoints: list[dict[str, Any]] = []
    for target in readout_horizons:
        source = trunk_segments if target <= stable_endpoints[-1] else final_branch["segments"]
        state = _endpoint_state(source, target)
        readout_endpoints.append(
            {
                "target_tokens": target,
                "plan_role": "trunk" if source is trunk_segments else "branch",
                "expected_step": state["step"],
                "expected_consumed_samples": state["consumed_samples"],
                "actual_tokens": state["actual_tokens"],
            }
        )
    fixed_final_step = int(final_branch["expected_end_step"])
    target_gbs = int(policy["target_global_batch_size"])
    fixed_dataset_samples = fixed_final_step * target_gbs
    trunk_end = trunk_segments[-1]
    return {
        "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
        "dataset_horizon_tokens": int(policy["dataset_horizon_tokens"]),
        "checkpoint_horizons_tokens": horizons,
        "readout_horizons_tokens": readout_horizons,
        "readout_endpoints": readout_endpoints,
        "stable_checkpoint_tokens": stable_endpoints,
        "trunk": {
            "target_end_tokens": stable_endpoints[-1],
            "expected_end_step": int(trunk_end["cumulative_end_step"]),
            "expected_consumed_samples": int(trunk_end["actual_end_samples"]),
            "actual_end_tokens": int(trunk_end["actual_end_tokens"]),
            "segments": trunk_segments,
        },
        "branches": branches,
        "fixed_dataset_final_cumulative_step": fixed_final_step,
        "fixed_dataset_sample_horizon": fixed_dataset_samples,
        "fixed_dataset_token_capacity": fixed_dataset_samples * int(policy["seq_length"]),
        "target_global_batch_size": target_gbs,
    }


def joint_moe_full_state_wsd_v2_scheduler_accounting(
    *,
    policy: dict[str, Any],
    endpoint_plan: dict[str, Any],
    branch_target_tokens: int | None = None,
) -> dict[str, int | str]:
    if not is_joint_moe_full_state_wsd_v2(policy):
        raise SystemExit("WSD accounting requires planning_method=joint_moe_full_state_wsd_v2")
    target = int(branch_target_tokens or policy["dataset_horizon_tokens"])
    branch = next(
        (item for item in endpoint_plan["branches"] if int(item["target_tokens"]) == target),
        None,
    )
    if branch is None:
        raise SystemExit(f"No joint-MoE v2 branch is defined for D={target}")

    first_gbs = int(policy["schedule"][0]["global_batch_size"])
    warmup_iters_at_first_gbs = scheduler_iter_for_tokens(
        tokens=int(policy["warmup_tokens"]),
        global_batch_size=first_gbs,
        seq_length=int(policy["seq_length"]),
    )
    warmup_samples = warmup_iters_at_first_gbs * first_gbs
    decay_start_samples = int(branch["expected_start_consumed_samples"])
    decay_end_samples = int(branch["expected_consumed_samples"])
    return {
        "style": "WSD",
        "wsd_decay_style": "linear",
        "target_tokens": target,
        "warmup_target_tokens": int(policy["warmup_tokens"]),
        "warmup_samples": warmup_samples,
        "warmup_actual_tokens": warmup_samples * int(policy["seq_length"]),
        "decay_start_samples": decay_start_samples,
        "lr_decay_samples": decay_end_samples,
        "wsd_decay_samples": decay_end_samples - decay_start_samples,
    }


def cumulative_step_for_tokens(segments: list[dict[str, Any]], target_tokens: int) -> int:
    if target_tokens <= 0:
        return 0
    for segment in segments:
        start = int(segment["target_start_tokens"])
        end = int(segment["target_end_tokens"])
        if int(target_tokens) <= end:
            local_tokens = max(0, int(target_tokens) - start)
            return int(segment["cumulative_start_step"]) + _ceil_div(local_tokens, int(segment["tokens_per_step"]))
    return int(segments[-1]["cumulative_end_step"])


def tokens_for_cumulative_step(segments: list[dict[str, Any]], step: int) -> int:
    if step <= 0:
        return 0
    for segment in segments:
        if int(step) <= int(segment["cumulative_end_step"]):
            local_steps = max(0, int(step) - int(segment["cumulative_start_step"]))
            return int(segment["actual_start_tokens"]) + local_steps * int(segment["tokens_per_step"])
    return int(segments[-1]["actual_end_tokens"])


def segment_for_token_position(segments: list[dict[str, Any]], target_tokens: int) -> dict[str, Any]:
    for segment in segments:
        if int(target_tokens) <= int(segment["target_end_tokens"]):
            return segment
    return segments[-1]


def segment_for_cumulative_step(segments: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for segment in segments:
        if int(step) <= int(segment["cumulative_end_step"]):
            return segment
    return segments[-1]


def scheduler_iter_for_tokens(*, tokens: int, global_batch_size: int, seq_length: int) -> int:
    if int(tokens) <= 0:
        return 0
    samples = _ceil_div(int(tokens), int(seq_length))
    return max(1, _ceil_div(samples, int(global_batch_size)))


def actual_tokens_from_segments(segments: list[dict[str, Any]]) -> int:
    return int(segments[-1]["actual_end_tokens"]) if segments else 0


def final_cumulative_step(segments: list[dict[str, Any]]) -> int:
    return int(segments[-1]["cumulative_end_step"]) if segments else 0
