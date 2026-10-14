"""Estimate campaign runtime from the completed D-ladder calibration wave."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.stage1_benchmarking.api import (
    _atomic_json,
    _load_json,
    _payload_sha256,
    _sha256_file,
)

if TYPE_CHECKING:
    from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
        CampaignProfile,
    )


RUNTIME_RECEIPT_SCHEMA = "stage1_jointmoe_benchmark_runtime/v1"
RUNTIME_ESTIMATE_SCHEMA = "stage1_jointmoe_benchmark_runtime_estimate/v1"


def _stratum(model: Mapping[str, Any]) -> tuple[int, str]:
    return int(model["experts"]), str(model["request_set"])


def build_runtime_estimate(
    models: Sequence[Mapping[str, Any]],
    calibration_receipts: Mapping[str, Mapping[str, Any]],
    *,
    campaign_id: str,
    profile_sha256: str,
    policy: str,
    safety_factor: float,
) -> dict[str, Any]:
    """Build a planning estimate; queue delay is intentionally out of scope."""

    if safety_factor < 1.0:
        raise ValueError("runtime-estimate safety factor must be at least one")
    calibration_models = [model for model in models if model["timing_calibration"]]
    missing = sorted(
        str(model["id"])
        for model in calibration_models
        if str(model["id"]) not in calibration_receipts
    )
    if missing:
        raise ValueError(f"missing calibration runtime receipts: {missing}")

    model_counts = Counter(_stratum(model) for model in models)
    calibration_by_stratum: dict[tuple[int, str], Mapping[str, Any]] = {}
    for model in calibration_models:
        model_id = str(model["id"])
        receipt = calibration_receipts[model_id]
        if receipt.get("schema") != RUNTIME_RECEIPT_SCHEMA or receipt.get("status") != "complete":
            raise ValueError(f"incomplete runtime receipt for {model_id}")
        if receipt.get("campaign_id") != campaign_id or receipt.get("model_id") != model_id:
            raise ValueError(f"runtime receipt identity drift for {model_id}")
        if receipt.get("profile", {}).get("sha256") != profile_sha256:
            raise ValueError(f"runtime receipt profile drift for {model_id}")
        timing = receipt.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"runtime receipt timing is missing for {model_id}")
        wall_seconds = float(timing.get("job_wall_seconds", 0))
        gpu_count = int(timing.get("gpu_count", 0))
        gpu_hours = float(timing.get("gpu_hours", -1))
        if wall_seconds <= 0 or gpu_count <= 0:
            raise ValueError(f"runtime receipt timing is invalid for {model_id}")
        if gpu_count != int(model["expert_parallel_size"]):
            raise ValueError(f"runtime receipt GPU topology drift for {model_id}")
        expected_gpu_hours = wall_seconds * gpu_count / 3600.0
        if not math.isclose(gpu_hours, expected_gpu_hours, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"runtime receipt GPU-hour arithmetic drift for {model_id}")
        key = _stratum(model)
        if key in calibration_by_stratum:
            raise ValueError(f"duplicate runtime calibration stratum: {key}")
        calibration_by_stratum[key] = receipt
    if set(calibration_by_stratum) != set(model_counts):
        raise ValueError("runtime calibration does not cover every E/request-set stratum")

    rows: list[dict[str, Any]] = []
    total_wall_seconds = 0.0
    total_gpu_hours = 0.0
    calibration_wall_seconds = 0.0
    calibration_gpu_hours = 0.0
    max_single_job_seconds = 0.0
    for experts, request_set in sorted(model_counts, key=lambda item: (item[0], item[1])):
        receipt = calibration_by_stratum[(experts, request_set)]
        timing = receipt["timing"]
        wall_seconds = float(timing["job_wall_seconds"])
        count = int(model_counts[(experts, request_set)])
        stratum_models = [
            model for model in models if _stratum(model) == (experts, request_set)
        ]
        gpu_counts = Counter(int(model["expert_parallel_size"]) for model in stratum_models)
        stratum_wall_seconds = wall_seconds * count
        stratum_gpu_hours = sum(
            wall_seconds * int(model["expert_parallel_size"]) / 3600.0
            for model in stratum_models
        )
        total_wall_seconds += stratum_wall_seconds
        total_gpu_hours += stratum_gpu_hours
        calibration_wall_seconds += wall_seconds
        calibration_gpu_hours += float(timing["gpu_hours"])
        max_single_job_seconds = max(max_single_job_seconds, wall_seconds)
        rows.append(
            {
                "experts": experts,
                "request_set": request_set,
                "calibration_model_id": receipt["model_id"],
                "models": count,
                "remaining_models": count - 1,
                "calibration_gpu_count": int(timing["gpu_count"]),
                "campaign_gpu_count_distribution": {
                    str(gpu_count): model_count
                    for gpu_count, model_count in sorted(gpu_counts.items())
                },
                "measured_job_wall_seconds": wall_seconds,
                "estimated_stratum_job_hours": stratum_wall_seconds / 3600.0,
                "estimated_stratum_gpu_hours": stratum_gpu_hours,
            }
        )

    remaining_wall_seconds = total_wall_seconds - calibration_wall_seconds
    remaining_gpu_hours = total_gpu_hours - calibration_gpu_hours
    payload: dict[str, Any] = {
        "schema": RUNTIME_ESTIMATE_SCHEMA,
        "status": "complete",
        "campaign_id": campaign_id,
        "profile_sha256": profile_sha256,
        "method": {
            "policy": policy,
            "calibration_ladder": "D",
            "assumption": (
                "D is the largest active model ladder; its measured runtime is reused "
                "within each E/request-set stratum as an extrapolative planning estimate."
            ),
            "safety_factor": safety_factor,
            "queue_time_included": False,
        },
        "counts": {
            "campaign_models": len(models),
            "calibration_models": len(calibration_models),
            "remaining_models": len(models) - len(calibration_models),
        },
        "strata": rows,
        "measured_calibration": {
            "serial_job_hours": calibration_wall_seconds / 3600.0,
            "gpu_hours": calibration_gpu_hours,
        },
        "point_estimate": {
            "serial_job_hours": total_wall_seconds / 3600.0,
            "gpu_hours": total_gpu_hours,
            "remaining_serial_job_hours": remaining_wall_seconds / 3600.0,
            "remaining_gpu_hours": remaining_gpu_hours,
            "unlimited_parallel_compute_floor_hours": max_single_job_seconds / 3600.0,
        },
        "planning_estimate": {
            "serial_job_hours": total_wall_seconds * safety_factor / 3600.0,
            "gpu_hours": total_gpu_hours * safety_factor,
            "remaining_serial_job_hours": (
                remaining_wall_seconds * safety_factor / 3600.0
            ),
            "remaining_gpu_hours": remaining_gpu_hours * safety_factor,
        },
        "interpretation": (
            "Serial job-hours and GPU-hours estimate benchmark execution only. Actual "
            "calendar time depends on scheduler queueing and concurrent A40 availability."
        ),
    }
    payload["estimate_sha256"] = _payload_sha256(payload)
    return payload


def write_runtime_estimate(profile: "CampaignProfile") -> Path:
    receipts: dict[str, Mapping[str, Any]] = {}
    receipt_artifacts: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for model_id in profile.calibration_model_ids:
        path = profile.runtime_receipt(model_id)
        if not path.is_file():
            missing.append(model_id)
            continue
        receipt = _load_json(path, label=f"{model_id} runtime receipt")
        for field in ("score_result", "metrics"):
            artifact = receipt.get(field)
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"runtime receipt does not seal {field} for {model_id}"
                )
            artifact_path = Path(str(artifact.get("path", ""))).resolve()
            if (
                not artifact_path.is_file()
                or _sha256_file(artifact_path) != artifact.get("sha256")
            ):
                raise ValueError(
                    f"runtime receipt {field} hash drift for {model_id}"
                )
        receipts[model_id] = receipt
        receipt_artifacts[model_id] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
    if missing:
        raise FileNotFoundError(
            "runtime estimate requires the complete calibration wave; missing: "
            + ", ".join(missing)
        )

    runtime_contract = profile.payload["runtime_estimation"]
    estimate = build_runtime_estimate(
        profile.payload["models"],
        receipts,
        campaign_id=str(profile.payload["campaign_id"]),
        profile_sha256=profile.source_sha256,
        policy=str(runtime_contract["policy"]),
        safety_factor=float(runtime_contract["safety_factor"]),
    )
    estimate.pop("estimate_sha256")
    estimate["calibration_receipts"] = receipt_artifacts
    estimate["estimate_sha256"] = _payload_sha256(estimate)
    output = profile.output_root / "runtime_estimate.json"
    _atomic_json(output, estimate)
    return output


__all__ = [
    "RUNTIME_ESTIMATE_SCHEMA",
    "RUNTIME_RECEIPT_SCHEMA",
    "build_runtime_estimate",
    "write_runtime_estimate",
]
