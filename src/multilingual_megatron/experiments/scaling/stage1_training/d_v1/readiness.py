"""Check D-specific evidence and promote it into a production runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.stage1_training.d_v1.spec import validate_d_v1_spec


SCHEMA = "stage1_d_sentinel_readiness/v2"
PROMOTION_SCHEMA = "stage1_d_runtime_promotion/v1"


class DReadinessError(ValueError):
    """Raised when D readiness evidence violates the contract."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DReadinessError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DReadinessError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload: Mapping[str, Any], *, field: str = "receipt_sha256") -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop(field, None)
    sealed[field] = _payload_sha256(sealed)
    return sealed


def _verify_seal(
    payload: Mapping[str, Any], *, field: str = "receipt_sha256", label: str
) -> None:
    candidate = dict(payload)
    expected = candidate.pop(field, None)
    if not isinstance(expected, str) or expected != _payload_sha256(candidate):
        raise DReadinessError(f"{label} self-hash mismatch")


def _manifest_without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.pop("manifest_sha256", None)
    return candidate


def _resolve_manifest_path(manifest: Mapping[str, Any], raw: object) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    return (Path(str(manifest["repo_root"])) / path).resolve()


def _verify_manifest(path: Path) -> dict[str, Any]:
    manifest = _load(path)
    expected = str(manifest.get("manifest_sha256", ""))
    if expected != _payload_sha256(_manifest_without_hash(manifest)):
        raise DReadinessError("D probe manifest self-hash mismatch")
    if (
        manifest.get("study_id") != "multilingual_moe_d_v1"
        or manifest.get("production_generation_enabled") is not False
    ):
        raise DReadinessError("D evidence must originate from a probe-only manifest")
    for raw, expected_hash in manifest.get("fixed_input_sha256", {}).items():
        source = _resolve_manifest_path(manifest, raw)
        if not source.is_file() or _sha256(source) != expected_hash:
            raise DReadinessError(f"D probe fixed input changed: {source}")
    return manifest


def _job(manifest: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    rows = [row for row in manifest.get("jobs", []) if row.get("job_id") == job_id]
    if len(rows) != 1:
        raise DReadinessError(f"D probe manifest lacks unique job {job_id}")
    return rows[0]


def _verify_job_receipt(
    manifest: Mapping[str, Any], control_root: Path, job_id: str
) -> dict[str, Any]:
    path = control_root / "receipts" / f"{job_id}.json"
    receipt = _load(path)
    _verify_seal(receipt, label=f"{job_id} receipt")
    if (
        receipt.get("manifest_sha256") != manifest.get("manifest_sha256")
        or receipt.get("job_id") != job_id
        or receipt.get("array_task_id") is not None
        or receipt.get("status") != "complete"
    ):
        raise DReadinessError(f"{job_id} receipt binding/status is invalid")
    expected_outputs = {
        str(_resolve_manifest_path(manifest, raw)): _resolve_manifest_path(manifest, raw)
        for raw in _job(manifest, job_id).get("outputs", [])
    }
    observed = receipt.get("output_sha256")
    if not isinstance(observed, Mapping):
        raise DReadinessError(f"{job_id} receipt lacks output hashes")
    normalized_observed = {
        str(_resolve_manifest_path(manifest, raw)): value
        for raw, value in observed.items()
    }
    if set(normalized_observed) != set(expected_outputs):
        raise DReadinessError(f"{job_id} receipt output set differs from its manifest")
    for key, output in expected_outputs.items():
        if (
            not output.is_file()
            or output.is_symlink()
            or normalized_observed[key] != _sha256(output)
        ):
            raise DReadinessError(f"{job_id} output identity mismatch: {output}")
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _spec_plan(spec_path: Path) -> dict[str, Any]:
    validate_d_v1_spec(spec_path=spec_path)
    spec = _load(spec_path)
    execution = spec["execution_policy"]
    return {
        "schema": SCHEMA,
        "study_id": spec["study_id"],
        "status": "blocked_pending_D_specific_measurements",
        "production_generation_enabled": False,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": _sha256(spec_path),
        "checker_code_sha256": _sha256(Path(__file__).resolve()),
        "model_shape": spec["axes"]["ladders"][0],
        "required_training_sentinels": execution["required_sentinels"],
        "required_long_throughput_probes": execution[
            "required_long_throughput_probes"
        ],
        "required_evaluator_sentinels": execution["required_evaluator_sentinels"],
        "required_fixed_lr_probes": execution["required_fixed_lr_probes"],
        "microbatch_candidates": execution["microbatch_candidates"],
        "storage_gate": execution["required_storage_gate"],
        "long_probe_step_time_cv_limit_pct": execution[
            "long_probe_step_time_cv_limit_pct"
        ],
        "long_probe_step_time_cv_limit_pct_overrides": execution[
            "long_probe_step_time_cv_limit_pct_overrides"
        ],
        "long_probe_p99_to_median_limit_overrides": execution[
            "long_probe_p99_to_median_limit_overrides"
        ],
        "gpu_hour_limit": execution["gpu_hour_limit"],
        "promotion_rule": execution["rule"],
    }


def _long_probe_errors(
    *, plan: Mapping[str, Any], sentinel: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    rows = sentinel.get("sentinels")
    if not isinstance(rows, Mapping):
        return ["sentinel gate lacks sentinel records"]
    for expected in plan["required_long_throughput_probes"]:
        job_id = str(expected["id"])
        row = rows.get(job_id)
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            errors.append(f"{job_id} is missing or incomplete")
            continue
        health = row.get("sentinel_health")
        metrics = health.get("metrics") if isinstance(health, Mapping) else None
        if (
            not isinstance(health, Mapping)
            or health.get("status") != "ok"
            or not isinstance(metrics, Mapping)
        ):
            errors.append(f"{job_id} lacks passing health metrics")
            continue
        try:
            values = {
                "steps": int(metrics["steps_counted"]),
                "tps": float(metrics["median_tokens_per_second"]),
                "mfu": float(metrics["mean_mfu_pct_of_peak"]),
                "cv": float(metrics["step_time_cv_pct"]),
                "median_step_ms": float(metrics["median_step_ms"]),
                "p99_step_ms": float(metrics["p99_step_ms"]),
                "reserved": float(metrics["max_reserved_gb"]),
                "vram": float(metrics["max_vram_used_pct"]),
            }
        except (KeyError, TypeError, ValueError):
            errors.append(f"{job_id} has incomplete throughput/memory metrics")
            continue
        if values["steps"] < 32:
            errors.append(f"{job_id} has fewer than 32 timed iterations")
        if not math.isfinite(values["tps"]) or values["tps"] <= 0:
            errors.append(f"{job_id} throughput is invalid")
        if not math.isfinite(values["mfu"]) or values["mfu"] <= 0:
            errors.append(f"{job_id} MFU is invalid")
        cv_limit = float(
            plan["long_probe_step_time_cv_limit_pct_overrides"].get(
                job_id, plan["long_probe_step_time_cv_limit_pct"]
            )
        )
        if not math.isfinite(values["cv"]) or values["cv"] >= cv_limit:
            errors.append(f"{job_id} step-time CV is not below {cv_limit:g}%")
        p99_to_median_limit = plan[
            "long_probe_p99_to_median_limit_overrides"
        ].get(job_id)
        if p99_to_median_limit is not None:
            ratio = (
                values["p99_step_ms"] / values["median_step_ms"]
                if values["median_step_ms"] > 0
                else math.inf
            )
            if not math.isfinite(ratio) or ratio >= float(p99_to_median_limit):
                errors.append(
                    f"{job_id} p99/median step-time ratio is not below "
                    f"{float(p99_to_median_limit):g}"
                )
        if not math.isfinite(values["reserved"]) or values["reserved"] > 84.0:
            errors.append(f"{job_id} reserved memory exceeds 84 GB")
        if not math.isfinite(values["vram"]) or values["vram"] > 90.0:
            errors.append(f"{job_id} VRAM usage exceeds 90%")
        if float(metrics.get("max_dropped_assignment_rate", 0.0)) != 0.0:
            errors.append(f"{job_id} dropped MoE assignments")
        if float(metrics.get("max_tokens_with_zero_kept_experts", 0.0)) != 0.0:
            errors.append(f"{job_id} has tokens with zero kept experts")
    return errors


def check_evidence(*, spec_path: Path, control_root: Path) -> dict[str, Any]:
    """Check hash-bound D probe, evaluator, LR, storage, and cost evidence."""

    plan = _spec_plan(spec_path)
    errors: list[str] = []
    manifest_path = control_root / "launch_manifest.json"
    try:
        manifest = _verify_manifest(manifest_path)
    except DReadinessError as exc:
        return plan | {
            "control_root": str(control_root.resolve()),
            "status": "blocked",
            "errors": [str(exc)],
        }

    required_jobs = ("sentinel_gate", "fixed_lr_schedule_gate", "storage_preflight")
    receipts: dict[str, Any] = {}
    for job_id in required_jobs:
        try:
            receipts[job_id] = _verify_job_receipt(manifest, control_root, job_id)
        except DReadinessError as exc:
            errors.append(str(exc))

    required_paths = {
        "sentinel_gate": control_root / "sentinel_gate.json",
        "fixed_lr_schedule_gate": control_root / "fixed_lr_schedule_gate.json",
        "storage_preflight": control_root / "storage_preflight.json",
        "gpu_hour_estimate": control_root / "gpu_hour_estimate.json",
    }
    evidence: dict[str, dict[str, Any]] = {}
    for name, path in required_paths.items():
        try:
            evidence[name] = _load(path)
        except DReadinessError as exc:
            errors.append(str(exc))

    sentinel = evidence.get("sentinel_gate", {})
    if sentinel.get("status") != "ok" or sentinel.get("errors"):
        errors.append("sentinel gate is not clean")
    expected_training = {item["id"] for item in plan["required_training_sentinels"]}
    expected_training |= {
        item["id"] for item in plan["required_long_throughput_probes"]
    }
    observed_training = sentinel.get("sentinels", {})
    if not isinstance(observed_training, Mapping) or not expected_training.issubset(
        observed_training
    ):
        errors.append("training sentinel coverage is incomplete")
    observed_eval = sentinel.get("integration_evaluations", {})
    if not isinstance(observed_eval, Mapping) or not set(
        plan["required_evaluator_sentinels"]
    ).issubset(observed_eval):
        errors.append("evaluator sentinel coverage is incomplete")
    errors.extend(_long_probe_errors(plan=plan, sentinel=sentinel))

    fixed_lr = evidence.get("fixed_lr_schedule_gate", {})
    if fixed_lr.get("status") != "ready" or fixed_lr.get("errors"):
        errors.append("fixed-LR gate is not ready")
    expected_lr = {item["id"] for item in plan["required_fixed_lr_probes"]}
    results = fixed_lr.get("results", {})
    if not isinstance(results, Mapping) or not expected_lr.issubset(results):
        errors.append("fixed-LR probe coverage is incomplete")
    elif any(
        not isinstance(results[job_id], Mapping) or not results[job_id]
        for job_id in expected_lr
    ):
        errors.append("fixed-LR probe evidence is empty")

    storage = evidence.get("storage_preflight", {})
    if storage.get("status") != "ok" or storage.get("errors"):
        errors.append("storage preflight is not clean")
    estimate = storage.get("storage_estimate", {})
    if not isinstance(estimate, Mapping):
        errors.append("storage preflight lacks its estimate")
    else:
        for field, minimum in plan["storage_gate"].items():
            if int(estimate.get(field, 0)) < int(minimum):
                errors.append(f"storage field {field} is below the frozen minimum")

    probe_cost = evidence.get("gpu_hour_estimate", {})
    if probe_cost != manifest.get("gpu_hour_estimate"):
        errors.append("probe GPU-hour estimate differs from the launch manifest")
    elif probe_cost.get("go_no_go") != "go":
        errors.append("probe cost forecast is not go")

    return plan | {
        "control_root": str(control_root.resolve()),
        "status": "ready_for_measured_runtime_promotion" if not errors else "blocked",
        "production_generation_enabled": False,
        "errors": errors,
        "probe_manifest": {
            "path": str(manifest_path.resolve()),
            "file_sha256": _sha256(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "git_provenance": manifest.get("git_provenance"),
        },
        "verified_receipts": receipts,
        "evidence_sha256": {
            name: _sha256(path)
            for name, path in required_paths.items()
            if path.is_file()
        },
        "next_step": "promote measured D throughput into a separate production runtime",
    }


def validate_runtime_promotion(*, runtime_path: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = _load(receipt_path)
    _verify_seal(receipt, label="D runtime promotion")
    if receipt.get("schema") != PROMOTION_SCHEMA or receipt.get("status") != "ready":
        raise DReadinessError("D runtime promotion is not ready")
    if (
        Path(str(receipt.get("runtime_path", ""))).resolve() != runtime_path.resolve()
        or receipt.get("runtime_file_sha256") != _sha256(runtime_path)
    ):
        raise DReadinessError("D runtime promotion does not bind this runtime file")
    if float(receipt.get("estimated_gpu_hours", math.inf)) > float(
        receipt.get("gpu_hour_limit", -1)
    ):
        raise DReadinessError("promoted D runtime exceeds its GPU-hour limit")
    return receipt


def _cost_only_spec_amendment(
    *, launch_spec: Mapping[str, Any], amended_spec: Mapping[str, Any]
) -> tuple[float, float, str]:
    """Require an amendment to change only the D cost ceiling and its rationale."""

    launch = json.loads(json.dumps(launch_spec))
    amended = json.loads(json.dumps(amended_spec))
    try:
        launch_execution = launch["execution_policy"]
        amended_execution = amended["execution_policy"]
        base_limit = float(launch_execution.pop("gpu_hour_limit"))
        amended_limit = float(amended_execution.pop("gpu_hour_limit"))
        launch_execution.pop("gpu_hour_limit_amendment")
        rationale = str(amended_execution.pop("gpu_hour_limit_amendment")).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise DReadinessError("D cost amendment fields are malformed") from exc
    if launch != amended:
        raise DReadinessError(
            "D cost amendment changes fields beyond the GPU-hour limit and rationale"
        )
    if (
        not math.isfinite(base_limit)
        or not math.isfinite(amended_limit)
        or amended_limit <= base_limit
        or not rationale
    ):
        raise DReadinessError("D cost amendment limit/rationale is invalid")
    return base_limit, amended_limit, rationale


def amend_cost_promotion(
    *,
    spec_path: Path,
    control_root: Path,
    runtime_output: Path,
    blocked_promotion_path: Path,
    promotion_output: Path,
    production_output_root: Path,
) -> dict[str, Any]:
    """Promote a measured D runtime after a cost-only pre-production amendment."""

    readiness = _load(control_root / "d_v1_readiness.json")
    _verify_seal(readiness, label="D readiness")
    if (
        readiness.get("status") != "ready_for_measured_runtime_promotion"
        or readiness.get("errors")
    ):
        raise DReadinessError("D readiness receipt is not clean")

    manifest_path = control_root / "launch_manifest.json"
    manifest = _load(manifest_path)
    expected_manifest_sha = str(manifest.get("manifest_sha256", ""))
    if expected_manifest_sha != _payload_sha256(_manifest_without_hash(manifest)):
        raise DReadinessError("D probe manifest self-hash mismatch")
    bound_manifest = readiness.get("probe_manifest", {})
    if (
        not isinstance(bound_manifest, Mapping)
        or bound_manifest.get("manifest_sha256") != expected_manifest_sha
        or bound_manifest.get("file_sha256") != _sha256(manifest_path)
    ):
        raise DReadinessError("D readiness does not bind the probe manifest")

    blocked = _load(blocked_promotion_path)
    _verify_seal(blocked, label="prior D runtime promotion")
    if (
        blocked.get("status") not in {"blocked_cost_exceeds_limit", "ready"}
        or Path(str(blocked.get("runtime_path", ""))).resolve()
        != runtime_output.resolve()
        or blocked.get("runtime_file_sha256") != _sha256(runtime_output)
    ):
        raise DReadinessError("prior D promotion does not bind the measured runtime")

    git_provenance = manifest.get("git_provenance", {})
    launch_repo = git_provenance.get("orchestration_repo", {})
    launch_commit = str(launch_repo.get("commit", "")).strip()
    repo_root = Path(str(manifest.get("repo_root", ""))).resolve()
    launch_spec_path = Path(str(manifest.get("spec_path", ""))).resolve()
    try:
        relative_spec = launch_spec_path.relative_to(repo_root)
    except ValueError as exc:
        raise DReadinessError("D launch spec is outside its repository") from exc
    if not launch_commit:
        raise DReadinessError("D launch manifest lacks its source commit")
    try:
        launch_spec_bytes = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "show",
                f"{launch_commit}:{relative_spec.as_posix()}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        launch_spec = json.loads(launch_spec_bytes)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise DReadinessError("cannot recover the launch-time D spec") from exc
    launch_spec_sha = hashlib.sha256(launch_spec_bytes).hexdigest()
    fixed = manifest.get("fixed_input_sha256", {})
    expected_launch_spec_sha = fixed.get(str(launch_spec_path))
    if expected_launch_spec_sha is None:
        expected_launch_spec_sha = fixed.get(relative_spec.as_posix())
    if launch_spec_sha != expected_launch_spec_sha:
        raise DReadinessError("launch-time D spec hash differs from the manifest")

    validate_d_v1_spec(spec_path=spec_path)
    amended_spec = _load(spec_path)
    base_limit, amended_limit, rationale = _cost_only_spec_amendment(
        launch_spec=launch_spec,
        amended_spec=amended_spec,
    )
    prior_limit = float(blocked.get("gpu_hour_limit", math.nan))
    if prior_limit not in {base_limit, amended_limit} or blocked.get(
        "spec_sha256"
    ) != launch_spec_sha:
        raise DReadinessError("prior D promotion does not match the cost contract")

    runtime = _load(runtime_output)
    if float(runtime.get("gpu_hour_limit", math.nan)) not in {
        base_limit,
        amended_limit,
    }:
        raise DReadinessError("measured D runtime uses an unrelated cost ceiling")
    runtime["gpu_hour_limit"] = amended_limit
    runtime["base_gpu_hour_limit"] = base_limit
    runtime["gpu_hour_limit_amendment"] = {
        "kind": "preproduction_cost_only",
        "launch_spec_sha256": launch_spec_sha,
        "amended_spec_sha256": _sha256(spec_path),
        "rationale": rationale,
    }
    runtime["production_promotion_receipt_path"] = str(promotion_output.resolve())
    write_json(runtime_output, runtime)

    cost_amendment = {
        "kind": "preproduction_cost_only",
        "prior_promotion_path": str(blocked_promotion_path.resolve()),
        "prior_promotion_file_sha256": _sha256(blocked_promotion_path),
        "prior_promotion_status": blocked["status"],
        "prior_promotion_gpu_hour_limit": prior_limit,
        "launch_commit": launch_commit,
        "launch_spec_sha256": launch_spec_sha,
        "amended_spec_sha256": _sha256(spec_path),
        "rationale": rationale,
    }
    write_json(
        promotion_output,
        _seal(
            {
                "schema": PROMOTION_SCHEMA,
                "status": "pending_cost_validation",
                "spec_sha256": _sha256(spec_path),
                "probe_manifest": readiness["probe_manifest"],
                "readiness_receipt_sha256": readiness["receipt_sha256"],
                "runtime_path": str(runtime_output.resolve()),
                "runtime_file_sha256": _sha256(runtime_output),
                "gpu_hour_limit": amended_limit,
                "base_gpu_hour_limit": base_limit,
                "cost_amendment": cost_amendment,
                "throughput_reports": blocked.get("throughput_reports", {}),
            }
        ),
    )

    from multilingual_megatron.experiments.scaling.stage1_training.d_v1 import workflow as d_v1_workflow

    production_manifest = d_v1_workflow.build_manifest(
        spec_path=spec_path,
        runtime_path=runtime_output,
        output_root=production_output_root,
        skip_promotion_validation=True,
    )
    estimated = float(
        production_manifest["gpu_hour_estimate"]["worst_case_estimated_gpu_hours"]
    )
    status = "ready" if estimated <= amended_limit else "blocked_cost_exceeds_limit"
    receipt = _seal(
        {
            "schema": PROMOTION_SCHEMA,
            "status": status,
            "spec_sha256": _sha256(spec_path),
            "probe_manifest": readiness["probe_manifest"],
            "readiness_receipt_sha256": readiness["receipt_sha256"],
            "runtime_path": str(runtime_output.resolve()),
            "runtime_file_sha256": _sha256(runtime_output),
            "gpu_hour_limit": amended_limit,
            "base_gpu_hour_limit": base_limit,
            "estimated_gpu_hours": estimated,
            "cost_amendment": cost_amendment,
            "throughput_reports": blocked.get("throughput_reports", {}),
            "storage_estimate": production_manifest["storage_estimate"],
            "production_role_counts": production_manifest["role_counts"],
            "production_waves": production_manifest.get("production_waves", []),
        }
    )
    write_json(promotion_output, receipt)
    return receipt


def _write_throughput_report(
    *, output: Path, global_batch_size: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "global_batch_size": global_batch_size,
        "rows": rows,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    write_json(output, payload)
    return {
        "report": str(output.resolve()),
        "file_sha256": _sha256(output),
        "payload_sha256": payload["payload_sha256"],
        "contract": (
            f"all four D cells; GBS{global_batch_size}; 134,217,728 tokens; "
            "32-step final window; ordinary FP32 optimizer moments"
        ),
    }


def promote_runtime(
    *,
    spec_path: Path,
    control_root: Path,
    runtime_output: Path,
    promotion_output: Path,
    production_output_root: Path,
) -> dict[str, Any]:
    readiness = check_evidence(spec_path=spec_path, control_root=control_root)
    if readiness["status"] != "ready_for_measured_runtime_promotion":
        raise DReadinessError("D evidence is not ready: " + "; ".join(readiness["errors"]))
    manifest = _load(control_root / "launch_manifest.json")
    gpu_hour_limit = float(readiness["gpu_hour_limit"])
    probe_runtime_path = Path(str(manifest["runtime_path"]))
    from multilingual_megatron.experiments.scaling.ab_v2.common import _load_object

    probe_runtime = _load_object(probe_runtime_path)
    sentinel = _load(control_root / "sentinel_gate.json")
    sentinel_rows = sentinel["sentinels"]
    reports: dict[str, Any] = {}
    throughput_matrix: dict[str, Any] = {}
    runtime_output.parent.mkdir(parents=True, exist_ok=True)
    report_root = promotion_output.parent / "throughput"
    report_root.mkdir(parents=True, exist_ok=True)
    for global_batch_size in (32, 64, 128):
        rows: list[dict[str, Any]] = []
        values: dict[str, float] = {}
        for experts in (1, 4, 16, 32):
            job_id = (
                f"sentinel_long_gbs{global_batch_size}_d_e{experts}_"
                f"k050_gbs{global_batch_size}"
            )
            health = sentinel_rows[job_id]["sentinel_health"]
            metrics = health["metrics"]
            value = float(metrics["median_tokens_per_second"])
            values[str(experts)] = value
            rows.append(
                {
                    "status": "complete",
                    "job_id": job_id,
                    "ladder": "D",
                    "experts": experts,
                    "GBS": global_batch_size,
                    "median_tokens_per_second": value,
                    "mean_mfu_pct_of_peak": float(metrics["mean_mfu_pct_of_peak"]),
                    "step_time_cv_pct": float(metrics["step_time_cv_pct"]),
                    "max_reserved_gb": float(metrics["max_reserved_gb"]),
                    "max_vram_used_pct": float(metrics["max_vram_used_pct"]),
                    "source_log_sha256": health["log_sha256"],
                }
            )
        throughput_matrix[str(global_batch_size)] = {"D": values}
        report_path = report_root / f"d_gbs{global_batch_size}_long_report.json"
        reports[f"GBS{global_batch_size}"] = _write_throughput_report(
            output=report_path,
            global_batch_size=global_batch_size,
            rows=rows,
        )

    integration = sentinel["integration_evaluations"]
    evaluation_tps = {
        str(experts): float(integration[f"sentinel_eval_d_e{experts}_eng"]["tokens_per_second"])
        for experts in (1, 4, 16, 32)
    }
    setup_seconds = max(float(row["setup_seconds"]) for row in integration.values())
    residual_seconds = max(
        float(row["residual_per_language_seconds"]) for row in integration.values()
    )

    runtime = dict(probe_runtime)
    runtime.update(
        {
            "study_id": "multilingual_moe_d_v1",
            "output_root": str(production_output_root),
            "workflow_profile_name": "d_v1_study_phase1_job",
            "workflow_attempt_id": "d_v1_study_phase1_20260723",
            "workflow_tag": "multilingual-moe-d-v1-study-phase1-20260723",
            "workflow_job_prefix": "dv1p1",
            "probe_only_unmeasured": False,
            "production_promotion_receipt_path": str(promotion_output.resolve()),
            "throughput_tokens_per_second": {
                "D": dict(throughput_matrix["128"]["D"])
            },
            "training_throughput_tokens_per_second_by_global_batch_size": throughput_matrix,
            "evaluation_throughput_tokens_per_second": {"D": evaluation_tps},
            "performance_budget": {
                "training_noncheckpoint_overhead_seconds_per_job": 120.0,
                "checkpoint_overhead_seconds": {"D": 90.0},
                "evaluation_setup_seconds": {"D": setup_seconds},
                "evaluation_residual_seconds_per_language": {"D": residual_seconds},
                "evidence": "D-specific study_phase1 long probes and evaluator sentinels",
                "training_throughput_evidence": reports,
            },
            "sentinel_throughput_safety_factor": 0.90,
            "gpu_hour_limit": gpu_hour_limit,
        }
    )
    runtime.pop("inherits", None)
    write_json(runtime_output, runtime)

    provisional = _seal(
        {
            "schema": PROMOTION_SCHEMA,
            "status": "pending_cost_validation",
            "spec_sha256": readiness["spec_sha256"],
            "probe_manifest": readiness["probe_manifest"],
            "readiness_receipt_sha256": _payload_sha256(readiness),
            "runtime_path": str(runtime_output.resolve()),
            "runtime_file_sha256": _sha256(runtime_output),
            "gpu_hour_limit": gpu_hour_limit,
            "throughput_reports": reports,
        }
    )
    write_json(promotion_output, provisional)

    from multilingual_megatron.experiments.scaling.stage1_training.d_v1 import workflow as d_v1_workflow

    production_manifest = d_v1_workflow.build_manifest(
        spec_path=spec_path,
        runtime_path=runtime_output,
        output_root=production_output_root,
        skip_promotion_validation=True,
    )
    estimated = float(
        production_manifest["gpu_hour_estimate"]["worst_case_estimated_gpu_hours"]
    )
    status = (
        "ready"
        if estimated <= gpu_hour_limit
        else "blocked_cost_exceeds_limit"
    )
    receipt = _seal(
        {
            **{key: value for key, value in provisional.items() if key != "receipt_sha256"},
            "status": status,
            "runtime_file_sha256": _sha256(runtime_output),
            "estimated_gpu_hours": estimated,
            "storage_estimate": production_manifest["storage_estimate"],
            "production_role_counts": production_manifest["role_counts"],
            "production_waves": production_manifest.get("production_waves", []),
        }
    )
    write_json(promotion_output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "promote", "amend-cost"))
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/scaling_law/multilingual_moe_d_v1/study_spec.json"),
    )
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--production-output-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = _spec_plan(args.spec)
    elif args.command == "check":
        if args.control_root is None:
            raise SystemExit("check requires --control-root")
        payload = check_evidence(spec_path=args.spec, control_root=args.control_root)
    elif args.command == "promote":
        if (
            args.control_root is None
            or args.runtime_output is None
            or args.production_output_root is None
        ):
            raise SystemExit(
                "promote requires --control-root, --runtime-output, and --production-output-root"
            )
        payload = promote_runtime(
            spec_path=args.spec,
            control_root=args.control_root,
            runtime_output=args.runtime_output,
            promotion_output=args.output,
            production_output_root=args.production_output_root,
        )
    else:
        if (
            args.control_root is None
            or args.runtime_output is None
            or args.production_output_root is None
        ):
            raise SystemExit(
                "amend-cost requires --control-root, --runtime-output, and "
                "--production-output-root; --output is the amended receipt"
            )
        blocked_path = args.output.with_name("runtime_promotion.json")
        payload = amend_cost_promotion(
            spec_path=args.spec,
            control_root=args.control_root,
            runtime_output=args.runtime_output,
            blocked_promotion_path=blocked_path,
            promotion_output=args.output,
            production_output_root=args.production_output_root,
        )
    write_json(args.output, _seal(payload) if "receipt_sha256" not in payload else payload)
    print(f"Wrote D readiness {args.command} to {args.output}: {payload['status']}")
    if args.command == "plan":
        return 0
    return 0 if payload["status"] in {"ready_for_measured_runtime_promotion", "ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DReadinessError",
    "amend_cost_promotion",
    "check_evidence",
    "promote_runtime",
    "validate_runtime_promotion",
]
