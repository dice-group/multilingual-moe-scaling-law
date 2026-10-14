"""Array execution and fail-closed scientific gates for the static DAG."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..campaign import audit_grid, scientific_result
from ..data import sha256_file
from ..execution import execute_task, load_manifest, read_json, reduce, result_path, write_json_once
from .contracts import (
    FINAL_SCHEMA,
    FOLDS,
    GATE_SCHEMA,
    BaselineRefitError,
    gate_path,
    job_by_id,
    load_workflow,
    seal,
    verify_seal,
)
from .planning import (
    manifest_path_for_job,
    plan_m1,
    plan_m2,
    plan_robust,
    write_index,
)
from .reporting import build_and_write_reports


def _binding(path: Path, embedded_field: str) -> dict[str, str]:
    payload = read_json(path)
    embedded = payload.get(embedded_field)
    if not isinstance(embedded, str):
        raise BaselineRefitError(f"{path} has no {embedded_field}")
    return {"path": str(path), "file_sha256": sha256_file(path), embedded_field: embedded}


def _read_gate(workflow: Mapping[str, Any], gate_id: str) -> Mapping[str, Any] | None:
    path = gate_path(workflow, gate_id)
    if not path.exists():
        return None
    payload = read_json(path)
    verify_seal(payload, "gate_sha256", label=gate_id)
    if payload.get("schema_version") != GATE_SCHEMA \
            or payload.get("workflow_sha256") != workflow["workflow_sha256"] \
            or payload.get("gate_id") != gate_id:
        raise BaselineRefitError(f"{gate_id} receipt belongs to another workflow")
    return payload


def _require_gate(workflow: Mapping[str, Any], gate_id: str) -> Mapping[str, Any]:
    payload = _read_gate(workflow, gate_id)
    if payload is None or payload.get("status") != "complete":
        raise BaselineRefitError(f"required gate is incomplete: {gate_id}")
    return payload


def _write_gate(
    workflow: Mapping[str, Any], gate_id: str, *, status: str,
    details: Mapping[str, Any], error: BaseException | None = None,
) -> Mapping[str, Any]:
    unsigned = {
        "schema_version": GATE_SCHEMA,
        "workflow_sha256": workflow["workflow_sha256"],
        "gate_id": gate_id,
        "status": status,
        "details": dict(details),
        "error": None if error is None else {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    payload = seal(unsigned, "gate_sha256")
    write_json_once(gate_path(workflow, gate_id), payload)
    return payload


def _reduce_and_audit(workflow: Mapping[str, Any], job_ids: Sequence[str]) -> dict[str, Any]:
    results = {}
    for job_id in job_ids:
        manifest_path = manifest_path_for_job(workflow, job_id)
        manifest = load_manifest(manifest_path)
        reduced = reduce(manifest)
        audit = audit_grid(manifest_path)
        if audit.get("status") != "complete":
            followup = audit.get("followup_manifest")
            suffix = "" if followup is None else f"; follow-up manifest: {followup}"
            raise BaselineRefitError(
                f"{job_id} audit status is {audit.get('status')}{suffix}"
            )
        result_file = result_path(manifest)
        scientific = scientific_result(result_file)
        if scientific["result_sha256"] != reduced["result_sha256"]:
            raise BaselineRefitError(f"{job_id} reduced result changed during audit")
        results[job_id] = {
            "path": str(result_file),
            "file_sha256": sha256_file(result_file),
            "result_sha256": scientific["result_sha256"],
            "audit": _binding(
                Path(str(manifest["execution"]["output_root"])) / "audits"
                    / f"{manifest['manifest_sha256'][:20]}.json",
                "audit_sha256",
            ),
        }
    return results


def _scale_gate(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    results = _reduce_and_audit(workflow, [f"grid.m0_ols.{fold}" for fold in FOLDS])
    path = write_index(workflow, name="robust", manifests=plan_robust(workflow))
    return {"results": results, "next_index": _binding(path, "index_sha256")}


def _m0_gate(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    results = _reduce_and_audit(workflow, [f"grid.robust.M0.{fold}" for fold in FOLDS])
    path = write_index(workflow, name="m1", manifests=plan_m1(workflow))
    return {"results": results, "next_index": _binding(path, "index_sha256")}


def _baseline_gate(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    jobs = [f"grid.robust.{model}.{fold}" for model in ("C0", "J0") for fold in FOLDS]
    return {"results": _reduce_and_audit(workflow, jobs)}


def _m1_gate(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    results = _reduce_and_audit(workflow, [f"grid.m1.{fold}" for fold in FOLDS])
    path = write_index(workflow, name="m2", manifests=plan_m2(workflow))
    return {"results": results, "next_index": _binding(path, "index_sha256")}


def _final_gate(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    for gate_id in ("gate.scale", "gate.m0", "gate.baselines", "gate.m1"):
        _require_gate(workflow, gate_id)
    results = _reduce_and_audit(workflow, [f"grid.m2.{fold}" for fold in FOLDS])
    reports = build_and_write_reports(workflow)
    return {"results": results, "reports": reports}


GATES: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
    "gate.scale": _scale_gate,
    "gate.m0": _m0_gate,
    "gate.baselines": _baseline_gate,
    "gate.m1": _m1_gate,
    "gate.final": _final_gate,
}


def _finalize(workflow: Mapping[str, Any], gate: Mapping[str, Any]) -> Mapping[str, Any]:
    reports = gate["details"]["reports"]
    unsigned = {
        "schema_version": FINAL_SCHEMA,
        "workflow_sha256": workflow["workflow_sha256"],
        "status": "complete",
        "strict_baseline_win": bool(reports["strict_baseline_win"]),
        "final_gate": {
            "path": str(gate_path(workflow, "gate.final")),
            "file_sha256": sha256_file(gate_path(workflow, "gate.final")),
            "gate_sha256": gate["gate_sha256"],
        },
        "reports": reports,
    }
    payload = seal(unsigned, "receipt_sha256")
    write_json_once(Path(str(workflow["output_root"])) / "receipts" / "finalization.json", payload)
    return payload


def execute_gate(workflow: Mapping[str, Any], gate_id: str) -> Mapping[str, Any]:
    existing = _read_gate(workflow, gate_id)
    if existing is not None:
        if existing.get("status") == "complete":
            if gate_id == "gate.final":
                final_path = Path(str(workflow["output_root"])) / "receipts" / "finalization.json"
                if not final_path.exists():
                    _finalize(workflow, existing)
            return existing
        raise BaselineRefitError(f"{gate_id} is already blocked: {existing.get('error')}")
    handler = GATES.get(gate_id)
    if handler is None:
        raise BaselineRefitError(f"unknown gate: {gate_id}")
    try:
        details = handler(workflow)
    except Exception as error:
        _write_gate(workflow, gate_id, status="blocked", details={}, error=error)
        raise
    receipt = _write_gate(workflow, gate_id, status="complete", details=details)
    if gate_id == "gate.final":
        _finalize(workflow, receipt)
    return receipt


def execute_job(
    workflow_manifest: Path, job_id: str, *, task_index: int | None = None,
) -> Mapping[str, Any]:
    workflow = load_workflow(workflow_manifest)
    job = job_by_id(workflow, job_id)
    if job["kind"] == "gate":
        if task_index is not None:
            raise BaselineRefitError("gate jobs do not accept a task index")
        return execute_gate(workflow, job_id)
    if task_index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise BaselineRefitError("array job needs --task-index or SLURM_ARRAY_TASK_ID")
        task_index = int(value)
    if not 0 <= int(task_index) < int(job["task_count"]):
        raise BaselineRefitError(f"task index is outside {job_id}")
    manifest = load_manifest(manifest_path_for_job(workflow, job_id))
    return execute_task(manifest, int(task_index))
