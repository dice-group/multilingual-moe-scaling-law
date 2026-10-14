"""Versioned, immutable remediation for a failed nested-warm grid audit."""
from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from .data import canonical_json_sha256, sha256_file
from .execution import (
    ExecutionError,
    _bounds,
    _direct_fit_record,
    _fit_record,
    _point,
    load_manifest,
    load_problem,
    read_json,
    require_clean_definitive_provenance,
    result_path,
    runtime_provenance,
    validate_recorded_provenance,
    write_json_once,
    write_text_once,
)
from .fit import (
    OBJECTIVE_RELATIVE_TOLERANCE,
    PREDICTION_ABSOLUTE_TOLERANCE,
    SharedParameters,
    StartPoint,
    direct_joint_check,
    equivalence_check,
    nested_warm_start,
    run_projected_start,
)

PLAN_SCHEMA = "stage1_fit_warm_audit_amendment_plan/v1"
RUN_SCHEMA = "stage1_fit_warm_audit_amendment_run/v1"
ACCEPTANCE_SCHEMA = "stage1_fit_warm_audit_amendment_acceptance/v1"
PROTOCOL = "stage1_fit_nested_warm_retry/v1"
STRICT_FTOL = 1e-14
STRICT_GTOL = 1e-8
STRICT_MAX_ITERATIONS = 20_000


class AmendmentError(RuntimeError):
    pass


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload); result[field] = canonical_json_sha256(payload); return result


def _sealed(path: Path, field: str, schema: str) -> Mapping[str, Any]:
    payload = read_json(path); unsigned = dict(payload); digest = unsigned.pop(field, None)
    if payload.get("schema_version") != schema or digest != canonical_json_sha256(unsigned):
        raise AmendmentError(f"invalid {schema} artifact: {path}")
    return payload


def _binding(path: Path, embedded_field: str) -> dict[str, Any]:
    payload = read_json(path); embedded = payload.get(embedded_field)
    if not isinstance(embedded, str):
        raise AmendmentError(f"{path} lacks {embedded_field}")
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), embedded_field: embedded}


def _verify_binding(binding: Mapping[str, Any], embedded_field: str) -> Path:
    if set(binding) != {"path", "file_sha256", embedded_field}:
        raise AmendmentError("artifact binding is malformed")
    path = Path(str(binding["path"]))
    if sha256_file(path) != binding["file_sha256"] or read_json(path).get(embedded_field) != binding[embedded_field]:
        raise AmendmentError(f"bound artifact changed: {path}")
    return path


def accepted_path(result: Mapping[str, Any], *, output_root: Path) -> Path:
    return output_root / "audit_amendments" / "accepted" / f"{str(result['manifest_sha256'])[:20]}.json"


def _run_path(plan: Mapping[str, Any]) -> Path:
    return Path(str(plan["output_root"])) / "audit_amendments" / "runs" / f"{plan['plan_sha256'][:20]}.json"


def _validate_result_manifest_binding(*, manifest_path: Path, manifest: Mapping[str, Any],
    result_path_: Path, result: Mapping[str, Any]) -> None:
    """Prove that one sealed reduced result was emitted by the bound manifest."""
    identity = {key: manifest["fit"][key] for key in
        ("model", "fold_id", "replicate_id", "objective", "bound_version")}
    if (result.get("manifest_sha256") != manifest.get("manifest_sha256")
            or Path(str(result.get("manifest_path"))).resolve() != manifest_path.resolve()
            or result.get("fit_rows_content_sha256") != manifest.get("input", {}).get("fit_rows_content_sha256")
            or result.get("fit_identity") != identity
            or result.get("fit_bounds") != manifest["fit"]["bounds"]
            or result.get("membership") != manifest.get("membership")
            or result.get("parents") != manifest.get("parents")
            or result.get("selection_report") != manifest.get("selection_report")
            or result.get("review_only") != manifest.get("review_only")
            or result.get("status") != "reduced" or not isinstance(result.get("selected"), Mapping)):
        raise AmendmentError(f"reduced result does not belong to its bound manifest: {result_path_}")
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != int(manifest["fit"]["start_count"]):
        raise AmendmentError("reduced result does not cover its manifest start set")
    receipt_path = Path(str(result.get("receipt_path")))
    receipt = _sealed(receipt_path, "receipt_sha256", "stage1_fit_complete/v2")
    successful = sum(isinstance(item, Mapping) and item.get("success") is True for item in attempts)
    if (result.get("receipt_sha256") != receipt["receipt_sha256"]
            or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
            or receipt.get("start_count") != len(attempts)
            or receipt.get("successful_starts") != successful or successful < 1):
        raise AmendmentError("completion receipt does not prove the bound result coverage")


def _validate_nested_warm_input(*, grid_manifest: Mapping[str, Any], grid_result_path: Path,
    grid_result: Mapping[str, Any], warm_manifest: Mapping[str, Any], warm_result: Mapping[str, Any]) -> None:
    """Recompute and bind the sole admissible nested warm start."""
    source_bindings = [item for item in grid_manifest.get("parents", [])
        if isinstance(item, Mapping) and item.get("kind") == "warm"]
    if len(source_bindings) != 1:
        raise AmendmentError("nested-model grid lacks one exact warm-source result")
    source_binding = source_bindings[0]; source_path = Path(str(source_binding.get("path")))
    if (sha256_file(source_path) != source_binding.get("file_sha256")
            or read_json(source_path).get("result_sha256") != source_binding.get("result_sha256")):
        raise AmendmentError("grid warm-source binding changed")
    from .campaign import scientific_result
    source = scientific_result(source_path)
    shared = source.get("selected", {}).get("shared_parameters")
    if not isinstance(shared, Mapping):
        raise AmendmentError("grid warm-source result lacks shared parameters")
    expected = nested_warm_start(grid_manifest["fit"]["model"], SharedParameters(**shared))
    explicit = warm_manifest.get("fit", {}).get("explicit_starts")
    if (warm_manifest.get("fit", {}).get("action") != "warm"
            or warm_manifest.get("fit", {}).get("start_mode") != "explicit"
            or warm_manifest.get("fit", {}).get("start_count") != 1
            or not isinstance(explicit, list) or len(explicit) != 1):
        raise AmendmentError("default warm artifact is not the exact one-start check")
    actual = explicit[0]; values = np.asarray(actual.get("values"), dtype=np.float64)
    if (set(actual) != {"start_id", "values", "kind"}
            or actual.get("start_id") != expected.start_id or actual.get("kind") != expected.kind
            or values.shape != expected.values.shape or not np.array_equal(values, expected.values)):
        raise AmendmentError("default warm manifest is not the derived nested warm start")
    invariant_fit = ("model", "fold_id", "objective", "delta", "bounds", "bound_version",
        "replicate_id", "outer_ftol", "outer_gtol", "outer_max_iterations", "grid_version")
    if (any(warm_manifest["fit"].get(key) != grid_manifest["fit"].get(key) for key in invariant_fit)
            or warm_manifest.get("input") != grid_manifest.get("input")
            or warm_manifest.get("membership") != grid_manifest.get("membership")
            or warm_manifest.get("selection_report") != grid_manifest.get("selection_report")):
        raise AmendmentError("default warm manifest changes the failed grid contract")
    grid_parent = {"kind": "grid", "path": str(grid_result_path.resolve()),
        "file_sha256": sha256_file(grid_result_path), "result_sha256": grid_result["result_sha256"]}
    actual_grid = [item for item in warm_manifest.get("parents", [])
        if isinstance(item, Mapping) and item.get("kind") == "grid"]
    if (actual_grid != [grid_parent]
            or [item for item in warm_manifest.get("parents", [])
                if isinstance(item, Mapping) and item.get("kind") != "grid"]
                != list(grid_manifest.get("parents", []))):
        raise AmendmentError("default warm manifest does not preserve the failed grid lineage")
    selected = warm_result.get("selected", {})
    attempts = warm_result.get("attempts", [])
    if (selected.get("start_id") != expected.start_id or len(attempts) != 1
            or attempts[0].get("start_id") != expected.start_id):
        raise AmendmentError("default warm result was not produced from the derived nested warm start")


def plan_warm_audit_amendment(*, grid_manifest_path: Path, warm_manifest_path: Path,
    profile: Mapping[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    """Freeze a conditional strict retry without modifying the historical fit."""
    from .campaign import AUDIT_SCHEMA, _completion_receipt, _reduced, audit_path

    grid_manifest = load_manifest(grid_manifest_path, require_current_provenance=False)
    warm_manifest = load_manifest(warm_manifest_path, require_current_provenance=False)
    grid_result_path = result_path(grid_manifest); warm_result_path = result_path(warm_manifest)
    grid_result = _reduced(grid_result_path); warm_result = _reduced(warm_result_path)
    _completion_receipt(grid_result); _completion_receipt(warm_result)
    _validate_result_manifest_binding(manifest_path=grid_manifest_path, manifest=grid_manifest,
        result_path_=grid_result_path, result=grid_result)
    _validate_result_manifest_binding(manifest_path=warm_manifest_path, manifest=warm_manifest,
        result_path_=warm_result_path, result=warm_result)
    _validate_nested_warm_input(grid_manifest=grid_manifest, grid_result_path=grid_result_path,
        grid_result=grid_result,
        warm_manifest=warm_manifest, warm_result=warm_result)
    grid_audit_path = audit_path(grid_result, output_root=grid_result_path.parent.parent)
    grid_audit = _sealed(grid_audit_path, "audit_sha256", AUDIT_SCHEMA)
    if grid_audit.get("status") != "failed" or grid_audit.get("reason") != "nested warm start disagrees":
        raise AmendmentError("amendment requires exactly a failed nested-warm audit")
    if grid_manifest["fit"]["model"] not in {"M1", "M2"} or grid_manifest["fit"]["action"] != "grid":
        raise AmendmentError("amendment target is not a definitive nested-model grid")
    parent_grid = [item for item in warm_manifest["parents"] if item.get("kind") == "grid"]
    if len(parent_grid) != 1 or parent_grid[0].get("result_sha256") != grid_result["result_sha256"]:
        raise AmendmentError("default warm artifact does not bind the failed grid")
    if grid_result.get("selected", {}).get("boundary_contacts"):
        raise AmendmentError("warm remediation cannot bypass a boundary failure")
    required_profile = {"partition", "account", "reservation", "memory", "time_limit", "blas_threads",
        "python_executable"}
    if (not required_profile <= set(profile) or int(profile["blas_threads"]) != 1
            or not isinstance(profile["python_executable"], str) or not profile["python_executable"].strip()):
        raise AmendmentError("audit-amendment profile is incomplete or not single-threaded")
    provenance = runtime_provenance()
    require_clean_definitive_provenance(provenance, review_only=False)
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "output_root": str(grid_result_path.parent.parent.resolve()),
        "grid_manifest": _binding(grid_manifest_path, "manifest_sha256"),
        "grid_result": _binding(grid_result_path, "result_sha256"),
        "failed_audit": _binding(grid_audit_path, "audit_sha256"),
        "default_warm_manifest": _binding(warm_manifest_path, "manifest_sha256"),
        "default_warm_result": _binding(warm_result_path, "result_sha256"),
        "thresholds": {"relative_objective": OBJECTIVE_RELATIVE_TOLERANCE,
            "maximum_absolute_prediction": PREDICTION_ABSOLUTE_TOLERANCE},
        "strict_retry": {"ftol": STRICT_FTOL, "gtol": STRICT_GTOL,
            "maximum_iterations": STRICT_MAX_ITERATIONS},
        "actions": ["strict_warm_retry", "three_start_basin", "projected_direct"],
        "execution": {key: profile[key] for key in sorted(required_profile)},
        "provenance": provenance,
    }
    plan = _seal(unsigned, "plan_sha256")
    root = Path(plan["output_root"]); plan_path = root / "audit_amendments" / "plans" / f"{plan['plan_sha256'][:20]}.json"
    write_json_once(plan_path, plan)
    script = _render_slurm(plan_path, plan)
    script_path = root / "audit_amendments" / "plans" / f"slurm-{plan['plan_sha256'][:20]}.sbatch"
    write_text_once(script_path, script)
    return plan, plan_path, script_path


def _render_slurm(plan_path: Path, plan: Mapping[str, Any]) -> str:
    execution = plan["execution"]
    python_executable = execution.get("python_executable")
    if not isinstance(python_executable, str) or not python_executable.strip():
        raise AmendmentError("audit-amendment plan lacks a profile-owned Python executable")
    lines = ["#!/bin/bash", "#SBATCH --job-name=s1fit-warm-audit-v2", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={execution['memory']}", f"#SBATCH --time={execution['time_limit']}"]
    if execution.get("partition"): lines.append(f"#SBATCH --partition={execution['partition']}")
    if execution.get("account"): lines.append(f"#SBATCH --account={execution['account']}")
    if execution.get("reservation"): lines.append(f"#SBATCH --reservation={execution['reservation']}")
    lines.extend(["", "export OMP_NUM_THREADS=1", "export OPENBLAS_NUM_THREADS=1", "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        f"PYTHONPATH=src:. {shlex.quote(python_executable)} -m "
        "multilingual_megatron.experiments.scaling.stage1_fit.cli run-warm-amendment "
        f"--plan {shlex.quote(str(plan_path.resolve()))}", ""])
    return "\n".join(lines)


def _validate_plan(path: Path, *, require_current: bool) -> Mapping[str, Any]:
    plan = _sealed(path, "plan_sha256", PLAN_SCHEMA)
    if plan.get("protocol") != PROTOCOL:
        raise AmendmentError("warm retry protocol changed")
    if require_current:
        if plan.get("provenance") != runtime_provenance():
            raise AmendmentError("amendment plan runtime changed")
    else:
        validate_recorded_provenance(plan.get("provenance"))
    paths = {}
    for key, field in (("grid_manifest", "manifest_sha256"), ("grid_result", "result_sha256"),
            ("failed_audit", "audit_sha256"), ("default_warm_manifest", "manifest_sha256"),
            ("default_warm_result", "result_sha256")):
        paths[key] = _verify_binding(plan[key], field)
    grid_manifest = load_manifest(paths["grid_manifest"], require_current_provenance=False)
    warm_manifest = load_manifest(paths["default_warm_manifest"], require_current_provenance=False)
    from .campaign import _reduced
    grid_result = _reduced(paths["grid_result"]); warm_result = _reduced(paths["default_warm_result"])
    _validate_result_manifest_binding(manifest_path=paths["grid_manifest"], manifest=grid_manifest,
        result_path_=paths["grid_result"], result=grid_result)
    _validate_result_manifest_binding(manifest_path=paths["default_warm_manifest"], manifest=warm_manifest,
        result_path_=paths["default_warm_result"], result=warm_result)
    _validate_nested_warm_input(grid_manifest=grid_manifest, grid_result_path=paths["grid_result"],
        grid_result=grid_result,
        warm_manifest=warm_manifest, warm_result=warm_result)
    try:
        require_clean_definitive_provenance(plan.get("provenance"), review_only=False)
    except ExecutionError as error:
        raise AmendmentError(str(error)) from error
    execution = plan.get("execution")
    if require_current and (not isinstance(execution, Mapping)
            or not isinstance(execution.get("python_executable"), str)
            or not str(execution["python_executable"]).strip()):
        raise AmendmentError("current amendment plan lacks a profile-owned Python executable")
    if plan.get("thresholds") != {"relative_objective": OBJECTIVE_RELATIVE_TOLERANCE,
            "maximum_absolute_prediction": PREDICTION_ABSOLUTE_TOLERANCE}:
        raise AmendmentError("scientific equivalence thresholds changed")
    if plan.get("strict_retry") != {"ftol": STRICT_FTOL, "gtol": STRICT_GTOL,
            "maximum_iterations": STRICT_MAX_ITERATIONS}:
        raise AmendmentError("strict warm retry settings changed")
    if plan.get("actions") != ["strict_warm_retry", "three_start_basin", "projected_direct"]:
        raise AmendmentError("audit-amendment action sequence changed")
    return plan


def _equivalence(retained: Mapping[str, Any], candidate: Mapping[str, Any], *, manifest: Mapping[str, Any]):
    from .campaign import _record_prediction
    return equivalence_check(float(retained["objective"]), float(candidate["objective"]),
        _record_prediction(retained, manifest=manifest), _record_prediction(candidate, manifest=manifest),
        objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
        prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)


def run_warm_audit_amendment(plan_path: Path) -> Mapping[str, Any]:
    """Execute the frozen retry, basin, and direct checks and seal their outcome."""
    plan = _validate_plan(plan_path, require_current=True)
    grid_manifest = load_manifest(Path(plan["grid_manifest"]["path"]), require_current_provenance=False)
    warm_manifest = load_manifest(Path(plan["default_warm_manifest"]["path"]), require_current_provenance=False)
    grid_result = read_json(Path(plan["grid_result"]["path"])); retained = grid_result["selected"]
    problem = load_problem(grid_manifest); bounds = _bounds(grid_manifest); delta = grid_manifest["fit"]["delta"]
    retry = run_projected_start(grid_manifest["fit"]["model"], _point(warm_manifest, 0), problem,
        delta=delta, bounds=bounds, ftol=STRICT_FTOL, gtol=STRICT_GTOL,
        maximum_iterations=STRICT_MAX_ITERATIONS)
    retry_record = _fit_record(retry); retry_check = _equivalence(retained, retry_record, manifest=grid_manifest)
    actions: list[Mapping[str, Any]] = [{"action": "strict_warm_retry", "settings": plan["strict_retry"],
        "record": retry_record, "relative_objective_difference": retry_check.relative_objective_difference,
        "maximum_absolute_prediction_difference": retry_check.maximum_absolute_prediction_difference,
        "resolved": retry_check.equivalent}]
    status = "failed"; reason: str | None = "strict nested-warm retry disagrees"
    if retry_check.equivalent:
        usable = [item for item in grid_result["attempts"] if item.get("success") and item.get("coordinates")]
        usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
        basin_records = []
        for item in usable[:3]:
            start = StartPoint(str(item["start_id"]), None, np.asarray(item["coordinates"], dtype=np.float64), "basin")
            fitted = run_projected_start(grid_manifest["fit"]["model"], start, problem, delta=delta, bounds=bounds,
                ftol=float(grid_manifest["fit"]["outer_ftol"]), gtol=float(grid_manifest["fit"]["outer_gtol"]),
                maximum_iterations=int(grid_manifest["fit"]["outer_max_iterations"]))
            basin_records.append(_fit_record(fitted))
        basin_checks = [_equivalence(retained, item, manifest=grid_manifest) for item in basin_records if item.get("success")]
        basin_resolved = len(basin_checks) == 3 and all(item.equivalent for item in basin_checks)
        actions.append({"action": "three_start_basin", "records": basin_records,
            "equivalence": [{"relative_objective_difference": item.relative_objective_difference,
                "maximum_absolute_prediction_difference": item.maximum_absolute_prediction_difference,
                "resolved": item.equivalent} for item in basin_checks], "resolved": basin_resolved})
        reason = "best basin unresolved"
        if basin_resolved:
            start = StartPoint("direct:best", None, np.asarray(retained["coordinates"], dtype=np.float64), "direct")
            projected = run_projected_start(grid_manifest["fit"]["model"], start, problem, delta=delta, bounds=bounds,
                ftol=float(grid_manifest["fit"]["outer_ftol"]), gtol=float(grid_manifest["fit"]["outer_gtol"]),
                maximum_iterations=int(grid_manifest["fit"]["outer_max_iterations"]))
            if not projected.success:
                raise AmendmentError("projected direct-check start failed")
            direct = direct_joint_check(projected, problem, delta=delta, bounds=bounds)
            actions.append({"action": "projected_direct", "projected": _fit_record(projected),
                "direct": _direct_fit_record(direct.direct),
                "relative_objective_difference": direct.equivalence.relative_objective_difference,
                "maximum_absolute_prediction_difference": direct.equivalence.maximum_absolute_prediction_difference,
                "retried": direct.retried, "resolved": direct.resolved})
            status = "complete" if direct.resolved else "failed"
            reason = None if direct.resolved else "projected/direct equivalence unresolved"
    unsigned = {"schema_version": RUN_SCHEMA, "protocol": PROTOCOL,
        "plan": _binding(plan_path, "plan_sha256"), "grid_manifest_sha256": grid_manifest["manifest_sha256"],
        "grid_result_sha256": grid_result["result_sha256"], "status": status, "reason": reason,
        "actions": actions, "provenance": runtime_provenance()}
    run = _seal(unsigned, "run_sha256"); run_path = _run_path(plan); write_json_once(run_path, run)
    acceptance_path = None
    if status == "complete":
        accepted_unsigned = {"schema_version": ACCEPTANCE_SCHEMA, "protocol": PROTOCOL,
            "grid_manifest": plan["grid_manifest"], "grid_result": plan["grid_result"],
            "failed_audit": plan["failed_audit"], "run": _binding(run_path, "run_sha256"),
            "status": "complete"}
        accepted = _seal(accepted_unsigned, "acceptance_sha256")
        acceptance_path = accepted_path(grid_result, output_root=Path(plan["output_root"]))
        write_json_once(acceptance_path, accepted)
    return {"status": status, "reason": reason, "run_path": str(run_path),
        "run_sha256": run["run_sha256"], "acceptance_path": None if acceptance_path is None else str(acceptance_path)}


def validate_accepted_amendment(result: Mapping[str, Any], failed_audit: Mapping[str, Any], *, output_root: Path) -> None:
    """Validate a complete amendment chain for an otherwise failed historical audit."""
    if failed_audit.get("status") != "failed" or failed_audit.get("reason") != "nested warm start disagrees":
        raise AmendmentError("only the registered nested-warm failure may be amended")
    path = accepted_path(result, output_root=output_root)
    accepted = _sealed(path, "acceptance_sha256", ACCEPTANCE_SCHEMA)
    if accepted.get("protocol") != PROTOCOL or accepted.get("status") != "complete":
        raise AmendmentError("audit amendment is not complete")
    grid_result_path = _verify_binding(accepted["grid_result"], "result_sha256")
    grid_manifest_path = _verify_binding(accepted["grid_manifest"], "manifest_sha256")
    failed_path = _verify_binding(accepted["failed_audit"], "audit_sha256")
    run_path = _verify_binding(accepted["run"], "run_sha256")
    if (read_json(grid_result_path).get("result_sha256") != result.get("result_sha256")
            or read_json(grid_manifest_path).get("manifest_sha256") != result.get("manifest_sha256")
            or read_json(failed_path).get("audit_sha256") != failed_audit.get("audit_sha256")):
        raise AmendmentError("audit amendment targets different historical evidence")
    run = _sealed(run_path, "run_sha256", RUN_SCHEMA)
    if run.get("protocol") != PROTOCOL or run.get("status") != "complete" or run.get("reason") is not None:
        raise AmendmentError("accepted audit-amendment run is incomplete")
    validate_recorded_provenance(run.get("provenance"))
    plan_path = _verify_binding(run["plan"], "plan_sha256")
    plan = _validate_plan(plan_path, require_current=False)
    if (plan.get("grid_manifest") != accepted.get("grid_manifest")
            or plan.get("grid_result") != accepted.get("grid_result")
            or plan.get("failed_audit") != accepted.get("failed_audit")
            or run.get("grid_manifest_sha256") != result.get("manifest_sha256")
            or run.get("grid_result_sha256") != result.get("result_sha256")):
        raise AmendmentError("audit-amendment lineage is inconsistent")
    actions = run.get("actions")
    if (not isinstance(actions, list)
            or not all(isinstance(item, Mapping) and bool(item.get("resolved")) for item in actions)
            or [item.get("action") for item in actions] != list(plan["actions"])):
        raise AmendmentError("audit-amendment actions are incomplete or unresolved")
