"""Training-only J0 numerical refinement, preserving both failed audits.

This is not an equivalence waiver for the old coefficients. Three original
grid basins must converge to a common, non-worse conditioned solution. The
new selected coefficients have their own result and audit hashes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np

from .. import campaign, execution, fit
from . import recovery_audit as original
from . import recovery_numerics as numerics
from .contracts import BaselineRefitError, repository_root, seal, verify_seal
from ..data import sha256_file
from .recovery import _validate_sources

PROTOCOL = "stage1_baseline_j0_conditioned_refit/v1"
MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.refit_audit"
SOURCE_FILES = ("refit_audit.py", "recovery_numerics.py", "recovery_audit.py")
SETTINGS = {"inner": numerics.INNER_SETTINGS, "outer": numerics.OUTER_SETTINGS,
    "objective_scale": numerics.OBJECTIVE_SCALE, "outer_gradient_max": 1e-8,
    "gradient_check_step": 1e-6, "gradient_check_atol": 1e-8,
    "objective_relative_tolerance": fit.OBJECTIVE_RELATIVE_TOLERANCE,
    "prediction_absolute_tolerance": fit.PREDICTION_ABSOLUTE_TOLERANCE}


def prepare(profile_path: Path) -> Path:
    profile = json.loads(profile_path.read_text())
    root = repository_root() / profile["output_root"]
    old_path = root / "audit_amendment" / "plan.json"
    old = original.load_plan(old_path, live=False)
    previous_run = root / "audit_amendment" / "run.json"
    old_run = execution.read_json(previous_run)
    verify_seal(old_run, "run_sha256", label="failed strict retry")
    if old_run["status"] != "failed" or old_run["plan"] != original.binding(old_path, "plan_sha256"):
        raise BaselineRefitError("refinement requires the preserved failed strict retry")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(),
        check=True, capture_output=True, text=True).stdout.strip()
    source = {"git_commit": commit, "files": {
        str((Path(__file__).parent / name).relative_to(repository_root())):
            sha256_file(Path(__file__).parent / name) for name in SOURCE_FILES}}
    _validate_sources(source, live=True)
    plan = seal({"schema_version": PROTOCOL, "settings": SETTINGS,
        "old_plan": original.binding(old_path, "plan_sha256"),
        "failed_strict_retry": original.binding(previous_run, "run_sha256"),
        "grid_manifest": old["grid_manifest"], "grid_result": old["grid_result"],
        "starts": old["starts"], "source": source,
        "numerical_provenance": execution.runtime_provenance(),
        "heldout_scores_used": False, "output_root": str(root.resolve()),
        "selection_rule": "minimum training objective among the three frozen refined basins",
        "old_coefficients_may_change": True, "original_failed_receipts_preserved": True,
    }, "plan_sha256")
    path = root / "numerical_refit" / "plan.json"
    execution.write_json_once(path, plan)
    logs = root / "logs" / "attempts" / plan["plan_sha256"][:12]
    logs.mkdir(parents=True, exist_ok=True)
    script = "\n".join(["#!/bin/bash", "#SBATCH --job-name=s1br-j0-conditioned-refit",
        "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={profile['audit_memory']}",
        f"#SBATCH --time={profile['audit_time_limit']}", f"#SBATCH --partition={profile['partition']}",
        f"#SBATCH --account={profile['account']}", f"#SBATCH --output={logs / 'refit-%j.out'}",
        "set -euo pipefail", f"cd {shlex.quote(str(repository_root()))}",
        *(f"export {name}=1" for name in original.THREADS), "export PYTHONPATH=src:.",
        shlex.join([profile["python_executable"], "-m", MODULE, "run", "--plan", str(path)]), ""])
    execution.write_text_once(path.parent / "run.sbatch", script)
    return path


def load_plan(path: Path, *, live=True):
    plan = execution.read_json(path)
    verify_seal(plan, "plan_sha256", label="conditioned refit plan")
    if plan.get("schema_version") != PROTOCOL or plan.get("settings") != SETTINGS \
            or plan.get("heldout_scores_used") is not False \
            or plan.get("old_coefficients_may_change") is not True:
        raise BaselineRefitError("conditioned refit protocol changed")
    _validate_sources(plan["source"], live=live)
    old_path, _ = original.bound(plan["old_plan"], "plan_sha256")
    old = original.load_plan(old_path, live=False)
    _, old_run = original.bound(plan["failed_strict_retry"], "run_sha256")
    if old_run["status"] != "failed" or old_run["plan"] != plan["old_plan"]:
        raise BaselineRefitError("failed strict retry binding changed")
    if any(plan[k] != old[k] for k in ("grid_manifest", "grid_result", "starts")):
        raise BaselineRefitError("refinement changed its original inputs or frozen starts")
    execution.validate_recorded_provenance(plan["numerical_provenance"])
    if live and not execution._current_provenance_matches(plan["numerical_provenance"]):
        raise BaselineRefitError("core numerical provenance changed")
    return plan


def record_stationarity(record, manifest):
    problem = execution.load_problem(manifest)
    prediction = campaign._record_prediction(record, manifest=manifest)
    value = fit.weighted_objective(prediction - problem.y, problem.weights, manifest["fit"]["delta"])
    if abs(value - record["objective"]) > 1e-14:
        raise BaselineRefitError("recorded numerical objective cannot be reproduced")
    _, _, cj, dj = fit.predictor_terms_and_jacobian("J0", record["coordinates"],
        problem.N, problem.D, problem.E, problem.K)
    coefficients = {v["language"]: v for v in record["language_coefficients"]}
    A = np.asarray([coefficients[v]["A"] for v in problem.language_ids])
    B = np.asarray([coefficients[v]["B"] for v in problem.language_ids])
    _, gradient = fit.weighted_objective_and_gradient(prediction - problem.y,
        A[:, None] * cj + B[:, None] * dj, problem.weights, manifest["fit"]["delta"])
    if np.max(np.abs(gradient)) > SETTINGS["outer_gradient_max"]:
        raise BaselineRefitError("conditioned refit is not stationary")
    return gradient


def run(path: Path):
    from scipy.optimize._numdiff import approx_derivative
    plan = load_plan(path)
    if not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(k) != "1" for k in original.THREADS):
        raise BaselineRefitError("numerical refit requires a single-threaded Slurm CPU allocation")
    root = path.parent
    if (root / "claim.json").exists():
        raise BaselineRefitError("numerical refit already claimed; preserve interrupted or failed attempts")
    execution.write_json_once(root / "claim.json", seal({"plan": original.binding(path, "plan_sha256"),
        "job_id": os.environ["SLURM_JOB_ID"], "one_shot": True}, "claim_sha256"))
    manifest = execution.load_manifest(Path(plan["grid_manifest"]["path"]))
    problem = execution.load_problem(manifest)
    profiler = numerics.J0Profile(problem, manifest["fit"]["delta"])
    bounds = execution._bounds(manifest)
    records, fitted = [], []
    for i, start in enumerate(plan["starts"]):
        result, gradient = profiler.run(fit.StartPoint(start["start_id"], i,
            np.asarray(start["values"]), "basin"), bounds)
        item = {"index": i, "start": start, "record": execution._fit_record(result),
                "gradient": gradient.tolist()}
        records.append(item)
        fitted.append(result)
        execution.write_json_once(root / f"start-{i}.json", seal(item, "task_sha256"))
        print(json.dumps({"start": i, "objective": result.objective,
            "success": result.success, "gradient_max": float(np.max(np.abs(gradient))),
            "iterations": result.iterations, "message": result.message}), flush=True)
    index = min(range(3), key=lambda i: (records[i]["record"]["objective"], records[i]["record"]["start_id"]))
    selected = records[index]["record"]
    checks = [original.check_record(selected, v["record"], manifest) for v in records]
    old = execution.read_json(Path(plan["grid_result"]["path"]))
    non_worse = selected["objective"] <= old["selected"]["objective"] + 1e-15
    direct = fit.direct_joint_check(fitted[index], problem, delta=manifest["fit"]["delta"], bounds=bounds)
    direct_record = execution._direct_fit_record(direct.direct, "J0")
    direct_check = original.check_record(selected, direct_record, manifest)
    numerical_gradient = approx_derivative(lambda x: profiler.evaluate(x)[0].objective,
        selected["coordinates"], method="3-point", abs_step=SETTINGS["gradient_check_step"])
    gradient_error = float(np.max(np.abs(numerical_gradient - np.asarray(records[index]["gradient"]))))
    stationary = all(np.max(np.abs(v["gradient"])) <= SETTINGS["outer_gradient_max"] for v in records)
    accepted = (all(v["resolved"] for v in checks) and non_worse and stationary
        and direct.resolved and direct_check["resolved"] and gradient_error <= SETTINGS["gradient_check_atol"])
    load_plan(path)
    audit = seal({"schema_version": PROTOCOL, "plan": original.binding(path, "plan_sha256"),
        "claim": original.binding(root / "claim.json", "claim_sha256"),
        "status": "complete" if accepted else "failed", "starts": records, "selected_index": index,
        "checks": checks, "non_worse": non_worse, "stationary": stationary,
        "direct_record": direct_record, "direct_check": direct_check,
        "direct_resolved": direct.resolved, "gradient_check_error": gradient_error,
        "numerical_gradient": numerical_gradient.tolist(), "heldout_scores_used": False,
        "old_vs_new": original.check_record(old["selected"], selected, manifest),
        "runtime_provenance": execution.runtime_provenance()}, "audit_sha256")
    execution.write_json_once(root / "audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("status", "checks", "direct_check", "gradient_check_error", "old_vs_new")}), flush=True)
    if not accepted:
        raise BaselineRefitError("conditioned refit failed its unchanged acceptance criteria")
    unsigned = dict(old)
    unsigned.pop("result_sha256")
    unsigned["selected"] = selected
    unsigned["numerical_refinement"] = {"protocol": PROTOCOL,
        "original_grid_result": plan["grid_result"], "plan": original.binding(path, "plan_sha256"),
        "audit": original.binding(root / "audit.json", "audit_sha256"),
        "selection_rule": plan["selection_rule"], "original_grid_attempts_retained": True}
    execution.write_json_once(root / "result.json", seal(unsigned, "result_sha256"))
    return validate_acceptance(path)


def validate_acceptance(path: Path):
    plan = load_plan(path, live=False)
    root = path.parent
    audit = execution.read_json(root / "audit.json")
    verify_seal(audit, "audit_sha256", label="conditioned refit audit")
    if audit.get("schema_version") != PROTOCOL or audit.get("status") != "complete" \
            or audit.get("plan") != original.binding(path, "plan_sha256") \
            or audit.get("heldout_scores_used") is not False:
        raise BaselineRefitError("conditioned numerical refit is not accepted")
    _, claim = original.bound(audit["claim"], "claim_sha256")
    if claim.get("one_shot") is not True or claim.get("plan") != audit["plan"] \
            or not str(claim.get("job_id", "")).isdigit():
        raise BaselineRefitError("refit one-shot claim changed")
    execution.validate_recorded_provenance(audit["runtime_provenance"])
    comparable = lambda value: {k: v for k, v in value.items() if k not in ("git_commit", "git_dirty")}
    if comparable(audit["runtime_provenance"]) != comparable(plan["numerical_provenance"]):
        raise BaselineRefitError("refit numerical environment changed")
    manifest = execution.read_json(Path(plan["grid_manifest"]["path"]))
    old = execution.read_json(Path(plan["grid_result"]["path"]))
    result = execution.read_json(root / "result.json")
    verify_seal(result, "result_sha256", label="conditioned numerical result")
    if len(audit.get("starts", [])) != 3:
        raise BaselineRefitError("refit lacks all three frozen starts")
    selected = result["selected"]
    selected_index = min(range(3), key=lambda i: (audit["starts"][i]["record"]["objective"],
                                                audit["starts"][i]["record"]["start_id"]))
    if audit["selected_index"] != selected_index or selected != audit["starts"][selected_index]["record"] \
            or selected["objective"] > old["selected"]["objective"] + 1e-15:
        raise BaselineRefitError("refit did not select the best non-worse training solution")
    for key in old:
        if key not in ("selected", "result_sha256") and result.get(key) != old[key]:
            raise BaselineRefitError(f"refit changed the original contract: {key}")
    for i, (item, start) in enumerate(zip(audit["starts"], plan["starts"], strict=True)):
        if item["start"] != start or item["index"] != i or item["record"]["start_index"] != i \
                or item["record"]["start_id"] != start["start_id"]:
            raise BaselineRefitError("refit changed its frozen basin starts")
        check = original.check_record(selected, item["record"], manifest)
        if check != audit["checks"][i] or not check["resolved"]:
            raise BaselineRefitError("refit basin agreement cannot be reproduced")
        gradient = record_stationarity(item["record"], manifest)
        if not np.allclose(gradient, item["gradient"], rtol=0, atol=1e-12):
            raise BaselineRefitError("refit stationarity cannot be reproduced")
    direct_check = original.check_record(selected, audit["direct_record"], manifest)
    if direct_check != audit["direct_check"] or not direct_check["resolved"] or not audit["direct_resolved"]:
        raise BaselineRefitError("refit direct-joint agreement cannot be reproduced")
    error = float(np.max(np.abs(np.asarray(audit["numerical_gradient"]) -
                               np.asarray(audit["starts"][selected_index]["gradient"]))))
    if error != audit["gradient_check_error"] or error > SETTINGS["gradient_check_atol"]:
        raise BaselineRefitError("refit gradient check failed")
    refinement = result["numerical_refinement"]
    if refinement["original_grid_result"] != plan["grid_result"] \
            or refinement["plan"] != audit["plan"] \
            or refinement["audit"] != original.binding(root / "audit.json", "audit_sha256"):
        raise BaselineRefitError("refined result lost its audit or original-grid binding")
    return {"status": "complete", "protocol": PROTOCOL, "plan": audit["plan"],
        "run": original.binding(root / "audit.json", "audit_sha256"),
        "refined_result": original.binding(root / "result.json", "result_sha256"),
        "original_result": plan["grid_result"], "failed_strict_retry": plan["failed_strict_retry"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--profile", type=Path, required=True)
    for name in ("run", "verify"):
        commands.add_parser(name).add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.profile))
    else:
        print(json.dumps(run(args.plan) if args.command == "run" else validate_acceptance(args.plan), indent=2))


if __name__ == "__main__":
    main()
