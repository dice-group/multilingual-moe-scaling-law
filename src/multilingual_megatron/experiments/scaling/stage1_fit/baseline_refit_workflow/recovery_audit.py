"""One-shot, training-only strict-basin amendment for the J0 baseline.

The original grid optimum and failed audit are immutable. Acceptance requires
all three originally selected basin starts, plus an independent direct check,
to reproduce that optimum under the original equivalence tolerances. This
module deliberately does not register the amendment with the generic fitter;
the baseline recovery report must explicitly validate and cite it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping

import numpy as np

from .. import campaign, execution
from ..amendment import _validate_result_manifest_binding
from ..data import sha256_file
from ..fit import (
    OBJECTIVE_RELATIVE_TOLERANCE, PREDICTION_ABSOLUTE_TOLERANCE,
    StartPoint, direct_joint_check, equivalence_check, run_projected_start,
)
from .contracts import BaselineRefitError, repository_root, seal, verify_seal

PROTOCOL = "stage1_baseline_j0_strict_basin/v1"
SETTINGS = {"ftol": 1e-14, "gtol": 1e-8, "maximum_iterations": 20_000}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TARGET = ("J0", "largest_N_act")
MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.recovery_audit"


def binding(path: Path, field: str) -> dict[str, str]:
    value = execution.read_json(path)
    verify_seal(value, field, label=str(path))
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), field: value[field]}


def bound(value: Mapping[str, str], field: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value["path"])
    if binding(path, field) != value:
        raise BaselineRefitError(f"amendment input changed: {path}")
    return path, execution.read_json(path)


def code_binding() -> dict[str, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(),
                            check=True, capture_output=True, text=True).stdout.strip()
    path = Path(__file__).resolve()
    relative = str(path.relative_to(repository_root()))
    content = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repository_root(),
                             check=True, capture_output=True).stdout
    if content != path.read_bytes():
        raise BaselineRefitError("strict-basin amendment source must be committed")
    return {"git_commit": commit, "relative_path": relative, "file_sha256": sha256_file(path)}


def validate_code(value: Mapping[str, str], *, live: bool) -> None:
    import hashlib
    content = subprocess.run(["git", "show", f"{value['git_commit']}:{value['relative_path']}"],
                             cwd=repository_root(), check=True, capture_output=True).stdout
    if hashlib.sha256(content).hexdigest() != value["file_sha256"]:
        raise BaselineRefitError("recorded amendment source differs from its commit")
    if live and sha256_file(repository_root() / value["relative_path"]) != value["file_sha256"]:
        raise BaselineRefitError("live amendment code changed after the plan was sealed")


def validate_target(manifest: Mapping[str, Any], result: Mapping[str, Any],
                    audit: Mapping[str, Any], basin: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (manifest["fit"]["model"], manifest["fit"]["fold_id"]) != TARGET:
        raise BaselineRefitError("this amendment is limited to J0/largest_N_act")
    if (audit.get("status"), audit.get("reason")) != ("failed", "best basin unresolved"):
        raise BaselineRefitError("this amendment requires the original failed basin audit")
    if audit.get("result_sha256") != result["result_sha256"] \
            or audit.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise BaselineRefitError("failed audit targets a different grid result")
    if result["selected"].get("boundary_contacts"):
        raise BaselineRefitError("a boundary failure cannot be repaired by a basin retry")
    expected = sorted((v for v in result["attempts"] if v.get("success") and v.get("coordinates")),
                      key=lambda v: (v["objective"], v["start_id"]))[:3]
    starts = [{"start_id": v["start_id"], "values": v["coordinates"], "kind": "basin"} for v in expected]
    actual = [{k: v[k] for k in ("start_id", "values", "kind")}
              for v in basin["fit"].get("explicit_starts", [])]
    if len(starts) != 3 or actual != starts:
        raise BaselineRefitError("strict retry must use the exact original three basin starts")
    for key in ("input", "membership", "review_only", "selection_report"):
        if basin.get(key) != manifest.get(key):
            raise BaselineRefitError(f"basin {key} differs from the original grid")
    for key in ("model", "fold_id", "objective", "delta", "bounds", "bound_version"):
        if basin["fit"][key] != manifest["fit"][key]:
            raise BaselineRefitError(f"basin fit {key} differs from the original grid")
    return starts


def prepare(profile_path: Path) -> Path:
    profile = json.loads(profile_path.read_text())
    path = Path(profile["target_manifest"])
    if not path.is_absolute():
        path = repository_root() / path
    manifest = execution.load_manifest(path)
    result_path = execution.result_path(manifest)
    result = campaign._reduced(result_path)
    campaign._completion_receipt(result)
    _validate_result_manifest_binding(manifest_path=path.resolve(), manifest=manifest,
                                     result_path_=result_path, result=result)
    audit_path = campaign.audit_path(result, output_root=result_path.parent.parent)
    audit = execution.read_json(audit_path)
    verify_seal(audit, "audit_sha256", label="original failed audit")
    action = next(v for v in audit["actions"] if v["action"] == "basin")
    basin_path = Path(action["manifest"])
    basin = execution.load_manifest(basin_path)
    starts = validate_target(manifest, result, audit, basin)
    output = Path(profile["output_root"])
    if not output.is_absolute():
        output = repository_root() / output
    if output.resolve() == result_path.parent.parent.resolve():
        raise BaselineRefitError("amendment must use a separate recovery output root")
    plan = seal({
        "schema_version": PROTOCOL, "protocol": PROTOCOL,
        "profile": {"path": str(profile_path.resolve()), "file_sha256": sha256_file(profile_path)},
        "grid_manifest": binding(path, "manifest_sha256"),
        "grid_result": binding(result_path, "result_sha256"),
        "failed_audit": binding(audit_path, "audit_sha256"),
        "basin_manifest": binding(basin_path, "manifest_sha256"),
        "strict_settings": SETTINGS, "starts": starts,
        "objective_relative_tolerance": OBJECTIVE_RELATIVE_TOLERANCE,
        "prediction_absolute_tolerance": PREDICTION_ABSOLUTE_TOLERANCE,
        "source": code_binding(), "numerical_provenance": execution.runtime_provenance(),
        "output_root": str(output.resolve()), "heldout_scores_used": False,
        "claim": "numerical equivalence only; original coefficients and selection unchanged",
    }, "plan_sha256")
    plan_path = output / "audit_amendment" / "plan.json"
    execution.write_json_once(plan_path, plan)
    log = output / "logs" / "attempts" / plan["plan_sha256"][:12]
    log.mkdir(parents=True, exist_ok=True)
    command = [str(Path(profile["python_executable"]).resolve()), "-m", MODULE,
               "run", "--plan", str(plan_path)]
    script = "\n".join([
        "#!/bin/bash", "#SBATCH --job-name=s1br-j0-strict-basin", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={profile['audit_memory']}", f"#SBATCH --time={profile['audit_time_limit']}",
        f"#SBATCH --partition={profile['partition']}", f"#SBATCH --account={profile['account']}",
        f"#SBATCH --output={log / 'strict-basin-%j.out'}", "set -euo pipefail",
        f"cd {shlex.quote(str(repository_root()))}",
        *(f"export {name}=1" for name in THREADS), "export PYTHONPATH=src:.",
        shlex.join(command), "",
    ])
    execution.write_text_once(output / "audit_amendment" / "run.sbatch", script)
    return plan_path


def load_plan(path: Path, *, live: bool = True) -> dict[str, Any]:
    plan = execution.read_json(path)
    verify_seal(plan, "plan_sha256", label="strict-basin plan")
    if plan.get("schema_version") != PROTOCOL or plan.get("protocol") != PROTOCOL \
            or plan.get("strict_settings") != SETTINGS or plan.get("heldout_scores_used") is not False \
            or plan.get("objective_relative_tolerance") != OBJECTIVE_RELATIVE_TOLERANCE \
            or plan.get("prediction_absolute_tolerance") != PREDICTION_ABSOLUTE_TOLERANCE:
        raise BaselineRefitError("strict-basin protocol changed")
    validate_code(plan["source"], live=live)
    execution.validate_recorded_provenance(plan["numerical_provenance"])
    if live and not execution._current_provenance_matches(plan["numerical_provenance"]):
        raise BaselineRefitError("numerical environment changed after amendment planning")
    mp, manifest = bound(plan["grid_manifest"], "manifest_sha256")
    rp, result = bound(plan["grid_result"], "result_sha256")
    _, audit = bound(plan["failed_audit"], "audit_sha256")
    _, basin = bound(plan["basin_manifest"], "manifest_sha256")
    execution.validate_manifest(manifest, require_current_provenance=live)
    _validate_result_manifest_binding(manifest_path=mp, manifest=manifest, result_path_=rp, result=result)
    if validate_target(manifest, result, audit, basin) != plan["starts"]:
        raise BaselineRefitError("amendment starts changed")
    return plan


def check_record(retained: Mapping[str, Any], record: Mapping[str, Any],
                 manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not record.get("success") or record.get("boundary_contacts"):
        return {"resolved": False, "reason": "optimizer failure or boundary contact"}
    check = equivalence_check(retained["objective"], record["objective"],
        campaign._record_prediction(retained, manifest=manifest),
        campaign._record_prediction(record, manifest=manifest))
    return {"resolved": check.equivalent,
            "relative_objective_difference": check.relative_objective_difference,
            "maximum_absolute_prediction_difference": check.maximum_absolute_prediction_difference}


def run(path: Path) -> dict[str, Any]:
    plan = load_plan(path)
    if not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(k) != "1" for k in THREADS):
        raise BaselineRefitError("formal recovery requires a Slurm CPU allocation and single-threaded BLAS")
    root = Path(plan["output_root"]) / "audit_amendment"
    if (root / "run.json").exists():
        return validate_acceptance(path)
    claim = seal({"plan": binding(path, "plan_sha256"), "job_id": os.environ["SLURM_JOB_ID"],
                  "status": "claimed_before_optimization", "one_shot": True}, "claim_sha256")
    if (root / "claim.json").exists():
        raise BaselineRefitError("amendment already claimed; do not silently retry an interrupted audit")
    execution.write_json_once(root / "claim.json", claim)
    manifest = execution.load_manifest(Path(plan["grid_manifest"]["path"]))
    retained = execution.read_json(Path(plan["grid_result"]["path"]))["selected"]
    problem = execution.load_problem(manifest)
    bounds = execution._bounds(manifest)
    records = []
    fitted_starts = []
    for i, start in enumerate(plan["starts"]):
        fitted = run_projected_start("J0", StartPoint(start["start_id"], i,
            np.asarray(start["values"], dtype=np.float64), "basin"), problem,
            delta=manifest["fit"]["delta"], bounds=bounds, **SETTINGS)
        record = execution._fit_record(fitted)
        checked = check_record(retained, record, manifest)
        item = {"index": i, "start": start, "record": record, "check": checked}
        records.append(item)
        fitted_starts.append(fitted)
        execution.write_json_once(root / f"start-{i}.json", seal(item, "task_sha256"))
        print(json.dumps({"start": i, "check": checked}), flush=True)
    direct_record = None
    direct_checks = None
    if all(v["check"]["resolved"] for v in records):
        direct = direct_joint_check(fitted_starts[0], problem, delta=manifest["fit"]["delta"], bounds=bounds)
        direct_record = execution._direct_fit_record(direct.direct, "J0")
        direct_checks = {"against_retained": check_record(retained, direct_record, manifest),
                         "projected_direct_resolved": direct.resolved, "retried": direct.retried}
    complete = all(v["check"]["resolved"] for v in records) and direct_checks is not None \
        and direct_checks["against_retained"]["resolved"] and direct_checks["projected_direct_resolved"]
    load_plan(path)  # refuse source or input drift during execution
    result = seal({"schema_version": PROTOCOL, "plan": binding(path, "plan_sha256"),
        "claim": binding(root / "claim.json", "claim_sha256"), "starts": records,
        "direct_record": direct_record, "direct_checks": direct_checks,
        "status": "complete" if complete else "failed", "heldout_scores_used": False,
        "provenance": execution.runtime_provenance()}, "run_sha256")
    execution.write_json_once(root / "run.json", result)
    if not complete:
        raise BaselineRefitError("strict basin amendment failed; original audit remains failed")
    return validate_acceptance(path)


def validate_acceptance(path: Path) -> dict[str, Any]:
    plan = load_plan(path, live=False)
    root = Path(plan["output_root"]) / "audit_amendment"
    result = execution.read_json(root / "run.json")
    verify_seal(result, "run_sha256", label="strict-basin run")
    if result.get("schema_version") != PROTOCOL or result.get("status") != "complete" \
            or result.get("heldout_scores_used") is not False or result.get("plan") != binding(path, "plan_sha256"):
        raise BaselineRefitError("strict-basin amendment is not accepted")
    _, claim = bound(result["claim"], "claim_sha256")
    if claim.get("plan") != result["plan"] or claim.get("one_shot") is not True \
            or claim.get("status") != "claimed_before_optimization" \
            or not str(claim.get("job_id", "")).isdigit():
        raise BaselineRefitError("strict-basin run does not bind its one-shot claim")
    execution.validate_recorded_provenance(result["provenance"])
    if {k: v for k, v in result["provenance"].items() if k not in ("git_commit", "git_dirty")} \
            != {k: v for k, v in plan["numerical_provenance"].items() if k not in ("git_commit", "git_dirty")}:
        raise BaselineRefitError("amendment runtime differs from its plan")
    manifest = execution.read_json(Path(plan["grid_manifest"]["path"]))
    retained = execution.read_json(Path(plan["grid_result"]["path"]))["selected"]
    if len(result.get("starts", [])) != 3:
        raise BaselineRefitError("strict-basin run lacks all three starts")
    for i, (item, start) in enumerate(zip(result["starts"], plan["starts"], strict=True)):
        if item["index"] != i or item["start"] != start \
                or item["record"].get("start_id") != start["start_id"] \
                or item["record"].get("start_index") != i:
            raise BaselineRefitError("strict-basin run changed the frozen starts")
        check = check_record(retained, item["record"], manifest)
        if check != item["check"] or not check["resolved"]:
            raise BaselineRefitError("strict-basin equivalence cannot be reproduced")
    direct = result.get("direct_record")
    if not direct or not check_record(retained, direct, manifest)["resolved"] \
            or not result.get("direct_checks", {}).get("projected_direct_resolved") \
            or not check_record(result["starts"][0]["record"], direct, manifest)["resolved"]:
        raise BaselineRefitError("direct-joint equivalence cannot be reproduced")
    return {"status": "complete", "protocol": PROTOCOL,
            "plan": binding(path, "plan_sha256"), "run": binding(root / "run.json", "run_sha256"),
            "original_result": plan["grid_result"], "original_failed_audit": plan["failed_audit"]}


def main() -> None:
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
