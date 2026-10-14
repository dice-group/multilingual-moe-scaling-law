"""Finish a blocked baseline refit without overwriting its failed receipts.

One explicit J0 numerical refit is accepted by this consumer only. Other cells
retain their ordinary immutable numerical audits. Reports live in the
separate recovery root and preserve the original baseline decision rule.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping

from .. import campaign, execution, report as fit_report
from ..data import sha256_file
from . import reporting
from .contracts import (
    BASELINES, CANDIDATES, FOLDS, MODELS, BaselineRefitError,
    load_workflow, repository_root, seal, verify_seal,
)
from .planning import manifest_path_for_job
from . import recovery_audit as amendment

PROTOCOL = "stage1_baseline_refit_recovery/v2"
MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.recovery"
SOURCE_FILES = ("recovery.py", "recovery_audit.py", "refit_audit.py", "recovery_numerics.py",
                "recovery_diagnostics.py", "reporting.py")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def _sources() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(),
                            check=True, capture_output=True, text=True).stdout.strip()
    files = {}
    for name in SOURCE_FILES:
        path = Path(__file__).parent / name
        relative = str(path.relative_to(repository_root()))
        content = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repository_root(),
                                 check=True, capture_output=True).stdout
        if content != path.read_bytes():
            raise BaselineRefitError(f"recovery source must be committed: {name}")
        files[relative] = sha256_file(path)
    return {"git_commit": commit, "files": files}


def _validate_sources(source: Mapping[str, Any], *, live: bool) -> None:
    for relative, expected in source["files"].items():
        content = subprocess.run(["git", "show", f"{source['git_commit']}:{relative}"],
                                 cwd=repository_root(), check=True, capture_output=True).stdout
        if hashlib.sha256(content).hexdigest() != expected:
            raise BaselineRefitError(f"recorded recovery source changed: {relative}")
        if live and sha256_file(repository_root() / relative) != expected:
            raise BaselineRefitError(f"live recovery source changed: {relative}")


def _original_workflow(path: Path) -> dict[str, Any]:
    # The original reporting implementation predates the explicit result-loader
    # hook. Recover its recorded Git bytes, not a false live-source equivalence.
    workflow = load_workflow(path, verify_live_files=False)
    for value in [workflow["profile_binding"], *workflow["inputs"].values()]:
        if sha256_file(Path(value["path"])) != value["file_sha256"]:
            raise BaselineRefitError("original workflow input/profile changed")
    source = workflow["workflow_source"]
    prefix = "src/multilingual_megatron/experiments/scaling/stage1_fit/"
    _validate_sources({"git_commit": source["git_commit"],
                       "files": {prefix + k: v for k, v in source["files"].items()}}, live=False)
    return workflow


def prepare(profile_path: Path, *, audit_job_id: str) -> Path:
    from . import refit_audit
    profile = json.loads(profile_path.read_text())
    if profile.get("schema_version") != "stage1_baseline_recovery_profile/v1" \
            or not audit_job_id.isdigit():
        raise BaselineRefitError("invalid recovery profile or audit job ID")
    workflow_path = _resolve(profile["workflow_manifest"])
    workflow = _original_workflow(workflow_path)
    output = _resolve(profile["output_root"])
    audit_plan_path = output / "numerical_refit" / "plan.json"
    audit_plan = refit_audit.load_plan(audit_plan_path)
    refit_audit.validate_acceptance(audit_plan_path)
    ordinary_path = output / "receipts" / "ordinary_audits.json"
    ordinary = execution.read_json(ordinary_path)
    verify_seal(ordinary, "receipt_sha256", label="ordinary audit collection")
    if ordinary.get("status") != "collected" or len(ordinary.get("cells", [])) != 15 \
            or any(v["status"] != "complete" for v in ordinary["cells"]
                   if (v["model"], v["fold"]) != ("J0", "largest_N_act")):
        raise BaselineRefitError("the other fourteen ordinary audits must be accepted before finalization")
    expected = manifest_path_for_job(workflow, "grid.robust.J0.largest_N_act")
    if audit_plan["grid_manifest"] != amendment.binding(expected, "manifest_sha256"):
        raise BaselineRefitError("J0 amendment belongs to another baseline workflow")
    manifests = {}
    for model in MODELS:
        for fold in FOLDS:
            job = f"grid.robust.{model}.{fold}" if model in BASELINES else f"grid.{model.lower()}.{fold}"
            path = manifest_path_for_job(workflow, job)
            manifests[f"{model}/{fold}"] = amendment.binding(path, "manifest_sha256")
    gates = {}
    for name in ("scale", "m0", "baselines", "m1", "final"):
        path = Path(workflow["output_root"]) / "control/baseline_refit/gates" / f"gate-{name}.json"
        gates[name] = amendment.binding(path, "gate_sha256")
    plan = seal({"schema_version": PROTOCOL, "protocol": PROTOCOL,
        "workflow": amendment.binding(workflow_path, "workflow_sha256"),
        "original_gates": gates, "manifests": manifests,
        "ordinary_audits": amendment.binding(ordinary_path, "receipt_sha256"),
        "audit_plan": amendment.binding(audit_plan_path, "plan_sha256"),
        "audit_job_id": audit_job_id, "source": _sources(),
        "profile": {"path": str(profile_path.resolve()), "file_sha256": sha256_file(profile_path)},
        "output_root": str(output), "fitter_provenance": execution.runtime_provenance(),
        "policy": {"overwrite_original_artifacts": False, "new_grid_search": False,
                   "amended_cell": "J0/largest_N_act", "changed_tolerances": False,
                   "changed_selection_rule": False, "heldout_metrics_select_audit_starts": False,
                   "amended_coefficients_have_new_result": True},
    }, "plan_sha256")
    path = output / "control" / "recovery_plan.json"
    execution.write_json_once(path, plan)
    logs = output / "logs" / "attempts" / plan["plan_sha256"][:12]
    logs.mkdir(parents=True, exist_ok=True)
    script = "\n".join([
        "#!/bin/bash", "#SBATCH --job-name=s1br-recovery-comparison", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={profile['completion_memory']}", f"#SBATCH --time={profile['completion_time_limit']}",
        f"#SBATCH --partition={profile['partition']}", f"#SBATCH --account={profile['account']}",
        f"#SBATCH --dependency=afterok:{audit_job_id}",
        f"#SBATCH --output={logs / 'complete-%j.out'}", "set -euo pipefail",
        f"cd {shlex.quote(str(repository_root()))}",
        *(f"export {name}=1" for name in amendment.THREADS), "export PYTHONPATH=src:.",
        shlex.join([str(Path(profile["python_executable"]).resolve()), "-m", MODULE,
                    "complete", "--plan", str(path)]), "",
    ])
    execution.write_text_once(output / "control" / "complete.sbatch", script)
    return path


def load_plan(path: Path, *, live: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = execution.read_json(path)
    verify_seal(plan, "plan_sha256", label="baseline recovery plan")
    if plan.get("schema_version") != PROTOCOL or plan.get("protocol") != PROTOCOL \
            or plan.get("policy") != {"overwrite_original_artifacts": False, "new_grid_search": False,
                "amended_cell": "J0/largest_N_act", "changed_tolerances": False,
                "changed_selection_rule": False, "heldout_metrics_select_audit_starts": False,
                "amended_coefficients_have_new_result": True}:
        raise BaselineRefitError("baseline recovery protocol changed")
    _validate_sources(plan["source"], live=live)
    if live and not execution._current_provenance_matches(plan["fitter_provenance"]):
        raise BaselineRefitError("fitter changed after recovery planning")
    execution.validate_recorded_provenance(plan["fitter_provenance"])
    wp, _ = amendment.bound(plan["workflow"], "workflow_sha256")
    workflow = _original_workflow(wp)
    for value in plan["original_gates"].values():
        amendment.bound(value, "gate_sha256")
    for value in plan["manifests"].values():
        amendment.bound(value, "manifest_sha256")
    amendment.bound(plan["audit_plan"], "plan_sha256")
    amendment.bound(plan["ordinary_audits"], "receipt_sha256")
    return plan, workflow


def amended_result_loader(plan: Mapping[str, Any]):
    from . import refit_audit
    evidence = refit_audit.validate_acceptance(Path(plan["audit_plan"]["path"]))
    target = Path(evidence["refined_result"]["path"]).resolve()

    def load(path: Path) -> Mapping[str, Any]:
        if path.resolve() != target:
            return campaign.scientific_result(path)
        _, result = amendment.bound(evidence["refined_result"], "result_sha256")
        campaign._completion_receipt(result)
        return {**result, "audit_amendment": evidence}

    return load


def complete(path: Path) -> dict[str, Any]:
    from . import refit_audit
    plan, workflow = load_plan(path)
    if not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(k) != "1" for k in amendment.THREADS):
        raise BaselineRefitError("recovery completion requires a Slurm CPU allocation and single-threaded BLAS")
    output = Path(plan["output_root"])
    if (output / "receipts" / "finalization.json").exists():
        return verify(path)
    load_result = amended_result_loader(plan)
    results = reporting.result_matrix(workflow)
    evidence = refit_audit.validate_acceptance(Path(plan["audit_plan"]["path"]))
    results["J0"]["largest_N_act"] = Path(evidence["refined_result"]["path"])
    for model in MODELS:
        for fold in FOLDS:
            key = f"{model}/{fold}"
            if key == "J0/largest_N_act":
                load_result(results[model][fold])
                continue
            mp = Path(plan["manifests"][key]["path"])
            manifest = execution.load_manifest(mp)
            rp = execution.result_path(manifest)
            if not rp.exists():
                print(f"Reducing {key} from existing chunks", flush=True)
                execution.reduce(manifest)
                gc.collect()
            result = campaign._reduced(rp)
            ap = campaign.audit_path(result, output_root=rp.parent.parent)
            if not ap.exists():
                print(f"Auditing {key}", flush=True)
                campaign.audit_grid(mp)
                gc.collect()
            # No new cell is silently amended if its ordinary checks fail.
            load_result(rp)
            print(f"Accepted {key}", flush=True)
    candidate_results = {model: results[model] for model in CANDIDATES}
    candidate = fit_report.build_candidate_report(
        input_receipt_path=Path(workflow["inputs"]["input_receipt"]["path"]),
        holdouts_path=Path(workflow["inputs"]["holdouts"]["path"]), results=candidate_results)
    reproduction = reporting.build_reproduction_report(
        workflow, fresh_candidate=candidate, fresh_results=candidate_results)
    reproduction_path = output / "reports" / "candidate_reproduction.json"
    execution.write_json_once(reproduction_path, reproduction)
    if reproduction["status"] != "complete":
        raise BaselineRefitError("fresh candidate fits do not reproduce the original comparison")
    candidate_path, _ = fit_report.write_candidate_report(output, candidate)
    comparison = reporting.build_comparison_report(workflow,
        candidate_report_path=candidate_path, reproduction_report_path=reproduction_path,
        results=results, result_loader=load_result)
    unsigned = dict(comparison)
    unsigned.pop("report_sha256")
    unsigned["recovery_plan"] = amendment.binding(path, "plan_sha256")
    comparison = seal(unsigned, "report_sha256")
    comparison_path = output / "reports" / "primary_baseline_comparison.json"
    execution.write_json_once(comparison_path, comparison)
    execution.write_text_once(output / "reports" / "primary_baseline_comparison.md",
        reporting._markdown(comparison) + "\nCompleted using a separately audited, conditioned J0 numerical refit; "
        "the original failed audit and blocked workflow gates are preserved.\n")
    from . import recovery_diagnostics
    diagnostics_path = output / "reports" / "residual_diagnostics.json"
    execution.write_json_once(diagnostics_path,
        recovery_diagnostics.build(comparison_path, results, load_result))
    load_plan(path)  # all original inputs and receipts must still be unchanged
    final = seal({"schema_version": PROTOCOL, "status": "complete",
        "plan": amendment.binding(path, "plan_sha256"),
        "audit_amendment": evidence,
        "reports": {"candidate": amendment.binding(candidate_path, "report_sha256"),
                    "reproduction": amendment.binding(reproduction_path, "report_sha256"),
                    "comparison": amendment.binding(comparison_path, "report_sha256"),
                    "residual_diagnostics": amendment.binding(diagnostics_path, "report_sha256")},
        "strict_baseline_win": comparison["strict_baseline_win"],
        "original_failed_receipts_preserved": True,
        "job_id": os.environ["SLURM_JOB_ID"]}, "receipt_sha256")
    execution.write_json_once(output / "receipts" / "finalization.json", final)
    return verify(path)


def collect_original_audits(profile_path: Path) -> dict[str, Any]:
    """Finish missing ordinary audits independently of the failed J0 cell.

    This does not authorize an amendment or publish a comparison. Existing
    reduced results and audits are only read; missing ones use the unchanged
    original numerical runner and its full chunk validation.
    """
    if not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(k) != "1" for k in amendment.THREADS):
        raise BaselineRefitError("ordinary audit collection requires a single-threaded Slurm CPU job")
    profile = json.loads(profile_path.read_text())
    workflow_path = _resolve(profile["workflow_manifest"])
    workflow = _original_workflow(workflow_path)
    source = _sources()
    receipt_path = _resolve(profile["output_root"]) / "receipts" / "ordinary_audits.json"
    if receipt_path.exists():
        raise BaselineRefitError("ordinary audit collection already has a receipt")
    records = []
    for model in MODELS:
        for fold in FOLDS:
            job = f"grid.robust.{model}.{fold}" if model in BASELINES else f"grid.{model.lower()}.{fold}"
            mp = manifest_path_for_job(workflow, job)
            manifest = execution.load_manifest(mp)
            rp = execution.result_path(manifest)
            print(f"Checking {model}/{fold}", flush=True)
            if not rp.exists():
                execution.reduce(manifest)
                gc.collect()
            result = campaign._reduced(rp)
            ap = campaign.audit_path(result, output_root=rp.parent.parent)
            if not ap.exists():
                print(f"Running ordinary audit for {model}/{fold}", flush=True)
                campaign.audit_grid(mp)
                gc.collect()
            audit = execution.read_json(ap)
            verify_seal(audit, "audit_sha256", label=str(ap))
            if audit["status"] == "complete":
                campaign.scientific_result(rp)
            item = {"model": model, "fold": fold, "manifest": amendment.binding(mp, "manifest_sha256"),
                    "result": amendment.binding(rp, "result_sha256"), "audit": amendment.binding(ap, "audit_sha256"),
                    "status": audit["status"], "reason": audit.get("reason")}
            records.append(item)
            print(json.dumps({k: item[k] for k in ("model", "fold", "status", "reason")}), flush=True)
    _validate_sources(source, live=True)
    receipt = seal({"schema_version": "stage1_baseline_ordinary_audit_collection/v1",
        "workflow": amendment.binding(workflow_path, "workflow_sha256"), "source": source,
        "job_id": os.environ["SLURM_JOB_ID"], "cells": records,
        "status": "collected", "all_accepted": all(v["status"] == "complete" for v in records)}, "receipt_sha256")
    execution.write_json_once(receipt_path, receipt)
    return receipt


def verify(path: Path) -> dict[str, Any]:
    plan, _ = load_plan(path, live=False)
    final_path = Path(plan["output_root"]) / "receipts" / "finalization.json"
    final = execution.read_json(final_path)
    verify_seal(final, "receipt_sha256", label="recovery finalization")
    if final.get("schema_version") != PROTOCOL or final.get("status") != "complete" \
            or final.get("plan") != amendment.binding(path, "plan_sha256") \
            or final.get("original_failed_receipts_preserved") is not True:
        raise BaselineRefitError("recovery finalization is not complete")
    for value in final["reports"].values():
        amendment.bound(value, "report_sha256")
    _, reproduction = amendment.bound(final["reports"]["reproduction"], "report_sha256")
    _, comparison = amendment.bound(final["reports"]["comparison"], "report_sha256")
    if reproduction["status"] != "complete" or comparison["status"] != "complete" \
            or final["strict_baseline_win"] != comparison["strict_baseline_win"]:
        raise BaselineRefitError("recovery report statuses disagree")
    load_result = amended_result_loader(plan)
    for source in comparison["source_results"]:
        result = load_result(Path(source["path"]))
        if sha256_file(Path(source["path"])) != source["file_sha256"] \
                or result["result_sha256"] != source["result_sha256"]:
            raise BaselineRefitError("a final comparison source changed")
    return {"status": "complete", "strict_baseline_win": final["strict_baseline_win"],
            "receipt": str(final_path), "reports": final["reports"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--audit-job-id", required=True)
    commands.add_parser("collect-original").add_argument("--profile", type=Path, required=True)
    for name in ("complete", "verify"):
        commands.add_parser(name).add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.profile, audit_job_id=args.audit_job_id))
    elif args.command == "collect-original":
        print(json.dumps(collect_original_audits(args.profile), indent=2))
    else:
        print(json.dumps(complete(args.plan) if args.command == "complete" else verify(args.plan), indent=2))


if __name__ == "__main__":
    main()
