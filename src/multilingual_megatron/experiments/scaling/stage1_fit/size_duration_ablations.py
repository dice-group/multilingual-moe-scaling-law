"""Two fixed, post-diagnosis ablations for the joint-story paper revision.

Reuse the exact M6-ND numerical implementation with fixed coordinates. J0-ND
has seven free coordinates; M5-ND has twelve. The former's ridge is identically
zero on its feasible set. Historical code, scientific gates and receipts stay
immutable. Real optimization runs only inside a one-core Slurm task.
"""
from __future__ import annotations

from .duration_ablation import FIXED, PROTOCOL, AblationBound, restrict, restricted_bounds, health, certify_attempts

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np
import pandas as pd

from . import execution as core, fit
from .m6_size_duration_workflow import execution as e, model as m, planning as p, reporting as r

MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.size_duration_ablations"
SCHEMA = "stage1_size_duration_ablations/v1"


def load(path):
    plan = p.read_sealed(path)
    if plan["schema_version"] != SCHEMA or plan["protocol"] != PROTOCOL:
        raise ValueError("ablation protocol changed")
    parent = p.load(p.diagnosis.bound_file(plan["parent"]))
    for item in [*plan["sources"], *plan["bindings"], plan["archive"], plan["script"]]:
        p.diagnosis.bound_file(item)
    if plan["environment"] != p.environment():
        raise ValueError("numerical environment changed")
    return plan, parent


def prepare(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    parent_path = p.diagnosis.bound_file(profile["parent"])
    parent = p.load(parent_path)
    comparison_path = p.diagnosis.bound_file(profile["comparison"])
    comparison = json.loads(comparison_path.read_text())
    root = p.repository_root()
    output = (root/profile["output_root"]).resolve()
    if output.parent != root/"result_analysis" or not output.name.startswith("stage1_size_duration_ablations_"):
        raise ValueError("ablation output must own a separate result directory")
    if output.exists():
        raise ValueError("ablation preparation is create-only")
    if str(Path(profile["python_executable"]).resolve()) != p.environment()["executable"]:
        raise ValueError("prepare with the configured bridge-train interpreter")
    tasks, bindings, preflight = [], [p.binding(profile_path), p.binding(comparison_path)], []
    for name in FIXED:
        for fold in p.FOLDS:
            jbox = None
            if name == "J0-ND":
                item, = [v for v in comparison["source_results"] if (v["model"], v["fold"]) == ("J0", fold)]
                source_path = p.diagnosis.bound_file(item)
                result = json.loads(source_path.read_text())
                manifest = core.load_manifest(Path(result["manifest_path"]))
                if result["membership"] != parent["folds"][fold]["manifest"]["membership"]:
                    raise ValueError("J0 and M6-ND memberships differ")
                if manifest["fit"]["delta"] != parent["folds"][fold]["manifest"]["fit"]["delta"]:
                    raise ValueError("J0 and M6-ND robust scales differ")
                anchors = sorted([a for a in result["attempts"] if a.get("success")],
                                 key=lambda a: (a["objective"], a["start_id"]))[:8]
                anchors = [dict(a, coordinates=[*a["coordinates"], *([0.]*8)]) for a in anchors]
                jbox = tuple(fit.OuterBound(**b) for b in result["fit_bounds"])
                bindings += [p.binding(source_path), p.binding(Path(result["manifest_path"]))]
            else:
                anchors = parent["folds"][fold]["anchors"]
            if len(anchors) != 8:
                raise ValueError("exactly eight successful parent anchors required")
            bounds = restricted_bounds(p.bounds(parent, fold), name, jbox)
            starts = []
            for i, a in enumerate(anchors):
                for j, zeta in enumerate(m.ZETA_STARTS):
                    x = restrict([*a["coordinates"], zeta], name)
                    starts.append({"start_id": f"{name}:{fold}:a{i}:z{j}", "fold": fold,
                                   "coordinates": x.tolist(), "parent_start_id": a["start_id"]})
            problem = p.fold_problem(parent, fold)
            x = np.asarray(starts[0]["coordinates"])
            cap, data, _, _ = m.terms(x, problem)
            base = "J0" if name == "J0-ND" else "M5"
            old = fit.predictor_terms_and_jacobian(base, x[:6] if base == "J0" else x[:11],
                problem.N, problem.D, problem.E, problem.K)
            if not all(np.allclose(a, b, rtol=1e-14, atol=1e-14) for a, b in zip((cap, data), old[:2])):
                raise ValueError("ablation null does not reproduce its parent")
            x[-1] = .1
            terms = m.terms(x, problem)
            errors = []
            for i, b in enumerate(bounds):
                if b.lower == b.upper:
                    continue
                step = np.eye(15)[i]*1e-6
                plus, minus = m.terms(x+step, problem), m.terms(x-step, problem)
                for k in (0, 1):
                    errors.append(float(np.max(abs((plus[k]-minus[k])/2e-6-terms[k+2][:, i])
                        /np.maximum(1., abs(terms[k+2][:, i])))))
            if max(errors) > 1e-7:
                raise ValueError("ablation basis gradient check failed")
            preflight.append({"model": name, "fold": fold, "null_parity": True, "basis_gradient_error": max(errors)})
            tasks.append({"task_id": len(tasks), "model": name, "fold": fold,
                          "bounds": [asdict(b) for b in bounds], "starts": starts})
    output.mkdir()
    (output/"logs").mkdir()
    sources = [p.binding(Path(__file__)), p.binding(root/"tests/test_stage1_size_duration_ablations.py")]
    # Parent p.load separately certifies every numerical dependency against its
    # original committed bytes. New code is preserved exactly in this archive.
    archive = output/"source.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        for item in sources:
            handle.add(item["path"], arcname=str(Path(item["path"]).relative_to(root)), recursive=False)
    script = output/"run.sbatch"
    lines = ["#!/bin/bash", "#SBATCH --job-name=story-ablation", "#SBATCH --array=0-5%6",
        f"#SBATCH --partition={profile['partition']}", f"#SBATCH --account={profile['account']}",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={profile['memory']}", f"#SBATCH --time={profile['time_limit']}",
        f"#SBATCH --output={output}/logs/%A_%a.out", "set -euo pipefail",
        f"cd {shlex.quote(str(root))}", "export PYTHONPATH=src:.",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"{shlex.quote(profile['python_executable'])} -m {MODULE} --plan {shlex.quote(str(output/'plan.json'))} worker --task \"$SLURM_ARRAY_TASK_ID\"", ""]
    core.write_text_once(script, "\n".join(lines))
    subprocess.run(["bash", "-n", str(script)], check=True)
    return p.write_sealed(output/"plan.json", {"schema_version": SCHEMA, "protocol": PROTOCOL,
        "parent": p.binding(parent_path), "sources": sources, "bindings": bindings,
        "archive": p.binding(archive), "script": p.binding(script), "environment": p.environment(),
        "source_provenance": "exact hash-bound source archive, including new uncommitted files",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "output_root": str(output), "tasks": tasks, "preflight": preflight,
        "previous_holdouts_inspected": True, "new_heldout_scores_computed": False})


def worker(plan_path, task_id):
    plan, parent = load(plan_path)
    if not 0 <= task_id < 6:
        raise ValueError("task outside fixed six-cell comparison")
    output = Path(plan["output_root"])
    path = output/f"cell-{task_id}.json"
    if path.exists():
        value = p.read_sealed(path)
        if value["plan_sha256"] != plan["sha256"]:
            raise ValueError("cell identity changed")
        return value
    task = plan["tasks"][task_id]
    context = e.runtime_context()
    bounds = tuple(AblationBound(**b) for b in task["bounds"])
    profile = m.Profile(p.fold_problem(parent, task["fold"]), parent["folds"][task["fold"]]["manifest"]["fit"]["delta"])
    attempts = []
    for start in task["starts"]:
        index = len(attempts)
        attempt_path = output/f"attempts/{task_id}-{index:02d}.json"
        if attempt_path.exists():
            receipt = p.read_sealed(attempt_path)
            if receipt["plan_sha256"] != plan["sha256"] or receipt["start"] != start:
                raise ValueError("attempt identity changed")
            attempt = receipt["result"]
        else:
            attempt = e.run_attempt(profile, start, bounds)
            p.write_sealed(attempt_path, {"plan_sha256": plan["sha256"], "start": start, "result": attempt})
        attempts.append(attempt)
        print(task["model"], task["fold"], index+1, attempt["success"], flush=True)
    checked = certify_attempts(profile, bounds, task["model"], attempts)
    return p.write_sealed(path, {"plan_sha256": plan["sha256"], "model": task["model"], "fold": task["fold"],
        "runtime": context, "attempts": attempts, **checked,
        "heldout_scores_computed": False})


def report(plan_path):
    plan, parent = load(plan_path)
    output = Path(plan["output_root"])
    cells = [p.read_sealed(output/f"cell-{i}.json") for i in range(6)]
    for i, cell in enumerate(cells):
        task = plan["tasks"][i]
        if cell["plan_sha256"] != plan["sha256"] or any(cell[k] != task[k] for k in ("model", "fold")):
            raise ValueError("cell identity changed")
    main_path = Path(parent["output_root"])/"reports/report.json"
    main = p.read_sealed(main_path)
    prediction_binding, = [v for v in main["outputs"] if Path(v["path"]).name == "predictions.parquet"]
    frame = pd.read_parquet(p.diagnosis.bound_file(prediction_binding))
    additions, statuses = [], {}
    for name in FIXED:
        subset = [c for c in cells if c["model"] == name]
        statuses[name] = "complete" if all(c["status"] == "healthy" for c in subset) else "unresolved"
        if statuses[name] == "complete":
            reduced = {"folds": {c["fold"]: {"selected": c["selected"]} for c in subset}}
            additions.append(r.candidate_rows(parent, reduced).assign(model=name))
    combined = pd.concat([frame, *additions], ignore_index=True)
    statistics = r.metrics_and_bootstrap(combined)
    if any(statistics["metrics"][name] != values for name, values in main["statistics"]["metrics"].items()):
        raise ValueError("ablation report changed historical metrics")
    path = output/"predictions.parquet"
    if path.exists():
        raise ValueError("report is create-only")
    combined.to_parquet(path, index=False)
    return p.write_sealed(output/"report.json", {"plan_sha256": plan["sha256"], "models": statuses,
        "status": "complete" if all(s == "complete" for s in statuses.values()) else "completed_with_unresolved_models",
        "cells": [p.binding(output/f"cell-{i}.json") for i in range(6)], "main_report": p.binding(main_path),
        "predictions": p.binding(path), "statistics": statistics,
        "starts_successful": sum(a["success"] for c in cells for a in c["attempts"]), "starts_total": 288,
        "audits_healthy": sum(a["status"] == "healthy" for c in cells for a in c["audits"]), "audits_expected": 24,
        "interpretation": PROTOCOL["scientific_status"], "historical_gate_changed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("--profile", type=Path, required=True)
    sub.add_parser("worker").add_argument("--task", type=int, required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.profile)
    elif args.plan is None:
        parser.error("--plan required")
    elif args.action == "worker":
        result = worker(args.plan, args.task)
    else:
        result = report(args.plan)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tasks", "attempts", "audits", "selected")}, indent=2))


if __name__ == "__main__":
    main()
