"""Same-optimizer M6 null control, frozen before opening M6-ND heldout scores.

No original artifact or main campaign source changes. Each of three CPU tasks
refits the same eight original M6 anchors with zeta fixed exactly to zero,
then runs the same three-basin-plus-direct numerical certificate (14 free
shared parameters). This separates equation improvement from numerical polish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import execution as core
from .m6_size_duration_workflow import execution as e, model as m, planning as p, reporting as r

NAME = "M6-null"
MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.m6_null_control"


def box(parent, fold):
    return (*p.bounds(parent, fold)[:14], SimpleNamespace(name="zeta", lower=0., upper=0., kind="fixed"))


def health(profile, result, bounds):
    if result["coordinates"][14] != 0.:
        raise ValueError("null control must fix zeta exactly to zero")
    value = m.numerical_health(profile, result, bounds)
    rejected = [c for c in value["rejected_boundary_contacts"] if c["name"] != "zeta"]
    return value | {"required_rank": 14, "rejected_boundary_contacts": rejected,
        "rank_scope": "14 free shared parameters, fixed zeta column zero; nuisance held fixed",
        "healthy": bool(value["shared_prediction_jacobian_rank"] == 14 and not rejected
            and value["minimum_alpha"] > 0 and value["minimum_beta"] > 0 and value["projected_gradient_inf"] <= 1e-8)}


def load(path):
    control = p.read_sealed(path)
    parent = p.load(p.diagnosis.bound_file(control["parent_plan"]))
    if parent["sha256"] != control["parent_sha256"]:
        raise ValueError("control parent changed")
    for item in [*control["sources"], control["archive"], control["script"]]:
        p.diagnosis.bound_file(item)
    p.verify_committed_sources(control["source_commit"], control["sources"])
    return control, parent


def prepare(parent_path):
    parent_path = Path(parent_path).resolve()
    parent = p.load(parent_path)
    if (parent_path.parent/"reports").exists():
        raise ValueError("freeze null control before opening main heldout scores")
    output = parent_path.parent/"null_control"
    if output.exists():
        raise ValueError("control packet is create-only")
    sources = [p.binding(Path(__file__)), p.binding(p.repository_root()/"tests/test_stage1_m6_null_control.py")]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p.repository_root(), text=True).strip()
    p.verify_committed_sources(commit, sources)
    output.mkdir()
    archive = output/"source.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        for item in sources:
            handle.add(item["path"], arcname=str(Path(item["path"]).relative_to(p.repository_root())), recursive=False)
    profile = json.loads(Path(parent["profile"]["path"]).read_text())
    script = output/"run.sbatch"
    text = ["#!/bin/bash", "#SBATCH --job-name=m6nd-null", "#SBATCH --array=0-2%3",
        f"#SBATCH --partition={profile['partition']}", f"#SBATCH --account={profile['account']}",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={profile['memory']}", f"#SBATCH --time={profile['time_limit']}",
        f"#SBATCH --output={parent_path.parent}/logs/null-%A_%a.out", "set -euo pipefail",
        f"cd {shlex.quote(str(p.repository_root()))}", "export PYTHONPATH=src:.",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"{shlex.quote(profile['python_executable'])} -m {MODULE} --control {shlex.quote(str(output/'plan.json'))} worker --task \"$SLURM_ARRAY_TASK_ID\"", ""]
    core.write_text_once(script, "\n".join(text))
    subprocess.run(["bash", "-n", str(script)], check=True)
    return p.write_sealed(output/"plan.json", {"schema_version": "stage1_m6_same_optimizer_null/v1",
        "parent_plan": p.binding(parent_path), "parent_sha256": parent["sha256"], "source_commit": commit,
        "sources": sources, "archive": p.binding(archive), "script": p.binding(script), "output_root": str(output),
        "protocol": "eight original M6 anchors per fold; same objective, rows, nuisance bounds and lambda=.01; zeta fixed exactly zero; three basin plus direct checks; same objective/prediction/KKT thresholds, free shared rank 14",
        "grid_starts": 24, "audit_starts": 12, "heldout_scores_opened_at_preparation": False,
        "changes_main_thresholds": False})


def worker(control_path, task):
    control, parent = load(control_path)
    if not 0 <= task < 3:
        raise ValueError("null task outside three folds")
    path = Path(control["output_root"])/f"fold-{task}.json"
    if path.exists():
        value = p.read_sealed(path)
        if value["control_sha256"] != control["sha256"]:
            raise ValueError("control result identity changed")
        return value
    context = e.runtime_context()
    fold = p.FOLDS[task]
    profile = m.Profile(p.fold_problem(parent, fold), parent["folds"][fold]["manifest"]["fit"]["delta"])
    bounds = box(parent, fold)
    starts = [{"fold": fold, "start_id": f"null:{a['start_id']}", "coordinates": [*a["coordinates"], 0.]}
        for a in parent["folds"][fold]["anchors"]]
    attempts = [e.run_attempt(profile, start, bounds) for start in starts]
    selected, audit_starts = e.select_audits(attempts)
    audits = []
    for kind, start in [*(('basin', a) for a in audit_starts), ('direct', selected)]:
        try:
            result = (m.Direct(profile, start, bounds).run() if kind == "direct"
                else profile.run(start["coordinates"], bounds))
            equivalent = m.equivalence(selected, result, profile.problem)
            numerical = health(profile, result, bounds)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            result = {"success": False, "message": str(error)}
            equivalent, numerical = {"equivalent": False}, {"healthy": False}
        audits.append({"kind": kind, "start_id": start["start_id"], "result": result,
            "equivalence": equivalent, "health": numerical,
            "status": "healthy" if equivalent["equivalent"] and numerical["healthy"] else "failed"})
    return p.write_sealed(path, {"control_sha256": control["sha256"], "fold": fold, "runtime": context,
        "attempts": attempts, "selected": selected, "audits": audits,
        "status": "healthy" if all(a["status"] == "healthy" for a in audits) else "failed",
        "heldout_scores_computed": False})


def report(control_path):
    control, parent = load(control_path)
    output = Path(control["output_root"])
    results = [p.read_sealed(output/f"fold-{i}.json") for i in range(3)]
    for i, value in enumerate(results):
        if value["control_sha256"] != control["sha256"] or value["fold"] != p.FOLDS[i] or value["status"] != "healthy":
            raise ValueError("null control lacks a complete healthy numerical audit")
    main_path = Path(parent["output_root"])/"reports/report.json"
    main = p.read_sealed(main_path)
    if main["status"] != "complete" or main["plan_sha256"] != parent["sha256"] or not main["numerical_audit_healthy"]:
        raise ValueError("main candidate must pass its numerical audits before scoring")
    predictions = [v for v in main["outputs"] if Path(v["path"]).name == "predictions.parquet"]
    if len(predictions) != 1:
        raise ValueError("missing main prediction artifact")
    frame = pd.read_parquet(p.diagnosis.bound_file(predictions[0]))
    reduced = {"folds": {v["fold"]: {"selected": v["selected"]} for v in results}}
    null = r.candidate_rows(parent, reduced).assign(model=NAME)
    combined = pd.concat([frame, null], ignore_index=True)
    statistics = r.metrics_and_bootstrap(combined)
    # Every preexisting metric must remain exactly unchanged in this add-on.
    for name, values in main["statistics"]["metrics"].items():
        if statistics["metrics"][name] != values:
            raise ValueError("null control changed a main reported metric")
    path = output/"predictions.parquet"
    if path.exists():
        raise ValueError("null predictions already exist")
    null.to_parquet(path, index=False)
    return p.write_sealed(output/"report.json", {"control_sha256": control["sha256"], "status": "complete",
        "main_report": p.binding(main_path), "fold_results": [v["sha256"] for v in results],
        "statistics": statistics, "predictions": p.binding(path), "changes_main_thresholds": False,
        "interpretation": "same-optimizer M6 null control, frozen before candidate heldout scoring; development evidence"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("--parent", type=Path, required=True)
    sub.add_parser("worker").add_argument("--task", type=int, required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.parent)
    elif args.control is None:
        parser.error("--control required")
    elif args.action == "worker":
        result = worker(args.control, args.task)
    else:
        result = report(args.control)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status") == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
