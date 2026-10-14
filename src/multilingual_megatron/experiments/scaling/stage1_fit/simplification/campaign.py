"""Frozen, CPU-only compact-law screen. No coefficient fitting sees test rows."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .. import execution as core
from .. import repetition_campaign as reuse
from . import model
from ..data import canonical_json_sha256
from ..m6_size_duration_workflow import planning

SCHEMA = "stage1_simplification/v1"
PROTOCOL = {
    "models": list(model.MODELS), "folds": ["largest_N_act", "E32", "K200_common_languages"],
    "equation": "c_t+A_t*E^delta*K^(phi+theta*log(E))*n^(-alpha)+B_t*E^omega*K^psi*n^zeta*d_t^(-beta)",
    "coordinates": "phi_e1,phi_e32 with theta=(phi_e32-phi_e1)/log(32); inherited endpoint bounds",
    "restrictions": {"S7-no-ND": "zeta=0", "S7-no-EK": "theta=0"},
    "shared_counts": {"S8": 8, "S7-no-ND": 7, "S7-no-EK": 7},
    "nuisance": "three coefficients per training language; original training-only bounds",
    "objective": "original trajectory weights, Huber delta and exactly mapped M6-ND ridge",
    "starts_per_cell": 24, "rng_seed": 20260912,
    "starts": "matching training-fold M5-ND projection; 23 seeded jitters; zeta cycles 0,.1,.3,.5",
    "selection": "lowest successful training objective per law/fold; reject lower nonstationary start; no law selection",
    "audits": "independent joint fit, at least two agreeing starts, fixed-nuisance shared Jacobian rank and boundary check",
    "gradient_tolerance": 1e-8, "objective_tolerance": 1e-6, "prediction_tolerance": 1e-4,
    "score_win_tolerance_nats": 1e-6, "retries": 0,
    "max_core_hours": 4., "allocation_cap_core_hours": 3.6,
    "cells": 9, "time_limit": "00:24:00", "concurrency": 8,
    "primary_cases": "three axis holdouts with original row membership and trajectory weighting",
    "diagnostics": "three frozen-fold L26 forecasts; largest-size per-horizon residuals",
    "comparators": "51 frozen baseline/control laws plus separately labelled two repetition candidates",
    "scientific_status": "post-observation development; not prospective confirmation",
    "deferred": ["new duration-split refits", "matched pooling refits", "all-grid refits",
                 "training-only selection between laws", "fresh training confirmation"],
}


def prepare(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    if Path(sys.executable).resolve() != Path(profile["python_executable"]).resolve():
        raise ValueError("use configured bridge-train interpreter")
    output = Path(profile["output_root"]).resolve()
    if output.exists():
        raise ValueError("prepare is create-only")
    prior_path = Path(profile["prior_plan"]).resolve()
    prior = reuse.load(prior_path)
    prior_root = prior_path.parent
    grid = pd.read_parquet(prior_root/"training_grid.parquet")
    statuses = pd.read_csv(prior_root/"baseline_status.csv")
    native = set(statuses.loc[statuses.status.eq("certified"), "model"])
    shared = {"C0", "J0", "J0-ND", "M0", "M1", "M2", "M5-ND", "M6", "M6-ND"}
    baseline_grid = pd.read_parquet(prior_root/"baseline_grid.parquet")
    baseline_endpoint = pd.read_csv(prior_root/"baseline_endpoint.csv")
    if len(native) != 42:
        raise ValueError("certified baseline inventory changed")
    for fold in PROTOCOL["folds"]:
        for frame in (baseline_grid, baseline_endpoint):
            if set(frame.loc[frame.fold.eq(fold), "model"]) != native | shared:
                raise ValueError("baseline scorecard must retain all 51 models")
    ablations_path = Path(profile["ablations_report"]).resolve()
    ablations = planning.read_sealed(ablations_path)
    anchor_plan_path = ablations_path.parent/"plan.json"
    anchor_plan = planning.read_sealed(anchor_plan_path)
    if ablations["plan_sha256"] != anchor_plan["sha256"]:
        raise ValueError("anchor report/plan mismatch")
    anchor_parent_path = planning.diagnosis.bound_file(anchor_plan["parent"])
    anchor_parent = planning.read_sealed(anchor_parent_path)
    anchors = {}
    inputs = [reuse.binding(profile_path), reuse.binding(prior_path), reuse.binding(ablations_path),
              reuse.binding(anchor_plan_path), reuse.binding(anchor_parent_path)]
    for entry in ablations["cells"]:
        path = planning.diagnosis.bound_file(entry)
        value = planning.read_sealed(path)
        if value["model"] == "M5-ND":
            if value["status"] != "healthy" or value["plan_sha256"] != anchor_plan["sha256"]:
                raise ValueError("unaudited training anchor")
            anchors[value["fold"]] = value["selected"]["coordinates"]
            inputs.append(reuse.binding(path))
    cells, packets = [], {}
    for fold_index, fold in enumerate(PROTOCOL["folds"]):
        previous = next(c for c in prior["cells"] if c["fold"] == fold)
        if any(previous["membership"].get(k) for k in ("train_readout_ids", "test_readout_ids")):
            raise ValueError("this screen requires whole-configuration splits")
        anchor_manifest = anchor_parent["folds"][fold]["manifest"]
        anchor_membership = {k: v for k, v in anchor_manifest["membership"].items() if k != "path"}
        previous_membership = {k: v for k, v in previous["membership"].items() if k != "path"}
        if anchor_membership != previous_membership:
            raise ValueError("warm-start training membership differs")
        if any(anchor_manifest["fit"][k] != previous[k] for k in ("delta", "regularization")):
            raise ValueError("warm-start fitting objective differs")
        training = reuse.partition(grid, previous["membership"], "train")
        testing = reuse.partition(grid, previous["membership"], "test")
        if training.trajectory_id.nunique() != 48 or testing.trajectory_id.nunique() != 16:
            raise ValueError("wrong configuration coverage")
        packets[fold] = training, testing
        parent_box = tuple(model.Bound(**b) for b in previous["bounds"][:15])
        for name in model.MODELS:
            box = model.bounds(parent_box, name)
            x = model.restrict(anchors[fold], name)
            low, high = np.array([(b.lower, b.upper) for b in box]).T
            rng = np.random.default_rng(PROTOCOL["rng_seed"]+fold_index)
            starts = [x.tolist()]
            for i in range(1, PROTOCOL["starts_per_cell"]):
                jitter = rng.normal(size=len(x))*np.array([.10 if b.kind == "positive" else .05 for b in box])
                start = np.clip(x+jitter, low+1e-5, high-1e-5)
                if box[-1].name == "zeta":
                    start[-1] = (0., .1, .3, .5)[i % 4]
                starts.append(start.tolist())
            cells.append({"cell_id": len(cells), "model": name, "fold": fold,
                          "membership": previous["membership"], "delta": previous["delta"],
                          "regularization": previous["regularization"],
                          "bounds": [asdict(b) for b in box], "starts": starts})
    # Preparation has not fitted or evaluated any candidate on heldout targets.
    output.mkdir(parents=True)
    for directory in ("logs", "packets", "cells", "attempts"):
        (output/directory).mkdir()
    for fold, (training, testing) in packets.items():
        training.to_parquet(output/"packets"/f"{fold}-train.parquet", index=False)
        testing.to_parquet(output/"packets"/f"{fold}-test.parquet", index=False)
    for filename in ("endpoint.parquet", "baseline_grid.parquet", "baseline_endpoint.csv", "baseline_status.csv"):
        shutil.copy2(prior_root/filename, output/filename)
    prior_report = core.read_json(prior_root/"report/report.json")
    inputs.append(reuse.binding(prior_root/"report/report.json"))
    for entry in prior_report["outputs"]:
        reuse.verify(entry)
        inputs.append(entry)
    for filename in ("predictions.parquet", "metrics.csv"):
        shutil.copy2(prior_root/"report"/filename, output/f"recent_{filename}")
    frozen = output/"source"/"stage1_fit"
    shutil.copytree(Path(__file__).parents[1], frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    test_path = Path(profile["test_file"]).resolve()
    shutil.copy2(test_path, output/"test_stage1_simplification.py")
    plan = {"schema": SCHEMA, "protocol": PROTOCOL, "profile": profile, "cells": cells,
            "environment": reuse.environment(), "inputs": inputs,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source": [reuse.binding(p) for p in sorted(frozen.rglob("*.py"))],
            "frozen_inputs": [reuse.binding(p) for p in sorted(output.glob("*.parquet"))]
                + [reuse.binding(p) for p in sorted(output.glob("*.csv"))]
                + [reuse.binding(p) for p in sorted((output/"packets").glob("*.parquet"))]
                + [reuse.binding(output/"test_stage1_simplification.py")],
            "new_heldout_scores_computed": False}
    command = shlex.join([profile["python_executable"], "-m", "stage1_fit.simplification",
                         "worker", "--plan", str(output/"plan.json")])
    script = ["#!/bin/bash", "#SBATCH --job-name=stage1_simple", "#SBATCH --array=0-8%8",
              f"#SBATCH --account={profile['account']}", f"#SBATCH --partition={profile['partition']}",
              "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
              f"#SBATCH --mem={profile['memory']}", "#SBATCH --time=00:24:00", "#SBATCH --no-requeue",
              f"#SBATCH --output={output}/logs/%A_%a.out", "set -euo pipefail",
              f"export PYTHONPATH={shlex.quote(str(output/'source'))}",
              "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
              command+' --task "$SLURM_ARRAY_TASK_ID"', ""]
    core.write_text_once(output/"run.sbatch", "\n".join(script))
    subprocess.run(["bash", "-n", str(output/"run.sbatch")], check=True)
    plan["script"] = reuse.binding(output/"run.sbatch")
    plan["sha256"] = canonical_json_sha256(plan)
    core.write_json_once(output/"plan.json", plan)
    return {"plan": str(output/"plan.json"), "sha256": plan["sha256"]}


def load(plan_path, *, training_cell=None):
    plan = core.read_json(Path(plan_path))
    unsigned = dict(plan)
    digest = unsigned.pop("sha256")
    if digest != canonical_json_sha256(unsigned) or plan["schema"] != SCHEMA or plan["protocol"] != PROTOCOL:
        raise ValueError("changed simplification plan")
    # Workers only open their own training packet; test targets are report-only.
    entries = plan["frozen_inputs"]
    if training_cell is not None:
        filename = plan["cells"][training_cell]["fold"]+"-train.parquet"
        entries = [v for v in entries if Path(v["path"]).name == filename]
        if len(entries) != 1:
            raise ValueError("missing training packet")
    for entry in [*entries, *plan["source"], plan["script"]]:
        reuse.verify(entry)
    if plan["environment"] != reuse.environment():
        raise ValueError("numerical environment changed")
    return plan


def worker(plan_path, task_id):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("fits require a one-core Slurm allocation")
    if task_id not in range(9):
        raise ValueError("invalid cell")
    plan = load(plan_path, training_cell=task_id)
    root = Path(plan_path).resolve().parent
    destination = root/"cells"/f"{task_id:03d}.json"
    if destination.exists() or list((root/"attempts").glob(f"{task_id:03d}-*.json")):
        raise ValueError("cell already started; no retry or overwrite")
    cell = plan["cells"][task_id]
    rows = pd.read_parquet(root/"packets"/f"{cell['fold']}-train.parquet")
    if set(rows.trajectory_id) != set(cell["membership"]["train_trajectory_ids"]):
        raise ValueError("training packet membership differs")
    profile = model.Profile(reuse.problem(rows, cell["regularization"]), cell["delta"], cell["model"])
    box = tuple(model.Bound(**b) for b in cell["bounds"])
    began = time.monotonic()
    attempts = []
    for index, initial in enumerate(cell["starts"]):
        try:
            if time.monotonic()-began > 22*60:
                raise RuntimeError("cell fitting time budget exhausted")
            attempt = profile.run(initial, box)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempt = {"success": False, "error": str(error), "traceback": traceback.format_exc()}
        attempt.update(task_id=index, cell_id=task_id, plan_sha256=plan["sha256"], initial=initial)
        core.write_json_once(root/"attempts"/f"{task_id:03d}-{index:02d}.json", attempt)
        attempts.append(attempt)
        print(cell["model"], cell["fold"], index, attempt["success"], flush=True)
    selected, audit = None, {"success": False}
    try:
        selected = reuse.selected_attempt(attempts)
        audit = model.audit(profile, selected, attempts, box)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        audit = {"success": False, "error": str(error), "traceback": traceback.format_exc()}
    result = {"cell_id": task_id, "model": cell["model"], "fold": cell["fold"],
              "plan_sha256": plan["sha256"], "selected": selected, "audit": audit,
              "successful_starts": sum(a["success"] for a in attempts), "attempts": len(attempts),
              "status": "healthy" if audit["success"] else "unresolved",
              "elapsed_seconds": time.monotonic()-began, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "heldout_targets_loaded": False}
    core.write_json_once(destination, result)
    return {k: v for k, v in result.items() if k not in ("selected", "audit")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "report"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.profile)
    elif args.action == "worker":
        result = worker(args.plan, args.task)
    else:
        from .reporting import report
        result = report(args.plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
