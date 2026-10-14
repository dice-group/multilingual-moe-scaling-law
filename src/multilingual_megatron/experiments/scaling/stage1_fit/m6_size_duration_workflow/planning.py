"""Frozen development protocol, source-byte archive, and review-only preparation."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import scipy

from .. import execution, fit, extrapolation_diagnostics as diagnosis
from ..baseline_refit_workflow.contracts import FOLDS, repository_root, seal, verify_seal
from ..data import sha256_file
from ..m5_m6_workflow import campaign
from . import model

SCHEMA = "stage1_m6_size_duration/v1"
MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration"
PROTOCOL = {
    "model": model.MODEL, "zeta_bounds": [0., .5], "zeta_starts": list(model.ZETA_STARTS),
    "anchor_count": 8, "chunk_size": 2, "folds": list(FOLDS), "grid_starts": 144,
    "pilot_tasks": [0, 1, 2], "pilot_reused_in_grid": True, "audit_starts": 12,
    "anchor_order": "successful original M6 lambda=.01 attempts by training objective then start_id",
    "audit_order": "best three successful grid start IDs by training objective then start_id; direct from best",
    "objective_tolerance": 1e-6, "prediction_tolerance": 1e-4, "gradient_tolerance": 1e-8,
    "rank_relative_tolerance": 1e-8, "bootstrap_replicates": 10000, "bootstrap_seed": 20260905,
    "short_total_tokens": 500039680., "long_total_tokens": 32000442368.,
    "pilot_max_projected_core_hours": 20., "pilot_cost_safety_multiplier": 2.,
    "gates": {"mean_rmse_below_m6": .049631289555012066,
        "largest_rmse_below_j0": .05098924060389545,
        "e32_rmse_at_most_m6": .037903055080416984,
        "k200_rmse_at_most_m6": .05304186691098596,
        "positive_zeta_all_folds": True,
        "short_C_D_absolute_gain_error_reduced": True,
        "largest_32B_E32_K12_K50_rmse_not_worse": True},
    "secondary": "E32 K200 minus K12 at matched target exposure, observed log-exposure interpolation, no extrapolation; report only",
    "evidence": "development on already inspected holdouts, not confirmatory; no new model training",
}


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "file_sha256": sha256_file(path)}


def environment():
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def read_sealed(path):
    value = execution.read_json(Path(path))
    verify_seal(value, "sha256", label=str(path))
    return value


def write_sealed(path, value):
    payload = seal(value, "sha256")
    execution.write_json_once(Path(path), payload)
    return payload


def load(path):
    plan = read_sealed(path)
    if plan["schema_version"] != SCHEMA or plan["protocol"] != PROTOCOL:
        raise ValueError("unknown or changed size-duration protocol")
    for value in [*plan["bindings"], *plan["sources"], plan["source_archive"], plan["profile"]]:
        diagnosis.bound_file(value)
    verify_committed_sources(plan["git_head"], plan["sources"])
    if environment() != plan["environment"]:
        raise ValueError("Python/numerical-library environment differs from prepared packet")
    validate_tasks(plan["tasks"])
    return plan


def verify_committed_sources(commit, sources):
    root = repository_root()
    for value in sources:
        relative = str(Path(value["path"]).relative_to(root))
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        if hashlib.sha256(content).hexdigest() != value["file_sha256"]:
            raise ValueError(f"fitting source differs from recorded commit: {relative}")


def tasks_for(anchors):
    starts = []
    for fold in FOLDS:
        for i, anchor in enumerate(anchors[fold]):
            for j, zeta in enumerate(model.ZETA_STARTS):
                starts.append({"fold": fold, "start_id": f"{fold}:a{i}:z{j}",
                    "anchor_start_id": anchor["start_id"], "coordinates": [*anchor["coordinates"], zeta]})
    return [{"task_id": i//2, "starts": starts[i:i+2]} for i in range(0, len(starts), 2)]


def validate_tasks(tasks):
    if len(tasks) != 72 or [t["task_id"] for t in tasks] != list(range(72)) or any(len(t["starts"]) != 2 for t in tasks):
        raise ValueError("grid must contain exactly 72 two-start tasks")
    ids = [s["start_id"] for t in tasks for s in t["starts"]]
    expected = [f"{f}:a{i}:z{j}" for f in FOLDS for i in range(8) for j in range(6)]
    if ids != expected:
        raise ValueError("start ordering/coverage changed")


def fold_problem(plan, fold):
    # Load only training rows; never restrict training to score_languages.
    return execution.load_problem(plan["folds"][fold]["manifest"])


def bounds(plan, fold):
    return tuple(fit.OuterBound(**v) for v in plan["folds"][fold]["bounds"])


def preflight(folds):
    checks = []
    for fold, value in folds.items():
        p = execution.load_problem(value["manifest"])
        old = value["m6_selected"]
        x = np.r_[old["coordinates"], 0.]
        cap, data, cj, dj = model.terms(x, p)
        a, b, aj, bj = fit.predictor_terms_and_jacobian("M6", x[:14], p.N, p.D, p.E, p.K,
            D_family=p.D_family, D_script=p.D_script, D_other=p.D_other)
        exact = all(np.array_equal(l, r) for l, r in ((cap, a), (data, b), (cj[:, :14], aj), (dj[:, :14], bj)))
        if not exact:
            raise ValueError("zeta=0 does not reproduce M6 exactly")
        box = value["bounds"]
        x = np.clip(x, [b["lower"]+1e-4 for b in box], [b["upper"]-1e-4 for b in box])
        x[14] = .1
        # Central differences of every basis derivative, no optimizer calls.
        _, _, cj, dj = model.terms(x, p)
        max_error = 0.
        for j in range(15):
            direction = np.eye(15)[j]*1e-6
            plus, minus = model.terms(x+direction, p), model.terms(x-direction, p)
            for k, analytic in ((0, cj), (1, dj)):
                numeric = (plus[k]-minus[k])/2e-6
                error = float(np.max(np.abs(numeric-analytic[:, j]) / np.maximum(1., np.abs(analytic[:, j]))))
                max_error = max(max_error, error)
        if max_error > 1e-7:
            raise ValueError(f"basis gradient check failed: {fold}: {max_error}")
        checks.append({"fold": fold, "training_rows": len(p.y), "exact_null_basis_and_jacobian": exact,
            "maximum_scaled_basis_gradient_error": max_error})
    return {"status": "passed", "optimizer_starts": 0, "checks": checks}


def scripts(output, profile):
    executable = shlex.quote(profile["python_executable"])
    plan_path = shlex.quote(str(output/"plan.json"))
    command = f"{executable} -m {MODULE} --plan {plan_path}"
    specifications = {
        "pilot": ("0-2%3", "worker --task \"$SLURM_ARRAY_TASK_ID\""),
        "pilot_gate": (None, "pilot-gate"),
        "grid": ("3-71%12", "worker --task \"$SLURM_ARRAY_TASK_ID\""),
        "reduce": (None, "reduce"),
        "audit": ("0-11%12", "audit --task \"$SLURM_ARRAY_TASK_ID\""),
        "report": (None, "report"),
    }
    paths = []
    for name, (array, action) in specifications.items():
        text = ["#!/bin/bash", f"#SBATCH --job-name=m6nd-{name}", f"#SBATCH --partition={profile['partition']}",
            f"#SBATCH --account={profile['account']}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={profile['memory']}",
            f"#SBATCH --time={profile['time_limit']}", f"#SBATCH --output={output}/logs/{name}-%A_%a.out"]
        if array:
            text.append(f"#SBATCH --array={array}")
        text += ["set -euo pipefail", f"cd {shlex.quote(str(repository_root()))}",
            "export PYTHONPATH=src:.", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
            f"{command} {action}", ""]
        path = output/"slurm"/f"{name}.sbatch"
        execution.write_text_once(path, "\n".join(text))
        subprocess.run(["bash", "-n", str(path)], check=True)
        paths.append(binding(path))
    return paths


def prepare(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    if profile["schema_version"] != SCHEMA or profile["protocol"] != "frozen-v1":
        raise ValueError("unknown profile")
    if str(Path(profile["python_executable"]).resolve()) != environment()["executable"]:
        raise ValueError("prepare with the configured bridge-train interpreter")
    root = repository_root()
    output = (root/profile["output_root"]).resolve()
    if output.parent != root/"result_analysis" or not output.name.startswith("stage1_m6_size_duration_"):
        raise ValueError("campaign must own a separate result_analysis/stage1_m6_size_duration_* directory")
    if output.exists():
        raise ValueError("prepare is create-only; verify existing packet or choose a new version")
    dp_path = diagnosis.bound_file(profile["diagnostics_profile"])
    dp, paths = diagnosis.load_profile(dp_path)
    predictions, evidence, _ = diagnosis.collect_predictions(dp, paths)
    index = campaign._index(paths["m6_index"])
    folds, anchors, bindings = {}, {}, [binding(p) for p in paths.values()]+[binding(dp_path)]
    parent = None
    if "numerical_parent_pilot" in profile:
        gate_path = diagnosis.bound_file(profile["numerical_parent_pilot"])
        gate = read_sealed(gate_path)
        parent_path = gate_path.parent/"plan.json"
        parent = read_sealed(parent_path)
        if gate["status"] != "blocked" or gate["heldout_scores_computed"] or gate["plan_sha256"] != parent["sha256"]:
            raise ValueError("numerical parent must be an exact blocked, unscored pilot")
        if parent["protocol"] != PROTOCOL:
            raise ValueError("numerical repair cannot change the scientific protocol")
        bindings += [binding(gate_path), binding(parent_path)]
    if "numerical_parent_audit" in profile:
        blocked_path = diagnosis.bound_file(profile["numerical_parent_audit"])
        blocked = read_sealed(blocked_path)
        parent_path = blocked_path.parent.parent/"plan.json"
        parent = read_sealed(parent_path)
        if blocked["status"] != "numerically_blocked" or blocked["heldout_scores_computed"] \
                or blocked["plan_sha256"] != parent["sha256"] or parent["protocol"] != PROTOCOL:
            raise ValueError("audit repair must preserve an exact unscored numerical parent and protocol")
        bindings += [binding(blocked_path), binding(parent_path)]
        for task_id, digest in enumerate(blocked["audit_sha256"]):
            audit_path = parent_path.parent/"audits"/f"{task_id:02d}.json"
            audit = read_sealed(audit_path)
            if audit["sha256"] != digest or audit["plan_sha256"] != parent["sha256"]:
                raise ValueError("parent numerical audit changed")
            if audit["status"] != "healthy":
                eq = audit["equivalence"]
                if not audit["health"]["healthy"] or eq["relative_objective_difference"] > 1e-6 \
                        or eq["maximum_absolute_prediction_difference"] > 1e-4 \
                        or audit["result"]["optimizer_success"] or audit["result"]["status"] != 2:
                    raise ValueError("repair is restricted to stationary line-search termination failures")
            bindings.append(binding(audit_path))
    for fold in FOLDS:
        manifest_path = Path(index["manifests"]["M6"]["0.01"][fold])
        result, manifest = campaign._campaign_result(manifest_path, index)
        candidates = sorted((a for a in result["attempts"] if a["success"]), key=lambda a: (a["objective"], a["start_id"]))[:8]
        if len(candidates) != 8 or manifest["fit"]["regularization"]["lambda"] != .01:
            raise ValueError("missing eight successful lambda=.01 M6 anchors")
        anchors[fold] = [{k: a[k] for k in ("start_id", "coordinates", "objective")} for a in candidates]
        folds[fold] = {"manifest": manifest, "m6_result_sha256": result["result_sha256"],
            "m6_selected": result["selected"], "anchors": anchors[fold],
            "bounds": manifest["fit"]["bounds"]+[asdict(fit.OuterBound("zeta", 0., .5, "nonnegative"))]}
        bindings += [binding(manifest_path), binding(execution.result_path(manifest)),
            binding(manifest["input"]["fit_rows_path"])]
    checks = preflight(folds)
    tasks = tasks_for(anchors)
    validate_tasks(tasks)
    if parent is not None and tasks != parent["tasks"]:
        raise ValueError("numerical repair changed frozen start coordinates or IDs")
    output.mkdir(parents=True)
    (output/"logs").mkdir()
    # Exact accepted baseline predictions are frozen, not recomputed by workers.
    baseline_path = output/"baseline_predictions.parquet"
    predictions.to_parquet(baseline_path, index=False)
    bindings += [binding(baseline_path), *scripts(output, profile)]
    source_paths = sorted(Path(__file__).parents[1].rglob("*.py"))
    source_paths += [root/"result_analysis/capacity_response"/name
                     for name in ("__init__.py", "matched_exposure.py", "tables.py")]
    source_paths += [root/"result_analysis/scaling_law_fit"/name
                     for name in ("__init__.py", "residuals.py")]
    source_paths += [root/"tests/test_stage1_m6_size_duration.py"]
    # Include package initializers outside stage1_fit as well.
    source_paths += [p/"__init__.py" for p in Path(__file__).parents[2:7] if (p/"__init__.py").is_file()]
    sources = [binding(p) for p in sorted(set(source_paths))]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    verify_committed_sources(commit, sources)
    archive = output/"source.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        for item in sources:
            handle.add(item["path"], arcname=str(Path(item["path"]).relative_to(root)), recursive=False)
    plan = {"schema_version": SCHEMA, "status": "prepared_unsubmitted", "protocol": PROTOCOL,
        "output_root": str(output), "profile": binding(profile_path), "environment": environment(),
        "source_mode": "committed and archived exact fitting source; unrelated working-tree changes allowed",
        "git_head": commit,
        "source_archive": binding(archive), "sources": sources, "bindings": bindings,
        "historical_evidence": evidence, "preflight": checks, "folds": folds, "tasks": tasks,
        "baseline_predictions": binding(baseline_path)}
    write_sealed(output/"plan.json", plan)
    load(output/"plan.json")
    return {"status": "prepared_unsubmitted", "plan": str(output/"plan.json"), "preflight": checks,
        "grid_starts": 144, "pilot_starts_reused": 6, "audit_starts": 12, "jobs_submitted": 0}
