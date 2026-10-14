"""Create-only, CPU-only first repetition round; no test-based law selector.

The worker executes frozen source and training-only packets. Reporting is a
separate action; inspected holdouts and L26 remain development diagnostics.
"""
from __future__ import annotations

from .artifacts import binding, verify, environment, validate_output
from .problems import inventories, partition, problem

import argparse
from dataclasses import asdict
import gzip
from importlib.metadata import version
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

from . import execution as core, fit, repetition_model as model
from .data import canonical_json_sha256, load_canonical_fit_data, FIT_ROW_COLUMNS

SCHEMA = "stage1_repetition_round/v1"
PROTOCOL = {
    "models": list(model.MODELS), "rate_starts": [0., .1, 1.], "rate_bounds": [0., 10.],
    "folds": ["largest_N_act", "E32", "K200_common_languages"],
    "equation": "L_inf + A*C6 + B*T6*(N/1e8)^zeta*(S_lambda(D_t,U_t)/D_t)^(-beta(E))",
    "M6-R_zeta": 0., "M6-ND-R_zeta_bounds": [0., .5],
    "composition": "raw family/script ratios and capacity remain unchanged",
    "selection": "lowest training objective per fixed law/fold; no law selection on test results",
    "scientific_status": "bounded post-observation development; not prospective confirmation",
    "gradient_tolerance": 1e-8, "objective_tolerance": 1e-6, "prediction_tolerance": 1e-4,
    "score_win_tolerance_nats": 1e-6, "starts": 18, "direct_audits": 6,
    "max_core_hours": 16., "retries": 0, "primary_only_first_round": True,
    "deferred": ["secondary_holdouts", "all_grid_comparison", "training_only_law_selection",
                 "GPU_evaluation_parity", "prospective_confirmation"],
}


def round_protocol(number):
    if number == 1:
        return PROTOCOL
    if number != 2:
        raise ValueError("unregistered repetition round")
    return PROTOCOL | {"models": ["M6-ND-R", "M6-ND-RN"], "round": 2,
        "equation": "v1 with lambda(N)=lambda_0*(N/1e8)^rho; rho=0 nests shared-rate control",
        "rho_bounds": [-2., 2.], "rho_starts": [-1., 0., 1.],
        "control_rate_multipliers": [.5, 1., 2.], "optimizer": "SLSQP, ftol=1e-12, maxiter=4000",
        "positive_exponent_constraint": "alpha_e1+gamma_n and alpha_e32+gamma_n >= 1e-6",
        "inner_amendment": "existing strict IRLS fallback for BVLS out-of-bounds initialization only",
        "max_core_hours": 12., "time_limit": "00:30:00",
        "reason": "v1 wins primary grid but not L26; test one size/repetition interaction and same-optimizer control"}


def prepare(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    protocol = round_protocol(profile.get("round", 1))
    if str(Path(sys.executable).resolve()) != str(Path(profile["python_executable"]).resolve()):
        raise ValueError("prepare with the configured bridge-train interpreter")
    output = validate_output(profile)
    if output.exists():
        raise ValueError("repetition preparation is create-only")
    comparison = Path(profile["comparison"]).resolve()
    receipt = json.loads((comparison/"manifest.json").read_text())
    for item in [*receipt["inputs"], *receipt["outputs"]]:
        verify(item)
    shared, literature = Path(profile["shared_fits"]), Path(profile["literature"])
    consumed = [profile_path, comparison/"manifest.json", Path(profile["parent_plan"]),
                *(shared/n for n in ("selected.json", "folds.json", "fit_rows.parquet", "predictions.parquet")),
                *(literature/n for n in ("packets.json.gz", "predictions.parquet", "model_status.csv"))]
    input_bindings = [binding(p) for p in consumed]
    upstream = None
    if "read_only_root" in profile:
        source = profile["read_only_root"]
        upstream = {"root": str(Path(source).resolve()),
                    "git_head": subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip(),
                    "git_status": subprocess.check_output(["git", "-C", source, "status", "--short"], text=True)}
    selected = json.loads((shared/"selected.json").read_text())
    folds = json.loads((shared/"folds.json").read_text())
    parent = json.loads(Path(profile["parent_plan"]).read_text())
    rows = pd.read_parquet(shared/"fit_rows.parquet")
    if "canonical_input_receipt" in profile:
        receipt_path = Path(profile["canonical_input_receipt"])
        canonical = json.loads(receipt_path.read_text())
        validated = load_canonical_fit_data(canonical["release_root"],
            archive_manifest_path=canonical["archive_manifest_path"],
            release_contract_path=canonical["release_contract_path"])
        if validated.content_sha256 != canonical["fit_rows_content_sha256"]:
            raise ValueError("canonical corrected release content changed")
        keys = ["trajectory_id", "readout_id", "language"]
        expected = pd.DataFrame([r.to_dict() for r in validated.rows])
        pd.testing.assert_frame_equal(
            rows[list(FIT_ROW_COLUMNS)].sort_values(keys).reset_index(drop=True),
            expected[list(FIT_ROW_COLUMNS)].sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=True)
        input_bindings.extend([binding(receipt_path), binding(canonical["archive_manifest_path"]),
                               binding(canonical["release_contract_path"]),
                               *(binding(t["path"]) for t in canonical["source_tables"])])
    packets = json.loads(gzip.decompress((literature/"packets.json.gz").read_bytes()))
    inv = inventories(packets)
    rows["U_t"] = [inv[l, int(k)] for l, k in zip(rows.language, rows.K)]
    endpoint = pd.read_parquet(comparison/"endpoint_inputs.parquet")
    endpoint["U_t"] = [inv[l, int(k)] for l, k in zip(endpoint.language, endpoint.K)]
    if len(endpoint) != 200 or endpoint.D_total.nunique() != 1 or int(endpoint.D_total.iloc[0]) != 68682776576:
        raise ValueError("wrong long-run endpoint")
    if rows.duplicated(["trajectory_id", "readout_id", "language"]).any():
        raise ValueError("duplicate grid row identity")
    cells, tasks, preflight, prior_inputs = [], [], [], []
    prior = None
    if profile.get("round", 1) == 2:
        prior_path = Path(profile["prior_round"]).resolve()/"plan.json"
        prior = load(prior_path)
        prior_inputs = [binding(prior_path), binding(prior_path.parent/"report/report.json")]
    for fold in protocol["folds"]:
        member = folds[fold]["membership"]
        pfit = parent["folds"][fold]["manifest"]["fit"]
        for key in ("train_trajectory_ids", "test_trajectory_ids", "score_languages"):
            if member[key] != parent["folds"][fold]["manifest"]["membership"][key]:
                raise ValueError("parent and frozen paper memberships differ")
        training = partition(rows, member, "train")
        testing = partition(rows, member, "test")
        if set(testing.language) != set(member["score_languages"]):
            raise ValueError("test language coverage differs")
        for name in protocol["models"]:
            if prior is None:
                source = selected[fold]["M6" if name == "M6-R" else "M6-ND"]
                x = [*fit.outer_from_shared("M6", fit.SharedParameters(**source["shared_parameters"])), source["zeta"], 0.]
            else:
                prior_cell, = [c for c in prior["cells"] if c["fold"] == fold and c["model"] == "M6-ND-R"]
                paths = [prior_path.parent/"fit"/f"{t['task_id']:03d}.json" for t in prior["tasks"]
                         if t["cell_id"] == prior_cell["cell_id"]]
                prior_inputs.extend(binding(p) for p in paths)
                x = list(selected_attempt([core.read_json(p) for p in paths])["coordinates"])
            bounds = [model.Bound(**b) for b in parent["folds"][fold]["bounds"]]
            if name == "M6-R":
                bounds[14] = model.Bound("zeta", 0., 0., "nonnegative")
            bounds.append(model.Bound("lambda_repetition", 0., 10., "nonnegative"))
            if name == "M6-ND-RN":
                x.append(0.)
                bounds.append(model.Bound("rho", -2., 2., "signed"))
            prob = problem(training, pfit["regularization"])
            old_terms = model.parent.terms(x[:15], prob)
            null = np.array(x); null[15] = 0.
            new_terms = model.terms(null, prob, training.U_t.to_numpy())
            for old, new in zip(old_terms[:2], new_terms[:2]):
                np.testing.assert_allclose(old, new, atol=1e-13, rtol=1e-13)
            preflight.append({"fold": fold, "model": name, "zero_rate_parent_parity": True})
            cell = {"cell_id": len(cells), "fold": fold, "model": name, "membership": member,
                    "delta": pfit["delta"], "regularization": pfit["regularization"],
                    "bounds": [asdict(b) for b in bounds]}
            cells.append(cell)
            if prior is None:
                starts = [[*x[:15], rate] for rate in protocol["rate_starts"]]
            elif name == "M6-ND-RN":
                starts = [[*x[:16], rho] for rho in protocol["rho_starts"]]
            else:
                starts = [[*x[:15], x[15]*multiplier] for multiplier in protocol["control_rate_multipliers"]]
            for initial in starts:
                tasks.append({"task_id": len(tasks), "cell_id": cell["cell_id"], "initial": initial})
    baseline = pd.read_parquet(shared/"predictions.parquet").query("split == 'test'")
    native = pd.read_parquet(literature/"predictions.parquet").query("split == 'test'")
    statuses = pd.read_csv(literature/"model_status.csv")
    eligible = set(statuses.loc[statuses.status.eq("certified"), "model"])
    native = native[native.model.isin(eligible)]
    keep = ["fold", "model", "trajectory_id", "readout_id", "language", "adjusted_loss", "prediction"]
    baselines = pd.concat([baseline[keep], native[keep]], ignore_index=True)
    endpoint_baselines = pd.read_csv(comparison/"predictions.csv")
    for entry in [*input_bindings, *receipt["inputs"], *receipt["outputs"], *prior_inputs]:
        verify(entry)
    output.mkdir(parents=True)
    rows.to_parquet(output/"training_grid.parquet", index=False)
    endpoint.to_parquet(output/"endpoint.parquet", index=False)
    # A standalone frozen package avoids dirty-worktree/commit changes at runtime.
    frozen = output/"source"/"stage1_fit"
    shutil.copytree(Path(__file__).parent, frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    baselines.to_parquet(output/"baseline_grid.parquet", index=False)
    shutil.copy2(comparison/"predictions.csv", output/"baseline_endpoint.csv")
    shutil.copy2(literature/"model_status.csv", output/"baseline_status.csv")
    plan = {"schema": SCHEMA, "protocol": protocol, "profile": profile, "cells": cells,
            "tasks": tasks, "preflight": preflight, "environment": environment(),
            "runtime": {"python_executable": str(Path(sys.executable).resolve()), "pyarrow": version("pyarrow")},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "upstream": upstream,
            "command": shlex.join([sys.executable, "-m", __spec__.name, "prepare", "--profile", str(profile_path)]),
            "baseline_models": {"grid": sorted(baselines.model.unique()),
                                "endpoint": sorted(endpoint_baselines.model.unique()),
                                "unresolved_literature": sorted(statuses.loc[~statuses.status.eq("certified"), "model"])},
            "inputs": [*input_bindings, *receipt["inputs"], *receipt["outputs"], *prior_inputs],
            "frozen_inputs": [binding(output/n) for n in ("training_grid.parquet", "endpoint.parquet",
                "baseline_grid.parquet", "baseline_endpoint.csv", "baseline_status.csv")],
            "source": [binding(p) for p in sorted(frozen.rglob("*.py"))]}
    plan["sha256"] = canonical_json_sha256(plan)
    core.write_json_once(output/"plan.json", plan)
    for action, count in (("fit", len(tasks)), ("audit", len(cells))):
        script, logs = render_job(output/"plan.json", plan, action, count)
        logs.mkdir(parents=True)
        path = output/f"{action}.sbatch"
        core.write_text_once(path, script)
        subprocess.run(["bash", "-n", str(path)], check=True)
    return plan


def render_job(plan_path, plan, action, count):
    """Group each create-only fit/audit invocation's logs; keep results stable."""
    if action not in ("fit", "audit") or type(count) is not int or count <= 0:
        raise ValueError("invalid repetition array")
    output = Path(plan_path).resolve().parent
    profile, protocol = plan["profile"], plan["protocol"]
    # Fit and audit are separate invocations; the plan hash binds their identity.
    logs = output/"logs"/"attempts"/f"{action}-{plan['sha256'][:12]}"
    command = shlex.join([profile["python_executable"], "-m", "stage1_fit.repetition_campaign",
                         action, "--plan", str(plan_path)])
    script = ["#!/bin/bash", f"#SBATCH --job-name=stage1_repetition_{action}",
        f"#SBATCH --account={profile['account']}", f"#SBATCH --partition={profile['partition']}",
        "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={profile['memory']}",
        f"#SBATCH --time={protocol.get('time_limit', '00:40:00')}",
        f"#SBATCH --array=0-{count-1}%{profile['concurrency']}",
        f"#SBATCH --output={shlex.quote(str(logs/(action+'-%A_%a.out')))}", "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(output/'source'))}",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        command+' --task "$SLURM_ARRAY_TASK_ID"', ""]
    return "\n".join(script), logs


def load(plan_path):
    plan = core.read_json(Path(plan_path))
    unsigned = dict(plan)
    digest = unsigned.pop("sha256")
    if (digest != canonical_json_sha256(unsigned) or plan["schema"] != SCHEMA
            or plan["protocol"] != round_protocol(plan["profile"].get("round", 1))):
        raise ValueError("changed repetition protocol")
    for entry in [*plan["frozen_inputs"], *plan["source"]]:
        verify(entry)
    if plan["environment"] != environment():
        raise ValueError("numerical environment changed")
    if "runtime" in plan and plan["runtime"] != {
            "python_executable": str(Path(sys.executable).resolve()), "pyarrow": version("pyarrow")}:
        raise ValueError("pinned interpreter or parquet runtime changed")
    return plan


def verify_preparation(plan_path):
    """Read-only check of the frozen package and its external input bindings."""
    plan = load(plan_path)
    for entry in plan["inputs"]:
        verify(entry)
    print(json.dumps({"status": "prepared_not_run", "plan_sha256": plan["sha256"],
                      "models": plan["protocol"]["models"], "fit_starts": len(plan["tasks"]),
                      "audit_cells": len(plan["cells"]), "input_bindings": len(plan["inputs"])}))


def selected_attempt(attempts):
    eligible = [r for r in attempts if r.get("success")]
    if not eligible:
        raise ValueError("no stationary successful start")
    best = min(eligible, key=lambda r: (r["objective"], r["task_id"]))
    lower = [r for r in attempts if r.get("objective", float("inf")) < best["objective"]*(1-1e-6)]
    if lower:
        raise ValueError("a nonstationary attempt has a better objective")
    return best


def worker(plan_path, action, task_id):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("real optimization requires a CPU Slurm allocation")
    plan = load(plan_path)
    root = Path(plan_path).resolve().parent
    destination = root/action/f"{task_id:03d}.json"
    if destination.exists():
        raise ValueError("attempt already exists; no overwrite or automatic retry")
    task = plan["tasks"][task_id] if action == "fit" else {"cell_id": task_id}
    cell = plan["cells"][task["cell_id"]]
    rows = partition(pd.read_parquet(root/"training_grid.parquet"), cell["membership"], "train")
    profile = model.Profile(problem(rows, cell["regularization"]), cell["delta"], rows.U_t.to_numpy(),
                            constrained=plan["profile"].get("round", 1) == 2)
    bounds = tuple(model.Bound(**b) for b in cell["bounds"])
    began = time.monotonic()
    try:
        if action == "fit":
            result = profile.run(task["initial"], bounds)
        else:
            attempts = [core.read_json(root/"fit"/f"{t['task_id']:03d}.json") for t in plan["tasks"]
                        if t["cell_id"] == cell["cell_id"]]
            best = selected_attempt(attempts)
            direct = model.Direct(profile, best, bounds).run()
            difference = abs(direct["objective"]-best["objective"])/max(abs(best["objective"]), 1e-30)
            maximum = float(np.max(abs(model.predict(profile.problem, profile.inventory, best)
                                      - model.predict(profile.problem, profile.inventory, direct))))
            _, gradient, _, jac = profile.evaluate(best["coordinates"])
            free = np.array([b.upper > b.lower for b in bounds])
            scaled = jac[:, free]*np.array([b.upper-b.lower for b in bounds])[free]
            singular = np.linalg.svd(scaled, compute_uv=False)
            rank = int(np.sum(singular > singular[0]*1e-8))
            best_prediction = model.predict(profile.problem, profile.inventory, best)
            basin = [a for a in attempts if a.get("success")
                     and abs(a["objective"]-best["objective"])/max(abs(best["objective"]), 1e-30) <= 1e-6
                     and np.max(abs(model.predict(profile.problem, profile.inventory, a)-best_prediction)) <= 1e-4]
            result = {"success": bool(direct["success"] and difference <= 1e-6 and maximum <= 1e-4),
                "selected_task_id": best["task_id"], "direct": direct,
                "basin_members": len(basin),
                "relative_objective_difference": difference, "maximum_prediction_difference": maximum,
                "projected_gradient_inf": model.parent.projected_gradient(best["coordinates"], gradient, bounds),
                "shared_prediction_jacobian_rank": rank, "free_shared_coordinates": int(free.sum()),
                "identifiability_scope": "nuisance held fixed; not global identifiability"}
    except (ValueError, RuntimeError, FloatingPointError) as error:
        result = {"success": False, "error": str(error), "traceback": traceback.format_exc()}
    result.update(task_id=task_id, cell_id=cell["cell_id"], model=cell["model"], fold=cell["fold"],
                  plan_sha256=plan["sha256"], slurm_job_id=os.environ["SLURM_JOB_ID"],
                  elapsed_seconds=time.monotonic()-began)
    core.write_json_once(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("language_coefficients", "direct", "traceback")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "fit", "audit", "report"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.profile)
    elif args.action == "verify":
        verify_preparation(args.plan)
    elif args.action == "report":
        from .repetition_reporting import report
        report(args.plan)
    else:
        worker(args.plan, args.action, args.task)


if __name__ == "__main__":
    main()
