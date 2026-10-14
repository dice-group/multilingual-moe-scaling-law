"""Create-only workers and training-only pilot/reduction gates. No submission API."""
from __future__ import annotations

from .model import run_attempt

import os
from pathlib import Path
import socket
import time

import numpy as np

from . import model, planning as p


def runtime_context():
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("real-data optimization requires a Slurm allocation")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"{key} must equal 1")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) != 1:
        raise ValueError("this campaign is one CPU core per task")
    return {"job_id": os.environ["SLURM_JOB_ID"], "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "hostname": socket.gethostname(),
        "unix_time": time.time()}


def artifact(plan, name):
    return Path(plan["output_root"])/name


def read_artifact(plan, name):
    value = p.read_sealed(artifact(plan, name))
    if value["plan_sha256"] != plan["sha256"]:
        raise ValueError(f"artifact belongs to a different campaign: {name}")
    return value


def gate_required(plan):
    if read_artifact(plan, "pilot_gate.json")["status"] != "passed":
        raise ValueError("pilot gate has not passed")


def validate_chunk(plan, task_id, chunk):
    task = plan["tasks"][task_id]
    if chunk["task_id"] != task_id or chunk["plan_sha256"] != plan["sha256"]:
        raise ValueError("chunk identity changed")
    records = chunk["attempts"]
    if len(records) != 2:
        raise ValueError("missing or duplicate chunk records")
    for record, start in zip(records, task["starts"]):
        if any(record[k] != start[k] for k in ("start_id", "fold")) or record["initial_coordinates"] != start["coordinates"]:
            raise ValueError("chunk start identity changed")
        if record["success"] and (not np.isfinite(record["objective"]) or len(record["coordinates"]) != 15):
            raise ValueError("invalid successful fit")
    return records


def worker(plan_path, task_id):
    plan = p.load(plan_path)
    if not 0 <= task_id < 72:
        raise ValueError("task outside frozen grid")
    if task_id not in p.PROTOCOL["pilot_tasks"]:
        gate_required(plan)
    path = artifact(plan, f"chunks/{task_id:03d}.json")
    if path.exists():
        return {"status": "already_complete", "attempts": validate_chunk(plan, task_id, p.read_sealed(path))}
    context = runtime_context()
    task = plan["tasks"][task_id]
    fold = task["starts"][0]["fold"]
    profile = model.Profile(p.fold_problem(plan, fold), plan["folds"][fold]["manifest"]["fit"]["delta"])
    attempts = [run_attempt(profile, start, p.bounds(plan, fold)) for start in task["starts"]]
    return p.write_sealed(path, {"plan_sha256": plan["sha256"], "task_id": task_id,
        "runtime": context, "attempts": attempts})


def pilot_decision(attempts):
    runtimes = [a["runtime_seconds"] for a in attempts]
    # Six largest-size starts cannot benchmark the other folds or direct solver.
    # Apply a conservative 2x multiplier to the slowest pilot start for all 156.
    estimate = max(runtimes)*156/3600*p.PROTOCOL["pilot_cost_safety_multiplier"]
    passed = len(attempts) == 6 and all(a["success"] for a in attempts) and estimate <= p.PROTOCOL["pilot_max_projected_core_hours"]
    return {"status": "passed" if passed else "blocked", "successful_starts": sum(a["success"] for a in attempts),
        "starts": len(attempts), "runtime_seconds": runtimes, "projected_core_hours_with_safety_factor": estimate,
        "heldout_scores_computed": False,
        "interpretation": "Convergence and runtime only; this is an estimate, not a hard campaign cost limit."}


def pilot_gate(plan_path):
    plan = p.load(plan_path)
    attempts = [a for t in p.PROTOCOL["pilot_tasks"]
        for a in validate_chunk(plan, t, read_artifact(plan, f"chunks/{t:03d}.json"))]
    return p.write_sealed(artifact(plan, "pilot_gate.json"), {"plan_sha256": plan["sha256"], **pilot_decision(attempts)})


def select_audits(attempts):
    successful = sorted((a for a in attempts if a["success"]), key=lambda a: (a["objective"], a["start_id"]))
    if len(successful) < 3:
        raise ValueError("fewer than three successful starts; audit cannot proceed")
    return successful[0], successful[:3]


def reduce(plan_path):
    plan = p.load(plan_path)
    gate_required(plan)
    # Exact coverage required. Failed optimizer attempts remain in the reduction.
    chunks = [read_artifact(plan, f"chunks/{t:03d}.json") for t in range(72)]
    attempts = [a for t, c in enumerate(chunks) for a in validate_chunk(plan, t, c)]
    folds, audit_tasks = {}, []
    for fold in p.FOLDS:
        rows = [a for a in attempts if a["fold"] == fold]
        selected, starts = select_audits(rows)
        folds[fold] = {"attempts": rows, "selected": selected}
        for kind, start in [*(('basin', a) for a in starts), ('direct', selected)]:
            audit_tasks.append({"task_id": len(audit_tasks), "fold": fold, "kind": kind, "start": start})
    return p.write_sealed(artifact(plan, "reduction.json"), {"plan_sha256": plan["sha256"],
        "chunks": [c["sha256"] for c in chunks], "folds": folds, "audit_tasks": audit_tasks,
        "heldout_scores_computed": False})


def audit(plan_path, task_id):
    plan = p.load(plan_path)
    reduction = read_artifact(plan, "reduction.json")
    if not 0 <= task_id < 12:
        raise ValueError("audit task outside frozen plan")
    task = reduction["audit_tasks"][task_id]
    path = artifact(plan, f"audits/{task_id:02d}.json")
    if path.exists():
        return read_artifact(plan, f"audits/{task_id:02d}.json")
    context = runtime_context()
    fold = task["fold"]
    profile = model.Profile(p.fold_problem(plan, fold), plan["folds"][fold]["manifest"]["fit"]["delta"])
    box = p.bounds(plan, fold)
    began = time.monotonic()
    try:
        result = (model.Direct(profile, task["start"], box).run() if task["kind"] == "direct"
            else profile.run(task["start"]["coordinates"], box))
        equivalent = model.equivalence(reduction["folds"][fold]["selected"], result, profile.problem)
        health = model.numerical_health(profile, result, box)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        result = {"success": False, "message": f"{type(error).__name__}: {error}", "runtime_seconds": time.monotonic()-began}
        equivalent, health = {"equivalent": False}, {"healthy": False}
    return p.write_sealed(path, {"plan_sha256": plan["sha256"], "reduction_sha256": reduction["sha256"],
        "task": task, "runtime": context, "result": result, "equivalence": equivalent, "health": health,
        "status": "healthy" if equivalent["equivalent"] and health["healthy"] else "failed"})
