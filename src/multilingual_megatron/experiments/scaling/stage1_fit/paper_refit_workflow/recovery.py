"""Bounded, create-only recovery of failed full-paper certification checks.

Original fits, their source snapshot and report remain immutable. This amendment
fixes physical bound roundoff and refines an independent audit, including a
cross-optimizer box audit; domains, fit winners and acceptance thresholds stay explicit.
"""
import argparse
import os
from pathlib import Path
import shlex
import signal
import time

import numpy as np

from .. import execution as io
from ..artifacts import copy_dependency_snapshot, environment
from ..law_search import model as shared
from . import backend, full_report, inputs, numerics, protocol, runner


def prepare(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    plan = runner.load(parent)
    if not output.is_relative_to(parent.parent) or output.exists():
        raise ValueError("use a new recovery directory inside the parent run")
    actions = []
    for task in plan["tasks"]:
        if task["stage"] != "certify":
            continue
        old = runner.receipt(parent, plan, task["id"])
        if old.get("certified"):
            continue
        if task["model"] in protocol.NATIVE:
            for index in task["dependencies"]:
                group = runner.receipt(parent, plan, index)
                if not group.get("certified"):
                    actions.append(dict(id=index, mode="native_bound_projection"))
            actions.append(dict(id=task["id"], mode="native_aggregate"))
        elif task["arm"].startswith("slsqp") and old.get("selected", {}).get("success"):
            actions.append(dict(id=task["id"], mode="tighter_joint_audit"))
        else:
            raise ValueError(f"no registered recovery for {task['cell']}")
    if not actions:
        raise ValueError("no failed checks need recovery")
    output.mkdir(parents=True)
    snapshot = output / "source/paper_fit_snapshot"
    copy_dependency_snapshot(snapshot, ("paper_refit_workflow.recovery", "paper_refit"))
    amendment = inputs.sealed(dict(schema="stage1_paper_refit_recovery/v1",
        parent=inputs.binding(parent), parent_sha256=plan["sha256"], actions=actions,
        originals=[inputs.binding(parent.parent / "results" / f"{a['id']:05d}.json") for a in actions],
        source=[inputs.local_binding(p, output) for p in sorted(snapshot.rglob("*.py"))],
        environment=environment(), audit_ftols=[1e-14, 1e-16],
        independent_box_audit="L-BFGS-B after bounded SLSQP refinement; original SLSQP winner unchanged",
        seconds=550,
        rationale="project unconditioned native amplitudes onto their unchanged physical bounds; "
                  "refine failed independent SLSQP audit, then cross-check the same box with L-BFGS-B; "
                  "retain winners and acceptance thresholds"))
    io.write_json_once(output / "plan.json", amendment)
    p = plan["profile"]
    script = ["#!/bin/bash", "#SBATCH --job-name=paper-refit-recovery",
        f"#SBATCH --account={p['account']}", f"#SBATCH --partition={p['partition']}",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={p['memory']}", "#SBATCH --time=10",
        f"#SBATCH --output={output}/recovery-%j.out", "set -euo pipefail",
        f"cd {shlex.quote(str(output))}", f"export PYTHONPATH={shlex.quote(str(snapshot.parent))}",
        "export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"exec {shlex.quote(p['python_executable'])} -m paper_fit_snapshot.paper_refit_workflow.recovery run --plan {shlex.quote(str(output / 'plan.json'))}", ""]
    io.write_text_once(output / "recovery.sbatch", "\n".join(script))
    return dict(plan=str(output / "plan.json"), actions=actions, submitted=False)


def refine_audit(task, old, input_path, amendment, deadline, sink):
    rows, sources, packets, cal = inputs.read_training(input_path, task["fold"])
    model = numerics.make_model(task["model"], rows, sources, packets, cal, task["arm"])
    profile = shared.Profile(model, cal["delta"], deadline)
    selected = old["selected"]
    prediction = numerics.predict(model, selected)
    gap = float(np.max(abs(prediction-numerics.independent(model, selected, rows, packets))))
    history, passed = [], False
    strategies = [(task["arm"], ftol) for ftol in amendment["audit_ftols"]]
    if task["arm"] == "slsqp_box" and amendment.get("independent_box_audit"):
        strategies.append(("lbfgsb_box", 1e-15))
    for audit_arm, ftol in strategies:
        model.solver_policy = audit_arm
        model.audit_ftol = ftol
        audit = shared.Direct(profile, selected).run()
        change = float(np.max(abs(prediction-numerics.predict(model, audit))))
        passed = bool(audit["success"] and gap <= 1e-10 and change <= 1e-4 and
            abs(audit["objective"]-selected["objective"]) <= 1e-6*max(abs(selected["objective"]), 1e-15))
        record = dict(optimizer=protocol.contract(audit_arm)["optimizer"], ftol=ftol,
                      audit=audit, prediction_change=change, passed=passed)
        history.append(record)
        io.write_json_once(sink / f"audit-{len(history)}.json", record)
        if passed:
            break
    return {k:v for k,v in old.items() if k != "sha256"} | dict(certified=passed, audit=audit,
        audit_recovery=history, independent_training_replay_max_gap=gap,
        error=None if passed else "tighter independent audit still unresolved")


def verified_overrides(parent, completion_path):
    """Verify both original and replacement receipts before reporting/adoption."""
    if completion_path is None:
        return {}
    parent, completion_path = Path(parent).resolve(), Path(completion_path).resolve()
    if not completion_path.is_relative_to(parent.parent):
        raise ValueError("recovery must be stored inside the parent run")
    completion = inputs.read_sealed(completion_path)
    amendment_path = inputs.local_path(parent.parent, completion["amendment"])
    amendment = inputs.read_sealed(amendment_path)
    if Path(inputs.verify(amendment["parent"])) != parent:
        raise ValueError("foreign recovery parent")
    plan = inputs.read_sealed(parent)
    if amendment["parent_sha256"] != plan["sha256"] or completion["parent_sha256"] != plan["sha256"]:
        raise ValueError("foreign recovery plan")
    for entry in amendment["source"]:
        inputs.local_path(amendment_path.parent, entry)
    for entry in amendment["originals"]:
        inputs.verify(entry)
    overrides = {}
    for entry in completion["overrides"]:
        path = inputs.local_path(parent.parent, entry)
        result = inputs.read_sealed(path)
        index = result["task"]["id"]
        old = runner.receipt(parent, plan, index)
        if (index in overrides or old.get("certified") or result["task"] != old["task"]
                or result["plan_sha256"] != plan["sha256"]
                or result["recovery_plan_sha256"] != amendment["sha256"]
                or result["original_receipt_sha256"] != old["sha256"]):
            raise ValueError("invalid recovery receipt lineage")
        overrides[index] = (path, result)
    if set(overrides) != {a["id"] for a in amendment["actions"]}:
        raise ValueError("incomplete recovery inventory")
    return overrides


def run(path):
    path = Path(path).resolve(); root = path.parent
    amendment = inputs.read_sealed(path)
    parent = Path(inputs.verify(amendment["parent"])); plan = runner.load(parent)
    for entry in amendment["source"]:
        inputs.local_path(root, entry)
    for entry in amendment["originals"]:
        inputs.verify(entry)
    if Path(__file__).resolve() != root / "source/paper_fit_snapshot/paper_refit_workflow/recovery.py":
        raise ValueError("execute the frozen recovery source")
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or any(os.environ.get(k) != "1" for k in runner.THREADS)
            or environment() != amendment["environment"]):
        raise ValueError("one-core Slurm allocation and prepared numerical environment required")
    if any(os.environ.get(k) not in (None, "", "0") for k in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")):
        raise ValueError("CPU-only recovery required")
    (root / "results").mkdir(); (root / "attempts").mkdir()
    def expired(signum, frame):
        raise TimeoutError("bounded recovery deadline")
    signal.signal(signal.SIGALRM, expired); signal.alarm(amendment["seconds"])
    deadline = time.monotonic()+amendment["seconds"]
    input_path = inputs.local_path(parent.parent, plan["input"])
    recovered, bindings = {}, []
    for action in amendment["actions"]:
        index = action["id"]; task = plan["tasks"][index]; old = runner.receipt(parent, plan, index)
        sink = root / "attempts" / f"{index:05d}"; sink.mkdir()
        began = time.monotonic()
        if action["mode"] == "tighter_joint_audit":
            result = refine_audit(task, old, input_path, amendment, deadline, sink)
        else:
            shards = [(plan["tasks"][i], recovered[i] if i in recovered else runner.receipt(parent, plan, i))
                      for i in task.get("dependencies", [])]
            result = backend.execute(task, input_path, plan["contracts"].get(task["cell"]), deadline, sink, shards)
        result = inputs.sealed({k:v for k,v in result.items() if k != "sha256"} | dict(
            plan_sha256=plan["sha256"], task=task, heldout_scores_computed=False,
            job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-began,
            recovery_plan_sha256=amendment["sha256"], original_receipt_sha256=old["sha256"]))
        target = root / "results" / f"{index:05d}.json"; io.write_json_once(target, result)
        recovered[index] = result; bindings.append(inputs.local_binding(target, parent.parent))
        print(dict(task=index, certified=result.get("certified"), seconds=result["elapsed_seconds"]), flush=True)
    completion = inputs.sealed(dict(parent_sha256=plan["sha256"], amendment=inputs.local_binding(path, parent.parent),
        overrides=bindings, all_recovered=all(r.get("certified") for r in recovered.values())))
    io.write_json_once(root / "completion.json", completion)
    verified_overrides(parent, root / "completion.json")
    summary = full_report.report(parent, recovery_path=root / "completion.json", output=root / "report")
    signal.alarm(0)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare" and args.output is None:
        parser.error("prepare requires --output")
    result = prepare(args.plan, args.output) if args.action == "prepare" else run(args.plan)
    print(result)
