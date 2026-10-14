"""Training-only repair of last-bit native box violations, with a new joint audit.

Original receipts and source stay immutable. No equation, bound, threshold,
start search, selector, or heldout response is changed. Other failures remain
unresolved. At most 42 single-core 15-minute audits fit within the baseline
phase's unused 20 core-hours, and the original 512 core-hour ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import numpy as np

from .. import execution as io
from . import data
from ..atlas_comparison_workflow import model as m
from ..atlas_comparison_workflow import numerics as n
from ..data import canonical_json_sha256
from .state import load_plan
from .protocol import check_budget
from ..artifacts import binding
from ..artifacts import verify

RULES=dict(schema="stage1_native_box_ulp_audit/v1",maximum_ulps=8,
           minutes_per_model=15,maximum_models=42,max_concurrent_cores=32,
           new_projected_starts=0,gradient_tolerance=1e-8,
           objective_tolerance=1e-6,prediction_tolerance=1e-4,
           heldout_responses=False,phase="baselines")


def seal_accounting(path, receipt):
    """A resumed controller may verify an existing receipt, never overwrite it."""
    if path.exists():
        if io.read_json(path) != receipt:
            raise ValueError("scheduler accounting changed after its receipt was sealed")
    else:
        io.write_json_once(path, receipt)


def canonical_box(spec,coordinates):
    """Return a correction only for an actual violation of at most eight ULPs."""
    x=np.asarray(coordinates,float)
    lo,hi=m.bounds(spec)
    if x.shape!=lo.shape or not np.isfinite(x).all():
        return None
    outside=(x<lo)|(x>hi)
    if not outside.any():
        return None
    y=np.clip(x,lo,hi)
    tolerance=8*np.abs(np.spacing(y))
    if np.any(abs(y[outside]-x[outside])>tolerance[outside]):
        return None
    return y


def repair_group(group,packet):
    if group.get("certified"):
        return group
    if "spec" not in group or "attempts" not in group:
        return group
    spec=m.Spec(**group["spec"])
    revised=[]
    for attempt in group.get("attempts",[]):
        if not attempt.get("optimizer_success") or "coordinates" not in attempt:
            continue
        x=canonical_box(spec,attempt["coordinates"])
        if x is None:
            continue
        try:
            record=n.record(spec,x,packet,packet,kind="box_ulp_canonicalization",index=attempt["index"],
                flag=attempt["optimizer_success"],message="roundoff-only box projection; original physical KKT recomputed",runtime=0.)
        except (ValueError,RuntimeError,FloatingPointError):
            continue
        record.update(original_coordinates=attempt["coordinates"],
                      max_coordinate_change=float(np.max(abs(x-attempt["coordinates"]))))
        revised.append(record)
    eligible=[r for r in revised if r.get("success")]
    if not eligible:
        return group | {"box_recovery":dict(status="no_certifiable_roundoff_repair",attempts=revised)}
    best=n.choose([*group.get("attempts",[]),*revised])
    audit=n.safe(n.direct,spec=spec,candidate=best,packet=packet,support=packet)
    certified=n.agreement(spec,best,audit,packet)
    return group | dict(selected=best,audit=audit,certified=certified,
        box_recovery=dict(status="certified" if certified else "independent_audit_failed",attempts=revised,
                          original_selected=group.get("selected"),original_audit=group.get("audit"),
                          maximum_ulps=8,new_projected_starts=0,heldout_scores_computed=False))


def prepare(parent_path):
    parent=load_plan(parent_path)
    root=Path(parent_path).resolve().parent
    output=root/"native_box_recovery"
    if output.exists():
        return io.read_json(output/"plan.json")
    tasks=[]
    for task in (t for t in parent["tasks"] if t["kind"]=="native"):
        path=root/"results"/f"{task['id']}.json"
        result=io.read_json(path)
        if result["plan_sha256"]!=parent["sha256"] or result["task_id"]!=task["id"]:
            raise ValueError("native receipt identity mismatch")
        if not result.get("certified"):
            tasks.append(dict(id=task["id"],index=len(tasks),phase="baselines",cores=1,minutes=15,original=binding(path)))
    if len(tasks)>42:
        raise ValueError("recovery panel exceeds original certified native panel")
    allocations=check_budget([*parent["tasks"],*tasks])
    output.mkdir()
    source=output/"source"/"stage1_recovery"
    shutil.copytree(Path(__file__).parents[1],source,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    plan=dict(rules=RULES,parent_plan=binding(Path(parent_path).resolve()),parent_plan_sha256=parent["sha256"],
        tasks=tasks,allocations_including_original_caps=allocations,profile=parent["profile"],
        training=parent["training"]["all_grid"],calibration=binding(root/"results/cal-all_grid.json"),
        source=[binding(p) for p in sorted(source.rglob("*.py"))])
    plan["sha256"]=canonical_json_sha256(plan)
    io.write_json_once(output/"plan.json",plan)
    return plan


def load(path):
    plan=io.read_json(Path(path));unsigned=dict(plan);digest=unsigned.pop("sha256")
    if digest!=canonical_json_sha256(unsigned) or plan["rules"]!=RULES:
        raise ValueError("recovery protocol changed")
    for entry in [plan["parent_plan"],plan["training"],plan["calibration"],*plan["source"],
                  *(t["original"] for t in plan["tasks"])]:
        verify(entry)
    parent=load_plan(plan["parent_plan"]["path"])
    if parent["sha256"]!=plan["parent_plan_sha256"]:
        raise ValueError("wrong parent plan")
    check_budget([*parent["tasks"],*plan["tasks"]])
    return plan


def worker(path,index):
    plan=load(path);root=Path(path).resolve().parent
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK")!="1":
        raise ValueError("recovery requires its single-core Slurm task")
    if any(os.environ.get(k) not in (None,"","0") for k in ("SLURM_GPUS","SLURM_GPUS_ON_NODE","SLURM_JOB_GPUS")):
        raise ValueError("GPU allocation is not authorized")
    task=plan["tasks"][index]
    destination=root/"results"/f"{task['id']}.json"
    if destination.exists():raise ValueError("immutable recovery already exists")
    _,_,packets=data.read_training(plan["training"])
    delta=io.read_json(verify(plan["calibration"]))["delta"]
    packets={p["language"]:p["train"]|{"delta":delta} for p in packets}
    original=io.read_json(verify(task["original"]))
    began=time.monotonic();groups=[];timed_out=False
    def expired(signum,frame):raise TimeoutError("recovery wall-time cap reached")
    signal.signal(signal.SIGALRM,expired);signal.alarm(870)
    try:
        for group in original.get("groups",[]):
            groups.append(repair_group(group,packets[group["language"]]))
    except TimeoutError:
        timed_out=True
        groups.extend(original["groups"][len(groups):])
    finally:signal.alarm(0)
    result=original | dict(groups=groups,certified=len(groups)==original.get("expected_languages") and
                            all(g.get("certified") for g in groups))
    receipt=dict(parent_plan_sha256=plan["parent_plan_sha256"],parent_result=task["original"],task_id=task["id"],
        recovery_plan=binding(Path(path).resolve()),result=result,elapsed_seconds=time.monotonic()-began,
        slurm_job_id=os.environ["SLURM_JOB_ID"],timed_out=timed_out,heldout_scores_computed=False,
        repaired_languages=[g["language"] for g in groups if g.get("box_recovery",{}).get("status")=="certified"])
    io.write_json_once(destination,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("result","parent_result","recovery_plan")}))


def drive(parent_path,poll_seconds=30):
    if os.environ.get("SLURM_JOB_ID"):raise ValueError("no nested submissions")
    parent=load_plan(parent_path);root=Path(parent_path).resolve().parent
    receipts=[io.read_json(p) for p in (root/"submissions").glob("*.json")]
    jobs=[r["job_id"] for r in receipts]
    while jobs and subprocess.check_output(["squeue","-h","-j",",".join(jobs),"-o","%i"],text=True).strip():
        time.sleep(poll_seconds)
    # Native recovery is in addition to, not a replacement for, all original
    # allocations. Refuse unknown/mismatched observed allocation counts.
    from .campaign import scheduler_records
    from .campaign import accounted_reservations
    accounted=accounted_reservations(parent,receipts,scheduler_records(jobs))
    plan=prepare(parent_path);path=root/"native_box_recovery/plan.json";plan=load(path)
    check_budget(plan["tasks"],accounted)
    if not plan["tasks"]:return
    output=path.parent;submission=output/"submission.json"
    if not submission.exists():
        intent=output/"submission_intent.json"
        if intent.exists():raise ValueError("reconcile unresolved recovery submission intent")
        p=plan["profile"];(output/"logs").mkdir(exist_ok=True)
        command=shlex.join([p["python_executable"],"-m","stage1_recovery.law_search.recovery","worker","--plan",str(path)])
        script="\n".join(["#!/bin/bash",f"#SBATCH --account={p['account']}",f"#SBATCH --partition={p['partition']}",
            "#SBATCH --job-name=lawsearch_box_audit","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=1",f"#SBATCH --mem={p['memory']}",
            "#SBATCH --time=15",f"#SBATCH --array=0-{len(plan['tasks'])-1}%32",
            f"#SBATCH --output={shlex.quote(str(output/'logs/audit-%A_%a.out'))}","set -euo pipefail",
            f"export PYTHONPATH={shlex.quote(str(output/'source'))}",
            "export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
            command+' --task "$SLURM_ARRAY_TASK_ID"',""])
        subprocess.run(["bash","-n"],input=script,text=True,check=True)
        io.write_text_once(output/"audit.sbatch",script)
        io.write_json_once(intent,dict(plan_sha256=plan["sha256"],script=binding(output/"audit.sbatch")))
        job=subprocess.check_output(["sbatch","--parsable",str(output/"audit.sbatch")],text=True).strip().split(";")[0]
        if not job.isdigit():raise ValueError("ambiguous recovery submission")
        io.write_json_once(submission,dict(job_id=job,plan_sha256=plan["sha256"],
            allocated_core_hours=len(plan["tasks"])*.25,task_ids=[t["id"] for t in plan["tasks"]]))
    receipt=io.read_json(submission)
    while subprocess.check_output(["squeue","-h","-j",receipt["job_id"],"-o","%i"],text=True).strip():
        time.sleep(poll_seconds)
    records=scheduler_records([receipt["job_id"]])
    recovery_parent={"tasks":plan["tasks"]}
    charges=accounted_reservations(recovery_parent,[receipt],records)
    check_budget(parent["tasks"],charges)
    for task in plan["tasks"]:
        if not (output/"results"/f"{task['id']}.json").exists():
            raise ValueError("recovery task ended without receipt; no automatic retry")
    seal_accounting(output/"accounting.json",dict(plan_sha256=plan["sha256"],records=records,
        reserved_core_hours=len(plan["tasks"])*.25,allocated_core_hours=sum(r["cores"]*r["seconds"]/3600 for r in records)))


def overlay(root,plan,task_id,original):
    path=root/"native_box_recovery/results"/f"{task_id}.json"
    if not path.exists():return original
    receipt=io.read_json(path)
    if receipt["parent_plan_sha256"]!=plan["sha256"] or receipt["task_id"]!=task_id:
        raise ValueError("foreign recovery receipt")
    if receipt["parent_result"]!=binding(root/"results"/f"{task_id}.json"):
        raise ValueError("original native result changed")
    verify(receipt["recovery_plan"])
    load(receipt["recovery_plan"]["path"])
    return receipt["result"] | {"box_recovery_receipt":binding(path)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("worker","drive"))
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--task",type=int)
    args=parser.parse_args()
    worker(args.plan,args.task) if args.action=="worker" else drive(args.plan)
