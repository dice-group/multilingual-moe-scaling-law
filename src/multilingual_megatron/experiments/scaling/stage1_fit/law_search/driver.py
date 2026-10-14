"""Login-host driver for the already registered, immutable CPU task graph.

No new tasks, retries, forms or thresholds are created. A kernel file lock
prevents two drivers. Worker submissions retain the frozen accounting guard.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from .. import execution as io
from .state import load_plan
from ..artifacts import binding
from ..artifacts import verify

STAGES=("calibration","inner","baselines","final","secondary","stability")


def stage_state(plan,root,stage):
    tasks=[t for t in plan["tasks"] if t["stage"]==stage]
    finished,failed=[],[]
    for task in tasks:
        path=root/"results"/f"{task['id']}.json"
        if not path.exists():
            continue
        result=io.read_json(path)
        if result["plan_sha256"] != plan["sha256"] or result["task_id"] != task["id"]:
            raise ValueError("driver encountered a foreign task receipt")
        finished.append(task["id"])
        if not result.get("certified"):
            failed.append(task["id"])
    return dict(stage=stage,expected=len(tasks),finished=len(finished),failed=failed,
                all_finished=len(finished)==len(tasks))


def final_accounting(plan,root,frozen):
    """Check the last wave too, including separately registered roundoff audits."""
    from .protocol import check_budget
    from .recovery import load as load_recovery
    from .recovery import seal_accounting
    paths=sorted((root/"submissions").glob("*.json"))
    receipts=[io.read_json(p) for p in paths]
    if any(not r.get("job_id") for r in receipts):
        raise ValueError("unresolved submission intent prevents final accounting")
    records=frozen.scheduler_records([r["job_id"] for r in receipts])
    charges=frozen.accounted_reservations(plan,receipts,records)
    recovery_path=root/"native_box_recovery/submission.json"
    if recovery_path.exists():
        recovery=load_recovery(recovery_path.parent/"plan.json")
        receipt=io.read_json(recovery_path)
        if receipt["plan_sha256"] != recovery["sha256"]:
            raise ValueError("foreign recovery submission")
        extra=frozen.scheduler_records([receipt["job_id"]])
        charges+=frozen.accounted_reservations(recovery,[receipt],extra)
        records+=extra
        paths.append(recovery_path)
    if any(r["state"] in ("RUNNING","PENDING","CONFIGURING","COMPLETING") for r in records):
        raise ValueError("final accounting requires finished allocations")
    phases=check_budget([],charges)
    result=dict(plan_sha256=plan["sha256"],phase_charged_core_hours=phases,
        charged_core_hours=sum(phases.values()),
        measured_allocated_core_hours=sum(r["cores"]*r["seconds"]/3600 for r in records),
        records=records,submissions=[binding(p) for p in paths],gpu_training=False)
    seal_accounting(root/"resource_accounting.json",result)
    return result


def run(plan_path,poll_seconds=30):
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("driver must not submit nested Slurm jobs")
    if not 5<=poll_seconds<=60:
        raise ValueError("poll interval must be between 5 and 60 seconds")
    plan=load_plan(plan_path,all_inputs=True)
    root=Path(plan_path).resolve().parent
    replay=io.read_json(root/"baseline_replay.json")
    if replay["plan_sha256"] != plan["sha256"] or not replay["complete"] or replay["max_absolute_prediction_difference"]>1e-10:
        raise ValueError("complete historical replay is required before the driver")
    for entry in [*replay["inputs"],*replay["source"]]:
        verify(entry)
    sys.path.insert(0,str(root/"source"))
    frozen=importlib.import_module("stage1_fit.law_search.campaign")
    selector=importlib.import_module("stage1_fit.law_search.reporting")
    # Only the source-bound reporting-only arithmetic amendment uses live code.
    from .reporting import report
    events=root/"driver_events";events.mkdir(exist_ok=True)
    def event(payload):
        payload=dict(payload,plan_sha256=plan["sha256"],pid=os.getpid(),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        io.write_json_once(events/f"{len(list(events.glob('*.json'))):05d}.json",payload)
        print(json.dumps(payload),flush=True)
    with (root/"driver.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        event(dict(action="start",source=binding(Path(__file__)),gpu_training=False))
        last=None
        try:
            for stage in STAGES:
                if stage=="final":
                    from .recovery import drive as recover
                    event(dict(action="roundoff_recovery_started",maximum_extra_core_hours=10.5))
                    recover(plan_path,poll_seconds)
                    event(dict(action="roundoff_recovery_finished"))
                if stage=="baselines" and not (root/"selection.json").exists():
                    selector.select(plan_path)
                    event(dict(action="selected",selection=binding(root/"selection.json")))
                while True:
                    state=stage_state(plan,root,stage)
                    if state!=last:
                        event(dict(action="progress",**state));last=state
                    receipts=[io.read_json(p) for p in sorted((root/"submissions").glob("*.json"))]
                    if any(not r.get("job_id") for r in receipts):
                        raise ValueError("unresolved submission intent; manual scheduler reconciliation required")
                    jobs=[r["job_id"] for r in receipts]
                    queued=(subprocess.check_output(["squeue","-h","-j",",".join(jobs),"-o","%i"],text=True).strip()
                            if jobs else "")
                    if queued:
                        time.sleep(poll_seconds)
                        continue
                    if state["all_finished"]:
                        if stage in ("calibration","inner") and state["failed"]:
                            raise ValueError(f"{stage} has uncertified tasks; selection is forbidden: {state['failed']}")
                        break
                    submitted={t for r in receipts for t in r["task_ids"]}
                    remaining=[t for t in plan["tasks"] if t["stage"]==stage and not (root/"results"/f"{t['id']}.json").exists()]
                    if any(t["id"] in submitted for t in remaining):
                        raise ValueError("a submitted task ended without its result receipt; no automatic retry")
                    if stage=="stability" and not io.read_json(root/"results/selected-all_grid.json").get("certified"):
                        event(dict(action="stability_blocked",reason="all-grid candidate is uncertified"))
                        break
                    # submit() verifies actual accounting, charges previous caps,
                    # and permits only one homogeneous <=32-core array at a time.
                    if (root/"native_box_recovery/accounting.json").exists():
                        from .protocol import check_budget
                        accounting=io.read_json(root/"native_box_recovery/accounting.json")
                        check_budget(plan["tasks"],[dict(phase="baselines",allocated_core_hours=
                            max(accounting["allocated_core_hours"],accounting["reserved_core_hours"]))])
                    frozen.submit(plan_path,stage)
                    time.sleep(poll_seconds)
            accounting=final_accounting(plan,root,frozen)
            event(dict(action="resources_audited",charged_core_hours=accounting["charged_core_hours"],
                measured_allocated_core_hours=accounting["measured_allocated_core_hours"]))
            if not (root/"report/report.json").exists():
                report(plan_path)
            event(dict(action="finished",report=binding(root/"report/report.json")))
        except Exception as error:
            event(dict(action="stopped",error=f"{type(error).__name__}: {error}"))
            raise


def detach(plan_path):
    """Freeze the reporting/controller code and give it a dedicated tmux session."""
    plan=load_plan(plan_path,all_inputs=True)
    root=Path(plan_path).resolve().parent
    if os.environ.get("SLURM_JOB_ID") or not shutil.which("tmux") or not shutil.which("conda"):
        raise ValueError("detached controller needs tmux and conda on the login host")
    with (root/"driver.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session="stage1-law-"+plan["sha256"][:12]
    existing=subprocess.run(["tmux","has-session","-t",session],capture_output=True)
    if existing.returncode==0:
        raise ValueError("the dedicated controller session already exists")
    launches=root/"driver_launches";launches.mkdir(exist_ok=True)
    index=len(list(launches.glob("*.json")))
    snapshot=root/f"analysis_source_{index:03d}"/"law_search_analysis"
    shutil.copytree(Path(__file__).parents[1],snapshot,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    prefix=str(Path(plan["profile"]["python_executable"]).parent.parent)
    command=shlex.join([shutil.which("conda"),"run","-p",prefix,"--no-capture-output","env",
        "PYTHONDONTWRITEBYTECODE=1","PYTHONPATH="+str(snapshot.parent),
        "OPENBLAS_NUM_THREADS=1","OMP_NUM_THREADS=1",
        "python","-m","law_search_analysis.law_search.driver","--plan",str(Path(plan_path).resolve())])
    # The new session runs only this graph; no existing tmux pane is used.
    subprocess.run(["tmux","new-session","-d","-s",session,"-c",plan["profile"]["workspace_root"],command],check=True)
    receipt=dict(session=session,command=command,plan_sha256=plan["sha256"],
        controller_source=binding(Path(__file__)),analysis_source=[binding(p) for p in sorted(snapshot.rglob("*.py"))])
    io.write_json_once(launches/f"{index:03d}.json",receipt)
    print(json.dumps(dict(session=session,receipt=str(launches/f"{index:03d}.json"))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--poll-seconds",type=int,default=30)
    parser.add_argument("--detach",action="store_true")
    args=parser.parse_args()
    detach(args.plan) if args.detach else run(args.plan,args.poll_seconds)
