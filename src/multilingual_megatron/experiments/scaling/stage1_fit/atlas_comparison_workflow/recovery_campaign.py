"""Bounded, sealed training-only numerical recovery of the complete v8 panel."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import argparse
import copy
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

import numpy as np

from .. import atlas_full_pilot as io
from ..execution import write_text_once
from . import array_execution as arrays, campaign as c, model as m, numerics as n
from . import recovery_numerics as r, reporting, reuse, snapshot

MODULE="multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.recovery_campaign"
ALLOCATIONS={"pilot":dict(cores=8,minutes=10),
             "main":dict(cores=8,minutes=8,jobs=128,parallel_jobs=128),
             "finalize":dict(cores=1,minutes=120)}


def protocol(parent):
    result=copy.deepcopy(parent)
    result.update(version="atlas_comparison/v9_recovery",recovery=r.SETTINGS,allocations=ALLOCATIONS,
        pilot_allocated_core_hours=arrays.allocated_core_hours(ALLOCATIONS["pilot"]),
        main_allocated_core_hours=arrays.allocated_core_hours(ALLOCATIONS["main"]),
        finalize_allocated_core_hours=arrays.allocated_core_hours(ALLOCATIONS["finalize"]),
        amendment="training-only recovery of every unresolved v8 target; preserve certified receipts, equations, packets, physical KKT and agreement thresholds; compatible bounded warm starts and independent joint-coordinate verification",
        recovery_pilot="first unresolved task per equation/fold/failure class, plus ten diagnostic cases; no heldout outcomes",
        recovery_selection="minimum certified objective no worse than any encountered feasible training objective within 1e-6 relative; distinct solver families including joint-coordinate solver")
    return result


def check_budget(prior):
    total=prior+1.+sum(arrays.allocated_core_hours(a) for a in ALLOCATIONS.values())
    if not np.isfinite(prior) or prior<0 or total>250. or max(map(arrays.concurrent_cores,ALLOCATIONS.values()))>1024:
        raise ValueError("recovery exceeds authorized cumulative CPU budget")
    return total


def failure_class(value):
    if value.get("selected") is None: return "no_selected"
    if not value.get("audit",{}).get("success"): return "audit_not_stationary"
    return "independent_disagreement"


def prepare(profile_path):
    profile=json.loads(Path(profile_path).read_text()); root=Path(profile["output_root"]).resolve()
    if root.exists() or root.parent!=Path.cwd()/"result_analysis": raise ValueError("new result root required")
    parent=io.read(profile["parent_plan"]); oldroot=Path(parent["output_root"])
    checked=io.read(oldroot/"audit.json"); completion=io.read(oldroot/"execution/finalize_completion.json")
    if (parent["protocol"]["version"]!="atlas_comparison/v8_parallel" or parent["environment"]!=c.environment()
            or checked["plan_sha256"]!=parent["sha256"] or completion["audit_sha256"]!=checked["sha256"]
            or not checked["complete"] or checked["missing_tasks"]): raise ValueError("requires complete audited v8 parent")
    bindings=[io.bind(profile_path),io.bind(profile["parent_plan"]),io.bind(oldroot/"audit.json"),
              io.bind(oldroot/"execution/finalize_completion.json"),parent["archive"],*parent["inputs"]]
    prior=parent["prior_allocated_core_hours"]
    accounting=[str(oldroot/"execution/slurm_accounting.json"),*profile["recovery_accounting"]]
    if len(set(accounting))!=len(accounting): raise ValueError("duplicate CPU accounting")
    for path in accounting:
        value=json.loads(Path(path).read_text())
        if value["state"]!="COMPLETED" or value["gpu_allocations"]!=0: raise ValueError("incomplete CPU accounting")
        prior+=value["allocated_cpu_seconds"]/3600; bindings.append(io.bind(path))
    maximum=check_budget(prior)
    for binding in [*bindings,*parent["sources"],*parent["packets"],parent["observations"],parent["historical_predictions"]]: io.verify(binding)
    source_map={Path(b["original_path"]):Path(b["path"]) for b in parent["sources"]}
    for path in (Path(__file__),Path(n.__file__),Path(r.__file__),Path(reporting.__file__),Path.cwd()/"tests/test_stage1_atlas_comparison.py"):
        source_map[path]=path
    runtime,sources,archive=snapshot.freeze(root,source_map,Path.cwd(),content_sources=source_map)
    (root/"logs").mkdir()
    newprotocol=protocol(parent["protocol"])
    coverage={row["task"]:row for row in checked["coverage"]}
    if len(coverage)!=parent["task_count"] or set(coverage)!=set(range(parent["task_count"])):
        raise ValueError("complete, unique parent population required")
    inherited=[]; descriptors={}; pilots={}; reasons={}
    for packet_index,packet_binding in enumerate(parent["packets"]):
        packet=io.read(packet_binding["path"]); values=[]; seeds=[]
        for index,spec in enumerate(c.PANEL):
            task=packet_index*len(c.PANEL)+index; bound=coverage[task]["receipt"]; io.verify(bound)
            value=io.read(bound["path"])
            restored=reuse.budget_receipt(value,parent,task,packet,spec.payload(),bound,coverage[task]["certified"])
            values.append((task,value,restored,bound)); bindings.append(bound)
            for point in r.seed_points(value): seeds.append(dict(spec=spec.payload(),point=point,receipt=bound))
        seed_path=root/f"seeds/{packet_index:04d}.json"
        io.write(seed_path,dict(packet_sha256=packet["sha256"],points=seeds,heldout_scores_computed=False))
        for task,value,restored,bound in values:
            if value["certified"]: inherited.append(restored); continue
            descriptors[str(task)]=dict(previous=bound,seeds=io.bind(seed_path))
            reason=failure_class(value); reasons[reason]=reasons.get(reason,0)+1
            pilots.setdefault((value["spec"]["name"],packet["fold"],reason),task)
    pilot_tasks=sorted(set(pilots.values()) | (set(profile["diagnostic_tasks"]) & set(map(int,descriptors))))
    shards=arrays.partition(parent["task_count"],newprotocol["panel"],inherited,ALLOCATIONS["main"]["jobs"])
    plan=io.write(root/"plan.json",dict(protocol=newprotocol,profile=profile,output_root=str(root),
        parent_plan=io.bind(profile["parent_plan"]),inputs=bindings,sources=sources,archive=archive,runtime_root=str(runtime),
        packets=parent["packets"],observations=parent["observations"],historical_predictions=parent["historical_predictions"],
        historical_metrics=parent["historical_metrics"],environment=c.environment(),task_count=parent["task_count"],
        recovery_inputs=descriptors,pilot_tasks=pilot_tasks,shards=shards,prior_allocated_core_hours=prior,
        maximum_cumulative_allocated_core_hours=maximum,failure_counts=reasons,new_heldout_scores_computed=False))
    for value in inherited: io.write(c.task_path(plan,value["task"]),value|dict(plan_sha256=plan["sha256"]))
    io.write(root/"execution/inheritance.json",dict(plan_sha256=plan["sha256"],certified_tasks=len(inherited),
        recovery_tasks=len(descriptors),status="every certified v8 receipt preserved; all failures retained as recovery inputs and required population"))
    runner=root/"run.sh"
    write_text_once(runner,"\n".join(("#!/bin/bash","set -euo pipefail",f"cd {shlex.quote(str(Path.cwd()))}",
        f"export PYTHONPATH={shlex.quote(str(runtime/'src'))}:.",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"exec {shlex.quote(profile['python_executable'])} -m {MODULE} \"$@\" --plan {shlex.quote(str(root/'plan.json'))}","")))
    scripts=[io.bind(runner)]
    for phase,a in ALLOCATIONS.items():
        jobs=a.get("jobs",1); extra=[f"#SBATCH --array=0-{jobs-1}%{a['parallel_jobs']}"] if jobs>1 else []
        logid="%A_%a" if jobs>1 else "%j"
        script=["#!/bin/bash",f"#SBATCH --job-name=atlas-recovery-{phase}",f"#SBATCH --account={profile['account']}",
            f"#SBATCH --partition={profile['partition']}","#SBATCH --nodes=1","#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={a['cores']}",f"#SBATCH --time={a['minutes']}",f"#SBATCH --mem={profile['memory']}",
            *extra,"#SBATCH --signal=B:USR1@60",f"#SBATCH --output={root}/logs/{phase}-{logid}.out",
            "set -euo pipefail",f"exec bash {shlex.quote(str(runner))} run --phase {phase}",""]
        target=root/f"{phase}.sbatch"; write_text_once(target,"\n".join(script)); scripts.append(io.bind(target))
        subprocess.run(["bash","-n",str(target)],check=True)
    io.write(root/"launch.json",dict(plan_sha256=plan["sha256"],scripts=scripts))
    return dict(plan=str(root/"plan.json"),sha256=plan["sha256"],inherited=len(inherited),recovery_tasks=len(descriptors),
        pilot_tasks=len(pilot_tasks),maximum_cumulative_core_hours=maximum)


def load(path,*,verify_inputs=False):
    plan=io.read(path); parent=io.read(plan["parent_plan"]["path"])
    if plan["protocol"]!=protocol(parent["protocol"]) or plan["environment"]!=c.environment():
        raise ValueError("recovery protocol or environment drift")
    if not Path(__file__).resolve().is_relative_to(Path(plan["runtime_root"])): raise ValueError("use frozen run.sh")
    for bound in [*plan["sources"],plan["archive"],plan["parent_plan"]]: io.verify(bound)
    if verify_inputs:
        for bound in [*plan["inputs"],*plan["packets"],plan["observations"],plan["historical_predictions"]]: io.verify(bound)
        for descriptor in plan["recovery_inputs"].values():
            for bound in descriptor.values(): io.verify(bound)
        launch=io.read(Path(plan["output_root"])/"launch.json")
        if launch["plan_sha256"]!=plan["sha256"]: raise ValueError("launch identity changed")
        for bound in launch["scripts"]: io.verify(bound)
    return plan


def work(root,plan_hash,task,packet_binding,descriptor,spec):
    for bound in (packet_binding,*descriptor.values()): io.verify(bound)
    packet=io.read(packet_binding["path"]); previous=io.read(descriptor["previous"]["path"])
    seeds=io.read(descriptor["seeds"]["path"])
    if seeds["packet_sha256"]!=packet["sha256"] or previous["packet_sha256"]!=packet["sha256"]:
        raise ValueError("recovery seed packet mismatch")
    result=r.recover(m.Spec(**spec),packet["train"],packet["support"],previous,seeds["points"])
    result.update(plan_sha256=plan_hash,task=task,packet_sha256=packet["sha256"],fold=packet["fold"],
        language=packet["language"],recovered_from=descriptor["previous"])
    io.write(Path(root)/f"results/{task:05d}.json",result)
    return dict(task=task,certified=result["certified"],runtime_seconds=result["runtime_seconds"])


def run(path,phase):
    plan=load(path,verify_inputs=phase!="main"); root=Path(plan["output_root"]); a=ALLOCATIONS[phase]
    index=arrays.array_index(a,os.environ); phase_id=phase if index is None else f"{phase}_{index:03d}"
    cores,limit=a["cores"],a["minutes"]*60
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK")!=str(cores): raise ValueError("bound Slurm CPU allocation required")
    if any(os.environ.get(k)!="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")):
        raise ValueError("single BLAS thread per worker required")
    if phase=="main":
        gate=io.read(root/"pilot_gate.json")
        if gate["plan_sha256"]!=plan["sha256"] or not gate["launch_allowed"]: raise ValueError("pilot gate failed")
    io.write(root/f"execution/{phase_id}.json",dict(plan_sha256=plan["sha256"],job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(),cores=cores,seconds_limit=limit,array_index=index,array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID")))
    began=time.monotonic()
    if phase=="finalize":
        checked=reporting.audit(path,loader=load); report=reporting.evaluate(path,loader=load)
        return io.write(root/"execution/finalize_completion.json",dict(plan_sha256=plan["sha256"],audit_sha256=checked["sha256"],
            report_sha256=report["sha256"],status=report["status"],wall_seconds=time.monotonic()-began,heldout_scores_computed=True))
    tasks=plan["pilot_tasks"] if phase=="pilot" else plan["shards"]["tasks"][index]
    remaining=iter(t for t in tasks if not c.task_path(plan,t).exists()); stop=[False]; completed=[]
    signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True)); signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    pool=ProcessPoolExecutor(cores,mp_context=multiprocessing.get_context("spawn")); pending={}
    try:
        while True:
            if time.monotonic()-began>limit-90: stop[0]=True
            while not stop[0] and len(pending)<cores:
                task=next(remaining,None)
                if task is None: break
                packet_index,spec_index=divmod(task,len(c.PANEL))
                pending[pool.submit(work,str(root),plan["sha256"],task,plan["packets"][packet_index],
                    plan["recovery_inputs"][str(task)],c.PANEL[spec_index].payload())]=task
            if not pending: break
            finished,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in finished:
                task=pending.pop(future)
                try: value=future.result()
                except Exception as error:
                    value=dict(task=task,certified=False,worker_error=f"{type(error).__name__}: {error}")
                    io.write(c.task_path(plan,task),value|dict(plan_sha256=plan["sha256"],heldout_scores_computed=False))
                completed.append(value); print(json.dumps(value),flush=True)
            if stop[0]:
                for process in multiprocessing.active_children(): process.terminate()
                break
    finally: pool.shutdown(wait=not stop[0],cancel_futures=True)
    return io.write(root/f"execution/{phase_id}_completion.json",dict(plan_sha256=plan["sha256"],completed_tasks=len(completed),
        stopped_at_budget=stop[0],wall_seconds=time.monotonic()-began,heldout_scores_computed=False))


def gate(path):
    plan=load(path); root=Path(plan["output_root"]); maxima={}; errors=[]; missing=[]; certified=0
    for task in plan["pilot_tasks"]:
        target=c.task_path(plan,task)
        if not target.exists(): missing.append(task); continue
        row=io.read(target)
        if row["plan_sha256"]!=plan["sha256"] or row["task"]!=task: raise ValueError("pilot receipt mismatch")
        if "worker_error" in row: errors.append(task); continue
        spec=m.Spec(**row["spec"]); key=(spec.form,spec.parameter_count,spec.signed)
        maxima[key]=max(maxima.get(key,0.),row["runtime_seconds"]); certified+=bool(row["certified"])
    forecast=0.
    for task in map(int,plan["recovery_inputs"]):
        spec=c.PANEL[task%len(c.PANEL)]
        forecast+=maxima.get((spec.form,spec.parameter_count,spec.signed),float("inf"))
    forecast*=2/3600
    allowed=not missing and not errors and forecast<=plan["protocol"]["main_allocated_core_hours"]
    return io.write(root/"pilot_gate.json",dict(plan_sha256=plan["sha256"],launch_allowed=bool(allowed),
        projected_core_hours=forecast if np.isfinite(forecast) else None,missing_tasks=missing,worker_errors=errors,
        certified_tasks=certified,completed_tasks=len(plan["pilot_tasks"])-len(missing),heldout_scores_computed=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","gate")); parser.add_argument("--profile"); parser.add_argument("--plan")
    parser.add_argument("--phase",choices=tuple(ALLOCATIONS)); args=parser.parse_args()
    value=prepare(args.profile) if args.action=="prepare" else gate(args.plan) if args.action=="gate" else run(args.plan,args.phase)
    print(json.dumps(value))
