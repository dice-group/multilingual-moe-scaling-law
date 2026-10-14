"""One-command Slurm submission with startup checks, collection and monitoring."""
from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import shlex
import subprocess

from .full_suite_campaign import checked, read, ref, save
from .contracts import _atomic_json

MODULE = "multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign"
MONITOR = "multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_monitor"


def prepare_controls(output, profile):
    slurm, resources = profile["slurm"], profile["slurm"]["aggregation"]
    controls = {}
    for name in ("gate", "monitor"):
        command = ([profile["python"], "-m", MODULE, "queue-gate", "--campaign", str(output/"campaign.json")]
                   if name=="gate" else [profile["python"], "-m", MONITOR, "--campaign", str(output/"campaign.json"), "--watch"])
        path=output/(name+".sbatch")
        path.write_text("\n".join([
            "#!/bin/bash", "#SBATCH --job-name=s1-extension-"+name, "#SBATCH --account="+slurm["account"],
            "#SBATCH --partition="+resources["partition"], "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task="+(str(resources["cpus_per_task"]) if name=="gate" else "1"),
            "#SBATCH --mem="+(resources["memory"] if name=="gate" else "8G"),
            "#SBATCH --time="+(resources["time"] if name=="gate" else "3-00:00:00"),
            "#SBATCH --output="+str(output/(name+"-%j.log")), "set -euo pipefail",
            "export PYTHONPATH="+shlex.quote(str(output/"source/src")+":"+str(output/"source")),
            "exec "+shlex.join(command), ""]))
        subprocess.run(["bash","-n",str(path)],check=True)
        controls[name]=ref(path)
    return controls


def finish_gate(path, campaign):
    from .candidate_campaign import smoke_gate, estimate_budget
    gate=smoke_gate(campaign)
    budget=estimate_budget(campaign,gate)
    bounds={j["id"]:j["walltime_minutes"]*60 for j in campaign["jobs"]}
    excessive=[j["id"] for j in budget["jobs"] if j["high_seconds"]>bounds[j["id"]]]
    save(path.parent/"a40_budget.json",dict(campaign=ref(path),**budget),"manifest_sha256")
    if excessive:
        raise ValueError("measured high runtime estimate exceeds prepared limits: "+", ".join(excessive))
    return save(path.parent/"a40_smoke_gate.json",dict(campaign=ref(path),**gate),"manifest_sha256")


def submit_all(path, campaign):
    """Queue the complete DAG from the login host; never submit nested jobs."""
    from .candidate_campaign import validate, result_for
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit only from the login host")
    if not campaign["profile_payload"].get("extension_protocol"):
        raise ValueError("unattended startup dependencies require an extension campaign")
    validate(campaign)
    for reference in campaign["queue_controls"].values():
        checked(reference)
    binary=Path(campaign["profile_payload"]["slurm"]["bin"])
    phase=campaign["evaluation_phase"]
    journal=path.parent/f"submission_{phase}.json"
    smoke_journal=path.parent/"submission_smoke.json"
    records, smoke_records = [], []
    gate_record=collection=monitor=None

    def enqueue(script, dependencies=None, condition="afterok", minutes=None):
        command=[str(binary/"sbatch"),"--parsable"]
        if minutes is not None:
            command.append(f"--time={minutes // 60:02d}:{minutes % 60:02d}:00")
        if dependencies:
            command += ["--dependency="+condition+":"+":".join(dependencies),"--kill-on-invalid-dep=yes"]
        command.append(script["path"])
        job_id=subprocess.check_output(command,text=True).strip().split(";")[0]
        if not job_id.isdecimal():
            raise ValueError("invalid Slurm submission response")
        return dict(slurm_job_id=job_id,command=command)

    def persist(status):
        _atomic_json(journal,dict(status=status,phase=phase,jobs=records,smoke_jobs=smoke_records,
                                 gate_job=gate_record,collection_job=collection,monitor_job=monitor))

    with (path.parent/"submission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if journal.exists():
            raise ValueError("submission journal already exists; inspect it before retrying")
        smoke_jobs=[j for j in campaign["jobs"] if j["phase"]=="smoke"]
        if smoke_journal.exists():
            previous=read(smoke_journal)
            if {j["id"] for j in previous["jobs"]}!={j["id"] for j in smoke_jobs}:
                raise ValueError("prior startup submission has incomplete membership")
            if any(result_for(j) is None for j in smoke_jobs):
                raise ValueError("existing startup jobs have not all completed; no duplicate submission")
            smoke_records=previous["jobs"]
            finish_gate(path,campaign)
            gate_dependency=[]
            persist("submitting")
        else:
            persist("submitting")
            for job in sorted(smoke_jobs,key=lambda j:(-j["E"],j["id"])):
                record=dict(id=job["id"],**enqueue(job["sbatch_script"]))
                smoke_records.append(record)
                _atomic_json(smoke_journal,dict(status="submitting",phase="smoke",jobs=smoke_records))
                persist("submitting")
            _atomic_json(smoke_journal,dict(status="submitted",phase="smoke",jobs=smoke_records))
            gate_record=enqueue(campaign["queue_controls"]["gate"],[r["slurm_job_id"] for r in smoke_records])
            gate_dependency=[gate_record["slurm_job_id"]]
            persist("submitting")
        estimates={}
        measured_limits={}
        budget_path=path.parent/"a40_budget.json"
        if budget_path.exists():
            budget=read(budget_path)
            estimates={j["id"]:j["central_seconds"] for j in budget["jobs"]}
            measured_limits={j["id"]:max(15,math.ceil((j["high_seconds"]+600)/60)) for j in budget["jobs"]}
        jobs=sorted((j for j in campaign["jobs"] if j["phase"]==phase),
                    key=lambda j:(-estimates.get(j["id"],j["walltime_minutes"]*60),-j["requests"],-j["E"],j["id"]))
        for job in jobs:
            minutes=min(job["walltime_minutes"],measured_limits.get(job["id"],job["walltime_minutes"]))
            records.append(dict(id=job["id"],walltime_minutes=minutes,**enqueue(job["sbatch_script"],gate_dependency,minutes=minutes)))
            persist("submitting")
        collection=enqueue(campaign["collection_job"],[r["slurm_job_id"] for r in records],"afterany")
        persist("submitting")
        monitor=enqueue(campaign["queue_controls"]["monitor"])
        persist("submitted")
    return dict(submitted_evaluation_jobs=len(records),startup_jobs=len(smoke_records),
                gate=gate_record,collection=collection,monitor=monitor)
