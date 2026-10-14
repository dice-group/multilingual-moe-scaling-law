"""Create-only search package, bounded CPU submissions, and evidence handoff.

Workers never submit jobs. The submission command runs on the login host and
refuses nested allocations, duplicate attempts, or concurrent campaign waves.
"""
from __future__ import annotations

from .state import load_plan
from .state import prerequisites
import argparse
from importlib.metadata import version
import gzip
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .. import execution as io
from .. import fit
from . import data
from . import model
from ..problems import counts
from ..data import canonical_json_sha256
from .protocol import PROTOCOL
from .protocol import SCHEMA
from .protocol import PRIMARY
from .protocol import INNER
from .protocol import SECONDARY
from .protocol import SHARED
from .protocol import check_budget
from ..artifacts import binding
from ..artifacts import verify
from ..artifacts import validate_output
from ..artifacts import environment
from ..problems import inventories
from ..repetition_campaign import load as load_control
from ..artifacts import copy_dependency_snapshot


def registered_tasks(specs):
    tasks = []
    def add(task_id,kind,fold,phase,minutes,stage,**kw):
        tasks.append(dict(id=task_id,kind=kind,fold=fold,phase=phase,minutes=minutes,cores=1,stage=stage,**kw))
    for fold in (*INNER,*SECONDARY,"all_grid"):
        add(f"cal-{fold}","calibrate",fold,"selection" if fold in INNER else "pilot",
            180 if fold == "all_grid" else 120,"calibration")
    for fold in INNER:
        for name in model.CANDIDATES:
            for tau in model.TAUS:
                add(f"fit-{fold}-{name}-t{tau:g}","shared",fold,"selection",120,"inner",model=name,tau=tau)
    for spec in specs:
        add(f"native-{spec['name']}","native","all_grid","baselines",150,"baselines",spec=spec)
    for name in SHARED:
        add(f"shared-{name}-all_grid","shared","all_grid","baselines",180,"baselines",model=name)
    for fold in (*PRIMARY,"all_grid"):
        for spec in model.POOLED:
            add(f"pooled-{spec.name}-{fold}","shared",fold,"baselines",180,"baselines",model=spec.name)
        for label,name,fixed in (("selected","selected",{}),("control-RN","RN",{}),("control-R","RN",{"rho":0.})):
            add(f"{label}-{fold}","shared",fold,"final",240,"final",model=name,fixed=fixed)
    for fold in SECONDARY:
        for label,name in (("selected","selected"),("control-RN","RN"),("shared-M6-ND","M6-ND")):
            add(f"{label}-{fold}","shared",fold,"final",120,"secondary",model=name)
    add("no_EK-all_grid","no_EK","all_grid","final",120,"stability",model="selected")
    for i,rho in enumerate(PROTOCOL["rho_profile_values"]):
        add(f"rho-{i:02d}","rho_profile","all_grid","stability",120,"stability",model="selected",rho=rho)
    for i in range(PROTOCOL["bootstrap_refits"]):
        add(f"bootstrap-{i:02d}","bootstrap","all_grid","stability",120,"stability",model="selected",replicate=i)
    check_budget(tasks)
    return tasks


def prepare(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    root = validate_output(profile)
    if root.exists():
        raise ValueError("search preparation is create-only; choose a new version before fitting")
    if str(Path(sys.executable).resolve()) != str(Path(profile["python_executable"]).resolve()):
        raise ValueError("prepare with the configured bridge-train interpreter")
    if profile["concurrency"] != PROTOCOL["max_concurrent_cores"]:
        raise ValueError("registered concurrency changed")
    control_path = Path(profile["prepared_control"]).resolve()
    control = load_control(control_path)
    for entry in control["inputs"]:
        verify(entry)
    inputs = [binding(profile_path),binding(Path(profile["input_profile"])),binding(control_path),
              *control["inputs"],*control["frozen_inputs"]]
    previous = json.loads(Path(profile["input_profile"]).read_text())
    literature,shared = Path(previous["literature"]),Path(previous["shared_fits"])
    native_packet_path = literature/"packets.json.gz"
    native_packets = json.loads(gzip.decompress(native_packet_path.read_bytes()))
    inv = inventories(native_packets)
    metadata = {}
    for packet in native_packets:
        lang = packet["language"]
        entry = dict(family=packet["family"],script=lang.rsplit("_",1)[1])
        if lang in metadata and metadata[lang] != entry:
            raise ValueError("source family changed across frozen packets")
        metadata[lang] = entry
    rows = pd.read_parquet(control_path.parent/"training_grid.parquet")
    receipt = json.loads(Path(previous["canonical_input_receipt"]).read_text())
    configs_path = Path(next(e["path"] for e in receipt["source_tables"] if e["name"] == "model_configs"))
    count_map,count_evidence = counts(pd.read_parquet(configs_path))
    rows["N_total"] = [count_map[str(l),int(e)] for l,e in zip(rows.ladder,rows.E)]
    endpoint = pd.read_parquet(control_path.parent/"endpoint.parquet")
    endpoint["N_total"] = profile["L26_configured_total_parameters"]
    endpoint["unigram_nll"] = -endpoint.unigram_log_probability_sum/endpoint.token_count
    folds = json.loads((shared/"folds.json").read_text())
    memberships,preflight = data.splits(rows,folds,receipt["fit_rows_content_sha256"])
    old_selected = json.loads((shared/"selected.json").read_text())
    parent = json.loads(Path(previous["parent_plan"]).read_text())
    calibration = {}
    for fold in PRIMARY:
        contract = parent["folds"][fold]["manifest"]
        for key in ("train_trajectory_ids","test_trajectory_ids","score_languages"):
            if memberships[fold][key] != contract["membership"][key]:
                raise ValueError("frozen baseline and candidate fold contracts disagree")
        record = old_selected[fold]["M6-ND"]
        x = [*fit.outer_from_shared("M6",fit.SharedParameters(**record["shared_parameters"])),record["zeta"]]
        calibration[fold] = dict(delta=contract["fit"]["delta"],regularization=contract["fit"]["regularization"],
                                 anchor=x,source="exact frozen outer training contract; not used for inner starts")
    status = pd.read_csv(literature/"model_status.csv")
    certified = sorted(status.loc[status.status.eq("certified"),"model"])
    if len(certified) != 42 or len(status)-len(certified) != 11:
        raise ValueError("native benchmark panel identity changed")
    selections_path = literature/"selected.json.gz"
    native_selected = json.loads(gzip.decompress(selections_path.read_bytes()))
    specs = {}
    for value in native_selected:
        spec = value["spec"]
        if spec["name"] in certified:
            if spec["name"] in specs and specs[spec["name"]] != spec:
                raise ValueError("native model spec varies across targets")
            specs[spec["name"]] = spec
    if set(specs) != set(certified):
        raise ValueError("missing certified native model definition")
    inputs += [binding(selections_path),binding(literature/"manifest.json"),binding(configs_path)]
    tasks = registered_tasks([specs[n] for n in certified])
    root.mkdir(parents=True)
    scoring = root/"scoring"; scoring.mkdir()
    rows.to_parquet(scoring/"grid.parquet",index=False)
    endpoint.to_parquet(scoring/"endpoint.parquet",index=False)
    for name in ("baseline_grid.parquet","baseline_endpoint.csv","baseline_status.csv"):
        shutil.copy2(control_path.parent/name,scoring/name)
    io.write_json_once(scoring/"exposure_contract.json",dict(metadata=metadata,
        inventories=[dict(language=l,K=k,U=u) for (l,k),u in sorted(inv.items())],counts=count_evidence))
    training, inner_scoring = {},{}
    covariates = rows.drop(columns=list(data.RESPONSES & set(rows.columns)))
    for fold,member in memberships.items():
        train = data.subset(rows,member,"train")
        training[fold] = data.write_training(root,fold,train,covariates,inv,metadata)
        if fold in INNER:
            test = data.subset(rows,member,"test")
            path = root/"inner_scoring"/fold; path.mkdir(parents=True)
            test.to_parquet(path/"rows.parquet",index=False)
            sources = data.source_arrays(test,covariates,inv,metadata)
            np.savez_compressed(path/"sources.npz",**sources.__dict__)
            inner_scoring[fold] = [binding(path/n) for n in ("rows.parquet","sources.npz")]
    frozen = root/"source"/"stage1_fit"
    copy_dependency_snapshot(frozen, ("law_search.__main__", "law_search.driver", "law_search.recovery", "law_search.atlas_followup"))
    for directory in ("results","logs","submissions"):
        (root/directory).mkdir()
    plan = dict(schema=SCHEMA,protocol=PROTOCOL,profile=profile,tasks=tasks,memberships=memberships,
        training=training,inner_scoring=inner_scoring,calibration=calibration,preflight=preflight,
        native_specs=[specs[n] for n in certified],unresolved=status.loc[~status.status.eq("certified"),"model"].tolist(),
        allocations=check_budget(tasks),environment=environment(),
        runtime={"python_executable":str(Path(sys.executable).resolve()),"pyarrow":version("pyarrow")},
        inputs=inputs,scoring=[binding(p) for p in sorted(scoring.iterdir())],
        source=[binding(p) for p in sorted(frozen.rglob("*.py"))],
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        git_status=subprocess.check_output(["git","status","--short"],text=True),upstream=control["upstream"])
    plan["sha256"] = canonical_json_sha256(plan)
    io.write_json_once(root/"plan.json",plan)
    print(json.dumps(dict(plan_sha256=plan["sha256"],tasks=len(tasks),reserved_core_hours=sum(plan["allocations"].values()),
                          training_packets=len(training),preflight={k:v for k,v in preflight.items() if k not in ("secondary_holdouts","generalization_holdout")})))
    return plan


def scheduler_records(job_ids):
    if not job_ids:
        return []
    output = subprocess.check_output(["sacct","-n","-P","-j",",".join(job_ids),
        "--format=JobID,State,AllocCPUS,ElapsedRaw,TimelimitRaw"],text=True)
    result = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts)<5 or "." in parts[0] or "_" not in parts[0] or not parts[2].isdigit():
            continue
        result.append(dict(job_id=parts[0],state=parts[1].split()[0],cores=int(parts[2]),
                           seconds=int(parts[3] or 0),limit_minutes=int(parts[4] or 0)))
    return result


def accounted_reservations(plan,receipts,accounting):
    """Charge at least each registered cap; unexpected allocation size fails closed."""
    measured={r["job_id"]:r for r in accounting}
    indexed={t["id"]:t for t in plan["tasks"]}
    reservations=[]
    for receipt in receipts:
        for index,task_id in enumerate(receipt["task_ids"]):
            task=indexed[task_id]
            row=measured.get(f"{receipt['job_id']}_{index}")
            if row is None:
                raise ValueError("scheduler accounting is incomplete; wait before submitting more work")
            if row["cores"] != 1:
                raise ValueError("observed allocation differs from the single-core contract")
            actual=row["cores"]*row["seconds"]/3600
            reservations.append(dict(phase=task["phase"],allocated_core_hours=max(task["minutes"]/60,actual)))
    return reservations


def render_array(plan_path,plan,tasks,submission_id):
    if not tasks or len({t["minutes"] for t in tasks}) != 1:
        raise ValueError("one array must have a homogeneous wall-time limit")
    root,p = Path(plan_path).resolve().parent,plan["profile"]
    ids = [t["id"] for t in tasks]
    command = shlex.join([p["python_executable"],"-m","stage1_fit.law_search","worker","--plan",str(Path(plan_path).resolve())])
    script = ["#!/bin/bash",f"#SBATCH --job-name=lawsearch_{submission_id}",
        f"#SBATCH --account={p['account']}",f"#SBATCH --partition={p['partition']}",
        "#SBATCH --ntasks=1","#SBATCH --cpus-per-task=1",f"#SBATCH --mem={p['memory']}",
        f"#SBATCH --time={tasks[0]['minutes']}",f"#SBATCH --array=0-{len(tasks)-1}%{min(32,len(tasks))}",
        f"#SBATCH --output={shlex.quote(str(root/'logs'/(submission_id+'-%A_%a.out')))}",
        "set -euo pipefail",f"export PYTHONPATH={shlex.quote(str(root/'source'))}",
        "export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "tasks=("+" ".join(shlex.quote(i) for i in ids)+")",command+' --task "${tasks[$SLURM_ARRAY_TASK_ID]}"',""]
    return "\n".join(script)


def submit(plan_path,stage,*,test_only=False,limit=None):
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("submit from login host; nested jobs are forbidden")
    plan = load_plan(plan_path,all_inputs=True)
    root = Path(plan_path).resolve().parent
    # Directory creation is atomic across concurrent command invocations.
    lock = root/"submissions"/"submit.lock"
    lock.mkdir()
    try:
        receipts = [io.read_json(p) for p in sorted((root/"submissions").glob("*.json"))]
        submitted = {i for r in receipts for i in r["task_ids"]}
        jobs = [r["job_id"] for r in receipts if r.get("job_id")]
        if any(not r.get("job_id") for r in receipts):
            raise ValueError("unresolved submission intent: reconcile scheduler before another submission")
        if jobs:
            queued = subprocess.check_output(["squeue","-h","-j",",".join(jobs),"-o","%i"],text=True).strip()
            if queued:
                raise ValueError("a campaign array is still queued/running; global 32-core cap forbids overlapping waves")
        candidates = [t for t in plan["tasks"] if t["stage"] == stage and t["id"] not in submitted]
        if not candidates:
            raise ValueError("no unsubmitted tasks remain in this stage")
        minutes = candidates[0]["minutes"]
        candidates = [t for t in candidates if t["minutes"] == minutes]
        ready = []
        for task in candidates:
            requirements = prerequisites(task,root,plan)
            if all(p.exists() and (p.name == "selection.json" or io.read_json(p).get("certified")) for p in requirements):
                ready.append(task)
        if limit is not None:
            if limit <= 0:
                raise ValueError("submission limit must be positive")
            ready = ready[:limit]
        if not ready:
            raise ValueError("stage prerequisites are missing or uncertified")
        accounting = scheduler_records(jobs)
        # All earlier jobs remain charged at their full allocated wall-time cap;
        # completed unused hours are not silently recycled into extra fits.
        check_budget(ready,accounted_reservations(plan,receipts,accounting))
        submission_id = f"{len(receipts):03d}-{stage}"
        script = render_array(plan_path,plan,ready,submission_id)
        subprocess.run(["bash","-n"],input=script,text=True,check=True)
        if test_only:
            print(subprocess.check_output(["sbatch","--test-only"],input=script,text=True).strip())
            return
        script_path = root/"submissions"/f"{submission_id}.sbatch"
        io.write_text_once(script_path,script)
        intent_path = root/"submissions"/f"{submission_id}-intent.json"
        io.write_json_once(intent_path,dict(task_ids=[t["id"] for t in ready],job_id=None,script=binding(script_path)))
        job_id = subprocess.check_output(["sbatch","--parsable",str(script_path)],text=True).strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError("ambiguous sbatch response; inspect immutable submission intent")
        io.write_json_once(root/"submissions"/f"{submission_id}.json",dict(job_id=job_id,task_ids=[t["id"] for t in ready],
            plan_sha256=plan["sha256"],reserved_core_hours=sum(t["minutes"]/60 for t in ready),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),previous_accounting=accounting,script=binding(script_path)))
        # Retain the intent as evidence outside the active receipt glob.
        intent_path.rename(intent_path.with_suffix(".resolved"))
        print(json.dumps(dict(job_id=job_id,tasks=len(ready),reserved_core_hours=sum(t["minutes"]/60 for t in ready))))
    finally:
        lock.rmdir()


def status(plan_path):
    plan = load_plan(plan_path); root = Path(plan_path).resolve().parent
    records = [(t,io.read_json(root/"results"/f"{t['id']}.json")) for t in plan["tasks"]
               if (root/"results"/f"{t['id']}.json").exists()]
    receipts = [io.read_json(p) for p in sorted((root/"submissions").glob("*.json"))]
    accounting = scheduler_records([r["job_id"] for r in receipts if r.get("job_id")])
    print(json.dumps(dict(plan_sha256=plan["sha256"],planned_tasks=len(plan["tasks"]),completed_receipts=len(records),
        certified=sum(bool(r.get("certified")) for _,r in records),
        unresolved=[dict(task=t["id"],error=r.get("error"),timed_out=r.get("timed_out")) for t,r in records if not r.get("certified")],
        allocated_finished_core_hours=sum(r["cores"]*r["seconds"]/3600 for r in accounting
            if r["state"] not in ("RUNNING","PENDING","CONFIGURING","COMPLETING")),
        reserved_submission_core_hours=sum(r.get("reserved_core_hours",0.) for r in receipts),
        accounting=accounting,selection_exists=(root/"selection.json").exists()),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","verify","worker","submit","status","select","report"))
    parser.add_argument("--profile",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--task")
    parser.add_argument("--stage",choices=("calibration","inner","baselines","final","secondary","stability"))
    parser.add_argument("--test-only",action="store_true")
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.profile)
    elif args.action == "verify":
        plan=load_plan(args.plan,all_inputs=True); print(json.dumps(dict(verified=True,plan_sha256=plan["sha256"])))
    elif args.action == "worker":
        from .worker import worker
        worker(args.plan,args.task)
    elif args.action == "submit":
        submit(args.plan,args.stage,test_only=args.test_only,limit=args.limit)
    elif args.action == "status":
        status(args.plan)
    else:
        from .reporting import select
        from .reporting import report
        (select if args.action == "select" else report)(args.plan)


if __name__ == "__main__":
    main()
