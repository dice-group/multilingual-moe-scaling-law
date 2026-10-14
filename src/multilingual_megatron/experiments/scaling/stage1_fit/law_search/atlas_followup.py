"""One bounded ATLAS-only numerical follow-up; RN and old receipts are immutable."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd

from .. import execution as io
from . import data
from ..atlas_comparison_workflow import model as m
from ..atlas_comparison_workflow import numerics as n
from ..atlas_comparison_workflow import recovery_numerics as recovery
from ..data import canonical_json_sha256
from .state import load_plan
from .campaign import scheduler_records
from .reporting import load_result
from ..scoring import predict_result
from ..scoring import checked_score
from ..artifacts import binding
from ..artifacts import verify

RULES = dict(schema="stage1_atlas_followup/v1",minutes_per_model=24,report_minutes=15,
    maximum_models=29,maximum_core_hours=12.,max_concurrent_cores=32,
    new_models="original native 16/64 starts followed by existing training-only recovery",
    existing_models="reuse certified language groups; existing training-only recovery on failures",
    numerical_recovery=recovery.SETTINGS,gradient_tolerance=1e-8,
    objective_tolerance=1e-6,prediction_tolerance=1e-4,equation_tolerance=1e-10,
    fit_support="all_grid only; original donors, inventories, Huber delta, bounds and equations",
    candidate="frozen full RN; diagnostic follow-up, not reselection or prospective validation",
    gpu_training=False)

EXTENDED_RULES = RULES | dict(schema="stage1_atlas_followup/v2",minutes_per_model=720,
    maximum_models=11,maximum_core_hours=132.25,
    authorization="2026-09-14 user explicitly authorized higher time limits and CPU hours to finish unfinished variants",
    resume="retain completed language fits from v1; fit or recover only uncertified groups",
    replaces_prior_resource_ceiling=True)


def budget(count, previous):
    extra=(count*RULES["minutes_per_model"]+RULES["report_minutes"])/60
    if not 0<count<=29 or extra>12 or previous["charged_core_hours"]+extra>512:
        raise ValueError("follow-up exceeds approved total CPU cap")
    if previous["phase_charged_core_hours"]["baselines"]+extra>200:
        raise ValueError("follow-up exceeds original baseline phase ceiling")
    return extra


def registered_specs(parent):
    names={s["name"] for s in parent["native_specs"]}|set(parent["unresolved"])
    available={s.name:s for s in m.panel()}
    if names-set(available):raise ValueError("registered native form missing from implementation")
    return {name:available[name] for name in sorted(names) if name.startswith("ATLAS-")}


def extended_budget(count):
    if not 0<count<=EXTENDED_RULES["maximum_models"]:
        raise ValueError("extended pass is limited to the eleven unfinished variants")
    return (count*EXTENDED_RULES["minutes_per_model"]+EXTENDED_RULES["report_minutes"])/60


def prepare(parent_path):
    parent=load_plan(parent_path,all_inputs=True);root=Path(parent_path).resolve().parent
    output=root/"atlas_followup_v1"
    if output.exists():raise ValueError("create-only follow-up already exists")
    tasks=[];previous={}
    specs=registered_specs(parent)
    for name,spec in specs.items():
        task_id="native-"+name
        if any(t["id"]==task_id for t in parent["tasks"]):
            result=load_result(root,parent,task_id)
            if result.get("certified"):continue
            previous[name]=result
        else:
            if name not in parent["unresolved"]:raise ValueError("unexpected new ATLAS form")
            previous[name]=dict(model=name,groups=[],expected_languages=200,certified=False)
        tasks.append(dict(id=name,index=len(tasks),spec=spec.payload(),new_model=not previous[name]["groups"]))
    accounting=io.read_json(root/"resource_accounting.json")
    extra=budget(len(tasks),accounting)
    output.mkdir()
    for task in tasks:
        path=output/"previous"/(task["id"]+".json")
        io.write_json_once(path,previous[task["id"]]);task["previous"]=binding(path)
    source=output/"source"/"atlas_followup"
    shutil.copytree(Path(__file__).parents[1],source,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    plan=dict(rules=RULES,parent=binding(Path(parent_path).resolve()),parent_sha256=parent["sha256"],
        atlas_panel=[s.payload() for s in specs.values()],
        profile=parent["profile"],tasks=tasks,training=parent["training"]["all_grid"],
        calibration=binding(root/"results/cal-all_grid.json"),
        prior_accounting=binding(root/"resource_accounting.json"),extra_reserved_core_hours=extra,
        rn=binding(root/"results/control-RN-all_grid.json"),
        source=[binding(p) for p in sorted(source.rglob("*.py"))])
    plan["sha256"]=canonical_json_sha256(plan)
    io.write_json_once(output/"plan.json",plan)
    print(dict(plan=str(output/"plan.json"),tasks=len(tasks),extra_core_hours=extra),flush=True)


def resume(previous_path):
    previous=load(previous_path);old_root=Path(previous_path).resolve().parent
    if previous["rules"]!=RULES:raise ValueError("this extension resumes v1 only")
    old_report=io.read_json(old_root/"report.json")
    if old_report["plan_sha256"]!=previous["sha256"]:raise ValueError("foreign previous report")
    output=old_root.parent/"atlas_followup_v2"
    if output.exists():raise ValueError("create-only extended pass already exists")
    tasks=[];reused={}
    for task in previous["tasks"]:
        path=old_root/"results"/(task["id"]+".json")
        result=io.read_json(path)
        if result["plan_sha256"]!=previous["sha256"] or result["task_sha256"]!=canonical_json_sha256(task):
            raise ValueError("foreign previous fit")
        reused[task["id"]]=binding(path)
        if not result["certified"]:
            tasks.append(task|dict(index=len(tasks),previous=binding(path),new_model=False))
    extra=extended_budget(len(tasks))
    output.mkdir()
    source=output/"source"/"atlas_followup"
    shutil.copytree(Path(__file__).parents[1],source,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    plan=previous|dict(rules=EXTENDED_RULES,tasks=tasks,extra_reserved_core_hours=extra,
        previous_followup=binding(Path(previous_path).resolve()),previous_report=binding(old_root/"report.json"),
        previous_results=reused,
        earlier_charged_core_hours=io.read_json(verify(previous["prior_accounting"]))["charged_core_hours"]+old_report["charged_extra_core_hours"],
        source=[binding(p) for p in sorted(source.rglob("*.py"))])
    plan.pop("sha256")
    plan["sha256"]=canonical_json_sha256(plan)
    io.write_json_once(output/"plan.json",plan)
    print(dict(plan=str(output/"plan.json"),tasks=len(tasks),minutes_per_model=720,extra_core_hours=extra),flush=True)


def load(path):
    plan=io.read_json(Path(path));unsigned=dict(plan);digest=unsigned.pop("sha256")
    if digest!=canonical_json_sha256(unsigned) or plan["rules"] not in (RULES,EXTENDED_RULES):
        raise ValueError("follow-up protocol changed")
    for entry in [plan["parent"],plan["training"],plan["calibration"],plan["prior_accounting"],*plan["source"]]:verify(entry)
    parent=load_plan(plan["parent"]["path"])
    if parent["sha256"]!=plan["parent_sha256"]:raise ValueError("foreign parent")
    if plan["rules"]==RULES:
        budget(len(plan["tasks"]),io.read_json(verify(plan["prior_accounting"])))
    else:
        if extended_budget(len(plan["tasks"]))!=plan["extra_reserved_core_hours"]:
            raise ValueError("extended allocation changed")
        verify(plan["previous_followup"]);verify(plan["previous_report"])
        for entry in plan["previous_results"].values():verify(entry)
    return plan


def recover_group(spec,packet,previous):
    if previous.get("certified"):return previous
    if not previous.get("attempts"):
        previous=previous | n.fit_group(spec,packet,packet)
    if previous.get("certified"):return previous
    fixed=recovery.recover(spec,packet,packet,previous)
    return previous | fixed | {"prior_fit":previous}


def worker(path,index):
    plan=load(path);root=Path(path).resolve().parent;task=plan["tasks"][index]
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK")!="1":
        raise ValueError("single-core Slurm allocation required")
    if any(os.environ.get(k) not in (None,"","0") for k in ("SLURM_GPUS","SLURM_GPUS_ON_NODE","SLURM_JOB_GPUS")):
        raise ValueError("GPU work is forbidden")
    if any(os.environ.get(k)!="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")):
        raise ValueError("single-thread numerical runtime required")
    if (root/"results"/(task["id"]+".json")).exists():raise ValueError("no automatic retry")
    _,_,packets=data.read_training(plan["training"])
    cal=io.read_json(verify(plan["calibration"]));spec=m.Spec(**task["spec"])
    old=io.read_json(verify(task["previous"]));by_language={g["language"]:g for g in old["groups"]}
    groups=[];began=time.monotonic();deadline=began+plan["rules"]["minutes_per_model"]*60-45
    def expired(signum,frame):raise TimeoutError("bounded model recovery expired")
    signal.signal(signal.SIGALRM,expired)
    for item in packets:
        packet=item["train"]|{"delta":cal["delta"]}
        previous=by_language.get(item["language"],dict(certified=False,spec=spec.payload()))
        group=previous
        if not previous.get("certified"):
            if time.monotonic()>=deadline:
                group=previous|dict(certified=False,error="model wall-time cap reached")
            else:
                signal.setitimer(signal.ITIMER_REAL,deadline-time.monotonic())
                try:group=recover_group(spec,packet,previous)
                except (TimeoutError,ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                    group=previous|dict(certified=False,error=f"{type(error).__name__}: {error}")
                finally:signal.setitimer(signal.ITIMER_REAL,0.)
        group=group|dict(language=item["language"],donors=item["donors"],
            unigram_nll=item["unigram_nll"],training_minimum=packet["training_minimum"])
        if group.get("certified"):group.pop("error",None)
        io.write_json_once(root/"groups"/task["id"]/(item["language"]+".json"),group)
        groups.append(group)
        print(dict(model=task["id"],language=item["language"],certified=group.get("certified"),
            elapsed_seconds=round(time.monotonic()-began,2)),flush=True)
    result=dict(model=task["id"],groups=groups,expected_languages=len(packets),
        certified=len(groups)==len(packets) and all(g.get("certified") for g in groups),
        plan_sha256=plan["sha256"],task_id=task["id"],task_sha256=canonical_json_sha256(task),
        previous=task["previous"],heldout_scores_computed=False,elapsed_seconds=time.monotonic()-began,
        slurm_job_id=os.environ["SLURM_JOB_ID"])
    io.write_json_once(root/"results"/(task["id"]+".json"),result)


def script(path,plan,report=False):
    root=Path(path).resolve().parent;p=plan["profile"]
    rules=plan.get("rules",RULES)
    action="report" if report else "worker"
    command=shlex.join([p["python_executable"],"-m","atlas_followup.law_search.atlas_followup",action,"--plan",str(Path(path).resolve())])
    lines=["#!/bin/bash",f"#SBATCH --account={p['account']}",f"#SBATCH --partition={p['partition']}",
        f"#SBATCH --job-name=atlas_followup_{action}","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={p['memory']}",f"#SBATCH --time={rules['report_minutes'] if report else rules['minutes_per_model']}",
        f"#SBATCH --output={shlex.quote(str(root/'logs'/(action+'-%A_%a.out')))}"]
    if not report:lines.append(f"#SBATCH --array=0-{len(plan['tasks'])-1}%{min(32,len(plan['tasks']))}")
    lines += ["set -euo pipefail",f"export PYTHONPATH={shlex.quote(str(root/'source'))}",
        "export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        command if report else command+' --task "$SLURM_ARRAY_TASK_ID"',""]
    return "\n".join(lines)


def active_parent_jobs(jobs):
    # Completed IDs expire from squeue; querying the user's active queue avoids
    # treating that expiry as an access failure or silently ignoring errors.
    queued=subprocess.check_output(["squeue","-h","-u",str(os.getuid()),"-o","%i"],text=True)
    return set(jobs)&{line.strip().split("_")[0] for line in queued.splitlines() if line.strip()}


def submit(path):
    if os.environ.get("SLURM_JOB_ID"):raise ValueError("no nested Slurm submissions")
    plan=load(path);root=Path(path).resolve().parent;parent=Path(plan["parent"]["path"]).parent
    if "previous_followup" in plan:
        previous_root=Path(plan["previous_followup"]["path"]).parent
        previous_jobs=[io.read_json(previous_root/(a+"_submission.json"))["job_id"] for a in ("worker","report")]
        if active_parent_jobs(previous_jobs):raise ValueError("previous follow-up is still active")
    for subdir in (parent/"submissions",parent/"native_box_recovery"):
        receipts=[io.read_json(p) for p in subdir.glob("*.json") if p.name=="submission.json" or subdir.name=="submissions"]
        jobs=[r["job_id"] for r in receipts if r.get("job_id")]
        if jobs and active_parent_jobs(jobs):
            raise ValueError("a parent campaign job is still active")
    for action in ("worker","report"):
        receipt=root/(action+"_submission.json")
        if receipt.exists():continue
        intent=root/(action+"_intent.json")
        if intent.exists():raise ValueError("reconcile existing submission intent; never submit a duplicate")
        (root/"logs").mkdir(exist_ok=True)
        content=script(path,plan,action=="report")
        subprocess.run(["bash","-n"],input=content,text=True,check=True)
        target=root/(action+".sbatch");io.write_text_once(target,content)
        io.write_json_once(intent,dict(plan_sha256=plan["sha256"],script=binding(target)))
        command=["sbatch","--parsable"]
        if action=="report":command.append("--dependency=afterany:"+io.read_json(root/"worker_submission.json")["job_id"])
        job=subprocess.check_output([*command,str(target)],text=True).strip().split(";")[0]
        if not job.isdigit():raise ValueError("ambiguous scheduler submission")
        io.write_json_once(receipt,dict(job_id=job,plan_sha256=plan["sha256"],script=binding(target),
            submitter_source=binding(Path(__file__))))
        print(dict(action=action,job_id=job),flush=True)


def report(path):
    plan=load(path);root=Path(path).resolve().parent;parent_root=Path(plan["parent"]["path"]).parent
    parent=load_plan(plan["parent"]["path"],all_inputs=True)
    if (root/"report.json").exists():raise ValueError("report is immutable")
    job=io.read_json(root/"worker_submission.json")["job_id"]
    accounting=scheduler_records([job]);observed={r["job_id"]:r for r in accounting}
    for task in plan["tasks"]:
        record=observed.get(f"{job}_{task['index']}")
        if record is None or record["cores"]!=1 or record["state"] in ("RUNNING","PENDING","COMPLETING","CONFIGURING"):
            raise ValueError("worker CPU accounting incomplete or unexpected")
    rules=plan["rules"]
    charged=sum(max(rules["minutes_per_model"]/60,r["cores"]*r["seconds"]/3600) for r in accounting)+rules["report_minutes"]/60
    if charged>rules["maximum_core_hours"]+1e-10:
        raise ValueError("actual CPU accounting exceeds budget")
    if rules==RULES and charged+io.read_json(verify(plan["prior_accounting"]))["charged_core_hours"]>512:
        raise ValueError("v1 CPU budget exceeded")
    endpoint=pd.read_parquet(parent_root/"scoring/endpoint.parquet")
    contract=io.read_json(parent_root/"scoring/exposure_contract.json")
    inv={(r["language"],r["K"]):r["U"] for r in contract["inventories"]}
    universe=endpoint.drop(columns=list(data.RESPONSES & set(endpoint.columns)))
    rn=io.read_json(verify(plan["rn"]))
    if not rn.get("certified") or rn.get("model")!="RN":raise ValueError("full RN must remain certified and frozen")
    rn_predictions=predict_result(rn,endpoint,universe,inv,contract["metadata"])
    rn_score=checked_score(rn_predictions,endpoint,True)
    if len(endpoint)!=200 or not rn_score["complete"]:raise ValueError("RN endpoint cohort changed")
    metrics=[];predictions=[];inputs=[]
    tasks={t["id"]:t for t in plan["tasks"]}
    for spec in (m.Spec(**s) for s in plan["atlas_panel"]):
        if spec.name in tasks:
            result_path=root/"results"/(spec.name+".json")
            result=io.read_json(result_path) if result_path.exists() else dict(groups=[],certified=False)
            if result_path.exists():
                if result["plan_sha256"]!=plan["sha256"] or result["task_sha256"]!=canonical_json_sha256(tasks[spec.name]):
                    raise ValueError("foreign recovery result")
                inputs.append(binding(result_path))
        elif spec.name in plan.get("previous_results",{}):
            entry=plan["previous_results"][spec.name]
            result=io.read_json(verify(entry));inputs.append(entry)
        else:result=load_result(parent_root,parent,"native-"+spec.name)
        pred=predict_result(result,endpoint,universe,inv,contract["metadata"])
        score=checked_score(pred,endpoint,True)
        complete=bool(result.get("certified") and score["complete"])
        metrics.append(dict(model=spec.name,certified=bool(result.get("certified")),
            rn_beats=bool(complete and rn_score["rmse"]<score["rmse"]-1e-6),**score))
        predictions.append(pred.assign(model=spec.name))
    pd.DataFrame(metrics).to_csv(root/"metrics.csv",index=False)
    pd.concat([*predictions,rn_predictions.assign(model="RN")],ignore_index=True).to_parquet(root/"predictions.parquet",index=False)
    result=dict(plan_sha256=plan["sha256"],rn=rn_score,atlas_models=len(metrics),
        certified_scoreable=sum(r["certified"] and r["complete"] for r in metrics),
        rn_audited_wins=sum(r["rn_beats"] for r in metrics),
        all_atlas_win=all(r["rn_beats"] for r in metrics),metrics=metrics,inputs=inputs,
        charged_extra_core_hours=charged,worker_allocated_core_hours=sum(r["cores"]*r["seconds"]/3600 for r in accounting),
        worker_accounting=accounting,report_allocation="one CPU, at most 15 minutes, charged in full",
        evidence="all-grid L26 comparison only; post-observation numerical follow-up, no RN reselection or refit",
        unchanged_parent_report=binding(parent_root/"report/report.json"))
    io.write_json_once(root/"report.json",result)
    print({k:v for k,v in result.items() if k not in ("metrics","inputs","worker_accounting")},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","resume","submit","worker","report"))
    parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--task",type=int)
    args=parser.parse_args()
    if args.action=="worker":worker(args.plan,args.task)
    else:globals()[args.action](args.plan)
