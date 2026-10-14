"""Bounded CPU workers. Imports never open scoring tables or choose a law."""
from __future__ import annotations

from ..shared_numerics import best_attempt
from ..shared_numerics import agreement
from ..shared_numerics import geometry
from ..shared_numerics import fit_cell

from dataclasses import asdict
from dataclasses import replace
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from .. import execution as io
from .. import fit
from . import data
from . import model as m
from ..atlas_comparison_workflow import model as native
from ..atlas_comparison_workflow import numerics
from ..data import canonical_json_sha256
from ..artifacts import binding
from ..artifacts import verify


def calibration(rows,sources,packets,deadline,sink):
    placeholder = fit.M5M6Regularization(0.,1.).as_dict()
    m0 = data.make_model("M0",rows,sources,packets,placeholder)
    ols = fit_cell(m0,None,deadline,sink=sink/"M0")
    io.write_json_once(sink/"M0.json",ols)
    if not ols["certified"]:
        raise ValueError("training-only M0 calibration is not certified")
    scale = fit.adaptive_huber_scale(m.predict(m0,ols["selected"])-m0.problem.y,m0.problem.weights)
    delta = scale.delta
    m2 = data.make_model("M2",rows,sources,packets,placeholder)
    shared = fit.shared_from_outer("M0",ols["selected"]["coordinates"])
    reference = fit_cell(m2,delta,deadline,anchor=fit.outer_from_shared("M2",shared),sink=sink/"M2")
    io.write_json_once(sink/"M2.json",reference)
    if not reference["certified"]:
        raise ValueError("training-only M2 reference is not certified")
    regularization = fit.M5M6Regularization(.01,reference["selected"]["objective"]).as_dict()
    parent = data.make_model("M6-ND",rows,sources,packets,regularization)
    shared = fit.shared_from_outer("M2",reference["selected"]["coordinates"])
    anchor = np.r_[fit.outer_from_shared("M6",shared),.05]
    parent_fit = fit_cell(parent,delta,deadline,anchor=anchor,sink=sink/"parent")
    io.write_json_once(sink/"parent.json",parent_fit)
    if not parent_fit["certified"]:
        raise ValueError("training-only M6-ND parent is not certified")
    return dict(certified=True,delta=delta,regularization=regularization,
                anchor=parent_fit["selected"]["coordinates"],scale=asdict(scale),
                calibration_inputs="assigned training responses only",heldout_scores_computed=False)


def bootstrap_rows(rows,sources,packets,index):
    """Frequency weights resample entire trajectories, including all their readouts."""
    rng = np.random.default_rng(20260913+index)
    frequencies = {}
    configs = rows.drop_duplicates("trajectory_id")
    for _,group in configs.groupby(["E","K"],sort=True):
        ids = group.trajectory_id.to_numpy()
        chosen = rng.choice(ids,size=len(ids),replace=True)
        frequencies.update({t:int(np.sum(chosen==t)) for t in ids})
    # Preserve scientific identities and all nuisance support, assigning zero
    # multiplicities by removing absent configurations before re-profiling.
    keep = rows.trajectory_id.map(frequencies).gt(0).to_numpy()
    result = rows[keep].reset_index(drop=True)
    new_sources = replace(sources,readout=sources.readout[keep],target=sources.target[keep])
    old_to_new = {old:new for new,old in enumerate(np.flatnonzero(keep))}
    updated = []
    for item in packets:
        mask = np.array([keep[i] for i in item["indices"]])
        if not mask.any():
            raise ValueError("bootstrap lost a target language; no silently changing cohort")
        p = dict(item["train"])
        for key in ("D","U","N","N_total","E","K","family_share","trajectory_ids","readout_ids","y","weights"):
            p[key] = np.asarray(p[key])[mask].tolist()
        p["training_minimum"] = float(min(p["y"]))
        updated.append(item | {"train":p,"indices":[old_to_new[i] for i,mk in zip(item["indices"],mask) if mk]})
    return result,new_sources,updated,frequencies


def execute(task, plan, root, deadline):
    rows,sources,packets = data.read_training(plan["training"][task["fold"]])
    sink = root/"attempts"/task["id"]
    if task["kind"] == "calibrate":
        return calibration(rows,sources,packets,deadline,sink)
    cal = plan["calibration"].get(task["fold"])
    dependencies = []
    if cal is None:
        path = root/"results"/f"cal-{task['fold']}.json"
        cal = io.read_json(path)
        if cal["plan_sha256"] != plan["sha256"] or not cal.get("certified"):
            raise ValueError("missing or uncertified training-only calibration")
        dependencies.append(binding(path))
    if task["kind"] == "native":
        spec = native.Spec(**task["spec"])
        groups = []
        # Native workers see training covariates only. Endpoint domain failures
        # remain visible later; never force positivity using endpoint responses.
        for item in packets:
            if time.monotonic() >= deadline:
                break
            p = dict(item["train"]); p["delta"] = cal["delta"]
            try:
                result = numerics.fit_group(spec,p,p)
            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as error:
                result = dict(certified=False,error=str(error))
            result.update(language=item["language"],donors=item["donors"],
                          unigram_nll=item["unigram_nll"],training_minimum=p["training_minimum"])
            io.write_json_once(sink/f"{item['language']}.json",result)
            groups.append(result)
        return dict(model=spec.name,groups=groups,expected_languages=len(packets),
                    certified=len(groups)==len(packets) and all(g["certified"] for g in groups),dependencies=dependencies)
    name,tau = task["model"],task.get("tau",0.)
    if name == "selected":
        path = root/"selection.json"
        selected = io.read_json(path)
        if selected["plan_sha256"] != plan["sha256"] or not selected["frozen_before_outer_scoring"]:
            raise ValueError("selection receipt does not match this campaign")
        name,tau = selected["chosen"]["model"],selected["chosen"]["tau"]
        dependencies.append(binding(path))
    frequencies = None
    if task["kind"] == "bootstrap":
        rows,sources,packets,frequencies = bootstrap_rows(rows,sources,packets,task["replicate"])
    candidate = data.make_model(name,rows,sources,packets,cal["regularization"],tau,cal["delta"])
    anchor = None
    if name in m.CANDIDATES:
        anchor = np.r_[cal["anchor"],.1,0.]
    elif name in ("M6-ND","M5-ND","J0-ND","M6"):
        anchor = cal["anchor"]
    if task["kind"] in ("rho_profile","bootstrap","no_EK"):
        path = root/"results"/"selected-all_grid.json"
        parent = io.read_json(path)
        if not parent.get("certified") or parent["plan_sha256"] != plan["sha256"]:
            raise ValueError("stability needs a certified, frozen all-grid candidate")
        anchor = parent["selected"]["coordinates"]
        dependencies.append(binding(path))
    fixed = dict(task.get("fixed",{}))
    if task["kind"] == "rho_profile":
        fixed["rho"] = task["rho"]
    if task["kind"] == "no_EK":
        # Refit common alpha/beta; remove E main effects, K slopes and K-size hinge.
        fixed.update({candidate.bounds[i].name:0. for i in range(4,11)})
        candidate.equalities=((0,1),(2,3))
        anchor=list(anchor);anchor[1]=anchor[0];anchor[3]=anchor[2]
    for i,b in enumerate(candidate.bounds):
        if b.name in fixed:
            value = float(fixed[b.name])
            if not b.lower <= value <= b.upper:
                raise ValueError("diagnostic fixed coordinate exceeds registered bounds")
            candidate.bounds[i] = replace(b,lower=value,upper=value)
    if frequencies is not None:
        weights = candidate.problem.weights*rows.trajectory_id.map(frequencies).to_numpy()
        candidate.problem = replace(candidate.problem,weights=weights/weights.sum())
    result = fit_cell(candidate,cal["delta"],deadline,anchor=anchor,sink=sink)
    result.update(delta=cal["delta"],regularization=cal["regularization"],dependencies=dependencies,
                  bootstrap_frequencies=frequencies,
                  bootstrap_scope="conditional on frozen procedure, delta and ridge reference; training minima refitted",
                  native_metadata=[{k:item[k] for k in ("language","donors","unigram_nll")} |
                      {"training_minimum":item["train"]["training_minimum"]} for item in packets])
    return result


def worker(plan_path, task_id):
    from .state import load_plan
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK","0")) != 1:
        raise ValueError("real fitting requires its single-core CPU Slurm task")
    if any(os.environ.get(k) not in (None,"","0") for k in ("SLURM_GPUS","SLURM_GPUS_ON_NODE","SLURM_JOB_GPUS")):
        raise ValueError("GPU allocation is not authorized for this campaign")
    plan = load_plan(plan_path)  # Source/runtime only: never hashes a heldout response file.
    task = next(t for t in plan["tasks"] if t["id"] == task_id)
    root = Path(plan_path).resolve().parent
    destination = root/"results"/f"{task_id}.json"
    if destination.exists() or (root/"attempts"/task_id).exists():
        raise ValueError("task already attempted; immutable receipts, no implicit retries")
    began = time.monotonic()
    def expired(signum,frame):
        raise TimeoutError("registered CPU wall-time budget reached")
    signal.signal(signal.SIGALRM,expired)
    seconds = max(1,int(task["minutes"]*60-30))
    signal.alarm(seconds)
    try:
        result = execute(task,plan,root,began+seconds-5)
    except (ValueError,RuntimeError,FloatingPointError,TimeoutError,OverflowError) as error:
        result = dict(certified=False,error=str(error),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
    result.update(task_id=task_id,fold=task["fold"],plan_sha256=plan["sha256"],
                  task_sha256=canonical_json_sha256(task),training=plan["training"][task["fold"]],
                  slurm_job_id=os.environ["SLURM_JOB_ID"],elapsed_seconds=time.monotonic()-began,
                  heldout_scores_computed=False)
    io.write_json_once(destination,result)
    print(f"{task_id}: certified={result['certified']} elapsed={result['elapsed_seconds']:.1f}s",flush=True)
