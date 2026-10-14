"""Create-only preparation, immutable CPU workers and one submission command."""
from .state import seal, cpu_guard, load, result_path, read_result, make_model
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

from .. import execution as io
from ..law_search import data
from ..data import canonical_json_sha256
from ..law_search.state import load_plan as load_parent
from ..artifacts import binding, environment, validate_output, verify
from ..artifacts import copy_dependency_snapshot
from .contracts import STAGES, cells, forms_for, plan_rules, reservation, rules_for, tasks, validate_profile
from .numerics import certify, collect_attempts, start_shard


def prepare(profile_path):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    rules = rules_for(profile)
    validate_profile(profile, rules)
    root = validate_output(profile)
    if root.exists():
        raise ValueError("create-only: campaign output already exists")
    if Path(sys.executable).resolve() != Path(profile["python_executable"]).resolve():
        raise ValueError("use the configured bridge-train interpreter")
    parent_path = Path(profile["parent_plan"]).resolve()
    parent = load_parent(parent_path, all_inputs=True)
    old = parent_path.parent
    inputs = [binding(profile_path), binding(parent_path)]
    inputs.extend(binding(path) for path in profile.get("reference_sources", []))
    calibrations = {}
    for fold in rules["folds"]:
        if fold in parent["calibration"]:
            calibrations[fold] = parent["calibration"][fold]
        else:
            path = old / "results" / f"cal-{fold}.json"
            receipt = io.read_json(path)
            if not receipt.get("certified") or receipt["plan_sha256"] != parent["sha256"]:
                raise ValueError("uncertified or foreign fold-local calibration")
            calibrations[fold] = {k: receipt[k] for k in ("anchor", "delta", "regularization")}
            inputs.append(binding(path))
    # Baseline table is an immutable reporting input, not a candidate selector.
    baseline = binding(profile["baseline_metrics"])
    from .reporting import complete_baseline_panel, prior_campaign_panel
    import pandas as pd
    original_baselines = pd.read_csv(verify(baseline))
    panel = complete_baseline_panel(original_baselines)
    prior, prior_cases = None, 0
    if profile.get("prior_campaign_plan"):
        prior_path = Path(profile["prior_campaign_plan"]).resolve()
        old_plan = io.read_json(prior_path)
        unsigned = {k: v for k, v in old_plan.items() if k != "sha256"}
        if canonical_json_sha256(unsigned) != old_plan["sha256"] or old_plan["parent_sha256"] != parent["sha256"]:
            raise ValueError("modified or incomparable prior campaign")
        prior_report = prior_path.parent / "report" / "report.json"
        if io.read_json(prior_report)["plan_sha256"] != old_plan["sha256"]:
            raise ValueError("foreign prior campaign report")
        prior = dict(plan=binding(prior_path), report=binding(prior_report),
                     metrics=binding(prior_path.parent / "report" / "metrics.csv"),
                     forms=forms_for(plan_rules(old_plan)), prefix="round2::")
        inputs.extend(prior[key] for key in ("plan", "report", "metrics"))
        previous = prior_campaign_panel(prior)
        prior_cases = len(previous)
        panel = pd.concat([panel, previous], ignore_index=True)
    saved_cases = len(original_baselines)+prior_cases
    baseline_coverage = dict(models=int(panel.model.nunique()), saved_cases=saved_cases,
                            absent_cases=len(panel)-saved_cases, registered_cases=len(panel))
    scoring = {p.name: binding(p) for p in (old / "scoring").iterdir() if p.is_file()}
    root.mkdir(parents=True)
    source = root / "source" / "stage1_fit"
    copy_dependency_snapshot(source, ("coverage_laws.__main__", "coverage_laws.recovery"))
    for directory in ("results", "attempts", "submissions", *(f"logs/attempts/{s}" for s in STAGES)):
        (root / directory).mkdir(parents=True)
    payload = dict(rules=rules, profile=profile, tasks=tasks(profile, rules), cells=cells(rules),
                   parent_sha256=parent["sha256"], calibrations=calibrations,
                   training={f: parent["training"][f] for f in rules["folds"]},
                   inner_scoring=parent["inner_scoring"], memberships=parent["memberships"],
                   scoring=scoring, baseline_metrics=baseline, baseline_coverage=baseline_coverage, inputs=inputs,
                   environment=environment(), reserved_core_hours=reservation(profile, rules),
                   source=[binding(p) for p in sorted(source.rglob("*.py"))])
    if prior is not None:
        payload["prior_campaign"] = prior
    seal(root / "plan.json", payload)
    plan = io.read_json(root / "plan.json")
    for stage in STAGES:
        rendered = script(root / "plan.json", plan, stage)
        subprocess.run(["bash", "-n"], input=rendered, text=True, check=True)
        io.write_text_once(root / "submissions" / f"{stage}.sbatch", rendered)
    for action in ("submit", "status", "verify"):
        command = [profile["python_executable"], "-m", "stage1_fit.coverage_laws", action,
                   "--plan", str(root / "plan.json")]
        wrapper = "\n".join(["#!/bin/bash", "set -euo pipefail",
                  f"cd {shlex.quote(profile['workspace_root'])}",
                  f"export PYTHONPATH={shlex.quote(str(root / 'source'))}",
                  "export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1",
                  "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
                  shlex.join(command)+' "$@"', ""])
        io.write_text_once(root / f"{action}.sh", wrapper)
    print(dict(plan=str(root / "plan.json"), starts=len(plan["tasks"])-len(cells(rules)),
               fit_cells=len(cells(rules)), max_cores=profile["concurrency"],
               reserved_core_hours=plan["reserved_core_hours"], submitted=False), flush=True)


def worker(path, task_id, stage):
    cpu_guard()
    plan, root = load(path), Path(path).resolve().parent
    task = next(t for t in plan["tasks"] if t["id"] == task_id and t["stage"] == stage)
    sink = root / "attempts" / f"{stage}--{task_id}"
    if result_path(root, task).exists():
        raise ValueError("immutable result already exists")
    sink.mkdir()  # atomic claim; interruption is not permission to overwrite
    began = time.monotonic()
    dependencies = []
    def expired(signum, frame):
        raise TimeoutError("registered CPU wall-time cap")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(task["minutes"]*60-30)
    try:
        model, rows, calibration = make_model(plan, task["fold"], task["form"])
        deadline = began+task["minutes"]*60-45
        if stage == "starts":
            value = start_shard(model, rows, calibration, task["start_indices"], deadline, sink)
        else:
            shards = [(t, read_result(root, plan, t)) for t in plan["tasks"]
                      if t["stage"] == "starts" and t["cell"] == task["cell"]]
            dependencies = [binding(result_path(root, t)) for t, _ in shards]
            value = certify(model, rows, calibration, collect_attempts(shards), deadline, sink)
    except Exception as error:
        value = dict(complete=False, certified=False, error=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
    value.update(plan_sha256=plan["sha256"], task_sha256=canonical_json_sha256(task),
                 task_id=task_id, stage=stage, form=task["form"], fold=task["fold"],
                 training=plan["training"][task["fold"]], dependencies=dependencies,
                 slurm_job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic()-began,
                 heldout_scores_computed=False)
    io.write_json_once(result_path(root, task), value)
    print(dict(task=task_id, complete=value["complete"], certified=value.get("certified")), flush=True)
    return bool(value.get("complete") and (stage == "starts" or value.get("certified")))


def script(path, plan, stage):
    if stage not in STAGES:
        raise ValueError("unregistered stage")
    root, p = Path(path).resolve().parent, plan["profile"]
    group = [t for t in plan["tasks"] if t["stage"] == stage]
    minutes = group[0]["minutes"] if group else p["analysis_minutes"]
    command = [p["python_executable"], "-m", "stage1_fit.coverage_laws",
               "worker" if group else stage, "--plan", str(Path(path).resolve())]
    lines = ["#!/bin/bash", f"#SBATCH --job-name=coverage-{stage}",
             f"#SBATCH --account={p['account']}", f"#SBATCH --partition={p['partition']}",
             "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1", "#SBATCH --no-requeue",
             f"#SBATCH --mem={p['memory']}", f"#SBATCH --time={minutes}",
             f"#SBATCH --output={shlex.quote(str(root / 'logs/attempts' / stage / '%A_%a.out'))}"]
    if group:
        lines.append(f"#SBATCH --array=0-{len(group)-1}%{min(len(group), p['concurrency'])}")
    lines += ["set -euo pipefail", f"cd {shlex.quote(p['workspace_root'])}",
              f"export PYTHONPATH={shlex.quote(str(root / 'source'))}",
              "export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1",
              "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"]
    if group:
        lines += ["tasks=("+" ".join(shlex.quote(t["id"]) for t in group)+")",
                  shlex.join(command)+f' --stage {stage} --task "${{tasks[$SLURM_ARRAY_TASK_ID]}}"']
    else:
        lines.append(shlex.join(command))
    return "\n".join([*lines, ""])


def submit(path, test_only=False):
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("no nested Slurm submission")
    plan, root = load(path, all_inputs=True), Path(path).resolve().parent
    rendered = {stage: script(path, plan, stage) for stage in STAGES}
    for stage, value in rendered.items():
        if (root / "submissions" / f"{stage}.sbatch").read_text() != value:
            raise ValueError("prepared submission script changed")
        subprocess.run(["bash", "-n"], input=value, text=True, check=True)
    if test_only:
        for stage in STAGES:
            result = subprocess.run(["sbatch", "--test-only", str(root / "submissions" / f"{stage}.sbatch")],
                                    text=True, capture_output=True, check=True, timeout=45)
            print(stage, result.stdout+result.stderr, flush=True)
        return
    if any((root / "results").iterdir()):
        raise ValueError("results already exist; reconcile rather than resubmit")
    claim = root / "submissions" / "claim"
    claim.mkdir()  # an ambiguous response leaves an auditable intent and blocks retries
    previous = None
    for stage in STAGES:
        script_path = root / "submissions" / f"{stage}.sbatch"
        intent = dict(plan_sha256=plan["sha256"], stage=stage, script=binding(script_path),
                      dependency=previous, reserved_core_hours=plan["reserved_core_hours"])
        io.write_json_once(claim / f"{stage}-intent.json", intent)
        command = ["sbatch", "--parsable"]
        if previous:
            command.append(f"--dependency=afterany:{previous}")
        response = subprocess.check_output([*command, str(script_path)], text=True, timeout=45)
        job = response.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("ambiguous scheduler response; reconcile recorded intent")
        io.write_json_once(root / "submissions" / f"{stage}.json", intent | {"job_id": job})
        previous = job
        print(dict(stage=stage, job_id=job), flush=True)


def status(path):
    plan, root = load(path), Path(path).resolve().parent
    values = [(t, read_result(root, plan, t)) for t in plan["tasks"] if result_path(root, t).exists()]
    submissions = [io.read_json(p) for p in sorted((root / "submissions").glob("*.json"))]
    return dict(plan_sha256=plan["sha256"], planned_start_shards=sum(t["stage"] == "starts" for t in plan["tasks"]),
                finished_start_shards=sum(t["stage"] == "starts" for t, _ in values),
                planned_fit_cells=len(cells(plan_rules(plan))), certified_fit_cells=sum(t["stage"] == "certify" and
                bool(r.get("certified")) for t, r in values),
                failed=[t["id"] for t, r in values if not r.get("complete") or
                        (t["stage"] == "certify" and not r.get("certified"))],
                submissions=submissions, selection=(root / "selection.json").exists(),
                report=(root / "report/report.json").exists(), max_concurrent_cores=plan["profile"]["concurrency"],
                reserved_core_hours=plan["reserved_core_hours"], gpu_training=False)
