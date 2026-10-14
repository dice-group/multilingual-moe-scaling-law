"""Create-only CPU fit plans, isolated workers, and explicit Slurm submission."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .. import execution as io
from . import inputs, backend, protocol
from ..artifacts import environment
from ..artifacts import copy_dependency_snapshot

SCHEMA = "stage1_paper_refit_execution/v1"
POOLS = ("grid", "native", "repetition", "shared")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def validate_profile(profile):
    if "full_paper_policy" in profile:
        protocol.contract(profile["full_paper_policy"])
    for key in ("account", "partition"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile[key]):
            raise ValueError(f"invalid Slurm {key}")
    if not re.fullmatch(r"[1-9][0-9]*[MG]", profile["memory"]):
        raise ValueError("memory must be an explicit positive M/G quantity")
    values = [profile["grid_chunk_size"], profile["starts_per_shard"], profile["max_array_size"],
              profile["max_concurrent_cores"], *profile["concurrency"].values(), *profile["minutes"].values()]
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("resource limits must be positive integers")
    if set(profile["concurrency"]) != set(POOLS) or set(profile["minutes"]) != {*POOLS, "certify", "report"}:
        raise ValueError("resource pools changed")
    if sum(profile["concurrency"].values()) > profile["max_concurrent_cores"]:
        raise ValueError("parallel arrays exceed the combined CPU limit")
    if not 0 < profile["max_reserved_core_hours"] < float("inf"):
        raise ValueError("a finite CPU reservation ceiling is required")


def tasks_for(models, cohorts, contracts, profile):
    if "full_paper_policy" in profile:
        return full_tasks(models, cohorts, profile)
    tasks, cell_tasks = [], {}

    def add(pool, model, fold, **kw):
        task = dict(id=len(tasks), stage="fit", pool=pool, model=model, fold=fold,
                    cell=f"{fold}--{model}", minutes=profile["minutes"][pool], **kw)
        tasks.append(task)
        cell_tasks.setdefault(task["cell"], []).append(task["id"])

    for model in models:
        for fold in inputs.FOLDS:
            if model in inputs.NATIVE:
                for language in cohorts[fold]:
                    add("native", model, fold, language=language)
            elif model == "RN14-stored-reuse":
                for begin in range(0, 48, profile["starts_per_shard"]):
                    add("repetition", model, fold, start_indices=list(range(begin, min(begin+profile["starts_per_shard"], 48))))
            elif fold == "all_grid":
                add("shared", model, fold)
            else:
                contract = contracts[f"{fold}--{model}"]
                count = len(contract["starts"]) if model == "M5-ND" else contract["fit"]["start_count"]
                for begin in range(0, count, profile["grid_chunk_size"]):
                    add("grid", model, fold, start_indices=list(range(begin, min(begin+profile["grid_chunk_size"], count))))
    for model in models:
        for fold in inputs.FOLDS:
            cell = f"{fold}--{model}"
            tasks.append(dict(id=len(tasks), stage="certify", pool="certify", model=model, fold=fold,
                              cell=cell, dependencies=cell_tasks[cell], minutes=profile["minutes"]["certify"]))
    return tasks



def full_tasks(models, cohorts, profile):
    tasks, groups = [], []
    for model in models:
        arms = [profile["full_paper_policy"]]
        if model == "RN14-stored-reuse" and profile.get("paired_controls", True):
            arms += [a for a in protocol.POLICIES if a not in arms]
        for fold in protocol.folds(model):
            for arm in arms:
                cell = f"{fold}--{model}--{arm}"
                dependencies = []
                native = model in protocol.NATIVE
                units = [dict(language=l) for l in cohorts[fold]] if native else [
                    dict(start_indices=list(range(i, min(i+profile["starts_per_shard"], protocol.STARTS))))
                    for i in range(0, protocol.STARTS, profile["starts_per_shard"])]
                pool = "native" if native else "repetition"
                for unit in units:
                    dependencies.append(len(tasks))
                    tasks.append(dict(id=len(tasks), stage="fit", pool=pool, model=model, fold=fold,
                                      cell=cell, arm=arm, minutes=profile["minutes"][pool], **unit))
                groups.append(dict(stage="certify", pool="certify", model=model, fold=fold,
                                   cell=cell, arm=arm, dependencies=dependencies, minutes=profile["minutes"]["certify"]))
    for group in groups:
        tasks.append(dict(id=len(tasks), **group))
    return tasks

def arrays_for(tasks, profile):
    arrays = []
    for pool in (*POOLS, "certify"):
        ids = [t["id"] for t in tasks if t["pool"] == pool]
        if not ids:
            continue
        if len(ids) > profile["max_array_size"]:
            raise ValueError(f"{pool} exceeds the configured Slurm array limit")
        arrays.append(dict(name=pool, tasks=ids, minutes=profile["minutes"][pool],
                           concurrency=min(len(ids), profile["concurrency"].get(pool, profile["max_concurrent_cores"]))))
    return arrays


def script(path, plan, name):
    root, profile = Path(path).resolve().parent, plan["profile"]
    array = next((a for a in plan["arrays"] if a["name"] == name), None)
    lines = ["#!/bin/bash", f"#SBATCH --job-name=paper-refit-{name}",
             f"#SBATCH --account={profile['account']}", f"#SBATCH --partition={profile['partition']}",
             "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1",
             f"#SBATCH --mem={profile['memory']}",
             f"#SBATCH --time={array['minutes'] if array else profile['minutes']['report']}",
             f"#SBATCH --output={shlex.quote(str(root / 'logs/attempts' / plan['sha256'][:12] / (name+'-%A_%a.out')))}"]
    if array:
        lines.append(f"#SBATCH --array=0-{len(array['tasks'])-1}%{array['concurrency']}")
    lines += ["set -euo pipefail", f"cd {shlex.quote(str(root))}",
              f"export PYTHONPATH={shlex.quote(str(root / 'source'))}", "export PYTHONDONTWRITEBYTECODE=1",
              "export "+" ".join(f"{key}=1" for key in THREADS)]
    command = [profile["python_executable"], "-m", "paper_fit_snapshot.paper_refit",
               "worker" if array else "report", "--plan", str(Path(path).resolve())]
    lines.append(shlex.join(command + ["--pool", name])+' --index "$SLURM_ARRAY_TASK_ID"' if array else shlex.join(command))
    return "\n".join([*lines, ""])


def prepare(profile_path, *, models=None, output=None):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    validate_profile(profile)
    if Path(profile["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("prepare with the configured bridge-train interpreter")
    resolve = lambda key: (profile_path.parent / profile[key]).resolve()
    source_input = resolve("inputs")
    root = Path(output).resolve() if output else resolve("output_root")
    if root.exists() or root.is_relative_to(source_input.parent) or source_input.is_relative_to(root):
        raise ValueError("execution requires a new directory separate from input evidence")
    allowed_models = (*protocol.MODELS, *protocol.ADDITIONAL_ABLATIONS) if "full_paper_policy" in profile else inputs.MODELS
    models = list(protocol.MODELS if "full_paper_policy" in profile else inputs.MODELS) if models is None else list(models)
    if not models or len(set(models)) != len(models) or set(models)-set(allowed_models):
        raise ValueError("choose unique models from the fixed main-paper panel")
    models = [n for n in allowed_models if n in models]
    inputs.verify_bundle(source_input)
    manifest = inputs.load(source_input)
    if "full_paper_policy" in profile and manifest.get("scope") != protocol.VERSION:
        raise ValueError("full paper requires the full input bundle")
    contracts, provenance = {}, [inputs.binding(profile_path), inputs.binding(source_input)]
    if "full_paper_policy" not in profile and set(models) & {"J0", "M2", "M5-ND"}:
        contracts, sources = backend.import_contracts(resolve("primary_baseline_comparison"), resolve("primary_ablation_plan"), source_input)
        provenance.extend(sources)
    cohorts = {f: inputs.read_training(source_input, f)[0].language.drop_duplicates().sort_values().tolist() for f in manifest["training"]}
    tasks = tasks_for(models, cohorts, contracts, profile)
    arrays = arrays_for(tasks, profile)
    hours = (sum(t["minutes"] for t in tasks)+profile["minutes"]["report"])/60
    if hours > profile["max_reserved_core_hours"]:
        raise ValueError(f"reservation {hours:g} core-hours exceeds the explicit profile ceiling")
    root.mkdir(parents=True)
    shutil.copytree(source_input.parent, root / "inputs")
    snapshot = root / "source/paper_fit_snapshot"
    copy_dependency_snapshot(snapshot, ("paper_refit",))
    io.write_json_once(root / "provenance.json", dict(inputs=provenance, calibration="reused training-only values and start anchors"))
    reference = None
    if profile.get("reference_metrics"):
        shutil.copyfile(resolve("reference_metrics"), root / "reference_metrics.csv")
        reference = inputs.local_binding(root / "reference_metrics.csv", root)
    reference_predictions = None
    if profile.get("reference_predictions"):
        shutil.copyfile(resolve("reference_predictions"), root / "reference_predictions.parquet")
        reference_predictions = inputs.local_binding(root / "reference_predictions.parquet", root)
    reference_documents = []
    for name, source in profile.get("reference_documents", {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid reference document name")
        destination = root / "reference_documents" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile((profile_path.parent / source).resolve(), destination)
        reference_documents.append(inputs.local_binding(destination, root))
    plan = inputs.sealed(dict(schema=SCHEMA, profile=profile, models=models, cohorts=cohorts,
        contracts=contracts, numerical_contract=protocol.contract(profile["full_paper_policy"]) if "full_paper_policy" in profile else None, tasks=tasks, arrays=arrays, reserved_core_hours=hours, environment=environment(),
        input=inputs.local_binding(root / "inputs/manifest.json", root),
        source=[inputs.local_binding(p, root) for p in sorted(snapshot.rglob("*.py"))],
        provenance=inputs.local_binding(root / "provenance.json", root), reference=reference,
        reference_predictions=reference_predictions, reference_documents=reference_documents,
        report_scope="three configuration-balanced holdouts and separate equal-language L26; fixed laws, no reselection"))
    path = root / "plan.json"
    io.write_json_once(path, plan)
    (root / "logs/attempts" / plan["sha256"][:12]).mkdir(parents=True)
    for name in [a["name"] for a in arrays]+["report"]:
        target = root / f"{name}.sbatch"
        io.write_text_once(target, script(path, plan, name))
        subprocess.run(["bash", "-n", str(target)], check=True, timeout=10)
    return dict(plan=str(path), models=models, tasks=len(tasks), fits=sum(t["stage"] == "certify" for t in tasks),
                arrays={a["name"]: len(a["tasks"]) for a in arrays}, reserved_core_hours=hours,
                max_concurrent_cores=sum(a["concurrency"] for a in arrays if a["name"] != "certify"), submitted=False)


def load(path):
    path = Path(path).resolve()
    plan = inputs.read_sealed(path)
    if plan["schema"] != SCHEMA:
        raise ValueError("wrong execution plan schema")
    validate_profile(plan["profile"])
    if "full_paper_policy" in plan["profile"] and plan.get("numerical_contract") != protocol.contract(plan["profile"]["full_paper_policy"]):
        raise ValueError("numerical policy changed")
    if plan["tasks"] != tasks_for(plan["models"], plan["cohorts"], plan["contracts"], plan["profile"]):
        raise ValueError("fit task inventory changed")
    if plan["arrays"] != arrays_for(plan["tasks"], plan["profile"]):
        raise ValueError("array inventory changed")
    hours = (sum(t["minutes"] for t in plan["tasks"])+plan["profile"]["minutes"]["report"])/60
    if hours != plan["reserved_core_hours"] or hours > plan["profile"]["max_reserved_core_hours"]:
        raise ValueError("CPU reservation changed")
    for entry in [plan["input"], *plan["source"], *plan.get("reference_documents", [])]:
        inputs.local_path(path.parent, entry)
    return plan


def receipt(path, plan, index):
    result = inputs.read_sealed(Path(path).resolve().parent / "results" / f"{index:05d}.json")
    if result["plan_sha256"] != plan["sha256"] or result["task"] != plan["tasks"][index]:
        raise ValueError("foreign or malformed task receipt")
    return result


def cpu_guard(path, plan):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("real fitting requires a one-core Slurm CPU allocation")
    if (any(os.environ.get(k) not in (None, "", "0") for k in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK"))
            or any(os.environ.get(k) not in (None, "") for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))):
        raise ValueError("GPU allocations are not allowed")
    if any(os.environ.get(k) != "1" for k in THREADS):
        raise ValueError("numerical libraries must use one thread")
    if environment() != plan["environment"] or Path(sys.executable).resolve() != Path(plan["profile"]["python_executable"]).resolve():
        raise ValueError("numerical runtime changed")
    if Path(__file__).resolve().parent != Path(path).resolve().parent / "source/paper_fit_snapshot/paper_refit_workflow":
        raise ValueError("workers must execute the prepared source snapshot")


def worker(path, pool, index):
    path = Path(path).resolve()
    plan = load(path)
    cpu_guard(path, plan)
    array, = [a for a in plan["arrays"] if a["name"] == pool]
    if not 0 <= index < len(array["tasks"]):
        raise ValueError("array index is out of range")
    task = plan["tasks"][array["tasks"][index]]
    sink = path.parent / "attempts" / f"{task['id']:05d}"
    sink.mkdir(parents=True, exist_ok=False)
    io.write_json_once(sink / "claim.json", dict(plan_sha256=plan["sha256"], task=task, job_id=os.environ["SLURM_JOB_ID"]))
    began = time.monotonic()
    seconds = max(1, task["minutes"]*60-30)

    def expired(signum, frame):
        raise TimeoutError("task wall-time allowance exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        shards = [(plan["tasks"][i], receipt(path, plan, i)) for i in task.get("dependencies", [])]
        result = backend.execute(task, inputs.local_path(path.parent, plan["input"]),
                                 plan["contracts"].get(task["cell"]), began+seconds, sink, shards)
    except Exception as error:
        result = dict(certified=False, complete=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    result = result | dict(plan_sha256=plan["sha256"], task=task, heldout_scores_computed=False,
                          elapsed_seconds=time.monotonic()-began, job_id=os.environ["SLURM_JOB_ID"])
    if "language" in task:
        result["language"] = task["language"]
    io.write_json_once(path.parent / "results" / f"{task['id']:05d}.json", inputs.sealed(result))
    return {k: result.get(k) for k in ("certified", "complete", "error", "elapsed_seconds")}


def submit(path, *, dry_run=False, test_only=False):
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("submit on the login host, not from a nested allocation")
    path = Path(path).resolve()
    plan = load(path)
    if environment() != plan["environment"] or Path(sys.executable).resolve() != Path(plan["profile"]["python_executable"]).resolve():
        raise ValueError("submit with the prepared numerical runtime")
    inputs.verify_bundle(inputs.local_path(path.parent, plan["input"]))
    names = [a["name"] for a in plan["arrays"]]+["report"]
    for name in names:
        target = path.parent / f"{name}.sbatch"
        if target.read_text() != script(path, plan, name):
            raise ValueError("prepared CPU script changed")
        subprocess.run(["bash", "-n", str(target)], check=True, timeout=10)
    if dry_run:
        return dict(submitted=False, arrays=[{k: v for k, v in a.items() if k != "tasks"} | {"tasks": len(a["tasks"])}
                                            for a in plan["arrays"]], reserved_core_hours=plan["reserved_core_hours"])
    # Inherited SBATCH_* options must not add GPUs or override resource limits.
    scheduler_env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    if test_only:
        for name in names:
            subprocess.run(["sbatch", "--test-only", str(path.parent / f"{name}.sbatch")], check=True, timeout=30, env=scheduler_env)
        return dict(submitted=False, scheduler_checks=len(names))
    submission = path.parent / "submission"
    submission.mkdir(exist_ok=False)
    jobs = {}
    for name in names:
        command = ["sbatch", "--parsable"]
        dependencies = list(jobs.values()) if name == "certify" else [jobs["certify"]] if name == "report" else []
        if dependencies:
            command += ["--dependency=afterany:"+":".join(dependencies)]
        command += [str(path.parent / f"{name}.sbatch")]
        io.write_json_once(submission / f"{name}-intent.json", dict(plan_sha256=plan["sha256"], command=command))
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30, env=scheduler_env)
        job_id = result.stdout.strip().split(";")[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("ambiguous scheduler response; reconcile the saved intent before retrying")
        jobs[name] = job_id
        io.write_json_once(submission / f"{name}.json", dict(plan_sha256=plan["sha256"], job_id=job_id))
    return dict(submitted=True, jobs=jobs)


def status(path, *, scheduler=False):
    path = Path(path).resolve()
    plan = load(path)
    counts = dict(pending=0, claimed_without_receipt=0, complete=0, incomplete=0, certified_cells=0, uncertified_cells=0)
    for task in plan["tasks"]:
        if (path.parent / "results" / f"{task['id']:05d}.json").exists():
            result = receipt(path, plan, task["id"])
            counts["complete" if result.get("complete") else "incomplete"] += 1
            if task["stage"] == "certify":
                counts["certified_cells" if result.get("certified") else "uncertified_cells"] += 1
        elif (path.parent / "attempts" / f"{task['id']:05d}").exists():
            counts["claimed_without_receipt"] += 1
        else:
            counts["pending"] += 1
    jobs = {}
    for item in (path.parent / "submission").glob("*.json"):
        value = io.read_json(item)
        if value["plan_sha256"] != plan["sha256"]:
            raise ValueError("submission receipt belongs to another plan")
        if "job_id" in value:
            jobs[item.stem] = value["job_id"]
    result = dict(tasks=len(plan["tasks"]), fits=sum(t["stage"] == "certify" for t in plan["tasks"]), counts=counts, jobs=jobs,
                  scope="artifact state, not scheduler liveness", report_exists=(path.parent / "report/report.json").exists())
    if scheduler and jobs:
        query = subprocess.run(["sacct", "-n", "-P", "-j", ",".join(jobs.values()),
                                "--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS"], capture_output=True, text=True, timeout=20)
        result["scheduler"] = dict(returncode=query.returncode, output=query.stdout, error=query.stderr)
    return result
