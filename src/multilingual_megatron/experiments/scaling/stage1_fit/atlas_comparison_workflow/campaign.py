"""Immutable packets and bounded Slurm execution; workers never read test losses."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy

from .. import atlas_full_pilot as io, extrapolation_diagnostics as diagnostic
from ..atlas_exposure import exposure_components
from ..problems import counts
from ..execution import trajectory_weights, write_text_once
from ..m6_composition import load_language_metadata
from ..m6_size_duration_workflow.reporting import candidate_rows, metrics_and_bootstrap
from . import model as m, numerics as n, reuse, snapshot, array_execution as arrays

MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison"
FOLDS = ("largest_N_act", "E32", "K200_common_languages")
PANEL = tuple(spec for spec in m.panel() if spec.form != "moe")
DEFERRED = tuple(spec for spec in m.panel() if spec.form == "moe")
PROTOCOL = {
    "version": "atlas_comparison/v8_parallel", "folds": list(FOLDS), "numerics": n.SETTINGS,
    "panel": [spec.payload() for spec in PANEL], "pooling": "all NEW forms per target; retained historical forms have shared nonlinear coefficients",
    "deferred_panel": [spec.payload() for spec in DEFERRED],
    "scope": "user-approved 53 native/control variants; 12 constructed MoE-ATLAS variants deferred, not defeated",
    "signed_domain": "D_eff/D_target >= 1e-10 on bound train and test COVARIATES; no test outcomes in worker packets",
    "data": "realized target/source exposures; exact original-document train-pool inventories before replicas, including EOD",
    "N_total": "configured design count, not runtime exact; same omitted small terms as declared active count",
    "loss": "raw NLL Huber with historical fold delta; normalized historical trajectory weights within target; no ridge",
    "family": "bound FineWeb2 family inventory; missing family is a disclosed singleton; per-language He-form adaptation",
    "selection": "minimum certified training objective, deterministic index tie; never heldout scores",
    "evidence": "development on already inspected holdouts; no independent confirmation or universal law claim",
    "source_mode": "isolated runtime and archive made from identical read-once source bytes; verify frozen files rather than unrelated live edits",
    "pilot_allocated_core_hours": 0., "main_allocated_core_hours": 1024*13/60,
    "finalize_allocated_core_hours": 2., "local_check_reserve_core_hours": 1.,
    "allocations": {"main": {"cores": 8, "minutes": 13, "jobs": 128, "parallel_jobs": 128},
                    "finalize": {"cores": 1, "minutes": 120}},
    "max_concurrent_cores": 1024, "maximum_total_core_hours": 250.,
    "task_partition": "disjoint frozen shards balanced by prior fitting runtime only; each task has one writer",
    "forecast_safety_factor": 2., "bootstrap_replicates": 10000, "bootstrap_seed": 20260906,
    "pilot_design": "original 63 native/control cases plus lexically first target per training K-support pattern on widest full/other settings and smallest training family share per fold for He; all selection uses covariates only",
    "amendment": "user explicitly requested high cluster parallelism; inherit every completed v7 fit including failures; 128 x 8 CPU shards, 13 minutes each; retain 250 total core-hour cap and all scientific settings",
}


def check_budget(prior_cpu_hours):
    allocated = sum(arrays.allocated_core_hours(value) for value in PROTOCOL["allocations"].values())
    if (prior_cpu_hours < 0 or not np.isfinite(prior_cpu_hours)
            or any(arrays.concurrent_cores(value) > PROTOCOL["max_concurrent_cores"] for value in PROTOCOL["allocations"].values())
            or allocated+prior_cpu_hours+PROTOCOL["local_check_reserve_core_hours"] > PROTOCOL["maximum_total_core_hours"]):
        raise ValueError("amended allocations exceed the authorized total CPU budget")


def environment():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)


def select(frame, membership, split):
    if set(membership["train_trajectory_ids"]) & set(membership["test_trajectory_ids"]):
        raise ValueError("overlapping trajectory split")
    rows = frame[frame.trajectory_id.isin(membership[f"{split}_trajectory_ids"])].copy()
    if membership[f"{split}_readout_ids"]:
        rows = rows[rows.readout_id.isin(membership[f"{split}_readout_ids"])]
    return rows


def covariates(rows, pivot, language, donors, inventories, metadata):
    """No losses, minima, weights, or optimizer settings enter this function."""
    source = pivot.loc[rows.readout_id]
    chosen = list(donors)
    if len(chosen) != 3:
        raise ValueError("full panel requires three training-selected donors")
    remainder = [key for key in pivot.columns if key not in [language, *chosen]]
    D = np.column_stack((rows.D_t, source[chosen], source[remainder].sum(axis=1)))
    U = np.column_stack((np.full(len(rows), inventories[language]),
        source[chosen].gt(0).to_numpy()*np.array([inventories[key] for key in chosen]),
        source[remainder].gt(0).to_numpy() @ np.array([inventories[key] for key in remainder])))
    family = metadata[language]["family"]
    members = [key for key in pivot.columns if key == language or
               (family is not None and metadata[key]["family"] == family)]
    share = source[members].sum(axis=1).to_numpy()/rows.D_total.to_numpy()
    if not np.allclose(D.sum(axis=1), rows.D_total, rtol=1e-12, atol=1e-6):
        raise ValueError("source exposures do not reconstruct D_total")
    return dict(D=D.tolist(), U=U.tolist(), N=rows.N_act.tolist(), N_total=rows.N_total.tolist(),
                E=rows.E.tolist(), K=rows.K.tolist(), family_share=share.tolist(),
                trajectory_ids=rows.trajectory_id.tolist(), readout_ids=rows.readout_id.tolist())


def packets_for(frame, membership, inventories, metadata, delta):
    """Cut the training split before ranking donors or deriving response settings."""
    train, test = select(frame, membership, "train"), select(frame, membership, "test")
    _, donors = exposure_components(train)
    train["weight"] = trajectory_weights(tuple(train.trajectory_id))
    pivot = frame.pivot(index="readout_id", columns="language", values="D_t").fillna(0.)
    selected_languages = set(membership["score_languages"] or test.language.unique())
    if not selected_languages <= set(train.language):
        raise ValueError("test target has no training support")
    result, observations = [], []
    for language in sorted(train.language.unique()):
        tr = train[train.language.eq(language)]
        te = test[test.language.eq(language)] if language in selected_languages else test.iloc[:0]
        features = covariates(tr, pivot, language, donors[language], inventories, metadata)
        heldout = covariates(te, pivot, language, donors[language], inventories, metadata)
        both = {key: features[key]+heldout[key] for key in features}
        unigram = float(tr.unigram_nll.iloc[0])
        if not np.allclose(tr.unigram_nll, unigram, rtol=0, atol=1e-12) or not np.allclose(te.unigram_nll, unigram, rtol=0, atol=1e-12):
            raise ValueError("language unigram offset is not constant")
        if not np.allclose(tr.raw_nll-tr.unigram_nll, tr.adjusted_loss, rtol=0, atol=1e-12):
            raise ValueError("raw and adjusted responses are inconsistent")
        raw = tr.raw_nll.to_numpy()
        if not np.isfinite(raw).all() or raw.min() <= 0:
            raise ValueError("invalid raw NLL training response")
        minimum = float(raw.min())
        features.update(y=raw.tolist(), weights=tr.weight.tolist(), delta=float(delta), training_minimum=minimum)
        heldout["training_minimum"] = minimum
        both["training_minimum"] = minimum
        result.append(dict(language=language, donors=list(donors[language]), unigram_nll=unigram,
            train=features, test=heldout, support=both, training_only_outcomes=True,
            family=metadata[language]["family"], missing_family_singleton=metadata[language]["family"] is None))
        observations.append(te)
    return result, pd.concat(observations, ignore_index=True)


def prepare(profile_path):
    profile = json.loads(Path(profile_path).read_text())
    root = Path(profile["output_root"]).resolve()
    if root.exists() or root.parent != Path.cwd()/"result_analysis" or not root.name.startswith("stage1_atlas_comparison_"):
        raise ValueError("comparison needs a new, separate result_analysis output root")
    parent = io.read(profile["parent_plan"])
    previous = io.read(profile["historical_report"])
    if not previous["numerical_audit_healthy"] or previous["plan_sha256"] != parent["sha256"]:
        raise ValueError("historical M6-ND result has no accepted numerical audit")
    inventories, bindings = io.inventories(profile["corpus_manifest"])
    bindings += [io.bind(profile_path), io.bind(profile["parent_plan"]), io.bind(profile["historical_report"])]
    prior_cpu_hours = 0.
    for prior in profile.get("prior_accounting", []):
        accounting = json.loads(Path(prior).read_text())
        if accounting["state"] != "COMPLETED" or accounting["gpu_allocations"] != 0:
            raise ValueError("prior accounting is incomplete or includes GPUs")
        prior_cpu_hours += accounting["allocated_cpu_seconds"]/3600
        bindings.append(io.bind(prior))
    check_budget(prior_cpu_hours)
    configs_path = Path(profile["model_configs"])
    bindings += [io.bind(configs_path), io.bind(profile["family_metadata"])]
    total_counts, count_evidence = counts(pd.read_parquet(configs_path))
    metadata = load_language_metadata(profile["family_metadata"], inventories)
    # Recompute the old predictions and metrics before reusing them. No new
    # law has been fitted or selected at this point.
    diag_profile, diag_paths = diagnostic.load_profile(Path(profile["diagnostic_profile"]))
    released_configs = diagnostic.release_table(diag_paths["release_manifest"], "model_configs")
    pd.testing.assert_frame_equal(pd.read_parquet(configs_path), released_configs, check_exact=True)
    old, reproduction, _ = diagnostic.collect_predictions(diag_profile, diag_paths)
    reduction_path = Path(profile["parent_plan"]).parent/"reduction.json"
    reduction = io.read(reduction_path)
    if reduction["plan_sha256"] != parent["sha256"]:
        raise ValueError("M6-ND reduction belongs to another plan")
    old = pd.concat((old, candidate_rows(parent, reduction)), ignore_index=True)
    metrics = metrics_and_bootstrap(old, replicates=100, seed=20260906)["metrics"]
    if max(abs(metrics[key][f]-previous["statistics"]["metrics"][key][f]) for key in metrics for f in (*FOLDS, "mean")) > 1e-12:
        raise ValueError("historical metric reproduction failed")
    bindings += [io.bind(profile["diagnostic_profile"]), io.bind(reduction_path), *[io.bind(p) for p in diag_paths.values()]]
    for bound in previous["outputs"]:
        io.verify(bound)
        if bound["path"].endswith("predictions.parquet"):
            stored = pd.read_parquet(bound["path"])
            keys = ["model", "fold", "split", "trajectory_id", "readout_id", "language"]
            a, b = old.sort_values(keys), stored.sort_values(keys)
            if list(map(tuple, a[keys].to_numpy())) != list(map(tuple, b[keys].to_numpy())) or not np.allclose(a.prediction, b.prediction, rtol=0, atol=1e-12):
                raise ValueError("historical prediction reconstruction failed")
            bindings.append(bound)
    packets, observations = [], []
    for fold in FOLDS:
        manifest = parent["folds"][fold]["manifest"]
        inp = manifest["input"]
        bound = io.bind(inp["fit_rows_path"])
        if bound["file_sha256"] != inp["fit_rows_file_sha256"]:
            raise ValueError("historical input changed")
        bindings.append(bound)
        frame = pd.read_parquet(bound["path"])
        frame["N_total"] = [total_counts[(str(ladder), int(e))] for ladder, e in zip(frame.ladder, frame.E)]
        values, obs = packets_for(frame, manifest["membership"], inventories, metadata, manifest["fit"]["delta"])
        packets.extend(value | {"fold": fold} for value in values)
        observations.append(obs.assign(fold=fold))
    root.mkdir(parents=True)
    (root/"logs").mkdir()
    packet_bindings, domain_compression = [], []
    for i, packet in enumerate(packets):
        reduced, multiplicities, inverse = n.compact_domain(packet["support"])
        for key in ("D", "U"):
            if not np.array_equal(packet["support"][key], np.asarray(reduced[key])[inverse]):
                raise ValueError("domain reduction did not preserve original exposure/inventory rows")
        domain_compression.append(dict(packet=i, original_rows=len(inverse), unique_rows=len(multiplicities),
                                       exact_exposure_inventory_reconstruction=True))
        path = root/f"packets/{i:04d}.json"
        io.write(path, packet)
        packet_bindings.append(io.bind(path))
    obs_path, old_path = root/"observations.parquet", root/"historical_predictions.parquet"
    pd.concat(observations, ignore_index=True).to_parquet(obs_path, index=False)
    old.to_parquet(old_path, index=False)
    prior_comparison = io.read(profile["parent_comparison_plan"])
    # Reuse the prior private runtime for every non-orchestration source, so
    # concurrent unrelated checkout edits cannot change the inherited science.
    source_map = {Path(b["original_path"]): Path(b["path"]) for b in prior_comparison["sources"]}
    amended = {Path(__file__), Path(reuse.__file__), Path(snapshot.__file__), Path(arrays.__file__),
               Path(__file__).parents[1]/"atlas_comparison.py", Path.cwd()/"tests/test_stage1_atlas_comparison.py"}
    source_map.update({path: path for path in amended})
    runtime, sources, archive = snapshot.freeze(root, source_map, Path.cwd(), content_sources=source_map)
    inherited, inherited_bindings = reuse.prepare_budget(profile["parent_comparison_plan"], PROTOCOL,
        environment(), packet_bindings, io.bind(obs_path), io.bind(old_path), sources, amended)
    bindings.extend(inherited_bindings)
    task_count = len(packets)*len(PANEL)
    pilot_tasks = sorted(prior_comparison["pilot_tasks"])
    shards = arrays.partition(task_count, PROTOCOL["panel"], inherited, PROTOCOL["allocations"]["main"]["jobs"])
    plan = io.write(root/"plan.json", dict(protocol=PROTOCOL, profile=profile, output_root=str(root),
        inputs=bindings, sources=sources, archive=archive, runtime_root=str(runtime), packets=packet_bindings,
        observations=io.bind(obs_path), historical_predictions=io.bind(old_path),
        historical_reproduction=reproduction, historical_metrics=metrics, parameter_count_audit=count_evidence,
        domain_compression=domain_compression,
        environment=environment(), task_count=task_count, pilot_tasks=pilot_tasks, shards=shards,
        prior_allocated_core_hours=prior_cpu_hours,
        new_heldout_scores_computed=False))
    reuse.write_budget(plan, inherited)
    launch_scripts = []
    runner = root/"run.sh"
    write_text_once(runner, "\n".join(("#!/bin/bash", "set -euo pipefail",
        f"cd {shlex.quote(str(Path.cwd()))}",
        f"export PYTHONPATH={shlex.quote(str(runtime/'src'))}:.",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"exec {shlex.quote(profile['python_executable'])} -m {MODULE} \"$@\" --plan {shlex.quote(str(root/'plan.json'))}", "")))
    subprocess.run(["bash", "-n", str(runner)], check=True)
    launch_scripts.append(io.bind(runner))
    for phase, allocation in PROTOCOL["allocations"].items():
        cores, minutes = allocation["cores"], allocation["minutes"]
        jobs = allocation.get("jobs", 1)
        array_options = ([f"#SBATCH --array=0-{jobs-1}%{allocation['parallel_jobs']}"] if jobs > 1 else [])
        log_id = "%A_%a" if jobs > 1 else "%j"
        script = ["#!/bin/bash", f"#SBATCH --job-name=atlas-{phase}",
            f"#SBATCH --partition={profile['partition']}", f"#SBATCH --account={profile['account']}",
            "#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={cores}",
            f"#SBATCH --mem={profile['memory']}", f"#SBATCH --time={minutes}",
            *array_options, "#SBATCH --signal=B:USR1@60", f"#SBATCH --output={root}/logs/{phase}-{log_id}.out",
            "set -euo pipefail", f"exec bash {shlex.quote(str(runner))} run --phase {phase}", ""]
        path = root/f"{phase}.sbatch"
        write_text_once(path, "\n".join(script))
        subprocess.run(["bash", "-n", str(path)], check=True)
        launch_scripts.append(io.bind(path))
    io.write(root/"launch.json", dict(plan_sha256=plan["sha256"], scripts=launch_scripts))
    return dict(plan=str(root/"plan.json"), runner=str(runner), sha256=plan["sha256"], packets=len(packets),
                model_variants=len(PANEL), deferred_variants=len(DEFERRED), tasks=task_count,
                pilot_tasks=len(pilot_tasks))


def load(path, *, verify_inputs=False):
    plan = io.read(path)
    if plan["protocol"] != PROTOCOL or plan["environment"] != environment():
        raise ValueError("comparison protocol or environment drift")
    snapshot.require_frozen_module(plan, __file__)
    for bound in [*plan["sources"], plan["archive"]]:
        io.verify(bound)
    if verify_inputs:
        for bound in [*plan["inputs"], *plan["packets"], plan["observations"], plan["historical_predictions"]]:
            io.verify(bound)
        launch = io.read(Path(plan["output_root"])/"launch.json")
        if launch["plan_sha256"] != plan["sha256"]:
            raise ValueError("launch scripts belong to another plan")
        for bound in launch["scripts"]:
            io.verify(bound)
    return plan


def task_path(plan, task):
    return Path(plan["output_root"])/f"results/{task:05d}.json"


def work(root, plan_hash, packet_binding, task, spec_payload):
    """One CPU process; its inputs contain no heldout outcomes."""
    io.verify(packet_binding)
    packet = io.read(packet_binding["path"])
    spec = m.Spec(**spec_payload)
    result = n.fit_group(spec, packet["train"], packet["support"])
    result.update(plan_sha256=plan_hash, task=task, packet_sha256=packet["sha256"],
                  fold=packet["fold"], language=packet["language"])
    io.write(Path(root)/f"results/{task:05d}.json", result)
    return dict(task=task, certified=result["certified"], runtime_seconds=result["runtime_seconds"])


def run(path, phase):
    # Array workers verify frozen sources plus their packet bytes on use.
    # Preparation and the final audit verify the full inherited input graph;
    # avoid 128 redundant reads of thousands of old receipts at startup.
    plan = load(path, verify_inputs=phase != "main")
    if phase not in PROTOCOL["allocations"]:
        raise ValueError("the parallel amendment inherits prior fits; only main and finalize are authorized")
    allocation = PROTOCOL["allocations"][phase]
    index = arrays.array_index(allocation, os.environ)
    phase_id = phase if index is None else f"{phase}_{index:03d}"
    cores, limit = allocation["cores"], allocation["minutes"]*60
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != str(cores):
        raise ValueError(f"real fits require a {cores}-core Slurm allocation")
    if any(os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("BLAS/OpenMP threads must be pinned to one")
    root = Path(plan["output_root"])
    if phase == "main":
        gate = io.read(root/"pilot_gate.json")
        if gate["plan_sha256"] != plan["sha256"] or not gate["launch_allowed"]:
            raise ValueError("training-only pilot did not authorize the full CPU budget")
    claim = root/f"execution/{phase_id}.json"
    io.write(claim, dict(plan_sha256=plan["sha256"], job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), phase=phase, array_index=index,
        array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"), cores=cores, seconds_limit=limit,
        maximum_allocated_core_hours=cores*limit/3600))
    if phase == "finalize":
        from . import reporting
        began = time.monotonic()
        audited = reporting.audit(path)
        report = reporting.evaluate(path)
        return io.write(root/"execution/finalize_completion.json", dict(plan_sha256=plan["sha256"],
            wall_seconds=time.monotonic()-began, audit_sha256=audited["sha256"],
            report_sha256=report["sha256"], status=report["status"], heldout_scores_computed=True))
    # Each frozen shard owns disjoint unfinished tasks. The evaluator still
    # requires all 450 targets before any comparator enters the main table.
    task_ids = plan["shards"]["tasks"][index]
    remaining = iter(task for task in task_ids if not task_path(plan, task).exists())
    began, stop, completed = time.monotonic(), [False], []
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    pool = ProcessPoolExecutor(max_workers=cores, mp_context=multiprocessing.get_context("spawn"))
    pending = {}
    try:
        while True:
            if time.monotonic()-began > limit-90:
                stop[0] = True
            while not stop[0] and len(pending) < cores:
                task = next(remaining, None)
                if task is None:
                    break
                packet_i, spec_i = divmod(task, len(PANEL))
                future = pool.submit(work, str(root), plan["sha256"], plan["packets"][packet_i], task, PANEL[spec_i].payload())
                pending[future] = task
            if not pending:
                break
            finished, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                try:
                    value = future.result()
                except Exception as error:
                    value = dict(task=task, certified=False, worker_error=f"{type(error).__name__}: {error}")
                    io.write(task_path(plan, task), value | {"plan_sha256": plan["sha256"], "heldout_scores_computed": False})
                completed.append(value)
                print(json.dumps(value), flush=True)
            if stop[0]:
                # Children may be mid-optimizer. Preserve only complete,
                # atomically sealed results; never extend the allocated cap.
                for process in multiprocessing.active_children():
                    process.terminate()
                break
    finally:
        pool.shutdown(wait=not stop[0], cancel_futures=True)
    return io.write(root/f"execution/{phase_id}_completion.json", dict(plan_sha256=plan["sha256"], array_index=index,
        completed_tasks=len(completed), wall_seconds=time.monotonic()-began,
        stopped_at_budget=stop[0], heldout_scores_computed=False))


def pilot_gate(path):
    plan = load(path)
    root = Path(plan["output_root"])
    rows, missing = [], []
    for task in plan["pilot_tasks"]:
        target = task_path(plan, task)
        if not target.exists():
            missing.append(task)
            continue
        value = io.read(target)
        if value["plan_sha256"] != plan["sha256"] or value["task"] != task:
            raise ValueError("pilot receipt identity mismatch")
        rows.append(value)
    # Per form/count/sign cost, using the worst pilot language/cap combination
    # avoids assuming that English predicts low-resource conditioning cost.
    maxima = {}
    for row in rows:
        if "spec" in row:
            spec = m.Spec(**row["spec"])
            key = (spec.form, spec.parameter_count, spec.signed)
            maxima[key] = max(maxima.get(key, 0.), row["runtime_seconds"])
    forecast = 0.
    for spec in PANEL:
        key = (spec.form, spec.parameter_count, spec.signed)
        forecast += maxima.get(key, float("inf"))*len(plan["packets"])
    forecast *= PROTOCOL["forecast_safety_factor"]/3600
    errors = [row["task"] for row in rows if "worker_error" in row]
    allowed = not missing and not errors and np.isfinite(forecast) and forecast <= PROTOCOL["main_allocated_core_hours"]
    # Coefficient rank and a failed start are not launch blockers. They will
    # remain explicit per-target certification requirements at evaluation.
    return io.write(root/"pilot_gate.json", dict(plan_sha256=plan["sha256"], launch_allowed=bool(allowed),
        projected_core_hours=float(forecast) if np.isfinite(forecast) else None,
        missing_tasks=missing, worker_errors=errors, certified_tasks=sum(row.get("certified", False) for row in rows),
        completed_tasks=len(rows), heldout_scores_computed=False,
        reason="within frozen runtime budget" if allowed else "incomplete pilot, worker errors, or forecast exceeds the bound main allocation"))
