"""Sealed, training-only ATLAS pilot. This CLI cannot score heldouts or submit jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import tarfile
import time

import numpy as np
import pandas as pd

from . import atlas_full_model as model
from .atlas_exposure import exposure_components
from .data import sha256_file
from .execution import trajectory_weights, write_json_once
from .baseline_refit_workflow.contracts import seal, verify_seal

MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.atlas_full_pilot"
PROTOCOL = {
    "version": "atlas_full_training_pilot/v1", "equation_source": "arxiv:2510.22037v2 equations 1-3",
    "languages": ["eng_Latn", "por_Latn", "swh_Latn"],
    "folds": ["largest_N_act", "E32", "K200_common_languages"],
    "N": "declared active parameters including embeddings; dense-law MoE adaptation",
    "U": "unique original-document tokens in selected training pool, including EOD, before replicas",
    "donors": "top three mean normalized co-exposures across training K levels only, lexical ties",
    "remainder_U": "sum training-pool U over present non-target, non-donor languages",
    "parameters": "ten independent coefficients per target; lambda shared across five components",
    "objective": "same fold Huber delta and trajectory weights, normalized within target; no ridge",
    "bounds": {name: [float(lo), None if not np.isfinite(hi) else float(hi)]
        for name, lo, hi in zip(model.NAMES, model.LOWER, model.UPPER)},
    "starts": [list(x) for x in model.STARTS], "solver": model.SETTINGS,
    "gradient_tolerance": 1e-8, "prediction_tolerance": 1e-4,
    "objective_relative_tolerance": 1e-6, "rank_relative_tolerance": 1e-7,
    "heldout_scoring": False, "automatic_full_campaign": False,
    "scope": "training-only numerical pilot, not baseline superiority or coefficient interpretation",
}


def read(path):
    value = json.loads(Path(path).read_text())
    verify_seal(value, "sha256", label=str(path))
    return value


def write(path, value):
    value = seal(value, "sha256")
    write_json_once(Path(path), value)
    return value


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "file_sha256": sha256_file(path)}


def verify(binding):
    if sha256_file(Path(binding["path"])) != binding["file_sha256"]:
        raise ValueError(f"changed bound file: {binding['path']}")


def inventories(corpus_path):
    corpus_path = Path(corpus_path)
    corpus = json.loads(corpus_path.read_text())
    hashes = json.loads((corpus_path.parent/"hash_manifest.json").read_text())
    if hashes["files"]["corpus_manifest.json"] != sha256_file(corpus_path):
        raise ValueError("corpus manifest hash mismatch")
    values, bindings = {}, [bind(corpus_path), bind(corpus_path.parent/"hash_manifest.json")]
    tokenizer = corpus["inputs"]["tokenizer_model"]["sha256"]
    for language, item in sorted(corpus["languages"].items()):
        receipt = item["train"]["unigram_counts"]
        path = corpus_path.parent/receipt["path"]
        binding = bind(path)
        if binding["file_sha256"] != receipt["sha256"] or binding["file_sha256"] != hashes["files"][receipt["path"]]:
            raise ValueError(f"inventory hash mismatch: {language}")
        value = json.loads(path.read_text())
        u = value["unique_train_tokens_including_special"]
        total = value["exposure_train_tokens_including_special"]
        if not (value["language"] == language and value["source_split"] == "train_only"
                and value["tokenizer_model_sha256"] == tokenizer and 0 < u <= total
                and total == item["train"]["selected_tokens"]
                and u == value["non_special_token_count"]+value["excluded_special_token_count"]):
            raise ValueError(f"invalid training inventory: {language}")
        values[language] = int(u)
        bindings.append(binding)
    return values, bindings


def packet(frame, membership, language, U, delta):
    """Select training before donor selection, minima, outcomes or weighting."""
    train_ids, test_ids = set(membership["train_trajectory_ids"]), set(membership["test_trajectory_ids"])
    if train_ids & test_ids:
        raise ValueError("overlapping train/test trajectories")
    train = frame[frame.trajectory_id.isin(train_ids)].copy()
    if membership["train_readout_ids"]:
        train = train[train.readout_id.isin(membership["train_readout_ids"])]
    _, donors = exposure_components(train)
    if language not in donors or len(donors[language]) != 3:
        raise ValueError("pilot target needs three observed training donors")
    train["weight"] = trajectory_weights(tuple(train.trajectory_id.astype(str)))
    target = train[train.language == language]
    pivot = train.pivot(index="readout_id", columns="language", values="D_t").fillna(0)
    source = pivot.loc[target.readout_id]
    chosen = list(donors[language])
    remainder = [name for name in pivot.columns if name not in [language, *chosen]]
    D = np.column_stack([target.D_t, source[chosen], source[remainder].sum(axis=1)])
    u_remainder = source[remainder].gt(0).to_numpy() @ np.array([U[name] for name in remainder])
    donor_u = source[chosen].gt(0).to_numpy()*np.array([U[name] for name in chosen])
    inventory = np.column_stack([np.full(len(target), U[language]), donor_u, u_remainder])
    if not np.allclose(D.sum(axis=1), target.D_total, rtol=1e-12, atol=1e-6):
        raise ValueError("ATLAS exposure partition does not reconstruct total tokens")
    if target.empty or not np.isfinite(target.adjusted_loss).all():
        raise ValueError("missing/nonfinite target training rows")
    floor = -float(target.unigram_nll.iloc[0])
    if not np.allclose(target.unigram_nll, -floor) or floor >= target.adjusted_loss.min():
        raise ValueError("invalid physical loss floor")
    return dict(language=language, donors=chosen, D=D.tolist(), U=inventory.tolist(),
        N=target.N_act.tolist(), y=target.adjusted_loss.tolist(), weights=target.weight.tolist(),
        physical_floor=floor, training_minimum=float(target.adjusted_loss.min()), delta=float(delta),
        trajectory_ids=target.trajectory_id.tolist(), readout_ids=target.readout_id.tolist(),
        training_only=True)


def prepare(profile_path):
    profile = json.loads(Path(profile_path).read_text())
    root = Path(profile["output_root"]).resolve()
    if root.exists():
        raise ValueError("pilot output root must be new")
    parent = read(profile["parent_plan"])
    U, bindings = inventories(profile["corpus_manifest"])
    bindings += [bind(profile_path), bind(profile["parent_plan"])]
    packets = []
    for fold in PROTOCOL["folds"]:
        manifest = parent["folds"][fold]["manifest"]
        inp = manifest["input"]
        b = bind(inp["fit_rows_path"])
        if b["file_sha256"] != inp["fit_rows_file_sha256"]:
            raise ValueError("historical fit snapshot changed")
        bindings.append(b)
        frame = pd.read_parquet(inp["fit_rows_path"])
        for language in PROTOCOL["languages"]:
            packets.append(packet(frame, manifest["membership"], language, U, manifest["fit"]["delta"]) | {"fold": fold})
    preflight = []
    for value in packets:
        x = np.array([.5, .3, .8, .4, .3, .2, .1, .2, .3, .4])
        _, jac = model.predict(x, value)
        maximum = 0.
        for j in range(10):
            h = np.eye(10)[j]*1e-6
            numeric = (model.predict(x+h, value)[0]-model.predict(x-h, value)[0])/2e-6
            maximum = max(maximum, float(np.max(np.abs(numeric-jac[:, j])/np.maximum(1., np.abs(jac[:, j])))))
        if maximum > 1e-7:
            raise ValueError("real-training-row ATLAS gradient check failed")
        preflight.append(dict(fold=value["fold"], language=value["language"],
            rows=len(value["y"]), maximum_scaled_gradient_error=maximum))
    source_root = Path(__file__).resolve().parent
    sources = [bind(p) for p in sorted(source_root.rglob("*.py"))]
    root.mkdir(parents=True)
    archive = root/"source.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for item in sources:
            tar.add(item["path"], arcname=str(Path(item["path"]).relative_to(source_root)), recursive=False)
    packet_bindings = []
    for index, value in enumerate(packets):
        path = root/f"input/{index:02d}.json"
        write(path, value)
        packet_bindings.append(bind(path))
    import scipy
    environment = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
    plan = write(root/"plan.json", dict(protocol=PROTOCOL, output_root=str(root),
        inputs=bindings, sources=sources, archive=bind(archive), packets=packet_bindings,
        inventory_languages=len(U), environment=environment, preflight=preflight,
        heldout_scores_computed=False))
    script = "\n".join(["#!/bin/bash", f"#SBATCH --partition={profile['partition']}",
        f"#SBATCH --account={profile['account']}", "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={profile['memory']}", f"#SBATCH --time={profile['time_limit']}",
        "#SBATCH --array=0-8%3", f"#SBATCH --output={root}/logs/%A_%a.out",
        "set -euo pipefail", f"cd {shlex.quote(str(Path.cwd()))}",
        "export PYTHONPATH=src:.", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"exec {shlex.quote(profile['python_executable'])} -m {MODULE} worker --plan {shlex.quote(str(root/'plan.json'))} --task \"$SLURM_ARRAY_TASK_ID\"", ""])
    (root/"logs").mkdir()
    with (root/"pilot.sbatch").open("x") as handle:
        handle.write(script)
    return {"plan": str(root/"plan.json"), "sha256": plan["sha256"], "tasks": len(packets), "starts": len(packets)*4}


def load(plan_path):
    plan = read(plan_path)
    if plan["protocol"] != PROTOCOL:
        raise ValueError("pilot protocol changed")
    for b in [*plan["sources"], plan["archive"], *plan["packets"]]:
        verify(b)
    import scipy
    if plan["environment"] != {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}:
        raise ValueError("pilot runtime environment changed")
    return plan


def worker(plan_path, task):
    plan = load(plan_path)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("real-data pilot requires one-core Slurm task")
    if any(os.environ.get(k) != "1" for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]):
        raise ValueError("numerical thread counts must be one")
    if task not in range(9):
        raise ValueError("invalid pilot task")
    path = Path(plan["output_root"])/f"results/{task:02d}.json"
    if path.exists():
        raise ValueError("pilot task already has a receipt")
    data = read(plan["packets"][task]["path"])
    attempts = []
    for i in range(4):
        began = time.monotonic()
        try:
            result = model.fit_start(data, i)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            result = dict(start_index=i, success=False, message=f"{type(error).__name__}: {error}", runtime_seconds=time.monotonic()-began)
        attempts.append(result)
    return write(path, dict(plan_sha256=plan["sha256"], task=task, fold=data["fold"],
        language=data["language"], attempts=attempts, heldout_scores_computed=False,
        runtime={"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname()}))


def summarize(plan_path):
    plan = load(plan_path)
    groups = []
    for task in range(9):
        result = read(Path(plan["output_root"])/f"results/{task:02d}.json")
        if result["plan_sha256"] != plan["sha256"] or result["task"] != task:
            raise ValueError("receipt belongs to different pilot")
        attempts = result["attempts"]
        good = sorted([a for a in attempts if a["success"]], key=lambda a: (a["objective"], a["start_index"]))
        best = good[0] if good else None
        equivalent = [] if best is None else [a for a in good
            if abs(a["objective"]-best["objective"]) <= 1e-6*max(abs(best["objective"]), 1e-15)
            and np.max(np.abs(np.array(a["training_predictions"])-best["training_predictions"])) <= 1e-4]
        groups.append(dict(task=task, fold=result["fold"], language=result["language"],
            successful_starts=len(good), equivalent_best_starts=len(equivalent),
            best_rank=None if best is None else best["normalized_jacobian_rank"],
            best_contacts=[] if best is None else best["boundary_contacts"],
            best_objective=None if best is None else best["objective"],
            runtime_seconds=sum(a["runtime_seconds"] for a in attempts)))
    stable = all(g["successful_starts"] == 4 and g["equivalent_best_starts"] >= 3 for g in groups)
    identified = all(g["best_rank"] == 10 and not g["best_contacts"] for g in groups)
    return write(Path(plan["output_root"])/"report.json", dict(plan_sha256=plan["sha256"],
        status="complete", groups=groups, convergence_and_training_prediction_gate=stable,
        interior_identification_gate=identified, ready_for_full_comparison=stable and identified,
        heldout_scores_computed=False, automatic_full_campaign=False,
        interpretation="Boundary/null or rank failure needs diagnosis, not automatic baseline rejection; pilot is not a full numerical audit."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker", "summarize"])
    parser.add_argument("--profile")
    parser.add_argument("--plan")
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    value = prepare(args.profile) if args.action == "prepare" else worker(args.plan, args.task) if args.action == "worker" else summarize(args.plan)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
