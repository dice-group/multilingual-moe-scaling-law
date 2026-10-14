"""Loss-only evaluation of the cooled JointMoE-LR L26/E32 endpoint.

Reuses the stage-owned evaluator; never trains, edits checkpoints, or adds
observations to the fitting grid. Submission runs an immutable source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from uuid import uuid4

from multilingual_megatron.repo import find_repo_root
from .evaluation_launch import EXPOSURE_SOURCE

from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import MODULE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import SOURCE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import LAUNCH_SOURCE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import PROFILE, PACKAGE_SOURCE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import allocation
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import attempt_logs
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import render_script
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import submission_command
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import step_command
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import worker_command


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify_files(entries):
    for entry in entries:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"changed input: {entry['path']}")


def languages(path):
    return sorted({lang for group in read(path).values() if isinstance(group, dict)
                   for lang in group.get("languages", [])})


def cooled(audit, scheduler, expected_step, expected_samples, complete):
    """Endpoint eligibility requires the full prescribed WSD duration."""
    return bool(complete and audit.valid_full_state
                and audit.train_state_step == expected_step
                and audit.consumed_train_samples == expected_samples
                and scheduler.get("lr_decay_style") == "WSD"
                and scheduler.get("num_steps") == expected_samples
                and scheduler.get("lr_decay_steps") == expected_samples
                and scheduler.get("min_lr") == 0.0)


def prepare(root, profile_path):
    from multilingual_megatron.pipeline.stages.training.checkpoint_state import (
        inspect_full_state_checkpoint, _load_torch_dict,
    )
    from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import load_evaluation_spec
    from multilingual_megatron.experiments.scaling.stage1_release.exposure import _exact_exposures
    from multilingual_megatron.experiments.scaling.stage1_fit.m6_composition import load_language_metadata

    cfg = read(profile_path)
    allocation(cfg, read(root / cfg["runtime"]))
    base = root / cfg["campaign_root"]
    subdir = Path(cfg["output_subdir"])
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("output_subdir must stay within the campaign")
    out = base / subdir
    if (out / "submission.json").exists() or (out / "submission.lock").exists():
        raise ValueError("submitted campaign is immutable; do not reprepare it")
    contract = read(root / cfg["contract"])
    endpoint = next(x for x in contract["checkpoint_contract"]["scientific_checkpoints"] if x["label"] == "final_wsd")
    step = endpoint["expected_step"]
    samples = endpoint["executable_consumed_tokens"] // 2048
    inventory = []
    selected = None
    inputs = [identity(profile_path), identity(root / cfg["contract"]), identity(root / cfg["runtime"])]
    for role in ("trunk", "cooldown"):
        for state_path in sorted((base / role / "runs").glob("*/segments/*/checkpoints/iter_*/train_state.pt")):
            cp = state_path.parent
            segment = cp.parent.parent
            state = _load_torch_dict(state_path)
            audit = inspect_full_state_checkpoint(checkpoint_root=cp.parent,
                checkpoint_step=int(state["step"]), expected_step=int(state["step"]),
                expected_consumed_samples=int(state["consumed_train_samples"]))
            scheduler = _load_torch_dict(cp / "common.pt").get("opt_param_scheduler", {})
            eligible = role == "cooldown" and cooled(audit, scheduler, step, samples,
                                                       (segment / ".training_complete").is_file())
            inventory.append({**audit.to_dict(), "role": role, "eligible": eligible,
                              "reason": "complete_final_cooldown" if eligible else "not_a_complete_prescribed_cooldown"})
            if eligible:
                if selected is not None:
                    raise ValueError("ambiguous cooled endpoint")
                selected = segment
                inputs.extend(identity(cp / name) for name in (".metadata", "train_state.pt", "common.pt", "run_config.yaml"))
    if selected is None:
        raise ValueError("no valid cooled final checkpoint")
    inputs.append(identity(selected / "bridge_command.json"))
    # Every segment used the same globally indexed stream. Prove this rather
    # than counting each segment's entire prefix repeatedly.
    cache_files = []
    for role in ("trunk", "cooldown"):
        for cache in sorted((base / role / "runs").glob("*/segments/*/dataset_cache")):
            entries = {}
            for suffix in ("dataset_index.npy", "dataset_sample_index.npy", "description.txt"):
                matches = list(cache.glob(f"*-BlendedDataset-train-{suffix}"))
                if len(matches) != 1:
                    raise ValueError(f"ambiguous or missing blend {suffix}: {cache}")
                entries[suffix] = identity(matches[0])
            cache_files.append(entries)
    if len(cache_files) != len(inventory):
        raise ValueError("missing segment stream evidence")
    for suffix in cache_files[0]:
        if len({item[suffix]["sha256"] for item in cache_files}) != 1:
            raise ValueError(f"training stream changed across segments: {suffix}")
    inputs.extend(entry for item in cache_files for entry in item.values())
    corpus = root / cfg["corpus_root"]
    tier = corpus / "tokenized/tiers/K200/megatron_data_manifest.json"
    inputs.append(identity(tier))
    exposures = _exact_exposures(index_path=Path(cache_files[0]["dataset_index.npy"]["path"]),
        description_path=Path(cache_files[0]["description.txt"]["path"]),
        tier_manifest_path=tier, end_samples=samples, sequence_length=2048)
    required = languages(root / "configs/language_groupings/scaling_fw2_v2_k200.json")
    if len(required) != 200 or set(exposures) != set(required):
        raise ValueError("K200 exposure/language mismatch")
    command = read(selected / "bridge_command.json")["cmd"]
    overrides = {}
    for item in command:
        if item.startswith(("model.", "mixed_precision=")):
            key, value = item.split("=", 1)
            overrides[key] = value
    model = {"num_layers": 26, "hidden_size": 1664, "ffn_hidden_size": 4992,
             "num_attention_heads": 26, "seq_length": 2048, "vocab_size": 64000,
             "num_moe_experts": 32, "moe_router_topk": 1,
             "moe_expert_capacity_factor": None, "moe_pad_expert_input_to_capacity": False,
             "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 4,
             "expert_model_parallel_size": 2}
    cohorts = {key: languages(root / f"configs/language_groupings/scaling_fw2_v2_k{k:03d}.json")
               for key, k in (("english", 1), ("k12", 12), ("k50", 50))}
    inputs.extend(identity(root / f"configs/language_groupings/scaling_fw2_v2_k{k:03d}.json") for k in (1, 12, 50, 200))
    spec = {"schema": "multilingual_per_language_checkpoint_eval/v1", "run_id": "l26_e32_jointmoe_lr_final",
            "output_dir": str(out / "final"), "model_recipe": {
                "name": command[command.index("--recipe") + 1],
                "overrides": [f"{key}={value}" for key, value in overrides.items()], "expected_model": model},
            "sequence_length": 2048, "eval_micro_batch_size": cfg["eval_micro_batch_size"],
            "context_token_id": 2, "training_seed": 1234,
            "require_moe_routing_diagnostics": False, "routing_diagnostics": {"enabled": False},
            "trained_languages": required, "required_languages": required, "cohorts": cohorts,
            "checkpoint": {"path": str(selected / "checkpoints"), "step": step},
            "validation_data_manifest": identity(corpus / "tokenized/validation_megatron_data_manifest.json"),
            "corpus_manifest": identity(corpus / "release_32b/corpus_manifest.json")}
    spec_path = out / "evaluation_spec.json"
    write(spec_path, spec)
    loaded = load_evaluation_spec(spec_path)
    inputs.extend([spec["validation_data_manifest"], spec["corpus_manifest"], identity(loaded.tokenizer_model_path)])
    inputs.extend({"path": str(x.path), "sha256": x.sha256} for x in loaded.unigrams.values())
    meta_path = root / cfg["language_metadata"]
    meta = load_language_metadata(meta_path, required)
    inputs.append(identity(meta_path))
    exposure_rows = [{"language": lang, "D_t": exposures[lang],
                      "family": meta[lang]["family"], "script": meta[lang]["script"]}
                     for lang in required]
    write(out / "exposure.json", exposure_rows)
    write(out / "checkpoint_inventory.json", inventory)
    plan = {"schema": "l26_e32_evaluation/v1", "root": str(root), "output": str(out),
            "profile": cfg, "runtime": read(root / cfg["runtime"]), "spec": identity(spec_path),
            "inputs": inputs + [identity(out / "exposure.json"), identity(out / "checkpoint_inventory.json")],
            "expected_step": step, "expected_samples": samples, "N_act": 1148878848,
            "N_act_nonembedding": 935886848, "E": 32, "K": 200, "model_seed": 1234,
            "independent_training_runs": 1, "held_out_from_fitting_grid": True}
    write(out / "plan.json", plan)
    return out / "plan.json"


def collect(plan):
    import pandas as pd
    out = Path(plan["output"])
    verify_files(plan["inputs"] + [plan["spec"]])
    manifest_path = out / "final/evaluation_manifest.json"
    manifest = read(manifest_path)
    spec = read(plan["spec"]["path"])
    if manifest["status"] != "complete" or manifest["checkpoint"]["step"] != plan["expected_step"]:
        raise ValueError("incomplete or wrong evaluation")
    if manifest["spec"]["sha256"] != plan["spec"]["sha256"]:
        raise ValueError("evaluation spec mismatch")
    if set(manifest["languages"]) != set(spec["required_languages"]):
        raise ValueError("incomplete language coverage")
    verify_files(list(manifest["languages"].values()) + [manifest["validation_report"]])
    exposures = {x["language"]: x for x in read(out / "exposure.json")}
    rows = []
    for lang, entry in manifest["languages"].items():
        loss = read(entry["path"])
        if loss["token_count"] != loss["backend"]["backend_token_count"] or loss["token_count"] <= 0:
            raise ValueError(f"token denominator mismatch: {lang}")
        rows.append({**exposures[lang], **{key: loss[key] for key in (
            "raw_nll", "atlas_tao_unigram_adjusted_loss", "nll_sum", "token_count", "unigram_log_probability_sum")},
            **{key: plan[key] for key in ("N_act", "N_act_nonembedding", "E", "K", "model_seed")},
            "D_total": plan["expected_samples"] * 2048, "checkpoint_step": plan["expected_step"]})
    frame = pd.DataFrame(rows).sort_values("language")
    frame.to_parquet(out / "observations.parquet", index=False)
    frame.to_csv(out / "observations.csv", index=False)
    write(out / "release.json", {"status": "complete", "evaluation": identity(manifest_path),
        "observations": identity(out / "observations.parquet"), "plan": identity(out / "plan.json"),
        "rows": len(frame), "independent_training_runs": 1, "held_out_from_fitting_grid": True})


def snapshot(root, destination):
    """Archive committed dependencies and overlay only this workflow's files."""
    destination.mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    def archive(repo, revision, target, paths=()):
        target.mkdir(parents=True, exist_ok=True)
        tar_path = destination / "archive.tar"
        subprocess.run(["git", "archive", "--format=tar", f"--output={tar_path}", revision, *paths], cwd=repo, check=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(target, filter="data")
        tar_path.unlink()
    archive(root, commit, destination, ("src", "configs"))
    bridge = subprocess.check_output(["git", "rev-parse", f"{commit}:external/Megatron-Bridge"], cwd=root, text=True).strip()
    bridge_repo = root / "external/Megatron-Bridge"
    mcore = subprocess.check_output(["git", "rev-parse", f"{bridge}:3rdparty/Megatron-LM"], cwd=bridge_repo, text=True).strip()
    archive(bridge_repo, bridge, destination / "external/Megatron-Bridge")
    archive(bridge_repo / "3rdparty/Megatron-LM", mcore, destination / "external/Megatron-Bridge/3rdparty/Megatron-LM")
    for relative in (SOURCE, LAUNCH_SOURCE, PACKAGE_SOURCE, EXPOSURE_SOURCE, PROFILE):
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination / relative)
    (destination / "outputs").symlink_to(root / "outputs", target_is_directory=True)
    files = [identity(p) for folder in ("src", "configs", "external") for p in sorted((destination / folder).rglob("*")) if p.is_file()]
    write(destination / "source_manifest.json", {"base_commit": commit, "bridge_commit": bridge,
        "megatron_commit": mcore, "overlays": [SOURCE, LAUNCH_SOURCE, PACKAGE_SOURCE, EXPOSURE_SOURCE, PROFILE], "files": files})
    for p in destination.rglob("*"):
        if p.is_file() and not p.is_symlink():
            p.chmod(p.stat().st_mode & ~0o222)


def submit(plan_path):
    plan = read(plan_path)
    allocation(plan["profile"], plan["runtime"])
    verify_files(plan["inputs"] + [plan["spec"]])
    out = Path(plan["output"])
    (out / "submission.lock").mkdir()
    plan["attempt_id"] = uuid4().hex
    logs = attempt_logs(out, plan["attempt_id"])
    logs.mkdir(parents=True)
    source = out / "source"
    snapshot(Path(plan["root"]), source)
    # Execute against sealed config inputs, not a later mutable checkout.
    for entry in plan["inputs"]:
        relative = Path(entry["path"]).relative_to(plan["root"])
        if relative.parts[0] == "configs":
            frozen = identity(source / relative)
            if frozen["sha256"] != entry["sha256"]:
                raise ValueError(f"snapshot config differs from preflight: {relative}")
            entry.update(frozen)
    write(plan_path, plan)
    script = out / "evaluate.slurm.sh"
    script.write_text(render_script(plan_path, plan, "run"))
    script.chmod(0o700)
    # Exercise the actual frozen recipe imports under the same module/Conda
    # setup before spending a GPU allocation. This is not a GPU-load proof.
    preflight_script = out / "preflight.sh"
    preflight_script.write_text(render_script(plan_path, plan, "preflight"))
    with (logs / "preflight.log").open("w") as log:
        subprocess.run(["bash", str(preflight_script)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    cmd = submission_command(plan, script)
    subprocess.run(["bash", "-n", str(script)], check=True)
    write(out / "launch_identity.json", {"plan": identity(plan_path),
        "source_manifest": identity(source / "source_manifest.json"), "script": identity(script),
        "preflight": identity(out / "preflight.json")})
    result = subprocess.run(cmd, capture_output=True, text=True)
    write(out / "submission.json", {"returncode": result.returncode, "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(), "command": cmd, "plan": identity(plan_path),
        "source_manifest": identity(source / "source_manifest.json"), "script": identity(script)})
    result.check_returncode()
    print("Submitted evaluation job", result.stdout.strip(), flush=True)


def execution_env(plan, work):
    """Shared import preflight and GPU worker environment; profile wins."""
    from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
        _prepare_pybind11_shim, _prepend_python_env_cuda_libs,
    )
    source = Path(plan["output"]) / "source"
    python = str(Path(plan["runtime"]["module_runtime"]["conda_env"]) / "bin/python")
    env = {**os.environ, **plan["profile"]["runtime_env"]}
    bridge = source / "external/Megatron-Bridge"
    work.mkdir(parents=True, exist_ok=True)
    shim = _prepare_pybind11_shim(work, env)
    env["PYTHONPATH"] = ":".join(str(p) for p in (shim, source / "src", source, bridge / "src", bridge / "3rdparty/Megatron-LM") if p)
    _prepend_python_env_cuda_libs(env, python)
    return env


def preflight(plan_path):
    plan = read(plan_path)
    allocation(plan["profile"], plan["runtime"])
    out = Path(plan["output"])
    verify_files(read(out / "source/source_manifest.json")["files"])
    work = out / "runtime/preflight"
    env = execution_env(plan, work)
    python = str(Path(plan["runtime"]["module_runtime"]["conda_env"]) / "bin/python")
    subprocess.run([python, "-m",
        "multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe", "--help"],
        cwd=work, env=env, check=True, timeout=240)
    write(out / "preflight.json", {"status": "passed", "scope": "frozen_recipe_imports_not_gpu_checkpoint_load",
        "plan": identity(plan_path), "source_manifest": identity(out / "source/source_manifest.json"),
        "runtime_env": plan["profile"]["runtime_env"]})


def run(plan_path, rank=False):
    plan = read(plan_path)
    cfg = plan["profile"]
    nodes, gpus, _ = allocation(cfg, plan["runtime"])
    source = Path(plan["output"]) / "source"
    receipt = read(Path(plan["output"]) / "launch_identity.json")
    verify_files([receipt["plan"], receipt["source_manifest"], receipt["script"], receipt["preflight"]])
    if not rank:
        verify_files(read(source / "source_manifest.json")["files"])
        verify_files(plan["inputs"] + [plan["spec"]])
        print(f"[EVAL_START] job={os.environ['SLURM_JOB_ID']} nodes={os.environ['SLURM_JOB_NODELIST']} gpus={nodes*gpus}", flush=True)
        subprocess.run(step_command(plan_path, plan), check=True)
        collect(plan)
        return
    node = os.environ["SLURM_PROCID"]
    work = Path(plan["output"]) / "runtime" / node
    env = execution_env(plan, work)
    master = subprocess.check_output([str(Path(cfg["slurm_bin"]) / "scontrol"), "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()[0]
    spec = read(plan["spec"]["path"])
    cmd = worker_command(plan, spec, node, master, os.environ["SLURM_JOB_ID"])
    subprocess.run(cmd, cwd=work, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "preflight", "run", "rank", "collect"))
    parser.add_argument("--profile", type=Path, default=Path(PROFILE))
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(find_repo_root(), args.profile.resolve()))
    elif args.action == "submit":
        submit(args.plan.resolve())
    elif args.action == "collect":
        collect(read(args.plan))
    elif args.action == "preflight":
        preflight(args.plan.resolve())
    else:
        run(args.plan.resolve(), rank=args.action == "rank")


if __name__ == "__main__":
    main()
