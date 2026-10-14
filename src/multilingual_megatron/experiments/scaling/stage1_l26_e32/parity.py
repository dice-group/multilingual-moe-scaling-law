"""Bounded, same-checkpoint PP4/PP1 loss diagnostic; never a training launch."""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

from multilingual_megatron.repo import find_repo_root
from .evaluation_launch import EXPOSURE_SOURCE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import execution_env
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import identity
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import read
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import verify_files
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import write
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import LAUNCH_SOURCE, SOURCE as EVALUATION_SOURCE, PACKAGE_SOURCE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import allocation
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import attempt_logs
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import render_script
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import submission_command
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import step_command
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import worker_command

MODULE = "multilingual_megatron.experiments.scaling.stage1_l26_e32.parity"
SOURCE = "src/multilingual_megatron/experiments/scaling/stage1_l26_e32/parity.py"
EVALUATOR = "src/multilingual_megatron/pipeline/stages/final_loss/per_language_checkpoint_eval.py"
INFERENCE = "src/multilingual_megatron/pipeline/backends/megatron_bridge/checkpoint_inference.py"
PROFILE = "configs/stage1/production/stage1_l26_e32_parity_v2.json"
MODES = ("pp4", "pp1")


def budget_hours(profile):
    hours, minutes, seconds = map(int, profile["time_limit"].split(":"))
    gpu_count = profile["nodes"] * profile["gpus_per_node"]
    result = gpu_count * (hours + minutes / 60 + seconds / 3600)
    consumed = profile.get("consumed_gpu_hours", 0)
    if ((profile["nodes"], profile["gpus_per_node"]) != (2, 4)
            or not 0 <= minutes < 60 or not 0 <= seconds < 60
            or hours < 0 or result <= 0
            or not 0 <= consumed < 4
            or not 0 < result + consumed <= profile["gpu_hours_cap"] <= 4):
        raise ValueError("parity allocation exceeds approved eight-GPU/four-hour contract")
    return result


def mode_spec(original, output, mode, languages):
    if mode not in MODES or set(languages) != {"eng_Latn", "nno_Latn"} or len(languages) != 2:
        raise ValueError("only the approved English/Nynorsk PP4/PP1 diagnostic is supported")
    spec = copy.deepcopy(original)
    spec.update(run_id=f"l26_e32_parity_{mode}", output_dir=str(output), diagnostic_languages=list(languages))
    recipe = spec["model_recipe"]
    if recipe["expected_model"]["pipeline_model_parallel_size"] != 4:
        raise ValueError("reference must be the original PP4 evaluation")
    if spec.get("require_moe_routing_diagnostics") or spec.get("routing_diagnostics", {}).get("enabled"):
        raise ValueError("reference must use the original loss-only evaluation path")
    if mode == "pp1":
        changes = {"model.pipeline_model_parallel_size": "1", "model.pipeline_model_parallel_layout": "null"}
        if not set(changes).issubset(x.split("=", 1)[0] for x in recipe["overrides"]):
            raise ValueError("missing explicit PP4 topology overrides")
        recipe["overrides"] = [f"{key}={changes[key]}" if (key := x.split("=", 1)[0]) in changes else x
                               for x in recipe["overrides"]]
        recipe["expected_model"]["pipeline_model_parallel_size"] = 1
    return spec


def loss_comparison(reference, observed, tolerance):
    """Require identical scored tokens and normalization, allowing only NLL roundoff."""
    if not 0 < tolerance <= 1e-4:
        raise ValueError("parity tolerance must be positive and at most 1e-4 nats")
    for key in ("language", "token_count", "target_token_ids_sha256", "unigram_counts_sha256",
                "unigram_log_probability_sum", "vocab_size", "special_token_ids", "unseen_target_token_count"):
        if reference[key] != observed[key]:
            raise ValueError(f"measurement identity mismatch: {key}")
    for key in ("checkpoint_step", "sequence_length", "eval_micro_batch_size", "validation_prefixes",
                "stream_tokens_including_special", "loss_statistic"):
        if reference["backend"][key] != observed["backend"][key]:
            raise ValueError(f"backend identity mismatch: {key}")
    for item in (reference, observed):
        n = item["token_count"]
        if n <= 0 or n != item["backend"]["backend_token_count"]:
            raise ValueError("invalid token denominator")
        if not math.isfinite(item["raw_nll"]) or not math.isclose(
                item["raw_nll"], item["nll_sum"] / n, rel_tol=0, abs_tol=1e-12):
            raise ValueError("invalid NLL sum or mean")
        adjusted = item["raw_nll"] + item["unigram_log_probability_sum"] / n
        if not math.isclose(adjusted, item["atlas_tao_unigram_adjusted_loss"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("invalid adjusted loss")
    delta = observed["raw_nll"] - reference["raw_nll"]
    return {"reference_raw_nll": reference["raw_nll"], "observed_raw_nll": observed["raw_nll"],
            "delta_nats": delta, "absolute_delta_nats": abs(delta), "tolerance_nats": tolerance,
            "token_count": observed["token_count"], "passed": abs(delta) <= tolerance}


def checkpoint_inventory(path):
    return [{"name": p.name, "size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(path.iterdir()) if p.is_file()]


def prior_usage(root, cfg, slurm_bin):
    previous = root / cfg["previous_attempt"]
    submission = read(previous / "submission.json")
    job = submission["stdout"].split(";")[0]
    if submission["returncode"] or not job.isdigit():
        raise ValueError("invalid previous submission receipt")
    stdout = subprocess.check_output([str(Path(slurm_bin)/"sacct"), "-j", job, "-X", "-n", "-P",
        "--starttime", cfg["accounting_start"], "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"], text=True)
    rows = [line.split("|") for line in stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != job or rows[0][1] not in ("FAILED", "COMPLETED", "TIMEOUT", "OUT_OF_MEMORY"):
        raise ValueError("previous job must have unambiguous terminal accounting")
    tres = dict(item.split("=", 1) for item in rows[0][3].split(","))
    gpus, seconds = int(tres["gres/gpu"]), int(rows[0][2])
    return {"job_id": job, "state": rows[0][1], "gpus": gpus, "elapsed_seconds": seconds,
            "gpu_hours": gpus*seconds/3600, "stdout": stdout, "submission": identity(previous/"submission.json")}


def prepare(root, profile_path):
    cfg = read(profile_path)
    base = (root / cfg["base_evaluation"]).resolve()
    parent = read(base / "plan.json")
    release = read(base / "release.json")
    if release["status"] != "complete":
        raise ValueError("original evaluation is not complete")
    verify_files([release["plan"], release["evaluation"], parent["spec"]])
    manifest = read(release["evaluation"]["path"])
    references = {lang: manifest["languages"][lang] for lang in cfg["languages"]}
    verify_files(list(references.values()))
    out = (root / cfg["output"]).resolve()
    out.relative_to(root / "outputs")
    profile = {**parent["profile"], **cfg}
    usage = prior_usage(root, cfg, profile["slurm_bin"])
    profile["consumed_gpu_hours"] = usage["gpu_hours"]
    allocation(profile, parent["runtime"])
    ceiling = budget_hours(profile)
    if not 0 < cfg["raw_nll_absolute_tolerance"] <= 1e-4:
        raise ValueError("invalid predeclared loss tolerance")
    original = read(parent["spec"]["path"])
    specs = {mode: mode_spec(original, out / mode, mode, cfg["languages"]) for mode in MODES}
    out.mkdir(parents=True, exist_ok=False)
    write(out / "prior_accounting.json", usage)
    source = out / "source"
    old_source = base / "source"
    provenance = read(old_source / "source_manifest.json")
    verify_files(provenance["files"])
    shutil.copytree(old_source, source, symlinks=True,
                    ignore=shutil.ignore_patterns("source_manifest.json", "__pycache__"))
    overlays = (SOURCE, EVALUATION_SOURCE, EVALUATOR, PROFILE, LAUNCH_SOURCE, PACKAGE_SOURCE, EXPOSURE_SOURCE, INFERENCE)
    for relative in overlays:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.chmod(0o644)
        shutil.copyfile(root / relative, target)
    files = [identity(p) for folder in ("src", "configs", "external")
             for p in sorted((source / folder).rglob("*")) if p.is_file()]
    write(source / "source_manifest.json", {key: provenance[key] for key in (
        "base_commit", "bridge_commit", "megatron_commit")} | {
        "parent_manifest": identity(old_source / "source_manifest.json"),
        "submission_checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "overlays": list(overlays), "files": files})
    for p in source.rglob("*"):
        if p.is_file() and not p.is_symlink():
            p.chmod(p.stat().st_mode & ~0o222)
    for mode, spec in specs.items():
        write(out / f"{mode}_spec.json", spec)
    from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import load_evaluation_spec
    parsed = [load_evaluation_spec(out / f"{mode}_spec.json") for mode in MODES]
    checkpoint = parsed[0].checkpoint_iteration_path
    inputs = [identity(out/"prior_accounting.json"), usage["submission"], identity(base / "release.json"), release["plan"], release["evaluation"], parent["spec"],
              *references.values(), *[identity(checkpoint / name) for name in (
                  ".metadata", "common.pt", "train_state.pt", "run_config.yaml")]]
    plan = {"schema": "l26_e32_checkpoint_parity/v1", "root": str(root), "output": str(out),
            "attempt_id": uuid4().hex,
            "profile": profile, "runtime": parent["runtime"], "inputs": inputs, "reference_losses": references,
            "specs": {mode: identity(out / f"{mode}_spec.json") for mode in MODES},
            "checkpoint": str(checkpoint), "checkpoint_inventory": checkpoint_inventory(checkpoint),
            "allocated_gpu_hours_ceiling": ceiling, "comparison_scope": "diagnostic_only_two_languages_no_retraining"}
    write(out / "plan.json", plan)
    return out / "plan.json"


def verify(plan):
    out = Path(plan["output"])
    budget_hours(plan["profile"])
    verify_files(plan["inputs"] + list(plan["specs"].values()))
    verify_files(read(out / "source/source_manifest.json")["files"])
    if checkpoint_inventory(Path(plan["checkpoint"])) != plan["checkpoint_inventory"]:
        raise ValueError("checkpoint files changed since preparation")


def config_check(plan):
    """Exercise both actual recipe configurations without creating a GPU model."""
    from multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe import (
        _load_recipe, _apply_dataset_override, _apply_cli_overrides,
    )
    from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import (
        load_evaluation_spec, configure_checkpoint_evaluation,
    )
    from torch.distributed.checkpoint import FileSystemReader
    metadata = FileSystemReader(plan["checkpoint"]).read_metadata()
    for mode, entry in plan["specs"].items():
        spec = load_evaluation_spec(Path(entry["path"]))
        config = _load_recipe(spec.recipe_name, None, False, spec.sequence_length, None)
        config = _apply_dataset_override(config, "llm-pretrain", spec.sequence_length)
        config = _apply_cli_overrides(config, list(spec.recipe_overrides))
        configure_checkpoint_evaluation(config, spec)
        if (not config.validation.skip_train or config.checkpoint.save is not None
                or config.checkpoint.load_rng or config.checkpoint.load_optim):
            raise ValueError("evaluation must not train, save, or load optimizer/RNG")
        print(f"[CONFIG_OK] mode={mode} PP={config.model.pipeline_model_parallel_size} "
              f"EP={config.model.expert_model_parallel_size} checkpoint={spec.checkpoint_step} "
              f"metadata_entries={len(metadata.state_dict_metadata)}", flush=True)


def preflight(plan_path):
    plan = read(plan_path)
    verify(plan)
    out = Path(plan["output"])
    work = out / "runtime/preflight"
    env = execution_env(plan, work)
    python = str(Path(plan["runtime"]["module_runtime"]["conda_env"]) / "bin/python")
    subprocess.run([python, "-m", MODULE, "config-check", "--plan", str(plan_path)],
                   cwd=work, env=env, check=True, timeout=240)
    write(out / "preflight.json", {"status": "passed", "plan": identity(plan_path),
          "source_manifest": identity(out / "source/source_manifest.json"),
          "scope": "frozen_imports_specs_recipe_contracts_and_checkpoint_metadata_not_gpu_load"})


def submit(plan_path):
    plan = read(plan_path)
    verify(plan)
    out = Path(plan["output"])
    (out / "submission.lock").mkdir()
    logs = attempt_logs(out, plan["attempt_id"])
    logs.mkdir(parents=True)
    script, preflight_script = out / "parity.slurm.sh", out / "preflight.sh"
    script.write_text(render_script(plan_path, plan, "run", module=MODULE))
    preflight_script.write_text(render_script(plan_path, plan, "preflight", module=MODULE))
    subprocess.run(["bash", "-n", str(script)], check=True)
    with (logs / "preflight.log").open("w") as log:
        subprocess.run(["bash", str(preflight_script)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    write(out / "launch_identity.json", {"plan": identity(plan_path), "script": identity(script),
          "source_manifest": identity(out / "source/source_manifest.json"), "preflight": identity(out / "preflight.json")})
    cmd = submission_command(plan, script, job_name="l26_e32_pp_parity")
    cmd.insert(1, "--no-requeue")
    result = subprocess.run(cmd, capture_output=True, text=True)
    write(out / "submission.json", {"returncode": result.returncode, "stdout": result.stdout.strip(),
          "stderr": result.stderr.strip(), "command": cmd, "plan": identity(plan_path)})
    result.check_returncode()
    print("Submitted parity job", result.stdout.strip(), flush=True)


def mode_results(plan, mode):
    spec = read(plan["specs"][mode]["path"])
    path = Path(spec["output_dir"]) / "diagnostic_manifest.json"
    manifest = read(path)
    if (manifest["status"] != "complete_diagnostic" or manifest["spec"] != plan["specs"][mode]
            or manifest["checkpoint"]["step"] != spec["checkpoint"]["step"]
            or manifest["checkpoint"]["root"] != spec["checkpoint"]["path"]
            or manifest["model_contract"] != spec["model_recipe"]["expected_model"]
            or set(manifest["languages"]) != set(plan["profile"]["languages"])
            or manifest["coverage"]["complete_full_evaluation"]):
        raise ValueError("invalid or incomplete diagnostic receipt")
    verify_files(list(manifest["languages"].values()))
    return {lang: read(entry["path"]) for lang, entry in manifest["languages"].items()}, identity(path)


def run(plan_path, mode=None):
    plan = read(plan_path)
    out, cfg = Path(plan["output"]), plan["profile"]
    verify_files(list(read(out / "launch_identity.json").values()))
    if mode is not None:
        node = os.environ["SLURM_PROCID"]
        work = out / "runtime" / mode / node
        env = execution_env(plan, work)
        master = subprocess.check_output([str(Path(cfg["slurm_bin"]) / "scontrol"), "show", "hostnames",
                                          os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()[0]
        spec = read(plan["specs"][mode]["path"])
        cmd = worker_command({**plan, "spec": plan["specs"][mode]}, spec, node, master,
                             int(os.environ["SLURM_JOB_ID"])+MODES.index(mode))
        subprocess.run(cmd, cwd=work, env=env, check=True)
        return
    verify(plan)
    print(f"[PARITY_START] job={os.environ['SLURM_JOB_ID']} nodes={os.environ['SLURM_JOB_NODELIST']} "
          f"gpus=8 gpu_hours_ceiling={plan['allocated_gpu_hours_ceiling']}", flush=True)
    results = {"original": {lang: read(entry["path"]) for lang, entry in plan["reference_losses"].items()}}
    comparisons, receipts = {}, {}
    started = time.monotonic()
    try:
        for current in MODES:
            verify(plan)
            if Path(read(plan["specs"][current]["path"])["output_dir"]).exists():
                raise ValueError(f"refusing to overwrite prior diagnostic outputs: {current}")
            print(f"[PARITY_MODE_START] {current}", flush=True)
            cmd = step_command(plan_path, plan, module=MODULE)+["--mode", current]
            cmd.insert(1, "--kill-on-bad-exit=1")
            subprocess.run(cmd, check=True)
            results[current], receipts[current] = mode_results(plan, current)
            for previous in ("original", "pp4") if current == "pp1" else ("original",):
                name = f"{current}_vs_{previous}"
                comparisons[name] = {lang: loss_comparison(results[previous][lang], results[current][lang],
                    cfg["raw_nll_absolute_tolerance"]) for lang in cfg["languages"]}
                write(out / f"{name}.json", comparisons[name])
                if not all(row["passed"] for row in comparisons[name].values()):
                    raise ValueError(f"measurement parity failed: {name}")
            print(f"[PARITY_MODE_PASSED] {current}", flush=True)
        verify(plan)
    except Exception as exc:
        write(out / "failure.json", {"status": "failed", "reason": str(exc), "comparisons": comparisons,
              "plan": identity(plan_path), "job_id": os.environ["SLURM_JOB_ID"]})
        raise
    write(out / "parity_report.json", {"status": "passed", "scope": "two_language_measurement_consistency_only",
          "plan": identity(plan_path), "job_id": os.environ["SLURM_JOB_ID"], "receipts": receipts,
          "comparisons": comparisons, "wall_seconds_inside_driver": time.monotonic()-started,
          "training_steps": 0, "all_language_parity_proven": False})
    print("[PARITY_PASSED] English/Nynorsk only; no training performed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "preflight", "config-check", "run", "rank"))
    parser.add_argument("--profile", type=Path, default=Path(PROFILE))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(find_repo_root(), args.profile.resolve()))
    elif args.action == "config-check":
        config_check(read(args.plan))
    elif args.action == "submit":
        submit(args.plan.resolve())
    elif args.action == "preflight":
        preflight(args.plan.resolve())
    else:
        if args.action == "rank" and args.mode is None:
            parser.error("rank requires --mode")
        run(args.plan.resolve(), mode=args.mode if args.action == "rank" else None)


if __name__ == "__main__":
    main()
