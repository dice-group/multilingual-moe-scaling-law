"""Read-only artifact audit for the L26/E32 prediction diagnosis."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import yaml

from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import identity
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import read
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import verify_files
from multilingual_megatron.pipeline.stages.training.metrics import parse_training_log_metrics


def require_equal_stream(old, new):
    keys = ("target_token_ids_sha256", "unigram_counts_sha256", "special_token_ids",
            "token_count", "vocab_size", "unigram_log_probability_sum")
    for key in keys:
        if key not in old or key not in new or old[key] != new[key]:
            raise ValueError(f"validation evidence missing or different: {key}")


def audit_resume(receipts):
    if len(receipts) != 8 or {r.get("rank") for r in receipts} != set(range(8)):
        raise ValueError("requires eight distinct rank resume receipts")
    expected = {"step": 217233, "consumed_train_samples": 26829184, "scheduler_num_steps": 26829184}
    for r in receipts:
        if r["status"] != "verified_before_first_update" or r["expected"] != expected or r["observed"] != expected:
            raise ValueError("wrong restored parent state")
        if Path(r["checkpoint_path"]).name != "iter_0217233" or "excluded" in Path(r["checkpoint_path"]).parts:
            raise ValueError("incorrect or excluded cooldown parent")


def audit(comparison, grid_release):
    bindings, ledger = [], []
    def bound(path, expected=None):
        item = identity(path)
        if expected is not None and item["sha256"] != expected:
            raise ValueError(f"changed audit input: {path}")
        bindings.append(item)
        return Path(path)
    def add(check, status, evidence):
        ledger.append(dict(check=check, status=status, evidence=evidence))
    manifest = read(bound(comparison/"manifest.json"))
    verify_files(manifest["inputs"]+manifest["outputs"]+[manifest["implementation"]])
    bindings.extend(manifest["inputs"]+manifest["outputs"])
    evaluation_release, = [Path(x["path"]) for x in manifest["inputs"] if Path(x["path"]).name == "release.json"]
    release = read(evaluation_release)
    verify_files([release[k] for k in ("evaluation", "observations", "plan")])
    plan = read(bound(release["plan"]["path"], release["plan"]["sha256"]))
    verify_files(plan["inputs"]+[plan["spec"]])
    bindings.extend(plan["inputs"]+[plan["spec"]])
    new = read(bound(release["evaluation"]["path"], release["evaluation"]["sha256"]))
    grid_manifest = read(bound(grid_release/"release_manifest.json"))
    table = grid_manifest["table_inventory"]["readouts"]
    table_path = grid_release/table["path"]
    # Table identity is independently bound by the Stage-1 release.
    readouts = pd.read_parquet(bound(table_path, table["sha256"]))
    row = readouts.loc[readouts.readout_id == "prod_eval_d_e32_k200_d32000000000"].iloc[0]
    old = read(bound(row.evaluation_manifest_path, row.evaluation_manifest_sha256))
    verify_files(list(old["languages"].values())+list(new["languages"].values()))
    bindings.extend(list(old["languages"].values())+list(new["languages"].values()))
    if set(old["languages"]) != set(new["languages"]) or len(new["languages"]) != 200:
        raise ValueError("requires matched 200-language observations")
    for language in sorted(new["languages"]):
        require_equal_stream(read(old["languages"][language]["path"]), read(new["languages"][language]["path"]))
    add("validation_targets", "equivalent", "All 200 target-token hashes, unigram hashes/sums, special masks, vocabularies and denominators agree exactly.")
    ospec = read(bound(old["spec"]["path"], old["spec"]["sha256"]))
    nspec = read(bound(new["spec"]["path"], new["spec"]["sha256"]))
    for key in ("sequence_length", "eval_micro_batch_size", "context_token_id", "training_seed",
                "validation_data_manifest", "corpus_manifest"):
        if ospec[key] != nspec[key]:
            raise ValueError(f"evaluation contract differs: {key}")
    add("validation_contract", "equivalent", "Sequence 2048, microbatch 8, context token 2, seed and frozen validation/corpus manifests match.")
    add("routing_evidence", "unresolved", "Historical diagnostics enabled; L26 PP4 diagnostics disabled. Matching masks/counts are not a routing or cross-topology logit proof.")
    checkpoint = Path(nspec["checkpoint"]["path"])/f"iter_{plan['expected_step']:07d}"
    segment = checkpoint.parent.parent
    dc = yaml.safe_load(bound(Path(row.checkpoint_iteration_path)/"run_config.yaml").read_text())
    lc = yaml.safe_load(bound(checkpoint/"run_config.yaml").read_text())
    differences = []
    for section in ("model", "optimizer", "dataset", "scheduler", "rng", "tokenizer", "train"):
        for key in sorted(set(dc[section])|set(lc[section])):
            a, b = dc[section].get(key), lc[section].get(key)
            if a != b:
                differences.append(dict(section=section, key=key, grid=json.dumps(a), l26=json.dumps(b)))
    if dc["dataset"]["blend_per_split"] != lc["dataset"]["blend_per_split"]:
        raise ValueError("training blend or weights changed")
    if dc["rng"] != lc["rng"] or dc["tokenizer"] != lc["tokenizer"]:
        raise ValueError("training seed or tokenizer changed")
    add("training_data_seed", "equivalent", "Saved training prefixes/order/weights, tokenizer and RNG config are identical to D/E32/K200.")
    optimizer_differences = [r["key"] for r in differences if r["section"] == "optimizer"]
    optimizer_status = "equivalent_except_size_lr" if optimizer_differences == ["lr"] else "potentially_consequential"
    add("optimizer", optimizer_status, f"Saved optimizer differences: {optimizer_differences}; this checks recipe execution, not LR optimality.")
    add("architecture_and_execution", "potentially_consequential", "L26 has 26x1664 vs D 20x1280, same 3x FFN and head dimension. PP4/EP2 selective recompute vs PP1/EP4 expert overlap; cross-topology equivalence is unmeasured.")
    receipts = [read(bound(segment/f"restored_resume_state.rank{i}.json")) for i in range(8)]
    audit_resume(receipts)
    final_audit = read(bound(segment/"checkpoint_state_audit.json"))
    if not final_audit["valid_full_state"] or final_audit["train_state_step"] != 269634 or final_audit["scheduler_num_steps"] != 33536512:
        raise ValueError("final full-state audit fails")
    if not (segment/".training_complete").is_file():
        raise ValueError("cooldown completion marker missing")
    add("cooldown_lineage", "verified", "All eight rank receipts restore step 217233/samples 26829184; completed final audit is step 269634/samples 33536512. Excluded failed attempt is not the parent.")
    campaign = Path(plan["root"])/plan["profile"]["campaign_root"]
    logs = sorted((campaign/"trunk/runs").glob("*/segments/*.train.log"))+[segment.parent/(segment.name+".train.log")]
    log_rows, segment_rows = [], []
    for path in logs:
        bound(path)
        metrics = parse_training_log_metrics(path, seq_length=2048, warmup_iters=0)
        log_rows.append(dict(log=str(path), **{k:metrics.get(k) for k in (
            "iteration_lines", "last_iteration", "last_consumed_samples", "finite_loss_and_gradients",
            "max_skipped_iterations", "max_nan_iterations", "nonfinite_lm_loss_count", "nonfinite_grad_norm_count")}))
        directory = path.with_name(path.name.removesuffix(".train.log"))
        state = read(bound(directory/"checkpoint_state_audit.json"))
        invocation = read(bound(directory/"bridge_invocation.json"))
        config = yaml.safe_load(bound(Path(state["checkpoint_path"])/"run_config.yaml").read_text())
        matches = re.findall(r"learning rate:\s*([0-9.eE+\-]+)", path.read_text())
        segment_rows.append(dict(segment=str(directory), step=state["train_state_step"],
            samples=state["consumed_train_samples"], scheduler_samples=state["scheduler_num_steps"],
            global_batch_size=invocation["global_batch_size"], configured_lr=config["optimizer"]["lr"],
            logged_peak_lr=max(map(float,matches)) if matches else None,
            logged_last_lr=float(matches[-1]) if matches else None,
            checkpoint_load=config["checkpoint"]["load"], load_optim=config["checkpoint"].get("load_optim"),
            load_rng=config["checkpoint"].get("load_rng"), source_cwd=invocation["cwd"]))
    good = len(log_rows) == 8 and all(r["finite_loss_and_gradients"] and r["max_skipped_iterations"] == 0 and r["max_nan_iterations"] == 0 for r in log_rows)
    previous_step = previous_samples = 0
    for r in segment_rows:
        if r["samples"]-previous_samples != (r["step"]-previous_step)*r["global_batch_size"] or r["scheduler_samples"] != r["samples"]:
            raise ValueError("step/sample/scheduler discontinuity across segments")
        if previous_step and (not r["load_optim"] or not r["load_rng"]):
            raise ValueError("resume disabled optimizer or RNG restoration")
        previous_step, previous_samples = r["step"], r["samples"]
    add("segment_continuity", "verified", "All eight step/sample deltas match their observed GBS, scheduler counts equal samples, and every resumed segment enables optimizer/RNG loading.")
    add("logged_training_health", "verified_logged_samples" if good else "unresolved", f"{len(log_rows)} production segments; finite loss/gradients and zero NaN/skip counters: {good}. Logged samples are not every optimizer update.")
    add("schedule", "matched_policy_not_optimality", f"Saved warmup steps {dc['scheduler']['lr_warmup_steps']} / {lc['scheduler']['lr_warmup_steps']}; WSD cooldown fractions {dc['scheduler']['wsd_decay_steps']/dc['scheduler']['lr_decay_steps']:.8f} / {lc['scheduler']['wsd_decay_steps']/lc['scheduler']['lr_decay_steps']:.8f}. LRs {dc['optimizer']['lr']} / {lc['optimizer']['lr']}; the larger model uses the smaller prescribed LR, not an experimentally proven optimum.")
    source = Path(plan["output"])/"source"
    code_paths = [source/"src/multilingual_megatron/pipeline/stages/final_loss/per_language_checkpoint_eval.py",
                  source/"external/Megatron-Bridge/src/megatron/bridge/training/eval.py"]
    for path in code_paths:
        bound(path)
    add("evaluation_code_review", "static_only", "Frozen stage constructs shifted causal targets and excludes padding/special targets. Bridge calls eval()/no_grad(), accumulates float64 sufficient statistics; stage broadcasts DP-reduced statistics from a pipeline-last rank. No independent L26 PP4 numerical parity control exists.")
    return dict(ledger=ledger, config_differences=differences, segment_states=segment_rows,
                training_log_health=log_rows, bindings=bindings, new_spec=nspec, old_spec=ospec,
                checkpoint=str(checkpoint), grid_checkpoint=row.checkpoint_iteration_path)
