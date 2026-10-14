from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...runtime import _abs, _build_run_label, _deep_merge, _dump_json, _log_path_component, _structured_logs_enabled
from ...tokenizer_manifest import resolve_tokenized_entry

from .config import as_positive_int, is_bridge_backend, resolve_train_iters, resolve_training_stage_config
from .batch_policy import (
    actual_tokens_from_segments,
    build_joint_moe_full_state_wsd_v2_plan,
    final_cumulative_step,
    is_joint_moe_full_state_wsd_v2,
    joint_moe_full_state_wsd_v2_scheduler_accounting,
    materialize_batch_segments,
    resolve_batch_policy,
    scheduler_iter_for_tokens,
)


TRAINING_MANIFEST_NAME = "training_manifest.json"
TRAINING_RUN_MANIFEST_NAME = "training_run_manifest.json"
TRAINING_CALIBRATION_NAME = "training_calibration.json"
TRAINING_CALIBRATION_TSV_NAME = "training_calibration.tsv"
TRAINING_CALIBRATIONS_NAME = "training_calibrations.json"
SEGMENT_PLAN_NAME = "segment_plan.json"

PRESPLIT_TRAINING_MANIFEST_PURPOSES = frozenset(
    {
        "nested_K_training_with_global_validation",
        "language_composition_K12_training_with_global_K196_validation",
    }
)


def _combined_prefix_for_partition(prefix: str) -> str | None:
    marker = "_text_document"
    if not prefix.endswith(marker):
        return None
    stem = prefix[: -len(marker)]
    base, separator, partition = stem.rpartition("_")
    if not separator or not partition.isdigit():
        return None
    combined = f"{base}{marker}"
    if Path(f"{combined}.bin").exists() and Path(f"{combined}.idx").exists():
        return combined
    return None


def _bounded_megatron_payload(payload: dict[str, Any], max_prefixes: int) -> tuple[dict[str, Any], bool]:
    prefixes = [str(item) for item in payload.get("prefixes", []) if str(item).strip()]
    if not prefixes:
        return dict(payload), False

    raw_keep = prefixes[:max_prefixes]
    keep = list(raw_keep)
    combined_by_partition: dict[str, str] = {}
    if max_prefixes == 1:
        combined = _combined_prefix_for_partition(keep[0])
        if combined:
            combined_by_partition[keep[0]] = combined
            keep = [combined]

    if keep == prefixes:
        return dict(payload), False

    keep_set = set(keep)
    bounded = dict(payload)
    bounded["prefixes"] = keep
    weights = payload.get("weights")
    if isinstance(weights, list):
        bounded["weights"] = None if len(keep) == 1 else [float(item) for item in weights[:max_prefixes]]
    prefixes_by_language = payload.get("prefixes_by_language")
    if isinstance(prefixes_by_language, dict):
        filtered: dict[str, list[str]] = {}
        for language, values in prefixes_by_language.items():
            kept_values = []
            for item in values:
                value = str(item)
                if value in keep_set:
                    kept_values.append(value)
                elif value in combined_by_partition:
                    kept_values.append(combined_by_partition[value])
            if kept_values:
                filtered[str(language)] = kept_values
        bounded["prefixes_by_language"] = filtered
        bounded["languages"] = list(filtered)
        docs = payload.get("documents_per_language")
        if isinstance(docs, dict):
            bounded["documents_per_language"] = {language: docs.get(language, 0) for language in filtered}
    return bounded, True


def _write_bounded_megatron_manifest(data_manifest: Path, output_dir: Path, max_prefixes: int | None) -> Path:
    if max_prefixes is None or max_prefixes <= 0 or not data_manifest.exists():
        return data_manifest

    payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    bounded, changed = _bounded_megatron_payload(payload, max_prefixes)
    splits = payload.get("splits")
    if isinstance(splits, dict):
        bounded_splits: dict[str, Any] = {}
        for split_name, split_payload in splits.items():
            if isinstance(split_payload, dict):
                bounded_split, split_changed = _bounded_megatron_payload(split_payload, max_prefixes)
                bounded_splits[str(split_name)] = bounded_split
                changed = changed or split_changed
            else:
                bounded_splits[str(split_name)] = split_payload
        if changed:
            bounded["splits"] = bounded_splits
    if not changed:
        return data_manifest

    bounded["source_data_manifest"] = str(data_manifest.resolve())
    bounded["bounded_max_prefixes"] = int(max_prefixes)

    output_dir.mkdir(parents=True, exist_ok=True)
    bounded_path = output_dir / "megatron_data_manifest.training_subset.json"
    _dump_json(bounded_path, bounded)
    return bounded_path


def _requires_presplit_train_only_mode(data_manifest: Path) -> bool:
    """Return true for frozen pre-split manifests and validate their train view."""

    if not data_manifest.is_file():
        return False
    payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != "fineweb_ab_v2_megatron_data/v1":
        return False
    if payload.get("purpose") not in PRESPLIT_TRAINING_MANIFEST_PURPOSES:
        raise SystemExit("A/B-v2 training requires a recognized pre-split training data manifest")
    splits = payload.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    if not isinstance(train, dict) or not train.get("prefixes"):
        raise SystemExit("A/B-v2 data manifest lacks its finalized train split")
    if payload.get("prefixes") != train.get("prefixes") or payload.get("weights") != train.get("weights"):
        raise SystemExit("A/B-v2 top-level blend differs from its finalized train split")
    return True


def _configured_override_values(common: dict[str, Any], run_cfg: dict[str, Any], key: str) -> list[str]:
    values: list[str] = []
    for source in (common.get("overrides", []), run_cfg.get("overrides", [])):
        if not isinstance(source, list):
            continue
        for raw in source:
            text = str(raw)
            field, separator, value = text.partition("=")
            if separator and field.lstrip("+").strip() == key:
                values.append(value.strip())
    return values


@dataclass(frozen=True, slots=True)
class TrainingRunPlan:
    run_cfg: dict[str, Any]
    run_id: str
    run_name: str
    output_dir: Path
    log_path: Path
    cmd: list[str]
    env: dict[str, str]
    tokenization: dict[str, Any] | None
    metadata: dict[str, Any]

    @property
    def expected_artifacts(self) -> tuple[Path, ...]:
        return (
            self.output_dir / "train_results.json",
            self.output_dir / "trainer_state.json",
            self.output_dir / ".training_complete",
            self.output_dir / TRAINING_RUN_MANIFEST_NAME,
        )


def _bridge_training_cmd(
    *,
    repo_root: Path,
    env: dict[str, str],
    output_dir: Path,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    tokenized_path: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    env = dict(env)
    bridge_root = _abs(repo_root, str(common.get("bridge_root", "external/Megatron-Bridge")))
    train = _deep_merge(common.get("train", {}), run_cfg.get("train", {}))
    eval_cfg = _deep_merge(common.get("eval", {}), run_cfg.get("eval", {}))

    recipe = str(run_cfg.get("recipe", common.get("recipe", "vanilla_gpt_pretrain_config")))
    dataset = str(run_cfg.get("dataset", common.get("dataset", "llm-pretrain")))
    nproc_per_node = int(run_cfg.get("nproc_per_node", common.get("nproc_per_node", 1)))
    nnodes = int(run_cfg.get("nnodes", common.get("nnodes", 1)))
    seq_length = int(run_cfg.get("seq_length", common.get("seq_length", train.get("seq_length", 512))))
    global_batch_size = int(train.get("global_batch_size", train.get("global_batch", 4)))
    micro_batch_size = int(train.get("micro_batch_size", train.get("micro_batch", 1)))
    train_iters = resolve_train_iters(
        train=train,
        common=common,
        global_batch_size=global_batch_size,
        seq_length=seq_length,
    )
    eval_interval = int(eval_cfg.get("eval_interval", eval_cfg.get("eval_steps", max(1, train_iters))))
    eval_iters = int(eval_cfg.get("eval_iters", 1))
    log_interval = int(train.get("log_interval", train.get("logging_steps", 1)))
    expected_tokens = int(train_iters) * int(global_batch_size) * int(seq_length)
    pipeline_python = str(env.get("PYTHON_EXECUTABLE", sys.executable)).strip() or sys.executable
    python_executable_value = run_cfg.get("python_executable", common.get("python_executable"))
    python_executable = str(python_executable_value).strip() if python_executable_value is not None else pipeline_python
    if not python_executable:
        python_executable = pipeline_python
    python_executable_path = Path(python_executable)
    if not python_executable_path.is_absolute() and (repo_root / python_executable_path).exists():
        python_executable = str(repo_root / python_executable_path)

    cmd = [
        pipeline_python,
        "-m",
        "multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training",
        "--bridge-root",
        str(bridge_root),
        "--output-dir",
        str(output_dir),
        "--recipe",
        recipe,
        "--dataset",
        dataset,
        "--nproc-per-node",
        str(nproc_per_node),
        "--nnodes",
        str(nnodes),
        "--seq-length",
        str(seq_length),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        str(eval_interval),
        "--eval-iters",
        str(eval_iters),
        "--global-batch-size",
        str(global_batch_size),
        "--micro-batch-size",
        str(micro_batch_size),
        "--log-interval",
        str(log_interval),
        "--launcher",
        str(run_cfg.get("launcher", common.get("launcher", "python"))),
        "--python-executable",
        python_executable,
    ]
    save_interval = as_positive_int(
        train.get(
            "save_interval",
            train.get(
                "save_interval_iters",
                common.get("save_interval", common.get("save_interval_iters")),
            ),
        )
    )
    if save_interval is not None:
        cmd.extend(["--save-interval", str(save_interval)])
    expected_end_step = as_positive_int(train.get("expected_end_step"))
    expected_consumed_samples = as_positive_int(train.get("expected_consumed_samples"))
    require_full_state_checkpoint = bool(train.get("require_full_state_checkpoint", False))
    if require_full_state_checkpoint:
        if expected_end_step is None or expected_consumed_samples is None:
            raise SystemExit(
                "train.require_full_state_checkpoint=true requires expected_end_step "
                "and expected_consumed_samples"
            )
        cmd.extend(
            [
                "--expected-end-step",
                str(expected_end_step),
                "--expected-consumed-samples",
                str(expected_consumed_samples),
                "--require-full-state-checkpoint",
            ]
        )

    launch_cfg = _deep_merge(common.get("launch", {}), run_cfg.get("launch", {}))
    if bool(launch_cfg.get("preflight_cuda", False)):
        cmd.append("--preflight-cuda")
    if bool(launch_cfg.get("nsys_profile", False)):
        cmd.append("--nsys-profile")
        nsys_output_dir = str(launch_cfg.get("nsys_output_dir", "")).strip()
        if nsys_output_dir:
            cmd.extend(["--nsys-output-dir", nsys_output_dir])
        nsys_bin = str(launch_cfg.get("nsys_bin", "")).strip()
        if nsys_bin:
            cmd.extend(["--nsys-bin", nsys_bin])
        nsys_trace = launch_cfg.get("nsys_trace", [])
        if isinstance(nsys_trace, list) and nsys_trace:
            cmd.extend(["--nsys-trace", ",".join(str(item) for item in nsys_trace if str(item).strip())])
        elif str(nsys_trace).strip():
            cmd.extend(["--nsys-trace", str(nsys_trace).strip()])
        nsys_extra_args = launch_cfg.get("nsys_extra_args", [])
        if isinstance(nsys_extra_args, list):
            for item in nsys_extra_args:
                if str(item).strip():
                    cmd.append(f"--nsys-extra-arg={item}")
    srun_args = launch_cfg.get("srun_args", [])
    if isinstance(srun_args, list):
        for item in srun_args:
            if str(item).strip():
                cmd.append(f"--srun-arg={item}")

    hf_path = str(run_cfg.get("hf_path", common.get("hf_path", ""))).strip()
    if hf_path:
        cmd.extend(["--hf-path", hf_path])

    tokenized_entry = resolve_tokenized_entry(tokenized_path, common, run_cfg)
    data_manifest = None
    max_manifest_prefixes = None
    presplit_train_only = False
    if dataset == "llm-pretrain":
        manifest_override = str(
            run_cfg.get(
                "data_manifest",
                run_cfg.get(
                    "megatron_data_manifest",
                    common.get("data_manifest", common.get("megatron_data_manifest", "")),
                ),
            )
        ).strip()
        data_manifest = tokenized_path / "megatron_data_manifest.json"
        if tokenized_entry is not None:
            data_manifest = Path(str(tokenized_entry["data_manifest"]))
        if manifest_override:
            data_manifest = _abs(repo_root, manifest_override)
        max_manifest_prefixes = as_positive_int(
            run_cfg.get(
                "data_manifest_max_prefixes",
                common.get("data_manifest_max_prefixes", common.get("max_data_manifest_prefixes")),
            )
        )
        data_manifest = _write_bounded_megatron_manifest(data_manifest, output_dir, max_manifest_prefixes)
        presplit_train_only = _requires_presplit_train_only_mode(data_manifest)
        cmd.extend(["--data-manifest", str(data_manifest)])

    for source in (common.get("overrides", []), run_cfg.get("overrides", [])):
        if isinstance(source, list):
            for override in source:
                cmd.extend(["--extra-override", str(override)])
    if presplit_train_only:
        split_overrides = _configured_override_values(common, run_cfg, "dataset.split")
        if any(value != "100,0,0" for value in split_overrides):
            raise SystemExit(
                "A/B-v2 finalized train data forbids an internal Bridge re-split; "
                "dataset.split must be 100,0,0"
            )
        # The ATLAS-style validation panel is evaluated by the dedicated
        # per-language evaluator.  Bridge training must consume the complete
        # finalized train prefixes and must not reserve another internal split.
        cmd.extend(["--extra-override", "dataset.split=100,0,0"])

    tokenizer_metadata: dict[str, Any] = {}
    if tokenized_entry is not None:
        tokenizer_type = str(tokenized_entry.get("tokenizer_type", "HuggingFaceTokenizer"))
        tokenizer_model = str(tokenized_entry.get("tokenizer_model", ""))
        trust_remote_code = bool(tokenized_entry.get("trust_remote_code", False))
        vocab_size = int(tokenized_entry.get("vocab_size", 0) or 0)
        if not tokenizer_model:
            raise SystemExit(f"Tokenized manifest entry for run {run_cfg.get('id', '')} is missing tokenizer_model")
        cmd.extend(["--extra-override", f"tokenizer.tokenizer_type={tokenizer_type}"])
        cmd.extend(["--extra-override", f"tokenizer.tokenizer_model={tokenizer_model}"])
        if trust_remote_code:
            cmd.extend(["--extra-override", "+tokenizer.hf_tokenizer_kwargs.trust_remote_code=true"])
        if vocab_size > 0:
            cmd.extend(["--extra-override", f"tokenizer.vocab_size={vocab_size}"])
            cmd.extend(["--extra-override", f"model.vocab_size={vocab_size}"])
        tokenizer_metadata = {
            "tokenizer_id": tokenized_entry.get("tokenizer_id"),
            "tokenizer_type": tokenizer_type,
            "tokenizer_model": tokenizer_model,
            "trust_remote_code": trust_remote_code,
            "vocab_size": vocab_size,
        }

    extra_env = _deep_merge(common.get("env", {}), run_cfg.get("env", {}))
    if isinstance(extra_env, dict):
        for key, value in extra_env.items():
            env[str(key)] = str(value)

    metadata = {
        "backend": str(run_cfg.get("backend", common.get("backend", "megatron_bridge"))),
        "recipe": recipe,
        "dataset": dataset,
        "bridge_root": str(bridge_root),
        "nproc_per_node": nproc_per_node,
        "nnodes": nnodes,
        "seq_length": seq_length,
        "train_iters": train_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "expected_tokens": expected_tokens,
        "expected_end_step": expected_end_step,
        "expected_consumed_samples": expected_consumed_samples,
        "require_full_state_checkpoint": require_full_state_checkpoint,
        "data_manifest": str(data_manifest) if data_manifest is not None else "",
        "data_manifest_max_prefixes": max_manifest_prefixes if data_manifest is not None else None,
        "presplit_train_only": presplit_train_only,
        "dataset_split": "100,0,0" if presplit_train_only else "runner_default",
        "tokenizer": tokenizer_metadata,
    }
    if require_full_state_checkpoint:
        metadata.update(
            {
                "final_checkpoint_root": str((output_dir / "checkpoints").resolve()),
                "final_checkpoint_step": int(expected_end_step),
                "final_checkpoint_expected_consumed_samples": int(
                    expected_consumed_samples
                ),
                "checkpoint_state_audit": str(
                    (output_dir / "checkpoint_state_audit.json").resolve()
                ),
            }
        )
    calibration_selection = run_cfg.get("calibration_selection")
    if isinstance(calibration_selection, dict):
        metadata["calibration_selection"] = calibration_selection
    return cmd, env, metadata


def _token_budget_tokens(train: dict[str, Any], common: dict[str, Any]) -> int:
    return as_positive_int(
        train.get(
            "token_budget_tokens",
            train.get("token_budget", common.get("token_budget_tokens", common.get("token_budget"))),
        )
    ) or 0


def _segment_scheduler_overrides(
    *,
    segment: dict[str, Any],
    policy: dict[str, Any],
    total_tokens: int,
    endpoint_plan: dict[str, Any] | None = None,
    branch_target_tokens: int | None = None,
) -> list[str]:
    seq_length = int(policy["seq_length"])
    gbs = int(segment["global_batch_size"])
    warmup_tokens = int(policy.get("warmup_tokens", 0) or 0)
    if is_joint_moe_full_state_wsd_v2(policy):
        if endpoint_plan is None:
            raise SystemExit("joint_moe_full_state_wsd_v2 scheduler planning requires an endpoint plan")
        accounting = joint_moe_full_state_wsd_v2_scheduler_accounting(
            policy=policy,
            endpoint_plan=endpoint_plan,
            branch_target_tokens=branch_target_tokens,
        )
        sample_fields = {
            "lr_decay_iters": int(accounting["lr_decay_samples"]),
            "lr_wsd_decay_iters": int(accounting["wsd_decay_samples"]),
            "lr_warmup_iters": int(accounting["warmup_samples"]),
        }
        for field, samples in sample_fields.items():
            if samples % gbs:
                raise SystemExit(
                    f"joint_moe_full_state_wsd_v2 {field} sample target {samples} is not divisible by GBS {gbs}"
                )
        return [
            "scheduler.override_opt_param_scheduler=true",
            "scheduler.use_checkpoint_opt_param_scheduler=false",
            "scheduler.lr_decay_style=WSD",
            "scheduler.lr_wsd_decay_style=linear",
            "scheduler.lr_warmup_fraction=null",
            "scheduler.lr_decay_samples=null",
            "scheduler.lr_wsd_decay_samples=null",
            "scheduler.lr_warmup_samples=0",
            f"scheduler.lr_decay_iters={sample_fields['lr_decay_iters'] // gbs}",
            f"scheduler.lr_wsd_decay_iters={sample_fields['lr_wsd_decay_iters'] // gbs}",
            f"scheduler.lr_warmup_iters={sample_fields['lr_warmup_iters'] // gbs}",
            "optimizer.min_lr=0.0",
        ]

    overrides = [
        "scheduler.override_opt_param_scheduler=true",
        "scheduler.use_checkpoint_opt_param_scheduler=false",
        f"scheduler.lr_decay_iters={scheduler_iter_for_tokens(tokens=total_tokens, global_batch_size=gbs, seq_length=seq_length)}",
    ]
    if warmup_tokens > 0:
        overrides.append(
            f"scheduler.lr_warmup_iters={scheduler_iter_for_tokens(tokens=warmup_tokens, global_batch_size=gbs, seq_length=seq_length)}"
        )
    return overrides


def _segmented_training_cmd(
    *,
    repo_root: Path,
    env: dict[str, str],
    output_dir: Path,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    tokenized_path: Path,
    run_name: str,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    train = _deep_merge(common.get("train", {}), run_cfg.get("train", {}))
    seq_length = int(run_cfg.get("seq_length", common.get("seq_length", train.get("seq_length", 2048))) or 2048)
    policy = resolve_batch_policy(train, seq_length=seq_length)
    if policy is None:
        return _bridge_training_cmd(
            repo_root=repo_root,
            env=env,
            output_dir=output_dir,
            run_cfg=run_cfg,
            common=common,
            tokenized_path=tokenized_path,
        )
    total_tokens = _token_budget_tokens(train, common)
    if total_tokens <= 0:
        raise SystemExit("training.train.batch_policy.enabled=true requires train.token_budget_tokens")
    plateau_mbs = int(train.get("micro_batch_size", train.get("micro_batch", 1)) or 1)
    data_parallel_size = int(policy.get("data_parallel_size", 1) or 1)
    endpoint_plan: dict[str, Any] | None = None
    branch_target_tokens: int | None = None
    branch_parent: dict[str, Any] | None = None
    fixed_dataset_sample_horizon: int | None = None
    if is_joint_moe_full_state_wsd_v2(policy):
        endpoint_plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=policy,
            plateau_micro_batch_size=plateau_mbs,
            data_parallel_size=data_parallel_size,
        )
        fixed_dataset_sample_horizon = int(endpoint_plan["fixed_dataset_sample_horizon"])
        plan_role = str(policy["plan_role"])
        if plan_role == "trunk":
            expected_budget = int(endpoint_plan["trunk"]["target_end_tokens"])
            segments = list(endpoint_plan["trunk"]["segments"])
        else:
            branch_target_tokens = int(policy["branch_target_tokens"])
            branch = next(
                item
                for item in endpoint_plan["branches"]
                if int(item["target_tokens"]) == branch_target_tokens
            )
            expected_budget = branch_target_tokens
            segments = list(branch["segments"])
            branch_parent = {
                "checkpoint_root": str(policy["branch_parent_checkpoint_root"]),
                "expected_step": int(branch["expected_start_step"]),
                "expected_consumed_samples": int(branch["expected_start_consumed_samples"]),
            }
        if total_tokens != expected_budget:
            raise SystemExit(
                f"joint_moe_full_state_wsd_v2 {plan_role} plan requires train.token_budget_tokens="
                f"{expected_budget} (got {total_tokens})"
            )
    else:
        segments = materialize_batch_segments(
            policy=policy,
            total_tokens=total_tokens,
            plateau_micro_batch_size=plateau_mbs,
            data_parallel_size=data_parallel_size,
        )
    segment_payloads: list[dict[str, Any]] = []
    segment_env = dict(env)
    run_env: dict[str, str] = {}
    previous_segment: dict[str, Any] | None = None
    require_full_state_checkpoint = fixed_dataset_sample_horizon is not None or bool(
        policy.get("require_full_state_checkpoint", False)
    )
    for segment in segments:
        segment_index = int(segment["segment_index"])
        segment_dir = output_dir / "segments" / f"{segment_index:02d}_gbs{segment['global_batch_size']}"
        segment_log = output_dir / "segments" / f"{segment_index:02d}_gbs{segment['global_batch_size']}.train.log"
        segment_run = dict(run_cfg)
        segment_run["id"] = f"{run_cfg['id']}__segment_{segment_index:02d}"
        segment_train = _deep_merge(segment_run.get("train", {}), {})
        segment_train["batch_policy"] = {"enabled": False}
        segment_train["global_batch_size"] = int(segment["global_batch_size"])
        segment_train["global_batch"] = int(segment["global_batch_size"])
        segment_train["micro_batch_size"] = int(segment["micro_batch_size"])
        segment_train["micro_batch"] = int(segment["micro_batch_size"])
        expected_end_step = int(segment["cumulative_end_step"])
        dataset_train_iters = expected_end_step
        if require_full_state_checkpoint:
            if fixed_dataset_sample_horizon is not None:
                gbs = int(segment["global_batch_size"])
                if fixed_dataset_sample_horizon % gbs:
                    raise SystemExit(
                        f"Fixed dataset sample horizon {fixed_dataset_sample_horizon} is not divisible by GBS {gbs}"
                    )
                dataset_train_iters = fixed_dataset_sample_horizon // gbs
            segment_train["expected_end_step"] = expected_end_step
            segment_train["expected_consumed_samples"] = int(segment["actual_end_samples"])
            segment_train["require_full_state_checkpoint"] = True
        segment_train["train_iters"] = dataset_train_iters
        segment_train["max_steps"] = dataset_train_iters
        segment_train["save_interval"] = expected_end_step
        segment_train["save_interval_iters"] = expected_end_step
        segment_train["token_budget_tokens"] = ""
        segment_train["token_budget"] = ""
        segment_run["train"] = segment_train
        overrides = list(segment_run.get("overrides", [])) if isinstance(segment_run.get("overrides", []), list) else []
        overrides.extend(
            _segment_scheduler_overrides(
                segment=segment,
                policy=policy,
                total_tokens=total_tokens,
                endpoint_plan=endpoint_plan,
                branch_target_tokens=branch_target_tokens,
            )
        )
        if fixed_dataset_sample_horizon is not None:
            overrides.append(f"train.exit_interval={expected_end_step}")
        overrides.extend(
            [
                f"checkpoint.save={segment_dir / 'checkpoints'}",
                f"checkpoint.save_interval={expected_end_step}",
            ]
        )
        if require_full_state_checkpoint:
            # A/B-v2 validates the full optimizer/scheduler/RNG checkpoint
            # immediately at each endpoint.  Synchronous save avoids auditing
            # a checkpoint whose async writer has not fully flushed yet.
            overrides.append("checkpoint.async_save=false")
        else:
            overrides.extend(
                [
                    "checkpoint.async_save=true",
                    "checkpoint.async_strategy=mcore",
                ]
            )
        initial_checkpoint: dict[str, Any] | None = None
        if previous_segment is None and branch_parent is not None:
            initial_checkpoint = dict(branch_parent)
            overrides.extend(
                [
                    f"checkpoint.load={branch_parent['checkpoint_root']}",
                    f"checkpoint.ckpt_step={branch_parent['expected_step']}",
                    "checkpoint.load_optim=true",
                    "checkpoint.load_rng=true",
                    "checkpoint.finetune=false",
                    "checkpoint.exit_on_missing_checkpoint=true",
                ]
            )
        if previous_segment is not None:
            if require_full_state_checkpoint:
                initial_checkpoint = {
                    "checkpoint_root": str(Path(str(previous_segment["output_dir"])) / "checkpoints"),
                    "expected_step": int(previous_segment["cumulative_end_step"]),
                    "expected_consumed_samples": int(previous_segment["actual_end_samples"]),
                }
            overrides.extend(
                [
                    f"checkpoint.load={Path(str(previous_segment['output_dir'])) / 'checkpoints'}",
                    f"checkpoint.ckpt_step={int(previous_segment['cumulative_end_step'])}",
                    "checkpoint.load_optim=true",
                    "checkpoint.load_rng=true",
                    "checkpoint.finetune=false",
                    "checkpoint.exit_on_missing_checkpoint=true",
                ]
            )
        segment_run["overrides"] = overrides
        segment_cmd, segment_env, segment_metadata = _bridge_training_cmd(
            repo_root=repo_root,
            env=env,
            output_dir=segment_dir,
            run_cfg=segment_run,
            common=common,
            tokenized_path=tokenized_path,
        )
        if fixed_dataset_sample_horizon is not None:
            scheduler_accounting = joint_moe_full_state_wsd_v2_scheduler_accounting(
                policy=policy,
                endpoint_plan=endpoint_plan,
                branch_target_tokens=branch_target_tokens,
            )
            segment_metadata.update(
                {
                    "planning_method": policy["planning_method"],
                    "plan_role": policy["plan_role"],
                    "dataset_train_iters": dataset_train_iters,
                    "fixed_dataset_sample_horizon": fixed_dataset_sample_horizon,
                    "expected_end_step": expected_end_step,
                    "expected_consumed_samples": int(segment["actual_end_samples"]),
                    "scheduler_accounting": scheduler_accounting,
                }
            )
        run_env = segment_env
        segment_payload = {
            **segment,
            "dataset_train_iters": dataset_train_iters,
            "fixed_dataset_sample_horizon": fixed_dataset_sample_horizon,
            "require_full_state_checkpoint": require_full_state_checkpoint,
            "run_id": segment_run["id"],
            "run_name": f"{run_name}.segment_{segment_index:02d}",
            "cmd": segment_cmd,
            "output_dir": str(segment_dir),
            "log_path": str(segment_log),
            "metadata": segment_metadata,
        }
        if initial_checkpoint is not None:
            segment_payload["initial_checkpoint"] = initial_checkpoint
        segment_payloads.append(segment_payload)
        previous_segment = segment_payload

    plan_path = output_dir / SEGMENT_PLAN_NAME
    plan_payload = {
        "schema_version": 2 if endpoint_plan is not None else 1,
        "kind": "segmented_training_plan",
        "run_id": str(run_cfg["id"]),
        "run_name": run_name,
        "cwd": str(repo_root),
        "output_dir": str(output_dir),
        "target_token_budget_tokens": int(total_tokens),
        "expected_tokens": actual_tokens_from_segments(segments),
        "final_cumulative_step": final_cumulative_step(segments),
        "expected_end_step": final_cumulative_step(segments),
        "expected_consumed_samples": int(segments[-1].get("actual_end_samples", 0)),
        "policy": policy,
        "plateau_micro_batch_size": plateau_mbs,
        "segments": segment_payloads,
    }
    if endpoint_plan is not None:
        plan_payload.update(
            {
                "planning_method": policy["planning_method"],
                "plan_role": policy["plan_role"],
                "fixed_dataset_sample_horizon": fixed_dataset_sample_horizon,
                "fixed_dataset_final_cumulative_step": endpoint_plan["fixed_dataset_final_cumulative_step"],
                "stable_checkpoint_tokens": endpoint_plan["stable_checkpoint_tokens"],
                "checkpoint_horizons_tokens": endpoint_plan["checkpoint_horizons_tokens"],
                "readout_horizons_tokens": endpoint_plan["readout_horizons_tokens"],
                "readout_endpoints": endpoint_plan["readout_endpoints"],
                "branch_target_tokens": branch_target_tokens,
                "branch_parent_checkpoint_root": (
                    branch_parent["checkpoint_root"] if branch_parent is not None else ""
                ),
            }
        )
    pipeline_python = str(run_env.get("PYTHON_EXECUTABLE", env.get("PYTHON_EXECUTABLE", sys.executable))).strip() or sys.executable
    cmd = [
        pipeline_python,
        "-m",
        "multilingual_megatron.pipeline.stages.training.segment_runner",
        "--segment-plan",
        str(plan_path),
        "--output-dir",
        str(output_dir),
        "--train-iters",
        str(final_cumulative_step(segments)),
        "--global-batch-size",
        str(int(policy["target_global_batch_size"])),
        "--micro-batch-size",
        str(plateau_mbs),
        "--seq-length",
        str(seq_length),
    ]
    metadata = {
        "backend": str(run_cfg.get("backend", common.get("backend", "megatron_bridge"))),
        "recipe": str(run_cfg.get("recipe", common.get("recipe", "vanilla_gpt_pretrain_config"))),
        "dataset": str(run_cfg.get("dataset", common.get("dataset", "llm-pretrain"))),
        "seq_length": seq_length,
        "train_iters": final_cumulative_step(segments),
        "eval_interval": "",
        "eval_iters": "",
        "global_batch_size": int(policy["target_global_batch_size"]),
        "micro_batch_size": plateau_mbs,
        "expected_tokens": actual_tokens_from_segments(segments),
        "target_token_budget_tokens": int(total_tokens),
        "segmented_training": True,
        "segment_plan_path": str(plan_path.resolve()),
        "segment_plan": plan_payload,
        "batch_policy": {
            "mode": policy["mode"],
            "planning_method": policy.get("planning_method", ""),
            "target_global_batch_size": int(policy["target_global_batch_size"]),
            "mbs_policy": policy["mbs_policy"],
            "schedule": policy["schedule"],
        },
        "segments": [
            {
                key: value
                for key, value in segment.items()
                if key
                in {
                    "segment_index",
                    "target_start_tokens",
                    "target_end_tokens",
                    "actual_start_tokens",
                    "actual_end_tokens",
                    "actual_start_samples",
                    "actual_end_samples",
                    "global_batch_size",
                    "micro_batch_size",
                    "tokens_per_step",
                    "segment_train_iters",
                    "cumulative_start_step",
                    "cumulative_end_step",
                    "expected_end_step",
                    "expected_consumed_samples",
                    "dataset_train_iters",
                    "fixed_dataset_sample_horizon",
                    "output_dir",
                    "log_path",
                }
            }
            for segment in segment_payloads
        ],
        "final_checkpoint_root": str(Path(str(segment_payloads[-1]["output_dir"])) / "checkpoints"),
        "final_checkpoint_step": int(segment_payloads[-1]["cumulative_end_step"]),
    }
    if endpoint_plan is not None:
        metadata.update(
            {
                "planning_method": policy["planning_method"],
                "plan_role": policy["plan_role"],
                "fixed_dataset_sample_horizon": fixed_dataset_sample_horizon,
                "fixed_dataset_final_cumulative_step": endpoint_plan["fixed_dataset_final_cumulative_step"],
                "expected_end_step": final_cumulative_step(segments),
                "expected_consumed_samples": int(segments[-1]["actual_end_samples"]),
                "stable_checkpoint_tokens": endpoint_plan["stable_checkpoint_tokens"],
                "checkpoint_horizons_tokens": endpoint_plan["checkpoint_horizons_tokens"],
                "readout_horizons_tokens": endpoint_plan["readout_horizons_tokens"],
                "readout_endpoints": endpoint_plan["readout_endpoints"],
                "branch_target_tokens": branch_target_tokens,
            }
        )
    calibration_selection = run_cfg.get("calibration_selection")
    if isinstance(calibration_selection, dict):
        metadata["calibration_selection"] = calibration_selection
    return cmd, run_env or dict(env), metadata


def describe_run_tokenization(
    *,
    tokenized_path: Path,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any] | None:
    return resolve_tokenized_entry(tokenized_path, common, run_cfg)


def build_training_cmd(
    *,
    repo_root: Path,
    env: dict[str, str],
    output_dir: Path,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    language_map_path: Path,
    tokenized_path: Path,
    run_name: str,
) -> tuple[list[str], dict[str, str]]:
    del language_map_path
    if not is_bridge_backend(common, run_cfg):
        kind = str(run_cfg.get("kind", ""))
        raise SystemExit(f"Unsupported training backend/kind in this repo: {kind!r}")
    cmd, run_env, _metadata = _segmented_training_cmd(
        repo_root=repo_root,
        env=env,
        output_dir=output_dir,
        run_cfg=run_cfg,
        common=common,
        tokenized_path=tokenized_path,
        run_name=run_name,
    )
    return cmd, run_env


def describe_training_run(
    ctx: Any,
    *,
    run: dict[str, Any],
    train_cfg_for_stage: dict[str, Any],
) -> tuple[str, str, Path, Path]:
    del train_cfg_for_stage
    run_id = str(run["id"])
    run_name_label = _build_run_label(ctx.base_run_label, run, default_num_experts=1)
    output_dir = ctx.train_root / run_id
    env = getattr(ctx, "env", {})
    if isinstance(env, dict) and _structured_logs_enabled(env):
        log_dir = ctx.logs_root / "stages" / "train" / _log_path_component(run_id)
        log_path = log_dir / "train.log"
    else:
        log_dir = ctx.logs_root / "train_runs" / run_id
        log_path = log_dir / f"{run_name_label}.train.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_name_label, output_dir, log_path


def build_training_run_plan(
    ctx: Any,
    *,
    run: dict[str, Any],
    common: dict[str, Any],
    output_dir: Path | None = None,
    log_path: Path | None = None,
) -> TrainingRunPlan:
    run_id, run_name, default_output_dir, default_log_path = describe_training_run(ctx, run=run, train_cfg_for_stage=common)
    output_dir = output_dir or default_output_dir
    log_path = log_path or default_log_path
    cmd, run_env, metadata = _segmented_training_cmd(
        repo_root=ctx.repo_root,
        env=ctx.env,
        output_dir=output_dir,
        run_cfg=run,
        common=common,
        tokenized_path=ctx.tokenized_dir,
        run_name=run_name,
    )
    tokenization = describe_run_tokenization(tokenized_path=ctx.tokenized_dir, run_cfg=run, common=common)
    return TrainingRunPlan(
        run_cfg=dict(run),
        run_id=run_id,
        run_name=run_name,
        output_dir=output_dir,
        log_path=log_path,
        cmd=cmd,
        env=run_env,
        tokenization=tokenization,
        metadata=metadata,
    )


def run_is_complete(plan: TrainingRunPlan) -> bool:
    return all(path.exists() for path in plan.expected_artifacts)


def core_run_artifacts_exist(plan: TrainingRunPlan) -> bool:
    return all(
        (plan.output_dir / name).exists()
        for name in ("train_results.json", "trainer_state.json", ".training_complete")
    )


def write_training_command_plan(plan: TrainingRunPlan) -> Path:
    path = plan.output_dir / "training_command_plan.json"
    segment_plan = plan.metadata.get("segment_plan")
    if isinstance(segment_plan, dict):
        _dump_json(plan.output_dir / SEGMENT_PLAN_NAME, segment_plan)
    metadata = dict(plan.metadata)
    metadata.pop("segment_plan", None)
    payload = {
        "schema_version": 1,
        "run_id": plan.run_id,
        "run_name": plan.run_name,
        "cmd": plan.cmd,
        "log_path": str(plan.log_path),
        "metadata": metadata,
    }
    _dump_json(path, payload)
    return path


def write_model_tokenizer_lineage(
    *,
    plan: TrainingRunPlan,
    model_name_or_path: str,
    tokenized_path: Path,
) -> Path:
    path = plan.output_dir / "model_tokenizer_lineage.json"
    _dump_json(
        path,
        {
            "run_id": plan.run_id,
            "model_name_or_path": model_name_or_path,
            "tokenized_path": str(tokenized_path.resolve()),
            "tokenization": plan.tokenization,
        },
    )
    return path


def write_training_run_manifest(
    *,
    plan: TrainingRunPlan,
    status: str,
    model_name_or_path: str,
    tokenized_path: Path,
) -> Path:
    path = plan.output_dir / TRAINING_RUN_MANIFEST_NAME
    artifacts = {
        "train_results": str((plan.output_dir / "train_results.json").resolve()),
        "trainer_state": str((plan.output_dir / "trainer_state.json").resolve()),
        "training_complete": str((plan.output_dir / ".training_complete").resolve()),
        "model_tokenizer_lineage": str((plan.output_dir / "model_tokenizer_lineage.json").resolve()),
        "command_plan": str((plan.output_dir / "training_command_plan.json").resolve()),
    }
    calibration_path = plan.output_dir / TRAINING_CALIBRATION_NAME
    calibration_tsv_path = plan.output_dir / TRAINING_CALIBRATION_TSV_NAME
    if calibration_path.exists():
        artifacts["calibration"] = str(calibration_path.resolve())
    if calibration_tsv_path.exists():
        artifacts["calibration_tsv"] = str(calibration_tsv_path.resolve())
    metadata = dict(plan.metadata)
    metadata.pop("segment_plan", None)
    payload = {
        "schema_version": 1,
        "stage": "train",
        "run_id": plan.run_id,
        "run_name": plan.run_name,
        "status": status,
        "backend": plan.metadata.get("backend"),
        "recipe": plan.metadata.get("recipe"),
        "dataset": plan.metadata.get("dataset"),
        "output_dir": str(plan.output_dir.resolve()),
        "log_path": str(plan.log_path.resolve()),
        "model_name_or_path": model_name_or_path,
        "tokenized_path": str(tokenized_path.resolve()),
        "tokenization": plan.tokenization,
        "run_cfg": plan.run_cfg,
        "metadata": metadata,
        "artifacts": artifacts,
    }
    _dump_json(path, payload)
    return path


def write_training_manifest(*, train_root: Path, plans: list[TrainingRunPlan]) -> Path:
    path = train_root / TRAINING_MANIFEST_NAME
    runs: dict[str, Any] = {}
    for plan in plans:
        run_manifest = plan.output_dir / TRAINING_RUN_MANIFEST_NAME
        status = "complete" if run_is_complete(plan) else "missing"
        if run_manifest.exists():
            try:
                payload = json.loads(run_manifest.read_text(encoding="utf-8"))
                status = str(payload.get("status", status))
            except Exception:
                status = "invalid_manifest"
        runs[plan.run_id] = {
            "status": status,
            "run_manifest": str(run_manifest.resolve()),
            "output_dir": str(plan.output_dir.resolve()),
            "train_results": str((plan.output_dir / "train_results.json").resolve()),
            "trainer_state": str((plan.output_dir / "trainer_state.json").resolve()),
            "training_complete": str((plan.output_dir / ".training_complete").resolve()),
        }
        calibration_path = plan.output_dir / TRAINING_CALIBRATION_NAME
        if calibration_path.exists():
            runs[plan.run_id]["calibration"] = str(calibration_path.resolve())
    _dump_json(
        path,
        {
            "schema_version": 1,
            "stage": "train",
            "train_root": str(train_root.resolve()),
            "runs": runs,
        },
    )
    return path


def run_training_variant(
    *,
    repo_root: Path,
    env: dict[str, str],
    output_dir: Path,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    language_map_path: Path,
    tokenized_path: Path,
    run_name: str,
    log_path: Path,
    dry_run: bool,
    executor: Any,
) -> None:
    cmd, run_env = build_training_cmd(
        repo_root=repo_root,
        env=env,
        output_dir=output_dir,
        run_cfg=run_cfg,
        common=common,
        language_map_path=language_map_path,
        tokenized_path=tokenized_path,
        run_name=run_name,
    )
    executor.run(cmd, run_env, dry_run, log_path)
