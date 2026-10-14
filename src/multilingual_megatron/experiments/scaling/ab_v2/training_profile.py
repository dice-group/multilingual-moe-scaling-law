"""Build immutable Megatron profile payloads and validate LR lineage."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    FIXED_PEAK_LEARNING_RATE,
    Paths,
    WorkflowError,
    _abs,
    _file_sha256,
    _language_map,
    _prescribed_peak_learning_rate,
    _runtime_primary_training_seed,
    _workflow_profile_name,
)
from .estimates import _canonical_batch_ramp_policy, _training_data_parallel_size
from .variant import active_variant


def _peak_lr_for_job(manifest: Mapping[str, Any], args: Mapping[str, Any]) -> float:
    if "fixed_peak_learning_rate" in manifest:
        try:
            manifest_lr = float(manifest["fixed_peak_learning_rate"])
            job_lr = float(args["peak_lr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError("historical training job lacks fixed peak LR") from exc
        if (
            not math.isclose(
                manifest_lr, FIXED_PEAK_LEARNING_RATE, rel_tol=0.0, abs_tol=0.0
            )
            or not math.isclose(job_lr, manifest_lr, rel_tol=0.0, abs_tol=0.0)
        ):
            raise WorkflowError("historical training job fixed peak LR drifted")
        return job_lr
    try:
        prescription = manifest["learning_rate_prescription"]
        ladder = str(args["ladder"])
        experts = int(args["experts"])
        base_lr = float(args["base_peak_lr"])
        multiplier = float(args["lr_multiplier"])
        job_lr = float(args["peak_lr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            "every H100 training job must declare its prescribed LR lineage"
        ) from exc
    if not isinstance(prescription, Mapping):
        raise WorkflowError("workflow manifest LR prescription is invalid")
    expected_base = _prescribed_peak_learning_rate(
        prescription, ladder=ladder, experts=experts
    )
    expected_lr = expected_base * multiplier
    if (
        args.get("learning_rate_policy")
        != "jointmoe_paper_rounded_coefficients_k_invariant"
        or not math.isclose(base_lr, expected_base, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(job_lr, expected_lr, rel_tol=0.0, abs_tol=0.0)
    ):
        raise WorkflowError(
            "training job peak learning rate differs from the manifest prescription"
        )
    return job_lr


def _training_overrides(
    *,
    experts: int,
    peak_lr: float,
    model_seed: int,
    tokenizer_model: Path,
    dataset_random_seed: int | None = None,
) -> list[str]:
    dataset_random_seed = (
        model_seed if dataset_random_seed is None else int(dataset_random_seed)
    )
    overrides = [
        "mixed_precision=bf16_mixed",
        "model.transformer_impl=transformer_engine",
        "model.attention_backend=auto",
        "model.cuda_graph_impl=none",
        "model.sequence_parallel=false",
        "model.tensor_model_parallel_size=1",
        "model.pipeline_model_parallel_size=1",
        "model.context_parallel_size=1",
        "model.expert_model_parallel_size=1",
        "model.expert_tensor_parallel_size=1",
        "model.virtual_pipeline_model_parallel_size=null",
        "model.gradient_accumulation_fusion=true",
        "model.cross_entropy_loss_fusion=true",
        "model.cross_entropy_fusion_impl=te",
        "model.apply_rope_fusion=true",
        "model.bias_dropout_fusion=true",
        "model.persist_layer_norm=true",
        "model.masked_softmax_fusion=true",
        "model.share_embeddings_and_output_weights=false",
        "model.should_pad_vocab=false",
        "optimizer.optimizer=adam",
        f"optimizer.lr={peak_lr:.17g}",
        "optimizer.min_lr=0.0",
        "optimizer.adam_beta1=0.9",
        "optimizer.adam_beta2=0.95",
        "optimizer.adam_eps=1e-8",
        "optimizer.weight_decay=0.1",
        "optimizer.clip_grad=1.0",
        "optimizer.bf16=true",
        "optimizer.fp16=false",
        "scheduler.start_weight_decay=0.1",
        "scheduler.end_weight_decay=0.1",
        "scheduler.weight_decay_incr_style=constant",
        "dist.distributed_backend=nccl",
        "profiling.record_memory_history=false",
        "profiling.use_pytorch_profiler=false",
        "profiling.use_nsys_profiler=false",
        "checkpoint.async_save=false",
        f"rng.seed={model_seed}",
        f"dataset.random_seed={dataset_random_seed}",
        "tokenizer.tokenizer_type=SentencePieceTokenizer",
        f"tokenizer.tokenizer_model={tokenizer_model}",
        "tokenizer.vocab_size=64000",
        "model.vocab_size=64000",
        "dataset.split=100,0,0",
        "dataset.num_dataset_builder_threads=2",
        "dataset.num_workers=2",
        "dataset.persistent_workers=true",
        "dataset.pin_memory=true",
        "validation.eval_iters=0",
    ]
    if experts > 1:
        overrides.extend(
            [
                f"model.num_moe_experts={experts}",
                "model.moe_router_topk=1",
                "model.moe_router_pre_softmax=true",
                "model.moe_aux_loss_coeff=0.01",
                "model.moe_z_loss_coeff=0.001",
                "model.moe_router_dtype=fp32",
                "model.moe_expert_capacity_factor=null",
                "model.moe_pad_expert_input_to_capacity=false",
                "model.moe_grouped_gemm=true",
                "model.moe_permute_fusion=true",
                "model.moe_router_fusion=true",
                "model.moe_shared_expert_overlap=false",
                "model.use_transformer_engine_op_fuser=true",
                "model.use_te_activation_func=true",
                "model.use_te_rng_tracker=true",
                "rng.te_rng_tracker=true",
                "model.moe_token_dispatcher_type=allgather",
                "model.overlap_moe_expert_parallel_comm=false",
                "model.delay_wgrad_compute=false",
            ]
        )
    else:
        overrides.extend(
            [
                "model.num_moe_experts=null",
                "model.use_transformer_engine_op_fuser=false",
            ]
        )
    transformed = active_variant().training_overrides_transform(
        overrides,
        experts=experts,
        peak_lr=peak_lr,
        model_seed=model_seed,
        dataset_random_seed=dataset_random_seed,
        tokenizer_model=tokenizer_model,
    )
    if not isinstance(transformed, list) or any(
        not isinstance(value, str) for value in transformed
    ):
        raise WorkflowError("workflow variant returned invalid training overrides")
    return transformed


def _evaluation_model_overrides(
    *, experts: int, model_seed: int, tokenizer_model: Path
) -> list[str]:
    """Reuse the training model execution contract for checkpoint evaluation.

    Evaluation must not silently fall back to a recipe default that differs
    from the checkpoint's training recipe, notably native rather than
    Transformer-Engine cross entropy. Optimizer, data-loader, checkpoint,
    tokenizer, and RNG overrides remain evaluation-owned concerns and are
    therefore intentionally excluded.
    """

    return [
        override
        for override in _training_overrides(
            experts=experts,
            peak_lr=1.0,
            model_seed=model_seed,
            dataset_random_seed=model_seed,
            tokenizer_model=tokenizer_model,
        )
        if override.startswith("model.") or override.startswith("mixed_precision=")
    ]


def _profile_payload(
    *,
    base: Mapping[str, Any],
    runtime: Mapping[str, Any],
    paths: Paths,
    job_root: Path,
    ladder: str,
    experts: int,
    k: int,
    horizons: Sequence[int],
    role: str,
    d_tokens: int,
    peak_lr: float,
    parent_checkpoint_root: str | None,
    model_seed: int | None = None,
    dataset_random_seed: int | None = None,
    sentinel: bool = False,
    sentinel_global_batch_size: int | None = None,
    sentinel_lr_warmup_steps: int | None = None,
    variant_arguments: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    # The inherited historical profile contains stale CUDA 13, /tmp, offline,
    # sample, tokenizer-training, async-checkpoint, and signal settings.  The
    # A/B-v2 job profile is constructed explicitly so none can leak through.
    _ = base
    profile: dict[str, Any] = {}
    tier_manifest = paths.tokenized / "tiers" / f"K{k}" / "megatron_data_manifest.json"
    if not tier_manifest.is_file():
        raise WorkflowError(f"missing frozen K{k} tier manifest: {tier_manifest}")
    profile["executor"] = {"type": "local"}
    profile["output_root"] = str(job_root / "pipeline")
    profile["logs_dir"] = str(job_root / "pipeline" / "logs")
    profile["language_map_path"] = str(_abs(_language_map(k)))
    profile["runtime"] = {"env": dict(runtime["training"]["runtime_env"])}
    profile["run_naming"] = {
        "experiment_tag": str(runtime.get("workflow_tag", "multilingual-moe-ab-v2")),
        "language_count_hint": k,
        "job_name_prefix": str(runtime.get("workflow_job_prefix", "abv2")),
    }
    profile["tokenization"] = {
        **dict(profile.get("tokenization", {})),
        "output_dir": str(paths.tokenized / "tiers" / f"K{k}"),
        "model_name": str(paths.tokenizer / "tokenizer.model"),
        "tokenizer_type": "SentencePieceTokenizer",
    }
    recipe = f"{'scaling_dense' if experts == 1 else 'scaling_switch_moe'}_scale_scaling_{ladder.lower()}_pretrain_config"
    run_id = (
        f"{str(runtime.get('workflow_job_prefix', 'abv2'))}_{role}_"
        f"{ladder.lower()}_e{experts}_k{k:03d}_d{d_tokens}"
    )
    mbs = int(runtime["training"]["micro_batch_size"][ladder][str(experts)])
    processes = active_variant().processes_for_experts(experts)
    if not isinstance(processes, int) or processes <= 0:
        raise WorkflowError("workflow variant returned an invalid process count")
    common: dict[str, Any] = {
        "enabled": True,
        "backend": "megatron_bridge",
        "bridge_root": "external/Megatron-Bridge",
        "dataset": "llm-pretrain",
        "recipe": recipe,
        "launcher": "python",
        "python_executable": str(runtime["bridge_python_executable"]),
        "nproc_per_node": processes,
        "seq_length": 2048,
        "output_dir": str(job_root / "runs"),
        "eval": {"eval_interval": 1_000_000_000, "eval_iters": 0},
        "launch": {"preflight_cuda": True},
        "overrides": _training_overrides(
            experts=experts,
            peak_lr=peak_lr,
            model_seed=int(
                model_seed
                if model_seed is not None
                else _runtime_primary_training_seed(runtime)
            ),
            dataset_random_seed=int(
                dataset_random_seed
                if dataset_random_seed is not None
                else _runtime_primary_training_seed(runtime)
            ),
            tokenizer_model=paths.tokenizer / "tokenizer.model",
        ),
    }
    if sentinel_lr_warmup_steps is not None:
        if not sentinel:
            raise WorkflowError(
                "stress LR schedule is valid only for sentinel-style probes"
            )
        if int(sentinel_lr_warmup_steps) <= 0:
            raise WorkflowError("stress LR warmup steps must be positive")
        common["overrides"].extend(
            [
                "scheduler.lr_decay_style=constant",
                "scheduler.lr_warmup_fraction=null",
                "scheduler.lr_warmup_samples=0",
                f"scheduler.lr_warmup_iters={int(sentinel_lr_warmup_steps)}",
                "scheduler.lr_decay_samples=null",
                "scheduler.lr_wsd_decay_samples=null",
                "scheduler.lr_wsd_decay_iters=null",
            ]
        )
    data_parallel_size = _training_data_parallel_size(
        runtime, ladder=ladder, experts=experts
    )
    selected_gbs = int(sentinel_global_batch_size or 128) if sentinel else 128
    if sentinel:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            derive_segment_micro_batch_size,
        )

        selected_mbs = derive_segment_micro_batch_size(
            plateau_micro_batch_size=mbs,
            global_batch_size=selected_gbs,
            data_parallel_size=data_parallel_size,
        )
    else:
        selected_mbs = mbs
    if selected_gbs % (selected_mbs * data_parallel_size):
        raise WorkflowError(
            f"GBS {selected_gbs} is not divisible by MBS {selected_mbs} and DP {data_parallel_size}"
        )
    train: dict[str, Any] = {
        "global_batch_size": selected_gbs,
        "micro_batch_size": selected_mbs,
        "log_interval": 10,
        "token_budget_tokens": d_tokens
        if role == "branch" or sentinel
        else int(0.8 * max(horizons)),
    }
    if sentinel:
        sentinel_gbs = int(train["global_batch_size"])
        sentinel_step = (int(d_tokens) + sentinel_gbs * 2048 - 1) // (
            sentinel_gbs * 2048
        )
        sentinel_samples = sentinel_step * sentinel_gbs
        train.update(
            {
                "train_iters": sentinel_step,
                "save_interval": sentinel_step,
                "save_interval_iters": sentinel_step,
                "expected_end_step": sentinel_step,
                "expected_consumed_samples": sentinel_samples,
                "require_full_state_checkpoint": True,
            }
        )
        train["batch_policy"] = {"enabled": False}
    else:
        train["batch_policy"] = _canonical_batch_ramp_policy(
            horizons=horizons,
            data_parallel_size=_training_data_parallel_size(
                runtime, ladder=ladder, experts=experts
            ),
        )
        train["batch_policy"]["plan_role"] = role
        if role == "branch":
            train["batch_policy"].update(
                {
                    "branch_target_tokens": d_tokens,
                    "branch_parent_checkpoint_root": str(parent_checkpoint_root or ""),
                }
            )
    common["train"] = train
    common["runs"] = [
        {
            "id": run_id,
            "kind": "megatron_bridge",
            "backend": "megatron_bridge",
            "recipe": recipe,
            "data_manifest": str(tier_manifest),
            "data_manifest_sha256": _file_sha256(tier_manifest),
            "train": train,
            "overrides": [],
        }
    ]
    profile["training"] = common
    profile["benchmark"] = {
        "enabled": False,
        "backend": "",
        "output_dir": str(job_root / "pipeline" / "benchmarks"),
        "include_flores": False,
    }
    profile["baseline_eval"] = {"enabled": False}
    profile["wandb"] = {"enabled": False, "mode": "disabled"}
    profile["wandb_mode"] = "disabled"
    profile["summary"] = {
        "summary_csv": str(job_root / "pipeline" / "summary.csv"),
        "gate_json": str(job_root / "pipeline" / "gate.json"),
        "min_relative_improvement": 0.0,
    }
    profile = active_variant().profile_transform(
        profile,
        runtime=runtime,
        paths=paths,
        job_root=job_root,
        ladder=ladder,
        experts=experts,
        k=k,
        role=role,
        d_tokens=d_tokens,
        arguments=dict(variant_arguments or {}),
    )
    if not isinstance(profile, dict):
        raise WorkflowError("workflow variant returned an invalid training profile")
    variant_runtime_env = active_variant().job_environment_transform(
        {},
        {
            "arguments": {
                "ladder": ladder,
                "experts": experts,
                "K": k,
                "role": role,
                "D": d_tokens,
                **dict(variant_arguments or {}),
            },
        },
        runtime,
    )
    if not isinstance(variant_runtime_env, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in variant_runtime_env.items()
    ):
        raise WorkflowError("workflow variant returned an invalid job environment")
    expected_runtime_env = dict(runtime["training"]["runtime_env"])
    expected_runtime_env.update(variant_runtime_env)
    _assert_clean_training_profile(
        profile,
        experts=experts,
        runtime=runtime,
        expected_runtime_env=expected_runtime_env,
    )
    defaults = {
        **profile,
        "sample": {
            "enabled": False,
            "source_type": "json",
            "output_dir": str(job_root / "pipeline" / "disabled_sample"),
        },
        "checkpoint_conversion": {
            "enabled": False,
            "mode": "manifest",
            "output_dir": str(job_root / "pipeline" / "disabled_checkpoint_conversion"),
        },
        "final_loss": {
            "enabled": False,
            "output_dir": str(job_root / "pipeline" / "disabled_final_loss"),
        },
    }
    return {
        "schema_version": 1,
        "defaults": defaults,
        "profiles": {_workflow_profile_name(runtime): profile},
    }, run_id


def _assert_clean_training_profile(
    profile: Mapping[str, Any],
    *,
    experts: int,
    runtime: Mapping[str, Any],
    expected_runtime_env: Mapping[str, str] | None = None,
) -> None:
    forbidden_keys = {"sample", "tokenizer_training", "stage_executors", "audit"}
    leaked = sorted(forbidden_keys.intersection(profile))
    if leaked:
        raise WorkflowError(
            f"resolved A/B-v2 profile contains legacy sections: {leaked}"
        )
    runtime_env = profile.get("runtime", {}).get("env")
    expected_env = (
        dict(expected_runtime_env)
        if expected_runtime_env is not None
        else runtime["training"]["runtime_env"]
    )
    if runtime_env != expected_env:
        raise WorkflowError(
            "resolved A/B-v2 runtime environment differs from the frozen clean "
            "env plus the variant-owned execution env"
        )
    serialized = json.dumps(profile, sort_keys=True)
    for forbidden in (
        "CUDA/13.0.0",
        "--signal",
        '"TMPDIR": "/tmp"',
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "BRIDGE_SKIP_JIT_WARMUP",
        '"CUDA_DEVICE_MAX_CONNECTIONS": "1"',
    ):
        if forbidden in serialized:
            raise WorkflowError(
                f"legacy resolved-profile value leaked into A/B-v2: {forbidden}"
            )
    overrides = profile.get("training", {}).get("overrides")
    if not isinstance(overrides, list):
        raise WorkflowError("resolved A/B-v2 profile lacks training overrides")
    graph_policy = runtime.get("training", {}).get("cuda_graph_impl_by_experts", {})
    if graph_policy and not isinstance(graph_policy, Mapping):
        raise WorkflowError("training.cuda_graph_impl_by_experts must be an object")
    cuda_graph_impl = str(
        graph_policy.get(str(experts), "none") if isinstance(graph_policy, Mapping) else "none"
    )
    if cuda_graph_impl not in {"none", "transformer_engine"}:
        raise WorkflowError(f"unsupported CUDA graph implementation {cuda_graph_impl!r}")
    required = {
        f"model.cuda_graph_impl={cuda_graph_impl}",
        "checkpoint.async_save=false",
        "dataset.persistent_workers=true",
        "dataset.pin_memory=true",
        (
            "model.use_transformer_engine_op_fuser=true"
            if experts > 1
            else "model.use_transformer_engine_op_fuser=false"
        ),
    }
    missing = sorted(required - set(overrides))
    if missing:
        raise WorkflowError(
            f"resolved A/B-v2 profile lacks performance contract: {missing}"
        )
