"""Build and execute A/B-v2 Megatron training and evaluation profiles."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_data import _languages_from_map
from .common import (
    FIXED_PEAK_LEARNING_RATE,
    GRAD_NORM_RE,
    H100_PEAK_TFLOPS_BF16,
    LEARNING_RATE_RE,
    LOSS_RE,
    NAN_ITER_RE,
    OPTIONAL_DEPENDENCY_PREFIX,
    PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS,
    PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS,
    Paths,
    SENTINEL_INTEGRATION_LANGUAGE,
    SKIPPED_RE,
    WorkflowError,
    _abs,
    _atomic_json,
    _file_sha256,
    _language_map,
    _load_object,
    _prescribed_peak_learning_rate,
    _run,
    _runtime_primary_training_seed,
    _workflow_attempt_id,
    _workflow_profile_name,
)
from .estimates import (
    _canonical_batch_ramp_policy,
    _job_horizons,
    _training_data_parallel_size,
)
from .variant import active_variant


def _run_training_job(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    paths: Paths,
    runtime: Mapping[str, Any],
) -> None:
    """Materialize a one-run local profile and execute it inside this allocation.

    The detailed profile builder is kept below so every training job records
    the exact resolved config it ran.
    """

    args = dict(job["arguments"])
    job_root = paths.jobs / str(job["job_id"])
    job_root.mkdir(parents=True, exist_ok=True)
    base: dict[str, Any] = {}
    peak_lr = _peak_lr_for_job(manifest, args)

    if job["action"] in {
        "training_sentinel",
        "fixed_lr_schedule_probe",
        "prescribed_lr_schedule_probe",
    }:
        is_schedule_probe = job["action"] != "training_sentinel"
        is_fixed_probe = job["action"] == "fixed_lr_schedule_probe"
        result = _execute_sentinel_profile(
            base=base,
            runtime=runtime,
            paths=paths,
            job_root=job_root,
            ladder=str(args["ladder"]),
            experts=int(args["experts"]),
            k=int(args["K"]),
            tokens=int(args["tokens"]),
            global_batch_size=int(args["global_batch_size"]),
            peak_lr=peak_lr,
            model_seed=int(args["model_seed"]),
            dataset_random_seed=int(args["dataset_random_seed"]),
            probe_role=(
                "fixed_lr_schedule_verification"
                if is_fixed_probe
                else "prescribed_lr_schedule_verification"
                if is_schedule_probe
                else str(args.get("sentinel_kind", "sentinel"))
            ),
            selected_lr_warmup_steps=(
                PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS if is_schedule_probe else None
            ),
            variant_arguments=args,
        )
    else:
        ladder = str(args["ladder"])
        horizons = _job_horizons(args, runtime)
        role = str(args["role"])
        parent = None
        if role == "branch":
            parent_job = paths.jobs / str(args["parent_job_id"]) / "result.json"
            parent = _branch_parent_for_horizon(
                _load_object(parent_job), int(args["D"])
            )
        result = _execute_training_role(
            base=base,
            runtime=runtime,
            paths=paths,
            job_root=job_root,
            ladder=ladder,
            experts=int(args["experts"]),
            k=int(args["K"]),
            horizons=horizons,
            role=role,
            d_tokens=int(args["D"]),
            peak_lr=peak_lr,
            parent_checkpoint_root=parent,
            model_seed=int(args["model_seed"]),
            dataset_random_seed=int(args["dataset_random_seed"]),
            variant_arguments=args,
        )
    result.update({"job_id": job["job_id"], "peak_lr": peak_lr, "arguments": args})
    _atomic_json(job_root / "result.json", result)


from .training_profile import (
    _assert_clean_training_profile,
    _evaluation_model_overrides,
    _peak_lr_for_job,
    _profile_payload,
    _training_overrides,
)
def _execute_training_role(
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
    variant_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config, run_id = _profile_payload(
        base=base,
        runtime=runtime,
        paths=paths,
        job_root=job_root,
        ladder=ladder,
        experts=experts,
        k=k,
        horizons=horizons,
        role=role,
        d_tokens=d_tokens,
        peak_lr=peak_lr,
        parent_checkpoint_root=parent_checkpoint_root,
        model_seed=model_seed,
        dataset_random_seed=dataset_random_seed,
        variant_arguments=variant_arguments,
    )
    config_path = job_root / "resolved_pipeline_config.json"
    _atomic_json(config_path, config)
    started = time.perf_counter()
    _run(
        [
            "python",
            "-m",
            "multilingual_megatron.pipeline",
            "--config",
            str(config_path),
            "--profile",
            _workflow_profile_name(runtime),
            "--start-at",
            "train",
            "--stop-after",
            "train",
            "--executor-type",
            "local",
            "--run-id",
            run_id,
            "--attempt-id",
            _workflow_attempt_id(runtime, role="production"),
        ]
    )
    manifests = list((job_root / "runs").rglob("training_run_manifest.json"))
    if len(manifests) != 1:
        raise WorkflowError(
            f"expected one training run manifest under {job_root}, found {len(manifests)}"
        )
    training_manifest = _load_object(manifests[0])
    metadata = training_manifest.get("metadata", {})
    segment_plan_path = _resolve_segment_plan_path(
        training_manifest_path=manifests[0], metadata=metadata
    )
    if role in {"trunk", "branch"} and not segment_plan_path.is_file():
        raise WorkflowError(
            f"segmented training manifest omitted its plan and no adjacent plan exists: {segment_plan_path}"
        )
    metadata_segments = metadata.get("segments", [])
    final_segment = (
        metadata_segments[-1]
        if isinstance(metadata_segments, list)
        and metadata_segments
        and isinstance(metadata_segments[-1], dict)
        else {}
    )
    actual_end_tokens = int(
        final_segment.get("actual_end_tokens", metadata.get("expected_tokens", 0)) or 0
    )
    expected_consumed_samples = int(
        final_segment.get(
            "expected_consumed_samples",
            metadata.get("final_checkpoint_expected_consumed_samples", 0),
        )
        or 0
    )
    training_health = _training_health_for_segment_plan(
        segment_plan_path,
        require_moe=experts > 1,
    )
    end_to_end_seconds = time.perf_counter() - started
    if not math.isfinite(end_to_end_seconds) or end_to_end_seconds <= 0:
        raise WorkflowError(f"invalid training end-to-end runtime under {job_root}")
    data_manifest = _resolved_profile_data_manifest(config, runtime)
    result = {
        "schema_version": 1,
        "status": "complete",
        "role": role,
        "training_manifest": str(manifests[0].resolve()),
        "training_manifest_sha256": _file_sha256(manifests[0]),
        "final_checkpoint_root": metadata.get("final_checkpoint_root", ""),
        "final_checkpoint_step": metadata.get(
            "final_checkpoint_step", metadata.get("train_iters", 0)
        ),
        "segment_plan": str(segment_plan_path.resolve())
        if segment_plan_path.is_file()
        else "",
        "actual_end_tokens": actual_end_tokens,
        "expected_consumed_samples": expected_consumed_samples,
        "training_health": training_health,
        "end_to_end_seconds": end_to_end_seconds,
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": _file_sha256(data_manifest),
    }
    if result["segment_plan"] and Path(str(result["segment_plan"])).is_file():
        result["segment_plan_sha256"] = _file_sha256(Path(str(result["segment_plan"])))
    return result


from .training_health import (
    _parse_sentinel_health,
    _resolve_segment_plan_path,
    _training_health_for_segment_plan,
)


def _resolved_profile_data_manifest(
    config: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Path:
    profiles = config.get("profiles")
    profile = (
        profiles.get(_workflow_profile_name(runtime))
        if isinstance(profiles, Mapping)
        else None
    )
    training = profile.get("training") if isinstance(profile, Mapping) else None
    runs = training.get("runs") if isinstance(training, Mapping) else None
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], Mapping):
        raise WorkflowError("resolved training profile lacks exactly one run")
    path = Path(str(runs[0].get("data_manifest", ""))).resolve()
    expected_hash = runs[0].get("data_manifest_sha256")
    if not path.is_file() or _file_sha256(path) != expected_hash:
        raise WorkflowError("resolved training data manifest identity changed")
    return path


def _execute_sentinel_profile(**kwargs: Any) -> dict[str, Any]:
    from multilingual_megatron.pipeline.stages.training.checkpoint_state import (
        inspect_full_state_checkpoint,
    )

    runtime = kwargs["runtime"]
    probe_role = str(kwargs.get("probe_role", "sentinel"))
    selected_lr_warmup_steps = kwargs.get("selected_lr_warmup_steps")
    config, run_id = _profile_payload(
        base=kwargs["base"],
        runtime=kwargs["runtime"],
        paths=kwargs["paths"],
        job_root=kwargs["job_root"],
        ladder=kwargs["ladder"],
        experts=kwargs["experts"],
        k=kwargs["k"],
        horizons=[kwargs["tokens"]],
        role=probe_role,
        d_tokens=kwargs["tokens"],
        peak_lr=kwargs["peak_lr"],
        parent_checkpoint_root=None,
        model_seed=int(
            kwargs.get("model_seed", _runtime_primary_training_seed(runtime))
        ),
        dataset_random_seed=int(
            kwargs.get(
                "dataset_random_seed", _runtime_primary_training_seed(runtime)
            )
        ),
        sentinel=True,
        sentinel_global_batch_size=int(kwargs["global_batch_size"]),
        sentinel_lr_warmup_steps=(
            int(selected_lr_warmup_steps)
            if selected_lr_warmup_steps is not None
            else None
        ),
        variant_arguments=kwargs.get("variant_arguments"),
    )
    # Performance experiments may supply explicit, recorded implementation
    # overrides without changing the frozen A/B-v2 runtime.  Production callers
    # never pass this field, so their resolved configuration is unchanged.
    performance_overrides = kwargs.get("performance_overrides")
    if performance_overrides is not None:
        if not isinstance(performance_overrides, Mapping):
            raise WorkflowError("performance_overrides must be a mapping")
        env_overrides = performance_overrides.get("env", {})
        training_overrides = performance_overrides.get("training", [])
        launch_overrides = performance_overrides.get("launch", {})
        if not isinstance(env_overrides, Mapping) or not isinstance(
            training_overrides, list
        ) or not isinstance(launch_overrides, Mapping):
            raise WorkflowError(
                "performance_overrides requires env/launch mappings and training list"
            )
        profiles = config.get("profiles")
        profile = (
            profiles.get(_workflow_profile_name(runtime))
            if isinstance(profiles, dict)
            else None
        )
        runtime_config = profile.get("runtime") if isinstance(profile, dict) else None
        config_env = runtime_config.get("env") if isinstance(runtime_config, dict) else None
        training = profile.get("training") if isinstance(profile, dict) else None
        if not isinstance(config_env, dict) or not isinstance(training, dict):
            raise WorkflowError("sentinel profile lacks mutable performance config")
        config_env.update({str(key): str(value) for key, value in env_overrides.items()})
        launch = training.get("launch")
        if not isinstance(launch, dict):
            raise WorkflowError("sentinel profile lacks mutable performance launch config")
        launch.update({str(key): value for key, value in launch_overrides.items()})
        existing = list(training.get("overrides", []))
        replacement_keys = {
            str(value).split("=", 1)[0]
            for value in training_overrides
            if isinstance(value, str) and "=" in value
        }
        if len(replacement_keys) != len(training_overrides):
            raise WorkflowError("every performance training override must be key=value")
        training["overrides"] = [
            value
            for value in existing
            if str(value).split("=", 1)[0] not in replacement_keys
        ] + [str(value) for value in training_overrides]
    config_path = kwargs["job_root"] / "resolved_pipeline_config.json"
    _atomic_json(config_path, config)
    started = time.perf_counter()
    _run(
        [
            "python",
            "-m",
            "multilingual_megatron.pipeline",
            "--config",
            str(config_path),
            "--profile",
            _workflow_profile_name(runtime),
            "--start-at",
            "train",
            "--stop-after",
            "train",
            "--executor-type",
            "local",
            "--run-id",
            run_id,
            "--attempt-id",
            _workflow_attempt_id(
                runtime,
                role=(
                    "prescribed_lr_schedule_probe"
                    if selected_lr_warmup_steps
                    else "sentinel"
                ),
            ),
        ]
    )
    manifests = list((kwargs["job_root"] / "runs").rglob("training_run_manifest.json"))
    if len(manifests) != 1:
        raise WorkflowError("sentinel did not produce exactly one training manifest")
    payload = _load_object(manifests[0])
    metadata = payload.get("metadata", {})
    log_path = Path(str(payload.get("log_path", "")))
    health = _parse_sentinel_health(
        log_path,
        require_moe=int(kwargs["experts"]) > 1,
        expected_learning_rate=(
            float(kwargs["peak_lr"]) if selected_lr_warmup_steps is not None else None
        ),
        allow_linear_warmup=selected_lr_warmup_steps is not None,
        throughput_window_size=(
            32 if str(probe_role).startswith("long_gbs") else 12
        ),
    )
    checkpoint_root_text = str(metadata.get("final_checkpoint_root", "")).strip()
    checkpoint_step = int(metadata.get("final_checkpoint_step", 0) or 0)
    expected_samples = int(
        metadata.get("final_checkpoint_expected_consumed_samples", 0) or 0
    )
    if not checkpoint_root_text or checkpoint_step <= 0 or expected_samples <= 0:
        raise WorkflowError(
            "sentinel manifest lacks exact full-state checkpoint metadata"
        )
    checkpoint_state = inspect_full_state_checkpoint(
        checkpoint_root=Path(checkpoint_root_text),
        expected_step=checkpoint_step,
        expected_consumed_samples=expected_samples,
    )
    if not checkpoint_state.valid_full_state:
        raise WorkflowError(
            f"sentinel full-state checkpoint is invalid: {checkpoint_state.errors}"
        )
    end_to_end_seconds = time.perf_counter() - started
    if not math.isfinite(end_to_end_seconds) or end_to_end_seconds <= 0:
        raise WorkflowError("sentinel end-to-end runtime measurement is invalid")
    tier_manifest = _resolved_profile_data_manifest(config, runtime)
    return {
        "schema_version": 1,
        "status": "complete",
        "role": probe_role,
        "fit_observation": False,
        "constant_after_warmup": selected_lr_warmup_steps is not None,
        "warmup_steps": selected_lr_warmup_steps,
        "warmup_tokens": (
            PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS
            if selected_lr_warmup_steps is not None
            else None
        ),
        "resolved_pipeline_config": str(config_path.resolve()),
        "resolved_pipeline_config_sha256": _file_sha256(config_path),
        "data_tier": f"K{int(kwargs['k'])}",
        "data_manifest": str(tier_manifest),
        "data_manifest_sha256": _file_sha256(tier_manifest),
        "training_manifest": str(manifests[0].resolve()),
        "training_manifest_sha256": _file_sha256(manifests[0]),
        "final_checkpoint_root": checkpoint_root_text,
        "final_checkpoint_step": checkpoint_step,
        "expected_consumed_samples": expected_samples,
        "actual_end_tokens": expected_samples * 2048,
        "checkpoint_state": checkpoint_state.to_dict(),
        "sentinel_health": health,
        "end_to_end_seconds": end_to_end_seconds,
    }


def _branch_parent_for_horizon(trunk_result: Mapping[str, Any], horizon: int) -> str:
    plan_path = Path(str(trunk_result.get("segment_plan", "")))
    if not plan_path.is_file():
        raise WorkflowError(f"trunk has no segment plan for D={horizon}: {plan_path}")
    plan = _load_object(plan_path)
    expected_tokens = int(0.8 * horizon)
    candidates = [
        row
        for row in plan.get("segments", [])
        if int(row.get("target_end_tokens", row.get("actual_end_tokens", 0)))
        == expected_tokens
    ]
    if len(candidates) != 1:
        # Endpoint planner records exact stable checkpoint tokens separately;
        # use cumulative consumed tokens when a rounded batch segment endpoint is shown.
        stable = plan.get("stable_checkpoint_tokens", {})
        target_step = (
            stable.get(str(horizon), stable.get(horizon))
            if isinstance(stable, dict)
            else None
        )
        candidates = [
            row
            for row in plan.get("segments", [])
            if int(row.get("cumulative_end_step", -1)) == int(target_step or -2)
        ]
    if len(candidates) != 1:
        raise WorkflowError(
            f"could not resolve unique 0.8D branch parent for D={horizon}"
        )
    return str(Path(str(candidates[0]["output_dir"])) / "checkpoints")


def _run_validation_job(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    paths: Paths,
    runtime: Mapping[str, Any],
) -> None:
    args = dict(job["arguments"])
    external_result = args.get("external_training_result")
    if external_result is None:
        training_result_path = paths.jobs / str(args["training_job_id"]) / "result.json"
    else:
        if not isinstance(external_result, Mapping) or set(external_result) != {
            "path",
            "sha256",
        }:
            raise WorkflowError("external training result must contain path and sha256")
        training_result_path = _abs(str(external_result["path"]))
        if (
            not training_result_path.is_file()
            or _file_sha256(training_result_path) != external_result["sha256"]
        ):
            raise WorkflowError("external training result identity changed")
    training_result = _load_object(training_result_path)
    job_root = paths.jobs / str(job["job_id"])
    job_root.mkdir(parents=True, exist_ok=True)
    spec = build_workflow_evaluation_spec(
        run_id=str(job["job_id"]),
        arguments=args,
        training_result=training_result,
        paths=paths,
        runtime=runtime,
        output_dir=job_root,
    )
    spec_path = job_root / "evaluation_spec.json"
    _atomic_json(spec_path, spec)
    _run(
        [
            "python",
            "-m",
            "multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_per_language_validation",
            "--bridge-root",
            str(_abs("external/Megatron-Bridge")),
            "--spec",
            str(spec_path),
            "--python-executable",
            str(runtime["bridge_python_executable"]),
            "--nproc-per-node",
            str(active_variant().processes_for_experts(int(args["experts"]))),
            "--preflight-cuda",
        ]
    )


def build_workflow_evaluation_spec(
    *,
    run_id: str,
    arguments: Mapping[str, Any],
    training_result: Mapping[str, Any],
    paths: Paths,
    runtime: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build the complete immutable spec consumed by the stage-owned evaluator."""

    args = dict(arguments)
    k = int(args["K"])
    ladder = str(args["ladder"])
    experts = int(args["experts"])
    trained_languages = _languages_from_map(_abs(_language_map(k)))
    integration_languages = args.get("integration_languages")
    if integration_languages is not None:
        if integration_languages != [SENTINEL_INTEGRATION_LANGUAGE]:
            raise WorkflowError(
                "evaluator integration sentinel must use exactly the frozen English language"
            )
        if SENTINEL_INTEGRATION_LANGUAGE not in trained_languages:
            raise WorkflowError(
                "English integration sentinel language is absent from the training tier"
            )
        trained_languages = [SENTINEL_INTEGRATION_LANGUAGE]
        required_languages = [SENTINEL_INTEGRATION_LANGUAGE]
    else:
        required_languages = (
            _languages_from_map(_abs(_language_map(200)))
            if k == 1
            else trained_languages
        )
    corpus_manifest = paths.release / "corpus_manifest.json"
    validation_manifest = paths.tokenized / "validation_megatron_data_manifest.json"
    shape = {
        "A": {
            "num_layers": 8,
            "hidden_size": 512,
            "ffn_hidden_size": 1536,
            "num_attention_heads": 8,
        },
        "B": {
            "num_layers": 12,
            "hidden_size": 768,
            "ffn_hidden_size": 2304,
            "num_attention_heads": 12,
        },
        "C": {
            "num_layers": 16,
            "hidden_size": 1024,
            "ffn_hidden_size": 3072,
            "num_attention_heads": 16,
        },
        "D": {
            "num_layers": 20,
            "hidden_size": 1280,
            "ffn_hidden_size": 3840,
            "num_attention_heads": 20,
        },
    }[ladder]
    expected_model: dict[str, Any] = {
        **shape,
        "seq_length": 2048,
        "vocab_size": 64000,
        "num_moe_experts": None if experts == 1 else experts,
        "cross_entropy_loss_fusion": True,
        "cross_entropy_fusion_impl": "te",
    }
    training_arguments = training_result.get("arguments", {})
    if not isinstance(training_arguments, Mapping):
        training_arguments = {}
    model_seed = int(
        args.get(
            "model_seed",
            training_arguments.get(
                "model_seed", _runtime_primary_training_seed(runtime)
            ),
        )
    )
    model_overrides = _evaluation_model_overrides(
        experts=experts,
        model_seed=model_seed,
        tokenizer_model=paths.tokenizer / "tokenizer.model",
    )
    if experts > 1:
        expected_model.update(
            {
                "moe_router_topk": 1,
                "moe_expert_capacity_factor": None,
                "moe_pad_expert_input_to_capacity": False,
                "moe_grouped_gemm": True,
                "moe_permute_fusion": True,
                "moe_router_fusion": True,
                "moe_token_dispatcher_type": "allgather",
            }
        )
    spec = {
        "schema": "multilingual_per_language_checkpoint_eval/v1",
        "run_id": str(run_id),
        "output_dir": str(output_dir),
        "model_recipe": {
            "name": f"{'scaling_dense' if experts == 1 else 'scaling_switch_moe'}_scale_scaling_{ladder.lower()}_pretrain_config",
            "overrides": model_overrides,
            "expected_model": expected_model,
        },
        "sequence_length": 2048,
        "training_seed": model_seed,
        # Evaluation has no backward/optimizer state. Keep its independently
        # benchmarked execution microbatch separate from the frozen training
        # batch-ramp microbatch; it does not change any token denominator or
        # optimizer update.
        "eval_micro_batch_size": int(
            runtime["training"].get(
                "evaluation_micro_batch_size", runtime["training"]["micro_batch_size"]
            )[ladder][str(experts)]
        ),
        "require_moe_routing_diagnostics": True,
        "context_token_id": 2,
        "trained_languages": trained_languages,
        "required_languages": required_languages,
        "cohorts": {
            "english": _languages_from_map(_abs(_language_map(1))),
            "k12": _languages_from_map(_abs(_language_map(12))),
            "k50": _languages_from_map(_abs(_language_map(50))),
        },
        "checkpoint": {
            "path": str(training_result["final_checkpoint_root"]),
            "step": int(training_result["final_checkpoint_step"]),
        },
        "validation_data_manifest": {
            "path": str(validation_manifest),
            "sha256": _file_sha256(validation_manifest),
        },
        "corpus_manifest": {
            "path": str(corpus_manifest),
            "sha256": _file_sha256(corpus_manifest),
        },
    }
    transformed = active_variant().evaluation_spec_transform(
        spec,
        arguments=args,
        training_result=training_result,
        paths=paths,
        runtime=runtime,
        output_dir=output_dir,
    )
    if not isinstance(transformed, dict):
        raise WorkflowError("workflow variant returned an invalid evaluation spec")
    return transformed
