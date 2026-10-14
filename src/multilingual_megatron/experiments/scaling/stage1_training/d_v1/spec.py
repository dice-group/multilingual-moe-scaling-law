"""Fail-closed scientific and sentinel contract for the Stage-1 D ladder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.ab_v2.spec import validate_ab_v2_spec
from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__)).resolve()
DEFAULT_SPEC = Path("configs/scaling_law/multilingual_moe_d_v1/study_spec.json")


class DSpecValidationError(ValueError):
    """Raised when D's inherited or sentinel-only contract drifts."""


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DSpecValidationError(f"cannot read D study spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DSpecValidationError("D study spec must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_d_v1_spec(
    *, repo_root: Path = REPO_ROOT, spec_path: Path = DEFAULT_SPEC
) -> None:
    """Validate D while intentionally refusing an unmeasured production layout."""

    spec_path = spec_path.resolve()
    spec = _load(spec_path)
    if spec.get("schema_version") != 1 or spec.get("study_id") != "multilingual_moe_d_v1":
        raise DSpecValidationError("D spec must use schema 1 and study id multilingual_moe_d_v1")
    inherited = spec.get("inherits")
    if not isinstance(inherited, Mapping):
        raise DSpecValidationError("D spec must declare sealed A/B-v2 inheritance")
    base = (spec_path.parent / str(inherited.get("spec_path"))).resolve()
    if _sha256(base) != inherited.get("spec_sha256"):
        raise DSpecValidationError("the inherited A/B-v2 spec is missing or changed")
    validate_ab_v2_spec(repo_root=repo_root, spec_path=base)

    ladder = spec.get("axes", {}).get("ladders")
    expected_ladder = [{
        "id": "D",
        "num_attention_heads": 20,
        "num_layers": 20,
        "hidden_size": 1280,
        "ffn_hidden_size": 3840,
        "N_act": 589824000,
        "N_act_nonembed": 425984000,
        "D_max_tokens": 32000000000,
    }]
    if spec.get("axes", {}).get("experts") != [1, 4, 16, 32]:
        raise DSpecValidationError("D expert axis drifted")
    if spec.get("axes", {}).get("language_counts") != [1, 12, 50, 200]:
        raise DSpecValidationError("D language-count axis drifted")
    if ladder != expected_ladder:
        raise DSpecValidationError("D model shape drifted")
    training = spec.get("training_policy", {})
    if training.get("peak_learning_rate") != 0.0002:
        raise DSpecValidationError("D must retain fixed LR=2e-4")
    if training.get("warmup_tokens") != 130000000:
        raise DSpecValidationError("D warmup-token contract drifted")
    if training.get("checkpoint_horizons_tokens", {}).get("D") != [
        500000000, 1000000000, 2000000000, 4000000000,
        8000000000, 16000000000, 32000000000,
    ]:
        raise DSpecValidationError("D checkpoint horizons drifted")
    execution = spec.get("execution_policy", {})
    if execution.get("status") != "sentinel_only_unmeasured":
        raise DSpecValidationError("D must remain sentinel-only before measurements")
    if execution.get("production_generation_enabled") is not False:
        raise DSpecValidationError("D production generation must fail closed")
    if execution.get("microbatch_candidates") != [1, 2, 4, 8, 16]:
        raise DSpecValidationError("D discovery microbatch candidates drifted")
    required_ids = {
        "sentinel_early_correctness_d_e1_k050_gbs32",
        "sentinel_early_correctness_d_e32_k050_gbs32",
        "sentinel_mid_ramp_d_e32_k050_gbs64",
        "sentinel_plateau_memory_perf_d_e4_k050_gbs128",
        "sentinel_plateau_memory_perf_d_e16_k050_gbs128",
        "sentinel_plateau_memory_perf_d_e32_k050_gbs128",
    }
    actual_ids = {item.get("id") for item in execution.get("required_sentinels", [])}
    if actual_ids != required_ids:
        raise DSpecValidationError("D training sentinel matrix is incomplete")
    if set(execution.get("required_evaluator_sentinels", [])) != {
        "sentinel_eval_d_e1_eng", "sentinel_eval_d_e4_eng",
        "sentinel_eval_d_e16_eng", "sentinel_eval_d_e32_eng",
    }:
        raise DSpecValidationError("D evaluator sentinel matrix is incomplete")
    if {item.get("id") for item in execution.get("required_fixed_lr_probes", [])} != {
        "fixed_lr_schedule_probe_d_e1_k001",
        "fixed_lr_schedule_probe_d_e32_k200",
    }:
        raise DSpecValidationError("D fixed-LR sentinel matrix is incomplete")
    long_probes = execution.get("required_long_throughput_probes", [])
    expected_long = {
        (experts, global_batch_size)
        for experts in (1, 4, 16, 32)
        for global_batch_size in (32, 64, 128)
    }
    actual_long = {
        (int(item.get("E", 0)), int(item.get("global_batch_sequences", 0)))
        for item in long_probes
        if isinstance(item, Mapping)
    }
    if actual_long != expected_long or any(
        int(item.get("tokens", 0)) != 134_217_728
        for item in long_probes
        if isinstance(item, Mapping)
    ):
        raise DSpecValidationError("D long-throughput probe matrix is incomplete")
    if execution.get("required_storage_gate") != {
        "retained_data_and_other_artifacts_reserve_bytes": 3298534883328,
        "required_post_allocation_free_headroom_bytes": 3298534883328,
    }:
        raise DSpecValidationError("D storage contract drifted")
    if execution.get("long_probe_step_time_cv_limit_pct") != 5.0:
        raise DSpecValidationError("D default long-probe CV limit must remain 5%")
    if execution.get("long_probe_step_time_cv_limit_pct_overrides") != {
        "sentinel_long_gbs32_d_e16_k050_gbs32": 8.0,
        "sentinel_long_gbs128_d_e16_k050_gbs128": 12.0,
        "sentinel_long_gbs32_d_e32_k050_gbs32": 10.0,
        "sentinel_long_gbs64_d_e32_k050_gbs64": 8.0,
        "sentinel_long_gbs128_d_e32_k050_gbs128": 12.0,
    }:
        raise DSpecValidationError("D scoped long-probe CV amendments drifted")
    if execution.get("long_probe_p99_to_median_limit_overrides") != {
        "sentinel_long_gbs32_d_e16_k050_gbs32": 1.35,
        "sentinel_long_gbs128_d_e16_k050_gbs128": 1.35,
        "sentinel_long_gbs32_d_e32_k050_gbs32": 1.35,
        "sentinel_long_gbs64_d_e32_k050_gbs64": 1.35,
        "sentinel_long_gbs128_d_e32_k050_gbs128": 1.35,
    }:
        raise DSpecValidationError("D scoped long-probe tail-latency amendments drifted")
    if not str(execution.get("stability_amendment", "")).strip():
        raise DSpecValidationError("D stability amendment rationale is missing")
    if execution.get("gpu_hour_limit") != 3200.0:
        raise DSpecValidationError("D cost cap must be 3,200 H100-hours")
    if not str(execution.get("gpu_hour_limit_amendment", "")).strip():
        raise DSpecValidationError("D cost-cap amendment rationale is missing")
    invalid_if = spec.get("run_validity_policy", {}).get("invalid_if")
    if set(invalid_if or []) != {
        "target token accounting is incomplete",
        "required full-state checkpoint is missing or corrupt",
        "loss or gradient is non-finite",
        "any required per-language evaluation is missing",
        "any token is dropped by MoE routing",
        "configuration or artifact hashes differ from the frozen plan",
        "expert-parallel routing replicas disagree during validation",
    }:
        raise DSpecValidationError("D run-validity rules are incomplete or drifted")


__all__ = ["DEFAULT_SPEC", "DSpecValidationError", "validate_d_v1_spec"]
