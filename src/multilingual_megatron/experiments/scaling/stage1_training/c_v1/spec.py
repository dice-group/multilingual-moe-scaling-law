"""Fail-closed contract for the C ladder of the multilingual MoE study.

The C ladder inherits the already sealed A/B-v2 scientific policy.  This file
only freezes the new model shape, horizons, and E32 execution layout, so a
later edit cannot silently change shared data, tokenizer, optimizer, or loss
decisions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.ab_v2.spec import validate_ab_v2_spec
from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__)).resolve()
DEFAULT_SPEC = Path("configs/scaling_law/multilingual_moe_c_v1/study_spec.json")


class CSpecValidationError(ValueError):
    """Raised when C's inherited or C-specific contract drifts."""


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CSpecValidationError(f"cannot read C study spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CSpecValidationError("C study spec must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CSpecValidationError(f"{field} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_c_v1_spec(*, repo_root: Path = REPO_ROOT, spec_path: Path = DEFAULT_SPEC) -> None:
    """Validate C and its exact dependency on the frozen A/B-v2 contract."""

    spec_path = spec_path.resolve()
    spec = _load(spec_path)
    if spec.get("schema_version") != 1 or spec.get("study_id") != "multilingual_moe_c_v1":
        raise CSpecValidationError("C spec must use schema 1 and study id multilingual_moe_c_v1")

    inherited = spec.get("inherits")
    if not isinstance(inherited, Mapping):
        raise CSpecValidationError("C spec must declare its sealed A/B-v2 inheritance")
    base_spec = _resolve(spec_path.parent, inherited.get("spec_path"), field="inherits.spec_path")
    expected_hash = inherited.get("spec_sha256")
    if not isinstance(expected_hash, str) or _sha256(base_spec) != expected_hash:
        raise CSpecValidationError("the inherited A/B-v2 spec is missing or its hash changed")
    validate_ab_v2_spec(repo_root=repo_root, spec_path=base_spec)

    expected = {
        "axes": {
            "experts": [1, 4, 16, 32],
            "language_counts": [1, 12, 50, 200],
            "ladders": [
                {
                    "id": "C",
                    "num_attention_heads": 16,
                    "num_layers": 16,
                    "hidden_size": 1024,
                    "ffn_hidden_size": 3072,
                    "N_act": 349175808,
                    "N_act_nonembed": 218103808,
                    "D_max_tokens": 32000000000,
                }
            ],
        },
        "training_policy": {
            "peak_learning_rate": 0.0002,
            "warmup_tokens": 130000000,
            "schedule": "warmup, stable, then linear decay to zero over the final 20% of D_max",
            "checkpoint_horizons_tokens": {
                "C": [
                    500000000,
                    1000000000,
                    2000000000,
                    4000000000,
                    8000000000,
                    16000000000,
                    32000000000,
                ]
            },
        },
        "execution_policy": {
            "E1_E4_E16": {
                "nproc_per_node": 1,
                "expert_model_parallel_size": 1,
                "data_parallel_size": 1,
                "moe_dispatcher": "allgather",
                "cuda_graph_impl": "transformer_engine",
                "cuda_graph_scope": "attn",
                "cuda_graph_warmup_steps": 5,
                "te_rng_tracker": True,
                "nccl_graph_register": 0,
                "optimizer_moments": "FP32 AdamW defaults",
            },
            "E32": {
                "nproc_per_node": 2,
                "expert_model_parallel_size": 2,
                "data_parallel_size": 2,
                "moe_dispatcher": "alltoall",
                "overlap_moe_expert_parallel_comm": True,
                "delay_wgrad_compute": True,
                "distributed_optimizer": True,
                "cuda_graph_impl": "transformer_engine",
                "cuda_graph_scope": "attn",
                "cuda_graph_warmup_steps": 5,
                "te_rng_tracker": True,
                "nccl_graph_register": 0,
                "runtime_environment": {
                    "PROOF_BIND_LOCAL_RANK_CPUS": "socket",
                    "PROOF_ENABLE_MOE_A2A_CPU_SPLIT_LISTS": "1",
                },
                "optimizer_moments": "FP32 AdamW defaults",
            },
        },
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise CSpecValidationError(f"C spec drift in {key}")

    invalid_if = spec.get("run_validity_policy", {}).get("invalid_if")
    required = {
        "target token accounting is incomplete",
        "required full-state checkpoint is missing or corrupt",
        "loss or gradient is non-finite",
        "any required per-language evaluation is missing",
        "any token is dropped by MoE routing",
        "configuration or artifact hashes differ from the frozen plan",
        "expert-parallel routing replicas disagree during validation",
    }
    if not isinstance(invalid_if, list) or set(invalid_if) != required:
        raise CSpecValidationError("C run-validity rules are incomplete or drifted")


__all__ = ["CSpecValidationError", "DEFAULT_SPEC", "validate_c_v1_spec"]
