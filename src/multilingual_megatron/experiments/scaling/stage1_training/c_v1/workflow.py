#!/usr/bin/env python3
"""C-ladder adapter for the shared immutable scaling workflow.

This module owns only the C-specific execution contract.  It supplies that
contract as :class:`WorkflowVariant` data to the shared workflow API; it never
patches A/B globals or private functions.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _workflow
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    REPO_ROOT,
    WorkflowError,
    _file_sha256,
    _load_object,
    _payload_sha256,
    _rel_or_abs,
)
from multilingual_megatron.experiments.scaling.ab_v2.contract import (
    _manifest_without_hash,
)
from multilingual_megatron.experiments.scaling.ab_v2.policy import AB_POLICY
from multilingual_megatron.experiments.scaling.ab_v2.variant import WorkflowVariant
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.spec import validate_c_v1_spec


DEFAULT_SPEC = Path("configs/scaling_law/multilingual_moe_c_v1/study_spec.json")
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_c_v1/workflow_runtime_full.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/h100_c_v1_full")
WORKFLOW_MODULE = "multilingual_megatron.experiments.scaling.stage1_training.c_v1.workflow"
C_LADDERS = ("C",)
C_MBS = {"C": {"1": 16, "4": 32, "16": 16, "32": 8}}
C_SENTINELS = {
    ("early_correctness", "C", 1, 32),
    ("early_correctness", "C", 32, 32),
    ("plateau_memory_perf", "C", 4, 128),
    ("plateau_memory_perf", "C", 16, 128),
    ("plateau_memory_perf", "C", 32, 128),
    ("mid_ramp_ga2", "C", 32, 64),
}
C_LR_PROBES = (("C", 1, 1), ("C", 32, 200))
C_E32_RUNTIME_ENV = {
    # These settings are part of the measured two-H100 EP2/all-to-all path.
    # They remain E32-only so the measured EP1 E1/E4/E16 contract is unchanged.
    "PROOF_BIND_LOCAL_RANK_CPUS": "socket",
    "PROOF_ENABLE_MOE_A2A_CPU_SPLIT_LISTS": "1",
}


def _replace_override(overrides: Sequence[str], value: str) -> list[str]:
    key = value.split("=", 1)[0]
    return [item for item in overrides if item.split("=", 1)[0] != key] + [value]


def _c_resource_selector(resource: str, arguments: Mapping[str, Any]) -> str:
    if (
        resource == "h100"
        and str(arguments.get("ladder", "")) == "C"
        and int(arguments.get("experts", 0) or 0) == 32
    ):
        return "h100_2"
    return resource


def _c_job_environment(
    _manifest: Mapping[str, Any], job: Mapping[str, Any], _runtime: Mapping[str, Any]
) -> Mapping[str, str]:
    arguments = job.get("arguments", {})
    if (
        isinstance(arguments, Mapping)
        and str(arguments.get("ladder", "")) == "C"
        and int(arguments.get("experts", 0) or 0) == 32
    ):
        return C_E32_RUNTIME_ENV
    return {}


def _c_training_overrides(overrides: list[str], *, experts: int, **_kwargs: Any) -> list[str]:
    """Apply the measured C execution route without changing study semantics."""

    for value in (
        "model.cuda_graph_impl=transformer_engine",
        "model.cuda_graph_scope=attn",
        "model.cuda_graph_warmup_steps=5",
        "model.use_te_activation_func=true",
        "model.use_te_rng_tracker=true",
        "rng.te_rng_tracker=true",
    ):
        overrides = _replace_override(overrides, value)
    if experts == 32:
        for value in (
            "model.expert_model_parallel_size=2",
            "model.moe_token_dispatcher_type=alltoall",
            "model.overlap_moe_expert_parallel_comm=true",
            "model.delay_wgrad_compute=true",
            "optimizer.use_distributed_optimizer=true",
            "ddp.use_distributed_optimizer=true",
        ):
            overrides = _replace_override(overrides, value)
    if any("precision_aware" in item or "exp_avg_dtype" in item for item in overrides):
        raise WorkflowError("C execution must retain ordinary FP32 AdamW moments")
    return overrides


def _c_processes(experts: int) -> int:
    return 2 if experts == 32 else 1


def _c_profile_transform(
    profile: dict[str, Any], *, experts: int, **_kwargs: Any
) -> dict[str, Any]:
    training = profile["training"]
    if experts == 32:
        if int(training["nproc_per_node"]) != 2:
            raise WorkflowError("C E32 must use two local processes for EP2")
        profile["runtime"]["env"].update(C_E32_RUNTIME_ENV)
    elif int(training["nproc_per_node"]) != 1:
        raise WorkflowError("C E1/E4/E16 must remain single-GPU EP1")
    return profile


def _c_evaluation_spec(
    spec: dict[str, Any], *, arguments: Mapping[str, Any], **_kwargs: Any
) -> dict[str, Any]:
    if int(arguments["experts"]) == 32:
        spec["model_recipe"]["expected_model"].update(
            {
                "expert_model_parallel_size": 2,
                "moe_token_dispatcher_type": "alltoall",
            }
        )
    return spec


C_POLICY = replace(
    AB_POLICY,
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=C_LADDERS,
    expected_micro_batch_size=C_MBS,
    required_sentinel_probes=frozenset(C_SENTINELS),
    fixed_lr_schedule_probe_cells=C_LR_PROBES,
    required_training_env_keys=frozenset(
        set(AB_POLICY.required_training_env_keys) | {"NCCL_GRAPH_REGISTER"}
    ),
    spec_validator=validate_c_v1_spec,
)
C_VARIANT = WorkflowVariant(
    identifier="c_v1",
    worker_module=WORKFLOW_MODULE,
    policy=C_POLICY,
    resource_selector=_c_resource_selector,
    job_environment_transform=_c_job_environment,
    training_overrides_transform=_c_training_overrides,
    profile_transform=_c_profile_transform,
    evaluation_spec_transform=_c_evaluation_spec,
    processes_for_experts=_c_processes,
    provenance_files=(Path(__file__), Path(__file__).with_name("spec.py"),
                      Path(__file__).with_name("__init__.py")),
)


def _parent_inputs(path: Path) -> list[Path]:
    """Return every runtime parent explicitly reached by JSON inheritance."""

    inputs: list[Path] = []
    current = path.resolve()
    while True:
        raw = json.loads(current.read_text(encoding="utf-8"))
        inherited = raw.get("inherits")
        if inherited is None:
            return inputs
        if not isinstance(inherited, str) or not inherited.strip():
            raise WorkflowError(f"invalid inherited runtime path in {current}")
        parent = Path(inherited)
        current = (
            parent.resolve()
            if parent.is_absolute()
            else (current.parent / parent).resolve()
        )
        if current in inputs:
            raise WorkflowError("runtime inheritance cycle")
        inputs.append(current)


def build_manifest(*, spec_path: Path, runtime_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _workflow.build_manifest(
        spec_path=spec_path,
        runtime_path=runtime_path,
        output_root=output_root,
        variant=C_VARIANT,
    )
    contract = dict(manifest["resolved_training_profile_contract"])
    contract.update(
        {
            "source": "configs/scaling_law/multilingual_moe_c_v1/study_spec.json",
            "cuda_graph_impl": "transformer_engine",
            "cuda_graph_scope": "attn",
            "cuda_graph_warmup_steps": 5,
            "nccl_graph_register": 0,
            "e32_execution": "EP2 all-to-all with overlap, delayed wgrad, and distributed optimizer",
            "e32_runtime_env": dict(C_E32_RUNTIME_ENV),
        }
    )
    manifest["resolved_training_profile_contract"] = contract
    fixed = dict(manifest["fixed_input_sha256"])
    for path in (
        *_parent_inputs(runtime_path),
        Path(spec_path).resolve().parent.parent / "multilingual_moe_ab_v2" / "study_spec.json",
    ):
        fixed[_rel_or_abs(path)] = _file_sha256(path)
    manifest["fixed_input_sha256"] = dict(sorted(fixed.items()))
    manifest["manifest_sha256"] = _payload_sha256(_manifest_without_hash(manifest))
    return manifest


def validate_manifest(
    payload: Mapping[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    if payload.get("study_id") != "multilingual_moe_c_v1":
        raise WorkflowError("C workflow manifest has the wrong study_id")
    if payload.get("workflow_module") != WORKFLOW_MODULE:
        raise WorkflowError("C workflow manifest must launch the C worker module")
    validate_c_v1_spec(repo_root=REPO_ROOT, spec_path=Path(str(payload["spec_path"])))
    result = _workflow.validate_manifest(
        dict(payload), verify_inputs=verify_inputs, variant=C_VARIANT
    )
    runtime = _load_object(Path(str(payload["runtime_path"])))
    expected_dp = {"C": {"1": 1, "4": 1, "16": 1, "32": 2}}
    if runtime.get("training", {}).get("data_parallel_size_by_ladder_and_experts") != expected_dp:
        raise WorkflowError("C must freeze DP=1 for E1/E4/E16 and DP=2 for E32 EP2")
    contract = payload.get("resolved_training_profile_contract", {})
    if not isinstance(contract, Mapping) or contract.get("cuda_graph_impl") != "transformer_engine":
        raise WorkflowError("C manifest must freeze the verified CUDA-graph path")
    if contract.get("e32_runtime_env") != C_E32_RUNTIME_ENV:
        raise WorkflowError("C manifest must freeze the measured E32 EP2 runtime environment")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    return _workflow.main(argv, variant=C_VARIANT)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["C_VARIANT", "build_manifest", "main", "validate_manifest"]
