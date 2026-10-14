#!/usr/bin/env python3
"""Clean Stage-1 C rerun using the B-selected JointMoE x1 LR law."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    REPO_ROOT,
    WorkflowError,
    _file_sha256,
    _load_object,
)
from multilingual_megatron.experiments.scaling.ab_v2.variant import (
    WorkflowVariant,
    variant_scope,
)
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.spec import validate_c_v1_spec
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.workflow import C_E32_RUNTIME_ENV
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.workflow import C_MBS
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.workflow import C_VARIANT
from multilingual_megatron.experiments.scaling.stage1_training.common import adopt_synchronous_preflight
from multilingual_megatron.experiments.scaling.stage1_training.common import validate_clean_production_manifest
from multilingual_megatron.experiments.scaling.stage1_training.common import validate_lr_only_runtime
from multilingual_megatron.experiments.scaling.stage1_training.common import validate_production_spec as _validate_jointmoe_production_spec


DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_c_v1/study_spec_jointmoe_x1_production_v1.json"
)
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_c_v1/workflow_runtime_jointmoe_x1_production_v1.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/stage1_c_jointmoe_x1_production_v1")
WORKFLOW_MODULE = (
    "multilingual_megatron.experiments.scaling.stage1_training.c_production"
)


def validate_production_spec(
    *, repo_root: Path = REPO_ROOT, spec_path: Path = DEFAULT_SPEC, **_kwargs: Any
) -> dict[str, Any]:
    return validate_production_spec_base(
        repo_root=repo_root,
        spec_path=spec_path,
    )


def validate_production_spec_base(
    *, repo_root: Path, spec_path: Path
) -> dict[str, Any]:
    return _validate_jointmoe_production_spec(
        repo_root=repo_root,
        spec_path=spec_path,
        ladder="C",
        parent_relative_path="study_spec.json",
        parent_sha256="f3bc43ec4a0ce0134dc1c2bc5101d2b68d676a53a55758dddf870366fb98473f",
        parent_validator=validate_c_v1_spec,
    )


C_JOINTMOE_POLICY = replace(
    C_VARIANT.policy,
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    expected_micro_batch_size=C_MBS,
    required_sentinel_probes=frozenset(),
    fixed_lr_schedule_probe_cells=(),
    prescribed_lr_schedule_probe_cells=(),
    spec_validator=validate_production_spec_base,
)
C_JOINTMOE_VARIANT = WorkflowVariant(
    identifier="stage1_c_jointmoe_x1_production_v1",
    worker_module=WORKFLOW_MODULE,
    policy=C_JOINTMOE_POLICY,
    resource_selector=C_VARIANT.resource_selector,
    job_environment_transform=C_VARIANT.job_environment_transform,
    training_overrides_transform=C_VARIANT.training_overrides_transform,
    profile_transform=C_VARIANT.profile_transform,
    evaluation_spec_transform=C_VARIANT.evaluation_spec_transform,
    processes_for_experts=C_VARIANT.processes_for_experts,
    throughput_evidence_required=C_VARIANT.throughput_evidence_required,
    authorized_production=True,
    provenance_files=(
        Path(__file__),
        Path(__file__).with_name("common.py"),
        *C_VARIANT.provenance_files,
    ),
)


def _systems_evidence_paths(runtime: Mapping[str, Any]) -> tuple[Path, ...]:
    expected = {
        "source_runtime_path": "configs/scaling_law/multilingual_moe_c_v1/workflow_runtime.json",
        "source_runtime_sha256": "e025b3eb3bb32462ce9c9297a80e03b445bd019a28f372159915f5853a6f8ecb",
        "scope": "topology, memory, throughput, evaluation rate, and walltime only; not LR validity",
    }
    if runtime.get("systems_evidence_adoption") != expected:
        raise WorkflowError("C systems-evidence adoption drifted")
    paths = [(REPO_ROOT / expected["source_runtime_path"]).resolve()]
    if _file_sha256(paths[0]) != expected["source_runtime_sha256"]:
        raise WorkflowError("C systems runtime hash drifted")
    if _load_object(paths[0]).get("study_id") != "multilingual_moe_c_v1":
        raise WorkflowError("C systems runtime targets another study")
    evidence = runtime["performance_budget"]["training_throughput_evidence"]
    paths.extend(
        (REPO_ROOT / str(evidence[key]["report"])).resolve()
        for key in ("GBS32", "GBS64", "GBS128")
    )
    return tuple(paths)


def build_manifest(
    *, spec_path: Path, runtime_path: Path, output_root: Path
) -> dict[str, Any]:
    manifest = _api.build_manifest(
        spec_path=spec_path,
        runtime_path=runtime_path,
        output_root=output_root,
        variant=C_JOINTMOE_VARIANT,
    )
    runtime = _load_object(runtime_path)
    contract = dict(manifest["resolved_training_profile_contract"])
    contract.update(
        {
            "source": str(DEFAULT_SPEC),
            "cuda_graph_impl": "transformer_engine",
            "cuda_graph_scope": "attn",
            "cuda_graph_warmup_steps": 5,
            "nccl_graph_register": 0,
            "e32_execution": "EP2 all-to-all with overlap, delayed wgrad, and distributed optimizer",
            "e32_runtime_env": dict(C_E32_RUNTIME_ENV),
            "only_scientific_change_from_historical_c": "peak learning rate",
        }
    )
    manifest["resolved_training_profile_contract"] = contract
    with variant_scope(C_JOINTMOE_VARIANT):
        adopt_synchronous_preflight(
            manifest,
            runtime,
            systems_evidence_paths=_systems_evidence_paths(runtime),
        )
    return manifest


def validate_manifest(
    payload: Mapping[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    if payload.get("study_id") != "multilingual_moe_c_v1":
        raise WorkflowError("C JointMoE manifest has the wrong study_id")
    if payload.get("workflow_module") != WORKFLOW_MODULE:
        raise WorkflowError("C JointMoE manifest has the wrong worker module")
    validate_production_spec_base(
        repo_root=REPO_ROOT, spec_path=Path(str(payload["spec_path"]))
    )
    result = _api.validate_manifest(
        dict(payload),
        verify_inputs=verify_inputs,
        variant=C_JOINTMOE_VARIANT,
    )
    validate_clean_production_manifest(payload, ladder="C")
    runtime = _load_object(Path(str(payload["runtime_path"])))
    _systems_evidence_paths(runtime)
    validate_lr_only_runtime(
        runtime,
        historical_runtime_path=Path(
            "configs/scaling_law/multilingual_moe_c_v1/workflow_runtime.json"
        ),
    )
    if runtime["training"]["data_parallel_size_by_ladder_and_experts"] != {
        "C": {"1": 1, "4": 1, "16": 1, "32": 2}
    }:
        raise WorkflowError("C JointMoE topology drifted")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "submit":
        print(
            "[ERROR] submit C and D through stage1_cd_jointmoe_x1_launch for the frozen heavy-first order",
            file=sys.stderr,
        )
        return 2
    return _api.main(
        values,
        variant=C_JOINTMOE_VARIANT,
        build_manifest_fn=build_manifest,
        validate_manifest_fn=validate_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C_JOINTMOE_VARIANT",
    "DEFAULT_OUTPUT",
    "DEFAULT_RUNTIME",
    "DEFAULT_SPEC",
    "build_manifest",
    "main",
    "validate_manifest",
]
