#!/usr/bin/env python3
"""Clean Stage-1 D rerun using the B-selected JointMoE x1 LR law."""

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
from multilingual_megatron.experiments.scaling.stage1_training.d_v1.spec import validate_d_v1_spec
from multilingual_megatron.experiments.scaling.stage1_training.d_v1.workflow import D_EP_RUNTIME_ENV
from multilingual_megatron.experiments.scaling.stage1_training.d_v1.workflow import D_MBS
from multilingual_megatron.experiments.scaling.stage1_training.d_v1.workflow import D_VARIANT
from multilingual_megatron.experiments.scaling.stage1_training.common import adopt_synchronous_preflight
from multilingual_megatron.experiments.scaling.stage1_training.common import validate_clean_production_manifest
from multilingual_megatron.experiments.scaling.stage1_training.common import validate_lr_only_runtime
from multilingual_megatron.experiments.scaling.stage1_training.common import validate_production_spec as _validate_jointmoe_production_spec


DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_d_v1/study_spec_jointmoe_x1_production_v1.json"
)
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_d_v1/workflow_runtime_jointmoe_x1_production_v1.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/stage1_d_jointmoe_x1_production_v1")
WORKFLOW_MODULE = (
    "multilingual_megatron.experiments.scaling.stage1_training.d_production"
)


def validate_production_spec_base(
    *, repo_root: Path, spec_path: Path
) -> dict[str, Any]:
    return _validate_jointmoe_production_spec(
        repo_root=repo_root,
        spec_path=spec_path,
        ladder="D",
        parent_relative_path="study_spec_launched_20260723.json",
        parent_sha256="bc6228d23db94fa5dfb0ddad1319e25811a51aed73785108af3acb33b3725d2b",
        parent_validator=validate_d_v1_spec,
    )


D_JOINTMOE_POLICY = replace(
    D_VARIANT.policy,
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    expected_micro_batch_size=D_MBS,
    required_sentinel_probes=frozenset(),
    fixed_lr_schedule_probe_cells=(),
    prescribed_lr_schedule_probe_cells=(),
    spec_validator=validate_production_spec_base,
)
D_JOINTMOE_VARIANT = WorkflowVariant(
    identifier="stage1_d_jointmoe_x1_production_v1",
    worker_module=WORKFLOW_MODULE,
    policy=D_JOINTMOE_POLICY,
    resource_selector=D_VARIANT.resource_selector,
    job_environment_transform=D_VARIANT.job_environment_transform,
    training_overrides_transform=D_VARIANT.training_overrides_transform,
    profile_transform=D_VARIANT.profile_transform,
    evaluation_spec_transform=D_VARIANT.evaluation_spec_transform,
    processes_for_experts=D_VARIANT.processes_for_experts,
    throughput_evidence_required=D_VARIANT.throughput_evidence_required,
    authorized_production=True,
    provenance_files=(
        Path(__file__),
        Path(__file__).with_name("common.py"),
        *D_VARIANT.provenance_files,
    ),
)


def _systems_evidence_paths(runtime: Mapping[str, Any]) -> tuple[Path, ...]:
    expected = {
        "source_runtime_path": "outputs/pipeline_runs/h100_d_v1_full_study_phase1_20260723/control/workflow_runtime_measured.json",
        "source_runtime_sha256": "ead3c73c6c056c2b114db91eeb029acd2ca7674495aecb945acd04472d2855df",
        "promotion_receipt_path": "outputs/pipeline_runs/h100_d_v1_full_study_phase1_20260723/control/runtime_promotion_cost_amended.json",
        "promotion_receipt_sha256": "94e8c38e261a4f7513fb44b8ef6529c50cdae387f8aaa7df9faafbafffe2cbe4",
        "scope": "topology, memory, throughput, evaluation rate, and walltime only; not LR validity",
    }
    if runtime.get("systems_evidence_adoption") != expected:
        raise WorkflowError("D systems-evidence adoption drifted")
    paths = [
        (REPO_ROOT / expected["source_runtime_path"]).resolve(),
        (REPO_ROOT / expected["promotion_receipt_path"]).resolve(),
    ]
    if _file_sha256(paths[0]) != expected["source_runtime_sha256"]:
        raise WorkflowError("D measured systems runtime hash drifted")
    if _file_sha256(paths[1]) != expected["promotion_receipt_sha256"]:
        raise WorkflowError("D systems promotion receipt hash drifted")
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
        variant=D_JOINTMOE_VARIANT,
    )
    runtime = _load_object(runtime_path)
    contract = dict(manifest["resolved_training_profile_contract"])
    contract.update(
        {
            "source": str(DEFAULT_SPEC),
            "cuda_graph_impl": "none",
            "d_execution": "E1/E4 EP1; E16 EP2; E32 EP4; all multi-rank paths use all-to-all",
            "ep_runtime_env": dict(D_EP_RUNTIME_ENV),
            "optimizer_moments": "ordinary FP32 AdamW defaults",
            "only_scientific_change_from_historical_d": "peak learning rate",
        }
    )
    manifest["resolved_training_profile_contract"] = contract
    with variant_scope(D_JOINTMOE_VARIANT):
        adopt_synchronous_preflight(
            manifest,
            runtime,
            systems_evidence_paths=_systems_evidence_paths(runtime),
        )
    return manifest


def validate_manifest(
    payload: Mapping[str, Any], *, verify_inputs: bool = True
) -> dict[str, Any]:
    if payload.get("study_id") != "multilingual_moe_d_v1":
        raise WorkflowError("D JointMoE manifest has the wrong study_id")
    if payload.get("workflow_module") != WORKFLOW_MODULE:
        raise WorkflowError("D JointMoE manifest has the wrong worker module")
    validate_production_spec_base(
        repo_root=REPO_ROOT, spec_path=Path(str(payload["spec_path"]))
    )
    result = _api.validate_manifest(
        dict(payload),
        verify_inputs=verify_inputs,
        variant=D_JOINTMOE_VARIANT,
    )
    validate_clean_production_manifest(payload, ladder="D")
    runtime = _load_object(Path(str(payload["runtime_path"])))
    _systems_evidence_paths(runtime)
    validate_lr_only_runtime(
        runtime,
        historical_runtime_path=Path(
            "outputs/pipeline_runs/h100_d_v1_full_study_phase1_20260723/control/workflow_runtime_measured.json"
        ),
    )
    if runtime["training"]["data_parallel_size_by_ladder_and_experts"] != {
        "D": {"1": 1, "4": 1, "16": 2, "32": 4}
    }:
        raise WorkflowError("D JointMoE topology drifted")
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
        variant=D_JOINTMOE_VARIANT,
        build_manifest_fn=build_manifest,
        validate_manifest_fn=validate_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "D_JOINTMOE_VARIANT",
    "DEFAULT_OUTPUT",
    "DEFAULT_RUNTIME",
    "DEFAULT_SPEC",
    "build_manifest",
    "main",
    "validate_manifest",
]
