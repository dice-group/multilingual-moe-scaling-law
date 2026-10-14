#!/usr/bin/env python3
"""Clean Stage-1 A+B production rerun using the JointMoE x1 LR law."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    EXPECTED_MICRO_BATCH_SIZE,
    PRESCRIBED_LR_SCHEDULE_PROBE_CELLS,
    REQUIRED_SENTINEL_PROBES,
    REQUIRED_TRAINING_ENV_KEYS,
    WorkflowError,
    _load_object,
)
from multilingual_megatron.experiments.scaling.ab_v2.variant import WorkflowPolicy, WorkflowVariant
from multilingual_megatron.experiments.scaling.ab_v2.spec import validate_ab_v2_spec


DEFAULT_SPEC = Path("configs/scaling_law/multilingual_moe_ab_v2/study_spec_jointmoe_x1_production_v1.json")
DEFAULT_RUNTIME = Path("configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime_jointmoe_x1_production_v1.json")
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/stage1_ab_jointmoe_x1_production_v1")
WORKFLOW_MODULE = "multilingual_megatron.experiments.scaling.stage1_training.ab_production"


def validate_production_spec(*, repo_root: Path, spec_path: Path, **kwargs: Any) -> dict[str, Any]:
    base = validate_ab_v2_spec(repo_root=repo_root, spec_path=spec_path, **kwargs)
    decision = _load_object(spec_path).get("production_lr_decision")
    expected = {
        "status": "authorized_with_disclosed_boundary",
        "selected_rule": "jointmoe_x1_aggregate_a_priori",
        "support": "B-ladder H100 confirmation: x1 is the aggregate leader and passes three positive checks",
        "limitation": "B/E1/K1 selected x4, so the preregistered confirmation decision is inconclusive_boundary",
        "deferred_boundary_cell": "B/E1/K1:x4",
        "automatic_per_cell_lr_selection": False,
        "inference_scope": "Prospective uniform LR-law choice for the clean A+B rerun; not a claim that x1 is optimal in every cell",
    }
    if decision != expected:
        raise WorkflowError("production LR decision overlay drifted")
    return {**base, "production_lr_decision": decision["status"]}


POLICY = WorkflowPolicy(
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=("A", "B"),
    expected_micro_batch_size=EXPECTED_MICRO_BATCH_SIZE,
    required_sentinel_probes=frozenset(REQUIRED_SENTINEL_PROBES),
    fixed_lr_schedule_probe_cells=(),
    required_training_env_keys=frozenset(REQUIRED_TRAINING_ENV_KEYS),
    spec_validator=validate_production_spec,
    prescribed_lr_schedule_probe_cells=PRESCRIBED_LR_SCHEDULE_PROBE_CELLS,
)
VARIANT = WorkflowVariant(
    identifier="stage1_ab_jointmoe_x1_production_v1",
    worker_module=WORKFLOW_MODULE,
    policy=POLICY,
    authorized_production=True,
    provenance_files=(Path(__file__),),
)


def build_manifest(*, spec_path: Path, runtime_path: Path, output_root: Path):
    return _api.build_manifest(spec_path=spec_path, runtime_path=runtime_path, output_root=output_root, variant=VARIANT)


def validate_manifest(payload, *, verify_inputs: bool = True):
    result = _api.validate_manifest(payload, verify_inputs=verify_inputs, variant=VARIANT)
    jobs = payload["jobs"]
    if len(jobs) != 406 or any(job["action"] == "checkpoint_prune" for job in jobs):
        raise WorkflowError("clean production manifest must contain 406 jobs and no pruning")
    if payload.get("role_counts") != {
        "prescribed_lr_schedule_probes": 4,
        "production_trunks": 32,
        "production_branches": 176,
        "production_evaluations": 176,
    }:
        raise WorkflowError("clean production role cardinality drifted")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    return _api.main(argv, variant=VARIANT)


if __name__ == "__main__":
    raise SystemExit(main())
