#!/usr/bin/env python3
"""Independent Stage-1 A-ladder JointMoE LR confirmation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    EXPECTED_MICRO_BATCH_SIZE,
    REQUIRED_TRAINING_ENV_KEYS,
    WorkflowError,
    _load_object,
)
from multilingual_megatron.experiments.scaling.ab_v2.variant import WorkflowPolicy, WorkflowVariant
from multilingual_megatron.experiments.scaling.ab_v2.spec import validate_ab_v2_spec


DEFAULT_SPEC = Path("configs/scaling_law/stage1_lr_confirmation_a_h100_v1/study_spec.json")
DEFAULT_RUNTIME = Path("configs/scaling_law/stage1_lr_confirmation_a_h100_v1/workflow_runtime.json")
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/stage1_lr_confirmation_a_h100_v1")
WORKFLOW_MODULE = "multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation_a"
PROTOCOL = "stage1_jointmoe_lr_confirmation_v1"


def validate_spec(*, repo_root: Path, spec_path: Path, **kwargs: Any) -> dict[str, Any]:
    base = validate_ab_v2_spec(repo_root=repo_root, spec_path=spec_path, **kwargs)
    row = _load_object(spec_path).get("lr_confirmation")
    if not isinstance(row, dict) or row.get("protocol") != PROTOCOL or row.get("ladder") != "A" or row.get("N_act_nonembed") != 27_262_976:
        raise WorkflowError("A-ladder LR-confirmation scientific contract drifted")
    if row.get("experts") != [1, 16, 32] or row.get("language_counts") != [1, 12, 200] or row.get("jointmoe_multipliers") != [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise WorkflowError("A-ladder LR-confirmation panel drifted")
    return {**base, "lr_confirmation_protocol": PROTOCOL, "trajectories": 54}


_REQUIRED_SENTINELS = frozenset(
    (kind, "A", experts, gbs)
    for experts in (1, 16, 32)
    for kind, gbs in (("early_correctness", 32), ("mid_ramp", 64), ("plateau_memory_perf", 128))
)
POLICY = WorkflowPolicy(
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=("A",),
    expected_micro_batch_size=EXPECTED_MICRO_BATCH_SIZE,
    required_sentinel_probes=_REQUIRED_SENTINELS,
    fixed_lr_schedule_probe_cells=(),
    required_training_env_keys=frozenset(REQUIRED_TRAINING_ENV_KEYS),
    spec_validator=validate_spec,
    prescribed_lr_schedule_probe_cells=(("A", 1, 12), ("A", 16, 12), ("A", 32, 12)),
)
VARIANT = WorkflowVariant(
    identifier="stage1_lr_confirmation_a_h100_v1",
    worker_module=WORKFLOW_MODULE,
    policy=POLICY,
    provenance_files=(Path(__file__),),
    throughput_evidence_required=lambda _runtime: False,
)


def build_manifest(*, spec_path: Path, runtime_path: Path, output_root: Path):
    return _api.build_manifest(spec_path=spec_path, runtime_path=runtime_path, output_root=output_root, variant=VARIANT)


def validate_manifest(payload, *, verify_inputs: bool = True):
    result = _api.validate_manifest(payload, verify_inputs=verify_inputs, variant=VARIANT)
    campaign = payload.get("lr_confirmation_campaign")
    if campaign != {
        "protocol": PROTOCOL,
        "ladder": "A",
        "trajectory_count": 54,
        "production_jobs_present": False,
        "manual_production_authorization_required": True,
        "checkpoint_retention": "retain every trunk and final 2B checkpoint through manual scientific review",
    }:
        raise WorkflowError("manifest lacks the frozen A-confirmation campaign")
    if len(payload["jobs"]) != 183 or any(job["action"] == "checkpoint_prune" for job in payload["jobs"]):
        raise WorkflowError("A-confirmation manifest must contain 183 jobs and no pruning")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    return _api.main(argv, variant=VARIANT)


if __name__ == "__main__":
    raise SystemExit(main())
