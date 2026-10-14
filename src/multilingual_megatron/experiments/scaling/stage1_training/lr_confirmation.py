#!/usr/bin/env python3
"""Dedicated Stage-1 JointMoE learning-rate confirmation workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.artifacts import (
    _verified_receipt_evidence,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    EXPECTED_MICRO_BATCH_SIZE,
    REQUIRED_TRAINING_ENV_KEYS,
    WorkflowError,
    _abs,
    _load_object,
)
from multilingual_megatron.experiments.scaling.ab_v2.variant import (
    WorkflowPolicy,
    WorkflowVariant,
)
from multilingual_megatron.experiments.scaling.ab_v2.spec import validate_ab_v2_spec


DEFAULT_SPEC = Path("configs/scaling_law/stage1_lr_confirmation_v1/study_spec.json")
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/stage1_lr_confirmation_v1/workflow_runtime.json"
)
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/stage1_lr_confirmation_h100_v1")
WORKFLOW_MODULE = (
    "multilingual_megatron.experiments.scaling.stage1_training.lr_confirmation"
)
PROTOCOL = "stage1_jointmoe_lr_confirmation_v1"


def validate_stage1_lr_confirmation_spec(
    *, repo_root: Path, spec_path: Path, **kwargs: Any
) -> dict[str, Any]:
    """Validate the inherited corpus/model contract and confirmation overlay."""

    base = validate_ab_v2_spec(
        repo_root=repo_root, spec_path=spec_path, **kwargs
    )
    spec = _load_object(spec_path)
    expected = {
        "protocol": PROTOCOL,
        "method_reference": (
            "Bjorck et al. arXiv:2409.19913 adapted five-point logarithmic bracket"
        ),
        "learning_rate_reference": (
            "JointMoE arXiv:2502.05172 printed rounded-coefficient law"
        ),
        "ladder": "B",
        "N_act_nonembed": 92_012_544,
        "experts": [1, 16, 32],
        "language_counts": [1, 12, 200],
        "jointmoe_multipliers": [0.25, 0.5, 1.0, 2.0, 4.0],
        "historical_fixed_control_peak_lr": 0.0002,
        "training_seed": 1234,
        "corpus_seed": 42,
        "nominal_horizon_tokens": 2_000_000_000,
        "stable_trunk_tokens": 1_600_000_000,
        "cooldown_tokens": 400_000_000,
        "primary_metric": "equal_language_atlas_tao_adjusted_loss",
        "negative_primary_values_are_valid": True,
        "lower_is_better": True,
        "selection_rule": {
            "equal_weight_per_E_K_cell": True,
            "x1_strictly_lowest_mean_regret": True,
            "minimum_x1_cell_wins_or_ties": 5,
            "x1_must_beat_fixed_2e4_in_aggregate": True,
            "outer_boundary_best_or_tied_is_inconclusive": True,
            "tie_absolute_tolerance": 1e-12,
        },
        "automatic_production_submission": False,
        "inference_scope": (
            "short 2B-token confirmation of a transferable LR prescription; "
            "not a long-horizon optimum proof"
        ),
    }
    if spec.get("lr_confirmation") != expected:
        raise WorkflowError("Stage-1 LR-confirmation scientific contract drifted")
    return {**base, "lr_confirmation_protocol": PROTOCOL, "trajectories": 54}


_REQUIRED_SENTINELS = frozenset(
    (kind, "B", experts, gbs)
    for experts in (1, 16, 32)
    for kind, gbs in (
        ("early_correctness", 32),
        ("mid_ramp", 64),
        ("plateau_memory_perf", 128),
    )
)

LR_CONFIRMATION_POLICY = WorkflowPolicy(
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=("B",),
    expected_micro_batch_size=EXPECTED_MICRO_BATCH_SIZE,
    required_sentinel_probes=_REQUIRED_SENTINELS,
    fixed_lr_schedule_probe_cells=(),
    required_training_env_keys=frozenset(REQUIRED_TRAINING_ENV_KEYS),
    spec_validator=validate_stage1_lr_confirmation_spec,
    prescribed_lr_schedule_probe_cells=(
        ("B", 1, 12),
        ("B", 16, 12),
        ("B", 32, 12),
    ),
)

LR_CONFIRMATION_VARIANT = WorkflowVariant(
    identifier="stage1_lr_confirmation_v1",
    worker_module=WORKFLOW_MODULE,
    policy=LR_CONFIRMATION_POLICY,
    provenance_files=(Path(__file__),),
)


def build_manifest(*, spec_path: Path, runtime_path: Path, output_root: Path):
    return _api.build_manifest(
        spec_path=spec_path,
        runtime_path=runtime_path,
        output_root=output_root,
        variant=LR_CONFIRMATION_VARIANT,
    )


def validate_manifest(payload, *, verify_inputs: bool = True):
    result = _api.validate_manifest(
        payload,
        verify_inputs=verify_inputs,
        variant=LR_CONFIRMATION_VARIANT,
    )
    campaign = payload.get("lr_confirmation_campaign")
    if campaign != {
        "protocol": PROTOCOL,
        "ladder": "B",
        "trajectory_count": 54,
        "production_jobs_present": False,
        "manual_production_authorization_required": True,
        "checkpoint_retention": (
            "retain all final 2B checkpoints through manual scientific review"
        ),
    }:
        raise WorkflowError("manifest lacks the frozen confirmation-only campaign")
    if any(job.get("stage") == "production" for job in payload["jobs"]):
        raise WorkflowError("confirmation manifest unexpectedly contains production jobs")
    return result


def plan_workflow(*, spec: Path, runtime: Path, output_root: Path, force: bool = False):
    return _api.plan_workflow(
        spec=spec,
        runtime=runtime,
        output_root=output_root,
        force=force,
        variant=LR_CONFIRMATION_VARIANT,
    )


def submit_workflow(
    *,
    manifest_path: Path,
    retry_failed: bool = False,
    dry_run: bool = False,
    through_stage: str | None = None,
):
    return _api.submit_workflow(
        manifest_path=manifest_path,
        retry_failed=retry_failed,
        dry_run=dry_run,
        through_stage=through_stage,
        variant=LR_CONFIRMATION_VARIANT,
        validate_manifest_fn=validate_manifest,
    )


def execute_job(manifest_path: Path, job_id: str) -> None:
    _api.execute_job(
        manifest_path,
        job_id,
        variant=LR_CONFIRMATION_VARIANT,
        validate_manifest_fn=validate_manifest,
    )


def _custom_submit(argv: Sequence[str]) -> int:
    command = argv[0]
    parser = argparse.ArgumentParser(prog=f"{WORKFLOW_MODULE} {command}")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    if command == "submit-campaign":
        parser.add_argument("--ack-manifest-sha256", required=True)
    args = parser.parse_args(argv[1:])
    manifest_path = _abs(args.manifest)
    manifest = _load_object(manifest_path)
    validate_manifest(manifest)
    if command == "submit-campaign":
        if args.ack_manifest_sha256 != manifest["manifest_sha256"]:
            raise WorkflowError("campaign acknowledgement does not match manifest hash")
        if not args.dry_run:
            _verified_receipt_evidence(
                manifest, "retention_prune_prescribed_lr_schedule_probes"
            )
        through_stage = None if args.dry_run else "lr_diagnostics"
    else:
        through_stage = "lr_verification"
    state = submit_workflow(
        manifest_path=manifest_path,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
        through_stage=through_stage,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in {"submit-preflight", "submit-campaign"}:
        try:
            return _custom_submit(values)
        except WorkflowError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
    if values and values[0] == "submit":
        print(
            "[ERROR] use submit-preflight or submit-campaign for this workflow",
            file=sys.stderr,
        )
        return 2
    return _api.main(values, variant=LR_CONFIRMATION_VARIANT)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_RUNTIME",
    "DEFAULT_SPEC",
    "LR_CONFIRMATION_POLICY",
    "LR_CONFIRMATION_VARIANT",
    "build_manifest",
    "execute_job",
    "main",
    "plan_workflow",
    "submit_workflow",
    "validate_manifest",
]
