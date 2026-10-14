"""Exploratory composition-aware M5/M6 screen on the frozen Stage 1 folds.

M5 is the exact M2-KN law under a manifest-bound ridge penalty.  M6 adds
three loss-blind exposure-composition coordinates.  This facade freezes the
derived component table, starts every fit from an audited exact-fold M2-KN
basin, and emits Slurm scripts without ever submitting them.

The three primary folds have already been inspected.  Consequently every
artifact produced here is permanently exploratory and cannot replace the
registered M2 result.  Passing the screen only authorizes a separately frozen
nested or prospective confirmation design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

# Updated mechanically after implementation changes; verified before any
# receipt is written or consumed.
SOURCE_BUNDLE: dict[str, str] = {
    "m5_m6_workflow/__init__.py":
        "bdf062696686073b85982fafc26d7a2e5bc3ff64e7cf30c4a35d8577e5d585b0",
    "m5_m6_workflow/analysis.py":
        "f4a9321659242202fcfedaea40a86105eb44aa8ab1e6fca44c31b4556db6dfec",
    "m5_m6_workflow/campaign.py":
        "b289cf5b34c68c400b0e043fb6dd9e5506570897e4f1b5d29e812d045948e4f7",
    "m5_m6_workflow/contracts.py":
        "cab271a5506f9f61a711e6834541343ebf71fa34a1f0e26b0407dd0694f0b2b8",
    "workflow_source.py":
        "5ac146969bd725320269635ae94c3a3686a871d7435239e9cda0beb5a2eac9fc",
}

from .m5_m6_workflow.contracts import (
    PROFILE_SCHEMA, COMPONENT_SCHEMA, INDEX_SCHEMA, AUDIT_INDEX_SCHEMA,
    AUDIT_SCHEMA, REPORT_SCHEMA, MODELS, MODEL_ORDER, COMPONENT_COLUMNS,
    ANCHOR_COUNT, RIDGE_LAMBDAS, M6_SEEDS, CHUNK_SIZE, EQUAL_FOLD_M2_GATE,
    K200_M1_GATE, MINIMUM_EQUIVALENT_BASIN_STARTS, M5M6Error,
    _repository_root, _resolve_path, _binding, load_profile, _fit_rows,
    _validate_sources, _write_component_parquet, prepare, _component_receipt,
    _lambda_key, _historical_m2kn_result, _retained_m2kn_anchors,
    composition_starts, _regularization, _bind_composition, _manifest_kwargs,
    _runner_source_binding,
)
from .m5_m6_workflow.campaign import (
    _write_index, _index, plan_screen, _iter_manifest_items, emit_slurm, status,
    reduce_index, _reduced_current_exploratory, _campaign_result, plan_audits,
)
from .m5_m6_workflow.analysis import (
    _audit_manifest_path, _record_prediction, _bound_scaled_rank,
    _shared_prediction_jacobian_rank, audit, _component_frame, _metric,
    build_report,
)

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--dry-run", action="store_true")
    plan = commands.add_parser("plan-screen")
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--component-receipt", type=Path)
    plan.add_argument("--dry-run", action="store_true")
    for name in ("emit-slurm", "status", "reduce-index"):
        command = commands.add_parser(name)
        command.add_argument("--index", type=Path, required=True)
    audits = commands.add_parser("plan-audits")
    audits.add_argument("--screen-index", type=Path, required=True)
    audit_command = commands.add_parser("audit")
    audit_command.add_argument("--screen-index", type=Path, required=True)
    audit_command.add_argument("--audit-index", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--screen-index", type=Path, required=True)
    report.add_argument("--audit", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect":
        result = prepare(args.profile, dry_run=True)
    elif args.command == "prepare":
        result = prepare(args.profile, dry_run=args.dry_run)
    elif args.command == "plan-screen":
        result = plan_screen(args.profile, component_receipt_path=args.component_receipt,
            dry_run=args.dry_run)
    elif args.command == "emit-slurm":
        result = emit_slurm(args.index)
    elif args.command == "status":
        result = status(args.index)
    elif args.command == "reduce-index":
        result = reduce_index(args.index)
    elif args.command == "plan-audits":
        result = plan_audits(args.screen_index)
    elif args.command == "audit":
        result = audit(args.screen_index, args.audit_index)
    else:
        result = build_report(args.screen_index, args.audit)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
