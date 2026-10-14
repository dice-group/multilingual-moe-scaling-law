"""Small public API for immutable scaling-workflow variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import contract, runner, scheduler
from .common import WorkflowError, _abs, _load_object
from .policy import AB_VARIANT
from .variant import WorkflowVariant, variant_scope


def build_manifest(
    *,
    spec_path: Path,
    runtime_path: Path,
    output_root: Path,
    variant: WorkflowVariant = AB_VARIANT,
) -> dict[str, object]:
    with variant_scope(variant):
        return contract.build_manifest(
            spec_path=spec_path,
            runtime_path=runtime_path,
            output_root=output_root,
        )


def validate_manifest(
    payload: dict[str, object],
    *,
    verify_inputs: bool = True,
    variant: WorkflowVariant = AB_VARIANT,
) -> dict[str, object]:
    with variant_scope(variant):
        return contract.validate_manifest(payload, verify_inputs=verify_inputs)


def plan_workflow(
    *,
    spec: Path,
    runtime: Path,
    output_root: Path,
    force: bool = False,
    variant: WorkflowVariant = AB_VARIANT,
    build_manifest_fn: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    with variant_scope(variant):
        return scheduler.plan_workflow(
            spec=spec,
            runtime=runtime,
            output_root=output_root,
            force=force,
            build_manifest_fn=(
                build_manifest_fn
                if build_manifest_fn is not None
                else lambda **kwargs: build_manifest(variant=variant, **kwargs)
            ),
        )


def submit_workflow(
    *,
    manifest_path: Path,
    retry_failed: bool = False,
    dry_run: bool = False,
    through_stage: str | None = None,
    selected_job_ids: Sequence[str] | None = None,
    variant: WorkflowVariant = AB_VARIANT,
    validate_manifest_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    with variant_scope(variant):
        validator = validate_manifest_fn or (
            lambda payload, **kwargs: validate_manifest(
                payload, variant=variant, **kwargs
            )
        )
        return scheduler.submit_workflow(
            manifest_path=manifest_path,
            retry_failed=retry_failed,
            dry_run=dry_run,
            through_stage=through_stage,
            selected_job_ids=selected_job_ids,
            validate_manifest_fn=validator,
        )


def reconcile_workflow(
    manifest_path: Path,
    *,
    variant: WorkflowVariant = AB_VARIANT,
    validate_manifest_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    with variant_scope(variant):
        validator = validate_manifest_fn or (
            lambda payload, **kwargs: validate_manifest(
                payload, variant=variant, **kwargs
            )
        )
        return scheduler.reconcile_workflow(
            manifest_path,
            validate_manifest_fn=validator,
        )


def collect_workflow(
    manifest_path: Path,
    *,
    variant: WorkflowVariant = AB_VARIANT,
    validate_manifest_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    with variant_scope(variant):
        validator = validate_manifest_fn or (
            lambda payload, **kwargs: validate_manifest(
                payload, variant=variant, **kwargs
            )
        )
        return scheduler.collect_workflow(
            manifest_path,
            validate_manifest_fn=validator,
        )


def execute_job(
    manifest_path: Path,
    job_id: str,
    *,
    variant: WorkflowVariant = AB_VARIANT,
    validate_manifest_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> None:
    with variant_scope(variant):
        validator = validate_manifest_fn or (
            lambda payload, **kwargs: validate_manifest(
                payload, variant=variant, **kwargs
            )
        )
        runner.execute_job(
            manifest_path,
            job_id,
            validate_manifest_fn=validator,
        )


def _parser(variant: WorkflowVariant) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--spec", type=Path, default=variant.policy.default_spec)
    plan.add_argument("--runtime", type=Path, default=variant.policy.default_runtime)
    plan.add_argument("--output-root", type=Path, default=variant.policy.default_output)
    plan.add_argument("--force-plan", action="store_true")
    for command in ("validate", "submit", "collect", "reconcile"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        if command == "submit":
            child.add_argument("--retry-failed", action="store_true")
            child.add_argument("--dry-run", action="store_true")
            child.add_argument(
                "--through-stage",
                choices=(
                    "pilot",
                    "tokenizer",
                    "corpus",
                    "tokenization",
                    "sentinel",
                    "lr_verification",
                    "lr_diagnostics",
                    "production",
                ),
                required=True,
                help="Submit through one resumable phase; array elements count against quota individually.",
            )
    execute = subparsers.add_parser("execute-job")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--job-id", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    variant: WorkflowVariant = AB_VARIANT,
    build_manifest_fn: Callable[..., dict[str, Any]] | None = None,
    validate_manifest_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> int:
    args = _parser(variant).parse_args(argv)
    try:
        if args.command == "plan":
            path = plan_workflow(
                spec=_abs(args.spec),
                runtime=_abs(args.runtime),
                output_root=_abs(args.output_root),
                force=args.force_plan,
                variant=variant,
                build_manifest_fn=build_manifest_fn,
            )
            manifest = _load_object(path)
            validator = validate_manifest_fn or (
                lambda payload, **kwargs: validate_manifest(
                    payload, variant=variant, **kwargs
                )
            )
            print(
                json.dumps(
                    {"manifest": str(path), **validator(manifest)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate":
            validator = validate_manifest_fn or (
                lambda payload, **kwargs: validate_manifest(
                    payload, variant=variant, **kwargs
                )
            )
            print(
                json.dumps(
                    validator(_load_object(_abs(args.manifest))),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "submit":
            state = submit_workflow(
                manifest_path=_abs(args.manifest),
                retry_failed=args.retry_failed,
                dry_run=args.dry_run,
                through_stage=args.through_stage,
                variant=variant,
                validate_manifest_fn=validate_manifest_fn,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if args.command == "reconcile":
            print(
                json.dumps(
                    reconcile_workflow(
                        _abs(args.manifest),
                        variant=variant,
                        validate_manifest_fn=validate_manifest_fn,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "collect":
            report = collect_workflow(
                _abs(args.manifest),
                variant=variant,
                validate_manifest_fn=validate_manifest_fn,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "ok" else 2
        if args.command == "execute-job":
            execute_job(
                _abs(args.manifest),
                args.job_id,
                variant=variant,
                validate_manifest_fn=validate_manifest_fn,
            )
            return 0
    except WorkflowError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)
