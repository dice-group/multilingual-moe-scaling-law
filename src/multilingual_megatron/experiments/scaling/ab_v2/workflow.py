#!/usr/bin/env python3
"""Public A/B-v2 CLI and API compatibility façade.

Study implementation belongs to ``experiments.scaling.ab_v2``.  This module is
intentionally limited to the stable A/B entry points used by existing Slurm
scripts; owner-level helpers are imported from their named modules instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import api as _api
from .common import (
    DEFAULT_OUTPUT,
    DEFAULT_RUNTIME,
    DEFAULT_SPEC,
    Paths,
    WorkflowError,
)
from .policy import AB_VARIANT


def build_manifest(*, spec_path: Path, runtime_path: Path, output_root: Path):
    return _api.build_manifest(
        spec_path=spec_path,
        runtime_path=runtime_path,
        output_root=output_root,
        variant=AB_VARIANT,
    )


def validate_manifest(payload, *, verify_inputs: bool = True):
    return _api.validate_manifest(
        payload,
        verify_inputs=verify_inputs,
        variant=AB_VARIANT,
    )


def plan_workflow(*, spec: Path, runtime: Path, output_root: Path, force: bool = False):
    return _api.plan_workflow(
        spec=spec,
        runtime=runtime,
        output_root=output_root,
        force=force,
        variant=AB_VARIANT,
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
        variant=AB_VARIANT,
        validate_manifest_fn=validate_manifest,
    )


def reconcile_workflow(manifest_path: Path):
    return _api.reconcile_workflow(
        manifest_path,
        variant=AB_VARIANT,
        validate_manifest_fn=validate_manifest,
    )


def collect_workflow(manifest_path: Path):
    return _api.collect_workflow(
        manifest_path,
        variant=AB_VARIANT,
        validate_manifest_fn=validate_manifest,
    )


def execute_job(manifest_path: Path, job_id: str) -> None:
    _api.execute_job(
        manifest_path,
        job_id,
        variant=AB_VARIANT,
        validate_manifest_fn=validate_manifest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return _api.main(argv, variant=AB_VARIANT)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AB_VARIANT",
    "DEFAULT_OUTPUT",
    "DEFAULT_RUNTIME",
    "DEFAULT_SPEC",
    "Paths",
    "WorkflowError",
    "build_manifest",
    "collect_workflow",
    "execute_job",
    "main",
    "plan_workflow",
    "reconcile_workflow",
    "submit_workflow",
    "validate_manifest",
]
