#!/usr/bin/env python3
"""Validate and submit clean C+D JointMoE-x1 manifests in heavy-first order."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2 import api as _api
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    REPO_ROOT,
    STATE_NAME,
    WorkflowError,
    _abs,
    _atomic_json,
    _load_object,
    _payload_sha256,
)
from multilingual_megatron.experiments.scaling.stage1_training.c_production import C_JOINTMOE_VARIANT
from multilingual_megatron.experiments.scaling.stage1_training.c_production import validate_manifest as validate_c_manifest
from multilingual_megatron.experiments.scaling.stage1_training.d_production import D_JOINTMOE_VARIANT
from multilingual_megatron.experiments.scaling.stage1_training.d_production import validate_manifest as validate_d_manifest


DEFAULT_C_MANIFEST = Path(
    "outputs/pipeline_runs/stage1_c_jointmoe_x1_production_v1/control/launch_manifest.json"
)
DEFAULT_D_MANIFEST = Path(
    "outputs/pipeline_runs/stage1_d_jointmoe_x1_production_v1/control/launch_manifest.json"
)
ACTIVE_AB_MANIFEST = Path(
    "outputs/pipeline_runs/stage1_ab_jointmoe_x1_production_v1/control/launch_manifest.json"
)
LAUNCH_CONTROL_ROOT = Path(
    "outputs/pipeline_runs/stage1_cd_jointmoe_x1_production_v1/control"
)
COHORTS = (
    ("D", 32, 4),
    ("D", 16, 2),
    ("C", 32, 2),
    ("D", 4, 1),
    ("D", 1, 1),
    ("C", 16, 1),
    ("C", 4, 1),
    ("C", 1, 1),
)
K_ORDER = (200, 50, 12, 1)


def _head_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise WorkflowError(f"cannot resolve repository HEAD: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _worktree_observation() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise WorkflowError(
            "cannot record informational worktree state: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    rows = [row for row in completed.stdout.splitlines() if row]
    return {
        "dirty": bool(rows),
        "porcelain_row_count": len(rows),
        "porcelain_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "policy": "informational_only_never_a_launch_gate",
    }


def _launch_controller_identity() -> dict[str, str]:
    """Require this launch-critical controller, but not the worktree, to be committed."""

    controller = Path(__file__).resolve()
    try:
        relative = controller.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise WorkflowError("combined launch controller is outside the repository") from exc
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relative.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise WorkflowError(
            "combined launch controller is not present in committed HEAD; commit before launch"
        )
    working_bytes = controller.read_bytes()
    if working_bytes != completed.stdout:
        raise WorkflowError(
            "combined launch controller differs from committed HEAD; commit the relevant change "
            "or restore the manifest-bound controller before launch"
        )
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(working_bytes).hexdigest(),
        "commit": _head_commit(),
    }


def _cohort_job_ids(
    manifest: Mapping[str, Any], *, ladder: str, experts: int
) -> list[str]:
    jobs = manifest["jobs"]
    by_id = {str(row["job_id"]): row for row in jobs}
    horizons = sorted(
        {
            int(row["arguments"]["D"])
            for row in jobs
            if str(row["job_id"]).startswith(
                f"prod_branch_{ladder.lower()}_e{experts}_"
            )
        },
        reverse=True,
    )
    if horizons != [32_000_000_000, 16_000_000_000, 8_000_000_000, 4_000_000_000, 2_000_000_000, 1_000_000_000, 500_000_000]:
        raise WorkflowError(f"{ladder}/E{experts} horizon set drifted")
    trunks = [
        f"prod_trunk_{ladder.lower()}_e{experts}_k{k:03d}" for k in K_ORDER
    ]
    branches = [
        f"prod_branch_{ladder.lower()}_e{experts}_k{k:03d}_d{horizon}"
        for horizon in horizons
        for k in K_ORDER
    ]
    evaluations = [job_id.replace("prod_branch_", "prod_eval_", 1) for job_id in branches]
    ordered = [*trunks, *branches, *evaluations]
    missing = [job_id for job_id in ordered if job_id not in by_id]
    if missing:
        raise WorkflowError(
            f"{ladder}/E{experts} heavy-first cohort is incomplete: "
            + ", ".join(missing[:8])
        )
    if len(ordered) != 60 or len(set(ordered)) != 60:
        raise WorkflowError(f"{ladder}/E{experts} cohort must contain 60 unique jobs")
    return ordered


def submission_batches(
    c_manifest: Mapping[str, Any], d_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    manifests = {"C": c_manifest, "D": d_manifest}
    batches: list[dict[str, Any]] = []
    for rank, (ladder, experts, gpus) in enumerate(COHORTS, start=1):
        batches.append(
            {
                "rank": rank,
                "label": f"{ladder}/E{experts}",
                "ladder": ladder,
                "experts": experts,
                "gpus": gpus,
                "job_ids": _cohort_job_ids(
                    manifests[ladder], ladder=ladder, experts=experts
                ),
            }
        )
    batches.extend(
        [
            {
                "rank": 9,
                "label": "D/final_summary",
                "ladder": "D",
                "experts": None,
                "gpus": 0,
                "job_ids": ["final_summary"],
            },
            {
                "rank": 10,
                "label": "C/final_summary",
                "ladder": "C",
                "experts": None,
                "gpus": 0,
                "job_ids": ["final_summary"],
            },
        ]
    )
    gpu_ids = [job_id for batch in batches[:8] for job_id in batch["job_ids"]]
    expected_gpu_ids = {
        str(row["job_id"])
        for manifest in (c_manifest, d_manifest)
        for row in manifest["jobs"]
        if str(row.get("resource", "")).startswith("h100")
    }
    if len(gpu_ids) != 480 or set(gpu_ids) != expected_gpu_ids:
        raise WorkflowError("combined heavy-first order does not cover exactly 480 GPU jobs")
    return batches


def validate_combined(
    c_manifest: Mapping[str, Any], d_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    c_result = validate_c_manifest(c_manifest)
    d_result = validate_d_manifest(d_manifest)
    if c_manifest["data_materialization"] != d_manifest["data_materialization"]:
        raise WorkflowError("C and D do not bind the same sealed corpus")
    if c_manifest["learning_rate_prescription"]["source"] != d_manifest[
        "learning_rate_prescription"
    ]["source"]:
        raise WorkflowError("C and D do not bind the same JointMoE prescription")
    batches = submission_batches(c_manifest, d_manifest)
    return {
        "status": "ok",
        "c": c_result,
        "d": d_result,
        "gpu_jobs": 480,
        "summary_jobs": 2,
        "batches": [
            {
                "rank": batch["rank"],
                "label": batch["label"],
                "gpus": batch["gpus"],
                "jobs": len(batch["job_ids"]),
            }
            for batch in batches
        ],
        "estimated_h100_hours": sum(
            float(manifest["gpu_hour_estimate"]["base_estimated_gpu_hours"])
            for manifest in (c_manifest, d_manifest)
        ),
    }


def _capacity_preflight(
    c_manifest: Mapping[str, Any], d_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Record live capacity without making storage a submission gate.

    The shared filesystem reports soft quota limits, and the manifest parameter-based checkpoint
    estimator is intentionally conservative.  Capacity therefore remains visible
    in the immutable launch receipt, but neither a low reading nor a failed
    statvfs probe may prevent the already validated Slurm DAG from being released.
    This function never deletes or prunes artifacts.
    """

    manifests = [c_manifest, d_manifest]
    active_ab_path = _abs(ACTIVE_AB_MANIFEST)
    if active_ab_path.is_file():
        manifests.append(_load_object(active_ab_path))
    checkpoint_bytes = sum(
        int(manifest["storage_estimate"]["checkpoint_peak_bytes"])
        for manifest in manifests
    )
    noncheckpoint_reserve = max(
        int(
            manifest["storage_estimate"].get(
                "retained_data_and_other_artifacts_reserve_bytes", 0
            )
        )
        for manifest in manifests
    )
    required_free_bytes = math.ceil(checkpoint_bytes * 1.10) + noncheckpoint_reserve
    output_parent = Path(str(c_manifest["output_root"])).resolve().parent
    report = {
        "status": "advisory_unavailable",
        "submission_allowed": True,
        "launch_blocking": False,
        "policy": "capacity_is_observed_and_reported_but_never_a_submission_gate",
        "filesystem_path": str(output_parent),
        "campaigns_included": [str(manifest["study_id"]) for manifest in manifests],
        "projected_checkpoint_bytes": checkpoint_bytes,
        "ten_percent_checkpoint_headroom_bytes": math.ceil(checkpoint_bytes * 0.10),
        "shared_noncheckpoint_reserve_bytes": noncheckpoint_reserve,
        "required_free_bytes": required_free_bytes,
        "required_free_bytes_semantics": "conservative_advisory_reference_only",
        "observed_free_bytes": None,
        "required_free_inodes": None,
        "required_free_inodes_semantics": "no_inode_threshold_is_enforced",
        "observed_free_inodes": None,
        "warnings": [],
        "automatic_cleanup_allowed": False,
    }
    try:
        stats = os.statvfs(output_parent)
    except OSError as exc:
        report["warnings"] = [
            "live statvfs capacity observation failed; submission remains allowed: "
            f"{type(exc).__name__}: {exc}"
        ]
        return report

    free_bytes = int(stats.f_bavail * stats.f_frsize)
    total_inodes = int(stats.f_files)
    free_inodes = int(stats.f_favail)
    advisory_free_inodes = max(100_000, math.ceil(total_inodes * 0.10))
    warnings: list[str] = []
    if free_bytes < required_free_bytes:
        warnings.append(
            "observed free bytes are below the conservative manifest-based "
            "projection; this is advisory and does not block submission"
        )
    if free_inodes < advisory_free_inodes:
        warnings.append(
            "observed free inodes are below the former ten-percent reference; "
            "this is advisory and does not block submission"
        )
    report.update(
        {
            "status": "advisory_warning" if warnings else "advisory_ok",
            "observed_free_bytes": free_bytes,
            "required_free_inodes": advisory_free_inodes,
            "required_free_inodes_semantics": (
                "former_ten_percent_reference_only_not_enforced"
            ),
            "observed_free_inodes": free_inodes,
            "warnings": warnings,
        }
    )
    return report


def _slurm_contract(*manifests: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the shared launch site contract from manifest-bound profiles."""

    partitions: set[str] = set()
    accounts: set[str] = set()
    maximum_hours: set[float] = set()
    runtime_paths: list[str] = []
    for manifest in manifests:
        runtime_path = Path(str(manifest["runtime_path"])).resolve()
        runtime = _load_object(runtime_path)
        runtime_paths.append(str(runtime_path))
        resources = runtime.get("resources")
        if not isinstance(resources, Mapping):
            raise WorkflowError(f"runtime profile has no resources mapping: {runtime_path}")
        used_resources = {
            str(job["resource"])
            for job in manifest["jobs"]
            if str(job.get("resource", "")).startswith("h100")
        }
        for resource_name in used_resources:
            resource = resources.get(resource_name)
            if not isinstance(resource, Mapping):
                raise WorkflowError(
                    f"runtime profile is missing used resource {resource_name!r}: {runtime_path}"
                )
            partition = str(resource.get("partition", "")).strip()
            if not partition:
                raise WorkflowError(f"resource {resource_name!r} has no partition")
            partitions.add(partition)
            for flag in resource.get("extra_sbatch_flags", []):
                if str(flag).startswith("--account="):
                    accounts.add(str(flag).split("=", 1)[1])
        scheduling = runtime.get("scheduling")
        if not isinstance(scheduling, Mapping):
            raise WorkflowError(f"runtime profile has no scheduling contract: {runtime_path}")
        maximum_hours.add(float(scheduling["maximum_job_hours"]))

    if len(partitions) != 1 or len(accounts) != 1 or len(maximum_hours) != 1:
        raise WorkflowError(
            "C/D runtime profiles disagree on Slurm partition, account, or maximum walltime"
        )
    hours = maximum_hours.pop()
    if hours <= 0 or not hours.is_integer():
        raise WorkflowError("maximum_job_hours must be a positive whole hour")
    total_hours = int(hours)
    days, hour = divmod(total_hours, 24)
    expected_max_time = f"{days}-{hour:02d}:00:00" if days else f"{hour:02d}:00:00"
    return {
        "partition": partitions.pop(),
        "account": accounts.pop(),
        "expected_partition_max_time": expected_max_time,
        "runtime_paths": runtime_paths,
    }


def _slurm_preflight(
    c_manifest: Mapping[str, Any], d_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _slurm_contract(c_manifest, d_manifest)
    partition = contract["partition"]
    account = contract["account"]
    commands = {
        name: shutil.which(name)
        for name in ("sbatch", "squeue", "sacct", "scontrol", "sacctmgr")
    }
    missing = [name for name, path in commands.items() if path is None]
    if missing:
        raise WorkflowError("required Slurm commands are missing: " + ", ".join(missing))
    completed = subprocess.run(
        [str(commands["scontrol"]), "show", "partition", partition, "--oneliner"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or f"PartitionName={partition}" not in completed.stdout:
        raise WorkflowError(f"{partition} partition is unavailable to the launch preflight")
    partition_fields = {
        key: value
        for token in completed.stdout.split()
        if "=" in token
        for key, value in (token.split("=", 1),)
    }
    if (
        partition_fields.get("State") != "UP"
        or partition_fields.get("MaxTime") != contract["expected_partition_max_time"]
    ):
        raise WorkflowError(
            f"{partition} is not UP with the profile-bound maximum walltime"
        )
    user = getpass.getuser()
    association = subprocess.run(
        [
            str(commands["sacctmgr"]),
            "--noheader",
            "--parsable2",
            "show",
            "assoc",
            "where",
            f"user={user}",
            f"account={account}",
            "format=User,Account",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if association.returncode or not any(
        row.split("|")[:2] == [user, account]
        for row in association.stdout.splitlines()
    ):
        raise WorkflowError(
            f"Slurm association {user}/{account} is unavailable to the launch preflight"
        )
    return {
        "status": "ok",
        "commands": commands,
        "partition": partition,
        "partition_state": partition_fields["State"],
        "partition_max_time": partition_fields["MaxTime"],
        "account": account,
        "runtime_paths": contract["runtime_paths"],
        "user": user,
        "association_rows": sum(
            row.split("|")[:2] == [user, account]
            for row in association.stdout.splitlines()
        ),
    }


def preflight(
    c_manifest: Mapping[str, Any],
    d_manifest: Mapping[str, Any],
    *,
    require_capacity: bool,
    require_slurm: bool,
) -> dict[str, Any]:
    validation = validate_combined(c_manifest, d_manifest)
    head = _head_commit()
    launch_commits = {
        str(manifest["git_provenance"]["orchestration_repo"]["commit"])
        for manifest in (c_manifest, d_manifest)
    }
    if launch_commits != {head}:
        raise WorkflowError(
            "C/D manifests are not bound to the current committed HEAD; regenerate both after commit"
        )
    report = {
        "status": "ok",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "head_commit": head,
        "launch_controller": _launch_controller_identity(),
        "worktree": _worktree_observation(),
        "validation": validation,
        "capacity": (
            _capacity_preflight(c_manifest, d_manifest)
            if require_capacity
            else {"status": "not_enforced_in_dry_run"}
        ),
        "slurm": (
            _slurm_preflight(c_manifest, d_manifest)
            if require_slurm
            else {"status": "not_enforced_in_dry_run"}
        ),
    }
    report["preflight_sha256"] = _payload_sha256(report)
    return report


def _manifest_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    state_path = Path(str(manifest["output_root"])) / "control" / STATE_NAME
    return _load_object(state_path)


def submit_heavy_first(
    *,
    c_manifest_path: Path,
    d_manifest_path: Path,
    dry_run: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    c_manifest_path = _abs(c_manifest_path)
    d_manifest_path = _abs(d_manifest_path)
    c_manifest = _load_object(c_manifest_path)
    d_manifest = _load_object(d_manifest_path)
    readiness = preflight(
        c_manifest,
        d_manifest,
        require_capacity=not dry_run,
        require_slurm=not dry_run,
    )
    batches = submission_batches(c_manifest, d_manifest)
    if dry_run:
        return {
            "status": "dry_run",
            "preflight": readiness,
            "submission_order": batches,
        }

    manifest_paths = {"C": c_manifest_path, "D": d_manifest_path}
    variants = {"C": C_JOINTMOE_VARIANT, "D": D_JOINTMOE_VARIANT}
    validators = {"C": validate_c_manifest, "D": validate_d_manifest}
    for batch in batches:
        ladder = str(batch["ladder"])
        _api.submit_workflow(
            manifest_path=manifest_paths[ladder],
            retry_failed=retry_failed,
            dry_run=False,
            through_stage=None,
            selected_job_ids=batch["job_ids"],
            variant=variants[ladder],
            validate_manifest_fn=validators[ladder],
        )

    states = {"C": _manifest_state(c_manifest), "D": _manifest_state(d_manifest)}
    slurm_ids = {
        ladder: {
            job_id: value.get("slurm_job_id")
            for job_id, value in state["jobs"].items()
            if value.get("slurm_job_id")
        }
        for ladder, state in states.items()
    }
    receipt = {
        "schema_version": 1,
        "status": "submitted",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "c_manifest": str(c_manifest_path),
        "c_manifest_sha256": c_manifest["manifest_sha256"],
        "d_manifest": str(d_manifest_path),
        "d_manifest_sha256": d_manifest["manifest_sha256"],
        "preflight": readiness,
        "submission_order": batches,
        "slurm_job_ids": slurm_ids,
        "worktree_cleanliness_was_a_gate": False,
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    target = _abs(LAUNCH_CONTROL_ROOT) / "heavy_first_launch_receipt.json"
    _atomic_json(target, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "preflight", "submit"):
        child = subparsers.add_parser(command)
        child.add_argument("--c-manifest", type=Path, default=DEFAULT_C_MANIFEST)
        child.add_argument("--d-manifest", type=Path, default=DEFAULT_D_MANIFEST)
        if command == "submit":
            child.add_argument("--dry-run", action="store_true")
            child.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        c_path = _abs(args.c_manifest)
        d_path = _abs(args.d_manifest)
        if args.command == "submit":
            result = submit_heavy_first(
                c_manifest_path=c_path,
                d_manifest_path=d_path,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
            )
        else:
            c_manifest = _load_object(c_path)
            d_manifest = _load_object(d_path)
            result = (
                validate_combined(c_manifest, d_manifest)
                if args.command == "validate"
                else preflight(
                    c_manifest,
                    d_manifest,
                    require_capacity=True,
                    require_slurm=True,
                )
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except WorkflowError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COHORTS",
    "K_ORDER",
    "preflight",
    "submission_batches",
    "submit_heavy_first",
    "validate_combined",
]
