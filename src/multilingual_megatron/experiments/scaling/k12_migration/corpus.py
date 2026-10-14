"""Build and adopt an immutable K16-to-K12 A/B-v2 corpus delta.

The completed K16 release is a predecessor, never a workspace.  This module
builds the successor plan from the predecessor's pinned calibration, source,
and tokenizer, freezes the exact language delta, and then performs two
storage-neutral joins:

* predecessor plus incrementally acquired production-bootstrap languages; and
* predecessor finalized languages plus a newly finalized physical-pool delta.

Every input is hash-verified.  Predecessor inputs are always copied and sealed
read-only so their inode metadata is never changed by the successor.  Files
already inside the successor may be hard-linked between worker and publication
trees.  Each mutating command emits a canonical, hash-addressed receipt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from multilingual_megatron.experiments.scaling.k12_migration.constants import (
    EXPECTED_ACQUISITION,
    EXPECTED_INCREASED,
    LANGUAGE_MAP_SCHEMA_VERSION,
    NEW_TIERS,
    OLD_TIERS,
    REMOVED_FROM_K16,
    SCHEMA_VERSION,
    SHARED_TIERS,
    TOOL_ID,
    UPSTREAM_ADOPTION_PATHS,
)
from multilingual_megatron.experiments.scaling.k12_migration.common import (
    K12MigrationError,
    _canonical_without,
    _expected_exposure_audit,
    _install_file,
    _install_or_verify,
    _load_bootstrap_manifest_only,
    _load_object,
    _load_release_manifests_only,
    _load_worker_receipt,
    _migration_roots,
    _physical_entry,
    _publish_verified_worker_file,
    _receipt,
    _reject_output_inside_inputs,
    _require_under,
    _safe_artifact,
    _subset_plan,
    _tier_exposure,
    _tier_languages,
    _validate_language_against_plan,
    _validate_transition_shape,
    _verify_bootstrap,
    _verify_bootstrap_entry,
    _verify_payload,
    _verify_plan,
    _verify_release,
    _write_frozen_json,
)
from multilingual_megatron.experiments.scaling.k12_migration.planning import (
    _load_migration,
    build_migration_contract,
    prepare_migration,
)
from multilingual_megatron.experiments.scaling.k12_migration.adoption import (
    _adopt_path,
    _adoption_source_files,
    _atomic_read_only_bytes,
    _render_rebased_adoption_json,
    _sealed_stat,
    _verify_completed_adoption,
    _verify_predecessor_seal_stats,
    adopt_upstream,
    verify_completed_adoption,
)
from multilingual_megatron.experiments.scaling.k12_migration.bootstrap import (
    _bootstrap_entry_source,
    bootstrap_merge_worker,
    finalize_bootstrap_merge,
    merge_bootstraps,
)
from multilingual_megatron.experiments.scaling.k12_migration.release import (
    _link_release_language,
    _load_parallel_release_context,
    _release_artifact_metadata,
    _source_prefix,
    _verify_cross_root_tokenizer_contract,
    _verify_delta_bootstrap_identity,
    _verify_release_language_artifacts,
    finalize_release_merge,
    merge_releases,
    release_merge_worker,
)


def _prepare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("prepare", help="build and freeze the K12 migration plan")
    parser.add_argument("--predecessor-plan", type=Path, required=True)
    parser.add_argument("--predecessor-release", type=Path, required=True)
    parser.add_argument("--adoption-manifest", type=Path, required=True)
    parser.add_argument("--language-map", action="append", default=[], help="K=path")
    parser.add_argument("--successor-plan", type=Path)
    parser.add_argument("--delta-plan", type=Path)
    parser.add_argument("--acquisition-language-map", type=Path)
    parser.add_argument("--migration-plan", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")


def _worker_language_index(value: int, parser: argparse.ArgumentParser) -> int:
    if value >= 0:
        return value
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        parser.error(
            "--language-index without a value requires SLURM_ARRAY_TASK_ID"
        )
    try:
        index = int(raw)
    except ValueError:
        parser.error("SLURM_ARRAY_TASK_ID must be an integer")
    if index < 0:
        parser.error("SLURM_ARRAY_TASK_ID must be non-negative")
    return index


def _add_parallel_merge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--language-index",
        type=int,
        nargs="?",
        const=-1,
        help="run one language worker; omit value inside a Slurm array",
    )
    mode.add_argument(
        "--merge-summary",
        action="store_true",
        help="verify every worker receipt and publish the final manifests",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _prepare_parser(subparsers)
    adopt = subparsers.add_parser(
        "adopt-upstream", help="adopt sealed upstream inputs into a new pipeline root"
    )
    adopt.add_argument("--predecessor-root", type=Path, required=True)
    adopt.add_argument("--successor-root", type=Path, required=True)
    adopt.add_argument("--predecessor-seal", type=Path, required=True)
    bootstrap = subparsers.add_parser(
        "bootstrap-merge", help="join predecessor and incremental bootstrap entries"
    )
    bootstrap.add_argument("--migration-plan", type=Path, required=True)
    bootstrap.add_argument("--predecessor-bootstrap", type=Path, required=True)
    bootstrap.add_argument("--incremental-bootstrap", type=Path, required=True)
    _add_parallel_merge_arguments(bootstrap)
    release = subparsers.add_parser(
        "release-merge", help="join predecessor release and finalized physical delta"
    )
    release.add_argument("--migration-plan", type=Path, required=True)
    release.add_argument("--predecessor-release", type=Path, required=True)
    release.add_argument("--delta-release", type=Path, required=True)
    release.add_argument("--bootstrap-manifest", type=Path, required=True)
    _add_parallel_merge_arguments(release)
    args = parser.parse_args()
    if args.command == "adopt-upstream":
        result = adopt_upstream(
            predecessor_root=args.predecessor_root,
            successor_root=args.successor_root,
            predecessor_seal_path=args.predecessor_seal,
        )
    elif args.command == "prepare":
        if not args.language_map:
            parser.error("prepare requires four --language-map K=path arguments")
        if not args.dry_run and not all(
            (
                args.successor_plan,
                args.delta_plan,
                args.acquisition_language_map,
                args.migration_plan,
                args.receipt,
            )
        ):
            parser.error(
                "non-dry-run prepare requires --successor-plan, --delta-plan, "
                "--acquisition-language-map, --migration-plan, and --receipt"
            )
        result = prepare_migration(
            predecessor_plan_path=args.predecessor_plan,
            predecessor_release_dir=args.predecessor_release,
            adoption_manifest_path=args.adoption_manifest,
            language_map_args=args.language_map,
            successor_plan_path=args.successor_plan,
            delta_plan_path=args.delta_plan,
            acquisition_language_map_path=args.acquisition_language_map,
            migration_plan_path=args.migration_plan,
            receipt_path=args.receipt,
            dry_run=args.dry_run,
        )
    elif args.command == "bootstrap-merge":
        if args.language_index is not None:
            if args.work_root is None:
                parser.error("bootstrap worker requires --work-root")
            result = bootstrap_merge_worker(
                migration_plan_path=args.migration_plan,
                predecessor_bootstrap_dir=args.predecessor_bootstrap,
                incremental_bootstrap_dir=args.incremental_bootstrap,
                work_root=args.work_root,
                language_index=_worker_language_index(
                    args.language_index, parser
                ),
            )
        elif args.merge_summary:
            if not all((args.work_root, args.output_dir, args.receipt)):
                parser.error(
                    "bootstrap summary requires --work-root, --output-dir, and --receipt"
                )
            result = finalize_bootstrap_merge(
                migration_plan_path=args.migration_plan,
                predecessor_bootstrap_dir=args.predecessor_bootstrap,
                incremental_bootstrap_dir=args.incremental_bootstrap,
                work_root=args.work_root,
                output_dir=args.output_dir,
                receipt_path=args.receipt,
            )
        else:
            if not all((args.output_dir, args.receipt)):
                parser.error("bootstrap scalar merge requires --output-dir and --receipt")
            result = merge_bootstraps(
                migration_plan_path=args.migration_plan,
                predecessor_bootstrap_dir=args.predecessor_bootstrap,
                incremental_bootstrap_dir=args.incremental_bootstrap,
                output_dir=args.output_dir,
                receipt_path=args.receipt,
                work_root=args.work_root,
            )
    else:
        if args.language_index is not None:
            if args.work_root is None:
                parser.error("release worker requires --work-root")
            result = release_merge_worker(
                migration_plan_path=args.migration_plan,
                predecessor_release_dir=args.predecessor_release,
                delta_release_dir=args.delta_release,
                bootstrap_manifest_path=args.bootstrap_manifest,
                work_root=args.work_root,
                language_index=_worker_language_index(
                    args.language_index, parser
                ),
            )
        elif args.merge_summary:
            if not all((args.work_root, args.output_dir, args.receipt)):
                parser.error(
                    "release summary requires --work-root, --output-dir, and --receipt"
                )
            result = finalize_release_merge(
                migration_plan_path=args.migration_plan,
                predecessor_release_dir=args.predecessor_release,
                delta_release_dir=args.delta_release,
                bootstrap_manifest_path=args.bootstrap_manifest,
                work_root=args.work_root,
                output_dir=args.output_dir,
                receipt_path=args.receipt,
            )
        else:
            if not all((args.output_dir, args.receipt)):
                parser.error("release scalar merge requires --output-dir and --receipt")
            result = merge_releases(
                migration_plan_path=args.migration_plan,
                predecessor_release_dir=args.predecessor_release,
                delta_release_dir=args.delta_release,
                bootstrap_manifest_path=args.bootstrap_manifest,
                output_dir=args.output_dir,
                receipt_path=args.receipt,
                work_root=args.work_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "K12MigrationError",
    "adopt_upstream",
    "bootstrap_merge_worker",
    "build_migration_contract",
    "finalize_bootstrap_merge",
    "finalize_release_merge",
    "merge_bootstraps",
    "merge_releases",
    "prepare_migration",
    "release_merge_worker",
    "verify_completed_adoption",
    "main",
]
