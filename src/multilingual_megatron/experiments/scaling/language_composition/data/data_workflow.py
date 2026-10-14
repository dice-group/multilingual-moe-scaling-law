#!/usr/bin/env python3
"""CPU-only preparation of the common corpus for both selected K12 arms."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.k12_migration import workflow as engine
from multilingual_megatron.experiments.scaling.ab_v2.artifact_data import (
    _run_data_audit,
    _run_megatron_data_smoke,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    Paths as ABPaths,
    _load_object as _load_inherited_object,
)
from multilingual_megatron.experiments.scaling.ab_v2.common_provenance import (
    _config_inheritance_chain,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    ARM_GROUPINGS,
    DESIGN_ID,
    REPO_ROOT,
    SELECTION_MANIFEST,
    CompositionStudyError,
    verify_selection_authority,
)
from multilingual_megatron.experiments.scaling.language_composition.data.data_artifacts import build_composition_manifests
from multilingual_megatron.experiments.scaling.language_composition.data.data_plan import EXPECTED_ACQUISITION, prepare_capacity_migration


DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "workflow_runtime_language_composition_data_v1.json"
)
DATA_AUDIT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/study_spec.json"
)
WORKFLOW_ID = "stage1_language_composition_data_v1"
WORKER_MODULE = (
    "multilingual_megatron.experiments.scaling."
    "language_composition.data.data_workflow"
)
EXPECTED_TRANSITION_COUNTS = {
    "changed_physical": 16,
    "increased_physical": 16,
    "decreased_physical": 0,
    "acquisition": 8,
    "delta_finalize": 16,
    "reused_physical": 184,
    "exposure_changed": 17,
}


def _jobs() -> list[dict[str, Any]]:
    jobs = deepcopy(engine._jobs())
    for row in jobs:
        if row["job_id"] == "acquisition_workers":
            row["array"] = "0-7%8"
            row["array_count"] = 8
        elif row["job_id"] == "finalizer_workers":
            row["array"] = "0-15%16"
            row["array_count"] = 16
    return jobs


def _fixed_additions(runtime_path: Path) -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    migration_package = (
        REPO_ROOT
        / "src/multilingual_megatron/experiments/scaling/k12_migration"
    ).resolve()
    audit_package = (
        REPO_ROOT
        / "src/multilingual_megatron/pipeline/stages/audit/fineweb_release"
    ).resolve()
    return (
        *_config_inheritance_chain(runtime_path),
        (REPO_ROOT / DATA_AUDIT_SPEC).resolve(),
        (REPO_ROOT / SELECTION_MANIFEST).resolve(),
        *((REPO_ROOT / path).resolve() for path in ARM_GROUPINGS.values()),
        *tuple(sorted(package.parent.rglob("*.py"))),
        *tuple(sorted(migration_package.glob("*.py"))),
        *tuple(sorted(audit_package.glob("*.py"))),
        (REPO_ROOT / "src/multilingual_megatron/pipeline/stages/audit/fineweb_v2_release.py").resolve(),
        (REPO_ROOT / "src/multilingual_megatron/experiments/scaling/ab_v2/artifact_data.py").resolve(),
        (REPO_ROOT / "src/multilingual_megatron/experiments/scaling/k12_migration/data_smoke.py").resolve(),
        (
            REPO_ROOT
            / "external/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets/indexed_dataset.py"
        ).resolve(),
    )


def build_manifest(
    *, runtime_path: Path, predecessor_root: Path, output_root: Path
) -> dict[str, Any]:
    runtime_path = runtime_path.resolve()
    predecessor_root = predecessor_root.resolve()
    output_root = output_root.resolve()
    base = engine.build_manifest(
        runtime_path=runtime_path,
        predecessor_root=predecessor_root,
        output_root=output_root,
    )
    jobs = _jobs()
    selection = verify_selection_authority()
    fixed = dict(base["fixed_inputs"])
    for path in _fixed_additions(runtime_path):
        if path.is_symlink() or not path.is_file():
            raise CompositionStudyError(f"fixed workflow input is missing: {path}")
        fixed[str(path)] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    estimate = dict(base["storage_estimate"])
    estimate["method"] = (
        "logical_predecessor_copy_bytes_plus_35_percent_composition_capacity_margin"
    )
    # The predecessor currently uses only a few thousand corpus entries.  A
    # deliberately much larger bound keeps the already-constrained shared
    # filesystem from failing late on inode exhaustion.
    estimate.update(
        {
            "migration_peak_additional_inodes": 100_000,
            "required_final_free_inodes": 100_000,
            "required_available_inodes_at_preflight": 200_000,
            "inode_policy": "fail before materialization with a 100k final inode reserve",
        }
    )
    manifest = dict(base)
    manifest.update(
        {
            "workflow_id": WORKFLOW_ID,
            "worker_module": WORKER_MODULE,
            "jobs": jobs,
            "submit_slots": sum(int(row["array_count"]) for row in jobs),
            "fixed_inputs": dict(sorted(fixed.items())),
            "storage_estimate": estimate,
            "selection_authority": selection,
            "scientific_contract": {
                "design_id": DESIGN_ID,
                "release_policy": "one common 32B physical release for both K12 arms",
                "canonical_nested_tiers_unchanged": [1, 12, 50, 200],
                "physical_capacity_rule": "elementwise maximum across reference, compact, and diverse exposures",
                "validation_frame": "K196 with primary unseen_177 cohort",
                "gpu_jobs": 0,
            },
            "contract": {
                **dict(base["contract"]),
                "acquisition_languages": sorted(EXPECTED_ACQUISITION),
                "finalizer_workers": 16,
                "changed_physical_languages": 16,
                "reused_physical_languages": 184,
                "exposure_changed_languages": 17,
                "selection_identity_sha256": selection[
                    "selection_identity_sha256"
                ],
                "common_release_for_arms": ["compact", "diverse"],
            },
        }
    )
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    validate_manifest(manifest, verify_fixed_inputs=True)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], *, verify_fixed_inputs: bool = True
) -> None:
    engine._verify_payload_hash(
        manifest, "manifest_sha256", label="composition data launch manifest"
    )
    if (
        manifest.get("schema_version") != engine.SCHEMA_VERSION
        or manifest.get("workflow_id") != WORKFLOW_ID
        or manifest.get("worker_module") != WORKER_MODULE
        or Path(str(manifest.get("repo_root", ""))).resolve() != REPO_ROOT
    ):
        raise CompositionStudyError("unsupported composition data manifest")
    paths = engine._paths_from_manifest(manifest)
    if paths.predecessor == paths.successor:
        raise CompositionStudyError("composition successor aliases its predecessor")
    contract = manifest.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("language_counts") != [1, 12, 50, 200]
        or contract.get("cpu_only") is not True
        or contract.get("acquisition_languages")
        != sorted(EXPECTED_ACQUISITION)
        or contract.get("finalizer_workers") != 16
        or contract.get("changed_physical_languages") != 16
        or contract.get("reused_physical_languages") != 184
        or contract.get("exposure_changed_languages") != 17
    ):
        raise CompositionStudyError("composition data transition contract drifted")
    resources = engine._validate_cpu_resources(manifest.get("resource_profiles"))
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("data"), dict):
        raise CompositionStudyError("composition data manifest lacks runtime data")
    engine._validate_tokenization_resource_contract(runtime["data"], resources)
    if manifest.get("engine_provenance", {}).get("megatron_bridge") != engine._bridge_revision():
        raise CompositionStudyError("Megatron-Bridge revision drifted")
    jobs = manifest.get("jobs")
    if jobs != _jobs():
        raise CompositionStudyError("composition data DAG drifted")
    seen: set[str] = set()
    for row in jobs:
        job_id = str(row["job_id"])
        if job_id in seen or any(str(value) not in seen for value in row["dependencies"]):
            raise CompositionStudyError("composition data DAG is not topological")
        if row["resource"] not in resources:
            raise CompositionStudyError(f"unknown CPU resource for {job_id}")
        seen.add(job_id)
    if int(manifest.get("submit_slots", -1)) != sum(
        int(row["array_count"]) for row in jobs
    ):
        raise CompositionStudyError("composition submit-slot count drifted")
    estimate = manifest.get("storage_estimate")
    if (
        not isinstance(estimate, dict)
        or int(estimate.get("required_final_free_headroom_bytes", 0)) < 3 * 1024**4
        or int(estimate.get("required_available_bytes_at_preflight", 0))
        != int(estimate.get("migration_peak_additional_bytes", -1))
        + int(estimate.get("required_final_free_headroom_bytes", -1))
        or int(estimate.get("required_available_inodes_at_preflight", 0))
        != int(estimate.get("migration_peak_additional_inodes", -1))
        + int(estimate.get("required_final_free_inodes", -1))
    ):
        raise CompositionStudyError("composition storage preflight contract drifted")
    selection = verify_selection_authority()
    if manifest.get("selection_authority") != selection:
        raise CompositionStudyError("composition selection authority drifted")
    if manifest.get("scientific_contract") != {
        "design_id": DESIGN_ID,
        "release_policy": "one common 32B physical release for both K12 arms",
        "canonical_nested_tiers_unchanged": [1, 12, 50, 200],
        "physical_capacity_rule": (
            "elementwise maximum across reference, compact, and diverse exposures"
        ),
        "validation_frame": "K196 with primary unseen_177 cohort",
        "gpu_jobs": 0,
    } or {
        "selection_identity_sha256": contract.get("selection_identity_sha256"),
        "common_release_for_arms": contract.get("common_release_for_arms"),
    } != {
        "selection_identity_sha256": selection["selection_identity_sha256"],
        "common_release_for_arms": ["compact", "diverse"],
    }:
        raise CompositionStudyError("composition scientific contract drifted")
    if verify_fixed_inputs:
        fixed = manifest.get("fixed_inputs")
        if not isinstance(fixed, dict) or not fixed:
            raise CompositionStudyError("composition data manifest lacks fixed inputs")
        for raw_path, identity in fixed.items():
            path = Path(str(raw_path))
            if (
                path.is_symlink()
                or not path.is_file()
                or not isinstance(identity, dict)
                or path.stat().st_size != int(identity.get("size_bytes", -1))
                or sha256_file(path) != identity.get("sha256")
            ):
                raise CompositionStudyError(f"fixed input changed: {path}")


def write_manifest(manifest: Mapping[str, Any]) -> Path:
    validate_manifest(manifest, verify_fixed_inputs=True)
    path = (
        Path(str(manifest["output_root"]))
        / "control/k12_migration/launch_manifest.json"
    )
    engine._write_frozen_json(path, manifest)
    return path


def _run_migration_prepare(
    manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None
) -> None:
    if task is not None:
        raise CompositionStudyError("migration preparation is not an array job")
    paths = engine._paths_from_manifest(manifest)
    prepare_capacity_migration(
        predecessor_plan_path=paths.predecessor_plan,
        predecessor_release_dir=paths.predecessor_release,
        adoption_manifest_path=paths.adoption_manifest,
        successor_plan_path=paths.successor_plan,
        delta_plan_path=paths.delta_plan,
        acquisition_language_map_path=paths.acquisition_map,
        migration_plan_path=paths.migration_plan,
        receipt_path=engine._completion_path(manifest, row, task),
    )


def _run_acquisition_worker(manifest: Mapping[str, Any], task: int) -> None:
    paths = engine._paths_from_manifest(manifest)
    languages = tuple(manifest["contract"]["acquisition_languages"])
    if task < 0 or task >= len(languages):
        raise CompositionStudyError("acquisition task index is out of range")
    language_root = paths.incremental_bootstrap / languages[task]
    partials = [
        language_root / name
        for name in ("train.jsonl", "validation_reserve.jsonl", "calibration.jsonl")
    ]
    partials.extend(
        path
        for pattern in (
            ".train.jsonl.*",
            ".validation_reserve.jsonl.*",
            ".calibration.jsonl.*",
        )
        for path in language_root.glob(pattern)
    )
    engine._quarantine_owned_partials(
        manifest, label=f"acquisition_{languages[task]}", candidates=partials
    )
    engine._run_command([*engine._bootstrap_args(manifest), "--language-index", str(task)])


def _run_completion_gate(
    manifest: Mapping[str, Any], row: Mapping[str, Any]
) -> None:
    paths = engine._paths_from_manifest(manifest)
    migration = engine._load_object(paths.migration_plan, label="composition migration plan")
    engine._verify_payload_hash(
        migration, "migration_plan_sha256", label="composition migration plan"
    )
    transition = migration.get("transition")
    if not isinstance(transition, dict) or transition.get("counts") != EXPECTED_TRANSITION_COUNTS:
        raise CompositionStudyError("materialized transition differs from the frozen delta")
    release = engine._load_object(paths.release / "corpus_manifest.json", label="composition release")
    engine._verify_payload_hash(release, "manifest_sha256", label="composition release")
    if release.get("ready") is not True or len(release.get("languages", {})) != 200:
        raise CompositionStudyError("composition release is not ready with 200 languages")
    report = engine._load_object(
        paths.tokenized / "tokenization_report.json", label="composition tokenization"
    )
    engine._verify_payload_hash(
        report, "report_payload_sha256", label="composition tokenization"
    )
    incremental = report.get("incremental_tokenization")
    reused = incremental.get("reused_items_by_split") if isinstance(incremental, dict) else None
    rebuilt = incremental.get("rebuilt_items_by_split") if isinstance(incremental, dict) else None
    if (
        not isinstance(reused, dict)
        or not isinstance(rebuilt, dict)
        or len(reused.get("train", [])) != 184
        or len(rebuilt.get("train", [])) != 16
    ):
        raise CompositionStudyError("incremental train-prefix reuse is not 184/16")
    composition_audit = build_composition_manifests(
        tokenized_root=paths.tokenized,
        sampling_plan_path=paths.successor_plan,
    )
    source_launch = paths.successor / "control/launch_manifest.json"
    engine._write_frozen_json(source_launch, manifest)
    ab_paths = ABPaths(paths.successor)
    _run_data_audit(manifest, ab_paths, study_spec_path=DATA_AUDIT_SPEC)
    _run_megatron_data_smoke(
        manifest,
        ab_paths,
        {"bridge_python_executable": manifest["runtime"]["python_executable"]},
    )
    result: dict[str, Any] = {
        "schema_version": engine.SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "migration_plan_sha256": migration["migration_plan_sha256"],
        "release_manifest_sha256": release["manifest_sha256"],
        "tokenization_report_sha256": report["report_payload_sha256"],
        "transition_counts": transition["counts"],
        "incremental_tokenization": {
            "reused_train": len(reused["train"]),
            "rebuilt_train": len(rebuilt["train"]),
            "reused_validation": len(reused.get("validation", [])),
            "rebuilt_validation": len(rebuilt.get("validation", [])),
        },
        "composition_manifest_audit_sha256": composition_audit["payload_sha256"],
        "data_audit": str((paths.successor / "control/data_audit.json").resolve()),
        "data_seal": str((paths.successor / "control/data_artifact_seal.json").resolve()),
        "reader_smoke": str((paths.successor / "control/megatron_data_smoke.json").resolve()),
        "next_step": "four authorized compact/diverse E1/E32 H100 smokes",
    }
    result["payload_sha256"] = canonical_json_sha256(result)
    engine._write_frozen_json(engine._completion_path(manifest, row, None), result)


def execute_job(
    manifest_path: Path, job_id: str, *, task: int | None = None
) -> None:
    manifest = engine._load_object(
        manifest_path.resolve(), label="composition data launch manifest"
    )
    validate_manifest(manifest, verify_fixed_inputs=True)
    row = engine._job_by_id(manifest, job_id)
    task_id = engine._task_index(row, str(task) if task is not None else None)
    receipt_path = engine._receipt_path(manifest, row, task_id)
    if receipt_path.is_file():
        engine._verify_receipt(manifest, row, task_id)
        return
    engine._require_dependencies(manifest, row)
    completion = engine._completion_path(manifest, row, task_id)
    if completion.is_file():
        engine._verify_completion_artifact(manifest, row, task_id)
        engine._write_receipt(manifest, row, task_id, recovered=True)
        return
    action = str(row["action"])
    if action == "migration_prepare":
        _run_migration_prepare(manifest, row, task_id)
    elif action == "acquisition_worker":
        if task_id is None:
            raise CompositionStudyError("acquisition worker lacks an array task")
        _run_acquisition_worker(manifest, task_id)
    elif action == "completion_gate":
        _run_completion_gate(manifest, row)
    else:
        engine._run_action(manifest, row, task_id)
    engine._write_receipt(manifest, row, task_id, recovered=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    plan.add_argument("--predecessor-root", type=Path)
    plan.add_argument("--output-root", type=Path)
    submit = sub.add_parser("submit")
    submit.add_argument("--manifest", type=Path, required=True)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--retry-failed", action="store_true")
    execute = sub.add_parser("execute", help=argparse.SUPPRESS)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--job-id", required=True)
    execute.add_argument("--task", type=int)
    status = sub.add_parser("status")
    status.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            runtime_path = (
                args.runtime.resolve()
                if args.runtime.is_absolute()
                else (REPO_ROOT / args.runtime).resolve()
            )
            runtime = _load_inherited_object(runtime_path)
            predecessor = (
                args.predecessor_root.resolve()
                if args.predecessor_root
                else Path(runtime["data"]["adoption_predecessor_root"]).resolve()
            )
            output = (
                args.output_root.resolve()
                if args.output_root
                else (REPO_ROOT / str(runtime["output_root"])).resolve()
            )
            manifest = build_manifest(
                runtime_path=runtime_path,
                predecessor_root=predecessor,
                output_root=output,
            )
            path = write_manifest(manifest)
            print(json.dumps({"manifest": str(path), "submit_slots": manifest["submit_slots"]}, indent=2))
            return 0
        manifest_path = args.manifest.resolve()
        if args.command == "submit":
            state = engine.submit_workflow(
                manifest_path,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
                validate_manifest_fn=validate_manifest,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0
        if args.command == "execute":
            execute_job(manifest_path, args.job_id, task=args.task)
            return 0
        if args.command == "status":
            report = engine.status(
                manifest_path, validate_manifest_fn=validate_manifest
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["complete"] else 1
    except (CompositionStudyError, engine.K12MigrationWorkflowError, OSError, ValueError) as exc:
        print(f"[language-composition-data] {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
