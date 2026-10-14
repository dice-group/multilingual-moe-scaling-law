#!/usr/bin/env python3
"""Plan and submit the CPU-only A/B-v2 K16-to-K12 corpus migration.

This experiment-layer workflow is deliberately separate from the A/B-v2 GPU
workflow.  It materializes the K12 successor corpus under a new run root using
the immutable migration contract, then incrementally tokenizes it by reusing
only hash-identical predecessor prefixes.  Every Slurm allocation executes a
local command; no allocated job submits another Slurm job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from multilingual_megatron.experiments.scaling.k12_migration import corpus as migration
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    _load_object as _load_inherited_object,
)
from multilingual_megatron.pipeline.executor import SlurmExecutor
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__)).resolve()
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime.json"
)
DEFAULT_PREDECESSOR = Path("outputs/pipeline_runs/h100_ab_v2_attempt7")
DEFAULT_OUTPUT = Path("outputs/pipeline_runs/h100_ab_v2_k12_fixed_lr_attempt3")
WORKFLOW_ID = "multilingual_moe_ab_v2_k12_migration"
SCHEMA_VERSION = 1
MANIFEST_NAME = "launch_manifest.json"
STATE_NAME = "submission_state.json"
MODULE = "multilingual_megatron.experiments.scaling.k12_migration.workflow"
MIGRATION_MODULE = (
    "multilingual_megatron.experiments.scaling.k12_migration.corpus"
)
BOOTSTRAP_MODULE = (
    "multilingual_megatron.pipeline.stages.sampling.tools.bootstrap_fineweb_v2"
)
FINALIZER_MODULE = (
    "multilingual_megatron.pipeline.stages.sampling.tools.parallel_finalize_fineweb_v2"
)
TOKENIZATION_MODULE = (
    "multilingual_megatron.pipeline.stages.tokenization.fineweb_v2_presplit"
)
LANGUAGE_COUNTS = (1, 12, 50, 200)
ADOPTION_COUNT = len(migration.UPSTREAM_ADOPTION_PATHS)
ACQUISITION_LANGUAGES = tuple(sorted(migration.EXPECTED_ACQUISITION))
FINALIZER_WORKERS = 15
TOKENIZATION_RANKS = 200
COPY_ARRAY_TASKS = 32
TOKENIZATION_PLAN_VERIFICATION_WORKERS = 32
MINIMUM_POST_MIGRATION_HEADROOM_BYTES = 3 * 1024**4
SLURM_FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "DEPENDENCY",
}
SLURM_ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "REQUEUED",
    "REQUEUE_FED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}


class K12MigrationWorkflowError(RuntimeError):
    """Raised when the migration DAG or its evidence is not trustworthy."""


def _abs(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(payload)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise K12MigrationWorkflowError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise K12MigrationWorkflowError(f"{label} must be a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise K12MigrationWorkflowError(f"frozen target is not a regular file: {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise K12MigrationWorkflowError(f"refusing to replace a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _verify_payload_hash(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    expected = payload.get(key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise K12MigrationWorkflowError(f"{label} lacks {key}")
    unhashed = dict(payload)
    unhashed.pop(key, None)
    if _canonical_hash(unhashed) != expected:
        raise K12MigrationWorkflowError(f"{label} payload hash mismatch")
    return expected


def _resolve_artifact(base: Path, raw: Any, *, label: str) -> Path:
    candidate = Path(str(raw or ""))
    if not str(candidate):
        raise K12MigrationWorkflowError(f"{label} path is empty")
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


@dataclass(frozen=True, slots=True)
class Paths:
    predecessor: Path
    successor: Path
    predecessor_release: Path
    predecessor_plan: Path
    predecessor_bootstrap: Path

    @property
    def control(self) -> Path:
        return self.successor / "control" / "k12_migration"

    @property
    def corpus(self) -> Path:
        return self.successor / "corpus"

    @property
    def migration(self) -> Path:
        return self.corpus / "k12_migration"

    @property
    def adoption_manifest(self) -> Path:
        return self.migration / "adoption_manifest.json"

    @property
    def successor_plan(self) -> Path:
        return self.corpus / "sampling_plan_32b.json"

    @property
    def delta_plan(self) -> Path:
        return self.migration / "delta_sampling_plan_32b.json"

    @property
    def acquisition_map(self) -> Path:
        return self.migration / "acquisition_language_map.json"

    @property
    def migration_plan(self) -> Path:
        return self.migration / "migration_plan.json"

    @property
    def incremental_bootstrap(self) -> Path:
        return self.migration / "incremental_bootstrap"

    @property
    def production_bootstrap(self) -> Path:
        return self.corpus / "production_bootstrap"

    @property
    def finalizer_work(self) -> Path:
        return self.migration / "parallel_finalize"

    @property
    def delta_release(self) -> Path:
        return self.migration / "delta_release_32b"

    @property
    def release(self) -> Path:
        return self.corpus / "release_32b"

    @property
    def tokenization_work(self) -> Path:
        return self.corpus / "tokenization_work"

    @property
    def tokenized(self) -> Path:
        return self.corpus / "tokenized"

    @property
    def tokenizer_model(self) -> Path:
        return self.corpus / "tokenizer" / "sp_unigram_64k" / "tokenizer.model"


def _paths_from_manifest(manifest: Mapping[str, Any]) -> Paths:
    raw = manifest.get("paths")
    if not isinstance(raw, dict):
        raise K12MigrationWorkflowError("migration manifest lacks paths")
    return Paths(
        predecessor=Path(str(manifest["predecessor_root"])).resolve(),
        successor=Path(str(manifest["output_root"])).resolve(),
        predecessor_release=Path(str(raw["predecessor_release"])).resolve(),
        predecessor_plan=Path(str(raw["predecessor_plan"])).resolve(),
        predecessor_bootstrap=Path(str(raw["predecessor_bootstrap"])).resolve(),
    )


def _language_map(k: int) -> Path:
    return _abs(f"configs/language_groupings/scaling_fw2_v2_k{k:03d}.json")


def _job(
    job_id: str,
    action: str,
    resource: str,
    *,
    dependencies: Sequence[str] = (),
    array: str | None = None,
    array_count: int = 1,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "action": action,
        "resource": resource,
        "dependencies": list(dependencies),
        "array": array,
        "array_count": array_count,
    }


def _jobs() -> list[dict[str, Any]]:
    """Return the frozen CPU-only dependency chain."""

    return [
        _job("storage_preflight", "storage_preflight", "cpu"),
        _job(
            "predecessor_seal_preflight",
            "predecessor_seal_verify",
            "cpu",
            dependencies=("storage_preflight",),
        ),
        _job(
            "adopt_copy",
            "adopt_copy",
            "cpu_streaming",
            dependencies=("predecessor_seal_preflight",),
            array=f"0-{ADOPTION_COUNT - 1}%{ADOPTION_COUNT}",
            array_count=ADOPTION_COUNT,
        ),
        _job("adopt_finalize", "adopt_finalize", "cpu", dependencies=("adopt_copy",)),
        _job(
            "migration_prepare",
            "migration_prepare",
            "cpu_streaming",
            dependencies=("adopt_finalize",),
        ),
        _job(
            "acquisition_preflight",
            "acquisition_preflight",
            "cpu",
            dependencies=("migration_prepare",),
        ),
        _job(
            "acquisition_workers",
            "acquisition_worker",
            "cpu_streaming",
            dependencies=("acquisition_preflight",),
            array=f"0-{len(ACQUISITION_LANGUAGES) - 1}%{len(ACQUISITION_LANGUAGES)}",
            array_count=len(ACQUISITION_LANGUAGES),
        ),
        _job(
            "acquisition_merge",
            "acquisition_merge",
            "cpu_streaming",
            dependencies=("acquisition_workers",),
        ),
        _job(
            "bootstrap_copy_workers",
            "bootstrap_copy_worker",
            "cpu_streaming",
            dependencies=("acquisition_merge",),
            array=f"0-{COPY_ARRAY_TASKS - 1}%{COPY_ARRAY_TASKS}",
            array_count=COPY_ARRAY_TASKS,
        ),
        _job(
            "bootstrap_join",
            "bootstrap_join",
            "cpu",
            dependencies=("bootstrap_copy_workers",),
        ),
        _job(
            "finalizer_prepare",
            "finalizer_prepare",
            "cpu",
            dependencies=("bootstrap_join",),
        ),
        _job(
            "finalizer_workers",
            "finalizer_worker",
            "cpu_streaming",
            dependencies=("finalizer_prepare",),
            array=f"0-{FINALIZER_WORKERS - 1}%{FINALIZER_WORKERS}",
            array_count=FINALIZER_WORKERS,
        ),
        _job(
            "finalizer_merge",
            "finalizer_merge",
            "cpu_streaming",
            dependencies=("finalizer_workers",),
        ),
        _job(
            "release_copy_workers",
            "release_copy_worker",
            "cpu_streaming",
            dependencies=("finalizer_merge",),
            array=f"0-{COPY_ARRAY_TASKS - 1}%{COPY_ARRAY_TASKS}",
            array_count=COPY_ARRAY_TASKS,
        ),
        _job(
            "release_join",
            "release_join",
            "cpu",
            dependencies=("release_copy_workers",),
        ),
        _job(
            "tokenize_plan",
            "tokenize_plan",
            "cpu_largemem",
            dependencies=("release_join",),
        ),
        _job(
            "tokenize_workers",
            "tokenize_worker",
            "cpu_streaming",
            dependencies=("tokenize_plan",),
            array="0-199%32",
            array_count=TOKENIZATION_RANKS,
        ),
        _job(
            "tokenize_merge",
            "tokenize_merge",
            "cpu_streaming",
            dependencies=("tokenize_workers",),
        ),
        _job(
            "predecessor_seal_postflight",
            "predecessor_seal_verify",
            "cpu",
            dependencies=("tokenize_merge",),
        ),
        _job(
            "migration_complete",
            "completion_gate",
            "cpu",
            dependencies=("predecessor_seal_postflight",),
        ),
    ]


def _fixed_input_paths(runtime_path: Path, predecessor: Path, paths: Paths) -> list[Path]:
    return [
        runtime_path,
        predecessor / "control" / "data_artifact_seal.json",
        predecessor / "control" / "data_audit.json",
        predecessor / "control" / "launch_manifest.json",
        paths.predecessor_plan,
        paths.predecessor_release / "corpus_manifest.json",
        paths.predecessor_release / "hash_manifest.json",
        paths.predecessor_bootstrap / "bootstrap_manifest.json",
        predecessor / "corpus" / "tokenized" / "megatron_data_manifest.json",
        predecessor / "corpus" / "tokenized" / "validation_megatron_data_manifest.json",
        predecessor / "corpus" / "tokenized" / "tokenization_report.json",
        _abs("configs/data_sources/fineweb2_v2_1_1_source_lock.json"),
        *(_language_map(k) for k in LANGUAGE_COUNTS),
        *sorted(Path(__file__).resolve().parent.glob("*.py")),
        _abs("src/multilingual_megatron/experiments/scaling/ab_v2/workflow.py"),
        _abs("src/multilingual_megatron/pipeline/stages/sampling/tools/bootstrap_fineweb_v2.py"),
        _abs("src/multilingual_megatron/pipeline/stages/sampling/fineweb_v2_policy.py"),
        _abs("src/multilingual_megatron/pipeline/stages/sampling/tools/build_fineweb_v2_sampling_plan.py"),
        _abs("src/multilingual_megatron/pipeline/stages/sampling/tools/parallel_finalize_fineweb_v2.py"),
        _abs("src/multilingual_megatron/pipeline/stages/sampling/tools/finalize_fineweb_v2_corpus.py"),
        _abs("src/multilingual_megatron/pipeline/stages/tokenization/fineweb_v2_presplit.py"),
        _abs("src/multilingual_megatron/pipeline/stages/tokenization/tools/run_megatron_preprocess.py"),
        _abs("external/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py"),
    ]


def _tree_logical_bytes(root: Path) -> int:
    if root.is_symlink():
        raise K12MigrationWorkflowError(f"storage-estimate input is a symlink: {root}")
    if root.is_file():
        return root.stat().st_size
    if not root.is_dir():
        raise K12MigrationWorkflowError(f"storage-estimate input is missing: {root}")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise K12MigrationWorkflowError(f"storage-estimate tree contains a symlink: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _walk_prefix_artifacts(value: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, dict):
        if {
            "prefix",
            "bin_size_bytes",
            "idx_size_bytes",
            "bin_sha256",
            "idx_sha256",
        }.issubset(value):
            rows.append(value)
        for child in value.values():
            rows.extend(_walk_prefix_artifacts(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_prefix_artifacts(child))
    return rows


def _predecessor_tokenized_bytes(predecessor: Path) -> int:
    tokenized = predecessor / "corpus" / "tokenized"
    unique: dict[str, int] = {}
    for name in ("megatron_data_manifest.json", "validation_megatron_data_manifest.json"):
        payload = _load_object(tokenized / name, label=f"predecessor {name}")
        for raw in _walk_prefix_artifacts(payload):
            prefix = str(raw["prefix"])
            size = int(raw["bin_size_bytes"]) + int(raw["idx_size_bytes"])
            if prefix in unique and unique[prefix] != size:
                raise K12MigrationWorkflowError(f"conflicting token-prefix size: {prefix}")
            unique[prefix] = size
    if not unique:
        raise K12MigrationWorkflowError("predecessor tokenized manifests have no prefix artifacts")
    return sum(unique.values())


def _storage_estimate(
    *, runtime: Mapping[str, Any], predecessor: Path, paths: Paths
) -> dict[str, Any]:
    adoption_bytes = sum(
        _tree_logical_bytes(predecessor / relative)
        for relative in migration.UPSTREAM_ADOPTION_PATHS
    )
    bootstrap_bytes = _tree_logical_bytes(paths.predecessor_bootstrap)
    release_bytes = _tree_logical_bytes(paths.predecessor_release)
    tokenized_bytes = _predecessor_tokenized_bytes(predecessor)
    predecessor_equivalent = adoption_bytes + bootstrap_bytes + release_bytes + tokenized_bytes
    # K12 enlarges eleven physical pools and also retains delta work products.
    # A 35% margin covers those additions without pretending to know final byte
    # counts before deterministic sampling/tokenization has run.
    growth_and_work_margin = int(math.ceil(predecessor_equivalent * 0.35))
    migration_peak = predecessor_equivalent + growth_and_work_margin
    storage = runtime.get("storage")
    if not isinstance(storage, dict):
        raise K12MigrationWorkflowError("runtime lacks storage policy")
    final_headroom = int(storage.get("post_allocation_free_headroom_bytes", 0))
    if final_headroom < MINIMUM_POST_MIGRATION_HEADROOM_BYTES:
        raise K12MigrationWorkflowError("migration requires at least 3 TiB final free headroom")
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "logical_predecessor_copy_bytes_plus_35_percent_k12_growth_margin",
        "components": {
            "upstream_adoption_bytes": adoption_bytes,
            "production_bootstrap_bytes": bootstrap_bytes,
            "finalized_release_bytes": release_bytes,
            "tokenized_prefix_bytes": tokenized_bytes,
            "growth_and_intermediate_work_margin_bytes": growth_and_work_margin,
        },
        "migration_peak_additional_bytes": migration_peak,
        "required_final_free_headroom_bytes": final_headroom,
        "required_available_bytes_at_preflight": migration_peak + final_headroom,
        "preflight_policy": "fail_before_any_successor_materialization",
    }


def _validate_cpu_resources(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise K12MigrationWorkflowError("runtime resources must be an object")
    selected: dict[str, Any] = {}
    for name in ("cpu", "cpu_largemem", "cpu_streaming"):
        profile = raw.get(name)
        if not isinstance(profile, dict):
            raise K12MigrationWorkflowError(f"runtime lacks resource profile {name}")
        partition = str(profile.get("partition", "")).lower()
        gres = str(profile.get("gres", "none")).lower()
        flags = [str(value) for value in profile.get("extra_sbatch_flags", [])]
        if "gpu" in partition or gres not in {"", "none", "cpu"}:
            raise K12MigrationWorkflowError(f"K12 migration resource {name} is not CPU-only")
        if any("gpu" in flag.lower() or "--gres" in flag.lower() for flag in flags):
            raise K12MigrationWorkflowError(f"K12 migration resource {name} requests a GPU")
        cpus_per_task = int(profile.get("cpus_per_task", 0))
        if cpus_per_task <= 0:
            raise K12MigrationWorkflowError(
                f"K12 migration resource {name} must provide at least one CPU"
            )
        selected[name] = dict(profile)
    if (
        int(selected["cpu_largemem"]["cpus_per_task"])
        < TOKENIZATION_PLAN_VERIFICATION_WORKERS
    ):
        raise K12MigrationWorkflowError(
            "cpu_largemem must provide at least 32 CPUs for tokenization-plan verification"
        )
    return selected


def _validate_tokenization_resource_contract(
    data: Mapping[str, Any], resources: Mapping[str, Any]
) -> None:
    workers_per_rank = int(data.get("tokenization_workers_per_rank", 0))
    streaming_cpus = int(resources["cpu_streaming"].get("cpus_per_task", 0))
    if workers_per_rank <= 0 or workers_per_rank > streaming_cpus:
        raise K12MigrationWorkflowError(
            "tokenization workers per rank must be positive and may not exceed "
            "cpu_streaming CPUs per task"
        )


def _bridge_revision() -> dict[str, Any]:
    bridge_root = _abs("external/Megatron-Bridge")
    commit = subprocess.run(
        ["git", "-C", str(bridge_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(bridge_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise K12MigrationWorkflowError(
            "Megatron-Bridge must be clean before freezing or running the K12 migration"
        )
    return {"root": str(bridge_root), "commit": commit, "clean": True}


def build_manifest(
    *, runtime_path: Path, predecessor_root: Path, output_root: Path
) -> dict[str, Any]:
    runtime_path = runtime_path.resolve()
    predecessor_root = predecessor_root.resolve()
    output_root = output_root.resolve()
    if predecessor_root == output_root:
        raise K12MigrationWorkflowError("successor and predecessor roots must differ")
    try:
        output_root.relative_to(predecessor_root)
    except ValueError:
        pass
    else:
        raise K12MigrationWorkflowError("successor root may not be inside predecessor root")
    # Runtime overlays use the same hash-pinned inheritance contract as the
    # training workflows.  Full historical runtime files resolve identically.
    runtime = _load_inherited_object(runtime_path)
    data = runtime.get("data")
    if not isinstance(data, dict) or data.get("mode") != "adopt_k12_migration":
        raise K12MigrationWorkflowError("runtime data.mode must be adopt_k12_migration")
    if _abs(str(runtime.get("output_root", ""))) != output_root:
        raise K12MigrationWorkflowError("runtime output_root differs from migration successor")
    if _abs(str(data.get("adoption_predecessor_root", ""))) != predecessor_root:
        raise K12MigrationWorkflowError("runtime adoption predecessor differs from migration input")
    if int(data.get("tokenization_array_ranks", -1)) != TOKENIZATION_RANKS:
        raise K12MigrationWorkflowError("K12 incremental tokenization requires 200 ranks")
    if str(data.get("tokenization_array")) != "0-199%32":
        raise K12MigrationWorkflowError("K12 tokenization array must be exactly 0-199%32")
    if (
        int(data.get("tokenization_plan_verification_workers", -1))
        != TOKENIZATION_PLAN_VERIFICATION_WORKERS
    ):
        raise K12MigrationWorkflowError(
            "K12 tokenization-plan verification must use exactly 32 workers"
        )
    resources = _validate_cpu_resources(runtime.get("resources"))
    _validate_tokenization_resource_contract(data, resources)
    submit_limit = int(runtime.get("max_submit_slots_per_invocation", 0))
    planned_slots = sum(int(row["array_count"]) for row in _jobs())
    if submit_limit <= 0 or planned_slots > submit_limit:
        raise K12MigrationWorkflowError(
            f"K12 migration needs {planned_slots} Slurm slots but runtime permits {submit_limit}"
        )

    seal_path = predecessor_root / "control" / "data_artifact_seal.json"
    seal = _load_object(seal_path, label="predecessor seal")
    seal_hash = _verify_payload_hash(seal, "seal_payload_sha256", label="predecessor seal")
    if seal.get("status") != "sealed" or Path(str(seal.get("release_root", ""))).resolve() != predecessor_root:
        raise K12MigrationWorkflowError("predecessor seal does not bind the predecessor root")

    predecessor_release = predecessor_root / "corpus" / "release_32b"
    release = _load_object(
        predecessor_release / "corpus_manifest.json", label="predecessor release"
    )
    if release.get("ready") is not True or release.get("release_status") != "ready":
        raise K12MigrationWorkflowError("predecessor release is not ready")
    if len(release.get("languages", {})) != 200:
        raise K12MigrationWorkflowError("predecessor release must contain 200 languages")
    inputs = release.get("inputs")
    if not isinstance(inputs, dict):
        raise K12MigrationWorkflowError("predecessor release lacks input lineage")
    predecessor_plan = _resolve_artifact(
        predecessor_release,
        (inputs.get("sampling_plan") or {}).get("path"),
        label="predecessor sampling plan",
    )
    predecessor_bootstrap_manifest = _resolve_artifact(
        predecessor_release,
        (inputs.get("bootstrap_manifest") or {}).get("path"),
        label="predecessor bootstrap manifest",
    )
    for candidate, label in (
        (predecessor_plan, "predecessor sampling plan"),
        (predecessor_bootstrap_manifest, "predecessor bootstrap manifest"),
    ):
        try:
            candidate.relative_to(predecessor_root)
        except ValueError as error:
            raise K12MigrationWorkflowError(f"{label} escapes predecessor root") from error
        if candidate.is_symlink() or not candidate.is_file():
            raise K12MigrationWorkflowError(f"{label} is missing: {candidate}")
    paths = Paths(
        predecessor=predecessor_root,
        successor=output_root,
        predecessor_release=predecessor_release,
        predecessor_plan=predecessor_plan,
        predecessor_bootstrap=predecessor_bootstrap_manifest.parent,
    )
    fixed_inputs: dict[str, dict[str, Any]] = {}
    for candidate in _fixed_input_paths(runtime_path, predecessor_root, paths):
        if candidate.is_symlink() or not candidate.is_file():
            raise K12MigrationWorkflowError(f"fixed input is missing: {candidate}")
        fixed_inputs[str(candidate)] = {
            "sha256": sha256_file(candidate),
            "size_bytes": candidate.stat().st_size,
        }
    jobs = _jobs()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "status": "planned",
        "repo_root": str(REPO_ROOT),
        "runtime_path": str(runtime_path),
        "predecessor_root": str(predecessor_root),
        "output_root": str(output_root),
        "predecessor_seal": {
            "path": str(seal_path),
            "file_sha256": sha256_file(seal_path),
            "payload_sha256": seal_hash,
        },
        "paths": {
            "predecessor_release": str(paths.predecessor_release),
            "predecessor_plan": str(paths.predecessor_plan),
            "predecessor_bootstrap": str(paths.predecessor_bootstrap),
            "predecessor_tokenized": str(predecessor_root / "corpus" / "tokenized"),
            "successor_plan": str(paths.successor_plan),
            "successor_release": str(paths.release),
            "successor_tokenized": str(paths.tokenized),
        },
        "contract": {
            "language_counts": list(LANGUAGE_COUNTS),
            "no_c_or_d_ladders": True,
            "cpu_only": True,
            "nested_slurm_submission": False,
            "adoption_copy_method": "parallel_verified_copy_no_predecessor_inode_sharing",
            "acquisition_languages": list(ACQUISITION_LANGUAGES),
            "changed_physical_languages": 15,
            "reused_physical_languages": 185,
            "finalizer_workers": FINALIZER_WORKERS,
            "parallel_copy_array_tasks": COPY_ARRAY_TASKS,
            "tokenization_ranks": TOKENIZATION_RANKS,
            "tokenization_reuse": "exact_source_and_tokenizer_identity_only",
            "tokenization_reuse_materialization": "verified_content_copy_distinct_inode",
            "predecessor_seal_verification": "before_and_after_materialization",
        },
        "runtime": {
            "python_executable": str(runtime.get("bridge_python_executable", sys.executable)),
            "data": dict(data),
            "seeds": dict(runtime.get("seeds", {})),
            "max_submit_slots_per_invocation": submit_limit,
        },
        "resource_profiles": resources,
        "engine_provenance": {"megatron_bridge": _bridge_revision()},
        "storage_estimate": _storage_estimate(
            runtime=runtime, predecessor=predecessor_root, paths=paths
        ),
        "fixed_inputs": fixed_inputs,
        "jobs": jobs,
        "submit_slots": planned_slots,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    validate_manifest(manifest, verify_fixed_inputs=True)
    return manifest


def _array_indices(row: Mapping[str, Any]) -> list[int]:
    count = int(row.get("array_count", 1))
    return list(range(count)) if row.get("array") else [0]


def validate_manifest(
    manifest: Mapping[str, Any], *, verify_fixed_inputs: bool = True
) -> None:
    _verify_payload_hash(manifest, "manifest_sha256", label="migration launch manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("workflow_id") != WORKFLOW_ID:
        raise K12MigrationWorkflowError("unsupported K12 migration manifest")
    if Path(str(manifest.get("repo_root", ""))).resolve() != REPO_ROOT:
        raise K12MigrationWorkflowError("migration manifest belongs to another repository")
    paths = _paths_from_manifest(manifest)
    if paths.predecessor == paths.successor:
        raise K12MigrationWorkflowError("migration manifest aliases predecessor and successor")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or contract.get("language_counts") != [1, 12, 50, 200]:
        raise K12MigrationWorkflowError("migration manifest is not the K12 contract")
    if contract.get("cpu_only") is not True or contract.get("no_c_or_d_ladders") is not True:
        raise K12MigrationWorkflowError("migration manifest broadens the approved scope")
    resources = _validate_cpu_resources(manifest.get("resource_profiles"))
    engine = manifest.get("engine_provenance")
    if (
        not isinstance(engine, dict)
        or engine.get("megatron_bridge") != _bridge_revision()
    ):
        raise K12MigrationWorkflowError("Megatron-Bridge revision drifted")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("data"), dict):
        raise K12MigrationWorkflowError("migration manifest lacks runtime data")
    _validate_tokenization_resource_contract(runtime["data"], resources)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or jobs != _jobs():
        raise K12MigrationWorkflowError("migration DAG differs from the frozen CPU-only DAG")
    seen: set[str] = set()
    for row in jobs:
        job_id = str(row["job_id"])
        if job_id in seen or any(str(value) not in seen for value in row["dependencies"]):
            raise K12MigrationWorkflowError("migration DAG is duplicated or not topological")
        if row["resource"] not in resources:
            raise K12MigrationWorkflowError(f"job {job_id} has an unknown resource")
        seen.add(job_id)
    if int(manifest.get("submit_slots", -1)) != sum(
        int(row["array_count"]) for row in jobs
    ):
        raise K12MigrationWorkflowError("migration submit-slot count changed")
    estimate = manifest.get("storage_estimate")
    if (
        not isinstance(estimate, dict)
        or int(estimate.get("required_final_free_headroom_bytes", 0))
        < MINIMUM_POST_MIGRATION_HEADROOM_BYTES
        or int(estimate.get("required_available_bytes_at_preflight", 0))
        != int(estimate.get("migration_peak_additional_bytes", -1))
        + int(estimate.get("required_final_free_headroom_bytes", -1))
    ):
        raise K12MigrationWorkflowError("migration storage estimate is invalid")
    if verify_fixed_inputs:
        fixed = manifest.get("fixed_inputs")
        if not isinstance(fixed, dict) or not fixed:
            raise K12MigrationWorkflowError("migration manifest lacks fixed inputs")
        for raw_path, identity in fixed.items():
            path = Path(str(raw_path))
            if path.is_symlink() or not path.is_file() or not isinstance(identity, dict):
                raise K12MigrationWorkflowError(f"fixed input is missing: {path}")
            if path.stat().st_size != int(identity.get("size_bytes", -1)) or sha256_file(path) != identity.get("sha256"):
                raise K12MigrationWorkflowError(f"fixed input changed after planning: {path}")


def write_manifest(manifest: Mapping[str, Any]) -> Path:
    validate_manifest(manifest, verify_fixed_inputs=True)
    path = Path(str(manifest["output_root"])) / "control" / "k12_migration" / MANIFEST_NAME
    _write_frozen_json(path, manifest)
    return path


def _job_by_id(manifest: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    for row in manifest["jobs"]:
        if row["job_id"] == job_id:
            return dict(row)
    raise K12MigrationWorkflowError(f"unknown migration job: {job_id}")


def _task_index(row: Mapping[str, Any], raw: str | None = None) -> int | None:
    if not row.get("array"):
        return None
    value = raw if raw is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None or not str(value).isdigit():
        raise K12MigrationWorkflowError(f"array job {row['job_id']} lacks a numeric task ID")
    index = int(value)
    if index not in _array_indices(row):
        raise K12MigrationWorkflowError(f"array index {index} is outside {row['job_id']}")
    return index


def _receipt_path(manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None) -> Path:
    suffix = f".{task:03d}" if task is not None else ""
    return (
        Path(str(manifest["output_root"]))
        / "control"
        / "k12_migration"
        / "receipts"
        / f"{row['job_id']}{suffix}.json"
    )


def _completion_path(
    manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None
) -> Path:
    paths = _paths_from_manifest(manifest)
    action = str(row["action"])
    if action == "storage_preflight":
        return paths.control / "storage_preflight.json"
    if action == "predecessor_seal_verify":
        return paths.control / "stage_outputs" / f"{row['job_id']}.json"
    if action == "adopt_copy":
        assert task is not None
        return paths.control / "stage_outputs" / f"adopt_copy.{task:03d}.json"
    if action == "adopt_finalize":
        return paths.adoption_manifest
    if action == "migration_prepare":
        return paths.control / "stage_outputs" / "migration_prepare.json"
    if action == "acquisition_preflight":
        return paths.incremental_bootstrap / "paths_preflight_summary.json"
    if action == "acquisition_worker":
        assert task is not None
        raw_languages = manifest.get("contract", {}).get(
            "acquisition_languages", ACQUISITION_LANGUAGES
        )
        languages = tuple(str(value) for value in raw_languages)
        if task < 0 or task >= len(languages):
            raise K12MigrationWorkflowError("acquisition task index is out of range")
        return paths.incremental_bootstrap / languages[task] / "bootstrap_report.json"
    if action == "acquisition_merge":
        return paths.incremental_bootstrap / "bootstrap_manifest.json"
    if action in {"bootstrap_copy_worker", "release_copy_worker"}:
        assert task is not None
        return paths.control / "stage_outputs" / f"{row['job_id']}.{task:03d}.json"
    if action == "bootstrap_join":
        return paths.control / "stage_outputs" / "bootstrap_join.json"
    if action == "finalizer_prepare":
        return paths.finalizer_work / "parallel_finalizer_plan.json"
    if action == "finalizer_worker":
        assert task is not None
        return paths.finalizer_work / "receipts" / f"rank_{task:03d}.json"
    if action == "finalizer_merge":
        return paths.delta_release / "corpus_manifest.json"
    if action == "release_join":
        return paths.control / "stage_outputs" / "release_join.json"
    if action == "tokenize_plan":
        return paths.tokenization_work / "plan" / "summary.json"
    if action == "tokenize_worker":
        assert task is not None
        return paths.tokenization_work / "ranks" / f"rank_{task:05d}" / "result.json"
    if action == "tokenize_merge":
        return paths.tokenized / "tokenization_report.json"
    if action == "completion_gate":
        return paths.control / "migration_complete.json"
    raise K12MigrationWorkflowError(f"no completion artifact for {action}")


_HASH_KEYS = {
    "storage_preflight": "payload_sha256",
    "predecessor_seal_verify": "payload_sha256",
    "adopt_copy": "payload_sha256",
    "adopt_finalize": "adoption_manifest_sha256",
    "migration_prepare": "receipt_sha256",
    "acquisition_preflight": "summary_sha256",
    "acquisition_worker": "report_payload_sha256",
    "acquisition_merge": "manifest_sha256",
    "bootstrap_copy_worker": "payload_sha256",
    "bootstrap_join": "receipt_sha256",
    "finalizer_prepare": "plan_sha256",
    "finalizer_worker": "receipt_sha256",
    "finalizer_merge": "manifest_sha256",
    "release_copy_worker": "payload_sha256",
    "release_join": "receipt_sha256",
    "tokenize_plan": "plan_payload_sha256",
    "tokenize_worker": "result_payload_sha256",
    "tokenize_merge": "report_payload_sha256",
    "completion_gate": "payload_sha256",
}


def _verify_adoption_copy(payload: Mapping[str, Any]) -> None:
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise K12MigrationWorkflowError("adoption-copy evidence has no files")
    destination_hashes: dict[str, str] = {}
    for relative, raw in files.items():
        if not isinstance(raw, dict):
            raise K12MigrationWorkflowError("adoption-copy file evidence is invalid")
        source = Path(str(raw.get("source", "")))
        destination = Path(str(raw.get("destination", "")))
        if (
            source.is_symlink()
            or destination.is_symlink()
            or not source.is_file()
            or not destination.is_file()
            or sha256_file(source) != raw.get("source_sha256")
            or sha256_file(destination) != raw.get("destination_sha256")
            or destination.stat().st_mode & 0o222
            or (source.stat().st_dev, source.stat().st_ino)
            == (destination.stat().st_dev, destination.stat().st_ino)
        ):
            raise K12MigrationWorkflowError(f"adoption-copy artifact changed: {relative}")
        destination_hashes[str(relative)] = str(raw["destination_sha256"])
    if canonical_json_sha256(destination_hashes) != payload.get("destination_tree_sha256"):
        raise K12MigrationWorkflowError("adoption-copy destination tree changed")


def _verify_copy_worker_bundle(
    payload: Mapping[str, Any], *, kind: str, task: int
) -> None:
    receipts = payload.get("worker_receipts")
    if not isinstance(receipts, dict) or not receipts:
        raise K12MigrationWorkflowError("parallel-copy bundle has no worker receipts")
    expected_indices = list(range(task, 200, COPY_ARRAY_TASKS))
    if (
        payload.get("status") != "complete"
        or payload.get("kind") != kind
        or payload.get("array_task") != task
        or payload.get("language_indices") != expected_indices
        or len(receipts) != len(expected_indices)
    ):
        raise K12MigrationWorkflowError("parallel-copy bundle coverage is invalid")
    for raw_index in expected_indices:
        key = str(int(raw_index))
        raw = receipts.get(key)
        if not isinstance(raw, dict):
            raise K12MigrationWorkflowError(f"parallel-copy bundle lacks index {key}")
        path = Path(str(raw.get("path", "")))
        if path.is_symlink() or not path.is_file() or sha256_file(path) != raw.get("sha256"):
            raise K12MigrationWorkflowError(f"parallel-copy worker receipt changed: {path}")
        worker = _load_object(path, label="parallel-copy worker receipt")
        _verify_payload_hash(worker, "receipt_sha256", label="parallel-copy worker receipt")
        expected_command = f"{kind}-merge-worker"
        if (
            int(worker.get("language_index", -1)) != int(raw_index)
            or worker.get("command") != expected_command
            or worker.get("predecessor_inode_sharing") is not False
        ):
            raise K12MigrationWorkflowError("parallel-copy worker is misbound or shares predecessor inodes")


def _manifest_acquisition_languages(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("contract", {}).get(
        "acquisition_languages", ACQUISITION_LANGUAGES
    )
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
        or len(raw) != len(set(raw))
    ):
        raise K12MigrationWorkflowError(
            "manifest acquisition_languages must be unique non-empty strings"
        )
    return tuple(str(value) for value in raw)


def _manifest_finalizer_workers(manifest: Mapping[str, Any]) -> int:
    raw = manifest.get("contract", {}).get("finalizer_workers", FINALIZER_WORKERS)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise K12MigrationWorkflowError(
            "manifest finalizer_workers must be a positive integer"
        )
    return raw


def _verify_completion_artifact(
    manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None
) -> dict[str, Any]:
    path = _completion_path(manifest, row, task)
    if path.is_symlink() or not path.is_file():
        raise K12MigrationWorkflowError(f"completion artifact is missing: {path}")
    payload = _load_object(path, label=f"{row['job_id']} completion artifact")
    action = str(row["action"])
    _verify_payload_hash(payload, _HASH_KEYS[action], label=f"{row['job_id']} completion artifact")
    if action == "adopt_copy":
        _verify_adoption_copy(payload)
    elif action == "storage_preflight":
        if payload.get("status") != "passed" or payload.get("go_no_go") != "go":
            raise K12MigrationWorkflowError("migration storage preflight is not passing")
    elif action in {"bootstrap_copy_worker", "release_copy_worker"}:
        assert task is not None
        _verify_copy_worker_bundle(
            payload,
            kind="bootstrap" if action == "bootstrap_copy_worker" else "release",
            task=task,
        )
    elif action == "predecessor_seal_verify":
        if payload.get("status") != "verified" or payload.get("predecessor_root") != str(
            _paths_from_manifest(manifest).predecessor
        ):
            raise K12MigrationWorkflowError("predecessor seal verification is invalid")
    elif action == "adopt_finalize":
        if payload.get("status") != "complete" or payload.get("tool_id") != migration.TOOL_ID:
            raise K12MigrationWorkflowError("upstream adoption is not complete")
    elif action in {"migration_prepare", "bootstrap_join", "release_join"}:
        if payload.get("status") != "complete":
            raise K12MigrationWorkflowError(f"{action} migration receipt is incomplete")
    elif action == "acquisition_preflight":
        acquisition_languages = _manifest_acquisition_languages(manifest)
        if (
            payload.get("languages") != list(acquisition_languages)
            or payload.get("language_map_sha256") != sha256_file(_language_map(200))
            or payload.get("language_selection_map_sha256")
            != sha256_file(_paths_from_manifest(manifest).acquisition_map)
        ):
            raise K12MigrationWorkflowError("acquisition preflight coverage changed")
    elif action == "acquisition_merge":
        acquisition_languages = _manifest_acquisition_languages(manifest)
        provenance = payload.get("provenance")
        if (
            set(payload.get("languages", {})) != set(acquisition_languages)
            or not isinstance(provenance, dict)
            or provenance.get("language_map_sha256") != sha256_file(_language_map(200))
            or provenance.get("language_selection_map_sha256")
            != sha256_file(_paths_from_manifest(manifest).acquisition_map)
        ):
            raise K12MigrationWorkflowError("incremental bootstrap coverage changed")
    elif action == "finalizer_worker":
        if payload.get("status") != "complete" or int(payload.get("rank", -1)) != task:
            raise K12MigrationWorkflowError("parallel finalizer receipt is invalid")
    elif action == "finalizer_merge":
        finalizer_workers = _manifest_finalizer_workers(manifest)
        if (
            payload.get("ready") is not True
            or len(payload.get("languages", {})) != finalizer_workers
        ):
            raise K12MigrationWorkflowError(
                "delta release language count differs from finalizer_workers"
            )
    elif action == "tokenize_worker":
        if int(payload.get("rank", -1)) != task:
            raise K12MigrationWorkflowError("tokenization rank result is misbound")
    elif action == "tokenize_merge":
        if (
            payload.get("pass") is not True
            or payload.get("language_count") != 200
            or payload.get("item_count") != 400
            or payload.get("maximum_tier") != "K200"
        ):
            raise K12MigrationWorkflowError("incremental tokenization report is not complete")
    return payload


def _write_receipt(
    manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None, *, recovered: bool
) -> dict[str, Any]:
    artifact = _completion_path(manifest, row, task)
    _verify_completion_artifact(manifest, row, task)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "job_id": row["job_id"],
        "array_task_id": task,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "status": "complete",
        "recovered_from_existing_artifact": recovered,
        "completion_artifact": {
            "path": str(artifact),
            "sha256": sha256_file(artifact),
        },
    }
    payload["receipt_sha256"] = _canonical_hash(payload)
    _write_frozen_json(_receipt_path(manifest, row, task), payload)
    return payload


def _verify_receipt(
    manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None
) -> dict[str, Any]:
    path = _receipt_path(manifest, row, task)
    receipt = _load_object(path, label=f"{row['job_id']} receipt")
    _verify_payload_hash(receipt, "receipt_sha256", label=f"{row['job_id']} receipt")
    if (
        receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("job_id") != row["job_id"]
        or receipt.get("array_task_id") != task
        or receipt.get("status") != "complete"
    ):
        raise K12MigrationWorkflowError(f"receipt is not bound to {row['job_id']}/{task}")
    artifact = _completion_path(manifest, row, task)
    if receipt.get("completion_artifact", {}).get("path") != str(artifact) or sha256_file(artifact) != receipt.get("completion_artifact", {}).get("sha256"):
        raise K12MigrationWorkflowError(f"receipt artifact changed for {row['job_id']}/{task}")
    _verify_completion_artifact(manifest, row, task)
    return receipt


def _job_complete(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    try:
        for task in (_array_indices(row) if row.get("array") else [None]):
            _verify_receipt(manifest, row, task)
    except (K12MigrationWorkflowError, FileNotFoundError):
        return False
    return True


def _require_dependencies(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    for dependency in row["dependencies"]:
        dependency_row = _job_by_id(manifest, str(dependency))
        if not _job_complete(manifest, dependency_row):
            raise K12MigrationWorkflowError(
                f"{row['job_id']} lacks hash-valid dependency evidence for {dependency}"
            )


def _copy_file_atomic(source: Path, destination: Path, payload: bytes | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        if payload is None:
            shutil.copy2(source, temporary)
        else:
            temporary.write_bytes(payload)
        os.chmod(temporary, temporary.stat().st_mode & ~0o222)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _run_adoption_copy(manifest: Mapping[str, Any], task: int) -> None:
    paths = _paths_from_manifest(manifest)
    relative_root = migration.UPSTREAM_ADOPTION_PATHS[task]
    source_root = paths.predecessor / relative_root
    destination_root = paths.successor / relative_root
    source_files = migration._adoption_source_files(source_root)
    rows: dict[str, Any] = {}
    destination_hashes: dict[str, str] = {}
    total_bytes = 0
    for child_relative, source in source_files:
        destination = (
            destination_root if child_relative == "." else destination_root / child_relative
        )
        full_relative = relative_root if child_relative == "." else f"{relative_root}/{child_relative}"
        transformed = migration._render_rebased_adoption_json(
            source=source,
            relative_root=full_relative,
            predecessor_root=paths.predecessor,
            successor_root=paths.successor,
        )
        expected = hashlib.sha256(transformed).hexdigest() if transformed is not None else sha256_file(source)
        if destination.exists() or destination.is_symlink():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or sha256_file(destination) != expected
                or destination.stat().st_mode & 0o222
                or (source.stat().st_dev, source.stat().st_ino)
                == (destination.stat().st_dev, destination.stat().st_ino)
            ):
                raise K12MigrationWorkflowError(f"partial adoption target differs: {destination}")
        else:
            _copy_file_atomic(source, destination, transformed)
        destination_hash = sha256_file(destination)
        destination_hashes[child_relative] = destination_hash
        total_bytes += destination.stat().st_size
        rows[child_relative] = {
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "source_sha256": sha256_file(source),
            "destination_sha256": destination_hash,
            "size_bytes": destination.stat().st_size,
            "method": "rebased_copy" if transformed is not None else "verified_full_copy",
            "same_inode": False,
        }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "task": task,
        "relative_root": relative_root,
        "status": "complete",
        "method": "parallel_verified_copy_no_predecessor_inode_sharing",
        "file_count": len(rows),
        "size_bytes": total_bytes,
        "destination_tree_sha256": canonical_json_sha256(destination_hashes),
        "files": rows,
    }
    result["payload_sha256"] = _canonical_hash(result)
    _write_frozen_json(_completion_path(manifest, _job_by_id(manifest, "adopt_copy"), task), result)


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def _quarantine_owned_partials(
    manifest: Mapping[str, Any], *, label: str, candidates: Sequence[Path]
) -> list[str]:
    """Atomically preserve incomplete successor-owned outputs before a retry."""

    paths = _paths_from_manifest(manifest)
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if not existing:
        return []
    quarantine = paths.control / "retry_quarantine" / f"{label}.{uuid.uuid4().hex}"
    quarantine.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, str]] = []
    for index, source in enumerate(existing):
        if source.is_symlink():
            raise K12MigrationWorkflowError(
                f"refusing to quarantine a symlinked workflow output: {source}"
            )
        try:
            source.resolve().relative_to(paths.successor)
        except ValueError as error:
            raise K12MigrationWorkflowError(
                f"retry-owned output escapes the successor root: {source}"
            ) from error
        destination = quarantine / f"{index:03d}_{source.name}"
        os.replace(source, destination)
        moved.append({"source": str(source), "quarantine": str(destination)})
    _atomic_json(
        quarantine / "quarantine.json",
        {
            "schema_version": SCHEMA_VERSION,
            "workflow_id": WORKFLOW_ID,
            "label": label,
            "reason": "incomplete workflow-owned output without a hash-valid completion receipt",
            "moved": moved,
        },
    )
    return [row["quarantine"] for row in moved]


def _bootstrap_args(manifest: Mapping[str, Any]) -> list[str]:
    paths = _paths_from_manifest(manifest)
    data = manifest["runtime"]["data"]
    seeds = manifest["runtime"]["seeds"]
    return [
        sys.executable,
        "-m",
        BOOTSTRAP_MODULE,
        "--source-lock",
        str(_abs("configs/data_sources/fineweb2_v2_1_1_source_lock.json")),
        "--rehydration-weights",
        str(paths.corpus / "source_lock" / "fineweb2-up-to-10-reps.json"),
        "--source-inventory",
        str(paths.corpus / "source_lock" / "fineweb2-language-distribution.csv"),
        "--language-map",
        str(_language_map(200)),
        "--language-selection-map",
        str(paths.acquisition_map),
        "--output-dir",
        str(paths.incremental_bootstrap),
        "--seed",
        str(seeds["data"]),
        "--validation-fraction",
        str(data["validation_candidate_fraction"]),
        "--target-unique-documents",
        str(data["bootstrap_target_unique_documents"]),
        "--target-utf8-bytes",
        str(data["bootstrap_target_utf8_bytes"]),
        "--calibration-documents",
        str(data["calibration_documents_per_language"]),
        "--sampling-plan",
        str(paths.successor_plan),
        "--production-safety-margin",
        str(data["production_safety_margin"]),
        "--incremental-from",
        str(paths.predecessor_bootstrap),
    ]


def _run_action(
    manifest: Mapping[str, Any], row: Mapping[str, Any], task: int | None
) -> None:
    paths = _paths_from_manifest(manifest)
    action = str(row["action"])
    if action == "storage_preflight":
        _run_storage_preflight(manifest, row)
        return
    if action == "predecessor_seal_verify":
        _run_predecessor_seal_verification(manifest, row)
        return
    if action == "adopt_copy":
        assert task is not None
        _run_adoption_copy(manifest, task)
        return
    if action == "adopt_finalize":
        _run_command(
            [
                sys.executable,
                "-m",
                MIGRATION_MODULE,
                "adopt-upstream",
                "--predecessor-root",
                str(paths.predecessor),
                "--successor-root",
                str(paths.successor),
                "--predecessor-seal",
                str(paths.predecessor / "control" / "data_artifact_seal.json"),
            ]
        )
        return
    if action == "migration_prepare":
        command = [
            sys.executable,
            "-m",
            MIGRATION_MODULE,
            "prepare",
            "--predecessor-plan",
            str(paths.predecessor_plan),
            "--predecessor-release",
            str(paths.predecessor_release),
            "--adoption-manifest",
            str(paths.adoption_manifest),
        ]
        for k in LANGUAGE_COUNTS:
            command.extend(["--language-map", f"K{k}={_language_map(k)}"])
        command.extend(
            [
                "--successor-plan",
                str(paths.successor_plan),
                "--delta-plan",
                str(paths.delta_plan),
                "--acquisition-language-map",
                str(paths.acquisition_map),
                "--migration-plan",
                str(paths.migration_plan),
                "--receipt",
                str(_completion_path(manifest, row, task)),
            ]
        )
        _run_command(command)
        return
    if action == "acquisition_preflight":
        _run_command([*_bootstrap_args(manifest), "--metadata-preflight"])
        return
    if action == "acquisition_worker":
        assert task is not None
        acquisition_languages = _manifest_acquisition_languages(manifest)
        language_root = paths.incremental_bootstrap / acquisition_languages[task]
        partials = [
            language_root / name
            for name in ("train.jsonl", "validation_reserve.jsonl", "calibration.jsonl")
        ]
        partials.extend(
            path
            for pattern in (".train.jsonl.*", ".validation_reserve.jsonl.*", ".calibration.jsonl.*")
            for path in language_root.glob(pattern)
        )
        _quarantine_owned_partials(
            manifest,
            label=f"acquisition_{acquisition_languages[task]}",
            candidates=partials,
        )
        _run_command([*_bootstrap_args(manifest), "--language-index", str(task)])
        return
    if action == "acquisition_merge":
        _run_command([*_bootstrap_args(manifest), "--merge-summary"])
        return
    if action == "bootstrap_copy_worker":
        assert task is not None
        _run_parallel_copy_worker(manifest, row, task, kind="bootstrap")
        return
    if action == "bootstrap_join":
        _run_command(
            [
                sys.executable,
                "-m",
                MIGRATION_MODULE,
                "bootstrap-merge",
                "--migration-plan",
                str(paths.migration_plan),
                "--predecessor-bootstrap",
                str(paths.predecessor_bootstrap),
                "--incremental-bootstrap",
                str(paths.incremental_bootstrap),
                "--work-root",
                str(paths.migration / "bootstrap_merge_work"),
                "--output-dir",
                str(paths.production_bootstrap),
                "--receipt",
                str(_completion_path(manifest, row, task)),
                "--merge-summary",
            ]
        )
        return
    if action == "finalizer_prepare":
        finalizer_workers = _manifest_finalizer_workers(manifest)
        _run_command(
            [
                sys.executable,
                "-m",
                FINALIZER_MODULE,
                "prepare",
                "--bootstrap-manifest",
                str(paths.production_bootstrap / "bootstrap_delta_manifest.json"),
                "--sampling-plan",
                str(paths.delta_plan),
                "--work-root",
                str(paths.finalizer_work),
                "--workers",
                str(finalizer_workers),
            ]
        )
        return
    if action == "finalizer_worker":
        assert task is not None
        finalizer_workers = _manifest_finalizer_workers(manifest)
        rank_parent = paths.finalizer_work / "ranks" / f"rank_{task:03d}"
        _quarantine_owned_partials(
            manifest,
            label=f"finalizer_rank_{task:03d}",
            candidates=[
                rank_parent / "release",
                *rank_parent.glob(".release.partial-*"),
            ],
        )
        index_root = Path(
            str(manifest["runtime"]["data"]["runtime_env"]["TMPDIR"])
        ) / f"abv2-k12-finalize-{os.environ.get('SLURM_JOB_ID', 'local')}-{task:03d}"
        index_root.mkdir(parents=True, exist_ok=True)
        _run_command(
            [
                sys.executable,
                "-m",
                FINALIZER_MODULE,
                "worker",
                "--work-root",
                str(paths.finalizer_work),
                "--tokenizer-model",
                str(paths.tokenizer_model),
                "--rank",
                str(task),
                "--workers",
                str(finalizer_workers),
                "--tokenization-workers",
                "2",
                "--index-work-root",
                str(index_root),
            ]
        )
        return
    if action == "finalizer_merge":
        _run_command(
            [
                sys.executable,
                "-m",
                FINALIZER_MODULE,
                "merge",
                "--work-root",
                str(paths.finalizer_work),
                "--output-dir",
                str(paths.delta_release),
                "--tokenizer-model",
                str(paths.tokenizer_model),
            ]
        )
        return
    if action == "release_copy_worker":
        assert task is not None
        _run_parallel_copy_worker(manifest, row, task, kind="release")
        return
    if action == "release_join":
        _run_command(
            [
                sys.executable,
                "-m",
                MIGRATION_MODULE,
                "release-merge",
                "--migration-plan",
                str(paths.migration_plan),
                "--predecessor-release",
                str(paths.predecessor_release),
                "--delta-release",
                str(paths.delta_release),
                "--bootstrap-manifest",
                str(paths.production_bootstrap / "bootstrap_manifest.json"),
                "--work-root",
                str(paths.migration / "release_merge_work"),
                "--output-dir",
                str(paths.release),
                "--receipt",
                str(_completion_path(manifest, row, task)),
                "--merge-summary",
            ]
        )
        return
    if action == "tokenize_plan":
        data = manifest["runtime"]["data"]
        _run_command(
            [
                sys.executable,
                "-m",
                TOKENIZATION_MODULE,
                "plan",
                "--corpus-manifest",
                str(paths.release / "corpus_manifest.json"),
                "--sampling-plan",
                str(paths.successor_plan),
                "--tokenizer-model",
                str(paths.tokenizer_model),
                "--plan-dir",
                str(paths.tokenization_work / "plan"),
                "--array-ranks",
                str(TOKENIZATION_RANKS),
                "--verification-workers",
                str(data["tokenization_plan_verification_workers"]),
            ]
        )
        return
    if action == "tokenize_worker":
        assert task is not None
        data = manifest["runtime"]["data"]
        _quarantine_owned_partials(
            manifest,
            label=f"tokenize_rank_{task:05d}",
            candidates=[paths.tokenization_work / "ranks" / f"rank_{task:05d}"],
        )
        _run_command(
            [
                sys.executable,
                "-m",
                TOKENIZATION_MODULE,
                "worker",
                "--plan-dir",
                str(paths.tokenization_work / "plan"),
                "--ranks-dir",
                str(paths.tokenization_work / "ranks"),
                "--rank",
                str(task),
                "--bridge-root",
                str(_abs("external/Megatron-Bridge")),
                "--python-executable",
                str(manifest["runtime"]["python_executable"]),
                "--workers",
                str(data["tokenization_workers_per_rank"]),
                "--partitions-per-language",
                str(data["tokenization_partitions_per_language"]),
                "--reuse-tokenized-root",
                str(paths.predecessor / "corpus" / "tokenized"),
                "--reuse-corpus-manifest",
                str(paths.predecessor_release / "corpus_manifest.json"),
            ]
        )
        return
    if action == "tokenize_merge":
        _run_command(
            [
                sys.executable,
                "-m",
                TOKENIZATION_MODULE,
                "merge",
                "--plan-dir",
                str(paths.tokenization_work / "plan"),
                "--ranks-dir",
                str(paths.tokenization_work / "ranks"),
                "--output-dir",
                str(paths.tokenized),
            ]
        )
        return
    if action == "completion_gate":
        _run_completion_gate(manifest, row)
        return
    raise K12MigrationWorkflowError(f"unimplemented migration action: {action}")


def _run_parallel_copy_worker(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    task: int,
    *,
    kind: str,
) -> None:
    """Run one bounded chunk of language-owned migration copy workers."""

    paths = _paths_from_manifest(manifest)
    if kind not in {"bootstrap", "release"}:
        raise K12MigrationWorkflowError(f"unknown copy-worker kind: {kind}")
    indices = list(range(task, 200, COPY_ARRAY_TASKS))
    work_root = paths.migration / f"{kind}_merge_work"
    receipts: dict[str, Any] = {}
    for index in indices:
        command = [
            sys.executable,
            "-m",
            MIGRATION_MODULE,
            f"{kind}-merge",
            "--migration-plan",
            str(paths.migration_plan),
        ]
        if kind == "bootstrap":
            command.extend(
                [
                    "--predecessor-bootstrap",
                    str(paths.predecessor_bootstrap),
                    "--incremental-bootstrap",
                    str(paths.incremental_bootstrap),
                ]
            )
            receipt = work_root / "receipts" / f"bootstrap.{index:03d}.json"
        else:
            command.extend(
                [
                    "--predecessor-release",
                    str(paths.predecessor_release),
                    "--delta-release",
                    str(paths.delta_release),
                    "--bootstrap-manifest",
                    str(paths.production_bootstrap / "bootstrap_manifest.json"),
                ]
            )
            receipt = work_root / "receipts" / f"release.{index:03d}.json"
        command.extend(
            [
                "--work-root",
                str(work_root),
                "--language-index",
                str(index),
            ]
        )
        _run_command(command)
        worker = _load_object(receipt, label=f"{kind} copy-worker receipt")
        _verify_payload_hash(worker, "receipt_sha256", label=f"{kind} copy-worker receipt")
        if worker.get("predecessor_inode_sharing") is not False:
            raise K12MigrationWorkflowError(f"{kind} worker shares a predecessor inode")
        receipts[str(index)] = {"path": str(receipt), "sha256": sha256_file(receipt)}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "status": "complete",
        "kind": kind,
        "array_task": task,
        "language_indices": indices,
        "worker_receipts": receipts,
        "predecessor_inode_sharing": False,
    }
    result["payload_sha256"] = _canonical_hash(result)
    _write_frozen_json(_completion_path(manifest, row, task), result)


def _run_storage_preflight(
    manifest: Mapping[str, Any], row: Mapping[str, Any]
) -> None:
    """Gate the copy-heavy migration before any corpus path is materialized."""

    paths = _paths_from_manifest(manifest)
    if paths.corpus.exists():
        materialized = [path for path in paths.corpus.rglob("*") if path.is_file()]
        if materialized:
            raise K12MigrationWorkflowError(
                "storage preflight must precede successor corpus materialization; found "
                + str(materialized[0])
            )
    paths.successor.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(paths.successor)
    available = int(stats.f_bavail * stats.f_frsize)
    total = int(stats.f_blocks * stats.f_frsize)
    available_inodes = int(stats.f_favail)
    total_inodes = int(stats.f_files)
    estimate = manifest["storage_estimate"]
    required = int(estimate["required_available_bytes_at_preflight"])
    required_inodes = int(estimate.get("required_available_inodes_at_preflight", 0))
    passed = available >= required and available_inodes >= required_inodes
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "status": "passed" if passed else "failed",
        "go_no_go": "go" if passed else "hold",
        "filesystem": {
            "path": str(paths.successor),
            "available_bytes": available,
            "total_bytes": total,
            "available_inodes": available_inodes,
            "total_inodes": total_inodes,
        },
        "storage_estimate": estimate,
        "required_available_bytes": required,
        "remaining_bytes_after_reservation": available - required,
        "required_available_inodes": required_inodes,
        "remaining_inodes_after_reservation": available_inodes - required_inodes,
    }
    result["payload_sha256"] = _canonical_hash(result)
    if not passed:
        _atomic_json(paths.control / "storage_preflight_last_failure.json", result)
        raise K12MigrationWorkflowError(
            "migration storage preflight failed: "
            f"bytes available={available} required={required}; "
            f"inodes available={available_inodes} required={required_inodes}"
        )
    _write_frozen_json(_completion_path(manifest, row, None), result)


def _run_completion_gate(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    paths = _paths_from_manifest(manifest)
    plan = _load_object(paths.migration_plan, label="K12 migration plan")
    _verify_payload_hash(plan, "migration_plan_sha256", label="K12 migration plan")
    transition = plan.get("transition")
    if not isinstance(transition, dict) or transition.get("counts") != {
        "changed_physical": 15,
        "increased_physical": 11,
        "decreased_physical": 4,
        "acquisition": 9,
        "delta_finalize": 15,
        "reused_physical": 185,
        "exposure_changed": 16,
    }:
        raise K12MigrationWorkflowError("K12 migration transition counts changed")
    release = _load_object(paths.release / "corpus_manifest.json", label="K12 release")
    _verify_payload_hash(release, "manifest_sha256", label="K12 release")
    if release.get("ready") is not True or len(release.get("languages", {})) != 200:
        raise K12MigrationWorkflowError("K12 release is not ready with 200 languages")
    report = _load_object(paths.tokenized / "tokenization_report.json", label="K12 tokenization")
    _verify_payload_hash(report, "report_payload_sha256", label="K12 tokenization")
    reuse = report.get("incremental_tokenization")
    if (
        not isinstance(reuse, dict)
        or reuse.get("enabled") is not True
        or reuse.get("materialization") != "verified_content_copy_distinct_inode"
    ):
        raise K12MigrationWorkflowError("incremental tokenization reuse is not enabled")
    reused = reuse.get("reused_items_by_split")
    rebuilt = reuse.get("rebuilt_items_by_split")
    if not isinstance(reused, dict) or not isinstance(rebuilt, dict):
        raise K12MigrationWorkflowError("incremental tokenization lacks item provenance")
    if len(reused.get("train", [])) != 185 or len(rebuilt.get("train", [])) != 15:
        raise K12MigrationWorkflowError("train-prefix reuse differs from the 185/15 K12 delta")
    if len(reused.get("validation", [])) < 191 or len(rebuilt.get("validation", [])) > 9:
        raise K12MigrationWorkflowError("validation-prefix drift exceeds the nine acquired languages")
    if set(report.get("tiers", {})) != {"K1", "K12", "K50", "K200"}:
        raise K12MigrationWorkflowError("tokenized tiers are not exactly K1/K12/K50/K200")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "migration_plan_sha256": plan["migration_plan_sha256"],
        "release_manifest_sha256": release["manifest_sha256"],
        "tokenization_report_sha256": report["report_payload_sha256"],
        "transition_counts": transition["counts"],
        "incremental_tokenization": {
            "reused_train": len(reused["train"]),
            "rebuilt_train": len(rebuilt["train"]),
            "reused_validation": len(reused["validation"]),
            "rebuilt_validation": len(rebuilt["validation"]),
        },
        "next_step": "canonical A/B-v2 data audit and seal; no GPU jobs submitted here",
    }
    result["payload_sha256"] = _canonical_hash(result)
    _write_frozen_json(_completion_path(manifest, row, None), result)


def _run_predecessor_seal_verification(
    manifest: Mapping[str, Any], row: Mapping[str, Any]
) -> None:
    """Verify the old stat-identity seal without rewriting any old artifact."""

    from multilingual_megatron.experiments.scaling.ab_v2.artifacts import (
        _verify_data_artifact_seal,
    )
    from multilingual_megatron.experiments.scaling.ab_v2.common import Paths as ABV2Paths

    paths = _paths_from_manifest(manifest)
    old_manifest_path = paths.predecessor / "control" / "launch_manifest.json"
    old_manifest = _load_object(old_manifest_path, label="predecessor launch manifest")
    evidence = _verify_data_artifact_seal(old_manifest, ABV2Paths(paths.predecessor))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "status": "verified",
        "phase": row["job_id"],
        "predecessor_root": str(paths.predecessor),
        "predecessor_manifest_sha256": old_manifest.get("manifest_sha256"),
        "seal": evidence,
    }
    result["payload_sha256"] = _canonical_hash(result)
    _write_frozen_json(_completion_path(manifest, row, None), result)


def execute_job(manifest_path: Path, job_id: str, *, task: int | None = None) -> None:
    manifest = _load_object(manifest_path.resolve(), label="migration launch manifest")
    validate_manifest(manifest, verify_fixed_inputs=True)
    row = _job_by_id(manifest, job_id)
    task_id = _task_index(row, str(task) if task is not None else None)
    receipt_path = _receipt_path(manifest, row, task_id)
    if receipt_path.is_file():
        _verify_receipt(manifest, row, task_id)
        return
    _require_dependencies(manifest, row)
    completion = _completion_path(manifest, row, task_id)
    if completion.is_file():
        _verify_completion_artifact(manifest, row, task_id)
        _write_receipt(manifest, row, task_id, recovered=True)
        return
    _run_action(manifest, row, task_id)
    _write_receipt(manifest, row, task_id, recovered=False)


def _resource_executor(manifest: Mapping[str, Any], resource: str) -> SlurmExecutor:
    raw = manifest["resource_profiles"][resource]
    return SlurmExecutor(
        partition=str(raw["partition"]),
        gres=str(raw.get("gres", "none")),
        mem=str(raw["mem"]),
        time_limit=str(raw["time"]),
        cpus_per_task=int(raw["cpus_per_task"]),
        module_init=str(raw.get("module_init", "")),
        modules=[str(value) for value in raw.get("modules", [])],
        conda_base=str(raw.get("conda_base", "")),
        conda_env=str(raw.get("conda_env", "")),
        extra_sbatch_flags=[str(value) for value in raw.get("extra_sbatch_flags", [])],
    )


def _runner_command(
    manifest_path: Path, job_id: str, python: str, *, module: str = MODULE
) -> list[str]:
    """Build a worker command, allowing experiment-owned workflow facades."""

    return [python, "-m", module, "execute", "--manifest", str(manifest_path), "--job-id", job_id]


def _job_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    data_env = manifest["runtime"]["data"].get("runtime_env", {})
    environment = {str(key): str(value) for key, value in data_env.items()}
    environment.update(
        {
            "PROOF_PIPELINE_REPO_ROOT": str(REPO_ROOT),
            "PYTHONPATH": "src:.",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment


def _load_state(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    workflow_id = str(manifest.get("workflow_id", WORKFLOW_ID))
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "workflow_id": workflow_id,
            "manifest_sha256": manifest["manifest_sha256"],
            "jobs": {},
        }
    state = _load_object(path, label="K12 migration submission state")
    if state.get("manifest_sha256") != manifest["manifest_sha256"] or state.get("workflow_id") != workflow_id:
        raise K12MigrationWorkflowError("submission state belongs to another migration plan")
    if not isinstance(state.get("jobs"), dict):
        raise K12MigrationWorkflowError("submission state jobs must be an object")
    return state


def _normalized_slurm_state(value: Any) -> str:
    return str(value or "").strip().split()[0].split("+")[0].upper()


def _query_slurm_states(job_ids: Sequence[str]) -> dict[str, str]:
    ids = sorted({str(value) for value in job_ids if str(value)})
    if not ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "-j",
            ",".join(ids),
            "--format=JobIDRaw,State",
            "--noheader",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise K12MigrationWorkflowError(
            f"sacct failed: {(result.stderr or result.stdout).strip()}"
        )
    observed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        if not job_id or "." in job_id:
            continue
        observed[job_id] = _normalized_slurm_state(parts[1])
    return observed


def _query_pending_reasons(job_ids: Sequence[str]) -> dict[str, str]:
    ids = sorted({str(value) for value in job_ids if str(value)})
    if not ids:
        return {}
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--jobs",
            ",".join(ids),
            "--format=%A|%T|%r",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise K12MigrationWorkflowError(
            f"squeue failed: {(result.stderr or result.stdout).strip()}"
        )
    reasons: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id, state, reason = (value.strip() for value in parts)
        if job_id and _normalized_slurm_state(state) == "PENDING":
            reasons[job_id] = reason
    return reasons


def _cancel_stale_dependency_jobs(job_ids: Sequence[str]) -> None:
    ids = sorted({str(value) for value in job_ids if str(value)})
    if not ids:
        return
    result = subprocess.run(
        ["scancel", "--quiet", *ids],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise K12MigrationWorkflowError(
            f"could not cancel stale dependent jobs: {(result.stderr or result.stdout).strip()}"
        )


def _aggregate_slurm_state(
    slurm_id: str, row: Mapping[str, Any], observed: Mapping[str, str]
) -> str:
    parent = _normalized_slurm_state(observed.get(slurm_id))
    if not row.get("array"):
        return parent or "ACCOUNTING_INCOMPLETE"
    expected = [f"{slurm_id}_{index}" for index in _array_indices(row)]
    task_states = {
        task: _normalized_slurm_state(observed.get(task))
        for task in expected
        if observed.get(task)
    }
    active = next(
        (state for state in task_states.values() if state in SLURM_ACTIVE_STATES),
        None,
    )
    if active:
        return active
    if len(task_states) == len(expected):
        if all(state == "COMPLETED" for state in task_states.values()):
            return "COMPLETED"
        failed = next(
            (state for state in task_states.values() if state in SLURM_FAILURE_STATES),
            None,
        )
        if failed and all(
            state == "COMPLETED" or state in SLURM_FAILURE_STATES
            for state in task_states.values()
        ):
            return failed
    if parent in SLURM_ACTIVE_STATES or parent in SLURM_FAILURE_STATES:
        return parent
    # A COMPLETED array parent without complete task accounting is not enough
    # evidence to retry or trust; sacct may still be converging.
    return "ACCOUNTING_INCOMPLETE"


def _refresh_submission_state(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    *,
    query_states: Callable[[Sequence[str]], Mapping[str, str]],
) -> dict[str, Any]:
    slurm_ids = [
        str(raw.get("slurm_job_id"))
        for raw in state["jobs"].values()
        if raw.get("slurm_job_id") and not str(raw.get("slurm_job_id")).startswith("dryrun_")
    ]
    observed = dict(query_states(slurm_ids)) if slurm_ids else {}
    for row in manifest["jobs"]:
        job_id = str(row["job_id"])
        raw = state["jobs"].get(job_id)
        if not isinstance(raw, dict):
            continue
        if _job_complete(manifest, row):
            raw["state"] = "COMPLETED_EVIDENCE"
            continue
        slurm_id = str(raw.get("slurm_job_id", ""))
        if slurm_id and not slurm_id.startswith("dryrun_"):
            raw["state"] = _aggregate_slurm_state(slurm_id, row, observed)
    return state


def _prepare_terminal_retries(
    manifest: Mapping[str, Any],
    state: dict[str, Any],
    *,
    query_pending_reasons: Callable[[Sequence[str]], Mapping[str, str]],
    cancel_jobs: Callable[[Sequence[str]], None],
) -> None:
    retrying: set[str] = set()
    pending_ids = [
        str(raw.get("slurm_job_id"))
        for raw in state["jobs"].values()
        if _normalized_slurm_state(raw.get("state")) == "PENDING"
        and raw.get("slurm_job_id")
    ]
    pending_reasons = dict(query_pending_reasons(pending_ids)) if pending_ids else {}
    stale_pending: list[str] = []
    for row in manifest["jobs"]:
        job_id = str(row["job_id"])
        if _job_complete(manifest, row):
            continue
        raw = state["jobs"].get(job_id)
        if not isinstance(raw, dict) or not raw.get("slurm_job_id"):
            continue
        current = _normalized_slurm_state(raw.get("state"))
        dependency_retry = any(str(value) in retrying for value in row["dependencies"])
        retryable = current in SLURM_FAILURE_STATES or current == "COMPLETED"
        if dependency_retry and current == "PENDING":
            slurm_id = str(raw["slurm_job_id"])
            reason = str(pending_reasons.get(slurm_id, ""))
            if "DEPENDENCY" not in reason.upper():
                raise K12MigrationWorkflowError(
                    f"cannot retry {job_id}: pending reason is not dependency-owned: {reason!r}"
                )
            stale_pending.append(slurm_id)
        elif dependency_retry and current in SLURM_ACTIVE_STATES:
            raise K12MigrationWorkflowError(
                f"cannot retry {job_id}: its old dependent allocation is still {current}"
            )
        if dependency_retry and current == "ACCOUNTING_INCOMPLETE":
            raise K12MigrationWorkflowError(
                f"cannot retry {job_id}: Slurm accounting is incomplete; reconcile again later"
            )
        if retryable or dependency_retry:
            retrying.add(job_id)
    # Explicit --retry-failed authorizes cancellation only for descendants of
    # a terminal failed dependency and only while Slurm reports a dependency
    # pending reason.  These allocations cannot be reused because their frozen
    # afterok IDs point to the failed predecessor allocation.
    cancel_jobs(stale_pending)
    for job_id in retrying:
        prior = dict(state["jobs"][job_id])
        history = list(prior.get("submission_history", []))
        history.append(
            {
                "slurm_job_id": prior.get("slurm_job_id"),
                "terminal_state": prior.get("state"),
                "reason": "terminal retry with hash-valid task receipts preserved",
                "stale_dependency_cancelled": str(prior.get("slurm_job_id"))
                in stale_pending,
            }
        )
        state["jobs"][job_id] = {"submission_history": history}


def submit_workflow(
    manifest_path: Path,
    *,
    dry_run: bool = False,
    retry_failed: bool = False,
    executor_factory: Callable[[Mapping[str, Any], str], Any] = _resource_executor,
    query_states: Callable[[Sequence[str]], Mapping[str, str]] = _query_slurm_states,
    query_pending_reasons: Callable[[Sequence[str]], Mapping[str, str]] = _query_pending_reasons,
    cancel_jobs: Callable[[Sequence[str]], None] = _cancel_stale_dependency_jobs,
    validate_manifest_fn: Callable[..., None] = validate_manifest,
) -> dict[str, Any]:
    if os.environ.get("SLURM_JOB_ID"):
        raise K12MigrationWorkflowError("refusing nested Slurm submission from an allocation")
    manifest_path = manifest_path.resolve()
    manifest = _load_object(manifest_path, label="migration launch manifest")
    validate_manifest_fn(manifest, verify_fixed_inputs=True)
    output = Path(str(manifest["output_root"])).resolve()
    state_path = output / "control" / "k12_migration" / STATE_NAME
    state = _load_state(state_path, manifest)
    if state["jobs"] and not dry_run:
        _refresh_submission_state(manifest, state, query_states=query_states)
        if retry_failed:
            _prepare_terminal_retries(
                manifest,
                state,
                query_pending_reasons=query_pending_reasons,
                cancel_jobs=cancel_jobs,
            )
        _atomic_json(state_path, state)
    python = str(manifest["runtime"]["python_executable"])
    for row in manifest["jobs"]:
        job_id = str(row["job_id"])
        if _job_complete(manifest, row):
            state["jobs"][job_id] = {
                **state["jobs"].get(job_id, {}),
                "state": "COMPLETED_EVIDENCE",
            }
            continue
        prior = state["jobs"].get(job_id, {})
        if prior.get("slurm_job_id"):
            current = _normalized_slurm_state(prior.get("state"))
            if current in SLURM_FAILURE_STATES or current == "COMPLETED":
                raise K12MigrationWorkflowError(
                    f"{job_id} is terminal without complete receipt evidence; "
                    "rerun submit with --retry-failed"
                )
            if current == "ACCOUNTING_INCOMPLETE":
                raise K12MigrationWorkflowError(
                    f"{job_id} has incomplete Slurm accounting; reconcile again later"
                )
            continue
        dependency_ids: list[str] = []
        for dependency in row["dependencies"]:
            dependency_row = _job_by_id(manifest, str(dependency))
            if _job_complete(manifest, dependency_row):
                continue
            dependency_state = state["jobs"].get(str(dependency), {})
            dependency_job = str(dependency_state.get("slurm_job_id", ""))
            if not dependency_job:
                raise K12MigrationWorkflowError(
                    f"cannot submit {job_id}: dependency {dependency} has no receipt or Slurm ID"
                )
            dependency_ids.append(dependency_job)
        dependency_spec = (
            "afterok:" + ":".join(dependency_ids) if dependency_ids else None
        )
        executor = executor_factory(manifest, str(row["resource"]))
        log = output / "logs" / "k12_migration" / f"{job_id}.controller.log"
        slurm_log = output / "logs" / "k12_migration" / (
            f"{job_id}.%A_%a.log" if row.get("array") else f"{job_id}.%j.log"
        )
        kwargs = {
            "cmd": _runner_command(
                manifest_path,
                job_id,
                python,
                module=str(manifest.get("worker_module", MODULE)),
            ),
            "env": _job_environment(manifest),
            "dry_run": dry_run,
            "log_path": log,
            "dependency": dependency_spec,
            "job_name": f"abv2_k12_{job_id}",
            "script_dir": output / "control" / "k12_migration" / "submitted_sbatch",
            "slurm_output": slurm_log,
        }
        if row.get("array"):
            slurm_id = executor.submit_array(array=str(row["array"]), **kwargs)
        else:
            slurm_id = executor.submit(**kwargs)
        if dry_run and not slurm_id:
            slurm_id = f"dryrun_{len(state['jobs']) + 1}"
        if not slurm_id:
            raise K12MigrationWorkflowError(f"Slurm returned no job ID for {job_id}")
        state["jobs"][job_id] = {
            "slurm_job_id": str(slurm_id),
            "state": "DRY_RUN" if dry_run else "SUBMITTED",
            "resource": row["resource"],
            "array": row.get("array"),
            "dependencies": list(row["dependencies"]),
            "slurm_dependency": dependency_spec,
            "submission_history": list(prior.get("submission_history", [])),
        }
        if not dry_run:
            _atomic_json(state_path, state)
    if not dry_run:
        _atomic_json(state_path, state)
    return state


def status(
    manifest_path: Path,
    *,
    verify_fixed_inputs: bool = True,
    validate_manifest_fn: Callable[..., None] = validate_manifest,
) -> dict[str, Any]:
    """Report receipt-backed migration completion.

    Normal migration operation keeps fixed-input verification enabled.  A
    downstream workflow may query a *completed* migration with it disabled so
    later changes to code outside the materialization path cannot invalidate
    immutable historical corpus receipts.
    """

    manifest = _load_object(manifest_path.resolve(), label="migration launch manifest")
    validate_manifest_fn(manifest, verify_fixed_inputs=verify_fixed_inputs)
    state_path = Path(str(manifest["output_root"])) / "control" / "k12_migration" / STATE_NAME
    state = _load_state(state_path, manifest)
    rows = []
    for row in manifest["jobs"]:
        job_id = str(row["job_id"])
        rows.append(
            {
                "job_id": job_id,
                "evidence_complete": _job_complete(manifest, row),
                "slurm": state["jobs"].get(job_id),
            }
        )
    return {
        "workflow_id": manifest["workflow_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "complete": all(row["evidence_complete"] for row in rows),
        "jobs": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="freeze the CPU-only K12 migration DAG")
    plan.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    plan.add_argument("--predecessor-root", type=Path, default=DEFAULT_PREDECESSOR)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    submit = sub.add_parser("submit", help="submit the frozen DAG with afterok dependencies")
    submit.add_argument("--manifest", type=Path, required=True)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument(
        "--retry-failed",
        action="store_true",
        help="reconcile terminal jobs, preserve valid task receipts, and rebuild stale dependencies",
    )
    execute = sub.add_parser("execute", help=argparse.SUPPRESS)
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--job-id", required=True)
    execute.add_argument("--task", type=int)
    report = sub.add_parser("status", help="verify receipts and show migration progress")
    report.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        manifest = build_manifest(
            runtime_path=_abs(args.runtime),
            predecessor_root=_abs(args.predecessor_root),
            output_root=_abs(args.output_root),
        )
        path = write_manifest(manifest)
        result: Any = {
            "manifest": str(path),
            "manifest_sha256": manifest["manifest_sha256"],
            "jobs": len(manifest["jobs"]),
            "submit_slots": manifest["submit_slots"],
            "cpu_only": True,
        }
    elif args.command == "submit":
        result = submit_workflow(
            args.manifest,
            dry_run=bool(args.dry_run),
            retry_failed=bool(args.retry_failed),
        )
    elif args.command == "execute":
        execute_job(args.manifest, args.job_id, task=args.task)
        result = {"status": "complete", "job_id": args.job_id, "task": args.task}
    else:
        result = status(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "K12MigrationWorkflowError",
    "build_manifest",
    "execute_job",
    "status",
    "submit_workflow",
    "validate_manifest",
    "write_manifest",
]
