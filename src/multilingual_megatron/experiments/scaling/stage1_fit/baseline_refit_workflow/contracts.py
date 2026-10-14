"""Immutable contracts and the static Slurm DAG for the baseline refit."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data import canonical_json_sha256, sha256_file
from ..execution import read_json
from ..fit import outer_parameter_names, start_count


PROFILE_SCHEMA = "stage1_fit_baseline_refit_profile/v1"
WORKFLOW_SCHEMA = "stage1_fit_baseline_refit_workflow/v1"
INDEX_SCHEMA = "stage1_fit_baseline_refit_index/v1"
GATE_SCHEMA = "stage1_fit_baseline_refit_gate/v1"
REPRODUCTION_SCHEMA = "stage1_fit_baseline_refit_reproduction/v1"
REPORT_SCHEMA = "stage1_fit_primary_baseline_comparison/v1"
FINAL_SCHEMA = "stage1_fit_baseline_refit_final/v1"
STATE_SCHEMA = "stage1_fit_baseline_refit_submission/v1"

FOLDS = ("largest_N_act", "E32", "K200_common_languages")
MODELS = ("C0", "M0", "J0", "M1", "M2")
BASELINES = ("C0", "M0", "J0")
CANDIDATES = ("M0", "M1", "M2")
EXPECTED_SELECTED_MODEL = "M2"
BOOTSTRAP_REPLICATES = 10_000

WORKFLOW_SOURCE_FILES = (
    "baseline_refit.py",
    "baseline_refit_workflow/__init__.py",
    "baseline_refit_workflow/contracts.py",
    "baseline_refit_workflow/planning.py",
    "baseline_refit_workflow/state.py",
    "baseline_refit_workflow/scheduler.py",
    "baseline_refit_workflow/gates.py",
    "baseline_refit_workflow/reporting.py",
)


class BaselineRefitError(RuntimeError):
    """Raised when the one-submit baseline refit loses scientific identity."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[6]


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = canonical_json_sha256(payload)
    return result


def verify_seal(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    digest = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not isinstance(digest, str) or digest != canonical_json_sha256(unsigned):
        raise BaselineRefitError(f"invalid {label} {field}")
    return digest


def _input_binding(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise BaselineRefitError(f"profile input {name} must contain path and sha256")
    path = resolve_path(value["path"])
    expected = str(value["sha256"])
    if not path.is_file() or sha256_file(path) != expected:
        raise BaselineRefitError(f"profile input {name} is missing or changed: {path}")
    return {"path": str(path), "file_sha256": expected}


def load_profile(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineRefitError(f"cannot read profile {path}: {error}") from error
    required = {
        "schema_version", "output_root", "mode", "python_executable",
        "chunk_size", "array_concurrency", "blas_threads", "memory",
        "time_limit", "m2_time_limit", "gate_time_limit", "partition",
        "account", "reservation", "inputs",
    }
    if not isinstance(raw, dict) or set(raw) != required \
            or raw.get("schema_version") != PROFILE_SCHEMA:
        raise BaselineRefitError("baseline-refit profile keys or schema changed")
    if raw.get("mode") != "slurm_array" or int(raw["chunk_size"]) != 12 \
            or int(raw["array_concurrency"]) < 1 or int(raw["blas_threads"]) != 1:
        raise BaselineRefitError("baseline-refit execution contract changed")
    for key in ("memory", "time_limit", "m2_time_limit", "gate_time_limit"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise BaselineRefitError(f"profile {key} is empty")
    python = resolve_path(raw["python_executable"])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise BaselineRefitError(f"profile Python is not executable: {python}")
    inputs = raw.get("inputs")
    expected_inputs = {"input_receipt", "holdouts", "historical_candidate_report"}
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise BaselineRefitError("baseline-refit profile input inventory changed")
    bound_inputs = {name: _input_binding(value, name=name) for name, value in inputs.items()}
    output_root = resolve_path(raw["output_root"])
    historical_root = Path(bound_inputs["input_receipt"]["path"]).parents[1]
    if output_root == historical_root or historical_root in output_root.parents:
        raise BaselineRefitError("new baseline refit must not write below its immutable source root")
    return {
        **raw,
        "output_root": str(output_root),
        "python_executable": str(python),
        "inputs": bound_inputs,
        "profile_path": str(path.resolve()),
        "profile_file_sha256": sha256_file(path),
    }


def control_root(workflow: Mapping[str, Any]) -> Path:
    return Path(str(workflow["output_root"])) / "control" / "baseline_refit"


def workflow_path(output_root: Path) -> Path:
    return output_root / "control" / "baseline_refit" / "workflow_manifest.json"


def index_path(workflow: Mapping[str, Any], name: str) -> Path:
    return control_root(workflow) / "indices" / f"{name}.json"


def gate_path(workflow: Mapping[str, Any], gate_id: str) -> Path:
    return control_root(workflow) / "gates" / f"{gate_id.replace('.', '-')}.json"


def source_binding() -> dict[str, Any]:
    root = source_root()
    files: dict[str, str] = {}
    for relative in WORKFLOW_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise BaselineRefitError(f"workflow source is missing: {path}")
        files[relative] = sha256_file(path)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root(), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaselineRefitError(f"cannot resolve repository commit: {error}") from error
    return {"git_commit": commit, "files": files}


def uncommitted_workflow_sources(binding: Mapping[str, Any]) -> list[str]:
    commit = str(binding.get("git_commit", ""))
    files = binding.get("files")
    if not commit or not isinstance(files, Mapping):
        raise BaselineRefitError("workflow source binding is malformed")
    changed: list[str] = []
    prefix = "src/multilingual_megatron/experiments/scaling/stage1_fit"
    for relative, expected in sorted(files.items()):
        repository_path = f"{prefix}/{relative}"
        try:
            committed = subprocess.run(
                ["git", "show", f"{commit}:{repository_path}"],
                cwd=repository_root(), check=True, capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            changed.append(str(relative))
            continue
        current = (source_root() / str(relative)).read_bytes()
        if committed != current or sha256_file(source_root() / str(relative)) != expected:
            changed.append(str(relative))
    return changed


def require_committed_workflow_sources(binding: Mapping[str, Any]) -> None:
    changed = uncommitted_workflow_sources(binding)
    if changed:
        raise BaselineRefitError(
            "submission requires committed baseline-refit source files: " + ", ".join(changed)
        )


def _array_job(*, job_id: str, stage: str, model: str, fold: str,
        index: str, dependencies: Sequence[str], dependency_mode: str,
        profile: Mapping[str, Any]) -> dict[str, Any]:
    count = math.ceil(start_count(model) / int(profile["chunk_size"]))
    return {
        "job_id": job_id,
        "kind": "array",
        "stage": stage,
        "model": model,
        "fold": fold,
        "manifest_index": index,
        "task_count": count,
        "array": f"0-{count - 1}%{int(profile['array_concurrency'])}",
        "dependencies": list(dependencies),
        "dependency_mode": dependency_mode,
        "memory": str(profile["memory"]),
        "time_limit": str(profile["m2_time_limit"] if model == "M2" else profile["time_limit"]),
        "cpus_per_task": int(profile["blas_threads"]),
    }


def _gate_job(*, job_id: str, dependencies: Sequence[str], profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "kind": "gate",
        "stage": job_id.removeprefix("gate."),
        "model": None,
        "fold": None,
        "manifest_index": None,
        "task_count": None,
        "array": None,
        "dependencies": list(dependencies),
        "dependency_mode": "afterany",
        "memory": str(profile["memory"]),
        "time_limit": str(profile["gate_time_limit"]),
        "cpus_per_task": 1,
    }


def build_jobs(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    scale_ids = [f"grid.m0_ols.{fold}" for fold in FOLDS]
    for fold, job_id in zip(FOLDS, scale_ids, strict=True):
        jobs.append(_array_job(job_id=job_id, stage="m0_ols", model="M0", fold=fold,
            index="m0_ols", dependencies=(), dependency_mode="afterok", profile=profile))
    jobs.append(_gate_job(job_id="gate.scale", dependencies=scale_ids, profile=profile))

    robust_ids: dict[str, list[str]] = {model: [] for model in BASELINES}
    for model in BASELINES:
        for fold in FOLDS:
            job_id = f"grid.robust.{model}.{fold}"
            robust_ids[model].append(job_id)
            jobs.append(_array_job(job_id=job_id, stage="robust", model=model, fold=fold,
                index="robust", dependencies=("gate.scale",), dependency_mode="afterok", profile=profile))
    jobs.append(_gate_job(job_id="gate.m0", dependencies=robust_ids["M0"], profile=profile))
    jobs.append(_gate_job(job_id="gate.baselines",
        dependencies=(*robust_ids["C0"], *robust_ids["J0"]), profile=profile))

    m1_ids = [f"grid.m1.{fold}" for fold in FOLDS]
    for fold, job_id in zip(FOLDS, m1_ids, strict=True):
        jobs.append(_array_job(job_id=job_id, stage="m1", model="M1", fold=fold,
            index="m1", dependencies=("gate.m0",), dependency_mode="afterok", profile=profile))
    jobs.append(_gate_job(job_id="gate.m1", dependencies=m1_ids, profile=profile))

    m2_ids = [f"grid.m2.{fold}" for fold in FOLDS]
    for fold, job_id in zip(FOLDS, m2_ids, strict=True):
        jobs.append(_array_job(job_id=job_id, stage="m2", model="M2", fold=fold,
            index="m2", dependencies=("gate.m1",), dependency_mode="afterok", profile=profile))
    jobs.append(_gate_job(job_id="gate.final",
        dependencies=(*m2_ids, "gate.baselines"), profile=profile))
    return jobs


def build_workflow(profile: Mapping[str, Any], initial_manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    jobs = build_jobs(profile)
    expected_initial = {f"grid.m0_ols.{fold}" for fold in FOLDS}
    if set(initial_manifests) != expected_initial:
        raise BaselineRefitError("initial M0-OLS manifest inventory changed")
    initial = {job_id: {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "phase": str(manifest["phase"]),
    } for job_id, manifest in sorted(initial_manifests.items())}
    registered_starts = sum(start_count(model) * len(FOLDS) for model in MODELS) + start_count("M0") * len(FOLDS)
    unsigned = {
        "schema_version": WORKFLOW_SCHEMA,
        "output_root": profile["output_root"],
        "profile": {key: profile[key] for key in (
            "output_root", "python_executable", "chunk_size", "array_concurrency",
            "blas_threads", "memory", "time_limit", "m2_time_limit",
            "gate_time_limit", "partition", "account", "reservation",
        )},
        "profile_binding": {"path": profile["profile_path"], "file_sha256": profile["profile_file_sha256"]},
        "inputs": dict(profile["inputs"]),
        "models": list(MODELS),
        "baselines": list(BASELINES),
        "candidates": list(CANDIDATES),
        "folds": list(FOLDS),
        "expected_selected_model": EXPECTED_SELECTED_MODEL,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "registered_optimizer_starts": registered_starts,
        "array_tasks": sum(int(job["task_count"] or 0) for job in jobs),
        "initial_manifests": initial,
        "jobs": jobs,
        "workflow_source": source_binding(),
        "model_outer_parameter_counts": {model: len(outer_parameter_names(model)) for model in MODELS},
        "claim_rule": {
            "fold_rule": "M2 RMSE is strictly below C0, M0, and J0 on every primary fold",
            "paired_rule": "p10 of each shared-trajectory bootstrap difference baseline_minus_M2 is above zero",
        },
    }
    return seal(unsigned, "workflow_sha256")


def validate_workflow(workflow: Mapping[str, Any], *, verify_live_files: bool = True) -> str:
    digest = verify_seal(workflow, "workflow_sha256", label="workflow")
    if workflow.get("schema_version") != WORKFLOW_SCHEMA \
            or tuple(workflow.get("models", ())) != MODELS \
            or tuple(workflow.get("baselines", ())) != BASELINES \
            or tuple(workflow.get("candidates", ())) != CANDIDATES \
            or tuple(workflow.get("folds", ())) != FOLDS \
            or workflow.get("expected_selected_model") != EXPECTED_SELECTED_MODEL \
            or workflow.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise BaselineRefitError("workflow scientific contract changed")
    profile = workflow.get("profile")
    if not isinstance(profile, Mapping) or workflow.get("output_root") != profile.get("output_root") \
            or workflow.get("jobs") != build_jobs(profile):
        raise BaselineRefitError("workflow Slurm DAG changed")
    if int(workflow.get("registered_optimizer_starts", -1)) != 199_530 \
            or int(workflow.get("array_tasks", -1)) != 16_632 \
            or len(workflow["jobs"]) != 23:
        raise BaselineRefitError("workflow workload accounting changed")
    if verify_live_files:
        bindings = [workflow.get("profile_binding"), *workflow.get("inputs", {}).values()]
        for binding in bindings:
            if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256"}:
                raise BaselineRefitError("workflow file binding is malformed")
            path = Path(str(binding["path"]))
            if not path.is_file() or sha256_file(path) != binding["file_sha256"]:
                raise BaselineRefitError(f"workflow-bound file changed: {path}")
        source = workflow.get("workflow_source")
        if not isinstance(source, Mapping) or not isinstance(source.get("files"), Mapping):
            raise BaselineRefitError("workflow source binding is malformed")
        for relative, expected in source["files"].items():
            path = source_root() / str(relative)
            if not path.is_file() or sha256_file(path) != expected:
                raise BaselineRefitError(f"workflow implementation source changed: {path}")
    return digest


def load_workflow(path: Path, *, verify_live_files: bool = True) -> dict[str, Any]:
    workflow = read_json(path)
    validate_workflow(workflow, verify_live_files=verify_live_files)
    return workflow


def job_by_id(workflow: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    rows = [row for row in workflow["jobs"] if row.get("job_id") == job_id]
    if len(rows) != 1:
        raise BaselineRefitError(f"unknown workflow job: {job_id}")
    return rows[0]
