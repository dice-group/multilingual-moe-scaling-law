"""One immutable manifest, atomic chunk worker, strict reducer, and Slurm array."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .data import CONTRACT_VERSION, canonical_json_bytes, canonical_json_sha256, sha256_file
from .fit import (
    D16_ADAPTIVE_PROTOCOL,
    EXPLORATORY_COMPOSITION_MODELS,
    EXPLORATORY_EXPOSURE_MODELS,
    M5M6Regularization,
    OUTER_FTOL, OUTER_GTOL, OUTER_MAX_ITERATIONS, START_GRID_ORDERING_VERSION,
    OuterBound, ProjectedProblem, SharedParameters, adaptive_huber_scale,
    StartPoint, d16_adaptive_tiers, data_column, direct_joint_check, initial_bounds, predict, run_projected_start,
    start_at, start_count, trajectory_weights,
)

MANIFEST_SCHEMA = "stage1_fit_work/v2"
CHUNK_SCHEMA = "stage1_fit_chunk/v2"
RESULT_SCHEMA = "stage1_fit_reduced/v2"
RECEIPT_SCHEMA = "stage1_fit_complete/v2"
Runner = Callable[[Mapping[str, Any], int], Mapping[str, Any]]

class ExecutionError(RuntimeError):
    pass

def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ExecutionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ExecutionError(f"non-finite JSON: {value}")))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionError(f"{path} is not a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ExecutionError(f"{path} is not canonical JSON")
    return value

def write_json_once(path: Path, payload: Mapping[str, Any]) -> bool:
    data = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise ExecutionError(f"refusing to overwrite immutable artifact: {path}")
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ExecutionError(f"immutable artifact collision: {path}")
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)

def write_text_once(path: Path, text: str) -> bool:
    data=text.encode("utf-8"); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()==data: return False
        raise ExecutionError(f"refusing to overwrite immutable artifact: {path}")
    descriptor,name=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.",suffix=".tmp"); temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        try: os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=data: raise ExecutionError(f"immutable artifact collision: {path}")
            return False
        return True
    finally: temporary.unlink(missing_ok=True)

def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload); result[field] = canonical_json_sha256(payload); return result

def _verify_seal(payload: Mapping[str, Any], field: str) -> str:
    digest = payload.get(field)
    unsigned = dict(payload); unsigned.pop(field, None)
    if not isinstance(digest, str) or digest != canonical_json_sha256(unsigned):
        raise ExecutionError(f"invalid {field}")
    return digest

SOURCE_FILES = (
    "data.py", "fit.py", "selection.py", "execution.py", "campaign.py",
    "amendment.py", "report.py", "cli.py", "interaction.py",
    "interaction_nested.py", "interaction_ablation.py",
)
OPTIONAL_HISTORICAL_SOURCE_FILES = {
    "amendment.py", "interaction.py", "interaction_nested.py",
    "interaction_ablation.py",
}


def source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in SOURCE_FILES:
        path = root / name
        if path.exists():
            content = path.read_bytes(); digest.update(name.encode()); digest.update(len(content).to_bytes(8,"big")); digest.update(content)
    return digest.hexdigest()


def source_sha256_at_commit(commit: str) -> str:
    """Recompute the recorded fitter source hash from an immutable Git commit."""
    if not isinstance(commit, str) or not commit:
        raise ExecutionError("recorded provenance lacks a Git commit")
    repository = Path(__file__).resolve().parents[5]
    digest = hashlib.sha256()
    prefix = "src/multilingual_megatron/experiments/scaling/stage1_fit"
    for name in SOURCE_FILES:
        try:
            content = subprocess.run(["git", "show", f"{commit}:{prefix}/{name}"], cwd=repository,
                check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError):
            if name in OPTIONAL_HISTORICAL_SOURCE_FILES:
                continue
            raise ExecutionError(f"cannot recover recorded fitter source {name} at {commit}")
        digest.update(name.encode()); digest.update(len(content).to_bytes(8, "big")); digest.update(content)
    return digest.hexdigest()


def validate_recorded_provenance(provenance: object) -> None:
    """Validate the fitter source binding without treating unrelated dirt as fatal."""
    if not isinstance(provenance, Mapping):
        raise ExecutionError("recorded provenance is malformed")
    commit = provenance.get("git_commit"); source = provenance.get("source_sha256")
    if not isinstance(source, str):
        raise ExecutionError("historical definitive work must bind a source hash")
    if source_sha256_at_commit(str(commit)) != source:
        raise ExecutionError("recorded source hash does not match its Git commit")

@lru_cache(maxsize=1)
def _repository_provenance() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[5]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository,
            check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionError(f"cannot record repository provenance: {error}") from error
    return (commit, dirty)

def runtime_provenance() -> dict[str, Any]:
    """Freeze the numerical/library and repository state used by each worker."""
    import pandas as pd
    import pyarrow
    import scipy
    commit, dirty = _repository_provenance()
    return {"source_sha256": source_sha256(), "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
        "platform": platform.platform(), "float_dtype": "float64", "git_commit": commit, "git_dirty": dirty}

def require_clean_definitive_provenance(provenance: object, *, review_only: bool) -> None:
    """Require committed fitter bytes; record but never gate on unrelated dirt."""
    if review_only:
        return
    try:
        validate_recorded_provenance(provenance)
    except ExecutionError as error:
        raise ExecutionError(
            "definitive work requires fitter source bytes bound to the recorded commit"
        ) from error

def _current_provenance_matches(recorded: Mapping[str, Any]) -> bool:
    """Compare runtime identity while allowing unrelated Git state to advance."""
    current = runtime_provenance()
    required = set(current) - {"git_commit", "git_dirty"}
    return all(recorded.get(field) == current.get(field) for field in required)

def _regularization_for_model(model: str,
    value: object) -> M5M6Regularization | None:
    if model not in EXPLORATORY_COMPOSITION_MODELS:
        if value is not None:
            raise ExecutionError(f"{model} does not accept M5/M6 regularization")
        return None
    if not isinstance(value, Mapping):
        raise ExecutionError(f"{model} requires manifest-bound M5/M6 regularization")
    try:
        return M5M6Regularization.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise ExecutionError(f"invalid {model} regularization: {error}") from error

def build_manifest(*, output_root: Path, fit_rows_path: Path, fit_rows_content_sha256: str,
    phase: str, model: str, fold_id: str, objective: str, chunk_size: int,
    delta: float | None = None, bounds: Sequence[OuterBound] | None = None,
    bound_version: int = 0, replicate_id: int | None = None,
    parent_result: Path | None = None, start_total: int | None = None,
    parent_results: Mapping[str, Path] | None = None,
    selection_report: Path | None = None,
    membership: Mapping[str, Any] | None = None,
    regularization: Mapping[str, Any] | None = None,
    explicit_starts: Sequence[StartPoint] | None = None,
    review_only: bool = False, action: str = "grid",
    blas_threads: int = 1, partition: str | None = None, account: str | None = None,
    reservation: str | None = None, memory: str = "2G", time_limit: str = "00:30:00",
    concurrency: int = 64) -> dict[str, Any]:
    if objective not in {"ols", "huber"} or (objective == "huber") != (delta is not None):
        raise ExecutionError("objective and delta disagree")
    active_regularization = _regularization_for_model(model, regularization)
    if chunk_size < 1 or blas_threads < 1 or concurrency < 1:
        raise ExecutionError("execution integers must be positive")
    if explicit_starts is not None and start_total is not None:
        raise ExecutionError("explicit starts and start_total are mutually exclusive")
    mode = "explicit" if explicit_starts is not None else "grid"
    total = len(explicit_starts) if explicit_starts is not None else (start_count(model) if start_total is None else int(start_total))
    if not 0 < total <= start_count(model) and mode == "grid":
        raise ExecutionError("invalid start count")
    if mode == "explicit" and total < 1:
        raise ExecutionError("explicit start list is empty")
    if mode == "grid" and total != start_count(model) and not review_only:
        raise ExecutionError("definitive grids must use every registered start")
    if mode == "explicit" and not review_only and action not in {"warm", "basin", "local", "direct", "d16_adaptive"}:
        raise ExecutionError("definitive explicit starts are reserved for derived numerical checks")
    if membership is None and not review_only:
        raise ExecutionError("definitive manifests require frozen membership")
    membership_kind = "" if membership is None else str(membership.get("kind", ""))
    is_d16 = membership_kind.startswith("d16:")
    if not review_only and is_d16:
        tiers = d16_adaptive_tiers(model)
        if action == "d16_adaptive" and (mode != "explicit" or total not in tiers[:-1]):
            raise ExecutionError("D16 adaptive manifest has an unregistered start tier")
        if action == "d16_full_grid" and (mode != "grid" or total != tiers[-1]):
            raise ExecutionError("D16 full fallback must use the complete registered grid")
        if action in {"warm", "basin", "local", "direct"} and mode != "explicit":
            raise ExecutionError("derived D16 numerical checks require explicit starts")
        if action not in {"d16_adaptive", "d16_full_grid", "warm", "basin", "local", "direct"}:
            raise ExecutionError("definitive D16 refits must use the adaptive protocol")
    if not is_d16 and action in {"d16_adaptive", "d16_full_grid"}:
        raise ExecutionError("D16 actions require a frozen D16 membership")
    active = tuple(initial_bounds(model) if bounds is None else bounds)
    tasks = [{"task_index": i, "task_id": f"task-{i:06d}", "start_begin": begin,
              "start_end": min(begin + chunk_size, total)}
             for i, begin in enumerate(range(0, total, chunk_size))]
    all_parents = dict(parent_results or {})
    if parent_result is not None:
        all_parents.setdefault("parent", parent_result)
    if is_d16 and not review_only:
        if "full_data" not in all_parents:
            raise ExecutionError("D16 refit lacks its selected all-data anchor")
        if action == "d16_full_grid" and "adaptive_parent" not in all_parents:
            raise ExecutionError("D16 full-grid fallback lacks its adaptive parent")
        if action in {"warm", "basin", "local", "direct"} and "grid" not in all_parents:
            raise ExecutionError("derived D16 check lacks its fitted-result parent")
    parents = []
    for kind, path in sorted(all_parents.items()):
        payload = read_json(Path(path))
        parents.append({"kind": kind, "path": str(Path(path).resolve()),
                        "file_sha256": sha256_file(Path(path)),
                        "result_sha256": payload.get("result_sha256")})
    selection = None
    if selection_report is not None:
        report = read_json(selection_report)
        selection = {"path": str(selection_report.resolve()), "file_sha256": sha256_file(selection_report),
            "report_sha256": report.get("report_sha256")}
    starts = None
    if explicit_starts is not None:
        starts = [{"start_id": item.start_id, "values": np.asarray(item.values, dtype=np.float64).tolist(),
                   "kind": item.kind} for item in explicit_starts]
    provenance = runtime_provenance()
    require_clean_definitive_provenance(provenance, review_only=review_only)
    fit_payload = {"model": model, "fold_id": fold_id, "replicate_id": replicate_id,
        "objective": objective, "delta": delta, "bound_version": bound_version,
        "bounds": [{"name": item.name, "lower": item.lower, "upper": item.upper,
                    "kind": item.kind, "expansions": item.expansions} for item in active],
        "grid_version": START_GRID_ORDERING_VERSION, "start_count": total,
        "start_mode": mode, "explicit_starts": starts, "action": action,
        "d16_adaptive_protocol": D16_ADAPTIVE_PROTOCOL if is_d16 and not review_only else None,
        "outer_ftol": OUTER_FTOL, "outer_gtol": OUTER_GTOL,
        "outer_max_iterations": OUTER_MAX_ITERATIONS}
    if active_regularization is not None:
        fit_payload["regularization"] = active_regularization.as_dict()
    unsigned = {
        "schema_version": MANIFEST_SCHEMA, "contract_version": CONTRACT_VERSION,
        "phase": phase,
        "input": {"fit_rows_path": str(fit_rows_path.resolve()), "fit_rows_file_sha256": sha256_file(fit_rows_path),
                  "fit_rows_content_sha256": fit_rows_content_sha256},
        "fit": fit_payload,
        "execution": {"output_root": str(output_root.resolve()), "chunk_size": chunk_size,
                      "blas_threads": blas_threads, "partition": partition, "account": account,
                      "reservation": reservation, "memory": memory, "time_limit": time_limit,
                      "concurrency": concurrency},
        "parents": parents, "selection_report": selection, "membership": None if membership is None else dict(membership),
        "review_only": bool(review_only), "tasks": tasks,
        "provenance": provenance,
    }
    return _seal(unsigned, "manifest_sha256")

def validate_manifest(manifest: Mapping[str, Any], *, verify_files: bool = True,
    require_current_provenance: bool = True) -> str:
    digest = _verify_seal(manifest, "manifest_sha256")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("contract_version") != CONTRACT_VERSION:
        raise ExecutionError("manifest schema or contract changed")
    fit = manifest.get("fit"); execution = manifest.get("execution"); tasks = manifest.get("tasks"); input_ = manifest.get("input")
    if not all(isinstance(value, Mapping) for value in (fit, execution, input_)) or not isinstance(tasks, list):
        raise ExecutionError("malformed manifest")
    model = str(fit.get("model"))
    if model in EXPLORATORY_COMPOSITION_MODELS:
        _regularization_for_model(model, fit.get("regularization"))
    elif "regularization" in fit:
        raise ExecutionError(f"{model} does not accept M5/M6 regularization")
    require_clean_definitive_provenance(manifest.get("provenance"),
        review_only=bool(manifest.get("review_only")))
    if require_current_provenance:
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping) or not _current_provenance_matches(provenance):
            raise ExecutionError("numerical environment or repository provenance changed")
    else:
        validate_recorded_provenance(manifest.get("provenance"))
    if fit["grid_version"] != START_GRID_ORDERING_VERSION:
        raise ExecutionError("grid contract changed")
    active_bounds = _bounds(manifest)
    if model == "M6":
        locked = {"eta_family": (0.0, 1.0), "eta_script": (0.0, 1.0),
            "gamma_other": (0.0, 0.5)}
        if (tuple(bound.name for bound in active_bounds)
                != tuple(bound.name for bound in initial_bounds("M6"))):
            raise ExecutionError("M6 bounds do not match its coordinate order")
        for bound in active_bounds:
            if bound.name in locked and ((bound.lower, bound.upper) != locked[bound.name]
                    or bound.expansions != 0):
                raise ExecutionError(f"M6 bound changed for {bound.name}")
    mode = fit.get("start_mode")
    if mode == "grid":
        if not 0 < fit["start_count"] <= start_count(fit["model"]):
            raise ExecutionError("grid start count is invalid")
        if fit["start_count"] != start_count(fit["model"]) and not manifest.get("review_only"):
            raise ExecutionError("partial grid is not definitive")
        if fit.get("explicit_starts") is not None:
            raise ExecutionError("grid manifest has explicit starts")
    elif mode == "explicit":
        starts = fit.get("explicit_starts")
        if not isinstance(starts, list) or len(starts) != fit["start_count"]:
            raise ExecutionError("explicit-start coverage changed")
        names = set()
        for item in starts:
            values = np.asarray(item.get("values"), dtype=np.float64)
            if not item.get("start_id") or item.get("start_id") in names or values.shape != (len(_bounds(manifest)),) or not np.isfinite(values).all():
                raise ExecutionError("explicit start is malformed")
            names.add(item["start_id"])
    else:
        raise ExecutionError("unknown start mode")
    membership_kind = str(manifest.get("membership", {}).get("kind", "")) if isinstance(manifest.get("membership"), Mapping) else ""
    is_d16 = membership_kind.startswith("d16:")
    if not manifest.get("review_only") and is_d16:
        tiers = d16_adaptive_tiers(fit["model"])
        if fit.get("d16_adaptive_protocol") != D16_ADAPTIVE_PROTOCOL:
            raise ExecutionError("D16 adaptive protocol changed")
        if fit.get("action") == "d16_adaptive" and (mode != "explicit" or fit["start_count"] not in tiers[:-1]):
            raise ExecutionError("D16 adaptive tier changed")
        if fit.get("action") == "d16_full_grid" and (mode != "grid" or fit["start_count"] != tiers[-1]):
            raise ExecutionError("D16 full-grid fallback changed")
        if fit.get("action") in {"warm", "basin", "local", "direct"} and mode != "explicit":
            raise ExecutionError("derived D16 numerical-check mode changed")
        if fit.get("action") not in {"d16_adaptive", "d16_full_grid", "warm", "basin", "local", "direct"}:
            raise ExecutionError("definitive D16 action changed")
        parent_kinds = {item.get("kind") for item in manifest.get("parents", []) if isinstance(item, Mapping)}
        if "full_data" not in parent_kinds or (fit.get("action") == "d16_full_grid" and "adaptive_parent" not in parent_kinds):
            raise ExecutionError("D16 adaptive lineage is incomplete")
        if fit.get("action") in {"warm", "basin", "local", "direct"} and "grid" not in parent_kinds:
            raise ExecutionError("derived D16 lineage is incomplete")
    elif fit.get("d16_adaptive_protocol") is not None or fit.get("action") in {"d16_adaptive", "d16_full_grid"}:
        raise ExecutionError("non-D16 manifest carries a D16 protocol")
    expected = 0
    for index, task in enumerate(tasks):
        if task != {"task_index": index, "task_id": f"task-{index:06d}", "start_begin": expected,
                    "start_end": min(expected + execution["chunk_size"], fit["start_count"])}:
            raise ExecutionError("tasks are not canonical contiguous chunks")
        expected = task["start_end"]
    if expected != fit["start_count"]:
        raise ExecutionError("task coverage is incomplete")
    if verify_files:
        path = Path(input_["fit_rows_path"])
        if sha256_file(path) != input_["fit_rows_file_sha256"]:
            raise ExecutionError("fit-row file hash changed")
        for parent in manifest.get("parents", []):
            parent_path = Path(parent["path"])
            if sha256_file(parent_path) != parent["file_sha256"]:
                raise ExecutionError("parent result file hash changed")
            if read_json(parent_path).get("result_sha256") != parent["result_sha256"]:
                raise ExecutionError("parent result embedded hash changed")
        selection = manifest.get("selection_report")
        if selection is not None:
            if not isinstance(selection, Mapping) or set(selection) != {"path", "file_sha256", "report_sha256"}:
                raise ExecutionError("selection report binding is malformed")
            selection_path = Path(str(selection["path"]))
            if sha256_file(selection_path) != selection["file_sha256"] or read_json(selection_path).get("report_sha256") != selection["report_sha256"]:
                raise ExecutionError("selection report changed")
        _validate_membership(manifest.get("membership"), review_only=bool(manifest.get("review_only")))
    return digest

def _validate_membership(membership: object, *, review_only: bool) -> None:
    if membership is None:
        if not review_only:
            raise ExecutionError("definitive manifest lacks membership")
        return
    if not isinstance(membership, Mapping):
        raise ExecutionError("membership is malformed")
    required = {"kind", "path", "file_sha256", "embedded_sha256", "membership_sha256", "train_trajectory_ids", "test_trajectory_ids", "score_languages", "train_readout_ids", "test_readout_ids"}
    if set(membership) != required:
        raise ExecutionError("membership keys changed")
    path = Path(str(membership["path"]))
    if sha256_file(path) != membership["file_sha256"]:
        raise ExecutionError("membership file hash changed")
    payload = read_json(path)
    payload_digest = _verify_seal(payload, "manifest_sha256")
    if payload_digest != membership["embedded_sha256"]:
        raise ExecutionError("membership embedded hash changed")
    unsigned = {key: membership[key] for key in sorted(required - {"path", "file_sha256", "embedded_sha256", "membership_sha256"})}
    if canonical_json_sha256(unsigned) != membership["membership_sha256"]:
        raise ExecutionError("membership content changed")
    for key in ("train_trajectory_ids", "test_trajectory_ids", "score_languages", "train_readout_ids", "test_readout_ids"):
        value = membership[key]
        if not isinstance(value, list) or value != sorted(set(str(item) for item in value)):
            raise ExecutionError(f"membership {key} is not canonical")

def bind_membership(path: Path, *, kind: str, membership_id: str | int | None = None) -> dict[str, Any]:
    """Freeze one declared trajectory, readout, all-data, or D16 membership."""
    payload = read_json(path)
    embedded = _verify_seal(payload, "manifest_sha256")
    train: list[str]; test: list[str] = []; score: list[str] = []; train_reads: list[str] = []; test_reads: list[str] = []
    if kind in {"primary", "random", "lowest"}:
        groups = payload.get("primary_folds", []) if kind == "primary" else [payload.get("random_holdout") if kind == "random" else payload.get("lowest_loss_extrapolation")]
        matches = [item for item in groups if isinstance(item, Mapping) and item.get("fold_id") == membership_id]
        if len(matches) != 1:
            raise ExecutionError("requested trajectory membership is absent")
        item = matches[0]; train = list(item["train_trajectory_ids"]); test = list(item["test_trajectory_ids"]); score = list(item["score_languages"])
    elif kind == "terminal":
        item = payload.get("terminal_continuation")
        if not isinstance(item, Mapping):
            raise ExecutionError("terminal membership is absent")
        train = list(item["trajectory_ids"]); test = list(item["trajectory_ids"])
        train_reads = list(item["train_readout_ids"]); test_reads = list(item["test_readout_ids"])
    elif kind == "compute":
        item = payload.get(str(membership_id))
        if not isinstance(item, Mapping) or item.get("fold_id") != membership_id:
            raise ExecutionError("logical-compute membership is absent")
        train = list(item["train_trajectory_ids"]); test = list(item["test_trajectory_ids"])
        score = list(item["score_languages"])
        train_reads = list(item["train_readout_ids"]); test_reads = list(item["test_readout_ids"])
    elif kind == "all":
        groups = payload.get("primary_folds", [])
        if not groups:
            raise ExecutionError("all-data membership lacks holdout population")
        train = sorted(set(groups[0]["train_trajectory_ids"]) | set(groups[0]["test_trajectory_ids"]))
        score = sorted({str(item) for group in groups for item in group.get("score_languages", [])})
    elif kind == "d16":
        matches = [item for item in payload.get("subsets", []) if item.get("replicate_id") == membership_id]
        if len(matches) != 1:
            raise ExecutionError("requested D16 subset is absent")
        train = list(matches[0]["trajectory_ids"])
    else:
        raise ExecutionError(f"unknown membership kind: {kind}")
    value = {"kind": kind if membership_id is None else f"{kind}:{membership_id}", "train_trajectory_ids": sorted(set(train)),
             "test_trajectory_ids": sorted(set(test)), "score_languages": sorted(set(score)),
             "train_readout_ids": sorted(set(train_reads)), "test_readout_ids": sorted(set(test_reads))}
    value["membership_sha256"] = canonical_json_sha256(value)
    value.update({"path": str(path.resolve()), "file_sha256": sha256_file(path), "embedded_sha256": embedded})
    return value

def _require_open_root(root: Path) -> None:
    if (root / "receipts" / "finalization.json").exists():
        raise ExecutionError(f"refusing work beneath finalized output root: {root}")

def write_manifest(manifest: Mapping[str, Any]) -> Path:
    digest = validate_manifest(manifest)
    root = Path(manifest["execution"]["output_root"])
    _require_open_root(root)
    path = root / "manifests" / f"{manifest['phase']}-{digest[:20]}.json"
    write_json_once(path, manifest); return path

def load_manifest(path: Path, *, verify_files: bool = True,
    require_current_provenance: bool = True) -> dict[str, Any]:
    manifest = read_json(path)
    validate_manifest(manifest, verify_files=verify_files,
        require_current_provenance=require_current_provenance)
    return manifest

def _bounds(manifest: Mapping[str, Any]) -> tuple[OuterBound, ...]:
    return tuple(OuterBound(**item) for item in manifest["fit"]["bounds"])

def _point(manifest: Mapping[str, Any], start_index: int) -> StartPoint:
    fit = manifest["fit"]
    if fit["start_mode"] == "grid":
        return start_at(fit["model"], start_index)
    item = fit["explicit_starts"][start_index]
    return StartPoint(str(item["start_id"]), start_index, np.asarray(item["values"], dtype=np.float64), str(item["kind"]))

def _start_id(manifest: Mapping[str, Any], start_index: int) -> str:
    return _point(manifest, start_index).start_id

def load_problem(manifest: Mapping[str, Any]) -> ProjectedProblem:
    import pandas as pd
    frame = pd.read_parquet(manifest["input"]["fit_rows_path"])
    membership = manifest.get("membership")
    _validate_membership(membership, review_only=bool(manifest.get("review_only")))
    if membership is None:
        raise ExecutionError("default fitting requires frozen membership")
    frame = frame[frame["trajectory_id"].isin(membership["train_trajectory_ids"])]
    if membership["train_readout_ids"]:
        frame = frame[frame["readout_id"].isin(membership["train_readout_ids"])]
    if frame.empty:
        raise ExecutionError("manifest membership selects no training rows")
    weights = trajectory_weights(tuple(frame["trajectory_id"].astype(str)))
    model = str(manifest["fit"]["model"])
    exposure_kwargs: dict[str, Any] = {}
    if model in EXPLORATORY_EXPOSURE_MODELS:
        required = {"D_transfer", "D_other"}
        if not required <= set(frame.columns):
            raise ExecutionError(f"{model} input lacks sealed exposure components")
        exposure_kwargs = {"D_transfer": frame["D_transfer"], "D_other": frame["D_other"]}
    elif model == "M6":
        required = {"D_family", "D_script", "D_other"}
        if not required <= set(frame.columns):
            raise ExecutionError("M6 input lacks sealed family, script, or other exposure")
        exposure_kwargs = {name: frame[name] for name in sorted(required)}
    regularization = _regularization_for_model(model,
        manifest["fit"].get("regularization"))
    return ProjectedProblem.from_arrays(y=frame["adjusted_loss"], weights=weights, N=frame["N_act"],
        D=frame[data_column(model)], E=frame["E"], K=frame["K"], language_ids=tuple(frame["language"].astype(str)),
        unigram_nll=frame["unigram_nll"], regularization=regularization,
        **exposure_kwargs)

def _fit_record(result: Any) -> dict[str, Any]:
    base = {"start_index": result.start_index, "start_id": result.start_id, "success": result.success,
            "status": result.status, "message": result.message, "objective": result.objective,
            "iterations": result.iterations, "function_evaluations": result.function_evaluations}
    if result.success and result.shared_parameters is not None and result.profile is not None:
        shared = result.shared_parameters.as_dict()
        if getattr(result, "model", None) in EXPLORATORY_EXPOSURE_MODELS:
            shared.update({"tau_transfer": float(result.shared_parameters.tau_transfer),
                "tau_other": float(result.shared_parameters.tau_other)})
        if getattr(result, "model", None) == "M6":
            shared.update({"eta_family": float(result.shared_parameters.eta_family),
                "eta_script": float(result.shared_parameters.eta_script),
                "gamma_other": float(result.shared_parameters.gamma_other)})
        base.update({"coordinates": result.coordinates.tolist(), "shared_parameters": shared,
            "language_coefficients": [{"language": item.language, "L_inf": item.L_inf, "A": item.A, "B": item.B,
                "floor_fraction": item.floor_fraction, "floor_saturated": item.floor_saturated}
                for item in result.profile.coefficients],
            "boundary_contacts": [{"name": item.name, "side": item.side, "value": item.value,
                "distance": item.distance, "interval_width": item.interval_width} for item in result.boundary_contacts]})
        if getattr(result, "model", None) in EXPLORATORY_COMPOSITION_MODELS:
            if result.objective_components is None:
                raise ExecutionError("successful M5/M6 fit lacks objective components")
            base["objective_components"] = result.objective_components.as_dict()
    return base

def _direct_fit_record(result: Any, model: str) -> dict[str, Any]:
    shared = result.shared_parameters.as_dict()
    if model in EXPLORATORY_EXPOSURE_MODELS:
        shared.update({"tau_transfer": float(result.shared_parameters.tau_transfer),
            "tau_other": float(result.shared_parameters.tau_other)})
    if model == "M6":
        shared.update({"eta_family": float(result.shared_parameters.eta_family),
            "eta_script": float(result.shared_parameters.eta_script),
            "gamma_other": float(result.shared_parameters.gamma_other)})
    record = {"success": result.success, "status": result.status, "message": result.message,
        "objective": result.objective, "outer_coordinates": result.outer_coordinates.tolist(),
        "shared_parameters": shared,
        "language_coefficients": [{"language": language, **values}
            for language, values in sorted(result.language_coefficients.items())],
        "predictions": result.prediction.tolist(), "iterations": result.iterations,
        "function_evaluations": result.function_evaluations}
    if model in EXPLORATORY_COMPOSITION_MODELS and result.success:
        if result.objective_components is None:
            raise ExecutionError("successful direct M5/M6 fit lacks objective components")
        record["objective_components"] = result.objective_components.as_dict()
    return record

def default_runner(manifest: Mapping[str, Any], start_index: int) -> Mapping[str, Any]:
    problem = load_problem(manifest)
    result = run_projected_start(manifest["fit"]["model"], _point(manifest, start_index),
        problem, delta=manifest["fit"]["delta"], bounds=_bounds(manifest))
    record = _fit_record(result)
    if manifest["fit"]["action"] == "direct" and result.success:
        check = direct_joint_check(result, problem, delta=manifest["fit"]["delta"], bounds=_bounds(manifest))
        record["direct_equivalence"] = {"resolved": check.resolved, "retried": check.retried,
            "relative_objective_difference": check.equivalence.relative_objective_difference,
            "maximum_absolute_prediction_difference": check.equivalence.maximum_absolute_prediction_difference}
        record["direct_fit"] = _direct_fit_record(check.direct, str(manifest["fit"]["model"]))
        record["success"] = check.resolved
        if not check.resolved:
            record["message"] = "projected/direct equivalence unresolved"
    return record

def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExecutionError("non-finite result")
    if isinstance(value, Mapping):
        for item in value.values(): _finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _finite_json(item)

def chunk_path(manifest: Mapping[str, Any], task_index: int) -> Path:
    return Path(manifest["execution"]["output_root"]) / "chunks" / manifest["manifest_sha256"][:20] / f"task-{task_index:06d}.json"

def _validate_objective_components(record: Mapping[str, Any]) -> None:
    components = record.get("objective_components")
    if not isinstance(components, Mapping) or set(components) != {
            "data_fit", "regularization", "total"}:
        raise ExecutionError("successful M5/M6 fit lacks canonical objective components")
    if any(isinstance(components[name], bool)
            or not isinstance(components[name], (int, float))
            or not math.isfinite(float(components[name]))
            for name in ("data_fit", "regularization", "total")):
        raise ExecutionError("M5/M6 objective components must be finite numbers")
    if (float(components["regularization"]) < 0.0
            or (float(components["data_fit"])
                + float(components["regularization"]) != float(components["total"]))
            or float(components["total"]) != float(record["objective"])):
        raise ExecutionError("M5/M6 objective components disagree with the fitted objective")

def validate_chunk(chunk: Mapping[str, Any], manifest: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    digest = _verify_seal(chunk, "chunk_sha256")
    if chunk.get("schema_version") != CHUNK_SCHEMA or chunk.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ExecutionError("chunk identity changed")
    if chunk.get("task") != task:
        raise ExecutionError("chunk task changed")
    records = chunk.get("records")
    if not isinstance(records, list) or [item.get("start_index") for item in records] != list(range(task["start_begin"], task["start_end"])):
        raise ExecutionError("chunk start coverage changed")
    for index,item in zip(range(task["start_begin"],task["start_end"]),records):
        if not isinstance(item,Mapping) or item.get("start_id")!=_start_id(manifest,index) or not isinstance(item.get("success"),bool):
            raise ExecutionError("chunk start identity or status changed")
        objective=item.get("objective")
        if item["success"] and (not isinstance(objective,(int,float)) or not math.isfinite(float(objective))):
            raise ExecutionError("successful start lacks a finite objective")
        if item["success"] and manifest["fit"]["model"] in EXPLORATORY_COMPOSITION_MODELS:
            _validate_objective_components(item)
            direct = item.get("direct_fit")
            if isinstance(direct, Mapping) and direct.get("success") is True:
                _validate_objective_components(direct)
    _finite_json(records); return digest

def execute_task(manifest: Mapping[str, Any], task_index: int, *, runner: Runner = default_runner) -> dict[str, Any]:
    validate_manifest(manifest)
    _require_open_root(Path(manifest["execution"]["output_root"]))
    if not 0 <= task_index < len(manifest["tasks"]): raise ExecutionError("task index out of range")
    task = manifest["tasks"][task_index]; path = chunk_path(manifest, task_index)
    if path.exists():
        chunk = read_json(path); validate_chunk(chunk, manifest, task); return {"status": "skipped_hash_valid", "path": str(path)}
    threads = str(manifest["execution"]["blas_threads"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[name] = threads
    active_runner=runner
    if runner is default_runner:
        problem=load_problem(manifest); bounds=_bounds(manifest)
        def active_runner(_manifest: Mapping[str,Any], index: int) -> Mapping[str,Any]:
            result=run_projected_start(manifest["fit"]["model"],_point(manifest,index),
                problem,delta=manifest["fit"]["delta"],bounds=bounds)
            record=_fit_record(result)
            if manifest["fit"]["action"]=="direct" and result.success:
                check=direct_joint_check(result,problem,delta=manifest["fit"]["delta"],bounds=bounds)
                record["direct_equivalence"]={"resolved":check.resolved,"retried":check.retried,
                    "relative_objective_difference":check.equivalence.relative_objective_difference,
                    "maximum_absolute_prediction_difference":check.equivalence.maximum_absolute_prediction_difference}
                record["direct_fit"]=_direct_fit_record(check.direct, str(manifest["fit"]["model"]))
                record["success"]=check.resolved
            return record
    records = []
    for index in range(task["start_begin"], task["start_end"]):
        started = time.perf_counter()
        try:
            record = dict(active_runner(manifest, index)); record["start_index"] = index
            _finite_json(record)
        except Exception as error:
            record = {"start_index": index, "start_id": _start_id(manifest,index), "success": False,
                      "status": -1, "message": f"{type(error).__name__}: {error}", "objective": None}
        record["runtime_seconds"] = time.perf_counter() - started; records.append(record)
    chunk = _seal({"schema_version": CHUNK_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
                   "task": task, "records": records}, "chunk_sha256")
    write_json_once(path, chunk); return {"status": "written", "path": str(path), "chunk_sha256": chunk["chunk_sha256"]}

def _without_runtime(value: Any) -> Any:
    if isinstance(value, Mapping): return {key: _without_runtime(item) for key,item in value.items() if key != "runtime_seconds"}
    if isinstance(value, list): return [_without_runtime(item) for item in value]
    return value

def result_path(manifest: Mapping[str, Any]) -> Path:
    root = Path(manifest["execution"]["output_root"])
    return root / "results" / f"{manifest['manifest_sha256'][:20]}.json"

def _diagnostic(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("start_index", "start_id", "success", "status", "message", "objective",
            "objective_components",
            "coordinates", "iterations", "function_evaluations", "boundary_contacts",
            "direct_equivalence")
    return {key: _without_runtime(record[key]) for key in keys if key in record}

def reduce(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    _require_open_root(Path(manifest["execution"]["output_root"]))
    chunks=[]; records=[]
    for task in manifest["tasks"]:
        path=chunk_path(manifest,task["task_index"])
        if not path.exists(): raise ExecutionError(f"missing chunk: {task['task_id']}")
        chunk=read_json(path); chunks.append(validate_chunk(chunk,manifest,task)); records.extend(chunk["records"])
    indices=[item["start_index"] for item in records]
    if indices != list(range(manifest["fit"]["start_count"])): raise ExecutionError("reducer coverage differs")
    successes=[item for item in records if item.get("success") is True and item.get("objective") is not None]
    selected=None if not successes else min(successes,key=lambda item:(float(item["objective"]),str(item["start_id"])))
    root=Path(manifest["execution"]["output_root"]); stem=manifest["manifest_sha256"][:20]
    receipt=_seal({"schema_version":RECEIPT_SCHEMA,"manifest_sha256":manifest["manifest_sha256"],
                   "chunk_sha256s":chunks,"start_count":len(records),"successful_starts":len(successes)},"receipt_sha256")
    receipt_path=root/"receipts"/f"{stem}.json"; write_json_once(receipt_path,receipt)
    scale=None
    if selected and manifest["fit"]["model"]=="M0" and manifest["fit"]["objective"]=="ols" and {"shared_parameters","language_coefficients"} <= set(selected):
        problem=load_problem(manifest); shared=SharedParameters(**selected["shared_parameters"])
        coefficients={item["language"]:item for item in selected["language_coefficients"]}
        nuisance=[coefficients[language] for language in problem.language_ids]
        prediction=predict("M0",shared,problem.N,problem.D,problem.E,problem.K,
            [item["L_inf"] for item in nuisance],[item["A"] for item in nuisance],[item["B"] for item in nuisance])
        robust=adaptive_huber_scale(prediction-problem.y,problem.weights)
        scale={"location":robust.location,"weighted_mad":robust.weighted_mad,"scale":robust.scale,
               "delta":robust.delta,"use_ols":robust.use_ols}
    result=_seal({"schema_version":RESULT_SCHEMA,"manifest_sha256":manifest["manifest_sha256"],
                  "manifest_path":str(root / "manifests" / f"{manifest['phase']}-{stem}.json"),
                  "receipt_path":str(receipt_path),"receipt_sha256":receipt["receipt_sha256"],
                  "fit_rows_content_sha256":manifest["input"]["fit_rows_content_sha256"],
                  "fit_identity":{key:manifest["fit"][key] for key in ("model","fold_id","replicate_id","objective","bound_version")},
                  "fit_bounds":manifest["fit"]["bounds"],
                  "membership":manifest.get("membership"),"review_only":manifest["review_only"],
                  "parents":manifest["parents"],
                  "selection_report":manifest["selection_report"],
                  "status":"reduced" if selected else "failed","selected":_without_runtime(selected),
                  "attempts":[_diagnostic(record) for record in records],"frozen_delta":manifest["fit"]["delta"],
                  "adaptive_huber_scale":scale},"result_sha256")
    write_json_once(result_path(manifest),result); return result

def run_serial(manifest: Mapping[str, Any], *, runner: Runner = default_runner) -> dict[str, Any]:
    for task in manifest["tasks"]: execute_task(manifest,task["task_index"],runner=runner)
    return reduce(manifest)

def render_slurm(manifest_path: Path, manifest: Mapping[str, Any]) -> tuple[str, str]:
    validate_manifest(manifest)
    execution=manifest["execution"]; count=len(manifest["tasks"])
    attempt_root = (Path(str(execution["output_root"])) / "logs" / "attempts"
        / str(manifest["manifest_sha256"])[:20])
    attempt_root.mkdir(parents=True, exist_ok=True)
    directives=["#!/bin/bash",f"#SBATCH --job-name=s1fit-{manifest['phase']}",f"#SBATCH --array=0-{count-1}%{execution['concurrency']}",
        f"#SBATCH --cpus-per-task={execution['blas_threads']}",f"#SBATCH --mem={execution['memory']}",f"#SBATCH --time={execution['time_limit']}",
        f"#SBATCH --output={attempt_root / 'slurm-%A_%a.out'}"]
    for key,flag in (("partition","partition"),("account","account"),("reservation","reservation")):
        if execution[key]: directives.append(f"#SBATCH --{flag}={execution[key]}")
    threads=execution["blas_threads"]
    body=[f"export {name}={threads}" for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")]
    body.append(f"PYTHONPATH=src:. {sys.executable} -m multilingual_megatron.experiments.scaling.stage1_fit.cli run-task --manifest {manifest_path} --task-index \"$SLURM_ARRAY_TASK_ID\"")
    script="\n".join(directives+[""]+body)+"\n"; return script,f"sbatch <script-for-{manifest['manifest_sha256'][:20]}>"
