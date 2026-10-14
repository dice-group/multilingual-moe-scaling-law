"""Post-hoc decomposition of M2's two expert-by-language interactions.

The two nine-parameter laws sit exactly between M1 and M2:

``M1-PhiE`` lets experts modify only the capacity-side language term, while
``M1-PsiE`` lets experts modify only the data-side language term.  This module
reuses the three frozen primary holdouts and the audited M1/M2 fits.  Every new
fit is permanently review-only and cannot replace the registered M2 result.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .campaign import _record_prediction, _records, scientific_result
from .data import canonical_json_sha256, load_fit_rows_snapshot, sha256_file
from .execution import (
    bind_membership,
    build_manifest,
    load_manifest,
    read_json,
    reduce,
    render_slurm,
    result_path,
    write_json_once,
    write_manifest,
    write_text_once,
)
from .fit import (
    EXPLORATORY_M1_ABLATION_MODELS,
    OBJECTIVE_RELATIVE_TOLERANCE,
    PREDICTION_ABSOLUTE_TOLERANCE,
    SharedParameters,
    StartPoint,
    deterministic_local_perturbations,
    effective_exponents,
    equivalence_check,
    initial_bounds,
    outer_from_shared,
    outer_parameter_names,
    predictor_terms,
    shared_from_outer,
)
from .interaction import (
    OFFICIAL_REPORT_SCHEMA,
    PRIMARY_FOLDS,
    _official_report,
    _profile,
    _reduced_exploratory,
    _result_parent,
    _seal,
    _sealed,
    _test_metrics,
)
from .selection import PAIRED_BOOTSTRAP_REPLICATES, select_model_paired_one_standard_error


MODEL_ORDER = ("M1",) + EXPLORATORY_M1_ABLATION_MODELS + ("M2",)
INDEX_SCHEMA = "stage1_m15_interaction_ablation_index/v1"
AUDIT_INDEX_SCHEMA = "stage1_m15_interaction_ablation_audit_index/v1"
AUDIT_SCHEMA = "stage1_m15_interaction_ablation_audit/v1"
REPORT_SCHEMA = "stage1_m15_interaction_ablation_report/v1"
SUBMISSION_SCHEMA = "stage1_m15_interaction_ablation_submission/v1"
ANCHORS_PER_PARENT = 8
STARTS_PER_FIT = 24


class AblationError(RuntimeError):
    """Raised when an ablation binding or numerical gate changes."""


def _official_paths(report: Mapping[str, Any], model: str) -> dict[str, Path]:
    paths = {
        str(item["fold_id"]): Path(str(item["path"]))
        for item in report.get("source_results", [])
        if item.get("role") == "primary" and item.get("model") == model
    }
    if set(paths) != set(PRIMARY_FOLDS):
        raise AblationError(f"official report does not bind exactly three {model} primary folds")
    return paths


def _retained_anchors(result: Mapping[str, Any], *, model: str, count: int) -> tuple[np.ndarray, ...]:
    selected = result.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("coordinates"), list):
        raise AblationError(f"official {model} result lacks selected coordinates")
    manifest = load_manifest(Path(str(result["manifest_path"])), require_current_provenance=False)
    if manifest["fit"]["model"] != model:
        raise AblationError(f"anchor manifest is not {model}")
    bounds = initial_bounds(model)
    widths = np.asarray([bound.upper - bound.lower for bound in bounds], dtype=np.float64)
    anchors: list[np.ndarray] = [np.asarray(selected["coordinates"], dtype=np.float64)]
    usable = [
        item for item in result.get("attempts", [])
        if item.get("success") and isinstance(item.get("objective"), (int, float))
        and isinstance(item.get("coordinates"), list)
    ]
    usable.sort(key=lambda item: (float(item["objective"]), str(item.get("start_id", ""))))
    for item in usable:
        if len(anchors) == count:
            break
        candidate = np.asarray(item["coordinates"], dtype=np.float64)
        if candidate.shape != anchors[0].shape or not np.isfinite(candidate).all():
            continue
        if all(float(np.max(np.abs((candidate - existing) / widths))) > 1e-6 for existing in anchors):
            anchors.append(candidate)
    if len(anchors) < count:
        for point in deterministic_local_perturbations(anchors[0], bounds, relative_step=0.02):
            anchors.append(np.asarray(point.values, dtype=np.float64))
            if len(anchors) == count:
                break
    width = len(outer_parameter_names(model))
    if len(anchors) != count or any(value.shape != (width,) for value in anchors):
        raise AblationError(f"could not construct {count} deterministic {model} anchors")
    return tuple(anchors)


def _collapse_m2(model: str, source: SharedParameters, *, endpoint: str) -> SharedParameters:
    if model not in EXPLORATORY_M1_ABLATION_MODELS or endpoint not in {"e1", "e32"}:
        raise AblationError("unknown intermediate model or collapse endpoint")
    values = source.as_dict()
    if model == "M1-PhiE":
        values["psi_0"] = source.psi_0 if endpoint == "e1" else source.psi_0 + source.psi_1 * math.log(32.0)
        values["psi_1"] = 0.0
    else:
        values["phi_0"] = source.phi_0 if endpoint == "e1" else source.phi_0 + source.phi_1 * math.log(32.0)
        values["phi_1"] = 0.0
    return SharedParameters(**values)


def ablation_starts(model: str, m1_result: Mapping[str, Any],
    m2_result: Mapping[str, Any]) -> tuple[StartPoint, ...]:
    """Return 24 deterministic starts spanning both exact parent basins."""
    if model not in EXPLORATORY_M1_ABLATION_MODELS:
        raise AblationError("unknown M1 interaction ablation")
    m1_anchors = _retained_anchors(m1_result, model="M1", count=ANCHORS_PER_PARENT)
    m2_anchors = _retained_anchors(m2_result, model="M2", count=ANCHORS_PER_PARENT)
    starts: list[StartPoint] = []
    for index, values in enumerate(m1_anchors):
        shared = shared_from_outer("M1", values)
        starts.append(StartPoint(f"ablation:m1:a{index}", None, outer_from_shared(model, shared), "m1_anchor"))
    for endpoint in ("e1", "e32"):
        for index, values in enumerate(m2_anchors):
            shared = _collapse_m2(model, shared_from_outer("M2", values), endpoint=endpoint)
            starts.append(StartPoint(
                f"ablation:m2-{endpoint}:a{index}", None,
                outer_from_shared(model, shared), f"m2_{endpoint}_anchor",
            ))
    if len(starts) != STARTS_PER_FIT or len({item.start_id for item in starts}) != STARTS_PER_FIT:
        raise AblationError("ablation start construction changed")
    return tuple(starts)


def _phase(fold: str, model: str) -> str:
    slug = model.lower().replace("-", "")
    return f"m15-ablation-{fold}-{slug}"


def _write_index(output_root: Path, payload: Mapping[str, Any]) -> Path:
    index = _seal({"schema_version": INDEX_SCHEMA, **dict(payload)}, "index_sha256")
    path = output_root / "ablation" / "screen_index.json"
    write_json_once(path, index)
    return path


def plan(*, profile_path: Path, input_receipt_path: Path, holdouts_path: Path,
    candidate_report_path: Path, dry_run: bool = False) -> dict[str, Any]:
    profile = _profile(profile_path)
    output_root = Path(str(profile["output_root"]))
    receipt = read_json(input_receipt_path)
    report = _official_report(candidate_report_path)
    if report.get("schema_version") != OFFICIAL_REPORT_SCHEMA:
        raise AblationError("official report schema changed")
    if report.get("input_sha256") != receipt.get("fit_rows_content_sha256"):
        raise AblationError("official report and input receipt differ")
    holdouts = _sealed(read_json(holdouts_path), "manifest_sha256")
    if holdouts.get("input_sha256") != receipt.get("fit_rows_content_sha256"):
        raise AblationError("holdouts and input receipt differ")
    m1_paths = _official_paths(report, "M1")
    m2_paths = _official_paths(report, "M2")
    planned: dict[str, dict[str, str]] = {model: {} for model in EXPLORATORY_M1_ABLATION_MODELS}
    totals = {"starts": 0, "tasks": 0}
    for fold in PRIMARY_FOLDS:
        membership = bind_membership(holdouts_path, kind="primary", membership_id=fold)
        m1 = scientific_result(m1_paths[fold]); m2 = scientific_result(m2_paths[fold])
        for result, expected_model in ((m1, "M1"), (m2, "M2")):
            identity = result.get("fit_identity", {})
            if identity.get("model") != expected_model or identity.get("fold_id") != fold:
                raise AblationError("official source has the wrong identity")
        if (m1["fit_identity"]["objective"] != m2["fit_identity"]["objective"]
                or m1.get("frozen_delta") != m2.get("frozen_delta")):
            raise AblationError("official M1 and M2 use different objectives")
        scale_path = _result_parent(m2, "scale_m0_ols")
        for model in EXPLORATORY_M1_ABLATION_MODELS:
            starts = ablation_starts(model, m1, m2)
            manifest = build_manifest(
                output_root=output_root,
                fit_rows_path=Path(str(receipt["fit_rows_path"])),
                fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]),
                phase=_phase(fold, model), model=model, fold_id=fold,
                objective=str(m2["fit_identity"]["objective"]), delta=m2.get("frozen_delta"),
                chunk_size=int(profile["chunk_size"]), bounds=initial_bounds(model),
                parent_results={"anchor_m1": m1_paths[fold], "anchor_m2": m2_paths[fold],
                    "scale_m0_ols": scale_path},
                selection_report=candidate_report_path, membership=membership,
                explicit_starts=starts, review_only=True, action="m15_ablation_screen",
                blas_threads=int(profile["blas_threads"]), partition=profile.get("partition"),
                account=profile.get("account"), reservation=profile.get("reservation"),
                memory=str(profile["memory"]), time_limit=str(profile["time_limit"]),
                concurrency=int(profile["array_concurrency"]),
            )
            totals["starts"] += int(manifest["fit"]["start_count"])
            totals["tasks"] += len(manifest["tasks"])
            if not dry_run:
                planned[model][fold] = str(write_manifest(manifest))
    expected = {"starts": 144, "tasks": 12}
    if totals != expected:
        raise AblationError(f"ablation workload changed: {totals} != {expected}")
    result: dict[str, Any] = {**totals, "written": not dry_run, "manifests": planned}
    if not dry_run:
        path = _write_index(output_root, {
            "exploratory": True, "output_root": str(output_root.resolve()),
            "profile": {"path": str(profile_path.resolve()), "file_sha256": sha256_file(profile_path)},
            "input_receipt": {"path": str(input_receipt_path.resolve()), "file_sha256": sha256_file(input_receipt_path)},
            "holdouts": {"path": str(holdouts_path.resolve()), "file_sha256": sha256_file(holdouts_path),
                "manifest_sha256": holdouts["manifest_sha256"]},
            "official_candidate_report": {"path": str(candidate_report_path.resolve()),
                "file_sha256": sha256_file(candidate_report_path), "report_sha256": report["report_sha256"]},
            "official_results": {"M1": {fold: str(path.resolve()) for fold, path in m1_paths.items()},
                "M2": {fold: str(path.resolve()) for fold, path in m2_paths.items()}},
            "manifests": planned, "expected": totals,
        })
        result["index_path"] = str(path)
    return result


def _index(path: Path) -> Mapping[str, Any]:
    value = _sealed(read_json(path), "index_sha256")
    if value.get("schema_version") not in {INDEX_SCHEMA, AUDIT_INDEX_SCHEMA}:
        raise AblationError("unknown ablation index schema")
    return value


def emit_slurm(index_path: Path) -> dict[str, Any]:
    index = _index(index_path); scripts: dict[str, str] = {}
    for model, folds in index["manifests"].items():
        for key, manifest_value in folds.items():
            manifest_path = Path(str(manifest_value)); manifest = load_manifest(manifest_path)
            script, _ = render_slurm(manifest_path, manifest)
            label = f"{model}/{key}"
            output = Path(str(index["output_root"])) / "slurm" / f"{model}-{key}-{manifest['manifest_sha256'][:12]}.sbatch"
            write_text_once(output, script); scripts[label] = str(output)
    return {"index_sha256": index["index_sha256"], "scripts": scripts,
        "count": len(scripts), "submitted": False}


def record_submission(index_path: Path, jobs: Sequence[str], *, attempt: int) -> dict[str, Any]:
    index = _index(index_path)
    expected = {f"{model}/{key}" for model, values in index["manifests"].items() for key in values}
    parsed: dict[str, str] = {}
    for item in jobs:
        key, separator, job_id = item.partition("=")
        if separator != "=" or not job_id.isdigit() or key in parsed:
            raise AblationError(f"malformed submission binding: {item}")
        parsed[key] = job_id
    if set(parsed) != expected or attempt < 1:
        raise AblationError("submission bindings do not exactly cover the index")
    payload = _seal({"schema_version": SUBMISSION_SCHEMA, "index_path": str(index_path.resolve()),
        "index_sha256": index["index_sha256"], "attempt": attempt, "jobs": parsed}, "submission_sha256")
    root = Path(str(index["output_root"]))
    path = root / "submissions" / f"{index['schema_version'].split('/')[0]}-attempt-{attempt:02d}.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def status(index_path: Path) -> dict[str, Any]:
    index = _index(index_path); rows = []
    for model, folds in index["manifests"].items():
        for key, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value)))
            root = Path(str(index["output_root"])); stem = manifest["manifest_sha256"][:20]
            chunk_root = root / "chunks" / stem
            present = sum((chunk_root / f"task-{task['task_index']:06d}.json").exists() for task in manifest["tasks"])
            rows.append({"key": f"{model}/{key}", "tasks": len(manifest["tasks"]), "chunks": present,
                "result": (root / "results" / f"{stem}.json").exists()})
    return {"complete": all(row["chunks"] == row["tasks"] for row in rows), "rows": rows}


def reduce_index(index_path: Path) -> dict[str, Any]:
    index = _index(index_path); paths: dict[str, dict[str, str]] = {}
    for model, folds in index["manifests"].items():
        paths[model] = {}
        for key, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value))); result = reduce(manifest)
            if result.get("status") != "reduced" or not result.get("selected"):
                raise AblationError(f"{model}/{key} produced no selected fit")
            paths[model][key] = str(result_path(manifest))
    return {"results": paths, "count": sum(len(value) for value in paths.values())}


def plan_audits(screen_index_path: Path) -> dict[str, Any]:
    screen = _index(screen_index_path)
    if screen.get("schema_version") != INDEX_SCHEMA:
        raise AblationError("audit planning requires the screen index")
    output_root = Path(str(screen["output_root"])); planned: dict[str, dict[str, str]] = {}
    totals = {"starts": 0, "tasks": 0}
    for model, folds in screen["manifests"].items():
        planned[model] = {}
        for fold, manifest_value in folds.items():
            base = load_manifest(Path(str(manifest_value))); result, _ = _reduced_exploratory(result_path(base))
            usable = [item for item in result.get("attempts", []) if item.get("success") and item.get("coordinates")]
            usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
            if len(usable) < 3:
                raise AblationError(f"{model}/{fold} lacks three successful screen starts")
            parents = {str(item["kind"]): Path(str(item["path"])) for item in base.get("parents", [])}
            parents["grid"] = result_path(base)
            selection = None if base.get("selection_report") is None else Path(str(base["selection_report"]["path"]))
            actions = (
                ("basin", tuple(StartPoint(f"basin:{index}", None, np.asarray(item["coordinates"], dtype=np.float64), "basin") for index, item in enumerate(usable[:3]))),
                ("direct", (StartPoint("direct:best", None, np.asarray(result["selected"]["coordinates"], dtype=np.float64), "direct"),)),
            )
            for action, starts in actions:
                manifest = build_manifest(
                    output_root=output_root, fit_rows_path=Path(str(base["input"]["fit_rows_path"])),
                    fit_rows_content_sha256=str(base["input"]["fit_rows_content_sha256"]),
                    phase=f"{base['phase']}-{action}", model=model, fold_id=fold,
                    objective=str(base["fit"]["objective"]), delta=base["fit"]["delta"],
                    chunk_size=int(base["execution"]["chunk_size"]), bounds=initial_bounds(model),
                    parent_results=parents, selection_report=selection, membership=base["membership"],
                    explicit_starts=starts, review_only=True, action=action,
                    blas_threads=int(base["execution"]["blas_threads"]),
                    partition=base["execution"].get("partition"), account=base["execution"].get("account"),
                    reservation=base["execution"].get("reservation"), memory=str(base["execution"]["memory"]),
                    time_limit=str(base["execution"]["time_limit"]), concurrency=int(base["execution"]["concurrency"]),
                )
                path = write_manifest(manifest); planned[model][f"{fold}:{action}"] = str(path)
                totals["starts"] += int(manifest["fit"]["start_count"]); totals["tasks"] += len(manifest["tasks"])
    if totals != {"starts": 24, "tasks": 12}:
        raise AblationError(f"audit workload changed: {totals}")
    payload = _seal({"schema_version": AUDIT_INDEX_SCHEMA, "exploratory": True,
        "output_root": str(output_root), "screen_index": {"path": str(screen_index_path.resolve()),
            "index_sha256": screen["index_sha256"]}, "manifests": planned, "expected": totals}, "index_sha256")
    path = output_root / "ablation" / "screen_audit_index.json"; write_json_once(path, payload)
    return {"index_path": str(path), **totals, "manifests": planned}


def _maximum_term_difference(first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray]) -> float:
    return max(float(np.max(np.abs(left - right))) for left, right in zip(first, second))


def audit(screen_index_path: Path, audit_index_path: Path) -> dict[str, Any]:
    screen = _index(screen_index_path); audit_index = _index(audit_index_path)
    if screen.get("schema_version") != INDEX_SCHEMA or audit_index.get("schema_version") != AUDIT_INDEX_SCHEMA:
        raise AblationError("audit indices have the wrong schema")
    if audit_index.get("screen_index", {}).get("index_sha256") != screen["index_sha256"]:
        raise AblationError("audit index does not bind the screen")
    items = []
    predictors = (np.asarray([1e8, 2e8, 4e8, 8e8]), np.asarray([1e8, 5e8, 2e9, 8e9]),
        np.asarray([1.0, 4.0, 16.0, 32.0]), np.asarray([1.0, 12.0, 50.0, 200.0]))
    for model, folds in screen["manifests"].items():
        for fold, base_value in folds.items():
            base_manifest = load_manifest(Path(str(base_value))); base_result, _ = _reduced_exploratory(result_path(base_manifest))
            basin_manifest = load_manifest(Path(str(audit_index["manifests"][model][f"{fold}:basin"])))
            direct_manifest = load_manifest(Path(str(audit_index["manifests"][model][f"{fold}:direct"])))
            basin_result, _ = _reduced_exploratory(result_path(basin_manifest))
            direct_result, _ = _reduced_exploratory(result_path(direct_manifest))
            retained = base_result["selected"]; retained_prediction = _record_prediction(retained, manifest=base_manifest)
            equivalent = 0
            for record in _records(basin_manifest):
                if not record.get("success") or not isinstance(record.get("objective"), (int, float)):
                    continue
                check = equivalence_check(float(retained["objective"]), float(record["objective"]), retained_prediction,
                    _record_prediction(record, manifest=basin_manifest), objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
                    prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)
                equivalent += int(check.equivalent)
            direct = direct_result.get("selected", {}).get("direct_equivalence", {})
            contacts = retained.get("boundary_contacts", [])
            selected_parameters = SharedParameters(**retained["shared_parameters"])
            m1 = scientific_result(Path(str(screen["official_results"]["M1"][fold])))
            m1_parameters = SharedParameters(**m1["selected"]["shared_parameters"])
            embedded_m1 = shared_from_outer(model, outer_from_shared(model, m1_parameters))
            m1_difference = _maximum_term_difference(predictor_terms("M1", m1_parameters, *predictors),
                predictor_terms(model, embedded_m1, *predictors))
            embedded_m2 = shared_from_outer("M2", outer_from_shared("M2", selected_parameters))
            m2_difference = _maximum_term_difference(predictor_terms(model, selected_parameters, *predictors),
                predictor_terms("M2", embedded_m2, *predictors))
            alpha, beta, _, _ = effective_exponents(selected_parameters, np.asarray([1.0, 4.0, 16.0, 32.0]), model=model)
            fixed_slope = selected_parameters.psi_1 if model == "M1-PhiE" else selected_parameters.phi_1
            complete = (equivalent >= 3 and bool(direct.get("resolved")) and not contacts
                and m1_difference <= 1e-12 and m2_difference <= 1e-12 and fixed_slope == 0.0
                and bool(np.all(alpha > 0.0)) and bool(np.all(beta > 0.0)))
            items.append({"model": model, "fold": fold, "status": "complete" if complete else "failed",
                "equivalent_basin_starts": equivalent, "direct_equivalence": direct,
                "boundary_contacts": contacts, "maximum_m1_nesting_term_difference": m1_difference,
                "maximum_m2_nesting_term_difference": m2_difference, "fixed_interaction_slope": fixed_slope,
                "minimum_alpha": float(np.min(alpha)), "minimum_beta": float(np.min(beta)),
                "base_result_sha256": base_result["result_sha256"],
                "basin_result_sha256": basin_result["result_sha256"],
                "direct_result_sha256": direct_result["result_sha256"]})
    status_value = "complete" if all(item["status"] == "complete" for item in items) else "failed"
    payload = _seal({"schema_version": AUDIT_SCHEMA, "status": status_value,
        "screen_index_sha256": screen["index_sha256"], "audit_index_sha256": audit_index["index_sha256"],
        "items": items}, "audit_sha256")
    path = Path(str(screen["output_root"])) / "reports" / "numerical_audit.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def report(screen_index_path: Path, audit_path: Path) -> dict[str, Any]:
    screen = _index(screen_index_path); audit_payload = _sealed(read_json(audit_path), "audit_sha256")
    if (audit_payload.get("schema_version") != AUDIT_SCHEMA or audit_payload.get("status") != "complete"
            or audit_payload.get("screen_index_sha256") != screen["index_sha256"]):
        raise AblationError("numerical audit is incomplete or mismatched")
    official = _official_report(Path(str(screen["official_candidate_report"]["path"])))
    receipt = read_json(Path(str(screen["input_receipt"]["path"])))
    frame = load_fit_rows_snapshot(receipt)
    errors: dict[str, dict[str, dict[str, float]]] = {}; metrics: dict[str, dict[str, Any]] = {}
    sources = []
    for model in ("M1", "M2"):
        errors[model] = {}; metrics[model] = {}
        for fold in PRIMARY_FOLDS:
            metric = official["primary_metrics"][model][fold]
            metrics[model][fold] = metric
            errors[model][fold] = {str(key): float(value) for key, value in metric["trajectory_mse"].items()}
    for model, folds in screen["manifests"].items():
        errors[model] = {}; metrics[model] = {}
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value))); result, _ = _reduced_exploratory(result_path(manifest))
            metric = _test_metrics(frame, manifest["membership"], result["selected"], model)
            metrics[model][fold] = metric; errors[model][fold] = metric["trajectory_mse"]
            sources.append({"model": model, "fold": fold, "result_path": str(result_path(manifest)),
                "result_sha256": result["result_sha256"], "selected_coordinates": result["selected"]["coordinates"],
                "shared_parameters": result["selected"]["shared_parameters"]})
    selection = select_model_paired_one_standard_error(errors, model_order=MODEL_ORDER,
        fold_order=PRIMARY_FOLDS, replicates=PAIRED_BOOTSTRAP_REPLICATES)
    observed = selection.observed_aggregate_rmse
    denominator = observed["M1"] - observed["M2"]
    recovery = {model: None if denominator == 0.0 else (observed["M1"] - observed[model]) / denominator
        for model in EXPLORATORY_M1_ABLATION_MODELS}
    payload = _seal({"schema_version": REPORT_SCHEMA, "status": "complete", "exploratory": True,
        "screen_index_sha256": screen["index_sha256"], "audit_sha256": audit_payload["audit_sha256"],
        "official_candidate_report": screen["official_candidate_report"], "model_order": list(MODEL_ORDER),
        "parameter_counts": {"M1": 8, "M1-PhiE": 9, "M1-PsiE": 9, "M2": 10},
        "primary_metrics": metrics, "selection": selection.to_dict(),
        "fraction_of_m1_to_m2_rmse_improvement_recovered": recovery, "source_results": sources,
        "claim_boundary": "Post-hoc nested decomposition of M2. Predictive attribution is not a causal mechanism, and registered M2 remains authoritative."},
        "report_sha256")
    root = Path(str(screen["output_root"])); json_path = root / "reports" / "ablation_report.json"
    write_json_once(json_path, payload)
    lines = ["# M2 interaction decomposition", "",
        "Post-hoc nested ablation; the registered M2 result remains authoritative.", "",
        f"Observed best: **{selection.observed_best_model}**  ",
        f"Paired one-SE choice: **{selection.selected_model}**", "",
        "| Model | Parameters | Equal-fold RMSE | M1-to-M2 improvement recovered |",
        "|---|---:|---:|---:|"]
    for model in MODEL_ORDER:
        recovered = "—" if model not in recovery else f"{100.0 * float(recovery[model]):.1f}%"
        lines.append(f"| {model} | {payload['parameter_counts'][model]} | {observed[model]:.8f} | {recovered} |")
    lines.extend(["", "| Model | Fold | RMSE |", "|---|---|---:|"])
    for model in MODEL_ORDER:
        for fold in PRIMARY_FOLDS:
            lines.append(f"| {model} | {fold} | {metrics[model][fold]['trajectory_balanced_rmse']:.8f} |")
    lines.extend(["", "The ablation localizes predictive support within M2; it does not causally identify capacity competition or transfer.", ""])
    markdown_path = root / "reports" / "ablation_report.md"
    write_text_once(markdown_path, "\n".join(lines))
    return {"path": str(json_path), "markdown_path": str(markdown_path), **payload}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); commands = root.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan"); plan_parser.add_argument("--profile", type=Path, required=True)
    plan_parser.add_argument("--input-receipt", type=Path, required=True); plan_parser.add_argument("--holdouts", type=Path, required=True)
    plan_parser.add_argument("--candidate-report", type=Path, required=True); plan_parser.add_argument("--dry-run", action="store_true")
    emit = commands.add_parser("emit-slurm"); emit.add_argument("--index", type=Path, required=True)
    submit = commands.add_parser("record-submission"); submit.add_argument("--index", type=Path, required=True)
    submit.add_argument("--job", action="append", default=[]); submit.add_argument("--attempt", type=int, default=1)
    status_parser = commands.add_parser("status"); status_parser.add_argument("--index", type=Path, required=True)
    reduce_parser = commands.add_parser("reduce-index"); reduce_parser.add_argument("--index", type=Path, required=True)
    audits = commands.add_parser("plan-audits"); audits.add_argument("--screen-index", type=Path, required=True)
    audit_parser = commands.add_parser("audit"); audit_parser.add_argument("--screen-index", type=Path, required=True)
    audit_parser.add_argument("--audit-index", type=Path, required=True)
    report_parser = commands.add_parser("report"); report_parser.add_argument("--screen-index", type=Path, required=True)
    report_parser.add_argument("--audit", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan":
        result = plan(profile_path=args.profile, input_receipt_path=args.input_receipt,
            holdouts_path=args.holdouts, candidate_report_path=args.candidate_report, dry_run=args.dry_run)
    elif args.command == "emit-slurm": result = emit_slurm(args.index)
    elif args.command == "record-submission": result = record_submission(args.index, args.job, attempt=args.attempt)
    elif args.command == "status": result = status(args.index)
    elif args.command == "reduce-index": result = reduce_index(args.index)
    elif args.command == "plan-audits": result = plan_audits(args.screen_index)
    elif args.command == "audit": result = audit(args.screen_index, args.audit_index)
    else: result = report(args.screen_index, args.audit)
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
