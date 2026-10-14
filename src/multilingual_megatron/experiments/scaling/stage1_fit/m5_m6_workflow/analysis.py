"""Numerical audits, metrics, and winner reporting for the M5/M6 screen."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..campaign import _records
from ..execution import (
    load_problem,
    load_manifest,
    read_json,
    result_path,
    write_json_once,
    write_text_once,
)
from ..fit import (
    OBJECTIVE_RELATIVE_TOLERANCE,
    PREDICTION_ABSOLUTE_TOLERANCE,
    OuterBound,
    SharedParameters,
    effective_exponents,
    equivalence_check,
    outer_parameter_names,
    predict,
    predictor_terms_and_jacobian,
)
from ..interaction import (
    PRIMARY_FOLDS,
    _official_report,
    _seal,
    _sealed,
    _wave_index,
)
from ..m6_composition import component_content_sha256
from ..selection import (
    PAIRED_BOOTSTRAP_REPLICATES,
    select_model_paired_one_standard_error,
    trajectory_balanced_r2,
    trajectory_balanced_rmse,
    trajectory_mean_squared_errors,
)
from .contracts import (
    AUDIT_INDEX_SCHEMA,
    AUDIT_SCHEMA,
    COMPONENT_COLUMNS,
    EQUAL_FOLD_M2_GATE,
    INDEX_SCHEMA,
    K200_M1_GATE,
    MINIMUM_EQUIVALENT_BASIN_STARTS,
    MODELS,
    M5M6Error,
    REPORT_SCHEMA,
    RIDGE_LAMBDAS,
    _historical_m2kn_result,
    _lambda_key,
)
from .campaign import (
    _campaign_result,
    _index,
    _iter_manifest_items,
    _reduced_current_exploratory,
)

def _audit_manifest_path(audits: Mapping[str, Any], model: str, lambda_key: str,
        fold: str, action: str) -> Path:
    return Path(str(audits["manifests"][model][lambda_key][f"{fold}:{action}"]))


def _record_prediction(record: Mapping[str, Any], *, manifest: Mapping[str, Any]) -> np.ndarray:
    """Evaluate one training-membership record, including M6 components."""
    if not {"shared_parameters", "language_coefficients"} <= set(record):
        raise M5M6Error("derived start lacks complete coefficients")
    problem = load_problem(manifest)
    shared = SharedParameters(**record["shared_parameters"])
    by_language = {str(item["language"]): item for item in record["language_coefficients"]}
    if not set(problem.language_ids) <= set(by_language):
        raise M5M6Error("derived start lacks a fitted language coefficient")
    nuisance = [by_language[str(language)] for language in problem.language_ids]
    kwargs: dict[str, Any] = {}
    if manifest["fit"]["model"] == "M6":
        kwargs = {"D_family": problem.D_family, "D_script": problem.D_script,
            "D_other": problem.D_other}
    return predict(manifest["fit"]["model"], shared, problem.N, problem.D,
        problem.E, problem.K, [item["L_inf"] for item in nuisance],
        [item["A"] for item in nuisance], [item["B"] for item in nuisance], **kwargs)


def _bound_scaled_rank(prediction_jacobian: Any, bounds: Sequence[OuterBound], *,
        relative_threshold: float = 1e-8) -> dict[str, Any]:
    """Return the numerical rank after scaling columns by bound width."""
    jacobian = np.asarray(prediction_jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[1] != len(bounds) or jacobian.shape[0] < 1 \
            or not np.isfinite(jacobian).all():
        raise M5M6Error("prediction Jacobian must be a finite row-by-bound matrix")
    if not math.isfinite(relative_threshold) or not 0.0 < relative_threshold < 1.0:
        raise M5M6Error("relative SVD threshold must lie strictly between zero and one")
    widths = np.asarray([item.upper - item.lower for item in bounds], dtype=np.float64)
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise M5M6Error("outer bound widths must be finite and positive")
    try:
        singular_values = np.linalg.svd(jacobian * widths[None, :],
            compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise M5M6Error("prediction Jacobian SVD did not converge") from error
    leading = float(singular_values[0]) if singular_values.size else 0.0
    absolute_threshold = float(relative_threshold * leading)
    rank = int(np.sum(singular_values > absolute_threshold)) if leading > 0.0 else 0
    return {
        "rows": int(jacobian.shape[0]), "columns": int(jacobian.shape[1]),
        "bound_widths": widths.tolist(), "singular_values": singular_values.tolist(),
        "relative_svd_threshold": float(relative_threshold),
        "absolute_svd_threshold": absolute_threshold, "rank": rank,
        "full_rank": rank == jacobian.shape[1],
    }


def _shared_prediction_jacobian_rank(record: Mapping[str, Any], *,
        manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Audit local shared-coordinate identifiability at the selected fit."""
    problem = load_problem(manifest)
    coordinates = np.asarray(record.get("coordinates"), dtype=np.float64)
    model = str(manifest["fit"]["model"])
    kwargs: dict[str, Any] = {}
    if model == "M6":
        kwargs = {"D_family": problem.D_family, "D_script": problem.D_script,
            "D_other": problem.D_other}
    _, _, capacity_jacobian, data_jacobian = predictor_terms_and_jacobian(
        model, coordinates, problem.N, problem.D, problem.E, problem.K, **kwargs)
    by_language = {str(item["language"]): item for item in record["language_coefficients"]}
    if not set(problem.language_ids) <= set(by_language):
        raise M5M6Error("rank audit lacks a fitted language coefficient")
    coefficient_a = np.asarray([by_language[language]["A"]
        for language in problem.language_ids], dtype=np.float64)
    coefficient_b = np.asarray([by_language[language]["B"]
        for language in problem.language_ids], dtype=np.float64)
    prediction_jacobian = (coefficient_a[:, None] * np.asarray(capacity_jacobian, dtype=np.float64)
        + coefficient_b[:, None] * np.asarray(data_jacobian, dtype=np.float64))
    bounds = tuple(OuterBound(**item) for item in manifest["fit"]["bounds"])
    result = _bound_scaled_rank(prediction_jacobian, bounds)
    result["required_rank"] = len(outer_parameter_names(model))
    result["full_rank"] = result["rank"] == result["required_rank"]
    return result


def audit(screen_index_path: Path, audit_index_path: Path) -> dict[str, Any]:
    screen = _index(screen_index_path, schemas=(INDEX_SCHEMA,))
    audits = _index(audit_index_path, schemas=(AUDIT_INDEX_SCHEMA,))
    if audits.get("screen_index", {}).get("index_sha256") != screen["index_sha256"]:
        raise M5M6Error("audit index belongs to a different screen")
    historical_screen = _wave_index(Path(str(screen["interaction_screen_index"]["path"])))
    items = []
    for model, lambda_key, fold, manifest_path in _iter_manifest_items(screen):
        base_result, base = _campaign_result(manifest_path, screen)
        basin_manifest = load_manifest(_audit_manifest_path(audits, model, lambda_key, fold, "basin"))
        direct_manifest = load_manifest(_audit_manifest_path(audits, model, lambda_key, fold, "direct"))
        basin_result, _ = _reduced_current_exploratory(result_path(basin_manifest))
        direct_result, _ = _reduced_current_exploratory(result_path(direct_manifest))
        retained = base_result["selected"]
        retained_prediction = _record_prediction(retained, manifest=base)
        equivalent = 0
        for record in _records(basin_manifest):
            if not record.get("success") or not isinstance(record.get("objective"), (int, float)):
                continue
            check = equivalence_check(float(retained["objective"]), float(record["objective"]),
                retained_prediction, _record_prediction(record, manifest=basin_manifest),
                objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
                prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)
            equivalent += int(check.equivalent)
        direct = direct_result.get("selected", {}).get("direct_equivalence", {})
        contacts = list(retained.get("boundary_contacts", []))
        allowed = [item for item in contacts if model == "M6"
            and item.get("name") in {"eta_family", "eta_script", "gamma_other"}
            and item.get("side") == "lower"]
        rejected = [item for item in contacts if item not in allowed]
        parameters = SharedParameters(**retained["shared_parameters"])
        alpha, beta, _, _ = effective_exponents(parameters,
            np.asarray([1.0, 4.0, 16.0, 32.0])[:, None],
            np.asarray([1.0, 12.0, 50.0, 200.0])[None, :], model=model)
        jacobian_rank = _shared_prediction_jacobian_rank(retained, manifest=base)
        zero_lambda_equivalence = None
        if model == "M5" and lambda_key == "0":
            predecessor, predecessor_manifest, _ = _historical_m2kn_result(historical_screen, fold)
            if predecessor_manifest.get("membership", {}).get("membership_sha256") \
                    != base.get("membership", {}).get("membership_sha256"):
                raise M5M6Error("zero-lambda M5 and M2-KN memberships differ")
            zero_check = equivalence_check(
                float(predecessor["selected"]["objective"]), float(retained["objective"]),
                _record_prediction(predecessor["selected"], manifest=predecessor_manifest),
                retained_prediction, objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
                prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)
            zero_lambda_equivalence = {
                "equivalent": zero_check.equivalent,
                "relative_objective_difference": zero_check.relative_objective_difference,
                "maximum_absolute_prediction_difference": zero_check.maximum_absolute_prediction_difference,
                "m2_kn_result_sha256": predecessor["result_sha256"],
            }
        healthy = equivalent >= MINIMUM_EQUIVALENT_BASIN_STARTS \
            and bool(direct.get("resolved")) and not rejected \
            and float(np.min(alpha)) > 0.0 and float(np.min(beta)) > 0.0 \
            and bool(jacobian_rank["full_rank"]) \
            and (zero_lambda_equivalence is None or bool(zero_lambda_equivalence["equivalent"]))
        items.append({
            "model": model, "lambda": float(lambda_key), "lambda_key": lambda_key,
            "fold": fold, "status": "healthy" if healthy else "failed",
            "equivalent_basin_starts": equivalent, "direct_equivalence": direct,
            "allowed_nested_null_contacts": allowed, "rejected_boundary_contacts": rejected,
            "minimum_alpha": float(np.min(alpha)), "minimum_beta": float(np.min(beta)),
            "bound_scaled_shared_prediction_jacobian": jacobian_rank,
            "zero_lambda_m2_kn_equivalence": zero_lambda_equivalence,
            "base_result_sha256": base_result["result_sha256"],
            "basin_result_sha256": basin_result["result_sha256"],
            "direct_result_sha256": direct_result["result_sha256"],
        })
    payload = _seal({
        "schema_version": AUDIT_SCHEMA, "status": "complete", "exploratory": True,
        "screen_index_sha256": screen["index_sha256"],
        "audit_index_sha256": audits["index_sha256"],
        "all_candidates_healthy": all(item["status"] == "healthy" for item in items),
        "items": items,
    }, "audit_sha256")
    path = Path(str(screen["output_root"])) / "reports" / "screen_numerical_audit.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def _component_frame(index: Mapping[str, Any]) -> Any:
    receipt_value = index.get("component_receipt", {})
    receipt = _sealed(read_json(Path(str(receipt_value.get("path")))), "receipt_sha256")
    if receipt.get("receipt_sha256") != receipt_value.get("receipt_sha256"):
        raise M5M6Error("index component receipt binding changed")
    import pandas as pd

    frame = pd.read_parquet(Path(str(receipt["snapshot"]["path"])))
    if component_content_sha256(frame) != receipt["snapshot"]["component_content_sha256"]:
        raise M5M6Error("component frame content changed")
    return frame


def _metric(frame: Any, membership: Mapping[str, Any], selected: Mapping[str, Any],
        model: str) -> dict[str, Any]:
    rows = frame[frame["trajectory_id"].isin(membership["test_trajectory_ids"])].copy()
    if membership["test_readout_ids"]:
        rows = rows[rows["readout_id"].isin(membership["test_readout_ids"])]
    if membership["score_languages"]:
        rows = rows[rows["language"].isin(membership["score_languages"])]
    coefficients = {str(item["language"]): item for item in selected["language_coefficients"]}
    if rows.empty or not set(rows["language"].astype(str)) <= set(coefficients):
        raise M5M6Error("held-out rows are empty or lack nuisance coefficients")
    nuisance = [coefficients[str(language)] for language in rows["language"]]
    kwargs: dict[str, Any] = {}
    if model == "M6":
        kwargs = {name: rows[name] for name in COMPONENT_COLUMNS}
    prediction = predict(model, SharedParameters(**selected["shared_parameters"]),
        rows["N_act"], rows["D_t"], rows["E"], rows["K"],
        [item["L_inf"] for item in nuisance], [item["A"] for item in nuisance],
        [item["B"] for item in nuisance], **kwargs)
    errors = trajectory_mean_squared_errors(rows["adjusted_loss"], prediction, rows["trajectory_id"])
    return {
        "trajectory_mse": errors,
        "trajectory_balanced_rmse": trajectory_balanced_rmse(
            rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "trajectory_balanced_r2": trajectory_balanced_r2(
            rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "rows": int(len(rows)), "trajectories": int(rows["trajectory_id"].nunique()),
    }


def build_report(screen_index_path: Path, audit_path: Path) -> dict[str, Any]:
    screen = _index(screen_index_path, schemas=(INDEX_SCHEMA,))
    audit_payload = _sealed(read_json(audit_path), "audit_sha256")
    if audit_payload.get("schema_version") != AUDIT_SCHEMA \
            or audit_payload.get("status") != "complete" \
            or audit_payload.get("screen_index_sha256") != screen["index_sha256"]:
        raise M5M6Error("numerical audit is incomplete or mismatched")
    frame = _component_frame(screen)
    official = _official_report(Path(str(screen["official_candidate_report"]["path"])))
    errors: dict[str, dict[str, dict[str, float]]] = {"M2": {}}
    metrics: dict[str, dict[str, Any]] = {"M1": {}, "M2": {}}
    for baseline in ("M1", "M2"):
        for fold in PRIMARY_FOLDS:
            metric = official["primary_metrics"][baseline][fold]
            metrics[baseline][fold] = metric
            if baseline == "M2":
                errors["M2"][fold] = {str(key): float(value)
                    for key, value in metric["trajectory_mse"].items()}
    candidate_order = []
    candidate_metrics: dict[str, dict[str, Any]] = {}
    source_results = []
    for model in MODELS:
        for strength in RIDGE_LAMBDAS:
            lambda_key = _lambda_key(strength)
            candidate = f"{model}@lambda={lambda_key}"
            candidate_order.append(candidate)
            errors[candidate] = {}
            candidate_metrics[candidate] = {}
            for fold in PRIMARY_FOLDS:
                manifest_path = Path(str(screen["manifests"][model][lambda_key][fold]))
                result, manifest = _campaign_result(manifest_path, screen)
                metric = _metric(frame, manifest["membership"], result["selected"], model)
                candidate_metrics[candidate][fold] = metric
                errors[candidate][fold] = metric["trajectory_mse"]
                source_results.append({
                    "candidate": candidate, "model": model, "lambda": strength, "fold": fold,
                    "result_path": str(result_path(manifest)), "result_sha256": result["result_sha256"],
                    "selected_coordinates": result["selected"]["coordinates"],
                    "shared_parameters": result["selected"]["shared_parameters"],
                    "objective_components": result["selected"].get("objective_components"),
                })
    selection = select_model_paired_one_standard_error(errors,
        model_order=("M2", *candidate_order), fold_order=PRIMARY_FOLDS,
        replicates=PAIRED_BOOTSTRAP_REPLICATES)
    health = {(str(item["model"]), str(item["lambda_key"]), str(item["fold"])):
        item["status"] == "healthy" for item in audit_payload["items"]}
    expected_health = {(model, _lambda_key(strength), fold)
        for model in MODELS for strength in RIDGE_LAMBDAS for fold in PRIMARY_FOLDS}
    if set(health) != expected_health or len(audit_payload["items"]) != len(expected_health):
        raise M5M6Error("numerical audit does not cover every model/lambda/fold cell exactly once")
    gates = {}
    passing = []
    for candidate in candidate_order:
        model, lambda_key = candidate.split("@lambda=", maxsplit=1)
        aggregate = float(selection.observed_aggregate_rmse[candidate])
        k200 = float(candidate_metrics[candidate]["K200_common_languages"]["trajectory_balanced_rmse"])
        healthy = all(health.get((model, lambda_key, fold), False) for fold in PRIMARY_FOLDS)
        item = {
            "observed_equal_fold_rmse": aggregate,
            "equal_fold_rmse_below_m2": aggregate < EQUAL_FOLD_M2_GATE,
            "k200_rmse": k200, "k200_rmse_below_m1": k200 < K200_M1_GATE,
            "all_three_folds_numerically_healthy": healthy,
        }
        item["passes"] = all((item["equal_fold_rmse_below_m2"],
            item["k200_rmse_below_m1"], healthy))
        gates[candidate] = item
        if item["passes"]:
            passing.append(candidate)
    winner = None if not passing else min(passing,
        key=lambda name: (selection.observed_aggregate_rmse[name], candidate_order.index(name)))
    advance = winner is not None
    decision = "advance_one_candidate_to_separately_frozen_confirmation" if advance \
        else "retain_registered_m2_and_stop"
    payload = _seal({
        "schema_version": REPORT_SCHEMA, "status": "complete", "exploratory": True,
        "screen_index_sha256": screen["index_sha256"],
        "audit_sha256": audit_payload["audit_sha256"],
        "parameter_counts": {"M2": 10, "M5": 11, "M6": 14},
        "locked_gates": {"equal_fold_rmse_below_m2": EQUAL_FOLD_M2_GATE,
            "k200_rmse_below_m1": K200_M1_GATE, "comparison": "strict"},
        "baseline_metrics": metrics, "candidate_metrics": candidate_metrics,
        "paired_one_standard_error_selection": selection.to_dict(),
        "candidate_gates": gates, "passing_candidates": passing,
        "selected_candidate": winner, "advance_nested_confirmation": advance,
        "decision": decision, "source_results": source_results,
        "remaining_confirmation_work": (
            "No nested manifests are emitted by this screen facade. A pass authorizes, but does not "
            "constitute, a separately frozen leakage-controlled inner/outer or prospective campaign."),
        "claim_boundary": (
            "Retrospective same-grid model development on already-inspected folds; registered M2 "
            "remains authoritative regardless of this screen outcome."),
    }, "report_sha256")
    root = Path(str(screen["output_root"]))
    json_path = root / "reports" / "screen_report.json"
    write_json_once(json_path, payload)
    lines = ["# Exploratory M5/M6 composition screen", "", f"Decision: **{decision}**", "",
        f"Selected gate-passing candidate: **{winner or 'none'}**", "",
        "| Candidate | Parameters | Equal-fold RMSE | K200 RMSE | Healthy | Pass |",
        "|---|---:|---:|---:|---|---|"]
    for candidate in candidate_order:
        model = candidate.split("@", maxsplit=1)[0]
        item = gates[candidate]
        lines.append(f"| {candidate} | {payload['parameter_counts'][model]} | "
            f"{item['observed_equal_fold_rmse']:.8f} | {item['k200_rmse']:.8f} | "
            f"{item['all_three_folds_numerically_healthy']} | {item['passes']} |")
    lines.extend(["", payload["claim_boundary"], "", payload["remaining_confirmation_work"], ""])
    markdown_path = root / "reports" / "screen_report.md"
    write_text_once(markdown_path, "\n".join(lines))
    return {"path": str(json_path), "markdown_path": str(markdown_path), **payload}
