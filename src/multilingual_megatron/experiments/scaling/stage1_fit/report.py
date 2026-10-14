"""Small, fail-closed Stage 1 fit reports; secondary diagnostics never select the law."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .campaign import CampaignError, GENERALIZATION_MODELS, _generalization_memberships, scientific_result
from .data import (RESAMPLING_MANIFEST_SCHEMA_VERSION, UNCERTAINTY_REPLICATES,
    UNCERTAINTY_SUBSAMPLE_SEED, UNCERTAINTY_SUBSET_TRAJECTORIES, Stage1FitDataError,
    canonical_json_sha256, load_fit_rows_snapshot, sha256_file)
from .execution import ExecutionError, bind_membership, load_manifest, read_json, validate_manifest, write_json_once, write_text_once
from .fit import (JOINT_MOE_E_MAX, JOINT_MOE_E_START, SharedParameters, d16_adaptive_tiers,
    data_column, outer_parameter_names, predict)
from .selection import (PAIRED_BOOTSTRAP_REPLICATES, select_model_paired_one_standard_error, trajectory_balanced_r2,
    trajectory_balanced_rmse, trajectory_mean_squared_errors)

REPORT_SCHEMA = "stage1_fit_candidate_report/v5"
SECONDARY_REPORT_SCHEMA = "stage1_fit_secondary_diagnostics/v1"
D17_LITERATURE_REPORT_SCHEMA = "stage1_fit_d17_literature/v2"
LOGICAL_COMPUTE_CONTRACT_SCHEMA = "stage1_fit_logical_compute_contract/v1"
LOGICAL_COMPUTE_FORMULA = "F=6*N_act*D_total"
GENERALIZATION_REPORT_SCHEMA = "stage1_fit_generalization_battery/v1"
D16_REPORT_SCHEMA = "stage1_fit_d16_report/v4"
FINAL_REPORT_SCHEMA = "stage1_fit_final_report/v1"
D17_BLOCK_REASON = "the remaining analysis-choice sensitivities in D17 are not executable"


class ReportError(RuntimeError):
    pass

def _snapshot(input_receipt_path: Path) -> tuple[Any, Mapping[str, Any], dict[str, str]]:
    receipt = read_json(input_receipt_path)
    try:
        frame = load_fit_rows_snapshot(receipt)
    except Stage1FitDataError as error:
        raise ReportError(str(error)) from error
    binding = {"path": str(input_receipt_path.resolve()),
        "file_sha256": sha256_file(input_receipt_path)}
    return frame, receipt, binding


def _result(path: Path) -> Mapping[str, Any]:
    try:
        return scientific_result(path)
    except CampaignError as error:
        raise ReportError(str(error)) from error


def _prediction(frame: Any, membership: Mapping[str, Any], selected: Mapping[str, Any], model: str,
    *, split: str = "test") -> tuple[Any, np.ndarray]:
    rows = frame[frame["trajectory_id"].isin(membership[f"{split}_trajectory_ids"])].copy()
    readouts = membership[f"{split}_readout_ids"]
    if readouts:
        rows = rows[rows["readout_id"].isin(readouts)]
    if membership["score_languages"]:
        rows = rows[rows["language"].isin(membership["score_languages"])]
    if rows.empty:
        raise ReportError(f"declared {split} membership has no scored rows")
    coefficients = {item["language"]: item for item in selected["language_coefficients"]}
    if not set(rows["language"].astype(str)).issubset(coefficients):
        raise ReportError(f"{split} language lacks a fitted nuisance coefficient")
    shared = SharedParameters(**selected["shared_parameters"])
    nuisance = [coefficients[str(language)] for language in rows["language"]]
    prediction = predict(model, shared, rows["N_act"], rows[data_column(model)], rows["E"], rows["K"],
        [item["L_inf"] for item in nuisance], [item["A"] for item in nuisance], [item["B"] for item in nuisance])
    return rows, np.asarray(prediction, dtype=np.float64)


def _metrics(frame: Any, membership: Mapping[str, Any], selected: Mapping[str, Any], model: str,
    *, split: str = "test") -> dict[str, Any]:
    rows, prediction = _prediction(frame, membership, selected, model, split=split)
    return {"trajectory_mse": trajectory_mean_squared_errors(rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "trajectory_balanced_rmse": trajectory_balanced_rmse(rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "trajectory_balanced_r2": trajectory_balanced_r2(rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "rows": int(len(rows)), "trajectories": int(rows["trajectory_id"].nunique())}


def _validate_result(*, result: Mapping[str, Any], input_sha256: str, membership: Mapping[str, Any], model: str,
    fold_id: str) -> None:
    identity = result.get("fit_identity", {})
    if result.get("fit_rows_content_sha256") != input_sha256:
        raise ReportError("result uses different fit rows")
    if result.get("membership", {}).get("membership_sha256") != membership["membership_sha256"]:
        raise ReportError("result uses different frozen membership")
    if (identity.get("model"), identity.get("fold_id")) != (model, fold_id):
        raise ReportError("result belongs to a different model or fold")
    parents = result.get("parents")
    if not isinstance(parents, list) or {item.get("kind") for item in parents if isinstance(item, Mapping)} < {"scale_m0_ols"}:
        raise ReportError("candidate result lacks its validated M0-OLS scale parent")


def _holdout(path: Path, input_sha256: str) -> Mapping[str, Any]:
    holdouts = read_json(path); unsigned = dict(holdouts); digest = unsigned.pop("manifest_sha256", None)
    if digest != canonical_json_sha256(unsigned) or holdouts.get("input_sha256") != input_sha256:
        raise ReportError("holdout manifest hash or input identity changed")
    return holdouts


def build_candidate_report(*, input_receipt_path: Path, holdouts_path: Path,
    results: Mapping[str, Mapping[str, Path]], bootstrap_replicates: int = PAIRED_BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    """Use only the three primary folds for the paired 10,000-resample selection."""
    if bootstrap_replicates != PAIRED_BOOTSTRAP_REPLICATES:
        raise ReportError("definitive candidate selection requires exactly 10,000 paired resamples")
    frame, receipt, input_binding = _snapshot(input_receipt_path)
    input_sha256 = str(receipt["fit_rows_content_sha256"]); holdouts = _holdout(holdouts_path, input_sha256)
    primary = {item["fold_id"]: bind_membership(holdouts_path, kind="primary", membership_id=item["fold_id"])
        for item in holdouts["primary_folds"]}
    errors: dict[str, dict[str, dict[str, float]]] = {}; metrics: dict[str, dict[str, Any]] = {}; sources = []
    for model in ("M0", "M1", "M2"):
        if set(results.get(model, {})) != set(primary):
            raise ReportError(f"{model} primary results are incomplete")
        errors[model] = {}; metrics[model] = {}
        for fold_id, membership in primary.items():
            path = results[model][fold_id]; reduced = _result(path)
            _validate_result(result=reduced, input_sha256=input_sha256, membership=membership, model=model, fold_id=fold_id)
            metric = _metrics(frame, membership, reduced["selected"], model)
            errors[model][fold_id] = metric["trajectory_mse"]; metrics[model][fold_id] = metric
            sources.append({"role": "primary", "model": model, "fold_id": fold_id, "path": str(path),
                "sha256": reduced["result_sha256"]})
    selection = select_model_paired_one_standard_error(errors, fold_order=tuple(primary), replicates=bootstrap_replicates)
    unsigned = {"schema_version": REPORT_SCHEMA, "input_sha256": input_sha256, "input_receipt": input_binding,
        "holdout_manifest_sha256": holdouts["manifest_sha256"], "source_results": sources,
        "primary_metrics": metrics, "selection": selection.to_dict(),
        "secondary_diagnostics": {"status": "publish_as_separate_post_selection_artifact"},
        "all_data_refit": {"status": "plan_after_candidate_selection", "selected_model": selection.selected_model},
        "d16_status": "deferred_until_candidate_selection", "d17_status": "blocked", "d17_block_reason": D17_BLOCK_REASON}
    return {**unsigned, "report_sha256": canonical_json_sha256(unsigned)}


def write_candidate_report(output_root: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    unsigned = dict(report); digest = unsigned.pop("report_sha256", None)
    if digest != canonical_json_sha256(unsigned):
        raise ReportError("candidate report hash differs")
    json_path = output_root / "reports" / "candidate_selection.json"; write_json_once(json_path, report)
    selection = report["selection"]
    lines = ["# Stage 1 candidate selection", "", f"Selected model: **{selection['selected_model']}**", "",
        "Primary validation uses only the three frozen trajectory holdouts and the paired 10,000-resample one-SE rule.",
        "Secondary random, terminal-continuation, and lowest-loss diagnostics are reported separately and do not select a model.",
        "D16 execution is deferred until this candidate result is accepted.", f"D17 is blocked: {report['d17_block_reason']}", ""]
    markdown = "\n".join(lines); path = output_root / "reports" / "fit_summary.md"; write_text_once(path, markdown)
    return json_path, path


def _sealed_report(path: Path, schema: str) -> Mapping[str, Any]:
    payload = read_json(path); unsigned = dict(payload); digest = unsigned.pop("report_sha256", None)
    if payload.get("schema_version") != schema or digest != canonical_json_sha256(unsigned):
        raise ReportError(f"invalid sealed report: {path}")
    return payload

def _selection_bound(result: Mapping[str, Any], candidate_path: Path,
    *, candidate: Mapping[str, Any]) -> None:
    binding = result.get("selection_report")
    expected = {"path": str(candidate_path.resolve()), "file_sha256": sha256_file(candidate_path),
        "report_sha256": candidate["report_sha256"]}
    if binding != expected:
        raise ReportError("post-selection result is not bound to this candidate report")

def build_secondary_diagnostics_report(*, candidate_report_path: Path, input_receipt_path: Path,
    holdouts_path: Path, selected_results: Mapping[str, Path], lowest_results: Mapping[str, Path]) -> dict[str, Any]:
    """Publish immutable D15 secondary diagnostics without rewriting candidate selection."""
    candidate = _sealed_report(candidate_report_path, REPORT_SCHEMA)
    frame, receipt, input_binding = _snapshot(input_receipt_path)
    input_sha256 = str(receipt["fit_rows_content_sha256"])
    if (candidate.get("input_sha256") != input_sha256
            or candidate.get("input_receipt") != input_binding):
        raise ReportError("secondary diagnostics use a different input snapshot")
    holdouts = _holdout(holdouts_path, input_sha256)
    if candidate.get("holdout_manifest_sha256") != holdouts["manifest_sha256"]:
        raise ReportError("secondary diagnostics use a different holdout manifest")
    selected_model = str(candidate["selection"]["selected_model"])
    if set(selected_results) != {"random", "terminal"}:
        raise ReportError("secondary diagnostics require random and terminal selected-model results")
    views = {"random": bind_membership(holdouts_path, kind="random", membership_id="random_13_of_64"),
        "terminal": bind_membership(holdouts_path, kind="terminal")}
    selected_metrics: dict[str, Any] = {}; sources: list[dict[str, Any]] = []
    for name, membership in views.items():
        result = _result(selected_results[name])
        _validate_result(result=result, input_sha256=input_sha256, membership=membership,
            model=selected_model, fold_id=name)
        _selection_bound(result, candidate_report_path, candidate=candidate)
        selected_metrics[name] = _metrics(frame, membership, result["selected"], selected_model)
        sources.append({"role": name, "model": selected_model, "path": str(selected_results[name]),
            "sha256": result["result_sha256"]})
    if set(lowest_results) != {"M0", "M1", "M2"}:
        raise ReportError("lowest-loss sensitivity requires M0, M1, and M2 results")
    lowest_metrics: dict[str, Any] = {}
    membership = bind_membership(holdouts_path, kind="lowest", membership_id="lowest_terminal_macro_loss_30")
    for model in ("M0", "M1", "M2"):
        result = _result(lowest_results[model])
        _validate_result(result=result, input_sha256=input_sha256, membership=membership, model=model, fold_id="lowest")
        _selection_bound(result, candidate_report_path, candidate=candidate)
        training = _metrics(frame, membership, result["selected"], model, split="train")
        extrapolation = _metrics(frame, membership, result["selected"], model)
        lowest_metrics[model] = {"training": training, "extrapolation": extrapolation,
            "training_plus_extrapolation_rmse": training["trajectory_balanced_rmse"] + extrapolation["trajectory_balanced_rmse"]}
        sources.append({"role": "lowest", "model": model, "path": str(lowest_results[model]),
            "sha256": result["result_sha256"]})
    unsigned = {"schema_version": SECONDARY_REPORT_SCHEMA,
        "source_candidate": {"path": str(candidate_report_path.resolve()), "file_sha256": sha256_file(candidate_report_path),
            "report_sha256": candidate["report_sha256"]},
        "input_sha256": input_sha256, "holdout_manifest_sha256": holdouts["manifest_sha256"],
        "selected_model_diagnostics": selected_metrics, "lowest_loss_all_candidates": lowest_metrics,
        "source_results": sources, "role": "secondary_not_used_for_candidate_selection"}
    return {**unsigned, "report_sha256": canonical_json_sha256(unsigned)}

def write_secondary_diagnostics_report(output_root: Path, report: Mapping[str, Any]) -> Path:
    unsigned = dict(report); digest = unsigned.pop("report_sha256", None)
    if report.get("schema_version") != SECONDARY_REPORT_SCHEMA or digest != canonical_json_sha256(unsigned):
        raise ReportError("secondary diagnostics report hash differs")
    path = output_root / "reports" / "secondary_diagnostics.json"
    write_json_once(path, report); return path


def _literature_diagnostic(result: Mapping[str, Any], model: str) -> dict[str, Any]:
    selected = result["selected"]
    path = Path(str(result["manifest_path"]))
    try:
        manifest = load_manifest(path)
    except ExecutionError:
        manifest = load_manifest(path, require_current_provenance=False)
    languages = len(selected["language_coefficients"]); shared = len(outer_parameter_names(model))
    attempts = result.get("attempts", []); successful = sum(item.get("success") is True for item in attempts)
    return {"objective": float(selected["objective"]), "shared_parameter_count": shared,
        "language_nuisance_parameter_count": 3 * languages, "total_parameter_count": shared + 3 * languages,
        "languages": languages, "start_count": int(manifest["fit"]["start_count"]),
        "successful_starts": successful, "failed_starts": len(attempts) - successful,
        "post_grid_audit": "accepted_by_scientific_result",
        "bound_version": int(manifest["fit"]["bound_version"]),
        "boundary_contacts": list(selected.get("boundary_contacts", []))}


def _scale_binding(result: Mapping[str, Any]) -> Mapping[str, Any]:
    parents = [item for item in result.get("parents", [])
        if isinstance(item, Mapping) and item.get("kind") == "scale_m0_ols"]
    if len(parents) != 1:
        raise ReportError("literature result lacks one exact M0-OLS scale parent")
    return parents[0]


def _logical_compute_contract(path: Path) -> tuple[Mapping[str, Any], dict[str, str]]:
    contract = read_json(path); unsigned = dict(contract); digest = unsigned.pop("contract_sha256", None)
    expected = {"schema_version", "logical_compute", "physical_compute", "admissible_domain",
        "budgets", "prediction_points"}
    if (set(unsigned) != expected or contract.get("schema_version") != LOGICAL_COMPUTE_CONTRACT_SCHEMA
            or digest != canonical_json_sha256(unsigned)):
        raise ReportError("logical-compute contract is malformed or unsealed")
    if contract["logical_compute"] != {"formula": LOGICAL_COMPUTE_FORMULA, "unit": "FLOPs"}:
        raise ReportError("logical-compute formula changed")
    if contract["physical_compute"] != "excluded_separate_measured_h100_ledger":
        raise ReportError("logical report must not contain a physical-compute ledger")
    domain = contract["admissible_domain"]
    if not isinstance(domain, Mapping) or set(domain) != {"label", "N_act", "D_total", "E", "K"}:
        raise ReportError("logical-compute admissible domain is malformed")
    if not isinstance(domain["label"], str) or not domain["label"].strip():
        raise ReportError("logical-compute domain needs a label")
    ranges: dict[str, tuple[float, float]] = {}
    for name in ("N_act", "D_total"):
        value = domain[name]
        if not isinstance(value, Mapping) or set(value) != {"minimum", "maximum"}:
            raise ReportError(f"{name} domain is malformed")
        lower, upper = float(value["minimum"]), float(value["maximum"])
        if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 < lower <= upper:
            raise ReportError(f"{name} domain is invalid")
        ranges[name] = (lower, upper)
    discrete: dict[str, tuple[int, ...]] = {}
    for name in ("E", "K"):
        values = domain[name]
        if (not isinstance(values, list) or not values or any(isinstance(item, bool) for item in values)
                or any(int(item) != item or int(item) < 1 for item in values)):
            raise ReportError(f"{name} domain is invalid")
        normalized = tuple(int(item) for item in values)
        if tuple(sorted(set(normalized))) != normalized:
            raise ReportError(f"{name} domain must be sorted and unique")
        discrete[name] = normalized
    budgets = contract["budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise ReportError("logical-compute contract needs budgets")
    budget_ids: set[str] = set()
    for budget in budgets:
        if not isinstance(budget, Mapping) or set(budget) != {"budget_id", "maximum_flops"}:
            raise ReportError("logical-compute budget is malformed")
        identifier = budget["budget_id"]; maximum = float(budget["maximum_flops"])
        if (not isinstance(identifier, str) or not identifier or identifier in budget_ids
                or not math.isfinite(maximum) or maximum <= 0.0):
            raise ReportError("logical-compute budget is invalid")
        budget_ids.add(identifier)
    points = contract["prediction_points"]
    if not isinstance(points, list) or not points:
        raise ReportError("logical-compute contract needs prediction points")
    point_ids: set[str] = set()
    for point in points:
        if not isinstance(point, Mapping) or set(point) != {"point_id", "N_act", "D_total", "E", "K", "D_t_by_language"}:
            raise ReportError("logical-compute prediction point is malformed")
        identifier = point["point_id"]; n_act = float(point["N_act"]); d_total = float(point["D_total"])
        experts, languages = point["E"], point["K"]
        if (not isinstance(identifier, str) or not identifier or identifier in point_ids
                or isinstance(experts, bool) or isinstance(languages, bool)
                or int(experts) != experts or int(languages) != languages):
            raise ReportError("logical-compute prediction point identity is invalid")
        point_ids.add(identifier); experts = int(experts); languages = int(languages)
        if (not ranges["N_act"][0] <= n_act <= ranges["N_act"][1]
                or not ranges["D_total"][0] <= d_total <= ranges["D_total"][1]
                or experts not in discrete["E"] or languages not in discrete["K"]):
            raise ReportError("logical-compute prediction point is outside the admissible domain")
        d_t = point["D_t_by_language"]
        if (not isinstance(d_t, Mapping) or len(d_t) != languages
                or any(not isinstance(language, str) or not language for language in d_t)
                or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in d_t.values())
                or not math.isclose(sum(float(value) for value in d_t.values()), d_total, rel_tol=1e-9, abs_tol=0.0)):
            raise ReportError("prediction point D_t values must be positive and sum to D_total")
    if any(not any(int(point["E"]) == expert
            and 6.0 * float(point["N_act"]) * float(point["D_total"]) <= float(budget["maximum_flops"])
            for point in points) for budget in budgets for expert in discrete["E"]):
        raise ReportError("every logical-compute budget and expert count needs an admissible prediction point")
    binding = {"path": str(path.resolve()), "file_sha256": sha256_file(path),
        "contract_sha256": str(digest)}
    return contract, binding


def _logical_compute_predictions(all_data: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model, result in all_data.items():
        selected = result["selected"]; coefficients = {item["language"]: item for item in selected["language_coefficients"]}
        point_predictions = []
        for point in contract["prediction_points"]:
            languages = tuple(sorted(str(item) for item in point["D_t_by_language"]))
            if not set(languages) <= set(coefficients):
                raise ReportError(f"{model} all-data fit lacks a declared prediction language")
            nuisance = [coefficients[language] for language in languages]
            data = ([float(point["D_total"])] * len(languages) if data_column(model) == "D_total"
                else [float(point["D_t_by_language"][language]) for language in languages])
            values = predict(model, SharedParameters(**selected["shared_parameters"]),
                np.full(len(languages), float(point["N_act"])), data,
                np.full(len(languages), int(point["E"])), np.full(len(languages), int(point["K"])),
                [item["L_inf"] for item in nuisance], [item["A"] for item in nuisance],
                [item["B"] for item in nuisance])
            point_predictions.append({"point_id": point["point_id"],
                "N_act": float(point["N_act"]), "D_total": float(point["D_total"]),
                "E": int(point["E"]), "K": int(point["K"]),
                "logical_flops": 6.0 * float(point["N_act"]) * float(point["D_total"]),
                "macro_adjusted_loss": float(np.mean(values)),
                "per_language_adjusted_loss": {language: float(value) for language, value in zip(languages, values)}})
        optima = []; optima_by_expert = []
        for budget in contract["budgets"]:
            eligible = [item for item in point_predictions if item["logical_flops"] <= float(budget["maximum_flops"])]
            best = min(eligible, key=lambda item: (item["macro_adjusted_loss"], item["point_id"]))
            optima.append({"budget_id": budget["budget_id"], "maximum_flops": float(budget["maximum_flops"]),
                "point_id": best["point_id"], "logical_flops": best["logical_flops"],
                "macro_adjusted_loss": best["macro_adjusted_loss"]})
            for expert in contract["admissible_domain"]["E"]:
                expert_eligible = [item for item in eligible if item["E"] == int(expert)]
                expert_best = min(expert_eligible,
                    key=lambda item: (item["macro_adjusted_loss"], item["point_id"]))
                optima_by_expert.append({"budget_id": budget["budget_id"],
                    "maximum_flops": float(budget["maximum_flops"]), "E": int(expert),
                    "point_id": expert_best["point_id"], "logical_flops": expert_best["logical_flops"],
                    "macro_adjusted_loss": expert_best["macro_adjusted_loss"]})
        output[model] = {"all_data_result_sha256": result["result_sha256"],
            "point_predictions": point_predictions, "optima_overall": optima,
            "optima_by_expert": optima_by_expert}
    return {"scope": "discrete_minimum_over_frozen_declared_points",
        "physical_compute_ledger": "excluded", "models": output}


def build_literature_comparison_report(*, candidate_report_path: Path, input_receipt_path: Path,
    holdouts_path: Path, logical_compute_contract_path: Path,
    results: Mapping[str, Mapping[str, Path]]) -> dict[str, Any]:
    """Compare harmonized literature forms without reopening candidate selection."""
    candidate = _sealed_report(candidate_report_path, REPORT_SCHEMA)
    if candidate.get("selection", {}).get("selected_model") != "M1":
        raise ReportError("the frozen literature comparison currently requires selected model M1")
    frame, receipt, input_binding = _snapshot(input_receipt_path)
    input_sha256 = str(receipt["fit_rows_content_sha256"]); holdouts = _holdout(holdouts_path, input_sha256)
    if candidate.get("input_sha256") != input_sha256 or candidate.get("input_receipt") != input_binding:
        raise ReportError("literature comparison uses a different input snapshot")
    if candidate.get("holdout_manifest_sha256") != holdouts["manifest_sha256"]:
        raise ReportError("literature comparison uses a different holdout manifest")
    memberships = {item["fold_id"]: bind_membership(holdouts_path, kind="primary", membership_id=item["fold_id"])
        for item in holdouts["primary_folds"]}
    memberships.update({"lowest": bind_membership(holdouts_path, kind="lowest",
        membership_id="lowest_terminal_macro_loss_30"), "all_data": bind_membership(holdouts_path, kind="all")})
    models = ("C0", "M0", "J0", "JH", "M1"); primary_models = ("C0", "M0", "J0", "M1")
    if set(results) != set(memberships) or any(set(results[view]) != set(models) for view in memberships):
        raise ReportError("literature comparison requires C0/M0/J0/JH/M1 for every frozen view")
    candidate_sources = {(item["model"], item["fold_id"]): item["sha256"]
        for item in candidate.get("source_results", []) if item.get("role") == "primary"}
    loaded: dict[str, dict[str, Mapping[str, Any]]] = {}; sources: list[dict[str, Any]] = []
    for view, membership in memberships.items():
        loaded[view] = {}
        for model in models:
            path = results[view][model]; result = _result(path)
            _validate_result(result=result, input_sha256=input_sha256, membership=membership, model=model, fold_id=view)
            if view not in {"lowest", "all_data"} and model in {"M0", "M1"}:
                if candidate_sources.get((model, view)) != result["result_sha256"]:
                    raise ReportError("primary M0/M1 result is not sealed by the candidate report")
            else:
                _selection_bound(result, candidate_report_path, candidate=candidate)
            loaded[view][model] = result
            sources.append({"view": view, "model": model, "path": str(path.resolve()),
                "sha256": result["result_sha256"]})
        scale = _scale_binding(loaded[view]["M0"])
        if any(_scale_binding(loaded[view][model]) != scale for model in models):
            raise ReportError("literature forms do not share the frozen fold-specific Huber scale")
    primary_metrics: dict[str, dict[str, Any]] = {model: {} for model in models}
    errors: dict[str, dict[str, dict[str, float]]] = {model: {} for model in primary_models}
    for view in memberships:
        if view in {"lowest", "all_data"}: continue
        membership = memberships[view]
        for model in models:
            result = loaded[view][model]
            training = _metrics(frame, membership, result["selected"], model, split="train")
            heldout = _metrics(frame, membership, result["selected"], model)
            primary_metrics[model][view] = {"training": training, "heldout": heldout,
                "fit": _literature_diagnostic(result, model)}
            if model in errors: errors[model][view] = heldout["trajectory_mse"]
    comparison = select_model_paired_one_standard_error(errors, model_order=primary_models,
        fold_order=tuple(item["fold_id"] for item in holdouts["primary_folds"]),
        replicates=PAIRED_BOOTSTRAP_REPLICATES)
    descriptive = comparison.to_dict(); reference = descriptive.pop("observed_best_model")
    for key in ("selected_model", "eligible_models", "eligibility_threshold"):
        descriptive.pop(key)
    descriptive["reference_model_for_paired_differences"] = reference
    descriptive["observed_rmse_rank"] = sorted(descriptive["observed_aggregate_rmse"],
        key=lambda model: (descriptive["observed_aggregate_rmse"][model], primary_models.index(model)))
    lowest_metrics = {}
    for model in models:
        result = loaded["lowest"][model]
        training = _metrics(frame, memberships["lowest"], result["selected"], model, split="train")
        heldout = _metrics(frame, memberships["lowest"], result["selected"], model)
        lowest_metrics[model] = {"training": training, "heldout": heldout,
            "training_plus_extrapolation_rmse": training["trajectory_balanced_rmse"] + heldout["trajectory_balanced_rmse"],
            "fit": _literature_diagnostic(result, model)}
    all_data = {model: {"result_sha256": loaded["all_data"][model]["result_sha256"],
        "shared_parameters": loaded["all_data"][model]["selected"]["shared_parameters"],
        "language_coefficients": loaded["all_data"][model]["selected"]["language_coefficients"],
        "fit": _literature_diagnostic(loaded["all_data"][model], model)} for model in models}
    compute_contract, compute_binding = _logical_compute_contract(logical_compute_contract_path)
    compute_predictions = _logical_compute_predictions(loaded["all_data"], compute_contract)
    unsigned = {"schema_version": D17_LITERATURE_REPORT_SCHEMA,
        "source_candidate": {"path": str(candidate_report_path.resolve()), "file_sha256": sha256_file(candidate_report_path),
            "report_sha256": candidate["report_sha256"]}, "input_sha256": input_sha256,
        "input_receipt": input_binding, "holdout_manifest_sha256": holdouts["manifest_sha256"],
        "fair_primary_models": list(primary_models), "published_input_sensitivity": "JH",
        "model_contracts": {"C0": {"label": "Chinchilla-form harmonized", "data": "D_t", "experts": "ignored", "languages": "ignored"},
            "M0": {"label": "ATLAS-form harmonized", "data": "D_t", "experts": "ignored", "languages": "K"},
            "J0": {"label": "Joint-MoE-form harmonized", "data": "D_t", "experts": "E", "languages": "ignored"},
            "JH": {"label": "Joint-MoE published-input sensitivity", "data": "D_total",
                "experts": "E_hat_fixed_published_endpoints", "E_start": JOINT_MOE_E_START,
                "E_max": JOINT_MOE_E_MAX, "languages": "ignored",
                "fitted_expert_coordinate": "(log(E_hat)-log(E_start))*log(32)/(log(E_hat(32))-log(E_start))",
                "coefficient_comparability": "affine-normalized; do not compare directly with published coefficients"},
            "M1": {"label": "selected multilingual-MoE law", "data": "D_t", "experts": "E", "languages": "K"}},
        "primary_metrics": primary_metrics, "paired_descriptive_comparison": descriptive,
        "lowest_loss_metrics": lowest_metrics, "all_data_fits_for_d20": all_data,
        "logical_compute_contract": compute_binding, "logical_compute_predictions": compute_predictions,
        "source_results": sources, "status": "literature_forms_complete_analysis_choice_sensitivities_pending"}
    return {**unsigned, "report_sha256": canonical_json_sha256(unsigned)}


def write_literature_comparison_report(output_root: Path, report: Mapping[str, Any]) -> Path:
    unsigned = dict(report); digest = unsigned.pop("report_sha256", None)
    if report.get("schema_version") != D17_LITERATURE_REPORT_SCHEMA or digest != canonical_json_sha256(unsigned):
        raise ReportError("literature comparison report hash differs")
    path = output_root / "reports" / "d17_literature_comparison.json"
    write_json_once(path, report); return path


def build_generalization_report(*, candidate_report_path: Path, input_receipt_path: Path,
    holdouts_path: Path, generalization_holdouts_path: Path,
    results: Mapping[str, Mapping[str, Path]]) -> dict[str, Any]:
    """Report fair supplemental views without selecting or aggregating models."""
    candidate = _sealed_report(candidate_report_path, REPORT_SCHEMA)
    frame, receipt, input_binding = _snapshot(input_receipt_path)
    input_sha256 = str(receipt['fit_rows_content_sha256']); holdouts = _holdout(holdouts_path, input_sha256)
    if (candidate.get('input_sha256') != input_sha256 or candidate.get('input_receipt') != input_binding
            or candidate.get('holdout_manifest_sha256') != holdouts['manifest_sha256']):
        raise ReportError('generalization battery differs from the selected candidate inputs')
    try:
        memberships = _generalization_memberships(holdouts_path=holdouts_path, generalization_path=generalization_holdouts_path, receipt=receipt)
    except CampaignError as error:
        raise ReportError(str(error)) from error
    if set(results) != set(memberships) or any(set(results[view]) != set(GENERALIZATION_MODELS) for view in memberships):
        raise ReportError('generalization battery requires exactly C0/M0/J0/M1 on all three views')
    metrics: dict[str, Any] = {}; views: dict[str, Any] = {}; sources = []
    for view, membership in memberships.items():
        loaded = {}
        for model in GENERALIZATION_MODELS:
            path = results[view][model]; result = _result(path)
            _validate_result(result=result, input_sha256=input_sha256, membership=membership, model=model, fold_id=view)
            _selection_bound(result, candidate_report_path, candidate=candidate)
            loaded[model] = result
            sources.append({'view': view, 'model': model, 'path': str(path.resolve()), 'file_sha256': sha256_file(path), 'result_sha256': result['result_sha256']})
        scale = _scale_binding(loaded['M0'])
        if any(_scale_binding(loaded[model]) != scale for model in GENERALIZATION_MODELS):
            raise ReportError('generalization forms do not share the fold-specific M0-OLS scale')
        scale_info = _result(Path(str(scale['path'])))['adaptive_huber_scale']
        expected_delta = None if scale_info['use_ols'] else float(scale_info['delta']); expected_objective = 'ols' if expected_delta is None else 'huber'
        if any(result.get('frozen_delta') != expected_delta or result.get('fit_identity', {}).get('objective') != expected_objective
                for result in loaded.values()):
            raise ReportError('generalization forms do not share the exact frozen Huber objective')
        m0_path = results[view]['M0']
        expected_warm = {'kind': 'warm', 'path': str(m0_path.resolve()), 'file_sha256': sha256_file(m0_path), 'result_sha256': loaded['M0']['result_sha256']}
        if [item for item in loaded['M1']['parents'] if item.get('kind') == 'warm'] != [expected_warm]:
            raise ReportError('M1 generalization fit does not use the exact M0 warm parent')
        metrics[view] = {model: {'training': _metrics(frame, membership, loaded[model]['selected'], model, split='train'),
            'heldout': _metrics(frame, membership, loaded[model]['selected'], model), 'fit': _literature_diagnostic(loaded[model], model)} for model in GENERALIZATION_MODELS}
        counts = {}
        for split in ('train', 'test'):
            rows = frame[frame['trajectory_id'].isin(membership[f'{split}_trajectory_ids'])]
            if membership[f'{split}_readout_ids']: rows = rows[rows['readout_id'].isin(membership[f'{split}_readout_ids'])]
            counts[f'{split}_readouts'] = int(rows['readout_id'].nunique())
        views[view] = {'evidence_class': 'independent_trajectory_holdout' if view == 'random_13_of_64' else 'dependent_repeated_checkpoint_diagnostic',
            'membership_sha256': membership['membership_sha256'], 'train_trajectories': len(membership['train_trajectory_ids']),
            'test_trajectories': len(membership['test_trajectory_ids']),
            'trajectory_overlap': len(set(membership['train_trajectory_ids']) & set(membership['test_trajectory_ids'])), **counts}
    supplement = read_json(generalization_holdouts_path)
    unsigned = {'schema_version': GENERALIZATION_REPORT_SCHEMA,
        'source_candidate': {'path': str(candidate_report_path.resolve()), 'file_sha256': sha256_file(candidate_report_path),
            'report_sha256': candidate['report_sha256']}, 'input_sha256': input_sha256, 'input_receipt': input_binding,
        'holdout_manifest': {'path': str(holdouts_path.resolve()), 'file_sha256': sha256_file(holdouts_path), 'manifest_sha256': holdouts['manifest_sha256']},
        'generalization_holdout_manifest': {'path': str(generalization_holdouts_path.resolve()),
            'file_sha256': sha256_file(generalization_holdouts_path), 'manifest_sha256': supplement['manifest_sha256']},
        'fair_models': list(GENERALIZATION_MODELS), 'views': views, 'metrics': metrics, 'source_results': sources,
        'role': 'post_selection_descriptive_no_model_selection_or_aggregate'}
    return {**unsigned, 'report_sha256': canonical_json_sha256(unsigned)}


def write_generalization_report(output_root: Path, report: Mapping[str, Any]) -> Path:
    unsigned = dict(report); digest = unsigned.pop('report_sha256', None)
    if report.get('schema_version') != GENERALIZATION_REPORT_SCHEMA or digest != canonical_json_sha256(unsigned): raise ReportError('generalization report hash differs')
    path = output_root / 'reports' / 'generalization_battery.json'
    write_json_once(path, report); return path


def _bound_activated(result: Mapping[str, Any]) -> bool:
    """Follow the immutable bound-repeat chain and count resolved contacts."""
    current = result; seen: set[str] = set()
    while True:
        selected = current.get("selected")
        if isinstance(selected, Mapping) and selected.get("boundary_contacts"):
            return True
        parents = [item for item in current.get("parents", [])
            if isinstance(item, Mapping) and item.get("kind") == "bound_repeat"]
        if len(parents) > 1:
            raise ReportError("D16 result has duplicate bound-repeat parents")
        if not parents:
            return False
        parent = parents[0]; path = Path(str(parent.get("path")))
        if sha256_file(path) != parent.get("file_sha256"):
            raise ReportError("D16 bound-repeat parent file changed")
        previous = read_json(path); unsigned = dict(previous); digest = unsigned.pop("result_sha256", None)
        if digest != canonical_json_sha256(unsigned) or digest != parent.get("result_sha256"):
            raise ReportError("D16 bound-repeat parent seal changed")
        identity = current.get("fit_identity", {}); previous_identity = previous.get("fit_identity", {})
        if (previous.get("fit_rows_content_sha256") != current.get("fit_rows_content_sha256")
                or previous.get("membership", {}).get("membership_sha256")
                    != current.get("membership", {}).get("membership_sha256")
                or previous_identity.get("model") != identity.get("model")
                or previous_identity.get("replicate_id") != identity.get("replicate_id")):
            raise ReportError("D16 bound-repeat chain changes fit identity")
        if previous_identity.get("bound_version") != identity.get("bound_version") - 1:
            raise ReportError("D16 bound-repeat versions are not contiguous")
        key = str(digest)
        if key in seen:
            raise ReportError("D16 bound-repeat chain contains a cycle")
        seen.add(key); current = previous

def _d16_adaptive_history(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    """Validate and summarize the immutable adaptive-parent chain."""
    history: list[dict[str, Any]] = []; current = result; seen: set[str] = set()
    final_manifest: Mapping[str, Any] | None = None
    while True:
        manifest_path = current.get("manifest_path")
        if not isinstance(manifest_path, str):
            raise ReportError("D16 result lacks its adaptive manifest")
        try:
            manifest = load_manifest(Path(manifest_path))
        except ExecutionError as error:
            raise ReportError(f"invalid D16 adaptive manifest: {error}") from error
        if final_manifest is None:
            final_manifest = manifest
        action = manifest.get("fit", {}).get("action")
        count = manifest.get("fit", {}).get("start_count")
        if action not in {"d16_adaptive", "d16_full_grid"} or count not in d16_adaptive_tiers(str(manifest["fit"]["model"])):
            raise ReportError("D16 result did not use a registered adaptive tier")
        audit_path_value = Path(manifest["execution"]["output_root"]) / "audits" / f"{manifest['manifest_sha256'][:20]}.json"
        audit_status = "missing"; reason = None
        if audit_path_value.exists():
            audit = read_json(audit_path_value); audit_unsigned = dict(audit); audit_digest = audit_unsigned.pop("audit_sha256", None)
            if (audit_digest != canonical_json_sha256(audit_unsigned)
                    or audit.get("manifest_sha256") != manifest["manifest_sha256"]
                    or audit.get("result_sha256") != current.get("result_sha256")):
                raise ReportError("D16 adaptive audit seal changed")
            audit_status = str(audit.get("status")); reason = audit.get("reason")
        history.append({"start_count": int(count), "mode": str(manifest["fit"]["start_mode"]),
            "audit_status": audit_status, "reason": reason})
        parents = [item for item in current.get("parents", [])
            if isinstance(item, Mapping) and item.get("kind") == "adaptive_parent"]
        if len(parents) > 1:
            raise ReportError("D16 result has duplicate adaptive parents")
        if not parents:
            break
        parent = parents[0]; path = Path(str(parent.get("path")))
        if sha256_file(path) != parent.get("file_sha256"):
            raise ReportError("D16 adaptive parent file changed")
        previous = read_json(path); unsigned = dict(previous); digest = unsigned.pop("result_sha256", None)
        if digest != canonical_json_sha256(unsigned) or digest != parent.get("result_sha256"):
            raise ReportError("D16 adaptive parent seal changed")
        if (previous.get("fit_rows_content_sha256") != current.get("fit_rows_content_sha256")
                or previous.get("membership", {}).get("membership_sha256") != current.get("membership", {}).get("membership_sha256")
                or previous.get("fit_identity", {}).get("model") != current.get("fit_identity", {}).get("model")
                or previous.get("fit_identity", {}).get("replicate_id") != current.get("fit_identity", {}).get("replicate_id")):
            raise ReportError("D16 adaptive chain changes fit identity")
        if str(digest) in seen:
            raise ReportError("D16 adaptive chain contains a cycle")
        seen.add(str(digest)); current = previous
    history.reverse()
    tiers = d16_adaptive_tiers(str(final_manifest["fit"]["model"]))
    counts = tuple(item["start_count"] for item in history)
    if not counts or counts[0] != tiers[0] or counts != tiers[:len(counts)]:
        raise ReportError("D16 adaptive tiers are not the frozen contiguous sequence")
    if any(item["audit_status"] != "needs_escalation" for item in history[:-1]):
        raise ReportError("D16 adaptive parent was not produced by a failed tier audit")
    return history, final_manifest

def _declared_prediction(selected: Mapping[str, Any], model: str, point: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"N_act", "E", "K", "D_t_by_language"}
    if set(point) != expected:
        raise ReportError(f"prediction point keys must be exactly {sorted(expected)}")
    d_t = point["D_t_by_language"]
    if not isinstance(d_t, Mapping) or not d_t:
        raise ReportError("prediction point needs a non-empty D_t_by_language object")
    n_act = float(point["N_act"]); experts = int(point["E"]); language_count = int(point["K"])
    if (not math.isfinite(n_act) or n_act <= 0.0 or experts != point["E"] or language_count != point["K"]
            or experts < 1 or language_count < 1):
        raise ReportError("prediction point N_act, E, or K is invalid")
    languages = tuple(sorted(str(item) for item in d_t))
    if len(languages) != language_count or any(not math.isfinite(float(d_t[item])) or float(d_t[item]) <= 0.0 for item in languages):
        raise ReportError("prediction point must declare one positive D_t for each of K languages")
    coefficients = {str(item["language"]): item for item in selected["language_coefficients"]}
    if not set(languages) <= set(coefficients):
        raise ReportError("prediction point language lacks a fitted nuisance coefficient")
    nuisance = [coefficients[language] for language in languages]
    values = predict(model, SharedParameters(**selected["shared_parameters"]), np.full(len(languages), n_act),
        [float(d_t[item]) for item in languages], np.full(len(languages), experts), np.full(len(languages), language_count),
        [item["L_inf"] for item in nuisance], [item["A"] for item in nuisance], [item["B"] for item in nuisance])
    per_language = {language: float(value) for language, value in zip(languages, values)}
    return {"macro_adjusted_loss": float(np.mean(values)), "per_language": per_language}

def build_d16_report(*, full_result: Path, resampling_path: Path, replicate_results: Mapping[int, Path],
    prediction_points: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the frozen 100-subset path after selection; never create subsets here."""
    full = _result(full_result)
    if full.get("membership", {}).get("kind") != "all":
        raise ReportError("D16 summary requires the selected all-data fit")
    if set(replicate_results) != set(range(100)):
        raise ReportError("D16 report requires all 100 frozen replicate identifiers")
    if not isinstance(prediction_points, Mapping) or not prediction_points:
        raise ReportError("D16 report requires at least one frozen prediction point")
    resampling = read_json(resampling_path); unsigned_resampling = dict(resampling)
    embedded = unsigned_resampling.pop("manifest_sha256", None)
    subsets = resampling.get("subsets")
    if (resampling.get("schema_version") != RESAMPLING_MANIFEST_SCHEMA_VERSION
            or embedded != canonical_json_sha256(unsigned_resampling)
            or resampling.get("input_sha256") != full.get("fit_rows_content_sha256")
            or resampling.get("seed") != UNCERTAINTY_SUBSAMPLE_SEED
            or resampling.get("subset_size") != UNCERTAINTY_SUBSET_TRAJECTORIES
            or not isinstance(subsets, list) or len(subsets) != UNCERTAINTY_REPLICATES
            or {item.get("replicate_id") for item in subsets if isinstance(item, Mapping)}
                != set(range(UNCERTAINTY_REPLICATES))):
        raise ReportError("resampling manifest is not the frozen D16 contract")
    resampling_binding = {"path": str(resampling_path.resolve()), "file_sha256": sha256_file(resampling_path),
        "manifest_sha256": embedded}
    expected_memberships = {replicate_id: bind_membership(resampling_path, kind="d16", membership_id=replicate_id)
        for replicate_id in range(UNCERTAINTY_REPLICATES)}
    if sha256_file(resampling_path) != resampling_binding["file_sha256"]:
        raise ReportError("resampling manifest changed while being bound")
    model = str(full["fit_identity"]["model"])
    full_predictions = {name: _declared_prediction(full["selected"], model, point)
        for name, point in sorted(prediction_points.items())}
    prediction_values = {name: {"macro_adjusted_loss": [],
        "per_language": {language: [] for language in value["per_language"]}}
        for name, value in full_predictions.items()}
    values: dict[str, list[float]] = {name: [] for name in full["selected"]["shared_parameters"]}
    failures: list[int] = []; boundary_activations: list[int] = []; nonidentified: list[int] = []
    escalated: list[int] = []; full_grid_fallbacks: list[int] = []; final_tier_counts: dict[str, int] = {}
    escalation_reasons: dict[str, list[str]] = {}
    for replicate_id in range(100):
        raw = read_json(replicate_results[replicate_id]); _sealed = dict(raw); digest = _sealed.pop("result_sha256", None)
        if digest != canonical_json_sha256(_sealed) or raw.get("review_only"):
            raise ReportError(f"invalid or review-only D16 replicate {replicate_id}")
        if (raw.get("fit_identity", {}).get("replicate_id") != replicate_id
                or raw.get("fit_identity", {}).get("model") != full.get("fit_identity", {}).get("model")
                or raw.get("membership") != expected_memberships[replicate_id]
                or raw.get("fit_rows_content_sha256") != full.get("fit_rows_content_sha256")
                or raw.get("selection_report") != full.get("selection_report")):
            raise ReportError(f"D16 replicate identity or frozen D16 membership changed: {replicate_id}")
        history, final_manifest = _d16_adaptive_history(raw)
        if history[-1]["audit_status"] not in {"complete", "failed"}:
            raise ReportError(f"D16 adaptive protocol is incomplete: {replicate_id}")
        anchors = [item for item in final_manifest.get("parents", [])
            if isinstance(item, Mapping) and item.get("kind") == "full_data"]
        if (len(anchors) != 1 or anchors[0].get("result_sha256") != full.get("result_sha256")
                or sha256_file(Path(str(anchors[0].get("path")))) != anchors[0].get("file_sha256")):
            raise ReportError(f"D16 replicate is not anchored to the selected all-data fit: {replicate_id}")
        final_count = int(history[-1]["start_count"])
        final_tier_counts[str(final_count)] = final_tier_counts.get(str(final_count), 0) + 1
        if len(history) > 1:
            escalated.append(replicate_id)
            reasons = [str(item["reason"]) for item in history[:-1] if item.get("reason")]
            if reasons:
                escalation_reasons[str(replicate_id)] = reasons
        if final_count == d16_adaptive_tiers(model)[-1]:
            full_grid_fallbacks.append(replicate_id)
        raw_selected = raw.get("selected")
        if raw_selected is not None and not isinstance(raw_selected, Mapping):
            raise ReportError(f"malformed D16 selected result: {replicate_id}")
        if _bound_activated(raw):
            boundary_activations.append(replicate_id)
        audit_file = replicate_results[replicate_id].parent.parent / "audits" / f"{raw['manifest_sha256'][:20]}.json"
        if audit_file.exists():
            audit = read_json(audit_file); audit_unsigned = dict(audit); audit_digest = audit_unsigned.pop("audit_sha256", None)
            if audit_digest != canonical_json_sha256(audit_unsigned):
                raise ReportError(f"invalid D16 audit: {replicate_id}")
            if "non-identification" in str(audit.get("reason")):
                nonidentified.append(replicate_id)
        try:
            result = _result(replicate_results[replicate_id])
        except ReportError:
            failures.append(replicate_id); continue
        for name, value in result["selected"]["shared_parameters"].items():
            values[name].append(float(value))
        for name, point in prediction_points.items():
            derived = _declared_prediction(result["selected"], model, point)
            prediction_values[name]["macro_adjusted_loss"].append(derived["macro_adjusted_loss"])
            for language, value in derived["per_language"].items():
                prediction_values[name]["per_language"][language].append(value)
    def interval(full_value: float, series: list[float]) -> dict[str, float | None]:
        return {"full_data": float(full_value),
            "p10": float(np.percentile(series, 10.0)) if series else None,
            "p90": float(np.percentile(series, 90.0)) if series else None}
    estimates = {name: {"full_data": float(full["selected"]["shared_parameters"][name]),
        "p10": float(np.percentile(series, 10.0)) if series else None,
        "p90": float(np.percentile(series, 90.0)) if series else None} for name, series in values.items()}
    prediction_estimates: dict[str, Any] = {}
    for name, full_value in full_predictions.items():
        macro = interval(full_value["macro_adjusted_loss"], prediction_values[name]["macro_adjusted_loss"])
        per_language = {language: interval(value, prediction_values[name]["per_language"][language])
            for language, value in full_value["per_language"].items()}
        prediction_estimates[name] = {"macro_adjusted_loss": macro, "per_language": per_language}
    unsigned = {"schema_version": D16_REPORT_SCHEMA, "full_result_sha256": full["result_sha256"],
        "resampling_manifest": resampling_binding,
        "shared_parameter_estimates": estimates, "convergence_failures": failures,
        "boundary_frequency": len(boundary_activations) / 100.0, "boundary_activation_replicates": boundary_activations,
        "nonidentification_frequency": len(nonidentified) / 100.0,
        "nonidentification_replicates": nonidentified,
        "adaptive_protocol": {"tiers": list(d16_adaptive_tiers(model)),
            "final_tier_counts": final_tier_counts,
            "escalated_replicates": escalated, "escalation_frequency": len(escalated) / 100.0,
            "escalation_reasons": escalation_reasons,
            "full_grid_fallback_replicates": full_grid_fallbacks,
            "full_grid_fallback_frequency": len(full_grid_fallbacks) / 100.0},
        "prediction_points": {key: dict(value) for key, value in sorted(prediction_points.items())},
        "derived_prediction_estimates": prediction_estimates,
        "d17_status": "blocked", "d17_block_reason": D17_BLOCK_REASON}
    return {**unsigned, "report_sha256": canonical_json_sha256(unsigned)}


def write_d16_report(output_root: Path, report: Mapping[str, Any]) -> Path:
    unsigned = dict(report); digest = unsigned.pop("report_sha256", None)
    if report.get("schema_version") != D16_REPORT_SCHEMA or digest != canonical_json_sha256(unsigned):
        raise ReportError("D16 report hash differs")
    path = output_root / "reports" / "d16_uncertainty.json"; write_json_once(path, report); return path


def build_final_report(*, candidate_report_path: Path, all_data_result: Path, d16_report_path: Path) -> dict[str, Any]:
    """Validate D16 lineage, then fail closed until all D17 work is sealed."""
    candidate = read_json(candidate_report_path); unsigned = dict(candidate); digest = unsigned.pop("report_sha256", None)
    if candidate.get("schema_version") != REPORT_SCHEMA or digest != canonical_json_sha256(unsigned):
        raise ReportError("candidate report hash differs")
    result = _result(all_data_result); selected_model = candidate["selection"]["selected_model"]
    identity = result.get("fit_identity", {})
    selection = result.get("selection_report")
    if (identity.get("model") != selected_model or result.get("membership", {}).get("kind") != "all"
            or result.get("fit_rows_content_sha256") != candidate.get("input_sha256")
            or not isinstance(selection, Mapping)
            or selection.get("path") != str(candidate_report_path.resolve())
            or selection.get("file_sha256") != sha256_file(candidate_report_path)
            or selection.get("report_sha256") != candidate.get("report_sha256")):
        raise ReportError("all-data result does not match the selected candidate and all-data membership")
    d16 = read_json(d16_report_path); d16_unsigned = dict(d16); d16_digest = d16_unsigned.pop("report_sha256", None)
    if (d16.get("schema_version") != D16_REPORT_SCHEMA or d16_digest != canonical_json_sha256(d16_unsigned)
            or d16.get("full_result_sha256") != result["result_sha256"]):
        raise ReportError("D16 report is not sealed for this all-data result")
    if d16.get("convergence_failures"):
        raise ReportError("finalization requires a complete D16 uncertainty campaign")
    raise ReportError(f"finalization blocked until D17 is complete: {D17_BLOCK_REASON}")


def write_final_report(output_root: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    unsigned = dict(report); digest = unsigned.pop("report_sha256", None)
    if report.get("schema_version") != FINAL_REPORT_SCHEMA or digest != canonical_json_sha256(unsigned):
        raise ReportError("final report hash differs")
    if report.get("d16_status") != "complete" or report.get("d17_status") != "complete":
        raise ReportError("finalization requires complete sealed D16 and D17 artifacts")
    report_path = output_root / "reports" / "final_fit.json"; write_json_once(report_path, report)
    receipt_unsigned = {"schema_version": "stage1_fit_finalization/v1", "final_report_sha256": report["report_sha256"],
        "final_report_path": str(report_path)}
    receipt = {**receipt_unsigned, "receipt_sha256": canonical_json_sha256(receipt_unsigned)}
    receipt_path = output_root / "receipts" / "finalization.json"; write_json_once(receipt_path, receipt)
    return report_path, receipt_path


def execution_summary(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path); validate_manifest(manifest)
    root = Path(manifest["execution"]["output_root"])
    result = read_json(root / "results" / f"{manifest['manifest_sha256'][:20]}.json")
    if result.get("status") not in {"reduced", "failed"}:
        raise ReportError("manifest has no reduced result")
    return {"phase": manifest["phase"], "model": manifest["fit"]["model"], "fold_id": manifest["fit"]["fold_id"],
        "attempted": len(result["attempts"]), "selected_start": None if not result.get("selected") else result["selected"]["start_id"],
        "objective": None if not result.get("selected") else result["selected"]["objective"], "result_sha256": result["result_sha256"]}
