"""Reproduction check and harmonized five-model primary comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from ..campaign import audit_path, scientific_result
from ..data import sha256_file
from ..execution import bind_membership, read_json, write_json_once, write_text_once
from ..fit import (
    OBJECTIVE_RELATIVE_TOLERANCE,
    PREDICTION_ABSOLUTE_TOLERANCE,
    equivalence_check,
)
from ..report import (
    REPORT_SCHEMA as CANDIDATE_REPORT_SCHEMA,
    _holdout,
    _metrics,
    _prediction,
    _snapshot,
    _validate_result,
    build_candidate_report,
    write_candidate_report,
)
from ..selection import select_model_paired_one_standard_error
from .contracts import (
    BASELINES,
    BOOTSTRAP_REPLICATES,
    CANDIDATES,
    EXPECTED_SELECTED_MODEL,
    FOLDS,
    MODELS,
    REPORT_SCHEMA,
    REPRODUCTION_SCHEMA,
    BaselineRefitError,
    seal,
    verify_seal,
)
from .planning import result_for_job


MODEL_CONTRACTS = {
    "C0": {"role": "literature_baseline", "form": "Chinchilla-style", "covariates": "N,D_t"},
    "M0": {"role": "literature_baseline", "form": "ATLAS-style capacity", "covariates": "N,D_t,K"},
    "J0": {"role": "literature_baseline", "form": "harmonized JointMoE", "covariates": "N,D_t,E"},
    "M1": {"role": "proposed_model", "form": "expert-dependent exponents", "covariates": "N,D_t,E,K"},
    "M2": {"role": "proposed_model", "form": "expert-language interactions", "covariates": "N,D_t,E,K"},
}


def _job_id(model: str, fold: str) -> str:
    if model in BASELINES:
        return f"grid.robust.{model}.{fold}"
    return f"grid.{model.lower()}.{fold}"


def result_matrix(workflow: Mapping[str, Any]) -> dict[str, dict[str, Path]]:
    return {
        model: {fold: result_for_job(workflow, _job_id(model, fold)) for fold in FOLDS}
        for model in MODELS
    }


def _historical_sources(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = report.get("source_results")
    if not isinstance(rows, list):
        raise BaselineRefitError("historical candidate report has no source results")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("model") not in CANDIDATES \
                or row.get("fold_id") not in FOLDS:
            raise BaselineRefitError("historical candidate source inventory changed")
        key = (str(row["model"]), str(row["fold_id"]))
        if key in result:
            raise BaselineRefitError("historical candidate report repeats a source result")
        result[key] = row
    if set(result) != {(model, fold) for model in CANDIDATES for fold in FOLDS}:
        raise BaselineRefitError("historical candidate source results are incomplete")
    return result


def build_reproduction_report(
    workflow: Mapping[str, Any],
    *,
    fresh_candidate: Mapping[str, Any],
    fresh_results: Mapping[str, Mapping[str, Path]],
) -> dict[str, Any]:
    historical_path = Path(str(workflow["inputs"]["historical_candidate_report"]["path"]))
    historical = read_json(historical_path)
    verify_seal(historical, "report_sha256", label="historical candidate report")
    if historical.get("schema_version") != CANDIDATE_REPORT_SCHEMA:
        raise BaselineRefitError("historical candidate report schema changed")
    historical_sources = _historical_sources(historical)
    frame, receipt, _ = _snapshot(Path(str(workflow["inputs"]["input_receipt"]["path"])))
    input_sha = str(receipt["fit_rows_content_sha256"])
    holdouts_path = Path(str(workflow["inputs"]["holdouts"]["path"]))
    holdouts = _holdout(holdouts_path, input_sha)
    if historical.get("input_sha256") != input_sha \
            or historical.get("holdout_manifest_sha256") != holdouts["manifest_sha256"]:
        raise BaselineRefitError("historical candidate report uses different fixed inputs")

    checks = []
    for model in CANDIDATES:
        for fold in FOLDS:
            membership = bind_membership(holdouts_path, kind="primary", membership_id=fold)
            fresh = scientific_result(fresh_results[model][fold])
            source = historical_sources[(model, fold)]
            historical_result_path = Path(str(source["path"]))
            historical_result = scientific_result(historical_result_path)
            if historical_result.get("result_sha256") != source.get("sha256"):
                raise BaselineRefitError("historical source result hash changed")
            _validate_result(result=fresh, input_sha256=input_sha, membership=membership,
                model=model, fold_id=fold)
            _validate_result(result=historical_result, input_sha256=input_sha,
                membership=membership, model=model, fold_id=fold)
            fresh_rows, fresh_prediction = _prediction(frame, membership, fresh["selected"], model)
            historical_rows, historical_prediction = _prediction(
                frame, membership, historical_result["selected"], model,
            )
            identity_columns = ["readout_id", "language"]
            if fresh_rows[identity_columns].astype(str).values.tolist() \
                    != historical_rows[identity_columns].astype(str).values.tolist():
                raise BaselineRefitError("historical and fresh prediction rows differ")
            comparison = equivalence_check(
                float(fresh["selected"]["objective"]),
                float(historical_result["selected"]["objective"]),
                fresh_prediction,
                historical_prediction,
            )
            checks.append({
                "model": model,
                "fold": fold,
                "equivalent": comparison.equivalent,
                "relative_objective_difference": comparison.relative_objective_difference,
                "maximum_absolute_prediction_difference": comparison.maximum_absolute_prediction_difference,
                "fresh_result_sha256": fresh["result_sha256"],
                "historical_result_sha256": historical_result["result_sha256"],
            })
    fresh_selected = fresh_candidate.get("selection", {}).get("selected_model")
    historical_selected = historical.get("selection", {}).get("selected_model")
    passed = all(item["equivalent"] for item in checks) \
        and fresh_selected == historical_selected == EXPECTED_SELECTED_MODEL
    unsigned = {
        "schema_version": REPRODUCTION_SCHEMA,
        "workflow_sha256": workflow["workflow_sha256"],
        "historical_candidate": {
            "path": str(historical_path),
            "file_sha256": sha256_file(historical_path),
            "report_sha256": historical["report_sha256"],
        },
        "objective_relative_tolerance": OBJECTIVE_RELATIVE_TOLERANCE,
        "prediction_absolute_tolerance": PREDICTION_ABSOLUTE_TOLERANCE,
        "historical_selected_model": historical_selected,
        "fresh_selected_model": fresh_selected,
        "checks": checks,
        "status": "complete" if passed else "failed",
    }
    return seal(unsigned, "report_sha256")


def _scale_parent(result: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [item for item in result.get("parents", [])
        if isinstance(item, Mapping) and item.get("kind") == "scale_m0_ols"]
    if len(matches) != 1:
        raise BaselineRefitError("robust fit lacks exactly one M0-OLS scale parent")
    return matches[0]


def _paired_baseline_minus_m2(
    errors: Mapping[str, Mapping[str, Mapping[str, float]]], baseline: str,
) -> dict[str, Any]:
    pair = select_model_paired_one_standard_error(
        {"M2": errors["M2"], baseline: errors[baseline]},
        model_order=("M2", baseline),
        fold_order=FOLDS,
        replicates=BOOTSTRAP_REPLICATES,
    )
    if pair.observed_best_model == "M2":
        lower, upper = pair.bootstrap_difference_percentiles_10_90[baseline]
        standard_error = pair.paired_standard_error_vs_best[baseline]
    else:
        raw_lower, raw_upper = pair.bootstrap_difference_percentiles_10_90["M2"]
        lower, upper = -raw_upper, -raw_lower
        standard_error = pair.paired_standard_error_vs_best["M2"]
    observed = pair.observed_aggregate_rmse[baseline] - pair.observed_aggregate_rmse["M2"]
    return {
        "observed_rmse_difference": observed,
        "observed_percent_improvement": 100.0 * observed / pair.observed_aggregate_rmse[baseline],
        "paired_standard_error": standard_error,
        "bootstrap_difference_percentiles_10_90": [lower, upper],
        "paired_support": lower > 0.0,
    }


def strict_baseline_decision(
    fold_wins: Mapping[str, Mapping[str, bool]],
    pairwise: Mapping[str, Mapping[str, Any]],
) -> bool:
    if set(fold_wins) != set(BASELINES) or set(pairwise) != set(BASELINES):
        raise BaselineRefitError("strict baseline decision has an incomplete baseline set")
    if any(set(fold_wins[baseline]) != set(FOLDS) for baseline in BASELINES):
        raise BaselineRefitError("strict baseline decision has an incomplete fold set")
    return all(all(bool(value) for value in fold_wins[baseline].values()) for baseline in BASELINES) \
        and all(bool(pairwise[baseline].get("paired_support")) for baseline in BASELINES)


def build_comparison_report(
    workflow: Mapping[str, Any],
    *,
    candidate_report_path: Path,
    reproduction_report_path: Path,
    results: Mapping[str, Mapping[str, Path]],
    result_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    # A versioned recovery may supply a validator for a narrowly amended
    # numerical audit. Ordinary reports still require scientific_result.
    load_result = scientific_result if result_loader is None else result_loader
    input_receipt = Path(str(workflow["inputs"]["input_receipt"]["path"]))
    holdouts_path = Path(str(workflow["inputs"]["holdouts"]["path"]))
    frame, receipt, input_binding = _snapshot(input_receipt)
    input_sha = str(receipt["fit_rows_content_sha256"])
    holdouts = _holdout(holdouts_path, input_sha)
    memberships = {
        fold: bind_membership(holdouts_path, kind="primary", membership_id=fold)
        for fold in FOLDS
    }
    metrics: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    errors: dict[str, dict[str, Mapping[str, float]]] = {model: {} for model in MODELS}
    parameter_counts: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    sources = []
    loaded: dict[str, dict[str, Mapping[str, Any]]] = {model: {} for model in MODELS}
    for model in MODELS:
        if set(results.get(model, {})) != set(FOLDS):
            raise BaselineRefitError(f"{model} result matrix is incomplete")
        for fold in FOLDS:
            path = results[model][fold]
            result = load_result(path)
            _validate_result(result=result, input_sha256=input_sha,
                membership=memberships[fold], model=model, fold_id=fold)
            loaded[model][fold] = result
            metric = _metrics(frame, memberships[fold], result["selected"], model)
            metrics[model][fold] = metric
            errors[model][fold] = metric["trajectory_mse"]
            selected = result["selected"]
            nuisance = selected["language_coefficients"]
            parameter_counts[model][fold] = {
                "outer_parameters": len(selected["coordinates"]),
                "fitted_languages": len(nuisance),
                "language_nuisance_parameters": 3 * len(nuisance),
                "total_fitted_parameters": len(selected["coordinates"]) + 3 * len(nuisance),
            }
            sources.append({
                "model": model,
                "fold": fold,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "result_sha256": result["result_sha256"],
                "audit": _report_binding(audit_path(result, output_root=Path(str(workflow["output_root"])))),
                **({"audit_amendment": result["audit_amendment"]}
                   if "audit_amendment" in result else {}),
            })
    for fold in FOLDS:
        scale = _scale_parent(loaded["M0"][fold])
        objective = loaded["M0"][fold]["fit_identity"]["objective"]
        delta = loaded["M0"][fold]["frozen_delta"]
        for model in MODELS:
            result = loaded[model][fold]
            if _scale_parent(result) != scale \
                    or result["fit_identity"]["objective"] != objective \
                    or result["frozen_delta"] != delta:
                raise BaselineRefitError(f"{fold} models do not share the exact robust fit contract")

    descriptive = select_model_paired_one_standard_error(
        errors, model_order=MODELS, fold_order=FOLDS, replicates=BOOTSTRAP_REPLICATES,
    ).to_dict()
    for key in ("selected_model", "eligible_models", "eligibility_threshold"):
        descriptive.pop(key, None)
    descriptive["observed_rmse_rank"] = sorted(
        MODELS,
        key=lambda model: (descriptive["observed_aggregate_rmse"][model], MODELS.index(model)),
    )
    pairwise = {baseline: _paired_baseline_minus_m2(errors, baseline) for baseline in BASELINES}
    fold_percent_improvement = {
        baseline: {
            fold: 100.0 * (
                metrics[baseline][fold]["trajectory_balanced_rmse"]
                - metrics["M2"][fold]["trajectory_balanced_rmse"]
            ) / metrics[baseline][fold]["trajectory_balanced_rmse"]
            for fold in FOLDS
        }
        for baseline in BASELINES
    }
    fold_wins = {
        baseline: {
            fold: metrics["M2"][fold]["trajectory_balanced_rmse"]
                < metrics[baseline][fold]["trajectory_balanced_rmse"]
            for fold in FOLDS
        }
        for baseline in BASELINES
    }
    strict_win = strict_baseline_decision(fold_wins, pairwise)
    candidate = read_json(candidate_report_path)
    verify_seal(candidate, "report_sha256", label="fresh candidate report")
    unsigned = {
        "schema_version": REPORT_SCHEMA,
        "workflow_sha256": workflow["workflow_sha256"],
        "input_sha256": input_sha,
        "input_receipt": input_binding,
        "holdout_manifest_sha256": holdouts["manifest_sha256"],
        "candidate_report": {
            "path": str(candidate_report_path),
            "file_sha256": sha256_file(candidate_report_path),
            "report_sha256": candidate["report_sha256"],
        },
        "reproduction_report": _report_binding(reproduction_report_path),
        "model_contracts": MODEL_CONTRACTS,
        "primary_metrics": metrics,
        "parameter_counts": parameter_counts,
        "paired_descriptive_comparison": descriptive,
        "pairwise_baseline_minus_m2": pairwise,
        "fold_percent_improvement_vs_m2": fold_percent_improvement,
        "strict_fold_wins": fold_wins,
        "strict_baseline_win": strict_win,
        "claim_rule": workflow["claim_rule"],
        "interpretation": "descriptive internal validation on the existing frozen trajectory holdouts",
        "source_results": sources,
        "status": "complete",
    }
    return seal(unsigned, "report_sha256")


def _report_binding(path: Path) -> dict[str, str]:
    payload = read_json(path)
    embedded = payload.get("audit_sha256", payload.get("report_sha256"))
    if not isinstance(embedded, str):
        raise BaselineRefitError(f"report or audit lacks an embedded hash: {path}")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "embedded_sha256": embedded,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    labels = {"C0": "Chinchilla-form (C0)", "M0": "ATLAS count/capacity-form (M0)",
              "J0": "JointMoE-form (J0)", "M1": "Our expert-dependent law (M1)",
              "M2": "Our expert-language interaction law (M2)"}
    lines = [
        "# Stage 1 harmonized primary baseline comparison",
        "",
        f"Strict M2 baseline win: **{str(bool(report['strict_baseline_win'])).lower()}**",
        "",
        "All models use the same adjusted loss, target-language exposure, trajectory weights,",
        "primary memberships, and fold-specific robust scale. The independent unit is a trajectory.",
        "These are harmonized literature forms; the ATLAS row is not its full adaptive-transfer/repetition model.",
        "",
        "| Law | Largest active size | E32 | K200, common languages | Equal-fold mean RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        values = [report["primary_metrics"][model][fold]["trajectory_balanced_rmse"] for fold in FOLDS]
        lines.append(f"| {labels[model]} | " + " | ".join(f"{v:.8f}" for v in values)
                     + f" | {sum(values) / len(values):.8f} |")
    lines.extend(("", "## M2 versus literature baselines", "",
        "Paired p10/p90 bounds form an 80% trajectory-bootstrap interval for baseline RMSE minus M2 RMSE.",
        "Positive differences favor M2. These are descriptive internal-validation intervals, not new prospective tests.", "",
        "| Baseline | Aggregate improvement | Paired p10 | Paired p90 | Supported |",
        "|---|---:|---:|---:|---|"))
    for baseline in BASELINES:
        row = report["pairwise_baseline_minus_m2"][baseline]
        lower, upper = row["bootstrap_difference_percentiles_10_90"]
        lines.append(
            f"| {labels[baseline]} | {row['observed_percent_improvement']:.4f}% | "
            f"{lower:.8f} | {upper:.8f} | {str(bool(row['paired_support'])).lower()} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_and_write_reports(workflow: Mapping[str, Any]) -> dict[str, Any]:
    results = result_matrix(workflow)
    candidate_results = {model: results[model] for model in CANDIDATES}
    candidate = build_candidate_report(
        input_receipt_path=Path(str(workflow["inputs"]["input_receipt"]["path"])),
        holdouts_path=Path(str(workflow["inputs"]["holdouts"]["path"])),
        results=candidate_results,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )
    reproduction = build_reproduction_report(
        workflow, fresh_candidate=candidate, fresh_results=candidate_results,
    )
    output = Path(str(workflow["output_root"]))
    reproduction_path = output / "reports" / "candidate_reproduction.json"
    write_json_once(reproduction_path, reproduction)
    if reproduction["status"] != "complete":
        raise BaselineRefitError("fresh M0-M2 campaign did not reproduce the historical candidate result")
    candidate_path, candidate_markdown = write_candidate_report(output, candidate)
    comparison = build_comparison_report(
        workflow, candidate_report_path=candidate_path,
        reproduction_report_path=reproduction_path, results=results,
    )
    comparison_path = output / "reports" / "primary_baseline_comparison.json"
    markdown_path = output / "reports" / "primary_baseline_comparison.md"
    write_json_once(comparison_path, comparison)
    write_text_once(markdown_path, _markdown(comparison))
    return {
        "reproduction": {"path": str(reproduction_path), "file_sha256": sha256_file(reproduction_path),
            "report_sha256": reproduction["report_sha256"]},
        "candidate": {"path": str(candidate_path), "file_sha256": sha256_file(candidate_path),
            "report_sha256": candidate["report_sha256"], "markdown": str(candidate_markdown)},
        "comparison": {"path": str(comparison_path), "file_sha256": sha256_file(comparison_path),
            "report_sha256": comparison["report_sha256"], "markdown": str(markdown_path)},
        "strict_baseline_win": comparison["strict_baseline_win"],
    }
