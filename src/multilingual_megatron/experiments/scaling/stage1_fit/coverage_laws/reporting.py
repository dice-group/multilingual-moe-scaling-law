"""Inner-only choice and a separate all-candidate developmental scorecard."""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as io
from ..law_search import data
from ..law_search import model as legacy
from ..data import canonical_json_sha256
from ..scoring import checked_score
from ..artifacts import binding, verify
from ..problems import problem
from . import replay
from .state import cpu_guard, seal, load, read_result, result_path
from .contracts import INNER, PRIMARY, RULES, choose, forms_for, plan_rules
from .model import Model, validate


def comparison_gate(candidate, baselines):
    eligible = baselines[baselines.certified & baselines.complete]
    best = eligible.loc[eligible.rmse.idxmin()] if len(eligible) else None
    unresolved = baselines.loc[~(baselines.certified & baselines.complete), "model"].tolist()
    win = bool(candidate["certified"] and candidate["complete"] and best is not None
               and candidate["rmse"] < best.rmse - RULES["win_tolerance_nats"])
    return dict(best_baseline=None if best is None else best.model,
                baseline_rmse=None if best is None else float(best.rmse),
                candidate_rmse=candidate.get("rmse"), baseline_models=len(baselines),
                unresolved=unresolved, win_among_scoreable=win,
                strict_all_baseline_win=bool(win and not unresolved))


def mean_metrics(metrics):
    """Every mean needs all three distinct grid holdouts with unchanged weights."""
    rows = metrics[(metrics.case == "grid") & metrics.fold.isin(PRIMARY)]
    means = []
    for (role, name), group in rows.groupby(["role", "model"]):
        complete = set(group.fold) == set(PRIMARY) and len(group) == len(PRIMARY) \
            and bool((group.complete & group.certified).all())
        means.append(dict(role=role, model=name, complete=complete,
                          mean_primary_rmse=float(group.rmse.mean()) if complete else None))
    return means


def complete_baseline_panel(frame):
    """Expose absent law/case fits, including later all-grid-only ATLAS forms."""
    frame = frame[frame.case.isin(["grid", "L26"])].copy()
    if frame.duplicated(["model", "fold", "case"]).any():
        raise ValueError("duplicate bound historical comparison")
    cases = [(fold, case) for fold in (*PRIMARY, "all_grid") for case in
             (["L26"] if fold == "all_grid" else ["grid", "L26"])]
    missing = []
    for name, group in frame.groupby("model", sort=True):
        categories = group.category.dropna().unique()
        if len(categories) != 1:
            raise ValueError("historical model category is ambiguous")
        present = set(zip(group.fold, group.case))
        for fold, case in cases:
            if (fold, case) not in present:
                missing.append(dict(model=name, fold=fold, case=case, category=categories[0],
                    certified=False, complete=False, rmse=None, rows=0,
                    missing_reason="no saved fit for this registered law/case"))
    result = pd.concat([frame, pd.DataFrame(missing)], ignore_index=True)
    result["role"] = "historical"
    return result


def prior_campaign_panel(entry):
    """All registered prior procedures, namespaced so old and fresh fits cannot merge."""
    original = pd.read_csv(verify(entry["metrics"]))
    rows = original[original.model.isin(entry["forms"])].copy()
    if set(rows.model) != set(entry["forms"]) or not rows.role.isin(["candidate", "matched_control"]).all():
        raise ValueError("incomplete or ambiguous prior-procedure comparison")
    rows["source_model"] = rows.model
    rows["model"] = entry["prefix"]+rows.model
    return complete_baseline_panel(rows)


def fitted_result(root, plan, name, fold):
    task = next(t for t in plan["tasks"] if t["stage"] == "certify" and t["form"] == name and t["fold"] == fold)
    try:
        return task, read_result(root, plan, task)
    except FileNotFoundError:
        return task, dict(certified=False, complete=False, error="missing certification receipt")


def predictions(result, name, rows):
    stored = rows.N_total.to_numpy() if "N_total" in rows else None
    model = Model(name, problem(rows, result["regularization"]), rows.U_t.to_numpy(),
                  stored_parameters=stored)
    # Prediction covariates must not redefine input-specific training restrictions.
    # Validate coordinates against the saved training bounds instead.
    from ..repetition_model import Bound
    model.bounds = [Bound(**b) for b in result["bounds"]]
    model.equalities = tuple(tuple(pair) for pair in result["equalities"])
    validate(model, result["selected"])
    first = legacy.predict(model, result["selected"])
    second = replay.predict(name, result["selected"], rows)
    np.testing.assert_allclose(first, second, atol=RULES["replay_tolerance"], rtol=0)
    return rows[data.KEYS+["adjusted_loss"]].assign(prediction=first)


def select(path):
    cpu_guard()
    plan, root = load(path), Path(path).resolve().parent
    rules = plan_rules(plan)
    forms = forms_for(rules)
    if (root / "selection.json").exists() or (root / "report").exists():
        raise ValueError("no reselection after a sealed choice/report")
    metrics, inputs, saved_predictions = [], [], []
    for fold in INNER:
        entries = plan["inner_scoring"][fold]
        row_entry = next(e for e in entries if Path(e["path"]).name == "rows.parquet")
        rows = pd.read_parquet(verify(row_entry))
        inputs.append(row_entry)
        for name in forms:
            task, result = fitted_result(root, plan, name, fold)
            if result_path(root, task).exists():
                inputs.append(binding(result_path(root, task)))
            if result.get("certified"):
                pred = predictions(result, name, rows)
                score = checked_score(pred, rows)
                saved_predictions.append(pred.assign(form=name, fold=fold))
            else:
                score = dict(complete=False, rmse=None, error=result.get("error"))
            metrics.append(dict(form=name, fold=fold, certified=bool(result.get("certified")), **score))
    if saved_predictions:
        pd.concat(saved_predictions, ignore_index=True).to_parquet(root / "inner_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(root / "inner_metrics.csv", index=False)
    try:
        decision = choose(metrics, rules) | {"passed": True}
    except ValueError as error:
        decision = dict(passed=False, chosen=None, error=str(error))
    seal(root / "selection.json", decision | dict(plan_sha256=plan["sha256"], inner_metrics=metrics,
         inputs=inputs, frozen_before_outer_scoring=True, outer_or_endpoint_responses_read=False,
         parallel_outer_training_fits_predeclared=True, evidence=rules["evidence"]))
    print({k: decision.get(k) for k in ("passed", "chosen", "error")}, flush=True)
    return decision["passed"]


def selection_receipt(root, plan):
    receipt = io.read_json(root / "selection.json")
    unsigned = {k: v for k, v in receipt.items() if k != "sha256"}
    if canonical_json_sha256(unsigned) != receipt["sha256"] or receipt["plan_sha256"] != plan["sha256"]:
        raise ValueError("foreign or modified selection")
    for entry in receipt["inputs"]:
        verify(entry)
    if not receipt["passed"]:
        raise ValueError("inner selection failed; see inner_metrics.csv")
    if not receipt["frozen_before_outer_scoring"] or receipt["outer_or_endpoint_responses_read"]:
        raise ValueError("invalid selection/scoring boundary")
    if receipt["chosen"] != choose(receipt["inner_metrics"], plan_rules(plan))["chosen"]:
        raise ValueError("registered choice does not reproduce")
    return receipt


def paired_metrics(frame, means, inner, pairs):
    """Matched differences, never a substitute for the sealed inner decision."""
    panel = frame[["model", "fold", "case", "certified", "complete", "rmse"]].copy()
    average = means.rename(columns={"mean_primary_rmse": "rmse"}).assign(
        fold="mean_primary", case="mean_primary", certified=means.complete)
    screen = inner.rename(columns={"form": "model"}).assign(case="inner")
    panel = pd.concat([panel, average, screen], ignore_index=True)
    output = []
    for raw, transformed in (pairs.items() if isinstance(pairs, dict) else pairs):
        cases = panel[panel.model.isin([raw, transformed])][["fold", "case"]].drop_duplicates()
        for fold, case in cases.itertuples(index=False, name=None):
            scores = []
            for name in (raw, transformed):
                row = panel[panel.model.eq(name)&panel.fold.eq(fold)&panel.case.eq(case)]
                valid = len(row) == 1 and bool(row.iloc[0].certified) and bool(row.iloc[0].complete)
                scores.append(float(row.iloc[0].rmse) if valid and pd.notna(row.iloc[0].rmse) else None)
            a, b = scores
            complete = a is not None and b is not None and np.isfinite([a, b]).all()
            output.append(dict(raw=raw, transformed=transformed, fold=fold, case=case,
                complete=bool(complete), raw_rmse=a, transformed_rmse=b,
                delta_rmse=b-a if complete else None,
                relative_improvement=1-b/a if complete and a > 0 else None,
                interpretation="negative delta favors transformed; developmental, not significance"))
    return pd.DataFrame(output)


def report(path):
    cpu_guard()
    plan, root = load(path), Path(path).resolve().parent
    rules = plan_rules(plan)
    forms, candidates = forms_for(rules), rules["candidates"]
    destination = root / "report"
    if destination.exists():
        raise ValueError("create-only report exists")
    try:
        selection = selection_receipt(root, plan)
    except (ValueError, FileNotFoundError) as error:
        destination.mkdir()
        io.write_json_once(destination / "report.json", dict(status="incomplete", error=str(error),
            plan_sha256=plan["sha256"], outer_scores_computed=False, evidence=rules["evidence"]))
        return False
    baseline = complete_baseline_panel(pd.read_csv(verify(plan["baseline_metrics"])))
    if plan.get("prior_campaign"):
        baseline = pd.concat([baseline, prior_campaign_panel(plan["prior_campaign"])], ignore_index=True)
    if baseline.duplicated(["model", "fold", "case"]).any() or set(baseline.model)&set(forms):
        raise ValueError("historical and fresh comparison identifiers collide")
    paths = {name: verify(entry) for name, entry in plan["scoring"].items()}
    grid, endpoint = pd.read_parquet(paths["grid.parquet"]), pd.read_parquet(paths["endpoint.parquet"])
    metrics, outputs, gates, failures, diagnostics, inputs = [], [], [], [], [], []
    for fold in (*PRIMARY, "all_grid"):
        for name in forms:
            task, result = fitted_result(root, plan, name, fold)
            if result_path(root, task).exists():
                inputs.append(binding(result_path(root, task)))
            if not result.get("certified"):
                failures.append(dict(form=name, fold=fold, error=result.get("error")))
            else:
                diagnostics.append(dict(form=name, fold=fold, **result.get("geometry", {})))
            for case in (["L26"] if fold == "all_grid" else ["grid", "L26"]):
                if case == "grid":
                    rows = data.subset(grid, plan["memberships"][fold], "test")
                else:
                    languages = endpoint.language if fold == "all_grid" else plan["memberships"][fold]["score_languages"]
                    rows = endpoint[endpoint.language.isin(languages)].reset_index(drop=True)
                if result.get("certified"):
                    pred = predictions(result, name, rows)
                    score = checked_score(pred, rows, endpoint=case == "L26")
                    outputs.append(pred.assign(form=name, model=name, fold=fold, case=case))
                else:
                    score = dict(complete=False, rmse=None, rows=0, expected_rows=len(rows))
                row = dict(model=name, form=name, fold=fold, case=case,
                           role="candidate" if name in candidates else "matched_control",
                           category="new_candidate" if name in candidates else "internal_control",
                           certified=bool(result.get("certified")), **score)
                metrics.append(row)
                panel = baseline[baseline.fold.eq(fold)&baseline.case.eq(case)]
                for label, subset in (("published_form_adaptations", panel[panel.category.eq("published_form_adaptation")]),
                                      ("all_historical_including_internal", panel)):
                    gate = comparison_gate(row, subset)
                    gates.append(dict(form=name, fold=fold, case=case, panel=label, **gate))
    frame = pd.concat([baseline, pd.DataFrame(metrics)], ignore_index=True)
    means = pd.DataFrame(mean_metrics(frame))
    for name in forms:
        value = means[means.model.eq(name)].iloc[0]
        candidate = dict(certified=bool(value.complete), complete=bool(value.complete), rmse=value.mean_primary_rmse)
        for label, names in (("published_form_adaptations", set(baseline[baseline.category.eq("published_form_adaptation")].model)),
                             ("all_historical_including_internal", set(baseline.model))):
            panel = means[means.model.isin(names)].rename(columns={"mean_primary_rmse": "rmse"}).copy()
            panel["certified"] = panel.complete
            gates.append(dict(form=name, fold="mean_primary", case="mean_primary", panel=label,
                              **comparison_gate(candidate, panel)))
    destination.mkdir()
    frame.to_csv(destination / "metrics.csv", index=False)
    means.to_csv(destination / "mean_primary_rmse.csv", index=False)
    if rules.get("paired_ablation"):
        paired_metrics(frame, means, pd.read_csv(root / "inner_metrics.csv"),
                       rules["paired_ablation"]).to_csv(destination / "paired_ablation.csv", index=False)
    pd.DataFrame(gates).to_csv(destination / "gates.csv", index=False)
    overview = []
    for name, group in frame.groupby("model", sort=True):
        row = dict(model=name, shared=forms.get(name), selected=name == selection["chosen"]["form"])
        for fold in PRIMARY:
            values = group[group.case.eq("grid")&group.fold.eq(fold)]
            row[fold] = float(values.rmse.iloc[0]) if len(values) == 1 and bool(
                values.certified.iloc[0] and values.complete.iloc[0]) else None
        values = group[group.case.eq("L26")&group.fold.eq("all_grid")]
        row["L26_all_grid"] = float(values.rmse.iloc[0]) if len(values) == 1 and bool(
            values.certified.iloc[0] and values.complete.iloc[0]) else None
        average = means[means.model.eq(name)]
        row["mean_primary_rmse"] = float(average.mean_primary_rmse.iloc[0]) if len(average) == 1 and bool(
            average.complete.iloc[0]) else None
        overview.append(row)
    pd.DataFrame(overview).to_csv(destination / "comparison.csv", index=False)
    if outputs:
        pd.concat(outputs, ignore_index=True).to_parquet(destination / "predictions.parquet", index=False)
    io.write_json_once(destination / "fit_diagnostics.json", diagnostics)
    inputs += [binding(root / "selection.json"), plan["baseline_metrics"], *plan["scoring"].values()]
    if plan.get("prior_campaign"):
        inputs.extend(plan["prior_campaign"][key] for key in ("plan", "report", "metrics"))
    io.write_json_once(destination / "report.json", dict(status="complete" if not failures else "partial",
        chosen=selection["chosen"], plan_sha256=plan["sha256"], failures=failures,
        evaluated_candidates=list(candidates), selected_after_outer_scoring=False,
        primary_average="arithmetic mean of three configuration-balanced RMSEs; L26 separate",
        evidence=rules["evidence"], l26_independent_training_runs=1, inputs=inputs,
        new_significance_test=False, reserved_core_hours=plan["reserved_core_hours"],
        deferred_diagnostics=["configuration-block refit stability", "allocation regret", "full matched-Dt contrasts"],
        post_report_diagnostics="diagnostics/receipt.json" if rules.get("diagnostics_after_report") else None,
        no_winner_guarantee=True))
    if rules.get("diagnostics_after_report"):
        from .diagnostics import diagnose
        diagnostic_complete = diagnose(path)
        return bool(not failures and diagnostic_complete)
    return not failures
