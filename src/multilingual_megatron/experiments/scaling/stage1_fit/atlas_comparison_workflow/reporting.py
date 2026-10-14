"""Coverage-first evaluation: an unresolved comparator is never a baseline loss."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import atlas_full_pilot as io
from ..execution import write_text_once
from ..m6_size_duration_workflow.reporting import metrics_and_bootstrap
from . import campaign as c, model as m, numerics as n

KEYS = ["fold", "trajectory_id", "readout_id", "language"]


def audit(path, *, loader=None):
    plan = (loader or c.load)(path, verify_inputs=True)
    coverage, errors, missing = [], [], []
    for task in range(plan["task_count"]):
        target = c.task_path(plan, task)
        if not target.exists():
            missing.append(task)
            continue
        value = io.read(target)
        if value["plan_sha256"] != plan["sha256"] or value["task"] != task:
            raise ValueError("result/plan identity mismatch")
        packet_index, spec_index = divmod(task, len(c.PANEL))
        packet = io.read(plan["packets"][packet_index]["path"])
        spec = c.PANEL[spec_index]
        valid = False
        reason = value.get("worker_error", "no certified selected fit")
        if "selected" in value and value["selected"] is not None:
            if value["packet_sha256"] != packet["sha256"] or value["spec"] != spec.payload():
                raise ValueError("result has different packet or equation")
            selected = value["selected"]
            v, g, _, _ = n.objective(spec, selected["coordinates"], packet["train"])
            kkt = n.stationarity(spec, np.asarray(selected["coordinates"]), g, packet["support"])
            valid = (value["certified"] and abs(v-selected["objective"]) <= 1e-12
                     and kkt <= 1e-8 and n.agreement(spec, selected, value["audit"], packet["train"]))
            if valid and "recovery" in value:
                from . import recovery_numerics as recovery
                partner=value["audit"]; ax=np.asarray(partner["coordinates"])
                av,ag,_,_=n.objective(spec,ax,packet["train"])
                minimum=value["recovery"]["minimum_feasible_objective"]
                lo,hi=m.bounds(spec)
                valid=bool(abs(av-partner["objective"])<=1e-12 and n.stationarity(spec,ax,ag,packet["support"])<=1e-8
                    and np.all(ax>=lo) and np.all(ax<=hi) and m.domain(spec,ax,packet["support"])[0].min()>=0
                    and recovery.pair(spec,[selected,partner],packet["train"],minimum)[1] is not None)
            reason = "certified" if valid else "stationarity or independent-audit disagreement"
        if not valid:
            errors.append(task)
        coverage.append(dict(task=task, model=spec.name, fold=packet["fold"], language=packet["language"],
            certified=bool(valid), reason=reason, receipt=io.bind(target)))
    target = Path(plan["output_root"])/"audit.json"
    return io.write(target, dict(plan_sha256=plan["sha256"], coverage=coverage, missing_tasks=missing,
        unresolved_tasks=errors, complete=not missing, all_certified=not missing and not errors,
        heldout_scores_computed=False))


def assert_same_rows(reference, candidate):
    if candidate.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise ValueError("duplicate scored observation")
    if set(map(tuple, candidate[KEYS].to_numpy())) != set(map(tuple, reference[KEYS].to_numpy())):
        raise ValueError("incomplete or changed scoring population")


def rmse(frame, column="squared_error"):
    return float(np.sqrt(frame.groupby("trajectory_id")[column].mean().mean()))


def candidate_predictions(spec, result, packet, tolerance):
    """Actual whole-target coefficient sets, never a row-wise oracle envelope."""
    best = result["selected"]
    accepted, coordinates = [], set()
    for row in [best, result["audit"], *result.get("audit_history", []), *result["attempts"], *result["probes"]]:
        if "objective" not in row or not row.get("optimizer_success"):
            continue
        if abs(row["objective"]-best["objective"]) > tolerance*max(abs(best["objective"]), 1e-15):
            continue
        key = tuple(row["coordinates"])
        if key in coordinates:
            continue
        lo, hi = m.bounds(spec)
        x = np.asarray(key)
        if np.any(x < lo) or np.any(x > hi) or m.domain(spec, x, packet["support"])[0].min() < 0:
            continue
        accepted.append(m.predict(spec, x, packet["test"])[0]-packet["unigram_nll"])
        coordinates.add(key)
    return np.asarray(accepted)


def evaluate(path, *, loader=None):
    plan = (loader or c.load)(path, verify_inputs=True)
    protocol=plan["protocol"]
    root = Path(plan["output_root"])
    checked = io.read(root/"audit.json")
    if checked["plan_sha256"] != plan["sha256"]:
        raise ValueError("audit belongs to another plan")
    output = root/"reports"
    if output.exists():
        raise ValueError("reports are create-only")
    output.mkdir()
    base = pd.read_parquet(plan["historical_predictions"]["path"])
    base = base[base.split.eq("test")].copy()
    reference = base[base.model.eq("M6-ND")]
    obs = pd.read_parquet(plan["observations"]["path"])
    assert_same_rows(reference, obs)
    coverage = pd.DataFrame(checked["coverage"])
    coverage.drop(columns=["receipt"], errors="ignore").to_csv(output/"numerical_coverage.csv", index=False)
    all_rows, status, sensitivity = [base], [], []
    certified_ids = set(coverage.loc[coverage.certified, "task"]) if not coverage.empty else set()
    for spec_index, spec in enumerate(c.PANEL):
        ids = [i*len(c.PANEL)+spec_index for i in range(len(plan["packets"]))]
        unresolved = [task for task in ids if task not in certified_ids]
        if unresolved:
            status.append(dict(model=spec.name, status="unresolved", unresolved_targets=len(unresolved)))
            continue
        parts, ambiguity = [], {tol: [] for tol in n.SETTINGS["near_optimal_tolerances"]}
        for packet_index, task in enumerate(ids):
            packet = io.read(plan["packets"][packet_index]["path"])
            if not packet["test"]["N"]:
                continue
            result = io.read(c.task_path(plan, task))
            rows = obs[obs.fold.eq(packet["fold"]) & obs.language.eq(packet["language"])].copy()
            expected = list(zip(packet["test"]["trajectory_ids"], packet["test"]["readout_ids"]))
            if list(zip(rows.trajectory_id, rows.readout_id)) != expected:
                raise ValueError("prediction order differs from bound observations")
            rows["prediction"] = m.predict(spec, result["selected"]["coordinates"], packet["test"])[0]-packet["unigram_nll"]
            rows["model"], rows["split"] = spec.name, "test"
            rows["residual"] = rows.prediction-rows.adjusted_loss
            rows["squared_error"] = rows.residual**2
            parts.append(rows)
            for tol in ambiguity:
                candidates = candidate_predictions(spec, result, packet, tol)
                if candidates.size == 0:
                    raise ValueError("certified winner missing from admissible prediction set")
                errors = (candidates-rows.adjusted_loss.to_numpy()[None, :])**2
                # Whole-target coefficients selected for this diagnostic only.
                # Use the actual fold's row weights, not equal target weights.
                fold_reference = reference[reference.fold.eq(packet["fold"])]
                sizes = fold_reference.groupby("trajectory_id").size()
                weights = 1./rows.trajectory_id.map(sizes).to_numpy()
                scores = errors @ weights
                a = rows[KEYS].copy()
                a["best_for_baseline_squared_error"] = errors[int(np.argmin(scores))]
                a["worst_for_baseline_squared_error"] = errors[int(np.argmax(scores))]
                a["max_prediction_range"] = np.max(candidates, axis=0)-np.min(candidates, axis=0)
                ambiguity[tol].append(a)
        frame = pd.concat(parts, ignore_index=True)
        assert_same_rows(reference, frame)
        all_rows.append(frame)
        status.append(dict(model=spec.name, status="certified", unresolved_targets=0))
        for tol, parts in ambiguity.items():
            frame_range = pd.concat(parts, ignore_index=True)
            for fold, group in frame_range.groupby("fold"):
                sensitivity.append(dict(model=spec.name, fold=fold, relative_training_tolerance=tol,
                    best_for_baseline_rmse=rmse(group, "best_for_baseline_squared_error"),
                    worst_for_baseline_rmse=rmse(group, "worst_for_baseline_squared_error"),
                    maximum_prediction_range=float(group.max_prediction_range.max()),
                    interpretation="whole-target coefficient-set oracle sensitivity, NOT deployable selection or confidence interval"))
    combined = pd.concat(all_rows, ignore_index=True)
    stats = metrics_and_bootstrap(combined, replicates=protocol["bootstrap_replicates"], seed=protocol["bootstrap_seed"])
    for name, values in plan["historical_metrics"].items():
        if any(abs(stats["metrics"][name][f]-values[f]) > 1e-12 for f in (*c.FOLDS, "mean")):
            raise ValueError("historical scores changed during evaluation")
    combined.to_parquet(output/"predictions.parquet", index=False)
    pd.DataFrame(status).to_csv(output/"model_status.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output/"prediction_sensitivity.csv", index=False)
    residuals, language_metrics = [], []
    for keys, group in combined.groupby(["model", "fold", "E", "K", "D_total"]):
        residuals.append(dict(zip(("model", "fold", "E", "K", "D_total"), keys)) |
                         dict(rmse=rmse(group), bias=float(group.groupby("trajectory_id").residual.mean().mean())))
    for (name, fold, language), group in combined.groupby(["model", "fold", "language"]):
        denominator = float(np.sum((group.adjusted_loss-group.adjusted_loss.mean())**2))
        language_metrics.append(dict(model=name, fold=fold, language=language,
            rmse=rmse(group), r2=None if denominator <= 1e-15 else float(1-group.squared_error.sum()/denominator)))
    pd.DataFrame(residuals).to_csv(output/"horizon_expert_coverage_residuals.csv", index=False)
    pd.DataFrame(language_metrics).to_csv(output/"per_language_metrics.csv", index=False)
    metrics = stats["metrics"]
    references = ["C0", "J0", "M0"]+[s.name for s in c.PANEL]
    complete = all(name in metrics for name in references)
    wins = {name: all(metrics["M6-ND"][f] < metrics[name][f] for f in c.FOLDS)
            for name in references if name in metrics}
    intervals = {item["reference"]: item for item in stats["paired_intervals"]}
    aggregate_supported = all(intervals[name]["p975"] < 0 for name in references if name in intervals)
    sensitivity_win = all(metrics["M6-ND"][row["fold"]] < row["best_for_baseline_rmse"] for row in sensitivity)
    decision = dict(panel_complete=complete, per_baseline_all_fold_wins=wins,
        comparison_scope=protocol["scope"], original_65_variant_panel_complete=False,
        deferred_models=[s.name for s in c.DEFERRED],
        all_evaluated_baselines_win=bool(complete and all(wins.values())),
        robust_advantage=bool(complete and all(wins.values()) and aggregate_supported and sensitivity_win),
        historical_long_horizon_gate="FAIL, unchanged; late largest-model E32 K12/K50 regressions remain",
        conclusion="No all-baseline claim while any required comparison is unresolved." if not complete else
                   "Use exact fold/interval/sensitivity results; no universal-law or independent-confirmation claim.")
    summary = ["# ATLAS / multilingual MoE baseline comparison", "", protocol["evidence"], "",
        protocol["scope"], "",
        f"Required native/control panel complete: **{complete}**. Robust advantage within that panel: **{decision['robust_advantage']}**.", "",
        "| Law | Largest model | E32 | K200 common | Mean RMSE |", "|---|---:|---:|---:|---:|"]
    for name, values in metrics.items():
        summary.append(f"| {name} | "+" | ".join(f"{values[f]:.6f}" for f in (*c.FOLDS, "mean"))+" |")
    unresolved_names = [s["model"] for s in status if s["status"] == "unresolved"]
    summary += ["", f"Unresolved comparator variants: {len(unresolved_names)}; see model_status.csv and numerical_coverage.csv.",
        "", "C0 uses target-language tokens, not total tokens. J0 uses raw log(E) and target tokens. "
        "JH-local uses the published fixed saturating expert transformation and total tokens. "
        "All new forms have per-language coefficients; historical C0/J0/M0 share nonlinear shapes. "
        "All 12 constructed ATLAS-moe variants are deferred, not defeated. Total counts are configured design counts, not runtime-exact measurements.",
        "", "Signed weights and numerical caps are disclosed study choices. A failed numerical audit is not a baseline loss. "
        "Coefficient-set oracle ranges are conservative sensitivities, not fitted selectors or confidence intervals.",
        "", "M6-ND's previously identified late largest-model E32/K12 and E32/K50 regressions remain. "
        "These comparisons do not establish general superiority over ATLAS on its original data or new mixtures.", ""]
    write_text_once(output/"summary.md", "\n".join(summary))
    write_text_once(output/"manuscript_comparison.tex", manuscript(stats, decision, len(unresolved_names)))
    return io.write(output/"report.json", dict(plan_sha256=plan["sha256"], audit_sha256=checked["sha256"],
        status="complete" if complete else "comparison_unresolved", statistics=stats, decision=decision,
        unresolved_models=unresolved_names, evidence=protocol["evidence"],
        outputs=[io.bind(p) for p in sorted(output.iterdir()) if p.is_file()]))


def manuscript(stats, decision, unresolved):
    """Evidence-gated prose, never manufacture a completed ATLAS win."""
    text = ("% Generated from the sealed ATLAS comparison; development evidence.\n"
        "\\paragraph{Expanded literature-form comparison.}\n"
        "We distinguish ATLAS's adaptive-transfer/repetition law from its separate language-count/capacity law. "
        "All new fits use training-only responses on the same trajectory-held-out splits; "
        "numerical transfer-weight domains, coefficient pooling, and active versus total parameter mappings are disclosed separately. "
        "The 53-variant native/control panel explicitly defers the 12 constructed ATLAS--JointMoE variants; "
        "they are not published ATLAS equations and no comparison against them is claimed.\n")
    if not decision["panel_complete"]:
        text += (f"The expanded comparison remains incomplete or uncertified ({unresolved} required variants); "
            "we therefore make no claim of outperforming full ATLAS or all literature baselines.\n")
    elif decision["robust_advantage"]:
        text += ("M6-ND has lower RMSE on each of the three evaluated holdouts than every comparator in this frozen panel, "
            "with a supported aggregate paired difference and robustness to the tested coefficient sensitivities.\n")
    else:
        text += "The completed panel does not support a robust all-baseline superiority claim; results must be stated by holdout.\n"
    return text+("These are post-selection development results on previously inspected data, not independent confirmation. "
        "The previously observed late-horizon largest-model E32/K12 and E32/K50 regressions remain limitations.\n")
