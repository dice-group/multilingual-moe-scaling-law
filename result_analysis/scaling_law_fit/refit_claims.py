"""Replay remaining manuscript diagnostics from a certified full-paper refit.

Preserves the historical interpolation knots, populations and weights. This
module evaluates frozen coefficients; it does not optimize or select a law.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dependency_replay import sha, write_json, KEYS
from .residuals import matched_sizes
from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import replay
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs, protocol


def run(profile_path, output):
    cfg = json.loads(profile_path.read_text())["replay"]
    root = Path(cfg["refit_plan"]).resolve().parent
    report_path = Path(cfg["refit_report"]).resolve()
    report = inputs.read_sealed(report_path)
    assert report["all_cells_certified"] and report["all_scoring_complete"]
    bound = {}

    def read(path):
        path = Path(path).resolve(); bound[str(path)] = sha(path)
        if path.suffix == ".parquet": return pd.read_parquet(path)
        if path.suffix == ".csv": return pd.read_csv(path)
        return json.loads(path.read_text())

    read(profile_path); read(report_path)
    bound[str(Path(__file__).resolve())] = sha(Path(__file__))
    bound[str(Path(replay.__file__))] = sha(Path(replay.__file__))
    bound[str(Path(matched_sizes.__code__.co_filename))] = sha(Path(matched_sizes.__code__.co_filename))
    plan = read(root / "plan.json")
    manifest_path = inputs.local_path(root, plan["input"])
    manifest = read(manifest_path)
    for entry in report["outputs"]: inputs.local_path(root, entry)
    grid = read(inputs.local_path(manifest_path.parent, manifest["scoring"]["grid.parquet"]))
    endpoint = read(inputs.local_path(manifest_path.parent, manifest["scoring"]["endpoint.parquet"]))
    coefficients = read(report_path.parent / "coefficients.json")["fits"]
    records = {(r["model"], r["fold"]): r["result"]["selected"] for r in coefficients
               if r["arm"] == "lbfgsb_box" and "selected" in r["result"]}
    saved = read(report_path.parent / "predictions.parquet")
    saved = saved[saved.arm.eq("lbfgsb_box")]
    ep = Path(cfg["evidence"])
    downstream = Path(cfg["output"])
    assert read(downstream / "completion.json")["complete"]
    output.mkdir(parents=True, exist_ok=False)

    def predict(name, fold, rows):
        selected = records[name, fold]
        x = selected["coordinates"]
        duration = name in ("J0-ND", "M5-ND", "M6-ND", "M6")
        selected = dict(selected, shared_parameters=asdict(fit.shared_from_outer(
            "M6" if duration else name, x[:14] if duration else x)), zeta=x[14] if duration else 0.)
        return replay.shared_equation(selected, rows)

    panels = []
    parity = 0.
    for name in ("J0", "J0-ND", "M1", "M2", "M5-ND", "M6", "M6-ND"):
        for fold in protocol.PRIMARY:
            languages = {r["language"] for r in records[name, fold]["language_coefficients"]}
            rows = grid[grid.language.isin(languages)].copy()
            rows["prediction"] = predict(name, fold, rows)
            rows["model"], rows["fold"] = name, fold
            rows["residual"] = rows.prediction - rows.adjusted_loss
            rows["squared_error"] = rows.residual**2
            check = saved[saved.model.eq(name) & saved.fold.eq(fold) & saved.case.eq("grid")].merge(
                rows[KEYS + ["prediction"]], on=KEYS, validate="one_to_one", suffixes=("", "_replay"))
            parity = max(parity, float(abs(check.prediction-check.prediction_replay).max()))
            panels.append(rows)
    assert parity <= 1e-10
    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(output / "primary_predictions.parquet", index=False)
    heldout = saved[saved.case.eq("grid") & saved.fold.isin(protocol.PRIMARY)].merge(
        grid[KEYS+["D_total", "K", "E", "ladder"]], on=KEYS, validate="many_to_one")
    heldout["squared_error"] = (heldout.prediction-heldout.adjusted_loss)**2
    heldout["residual"] = heldout.prediction-heldout.adjusted_loss
    horizon = heldout.groupby(["model", "fold", "D_total", "trajectory_id"])[["squared_error", "residual"]].mean()
    horizon = horizon.groupby(["model", "fold", "D_total"]).mean().reset_index()
    horizon["rmse"] = np.sqrt(horizon.squared_error)
    horizon.to_csv(output / "horizon_metrics.csv", index=False)
    sizes = matched_sizes(panel[panel.model.isin(["J0", "M6", "M6-ND"])], replicates=10000, seed=20260905)
    (output / "reports").mkdir()
    sizes.to_csv(output / "reports/matched_sizes.csv", index=False)

    # Reuse the registered reference brackets; independently reconstruct truth.
    gaps = read(ep / "combined_baseline/matched_gap_points.csv")
    indexed = panel.set_index(["fold", "model", "readout_id", "language"])
    def values(column, field):
        index = pd.MultiIndex.from_arrays([gaps.fold, gaps.model, gaps[field], gaps.language], names=indexed.index.names)
        return indexed[column].loc[index].to_numpy()
    w = gaps.reference_upper_log_weight.to_numpy()
    def contrast(column):
        return values(column, "target_readout_id") - (1-w)*values(column, "reference_lower_readout_id") - w*values(column, "reference_upper_readout_id")
    np.testing.assert_allclose(contrast("adjusted_loss"), gaps.observed, atol=1e-12, rtol=0)
    gaps["historical_prediction"] = gaps.predicted
    gaps["predicted"] = contrast("prediction")
    gaps.to_csv(output / "matched_gap_points.csv", index=False)
    macro = gaps.groupby(["model", "fold", "ladder", "E", "D_total"], as_index=False)[["observed", "predicted"]].mean()
    macro = macro[~macro.fold.eq("E32") | macro.E.eq(32)].copy()
    macro["squared_error"] = (macro.predicted-macro.observed)**2
    by_size = macro.groupby(["model", "fold", "ladder"]).squared_error.mean()
    contrasts = by_size.groupby(["model", "fold"]).mean().pow(.5).unstack("fold")
    contrasts.to_csv(output / "expansion_gap_rmse.csv")

    # Earlier shared-law rows in the policy table use the same held-out panel.
    features = read(downstream / "policy_inputs.parquet")
    truth = read(downstream / "policy_observed.parquet")
    context = read(ep / "primary_fits/policy_context.json")
    policy = []
    for name in ("J0", "J0-ND", "M6"):
        p = features[["config", "language", "FLOPs", "N_total"]].copy()
        p["prediction"] = predict(name, "K200_common_languages", features)
        for cohort, languages in context["cohorts"].items():
            means = p[p.language.isin(languages)].groupby(["config", "FLOPs", "N_total"], as_index=False).prediction.mean()
            obs = truth[truth.language.isin(languages)].groupby(["config", "FLOPs", "reference"], as_index=False)["observed"].mean()
            joined = means.merge(obs, on=["config", "FLOPs"], validate="many_to_many")
            for cap in (.4e9, .7e9, 2.2e9):
                for (reference, budget), g in joined[joined.N_total.le(cap)].groupby(["reference", "FLOPs"]):
                    chosen, best = g.loc[g.prediction.idxmin()], g.loc[g.observed.idxmin()]
                    policy.append(dict(model=name, cohort=cohort, reference=reference, cap=cap,
                        FLOPs=budget, chosen=chosen.config, best=best.config, regret=chosen.observed-best.observed))
    policy = pd.concat([pd.DataFrame(policy), read(downstream / "policy_decisions.csv")], ignore_index=True)
    policy.to_csv(output / "policy_decisions.csv", index=False)
    metrics = policy.groupby(["model", "cohort", "reference", "cap"], as_index=False).agg(
        mean_regret=("regret", "mean"), max_regret=("regret", "max"))
    metrics.to_csv(output / "policy_metrics.csv", index=False)

    # The legacy eight-knot comparison interpolates predictions at saved readouts.
    old = read(ep / "combined_baseline/allocation_candidates.csv")
    common = context["cohorts"]["common12"]
    knot_rows = []
    for row in old.itertuples():
        ladder, expert = row.config.split("/E")
        g = panel[panel.model.eq(row.model) & panel.fold.eq("K200_common_languages") &
                  panel.ladder.eq(ladder) & panel.E.eq(int(expert)) & panel.K.eq(row.K) & panel.language.isin(common)]
        values_new, observed = [], []
        for _, language in g.groupby("language"):
            language = language.sort_values("D_t")
            q = row.FLOPs / (6*language.N_act.iloc[0])
            assert language.D_total.min()*(1-1e-12) <= q <= language.D_total.max()*(1+1e-12)
            dt = np.interp(q, language.D_total, language.D_t)
            values_new.append(np.interp(np.log(dt), np.log(language.D_t), language.prediction))
            observed.append(np.interp(np.log(dt), np.log(language.D_t), language.adjusted_loss))
        assert len(values_new) == 12
        assert abs(np.mean(observed)-row.observed) < 1e-10
        knot_rows.append(dict(model=row.model, config=row.config, K=row.K, FLOPs=row.FLOPs,
                              prediction=float(np.mean(values_new)), observed=row.observed))
    knots = pd.DataFrame(knot_rows)
    decisions = []
    for (name, k, budget), g in knots.groupby(["model", "K", "FLOPs"]):
        chosen, best = g.loc[g.prediction.idxmin()], g.loc[g.observed.idxmin()]
        decisions.append(dict(model=name, K=k, FLOPs=budget, chosen=chosen.config, best=best.config,
                              correct=chosen.config == best.config, regret=chosen.observed-best.observed))
    pd.DataFrame(decisions).to_csv(output / "knots_decisions.csv", index=False)

    # Retain, rather than hide, the composition law's late-duration errors.
    tail = heldout[heldout.fold.eq("largest_N_act") & heldout.E.eq(32) & heldout.K.eq(12) & heldout.D_total.gt(31e9)]
    tail = tail.groupby("model").squared_error.mean().pow(.5).to_dict()
    large = saved[saved.case.eq("L26") & saved.fold.eq("E32") & saved.model.eq("M6-ND")].merge(
        endpoint[KEYS+["D_t", "U_t"]], on=KEYS, validate="one_to_one")
    large["squared_error"] = (large.prediction-large.adjusted_loss)**2
    reuse = {}
    for label, mask in [("over_four", large.D_t.gt(4*large.U_t)),
                        ("one_to_four", large.D_t.gt(large.U_t) & large.D_t.le(4*large.U_t))]:
        g = large[mask]
        reuse[label] = dict(languages=len(g), rmse=float(np.sqrt(g.squared_error.mean())),
                            squared_error_share=float(g.squared_error.sum()/large.squared_error.sum()))
    checks = dict(complete=True, report_sha256=report["sha256"], prediction_oracle_max_error=parity,
                  largest_size_32B_E32_K12_rmse=tail, L26_E32_M6ND_reuse_buckets=reuse,
                  inputs=bound, coefficient_fitting=False)
    for path, digest in bound.items(): assert sha(Path(path)) == digest
    checks["outputs"] = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "receipt.json", checks)
    return dict(complete=True, output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.profile.resolve(), args.output.resolve()))
