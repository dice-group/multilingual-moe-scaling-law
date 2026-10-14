"""Descriptive residual statistics; no optimizer or law-selection interface."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.contracts import FOLDS

FOCUS_MODELS = ("J0", "M2", "M6")


def summary(rows: pd.DataFrame) -> dict:
    per = rows.groupby("trajectory_id", sort=True).agg(
        mse=("squared_error", "mean"), bias=("residual", "mean"),
        influence=("huber_influence_ratio", "mean"))
    return {"rows": len(rows), "trajectories": len(per),
        "rmse": float(np.sqrt(per.mse.mean())), "bias": float(per.bias.mean()),
        "mean_huber_influence_ratio": float(per.influence.mean()),
        "negative_residual_row_fraction": float(rows.residual.lt(0).mean()),
        "clipped_row_fraction": float(rows.huber_influence_ratio.lt(1).mean())}


def residual_groups(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    axes = [("ladder",), ("E",), ("K",), ("D_total",), ("language",),
            ("ladder", "D_total"), ("D_total", "E"), ("D_total", "K"), ("D_total", "E", "K")]
    for (model, fold, split), rows in frame.groupby(["model", "fold", "split"], sort=True):
        total_mse = float(np.dot(rows.weight, rows.squared_error))
        total_huber = float(np.dot(rows.weight, rows.huber_loss))
        views = [("all", "all", rows)]
        for columns in axes:
            for values, group in rows.groupby(list(columns), observed=True, sort=True):
                views.append(("/".join(columns), "/".join(str(v) for v in values), group))
        for axis, value, group in views:
            records.append({"model": model, "fold": fold, "split": split, "axis": axis, "value": value,
                **summary(group), "mse_share": float(np.dot(group.weight, group.squared_error) / total_mse) if total_mse else 0.,
                "huber_loss_share": float(np.dot(group.weight, group.huber_loss) / total_huber) if total_huber else 0.,
                "base_weight_share": float(group.weight.sum()),
                "mean_total_tokens_per_active_parameter": float(group.total_tokens_per_active_parameter.mean()),
                "mean_target_tokens_per_active_parameter": float(group.target_tokens_per_active_parameter.mean())})
    return pd.DataFrame(records)


def bootstrap_mean(values: np.ndarray, *, replicates: int, seed: int) -> tuple[float, float]:
    """Resample complete trajectory or complete matched-trajectory-pair means."""
    if not len(values):
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(samples, [.025, .975]))


def matched_sizes(frame: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    """Exact language/E/K/token matches only. Unmatched horizons stay visible."""
    frame = frame[frame.fold.eq("largest_N_act")]
    keys = ["E", "K", "language", "D_total"]
    exposures = [name for name in ["D_t", "D_family", "D_script", "D_other"] if name in frame]
    columns = keys + ["trajectory_id", "residual", "prediction", "adjusted_loss"] + exposures
    records = []
    for model, rows in frame.groupby("model", sort=True):
        for left_ladder, right_ladder in combinations(sorted(rows.ladder.unique()), 2):
            left = rows[rows.ladder.eq(left_ladder)][columns]
            right = rows[rows.ladder.eq(right_ladder)][columns]
            if left.duplicated(keys).any() or right.duplicated(keys).any():
                raise ValueError("ambiguous exact size matching")
            pairs = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
            for horizon in sorted(set(left.D_total) | set(right.D_total)):
                pair = pairs[pairs.D_total.eq(horizon)].copy()
                nl, nr = int(left.D_total.eq(horizon).sum()), int(right.D_total.eq(horizon).sum())
                row = {"model": model, "left_ladder": left_ladder, "right_ladder": right_ladder,
                    "D_total": horizon, "left_rows": nl, "right_rows": nr, "matched_rows": len(pair),
                    "left_coverage": len(pair) / nl if nl else 0., "right_coverage": len(pair) / nr if nr else 0.,
                    "matching": "exact language, E, K and actual D_total; no interpolation"}
                if not pair.empty:
                    exposure_difference = max((float((
                        (pair[f"{name}_right"] - pair[f"{name}_left"]).abs()
                        / np.maximum(pair[f"{name}_left"].abs(), 1.)).max()) for name in exposures), default=0.)
                    row["maximum_relative_exposure_difference"] = exposure_difference
                    row["target_and_component_exposure_matched"] = exposure_difference <= 1e-12
                    pair["error_gap"] = pair.residual_right - pair.residual_left
                    pair["observed_size_gain"] = pair.adjusted_loss_left - pair.adjusted_loss_right
                    pair["predicted_size_gain"] = pair.prediction_left - pair.prediction_right
                    # One stratum contains the entire left/right trajectory pair.
                    per = pair.groupby(["trajectory_id_left", "trajectory_id_right"], sort=True)[
                        ["error_gap", "observed_size_gain", "predicted_size_gain"]].mean()
                    lower, upper = bootstrap_mean(per.error_gap.to_numpy(), replicates=replicates, seed=seed)
                    row.update({"trajectory_pairs": len(per), "right_minus_left_bias": float(per.error_gap.mean()),
                        "p025": lower, "p975": upper, "observed_size_gain": float(per.observed_size_gain.mean()),
                        "predicted_size_gain": float(per.predicted_size_gain.mean())})
                records.append(row)
    return pd.DataFrame(records)


def learning_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Secants between actual validation readouts, not derivatives or replicates."""
    rows = frame[frame.fold.eq("largest_N_act")].sort_values(["model", "trajectory_id", "language", "D_total"]).copy()
    ids = ["model", "trajectory_id", "language"]
    for name in ("D_total", "adjusted_loss", "prediction"):
        rows[f"previous_{name}"] = rows.groupby(ids, sort=False)[name].shift()
    rows = rows[rows.previous_D_total.notna()].copy()
    span = np.log(rows.D_total / rows.previous_D_total)
    if (span <= 0).any():
        raise ValueError("non-increasing or duplicate learning-curve horizon")
    rows["observed_slope"] = (rows.adjusted_loss - rows.previous_adjusted_loss) / span
    rows["predicted_slope"] = (rows.prediction - rows.previous_prediction) / span
    rows["slope_error"] = rows.predicted_slope - rows.observed_slope
    records = []
    for keys, group in rows.groupby(["model", "ladder", "previous_D_total", "D_total"], sort=True):
        per = group.groupby("trajectory_id")[["observed_slope", "predicted_slope", "slope_error"]].mean()
        records.append(dict(zip(["model", "ladder", "start_D_total", "end_D_total"], keys)) |
            {"rows": len(group), "trajectories": len(per), **{k: float(v) for k, v in per.mean().items()}})
    return pd.DataFrame(records)


def pairwise_bootstrap(frame: pd.DataFrame, *, replicates: int, seed: int) -> dict:
    """Shared-union trajectory resampling preserves overlapping-fold dependence."""
    rows = frame[frame.split.eq("test") & frame.model.isin(FOCUS_MODELS)]
    per = rows.groupby(["model", "fold", "trajectory_id"]).squared_error.mean()
    union = sorted(rows.trajectory_id.unique())
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(union), np.full(len(union), 1 / len(union)), size=replicates)
    observed, bootstrap, fold_rmse = {}, {}, {}
    support = None
    for model in FOCUS_MODELS:
        samples, metrics, this_support = [], {}, []
        for fold in FOLDS:
            values = per.loc[(model, fold)].reindex(union)
            mask = values.notna().to_numpy()
            this_support.append(mask)
            denominator = counts[:, mask].sum(axis=1)
            score = np.sqrt((counts[:, mask] @ values.to_numpy()[mask]) / np.maximum(denominator, 1))
            score[denominator == 0] = np.nan
            samples.append(score)
            metrics[fold] = float(np.sqrt(values.mean()))
        if support is not None and not np.array_equal(support, this_support):
            raise ValueError("paired bootstrap requires identical trajectory support")
        support = this_support
        observed[model] = float(np.mean(list(metrics.values())))
        bootstrap[model] = np.mean(samples, axis=0)
        fold_rmse[model] = metrics
    comparisons = []
    for reference, candidate in combinations(FOCUS_MODELS, 2):
        diff = bootstrap[candidate] - bootstrap[reference]
        valid = diff[np.isfinite(diff)]
        comparisons.append({"reference": reference, "candidate": candidate,
            "candidate_minus_reference_rmse": observed[candidate] - observed[reference],
            "p025": float(np.quantile(valid, .025)), "p975": float(np.quantile(valid, .975)),
            "valid_replicates": len(valid)})
    return {"replicates": replicates, "seed": seed, "unit": "whole trajectory, shared weights across overlapping folds",
        "observed_equal_fold_rmse": observed, "fold_rmse": fold_rmse, "comparisons": comparisons,
        "interpretation": "Descriptive 95% intervals conditional on retained fits; no correction for prior model search."}


def exposure_bins(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    axes = ["total_tokens_per_active_parameter", "target_tokens_per_active_parameter", "total_tokens_per_nonembedding_parameter"]
    for axis in axes:
        # Fixed count, loss-blind log spacing; common bin edges for all models/folds.
        values = frame[axis].to_numpy()
        edges = np.geomspace(values.min() * (1 - 1e-12), values.max() * (1 + 1e-12), 9)
        for (model, fold, split), rows in frame.groupby(["model", "fold", "split"], sort=True):
            for number in range(8):
                group = rows[(rows[axis] >= edges[number]) & (rows[axis] < edges[number + 1])]
                if not group.empty:
                    records.append({"axis": axis, "lower": edges[number], "upper": edges[number + 1],
                        "model": model, "fold": fold, "split": split, **summary(group)})
    return pd.DataFrame(records)


def analyze(frame: pd.DataFrame, *, replicates: int, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    return {"residual_groups": residual_groups(frame),
        "matched_sizes": matched_sizes(frame, replicates=replicates, seed=seed),
        "learning_slopes": learning_slopes(frame), "exposure_bins": exposure_bins(frame)}, \
        pairwise_bootstrap(frame, replicates=replicates, seed=seed)


def write_plots(output: Path, frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = tables["residual_groups"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, model in zip(axes, FOCUS_MODELS):
        rows = frame[frame.model.eq(model) & frame.fold.eq("largest_N_act")]
        for ladder, part in rows.groupby("ladder", sort=True):
            points = []
            for horizon, h in part.groupby("D_total", sort=True):
                points.append((horizon / 1e9, summary(h)["bias"]))
            ax.plot(*np.array(points).T, marker="o", label=f"{ladder} ({'held out' if part.split.eq('test').all() else 'fit'})")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(xscale="log", xlabel="Total training tokens (B)", title=model)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Prediction − observation\nnegative = overly optimistic")
    fig.suptitle("Largest-size fold: training-size and held-out-size residuals")
    fig.tight_layout()
    fig.savefig(output / "size_horizon_bias.png", dpi=160)
    fig.savefig(output / "size_horizon_bias.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True, layout="constrained")
    for ax, model in zip(axes, FOCUS_MODELS):
        part = groups[groups.model.eq(model) & groups.fold.eq("largest_N_act") & groups.split.eq("test") & groups.axis.eq("D_total/E")].copy()
        part[["horizon", "E"]] = part.value.str.split("/", expand=True).astype(float)
        pivot = part.pivot(index="E", columns="horizon", values="bias")
        im = ax.imshow(pivot, aspect="auto", cmap="coolwarm", vmin=-.15, vmax=.15)
        ax.set_xticks(range(len(pivot.columns)), [f"{v / 1e9:.3g}" for v in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
        ax.set(xlabel="Total tokens (B)", title=model)
    axes[0].set_ylabel("Experts")
    fig.colorbar(im, ax=axes, label="Prediction − observation", shrink=.8)
    fig.suptitle("Largest held-out model: expert × horizon error")
    fig.savefig(output / "largest_expert_horizon.png", dpi=160)
    fig.savefig(output / "largest_expert_horizon.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    pairs = tables["matched_sizes"]
    pairs = pairs[pairs.left_ladder.eq("C") & pairs.right_ladder.eq("D") & pairs.matched_rows.gt(0)]
    observed = pairs[pairs.model.eq("M6")].sort_values("D_total")
    ax.plot(observed.D_total / 1e9, observed.observed_size_gain, color="black", marker="o", linewidth=2, label="Observed")
    for model in FOCUS_MODELS:
        part = pairs[pairs.model.eq(model)].sort_values("D_total")
        ax.plot(part.D_total / 1e9, part.predicted_size_gain, linestyle="--", label=f"{model} predicted")
    ax.set(xscale="log", xlabel="Total training tokens (B), exact matched support",
        ylabel="Loss reduction from size C to D", title="Size benefit changes with training duration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "matched_size_gain.png", dpi=160)
    fig.savefig(output / "matched_size_gain.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def markdown(report: dict, tables: dict[str, pd.DataFrame]) -> str:
    groups = tables["residual_groups"]
    lines = ["# Largest-model extrapolation: reproduced diagnostic metrics", "",
        report["claim_boundary"], "", "No new laws were fitted; M6 is fixed at lambda=0.01.", "",
        "## Accepted held-out scores", "",
        "| Law | Largest size | E32 | K200 common | Equal-fold mean RMSE |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for model in ("C0", "M0", "J0", "M1", "M2", "M6"):
        part = groups[groups.model.eq(model) & groups.split.eq("test") & groups.axis.eq("all")].set_index("fold")
        values = [part.loc[f].rmse for f in FOLDS]
        lines.append(f"| {model} | " + " | ".join(f"{v:.8f}" for v in [*values, np.mean(values)]) + " |")
    lines += ["", "C0 = harmonized Chinchilla; M0 = ATLAS count/capacity form, not full transfer/repetition ATLAS; "
        "J0 = harmonized JointMoE; M1/M2/M6 = developed extensions.", "",
        "## Largest-size heldout: horizon breakdown", "",
        "| Law | Total tokens (B) | Bias | RMSE | Fold MSE share |", "| --- | ---: | ---: | ---: | ---: |"]
    part = groups[groups.model.isin(FOCUS_MODELS) & groups.fold.eq("largest_N_act") & groups.split.eq("test") & groups.axis.eq("D_total")]
    for row in part.itertuples():
        lines.append(f"| {row.model} | {float(row.value)/1e9:g} | {row.bias:.5f} | {row.rmse:.5f} | {row.mse_share:.1%} |")
    lines += ["", "## Training-objective diagnostic", "",
        "The following is evaluated on training rows, not held-out rows. Huber influence ratio is "
        "min(1, delta/abs(error)); it is a relative gradient influence, not a sample exclusion.", "",
        "| Law | Total tokens (B) | Training RMSE | Clipped rows | Mean influence ratio | Squared-error share | Huber-loss share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    part = groups[groups.model.isin(FOCUS_MODELS) & groups.fold.eq("largest_N_act") & groups.split.eq("train") & groups.axis.eq("D_total")]
    for row in part.itertuples():
        lines.append(f"| {row.model} | {float(row.value)/1e9:g} | {row.rmse:.5f} | {row.clipped_row_fraction:.1%} | "
            f"{row.mean_huber_influence_ratio:.5f} | {row.mse_share:.1%} | {row.huber_loss_share:.1%} |")
    lines += ["", "## Interpretation safeguards", "",
        "- Training residuals are in-sample; test residuals are already inspected development holdouts.",
        "- Matched-size CSV includes missing overlap and uses exact language/E/K/actual-token matching, with no extrapolation.",
        "- Learning slopes are secants between cooldown readouts sharing trunks, not independent learning trajectories.",
        "- Local numerical full-rank/basin audits do not establish global parameter identification or a causal mechanism.",
        "- Training telemetry is release-bound and restricted to each readout's branch and available parent prefix.",
        "- M6 predictor/core and component-builder bytes are verified. The older orchestration receipt hash differs "
        "from the retained runner snapshot; this report does not certify replay of that historical runner.", "",
        "See `report.json` for source bindings and descriptive paired 95% intervals; `training_evidence.csv`, "
        "`matched_sizes.csv`, `learning_slopes.csv`, and `exposure_bins.csv` contain the supporting panels.", ""]
    return "\n".join(lines)
