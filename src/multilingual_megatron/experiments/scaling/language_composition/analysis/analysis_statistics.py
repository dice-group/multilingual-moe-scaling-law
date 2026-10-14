"""Seed-blocked empirical and prediction summaries, independent of law form."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import EXPERTS, HORIZONS, SEEDS

CELL = ["arm", "seed", "E", "horizon"]


def describe(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("report exactly three finite seed-block estimates")
    return {"mean": float(values.mean()), "sample_standard_deviation": float(values.std(ddof=1)),
            "range": [float(values.min()), float(values.max())], "n_independent_seeds": 3}


def validate_losses(rows):
    expected = set(itertools.product(("compact", "diverse", "reference"), SEEDS, EXPERTS, HORIZONS))
    if set(map(tuple, rows[CELL].drop_duplicates().to_numpy())) != expected:
        raise ValueError("incomplete empirical factorial")
    if rows.duplicated(CELL + ["language"]).any() or len(rows) != 162 * 196:
        raise ValueError("duplicate or missing language observations")
    if not rows.groupby(CELL).size().eq(196).all():
        raise ValueError("evaluation lacks K196 coverage")
    if not np.isfinite(rows[["raw_nll", "adjusted_loss", "unigram_nll"]]).all().all():
        raise ValueError("non-finite observed losses")
    if not np.allclose(rows.raw_nll - rows.unigram_nll, rows.adjusted_loss, rtol=0, atol=1e-12):
        raise ValueError("adjusted/raw loss identity failed")
    for column in ("target_hash", "unigram_hash", "token_count", "unigram_nll", "vocab_size", "special_tokens"):
        if not rows.groupby("language")[column].nunique().eq(1).all():
            raise ValueError(f"evaluation response changes between arms: {column}")


def empirical(rows, cohorts):
    validate_losses(rows)
    means = []
    for name, languages in cohorts.items():
        if not languages:
            continue  # The frozen compact_diverse-only cohort is empty.
        if not set(languages) <= set(rows.language):
            raise ValueError("missing cohort language")
        group = rows[rows.language.isin(languages)].groupby(CELL)[["adjusted_loss", "raw_nll"]].mean().reset_index()
        group["cohort"] = name
        group["language_count"] = len(languages)
        means.append(group)
    cells = pd.concat(means, ignore_index=True)
    contrasts = []
    for (cohort, horizon, seed), block in cells.groupby(["cohort", "horizon", "seed"]):
        table = block.pivot(index="arm", columns="E", values="adjusted_loss")
        gains = table[1] - table[32]
        contrasts.append({"cohort": cohort, "horizon": horizon, "seed": seed,
            **{f"{a}_moe_gain": float(gains[a]) for a in gains.index},
            "paired_diverse_minus_compact_gain": float(gains["diverse"] - gains["compact"])})
    return cells, pd.DataFrame(contrasts)


def arm_contrasts(rows, value_columns):
    keys = ["seed", "E", "horizon", "language"]
    selected = rows[rows.arm.isin(["compact", "diverse"])]
    if selected.duplicated(["arm"] + keys).any():
        raise ValueError("duplicate paired prediction row")
    left = selected[selected.arm.eq("compact")][keys + value_columns]
    right = selected[selected.arm.eq("diverse")][keys + value_columns]
    paired = left.merge(right, on=keys, suffixes=("_compact", "_diverse"), validate="one_to_one", how="outer", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("unpaired composition observations")
    result = paired[keys].copy()
    for c in value_columns:
        result[c] = paired[f"{c}_diverse"] - paired[f"{c}_compact"]
    return result


def prediction_metrics(predictions, spec):
    metrics, contrast_rows = [], []
    expected = set(itertools.product(SEEDS, EXPERTS, HORIZONS, spec["shared_languages"]))
    for (model, fold), rows in predictions.groupby(["model", "fold"]):
        contrasts = arm_contrasts(rows, ["prediction", "adjusted_loss"])
        if set(map(tuple, contrasts[["seed", "E", "horizon", "language"]].to_numpy())) != expected:
            raise ValueError("incomplete common7 prediction panel")
        contrasts["model"], contrasts["fold"] = model, fold
        contrasts["error"] = contrasts.prediction - contrasts.adjusted_loss
        contrast_rows.append(contrasts)
        for (seed, horizon), block in contrasts.groupby(["seed", "horizon"]):
            absolute = rows[rows.seed.eq(seed) & rows.horizon.eq(horizon)]
            residual = absolute.prediction - absolute.adjusted_loss
            expert = block.pivot(index="language", columns="E", values="error")
            metrics.append({"model": model, "fold": fold, "seed": seed, "horizon": horizon,
                "contrast_rmse": float(np.sqrt(np.mean(block.error**2))),
                "contrast_mae": float(np.mean(abs(block.error))),
                "absolute_rmse": float(np.sqrt(np.mean(residual**2))),
                "absolute_mae": float(np.mean(abs(residual))),
                "interaction_rmse": float(np.sqrt(np.mean((expert[1] - expert[32])**2))),
                "observed_shift_rms": float(np.sqrt(np.mean(block.adjusted_loss**2)))})
    metrics = pd.DataFrame(metrics)
    if set(zip(metrics.model, metrics.fold)) != set(itertools.product(spec["models"], spec["folds"])):
        raise ValueError("missing model/fit scores")
    scores = [c for c in metrics if c not in ("model", "fold", "seed", "horizon")]
    # Average SCORES, not predictions; fixed fits are sensitivity, not replication.
    seeds = metrics.groupby(["model", "seed", "horizon"])[scores].mean().reset_index()
    zero = seeds[seeds.model.eq(spec["models"][0])].copy()
    zero["model"] = "zero_shift"
    zero["contrast_rmse"] = zero.observed_shift_rms
    for col in ("absolute_rmse", "absolute_mae", "interaction_rmse", "contrast_mae"):
        zero[col] = np.nan  # zero-shift is a contrast control, not a loss predictor.
    seeds = pd.concat([seeds, zero], ignore_index=True)
    paired = metrics.pivot(index=["fold", "seed", "horizon"], columns="model", values="contrast_rmse")
    paired["full_minus_coverage"] = paired["M6-ND"] - paired["M5-ND"]
    differences = paired[["full_minus_coverage"]].reset_index()
    return metrics, seeds, pd.concat(contrast_rows, ignore_index=True), differences
