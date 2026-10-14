"""Seed-blocked composition scores; ported from the 2026-09-21 replay.

Only the two required estimators are retained. Fits are averaged as scores,
not treated as independent training repetitions.
"""
import itertools

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import EXPERTS, HORIZONS, SEEDS


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
