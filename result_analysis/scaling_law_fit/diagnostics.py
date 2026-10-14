"""Prediction diagnostics with matched populations and trajectory weighting."""
import numpy as np
import pandas as pd

KEYS = ["fold", "trajectory_id", "readout_id", "language"]


def matched_rows(left, right):
    """Reject missing, extra, or duplicate observations before comparing laws."""
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise ValueError("duplicate prediction membership")
    joined = left.merge(right, on=KEYS, how="outer", validate="one_to_one",
                        suffixes=("_5", "_6"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("prediction scoring populations differ")
    return joined.drop(columns="_merge")


def composition_diagnostics(predictions, metadata):
    heldout = predictions.loc[predictions.split.eq("test")]
    columns = KEYS + ["squared_error"]
    paired = matched_rows(heldout.loc[heldout.model.eq("M5-ND"), columns],
                          heldout.loc[heldout.model.eq("M6-ND"), columns])
    meta = metadata.loc[metadata.tier.eq("K200")].copy()
    if meta.language.duplicated().any():
        raise ValueError("duplicate language metadata")
    paired = paired.merge(meta, on="language", how="left", validate="many_to_one")
    if paired["name"].isna().any():
        raise ValueError("missing language metadata")
    paired["family"] = paired.family.fillna("Unassigned")
    paired["resource_availability"] = paired.resource_availability.fillna("UNKNOWN").str.upper()
    records = []
    for fold, rows in paired.groupby("fold", sort=True):
        rows = rows.copy()
        rows["w"] = 1 / (rows.trajectory_id.nunique() * rows.groupby("trajectory_id").language.transform("size"))
        if not np.isclose(rows.w.sum(), 1):
            raise ValueError("fold weights do not sum to one")
        for dimension in ("all", "language", "resource_availability", "family", "production_script"):
            groups = [("All", rows)] if dimension == "all" else rows.groupby(dimension, sort=True, dropna=False)
            for group, part in groups:
                weight = float(part.w.sum())
                mse5 = float(np.sum(part.w * part.squared_error_5) / weight)
                mse6 = float(np.sum(part.w * part.squared_error_6) / weight)
                records.append(dict(fold=fold, dimension=dimension, group=group,
                    rows=len(part), languages=part.language.nunique(), trajectories=part.trajectory_id.nunique(),
                    scoring_weight=weight, rmse_m5=np.sqrt(mse5), rmse_m6=np.sqrt(mse6),
                    rmse_change=np.sqrt(mse6)-np.sqrt(mse5),
                    fold_mse_contribution=weight*(mse6-mse5)))
    result = pd.DataFrame(records)
    for fold, rows in result.groupby("fold"):
        total = rows.loc[rows.dimension.eq("all"), "fold_mse_contribution"].iloc[0]
        for dimension, part in rows.loc[~rows.dimension.eq("all")].groupby("dimension"):
            np.testing.assert_allclose(part.fold_mse_contribution.sum(), total, rtol=1e-10, atol=1e-14)
    return result


def horizon_diagnostics(predictions):
    rows = predictions.loc[predictions.split.eq("test")]
    keys = ["model", "fold", "D_total", "trajectory_id"]
    main = rows.groupby(keys)[["squared_error", "residual"]].mean().groupby(keys[:-1]).mean().reset_index()
    main["rmse"] = np.sqrt(main.squared_error)
    tail = rows.loc[rows.ladder.eq("D") & rows.E.eq(32) & rows.K.isin([12, 50])]
    keys = ["model", "fold", "K", "D_total", "trajectory_id"]
    tail = tail.groupby(keys)[["squared_error", "residual"]].mean().groupby(keys[:-1]).mean().reset_index()
    tail["rmse"] = np.sqrt(tail.squared_error)
    return main, tail
