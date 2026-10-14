"""Post-hoc residual panel for the accepted recovered baseline comparison.

Groups are descriptive views of the same held-out trajectories. They do not
change scoring support, select a law, or constitute new independent tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import execution, report as fit_report
from ..data import sha256_file
from .contracts import FOLDS, MODELS, seal, verify_seal


def summary(rows):
    per = rows.groupby("trajectory_id").agg(mse=("squared_error", "mean"),
        bias=("residual", "mean"))
    return {"rmse": float(np.sqrt(per["mse"].mean())),
        "mean_prediction_minus_observed": float(per["bias"].mean()),
        "rows": len(rows), "trajectories": len(per),
        "negative_residual_row_fraction": float((rows["residual"] < 0).mean())}


def build(comparison_path: Path, results, result_loader):
    comparison = execution.read_json(comparison_path)
    verify_seal(comparison, "report_sha256", label="accepted primary comparison")
    if comparison["status"] != "complete":
        raise ValueError("residual diagnostics require a completed comparison")
    frame, _, _ = fit_report._snapshot(Path(comparison["input_receipt"]["path"]))
    summaries = []
    for model in MODELS:
        for fold in FOLDS:
            result = result_loader(results[model][fold])
            rows, predictions = fit_report._prediction(frame, result["membership"], result["selected"], model)
            rows["residual"] = predictions - rows["adjusted_loss"]
            rows["squared_error"] = rows["residual"] ** 2
            counts = rows.groupby("trajectory_id")["trajectory_id"].transform("size")
            rows["fold_weighted_squared_error"] = rows["squared_error"] / counts / rows["trajectory_id"].nunique()
            total_mse = float(rows["fold_weighted_squared_error"].sum())
            expected_rmse = comparison["primary_metrics"][model][fold]["trajectory_balanced_rmse"]
            if not np.isclose(np.sqrt(total_mse), expected_rmse, rtol=0, atol=1e-12):
                raise ValueError("residual panel does not reproduce the accepted primary score")
            groups = [{"axis": "all", "value": "all", **summary(rows), "fold_mse_share": 1.0}]
            for axis in ("ladder", "E", "K", "D_total", "trajectory_id"):
                for value, subset in rows.groupby(axis):
                    groups.append({"axis": axis, "value": str(value), **summary(subset),
                        "fold_mse_share": float(subset["fold_weighted_squared_error"].sum() / total_mse)
                            if total_mse else 0.0})
            summaries.append({"model": model, "fold": fold, "groups": groups})
    return seal({"schema_version": "stage1_baseline_recovered_residuals/v1", "status": "complete",
        "comparison": {"path": str(comparison_path), "file_sha256": sha256_file(comparison_path),
                       "report_sha256": comparison["report_sha256"]},
        "review_only": True, "selects_model": False,
        "weighting": "equal trajectories, then equal scored rows within each trajectory",
        "warning": "Post-hoc views of already inspected holdouts. Horizon groups can differ in ladder support.",
        "residual_sign": "prediction minus observed; negative is overly optimistic",
        "summaries": summaries}, "report_sha256")
