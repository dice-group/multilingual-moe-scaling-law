"""Score certified fixed-law fits; keep incomplete comparisons unavailable."""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as io
from ..law_search import data
from . import inputs, backend, runner
from ..scoring import checked_score, predict_result


def report(path):
    path = Path(path).resolve()
    plan = runner.load(path)
    if "full_paper_policy" in plan["profile"]:
        from .full_report import report as full_report
        return full_report(path)
    root = path.parent
    output = root / "report"
    if output.exists():
        raise ValueError("report is create-only; preserve the existing assessment")
    input_path = inputs.local_path(root, plan["input"])
    manifest = inputs.load(input_path)
    evidence = input_path.parent
    grid, endpoint = [pd.read_parquet(inputs.local_path(evidence, manifest["scoring"][name]))
                      for name in ("grid.parquet", "endpoint.parquet")]
    exposure = io.read_json(inputs.local_path(evidence, manifest["scoring"]["exposure_contract.json"]))
    inventories = {(e["language"], int(e["K"])): e["U"] for e in exposure["inventories"]}
    metrics, predictions, receipts = [], [], []
    for task in (t for t in plan["tasks"] if t["stage"] == "certify"):
        fold, name = task["fold"], task["model"]
        row = dict(model=name, fold=fold, case="L26" if fold == "all_grid" else "grid", certified=False, complete=False, rmse=None)
        result_path = root / "results" / f"{task['id']:05d}.json"
        if not result_path.exists():
            metrics.append(row | {"error": "missing certification receipt"})
            continue
        result = runner.receipt(path, plan, task["id"])
        receipts.append(inputs.local_binding(result_path, root))
        if not result.get("certified") or not result.get("complete"):
            metrics.append(row | {"error": result.get("error", "numerical certification failed")})
            continue
        rows = endpoint.copy() if fold == "all_grid" else data.subset(grid, manifest["memberships"][fold], "test")
        if name in inputs.NATIVE:
            universe = (endpoint if fold == "all_grid" else grid)
            universe = universe.drop(columns=list(data.RESPONSES & set(universe)))
            predicted = predict_result(result, rows, universe, inventories, exposure["metadata"])
        else:
            _, _, _, calibration = inputs.read_training(input_path, fold)
            predicted = rows[data.KEYS+["adjusted_loss"]].copy()
            predicted["prediction"], _ = backend.predict_shared(name, result["selected"], rows, calibration["regularization"])
        score = checked_score(predicted, rows, endpoint=fold == "all_grid")
        metrics.append(row | score | dict(certified=True))
        predictions.append(predicted.assign(model=name, fold=fold, case=row["case"]))
    frame = pd.DataFrame(metrics)
    available = frame[frame.certified & frame.complete]
    table = available.pivot(index="model", columns="fold", values="rmse").reindex(index=plan["models"], columns=inputs.FOLDS)
    table = table.rename(columns={"all_grid": "L26_all_grid"})
    table["mean_primary_rmse"] = table[list(inputs.PRIMARY)].mean(axis=1, skipna=False)
    reference_gap, matches = None, None
    if plan.get("reference"):
        reference = pd.read_csv(inputs.local_path(root, plan["reference"])).set_index("model")
        reference_gap = table-reference.reindex(index=table.index, columns=table.columns)
        matches = bool(np.isfinite(reference_gap.to_numpy()).all() and (reference_gap.abs() <= 5.0001e-6).all().all())
    output.mkdir()
    frame.to_csv(output / "metrics.csv", index=False)
    table.to_csv(output / "comparison.csv", index_label="model")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    if reference_gap is not None:
        reference_gap.to_csv(output / "reference_rmse_difference.csv", index_label="model")
    summary = dict(plan_sha256=plan["sha256"], models=plan["models"], complete_cells=len(available),
        expected_cells=len(frame), all_cells_certified=len(available) == len(frame),
        matches_reference_to_display_precision=matches, receipts=receipts,
        outputs=[inputs.local_binding(p, root) for p in sorted(output.iterdir())],
        scope="fixed-law coefficient refits; configuration-balanced primary RMSE; equal-language one-run L26; no reselection",
        calibration="training-only calibration and declared parent start anchors reused; not recalibrated",
        manuscript_changed=False)
    io.write_json_once(output / "report.json", inputs.sealed(summary))
    return summary
