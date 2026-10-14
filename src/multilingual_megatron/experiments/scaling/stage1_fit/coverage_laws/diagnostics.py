"""Frozen fixed-coefficient contrast checks; never fit or select equations."""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as io
from ..law_search import data
from ..artifacts import binding, verify
from . import replay
from .state import cpu_guard, load
from .contracts import forms_for
from .reporting import fitted_result, selection_receipt

FIELDS = ("adjusted_loss", "prediction", "floor", "capacity", "data", "repetition_correction")


def contributions(name, record, rows):
    result = rows.copy()
    result["floor"], result["capacity"], result["data"] = replay.components(name, record, rows)
    result["prediction"] = result.floor+result.capacity+result.data
    coordinates = list(record["coordinates"])
    coordinates[15] = 0.
    no_repeat = record | dict(coordinates=coordinates)
    result["repetition_correction"] = result.prediction-replay.predict(name, no_repeat, rows)
    return result


def expert_coverage(rows, regime):
    """Four D-size corners, common target support; never cross an inventory boundary."""
    corners = rows[rows.trajectory_id.str.startswith("D:") & rows.K.isin([12, 200]) & rows.E.isin([1, 32])]
    records = []
    for language, group in corners.groupby("language"):
        if regime == "pre_repeat":
            group = group[group.D_t <= group.U_t]
        elif regime == "repeated":
            group = group[group.D_t > group.U_t]
        elif regime != "all_support":
            raise ValueError("unknown exposure regime")
        curves = {key: v.sort_values("D_t") for key, v in group.groupby(["K", "E"])}
        if set(curves) != {(12, 1), (12, 32), (200, 1), (200, 32)} or any(len(c) < 2 for c in curves.values()):
            continue
        if group.U_t.nunique() != 1 or any(c.D_t.duplicated().any() for c in curves.values()):
            raise ValueError("ambiguous contrast inventory or exposure")
        low, high = max(c.D_t.min() for c in curves.values()), min(c.D_t.max() for c in curves.values())
        if low >= high:
            continue
        for fraction in np.linspace(0., 1., 5):
            logd = (1-fraction)*np.log(low)+fraction*np.log(high)
            record = dict(language=language, fraction=fraction, D_t=np.exp(logd), U_t=group.U_t.iloc[0])
            for field in FIELDS:
                values = {key: np.interp(logd, np.log(c.D_t), c[field]) for key, c in curves.items()}
                record[field] = values[200, 1]-values[200, 32]-values[12, 1]+values[12, 32]
            records.append(record)
    return records


def size_duration(rows):
    """C-D gain change at first/last exact matched post-pass nodes; at least two."""
    sizes = rows[rows.trajectory_id.str.match(r"[CD]:")]
    for _, group in sizes.groupby(["E", "K", "language"]):
        if group.U_t.nunique() != 1:
            raise ValueError("size contrast requires equal invariant C/D inventory")
    post = rows[rows.D_t > rows.U_t]
    keys = ["E", "K", "language", "D_t", "U_t"]
    c, d = [post[post.trajectory_id.str.startswith(f"{ladder}:")] for ladder in ("C", "D")]
    matched = c.merge(d, on=keys, suffixes=("_C", "_D"), validate="one_to_one")
    records = []
    for (e, k, language), group in matched.groupby(["E", "K", "language"]):
        group = group.sort_values("D_t")
        if len(group) < 2:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        record = dict(E=e, K=k, language=language, anchors=len(group),
                      early_D_t=first.D_t, late_D_t=last.D_t)
        for field in FIELDS:
            record[f"change_{field}"] = (last[f"{field}_C"]-last[f"{field}_D"])-(
                first[f"{field}_C"]-first[f"{field}_D"])
        records.append(record)
    return records


def diagnose(path):
    cpu_guard()
    plan, root = load(path), Path(path).resolve().parent
    selection_receipt(root, plan)
    report_path = root / "report" / "report.json"
    report = io.read_json(report_path)
    if report["plan_sha256"] != plan["sha256"]:
        raise ValueError("foreign score report")
    destination = root / "diagnostics"
    if destination.exists():
        raise ValueError("create-only diagnostics exist")
    grid = pd.read_parquet(verify(plan["scoring"]["grid.parquet"]))
    saved_path = root / "report" / "predictions.parquet"
    saved = pd.read_parquet(saved_path)
    inputs = [binding(report_path), binding(saved_path), plan["scoring"]["grid.parquet"]]
    contrasts, sizes, failures = [], [], []
    for fold in ("largest_N_act", "all_grid"):
        for name in forms_for(plan["rules"]):
            task, result = fitted_result(root, plan, name, fold)
            if not result.get("certified"):
                failures.append(dict(form=name, fold=fold, error=result.get("error")))
                continue
            from .state import result_path
            inputs.append(binding(result_path(root, task)))
            rows = contributions(name, result["selected"], grid)
            if fold == "largest_N_act":
                check = saved[saved.model.eq(name)&saved.fold.eq(fold)&saved.case.eq("grid")]
                joined = check.merge(rows, on=data.KEYS, validate="one_to_one", suffixes=("_saved", "_replay"))
                if len(joined) != len(check):
                    raise ValueError("saved prediction coverage changed")
                np.testing.assert_allclose(joined.prediction_saved, joined.prediction_replay, rtol=0, atol=1e-10)
            for regime in ("all_support", "pre_repeat", "repeated"):
                contrasts.extend(dict(model=name, fit_fold=fold, regime=regime, **r)
                                 for r in expert_coverage(rows, regime))
            sizes.extend(dict(model=name, fit_fold=fold, **r) for r in size_duration(rows))
    contrast = pd.DataFrame(contrasts, columns=["model", "fit_fold", "regime", "language",
        "fraction", "D_t", "U_t", *FIELDS])
    size = pd.DataFrame(sizes, columns=["model", "fit_fold", "E", "K", "language", "anchors",
        "early_D_t", "late_D_t", *(f"change_{field}" for field in FIELDS)])
    summaries = []
    for (name, fold, regime), group in contrast.groupby(["model", "fit_fold", "regime"]):
        summaries.append(dict(model=name, fit_fold=fold, regime=regime, languages=group.language.nunique(),
            points=len(group), observed=group.adjusted_loss.mean(), predicted=group.prediction.mean(),
            capacity=group.capacity.mean(), data=group.data.mean(),
            repetition_correction=group.repetition_correction.mean(),
            rmse=np.sqrt(np.mean((group.prediction-group.adjusted_loss)**2))))
    size["squared_error"] = (size.change_prediction-size.change_adjusted_loss)**2
    groupkeys = ["model", "fit_fold", "E", "K"]
    value_columns = [c for c in size if c.startswith("change_")]+["squared_error"]
    size[value_columns] = size[value_columns].astype(float)
    blocks = size.groupby(groupkeys)[value_columns].mean().reset_index()
    size_summary = blocks.groupby(["model", "fit_fold"]).squared_error.mean().pow(.5).rename("block_balanced_rmse").reset_index()
    destination.mkdir()
    for name, frame in (("contrast_points", contrast), ("contrast_summary", pd.DataFrame(summaries)),
                        ("size_duration_points", size), ("size_duration_blocks", blocks), ("size_duration_summary", size_summary)):
        frame.to_csv(destination / f"{name}.csv", index=False)
    io.write_json_once(destination / "receipt.json", dict(plan_sha256=plan["sha256"], inputs=inputs,
        outputs=[binding(p) for p in sorted(destination.glob("*.csv"))], failures=failures,
        contrast_points=len(contrast), size_duration_pairs=len(size),
        complete=bool(not failures and len(contrast) and len(size)),
        fitting=False, selection=False, endpoint_responses_read=False,
        evidence="developmental nested contrasts, not independent seeds; largest-size C is training and D held out; all_grid is in-sample",
        historical_missing_contrast_gate_replaced=False))
    return bool(not failures and len(contrast) and len(size))
