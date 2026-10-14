"""Full-paper coefficient/prediction export; numerical success is not paper adoption."""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as io
from ..law_search import data
from ..scoring import checked_score, predict_result
from . import inputs, runner, protocol, numerics


def shared_predictions(task, result, rows, universe, inventories, metadata, input_path):
    _, _, training_packets, cal = inputs.read_training(input_path, task["fold"])
    packets = []
    if task["model"].startswith("Pooled-"):
        for item in training_packets:
            idx = np.flatnonzero(rows.language.eq(item["language"]))
            if not len(idx):
                continue
            packet = data.donor_covariates(rows.iloc[idx], universe, inventories, metadata,
                                          item["language"], item["donors"])
            packet["training_minimum"] = item["train"]["training_minimum"]
            packets.append(dict(indices=idx.tolist(), train=packet))
    model = numerics.make_model(task["model"], rows, None, packets, cal, task["arm"])
    values = numerics.predict(model, result["selected"])
    independent = numerics.independent(model, result["selected"], rows, packets)
    np.testing.assert_allclose(values, independent, atol=1e-10, rtol=0)
    return rows[data.KEYS+["adjusted_loss"]].assign(prediction=values)


def reference_comparison(predictions, reference_path):
    """Compare exact populations; absent historical cells remain explicitly absent."""
    reference = pd.read_parquet(reference_path)
    rank = {"final_law": 0, "baselines": 1, "expert_ablation": 2, "coverage_ablation": 3}
    reference = reference.assign(priority=reference.fit_campaign.map(rank)).sort_values("priority")
    keys = ["model", "fold", "case", *data.KEYS]
    reference = reference.drop_duplicates(keys, keep="first")
    records = []
    for (model, fold, case, arm), part in predictions.groupby(["model", "fold", "case", "arm"]):
        old = reference[reference.model.eq(model) & reference.fold.eq(fold) & reference.case.eq(case)]
        paired = part.merge(old[keys+["adjusted_loss", "prediction"]], on=keys, how="outer",
                            suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
        matching = bool(paired._merge.eq("both").all())
        record = dict(model=model, fold=fold, case=case, arm=arm, matching_population=matching)
        if matching:
            np.testing.assert_allclose(paired.adjusted_loss_new, paired.adjusted_loss_old, atol=1e-12, rtol=0)
            record["max_prediction_change"] = float(abs(paired.prediction_new-paired.prediction_old).max())
            record["old_rmse"] = checked_score(old, old, endpoint=case == "L26")["rmse"]
            record["new_rmse"] = checked_score(part, part, endpoint=case == "L26")["rmse"]
            record["rmse_change"] = record["new_rmse"]-record["old_rmse"]
        records.append(record)
    return pd.DataFrame(records)


def report(path, *, recovery_path=None, output=None):
    path = Path(path).resolve(); root = path.parent
    plan = runner.load(path)
    from .recovery import verified_overrides
    overrides = verified_overrides(path, recovery_path)
    output = Path(output).resolve() if output is not None else root / "report"
    if not output.is_relative_to(root):
        raise ValueError("report must be inside the run directory")
    if output.exists():
        raise ValueError("report is create-only")
    input_path = inputs.local_path(root, plan["input"])
    manifest = inputs.load(input_path)
    evidence = input_path.parent
    grid, endpoint = [pd.read_parquet(inputs.local_path(evidence, manifest["scoring"][n]))
                      for n in ("grid.parquet", "endpoint.parquet")]
    exposure = io.read_json(inputs.local_path(evidence, manifest["scoring"]["exposure_contract.json"]))
    inventories = {(e["language"], int(e["K"])): e["U"] for e in exposure["inventories"]}
    metrics, predictions, coefficients, diagnostics = [], [], [], []
    for task in (t for t in plan["tasks"] if t["stage"] == "certify"):
        identity = {k: task[k] for k in ("model", "fold", "arm")}
        result_path = root / "results" / f"{task['id']:05d}.json"
        if task["id"] in overrides:
            result_path, result = overrides[task["id"]]
        else:
            result = runner.receipt(path, plan, task["id"]) if result_path.exists() else {}
        complete = bool(result.get("certified") and result.get("complete"))
        diagnostics.append(identity | dict(certified=complete, error=result.get("error"),
            effective_exponents=result.get("effective_exponents"),
            positive_effective_exponents=result.get("positive_effective_exponents"),
            training_objective=(result.get("selected") or {}).get("objective")))
        coefficients.append(identity | dict(result=result, receipt=str(result_path.relative_to(root))))
        cases = ["L26"] if task["fold"] == "all_grid" else ["grid"]
        if task["fold"] in protocol.PRIMARY:
            cases.append("L26")
        for case in cases:
            row = identity | dict(case=case, certified=complete, complete=False, rmse=None)
            if not complete:
                metrics.append(row); continue
            rows = endpoint.copy() if case == "L26" else data.subset(grid, manifest["memberships"][task["fold"]], "test")
            if case == "L26" and task["fold"] != "all_grid":
                rows = rows[rows.language.isin(manifest["memberships"][task["fold"]]["score_languages"])].reset_index(drop=True)
            universe = endpoint if case == "L26" else grid
            universe = universe.drop(columns=list(data.RESPONSES & set(universe)))
            try:
                if task["model"] in protocol.NATIVE:
                    predicted = predict_result(result, rows, universe, inventories, exposure["metadata"])
                else:
                    predicted = shared_predictions(task, result, rows, universe, inventories, exposure["metadata"], input_path)
                score = checked_score(predicted, rows, endpoint=case == "L26")
                metrics.append(row | score)
                predictions.append(predicted.assign(**identity, case=case))
            except (ValueError, FloatingPointError, AssertionError) as error:
                metrics.append(row | dict(error=str(error)))
    frame = pd.DataFrame(metrics)
    prediction = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    output.mkdir()
    frame.to_csv(output / "metrics.csv", index=False)
    if len(prediction):
        prediction.to_parquet(output / "predictions.parquet", index=False)
    io.write_json_once(output / "coefficients.json", dict(fits=coefficients))
    io.write_json_once(output / "fit_diagnostics.json", dict(fits=diagnostics))
    for arm in protocol.POLICIES:
        valid = frame[frame.arm.eq(arm) & frame.certified & frame.complete & (
            (frame.case.eq("grid") & frame.fold.isin(protocol.PRIMARY)) |
            (frame.case.eq("L26") & frame.fold.eq("all_grid")))]
        table = valid.pivot(index="model", columns="fold", values="rmse").reindex(
            index=plan["models"], columns=(*protocol.PRIMARY, "all_grid"))
        table["mean_primary_rmse"] = table[list(protocol.PRIMARY)].mean(axis=1, skipna=False)
        table.rename(columns={"all_grid": "L26_all_grid"}).to_csv(output / f"comparison_{arm}.csv", index_label="model")
    if plan.get("reference_predictions") and len(prediction):
        reference_comparison(prediction, inputs.local_path(root, plan["reference_predictions"])).to_csv(
            output / "reference_prediction_comparison.csv", index=False)
    paired = frame.pivot(index=["model", "fold", "case"], columns="arm", values="rmse")
    paired.to_csv(output / "paired_solver_scores.csv")
    pending = ["fresh composition contrasts and paired configuration bootstrap",
               "fresh interaction/expansion-gap diagnostics",
               "fresh allocation surfaces, supported policy regret and doubling diagnostics",
               "replay measured findings, preservation, seed/LR, benchmark and routing evidence",
               "claim-by-claim manuscript/appendix and figure/table review"]
    summary = dict(plan_sha256=plan["sha256"], fit_cells=len(diagnostics),
        recovery=inputs.local_binding(Path(recovery_path).resolve(), root) if recovery_path else None,
        certified_fit_cells=sum(d["certified"] for d in diagnostics),
        all_cells_certified=all(d["certified"] for d in diagnostics),
        all_scoring_complete=bool(frame.complete.all()), manuscript_changed=False,
        pending_paper_adoption_gates=pending, paper_ready=False,
        scope="all named manuscript laws and associated inner checks; historical search and neural training are not repeated",
        outputs=[inputs.local_binding(p, root) for p in sorted(output.iterdir())])
    io.write_json_once(output / "report.json", inputs.sealed(summary))
    return summary
