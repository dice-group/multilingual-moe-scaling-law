"""Re-score fixed predictions, retain selection boundaries, and replay coefficients."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from result_analysis import evidence as io
from result_analysis.checks import Checks
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import replay
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import replay as law_search_replay
from . import composition as analysis_statistics
from . import plots

FOLDS = ("largest_N_act", "E32", "K200_common_languages")
KEYS = ["trajectory_id", "readout_id", "language"]
SELECTED = "RN13-psi-tied"
EXPLORATORY = "RN14-stored-reuse"


from .analysis import grid_scores, endpoint_scores


def run(evidence: io.Evidence, output: Path) -> dict:
    output = io.output_directory(output, evidence)
    current = evidence.analysis.get("version") == "stage1-paper-v2"
    main_law = evidence.analysis["main_law"] if current else SELECTED
    checks = Checks()
    selection = evidence.json("law/selection.json")
    if selection["chosen"]["form"] != SELECTED:
        raise ValueError("selected law changed; update the explicit paper contract")
    paper = evidence.analysis if current else evidence.json("paper/scripts/selected_law_metrics.json")
    if not current and io.sha256(evidence.path("law/report/comparison.csv")) != paper["sha256"]:
        raise ValueError("paper-to-campaign binding changed")
    saved = evidence.table("law/report/predictions.parquet")
    grid = saved[saved.case.eq("grid") & saved.fold.isin(FOLDS)]
    baseline = evidence.table("baselines/predictions.parquet")
    status = evidence.table("baselines/metrics.csv")
    status.to_csv(output / "baseline_status.csv", index=False)
    valid = status.loc[status.certified.eq(True) & status.complete.eq(True), ["model", "fold", "case"]]
    if valid.duplicated().any():
        raise ValueError("duplicate baseline certification identity")
    baseline = baseline.merge(valid, on=["model", "fold", "case"], validate="many_to_one")
    base = baseline[baseline.case.eq("grid") & baseline.fold.isin(FOLDS) & ~baseline.model.isin(grid.model.unique())]
    scored = pd.concat([grid, base], ignore_index=True)
    table = grid_scores(scored).reindex(columns=FOLDS)
    table = table.reindex(sorted(set(table.index) | set(status.model)))
    table["mean_primary_rmse"] = table[list(FOLDS)].mean(axis=1, skipna=False)
    endpoint = saved[saved.case.eq("L26") & saved.fold.eq("all_grid")]
    old = baseline[baseline.case.eq("L26") & baseline.fold.eq("all_grid") & ~baseline.model.isin(endpoint.model.unique())]
    table["L26_all_grid"] = endpoint_scores(pd.concat([endpoint, old], ignore_index=True))
    original = evidence.table("law/report/comparison.csv").set_index("model")
    columns = [*FOLDS, "mean_primary_rmse", "L26_all_grid"]
    paper_models = [r["model"] for r in paper["main_table"] if r.get("model")] if current else [r["model"] for r in paper["rows"]]
    np.testing.assert_allclose(table.loc[paper_models, columns], original.loc[paper_models, columns],
                               atol=2e-12, rtol=0, equal_nan=True)

    # Refuse cross-population comparisons for every model in the main table.
    memberships = grid.loc[grid.model.eq(SELECTED), ["fold", *KEYS, "adjusted_loss"]]
    for model in paper_models:
        other = scored.loc[scored.model.eq(model), ["fold", *KEYS, "adjusted_loss"]]
        paired = memberships.merge(other, on=["fold", *KEYS], how="outer", validate="one_to_one", indicator=True)
        if not paired._merge.eq("both").all():
            raise ValueError(f"different scoring population for {model}")
        np.testing.assert_allclose(paired.adjusted_loss_x, paired.adjusted_loss_y, atol=1e-12, rtol=0)

    # Re-evaluate both named laws from saved coefficients, independently of optimizer code.
    features = evidence.table("scoring/grid.parquet")
    large = evidence.table("scoring/endpoint.parquet")
    plan = evidence.json("law/plan.json")
    replay_checks = []
    for model in (SELECTED, EXPLORATORY):
        for fold in (*FOLDS, "all_grid"):
            record = evidence.json(f"law/results/certify--{fold}--{model}.json")
            if not record["certified"] or not record["complete"] or record["plan_sha256"] != plan["sha256"]:
                raise ValueError(f"uncertified fit: {model}/{fold}")
            part = saved[saved.model.eq(model) & saved.fold.eq(fold)
                         & saved.case.eq("L26" if fold == "all_grid" else "grid")]
            source = large if fold == "all_grid" else features
            join_keys = ["language"] if fold == "all_grid" else KEYS
            merged = part[join_keys + ["prediction"]].merge(source, on=join_keys, validate="one_to_one")
            if len(merged) != len(part) or merged.empty:
                raise ValueError("coefficient replay population changed")
            predicted = replay.predict(model, record["selected"], merged)
            delta = float(np.max(abs(predicted - merged.prediction.to_numpy())))
            np.testing.assert_allclose(predicted, merged.prediction, atol=1e-10, rtol=0)
            replay_checks.append(dict(model=model, fold=fold, rows=len(part), max_abs_difference=delta))
    pd.DataFrame(replay_checks).to_csv(output / "coefficient_replay.csv", index=False)
    table["role"] = "comparison (see original campaign provenance)"
    table.loc[SELECTED, "role"] = "inner-selected"
    table.loc[EXPLORATORY, "role"] = "exploratory; outer scores known"
    if current:
        table.loc[SELECTED, "role"] = "historical inner-selected control"
        table.loc[main_law, "role"] = "final proposed law; coefficient-held-out developmental evaluation"
    table.to_csv(output / "all_available_scores.csv", index_label="model")
    main = table.loc[paper_models]
    main.to_csv(output / "paper_validation.csv", index_label="model")
    main[columns].to_latex(output / "paper_validation.tex", float_format="%.5f", escape=True)
    if current:
        displayed = manuscript_scores(evidence.path("paper/tables/main/table_validation.tex").read_text())
        checks.require("main-table model order", list(displayed.index) == paper_models,
                       source="paper/tables/main/table_validation.tex")
        checks.close("all 40 current main-table RMSE values", main[columns], displayed[columns],
                     atol=5.0001e-6, source="paper/tables/main/table_validation.tex",
                     scope="three configuration-balanced folds, their arithmetic mean, equal-language L26")
        checks.close("certified coefficient replay", [r["max_abs_difference"] for r in replay_checks],
                     np.zeros(len(replay_checks)), source="law/results", atol=1e-10)

    comp_predictions = evidence.table("replay/composition_predictions.parquet")
    spec = evidence.json("study/composition_prediction/inputs/spec.json")
    spec["models"] = sorted(comp_predictions.model.unique())
    _, seeds, _, _ = analysis_statistics.prediction_metrics(comp_predictions, spec)
    comp = seeds[seeds.horizon.eq(16_000_000_000)].groupby("model")[["contrast_rmse", "absolute_rmse", "interaction_rmse"]].mean()
    comp_saved = evidence.table("replay/composition_primary.csv").set_index("model")
    np.testing.assert_allclose(comp.sort_index(), comp_saved.loc[comp.index].sort_index(), atol=1e-12, rtol=0)
    comp.to_csv(output / "composition_scores.csv")
    plots.render(main, output, main_law=main_law)
    claims = {
        "selected_law": SELECTED, "selected_shared_parameters": 13, "language_local_parameters": 3,
        "selected_scores": {c: float(table.loc[SELECTED, c]) for c in columns},
        "exploratory_law": EXPLORATORY,
        "exploratory_scores": {c: float(table.loc[EXPLORATORY, c]) for c in columns},
        "relative_mean_improvement_over_M2": float(1-table.loc[SELECTED, "mean_primary_rmse"]/table.loc["M2", "mean_primary_rmse"]),
        "coverage_ablation_K200_relative_improvement_J0ND_to_M5ND": float(1-table.loc["M5-ND", "K200_common_languages"]/table.loc["J0-ND", "K200_common_languages"]),
        "composition_contrast_rmse": {name: float(comp.loc[name, "contrast_rmse"]) for name in (SELECTED, "M5-ND", "M6-ND")},
        "coefficient_replay_max_abs_difference": max(r["max_abs_difference"] for r in replay_checks),
        "caveats": ["selection is inner-only; later form development saw outer/L26 outcomes",
                    "L26 is one large run, not 200 independent experiments",
                    "no claim of winning every internal baseline/split; missing endpoint scores stay missing",
                    "the selected law does not inherit M6-ND's composition advantage"],
    }
    if current:
        from . import ablations
        ablation_claims = ablations.run(evidence, output, checks)
        claims.update(main_law=main_law, main_shared_parameters=14,
                      main_scores={c: float(table.loc[main_law, c]) for c in columns},
                      historical_inner_selected=SELECTED,
                      main_composition_scores={c: float(comp.loc[main_law, c]) for c in comp.columns},
                      ablations=ablation_claims,
                      reproduction_scope="RMSE from row-level predictions; RN13/RN14 coefficient replay; current-table parity; no optimizer rerun")
        # Do not describe the old inner-selected control as the main-paper selection.
        claims["selected_law"] = SELECTED
        claims["selection_role"] = "historical inner selection, distinct from final proposed law"
        claims["historical_inner_selection"] = {"law": SELECTED, "shared_parameters": 13,
                                               "scores": claims.pop("selected_scores")}
        for key in ("selected_law", "selected_shared_parameters", "exploratory_law", "exploratory_scores"):
            claims.pop(key)
        claims["relative_mean_improvement_over_M2"] = float(1-table.loc[main_law, "mean_primary_rmse"]/table.loc["M2", "mean_primary_rmse"])
        claims["caveats"][0] = "RN13 was historically inner-selected; RN14 is the final proposed law after developmental comparison"
        claims["caveats"][-1] = "RN14 does not inherit M6-ND's composition advantage; later form development saw outer/L26 outcomes"
        claims["checks_passed"] = checks.save(output)
    return io.finish(output, evidence, "scaling_law", claims,
                     [Path(__file__), Path(plots.__file__), Path(io.__file__), Path(replay.__file__),
                      Path(law_search_replay.__file__), Path(analysis_statistics.__file__),
                      *([Path(ablations.__file__)] if current else [])])


def manuscript_scores(text):
    """Read source-labelled rows; never infer model identity from presentation labels."""
    records, model = [], None
    for line in text.splitlines():
        if line.startswith("% Source model:"):
            model = line.split(":", 1)[1].strip()
        elif model and "&" in line:
            cells = line.split("&")[-5:]
            values = [float(re.search(r"[+-]?\d+\.\d+", cell).group()) for cell in cells]
            records.append(dict(model=model, **dict(zip([*FOLDS, "mean_primary_rmse", "L26_all_grid"], values))))
            model = None
    if not records or len({r["model"] for r in records}) != len(records):
        raise ValueError("missing or duplicate model identities in manuscript table")
    return pd.DataFrame(records).set_index("model")
