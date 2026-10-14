"""Rescore certified expert-coordinate and exponent ablations, without refitting."""
import numpy as np
import pandas as pd


def run(evidence, output, checks):
    # Import here to avoid an import-time cycle with the public scorer.
    from .analysis import FOLDS, grid_scores, endpoint_scores
    summaries = {}
    for campaign in ("coverage", "expert"):
        predictions = evidence.table(f"{campaign}/predictions.parquet")
        status = evidence.table(f"{campaign}/metrics.csv")
        reference = evidence.table(f"{campaign}/comparison.csv").set_index("model")
        models = sorted(predictions.model.unique())
        status = status[status.model.isin(models)]
        reference = reference.loc[models]
        keys = ["model", "case", "fold"]
        valid = status[status.certified.eq(True) & status.complete.eq(True)]
        checks.require(f"{campaign} unique certification cells", not valid.duplicated(keys).any())
        scored = predictions.merge(valid[keys], on=keys, validate="many_to_one")
        sizes = scored.groupby(keys).size().rename("actual_rows").reset_index()
        sizes = sizes.merge(valid[keys+["rows"]], on=keys, validate="one_to_one")
        checks.close(f"{campaign} certified row counts", sizes.actual_rows, sizes.rows)
        table = grid_scores(scored[scored.case.eq("grid") & scored.fold.isin(FOLDS)]).reindex(columns=FOLDS)
        table["mean_primary_rmse"] = table[list(FOLDS)].mean(axis=1, skipna=False)
        table["L26_all_grid"] = endpoint_scores(scored[scored.case.eq("L26") & scored.fold.eq("all_grid")])
        table = table.reindex(reference.index)
        for column in table:
            mask = table[column].notna() & reference[column].notna()
            checks.close(f"{campaign} {column}", table.loc[mask, column], reference.loc[mask, column],
                         atol=2e-12, source=f"{campaign}/comparison.csv")
        table.to_csv(output / f"{campaign}_ablation_scores.csv")
        missing = status[~(status.certified.eq(True) & status.complete.eq(True))]
        missing.to_csv(output / f"{campaign}_uncertified_cells.csv", index=False)
        # Preserve diagnostics, including the inner validation tradeoff. These
        # are optimizer receipts, not statistics recomputed from outer errors.
        evidence.table(f"{campaign}/paired_ablation.csv").to_csv(output / f"{campaign}_paired_diagnostics.csv", index=False)
        summaries[campaign] = {"rescored_models": int(table.notna().any(axis=1).sum()), "models": models,
                               "uncertified_cells": len(missing),
                               "missing_policy": "leave unavailable splits and their mean missing; no significance claim"}
    return summaries
