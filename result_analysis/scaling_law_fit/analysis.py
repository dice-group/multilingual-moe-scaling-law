"""Score new prediction panels without fixed laws, folds or language counts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Historical defaults retained for callers of the lower-level scoring API.
FOLDS = ("largest_N_act", "E32", "K200_common_languages")
KEYS = ["trajectory_id", "readout_id", "language"]
MAIN_MODELS = ("Chinchilla-target-local", "JointMoE-JH-local",
               "ATLAS-full-active-positive-T100-L10", "ATLAS-full-total-positive-T100-L10",
               "J0", "M2", "M5-ND", "RN14-stored-reuse")


def load_fit_inputs(data):
    """Read measured covariates and the fitting taxonomy, including its missing-family policy."""
    data = Path(data)
    grid = pd.read_parquet(data / "main_grid.parquet").query("metric_role == 'trained'").reset_index(drop=True)
    endpoint = pd.read_parquet(data / "large_model.parquet")
    # The corpus catalogue in languages.parquet has older, incomplete family labels.
    # This endpoint preserves the exact 200-language taxonomy used by the fits.
    taxonomy = endpoint.set_index("language")[["family", "script"]]
    if taxonomy.index.duplicated().any() or not set(grid.language) <= set(taxonomy.index):
        raise ValueError("Incomplete or duplicate fitting taxonomy")
    metadata = taxonomy.astype(object).where(taxonomy.notna(), None).to_dict("index")
    inventory = pd.concat([grid, endpoint])[["language", "K", "U_t"]].drop_duplicates()
    if inventory.duplicated(["language", "K"]).any() or inventory.U_t.isna().any():
        raise ValueError("Inconsistent fitting inventory")
    inventories = inventory.set_index(["language", "K"]).U_t.to_dict()
    return grid, endpoint, inventories, metadata


def load_predictions(data, *, arm=None, revision=None, campaign=None):
    """Select one recorded revision and optimizer arm, using the package's adoption record."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    predictions = pd.read_parquet(data / "fit_predictions.parquet")
    selected = manifest.get("fit_revision")
    if selected is None:
        if arm is not None or revision is not None:
            raise ValueError("Historical package has no recorded optimizer arms")
        return predictions[predictions.fit_campaign.eq(campaign or "final_law")].copy()
    revision = revision or selected["revision"]
    historical = revision == "historical"
    arm = arm or ("historical" if historical else selected["default_arm"])
    campaign = campaign or ("final_law" if historical else selected["fit_campaign"])
    result = predictions[predictions.fit_revision.eq(revision) & predictions.arm.eq(arm)
                         & predictions.fit_campaign.eq(campaign)].copy()
    if result.empty or result.duplicated(["model", "fold", "case", *KEYS]).any():
        raise ValueError("Empty or ambiguous fit revision/arm; select one campaign explicitly")
    return result


def parameter_record(parameters, model, fold, *, revision, arm="lbfgsb_box"):
    """Reconstruct coefficient records from the ordinary parameter Parquet."""
    rows = parameters[parameters.model.eq(model) & parameters.fold.eq(fold)
                      & parameters.fit_revision.eq(revision) & parameters.arm.eq(arm)]
    if rows.empty or rows.duplicated(["scope", "language", "parameter"]).any():
        raise ValueError("Missing or ambiguous coefficient set")

    def coordinates(group):
        group = group.sort_values("coordinate_index")
        if group.coordinate_index.tolist() != list(range(len(group))):
            raise ValueError("Incomplete coefficient coordinates")
        return group.value.tolist()

    if rows.scope.eq("native_language").all():
        groups = []
        for language, group in rows.groupby("language", sort=True):
            first = group.iloc[0]
            groups.append(dict(language=language, selected=dict(coordinates=coordinates(group)),
                spec=json.loads(first.model_spec_json), donors=list(first.donors),
                training_minimum=float(first.training_minimum), unigram_nll=float(first.unigram_nll)))
        return dict(model=model, groups=groups)
    shared = rows[rows.scope.eq("shared")]
    if shared.empty:
        raise ValueError("Missing shared coefficients")
    local = rows[rows.scope.eq("language")].pivot(index="language", columns="parameter", values="value")
    if local.empty or local[["A", "B", "L_inf"]].isna().any().any():
        raise ValueError("Missing language amplitudes or floors")
    coefficients = [{k: v for k, v in row.items() if k == "language" or pd.notna(v)}
                    for row in local.reset_index().to_dict("records")]
    return dict(model=model, selected=dict(coordinates=coordinates(shared), language_coefficients=coefficients))


def from_parquet(data):
    """Rescore the adopted main-paper laws and full comparison inventory."""
    selected = load_predictions(data)
    endpoint = selected[selected.case.eq("L26") & selected.fold.eq("all_grid")]
    options = dict(endpoint_predictions=endpoint, endpoint_languages=sorted(endpoint.language.unique())) if len(endpoint) else {}
    scores = analyze(selected[selected.case.eq("grid")], folds=FOLDS, **options)
    main = [m for m in MAIN_MODELS if m in scores.index]
    if not main:
        main = list(scores.index)
    return {"scores": scores.loc[main], "all_models": scores}


def grid_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Average squared errors within trajectory, then across trajectories."""
    if frame.duplicated(["model", "fold", *KEYS]).any():
        raise ValueError("duplicate held-out prediction membership")
    if not np.isfinite(frame[["adjusted_loss", "prediction"]]).all().all():
        raise ValueError("nonfinite prediction")
    work = frame.assign(squared_error=(frame.prediction - frame.adjusted_loss)**2)
    return (work.groupby(["model", "fold", "trajectory_id"]).squared_error.mean()
            .groupby(["model", "fold"]).mean().pow(.5).unstack("fold"))


def endpoint_scores(frame: pd.DataFrame, expected_languages=200) -> pd.Series:
    """Equal-language endpoint RMSE; incomplete cohorts remain unavailable."""
    if frame.duplicated(["model", "language"]).any():
        raise ValueError("duplicate L26 prediction membership")
    work = frame.assign(squared_error=(frame.prediction - frame.adjusted_loss)**2)
    complete = work.groupby("model").language.nunique().eq(expected_languages)
    if not np.isfinite(work[["adjusted_loss", "prediction"]]).all().all():
        raise ValueError("nonfinite L26 prediction")
    return work[work.model.isin(complete[complete].index)].groupby("model").squared_error.mean().pow(.5)


def analyze(predictions, *, folds, endpoint_predictions=None, endpoint_languages=None):
    """Configuration-balanced RMSE on exactly paired model populations.

    Missing folds retain NaN, including the mean. Available models in each fold
    must have identical row identities and targets. Endpoints require an
    explicit language cohort and represent one run, not language replicates.
    No model selection, optimizer fit or comparison to published values occurs.
    """
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("unique nonempty folds required")
    data = predictions[predictions.fold.isin(folds)].copy()
    if data.empty or data[["model", "fold", *KEYS]].isna().any().any():
        raise ValueError("empty prediction panel or missing identity")
    if data.duplicated(["model", "fold", *KEYS]).any():
        raise ValueError("duplicate held-out prediction membership")
    for fold, panel in data.groupby("fold"):
        reference = None
        for model, part in panel.groupby("model"):
            part = part.set_index(KEYS).sort_index().adjusted_loss
            if reference is not None and (not part.index.equals(reference.index)
                    or not np.allclose(part, reference, rtol=0, atol=1e-12)):
                raise ValueError(f"different scoring population or targets: {fold}/{model}")
            reference = part
    scores = grid_scores(data).reindex(columns=folds)
    scores["mean_rmse"] = scores[list(folds)].mean(axis=1, skipna=False)
    if endpoint_predictions is not None:
        if endpoint_predictions.empty or endpoint_predictions[["model", "language"]].isna().any().any():
            raise ValueError("empty endpoint panel or missing identity")
        if endpoint_predictions.duplicated(["model", "language"]).any():
            raise ValueError("duplicate endpoint prediction membership")
        if endpoint_languages is None:
            raise ValueError("explicit endpoint language cohort required")
        languages = tuple(endpoint_languages)
        if not languages or len(set(languages)) != len(languages):
            raise ValueError("unique nonempty endpoint cohort required")
        targets = None
        for model, part in endpoint_predictions.groupby("model"):
            if set(part.language) != set(languages):
                raise ValueError(f"incomplete or foreign endpoint cohort: {model}")
            values = part.set_index("language").sort_index().adjusted_loss
            if targets is not None and (not values.index.equals(targets.index)
                    or not np.allclose(values, targets, rtol=0, atol=1e-12)):
                raise ValueError("different endpoint targets")
            targets = values
        scores = scores.reindex(scores.index.union(endpoint_predictions.model.unique(), sort=False))
        scores["endpoint_rmse"] = endpoint_scores(endpoint_predictions, expected_languages=len(languages))
    return scores


def manuscript_scores(text):
    """Compatibility helper; manuscript parsing belongs to reproduction."""
    from .reproduce import manuscript_scores as read_published
    return read_published(text)


def run(evidence, output):
    """Compatibility entrypoint for the separate published-result reproduction."""
    from .reproduce import run as reproduce
    return reproduce(evidence, output)
