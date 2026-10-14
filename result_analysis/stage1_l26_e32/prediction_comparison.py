"""Replay frozen paper fits on the independent cooled L26/E32 endpoint.

No fitting or selection: retain every certified model, separate training folds,
and report invalid extrapolations instead of clipping or dropping languages.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import identity
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import read
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import verify_files
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import write
from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.model import Spec
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.model import predict as native_predict
from multilingual_megatron.experiments.scaling.stage1_fit.m6_composition import composition_exposure_components


def shared_predict(selected, rows):
    # All normalized shared forms are nested within M6; ND adds its fixed
    # size multiplier only to the data term. Replay checks guard this mapping.
    cap, data = fit.predictor_terms("M6", fit.SharedParameters(**selected["shared_parameters"]),
        rows.N_act, rows.D_t, rows.E, rows.K,
        **{k: rows[k] for k in ("D_family", "D_script", "D_other")})
    by_language = {v["language"]: v for v in selected["language_coefficients"]}
    c = np.array([[by_language[l][k] for k in ("L_inf", "A", "B")] for l in rows.language])
    return c[:, 0] + c[:, 1]*cap + c[:, 2]*data*(rows.N_act.to_numpy()/1e8)**selected["zeta"]


def endpoint_packet(packet, rows, families, total_parameters):
    """Preserve frozen donor selection and unique inventory, update exposures."""
    language = packet["language"]
    indexed = rows.set_index("language")
    row = indexed.loc[language]
    donors = packet["donors"]
    if len(set([language, *donors])) != 4:
        raise ValueError("target and three donors must be distinct")
    d = [int(indexed.loc[l, "D_t"]) for l in [language, *donors]]
    d += [int(row.D_total)-sum(d)]
    # Support includes input-only K200 points even in the K200 holdout.
    support = packet["support"]
    inventories = np.asarray(support["U"])[np.asarray(support["K"]) == 200]
    if not len(inventories) or not np.all(inventories == inventories[0]):
        raise ValueError("no unambiguous frozen K200 unique inventory")
    family = families[language]
    family_tokens = sum(int(indexed.loc[l, "D_t"]) for l in indexed.index if families[l] == family)
    return {"N": [row.N_act], "N_total": [total_parameters], "E": [row.E], "K": [row.K],
        "D": [d], "U": [inventories[0]], "family_share": [family_tokens/row.D_total],
        "training_minimum": packet["train"]["training_minimum"]}


def summarize(predictions, expected):
    output = []
    for (fold, model), group in predictions.groupby(["fold", "model"]):
        valid = np.isfinite(group.prediction)
        complete = bool(valid.all() and set(group.language) == set(expected[fold]))
        error = group.prediction-group.observed
        output.append({"fold": fold, "model": model, "languages": len(group),
            "valid_languages": int(valid.sum()), "complete": complete,
            **({"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(abs(error))),
                "bias": float(error.mean()), "max_abs_error": float(abs(error).max()),
                "observed_mean": float(group.observed.mean()),
                "predicted_mean": float(group.prediction.mean())} if complete else {})})
    return pd.DataFrame(output).sort_values(["fold", "rmse", "model"])


def compare(evaluation, shared_bundle, literature, architecture, output, all_data_report=None):
    if output.exists():
        raise ValueError("comparison output is create-only")
    bindings = []
    def bound(path, expected=None):
        entry = identity(path)
        if expected is not None and entry["sha256"] != expected:
            raise ValueError(f"changed source: {path}")
        bindings.append(entry)
        return path
    release = read(bound(evaluation/"release.json"))
    if release["status"] != "complete" or release["rows"] != 200:
        raise ValueError("requires complete 200-language release")
    verify_files([release[k] for k in ("evaluation", "observations", "plan")])
    rows = pd.read_parquet(bound(Path(release["observations"]["path"]), release["observations"]["sha256"]))
    if len(rows) != 200 or rows.language.nunique() != 200 or set(rows.checkpoint_step) != {269634}:
        raise ValueError("wrong endpoint or language coverage")
    rows = rows.rename(columns={"atlas_tao_unigram_adjusted_loss": "adjusted_loss"})
    rows["trajectory_id"], rows["readout_id"] = "L26:E32:K200", "final_269634"
    metadata = {r.language: {"family": None if pd.isna(r.family) else r.family, "script": r.script}
                for r in rows.itertuples()}
    components, _ = composition_exposure_components(rows, metadata)
    rows = rows.join(components.drop(columns="D_target"))
    arch = read(bound(architecture))
    # Bind actual stored parameter geometry rather than infer N_total from N_act.
    counts = arch["frozen_model"]
    total_parameters = counts["total_parameters"]
    if counts["active_parameters"] != int(rows.N_act.iloc[0]):
        raise ValueError("architecture parameter count differs from evaluation")
    manifest = read(bound(shared_bundle/"manifest.json"))
    def shared_file(name):
        name = "evidence/fits/"+name
        return bound(shared_bundle/name, manifest["files"][name])
    selected = read(shared_file("selected.json"))
    historical = pd.read_parquet(shared_file("predictions.parquet"))
    reference_rows = pd.read_parquet(shared_file("fit_rows.parquet"))
    # Same loss offset and target count are necessary for direct law comparison.
    for r in rows.itertuples():
        old = reference_rows[reference_rows.language == r.language]
        np.testing.assert_allclose(old.unigram_nll, -r.unigram_log_probability_sum/r.token_count, atol=1e-10, rtol=0)
        if not (old.validation_token_count == r.token_count).all():
            raise ValueError(f"validation stream differs: {r.language}")
    expected, records, replay_errors = {}, [], []
    for fold, models in selected.items():
        populations = [{c["language"] for c in s["language_coefficients"]} for s in models.values()]
        if any(p != populations[0] for p in populations):
            raise ValueError("shared models have different language coverage")
        expected[fold] = sorted(populations[0])
        target = rows[rows.language.isin(expected[fold])]
        for name, coefficients in models.items():
            old = historical[(historical.fold == fold) & (historical.model == name)]
            replay = shared_predict(coefficients, old)
            np.testing.assert_allclose(replay, old.prediction, atol=1e-10, rtol=0)
            replay_errors.append(float(np.max(abs(replay-old.prediction))))
            for r, value in zip(target.itertuples(), shared_predict(coefficients, target)):
                records.append(dict(fold=fold, model=name, language=r.language,
                    observed=r.adjusted_loss, prediction=float(value), status="valid"))
    lm = read(bound(literature/"manifest.json"))
    def lit_file(name):
        return bound(literature/name, lm["outputs"][name])
    packets = json.loads(gzip.decompress(lit_file("packets.json.gz").read_bytes()))
    fits = json.loads(gzip.decompress(lit_file("selected.json.gz").read_bytes()))
    statuses = pd.read_csv(lit_file("model_status.csv"))
    certified = set(statuses.loc[statuses.status == "certified", "model"])
    native_reference = pd.read_parquet(lit_file("predictions.parquet"))
    references = {key: group for key, group in native_reference.groupby(["model", "fold", "language"])}
    packet_index = {(p["fold"], p["language"]): p for p in packets}
    families = {p["language"]: p["family"] for p in packets if p["fold"] == "largest_N_act"}
    for s in fits:
        if s["model"] not in certified or not s["certified"]:
            raise ValueError("uncertified native fit")
        packet = packet_index[s["fold"], s["language"]]
        spec = Spec(**s["spec"])
        old = references[s["model"], s["fold"], s["language"]].set_index(["trajectory_id", "readout_id"])
        keys = list(zip(packet["test"]["trajectory_ids"], packet["test"]["readout_ids"]))
        replay = native_predict(spec, s["coordinates"], packet["test"])[0]-packet["unigram_nll"]
        np.testing.assert_allclose(replay, old.loc[keys, "prediction"], atol=1e-10, rtol=0)
        replay_errors.append(float(np.max(abs(replay-old.loc[keys, "prediction"].to_numpy()))))
        target = endpoint_packet(packet, rows, families, total_parameters)
        try:
            value = float(native_predict(spec, s["coordinates"], target)[0][0]-packet["unigram_nll"])
            status = "valid"
        except (ValueError, FloatingPointError) as error:
            value, status = np.nan, str(error)
        r = rows.set_index("language").loc[s["language"]]
        records.append(dict(fold=s["fold"], model=s["model"], language=s["language"],
            observed=r.adjusted_loss, prediction=value, status=status))
    predictions = pd.DataFrame(records)
    all_data_check = None
    if all_data_report is not None:
        forecast = read(bound(all_data_report))
        result_binding, audit_binding = forecast["all_data_result"], forecast["all_data_audit"]
        result = read(bound(Path(result_binding["path"]), result_binding["file_sha256"]))
        audit = read(bound(Path(audit_binding["path"]), audit_binding["file_sha256"]))
        if audit["status"] != "complete" or audit["result_sha256"] != result["result_sha256"]:
            raise ValueError("all-data fit lacks matching complete audit")
        if result["membership"]["test_trajectory_ids"]:
            raise ValueError("requested all-data fit has a held-out training trajectory")
        coefficients = dict(result["selected"], zeta=0.)
        original = rows.copy()
        original["D_total"] = forecast["l26_e32"]["D_total"]
        original["D_t"] = original.language.map(forecast["l26_e32"]["D_t_by_language"])
        original_prediction = shared_predict(coefficients, original)
        np.testing.assert_allclose(original_prediction.mean(), forecast["l26_e32"]["macro_adjusted_loss"], atol=1e-10, rtol=0)
        values = shared_predict(coefficients, rows)
        all_data_check = {"original_macro_prediction": float(original_prediction.mean()),
                          "exact_exposure_macro_prediction": float(values.mean())}
        expected["all_data"] = sorted(rows.language)
        predictions = pd.concat([predictions, pd.DataFrame({"fold": "all_data", "model": "M2",
            "language": rows.language, "observed": rows.adjusted_loss, "prediction": values,
            "status": "valid"})], ignore_index=True)
    if predictions.duplicated(["fold", "model", "language"]).any():
        raise ValueError("duplicate prediction identity")
    metrics = summarize(predictions, expected)
    output.mkdir(parents=True)
    predictions.to_csv(output/"predictions.csv", index=False)
    metrics.to_csv(output/"metrics.csv", index=False)
    rows.to_parquet(output/"endpoint_inputs.parquet", index=False)
    write(output/"manifest.json", {"inputs": bindings, "implementation": identity(__file__),
        "outputs": [identity(output/name) for name in ("predictions.csv", "metrics.csv", "endpoint_inputs.parquet")],
        "maximum_historical_replay_error": max(replay_errors), "refits": 0,
        "all_data_forecast_replay": all_data_check,
        "independent_training_runs": 1, "loss": "unigram_adjusted_nats; equal language weighting",
        "scope": "frozen fold fits and separately labelled existing all-data M2; no pooling across folds or winner selection for deployment",
        "excluded_uncertified_models": statuses.loc[statuses.status != "certified", "model"].tolist()})
    print(metrics.groupby("fold").head(8).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "shared-bundle", "literature", "architecture", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--all-data-report", type=Path)
    args = parser.parse_args()
    compare(args.evaluation, args.shared_bundle, args.literature, args.architecture, args.output, args.all_data_report)


if __name__ == "__main__":
    main()
