"""Existing-evidence-only diagnosis of held-out L26/E32 predictions.

Never fits coefficients, modifies old outputs, or submits jobs. Centered errors,
term contributions and derivatives are diagnostics, not improved forecasts.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import identity
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import read
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import verify_files
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import write
from result_analysis.stage1_l26_e32.prediction_comparison import endpoint_packet
from result_analysis.stage1_l26_e32.prediction_comparison import shared_predict
from result_analysis.stage1_l26_e32.evidence import audit
from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.model import Spec
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.model import predict as native_predict
from multilingual_megatron.experiments.scaling.stage1_l26_e32 import evaluation
from multilingual_megatron.experiments.scaling.stage1_fit import atlas_full_model
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import model as atlas_model
from result_analysis.scaling_law_fit import residuals
from result_analysis.scaling_law_fit.residuals import learning_slopes

NATIVE = ("ATLAS-full-active-positive-T100-L10", "ATLAS-full-total-positive-T100-L10",
          "ATLAS-other-active-positive-T100-L100", "JointMoE-JH-local", "Chinchilla-total-local")
KEYS = ["trajectory_id", "readout_id", "language"]


def error_decomposition(observed, predicted):
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != predicted.shape:
        raise ValueError("prediction and observation shapes differ")
    error = predicted-observed
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError("requires complete finite predictions")
    bias, mse = float(error.mean()), float(np.mean(error**2))
    variance = float(np.mean((error-bias)**2))
    np.testing.assert_allclose(mse, bias*bias+variance, atol=1e-12, rtol=1e-12)
    return dict(rmse=np.sqrt(mse), bias=bias, squared_bias=bias*bias,
        residual_variance=variance, centered_rmse_diagnostic=np.sqrt(variance),
        optimistic_languages=int((error < 0).sum()), languages=len(error))


def shared_terms(selected, rows):
    cap, data = fit.predictor_terms("M6", fit.SharedParameters(**selected["shared_parameters"]),
        rows.N_act, rows.D_t, rows.E, rows.K,
        **{k: rows[k] for k in ("D_family", "D_script", "D_other")})
    c = pd.DataFrame(selected["language_coefficients"]).set_index("language").loc[rows.language]
    terms = np.column_stack((c.L_inf, c.A.to_numpy()*cap,
        c.B.to_numpy()*data*(rows.N_act.to_numpy()/1e8)**selected["zeta"]))
    np.testing.assert_allclose(terms.sum(axis=1), shared_predict(selected, rows), atol=1e-12, rtol=0)
    return terms


def native_terms(spec, coordinates, packet, unigram_nll):
    values, jac = native_predict(spec, coordinates, packet)
    # The first three coordinates multiply the floor, capacity and data bases.
    terms = jac[:, :3]*np.asarray(coordinates[:3])
    terms[:, 0] -= unigram_nll
    np.testing.assert_allclose(terms.sum(axis=1), values-unigram_nll, atol=1e-10, rtol=0)
    return terms


def matched_experts(predictions):
    keys = ["fold", "model", "ladder", "K", "language", "N_act", "D_total"]
    left, right = (predictions[predictions.E == e].copy() for e in (1, 32))
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("duplicate expert contrast keys")
    pairs = left.merge(right, on=keys, suffixes=("_e1", "_e32"), validate="one_to_one")
    if len(pairs) != len(left) or len(pairs) != len(right):
        raise ValueError("incomplete expert contrast coverage")
    for name in ("D_t", "D_family", "D_script", "D_other"):
        if not np.array_equal(pairs[name+"_e1"], pairs[name+"_e32"]):
            raise ValueError("expert contrast has unmatched realized exposure")
    pairs["observed_gain"] = pairs.adjusted_loss_e1-pairs.adjusted_loss_e32
    pairs["predicted_gain"] = pairs.prediction_e1-pairs.prediction_e32
    pairs["gain_error"] = pairs.predicted_gain-pairs.observed_gain
    # Equal trajectory-pair weighting, not independent language replication.
    group = ["fold", "model", "ladder", "K", "D_total"]
    summary = pairs.groupby(group)[["observed_gain", "predicted_gain", "gain_error"]].mean().reset_index()
    summary["languages"] = pairs.groupby(group).size().to_numpy()
    return pairs, summary


def run(comparison, grid_release, output):
    if output.exists():
        raise ValueError("diagnosis output is create-only")
    evidence = audit(comparison, grid_release)
    manifest = read(comparison/"manifest.json")
    def source(suffix):
        paths = [Path(x["path"]) for x in manifest["inputs"] if x["path"].endswith(suffix)]
        if len(paths) != 1:
            raise ValueError(f"ambiguous source: {suffix}")
        return paths[0]
    endpoint = pd.read_parquet(comparison/"endpoint_inputs.parquet")
    predictions = pd.read_csv(comparison/"predictions.csv")
    old = pd.read_parquet(source("evidence/fits/fit_rows.parquet"))
    historical = pd.read_parquet(source("evidence/fits/predictions.parquet"))
    selected = read(source("evidence/fits/selected.json"))
    packets = json.loads(gzip.decompress(source("packets.json.gz").read_bytes()))
    native_fits = json.loads(gzip.decompress(source("selected.json.gz").read_bytes()))
    packet_index = {(p["fold"], p["language"]): p for p in packets}
    families = {p["language"]: p["family"] for p in packets if p["fold"] == "largest_N_act"}
    architecture = read(source("architecture_stage1_l26_e32_v1.json"))["frozen_model"]
    native_fits = [s for s in native_fits if s["model"] in NATIVE]
    decomposed = []
    for (fold, model), rows in predictions.groupby(["fold", "model"]):
        if np.isfinite(rows.prediction).all():
            decomposed.append(dict(fold=fold, model=model, **error_decomposition(rows.observed, rows.prediction)))
    decomposition = pd.DataFrame(decomposed)
    term_rows, sensitivities, native_history = [], [], []
    # The endpoint's unique target inventory comes from the frozen K200 packet.
    unique = {}
    for language in endpoint.language:
        p = packet_index["largest_N_act", language]
        inputs = endpoint_packet(p, endpoint, families, architecture["total_parameters"])
        unique[language] = float(inputs["U"][0][0])
    endpoint["unique_training_tokens"] = endpoint.language.map(unique)
    endpoint["exposure_over_unique_inventory"] = endpoint.D_t/endpoint.unique_training_tokens
    endpoint["exposure_quartile"] = pd.qcut(endpoint.D_t, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    endpoint["repetition_bin"] = pd.cut(endpoint.exposure_over_unique_inventory,
        [0, 1, 4, float("inf")], labels=["<=1", "1-4", ">4"], include_lowest=True)
    for fold, models in selected.items():
        for name, coefficients in models.items():
            langs = {c["language"] for c in coefficients["language_coefficients"]}
            rows = endpoint[endpoint.language.isin(langs)]
            values = shared_terms(coefficients, rows)
            slopes = {}
            for axis in ("size", "duration"):
                plus, minus = rows.copy(), rows.copy()
                columns = ["N_act"] if axis == "size" else ["D_t", "D_family", "D_script", "D_other", "D_total"]
                for column in columns:
                    plus[column] *= np.exp(1e-5)
                    minus[column] *= np.exp(-1e-5)
                slopes[axis] = (shared_predict(coefficients, plus)-shared_predict(coefficients, minus))/2e-5
            for language, parts, sn, sd in zip(rows.language, values, slopes["size"], slopes["duration"]):
                term_rows.append(dict(fold=fold, model=name, language=language,
                    floor=parts[0], capacity=parts[1], data=parts[2], log_size_slope=sn, log_duration_slope=sd))
    for s in native_fits:
        p = packet_index[s["fold"], s["language"]]
        spec = Spec(**s["spec"])
        target = endpoint_packet(p, endpoint, families, architecture["total_parameters"])
        parts = native_terms(spec, s["coordinates"], target, p["unigram_nll"])[0]
        slopes = {}
        for axis in ("size", "duration"):
            key = ("N_total" if spec.parameter_count == "total" else "N") if axis == "size" else "D"
            plus = dict(target, **{key: np.asarray(target[key])*np.exp(1e-5)})
            minus = dict(target, **{key: np.asarray(target[key])*np.exp(-1e-5)})
            slopes[axis] = float((native_predict(spec, s["coordinates"], plus)[0][0]-native_predict(spec, s["coordinates"], minus)[0][0])/2e-5)
        term_rows.append(dict(fold=s["fold"], model=s["model"], language=s["language"],
            floor=parts[0], capacity=parts[1], data=parts[2], log_size_slope=slopes["size"], log_duration_slope=slopes["duration"]))
        e1, e32 = (native_predict(spec, s["coordinates"], dict(target, E=[e]))[0][0] for e in (1, 32))
        sensitivities.append(dict(fold=s["fold"], model=s["model"], language=s["language"],
            expert_only_change=float(e32-e1), mapping=spec.parameter_count))
        support = p["support"]
        values = native_predict(spec, s["coordinates"], support)[0]-p["unigram_nll"]
        frame = pd.DataFrame({"trajectory_id": support["trajectory_ids"], "readout_id": support["readout_ids"],
            "language": s["language"], "prediction": values, "fold": s["fold"], "model": s["model"]})
        frame = frame.merge(old, on=KEYS, validate="one_to_one")
        frame["split"] = np.where(frame.readout_id.isin(p["train"]["readout_ids"]), "train", "test")
        native_history.append(frame)
    terms = pd.DataFrame(term_rows)
    full_history = pd.concat([historical, *native_history], ignore_index=True)
    pair_rows, pair_summary = matched_experts(full_history)
    comparisons = []
    neighbors = old[(old.ladder == "D") & (old.E == 32) & (old.K == 200) & (old.D_total >= 8e9)]
    for horizon, rows in neighbors.groupby("D_total"):
        merged = rows.merge(endpoint, on="language", suffixes=("_grid", "_l26"), validate="one_to_one")
        delta = merged.adjusted_loss_l26-merged.adjusted_loss_grid
        comparisons.append(dict(grid_tokens=horizon, languages=len(merged),
            grid_mean=merged.adjusted_loss_grid.mean(), l26_mean=merged.adjusted_loss_l26.mean(),
            observed_l26_minus_grid=delta.mean(), l26_worse_languages=int((delta > 0).sum())))
    growth = []
    for (fold, model), group in full_history.groupby(["fold", "model"]):
        neighbor = group[(group.ladder == "D") & (group.E == 32) & (group.K == 200)]
        for horizon, nrows in neighbor[neighbor.D_total >= 8e9].groupby("D_total"):
            target = predictions[(predictions.fold == fold) & (predictions.model == model)]
            merged = nrows.merge(target, on="language", suffixes=("_grid", "_l26"), validate="one_to_one")
            growth.append(dict(fold=fold, model=model, grid_tokens=horizon, languages=len(merged),
                predicted_l26_gain=float((merged.prediction_grid-merged.prediction_l26).mean()),
                observed_l26_gain=float((merged.adjusted_loss-merged.observed).mean()),
                grid_bias=float((merged.prediction_grid-merged.adjusted_loss).mean()),
                endpoint_bias=float((merged.prediction_l26-merged.observed).mean()),
                excess_predicted_gain=float(((merged.prediction_grid-merged.prediction_l26)-(merged.adjusted_loss-merged.observed)).mean())))
    joined = predictions.merge(endpoint, on="language", validate="many_to_one")
    joined["error"] = joined.prediction-joined.observed
    group_rows = []
    for (fold, model), part in joined.groupby(["fold", "model"]):
        if not np.isfinite(part.error).all():
            continue
        total = float(np.sum(part.error**2))
        for axis in ("exposure_quartile", "repetition_bin", "family", "script"):
            for label, group in part.groupby(axis, observed=True, dropna=False):
                group_rows.append(dict(fold=fold, model=model, axis=axis, group=str(label),
                    squared_error_share=float(np.sum(group.error**2)/total),
                    **error_decomposition(group.observed, group.prediction)))
    terms = terms.merge(endpoint[["language", "exposure_over_unique_inventory"]], on="language", validate="many_to_one")
    tables = {"error_decomposition": decomposition, "term_contributions": terms,
        "term_means": terms.groupby(["fold", "model"])[["floor", "capacity", "data", "log_size_slope", "log_duration_slope"]].mean().reset_index(),
        "expert_only_sensitivity": pd.DataFrame(sensitivities), "matched_expert_pairs": pair_rows,
        "matched_expert_summary": pair_summary, "grid_neighbors": pd.DataFrame(comparisons),
        "predicted_vs_observed_growth": pd.DataFrame(growth), "residual_groups": pd.DataFrame(group_rows),
        "language_errors": joined, "historical_learning_slopes": learning_slopes(full_history),
        "configuration_differences": pd.DataFrame(evidence["config_differences"]),
        "segment_states": pd.DataFrame(evidence["segment_states"]),
        "training_log_health": pd.DataFrame(evidence["training_log_health"])}
    # Fail closed if any input changed during this read-only analysis.
    verify_files(evidence["bindings"])
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output/(name+".csv"), index=False)
    write(output/"comparability.json", evidence["ledger"])
    # Retain the exact diagnostic implementation even if the checkout changes.
    source_files = [Path(__file__), Path(__file__).with_name("evidence.py")]
    dependencies = [Path(fit.__file__), Path(__file__).with_name("prediction_comparison.py"),
        Path(evaluation.__file__),
        Path(atlas_model.__file__),
        Path(atlas_full_model.__file__),
        Path(residuals.__file__)]
    (output/"source").mkdir()
    for path in source_files:
        shutil.copyfile(path, output/"source"/path.name)
    write(output/"manifest.json", {"inputs": evidence["bindings"],
        "implementation": [identity(p) for p in sorted((output/"source").iterdir())],
        "dependencies": [identity(p) for p in dependencies],
        "outputs": [identity(p) for p in sorted(output.iterdir()) if p.is_file()],
        "refits": 0, "gpu_jobs": 0, "independent_training_runs": 1,
        "centered_error_scope": "descriptive decomposition using observed outcomes, never a deployable or corrected forecast"})
    print(decomposition[(decomposition.fold == "E32") & decomposition.model.isin(["M6-ND", *NATIVE])].to_string(index=False))
    print(tables["grid_neighbors"].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "grid-release", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison.resolve(), args.grid_release.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
