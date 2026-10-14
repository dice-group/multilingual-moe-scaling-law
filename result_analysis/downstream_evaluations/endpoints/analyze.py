"""Compare complete final-checkpoint benchmarks with sealed validation losses.

Descriptive endpoint comparisons only. Language rows and settings are not
independent training replicates. Generated evidence requires a fresh directory.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from result_analysis.capacity_response.tables import (
    load_primary, load_release_table, sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = {
    "belebele_cloze": "Belebele", "global_mmlu_full": "Global MMLU",
    "okapi_hellaswag": "HellaSwag", "uhura_arc_easy": "ARC Easy / Uhura",
    "xcopa": "XCOPA", "xstorycloze": "XStoryCloze",
}
# Explicit additions for full Global MMLU's languages, using corpus identifiers.
MAP_ADDITIONS = {
    "cs": "ces_Latn", "el": "ell_Grek", "fa": "fas_Arab",
    "fil": "fil_Latn", "he": "heb_Hebr", "ig": "ibo_Latn",
    "ky": "kir_Cyrl", "lt": "lit_Latn", "mg": "plt_Latn",
    "ms": "zsm_Latn", "ny": "nya_Latn", "pl": "pol_Latn",
    "si": "sin_Sinh", "sn": "sna_Latn", "so": "som_Latn",
}
METRICS = ("acc_norm", "acc")
SCOPES = ("common12", "common12_non_english", "english", "trained", "all_evaluated")
CELL = ["ladder", "E", "K", "seed", "benchmark", "scope"]


def sealed(path, field="manifest_sha256"):
    obj = json.loads(path.read_text())
    expected = obj.pop(field)
    assert canonical_json_sha256(obj) == expected, f"Payload hash mismatch: {path}"
    return obj


def referenced(reference):
    path = Path(reference["path"])
    assert sha256_file(path) == reference["sha256"], f"File hash mismatch: {path}"
    return path


def rho(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    return float(spearmanr(x, y).statistic) if len(x) > 2 and len(set(x)) > 1 and len(set(y)) > 1 else np.nan


def load_inputs(root, campaign_root, source_root, mapping_path):
    campaign = sealed(campaign_root / "campaign.json")
    metrics = sealed(campaign_root / "metrics.json")
    assert metrics["status"] == "complete" and not metrics["incomplete_jobs"]
    bench = pd.DataFrame(metrics["metrics"])
    assert len(bench) == 18532 and bench.model_id.nunique() == 82
    assert set(bench.benchmark) == set(BENCHMARKS)
    assert not bench.duplicated(["model_id", "benchmark", "language"]).any()
    assert bench.groupby(["benchmark", "language"]).samples.nunique().eq(1).all()
    assert bench.groupby(["model_id", "benchmark"]).size().eq(
        bench.groupby(["model_id", "benchmark"]).size().index.get_level_values("benchmark").map(
            {b: c["languages"] for b, c in campaign["inputs"].items()})).all()

    receipt = sealed(source_root / "preparation_receipt.json", "receipt_sha256")
    models = {m["id"]: m for m in receipt["models"] if m["is_final_readout"]}
    assert set(models) == set(bench.model_id)
    primary = load_primary(root)
    endpoint = primary.sort_values("D_total").groupby(["ladder", "E", "K", "language"]).tail(1).copy()
    endpoint["seed"] = 1234
    readouts = load_release_table(root, "readouts", ["readout_id", "checkpoint_iteration_path"])
    checkpoint_by_readout = readouts.set_index("readout_id").checkpoint_iteration_path.to_dict()
    bindings = []
    seed_losses = []
    for model_id, model in models.items():
        jobs = [j for j in campaign["jobs"] if j["model_id"] == model_id]
        assert len(jobs) == 6
        for job in jobs:
            spec_path = referenced(job["spec"])
            spec = sealed(spec_path, "spec_sha256")
            assert spec["checkpoint"]["iteration_path"] == model["checkpoint"]["path"]
            assert spec["checkpoint"]["tokens_seen"] == model["checkpoint"]["tokens_seen"]
        if model["seed"] == 1234:
            assert checkpoint_by_readout[model["readout_id"]] == model["checkpoint"]["path"]
        else:
            em_path = referenced(model["evaluation_manifest"])
            em = sealed(em_path, "result_sha256")
            report_path = referenced(em["validation_report"])
            report = sealed(report_path, "report_sha256")
            assert report["status"] == "complete"
            for row in report["languages"]:
                if row["role"] == "trained":
                    seed_losses.append({"ladder": model["ladder"], "E": model["experts"],
                        "K": model["languages"], "seed": model["seed"], "language": row["language"],
                        "raw_nll": row["raw_nll"], "adjusted_loss": row["atlas_tao_unigram_adjusted_loss"],
                        "D_t": np.nan, "readout_id": model["readout_id"]})
        bindings.append({"model_id": model_id, "ladder": model["ladder"], "E": model["experts"],
                         "K": model["languages"], "seed": model["seed"],
                         "D_total": model["checkpoint"]["tokens_seen"], "N_act": model["active_parameters"],
                         "N_total": model["configured_total_parameters"], "readout_id": model["readout_id"],
                         "checkpoint_path": model["checkpoint"]["path"]})
    metadata = pd.DataFrame(bindings)
    assert len(metadata) == 82 and metadata.query("seed == 1234").shape[0] == 64
    loss_cols = ["ladder", "E", "K", "seed", "language", "raw_nll", "adjusted_loss", "D_t", "readout_id"]
    losses = pd.concat([endpoint[loss_cols], pd.DataFrame(seed_losses)[loss_cols]], ignore_index=True)
    assert not losses.duplicated(["ladder", "E", "K", "seed", "language"]).any()
    sets = {int(k): set(g.language) for k, g in endpoint.groupby("K")}
    assert sets[1] < sets[12] < sets[50] < sets[200]
    mapping = json.loads(mapping_path.read_text())
    short = mapping["short_code_to_corpus"] | MAP_ADDITIONS
    assert set(MAP_ADDITIONS.values()) <= sets[200]

    def corpus(benchmark, language):
        if benchmark == "belebele_cloze":
            # Preserve the paper's one-to-one Mandarin policy; do not merge
            # traditional and simplified benchmark versions into one loss row.
            return {"zho_Hans": "cmn_Hani", "zho_Hant": None, "est_Latn": "ekk_Latn"}.get(language, language)
        assert language in short, f"Unresolved short language code: {language}"
        return short[language]

    bench["corpus_language"] = [corpus(b, l) for b, l in zip(bench.benchmark, bench.language)]
    bench = bench.merge(metadata, on=["model_id", "ladder", "E", "K", "seed"], validate="many_to_one")
    bench = bench.merge(losses.rename(columns={"language": "corpus_language"}).drop(columns="readout_id"),
                        on=["ladder", "E", "K", "seed", "corpus_language"], how="left", validate="many_to_one")
    bench["trained"] = [l in sets[k] for l, k in zip(bench.corpus_language, bench.K)]
    assert bench.loc[bench.trained, "raw_nll"].notna().all()
    assert bench.loc[~bench.trained, "raw_nll"].isna().all()
    bench["in_common12"] = bench.corpus_language.isin(sets[12])
    bench["is_english"] = bench.corpus_language.eq("eng_Latn")
    bench["in_common12_non_english"] = bench.in_common12 & ~bench.is_english
    return campaign, metrics, primary, bench, metadata, sets


def aggregate(bench):
    records = []
    for (model, benchmark), g in bench.groupby(["model_id", "benchmark"]):
        for scope in SCOPES:
            sub = g if scope == "all_evaluated" else g.loc[g[{"common12": "in_common12", "common12_non_english": "in_common12_non_english", "english": "is_english", "trained": "trained"}[scope]]]
            if sub.empty:
                continue
            r = sub.iloc[0]
            row = {"model_id": model, "benchmark": benchmark, "scope": scope,
                   **{c: r[c] for c in ["ladder", "E", "K", "seed", "N_act", "N_total", "D_total"]},
                   "languages": len(sub), "samples": int(sub.samples.sum()),
                   "acc": sub.acc.mean(), "acc_norm": sub.acc_norm.mean()}
            for col in ["raw_nll", "adjusted_loss", "D_t"]:
                row[col] = sub[col].mean() if sub[col].notna().all() else np.nan
            records.append(row)
    return pd.DataFrame(records)


def paired_contrasts(agg, axis, pairs):
    keys = [c for c in CELL if c != axis]
    values = [axis, "acc", "acc_norm", "raw_nll", "languages", "D_total", "N_act"]
    # N_act is not a pairing key: size contrasts intentionally change it.
    values = list(dict.fromkeys(values))
    rows = []
    for low, high in pairs:
        a, b = agg.loc[agg[axis].eq(low)], agg.loc[agg[axis].eq(high)]
        merged = a[keys + values].merge(b[keys + values], on=keys, suffixes=("_low", "_high"), validate="one_to_one")
        merged["axis"], merged["low"], merged["high"] = axis, str(low), str(high)
        merged["loss_gain"] = merged.raw_nll_low - merged.raw_nll_high
        for metric in METRICS:
            merged[metric + "_gain_pp"] = 100 * (merged[metric + "_high"] - merged[metric + "_low"])
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def coverage_interactions(coverage):
    keys = ["ladder", "seed", "benchmark", "scope", "low", "high"]
    a = coverage.loc[coverage.E.eq(1)]
    b = coverage.loc[coverage.E.eq(32)]
    cols = keys + ["loss_gain", "acc_gain_pp", "acc_norm_gain_pp"]
    out = a[cols].merge(b[cols], on=keys, suffixes=("_E1", "_E32"), validate="one_to_one")
    out["loss_interaction_nats"] = out.loss_gain_E32 - out.loss_gain_E1
    for metric in METRICS:
        out[metric + "_interaction_pp"] = out[metric + "_gain_pp_E32"] - out[metric + "_gain_pp_E1"]
    return out


def summarize(agg, expert, size, interactions):
    correlations, overview = [], []
    for benchmark in BENCHMARKS:
        for scope, ks in [("common12", [12, 50, 200]), ("common12_non_english", [12, 50, 200]), ("english", [1, 12, 50, 200])]:
            group = agg.loc[agg.benchmark.eq(benchmark) & agg.scope.eq(scope) & agg.seed.eq(1234) & agg.K.isin(ks)]
            if group.empty:
                continue
            assert len(group) == 4 * 4 * len(ks) and group.raw_nll.notna().all()
            for metric in METRICS:
                for k in [0] + ks:
                    sub = group if k == 0 else group.loc[group.K.eq(k)]
                    correlations.append({"benchmark": benchmark, "scope": scope, "K": k, "metric": metric,
                                         "models": len(sub), "languages": int(sub.languages.iloc[0]),
                                         "spearman_lower_loss_vs_accuracy": rho(-sub.raw_nll, sub[metric]),
                                         "pearson_lower_loss_vs_accuracy": float(np.corrcoef(-sub.raw_nll, sub[metric])[0, 1])})
        for metric in METRICS:
            x = expert.loc[expert.benchmark.eq(benchmark) & expert.scope.eq("common12") & expert.seed.eq(1234) & expert.K.isin([12, 50, 200])]
            endpoints = x.loc[x.low.eq("1") & x.high.eq("32")]
            adjacent = x.loc[[p in {(1, 4), (4, 16), (16, 32)} for p in zip(x.low.astype(int), x.high.astype(int))]]
            assert len(endpoints) == 12 and len(adjacent) == 36
            s = size.loc[size.benchmark.eq(benchmark) & size.scope.eq("common12") & size.seed.eq(1234) & size.K.isin([12, 50, 200]) & size.low.eq("C") & size.high.eq("D")]
            assert len(s) == 12 and np.array_equal(s.D_total_low, s.D_total_high)
            inter = interactions.loc[interactions.benchmark.eq(benchmark) & interactions.scope.eq("common12") & interactions.seed.eq(1234) & interactions.low.eq("12") & interactions.high.eq("200")]
            d = agg.loc[agg.benchmark.eq(benchmark) & agg.scope.eq("common12") & agg.seed.eq(1234) & agg.ladder.eq("D") & agg.E.eq(32) & agg.K.eq(200)].iloc[0]
            full = expert.loc[expert.benchmark.eq(benchmark) & expert.scope.eq("all_evaluated") & expert.seed.eq(1234) & expert.K.isin([12, 50, 200]) & expert.low.eq("1") & expert.high.eq("32")]
            corr = next(r for r in correlations if r["benchmark"] == benchmark and r["scope"] == "common12" and r["metric"] == metric and r["K"] == 0)
            row = {"benchmark": benchmark, "metric": metric, "common12_languages": corr["languages"],
                   "loss_accuracy_spearman_48": corr["spearman_lower_loss_vs_accuracy"],
                   "expert_gain_mean_pp": endpoints[metric + "_gain_pp"].mean(),
                   "expert_gain_min_pp": endpoints[metric + "_gain_pp"].min(), "expert_gain_max_pp": endpoints[metric + "_gain_pp"].max(),
                   "expert_positive_of12": int((endpoints[metric + "_gain_pp"] > 1e-10).sum()),
                   "expert_loss_improves_of12": int((endpoints.loss_gain > 0).sum()),
                   "adjacent_expert_positive_of36": int((adjacent[metric + "_gain_pp"] > 1e-10).sum()),
                   "C_to_D_mean_gain_pp": s[metric + "_gain_pp"].mean(),
                   "C_to_D_positive_of12": int((s[metric + "_gain_pp"] > 1e-10).sum()),
                   "C_to_D_loss_improves_of12": int((s.loss_gain > 0).sum()),
                   "coverage_interaction_mean_pp": inter[metric + "_interaction_pp"].mean(),
                   "coverage_interaction_positive_of4": int((inter[metric + "_interaction_pp"] > 1e-10).sum()),
                   "loss_interaction_positive_of4": int((inter.loss_interaction_nats > 0).sum()),
                   "D_coverage_interaction_pp": inter.loc[inter.ladder.eq("D"), metric + "_interaction_pp"].iloc[0],
                   "D_E32_K200_accuracy_percent": 100 * d[metric],
                   "full_suite_expert_mean_pp": full[metric + "_gain_pp"].mean(),
                   "full_suite_expert_positive_of12": int((full[metric + "_gain_pp"] > 1e-10).sum())}
            overview.append(row)
    return pd.DataFrame(correlations), pd.DataFrame(overview)


def main(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    campaign, metrics, primary, bench, metadata, sets = load_inputs(args.root, args.campaign, args.source_campaign, args.membership_map)
    agg = aggregate(bench)
    expert = paired_contrasts(agg, "E", list(combinations([1, 4, 16, 32], 2)))
    size = paired_contrasts(agg, "ladder", [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")])
    # Changing trained populations are excluded from K contrasts.
    coverage = paired_contrasts(agg.loc[~agg.scope.eq("trained")], "K", list(combinations([1, 12, 50, 200], 2)))
    assert coverage.languages_low.eq(coverage.languages_high).all()
    interactions = coverage_interactions(coverage)
    correlations, overview = summarize(agg, expert, size, interactions)
    seed = interactions.loc[interactions.ladder.eq("B") & interactions.low.eq("12") & interactions.high.eq("200")]
    seed_summary = seed.groupby(["benchmark", "scope"]).agg(
        seeds=("seed", "nunique"), norm_mean=("acc_norm_interaction_pp", "mean"),
        norm_sd=("acc_norm_interaction_pp", "std"), norm_min=("acc_norm_interaction_pp", "min"),
        norm_max=("acc_norm_interaction_pp", "max"), norm_positive=("acc_norm_interaction_pp", lambda x: int((x > 1e-10).sum())),
        raw_mean=("acc_interaction_pp", "mean"), raw_min=("acc_interaction_pp", "min"), raw_max=("acc_interaction_pp", "max"),
        raw_positive=("acc_interaction_pp", lambda x: int((x > 1e-10).sum())),
        loss_mean=("loss_interaction_nats", "mean"), loss_positive=("loss_interaction_nats", lambda x: int((x > 0).sum())),
    ).reset_index()
    assert seed_summary.seeds.eq(3).all()
    language_correlations = []
    for (benchmark, language), g in bench.loc[bench.seed.eq(1234) & bench.K.isin([12, 50, 200]) & bench.in_common12].groupby(["benchmark", "language"]):
        assert len(g) == 48
        language_correlations.append({"benchmark": benchmark, "language": language, "models": 48,
            "spearman_norm": rho(-g.raw_nll, g.acc_norm), "spearman_raw": rho(-g.raw_nll, g.acc),
            "accuracy_norm_min": g.acc_norm.min(), "accuracy_norm_max": g.acc_norm.max()})
    membership = bench.loc[bench.model_id.eq(metadata.model_id.iloc[0]), ["benchmark", "language", "corpus_language", "in_common12", "is_english"]].copy()
    for k in [1, 12, 50, 200]:
        membership[f"trained_at_K{k}"] = membership.corpus_language.isin(sets[k])
    outputs = {"language_loss_accuracy": bench, "model_bindings": metadata, "language_membership": membership,
               "cohort_metrics": agg, "expert_contrasts": expert, "size_contrasts": size,
               "coverage_contrasts": coverage, "coverage_interactions": interactions,
               "loss_accuracy_correlations": correlations, "benchmark_summary": overview,
               "b_seed_interactions": seed, "b_seed_summary": seed_summary,
               "language_correlations": pd.DataFrame(language_correlations)}
    args.output.mkdir(parents=True)
    for name, frame in outputs.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    inputs = {"metrics": args.campaign / "metrics.json", "campaign": args.campaign / "campaign.json",
              "source_preparation": args.source_campaign / "preparation_receipt.json", "membership_map": args.membership_map,
              "loss_release": args.root / "result_analysis/stage1_abcd_jointmoe_x1_v1/release_manifest.json"}
    receipt = {"status": "verified", "inputs": {n: {"path": str(p.resolve()), "sha256": sha256_file(p)} for n, p in inputs.items()},
               "script_sha256": sha256_file(Path(__file__)), "models": 82, "primary_models": 64,
               "benchmark_rows": len(bench), "checks": {"sealed_sources_verified": True, "all_492_checkpoint_bindings_match": True,
               "primary_loss_release_verified": True, "seed_loss_reports_verified": True,
               "identical_sample_counts_across_models": True, "fixed_cohort_pairing": True, "C_D_total_tokens_match": True},
               "mapping_additions": MAP_ADDITIONS, "belebele_aliases": {"est_Latn": "ekk_Latn", "zho_Hans": "cmn_Hani"},
               "scope": "Final checkpoints; fixed total horizon within ladder; no matched-target-exposure benchmark or fitted benchmark scaling law.",
               "seed_scope": "A/C/D and B/E4,K50 have one seed; B/E1,16,32 x K1,12,200 have three seeds.",
               "uncertainty": "Correlations and setting/language counts are descriptive. No significance tests or pseudo-replicated confidence intervals.",
               "outputs": {n: {"rows": len(f), "sha256": sha256_file(args.output / f"{n}.csv")} for n, f in outputs.items()}}
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(overview.to_string(index=False))
    print("B SEED INTERACTIONS, FIXED COMMON12")
    print(seed_summary.loc[seed_summary.scope.eq("common12")].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--membership-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
