"""Balanced, provenance-checked endpoint contrasts across completed campaigns.

Freeze partial availability before inspecting outcomes. No model selection by
score, pooled-size correlation, or matched-exposure inference from endpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign import load, result_for, reduce_scores
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign import checked, read, ref, save
from result_analysis.downstream_evaluations.endpoints.analyze import load_inputs, sealed

LOSS = {"flores_mono", "ntrex_mono"}
IDENTITY = ["model_id", "ladder", "E", "K", "seed", "benchmark"]
METRICS = ["primary", "acc", "acc_token", "acc_byte", "margin", "macro_f1", "weighted_f1",
           "external_nll", "unigram_adjusted_loss", "bits_per_byte"]
PAIRS = [(1, 32), (1, 4), (4, 32), (16, 32), (4, 16)]


def orientation(benchmark, metric):
    return -1 if metric in {"external_nll", "unigram_adjusted_loss", "bits_per_byte"} or (metric == "primary" and benchmark in LOSS) else 1


def scale(benchmark, metric):
    return 100 if metric.startswith("acc") or metric in {"macro_f1", "weighted_f1"} or (metric == "primary" and benchmark not in LOSS) else 1


def main(root: Path, output: Path, require_complete: bool = False):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'endpoint_diagnostics_source.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    base = root / "outputs/stage1_benchmark_jointmoe"
    extension_root = base / "stage1_extension_full_a40_20260921_ready"
    extension = load(extension_root / "campaign.json")
    available = [j for j in extension["jobs"] if j["phase"] == "full"
                 and (extension_root / "scores" / j["id"] / "scores.json").is_file()]
    if require_complete:
        assert len(available) == 410, "Incomplete extension campaign"
    save(output / "snapshot_inventory.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
         campaign=ref(extension_root / "campaign.json"), jobs=[dict(id=j["id"], spec=j["spec"]) for j in available]), "manifest_sha256")
    old_root = base / "stage1_full_final_h100_20260920"
    source_root = base / "stage1_abcd_jointmoe_all_readouts_a40_v2"
    mapping = root / "external/research/scaling/stage1_paper_naacl2027_scaling_laws/tables/benchmark_training_membership_map.json"
    _, _, primary, old, metadata, sets = load_inputs(root, old_root, source_root, mapping)
    endpoint = primary.sort_values("D_total").groupby(["ladder", "E", "K", "language"]).tail(1).copy()
    endpoint["seed"] = 1234
    val = endpoint.rename(columns={"language": "corpus_language"}).merge(
        metadata[["model_id", "ladder", "E", "K", "seed"]], on=["ladder", "E", "K", "seed"], validate="many_to_one")
    val_cols = ["model_id", "corpus_language", "raw_nll", "adjusted_loss"]
    seed_val = old.loc[old.seed.ne(1234) & old.raw_nll.notna(), val_cols].drop_duplicates()
    val = pd.concat([val[val_cols], seed_val], ignore_index=True).rename(columns={"raw_nll": "validation_nll", "adjusted_loss": "validation_adjusted_loss"})
    assert not val.duplicated(["model_id", "corpus_language"]).any()
    models = metadata.set_index("model_id")
    metadata.to_csv(output / "model_bindings.csv", index=False)

    old["primary"] = old.acc_norm
    old["panel"] = "original_complete"
    frames = [old.drop(columns=["raw_nll", "adjusted_loss"])]
    candidate_root = base / "stage1_candidate_full_a40_20260921"
    candidate = load(candidate_root / "campaign.json")
    candidate_metrics = sealed(candidate_root / "metrics.json")
    assert candidate_metrics["status"] == "complete" and not candidate_metrics["incomplete_jobs"]
    assert candidate_metrics["campaign"] == ref(candidate_root / "campaign.json")
    candidates = [dict(v, primary=v[v["primary_metric"]],
                       margin=v.get("margin_pmi", v.get("margin")) if v["benchmark"] == "x_csqa" else v.get("margin"),
                       panel="candidate_complete") for v in candidate_metrics["metrics"] if v["group"] == "all"]
    frames.append(pd.DataFrame(candidates))
    # Bind all supplementary scores to the same checkpoint inventory as the
    # original study and sealed validation endpoints, including B repeats.
    for j in [j for j in candidate["jobs"] if j["phase"] == "full"] + available:
        spec = read(checked(j["spec"]))
        assert spec["checkpoint"]["iteration_path"] == models.loc[j["model_id"], "checkpoint_path"]

    extension_rows, receipts = [], []
    if require_complete:
        metrics = sealed(extension_root / "metrics.json")
        assert metrics['status'] == 'complete' and not metrics['incomplete_jobs']
        assert metrics['campaign'] == ref(extension_root / 'campaign.json')
        extension_rows = [dict(v, primary=v[v['primary_metric']], panel='extension_complete')
                          for v in metrics['metrics'] if v['group'] == 'all']
        for row in extension_rows:
            if 'raw_nll' in row:
                row['external_nll'] = row.pop('raw_nll')
        assert len(extension_rows) == 117588
        # Full collection has already reduced every sealed score. The separate
        # example-level audit revalidates raw scores used for diagnostics.
    current, manifest = None, None
    for index, j in enumerate(sorted([] if require_complete else available, key=lambda j: (j["benchmark"], j["id"])), 1):
        result = result_for(j)
        assert result is not None
        if current != j["benchmark"]:
            current = j["benchmark"]
            manifest = None
            manifest = read(checked(extension["inputs"][current]["request_manifest"]))
        values = reduce_scores(extension, manifest, result["scores"])
        assert len(values) == extension["inputs"][current]["languages"]
        assert sum(v["samples"] for v in values) == j["documents"]
        for value in values:
            row = dict(value, **{k: j[k] for k in IDENTITY}, primary=value[value["primary_metric"]], panel="extension_partial")
            if "raw_nll" in row:
                row["external_nll"] = row.pop("raw_nll")
            extension_rows.append(row)
        receipts.append(dict(id=j["id"], result_sha256=result["result_sha256"], score_spec=j["spec"]))
        del result
        print(f"Reduced frozen extension snapshot {index}/{len(available)}: {j['id']}", flush=True)
    frames.append(pd.DataFrame(extension_rows))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.merge(val, on=["model_id", "corpus_language"], how="left", validate="many_to_one")
    if require_complete:
        assert frame.groupby('benchmark').model_id.nunique().eq(82).all()
        assert frame.benchmark.nunique() == 14
        for campaign, panel in [(candidate, 'candidate_complete'), (extension, 'extension_complete')]:
            for j in (j for j in campaign['jobs'] if j['phase'] == 'full'):
                values = frame[(frame.model_id == j['model_id']) & (frame.benchmark == j['benchmark'])]
                assert len(values) == campaign['inputs'][j['benchmark']]['languages']
                assert values.samples.sum() == j['documents']
    assert not frame.duplicated(["model_id", "benchmark", "language"]).any()
    assert frame.groupby(["benchmark", "language"]).samples.nunique().eq(1).all()
    frame.to_csv(output / "language_metrics_with_validation.csv", index=False)
    pd.DataFrame([dict(**{k:j[k] for k in IDENTITY}, samples=j['documents']) for j in available]).to_csv(output / "extension_availability.csv", index=False)

    # Average regional varieties within corpus language before language-macro
    # aggregation. Keep English and non-English sensitivities separate.
    extra = ["validation_nll", "validation_adjusted_loss", "chance", "majority_baseline"]
    selected = frame[frame.K.isin([12, 50, 200]) & frame.corpus_language.isin(sets[12])].copy()
    fields = [m for m in METRICS + extra if m in selected]
    languages = selected.groupby(IDENTITY + ["corpus_language", "panel"], dropna=False)[fields].mean().reset_index()
    languages.to_csv(output / "fixed_cohort_language_metrics.csv", index=False)
    cohorts, gains, interactions = [], [], []
    for cohort in ["common12", "common12_non_english", "english"]:
        data = languages if cohort == "common12" else languages[languages.corpus_language.ne("eng_Latn") if cohort.endswith("non_english") else languages.corpus_language.eq("eng_Latn")]
        for key, group in data.groupby(IDENTITY + ["panel"]):
            assert group.validation_nll.notna().all(), (key, "unmatched validation languages")
            row = dict(zip(IDENTITY + ["panel"], key), cohort=cohort, languages=len(group))
            row.update({m:group[m].mean() for m in fields})
            cohorts.append(row)
        for key, group in data.groupby(["benchmark", "ladder", "seed", "panel"]):
            benchmark, ladder, seed, panel = key
            identity = dict(benchmark=benchmark, ladder=ladder, seed=seed, panel=panel, cohort=cohort)
            for metric in METRICS:
                if metric not in group or group[metric].isna().all():
                    continue
                wide = group.pivot(index="corpus_language", columns=["K", "E"], values=metric)
                loss = group.pivot(index="corpus_language", columns=["K", "E"], values="validation_nll")
                sign, unit = orientation(benchmark, metric), scale(benchmark, metric)
                for low, high in PAIRS:
                    for k in (12,50,200):
                        if not {(k,low),(k,high)} <= set(wide.columns):
                            continue
                        pair = wide[[(k,low),(k,high)]]
                        assert pair.notna().all().all(), (identity, metric, k, low, high)
                        diff = sign * (pair[k,high] - pair[k,low]) * unit
                        vd = loss[k,low] - loss[k,high]
                        gains.append(dict(**identity, metric=metric, E_low=low, E_high=high, K=k,
                            languages=len(diff), gain=diff.mean(), languages_improved=int((diff>1e-12).sum()),
                            languages_worsened=int((diff < -1e-12).sum()), validation_nll_reduction=vd.mean()))
                    columns = [(k,e) for k in (12,200) for e in (low,high)]
                    if not set(columns) <= set(wide.columns):
                        continue
                    assert wide[columns].notna().all().all()
                    gain12 = sign*(wide[12,high]-wide[12,low])*unit
                    gain200 = sign*(wide[200,high]-wide[200,low])*unit
                    diff = gain200-gain12
                    vd = (loss[200,low]-loss[200,high])-(loss[12,low]-loss[12,high])
                    interactions.append(dict(**identity, metric=metric, E_low=low, E_high=high,
                        languages=len(diff), gain_K12=gain12.mean(), gain_K200=gain200.mean(),
                        interaction=diff.mean(), languages_positive=int((diff>1e-12).sum()),
                        languages_negative=int((diff < -1e-12).sum()),
                        validation_nll_interaction=vd.mean(),
                        coverage_penalty_low=(sign*(wide[12,low]-wide[200,low])*unit).mean(),
                        coverage_penalty_high=(sign*(wide[12,high]-wide[200,high])*unit).mean()))
    tables = {"cohort_metrics":pd.DataFrame(cohorts), "expert_gains":pd.DataFrame(gains), "coverage_interactions":pd.DataFrame(interactions)}
    for name, table in tables.items():
        table.to_csv(output / (name+".csv"), index=False)
    contrasts = tables["coverage_interactions"]
    headline = contrasts.query("cohort == 'common12' and metric == 'primary' and E_low == 1 and E_high == 32 and seed == 1234")
    # All complete original and candidate tasks must have all four ladders.
    if require_complete:
        assert len(headline) == 14*4
    else:
        assert len(headline[headline.panel.isin(['original_complete', 'candidate_complete'])]) == 9*4
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = headline[~headline.benchmark.isin(LOSS)].pivot(index="benchmark",columns="ladder",values="interaction")
    fig, ax = plt.subplots(figsize=(8,7))
    heat = ax.imshow(matrix.values,cmap="RdYlGn",vmin=-3,vmax=3,aspect="auto")
    ax.set_xticks(range(4),matrix.columns);ax.set_yticks(range(len(matrix)),matrix.index)
    for i in range(len(matrix)):
        for j in range(4):ax.text(j,i,f"{matrix.iloc[i,j]:+.2f}",ha="center",va="center",fontsize=9)
    ax.set_title("Change in E32−E1 accuracy gain: K200 minus K12\nFixed shared languages; seed 1234; percentage points")
    fig.colorbar(heat,ax=ax,label="Positive = greater expert benefit at K200")
    fig.text(.5,.01,"Final checkpoints at fixed total training budget; not the paper's matched-exposure estimator.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(output/'endpoint_interactions.png',dpi=180);fig.savefig(output/'endpoint_interactions.pdf');plt.close(fig)
    save(output / "analysis_receipt.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source=ref(snapshot), original_metrics=ref(old_root/'metrics.json'), candidate_metrics=ref(candidate_root/'metrics.json'),
        extension_metrics=ref(extension_root/'metrics.json') if require_complete else None,
        extension_snapshot=ref(output/'snapshot_inventory.json'), extension_results=receipts,
        verified_checks=dict(exact_checkpoint_bindings=True,complete_candidate_matrix=True,fixed_language_pairing=True,complete_extension_matrix=require_complete,complete_partial_result_reduction=not require_complete),
        interpretation="Descriptive endpoint contrasts. Positive interaction means expert gain increases from K12 to K200. Not matched target exposure; languages and items are nested, not independent training seeds. Contrasts require complete paired cells; complete mode uses all three sealed collections, partial mode selects only by completion status.",
        outputs={name:ref(output/(name+'.csv')) for name in tables}), "manifest_sha256")
    print("FULL MAIN-SEED PRIMARY INTERACTIONS");print(headline.to_string(index=False))
    print("PARTIAL EXTENSION PRIMARY AND LOSS INTERACTIONS");print(contrasts.query("panel == 'extension_partial' and cohort == 'common12' and metric in ['primary','external_nll']").to_string(index=False))
    print("B SEEDS, PRIMARY E1 TO E32");print(contrasts.query("panel == 'candidate_complete' and ladder == 'B' and cohort == 'common12' and metric == 'primary' and E_low == 1 and E_high == 32").to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete', action='store_true')
    args=parser.parse_args();main(args.root.resolve(),args.output.resolve(),args.require_complete)
