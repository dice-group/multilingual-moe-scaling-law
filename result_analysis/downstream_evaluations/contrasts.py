"""Language-balanced downstream contrasts.

Varieties are averaged before corpus languages. Endpoints match total tokens,
not target exposure. Metric signs and all expert comparisons are preserved.
"""
import numpy as np
import pandas as pd

LOSS = {"flores_mono", "ntrex_mono"}
IDENTITY = ["model_id", "ladder", "E", "K", "seed", "benchmark"]
METRICS = ["primary", "acc", "acc_token", "acc_byte", "margin", "macro_f1", "weighted_f1",
           "external_nll", "unigram_adjusted_loss", "bits_per_byte"]
PAIRS = [(1, 32), (1, 4), (4, 32), (16, 32), (4, 16)]


def orientation(benchmark, metric):
    return -1 if metric in {"external_nll", "unigram_adjusted_loss", "bits_per_byte"} or (metric == "primary" and benchmark in LOSS) else 1


def scale(benchmark, metric):
    return 100 if metric.startswith("acc") or metric in {"macro_f1", "weighted_f1"} or (metric == "primary" and benchmark not in LOSS) else 1


def aggregate(frame, common12, *, coverages=(12, 50, 200), coverage_pair=(12, 200),
              expert_pairs=PAIRS, cohorts=None, metrics=METRICS,
              metric_directions=None, metric_scales=None):
    extra = ["validation_nll", "validation_adjusted_loss", "chance", "majority_baseline"]
    selected = frame[frame.K.isin(coverages) & frame.corpus_language.isin(common12)].copy()
    if "validation_nll" not in selected:
        selected["validation_nll"] = np.nan
    fields = [m for m in list(metrics) + extra if m in selected]
    languages = selected.groupby(IDENTITY + ["corpus_language", "panel"], dropna=False)[fields].mean().reset_index()
    cohort_map = cohorts if cohorts is not None else {
        "common12": common12,
        "common12_non_english": [x for x in common12 if x != "eng_Latn"],
        "english": ["eng_Latn"]}
    cohorts, gains, interactions = [], [], []
    low_k, high_k = coverage_pair
    for cohort, members in cohort_map.items():
        data = languages[languages.corpus_language.isin(members)]
        for key, group in data.groupby(IDENTITY + ["panel"]):
            if metric_directions is None and not group.validation_nll.notna().all():
                raise ValueError(f"unmatched validation languages: {key}")
            row = dict(zip(IDENTITY + ["panel"], key), cohort=cohort, languages=len(group))
            row.update({m:group[m].mean() for m in fields})
            cohorts.append(row)
        for key, group in data.groupby(["benchmark", "ladder", "seed", "panel"]):
            benchmark, ladder, seed, panel = key
            identity = dict(benchmark=benchmark, ladder=ladder, seed=seed, panel=panel, cohort=cohort)
            for metric in metrics:
                if metric not in group or group[metric].isna().all():
                    continue
                wide = group.pivot(index="corpus_language", columns=["K", "E"], values=metric)
                loss = group.pivot(index="corpus_language", columns=["K", "E"], values="validation_nll")
                sign = orientation(benchmark, metric) if metric_directions is None else metric_directions[benchmark, metric]
                unit = scale(benchmark, metric) if metric_scales is None else metric_scales.get((benchmark, metric), 1)
                for low, high in expert_pairs:
                    for k in coverages:
                        if not {(k,low),(k,high)} <= set(wide.columns):
                            continue
                        pair = wide[[(k,low),(k,high)]]
                        if not pair.notna().all().all():
                            raise ValueError(f"unpaired benchmark cells: {identity}/{metric}/{k}/{low}/{high}")
                        diff = sign * (pair[k,high] - pair[k,low]) * unit
                        vd = loss[k,low] - loss[k,high]
                        gains.append(dict(**identity, metric=metric, E_low=low, E_high=high, K=k,
                            languages=len(diff), gain=diff.mean(), languages_improved=int((diff>1e-12).sum()),
                            languages_worsened=int((diff < -1e-12).sum()), validation_nll_reduction=vd.mean()))
                    columns = [(k,e) for k in coverage_pair for e in (low,high)]
                    if not set(columns) <= set(wide.columns):
                        continue
                    if not wide[columns].notna().all().all():
                        raise ValueError(f"incomplete benchmark interaction: {identity}/{metric}")
                    gain12 = sign*(wide[low_k,high]-wide[low_k,low])*unit
                    gain200 = sign*(wide[high_k,high]-wide[high_k,low])*unit
                    diff = gain200-gain12
                    vd = (loss[high_k,low]-loss[high_k,high])-(loss[low_k,low]-loss[low_k,high])
                    interactions.append(dict(**identity, metric=metric, E_low=low, E_high=high,
                        languages=len(diff), **{f"gain_K{low_k}": gain12.mean(), f"gain_K{high_k}": gain200.mean()},
                        interaction=diff.mean(), languages_positive=int((diff>1e-12).sum()),
                        languages_negative=int((diff < -1e-12).sum()),
                        validation_nll_interaction=vd.mean(),
                        coverage_penalty_low=(sign*(wide[low_k,low]-wide[high_k,low])*unit).mean(),
                        coverage_penalty_high=(sign*(wide[low_k,high]-wide[high_k,high])*unit).mean()))
    tables = {"cohort_metrics":pd.DataFrame(cohorts), "expert_gains":pd.DataFrame(gains), "coverage_interactions":pd.DataFrame(interactions)}
    tables["language_metrics"] = languages
    return tables
