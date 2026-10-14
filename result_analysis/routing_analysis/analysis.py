"""Routing metrics and fixed-cohort comparisons; no causal interpretation."""
from itertools import combinations

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_routing_repair import reporting


def from_parquet(data):
    """Recompute routing summaries and comparisons from saved real-token counts."""
    counts = pd.read_parquet(data / "routing_counts.parquet", columns=[
        *reporting.ID_FIELDS, "language", "layer_index", "expert_token_counts"])
    losses = pd.read_parquet(data / "routing_losses.parquet")
    columns = ["E", "readout_id", "language", "ladder", "K", "model_seed", "D_total",
               "nll_sum", "token_count", "target_token_ids_sha256"]
    dense = pd.concat([pd.read_parquet(data / f"{name}.parquet", columns=columns)
                       for name in ("main_grid", "seed_comparison")], ignore_index=True)
    metadata = pd.read_parquet(data / "languages.parquet")
    cohorts = {f"common{k}": sorted(metadata.loc[metadata[f"in_k{k}"], "language"]) for k in (12,50)}
    checkpoints, languages = summaries_from_counts(counts, losses, dense, cohorts=cohorts)
    return {"checkpoint_summary": checkpoints, "language_metrics": languages,
            **analyze(checkpoints, languages, cohorts=tuple(cohorts))}


def summaries_from_counts(counts, losses, dense_losses, *, cohorts):
    """Reconstruct point estimates from portable count and validation parquets.

    cohorts maps names to exact language memberships. Bootstrap blocks are optional and unused here; this function computes
    the point estimates consumed by analyze().
    """
    from multilingual_megatron.experiments.scaling.stage1_routing_repair.analysis import entropy, js_matrix
    if counts.duplicated(["id", "language", "layer_index"]).any():
        raise ValueError("duplicate routing count row")
    dense = dense_losses.loc[dense_losses.E.eq(1)].drop_duplicates(["readout_id", "language"])
    dense_keys = ["ladder", "K", "model_seed", "D_total", "language"]
    if dense.duplicated(dense_keys).any():
        raise ValueError("ambiguous dense control")
    dense = dense.set_index(dense_keys)
    sparse = losses.set_index(["id", "language"])
    checkpoints, language_rows = [], []
    for model, group in counts.groupby("id", sort=True):
        meta = {k: group.iloc[0][k] for k in reporting.ID_FIELDS}
        for cohort, members in cohorts.items():
            members = sorted(members)
            if len(set(members)) != len(members) or len(members) < 2:
                raise ValueError("cohorts require unique language members")
            if not set(members) <= set(group.language):
                continue
            layers, breadth, distance = [], [], []
            for _, layer in group.groupby("layer_index", sort=True):
                ordered = layer.set_index("language").loc[members]
                values = np.stack(ordered.expert_token_counts).astype(float)
                if not np.isfinite(values).all() or (values < 0).any() or (values.sum(1) <= 0).any():
                    raise ValueError("invalid routing counts")
                p = values / values.sum(1, keepdims=True)
                layers.append(reporting.routing_metrics(p))
                breadth.append(np.exp(entropy(p)) / meta["experts"])
                distance.append(js_matrix(p).sum(1) / (len(members)-1))
            summary, _ = reporting.summarize_layers(layers)
            checkpoints.append({**meta, "cohort": cohort, "cohort_size": len(members), **summary})
            for i, language in enumerate(members):
                s = sparse.loc[(model, language)]
                d = dense.loc[(meta["ladder"], meta["languages"], meta["seed"], meta["tokens_seen"], language)]
                if s.token_count != d.token_count or s.target_token_ids_sha256 != d.target_token_ids_sha256:
                    raise ValueError("dense/sparse validation targets differ")
                gain = float((d.nll_sum-s.nll_sum)/s.token_count)
                language_rows.append({**meta, "cohort": cohort, "cohort_size": len(members),
                    "language": language, "effective_expert_fraction": float(np.median(breadth, axis=0)[i]),
                    "mean_pair_js_nats": float(np.median(distance, axis=0)[i]), "moe_gain_nats": gain})
    return pd.DataFrame(checkpoints), pd.DataFrame(language_rows)


def from_counts(counts):
    """Summarize [layer, language, expert] real-token counts with equal language weights.

    Supply one checkpoint and one fixed language cohort. Layers and languages
    are not independent training replicates; no confidence interval is inferred.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or min(counts.shape) == 0 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("finite nonnegative layer/language/expert counts required")
    totals = counts.sum(axis=-1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("empty language/layer routing counts")
    layers = [reporting.routing_metrics(p) for p in counts / totals]
    summary, _ = reporting.summarize_layers(layers)
    return {"layers": pd.DataFrame(layers), "summary": summary}


def analyze(checkpoints, languages, *, cohorts=("common12", "common50")):
    """Recompute coverage deltas and within-checkpoint loss associations.

    Inputs are checkpoint and per-language routing summaries. Match coverage
    comparisons within size, expert count, seed, horizon, and identical cohort.
    Native/trained cohorts cannot support fixed-language coverage comparisons.
    """
    if not cohorts or "trained" in cohorts or len(set(cohorts)) != len(cohorts):
        raise ValueError("unique fixed cohorts required; trained cohorts change population")
    summary = checkpoints[checkpoints.cohort.isin(cohorts)].copy()
    language = languages[languages.cohort.isin(cohorts)].copy()
    group_keys = ["ladder", "experts", "seed", "tokens_seen", "cohort"]
    if summary.empty or set(summary.cohort) != set(cohorts):
        raise ValueError("missing requested checkpoint cohorts")
    if summary.duplicated(["id", "cohort"]).any() or summary.duplicated(group_keys+["languages"]).any():
        raise ValueError("duplicate checkpoint/cohort or coverage identity")
    if language.duplicated(["id", "cohort", "language"]).any():
        raise ValueError("duplicate language observations")
    groups = {(model, cohort): part for (model, cohort), part in language.groupby(["id", "cohort"])}
    if set(groups) != set(zip(summary.id, summary.cohort)):
        raise ValueError("checkpoint/language populations differ")
    memberships = {}
    associations = []
    for row in summary.to_dict("records"):
        part = groups[row["id"], row["cohort"]]
        members = frozenset(part.language)
        if len(members) != row["cohort_size"] or memberships.setdefault(row["cohort"], members) != members:
            raise ValueError("fixed language cohort differs across checkpoints")
        for field in (*reporting.ID_FIELDS, "cohort_size"):
            if not part[field].eq(row[field]).all():
                raise ValueError(f"inconsistent language metadata: {field}")
        for predictor in ("effective_expert_fraction", "mean_pair_js_nats"):
            rho, reason = reporting.spearman(part[predictor], part.moe_gain_nats)
            associations.append({**{k: row[k] for k in reporting.ID_FIELDS},
                "cohort": row["cohort"], "predictor": predictor, "n_languages": len(part),
                "spearman_rho": rho, "undefined_reason": reason})
    if not np.isfinite(summary[list(reporting.METRICS)]).all().all():
        raise ValueError("nonfinite fixed-cohort routing metrics")
    contrasts = []
    for key, group in summary.groupby(group_keys, sort=True):
        for left, right in combinations(group.sort_values("languages").to_dict("records"), 2):
            for metric in reporting.METRICS:
                contrasts.append({**dict(zip(group_keys, key)), "K_from": left["languages"],
                    "K_to": right["languages"], "metric": metric, "delta": right[metric]-left[metric]})
    return {"coverage_contrasts": pd.DataFrame(contrasts, columns=group_keys+["K_from", "K_to", "metric", "delta"]),
            "loss_associations": pd.DataFrame(associations)}
