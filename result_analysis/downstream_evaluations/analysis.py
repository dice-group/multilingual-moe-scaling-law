"""Language-balanced downstream comparisons for explicitly selected panels."""
import numpy as np

from . import contrasts


def from_parquet(data):
    """Recompute the main benchmark comparisons from language-level scores."""
    import pandas as pd
    frame = pd.read_parquet(data / "benchmarks.parquet")
    metadata = pd.read_parquet(data / "languages.parquet")
    languages = sorted(metadata.loc[metadata.in_k12, "language"])
    loss_tasks = {"flores_mono", "ntrex_mono"}
    directions = {(b, "primary"): -1 if b in loss_tasks else 1 for b in frame.benchmark.unique()}
    scales = {(b, "primary"): 1 if b in loss_tasks else 100 for b in frame.benchmark.unique()}
    return analyze(frame, languages=languages, coverages=(12, 200), coverage_pair=(12, 200),
                   expert_pairs=((1, 32),), metric_directions=directions, metric_scales=scales)


def analyze(frame, *, languages, coverages, coverage_pair, expert_pairs,
            metric_directions, metric_scales=None, cohort_name="selected"):
    """Compare a fixed language cohort with explicit metric meaning.

    directions maps (benchmark, metric) to +1 (higher better) or -1 (lower
    better); scales defaults to 1, e.g. use 100 for accuracy percentage points.
    Varieties are averaged within corpus languages first. Only supplied cells
    are analyzed; this does not assert the published 14-task/82-model inventory.
    """
    languages, coverages = tuple(languages), tuple(coverages)
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("unique nonempty language cohort required")
    if len(coverage_pair) != 2 or coverage_pair[0] >= coverage_pair[1] or not set(coverage_pair) <= set(coverages):
        raise ValueError("increasing coverage pair must belong to selected coverages")
    if not expert_pairs or any(a >= b or a <= 0 for a, b in expert_pairs):
        raise ValueError("increasing positive expert pairs required")
    data = frame[frame.K.isin(coverages) & frame.corpus_language.isin(languages)].copy()
    if data.empty or data.duplicated(["model_id", "benchmark", "language"]).any():
        raise ValueError("empty or duplicate endpoint-language panel")
    if data[contrasts.IDENTITY+["language", "corpus_language", "panel"]].isna().any().any():
        raise ValueError("missing endpoint identity")
    if "samples" in data and not data.groupby(["benchmark", "language"]).samples.nunique().eq(1).all():
        raise ValueError("different benchmark items across models")
    metrics = tuple(dict.fromkeys(metric for _, metric in metric_directions))
    if not metrics or not set(metrics) <= set(data):
        raise ValueError("explicit directions for existing metrics required")
    for benchmark, group in data.groupby("benchmark"):
        for metric in metrics:
            if not np.isfinite(group[metric].dropna()).all():
                raise ValueError(f"nonfinite metric: {benchmark}/{metric}")
            if group[metric].notna().any() and metric_directions.get((benchmark, metric)) not in (-1, 1):
                raise ValueError(f"missing metric direction: {benchmark}/{metric}")
    if metric_scales and any(not np.isfinite(v) or v <= 0 for v in metric_scales.values()):
        raise ValueError("positive finite metric scales required")
    return contrasts.aggregate(data, languages, coverages=coverages, coverage_pair=coverage_pair,
        expert_pairs=expert_pairs, cohorts={cohort_name: languages}, metrics=metrics,
        metric_directions=metric_directions, metric_scales=metric_scales or {})


def compare(*args, **kwargs):
    """Compatibility helper for strict comparison against a published table."""
    from .reproduce import compare as compare_published
    return compare_published(*args, **kwargs)


def run(evidence, output):
    """Compatibility entrypoint for the separate published-result reproduction."""
    from .reproduce import run as reproduce
    return reproduce(evidence, output)
