"""Compute coverage gaps and gap reductions from observed losses."""
from .matched_exposure import coverage_gap, summarize


def from_parquet(data):
    """Default main-grid comparison using only the portable observation table."""
    import pandas as pd
    rows = pd.read_parquet(data / "main_grid.parquet").query("metric_role == 'trained'")
    rows = rows.rename(columns={"E": "expert_count", "K": "language_coverage",
        "N_act": "active_parameters", "D_total": "total_tokens", "D_t": "target_exposure"})
    return analyze(rows, reference_coverage=12, target_coverage=200,
                   expected_expert_counts=(1, 4, 16, 32), reference_expert_count=1)


def analyze(rows, *, reference_coverage, target_coverage,
            expected_expert_counts=None, reference_expert_count=None):
    """Return matched points, size-level coverage gaps and an equal-size summary."""
    points = coverage_gap(rows, reference_coverage, target_coverage)
    by_size, summary = summarize(points, expected_expert_counts, reference_expert_count)
    return {"matched_points": points, "by_size": by_size, "summary": summary}


def run(evidence, output):
    """Reproduce the reported comparisons and export their tables and figures."""
    from .reproduce import run as reproduce
    return reproduce(evidence, output)
