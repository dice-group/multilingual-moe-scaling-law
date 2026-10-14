"""Coverage gaps at matched target exposure, without extrapolation."""
import numpy as np
import pandas as pd


def coverage_gap(rows, reference_coverage=12, target_coverage=200):
    """Subtract same-language reference loss at the target's realized exposure.

    Interpolate reference loss in log target exposure. Omit target points
    outside reference support; never substitute total tokens / language coverage.
    """
    fields = ["ladder", "expert_count", "language_coverage", "language",
              "trajectory_id", "readout_id", "active_parameters", "total_tokens",
              "target_exposure", "raw_nll", "adjusted_loss"]
    if not 0 < reference_coverage < target_coverage:
        raise ValueError("positive increasing language coverage required")
    data = rows.loc[rows.language_coverage.isin([reference_coverage, target_coverage]), fields]
    numeric = ["expert_count", "language_coverage", "active_parameters",
               "total_tokens", "target_exposure", "raw_nll", "adjusted_loss"]
    if data.empty or data.isna().any().any() or not np.isfinite(data[numeric]).all().all():
        raise ValueError("empty or nonfinite observations")
    if not data[numeric[:-2]].gt(0).all().all():
        raise ValueError("positive counts, parameters and target exposure required")
    if data.duplicated(["trajectory_id", "readout_id", "language"]).any():
        raise ValueError("duplicate observation")
    keys = ["ladder", "expert_count", "language"]
    if data.groupby(keys + ["language_coverage"]).trajectory_id.nunique().gt(1).any():
        raise ValueError("select one trajectory per architecture, coverage and language")
    if data.groupby(["ladder", "expert_count"]).active_parameters.nunique().gt(1).any():
        raise ValueError("matched architectures must have equal active parameters")
    references = {}
    for key, group in data[data.language_coverage.eq(reference_coverage)].groupby(keys, observed=True):
        curve = group.sort_values("target_exposure")
        if curve.target_exposure.duplicated().any():
            raise ValueError("duplicate reference target exposure")
        references[key] = curve
    records = []
    for row in data[data.language_coverage.eq(target_coverage)].itertuples(index=False):
        reference = references.get((row.ladder, row.expert_count, row.language))
        if reference is None or not reference.target_exposure.min() <= row.target_exposure <= reference.target_exposure.max():
            continue
        x = np.log(reference.target_exposure.to_numpy(float))
        query = np.log(float(row.target_exposure))
        reference_loss = np.interp(query, x, reference.adjusted_loss.to_numpy(float))
        gap = float(row.adjusted_loss - reference_loss)
        raw_gap = float(row.raw_nll - np.interp(query, x, reference.raw_nll.to_numpy(float)))
        # The language-specific loss offset must cancel in a same-language contrast.
        if not np.isclose(gap, raw_gap, atol=2e-12):
            raise ValueError("adjusted and raw coverage gaps disagree")
        records.append(dict(ladder=row.ladder, active_parameters=row.active_parameters,
            expert_count=row.expert_count, language=row.language,
            reference_coverage=reference_coverage, target_coverage=target_coverage,
            reference_trajectory_id=reference.trajectory_id.iloc[0],
            target_trajectory_id=row.trajectory_id, target_readout_id=row.readout_id,
            total_tokens=row.total_tokens, target_exposure=row.target_exposure,
            reference_loss=reference_loss, target_loss=row.adjusted_loss, coverage_gap=gap))
    if not records:
        raise ValueError("no matched target exposure support")
    return pd.DataFrame(records).sort_values(
        ["ladder", "expert_count", "language", "target_exposure"]).reset_index(drop=True)


def summarize(points, expected_expert_counts=None, reference_expert_count=None):
    """Average nested rows within active-size groups, then weight groups equally.

    gap_reduction is fractional relative to the requested expert count.
    A zero reference gap cannot define a relative reduction.
    """
    by_size = (points.groupby(["ladder", "active_parameters", "expert_count"],
                             as_index=False, observed=True).coverage_gap.mean()
               .sort_values(["expert_count", "active_parameters"]))
    summary = by_size.groupby("expert_count", as_index=False, observed=True).coverage_gap.mean()
    if expected_expert_counts is not None and tuple(summary.expert_count) != tuple(expected_expert_counts):
        raise ValueError("missing expected expert count")
    if reference_expert_count is not None:
        reference = summary.loc[summary.expert_count.eq(reference_expert_count), "coverage_gap"]
        if len(reference) != 1 or reference.iloc[0] == 0:
            raise ValueError("nonzero reference coverage gap required for gap_reduction")
        summary["gap_reduction"] = 1 - summary.coverage_gap / reference.iloc[0]
    return by_size, summary
