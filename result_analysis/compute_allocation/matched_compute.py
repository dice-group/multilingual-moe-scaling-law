"""Derive observed-support stored-expansion allocation contrasts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_BUDGETS = (2.0, 4.0)
DEFAULT_COVERAGES = (12, 50, 200)
MAX_TOTAL_PARAMETER_MISMATCH = 0.10


def select_allocation_pairs(
    frame: pd.DataFrame,
    maximum_mismatch: float = MAX_TOTAL_PARAMETER_MISMATCH,
) -> pd.DataFrame:
    """Select every loss-blind near-total-parameter pair with different R."""
    configs = (
        frame[["ladder", "E", "N_act", "N_total", "R"]]
        .drop_duplicates()
        .sort_values(["N_total", "N_act", "E"])
        .reset_index(drop=True)
    )
    configs["config"] = configs.ladder.astype(str) + "/E" + configs.E.astype(str)
    rows: list[dict[str, object]] = []
    for left_index, left in configs.iterrows():
        for _, right in configs.iloc[left_index + 1 :].iterrows():
            mismatch = max(left.N_total, right.N_total) / min(left.N_total, right.N_total) - 1
            if (
                mismatch > maximum_mismatch
                or left.N_act == right.N_act
                or np.isclose(left.R, right.R)
            ):
                continue
            higher_r, lower_r = (left, right) if left.R > right.R else (right, left)
            rows.append(
                {
                    "higher_R": higher_r.config,
                    "lower_R": lower_r.config,
                    "higher_R_value": higher_r.R,
                    "lower_R_value": lower_r.R,
                    "higher_R_N_act": higher_r.N_act,
                    "lower_R_N_act": lower_r.N_act,
                    "higher_R_N_total": higher_r.N_total,
                    "lower_R_N_total": lower_r.N_total,
                    "N_total_mismatch_pct": 100 * mismatch,
                }
            )
    pairs = pd.DataFrame(rows).sort_values("N_total_mismatch_pct").reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"P{index}" for index in range(1, len(pairs) + 1)])
    pairs.insert(1, "pair", pairs.higher_R + " vs " + pairs.lower_R)
    return pairs


def interpolate_fixed_compute(
    frame: pd.DataFrame,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    coverages: Sequence[int] = DEFAULT_COVERAGES,
) -> pd.DataFrame:
    """Interpolate empirical common-language loss at fixed logical compute."""
    language_sets = [set(frame.loc[frame.K.eq(coverage), "language"]) for coverage in coverages]
    languages = set.intersection(*language_sets)
    observed = (
        frame.loc[frame.K.isin(coverages) & frame.language.isin(languages)]
        .groupby(
            ["ladder", "E", "K", "D_total", "N_act", "N_total", "R"],
            as_index=False,
            observed=True,
        )
        .agg(
            adjusted_loss=("adjusted_loss", "mean"),
            raw_nll=("raw_nll", "mean"),
            languages=("language", "nunique"),
        )
    )
    observed["config"] = observed.ladder.astype(str) + "/E" + observed.E.astype(str)
    observed["F_EFLOPs"] = (
        6.0 * observed.N_act.astype(float) * observed.D_total.astype(float) / 1e18
    )
    if not observed.languages.eq(len(languages)).all():
        raise ValueError("incomplete common-language allocation curves")

    rows: list[dict[str, object]] = []
    for (coverage, config), curve in observed.groupby(["K", "config"], observed=True):
        curve = curve.sort_values("F_EFLOPs")
        if curve.F_EFLOPs.duplicated().any():
            raise ValueError(f"duplicate compute support for {config}, K={coverage}")
        for budget in budgets:
            if not curve.F_EFLOPs.min() <= budget <= curve.F_EFLOPs.max():
                raise ValueError(f"budget {budget:g} is outside support for {config}, K={coverage}")
            first = curve.iloc[0]
            rows.append(
                {
                    "K": coverage,
                    "config": config,
                    "F_EFLOPs": budget,
                    "N_act": first.N_act,
                    "N_total": first.N_total,
                    "R": first.R,
                    "adjusted_loss": np.interp(
                        np.log(budget),
                        np.log(curve.F_EFLOPs),
                        curve.adjusted_loss,
                    ),
                    "raw_nll": np.interp(
                        np.log(budget),
                        np.log(curve.F_EFLOPs),
                        curve.raw_nll,
                    ),
                }
            )
    fixed = pd.DataFrame(rows)
    expected = observed.config.nunique() * len(coverages) * len(budgets)
    if len(fixed) != expected:
        raise ValueError("incomplete fixed-compute allocation support")
    return fixed.sort_values(["config", "K", "F_EFLOPs"]).reset_index(drop=True)


def allocation_contrasts(
    fixed: pd.DataFrame,
    pairs: pd.DataFrame,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    coverages: Sequence[int] = DEFAULT_COVERAGES,
) -> pd.DataFrame:
    """Compare higher-R and lower-R allocations at fixed compute."""
    index = fixed.set_index(["config", "F_EFLOPs", "K"])
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        for budget in budgets:
            result: dict[str, object] = {
                "pair_id": pair.pair_id,
                "pair": pair.pair,
                "higher_R": pair.higher_R,
                "lower_R": pair.lower_R,
                "N_total_mismatch_pct": pair.N_total_mismatch_pct,
                "F_EFLOPs": budget,
            }
            for coverage in coverages:
                higher = index.loc[(pair.higher_R, budget, coverage)]
                lower = index.loc[(pair.lower_R, budget, coverage)]
                advantage = float(lower.adjusted_loss - higher.adjusted_loss)
                raw_advantage = float(lower.raw_nll - higher.raw_nll)
                if not np.isclose(advantage, raw_advantage, atol=2e-12):
                    raise ValueError("adjusted and raw allocation contrasts disagree")
                result[f"K{coverage}"] = advantage
            result["K200_minus_K12"] = float(result["K200"]) - float(result["K12"])
            rows.append(result)
    contrasts = pd.DataFrame(rows)
    if len(contrasts) != len(pairs) * len(budgets):
        raise ValueError("incomplete allocation contrasts")
    return contrasts


def write_allocation_tables(
    output: Path,
    pairs: pd.DataFrame,
    fixed: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    """Write regenerable allocation tables."""
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(tables / "allocation_pairs.csv", index=False)
    fixed.to_csv(tables / "allocation_fixed_compute.csv", index=False)
    contrasts.to_csv(tables / "allocation_contrasts.csv", index=False)


def isoflop_coverage_cost(
    frame: pd.DataFrame,
    budgets: Sequence[float] = (2.0, 4.0),
) -> pd.DataFrame:
    """Interpolate common-language K50/K200 loss differences at fixed FLOPs."""
    languages = set(frame.loc[frame.K.eq(12), "language"])
    observed = (
        frame.loc[frame.language.isin(languages) & frame.K.isin((12, 50, 200))]
        .groupby(["ladder", "E", "K", "D_total", "N_act"], as_index=False, observed=True)
        .adjusted_loss.mean()
    )
    observed["F_EFLOPs"] = (
        6.0 * observed.N_act.astype(float) * observed.D_total.astype(float) / 1e18
    )
    rows: list[dict[str, object]] = []
    for (ladder, expert, coverage, n_act), curve in observed.groupby(
        ["ladder", "E", "K", "N_act"], observed=True
    ):
        curve = curve.sort_values("F_EFLOPs")
        for budget in budgets:
            if curve.F_EFLOPs.min() <= budget <= curve.F_EFLOPs.max():
                rows.append(
                    {
                        "ladder": ladder,
                        "N_act": n_act,
                        "E": expert,
                        "K": coverage,
                        "F_EFLOPs": budget,
                        "loss": np.interp(
                            np.log(budget),
                            np.log(curve.F_EFLOPs),
                            curve.adjusted_loss,
                        ),
                    }
                )
    fixed = pd.DataFrame(rows)
    config_count = observed[["ladder", "E", "N_act"]].drop_duplicates().shape[0]
    pivot = fixed.pivot(
        index=["ladder", "N_act", "E", "F_EFLOPs"],
        columns="K",
        values="loss",
    )
    if len(pivot) != config_count * len(budgets) or pivot[[12, 50, 200]].isna().any().any():
        raise ValueError("incomplete shared IsoFLOP support")
    costs = pivot.assign(K50=lambda x: x[50] - x[12], K200=lambda x: x[200] - x[12])
    return (
        costs[["K50", "K200"]]
        .rename_axis(columns="comparison")
        .stack()
        .rename("cost")
        .reset_index()
    )
