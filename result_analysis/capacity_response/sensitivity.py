"""Compare target-exposure coverage gaps across interpolation methods."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
from .tables import matched_pressure


def coverage_gap_sensitivity(frame, bank, cohort, *, expert_counts, methods):
    records, supports = [], []
    for target_k in (50, 200):
        for ladder in "ABCD":
            # A single reference-equivalent token axis is shared across all
            # experts and languages within this panel. Individual D_t is exact.
            lower, upper = [], []
            for e in expert_counts:
                old, new = bank[f"{ladder}/E{e}", 12], bank[f"{ladder}/E{e}", target_k]
                for language in cohort:
                    a, b = old.positions[language], new.positions[language]
                    lo = max(old.exposure[0, a], new.exposure[0, b])
                    hi = min(old.exposure[-1, a], new.exposure[-1, b])
                    lower.append(float(old.tokens_at_exposure(language, lo)))
                    upper.append(float(old.tokens_at_exposure(language, hi)))
            lo, hi = max(lower), min(upper)
            supports.append(dict(target_K=target_k, ladder=ladder, lower=lo, upper=hi))
            grid = np.geomspace(lo, hi, 101)
            for e in expert_counts:
                old, new = bank[f"{ladder}/E{e}", 12], bank[f"{ladder}/E{e}", target_k]
                for method in methods:
                    delta = []
                    for language in cohort:
                        a, b = old.positions[language], new.positions[language]
                        exposure = np.exp(np.interp(np.log(grid), np.log(old.tokens), np.log(old.exposure[:, a])))
                        target_tokens = new.tokens_at_exposure(language, exposure)
                        delta.append(new.at_tokens(target_tokens, method)[:, b]-old.at_tokens(grid, method)[:, a])
                    for tokens, mean in zip(grid, np.mean(delta, axis=0)):
                        records.append(dict(target_K=target_k, ladder=ladder, E=e, method=method,
                                            reference_equivalent_tokens=tokens, pressure=mean))
    points = []
    for target_k in (50, 200):
        anchored = matched_pressure(frame, 12, target_k)
        for row in anchored.itertuples(index=False):
            old = bank[f"{row.ladder}/E{row.E}", 12]
            points.append(dict(target_K=target_k, ladder=row.ladder, E=row.E,
                               D_total=row.D_total, language=row.language, pressure=row.pressure,
                               reference_equivalent_tokens=float(old.tokens_at_exposure(row.language, row.D_t))))
    anchors = pd.DataFrame(points)
    table, support = pd.DataFrame(records), pd.DataFrame(supports)
    standardized = []
    for target_k, group in support.groupby("target_K"):
        lo, hi = group.lower.max(), group.upper.min()
        grid = np.geomspace(lo, hi, 101)
        for (ladder, e, method), curve in table.loc[table.target_K.eq(target_k)].groupby(["ladder", "E", "method"]):
            values = np.interp(np.log(grid), np.log(curve.reference_equivalent_tokens), curve.pressure)
            standardized.append(dict(target_K=target_k, ladder=ladder, E=e, method=method,
                lower_reference_tokens=lo, upper_reference_tokens=hi,
                log_exposure_average=float(np.trapezoid(values, np.log(grid))/np.log(hi/lo))))
    return table, support, anchors, pd.DataFrame(standardized)


def plot_coverage_gap_sensitivity(table, support, anchors, *, expert_colors):
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharey=True, layout="constrained")
    for row, target_k in enumerate((50, 200)):
        limits = support.loc[support.target_K.eq(target_k)]
        common_max, maximum = limits.upper.min()/1e9, limits.upper.max()/1e9
        for column, ladder in enumerate("ABCD"):
            ax = axes[row, column]
            ax.axvspan(common_max, maximum*1.04, color=".95", zorder=0)
            ax.axvline(common_max, color=".65", linestyle=":", linewidth=.8)
            for e, color in expert_colors.items():
                group = table.loc[table.target_K.eq(target_k) & table.ladder.eq(ladder) & table.E.eq(e)]
                grid = group.pivot(index="reference_equivalent_tokens", columns="method", values="pressure")
                ax.fill_between(grid.index/1e9, grid.min(axis=1), grid.max(axis=1), color=color, alpha=.16, linewidth=0)
                ax.plot(grid.index/1e9, grid.smooth, color=color, linewidth=1.7)
                points = anchors.loc[anchors.target_K.eq(target_k) & anchors.ladder.eq(ladder) & anchors.E.eq(e)]
                points = points.groupby("D_total")[["reference_equivalent_tokens", "pressure"]].mean()
                ax.scatter(points.reference_equivalent_tokens/1e9, points.pressure, s=14, color=color,
                           edgecolors="white", linewidth=.3, zorder=3)
            ax.axhline(0, color="black", linewidth=.7)
            ax.set(xscale="log", xlim=(.48, maximum*1.04), title=f"{ladder}: {dict(A=93,B=190,C=349,D=590)[ladder]}M active")
            ticks = [x for x in (.5, 1, 2, 4, 8, 16) if x <= maximum*1.04]
            ax.set_xticks(ticks, labels=[f"{x:g}" for x in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
            if row == 1:
                ax.set_xlabel("K12-equivalent training tokens (B)")
        axes[row, 0].set_ylabel(f"K12 → K{target_k}\nMatched-exposure loss change")
    fig.legend(handles=[Line2D([], [], color=color, label=f"E{e}") for e, color in expert_colors.items()]+[
        Line2D([], [], marker="o", color=".5", linestyle="none", label="Checkpoint-anchored contrasts"),
        Patch(facecolor=".85", label="Band: estimator sensitivity")],
        loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    return fig
