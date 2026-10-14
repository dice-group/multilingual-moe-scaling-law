"""Publication-style figures for the Stage 1 pressure analyses."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


PALETTE = plt.colormaps["inferno"]
MARKERS = ("o", "s", "^", "D", "P", "X")
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.14}
K_COLORS = {12: "#4B0D78", 200: "#FF8A61"}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def _ladder_style(frame: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Derive colors, markers, and active-size labels from the data."""
    sizes = frame.groupby("ladder", observed=True).N_act.median().sort_values()
    positions = np.linspace(0.18, 0.78, len(sizes))
    colors = {
        ladder: matplotlib.colors.to_hex(color)
        for ladder, color in zip(sizes.index, PALETTE(positions))
    }
    markers = {ladder: marker for ladder, marker in zip(sizes.index, cycle(MARKERS))}
    labels = {ladder: f"{size / 1e6:.0f}M active" for ladder, size in sizes.items()}
    return colors, markers, labels


def _style(axis: plt.Axes) -> None:
    axis.axhline(0, color="#555555", lw=0.8, ls="--")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.minorticks_off()


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(figures / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def _draw_primary(axis: plt.Axes, by_ladder: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors, markers, labels = _ladder_style(by_ladder)
    for ladder, rows in by_ladder.groupby("ladder", observed=True):
        rows = rows.sort_values("E")
        axis.plot(
            rows.E,
            rows.pressure,
            marker=markers[ladder],
            color=colors[ladder],
            alpha=0.72,
            lw=1.35,
            ms=4,
            label=labels[ladder],
            zorder=1,
        )
    axis.plot(
        summary.E,
        summary.pressure,
        "o-",
        color="#202020",
        lw=2.7,
        ms=6,
        label="equal-size mean",
        zorder=3,
    )
    for row in summary.itertuples(index=False):
        axis.annotate(
            f"{row.pressure:.4f}",
            (row.E, row.pressure),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            bbox=LABEL_BOX,
        )
    pressure = summary.set_index("E").pressure
    attenuation = 1 - pressure.loc[32] / pressure.loc[1]
    axis.text(
        0.64,
        0.94,
        f"{pressure.loc[1]:.4f} → {pressure.loc[32]:.4f}\n"
        f"{attenuation:.1%} smaller",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=LABEL_BOX,
    )
    axis.set(
        xscale="log",
        xticks=summary.E,
        xticklabels=[f"E={expert}" for expert in summary.E],
        ylabel="Matched-$D_t$ loss difference\n($K=200 - K=12$)",
    )
    axis.tick_params(axis="x", labelsize=8.5)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=7.8,
    )


def _draw_seed(axis: plt.Axes, seed: pd.DataFrame) -> None:
    experts = (1, 16, 32)
    positions = np.arange(len(experts))
    colors = PALETTE(np.linspace(0.28, 0.82, len(seed)))
    for color, marker, linestyle, row in zip(
        colors,
        MARKERS,
        ("-", "--", "-."),
        seed.itertuples(index=False),
    ):
        values = [getattr(row, f"E{expert}_pressure") for expert in experts]
        axis.plot(
            positions,
            values,
            color=color,
            marker=marker,
            ls=linestyle,
            lw=2,
            ms=5,
            label=f"seed {row.seed}",
        )
    minimum, maximum = seed.relative_attenuation.min(), seed.relative_attenuation.max()
    axis.text(
        0.04,
        0.32,
        f"{len(seed)}/{len(seed)} seeds show reduction\n{minimum:.1%}–{maximum:.1%}",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=LABEL_BOX,
    )
    axis.set(
        xticks=positions,
        xticklabels=[f"E={expert}" for expert in experts],
        ylabel="Matched-$D_t$ loss difference",
    )
    axis.tick_params(axis="x", labelsize=8.5)
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def save_primary_figures(
    output: Path,
    by_ladder: pd.DataFrame,
    summary: pd.DataFrame,
    seed: pd.DataFrame,
) -> None:
    """Save a combined view and title-free panels for LaTeX."""
    figure, (primary, replication) = plt.subplots(
        1,
        2,
        figsize=(10.6, 3.8),
        gridspec_kw={"width_ratios": [1.25, 1]},
    )
    _draw_primary(primary, by_ladder, summary)
    _draw_seed(replication, seed)
    for axis in (primary, replication):
        _style(axis)
    figure.tight_layout(rect=[0, 0.10, 1, 1])
    _save(figure, output, "matched_exposure_pressure")

    for stem, draw, arguments, rect, size in (
        (
            "matched_exposure_primary",
            _draw_primary,
            (by_ladder, summary),
            [0, 0.14, 1, 1],
            (4.0, 2.75),
        ),
        ("matched_exposure_seed", _draw_seed, (seed,), [0, 0, 1, 1], (2.75, 2.75)),
    ):
        figure, axis = plt.subplots(figsize=size)
        draw(axis, *arguments)
        _style(axis)
        figure.tight_layout(rect=rect)
        _save(figure, output, stem)


def save_coverage_contrasts(
    output: Path,
    contrasts: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Save all matched-exposure coverage contrasts for the appendix."""
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 5.4), sharex=True)
    all_ladders = pd.concat([by_ladder for by_ladder, _ in contrasts.values()])
    colors, markers, labels = _ladder_style(all_ladders)
    for axis, ((reference_k, target_k), (by_ladder, summary)) in zip(axes.flat, contrasts.items()):
        for ladder, rows in by_ladder.groupby("ladder", observed=True):
            rows = rows.sort_values("E")
            axis.plot(
                rows.E,
                rows.pressure,
                marker=markers[ladder],
                color=colors[ladder],
                alpha=0.70,
                lw=1.15,
                ms=3.5,
            )
        axis.plot(summary.E, summary.pressure, "o-", color="#202020", lw=2.2, ms=5)
        _style(axis)
        axis.set(
            xscale="log",
            xticks=summary.E,
            xticklabels=[f"E={expert}" for expert in summary.E],
            title=rf"$K={reference_k}\rightarrow K={target_k}$",
            ylabel="Matched-$D_t$ loss difference",
        )
        axis.tick_params(axis="x", labelsize=8)
    handles = [
        *(
            Line2D(
                [],
                [],
                color=colors[ladder],
                marker=markers[ladder],
                lw=1.15,
                markersize=4,
                label=labels[ladder],
            )
            for ladder in labels
        ),
        Line2D([], [], color="#202020", marker="o", lw=2.2, markersize=5, label="equal-size mean"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    _save(figure, output, "matched_exposure_coverage_contrasts")


def save_expert_gain_by_coverage(
    output: Path,
    multilingual: pd.DataFrame,
    english: pd.DataFrame,
) -> None:
    """Plot descriptive E1-to-E32 gains for fixed language cohorts."""
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    colors, markers, labels = _ladder_style(pd.concat([multilingual, english]))
    for axis, gains, title in (
        (axes[0], multilingual, "Shared 12-language cohort"),
        (axes[1], english, "English across all coverage tiers"),
    ):
        by_ladder = gains.groupby(
            ["ladder", "N_act", "K"], as_index=False, observed=True
        ).gain.mean()
        mean = by_ladder.groupby("K", as_index=False, observed=True).gain.mean()
        for ladder, rows in by_ladder.groupby("ladder", observed=True):
            rows = rows.sort_values("K")
            axis.plot(
                rows.K,
                rows.gain,
                marker=markers[ladder],
                color=colors[ladder],
                alpha=0.72,
                lw=1.3,
                ms=4,
                label=labels[ladder],
            )
        axis.plot(mean.K, mean.gain, "o-", color="#202020", lw=2.7, ms=6, label="equal-size mean")
        axis.set(
            xscale="log",
            xticks=mean.K,
            xticklabels=[f"K={coverage}" for coverage in mean.K],
            ylabel="Mean loss reduction ($E=1 - E=32$)",
            title=title,
        )
        _style(axis)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    figure.tight_layout()
    _save(figure, output, "expert_gain_by_coverage")


def save_isoflop_coverage_cost(output: Path, costs: pd.DataFrame) -> None:
    """Plot fixed-logical-compute K50/K200 differences over expert count."""
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    colors, markers, labels = _ladder_style(costs)
    linestyles = {"K50": "--", "K200": "-"}
    for axis, (budget, panel) in zip(axes, costs.groupby("F_EFLOPs", observed=True)):
        for comparison, rows in panel.groupby("comparison", observed=True):
            for ladder, curve in rows.groupby("ladder", observed=True):
                curve = curve.sort_values("E")
                axis.plot(
                    curve.E,
                    curve.cost,
                    linestyles[comparison],
                    marker=markers[ladder],
                    color=colors[ladder],
                    alpha=0.68,
                    lw=1.25,
                    ms=4,
                )
            mean = rows.groupby("E", as_index=False, observed=True).cost.mean().sort_values("E")
            axis.plot(
                mean.E,
                mean.cost,
                linestyles[comparison],
                marker="o",
                color="#202020",
                lw=2.6,
                ms=5.5,
                label=f"{comparison.replace('K', 'K=')} - K=12 mean",
            )
        axis.set(
            xscale="log",
            xticks=(1, 4, 16, 32),
            xticklabels=("E=1", "E=4", "E=16", "E=32"),
            title=f"{budget:g} logical EFLOPs",
            ylabel="Fixed-compute loss difference",
        )
        _style(axis)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7.5)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=colors[ladder],
                marker=markers[ladder],
                lw=1.25,
                markersize=4,
                label=labels[ladder],
            )
            for ladder in labels
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    _save(figure, output, "isoflop_coverage_cost")
