"""Compact publication figures from canonical benchmark visualization tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    BenchmarkAnalysisError,
)
from result_analysis.downstream_evaluations.campaign.visualization_data import (
    EXPERTS,
    LADDERS,
    LANGUAGE_COUNTS,
)
from result_analysis.downstream_evaluations.campaign.visualization_plots import (
    configure_matplotlib,
    save_figure,
)


TASKS = (
    "xstorycloze",
    "uhura_arc_easy",
    "okapi_hellaswag",
    "belebele_cloze",
    "xcopa",
    "global_mmlu_cloze",
)
TASK_LABELS = {
    "xstorycloze": "XStoryCloze",
    "uhura_arc_easy": "ARC Easy",
    "okapi_hellaswag": "HellaSwag",
    "belebele_cloze": "Belebele",
    "xcopa": "XCOPA",
    "global_mmlu_cloze": "Global MMLU",
}
PAPER_FIGURE_FILES = {
    "task_signal": "benchmark_task_signal_and_learning",
    "b_seed_task_curves": "benchmark_b_seed_task_compute_curves",
    "full_grid_compute_curves": "benchmark_full_grid_compute_curves",
    "composite_slope_map": "benchmark_composite_compute_slope_map",
}


def _frame(rows: Sequence[Mapping[str, Any]]) -> Any:
    import pandas as pd

    return pd.DataFrame([dict(row) for row in rows])


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _palette(plt: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    palette = plt.colormaps["inferno"]
    coverage_colors = dict(zip(LANGUAGE_COUNTS, palette([0.15, 0.36, 0.58, 0.80])))
    expert_colors = dict(zip(EXPERTS, palette([0.10, 0.34, 0.58, 0.82])))
    return coverage_colors, expert_colors


def plot_task_signal(
    endpoint_rows: Sequence[Mapping[str, Any]],
    task_slope_rows: Sequence[Mapping[str, Any]],
    *,
    figure_root: Path,
    stem: str,
) -> dict[str, Any]:
    """Show final signal relative to chance and task-wise compute slopes."""

    plt, np, TwoSlopeNorm, _line2d, _patch = configure_matplotlib()
    _style(plt)
    endpoints = _frame(endpoint_rows)
    slopes = _frame(task_slope_rows)
    coverage_colors, _expert_colors = _palette(plt)
    fig, (signal, learning) = plt.subplots(
        1, 2, figsize=(7.05, 2.95), gridspec_kw={"width_ratios": [1.18, 1]}
    )
    positions = np.arange(len(TASKS))
    offsets = dict(zip(LANGUAGE_COUNTS, (-0.24, -0.08, 0.08, 0.24)))
    markers = dict(zip(LANGUAGE_COUNTS, ("o", "s", "^", "D")))
    for languages in LANGUAGE_COUNTS:
        rows = (
            endpoints.loc[endpoints.K.eq(languages)]
            .set_index("benchmark")
            .loc[list(TASKS)]
        )
        y_values = positions + offsets[languages]
        signal.hlines(
            y_values,
            rows.minimum,
            rows.maximum,
            color=coverage_colors[languages],
            linewidth=1.0,
            alpha=0.48,
        )
        signal.scatter(
            rows["mean"],
            y_values,
            color=coverage_colors[languages],
            marker=markers[languages],
            s=31,
            edgecolor="white",
            linewidth=0.55,
            label=f"K={languages}{' (English)' if languages == 1 else ''}",
            zorder=3,
        )
    chance = (
        endpoints.drop_duplicates("benchmark")
        .set_index("benchmark")
        .loc[list(TASKS)]
    )
    labels = [
        f"{TASK_LABELS[task]}  "
        f"({100 * chance.loc[task, 'random_choice_accuracy']:.0f}% chance)"
        for task in TASKS
    ]
    signal.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
    signal.set_yticks(positions, labels)
    signal.invert_yaxis()
    signal.set_xlabel("Final accuracy above random choice (pp)")
    signal.set_title("Final task signal across the grid", fontweight="bold")
    signal.grid(axis="x", color="#DDDDDD", linewidth=0.6, alpha=0.65)
    signal.legend(loc="lower right", frameon=False, ncol=2)

    matrix = np.array(
        [
            [
                slopes.loc[
                    slopes.K.eq(languages) & slopes.benchmark.eq(task), "median"
                ].iloc[0]
                for languages in LANGUAGE_COUNTS
            ]
            for task in TASKS
        ]
    )
    bound = max(abs(matrix.min()), abs(matrix.max()), 0.25)
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list(
        "paper_diverging",
        ("#355F8D", "#F4F1EC", "#9C2F7F", "#F47A48", "#F6C65B"),
    )
    image = learning.imshow(
        matrix,
        aspect="auto",
        cmap=colormap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    learning.set_xticks(
        range(4),
        ["K=1\nEnglish", "K=12\nMulti.", "K=50\nMulti.", "K=200\nMulti."],
    )
    learning.set_yticks(range(6), [TASK_LABELS[task] for task in TASKS])
    learning.set_title("Median gain per compute doubling", fontweight="bold")
    learning.tick_params(length=0)
    for row_index, task in enumerate(TASKS):
        for column_index, languages in enumerate(LANGUAGE_COUNTS):
            cell = slopes.loc[
                slopes.K.eq(languages) & slopes.benchmark.eq(task)
            ].iloc[0]
            value = float(cell["median"])
            color = "white" if abs(value) > 0.55 * bound else "#202020"
            learning.text(
                column_index,
                row_index,
                f"{value:+.2f}\n{int(cell.positive)}/{int(cell.n)} ↑",
                ha="center",
                va="center",
                fontsize=5.9,
                color=color,
            )
    colorbar = fig.colorbar(
        image, ax=learning, orientation="horizontal", fraction=0.07, pad=0.16
    )
    colorbar.set_label("Accuracy gain (pp per logical-compute doubling)")
    fig.text(
        0.5,
        -0.005,
        "Left: mean and min–max over 16 canonical final configurations per K. "
        "Right: median over the same 16 trajectories; arrows report positive "
        "trajectories. K=1 is English-only; K≥12 is multilingual. "
        r"Logical compute = $6N_{\mathrm{act}}D$.",
        ha="center",
        va="top",
        fontsize=6.1,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    references = save_figure(fig, figure_root, stem)
    plt.close(fig)
    return references


def plot_b_seed_task_curves(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    figure_root: Path,
    stem: str,
) -> dict[str, Any]:
    """Show the bounded B three-seed task learning curves."""

    plt, _np, _norm, Line2D, _patch = configure_matplotlib()
    _style(plt)
    curves = _frame(curve_rows)
    coverage_colors, _expert_colors = _palette(plt)
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.35))
    styles = {1: (0, (3, 2)), 32: "-"}
    for axis, task in zip(axes.flat, TASKS):
        panel = curves.loc[curves.benchmark.eq(task)]
        for languages in (1, 12, 200):
            for experts in (1, 32):
                line = panel.loc[
                    panel.K.eq(languages) & panel.E.eq(experts)
                ].sort_values("compute_eflops")
                axis.plot(
                    line.compute_eflops,
                    line["mean"],
                    color=coverage_colors[languages],
                    linestyle=styles[experts],
                    linewidth=1.75 if experts == 32 else 1.25,
                    marker="o" if experts == 32 else None,
                    markersize=3.0,
                    alpha=0.92,
                )
                axis.fill_between(
                    line.compute_eflops,
                    line.minimum,
                    line.maximum,
                    color=coverage_colors[languages],
                    alpha=0.08,
                    linewidth=0,
                )
        axis.axhline(0, color="#555555", linewidth=0.75, linestyle="--")
        axis.set_xscale("log")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55, alpha=0.65)
        axis.set_title(TASK_LABELS[task], fontweight="bold")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy above chance (pp)")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Logical compute $6N_{\mathrm{act}}D$ (EFLOPs)")
    coverage_handles = [
        Line2D(
            [],
            [],
            color=coverage_colors[languages],
            linewidth=2,
            label=f"K={languages} "
            f"({'English' if languages == 1 else 'multilingual'})",
        )
        for languages in (1, 12, 200)
    ]
    expert_handles = [
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=styles[experts],
            linewidth=1.6,
            marker="o" if experts == 32 else None,
            markersize=3,
            label=f"E={experts}",
        )
        for experts in (1, 32)
    ]
    fig.legend(
        handles=coverage_handles + expert_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.064),
    )
    fig.text(
        0.5,
        0.018,
        "B active size (190M), three-seed means with observed min–max ribbons. "
        "Logical compute excludes routing, communication, and other system overhead.",
        ha="center",
        fontsize=6.2,
    )
    fig.tight_layout(rect=(0, 0.145, 1, 1))
    references = save_figure(fig, figure_root, stem)
    plt.close(fig)
    return references


def plot_full_grid_compute_curves(
    plot_rows: Sequence[Mapping[str, Any]],
    *,
    figure_root: Path,
    stem: str,
) -> dict[str, Any]:
    """Plot all empirical K/E/active-size trajectories compactly."""

    plt, _np, _norm, Line2D, _patch = configure_matplotlib()
    _style(plt)
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.8,
        }
    )
    curves = _frame(plot_rows).rename(
        columns={"experts": "E", "languages": "K"}
    )
    coverage_colors, expert_colors = _palette(plt)
    del coverage_colors
    line_styles = dict(zip(EXPERTS, ("-", "--", ":", "-.")))
    markers = dict(zip(LADDERS, ("o", "s", "^", "D")))
    labels = (
        curves.groupby("ladder", observed=True)
        .active_parameters.mean()
        .round(-6)
        .astype(int)
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.7), sharey=False)
    for axis, languages in zip(axes, LANGUAGE_COUNTS):
        panel = curves.loc[curves.K.eq(languages)]
        for ladder in LADDERS:
            for experts in EXPERTS:
                line = panel.loc[
                    panel.ladder.eq(ladder) & panel.E.eq(experts)
                ].sort_values("logical_compute_eflops")
                axis.plot(
                    line.logical_compute_eflops,
                    line.chance_adjusted_change_from_0p5b_pp_mean,
                    color=expert_colors[experts],
                    marker=markers[ladder],
                    linestyle=line_styles[experts],
                    markersize=2.7,
                    linewidth=1.15,
                    alpha=0.72,
                )
                if line.n_seeds.max() == 3:
                    axis.fill_between(
                        line.logical_compute_eflops,
                        line.chance_adjusted_change_from_0p5b_pp_min,
                        line.chance_adjusted_change_from_0p5b_pp_max,
                        color=expert_colors[experts],
                        alpha=0.07,
                        linewidth=0,
                    )
        axis.axhline(0, color="#555555", linewidth=0.75, linestyle="--")
        axis.set_xscale("log")
        suffix = "English" if languages == 1 else "multilingual"
        axis.set_title(f"K={languages}  ({suffix})", fontweight="bold")
        axis.set_xlabel(r"$6N_{\mathrm{act}}D$ (EFLOPs)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55, alpha=0.65)
    axes[0].set_ylabel("Chance-adjusted score change\nfrom the 0.5B checkpoint (pp)")
    expert_handles = [
        Line2D(
            [],
            [],
            color=expert_colors[experts],
            linestyle=line_styles[experts],
            linewidth=1.8,
            label=f"E={experts}",
        )
        for experts in EXPERTS
    ]
    size_handles = [
        Line2D(
            [],
            [],
            color="#444444",
            marker=markers[ladder],
            linewidth=0,
            markersize=4,
            label=f"{labels[ladder] / 1e6:.0f}M active",
        )
        for ladder in LADDERS
    ]
    fig.legend(
        handles=expert_handles + size_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.6,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.tight_layout(rect=(0, 0.235, 1, 1))
    references = save_figure(fig, figure_root, stem)
    plt.close(fig)
    return references


def plot_composite_slope_map(
    slope_rows: Sequence[Mapping[str, Any]],
    plot_rows: Sequence[Mapping[str, Any]],
    *,
    figure_root: Path,
    stem: str,
) -> dict[str, Any]:
    """Show chance-adjusted compute slopes over active size, E, and K."""

    plt, np, TwoSlopeNorm, _line2d, _patch = configure_matplotlib()
    _style(plt)
    slopes = _frame(slope_rows).rename(
        columns={"experts": "E", "languages": "K"}
    )
    curves = _frame(plot_rows)
    labels = (
        curves.groupby("ladder", observed=True)
        .active_parameters.mean()
        .round(-6)
        .astype(int)
    )
    value_field = "mean_chance_adjusted_slope_pp_per_actual_token_doubling"
    values = slopes[value_field].to_numpy(float)
    bound = max(abs(values.min()), abs(values.max()), 0.2)
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list(
        "paper_diverging",
        ("#355F8D", "#F4F1EC", "#9C2F7F", "#F47A48", "#F6C65B"),
    )
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.75))
    image = None
    for axis, languages in zip(axes, LANGUAGE_COUNTS):
        matrix = np.zeros((4, 4))
        counts = np.zeros((4, 4), dtype=int)
        for row_index, ladder in enumerate(LADDERS):
            for column_index, experts in enumerate(EXPERTS):
                cell = slopes.loc[
                    slopes.K.eq(languages)
                    & slopes.ladder.eq(ladder)
                    & slopes.E.eq(experts)
                ].iloc[0]
                matrix[row_index, column_index] = cell[value_field]
                counts[row_index, column_index] = cell.n_trajectories
        image = axis.imshow(matrix, aspect="auto", cmap=colormap, norm=norm)
        suffix = "English" if languages == 1 else "multilingual"
        axis.set_title(f"K={languages}  ({suffix})", fontweight="bold")
        axis.set_xticks(range(4), [f"E={experts}" for experts in EXPERTS])
        if languages == 1:
            axis.set_yticks(
                range(4), [f"{labels[ladder] / 1e6:.0f}M" for ladder in LADDERS]
            )
            axis.set_ylabel("Active parameters")
        else:
            axis.set_yticks(range(4), [])
        axis.tick_params(length=0)
        for row_index in range(4):
            for column_index in range(4):
                value = matrix[row_index, column_index]
                color = "white" if abs(value) > 0.55 * bound else "#202020"
                suffix = "\n3 seeds" if counts[row_index, column_index] == 3 else ""
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.2f}{suffix}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=color,
                )
    if image is None:
        raise BenchmarkAnalysisError("empty composite slope map")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.34, wspace=0.18)
    color_axis = fig.add_axes((0.35, 0.175, 0.30, 0.035))
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        "Chance-adjusted benchmark gain (pp per logical-compute doubling)"
    )
    fig.text(
        0.5,
        0.025,
        "Each cell is an OLS slope over checkpoints within one trajectory; "
        "replicated B cells show the mean of three seed-wise slopes. K=1 is "
        "English-only; K≥12 is multilingual. "
        r"Logical compute = $6N_{\mathrm{act}}D$ and excludes system overhead.",
        ha="center",
        fontsize=5.8,
    )
    references = save_figure(fig, figure_root, stem)
    plt.close(fig)
    return references


__all__ = [
    "PAPER_FIGURE_FILES",
    "plot_b_seed_task_curves",
    "plot_composite_slope_map",
    "plot_full_grid_compute_curves",
    "plot_task_signal",
]
