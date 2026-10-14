"""Audit-oriented plots for the sealed Stage-1 benchmark analysis."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from result_analysis.downstream_evaluations.campaign.analysis_artifacts import (
    BenchmarkAnalysisError,
    _artifact_reference,
)
from result_analysis.downstream_evaluations.campaign.visualization_data import (
    EXPECTED_READOUTS_BY_LADDER,
    EXPERTS,
    LADDERS,
    LANGUAGE_COUNTS,
)


EXPERT_COLORS = {
    1: "#4D4D4D",
    4: "#0072B2",
    16: "#009E73",
    32: "#D55E00",
}
TOKEN_LABELS = {
    500_000_000: "0.5",
    1_000_000_000: "1",
    2_000_000_000: "2",
    4_000_000_000: "4",
    8_000_000_000: "8",
    16_000_000_000: "16",
    32_000_000_000: "32",
}


def configure_matplotlib() -> tuple[Any, Any, Any, Any, Any]:
    """Configure a deterministic headless Matplotlib process on first use."""

    config_root = Path(tempfile.gettempdir()) / "multilingual-megatron-matplotlib"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_root))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 260,
        }
    )
    return plt, np, TwoSlopeNorm, Line2D, Patch


def save_figure(fig: Any, figure_root: Path, stem: str) -> dict[str, Any]:
    """Save deterministic PDF/PNG variants and return artifact references."""

    figure_root.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_root / f"{stem}.pdf"
    png_path = figure_root / f"{stem}.png"
    metadata = {
        "Title": stem.replace("_", " "),
        "Author": "Stage-1 benchmark visualization generator",
        "Creator": "stage1_benchmarking.visualization_plots",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(
        png_path,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib"},
    )
    pdf_reference = _artifact_reference(pdf_path)
    pdf_reference["bytes"] = pdf_path.stat().st_size
    png_reference = _artifact_reference(png_path)
    png_reference["bytes"] = png_path.stat().st_size
    return {"pdf": pdf_reference, "png": png_reference}


def _row_limits(
    plot_rows: Sequence[Mapping[str, Any]],
    *,
    languages: int,
    field_prefix: str,
) -> tuple[float, float]:
    selected = [row for row in plot_rows if int(row["languages"]) == languages]
    values = [
        100.0 * float(row[f"{field_prefix}_{suffix}"])
        for row in selected
        for suffix in ("min", "max")
    ]
    values.append(100.0 / 3.0 if field_prefix == "primary_score" else 0.0)
    lower, upper = min(values), max(values)
    padding = max(0.45, (upper - lower) * 0.07)
    return lower - padding, upper + padding


def plot_scaling_grid(
    plot_rows: Sequence[Mapping[str, Any]],
    *,
    figure_root: Path,
    stem: str,
    field_prefix: str,
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    """Render the complete K-by-active-size scaling grid."""

    plt, _np, _two_slope_norm, Line2D, Patch = configure_matplotlib()
    fig, axes = plt.subplots(4, 4, figsize=(13.3, 10.4), squeeze=False)
    limits = {
        languages: _row_limits(
            plot_rows, languages=languages, field_prefix=field_prefix
        )
        for languages in LANGUAGE_COUNTS
    }
    for row_index, languages in enumerate(LANGUAGE_COUNTS):
        for column_index, ladder in enumerate(LADDERS):
            axis = axes[row_index][column_index]
            cell = [
                row
                for row in plot_rows
                if row["ladder"] == ladder and int(row["languages"]) == languages
            ]
            for experts in EXPERTS:
                series = sorted(
                    (row for row in cell if int(row["experts"]) == experts),
                    key=lambda row: int(row["nominal_tokens"]),
                )
                if len(series) != EXPECTED_READOUTS_BY_LADDER[ladder]:
                    raise BenchmarkAnalysisError(
                        f"plot series drift for {(ladder, experts, languages)}"
                    )
                x_values = [
                    math.log2(float(row["nominal_tokens"]) / 500_000_000.0)
                    for row in series
                ]
                means = [
                    100.0 * float(row[f"{field_prefix}_mean"]) for row in series
                ]
                color = EXPERT_COLORS[experts]
                axis.plot(
                    x_values,
                    means,
                    color=color,
                    marker="o",
                    markersize=3.2,
                    markeredgewidth=0.0,
                    zorder=3,
                )
                if int(series[0]["n_seeds"]) == 3:
                    lower = [
                        100.0 * float(row[f"{field_prefix}_min"]) for row in series
                    ]
                    upper = [
                        100.0 * float(row[f"{field_prefix}_max"]) for row in series
                    ]
                    axis.fill_between(
                        x_values,
                        lower,
                        upper,
                        color=color,
                        alpha=0.16,
                        linewidth=0.0,
                        zorder=1,
                    )
            baseline = 100.0 / 3.0 if field_prefix == "primary_score" else 0.0
            axis.axhline(baseline, color="0.55", linestyle=":", linewidth=0.9)
            available_tokens = sorted({int(row["nominal_tokens"]) for row in cell})
            x_ticks = [
                math.log2(float(tokens) / 500_000_000.0)
                for tokens in available_tokens
            ]
            axis.set_xticks(
                x_ticks, [TOKEN_LABELS[tokens] for tokens in available_tokens]
            )
            axis.set_xlim(-0.22, 6.22)
            axis.set_ylim(*limits[languages])
            axis.grid(axis="y", color="0.90", linewidth=0.6)
            if row_index == 0:
                axis.set_title(f"Ladder {ladder}", fontweight="bold", pad=5)
            if column_index == 0:
                axis.set_ylabel(f"K={languages}\n{ylabel}")
            if row_index == len(LANGUAGE_COUNTS) - 1:
                axis.set_xlabel("Nominal tokens (billions)")
    handles = [
        Line2D(
            [0],
            [0],
            color=EXPERT_COLORS[experts],
            marker="o",
            markersize=3.8,
            label=f"E={experts}",
        )
        for experts in EXPERTS
    ]
    handles.append(
        Patch(
            facecolor="0.5",
            alpha=0.16,
            edgecolor="none",
            label="B replicated cells: 3-seed min-max",
        )
    )
    fig.suptitle(title, fontsize=12.2, fontweight="bold", y=0.992)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
    )
    baseline_text = (
        "Dotted line: task-equal random-choice baseline (33.33%). "
        if field_prefix == "primary_score"
        else "Dotted line: each trajectory's 0.5B baseline. "
    )
    fig.text(
        0.5,
        0.012,
        baseline_text
        + "B/E={1,16,32}/K={1,12,200}: mean and observed range over 3 seeds; "
        "all other curves: one seed.\n"
        "K1 and K>=12 use different six-task request sets; compare within K. "
        "Rows use K-specific y-ranges.",
        ha="center",
        va="bottom",
        fontsize=7.6,
    )
    fig.subplots_adjust(
        left=0.073,
        right=0.992,
        bottom=0.075,
        top=0.915,
        hspace=0.34,
        wspace=0.24,
    )
    references = save_figure(fig, figure_root, stem)
    plt.close(fig)
    return references


def plot_slope_heatmap(
    slope_rows: Sequence[Mapping[str, Any]],
    *,
    figure_root: Path,
    stem: str,
) -> dict[str, Any]:
    """Render the audit-oriented raw-score slope heatmap."""

    plt, np, TwoSlopeNorm, _line2d, _patch = configure_matplotlib()
    values = [
        float(row["mean_slope_pp_per_actual_token_doubling"]) for row in slope_rows
    ]
    magnitude = max(abs(min(values)), abs(max(values)), 0.25)
    norm = TwoSlopeNorm(vmin=-magnitude, vcenter=0.0, vmax=magnitude)
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.7), squeeze=False)
    image = None
    for panel_index, languages in enumerate(LANGUAGE_COUNTS):
        axis = axes[0][panel_index]
        matrix = np.zeros((len(LADDERS), len(EXPERTS)), dtype=float)
        counts = np.zeros_like(matrix, dtype=int)
        for row_index, ladder in enumerate(LADDERS):
            for column_index, experts in enumerate(EXPERTS):
                matches = [
                    row
                    for row in slope_rows
                    if row["ladder"] == ladder
                    and int(row["experts"]) == experts
                    and int(row["languages"]) == languages
                ]
                if len(matches) != 1:
                    raise BenchmarkAnalysisError(
                        f"slope heatmap cell drift for {(ladder, experts, languages)}"
                    )
                matrix[row_index, column_index] = float(
                    matches[0]["mean_slope_pp_per_actual_token_doubling"]
                )
                counts[row_index, column_index] = int(
                    matches[0]["n_trajectories"]
                )
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_title(f"K={languages}", fontweight="bold")
        axis.set_xticks(range(len(EXPERTS)), [f"E{experts}" for experts in EXPERTS])
        if panel_index == 0:
            axis.set_yticks(range(len(LADDERS)), [f"Ladder {item}" for item in LADDERS])
        else:
            axis.set_yticks(range(len(LADDERS)), [])
        for row_index in range(len(LADDERS)):
            for column_index in range(len(EXPERTS)):
                value = matrix[row_index, column_index]
                count = counts[row_index, column_index]
                color = "white" if abs(value) > magnitude * 0.52 else "black"
                label = f"{value:+.2f}" + ("\n3 seeds" if count == 3 else "")
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7.5,
                )
        axis.tick_params(length=0)
    if image is None:
        raise BenchmarkAnalysisError("slope heatmap produced no image")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.29, top=0.81, wspace=0.18)
    colorbar_axis = fig.add_axes((0.34, 0.18, 0.32, 0.035))
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Mean score slope (pp per actual-token doubling)")
    fig.suptitle(
        "Benchmark learning-curve slopes across the Stage-1 grid",
        fontsize=11.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.018,
        "OLS slopes are descriptive within-trajectory summaries. B replicated cells "
        "show the mean of 3 independent-seed slopes; all others are one seed.\n"
        "K1 and K>=12 use different six-task request sets; slope magnitude is not a "
        "cross-K estimand.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    references = save_figure(fig, figure_root, stem)
    plt.close(fig)
    return references


__all__ = [
    "configure_matplotlib",
    "plot_scaling_grid",
    "plot_slope_heatmap",
    "save_figure",
]
