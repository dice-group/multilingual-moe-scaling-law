"""Standalone bottom-right checkpoint panel with data-fitted axes and trunk lines."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterSciNotation, MultipleLocator, NullFormatter
import numpy as np

from result_analysis.compute_allocation.review.compute_projection import (
    COLORS, MARKERS, STYLE, checkpoint_means, legend,
)
from result_analysis.capacity_response.tables import (
    load_primary, RELEASE_MANIFEST_SHA256, sha256_file,
)


def size_panels(means, output, paper=False, inferno=False):
    """Separate active sizes without aggregating or dropping trajectories."""
    colors = dict(zip(COLORS, matplotlib.colormaps["inferno"]([.02, .32, .62, .82]))) if inferno else COLORS
    fig, axes = plt.subplots(1, 4, figsize=(16 / 2.54, 2.5) if paper else (15, 4.8), sharex=True, sharey=True)
    xmin, xmax = means.F_total_flops.agg(["min", "max"])
    ymin, ymax = means.mean_adjusted_loss.agg(["min", "max"])
    counts = {}
    for ax, (ladder, rows) in zip(axes.flat, means.groupby("ladder", sort=True)):
        assert rows.N_act.nunique() == 1 and rows.trajectory_id.nunique() == 16
        for _, group in rows.groupby("trajectory_id", sort=True):
            group = group.sort_values("F_total_flops")
            e, k = int(group.E.iloc[0]), int(group.K.iloc[0])
            ax.plot(group.F_total_flops, group.mean_adjusted_loss,
                    color=colors[k], lw=.6 if paper else 1.0, alpha=1 if inferno else (.85 if paper else .75), zorder=1)
            ax.scatter(group.F_total_flops, group.mean_adjusted_loss,
                       color=colors[k], marker=MARKERS[e], s=8 if paper else 25, alpha=1 if paper else .95,
                       edgecolors="white", linewidths=.2 if paper else .35, zorder=2)
        counts[ladder] = len(rows)
        title = f"({chr(97 + len(counts) - 1)}) {rows.N_act.iloc[0]/1e6:.0f}M" if paper else f"{rows.N_act.iloc[0]/1e6:.0f}M active parameters"
        ax.set_title(title, fontsize=8.5 if paper else 11, pad=5 if paper else 6)
        ax.set_xscale("log")
        ax.set_xlim(xmin / 1.12, xmax * 1.12)
        ax.set_ylim(ymin - .055 * (ymax - ymin), ymax + .055 * (ymax - ymin))
        ax.xaxis.set_major_locator(FixedLocator([1e18, 1e19, 1e20]))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.grid(which="major", alpha=.15, lw=.6)
        if paper:
            ax.tick_params(which="major", labelsize=7.5, length=2.5, width=.5, pad=2)
            ax.tick_params(which="minor", length=1.3, width=.4)
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
    assert sum(counts.values()) == 400 and len(counts) == 4
    if paper:
        handles = [Line2D([], [], color=color, lw=1.2, label=f"$K={k}$") for k, color in colors.items()]
        handles += [Line2D([], [], color="#333333", ls="none", marker=m, ms=3.2, label=f"$E={e}$") for e, m in MARKERS.items()]
        fig.legend(handles=handles, ncol=8, loc="upper center", bbox_to_anchor=(.52, .995),
                   frameon=False, fontsize=7.5, handlelength=1.2, handletextpad=.35, columnspacing=.8)
        fig.supxlabel(r"Total training compute $F=6N_{\mathrm{act}}D_{\mathrm{total}}$ (FLOPs)", y=.035, fontsize=8.5)
        axes[0].set_ylabel("Mean adjusted loss (nats)", fontsize=8.5, labelpad=4)
        fig.subplots_adjust(left=.09, right=.99, top=.79, bottom=.205, wspace=.13)
    else:
        fig.suptitle("Checkpoint mean loss across training compute", fontsize=15, y=.99)
        legend(fig, .93)
        fig.supxlabel(r"Total training compute $F=6N_{\mathrm{act}}D_{\mathrm{total}}$ (FLOPs; log scale)", y=.105)
        axes[0].set_ylabel("Mean adjusted loss (nats; lower is better)", fontsize=10)
        fig.text(.52, .035,
             "All 400 checkpoints · Lines connect checkpoints from the same trajectory · Identical axes across panels\n"
             "Each point averages its K trained languages equally; language sets differ across K.",
             ha="center", fontsize=8, linespacing=1.6)
        fig.subplots_adjust(left=.06, right=.985, top=.755, bottom=.225, wspace=.10)
    name = "checkpoint_mean_total_compute_by_size" + ("_paper" if paper else "") + ("_inferno" if inferno else "")
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220)
    plt.close(fig)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = load_primary(root)
    frame["F_t_eflops"] = 6.0 * frame.N_act.astype(float) * frame.D_t.astype(float) / 1e18
    frame["F_total_eflops"] = 6.0 * frame.N_act.astype(float) * frame.D_total.astype(float) / 1e18
    means, error = checkpoint_means(frame, root)
    means["F_total_flops"] = means.F_total_eflops * 1e18
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    drawn = 0
    for _, group in means.groupby("trajectory_id", sort=True):
        assert group[["ladder", "E", "K", "N_act"]].nunique().eq(1).all()
        group = group.sort_values("F_total_eflops")
        assert np.all(np.diff(group.F_total_eflops) > 0)
        e, k = int(group.E.iloc[0]), int(group.K.iloc[0])
        ax.plot(group.F_total_flops, group.mean_adjusted_loss,
                color=COLORS[k], lw=.85, alpha=.55, zorder=1)
        ax.scatter(group.F_total_flops, group.mean_adjusted_loss,
                   color=COLORS[k], marker=MARKERS[e], s=25, alpha=.9,
                   edgecolors="white", linewidths=.3, zorder=2)
        drawn += len(group)
    assert drawn == 400 and means.trajectory_id.nunique() == 64
    xmin, xmax = means.F_total_flops.agg(["min", "max"])
    ymin, ymax = means.mean_adjusted_loss.agg(["min", "max"])
    ax.set_xscale("log")
    ax.set_xlim(xmin / 1.12, xmax * 1.12)
    ax.set_ylim(ymin - .055 * (ymax - ymin), ymax + .055 * (ymax - ymin))
    ax.xaxis.set_major_locator(FixedLocator([3e17, 1e18, 3e18, 1e19, 3e19, 1e20]))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(np.inf, np.inf)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MultipleLocator(.25))
    ax.grid(which="major", alpha=.18, lw=.6)
    ax.set_xlabel(r"Total training compute $F=6N_{\mathrm{act}}D_{\mathrm{total}}$ (FLOPs; log scale)", labelpad=9)
    ax.set_ylabel("Mean unigram-adjusted validation loss (nats; lower is better)", labelpad=9)
    fig.suptitle("Checkpoint mean loss across training compute", fontsize=15, y=.98)
    legend(fig, .94)
    fig.text(.52, .033,
             "400 checkpoints · 64 training trajectories · 4 active model sizes\n"
             "Lines connect checkpoints from the same trajectory. Each point averages its K trained languages equally; language sets differ across K.",
             ha="center", fontsize=8.3, linespacing=1.6)
    fig.subplots_adjust(left=.105, right=.985, bottom=.17, top=.845)
    name = "checkpoint_mean_total_compute_trajectories"
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220)
    panel_counts = size_panels(means, output)
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 8, "axes.titleweight": "normal"}):
        size_panels(means, output, paper=True)
        size_panels(means, output, paper=True, inferno=True)
    metadata = {
        "source_panel": "language_vs_checkpoint_mean_compute_projection.pdf: bottom right",
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "generator_sha256": sha256_file(Path(__file__)),
        "points": drawn, "trajectories": int(means.trajectory_id.nunique()),
        "by_size_preview": "checkpoint_mean_total_compute_by_size.pdf",
        "by_size_point_counts": panel_counts,
        "paper_preview": "checkpoint_mean_total_compute_by_size_paper.pdf",
        "paper_width_cm": 16,
        "inferno_preview": "checkpoint_mean_total_compute_by_size_paper_inferno.pdf",
        "inferno_positions_by_K": {1: .02, 12: .32, 50: .62, 200: .82},
        "released_aggregate_max_abs_error": error,
        "x_limits_flops": list(ax.get_xlim()), "y_limits_nats": list(ax.get_ylim()),
        "x_scale": "log", "y_scale": "linear",
        "lines": "Straight connections between measured endpoints in log-compute coordinates; grouped by trajectory_id; no smoothing or law fitting.",
        "mean": "Equal-language mean across the K trained languages; language membership changes across K.",
    }
    (output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        main()
