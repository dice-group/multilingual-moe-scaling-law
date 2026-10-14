"""Compact Inferno paper export from verified routing summary tables.

Presentation only: never regenerate or overwrite the corrected analysis release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import identity, read, verify, write


def export(report_root: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    report_root, output = report_root.resolve(), output.resolve()
    release_path = report_root / "release_manifest.json"
    release = read(release_path)
    if release["schema"] != "stage1_routing_report/v1" or release["status"] != "complete":
        raise ValueError("complete corrected routing report required")
    table_path = report_root / "tables/checkpoint_summary.parquet"
    table_entry = {"path": str(table_path), "sha256": release["outputs"]["tables/checkpoint_summary.parquet"]["sha256"]}
    verify(table_entry)
    campaign_entry = next(e for e in release["inputs"] if e["sha256"] == release["campaign_sha256"])
    verify(campaign_entry)
    campaign = read(campaign_entry["path"])
    frame = pd.read_parquet(table_path)
    frame = frame[(frame.cohort == "common12") & (frame.seed == 1234)].copy()
    expected = {(l, e, k) for l in "ABCD" for e in (4, 16, 32) for k in (12, 50, 200)}
    if len(frame) != 36 or set(zip(frame.ladder, frame.experts, frame.languages)) != expected:
        raise ValueError("incomplete or duplicated common12 primary-seed panel")
    models = {t["model"]["id"]: t["model"] for t in campaign["tasks"] if t["kind"] == "production"}
    frame["active_nonembedding_parameters"] = [models[i]["active_nonembedding_parameters"] for i in frame.id]
    if any(models[r.id]["tokens_seen"] != r.tokens_seen for r in frame.itertuples()):
        raise ValueError("checkpoint horizon differs from sealed campaign")
    for suffix in ("", "_ci_low", "_ci_high"):
        frame["normalized_mi" + suffix] = frame["mi_nats" + suffix] / np.log(frame.experts)
    if not np.allclose(frame.normalized_mi, frame.mi_over_log_experts, rtol=0, atol=1e-14):
        raise ValueError("normalized MI differs from released statistic")
    positions = {4: .32, 16: .62, 32: .82}
    colors = {e: plt.colormaps["inferno"](p) for e, p in positions.items()}
    markers = {12: "o", 50: "s", 200: "^"}
    offsets = {12: -.075, 50: 0, 200: .075}
    style = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
             "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
             "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .5,
             "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42}
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 3.25))
        for ax, metric, title, ylabel in zip(
                axes, ("normalized_mi", "effective_expert_fraction"),
                ("(a) Language-specific routing", "(b) Per-language expert breadth"),
                ("Normalized language–expert MI", "Within-language effective expert fraction"), strict=True):
            for e, color in colors.items():
                for k, marker in markers.items():
                    rows = frame[(frame.experts == e) & (frame.languages == k)].set_index("ladder").loc[list("ABCD")]
                    x = np.arange(4) + offsets[k]
                    ax.plot(x, rows[metric], color=color, marker=marker, lw=.9, ms=3.6,
                            markeredgecolor="white", markeredgewidth=.3, zorder=3)
                    low, high = rows[metric + "_ci_low"], rows[metric + "_ci_high"]
                    ax.vlines(x, low, high, color=color, lw=.6)
                    ax.hlines(low, x - .022, x + .022, color=color, lw=.6)
                    ax.hlines(high, x - .022, x + .022, color=color, lw=.6)
            labels = []
            for ladder in "ABCD":
                rows = frame[frame.ladder == ladder]
                if rows.active_nonembedding_parameters.nunique() != 1 or rows.tokens_seen.nunique() != 1:
                    raise ValueError("size/horizon varies inside a plotted ladder")
                row = rows.iloc[0]
                labels.append(f"{ladder}\n{row.active_nonembedding_parameters / 1e6:.0f}M\n{row.tokens_seen / 1e9:.0f}B tokens")
            ax.set_xticks(range(4), labels=labels)
            ax.set_xlim(-.25, 3.25)
            ax.set_title(title, pad=6)
            ax.set_ylabel(ylabel, labelpad=4)
            ax.grid(axis="y", alpha=.15, lw=.6)
            ax.tick_params(length=2.5, width=.5, pad=2)
            ax.set_axisbelow(True)
        axes[0].set_ylim(0, frame.normalized_mi_ci_high.max() * 1.10)
        axes[1].set_ylim(max(0, np.floor((frame.effective_expert_fraction_ci_low.min() - .07) * 20) / 20), 1.015)
        expert_handles = [Line2D([], [], color=c, lw=1.2, label=f"$E={e}$") for e, c in colors.items()]
        coverage_handles = [Line2D([], [], color=".25", marker=m, ls="none", ms=3.6, label=f"$K={k}$") for k, m in markers.items()]
        fig.legend(handles=expert_handles, title="Expert count", loc="upper center", bbox_to_anchor=(.30, 1.0),
                   ncol=3, frameon=False, fontsize=7, title_fontsize=7.5, handlelength=1.2, columnspacing=1)
        fig.legend(handles=coverage_handles, title="Training languages", loc="upper center", bbox_to_anchor=(.79, 1.0),
                   ncol=3, frameon=False, fontsize=7, title_fontsize=7.5, handlelength=1, columnspacing=1)
        fig.supxlabel("Active-size ladder (non-embedding parameters); final training tokens", y=.115, fontsize=8)
        fig.text(.53, .025, "Common12, equal language weights, seed 1234. Points: layer medians; bars: 95% block intervals.\n"
                 "Final horizons differ; cross-size curves are descriptive. Intervals are often smaller than markers.",
                 ha="center", va="bottom", fontsize=6.8)
        fig.subplots_adjust(left=.085, right=.975, bottom=.35, top=.77, wspace=.34)
        for extension in ("pdf", "png"):
            kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
            fig.savefig(output / f"fig_routing_summary_inferno.{extension}", dpi=240, **kwargs)
        plt.close(fig)
    frame.to_csv(output / "plotted_points.csv", index=False, float_format="%.17g")
    caption = ("Corrected final-checkpoint routing on a fixed 12-language cohort (seed 1234, equal language weights). "
               "Colors identify expert count and marker shapes identify training-language coverage. "
               "Left: language–expert mutual information divided by log(E). Right: the arithmetic mean across languages "
               "of exp(entropy)/E. Points are medians across layers; bars are 95% sequence-block bootstrap intervals "
               "conditional on the trained checkpoint, not layer IQRs or training-seed confidence intervals. "
               "A/B/C/D final horizons are approximately 8/16/32/32B tokens; connecting lines describe endpoints, "
               "not an isolated model-size effect. The three B training seeds remain in the companion report.")
    (output / "caption.md").write_text(caption + "\n")
    for entry in (table_entry, campaign_entry):
        verify(entry)
    manifest = {"schema": "stage1_routing_paper_figure/v1", "status": "complete", "points": len(frame),
                "source_report": identity(release_path), "source_table": table_entry, "campaign": campaign_entry,
                "generator": identity(__file__), "palette": "inferno", "positions_by_experts": positions,
                "style": style, "caption": caption,
                "outputs": {p.name: identity(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write(output / "figure_manifest.json", manifest)
    return {"status": "complete", "points": len(frame), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.report, args.output), indent=2))


if __name__ == "__main__":
    main()
