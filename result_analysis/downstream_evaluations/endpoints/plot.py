"""Render descriptive full-benchmark/loss comparisons from verified CSV tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from result_analysis.downstream_evaluations.endpoints.analyze import BENCHMARKS
from result_analysis.capacity_response.tables import sha256_file

COLORS = {12: "#48186a", 50: "#bc3754", 200: "#ed9b27"}
MARKERS = dict(zip("ABCD", ["o", "s", "^", "D"]))
SIZES = {"A": "93M / 8B", "B": "190M / 16B", "C": "349M / 32B", "D": "590M / 32B"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.16, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(source, output):
    receipt = json.loads((source / "verification.json").read_text())
    names = ["cohort_metrics", "benchmark_summary", "coverage_interactions"]
    frames = {}
    for name in names:
        path = source / f"{name}.csv"
        assert sha256_file(path) == receipt["outputs"][name]["sha256"]
        frames[name] = pd.read_csv(path)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    data = frames["cohort_metrics"]
    summary = frames["benchmark_summary"].query("metric == 'acc_norm'").set_index("benchmark")
    selected = data.query("scope == 'common12' and seed == 1234 and K in [12,50,200]")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    for ax, (benchmark, label) in zip(axes.flat, BENCHMARKS.items()):
        panel = selected.loc[selected.benchmark.eq(benchmark)]
        for (k, ladder), g in panel.groupby(["K", "ladder"]):
            ax.scatter(g.raw_nll, 100 * g.acc_norm, color=COLORS[k], marker=MARKERS[ladder],
                       s=28, alpha=.85, linewidth=.35, edgecolors="white")
        n = int(summary.loc[benchmark, "common12_languages"])
        ax.set_title(f"{label} · {n} shared language{'s' if n != 1 else ''}", loc="left", fontsize=10)
        ax.set_xlabel("Validation NLL (lower is better)")
        ax.set_ylabel("Normalized accuracy (%)")
        ax.text(.96, .95, f"ρ(−loss, accuracy) = {summary.loc[benchmark, 'loss_accuracy_spearman_48']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
        style(ax)
    handles = [Line2D([], [], color=c, marker="o", linestyle="", label=f"K={k}") for k,c in COLORS.items()]
    handles += [Line2D([], [], color=".3", marker=m, linestyle="", label=SIZES[l]) for l,m in MARKERS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=7, frameon=False, fontsize=8)
    fig.suptitle("Lower validation loss generally predicts better benchmark accuracy", fontsize=14, y=1.01)
    fig.text(.5, .005, "48 primary endpoints per panel; fixed benchmark-supported K12 languages; one seed per setting.\n"
             "Markers show active parameters / training tokens. Correlations are descriptive; no fitted benchmark scaling law.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .91))
    save(fig, output, "loss_accuracy_by_benchmark")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7))
    for ax, (benchmark, label) in zip(axes.flat, BENCHMARKS.items()):
        panel = selected.loc[selected.benchmark.eq(benchmark) & selected.ladder.eq("D")]
        for k, g in panel.groupby("K"):
            g = g.sort_values("E")
            ax.plot(g.E, 100 * g.acc_norm, color=COLORS[k], marker="o", linewidth=1.6, markersize=4, label=f"K={k}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 32], labels=["1", "4", "16", "32"])
        ax.set_xlabel("Experts")
        ax.set_ylabel("Normalized accuracy (%)")
        ax.set_title(label, loc="left", fontsize=10)
        style(ax)
    fig.legend(handles=handles[:3], loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.suptitle("Expert scaling by benchmark at 590M active parameters and 32B tokens", fontsize=14, y=1.01)
    fig.text(.5, .005, "Fixed benchmark-supported K12 languages across K; ARC Easy / Uhura is English-only in this cohort.\n"
             "Complete benchmark splits; one training seed. Connecting lines are guides between observed settings.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .91))
    save(fig, output, "expert_scaling_by_benchmark")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    inter = frames["coverage_interactions"].query("scope == 'common12' and low == 12 and high == 200")
    for ax, (benchmark, label) in zip(axes.flat, BENCHMARKS.items()):
        panel = inter.loc[inter.benchmark.eq(benchmark)]
        for r in panel.itertuples():
            ax.scatter(r.loss_interaction_nats, r.acc_norm_interaction_pp,
                       marker=MARKERS[r.ladder], color="#bc3754" if r.seed == 1234 else "#267d9b",
                       s=48, facecolors=None if r.seed == 1234 else "none", linewidth=1)
            if r.seed == 1234:
                ax.annotate(r.ladder, (r.loss_interaction_nats, r.acc_norm_interaction_pp),
                            xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.axhline(0, color=".35", linewidth=.8)
        ax.axvline(0, color=".35", linewidth=.8)
        ax.set_xlabel("Loss interaction (nats; positive favors K200)")
        ax.set_ylabel("Accuracy interaction (pp)")
        ax.set_title(label, loc="left", fontsize=10)
        ax.margins(.15)
        style(ax)
    fig.suptitle("The expert–coverage interaction does not consistently transfer to accuracy", fontsize=14, y=1.01)
    fig.text(.5, .005, "Each interaction compares the E1→E32 gain at K200 against that at K12. Positive means experts help more at K200.\n"
             "Fixed total tokens within size; target-language exposure changes with K. A–D: seed 1234; open squares: B seeds 5678 and 91011.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .97))
    save(fig, output, "loss_vs_accuracy_interaction")
    (output / "manifest.json").write_text(json.dumps({"source": str(source.resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {n: receipt["outputs"][n] for n in names},
        "figures": {p.name: sha256_file(p) for p in output.iterdir() if p.suffix in {".pdf", ".png"}}}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
