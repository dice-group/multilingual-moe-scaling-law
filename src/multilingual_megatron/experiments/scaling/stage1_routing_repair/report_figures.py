"""Paper-sized figures and traceable prose for the corrected routing report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLORS = {4: "#258b9c", 16: "#d99125", 32: "#ca495e"}
MARKERS = {4: "o", 16: "s", 32: "^"}
SEEDS = (1234, 5678, 91011)
LABELS = {"mi_nats": "Language–expert MI (nats)",
          "effective_expert_fraction": "Effective expert fraction"}


def _save(fig, root, name):
    import matplotlib.pyplot as plt

    fig.savefig(root / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(root / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _intervals(ax, x, rows, metric, **style):
    """Draw percentile bounds directly: they need not contain the point estimate."""
    ax.plot(x, rows[metric], **style)
    ax.vlines(x, rows[metric + "_ci_low"], rows[metric + "_ci_high"],
              color=style["color"], linewidth=.85, alpha=.8)


def _cohort_figure(summary, cohort, root):
    import matplotlib.pyplot as plt

    data = summary[(summary.cohort == cohort) & (summary.seed == 1234)]
    ks = [12, 50, 200] if cohort == "common12" else [50, 200]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.85), sharex=True, sharey="row")
    for j, ladder in enumerate("ABCD"):
        group = data[data.ladder == ladder]
        if len(group) != 3 * len(ks):
            raise ValueError(f"incomplete plotted cohort {cohort}/{ladder}")
        first = group.iloc[0]
        for i, metric in enumerate(LABELS):
            ax = axes[i, j]
            for e in COLORS:
                rows = group[group.experts == e].set_index("languages").loc[ks]
                x = np.arange(len(ks))
                _intervals(ax, x, rows, metric, color=COLORS[e], marker=MARKERS[e],
                           markersize=3.5, linewidth=1, label=f"E={e}")
            ax.set_xticks(range(len(ks)), labels=ks)
            ax.set_xlim(-.22, len(ks) - .78)
            ax.grid(axis="y", color=".91", linewidth=.5)
            if i == 0:
                ax.set_title(f"{ladder} · {first.active_parameters / 1e6:.1f}M active\n{first.tokens_seen / 1e9:.0f}B final tokens", fontsize=8)
            else:
                ax.set_xlabel("Training languages K")
            if j == 0:
                ax.set_ylabel("Language–expert MI\n(nats)" if metric == "mi_nats" else "Effective expert\nfraction")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylim(0, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.53, .955))
    fig.suptitle(f"Fixed {cohort.removeprefix('common')}-language cohort · equal language weights · seed 1234", y=.995, fontsize=9)
    fig.text(.53, .005, "Median across layers; 95% block intervals are often smaller than the markers.\nIntervals describe evaluation uncertainty; final training horizons differ across columns.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.10, right=.995, bottom=.20, top=.76, wspace=.13, hspace=.23)
    _save(fig, root, f"routing_final_{cohort}")


def _seed_figure(contrasts, root):
    import matplotlib.pyplot as plt

    selected = contrasts[(contrasts.ladder == "B") & (contrasts.cohort == "common12")
                         & (contrasts.K_from == 12) & (contrasts.K_to == 200)
                         & contrasts.experts.isin([16, 32])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    for ax, metric in zip(axes, LABELS, strict=True):
        ax.axhline(0, color=".4", linestyle="--", linewidth=.7)
        for j, e in enumerate((16, 32)):
            rows = selected[(selected.experts == e) & (selected.metric == metric)].set_index("seed").loc[list(SEEDS)]
            x = j + np.array([-.18, 0, .18])
            for i, seed in enumerate(SEEDS):
                row = rows.loc[seed]
                ax.plot(x[i], row.delta, color=COLORS[e], marker=("o", "s", "^")[i], markersize=5, linestyle="none")
                ax.vlines(x[i], row.ci_low, row.ci_high, color=COLORS[e], linewidth=1)
            ax.plot([j - .28, j + .28], [rows.delta.mean()] * 2, color=".25", linewidth=1)
        ax.set_xticks([0, 1], labels=["E=16", "E=32"])
        ax.set_ylabel("Δ " + LABELS[metric])
        ax.set_xlim(-.48, 1.48)
        ax.grid(axis="y", color=".92", linewidth=.5)
    handles = [plt.Line2D([], [], color=".35", marker=marker, linestyle="none", label=f"Seed {seed}")
               for seed, marker in zip(SEEDS, ("o", "s", "^"), strict=True)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .95), ncol=3, frameon=False)
    fig.suptitle("B ladder · K=200 minus K=12 · common12 · 16B final tokens", y=1.02, fontsize=9)
    fig.text(.53, .005, "Points are individual training seeds; bars are paired sequence-block intervals.\nShort horizontal lines show the three-seed mean, not a seed confidence interval.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.095, right=.995, bottom=.26, top=.76, wspace=.35)
    _save(fig, root, "routing_b_seed_contrasts")


def _association_figure(associations, root):
    import matplotlib.pyplot as plt

    data = associations[(associations.cohort == "common50") & (associations.languages == 200)]
    row_keys = [("A", 1234), ("B", 1234), ("B", 5678), ("B", 91011), ("C", 1234), ("D", 1234)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True, layout="constrained")
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad(".93")
    for ax, predictor, title in zip(axes, ("effective_expert_fraction", "mean_pair_js_nats"),
                                     ("Expert breadth vs MoE gain", "Routing distinctness vs MoE gain"), strict=True):
        values = np.full((len(row_keys), 3), np.nan)
        for i, (ladder, seed) in enumerate(row_keys):
            for j, e in enumerate(COLORS):
                rows = data[(data.ladder == ladder) & (data.seed == seed) & (data.experts == e) & (data.predictor == predictor)]
                if len(rows) == 1:
                    if rows.iloc[0].n_languages != 50:
                        raise ValueError("association plot does not use the declared common50")
                    values[i, j] = rows.iloc[0].spearman_rho
                elif len(rows) > 1 or not (ladder == "B" and seed != 1234 and e == 4):
                    raise ValueError("missing or duplicated association plot cell")
                text = f"{values[i, j]:+.2f}" if np.isfinite(values[i, j]) else "—"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if abs(values[i, j]) > .65 else "black")
        im = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1, 2], labels=["E=4", "E=16", "E=32"])
        ax.set_yticks(range(len(row_keys)), labels=[f"{l} · seed {s}" for l, s in row_keys])
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=list(axes), label="Within-checkpoint Spearman ρ", fraction=.035, pad=.03)
    fig.suptitle("Exploratory associations · K=200 · common50 (50 languages per cell)\nPositive MoE gain = lower loss than matched dense baseline", fontsize=9)
    _save(fig, root, "routing_loss_associations")


def render_figures(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output) / "figures"
    root.mkdir()
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"}):
        for cohort in ("common12", "common50"):
            _cohort_figure(tables["checkpoint_summary"], cohort, root)
        _seed_figure(tables["coverage_contrasts"], root)
        _association_figure(tables["loss_associations"], root)


def _markdown(frame, columns, digits=4):
    """Small Markdown tables without an optional tabulate dependency."""
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def write_findings(tables, audit, output):
    summary, contrasts = tables["checkpoint_summary"], tables["coverage_contrasts"]
    primary = summary[(summary.cohort == "common12") & (summary.seed == 1234)]
    seed_contrasts = contrasts[(contrasts.ladder == "B") & (contrasts.cohort == "common12")
                              & (contrasts.K_from == 12) & (contrasts.K_to == 200)
                              & contrasts.experts.isin([16, 32]) & (contrasts.metric == "mi_nats")]
    padding = tables["padding_impact"]
    padding = padding[padding.cohort.isin(["common12", "common50"])]
    associations = tables["loss_associations"]
    focus = associations[(associations.cohort == "common50") & (associations.languages == 200)]
    illustration = primary[(primary.experts == 32) & (primary.languages == 200)]
    findings = {"primary_common12_mi_range_by_ladder": {}, "B_seed_K200_minus_K12_mi": [],
                "E32_K200_common12": illustration[["ladder", "mi_nats", "effective_expert_fraction"]].to_dict(orient="records"),
                "padding_common_cohorts": {"max_abs_median_mi_change": float(padding.valid_minus_all_mi_nats.abs().max()),
                                           "median_abs_median_mi_change": float(padding.valid_minus_all_mi_nats.abs().median())},
                "association_common50_K200": []}
    for ladder, rows in primary.groupby("ladder"):
        findings["primary_common12_mi_range_by_ladder"][ladder] = {"min": float(rows.mi_nats.min()), "max": float(rows.mi_nats.max())}
    for e, rows in seed_contrasts.groupby("experts"):
        findings["B_seed_K200_minus_K12_mi"].append({"experts": int(e), "positive_seeds": int((rows.delta > 0).sum()),
                                                   "seeds": len(rows), "mean": float(rows.delta.mean()),
                                                   "min": float(rows.delta.min()), "max": float(rows.delta.max())})
    for predictor, rows in focus.groupby("predictor"):
        findings["association_common50_K200"].append({"predictor": predictor, "positive": int((rows.spearman_rho > 0).sum()),
                                                     "negative": int((rows.spearman_rho < 0).sum()), "cells": len(rows),
                                                     "rho_min": float(rows.spearman_rho.min()), "rho_max": float(rows.spearman_rho.max())})
    (output / "findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True, allow_nan=False) + "\n")
    seed_summary = pd.DataFrame(findings["B_seed_K200_minus_K12_mi"])
    range_frame = pd.DataFrame.from_dict(findings["primary_common12_mi_range_by_ladder"], orient="index").rename_axis("ladder").reset_index()
    association_by_predictor = {row['predictor']: row for row in findings['association_common50_K200']}
    distinct = association_by_predictor['mean_pair_js_nats']
    breadth = association_by_predictor['effective_expert_fraction']
    text = f"""# Corrected final-checkpoint routing analysis

Exploratory analysis after preliminary inspection. The independent replication
unit is a trained trajectory. This report does not estimate the paper's
matched-language-exposure attenuation or establish a causal routing mechanism.

## Evidence and definitions

All {audit['checkpoints']} sparse final checkpoints and {audit['language_streams']:,}
trained-language streams passed the reporting checks, with {audit['dense_controls']}
matched dense controls. There are {audit['valid_input_positions']:,} valid input
positions and {audit['scored_target_tokens']:,} scored targets. Historical
scored-token counts and NLL sums are exactly unchanged.

Common12 comparisons use the same 12 languages for K=12/50/200; common50 uses
the same 50 for K=50/200. Languages are weighted equally. Every summary first
computes a layer statistic, then takes the median across layers. MI is
H(mean p) minus mean H(p), in nats. Expert breadth is the arithmetic mean of
exp(H(p_language))/E, not exp(mean language entropy)/E. K=1 has no language MI.
Full trained-cohort summaries and all 4,008 per-language loss gains remain in
the tables; changes in cohort membership are not treated as routing effects.

The 95% intervals use 1,000 shared draws of 128-sequence blocks, seed 20260907.
The checkpoint median is recomputed inside each draw. These intervals describe
evaluation sampling conditional on a trained model, not seed uncertainty.
Actual evaluator microbatch sizes are recorded separately from training sizes.
The language artifact's input_sha256 includes NLL, so joins instead verify
target-token hashes, tokenizer, stream layout and scoring metadata.

## Routing pattern at final checkpoints

![Common12 routing](figures/routing_final_common12.png)

The range of checkpoint-median MI across expert counts and training K, for
seed1234 and common12, is:

{_markdown(range_frame, ['ladder', 'min', 'max'])}

For a concrete matched-E/K illustration, E32/K200 on common12 gives:

{_markdown(illustration, ['ladder', 'mi_nats', 'effective_expert_fraction'])}

At these endpoints, C/D distribute each language more broadly across experts
and show less difference between languages than A/B. The full figure retains
all E/K cells; this illustration is not an additional selected test.

These are endpoint descriptions. A/B/C/D have approximately 8/16/32/32B
training tokens, respectively. Differences across these columns cannot
isolate model size. Exact horizons and active parameter counts are preserved
in checkpoint_summary. Expert identities are meaningful only within a
checkpoint and layer. Marginal routing overlap cannot establish semantic
sharing, token-feature specialization, transfer, or reduced interference.

Common50 robustness is in [the companion figure](figures/routing_final_common50.pdf).
Layer-level evidence remains in the immutable corrected release's
cohort_summary.parquet and common12_by_layer.pdf; medians do not replace it.

## Training-seed replication

![Individual B seeds](figures/routing_b_seed_contrasts.png)

For B/common12, K=200 minus K=12 checkpoint-median MI:

{_markdown(seed_summary, ['experts', 'positive_seeds', 'seeds', 'mean', 'min', 'max'])}

All three seeds are shown individually. The seed range is descriptive, and
seed disagreement remains part of the result. Additional seeds exist only
for B/E16,32/K1,12,200. These routing contrasts are distinct from the paper's
three-seed loss attenuation analysis.

## Exploratory routing–loss associations

![Loss associations](figures/routing_loss_associations.png)

MoE gain is matched dense minus sparse adjusted loss; the raw-loss difference
agrees to numerical precision because the scored stream and unigram reference
are identical. For each checkpoint/cohort, Spearman rho compares 12 or 50
languages. Breadth is median layer exp(H(p_language))/E. Distinctness is median
layer mean pairwise Jensen–Shannon divergence to other cohort languages.

{_markdown(pd.DataFrame(findings['association_common50_K200']), ['predictor', 'positive', 'negative', 'cells', 'rho_min', 'rho_max'])}

More distinct language routing is positively associated with MoE gain in
{distinct['positive']} of these {distinct['cells']} checkpoint comparisons; broader expert use is negatively
associated in {breadth['negative']} of {breadth['cells']}. Magnitudes and directions vary across settings. These
counts describe the displayed checkpoints and are not a replication test.

This table describes common50/K200. All 144 eligible associations are retained
in loss_associations, including common12 and other K settings. No pooled
cross-model correlation, p-values, or causal interpretation is used. Languages
are a fixed cohort, and their correlation is not independent training replication.

## Effect of excluding padding

Across the 72 common-cohort checkpoint summaries, the maximum absolute change
in median MI from corrected all-position to valid-position counts is
{findings['padding_common_cohorts']['max_abs_median_mi_change']:.6f} nats;
the median absolute change is {findings['padding_common_cohorts']['median_abs_median_mi_change']:.6f} nats.
The signed change is valid minus all. Historical-v1 minus corrected-all values
are preserved separately because they can include old rank-reconciliation
differences. The old paper plot also used matched 8B readouts and different
weighting; it is not a padding-only comparison with these final endpoints.

## Reproducibility and review

The release_manifest binds inputs, implementation files, analysis_spec, tables,
figures, and this report. Generated numerical claims are recorded in findings.json
and draft_findings.tex. Every plotted point can be traced to the CSV/Parquet
tables. The four PDF figures have PNG previews. Re-run the report CLI with
--verify for read-only integrity checks, adding --replay for an exact numerical
replay. Outputs were written separately from the sealed campaign; manuscript
sources were not edited. No GPU work was required.
"""
    (output / "README.md").write_text(text)
    parts = [r"% Standalone draft only; not inserted into the manuscript.",
             r"% Source rows: checkpoint_summary (common12, seed1234), coverage_contrasts (B, common12, K12->200, mi_nats).",
             r"\paragraph{Exploratory routing diagnostics.}",
             r"We analyze corrected routing counts at 60 sparse final checkpoints, excluding synthetic padding while retaining real special tokens.",
             r"On a fixed 12-language cohort, we summarize language--expert mutual information and effective expert breadth by their median across layers."]
    for ladder, values in findings["primary_common12_mi_range_by_ladder"].items():
        parts.append(f"For ladder {ladder}, checkpoint-median MI spans {values['min']:.3f}--{values['max']:.3f} nats across the evaluated expert counts and training coverages (seed 1234).")
    parts.append(r"Final training horizons differ (approximately 8B, 16B, 32B, and 32B tokens for A--D), so these endpoint differences do not identify a model-size effect.")
    for row in findings["B_seed_K200_minus_K12_mi"]:
        parts.append(f"For B/E{row['experts']}, the K=200 minus K=12 MI contrast is positive in {row['positive_seeds']} of three training seeds, with a range of {row['min']:+.4f} to {row['max']:+.4f} nats.")
    parts.append(f"Across the {distinct['cells']} common50/K200 checkpoint comparisons, routing distinctness has a positive Spearman association with matched MoE loss gain in {distinct['positive']} cases; expert breadth has a negative association in {breadth['negative']} cases. These are descriptive counts over the available checkpoints, not a replication test.")
    parts.append(r"Within-checkpoint routing--loss correlations are exploratory descriptions of fixed language cohorts; they do not establish that routing explains the attenuation result.")
    parts += ["", r"% Optional caption macros for subsequent manuscript integration:",
              r"\newcommand{\RoutingFinalCaption}{Corrected final-checkpoint routing on a fixed 12-language cohort. Columns show A--D with their final training horizons; colors identify expert count. Points summarize layers by their median. Top: language--expert mutual information under equal language weights. Bottom: arithmetic mean within-language effective expert fraction. Intervals are 95\% sequence-block bootstrap intervals conditional on each trained checkpoint (seed 1234), not uncertainty across training seeds. See the companion figure for the three B seeds.}",
              r"\newcommand{\RoutingCommonFiftyCaption}{The same analysis on the fixed 50-language cohort, comparing K=50 with K=200.}",
              r"\newcommand{\RoutingSeedCaption}{B/common12, K=200 minus K=12. Points show all three training seeds; intervals use paired block draws. Horizontal ticks show the three-seed mean.}",
              r"\newcommand{\RoutingAssociationCaption}{Exploratory within-checkpoint Spearman correlations across common50 languages at K=200. Positive MoE gain means lower loss than the matched dense baseline. No causal or significance claim is made; gray cells are unavailable B/E4 additional seeds.}"]
    (output / "draft_findings.tex").write_text("\n".join(parts) + "\n")
