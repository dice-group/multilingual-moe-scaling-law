"""Small, deterministic paper exports and a complete analysis report."""
from __future__ import annotations

import json

import numpy as np

LABELS = {"M6-ND": "Full law", "M5-ND": "+ coverage", "M6": "Separable law",
          "J0-ND": "MoE + size--data", "J0": "MoE baseline", "C0": "Chinchilla-form",
          "M0": "Count/capacity-form", "M1": "Expert-dependent law", "zero_shift": "Zero shift"}
ORDER = tuple(LABELS)


def export(output, report, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    derived = output / "derived"
    seeds = frames["seed_metrics"]
    primary = seeds[seeds.horizon.eq(16_000_000_000)]
    empirical = frames["empirical_contrasts"]
    original = empirical[empirical.cohort.eq("unseen_177") & empirical.horizon.eq(16_000_000_000)].sort_values("seed")
    support = json.loads((output / "inputs/composition_support.json").read_text())
    parity = json.loads((output / "inputs/manifest.json").read_text())
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    def save(fig, name):
        fig.savefig(derived / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(derived / f"{name}.png", dpi=150, bbox_inches="tight", metadata={"Software": "composition analysis v1"})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4), constrained_layout=True)
    axes[0].axhline(0, color="0.5", linewidth=.8)
    axes[0].scatter(range(3), original.paired_diverse_minus_compact_gain, color="#2463a6", s=48)
    axes[0].set(xticks=range(3), xticklabels=original.seed.astype(str), xlabel="Training seed",
                ylabel="Change in E1−E32 gain (nats)", title="Original primary: unseen 177, 16B")
    for i, model in enumerate(ORDER):
        vals = primary[primary.model.eq(model)].sort_values("seed").contrast_rmse.to_numpy()
        axes[1].scatter(vals, np.full(3, i), s=21, alpha=.75, color="#bb542d" if model == "M6-ND" else "#2463a6")
        axes[1].plot([vals.mean()], [i], "|", color="black", markersize=12)
    axes[1].set(yticks=range(len(ORDER)), yticklabels=[LABELS[m].replace("--", "–") for m in ORDER],
                xlabel="Arm-contrast RMSE (nats; lower is better)", title="Secondary: shared 7, 16B")
    axes[1].invert_yaxis()
    save(fig, "composition_results")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for model in ("M6-ND", "M5-ND", "M6", "J0-ND", "J0", "zero_shift"):
        group = seeds[seeds.model.eq(model)].groupby("horizon")
        curve = group.contrast_rmse.mean()
        axes[0].plot(curve.index / 1e9, curve, marker="o", ms=3, label=LABELS[model].replace("--", "–"))
        if model != "zero_shift":
            absolute = group.absolute_rmse.mean()
            axes[1].plot(absolute.index / 1e9, absolute, marker="o", ms=3, label=LABELS[model].replace("--", "–"))
    for ax, title in zip(axes, ("Mixture-shift prediction", "Absolute loss prediction")):
        ax.set(xscale="log", xticks=[.5, 1, 2, 4, 8, 16], xlabel="Nominal training tokens (B)",
               ylabel="RMSE (nats)", title=title)
        ax.xaxis.set_major_formatter(ScalarFormatter())
    axes[0].legend(fontsize=8)
    save(fig, "composition_prediction_horizons")

    cohorts = sorted(empirical.cohort.unique())
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    for ax, cohort in zip(axes.flat, cohorts):
        for seed, group in empirical[empirical.cohort.eq(cohort)].groupby("seed"):
            group = group.sort_values("horizon")
            ax.plot(group.horizon / 1e9, group.paired_diverse_minus_compact_gain, marker="o", ms=3, label=str(seed))
        ax.axhline(0, color="0.5", linewidth=.6)
        ax.set(title=cohort.replace("_", " "), xscale="log", xticks=[.5, 2, 8, 16], xlabel="Tokens (B)", ylabel="MoE-gain difference (nats)")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    for ax in list(axes.flat)[len(cohorts):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    save(fig, "composition_empirical_horizons")

    tex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & Seed 1234 & Seed 5678 & Seed 91011 & Mean \\", r"\midrule"]
    for name in ORDER:
        vals = primary[primary.model.eq(name)].sort_values("seed").contrast_rmse.to_numpy()
        tex.append(LABELS[name] + " & " + " & ".join(f"{v:.5f}" for v in [*vals, vals.mean()]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (derived / "table_prediction.tex").write_text("\n".join(tex) + "\n")
    tex = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Absolute RMSE & Absolute MAE & Interaction RMSE \\", r"\midrule"]
    for name in ORDER[:-1]:
        vals = primary[primary.model.eq(name)][["absolute_rmse", "absolute_mae", "interaction_rmse"]].mean()
        tex.append(LABELS[name] + " & " + " & ".join(f"{v:.5f}" for v in vals) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (derived / "table_absolute.tex").write_text("\n".join(tex) + "\n")
    tex = [r"\begin{tabular}{lrrr}", r"\toprule", r"Seed & Compact gain & Diverse gain & Difference \\", r"\midrule"]
    for row in original.itertuples():
        tex.append(f"{row.seed} & {row.compact_moe_gain:.5f} & {row.diverse_moe_gain:.5f} & {row.paired_diverse_minus_compact_gain:.5f}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (derived / "table_empirical.tex").write_text("\n".join(tex) + "\n")
    lines = ["# Completed language-composition analysis", "",
        "All 162 evaluations and 31,752 language losses are included. No models were fitted or selected.", "",
        "## Original empirical primary", "",
        "Unseen means absent from all three training mixtures, not absent from tokenizer construction or historical law fitting.",
        "The primary compares diverse-minus-compact E1−E32 gains at 16B on unseen177.", "",
        "| Seed | Compact gain | Diverse gain | Difference |", "|---|---:|---:|---:|"]
    for row in original.itertuples():
        lines.append(f"| {row.seed} | {row.compact_moe_gain:.8f} | {row.diverse_moe_gain:.8f} | {row.paired_diverse_minus_compact_gain:.8f} |")
    lines += ["", "The mixed directions support neither a consistent diversity advantage nor equivalence.", "",
        "## Frozen-law secondary analysis", "",
        "Primary prediction metric: 16B diverse-minus-compact loss contrast RMSE on the common seven training languages and E1/E16/E32. "
        "Each seed's score averages three fixed-fit RMSEs; fits and nested languages/readouts are not independent replications.", "",
        "| Model | Mean contrast RMSE | Seed SD | Minimum | Maximum |", "|---|---:|---:|---:|---:|"]
    for name in ORDER:
        d = report["primary_predictive_scores"][name]
        lines.append(f"| {LABELS[name]} ({name}) | {d['mean']:.8f} | {d['sample_standard_deviation']:.8f} | {d['range'][0]:.8f} | {d['range'][1]:.8f} |")
    difference = report["full_minus_coverage_summary"]
    means = primary.groupby("model").mean(numeric_only=True)
    macros = {
        "LCompFullContrast": f"{means.loc['M6-ND', 'contrast_rmse']:.5f}",
        "LCompCoverageContrast": f"{means.loc['M5-ND', 'contrast_rmse']:.5f}",
        "LCompContrastImprovement": f"{100 * (1 - means.loc['M6-ND', 'contrast_rmse'] / means.loc['M5-ND', 'contrast_rmse']):.1f}\\%",
        "LCompFullAbsolute": f"{means.loc['M6-ND', 'absolute_rmse']:.5f}",
        "LCompCoverageAbsolute": f"{means.loc['M5-ND', 'absolute_rmse']:.5f}",
    }
    (derived / "numbers.tex").write_text("\n".join(
        "\\newcommand{\\" + name + "}{" + value + "}" for name, value in macros.items()) + "\n")
    lines += ["", f"Paired Full-minus-coverage RMSE: mean {difference['mean']:.8f}; sample SD {difference['sample_standard_deviation']:.8f}; range {difference['range']}. Negative favors Full.",
        "", "See fit_metrics.csv for every retained fit and paired_fit_differences.csv for all paired comparisons. "
        "seed_metrics.csv includes absolute RMSE/MAE and the E1−E32 interaction error. All horizons are retained.",
        "", "## Interpretation and limits", "",
        "- Full has slightly worse absolute RMSE than coverage for all three seed means. The gain concerns composition contrasts, not universal prediction superiority.",
        "- Separable M6 beats Full on contrast RMSE in the E32 and largest-size historical fits; Full's average advantage comes from the K200 fit. This does not establish necessity of all 15 coordinates.",
        "- This secondary analysis was specified after aggregate empirical outcomes were inspected; it is not preregistered prospective validation.",
        "- The primary law panel contains six Indo-European targets and Chinese. All 24 new-arm target/arm pairs have zero same-script/outside-family exposure. The script coefficient is untested.",
        "- Fixed K12, B size, three seeds, 0.5–16B horizons and one deliberately selected mixture pair do not validate cross-size, K50/K200, or longer-horizon generalization.",
        "- Mixture selection changes language identity and domain as well as exposure composition; the fitted coordinates do not identify causal linguistic mechanisms.",
        "- The reused reference trajectories are historical context. They are excluded from new-mixture prediction scoring.",
        "- Existing laws require positive target exposure; no epsilon or new nuisance coefficients are introduced for unseen177.",
        "- Old/new normalization offsets are derived from fixed unigram evidence, not prediction residuals. Raw losses and exact target identities are checked on all 54 reused checkpoints.",
        "- C0 is the retained Chinchilla-form target-exposure control; J0 is the harmonized MoE baseline. M0 is a count/capacity form; M1 is study-developed. These are not all possible literature adaptations.",
        "- Routing counters are not analyzed; the separate padding-repair work does not supply evidence here.", "",
        f"Historical replay: {sum(r['rows'] for r in report['historical_replay']):,} predictions; maximum error {max(r['maximum_error'] for r in report['historical_replay']):.3g}.",
        f"Source binding inventory: {len(parity['source_bindings']):,} files. Full composition support: inputs/composition_support.json ({len(support)} rows).", ""]
    (derived / "REPORT.md").write_text("\n".join(lines))
