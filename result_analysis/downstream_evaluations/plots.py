"""External transfer and paired decision-change review figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(headline, losses, answers, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for benchmark, group in losses.groupby("benchmark"):
        axes[0].plot(group.ladder, group.attenuation_percent, marker="o", label=benchmark)
    axes[0].set(ylabel="Coverage-penalty attenuation (%)", xlabel="Size ladder", title="External text loss")
    axes[0].legend(frameon=False)
    accuracy = headline.query("ladder == 'D'").loc[lambda f: ~f.benchmark.isin(["flores_mono", "ntrex_mono"])].sort_values("interaction")
    axes[1].barh(accuracy.benchmark, accuracy.interaction, color=np.where(accuracy.interaction > 0, "#21867a", "#be6b4a"))
    axes[1].axvline(0, color=".4", lw=.7)
    axes[1].set(xlabel="E32−E1 gain at K200 minus K12 (pp)", title="Accuracy transfer is task-dependent (D)")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"external_transfer.{ext}", dpi=150)
    plt.close(fig)
    row = answers.query("benchmark == 'okapi_hellaswag' and ladder == 'D'").iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    for i, k in enumerate((12, 200)):
        axes[0].bar(np.arange(2)+(i-.5)*.35, [row[f"gold_nll_gain_K{k}"], row[f"wrong_nll_gain_K{k}"]], .35, label=f"K={k}")
        axes[1].bar(np.arange(2)+(i-.5)*.35, [row[f"corrected_pp_K{k}"], row[f"broken_pp_K{k}"]], .35)
    axes[0].set(xticks=[0, 1], xticklabels=["Correct", "Mean distractor"], ylabel="E1−E32 NLL (nats)")
    axes[0].legend(frameon=False)
    axes[1].set(xticks=[0, 1], xticklabels=["Corrected", "Broken"], ylabel="Items (percentage points)")
    axes[2].bar([0, 1], [row.primary_gain_K12, row.primary_gain_K200])
    axes[2].set(xticks=[0, 1], xticklabels=["K=12", "K=200"], ylabel="E32−E1 accuracy (pp)")
    fig.suptitle("HellaSwag D: likelihood gains and answer changes are distinct")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"decision_accounting.{ext}", dpi=150)
    plt.close(fig)
