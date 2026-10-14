"""Plot coverage gaps and expert gains from derived tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(ladder, summary, response, gains, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for name, group in ladder.groupby("ladder"):
        axes[0].plot(group.E, group.pressure, ".-", label=f"Size {name}", alpha=.6)
    axes[0].plot(summary.E, summary.pressure, "ko-", label="Equal-size average", linewidth=2)
    axes[0].set(xlabel="Experts E", ylabel="K200 − K12 loss (matched exposure)")
    axes[0].axhline(0, color="grey", linewidth=.6)
    axes[0].legend(fontsize=8)
    for (size, expert), group in response.loc[response.ladder.eq("D")].groupby(["ladder", "E"]):
        group = group.sort_values("FLOPs")
        axes[1].semilogx(group.FLOPs, group.response, label=f"E{expert}")
    axes[1].axhline(0, color="grey", linewidth=.6)
    axes[1].set(xlabel="K200 logical FLOPs (size D)", ylabel="Matched-exposure response")
    axes[1].legend(fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"capacity_response.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    for axis, coverage in zip(axes, (12, 200)):
        for size, group in gains[gains.K.eq(coverage)].groupby("ladder"):
            group = group.sort_values("FLOPs")
            axis.semilogx(group.FLOPs, group.gain, ".-", label=f"Size {size}")
        axis.set(title=f"K{coverage}, common12", xlabel="Logical FLOPs", ylabel="E1 minus E32 loss")
        axis.legend(fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"equal_compute_expert_gains.{suffix}", dpi=180)
    plt.close(fig)
