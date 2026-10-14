"""Mean recovery, individual loss changes, and distinct compute objectives."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(frame, summary, exposure, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    means = frame[frame.objective.eq("mean")].sort_values("loss_delta")
    colors = plt.colormaps["inferno"]([.25, .65])
    axes[0].bar(range(len(means)), means.loss_delta,
                color=[colors[0] if x <= 1e-9 else colors[1] for x in means.loss_delta])
    axes[0].axhline(0, color="grey", lw=.6)
    axes[0].set(xlabel="Incumbent languages (sorted)", ylabel="Expanded minus reference loss",
                title=f"Mean recovery: {int(summary.loc['mean', 'protected_languages'])} of {len(means)} languages protected")
    axes[1].bar(["Exposure", "Mean loss", "All languages"],
                [exposure, summary.loc["mean", "compute_multiplier"], summary.loc["all", "compute_multiplier"]],
                color=plt.colormaps["inferno"]([.2, .5, .75]))
    axes[1].set(ylabel="Logical-compute multiplier", title="Different preservation objectives")
    for ext in ("pdf", "png"):
        fig.savefig(output / f"language_preservation.{ext}", dpi=180)
    plt.close(fig)
