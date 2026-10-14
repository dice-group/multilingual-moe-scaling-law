"""Review figures for fixed-law and observation-based allocation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def render(winners, observed, output):
    colors = dict(zip((1, 4, 16, 32), plt.colormaps["inferno"]([.12, .36, .60, .79])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, k in zip(axes[:2], (1, 200)):
        for cap, group in winners.query("fold == 'E32' and cap in [400000000,1200000000,5100000000]").loc[lambda f: f.K.eq(k)].groupby("cap"):
            group = group.sort_values("F_EF").reset_index(drop=True)
            cuts = np.r_[0, np.flatnonzero((group.E.to_numpy()[1:] != group.E.to_numpy()[:-1]) |
                (group.horizon_extrapolation.to_numpy()[1:] != group.horizon_extrapolation.to_numpy()[:-1]))+1, len(group)]
            for start, stop in zip(cuts[:-1], cuts[1:]):
                part = group.iloc[start:min(stop+1, len(group))]
                ax.plot(part.F_EF*1e18, part.loss, color=colors[int(part.E.iloc[0])],
                        ls="--" if part.horizon_extrapolation.iloc[0] else "-")
            ax.annotate(f"{cap/1e9:g}B cap", (group.F_EF.iloc[-1]*1e18, group.loss.iloc[-1]), ha="right")
        ax.set(xscale="log", xlabel="Logical training FLOPs", ylabel="Predicted native-language mean loss", title=f"M6-ND, K={k}")
    for k, style in ((12, "--"), (200, "-")):
        part = observed.query("method == 'smooth'").loc[lambda f: f.K.eq(k)]
        wide = part.pivot(index="F", columns="config", values="loss")
        axes[2].plot(wide.index*1e18, wide["A/E16"]-wide["B/E4"], ls=style, label=f"K={k}")
    axes[2].axhline(0, color=".6", lw=.7)
    axes[2].set(xscale="log", xlabel="Logical training FLOPs", ylabel="A/E16 minus B/E4 loss", title="Observed common12, 0.4B cap")
    axes[2].legend(frameon=False)
    fig.suptitle("Capacity allocation; dashed fitted curves exceed study token horizons", fontsize=11)
    fig.legend(handles=[Line2D([], [], color=color, label=f"Fitted E={e}") for e, color in colors.items()],
               loc="outside lower center", ncols=4, frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"capacity_allocation.{ext}", dpi=150)
    plt.close(fig)


def save_allocation_figure(output, contrasts) -> None:
    """Plot the eight empirical higher-R allocation comparisons."""
    panel = contrasts.sort_values(["pair_id", "F_EFLOPs"]).reset_index(drop=True)
    y = np.arange(len(panel))
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    axis.hlines(y, panel.K12, panel.K200, color="#AAAAAA", lw=1.8, zorder=1)
    axis.scatter(panel.K12, y, marker="s", s=42, color="#4B0D78", label="K=12", zorder=2)
    axis.scatter(panel.K200, y, marker="o", s=45, color="#FF8A61", label="K=200", zorder=2)
    for row_index, row in panel.iterrows():
        axis.annotate(
            f"{row.K200_minus_K12:+.3f}",
            (row.K200, row_index),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#444444",
        )
    labels = [
        f"{row.pair_id} {row.higher_R}–{row.lower_R} · {row.F_EFLOPs:g}F"
        for row in panel.itertuples(index=False)
    ]
    axis.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Higher-$R$ allocation advantage\n(adjusted loss)",
    )
    axis.invert_yaxis()
    axis.legend(loc="lower center", bbox_to_anchor=(0.63, 1.01), ncol=2, frameon=False)
    axis.axvline(0, color="#555555", lw=0.8, ls="--")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"allocation_amplification.{suffix}", dpi=180)
    plt.close(figure)
