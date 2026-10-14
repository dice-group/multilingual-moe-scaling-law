"""Validation visualization with selected and exploratory laws labeled separately."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(table, output, *, main_law="RN13-psi-tied"):
    columns = ["largest_N_act", "E32", "K200_common_languages", "L26_all_grid"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True, layout="constrained")
    names = list(table.index)
    colors = ["#187b80" if name == main_law else "#b5bbc3" for name in names]
    for axis, col in zip(axes, columns):
        axis.barh(names, table[col], color=colors)
        axis.set(title=col.replace("_", " "), xlabel="RMSE (lower is better)")
    axes[0].invert_yaxis()
    fig.suptitle("Final proposed RN14 stored-reuse law (teal)" if main_law == "RN14-stored-reuse"
                 else "Historical inner-selected RN13 law (teal)")
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"validation.{suffix}", dpi=180)
    plt.close(fig)
