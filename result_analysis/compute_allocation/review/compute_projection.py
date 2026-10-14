"""Review plots of Stage 1 adjusted loss against language-attributed compute.

Run from the study root with PYTHONPATH=src:. and the bridge-train interpreter.
Inputs are verified by the existing primary-paper release loader; no fits or
training evidence are changed. This directory is an exploratory analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from result_analysis.capacity_response.tables import (
    RELEASE,
    RELEASE_MANIFEST_SHA256,
    load_primary,
    load_release_table,
    matched_pressure,
    sha256_file,
    summarize,
    validate_primary_support,
)

COLORS = {1: "#666666", 12: "#4B0D78", 50: "#0072B2", 200: "#DF6835"}
MARKERS = {1: "o", 4: "s", 16: "^", 32: "D"}
NAMES = {
    "cmn_Hani": "Mandarin", "deu_Latn": "German", "eng_Latn": "English",
    "fra_Latn": "French", "hin_Deva": "Hindi", "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese", "pol_Latn": "Polish", "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian", "spa_Latn": "Spanish", "vie_Latn": "Vietnamese",
}
X_LABEL = r"Language-attributed logical compute $F_t=6N_{\mathrm{act}}D_t$ (EFLOPs)"
MEAN_X_LABEL = r"Mean language-attributed compute $\overline{F_t}=F_{\mathrm{total}}/K$ (EFLOPs)"
TOTAL_X_LABEL = r"Total logical training compute $F=6N_{\mathrm{act}}D$ (EFLOPs)"
STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold",
}


def legend(fig: plt.Figure, y: float) -> None:
    handles = [
        Line2D([], [], color=color, lw=3, label=f"K = {k}")
        for k, color in COLORS.items()
    ] + [
        Line2D([], [], color="#333333", marker=marker, ls="none", ms=6,
               label=f"E = {e}" + (" (dense)" if e == 1 else ""))
        for e, marker in MARKERS.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=8, frameon=False, columnspacing=1.7, handlelength=1.3)


def style(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.16)
    ax.tick_params(axis="both", labelsize=9)


def draw_compute_points(
    ax: plt.Axes, frame: pd.DataFrame, column: str, *, means: bool = False,
) -> None:
    loss = "mean_adjusted_loss" if means else "adjusted_loss"
    drawn = 0
    for k in reversed(COLORS):
        for e, marker in MARKERS.items():
            rows = frame.loc[frame.K.eq(k) & frame.E.eq(e)]
            ax.scatter(rows[column], rows[loss], c=COLORS[k], marker=marker,
                       s=38 if means else 12, alpha=0.82 if means else 0.22,
                       linewidths=0.35 if means else 0,
                       edgecolors="white" if means else None,
                       rasterized=not means)
            drawn += len(rows)
    assert drawn == len(frame)
    style(ax)


def pooled_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, column, title in zip(axes, ("F_t_eflops", "F_total_eflops"), (
        "Your proposed projection: target-language tokens",
        "Budget view: all training tokens",
    )):
        draw_compute_points(ax, frame, column)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(X_LABEL if column == "F_t_eflops" else
                      r"Total logical training compute $F=6N_{\mathrm{act}}D$ (EFLOPs)")
    axes[0].set_ylabel("Unigram-adjusted validation loss (nats; lower is better)")
    fig.suptitle("All 26,300 trained-language observations", y=0.98, fontsize=16)
    legend(fig, 0.94)
    fig.text(0.5, 0.035,
             "200 languages | 4 active sizes | 400 cooled endpoints from 64 shared trunks | "
             "One main-grid seed\n"
             "Each panel contains every observed row. K200 supplies 76% of points; "
             "pooled color separation does not isolate coverage effects.",
             ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=0.79, bottom=0.20, left=0.065, right=0.99, wspace=0.08)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"all_languages_compute_projection.{suffix}", dpi=190)
    plt.close(fig)


def checkpoint_means(frame: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, float]:
    """One equally weighted trained-language mean per uniquely identified readout."""
    groups = frame.groupby("readout_id", sort=True)
    fixed = ["trajectory_id", "ladder", "E", "K", "N_act", "N_total", "D_total", "F_total_eflops"]
    assert groups[fixed].nunique().eq(1).all().all()
    means = groups.agg(
        trajectory_id=("trajectory_id", "first"), ladder=("ladder", "first"),
        E=("E", "first"), K=("K", "first"), N_act=("N_act", "first"),
        N_total=("N_total", "first"), D_total=("D_total", "first"),
        language_count=("language", "nunique"),
        mean_adjusted_loss=("adjusted_loss", "mean"),
        mean_raw_nll=("raw_nll", "mean"), mean_D_t=("D_t", "mean"),
        mean_F_t_eflops=("F_t_eflops", "mean"),
        F_total_eflops=("F_total_eflops", "first"),
    ).reset_index()
    assert len(means) == 400 and means.readout_id.is_unique
    assert means.language_count.eq(means.K).all()
    # This is a mean over the realized exposures, not uniform sampling per language.
    assert np.allclose(means.mean_F_t_eflops, means.F_total_eflops / means.K,
                       rtol=1e-14, atol=0)
    declared = load_release_table(root, "validation_aggregates", (
        "readout_id", "aggregate_kind", "cohort", "status", "metric_name",
        "metric_value", "aggregate_language_count",
    ))
    declared = declared.loc[
        declared.aggregate_kind.eq("declared_metric") & declared.cohort.eq("primary")
    ]
    assert len(declared) == 400 and declared.status.eq("complete").all()
    assert declared.metric_name.eq(
        "trained_equal_language_atlas_tao_unigram_adjusted_loss"
    ).all()
    checked = means.merge(declared, on="readout_id", validate="one_to_one")
    assert len(checked) == 400
    assert checked.language_count.eq(checked.aggregate_language_count).all()
    error = float((checked.mean_adjusted_loss - checked.metric_value).abs().max())
    assert error < 2e-12, error
    return means, error


def checkpoint_plots(frame: pd.DataFrame, means: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, column, title, xlabel in zip(axes,
        ("mean_F_t_eflops", "F_total_eflops"),
        ("Mean compute per trained language", "Total training compute"),
        (MEAN_X_LABEL, TOTAL_X_LABEL),
    ):
        draw_compute_points(ax, means, column, means=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Mean unigram-adjusted validation loss (nats; lower is better)")
    fig.suptitle("Mean adjusted loss at each of the 400 checkpoints", y=0.98, fontsize=16)
    legend(fig, 0.94)
    fig.text(0.5, 0.035,
             "Each point averages the K trained languages equally at one checkpoint; "
             "language membership changes with K.\n"
             "Left: arithmetic mean of the realized language compute values. "
             "Right: the checkpoint's full training compute. One main-grid seed.",
             ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=0.79, bottom=0.20, left=0.065, right=0.99, wspace=0.08)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"checkpoint_mean_compute_projection.{suffix}", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex="col", sharey=True)
    for row, data in enumerate((frame, means)):
        is_mean = row == 1
        for col, coordinate in enumerate(("mean_F_t_eflops" if is_mean else "F_t_eflops",
                                           "F_total_eflops")):
            ax = axes[row, col]
            draw_compute_points(ax, data, coordinate, means=is_mean)
            ax.set_title(("Checkpoint means" if is_mean else "Individual languages")
                         + (" — language compute" if col == 0 else " — total compute"), fontsize=11)
            ax.set_xlabel((MEAN_X_LABEL if is_mean else X_LABEL) if col == 0 else TOTAL_X_LABEL)
            ax.tick_params(axis="x", labelbottom=True)
        axes[row, 0].set_ylabel(("Mean adjusted loss" if is_mean else "Adjusted loss")
                               + " (nats; lower is better)")
    fig.suptitle("Individual languages and checkpoint means", y=0.985, fontsize=16)
    legend(fig, 0.958)
    fig.text(0.5, 0.025,
             "Top: 26,300 language observations. Bottom: 400 equal-language checkpoint means. "
             "Color = K; shape = E.\n"
             "Columns share x limits; all panels share y limits. "
             "Checkpoint means average different language sets at different K.",
             ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=0.87, bottom=0.11, left=0.065, right=0.99,
                        wspace=0.08, hspace=0.30)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"language_vs_checkpoint_mean_compute_projection.{suffix}", dpi=190)
    plt.close(fig)


def language_plot(frame: pd.DataFrame, ladder: str, languages: list[str]) -> plt.Figure:
    selected = frame.loc[frame.ladder.eq(ladder) & frame.language.isin(languages)]
    assert selected.N_act.nunique() == 1
    n_act = selected.N_act.iloc[0]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5), sharex=True, sharey=False)
    drawn = 0
    for language, ax in zip(languages, axes.flat):
        subset = selected.loc[selected.language.eq(language)]
        for k in reversed(COLORS):
            for e, marker in MARKERS.items():
                rows = subset.loc[subset.K.eq(k) & subset.E.eq(e)].sort_values("F_t_eflops")
                if rows.empty:
                    continue
                assert rows.trajectory_id.nunique() == 1
                ax.plot(rows.F_t_eflops, rows.adjusted_loss, color=COLORS[k],
                        lw=0.6, alpha=0.3, zorder=1)
                ax.scatter(rows.F_t_eflops, rows.adjusted_loss, color=COLORS[k],
                           marker=marker, s=21, alpha=0.8, linewidths=0.35,
                           edgecolors="white", zorder=2)
                drawn += len(rows)
        ax.set_title(f"{NAMES[language]}  ({language})", fontsize=10)
        style(ax)
    assert drawn == len(selected)
    fig.suptitle(f"Compare K and E within each language at {n_act / 1e6:.1f}M active parameters",
                 y=0.99, fontsize=15)
    legend(fig, 0.955)
    fig.supxlabel(X_LABEL, y=0.065)
    fig.supylabel("Unigram-adjusted validation loss (nats; lower is better)", x=0.013)
    fig.text(0.5, 0.022,
             f"Size {ladder} | {len(selected):,} observed rows | "
             "12 shared training languages | K1 is English only\n"
             "Vertical scales differ by language. Points are measured endpoints; "
             "faint lines join endpoints sharing a trunk. No fitted curves or confidence bands.",
             ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=0.875, bottom=0.115, left=0.075, right=0.99,
                        hspace=0.35, wspace=0.29)
    return fig


def matched_size_example(frame: pd.DataFrame) -> list[dict]:
    """English/K12/E1 at the geometric midpoint of all four sizes' shared support."""
    selected = frame.loc[
        frame.language.eq("eng_Latn") & frame.K.eq(12) & frame.E.eq(1)
    ]
    bounds = selected.groupby("ladder").F_t_eflops.agg(["min", "max"])
    lower, upper = bounds["min"].max(), bounds["max"].min()
    assert lower < upper
    target = float(np.sqrt(lower * upper))
    rows = []
    for ladder, group in selected.groupby("ladder"):
        group = group.sort_values("F_t_eflops")
        x = group.F_t_eflops.to_numpy(float)
        if not x[0] <= target <= x[-1]:
            continue
        right = np.searchsorted(x, target)
        brackets = group.iloc[[right - 1, right]]
        n_act = int(group.N_act.iloc[0])
        rows.append({
            "language": "eng_Latn", "K": 12, "E": 1, "ladder": ladder,
            "F_t_eflops": target, "N_act": n_act,
            "D_t_at_target": target * 1e18 / (6.0 * n_act),
            "interpolated_adjusted_loss": float(np.interp(
                np.log(target), np.log(x), group.adjusted_loss.to_numpy(float))),
            "left_readout": str(brackets.readout_id.iloc[0]),
            "right_readout": str(brackets.readout_id.iloc[1]),
        })
    assert len(rows) == 4
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = load_primary(root)
    # Compute exceeds signed int64 at larger endpoints; cast BEFORE multiplying.
    frame["F_t_eflops"] = 6.0 * frame.N_act.astype(float) * frame.D_t.astype(float) / 1e18
    frame["F_total_eflops"] = 6.0 * frame.N_act.astype(float) * frame.D_total.astype(float) / 1e18
    frame["realized_language_share"] = frame.D_t / frame.D_total
    assert (frame[["F_t_eflops", "F_total_eflops"]] > 0).all().all()
    assert np.isfinite(frame[["F_t_eflops", "F_total_eflops"]].to_numpy()).all()
    assert not frame.duplicated(["readout_id", "language"]).any()
    assert np.allclose(frame.F_t_eflops, frame.realized_language_share * frame.F_total_eflops,
                       rtol=1e-14, atol=0)
    by_readout = frame.groupby("readout_id")
    assert np.array_equal(by_readout.D_t.sum(), by_readout.D_total.first())
    assert np.allclose(by_readout.F_t_eflops.sum(), by_readout.F_total_eflops.first(), rtol=1e-14)
    unigram = frame.raw_nll - frame.adjusted_loss
    ref_spread = unigram.groupby(frame.language).agg(lambda x: x.max() - x.min()).max()
    assert ref_spread < 2e-12
    frame.to_csv(output / "points.csv", index=False)
    pooled_plot(frame, output)
    means, checkpoint_mean_error = checkpoint_means(frame, root)
    means.to_csv(output / "checkpoint_means.csv", index=False)
    checkpoint_plots(frame, means, output)
    languages = sorted(frame.loc[frame.K.eq(12), "language"].unique())
    assert len(languages) == 12
    with PdfPages(output / "common12_by_language_all_sizes.pdf") as pdf:
        for ladder in sorted(frame.ladder.unique()):
            fig = language_plot(frame, ladder, languages)
            pdf.savefig(fig)
            fig.savefig(output / f"common12_by_language_{ladder}.png", dpi=160)
            plt.close(fig)

    pressure = matched_pressure(frame)
    validate_primary_support(pressure)
    _, summary = summarize(pressure)
    pressure.to_csv(output / "matched_pressure_points.csv", index=False)
    summary.to_csv(output / "matched_pressure_summary.csv", index=False)
    examples = matched_size_example(frame)
    pd.DataFrame(examples).to_csv(output / "same_compute_different_sizes.csv", index=False)
    manifest = {
        "status": "exploratory_plot_review", "release": str(RELEASE),
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "source_verification": "load_primary checks manifest, three table hashes, counts and joins; checkpoint means also verify validation_aggregates",
        "loader_sha256": sha256_file(root / "result_analysis/capacity_response/tables.py"),
        "generator_sha256": sha256_file(Path(__file__)),
        "observations": len(frame), "trajectories": frame.trajectory_id.nunique(),
        "endpoints": frame.readout_id.nunique(), "languages": frame.language.nunique(),
        "rows_by_K": {str(k): len(g) for k, g in frame.groupby("K")},
        "active_parameters_by_ladder": frame.groupby("ladder").N_act.first().to_dict(),
        "F_t_eflops_range": [frame.F_t_eflops.min(), frame.F_t_eflops.max()],
        "F_total_eflops_range": [frame.F_total_eflops.min(), frame.F_total_eflops.max()],
        "unigram_adjusted_loss_range": [frame.adjusted_loss.min(), frame.adjusted_loss.max()],
        "exact_target_exposure_sum_equals_total_at_every_endpoint": True,
        "checkpoint_mean_definition": "Arithmetic mean of adjusted loss over all K trained languages at each readout",
        "checkpoint_mean_left_coordinate": "Arithmetic mean of realized F_t; equals F_total/K by exposure closure",
        "checkpoint_mean_rows": len(means),
        "checkpoint_mean_max_difference_from_released_primary_metric": checkpoint_mean_error,
        "matched_pressure_summary": summary.to_dict(orient="records"),
        "same_compute_different_sizes": examples,
        "caveats": [
            "F_t attributes logical compute by token language; it excludes other-language training.",
            "Constant F_t does not fix N_act and D_t separately.",
            "Adjusted loss removes a fixed unigram reference, not all language-specific variation.",
            "All main-grid rows share one seed; endpoints and languages are nested observations.",
            "Expert recipes differ in stored parameters and prescribed learning rate.",
            "Logical compute does not measure GPU-hours or systems overhead.",
        ],
        "generated_files": {
            p.name: sha256_file(p) for p in sorted(output.iterdir())
            if p.suffix in {".csv", ".png", ".pdf"}
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "generated_files"}, indent=2))


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        main()
