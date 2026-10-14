"""Observed study inventory, architecture geometry, and training compute."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from result_analysis import evidence as io
from result_analysis.checks import Checks


def run(evidence, output):
    evidence.require_current()
    output = io.output_directory(output, evidence)
    checks = Checks()
    rows = evidence.table("study/primary_fits/fit_rows.parquet")
    keys = ["trajectory_id", "readout_id", "language"]
    checks.require("unique observed rows", not rows.duplicated(keys).any())
    checks.close("paper observation/trajectory/readout counts", [len(rows), rows.trajectory_id.nunique(), rows.readout_id.nunique()],
                 [26300, 64, 400], source="paper/my_content.tex")
    checks.close("unigram-adjusted loss definition", rows.adjusted_loss, rows.raw_nll-rows.unigram_nll)
    checks.close("exposure components partition total tokens", rows.D_t+rows.D_family+rows.D_script+rows.D_other,
                 rows.D_total, atol=1e-3)
    geometry = evidence.table("scoring/grid.parquet")[["ladder", "E", "N_act", "N_total"]].drop_duplicates().sort_values(["ladder", "E"])
    checks.require("sixteen architecture cells", len(geometry) == 16 and not geometry.duplicated(["ladder", "E"]).any())
    geometry.to_csv(output / "architecture_geometry.csv", index=False)
    group = ["trajectory_id", "readout_id", "ladder", "E", "K", "N_act", "D_total"]
    endpoints = rows.groupby(group, as_index=False).agg(mean_raw_nll=("raw_nll", "mean"),
                                                       mean_adjusted_loss=("adjusted_loss", "mean"), languages=("language", "nunique"))
    checks.close("native languages at every endpoint", endpoints.languages, endpoints.K)
    endpoints["FLOPs"] = 6*endpoints.N_act*endpoints.D_total
    endpoints.to_csv(output / "compute_overview.csv", index=False)
    trajectories = endpoints.sort_values("D_total").groupby("trajectory_id").agg(
        ladder=("ladder", "first"), E=("E", "first"), K=("K", "first"), N_act=("N_act", "first"),
        min_tokens=("D_total", "min"), max_tokens=("D_total", "max"), readouts=("readout_id", "size"))
    trajectories.to_csv(output / "trajectory_inventory.csv")
    readouts = evidence.table("release/tables/readouts.parquet")
    attribution = evidence.table("release/tables/readout_compute_attribution.parquet")
    training = evidence.table("release/tables/training_summary.parquet")
    systems = readouts[["readout_id", "ladder", "experts", "language_count", "nominal_tokens", "training_job_id"]].merge(
        attribution[["readout_id", "logical_readout_attributed_productive_gpu_hours"]], on="readout_id", validate="one_to_one").merge(
        training[["training_job_id", "mean_mfu_pct_of_configured_peak", "mean_tokens_per_second_per_gpu"]],
        on="training_job_id", validate="many_to_one")
    gpu = systems.pivot(index=["ladder", "language_count", "nominal_tokens"], columns="experts",
                        values="logical_readout_attributed_productive_gpu_hours").dropna()
    checks.require("100 matched physical-cost comparisons", len(gpu) == 100)
    ratios = {e: float((gpu[e]/gpu[1]).median()) for e in (4, 16, 32)}
    mfu = systems.groupby("experts").mean_mfu_pct_of_configured_peak.mean()
    checks.close("paper GPU-hour ratios", list(ratios.values()), [1.17, 1.44, 1.69], atol=.0051)
    checks.close("paper MFU endpoints", mfu.loc[[1, 32]], [35.7, 22.0], atol=.051)
    systems.to_csv(output / "physical_compute_by_readout.csv", index=False)
    pd.DataFrame({"E": list(ratios), "median_gpu_hour_ratio_vs_E1": list(ratios.values())}).to_csv(output / "physical_compute_ratios.csv", index=False)
    evidence.table("release/tables/unique_physical_study_compute.parquet").to_csv(output / "retained_unique_physical_study_compute.csv", index=False)
    evidence.table("study/paper_exports/language_tier_metadata.csv").to_csv(output / "language_metadata.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True, layout="constrained")
    colors = dict(zip((1, 12, 50, 200), plt.colormaps["viridis"]([.1, .35, .6, .85])))
    markers = dict(zip((1, 4, 16, 32), ("o", "s", "^", "D")))
    for ax, (ladder, frame) in zip(axes, endpoints.groupby("ladder")):
        for (e, k), part in frame.groupby(["E", "K"]):
            part = part.sort_values("FLOPs")
            ax.plot(part.FLOPs, part.mean_raw_nll, color=colors[k], marker=markers[e], markersize=2, lw=.8)
        ax.set(xscale="log", xlabel="Logical training FLOPs", title=f"{ladder}: {frame.N_act.iloc[0]/1e6:.1f}M active")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Mean loss over each K's trained languages")
    handles = [Line2D([], [], color=color, label=f"K={k}") for k, color in colors.items()]
    handles += [Line2D([], [], color=".4", marker=marker, ls="", label=f"E={e}") for e, marker in markers.items()]
    fig.legend(handles=handles, loc="outside lower center", ncols=8, frameon=False, fontsize=8)
    fig.suptitle("64 trajectories / 400 dependent endpoints; native language sets differ across K", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"compute_overview.{ext}", dpi=150)
    plt.close(fig)
    claims = {"training_trajectories": 64, "dependent_endpoints": 400, "language_readouts": len(rows),
              "physical_gpu_hour_ratios": ratios, "mean_mfu_percent": mfu.to_dict(),
              "architecture_cells": len(geometry), "checks_passed": checks.save(output),
              "reproduction_scope": "observed grid inventory, geometry, all compute-overview points and matched physical-cost summaries",
              "caveats": ["native-language means are not matched coverage contrasts", "logical compute is 6*N_act*D_total, not measured GPU-hours",
                          "physical costs derive from sealed training telemetry; raw logs are not reparsed"]}
    return io.finish(output, evidence, "study_setup", claims, [Path(__file__), Path(io.__file__)])


if __name__ == "__main__":
    from result_analysis.cli import module_main
    module_main(run, __doc__)
