"""Audited, descriptive comparison; no fitted model or threshold is selected here."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from result_analysis.scaling_law_fit import residuals as analysis
from .. import extrapolation_diagnostics as diagnosis
from .. import execution as core_execution
from . import model, planning as p
from .execution import artifact, read_artifact


def features(rows):
    return SimpleNamespace(N=rows.N_act.to_numpy(), D=rows.D_t.to_numpy(), E=rows.E.to_numpy(), K=rows.K.to_numpy(),
        D_family=rows.D_family.to_numpy(), D_script=rows.D_script.to_numpy(), D_other=rows.D_other.to_numpy(),
        language_ids=tuple(rows.language))


def candidate_rows(plan, reduction):
    frames = []
    for fold in p.FOLDS:
        manifest = plan["folds"][fold]["manifest"]
        frame = pd.read_parquet(manifest["input"]["fit_rows_path"])
        member = manifest["membership"]
        for split in ("train", "test"):
            rows = frame[frame.trajectory_id.isin(member[f"{split}_trajectory_ids"])].copy()
            if member[f"{split}_readout_ids"]:
                rows = rows[rows.readout_id.isin(member[f"{split}_readout_ids"])].copy()
            if split == "test" and member["score_languages"]:
                rows = rows[rows.language.isin(member["score_languages"])].copy()
            rows["prediction"] = model.predict(features(rows), reduction["folds"][fold]["selected"])
            frames.append(diagnosis.annotate(rows, model=model.MODEL, fold=fold, split=split, delta=manifest["fit"]["delta"]))
    return pd.concat(frames, ignore_index=True)


def metrics_and_bootstrap(frame, replicates=10000, seed=20260905):
    rows = frame[frame.split.eq("test")]
    keys = ["fold", "trajectory_id", "readout_id", "language"]
    reference = None
    for _, group in rows.groupby("model"):
        if group.duplicated(keys).any():
            raise ValueError("duplicate heldout observation")
        ids = set(map(tuple, group[keys].to_numpy()))
        if reference is not None and ids != reference:
            raise ValueError("baseline and candidate heldout memberships differ")
        reference = ids
    per = rows.groupby(["model", "fold", "trajectory_id"]).squared_error.mean()
    union = sorted(rows.trajectory_id.unique())
    counts = np.random.default_rng(seed).multinomial(len(union), np.full(len(union), 1/len(union)), size=replicates)
    metrics, samples = {}, {}
    for name in sorted(rows.model.unique()):
        metrics[name], fold_samples = {}, []
        for fold in p.FOLDS:
            values = per.loc[(name, fold)].reindex(union)
            present = values.notna().to_numpy()
            denominator = counts[:, present].sum(axis=1)
            if (denominator == 0).any():
                raise ValueError("bootstrap draw has no fold support")
            fold_samples.append(np.sqrt(counts[:, present] @ values[present].to_numpy()/denominator))
            metrics[name][fold] = float(np.sqrt(values[present].mean()))
        metrics[name]["mean"] = float(np.mean(list(metrics[name].values())))
        samples[name] = np.mean(fold_samples, axis=0)
    intervals = [{"reference": name, "candidate": model.MODEL,
        "difference": metrics[model.MODEL]["mean"]-metrics[name]["mean"],
        "p025": float(np.quantile(samples[model.MODEL]-values, .025)),
        "p975": float(np.quantile(samples[model.MODEL]-values, .975))}
        for name, values in samples.items() if name != model.MODEL]
    return {"metrics": metrics, "paired_intervals": intervals, "replicates": replicates,
        "unit": "whole trajectory; common bootstrap multiplicities across overlapping folds and models",
        "interpretation": "descriptive intervals conditional on fitted laws; not search-corrected, not independent-seed uncertainty"}


def pressure_curves(frame):
    from result_analysis.capacity_response.tables import matched_pressure
    subset = frame[frame.fold.eq("largest_N_act") & frame.E.eq(32) & frame.model.isin(["M6", "J0", model.MODEL])]
    curves = []
    keys = ["ladder", "E", "language", "target_trajectory_id", "target_readout_id", "D_t"]
    for name, rows in subset.groupby("model"):
        observed = matched_pressure(rows)
        predicted_rows = rows.copy()
        predicted_rows["adjusted_loss"] = rows.prediction
        predicted_rows["raw_nll"] = rows.prediction+rows.unigram_nll
        predicted = matched_pressure(predicted_rows)[keys+["pressure"]].rename(columns={"pressure": "predicted_pressure"})
        merged = observed.merge(predicted, on=keys, validate="one_to_one")
        merged["model"] = name
        merged["pressure_error"] = merged.predicted_pressure-merged.pressure
        merged["support"] = "heldout largest model" if set(merged.ladder) == {"D"} else "D heldout; A-C training trajectories"
        curves.append(merged)
    return pd.concat(curves, ignore_index=True)


def gates(statistics, frame, sizes, reduction):
    m = statistics["metrics"][model.MODEL]
    limits = p.PROTOCOL["gates"]
    checks = {
        "mean_rmse": m["mean"] < limits["mean_rmse_below_m6"],
        "largest_rmse": m["largest_N_act"] < limits["largest_rmse_below_j0"],
        "e32_rmse": m["E32"] <= limits["e32_rmse_at_most_m6"],
        "k200_rmse": m["K200_common_languages"] <= limits["k200_rmse_at_most_m6"],
        "positive_zeta_all_folds": all(reduction["folds"][f]["selected"]["coordinates"][14] > 1e-8 for f in p.FOLDS),
    }
    short = sizes[sizes.left_ladder.eq("C") & sizes.right_ladder.eq("D") & sizes.D_total.eq(p.PROTOCOL["short_total_tokens"])]
    if short.model.duplicated().any() or not {model.MODEL, "M6"} <= set(short.model):
        raise ValueError("missing unambiguous short-horizon size contrast")
    short = short.set_index("model")
    if not short.target_and_component_exposure_matched.all():
        raise ValueError("size contrasts do not match exposure")
    error = abs(short.predicted_size_gain-short.observed_size_gain)
    checks["short_C_D_absolute_gain_error_reduced"] = error[model.MODEL] < error["M6"]
    tails = {}
    for k in (12, 50):
        rows = frame[frame.fold.eq("largest_N_act") & frame.split.eq("test") & frame.E.eq(32)
            & frame.K.eq(k) & frame.D_total.eq(p.PROTOCOL["long_total_tokens"])]
        rmse = {name: analysis.summary(group)["rmse"] for name, group in rows.groupby("model")}
        checks[f"largest_32B_E32_K{k}"] = rmse[model.MODEL] <= rmse["M6"]
        tails[str(k)] = rmse
    return {"passed": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()},
        "short_C_D_absolute_gain_error": {k: float(v) for k, v in error.items()}, "largest_32B_E32_tail_rmse": tails}


def figures(output, frame, pressure):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    heldout = frame[frame.fold.eq("largest_N_act") & frame.split.eq("test") & frame.model.isin(["J0", "M6", model.MODEL])]
    for name, group in heldout.groupby("model"):
        means = group.groupby(["D_total", "trajectory_id"])[["residual", "squared_error"]].mean().groupby("D_total").mean()
        axes[0].plot(means.index/1e9, means.residual, marker="o", label=name)
        axes[1].plot(means.index/1e9, np.sqrt(means.squared_error), marker="o", label=name)
    for ax, label in zip(axes, ("Prediction minus observation (nats)", "Trajectory-balanced RMSE (nats)")):
        ax.set(xscale="log", xlabel="Total training tokens (billions)", ylabel=label)
        ax.legend()
    axes[0].axhline(0, color="grey", linewidth=.6)
    for suffix in ("png", "pdf"):
        path = output/f"largest_horizon.{suffix}"
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    for ax, ladder in zip(axes, sorted(pressure.ladder.unique())):
        rows = pressure[pressure.ladder.eq(ladder)]
        observed = rows[rows.model.eq("M6")].groupby("D_total").pressure.mean()
        ax.plot(observed.index/1e9, observed, color="black", label="Observed, interpolated reference")
        for name, group in rows.groupby("model"):
            values = group.groupby("D_total").predicted_pressure.mean()
            ax.plot(values.index/1e9, values, linestyle="--", label=name)
        ax.axhline(0, color="grey", linewidth=.5)
        ax.set(title=f"{ladder}: E32, K200−K12", xscale="log", xlabel="K200 total tokens (B)", ylabel="Matched-exposure pressure")
    axes[-1].legend(fontsize=6)
    for suffix in ("png", "pdf"):
        path = output/f"e32_pressure.{suffix}"
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def report(plan_path):
    plan = p.load(plan_path)
    reduction = read_artifact(plan, "reduction.json")
    audits = [read_artifact(plan, f"audits/{i:02d}.json") for i in range(12)]
    for i, audit in enumerate(audits):
        if audit["reduction_sha256"] != reduction["sha256"] or audit["task"] != reduction["audit_tasks"][i]:
            raise ValueError("audit identity changed")
    audit_healthy = all(a["status"] == "healthy" for a in audits)
    output = artifact(plan, "reports")
    if output.exists():
        raise ValueError("report directory already exists; reports are create-only")
    output.mkdir()
    base = {"plan_sha256": plan["sha256"], "reduction_sha256": reduction["sha256"],
        "audit_sha256": [a["sha256"] for a in audits], "numerical_audit_healthy": audit_healthy,
        "evidence": p.PROTOCOL["evidence"], "zeta": {f: reduction["folds"][f]["selected"]["coordinates"][14] for f in p.FOLDS}}
    if not audit_healthy:
        return p.write_sealed(output/"report.json", base | {"status": "numerically_blocked",
            "heldout_scores_computed": False, "failed_audits": [a["task"]["task_id"] for a in audits if a["status"] != "healthy"]})
    frame = pd.concat([pd.read_parquet(p.diagnosis.bound_file(plan["baseline_predictions"])), candidate_rows(plan, reduction)], ignore_index=True)
    statistics = metrics_and_bootstrap(frame)
    sizes = analysis.matched_sizes(frame, replicates=p.PROTOCOL["bootstrap_replicates"], seed=p.PROTOCOL["bootstrap_seed"])
    decision = gates(statistics, frame, sizes, reduction)
    pressure = pressure_curves(frame)
    paths = []
    for name, table in (("residual_groups", analysis.residual_groups(frame)), ("matched_sizes", sizes),
                        ("learning_slopes", analysis.learning_slopes(frame)), ("e32_pressure", pressure)):
        path = output/f"{name}.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    path = output/"predictions.parquet"
    frame.to_parquet(path, index=False)
    paths.append(path)
    paths += figures(output, frame, pressure)
    lines = ["# M6-ND development comparison", "", "Numerical audits: 12/12 healthy.",
        "", f"Frozen improvement gate: {'PASS' if decision['passed'] else 'FAIL'}.", "",
        "| Law | Largest N | E32 | K200 | Mean |", "|---|---:|---:|---:|---:|"]
    for name, m in statistics["metrics"].items():
        lines.append(f"| {name} | {m[p.FOLDS[0]]:.8f} | {m[p.FOLDS[1]]:.8f} | {m[p.FOLDS[2]]:.8f} | {m['mean']:.8f} |")
    lines += ["", "Labels: C0 is the Chinchilla-style total-token control; J0 is the audited JointMoE baseline. "
        "M0 is a target-token baseline; M1/M2/M6 are study-developed laws. No faithful ATLAS refit is claimed by this comparison.",
        "", "E32 pressure is a multilingual K200-minus-K12 contrast, not prediction error. Both observed and predicted "
        "curves use the same log-exposure interpolation on observed common support; no reference extrapolation. "
        "Only ladder D is size-heldout. The secondary curve check does not tune or override the frozen primary gate.",
        "", "Already-inspected holdouts, nested checkpoints/languages, and one-seed recipe-conditional data: "
        "even a pass is development evidence. A fresh independent confirmation is needed before a stronger claim.", ""]
    core_execution.write_text_once(output/"summary.md", "\n".join(lines))
    paths.append(output/"summary.md")
    return p.write_sealed(output/"report.json", base | {"status": "complete", "statistics": statistics,
        "decision": decision, "outputs": [p.binding(path) for path in paths]})
