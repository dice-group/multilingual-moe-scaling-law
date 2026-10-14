"""Recompute downstream paper estimators from certified full-refit coefficients."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from result_analysis import evidence
from .dependency_replay import Replay, write_json, sha
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs, protocol, recovery, runner
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import diagnostics
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow.reporting import metrics_and_bootstrap


def run(profile_path):
    import json
    from types import SimpleNamespace
    profile_path = Path(profile_path).resolve()
    cfg = json.loads(profile_path.read_text())
    args = SimpleNamespace(**{k:Path(v).resolve() for k,v in cfg["replay"].items()})
    if args.output.exists():
        raise ValueError("review output is create-only")
    work = Replay(args)
    work.bind(profile_path); work.bind(Path(__file__))
    plan = runner.load(args.refit_plan)
    report = inputs.read_sealed(args.refit_report)
    if not report["all_cells_certified"] or not report["all_scoring_complete"]:
        raise ValueError("complete certification and scoring required before downstream review")
    overrides = recovery.verified_overrides(args.refit_plan,
        inputs.local_path(args.refit_plan.parent, report["recovery"]) if report.get("recovery") else None)
    work.composition(); work.policy(); work.knots()
    work.finish("core")
    # Same seed and paired WHOLE-CONFIGURATION draws as the original interval.
    predictions = work.read(args.refit_report.parent / "predictions.parquet")
    primary = predictions[predictions.arm.eq("lbfgsb_box") & predictions.case.eq("grid") & predictions.fold.isin(protocol.PRIMARY)].copy()
    primary["squared_error"] = (primary.prediction-primary.adjusted_loss)**2
    primary["split"] = "test"
    bootstrap = metrics_and_bootstrap(primary)
    write_json(args.output / "configuration_bootstrap.json", bootstrap)
    work.bind(Path(metrics_and_bootstrap.__code__.co_filename))
    # Recompute the adverse contrast and size-duration diagnostics as well.
    contrasts, sizes = [], []
    for task in plan["tasks"]:
        if (task["stage"] != "certify" or task["arm"] != "lbfgsb_box" or
                task["model"] not in ("RN13-psi-tied", "RN14-no-ridge", "RN14-stored-reuse") or
                task["fold"] not in ("largest_N_act", "all_grid")):
            continue
        i = task["id"]
        path, result = overrides.get(i, (args.refit_plan.parent / "results" / f"{i:05d}.json",
                                         runner.receipt(args.refit_plan, plan, i)))
        work.bind(path)
        rows = diagnostics.contributions(task["model"], result["selected"], work.grid)
        for regime in ("all_support", "pre_repeat", "repeated"):
            contrasts.extend(dict(model=task["model"], fit_fold=task["fold"], regime=regime, **row)
                             for row in diagnostics.expert_coverage(rows, regime))
        sizes.extend(dict(model=task["model"], fit_fold=task["fold"], **row)
                     for row in diagnostics.size_duration(rows))
    contrasts, sizes = pd.DataFrame(contrasts), pd.DataFrame(sizes)
    contrasts["squared_error"] = (contrasts.prediction-contrasts.adjusted_loss)**2
    summary = contrasts.groupby(["model", "fit_fold", "regime"]).agg(
        observed=("adjusted_loss", "mean"), predicted=("prediction", "mean"),
        mse=("squared_error", "mean"), languages=("language", "nunique"), points=("language", "size")).reset_index()
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    work.save("contrast_points.csv", contrasts); work.save("contrast_summary.csv", summary)
    work.save("size_duration_points.csv", sizes)
    work.bind(Path(diagnostics.__file__))
    work.finish("diagnostics")
    work.continuous(); work.finish("continuous")
    from result_analysis.compute_allocation import doubling
    doubling_cfg = json.loads(Path(cfg["doubling_config"]).read_text())
    doubling_cfg.update(campaign_plan=cfg["coverage_plan"], refit_plan=str(args.refit_plan),
                        refit_report=str(args.refit_report), output_root=str(args.output / "doubling"))
    work.bind(Path(cfg["doubling_config"]).resolve()); work.bind(Path(doubling.__file__))
    write_json(args.output / "doubling_config.json", doubling_cfg)
    # Historical campaigns use the same frozen package name. Isolate the
    # coverage campaign so Python cannot reuse the repetition campaign's modules.
    import subprocess
    import sys
    with (args.output / "doubling.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "result_analysis.compute_allocation.doubling",
                        "--config", str(args.output / "doubling_config.json")],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    work.finish("all")
    outputs = {str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "completion.json", dict(complete=True, report_sha256=report["sha256"],
        inputs=work.inputs, outputs=outputs, fitting=False, new_neural_evaluation=False,
        independent_units="configurations; languages, readouts, horizons and budgets remain nested"))
    return dict(complete=True, output=str(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    print(run(parser.parse_args().profile))
