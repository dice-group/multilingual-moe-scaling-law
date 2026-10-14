"""Replay the final-law coverage ablation on saved held-out allocation inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.replay import predict


def run(root, control, output):
    prior = root / "review/fresh_prediction_checks_v2"
    paths = [prior / "policy_inputs.parquet", prior / "policy_observed.parquet",
             prior / "policy_metrics.csv", root / "inputs/scoring/grid.parquet",
             root / "recovery/v2/report/coefficients.json", control / "report/coefficients.json"]
    features, truth, previous, grid = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in paths[:4]]
    cohorts = dict(common12=set(grid[grid.K.eq(12)].language), incumbent50=set(features.language))
    results = []
    for name, path in zip(("RN14-stored-reuse", "RN12-stored-common-K"), paths[4:]):
        records = json.loads(path.read_text())["fits"]
        record, = [r["result"]["selected"] for r in records if r["model"] == name
                   and r["fold"] == "K200_common_languages" and r["arm"] == "lbfgsb_box"]
        frame = features[["config", "language", "FLOPs", "N_total"]].copy()
        frame["prediction"] = predict(name, record, features)
        for cohort, languages in cohorts.items():
            means = frame[frame.language.isin(languages)].groupby(["config", "FLOPs", "N_total"], as_index=False).prediction.mean()
            obs = truth[truth.language.isin(languages)].groupby(["config", "FLOPs", "reference"], as_index=False)["observed"].mean()
            joined = means.merge(obs, on=["config", "FLOPs"], validate="many_to_many")
            for cap in (.4e9, .7e9, 2.2e9):
                for (method, budget), g in joined[joined.N_total.le(cap)].groupby(["reference", "FLOPs"]):
                    chosen, best = g.loc[g.prediction.idxmin()], g.loc[g.observed.idxmin()]
                    results.append(dict(model=name, cohort=cohort, reference=method, cap=cap,
                        FLOPs=budget, chosen=chosen.config, best=best.config,
                        regret=chosen.observed-best.observed, correct=chosen.config == best.config))
    decisions = pd.DataFrame(results)
    metrics = decisions.groupby(["model", "cohort", "reference", "cap"], as_index=False).agg(
        mean_regret=("regret", "mean"), max_regret=("regret", "max"), correct_choices=("correct", "sum"))
    keys = ["model", "cohort", "reference", "cap"]
    check = metrics[metrics.model.eq("RN14-stored-reuse")].merge(previous, on=keys, validate="one_to_one", suffixes=("", "_old"))
    assert len(check) == 18
    np.testing.assert_allclose(check.max_regret, check.max_regret_old, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(check.correct_choices, check.correct_choices_old)
    output.mkdir(parents=True, exist_ok=False)
    decisions.to_csv(output / "decisions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "receipt.json").write_text(json.dumps(dict(inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths+[Path(__file__)]}, parent_metrics_reproduced=True, refitting=False), indent=2)+"\n")
    print(metrics.query("cohort == 'incumbent50' and reference == 'smooth'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("root", "control", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.control, args.output)
