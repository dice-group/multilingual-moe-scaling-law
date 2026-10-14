"""Check K12 composition sensitivity of the measured K200 coverage gap.

No training or law fitting. Reuse the paper's exact exposures and within-support
interpolation; retain all three seeds, expert counts, and baseline mixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import multilingual_megatron.experiments.scaling.stage1_release.exposure as exposure_source
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from multilingual_megatron.experiments.scaling.stage1_release.exposure import (
    _exact_exposures, _resolve_training_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in bindings:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            bindings[str(path)] = digest.hexdigest()
        if expected is not None and bindings[str(path)] != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    bind(Path(__file__))
    bind(Path(exposure_source.__file__))
    comp = root / "result_analysis/stage1_language_composition_v1/inputs"
    cm = read(comp / "manifest.json")
    spec = read(comp / "spec.json", cm["files"]["spec.json"])
    common = sorted(spec["shared_languages"])
    losses = pd.read_parquet(bind(comp / "losses.parquet", cm["files"]["losses.parquet"]))
    exposures = pd.read_parquet(bind(comp / "exposures.parquet", cm["files"]["exposures.parquet"]))
    losses = losses[losses.language.isin(common)].copy()
    assert np.allclose(losses.adjusted_loss, losses.raw_nll-losses.unigram_nll, rtol=0, atol=1e-12)
    for col in ("target_hash", "token_count", "vocab_size", "special_tokens", "unigram_nll"):
        assert losses.groupby("language")[col].nunique().eq(1).all(), col
    identity = losses.drop_duplicates("language").set_index("language")
    new = losses.query("arm != 'reference'").merge(
        exposures, on=["arm", "seed", "E", "horizon", "language"], validate="one_to_one")
    new = new.rename(columns={"job_id": "readout_id"})
    cols = ["arm", "seed", "E", "horizon", "language", "D_total", "D_t", "raw_nll", "readout_id"]
    rows = new[cols].to_dict("records")

    release = root / "result_analysis/stage1_abcd_jointmoe_x1_v1"
    rm = read(release / "release_manifest.json")

    def table(name):
        return pd.read_parquet(bind(release / "tables" / f"{name}.parquet",
                                    rm["table_inventory"][name]["sha256"]))

    val, exp = table("validation_language"), table("data_exposure")
    k200_languages = set(val.query("language_count == 200 and metric_role == 'trained'").language)
    assert len(k200_languages) == 200
    assert all(set(block.language) <= k200_languages for _, block in exposures.groupby("arm"))
    reference_languages = sorted(val.query("ladder == 'B' and language_count == 12 and metric_role == 'trained'").language.unique())
    assert len(common) == 7 and len(reference_languages) == 12
    historical = val.query("ladder == 'B' and language_count in [12, 200] and experts in [1, 16, 32]")
    historical = historical[historical.language.isin(reference_languages)]
    historical = historical.merge(exp[["readout_id", "language", "realized_self_exposure_tokens"]],
                                  on=["readout_id", "language"], validate="one_to_one")

    def check_identity(language, raw):
        if language not in common:
            return
        expected = identity.loc[language]
        assert raw["target_token_ids_sha256"] == expected.target_hash
        assert raw["token_count"] == expected.token_count
        assert raw["vocab_size"] == expected.vocab_size
        special = raw.get("special_token_ids", raw.get("special_token_ids_json"))
        if isinstance(special, str):
            special = json.loads(special)
        assert special == json.loads(expected.special_tokens)

    for r in historical.to_dict("records"):
        check_identity(r["language"], r)
        rows.append(dict(arm="reference" if r["language_count"] == 12 else "K200", seed=1234,
                         E=r["experts"], horizon=r["nominal_tokens"], language=r["language"],
                         D_total=r["actual_tokens"], D_t=r["realized_self_exposure_tokens"],
                         raw_nll=r["raw_nll"], readout_id=r["readout_id"]))

    seedroot = root / "outputs/pipeline_runs/stage1_b_seed_confirmation_v1"
    summary = read(seedroot / "control/seed_confirmation_summary.json")
    assert summary["status"] == "complete" and not summary["errors"]
    assert summary["observed_results"] == summary["expected_results"] == 108
    selected = [r for r in summary["results"] if r["K"] in (12, 200)]
    assert len(selected) == 72
    for count, r in enumerate(selected, 1):
        ep = seedroot / "jobs" / r["job_id"] / "evaluation_manifest.json"
        evaluation = read(ep)
        assert evaluation["status"] == "complete"
        tp, index, description, training = _resolve_training_inputs(ep)
        bind(tp)
        assert index is not None and description is not None
        tier = Path(training["run_cfg"]["data_manifest"])
        for path in (index, description, tier):
            bind(path)
        seq = int(training["metadata"]["seq_length"])
        actual = int(r["actual_end_tokens"])
        assert seq == 2048 and actual % seq == 0
        assert int(training["metadata"]["segments"][-1]["actual_end_tokens"]) == actual
        realized = _exact_exposures(index_path=index, description_path=description,
                                   tier_manifest_path=tier, end_samples=actual // seq,
                                   sequence_length=seq)
        assert len(realized) == r["K"] and sum(realized.values()) == actual
        for language in reference_languages:
            artifact = evaluation["languages"][language]
            path = Path(artifact["path"])
            path = path if path.is_absolute() else ep.parent / path
            raw = read(path, artifact["sha256"])
            check_identity(language, raw)
            rows.append(dict(arm="reference" if r["K"] == 12 else "K200", seed=r["model_seed"],
                             E=r["experts"], horizon=r["D"], language=language,
                             D_total=actual, D_t=realized[language], raw_nll=raw["raw_nll"],
                             readout_id=r["job_id"]))
        if count % 18 == 0:
            print(f"Verified {count}/72 historical seed evaluations", flush=True)

    frame = pd.DataFrame(rows).sort_values(["arm", "seed", "E", "language", "horizon"])
    keys = ["arm", "seed", "E", "horizon", "language"]
    assert not frame.duplicated(keys).any()
    assert frame.groupby(["arm", "seed", "E", "language"]).size().eq(6).all()
    old = frame.query("arm == 'reference' and language in @common")
    parity = old.merge(losses.query("arm == 'reference'"), on=["arm", "seed", "E", "horizon", "language"],
                       suffixes=("_old", "_new"), validate="one_to_one")
    reference_parity_error = float((parity.raw_nll_old - parity.raw_nll_new).abs().max())
    assert len(parity) == 378 and reference_parity_error < 1e-4

    def contrasts(method):
        references = {key: block.sort_values("D_t") for key, block in
                      frame.query("arm != 'K200'").groupby(["arm", "seed", "E", "language"])}
        result = []
        for t in frame.query("arm == 'K200'").itertuples(index=False):
            for arm in ("reference", "compact", "diverse"):
                ref = references.get((arm, t.seed, t.E, t.language))
                if ref is None or not ref.D_t.min() <= t.D_t <= ref.D_t.max():
                    continue
                x = ref.D_t.to_numpy()
                y = ref.raw_nll.to_numpy()
                if method == "linear_D_t":
                    loss = float(np.interp(t.D_t, x, y))
                elif method == "pchip_log_D_t":
                    loss = float(PchipInterpolator(np.log(x), y, extrapolate=False)(np.log(t.D_t)))
                else:
                    loss = float(np.interp(np.log(t.D_t), np.log(x), y))
                upper = min(int(np.searchsorted(x, t.D_t)), len(x) - 1)
                lower = upper if x[upper] == t.D_t else max(0, upper - 1)
                result.append(dict(arm=arm, seed=t.seed, E=t.E, language=t.language,
                                   horizon=t.horizon, target_D_total=t.D_total, target_D_t=t.D_t,
                                   reference_lower_D_t=x[lower], reference_upper_D_t=x[upper],
                                   target_raw_nll=t.raw_nll, reference_raw_nll=loss,
                                   gap=t.raw_nll-loss, method=method))
        return pd.DataFrame(result)

    points = contrasts("log_linear_D_t")
    robust = root / "external/research/scaling/stage1_paper_naacl2027_scaling_laws/tables/matched_exposure_robustness_v1"
    bm = read(robust / "manifest.json")
    expected_hash = next(v["sha256"] for v in bm["outputs"] if v["path"].endswith("/seed_pressure_points.csv"))
    previous = pd.read_csv(bind(robust / "seed_pressure_points.csv", expected_hash))
    reproduction = points.query("arm == 'reference'").merge(previous,
        left_on=["seed", "E", "language", "target_D_total"],
        right_on=["seed", "E", "language", "target_D_total"], validate="one_to_one")
    reproduction_error = float((reproduction.gap - reproduction.raw_nll_delta).abs().max())
    assert len(reproduction) == len(previous) == 432 and reproduction_error < 1e-10

    # Use the same languages and K200 horizons in every arm, seed and expert cell.
    shared = points.query("language in @common")
    support = shared.groupby(["horizon", "language"]).size()
    supported_horizons = sorted(h for h, block in support.groupby(level="horizon")
                               if len(block) == len(common) and block.eq(27).all())
    assert supported_horizons
    all_methods = pd.concat([points, contrasts("linear_D_t"), contrasts("pchip_log_D_t")], ignore_index=True)
    chosen = all_methods.query("language in @common and horizon in @supported_horizons").copy()
    assert chosen.groupby(["method", "arm", "seed", "E"]).size().eq(len(common)*len(supported_horizons)).all()
    seed_means = chosen.groupby(["method", "arm", "seed", "E"]).gap.mean().unstack("E")
    seed_means.columns = [f"E{e}_gap" for e in seed_means.columns]
    seed_means = seed_means.reset_index()
    seed_means["absolute_reduction"] = seed_means.E1_gap - seed_means.E32_gap
    seed_means["reduction_percent"] = np.where(seed_means.E1_gap > 0,
                                              100*seed_means.absolute_reduction/seed_means.E1_gap, np.nan)
    averages = seed_means.groupby(["method", "arm"])[["E1_gap", "E16_gap", "E32_gap", "absolute_reduction"]].mean().reset_index()
    averages["reduction_percent"] = np.where(averages.E1_gap > 0,
                                             100*averages.absolute_reduction/averages.E1_gap, np.nan)
    per_horizon = chosen.groupby(["method", "arm", "seed", "E", "horizon"], as_index=False).gap.mean()
    per_language = chosen.groupby(["method", "arm", "seed", "E", "language"], as_index=False).gap.mean()
    for name, data in (("observations", frame), ("matched_points", chosen), ("summary_by_seed", seed_means),
                       ("summary", averages), ("by_horizon", per_horizon), ("by_language", per_language)):
        data.to_csv(out / f"{name}.csv", index=False, float_format="%.15g")
    receipt = dict(status="complete", common_languages=common,
                   target_horizons_tokens=supported_horizons, active_size="190M", seeds=[1234, 5678, 91011],
                   experts=[1, 16, 32], main_method="log-linear in realized target exposure, no extrapolation",
                   aggregation="equal languages, horizons, and seeds; reduction is ratio of aggregated gaps",
                   validation_token_identity="exact target hashes and counts matched across arms and K200",
                   all_K12_mixtures_nested_in_K200=True,
                   reference_reevaluation_max_raw_loss_difference=reference_parity_error,
                   paper_common12_matched_points_reproduced=len(reproduction),
                   paper_common12_max_gap_difference=reproduction_error,
                   scope="Baseline-composition sensitivity, one active size and one shared K200 mixture; correlated mixture contrasts, not independent expansion experiments.",
                   inputs=bindings)
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("\nPrimary result (gap in nats):")
    print(averages.query("method == 'log_linear_D_t'").to_string(index=False))
    print("\nBy seed:")
    print(seed_means.query("method == 'log_linear_D_t'").to_string(index=False))
    print("\nK200 horizons:", supported_horizons)
    print("Paper estimator reproduction error:", reproduction_error)


if __name__ == "__main__":
    main()
