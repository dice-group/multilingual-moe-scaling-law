"""Joint capacity/coverage contrasts from sealed observations, without a fitted law.

Windows depend only on observed exposure support. Each target language is
evaluated on exactly the same window across all included N, E, and K settings.
Language curves, not interpolation grid points, are the units of aggregation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from result_analysis.capacity_response.tables import (  # noqa: E402
    COVERAGE_CONTRASTS, EXPERTS, RELEASE_MANIFEST_SHA256,
    load_primary, matched_pressure, sha256_file,
)

LADDERS = tuple("ABCD")
STAGES = (0., .25, .5, .75, 1.)
METHODS = ("linear_log", "pchip_log", "linear_tokens")


def evaluate(curve, lo, hi, method):
    """Endpoint/quarter-window losses and a mean uniform in log exposure.

    Linear-log and PCHIP means use their exact integrals. Linear-token means
    also use the analytic integral after mapping the linear segment to log D.
    No method extrapolates. Both losses in a contrast share the same window.
    """
    x = np.log(curve.D_t.to_numpy(float))
    y = curve.adjusted_loss.to_numpy(float)
    a, b = np.log(lo), np.log(hi)
    assert x[0] <= a + 1e-12 and b <= x[-1] + 1e-12 and b > a
    knots = np.unique(np.r_[a, x[(x > a) & (x < b)], b])
    stages = a + np.asarray(STAGES) * (b-a)
    if method == "linear_log":
        values = np.interp(stages, x, y)
        mean = trapezoid(np.interp(knots, x, y), knots) / (b-a)
    elif method == "pchip_log":
        f = PchipInterpolator(x, y, extrapolate=False)
        values = f(stages)
        mean = f.integrate(a, b) / (b-a)
    else:
        tokens = np.exp(x)
        values = np.interp(np.exp(stages), tokens, y)
        endpoints = np.exp(knots)
        endpoint_y = np.interp(endpoints, tokens, y)
        slope = np.diff(endpoint_y) / np.diff(endpoints)
        intercept = endpoint_y[:-1] - slope * endpoints[:-1]
        mean = np.sum(slope*np.diff(endpoints) + intercept*np.diff(knots))/(b-a)
    return dict(zip(("early", "quarter", "middle", "three_quarter", "late"), values)) | {"log_mean": float(mean)}


def contrast_rows(losses, comparisons):
    index = ["analysis", "method", "language", "ladder", "E", "N_act", "summary"]
    pivot = losses.pivot(index=index, columns="K", values="loss")
    rows = []
    for a, b in comparisons:
        if a not in pivot or b not in pivot:
            continue
        r = pivot[[a, b]].dropna().reset_index().rename(columns={a: "reference_loss", b: "target_loss"})
        r["reference_K"], r["target_K"] = a, b
        r["response"] = r.target_loss-r.reference_loss
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def main(root: Path, output: Path):
    """Write a new analysis run; never replace an existing evidence directory."""
    frame = load_primary(root)
    output.mkdir(parents=True, exist_ok=False)
    # Within-language unigram subtraction cancels in every contrast here.
    assert (frame.assign(offset=frame.raw_nll-frame.adjusted_loss)
            .groupby("language").offset.agg(lambda x: x.max()-x.min()) < 2e-12).all()
    curves = {key: g.sort_values("D_t") for key, g in frame.groupby(["language", "ladder", "E", "K"])}
    assert all(not c.D_t.duplicated().any() for c in curves.values())
    n_map = frame.groupby("ladder").N_act.first().to_dict()
    assert frame.groupby("ladder").N_act.nunique().eq(1).all()
    lang_sets = {int(k): set(g.language) for k, g in frame.groupby("K")}
    plans = []
    for a, b in COVERAGE_CONTRASTS:
        plans.append((f"K{a}_to_K{b}", (a, b), sorted(lang_sets[a] & lang_sets[b]), LADDERS))
    # Exact cross-transition comparison on one cohort and one window per language.
    plans.append(("common12_all_K", (12, 50, 200), sorted(lang_sets[12]), LADDERS))
    # English has no full-four-size K1/K200 overlap; retain the supported larger sizes.
    plans.append(("K1_to_K200_BCD", (1, 200), sorted(lang_sets[1]), tuple("BCD")))
    # Pair-specific size windows retain more support than the full-four-size window.
    for a, b in ((12, 50), (12, 200), (50, 200)):
        for small, large in zip(LADDERS[:-1], LADDERS[1:]):
            plans.append((f"K{a}_to_K{b}_{small}{large}", (a, b), sorted(lang_sets[a]), (small, large)))

    windows, losses, support_curves = [], [], []
    for name, ks, languages, ladders in plans:
        for lang in languages:
            required = [(lang, lad, e, k) for lad in ladders for e in EXPERTS for k in ks]
            assert all(key in curves for key in required)
            lo = max(float(curves[key].D_t.min()) for key in required)
            hi = min(float(curves[key].D_t.max()) for key in required)
            eligible = hi > lo
            windows.append(dict(analysis=name, language=lang, ladders="".join(ladders), Ks=",".join(map(str, ks)),
                                min_D_t=lo, max_D_t=hi, width_factor=hi/lo, eligible=eligible))
            if not eligible:
                continue
            for key in required:
                _, lad, e, k = key
                c = curves[key]
                # Record actual readout support; interpolation nodes never count as data.
                support_curves.append(dict(analysis=name, language=lang, ladder=lad, E=e, K=k,
                    source_trajectory=c.trajectory_id.iloc[0], observed_endpoints=len(c),
                    endpoints_inside_window=int(c.D_t.between(lo, hi).sum()),
                    min_D_t=lo, max_D_t=hi))
                for method in METHODS:
                    for summary, loss in evaluate(c, lo, hi, method).items():
                        losses.append(dict(analysis=name, method=method, language=lang, ladder=lad,
                            N_act=n_map[lad], E=e, K=k, summary=summary, loss=float(loss)))
    w = pd.DataFrame(windows)
    l = pd.DataFrame(losses)
    p = contrast_rows(l, COVERAGE_CONTRASTS)
    w.to_csv(output/"support_windows.csv", index=False)
    pd.DataFrame(support_curves).to_csv(output/"source_curve_support.csv", index=False)
    l.to_csv(output/"language_losses.csv", index=False)
    p.to_csv(output/"language_responses.csv", index=False)

    group = ["analysis", "method", "reference_K", "target_K", "ladder", "N_act", "E", "summary"]
    summary = p.groupby(group).agg(
        languages=("language", "nunique"), reference_loss=("reference_loss", "mean"),
        target_loss=("target_loss", "mean"), response=("response", "mean"),
        median_response=("response", "median"), q10_response=("response", lambda x: x.quantile(.1)),
        q90_response=("response", lambda x: x.quantile(.9)),
        negative_languages=("response", lambda x: int((x < 0).sum())),
    ).reset_index()
    summary.to_csv(output/"response_summary.csv", index=False)

    # Expert contrasts retain each language; paired baseline/target losses verify identity.
    key = ["analysis", "method", "reference_K", "target_K", "language", "ladder", "N_act", "summary"]
    ep = p.pivot(index=key, columns="E", values=["response", "reference_loss", "target_loss"])
    erows = []
    for small, large in ((1,4), (4,16), (16,32), (1,32)):
        q = ep.index.to_frame(index=False)
        q["low_E"], q["high_E"] = small, large
        q["expert_response_improvement"] = (ep["response",small]-ep["response",large]).to_numpy()
        q["expert_loss_gain_reference"] = (ep["reference_loss",small]-ep["reference_loss",large]).to_numpy()
        q["expert_loss_gain_target"] = (ep["target_loss",small]-ep["target_loss",large]).to_numpy()
        assert np.allclose(q.expert_response_improvement,
            q.expert_loss_gain_target-q.expert_loss_gain_reference, atol=1e-12, rtol=0)
        erows.append(q)
    ec = pd.concat(erows, ignore_index=True)
    ec.to_csv(output/"expert_contrasts.csv", index=False)
    egroup = [x for x in key if x != "language"]+["low_E", "high_E"]
    es = ec.groupby(egroup).agg(
        languages=("language", "nunique"), response_improvement=("expert_response_improvement", "mean"),
        positive_languages=("expert_response_improvement", lambda x: int((x > 0).sum())),
        loss_gain_reference=("expert_loss_gain_reference", "mean"), loss_gain_target=("expert_loss_gain_target", "mean"),
        positive_target_loss_gain_languages=("expert_loss_gain_target", lambda x: int((x > 0).sum())),
    ).reset_index()
    es.to_csv(output/"expert_contrast_summary.csv", index=False)

    nkey = [x for x in key if x not in ("ladder", "N_act")]+["E"]
    npiv = p.pivot(index=nkey, columns="ladder", values=["response", "reference_loss", "target_loss"])
    nrows = []
    for small, large in (("A","B"), ("B","C"), ("C","D"), ("A","D")):
        q = npiv.index.to_frame(index=False)
        q["low_N"], q["high_N"] = small, large
        q["active_response_improvement"] = (npiv["response",small]-npiv["response",large]).to_numpy()
        q["active_loss_gain_reference"] = (npiv["reference_loss",small]-npiv["reference_loss",large]).to_numpy()
        q["active_loss_gain_target"] = (npiv["target_loss",small]-npiv["target_loss",large]).to_numpy()
        q = q.dropna()
        assert np.allclose(q.active_response_improvement,
            q.active_loss_gain_target-q.active_loss_gain_reference, atol=1e-12, rtol=0)
        nrows.append(q)
    nc = pd.concat(nrows, ignore_index=True)
    nc.to_csv(output/"active_contrasts.csv", index=False)
    ngroup = [x for x in nkey if x != "language"]+["low_N", "high_N"]
    ns = nc.groupby(ngroup).agg(
        languages=("language", "nunique"), response_improvement=("active_response_improvement", "mean"),
        positive_languages=("active_response_improvement", lambda x: int((x > 0).sum())),
        loss_gain_reference=("active_loss_gain_reference", "mean"), loss_gain_target=("active_loss_gain_target", "mean"),
        positive_target_loss_gain_languages=("active_loss_gain_target", lambda x: int((x > 0).sum())),
    ).reset_index()
    ns.to_csv(output/"active_contrast_summary.csv", index=False)

    triple_index = [x for x in key if x not in ("ladder", "N_act")]
    ex = ec.query("low_E==1 and high_E==32").pivot(index=triple_index, columns="ladder", values="expert_response_improvement")
    tr = ex.index.to_frame(index=False)
    tr["expert_response_improvement_A"] = ex.A.to_numpy()
    tr["expert_response_improvement_D"] = ex.D.to_numpy()
    # Positive = more additional expert relief at large N; negative = less.
    tr["change_of_expert_response_improvement_D_minus_A"] = (ex.D-ex.A).to_numpy()
    tr = tr.dropna()
    tr.to_csv(output/"three_way_contrasts.csv", index=False)
    tg = [x for x in triple_index if x != "language"]
    ts = tr.groupby(tg).agg(languages=("language", "nunique"),
        expert_response_improvement_A=("expert_response_improvement_A", "mean"),
        expert_response_improvement_D=("expert_response_improvement_D", "mean"),
        three_way_change=("change_of_expert_response_improvement_D_minus_A", "mean"),
        diminishing_languages=("change_of_expert_response_improvement_D_minus_A", lambda x: int((x < 0).sum())),
    ).reset_index()
    ts.to_csv(output/"three_way_summary.csv", index=False)

    # On common cohort + exposure windows, adjacent expansions must add exactly.
    cp = p.query("analysis=='common12_all_K'").pivot(
        index=["method", "language", "ladder", "E", "summary"], columns=["reference_K", "target_K"], values="response")
    closure_error = float(np.max(abs(cp[12,200]-cp[12,50]-cp[50,200])))
    assert closure_error < 2e-12
    # Reproduce the previous headline, which uses a different support/weighting rule.
    old = matched_pressure(frame,12,200).groupby(["ladder", "E"]).pressure.mean().groupby("E").mean()
    assert abs(100*(1-old[32]/old[1])-82.981784) < .02

    main_names = ("K12_to_K50", "K12_to_K200", "K50_to_K200")
    selected = summary[summary.analysis.isin(main_names)]
    selected.query("method=='linear_log' and summary=='log_mean'").to_csv(output/"main_response_matrix.csv", index=False)
    sensitivity = ts[ts.analysis.isin(main_names) & ts.summary.eq("log_mean")]
    sensitivity.to_csv(output/"interpolation_sensitivity.csv", index=False)
    # Alternative support: target losses only at actually observed endpoints
    # shared by all 16 N/E configurations; reference losses retain log interpolation.
    anchor_rows = []
    for a, b in ((12,50), (12,200), (50,200)):
        observed = matched_pressure(frame, a, b).pivot(
            index=["language", "D_t"], columns=["ladder", "E"], values="pressure").dropna()
        language_means = observed.groupby("language").mean()
        for (lad, e), value in language_means.mean().items():
            anchor_rows.append(dict(reference_K=a,target_K=b,ladder=lad,E=e,
                languages=len(language_means),observed_language_exposure_cells=len(observed),response=float(value)))
    pd.DataFrame(anchor_rows).to_csv(output/"observed_target_anchor_sensitivity.csv", index=False)

    native = selected.query("method=='linear_log'")
    endpoints = native.pivot(index=["analysis", "ladder", "E"], columns="summary",
        values=["response", "reference_loss", "target_loss"])
    expo_experts = es[es.analysis.isin(main_names)].query(
        "method=='linear_log' and low_E==1 and high_E==32").pivot(
            index=["analysis", "ladder"], columns="summary", values=["response_improvement", "loss_gain_target"])
    pairwise = ns[ns.analysis.str.endswith(("_AB","_BC","_CD"))].query(
        "method=='linear_log' and summary=='log_mean'")
    exposure_audit = dict(
        aggregate_response_increases_early_to_late=int((endpoints["response","late"]>endpoints["response","early"]).sum()),
        aggregate_N_E_transition_settings=len(endpoints),
        reference_loss_falls=int((endpoints["reference_loss","late"]<endpoints["reference_loss","early"]).sum()),
        target_loss_falls=int((endpoints["target_loss","late"]<endpoints["target_loss","early"]).sum()),
        absolute_E1_to_E32_target_loss_gain_grows=int((expo_experts["loss_gain_target","late"]>expo_experts["loss_gain_target","early"]).sum()),
        expert_response_improvement_grows=int((expo_experts["response_improvement","late"]>expo_experts["response_improvement","early"]).sum()),
        aggregate_N_transition_settings=len(expo_experts),
        wider_pairwise_size_response_improvement_positive=int(pairwise.response_improvement.gt(0).sum()),
        wider_pairwise_size_comparisons=len(pairwise),
        wider_pairwise_size_target_loss_gain_positive=int(pairwise.loss_gain_target.gt(0).sum()),
    )
    (output/"exposure_audit.json").write_text(json.dumps(exposure_audit, indent=2)+"\n")
    metadata = dict(
        release_manifest_sha256=RELEASE_MANIFEST_SHA256, builder_sha256=sha256_file(Path(__file__)),
        observations=len(frame), trajectories=int(frame.trajectory_id.nunique()), readouts=int(frame.readout_id.nunique()),
        estimator="Equal-language means of per-language signed coverage responses; shared exposure window across all included N/E/K settings. Log-mean uses exact integration of within-support interpolated losses in log D_t, not the count of endpoints.",
        uncertainty="Descriptive primary-seed analysis. Languages, checkpoints, contrasts and interpolated stages are not independent training replications. Quantiles describe language heterogeneity, not confidence intervals.",
        limitations=["Per-language exposure windows differ; early/late do not denote one absolute D_t across languages.",
            "Full-four-size windows end at support allowed by the smallest size; later large-model training is excluded from that estimand.",
            "K changes mixture composition as well as count. E expands stored parameters. N changes active and stored capacity.",
            "Matching D_t is not matching total tokens, logical FLOPs, stored capacity or hardware cost.",
            "No law fitting, extrapolation, significance test or parameter-identifiability inference is used."],
        closure_max_abs_error=closure_error,
        original_headline=dict(E1=float(old[1]),E32=float(old[32]),attenuation_percent=float(100*(1-old[32]/old[1]))),
    )
    (output/"verification.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("SHARED-WINDOW RESPONSE MATRICES (equal languages; integrated log D_t):")
    for name in main_names:
        q = selected.query("analysis==@name and method=='linear_log' and summary=='log_mean'")
        print(name)
        print(q.pivot(index="ladder", columns="E", values="response").round(5).to_string())
    print("THREE-WAY CONTRAST SENSITIVITY:")
    print(sensitivity[["analysis", "method", "expert_response_improvement_A", "expert_response_improvement_D", "three_way_change", "diminishing_languages", "languages"]].round(5).to_string(index=False))
    print("EARLY/LATE RESPONSE:")
    print(selected.query("method=='linear_log' and summary in ['early','late'] and E in [1,32]")[["analysis","ladder","E","summary","response","negative_languages","languages"]].round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing the sealed release")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    args = parser.parse_args()
    main(args.root, args.output)
