"""Continuous M6-ND allocation with fixed coefficients and recorded geometry.

Optimizing the allocation is not refitting a scaling law. Stored size is
piecewise linear in active size at each E; horizons use the original log rule.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model

COMPONENTS = ["D_t", "D_family", "D_script", "D_other"]


def problem(shares, size, tokens):
    return SimpleNamespace(N=np.full(len(shares), size), D=shares.D_t.to_numpy()*tokens,
        E=shares.E.to_numpy(), K=shares.K.to_numpy(), language_ids=tuple(shares.language),
        **{c: shares[c].to_numpy()*tokens for c in COMPONENTS[1:]})


def reduce_laws(records, shares, checks):
    """Reduce each native-language mean to base+C*n^-alpha+B*n^zeta*d^-beta."""
    laws = []
    for fold, record in records.items():
        coefficients = pd.DataFrame(record["language_coefficients"]).set_index("language")
        for k, s in shares.groupby("K"):
            checks.require(f"native allocation languages {fold}/{k}", len(s) == k)
            cs = coefficients.loc[s.language]
            for e in (1, 4, 16, 32):
                s = s.assign(E=e)
                cap, data, *_ = model.terms(record["coordinates"], problem(s, 1e8, 1e9))
                cap2, data2, *_ = model.terms(record["coordinates"], problem(s, 2e8, 1e9))
                _, data_d, *_ = model.terms(record["coordinates"], problem(s, 1e8, 2e9))
                p = dict(fold=fold, K=k, E=e, base=float(np.mean(cs.L_inf.to_numpy()+s.unigram_nll)),
                         C=float(np.mean(cs.A.to_numpy()*cap)), B=float(np.mean(cs.B.to_numpy()*data)),
                         alpha=float(-np.log2(cap2[0]/cap[0])), beta=float(-np.log2(data_d[0]/data[0])),
                         zeta=float(np.log2(data2[0]/data[0])))
                checks.require(f"convex log-size objective {fold}/{k}/{e}",
                               min(p["alpha"], p["beta"], p["C"], p["B"], p["zeta"]+p["beta"]) > 0)
                for n, d in ((1.35e8, 3e9), (5e8, 40e9)):
                    direct = np.mean(model.predict(problem(s, n, d), record)+s.unigram_nll)
                    reduced = p["base"]+p["C"]*(n/1e8)**(-p["alpha"])+p["B"]*(n/1e8)**p["zeta"]*(d/1e9)**(-p["beta"])
                    checks.close(f"reduced predictor {fold}/{k}/{e}/{n}", reduced, direct, atol=1e-10)
                laws.append(p)
    return pd.DataFrame(laws)


def optimum(p, geometry, cap, budgets):
    """Analytic optimum on the feasible fixed-E interval (vectorized budgets)."""
    geometry = geometry.sort_values("N_act")
    f = np.asarray(budgets, float)
    if cap < geometry.N_total.min() or np.any(f <= 0):
        raise ValueError("infeasible storage cap or nonpositive compute")
    lower = geometry.N_act.min()/1e8
    upper = np.minimum(np.interp(cap, geometry.N_total, geometry.N_act), f/(6*.5e9))/1e8
    if np.any(upper < lower):
        raise ValueError("no allocation with minimum 0.5B tokens")
    q = p.B*(f/.6e18)**(-p.beta)
    power = p.beta+p.zeta
    n = np.clip((p.alpha*p.C/(power*q))**(1/(p.alpha+power)), lower, upper)
    loss = p.base+p.C*n**(-p.alpha)+q*n**power
    tokens = f/(6*n*1e8)
    stored = np.interp(n*1e8, geometry.N_act, geometry.N_total)
    horizon = np.exp(np.interp(np.log(n*1e8), np.log(geometry.N_act), np.log([8e9, 16e9, 32e9, 32e9])))
    return dict(N_act=n*1e8, N_total=stored, D_total=tokens, loss=loss,
                horizon_extrapolation=tokens>horizon)


def run(evidence, output, checks, geometry):
    rows = evidence.table("study/primary_fits/fit_rows.parquet")
    shares = rows.copy()
    shares[COMPONENTS] = shares[COMPONENTS].div(shares.D_total, axis=0)
    shares = shares.groupby(["K", "language"])[COMPONENTS+["unigram_nll"]].mean().reset_index()
    frozen_shares = evidence.table("allocation/exposure_shares.csv")
    keys = ["K", "language"]
    checks.close("allocation shares from observed exposures", shares.set_index(keys).sort_index(),
                 frozen_shares.set_index(keys).sort_index(), source="allocation/exposure_shares.csv")
    shares.to_csv(output / "exposure_shares.csv", index=False)
    laws = reduce_laws(evidence.json("allocation/frozen_coefficients.json"), shares, checks)
    keys = ["fold", "K", "E"]
    reference = evidence.table("allocation/reduced_mean_laws.csv")
    checks.close("all reduced laws", laws.set_index(keys).sort_index(), reference.set_index(keys).sort_index(), atol=1e-10)
    laws.to_csv(output / "reduced_mean_laws.csv", index=False)
    lookup = laws.set_index(keys)
    original = evidence.table("allocation/candidate_optima.csv.gz")
    candidates = []
    for (fold, k, cap, e), group in original.groupby(["fold", "K", "cap", "E"]):
        f = group.F_EF.to_numpy()*1e18
        computed = optimum(lookup.loc[fold, k, e], geometry[geometry.E.eq(e)], cap, f)
        checks.close(f"candidate loss {fold}/{k}/{cap}/{e}", computed["loss"], group.loss, atol=1e-10)
        checks.close(f"candidate size {fold}/{k}/{cap}/{e}", computed["N_act"], group.N_act, atol=1e-4)
        checks.require(f"candidate storage {fold}/{k}/{cap}/{e}", np.all(computed["N_total"] <= cap*(1+1e-12)))
        candidates.append(pd.DataFrame(dict(fold=fold, K=k, cap=cap, E=e, F_EF=group.F_EF.to_numpy(), **computed)))
    candidates = pd.concat(candidates, ignore_index=True)
    winner_keys = ["fold", "K", "cap", "F_EF"]
    winners = candidates.loc[candidates.groupby(winner_keys).loss.idxmin()].sort_values(winner_keys)
    original_winners = evidence.table("allocation/selected_optima.csv").sort_values(winner_keys)
    checks.close("all fitted winner identities", winners.E, original_winners.E)
    checks.close("all fitted winner losses", winners.loss, original_winners.loss, atol=1e-10)
    candidates.to_csv(output / "fitted_candidates.csv.gz", index=False)
    winners.to_csv(output / "fitted_winners.csv", index=False)
    transitions = []
    for (fold, k, cap), group in winners.groupby(["fold", "K", "cap"]):
        f, experts = group.F_EF.to_numpy()*1e18, group.E.to_numpy()
        def value(e, budget):
            return float(optimum(lookup.loc[fold, k, e], geometry[geometry.E.eq(e)], cap, budget)["loss"])
        feasible = geometry.groupby("E").N_total.min()
        feasible = feasible[feasible.le(cap)].index
        for i in np.flatnonzero(experts[1:] != experts[:-1]):
            a, b = int(experts[i]), int(experts[i+1])
            root = np.exp(brentq(lambda logf: value(a, np.exp(logf))-value(b, np.exp(logf)), np.log(f[i]), np.log(f[i+1])))
            checks.close(f"switch lower envelope {fold}/{k}/{cap}/{a}/{b}", value(a, root), min(value(e, root) for e in feasible), atol=1e-10)
            transitions.append(dict(fold=fold, K=k, cap=cap, before=a, after=b, FLOPs=root))
    transitions = pd.DataFrame(transitions)
    transitions.to_csv(output / "fitted_switches.csv", index=False)
    main = transitions.query("fold == 'E32' and cap == 5100000000 and before == 32 and after == 16").set_index("K").FLOPs
    checks.close("paper M6-ND switch ratio", main.loc[200]/main.loc[1], 2.70, atol=.005, source="paper/my_content.tex")
    return winners, {"predictor": "M6-ND", "fold": "E32", "K1_switch_FLOPs": main.loc[1],
                     "K200_switch_FLOPs": main.loc[200], "switch_delay": main.loc[200]/main.loc[1],
                     "candidate_allocations": len(candidates), "coefficients_refitted": False}
