"""Shared frozen-law loading, prediction, and numerical allocation checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

MODELS = ("M6-ND", "M5-ND", "J0-ND")
EXPERTS = (1, 4, 16, 32)
COMPONENTS = ("D_t", "D_family", "D_script", "D_other")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(bundle):
    manifest = json.loads((bundle/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(bundle/name) != digest:
            raise ValueError(f"Changed bundle input: {name}")
    fit = bundle/"evidence/fits"
    rows = pd.read_parquet(fit/"fit_rows.parquet")
    selected = json.loads((fit/"selected.json").read_text())
    context = json.loads((fit/"policy_context.json").read_text())
    recipes = pd.read_parquet(fit/"policy_rows.parquet")
    recipes = recipes[["ladder", "E", "N_act", "N_total"]].drop_duplicates()
    recipes = recipes.sort_values(["ladder", "E"]).reset_index(drop=True)
    recipes["config"] = recipes.ladder+"/E"+recipes.E.astype(str)
    horizon = rows.groupby(["ladder", "E"]).D_total.agg(["min", "max"])
    recipes = recipes.join(horizon, on=["ladder", "E"])
    if len(recipes) != 16 or rows.duplicated(["trajectory_id", "readout_id", "language"]).any():
        raise ValueError("Unexpected recipe or row population")
    # Fixed, outcome-independent exposure shares from the longest dense ladder.
    reference = rows.query("ladder == 'D' and E == 1 and K > 1")
    reference = reference.sort_values("D_total").groupby(["K", "language"]).tail(1)
    reference = reference.set_index(["K", "language"])
    shares = reference[list(COMPONENTS)].div(reference.D_total, axis=0)
    np.testing.assert_allclose(shares.sum(axis=1), 1., atol=1e-12)
    spec = importlib.util.spec_from_file_location("allocation_portable", bundle/"scripts/regenerate.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    return rows, selected, context, recipes, shares, replay


def coefficients(fit, shares, k, e, languages):
    """Per-language a,b in L=Linf+a*n^-alpha+b*d^-beta*n^zeta."""
    p = fit["shared_parameters"]
    h = max(0., np.log(k/12))/np.log(200/12)
    alpha = p["alpha_0"]+p["alpha_1"]*np.log(e)+p["gamma_n"]*h
    beta = p["beta_0"]+p["beta_1"]*np.log(e)+p["gamma_d"]*h
    phi = p["phi_0"]+p["phi_1"]*np.log(e)
    psi = p["psi_0"]+p["psi_1"]*np.log(e)
    if alpha <= 0 or beta <= 0 or fit["zeta"] < 0:
        raise ValueError("Convex fixed-E optimum is not available")
    s = shares.loc[k].loc[languages]
    t, f, script, other = (s[c].to_numpy() for c in COMPONENTS)
    effective = t*(1+f/t)**p["eta_family"]*(1+script/t)**p["eta_script"]
    c = pd.DataFrame(fit["language_coefficients"]).set_index("language").loc[languages]
    a = c.A.to_numpy()*e**p["delta"]*k**phi*(1+other/(t+f+script))**p["gamma_other"]
    b = c.B.to_numpy()*e**p["omega"]*k**psi*effective**(-beta)
    return c.L_inf.to_numpy(), a, b, alpha, beta, fit["zeta"]


def values(c, n, d):
    floor, a, b, alpha, beta, zeta = c
    return floor+a*np.asarray(n)[..., None]**(-alpha)+b*np.asarray(d)[..., None]**(-beta)*np.asarray(n)[..., None]**zeta


def optimum(c, compute):
    _, a, b, alpha, beta, zeta = c
    power = beta/(alpha+beta+zeta)
    n = (alpha*a.mean()/((beta+zeta)*b.mean()))**(1/(alpha+beta+zeta))*(compute/.6)**power
    return n, compute/(.6*n), power


def numerical_checks(rows, selected, shares, replay, languages):
    maximum, shift, derivative_error, optimum_error = 0., 0., 0., 0.
    elasticities = []
    for fold, fits in selected.items():
        for model in MODELS:
            fit = fits[model]
            for k in (12, 50, 200):
                for e in EXPERTS:
                    c = coefficients(fit, shares, k, e, languages)
                    part = rows.loc[rows.K.eq(k)&rows.E.eq(e)&rows.language.isin(languages)].copy()
                    standardized = part.copy()
                    for name in COMPONENTS:
                        standardized[name] = standardized.D_total*standardized.language.map(shares.loc[k][name])
                    analytic = np.concatenate([values(c, 1.9, d) for d in (.5, 3., 16.)])
                    probe = pd.concat([pd.DataFrame({"language":languages, "N_act":1.9e8, "D_total":d*1e9, "E":e, "K":k}) for d in (.5, 3., 16.)], ignore_index=True)
                    for name in COMPONENTS:
                        probe[name] = probe.D_total*probe.language.map(shares.loc[k][name])
                    maximum = max(maximum, float(np.max(np.abs(analytic-replay.model_prediction(model, fit, probe)))))
                    shift = max(shift, float(np.max(np.abs(replay.model_prediction(model, fit, standardized)-replay.model_prediction(model, fit, part)))))
                    n, d, power = optimum(c, 3.)
                    fun = lambda logn: float(values(c, np.exp(logn), 3./(.6*np.exp(logn))).mean())
                    result = minimize_scalar(fun, bounds=(np.log(n)-2, np.log(n)+2), method="bounded", options={"xatol":1e-10})
                    optimum_error = max(optimum_error, abs(np.exp(result.x)/n-1))
                    eps = 1e-4
                    derivative_error = max(derivative_error, abs((fun(np.log(n)+eps)-fun(np.log(n)-eps))/(2*eps)))
                    elasticities.append(dict(fold=fold, model=model, K=k, E=e, alpha=c[3], beta=c[4], zeta=c[5], size_elasticity=power, duration_elasticity=1-power))
    if maximum > 1e-10 or derivative_error > 1e-7 or optimum_error > 1e-5:
        raise ValueError("Analytic formula verification failed")
    return pd.DataFrame(elasticities), dict(analytic_vs_portable_max_error=maximum, fixed_share_vs_realized_max_prediction_shift=shift, optimum_relative_error=optimum_error, stationary_derivative_error=derivative_error)
