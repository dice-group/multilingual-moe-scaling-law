"""Explicit literature-form contracts; no optimizer or outcome selection here.

All new fits predict raw NLL. Subtracting the bound per-language unigram NLL
recovers the study response. ATLAS bounds/sign choices are study assumptions,
not undocumented claims about the authors' implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..atlas_full_model import saturation
from ..fit import expert_log_coordinate

DOMAIN_MARGIN = 1e-10  # D_eff / D_target; loss-blind declared input domain.


@dataclass(frozen=True)
class Spec:
    name: str
    form: str
    parameter_count: str = "active"
    signed: bool = False
    tau_cap: float = 100.
    lambda_cap: float = 10.

    def payload(self):
        return asdict(self)


def panel():
    result = [Spec("Chinchilla-total-local", "chinchilla"),
              Spec("Chinchilla-target-local", "target"),
              Spec("JointMoE-JH-local", "joint"),
              Spec("ATLAS-capacity-local", "capacity"),
              Spec("He-family-local", "he")]
    for form in ("full", "other", "moe"):
        for count in (("active", "total") if form != "moe" else ("active",)):
            for signed in (False, True):
                for tau in (100., 1000., 10000.):
                    for rate in (10., 100.):
                        name = f"ATLAS-{form}-{count}-{'signed' if signed else 'positive'}-T{tau:g}-L{rate:g}"
                        result.append(Spec(name, form, count, signed, tau, rate))
    return tuple(result)


def names(spec):
    if spec.form in ("moe", "joint"):
        nonlinear = ("alpha_e1", "alpha_e32", "beta_e1", "beta_e32", "delta", "omega")
    else:
        nonlinear = ("alpha", "beta")
    if spec.form == "capacity":
        nonlinear += ("phi", "psi")
    if spec.form == "he":
        nonlinear += ("gamma",)
    if spec.form in ("full", "moe"):
        nonlinear += ("lambda", "tau_1", "tau_2", "tau_3", "tau_other")
    if spec.form == "other":
        nonlinear += ("lambda", "tau_other")
    return ("floor_fraction", "A", "B", *nonlinear)


def bounds(spec):
    lower, upper = [], []
    for name in names(spec):
        if name == "floor_fraction":
            lo, hi = 1e-10, 1.-1e-10
        elif name in ("A", "B"):
            lo, hi = 0., np.inf
        elif name.startswith(("alpha", "beta")):
            lo, hi = .01, 2.
        elif name in ("delta", "omega", "phi", "psi"):
            lo, hi = -1., 1.
        elif name == "gamma":
            lo, hi = 0., 2.
        elif name == "lambda":
            lo, hi = 0., spec.lambda_cap
        else:
            lo, hi = (-spec.tau_cap if spec.signed else 0.), spec.tau_cap
        lower.append(lo)
        upper.append(hi)
    return np.array(lower), np.array(upper)


def components(spec, packet):
    D, U = np.asarray(packet["D"], float), np.asarray(packet["U"], float)
    if D.shape != U.shape or D.ndim != 2 or D.shape[1] != 5:
        raise ValueError("ATLAS needs five separate exposure and inventory columns")
    if spec.form == "other":
        D = np.column_stack((D[:, 0], D[:, 1:].sum(axis=1)))
        U = np.column_stack((U[:, 0], U[:, 1:].sum(axis=1)))
    return D, U


def exposure(spec, x, packet):
    """Effective exposure and its complete physical-coordinate derivative."""
    labels = names(spec)
    jac = np.zeros((len(packet["N"]), len(labels)))
    if "lambda" not in labels:
        D = np.asarray(packet["D"], float)
        return (D[:, 0] if spec.form in ("target", "capacity") else D.sum(axis=1)), jac
    D, U = components(spec, packet)
    rate_index = labels.index("lambda")
    tau_indices = [i for i, key in enumerate(labels) if key.startswith("tau_")]
    sat, dsat = saturation(D, U, x[rate_index])
    weights = np.r_[1., np.asarray(x)[tau_indices]]
    jac[:, rate_index] = dsat @ weights
    jac[:, tau_indices] = sat[:, 1:]
    return sat @ weights, jac


def domain(spec, x, packet):
    value, jac = exposure(spec, x, packet)
    target = np.asarray(packet["D"], float)[:, 0]
    return value / target - DOMAIN_MARGIN, jac / target[:, None]


def predict(spec, x, packet, *, trial_extension=False):
    """Raw NLL and analytic Jacobian; accepted predictions never clip exposure.

    The least-squares audit may visit infeasible trial points. Its explicit
    constant extension is paired with separate domain residuals, and never
    used for accepting or scoring a fit.
    """
    labels = names(spec)
    x = np.asarray(x, float)
    if x.shape != (len(labels),) or not np.isfinite(x).all():
        raise ValueError("wrong or nonfinite model coordinates")
    values = dict(zip(labels, x))
    effective, exposure_jac = exposure(spec, x, packet)
    threshold = DOMAIN_MARGIN * np.asarray(packet["D"], float)[:, 0]
    if not trial_extension and (np.any(effective < threshold) or not np.isfinite(effective).all()):
        raise ValueError("effective exposure outside declared positive domain")
    invalid = effective < threshold
    effective = np.maximum(effective, threshold) if trial_extension else effective
    exposure_jac = exposure_jac.copy()
    if trial_extension:
        exposure_jac[invalid] = 0.
    n = np.asarray(packet["N_total" if spec.parameter_count == "total" else "N"], float)
    logn, logd = np.log(n/1e8), np.log(effective/1e9)
    nrows = len(n)
    cj, dj = np.zeros((nrows, len(x))), np.zeros((nrows, len(x)))
    if spec.form in ("moe", "joint"):
        g = expert_log_coordinate("JH", packet["E"])
        h = g / np.log(32.)
        alpha = values["alpha_e1"]*(1-h)+values["alpha_e32"]*h
        beta = values["beta_e1"]*(1-h)+values["beta_e32"]*h
        logcap = values["delta"]*g-alpha*logn
        logdata = values["omega"]*g-beta*logd
        for key, value in (("alpha_e1", -(1-h)*logn), ("alpha_e32", -h*logn), ("delta", g)):
            cj[:, labels.index(key)] = value
        for key, value in (("beta_e1", -(1-h)*logd), ("beta_e32", -h*logd), ("omega", g)):
            dj[:, labels.index(key)] = value
    else:
        alpha, beta = values["alpha"], values["beta"]
        logcap, logdata = -alpha*logn, -beta*logd
        cj[:, labels.index("alpha")] = -logn
        dj[:, labels.index("beta")] = -logd
    if spec.form == "capacity":
        logk = np.log(packet["K"])
        logcap += values["phi"]*logk
        logdata += values["psi"]*logk
        cj[:, labels.index("phi")] = logk
        dj[:, labels.index("psi")] = logk
    dj -= np.asarray(beta)[..., None]*exposure_jac/effective[:, None]
    cap, data = np.exp(logcap), np.exp(logdata)
    prediction = x[0]*packet["training_minimum"]+x[1]*cap+x[2]*data
    jac = x[1]*cap[:, None]*cj+x[2]*data[:, None]*dj
    jac[:, :3] = np.column_stack((np.full(nrows, packet["training_minimum"]), cap, data))
    if spec.form == "he":
        share = np.asarray(packet["family_share"], float)
        if np.any(share <= 0) or np.any(share > 1+1e-12):
            raise ValueError("invalid family sampling share")
        factor = share**(-values["gamma"])
        prediction *= factor
        jac *= factor[:, None]
        jac[:, labels.index("gamma")] = -prediction*np.log(share)
    if not np.isfinite(prediction).all() or not np.isfinite(jac).all():
        raise FloatingPointError("nonfinite prediction or derivative")
    return prediction, jac
