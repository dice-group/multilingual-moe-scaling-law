"""ATLAS equations 1--3, with per-target coefficients and analytic derivatives.

The parameterization and bounds are harmonized study choices, not author code.
U counts unique original-document training tokens, before rehydration copies.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import least_squares

from .baseline_refit_workflow.recovery_numerics import weighted_huber_loss

NAMES = ("floor_fraction", "A", "B", "alpha", "beta", "lambda",
         "tau_1", "tau_2", "tau_3", "tau_other")
LOWER = np.array([1e-10, 0., 0., .01, .01, 0., 0., 0., 0., 0.])
UPPER = np.array([1.-1e-10, np.inf, np.inf, 2., 2., 10., 100., 100., 100., 100.])
STARTS = ((.3, .3, .01, .01), (.5, .2, .1, .1),
          (.2, .5, 1., 1.), (.8, .15, 3., .001))
SETTINGS = dict(ftol=1e-13, xtol=1e-13, gtol=1e-11, max_nfev=2000)


def saturation(D, U, rate):
    """Return S and dS/dlambda, including the exact lambda=0 limit."""
    D, U = np.broadcast_arrays(np.asarray(D, float), np.asarray(U, float))
    if not np.isfinite(D).all() or not np.isfinite(U).all() or np.any(D < 0) or np.any(U < 0):
        raise ValueError("invalid ATLAS exposure/inventory")
    if np.any((U == 0) & (D != 0)) or not np.isfinite(rate) or rate < 0:
        raise ValueError("positive exposure needs positive inventory; rate must be nonnegative")
    safe_u = np.where(U > 0, U, 1.)
    r = np.maximum(D / safe_u - 1., 0.)
    z = rate * r
    small = z < 1e-4
    factor = np.empty_like(z)
    derivative = np.empty_like(z)
    factor[small] = 1-z[small]/2+z[small]**2/6-z[small]**3/24+z[small]**4/120
    derivative[small] = -.5+z[small]/3-z[small]**2/8+z[small]**3/30
    factor[~small] = -np.expm1(-z[~small])/z[~small]
    derivative[~small] = ((z[~small]+1)*np.exp(-z[~small])-1)/z[~small]**2
    value = np.where(D <= U, D, U*(1+r*factor))
    grad = np.where(D <= U, 0., U*r*r*derivative)
    return value, grad


def predict(x, packet):
    """Prediction and unweighted full ten-coordinate Jacobian."""
    x = np.asarray(x, float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("ATLAS needs ten finite coordinates")
    D, U = np.asarray(packet["D"]), np.asarray(packet["U"])
    if D.ndim != 2 or D.shape[1] != 5 or D.shape != U.shape:
        raise ValueError("target, three donors and remainder must be separate columns")
    sat, dsat = saturation(D, U, x[5])
    weights = np.r_[1., x[6:]]
    effective = sat @ weights
    if np.any(effective <= 0):
        raise ValueError("effective exposure must be positive")
    logn = np.log(np.asarray(packet["N"])/1e8)
    logd = np.log(effective/1e9)
    cap, data = np.exp(-x[3]*logn), np.exp(-x[4]*logd)
    span = packet["training_minimum"]-packet["physical_floor"]
    prediction = packet["physical_floor"]+x[0]*span+x[1]*cap+x[2]*data
    jac = np.zeros((len(D), 10))
    jac[:, 0], jac[:, 1], jac[:, 2] = span, cap, data
    jac[:, 3], jac[:, 4] = -x[1]*cap*logn, -x[2]*data*logd
    multiplier = -x[2]*x[4]*data/effective
    jac[:, 5] = multiplier*(dsat @ weights)
    jac[:, 6:] = multiplier[:, None]*sat[:, 1:]
    return prediction, jac


def fit_start(packet, index):
    """Independent bounded robust least squares; no holdout data accepted."""
    began = time.monotonic()
    alpha, beta, rate, tau = STARTS[index]
    x = np.array([.8, .5, .5, alpha, beta, rate, tau, tau, tau, tau])
    y = np.asarray(packet["y"])
    weights = np.asarray(packet["weights"])
    weights = weights/weights.sum()  # Constant rescaling within a target.
    result = least_squares(lambda q: predict(q, packet)[0]-y, x,
        jac=lambda q: predict(q, packet)[1], bounds=(LOWER, UPPER),
        loss=weighted_huber_loss(weights), f_scale=packet["delta"],
        x_scale="jac", **SETTINGS)
    pred, jac = predict(result.x, packet)
    residual = pred-y
    gradient = jac.T @ (weights*np.clip(residual, -packet["delta"], packet["delta"]))
    kkt = float(np.max(np.abs(result.x-np.clip(result.x-gradient, LOWER, UPPER))))
    weighted_jac = np.sqrt(weights[:, None])*jac
    norms = np.linalg.norm(weighted_jac, axis=0)
    singular = np.linalg.svd(weighted_jac/np.maximum(norms, 1e-30), compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-7))
    contacts = [f"{name}:{side}" for i, name in enumerate(NAMES)
        for side, bound in (("lower", LOWER[i]), ("upper", UPPER[i]))
        if np.isfinite(bound) and abs(result.x[i]-bound) <= 1e-6*max(1., abs(bound))]
    return dict(start_index=index, initial=x.tolist(), coordinates=result.x.tolist(),
        optimizer_success=bool(result.success), success=bool(result.success and kkt <= 1e-8),
        message=str(result.message), objective=float(result.cost), nfev=int(result.nfev),
        projected_gradient_inf=kkt, normalized_jacobian_rank=rank,
        normalized_jacobian_singular_values=singular.tolist(), boundary_contacts=contacts,
        training_predictions=pred.tolist(), runtime_seconds=time.monotonic()-began)
