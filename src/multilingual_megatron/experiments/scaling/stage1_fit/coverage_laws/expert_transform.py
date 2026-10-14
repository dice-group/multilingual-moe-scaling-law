"""JointMoE expert coordinate and analytic correction to the existing RN terms.

The two new coordinates are log(E_start) and r=E_start/E_max. r=0
includes the infinite-E_max limit, and (0,0) exactly nests raw E.
Endpoint-normalized h keeps the original exponent bounds meaningful.
"""
import numpy as np

from ..repetition_model import saturation


def log_coordinate(e, log_start, ratio):
    """Log of JointMoE's harmonic transform, and its two derivatives."""
    e = np.asarray(e, dtype=float)
    if not np.isfinite(e).all() or np.any(e < 1):
        raise ValueError("expert counts must be finite and >= 1")
    s, t = np.exp(log_start), e-1.
    a = 1.-ratio
    numerator = s+a*t
    offset = ratio*a*t/s
    value = np.log(numerator)-np.log1p(offset)
    jac = np.stack((s/numerator+offset/(1.+offset),
                    -t/numerator-(1.-2.*ratio)*t/s/(1.+offset)), axis=-1)
    return value, jac


def coordinates(e, log_start, ratio):
    loge, jac = log_coordinate(e, log_start, ratio)
    endpoint, end_jac = log_coordinate(32., log_start, ratio)
    span = endpoint-log_start
    h = (loge-log_start)/span
    origin_jac = np.array([1., 0.])
    dh = (jac-origin_jac-h[..., None]*(end_jac-origin_jac))/span
    return loge, h, jac, dh


def correct(x, problem, inventory, stored, cap, term, cj, dj):
    """Change only expert geometry; reuse the canonical RN/repetition bases."""
    p = problem
    loge, h, de, dh = coordinates(p.E, x[17], x[18])
    raw_loge = np.log(p.E)
    delta_e, delta_h = loge-raw_loge, h-raw_loge/np.log(32.)
    logn, logk = np.log(p.N/1e8), np.log(p.K)
    size = p.N if stored is None else stored
    rate = x[15]*(size/1e8)**x[16]
    useful, ds = np.empty_like(p.D), np.empty_like(p.D)
    for value in np.unique(rate):
        mask = rate == value
        useful[mask], ds[mask] = saturation(p.D[mask], inventory[mask], value)
    logd = np.log(useful/1e9)
    capacity_slope = (x[7]-x[6])*logk-(x[1]-x[0])*logn
    data_slope = (x[9]-x[8])*logk-(x[3]-x[2])*logd
    cf = np.exp(x[4]*delta_e+delta_h*capacity_slope)
    df = np.exp(x[5]*delta_e+delta_h*data_slope)
    cap, term, cj, dj = cap*cf, term*df, cj*cf[:, None], dj*df[:, None]
    for index, derivative in ((0, delta_h*logn), (1, -delta_h*logn),
                              (4, delta_e), (6, -delta_h*logk), (7, delta_h*logk)):
        cj[:, index] += cap*derivative
    for index, derivative in ((2, delta_h*logd), (3, -delta_h*logd),
                              (5, delta_e), (8, -delta_h*logk), (9, delta_h*logk)):
        dj[:, index] += term*derivative
    # The beta change also changes the lambda/rho derivatives of useful exposure.
    correction = -term*delta_h*(x[3]-x[2])*ds/useful
    dj[:, 15] += correction*(size/1e8)**x[16]
    dj[:, 16] += correction*rate*np.log(size/1e8)
    cj[:, 17:] = cap[:, None]*(x[4]*de+capacity_slope[:, None]*dh)
    dj[:, 17:] = term[:, None]*(x[5]*de+data_slope[:, None]*dh)
    return cap, term, cj, dj
