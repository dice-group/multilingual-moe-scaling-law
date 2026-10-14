"""Conditioned variable projection, independent joint audit, and ambiguity probes."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize
from scipy.stats import qmc

from ..atlas_full_model import STARTS
from ..baseline_refit_workflow.recovery_numerics import weighted_huber_loss
from . import model as m

SETTINGS = {
    "initial_starts": 16, "escalated_starts": 64, "seed": 20260906,
    "gradient_tolerance": 1e-8, "objective_tolerance": 1e-6,
    "prediction_tolerance": 1e-4, "near_optimal_tolerances": [1e-6, 1e-5],
    "inner_kkt": 1e-11, "rank_relative": 1e-7, "maxiter": 2000,
    "direct_max_nfev": 4000, "objective_scale": 1e4,
    "deduplicate_domain": "exact D/U rows only; never training responses",
    "audit_seed_refinement_rounds": 2,
    "kkt_multiplier_solver": "bvls",
    "inner_physical_kkt": 1e-8,
}


def encode(spec, x):
    z = np.asarray(x, float).copy()
    for i, name in enumerate(m.names(spec)):
        if name.startswith("tau_"):
            z[i] = np.arcsinh(x[i])
        elif name == "lambda":
            z[i] = np.log1p(x[i])
    return z


def decode(spec, z):
    x, chain = np.asarray(z, float).copy(), np.ones(len(z))
    for i, name in enumerate(m.names(spec)):
        if name.startswith("tau_"):
            x[i], chain[i] = np.sinh(z[i]), np.cosh(z[i])
        elif name == "lambda":
            x[i], chain[i] = np.expm1(z[i]), np.exp(z[i])
    lo, hi = m.bounds(spec)
    return np.clip(x, lo, hi), chain


def starts(spec):
    lo, hi = m.bounds(spec)
    # Only nonlinear coordinates are initialized; the convex inner solve
    # refits all three amplitudes at every evaluation.
    zlo, zhi = encode(spec, lo), encode(spec, hi)
    samples = qmc.Sobol(len(lo)-3, scramble=True, seed=SETTINGS["seed"]).random_base2(6)
    result = []
    for row in samples:
        z = np.r_[.5, .5, .5, zlo[3:]+row*(zhi[3:]-zlo[3:])]
        x, _ = decode(spec, z)
        # Feasible initial signed starts; the optimizer can subsequently
        # discover interference. No test outcome defines an initialization.
        for j, name in enumerate(m.names(spec)):
            if name.startswith("tau_"):
                x[j] = abs(x[j])
        result.append(x)
    for i, (alpha, beta, rate, tau) in enumerate(STARTS):
        for j, name in enumerate(m.names(spec)):
            if name.startswith("alpha"):
                result[i][j] = alpha
            elif name.startswith("beta"):
                result[i][j] = beta
            elif name == "lambda":
                result[i][j] = rate
            elif name.startswith("tau_"):
                result[i][j] = tau
            elif name in ("delta", "omega", "phi", "psi", "gamma"):
                result[i][j] = 0.
    return result


def objective(spec, x, packet):
    pred, jac = m.predict(spec, x, packet)
    weights = np.asarray(packet["weights"], float)
    weights = weights/weights.sum()
    residual = pred-np.asarray(packet["y"], float)
    delta = packet["delta"]
    loss = np.where(abs(residual) <= delta, .5*residual**2, delta*(abs(residual)-.5*delta))
    return float(weights @ loss), jac.T @ (weights*np.clip(residual, -delta, delta)), pred, jac


def compact_domain(support):
    """Remove exactly repeated inequalities, preserving their full feasible set.

    Domain constraints depend on D/U, not N/E/K or loss. Multiplicities retain
    the identical joint-audit penalty even at infeasible trial points. The
    full, original support is still used for final physical KKT certification.
    """
    values = np.column_stack((support["D"], support["U"]))
    _, indices, inverse, multiplicities = np.unique(values, axis=0, return_index=True,
                                                   return_inverse=True, return_counts=True)
    result = dict(support)
    for key in ("D", "U", "N", "N_total", "E", "K", "family_share", "trajectory_ids", "readout_ids"):
        if key in support:
            result[key] = np.asarray(support[key])[indices]
    return result, multiplicities, inverse


def inner(spec, x, packet, *, physical_kkt=None):
    if physical_kkt is None:
        physical_kkt = SETTINGS["inner_physical_kkt"]
    _, jac = m.predict(spec, x, packet)
    design = jac[:, :3]
    weights = np.asarray(packet["weights"], float)
    weights = weights/weights.sum()
    scales = np.sqrt(np.sum(weights[:, None]*design**2, axis=0))
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("degenerate convex design")
    design = design/scales
    physical_lo, physical_hi = m.bounds(spec)
    lo, hi = physical_lo[:3]*scales, physical_hi[:3]*scales
    robust = np.ones(len(weights))
    for _ in range(500):
        root = np.sqrt(weights*robust)
        result = lsq_linear(root[:, None]*design, root*np.asarray(packet["y"]),
                           bounds=(lo, hi), method="bvls", tol=1e-14, max_iter=1000)
        if not result.success:
            raise RuntimeError("convex bounded least-squares failure")
        residual = design @ result.x-np.asarray(packet["y"])
        gradient = design.T @ (weights*np.clip(residual, -packet["delta"], packet["delta"]))
        conditioned_ok = np.max(abs(result.x-np.clip(result.x-gradient, lo, hi))) <= SETTINGS["inner_kkt"]
        # Undoing conditioning can round an active bound one ULP outside the
        # physical box. Project back to the ORIGINAL box before its unchanged
        # physical KKT and strict feasibility checks.
        physical = np.clip(result.x/scales, physical_lo[:3], physical_hi[:3])
        physical_error = np.max(abs(physical-np.clip(
            physical-gradient*scales, physical_lo[:3], physical_hi[:3])))
        if conditioned_ok and (physical_kkt is None or physical_error <= physical_kkt):
            return physical
        robust = np.minimum(1., packet["delta"]/np.maximum(abs(residual), 1e-300))
    raise RuntimeError("convex Huber solve did not pass unchanged KKT threshold")


def stationarity(spec, x, gradient, support):
    """Physical-coordinate box/nonlinear KKT, including complementarity."""
    lo, hi = m.bounds(spec)
    slack, normals = [], []
    eye = np.eye(len(x))
    for i in range(len(x)):
        if x[i]-lo[i] <= 1e-7*max(1., abs(lo[i])):
            slack.append(max(0., x[i]-lo[i])); normals.append(eye[i])
        if np.isfinite(hi[i]) and hi[i]-x[i] <= 1e-7*max(1., abs(hi[i])):
            slack.append(max(0., hi[i]-x[i])); normals.append(-eye[i])
    domain, dj = m.domain(spec, x, support)
    if np.min(domain) < -1e-12:
        return float("inf")
    if spec.signed:
        for i in np.flatnonzero(domain <= 1e-8):
            slack.append(max(0., domain[i])); normals.append(dj[i])
    if not normals:
        return float(np.max(abs(gradient)))
    matrix = np.asarray(normals).T
    # TRF's absolute optimality stopping rule can leave spurious ~1e-7
    # multipliers when the true constrained residual is ~1e-11. Solve the
    # SAME nonnegative multiplier problem with an active-set method instead;
    # neither the constraint cone nor the 1e-8 acceptance threshold changes.
    result = lsq_linear(matrix, gradient, bounds=(0., np.inf), method="bvls", tol=1e-14, max_iter=1000)
    if not result.success:
        return float("inf")
    return float(max(np.max(abs(gradient-matrix@result.x)), np.max(result.x*np.asarray(slack))))


def record(spec, x, packet, support, *, kind, index, flag, message, runtime):
    value, gradient, _, _ = objective(spec, x, packet)
    kkt = stationarity(spec, x, gradient, support)
    lo, hi = m.bounds(spec)
    feasible = (np.min(m.domain(spec, x, support)[0]) >= 0.
                and np.all(x >= lo) and np.all(x <= hi))
    return dict(kind=kind, index=index, coordinates=np.asarray(x).tolist(), objective=value,
                optimizer_success=bool(flag), kkt=kkt if np.isfinite(kkt) else None,
                success=bool(flag and feasible and kkt <= SETTINGS["gradient_tolerance"]),
                message=str(message), runtime_seconds=runtime)


def run_start(spec, initial, packet, support, index, *, fixed=None, physical_kkt=None):
    began = time.monotonic()
    fixed = fixed or {}
    lo, hi = m.bounds(spec)
    zlo, zhi = encode(spec, lo), encode(spec, hi)
    initial = np.asarray(initial, float).copy()
    for j, value in fixed.items():
        initial[j] = value
    free = [i for i in range(3, len(lo)) if i not in fixed]
    seed = encode(spec, initial)
    constraint_support, _, _ = compact_domain(support)
    cache = {}

    def unpack(q):
        z = seed.copy(); z[free] = q
        return decode(spec, z)

    def constraint(q):
        key = np.asarray(q).tobytes()
        if cache.get("key") != key:
            x, chain = unpack(q)
            domain, jac = m.domain(spec, x, constraint_support)
            cache.update(key=key, domain=domain, jac=(jac*chain)[:, free])
        return cache["domain"], cache["jac"]

    def evaluate(q):
        x, chain = unpack(q)
        domain, dj = constraint(q)
        if np.min(domain) < 0:
            # Explicit infeasible-trial merit, never accepted as an objective.
            bad = np.minimum(domain, 0.)
            return 1e8+1e8*float(bad@bad), 2e8*dj.T@bad
        x[:3] = inner(spec, x, packet, physical_kkt=physical_kkt)
        value, gradient, _, _ = objective(spec, x, packet)
        return value*SETTINGS["objective_scale"], (gradient*chain)[free]*SETTINGS["objective_scale"]

    constraints = ()
    if spec.signed:
        constraints = ({"type": "ineq", "fun": lambda q: constraint(q)[0],
                        "jac": lambda q: constraint(q)[1]},)
    method = "SLSQP" if spec.signed else "L-BFGS-B"
    options = {"maxiter": SETTINGS["maxiter"], "ftol": 1e-12 if spec.signed else 1e-15}
    if not spec.signed:
        options.update(gtol=1e-6, maxls=50, maxcor=20)
    result = minimize(evaluate, seed[free], jac=True, method=method,
                      bounds=list(zip(zlo[free], zhi[free])), constraints=constraints, options=options)
    x, _ = unpack(result.x)
    x[:3] = inner(spec, x, packet, physical_kkt=physical_kkt)
    return record(spec, x, packet, support, kind="profile" if fixed else "start", index=index,
                  flag=result.success, message=result.message, runtime=time.monotonic()-began) | {"fixed": fixed}


def direct(spec, candidate, packet, support, *, method="trf", max_nfev=None, gtol=1e-12):
    """Independent joint least-squares in original, not projected coordinates."""
    began = time.monotonic()
    weights = np.asarray(packet["weights"], float)
    weights = weights/weights.sum()
    constraint_support, multiplicities, _ = compact_domain(support)
    penalty_scale = 1e4*np.sqrt(multiplicities)
    nsupport = len(constraint_support["N"])
    # Quadratic domain residuals are separate from weighted Huber data loss.
    loss_data = weighted_huber_loss(weights)
    def loss(z):
        tail = np.vstack((z[len(weights):], np.ones(nsupport), np.zeros(nsupport)))
        return np.column_stack((loss_data(z[:len(weights)]), tail))
    def residual(x):
        pred, _ = m.predict(spec, x, packet, trial_extension=True)
        domain, _ = m.domain(spec, x, constraint_support)
        return np.r_[pred-np.asarray(packet["y"]), penalty_scale*np.minimum(domain, 0.)]
    def jacobian(x):
        _, jac = m.predict(spec, x, packet, trial_extension=True)
        domain, dj = m.domain(spec, x, constraint_support)
        return np.vstack((jac, penalty_scale[:, None]*dj*(domain < 0)[:, None]))
    result = least_squares(residual, candidate["coordinates"], jac=jacobian, method=method,
        bounds=m.bounds(spec), loss=loss, f_scale=packet["delta"], x_scale="jac",
        ftol=1e-14, xtol=1e-14, gtol=gtol, max_nfev=max_nfev or SETTINGS["direct_max_nfev"])
    return record(spec, result.x, packet, support, kind="direct", index=-1,
        flag=result.success, message=result.message, runtime=time.monotonic()-began) | {"solver": method}


def safe(function, **kwargs):
    began = time.monotonic()
    try:
        return function(**kwargs)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {"success": False, "kind": function.__name__, "index": kwargs.get("index", -1),
                "message": f"{type(error).__name__}: {error}", "runtime_seconds": time.monotonic()-began}


def choose(attempts):
    good = [a for a in attempts if a.get("success")]
    return min(good, key=lambda a: (a["objective"], a["index"])) if good else None


def agreement(spec, left, right, packet):
    if left is None or not right.get("success"):
        return False
    difference = abs(left["objective"]-right["objective"])/max(abs(left["objective"]), 1e-15)
    a, b = (m.predict(spec, row["coordinates"], packet)[0] for row in (left, right))
    return bool(difference <= SETTINGS["objective_tolerance"] and np.max(abs(a-b)) <= SETTINGS["prediction_tolerance"])


def fit_group(spec, packet, support):
    began = time.monotonic()
    seeds = starts(spec)
    attempts = [safe(run_start, spec=spec, initial=seeds[i], packet=packet, support=support, index=i) for i in range(16)]
    best = choose(attempts)
    audit = safe(direct, spec=spec, candidate=best, packet=packet, support=support) if best else {"success": False}
    audit_history = [audit]
    first, second = choose(attempts[:8]), choose(attempts[8:])
    improved = first is None or (second is not None and
        first["objective"]-second["objective"] > 1e-6*max(abs(first["objective"]), 1e-15))
    escalated = improved or not agreement(spec, best, audit, packet)
    if escalated:
        attempts.extend(safe(run_start, spec=spec, initial=seeds[i], packet=packet, support=support, index=i) for i in range(16, 64))
        best = choose(attempts)
        audit = safe(direct, spec=spec, candidate=best, packet=packet, support=support) if best else {"success": False}
        audit_history.append(audit)
    probes = []
    geometry = {}
    if best is not None:
        x = np.asarray(best["coordinates"])
        _, _, _, jac = objective(spec, x, packet)
        scale = np.maximum(abs(x), 1.)
        scaled = jac*scale
        singular = np.linalg.svd(scaled, compute_uv=False)
        rank = int(np.sum(singular > singular[0]*SETTINGS["rank_relative"]))
        lo, hi = m.bounds(spec)
        contacts = [key for i, key in enumerate(m.names(spec))
                    if abs(x[i]-lo[i]) <= 1e-6*max(1., abs(lo[i])) or
                    (np.isfinite(hi[i]) and abs(x[i]-hi[i]) <= 1e-6*max(1., abs(hi[i])))]
        geometry = dict(rank=rank, coordinate_count=len(x), singular_values=singular.tolist(), contacts=contacts)
        for j, key in enumerate(m.names(spec)):
            if key != "lambda" and not key.startswith("tau_"):
                continue
            if rank == len(x) and key not in contacts:
                continue
            values = sorted(set([0., float(x[j]/2), float(hi[j])]+([float(lo[j])] if spec.signed and key.startswith("tau_") else [])))
            for value in values:
                probe = safe(run_start, spec=spec, initial=x, packet=packet, support=support,
                             index=1000+len(probes), fixed={j: value})
                probes.append(probe)
        better = [a for a in probes if "objective" in a and a["objective"] < best["objective"]]
        if better:
            anchor = min(better, key=lambda a: a["objective"])
            attempts.append(safe(run_start, spec=spec, initial=anchor["coordinates"], packet=packet,
                                 support=support, index=64))
            best = choose(attempts)
            audit = safe(direct, spec=spec, candidate=best, packet=packet, support=support)
            audit_history.append(audit)
    # An independent audit can reveal a better training basin. Follow it with
    # a NEW variable-projection solve and another joint audit, never accept an
    # audit as evidence of agreement with itself or change the heldout law.
    for round_index in range(SETTINGS["audit_seed_refinement_rounds"]):
        if best is None or agreement(spec, best, audit, packet) or not audit.get("success"):
            break
        if audit["objective"] >= best["objective"]:
            break
        attempts.append(safe(run_start, spec=spec, initial=audit["coordinates"], packet=packet,
                             support=support, index=200+round_index))
        best = choose(attempts)
        audit = safe(direct, spec=spec, candidate=best, packet=packet, support=support)
        audit_history.append(audit)
    if best is not None:
        x = np.asarray(best["coordinates"])
        jac = objective(spec, x, packet)[3]
        singular = np.linalg.svd(jac*np.maximum(abs(x), 1.), compute_uv=False)
        lo, hi = m.bounds(spec)
        geometry = dict(rank=int(np.sum(singular > singular[0]*SETTINGS["rank_relative"])),
            coordinate_count=len(x), singular_values=singular.tolist(), point="final selected coefficients",
            contacts=[key for i, key in enumerate(m.names(spec)) if
                abs(x[i]-lo[i]) <= 1e-6*max(1., abs(lo[i])) or
                (np.isfinite(hi[i]) and abs(x[i]-hi[i]) <= 1e-6*max(1., abs(hi[i])))])
    return dict(spec=spec.payload(), attempts=attempts, selected=best, audit=audit, probes=probes,
                audit_history=audit_history, geometry=geometry, escalated=escalated,
                certified=agreement(spec, best, audit, packet),
                runtime_seconds=time.monotonic()-began, heldout_scores_computed=False)
