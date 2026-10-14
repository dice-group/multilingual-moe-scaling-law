"""One shared exponent on M6's data term; historical fitting code is untouched.

L = L_inf + A * capacity_M6 + B * data_M6 * (N / 1e8)**zeta.
The conditioned convex inner problem and the original ridge penalty are kept.
"""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
from scipy.optimize import minimize

from .. import fit
from ..baseline_refit_workflow.recovery_numerics import (
    OBJECTIVE_SCALE, OUTER_SETTINGS as ORIGINAL_OUTER_SETTINGS,
)
from .numerics import solve_language

MODEL = "M6-ND"
# The optimizer sees a 10000x scaled objective. A 1e-9 unscaled stopping
# threshold is ten times stricter than the frozen 1e-8 KKT acceptance check.
# The original 1e-10 scaled setting demanded 1e-14 unscaled and could produce
# line-search failure at an already stationary solution. Do not waive flags.
OPTIMIZER_UNSCALED_GTOL = 1e-9
OUTER_SETTINGS = dict(ORIGINAL_OUTER_SETTINGS, gtol=OBJECTIVE_SCALE*OPTIMIZER_UNSCALED_GTOL)
ZETA_STARTS = (0., .01, .03, .1, .3, .5)
NULL_CONTACTS = {"zeta", "eta_family", "eta_script", "gamma_other"}


def terms(coordinates, problem, *, require_positive_exponents=True):
    x = np.asarray(coordinates, dtype=float)
    if x.shape != (15,) or not np.isfinite(x).all():
        raise ValueError("M6-ND needs 15 finite shared coordinates")
    p = problem
    cap, data, cj, dj = fit.predictor_terms_and_jacobian(
        "M6", x[:14], p.N, p.D, p.E, p.K,
        D_family=p.D_family, D_script=p.D_script, D_other=p.D_other, require_positive_exponents=require_positive_exponents)
    log_n = np.log(p.N / fit.N_SCALE)
    factor = np.exp(x[14] * log_n)
    modified = data * factor
    return cap, modified, np.column_stack((cj, np.zeros(len(cap)))), np.column_stack((dj * factor[:, None], modified * log_n))


def penalty(coordinates, problem):
    value, gradient = fit.regularization_objective_and_gradient("M6", coordinates[:14], problem.regularization)
    return value, np.r_[gradient, 0.]


def projected_gradient(x, gradient, bounds):
    lower, upper = np.asarray([(b.lower, b.upper) for b in bounds]).T
    return float(np.max(np.abs(x - np.clip(x - gradient, lower, upper))))


class Profile:
    def __init__(self, problem, delta):
        self.problem, self.delta = problem, delta
        ids = np.asarray(problem.language_ids)
        self.groups = [(name, np.flatnonzero(ids == name)) for name in sorted(set(ids))]
        self.fallback_solves = 0

    def inner_problem(self, indices, cap, data):
        p = self.problem
        return fit.LanguageProblem(p.y[indices], cap[indices], data[indices], p.weights[indices],
            float(p.physical_floors[indices[0]]), float(p.y[indices].min()))

    def evaluate(self, x):
        p = self.problem
        cap, data, cj, dj = terms(x, p)
        prediction = np.empty_like(p.y)
        jacobian = np.empty((len(p.y), 15))
        coefficients = []
        for name, indices in self.groups:
            solution = solve_language(self.inner_problem(indices, cap, data), self.delta, language=name)
            self.fallback_solves += int(solution.message.startswith("IRLS refinement"))
            prediction[indices] = solution.prediction
            jacobian[indices] = solution.A * cj[indices] + solution.B * dj[indices]
            coefficients.append({k: getattr(solution, k) for k in (
                "language", "L_inf", "A", "B", "floor_fraction", "floor_saturated", "A_zero_contact", "B_zero_contact")})
        objective, gradient = fit.weighted_objective_and_gradient(prediction-p.y, jacobian, p.weights, self.delta)
        ridge, ridge_gradient = penalty(x, p)
        return {"coordinates": np.asarray(x).tolist(), "objective": float(objective+ridge),
            "objective_components": {"data_fit": float(objective), "regularization": float(ridge)},
            "language_coefficients": coefficients}, gradient+ridge_gradient, prediction, jacobian

    def run(self, coordinates, bounds):
        began = time.monotonic()
        initial_fallbacks = self.fallback_solves
        def objective(x):
            value, gradient, _, _ = self.evaluate(x)
            return OBJECTIVE_SCALE * value["objective"], OBJECTIVE_SCALE * gradient
        result = minimize(objective, coordinates, method="L-BFGS-B", jac=True,
            bounds=[(b.lower, b.upper) for b in bounds], options=OUTER_SETTINGS)
        record, gradient, _, _ = self.evaluate(result.x)
        kkt = projected_gradient(result.x, gradient, bounds)
        return record | {"success": bool(result.success and kkt <= 1e-8), "optimizer_success": bool(result.success),
            "status": int(result.status), "message": str(result.message), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "projected_gradient_inf": kkt,
            "runtime_seconds": time.monotonic()-began,
            "inner_fallback_solves": self.fallback_solves-initial_fallbacks,
            "boundary_contacts": [asdict(c) for c in fit.boundary_contacts(result.x, bounds)]}


class Direct:
    """Independent joint optimization: no inner profiling during objective calls.

Nuisance coordinates are logit(floor fraction), log(A), log(B), as in the
historical direct check. Initialization is the selected profiled solution.
"""
    def __init__(self, profile, selected, bounds):
        self.profile, self.outer_bounds = profile, bounds
        self.initial = list(selected["coordinates"])
        coefficients = {c["language"]: c for c in selected["language_coefficients"]}
        self.bounds = [(b.lower, b.upper) for b in bounds]
        for name, _ in profile.groups:
            c = coefficients[name]
            fraction = np.clip(c["floor_fraction"], 1e-12, 1-1e-12)
            self.initial.extend([float(np.log(fraction/(1-fraction))), float(np.log(c["A"])), float(np.log(c["B"]))])
            self.bounds.extend([(None, None), (None, None), (None, None)])

    def evaluate(self, vector):
        from scipy.special import expit
        p, delta = self.profile.problem, self.profile.delta
        x = np.asarray(vector[:15])
        cap, data, cj, dj = terms(x, p)
        prediction = np.empty_like(p.y)
        gradient = np.zeros(len(vector))
        coefficients = []
        objective = 0.
        for i, (name, indices) in enumerate(self.profile.groups):
            offset = 15 + 3*i
            c, a, b = vector[offset:offset+3]
            fraction, A, B = expit(c), np.exp(a), np.exp(b)
            low, high = self.profile.inner_problem(indices, cap, data).strict_bounds
            interval = high[0]-low[0]
            floor = low[0]+interval*fraction
            prediction[indices] = floor+A*cap[indices]+B*data[indices]
            residual = prediction[indices]-p.y[indices]
            loss, score = fit.loss_and_score(residual, delta)
            weighted_score = p.weights[indices]*score
            objective += float(p.weights[indices] @ loss)
            gradient[:15] += (A*cj[indices]+B*dj[indices]).T @ weighted_score
            gradient[offset:offset+3] = [weighted_score.sum()*interval*fraction*(1-fraction),
                weighted_score @ (A*cap[indices]), weighted_score @ (B*data[indices])]
            coefficients.append({"language": name, "L_inf": float(floor), "A": float(A), "B": float(B),
                "floor_fraction": float(fraction)})
        ridge, rg = penalty(x, p)
        gradient[:15] += rg
        if not np.isfinite(prediction).all() or not np.isfinite(gradient).all():
            raise ValueError("non-finite direct joint evaluation")
        return float(objective+ridge), gradient, prediction, coefficients

    def run(self):
        began = time.monotonic()
        def objective(x):
            value, gradient, _, _ = self.evaluate(x)
            return OBJECTIVE_SCALE*value, OBJECTIVE_SCALE*gradient
        result = minimize(objective, self.initial, method="L-BFGS-B", jac=True,
            bounds=self.bounds, options=OUTER_SETTINGS)
        value, gradient, _, coefficients = self.evaluate(result.x)
        kkt = max(projected_gradient(result.x[:15], gradient[:15], self.outer_bounds),
            float(np.max(np.abs(gradient[15:]))))
        return {"success": bool(result.success and kkt <= 1e-8), "optimizer_success": bool(result.success),
            "objective": value, "coordinates": result.x[:15].tolist(), "language_coefficients": coefficients,
            "message": str(result.message), "status": int(result.status), "iterations": int(result.nit),
            "function_evaluations": int(result.nfev), "projected_gradient_inf": kkt,
            "runtime_seconds": time.monotonic()-began}


def predict(problem, selected):
    cap, data, _, _ = terms(selected["coordinates"], problem)
    by_language = {c["language"]: c for c in selected["language_coefficients"]}
    c = [by_language[name] for name in problem.language_ids]
    return np.asarray([v["L_inf"] for v in c]) + np.asarray([v["A"] for v in c])*cap + np.asarray([v["B"] for v in c])*data


def equivalence(left, right, problem):
    relative = abs(left["objective"]-right["objective"])/max(abs(left["objective"]), np.finfo(float).tiny)
    maximum = float(np.max(np.abs(predict(problem, left)-predict(problem, right))))
    return {"equivalent": bool(left["success"] and right["success"] and relative <= 1e-6 and maximum <= 1e-4),
        "relative_objective_difference": float(relative), "maximum_absolute_prediction_difference": maximum}


def numerical_health(profile, selected, bounds):
    _, gradient, _, jacobian = profile.evaluate(selected["coordinates"])
    widths = np.array([b.upper-b.lower for b in bounds])
    singular = np.linalg.svd(jacobian*widths[None, :], compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-8))
    contacts = [asdict(c) for c in fit.boundary_contacts(selected["coordinates"], bounds)]
    rejected = [c for c in contacts if c["side"] != "lower" or c["name"] not in NULL_CONTACTS]
    alpha, beta, _, _ = fit.effective_exponents(fit.shared_from_outer("M6", selected["coordinates"][:14]),
        profile.problem.E, profile.problem.K, model="M6")
    kkt = projected_gradient(np.array(selected["coordinates"]), gradient, bounds)
    return {"healthy": bool(rank == 15 and not rejected and min(alpha) > 0 and min(beta) > 0 and kkt <= 1e-8),
        "shared_prediction_jacobian_rank": rank, "required_rank": 15, "singular_values": singular.tolist(),
        "rank_scope": "bound-scaled shared prediction Jacobian, nuisance held fixed; not global identifiability",
        "rejected_boundary_contacts": rejected, "boundary_contacts": contacts,
        "minimum_alpha": float(min(alpha)), "minimum_beta": float(min(beta)), "projected_gradient_inf": kkt}


def run_attempt(profile, start, bounds):
    began = time.monotonic()
    try:
        record = profile.run(start["coordinates"], bounds)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        record = {"success": False, "message": f"{type(error).__name__}: {error}",
            "runtime_seconds": time.monotonic()-began}
    return record | {"start_id": start["start_id"], "fold": start["fold"], "initial_coordinates": start["coordinates"]}
