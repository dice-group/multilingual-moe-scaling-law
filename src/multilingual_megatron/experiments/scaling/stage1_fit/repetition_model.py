"""Bounded post-diagnosis extension; historical M6/M6-ND code stays unchanged.

Replace the useful target exposure in the data term by ATLAS S_lambda(D_t,U_t),
preserving the raw family/script composition ratios and capacity term. One
shared rate governs diminishing returns; it is not an overtraining penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit

from . import fit
from .atlas_full_model import saturation
from .m6_size_duration_workflow import model as parent
from .m6_size_duration_workflow.numerics import refine_language, solve_language as original_solve

MODELS = ("M6-R", "M6-ND-R")


def solve_language(problem, delta, *, language=None):
    try:
        return original_solve(problem, delta, language=language)
    except ValueError as error:
        # BVLS can return an initializer a few ulps outside the same box.
        # The existing IRLS solver avoids that handoff, retaining stricter KKT.
        if str(error) != "Initial guess is outside of provided bounds":
            raise
        return refine_language(problem, delta, language=language)


def optimize(objective, initial, bounds, constrained):
    if not constrained:
        return minimize(objective, initial, method="L-BFGS-B", jac=True,
                        bounds=bounds, options=parent.OUTER_SETTINGS)
    # Preserve positive alpha(E,K), including the K200 support of a K200 holdout.
    matrix = np.zeros((2, len(initial)))
    matrix[0, [0, 10]], matrix[1, [1, 10]] = 1., 1.
    return minimize(objective, initial, method="SLSQP", jac=True, bounds=bounds,
                    constraints=LinearConstraint(matrix, 1e-6, np.inf),
                    options={"ftol": 1e-12, "maxiter": 4000})


@dataclass(frozen=True)
class Bound:
    name: str
    lower: float
    upper: float
    kind: str
    expansions: int = 0


def terms(x, problem, inventory, *, require_positive_exponents=True):
    x = np.asarray(x, dtype=float)
    inventory = np.asarray(inventory, dtype=float)
    if x.shape not in ((16,), (17,)) or not np.isfinite(x).all():
        raise ValueError("repetition model requires 16 or 17 finite coordinates")
    if inventory.shape != problem.D.shape or np.any(inventory <= 0):
        raise ValueError("aligned positive target inventories required")
    cap, data, cj, dj = parent.terms(x[:15], problem, require_positive_exponents=require_positive_exponents)
    logn = np.log(problem.N/fit.N_SCALE)
    size_factor = np.exp(x[16]*logn) if len(x) == 17 else np.ones_like(logn)
    rate = x[15]*size_factor
    useful, derivative = np.empty_like(problem.D), np.empty_like(problem.D)
    for value in np.unique(rate):
        mask = rate == value
        useful[mask], derivative[mask] = saturation(problem.D[mask], inventory[mask], value)
    log_ratio = np.log(useful / problem.D)
    q = np.log(problem.E) / np.log(32.)
    beta = x[2]*(1-q) + x[3]*q
    factor = np.exp(-beta*log_ratio)
    data = data*factor
    dj = dj*factor[:, None]
    dj[:, 2] -= data*log_ratio*(1-q)
    dj[:, 3] -= data*log_ratio*q
    rate_gradient = -data*beta*derivative/useful
    columns = [dj, rate_gradient*size_factor]
    if len(x) == 17:
        columns.append(rate_gradient*rate*logn)
    return (cap, data, np.column_stack((cj, np.zeros((len(cap), len(x)-15)))),
            np.column_stack(columns))


def penalty(x, problem):
    value, gradient = parent.penalty(x[:15], problem)
    return value, np.r_[gradient, np.zeros(len(x)-15)]


class Profile(parent.Profile):
    """Reuse the conditioned Huber inner solver and audited outer optimizer."""

    def __init__(self, problem, delta, inventory, *, constrained=False):
        super().__init__(problem, delta)
        self.inventory = np.asarray(inventory, dtype=float)
        self.constrained = constrained

    def evaluate(self, x):
        p = self.problem
        cap, data, cj, dj = terms(x, p, self.inventory)
        prediction = np.empty_like(p.y)
        jacobian = np.empty((len(p.y), len(x)))
        coefficients = []
        for language, indices in self.groups:
            solution = solve_language(self.inner_problem(indices, cap, data), self.delta, language=language)
            self.fallback_solves += int(solution.message.startswith("IRLS refinement"))
            prediction[indices] = solution.prediction
            jacobian[indices] = solution.A*cj[indices] + solution.B*dj[indices]
            coefficients.append({key: getattr(solution, key) for key in (
                "language", "L_inf", "A", "B", "floor_fraction", "floor_saturated",
                "A_zero_contact", "B_zero_contact")})
        value, gradient = fit.weighted_objective_and_gradient(prediction-p.y, jacobian, p.weights, self.delta)
        ridge, rg = penalty(x, p)
        return ({"coordinates": np.asarray(x).tolist(), "objective": float(value+ridge),
                 "objective_components": {"data_fit": float(value), "regularization": float(ridge)},
                 "language_coefficients": coefficients}, gradient+rg, prediction, jacobian)

    def run(self, coordinates, bounds):
        if not self.constrained:
            return super().run(coordinates, bounds)
        began = time.monotonic()
        def objective(x):
            value, gradient, _, _ = self.evaluate(x)
            return parent.OBJECTIVE_SCALE*value["objective"], parent.OBJECTIVE_SCALE*gradient
        result = optimize(objective, coordinates, [(b.lower, b.upper) for b in bounds], True)
        record, gradient, _, _ = self.evaluate(result.x)
        kkt = parent.projected_gradient(result.x, gradient, bounds)
        return record | {"success": bool(result.success and kkt <= 1e-8),
            "optimizer_success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "projected_gradient_inf": kkt, "runtime_seconds": time.monotonic()-began,
            "boundary_contacts": [vars(c) for c in fit.boundary_contacts(result.x, bounds)]}


def predict(problem, inventory, selected):
    cap, data, _, _ = terms(selected["coordinates"], problem, inventory)
    by_language = {v["language"]: v for v in selected["language_coefficients"]}
    c = np.array([[by_language[l][k] for k in ("L_inf", "A", "B")] for l in problem.language_ids])
    return c[:, 0] + c[:, 1]*cap + c[:, 2]*data


class Direct:
    """Independent joint optimization of shared and nuisance coordinates."""

    def __init__(self, profile, selected, bounds):
        self.profile, self.outer_bounds = profile, bounds
        self.nouter = len(selected["coordinates"])
        self.initial = list(selected["coordinates"])
        self.bounds = [(b.lower, b.upper) for b in bounds]
        coefficients = {v["language"]: v for v in selected["language_coefficients"]}
        for language, _ in profile.groups:
            c = coefficients[language]
            f = np.clip(c["floor_fraction"], 1e-12, 1-1e-12)
            self.initial.extend([float(np.log(f/(1-f))), float(np.log(c["A"])), float(np.log(c["B"]))])
            self.bounds.extend([(None, None)]*3)

    def evaluate(self, vector):
        p, delta = self.profile.problem, self.profile.delta
        cap, data, cj, dj = terms(vector[:self.nouter], p, self.profile.inventory)
        prediction, gradient = np.empty_like(p.y), np.zeros(len(vector))
        objective, coefficients = 0., []
        for i, (language, indices) in enumerate(self.profile.groups):
            offset = self.nouter+3*i
            c, a, b = vector[offset:offset+3]
            fraction, A, B = expit(c), np.exp(a), np.exp(b)
            low, high = self.profile.inner_problem(indices, cap, data).strict_bounds
            interval = high[0]-low[0]
            floor = low[0]+interval*fraction
            prediction[indices] = floor+A*cap[indices]+B*data[indices]
            loss, score = fit.loss_and_score(prediction[indices]-p.y[indices], delta)
            weighted = p.weights[indices]*score
            objective += float(p.weights[indices] @ loss)
            gradient[:self.nouter] += (A*cj[indices]+B*dj[indices]).T @ weighted
            gradient[offset:offset+3] = [weighted.sum()*interval*fraction*(1-fraction),
                weighted @ (A*cap[indices]), weighted @ (B*data[indices])]
            coefficients.append({"language": language, "L_inf": float(floor), "A": float(A),
                                 "B": float(B), "floor_fraction": float(fraction)})
        ridge, rg = penalty(vector[:self.nouter], p)
        gradient[:self.nouter] += rg
        if not np.isfinite(prediction).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite joint fit")
        return float(objective+ridge), gradient, prediction, coefficients

    def run(self):
        began = time.monotonic()
        def objective(x):
            value, gradient, _, _ = self.evaluate(x)
            return parent.OBJECTIVE_SCALE*value, parent.OBJECTIVE_SCALE*gradient
        result = optimize(objective, self.initial, self.bounds, self.profile.constrained)
        value, gradient, _, coefficients = self.evaluate(result.x)
        kkt = max(parent.projected_gradient(result.x[:self.nouter], gradient[:self.nouter], self.outer_bounds),
                  float(np.max(np.abs(gradient[self.nouter:]))))
        return {"success": bool(result.success and kkt <= 1e-8), "optimizer_success": bool(result.success),
                "objective": value, "coordinates": result.x[:self.nouter].tolist(),
                "language_coefficients": coefficients, "projected_gradient_inf": kkt,
                "message": str(result.message), "iterations": int(result.nit),
                "runtime_seconds": time.monotonic()-began}
