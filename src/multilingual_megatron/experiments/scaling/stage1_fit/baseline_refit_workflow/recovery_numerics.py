"""Conditioned, analytic-gradient J0 profiling for a numerical refit.

The loss, observations, bounds, and language weights are unchanged. Rescaling
each convex inner objective by its weight sum and its design columns changes
only numerical conditioning. This solver is separate from the historical
fitter so that previously recorded numerical provenance remains reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize

from .. import fit

INNER_SETTINGS = {"ftol": 1e-14, "xtol": 1e-14, "gtol": 1e-13, "max_nfev": 2000}
OUTER_SETTINGS = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 20000, "maxls": 50, "maxcor": 20}
OBJECTIVE_SCALE = 10000.0


def weighted_huber_loss(weights):
    """SciPy's rho, rho-prime, rho-double-prime with exact row weights."""
    weights = np.asarray(weights, dtype=np.float64)

    def loss(z):
        rho = np.empty((3, z.size), dtype=np.float64)
        inside = z <= 1
        rho[0, inside] = z[inside]
        rho[1, inside] = 1
        rho[2, inside] = 0
        root = np.sqrt(z[~inside])
        rho[0, ~inside] = 2 * root - 1
        rho[1, ~inside] = 1 / root
        rho[2, ~inside] = -.5 / root ** 3
        return rho * weights[None, :]

    return loss


def solve_language(problem: fit.LanguageProblem, delta, *, language=None):
    weights = problem.weights / problem.weights.sum()
    scales = np.sqrt(np.sum(weights[:, None] * problem.design ** 2, axis=0))
    design = problem.design / scales[None, :]
    lower, upper = (v * scales for v in problem.strict_bounds)
    root = np.sqrt(weights)
    initial = lsq_linear(root[:, None] * design, root * problem.y,
        bounds=(lower, upper), method="bvls", tol=1e-14, max_iter=2000)
    if not initial.success:
        raise fit.InnerSolveError("conditioned bounded initializer failed")
    # Return a fresh Jacobian: SciPy robust least squares scales it in place.
    if delta is None:
        result = least_squares(lambda z: root * (design @ z - problem.y), initial.x,
            jac=lambda z: root[:, None] * design, bounds=(lower, upper),
            **INNER_SETTINGS)
    else:
        result = least_squares(lambda z: design @ z - problem.y, initial.x,
            jac=lambda z: design.copy(), bounds=(lower, upper),
            loss=weighted_huber_loss(weights), f_scale=delta, **INNER_SETTINGS)
    values = result.x / scales
    solution = fit._solution_from_result(problem=problem, values=values, delta=delta,
        success=result.success, status=result.status, message=str(result.message),
        iterations=result.nfev, function_evaluations=result.nfev, language=language)
    _, gradient = fit.language_objective_and_gradient(values, problem, delta)
    gradient = gradient / (scales * problem.weights.sum())
    # A box-constrained KKT residual; active coefficients need not have zero
    # ordinary gradient. The projected residual is in conditioned coordinates.
    residual = result.x - np.clip(result.x - gradient, lower, upper)
    if not result.success or np.max(np.abs(residual)) > 1e-9:
        raise fit.InnerSolveError(f"conditioned inner KKT failure for {language}: {np.max(np.abs(residual))}")
    return solution


class J0Profile:
    def __init__(self, problem, delta):
        self.problem = problem
        self.delta = delta
        identifiers = np.asarray(problem.language_ids)
        self.groups = [(name, np.flatnonzero(identifiers == name)) for name in sorted(set(identifiers))]

    def evaluate(self, coordinates):
        p = self.problem
        cap, data, cap_jac, data_jac = fit.predictor_terms_and_jacobian(
            "J0", coordinates, p.N, p.D, p.E, p.K)
        predictions = np.empty_like(p.y)
        jacobian = np.empty((p.y.size, len(coordinates)))
        solutions = []
        for name, indices in self.groups:
            inner = fit.LanguageProblem(p.y[indices], cap[indices], data[indices],
                p.weights[indices], float(p.physical_floors[indices[0]]), float(p.y[indices].min()))
            solution = solve_language(inner, self.delta, language=name)
            solutions.append(solution)
            predictions[indices] = solution.prediction
            jacobian[indices] = solution.A * cap_jac[indices] + solution.B * data_jac[indices]
        residual = predictions - p.y
        objective, gradient = fit.weighted_objective_and_gradient(residual, jacobian, p.weights, self.delta)
        profile = fit.ProfileSolution(objective, predictions, residual, tuple(solutions), True)
        evaluated = fit.ProjectedEvaluation(np.asarray(coordinates).copy(),
            fit.shared_from_outer("J0", coordinates), objective, predictions, profile)
        return evaluated, gradient

    def run(self, start, bounds):
        def objective(coordinates):
            evaluation, gradient = self.evaluate(coordinates)
            return evaluation.objective * OBJECTIVE_SCALE, gradient * OBJECTIVE_SCALE
        result = minimize(objective, start.values, method="L-BFGS-B", jac=True,
            bounds=[(v.lower, v.upper) for v in bounds], options=OUTER_SETTINGS)
        final, gradient = self.evaluate(result.x)
        return fit.ProjectedStartResult("J0", start.start_id, start.start_index,
            bool(result.success), int(result.status), str(result.message), final.objective,
            result.x, final.shared_parameters, final.prediction, final.profile,
            int(result.nit), int(result.nfev), fit.boundary_contacts(result.x, bounds)), gradient
