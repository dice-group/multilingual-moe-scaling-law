"""Accuracy-preserving inner fallback for nonstationary M6-ND trial points.

The historical conditioned solver occasionally terminates on relative cost
while its KKT residual is still above 1e-9. Keep that threshold, and refine the
same convex Huber objective by bounded iteratively reweighted least squares.
Neither historical code nor data, objective weights, or constraints change.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .. import fit
from ..baseline_refit_workflow.recovery_numerics import solve_language as original_solve


def refine_language(problem, delta, *, language=None):
    weights = problem.weights/problem.weights.sum()
    scales = np.sqrt(np.sum(weights[:, None]*problem.design**2, axis=0))
    design = problem.design/scales[None, :]
    lower, upper = (v*scales for v in problem.strict_bounds)
    robust_weights = np.ones(len(problem.y))
    previous_objective = np.inf
    for iteration in range(5000):
        root = np.sqrt(weights*robust_weights)
        result = lsq_linear(root[:, None]*design, root*problem.y, bounds=(lower, upper),
            method="bvls", tol=1e-14, max_iter=2000)
        if not result.success:
            raise fit.InnerSolveError("IRLS bounded quadratic subproblem failed")
        values = result.x/scales
        objective, gradient = fit.language_objective_and_gradient(values, problem, delta)
        gradient /= scales*problem.weights.sum()
        kkt = float(np.max(np.abs(result.x-np.clip(result.x-gradient, lower, upper))))
        if kkt <= 1e-10:
            return fit._solution_from_result(problem=problem, values=values, delta=delta,
                success=True, status=0, message=f"IRLS refinement: normalized KKT={kkt:.3g}",
                iterations=iteration+1, function_evaluations=iteration+1, language=language)
        if objective > previous_objective+1e-13*max(1., abs(previous_objective)):
            raise fit.InnerSolveError("IRLS failed Huber descent check")
        previous_objective = objective
        residual = problem.design@values-problem.y
        robust_weights = (np.ones(len(residual)) if delta is None else
            np.minimum(1., delta/np.maximum(np.abs(residual), np.finfo(float).tiny)))
    raise fit.InnerSolveError(f"IRLS failed unchanged 1e-9 KKT contract for {language}: {kkt}")


def solve_language(problem, delta, *, language=None):
    try:
        return original_solve(problem, delta, language=language)
    except fit.InnerSolveError:
        return refine_language(problem, delta, language=language)
