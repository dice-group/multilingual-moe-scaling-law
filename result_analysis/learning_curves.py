"""Canonical bounded learning-curve estimators for descriptive paper analysis.

Six cubic B-spline coefficients are constrained to be nonincreasing. A penalty
on integrated squared second derivative controls curvature in normalized log
tokens. This shares a smoothing strength across trajectories, not their losses
or multilingual interactions. It supplies no training-seed confidence interval.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator, interp1d
from scipy.optimize import brentq, lsq_linear


PENALTIES = tuple(np.logspace(-7, -1, 9))


@lru_cache(maxsize=1)
def _basis():
    knots = np.r_[np.zeros(4), 1/3, 2/3, np.ones(4)]
    spline = BSpline(knots, np.eye(6), 3, extrapolate=False)
    # c_j = intercept - sum_{i=1}^j decrement_i, with decrement_i >= 0.
    transform = -np.tril(np.ones((6, 6)))
    transform[:, 0] = 1
    # Gauss quadrature integrates the piecewise quadratic squared curvature.
    nodes, weights = np.polynomial.legendre.leggauss(4)
    locations = np.concatenate([(nodes+1)/6 + i/3 for i in range(3)])
    quadrature = np.tile(weights/6, 3)
    roughness = np.sqrt(quadrature)[:, None] * (spline(locations, nu=2) @ transform)
    return spline, transform, roughness


@dataclass(frozen=True)
class LearningCurve:
    x: np.ndarray
    y: np.ndarray
    penalty: float
    coefficients: np.ndarray

    @classmethod
    def fit(cls, x, y, penalty, monotone=True):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if (x.ndim != 1 or y.shape != x.shape or len(x) < 3
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(x <= 0) or np.any(np.diff(x) <= 0)
                or not np.isfinite(penalty) or penalty <= 0):
            raise ValueError("invalid learning-curve data or penalty")
        basis, transform, roughness = _basis()
        z = np.log(x/x[0]) / np.log(x[-1]/x[0])
        design = basis(z) @ transform
        centered = y - y.mean()
        matrix = np.vstack([design/np.sqrt(len(x)), np.sqrt(penalty)*roughness])
        target = np.r_[centered/np.sqrt(len(x)), np.zeros(len(roughness))]
        if monotone:
            solution = lsq_linear(matrix, target, bounds=(np.r_[-np.inf, np.zeros(5)], np.inf),
                                  method="bvls", tol=1e-12, max_iter=200)
            if not solution.success:
                raise RuntimeError(f"curve smoothing failed: {solution.message}")
            parameters = solution.x
        else:
            parameters = np.linalg.lstsq(matrix, target, rcond=None)[0]
        coefficients = transform @ parameters + y.mean()
        if monotone and np.max(np.diff(coefficients)) > 1e-10:
            raise ValueError("monotonicity constraint failed")
        return cls(x.copy(), y.copy(), float(penalty), coefficients)

    def __call__(self, x, method="smooth"):
        values = np.asarray(x, float)
        if (not np.isfinite(values).all()
                or np.any(values < self.x[0]*(1-1e-12))
                or np.any(values > self.x[-1]*(1+1e-12))):
            raise ValueError("evaluation outside observed learning-curve support")
        values = np.clip(values, self.x[0], self.x[-1])
        if method == "linear_log":
            return np.interp(np.log(values), np.log(self.x), self.y)
        if method == "pchip_log":
            return PchipInterpolator(np.log(self.x), self.y, extrapolate=False)(np.log(values))
        if method != "smooth":
            raise ValueError(f"unknown curve method: {method}")
        basis, _, _ = _basis()
        z = np.clip(np.log(values/self.x[0])/np.log(self.x[-1]/self.x[0]), 0, 1)
        return basis(z) @ self.coefficients


def invert_monotone(function, target, lower, upper):
    """Invert a decreasing function, retaining left/right censoring explicitly."""
    start, end = float(function(lower)), float(function(upper))
    if target >= start:
        return float(lower), "left_censored"
    if target < end:
        return None, "unreached"
    root = float(np.exp(brentq(lambda log_x: float(function(np.exp(log_x)))-target,
                               np.log(lower), np.log(upper), xtol=1e-12)))
    if not np.isclose(function(root), target, rtol=0, atol=2e-10):
        raise ValueError("inverse failed forward verification")
    return root, "bracketed"


def first_feasible_piecewise(polynomial):
    """First point where every component polynomial is <= 0, without monotonicity.

    All real polynomial roots and both support endpoints are checked. This
    retains early feasible intervals even if losses subsequently rise again.
    The caller supplies one component for a mean objective or one per language.
    """
    if polynomial.c.ndim != 3:
        raise ValueError("expected an explicit component dimension")
    lower, upper = polynomial.x[[0, -1]]
    roots = polynomial.solve(0, extrapolate=False)
    roots = np.concatenate([np.asarray(values, float) for values in roots.flat])
    locations = np.unique(np.r_[lower, roots[np.isfinite(roots) & (roots >= lower) & (roots <= upper)], upper])
    feasible = np.max(polynomial(locations), axis=-1) <= 1e-10
    if not feasible.any():
        return None, "unreached"
    point = float(locations[np.flatnonzero(feasible)[0]])
    return point, "left_censored" if point == lower else "bracketed"


def horizon_cv(curves, penalties=PENALTIES):
    """Equal-trajectory CV over interior readouts; endpoints remain in support.

    This tunes interpolation/smoothing within observed training trajectories.
    It does not assess independent-trajectory or training-seed generalization.
    """
    scores = []
    for penalty in penalties:
        errors = []
        for name, (x, y) in curves.items():
            for omitted in range(1, len(x)-1):
                keep = np.arange(len(x)) != omitted
                curve = LearningCurve.fit(x[keep], y[keep], penalty)
                errors.append((name, (float(curve(x[omitted]))-y[omitted])**2))
        per_trajectory = {}
        for name, error in errors:
            per_trajectory.setdefault(name, []).append(error)
        scores.append(dict(penalty=float(penalty), mse=float(np.mean([
            np.mean(values) for values in per_trajectory.values()])),
            trajectories=len(per_trajectory), omitted_readouts=len(errors)))
    # An exact tie favors more regularization; no outcome-based visual tuning.
    selected = min(scores, key=lambda row: (row["mse"], -row["penalty"]))["penalty"]
    return selected, scores


def allocation_metrics(observed, predicted, tolerance=0.01):
    """Score recipe-by-budget matrices with the same feasible recipe ordering."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if (observed.ndim != 2 or observed.shape != predicted.shape
            or observed.shape[0] < 2 or not np.isfinite(observed).all()
            or not np.isfinite(predicted).all() or tolerance < 0):
        raise ValueError("invalid allocation comparison")
    chosen = predicted.argmin(axis=0)
    regret = observed[chosen, np.arange(observed.shape[1])] - observed.min(axis=0)
    a, b = np.triu_indices(observed.shape[0], k=1)
    truth_margin, model_margin = observed[a]-observed[b], predicted[a]-predicted[b]
    resolved = np.abs(truth_margin) > tolerance
    return dict(chosen_indices=chosen, regret=regret,
                winner_matches=chosen == observed.argmin(axis=0),
                within_tolerance=regret <= tolerance,
                pairwise_margin_rmse=float(np.sqrt(np.mean((truth_margin-model_margin)**2))),
                resolved_pairwise_accuracy=float(np.mean(
                    np.sign(truth_margin[resolved]) == np.sign(model_margin[resolved])))
                    if resolved.any() else None)


class RecipeCurve:
    """Vectorized language curves sharing the same observed training horizons."""

    def __init__(self, frame, penalties):
        self.languages = sorted(frame.language.unique())
        self.positions = {language: i for i, language in enumerate(self.languages)}
        pivot = frame.pivot(index="D_total", columns="language", values="adjusted_loss")
        self.tokens = pivot.index.to_numpy(float)
        self.loss = pivot[self.languages].to_numpy(float)
        self.exposure = frame.pivot(index="D_total", columns="language", values="D_t")[self.languages].to_numpy(float)
        if not np.isfinite(self.loss).all() or not np.isfinite(self.exposure).all():
            raise ValueError("incomplete recipe language/horizon matrix")
        self.n = int(frame.N_act.iloc[0])
        self.stored = int(frame.N_total.iloc[0])
        self.e = int(frame.E.iloc[0])
        self.k = int(frame.K.iloc[0])
        self.name = f"{frame.ladder.iloc[0]}/E{self.e}"
        self.trajectory = str(frame.trajectory_id.iloc[0])
        self.fmin, self.fmax = 6*self.n*self.tokens[[0, -1]]/1e18
        self.fits = {}
        for method, penalty in penalties.items():
            coefficients = np.stack([LearningCurve.fit(self.tokens, y, penalty).coefficients
                                     for y in self.loss.T], axis=1)
            self.fits[method] = LearningCurve(self.tokens, self.loss.mean(axis=1), penalty, coefficients)
        free = np.stack([LearningCurve.fit(self.tokens, y, penalties["smooth"], monotone=False).coefficients
                         for y in self.loss.T], axis=1)
        self.fits["smooth_unconstrained"] = LearningCurve(self.tokens, self.loss.mean(axis=1), penalties["smooth"], free)
        self.linear = interp1d(np.log(self.tokens), self.loss, axis=0, bounds_error=True)
        self.pchip = PchipInterpolator(np.log(self.tokens), self.loss, axis=0, extrapolate=False)

    def indices(self, cohort):
        return [self.positions[language] for language in cohort]

    def at_tokens(self, tokens, method="smooth"):
        tokens = np.asarray(tokens, float)
        if (np.any(tokens < self.tokens[0]*(1-1e-12))
                or np.any(tokens > self.tokens[-1]*(1+1e-12))):
            raise ValueError("recipe query outside observed support")
        tokens = np.clip(tokens, self.tokens[0], self.tokens[-1])
        if method in self.fits:
            return self.fits[method](tokens)
        if method == "linear_log":
            return self.linear(np.log(tokens))
        if method == "pchip_log":
            return self.pchip(np.log(tokens))
        raise ValueError(f"unknown estimator {method}")

    def values(self, compute, cohort, method="smooth"):
        return self.at_tokens(np.asarray(compute)*1e18/(6*self.n), method)[..., self.indices(cohort)]

    def macro(self, compute, cohort, method="smooth"):
        return self.values(compute, cohort, method).mean(axis=-1)

    def tokens_at_exposure(self, language, exposure):
        values = self.exposure[:, self.positions[language]]
        if (np.any(np.asarray(exposure) < values[0]*(1-1e-12))
                or np.any(np.asarray(exposure) > values[-1]*(1+1e-12))):
            raise ValueError("exposure matching would extrapolate")
        return np.exp(np.interp(np.log(exposure), np.log(values), np.log(self.tokens)))
