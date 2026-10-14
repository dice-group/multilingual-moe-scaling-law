"""Equation kernels and restrictions; no files, fitting or scoring responses."""
from dataclasses import replace

import numpy as np

from .. import fit
from ..law_search import model as legacy
from .. import repetition_model
from .contracts import ABLATIONS, BASELINES, FORMS, RULES


class Model(legacy.Model):
    """An explicit 18-slot embedding; fixed slots are not fitted parameters."""
    def __init__(self, name, problem, inventory, *, no_ek=False):
        super().__init__("RN", problem, inventory)
        self.name, self.no_ek = name, no_ek
        self.powered = name in BASELINES or FORMS.get(name, {}).get("shape") == "power"
        self.bounds.append(repetition_model.Bound("q", 1., 1., "positive"))
        fixed = {}
        if name in ABLATIONS:
            fixed.update({i: 0. for i in ABLATIONS[name]})
        elif name in FORMS:
            spec = FORMS[name]
            fixed.update({i: 0. for i in range(10, 14)})
            if spec["shape"] != "mult":
                fixed[14] = 0.
            if spec["reuse"] == "none":
                fixed.update({15: 0., 16: 0.})
            elif spec["reuse"] == "constant":
                fixed[16] = 0.
        elif name in BASELINES:
            fixed.update({i: 0. for i in range(6, 17)})
            if name != BASELINES[2]:
                fixed.update({4: 0., 5: 0.})
                self.equalities = ((0, 1), (2, 3))
            self.problem = replace(problem, regularization=replace(problem.regularization, strength=0.))
        else:
            raise ValueError(f"unregistered form {name}")
        for index, value in fixed.items():
            self.bounds[index] = replace(self.bounds[index], lower=value, upper=value)
        if self.powered:
            self.bounds[17] = repetition_model.Bound("q", *RULES["q_bounds"], "positive")
        if no_ek:
            if name not in FORMS:
                raise ValueError("E-by-K ablation applies only to a registered simple candidate")
            self.equalities = (*self.equalities, (6, 7), (8, 9))

    def terms(self, x):
        cap, data, cj, dj = repetition_model.terms(np.asarray(x)[:17], self.problem, self.inventory,
            require_positive_exponents=getattr(self, "require_positive_exponents", True))
        return cap, data, np.column_stack((cj, np.zeros(len(cap)))), np.column_stack((dj, np.zeros(len(cap))))

    def penalty(self, x):
        value, gradient = repetition_model.penalty(np.asarray(x)[:17], self.problem)
        return value, np.r_[gradient, 0.]


def combine(log_c, log_t, q):
    """Stable powered sum and derivatives w.r.t. log(C), log(T), q."""
    if not np.isfinite(q) or q <= 0:
        raise ValueError("positive finite q required")
    left, right = np.asarray(log_c) / q, np.asarray(log_t) / q
    joined = np.logaddexp(left, right)
    a, b = np.exp(left - joined), np.exp(right - joined)
    value = np.exp(q * joined)
    entropy = -(a * (left - joined) + b * (right - joined))
    if not np.isfinite(value).all():
        raise FloatingPointError("powered sum overflow")
    return value, value * a, value * b, value * entropy


def prediction(model, record, *, jacobian=False):
    x = np.asarray(record["coordinates"], float)
    cap, term, cj, dj = model.terms(x)
    coefficients = {c["language"]: c for c in record["language_coefficients"]}
    c = np.array([[coefficients[l][k] for k in ("L_inf", "A", "B")] for l in model.problem.language_ids])
    if np.any(c[:, 1:] <= 0):
        raise ValueError("positive amplitudes required")
    if not model.powered:
        pred = c[:, 0] + c[:, 1] * cap + c[:, 2] * term
        shape = c[:, 1, None] * cj + c[:, 2, None] * dj
        nuisance = np.column_stack((np.ones(len(pred)), cap, term))
    else:
        value, dc, dt, dq = combine(np.log(c[:, 1]) + np.log(cap), np.log(c[:, 2]) + np.log(term), x[17])
        pred = c[:, 0] + value
        shape = dc[:, None] * (cj / cap[:, None]) + dt[:, None] * (dj / term[:, None])
        shape[:, 17] += dq
        nuisance = np.column_stack((np.ones(len(pred)), dc / c[:, 1], dt / c[:, 2]))
    if not np.isfinite(pred).all():
        raise FloatingPointError("nonfinite predictions")
    return (pred, shape, nuisance) if jacobian else pred


def validate(model, record):
    x = np.asarray(record["coordinates"], float)
    if x.shape != (18,) or not np.isfinite(x).all():
        raise ValueError("expected 18 finite embedded coordinates")
    for value, bound in zip(x, model.bounds):
        if not bound.lower - 1e-12 <= value <= bound.upper + 1e-12:
            raise ValueError(f"coordinate outside frozen bounds: {bound.name}")
    for first, second in model.equalities:
        if abs(x[first] - x[second]) > 1e-10:
            raise ValueError("coordinate equality violated")


class Reduction:
    """Remove exact fixed coordinates and ties before joint optimization."""
    def __init__(self, model):
        groups = [{i} for i in range(len(model.bounds))]
        for a, b in model.equalities:
            left, right = next(g for g in groups if a in g), next(g for g in groups if b in g)
            if left is not right:
                left.update(right)
                groups.remove(right)
        self.fixed = np.array([b.lower for b in model.bounds])
        self.groups, self.bounds = [], []
        for group in groups:
            low = max(model.bounds[i].lower for i in group)
            high = min(model.bounds[i].upper for i in group)
            if low > high:
                raise ValueError("incompatible coordinate restrictions")
            if low == high:
                self.fixed[list(group)] = low
            else:
                self.groups.append(sorted(group))
                self.bounds.append((low, high))
        self.matrix = np.zeros((len(model.bounds), len(self.groups)))
        for i, group in enumerate(self.groups):
            self.matrix[group, i] = 1.

    def pack(self, x):
        return np.array([np.clip(np.mean(np.asarray(x)[g]), *b) for g, b in zip(self.groups, self.bounds)])

    def expand(self, packed):
        result = self.fixed.copy()
        for group, value in zip(self.groups, packed):
            result[group] = value
        return result
