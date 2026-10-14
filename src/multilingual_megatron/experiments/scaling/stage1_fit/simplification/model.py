"""Compact restrictions of the audited M6-ND law, with exact chain rules.

Endpoint coordinates preserve the parent's rectangular feasible domain:
theta = (phi_e32 - phi_e1) / log(32). Historical fit code is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .. import fit
from ..m6_size_duration_workflow import model as parent

MODELS = ("S8", "S7-no-ND", "S7-no-EK")
NAMES = ("alpha", "beta", "delta", "omega", "phi_e1", "phi_e32", "psi", "zeta")


@dataclass(frozen=True)
class Bound:
    name: str
    lower: float
    upper: float
    kind: str
    expansions: int = 0


def mapping(name):
    if name not in MODELS:
        raise ValueError("unknown compact law")
    matrix = np.zeros((15, 8))
    for target, source in enumerate((0, 0, 1, 1, 2, 3, 4, 5, 6, 6)):
        matrix[target, source] = 1.
    matrix[14, 7] = 1.
    names = list(NAMES)
    if name == "S7-no-ND":
        matrix = matrix[:, :7]
        names.pop()
    elif name == "S7-no-EK":
        matrix[:, 4] += matrix[:, 5]
        matrix = np.delete(matrix, 5, axis=1)
        names.pop(5)
    return matrix, names


def bounds(parent_bounds, name):
    matrix, names = mapping(name)
    result = []
    for col, label in enumerate(names):
        sources = [parent_bounds[i] for i in np.flatnonzero(matrix[:, col])]
        low, high = max(b.lower for b in sources), min(b.upper for b in sources)
        if low >= high:
            raise ValueError("empty tied-coordinate interval")
        result.append(Bound(label, low, high, sources[0].kind))
    return tuple(result)


def restrict(x, name):
    matrix, _ = mapping(name)
    return np.linalg.lstsq(matrix, np.asarray(x), rcond=None)[0]


def terms(x, problem, name):
    matrix, _ = mapping(name)
    cap, data, cj, dj = parent.terms(matrix @ np.asarray(x), problem)
    return cap, data, cj @ matrix, dj @ matrix


class Profile(parent.Profile):
    def __init__(self, problem, delta, name):
        super().__init__(problem, delta)
        self.name = name
        self.matrix, self.names = mapping(name)

    def evaluate(self, x):
        result, gradient, prediction, jacobian = super().evaluate(self.matrix @ np.asarray(x))
        result["coordinates"] = np.asarray(x).tolist()
        return result, self.matrix.T @ gradient, prediction, jacobian @ self.matrix


def expanded(selected, name):
    matrix, _ = mapping(name)
    return dict(selected, coordinates=(matrix @ np.asarray(selected["coordinates"])).tolist())


def predict(problem, selected, name):
    return parent.predict(problem, expanded(selected, name))


class Direct:
    """Joint fit in reduced shared coordinates and original nuisance coordinates."""
    def __init__(self, profile, selected, box):
        self.matrix = profile.matrix
        self.nouter = len(box)
        self.box = box
        full_box = [Bound(str(i), -2., 2., "signed") for i in range(15)]
        original_profile = parent.Profile(profile.problem, profile.delta)
        self.original = parent.Direct(original_profile, expanded(selected, profile.name), full_box)
        self.initial = np.r_[selected["coordinates"], self.original.initial[15:]]
        self.bounds = [(b.lower, b.upper) for b in box] + self.original.bounds[15:]

    def evaluate(self, x):
        vector = np.r_[self.matrix @ x[:self.nouter], x[self.nouter:]]
        value, gradient, prediction, coefficients = self.original.evaluate(vector)
        return value, np.r_[self.matrix.T @ gradient[:15], gradient[15:]], prediction, coefficients

    def run(self):
        began = time.monotonic()
        def objective(x):
            value, gradient, _, _ = self.evaluate(x)
            return parent.OBJECTIVE_SCALE*value, parent.OBJECTIVE_SCALE*gradient
        result = minimize(objective, self.initial, method="L-BFGS-B", jac=True,
                          bounds=self.bounds, options=parent.OUTER_SETTINGS)
        value, gradient, _, coefficients = self.evaluate(result.x)
        kkt = max(parent.projected_gradient(result.x[:self.nouter], gradient[:self.nouter], self.box),
                  float(np.max(np.abs(gradient[self.nouter:]))))
        return {"success": bool(result.success and kkt <= 1e-8), "objective": value,
                "coordinates": result.x[:self.nouter].tolist(), "language_coefficients": coefficients,
                "optimizer_success": bool(result.success), "message": str(result.message),
                "projected_gradient_inf": kkt, "runtime_seconds": time.monotonic()-began}


def audit(profile, chosen, attempts, box):
    direct = Direct(profile, chosen, box).run()
    reference = predict(profile.problem, chosen, profile.name)
    def agrees(other):
        relative = abs(other["objective"]-chosen["objective"])/max(abs(chosen["objective"]), 1e-30)
        maximum = float(np.max(abs(predict(profile.problem, other, profile.name)-reference)))
        return relative, maximum
    relative, maximum = agrees(direct)
    basin = [a for a in attempts if a.get("success") and agrees(a)[0] <= 1e-6 and agrees(a)[1] <= 1e-4]
    distinct = len({tuple(a["initial"]) for a in basin})
    _, gradient, _, jac = profile.evaluate(chosen["coordinates"])
    singular = np.linalg.svd(jac*np.array([b.upper-b.lower for b in box]), compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-8))
    contacts = [vars(c) for c in fit.boundary_contacts(chosen["coordinates"], box)]
    rejected = [c for c in contacts if not (c["name"] == "zeta" and c["side"] == "lower")]
    kkt = parent.projected_gradient(np.asarray(chosen["coordinates"]), gradient, box)
    return {"success": bool(direct["success"] and relative <= 1e-6 and maximum <= 1e-4
                            and distinct >= 2 and rank == len(box) and not rejected and kkt <= 1e-8),
            "direct": direct, "relative_objective_difference": relative,
            "maximum_prediction_difference": maximum, "basin_members": len(basin),
            "distinct_basin_initializations": distinct,
            "shared_prediction_jacobian_rank": rank, "required_rank": len(box),
            "rank_scope": "nuisance held fixed; not global parameter identifiability",
            "singular_values": singular.tolist(), "boundary_contacts": contacts,
            "rejected_boundary_contacts": rejected, "projected_gradient_inf": kkt}
