"""Fixed size-duration restrictions and their unchanged numerical audits."""
from dataclasses import dataclass
import numpy as np
from . import fit
from .m6_size_duration_workflow import model as m
from .baseline_refit_workflow import contracts as p


FIXED = {"J0-ND": tuple(range(6, 14)), "M5-ND": (11, 12, 13)}


PROTOCOL = {
    "models": list(FIXED), "folds": list(p.FOLDS), "starts_per_cell": 48,
    "zeta_starts": list(m.ZETA_STARTS), "zeta_bounds": [0., .5],
    "anchors": "eight successful parent starts ordered by training objective then start_id",
    "audits": "three projected restarts and one independent joint optimization from best",
    "objective_tolerance": 1e-6, "prediction_tolerance": 1e-4, "gradient_tolerance": 1e-8,
    "rank_relative_tolerance": 1e-8, "free_shared_coordinates": {"J0-ND": 7, "M5-ND": 12},
    "scientific_status": "post-diagnosis development ablations; no prospective test or new law search",
    "selection": "training objective only; report all completed cells and all failures",
    "retries": 0, "holdout_selection": False,
}


@dataclass(frozen=True)
class AblationBound:
    name: str
    lower: float
    upper: float
    kind: str
    expansions: int = 0

    def __post_init__(self):
        if not np.isfinite([self.lower, self.upper]).all() or self.lower > self.upper:
            raise ValueError("invalid ablation interval")


def restrict(coordinates, name):
    x = np.asarray(coordinates, dtype=float).copy()
    if x.shape != (15,) or name not in FIXED:
        raise ValueError("unknown ablation or coordinate shape")
    x[list(FIXED[name])] = 0.
    return x


def restricted_bounds(parent_bounds, name, j0_bounds=None):
    bounds = list(parent_bounds)
    if name == "J0-ND" and j0_bounds is not None:
        bounds[:6] = j0_bounds
    for i in FIXED[name]:
        bounds[i] = AblationBound(bounds[i].name, 0., 0., "nonnegative")
    return tuple(bounds)


def health(profile, selected, bounds, name):
    if np.any(np.asarray(selected["coordinates"])[list(FIXED[name])] != 0.):
        raise ValueError("an ablated coordinate is not exactly zero")
    value = m.numerical_health(profile, selected, bounds)
    fixed_names = {bounds[i].name for i in FIXED[name]}
    rejected = [c for c in value["rejected_boundary_contacts"] if c["name"] not in fixed_names]
    rank = PROTOCOL["free_shared_coordinates"][name]
    return value | {"required_rank": rank, "rejected_boundary_contacts": rejected,
        "rank_scope": "free shared-coordinate prediction Jacobian; nuisance held fixed; not global identification",
        "healthy": bool(value["shared_prediction_jacobian_rank"] == rank and not rejected
            and value["minimum_alpha"] > 0 and value["minimum_beta"] > 0
            and value["projected_gradient_inf"] <= 1e-8)}


def certify_attempts(profile, bounds, name, attempts):
    """Audit the three best successful starts and an independent direct fit."""
    successful = sorted([a for a in attempts if a["success"]], key=lambda a: (a["objective"], a["start_id"]))
    selected = successful[0] if successful else None
    audits = []
    if len(successful) >= 3:
        for kind, start in [*(("basin", a) for a in successful[:3]), ("direct", selected)]:
            try:
                result = m.Direct(profile, start, bounds).run() if kind == "direct" else profile.run(start["coordinates"], bounds)
                equivalent = m.equivalence(selected, result, profile.problem)
                numerical = health(profile, result, bounds, name)
                status = "healthy" if equivalent["equivalent"] and numerical["healthy"] else "failed"
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                result, equivalent, numerical, status = {"success": False, "message": str(error)}, {}, {}, "failed"
            audits.append({"kind": kind, "start_id": start["start_id"], "result": result,
                           "equivalence": equivalent, "health": numerical, "status": status})
    return dict(selected=selected, audits=audits,
                status="healthy" if len(audits) == 4 and all(a["status"] == "healthy" for a in audits) else "unresolved")
