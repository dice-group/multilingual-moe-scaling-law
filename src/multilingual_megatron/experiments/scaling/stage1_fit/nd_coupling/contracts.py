"""Frozen equations, task graph and selection decisions, independent of data."""
import math

from ..law_search.protocol import INNER
from ..law_search.protocol import PRIMARY

OUTER = (*PRIMARY, "all_grid")
FORMS = {
    f"EK-{shape}-{reuse}": dict(shape=shape, reuse=reuse,
                              shared=10 + (shape != "add") + extra)
    for shape in ("add", "mult", "power")
    for reuse, extra in (("none", 0), ("constant", 1), ("size", 2))
}
ABLATIONS = {"RN-no-zeta": (14,), "RN-no-rho": (16,), "RN-no-zeta-rho": (14, 16)}
BASELINES = ("Skaling-local-active-Dt", "Skaling-pooled-active-Dt",
             "JointMoE-Skaling-pooled-active-Dt")
CONTROLS = ("RN", "RN-no-composition")
INTERNAL_OLD = ("M0", "M1", "M2", "M6", "M6-ND", "J0-ND", "M5-ND")
STAGES = ("pilot", "pilot-gate", "inner", "select", "final", "native", "stability", "report")
RULES = dict(
    schema="stage1_nd_coupling/v1", forms=FORMS, rn_ablations=ABLATIONS,
    new_baselines=BASELINES, controls=CONTROLS, inner=INNER, outer=OUTER,
    initial_starts=24, maximum_starts=48, seed=2026091903,
    zeta_bounds=(0., .5), q_bounds=(.01, 2.), rate_bounds=(0., 10.), rho_bounds=(-2., 2.),
    ridge_lambda=.01, rho_penalty_tau=0., simplicity_band=1.01, per_view_guard=1.02,
    gradient_tolerance=1e-8, objective_tolerance=1e-6, prediction_tolerance=1e-4,
    replay_tolerance=1e-10, win_tolerance_nats=1e-6,
    fit_minutes=240, native_minutes=720, analysis_minutes=120,
    max_concurrent_cores=8, max_core_hours=512., bootstrap_refits=20,
    mc_draws=999999, bootstrap_draws=100000, mc_seed=2026091904, bootstrap_seed=2026091905,
    retries=0, gpu_training=False,
    evidence="post-observation development; fixed-recipe nested readouts and one L26 run are not independent confirmation",
    selection="minimum worst inner RMSE ratio to RN; 1% simplicity band, 2% per-view guard; shared count, mean, registered order",
    primary_metric="configuration-balanced RMSE; arithmetic mean of three primary RMSEs",
    endpoint_metric="equal-language RMSE; separate 200/200/50/200 cohorts",
    inference="paired whole-configuration swaps; overlap-preserving mean; Holm across frozen selected-law contrasts",
    pilot="first size-C additive/no-repetition and powered/no-repetition fits; require both certified within caps",
)


def tasks():
    result = []
    pilot = {(INNER[0], "EK-add-none"), (INNER[0], "EK-power-none")}
    for fold in INNER:
        for form in FORMS:
            result.append(dict(id=f"inner-{fold}-{form}", stage="pilot" if (fold, form) in pilot else "inner",
                               kind="candidate", fold=fold, form=form, minutes=RULES["fit_minutes"]))
    for fold in OUTER:
        for form, kind in [("selected", "candidate"), ("selected-no-EK", "no_EK"),
                           *((name, "ablation") for name in ABLATIONS),
                           *((name, "baseline") for name in BASELINES)]:
            native = form == BASELINES[0]
            result.append(dict(id=f"final-{fold}-{form}", stage="native" if native else "final",
                               kind=kind, fold=fold, form=form,
                               minutes=RULES["native_minutes"] if native else RULES["fit_minutes"]))
    for index in range(RULES["bootstrap_refits"]):
        result.append(dict(id=f"bootstrap-{index:02d}", stage="stability", kind="bootstrap",
                           fold="all_grid", form="selected", replicate=index, minutes=RULES["fit_minutes"]))
    check_budget(result)
    return result


def check_budget(entries):
    if any(t["minutes"] <= 0 or t.get("cores", 1) != 1 for t in entries):
        raise ValueError("only positive-duration single-core tasks are permitted")
    # Three analysis stages, each single-core. Submission stages never overlap.
    hours = (sum(t["minutes"] for t in entries) + 3 * RULES["analysis_minutes"]) / 60
    if hours > RULES["max_core_hours"]:
        raise ValueError("CPU reservation exceeds frozen budget")
    return hours


def choose(records):
    expected = {(name, fold) for name in (*FORMS, "RN") for fold in INNER}
    keys = [(r["form"], r["fold"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("selection requires exactly the complete registered inner panel")
    if any(not r["certified"] or not r["complete"] or not math.isfinite(r["rmse"])
           or r["rmse"] <= 0 for r in records):
        raise ValueError("uncertified or incomplete inner panel; no selection")
    scores = {(r["form"], r["fold"]): r["rmse"] for r in records}
    candidates = []
    for order, (name, spec) in enumerate(FORMS.items()):
        ratios = [scores[name, fold] / scores["RN", fold] for fold in INNER]
        candidates.append(dict(form=name, shared=spec["shared"], ratios=ratios,
                               worst=max(ratios), mean=sum(ratios) / len(ratios), order=order))
    best = min(candidates, key=lambda r: (r["worst"], r["mean"], r["order"]))
    eligible = [r for r in candidates if r["worst"] <= RULES["simplicity_band"] * best["worst"]
                and all(a <= RULES["per_view_guard"] * b for a, b in zip(r["ratios"], best["ratios"]))]
    return dict(chosen=min(eligible, key=lambda r: (r["shared"], r["mean"], r["order"])),
                best_worst=best, eligible=eligible, procedures=candidates)


def category(name):
    if name in INTERNAL_OLD or name in CONTROLS or name == BASELINES[2]:
        return "internal_control"
    if name in ABLATIONS or name.endswith("-no-EK"):
        return "diagnostic_ablation"
    if name in FORMS:
        return "candidate"
    if name in BASELINES or name == "C0" or name == "J0" or name.startswith(
            ("ATLAS-", "Chinchilla-", "JointMoE-", "He-", "Pooled-ATLAS-", "Pooled-JointMoE-")):
        return "published_form_adaptation"
    raise ValueError(f"unregistered comparator category: {name}")
