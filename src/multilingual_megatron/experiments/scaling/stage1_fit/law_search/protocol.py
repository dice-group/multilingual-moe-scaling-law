"""Versioned scientific decisions, independent of results and scheduler state."""
from .model import CANDIDATES
from .model import PARAMETER_COUNTS
from .model import POOLED
from .model import TAUS

SCHEMA = "stage1_law_search/v1"
PRIMARY = ("largest_N_act", "E32", "K200_common_languages")
INNER = ("inner_size_C", "inner_E16", "inner_K50", "inner_terminal")
SECONDARY = ("random_13_of_64", "lowest_terminal_macro_loss_30", "terminal", "high_compute_20")
SHARED = ("C0", "M0", "J0", "M1", "M2", "M6", "M6-ND", "J0-ND", "M5-ND")
PROTOCOL = {
    "schema": SCHEMA, "candidates": list(CANDIDATES), "shared_parameter_counts": PARAMETER_COUNTS,
    "rho_penalty_tau": list(TAUS), "rho_bounds": [-2., 2.], "rate_bounds": [0., 10.],
    "rho_penalty": "0.5 * Qref_M2 * tau * rho**2; existing M6 ridge also retained",
    "source_equation": "S_t + eta_family*sum_j_in_family(S_j) + eta_script*sum_j_in_disjoint_script(S_j)",
    "source_inventory": "exact (K, language) original-document token inventory, before replicas",
    "capacity": "unchanged M6 capacity and other-language dilution term",
    "structural_nulls": "fix input-unidentified coordinates to zero; never infer nulls from test loss",
    "outer_folds": list(PRIMARY), "inner_folds": list(INNER), "secondary_folds": list(SECONDARY),
    "selection_population": "intersection of all primary training partitions: A/B/C x E1/4/16 x K1/12/50",
    "inner_K50_scoring": "languages shared with K<=12 training",
    "calibration": "training-only OLS M0 -> residual-MAD delta -> Huber M2 Qref -> M6-ND parent",
    "selection": "minimize worst inner RMSE / unregularized RN RMSE; no outer or L26 responses",
    "simplicity_band": 1.01, "individual_view_guard": 1.02,
    "ties": "eligible nominal shared count, then mean normalized RMSE, then registered form/tau order",
    "primary_metric": "configuration-balanced RMSE across every scored language and readout",
    "endpoint_metric": "equal-language RMSE; 200/200/50 fold forecasts plus matched all-grid 200",
    "required_cases": 7, "win_tolerance_nats": 1e-6,
    "coverage": "exact row identity and finite predictions; unresolved and domain-invalid are never defeated",
    "baseline_panel": "42 certified native published-law adaptations + original 9 shared + 4 matched pooled",
    "pooled": [s.payload() for s in POOLED], "shared": list(SHARED),
    "unresolved_native": 11, "internal_controls": ["RN", "R"],
    "initial_starts": 24, "escalated_starts": 48, "native_starts": [16, 64],
    "escalation": "numeric audit failure or training-basin improvement only; never score-triggered",
    "gradient_tolerance": 1e-8, "objective_tolerance": 1e-6, "prediction_tolerance": 1e-4,
    "independent_equation_tolerance": 1e-10, "seed": 20260913,
    "bootstrap_refits": 20, "rho_profile_values": [-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.],
    "bootstrap_unit": "whole training trajectory, stratified by E and K",
    "uncertainty": "paired primary configuration bootstrap; endpoint cohort sensitivity is not seed uncertainty",
    "max_core_hours": 512., "max_concurrent_cores": 32,
    "phase_core_hours": {"pilot": 12., "selection": 128., "baselines": 200., "final": 96., "stability": 76.},
    "gpu_training_authorized": False, "fresh_confirmation": "costed proposal only; two new L26 seeds at 32B and 68.68B",
    "evidence": "post-observation development; neither nested readouts nor one larger run prove generalization",
    "outcomes": ["achieved", "fragile", "partial", "incomplete"],
    "freeze": "one globally selected form and tau, immutable before outer and L26 scoring; no reselection",
}


def choose_procedure(records):
    """Require all 60 certified inner fits before a deterministic global choice."""
    expected = {(name, tau, fold) for name in CANDIDATES for tau in TAUS for fold in INNER}
    keys = [(r["model"], r["tau"], r["fold"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("inner selector needs exactly the complete registered panel")
    if any(not r["certified"] or not r["complete"] or not 0 < r["rmse"] < float("inf") for r in records):
        raise ValueError("incomplete or unaudited inner comparison; no selection")
    scores = {key: row["rmse"] for key, row in zip(keys, records)}
    procedures = []
    for index, (name, tau) in enumerate((n,t) for n in CANDIDATES for t in TAUS):
        ratios = [scores[name,tau,f]/scores["RN",0.,f] for f in INNER]
        procedures.append(dict(model=name, tau=tau, ratios=ratios, worst=max(ratios),
                               mean=sum(ratios)/len(ratios), shared=PARAMETER_COUNTS[name], order=index))
    best = min(procedures, key=lambda r: (r["worst"],r["mean"],r["order"]))
    eligible = [r for r in procedures if r["worst"] <= 1.01*best["worst"] and
                all(a <= 1.02*b for a,b in zip(r["ratios"],best["ratios"]))]
    chosen = min(eligible, key=lambda r:(r["shared"],r["mean"],r["order"]))
    return dict(chosen=chosen, best_worst_procedure=best, eligible=eligible, procedures=procedures)


def check_budget(tasks, reservations=()):
    """Reservations cover unfinished jobs at their full wall-time, not CPUTime."""
    amounts = {p: 0. for p in PROTOCOL["phase_core_hours"]}
    for task in tasks:
        if task["cores"] != 1 or task["minutes"] <= 0:
            raise ValueError("search requires bounded single-core CPU tasks")
        amounts[task["phase"]] += task["cores"]*task["minutes"]/60
    for entry in reservations:
        value = entry["allocated_core_hours"]
        if not 0 <= value < float("inf"):
            raise ValueError("invalid prior resource accounting")
        amounts[entry["phase"]] += value
    if any(amounts[p] > limit+1e-10 for p,limit in PROTOCOL["phase_core_hours"].items()):
        raise ValueError("phase allocation exceeds registered budget")
    if sum(amounts.values()) > PROTOCOL["max_core_hours"]+1e-10:
        raise ValueError("total allocated and reserved CPU budget exceeded")
    return amounts
