"""Scientific decisions and deterministic task/selection contracts; no I/O."""
from copy import deepcopy
import math

from ..law_search.protocol import INNER
from ..law_search.protocol import PRIMARY

CANDIDATES = {"RN12-common-K": 12, "RN13-capacity-tied": 13, "Shared-load-13": 13}
CONTROLS = {"RN14-matched": 14, "RN14-no-ridge": 14, "Shared-load-chi0": 12}
FORMS = {**CANDIDATES, **CONTROLS}
FOLDS = (*INNER, *PRIMARY, "all_grid")
SCREEN = INNER[:3]
STAGES = ("starts", "certify", "select", "report")
RULES = dict(
    schema="stage1_coverage_laws/v1", candidates=CANDIDATES, controls=CONTROLS,
    folds=list(FOLDS), screen=list(SCREEN), diagnostic_inner="inner_terminal",
    starts=48, starts_per_shard=4, seed=2026092101, polish_attempts=2,
    kappa_bounds=[0., 10.], chi_bounds=[0., 2.], phi_shared_load_bounds=[0., 1.],
    repetition="inherited exponential S(D_t,U_t;lambda*n^rho)",
    local_coefficients=["L_inf", "A", "B"], tau=0.,
    objective="inherited training-only Huber/calibration/floors/configuration weights",
    regularization="A/B/RN14-matched: exact inherited ridge; load forms and RN14-no-ridge: zero ridge",
    optimizer="inherited SLSQP; analytic gradients; ftol=1e-12; maxiter=4000",
    gradient_tolerance=1e-8, objective_tolerance=1e-6, prediction_tolerance=1e-4,
    replay_tolerance=1e-10, simplicity_band=1.01, per_view_guard=1.02,
    rn14_guard=1.02, win_tolerance_nats=1e-6,
    selection="3 configuration-level inner folds; minimum worst RMSE ratio to matched RN14; 1% band, 2% best-view/RN14 guards; count, mean, registered order",
    finalists=list(CANDIDATES),
    parallel_final_fits="all forms/folds predeclared; isolated training packets/anchors; no score until sealed inner choice",
    retry_policy="no blind resubmission; interrupted intents require reconciliation",
    gpu_training=False,
    evidence="developmental known grid/L26 outcomes; nested readouts are not independent seeds",
)

ROUND2_CANDIDATES = {"Cap-burden-13": 13, "RN14-stored-data": 14,
                     "RN14-psi-tied-zetaE": 14}
ROUND2_CONTROLS = {"Cap-burden-chi0": 12, "RN14-stored-reuse": 14,
                  "RN13-psi-tied": 13, "RN14-no-ridge": 14}
ROUND2_FORMS = {**ROUND2_CANDIDATES, **ROUND2_CONTROLS}
ROUND2_RULES = RULES | dict(
    schema="stage1_law_improvement_round2/v1", protocol="law_improvement_round2_v1",
    candidates=ROUND2_CANDIDATES, controls=ROUND2_CONTROLS,
    reference="RN14-no-ridge", finalists=list(ROUND2_CANDIDATES),
    kappa_bounds=[0., 1.],
    repetition="target-only exponential S(D_t,U_t;lambda*x^rho), declared active/stored x per form",
    regularization="zero ridge for all round-2 candidates and controls",
    selection="3 configuration-level inner folds; minimum worst RMSE ratio to RN14-no-ridge; 1% band, 2% best-procedure-view/reference guards; count, mean, registered order",
)

ROUND3_CANDIDATES = {"RN13-psi-tied-stored": 13, "Cap-burden-stored-13": 13,
                     "Cap-burden-stored-gamma14": 14}
ROUND3_CONTROLS = {"RN14-stored-reuse": 14, "RN13-psi-tied": 13,
                  "Cap-burden-13": 13, "RN14-no-ridge": 14}
ROUND3_FORMS = {**ROUND3_CANDIDATES, **ROUND3_CONTROLS}
EXPERT_PAIRS = {"RN13-psi-tied": "RN15-psi-tied-ehat",
                "RN14-stored-reuse": "RN16-stored-reuse-ehat"}
EXPERT_FORMS = {"RN13-psi-tied": 13, "RN15-psi-tied-ehat": 15,
                "RN14-stored-reuse": 14, "RN16-stored-reuse-ehat": 16}
EXPERT_RULES = RULES | dict(
    schema="stage1_expert_transform/v1", protocol="expert_transform_v1",
    candidates={"RN15-psi-tied-ehat": 15},
    controls={k: v for k, v in EXPERT_FORMS.items() if k != "RN15-psi-tied-ehat"},
    reference="RN13-psi-tied", eligible=["RN13-psi-tied", "RN15-psi-tied-ehat"],
    finalists=["RN15-psi-tied-ehat"], paired_ablation=EXPERT_PAIRS,
    expert_transform="JointMoE Eq. e_transformation; s=exp(log_E_start), r=s/E_max; raw limit s=1,r=0",
    log_E_start_bounds=[-math.log(16.), math.log(16.)], expert_ratio_bounds=[0., .95],
    expert_interpolation="h=(log(Ehat(E))-log(Ehat(1)))/(log(Ehat(32))-log(Ehat(1)))",
    transform_scope="all explicit E prefactors and exponent interpolation; observed N_total, N_act, K, D, U unchanged",
    transform_starts="same 48 inherited core starts; first 12 exact raw limit, 36 loss-blind Sobol shape starts",
    repetition="unchanged active-size RN13 and stored-size RN14 target-only kernels",
    regularization="zero ridge in both matched pairs",
    selection="paper RN13 pair only; inherited inner 3-view minimax/1% simplicity/2% guards; stored pair is secondary sensitivity",
)
EXPERT_PARALLEL_RULES = EXPERT_RULES | dict(
    schema="stage1_expert_transform/parallel_v1", protocol="expert_transform_parallel_v1",
    starts_per_shard=1, max_concurrency=1536,
    execution_amendment="one restart per task; same 48 starts, equations, certification and selection; no results observed",
)
DATA_EXPONENT_FORMS = {"RN14-stored-data-exponent": 14,
                      "RN14-stored-shared-exponent": 14,
                      "RN15-stored-separate-exponents": 15}
DATA_EXPONENT_RULES = RULES | dict(
    schema="stage1_coverage_exponents/v1", protocol="coverage_exponents_v1",
    candidates=DATA_EXPONENT_FORMS, controls={"RN14-stored-reuse": 14},
    reference="RN14-stored-reuse",
    eligible=["RN14-stored-reuse", *DATA_EXPONENT_FORMS],
    finalists=list(DATA_EXPONENT_FORMS), starts_per_shard=1, max_concurrency=1536,
    paired_ablation=[["RN14-stored-reuse", name] for name in DATA_EXPONENT_FORMS],
    gamma_d_bounds="same signed bounds as inherited gamma_n; zero when training K<=12",
    positivity="alpha and beta endpoint-plus-coverage slopes >=1e-6 over E=1..32,K=1..200",
    repetition="unchanged stored-size S(D_t,U_t;lambda*(N_total/1e8)^rho); beta(E,K) acts on S/1e9",
    regularization="zero ridge for all four forms",
    starts_description="matched 48 inherited core starts; data-only relocates gamma_n start to gamma_d; shared ties both; separate first12 gamma_d=0 and36 signed Sobol starts; feasibility repair only",
    selection="all four eligible on 3 inner configuration folds; inherited minimax ratios, 1% simplicity band, 2% guards, count/mean/order; no automatic paper promotion",
    scope="question 1 only: exponent ablation; no resource-allocation or doubling study",
)
ALL_FORMS = {**FORMS, **ROUND2_FORMS, **ROUND3_FORMS, **EXPERT_FORMS, **DATA_EXPONENT_FORMS,
             "RN12-stored-common-K": 12}
ROUND3_RULES = RULES | dict(
    schema="stage1_law_improvement_round3/v1", protocol="law_improvement_round3_v1",
    candidates=ROUND3_CANDIDATES, controls=ROUND3_CONTROLS,
    reference="RN14-no-ridge", eligible=list(ROUND3_FORMS), finalists=list(ROUND3_CANDIDATES),
    kappa_bounds=[0., 1.],
    repetition="unchanged target-only exponential kernel; stored size only in declared repetition rates",
    regularization="zero ridge for every round-3 procedure",
    selection="all seven procedures eligible prospectively for this developmental round; inherited 3-view ratios to RN14-no-ridge, 1% band, 2% best-procedure/reference guards, count/mean/order",
    diagnostics_after_report=True,
)


def rules_for(profile=None):
    """Finite named protocols; absent profile key preserves the sealed v1 rules."""
    protocol = (profile or {}).get("protocol")
    registry = {None: RULES, "law_improvement_round2_v1": ROUND2_RULES,
                "law_improvement_round3_v1": ROUND3_RULES,
                "expert_transform_v1": EXPERT_RULES,
                "expert_transform_parallel_v1": EXPERT_PARALLEL_RULES,
                "coverage_exponents_v1": DATA_EXPONENT_RULES}
    if protocol not in registry:
        raise ValueError(f"unregistered coverage-law protocol: {protocol}")
    return deepcopy(registry[protocol])


def plan_rules(plan):
    """Reject profile/rules disagreement, while supporting old rules-less fixtures."""
    expected = rules_for(plan.get("profile"))
    if plan.get("rules", expected) != expected:
        raise ValueError("sealed protocol rules changed")
    return expected


def forms_for(rules=None):
    rules = RULES if rules is None else rules
    return {**rules["candidates"], **rules["controls"]}


def cells(rules=None):
    rules = RULES if rules is None else rules
    return [dict(id=f"{fold}--{form}", fold=fold, form=form)
            for fold in rules["folds"] for form in forms_for(rules)]


def tasks(profile, rules=None):
    rules = rules_for(profile) if rules is None else rules
    result = []
    for cell in cells(rules):
        for first in range(0, rules["starts"], rules["starts_per_shard"]):
            result.append(cell | dict(id=f"{cell['id']}--{first:02d}", cell=cell["id"], stage="starts",
                          start_indices=list(range(first, first + rules["starts_per_shard"])),
                          minutes=profile["start_minutes"]))
    result += [cell | dict(stage="certify", cell=cell["id"], minutes=profile["certify_minutes"])
               for cell in cells(rules)]
    return result


def reservation(profile, rules=None):
    return (sum(t["minutes"] for t in tasks(profile, rules)) + 2 * profile["analysis_minutes"]) / 60


def validate_profile(profile, rules=None):
    expected = rules_for(profile)
    if rules is not None and rules != expected:
        raise ValueError("profile and protocol rules disagree")
    for key in ("concurrency", "start_minutes", "certify_minutes", "analysis_minutes"):
        if type(profile[key]) is not int or profile[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    maximum = expected.get("max_concurrency", 256)
    if profile["concurrency"] > maximum:
        raise ValueError(f"this campaign permits at most {maximum} concurrent single-core tasks")
    if reservation(profile, expected) > profile["max_reserved_core_hours"]:
        raise ValueError("resource reservation exceeds profile ceiling")
    for key in ("account", "partition", "memory", "workspace_root", "python_executable"):
        if not profile[key] or any(c in profile[key] for c in "\n\r\x00"):
            raise ValueError(f"invalid scheduler/profile field: {key}")


def choose(records, rules=None):
    """Only declared eligible forms and screening folds enter the choice."""
    rules = RULES if rules is None else rules
    reference, screen = rules.get("reference", "RN14-matched"), rules["screen"]
    forms = forms_for(rules)
    names = rules.get("eligible", [*rules["candidates"], reference])
    if len(names) != len(set(names)) or not set(names) <= set(forms) or reference not in names:
        raise ValueError("invalid declared selection eligibility")
    expected = {(name, fold) for name in names for fold in screen}
    selected = [r for r in records if r["form"] in names and r["fold"] in screen]
    keys = [(r["form"], r["fold"]) for r in selected]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("selection requires the complete registered screening panel")
    if any(not r["certified"] or not r["complete"] or not math.isfinite(r["rmse"])
           or r["rmse"] <= 0 for r in selected):
        raise ValueError("uncertified/incomplete inner comparison; no selection")
    lookup = {(r["form"], r["fold"]): r["rmse"] for r in selected}
    procedures = []
    for order, name in enumerate(names):
        ratios = [lookup[name, fold] / lookup[reference, fold] for fold in screen]
        procedures.append(dict(form=name, shared=forms[name], ratios=ratios,
                               worst=max(ratios), mean=sum(ratios)/len(ratios), order=order))
    best = min(procedures, key=lambda r: (r["worst"], r["mean"], r["order"]))
    eligible = [r for r in procedures if r["worst"] <= rules["simplicity_band"]*best["worst"]
                and max(r["ratios"]) <= rules["rn14_guard"]
                and all(a <= rules["per_view_guard"]*b for a, b in zip(r["ratios"], best["ratios"]))]
    if not eligible:
        raise ValueError("no form satisfies the registered guards")
    return dict(chosen=min(eligible, key=lambda r: (r["shared"], r["mean"], r["order"])),
                best_worst=best, eligible=eligible, procedures=procedures)
