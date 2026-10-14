"""Fixed manuscript-wide refit inventory and outcome-independent adoption gates."""
from ..law_search.protocol import PRIMARY, INNER

VERSION = "paper_full_refit_20260926_v1"
NATIVE = ("Chinchilla-total-local", "Chinchilla-target-local", "JointMoE-JH-local",
          "ATLAS-capacity-local", "He-family-local", "ATLAS-full-active-positive-T100-L10",
          "ATLAS-full-total-positive-T100-L10")
SHARED = ("C0", "J0", "J0-ND", "M0", "M1", "M1-PhiE", "M1-PsiE", "M2",
          "M5-ND", "M6", "M6-ND", "RN", "RN13-psi-tied", "RN14-no-ridge",
          "RN14-stored-reuse", "RN16-stored-reuse-ehat", "RN14-stored-data-exponent",
          "RN14-stored-shared-exponent", "RN15-stored-separate-exponents",
          "Pooled-JointMoE-JH", "Pooled-ATLAS-full-active", "Pooled-ATLAS-full-total",
          "JointMoE-Skaling-pooled-active-Dt")
MODELS = (*NATIVE, *SHARED)
ADDITIONAL_ABLATIONS = ("RN12-stored-common-K",)
INNER_MODELS = ("RN13-psi-tied", "RN14-stored-reuse", "RN16-stored-reuse-ehat",
                "RN14-stored-data-exponent", "RN14-stored-shared-exponent",
                "RN15-stored-separate-exponents")
FOLDS = (*PRIMARY, "all_grid", *INNER)
POLICIES = ("lbfgsb_box", "slsqp_box", "slsqp_positive")
STARTS = 48
SEED = 2026092601


def folds(model):
    return (*PRIMARY, "all_grid", *INNER) if model in INNER_MODELS else (*PRIMARY, "all_grid")


def cells():
    return [dict(model=m, fold=f, coefficient_scope="language_local" if m in NATIVE else
                 "shared_with_language_local_amplitudes") for m in MODELS for f in folds(m)]


def contract(policy):
    if policy not in POLICIES:
        raise ValueError("unknown full-paper numerical policy")
    return dict(version=VERSION, policy=policy, shared_starts=STARTS, seed=SEED,
        optimizer="L-BFGS-B" if policy == "lbfgsb_box" else "SLSQP",
        positive_effective_exponents=policy == "slsqp_positive",
        objective="inherited fold-local weighted Huber and model-specific ridge; no log-loss change",
        inner_solver="conditioned convex Huber with IRLS fallback; powered-sum uses joint fitting",
        lbfgsb=dict(ftol=1e-15, gtol=1e-5, maxiter=20000, maxls=50, maxcor=20),
        slsqp=dict(ftol=1e-12, maxiter=4000), objective_scale=1e4,
        kkt_tolerance=1e-8, relative_objective_tolerance=1e-6, prediction_tolerance=1e-4,
        replay_tolerance=1e-10, native_starts=[16, 64],
        restart_change="48 matched training-only shared starts replace historical heterogeneous grids; controls use identical starts",
        acceptance="training-only convergence, original-coordinate KKT, restart completeness, joint audit, independent equation replay",
        positivity="diagnostic only for box arms; negative optima retained and block automatic adoption, never censored",
        recovery="no outcome-driven restart escalation or hyperparameter tuning",
        sealing="archive every outcome; manuscript adoption is a separate complete-evidence review")
