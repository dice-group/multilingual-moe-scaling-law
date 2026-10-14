"""Exact RN restrictions and shared-load kernel with analytic derivatives."""
from dataclasses import replace

import numpy as np
from scipy.stats import qmc

from ..law_search import model as legacy
from .. import repetition_model
from ..nd_coupling.model import Reduction
from .contracts import ALL_FORMS, DATA_EXPONENT_FORMS, EXPERT_FORMS, EXPERT_RULES, ROUND2_FORMS, ROUND3_FORMS, RULES


class Model(legacy.Model):
    """19-slot embedding: RN's 17 coordinates followed by kappa and chi."""
    def __init__(self, name, problem, inventory, *, stored_parameters=None):
        if name not in ALL_FORMS:
            raise ValueError(f"unknown coverage law: {name}")
        super().__init__("RN-no-composition", problem, inventory)
        self.name = name
        self.expert_transform = name.endswith("-ehat")
        self.data_coverage = name in DATA_EXPONENT_FORMS
        self.load_form = name.startswith("Shared-load")
        self.burden_form = name.startswith("Cap-burden")
        self.stored_form = name in ("RN14-stored-data", "RN14-stored-reuse", "RN12-stored-common-K",
            "RN13-psi-tied-stored", "Cap-burden-stored-13", "Cap-burden-stored-gamma14",
            "RN16-stored-reuse-ehat") or self.data_coverage
        self.stored_parameters = None
        if self.stored_form:
            stored = np.asarray(stored_parameters, dtype=float)
            if (stored.shape != problem.N.shape or not np.isfinite(stored).all()
                    or np.any(stored < problem.N)):
                raise ValueError("aligned finite stored parameter counts >= active counts required")
            self.stored_parameters = stored
        self.bounds += [repetition_model.Bound("kappa", 0., 0., "nonnegative"),
                        repetition_model.Bound("chi", 0., 0., "nonnegative")]
        if self.data_coverage:
            self.bounds[17] = replace(self.bounds[10], name="gamma_d")
            if name == "RN14-stored-data-exponent":
                self.bounds[10] = replace(self.bounds[10], lower=0., upper=0.)
            elif name == "RN14-stored-shared-exponent":
                self.equalities = ((10, 17),)
        if name in ("RN12-common-K", "RN12-stored-common-K"):
            self.equalities = ((6, 7), (8, 9))
        elif name == "RN13-capacity-tied":
            self.equalities = ((6, 7),)
        elif name in ("RN13-psi-tied", "RN14-psi-tied-zetaE", "RN13-psi-tied-stored", "RN15-psi-tied-ehat"):
            self.equalities = ((8, 9),)
        if self.expert_transform:
            self.bounds[17] = repetition_model.Bound("log_E_start", *EXPERT_RULES["log_E_start_bounds"], "signed")
            self.bounds[18] = repetition_model.Bound("E_start_over_E_max", *EXPERT_RULES["expert_ratio_bounds"], "nonnegative")
        if name == "RN14-psi-tied-zetaE":
            self.bounds[14] = replace(self.bounds[14], name="zeta_e1")
            self.bounds[17] = repetition_model.Bound("zeta_e32", 0., .5, "nonnegative")
        if self.burden_form:
            fixed = (6, 7) if name == "Cap-burden-stored-gamma14" else (6, 7, 10)
            for index in fixed:
                self.bounds[index] = replace(self.bounds[index], lower=0., upper=0.)
            self.bounds[17] = repetition_model.Bound("kappa", 0., 1., "nonnegative")
            self.bounds[18] = repetition_model.Bound("chi", 0., 2., "nonnegative")
            if name == "Cap-burden-chi0":
                self.bounds[18] = replace(self.bounds[18], upper=0.)
            if np.all(problem.K == 1):
                for index in (17, 18):
                    self.bounds[index] = replace(self.bounds[index], lower=0., upper=0.)
        if self.load_form:
            for index in (7, 9, 10):
                self.bounds[index] = replace(self.bounds[index], lower=0., upper=0.)
            self.bounds[6] = replace(self.bounds[6], lower=0., upper=1.)
            self.bounds[17] = replace(self.bounds[17], lower=RULES["kappa_bounds"][0], upper=RULES["kappa_bounds"][1])
            self.bounds[18] = replace(self.bounds[18], lower=RULES["chi_bounds"][0], upper=RULES["chi_bounds"][1])
            if name == "Shared-load-chi0":
                self.bounds[18] = replace(self.bounds[18], lower=0., upper=0.)
            if np.all(problem.K == 1):
                for index in (6, 8, 17, 18):
                    self.bounds[index] = replace(self.bounds[index], lower=0., upper=0.)

    def constraint_matrix(self, size):
        matrix = super().constraint_matrix(size)
        if not getattr(self, "coupled_exponent_constraints", getattr(self, "require_positive_exponents", True)):
            return None
        if not self.data_coverage:
            return matrix
        extra = np.zeros((2, size))
        extra[0, [2, 17]], extra[1, [3, 17]] = 1., 1.
        return np.vstack((matrix, extra))

    def terms(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (19,) or not np.isfinite(x).all():
            raise ValueError("expected 19 finite embedded coordinates")
        base = x[:17].copy()
        if self.load_form:
            base[6:14] = 0.
        if self.stored_form:
            # Start with unsaturated target exposure; apply the stored-size
            # kernel below, including its own lambda/rho derivatives.
            base[15:17] = 0.
        cap, term, cj, dj = repetition_model.terms(base, self.problem, self.inventory,
            require_positive_exponents=getattr(self, "require_positive_exponents", True))
        cj, dj = (np.column_stack((j, np.zeros((len(cap), 2)))) for j in (cj, dj))
        if self.stored_form:
            term, dj = self.stored_terms(x, term, dj)
        if self.expert_transform:
            from .expert_transform import correct
            return correct(x, self.problem, self.inventory, self.stored_parameters, cap, term, cj, dj)
        if self.name == "RN14-psi-tied-zetaE":
            h = np.log(self.problem.E)/np.log(32.)
            logn = np.log(self.problem.N/1e8)
            factor = np.exp(h*(x[17]-x[14])*logn)
            term, dj = term*factor, dj*factor[:, None]
            dj[:, 14], dj[:, 17] = term*(1-h)*logn, term*h*logn
        if self.burden_form:
            p = self.problem
            logn, loge, logk = np.log(p.N/1e8), np.log(p.E), np.log(p.K)
            h = loge/np.log(32.)
            original = np.exp(x[4]*loge-(x[0]*(1-h)+x[1]*h)*logn)
            unit_burden = logk*np.exp(-x[0]*logn-x[18]*loge)
            burden = x[17]*unit_burden
            cap = original+burden
            cj[:] = 0.
            cj[:, 0] = -logn*((1-h)*original+burden)
            cj[:, 1], cj[:, 4] = -logn*h*original, loge*original
            cj[:, 17], cj[:, 18] = unit_burden, -loge*burden
            if self.name == "Cap-burden-stored-gamma14":
                hinge = np.maximum(np.log(p.K/12.), 0.)/np.log(200./12.)
                factor = np.exp(-x[10]*hinge*logn)
                cap, cj = cap*factor, cj*factor[:, None]
                cj[:, 10] = -cap*hinge*logn
        if not self.load_form:
            return cap, term, cj, dj
        loge = np.log(self.problem.E)
        relief = (self.problem.K-1.)*np.exp(-x[18]*loge)
        load = 1.+x[17]*relief
        logload = np.log1p(x[17]*relief)
        cap_factor, term_factor = np.exp(x[6]*logload), np.exp(x[8]*logload)
        cap, term = cap*cap_factor, term*term_factor
        cj, dj = cj*cap_factor[:, None], dj*term_factor[:, None]
        cj[:, 6:14], dj[:, 6:14] = 0., 0.
        cj[:, 6], dj[:, 8] = cap*logload, term*logload
        for index, derivative in ((17, relief/load), (18, -x[17]*relief/load*loge)):
            cj[:, index], dj[:, index] = cap*x[6]*derivative, term*x[8]*derivative
        return cap, term, cj, dj

    def stored_terms(self, x, term, jac):
        """Target-only saturation at observed stored size, not E*N_active."""
        p = self.problem
        logm = np.log(self.stored_parameters/1e8)
        size_factor = np.exp(x[16]*logm)
        rate = x[15]*size_factor
        useful, derivative = np.empty_like(p.D), np.empty_like(p.D)
        for value in np.unique(rate):
            mask = rate == value
            useful[mask], derivative[mask] = repetition_model.saturation(
                p.D[mask], self.inventory[mask], value)
        h = np.log(p.E)/np.log(32.)
        beta = x[2]*(1-h)+x[3]*h
        log_ratio = np.log(useful/p.D)
        log_factor = -beta*log_ratio
        if self.data_coverage:
            hinge = np.maximum(np.log(p.K/12.), 0.)/np.log(200./12.)
            log_factor -= x[17]*hinge*np.log(useful/1e9)
        if self.name == "RN14-stored-data":
            log_factor += x[14]*(logm-np.log(p.N/1e8))
        factor = np.exp(log_factor)
        term, jac = term*factor, jac*factor[:, None]
        jac[:, 2] -= term*log_ratio*(1-h)
        jac[:, 3] -= term*log_ratio*h
        if self.data_coverage:
            beta = beta+x[17]*hinge
            jac[:, 17] = -term*hinge*np.log(useful/1e9)
        rate_gradient = -term*beta*derivative/useful
        jac[:, 15], jac[:, 16] = rate_gradient*size_factor, rate_gradient*rate*logm
        if self.name == "RN14-stored-data":
            jac[:, 14] = term*logm
        return term, jac

    def penalty(self, x):
        if self.load_form or self.data_coverage or self.name in ROUND2_FORMS or self.name in ROUND3_FORMS or self.name in EXPERT_FORMS:
            return 0., np.zeros(len(x))
        value, gradient = repetition_model.penalty(np.asarray(x)[:17], self.problem)
        return value, np.r_[gradient, 0., 0.]


def starts(model, anchor):
    """Deterministic fold-local starts; no all-grid/outer anchor enters inner fits."""
    # The same original RN Sobol coordinates feed all restrictions.
    base = legacy.Model("RN-no-composition", model.problem, model.inventory)
    base_anchor = np.r_[np.asarray(anchor, float), .1, 0.]
    if base_anchor.shape != (17,):
        raise ValueError("expected the 15-coordinate training-only M6-ND anchor")
    result = np.column_stack((legacy.starts(base, base_anchor, RULES["seed"]), np.zeros((48, 2))))
    if model.load_form:
        sample = qmc.Sobol(2, scramble=True, seed=RULES["seed"]+1).random_base2(6)[:48]
        result[:, 17] = 10.**(-3.+4.*sample[:, 0])
        result[:, 18] = 2.*sample[:, 1]
        # Loss-blind anchors span no relief, moderate and inverse-E relief.
        result[:12, 17] = np.tile([.01, .1, 1., 10.], 3)
        result[:12, 18] = np.repeat([0., .5, 1.], 4)
    elif model.burden_form:
        sample = qmc.Sobol(2, scramble=True, seed=RULES["seed"]+2).random_base2(6)[:48]
        result[:, 17], result[:, 18] = sample[:, 0], 2.*sample[:, 1]
        result[:12, 17] = np.tile([0., .01, .1, 1.], 3)
        result[:12, 18] = np.repeat([0., .5, 1.], 4)
        if model.name == "Cap-burden-stored-gamma14":
            result[:12, 10] = 0.  # exact nested-null starts, declared before fitting
    elif model.name == "RN14-psi-tied-zetaE":
        sample = qmc.Sobol(1, scramble=True, seed=RULES["seed"]+3).random_base2(6)[:48, 0]
        result[:, 17] = .5*sample
        result[:12, 17] = result[:12, 14]
    elif model.expert_transform:
        sample = qmc.Sobol(2, scramble=True, seed=RULES["seed"]+4).random_base2(6)[:48]
        for j in (17, 18):
            bound = model.bounds[j]
            result[:, j] = bound.lower+(bound.upper-bound.lower)*sample[:, j-17]
        result[:12, 17:] = 0.  # paired raw-null starts, not outcome-selected anchors
    elif model.data_coverage:
        if model.name == "RN14-stored-data-exponent":
            result[:, 17], result[:, 10] = result[:, 10].copy(), 0.
        elif model.name == "RN14-stored-shared-exponent":
            result[:, 17] = result[:, 10]
        else:
            sample = qmc.Sobol(1, scramble=True, seed=RULES["seed"]+5).random_base2(6)[:48, 0]
            result[:, 17] = .1*(sample-.5)
            result[:12, 17] = 0.
        for row in result:
            if min(row[2], row[3])+row[17] < 1e-6:
                row[17] = 0.
                if model.equalities == ((10, 17),):
                    row[10] = 0.
    reduction = Reduction(model)
    return np.array([reduction.expand(reduction.pack(row)) for row in result])


def validate(model, record):
    x = np.asarray(record["coordinates"], float)
    if x.shape != (19,) or not np.isfinite(x).all():
        raise ValueError("invalid embedded fitted coordinates")
    if any(not b.lower-1e-12 <= value <= b.upper+1e-12 for value, b in zip(x, model.bounds)):
        raise ValueError("fitted coordinate violates registered bounds")
    if any(abs(x[a]-x[b]) > 1e-10 for a, b in model.equalities):
        raise ValueError("fitted coordinate violates equality")
    matrix = model.constraint_matrix(len(x))
    if matrix is not None and np.min(matrix@x) < 1e-6-1e-12:
        raise ValueError("capacity or data exponent positivity violated")
