"""Matched full-paper solver experiment; historical fitting defaults are untouched."""
from dataclasses import asdict
from copy import copy
from types import SimpleNamespace
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit
from scipy.stats import qmc

from .. import execution as io, fit
from ..law_search import model as shared, data, replay as oracle
from ..coverage_laws import model as coverage, replay as coverage_oracle
from ..coverage_laws.contracts import ALL_FORMS
from ..nd_coupling import model as powered
from ..problems import problem
from ..shared_numerics import best_attempt
from . import protocol


def optimize(objective, initial, bounds, model):
    """Eliminate fixed/tied coordinates identically in both optimizer arms."""
    settings = protocol.contract(model.solver_policy)
    reduction = powered.Reduction(SimpleNamespace(bounds=bounds, equalities=model.equalities))
    seed = reduction.pack(initial)

    def evaluate(z):
        value, gradient = objective(reduction.expand(z))
        return value, np.asarray(gradient) @ reduction.matrix

    constraints = []
    matrix = model.constraint_matrix(len(initial))
    if matrix is not None:
        if settings["optimizer"] == "L-BFGS-B":
            raise ValueError("L-BFGS-B must not silently discard coupled constraints")
        constraints = [LinearConstraint(matrix @ reduction.matrix,
                        1e-6-matrix @ reduction.expand(np.zeros(len(reduction.bounds))), np.inf)]
    options = dict(settings["lbfgsb"] if settings["optimizer"] == "L-BFGS-B" else settings["slsqp"])
    if hasattr(model, "audit_ftol"):
        if not 0 < model.audit_ftol <= options["ftol"]:
            raise ValueError("audit refinement may only tighten optimizer ftol")
        options["ftol"] = model.audit_ftol
    result = minimize(evaluate, seed, jac=True, method=settings["optimizer"],
                      bounds=reduction.bounds, constraints=constraints, options=options)
    result.x = reduction.expand(result.x)
    return result


def make_model(name, rows, sources, packets, calibration, policy):
    settings = protocol.contract(policy)
    regularization = dict(calibration["regularization"])
    if name in ALL_FORMS or name.startswith("Pooled-") or name == "JointMoE-Skaling-pooled-active-Dt":
        regularization["lambda"] = 0.
    p = problem(rows, regularization)
    if name in ALL_FORMS:
        model = coverage.Model(name, p, rows.U_t, stored_parameters=rows.N_total)
    elif name == "JointMoE-Skaling-pooled-active-Dt":
        model = powered.Model(name, p, rows.U_t)
    else:
        model = data.make_model(name, rows, sources, packets, regularization, delta=calibration["delta"])
    model.solver_policy = policy
    model.require_positive_exponents = False
    model.coupled_exponent_constraints = settings["positive_effective_exponents"]
    return model


def starts(model, calibration):
    """Same 48 starts in every arm, including coverage of the relaxed domain."""
    if model.name == "RN12-stored-common-K":
        parent = copy(model)
        parent.name, parent.equalities = "RN14-stored-reuse", ()
        seeds = starts(parent, calibration).copy()
        for a, b in model.equalities:
            seeds[:, a] = seeds[:, b] = (seeds[:, a]+seeds[:, b])/2
        return seeds
    # Historical near-origin/parent starts plus independent full-box exploration.
    if model.name in ALL_FORMS:
        seeds = coverage.starts(model, calibration["anchor"])
    else:
        anchor = calibration["anchor"] if model.name in ("M5-ND", "M6-ND", "M6") else None
        if model.name == "M6":
            anchor = anchor[:14] + [0.] if isinstance(anchor, list) else np.r_[anchor[:14], 0.]
        seeds = shared.starts(model, anchor=anchor, seed=protocol.SEED)
    reduction = powered.Reduction(model)
    bounds = np.asarray(reduction.bounds)
    sample = qmc.Sobol(len(bounds), scramble=True, seed=protocol.SEED).random_base2(4)
    # This replaces the same final 16 seeds in ALL arms. SLSQP receives the
    # same potentially infeasible points; only it imposes the inequalities.
    seeds[-16:] = [reduction.expand(bounds[:, 0]+u*(bounds[:, 1]-bounds[:, 0])) for u in sample]
    return np.array([reduction.expand(reduction.pack(x)) for x in seeds])


def predict(model, selected):
    return powered.prediction(model, selected) if getattr(model, "powered", False) else shared.predict(model, selected)


def independent(model, selected, rows, packets):
    name, x = model.name, np.asarray(selected["coordinates"])
    if name in ALL_FORMS:
        return coverage_oracle.predict(name, selected, rows)
    coeff = {c["language"]: c for c in selected["language_coefficients"]}
    c = np.array([[coeff[l][k] for k in ("L_inf", "A", "B")] for l in rows.language])
    if name.startswith("Pooled-"):
        out = np.empty(len(rows))
        for item in packets:
            idx = np.asarray(item["indices"], int)
            a = oracle.native(model.spec.payload(), np.r_[0., 1., 0., x], item["train"])
            b = oracle.native(model.spec.payload(), np.r_[0., 0., 1., x], item["train"])
            out[idx] = c[idx, 0]+c[idx, 1]*a+c[idx, 2]*b
        return out
    if getattr(model, "powered", False):
        n, e = rows.N_act.to_numpy()/1e8, rows.E.to_numpy()
        h = np.log(e)/np.log(32.)
        a = c[:, 1]*e**x[4]*n**(-(x[0]*(1-h)+x[1]*h))
        b = c[:, 2]*e**x[5]*(rows.D_t.to_numpy()/1e9)**(-(x[2]*(1-h)+x[3]*h))
        return c[:, 0]+np.exp(x[17]*np.logaddexp(np.log(a)/x[17], np.log(b)/x[17]))
    if name in shared.CANDIDATES:
        a, b = oracle.candidate(name, x, rows, model.sources)
        return c[:, 0]+c[:, 1]*a+c[:, 2]*b
    duration = name in ("M5-ND", "M6-ND", "J0-ND", "M6")
    parameters = fit.shared_from_outer("M6" if duration else name, x[:14] if duration else x)
    return oracle.shared_equation(dict(shared_parameters=asdict(parameters), zeta=x[14] if duration else 0.,
                                      language_coefficients=selected["language_coefficients"]), rows)


class PoweredDirect(shared.Direct):
    """Joint fitting for the nonseparable powered sum; never call it convex."""
    def evaluate(self, z):
        x, p = np.asarray(z[:self.n]), self.profile.problem
        coeff, derivative = [], []
        cap, term, _, _ = self.model.terms(x)
        for i, (language, idx) in enumerate(self.profile.groups):
            off = self.n+3*i
            low, high = self.profile.inner_problem(idx, cap, term).strict_bounds
            f = expit(z[off]); a, b = np.exp(z[off+1:off+3])
            coeff.append(dict(language=language, L_inf=low[0]+(high[0]-low[0])*f, A=a, B=b))
            derivative.append([(high[0]-low[0])*f*(1-f), a, b])
        record = dict(coordinates=x, language_coefficients=coeff)
        pred, jac, nuisance = powered.prediction(self.model, record, jacobian=True)
        value, score = fit.loss_and_score(pred-p.y, self.profile.delta)
        weighted = p.weights*score
        ridge, rg = self.model.penalty(x)
        grad = np.r_[jac.T@weighted+rg, np.zeros(3*len(coeff))]
        self.physical_gradient = grad.copy()
        for i, (_, idx) in enumerate(self.profile.groups):
            off = self.n+3*i
            physical = nuisance[idx].T@weighted[idx]
            grad[off:off+3] = physical*np.asarray(derivative[i])
            self.physical_gradient[off:off+3] = physical
        return float(p.weights@value+ridge), grad, pred, coeff

    def run(self, **kwargs):
        def objective(z):
            value, gradient, _, _ = self.evaluate(z)
            return 1e4*value, 1e4*gradient
        result = optimize(objective, self.initial, self.bounds, self.model)
        value, _, _, coeff = self.evaluate(result.x)
        x, bounds = list(result.x[:self.n]), list(self.model.bounds)
        cap, term, _, _ = self.model.terms(x)
        for c, (_, idx) in zip(coeff, self.profile.groups):
            low, high = self.profile.inner_problem(idx, cap, term).strict_bounds
            x.extend([c[k] for k in ("L_inf", "A", "B")])
            bounds.extend(shared.repetition.Bound(k, float(a), float(b), "signed")
                          for k, a, b in zip(("floor", "A", "B"), low, high))
        kkt = shared.stationarity(x, self.physical_gradient, bounds, self.model)
        return dict(success=bool(result.success and kkt <= 1e-8), optimizer_success=bool(result.success),
                    objective=value, coordinates=result.x[:self.n].tolist(), language_coefficients=coeff,
                    kkt=float(kkt) if np.isfinite(kkt) else None, message=str(result.message))


def execute(task, rows, sources, packets, calibration, policy, deadline, sink, shards):
    model = make_model(task["model"], rows, sources, packets, calibration, policy)
    profile = shared.Profile(model, calibration["delta"], deadline)
    direct = PoweredDirect if getattr(model, "powered", False) else shared.Direct
    seeds = starts(model, calibration)

    def run(seed):
        if time.monotonic() >= deadline:
            raise TimeoutError("full-paper refit deadline")
        if getattr(model, "powered", False):
            initializer = profile.evaluate(seed)[0]
            return direct(profile, initializer).run()
        return profile.run(seed)

    if task["stage"] == "fit":
        attempts = []
        for index in task["start_indices"]:
            try:
                record = run(seeds[index])
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                if isinstance(error, TimeoutError):
                    raise
                record = dict(success=False, error=f"{type(error).__name__}: {error}")
            record.update(index=index, initial=seeds[index].tolist())
            io.write_json_once(sink / f"start-{index:02d}.json", record)
            attempts.append(record)
        return dict(complete=True, attempts=attempts)
    attempts = []
    for source_task, result in shards:
        if not result.get("complete") or [a["index"] for a in result["attempts"]] != source_task["start_indices"]:
            raise ValueError("incomplete or changed restart shard")
        attempts.extend(result["attempts"])
    if sorted(a["index"] for a in attempts) != list(range(protocol.STARTS)):
        raise ValueError("48 unique restarts required")
    selected = best_attempt(attempts)
    audit, gap, passed = None, None, False
    if selected:
        audit = direct(profile, selected).run()
        a, b = predict(model, selected), predict(model, audit)
        passed = bool(audit["success"] and abs(selected["objective"]-audit["objective"]) <=
                      1e-6*max(abs(selected["objective"]), 1e-15) and np.max(abs(a-b)) <= 1e-4)
        gap = float(np.max(abs(a-independent(model, selected, rows, packets))))
        passed = passed and gap <= 1e-10
    x = dict(zip((b.name for b in model.bounds), selected["coordinates"])) if selected else {}
    exponents = {}
    for side, slope in (("alpha", "gamma_n"), ("beta", "gamma_d")):
        for endpoint in ("e1", "e32"):
            key = f"{side}_{endpoint}"
            if key in x:
                exponents[key+"_K200"] = x[key]+x.get(slope, 0.)
    return dict(certified=passed, complete=True, selected=selected, audit=audit,
        bounds=[asdict(b) for b in model.bounds], equalities=list(model.equalities),
        initial_starts=protocol.STARTS, attempts=attempts, effective_exponents=exponents,
        positive_effective_exponents=bool(exponents and min(exponents.values()) > 0),
        independent_training_replay_max_gap=gap, numerical_policy=protocol.contract(policy),
        delta=calibration["delta"], regularization=asdict(model.problem.regularization),
        error=None if passed else "restart/stationarity/joint-audit/replay contract unresolved")
