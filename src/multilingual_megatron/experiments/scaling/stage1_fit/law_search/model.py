"""Registered finite-inventory candidates and shared-coefficient fit numerics.

No filesystem, scheduler, model selection, or held-out response access here.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import lsq_linear
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc

from .. import fit
from .. import repetition_model as repetition
from ..atlas_full_model import saturation
from ..atlas_comparison_workflow import model as native
from ..m6_size_duration_workflow import model as duration

CANDIDATES = ("RN", "RN-no-composition", "RN-no-gamma", "SOURCE-RN", "SOURCE-RP")
PARAMETER_COUNTS = dict(zip(CANDIDATES, (17, 14, 16, 17, 17)))
TAUS = (0., .01, .1)
POOLED = (
    native.Spec("Pooled-ATLAS-full-active", "full"),
    native.Spec("Pooled-ATLAS-full-total", "full", parameter_count="total"),
    native.Spec("Pooled-ATLAS-other-active", "other", lambda_cap=100.),
    native.Spec("Pooled-JointMoE-JH", "joint"),
)


@dataclass
class Sources:
    D: np.ndarray
    U: np.ndarray
    N: np.ndarray
    P: np.ndarray
    readout: np.ndarray
    target: np.ndarray
    family: np.ndarray
    script: np.ndarray

    def effective(self, x, stored=False):
        count = self.P if stored else self.N
        logsize = np.log(count/fit.N_SCALE)
        factor = np.exp(x[16]*logsize)
        rate = x[15]*factor
        sat, dr = np.empty_like(self.D), np.empty_like(self.D)
        for value in np.unique(rate):
            mask = rate == value
            sat[mask], dr[mask] = saturation(self.D[mask], self.U[mask], value)
        dlambda = dr*factor[:, None]
        drho = dr*(rate*logsize)[:, None]
        r, t = self.readout, self.target
        family = (sat @ self.family.T)[r, t]
        script = (sat @ self.script.T)[r, t]
        effective = sat[r, t]+x[11]*family+x[12]*script
        derivatives = []
        for jac in (dlambda, drho):
            derivatives.append(jac[r, t]+x[11]*(jac @ self.family.T)[r, t]
                               +x[12]*(jac @ self.script.T)[r, t])
        return effective, family, script, *derivatives


def candidate_terms(name, x, problem, inventory, sources=None, *, require_positive_exponents=True):
    if name not in CANDIDATES:
        raise ValueError("unregistered candidate")
    if not name.startswith("SOURCE"):
        return repetition.terms(x, problem, inventory, require_positive_exponents=require_positive_exponents)
    if sources is None:
        raise ValueError("source-wise exposures and inventories required")
    base = np.asarray(x[:15]).copy()
    base[11:13] = 0.
    cap, data, cj, dj = duration.terms(base, problem, require_positive_exponents=require_positive_exponents)
    dj[:, 11:13] = 0.
    effective, family, script, dlambda, drho = sources.effective(x, name == "SOURCE-RP")
    if not np.isfinite(effective).all() or np.any(effective <= 0):
        raise ValueError("nonpositive useful exposure")
    h = np.log(problem.E)/np.log(32.)
    beta = x[2]*(1-h)+x[3]*h
    logratio = np.log(effective/problem.D)
    factor = np.exp(-beta*logratio)
    data *= factor
    dj *= factor[:, None]
    dj[:, 2] -= data*logratio*(1-h)
    dj[:, 3] -= data*logratio*h
    dj[:, 11] = -data*beta*family/effective
    dj[:, 12] = -data*beta*script/effective
    dj = np.column_stack((dj, -data*beta*dlambda/effective, -data*beta*drho/effective))
    return cap, data, np.column_stack((cj, np.zeros((len(cap), 2)))), dj


def candidate_bounds(name, problem, inventory, sources=None):
    bounds = [repetition.Bound(b.name, b.lower, b.upper, b.kind) for b in fit.initial_bounds("M6")]
    bounds += [repetition.Bound("zeta", 0., .5, "nonnegative"),
               repetition.Bound("lambda_repetition", 0., 10., "nonnegative"),
               repetition.Bound("rho", -2., 2., "signed")]
    fixed = set()
    if name == "RN-no-composition":
        fixed.update((11, 12, 13))
    if name == "RN-no-gamma" or np.max(problem.K) <= 12:
        fixed.add(10)
    if np.all(problem.D_family == 0):
        fixed.add(11)
    if np.all(problem.D_script == 0):
        fixed.add(12)
    if np.all(problem.D_other == 0):
        fixed.add(13)
    no_repetition = (np.all(sources.D <= sources.U) if name.startswith("SOURCE") and sources is not None
                     else np.all(problem.D <= inventory))
    if no_repetition:
        fixed.update((15, 16))
    for i in fixed:
        bounds[i] = repetition.Bound(bounds[i].name, 0., 0., bounds[i].kind)
    return bounds


class Model:
    def __init__(self, name, problem, inventory, sources=None, tau=0., packets=None):
        self.name, self.problem, self.inventory = name, problem, np.asarray(inventory)
        self.sources, self.tau, self.packets = sources, float(tau), packets
        self.equalities = ()
        self.spec = next((s for s in POOLED if s.name == name), None)
        if name in CANDIDATES:
            self.bounds = candidate_bounds(name, problem, self.inventory, sources)
        elif self.spec:
            lo, hi = native.bounds(self.spec)
            self.bounds = [repetition.Bound(n, float(a), float(b), "signed")
                           for n, a, b in zip(native.names(self.spec)[3:], lo[3:], hi[3:])]
        elif name in ("M6-ND", "M5-ND", "J0-ND", "M6"):
            self.bounds = candidate_bounds("RN", problem, self.inventory)[:15]
            fixed = {"M6-ND": (), "M5-ND": (11, 12, 13), "J0-ND": tuple(range(6,14)), "M6": (14,)}[name]
            for i in fixed:
                self.bounds[i] = repetition.Bound(self.bounds[i].name, 0., 0., "nonnegative")
        else:
            self.bounds = list(fit.initial_bounds(name))

    def terms(self, x):
        p = self.problem
        if self.name in CANDIDATES:
            return candidate_terms(self.name, x, p, self.inventory, self.sources,
                require_positive_exponents=getattr(self, "require_positive_exponents", True))
        if self.spec:
            cap, data = np.empty_like(p.y), np.empty_like(p.y)
            cj, dj = np.zeros((len(p.y), len(x))), np.zeros((len(p.y), len(x)))
            for indices, packet in self.packets:
                vector = np.r_[0., 1., 0., x]
                cap[indices], jac = native.predict(self.spec, vector, packet)
                cj[indices] = jac[:, 3:]
                vector[1:3] = (0., 1.)
                data[indices], jac = native.predict(self.spec, vector, packet)
                dj[indices] = jac[:, 3:]
            return cap, data, cj, dj
        if self.name in ("M6-ND", "M5-ND", "J0-ND", "M6"):
            return duration.terms(x, p, require_positive_exponents=getattr(self, "require_positive_exponents", True))
        return fit.predictor_terms_and_jacobian(self.name, x, p.N, p.D, p.E, p.K)

    def penalty(self, x):
        if self.name in CANDIDATES:
            value, grad = repetition.penalty(x, self.problem)
            scale = self.tau*self.problem.regularization.m2_reference_objective
            return value+.5*scale*x[16]**2, grad+np.eye(len(x))[16]*scale*x[16]
        if self.name in ("M6-ND", "M5-ND", "J0-ND", "M6"):
            return duration.penalty(x, self.problem)
        return 0., np.zeros(len(x))

    def constraint_matrix(self, size):
        if not getattr(self, "coupled_exponent_constraints", getattr(self, "require_positive_exponents", True)):
            return None
        labels = [b.name for b in self.bounds]
        if "gamma_n" not in labels:
            return None
        matrix = np.zeros((2, size))
        for row, alpha in enumerate(("alpha_e1", "alpha_e32")):
            matrix[row, labels.index(alpha)] = 1.
            matrix[row, labels.index("gamma_n")] = 1.
        return matrix


class Profile(duration.Profile):
    def __init__(self, model, delta, deadline=None):
        super().__init__(model.problem, delta)
        self.model, self.deadline = model, deadline

    def evaluate(self, x):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("registered optimization deadline reached")
        p = self.problem
        cap, data, cj, dj = self.model.terms(x)
        prediction, jacobian, coefficients = np.empty_like(p.y), np.empty((len(p.y),len(x))), []
        for language, indices in self.groups:
            solution = repetition.solve_language(self.inner_problem(indices, cap, data), self.delta, language=language)
            prediction[indices] = solution.prediction
            jacobian[indices] = solution.A*cj[indices]+solution.B*dj[indices]
            coefficients.append({k: getattr(solution,k) for k in ("language","L_inf","A","B","floor_fraction")})
        value, gradient = fit.weighted_objective_and_gradient(prediction-p.y, jacobian, p.weights, self.delta)
        ridge, rg = self.model.penalty(x)
        return {"coordinates": np.asarray(x).tolist(), "objective": float(value+ridge),
                "language_coefficients": coefficients}, gradient+rg, prediction, jacobian

    def run(self, initial):
        bounds = self.model.bounds
        def objective(x):
            record, grad, _, _ = self.evaluate(x)
            return 1e4*record["objective"], 1e4*grad
        result = optimize(objective, initial, bounds, self.model)
        record, gradient, _, _ = self.evaluate(result.x)
        kkt = stationarity(result.x, gradient, bounds, self.model)
        return record | {"success": bool(result.success and kkt <= 1e-8),
                         "optimizer_success": bool(result.success), "kkt": kkt if np.isfinite(kkt) else None,
                         "message": str(result.message), "iterations": int(result.nit)}


def optimize(objective, initial, bounds, model, *, ftol=1e-12):
    if hasattr(model, "solver_policy"):
        from ..paper_refit_workflow.numerics import optimize as controlled_optimize
        return controlled_optimize(objective, initial, bounds, model)
    matrix = model.constraint_matrix(len(initial))
    constraints = [] if matrix is None else [LinearConstraint(matrix, 1e-6, np.inf)]
    for first,second in model.equalities:
        row = np.zeros(len(initial)); row[first],row[second] = 1.,-1.
        constraints.append(LinearConstraint(row,0.,0.))
    return minimize(objective, initial, method="SLSQP", jac=True,
                    bounds=[(b.lower,b.upper) for b in bounds], constraints=constraints,
                    options={"ftol": ftol, "maxiter": 4000})


def stationarity(x, gradient, bounds, model):
    x, gradient = np.asarray(x), np.asarray(gradient)
    free = np.array([b.lower != b.upper for b in bounds])
    normals = []
    slacks = []
    for i,b in enumerate(bounds):
        if not free[i]:
            continue
        if x[i] < b.lower-1e-12 or x[i] > b.upper+1e-12:
            return float("inf")
        if np.isfinite(b.lower) and x[i]-b.lower <= 1e-7*max(1.,abs(b.lower)):
            normals.append(np.eye(len(x))[i])
            slacks.append(max(0.,x[i]-b.lower))
        if np.isfinite(b.upper) and b.upper-x[i] <= 1e-7*max(1.,abs(b.upper)):
            normals.append(-np.eye(len(x))[i])
            slacks.append(max(0.,b.upper-x[i]))
    matrix = model.constraint_matrix(len(x))
    if matrix is not None:
        slack = matrix@x-1e-6
        if np.min(slack) < -1e-12:
            return float("inf")
        active = slack <= 1e-7
        normals.extend(matrix[active]); slacks.extend(np.maximum(slack[active],0.))
    for first,second in model.equalities:
        if abs(x[first]-x[second]) > 1e-10:
            return float("inf")
        row = np.zeros(len(x)); row[first],row[second] = 1.,-1.
        normals.extend((row,-row)); slacks.extend((0.,0.))
    g = gradient[free]
    if not len(g):
        return 0.
    if normals:
        a = np.asarray(normals).T[free]
        result = lsq_linear(a,g,bounds=(0.,np.inf),method="bvls",tol=1e-14)
        g = g-a@result.x
        if not result.success:
            return float("inf")
        return float(max(np.max(abs(g)),np.max(result.x*np.asarray(slacks))))
    return float(np.max(abs(g)))


def predict(model, selected):
    cap, data, _, _ = model.terms(selected["coordinates"])
    coefficients = {c["language"]: c for c in selected["language_coefficients"]}
    c = np.array([[coefficients[l][k] for k in ("L_inf","A","B")] for l in model.problem.language_ids])
    return c[:,0]+c[:,1]*cap+c[:,2]*data


class Direct:
    def __init__(self, profile, selected):
        self.profile, self.model = profile, profile.model
        self.n = len(selected["coordinates"])
        self.initial = list(selected["coordinates"])
        self.bounds = list(self.model.bounds)
        coefficients = {c["language"]:c for c in selected["language_coefficients"]}
        cap,data,_,_=self.model.terms(selected["coordinates"])
        for language,idx in profile.groups:
            c = coefficients[language]
            low,high=profile.inner_problem(idx,cap,data).strict_bounds
            f = np.clip((c["L_inf"]-low[0])/(high[0]-low[0]),1e-12,1-1e-12)
            self.initial += [float(np.log(f/(1-f))),float(np.log(c["A"])),float(np.log(c["B"]))]
            self.bounds += [repetition.Bound("floor",-np.inf,np.inf,"signed"),
                repetition.Bound("A",float(np.log(low[1])),np.inf,"signed"),
                repetition.Bound("B",float(np.log(low[2])),np.inf,"signed")]

    def evaluate(self,z):
        if self.profile.deadline is not None and time.monotonic() >= self.profile.deadline:
            raise TimeoutError("registered direct-audit deadline reached")
        p = self.profile.problem
        cap,data,cj,dj = self.model.terms(z[:self.n])
        prediction,gradient,coefficients = np.empty_like(p.y),np.zeros(len(z)),[]
        value = 0.
        for i,(language,idx) in enumerate(self.profile.groups):
            off = self.n+3*i
            fraction,A,B = expit(z[off]),np.exp(z[off+1]),np.exp(z[off+2])
            low,high = self.profile.inner_problem(idx,cap,data).strict_bounds
            span = high[0]-low[0]
            floor = low[0]+span*fraction
            prediction[idx] = floor+A*cap[idx]+B*data[idx]
            loss,score = fit.loss_and_score(prediction[idx]-p.y[idx],self.profile.delta)
            weighted = p.weights[idx]*score
            value += p.weights[idx]@loss
            gradient[:self.n] += (A*cj[idx]+B*dj[idx]).T@weighted
            gradient[off:off+3] = [weighted.sum()*span*fraction*(1-fraction),weighted@(A*cap[idx]),weighted@(B*data[idx])]
            coefficients.append(dict(language=language,L_inf=float(floor),A=float(A),B=float(B),floor_fraction=float(fraction)))
        ridge,rg = self.model.penalty(z[:self.n]); gradient[:self.n] += rg
        if not np.isfinite(prediction).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite direct fit")
        return float(value+ridge),gradient,prediction,coefficients

    def run(self, *, ftol=1e-12):
        def objective(z):
            value,gradient,_,_ = self.evaluate(z)
            return 1e4*value,1e4*gradient
        result = optimize(objective,self.initial,self.bounds,self.model,ftol=ftol)
        value,gradient,prediction,coefficients = self.evaluate(result.x)
        transformed_kkt = stationarity(result.x,gradient,self.bounds,self.model)
        # Certify the original physical objective, not merely a small log-space
        # gradient caused by a vanishing amplitude or sigmoid derivative.
        physical,physical_gradient = list(result.x[:self.n]),list(gradient[:self.n])
        bounds=list(self.model.bounds)
        cap,data,_,_=self.model.terms(result.x[:self.n])
        p=self.profile.problem
        for c,(_,idx) in zip(coefficients,self.profile.groups):
            low,high=self.profile.inner_problem(idx,cap,data).strict_bounds
            weighted=p.weights[idx]*fit.loss_and_score(prediction[idx]-p.y[idx],self.profile.delta)[1]
            physical += [c["L_inf"],c["A"],c["B"]]
            physical_gradient += [float(weighted.sum()),float(weighted@cap[idx]),float(weighted@data[idx])]
            bounds += [repetition.Bound(k,float(a),float(b),"signed") for k,a,b in zip(("floor","A","B"),low,high)]
        kkt=stationarity(physical,physical_gradient,bounds,self.model)
        return dict(success=bool(result.success and kkt<=1e-8),optimizer_success=bool(result.success),
                    objective=value,coordinates=result.x[:self.n].tolist(),language_coefficients=coefficients,
                    kkt=kkt if np.isfinite(kkt) else None,
                    transformed_kkt=transformed_kkt if np.isfinite(transformed_kkt) else None,
                    message=str(result.message),iterations=int(result.nit))


def starts(model, anchor=None, seed=20260913):
    bounds = model.bounds
    lo,hi = np.array([[b.lower,b.upper] for b in bounds]).T
    sample = qmc.Sobol(len(bounds),scramble=True,seed=seed).random_base2(6)[:48]
    values = lo+sample*(hi-lo)
    labels = [b.name for b in bounds]
    for i,label in enumerate(labels):
        if label.startswith(("alpha","beta")):
            values[:,i] = np.clip(.15+.6*sample[:,i],lo[i],hi[i])
        elif label.startswith(("phi","psi")) or label in ("delta","omega","gamma_n"):
            values[:,i] = np.clip((sample[:,i]-.5)*.1,lo[i],hi[i])
        elif label in ("lambda_repetition","lambda"):
            values[:,i] = np.clip(10**(-3+4*sample[:,i]),lo[i],hi[i])
        elif label.startswith("eta") or label in ("gamma_other","zeta"):
            values[:,i] = np.clip(sample[:,i]*.2,lo[i],hi[i])
    if anchor is not None:
        a = np.asarray(anchor,float)
        if len(a) != len(bounds):
            raise ValueError("wrong training-only anchor dimension")
        for j,(rate,rho) in enumerate((r,p) for r in (0.,.01,.1,1.) for p in (-1.,0.,1.)):
            if j and "lambda_repetition" not in labels:
                break
            values[j] = a
            if "lambda_repetition" in labels:
                values[j,labels.index("lambda_repetition")] = rate
                values[j,labels.index("rho")] = rho
    values = np.clip(values,lo,hi)
    matrix = model.constraint_matrix(len(bounds))
    if matrix is not None:
        for row in values:
            if np.min(matrix@row) < 1e-6:
                row[labels.index("gamma_n")] = max(lo[labels.index("gamma_n")],0.)
    return values
