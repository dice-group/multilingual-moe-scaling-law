"""Shared-model multistart fitting and independent certification; no scheduler."""
from dataclasses import asdict
import time
import numpy as np
from . import execution as io
from .law_search import model as m


def best_attempt(attempts):
    good = [a for a in attempts if a.get("success")]
    if not good:
        return None
    best = min(good,key=lambda a:(a["objective"],a["index"]))
    if any(a.get("objective",float("inf")) < best["objective"]-1e-6*max(abs(best["objective"]),1e-15) for a in attempts):
        return None
    return best


def agreement(model, best, audit):
    if not best or not audit or not audit.get("success"):
        return False
    return bool(abs(best["objective"]-audit["objective"]) <= 1e-6*max(abs(best["objective"]),1e-15)
                and np.max(abs(m.predict(model,best)-m.predict(model,audit))) <= 1e-4)


def geometry(profile, selected):
    """Eliminate each language's three nuisance directions before shared rank."""
    _,_,pred,jac = profile.evaluate(selected["coordinates"])
    cap,data_term,_,_ = profile.model.terms(selected["coordinates"])
    robust = np.ones(len(pred)) if profile.delta is None else (abs(pred-profile.problem.y) < profile.delta).astype(float)
    root = np.sqrt(profile.problem.weights*robust)
    residualized = jac*root[:,None]
    for _,indices in profile.groups:
        design = np.column_stack((np.ones(len(indices)),cap[indices],data_term[indices]))*root[indices,None]
        residualized[indices] -= design@np.linalg.lstsq(design,residualized[indices],rcond=1e-12)[0]
    bounds = profile.model.bounds
    free = np.array([b.lower != b.upper for b in bounds])
    width = np.array([b.upper-b.lower for b in bounds])
    singular = np.linalg.svd(residualized[:,free]*width[free],compute_uv=False)
    rank = int(np.sum(singular > (singular[0]*1e-7 if len(singular) else 0.)))
    contacts = [f"{b.name}:{side}" for x,b in zip(selected["coordinates"],bounds) if b.lower != b.upper
                for side,v in (("lower",b.lower),("upper",b.upper)) if abs(x-v) <= 1e-6*max(1.,abs(v))]
    return dict(nuisance_adjusted_rank=rank,free_shared=int(free.sum()),singular_values=singular.tolist(),
                contacts=contacts,structural_nulls=[b.name for b in bounds if b.lower == b.upper])


def fit_cell(model, delta, deadline, *, anchor=None, sink=None, minimum=24, maximum=48):
    profile = m.Profile(model,delta,deadline)
    attempts, audits = [],[]
    seeds = m.starts(model,anchor)
    def run(index, seed):
        try:
            result = profile.run(seed)
        except (ValueError,RuntimeError,FloatingPointError,OverflowError) as error:
            result = dict(success=False,error=str(error))
        result.update(index=index,initial=np.asarray(seed).tolist())
        attempts.append(result)
        if sink:
            io.write_json_once(sink/f"start-{index:03d}.json",result)
        print(f"start {index}: success={result['success']} objective={result.get('objective')}",flush=True)
    def audit(best):
        try:
            result = m.Direct(profile,best).run() if best else dict(success=False,error="no stationary basin")
        except (ValueError,RuntimeError,FloatingPointError,OverflowError) as error:
            result = dict(success=False,error=str(error))
        audits.append(result)
        return result
    timed_out = False
    escalated = False
    try:
        for index in range(minimum):
            run(index,seeds[index])
        best = best_attempt(attempts)
        audited = audit(best)
        first = best_attempt(attempts[:minimum//2]); second = best_attempt(attempts[minimum//2:])
        improved = first is None or (second is not None and
            first["objective"]-second["objective"] > 1e-6*max(abs(first["objective"]),1e-15))
        escalated = not agreement(model,best,audited) or improved
        if escalated:
            for index in range(minimum,maximum):
                run(index,seeds[index])
            best = best_attempt(attempts)
            audited = audit(best)
        # Training-only numerical recovery, not another form or a bound expansion.
        for index in range(2):
            if agreement(model,best,audited) or not audited.get("success") or not best:
                break
            if audited["objective"] >= best["objective"]:
                break
            run(maximum+index,audited["coordinates"])
            best = best_attempt(attempts)
            audited = audit(best)
    except TimeoutError:
        timed_out = True
    best = best_attempt(attempts)
    audited = audits[-1] if audits else None
    near = [a for a in attempts if best and a.get("success") and
            abs(a["objective"]-best["objective"]) <= 1e-5*max(abs(best["objective"]),1e-15)]
    geo = {}
    if best and time.monotonic() < deadline:
        geo = geometry(profile,best)
    return dict(selected=best,audit=audited,attempts=attempts,audit_history=audits,
                near_equivalent=near,geometry=geo,escalated=escalated,timed_out=timed_out,
                certified=bool(not timed_out and agreement(model,best,audited)),
                bounds=[asdict(b) for b in model.bounds],model=model.name,tau=model.tau,
                heldout_scores_computed=False)
