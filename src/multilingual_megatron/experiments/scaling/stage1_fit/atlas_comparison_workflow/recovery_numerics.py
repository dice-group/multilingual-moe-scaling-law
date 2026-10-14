"""Training-only numerical recovery; equations and final KKT tests are unchanged."""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import least_squares, minimize

from . import model as m, numerics as n

SETTINGS = dict(max_anchors=4, max_rounds=2, direct_max_nfev=8000,
    gradient_root_max_nfev=1000, newton_iterations=80, joint_sqp_maxiter=4000,
    acceptance="unchanged physical KKT <=1e-8 and independent objective/prediction agreement; no worse than lowest known feasible training loss within 1e-6 relative",
    selection="training-only lowest feasible anchors; compatible cap/sign warm starts; never heldout outcomes")


def family(row):
    if row.get("kind") == "direct": return "joint_"+row.get("solver","trf")
    if row.get("kind") in ("start","profile"): return "variable_projection"
    return row.get("kind","unknown")


def seed_points(result):
    """Small deterministic training-only seed pool from a sealed target fit."""
    rows=[result.get("selected"),result.get("audit"),*result.get("audit_history",[]),
          *result.get("attempts",[])]
    ordered=sorted((r for r in rows if r and "coordinates" in r and np.isfinite(r.get("objective",np.nan))),
        key=lambda r:r["objective"])
    points=[]; seen=set()
    for row in ordered:
        key=tuple(row["coordinates"])
        if key not in seen: points.append(row); seen.add(key)
        if len(points)==2: break
    return points


def pair(spec, attempts, packet, minimum):
    """Never hide a better encountered feasible point merely to get agreement."""
    good=sorted((r for r in attempts if r.get("success")),key=lambda r:r["objective"])
    if not good: return None,None
    best=good[0]
    if best["objective"]-minimum > 1e-6*max(abs(minimum),1e-15): return best,None
    for alternative in good[1:]:
        families={family(best),family(alternative)}
        if len(families)==2 and any(f.startswith("joint_") for f in families) and n.agreement(spec,best,alternative,packet):
            return best,alternative
    return best,None


def recover(spec, packet, support, previous, neighbors=(), *, joint_method="SLSQP"):
    began=time.monotonic()
    lo,hi=m.bounds(spec)
    attempts=[]; frontier=[]; seen=set(); minimum=float("inf")

    def add(row, *, trust_status=False):
        nonlocal minimum
        if not row or "coordinates" not in row: return
        x=np.asarray(row["coordinates"],float)
        if x.shape!=lo.shape or not np.isfinite(x).all() or np.any(x<lo) or np.any(x>hi): return
        if m.domain(spec,x,support)[0].min()<0: return
        value,g,_,_=n.objective(spec,x,packet)
        if not np.isfinite(value) or not np.isfinite(g).all(): return
        minimum=min(minimum,value)
        checked=dict(row,objective=value)
        if trust_status:
            checked["kkt"]=n.stationarity(spec,x,g,support)
            checked["success"]=bool(row.get("success") and checked["kkt"]<=1e-8)
        else:
            checked["success"]=False  # a neighbor/failed start is only a seed
        attempts.append(checked)
        key=tuple(x)
        if key not in seen:
            seen.add(key); frontier.append(checked)

    for row in [previous.get("selected"),previous.get("audit"),*previous.get("audit_history",[]),
                *previous.get("attempts",[]),*previous.get("probes",[])]: add(row,trust_status=True)
    for row in neighbors:
        other=m.Spec(**row["spec"])
        if other.form==spec.form and other.parameter_count==spec.parameter_count:
            add(row["point"])
    old_attempt_count=len(attempts)
    checked_anchors=set()
    for round_index in range(SETTINGS["max_rounds"]):
        anchors=[]
        for point in sorted(frontier,key=lambda r:r["objective"]):
            key=tuple(point["coordinates"])
            if key not in checked_anchors:
                anchors.append(point); checked_anchors.add(key)
            if len(anchors)==SETTINGS["max_anchors"]: break
        for anchor in anchors:
            options=dict(spec=spec,packet=packet,support=support)
            polished=n.safe(polish,**options,initial=anchor["coordinates"],index=len(attempts))
            add(polished,trust_status=True)
            candidate=polished if "coordinates" in polished and polished.get("objective",float("inf"))<=anchor["objective"]+1e-14 else anchor
            audited=n.safe(joint_sqp,**options,initial=candidate["coordinates"],index=len(attempts), method=joint_method)
            add(audited,trust_status=True)
            best,audit=pair(spec,attempts,packet,minimum)
            if audit is not None: break
            dogbox=n.safe(n.direct,**options,candidate=candidate,method="dogbox",max_nfev=SETTINGS["direct_max_nfev"])
            add(dogbox,trust_status=True)
            best,audit=pair(spec,attempts,packet,minimum)
            if audit is not None: break
            rooted=n.safe(gradient_root,**options,initial=candidate["coordinates"],index=len(attempts))
            add(rooted,trust_status=True)
            if "coordinates" in rooted:
                check=n.safe(joint_sqp,**options,initial=rooted["coordinates"],index=len(attempts))
                add(check,trust_status=True)
            best,audit=pair(spec,attempts,packet,minimum)
            if audit is not None: break
        best,audit=pair(spec,attempts,packet,minimum)
        if audit is not None: break
    best,audit=pair(spec,attempts,packet,minimum)
    geometry={}
    if best is not None:
        x=np.asarray(best["coordinates"]); jac=n.objective(spec,x,packet)[3]
        singular=np.linalg.svd(jac*np.maximum(abs(x),1.),compute_uv=False)
        geometry=dict(rank=int(np.sum(singular>singular[0]*n.SETTINGS["rank_relative"])),
            coordinate_count=len(x),singular_values=singular.tolist(),point="recovered selected coefficients",
            contacts=[key for i,key in enumerate(m.names(spec)) if abs(x[i]-lo[i])<=1e-6*max(1.,abs(lo[i])) or
                (np.isfinite(hi[i]) and abs(x[i]-hi[i])<=1e-6*max(1.,abs(hi[i])))])
    return dict(spec=spec.payload(),selected=best,audit=audit or {"success":False},attempts=attempts,
        probes=[],audit_history=[],geometry=geometry,escalated=True,certified=audit is not None,
        runtime_seconds=time.monotonic()-began,heldout_scores_computed=False,
        recovery=dict(minimum_feasible_objective=minimum if np.isfinite(minimum) else None,
            old_attempts=old_attempt_count,new_attempts=len(attempts)-old_attempt_count,
            independent_families=[family(best),family(audit)] if audit is not None else [],
            method=SETTINGS))


def gradient_root(spec, initial, packet, support, index=0):
    """Local physical-gradient root correction, accepted only without loss regression."""
    began=time.monotonic()
    lo,hi=m.bounds(spec); x0=np.clip(np.asarray(initial,float),lo,hi)
    original,g,_,_=n.objective(spec,x0,packet)
    active=((x0-lo<=1e-7*np.maximum(1.,abs(lo))) & (g>0)) | (np.isfinite(hi) &
            (hi-x0<=1e-7*np.maximum(1.,abs(hi))) & (g<0))
    free=np.flatnonzero(~active)
    reduced=n.compact_domain(support)[0]
    def unpack(z):
        x=x0.copy(); x[free]=z; return x
    def residual(z):
        x=unpack(z); domain,dj=m.domain(spec,x,reduced)
        if domain.min()<0:
            return np.r_[np.full(len(free),1e4),1e4*np.minimum(domain,0.)]
        return np.r_[n.objective(spec,x,packet)[1][free],1e4*np.minimum(domain,0.)]
    result=least_squares(residual,x0[free],jac="3-point",method="dogbox",bounds=(lo[free],hi[free]),
        x_scale="jac",ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=SETTINGS["gradient_root_max_nfev"])
    x=unpack(result.x)
    value=n.objective(spec,x,packet)[0]
    return n.record(spec,x,packet,support,kind="physical_gradient_root",index=index,
        flag=result.success and value<=original+1e-14,message=result.message,runtime=time.monotonic()-began)


def analytic_hessian(spec, x, packet):
    """Exact full-ATLAS Huber Hessian; objective/gradient stay in numerics.py."""
    if spec.form!="full": raise ValueError("analytic recovery Hessian supports full ATLAS only")
    x=np.asarray(x,float); D,U=m.components(spec,packet)
    sat,dsat=m.saturation(D,U,x[5]); ratio=np.maximum(D/np.where(U>0,U,1.)-1.,0.)
    z=x[5]*ratio; small=z<.05; second=np.empty_like(z)
    a=z[small]
    second[small]=1/3-a/4+a*a/10-a**3/36+a**4/168-a**5/960+a**6/6480
    a=z[~small]
    second[~small]=(2-np.exp(-a)*(a*a+2*a+2))/(a**3)
    d2sat=np.where(D>U,U*ratio**3*second,0.)
    effective,ej=m.exposure(spec,x,packet); dim=len(x); count=len(effective)
    eh=np.zeros((count,dim,dim)); weights=np.r_[1.,x[6:]]
    eh[:,5,5]=d2sat@weights; eh[:,5,6:]=dsat[:,1:]; eh[:,6:,5]=dsat[:,1:]
    logn=np.log(np.asarray(packet['N_total' if spec.parameter_count=='total' else 'N'])/1e8)
    logd=np.log(effective/1e9); cap=np.exp(-x[3]*logn); data=np.exp(-x[4]*logd)
    cj=np.zeros((count,dim)); cj[:,3]=-logn
    normalized=ej/effective[:,None]
    dj=-x[4]*normalized; dj[:,4]-=logd
    dh=-x[4]*(eh/effective[:,None,None]-normalized[:,:,None]*normalized[:,None,:])
    dh[:,4,:]-=normalized; dh[:,:,4]-=normalized
    ph=x[1]*cap[:,None,None]*(cj[:,:,None]*cj[:,None,:])
    ph+=x[2]*data[:,None,None]*(dj[:,:,None]*dj[:,None,:]+dh)
    ph[:,1,:]+=cap[:,None]*cj; ph[:,:,1]+=cap[:,None]*cj
    ph[:,2,:]+=data[:,None]*dj; ph[:,:,2]+=data[:,None]*dj
    prediction,jac=m.predict(spec,x,packet); residual=prediction-np.asarray(packet['y'])
    w=np.asarray(packet['weights'],float); w=w/w.sum(); delta=packet['delta']
    return jac.T@((w*(abs(residual)<delta))[:,None]*jac)+np.einsum('i,ijk->jk',w*np.clip(residual,-delta,delta),ph)


def analytic_polish(spec, initial, packet, support, index=0):
    return polish(spec,initial,packet,support,index,maxiter=200,exact_hessian=True)


def polish(spec, initial, packet, support, index=0, maxiter=80, exact_hessian=False):
    """Projected Newton correction of the original analytic gradient.

    Finite differences form a small Hessian, not a new fitted equation.
    Objective and physical-coordinate KKT are checked on every accepted step.
    """
    began = time.monotonic()
    lo, hi = m.bounds(spec)
    x = np.clip(np.asarray(initial, float), lo, hi)
    for iteration in range(maxiter):
        value, g, _, jac = n.objective(spec, x, packet)
        kkt = n.stationarity(spec, x, g, support)
        if kkt <= 5e-9:
            break
        # Exact bound projection avoids the small inward displacement made by
        # reflective least squares, but only accept a feasible descent step.
        snapped = x.copy()
        lower_active = (x-lo <= 1e-7*np.maximum(1., abs(lo))) & (g > 0)
        upper_active = np.isfinite(hi) & (hi-x <= 1e-7*np.maximum(1., abs(hi))) & (g < 0)
        snapped[lower_active] = lo[lower_active]; snapped[upper_active] = hi[upper_active]
        if m.domain(spec, snapped, support)[0].min() >= 0:
            sv, sg, _, sj = n.objective(spec, snapped, packet)
            if sv <= value+1e-15 and n.stationarity(spec,snapped,sg,support) < kkt:
                x, value, g, jac = snapped, sv, sg, sj
                kkt = n.stationarity(spec,x,g,support)
        free = np.flatnonzero(~(lower_active | upper_active))
        if not len(free): break
        weights = np.asarray(packet["weights"]); weights=weights/weights.sum()
        scale = 1./np.maximum(np.sqrt(np.sum(weights[:,None]*jac[:,free]**2,axis=0)),1e-8)
        scale = np.clip(scale,1e-12,1e6)
        gz = g[free]*scale
        hessian = np.empty((len(free),len(free)))
        if exact_hessian:
            hessian=analytic_hessian(spec,x,packet)[np.ix_(free,free)]*scale[:,None]*scale[None,:]
        for j, coordinate in (() if exact_hessian else enumerate(free)):
            step = 1e-4*scale[j]
            xp, xm = x.copy(), x.copy()
            xp[coordinate] = min(hi[coordinate], x[coordinate]+step)
            xm[coordinate] = max(lo[coordinate], x[coordinate]-step)
            for _ in range(30):
                if (m.domain(spec,xp,support)[0].min() >= 0 and m.domain(spec,xm,support)[0].min() >= 0): break
                xp[coordinate]=(xp[coordinate]+x[coordinate])/2
                xm[coordinate]=(xm[coordinate]+x[coordinate])/2
            distance=(xp[coordinate]-xm[coordinate])/scale[j]
            if distance == 0: hessian[:,j]=0.; continue
            gp=n.objective(spec,xp,packet)[1][free]*scale
            gm=n.objective(spec,xm,packet)[1][free]*scale
            hessian[:,j]=(gp-gm)/distance
        hessian=(hessian+hessian.T)/2
        eigenvalues, vectors=np.linalg.eigh(hessian)
        floor=max(float(np.max(abs(eigenvalues)))*(1e-14 if exact_hessian else 1e-9),1e-15 if exact_hessian else 1e-12)
        dz=-(vectors @ ((vectors.T@gz)/np.maximum(eigenvalues,floor)))
        if np.linalg.norm(dz)>1.: dz=dz/np.linalg.norm(dz)
        accepted=False
        for backtrack in range(35):
            trial=x.copy(); trial[free]=np.clip(x[free]+(0.5**backtrack)*scale*dz,lo[free],hi[free])
            if m.domain(spec,trial,support)[0].min()<0: continue
            tv,tg,_,_=n.objective(spec,trial,packet)
            tk=n.stationarity(spec,trial,tg,support)
            if tv <= value+1e-15 and (tv < value-1e-14 or tk < kkt):
                x=trial; accepted=True; break
        if not accepted: break
    return n.record(spec,x,packet,support,kind="analytic_newton_polish" if exact_hessian else "physical_newton_polish",index=index,
        flag=n.stationarity(spec,x,n.objective(spec,x,packet)[1],support)<=1e-8,
        message=f"original-coordinate KKT polish; {iteration+1} iterations",runtime=time.monotonic()-began)


def joint_sqp(spec, initial, packet, support, index=0, *, method="SLSQP"):
    """Independent full-coordinate SQP with a fixed Jacobian-based scale."""
    began = time.monotonic()
    if method not in ("SLSQP", "L-BFGS-B") or (spec.signed and method != "SLSQP"):
        raise ValueError("unsupported optimizer for native constraint domain")
    lo, hi = m.bounds(spec)
    x0 = np.clip(np.asarray(initial, float), lo, hi)
    _, _, _, jac = n.objective(spec, x0, packet)
    weights = np.asarray(packet["weights"]); weights = weights/weights.sum()
    scales = 1./np.maximum(np.sqrt(np.sum(weights[:, None]*jac**2, axis=0)), 1e-8)
    scales = np.clip(scales, 1e-12, 1e6)
    zlo, zhi = (lo-x0)/scales, (hi-x0)/scales
    reduced = n.compact_domain(support)[0]

    def unpack(z):
        return np.clip(x0+scales*z, lo, hi)

    def objective(z):
        x = unpack(z)
        domain, dj = m.domain(spec, x, reduced)
        if domain.min() < 0:
            bad = np.minimum(domain, 0.)
            return 1e8+1e8*float(bad@bad), 2e8*scales*(dj.T@bad)
        value, gradient, _, _ = n.objective(spec, x, packet)
        return value*1e4, gradient*scales*1e4

    constraints = ()
    if spec.signed:
        constraints = ({"type": "ineq", "fun": lambda z: m.domain(spec, unpack(z), reduced)[0],
                        "jac": lambda z: m.domain(spec, unpack(z), reduced)[1]*scales},)
    options = ({"maxiter": 4000, "ftol": 1e-14} if method == "SLSQP" else
               {"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-5, "maxls": 50, "maxcor": 20})
    result = minimize(objective, np.zeros(len(x0)), method=method, jac=True,
        bounds=list(zip(zlo, zhi)), constraints=constraints, options=options)
    return n.record(spec, unpack(result.x), packet, support, kind="joint_scaled_sqp" if method == "SLSQP" else "joint_scaled_lbfgsb", index=index,
        flag=result.success, message=result.message, runtime=time.monotonic()-began)
