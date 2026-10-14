"""Independent NumPy-only equation and score replay. No fitter imports."""
import numpy as np


def saturate(D,U,rate):
    D,U,rate = np.broadcast_arrays(np.asarray(D,float),np.asarray(U,float),np.asarray(rate,float))
    if np.any((D>0)&(U<=0)) or np.any(rate<0):
        raise ValueError("invalid saturation domain")
    result = D.copy()
    active = (D>U)&(rate>0)
    excess = D[active]/U[active]-1.
    result[active] = U[active]*(1.-np.expm1(-rate[active]*excess)/rate[active])
    return result


def candidate(name,x,rows,sources):
    x = np.asarray(x)
    n,e,k = rows.N_act.to_numpy()/1e8,rows.E.to_numpy(),rows.K.to_numpy()
    h = np.log(e)/np.log(32.)
    hinge = np.maximum(np.log(k/12.),0.)/np.log(200./12.)
    alpha = x[0]*(1-h)+x[1]*h+x[10]*hinge
    beta = x[2]*(1-h)+x[3]*h
    phi,psi = x[6]*(1-h)+x[7]*h,x[8]*(1-h)+x[9]*h
    target = rows.D_t.to_numpy()
    cap = e**x[4]*k**phi*n**(-alpha)*(1+rows.D_other.to_numpy()/(
        target+rows.D_family.to_numpy()+rows.D_script.to_numpy()))**x[13]
    if name.startswith("SOURCE"):
        size = sources.P if name == "SOURCE-RP" else sources.N
        sat = saturate(sources.D,sources.U,(x[15]*(size/1e8)**x[16])[:,None])
        r,t = sources.readout,sources.target
        effective = sat[r,t]+x[11]*(sat@sources.family.T)[r,t]+x[12]*(sat@sources.script.T)[r,t]
    else:
        effective = saturate(target,rows.U_t.to_numpy(),x[15]*n**x[16])
        effective *= (1+rows.D_family.to_numpy()/target)**x[11]*(1+rows.D_script.to_numpy()/target)**x[12]
    return np.asarray(cap),e**x[5]*k**psi*(effective/1e9)**(-beta)*n**x[14]


def native(spec,x,packet):
    form = spec["form"]
    x = np.asarray(x,float)
    D,U = np.asarray(packet["D"]),np.asarray(packet["U"])
    n = np.asarray(packet["N_total"] if spec["parameter_count"] == "total" else packet["N"])/1e8
    e,k = np.asarray(packet["E"]),np.asarray(packet["K"])
    if form in ("full","other","moe"):
        offset = 9 if form == "moe" else 5
        if form == "other":
            D,U = np.column_stack((D[:,0],D[:,1:].sum(axis=1))),np.column_stack((U[:,0],U[:,1:].sum(axis=1)))
        useful = saturate(D,U,x[offset])@np.r_[1.,x[offset+1:]]
    else:
        useful = D[:,0] if form in ("target","capacity") else D.sum(axis=1)
    if np.any(useful/np.asarray(packet["D"])[:,0] < 1e-10):
        raise ValueError("effective exposure outside declared domain")
    if form in ("joint","moe"):
        bridge = 1/(1/2.0732-1/290.4521)
        transformed = 1/(1/(e-1+bridge)+1/290.4521)
        transformed32 = 1/(1/(32-1+bridge)+1/290.4521)
        g = (np.log(transformed)-np.log(2.0732))*np.log(32.)/(np.log(transformed32)-np.log(2.0732))
        h = g/np.log(32.)
        logcap = x[7]*g-(x[3]*(1-h)+x[4]*h)*np.log(n)
        logterm = x[8]*g-(x[5]*(1-h)+x[6]*h)*np.log(useful/1e9)
    else:
        logcap,logterm = -x[3]*np.log(n),-x[4]*np.log(useful/1e9)
    if form == "capacity":
        logcap,logterm = logcap+x[5]*np.log(k),logterm+x[6]*np.log(k)
    cap,term = np.exp(logcap),np.exp(logterm)
    prediction = x[0]*packet["training_minimum"]+x[1]*cap+x[2]*term
    if form == "he":
        prediction *= np.asarray(packet["family_share"])**(-x[5])
    return prediction


def score(observed,prediction,trajectory_ids,endpoint=False):
    observed,prediction = np.asarray(observed),np.asarray(prediction)
    if observed.shape != prediction.shape or not len(observed) or not np.isfinite(prediction).all():
        raise ValueError("incomplete independent score replay")
    error = prediction-observed
    if endpoint:
        weights = np.full(len(error),1/len(error))
    else:
        ids,counts = np.unique(trajectory_ids,return_counts=True)
        weights = np.array([1/(len(ids)*counts[np.flatnonzero(ids==i)[0]]) for i in trajectory_ids])
    return dict(rmse=float(np.sqrt(np.sum(weights*error*error))),mae=float(np.sum(weights*abs(error))),
                bias=float(np.sum(weights*error)))


def shared_equation(record,rows):
    p=record["shared_parameters"]
    n,e,k=rows.N_act.to_numpy()/1e8,rows.E.to_numpy(),rows.K.to_numpy()
    loge=np.log(e)
    hinge=np.maximum(np.log(k/12.),0.)/np.log(200./12.)
    alpha=p["alpha_0"]+p["alpha_1"]*loge+p.get("gamma_n",0.)*hinge
    beta=p["beta_0"]+p["beta_1"]*loge+p.get("gamma_d",0.)*hinge
    phi=p["phi_0"]+p["phi_1"]*loge
    psi=p["psi_0"]+p["psi_1"]*loge
    dt,df,ds,do=(rows[c].to_numpy() for c in ("D_t","D_family","D_script","D_other"))
    cap=e**p["delta"]*k**phi*n**(-alpha)*(1+do/(dt+df+ds))**p.get("gamma_other",0.)
    effective=dt*(1+df/dt)**p.get("eta_family",0.)*(1+ds/dt)**p.get("eta_script",0.)
    term=e**p["omega"]*k**psi*(effective/1e9)**(-beta)*n**record["zeta"]
    nuisance={r["language"]:r for r in record["language_coefficients"]}
    c=np.array([[nuisance[l][k] for k in ("L_inf","A","B")] for l in rows.language])
    return c[:,0]+c[:,1]*cap+c[:,2]*term
