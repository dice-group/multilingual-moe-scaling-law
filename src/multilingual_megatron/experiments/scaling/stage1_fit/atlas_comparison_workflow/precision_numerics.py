"""Extended-precision diagnostics and coordinate recovery for unchanged full ATLAS.

Existing float64 model.py/numerics.py remain the final acceptance authority.
Binary64 inputs are lifted exactly, never reconstructed from rounded decimals.
"""
from __future__ import annotations

import time
import numpy as np
import mpmath as mp
from scipy.optimize import minimize, LinearConstraint

from . import model as m, numerics as n

LD=np.longdouble
SETTINGS=dict(newton_iterations=300,sqp_iterations=3000,svd_relative_floor=1e-10,
    precision='numpy longdouble, independently checked with 60/90-decimal mpmath',
    final_checks='unchanged original float64 physical KKT, objective and prediction agreement; higher-precision diagnostics are additional checks',
    selection='training-only; original bounds and all covariate-only domain inequalities preserved')


def lift(value):
    return np.asarray(value,dtype=float).astype(LD)


def saturation(D,U,rate):
    ratio=np.maximum(D/np.where(U>0,U,LD(1))-1,0); z=rate*ratio
    small=z<LD('0.0001'); f=np.empty_like(z); df=np.empty_like(z); ddf=np.empty_like(z)
    a=z[small]
    f[small]=1-a/2+a*a/6-a**3/24+a**4/120
    df[small]=-LD('.5')+a/3-a*a/8+a**3/30
    ddf[small]=LD(1)/3-a/4+a*a/10
    a=z[~small]
    f[~small]=-np.expm1(-a)/a
    df[~small]=((a+1)*np.exp(-a)-1)/(a*a)
    ddf[~small]=(2-np.exp(-a)*(a*a+2*a+2))/(a**3)
    transition=(z>=LD('0.0001')) & (z<LD('.05'))
    a=z[transition]
    ddf[transition]=LD(1)/3-a/4+a*a/10-a**3/36+a**4/168-a**5/960+a**6/6480
    return (np.where(D<=U,D,U*(1+ratio*f)),np.where(D<=U,0,U*ratio**2*df),
            np.where(D<=U,0,U*ratio**3*ddf))


class Evaluator:
    def __init__(self,spec,packet):
        if spec.form!='full': raise ValueError('precision recovery supports full ATLAS only')
        self.spec=spec; self.D=lift(packet['D']); self.U=lift(packet['U'])
        self.logn=np.log(lift(packet['N_total' if spec.parameter_count=='total' else 'N'])/LD(1e8))
        self.minimum=LD(float(packet['training_minimum']))
        self.y=lift(packet['y']) if 'y' in packet else None
        self.weights=lift(packet['weights']) if 'weights' in packet else None
        if self.weights is not None: self.weights=self.weights/self.weights.sum()
        self.delta=LD(float(packet['delta'])) if 'delta' in packet else None

    def exposure(self,x):
        x=np.asarray(x,dtype=LD); sat,dsat,d2sat=saturation(self.D,self.U,x[5])
        weights=np.r_[LD(1),x[6:]]; eff=np.sum(sat*weights,axis=1,dtype=LD)
        ej=np.zeros((len(eff),10),dtype=LD); ej[:,5]=np.sum(dsat*weights,axis=1,dtype=LD); ej[:,6:]=sat[:,1:]
        return eff,ej,dsat,d2sat

    def domain(self,x):
        eff,jac,_,_=self.exposure(x)
        return eff/self.D[:,0]-LD(m.DOMAIN_MARGIN),jac/self.D[:,0,None]

    def evaluate(self,x,*,hessian=False):
        x=np.asarray(x,dtype=LD); eff,ej,dsat,d2sat=self.exposure(x)
        if np.any(eff<LD(m.DOMAIN_MARGIN)*self.D[:,0]): raise ValueError('outside original positive-exposure domain')
        logd=np.log(eff/LD(1e9)); cap=np.exp(-x[3]*self.logn); data=np.exp(-x[4]*logd)
        cj=np.zeros((len(eff),10),dtype=LD); cj[:,3]=-self.logn
        normalized=ej/eff[:,None]; dj=-x[4]*normalized; dj[:,4]-=logd
        pred=x[0]*self.minimum+x[1]*cap+x[2]*data
        jac=x[1]*cap[:,None]*cj+x[2]*data[:,None]*dj
        jac[:,0]=self.minimum; jac[:,1]=cap; jac[:,2]=data
        if self.y is None: return pred,jac
        residual=pred-self.y; psi=np.clip(residual,-self.delta,self.delta)
        loss=np.where(abs(residual)<=self.delta,residual**2/2,self.delta*(abs(residual)-self.delta/2))
        value=np.sum(self.weights*loss,dtype=LD); gradient=np.sum(jac*(self.weights*psi)[:,None],axis=0,dtype=LD)
        if not hessian: return value,gradient,pred,jac
        eh=np.zeros((len(eff),10,10),dtype=LD); weights=np.r_[LD(1),x[6:]]
        eh[:,5,5]=np.sum(d2sat*weights,axis=1,dtype=LD); eh[:,5,6:]=dsat[:,1:]; eh[:,6:,5]=dsat[:,1:]
        dh=-x[4]*(eh/eff[:,None,None]-normalized[:,:,None]*normalized[:,None,:])
        dh[:,4,:]-=normalized; dh[:,:,4]-=normalized
        ph=x[1]*cap[:,None,None]*(cj[:,:,None]*cj[:,None,:])
        ph+=x[2]*data[:,None,None]*(dj[:,:,None]*dj[:,None,:]+dh)
        ph[:,1,:]+=cap[:,None]*cj; ph[:,:,1]+=cap[:,None]*cj
        ph[:,2,:]+=data[:,None]*dj; ph[:,:,2]+=data[:,None]*dj
        h=np.sum(jac[:,:,None]*jac[:,None,:]*(self.weights*(abs(residual)<self.delta))[:,None,None],axis=0,dtype=LD)
        h+=np.sum(ph*(self.weights*psi)[:,None,None],axis=0,dtype=LD)
        return value,gradient,pred,jac,h


def mp_objective(spec,x,packet,dps=60):
    """Independent scalar arbitrary-precision equation and gradient evaluation."""
    if spec.form!='full': raise ValueError('full ATLAS only')
    with mp.workdps(dps):
        X=[mp.mpf(float(v)) for v in x]; w=[mp.mpf(float(v)) for v in packet['weights']]; sw=mp.fsum(w)
        floor=mp.mpf(float(packet['training_minimum'])); delta=mp.mpf(float(packet['delta'])); loss=[]; grad=[[] for _ in X]; predictions=[]
        Ns=packet['N_total' if spec.parameter_count=='total' else 'N']
        for i,(ds,us,N,y) in enumerate(zip(packet['D'],packet['U'],Ns,packet['y'])):
            sat=[]; dsat=[]
            for dv,uv in zip(ds,us):
                D,U=mp.mpf(float(dv)),mp.mpf(float(uv))
                if D<=U: sat.append(D); dsat.append(mp.mpf(0)); continue
                ratio=D/U-1; z=X[5]*ratio
                if z<mp.mpf('0.0001'):
                    f=1-z/2+z*z/6-z**3/24+z**4/120; df=-mp.mpf('.5')+z/3-z*z/8+z**3/30
                else: f=-mp.expm1(-z)/z; df=((z+1)*mp.exp(-z)-1)/(z*z)
                sat.append(U*(1+ratio*f)); dsat.append(U*ratio*ratio*df)
            tau=[mp.mpf(1),*X[6:]]; eff=mp.fsum(a*b for a,b in zip(sat,tau))
            if eff<mp.mpf(m.DOMAIN_MARGIN)*mp.mpf(float(ds[0])): raise ValueError('mpmath domain violation')
            ln=mp.log(mp.mpf(float(N))/10**8); ld=mp.log(eff/10**9); cap=mp.exp(-X[3]*ln); data=mp.exp(-X[4]*ld)
            prediction=X[0]*floor+X[1]*cap+X[2]*data; residual=prediction-mp.mpf(float(y)); psi=max(-delta,min(delta,residual))
            jac=[floor,cap,data,-X[1]*cap*ln,-X[2]*data*ld,
                 -X[2]*X[4]*data*mp.fsum(a*b for a,b in zip(dsat,tau))/eff,
                 *[-X[2]*X[4]*data*a/eff for a in sat[1:]]]
            loss.append(w[i]/sw*(residual*residual/2 if abs(residual)<=delta else delta*(abs(residual)-delta/2)))
            for j,g in enumerate(jac): grad[j].append(w[i]/sw*psi*g)
            predictions.append(float(prediction))
        return float(mp.fsum(loss)),np.array([float(mp.fsum(g)) for g in grad]),np.array(predictions)


def precision_diagnostic(spec,x,packet,support):
    v,g,p,j=n.objective(spec,x,packet); extended=Evaluator(spec,packet).evaluate(x)
    a=mp_objective(spec,x,packet,60); b=mp_objective(spec,x,packet,90)
    domain=Evaluator(spec,support).domain(x)[0]
    return dict(float64_objective=v,longdouble_objective=float(extended[0]),mp60_objective=a[0],mp90_objective=b[0],
        float64_kkt=n.stationarity(spec,np.asarray(x),g,support),
        longdouble_gradient_kkt=n.stationarity(spec,np.asarray(x),np.asarray(extended[1],float),support),
        mp60_gradient_kkt=n.stationarity(spec,np.asarray(x),a[1],support),
        mp90_gradient_kkt=n.stationarity(spec,np.asarray(x),b[1],support),
        max_float64_mp_gradient_difference=float(np.max(abs(g-b[1]))),
        max_longdouble_mp_gradient_difference=float(np.max(abs(np.asarray(extended[1],float)-b[1]))),
        max_mp60_mp90_gradient_difference=float(np.max(abs(a[1]-b[1]))),
        max_float64_mp_prediction_difference=float(np.max(abs(p-b[2]))),
        minimum_longdouble_domain=float(domain.min()),heldout_scores_computed=False)


def whitening(jac,weights,free):
    root=np.sqrt(np.asarray(weights,float)); design=np.asarray(jac[:,free],float)*root[:,None]
    norms=np.maximum(np.linalg.norm(design,axis=0),1e-12); _,singular,vt=np.linalg.svd(design/norms,full_matrices=False)
    singular=np.maximum(singular,singular[0]*SETTINGS['svd_relative_floor'])
    return (vt.T/singular)/norms[:,None]


def newton(spec,initial,packet,support,index=0):
    began=time.monotonic(); lo,hi=(lift(v) for v in m.bounds(spec)); x=np.clip(lift(initial),lo,hi)
    train=Evaluator(spec,packet); domain=Evaluator(spec,support)
    for iteration in range(SETTINGS['newton_iterations']):
        value,g,p,jac,h=train.evaluate(x,hessian=True)
        original=n.safe(n.record,spec=spec,x=np.asarray(x,float),packet=packet,support=support,
            kind='precision_newton',index=index,flag=True,message='iteration check',runtime=0.)
        if original.get('success') and n.stationarity(spec,np.asarray(x,float),np.asarray(g,float),support)<=5e-9: break
        active_lo=(x-lo<=LD('1e-7')*np.maximum(1,abs(lo))) & (g>0)
        active_hi=np.isfinite(hi) & (hi-x<=LD('1e-7')*np.maximum(1,abs(hi))) & (g<0)
        snapped=x.copy(); snapped[active_lo]=lo[active_lo]; snapped[active_hi]=hi[active_hi]
        if domain.domain(snapped)[0].min()>=0 and train.evaluate(snapped)[0]<=value+LD('1e-16'):
            x=snapped; value,g,p,jac,h=train.evaluate(x,hessian=True)
        free=np.flatnonzero(~(active_lo|active_hi))
        if not len(free): break
        transform=whitening(jac,train.weights,free).astype(LD)
        hz=np.asarray(transform.T@h[np.ix_(free,free)]@transform,float); hz=(hz+hz.T)/2
        gz=np.asarray(transform.T@g[free],float); eig,vec=np.linalg.eigh(hz)
        floor=max(np.max(abs(eig))*1e-12,1e-12); dz=-(vec@((vec.T@gz)/np.maximum(eig,floor)))
        if np.linalg.norm(dz)>1: dz=dz/np.linalg.norm(dz)
        step=transform@dz.astype(LD); alpha=LD(1)
        for k,coordinate in enumerate(free):
            if step[k]>0: alpha=min(alpha,(hi[coordinate]-x[coordinate])/step[k])
            elif step[k]<0: alpha=min(alpha,(lo[coordinate]-x[coordinate])/step[k])
        accepted=False
        oldk=n.stationarity(spec,np.asarray(x,float),np.asarray(g,float),support)
        for backtrack in range(45):
            trial=x.copy(); trial[free]=np.clip(x[free]+alpha*LD(.5)**backtrack*step,lo[free],hi[free])
            if domain.domain(trial)[0].min()<0: continue
            tv,tg,_,_=train.evaluate(trial)
            tk=n.stationarity(spec,np.asarray(trial,float),np.asarray(tg,float),support)
            if tv<=value+LD('1e-19') and (tv<value-LD('1e-19') or tk<oldk): x=trial; accepted=True; break
        if not accepted: break
    return n.record(spec,np.asarray(x,float),packet,support,kind='precision_newton',index=index,
        flag=True,message=f'extended precision, rotated coordinates; {iteration+1} iterations',runtime=time.monotonic()-began)


def joint_sqp(spec,initial,packet,support,index=0):
    began=time.monotonic(); lo,hi=m.bounds(spec); x0=lift(np.clip(np.asarray(initial,float),lo,hi))
    train=Evaluator(spec,packet); domain=Evaluator(spec,n.compact_domain(support)[0]); _,_,_,jac=train.evaluate(x0)
    transform=whitening(jac,train.weights,np.arange(len(x0))).astype(LD)
    row_norm=np.maximum(np.linalg.norm(np.asarray(transform,float),axis=1),1e-12)
    def unpack(z): return np.clip(x0+transform@np.asarray(z,dtype=LD),lift(lo),lift(hi))
    def objective(z):
        x=unpack(z); slack,dj=domain.domain(x)
        if slack.min()<0:
            bad=np.minimum(slack,0); return 1e8+float(bad@bad)*1e8,np.asarray(2e8*transform.T@dj.T@bad,float)
        value,g,_,_=train.evaluate(x); return float(value*1e4),np.asarray(transform.T@g*1e4,float)
    constraints=[LinearConstraint(np.asarray(transform,float)/row_norm[:,None],(lo-np.asarray(x0,float))/row_norm,(hi-np.asarray(x0,float))/row_norm),
        dict(type='ineq',fun=lambda z:np.asarray(domain.domain(unpack(z))[0],float),
            jac=lambda z:np.asarray(domain.domain(unpack(z))[1]@transform,float))]
    result=minimize(objective,np.zeros(len(x0)),jac=True,method='SLSQP',constraints=constraints,
        options=dict(maxiter=SETTINGS['sqp_iterations'],ftol=1e-14))
    return n.record(spec,np.asarray(unpack(result.x),float),packet,support,kind='joint_precision_sqp',index=index,
        flag=result.success,message=result.message,runtime=time.monotonic()-began)


def recover_case(spec,packet,previous):
    """Bounded representative-case runner, sharing the original consolidation gate."""
    from . import recovery_merge
    began=time.monotonic(); minimum=previous['recovery']['minimum_feasible_objective']
    near=[a for a in previous['attempts'] if a['objective']<=minimum+1e-6*abs(minimum)]
    lowest=min(near,key=lambda a:a['objective'])
    stationary=min(near,key=lambda a:a.get('kkt') if a.get('kkt') is not None else float('inf'))
    anchors=[]; seen=set()
    for row in (stationary,lowest,previous.get('selected')):
        if row and tuple(row['coordinates']) not in seen: anchors.append(row); seen.add(tuple(row['coordinates']))
    diagnostic=precision_diagnostic(spec,stationary['coordinates'],packet['train'],packet['support'])
    additions=[]
    for index,anchor in enumerate(anchors):
        options=dict(spec=spec,packet=packet['train'],support=packet['support'],index=index)
        polished=n.safe(newton,**options,initial=anchor['coordinates']); additions.append(polished)
        candidate=polished if 'coordinates' in polished and polished['objective']<=anchor['objective']+1e-12 else anchor
        audited=n.safe(joint_sqp,**options,initial=candidate['coordinates']); additions.append(audited)
        if 'coordinates' in audited:
            additions.append(n.safe(newton,**options,initial=audited['coordinates']))
        value=recovery_merge.consolidate(spec,packet,previous,additions)
        if value['certified']: break
    value=recovery_merge.consolidate(spec,packet,previous,additions)
    final_checks=[]
    if value['certified']:
        for point in (value['selected'],value['audit']):
            final_checks.append(precision_diagnostic(spec,point['coordinates'],packet['train'],packet['support']))
        if any(check['mp90_gradient_kkt']>1e-8 or check['minimum_longdouble_domain']<0 for check in final_checks):
            value['certified']=False
            # Preserve attempts but prevent a later consolidation from ignoring
            # a contradictory higher-precision gradient check.
            for row in additions:
                if 'coordinates' in row:
                    _,g,_,_=Evaluator(spec,packet['train']).evaluate(row['coordinates'])
                    row['success']=bool(row.get('success') and n.stationarity(spec,np.asarray(row['coordinates']),np.asarray(g,float),packet['support'])<=1e-8)
    return dict(results=[dict(method=row.get('kind','error'),result=row) for row in additions],
        certified=value['certified'],diagnostic_before=diagnostic,diagnostic_after=final_checks,
        selected=value['selected'],audit=value['audit'],minimum_feasible_objective=value['recovery']['minimum_feasible_objective'],
        runtime_seconds=time.monotonic()-began,heldout_scores_computed=False)
