"""Descriptive factorial contrasts on observed support; no fitted scaling law."""
import argparse
import itertools as it
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from result_analysis.capacity_response.grid.analyze import evaluate,METHODS
from result_analysis.capacity_response.tables import load_primary,sha256_file,RELEASE_MANIFEST_SHA256


def main(source,matched,output):
    f=load_primary(source)
    output.mkdir(parents=True,exist_ok=False)
    sets={k:set(g.language) for k,g in f.groupby('K')}
    assert all(sets[a]<=sets[b] for a,b in zip((1,12,50),(12,50,200)))
    offsets=f.assign(offset=f.raw_nll-f.adjusted_loss).groupby('language').offset
    assert (offsets.max()-offsets.min()).max()<2e-12
    native=f.groupby(['ladder','N_act','E','K','D_total'],as_index=False).raw_nll.mean()
    native.to_csv(output/'native_observed.csv',index=False)
    observed=[]
    for cohort, selected in [('native',f),('english',f[f.language.isin(sets[1])]),
            ('common12',f[f.language.isin(sets[12]) & (f.K>=12)]),
            ('common50',f[f.language.isin(sets[50]) & (f.K>=50)])]:
        means=selected.groupby(['ladder','N_act','K','D_total']).apply(
            lambda g: g.groupby('E').raw_nll.mean(),include_groups=False)
        for a,b in it.combinations((1,4,16,32),2):
            table=(means[a]-means[b]).rename('gain').reset_index()
            table['cohort'],table['low_E'],table['high_E']=cohort,a,b
            table['F_EF']=6*table.N_act.astype(float)*table.D_total.astype(float)/1e18
            observed.append(table)
    pd.concat(observed).to_csv(output/'observed_expert_gains.csv',index=False)
    rows,windows=[],[]
    for regime in ('tokens','compute'):
        for scope in ('ABCD','A','B','C','D'):
            sub=f[f.ladder.isin(list(scope))].copy()
            sub['axis']=sub.D_total.astype(float)
            if regime=='compute':
                sub['axis']*=6*sub.N_act.astype(float)/1e18
            bounds=sub.groupby(['ladder','E','K']).axis.agg(['min','max'])
            lo,hi=bounds['min'].max(),bounds['max'].min()
            windows.append(dict(regime=regime,analysis=scope,lo=lo,hi=hi))
            for (lang,n,e,k),g in sub.groupby(['language','ladder','E','K']):
                curve=pd.DataFrame({'D_t':g.axis,'adjusted_loss':g.raw_nll}).sort_values('D_t')
                for method in METHODS:
                    for stage,loss in evaluate(curve,lo,hi,method).items():
                        rows.append((regime,scope,method,lang,n,e,k,stage,loss))
    cols=['regime','analysis','method','language','ladder','E','K','summary','loss']
    losses=pd.DataFrame(rows,columns=cols)
    prior=pd.read_csv(matched/'language_losses.csv').assign(regime='target_exposure')
    losses=pd.concat([losses,prior[cols]],ignore_index=True)
    losses.to_csv(output/'language_losses.csv.gz',index=False)
    pd.DataFrame(windows).to_csv(output/'budget_support.csv',index=False)
    pd.read_csv(matched/'support_windows.csv').to_csv(output/'target_exposure_support.csv',index=False)
    print('Loss evaluation complete',len(losses),flush=True)
    results=[]
    groupcols=['regime','analysis','method','summary']
    for keys,g in losses.groupby(groupcols,sort=False):
        cohorts=[('incumbent',g)] if keys[0]=='target_exposure' else [
            ('native',g),('english',g[g.language.isin(sets[1])]),
            ('common12',g[g.language.isin(sets[12]) & (g.K>=12)]),
            ('common50',g[g.language.isin(sets[50]) & (g.K>=50)])]
        for cohort,sub in cohorts:
            curves={key:s.set_index('language').loss for key,s in sub.groupby(['ladder','E','K'])}
            ns,es,ks=sorted(sub.ladder.unique()),sorted(sub.E.unique()),sorted(sub.K.unique())
            def add(kind,v,n0='',n1='',e0='',e1='',k0='',k1=''):
                v=v.dropna().to_numpy(float)
                if not len(v): return
                loo=(v.sum()-v)/(len(v)-1) if len(v)>1 else v
                results.append((*keys,cohort,kind,n0,n1,e0,e1,k0,k1,len(v),v.mean(),np.median(v),
                    np.quantile(v,.1),np.quantile(v,.9),int((v>0).sum()),loo.min(),loo.max()))
            def eg(n,a,b,k): return curves[n,a,k]-curves[n,b,k]
            for n,k,(a,b) in it.product(ns,ks,it.combinations(es,2)):
                add('E_gain',eg(n,a,b,k),n0=n,e0=a,e1=b,k0=k)
            for e,k,(a,b) in it.product(es,ks,it.combinations(ns,2)):
                add('N_gain',curves[a,e,k]-curves[b,e,k],n0=a,n1=b,e0=e,k0=k)
            for k,(a,b),(c,d) in it.product(ks,it.combinations(ns,2),it.combinations(es,2)):
                add('NE',eg(b,c,d,k)-eg(a,c,d,k),a,b,c,d,k)
            # Native K means have different evaluation populations: no K subtraction.
            if cohort=='native': continue
            for a,b in it.combinations(ks,2):
                for n,e in it.product(ns,es):
                    add('K_response',curves[n,e,b]-curves[n,e,a],n0=n,e0=e,k0=a,k1=b)
                for n,(c,d) in it.product(ns,it.combinations(es,2)):
                    add('EK',eg(n,c,d,b)-eg(n,c,d,a),n0=n,e0=c,e1=d,k0=a,k1=b)
                for e,(c,d) in it.product(es,it.combinations(ns,2)):
                    add('NK',(curves[c,e,b]-curves[d,e,b])-(curves[c,e,a]-curves[d,e,a]),c,d,e,'',a,b)
                for (c,d),(h,j) in it.product(it.combinations(ns,2),it.combinations(es,2)):
                    add('NEK',(eg(d,h,j,b)-eg(d,h,j,a))-(eg(c,h,j,b)-eg(c,h,j,a)),c,d,h,j,a,b)
    names=groupcols+['cohort','contrast','low_N','high_N','low_E','high_E','low_K','high_K',
        'languages','mean','median','q10','q90','positive_languages','loo_min','loo_max']
    c=pd.DataFrame(results,columns=names)
    c.to_csv(output/'contrasts.csv.gz',index=False)
    ident=[x for x in names[:12] if x not in ('method','summary')]
    s=c.pivot(index=ident+['summary'],columns='method',values='mean').reset_index()
    s['method_spread']=s[list(METHODS)].max(axis=1)-s[list(METHODS)].min(axis=1)
    s['sign_disagreement']=(s[list(METHODS)].min(axis=1)<0)&(s[list(METHODS)].max(axis=1)>0)
    s.to_csv(output/'interpolation_sensitivity.csv.gz',index=False)
    t=c[c.method=='linear_log'].pivot(index=ident,columns='summary',values='mean').reset_index()
    t['late_minus_early']=t.late-t.early
    t.to_csv(output/'budget_changes.csv',index=False)
    p=native.pivot(index=['ladder','K','D_total'],columns='E',values='raw_nll')
    closure=float(((p[1]-p[32])-((p[1]-p[4])+(p[4]-p[16])+(p[16]-p[32]))).abs().max())
    assert closure<1e-12
    # Independent endpoint reconstruction checks raw aggregation and units.
    endpoint=c.query("regime=='tokens' and analysis=='ABCD' and cohort=='native' and contrast=='E_gain' and method=='linear_log' and summary=='early'")
    errors=[]
    first=native[native.D_total==native.D_total.min()].set_index(['ladder','E','K']).raw_nll
    for r in endpoint.itertuples():
        direct=first[r.low_N,int(r.low_E),int(r.low_K)]-first[r.low_N,int(r.high_E),int(r.low_K)]
        errors.append(abs(r.mean-direct))
    assert max(errors)<1e-12 and len(errors)==96
    # Replay every reused linear-log matched-exposure stage directly from sealed data.
    support=pd.read_csv(matched/'support_windows.csv').set_index(['analysis','language'])
    raw_curves={key:g.sort_values('D_t') for key,g in f.groupby(['language','ladder','E','K'])}
    replay=prior[(prior.method=='linear_log') & prior.summary.isin(['early','middle','late'])]
    replay_error=0.
    for r in replay.itertuples():
        window=support.loc[r.analysis,r.language]
        fraction={'early':0.,'middle':.5,'late':1.}[r.summary]
        x=(1-fraction)*np.log(window.min_D_t)+fraction*np.log(window.max_D_t)
        raw=raw_curves[r.language,r.ladder,r.E,r.K]
        direct=np.interp(x,np.log(raw.D_t.to_numpy(float)),raw.adjusted_loss.to_numpy(float))
        replay_error=max(replay_error,abs(direct-r.loss))
    assert replay_error<1e-12
    receipt=dict(source=str(source),release_manifest_sha256=RELEASE_MANIFEST_SHA256,
        observations=len(f),trajectories=f.trajectory_id.nunique(),readouts=f.readout_id.nunique(),
        loss_rows=len(losses),contrast_rows=len(c),observed_chain_closure=closure,
        native_endpoint_checks=len(errors),native_endpoint_max_error=max(errors),
        matched_replay_checks=len(replay),matched_replay_max_error=replay_error,
        inputs={str(matched/x):sha256_file(matched/x) for x in ['language_losses.csv','support_windows.csv','verification.json']},
        builder_sha256=sha256_file(Path(__file__)),
        limitations='One primary seed. Language quantiles and leave-one-out ranges are descriptive, not training confidence intervals. Matched exposure reuses previously audited curves. No extrapolation or fitting.')
    (output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','matched','output'): p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    main(a.source,a.matched,a.output)
