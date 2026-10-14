"""Benchmark cohorts and observed token exposure; writes only local analysis artifacts."""
from pathlib import Path
import argparse
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from result_analysis.capacity_response.tables import load_primary, load_release_table

ROOT=Path(__file__).resolve().parents[3]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def macro(g,col): return g.groupby('task')[col].mean().mean()

def main(root: Path, tables: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((tables/'benchmark_artifact_manifest.json').read_text())
    mapping=json.loads((tables/'benchmark_training_membership_map.json').read_text())
    planpath=Path(mapping['sampling_plan']['path'])
    assert sha(planpath)==mapping['sampling_plan']['sha256']
    plan=json.loads(planpath.read_text())
    sets={k:set(plan['tiers'][f'K{k}']['languages']) for k in [12,50,200]}
    sets[1]={'eng_Latn'}
    assert sets[1]<sets[12]<sets[50]<sets[200]
    primary=load_primary(root)
    ep=primary.loc[primary.ladder.eq('D')].sort_values('D_total').groupby(['E','K','language']).tail(1)
    assert ep.groupby(['K','language']).D_t.nunique().eq(1).all()
    assert ep.D_total.nunique()==ep.N_act.nunique()==1
    for k,s in sets.items(): assert set(ep.loc[ep.K.eq(k),'language'])==s
    unique_ep=ep.loc[ep.E.eq(1)]
    np.testing.assert_allclose(unique_ep.groupby('K').D_t.sum(),ep.D_total.iloc[0],rtol=0,atol=0)
    readouts=load_release_table(root,'readouts',['readout_id','checkpoint_iteration_path'])
    ep=ep.merge(readouts,on='readout_id',validate='many_to_one')
    records=[]
    for pair in manifest['metric_score_pairs']:
        path=Path(pair['metrics_path'])
        assert sha(path)==pair['metrics_file_sha256']
        payload=json.loads(path.read_text())
        assert payload['status']=='complete'
        e,k=pair['E'],pair['K']
        assert ep.loc[ep.E.eq(e)&ep.K.eq(k),'checkpoint_iteration_path'].unique().tolist()==[payload['checkpoint']['iteration_path']]
        for task in payload['benchmarks']:
            for lang in task['per_language']:
                records.append(dict(E=e,K=k,task=task['benchmark'],language=lang['language'],documents=lang['documents'],normalized_accuracy=lang['acc_norm'],raw_accuracy=lang['acc']))
    d=pd.DataFrame(records)
    assert len(d)==2376 and not d.duplicated(['E','K','task','language']).any()
    def corpus(task,lang):
        if task=='belebele_cloze': return 'cmn_Hani' if lang=='zho_Hans' else None if lang=='zho_Hant' else lang
        return mapping['short_code_to_corpus'].get(lang)
    d['corpus_language']=[corpus(t,l) for t,l in zip(d.task,d.language)]
    d['training_status']=['mapping_unresolved' if l is None else 'trained' if l in sets[k] else 'untrained' for l,k in zip(d.corpus_language,d.K)]
    d['first_training_K']=[next((k for k in [1,12,50,200] if l in sets[k]),None) for l in d.corpus_language]
    cols=['E','K','language','D_t','adjusted_loss','raw_nll','readout_id']
    d=d.merge(ep[cols].rename(columns={'language':'corpus_language'}),on=['E','K','corpus_language'],how='left',validate='many_to_one')
    assert d.loc[d.training_status.eq('trained'),'D_t'].notna().all()
    d.loc[d.training_status.eq('untrained'),'D_t']=0.0
    # Unknown mappings stay missing; no fictitious zero exposure or loss.
    config=ep.groupby(['E','K'])[['D_total','N_act','N_total']].first().reset_index()
    d=d.merge(config,on=['E','K'],validate='many_to_one')
    d['exposure_fraction']=d.D_t/d.D_total
    d.to_csv(output/'benchmark_exposure_language_table.csv',index=False)
    d.loc[d.E.eq(1)].groupby(['K','task','training_status'],dropna=False).agg(languages=('language','size'),min_D_t=('D_t','min'),median_D_t=('D_t','median'),max_D_t=('D_t','max')).reset_index().to_csv(output/'exposure_coverage.csv',index=False)
    expert_cells=[];expert_summary=[];expert_tasks=[]
    for k,g in d.groupby('K'):
        for status in ['all_evaluated','trained','untrained','mapping_unresolved']:
            sub=g if status=='all_evaluated' else g.loc[g.training_status.eq(status)]
            if sub.empty: continue
            for metric in ['normalized_accuracy','raw_accuracy']:
                w=sub.pivot(index=['task','language'],columns='E',values=metric)
                assert w.notna().all().all()
                for a,b in combinations([1,4,16,32],2):
                    c=((w[b]-w[a])*100).rename('gain_pp').reset_index()
                    c=c.merge(sub.loc[sub.E.eq(a),['task','language','D_t','corpus_language']],on=['task','language'],validate='one_to_one')
                    c=c.assign(K=k,population=status,metric=metric,E_low=a,E_high=b)
                    expert_cells.append(c)
                    tasks=c.groupby('task').gain_pp.mean()
                    expert_tasks.append(tasks.rename('gain_pp').reset_index().assign(K=k,population=status,metric=metric,E_low=a,E_high=b))
                    expert_summary.append(dict(K=k,population=status,metric=metric,E_low=a,E_high=b,gain_pp=tasks.mean(),tasks=len(tasks),cells=len(c),positive=(c.gain_pp>1e-9).sum(),negative=(c.gain_pp< -1e-9).sum(),loto_min=min(tasks.drop(t).mean() for t in tasks.index) if len(tasks)>1 else np.nan,loto_max=max(tasks.drop(t).mean() for t in tasks.index) if len(tasks)>1 else np.nan))
    ec=pd.concat(expert_cells,ignore_index=True);es=pd.DataFrame(expert_summary)
    ec.to_csv(output/'exposure_expert_cell_contrasts.csv',index=False)
    es.to_csv(output/'exposure_expert_summary.csv',index=False)
    pd.concat(expert_tasks).to_csv(output/'exposure_expert_tasks.csv',index=False)
    # Pair task+language identities before assigning cohorts; K1 lacks some English tasks at higher K.
    pairs=[]
    fields=['E','task','language','corpus_language','training_status','D_t','adjusted_loss','normalized_accuracy','raw_accuracy']
    for ka,kb in combinations([1,12,50,200],2):
        a=d.loc[d.K.eq(ka),fields];b=d.loc[d.K.eq(kb),fields]
        p=a.merge(b,on=['E','task','language'],suffixes=('_low','_high'),validate='one_to_one')
        p['cohort']=['mapping_unresolved' if 'mapping_unresolved' in (x,y) else 'shared_trained' if x==y=='trained' else 'newly_trained' if x=='untrained' and y=='trained' else 'untrained_both' for x,y in zip(p.training_status_low,p.training_status_high)]
        p['K_low']=ka;p['K_high']=kb
        p['exposure_ratio']=p.D_t_high/p.D_t_low.replace(0,np.nan)
        p['loss_improvement']=p.adjusted_loss_low-p.adjusted_loss_high
        for metric in ['normalized_accuracy','raw_accuracy']:
            c=p.copy();c['metric']=metric;c['accuracy_change_pp']=100*(c[metric+'_high']-c[metric+'_low']);pairs.append(c)
    pc=pd.concat(pairs,ignore_index=True);pc.to_csv(output/'exposure_coverage_cell_contrasts.csv',index=False)
    summary=[];interactions=[]
    keys=['K_low','K_high','cohort','metric']
    for key,g in pc.groupby(keys):
        for e,c in g.groupby('E'):
            summary.append(dict(zip(keys,key),E=e,change_pp=macro(c,'accuracy_change_pp'),tasks=c.task.nunique(),cells=len(c),median_exposure_ratio=c.exposure_ratio.median(),min_exposure_ratio=c.exposure_ratio.min(),max_exposure_ratio=c.exposure_ratio.max(),median_D_t_low=c.D_t_low.median(),median_D_t_high=c.D_t_high.median(),mean_loss_improvement=macro(c,'loss_improvement')))
        w=g.pivot(index=['task','language'],columns='E',values='accuracy_change_pp')
        for a,b in combinations([1,4,16,32],2):
            t=(w[b]-w[a]).groupby('task').mean()
            interactions.append(dict(zip(keys,key),E_low=a,E_high=b,interaction_pp=t.mean(),loto_min=min(t.drop(x).mean() for x in t.index) if len(t)>1 else np.nan,loto_max=max(t.drop(x).mean() for x in t.index) if len(t)>1 else np.nan))
    cs=pd.DataFrame(summary);cs.to_csv(output/'exposure_coverage_summary.csv',index=False)
    pd.DataFrame(interactions).to_csv(output/'exposure_coverage_interactions.csv',index=False)
    # Within K expert gains versus exposure, avoiding pooled K differences.
    correlations=[]
    for (k,t),g in ec.query('population=="trained" and metric=="normalized_accuracy" and E_low==1 and E_high==32').groupby(['K','task']):
        correlations.append(dict(K=k,task=t,languages=len(g),spearman_exposure_vs_expert_gain=float(spearmanr(g.D_t,g.gain_pp).statistic) if len(g)>=4 and g.D_t.nunique()>1 and g.gain_pp.nunique()>1 else np.nan))
    pd.DataFrame(correlations).to_csv(output/'exposure_gain_correlations.csv',index=False)
    checks=dict(rows=len(d),verified_metric_files=len(manifest['metric_score_pairs']),checkpoint_paths_match=True,exposure_identical_across_E=True,exposure_sums_to_total=True,total_tokens=int(ep.D_total.iloc[0]),active_parameters=int(ep.N_act.iloc[0]),source_manifest_sha256=sha(tables/'benchmark_artifact_manifest.json'),script_sha256=sha(__file__))
    (output/'exposure_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('EXPERT GAINS');print(es.query('E_low==1 and E_high==32 and metric=="normalized_accuracy"').to_string(index=False))
    print('SHARED COVERAGE');print(cs.query('cohort=="shared_trained" and metric=="normalized_accuracy"').to_string(index=False))
    print('COHORT K12 TO K200');print(cs.query('K_low==12 and K_high==200 and metric=="normalized_accuracy"').to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository containing the sealed primary release')
    parser.add_argument('--tables', type=Path, required=True, help='Directory containing benchmark manifest and membership map')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
    args=parser.parse_args()
    main(args.root, args.tables, args.output)
