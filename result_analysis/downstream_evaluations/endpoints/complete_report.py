"""Verified synthesis of all completed endpoint and example-level diagnostics."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import sealed
from .endpoint_diagnostics import LOSS
from .example_diagnostics import CLASSIFICATION
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign import checked, ref, save

NAMES={'belebele_cloze':'Belebele','global_mmlu_full':'Global MMLU','okapi_hellaswag':'HellaSwag',
       'uhura_arc_easy':'ARC/Uhura','xcopa':'XCOPA','xstorycloze':'XStoryCloze','multiblimp':'MultiBLiMP',
       'x_csqa':'X-CSQA','xcomps':'XCOMPS','flores_mono':'FLORES','ntrex_mono':'NTREX',
       'arc_easy_eu20':'ARC Easy EU20','sib200':'SIB200','taxi1500':'TAXI-1500'}


def markdown(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)): return '—' if np.isnan(v) else f'{v:.3f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def finish(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main(root, analysis, report_name='report_final'):
    report=analysis/report_name;report.mkdir(exist_ok=False)
    figures=report/'figures';figures.mkdir();tables=report/'tables';tables.mkdir()
    source=report/'complete_report_source.py';source.write_bytes(Path(__file__).read_bytes())
    receipt=sealed(analysis/'analysis_receipt.json')
    for r in receipt['outputs'].values(): checked(r)
    for key in ['original_metrics','candidate_metrics','extension_metrics','source']: checked(receipt[key])
    examples=analysis/'examples';example_receipt=sealed(examples/'verification.json')
    assert example_receipt['raw_jobs_verified']==356
    for r in example_receipt['output_files'].values():checked(r)
    assert example_receipt['analysis_receipt']==ref(analysis/'analysis_receipt.json')
    d=pd.read_csv(analysis/'coverage_interactions.csv');c=pd.read_csv(analysis/'cohort_metrics.csv')
    f=pd.read_csv(analysis/'fixed_cohort_language_metrics.csv');models=pd.read_csv(analysis/'model_bindings.csv')
    raw=pd.read_csv(analysis/'language_metrics_with_validation.csv',low_memory=False)
    ex=pd.read_csv(examples/'language_examples.csv');ci=pd.read_csv(examples/'paired_bootstrap.csv')
    h=d.query("cohort=='common12' and metric=='primary' and E_low==1 and E_high==32 and seed==1234").copy()
    assert len(h)==56 and set(h.benchmark)==set(NAMES)
    seeds=d.query("cohort=='common12' and metric=='primary' and E_low==1 and E_high==32 and ladder=='B'")
    assert len(seeds)==42
    checks={}
    data_audits=[]
    for campaign_ref in example_receipt['inputs']:
        if Path(campaign_ref['path']).name != 'campaign.json': continue
        campaign=sealed(checked(campaign_ref))
        data_ref=campaign.get('data_manifest') or campaign['data_manifests']['full']
        data=sealed(checked(data_ref));assert data['sampling'] is None
        assert not data['unsupported_long_targets']
        audit_ref=data.get('exclusion_audit') or data['approved_exclusions']
        checked(audit_ref)
        excluded=data.get('exclusions',data.get('excluded_documents',[]))
        reasons=Counter(x.get('reason','unspecified') for x in excluded) if isinstance(excluded,list) else {'total':excluded}
        data_audits.append(dict(campaign=campaign_ref,data=data_ref,exclusion_audit=audit_ref,
                               counts_by_reason=dict(reasons),note='Unavailable language groups and unavailable examples are distinct units; counts are not interchangeable.'))
    save(tables/'data_exclusion_audit.json',dict(audits=data_audits),'manifest_sha256')
    # Independent reproduction against the historical canonical analysis.
    old=root/'outputs/benchmark_review/full_final_scaling_20260920_v2'
    old_ver=json.loads((old/'verification.json').read_text())
    assert ref(old/'coverage_interactions.csv')['sha256']==old_ver['outputs']['coverage_interactions']['sha256']
    prev=pd.read_csv(old/'coverage_interactions.csv').query("scope=='common12' and low==12 and high==200 and seed==1234")
    merged=h.merge(prev,on=['benchmark','ladder','seed'],validate='one_to_one')
    assert len(merged)==24
    error=float((merged.interaction-merged.acc_norm_interaction_pp).abs().max());assert error<1e-10
    checks['original_24_max_difference']=error
    paper_dir=old/'paper_comparison'
    paper_receipt=json.loads((paper_dir/'verification.json').read_text())
    paper_path=paper_dir/'expert_gains_equal_task.csv'
    assert ref(paper_path)['sha256']==paper_receipt['outputs'][paper_path.name]['sha256']
    paper_comparison=pd.read_csv(paper_path)
    paper_comparison.to_csv(tables/'original_paper_vs_full_expert_gains.csv',index=False)
    checks['historical_paper_comparison_verified']=True
    # Independent candidate summary arithmetic for all three families and seeds.
    candidate=sealed(Path(receipt['candidate_metrics']['path']))
    sm=pd.DataFrame(candidate['benchmark_summary']).query("cohort=='common12' and E in [1,32] and K in [12,200]")
    errors=[]
    for (b,l,s),g in sm.groupby(['benchmark','ladder','seed']):
        wide=g.set_index(['E','K']).language_macro_accuracy
        value=100*(wide.loc[32,200]-wide.loc[1,200]-wide.loc[32,12]+wide.loc[1,12])
        expected=d.query("benchmark==@b and ladder==@l and seed==@s and cohort=='common12' and metric=='primary' and E_low==1 and E_high==32").interaction.item()
        errors.append(abs(value-expected))
    assert len(errors)==18 and max(errors)<1e-10;checks['candidate_18_max_difference']=max(errors)
    # Every example-bootstrap point estimate reproduces its canonical language cell.
    lang=raw.set_index(['benchmark','ladder','seed','E','K','language'])
    errors=[]
    for row in ci.query("contrast=='interaction'").itertuples():
        col='primary' if row.metric=='primary' else row.metric
        values=[lang.loc[(row.benchmark,row.ladder,row.seed,e,k,row.language),col] for e,k in [(1,12),(32,12),(1,200),(32,200)]]
        orientation=-1 if row.benchmark in LOSS else 100
        expected=orientation*(values[3]-values[2]-values[1]+values[0])
        errors.append(abs(row.estimate-expected))
    assert max(errors)<1e-10;checks['paired_point_estimates_checked']=len(errors);checks['paired_max_difference']=max(errors)
    # Reference offsets must cancel in fixed-language contrasts.
    n=d.query("benchmark in @LOSS and metric=='external_nll'")
    a=d.query("benchmark in @LOSS and metric=='unigram_adjusted_loss'")
    pairs=n.merge(a,on=['benchmark','ladder','seed','cohort','E_low','E_high'],suffixes=('_raw','_adj'),validate='one_to_one')
    assert np.allclose(pairs.interaction_raw,pairs.interaction_adj,atol=1e-12,rtol=0)
    checks['raw_adjusted_contrasts']=len(pairs)
    h.to_csv(tables/'headline_interactions.csv',index=False)
    seeds.to_csv(tables/'b_seed_interactions.csv',index=False)
    seed_summary=seeds.groupby('benchmark').interaction.agg(['mean','min','max']).reset_index()
    seed_summary.to_csv(tables/'b_seed_summary.csv',index=False)
    # Same language cohort and checkpoints, with actual positive denominators.
    losses=[]
    for row in n.query("cohort=='common12' and E_low==1 and E_high==32").itertuples():
        cells=c.query("benchmark==@row.benchmark and ladder==@row.ladder and seed==@row.seed and cohort=='common12' and E in [1,32] and K in [12,200]").set_index(['E','K'])
        vl=cells.loc[1,200].validation_nll-cells.loc[1,12].validation_nll
        vh=cells.loc[32,200].validation_nll-cells.loc[32,12].validation_nll
        losses.append(dict(benchmark=row.benchmark,ladder=row.ladder,seed=row.seed,languages=row.languages,
                          penalty_E1=row.coverage_penalty_low,penalty_E32=row.coverage_penalty_high,
                          interaction=row.interaction,attenuation_percent=100*row.interaction/row.coverage_penalty_low if row.coverage_penalty_low>0 else np.nan,
                          validation_penalty_E1=vl,validation_penalty_E32=vh,validation_interaction=vl-vh,
                          validation_attenuation_percent=100*(vl-vh)/vl if vl>0 else np.nan,
                          positive_languages=row.languages_positive))
    losses=pd.DataFrame(losses);losses.to_csv(tables/'external_validation_loss_comparison.csv',index=False)
    # Coverage is descriptive, including all released varieties, not only common12.
    coverage=[]
    for b,g in raw.groupby('benchmark'):
        one=g[g.model_id==g.model_id.iloc[0]]
        coverage.append(dict(benchmark=b,models=g.model_id.nunique(),language_variants=len(one),items_per_model=int(one.samples.sum()),
                             min_items_per_variant=int(one.samples.min()),max_items_per_variant=int(one.samples.max()),
                             common12_languages=int(h[h.benchmark==b].languages.iloc[0])))
    pd.DataFrame(coverage).to_csv(tables/'coverage_inventory.csv',index=False)
    models.groupby('ladder')[['N_act','D_total']].agg(['min','max']).to_csv(tables/'model_scale_and_budget.csv')
    d.query("E_low==1 and E_high==32").to_csv(tables/'metric_and_language_sensitivities.csv',index=False)
    # Macro average regional variants first, exactly as in the endpoint analysis.
    keys=['benchmark','ladder','seed','E','K','corpus_language']
    metrics=[x for x in ['primary','gold_nll','wrong_nll','margin','margin_raw','margin_token','accuracy_raw','accuracy_token','accuracy_char','dominant_fraction','labels_predicted','macro_f1','majority_baseline'] if x in ex]
    means=ex.groupby(keys)[metrics].mean().groupby(keys[:-1]).mean().reset_index()
    means.to_csv(tables/'example_diagnostic_means.csv',index=False)
    example_inter=[]
    for (b,l,s),g in means.query("E in [1,32] and K in [12,200]").groupby(['benchmark','ladder','seed']):
        if l not in ['B','D']:continue
        wide=g.set_index(['E','K'])
        if len(wide)!=4:continue
        for metric in metrics:
            vals=[wide.loc[e,k][metric] for e,k in [(1,12),(32,12),(1,200),(32,200)]]
            if not np.isfinite(vals).all():continue
            factor=-1 if metric in ['gold_nll','wrong_nll'] else 100 if metric.startswith('accuracy') or (metric=='primary' and b not in LOSS) else 1
            x,y=factor*(vals[1]-vals[0]),factor*(vals[3]-vals[2])
            example_inter.append(dict(benchmark=b,ladder=l,seed=s,metric=metric,gain_K12=x,gain_K200=y,interaction=y-x))
    ei=pd.DataFrame(example_inter);ei.to_csv(tables/'example_diagnostic_interactions.csv',index=False)
    cir=ci.query("contrast=='interaction'").groupby(['benchmark','ladder','seed','metric']).agg(
        variants=('estimate','size'),positive_point=('estimate',lambda x:int((x>0).sum())),
        lower_above_zero=('low',lambda x:int((x>0).sum())),upper_below_zero=('high',lambda x:int((x<0).sum())))
    cir.to_csv(tables/'conditional_interval_summary.csv')
    cls=c.query("benchmark in @CLASSIFICATION and cohort=='common12'").copy()
    cls['above_majority_pp']=100*(cls.primary-cls.majority_baseline)
    cls['majority_macro_f1']=2*cls.majority_baseline/(1+cls.majority_baseline)/cls.benchmark.map({'sib200':7,'taxi1500':6})
    cls.to_csv(tables/'classification_baselines.csv',index=False)
    classification_overview=cls.groupby('benchmark').agg(cells=('primary','size'),
        above_majority=('above_majority_pp',lambda v:int((v>1e-12).sum())),
        minimum_accuracy=('primary','min'),maximum_accuracy=('primary','max'),
        minimum_macro_f1=('macro_f1','min'),maximum_macro_f1=('macro_f1','max')).reset_index()
    classification_overview.to_csv(tables/'classification_overview.csv',index=False)
    cls_d=cls.query("ladder=='D' and seed==1234 and E in [1,32] and K in [12,200]")[['benchmark','E','K','primary','macro_f1','majority_baseline','majority_macro_f1']]
    # Paper-ready figures, with distinct axes for distinct units.
    accuracy=h[~h.benchmark.isin(LOSS)].pivot(index='benchmark',columns='ladder',values='interaction')
    fig,ax=plt.subplots(figsize=(8,7));v=float(np.abs(accuracy.values).max())
    im=ax.imshow(accuracy.values,cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
    ax.set_xticks(range(4),accuracy.columns);ax.set_yticks(range(len(accuracy)),[NAMES[b] for b in accuracy.index])
    for i,j in np.ndindex(accuracy.shape):ax.text(j,i,f'{accuracy.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=9)
    ax.set_title('Accuracy interaction: expert gain at K200 minus K12\nE1→E32; fixed shared languages; main seed')
    fig.colorbar(im,ax=ax,label='Percentage points; positive = greater benefit at K200');finish(fig,figures,'accuracy_interactions')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,label in zip(axes,['external_nll','bits_per_byte'],['Nats/token','Bits/UTF-8 byte']):
        for b in sorted(LOSS):
            rows=d.query("benchmark==@b and metric==@metric and cohort=='common12' and seed==1234 and E_low==1 and E_high==32").sort_values('ladder')
            ax.plot(rows.ladder,rows.interaction,marker='o',label=NAMES[b])
        if metric=='external_nll':
            rows=losses.query("benchmark=='flores_mono' and seed==1234").sort_values('ladder')
            ax.plot(rows.ladder,rows.validation_interaction,marker='s',label='Validation',linestyle='--')
        ax.axhline(0,color='gray',lw=.8);ax.set_ylabel(label);ax.set_xlabel('Model scale / training budget');ax.legend()
    fig.suptitle('External predictive-loss interactions, E1→E32, K12→K200');finish(fig,figures,'external_loss_interactions')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,l in zip(axes.flat,['A','B','C','D']):
        for b in sorted(LOSS):
            g=losses.query('benchmark==@b and ladder==@l and seed==1234').iloc[0]
            ax.plot([1,32],[g.penalty_E1,g.penalty_E32],marker='o',label=NAMES[b])
        ax.plot([1,32],[g.validation_penalty_E1,g.validation_penalty_E32],marker='s',linestyle='--',label='Validation')
        ax.set_title(l);ax.set_xticks([1,32]);ax.set_ylabel('K200−K12 penalty (nats/token)');ax.set_xlabel('Experts');ax.legend(fontsize=8)
    finish(fig,figures,'loss_coverage_penalties')
    for l in ['A','B','C','D']:
        fig,axes=plt.subplots(4,4,figsize=(14,12))
        for ax,b in zip(axes.flat,NAMES):
            for k in [12,50,200]:
                g=c.query("ladder==@l and benchmark==@b and seed==1234 and cohort=='common12' and K==@k").sort_values('E')
                ax.plot(g.E,g.primary*(1 if b in LOSS else 100),marker='.',label=f'K{k}')
            ax.set_title(NAMES[b]);ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,32],[1,4,16,32]);ax.set_ylabel('Bits/byte' if b in LOSS else 'Accuracy (%)')
            if b in CLASSIFICATION:ax.axhline(g.majority_baseline.iloc[0]*100,color='gray',linestyle=':',label='Majority')
        for ax in axes.flat[14:]:ax.axis('off')
        axes.flat[0].legend(fontsize=8);meta=models[models.ladder==l].iloc[0]
        fig.suptitle(f"Scale {l}: {meta.N_act/1e6:.1f}M active parameters, {meta.D_total/1e9:.1f}B training tokens",fontsize=14)
        finish(fig,figures,'expert_curves_'+l)
    for group,label in [('accuracy','Percentage-point interaction'),('loss','Bits/byte interaction')]:
        data=seeds[seeds.benchmark.isin(LOSS) if group=='loss' else ~seeds.benchmark.isin(LOSS)]
        names=sorted(data.benchmark.unique());fig,ax=plt.subplots(figsize=(9,max(3,len(names)*.45)))
        for si,seed in enumerate([1234,5678,91011]):
            vals=data.query('seed==@seed').set_index('benchmark').loc[names].interaction
            ax.scatter(vals,np.arange(len(names))+(si-1)*.12,label=str(seed),s=30)
        ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(names)),[NAMES[b] for b in names]);ax.set_xlabel(label);ax.set_title('B-scale training-seed sensitivity');ax.legend(title='Seed');finish(fig,figures,'seed_interactions_'+group)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,b in zip(axes,sorted(CLASSIFICATION)):
        g=means.query('benchmark==@b and K in [12,200] and seed==1234')
        for l in ['A','B','C','D']:
            for k in [12,200]:
                z=g.query('ladder==@l and K==@k').sort_values('E')
                ax.plot(z.E,z.dominant_fraction*100,marker='.',linestyle='-' if k==200 else '--',label=f'{l} K{k}')
        ax.set_title(NAMES[b]);ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,32],[1,4,16,32]);ax.set_ylabel('Dominant predicted label (%)');ax.set_xlabel('Experts')
    axes[1].legend(fontsize=7,ncol=2);finish(fig,figures,'classification_label_concentration')
    detail=ei.query("ladder=='D' and seed==1234")
    tasks=sorted(set(detail[detail.metric=='gold_nll'].benchmark))
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for index,(metric,label) in enumerate([('gold_nll','Correct answer'),('wrong_nll','Mean distractor')]):
        values=detail[detail.metric==metric].set_index('benchmark').loc[tasks].interaction
        axes[0].barh(np.arange(len(tasks))+(index-.5)*.3,values,height=.28,label=label)
    values=detail[detail.metric=='primary'].set_index('benchmark').loc[tasks].interaction
    axes[1].barh(np.arange(len(tasks)),values,height=.6)
    for ax in axes:ax.axvline(0,color='gray',lw=.8)
    axes[0].set_yticks(range(len(tasks)),[NAMES[b] for b in tasks]);axes[0].legend(fontsize=8)
    axes[0].set_xlabel('Change in expert NLL reduction (nats/token)')
    axes[1].set_xlabel('Change in expert accuracy gain (percentage points)')
    fig.suptitle('D-scale likelihood versus answer discrimination\nE1→E32 benefit at K200 minus K12; fixed common-language panel')
    finish(fig,figures,'likelihood_vs_discrimination')
    # Uniform task-by-task interpretation; classify by signs without p-value claims.
    decisions=[]
    for b in NAMES:
        main=h[h.benchmark==b].set_index('ladder').interaction
        ss=seeds[seeds.benchmark==b].interaction
        interpretation=('Classification limitation; examine majority baseline and F1' if b in CLASSIFICATION else
                        'Direction consistent across four scales and three B seeds' if (main>0).all() and (ss>0).all() else
                        'Negative interaction at all four main-seed scales' if (main<0).all() else
                        'Mixed by scale and/or seed')
        decisions.append(dict(benchmark=NAMES[b],languages=int(h[h.benchmark==b].languages.iloc[0]),
                              **{l:main[l] for l in ['A','B','C','D']},B_seed_min=ss.min(),B_seed_max=ss.max(),interpretation=interpretation))
    decisions=pd.DataFrame(decisions);decisions.to_csv(tables/'benchmark_findings.csv',index=False)
    loss_table=losses.query('seed==1234')[['benchmark','ladder','penalty_E1','penalty_E32','attenuation_percent','validation_attenuation_percent']].copy()
    loss_table['benchmark']=loss_table.benchmark.map(NAMES)
    # Export a self-contained LaTeX table without modifying the paper.
    lines=['% Endpoint contrasts: fixed total budget, not matched target exposure.',r'\begin{tabular}{lrrrr}',r'\hline',r'Task & A & B & C & D \\',r'\hline']
    for row in decisions.itertuples():
        unit=' (bits/byte)' if row.benchmark in ['FLORES','NTREX'] else ' (pp)'
        lines.append(row.benchmark+unit+' & '+' & '.join(f'{getattr(row,l):+.3f}' for l in ['A','B','C','D'])+r' \\')
    lines.extend([r'\hline',r'\end{tabular}']);(report/'proposed_benchmark_table.tex').write_text('\n'.join(lines)+'\n')
    summary=f'''# Complete benchmark evidence: task-dependent transfer

All 14 families × 82 final checkpoints are complete (1,148 evaluations). The
64 main configurations span A–D, E1/4/16/32, K1/12/50/200, seed 1234. The 18 B
repeats use E1/16/32, K1/12/200, seeds 5678 and 91011. No intermediate checkpoints
or new evaluations enter this analysis.

## Main result

The complete evidence supports external predictive-loss transfer and consistent
grammar-accuracy interactions, while downstream answer discrimination remains
task-dependent. FLORES and NTREX have positive E1→E32 coverage interactions at
all four main-seed scales and across all three B seeds. MultiBLiMP accuracy
also has this direction at all four scales and three B seeds. This does not
establish a universal downstream accuracy law.

Positive interaction means the E32-over-E1 benefit is greater at K200 than K12.
The primary panel fixes each benchmark's supported K12 training languages.
Accuracy values below are percentage points; FLORES/NTREX values are bits/byte.
Regional variants are averaged within corpus language before language-macro
aggregation. Cross-task language panels differ; ARC/Uhura's panel is English-only.

{markdown(decisions)}

## External text likelihood

This is the full E1→E32 contrast, superseding the earlier partial D/E16→E32
comparison. The columns compare exactly the same language cohort and checkpoints
for external and validation loss. All baseline penalties in this table are positive.

{markdown(loss_table)}

FLORES main-seed penalty reductions are 22–58%; NTREX reductions are 20–62%.
Their B-repeat signs remain positive. Raw and unigram-adjusted loss interactions
agree because the fixed-language reference offsets cancel. Bits/byte preserves
the positive aggregate directions, including the non-English sensitivities.
These are normalizations of the same likelihoods, not independent replications.
Perplexity is available in the collected metrics but is not treated as another
independent result. Whitespace word perplexity remains deliberately unreported.

The paper's 83% result controls realized target-language exposure. These final
checkpoints instead fix total training budget within each scale: do not compare
these attenuation percentages as estimates of the same quantity. Endpoint
agreement supports the direction and external-domain applicability, not the
matched-exposure effect magnitude. A–D also differ in active size and sometimes
training budget; their curves are not a pure size intervention.

## Accuracy and diagnostic interpretation

MultiBLiMP is the clearest completed accuracy corroboration. Its non-English
and length-normalization sensitivities preserve the main direction. However,
likelihood-margin interactions are not uniformly positive; high grammar accuracy
creates ceiling and threshold effects. Describe this as accuracy-level transfer.

X-CSQA remains seed-sensitive; XCOMPS changes interaction sign across scales and
has negative interactions across all three B seeds. HellaSwag and XStoryCloze
retain negative interactions at all four main-seed scales. Full ARC Easy EU20
is mixed across scales, so its earlier positive D/E4→E32 partial result cannot
stand in for the full grid. These counterexamples must remain visible.

SIB200 and TAXI require majority baselines, F1 and prediction concentration.
Positive raw-accuracy interactions are not sufficient when classification is
near baseline or concentrated on a small subset of labels. The complete
classification baseline table and confusion matrices retain every main and
repeat checkpoint on the fixed common-language panel. K1 entries in those
prediction diagnostics include languages not trained at K1 and are not used
for multilingual coverage or matched validation-loss contrasts.

{markdown(classification_overview)}

This overview contains the 60 available K12/K50/K200 checkpoints. All 60 TAXI
accuracy means remain below its majority-class baseline. SIB exceeds majority
accuracy in 21/60 cells, reaching 33.7%; it is therefore not uniformly at chance
or devoid of learned signal. Its accuracy and macro-F1 interactions disagree
at several scales. For example, D has a −1.348-point accuracy interaction but
a +3.579-point macro-F1 interaction. TAXI at D has negative interactions in
both accuracy (−3.303 points) and macro-F1 (−3.502 points), superseding any
generalization from the earlier favorable E4→E16 accuracy-only comparison.

D-scale baseline-aware cells (accuracy and F1 are fractions):

{markdown(cls_d)}

`example_diagnostic_interactions.csv` separates gains in correct-answer NLL,
distractor NLL and relative margins from accuracy. Greater likelihood gain can
coexist with smaller accuracy gain when distractors improve too. Corrected and
broken answers are paired by exact frozen example. These are descriptive
explanations of score behavior, not identified causal mechanisms.

## Robustness and evidence boundaries

- All three B seeds are shown separately; their range is descriptive, not a
  population confidence interval. No seed replication is claimed at A/C/D.
- 2,000-resample paired percentile intervals are computed within each language
  for D and all B seeds, E1/E32 × K12/K200. Resampling retains source-document
  blocks and every NTREX window. Intervals quantify evaluation-sample uncertainty
  conditional on trained checkpoints, not training-seed uncertainty.
- No pooled multilingual interval or significance tally is claimed: translated
  examples are dependent, benchmark families can share sources, and multiple
  exploratory contrasts have not been corrected into confirmatory tests.
  Interval rows retain released language varieties (for example, 18 NTREX
  varieties map to the 12-language primary cohort); these are not 18 independent
  training languages. Primary cohort means collapse the variants first.
- FLORES, SIB and Belebele share a source family. NTREX provides a separate
  news evaluation, but this analysis does not establish absence of training-data
  contamination or broad out-of-distribution generalization.
- Dataset exclusions are those frozen before evaluation. Near-chance tasks are
  retained; low scores alone do not prove the task is inappropriate or the model
  is too small. Label concentration does not identify its architectural or
  prompting cause.
- The original Global MMLU protocol explicitly allows 62 left-context-truncated
  option requests per model; answers are complete. This audit preserves that
  frozen protocol and checks identical token/truncation counts across paired
  models. The supplementary campaigns retain their zero-truncation checks.

## Relation to the paper's original 100-example results

The prior verified paper comparison remains applicable: the D-scale six-task
trained-language equal-task expert gains were +1.638/+0.687/+0.684 pp at
K12/K50/K200, versus +1.683/+0.580/+0.607 pp on the expanded complete suite.
Thus fuller evaluation preserved the broad original accuracy interpretation.
This is not a pure sample-size experiment: Global MMLU Lite changed to full
MMLU and some language/question sets expanded. The new tasks add evidence about
which measurements transfer; they do not erase the original counterexamples.

## Recommendation

Keep the validation-loss claim and distinguish three evidence layers:
external predictive likelihood (consistent direction), linguistic discrimination
(MultiBLiMP accuracy, with margin/ceiling qualifications), and broad downstream
accuracy (mixed transfer). Keep SIB/TAXI as protocol/capability limitations
unless their baseline-aware evidence supports a narrower positive claim.
This is a retrospective analysis: additional benchmarks were motivated by the
earlier results. Within the complete analysis, every evaluated family and its
frozen primary metric is retained; no favorable-outcome filtering is applied.

Proposed manuscript wording and a LaTeX table are provided alongside this report,
without editing the manuscript. Figures use separate accuracy and loss units.
All tables are machine-readable; `verification.json` records independent checks
and artifact hashes. The maintained reproduction instructions are in the analysis
README. No GPU jobs were launched for this work.
'''
    (report/'findings.md').write_text(summary)
    (report/'proposed_paper_text.md').write_text('''# Proposed paper material — not applied to manuscript

## Main-text paragraph

We evaluate 14 benchmark families at all 82 final checkpoints. At fixed total
training budget, increasing expert count from 1 to 32 attenuates the K12-to-K200
loss penalty on both FLORES and NTREX across all four model scales; the direction
holds across three training seeds at the B scale. The main-seed reductions span
22–58% on FLORES and 20–62% on NTREX. These endpoint comparisons corroborate the
direction on external text but do not reproduce the matched-target-exposure
estimand of our primary analysis. MultiBLiMP accuracy shows the same interaction
direction across four scales and three B seeds, whereas commonsense and reasoning
accuracy remain task- and scale-dependent. Classification diagnostics further
show that accuracy gains can reflect label concentration rather than improved
balanced discrimination.

## Appendix methods and limitations

Use the fixed benchmark-supported K12 language panel for K12/K50/K200 comparisons;
average regional varieties within corpus language and then weight languages
equally. Report English and non-English sensitivities, benchmark-specific primary
metrics, all B seeds and both absolute coverage penalties. Endpoint comparisons
do not control target-language exposure. Report paired within-language evaluation
bootstrap intervals separately from training-seed ranges; neither translated
languages nor related benchmark families count as independent training replicas.
Retain negative benchmark interactions and majority/F1 diagnostics alongside
positive results. Task likelihood, relative answer margins and thresholded
accuracy are distinct measurements.

## Figure captions

**External predictive-loss interactions.** E32-over-E1 benefit at K200 minus
that at K12, evaluated on each task's fixed common-language cohort. Left: raw
NLL, including validation loss at matching checkpoints. Right: bits per UTF-8
byte for the same external likelihood evaluations. Panels have different units.
These fixed-total-budget endpoint contrasts are not matched-exposure estimates.

**Accuracy interactions.** Each cell reports the change in E32-over-E1 accuracy
gain between K12 and K200, in percentage points, for the main seed. Positive
values indicate greater expert benefit at broader coverage. Language support
differs by benchmark; the ARC/Uhura common panel contains English only. SIB200
and TAXI must be interpreted alongside majority baselines and macro-F1.

**Seed sensitivity.** B-scale interactions for three separately trained seeds.
Dots are observed training runs, not confidence intervals. Separate figures
show accuracy and text-loss metrics. Missing E4/K50 repeat settings are not
imputed.

**Classification concentration.** Mean per-language dominant-label prediction
share across experts and model scales at K12 and K200, with regional variants
combined before language averaging. Dominant labels may differ across cells.
This diagnoses prediction concentration without identifying its cause.
''')
    outputs={str(p.relative_to(report)):ref(p) for p in report.rglob('*') if p.is_file()}
    save(report/'verification.json',dict(status='verified',endpoint_receipt=ref(analysis/'analysis_receipt.json'),
         example_receipt=ref(examples/'verification.json'),source=ref(source),checks=checks,
         historical_paper_comparison=ref(paper_dir/'verification.json'),outputs=outputs),'manifest_sha256')
    print(markdown(loss_table));print('Verified complete synthesis:',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--report-name', default='report_final')
    a=p.parse_args();main(a.root.resolve(),a.analysis.resolve(),a.report_name)
