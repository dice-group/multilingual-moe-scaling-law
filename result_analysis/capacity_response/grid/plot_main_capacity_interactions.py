"""Paper figure: both capacity gains and the signed matched-exposure response."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from result_analysis.capacity_response.tables import load_primary,sha256_file


def main(root,audit,output):
    output.mkdir(parents=True,exist_ok=False)
    raw=load_primary(root)
    languages=sorted(raw[raw.K==12].language.unique())
    raw=raw[raw.language.isin(languages)&raw.K.isin([12,200])]
    means=raw.groupby(['ladder','N_act','E','K','D_total'],as_index=False).agg(loss=('raw_nll','mean'),languages=('language','nunique'))
    assert means.languages.eq(12).all()
    a=means.pivot(index=['ladder','N_act','K','D_total'],columns='E',values='loss')
    a['gain']=a[1]-a[32];a=a.reset_index()
    a['FLOPs']=6*a.N_act.astype(float)*a.D_total.astype(float)
    observed=pd.read_csv(audit/'observed_expert_gains.csv').query("cohort=='common12' and low_E==1 and high_E==32 and K in [12,200]")
    parity=a.merge(observed,on=['ladder','N_act','K','D_total'],suffixes=('_direct','_audit'),validate='one_to_one')
    error_a=float(abs(parity.gain_direct-parity.gain_audit).max())
    assert len(a)==50 and len(parity)==50 and error_a<1e-12
    b=means[means.ladder.isin(['A','D'])].pivot(index=['E','K','D_total'],columns='ladder',values='loss').dropna().reset_index()
    b['gain']=b.A-b.D
    assert len(b)==40
    l=pd.read_csv(audit/'language_losses.csv.gz').query("regime=='target_exposure' and analysis=='K12_to_K200' and method=='linear_log' and E in [1,32] and summary!='log_mean'")
    p=l.pivot(index=['language','ladder','E','summary'],columns='K',values='loss')
    p['response']=p[200]-p[12]
    assert p.groupby(['ladder','E','summary']).size().eq(12).all()
    c=p.groupby(['ladder','E','summary']).response.mean().reset_index()
    c['position']=c.summary.map(dict(early=0.,quarter=.25,middle=.5,three_quarter=.75,late=1.))
    contrasts=pd.read_csv(audit/'contrasts.csv.gz')
    check=contrasts.query("regime=='target_exposure' and analysis=='K12_to_K200' and method=='linear_log' and contrast=='K_response' and low_E in [1,32] and summary!='log_mean'")
    m=c.merge(check,left_on=['ladder','E','summary'],right_on=['low_N','low_E','summary'],validate='one_to_one')
    error_c=float(abs(m.response-m['mean']).max());assert len(m)==40 and error_c<1e-12
    check_b=contrasts.query("regime=='tokens' and analysis=='ABCD' and cohort=='common12' and method=='linear_log' and contrast=='N_gain' and low_N=='A' and high_N=='D' and low_K in [12,200] and summary in ['early','late']")
    errors=[]
    for r in check_b.itertuples():
        v=b[(b.E==r.low_E)&(b.K==r.low_K)].sort_values('D_total')
        errors.append(abs(float(v.gain.iloc[0 if r.summary=='early' else -1])-r.mean))
    assert len(errors)==16 and max(errors)<1e-12
    a.to_csv(output/'panel_a_expert_gain.csv',index=False)
    b.to_csv(output/'panel_b_active_size_gain.csv',index=False)
    c.to_csv(output/'panel_c_matched_response.csv',index=False)
    colors=plt.colormaps['inferno']([.14,.37,.60,.78])
    nc=dict(zip('ABCD',colors));ec=dict(zip([1,4,16,32],colors))
    nlabels=dict(zip('ABCD',['92.8M','190.3M','349.2M','589.8M']))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,'axes.titlesize':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.45))
    fig.subplots_adjust(left=.079,right=.985,bottom=.19,top=.68,wspace=.43)
    aa,bb,cc=axes
    for n in 'ABCD':
        for k,style in [(12,'--'),(200,'-')]:
            g=a[(a.ladder==n)&(a.K==k)].sort_values('D_total')
            aa.plot(g.FLOPs,g.gain,color=nc[n],ls=style,marker='o',ms=2,lw=1)
        for e,style in [(1,'--'),(32,'-')]:
            g=c[(c.ladder==n)&(c.E==e)].sort_values('position')
            cc.plot(g.position,g.response,color=nc[n],ls=style,marker='o',mfc='white',ms=2.6,lw=1)
    for e in [1,4,16,32]:
        for k,style in [(12,'--'),(200,'-')]:
            g=b[(b.E==e)&(b.K==k)].sort_values('D_total')
            bb.plot(g.D_total/1e9,g.gain,color=ec[e],ls=style,marker='o',ms=2,lw=1)
    for ax,title in zip(axes,['(a) Expert capacity gain','(b) Active capacity gain','(c) Multilingual response']):
        ax.set_title(title,pad=48)
        ax.grid(axis='y',alpha=.18,lw=.5)
        ax.axhline(0,color='.5',lw=.65,zorder=0)
        ax.tick_params(labelsize=7,length=2.5)
    aa.set_ylabel('Loss advantage: E1 − E32 (nats)')
    bb.set_ylabel('Loss advantage: 92.8M − 589.8M (nats)')
    cc.set_ylabel('Loss difference: K200 − K12 (nats)')
    aa.set_ylim(0,.33);bb.set_ylim(bottom=0)
    cc.set_ylim(-.11,.19);cc.set_yticks([-.1,0,.1])
    aa.set_xscale('log');aa.set_xticks([1e18,1e19,1e20],labels=[r'$10^{18}$',r'$10^{19}$',r'$10^{20}$']);aa.set_xlabel('Training FLOPs')
    for ax,ticks in [(bb,[.5,1,2,4,8])]:
        ax.set_xscale('log',base=2);ax.set_xticks(ticks,labels=[str(t) for t in ticks]);ax.set_xlabel('Total training tokens (B)')
    cc.set_xticks([0,.5,1],labels=['Early','Middle','Late']);cc.set_xlabel('Relative target exposure')
    for ax,col,labels,title in [(aa,nc,nlabels,'Active parameters'),(bb,ec,{e:f'E = {e}' for e in ec},'Expert count'),(cc,nc,nlabels,'Active parameters')]:
        legend=ax.legend([Line2D([],[],color=col[k],lw=1.4) for k in col],[labels[k] for k in col],
            title=title,title_fontsize=7,fontsize=6.8,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.015),
            frameon=False,columnspacing=.8,handlelength=1.4,handletextpad=.35,borderaxespad=0)
    # Separate line-style keys under the x axes prevent color-role ambiguity.
    for ax,labels in [(aa,['K12','K200']),(bb,['K12','K200']),(cc,['E1','E32'])]:
        for xpos,style,label in zip([.14,.64],['--','-'],labels):
            ax.plot([xpos-.1,xpos+.03],[-.31,-.31],transform=ax.transAxes,color='.3',ls=style,lw=1,clip_on=False)
            ax.text(xpos+.06,-.31,label,transform=ax.transAxes,va='center',fontsize=7)
    fig.savefig(output/'Main_result1_capacity_interactions.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/'Main_result1_capacity_interactions.png',dpi=300)
    plt.close(fig)
    caption=r'''\textbf{Active and expert capacity jointly shape the response to multilingual breadth.}
All panels report equal-weight mean validation-loss contrasts over the same 12 languages shared by the K12 and K200 mixtures.
(a) Expert expansion gain, $L_{E=1}-L_{E=32}$, versus training FLOPs ($F=6N_{\mathrm{act}}D_{\mathrm{total}}$) at each active size. Within each size, the expert comparison matches total tokens and logical compute; different sizes receive different token counts at equal compute.
(b) Active-size gain, $L_{N=92.8\mathrm{M}}-L_{N=589.8\mathrm{M}}$, versus total tokens at each expert count.
Dashed/solid lines in (a--b) denote K12/K200; filled markers are observed cooled readouts and lines connect them.
Panel (a) retains each size's observed horizon; (b) stops at the common 8B-token horizon.
(c) Signed expansion response, $L_{K=200}-L_{K=12}$, at matched realized target-language exposure, with dashed/solid lines denoting E1/E32.
Negative values indicate lower loss under broader training. Open markers use interpolation linear in log exposure on windows shared across all active sizes and expert counts for each language.
Early, middle, and late denote normalized positions in each language's own log-exposure window, not identical absolute exposure across languages.
Matching target exposure requires more total training tokens at K200; it does not match total compute.
Colors encode active size in (a,c), expert count in (b). The primary grid has one seed; these are descriptive contrasts, not independent replications or fitted-law predictions.'''
    (output/'caption.tex').write_text(caption+'\n')
    (output/'figure_snippet.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{Main_result1_capacity_interactions.pdf}\n\\caption{'+caption+'}\n\\label{fig:result1-capacity-interactions}\n\\end{figure*}\n')
    (output/'verification.json').write_text(json.dumps(dict(panel_a_checks=50,panel_a_max_error=error_a,
        panel_b_endpoint_checks=16,panel_b_max_error=max(errors),panel_c_checks=40,panel_c_max_error=error_c,
        cohort=languages,input_hashes={str(audit/f):sha256_file(audit/f) for f in ['observed_expert_gains.csv','language_losses.csv.gz','contrasts.csv.gz']},
        builder_sha256=sha256_file(Path(__file__))),indent=2)+'\n')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['root','audit','output']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();main(a.root,a.audit,a.output)
