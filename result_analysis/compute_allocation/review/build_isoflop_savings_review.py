"""Observed-grid IsoFLOP profiles and equal-loss compute savings across K."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from result_analysis.capacity_response.tables import sha256_file
KS=(1,12,50,200);ES=(1,4,16,32);LS=tuple('ABCD')
COL=dict(zip(ES,plt.get_cmap('inferno')([.06,.34,.62,.80])))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=a.source/'native_mean_learning_curves.csv'
    assert sha256_file(source)==json.loads((a.source/'manifest.json').read_text())['outputs'][source.name]
    data=pd.read_csv(source);assert data.languages.eq(data.K).all()
    data['F']=6.0*data.N_act.astype(float)*data.D_total.astype(float)/1e18
    curves={key:g.sort_values('F') for key,g in data.groupby(['K','ladder','E'])}
    assert len(curves)==64 and all(np.all(np.diff(g.mean_loss)<0) for g in curves.values())
    lo=max(g.F.min() for g in curves.values());hi=min(g.F.max() for g in curves.values())
    def loss(g,f):
        assert g.F.min()*(1-1e-12)<=f<=g.F.max()*(1+1e-12)
        return float(np.interp(np.log(f),np.log(g.F),g.mean_loss))
    profiles=[]
    for k in KS:
        for f in [2.,4.]:
            assert lo<f<hi
            for e in ES:
                batch=[]
                for l in LS:
                    g=curves[k,l,e];n=int(g.N_act.iloc[0])
                    batch.append(dict(K=k,F_EF=f,E=e,ladder=l,N_act=n,D_total=f*1e18/(6*n),mean_loss=loss(g,f)))
                best=min(r['mean_loss'] for r in batch)
                profiles.extend(dict(r,best_tested=r['mean_loss']==best) for r in batch)
    profiles=pd.DataFrame(profiles);profiles.to_csv(out/'isoflop_profiles.csv',index=False)
    budgets=np.unique(np.r_[np.geomspace(lo,hi,151),2.,4.])
    savings=[];audit=[];max_error=0.
    for k in KS:
        for f in budgets:
            dense=[(loss(curves[k,l,1],f),l) for l in LS]
            target,dense_l=min(dense)
            for e in [4,16,32]:
                candidates=[]
                for l in LS:
                    g=curves[k,l,e]
                    if target>g.mean_loss.iloc[0]:
                        # Earlier attainment is unknown. Retain first demonstrated
                        # success as an upper bound on required compute.
                        required=float(g.F.iloc[0]);status='first_readout_already_better'
                    elif target<g.mean_loss.iloc[-1]:
                        audit.append(dict(K=k,dense_F_EF=f,E=e,ladder=l,status='not_reached_in_support'))
                        continue
                    else:
                        required=float(np.exp(np.interp(target,g.mean_loss.to_numpy()[::-1],np.log(g.F.to_numpy()[::-1]))))
                        err=abs(loss(g,required)-target);max_error=max(max_error,err);assert err<1e-11
                        status='bracketed_equal_loss'
                    candidates.append((required,l,status))
                    audit.append(dict(K=k,dense_F_EF=f,E=e,ladder=l,status=status,required_F_EF=required))
                if not candidates:continue
                required,l,status=min(candidates)
                savings.append(dict(K=k,E=e,dense_F_EF=f,dense_ladder=dense_l,target_loss=target,
                                    moe_ladder=l,moe_F_EF=required,savings_percent=100*(1-required/f),status=status))
    savings=pd.DataFrame(savings);savings.to_csv(out/'equal_loss_compute_savings.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'inverse_support_audit.csv',index=False)
    assert len(savings)==len(budgets)*4*3
    fig,axes=plt.subplots(2,4,figsize=(15,10),squeeze=False)
    fig.subplots_adjust(left=.075,right=.978,bottom=.22,top=.80,wspace=.27,hspace=.65)
    fig.suptitle('Expert capacity, training allocation, and compute savings across K',x=.52,y=.97,fontsize=20,weight='bold',color='#183249')
    fig.text(.5,.927,'Observed-grid analogue of JointMoE: four tested active sizes; within-support interpolation along training trajectories',ha='center',fontsize=11,color='#526472')
    fig.legend(handles=[Line2D([],[],color=COL[e],lw=2,label=f'E = {e}') for e in ES],loc='upper center',bbox_to_anchor=(.32,.892),ncol=4,frameon=False,title='Experts (both rows)',fontsize=10,title_fontsize=10)
    fig.legend(handles=[Line2D([],[],color='#526472',ls='none',marker=m,ms=9,label=f'{f:g} EFLOPs') for f,m in [(2.,'*'),(4.,'D')]],loc='upper center',bbox_to_anchor=(.79,.892),ncol=2,frameon=False,title='Best tested size per profile',fontsize=10,title_fontsize=10)
    for j,k in enumerate(KS):
        ax=axes[0,j]
        for f,marker in [(2.,'*'),(4.,'D')]:
            for e in ES:
                g=profiles[(profiles.K==k)&(profiles.F_EF==f)&(profiles.E==e)].sort_values('D_total')
                ax.plot(g.D_total/1e9,g.mean_loss,color=COL[e],lw=1.2,marker='o',ms=3,mfc='white')
                best=g[g.best_tested];ax.scatter(best.D_total/1e9,best.mean_loss,color=COL[e],marker=marker,s=65 if marker=='*' else 35,zorder=4,edgecolors='white',linewidths=.5)
        ax.set_title(f'K = {k}',fontsize=13,pad=9);ax.set_xscale('log');ax.set_xlabel('Total training tokens (billions)',fontsize=9.5)
        ax.set_xlim(.5,8.5);ax.xaxis.set_major_locator(FixedLocator([.5,1,2,4,8]));ax.xaxis.set_major_formatter(FixedFormatter(['0.5','1','2','4','8']));ax.xaxis.set_minor_locator(NullLocator())
        if j==0:ax.set_ylabel('Mean validation loss (nats)',fontsize=11)
        ax=axes[1,j]
        for e in [4,16,32]:
            g=savings[(savings.K==k)&(savings.E==e)].sort_values('dense_F_EF')
            ax.plot(g.dense_F_EF,g.savings_percent,color=COL[e],lw=1.9)
            censored=g[g.status.ne('bracketed_equal_loss')]
            if len(censored):ax.scatter(censored.dense_F_EF,censored.savings_percent,color=COL[e],marker='^',s=12)
        ax.axhline(0,color='#647786',ls='--',lw=.7);ax.set_xscale('log');ax.set_xlim(lo,hi)
        ax.xaxis.set_major_locator(FixedLocator([2,3,4]));ax.xaxis.set_major_formatter(FixedFormatter(['2','3','4']));ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Dense-model budget (EFLOPs)',fontsize=9.5)
        if j==0:ax.set_ylabel('Compute savings at matched loss (%)',fontsize=11)
    for row in axes:
        low=min(ax.dataLim.ymin for ax in row);high=max(ax.dataLim.ymax for ax in row)
        if row is axes[1]:low=min(0,low)
        pad=(high-low)*.08
        for ax in row:ax.set_ylim(low-pad,high+pad)
    for ax in axes.flat:
        ax.grid(alpha=.18,lw=.6);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)
    fig.text(.075,.135,'Top: fixed F ≈ 6 N_act D_total. Small circles are evaluated tested sizes; connecting lines do not predict untested sizes. Symbols mark discrete minima.',fontsize=9.5,color='#526472')
    fig.text(.075,.109,'Bottom: 100 × (1 − F_MoE / F_dense), matching the best dense loss among four sizes. MoE cost is minimized over supported size trajectories.',fontsize=9.5,color='#526472')
    fig.text(.075,.083,'Interpolation is linear in log compute; no fitted law or extrapolation. Dense budgets retain all four sizes (≈1.77–4.45 EFLOPs).',fontsize=9.5,color='#526472')
    fig.text(.075,.057,'Each K uses its own equal-language mean. Primary seed only; logical FLOPs exclude storage/runtime costs. These are supported-grid, not global optima.',fontsize=9.5,color='#526472')
    stem='IsoFLOP_and_compute_savings_by_K'
    fig.savefig(out/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(out/f'{stem}.png',dpi=165);plt.close(fig)
    receipt=dict(source_sha256=sha256_file(source),builder_sha256=sha256_file(Path(__file__)),common_dense_compute_EF=[lo,hi],
                 isoflop_points=len(profiles),savings_points=len(savings),max_inversion_forward_error=max_error,
                 selected_attainment_status=savings.status.value_counts().to_dict(),inverse_candidate_status=pd.DataFrame(audit).status.value_counts().to_dict(),
                 outputs={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'pdf':str((out/f'{stem}.pdf').resolve()),'status':receipt['selected_attainment_status'],'max_error':max_error}))


if __name__=='__main__':
    with plt.rc_context({'font.family':'DejaVu Sans','pdf.fonttype':42}):main()
