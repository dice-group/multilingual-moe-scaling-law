"""Continuous weight-constrained frontiers of the frozen M6-ND law."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as nd
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.contracts import verify_seal
from result_analysis.capacity_response.tables import load_release_table,sha256_file


def problem(g,N,D):
    return SimpleNamespace(N=np.full(len(g),N),D=g.D_t.to_numpy()*D,E=g.E.to_numpy(),K=g.K.to_numpy(),
        D_family=g.D_family.to_numpy()*D,D_script=g.D_script.to_numpy()*D,D_other=g.D_other.to_numpy()*D,
        language_ids=tuple(g.language))


def main(root,output,caps=None,max_flops=2e20):
    output.mkdir(parents=True,exist_ok=False)
    reduction_path=root/'result_analysis/stage1_m6_size_duration_v3/reduction.json'
    reduction=json.loads(reduction_path.read_text());verify_seal(reduction,'sha256',label=str(reduction_path))
    pred_path=root/'result_analysis/stage1_size_duration_ablations_v1/predictions.parquet'
    allrows=pd.read_parquet(pred_path)
    rows=allrows.query("model=='M6-ND' and fold=='E32'").copy()
    assert len(rows)==26300 and not rows.duplicated(['trajectory_id','readout_id','language']).any()
    shares=rows.copy()
    for col in ['D_t','D_family','D_script','D_other']:shares[col]/=shares.D_total
    shares=shares.groupby(['K','language'],as_index=False)[['D_t','D_family','D_script','D_other','unigram_nll']].mean()
    assert np.max(abs(shares[['D_t','D_family','D_script','D_other']].sum(axis=1)-1))<1e-10
    shares.to_csv(output/'exposure_shares.csv',index=False)
    cfg=load_release_table(root,'model_configs',['ladder','experts','declared_active_parameters','configured_design_total_parameters'])
    cfg=cfg.rename(columns={'experts':'E','declared_active_parameters':'N','configured_design_total_parameters':'stored'}).sort_values('N')
    ns=cfg[cfg.E==1].N.to_numpy(float)
    caps=caps or [.4e9,1.2e9,5.1e9];experts=[1,4,16,32]
    budgets=np.geomspace(.3,max_flops/1e18,700)
    records=[];checks=[];laws=[];selections={}
    for fold in ['E32','largest_N_act']:
        selected=reduction['folds'][fold]['selected'];assert selected['success']
        selections[fold]={k:selected[k] for k in ['coordinates','language_coefficients']}
        old=allrows[(allrows.model=='M6-ND')&(allrows.fold==fold)]
        p=SimpleNamespace(N=old.N_act.to_numpy(),D=old.D_t.to_numpy(),E=old.E.to_numpy(),K=old.K.to_numpy(),
            D_family=old.D_family.to_numpy(),D_script=old.D_script.to_numpy(),D_other=old.D_other.to_numpy(),language_ids=tuple(old.language))
        parity=float(np.max(abs(nd.predict(p,selected)-old.prediction.to_numpy())))
        assert parity<1e-10
        checks.append(dict(fold=fold,historical_rows=len(old),historical_prediction_max_error=parity))
        cs=pd.DataFrame(selected['language_coefficients']).set_index('language')
        for k in [1,12,50,200]:
            g=shares[shares.K==k].copy();assert len(g)==k
            coeff=cs.loc[g.language]
            base=float(np.mean(coeff.L_inf.to_numpy()+g.unigram_nll.to_numpy()))
            for e in experts:
                g['E']=e
                cap_term,data_term,*_=nd.terms(selected['coordinates'],problem(g,1e8,1e9))
                cap2,data2,*_=nd.terms(selected['coordinates'],problem(g,2e8,1e9))
                _,dataD,*_=nd.terms(selected['coordinates'],problem(g,1e8,2e9))
                alpha=float(-np.log(cap2[0]/cap_term[0])/np.log(2))
                beta=float(-np.log(dataD[0]/data_term[0])/np.log(2))
                zeta=float(np.log(data2[0]/data_term[0])/np.log(2))
                C=float(np.mean(coeff.A.to_numpy()*cap_term));B=float(np.mean(coeff.B.to_numpy()*data_term))
                assert min(alpha,beta,C,B,zeta+beta)>0
                laws.append(dict(fold=fold,K=k,E=e,base=base,C=C,B=B,alpha=alpha,beta=beta,zeta=zeta))
                # Independent native-mean prediction checks of the reduced expression.
                for N,D in [(1.35e8,3e9),(5e8,40e9)]:
                    direct=np.mean(nd.predict(problem(g,N,D),selected)+g.unigram_nll.to_numpy())
                    reduced=base+C*(N/1e8)**(-alpha)+B*(N/1e8)**zeta*(D/1e9)**(-beta)
                    assert abs(direct-reduced)<1e-10
                geom=cfg[cfg.E==e].sort_values('N')
                for cap in caps:
                    if cap<geom.stored.min():continue
                    upper=float(np.interp(cap,geom.stored,geom.N))
                    for F in budgets:
                        # D>=0.5B, N within tested active-size range; continuous storage interpolation.
                        nlo=ns.min()/1e8;nhi=min(upper,F*1e18/(6*.5e9))/1e8
                        if nhi<nlo:continue
                        data_coef=B*(F/0.6)**(-beta)
                        optimum=(alpha*C/((zeta+beta)*data_coef))**(1/(alpha+zeta+beta))
                        n=float(np.clip(optimum,nlo,nhi));N=n*1e8;D=F*1e18/(6*N)
                        value=base+C*n**(-alpha)+data_coef*n**(zeta+beta)
                        stored=float(np.interp(N,geom.N,geom.stored))
                        horizon=float(np.exp(np.interp(np.log(N),np.log(ns),np.log([8e9,16e9,32e9,32e9]))))
                        assert stored<=cap*(1+1e-12) and abs(6*N*D/1e18-F)<1e-9
                        # Convex in log N: confirm analytic optimum against a dense grid.
                        grid=np.geomspace(nlo,nhi,65)
                        assert value<=np.min(base+C*grid**(-alpha)+data_coef*grid**(zeta+beta))+1e-10
                        records.append(dict(fold=fold,K=k,cap=cap,F_EF=F,E=e,N_act=N,N_total=stored,D_total=D,loss=value,
                                            horizon_extrapolation=D>horizon))
    candidates=pd.DataFrame(records)
    winners=candidates.loc[candidates.groupby(['fold','K','cap','F_EF']).loss.idxmin()].sort_values(['fold','K','cap','F_EF'])
    candidates.to_csv(output/'candidate_optima.csv.gz',index=False);winners.to_csv(output/'selected_optima.csv',index=False)
    pd.DataFrame(laws).to_csv(output/'reduced_mean_laws.csv',index=False)
    (output/'frozen_coefficients.json').write_text(json.dumps(selections,indent=2)+'\n')
    colors=dict(zip(experts,plt.colormaps['inferno']([.12,.36,.60,.79])))
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    with PdfPages(output/'Fitted_law_weights_only.pdf') as pdf:
        for fold in ['E32','largest_N_act']:
            fig,axes=plt.subplots(1,4,figsize=(16,5.5))
            fig.subplots_adjust(left=.055,right=.975,top=.77,bottom=.25,wspace=.24)
            fig.suptitle('Fitted scaling law: expert choices under weight caps',fontsize=17,y=.975)
            fig.text(.5,.902,f'Frozen M6-ND · {fold} held-out fit · continuous active-size and token allocation',ha='center',fontsize=10,color='#555555')
            fig.legend([Line2D([],[],color=colors[e],lw=2.5) for e in experts],[f'E = {e}' for e in experts],loc='upper center',bbox_to_anchor=(.5,.875),ncol=4,frameon=False)
            for ax,k in zip(axes,[1,12,50,200]):
                for cap in caps:
                    g=winners[(winners.fold==fold)&(winners.K==k)&(winners.cap==cap)].sort_values('F_EF')
                    assert (np.diff(g.loss)<=1e-9).all()
                    g=g.reset_index(drop=True)
                    cuts=np.r_[0,np.flatnonzero((g.E.to_numpy()[1:]!=g.E.to_numpy()[:-1])|
                        (g.horizon_extrapolation.to_numpy()[1:]!=g.horizon_extrapolation.to_numpy()[:-1]))+1,len(g)]
                    for start,stop in zip(cuts[:-1],cuts[1:]):
                        segment=g.iloc[start:min(stop+1,len(g))]
                        ax.plot(segment.F_EF*1e18,segment.loss,color=colors[int(segment.E.iloc[0])],lw=1.8,
                                ls='--' if segment.horizon_extrapolation.iloc[0] else '-')
                    end=g.iloc[-1]
                    ax.annotate(f'{cap/1e9:g}B',xy=(end.F_EF*1e18,end.loss),xytext=(-3,7),textcoords='offset points',ha='right',fontsize=10)
                ax.set(title=f'K = {k}',xlabel='Training FLOPs')
                ax.set_xscale('log');ax.set_xlim(.3e18,max_flops*1.2);ax.set_xticks([1e18,1e19,1e20,1e21],labels=[r'$10^{18}$',r'$10^{19}$',r'$10^{20}$',r'$10^{21}$']);ax.grid(alpha=.17)
            axes[0].set_ylabel('Predicted mean loss (nats)')
            fig.text(.055,.135,'Curve labels: stored-parameter caps. Color: selected expert count. Each K averages its own trained languages.',fontsize=10)
            fig.text(.055,.087,'Solid: within interpolated study token horizons. Dashed: longer-token extrapolation. Active size restricted to 92.8–589.8M.\n'
                     'Continuous weight accounting interpolates the tested architectures; E ∈ {1, 4, 16, 32}. Frozen fold fits, not a new full-grid refit.',fontsize=9,color='#555555',va='top',linespacing=1.5)
            pdf.savefig(fig);fig.savefig(output/f'Weights_only_{fold}.png',dpi=170);plt.close(fig)
    receipt=dict(inputs={str(p):sha256_file(p) for p in [reduction_path,pred_path]},fold_checks=checks,
        candidate_optima=len(candidates),selected_optima=len(winners),analytic_optimum_dense_grid_checks=len(candidates),
        storage='piecewise linear N_total(N_act) for each tested E; exact at architecture anchors',
        exposure='mean realized composition shares at each K, scaled continuously with total tokens',
        mean_prediction_parity_checks=64,scope='Frozen M6-ND folds; no refit; longer-token extrapolation is marked',builder_sha256=sha256_file(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--caps',type=float,nargs='+',help='Stored-parameter caps in billions')
    p.add_argument('--max-flops',type=float,default=2e20)
    a=p.parse_args();main(a.root,a.output,[v*1e9 for v in a.caps] if a.caps else None,a.max_flops)
