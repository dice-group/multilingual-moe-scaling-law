"""Equal-compute mean capacity gains on the full-grid shared support at K200."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,LogFormatterSciNotation,NullLocator


def main(source,output):
    output.mkdir(parents=True,exist_ok=False)
    path=source/'native_mean_learning_curves.csv';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==json.loads((source/'manifest.json').read_text())['outputs'][path.name]
    d=pd.read_csv(path).query('K==200').copy();assert len(d)==100 and d.languages.eq(200).all()
    d['FLOPs']=6*d.N_act.astype(float)*d.D_total.astype(float)
    curves={key:g.sort_values('FLOPs') for key,g in d.groupby(['ladder','E'])}
    lo=max(g.FLOPs.min() for g in curves.values());hi=min(g.FLOPs.max() for g in curves.values())
    x=np.unique(np.r_[np.geomspace(lo,hi,81),d.FLOPs[(d.FLOPs>=lo)&(d.FLOPs<=hi)],2e18,4e18])
    assert x.min()>=lo and x.max()<=hi
    predictions={key:np.interp(np.log(x),np.log(g.FLOPs),g.mean_loss) for key,g in curves.items()}
    records=[];errors=[]
    for e in [4,16,32]:
        v=np.mean([predictions[n,1]-predictions[n,e] for n in 'ABCD'],axis=0)
        direct=np.mean([predictions[n,1] for n in 'ABCD'],axis=0)-np.mean([predictions[n,e] for n in 'ABCD'],axis=0)
        errors.extend(abs(v-direct));records.extend(dict(panel='expert',target=str(e),FLOPs=f,mean_gain=y) for f,y in zip(x,v))
    for n in 'BCD':
        v=np.mean([predictions['A',e]-predictions[n,e] for e in [1,4,16,32]],axis=0)
        direct=np.mean([predictions['A',e] for e in [1,4,16,32]],axis=0)-np.mean([predictions[n,e] for e in [1,4,16,32]],axis=0)
        errors.extend(abs(v-direct));records.extend(dict(panel='active',target=n,FLOPs=f,mean_gain=y) for f,y in zip(x,v))
    assert max(errors)<1e-12
    results=pd.DataFrame(records);results.to_csv(output/'mean_compute_gains.csv',index=False)
    pd.DataFrame([dict(ladder=n,E=e,FLOPs=f,mean_loss=y) for (n,e),ys in predictions.items() for f,y in zip(x,ys)]).to_csv(output/'interpolated_losses.csv',index=False)
    colors=plt.colormaps['inferno']([.2,.5,.78])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.085,right=.98,bottom=.25,top=.78,wspace=.14)
    fig.suptitle('Mean capacity gains at equal training compute · K = 200',fontsize=15,y=.975)
    for ax,panel,targets,labels,title in [
        (axes[0],'expert',['4','16','32'],['E1 → E4','E1 → E16','E1 → E32'],'(a) Expert gain, averaged over active sizes'),
        (axes[1],'active',['B','C','D'],['92.8M → 190.3M','92.8M → 349.2M','92.8M → 589.8M'],'(b) Active-size gain, averaged over E')]:
        for target,label,color in zip(targets,labels,colors):
            g=results[(results.panel==panel)&(results.target==target)].sort_values('FLOPs')
            ax.plot(g.FLOPs,g.mean_gain,color=color,lw=2,label=label)
        ax.set_title(title,fontsize=11,pad=12);ax.set_xscale('log')
        ax.set_xticks([2e18,3e18,4e18],labels=[r'$2\times10^{18}$',r'$3\times10^{18}$',r'$4\times10^{18}$'])
        ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel('Training FLOPs')
        ax.axhline(0,color='.45',lw=.85);ax.grid(alpha=.18,lw=.6)
        ax.legend(loc='lower left' if panel=='expert' else 'upper left',frameon=False,fontsize=9)
    axes[0].set_ylabel('Mean loss reduction at equal compute (nats)')
    fig.text(.085,.11,'Equal means over 200 languages and four settings of the other capacity axis. Negative values mean the expansion has higher loss.',fontsize=8.5)
    fig.text(.085,.047,'All configurations share 1.77–4.45 × 10¹⁸ FLOPs. Interpolation is linear in log FLOPs; no extrapolation or fitted law.\nAt equal compute, larger active models receive fewer training tokens. Logical compute F = 6 N_act D_total; primary seed only.',fontsize=8.5,color='.35',linespacing=1.5)
    fig.savefig(output/'K200_mean_capacity_gains_compute.pdf');fig.savefig(output/'K200_mean_capacity_gains_compute.png',dpi=220);plt.close(fig)
    (output/'verification.json').write_text(json.dumps(dict(source_sha256=digest,compute_min=lo,compute_max=hi,grid_points=len(x),mean_checks=len(errors),max_error=max(errors),builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(results[results.FLOPs==4e18].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
