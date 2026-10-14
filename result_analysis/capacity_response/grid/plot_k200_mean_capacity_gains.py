"""Equal-configuration average capacity gains at K200 on common token support."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    output.mkdir(parents=True,exist_ok=False)
    path=source/'native_mean_learning_curves.csv'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==json.loads((source/'manifest.json').read_text())['outputs'][path.name]
    d=pd.read_csv(path).query('K==200').copy()
    assert len(d)==100 and d.languages.eq(200).all()
    common=set.intersection(*[set(g.D_total) for _,g in d.groupby(['ladder','E'])])
    d=d[d.D_total.isin(common)]
    assert len(common)==5 and len(d)==80
    p=d.pivot(index=['ladder','D_total'],columns='E',values='mean_loss')
    er=[]
    for e in [4,16,32]:
        r=(p[1]-p[e]).rename('gain').reset_index();r['E']=e;er.append(r)
    expert=pd.concat(er,ignore_index=True)
    expert_mean=expert.groupby(['E','D_total'],as_index=False).agg(mean_gain=('gain','mean'),configurations=('gain','size'))
    p=d.pivot(index=['E','D_total'],columns='ladder',values='mean_loss')
    nr=[]
    for n in ['B','C','D']:
        r=(p.A-p[n]).rename('gain').reset_index();r['ladder']=n;nr.append(r)
    active=pd.concat(nr,ignore_index=True)
    active_mean=active.groupby(['ladder','D_total'],as_index=False).agg(mean_gain=('gain','mean'),configurations=('gain','size'))
    assert expert_mean.configurations.eq(4).all() and active_mean.configurations.eq(4).all()
    # Independent difference-of-means check, distinct from averaging paired gains.
    em=d.groupby(['E','D_total']).mean_loss.mean();nm=d.groupby(['ladder','D_total']).mean_loss.mean()
    errors=[abs(r.mean_gain-(em[1,r.D_total]-em[r.E,r.D_total])) for r in expert_mean.itertuples()]
    errors += [abs(r.mean_gain-(nm['A',r.D_total]-nm[r.ladder,r.D_total])) for r in active_mean.itertuples()]
    assert len(errors)==30 and max(errors)<1e-12
    expert_mean.to_csv(output/'mean_expert_gains.csv',index=False)
    active_mean.to_csv(output/'mean_active_size_gains.csv',index=False)
    expert.to_csv(output/'individual_expert_gains.csv',index=False)
    active.to_csv(output/'individual_active_size_gains.csv',index=False)
    colors=plt.colormaps['inferno']([.2,.5,.78])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.78,wspace=.12)
    fig.suptitle('Mean capacity gains during multilingual training · K = 200',fontsize=15,y=.975)
    for e,color in zip([4,16,32],colors):
        g=expert_mean[expert_mean.E==e].sort_values('D_total')
        axes[0].plot(g.D_total/1e9,g.mean_gain,color=color,marker='o',ms=4,lw=1.9,label=f'E1 → E{e}')
    for n,label,color in zip(['B','C','D'],['190.3M','349.2M','589.8M'],colors):
        g=active_mean[active_mean.ladder==n].sort_values('D_total')
        axes[1].plot(g.D_total/1e9,g.mean_gain,color=color,marker='o',ms=4,lw=1.9,label=f'92.8M → {label}')
    for ax,title in zip(axes,['(a) Expert gain, averaged over active sizes','(b) Active-size gain, averaged over E']):
        ax.set_title(title,fontsize=11,pad=12)
        ax.set_xscale('log',base=2);ax.set_xticks([.5,1,2,4,8],labels=['0.5','1','2','4','8'])
        ax.set_xlabel('Total training tokens (billions)');ax.set_ylim(0,.51)
        ax.grid(alpha=.18,lw=.6);ax.legend(loc='upper left',frameon=False,fontsize=9)
    axes[0].set_ylabel('Mean loss reduction (nats)')
    fig.text(.08,.105,'Equal-weight mean across all 200 languages, then equally across four active sizes (a) or four E values (b).',fontsize=9)
    fig.text(.08,.045,'Same configurations at every token budget; common observed range 0.5–8B tokens. Lines connect observed contrasts.\nFixed-token comparisons, not equal-compute or equal-storage investments. Primary seed only.',fontsize=8.5,color='.35',linespacing=1.5)
    fig.savefig(output/'K200_mean_capacity_gains.pdf');fig.savefig(output/'K200_mean_capacity_gains.png',dpi=220);plt.close(fig)
    (output/'verification.json').write_text(json.dumps(dict(source_sha256=digest,configurations_per_mean=4,common_token_budgets=sorted(common),
        mean_checks=len(errors),max_error=max(errors),builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(expert_mean[expert_mean.D_total==max(common)].to_string(index=False))
    print(active_mean[active_mean.D_total==max(common)].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
