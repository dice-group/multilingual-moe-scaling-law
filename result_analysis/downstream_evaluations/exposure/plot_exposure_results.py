"""Review figures from verified exposure-aware benchmark tables."""
from pathlib import Path
import argparse
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ES=[1,4,16,32]; X=np.arange(4)
C=dict(zip([1,12,50,200],matplotlib.colormaps['inferno']([.02,.32,.62,.82])))
STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180}
def style(ax):
    ax.set_xticks(X,[str(e) for e in ES]);ax.set_xlabel('Experts E');ax.grid(axis='y',alpha=.16,lw=.5);ax.set_axisbelow(True)
    ax.axhline(0,color='.45',lw=.7,zorder=1)
def save(fig,name,output):
    for ext in ['pdf','png']:fig.savefig(output/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def caption(fig,text):fig.text(.5,.015,text,ha='center',va='bottom',fontsize=8,color='.3')

def main(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(STYLE):
        d=pd.read_csv(source/'benchmark_exposure_language_table.csv')
        t=d.query('training_status=="trained" and K>1').groupby(['K','task','E']).normalized_accuracy.mean().mul(100)
        fig,axes=plt.subplots(1,3,figsize=(9,3.6),sharey=True)
        for ax,k in zip(axes,[12,50,200]):
            w=t.loc[k].unstack('E')[ES];g=w.subtract(w[1],axis=0)
            for _,r in g.iterrows():ax.plot(X,r.to_numpy(),color='.5',lw=.7,alpha=.35,marker='o',ms=2)
            mean=g.mean();ax.plot(X,mean.to_numpy(),color=C[k],lw=2,marker='o',ms=4,zorder=4)
            ax.annotate(f'{mean[32]:+.2f}',(3,mean[32]),xytext=(-4,8),textcoords='offset points',ha='right',color=C[k],weight='bold')
            cells=d.query('training_status=="trained" and K==@k and E==1').shape[0]
            ax.set_title(f'K = {k}  ·  {cells} task–language cells');style(ax)
        axes[0].set_ylabel('Accuracy change vs E1 (percentage points)')
        fig.legend([Line2D([],[],color='.45',lw=.7,alpha=.5),Line2D([],[],color=C[12],lw=2,marker='o',ms=4)],['Individual benchmarks','Equal-task mean (color denotes K)'],loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False,fontsize=8)
        fig.suptitle('Expert gains on trained languages',y=1.01,fontsize=12)
        caption(fig,'Same language exposure, active size and total tokens within each panel. Cohorts differ across panels.\nNormalized accuracy; six tasks. Lines connect measured configurations; no uncertainty intervals.')
        fig.subplots_adjust(top=.73,bottom=.28,wspace=.14)
        save(fig,'trained_language_expert_gains',output)

        s=pd.read_csv(source/'exposure_coverage_summary.csv')
        s=s.query('metric=="normalized_accuracy"')
        fig,axes=plt.subplots(1,2,figsize=(8,3.8))
        for k in [50,200]:
            g=s.query('K_low==12 and K_high==@k and cohort=="shared_trained"').set_index('E').loc[ES]
            for ax,col,sign in [(axes[0],'mean_loss_improvement',-1),(axes[1],'change_pp',1)]:
                ax.plot(X,sign*g[col].to_numpy(),color=C[k],lw=1.7,marker='o' if k==50 else 's',ms=4,label=f'K12 → K{k}');style(ax)
            for ax,col,sign,fmt in [(axes[0],'mean_loss_improvement',-1,'.3f'),(axes[1],'change_pp',1,'.2f')]:
                for e in [32]:
                    v=sign*g.loc[e,col];ax.annotate(format(v,fmt),(ES.index(e),v),xytext=(4 if e==1 else -4,7),textcoords='offset points',ha='left' if e==1 else 'right',fontsize=8,color=C[k])
        axes[0].set_title('(a) Validation-loss penalty');axes[0].set_ylabel('Increase in adjusted loss (lower is better)');axes[0].set_ylim(-.01,.255)
        axes[1].set_title('(b) Benchmark accuracy change');axes[1].set_ylabel('Change in percentage points (higher is better)');axes[1].set_ylim(-4.2,.8)
        axes[0].legend(frameon=False,fontsize=8,loc='upper left')
        fig.suptitle('Same original languages: loss and accuracy respond differently',fontsize=12,y=1.02)
        caption(fig,'Fixed original-K12 cohort: 38 task–language cells, six tasks; identical weights for loss and accuracy.\nRetained exposure: 51.32% at K50; 29.33% at K200. Each ratio is identical across E.')
        fig.subplots_adjust(top=.84,bottom=.27,wspace=.37)
        save(fig,'shared_languages_loss_vs_accuracy',output)

        fig,axes=plt.subplots(1,3,figsize=(9,3.6),sharey=True)
        styles={'shared_trained':('-', 'o','Original languages'),'newly_trained':('--','s','Newly trained languages'),'untrained_both':(':','^','Untrained in both')}
        for ax,(a,b) in zip(axes,[(12,50),(12,200),(50,200)]):
            for cohort,(ls,m,label) in styles.items():
                g=s.query('K_low==@a and K_high==@b and cohort==@cohort').set_index('E').loc[ES]
                color='.5' if cohort=='untrained_both' else C[b]
                ax.plot(X,g.change_pp.to_numpy(),color=color,ls=ls,marker=m,ms=3.5,lw=1.5)
            style(ax);ax.set_title(f'K{a} → K{b}')
        axes[0].set_ylabel('Accuracy change (percentage points)')
        fig.legend([Line2D([],[],color='.3' if c!='untrained_both' else '.5',ls=v[0],marker=v[1],ms=3.5) for c,v in styles.items()],[v[2] for v in styles.values()],loc='upper center',bbox_to_anchor=(.5,.9),ncol=3,frameon=False,fontsize=8)
        fig.suptitle('Coverage expansion: separate gains and costs',fontsize=12,y=1.02)
        caption(fig,'Fixed task–language identities within each curve; cohort sizes and task coverage differ.\nNormalized accuracy, equal-task means. Total tokens fixed; per-language exposure changes.')
        fig.subplots_adjust(top=.74,bottom=.27,wspace=.14)
        save(fig,'coverage_expansion_cohorts',output)
        inputs=['benchmark_exposure_language_table.csv','exposure_coverage_summary.csv']
        (output/'exposure_plot_receipt.json').write_text(json.dumps({'inputs':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in inputs},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':['trained_language_expert_gains','shared_languages_loss_vs_accuracy','coverage_expansion_cohorts'],'scope':'review figures; observed endpoints; no fitted curves or confidence intervals'},indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Output directory from exposure_analysis.py')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
    args=parser.parse_args()
    main(args.source, args.output)
