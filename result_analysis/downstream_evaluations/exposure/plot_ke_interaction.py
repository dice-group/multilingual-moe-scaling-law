"""Aggregate and task-wise K × E interactions on all benchmark-supported languages trained at each K."""
from pathlib import Path
import argparse
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

E=[1,4,16,32];KS=[12,50,200]
COLORS=dict(zip(KS,matplotlib.colormaps['inferno']([.32,.62,.82])))
MARKERS={12:'o',50:'s',200:'^'}
NAMES={'belebele_cloze':'Belebele','global_mmlu_cloze':'Global MMLU','okapi_hellaswag':'HellaSwag','uhura_arc_easy':'Uhura (ARC-Easy)','xcopa':'XCOPA','xstorycloze':'XStoryCloze'}
ORDER=['belebele_cloze','okapi_hellaswag','xstorycloze','xcopa','uhura_arc_easy','global_mmlu_cloze']
STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42}
def setup(ax):
    ax.set_xscale('log',base=2);ax.set_xticks(E,[str(e) for e in E]);ax.minorticks_off()
    ax.set_xlabel('Number of experts E');ax.grid(axis='y',alpha=.18,lw=.6);ax.set_axisbelow(True)
def draw(ax,frame):
    for k in KS:
        g=frame.loc[frame.K.eq(k)].set_index('E').loc[E]
        ax.plot(E,g.accuracy_percent,color=COLORS[k],marker=MARKERS[k],ms=4.2,lw=1.6,label=f'K = {k}')
    setup(ax)
def save(fig,stem,output,paper):
    if paper:
        stem += '_paper'
        for text in list(fig.texts): text.remove()
    for ext in ['pdf','png']:fig.savefig(output/f'{stem}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main(source: Path, output: Path, paper=False):
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(STYLE):
        if paper: plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
        source=source/'benchmark_exposure_language_table.csv';d=pd.read_csv(source)
        # Include all supported trained languages separately for each coverage setting.
        p=d.loc[d.K.isin(KS)&d.training_status.eq('trained')].copy()
        assert not p.duplicated(['K','E','task','language']).any()
        assert p.groupby(['K','task','language']).size().eq(4).all()
        assert p.groupby(['K','task','language']).D_t.nunique().eq(1).all()
        counts=p.loc[p.E.eq(1)].groupby('K').size().to_dict()
        assert counts=={12:38,50:99,200:172}
        task=p.groupby(['task','K','E']).agg(accuracy_percent=('normalized_accuracy',lambda x:100*x.mean()),languages=('language','size')).reset_index()
        agg=task.groupby(['K','E']).accuracy_percent.mean().reset_index()
        task.to_csv(output/'ke_interaction_by_benchmark.csv',index=False);agg.to_csv(output/'ke_interaction_aggregate.csv',index=False)
        handles=[Line2D([],[],color=COLORS[k],marker=MARKERS[k],lw=1.6,ms=4.2,label=f'K = {k}') for k in KS]
        fig,ax=plt.subplots(figsize=(3.25,2.5) if paper else (5.2,3.8));draw(ax,agg)
        ax.set_ylabel('Benchmark accuracy (%)');ax.set_title('' if paper else 'Language coverage × expert count',pad=12)
        ax.legend(handles=handles,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.0),fontsize=7 if paper else 8,handlelength=1.3,columnspacing=1)
        lo,hi=agg.accuracy_percent.min(),agg.accuracy_percent.max()
        ax.set_ylim(lo-.6,hi+1.5)
        fig.text(.5,.025,'All supported languages trained at each K · Equal mean over six benchmarks\nLanguage sets and per-language exposure differ across K; 32B total training tokens.',ha='center',fontsize=8,color='.3')
        fig.subplots_adjust(bottom=.2 if paper else .25,top=.97 if paper else .88,left=.17 if paper else .125)
        save(fig,'benchmark_ke_interaction',output,paper)
        fig,axes=plt.subplots(2,3,figsize=(6.3,4.6) if paper else (9,5.9))
        for ax,t in zip(axes.flat,ORDER):
            g=task.loc[task.task.eq(t)];draw(ax,g)
            counts_task=g.groupby('K').languages.first().loc[KS]
            ns=' / '.join(str(int(n)) for n in counts_task)
            ax.set_title(f'{NAMES[t]}\n{ns} languages',fontsize=8 if paper else 9)
            ax.set_ylabel('Accuracy (%)')
            lo,hi=g.accuracy_percent.min(),g.accuracy_percent.max();pad=max(.7,(hi-lo)*.16);ax.set_ylim(lo-pad,hi+pad)
        fig.suptitle('Language coverage × expert count, by benchmark',fontsize=12,y=.995)
        fig.legend(handles=handles,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1 if paper else .96))
        fig.text(.5,.015,'Language counts follow K12 / K50 / K200; all supported trained languages included. Panel y-ranges differ.\nPoints are measured configurations; connecting lines are guides. One seed, 100 questions per language/task.',ha='center',fontsize=8,color='.3')
        fig.subplots_adjust(top=.87 if paper else .82,bottom=.1 if paper else .16,hspace=.72 if paper else .52,wspace=.40 if paper else .34)
        save(fig,'benchmark_ke_interaction_per_task',output,paper)
        receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cohort':'all benchmark-supported trained languages at each K; fixed within K across E, changing across K','task_language_cells_by_K':counts,'metric':'normalized_accuracy','aggregate':'equal language mean within task, equal mean across six tasks','uncertainty':'not estimated; descriptive single-seed results','K1':'not included in this three-coverage multilingual comparison'}
        (output/'ke_interaction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(agg.pivot(index='E',columns='K',values='accuracy_percent').round(3).to_string())
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Output directory from exposure_analysis.py')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
    parser.add_argument('--paper', action='store_true', help='Render the compact paper layout')
    args=parser.parse_args()
    main(args.source, args.output, args.paper)
