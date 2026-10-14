"""Near-matched stored-capacity contrasts: observed interpolation and frozen law."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext,NullLocator
from result_analysis.compute_allocation.law_allocation import load,coefficients,values,sha

def main(bundle: Path, output: Path):
 rows,fits,_,recipes,shares,replay=load(bundle)
 output.mkdir(parents=True, exist_ok=False)
 fit=fits['K200_common_languages']['M6-ND']
 langs=sorted(rows.loc[rows.K.eq(12),'language'].unique());assert len(langs)==12
 pairs=[('A/E16','B/E4'),('C/E32','D/E16')]
 colors=dict(zip([12,50,200],plt.get_cmap('inferno')([.32,.62,.82])))
 records=[];evidence=[];summaries=[]
 for pa,pb in pairs:
  members=[recipes.loc[recipes.config.eq(c)].iloc[0] for c in [pa,pb]]
  samples={}
  low=[];high=[]
  for k in colors:
   for m in members:
    sub=rows.loc[rows.K.eq(k)&rows.E.eq(m.E)&rows.ladder.eq(m.ladder)&rows.language.isin(langs)]
    cs=[sub.loc[sub.language.eq(lang)].sort_values('D_total') for lang in langs]
    samples[(k,m.config)]=cs
    low += [6*m.N_act*c.D_total.min() for c in cs]
    high += [6*m.N_act*c.D_total.max() for c in cs]
  lo=max(low);hi=min(high);assert lo<hi
  grid=np.geomspace(lo,hi,401)
  for k in colors:
   measured=[];predicted=[]
   for m in members:
    d=grid/(6*m.N_act);cs=samples[(k,m.config)]
    assert all(d.min()>=c.D_total.min()*(1-1e-12) and d.max()<=c.D_total.max()*(1+1e-12) for c in cs)
    measured.append(np.mean([np.interp(np.log(d),np.log(c.D_total),c.adjusted_loss) for c in cs],axis=0))
    predicted.append(values(coefficients(fit,shares,k,int(m.E),langs),m.N_act/1e8,d/1e9).mean(axis=1))
   delta=measured[0]-measured[1];law=predicted[0]-predicted[1]
   records.append(pd.DataFrame(dict(pair=pa+' vs '+pb,K=k,F=grid,observed_delta=delta,predicted_delta=law)))
   # Checkpoint anchors from either member; the other member is interpolated.
   for index,m in enumerate(members):
    cs=samples[(k,m.config)]
    checkpoints=np.unique(np.concatenate([6*m.N_act*c.D_total.to_numpy() for c in cs]))
    for f in checkpoints[(checkpoints>=lo)&(checkpoints<=hi)]:
     losses=[]
     for member in members:
      d=f/(6*member.N_act)
      losses.append(np.mean([np.interp(np.log(d),np.log(c.D_total),c.adjusted_loss) for c in samples[(k,member.config)]]))
     evidence.append(dict(pair=pa+' vs '+pb,K=k,F=f,delta=losses[0]-losses[1],anchor=m.config,marker='o' if index==0 else 's'))
  summaries.append(dict(pair=pa+' vs '+pb,mismatch_pct=100*(max(m.N_total for m in members)/min(m.N_total for m in members)-1),F_min=lo,F_max=hi))
 frame=pd.concat(records,ignore_index=True);points=pd.DataFrame(evidence)
 frame.to_csv(output/'curves.csv',index=False);points.to_csv(output/'checkpoint_anchors.csv',index=False)
 with plt.rc_context({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True,'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
  fig,axes=plt.subplots(1,2,figsize=(16/2.54,3.2),sharey=True)
  titles=['(a) $\\approx$0.38B stored parameters','(b) $\\approx$5.02B stored parameters']
  descriptions=['93M / E16 - 190M / E4','349M / E32 - 590M / E16']
  for ax,summary,title,desc in zip(axes,summaries,titles,descriptions):
   ax.axhspan(-.005,.005,color='.94',zorder=0)
   ax.axhline(0,color='.45',lw=.65)
   for k,color in colors.items():
    sub=frame.loc[frame.pair.eq(summary['pair'])&frame.K.eq(k)]
    ax.plot(sub.F,sub.observed_delta,color=color,lw=1.25)
    ax.plot(sub.F,sub.predicted_delta,color=color,lw=1.05,ls='--',alpha=.8)
    p=points.loc[points.pair.eq(summary['pair'])&points.K.eq(k)]
    for marker,g in p.groupby('marker'):
     ax.scatter(g.F,g.delta,marker=marker,s=11,facecolors='white',edgecolors=color,linewidths=.7,zorder=5)
   ax.set(xscale='log',xlim=(summary['F_min'],summary['F_max']))
   ax.set_title(title+'\n'+desc,fontsize=8,pad=7)
   ax.xaxis.set_major_formatter(LogFormatterMathtext());ax.xaxis.set_minor_locator(NullLocator())
   ticks=[t for t in [6e17,1e18,2e18,4e18,1e19,3e19,6e19] if summary['F_min']<=t<=summary['F_max']]
   ax.set_xticks(ticks,[rf'$10^{{{int(np.floor(np.log10(t)))}}}$' if np.isclose(t/10**int(np.floor(np.log10(t))),1) else rf'${t/10**int(np.floor(np.log10(t))):g}\times10^{{{int(np.floor(np.log10(t)))}}}$' for t in ticks])
   ax.tick_params(labelsize=7,length=2.5,width=.5,pad=3)
   ax.grid(axis='y',color='.92',lw=.45)
   for spine in ax.spines.values():spine.set_linewidth(.5)
   ax.text(.98,.04,f'Stored mismatch: {summary["mismatch_pct"]:.2f}%',transform=ax.transAxes,ha='right',fontsize=7,color='.4')
  lo=min(frame.observed_delta.min(),frame.predicted_delta.min());hi=max(frame.observed_delta.max(),frame.predicted_delta.max())
  axes[0].set_ylim(lo-.015,hi+.015)
  handles=[Line2D([],[],color=colors[k],lw=1.5,label=f'K = {k}') for k in colors]
  handles += [Line2D([],[],color='.4',lw=1.2,label='Observed interpolation'),Line2D([],[],color='.4',lw=1.1,ls='--',label='Full law')]
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1),ncol=5,frameon=False,fontsize=7.5,columnspacing=1.2)
  fig.subplots_adjust(left=.10,right=.96,top=.76,bottom=.27,wspace=.15)
  fig.supxlabel('Total training compute (FLOPs)',fontsize=8.5,y=.14)
  fig.supylabel('Loss difference: expert-heavy - active-heavy (nats)',fontsize=8.5,x=.008)
  for ext in ['pdf','png']:fig.savefig(output/f'matched_pairs_observed_and_law.{ext}',dpi=300)
  plt.close(fig)
 (output/'caption.md').write_text('Near-matched stored-parameter comparisons for A/E16 versus B/E4 (1.46% mismatch) and C/E32 versus D/E16 (0.33%). Difference is expert-heavy minus active-heavy common-12 mean adjusted loss. All K12/K50/K200 curves within each panel share the same compute support, intersected across both architectures and all languages. Solid curves use log-token interpolation of actual observed language losses; circle/square anchors correspond to checkpoints of the first/second architecture, with the counterpart interpolated, not independent matched experiments. Dashed curves use the frozen M6-ND K200-holdout law and fixed exposure shares. No law refitting or duration extrapolation in these comparisons. Gray band is ±0.005 nats, not confidence. The curves compare the specified pair only, not all eligible configurations. Predictions are internal development evidence.\n')
 (output/'manifest.json').write_text(json.dumps(dict(generator_sha256=sha(Path(__file__)),bundle_manifest_sha256=sha(bundle/'manifest.json'),pairs=summaries,curve_rows=len(frame),anchors=len(points)),indent=2)+'\n')
 print(json.dumps(summaries,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--bundle', type=Path, required=True, help='Verified joint-story supplement')
 parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
 args=parser.parse_args()
 main(args.bundle, args.output)
