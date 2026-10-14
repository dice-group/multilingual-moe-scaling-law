"""Joint MoE Figure-2-style iso-compute profiles, using our full multilingual law."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from scipy.optimize import minimize_scalar
from result_analysis.compute_allocation.law_allocation import load,coefficients,values,optimum,sha

def main(bundle: Path, output: Path):
 rows,fits,_,_,shares,replay=load(bundle)
 output.mkdir(parents=True, exist_ok=False)
 fit=fits['K200_common_languages']['M6-ND']
 langs=sorted(rows.loc[rows.K.eq(12),'language'].unique());assert len(langs)==12
 computes=[2e18,8e18,32e18];shapes=['*','D','X'];es=[1,4,16,32]
 colors=dict(zip(es,plt.get_cmap('inferno')([.02,.32,.62,.82])))
 minima=[]
 for k in [12,50,200]:
  for f in computes:
   for e in es:
    c=coefficients(fit,shares,k,e,langs)
    n,d,_=optimum(c,f/1e18)
    loss=float(values(c,n,d).mean())
    numerical=minimize_scalar(lambda logd:float(values(c,f/(6*np.exp(logd))/1e8,np.exp(logd)/1e9).mean()),bounds=(np.log(d*1e9)-3,np.log(d*1e9)+3),method='bounded',options={'xatol':1e-12})
    np.testing.assert_allclose(np.exp(numerical.x),d*1e9,rtol=1e-5)
    np.testing.assert_allclose(6*(n*1e8)*(d*1e9),f,rtol=1e-12)
    probe=pd.DataFrame({'language':langs,'K':k,'E':e,'N_act':n*1e8,'D_total':d*1e9})
    for name in ['D_t','D_family','D_script','D_other']:
     probe[name]=probe.D_total*probe.language.map(shares.loc[k][name])
    np.testing.assert_allclose(replay.model_prediction('M6-ND',fit,probe).mean(),loss,atol=1e-10,rtol=0)
    minima.append(dict(K=k,E=e,F=f,N_act=n*1e8,D_total=d*1e9,loss=loss))
 mins=pd.DataFrame(minima)
 xlo=mins.D_total.min()*.68;xhi=mins.D_total.max()*1.7
 tokens=np.geomspace(xlo,xhi,500)
 tables=[]
 for k in [12,50,200]:
  for f in computes:
   for e in es:
    c=coefficients(fit,shares,k,e,langs)
    active=f/(6*tokens)
    tables.append(pd.DataFrame(dict(K=k,E=e,F=f,D_total=tokens,N_act=active,loss=values(c,active/1e8,tokens/1e9).mean(axis=1))))
 curves=pd.concat(tables,ignore_index=True)
 with plt.rc_context({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
  fig,axes=plt.subplots(1,3,figsize=(16/2.54,3.3),sharex=True,sharey=True)
  for ax,k in zip(axes,[12,50,200]):
   for f,marker in zip(computes,shapes):
    for e in es:
     curve=curves.loc[curves.K.eq(k)&curves.F.eq(f)&curves.E.eq(e)]
     point=mins.loc[mins.K.eq(k)&mins.F.eq(f)&mins.E.eq(e)].iloc[0]
     ax.plot(curve.D_total/1e9,curve.loss,color=colors[e],lw=.95)
     ax.scatter([point.D_total/1e9],[point.loss],marker=marker,color=colors[e],s=24 if marker=='*' else 15,zorder=4)
   ax.set(xscale='log',xlim=(xlo/1e9,xhi/1e9))
   ax.set_title(f'$K={k}$',fontsize=9,pad=7)
   ticks=[t for t in [1,2,4,8,16,32,64,128] if xlo/1e9<=t<=xhi/1e9]
   ax.set_xticks(ticks,[str(t) for t in ticks]);ax.xaxis.set_minor_locator(NullLocator())
   ax.grid(color='.92',lw=.45);ax.tick_params(labelsize=7,length=2.5,width=.5,pad=3)
   for spine in ax.spines.values():spine.set_linewidth(.5)
  # Shared linear loss range; no vertical shifting or per-budget normalization.
  axes[0].set_ylim(curves.loss.min()-.035,curves.loss.max()+.025)
  handles=[Line2D([],[],color=colors[e],lw=1.2,label=f'E = {e}') for e in es]
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),ncol=4,frameon=False,fontsize=7.5)
  handles=[Line2D([],[],color='.3',marker=m,ls='none',ms=5,label=label) for m,label in zip(shapes,[r'$2\times10^{18}$',r'$8\times10^{18}$',r'$3.2\times10^{19}$ FLOPs'])]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.035),ncol=3,frameon=False,fontsize=7.5)
  fig.subplots_adjust(left=.095,right=.99,top=.81,bottom=.28,wspace=.16)
  fig.supxlabel('Training tokens (billions)',fontsize=8.5,y=.145)
  fig.supylabel('Predicted unigram-adjusted loss (nats)',fontsize=8.5,x=.008)
  fig.text(.54,.012,'Common 12-language mean · Continuous full-law optima · No stored-parameter cap',ha='center',fontsize=6.5)
  for ext in ['pdf','png']:fig.savefig(output/f'jointmoe_style_isocompute.{ext}',dpi=300)
  plt.close(fig)
 mins.to_csv(output/'predicted_optima.csv',index=False);curves.to_csv(output/'predicted_curves.csv',index=False)
 render_active_axis(curves, mins, output)
 (output/'caption.md').write_text('Joint MoE Figure 2(a) visual structure applied to our frozen M6-ND law: predicted mean unigram-adjusted loss versus total training tokens, at three fixed compute budgets. Each budget has four expert-count curves, with a shared color per E. Marker shape identifies compute budget and marker position gives the continuous loss-minimizing size/token allocation. N_act=F/(6 D_total) along every curve. Panels K12/K50/K200 use the identical common-12 cohort, shared axes and fixed exposure shares from the longest dense ladder. K200 is excluded from coefficient fitting; the broader grid was inspected during model development. No observations are plotted and no additional models were trained. Continuous sizes and durations may exceed experimental support. No stored-parameter constraint is imposed, so this figure alone does not establish the matched-storage allocation claim. Minima were verified numerically and against the portable full-law predictor.\n')
 (output/'manifest.json').write_text(json.dumps(dict(generator_sha256=sha(Path(__file__)),bundle_manifest_sha256=sha(bundle/'manifest.json'),compute_budgets=computes,cohort=langs,token_limits=[xlo,xhi],minimum_count=len(mins)),indent=2)+'\n')
 print(mins.groupby('F').agg(min_tokens=('D_total','min'),max_tokens=('D_total','max')).to_string())

def render_active_axis(curves, mins, output: Path):
 for table in (curves,mins):
  np.testing.assert_allclose(6*table.N_act*table.D_total,table.F,rtol=1e-12)
 colors=dict(zip([1,4,16,32],plt.get_cmap('inferno')([.02,.32,.62,.82])))
 computes=[2e18,8e18,32e18];shapes=['*','D','X']
 xlo=curves.N_act.min()/1e6;xhi=curves.N_act.max()/1e6
 with plt.rc_context({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True,'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
  fig,axes=plt.subplots(1,3,figsize=(16/2.54,3.3),sharex=True,sharey=True)
  for ax,k in zip(axes,[12,50,200]):
   for f,marker in zip(computes,shapes):
    for e in colors:
     curve=curves.loc[curves.K.eq(k)&curves.F.eq(f)&curves.E.eq(e)].sort_values('N_act')
     point=mins.loc[mins.K.eq(k)&mins.F.eq(f)&mins.E.eq(e)].iloc[0]
     ax.plot(curve.N_act/1e6,curve.loss,color=colors[e],lw=.95)
     ax.scatter([point.N_act/1e6],[point.loss],marker=marker,color=colors[e],s=24 if marker=='*' else 15,zorder=4)
   ax.set(xscale='log',xlim=(xlo,xhi))
   ax.set_title(f'$K={k}$',fontsize=9,pad=7)
   ticks=[v for v in [20,100,500,2500] if xlo<=v<=xhi]
   ax.set_xticks(ticks,[str(v) for v in ticks]);ax.xaxis.set_minor_locator(NullLocator())
   ax.grid(color='.92',lw=.45);ax.tick_params(labelsize=7,length=2.5,width=.5,pad=3)
   for spine in ax.spines.values():spine.set_linewidth(.5)
  axes[0].set_ylim(curves.loss.min()-.035,curves.loss.max()+.025)
  handles=[Line2D([],[],color=colors[e],lw=1.2,label=f'E = {e}') for e in colors]
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),ncol=4,frameon=False,fontsize=7.5)
  handles=[Line2D([],[],color='.3',marker=m,ls='none',ms=5,label=label) for m,label in zip(shapes,[r'$2\times10^{18}$',r'$8\times10^{18}$',r'$3.2\times10^{19}$ FLOPs'])]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.035),ncol=3,frameon=False,fontsize=7.5)
  fig.subplots_adjust(left=.095,right=.96,top=.81,bottom=.28,wspace=.16)
  fig.supxlabel('Active parameters (millions)',fontsize=8.5,y=.145)
  fig.supylabel('Predicted unigram-adjusted loss (nats)',fontsize=8.5,x=.008)
  for ext in ['pdf','png']:fig.savefig(output/f'jointmoe_style_active_parameters.{ext}',dpi=300)
  plt.close(fig)
 (output/'active_axis_caption.md').write_text('Exact predictions and optima from the token-axis version, re-expressed against active parameter count in millions. Each curve holds compute fixed, so D_total=F/(6 N_act). Colors encode expert count, marker shapes encode compute budget. All three K panels score the same common 12 languages. Losses, budget values, points and y-axis range are unchanged. Curve domains differ between budgets because the original shared token range maps to different active-size ranges. Full-law predictions are continuous and may extrapolate size and duration beyond observed support. No stored-parameter cap or additional experimental evidence.\n')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--bundle', type=Path, required=True, help='Verified joint-story supplement')
 parser.add_argument('--output', type=Path, required=True, help='New output directory; must not exist')
 args=parser.parse_args()
 main(args.bundle, args.output)
