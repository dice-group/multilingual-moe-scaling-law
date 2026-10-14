"""Observed mean loss versus total tokens; each K uses its own language cohort."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from result_analysis.capacity_response.tables import (
    load_primary, sha256_file, RELEASE_MANIFEST_SHA256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = load_primary(args.root.resolve())
    cohorts = {int(k): sorted(g.language.unique()) for k,g in data.groupby('K')}
    assert set(cohorts) == {1,12,50,200}
    assert cohorts[1] == ['eng_Latn']
    for (k,readout),g in data.groupby(['K','readout_id']):
        assert len(g) == k and sorted(g.language) == cohorts[k], (k,readout)
    keys = ['trajectory_id','readout_id','ladder','N_act','E','K','D_total']
    means = data.groupby(keys,as_index=False).agg(
        mean_loss=('raw_nll','mean'), languages=('language','nunique'))
    assert len(means) == data.readout_id.nunique() == 400
    assert means.languages.eq(means.K).all()
    assert np.isfinite(means.mean_loss).all()
    assert means.groupby(['K','ladder','E']).ngroups == 64
    means.to_csv(out/'native_mean_learning_curves.csv',index=False)

    ladders = tuple('ABCD')
    experts = (1,4,16,32)
    colors = dict(zip(ladders,plt.get_cmap('inferno')([.06,.34,.62,.80])))
    markers = dict(zip(experts,('o','s','^','D')))
    sizes = means.groupby('ladder').N_act.first()/1e6
    fig,axes = plt.subplots(2,2,figsize=(12.8,11.4),sharex=True,sharey=False)
    fig.subplots_adjust(left=.09,right=.97,bottom=.18,top=.78,wspace=.15,hspace=.24)
    fig.suptitle('Mean validation loss as training tokens increase',x=.53,y=.97,
                 fontsize=20,weight='bold',color='#183249')
    fig.text(.5,.918,'Equal language means within each K; loss-axis ranges are specific to each panel',
             ha='center',fontsize=12,color='#526472')
    size_handles = [Line2D([],[],color=colors[l],lw=2,label=f'{sizes[l]:.0f}M') for l in ladders]
    expert_handles = [Line2D([],[],color='#526472',ls='none',marker=markers[e],
                            markerfacecolor='white',markersize=6,label=f'E = {e}') for e in experts]
    fig.legend(handles=size_handles,title=r'Active parameters $N_{\mathrm{act}}$ — colour',
               loc='upper center',bbox_to_anchor=(.29,.89),ncol=4,frameon=False,fontsize=10,title_fontsize=11)
    fig.legend(handles=expert_handles,title='Expert count — marker shape',
               loc='upper center',bbox_to_anchor=(.76,.89),ncol=4,frameon=False,fontsize=10,title_fontsize=11)
    for ax,k in zip(axes.flat,(1,12,50,200)):
        panel = means[means.K.eq(k)]
        low,high = panel.mean_loss.min(),panel.mean_loss.max()
        for l in ladders:
            for e in experts:
                g = panel[panel.ladder.eq(l)&panel.E.eq(e)].sort_values('D_total')
                assert len(g)>=2 and g.D_total.is_monotonic_increasing and g.D_total.is_unique
                ax.plot(g.D_total/1e9,g.mean_loss,color=colors[l],marker=markers[e],
                        lw=.95,ms=5.7,mfc='white',mew=1.2,alpha=1.)
        ax.set_title(f'K = {k}'+('  ·  English' if k==1 else f'  ·  mean of {k} languages'),fontsize=13,pad=9)
        ax.set_xscale('log');ax.set_xlim(.45,36)
        ax.set_ylim(low-.035,high+.035)
        ax.xaxis.set_major_locator(FixedLocator([.5,1,2,4,8,16,32]))
        ax.xaxis.set_major_formatter(FixedFormatter(['0.5','1','2','4','8','16','32']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2,lw=.6);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=10,labelbottom=True)
    fig.supxlabel('Total training tokens (billions; logarithmic scale)',y=.115,fontsize=12)
    fig.supylabel('Mean validation cross-entropy (nats; lower is better)',x=.022,fontsize=12)
    fig.text(.09,.069,'Markers are observed readouts; lines connect them. No fitted law or extrapolation. Primary seed only.',fontsize=10,color='#526472')
    fig.text(.09,.041,'Language populations differ between K panels, so cross-panel loss differences also reflect cohort composition.',fontsize=10,color='#526472')
    stem = 'Native_mean_loss_vs_tokens'
    fig.savefig(out/f'{stem}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/f'{stem}.png',dpi=180)
    with PdfPages(out/'Inferno_full_and_zoom.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
        pdf.savefig(fig)
        fig.suptitle('Mean validation loss: 4–32B token detail',x=.53,y=.97,
                     fontsize=20,weight='bold',color='#183249')
        for ax,k in zip(axes.flat,(1,12,50,200)):
            visible=means[means.K.eq(k)&means.D_total.ge(4e9)]
            ax.set_xlim(3.8,36)
            ax.set_ylim(visible.mean_loss.min()-.035,visible.mean_loss.max()+.035)
            ax.xaxis.set_major_locator(FixedLocator([4,8,16,32]))
            ax.xaxis.set_major_formatter(FixedFormatter(['4','8','16','32']))
        fig.savefig(out/f'{stem}_zoom.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(out/f'{stem}_zoom.png',dpi=180)
        pdf.savefig(fig)
    plt.close(fig)
    manifest = dict(source_root=str(args.root.resolve()),release_manifest_sha256=RELEASE_MANIFEST_SHA256,
                    builder_sha256=sha256_file(Path(__file__)),mean='Unweighted arithmetic mean of raw_nll across the K trained languages at each observed readout',
                    points=len(means),trajectories=64,cohorts=cohorts,
                    x_axis='Actual total training tokens; log scale',y_axis='Linear raw mean loss; per-panel ranges',
                    palette='inferno at 0.06, 0.34, 0.62, 0.80',zoom='4–32B tokens; full curves retained on first page',no_law_fitting=True,
                    outputs={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out/f'{stem}.pdf')


if __name__=='__main__':
    with plt.rc_context({'font.family':'DejaVu Sans','pdf.fonttype':42}):
        main()
