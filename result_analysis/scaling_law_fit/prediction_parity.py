"""Held-out observed-versus-predicted loss, at the manuscript's 16 cm width.

Run with bridge-train Python and PYTHONPATH=src:.; no refitting or paper edits.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, LogLocator, LogFormatterMathtext
from result_analysis.capacity_response.tables import load_primary, sha256_file
from result_analysis.compute_allocation.law_allocation import load

FOLDS = [('largest_N_act', '(a) Largest size held out'),
         ('E32', r'(b) $E=32$ held out'),
         ('K200_common_languages', r'(c) $K=200$ held out')]
KEYS = ['trajectory_id', 'readout_id', 'language']


def main(root, bundle, output, compact=False):
    output.mkdir(parents=True, exist_ok=False)
    source = root / "result_analysis/stage1_size_duration_ablations_v1/predictions.parquet"
    stem = "heldout_prediction_parity_compact" if compact else "heldout_prediction_parity"
    primary = load_primary(root)
    rows, fits, _, _, _, replay = load(bundle)
    predictions = pd.read_parquet(source).query('model == "M6-ND" and split == "test"').copy()
    assert len(predictions) == 18939 and not predictions.duplicated(['fold', *KEYS]).any()
    checked = predictions.merge(primary[KEYS + ['adjusted_loss']], on=KEYS,
                                 validate='many_to_one', suffixes=('', '_release'))
    assert len(checked) == len(predictions)
    np.testing.assert_allclose(checked.adjusted_loss, checked.adjusted_loss_release, atol=1e-12, rtol=0)
    metrics_path = root / 'result_analysis/stage1_reviewer_revision_v1/holdout_metrics.csv'
    expected = pd.read_csv(metrics_path).set_index('model').loc['M6-ND']
    report = {}
    for fold, _ in FOLDS:
        part = predictions.loc[predictions.fold.eq(fold)]
        assert part.trajectory_id.nunique() == 16
        assert (part.ladder.eq('D').all() if fold == 'largest_N_act' else
                part.E.eq(32).all() if fold == 'E32' else part.K.eq(200).all())
        source_rows = part[KEYS].merge(rows, on=KEYS, validate='one_to_one')
        regenerated = replay.model_prediction('M6-ND', fits[fold]['M6-ND'], source_rows)
        np.testing.assert_allclose(regenerated, part.prediction, atol=1e-9, rtol=0)
        error = (part.prediction-part.adjusted_loss)**2
        rmse = float(np.sqrt(error.groupby(part.trajectory_id).mean().mean()))
        np.testing.assert_allclose(rmse, expected[fold], atol=1e-12, rtol=0)
        report[fold] = dict(observations=len(part), trajectories=16, languages=part.language.nunique(),
                            trajectory_balanced_rmse=rmse,
                            replay_max_abs_error=float(np.max(abs(regenerated-part.prediction))))
    predictions[['fold', *KEYS, 'ladder','E','K','N_act','D_t','adjusted_loss','prediction']].to_csv(output/'heldout_points.csv',index=False)
    limits = (-5.65, -1.65)
    assert predictions[['adjusted_loss','prediction']].min().min() > limits[0]
    assert predictions[['adjusted_loss','prediction']].max().max() < limits[1]
    cmap = LinearSegmentedColormap.from_list('inferno_paper', plt.get_cmap('inferno')(np.linspace(.08,.85,256)))
    with plt.rc_context({'font.family':'STIXGeneral','mathtext.fontset':'stix',
                         'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig, axes = plt.subplots(1,3,figsize=(16/2.54,2.35 if compact else 2.65),sharex=True,sharey=True)
        bins=[]
        for ax,(fold,title) in zip(axes,FOLDS):
            part=predictions.loc[predictions.fold.eq(fold)]
            hb=ax.hexbin(part.adjusted_loss,part.prediction,gridsize=75,extent=(*limits,*limits),
                         mincnt=1,cmap=cmap,linewidths=0,rasterized=True)
            assert int(hb.get_array().sum()) == len(part)
            bins.append(hb)
            ax.plot(limits,limits,color='#555555',lw=.65,ls=(0,(4,3)),zorder=3)
            ax.set(xlim=limits,ylim=limits,aspect='equal')
            ax.set_title(title,fontsize=8.5,pad=7)
            stat=report[fold]
            ax.text(.035,.96,f'RMSE = {stat["trajectory_balanced_rmse"]:.5f}\n{len(part):,} observations',
                    transform=ax.transAxes,va='top',fontsize=7.2,linespacing=1.5)
            ax.xaxis.set_major_locator(MultipleLocator(1));ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.tick_params(labelsize=7.5,length=2.5,width=.5,pad=2)
            for spine in ax.spines.values():spine.set_linewidth(.5)
        maximum=max(float(h.get_array().max()) for h in bins)
        for h in bins:h.set_norm(LogNorm(vmin=1,vmax=maximum))
        fig.subplots_adjust(left=.085,right=.90,bottom=.20 if compact else .22,top=.91 if compact else .88,wspace=.14)
        cax=fig.add_axes([.923,.28,.012,.47])
        cb=fig.colorbar(bins[0],cax=cax)
        cb.set_label('Observations / bin',fontsize=7.2,labelpad=3)
        cb.locator=LogLocator(base=10,numticks=3);cb.formatter=LogFormatterMathtext();cb.update_ticks()
        cb.ax.tick_params(labelsize=6.5,length=2,width=.5);cb.outline.set_linewidth(.5)
        fig.supxlabel('Observed unigram-adjusted loss (nats)',fontsize=8.5,y=.02 if compact else .045)
        fig.supylabel('Predicted unigram-adjusted loss (nats)',fontsize=8.5,x=.008)
        for ext in ('pdf','png'):fig.savefig(output/f'{stem}.{ext}',dpi=300)
        plt.close(fig)
    caption = (r'\textbf{Held-out loss predictions of the full scaling law.} '
        r'Observed versus predicted unigram-adjusted loss for the largest active model size, '
        r'$E=32$, and $K=200$ holdouts. Each panel uses its own frozen M6-ND fit, '
        r'excluding the displayed trajectories from coefficient fitting. '
        r'All held-out language/checkpoint observations are shown; the $K=200$ panel evaluates '
        r'the 50 languages shared with $K=50$. The dashed diagonal denotes perfect prediction. '
        r'Hexagon color gives observation count with a shared logarithmic scale; it does not represent confidence. '
        r'Annotated RMSE balances trajectories and matches Table~\ref{tab:joint-story-validation}. '
        r'The 16 held-out trajectories per panel contain dependent checkpoints and language observations, '
        r'and the holdouts overlap. These are internal validation results: the grid was inspected during model development.')
    snippet = '\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'+stem+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:heldout-parity}\n\\end{figure*}\n'
    (output/'heldout_prediction_parity.tex').write_text(snippet)
    (output/'caption.md').write_text('Three-panel full-law parity plot. Linear, equal x/y scales; all held-out observations are included. Diagonal: perfect prediction. Inferno hexagons: raw counts (shared logarithmic color scale), not trajectory-balanced weights or confidence. Annotated RMSE is trajectory-balanced and matches Table 1. K200 scores the common50 cohort; other folds score 200 languages. No train observations, refitting, or extrapolated synthetic points. Observations and overlapping holdouts are dependent; validation is internal.\n')
    (output/'manifest.json').write_text(json.dumps({'model':'M6-ND','sources':{str(source.relative_to(root)):sha256_file(source),str(metrics_path.relative_to(root)):sha256_file(metrics_path),'bundle_manifest':sha256_file(bundle/'manifest.json')},'generator_sha256':sha256_file(Path(__file__)),'folds':report,'axis_limits':limits,'density_gridsize':75,'density_count_max':maximum,'width_cm':16},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.bundle.resolve(), args.output.resolve(), compact=args.compact)
