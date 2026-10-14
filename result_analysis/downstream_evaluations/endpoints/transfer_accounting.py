"""Account for endpoint accuracy changes using sealed paired-score diagnostics.

No evaluation, fitting, metric selection, or modification of sealed inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(analysis: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    report = analysis / 'report_final'
    bindings = {}
    for receipt in [report / 'verification.json', analysis / 'examples/verification.json']:
        def visit(value):
            if isinstance(value, dict):
                if 'path' in value and 'sha256' in value:
                    bindings[str(Path(value['path']).resolve())] = value['sha256']
                for v in value.values():
                    visit(v)
            elif isinstance(value, list):
                for v in value:
                    visit(v)
        visit(json.loads(receipt.read_text()))
    sources = {}

    def read(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == bindings[str(path.resolve())], path
        sources[str(path.resolve())] = digest
        return pd.read_csv(path)

    means = read(report / 'tables/example_diagnostic_means.csv')
    primary = read(report / 'tables/headline_interactions.csv')
    seeds = read(report / 'tables/b_seed_interactions.csv')
    sensitivity = read(report / 'tables/metric_and_language_sensitivities.csv')
    languages = read(analysis / 'examples/language_examples.csv')
    flips = read(analysis / 'examples/paired_answer_changes.csv')
    classification = read(report / 'tables/classification_overview.csv')
    mapping = languages[['benchmark', 'language', 'corpus_language']].drop_duplicates()
    assert not mapping.duplicated(['benchmark', 'language']).any()
    flips = flips.merge(mapping, on=['benchmark', 'language'], validate='many_to_one')
    assert flips.corpus_language.notna().all()
    for name in ['corrected', 'broken']:
        flips[name + '_pp'] = 100 * flips[name] / flips.samples
    assert np.allclose(flips.corrected_pp - flips.broken_pp, flips.net_gain_pp, atol=1e-10)
    keys = ['benchmark', 'ladder', 'seed', 'K']
    rates = ['corrected_pp', 'broken_pp', 'net_gain_pp']
    # Match the primary language macro: varieties first, then corpus languages.
    macro = flips.groupby(keys + ['corpus_language'])[rates].mean().groupby(keys).mean().reset_index()
    outputs = []
    for (task, ladder, seed), group in means.groupby(['benchmark', 'ladder', 'seed']):
        if ladder not in ['B', 'D']:
            continue  # Paired example transitions were audited at B and D.
        if task not in set(flips.benchmark):
            continue  # External text likelihood has no answer decisions.
        cells = group.set_index(['E', 'K'])
        if not all(pair in cells.index for pair in [(1,12),(32,12),(1,200),(32,200)]):
            continue
        row = dict(benchmark=task, ladder=ladder, seed=int(seed))
        for k in [12, 200]:
            for metric in ['primary', 'accuracy_raw', 'accuracy_token', 'accuracy_char',
                           'gold_nll', 'wrong_nll', 'margin', 'margin_token', 'macro_f1']:
                a, b = cells.loc[(1,k),metric], cells.loc[(32,k),metric]
                if pd.notna(a) and pd.notna(b):
                    sign = -1 if metric.endswith('nll') else 1
                    factor = 100 if metric == 'primary' or metric.startswith('accuracy') or metric == 'macro_f1' else 1
                    row[f'{metric}_gain_K{k}'] = sign * factor * (b-a)
            part = macro[(macro.benchmark==task)&(macro.ladder==ladder)&(macro.seed==seed)&(macro.K==k)]
            assert len(part) == 1, (task, ladder, seed, k)
            for rate in rates:
                row[f'{rate}_K{k}'] = float(part.iloc[0][rate])
            assert abs(row[f'primary_gain_K{k}']-row[f'net_gain_pp_K{k}']) < 1e-10
        for metric in ['primary', 'accuracy_raw', 'accuracy_token', 'accuracy_char', 'gold_nll', 'wrong_nll', 'margin', 'margin_token', 'macro_f1']:
            if f'{metric}_gain_K12' in row:
                row[metric+'_interaction'] = row[f'{metric}_gain_K200']-row[f'{metric}_gain_K12']
        for rate in ['corrected_pp', 'broken_pp']:
            row[rate+'_change'] = row[rate+'_K200']-row[rate+'_K12']
        assert abs(row['primary_interaction']-(row['corrected_pp_change']-row['broken_pp_change'])) < 1e-10
        ref = seeds if ladder == 'B' else primary
        ref = ref[(ref.benchmark==task)&(ref.ladder==ladder)&(ref.seed==seed)]
        assert len(ref)==1 and abs(row['primary_interaction']-float(ref.iloc[0].interaction))<1e-10
        outputs.append(row)
    accounting = pd.DataFrame(outputs)
    assert len(accounting)==48  # 12 accuracy configurations x (D + three B seeds)
    accounting.to_csv(output / 'score_and_transition_accounting.csv', index=False)
    macro.to_csv(output / 'language_macro_transitions.csv', index=False)
    sensitivity.to_csv(output / 'metric_and_cohort_sensitivities.csv', index=False)
    classification.to_csv(output / 'classification_overview.csv', index=False)

    lines = ['# Benchmark transfer: exact score accounting', '',
             'All inputs are hash-checked against the completed analysis. Fixed benchmark-supported K12 languages only; varieties are collapsed before equal-language aggregation.', '',
             'Accuracy gain = corrected fraction minus broken fraction. This is exact accounting of decision changes, not identification of a training mechanism.', '',
             '## Largest-size endpoint diagnostics', '',
             '| Task | Corrected K12 / K200 (pp) | Broken K12 / K200 (pp) | Expert gain K12 / K200 (pp) | Primary interaction | Raw interaction | Token interaction |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in outputs:
        if row['ladder']!='D': continue
        def pair(key): return f"{row[key+'_K12']:.3f} / {row[key+'_K200']:.3f}"
        lines.append(f"| {row['benchmark']} | {pair('corrected_pp')} | {pair('broken_pp')} | {pair('primary_gain')} | {row['primary_interaction']:+.3f} | {row['accuracy_raw_interaction']:+.3f} | {row['accuracy_token_interaction']:+.3f} |")
    lines += ['', '## Interpretation and unresolved causes', '',
              '- HellaSwag: compare correct-answer and mean-distractor NLL gains separately from the margin against the strongest distractor. These are different statistics. Mean margins do not determine accuracy: accuracy depends on which individual decisions cross the boundary (and the frozen tie rule).',
              '- XStoryCloze: raw and normalized scoring can give different interaction signs. This establishes metric sensitivity, not which metric is intrinsically correct. Keep the frozen character-normalized primary result.',
              '- MultiBLiMP: positive accuracy interaction can coexist with negative mean-margin interaction. Apply the same interpretive standard to supporting and opposing tasks.',
              '- SIB/TAXI: majority baselines, F1, and label concentration diagnose observed behavior. They do not establish that insufficient size or a particular prompt caused it.',
              '- Non-English and score-normalization sensitivities are preserved for every family in the accompanying table. Fixed language cohorts rule out changing cohort membership as an explanation for the primary interaction.',
              '- Final endpoints match total tokens, not target-language exposure. A direct test of the matched-exposure benchmark interaction requires additional evaluated horizons. No such inference is made from these endpoints.',
              '- Exact causal separation of exposure dilution, cross-language interference, model size, and prompting is not identified by this accounting. Controlled interventions are needed; do not relabel adverse results as benchmark bugs.',
              '', 'The new evidence extends the paper from in-domain predictive loss to external text loss and selected linguistic discrimination, while locating the boundary at task-specific decision behavior. All evaluated tasks remain reported.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(9,2.8),layout='constrained')
    colors=plt.get_cmap('inferno')([.25,.65])
    d=accounting[(accounting.ladder=='D')&(accounting.benchmark=='okapi_hellaswag')].iloc[0]
    for j,k in enumerate([12,200]):
        axes[0].bar(np.arange(2)+(j-.5)*.32,[d[f'gold_nll_gain_K{k}'],d[f'wrong_nll_gain_K{k}']],.32,color=colors[j],label=f'K={k}')
        axes[1].bar(np.arange(2)+(j-.5)*.32,[d[f'corrected_pp_K{k}'],d[f'broken_pp_K{k}']],.32,color=colors[j])
    axes[0].set_xticks([0,1],['Correct answer','Mean distractor']);axes[0].set_ylabel('E1−E32 NLL (nats/token)');axes[0].set_title('(a) Likelihood improves');axes[0].legend(frameon=False)
    axes[1].set_xticks([0,1],['Corrected','Broken']);axes[1].set_ylabel('Fraction of items (percentage points)');axes[1].set_title('(b) Decisions change both ways')
    axes[2].bar([0,1],[d.primary_gain_K12,d.primary_gain_K200],color=colors)
    axes[2].set_xticks([0,1],['K=12','K=200']);axes[2].set_ylabel('E32−E1 accuracy (percentage points)');axes[2].set_title('(c) Net accuracy gain stays positive')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for ext in ['pdf','png']: fig.savefig(output/f'hellaswag_transfer_accounting.{ext}',dpi=180)
    plt.close(fig)
    (output/'transfer_accounting_source.py').write_bytes(Path(__file__).read_bytes())
    artifacts={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'verification.json').write_text(json.dumps({'status':'passed','contrasts':len(accounting),'checks':['input hashes','within-language flip identity','variety then language macro','all 48 canonical interactions','gain = corrected minus broken'], 'sources':sources,'outputs':artifacts},indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.analysis,args.output)
