"""Export saved final-law evidence without training, fitting, or modifying sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

MODEL = 'RN12-stored-common-K'
PARENT = 'RN14-stored-reuse'
FOLDS = ('largest_N_act', 'E32', 'K200_common_languages', 'all_grid')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def export(selected, parent, output):
    selected, parent, output = (Path(p).resolve() for p in (selected, parent, output))
    for source in (selected, parent):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('delivery must be separate from original evidence')
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(root, name, label):
        path = root / name
        sources[label] = digest(path)
        return json.loads(path.read_text())

    def copy(root, name, target):
        path = root / name
        sources[target] = digest(path)
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    coefficients = read(selected, 'report/coefficients.json', 'original_selected_coefficients')
    plan = read(selected, 'plan.json', 'original_plan')
    inputs = read(selected, 'inputs/manifest.json', 'original_input_manifest')
    fits, certification = [], []
    for row in coefficients['fits']:
        if row['model'] != MODEL or row['arm'] != 'lbfgsb_box':
            raise ValueError('unexpected model/optimizer arm in final selected evidence')
        result = row['result']
        if not result['certified'] or not result['complete']:
            raise ValueError('uncertified or incomplete selected fit')
        receipt_path = selected / row['receipt']
        receipt = json.loads(receipt_path.read_text())
        if receipt != result:
            raise ValueError('coefficient report disagrees with original certification')
        fits.append({key: row[key] for key in ('model', 'fold', 'arm')} |
                    {'selected': result['selected']})
        keys = ('certified', 'complete', 'delta', 'bounds', 'equalities', 'effective_exponents',
                'independent_training_replay_max_gap', 'initial_starts', 'numerical_policy',
                'positive_effective_exponents', 'regularization', 'error', 'heldout_scores_computed')
        certification.append({'model': MODEL, 'fold': row['fold'], 'arm': row['arm'],
            'original_receipt_sha256': digest(receipt_path),
            **{key: result[key] for key in keys},
            'selected_kkt': result['selected']['kkt'], 'selected_objective': result['selected']['objective']})
    if sorted(row['fold'] for row in fits) != sorted(FOLDS):
        raise ValueError('missing or duplicate selected folds')
    write(output/'coefficients.json', {'fits': fits})
    write(output/'certification.json', {'scope': 'Extracted original numerical certification; no recertification or new fitting.',
                                       'fits': certification})
    membership_keys = ('kind', 'score_languages', 'train_readout_ids', 'test_readout_ids',
                       'train_trajectory_ids', 'test_trajectory_ids', 'membership_sha256')
    write(output/'memberships.json', {fold: {key: value for key, value in inputs['memberships'][fold].items()
            if key in membership_keys} for fold in FOLDS})
    write(output/'model_contract.json', {
        'model': MODEL, 'shared_coefficients': 12, 'language_coefficients': ['L_inf', 'A', 'B'],
        'expanded_coordinate_order': [b['name'] for b in certification[0]['bounds']],
        'coordinate_equalities': [[6, 7], [8, 9]],
        'normalization': {'active_parameters': 1e8, 'total_parameters': 1e8, 'training_tokens': 1e9,
                          'gK': 'max(log(K/12), 0) / log(200/12)',
                          'expert_interpolation': 'log(E)/log(32)'},
        'numerical_contract': plan['numerical_contract'], 'report_scope': plan['report_scope'],
        'evaluation': inputs['evaluation'], 'evidence': inputs['evidence'],
        'scope': 'Saved fixed-law replay. Structured development holdouts; L26 is a developmental stress test.',
        'allocation_predictor': 'M6-ND (separate composition-aware law); no allocation is reassigned to RN12.',
    })
    for name in ('predictions.parquet', 'metrics.csv', 'comparison_lbfgsb_box.csv'):
        copy(selected, 'report/'+name, name)
    for name in ('grid.parquet', 'endpoint.parquet'):
        copy(selected, 'inputs/scoring/'+name, 'scoring/'+name)
    # Keep only the compact parent subset needed for the published mixture check.
    all_parent = read(parent, 'recovery/v2/report/coefficients.json', 'original_parent_coefficients')
    parent_fits = [{key: r[key] for key in ('model', 'fold', 'arm')} |
                   {'selected': r['result']['selected']} for r in all_parent['fits']
                   if r['model'] == PARENT and r['arm'] == 'lbfgsb_box' and r['fold'] in FOLDS]
    if sorted(r['fold'] for r in parent_fits) != sorted(FOLDS):
        raise ValueError('missing or duplicate parent folds')
    write(output/'composition/parent_coefficients.json', {'fits': parent_fits})
    old = parent/'review/fresh_prediction_checks_v2'
    composition = pd.read_parquet(old/'composition_inputs.parquet')
    sources['original_composition_inputs'] = digest(old/'composition_inputs.parquet')
    # inventory_source is the only author-path column; all scientific values remain unchanged.
    composition['inventory_source'] = composition.inventory_source.map(
        lambda value: 'source:' + hashlib.sha256(str(value).encode()).hexdigest())
    composition.to_parquet(output/'composition/inputs.parquet', index=False)
    reference = pd.read_csv(old/'composition_fit_metrics.csv')
    sources['original_parent_composition_metrics'] = digest(old/'composition_fit_metrics.csv')
    reference[reference.model.eq(PARENT)].to_csv(output/'composition/parent_metrics.csv', index=False)
    text = '''# Selected repetition-aware law: saved evidence

This versioned package contains RN12-stored-common-K, not the older RN14 law.
Predictions, grid/endpoint inputs and reference metrics are original saved bytes;
coefficient and certification JSONs are explicit numerical subsets of the original
reports. Composition inputs retain all scientific columns; only inventory paths
are replaced with opaque labels. The manifest binds every delivered file.

Run from the repository root (CPU only, no refitting):

```sh
PYTHONPATH=src:. python -m result_analysis.scaling_law_fit.submission_replay \\
  --output outputs/submission_replay
```

This checks literal coefficient predictions, scoring membership, RMSE/R2 and
composition transfer for the selected law, and the saved JointMoE D/Dt baselines.
L26 is developmental evidence; no untouched validation is implied. Allocation
uses the separately named M6-ND law. Original optimizer restart histories and
private runtime metadata are not needed for this saved-data replay.
'''
    (output/'README.md').write_text(text)
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'manifest.json', {'schema': 'stage1-selected-law/v1', 'model': MODEL,
        'arm': 'lbfgsb_box', 'folds': list(FOLDS), 'files': files, 'original_sources_sha256': sources,
        'exporter_sha256': digest(__file__), 'new_training': False, 'new_fitting': False,
        'provenance': 'Original source digests are lineage only; files binds the delivered numerical projection.'})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('selected', 'parent', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(f'Exported {len(export(args.selected, args.parent, args.output))} files')


if __name__ == '__main__':
    main()
