"""Verify the delivered RN12 and JointMoE evidence without original campaigns."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .jointmoe_token_contrast import literal
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.replay import predict

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'RN12-stored-common-K'
PRIMARY = ('largest_N_act', 'E32', 'K200_common_languages')
FOLDS = (*PRIMARY, 'all_grid')
KEYS = ['trajectory_id', 'readout_id', 'language']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_manifest(root, schema=None):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if schema:
        require(manifest.get('schema') == schema, 'unexpected evidence schema')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path != root, f'unsafe evidence path: {name}')
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'file hash mismatch: {name}')
    return manifest


def align(saved, measured):
    """Fail on duplicate, missing, extra, or response-mismatched scoring rows."""
    require(not saved.duplicated(KEYS).any() and not measured.duplicated(KEYS).any(),
            'duplicate scoring identities')
    joined = saved.merge(measured, on=KEYS, how='outer', validate='one_to_one',
                         suffixes=('_saved', ''), indicator=True)
    require(joined._merge.eq('both').all(), 'scoring membership differs from frozen fold')
    np.testing.assert_allclose(joined.adjusted_loss_saved, joined.adjusted_loss, atol=1e-12, rtol=0)
    return joined.drop(columns='_merge')


def scores(rows, *, endpoint=False):
    require(len(rows) > 0, 'empty scoring population')
    require(np.isfinite(rows[['prediction', 'adjusted_loss']].to_numpy()).all(), 'nonfinite score')
    if endpoint:
        require(not rows.language.duplicated().any(), 'endpoint must have one row per language')
        weights = np.ones(len(rows))
    else:
        weights = 1/rows.groupby('trajectory_id').trajectory_id.transform('size').to_numpy()
    weights /= weights.sum()
    y = rows.adjusted_loss.to_numpy()
    err = rows.prediction.to_numpy()-y
    mse = float(np.dot(weights, err**2))
    variance = float(np.dot(weights, (y-np.dot(weights, y))**2))
    require(variance > 0, 'R2 requires nonconstant response')
    return dict(rows=len(rows), rmse=float(np.sqrt(mse)), r2=1-mse/variance)


def fold_rows(grid, endpoint, membership, case, *, all_grid=False):
    if case == 'L26':
        if all_grid:
            return endpoint.copy()
        return endpoint[endpoint.language.isin(membership['score_languages'])].copy()
    require(case == 'grid', 'unknown scoring case')
    train, test = set(membership['train_trajectory_ids']), set(membership['test_trajectory_ids'])
    require(not train & test, 'overlapping training/evaluation trajectories')
    rows = grid[grid.trajectory_id.isin(test) & grid.language.isin(membership['score_languages'])]
    if membership['test_readout_ids']:
        rows = rows[rows.readout_id.isin(membership['test_readout_ids'])]
    return rows.copy()


def verify_selected(root):
    root = Path(root)
    manifest = verify_manifest(root, 'stage1-selected-law/v1')
    require(manifest['model'] == MODEL and tuple(manifest['folds']) == FOLDS, 'wrong selected model/folds')
    fits = json.loads((root/'coefficients.json').read_text())['fits']
    require(sorted(f['fold'] for f in fits) == sorted(FOLDS), 'missing or duplicate fits')
    require(all(f['model'] == MODEL and f['arm'] == 'lbfgsb_box' for f in fits), 'wrong law/arm')
    certs = {f['fold']: f for f in json.loads((root/'certification.json').read_text())['fits']}
    require(set(certs) == set(FOLDS), 'missing certification')
    membership = json.loads((root/'memberships.json').read_text())
    grid = pd.read_parquet(root/'scoring/grid.parquet')
    endpoint = pd.read_parquet(root/'scoring/endpoint.parquet')
    saved = pd.read_parquet(root/'predictions.parquet')
    require(saved.model.eq(MODEL).all() and saved.arm.eq('lbfgsb_box').all(), 'wrong prediction attribution')
    require(set(saved.fold) == set(FOLDS), 'wrong prediction folds')
    references = pd.read_csv(root/'metrics.csv').set_index(['fold', 'case'])
    require(not references.index.duplicated().any(), 'duplicate reference metrics')
    records, max_errors = [], []
    for fit in fits:
        fold, record = fit['fold'], fit['selected']
        cert = certs[fold]
        require(cert['certified'] and cert['complete'], 'uncertified selected coefficients')
        x = np.asarray(record['coordinates'])
        bounds = cert['bounds']
        require(len(x) == len(bounds), 'coefficient dimensionality differs')
        free = sum(b['lower'] != b['upper'] for b in bounds)-len(cert['equalities'])
        require(free == 12, 'selected law must have 12 independent shared coefficients')
        for a, b in cert['equalities']:
            require(x[a] == x[b], 'tied coverage coefficients disagree')
        for value, bound in zip(x, bounds):
            require(bound['lower']-1e-12 <= value <= bound['upper']+1e-12, 'coefficient outside frozen bound')
        for case, part in saved[saved.fold.eq(fold)].groupby('case'):
            measured = fold_rows(grid, endpoint, membership[fold], case, all_grid=fold == 'all_grid')
            joined = align(part, measured)
            actual = predict(MODEL, record, joined)
            np.testing.assert_allclose(actual, joined.prediction, atol=1e-10, rtol=0)
            max_errors.append(float(np.max(abs(actual-joined.prediction))))
            score = scores(joined, endpoint=case == 'L26')
            ref = references.loc[(fold, case)]
            require(score['rows'] == ref['rows'] == ref['expected_rows'], 'score population count differs')
            np.testing.assert_allclose(score['rmse'], ref.rmse, atol=1e-12, rtol=0)
            records.append(dict(model=MODEL, fold=fold, case=case, **score))
    result = pd.DataFrame(records)
    require(set(zip(result.fold, result.case)) == set(references.index), 'missing or extra metric cells')
    comparison = pd.read_csv(root/'comparison_lbfgsb_box.csv').set_index('model').loc[MODEL]
    for fold in PRIMARY:
        np.testing.assert_allclose(result.query('fold == @fold and case == "grid"').rmse.item(),
                                   comparison[fold], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.query('fold == "all_grid"').rmse.item(), comparison.L26_all_grid,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.query('case == "grid"').rmse.mean(), comparison.mean_primary_rmse,
                               atol=1e-12, rtol=0)
    return result, max(max_errors), (grid, endpoint, membership)


def verify_composition(root):
    root = Path(root)
    rows = pd.read_parquet(root/'composition/inputs.parquet')
    records = []
    for path in (root/'coefficients.json', root/'composition/parent_coefficients.json'):
        for fit in json.loads(path.read_text())['fits']:
            if fit['fold'] not in PRIMARY:
                continue
            p = rows.copy()
            p['prediction'] = predict(fit['model'], fit['selected'], p)+p.old_unigram_nll-p.unigram_nll
            for (seed, horizon), g in p.groupby(['seed', 'horizon']):
                wide = g.pivot(index=['E', 'language'], columns='arm', values=['prediction', 'adjusted_loss'])
                error = (wide.prediction.diverse-wide.prediction.compact)-(wide.adjusted_loss.diverse-wide.adjusted_loss.compact)
                records.append(dict(model=fit['model'], fold=fit['fold'], seed=seed, horizon=horizon,
                    absolute_rmse=float(np.sqrt(np.mean((g.prediction-g.adjusted_loss)**2))),
                    contrast_rmse=float(np.sqrt(np.mean(error**2)))))
    result = pd.DataFrame(records)
    old = pd.read_csv(root/'composition/parent_metrics.csv')
    keys = ['model', 'fold', 'seed', 'horizon']
    check = result[result.model.eq('RN14-stored-reuse')].merge(old, on=keys, how='outer',
                validate='one_to_one', suffixes=('', '_old'), indicator=True)
    require(len(check) == 54 and check._merge.eq('both').all(), 'parent composition population differs')
    for field in ('absolute_rmse', 'contrast_rmse'):
        np.testing.assert_allclose(check[field], check[field+'_old'], atol=1e-10, rtol=0)
    return result


def verify_baselines(root, grid, endpoint, memberships):
    root = Path(root)
    verify_manifest(root)
    fits = json.loads((root/'coefficients.json').read_text())['fits']
    saved = pd.read_parquet(root/'predictions.parquet')
    reference = pd.read_csv(root/'comparison.csv').set_index('token')
    require(set(saved.token) == {'D', 'Dt'} and set(saved.fold) == set(FOLDS), 'wrong baseline cells')
    records, errors = [], []
    for (fold, token), part in saved.groupby(['fold', 'token']):
        measured = fold_rows(grid, endpoint, memberships[fold], 'L26' if fold == 'all_grid' else 'grid',
                             all_grid=fold == 'all_grid')
        joined = align(part, measured)
        fs = [f for f in fits if f['fold'] == fold and f['token'] == token]
        require(len({f['language'] for f in fs}) == len(fs), 'duplicate baseline coefficient set')
        require({f['language'] for f in fs} == set(joined.language), 'baseline calibration set differs')
        for fit in fs:
            require(fit['certified'], 'uncertified baseline fit')
            rows = joined[joined.language.eq(fit['language'])]
            packet = dict(N=rows.N_act.to_numpy(), E=rows.E.to_numpy(),
                D=np.column_stack([rows.D_t, rows.D_total-rows.D_t]), training_minimum=fit['training_minimum'])
            actual = literal(fit['selected']['coordinates'], packet, token)-fit['unigram_nll']
            np.testing.assert_allclose(actual, rows.prediction, atol=1e-10, rtol=0)
            errors.append(float(np.max(abs(actual-rows.prediction))))
        score = scores(joined, endpoint=fold == 'all_grid')
        np.testing.assert_allclose(score['rmse'], reference.loc[token, fold], atol=1e-12, rtol=0)
        records.append(dict(model='JointMoE', token=token, fold=fold, **score))
    result = pd.DataFrame(records)
    require(set(zip(result.fold, result.token)) == {(fold, token) for fold in FOLDS for token in ('D', 'Dt')},
            'missing baseline metric cells')
    for token in ('D', 'Dt'):
        np.testing.assert_allclose(result[result.token.eq(token) & result.fold.isin(PRIMARY)].rmse.mean(),
                                   reference.loc[token, 'mean_primary_rmse'], atol=1e-12, rtol=0)
    return result, max(errors)


def run(selected, baselines, output):
    selected, baselines, output = (Path(p).resolve() for p in (selected, baselines, output))
    require(not output.exists(), 'output must be a fresh directory')
    for source in (selected, baselines):
        require(not output.is_relative_to(source) and not source.is_relative_to(output), 'output overlaps evidence')
    metrics, error, features = verify_selected(selected)
    composition = verify_composition(selected)
    baseline_scores, baseline_error = verify_baselines(baselines, *features)
    output.mkdir(parents=True)
    metrics.to_csv(output/'selected_scores.csv', index=False)
    composition.to_csv(output/'composition_scores.csv', index=False)
    baseline_scores.to_csv(output/'jointmoe_scores.csv', index=False)
    summary = {'schema': 'stage1-submission-replay/v1', 'model': MODEL, 'new_training': False, 'new_fitting': False,
        'selected_prediction_max_abs_error': error, 'jointmoe_prediction_max_abs_error': baseline_error,
        'selected_metric_cells': len(metrics), 'jointmoe_metric_cells': len(baseline_scores),
        'composition_cells': len(composition),
        'input_manifests': {name: hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
                            for name, root in [('selected', selected), ('jointmoe', baselines)]}}
    (output/'receipt.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(metrics.to_string(index=False))
    print(baseline_scores.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=ROOT/'stage1_selected_law_v1')
    parser.add_argument('--baselines', type=Path, default=ROOT/'stage1_jointmoe_token_contrast_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.selected, args.baselines, args.output)


if __name__ == '__main__':
    main()
