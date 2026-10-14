"""Matched native JointMoE D versus D_t refits using immutable campaign kernels.

Only the exposure input changes. E_start/E_max remain fixed at the published
values used by the existing baseline; this is not a refit of those two values.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def exposure_packet(packet, token):
    import numpy as np
    result = dict(packet)
    if token == 'Dt':
        # Native JointMoE consumes the row sum; retain the target column only.
        # No target values, weights, inventories, or model coordinates change.
        d = np.asarray(packet['D'], float).copy()
        d[:, 1:] = 0.
        result['D'] = d.tolist()
    return result


def literal(x, packet, token):
    """Independent equation using original token columns, without adapter."""
    import numpy as np
    e = np.asarray(packet['E'], float)
    start, maximum = 2.0732, 290.4521
    bridge = 1 / (1 / start - 1 / maximum)
    ehat = 1 / (1 / (e - 1 + bridge) + 1 / maximum)
    ehat32 = 1 / (1 / (31 + bridge) + 1 / maximum)
    g = (np.log(ehat) - np.log(start)) * np.log(32) / (np.log(ehat32) - np.log(start))
    h = g / np.log(32)
    d = np.asarray(packet['D'], float)
    d = d[:, 0] if token == 'Dt' else d.sum(axis=1)
    n = np.asarray(packet['N'], float) / 1e8
    return (x[0] * packet['training_minimum']
            + x[1] * np.exp(x[7] * g) * n ** (-(x[3]*(1-h) + x[4]*h))
            + x[2] * np.exp(x[8] * g) * (d/1e9) ** (-(x[5]*(1-h) + x[6]*h)))


def worker(task):
    from paper_fit_snapshot.atlas_comparison_workflow import model, numerics, recovery_numerics
    import numpy as np
    fold, token, item, output = task
    began = time.monotonic()
    spec = model.Spec('JointMoE-JH-local', 'joint')
    train, support = (exposure_packet(item[k], token) for k in ('train', 'support'))
    result = numerics.fit_group(spec, train, support)
    if not result['certified']:
        initial = result
        result = recovery_numerics.recover(spec, train, support, initial, joint_method='L-BFGS-B')
        result['initial_fit'] = initial
    if result['certified']:
        x = result['selected']['coordinates']
        actual = model.predict(spec, x, train)[0]
        expected = literal(x, item['train'], token)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        result['literal_replay_max_error'] = float(np.max(abs(actual-expected)))
    result.update(fold=fold, token=token, language=item['language'], donors=item['donors'],
                  unigram_nll=item['unigram_nll'], training_minimum=train['training_minimum'],
                  elapsed_seconds=time.monotonic()-began)
    write(Path(output)/'fits'/f'{fold}--{token}--{item["language"]}.json', result)
    return fold, token, item['language'], result['certified']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    root, out = args.campaign.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Output is create-only')
    sys.path.insert(0, str(root/'source'))
    import numpy as np
    import pandas as pd
    from paper_fit_snapshot.paper_refit_workflow import inputs
    from paper_fit_snapshot.atlas_comparison_workflow import model, numerics
    from paper_fit_snapshot.law_search import data
    from paper_fit_snapshot.scoring import checked_score
    manifest_path = root/'inputs/manifest.json'
    manifest = inputs.load(manifest_path)
    folds = (*inputs.PRIMARY, 'all_grid')
    out.mkdir(parents=True)
    (out/'fits').mkdir()
    source = Path(__file__).resolve()
    (out/'runner.py').write_bytes(source.read_bytes())
    write(out/'protocol.json', dict(campaign=str(root), input_sha256=manifest['sha256'],
        runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), workers=args.workers,
        threads_per_worker=1, tokens=['D','Dt'], folds=list(folds), settings=numerics.SETTINGS,
        expert_transform=dict(E_start=2.0732, E_max=290.4521, fitted=False),
        coefficient_scope='nine per language, identical in both arms',
        changed='Only scalar exposure D_total versus D_t; original immutable native kernels',
        selection='training objectives and numerical certification only; no evaluation-based selection'))
    tasks = []
    for fold in folds:
        rows, _, packets, _ = inputs.read_training(manifest_path, fold)
        inputs.native_groups(rows, packets)
        tasks.extend((fold, token, item, str(out)) for token in ('D','Dt') for item in packets)
    began = time.monotonic()
    print(f'Starting {len(tasks)} language fits with {args.workers} workers', flush=True)
    certified = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            fold, token, language, passed = future.result()
            certified += passed
            if done % 50 == 0 or not passed:
                print(f'{done}/{len(tasks)} complete; {certified} certified; {time.monotonic()-began:.1f}s; {fold}/{token}/{language}', flush=True)
    if certified != len(tasks):
        raise RuntimeError(f'{len(tasks)-certified} uncertified fits; evaluation withheld')
    # Evaluation response files are opened only after all fits are frozen.
    grid, endpoint = [pd.read_parquet(inputs.local_path(manifest_path.parent, manifest['scoring'][n]))
                      for n in ('grid.parquet','endpoint.parquet')]
    metrics, predictions = [], []
    spec = model.Spec('JointMoE-JH-local','joint')
    for fold in folds:
        rows = endpoint.copy() if fold == 'all_grid' else data.subset(grid, manifest['memberships'][fold], 'test')
        for token in ('D','Dt'):
            frame = rows[data.KEYS+['adjusted_loss']].copy()
            frame['prediction'] = np.nan
            for language, part in rows.groupby('language'):
                result = json.loads((out/'fits'/f'{fold}--{token}--{language}.json').read_text())
                # JointMoE needs only N, E and scalar exposure, not donor identities.
                d = np.zeros((len(part),5)); d[:,0] = part.D_t
                d[:,1] = part.D_total-part.D_t
                packet = dict(N=part.N_act.to_numpy(), E=part.E.to_numpy(), D=d,
                              training_minimum=result['training_minimum'])
                actual = model.predict(spec, result['selected']['coordinates'], exposure_packet(packet,token))[0]
                expected = literal(result['selected']['coordinates'], packet,token)
                np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
                frame.loc[part.index,'prediction'] = actual-result['unigram_nll']
            score = checked_score(frame,rows,endpoint=fold=='all_grid')
            assert score['complete']
            metrics.append(dict(token=token,fold=fold,**score))
            predictions.append(frame.assign(token=token,fold=fold))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    table = metrics.pivot(index='token',columns='fold',values='rmse')
    table['mean_primary_rmse'] = table[list(inputs.PRIMARY)].mean(axis=1)
    table.to_csv(out/'comparison.csv')
    old = pd.read_csv(root/'recovery/v2/report/comparison_lbfgsb_box.csv').set_index('model').loc['JointMoE-JH-local']
    gaps = {fold:float(table.loc['D',fold]-old['L26_all_grid' if fold=='all_grid' else fold]) for fold in folds}
    write(out/'verification.json',dict(certified=certified,total=len(tasks),historical_D_rmse_difference=gaps,
        elapsed_seconds=time.monotonic()-began,manuscript_changed=False))
    print(table.to_string(),flush=True)
    print('Historical D RMSE differences:',gaps,flush=True)


if __name__ == '__main__':
    main()
