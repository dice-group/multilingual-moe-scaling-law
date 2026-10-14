"""Compare two recorded LR recipes with the existing RQ1--RQ3 estimators.

Run: python -m result_analysis.study_setup.lr_sensitivity --config PATH
No fitting, training, manuscript changes, or replacement of sealed evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from result_analysis.capacity_response import analysis as pressure
from result_analysis.compute_allocation import analysis as allocation
from result_analysis.language_preservation import analysis as preservation
from result_analysis.language_preservation.sensitivity import restore
from result_analysis.learning_curves import RecipeCurve

KEYS = ['trajectory_id', 'readout_id', 'language']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_release(root, sources):
    def read(name):
        path = root / 'tables' / (name + '.parquet')
        sources[str(path)] = digest(path)
        return pd.read_parquet(path)
    val, exposure, models = read('validation_language'), read('data_exposure'), read('model_configs')
    geometry = models.rename(columns={'experts': 'E', 'declared_active_parameters': 'N_act',
        'configured_design_total_parameters': 'N_total'})[['ladder', 'E', 'N_act', 'N_total']]
    rows = val.query("metric_role == 'trained'").merge(exposure[KEYS + ['realized_self_exposure_tokens']],
        on=KEYS, validate='one_to_one').rename(columns={'experts': 'E', 'language_count': 'K',
        'actual_tokens': 'D_total', 'atlas_tao_unigram_adjusted_loss': 'adjusted_loss',
        'realized_self_exposure_tokens': 'D_t'}).merge(geometry, on=['ladder', 'E'], validate='many_to_one')
    trajectories, configs, readouts = read('trajectories'), read('training_configs'), read('readouts')
    assert len(rows) == 26300 and rows.readout_id.nunique() == 400 and rows.trajectory_id.nunique() == 64
    assert not rows.duplicated(KEYS).any()
    assert np.isfinite(rows[['D_t', 'D_total', 'N_act', 'N_total', 'adjusted_loss', 'raw_nll']]).all().all()
    assert rows.groupby(['ladder', 'E', 'K']).trajectory_id.nunique().eq(1).all()
    return rows, geometry, models, trajectories, configs, readouts


def run(config_path):
    cfg = json.loads(config_path.read_text())
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    sources = {str(config_path): digest(config_path)}
    panels = {name: load_release(Path(path), sources) for name, path in cfg['releases'].items()}
    main = pd.read_parquet(Path(cfg['primary_data']) / 'main_grid.parquet').query("metric_role == 'trained'")
    sources[str(Path(cfg['primary_data']) / 'main_grid.parquet')] = digest(Path(cfg['primary_data']) / 'main_grid.parquet')
    a = panels['expert_aware'][0].set_index(KEYS).sort_index()
    b = main.set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_allclose(a[['D_t','D_total','N_act','N_total','adjusted_loss','raw_nll']],
                               b[['D_t','D_total','N_act','N_total','adjusted_loss','raw_nll']], atol=1e-12, rtol=0)
    cohorts = {k: sorted(main.loc[main.K.eq(k), 'language'].unique()) for k in (12,50)}
    audits, rq1, rq2, rq3, winners = [], [], [], [], []
    base = panels['expert_aware']
    fields = ['ladder','experts','num_layers','hidden_size','ffn_hidden_size','num_attention_heads',
              'sequence_length','vocab_size','declared_active_parameters','configured_design_total_parameters']
    for name, (rows, geometry, models, trajectories, configs, readouts) in panels.items():
        dest = out / name; dest.mkdir()
        pd.testing.assert_frame_equal(models[fields].sort_values(fields[:2]).reset_index(drop=True),
            base[2][fields].sort_values(fields[:2]).reset_index(drop=True), check_dtype=False)
        for k, cohort in cohorts.items():
            assert sorted(rows.loc[rows.K.eq(k),'language'].unique()) == cohort
        aligned = rows.set_index(KEYS).sort_index()
        pd.testing.assert_index_equal(aligned.index, a.index)
        audit = dict(recipe=name, trajectories=64, checkpoints=400, observations=len(rows),
            model_seeds=sorted(trajectories.model_seed.unique().tolist()),
            dataset_seeds=sorted(trajectories.dataset_shuffle_seed.unique().tolist()),
            max_token_difference=float(abs(aligned.D_total-a.D_total).max()),
            max_exposure_difference=float(abs(aligned.D_t-a.D_t).max()))
        for field in ['sequence_length','warmup_tokens','decay_fraction','schedule_mode']:
            audit[field] = sorted(configs[field].dropna().unique().tolist())
        audit['data_manifest_hashes_match'] = set(configs.data_manifest_sha256)==set(base[4].data_manifest_sha256)
        assert audit['data_manifest_hashes_match']
        for field in ['model_seed','dataset_shuffle_seed']:
            assert set(trajectories[field])==set(base[3][field])
        for field in ['sequence_length','warmup_tokens','decay_fraction','schedule_mode']:
            assert set(configs[field].dropna())==set(base[4][field].dropna())
        audits.append(audit)
        rows.to_parquet(dest / 'main_grid.parquet', index=False)
        results = pressure.from_parquet(dest)
        for key, table in results.items(): table.to_csv(dest / ('rq1_'+key+'.csv'), index=False)
        rq1.append(results['summary'].assign(recipe=name))
        for cap in cfg['allocation_caps']:
            results2 = allocation.analyze(rows, geometry, languages=cohorts[12], coverages=(12,200),
                cap=cap, penalty=cfg['penalty'], methods=cfg['methods'])
            results2['loss_curves'].to_csv(dest / f'rq2_curves_{cap}.csv', index=False)
            rq2.append(results2['switches'].assign(recipe=name))
            curves = results2['loss_curves']
            best = curves.loc[curves.groupby(['K','method','F']).loss.idxmin()]
            for (k, method), part in best.groupby(['K','method']):
                switches = results2['switches'].query('K==@k and method==@method')
                winners.append(dict(recipe=name,cap=cap,K=k,method=method,
                    lower_EFLOPs=float(part.F.min()),upper_EFLOPs=float(part.F.max()),
                    winners=','.join(part.sort_values('F').config.unique()),switches=len(switches)))
            if cap==cfg['allocation_caps'][0]:
                fine = allocation.analyze(rows,geometry,languages=cohorts[12],coverages=(12,200),
                    cap=cap,penalty=cfg['penalty'],methods=cfg['methods'],grid_points=721)['switches']
                coarse = results2['switches']
                assert fine[['K','method','below','above']].equals(coarse[['K','method','below','above']])
                np.testing.assert_allclose(fine.F.to_numpy(float),coarse.F.to_numpy(float),atol=1e-9,rtol=0)
        # Use the canonical primary preservation API and cross-check the existing
        # multi-interpolator estimator below, rather than duplicating inversion.
        primary = preservation.analyze(rows, geometry, languages=cohorts[50], base_k=50,target_k=200,
            cap=cfg['preservation_cap'],baseline_f=cfg['reference_EFLOPs'],penalty=cfg['penalty'])
        for key, table in primary.items(): table.to_csv(dest / ('rq3_'+key+'.csv'), index=key=='summary')
        bank = []
        for (_, _, k), group in rows[rows.K.isin([50,200]) & rows.language.isin(cohorts[50])
                                    & rows.N_total.le(cfg['preservation_cap'])].groupby(['ladder','E','K']):
            bank.append(RecipeCurve(group, {'smooth': cfg['penalty']}))
        per_language = []
        for method in cfg['methods']:
            f = cfg['reference_EFLOPs']; cohort = cohorts[50]
            eligible = [r for r in bank if r.k==50 and r.fmin <= f <= r.fmax]
            reference = min(eligible, key=lambda r:(r.macro(f,cohort,method),r.stored))
            values = reference.values(f,cohort,method)
            for objective, tolerance in [('mean',0.),('all',0.),('all',.01)]:
                target = restore([r for r in bank if r.k==200], values, cohort, method, objective,tolerance)
                record = dict(recipe=name,method=method,objective=objective,tolerance=tolerance,
                    reference_config=reference.name,compute_multiplier=target['F']/f, **target)
                if target['config'] is not None:
                    chosen = next(r for r in bank if r.k==200 and r.name==target['config'])
                    deltas = chosen.values(target['F'],cohort,method)-values
                    record.update(regressions=int((deltas>1e-9).sum()),above_001=int((deltas>.01+1e-9).sum()),
                                  above_002=int((deltas>.02+1e-9).sum()))
                    per_language.extend(dict(recipe=name,method=method,objective=objective,tolerance=tolerance,
                        language=l,delta=float(d)) for l,d in zip(cohort,deltas))
                    if method=='smooth' and tolerance==0:
                        expected = primary['policies'].set_index('objective').loc[objective]
                        assert chosen.name==expected.config and reference.name==expected.baseline_config
                        np.testing.assert_allclose(record['compute_multiplier'],expected.compute_multiplier,atol=1e-9,rtol=0)
                rq3.append(record)
        pd.DataFrame(per_language).to_csv(dest/'rq3_language_deltas_all_methods.csv',index=False)
    p1,p2,p3 = pd.concat(rq1),pd.concat(rq2),pd.DataFrame(rq3)
    # Published-result parity gate applies to the current recipe only.
    refs=Path(cfg['reference_outputs'])
    saved=pd.read_csv(refs/'compute_allocation/empirical_switches.csv')
    keys=['cap','K','method','below','above']
    got=p2.query("recipe=='expert_aware' and cap==400000000").set_index(keys).sort_index()
    expected=saved.set_index(keys).sort_index()
    pd.testing.assert_index_equal(got.index,expected.index, exact=False)
    np.testing.assert_allclose(got.F.to_numpy(float),expected.F.to_numpy(float),atol=1e-9,rtol=0)
    expected=pd.read_csv(refs/'language_preservation/preservation_summary.csv').set_index('objective')
    got=p3.query("recipe=='expert_aware' and method=='smooth' and tolerance==0").set_index('objective')
    np.testing.assert_allclose(got.loc[expected.index,'compute_multiplier'],expected.compute_multiplier,atol=1e-9,rtol=0)
    np.testing.assert_array_equal(50-got.loc[expected.index,'regressions'],expected.protected_languages)
    saved1=pd.read_csv(refs/'capacity_response/pressure_summary.csv')
    np.testing.assert_allclose(p1.query("recipe=='expert_aware'").set_index('expert_count').coverage_gap,
                               saved1.set_index('E').pressure,atol=1e-12,rtol=0)
    pd.DataFrame(winners).to_csv(out/'rq2_support_and_winners.csv',index=False)
    p1.to_csv(out/'rq1_summary.csv',index=False);p2.to_csv(out/'rq2_switches.csv',index=False);p3.to_csv(out/'rq3_summary.csv',index=False)
    pd.DataFrame(audits).to_json(out/'input_audit.json',orient='records',indent=2)
    code=[Path(__file__),Path(pressure.__file__),Path(allocation.__file__),Path(preservation.__file__),
          Path('result_analysis/capacity_response/matched_exposure.py'),Path('result_analysis/compute_allocation/observed.py'),
          Path('result_analysis/language_preservation/estimators.py'),Path('result_analysis/language_preservation/sensitivity.py'),
          Path('result_analysis/learning_curves.py')]
    (out/'receipt.json').write_text(json.dumps(dict(status='complete',sources=sources,
        code={str(p):digest(p) for p in code},parity='current measurements and RQ1/RQ2/RQ3 published results passed; allocation 181 vs 721 budget points matched',
        scope='Descriptive optimization-recipe sensitivity; one seed per grid; no fitting or independent seed uncertainty'),indent=2)+'\n')
    print(p1.to_string(index=False));print(p2.to_string(index=False));print(p3.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
