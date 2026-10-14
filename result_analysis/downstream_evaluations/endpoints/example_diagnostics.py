"""Paired example diagnostics and conditional evaluation-sample uncertainty.

No cross-language bootstrap: translated examples are not independent replicas.
All NTREX windows stay together in their original document.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import sealed
from multilingual_megatron.contracts.hashing import canonical_json_sha256
from .endpoint_diagnostics import LOSS
from multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign import result_for
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign import checked, read, ref, save

CLASSIFICATION = {'sib200', 'taxi1500'}
MINIMAL = {'multiblimp', 'xcomps'}
CAMPAIGNS = ['stage1_full_final_h100_20260920', 'stage1_candidate_full_a40_20260921', 'stage1_extension_full_a40_20260921_ready']
CELLS = [(1, 12), (32, 12), (1, 200), (32, 200)]


def f1_scores(gold, pred, classes):
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (gold, pred), 1)
    support, predicted = cm.sum(1), cm.sum(0)
    f1 = np.divide(2*np.diag(cm), support+predicted, out=np.zeros(classes), where=(support+predicted)>0)
    return cm, float(f1.mean()), float(f1 @ support / len(gold))


def paired_bootstrap(numerators, denominators, blocks, *, seed=20260921, repetitions=2000):
    """Four paired cell ratios; resample source blocks, not independent windows."""
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    assert numerators.shape == denominators.shape and numerators.shape[1] == 4
    _, inverse = np.unique(np.asarray(blocks, dtype=str), return_inverse=True)
    count = int(inverse.max()) + 1
    sums = np.zeros((count, 8))
    np.add.at(sums, inverse, np.column_stack([numerators, denominators]))
    # Identical blocks can be collapsed without changing the bootstrap law.
    unique, counts = np.unique(sums, axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repetitions, 50):
        weights = rng.multinomial(count, counts/count, size=min(50, repetitions-start))
        totals = weights @ unique
        ratios = totals[:, :4] / totals[:, 4:]
        gains = ratios[:, [1, 3]] - ratios[:, [0, 2]]
        estimates.append(np.column_stack([gains, gains[:, 1]-gains[:, 0]]))
    estimates = np.concatenate(estimates)
    point = numerators.sum(0)/denominators.sum(0)
    point = np.array([point[1]-point[0], point[3]-point[2], point[3]-point[2]-point[1]+point[0]])
    limits = np.quantile(estimates, [.025, .975], axis=0)
    return point, limits, count


def observations(task, by_id, benchmark):
    docs = task['documents']
    if benchmark in LOSS:
        nll, tokens, byte_count = [], [], []
        for d in docs:
            rows = [by_id[r] for r in d['conditional']]
            assert [r['scored_tokens'] for r in rows] == d['token_lengths']
            assert sum(d['token_lengths']) == d['target_tokens']
            nll.append(-math.fsum(r['loglikelihood'] for r in rows))
            tokens.append(d['target_tokens']); byte_count.append(d['utf8_bytes'])
        return dict(nll=np.array(nll), tokens=np.array(tokens), bytes=np.array(byte_count))
    result = defaultdict(list)
    for d in docs:
        rows = [by_id[r] for r in d['conditional']]
        values = np.array([r['loglikelihood'] for r in rows])
        tokens = np.array([r['scored_tokens'] for r in rows])
        if 'token_lengths' in d:
            assert tokens.tolist() == d['token_lengths']
        assert np.isfinite(values).all() and (tokens > 0).all()
        gold = d['target']
        char = np.array(d.get('char_lengths', d.get('choice_lengths')))
        byte_count = np.array(d.get('byte_lengths', char))
        scores = dict(raw=values, token=values/tokens, char=values/char, byte=values/byte_count)
        if benchmark == 'x_csqa':
            scores['pmi'] = values - np.array([by_id[r]['loglikelihood'] for r in d['unconditional']])
        primary = 'pmi' if benchmark == 'x_csqa' else 'raw' if benchmark in MINIMAL | CLASSIFICATION else 'char'
        result['gold'].append(gold); result['pred'].append(int(np.argmax(scores[primary])))
        result['chance'].append(1/len(values))
        result['gold_nll'].append(-values[gold]/tokens[gold])
        result['wrong_nll'].append(float(np.delete(-values/tokens, gold).mean()))
        for name, score in scores.items():
            correct = float(score[0] > score[1]) if benchmark in MINIMAL else float(np.argmax(score) == gold)
            margin = score[gold] - np.delete(score, gold).max()
            result['accuracy_'+name].append(correct)
            result['margin_'+name].append(margin)
            if name == primary:
                result['primary'].append(correct); result['margin'].append(margin)
    return {k:np.asarray(v) for k,v in result.items()}


def main(root, analysis, output):
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot = output / 'example_diagnostics_source.py'
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    receipt = sealed(analysis/'analysis_receipt.json')
    assert receipt['verified_checks']['complete_extension_matrix']
    canonical = pd.read_csv(analysis/'language_metrics_with_validation.csv', low_memory=False)
    fixed = pd.read_csv(analysis/'fixed_cohort_language_metrics.csv')
    common = set(fixed.corpus_language)
    canonical = canonical.set_index(['model_id','benchmark','language'])
    rows, confusion, transitions, bootstrap, verification, inputs = [], [], [], [], [], []
    base = root/'outputs/stage1_benchmark_jointmoe'
    for directory in CAMPAIGNS:
        campaign = sealed(base/directory/'campaign.json')
        inputs.append(ref(base/directory/'campaign.json'))
        for benchmark in campaign['inputs']:
            manifest_ref = campaign['inputs'][benchmark]['request_manifest']
            manifest = sealed(checked(manifest_ref)); inputs.append(manifest_ref)
            task_maps = {}
            for task in manifest['tasks']:
                language = task['language']
                match = canonical.loc[(slice(None),benchmark,language), :]
                if match.iloc[0].corpus_language in common:
                    task_maps[task['task']] = task
            assert task_maps
            # Original tasks store request/option membership outside documents.
            old_requests = defaultdict(dict)
            for request in manifest['requests']:
                if request['task'] in task_maps and 'conditional' not in task_maps[request['task']]['documents'][0]:
                    old_requests[(request['task'],request['doc_id'])][request['choice_index']] = request['request_id']
            for task in task_maps.values():
                for d in task['documents']:
                    if 'conditional' not in d:
                        choices = old_requests[(task['task'],d['doc_id'])]
                        d['conditional'] = [choices[i] for i in range(len(choices))]
            selected_ids = {r for t in task_maps.values() for d in t['documents'] for r in d['conditional']+d.get('unconditional',[])}
            request_ids = {r['request_id'] for r in manifest['requests']}
            assert len(request_ids) == len(manifest['requests'])
            del old_requests, manifest
            gc.collect()
            jobs = [j for j in campaign['jobs'] if j.get('phase','full')=='full' and j['benchmark']==benchmark and
                    (benchmark in CLASSIFICATION or (j['ladder'] in ['B','D'] and j['E'] in [1,32] and j['K'] in [12,200]))]
            assert len(jobs) == (82 if benchmark in CLASSIFICATION else 16)
            cells = {}
            reference_token_counts = None
            for ji, job in enumerate(jobs):
                if directory == CAMPAIGNS[0]:
                    spec = read(checked(job['spec']))
                    result = sealed(Path(spec['output_dir'])/'scores.json', 'result_sha256')
                    result['result_sha256'] = canonical_json_sha256(result)
                    assert result['status'] == 'complete'
                    assert result['score_spec']['sha256'] == job['spec']['sha256']
                    assert result['request_manifest']['sha256'] == manifest_ref['sha256']
                    assert result['parity_gate']['passed'] and result['replica_gate']['passed']
                    # The original frozen audit explicitly permits these 62
                    # left-context truncations; answer targets remain complete.
                    assert result['counts']['truncated_requests'] == (62 if benchmark == 'global_mmlu_full' else 0)
                    if job['E'] > 1:
                        assert result['moe_routing']['evidence_available']
                        assert result['moe_routing']['totals']['dropped_assignments'] == 0
                else:
                    result = result_for(job)
                assert result is not None
                assert {r['request_id'] for r in result['scores']} == request_ids
                assert len(result['scores']) == len(request_ids)
                by_id = {r['request_id']:r for r in result['scores'] if r['request_id'] in selected_ids}
                assert len(by_id) == len(selected_ids)
                assert all(np.isfinite(r['loglikelihood']) and r['scored_tokens']>0 for r in by_id.values())
                token_counts = {rid:(r['scored_tokens'],r.get('truncated_tokens',0)) for rid,r in by_id.items()}
                if reference_token_counts is None: reference_token_counts = token_counts
                else: assert token_counts == reference_token_counts
                verification.append(dict(job=job['id'],result_sha256=result['result_sha256'],score_spec=job['spec'],selected_requests=len(by_id),context_truncated_requests=result['counts']['truncated_requests']))
                del result
                identity = {k:job[k] for k in ['model_id','ladder','E','K','seed','benchmark']}
                for task in task_maps.values():
                    language = task['language']; expected = canonical.loc[(job['model_id'],benchmark,language)]
                    obs = observations(task,by_id,benchmark)
                    item = dict(**identity, language=language, corpus_language=expected.corpus_language, samples=len(task['documents']))
                    if benchmark in LOSS:
                        item.update(primary=obs['nll'].sum()/obs['bytes'].sum()/math.log(2),
                                    external_nll=obs['nll'].sum()/obs['tokens'].sum())
                    else:
                        item.update({k:float(v.mean()) for k,v in obs.items() if k not in ['gold','pred']})
                        if benchmark in CLASSIFICATION:
                            classes=max(d['classes'] for d in task['documents'])
                            cm, macro, weighted = f1_scores(obs['gold'],obs['pred'],classes)
                            item.update(macro_f1=macro,weighted_f1=weighted,majority_baseline=cm.sum(1).max()/cm.sum(),
                                        dominant_label=int(cm.sum(0).argmax()),dominant_fraction=cm.sum(0).max()/cm.sum(),labels_predicted=int((cm.sum(0)>0).sum()))
                            for key in ['macro_f1','weighted_f1','majority_baseline']:
                                assert abs(item[key]-expected[key])<1e-12
                            for g,p in np.ndindex(cm.shape):
                                confusion.append(dict(**identity,language=language,gold=g,predicted=p,count=int(cm[g,p])))
                    assert abs(item['primary']-expected.primary)<1e-12,(identity,language,item['primary'],expected.primary)
                    rows.append(item)
                    if job['ladder'] in ['B','D'] and (job['E'],job['K']) in CELLS:
                        cells[(job['ladder'],job['seed'],language,job['E'],job['K'])] = obs
                del by_id
                print(f"Verified examples {benchmark} {ji+1}/{len(jobs)} {job['model_id']}",flush=True)
            for ladder, seed in [('D',1234),('B',1234),('B',5678),('B',91011)]:
                for task in task_maps.values():
                    language=task['language']; group=[cells[(ladder,seed,language,e,k)] for e,k in CELLS]
                    blocks=[str(d.get('source_id',d['doc_id'])) for d in task['documents']]
                    identity=dict(benchmark=benchmark,ladder=ladder,seed=seed,language=language)
                    if benchmark in LOSS:
                        assert all(np.array_equal(group[0]['tokens'],g['tokens']) and np.array_equal(group[0]['bytes'],g['bytes']) for g in group)
                        quantities=[('external_nll',-np.column_stack([g['nll'] for g in group]),np.column_stack([g['tokens'] for g in group])),
                                    ('bits_per_byte',-np.column_stack([g['nll']/math.log(2) for g in group]),np.column_stack([g['bytes'] for g in group]))]
                    else:
                        quantities=[('primary',100*np.column_stack([g['primary'] for g in group]),np.ones((len(blocks),4)))]
                        for index,k in [(0,12),(2,200)]:
                            a,b=group[index]['primary'].astype(bool),group[index+1]['primary'].astype(bool)
                            transitions.append(dict(**identity,K=k,samples=len(a),corrected=int((~a&b).sum()),broken=int((a&~b).sum()),net_gain_pp=100*float((b.astype(float)-a).mean())))
                    for metric,num,den in quantities:
                        salt=int.from_bytes(hashlib.sha256(f'{benchmark}:{ladder}:{seed}:{language}:{metric}'.encode()).digest()[:4],'big')
                        point,intervals,count=paired_bootstrap(num,den,blocks,seed=20260921+salt)
                        for i,contrast in enumerate(['gain_K12','gain_K200','interaction']):
                            bootstrap.append(dict(**identity,metric=metric,contrast=contrast,estimate=point[i],low=intervals[0,i],high=intervals[1,i],source_blocks=count,repetitions=2000))
            del cells, task_maps
            gc.collect()
            # Checkpoint after each family so the verified progress is inspectable.
            for name,data in [('language_examples',rows),('classification_confusion',confusion),('paired_answer_changes',transitions),('paired_bootstrap',bootstrap)]:
                if data:pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    assert len(verification)==356
    save(output/'verification.json',dict(status='verified',source=ref(source_snapshot),analysis_receipt=ref(analysis/'analysis_receipt.json'),
         inputs=inputs,scores=verification,raw_jobs_verified=len(verification),
         uncertainty='Paired 2000-resample percentile intervals within each language; document/source IDs are blocks; NTREX windows kept together. Conditional on fixed checkpoints. No pooled multilingual or training-population intervals. Salted deterministic base seed 20260921.',
         output_files={p.name:ref(p) for p in output.glob('*.csv')}),'manifest_sha256')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.analysis.resolve(),a.output.resolve())
