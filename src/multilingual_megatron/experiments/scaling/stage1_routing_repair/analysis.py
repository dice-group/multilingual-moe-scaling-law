"""Fixed-cohort, layer-local routing comparisons with paired sequence blocks."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .campaign import identity, read, sha, verify, write


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1)


def mutual_information(probabilities):
    """I(language;expert) under an explicitly uniform language prior, in nats."""
    p = np.asarray(probabilities,dtype=np.float64)
    return entropy(p.mean(axis=0))-entropy(p).mean(axis=0)


def js_matrix(probabilities):
    p = np.asarray(probabilities,dtype=np.float64)
    mean = (p[:,None,:]+p[None,:,:])/2
    return entropy(mean)-(entropy(p)[:,None]+entropy(p)[None,:])/2


def bootstrap_probabilities(layer, language, replicates, seed):
    blocks = np.asarray([value for _,value in sorted(layer['blocks'].items(),key=lambda x:int(x[0])) if sum(value)>0],dtype=np.float64)
    # Same stream, block layout and seed across checkpoints: paired resampling.
    language_seed = int.from_bytes(hashlib.sha256(f'{seed}:{language}'.encode()).digest()[:8],'little')
    rng = np.random.default_rng(language_seed)
    weights = rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=replicates)
    counts = weights @ blocks
    return counts/counts.sum(-1,keepdims=True)


def collect(campaign):
    from .execution import validate_attempt
    root = Path(campaign['root'])
    rows = []
    results = []
    for task in campaign['tasks']:
        if task['kind'] != 'production':
            continue
        path = root/'results'/f"{task['id']}.json"
        state = 'missing'
        error = None
        if path.exists():
            try:
                result = read(path)
                if result['status'] != 'complete' or result['task_id'] != task['id']:
                    raise ValueError('result identity/status differs from the task')
                verify(result['receipt'])
                receipt = read(result['receipt']['path'])
                if receipt['status'] != 'complete' or receipt['task_id'] != task['id'] or receipt['attempt'] != result['attempt']:
                    raise ValueError('selected attempt receipt disagrees with the result')
                for entry in result['manifests'].values():
                    verify(entry)
                checked = validate_attempt(campaign,task,result['attempt'])
                if checked['manifests'] != result['manifests']:
                    raise ValueError('selected manifests differ from the verified attempt')
                state = 'complete'
                results.append({'task_id':task['id'],**identity(path)})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                state, error = 'invalid', repr(exc)
        rows.append({'task_id':task['id'],'status':state,'expected_languages':len(task['languages']),'error':error})
    complete = len(results)==60
    gate_error = None
    if complete and campaign['runtime'].get('gate_policy') == 'fixed_execution_v3':
        from .recovery import require_controls
        try:
            require_controls(campaign)
            pilots = read(root/'pilots_complete.json')
            if pilots['status'] != 'complete':
                raise ValueError('pilots did not pass')
            for entry in pilots['pilots']:
                verify(entry)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            complete, gate_error = False, repr(exc)
    accounting_path = root/'scheduler_status.psv'
    accounting = []
    accounting_errors = []
    metadata_path = root/'scheduler_status.json'
    try:
        metadata = read(metadata_path) if metadata_path.exists() else {'status':'unverified_snapshot'}
        fresh = metadata['status']=='queried'
        if fresh:
            verify(metadata['snapshot'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        metadata, fresh = {'status':'invalid_snapshot','error':repr(exc)}, False
    if accounting_path.exists():
        for line in accounting_path.read_text().splitlines():
            fields = line.split('|')
            if len(fields) < 6:
                accounting_errors.append(line)
                continue
            resources = dict(item.split('=',1) for item in fields[4].split(',') if '=' in item)
            try:
                gpus = int(resources.get('gres/gpu', resources.get('gres/gpu:h100', 0)))
                elapsed = int(fields[3] or 0)
            except ValueError:
                accounting_errors.append(line)
                continue
            accounting.append({'job_id':fields[0], 'state':fields[2], 'elapsed_seconds':elapsed,
                'allocated_gpus':gpus, 'allocated_gpu_hours':gpus*elapsed/3600, 'exit_code':fields[5]})
    write(root/'runtime_accounting.json', {'scheduler_accounting_available':fresh and not accounting_errors,
        'snapshot_is_stale_or_unverified':not fresh, 'query':metadata,'parse_errors':accounting_errors,
        'allocated_gpu_hours':sum(row['allocated_gpu_hours'] for row in accounting), 'jobs':accounting})
    write(root/'completeness.json',{'status':'complete' if complete else 'incomplete',
        'expected_checkpoints':60,'complete_checkpoints':len(results),
        'expected_language_streams':4008,'models':rows,'results':results,'gate_error':gate_error})
    report = root/'analysis'/'README.md'
    report.parent.mkdir(exist_ok=True)
    if not complete:
        report.write_text(f"Routing repair incomplete: {len(results)}/60 checkpoints validated.\n\n"
            "No corrected scientific release was produced. See ../completeness.json, attempt receipts, and scheduler_status.psv.\n")
        return {'status':'incomplete','complete_checkpoints':len(results)}
    try:
        return analyze(campaign)
    except Exception as exc:
        write(root/'analysis_failure.json',{'status':'failed','error':repr(exc)})
        report.write_text(f'All checkpoint results validated, but analysis failed: {exc!r}.\n'
            'No new complete analysis release was produced. See ../analysis_failure.json.\n')
        raise


def analyze(campaign):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(campaign['root'])
    completion = read(root/'completeness.json')
    if completion['status']!='complete' or len(completion['results'])!=60:
        raise ValueError('analysis requires all 60 verified final checkpoints')
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    layer_rows = []
    summary_rows = []
    pair_rows = []
    seed_rows = []
    bootstrap = {}
    comparison_rows = []
    reps = campaign['runtime']['analysis']['bootstrap_replicates']
    seed = campaign['runtime']['analysis']['seed']
    valid_total = 0
    scored_total = 0
    for task in campaign['tasks']:
        if task['kind']!='production':
            continue
        model = task['model']
        identity_row = {key:model[key] for key in ('id','ladder','experts','languages','seed','tokens_seen')}
        identity_row.update(micro_batch_size=model['micro_batch_size'],
            expert_parallel_size=model['expert_parallel_size'], allocated_gpus=task['gpus'],
            moe_dispatcher=model['moe_dispatcher'])
        result = read(root/'results'/f"{task['id']}.json")
        verify(result['manifests']['corrected'])
        manifest = read(result['manifests']['corrected']['path'])
        old_manifest = read(model['evaluation_manifest'])
        language_rows = {}
        old_rows = {}
        for language,entry in manifest['languages'].items():
            verify(entry)
            row = read(entry['path'])
            old_entry = old_manifest['languages'][language]
            verify(old_entry)
            language_rows[language] = row
            old_rows[language] = read(old_entry['path'])
            scored_total += row['token_count']
            valid_total += row['backend']['stream_tokens_including_special']
            for layer in row['moe_routing']['layers']:
                old_layer = old_rows[language]['moe_routing']['layers'][layer['layer_index']]
                count = np.asarray(layer['expert_token_counts'],dtype=np.float64)
                p = count/count.sum()
                h = float(entropy(p))
                layer_rows.append({**identity_row,'language':language,'layer_index':layer['layer_index'],
                    'valid_positions':layer['forwarded_tokens'],'padding_positions':layer['padding_positions'],
                    'padding_fraction':layer['padding_positions']/layer['all_positions'],
                    'expert_token_counts':layer['expert_token_counts'],'padding_expert_token_counts':layer['padding_expert_token_counts'],
                    'all_expert_token_counts':layer['all_expert_token_counts'],
                    'historical_expert_token_counts':old_layer['expert_token_counts'],
                    'historical_forwarded_positions':old_layer['forwarded_tokens'],
                    'entropy_nats':h,'effective_experts':math.exp(h),'effective_expert_fraction':math.exp(h)/model['experts'],
                    'router_z_loss':layer['router_z_loss_unweighted'],'switch_aux_loss':layer['switch_aux_loss_unweighted'],
                    'language_artifact':entry['path'],'language_artifact_sha256':entry['sha256']})
        cohorts = {'trained':task['languages']}
        for name in ('k12','k50'):
            members = campaign['cohorts'][name]
            if set(members).issubset(task['languages']):
                cohorts['common'+name[1:]] = members
        n_layers = len(next(iter(language_rows.values()))['moe_routing']['layers'])
        for cohort,languages in cohorts.items():
            for layer_index in range(n_layers):
                layers = [language_rows[l]['moe_routing']['layers'][layer_index] for l in languages]
                p = np.asarray([np.asarray(l['expert_token_counts'])/sum(l['expert_token_counts']) for l in layers])
                all_p = np.asarray([np.asarray(l['all_expert_token_counts'])/sum(l['all_expert_token_counts']) for l in layers])
                historical = [old_rows[l]['moe_routing']['layers'][layer_index] for l in languages]
                old_p = np.asarray([np.asarray(l['expert_token_counts'])/sum(l['expert_token_counts']) for l in historical])
                mi = float(mutual_information(p)) if len(languages)>1 else None
                sampled = np.stack([bootstrap_probabilities(l,language,reps,seed) for l,language in zip(layers,languages)])
                sampled_mi = mutual_information(sampled) if len(languages)>1 else np.zeros(reps)
                lo,hi = np.quantile(sampled_mi,[.025,.975])
                key = (model['ladder'],model['experts'],model['seed'],model['tokens_seen'],cohort,layer_index)
                bootstrap.setdefault(key,{})[model['languages']] = sampled_mi
                summary_rows.append({**identity_row,'cohort':cohort,'cohort_size':len(languages),'layer_index':layer_index,
                    'language_expert_mi_nats':mi,'mi_over_log_experts':mi/math.log(model['experts']) if mi is not None else None,
                    'mi_block_ci_low':float(lo) if mi is not None else None,'mi_block_ci_high':float(hi) if mi is not None else None,
                    'mean_language_effective_experts':float(np.exp(entropy(p)).mean()),
                    'new_all_position_mi_nats':float(mutual_information(all_p)) if mi is not None else None,
                    'historical_v1_mi_nats':float(mutual_information(old_p)) if mi is not None else None,
                    'equal_language_weights':True})
                # Matrices use language labels. Expert IDs are never aligned
                # between independently trained models or different layers.
                if cohort.startswith('common'):
                    js = js_matrix(p)
                    for i,left in enumerate(languages):
                        for j in range(i+1,len(languages)):
                            pair_rows.append({**identity_row,'cohort':cohort,'layer_index':layer_index,
                                'language_a':left,'language_b':languages[j],'js_nats':float(js[i,j])})
    if valid_total!=64201922812 or scored_total!=64142756068:
        raise ValueError(f'campaign token totals differ: valid={valid_total}, scored={scored_total}')
    for key,by_k in bootstrap.items():
        ladder,e,model_seed,tokens,cohort,layer = key
        if not cohort.startswith('common'):
            continue
        ks = sorted(by_k)
        for i,k1 in enumerate(ks):
            for k2 in ks[i+1:]:
                delta = by_k[k2]-by_k[k1]
                lo,hi = np.quantile(delta,[.025,.975])
                comparison_rows.append({'ladder':ladder,'experts':e,'seed':model_seed,'tokens_seen':tokens,
                    'cohort':cohort,'layer_index':layer,'K_from':k1,'K_to':k2,
                    'paired_bootstrap_mean_delta_mi_nats':float(delta.mean()),'delta_ci_low':float(lo),'delta_ci_high':float(hi)})
    groups = {}
    for row in summary_rows:
        if row['ladder']=='B' and row['experts'] in (16,32) and row['languages'] in (12,200) and row['cohort']=='common12':
            key = (row['experts'],row['languages'],row['layer_index'])
            groups.setdefault(key,[]).append(row)
    for (e,k,layer),rows in groups.items():
        if {r['seed'] for r in rows}!={1234,5678,91011}:
            raise ValueError('B three-seed comparison incomplete')
        vals = [r['language_expert_mi_nats'] for r in rows]
        seed_rows.append({'ladder':'B','experts':e,'languages':k,'layer_index':layer,
            'mean_mi_nats':float(np.mean(vals)),'min_mi_nats':min(vals),'max_mi_nats':max(vals),'n_training_seeds':3})
    tables = {'routing_languages':layer_rows,'cohort_summary':summary_rows,'language_pair_js':pair_rows,
        'paired_K_comparisons':comparison_rows,'B_seed_summary':seed_rows}
    inventory = {}
    for name,rows in tables.items():
        path = out/f'{name}.parquet'
        pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
        inventory[name] = {**identity(path),'rows':len(rows)}
    _plot(summary_rows,out)
    report = (
        f"Corrected final routing release: 60 sparse checkpoints, 4,008 trained-language streams.\n\n"
        f"Valid input positions: {valid_total:,}; scored targets: {scored_total:,}. All checkpoint and loss-parity gates passed.\n\n"
        "Compare common12 across K=12/50/200 and common50 across K=50/200, within a ladder, E, seed and final token horizon. "
        "The own-training-set summaries are descriptive. Languages have equal weight. Expert identities are layer-local. "
        "K=1 has no language-expert mutual information.\n\n"
        "Tables preserve historical v1, corrected all-position, and corrected valid-position MI on identical cohorts. "
        "Differences between old and new all-position values can expose the old rank reconciliation defect.\n\n"
        "Intervals use 1,000 paired bootstrap draws of 128-sequence blocks within each language. They describe evaluation-sample "
        "uncertainty conditional on each trained model. B seed summaries report all three seeds plus mean and range, not a seed CI.\n\n"
        +campaign.get('claim_boundary', '')+"\n\nNo manuscript files were changed.\n")
    (out/'README.md').write_text(report)
    release = {'schema':'stage1_corrected_routing_release/v2','status':'complete','campaign_sha256':sha(root/'campaign.json'),
        'gate_policy':campaign['runtime'].get('gate_policy','cross_batch_v2'),
        'table_inventory':inventory,'valid_input_positions':valid_total,'scored_target_tokens':scored_total,
        'bootstrap':campaign['runtime']['analysis'],'claim_boundary':campaign.get('claim_boundary', '')}
    write(out/'release_manifest.json',release)
    return release


def _plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(4,3,figsize=(12,11),sharex=False,sharey=True)
    for i,ladder in enumerate('ABCD'):
        for j,e in enumerate((4,16,32)):
            ax = axes[i,j]
            for k in (12,50,200):
                subset = sorted((r for r in rows if r['ladder']==ladder and r['experts']==e and r['languages']==k and r['seed']==1234 and r['cohort']=='common12'),key=lambda r:r['layer_index'])
                ax.plot([r['layer_index']+1 for r in subset],[r['language_expert_mi_nats'] for r in subset],label=f'K={k}')
            ax.set_title(f'{ladder}, E={e} (final horizon)')
            ax.set_xlabel('Layer')
            ax.set_ylabel('I(language; expert), nats')
    axes[0,0].legend()
    fig.suptitle('Common12, equal language weights; primary seed 1234')
    fig.tight_layout()
    fig.savefig(out/'common12_by_layer.pdf')
    fig.savefig(out/'common12_by_layer.png',dpi=160)
    plt.close(fig)
