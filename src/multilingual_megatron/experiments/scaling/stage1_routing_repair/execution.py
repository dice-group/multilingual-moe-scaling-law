"""Run a sealed checkpoint task and validate its results before release."""
from __future__ import annotations

import math
import errno
import os
import subprocess
import time
from pathlib import Path

from .campaign import identity, read, sha, verify, write


def loss_parity(observed, reference, tolerances):
    if observed['token_count'] != reference['token_count']:
        raise ValueError('scored-token denominator changed')
    nll = float(observed['nll_sum']) / observed['token_count']
    old = float(reference['nll_sum']) / reference['token_count']
    delta = abs(nll-old)
    relative = abs(observed['nll_sum']-reference['nll_sum']) / max(abs(reference['nll_sum']),1e-30)
    if not math.isfinite(nll) or delta > tolerances['raw_nll_abs_tolerance'] or relative > tolerances['nll_sum_relative_tolerance']:
        raise ValueError(f'loss parity failed: raw NLL delta={delta}, sum relative delta={relative}')
    return {'raw_nll_abs_difference':delta,'nll_sum_relative_difference':relative,'scored_tokens':observed['token_count']}


def load_variant_evidence(task, entry, *, baseline=False):
    """Verify an evaluator artifact independently of its comparison reference."""
    verify(entry)
    spec = read(entry['path'])
    output = Path(spec['output_dir'])
    if not (output / '.validation_complete').is_file():
        raise ValueError(f'validation completion marker missing: {output}')
    manifest = read(output / 'evaluation_manifest.json')
    if manifest['status'] != 'complete' or manifest['spec']['sha256'] != entry['sha256']:
        raise ValueError('evaluation did not complete the frozen spec')
    if manifest['checkpoint']['step'] != task['model']['checkpoint_step']:
        raise ValueError('loaded checkpoint step differs from selected final checkpoint')
    if set(manifest['languages']) != set(task['languages']):
        raise ValueError('language coverage differs from frozen task')
    rows = {}
    for language, artifact in manifest['languages'].items():
        verify(artifact)
        row = read(artifact['path'])
        if row['token_count'] <= 0 or not math.isfinite(row['nll_sum']):
            raise ValueError('nonfinite or empty language loss')
        if not baseline:
            routing = row['moe_routing']
            if routing['schema'] != 'multilingual_moe_routing_diagnostics/v2' or not routing['evidence_available'] or routing['evidence_errors']:
                raise ValueError('required v2 routing evidence unavailable')
            expected_layers = spec['model_recipe']['expected_model']['num_layers']
            if [r['layer_index'] for r in routing['layers']] != list(range(expected_layers)):
                raise ValueError('layer coverage differs')
            n = row['backend']['stream_tokens_including_special']
            for layer in routing['layers']:
                counts = layer['expert_token_counts']
                if len(counts) != task['model']['experts'] or layer['topk'] != 1:
                    raise ValueError('expert/top-k contract changed')
                if sum(counts) != n or layer['forwarded_tokens'] != n or layer['dropped_assignments']:
                    raise ValueError('valid assignment count conservation failed')
                if [a+b for a,b in zip(counts,layer['padding_expert_token_counts'])] != layer['all_expert_token_counts']:
                    raise ValueError('valid + padding != all assignments')
                if [sum(v[i] for v in layer['blocks'].values()) for i in range(len(counts))] != counts:
                    raise ValueError('block counts do not reconstruct layer counts')
        rows[language] = row
    return rows, identity(output / 'evaluation_manifest.json')


def validate_variant(task, entry, campaign, *, baseline=False):
    rows, manifest = load_variant_evidence(task, entry, baseline=baseline)
    historical = read(task['model']['evaluation_manifest'])
    parity = {}
    for language, row in rows.items():
        reference = historical['languages'][language]
        verify(reference)
        parity[language] = loss_parity(row, read(reference['path']), campaign['runtime']['parity'])
    return rows, parity, manifest


def validate_attempt(campaign, task, attempt):
    variants = task['specs'][str(attempt)]
    fixed_execution = campaign['runtime'].get('gate_policy') == 'fixed_execution_v3'
    expected = {'baseline','corrected','extra_padding'} if task['kind']=='pilot' else {'corrected'}
    if task['kind']=='pilot' and not fixed_execution:
        expected.add('alternate_batch')
    if set(variants) != expected:
        raise ValueError('missing or unexpected evaluation variant')
    values = {}
    parity = {}
    manifests = {}
    for name, entry in variants.items():
        values[name], parity[name], manifests[name] = validate_variant(task,entry,campaign,baseline=name=='baseline')
    if task['kind'] == 'pilot':
        for language in task['languages']:
            corrected = values['corrected'][language]
            for name in sorted(expected - {'corrected'}):
                loss_parity(values[name][language],corrected,campaign['runtime']['parity'])
            for name in sorted(expected - {'corrected','baseline'}):
                other = values[name][language]
                for left,right in zip(corrected['moe_routing']['layers'],other['moe_routing']['layers'],strict=True):
                    # Any changed expert decision/count blocks publication; no
                    # automatic tolerance widening or topology changes.
                    for key in ('expert_token_counts','trace'):
                        if left[key] != right[key]:
                            raise ValueError(f'{name}: real routing changed in {language}, layer {left["layer_index"]}, {key}')
    return {'status':'complete','task_id':task['id'],'attempt':attempt,'manifests':manifests,'parity':parity}


def scheduler_state(job_id, executable='sacct'):
    result = subprocess.run([executable,'-n','-X','-P','-j',str(job_id),'--format=JobIDRaw,State'],capture_output=True,text=True,check=True,timeout=30)
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if fields[0] == str(job_id):
            return fields[1].split()[0].rstrip('+')
    return 'UNKNOWN'


def retry_allowed(campaign, task_id, attempt):
    if attempt == 1:
        return True
    root = Path(campaign['root'])
    # Successful or scientifically invalid attempts stop all subsequent work.
    for previous in range(1,attempt):
        receipt = root / 'attempts' / task_id / str(previous) / 'receipt.json'
        if receipt.exists():
            status = read(receipt)['status']
            if status in ('complete','invalid','skipped'):
                return False
    journal = read(root / 'submission.json')
    prior = root / 'attempts' / task_id / str(attempt-1) / 'receipt.json'
    if prior.exists() and read(prior)['status']=='infrastructure_failure':
        return True
    job = journal['jobs'][f'{task_id}.a{attempt-1}']['job_id']
    executable = campaign['runtime'].get('slurm_commands', {}).get('sacct', 'sacct')
    return scheduler_state(job, executable) in campaign['runtime']['retries']['allowed_scheduler_states']


def run_task(campaign, task_id, attempt):
    root = Path(campaign['root'])
    task = next(t for t in campaign['tasks'] if t['id']==task_id)
    receipt_path = root / 'attempts' / task_id / str(attempt) / 'receipt.json'
    if receipt_path.exists():
        raise ValueError('attempt already has a receipt; refusing to overwrite')
    if not retry_allowed(campaign,task_id,attempt):
        write(receipt_path,{'status':'skipped','reason':'predecessor is not a verified infrastructure failure'})
        return
    seal = read(root / 'input_seal.json')
    if seal['status']!='complete' or seal['campaign_sha256'] != sha(root/'campaign.json'):
        raise ValueError('shared input seal missing or invalid')
    if campaign['runtime'].get('gate_policy') == 'fixed_execution_v3':
        from .recovery import require_controls
        require_controls(campaign)
    started = time.time()
    try:
        # Every selected checkpoint is freshly loaded. Matching files alone are
        # never reported as a successful load. Hash just this model's shards.
        checkpoint = read(task['specs'][str(attempt)]['corrected']['path'])['checkpoint']['path']
        for entry in seal['checkpoint_files']:
            if Path(entry['path']).is_relative_to(checkpoint):
                verify(entry)
        for entry in campaign['source']['files']:
            verify(entry)
        variants = task['specs'][str(attempt)]
        for variant in ('baseline', 'corrected', 'extra_padding', 'alternate_batch'):
            if variant not in variants:
                continue
            entry = variants[variant]
            verify(entry)
            output = Path(read(entry['path'])['output_dir'])
            if output.exists():
                raise ValueError(f'attempt output already exists: {output}')
            log_dir = root / 'logs' / 'attempts' / f'{task_id}.a{attempt}' / variant
            command = [campaign['runtime']['python'],'-m',
                'multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_per_language_validation',
                '--bridge-root',str(root/'source/external/Megatron-Bridge'),
                '--spec',entry['path'],'--nproc-per-node',str(task['gpus']),
                '--python-executable',campaign['runtime']['python'],'--preflight-cuda','--log-dir',str(log_dir)]
            subprocess.run(command,check=True,cwd=root/'source')
            # Stop pilot immediately on the first invalid variant.
            validate_variant(task,entry,campaign,baseline=variant=='baseline')
        receipt = validate_attempt(campaign,task,attempt)
        receipt['runtime_seconds'] = time.time()-started
        receipt['gpu_hours'] = receipt['runtime_seconds']*task['gpus']/3600
        write(receipt_path,receipt)
    except Exception as exc:
        status = 'invalid'
        if isinstance(exc, subprocess.CalledProcessError):
            # Only a subsequently verified Slurm infrastructure state permits
            # retry; FAILED, timeout, OOM, and unknown remain non-retryable.
            status = 'runtime_failure'
        elif isinstance(exc, OSError) and exc.errno in (errno.ESTALE, errno.ETIMEDOUT):
            status = 'infrastructure_failure'
        write(receipt_path,{'status':status,'error':repr(exc),'runtime_seconds':time.time()-started})
        raise


def complete_task(campaign, task_id):
    root = Path(campaign['root'])
    task = next(t for t in campaign['tasks'] if t['id']==task_id)
    candidates = []
    for attempt in range(1, campaign['runtime'].get('retries', {}).get('maximum_attempts', 3) + 1):
        path = root / 'attempts' / task_id / str(attempt) / 'receipt.json'
        if path.exists() and read(path)['status']=='complete':
            # Revalidate artifacts independently in the CPU completion gate.
            receipt = validate_attempt(campaign,task,attempt)
            candidates.append((path,receipt))
    if len(candidates)!=1:
        raise ValueError(f'{task_id}: expected one verified success, got {len(candidates)}')
    path, receipt = candidates[0]
    write(root/'results'/f'{task_id}.json',{'status':'complete','receipt':identity(path),**receipt})


def pilot_gate(campaign):
    root = Path(campaign['root'])
    if campaign['runtime'].get('gate_policy') == 'fixed_execution_v3':
        from .recovery import require_controls
        require_controls(campaign)
    entries = []
    for name in ('pilot_b','pilot_c','pilot_d'):
        path = root / 'results' / f'{name}.json'
        if not path.exists() or read(path)['status']!='complete':
            raise ValueError(f'pilot has not passed: {name}')
        entries.append(identity(path))
    write(root/'pilots_complete.json',{'status':'complete','pilots':entries})
