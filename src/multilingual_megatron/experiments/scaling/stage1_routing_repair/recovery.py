"""Recover a frozen campaign only after independent batch-sensitivity controls."""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
from pathlib import Path

from .campaign import identity, read, sha, verify, write


def _without_output(spec):
    return {key: value for key, value in spec.items() if key not in ('run_id', 'output_dir')}


def freeze_recovery_sources(repo, destination, recovery):
    """Keep the evaluated engine and collector; overlay explicit orchestration files."""
    verify(recovery['base_campaign'])
    base = read(recovery['base_campaign']['path'])
    source = Path(base['root'])/'source'
    if destination.resolve().is_relative_to(Path(base['root']).resolve()):
        raise ValueError('recovery must not write inside the original campaign')
    for entry in base['source']['files']:
        verify(entry)
        target = destination/Path(entry['path']).relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry['path'], target)
        verify({'path': str(target), 'sha256': entry['sha256']})
    package = Path('src/multilingual_megatron/experiments/scaling/stage1_routing_repair')
    for name in recovery['source_overlays']:
        path = Path(name)
        allowed = (path.parent == package and path.suffix == '.py') or name in (
            'src/multilingual_megatron/pipeline/executor.py',
            'configs/benchmarks/stage1_routing_repair_final_v3.json')
        if not allowed:
            raise ValueError(f'recovery overlay cannot change evaluation code: {name}')
        entry = identity(repo/path)
        target = destination/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry['path'], target)
        verify({'path': str(target), 'sha256': entry['sha256']})
    return [identity(p) for p in sorted(destination.rglob('*')) if p.is_file()]


def verify_panel_execution(tasks, base):
    """Check preserved production contracts and matched execution within ladder/E."""
    previous = {t['id']: t for t in base['tasks']}
    groups = {}
    if {t['id'] for t in tasks} != set(previous):
        raise ValueError('recovery changed the checkpoint/pilot panel')
    for task in tasks:
        old = previous[task['id']]
        if any(task[key] != old[key] for key in ('model', 'gpus', 'languages', 'kind')):
            raise ValueError(f"recovery changed the task contract: {task['id']}")
        for variants in task['specs'].values():
            for name, entry in variants.items():
                reference = old['specs']['1'][name]
                verify(reference)
                if _without_output(read(entry['path'])) != _without_output(read(reference['path'])):
                    raise ValueError(f"recovery changed evaluation settings: {task['id']}/{name}")
        if task['kind'] == 'production':
            spec = read(task['specs']['1']['corrected']['path'])
            setting = (task['gpus'], spec['eval_micro_batch_size'], spec['model_recipe'])
            key = (task['model']['ladder'], task['model']['experts'])
            if key in groups and groups[key] != setting:
                raise ValueError(f'unmatched evaluation settings across K/seeds: {key}')
            groups[key] = setting


def validate_control_task(campaign, diagnosis, task, base):
    """Recompute comparisons from evaluator artifacts, not receipt assertions."""
    from .execution import load_variant_evidence, loss_parity, validate_variant
    old = next(t for t in base['tasks'] if t['id'] == task['id'])
    if any(task[key] != old[key] for key in ('model', 'gpus', 'languages')) or task['old_specs'] != old['specs']['1']:
        raise ValueError('control does not reference the original pilot')
    receipt_path = Path(diagnosis['root'])/'receipts'/f"{task['id']}.json"
    receipt = read(receipt_path)
    if receipt.get('execution_status') != 'complete' or receipt.get('task') != task['id']:
        raise ValueError(f"control execution incomplete: {task['id']}")
    evidence = [identity(receipt_path)]
    rows = {}
    labels = ('original_disabled', 'half_disabled', 'half_enabled_repeat')
    if set(task['specs']) != set(labels):
        raise ValueError('control variants differ from the declared design')
    for label in labels:
        entry = task['specs'][label]
        reference = task['old_specs']['baseline' if label == 'original_disabled' else 'alternate_batch']
        verify(reference)
        expected = copy.deepcopy(read(reference['path']))
        if label == 'half_disabled':
            expected['routing_diagnostics']['enabled'] = False
            expected['require_moe_routing_diagnostics'] = False
        verify(entry)
        if _without_output(read(entry['path'])) != _without_output(expected):
            raise ValueError(f'control settings changed beyond diagnostics/batch size: {label}')
        rows[label], manifest = load_variant_evidence(task, entry, baseline=label != 'half_enabled_repeat')
        evidence.extend((entry, manifest))
    rows['previous_original'], _, manifest = validate_variant(old, task['old_specs']['baseline'], base, baseline=True)
    evidence.append(manifest)
    rows['previous_half'], manifest = load_variant_evidence(old, task['old_specs']['alternate_batch'])
    evidence.append(manifest)
    comparisons = {}
    for left, right in (('original_disabled', 'previous_original'),
                        ('half_disabled', 'half_enabled_repeat'),
                        ('half_enabled_repeat', 'previous_half')):
        comparisons[f'{left}_vs_{right}'] = {}
        for language in task['languages']:
            a, b = rows[left][language], rows[right][language]
            comparison = loss_parity(a, b, campaign['runtime']['parity'])
            comparison['nll_sum_exactly_equal'] = a['nll_sum'] == b['nll_sum']
            if left == 'half_enabled_repeat':
                for x, y in zip(a['moe_routing']['layers'], b['moe_routing']['layers'], strict=True):
                    if any(x[key] != y[key] for key in ('expert_token_counts', 'blocks', 'trace')):
                        raise ValueError(f'fixed-batch routing repeat changed: {language}, layer {x["layer_index"]}')
                comparison['real_counts_blocks_and_bounded_trace_equal'] = True
            comparisons[f'{left}_vs_{right}'][language] = comparison
    sensitivity = {}
    for language in task['languages']:
        a, b = rows['half_disabled'][language], rows['original_disabled'][language]
        if a['token_count'] != b['token_count']:
            raise ValueError('changing batch size changed the scored-token denominator')
        delta = abs(a['nll_sum']-b['nll_sum'])/b['token_count']
        relative = abs(a['nll_sum']-b['nll_sum'])/max(abs(b['nll_sum']), 1e-30)
        tolerance = campaign['runtime']['parity']
        sensitivity[language] = {'raw_nll_absolute_difference': delta,
            'nll_sum_relative_difference': relative,
            'exceeds_original_tolerances': delta > tolerance['raw_nll_abs_tolerance'] or relative > tolerance['nll_sum_relative_tolerance']}
    if not any(row['exceeds_original_tolerances'] for row in sensitivity.values()):
        raise ValueError('disabled controls did not reproduce the observed batch-size sensitivity')
    return {'task': task['id'], 'comparisons': comparisons, 'batch_sensitivity': sensitivity}, evidence


def controls_gate(campaign):
    """CPU prerequisite for every recovery GPU attempt; failure leaves no success seal."""
    root = Path(campaign['root'])
    output = root/'controls_complete.json'
    if output.exists():
        return require_controls(campaign)
    seal = read(root/'input_seal.json')
    if seal['status'] != 'complete' or seal['campaign_sha256'] != sha(root/'campaign.json'):
        raise ValueError('input seal is missing or belongs to a different campaign')
    recovery = campaign['runtime']['recovery']
    evidence = [recovery[key] for key in ('base_campaign', 'control_manifest', 'collector_test')]
    for entry in evidence:
        verify(entry)
    base = read(recovery['base_campaign']['path'])
    diagnosis = read(recovery['control_manifest']['path'])
    if diagnosis['source_campaign']['sha256'] != recovery['base_campaign']['sha256']:
        raise ValueError('controls used a different source campaign')
    for key in ('parity', 'environment', 'python', 'bridge_commit'):
        if campaign['runtime'][key] != base['runtime'][key] or diagnosis['runtime'][key] != base['runtime'][key]:
            raise ValueError(f'recovery changed a frozen runtime contract: {key}')
    evidence.extend((diagnosis['source_campaign'], diagnosis['input_seal'], diagnosis['runner']))
    for entry in evidence + base['source']['files'] + campaign['source']['files']:
        verify(entry)
    if {t['id'] for t in diagnosis['tasks']} != {'pilot_c', 'pilot_d'}:
        raise ValueError('both C and D controls are required')
    reports = []
    for task in diagnosis['tasks']:
        report, artifacts = validate_control_task(campaign, diagnosis, task, base)
        reports.append(report)
        evidence.extend(artifacts)
    # Exercise collector partition/padding/ownership invariance with fixed
    # assignments, using the original tests and frozen evaluated implementation.
    test = Path(recovery['collector_test']['path'])
    env = dict(os.environ, PYTHONPATH=f'{root}/source/src:{root}/source')
    log = root/'logs/collector_contract_tests.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([campaign['runtime']['python'], '-m', 'unittest', 'discover',
            '-s', str(test.parent), '-p', test.name, '-v'], cwd=root/'source', env=env,
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    evidence.append(identity(log))
    result = {'schema': 'stage1_routing_recovery_controls/v1', 'status': 'complete',
        'campaign_sha256': sha(root/'campaign.json'), 'controls': reports,
        'evidence': evidence, 'gate_policy': 'fixed_execution_v3'}
    write(output, result)
    (root/'controls_complete.sha256').write_text(sha(output)+'\n')
    return result


def require_controls(campaign):
    root = Path(campaign['root'])
    path = root/'controls_complete.json'
    if sha(path) != (root/'controls_complete.sha256').read_text().strip():
        raise ValueError('recovery control seal changed')
    result = read(path)
    if result['status'] != 'complete' or result['campaign_sha256'] != sha(root/'campaign.json'):
        raise ValueError('recovery controls have not passed for this campaign')
    for entry in result['evidence']:
        verify(entry)
    return result
