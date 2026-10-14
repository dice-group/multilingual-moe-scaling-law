"""Freeze the scientific panel, source code, and evaluation contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

SCHEMA = "stage1_routing_repair_campaign/v2"
RECOVERY_SCHEMA = "stage1_routing_repair_campaign/v3"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def verify(entry):
    if sha(entry['path']) != entry['sha256']:
        raise ValueError(f"sealed artifact changed: {entry['path']}")


def load_campaign(root):
    root = Path(root).resolve()
    manifest = root / 'campaign.json'
    expected = (root / 'campaign.sha256').read_text().strip()
    if sha(manifest) != expected:
        raise ValueError('campaign manifest hash drift')
    result = read(manifest)
    if result['schema'] not in (SCHEMA, RECOVERY_SCHEMA):
        raise ValueError('unsupported campaign version')
    return result


def checkpoint_inventory(spec):
    root = Path(spec['checkpoint']['path'])
    step = spec['checkpoint']['step']
    if int((root / 'latest_checkpointed_iteration.txt').read_text()) != step:
        raise ValueError(f'checkpoint tracker drift: {root}')
    candidates = [root / f'iter_{step:07d}', root / f'iter_{step:010d}', root / f'iter_{step}']
    directory = next((p for p in candidates if p.is_dir()), None)
    if directory is None or not (directory / '.metadata').is_file():
        raise ValueError(f'checkpoint metadata missing: {root}')
    files = sorted(directory.rglob('*.distcp'))
    if not files or any(p.stat().st_size == 0 for p in files):
        raise ValueError(f'checkpoint shards missing/empty: {directory}')
    return [identity(directory / '.metadata')], [
        {'path': str(p), 'size_bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files]


def validate_panel(models):
    expected = {(l,e,k,1234) for l in 'ABCD' for e in (4,16,32) for k in (1,12,50,200)}
    expected |= {('B',e,k,s) for e in (16,32) for k in (1,12,200) for s in (5678,91011)}
    cells = {(m['ladder'],m['experts'],m['languages'],m['seed']) for m in models}
    if len(models) != 60 or cells != expected or sum(m['languages'] for m in models) != 4008:
        raise ValueError('final sparse panel is not the declared 60-checkpoint / 4008-stream design')
    tokens = {'A':8000372736,'B':16000483328,'C':32000442368,'D':32000442368}
    if any(m['tokens_seen'] != tokens[m['ladder']] or not m['is_final_readout'] for m in models):
        raise ValueError('final training horizon drift')


def freeze_sources(repo, destination):
    """Copy runnable sources, including uncommitted repair files, without data or git state."""
    pairs = [(repo / 'src', destination / 'src'), (repo / 'configs', destination / 'configs'),
        (repo / 'external/Megatron-Bridge/src', destination / 'external/Megatron-Bridge/src'),
        (repo / 'external/Megatron-Bridge/3rdparty/Megatron-LM/megatron', destination / 'external/Megatron-Bridge/3rdparty/Megatron-LM/megatron')]
    for source, target in pairs:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for copied in target.rglob('*'):
            if copied.is_file() and sha(copied) != sha(source / copied.relative_to(target)):
                raise ValueError(f'source changed during freeze: {copied}')
    for name in ('pyproject.toml',):
        shutil.copy2(repo / name, destination / name)
    # Stage recipes use the package tree; Bridge and Core remain unmodified copies.
    return [identity(p) for p in sorted(destination.rglob('*')) if p.is_file()]


def prepare(profile_path):
    # Checkpoint resolution needs the training runtime; summary replay does not.
    from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_models import resolve_checkpoint_models

    profile_path = Path(profile_path).resolve()
    profile = read(profile_path)
    if profile['schema'] not in ('stage1_routing_repair_profile/v2', 'stage1_routing_repair_profile/v3'):
        raise ValueError('invalid profile')
    recovery = profile['schema'].endswith('/v3')
    if recovery != (profile.get('gate_policy') == 'fixed_execution_v3'):
        raise ValueError('profile version and pilot gate policy disagree')
    attempts = profile.get('retries', {}).get('maximum_attempts', 3)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 3:
        raise ValueError('maximum_attempts must be between one and three')
    if recovery:
        for key in ('base_campaign', 'control_manifest', 'collector_test'):
            entry = profile['recovery'][key]
            entry['path'] = str((profile_path.parent / entry['path']).resolve())
            verify(entry)
    repo = (profile_path.parent / profile['repo_root']).resolve()
    root = (profile_path.parent / profile['output_root']).resolve()
    if root.exists():
        if (root / 'campaign.sha256').exists():
            existing = load_campaign(root)
            if existing['profile']['sha256'] != sha(profile_path):
                raise ValueError('existing campaign uses a different profile')
            return existing
        raise ValueError(f'incomplete preparation exists; retain it and use a new output root: {root}')
    release_path = (profile_path.parent / profile['source']['release_manifest']).resolve()
    verify({'path':str(release_path),'sha256':profile['source']['release_manifest_sha256']})
    models, _ = resolve_checkpoint_models(profile, read(release_path), release_path)
    models = [m for m in models if m['experts'] > 1]
    validate_panel(models)
    bridge = (profile_path.parent / profile['bridge_root']).resolve()
    commit = subprocess.check_output(['git','-C',str(bridge),'rev-parse','HEAD'], text=True).strip()
    if commit != profile['bridge_commit']:
        raise ValueError('Bridge commit drift')
    root.mkdir(parents=True)
    profile['repo_root'] = str(repo)
    profile['bridge_root'] = str(bridge)
    profile['output_root'] = str(root)
    tasks = []
    checkpoint_files = {}
    cohorts = None
    historical = 0.
    for model in models:
        source = read(model['evaluation_spec'])
        original = Path(model['evaluation_spec']).parent
        invocation = read(original / 'bridge_validation_command.json')
        manifest = read(model['evaluation_manifest'])
        timing = read(original / 'bridge_validation_launcher_result.json')
        metadata, shards = checkpoint_inventory(source)
        for entry in metadata + shards:
            checkpoint_files[entry['path']] = entry
        languages = source['trained_languages']
        if len(languages) != model['languages']:
            raise ValueError('trained language coverage differs from selected model')
        selected_seconds = sum(manifest['languages'][l]['forward_seconds'] for l in languages)
        total_seconds = sum(x['forward_seconds'] for x in manifest['languages'].values())
        estimated_seconds = selected_seconds + max(0., timing['runtime_seconds'] - total_seconds)
        gpus = invocation['nproc_per_node'] * invocation['nnodes']
        historical += estimated_seconds * gpus / 3600
        cohorts = source['cohorts'] if cohorts is None else cohorts
        if cohorts != source['cohorts']:
            raise ValueError('cohort identity drift')
        source['required_languages'] = languages
        source['routing_diagnostics'] = {'enabled':True,'extra_padding_batches':0}
        source['require_moe_routing_diagnostics'] = True
        tasks.append({'id':model['id'], 'kind':'production', 'model':model, 'source_spec':source,
            'languages':languages, 'gpus':gpus, 'estimated_seconds':estimated_seconds,
            'time_minutes':15 * math.ceil(max(30., (2*estimated_seconds + 900)/60)/15)})
    for ladder in 'BCD':
        base = next(t for t in tasks if t['model']['ladder']==ladder and t['model']['experts']==32 and t['model']['languages']==200 and t['model']['seed']==1234)
        pilot = copy.deepcopy(base)
        pilot.update(id=f'pilot_{ladder.lower()}',kind='pilot',languages=(cohorts['k12']+['bam_Latn']) if ladder=='B' else ['eng_Latn','bam_Latn'])
        # Recovery keeps the original batch shape. V2 also required full-model
        # invariance after halving the microbatch; its failed gate is preserved.
        pilot['time_minutes'] = profile.get('pilot_time_minutes', {}).get(ladder, 120)
        tasks.append(pilot)
    for task in tasks:
        task['specs'] = {}
        variants = ('baseline','corrected','extra_padding') if task['kind']=='pilot' else ('corrected',)
        if task['kind']=='pilot' and not recovery:
            variants += ('alternate_batch',)
        for attempt in range(1, attempts + 1):
            task['specs'][str(attempt)] = {}
            for variant in variants:
                spec = copy.deepcopy(task['source_spec'])
                spec['run_id'] = f"{profile['campaign_id']}_{task['id']}_a{attempt}_{variant}"
                spec['output_dir'] = str(root / 'attempts' / task['id'] / str(attempt) / variant)
                spec['trained_languages'] = spec['required_languages'] = task['languages']
                if variant == 'baseline':
                    spec['routing_diagnostics']['enabled'] = False
                    spec['require_moe_routing_diagnostics'] = False
                elif variant == 'extra_padding':
                    spec['routing_diagnostics']['extra_padding_batches'] = 1
                elif variant == 'alternate_batch':
                    spec['eval_micro_batch_size'] = max(1,spec['eval_micro_batch_size']//2)
                path = root / 'specs' / task['id'] / str(attempt) / f'{variant}.json'
                write(path,spec)
                task['specs'][str(attempt)][variant] = identity(path)
        del task['source_spec']
    if recovery:
        from .recovery import freeze_recovery_sources, verify_panel_execution
        source_inventory = freeze_recovery_sources(repo, root/'source', profile['recovery'])
        verify_panel_execution(tasks, read(profile['recovery']['base_campaign']['path']))
    else:
        source_inventory = freeze_sources(repo, root / 'source')
    source_info = {'superproject_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'bridge_commit':commit, 'core_commit':subprocess.check_output(['git','-C',str(bridge / '3rdparty/Megatron-LM'),'rev-parse','HEAD'],text=True).strip(),
        'files':source_inventory}
    if recovery:
        source_info.update(base_campaign=profile['recovery']['base_campaign'],
            overlays=[identity(root/'source'/name) for name in profile['recovery']['source_overlays']])
    campaign = {'schema':RECOVERY_SCHEMA if recovery else SCHEMA,'id':profile['campaign_id'],'root':str(root),'profile':identity(profile_path),
        'runtime':profile,'release':identity(release_path),'source':source_info,'tasks':tasks,'cohorts':cohorts,
        'checkpoint_files':list(checkpoint_files.values()),'checkpoint_count':60,'language_stream_count':4008,
        'historical_gpu_hours':historical}
    write(root / 'campaign.json',campaign)
    (root / 'campaign.sha256').write_text(sha(root / 'campaign.json')+'\n')
    return campaign


def seal(campaign):
    """CPU gate: shared data hashes, code/spec identities, and all shard hashes."""
    from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import load_evaluation_spec
    root = Path(campaign['root'])
    verify(campaign['profile'])
    verify(campaign['release'])
    for entry in campaign['source']['files']:
        verify(entry)
    for task in campaign['tasks']:
        verify({'path':task['model']['evaluation_spec'],'sha256':task['model']['evaluation_spec_sha256']})
        for variants in task['specs'].values():
            for entry in variants.values():
                verify(entry)
    # One K200 spec covers every shared validation prefix, unigram and tokenizer.
    full = next(t for t in campaign['tasks'] if t['kind']=='production' and t['model']['languages']==200)
    load_evaluation_spec(Path(full['specs']['1']['corrected']['path']))
    sealed = []
    for index, entry in enumerate(campaign['checkpoint_files'], 1):
        path = Path(entry['path'])
        if path.stat().st_size != entry['size_bytes'] or ('mtime_ns' in entry and path.stat().st_mtime_ns != entry['mtime_ns']):
            raise ValueError(f'checkpoint changed after preparation: {path}')
        if 'sha256' in entry:
            verify(entry)
        sealed.append(identity(path))
        print(f"[routing-seal] {index}/{len(campaign['checkpoint_files'])}: {path.name}, {entry['size_bytes']} bytes", flush=True)
    write(root / 'input_seal.json',{'status':'complete','campaign_sha256':sha(root/'campaign.json'),'checkpoint_files':sealed})
