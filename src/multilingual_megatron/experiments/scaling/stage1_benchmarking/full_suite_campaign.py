"""Prepare, validate, submit and collect the complete final-checkpoint suite.

Preparation is CPU-only. Submission queues every checkpoint x benchmark job
independently, followed by a CPU collection job dependent on their completion.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from .contracts import _atomic_json, _verify_payload_hash
from .full_suite_budget import estimate
from .full_suite_data import aggregate_metrics

MODULE = 'multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign'
NATIVE = 'multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_native_benchmark'


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def checked(reference):
    path = Path(reference['path'])
    if sha256_file(path) != reference['sha256']:
        raise ValueError(f'artifact drift: {path}')
    return path


def save(path, payload, field=None):
    if field:
        payload[field] = canonical_json_sha256(payload)
    _atomic_json(path, payload)
    return ref(path)


def freeze_source(repo, output, profile):
    """Copy orchestration sources and seal the shared external engine files."""
    source = output / 'source'
    for name in ('src', 'configs'):
        shutil.copytree(repo / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'external').mkdir()
    for name, key in [('Megatron-Bridge', 'bridge_root'), ('lm-evaluation-harness', 'harness_root')]:
        (source / 'external' / name).symlink_to((repo / profile[key]).resolve(), target_is_directory=True)
    references = [ref(path) for path in sorted(source.rglob('*')) if path.is_file()]
    roots = {(repo / profile['bridge_root']).resolve(), (repo / profile['harness_root']).resolve(),
             (repo / profile['bridge_root'] / '3rdparty/Megatron-LM').resolve()}
    engines = []
    for root in sorted(roots):
        engines.append({'path': str(root), 'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()})
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'], text=True).split('\0')
        references.extend(ref(root / name) for name in names if name and Path(name).suffix in
                          {'.py', '.yaml', '.yml', '.json', '.tsv', '.toml'} and (root / name).is_file())
    return save(output / 'source_manifest.json', {'files': references, 'engines': engines})


def prepare_benchmark_inputs(data, output):
    from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import combine_token_caches
    inputs = {}
    for family in data['families']:
        benchmark = family['benchmark']
        request = ref(checked(family))
        languages = [x for x in data['languages'] if x['benchmark'] == benchmark]
        for language in languages:
            payload = read(checked(language['token_cache']))
            if payload['request_sha256'] != language['sha256'] or payload['requests'] != language['requests']:
                raise ValueError('language token cache does not match requests')
        token_cache = combine_token_caches([x['token_cache'] for x in languages],
            output / 'inputs' / benchmark / 'token_cache', request_sha256=request['sha256'],
            tokenizer=data['tokenizer'], sequence_length=data['sequence_length'])
        inputs[benchmark] = {'request_manifest': request, 'token_cache': token_cache,
                             'documents': family['documents'], 'requests': family['requests'],
                             'languages': family['languages']}
    return inputs


def write_job(output, job, profile, repo, *, module=MODULE, job_prefix='s1-full-'):
    directory = output / 'jobs' / job['id']
    directory.mkdir(parents=True)
    runtime = ['#!/bin/bash', 'set -euo pipefail',
               'for campaign_env_key in ${!PROOF_@} ${!BRIDGE_@} ${!NVTE_@} ${!NCCL_@}; do unset "$campaign_env_key"; done',
               'unset MAMBA_DETERMINISTIC TRITON_CACHE_AUTOTUNING TRITON_CACHE_DIR',
               'source ' + shlex.quote(profile['module_init']), 'module purge', 'module load ' + shlex.join(profile['modules'])]
    runtime += ['export ' + key + '=' + shlex.quote(str(value)) for key, value in sorted(profile['environment'].items())]
    runtime += ['export PROOF_PIPELINE_REPO_ROOT=' + shlex.quote(str(output / 'source')),
                'export PYTHONPATH=' + shlex.quote(str(output / 'source/src') + ':' + str(output / 'source')),
                'unset MASTER_ADDR MASTER_PORT',
                'cd ' + shlex.quote(str(repo)),
                'exec ' + shlex.join([profile['python'], '-m', module, 'run', '--campaign', str(output / 'campaign.json'), '--job', job['id']])]
    run = directory / 'run.sh'
    run.write_text('\n'.join(runtime) + '\n')
    slurm = profile['slurm']
    minutes = job['walltime_minutes']
    lines = ['#!/bin/bash', '#SBATCH --job-name=' + job_prefix + job['id'],
             '#SBATCH --account=' + slurm['account'], '#SBATCH --partition=' + slurm['partition'],
             '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --gres=gpu:' + slurm['gpu_type'] + ':1',
             '#SBATCH --cpus-per-task=' + str(slurm['cpus_per_task']), '#SBATCH --mem=' + job['memory'],
             f'#SBATCH --time={minutes // 60:02d}:{minutes % 60:02d}:00',
             '#SBATCH --output=' + str(directory / 'slurm-%j.log'), 'set -euo pipefail',
             'exec ' + shlex.join([str(Path(slurm['bin']) / 'srun'), '--nodes=1', '--ntasks=1', '--cpu-bind=none',
                                   '--cpus-per-task=' + str(slurm['cpus_per_task']), 'bash', str(run)])]
    script = directory / 'job.sbatch'
    script.write_text('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(run)], check=True)
    subprocess.run(['bash', '-n', str(script)], check=True)
    return ref(run), ref(script)


def write_collection_job(output, profile, *, module=MODULE, job_prefix='s1-full-'):
    slurm, resources = profile['slurm'], profile['slurm']['aggregation']
    path = output / 'collect.sbatch'
    lines = ['#!/bin/bash', '#SBATCH --job-name=' + job_prefix + 'collect', '#SBATCH --account=' + slurm['account'],
             '#SBATCH --partition=' + resources['partition'], '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             '#SBATCH --cpus-per-task=' + str(resources['cpus_per_task']), '#SBATCH --mem=' + resources['memory'],
             '#SBATCH --time=' + resources['time'], '#SBATCH --output=' + str(output / 'collect-%j.log'),
             'set -euo pipefail', 'export PYTHONPATH=' + shlex.quote(str(output / 'source/src') + ':' + str(output / 'source')),
             'exec ' + shlex.join([profile['python'], '-m', module, 'collect', '--campaign', str(output / 'campaign.json'), '--require-complete'])]
    path.write_text('\n'.join(lines) + '\n')
    subprocess.run(['bash', '-n', str(path)], check=True)
    return ref(path)


def write_entrypoints(output, profile, campaign, budget):
    """Leave one submit command that uses the frozen orchestration snapshot."""
    for command in ('validate', 'submit', 'collect'):
        path = output / (command + '.sh')
        lines = ['#!/bin/bash', 'set -euo pipefail',
                 'export PYTHONPATH=' + shlex.quote(str(output / 'source/src') + ':' + str(output / 'source')),
                 'exec ' + shlex.join(['conda', 'run', '--no-capture-output', '-p', str(Path(profile['python']).parents[1]),
                                      'python', '-m', MODULE, command, '--campaign', str(output / 'campaign.json')])]
        path.write_text('\n'.join(lines) + '\n')
        subprocess.run(['bash', '-n', str(path)], check=True)
    cost = budget['cohorts']['all_82']
    (output / 'README.md').write_text(f'''# Complete Stage 1 benchmark queue package

Prepared, not submitted: {len(campaign['jobs'])} independent GPU jobs
(82 final checkpoints x 6 benchmarks), plus one automatic CPU collection job.
Every GPU job requests one H100. Each benchmark includes all its languages.
The full suite contains {campaign['documents_per_checkpoint']:,} valid questions
per checkpoint. Exactly 55 approved invalid questions are excluded; see
excluded_questions.csv. There are no sampling caps.

From the login host, queue everything with:

```bash
bash {output}/submit.sh
```

The command validates frozen inputs, queues all GPU jobs without dependencies
between them or an application concurrency limit, and queues CPU collection
after the GPU jobs finish. Slurm chooses compute allocation. Job IDs are saved
in submission.json. Repeated submission is refused to prevent duplicate jobs.
If submission is interrupted, inspect that journal before retrying.

The CPU collection job writes:

- benchmark_metrics.csv: 492 model-benchmark summaries, including language-macro
  and example-micro accuracy and normalized accuracy.
- language_metrics.csv: 18,532 model-benchmark-language rows when complete.
- metrics.json: provenance, metrics, completion status and missing-job IDs.

Missing GPU results produce a partial report and a nonzero collection exit.
Invalid score artifacts fail validation. No GPU is allocated for collection.
Optional manual commands are validate.sh and collect.sh in this directory.

The measured-calibration forecast is {cost['h100_low_hours']:.0f}–{cost['h100_high_hours']:.0f}
H100-hours; reserve about {math.ceil(cost['reserve_hours'] / 10) * 10} H100-hours.
This is an extrapolation, not a completed-campaign runtime. Pilot cost is additional.
Inspect campaign.json, h100_budget.json, source_manifest.json, inputs/, specs/
and jobs/. Preserve these artifacts and the referenced data/engine paths while
jobs are queued or running. The scripts execute the sealed source snapshot.
''')


def prepare(profile_path, output):
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import load_native_benchmark_score_spec
    profile_path, output = profile_path.resolve(), output.resolve()
    profile = read(profile_path)
    repo = (profile_path.parent / profile['repo_root']).resolve()
    if (output / 'campaign.json').exists() or (output / 'source').exists():
        raise ValueError('refusing to overwrite a prepared campaign')
    data_path = repo / profile['data_manifest']
    data = read(data_path)
    _verify_payload_hash(data, field='manifest_sha256', label='full data')
    if data['sampling'] is not None or data['unsupported_long_targets']:
        raise ValueError('sampled or unsupported full data')
    if len(data['languages']) != 226 or data['excluded_documents'] != 55:
        raise ValueError('full-suite coverage differs from the approved inventory')
    if (sum(x['documents'] for x in data['languages']) != 1021935
            or sum(x['requests'] for x in data['languages']) != 4042495):
        raise ValueError('unexpected full-suite document or request count')
    legacy = read(repo / profile['legacy_budget'])
    models = legacy['models']
    if len(models) != 82 or len({x['model_id'] for x in models}) != 82:
        raise ValueError('expected exactly 82 final checkpoints')
    budget = estimate(models, data, repo / profile['pilot_root'])
    output.mkdir(parents=True, exist_ok=True)
    budget_ref = save(output / 'h100_budget.json', budget)
    inputs = prepare_benchmark_inputs(data, output)
    source_ref = freeze_source(repo, output, profile)
    jobs, cache = [], {}
    source_campaign = Path(legacy['source_campaign'])
    for model in sorted(budget['models'], key=lambda x: (x['ladder'], x['E'], x['K'], x['seed'])):
        name = model['model_id']
        receipt_path = source_campaign / 'runtime' / (name + '.json')
        old = read(receipt_path)
        if old['status'] != 'complete' or old['trajectory_readout']['index'] != old['trajectory_readout']['count']:
            raise ValueError('source is not a complete final checkpoint')
        if (old['model_id'], old['ladder'], old['experts'], old['languages'], old['seed']) != (name, model['ladder'], model['E'], model['K'], model['seed']):
            raise ValueError('final-checkpoint metadata mismatch')
        source = source_campaign / 'specs' / old['request_set'] / (name + '.score_spec.json')
        for benchmark, family in inputs.items():
            job_id = name + '__' + benchmark
            spec = read(source)
            _verify_payload_hash(spec, field='spec_sha256', label='source score spec')
            spec.pop('spec_sha256')
            spec['run_id'] = output.name + '_' + job_id
            spec['output_dir'] = str(output / 'scores' / job_id)
            spec['request_manifest'] = family['request_manifest']
            spec['token_cache'] = family['token_cache']
            spec['model_recipe']['expected_model']['expert_model_parallel_size'] = 1
            spec['model_recipe']['overrides'] = [x for x in spec['model_recipe']['overrides'] if not x.startswith('model.expert_model_parallel_size=')] + ['model.expert_model_parallel_size=1']
            cost = model['benchmarks_h100'][benchmark]
            slurm = profile['slurm']
            minutes = max(slurm['minimum_minutes'], math.ceil((cost['reserve_hours'] * 60 + slurm['extra_minutes']) / slurm['round_minutes']) * slurm['round_minutes'])
            spec['evaluation'].update(expected_world_size=1, required_gpu_name_substring='H100',
                                      micro_batch_size=model['h100_micro_batch_size'], max_gpu_hours=minutes / 60)
            spec['evaluation'].pop('batching', None)
            spec['full_suite_provenance'] = {'source_score_spec': ref(source), 'source_final_runtime': ref(receipt_path),
                                            'data_manifest': ref(data_path), 'profile': ref(profile_path), 'benchmark': benchmark,
                                            'scope': 'all languages of this benchmark, final checkpoint including K1'}
            spec_ref = save(output / 'specs' / (job_id + '.score_spec.json'), spec, 'spec_sha256')
            load_native_benchmark_score_spec(Path(spec_ref['path']), request_manifest_cache=cache)
            job = {'id': job_id, 'model_id': name, 'benchmark': benchmark, 'ladder': model['ladder'],
                   'E': model['E'], 'K': model['K'], 'seed': model['seed'], 'gpus': 1,
                   'memory': slurm['memory_by_benchmark'][benchmark], 'walltime_minutes': minutes, 'spec': spec_ref,
                   'estimated_hours': [cost['h100_low_hours'], cost['h100_high_hours']]}
            job['run_script'], job['sbatch_script'] = write_job(output, job, profile, repo)
            jobs.append(job)
    validate_matrix(jobs, inputs)
    campaign = {'schema': 'stage1_full_final_campaign/v1', 'status': 'prepared_not_submitted',
                'profile': ref(profile_path), 'profile_payload': profile, 'repo': str(repo),
                'source_manifest': source_ref, 'data_manifest': ref(data_path), 'inputs': inputs,
                'budget': budget_ref, 'jobs': jobs, 'collection_job': write_collection_job(output, profile),
                'submission_policy': 'queue all 492 GPU jobs independently; no concurrency limit or dependency between GPU jobs',
                'documents_per_checkpoint': sum(x['documents'] for x in data['languages']),
                'requests_per_checkpoint': sum(x['requests'] for x in data['languages']),
                'excluded_documents': data['excluded_documents'], 'automatic_batching': 'rejected by full-vector prediction checks; fixed measured batches retained'}
    save(output / 'campaign.json', campaign, 'manifest_sha256')
    write_entrypoints(output, profile, campaign, budget)
    print(json.dumps({'campaign': str(output / 'campaign.json'), 'jobs': len(jobs), 'budget': budget['cohorts']}, indent=2))


def validate_matrix(jobs, inputs):
    identities = {(x['model_id'], x['benchmark']) for x in jobs}
    models = {x['model_id'] for x in jobs}
    if len(jobs) != 492 or len(models) != 82 or len(inputs) != 6 or identities != {(m, b) for m in models for b in inputs}:
        raise ValueError('expected exactly one job for each of 82 final models x 6 benchmarks')
    if len({x['id'] for x in jobs}) != len(jobs):
        raise ValueError('duplicate job identifier')


def load_campaign(path):
    campaign = read(path)
    _verify_payload_hash(campaign, field='manifest_sha256', label='campaign')
    if campaign['schema'] != 'stage1_full_final_campaign/v1':
        raise ValueError('invalid full campaign')
    return campaign


def validate(campaign, *, all_specs=True):
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import load_native_benchmark_score_spec
    validate_matrix(campaign['jobs'], campaign['inputs'])
    for key in ('profile', 'source_manifest', 'data_manifest', 'budget', 'collection_job'):
        checked(campaign[key])
    for item in read(Path(campaign['source_manifest']['path']))['files']:
        checked(item)
    for family in campaign['inputs'].values():
        checked(family['request_manifest'])
        for item in read(checked(family['token_cache']))['files'].values():
            checked(item)
    cache = {}
    for job in campaign['jobs']:
        for key in ('spec', 'run_script', 'sbatch_script'):
            checked(job[key])
        if all_specs:
            load_native_benchmark_score_spec(Path(job['spec']['path']), request_manifest_cache=cache)
    return {'validated_jobs': len(campaign['jobs']), 'documents_per_checkpoint': campaign['documents_per_checkpoint'],
            'requests_per_checkpoint': campaign['requests_per_checkpoint'], 'status': 'ready_not_submitted'}


def submit(path, campaign):
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit from the login host; nested submission is forbidden')
    validate(campaign)
    journal = path.parent / 'submission.json'
    with (path.parent / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if journal.exists():
            raise ValueError('submission journal already exists; inspect it before any retry')
        profile = campaign['profile_payload']
        records = []
        _atomic_json(journal, {'status': 'submitting', 'jobs': records})
        for job in campaign['jobs']:
            command = [str(Path(profile['slurm']['bin']) / 'sbatch'), '--parsable']
            command.append(job['sbatch_script']['path'])
            job_id = subprocess.check_output(command, text=True).strip().split(';')[0]
            if not job_id.isdecimal():
                raise RuntimeError('invalid sbatch response')
            records.append({'id': job['id'], 'slurm_job_id': job_id, 'command': command})
            _atomic_json(journal, {'status': 'submitting', 'jobs': records})
        # Completed jobs need no dependency and may already have left slurmctld.
        queued = set(subprocess.check_output([str(Path(profile['slurm']['bin']) / 'squeue'), '--me', '--noheader', '--format=%A'], text=True).split())
        pending = [r['slurm_job_id'] for r in records if r['slurm_job_id'] in queued]
        command = [str(Path(profile['slurm']['bin']) / 'sbatch'), '--parsable']
        if pending:
            command.append('--dependency=afterany:' + ':'.join(pending))
        command.append(campaign['collection_job']['path'])
        collection_id = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not collection_id.isdecimal():
            raise RuntimeError('invalid collection-job response')
        _atomic_json(journal, {'status': 'submitted', 'jobs': records, 'collection_job': {'slurm_job_id': collection_id, 'command': command}})
    return {'submitted_gpu_jobs': len(records), 'automatic_collection_job': collection_id}


def run(campaign, job_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU scoring requires a Slurm allocation')
    job = next(x for x in campaign['jobs'] if x['id'] == job_id)
    checked(job['spec'])
    spec = read(job['spec']['path'])
    directory = Path(spec['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'scores.json').exists() or (directory / 'launch.json').exists():
            raise ValueError('refusing to overwrite an existing scoring attempt')
        for reference in read(checked(campaign['source_manifest']))['files']:
            checked(reference)
        profile = campaign['profile_payload']
        command = [profile['python'], '-m', NATIVE, '--bridge-root', str(Path(campaign['repo']) / profile['bridge_root']),
                   '--spec', job['spec']['path'], '--nproc-per-node', '1', '--python-executable', profile['python'], '--preflight-cuda']
        _atomic_json(directory / 'launch.json', {'slurm_job_id': os.environ['SLURM_JOB_ID'],
                     'node': os.environ.get('SLURMD_NODENAME'), 'gpus': 1, 'command': command})
        subprocess.run(command, check=True)


def collect(path, campaign):
    rows, incomplete = [], []
    summaries = []
    current_benchmark, manifest = None, None
    for job in sorted(campaign['jobs'], key=lambda j: (j['benchmark'], j['model_id'])):
        if job['benchmark'] != current_benchmark:
            current_benchmark = job['benchmark']
            manifest = read(checked(campaign['inputs'][current_benchmark]['request_manifest']))
        spec = read(checked(job['spec']))
        score_path = Path(spec['output_dir']) / 'scores.json'
        if not score_path.exists():
            incomplete.append(job['id'])
            continue
        result = read(score_path)
        _verify_payload_hash(result, field='result_sha256', label='score result')
        if result['status'] != 'complete' or result['score_spec']['sha256'] != job['spec']['sha256'] or result['request_manifest']['sha256'] != campaign['inputs'][current_benchmark]['request_manifest']['sha256']:
            raise ValueError('score provenance mismatch')
        if not result['parity_gate']['passed'] or not result['replica_gate']['passed']:
            raise ValueError('failed score gates')
        if job['E'] > 1 and (not result['moe_routing']['evidence_available'] or result['moe_routing']['totals']['dropped_assignments'] != 0):
            raise ValueError('missing routing evidence or dropped expert assignments')
        metrics = aggregate_metrics(manifest, result['scores'])
        identity = {key: job[key] for key in ('model_id', 'ladder', 'E', 'K', 'seed', 'benchmark')}
        rows.extend({'job_id': job['id'], **identity, **metric} for metric in metrics)
        count = sum(x['samples'] for x in metrics)
        summaries.append({**identity, 'languages': len(metrics), 'samples': count,
            **{f'language_macro_{key}': sum(x[key] for x in metrics) / len(metrics) for key in ('acc', 'acc_norm')},
            **{f'example_micro_{key}': sum(x[key] * x['samples'] for x in metrics) / count for key in ('acc', 'acc_norm')}})
    payload = {'status': 'complete' if not incomplete else 'partial', 'metrics': rows, 'benchmark_summary': summaries, 'incomplete_jobs': incomplete,
               'campaign': ref(path), 'exclusion_policy': '55 explicitly approved invalid source questions; no sampling caps'}
    save(path.parent / 'metrics.json', payload, 'manifest_sha256')
    for filename, records in [('language_metrics.csv', rows), ('benchmark_metrics.csv', summaries)]:
        with (path.parent / filename).open('w', newline='') as handle:
            if records:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return {'status': payload['status'], 'completed_benchmark_jobs': len(summaries), 'metric_rows': len(rows), 'incomplete_jobs': len(incomplete)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--profile', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    for name in ('validate', 'submit', 'run', 'collect'):
        command = commands.add_parser(name)
        command.add_argument('--campaign', type=Path, required=True)
        if name == 'run':
            command.add_argument('--job', required=True)
        if name == 'collect':
            command.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        return prepare(args.profile, args.output)
    campaign = load_campaign(args.campaign)
    if args.command == 'validate':
        result = validate(campaign)
    elif args.command == 'submit':
        result = submit(args.campaign, campaign)
    elif args.command == 'run':
        result = run(campaign, args.job)
    else:
        result = collect(args.campaign, campaign)
    print(json.dumps(result, indent=2))
    if args.command == 'collect' and args.require_complete and result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
