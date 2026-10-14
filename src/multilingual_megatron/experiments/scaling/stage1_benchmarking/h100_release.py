"""Seal and validate a uniform-H100 replacement for all Stage 1 paper tasks.

Historical campaigns are immutable lineage inputs. Their task-specific reducers
remain authoritative; this module owns only the shared execution and submission
contract. Preparing a release never submits it.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
from collections import Counter, defaultdict
from statistics import median

from . import full_suite_campaign as runtime
from .contracts import _verify_payload_hash
from .full_suite_campaign import checked, read, ref, save

MODULE = 'multilingual_megatron.experiments.scaling.stage1_benchmarking.h100_release'
SCHEMA = 'stage1_h100_paper_release/v1'


def packages():
    return sorted((d.metadata['Name'].lower(), d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name'))


def batch_map(profile):
    return profile.get('batches_by_architecture',profile.get('batches_by_tier'))


def batch_for(batches,ladder,experts):
    value=batches.get(f'{ladder}_e{experts}',batches.get(ladder))
    if not isinstance(value,int) or isinstance(value,bool) or value<1:raise ValueError('missing fixed architecture batch')
    return value


def check_gate(gate):
    expected={(a,e,200) for a in 'ABCD' for e in (1,4,16,32)}|{(a,e,12) for a in 'ABCD' for e in (1,32)}
    rows=gate.get('comparisons',[])
    if (not gate.get('complete') or not gate.get('passed') or len(rows)!=24
            or {(r['ladder'],r['E'],r['K']) for r in rows}!=expected
            or any(not r['passed'] or not math.isfinite(r['max_score_difference']) or not math.isfinite(r['max_per_token_difference']) for r in rows)):
        raise ValueError('numerical gate lacks the complete accepted architecture/coverage panel')
    if gate.get('acceptance_policy')=='model_specific_memory_safe_with_reported_sensitivity':
        # User prefers efficient model-specific batches. Cross-batch agreement
        # is measured and disclosed, never relabelled as numerical invariance.
        metrics=gate.get('metric_differences',[]);contrasts=gate.get('paired_contrasts',[])
        if (not gate.get('runtime_checks_passed') or not gate.get('capacity_checks_passed')
                or len(metrics)!=672 or len(contrasts)!=112
                or any(not math.isfinite(x['absolute_difference']) for x in metrics+contrasts)):
            raise ValueError('model-specific release lacks complete runtime/capacity/metric evidence')
    elif any(r['max_score_difference']>.01 or r['max_per_token_difference']>1e-5 or r['decision_changes']!=0 for r in rows):
        raise ValueError('numerical gate failed strict cross-batch tolerances')


def validate_matrix(jobs):
    expected = {(a,e,k,1234) for a in 'ABCD' for e in (1,4,16,32) for k in (1,12,50,200)}
    expected |= {('B',e,k,s) for e in (1,16,32) for k in (1,12,200) for s in (5678,91011)}
    families = {j['benchmark'] for j in jobs}
    observed = {(j['benchmark'],j['ladder'],j['E'],j['K'],j['seed']) for j in jobs}
    if len(jobs)!=1148 or len(families)!=14 or len({j['id'] for j in jobs})!=1148 or observed!={(b,*x) for b in families for x in expected}:
        raise ValueError('need exactly all 82 final checkpoints x 14 benchmarks')


def check_spec(payload, original, batch):
    for key in ('checkpoint','tokenizer','model_recipe','request_manifest','token_cache'):
        if payload[key]!=original[key]:raise ValueError(f'changed scientific binding: {key}')
    evaluation=payload['evaluation']
    runtime_fields={'required_gpu_name_substring','expected_world_size','micro_batch_size',
                    'minimum_sequence_width','parity_requests','max_gpu_hours','batching'}
    if ({k:v for k,v in evaluation.items() if k not in runtime_fields}!=
            {k:v for k,v in original['evaluation'].items() if k not in runtime_fields}):
        raise ValueError('changed scientific evaluation settings')
    if (evaluation['required_gpu_name_substring']!='H100' or evaluation['expected_world_size']!=1
            or evaluation['micro_batch_size']!=batch or evaluation.get('batching') is not None
            or evaluation.get('minimum_sequence_width',1)!=1):
        raise ValueError('H100/fixed-batch execution contract mismatch')


def estimate(campaigns, pilot, batches):
    rates={}
    for job in pilot['jobs']:
        if job['variant']!='candidate' or job['K']!=200:continue
        spec=read(checked(job['spec']))
        if spec['evaluation']['micro_batch_size']!=batch_for(batches,job['ladder'],job['E']):raise ValueError('timing batch differs from accepted batch')
        result=read(Path(spec['output_dir'])/'scores.json');timing=result['timing'];bins=defaultdict(list)
        for b in timing['forward_batches'][2:]:
            bins[1 << (max(1,b['width'])-1).bit_length()].append(b['seconds']/b['requests'])
        if not bins:raise ValueError('missing measured forward timings')
        overhead=max(0,(timing['scoring_seconds']-sum(b['seconds'] for b in timing['forward_batches']))/result['counts']['requests'])
        rates[job['ladder'],job['E']]=({w:median(v) for w,v in bins.items()},overhead)
    estimates={}
    for campaign in campaigns.values():
        data=read(checked(campaign.get('data_manifest') or campaign['data_manifests']['full']))
        histograms=defaultdict(Counter)
        for language in data['languages']:
            for w,n in language['width_histogram'].items():histograms[language['benchmark']][int(w)]+=n
        for job in campaign['jobs']:
            if job.get('phase')=='smoke':continue
            hist=histograms[job['benchmark']];rates_by_width,overhead=rates[job['ladder'],job['E']]
            seconds=120+sum(hist.values())*overhead
            for w,n in hist.items():
                bucket=1 << (max(1,w)-1).bit_length()
                nearest=min(rates_by_width,key=lambda k:abs(k.bit_length()-bucket.bit_length()))
                seconds+=n*rates_by_width[nearest]*max(1,bucket/nearest)
            estimates[job['id']]=dict(estimated_seconds=seconds,walltime_minutes=max(20,math.ceil((2*seconds+600)/300)*5))
    return estimates


def prepare(profile_path, gate_path, output):
    profile_path,gate_path,output=profile_path.resolve(),gate_path.resolve(),output.resolve()
    profile=read(profile_path);batches=batch_map(profile);repo=(profile_path.parent/profile['repo_root']).resolve()
    gate=read(gate_path);_verify_payload_hash(gate,field='manifest_sha256',label='numerical gate')
    check_gate(gate)
    pilot=read(checked(gate['campaign']))
    if [list(x) for x in packages()]!=read(checked(gate['runtime_packages'])):
        raise ValueError('installed packages differ from the validation node')
    device_lines=checked(gate['device']).read_text().strip().splitlines()
    if len(device_lines)!=2:raise ValueError('expected one validated GPU')
    device=[x.strip() for x in device_lines[1].split(',')]
    if (device[0],int(device[2].split()[0]),device[3])!=(profile['gpu_name'],profile['gpu_memory_mib'],profile['driver_version']):
        raise ValueError('profile hardware/driver differs from validated GPU')
    if pilot['profile_payload']['python']!=profile['python'] or pilot['profile_payload']['environment']!=profile['environment']:
        raise ValueError('runtime differs from numerical pilot')
    if output.exists():raise ValueError('use a fresh output directory')
    originals={name:runtime.load_campaign(repo/path) if name=='main' else read(repo/path) for name,path in profile['source_campaigns'].items()}
    estimates=estimate(originals,pilot,batches)
    output.mkdir(parents=True)
    source=runtime.freeze_source(repo,output,profile)
    # New orchestration is allowed; numerical implementation must be pilot-identical.
    old_source_root=Path(pilot['source_manifest']['path']).parent/'source'
    for entry in read(checked(pilot['source_manifest']))['files']:
        if not Path(entry['path']).is_relative_to(old_source_root):checked(entry)
    old={str(Path(e['path']).relative_to(Path(pilot['source_manifest']['path']).parent/'source')):e['sha256']
         for e in read(checked(pilot['source_manifest']))['files'] if '/source/src/' in e['path']}
    reducers={'full_suite_data.py','candidate_data.py','extension_data.py'}
    numerical={p:h for p,h in old.items() if '/pipeline/' in p or '/recipes/' in p
               or ('/stage1_benchmarking/' in p and Path(p).name in reducers)}
    for p,h in numerical.items():checked(dict(path=str(output/'source'/p),sha256=h))
    environment=save(output/'environment.json',dict(packages=packages(),pilot_packages=gate['runtime_packages'],pilot_device=gate['device'],gpu_name=profile['gpu_name'],driver_version=profile['driver_version'],
                                                    gpu_memory_mib=profile['gpu_memory_mib'],python=profile['python']))
    gate_results=[]
    for job in pilot['jobs']:
        s=read(checked(job['spec']));gate_results.append(ref(Path(s['output_dir'])/'scores.json'))
    # Preserve supplementary metric-gate evidence as well as timing-panel scores.
    for entry in gate.get('score_results',[]):
        checked(entry)
        if entry not in gate_results:gate_results.append(entry)
    group_refs={};all_jobs=[]
    for name,original in originals.items():
        jobs=[]
        for historical in original['jobs']:
            if historical.get('phase')=='smoke':continue
            job=copy.deepcopy(historical)
            for key in ('reused','origin_campaign','original_slurm_job_id'):job.pop(key,None)
            old_spec=read(checked(historical['spec']));spec=copy.deepcopy(old_spec);spec.pop('spec_sha256')
            spec.update(run_id=output.name+'_'+job['id'],output_dir=str(output/'scores'/job['id']),profile=ref(profile_path))
            # Historical execution metadata remains reachable through the exact
            # parent spec; do not present its A40 profile as this run's profile.
            spec.pop('candidate_provenance',None);spec.pop('full_suite_provenance',None)
            spec['evaluation'].update(micro_batch_size=batch_for(batches,job['ladder'],job['E']),minimum_sequence_width=1,
                required_gpu_name_substring='H100',expected_world_size=1,parity_requests=32,max_gpu_hours=estimates[job['id']]['walltime_minutes']/60)
            spec['evaluation'].pop('batching',None)
            spec['h100_release_provenance']=dict(historical_spec=historical['spec'],numerical_gate=ref(gate_path))
            check_spec(spec,old_spec,batch_for(batches,job['ladder'],job['E']))
            job.update(spec=save(output/'specs'/(job['id']+'.json'),spec,'spec_sha256'),gpus=1,memory=profile['slurm']['memory'],
                       **estimates[job['id']],group=name)
            job['run_script'],job['sbatch_script']=runtime.write_job(output,job,profile,repo,module=MODULE,job_prefix='s1-h100-paper-')
            jobs.append(job);all_jobs.append(job)
        group=copy.deepcopy(original)
        for key in ('manifest_sha256','reuse_campaign','budget','max_pilot_allocated_gpu_hours','queue_controls','collection_job'):
            group.pop(key,None)
        group.update(jobs=jobs,source_manifest=source,repo=str(repo),status='prepared_not_submitted',
                     profile_payload=copy.deepcopy({**original['profile_payload'],**profile}),profile=ref(profile_path))
        batch_policy='Fixed batches by architecture '+json.dumps(batches,sort_keys=True)+'; no adaptive enlargement or OOM fallback.'
        group['execution_policy']='Fresh single-H100 evaluations, no result reuse; root release owns budget and collection. '+batch_policy
        for protocol in (group.get('protocol'),group['profile_payload'].get('protocol')):
            if protocol is not None:protocol['batching']=batch_policy
        group_refs[name]=save(output/name/'campaign.json',group,'manifest_sha256')
    validate_matrix(all_jobs)
    budget=save(output/'budget.json',dict(method='Measured fixed-batch width-bucket medians; full request-width histograms; measured host/diagnostic overhead; 120s startup per job. Wall limits 2x estimate + 10min. No queue time; provisional extrapolation.',
        estimated_gpu_hours=sum(j['estimated_seconds'] for j in all_jobs)/3600,
        timeout_gpu_hours=sum(j['walltime_minutes'] for j in all_jobs)/60,
        by_tier={a:sum(j['estimated_seconds'] for j in all_jobs if j['ladder']==a)/3600 for a in 'ABCD'}))
    campaign=dict(schema=SCHEMA,repo=str(repo),profile=ref(profile_path),profile_payload=profile,source_manifest=source,
        environment=environment,gate=ref(gate_path),gate_scores=gate_results,groups=group_refs,jobs=all_jobs,budget=budget,
        batch_selection=ref(pilot.get('batch_selection_path',pilot.get('rejected_candidates'))),
        status='prepared_not_submitted',collection_job=runtime.write_collection_job(output,profile,module=MODULE,job_prefix='s1-h100-paper-'))
    save(output/'campaign.json',campaign,'manifest_sha256')
    for command in ('validate','submit','collect'):
        script=output/(command+'.sh')
        script.write_text('\n'.join(['#!/bin/bash','set -euo pipefail',
            'export PYTHONPATH='+shlex.quote(str(output/'source/src')+':'+str(output/'source')),
            'exec '+shlex.join([profile['python'],'-m',MODULE,command,'--campaign',str(output/'campaign.json')])+' "$@"','']))
        subprocess.run(['bash','-n',str(script)],check=True)
    return validate(output/'campaign.json')


def validate(path):
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import load_native_benchmark_score_spec
    campaign=read(path);_verify_payload_hash(campaign,field='manifest_sha256',label='H100 release')
    if campaign['schema']!=SCHEMA:raise ValueError('incorrect release schema')
    validate_matrix(campaign['jobs'])
    for key in ('profile','source_manifest','environment','gate','budget','collection_job','batch_selection'):checked(campaign[key])
    if read(checked(campaign['batch_selection']))['batches']!=batch_map(campaign['profile_payload']):
        raise ValueError('profile batch sizes differ from selected tier policy')
    for entry in read(checked(campaign['batch_selection'])).get('evidence',[]):checked(entry)
    gate=read(checked(campaign['gate']))
    check_gate(gate)
    for entry in campaign['gate_scores']:checked(entry)
    for entry in read(checked(campaign['source_manifest']))['files']:checked(entry)
    if [list(x) for x in packages()]!=read(checked(campaign['environment']))['packages']:raise ValueError('Python package environment drift')
    cache={};seen_caches=set()
    original_jobs={}
    for group_name,source_path in campaign['profile_payload']['source_campaigns'].items():
        original=read(Path(campaign['repo'])/source_path)
        _verify_payload_hash(original,field='manifest_sha256',label='historical campaign')
        original_jobs[group_name]={j['id']:j for j in original['jobs'] if j.get('phase')!='smoke'}
    group_jobs=[]
    for group_ref in campaign['groups'].values():
        group=read(checked(group_ref))
        group_jobs.extend(group['jobs'])
        for family in group['inputs'].values():
            checked(family['request_manifest'])
            cache_path=checked(family['token_cache'])
            if str(cache_path) not in seen_caches:
                for entry in read(cache_path)['files'].values():checked(entry)
                seen_caches.add(str(cache_path))
    if sorted(group_jobs,key=lambda j:j['id'])!=sorted(campaign['jobs'],key=lambda j:j['id']):
        raise ValueError('collector job inventory differs from submission inventory')
    current_family=None
    for job in sorted(campaign['jobs'],key=lambda j:(j['benchmark'],j['id'])):
        if job['benchmark']!=current_family:
            cache.clear()
            current_family=job['benchmark']
        for key in ('spec','run_script','sbatch_script'):checked(job[key])
        payload=read(checked(job['spec']));original=read(checked(payload['h100_release_provenance']['historical_spec']))
        historical=original_jobs[job['group']][job['id']]
        if (payload['h100_release_provenance']['historical_spec']!=historical['spec'] or
                any(job[k]!=historical[k] for k in ('model_id','benchmark','ladder','E','K','seed'))):
            raise ValueError('checkpoint/job identity drift')
        check_spec(payload,original,batch_for(batch_map(campaign['profile_payload']),job['ladder'],job['E']))
        load_native_benchmark_score_spec(Path(job['spec']['path']),request_manifest_cache=cache)
    result=dict(status='submission_ready_not_submitted',jobs=len(campaign['jobs']),gpu='H100',
                batches=batch_map(campaign['profile_payload']),campaign=ref(path))
    save(Path(path).parent/'validation_receipt.json',result)
    return result


def run(campaign,job_id):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU scoring requires a Slurm allocation')
    expected=read(checked(campaign['environment']))
    if [list(x) for x in packages()]!=expected['packages']:raise ValueError('Python package environment drift')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    # nvidia-smi may list other GPUs on a shared node; CUDA name/visible world
    # size are independently enforced by the scorer. Require all listed H100s
    # to have the calibrated model, driver and memory capacity.
    if not lines or any([x.strip() for x in line.split(',')]!=[expected['gpu_name'],expected['driver_version'],str(expected['gpu_memory_mib'])] for line in lines):
        raise ValueError('GPU hardware/driver differs from validated runtime')
    job=next(j for j in campaign['jobs'] if j['id']==job_id)
    directory=Path(read(checked(job['spec']))['output_dir'])
    if (directory/'hardware.json').exists():raise ValueError('refusing to overwrite hardware receipt')
    save(directory/'hardware.json',dict(observed_gpu_driver_memory=lines,environment=campaign['environment'],
                                       slurm_job_id=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME')))
    runtime.run(campaign,job_id)


def submit(path,campaign):
    if os.environ.get('SLURM_JOB_ID'):raise RuntimeError('submit from login host, never inside a holder')
    validate(path)
    root=path.parent;records=[];slurm=Path(campaign['profile_payload']['slurm']['bin'])
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'submission.json').exists():raise ValueError('existing submission journal; refusing duplicates')
        save(root/'submission.json',dict(status='submitting',jobs=records))
        for job in sorted(campaign['jobs'],key=lambda j:-j['estimated_seconds']):
            jid=subprocess.check_output([str(slurm/'sbatch'),'--parsable',job['sbatch_script']['path']],text=True).strip().split(';')[0]
            if not jid.isdecimal():raise ValueError('invalid Slurm response')
            records.append(dict(id=job['id'],slurm_job_id=jid));save(root/'submission.json',dict(status='submitting',jobs=records))
        queued=set(subprocess.check_output([str(slurm/'squeue'),'--me','--noheader','--format=%A'],text=True).split())
        pending=[r['slurm_job_id'] for r in records if r['slurm_job_id'] in queued]
        command=[str(slurm/'sbatch'),'--parsable']
        if pending:command.append('--dependency=afterany:'+':'.join(pending))
        command.append(campaign['collection_job']['path'])
        collector=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not collector.isdecimal():raise ValueError('invalid collector response')
        save(root/'submission.json',dict(status='submitted',jobs=records,collector=collector))
    return dict(submitted_jobs=len(records),collector=collector)


def collect(campaign):
    from .candidate_campaign import collect as supplementary_collect
    results={}
    for name,entry in campaign['groups'].items():
        path=checked(entry);group=read(path)
        results[name]=(runtime.collect if name=='main' else supplementary_collect)(path,group)
    return dict(status='complete' if all(r['status']=='complete' for r in results.values()) else 'partial',groups=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--profile',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name in ('validate','run','submit','collect'):
        p=sub.add_parser(name);p.add_argument('--campaign',type=Path,required=True)
        if name=='run':p.add_argument('--job',required=True)
        if name=='collect':p.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.profile,args.gate,args.output)
    elif args.command=='validate':result=validate(args.campaign)
    else:
        campaign=read(args.campaign);_verify_payload_hash(campaign,field='manifest_sha256',label='H100 release')
        if args.command=='run':run(campaign,args.job);result=dict(complete=args.job)
        elif args.command=='submit':result=submit(args.campaign,campaign)
        else:
            result=collect(campaign);save(args.campaign.parent/'collection_receipt.json',result)
            if args.require_complete and result['status']!='complete':raise RuntimeError('incomplete benchmark collection')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
