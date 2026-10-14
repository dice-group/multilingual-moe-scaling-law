"""Register an idempotent Slurm DAG from the login host, then release its root."""
from __future__ import annotations

import fcntl
import getpass
import os
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from multilingual_megatron.pipeline.executor import SlurmExecutor
from .campaign import identity, read, sha, verify, write

MODULE = 'multilingual_megatron.experiments.scaling.stage1_routing_repair'


def build_graph(campaign):
    nodes = [{'id':'seal','action':'seal','task':None,'attempt':None,'dependencies':[]}]
    prerequisite = 'seal'
    if campaign['runtime'].get('gate_policy') == 'fixed_execution_v3':
        prerequisite = 'controls'
        jobs = campaign['runtime']['recovery']['wait_for_jobs']
        if not jobs or any(not str(job).isdigit() for job in jobs):
            raise ValueError('recovery requires explicit external control job IDs')
        nodes.append({'id':'controls','action':'controls','task':None,'attempt':None,
            'dependencies':[('afterok','seal')],
            'external_dependencies':[('afterany',str(job)) for job in jobs]})
    def task_nodes(task, prerequisite):
        attempts = []
        for attempt in range(1,campaign['runtime'].get('retries',{}).get('maximum_attempts',3)+1):
            key = f"{task['id']}.a{attempt}"
            deps = [('afterok',prerequisite)]
            if attempts:
                # AND: if any preceding attempt succeeds, every later retry is
                # invalidated and cancelled by Slurm's kill-on-invalid-dep.
                deps += [('afternotok',p) for p in attempts]
            nodes.append({'id':key,'action':'run','task':task['id'],'attempt':attempt,'dependencies':deps})
            attempts.append(key)
        gate = f"{task['id']}.complete"
        nodes.append({'id':gate,'action':'complete','task':task['id'],'attempt':None,'dependencies':[('afterany',p) for p in attempts]})
        return gate
    by_id = {t['id']:t for t in campaign['tasks']}
    gate_b = task_nodes(by_id['pilot_b'],prerequisite)
    gate_c = task_nodes(by_id['pilot_c'],prerequisite)
    gate_d = task_nodes(by_id['pilot_d'],prerequisite)
    nodes.append({'id':'pilots','action':'pilots','task':None,'attempt':None,
        'dependencies':[('afterany',p) for p in (gate_b,gate_c,gate_d)]})
    gates = [task_nodes(task,'pilots') for task in campaign['tasks'] if task['kind']=='production']
    nodes.append({'id':'collect','action':'collect','task':None,'attempt':None,
        'dependencies':[('afterany',p) for p in gates+['pilots']]})
    return nodes


def dependency_text(node, jobs):
    groups = {}
    for kind, key in node['dependencies']:
        groups.setdefault(kind,[]).append(str(jobs[key]['job_id']))
    for kind, job_id in node.get('external_dependencies',[]):
        groups.setdefault(kind,[]).append(str(job_id))
    return ','.join(kind+':'+':'.join(ids) for kind,ids in groups.items()) or None


def executor_for(campaign,node):
    runtime = campaign['runtime']
    gpu = node['action']=='run'
    settings = dict(runtime['resources']['h100' if gpu else 'cpu'])
    task = next((t for t in campaign['tasks'] if t['id']==node['task']),None)
    if gpu:
        settings.update(runtime.get('task_resources',{}).get(f"{task['model']['ladder']}/{task['gpus']}",{}))
    flags = list(settings.pop('extra_sbatch_flags',[]))
    settings['time_limit'] = settings.pop('time')
    if gpu:
        if 'task_resources' not in runtime:  # Preserve legacy V2 requests.
            settings['gres'] = f"gpu:h100:{task['gpus']}"
            settings['cpus_per_task'] = 64 if task['gpus']==4 else 32
            settings['mem'] = '256G' if task['model']['ladder']=='D' and task['gpus']>1 else '128G'
        if int(settings['gres'].rsplit(':',1)[-1]) != task['gpus']:
            raise ValueError(f"GPU allocation differs from frozen execution: {task['id']}")
        minutes = task['time_minutes']
        settings['time_limit'] = f'{minutes//60:02d}:{minutes%60:02d}:00'
    else:
        settings['time_limit'] = runtime.get('cpu_time_limits',{}).get(node['action'],settings['time_limit'])
    flags += [f"--account={runtime['account']}",f"--qos={runtime['qos']}",
        '--no-requeue','--kill-on-invalid-dep=yes']
    if node['id']=='seal':
        flags.append('--hold')
    return SlurmExecutor(**settings,extra_sbatch_flags=flags)


def slurm_command(campaign, name):
    return campaign['runtime'].get('slurm_commands',{}).get(name,name)


@contextmanager
def submission_path(campaign):
    """The shared executor finds sbatch through the profile-selected directory."""
    executable = slurm_command(campaign,'sbatch')
    previous = os.environ.get('PATH')
    try:
        if Path(executable).is_absolute():
            os.environ['PATH'] = str(Path(executable).parent)+os.pathsep+(previous or os.defpath)
            if shutil.which('sbatch') != executable:
                raise ValueError('profile sbatch cannot be resolved')
        yield
    finally:
        if previous is None:
            os.environ.pop('PATH',None)
        else:
            os.environ['PATH'] = previous


def recover_intent(comment, journal, campaign):
    """Recover a submit/receipt interruption without submitting a duplicate job."""
    user = getpass.getuser()
    queue = subprocess.check_output([slurm_command(campaign,'squeue'),'-h','-u',user,'-o','%i|%k'],text=True,timeout=30)
    accounting = subprocess.check_output([slurm_command(campaign,'sacct'),'-n','-X','-P','-u',user,'-S',journal['created_date'],
        '--format=JobIDRaw,Comment%200'],text=True,timeout=30)
    matches = set()
    for line in (queue+'\n'+accounting).splitlines():
        fields = line.strip().split('|')
        if len(fields)>1 and fields[1].strip()==comment:
            matches.add(fields[0].strip())
    if len(matches)>1:
        raise ValueError(f'duplicate scheduler identity: {comment}: {matches}')
    if matches:
        return matches.pop()
    # The absence of a receipt immediately after an ambiguous sbatch is not
    # sufficient evidence that no job exists. Require explicit recovery later.
    raise ValueError(f'unresolved submission intent {comment}; no duplicate submitted; inspect scheduler/log receipts')


def job_request(campaign, node):
    root = Path(campaign['root'])
    source = root/'source'
    env = dict(campaign['runtime']['environment'])
    env.update(PROOF_PIPELINE_REPO_ROOT=str(source),PYTHONPATH=f'{source}/src:{source}',
        LD_LIBRARY_PATH=str(Path(campaign['runtime']['python']).parent.parent/'lib'))
    command = [campaign['runtime']['python'],'-m',MODULE,node['action'],'--campaign',str(root)]
    if node['task']:
        command += ['--task',node['task']]
    if node['attempt']:
        command += ['--attempt',str(node['attempt'])]
    return executor_for(campaign,node), command, env, root/'logs'/'attempts'/node['id']


def validate_submission(campaign, graph):
    """Render every request and check shell syntax without a scheduler mutation."""
    root = Path(campaign['root'])
    for entry in campaign['source']['files']:
        verify(entry)
    for task in campaign['tasks']:
        for variants in task['specs'].values():
            for entry in variants.values():
                verify(entry)
    for name, executable in campaign['runtime'].get('slurm_commands',{}).items():
        if not Path(executable).is_absolute() or not os.access(executable,os.X_OK):
            raise ValueError(f'unavailable profile scheduler command: {name}: {executable}')
    scripts = []
    for index,node in enumerate(graph):
        executor,command,env,logs = job_request(campaign,node)
        path = root/'preflight_scripts'/f'{index:03d}_{node["id"]}.sh'
        path.parent.mkdir(exist_ok=True)
        path.write_text(executor._build_job_script(command,env,f'rr_preflight_{index}',str(logs/'slurm.log')))
        path.chmod(0o600)
        subprocess.run(['bash','-n',str(path)],check=True,capture_output=True,text=True)
        scripts.append(identity(path))
    result = {'status':'dry_run','nodes':len(graph),'scripts':scripts}
    write(root/'submission_preflight.json',result)
    return {'status':'dry_run','nodes':len(graph),'shell_scripts_checked':len(scripts)}


def definite_rejection(exc):
    message = str(exc.stderr or '').lower()
    # Communication errors remain ambiguous even when sbatch exits nonzero.
    return any(text in message for text in ('invalid account', 'invalid qos',
        'invalid partition', 'invalid dependency', 'invalid generic resource',
        'requested node configuration is not available'))


def submit(campaign, *, dry_run=False):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('submit from the login host; nested sbatch is forbidden')
    root = Path(campaign['root'])
    graph = build_graph(campaign)
    write(root/'submission_plan.json',{'nodes':graph,'node_count':len(graph),
        'policy':'all eligible; production after all three pilots; at most three attempts; no automatic scientific retries'})
    if dry_run:
        return validate_submission(campaign,graph)
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path = root/'submission.json'
        journal = read(path) if path.exists() else {'created_date':datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'campaign_sha256':sha(root/'campaign.json'),'jobs':{},'released':False}
        if journal['campaign_sha256'] != sha(root/'campaign.json'):
            raise ValueError('submission belongs to a different campaign')
        prefix = journal['campaign_sha256'][:10]
        for index,node in enumerate(graph):
            key = node['id']
            previous = journal['jobs'].get(key)
            if previous and previous.get('job_id'):
                continue
            version = 'rr3' if campaign['runtime'].get('gate_policy') == 'fixed_execution_v3' else 'rr2'
            comment = f'{version}.{prefix}.{index}'
            if previous and previous['state'] != 'rejected':
                previous['job_id'] = recover_intent(comment,journal,campaign)
                write(path,journal)
                continue
            journal['jobs'][key] = {'comment':comment,'state':'submitting',
                'rejections':previous.get('rejections',[]) if previous else []}
            write(path,journal)
            executor,command,env,log_dir = job_request(campaign,node)
            executor.extra_sbatch_flags.append(f'--comment={comment}')
            try:
                with submission_path(campaign):
                    job_id = executor.submit(command,env,False,log_dir/'submission.log',
                        dependency=dependency_text(node,journal['jobs']),job_name=f'{version}_{prefix}_{index}',
                        script_dir=log_dir,slurm_output=log_dir/'slurm.log')
            except subprocess.CalledProcessError as exc:
                if definite_rejection(exc):
                    journal['jobs'][key]['state'] = 'rejected'
                    journal['jobs'][key]['rejections'].append({'error':str(exc.stderr),
                        'time_utc':datetime.now(timezone.utc).isoformat()})
                    write(path,journal)
                raise
            journal['jobs'][key].update(job_id=job_id,state='submitted')
            write(path,journal)
        if not journal['released']:
            root_id = journal['jobs']['seal']['job_id']
            state = subprocess.check_output([slurm_command(campaign,'scontrol'),'show','job',str(root_id)],text=True,timeout=30)
            if 'JobHeldUser' in state or 'JobHeldAdmin' in state:
                subprocess.run([slurm_command(campaign,'scontrol'),'release',str(root_id)],check=True,timeout=30)
            journal['released'] = True
            write(path,journal)
        return journal


def status(campaign):
    root = Path(campaign['root'])
    journal = read(root/'submission.json')
    ids = ','.join(str(j['job_id']) for j in journal['jobs'].values() if j.get('job_id'))
    metadata = {'queried_at_utc':datetime.now(timezone.utc).isoformat(),
        'command':slurm_command(campaign,'sacct'),'requested_job_ids':ids.split(',')}
    try:
        result = subprocess.check_output([slurm_command(campaign,'sacct'),'-n','-X','-P','-j',ids,
            '--format=JobIDRaw,JobName%60,State,ElapsedRaw,AllocTRES%100,ExitCode'],text=True,timeout=30)
    except Exception as exc:
        write(root/'scheduler_status.json',{**metadata,'status':'query_failed','error':repr(exc)})
        raise
    (root/'scheduler_status.psv').write_text(result)
    observed = {line.split('|')[0].strip() for line in result.splitlines()}
    write(root/'scheduler_status.json',{**metadata,'status':'queried',
        'snapshot':identity(root/'scheduler_status.psv'), 'missing_job_ids':sorted(set(ids.split(','))-observed)})
    return result
