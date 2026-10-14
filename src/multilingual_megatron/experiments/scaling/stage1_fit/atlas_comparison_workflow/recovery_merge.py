"""Consolidate sealed training-only recovery attempts; never run an optimizer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import time

import numpy as np

from .. import atlas_full_pilot as io
from ..execution import write_text_once
from . import campaign as c, model as m, numerics as n, recovery_numerics as r, reporting, snapshot

MODULE="multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.recovery_merge"


def consolidate(spec, packet, previous, additions):
    """Recheck every feasible training point, not just the pilot's winners."""
    lo,hi=m.bounds(spec); attempts=[]; minimum=float('inf')
    rows=[previous.get('selected'),previous.get('audit'),*previous.get('audit_history',[]),
          *previous.get('attempts',[]),*previous.get('probes',[]),*additions]
    for row in rows:
        if not row or 'coordinates' not in row: continue
        x=np.asarray(row['coordinates'],float)
        if x.shape!=lo.shape or not np.isfinite(x).all() or np.any(x<lo) or np.any(x>hi): continue
        if m.domain(spec,x,packet['support'])[0].min()<0: continue
        value,g,_,_=n.objective(spec,x,packet['train'])
        if not np.isfinite(value) or not np.isfinite(g).all(): continue
        minimum=min(minimum,value); kkt=n.stationarity(spec,x,g,packet['support'])
        attempts.append(row|dict(objective=value,kkt=kkt if np.isfinite(kkt) else None,
            success=bool(row.get('success') and row.get('optimizer_success') and kkt<=1e-8)))
    best,audit=r.pair(spec,attempts,packet['train'],minimum)
    result={k:v for k,v in previous.items() if k not in ('sha256','plan_sha256')}
    geometry={}
    if best:
        x=np.asarray(best['coordinates']); jac=n.objective(spec,x,packet['train'])[3]
        singular=np.linalg.svd(jac*np.maximum(abs(x),1.),compute_uv=False)
        geometry=dict(rank=int(np.sum(singular>singular[0]*n.SETTINGS['rank_relative'])),coordinate_count=len(x),
            singular_values=singular.tolist(),point='consolidated training-selected fit')
    result.update(selected=best,audit=audit or {'success':False},attempts=attempts,probes=[],audit_history=[],
        geometry=geometry,certified=audit is not None,heldout_scores_computed=False,
        recovery=dict(minimum_feasible_objective=minimum if np.isfinite(minimum) else None,
            independent_families=[r.family(best),r.family(audit)] if audit is not None else [],
            method='training-only consolidation of v9 and exact-Hessian pilot; all feasible points retained in minimum-loss guard',
            additional_fitting=False))
    return result


def prepare(profile_path):
    profile=json.loads(Path(profile_path).read_text()); root=Path(profile['output_root']).resolve()
    if root.exists() or root.parent!=Path.cwd()/'result_analysis': raise ValueError('new output root required')
    parent=io.read(profile['parent_plan']); oldroot=Path(parent['output_root'])
    audit=io.read(oldroot/'audit.json'); done=io.read(oldroot/'execution/finalize_completion.json')
    lab=io.read(profile['lab_plan']); labroot=Path(lab['profile']['output_root']); completed=io.read(labroot/'completion.json')
    if (done['plan_sha256']!=parent['sha256'] or done['audit_sha256']!=audit['sha256'] or not audit['complete']
            or completed['plan_sha256']!=lab['sha256'] or lab['parent_sha256']!=parent['sha256']
            or set(lab['profile']['tasks'])!=set(audit['unresolved_tasks']) or completed['cases']!=len(audit['unresolved_tasks'])):
        raise ValueError('requires the complete v9 audit and all unresolved exact-curvature cases')
    inputs=[io.bind(profile_path),io.bind(profile['parent_plan']),io.bind(oldroot/'audit.json'),
        io.bind(oldroot/'execution/finalize_completion.json'),io.bind(profile['lab_plan']),io.bind(labroot/'completion.json'),
        *parent['inputs'],parent['archive'],lab['archive'],*lab['sources'],lab['profile_binding']]
    prior=parent['prior_allocated_core_hours']
    for path in profile['additional_accounting']:
        account=json.loads(Path(path).read_text())
        if account['state']!='COMPLETED' or account['gpu_allocations']!=0: raise ValueError('terminal CPU accounting required')
        prior+=account['allocated_cpu_seconds']/3600; inputs.append(io.bind(path))
    if prior+3>250: raise ValueError('consolidation exceeds cumulative CPU cap')
    for b in [*inputs,*parent['sources'],*parent['packets'],parent['observations'],parent['historical_predictions']]: io.verify(b)
    sources={Path(b['original_path']):Path(b['path']) for b in parent['sources']}
    for path in (Path(__file__),Path(r.__file__),Path.cwd()/'tests/test_stage1_atlas_comparison.py'): sources[path]=path
    runtime,bindings,archive=snapshot.freeze(root,sources,Path.cwd(),content_sources=sources)
    rows=[]; recovered=0
    for item in audit['coverage']:
        io.verify(item['receipt']); inputs.append(item['receipt']); value=io.read(item['receipt']['path']); task=item['task']
        if value['plan_sha256']!=parent['sha256'] or value['task']!=task: raise ValueError('parent receipt mismatch')
        if not item['certified']:
            bound=io.bind(labroot/f'results/{task:05d}.json'); inputs.append(bound); extra=io.read(bound['path'])
            if extra['plan_sha256']!=lab['sha256'] or extra['task']!=task: raise ValueError('pilot receipt mismatch')
            packet=io.read(parent['packets'][task//len(c.PANEL)]['path'])
            if value['packet_sha256']!=packet['sha256'] or value['spec']!=c.PANEL[task%len(c.PANEL)].payload(): raise ValueError('equation or packet changed')
            value=consolidate(c.PANEL[task%len(c.PANEL)],packet,value,[row['result'] for row in extra['results']])
            value['exact_curvature_receipt']=bound; recovered+=bool(value['certified'])
        rows.append({k:v for k,v in value.items() if k not in ('sha256','plan_sha256')})
    protocol=parent['protocol']|dict(version='atlas_comparison/v10_consolidated',
        amendment='consolidate all 139 exact-curvature pilot cases with v9, rechecking every feasible training objective; no new fitting, equations, data, bounds or acceptance thresholds',
        allocations={'finalize':dict(cores=1,minutes=120)},pilot_allocated_core_hours=0.,main_allocated_core_hours=0.)
    plan=io.write(root/'plan.json',dict(protocol=protocol,profile=profile,output_root=str(root),runtime_root=str(runtime),
        sources=bindings,archive=archive,inputs=inputs,packets=parent['packets'],observations=parent['observations'],
        historical_predictions=parent['historical_predictions'],historical_metrics=parent['historical_metrics'],
        environment=c.environment(),task_count=parent['task_count'],prior_allocated_core_hours=prior,
        maximum_cumulative_allocated_core_hours=prior+3,new_heldout_scores_computed=False))
    for value in rows: io.write(c.task_path(plan,value['task']),value|dict(plan_sha256=plan['sha256']))
    io.write(root/'execution/consolidation.json',dict(plan_sha256=plan['sha256'],newly_certified=recovered,
        unresolved=sum(not v['certified'] for v in rows),heldout_scores_computed=False,additional_fitting=False))
    runner=root/'run.sh'; write_text_once(runner,'\n'.join(('#!/bin/bash','set -euo pipefail',f'cd {shlex.quote(str(Path.cwd()))}',
        f"export PYTHONPATH={shlex.quote(str(runtime/'src'))}:.",
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        f"exec {shlex.quote(profile['python_executable'])} -m {MODULE} finalize --plan {shlex.quote(str(root/'plan.json'))}",'')))
    job=root/'finalize.sbatch'; write_text_once(job,'\n'.join(('#!/bin/bash','#SBATCH --job-name=atlas-consolidate',
        f"#SBATCH --account={profile['account']}",f"#SBATCH --partition={profile['partition']}",
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=1','#SBATCH --time=120',
        f"#SBATCH --mem={profile['memory']}",f'#SBATCH --output={root}/finalize-%j.out','set -euo pipefail',
        f'exec bash {shlex.quote(str(runner))}','')))
    io.write(root/'launch.json',dict(plan_sha256=plan['sha256'],scripts=[io.bind(runner),io.bind(job)]))
    return dict(plan=str(root/'plan.json'),sha256=plan['sha256'],newly_certified=recovered,unresolved=sum(not v['certified'] for v in rows))


def load(path,*,verify_inputs=False):
    plan=io.read(path)
    if plan['protocol']['version']!='atlas_comparison/v10_consolidated' or plan['environment']!=c.environment(): raise ValueError('consolidation drift')
    if not Path(__file__).resolve().is_relative_to(Path(plan['runtime_root'])): raise ValueError('use frozen run.sh')
    for b in [*plan['sources'],plan['archive']]: io.verify(b)
    if verify_inputs:
        for b in [*plan['inputs'],*plan['packets'],plan['observations'],plan['historical_predictions']]: io.verify(b)
        launch=io.read(Path(plan['output_root'])/'launch.json')
        if launch['plan_sha256']!=plan['sha256']: raise ValueError('launch mismatch')
        for b in launch['scripts']: io.verify(b)
    return plan


def finalize(path):
    plan=load(path); root=Path(plan['output_root'])
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='1': raise ValueError('bound one-CPU allocation required')
    if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')): raise ValueError('single CPU thread required')
    io.write(root/'execution/finalize.json',dict(plan_sha256=plan['sha256'],job_id=os.environ['SLURM_JOB_ID'],cores=1,seconds_limit=7200))
    began=time.monotonic(); audit=reporting.audit(path,loader=load); report=reporting.evaluate(path,loader=load)
    return io.write(root/'execution/finalize_completion.json',dict(plan_sha256=plan['sha256'],audit_sha256=audit['sha256'],
        report_sha256=report['sha256'],status=report['status'],wall_seconds=time.monotonic()-began,heldout_scores_computed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','finalize'))
    parser.add_argument('--profile'); parser.add_argument('--plan'); args=parser.parse_args()
    print(json.dumps(prepare(args.profile) if args.action=='prepare' else finalize(args.plan)))
