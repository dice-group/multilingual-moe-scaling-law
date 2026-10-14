"""One entry point for preparation, gated submission, execution and analysis."""
import argparse
import json
from pathlib import Path

from .campaign import load_campaign, prepare, seal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('prepare','validate','submit','status','seal','controls','run','complete','pilots','collect','analyze','report'))
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--task')
    parser.add_argument('--attempt',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output',type=Path,help='Separate output directory for report')
    parser.add_argument('--verify',action='store_true',help='Verify existing report without rewriting it')
    parser.add_argument('--replay',action='store_true',help='With report --verify, replay all numerical tables')
    args = parser.parse_args()
    if args.action != 'report' and (args.output or args.verify or args.replay):
        parser.error('--output/--verify/--replay are report-only options')
    if args.action == 'report' and (not args.output or args.dry_run or (args.replay and not args.verify)):
        parser.error('report requires --output, does not use --dry-run, and --replay requires --verify')
    if args.action=='prepare':
        if not args.profile:
            parser.error('prepare requires --profile')
        campaign = prepare(args.profile)
        result = {'campaign':campaign['root'],'checkpoints':campaign['checkpoint_count'],
            'language_streams':campaign['language_stream_count'],'historical_gpu_hours':campaign['historical_gpu_hours']}
    else:
        if not args.campaign:
            parser.error('--campaign is required')
        campaign = load_campaign(args.campaign)
        if args.action == 'report':
            from .reporting import report
            result = report(campaign,args.output,verify_only=args.verify,replay=args.replay)
        elif args.action in ('submit','status','validate'):
            from .scheduler import submit, status
            result = status(campaign) if args.action=='status' else submit(campaign,dry_run=args.dry_run or args.action=='validate')
            if args.action=='submit' and not args.dry_run:
                result = {'registered_jobs':len(result['jobs']), 'released':result['released'],
                    'seal_job':result['jobs']['seal']['job_id'], 'collector_job':result['jobs']['collect']['job_id'],
                    'journal':str(Path(campaign['root'])/'submission.json')}
        elif args.action=='seal':
            result = seal(campaign)
        elif args.action=='controls':
            from .recovery import controls_gate
            result = controls_gate(campaign)
        elif args.action in ('run','complete','pilots'):
            from .execution import run_task, complete_task, pilot_gate
            if args.action!='pilots' and not args.task:
                parser.error('--task is required')
            result = pilot_gate(campaign) if args.action=='pilots' else (run_task(campaign,args.task,args.attempt) if args.action=='run' else complete_task(campaign,args.task))
        else:
            from .analysis import collect, analyze
            if args.action=='collect':
                from .scheduler import status
                # Always preserve collection, even if accounting is temporarily unavailable.
                try:
                    status(campaign)
                except Exception as exc:
                    print(f'scheduler accounting unavailable: {exc}',flush=True)
            result = collect(campaign) if args.action=='collect' else analyze(campaign)
    if result is not None:
        print(json.dumps(result,indent=2) if not isinstance(result,str) else result,flush=True)


if __name__=='__main__':
    main()
