"""Public fixed-law refit CLI; preparation never submits jobs."""
import argparse
from .paper_refit_workflow import protocol
import json
from pathlib import Path
from .paper_refit_workflow.inputs import (
    cells,
    sealed,
    read_sealed,
    local_binding,
    local_path,
    source_input,
    validate_split,
    native_groups,
    load,
    read_training,
    verify_bundle,
    prepare,
    SCHEMA,
    FOLDS,
    NATIVE,
    SHARED,
    MODELS,
    binding,
    verify,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="copy and validate the main-paper fitting inputs")
    prep.add_argument("--profile", type=Path, required=True)
    prep.add_argument("--output", type=Path, help="new output directory; otherwise use the profile")
    check = commands.add_parser("verify", help="validate a bundle without its source campaigns")
    check.add_argument("--manifest", type=Path, required=True)
    plan = commands.add_parser("plan", help="prepare numerical contracts, source snapshot, and CPU arrays")
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--output", type=Path)
    plan.add_argument("--models", nargs="+", choices=tuple(dict.fromkeys((*MODELS, *protocol.MODELS, *protocol.ADDITIONAL_ABLATIONS))))
    seal = commands.add_parser("seal", help="seal a complete full-paper refit after bound downstream review")
    seal.add_argument("--plan", type=Path, required=True)
    seal.add_argument("--review", type=Path, required=True)
    seal.add_argument("--report", type=Path, help="create-only amended report inside the run directory")
    for name in ("submit", "status", "worker", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", type=Path, required=True)
        if name == "submit":
            mode = sub.add_mutually_exclusive_group()
            mode.add_argument("--dry-run", action="store_true")
            mode.add_argument("--test-only", action="store_true")
        elif name == "status":
            sub.add_argument("--scheduler", action="store_true")
        elif name == "worker":
            sub.add_argument("--pool", required=True)
            sub.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.profile, output=args.output)
    elif args.command == "verify":
        result = verify_bundle(args.manifest)
    elif args.command == "report":
        from .paper_refit_workflow.report import report
        result = report(args.plan)
    elif args.command == "seal":
        from .paper_refit_workflow.seal import seal
        result = seal(args.plan, args.review, report_path=args.report)
    else:
        from .paper_refit_workflow import runner
        if args.command == "plan":
            result = runner.prepare(args.profile, models=args.models, output=args.output)
        elif args.command == "submit":
            result = runner.submit(args.plan, dry_run=args.dry_run, test_only=args.test_only)
        elif args.command == "status":
            result = runner.status(args.plan, scheduler=args.scheduler)
        else:
            result = runner.worker(args.plan, args.pool, args.index)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
