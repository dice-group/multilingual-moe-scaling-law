"""Canonical CLI for the submission-ready coverage-law comparison."""
import argparse
import json
from pathlib import Path

from . import campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--profile", type=Path, required=True)
    for name in ("verify", "submit", "worker", "select", "report", "status", "diagnose"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        if name == "submit":
            command.add_argument("--test-only", action="store_true")
        if name == "worker":
            command.add_argument("--task", required=True)
            command.add_argument("--stage", choices=("starts", "certify"), required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        campaign.prepare(args.profile)
    elif args.action == "verify":
        print(json.dumps(dict(verified=True, plan_sha256=campaign.load(args.plan, all_inputs=True)["sha256"])))
    elif args.action == "submit":
        campaign.submit(args.plan, args.test_only)
    elif args.action == "status":
        print(json.dumps(campaign.status(args.plan), indent=2))
    elif args.action == "worker":
        return 0 if campaign.worker(args.plan, args.task, args.stage) else 1
    elif args.action == "diagnose":
        from .diagnostics import diagnose
        return 0 if diagnose(args.plan) else 1
    else:
        from . import reporting
        return 0 if getattr(reporting, args.action)(args.plan) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
