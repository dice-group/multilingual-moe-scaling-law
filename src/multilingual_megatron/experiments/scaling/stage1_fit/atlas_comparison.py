"""Prepare, run, audit and report the bounded ATLAS comparison."""
from __future__ import annotations

import argparse
import json

from .atlas_comparison_workflow import campaign, reporting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "pilot-gate", "audit", "evaluate"))
    parser.add_argument("--profile")
    parser.add_argument("--plan")
    parser.add_argument("--phase", choices=("pilot", "main", "finalize"))
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.profile: parser.error("prepare requires --profile")
        result = campaign.prepare(args.profile)
    else:
        if not args.plan: parser.error("action requires --plan")
        if args.action == "run":
            if not args.phase: parser.error("run requires --phase")
            result = campaign.run(args.plan, args.phase)
        elif args.action == "pilot-gate": result = campaign.pilot_gate(args.plan)
        elif args.action == "audit": result = reporting.audit(args.plan)
        else: result = reporting.evaluate(args.plan)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
