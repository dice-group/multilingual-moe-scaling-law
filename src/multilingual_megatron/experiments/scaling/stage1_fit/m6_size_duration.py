"""CLI for the isolated, frozen M6-ND CPU campaign; never submits jobs itself."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    from .m6_size_duration_workflow import execution, planning, reporting
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("--profile", type=Path, required=True)
    sub.add_parser("verify")
    sub.add_parser("worker").add_argument("--task", type=int, required=True)
    sub.add_parser("pilot-gate")
    sub.add_parser("reduce")
    sub.add_parser("audit").add_argument("--task", type=int, required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.action == "prepare":
        result = planning.prepare(args.profile)
    else:
        if args.plan is None:
            parser.error("--plan is required")
        if args.action == "verify":
            plan = planning.load(args.plan)
            result = {"status": "verified", "plan_sha256": plan["sha256"], "tasks": len(plan["tasks"]),
                "preflight": plan["preflight"]}
        elif args.action in {"worker", "audit"}:
            result = getattr(execution, args.action)(args.plan, args.task)
        elif args.action == "report":
            result = reporting.report(args.plan)
        else:
            result = getattr(execution, args.action.replace("-", "_"))(args.plan)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status") in {"blocked", "failed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
