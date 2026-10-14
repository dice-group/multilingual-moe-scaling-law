"""One-submit Stage 1 C0/M0/J0/M1/M2 primary-holdout refit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .baseline_refit_workflow.gates import execute_job
from .baseline_refit_workflow.scheduler import inspect, status, submit


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("inspect", "submit", "status"):
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
    execute = commands.add_parser("execute", help=argparse.SUPPRESS)
    execute.add_argument("--workflow-manifest", type=Path, required=True)
    execute.add_argument("--job-id", required=True)
    execute.add_argument("--task-index", type=int)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect(args.profile)
    elif args.command == "submit":
        result = submit(args.profile)
    elif args.command == "status":
        result = status(args.profile)
    else:
        result = execute_job(
            args.workflow_manifest, args.job_id, task_index=args.task_index,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
