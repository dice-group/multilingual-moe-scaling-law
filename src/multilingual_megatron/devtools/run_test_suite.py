"""Run one or more curated repository unittest suites."""

from __future__ import annotations

import argparse
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path

from multilingual_megatron.devtools.test_suites import (
    RUNNABLE_SUITES,
    load_suite_modules,
    suite_manifest_path,
)
from multilingual_megatron.repo import find_repo_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="+", choices=RUNNABLE_SUITES)
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="print selected modules without running them",
    )
    parser.add_argument("--verbosity", type=int, choices=(1, 2), default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = find_repo_root(Path(__file__))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    modules: list[str] = []
    for suite in args.suite:
        for module in load_suite_modules(suite_manifest_path(repo_root, suite)):
            if module not in modules:
                modules.append(module)

    if args.list_modules:
        print("\n".join(modules))
        return 0

    loaded = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(loaded)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
