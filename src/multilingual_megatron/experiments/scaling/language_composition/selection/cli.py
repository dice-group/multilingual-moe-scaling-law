"""Command-line facade for the one maintained language-composition workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from multilingual_megatron.repo import find_repo_root

from multilingual_megatron.experiments.scaling.language_composition.selection.common import DEFAULT_SPEC, SelectionError, load_inputs
from multilingual_megatron.experiments.scaling.language_composition.selection.optimization import SolverSettings
from multilingual_megatron.experiments.scaling.language_composition.selection.workflow import select_and_write, verify_selection


DEFAULT_OUTPUT = Path("result_analysis/language_composition_selection_v2")
DEFAULT_GROUPINGS = Path("configs/language_groupings")


def _resolve(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _artifact_overrides(values: Sequence[str], root: Path) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise SelectionError(
                f"Artifact override must use NAME=PATH syntax: {value!r}"
            )
        if name in overrides:
            raise SelectionError(f"Duplicate artifact override: {name}")
        overrides[name] = _resolve(Path(raw_path), root)
    return overrides


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override one hash-pinned external artifact path.",
    )
    parser.add_argument(
        "--sampling-plan",
        type=Path,
        help="Relocate the hash-pinned sampling plan without changing its identity.",
    )
    parser.add_argument("--arm-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solver-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive or verify the strict K196, four-axis, resource-matched K12 arms."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Solve, gate, and seal both arms.")
    _add_shared_arguments(select)
    select.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    select.add_argument("--grouping-dir", type=Path, default=DEFAULT_GROUPINGS)
    verify = commands.add_parser(
        "verify", help="Rebuild distances and independently re-solve a sealed result."
    )
    _add_shared_arguments(verify)
    verify.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT / "selection_manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_repo_root(Path(__file__)).resolve()
    settings = SolverSettings(
        arm_workers=args.arm_workers,
        solver_threads=args.solver_threads,
        show_progress=args.solver_progress,
    )
    try:
        inputs = load_inputs(
            _resolve(args.spec, root),
            repo_root=root,
            artifact_dir=_resolve(args.artifact_dir, root),
            sampling_plan=(
                _resolve(args.sampling_plan, root) if args.sampling_plan else None
            ),
            artifact_overrides=_artifact_overrides(args.artifact, root),
        )
        if args.command == "select":
            receipt = select_and_write(
                inputs,
                output_dir=_resolve(args.output_dir, root),
                grouping_dir=_resolve(args.grouping_dir, root),
                solver_settings=settings,
            )
            summary = {
                "status": receipt["status"],
                "selection_identity_sha256": receipt[
                    "selection_identity_sha256"
                ],
                "compact": receipt["selection"]["compact"]["languages"],
                "diverse": receipt["selection"]["diverse"]["languages"],
                "all_directional_gates_passed": receipt["selection"]["gates"][
                    "all_directional_gates_passed"
                ],
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if receipt["status"] == "passed" else 2
        result = verify_selection(
            _resolve(args.manifest, root),
            inputs,
            solver_settings=settings,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (SelectionError, OSError, ValueError) as exc:
        print(f"[language-composition] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
