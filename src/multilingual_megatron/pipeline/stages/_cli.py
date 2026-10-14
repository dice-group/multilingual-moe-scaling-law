"""Shared CLI helper for package-level stage commands."""

from __future__ import annotations

from multilingual_megatron.pipeline.cli import main as pipeline_main


def run_single_stage(stage_name: str) -> int:
    """Run the top-level pipeline CLI constrained to one stage.

    Stage packages expose this for discoverability while keeping argument
    behavior identical to the main pipeline CLI.
    """

    import sys

    argv = sys.argv[1:]
    if "--start-at" not in argv:
        argv.extend(["--start-at", stage_name])
    if "--stop-after" not in argv:
        argv.extend(["--stop-after", stage_name])
    sys.argv = [sys.argv[0], *argv]
    return pipeline_main()
