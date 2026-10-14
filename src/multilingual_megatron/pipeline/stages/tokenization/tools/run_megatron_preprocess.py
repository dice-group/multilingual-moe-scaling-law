"""Run Megatron's preprocessing script with orchestration-owned import fixes.

The cluster training environment can expose ``nvidia_resiliency_ext`` as a
namespace package without ``__version__``.  Importing Megatron-Core then fails
before the CPU-only preprocessing entrypoint is reached.  Training already
handles optional dependency compatibility in its orchestration wrapper; this
small entrypoint applies the corresponding non-invasive fix without modifying
vendored Megatron-LM.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--script", required=True)
    args, forwarded = parser.parse_known_args(argv)
    script = Path(args.script).resolve()
    if not script.is_file():
        raise SystemExit(f"missing Megatron preprocessing script: {script}")

    try:
        import nvidia_resiliency_ext as nvrx
    except ImportError:
        nvrx = None
    if nvrx is not None and not hasattr(nvrx, "__version__"):
        # The installed checkpointing symbols are from the 0.6 API, but the
        # namespace-package build omitted only its top-level version marker.
        nvrx.__version__ = "0.6.0"

    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
