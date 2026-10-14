"""Source-checkout entry point for the CPU paper reproduction workflow."""
from pathlib import Path
import sys

from multilingual_megatron.repo import find_repo_root


def main(argv=None):
    root = find_repo_root(Path(__file__))
    if not (root / "result_analysis/reproduce.py").is_file():
        raise SystemExit("Run from the source checkout or reviewer export; a wheel alone does not include result_analysis.")
    sys.path.insert(0, str(root))
    from result_analysis.reproduce import main as reproduce
    return reproduce(argv)


if __name__ == "__main__":
    raise SystemExit(main())
