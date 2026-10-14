"""Compatibility import; implementation lives in multilingual_megatron.experiments.scaling.language_composition.selection.cli."""

import sys
from importlib import import_module

_implementation = import_module("multilingual_megatron.experiments.scaling.language_composition.selection.cli")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
