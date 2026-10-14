"""Compatibility import; implementation lives in multilingual_megatron.experiments.scaling.language_composition.selection.optimization."""

import sys
from importlib import import_module

_implementation = import_module("multilingual_megatron.experiments.scaling.language_composition.selection.optimization")

sys.modules[__name__] = _implementation
