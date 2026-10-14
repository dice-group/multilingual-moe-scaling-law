"""Compatibility import; implementation lives in multilingual_megatron.experiments.scaling.language_composition.selection.distances."""

import sys
from importlib import import_module

_implementation = import_module("multilingual_megatron.experiments.scaling.language_composition.selection.distances")

sys.modules[__name__] = _implementation
