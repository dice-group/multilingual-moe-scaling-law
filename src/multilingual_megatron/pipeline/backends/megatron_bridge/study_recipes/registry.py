# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Register Stage 1 dense and Switch-MoE recipe names."""

from __future__ import annotations

from ..scaling_registry import (
    ARCHITECTURE_FAMILIES,
    SCALE_SPECS,
    recipe_name as _scaling_recipe_name,
)
from .generic import (
    _scaling_dense_config,
    _scaling_switch_moe_config,
)
from .registration import _make_scaling_recipe


RECIPE_EXPORTS: dict[str, object] = {}


def _export(name: str, recipe: object) -> None:
    RECIPE_EXPORTS[name] = recipe
    globals()[name] = recipe


_SCALING_BUILDERS = {
    "scaling_dense": _scaling_dense_config,
    "scaling_switch_moe": _scaling_switch_moe_config,
}


for _scale_label in SCALE_SPECS:
    for _family, _family_cfg in ARCHITECTURE_FAMILIES.items():
        _builder_name = str(_family_cfg["builder"])
        _recipe_name = _scaling_recipe_name(_family, _scale_label)
        _export(
            _recipe_name,
            _make_scaling_recipe(
                _recipe_name,
                _scale_label,
                str(_family_cfg["arch_key"]),
                _SCALING_BUILDERS[_builder_name],
            ),
        )


__all__ = tuple(sorted(RECIPE_EXPORTS))
