#!/usr/bin/env python3
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

"""Compatibility facade for study-local Megatron Bridge recipes.

Implementations are owned by architecture-family modules under
``study_recipes``. This module retains every existing recipe lookup name used
by pipeline profiles and ``run_study_bridge_recipe``.
"""

from __future__ import annotations

from .study_recipes.registry import RECIPE_EXPORTS


globals().update(RECIPE_EXPORTS)

# Preserve the public module path used by serialized production recipes.
for _recipe in RECIPE_EXPORTS.values():
    _recipe.__module__ = __name__

__all__ = tuple(sorted(RECIPE_EXPORTS))
