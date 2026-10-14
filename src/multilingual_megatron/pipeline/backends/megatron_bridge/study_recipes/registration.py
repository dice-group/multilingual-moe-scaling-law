"""Stage 1 scaling-recipe factory."""

from __future__ import annotations

from ..scaling_registry import SCALE_SPECS, scale_spec as _scaling_scale_spec


def _make_scaling_recipe(name: str, scale_label: str, arch_key: str, builder):
    def _recipe() -> ConfigContainer:
        return builder(_scaling_scale_spec(scale_label, arch_key))

    _recipe.__name__ = name
    _recipe.__qualname__ = name
    _recipe.__module__ = builder.__module__
    target = SCALE_SPECS[scale_label].get("target_active_params", SCALE_SPECS[scale_label].get("target_total_params"))
    _recipe.__doc__ = (
        f"Return the Stage 1 scaling config for "
        f"{name.removesuffix('_pretrain_config')} targeting about {target:,} parameters."
    )
    return _recipe
