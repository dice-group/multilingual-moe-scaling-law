"""The immutable A/B policy used by the shared workflow implementation."""

from __future__ import annotations

from multilingual_megatron.experiments.scaling.ab_v2.spec import validate_ab_v2_spec

from .common import (
    DEFAULT_OUTPUT,
    DEFAULT_RUNTIME,
    DEFAULT_SPEC,
    EXPECTED_MICRO_BATCH_SIZE,
    FIXED_LR_SCHEDULE_PROBE_CELLS,
    LADDERS,
    PRESCRIBED_LR_SCHEDULE_PROBE_CELLS,
    REQUIRED_SENTINEL_PROBES,
    REQUIRED_TRAINING_ENV_KEYS,
)
from .variant import WorkflowPolicy, WorkflowVariant, set_default_variant


AB_POLICY = WorkflowPolicy(
    default_spec=DEFAULT_SPEC,
    default_runtime=DEFAULT_RUNTIME,
    default_output=DEFAULT_OUTPUT,
    ladders=LADDERS,
    expected_micro_batch_size=EXPECTED_MICRO_BATCH_SIZE,
    required_sentinel_probes=frozenset(REQUIRED_SENTINEL_PROBES),
    fixed_lr_schedule_probe_cells=FIXED_LR_SCHEDULE_PROBE_CELLS,
    required_training_env_keys=frozenset(REQUIRED_TRAINING_ENV_KEYS),
    spec_validator=validate_ab_v2_spec,
    prescribed_lr_schedule_probe_cells=PRESCRIBED_LR_SCHEDULE_PROBE_CELLS,
)

AB_VARIANT = WorkflowVariant(
    identifier="ab_v2",
    worker_module="multilingual_megatron.experiments.scaling.ab_v2.workflow",
    policy=AB_POLICY,
)

set_default_variant(AB_VARIANT)
