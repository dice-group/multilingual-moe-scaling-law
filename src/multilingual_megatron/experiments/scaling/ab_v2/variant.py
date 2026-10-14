"""Explicit immutable study-variant contracts for scaling workflows.

The A/B, C, and D ladders share one orchestration implementation but have
different, reviewable execution contracts.  Variants replace the old pattern
of mutating module globals and monkey-patching private helpers at import time.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


SpecValidator = Callable[..., None]
ResourceSelector = Callable[[str, Mapping[str, Any]], str]
JobEnvironmentTransform = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, str]
]
TrainingOverridesTransform = Callable[..., list[str]]
ProfileTransform = Callable[..., dict[str, Any]]
EvaluationSpecTransform = Callable[..., dict[str, Any]]
ProcessCount = Callable[[int], int]
ThroughputEvidenceRequired = Callable[[Mapping[str, Any]], bool]
ProductionTopologyValidator = Callable[[Sequence[Mapping[str, Any]]], None]
CustomActionHandler = Callable[
    [Mapping[str, Any], Mapping[str, Any], Any, Mapping[str, Any]], bool
]


def _identity_resource(resource: str, _arguments: Mapping[str, Any]) -> str:
    return resource


def _empty_environment(
    _manifest: Mapping[str, Any],
    _job: Mapping[str, Any],
    _runtime: Mapping[str, Any],
) -> Mapping[str, str]:
    return {}


def _identity_training_overrides(
    overrides: list[str], **_kwargs: Any
) -> list[str]:
    return overrides


def _identity_profile(profile: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    return profile


def _identity_evaluation_spec(
    spec: dict[str, Any], **_kwargs: Any
) -> dict[str, Any]:
    return spec


def _single_process(_experts: int) -> int:
    return 1


def _throughput_evidence_required(_runtime: Mapping[str, Any]) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class WorkflowPolicy:
    """Values that define a ladder's immutable scientific contract."""

    default_spec: Path
    default_runtime: Path
    default_output: Path
    ladders: tuple[str, ...]
    expected_micro_batch_size: Mapping[str, Mapping[str, int]]
    required_sentinel_probes: frozenset[tuple[str, str, int, int]]
    fixed_lr_schedule_probe_cells: tuple[tuple[str, int, int], ...]
    required_training_env_keys: frozenset[str]
    spec_validator: SpecValidator
    prescribed_lr_schedule_probe_cells: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class WorkflowVariant:
    """An explicit execution adapter for one ladder family.

    The policy is recorded in the manifest and every non-default behavior is
    a named callback owned by the variant module.  The callbacks are invoked
    only within :func:`variant_scope`, which is entered at public workflow
    boundaries rather than through import-time mutation.
    """

    identifier: str
    worker_module: str
    policy: WorkflowPolicy
    resource_selector: ResourceSelector = _identity_resource
    job_environment_transform: JobEnvironmentTransform = _empty_environment
    training_overrides_transform: TrainingOverridesTransform = (
        _identity_training_overrides
    )
    profile_transform: ProfileTransform = _identity_profile
    evaluation_spec_transform: EvaluationSpecTransform = _identity_evaluation_spec
    processes_for_experts: ProcessCount = _single_process
    throughput_evidence_required: ThroughputEvidenceRequired = (
        _throughput_evidence_required
    )
    production_training_seeds: tuple[int, ...] | None = None
    production_topology_validator: ProductionTopologyValidator | None = None
    custom_action_handler: CustomActionHandler | None = None
    sentinel_startup_recovery_actions: frozenset[str] = frozenset()
    authorized_production: bool = False
    provenance_files: tuple[Path, ...] = ()

    def manifest_descriptor(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "worker_module": self.worker_module,
            "ladders": list(self.policy.ladders),
            "sentinel_startup_recovery_actions": sorted(
                self.sentinel_startup_recovery_actions
            ),
        }


_DEFAULT_VARIANT: WorkflowVariant | None = None
_ACTIVE_VARIANT: ContextVar[WorkflowVariant | None] = ContextVar(
    "ab_v2_active_workflow_variant", default=None
)


def set_default_variant(variant: WorkflowVariant) -> None:
    """Install the base A/B variant once during package initialization."""

    global _DEFAULT_VARIANT
    if _DEFAULT_VARIANT is not None and _DEFAULT_VARIANT != variant:
        raise RuntimeError("the default workflow variant is already configured")
    _DEFAULT_VARIANT = variant


def active_variant() -> WorkflowVariant:
    variant = _ACTIVE_VARIANT.get() or _DEFAULT_VARIANT
    if variant is None:
        raise RuntimeError("workflow variant was requested before package initialization")
    return variant


def active_policy() -> WorkflowPolicy:
    return active_variant().policy


@contextmanager
def variant_scope(variant: WorkflowVariant) -> Iterator[None]:
    """Bind one immutable variant to a single plan/validate/run operation."""

    token = _ACTIVE_VARIANT.set(variant)
    try:
        yield
    finally:
        _ACTIVE_VARIANT.reset(token)


def variant_input_files() -> Sequence[Path]:
    """Return additional variant-owned files that must enter a manifest hash."""

    return active_variant().provenance_files
