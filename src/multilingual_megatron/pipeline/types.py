"""Shared stage contracts for pipeline packages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Expected output artifact owned by a stage."""

    path: Path
    required: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class StagePlan:
    """Resolved execution plan for a stage."""

    stage: str
    enabled: bool = True
    reason: str = ""
    expected_artifacts: tuple[ArtifactSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Result returned by a stage implementation."""

    stage: str
    status: str = "ok"
    artifacts: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    """Common public interface for all pipeline stages."""

    name: str

    def plan(self, ctx: Any) -> StagePlan:
        """Resolve the stage plan from the pipeline context."""

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        """Execute the stage plan."""

    def validate(self, ctx: Any, result: StageResult) -> None:
        """Validate stage artifacts after execution."""


class FunctionStage:
    """Adapter for legacy stage functions while packages are migrated."""

    def __init__(self, name: str, handler: Callable[[Any], None], *, enabled_key: str | None = None) -> None:
        self.name = name
        self._handler = handler
        self._enabled_key = enabled_key

    def plan(self, ctx: Any) -> StagePlan:
        cfg = getattr(ctx, self._enabled_key, {}) if self._enabled_key else {}
        enabled = bool(cfg.get("enabled", True)) if isinstance(cfg, dict) else True
        return StagePlan(stage=self.name, enabled=enabled)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        self._handler(ctx)
        return StageResult(stage=self.name)

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


class NoOpStage:
    """Disabled placeholder for stages that are architectural slots in v1."""

    def __init__(self, name: str, *, reason: str) -> None:
        self.name = name
        self._reason = reason

    def plan(self, ctx: Any) -> StagePlan:
        del ctx
        return StagePlan(stage=self.name, enabled=False, reason=self._reason)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        del ctx
        return StageResult(stage=self.name, status="skipped", metadata={"reason": plan.reason or self._reason})

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result
