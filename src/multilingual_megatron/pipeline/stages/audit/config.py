from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass(frozen=True, slots=True)
class AuditConfig:
    enabled: bool
    backend: str
    output_path: Path
    require_prefix_files: bool


def resolve_audit_config(ctx: Any) -> AuditConfig:
    raw_cfg = getattr(ctx, "audit_cfg", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    tokenization_cfg = getattr(ctx, "tokenize_cfg", {})
    if not isinstance(tokenization_cfg, dict):
        tokenization_cfg = {}

    backend = str(raw_cfg.get("backend", tokenization_cfg.get("backend", "megatron_gpt"))).strip().lower()
    raw_output = str(raw_cfg.get("output_path", raw_cfg.get("audit_json", ""))).strip()
    output_path = Path(raw_output) if raw_output else Path(ctx.output_root) / "audit" / "tokenization_audit.json"
    if not output_path.is_absolute():
        output_path = Path(ctx.repo_root) / output_path

    return AuditConfig(
        enabled=_as_bool(raw_cfg.get("enabled", True)),
        backend=backend,
        output_path=output_path,
        require_prefix_files=_as_bool(raw_cfg.get("require_prefix_files", True)),
    )
