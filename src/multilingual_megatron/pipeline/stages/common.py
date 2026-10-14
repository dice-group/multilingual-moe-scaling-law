"""Common helpers for proof-pipeline stage handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def lm_eval_harness_root(repo_root: Path) -> Path | None:
    candidate = repo_root / "external" / "lm-evaluation-harness"
    return candidate if (candidate / "lm_eval").exists() else None


def ensure_lm_eval_layout(repo_root: Path, logger: Any) -> None:
    if lm_eval_harness_root(repo_root) is None:
        logger.warning("[WARN] local lm-evaluation-harness submodule is unavailable under external/")
