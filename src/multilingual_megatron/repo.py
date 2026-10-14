"""Repository path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(anchor: Path | None = None) -> Path:
    """Resolve the study repository root for repo-local orchestration tools."""
    env_root = str(os.environ.get("PROOF_PIPELINE_REPO_ROOT", "")).strip()
    if env_root:
        return Path(env_root).resolve()

    candidates: list[Path] = [Path.cwd().resolve()]
    if anchor is not None:
        resolved = anchor.resolve()
        candidates.extend([resolved, *resolved.parents])

    for candidate in candidates:
        if (candidate / "configs").exists() and (candidate / "src" / "multilingual_megatron").exists():
            return candidate
    return Path.cwd().resolve()
