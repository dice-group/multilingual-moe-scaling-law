"""Persist and normalize A/B-v2 scheduler state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import WorkflowError, _load_object


def _empty_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "jobs": {},
    }


def _load_state(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _empty_state(manifest)
    state = _load_object(path)
    if state.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise WorkflowError("workflow state belongs to a different launch manifest")
    if not isinstance(state.get("jobs"), dict):
        raise WorkflowError("workflow state jobs must be an object")
    return state


def _normalized_slurm_state(value: Any) -> str:
    parts = str(value or "").strip().split(maxsplit=1)
    return parts[0].rstrip("+") if parts else ""
