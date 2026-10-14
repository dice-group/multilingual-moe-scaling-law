"""Mutable scheduler state kept separate from immutable scientific artifacts."""
from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..data import canonical_json_bytes, canonical_json_sha256
from ..execution import read_json
from .contracts import STATE_SCHEMA, BaselineRefitError, control_root


def state_path(workflow: Mapping[str, Any]) -> Path:
    return control_root(workflow) / "submission_state.json"


def _state_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("state_sha256", None)
    unsigned["state_sha256"] = canonical_json_sha256(unsigned)
    return unsigned


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    payload = _state_payload(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(workflow: Mapping[str, Any]) -> dict[str, Any]:
    path = state_path(workflow)
    if not path.exists():
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return {
            "schema_version": STATE_SCHEMA,
            "workflow_sha256": workflow["workflow_sha256"],
            "attempt_id": f"{attempt}-{str(workflow['workflow_sha256'])[:8]}",
            "jobs": {},
        }
    state = read_json(path)
    digest = state.get("state_sha256")
    unsigned = dict(state)
    unsigned.pop("state_sha256", None)
    if digest != canonical_json_sha256(unsigned) \
            or state.get("schema_version") != STATE_SCHEMA \
            or state.get("workflow_sha256") != workflow["workflow_sha256"] \
            or not isinstance(state.get("jobs"), dict):
        raise BaselineRefitError("submission state is invalid or belongs to another workflow")
    state.pop("state_sha256", None)
    return state


@contextmanager
def submission_lock(workflow: Mapping[str, Any]) -> Iterator[None]:
    path = control_root(workflow) / ".submit.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
