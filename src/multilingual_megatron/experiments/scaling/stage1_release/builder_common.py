"""Canonical scalar, JSON-cell, timestamp, and process helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    _canonical_bytes,
    _json_cell,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_text(command: Sequence[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _columnar_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert nested cells to canonical JSON while preserving scalar types."""

    result: list[dict[str, Any]] = []
    for source in records:
        row: dict[str, Any] = {}
        for key, value in source.items():
            if isinstance(value, (Mapping, list, tuple, set)):
                row[str(key)] = _json_cell(_json_safe(value))
            else:
                row[str(key)] = _json_safe(value)
        result.append(row)
    return result


def _canonical_qa_evidence_cell(value: Any) -> str | None:
    """Render heterogeneous observed/expected QA evidence as canonical JSON."""

    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return _json_cell(_json_safe(value))


def _assign_qa_issue_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach stable, unique IDs without treating build timestamps as identity."""

    occurrences: defaultdict[str, int] = defaultdict(int)
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        identity_payload = {
            key: value
            for key, value in sorted(row.items())
            if key not in {"checked_at_utc", "qa_issue_id"}
        }
        digest = hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        row["qa_issue_id"] = f"qa-{digest[:24]}-{occurrence:03d}"
        result.append(row)
    return result
