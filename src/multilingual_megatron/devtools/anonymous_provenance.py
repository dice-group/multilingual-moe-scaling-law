"""Anonymous labels for author-side source provenance, not dependency revisions."""

from __future__ import annotations

import hashlib
import re
from typing import Any


GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)
AUTHOR_SOURCE_FIELDS = frozenset({
    "paper_commit", "baseline_commit", "source_commit", "git_commit", "base_commit",
    "preparation_base_commit", "retry_preparation_base_commit",
    "terminal_gate_fix_base_commit", "calibration_source_commit",
    "preserved_m6_source_commit", "bridge_commit", "harness_commit",
})


def anonymous_source_id(value: str) -> str:
    """Map a source commit to a stable opaque label; preserve labels and fixtures."""
    if GIT_OBJECT_ID.fullmatch(value):
        return "snapshot-" + hashlib.sha256(value.encode()).hexdigest()[:16]
    return value


def anonymize_text_source_ids(text: str) -> str:
    """Redact JSON source fields without rewriting numeric representations."""
    keys = "|".join(re.escape(key) for key in sorted(AUTHOR_SOURCE_FIELDS))
    pattern = re.compile(r'("(?:' + keys + r')"\s*:\s*")([0-9a-f]{40})(")', re.I)
    return pattern.sub(lambda m: m[1] + anonymous_source_id(m[2]) + m[3], text)


def anonymize_metadata(value: Any) -> Any:
    """Redact explicit author-source fields, retaining public revisions/checksums."""
    if isinstance(value, dict):
        return {
            key: anonymous_source_id(item)
            if key in AUTHOR_SOURCE_FIELDS and isinstance(item, str)
            else anonymize_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [anonymize_metadata(item) for item in value]
    return value
