"""Explicit scheduling amendments for already submitted, sealed studies.

Apply only after validating the original manifest. Keep its identity and scientific
receipts intact; the separate amendment records the operational policy change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multilingual_megatron.experiments.scaling.ab_v2.common import WorkflowError


def apply_scheduling_amendment(payload: dict[str, Any], control: Path) -> None:
    path = control / "scheduling_amendment.json"
    if not path.exists():
        return
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version") != 1
        or amendment.get("manifest_sha256") != payload["manifest_sha256"]
        or amendment.get("policy") != "all_trunks_after_authorization"
        or not amendment.get("authorization")
    ):
        raise WorkflowError("invalid language-composition scheduling amendment")
    jobs = payload["jobs"]
    # Preserve every data, branch, evaluation and retention dependency. Only the
    # artificial cross-trajectory barriers are removed.
    cleanups = [
        row["job_id"] for row in jobs
        if row["job_id"].startswith(("retention_prune_compact_", "retention_prune_diverse_"))
    ]
    trunks = [row for row in jobs if row["job_id"].startswith("lc_trunk_")]
    summaries = [row for row in jobs if row["job_id"] == "language_composition_summary"]
    if len(trunks) != 18 or len(cleanups) != 18 or len(summaries) != 1:
        raise WorkflowError("unexpected language-composition scheduling topology")
    for row in trunks:
        row["dependencies"] = ["language_composition_authorization_gate"]
    summary = summaries[0]
    summary["dependencies"] = list(dict.fromkeys([*summary["dependencies"], *cleanups]))
