"""Verify the external B-ladder LR evidence that authorizes the A/B rerun."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import (
    Paths,
    WorkflowError,
    _abs,
    _atomic_json,
    _file_sha256,
    _load_object,
    _payload_sha256,
)


def validate_production_lr_evidence(args: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen B-ladder evidence without creating a Slurm gate."""

    gate_path = _abs(str(args["source_gate_path"]))
    recovery_path = _abs(str(args["source_recovery_path"]))
    if _file_sha256(gate_path) != args["source_gate_file_sha256"]:
        raise WorkflowError("external LR-confirmation gate file hash drifted")
    if _file_sha256(recovery_path) != args["source_recovery_file_sha256"]:
        raise WorkflowError("external LR-confirmation recovery evidence hash drifted")
    source = _load_object(gate_path)
    recovery = _load_object(recovery_path)
    analysis = source.get("analysis", {})
    if (
        source.get("gate_sha256") != args["source_gate_payload_sha256"]
        or _payload_sha256({k: v for k, v in source.items() if k != "gate_sha256"})
        != args["source_gate_payload_sha256"]
        or recovery.get("manifest_sha256") != args["source_manifest_sha256"]
        or analysis.get("decision") != args["observed_decision"]
        or analysis.get("boundary_cells") != ["B/E1/K1"]
        or not analysis.get("checks", {}).get(
            "x1_strictly_lowest_equal_cell_mean_regret"
        )
        or not analysis.get("checks", {}).get(
            "x1_wins_or_ties_at_least_required_cells"
        )
        or not analysis.get("checks", {}).get(
            "x1_beats_fixed_2e4_in_equal_cell_mean_primary_loss"
        )
    ):
        raise WorkflowError("external LR-confirmation evidence content drifted")
    return {
        "schema_version": 1,
        "status": "authorized_with_disclosed_boundary",
        "selected_rule": args["selected_rule"],
        "observed_decision": args["observed_decision"],
        "deferred_boundary_cell": args["deferred_boundary_cell"],
        "source_gate": str(gate_path),
        "source_gate_sha256": args["source_gate_file_sha256"],
        "source_manifest_sha256": args["source_manifest_sha256"],
        "interpretation": (
            "Manual prospective authorization of aggregate JointMoE x1 for A/B; "
            "the B/E1/K1 x4 boundary observation remains disclosed and deferred."
        ),
    }


def _run_production_lr_evidence_gate(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    report = validate_production_lr_evidence(job["arguments"])
    report["gate_sha256"] = _payload_sha256(report)
    _atomic_json(paths.control / "production_lr_evidence_gate.json", report)
