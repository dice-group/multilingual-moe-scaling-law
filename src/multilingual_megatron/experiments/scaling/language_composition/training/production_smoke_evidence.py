"""Validate disclosed-boundary reuse of immutable language-composition smokes."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.ab_v2.artifacts import (
    _verified_receipt_evidence,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import (
    Paths,
    WorkflowError,
    _atomic_json,
    _file_sha256,
    _load_object,
    _payload_sha256,
)

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    DESIGN_ID,
    SMOKE_GLOBAL_BATCH_SIZE,
    SMOKE_MAX_STEP_TIME_CV_PCT,
    SMOKE_TOKENS,
)


DECISION = "authorized_with_disclosed_boundary"
EVIDENCE_GATE_JOB_ID = "language_composition_smoke_evidence_gate"
EVIDENCE_GATE_FILENAME = "language_composition_smoke_evidence_gate.json"
EVIDENCE_GATE_SCHEMA = "stage1_language_composition_smoke_evidence_gate/v1"
AUTHORIZATION_SCHEMA = "stage1_language_composition_authorization/v2"
SMOKE_JOB_CONTRACT = {
    "smoke_compact_b_e1_s1234": ("compact", 1),
    "smoke_compact_b_e32_s1234": ("compact", 32),
    "smoke_diverse_b_e1_s1234": ("diverse", 1),
    "smoke_diverse_b_e32_s1234": ("diverse", 32),
}
ALLOWED_WAIVED_ERRORS = (
    "smoke_diverse_b_e1_s1234: steady-state stability",
)
CONFIG_KEYS = {
    "source_manifest",
    "source_manifest_file_sha256",
    "source_manifest_payload_sha256",
    "source_smoke_gate",
    "source_smoke_gate_file_sha256",
    "source_smoke_gate_payload_sha256",
    "decision",
    "waived_errors",
    "rationale",
    "authorized_by",
    "authorized_at_utc",
}


def validate_evidence_config(
    raw: Any, *, require_files: bool
) -> dict[str, Any]:
    """Validate the exact, user-authorized source-evidence descriptor."""

    if not isinstance(raw, Mapping) or set(raw) != CONFIG_KEYS:
        raise WorkflowError("production smoke-evidence contract is incomplete")
    evidence = dict(raw)
    hashes = (
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "source_smoke_gate_file_sha256",
        "source_smoke_gate_payload_sha256",
    )
    if any(
        not isinstance(evidence.get(key), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(evidence[key])) is None
        for key in hashes
    ):
        raise WorkflowError("production smoke-evidence hashes are malformed")
    if (
        evidence.get("decision") != DECISION
        or tuple(evidence.get("waived_errors", ())) != ALLOWED_WAIVED_ERRORS
        or not str(evidence.get("rationale", "")).strip()
        or not str(evidence.get("authorized_by", "")).strip()
        or not str(evidence.get("authorized_at_utc", "")).endswith("Z")
    ):
        raise WorkflowError("production smoke-evidence decision drifted")
    for key in ("source_manifest", "source_smoke_gate"):
        value = evidence.get(key)
        if not isinstance(value, str) or not value.strip():
            raise WorkflowError(f"production smoke evidence lacks {key}")
        path = Path(value).resolve()
        if require_files and (path.is_symlink() or not path.is_file()):
            raise WorkflowError(f"production smoke evidence is missing: {path}")
    if require_files:
        _validated_source(evidence)
    return evidence


def _validated_source(
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_manifest_path = Path(str(evidence["source_manifest"])).resolve()
    source_gate_path = Path(str(evidence["source_smoke_gate"])).resolve()
    if _file_sha256(source_manifest_path) != evidence["source_manifest_file_sha256"]:
        raise WorkflowError("production smoke source-manifest file changed")
    if _file_sha256(source_gate_path) != evidence["source_smoke_gate_file_sha256"]:
        raise WorkflowError("production smoke source-gate file changed")
    source_manifest = _load_object(source_manifest_path)
    manifest_candidate = dict(source_manifest)
    embedded_manifest_hash = manifest_candidate.pop("manifest_sha256", None)
    if (
        embedded_manifest_hash != evidence["source_manifest_payload_sha256"]
        or embedded_manifest_hash != _payload_sha256(manifest_candidate)
        or source_manifest.get("language_composition_design", {}).get("design_id")
        != DESIGN_ID
    ):
        raise WorkflowError("production smoke source manifest is invalid")
    source_root = Path(str(source_manifest.get("output_root", ""))).resolve()
    if source_manifest_path != source_root / "control" / "launch_manifest.json":
        raise WorkflowError("production smoke source manifest is outside its root")
    if source_gate_path != source_root / "control" / "language_composition_smoke_gate.json":
        raise WorkflowError("production smoke source gate is outside its root")

    source_gate = _load_object(source_gate_path)
    gate_candidate = dict(source_gate)
    embedded_gate_hash = gate_candidate.pop("payload_sha256", None)
    if (
        embedded_gate_hash != evidence["source_smoke_gate_payload_sha256"]
        or embedded_gate_hash != _payload_sha256(gate_candidate)
        or source_gate.get("schema") != "stage1_language_composition_smoke_gate/v2"
        or source_gate.get("design_id") != DESIGN_ID
        or source_gate.get("manifest_sha256") != embedded_manifest_hash
        or source_gate.get("status") != "failed"
        or tuple(source_gate.get("errors", ())) != ALLOWED_WAIVED_ERRORS
    ):
        raise WorkflowError("production smoke source gate is not the disclosed failure")
    rows = source_gate.get("evidence")
    if not isinstance(rows, Mapping) or set(rows) != set(SMOKE_JOB_CONTRACT):
        raise WorkflowError("production smoke source gate has incomplete evidence")

    observed: dict[str, Any] = {}
    for job_id, (arm, experts) in SMOKE_JOB_CONTRACT.items():
        row = rows[job_id]
        if not isinstance(row, Mapping):
            raise WorkflowError(f"production smoke gate row is malformed: {job_id}")
        expected_errors = ["steady-state stability"] if (
            f"{job_id}: steady-state stability" in ALLOWED_WAIVED_ERRORS
        ) else []
        if row.get("errors") != expected_errors:
            raise WorkflowError(f"production smoke gate row errors changed: {job_id}")
        result_path = source_root / "jobs" / job_id / "result.json"
        receipt_path = source_root / "control" / "receipts" / f"{job_id}.json"
        result_descriptor = row.get("result")
        receipt_descriptor = row.get("receipt")
        if (
            not isinstance(result_descriptor, Mapping)
            or not isinstance(receipt_descriptor, Mapping)
            or Path(str(result_descriptor.get("path", ""))).resolve() != result_path
            or Path(str(receipt_descriptor.get("path", ""))).resolve() != receipt_path
            or _file_sha256(result_path) != result_descriptor.get("sha256")
            or _file_sha256(receipt_path) != receipt_descriptor.get("sha256")
        ):
            raise WorkflowError(f"production smoke evidence identity changed: {job_id}")
        receipt = _verified_receipt_evidence(source_manifest, job_id)
        result = _load_object(result_path)
        arguments = result.get("arguments")
        health = result.get("sentinel_health")
        metrics = health.get("metrics") if isinstance(health, Mapping) else None
        if (
            result.get("status") != "complete"
            or not isinstance(arguments, Mapping)
            or arguments.get("composition_arm") != arm
            or int(arguments.get("experts", 0)) != experts
            or int(arguments.get("tokens", 0)) != SMOKE_TOKENS
            or int(arguments.get("global_batch_size", 0))
            != SMOKE_GLOBAL_BATCH_SIZE
            or int(result.get("actual_end_tokens", 0)) < SMOKE_TOKENS
            or result.get("checkpoint_state", {}).get("valid_full_state") is not True
            or not isinstance(health, Mapping)
            or health.get("status") != "ok"
            or health.get("errors") != []
            or not isinstance(metrics, Mapping)
            or metrics.get("finite_loss_and_gradients") is not True
            or int(metrics.get("max_skipped_iterations", -1)) != 0
            or int(metrics.get("max_nan_iterations", -1)) != 0
            or int(metrics.get("nonfinite_lm_loss_count", -1)) != 0
            or int(metrics.get("nonfinite_grad_norm_count", -1)) != 0
        ):
            raise WorkflowError(f"production smoke health changed: {job_id}")
        if experts == 32 and (
            float(metrics.get("max_dropped_assignment_rate", math.nan)) != 0.0
            or float(metrics.get("max_tokens_with_zero_kept_experts", math.nan))
            != 0.0
        ):
            raise WorkflowError(f"production smoke MoE routing changed: {job_id}")
        try:
            throughput_fraction = float(row["throughput_fraction"])
            cv_pct = float(row["step_time_cv_pct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"production smoke performance row is malformed: {job_id}"
            ) from exc
        if not math.isfinite(throughput_fraction) or throughput_fraction < 0.85:
            raise WorkflowError(f"production smoke throughput floor failed: {job_id}")
        if expected_errors:
            if not math.isfinite(cv_pct) or cv_pct < SMOKE_MAX_STEP_TIME_CV_PCT:
                raise WorkflowError(f"production smoke waived boundary changed: {job_id}")
        elif not math.isfinite(cv_pct) or cv_pct >= SMOKE_MAX_STEP_TIME_CV_PCT:
            raise WorkflowError(f"production smoke unwaived stability failed: {job_id}")
        observed[job_id] = {
            "composition_arm": arm,
            "experts": experts,
            "actual_end_tokens": result["actual_end_tokens"],
            "valid_full_state": True,
            "median_tokens_per_second": row["median_tokens_per_second"],
            "throughput_fraction": throughput_fraction,
            "step_time_cv_pct": cv_pct,
            "waived_errors": expected_errors,
            "result": {
                "path": str(result_path),
                "sha256": _file_sha256(result_path),
            },
            "receipt": receipt,
        }
    return source_manifest, source_gate, observed


def fixed_input_paths(evidence: Mapping[str, Any]) -> set[Path]:
    """Return the complete immutable external smoke-evidence closure."""

    source_manifest, source_gate, _observed = _validated_source(evidence)
    paths = {
        Path(str(evidence["source_manifest"])).resolve(),
        Path(str(evidence["source_smoke_gate"])).resolve(),
    }
    for row in source_gate["evidence"].values():
        paths.add(Path(str(row["result"]["path"])).resolve())
        paths.add(Path(str(row["receipt"]["path"])).resolve())
    if source_manifest.get("manifest_sha256") != evidence["source_manifest_payload_sha256"]:
        raise WorkflowError("production smoke source manifest identity changed")
    return paths


def run_evidence_gate(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    evidence = validate_evidence_config(job.get("arguments"), require_files=True)
    source_manifest, source_gate, observed = _validated_source(evidence)
    sealed = manifest.get("data_materialization", {}).get("sealed_corpus")
    source_sealed = source_manifest.get("data_materialization", {}).get(
        "sealed_corpus"
    )
    if (
        not isinstance(sealed, Mapping)
        or not isinstance(source_sealed, Mapping)
        or sealed.get("data_seal_sha256") != source_sealed.get("data_seal_sha256")
    ):
        raise WorkflowError("production and source smokes use different sealed data")
    report: dict[str, Any] = {
        "schema": EVIDENCE_GATE_SCHEMA,
        "status": DECISION,
        "design_id": DESIGN_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_manifest": {
            "path": str(Path(str(evidence["source_manifest"])).resolve()),
            "file_sha256": evidence["source_manifest_file_sha256"],
            "payload_sha256": source_manifest["manifest_sha256"],
        },
        "source_smoke_gate": {
            "path": str(Path(str(evidence["source_smoke_gate"])).resolve()),
            "file_sha256": evidence["source_smoke_gate_file_sha256"],
            "payload_sha256": source_gate["payload_sha256"],
            "observed_status": source_gate["status"],
        },
        "decision": evidence["decision"],
        "waived_errors": list(evidence["waived_errors"]),
        "rationale": evidence["rationale"],
        "authorized_by": evidence["authorized_by"],
        "authorized_at_utc": evidence["authorized_at_utc"],
        "verified_smokes": observed,
        "claim_boundary": (
            "Only the disclosed diverse/E1 final-window timing-CV failure is waived; "
            "training health, checkpoint, throughput-floor, and MoE-routing gates remain mandatory."
        ),
    }
    report["payload_sha256"] = _payload_sha256(report)
    _atomic_json(paths.control / EVIDENCE_GATE_FILENAME, report)
