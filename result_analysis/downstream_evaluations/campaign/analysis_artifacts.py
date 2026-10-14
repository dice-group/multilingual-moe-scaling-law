"""Artifact IO and fail-closed contracts for Stage-1 benchmark analysis."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.contracts.hashing import (
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
    CampaignProfile,
)


ANALYSIS_BUNDLE_SCHEMA = "stage1_jointmoe_benchmark_analysis_bundle/v1"
INTEGRITY_SCHEMA = "stage1_jointmoe_benchmark_analysis_integrity/v1"
FINDINGS_SCHEMA = "stage1_jointmoe_benchmark_initial_findings/v1"
EXPECTED_CAMPAIGN_ID = "stage1_abcd_jointmoe_all_readouts_a40_v2"
EXPECTED_TABLE_ROWS = {
    "model_readouts": 508,
    "benchmark_metrics": 3_048,
    "language_metrics": 73_728,
}
EXPECTED_TRAJECTORIES = 82
EXPECTED_FINAL_READOUTS = 82
EXPECTED_SOURCE_COHORTS = {"canonical_release": 400, "b_seed_confirmation": 108}
EXPECTED_LADDERS = {"A": 80, "B": 204, "C": 112, "D": 112}
REPLICATED_B_EXPERTS = (1, 16, 32)
REPLICATED_B_LANGUAGES = (1, 12, 200)
REPLICATED_B_SEEDS = (1234, 5678, 91011)
REPLICATED_B_TOKENS = (
    500_000_000,
    1_000_000_000,
    2_000_000_000,
    4_000_000_000,
    8_000_000_000,
    16_000_000_000,
)
COMMON_ENGLISH_SLICE = (
    ("belebele_cloze", "eng_Latn"),
    ("xstorycloze", "en"),
    ("uhura_arc_easy", "en"),
    ("global_mmlu_cloze", "en"),
)
POST_HOC_THRESHOLDS = (0.0, 0.05, 0.10, 0.20)
FLOAT_TOLERANCE = 1e-12


class BenchmarkAnalysisError(ValueError):
    """Raised when sealed inputs or analysis-grain contracts drift."""


ANALYSIS_IMPLEMENTATION_FILES = {
    "facade": Path(__file__).with_name("analysis.py").resolve(),
    "artifacts": Path(__file__).resolve(),
    "scores": Path(__file__).with_name("analysis_scores.py").resolve(),
    "replication": Path(__file__).with_name("analysis_replication.py").resolve(),
    "reporting": Path(__file__).with_name("analysis_reporting.py").resolve(),
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkAnalysisError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkAnalysisError(f"{label} must contain a JSON object: {path}")
    return payload


def _verify_payload_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise BenchmarkAnalysisError(f"{label}.{field} is not a SHA-256 digest")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    actual = canonical_json_sha256(unhashed)
    if actual != expected:
        raise BenchmarkAnalysisError(
            f"{label} payload hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def _resolved_reference(
    reference: Mapping[str, Any], *, expected_path: Path, label: str
) -> Path:
    raw_path = reference.get("path")
    expected_sha = reference.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
        raise BenchmarkAnalysisError(f"invalid {label} file reference")
    path = Path(raw_path).resolve()
    if path != expected_path.resolve():
        raise BenchmarkAnalysisError(
            f"{label} path drift: expected {expected_path.resolve()}, got {path}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise BenchmarkAnalysisError(
            f"{label} hash mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return path


def _read_jsonl(path: Path, *, columns: Sequence[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_columns = set(columns)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkAnalysisError(
                    f"invalid JSONL in {label} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or set(row) != expected_columns:
                raise BenchmarkAnalysisError(
                    f"{label} column drift at line {line_number}"
                )
            rows.append(row)
    return rows


def _validate_csv(path: Path, *, columns: Sequence[str], rows: int, label: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise BenchmarkAnalysisError(f"{label} CSV header drift")
        observed = sum(1 for _row in reader)
    if observed != rows:
        raise BenchmarkAnalysisError(
            f"{label} CSV row-count drift: expected {rows}, got {observed}"
        )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True)
    return value


def _write_table(
    root: Path, name: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows:
        raise BenchmarkAnalysisError(f"analysis table {name} has no rows")
    columns = list(rows[0])
    expected_columns = set(columns)
    if any(set(row) != expected_columns for row in rows):
        raise BenchmarkAnalysisError(f"analysis table {name} has inconsistent columns")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row[key]) for key in columns})
    csv_path = root / f"{name}.csv"
    _atomic_text(csv_path, csv_buffer.getvalue())

    jsonl_path = root / f"{name}.jsonl"
    _atomic_text(
        jsonl_path,
        "".join(
            json.dumps(dict(row), allow_nan=False, ensure_ascii=False, sort_keys=True)
            + "\n"
            for row in rows
        ),
    )
    return {
        "rows": len(rows),
        "columns": columns,
        "grain": _table_grain(name),
        "csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "jsonl": {"path": str(jsonl_path), "sha256": sha256_file(jsonl_path)},
    }


def _table_grain(name: str) -> str:
    grains = {
        "benchmark_chance_levels": "one row per benchmark family",
        "common_english_slice": "one row per frozen benchmark-language intersection",
        "model_scores": "one row per checkpoint readout",
        "trajectory_dynamics": "one row per independently trained trajectory",
        "threshold_crossings": (
            "one row per trajectory, score scope, and post-hoc threshold"
        ),
        "final_endpoints": "one row per independently trained trajectory",
        "canonical_final_grid": "one row per canonical seed-1234 design cell",
        "b_seed_cell_summary": "one row per replicated B E/K/token cell",
        "b_seed_contrasts_seedwise": (
            "one row per replicated B matched contrast and training seed"
        ),
        "b_seed_contrast_summary": (
            "one row per replicated B matched contrast summarized over three seeds"
        ),
    }
    try:
        return grains[name]
    except KeyError as exc:
        raise BenchmarkAnalysisError(f"missing declared grain for table {name}") from exc



def _load_sealed_inputs(
    profile: CampaignProfile,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    if profile.payload.get("campaign_id") != EXPECTED_CAMPAIGN_ID:
        raise BenchmarkAnalysisError(
            f"analysis supports only {EXPECTED_CAMPAIGN_ID!r}"
        )
    output_root = profile.output_root
    report_path = output_root / "comparison_report.json"
    report = _load_object(report_path, label="benchmark comparison report")
    if (
        report.get("schema") != "stage1_jointmoe_benchmark_report/v1"
        or report.get("status") != "complete"
        or report.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or report.get("profile", {}).get("sha256") != profile.source_sha256
    ):
        raise BenchmarkAnalysisError("benchmark comparison report identity drift")
    _verify_payload_hash(report, field="report_sha256", label="comparison report")

    table_reference = report.get("analysis_tables")
    if not isinstance(table_reference, dict):
        raise BenchmarkAnalysisError("comparison report lacks analysis-table reference")
    manifest_path = _resolved_reference(
        {
            "path": table_reference.get("manifest"),
            "sha256": table_reference.get("sha256"),
        },
        expected_path=output_root / "analysis_tables" / "manifest.json",
        label="analysis-table manifest",
    )
    table_manifest = _load_object(manifest_path, label="analysis-table manifest")
    if (
        table_manifest.get("schema")
        != "stage1_jointmoe_benchmark_table_manifest/v1"
        or table_manifest.get("status") != "complete"
        or table_manifest.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or table_manifest.get("profile", {}).get("sha256") != profile.source_sha256
    ):
        raise BenchmarkAnalysisError("analysis-table manifest identity drift")
    manifest_payload_hash = _verify_payload_hash(
        table_manifest,
        field="manifest_sha256",
        label="analysis-table manifest",
    )
    if table_reference.get("manifest_sha256") != manifest_payload_hash:
        raise BenchmarkAnalysisError("report/table-manifest payload hash drift")

    table_specs = table_manifest.get("tables")
    if not isinstance(table_specs, dict) or set(table_specs) != set(EXPECTED_TABLE_ROWS):
        raise BenchmarkAnalysisError("canonical analysis-table inventory drift")
    tables: dict[str, list[dict[str, Any]]] = {}
    for name, expected_rows in EXPECTED_TABLE_ROWS.items():
        spec = table_specs[name]
        if not isinstance(spec, dict) or spec.get("rows") != expected_rows:
            raise BenchmarkAnalysisError(f"{name} declared row-count drift")
        columns = spec.get("columns")
        if not isinstance(columns, list) or not columns:
            raise BenchmarkAnalysisError(f"{name} lacks a column contract")
        jsonl_path = _resolved_reference(
            spec.get("jsonl", {}),
            expected_path=output_root / "analysis_tables" / f"{name}.jsonl",
            label=f"{name} JSONL",
        )
        csv_path = _resolved_reference(
            spec.get("csv", {}),
            expected_path=output_root / "analysis_tables" / f"{name}.csv",
            label=f"{name} CSV",
        )
        rows = _read_jsonl(jsonl_path, columns=columns, label=name)
        if len(rows) != expected_rows:
            raise BenchmarkAnalysisError(
                f"{name} JSONL row-count drift: expected {expected_rows}, got {len(rows)}"
            )
        _validate_csv(csv_path, columns=columns, rows=expected_rows, label=name)
        tables[name] = rows

    request_manifests: dict[str, dict[str, Any]] = {}
    for request_set in ("english", "multilingual"):
        path = profile.request_manifest(request_set)
        payload = _load_object(path, label=f"{request_set} frozen requests")
        if (
            payload.get("schema") != "stage1_lm_eval_frozen_requests/v1"
            or payload.get("profile", {}).get("sha256") != profile.source_sha256
        ):
            raise BenchmarkAnalysisError(f"{request_set} request-manifest identity drift")
        _verify_payload_hash(
            payload,
            field="manifest_sha256",
            label=f"{request_set} frozen requests",
        )
        request_manifests[request_set] = payload

    status_path = output_root / "campaign_status.json"
    status = _load_object(status_path, label="campaign status")
    _verify_payload_hash(status, field="status_sha256", label="campaign status")
    if (
        status.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or status.get("profile", {}).get("sha256") != profile.source_sha256
        or status.get("counts")
        != {"expected": 508, "complete": 508, "partial": 0, "pending": 0}
        or status.get("ready_for_report") is not True
    ):
        raise BenchmarkAnalysisError("campaign status is not a complete 508-job cohort")

    return report, table_manifest, tables, request_manifests, status



def _validate_submission_receipt(
    profile: CampaignProfile, model_scores: Sequence[Mapping[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    path = profile.output_root / "slurm" / "submission_all_receipt.json"
    receipt = _load_object(path, label="all-readout submission receipt")
    jobs = receipt.get("jobs")
    if (
        receipt.get("schema") != "stage1_jointmoe_benchmark_submission/v1"
        or receipt.get("cohort") != "all"
        or receipt.get("profile", {}).get("sha256") != profile.source_sha256
        or not isinstance(jobs, dict)
        or len(jobs) != 508
    ):
        raise BenchmarkAnalysisError("all-readout submission receipt identity drift")
    score_job_ids = {
        str(row["model_id"]): str(row["slurm_job_id"]) for row in model_scores
    }
    receipt_job_ids = {
        str(model_id): str(value.get("job_id"))
        for model_id, value in jobs.items()
        if isinstance(value, dict)
    }
    if receipt_job_ids != score_job_ids:
        raise BenchmarkAnalysisError("submission/runtime Slurm job identity drift")
    for label, expected_name in (
        ("jobs_json", "submitted_all_jobs.json"),
        ("jobs_tsv", "submitted_all_jobs.tsv"),
    ):
        reference = receipt.get(label)
        if not isinstance(reference, dict):
            raise BenchmarkAnalysisError(f"submission receipt lacks {label}")
        _resolved_reference(
            reference,
            expected_path=profile.output_root / "slurm" / expected_name,
            label=f"submission {label}",
        )
    return path, receipt


def _artifact_reference(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_generated_bundle(manifest_path: Path | str) -> dict[str, Any]:
    """Fail closed if any generated bundle artifact no longer matches its manifest."""

    path = Path(manifest_path)
    payload = _load_object(path, label="scientific analysis manifest")
    if payload.get("schema") != ANALYSIS_BUNDLE_SCHEMA or payload.get("status") != "complete":
        raise BenchmarkAnalysisError("scientific analysis manifest identity drift")
    _verify_payload_hash(payload, field="manifest_sha256", label="analysis bundle")
    output_root_value = payload.get("output_root")
    if not isinstance(output_root_value, str):
        raise BenchmarkAnalysisError("analysis bundle lacks output_root")
    output_root = Path(output_root_value).resolve()
    if path.resolve() != output_root / "manifest.json":
        raise BenchmarkAnalysisError("analysis bundle manifest/output_root drift")
    implementations = payload.get("implementations")
    if not isinstance(implementations, dict) or set(implementations) != set(
        ANALYSIS_IMPLEMENTATION_FILES
    ):
        raise BenchmarkAnalysisError("analysis implementation inventory drift")
    for name, expected_path in ANALYSIS_IMPLEMENTATION_FILES.items():
        reference = implementations[name]
        if not isinstance(reference, dict):
            raise BenchmarkAnalysisError(f"invalid analysis implementation {name}")
        _resolved_reference(
            reference,
            expected_path=expected_path,
            label=f"analysis implementation {name}",
        )
    artifacts = payload.get("artifacts")
    tables = payload.get("tables")
    if not isinstance(artifacts, dict) or not isinstance(tables, dict):
        raise BenchmarkAnalysisError("analysis bundle lacks artifact inventory")
    expected_artifacts = {
        "readme": "README.md",
        "integrity_summary": "integrity_summary.json",
        "initial_findings": "initial_findings.json",
        "initial_analysis": "initial_analysis.md",
    }
    if set(artifacts) != set(expected_artifacts):
        raise BenchmarkAnalysisError("generated artifact inventory drift")
    for label, reference in artifacts.items():
        if not isinstance(reference, dict):
            raise BenchmarkAnalysisError(f"invalid generated artifact reference: {label}")
        _resolved_reference(
            reference,
            expected_path=output_root / expected_artifacts[label],
            label=label,
        )
    expected_tables = {
        "benchmark_chance_levels",
        "common_english_slice",
        "model_scores",
        "trajectory_dynamics",
        "threshold_crossings",
        "final_endpoints",
        "canonical_final_grid",
        "b_seed_cell_summary",
        "b_seed_contrasts_seedwise",
        "b_seed_contrast_summary",
    }
    if set(tables) != expected_tables:
        raise BenchmarkAnalysisError("generated table inventory drift")
    for name, spec in tables.items():
        if not isinstance(spec, dict):
            raise BenchmarkAnalysisError(f"invalid generated table spec: {name}")
        columns = spec.get("columns")
        row_count = spec.get("rows")
        if not isinstance(columns, list) or not isinstance(row_count, int):
            raise BenchmarkAnalysisError(f"invalid generated table contract: {name}")
        jsonl = spec.get("jsonl", {})
        csv_ref = spec.get("csv", {})
        jsonl_path = _resolved_reference(
            jsonl,
            expected_path=output_root / "tables" / f"{name}.jsonl",
            label=f"{name} JSONL",
        )
        csv_path = _resolved_reference(
            csv_ref,
            expected_path=output_root / "tables" / f"{name}.csv",
            label=f"{name} CSV",
        )
        rows = _read_jsonl(jsonl_path, columns=columns, label=name)
        if len(rows) != row_count:
            raise BenchmarkAnalysisError(f"generated {name} row-count drift")
        _validate_csv(csv_path, columns=columns, rows=row_count, label=name)
    return payload




__all__ = [
    "ANALYSIS_BUNDLE_SCHEMA",
    "ANALYSIS_IMPLEMENTATION_FILES",
    "COMMON_ENGLISH_SLICE",
    "EXPECTED_CAMPAIGN_ID",
    "EXPECTED_FINAL_READOUTS",
    "EXPECTED_LADDERS",
    "EXPECTED_SOURCE_COHORTS",
    "EXPECTED_TRAJECTORIES",
    "FINDINGS_SCHEMA",
    "FLOAT_TOLERANCE",
    "INTEGRITY_SCHEMA",
    "POST_HOC_THRESHOLDS",
    "REPLICATED_B_EXPERTS",
    "REPLICATED_B_LANGUAGES",
    "REPLICATED_B_SEEDS",
    "REPLICATED_B_TOKENS",
    "BenchmarkAnalysisError",
    "validate_generated_bundle",
]
