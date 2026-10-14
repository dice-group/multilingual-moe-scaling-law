"""Build sealed Stage-1 A--D readout and validation-analysis tables.

The Stage-1 result panel has two source shapes:

* A/B readouts are enumerated by one completed ``final_summary.json``.
* C/D readouts are enumerated by a cross-root ledger because validation
  recovery jobs may have written equivalent results outside the training root.

This module normalizes those sources without choosing a new scientific result.
It verifies the result/report/language-artifact seals, retains every C/D
candidate, and gives semantically equivalent duplicate evaluations a
path-independent sufficient-statistics identity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CANONICAL_READOUT_COUNTS: dict[str, int] = {
    "A": 80,
    "B": 96,
    "C": 112,
    "D": 112,
}

_READOUT_ID = re.compile(
    r"^prod_eval_([abcd])_e(1|4|16|32)_k(001|012|050|200)_d([1-9][0-9]*)$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LANGUAGE_STAT_FIELDS = (
    "schema",
    "language",
    "role",
    "required",
    "token_count",
    "nll_sum",
    "raw_nll",
    "adjusted_nll_sum",
    "atlas_tao_unigram_adjusted_loss",
    "unigram_log_probability_sum",
    "unseen_target_token_count",
    "input_sha256",
    "target_token_ids_sha256",
    "unigram_counts_sha256",
    "vocab_size",
    "special_token_ids",
)
_ARTIFACT_STAT_FIELDS = tuple(
    field for field in _LANGUAGE_STAT_FIELDS if field not in {"role", "required"}
)


class Stage1ValidationError(RuntimeError):
    """Raised when a Stage-1 validation source is incomplete or inconsistent."""


@dataclass(frozen=True)
class Stage1ValidationTables:
    """Columnar-ready records for the central Stage-1 study release."""

    readouts: tuple[dict[str, Any], ...]
    validation_aggregates: tuple[dict[str, Any], ...]
    validation_language: tuple[dict[str, Any], ...]
    validation_candidates: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    provenance: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _ValidatedEvaluation:
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_file_sha256: str
    report_path: Path
    report: dict[str, Any]
    report_file_sha256: str
    semantic_sha256: str
    spec_path: Path
    spec_file_sha256: str
    language_artifacts: dict[str, tuple[Path, str]]


def _canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage1ValidationError(f"Payload is not canonical JSON: {exc}") from exc


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Stage1ValidationError(f"Cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ValidationError(f"Cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage1ValidationError(f"Expected a JSON object: {path}")
    return payload


def _resolve_path(raw: Any, *, relative_to: Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage1ValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise Stage1ValidationError(f"{label} is not finite: {value!r}")
    return result


def _optional_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


def _readout_parts(readout_id: str) -> tuple[str, int, int, int]:
    match = _READOUT_ID.fullmatch(readout_id)
    if match is None:
        raise Stage1ValidationError(f"Non-canonical Stage-1 readout ID: {readout_id}")
    ladder, experts, language_count, nominal_tokens = match.groups()
    return (
        ladder.upper(),
        int(experts),
        int(language_count),
        int(nominal_tokens),
    )


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    observed = str(payload.get(field, ""))
    if not _SHA256.fullmatch(observed):
        raise Stage1ValidationError(f"{label} has no valid {field}")
    hash_input = dict(payload)
    hash_input.pop(field, None)
    expected = _payload_sha256(hash_input)
    if observed != expected:
        raise Stage1ValidationError(
            f"{label} embedded {field} mismatch: {observed} != {expected}"
        )
    return observed


def _verified_reference(
    reference: Any,
    *,
    owner_path: Path,
    label: str,
    required_parent: Path | None = None,
) -> tuple[Path, str]:
    if not isinstance(reference, Mapping):
        raise Stage1ValidationError(f"{label} is not a hash reference")
    path = _resolve_path(reference.get("path", ""), relative_to=owner_path.parent)
    declared = str(reference.get("sha256", ""))
    if not path.is_file() or not _SHA256.fullmatch(declared):
        raise Stage1ValidationError(f"{label} reference is incomplete: {reference}")
    if required_parent is not None:
        required_parent = required_parent.resolve()
        if path != required_parent and required_parent not in path.parents:
            raise Stage1ValidationError(f"{label} escaped {required_parent}: {path}")
    observed = _file_sha256(path)
    if observed != declared:
        raise Stage1ValidationError(
            f"{label} file hash mismatch: {observed} != {declared}"
        )
    return path, observed


def _language_semantic_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _LANGUAGE_STAT_FIELDS if field not in row]
    if missing:
        raise Stage1ValidationError(
            f"Language {row.get('language')!r} lacks fields: {missing}"
        )
    return {field: row[field] for field in _LANGUAGE_STAT_FIELDS}


def semantic_sufficient_statistics_sha256(report: Mapping[str, Any]) -> str:
    """Return a path- and runtime-independent identity for validation evidence.

    The identity includes the metric definition, coverage/roles, every
    per-language sufficient statistic, and every reported aggregate. It omits
    run IDs, source paths, timing, and routing telemetry, which may legitimately
    differ between recovery evaluations of the same checkpoint.
    """

    languages = report.get("languages")
    if not isinstance(languages, Sequence) or isinstance(languages, (str, bytes)):
        raise Stage1ValidationError("Validation report has no language rows")
    rows: list[dict[str, Any]] = []
    for raw in languages:
        if not isinstance(raw, Mapping):
            raise Stage1ValidationError("Validation report language row is malformed")
        rows.append(_language_semantic_payload(raw))
    rows.sort(key=lambda item: str(item["language"]))
    return _payload_sha256(
        {
            "schema": "stage1_validation_sufficient_statistics/v1",
            "metric_spec": report.get("metric_spec"),
            "coverage": report.get("coverage"),
            "aggregates": report.get("aggregates"),
            "languages": rows,
        }
    )


def _verify_language_numbers(row: Mapping[str, Any], *, label: str) -> None:
    try:
        token_count = int(row["token_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage1ValidationError(f"{label} has invalid token_count") from exc
    if token_count <= 0:
        raise Stage1ValidationError(f"{label} has non-positive token_count")
    nll_sum = _finite_float(row.get("nll_sum"), label=f"{label}.nll_sum")
    raw_nll = _finite_float(row.get("raw_nll"), label=f"{label}.raw_nll")
    adjusted_sum = _finite_float(
        row.get("adjusted_nll_sum"), label=f"{label}.adjusted_nll_sum"
    )
    adjusted_loss = _finite_float(
        row.get("atlas_tao_unigram_adjusted_loss"),
        label=f"{label}.atlas_tao_unigram_adjusted_loss",
    )
    unigram_sum = _finite_float(
        row.get("unigram_log_probability_sum"),
        label=f"{label}.unigram_log_probability_sum",
    )
    if not math.isclose(raw_nll, nll_sum / token_count, rel_tol=1e-12, abs_tol=1e-12):
        raise Stage1ValidationError(f"{label} raw-NLL denominator mismatch")
    if not math.isclose(
        adjusted_loss,
        adjusted_sum / token_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise Stage1ValidationError(f"{label} adjusted-loss denominator mismatch")
    if not math.isclose(
        adjusted_sum,
        nll_sum + unigram_sum,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise Stage1ValidationError(f"{label} adjusted sufficient-statistic mismatch")


def _verify_role_aggregate(
    aggregate: Any,
    *,
    role: str,
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if not isinstance(aggregate, Mapping):
        raise Stage1ValidationError(f"{label} aggregate is missing")
    role_rows = [row for row in rows if row.get("role") == role]
    languages = sorted(str(row["language"]) for row in role_rows)
    if aggregate.get("languages") != languages:
        raise Stage1ValidationError(f"{label} language membership mismatch")
    if int(aggregate.get("language_count", -1)) != len(role_rows):
        raise Stage1ValidationError(f"{label} language count mismatch")
    token_count = sum(int(row["token_count"]) for row in role_rows)
    if int(aggregate.get("token_count", -1)) != token_count:
        raise Stage1ValidationError(f"{label} token count mismatch")
    if not role_rows:
        if aggregate.get("status") != "empty":
            raise Stage1ValidationError(f"{label} empty aggregate status mismatch")
        for key in (
            "equal_language_atlas_tao_adjusted_loss",
            "equal_language_raw_nll",
            "token_micro_atlas_tao_adjusted_loss",
            "token_micro_raw_nll",
        ):
            if aggregate.get(key) is not None:
                raise Stage1ValidationError(f"{label}.{key} should be null")
        return
    if aggregate.get("status") != "complete":
        raise Stage1ValidationError(f"{label} complete aggregate status mismatch")
    expected = {
        "equal_language_atlas_tao_adjusted_loss": sum(
            float(row["atlas_tao_unigram_adjusted_loss"]) for row in role_rows
        )
        / len(role_rows),
        "equal_language_raw_nll": sum(float(row["raw_nll"]) for row in role_rows)
        / len(role_rows),
        "token_micro_atlas_tao_adjusted_loss": sum(
            float(row["adjusted_nll_sum"]) for row in role_rows
        )
        / token_count,
        "token_micro_raw_nll": sum(float(row["nll_sum"]) for row in role_rows)
        / token_count,
    }
    for key, value in expected.items():
        observed = _finite_float(aggregate.get(key), label=f"{label}.{key}")
        if not math.isclose(observed, value, rel_tol=1e-12, abs_tol=1e-12):
            raise Stage1ValidationError(f"{label}.{key} reconstruction mismatch")


def _validate_evaluation(manifest_path: Path) -> _ValidatedEvaluation:
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != "multilingual_per_language_checkpoint_eval_result/v1"
        or manifest.get("status") != "complete"
    ):
        raise Stage1ValidationError(
            f"Evaluation manifest is not complete/sealed: {manifest_path}"
        )
    _verify_embedded_hash(
        manifest,
        field="result_sha256",
        label=f"evaluation manifest {manifest_path}",
    )
    manifest_file_sha256 = _file_sha256(manifest_path)

    marker = manifest_path.parent / ".validation_complete"
    if not marker.is_file():
        raise Stage1ValidationError(f"Validation completion marker missing: {marker}")

    spec_path, spec_sha256 = _verified_reference(
        manifest.get("spec"),
        owner_path=manifest_path,
        label=f"{manifest_path} spec",
        required_parent=manifest_path.parent,
    )
    spec = _load_json(spec_path)
    if spec.get("schema") != "multilingual_per_language_checkpoint_eval/v1":
        raise Stage1ValidationError(f"Unexpected evaluation spec schema: {spec_path}")

    report_path, report_file_sha256 = _verified_reference(
        manifest.get("validation_report"),
        owner_path=manifest_path,
        label=f"{manifest_path} validation report",
        required_parent=manifest_path.parent,
    )
    report = _load_json(report_path)
    if (
        report.get("schema") != "multilingual_validation_report/v1"
        or report.get("status") != "complete"
    ):
        raise Stage1ValidationError(f"Validation report is incomplete: {report_path}")
    report_payload_sha256 = _verify_embedded_hash(
        report,
        field="report_sha256",
        label=f"validation report {report_path}",
    )
    descriptor_report_sha256 = str(
        manifest["validation_report"].get("report_sha256", "")
    )
    if descriptor_report_sha256 != report_payload_sha256:
        raise Stage1ValidationError(
            f"Validation report payload identity mismatch: {report_path}"
        )

    coverage = report.get("coverage")
    manifest_coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping) or not isinstance(manifest_coverage, Mapping):
        raise Stage1ValidationError(f"Validation coverage is missing: {report_path}")
    required = list(coverage.get("required_languages", []))
    observed = list(coverage.get("observed_languages", []))
    trained = list(coverage.get("trained_languages", []))
    zero_shot = list(coverage.get("zero_shot_languages", []))
    if (
        not required
        or observed != sorted(required)
        or coverage.get("missing_required_languages") != []
        or sorted(trained + zero_shot) != sorted(required)
        or set(trained).intersection(zero_shot)
        or manifest_coverage.get("required_languages") != required
        or manifest_coverage.get("evaluated_languages") != observed
        or manifest_coverage.get("trained_languages") != trained
        or manifest_coverage.get("missing_required_languages") != []
    ):
        raise Stage1ValidationError(f"Validation coverage mismatch: {report_path}")

    raw_rows = report.get("languages")
    descriptors = manifest.get("languages")
    if not isinstance(raw_rows, list) or not isinstance(descriptors, Mapping):
        raise Stage1ValidationError(
            f"Validation language data is missing: {report_path}"
        )
    rows: list[dict[str, Any]] = []
    by_language: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise Stage1ValidationError(f"Malformed language row: {report_path}")
        language = str(raw.get("language", ""))
        if not language or language in by_language:
            raise Stage1ValidationError(
                f"Missing/duplicate language row {language!r}: {report_path}"
            )
        _language_semantic_payload(raw)
        _verify_language_numbers(raw, label=f"{report_path}/{language}")
        expected_role = "trained" if language in trained else "zero_shot"
        if (
            raw.get("role") != expected_role
            or raw.get("required") is not True
            or raw.get("schema") != "multilingual_validation_language_loss/v1"
        ):
            raise Stage1ValidationError(
                f"Language role/schema mismatch for {language}: {report_path}"
            )
        rows.append(raw)
        by_language[language] = raw
    if set(by_language) != set(required) or set(descriptors) != set(required):
        raise Stage1ValidationError(f"Language coverage changed: {manifest_path}")

    language_artifacts: dict[str, tuple[Path, str]] = {}
    for language in sorted(required):
        descriptor = descriptors[language]
        language_path, language_sha256 = _verified_reference(
            descriptor,
            owner_path=manifest_path,
            label=f"{manifest_path}/{language}",
            required_parent=manifest_path.parent / "languages",
        )
        artifact = _load_json(language_path)
        report_row = by_language[language]
        for field in _ARTIFACT_STAT_FIELDS:
            if artifact.get(field) != report_row.get(field):
                raise Stage1ValidationError(
                    f"{language} artifact/report field mismatch: {field}"
                )
        if int(descriptor.get("token_count", -1)) != int(
            report_row["token_count"]
        ) or not math.isclose(
            _finite_float(
                descriptor.get("nll_sum"),
                label=f"{manifest_path}/{language}.descriptor_nll_sum",
            ),
            float(report_row["nll_sum"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise Stage1ValidationError(
                f"{language} manifest/report sufficient-statistic mismatch"
            )
        language_artifacts[language] = (language_path, language_sha256)

    aggregates = report.get("aggregates")
    if not isinstance(aggregates, Mapping):
        raise Stage1ValidationError(f"Validation aggregates are missing: {report_path}")
    _verify_role_aggregate(
        aggregates.get("trained_languages"),
        role="trained",
        rows=rows,
        label=f"{report_path}.trained_languages",
    )
    _verify_role_aggregate(
        aggregates.get("zero_shot_languages"),
        role="zero_shot",
        rows=rows,
        label=f"{report_path}.zero_shot_languages",
    )
    trained_aggregate = aggregates["trained_languages"]
    for name, key in (
        ("primary", "equal_language_atlas_tao_adjusted_loss"),
        ("secondary", "equal_language_raw_nll"),
    ):
        descriptor = aggregates.get(name)
        if not isinstance(descriptor, Mapping):
            raise Stage1ValidationError(f"{report_path} lacks {name} aggregate")
        value = _finite_float(
            descriptor.get("value"), label=f"{report_path}.{name}.value"
        )
        expected = _finite_float(
            trained_aggregate.get(key),
            label=f"{report_path}.trained_languages.{key}",
        )
        if (
            value != expected
            or descriptor.get("languages") != trained_aggregate.get("languages")
            or int(descriptor.get("language_count", -1)) != len(trained)
        ):
            raise Stage1ValidationError(f"{report_path} {name} aggregate mismatch")

    return _ValidatedEvaluation(
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        report_path=report_path,
        report=report,
        report_file_sha256=report_file_sha256,
        semantic_sha256=semantic_sufficient_statistics_sha256(report),
        spec_path=spec_path,
        spec_file_sha256=spec_sha256,
        language_artifacts=language_artifacts,
    )


def _training_metadata(
    evaluation: _ValidatedEvaluation,
    *,
    expected_training_job_id: str | None,
) -> dict[str, Any]:
    checkpoint = evaluation.manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise Stage1ValidationError(
            f"Evaluation checkpoint is missing: {evaluation.manifest_path}"
        )
    checkpoint_root = _resolve_path(
        checkpoint.get("root", ""), relative_to=evaluation.manifest_path.parent
    )
    try:
        checkpoint_step = int(checkpoint["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage1ValidationError(
            f"Checkpoint step is invalid: {evaluation.manifest_path}"
        ) from exc
    if checkpoint_step <= 0 or len(checkpoint_root.parents) < 3:
        raise Stage1ValidationError(
            f"Checkpoint binding is invalid: {evaluation.manifest_path}"
        )
    run_root = checkpoint_root.parents[2]
    training_manifest_path = run_root / "training_run_manifest.json"
    training = _load_json(training_manifest_path)
    if training.get("status") != "complete":
        raise Stage1ValidationError(
            f"Training run is not complete: {training_manifest_path}"
        )
    metadata = training.get("metadata")
    if not isinstance(metadata, Mapping):
        raise Stage1ValidationError(
            f"Training metadata is missing: {training_manifest_path}"
        )
    segments = metadata.get("segments")
    if not isinstance(segments, list):
        raise Stage1ValidationError(
            f"Training segments are missing: {training_manifest_path}"
        )
    matches = [
        segment
        for segment in segments
        if isinstance(segment, Mapping)
        and int(segment.get("cumulative_end_step", -1)) == checkpoint_step
    ]
    if len(matches) != 1:
        raise Stage1ValidationError(
            f"Cannot bind checkpoint step {checkpoint_step} to one segment: "
            f"{training_manifest_path}"
        )
    segment = matches[0]
    parts = [part for part in checkpoint_root.parts if part.startswith("prod_branch_")]
    derived_training_job_id = parts[-1] if parts else None
    training_job_id = expected_training_job_id or derived_training_job_id
    if not training_job_id:
        raise Stage1ValidationError(
            f"Cannot derive training job ID from {checkpoint_root}"
        )
    if (
        expected_training_job_id
        and derived_training_job_id
        and expected_training_job_id != derived_training_job_id
    ):
        raise Stage1ValidationError(
            f"Training job mismatch: {expected_training_job_id} != "
            f"{derived_training_job_id}"
        )
    return {
        "training_job_id": training_job_id,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_iteration_path": str(
            _resolve_path(
                checkpoint.get("iteration_path", checkpoint_root),
                relative_to=evaluation.manifest_path.parent,
            )
        ),
        "checkpoint_step": checkpoint_step,
        "actual_tokens": int(segment["actual_end_tokens"]),
        "expected_consumed_samples": int(segment["expected_consumed_samples"]),
        "training_manifest_path": str(training_manifest_path.resolve()),
        "training_manifest_sha256": _file_sha256(training_manifest_path),
    }


def _artifact_record(
    *,
    readout_id: str | None,
    artifact_type: str,
    path: Path,
    sha256: str,
    source_path: Path,
    source_sha256: str,
    selected: bool | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    identity = {
        "readout_id": readout_id,
        "artifact_type": artifact_type,
        "path": str(path.resolve()),
        "candidate_path": (
            None if candidate_path is None else str(candidate_path.resolve())
        ),
    }
    return {
        "artifact_id": _payload_sha256(identity),
        **identity,
        "sha256": sha256,
        "selected": selected,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "provenance_class": "observed",
    }


def _provenance_record(
    *,
    readout_id: str | None,
    relation: str,
    subject_path: Path,
    object_path: Path,
    source_path: Path,
    source_sha256: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "readout_id": readout_id,
        "relation": relation,
        "subject_path": str(subject_path.resolve()),
        "object_path": str(object_path.resolve()),
    }
    return {
        "provenance_id": _payload_sha256(identity),
        **identity,
        "details_json": json.dumps(
            dict(details or {}),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "provenance_class": "observed",
    }


def _json_cell(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_aggregates(
    *,
    readout: Mapping[str, Any],
    evaluation: _ValidatedEvaluation,
) -> list[dict[str, Any]]:
    report = evaluation.report
    raw_aggregates = report["aggregates"]
    rows: list[dict[str, Any]] = []
    common = {
        "readout_id": readout["readout_id"],
        "trajectory_id": readout["trajectory_id"],
        "ladder": readout["ladder"],
        "experts": readout["experts"],
        "language_count": readout["language_count"],
        "nominal_tokens": readout["nominal_tokens"],
        "actual_tokens": readout["actual_tokens"],
        "report_path": str(evaluation.report_path),
        "report_sha256": evaluation.report_file_sha256,
        "semantic_sufficient_statistics_sha256": evaluation.semantic_sha256,
        "source_path": str(evaluation.report_path),
        "source_sha256": evaluation.report_file_sha256,
        "provenance_class": "observed",
    }

    for cohort in ("trained_languages", "zero_shot_languages"):
        aggregate = raw_aggregates.get(cohort)
        if not isinstance(aggregate, Mapping):
            raise Stage1ValidationError(
                f"{evaluation.report_path} lacks {cohort} aggregate"
            )
        rows.append(
            {
                **common,
                "aggregate_kind": "metric_role",
                "cohort": cohort,
                "status": aggregate.get("status"),
                "metric_name": None,
                "metric_value": None,
                "aggregate_language_count": int(aggregate.get("language_count", 0)),
                "aggregate_token_count": int(aggregate.get("token_count", 0)),
                "equal_language_adjusted_loss": _optional_float(
                    aggregate.get("equal_language_atlas_tao_adjusted_loss"),
                    label=f"{evaluation.report_path}.{cohort}.adjusted",
                ),
                "equal_language_raw_nll": _optional_float(
                    aggregate.get("equal_language_raw_nll"),
                    label=f"{evaluation.report_path}.{cohort}.raw",
                ),
                "token_micro_adjusted_loss": _optional_float(
                    aggregate.get("token_micro_atlas_tao_adjusted_loss"),
                    label=f"{evaluation.report_path}.{cohort}.micro_adjusted",
                ),
                "token_micro_raw_nll": _optional_float(
                    aggregate.get("token_micro_raw_nll"),
                    label=f"{evaluation.report_path}.{cohort}.micro_raw",
                ),
                "languages_json": _json_cell(aggregate.get("languages", [])),
                "membership_sha256": None,
                "missing_languages_json": "[]",
                "reason": None,
            }
        )

    fixed = raw_aggregates.get("fixed_common_cohorts", {})
    if not isinstance(fixed, Mapping):
        raise Stage1ValidationError(
            f"{evaluation.report_path} fixed cohorts are malformed"
        )
    for cohort, aggregate in sorted(fixed.items()):
        if not isinstance(aggregate, Mapping):
            raise Stage1ValidationError(
                f"{evaluation.report_path} cohort {cohort} is malformed"
            )
        rows.append(
            {
                **common,
                "aggregate_kind": "fixed_common_cohort",
                "cohort": str(cohort),
                "status": aggregate.get("status"),
                "metric_name": None,
                "metric_value": None,
                "aggregate_language_count": (
                    None
                    if aggregate.get("language_count") is None
                    else int(aggregate["language_count"])
                ),
                "aggregate_token_count": (
                    None
                    if aggregate.get("token_count") is None
                    else int(aggregate["token_count"])
                ),
                "equal_language_adjusted_loss": _optional_float(
                    aggregate.get("equal_language_atlas_tao_adjusted_loss"),
                    label=f"{evaluation.report_path}.{cohort}.adjusted",
                ),
                "equal_language_raw_nll": _optional_float(
                    aggregate.get("equal_language_raw_nll"),
                    label=f"{evaluation.report_path}.{cohort}.raw",
                ),
                "token_micro_adjusted_loss": _optional_float(
                    aggregate.get("token_micro_atlas_tao_adjusted_loss"),
                    label=f"{evaluation.report_path}.{cohort}.micro_adjusted",
                ),
                "token_micro_raw_nll": _optional_float(
                    aggregate.get("token_micro_raw_nll"),
                    label=f"{evaluation.report_path}.{cohort}.micro_raw",
                ),
                "languages_json": _json_cell(aggregate.get("languages", [])),
                "membership_sha256": aggregate.get("membership_sha256"),
                "missing_languages_json": _json_cell(
                    aggregate.get("missing_trained_languages", [])
                ),
                "reason": aggregate.get("reason"),
            }
        )

    for name in ("primary", "secondary"):
        aggregate = raw_aggregates[name]
        rows.append(
            {
                **common,
                "aggregate_kind": "declared_metric",
                "cohort": name,
                "status": "complete",
                "metric_name": aggregate.get("name"),
                "metric_value": _finite_float(
                    aggregate.get("value"),
                    label=f"{evaluation.report_path}.{name}.value",
                ),
                "aggregate_language_count": int(aggregate["language_count"]),
                "aggregate_token_count": None,
                "equal_language_adjusted_loss": None,
                "equal_language_raw_nll": None,
                "token_micro_adjusted_loss": None,
                "token_micro_raw_nll": None,
                "languages_json": _json_cell(aggregate.get("languages", [])),
                "membership_sha256": None,
                "missing_languages_json": "[]",
                "reason": None,
            }
        )
    return rows


def _normalize_languages(
    *,
    readout: Mapping[str, Any],
    evaluation: _ValidatedEvaluation,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    language_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for raw in sorted(
        evaluation.report["languages"], key=lambda item: item["language"]
    ):
        language = str(raw["language"])
        artifact_path, artifact_sha256 = evaluation.language_artifacts[language]
        sufficient_statistics = _language_semantic_payload(raw)
        language_rows.append(
            {
                "readout_id": readout["readout_id"],
                "trajectory_id": readout["trajectory_id"],
                "ladder": readout["ladder"],
                "experts": readout["experts"],
                "language_count": readout["language_count"],
                "nominal_tokens": readout["nominal_tokens"],
                "actual_tokens": readout["actual_tokens"],
                "language": language,
                "metric_role": raw["role"],
                "required": bool(raw["required"]),
                "token_count": int(raw["token_count"]),
                "nll_sum": float(raw["nll_sum"]),
                "raw_nll": float(raw["raw_nll"]),
                "adjusted_nll_sum": float(raw["adjusted_nll_sum"]),
                "atlas_tao_unigram_adjusted_loss": float(
                    raw["atlas_tao_unigram_adjusted_loss"]
                ),
                "unigram_log_probability_sum": float(
                    raw["unigram_log_probability_sum"]
                ),
                "unseen_target_token_count": int(raw["unseen_target_token_count"]),
                "vocab_size": int(raw["vocab_size"]),
                "special_token_ids_json": _json_cell(raw["special_token_ids"]),
                # The upstream field is a hash of the full language-loss
                # record and therefore includes the model-dependent NLL.
                "input_sha256": raw["input_sha256"],
                "input_sha256_semantics": (
                    "outcome_inclusive_multilingual_validation_language_loss_v1_"
                    "record_hash"
                ),
                "validation_input_identity_sha256": _payload_sha256(
                    {
                        "schema": "multilingual_validation_input_identity/v1",
                        "language": language,
                        "token_count": int(raw["token_count"]),
                        "target_token_ids_sha256": raw["target_token_ids_sha256"],
                        "unigram_counts_sha256": raw["unigram_counts_sha256"],
                        "vocab_size": int(raw["vocab_size"]),
                        "special_token_ids": list(raw["special_token_ids"]),
                    }
                ),
                "target_token_ids_sha256": raw["target_token_ids_sha256"],
                "unigram_counts_sha256": raw["unigram_counts_sha256"],
                "language_sufficient_statistics_sha256": _payload_sha256(
                    sufficient_statistics
                ),
                "readout_sufficient_statistics_sha256": evaluation.semantic_sha256,
                "language_artifact_path": str(artifact_path),
                "language_artifact_sha256": artifact_sha256,
                "report_path": str(evaluation.report_path),
                "report_sha256": evaluation.report_file_sha256,
                "source_path": str(evaluation.report_path),
                "source_sha256": evaluation.report_file_sha256,
                "provenance_class": "observed",
            }
        )
        artifacts.append(
            _artifact_record(
                readout_id=str(readout["readout_id"]),
                artifact_type="validation_language",
                path=artifact_path,
                sha256=artifact_sha256,
                source_path=evaluation.manifest_path,
                source_sha256=evaluation.manifest_file_sha256,
                selected=True,
            )
        )
        provenance.append(
            _provenance_record(
                readout_id=str(readout["readout_id"]),
                relation="validation_report_summarizes_language_artifact",
                subject_path=evaluation.report_path,
                object_path=artifact_path,
                source_path=evaluation.manifest_path,
                source_sha256=evaluation.manifest_file_sha256,
                details={"language": language, "metric_role": raw["role"]},
            )
        )
    return language_rows, artifacts, provenance


def _core_artifacts_and_provenance(
    *,
    readout_id: str,
    evaluation: _ValidatedEvaluation,
    training: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_path = Path(str(training["training_manifest_path"]))
    artifacts = [
        _artifact_record(
            readout_id=readout_id,
            artifact_type="evaluation_manifest",
            path=evaluation.manifest_path,
            sha256=evaluation.manifest_file_sha256,
            source_path=source_path,
            source_sha256=source_sha256,
            selected=True,
        ),
        _artifact_record(
            readout_id=readout_id,
            artifact_type="validation_report",
            path=evaluation.report_path,
            sha256=evaluation.report_file_sha256,
            source_path=evaluation.manifest_path,
            source_sha256=evaluation.manifest_file_sha256,
            selected=True,
        ),
        _artifact_record(
            readout_id=readout_id,
            artifact_type="evaluation_spec",
            path=evaluation.spec_path,
            sha256=evaluation.spec_file_sha256,
            source_path=evaluation.manifest_path,
            source_sha256=evaluation.manifest_file_sha256,
            selected=True,
        ),
        _artifact_record(
            readout_id=readout_id,
            artifact_type="training_manifest",
            path=training_path,
            sha256=str(training["training_manifest_sha256"]),
            source_path=evaluation.manifest_path,
            source_sha256=evaluation.manifest_file_sha256,
            selected=True,
        ),
    ]
    provenance = [
        _provenance_record(
            readout_id=readout_id,
            relation="evaluation_manifest_references_validation_report",
            subject_path=evaluation.manifest_path,
            object_path=evaluation.report_path,
            source_path=evaluation.manifest_path,
            source_sha256=evaluation.manifest_file_sha256,
        ),
        _provenance_record(
            readout_id=readout_id,
            relation="evaluation_manifest_references_evaluation_spec",
            subject_path=evaluation.manifest_path,
            object_path=evaluation.spec_path,
            source_path=evaluation.manifest_path,
            source_sha256=evaluation.manifest_file_sha256,
        ),
        _provenance_record(
            readout_id=readout_id,
            relation="evaluation_uses_training_checkpoint",
            subject_path=evaluation.manifest_path,
            object_path=Path(str(training["checkpoint_root"])),
            source_path=evaluation.manifest_path,
            source_sha256=evaluation.manifest_file_sha256,
            details={
                "checkpoint_step": training["checkpoint_step"],
                "training_job_id": training["training_job_id"],
            },
        ),
    ]
    return artifacts, provenance


def _normalize_readout(
    *,
    readout_id: str,
    evaluation: _ValidatedEvaluation,
    training: Mapping[str, Any],
    source_family: str,
    source_path: Path,
    source_sha256: str,
    selection_reason: str,
    candidate_count: int,
    valid_candidate_count: int,
) -> dict[str, Any]:
    ladder, experts, language_count, nominal_tokens = _readout_parts(readout_id)
    aggregates = evaluation.report["aggregates"]
    primary = aggregates["primary"]
    secondary = aggregates["secondary"]
    return {
        "readout_id": readout_id,
        "trajectory_id": f"{ladder}:E{experts}:K{language_count}",
        "ladder": ladder,
        "experts": experts,
        "language_count": language_count,
        "nominal_tokens": nominal_tokens,
        "actual_tokens": int(training["actual_tokens"]),
        "expected_consumed_samples": int(training["expected_consumed_samples"]),
        "training_job_id": training["training_job_id"],
        "checkpoint_root": training["checkpoint_root"],
        "checkpoint_iteration_path": training["checkpoint_iteration_path"],
        "checkpoint_step": int(training["checkpoint_step"]),
        "training_manifest_path": training["training_manifest_path"],
        "training_manifest_sha256": training["training_manifest_sha256"],
        "evaluation_manifest_path": str(evaluation.manifest_path),
        "evaluation_manifest_sha256": evaluation.manifest_file_sha256,
        "evaluation_result_sha256": evaluation.manifest["result_sha256"],
        "validation_report_path": str(evaluation.report_path),
        "validation_report_sha256": evaluation.report_file_sha256,
        "validation_report_payload_sha256": evaluation.report["report_sha256"],
        "semantic_sufficient_statistics_sha256": evaluation.semantic_sha256,
        "primary_metric_name": primary["name"],
        "primary_metric_value": float(primary["value"]),
        "secondary_metric_name": secondary["name"],
        "secondary_metric_value": float(secondary["value"]),
        "trained_language_count": len(
            evaluation.report["coverage"]["trained_languages"]
        ),
        "zero_shot_language_count": len(
            evaluation.report["coverage"]["zero_shot_languages"]
        ),
        "required_language_count": len(
            evaluation.report["coverage"]["required_languages"]
        ),
        "selection_reason": selection_reason,
        "candidate_count": candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "source_family": source_family,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "provenance_class": "observed",
    }


def _append_selected_evaluation(
    *,
    readout: dict[str, Any],
    evaluation: _ValidatedEvaluation,
    training: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    readouts: list[dict[str, Any]],
    validation_aggregates: list[dict[str, Any]],
    validation_language: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    readouts.append(readout)
    validation_aggregates.extend(
        _normalize_aggregates(readout=readout, evaluation=evaluation)
    )
    language_rows, language_artifacts, language_provenance = _normalize_languages(
        readout=readout, evaluation=evaluation
    )
    validation_language.extend(language_rows)
    artifacts.extend(language_artifacts)
    provenance.extend(language_provenance)
    core_artifacts, core_provenance = _core_artifacts_and_provenance(
        readout_id=str(readout["readout_id"]),
        evaluation=evaluation,
        training=training,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    artifacts.extend(core_artifacts)
    provenance.extend(core_provenance)


def build_direct_cd_ledger(
    *,
    cd_result_roots: Mapping[str, Path],
) -> dict[str, Any]:
    """Create a sealed one-candidate C/D ledger from completed final summaries.

    The clean JointMoE rerun has one canonical evaluation location per C/D
    readout, unlike the historical quota-recovery campaign.  This adapter keeps
    the established ledger-backed normalization path while independently
    checking every summary row, evaluation seal, metric, and training cutoff.
    """

    if set(cd_result_roots) != {"C", "D"}:
        raise Stage1ValidationError("direct C/D roots must contain exactly C and D")
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_ladder in ("C", "D"):
        result_root = Path(cd_result_roots[expected_ladder]).resolve()
        summary_path = result_root / "control" / "final_summary.json"
        summary = _load_json(summary_path)
        rows = summary.get("results")
        if (
            summary.get("status") != "complete"
            or summary.get("errors") != []
            or not isinstance(rows, list)
            or int(summary.get("observed_results", -1))
            != int(summary.get("expected_results", -2))
            or len(rows) != CANONICAL_READOUT_COUNTS[expected_ladder]
        ):
            raise Stage1ValidationError(
                f"{expected_ladder} final summary is incomplete: {summary_path}"
            )
        summary_sha256 = _file_sha256(summary_path)
        summaries.append(
            {
                "ladder": expected_ladder,
                "path": str(summary_path),
                "sha256": summary_sha256,
                "results": len(rows),
            }
        )
        for row in sorted(rows, key=lambda item: str(item.get("job_id"))):
            if not isinstance(row, Mapping):
                raise Stage1ValidationError(
                    f"Malformed {expected_ladder} summary row: {row!r}"
                )
            readout_id = str(row.get("job_id", ""))
            ladder, experts, language_count, nominal_tokens = _readout_parts(readout_id)
            if ladder != expected_ladder or readout_id in seen:
                raise Stage1ValidationError(
                    f"Unexpected or duplicate direct C/D readout: {readout_id}"
                )
            seen.add(readout_id)
            observed_cell = (
                str(row.get("ladder")),
                int(row.get("experts", -1)),
                int(row.get("K", -1)),
                int(row.get("nominal_D_tokens", -1)),
            )
            if (
                observed_cell != (ladder, experts, language_count, nominal_tokens)
                or int(row.get("D", -1)) != nominal_tokens
                or row.get("purpose") != "production"
            ):
                raise Stage1ValidationError(
                    f"Direct C/D summary axes disagree with {readout_id}"
                )
            manifest_path = result_root / "jobs" / readout_id / "evaluation_manifest.json"
            evaluation = _validate_evaluation(manifest_path)
            if evaluation.manifest_file_sha256 != str(row.get("sha256", "")):
                raise Stage1ValidationError(
                    f"Direct C/D summary evaluation hash mismatch: {readout_id}"
                )
            training_job_id = str(row.get("training_job_id", ""))
            training = _training_metadata(
                evaluation,
                expected_training_job_id=training_job_id or None,
            )
            if (
                int(row.get("actual_end_tokens", -1)) != int(training["actual_tokens"])
                or int(row.get("expected_consumed_samples", -1))
                != int(training["expected_consumed_samples"])
            ):
                raise Stage1ValidationError(
                    f"Direct C/D training cutoff mismatch for {readout_id}"
                )
            primary = float(evaluation.report["aggregates"]["primary"]["value"])
            secondary = float(evaluation.report["aggregates"]["secondary"]["value"])
            if not math.isclose(
                float(row["equal_language_atlas_tao_adjusted_loss"]),
                primary,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(row["raw_token_nll"]),
                secondary,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise Stage1ValidationError(
                    f"Direct C/D summary metric mismatch for {readout_id}"
                )
            records.append(
                {
                    "canonical_id": readout_id,
                    "ladder": ladder,
                    "candidate_count": 1,
                    "valid_candidate_count": 1,
                    "selected_manifest": str(manifest_path.resolve()),
                    "selection_reason": "canonical_direct_final_summary",
                    "source_final_summary": str(summary_path),
                    "source_final_summary_sha256": summary_sha256,
                    "candidates": [
                        {
                            "path": str(manifest_path.resolve()),
                            "valid": True,
                            "error": None,
                            "details": {
                                "manifest_file_sha256": evaluation.manifest_file_sha256,
                                "result_sha256": evaluation.manifest["result_sha256"],
                                "semantic_sufficient_statistics_sha256": evaluation.semantic_sha256,
                            },
                        }
                    ],
                }
            )
    expected = CANONICAL_READOUT_COUNTS["C"] + CANONICAL_READOUT_COUNTS["D"]
    if len(records) != expected:
        raise Stage1ValidationError(
            f"Expected {expected} direct C/D records, got {len(records)}"
        )
    unsigned = {
        "schema": "stage1_cd_cross_root_analysis_ledger/v1",
        "source_mode": "direct_final_summaries",
        "expected_results": expected,
        "selected_results": len(records),
        "source_summaries": summaries,
        "records": sorted(records, key=lambda item: str(item["canonical_id"])),
    }
    return {**unsigned, "ledger_sha256": _payload_sha256(unsigned)}


def build_stage1_ad_validation(
    *,
    ab_result_root: Path,
    cd_ledger_path: Path,
    expected_ladder_counts: Mapping[str, int] | None = CANONICAL_READOUT_COUNTS,
) -> Stage1ValidationTables:
    """Build the canonical A--D readout and validation registry.

    ``expected_ladder_counts=None`` disables only the canonical panel-size
    check, which is useful for bounded fixtures. All identity, completeness,
    metric, and lineage checks remain enabled.
    """

    ab_result_root = ab_result_root.resolve()
    cd_ledger_path = cd_ledger_path.resolve()
    summary_path = ab_result_root / "control" / "final_summary.json"
    summary = _load_json(summary_path)
    summary_sha256 = _file_sha256(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("errors") != []
        or int(summary.get("observed_results", -1))
        != int(summary.get("expected_results", -2))
    ):
        raise Stage1ValidationError(f"A/B final summary is incomplete: {summary_path}")
    summary_rows = summary.get("results")
    if not isinstance(summary_rows, list):
        raise Stage1ValidationError(f"A/B final summary has no results: {summary_path}")

    ledger = _load_json(cd_ledger_path)
    ledger_sha256 = _file_sha256(cd_ledger_path)
    if ledger.get("schema") != "stage1_cd_cross_root_analysis_ledger/v1":
        raise Stage1ValidationError(f"Unexpected C/D ledger schema: {cd_ledger_path}")
    _verify_embedded_hash(
        ledger,
        field="ledger_sha256",
        label=f"C/D ledger {cd_ledger_path}",
    )
    ledger_records = ledger.get("records")
    if (
        not isinstance(ledger_records, list)
        or int(ledger.get("selected_results", -1)) != len(ledger_records)
        or int(ledger.get("expected_results", -2)) != len(ledger_records)
    ):
        raise Stage1ValidationError(f"C/D ledger is incomplete: {cd_ledger_path}")

    readouts: list[dict[str, Any]] = []
    validation_aggregates: list[dict[str, Any]] = []
    validation_language: list[dict[str, Any]] = []
    validation_candidates: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = [
        _artifact_record(
            readout_id=None,
            artifact_type="ab_final_summary",
            path=summary_path,
            sha256=summary_sha256,
            source_path=summary_path,
            source_sha256=summary_sha256,
            selected=True,
        ),
        _artifact_record(
            readout_id=None,
            artifact_type="cd_cross_root_ledger",
            path=cd_ledger_path,
            sha256=ledger_sha256,
            source_path=cd_ledger_path,
            source_sha256=ledger_sha256,
            selected=True,
        ),
    ]
    provenance: list[dict[str, Any]] = []

    seen: set[str] = set()
    for summary_row in sorted(summary_rows, key=lambda item: str(item.get("job_id"))):
        if not isinstance(summary_row, Mapping):
            raise Stage1ValidationError(f"Malformed A/B summary row: {summary_row!r}")
        readout_id = str(summary_row.get("job_id", ""))
        ladder, experts, language_count, nominal_tokens = _readout_parts(readout_id)
        if ladder not in {"A", "B"}:
            raise Stage1ValidationError(f"Non-A/B row in final summary: {readout_id}")
        if readout_id in seen:
            raise Stage1ValidationError(f"Duplicate readout ID: {readout_id}")
        seen.add(readout_id)
        observed_cell = (
            str(summary_row.get("ladder")),
            int(summary_row.get("experts", -1)),
            int(summary_row.get("K", -1)),
            int(summary_row.get("nominal_D_tokens", -1)),
        )
        if observed_cell != (ladder, experts, language_count, nominal_tokens):
            raise Stage1ValidationError(
                f"A/B summary axes disagree with {readout_id}: {observed_cell}"
            )
        if (
            int(summary_row.get("D", -1)) != nominal_tokens
            or summary_row.get("purpose") != "production"
        ):
            raise Stage1ValidationError(f"Invalid A/B production row: {readout_id}")
        manifest_path = (
            ab_result_root / "jobs" / readout_id / "evaluation_manifest.json"
        )
        evaluation = _validate_evaluation(manifest_path)
        if _file_sha256(manifest_path) != str(summary_row.get("sha256", "")):
            raise Stage1ValidationError(
                f"A/B summary evaluation hash mismatch: {readout_id}"
            )
        training_job_id = str(summary_row.get("training_job_id", ""))
        training = _training_metadata(
            evaluation, expected_training_job_id=training_job_id
        )
        if int(summary_row.get("actual_end_tokens", -1)) != int(
            training["actual_tokens"]
        ) or int(summary_row.get("expected_consumed_samples", -1)) != int(
            training["expected_consumed_samples"]
        ):
            raise Stage1ValidationError(
                f"A/B training cutoff mismatch for {readout_id}"
            )
        primary = float(evaluation.report["aggregates"]["primary"]["value"])
        secondary = float(evaluation.report["aggregates"]["secondary"]["value"])
        if not math.isclose(
            float(summary_row["equal_language_atlas_tao_adjusted_loss"]),
            primary,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(summary_row["raw_token_nll"]),
            secondary,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise Stage1ValidationError(f"A/B summary metric mismatch for {readout_id}")
        readout = _normalize_readout(
            readout_id=readout_id,
            evaluation=evaluation,
            training=training,
            source_family="ab_final_summary",
            source_path=summary_path,
            source_sha256=summary_sha256,
            selection_reason="canonical_ab_final_summary",
            candidate_count=1,
            valid_candidate_count=1,
        )
        validation_candidates.append(
            {
                "readout_id": readout_id,
                "ladder": ladder,
                "candidate_manifest_path": str(manifest_path.resolve()),
                "candidate_manifest_sha256": evaluation.manifest_file_sha256,
                "candidate_result_sha256": evaluation.manifest["result_sha256"],
                "candidate_report_path": str(evaluation.report_path),
                "candidate_report_sha256": evaluation.report_file_sha256,
                "semantic_sufficient_statistics_sha256": evaluation.semantic_sha256,
                "ledger_valid": True,
                "selected": True,
                "semantically_matches_selected": True,
                "selection_reason": "canonical_ab_final_summary",
                "ledger_details_json": "{}",
                "ledger_error": None,
                "source_path": str(summary_path),
                "source_sha256": summary_sha256,
                "provenance_class": "observed",
            }
        )
        _append_selected_evaluation(
            readout=readout,
            evaluation=evaluation,
            training=training,
            source_path=summary_path,
            source_sha256=summary_sha256,
            readouts=readouts,
            validation_aggregates=validation_aggregates,
            validation_language=validation_language,
            artifacts=artifacts,
            provenance=provenance,
        )
        provenance.append(
            _provenance_record(
                readout_id=readout_id,
                relation="ab_final_summary_selects_evaluation",
                subject_path=summary_path,
                object_path=manifest_path,
                source_path=summary_path,
                source_sha256=summary_sha256,
            )
        )

    for ledger_record in sorted(
        ledger_records, key=lambda item: str(item.get("canonical_id"))
    ):
        if not isinstance(ledger_record, Mapping):
            raise Stage1ValidationError(
                f"Malformed C/D ledger record: {ledger_record!r}"
            )
        readout_id = str(ledger_record.get("canonical_id", ""))
        ladder, _, _, _ = _readout_parts(readout_id)
        if ladder not in {"C", "D"}:
            raise Stage1ValidationError(f"Non-C/D row in ledger: {readout_id}")
        if str(ledger_record.get("ladder")) != ladder:
            raise Stage1ValidationError(f"C/D ledger ladder mismatch: {readout_id}")
        if readout_id in seen:
            raise Stage1ValidationError(f"Duplicate readout ID: {readout_id}")
        seen.add(readout_id)
        candidates = ledger_record.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise Stage1ValidationError(f"C/D ledger has no candidates: {readout_id}")
        if int(ledger_record.get("candidate_count", -1)) != len(candidates):
            raise Stage1ValidationError(f"C/D candidate count mismatch: {readout_id}")
        selected_path = _resolve_path(
            ledger_record.get("selected_manifest", ""),
            relative_to=cd_ledger_path.parent,
        )
        evaluations: dict[Path, _ValidatedEvaluation] = {}
        valid_count = 0
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise Stage1ValidationError(f"Malformed C/D candidate for {readout_id}")
            candidate_path = _resolve_path(
                candidate.get("path", ""), relative_to=cd_ledger_path.parent
            )
            ledger_valid = candidate.get("valid") is True
            evaluation: _ValidatedEvaluation | None = None
            if ledger_valid:
                valid_count += 1
                evaluation = _validate_evaluation(candidate_path)
                details = candidate.get("details")
                if not isinstance(details, Mapping):
                    raise Stage1ValidationError(
                        f"Valid C/D candidate lacks details: {candidate_path}"
                    )
                if str(
                    details.get("manifest_file_sha256")
                ) != evaluation.manifest_file_sha256 or str(
                    details.get("result_sha256")
                ) != str(evaluation.manifest["result_sha256"]):
                    raise Stage1ValidationError(
                        f"C/D ledger candidate identity changed: {candidate_path}"
                    )
                evaluations[candidate_path] = evaluation
                artifacts.extend(
                    [
                        _artifact_record(
                            readout_id=readout_id,
                            artifact_type="evaluation_candidate_manifest",
                            path=candidate_path,
                            sha256=evaluation.manifest_file_sha256,
                            source_path=cd_ledger_path,
                            source_sha256=ledger_sha256,
                            selected=candidate_path == selected_path,
                            candidate_path=candidate_path,
                        ),
                        _artifact_record(
                            readout_id=readout_id,
                            artifact_type="evaluation_candidate_report",
                            path=evaluation.report_path,
                            sha256=evaluation.report_file_sha256,
                            source_path=candidate_path,
                            source_sha256=evaluation.manifest_file_sha256,
                            selected=candidate_path == selected_path,
                            candidate_path=candidate_path,
                        ),
                    ]
                )
            validation_candidates.append(
                {
                    "readout_id": readout_id,
                    "ladder": ladder,
                    "candidate_manifest_path": str(candidate_path),
                    "candidate_manifest_sha256": (
                        None if evaluation is None else evaluation.manifest_file_sha256
                    ),
                    "candidate_result_sha256": (
                        None
                        if evaluation is None
                        else evaluation.manifest["result_sha256"]
                    ),
                    "candidate_report_path": (
                        None if evaluation is None else str(evaluation.report_path)
                    ),
                    "candidate_report_sha256": (
                        None if evaluation is None else evaluation.report_file_sha256
                    ),
                    "semantic_sufficient_statistics_sha256": (
                        None if evaluation is None else evaluation.semantic_sha256
                    ),
                    "ledger_valid": ledger_valid,
                    "selected": candidate_path == selected_path,
                    "semantically_matches_selected": None,
                    "selection_reason": (
                        ledger_record.get("selection_reason")
                        if candidate_path == selected_path
                        else None
                    ),
                    "ledger_details_json": _json_cell(candidate.get("details", {})),
                    "ledger_error": candidate.get("error"),
                    "source_path": str(cd_ledger_path),
                    "source_sha256": ledger_sha256,
                    "provenance_class": "observed",
                }
            )
        if valid_count != int(ledger_record.get("valid_candidate_count", -1)):
            raise Stage1ValidationError(
                f"C/D valid-candidate count mismatch: {readout_id}"
            )
        if selected_path not in evaluations:
            raise Stage1ValidationError(
                f"Selected C/D candidate is not valid: {readout_id}"
            )
        semantic_hashes = {
            evaluation.semantic_sha256 for evaluation in evaluations.values()
        }
        if len(semantic_hashes) != 1:
            raise Stage1ValidationError(
                f"Valid C/D duplicates are not semantically identical: {readout_id}"
            )
        checkpoint_bindings = {
            (
                str(
                    _resolve_path(
                        evaluation.manifest.get("checkpoint", {}).get("root", ""),
                        relative_to=evaluation.manifest_path.parent,
                    )
                ),
                int(evaluation.manifest.get("checkpoint", {}).get("step", -1)),
            )
            for evaluation in evaluations.values()
        }
        if len(checkpoint_bindings) != 1:
            raise Stage1ValidationError(
                f"Valid C/D duplicates do not evaluate one checkpoint: {readout_id}"
            )
        selected = evaluations[selected_path]
        for candidate_row in validation_candidates:
            if (
                candidate_row["readout_id"] == readout_id
                and candidate_row["ledger_valid"]
            ):
                candidate_row["semantically_matches_selected"] = (
                    candidate_row["semantic_sufficient_statistics_sha256"]
                    == selected.semantic_sha256
                )
        training = _training_metadata(selected, expected_training_job_id=None)
        readout = _normalize_readout(
            readout_id=readout_id,
            evaluation=selected,
            training=training,
            source_family="cd_cross_root_ledger",
            source_path=cd_ledger_path,
            source_sha256=ledger_sha256,
            selection_reason=str(ledger_record.get("selection_reason", "")),
            candidate_count=len(candidates),
            valid_candidate_count=valid_count,
        )
        _append_selected_evaluation(
            readout=readout,
            evaluation=selected,
            training=training,
            source_path=cd_ledger_path,
            source_sha256=ledger_sha256,
            readouts=readouts,
            validation_aggregates=validation_aggregates,
            validation_language=validation_language,
            artifacts=artifacts,
            provenance=provenance,
        )
        provenance.append(
            _provenance_record(
                readout_id=readout_id,
                relation="cd_cross_root_ledger_selects_evaluation",
                subject_path=cd_ledger_path,
                object_path=selected_path,
                source_path=cd_ledger_path,
                source_sha256=ledger_sha256,
                details={
                    "selection_reason": ledger_record.get("selection_reason"),
                    "candidate_count": len(candidates),
                    "valid_candidate_count": valid_count,
                    "semantic_sufficient_statistics_sha256": selected.semantic_sha256,
                },
            )
        )

    counts = Counter(str(row["ladder"]) for row in readouts)
    if expected_ladder_counts is not None:
        expected = {
            str(key): int(value) for key, value in expected_ladder_counts.items()
        }
        observed = {ladder: counts.get(ladder, 0) for ladder in expected}
        extras = sorted(set(counts) - set(expected))
        if observed != expected or extras:
            raise Stage1ValidationError(
                f"Stage-1 readout panel mismatch: observed={observed}, "
                f"expected={expected}, extras={extras}"
            )
        expected_ab = expected.get("A", 0) + expected.get("B", 0)
        expected_cd = expected.get("C", 0) + expected.get("D", 0)
        if len(summary_rows) != expected_ab or len(ledger_records) != expected_cd:
            raise Stage1ValidationError(
                "Stage-1 source family counts disagree with expected ladder counts"
            )

    readouts.sort(
        key=lambda row: (
            row["ladder"],
            row["experts"],
            row["language_count"],
            row["nominal_tokens"],
        )
    )
    validation_aggregates.sort(
        key=lambda row: (
            row["readout_id"],
            row["aggregate_kind"],
            row["cohort"],
        )
    )
    validation_language.sort(key=lambda row: (row["readout_id"], row["language"]))
    validation_candidates.sort(
        key=lambda row: (row["readout_id"], row["candidate_manifest_path"])
    )
    artifacts.sort(
        key=lambda row: (
            str(row["readout_id"]),
            row["artifact_type"],
            row["path"],
        )
    )
    provenance.sort(
        key=lambda row: (
            str(row["readout_id"]),
            row["relation"],
            row["object_path"],
        )
    )
    return Stage1ValidationTables(
        readouts=tuple(readouts),
        validation_aggregates=tuple(validation_aggregates),
        validation_language=tuple(validation_language),
        validation_candidates=tuple(validation_candidates),
        artifacts=tuple(artifacts),
        provenance=tuple(provenance),
    )


__all__ = [
    "CANONICAL_READOUT_COUNTS",
    "Stage1ValidationError",
    "Stage1ValidationTables",
    "build_stage1_ad_validation",
    "semantic_sufficient_statistics_sha256",
]
