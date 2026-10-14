"""Validate evaluation artifacts and expose their study metrics.

This is deliberately separate from launch and scheduler concerns: it owns the
identity chain from an evaluation manifest to its per-language evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .common import WorkflowError, _file_sha256, _load_object, _payload_sha256


def _validation_report_from_eval(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load an evaluation and its report only after verifying both identities."""

    payload = _load_object(path)
    embedded_result_hash = payload.get("result_sha256")
    result_without_hash = dict(payload)
    result_without_hash.pop("result_sha256", None)
    if embedded_result_hash != _payload_sha256(result_without_hash):
        raise WorkflowError(f"evaluation manifest result hash mismatch: {path}")
    descriptor = payload.get("validation_report")
    if not isinstance(descriptor, dict):
        raise WorkflowError(f"evaluation manifest lacks validation_report: {path}")
    report_path = Path(str(descriptor.get("path", "")))
    if not report_path.is_absolute():
        report_path = path.parent / report_path
    report_path = report_path.resolve()
    expected_file_hash = str(descriptor.get("sha256", ""))
    if not report_path.is_file() or _file_sha256(report_path) != expected_file_hash:
        raise WorkflowError(f"validation report identity mismatch: {report_path}")
    report = _load_object(report_path)
    expected_report_hash = report.get("report_sha256")
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256", None)
    if expected_report_hash != descriptor.get(
        "report_sha256"
    ) or expected_report_hash != _payload_sha256(report_without_hash):
        raise WorkflowError(f"validation report payload hash mismatch: {report_path}")
    return payload, report, report_path


def _metric_from_eval(path: Path) -> float:
    _payload, report, report_path = _validation_report_from_eval(path)
    aggregates = report.get("aggregates")
    primary = aggregates.get("primary") if isinstance(aggregates, dict) else None
    candidate = primary.get("value") if isinstance(primary, dict) else None
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        value = math.nan
    if math.isfinite(value):
        return value
    raise WorkflowError(
        f"validation report has no finite aggregates.primary.value: {report_path}"
    )


def _secondary_metric_from_eval(path: Path) -> float:
    _payload, report, report_path = _validation_report_from_eval(path)
    aggregates = report.get("aggregates")
    secondary = aggregates.get("secondary") if isinstance(aggregates, dict) else None
    candidate = secondary.get("value") if isinstance(secondary, dict) else None
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        value = math.nan
    if math.isfinite(value):
        return value
    raise WorkflowError(
        f"validation report has no finite aggregates.secondary.value: {report_path}"
    )


def _validate_full_evaluation_artifacts(path: Path) -> dict[str, Any]:
    payload, report, report_path = _validation_report_from_eval(path)
    if payload.get("status") != "complete" or report.get("status") != "complete":
        raise WorkflowError(f"evaluation is not complete: {path}")
    coverage = payload.get("coverage")
    languages = payload.get("languages")
    if not isinstance(coverage, dict) or not isinstance(languages, dict):
        raise WorkflowError(f"evaluation lacks coverage/language descriptors: {path}")
    required = coverage.get("required_languages")
    evaluated = coverage.get("evaluated_languages")
    if (
        not isinstance(required, list)
        or evaluated != sorted(required)
        or coverage.get("missing_required_languages") != []
        or set(languages) != set(required)
    ):
        raise WorkflowError(f"evaluation language coverage is incomplete: {path}")
    report_coverage = report.get("coverage")
    if (
        not isinstance(report_coverage, dict)
        or report_coverage.get("required_languages") != required
        or report_coverage.get("observed_languages") != sorted(required)
        or report_coverage.get("missing_required_languages") != []
    ):
        raise WorkflowError(f"validation report coverage is incomplete: {report_path}")
    is_moe = payload.get("moe_routing_diagnostics", {}).get("model_is_moe") is True
    artifact_hashes: dict[str, str] = {}
    for language in required:
        descriptor = languages.get(language)
        if not isinstance(descriptor, dict):
            raise WorkflowError(f"invalid {language} descriptor in {path}")
        language_path = Path(str(descriptor.get("path", "")))
        if not language_path.is_absolute():
            language_path = path.parent / language_path
        language_path = language_path.resolve()
        if not language_path.is_file() or _file_sha256(language_path) != descriptor.get(
            "sha256"
        ):
            raise WorkflowError(
                f"{language} artifact identity mismatch: {language_path}"
            )
        item = _load_object(language_path)
        try:
            token_count = int(item["token_count"])
            backend_count = int(item["backend"]["backend_token_count"])
            nll_sum = float(item["nll_sum"])
            raw_nll = float(item["raw_nll"])
            adjusted_sum = float(item["adjusted_nll_sum"])
            adjusted = float(item["atlas_tao_unigram_adjusted_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"invalid {language} metric artifact: {exc}") from exc
        if (
            item.get("language") != language
            or token_count <= 0
            or token_count != backend_count
            or token_count != int(descriptor.get("token_count", -1))
            or not all(
                math.isfinite(value)
                for value in (nll_sum, raw_nll, adjusted_sum, adjusted)
            )
            or not math.isclose(
                raw_nll, nll_sum / token_count, rel_tol=1e-12, abs_tol=1e-12
            )
            or not math.isclose(
                adjusted, adjusted_sum / token_count, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise WorkflowError(f"invalid exact metric/denominator for {language}")
        routing = item.get("moe_routing")
        if (
            not isinstance(routing, dict)
            or routing.get("evidence_available") is not True
        ):
            raise WorkflowError(f"missing routing evidence for {language}")
        if is_moe:
            dropped = routing.get("totals", {}).get("dropped_assignments")
            if routing.get("applicable") is not True or dropped != 0:
                raise WorkflowError(f"invalid MoE routing/drop evidence for {language}")
        elif routing.get("applicable") is not False:
            raise WorkflowError(f"invalid dense routing marker for {language}")
        artifact_hashes[language] = str(descriptor["sha256"])
    for metric_name in ("primary", "secondary"):
        try:
            value = float(report["aggregates"][metric_name]["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"missing report {metric_name} metric: {path}") from exc
        if not math.isfinite(value):
            raise WorkflowError(f"non-finite report {metric_name} metric: {path}")
    return {
        "evaluation_manifest_sha256": _file_sha256(path),
        "validation_report_sha256": _file_sha256(report_path),
        "language_count": len(required),
        "language_artifact_set_sha256": _payload_sha256(artifact_hashes),
        "model_is_moe": is_moe,
    }


def _cohort_metric_from_eval(path: Path, cohort: str) -> float:
    _payload, report, report_path = _validation_report_from_eval(path)
    try:
        descriptor = report["aggregates"]["fixed_common_cohorts"][cohort]
        value = float(descriptor["equal_language_atlas_tao_adjusted_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            f"validation report lacks finite {cohort} cohort metric: {report_path}"
        ) from exc
    if descriptor.get("status") != "complete" or not math.isfinite(value):
        raise WorkflowError(
            f"validation report has invalid {cohort} cohort metric: {report_path}"
        )
    return value
