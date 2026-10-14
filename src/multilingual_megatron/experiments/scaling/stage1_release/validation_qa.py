"""Independent QA for Stage-1 A--D language and validation release tables.

These helpers deliberately consume the normalized release records rather than
the raw report-building implementation.  Aggregate values are reconstructed
from per-language sufficient statistics, validation identities are checked
against their canonical hash inputs, and routing conservation is checked
against the declared model dimensions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


_LANGUAGE_ID = re.compile(r"^([a-z]{3})_([A-Z][a-z]{3})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIER_SIZES = (1, 12, 50, 200)
_TIER_PATH = "configs/language_groupings/scaling_fw2_v2_k{size:03d}.json"
_LEGACY_PATH = "configs/scaling_law/multilingual_moe_v1/language_sets.json"
_LANGUAGE_LOSS_SCHEMA = "multilingual_validation_language_loss/v1"


class Stage1ReleaseQAError(RuntimeError):
    """Raised when a release invariant cannot be verified."""


@dataclass(frozen=True)
class QAResult:
    """Structured, table-ready QA result."""

    summary: dict[str, Any]
    checks: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LanguageDimensions:
    """Canonical language dimension and ordered tier-membership bridge."""

    languages: tuple[dict[str, Any], ...]
    memberships: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]
    qa: tuple[dict[str, Any], ...]


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
        raise Stage1ReleaseQAError(f"Value is not canonical JSON: {exc}") from exc


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Stage1ReleaseQAError(f"Cannot hash source {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ReleaseQAError(f"Cannot load JSON source {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage1ReleaseQAError(f"Expected a JSON object: {path}")
    return payload


def _json_cell(value: Any) -> str:
    return _canonical_bytes(value).decode("utf-8")


def _parse_json_cell(value: Any, *, label: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise Stage1ReleaseQAError(f"{label} is not valid JSON: {value!r}") from exc
    return value


def _language_parts(language: str) -> tuple[str, str]:
    match = _LANGUAGE_ID.fullmatch(language)
    if match is None:
        raise Stage1ReleaseQAError(f"Invalid production language ID: {language!r}")
    return match.group(1), match.group(2)


def _ordered_language_hash(languages: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for language in languages:
        digest.update(language.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage1ReleaseQAError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise Stage1ReleaseQAError(f"{label} is not finite: {value!r}")
    return result


def _optional_finite(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label=label)


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage1ReleaseQAError(f"{label} is not an integer: {value!r}") from exc
    if minimum is not None and result < minimum:
        raise Stage1ReleaseQAError(f"{label} must be >= {minimum}: {result}")
    return result


def _assert_close(
    observed: Any,
    expected: float | None,
    *,
    label: str,
    tolerance: float,
) -> float:
    if expected is None:
        if observed is not None:
            raise Stage1ReleaseQAError(f"{label}: expected null, got {observed!r}")
        return 0.0
    value = _finite(observed, label=label)
    error = abs(value - expected)
    if not math.isclose(
        value,
        expected,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise Stage1ReleaseQAError(
            f"{label}: observed {value!r}, independently reconstructed {expected!r}"
        )
    return error


def _source_row(
    *,
    source_id: str,
    source_type: str,
    path: Path,
    embedded_sha256: str | None,
    language_count: int | None,
) -> dict[str, Any]:
    return {
        "language_metadata_source_id": source_id,
        "source_type": source_type,
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "embedded_sha256": embedded_sha256,
        "language_count": language_count,
        "provenance_class": "observed",
    }


def build_language_dimensions(
    repo_root: Path,
    corpus_language_rows: Sequence[Mapping[str, Any]],
) -> LanguageDimensions:
    """Build the 200-language dimension and 263 ordered tier memberships.

    Current language IDs and scripts come exclusively from the sealed
    ``scaling_fw2_v2`` grouping files.  Legacy family/name/resource metadata is
    joined only when an ISO-639-3 prefix maps to exactly one legacy language
    record.  A legacy script is never promoted to the current script field.
    """

    repo_root = repo_root.resolve()
    tiers: dict[int, list[str]] = {}
    tier_context: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    for size in _TIER_SIZES:
        path = (repo_root / _TIER_PATH.format(size=size)).resolve()
        payload = _load_json(path)
        group = payload.get("0")
        if not isinstance(group, Mapping):
            raise Stage1ReleaseQAError(f"Language grouping lacks group 0: {path}")
        languages = group.get("languages")
        if not isinstance(languages, list) or any(
            not isinstance(item, str) for item in languages
        ):
            raise Stage1ReleaseQAError(f"Language grouping is malformed: {path}")
        if len(languages) != size or len(languages) != len(set(languages)):
            raise Stage1ReleaseQAError(
                f"K{size} must contain {size} unique languages: {path}"
            )
        for language in languages:
            _language_parts(language)
        embedded = str(payload.get("language_set_sha256", ""))
        observed = _ordered_language_hash(languages)
        if embedded != observed:
            raise Stage1ReleaseQAError(
                f"K{size} ordered membership hash mismatch: {embedded} != {observed}"
            )
        source_id = str(payload.get("language_set_id", ""))
        if not source_id:
            raise Stage1ReleaseQAError(f"Language set ID is missing: {path}")
        tiers[size] = list(languages)
        tier_context[size] = {
            "path": path,
            "source_id": source_id,
            "file_sha256": _file_sha256(path),
            "ordered_membership_sha256": observed,
        }
        sources.append(
            _source_row(
                source_id=source_id,
                source_type="production_language_tier",
                path=path,
                embedded_sha256=embedded,
                language_count=size,
            )
        )
        qa.append(
            {
                "check_id": f"language_tier_k{size}_seal",
                "category": "language_dimension",
                "status": "pass",
                "observed": size,
                "expected": size,
                "source_path": str(path),
                "source_sha256": tier_context[size]["file_sha256"],
                "provenance_class": "derived",
            }
        )

    if not (
        set(tiers[1]).issubset(tiers[12])
        and set(tiers[12]).issubset(tiers[50])
        and set(tiers[50]).issubset(tiers[200])
    ):
        raise Stage1ReleaseQAError(
            "Production K1/K12/K50/K200 language tiers are not nested"
        )
    qa.append(
        {
            "check_id": "language_tiers_nested",
            "category": "language_dimension",
            "status": "pass",
            "observed": "K1 subset K12 subset K50 subset K200",
            "expected": "K1 subset K12 subset K50 subset K200",
            "source_path": _json_cell(
                [str(tier_context[size]["path"]) for size in _TIER_SIZES]
            ),
            "source_sha256": _payload_sha256(
                {str(size): tier_context[size]["file_sha256"] for size in _TIER_SIZES}
            ),
            "provenance_class": "derived",
        }
    )

    legacy_path = (repo_root / _LEGACY_PATH).resolve()
    legacy = _load_json(legacy_path)
    legacy_sha256 = _file_sha256(legacy_path)
    sources.append(
        _source_row(
            source_id=str(legacy.get("study_id", "multilingual_moe_v1")),
            source_type="legacy_language_metadata",
            path=legacy_path,
            embedded_sha256=None,
            language_count=None,
        )
    )
    raw_sets = legacy.get("sets")
    if not isinstance(raw_sets, Mapping):
        raise Stage1ReleaseQAError(f"Legacy language sets are missing: {legacy_path}")
    legacy_by_prefix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    source_sets_by_id: dict[str, set[str]] = defaultdict(set)
    for set_name, raw_set in raw_sets.items():
        if not isinstance(raw_set, Mapping) or not isinstance(
            raw_set.get("languages"), list
        ):
            raise Stage1ReleaseQAError(f"Legacy language set {set_name!r} is malformed")
        for raw in raw_set["languages"]:
            if not isinstance(raw, Mapping):
                raise Stage1ReleaseQAError(
                    f"Legacy language row in {set_name!r} is malformed"
                )
            legacy_id = str(raw.get("id", ""))
            prefix, _ = _language_parts(legacy_id)
            material = {
                "id": legacy_id,
                "family": raw.get("family"),
                "name": raw.get("name"),
                "resource_availability": raw.get("resource_availability"),
                "script": raw.get("script"),
            }
            prior = legacy_by_prefix[prefix].get(legacy_id)
            if prior is not None and prior != material:
                raise Stage1ReleaseQAError(
                    f"Legacy metadata disagrees across sets for {legacy_id}"
                )
            legacy_by_prefix[prefix][legacy_id] = material
            source_sets_by_id[legacy_id].add(str(set_name))

    corpus_by_language: dict[str, Mapping[str, Any]] = {}
    for row in corpus_language_rows:
        language = str(row.get("language", ""))
        if not language or language in corpus_by_language:
            raise Stage1ReleaseQAError(
                f"Corpus language rows have missing/duplicate ID {language!r}"
            )
        corpus_by_language[language] = row
    production_set = set(tiers[200])
    if set(corpus_by_language) != production_set:
        missing = sorted(production_set - set(corpus_by_language))
        extra = sorted(set(corpus_by_language) - production_set)
        raise Stage1ReleaseQAError(
            f"Corpus/K200 language mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    ranks = {
        size: {language: index for index, language in enumerate(values, start=1)}
        for size, values in tiers.items()
    }
    languages: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    mismatch_count = 0
    for k200_rank, language in enumerate(tiers[200], start=1):
        prefix, production_script = _language_parts(language)
        candidates = legacy_by_prefix.get(prefix, {})
        metadata: Mapping[str, Any] | None
        if len(candidates) == 1:
            metadata = next(iter(candidates.values()))
            method = "unique_iso639_3_prefix"
        elif not candidates:
            metadata = None
            method = "no_legacy_iso639_3_prefix"
        else:
            metadata = None
            method = "ambiguous_legacy_iso639_3_prefix"
        method_counts[method] += 1
        legacy_id = None if metadata is None else str(metadata["id"])
        legacy_script = None if metadata is None else metadata.get("script")
        stale_script_mismatch = bool(
            metadata is not None and legacy_script != production_script
        )
        mismatch_count += int(stale_script_mismatch)
        corpus = corpus_by_language[language]
        corpus_inventory = dict(corpus)
        row = {
            "language": language,
            "iso639_3": prefix,
            "production_script": production_script,
            "k200_rank": k200_rank,
            "in_k1": language in ranks[1],
            "k1_rank": ranks[1].get(language),
            "in_k12": language in ranks[12],
            "k12_rank": ranks[12].get(language),
            "in_k50": language in ranks[50],
            "k50_rank": ranks[50].get(language),
            "in_k200": True,
            "family": None if metadata is None else metadata.get("family"),
            "name": None if metadata is None else metadata.get("name"),
            "resource_availability": (
                None if metadata is None else metadata.get("resource_availability")
            ),
            "legacy_metadata_join_method": method,
            "legacy_metadata_source_language_id": legacy_id,
            "legacy_metadata_source_set_ids_json": _json_cell(
                [] if legacy_id is None else sorted(source_sets_by_id[legacy_id])
            ),
            "legacy_script": legacy_script,
            "stale_script_mismatch": stale_script_mismatch,
            "production_k200_source_id": tier_context[200]["source_id"],
            "production_k200_source_path": str(tier_context[200]["path"]),
            "production_k200_source_file_sha256": tier_context[200]["file_sha256"],
            "production_k200_ordered_membership_sha256": tier_context[200][
                "ordered_membership_sha256"
            ],
            "legacy_metadata_source_id": str(
                legacy.get("study_id", "multilingual_moe_v1")
            ),
            "legacy_metadata_source_path": str(legacy_path),
            "legacy_metadata_source_file_sha256": legacy_sha256,
            "corpus_inventory_json": _json_cell(corpus_inventory),
            "provenance_class": "observed",
        }
        for key in (
            "dataset_release_id",
            "exact_unique_language_tokens_including_eod",
            "unique_language_tokens_lower_bound_including_eod",
            "train_status",
            "train_target_tokens",
            "train_selected_tokens",
            "train_shortfall_tokens",
            "train_emitted_documents",
            "train_path",
            "train_sha256",
            "validation_status",
            "validation_target_tokens",
            "validation_selected_tokens",
            "validation_shortfall_tokens",
            "validation_emitted_documents",
            "validation_path",
            "validation_sha256",
            "complete_source_scan",
            "acquisition_status",
            "source_coverage_report_path",
            "source_coverage_report_payload_sha256",
            "corpus_manifest_path",
            "corpus_manifest_sha256",
        ):
            row[key] = corpus.get(key)
        languages.append(row)

    memberships: list[dict[str, Any]] = []
    for size in _TIER_SIZES:
        context = tier_context[size]
        for rank, language in enumerate(tiers[size], start=1):
            prefix, script = _language_parts(language)
            memberships.append(
                {
                    "language_set_id": context["source_id"],
                    "tier": f"K{size}",
                    "language_count": size,
                    "tier_rank": rank,
                    "language": language,
                    "iso639_3": prefix,
                    "production_script": script,
                    "ordered_membership_sha256": context["ordered_membership_sha256"],
                    "source_path": str(context["path"]),
                    "source_file_sha256": context["file_sha256"],
                    "provenance_class": "observed",
                }
            )
    if len(languages) != 200 or len(memberships) != 263:
        raise Stage1ReleaseQAError(
            f"Language dimension cardinality mismatch: "
            f"{len(languages)} languages/{len(memberships)} memberships"
        )
    qa.extend(
        (
            {
                "check_id": "language_dimension_cardinality",
                "category": "language_dimension",
                "status": "pass",
                "observed": len(languages),
                "expected": 200,
                "source_path": str(tier_context[200]["path"]),
                "source_sha256": tier_context[200]["file_sha256"],
                "provenance_class": "derived",
            },
            {
                "check_id": "language_tier_membership_cardinality",
                "category": "language_dimension",
                "status": "pass",
                "observed": len(memberships),
                "expected": 263,
                "source_path": _json_cell(
                    [str(tier_context[size]["path"]) for size in _TIER_SIZES]
                ),
                "source_sha256": _payload_sha256(
                    {
                        str(size): tier_context[size]["file_sha256"]
                        for size in _TIER_SIZES
                    }
                ),
                "provenance_class": "derived",
            },
            {
                "check_id": "legacy_metadata_join_coverage",
                "category": "language_dimension",
                "status": "pass",
                "observed": _json_cell(dict(sorted(method_counts.items()))),
                "expected": "explicit unique, missing, or ambiguous join status",
                "source_path": str(legacy_path),
                "source_sha256": legacy_sha256,
                "provenance_class": "derived",
            },
            {
                "check_id": "legacy_stale_script_disclosure",
                "category": "language_dimension",
                "status": "pass",
                "observed": mismatch_count,
                "expected": "all prefix-joined script mismatches explicitly flagged",
                "source_path": str(legacy_path),
                "source_sha256": legacy_sha256,
                "provenance_class": "derived",
            },
        )
    )
    return LanguageDimensions(
        languages=tuple(languages),
        memberships=tuple(memberships),
        sources=tuple(sources),
        qa=tuple(qa),
    )


def _cohort_hash(name: str, languages: Sequence[str]) -> str:
    return _payload_sha256({"name": name, "languages": list(languages)})


def build_validation_cohorts(
    validation_aggregates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Deduplicate reported cohort memberships and verify membership seals."""

    collected: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for row in validation_aggregates:
        kind = str(row.get("aggregate_kind", ""))
        cohort = str(row.get("cohort", ""))
        raw_languages = _parse_json_cell(
            row.get("languages_json", []),
            label=f"{row.get('readout_id')}/{kind}/{cohort}.languages_json",
        )
        if not isinstance(raw_languages, list) or any(
            not isinstance(language, str) for language in raw_languages
        ):
            raise Stage1ReleaseQAError(
                f"Malformed cohort membership: {row.get('readout_id')}/{cohort}"
            )
        if len(raw_languages) != len(set(raw_languages)):
            raise Stage1ReleaseQAError(
                f"Duplicate cohort language: {row.get('readout_id')}/{cohort}"
            )
        languages = tuple(raw_languages)
        computed_hash = _cohort_hash(cohort, languages)
        declared_hash = row.get("membership_sha256")
        if declared_hash is not None and declared_hash != computed_hash:
            raise Stage1ReleaseQAError(
                f"Cohort membership hash mismatch for {row.get('readout_id')}/{cohort}"
            )
        key = (kind, cohort, languages)
        record = collected.setdefault(
            key,
            {
                "validation_cohort_id": (f"{kind}:{cohort}:{computed_hash[:16]}"),
                "aggregate_kind": kind,
                "cohort": cohort,
                "language_count": len(languages),
                "languages_json": _json_cell(languages),
                "membership_sha256": computed_hash,
                "declared_membership_sha256": declared_hash,
                "readout_ids": set(),
                "statuses": set(),
                "provenance_class": "derived",
            },
        )
        if record["declared_membership_sha256"] not in (None, declared_hash):
            raise Stage1ReleaseQAError(
                f"Conflicting declared hashes for {kind}/{cohort}"
            )
        if record["declared_membership_sha256"] is None:
            record["declared_membership_sha256"] = declared_hash
        record["readout_ids"].add(str(row.get("readout_id", "")))
        record["statuses"].add(str(row.get("status", "")))
    result: list[dict[str, Any]] = []
    for key in sorted(collected):
        raw = collected[key]
        result.append(
            {
                **{
                    field: value
                    for field, value in raw.items()
                    if field not in {"readout_ids", "statuses"}
                },
                "readout_count": len(raw["readout_ids"]),
                "statuses_json": _json_cell(sorted(raw["statuses"])),
            }
        )
    return tuple(result)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["language"]))
    if not ordered:
        return {
            "status": "empty",
            "language_count": 0,
            "token_count": 0,
            "languages": [],
            "equal_language_adjusted_loss": None,
            "equal_language_raw_nll": None,
            "token_micro_adjusted_loss": None,
            "token_micro_raw_nll": None,
        }
    token_count = sum(
        _integer(row["token_count"], label="token_count", minimum=1) for row in ordered
    )
    return {
        "status": "complete",
        "language_count": len(ordered),
        "token_count": token_count,
        "languages": [str(row["language"]) for row in ordered],
        "equal_language_adjusted_loss": math.fsum(
            _finite(
                row["atlas_tao_unigram_adjusted_loss"],
                label="atlas_tao_unigram_adjusted_loss",
            )
            for row in ordered
        )
        / len(ordered),
        "equal_language_raw_nll": math.fsum(
            _finite(row["raw_nll"], label="raw_nll") for row in ordered
        )
        / len(ordered),
        "token_micro_adjusted_loss": math.fsum(
            _finite(row["adjusted_nll_sum"], label="adjusted_nll_sum")
            for row in ordered
        )
        / token_count,
        "token_micro_raw_nll": math.fsum(
            _finite(row["nll_sum"], label="nll_sum") for row in ordered
        )
        / token_count,
    }


def _compare_membership(
    row: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    observed = _parse_json_cell(row.get("languages_json", []), label=label)
    if observed != list(expected):
        raise Stage1ReleaseQAError(
            f"{label}: observed membership {observed!r}, expected {list(expected)!r}"
        )


def _compare_aggregate(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    tolerance: float,
) -> float:
    if row.get("status") != expected["status"]:
        raise Stage1ReleaseQAError(
            f"{label}.status: {row.get('status')!r} != {expected['status']!r}"
        )
    if (
        _integer(
            row.get("aggregate_language_count"),
            label=f"{label}.language_count",
            minimum=0,
        )
        != expected["language_count"]
    ):
        raise Stage1ReleaseQAError(f"{label}.language_count mismatch")
    if (
        _integer(
            row.get("aggregate_token_count"),
            label=f"{label}.token_count",
            minimum=0,
        )
        != expected["token_count"]
    ):
        raise Stage1ReleaseQAError(f"{label}.token_count mismatch")
    _compare_membership(row, expected["languages"], label=f"{label}.languages")
    errors = [
        _assert_close(
            row.get(field),
            expected[field],
            label=f"{label}.{field}",
            tolerance=tolerance,
        )
        for field in (
            "equal_language_adjusted_loss",
            "equal_language_raw_nll",
            "token_micro_adjusted_loss",
            "token_micro_raw_nll",
        )
    ]
    return max(errors, default=0.0)


def reconstruct_and_verify_aggregates(
    validation_language: Sequence[Mapping[str, Any]],
    validation_aggregates: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-10,
) -> QAResult:
    """Independently reconstruct every released validation aggregate."""

    if tolerance <= 0 or not math.isfinite(tolerance):
        raise Stage1ReleaseQAError("Aggregate tolerance must be finite and positive")
    by_readout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_language: set[tuple[str, str]] = set()
    for row in validation_language:
        readout_id = str(row.get("readout_id", ""))
        language = str(row.get("language", ""))
        key = (readout_id, language)
        if not readout_id or not language or key in seen_language:
            raise Stage1ReleaseQAError(
                f"Missing/duplicate validation language identity: {key}"
            )
        seen_language.add(key)
        token_count = _integer(
            row.get("token_count"),
            label=f"{readout_id}/{language}.token_count",
            minimum=1,
        )
        nll_sum = _finite(row.get("nll_sum"), label=f"{readout_id}/{language}.nll_sum")
        adjusted_sum = _finite(
            row.get("adjusted_nll_sum"),
            label=f"{readout_id}/{language}.adjusted_nll_sum",
        )
        _assert_close(
            row.get("raw_nll"),
            nll_sum / token_count,
            label=f"{readout_id}/{language}.raw_nll",
            tolerance=tolerance,
        )
        _assert_close(
            row.get("atlas_tao_unigram_adjusted_loss"),
            adjusted_sum / token_count,
            label=f"{readout_id}/{language}.adjusted_loss",
            tolerance=tolerance,
        )
        by_readout[readout_id].append(row)

    aggregate_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in validation_aggregates:
        key = (
            str(row.get("readout_id", "")),
            str(row.get("aggregate_kind", "")),
            str(row.get("cohort", "")),
        )
        if key in aggregate_by_key:
            raise Stage1ReleaseQAError(f"Duplicate validation aggregate: {key}")
        aggregate_by_key[key] = row
    if set(readout for readout, _, _ in aggregate_by_key) != set(by_readout):
        raise Stage1ReleaseQAError(
            "Aggregate and per-language readout sets do not match"
        )

    checks: list[dict[str, Any]] = []
    max_error = 0.0
    for readout_id, language_rows in sorted(by_readout.items()):
        trained = [row for row in language_rows if row.get("metric_role") == "trained"]
        zero_shot = [
            row for row in language_rows if row.get("metric_role") == "zero_shot"
        ]
        if len(trained) + len(zero_shot) != len(language_rows) or not trained:
            raise Stage1ReleaseQAError(
                f"{readout_id} has invalid trained/zero-shot role partition"
            )
        role_rows = {
            "trained_languages": trained,
            "zero_shot_languages": zero_shot,
        }
        for cohort, selected in role_rows.items():
            key = (readout_id, "metric_role", cohort)
            if key not in aggregate_by_key:
                raise Stage1ReleaseQAError(f"Missing aggregate {key}")
            expected = _aggregate_rows(selected)
            error = _compare_aggregate(
                aggregate_by_key[key],
                expected,
                label="/".join(key),
                tolerance=tolerance,
            )
            max_error = max(max_error, error)
            checks.append(
                {
                    "check_id": "/".join(key),
                    "category": "validation_aggregate_reconstruction",
                    "status": "pass",
                    "max_absolute_error": error,
                    "language_count": expected["language_count"],
                    "token_count": expected["token_count"],
                    "provenance_class": "derived",
                }
            )

        trained_expected = _aggregate_rows(trained)
        declared = {
            "primary": (
                "trained_equal_language_atlas_tao_unigram_adjusted_loss",
                trained_expected["equal_language_adjusted_loss"],
            ),
            "secondary": (
                "trained_equal_language_raw_nll",
                trained_expected["equal_language_raw_nll"],
            ),
        }
        for cohort, (metric_name, metric_value) in declared.items():
            key = (readout_id, "declared_metric", cohort)
            row = aggregate_by_key.get(key)
            if row is None:
                raise Stage1ReleaseQAError(f"Missing aggregate {key}")
            if row.get("status") != "complete" or row.get("metric_name") != metric_name:
                raise Stage1ReleaseQAError(f"{'/'.join(key)} definition mismatch")
            if _integer(
                row.get("aggregate_language_count"),
                label=f"{'/'.join(key)}.language_count",
                minimum=1,
            ) != len(trained):
                raise Stage1ReleaseQAError(f"{'/'.join(key)} language count mismatch")
            _compare_membership(
                row,
                trained_expected["languages"],
                label=f"{'/'.join(key)}.languages",
            )
            error = _assert_close(
                row.get("metric_value"),
                metric_value,
                label=f"{'/'.join(key)}.metric_value",
                tolerance=tolerance,
            )
            max_error = max(max_error, error)
            checks.append(
                {
                    "check_id": "/".join(key),
                    "category": "validation_declared_metric_reconstruction",
                    "status": "pass",
                    "max_absolute_error": error,
                    "language_count": len(trained),
                    "token_count": None,
                    "provenance_class": "derived",
                }
            )

        by_language = {str(row["language"]): row for row in language_rows}
        trained_set = {str(row["language"]) for row in trained}
        fixed_keys = sorted(
            key
            for key in aggregate_by_key
            if key[0] == readout_id and key[1] == "fixed_common_cohort"
        )
        if not fixed_keys:
            raise Stage1ReleaseQAError(f"{readout_id} has no fixed cohorts")
        for key in fixed_keys:
            row = aggregate_by_key[key]
            members = _parse_json_cell(
                row.get("languages_json", []),
                label=f"{'/'.join(key)}.languages",
            )
            if not isinstance(members, list) or len(members) != len(set(members)):
                raise Stage1ReleaseQAError(f"{'/'.join(key)} has malformed membership")
            declared_hash = row.get("membership_sha256")
            expected_hash = _cohort_hash(key[2], members)
            if declared_hash != expected_hash:
                raise Stage1ReleaseQAError(f"{'/'.join(key)} membership hash mismatch")
            missing_trained = sorted(set(members) - trained_set)
            if missing_trained:
                if (
                    row.get("status") != "not_applicable"
                    or row.get("reason") != "not_all_cohort_languages_were_trained"
                    or _parse_json_cell(
                        row.get("missing_languages_json", []),
                        label=f"{'/'.join(key)}.missing_languages",
                    )
                    != missing_trained
                ):
                    raise Stage1ReleaseQAError(
                        f"{'/'.join(key)} applicability mismatch"
                    )
                for field in (
                    "aggregate_language_count",
                    "aggregate_token_count",
                    "equal_language_adjusted_loss",
                    "equal_language_raw_nll",
                    "token_micro_adjusted_loss",
                    "token_micro_raw_nll",
                ):
                    if row.get(field) is not None:
                        raise Stage1ReleaseQAError(
                            f"{'/'.join(key)}.{field} should be null"
                        )
                error = 0.0
                count = None
                tokens = None
            else:
                missing_observations = sorted(set(members) - set(by_language))
                if missing_observations:
                    raise Stage1ReleaseQAError(
                        f"{'/'.join(key)} lacks observations {missing_observations}"
                    )
                expected = _aggregate_rows(
                    [by_language[language] for language in members]
                )
                error = _compare_aggregate(
                    row,
                    expected,
                    label="/".join(key),
                    tolerance=tolerance,
                )
                count = expected["language_count"]
                tokens = expected["token_count"]
            max_error = max(max_error, error)
            checks.append(
                {
                    "check_id": "/".join(key),
                    "category": "validation_fixed_cohort_reconstruction",
                    "status": "pass",
                    "max_absolute_error": error,
                    "language_count": count,
                    "token_count": tokens,
                    "provenance_class": "derived",
                }
            )

    if len(checks) != len(validation_aggregates):
        unhandled = sorted(
            set(aggregate_by_key)
            - {tuple(check["check_id"].split("/", 2)) for check in checks}
        )
        raise Stage1ReleaseQAError(
            f"Not every aggregate was independently verified: {unhandled[:5]}"
        )
    return QAResult(
        summary={
            "status": "pass",
            "readout_count": len(by_readout),
            "language_observation_count": len(validation_language),
            "aggregate_count": len(validation_aggregates),
            "verified_aggregate_count": len(checks),
            "max_absolute_error": max_error,
            "tolerance": tolerance,
            "provenance_class": "derived",
        },
        checks=tuple(checks),
    )


def validate_validation_identities(
    validation_language: Sequence[Mapping[str, Any]],
) -> QAResult:
    """Verify raw record hashes and stable, outcome-free input identities."""

    by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    global_vocab_specs: set[tuple[int, tuple[int, ...]]] = set()
    outcome_inclusive_record_hashes_verified = 0
    validation_input_identity_hashes_verified = 0
    for row in validation_language:
        readout_id = str(row.get("readout_id", ""))
        language = str(row.get("language", ""))
        key = (readout_id, language)
        if not readout_id or not language or key in seen:
            raise Stage1ReleaseQAError(f"Missing/duplicate validation identity: {key}")
        seen.add(key)
        for field in (
            "target_token_ids_sha256",
            "unigram_counts_sha256",
            "input_sha256",
            "validation_input_identity_sha256",
        ):
            value = str(row.get(field, ""))
            if not _SHA256.fullmatch(value):
                raise Stage1ReleaseQAError(
                    f"{readout_id}/{language}.{field} is not SHA-256"
                )
        token_count = _integer(
            row.get("token_count"),
            label=f"{readout_id}/{language}.token_count",
            minimum=1,
        )
        nll_sum = _finite(row.get("nll_sum"), label=f"{readout_id}/{language}.nll_sum")
        expected_input_sha256 = _payload_sha256(
            {
                "schema": _LANGUAGE_LOSS_SCHEMA,
                "language": language,
                "nll_sum_hex": nll_sum.hex(),
                "token_count": token_count,
                "target_token_ids_sha256": row["target_token_ids_sha256"],
                "unigram_counts_sha256": row["unigram_counts_sha256"],
            }
        )
        if row["input_sha256"] != expected_input_sha256:
            raise Stage1ReleaseQAError(
                f"{readout_id}/{language} input SHA-256 reconstruction mismatch"
            )
        outcome_inclusive_record_hashes_verified += 1
        raw_special = _parse_json_cell(
            row.get("special_token_ids_json"),
            label=f"{readout_id}/{language}.special_token_ids_json",
        )
        if not isinstance(raw_special, list) or any(
            not isinstance(value, int) for value in raw_special
        ):
            raise Stage1ReleaseQAError(
                f"{readout_id}/{language} special-token IDs are malformed"
            )
        vocab_size = _integer(
            row.get("vocab_size"),
            label=f"{readout_id}/{language}.vocab_size",
            minimum=1,
        )
        expected_validation_input_identity = _payload_sha256(
            {
                "schema": "multilingual_validation_input_identity/v1",
                "language": language,
                "token_count": token_count,
                "target_token_ids_sha256": row["target_token_ids_sha256"],
                "unigram_counts_sha256": row["unigram_counts_sha256"],
                "vocab_size": vocab_size,
                "special_token_ids": raw_special,
            }
        )
        if (
            row["validation_input_identity_sha256"]
            != expected_validation_input_identity
        ):
            raise Stage1ReleaseQAError(
                f"{readout_id}/{language} validation input identity mismatch"
            )
        validation_input_identity_hashes_verified += 1
        global_vocab_specs.add((vocab_size, tuple(raw_special)))
        by_language[language].append(row)

    if len(global_vocab_specs) != 1:
        raise Stage1ReleaseQAError(
            f"Validation vocabulary/special-token drift: {sorted(global_vocab_specs)}"
        )
    checks: list[dict[str, Any]] = []
    for language, rows in sorted(by_language.items()):
        identities = {
            "target_token_ids_sha256": {
                str(row["target_token_ids_sha256"]) for row in rows
            },
            "unigram_counts_sha256": {
                str(row["unigram_counts_sha256"]) for row in rows
            },
            "token_count": {int(row["token_count"]) for row in rows},
            "validation_input_identity_sha256": {
                str(row["validation_input_identity_sha256"]) for row in rows
            },
        }
        drifting = {
            field: sorted(values)
            for field, values in identities.items()
            if len(values) != 1
        }
        if drifting:
            raise Stage1ReleaseQAError(
                f"Validation input drift for {language}: {drifting}"
            )
        checks.append(
            {
                "check_id": f"validation_identity/{language}",
                "category": "validation_identity",
                "status": "pass",
                "observation_count": len(rows),
                "target_token_ids_sha256": next(
                    iter(identities["target_token_ids_sha256"])
                ),
                "unigram_counts_sha256": next(
                    iter(identities["unigram_counts_sha256"])
                ),
                "token_count": next(iter(identities["token_count"])),
                "provenance_class": "derived",
            }
        )
    vocab_size, special_ids = next(iter(global_vocab_specs))
    checks.append(
        {
            "check_id": "validation_identity/global_vocabulary",
            "category": "validation_identity",
            "status": "pass",
            "observation_count": len(validation_language),
            "vocab_size": vocab_size,
            "special_token_ids_json": _json_cell(special_ids),
            "provenance_class": "derived",
        }
    )
    return QAResult(
        summary={
            "status": "pass",
            "observation_count": len(validation_language),
            "language_count": len(by_language),
            "outcome_inclusive_record_hashes_verified": (
                outcome_inclusive_record_hashes_verified
            ),
            "validation_input_identity_hashes_verified": (
                validation_input_identity_hashes_verified
            ),
            "input_hash_semantics": (
                "input_sha256 is the outcome-inclusive language-loss record hash; "
                "validation_input_identity_sha256 is the outcome-free input seal"
            ),
            "vocab_size": vocab_size,
            "special_token_ids_json": _json_cell(special_ids),
            "provenance_class": "derived",
        },
        checks=tuple(checks),
    )


def validate_routing_rows(
    routing_rows: Sequence[Mapping[str, Any]],
    model_configs: Sequence[Mapping[str, Any]],
) -> QAResult:
    """Verify routing dimensions, assignment conservation, and zero drops."""

    configs: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in model_configs:
        key = (str(row.get("ladder", "")), int(row.get("experts", -1)))
        if key in configs:
            raise Stage1ReleaseQAError(f"Duplicate model configuration: {key}")
        configs[key] = row
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    total_assignments = 0
    total_dropped = 0
    for row in routing_rows:
        readout_id = str(row.get("readout_id", ""))
        language = str(row.get("language", ""))
        ladder = str(row.get("ladder", ""))
        experts = _integer(
            row.get("experts"),
            label=f"{readout_id}/{language}.experts",
            minimum=1,
        )
        if experts == 1:
            raise Stage1ReleaseQAError(
                f"Dense control has applicable routing evidence: {readout_id}/{language}"
            )
        config = configs.get((ladder, experts))
        if config is None:
            raise Stage1ReleaseQAError(
                f"No model configuration for routing row {ladder}:E{experts}"
            )
        runtime_experts = _integer(
            config.get("num_moe_experts_runtime"),
            label=f"{ladder}:E{experts}.num_moe_experts_runtime",
            minimum=1,
        )
        if runtime_experts != experts:
            raise Stage1ReleaseQAError(
                f"Runtime expert dimension mismatch for {ladder}:E{experts}"
            )
        topk = _integer(
            row.get("topk"),
            label=f"{readout_id}/{language}.topk",
            minimum=1,
        )
        configured_topk = _integer(
            config.get("moe_router_topk"),
            label=f"{ladder}:E{experts}.moe_router_topk",
            minimum=1,
        )
        if topk != configured_topk:
            raise Stage1ReleaseQAError(
                f"Routing top-k mismatch for {readout_id}/{language}"
            )
        raw_counts = _parse_json_cell(
            row.get("expert_token_counts_json"),
            label=f"{readout_id}/{language}.expert_token_counts_json",
        )
        if (
            not isinstance(raw_counts, list)
            or len(raw_counts) != experts
            or any(not isinstance(value, int) or value < 0 for value in raw_counts)
        ):
            raise Stage1ReleaseQAError(
                f"Expert-count dimension mismatch for {readout_id}/{language}"
            )
        expected = _integer(
            row.get("expected_assignments"),
            label=f"{readout_id}/{language}.expected_assignments",
            minimum=0,
        )
        kept = _integer(
            row.get("kept_assignments"),
            label=f"{readout_id}/{language}.kept_assignments",
            minimum=0,
        )
        dropped = _integer(
            row.get("dropped_assignments"),
            label=f"{readout_id}/{language}.dropped_assignments",
            minimum=0,
        )
        forwarded = _integer(
            row.get("forwarded_tokens"),
            label=f"{readout_id}/{language}.forwarded_tokens",
            minimum=0,
        )
        if expected != kept + dropped:
            raise Stage1ReleaseQAError(
                f"Assignment conservation mismatch for {readout_id}/{language}"
            )
        if sum(raw_counts) != kept:
            raise Stage1ReleaseQAError(
                f"Expert-count sum mismatch for {readout_id}/{language}"
            )
        if expected != forwarded * topk:
            raise Stage1ReleaseQAError(
                f"Forwarded-token/top-k mismatch for {readout_id}/{language}"
            )
        drop_rate = _finite(
            row.get("dropped_assignment_rate"),
            label=f"{readout_id}/{language}.dropped_assignment_rate",
        )
        if dropped != 0 or drop_rate != 0.0:
            raise Stage1ReleaseQAError(
                f"Non-zero routing drops for {readout_id}/{language}"
            )
        layer_index = _integer(
            row.get("layer_index"),
            label=f"{readout_id}/{language}.layer_index",
            minimum=0,
        )
        total_assignments += expected
        total_dropped += dropped
        groups[(readout_id, language, ladder, experts)].append(
            {**row, "_layer_index": layer_index}
        )

    checks: list[dict[str, Any]] = []
    for (readout_id, language, ladder, experts), rows in sorted(groups.items()):
        config = configs[(ladder, experts)]
        num_layers = _integer(
            config.get("num_layers"),
            label=f"{ladder}:E{experts}.num_layers",
            minimum=1,
        )
        observed_layers = sorted(int(row["_layer_index"]) for row in rows)
        expected_layers = list(range(num_layers))
        if observed_layers != expected_layers:
            raise Stage1ReleaseQAError(
                f"Routing layer dimension mismatch for {readout_id}/{language}: "
                f"{observed_layers} != {expected_layers}"
            )
        checks.append(
            {
                "check_id": f"routing/{readout_id}/{language}",
                "category": "routing_conservation",
                "status": "pass",
                "ladder": ladder,
                "experts": experts,
                "layer_count": num_layers,
                "expected_assignments": sum(
                    int(row["expected_assignments"]) for row in rows
                ),
                "dropped_assignments": 0,
                "provenance_class": "derived",
            }
        )
    return QAResult(
        summary={
            "status": "pass",
            "routing_row_count": len(routing_rows),
            "readout_language_count": len(groups),
            "expected_assignments": total_assignments,
            "dropped_assignments": total_dropped,
            "provenance_class": "derived",
        },
        checks=tuple(checks),
    )


def validate_duplicate_candidate_routing(
    validation_candidates: Sequence[Mapping[str, Any]],
) -> QAResult:
    """Verify routing equality for every valid duplicate C/D evaluation."""

    by_readout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_candidates:
        if row.get("ledger_valid") is True:
            by_readout[str(row.get("readout_id", ""))].append(row)
    duplicate_groups = {
        readout_id: rows for readout_id, rows in by_readout.items() if len(rows) > 1
    }
    checks: list[dict[str, Any]] = []
    candidate_count = 0
    language_payload_count = 0
    for readout_id, rows in sorted(duplicate_groups.items()):
        candidate_hashes: dict[str, str] = {}
        candidate_language_counts: dict[str, int] = {}
        seen_paths: set[Path] = set()
        for row in rows:
            manifest_path = Path(str(row.get("candidate_manifest_path", ""))).resolve()
            if manifest_path in seen_paths:
                raise Stage1ReleaseQAError(
                    f"Duplicate candidate path for {readout_id}: {manifest_path}"
                )
            seen_paths.add(manifest_path)
            manifest = _load_json(manifest_path)
            declared_manifest_sha = row.get("candidate_manifest_sha256")
            if (
                declared_manifest_sha is not None
                and _file_sha256(manifest_path) != declared_manifest_sha
            ):
                raise Stage1ReleaseQAError(
                    f"Candidate manifest identity changed: {manifest_path}"
                )
            descriptors = manifest.get("languages")
            if not isinstance(descriptors, Mapping) or not descriptors:
                raise Stage1ReleaseQAError(
                    f"Candidate has no language artifacts: {manifest_path}"
                )
            routing_by_language: dict[str, Any] = {}
            for language, descriptor in sorted(descriptors.items()):
                if not isinstance(descriptor, Mapping):
                    raise Stage1ReleaseQAError(
                        f"Malformed candidate language descriptor: "
                        f"{manifest_path}/{language}"
                    )
                artifact_path = Path(str(descriptor.get("path", "")))
                if not artifact_path.is_absolute():
                    artifact_path = manifest_path.parent / artifact_path
                artifact_path = artifact_path.resolve()
                declared_sha = str(descriptor.get("sha256", ""))
                if (
                    not artifact_path.is_file()
                    or not _SHA256.fullmatch(declared_sha)
                    or _file_sha256(artifact_path) != declared_sha
                ):
                    raise Stage1ReleaseQAError(
                        f"Candidate language artifact identity changed: {artifact_path}"
                    )
                artifact = _load_json(artifact_path)
                routing = artifact.get("moe_routing")
                if not isinstance(routing, Mapping):
                    raise Stage1ReleaseQAError(
                        f"Candidate lacks routing evidence: {artifact_path}"
                    )
                routing_by_language[str(language)] = routing
            routing_hash = _payload_sha256(
                {
                    "schema": "stage1_duplicate_candidate_routing/v1",
                    "languages": routing_by_language,
                }
            )
            candidate_hashes[str(manifest_path)] = routing_hash
            candidate_language_counts[str(manifest_path)] = len(routing_by_language)
            candidate_count += 1
            language_payload_count += len(routing_by_language)
        if len(set(candidate_hashes.values())) != 1:
            raise Stage1ReleaseQAError(
                f"Valid duplicate candidates have different routing evidence: "
                f"{readout_id}"
            )
        if len(set(candidate_language_counts.values())) != 1:
            raise Stage1ReleaseQAError(
                f"Valid duplicate candidates have different routing coverage: "
                f"{readout_id}"
            )
        checks.append(
            {
                "check_id": f"duplicate_candidate_routing/{readout_id}",
                "category": "duplicate_candidate_routing",
                "status": "pass",
                "candidate_count": len(rows),
                "language_count": next(iter(candidate_language_counts.values())),
                "routing_sha256": next(iter(candidate_hashes.values())),
                "candidate_paths_json": _json_cell(sorted(candidate_hashes)),
                "provenance_class": "derived",
            }
        )
    return QAResult(
        summary={
            "status": "pass",
            "duplicate_readout_count": len(duplicate_groups),
            "candidate_count": candidate_count,
            "language_routing_payload_count": language_payload_count,
            "provenance_class": "derived",
        },
        checks=tuple(checks),
    )


__all__ = [
    "LanguageDimensions",
    "QAResult",
    "Stage1ReleaseQAError",
    "build_language_dimensions",
    "build_validation_cohorts",
    "reconstruct_and_verify_aggregates",
    "validate_duplicate_candidate_routing",
    "validate_routing_rows",
    "validate_validation_identities",
]
