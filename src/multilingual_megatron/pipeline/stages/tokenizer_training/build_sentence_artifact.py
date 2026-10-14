#!/usr/bin/env python3
"""Build the frozen alpha-smoothed sentence artifact for tokenizer training.

The input manifest names exactly one *train* JSONL per language.  Declared
validation paths are used only for path-overlap checks and are never opened.
Production mode requires the frozen K200 map and exactly 100,000,000 output
sentences; smaller fixtures are available only with ``--test-mode``.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from multilingual_megatron.pipeline.tokenizer_sentence_contract import (
    ACQUISITION_PLAN_TOOL_ID,
    SENTENCE_SLOT_COUNTER_ID,
    SPLITTER_ID,
    TokenizerSentenceContractError,
    sentence_slot_gate_contract,
    sentence_splitter_contract,
    split_sentence_like,
    tokenizer_acquisition_target_contract,
    verify_acquisition_plan,
)

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    alpha_probabilities,
    canonical_json_sha256,
    largest_remainder_quotas,
    sha256_file,
)


TOOL_ID = "tokenizer_sentence_artifact_v1"
SCHEMA_VERSION = 1
PRODUCTION_LANGUAGE_COUNT = 200
PRODUCTION_TARGET_SENTENCES = 100_000_000
PRODUCTION_ALPHA = 0.3
DEFAULT_MAX_SENTENCE_CHARS = 60_000
DEFAULT_ACQUISITION_SAFETY_MARGIN = 1.10
INTERLEAVER_ID = "smooth_weighted_finish_time_heap_v1"
WITHIN_LANGUAGE_DRAW_ID = "seeded_sha256_rehydrated_sentence_permutation_v1"
AVAILABILITY_ESTIMATOR_ID = "pinned_documents_times_bootstrap_sentences_per_train_document_v1"
_VIRTUAL_TIME_SCALE = 1 << 64
_DOC_ID_PARTITION_COUNT = 256
_SENTENCE_HASH_BUCKET_COUNT = 256
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SentenceArtifactError = TokenizerSentenceContractError


@dataclass(frozen=True, slots=True)
class TrainInput:
    language: str
    path: Path
    expected_sha256: str
    expected_size_bytes: int
    bootstrap_report_path: Path
    bootstrap_report_sha256: str
    bootstrap_report_payload_sha256: str


class _DocumentIdAudit:
    """Disk-partitioned exact duplicate check without an optional DB runtime."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.handles: dict[int, BinaryIO] = {}

    def add(self, *, doc_id: str, language: str) -> None:
        partition = hashlib.sha256(doc_id.encode("utf-8")).digest()[0]
        handle = self.handles.get(partition)
        if handle is None:
            path = self.root / f"doc-ids-{partition:03d}.jsonl"
            handle = path.open("ab")
            self.handles[partition] = handle
        handle.write(_canonical_line({"doc_id": doc_id, "language": language}))

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def validate_unique(self) -> None:
        self.close()
        for partition in range(_DOC_ID_PARTITION_COUNT):
            path = self.root / f"doc-ids-{partition:03d}.jsonl"
            if not path.exists():
                continue
            seen: dict[str, str] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    doc_id = str(row["doc_id"])
                    language = str(row["language"])
                    previous = seen.get(doc_id)
                    if previous is not None:
                        raise SentenceArtifactError(
                            f"overlapping/duplicate doc_id {doc_id!r}: "
                            f"first={previous}, again={language}"
                        )
                    seen[doc_id] = language


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SentenceArtifactError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SentenceArtifactError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SentenceArtifactError(f"{label} must be a JSON object: {path}")
    return payload


def _load_languages(path: Path, *, test_mode: bool) -> tuple[str, ...]:
    payload = _load_object(path, label="language map")
    raw_languages: list[Any] = []
    if isinstance(payload.get("languages"), list):
        raw_languages = payload["languages"]
    else:
        for key in sorted((key for key in payload if str(key).isdigit()), key=int):
            group = payload[key]
            if not isinstance(group, dict) or not isinstance(group.get("languages"), list):
                raise SentenceArtifactError(f"invalid language group {key!r}: {path}")
            raw_languages.extend(group["languages"])
    languages: list[str] = []
    for value in raw_languages:
        language = str(value)
        if not language or language != language.strip():
            raise SentenceArtifactError("language IDs must be non-blank and have no outer whitespace")
        if language in languages:
            raise SentenceArtifactError(f"duplicate language in language map: {language}")
        languages.append(language)
    if not languages:
        raise SentenceArtifactError("language map contains no languages")
    if not test_mode and len(languages) != PRODUCTION_LANGUAGE_COUNT:
        raise SentenceArtifactError(
            f"production language map must contain exactly {PRODUCTION_LANGUAGE_COUNT} languages"
        )
    return tuple(languages)


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise SentenceArtifactError(f"{label} must be a lowercase SHA-256")
    return digest


def _resolve_path(manifest_path: Path, value: Any, *, label: str, require_file: bool) -> Path:
    text = str(value or "").strip()
    if not text:
        raise SentenceArtifactError(f"{label}.path must be set")
    path = Path(text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise SentenceArtifactError(f"{label} does not exist: {path}")
    if path.suffix != ".jsonl":
        raise SentenceArtifactError(f"{label} must be an uncompressed .jsonl file")
    return path


def _parse_inputs(
    manifest_path: Path,
    *,
    languages: tuple[str, ...],
) -> tuple[list[TrainInput], dict[str, list[str]], dict[str, dict[str, Any]]]:
    payload = _load_object(manifest_path, label="bootstrap train manifest")
    if payload.get("schema_version") != 1 or payload.get("tool_id") != "fineweb_ab_v2_bootstrap_v1":
        raise SentenceArtifactError(
            "input manifest must be bootstrap_summary.json from fineweb_ab_v2_bootstrap_v1"
        )
    summary_hash = payload.get("summary_sha256")
    summary_without_hash = {key: value for key, value in payload.items() if key != "summary_sha256"}
    if summary_hash != canonical_json_sha256(summary_without_hash):
        raise SentenceArtifactError("bootstrap summary payload hash is corrupt")
    raw_languages = payload.get("languages")
    if not isinstance(raw_languages, list):
        raise SentenceArtifactError("bootstrap summary languages must be a list")
    provided_order = tuple(map(str, raw_languages))
    if provided_order != languages:
        required = set(languages)
        provided = set(provided_order)
        raise SentenceArtifactError(
            "bootstrap language mismatch: "
            f"missing={sorted(required - provided)}, unexpected={sorted(provided - required)}, "
            "or ordering differs from the frozen language map"
        )
    expected_report_hashes = payload.get("report_payload_sha256")
    if not isinstance(expected_report_hashes, dict) or set(expected_report_hashes) != set(languages):
        raise SentenceArtifactError("bootstrap summary report_payload_sha256 coverage is incomplete")

    train_inputs: list[TrainInput] = []
    validation_paths: dict[str, list[str]] = {}
    report_audit: dict[str, dict[str, Any]] = {}
    resolved_train_paths: dict[Path, str] = {}
    resolved_validation_paths: dict[Path, str] = {}
    for language in languages:
        report_path = (manifest_path.parent / language / "bootstrap_report.json").resolve()
        report = _load_object(report_path, label=f"bootstrap report for {language}")
        if report.get("schema_version") != 1 or report.get("tool_id") != "fineweb_ab_v2_bootstrap_v1":
            raise SentenceArtifactError(f"invalid bootstrap report schema for {language}")
        if report.get("status") != "complete":
            raise SentenceArtifactError(f"bootstrap report is not complete for {language}")
        report_payload_hash = report.get("report_payload_sha256")
        report_without_hash = {
            key: value for key, value in report.items() if key != "report_payload_sha256"
        }
        if report_payload_hash != canonical_json_sha256(report_without_hash):
            raise SentenceArtifactError(f"bootstrap report payload hash is corrupt for {language}")
        if report_payload_hash != expected_report_hashes[language]:
            raise SentenceArtifactError(f"bootstrap summary/report hash mismatch for {language}")
        run_config = report.get("run_config")
        if not isinstance(run_config, dict) or str(run_config.get("language", "")) != language:
            raise SentenceArtifactError(f"bootstrap report language mismatch for {language}")
        outputs = report.get("outputs")
        if not isinstance(outputs, dict):
            raise SentenceArtifactError(f"bootstrap report outputs are missing for {language}")
        train = outputs.get("train.jsonl")
        if not isinstance(train, dict):
            raise SentenceArtifactError(f"bootstrap report for {language} is missing train.jsonl")
        train_path = _resolve_path(
            report_path,
            train.get("path"),
            label=f"{language}.outputs.train.jsonl",
            require_file=True,
        )
        if train_path in resolved_train_paths:
            raise SentenceArtifactError(
                f"train path overlap between {resolved_train_paths[train_path]} and {language}: {train_path}"
            )
        resolved_train_paths[train_path] = language
        train_inputs.append(
            TrainInput(
                language=language,
                path=train_path,
                expected_sha256=_sha256(
                    train.get("sha256"),
                    label=f"{language}.outputs.train.jsonl.sha256",
                ),
                expected_size_bytes=int(train.get("bytes", -1)),
                bootstrap_report_path=report_path,
                bootstrap_report_sha256=sha256_file(report_path),
                bootstrap_report_payload_sha256=str(report_payload_hash),
            )
        )
        if train_inputs[-1].expected_size_bytes < 0:
            raise SentenceArtifactError(
                f"bootstrap report for {language} has invalid train.jsonl byte count"
            )

        declared: list[str] = []
        for split_name in ("validation_reserve.jsonl",):
            split = outputs.get(split_name)
            if not isinstance(split, dict):
                raise SentenceArtifactError(f"bootstrap report for {language} is missing {split_name}")
            validation_path = _resolve_path(
                report_path,
                split.get("path"),
                label=f"{language}.outputs.{split_name}",
                require_file=False,
            )
            declared.append(str(validation_path))
            resolved_validation_paths[validation_path] = f"{language}.{split_name}"
        validation_paths[language] = declared
        run_settings = run_config.get("settings")
        if not isinstance(run_settings, dict):
            raise SentenceArtifactError(f"bootstrap report settings are missing for {language}")
        try:
            legacy_validation_fraction = float(run_settings["validation_fraction"])
            validation_fraction = float(
                run_settings["validation_candidate_fraction"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SentenceArtifactError(
                f"bootstrap validation candidate fraction is invalid for {language}"
            ) from exc
        candidate_contract = run_config.get("validation_candidate_contract")
        if (
            not 0.0 < validation_fraction < 1.0
            or legacy_validation_fraction != validation_fraction
            or candidate_contract
            != {
                "membership_namespace": "validation-reserve-v1",
                "excluded_from_tokenizer_training": True,
                "unused_candidates_return_to_model_training_selection_pool": True,
                "preserved_train_rehydration_multiplier": True,
            }
        ):
            raise SentenceArtifactError(
                f"bootstrap validation candidate contract is invalid for {language}"
            )
        targets = report.get("targets")
        if not isinstance(targets, dict):
            raise SentenceArtifactError(f"bootstrap target audit is missing for {language}")
        availability_policy = run_config.get("availability_policy")
        if not isinstance(availability_policy, dict):
            raise SentenceArtifactError(
                f"bootstrap source-availability audit is missing for {language}"
            )
        full_source_documents: int | None = None
        if language == "eng_Latn":
            lineage = report.get("lineage")
            if not isinstance(lineage, dict):
                raise SentenceArtifactError("English bootstrap lineage is missing")
            source_lock_path = Path(str(lineage.get("source_lock_path", ""))).resolve()
            if (
                not source_lock_path.is_file()
                or sha256_file(source_lock_path) != run_config.get("source_lock_sha256")
            ):
                raise SentenceArtifactError(
                    "English bootstrap source-lock path/hash is invalid"
                )
            source_lock = _load_object(source_lock_path, label="English source lock")
            english = source_lock.get("english_corpus")
            raw_availability = english.get("availability") if isinstance(english, dict) else None
            if isinstance(raw_availability, dict):
                try:
                    full_source_documents = int(raw_availability["reported_documents"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise SentenceArtifactError(
                        "English reported document availability is invalid"
                    ) from exc
        else:
            try:
                full_source_documents = int(
                    availability_policy["pinned_available_documents"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SentenceArtifactError(
                    f"pinned document availability is invalid for {language}"
                ) from exc
        if full_source_documents is None or full_source_documents <= 0:
            raise SentenceArtifactError(
                f"full pinned document availability is missing for {language}"
            )
        reported_counts = report.get("counts")
        if not isinstance(reported_counts, dict):
            raise SentenceArtifactError(
                f"bootstrap report counts are missing for {language}"
            )

        report_audit[language] = {
            "path": str(report_path),
            "file_sha256": sha256_file(report_path),
            "report_payload_sha256": report_payload_hash,
            "run_config_sha256": report.get("run_config_sha256"),
            "availability_inputs": {
                "pinned_full_source_documents": full_source_documents,
                "validation_fraction": validation_fraction,
                "complete_source_scan": bool(
                    targets.get("source_exhausted")
                    or targets.get("pinned_inventory_exhausted")
                ),
                "true_source_exhausted": bool(targets.get("source_exhausted")),
            },
            "pilot_requested_settings": targets.get("requested"),
            "incremental_extension": run_config.get("incremental_extension"),
            "tokenizer_sentence_acquisition_target": run_config.get(
                "tokenizer_sentence_acquisition_target"
            ),
            "reported_tokenizer_sentence_slot_gate": targets.get(
                "tokenizer_sentence_slot_gate"
            ),
            "reported_unique_train_sentences": reported_counts.get(
                "unique_train_sentences"
            ),
            "reported_rehydrated_train_sentence_slots": reported_counts.get(
                "rehydrated_train_sentence_slots"
            ),
        }

    overlaps = sorted(set(resolved_train_paths) & set(resolved_validation_paths))
    if overlaps:
        details = [
            f"{resolved_train_paths[path]} train == {resolved_validation_paths[path]}"
            for path in overlaps
        ]
        raise SentenceArtifactError("train/validation path overlap: " + "; ".join(details))
    return train_inputs, validation_paths, report_audit


def _row_doc_id(row: Mapping[str, Any], *, label: str) -> str:
    value = row.get("document_id")
    if not isinstance(value, str) or not value or value != value.strip():
        raise SentenceArtifactError(
            f"{label}.document_id must be a non-blank stable string"
        )
    return value


def _check_row_train_language(row: Mapping[str, Any], *, language: str, label: str) -> None:
    lineage = row.get("lineage")
    if not isinstance(lineage, dict):
        raise SentenceArtifactError(f"{label} has no finalized bootstrap lineage object")
    declared_language = row.get("language")
    if str(declared_language or "") != language or str(lineage.get("subset", "")) != language:
        raise SentenceArtifactError(
            f"{label} language mismatch: expected {language}, got {declared_language}"
        )
    declared_split = lineage.get("split")
    if str(declared_split or "") != "train":
        raise SentenceArtifactError(f"{label} is not a train row: split={declared_split!r}")
    if row.get("artifact_role") != "train":
        raise SentenceArtifactError(
            f"{label} is not a train artifact: artifact_role={row.get('artifact_role')!r}"
        )


def _canonical_line(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _spool_language(
    spec: TrainInput,
    *,
    spool_path: Path,
    document_audit: _DocumentIdAudit,
    max_chars: int,
    seed: int,
    materialize_spool: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    document_count = 0
    sentence_count = 0
    unique_sentence_count = 0
    text_utf8_bytes = 0
    # A full in-memory shuffle is not viable for the production pool.  Hash
    # partitioning followed by an in-bucket sort is an exact SHA-256 ordering
    # with bounded memory (one 1/256 bucket for one language at a time).
    with tempfile.TemporaryDirectory(
        prefix=f".{spec.language}-sentence-hash-",
        dir=spool_path.parent,
    ) as bucket_dir_text:
        bucket_dir = Path(bucket_dir_text)
        bucket_handles: dict[int, BinaryIO] = {}
        try:
            with spec.path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    digest.update(raw_line)
                    size_bytes += len(raw_line)
                    label = f"{spec.language}:{spec.path.name}:{line_number}"
                    if not raw_line.strip():
                        raise SentenceArtifactError(f"blank input line: {label}")
                    try:
                        row = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SentenceArtifactError(f"invalid UTF-8 JSON object: {label}") from exc
                    if not isinstance(row, dict):
                        raise SentenceArtifactError(f"input row is not an object: {label}")
                    _check_row_train_language(row, language=spec.language, label=label)
                    doc_id = _row_doc_id(row, label=label)
                    document_audit.add(doc_id=doc_id, language=spec.language)
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise SentenceArtifactError(f"blank or missing text: {label}")
                    text_utf8_bytes += len(text.encode("utf-8"))
                    segments = list(split_sentence_like(text, max_chars=max_chars))
                    if not segments:
                        raise SentenceArtifactError(f"sentence splitter produced no text: {label}")
                    lineage = row.get("lineage")
                    if not isinstance(lineage, dict):
                        raise SentenceArtifactError(f"missing finalized bootstrap lineage: {label}")
                    multiplier = lineage.get("rehydration_multiplier")
                    if (
                        not isinstance(multiplier, int)
                        or isinstance(multiplier, bool)
                        or multiplier < 1
                    ):
                        raise SentenceArtifactError(f"invalid rehydration_multiplier: {label}")
                    for sentence_index, sentence in enumerate(segments):
                        encoded = _canonical_line(
                            {
                                "doc_id": doc_id,
                                "language": spec.language,
                                "sentence_index": sentence_index,
                                "text": sentence,
                            }
                        )
                        for replica_index in range(multiplier):
                            if materialize_spool:
                                order_digest = hashlib.sha256(
                                    (
                                        f"{int(seed)}\0{WITHIN_LANGUAGE_DRAW_ID}\0{spec.language}\0"
                                        f"{doc_id}\0{sentence_index}\0{replica_index}"
                                    ).encode("utf-8")
                                ).digest()
                                bucket_index = order_digest[0]
                                handle = bucket_handles.get(bucket_index)
                                if handle is None:
                                    handle = (bucket_dir / f"{bucket_index:03d}.rows").open("ab")
                                    bucket_handles[bucket_index] = handle
                                handle.write(order_digest.hex().encode("ascii") + b"\t" + encoded)
                            sentence_count += 1
                        unique_sentence_count += 1
                    document_count += 1
        finally:
            for handle in bucket_handles.values():
                handle.close()

        if materialize_spool:
            with spool_path.open("wb") as spool:
                for bucket_index in range(_SENTENCE_HASH_BUCKET_COUNT):
                    bucket_path = bucket_dir / f"{bucket_index:03d}.rows"
                    if not bucket_path.is_file():
                        continue
                    rows = bucket_path.read_bytes().splitlines(keepends=True)
                    rows.sort()
                    for raw in rows:
                        separator = raw.find(b"\t")
                        if separator != 64:
                            raise AssertionError(f"invalid seeded sentence bucket: {bucket_path}")
                        spool.write(raw[separator + 1 :])
                spool.flush()
                os.fsync(spool.fileno())

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != spec.expected_sha256:
        raise SentenceArtifactError(
            f"input SHA-256 mismatch for {spec.language}: "
            f"expected {spec.expected_sha256}, got {actual_sha256}"
        )
    if size_bytes != spec.expected_size_bytes:
        raise SentenceArtifactError(
            f"input byte-size mismatch for {spec.language}: "
            f"expected {spec.expected_size_bytes}, got {size_bytes}"
        )
    if document_count == 0 or sentence_count == 0:
        raise SentenceArtifactError(f"missing train documents/sentences for {spec.language}")
    return {
        "path": str(spec.path),
        "expected_sha256": spec.expected_sha256,
        "actual_sha256": actual_sha256,
        "size_bytes": size_bytes,
        "document_count": document_count,
        "text_utf8_bytes": text_utf8_bytes,
        "unique_sentence_count": unique_sentence_count,
        "rehydrated_sentence_count": sentence_count,
        "bootstrap_report": {
            "path": str(spec.bootstrap_report_path),
            "file_sha256": spec.bootstrap_report_sha256,
            "report_payload_sha256": spec.bootstrap_report_payload_sha256,
        },
    }


def _round_fraction_half_up(value: Fraction) -> int:
    if value <= 0:
        raise SentenceArtifactError("sentence availability estimate must be positive")
    numerator = value.numerator
    denominator = value.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def _sentence_availability(
    *,
    language: str,
    input_audit: Mapping[str, Any],
    report_audit: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Estimate the full post-split unique sentence inventory for alpha q_l.

    Exact full-corpus sentence counts would require preprocessing every source
    before tokenizer training.  For an exhausted source we already have that
    exact count.  Otherwise the frozen estimator scales the deterministic
    bootstrap's unique train sentences per train document to the pinned full
    document inventory and applies the frozen document-level validation share.
    """

    observed_documents = int(input_audit["document_count"])
    observed_sentences = int(input_audit["unique_sentence_count"])
    raw_inputs = report_audit.get("availability_inputs")
    if not isinstance(raw_inputs, dict):
        raise SentenceArtifactError(f"availability inputs are missing for {language}")
    full_documents = int(raw_inputs["pinned_full_source_documents"])
    validation_fraction = Fraction(str(raw_inputs["validation_fraction"]))
    complete_source_scan = bool(raw_inputs["complete_source_scan"])
    if observed_documents <= 0 or observed_sentences <= 0 or full_documents <= 0:
        raise SentenceArtifactError(f"invalid sentence availability inputs for {language}")
    if complete_source_scan:
        estimated = observed_sentences
        basis = "complete_source_scan_exact_post_split_sentences"
        formula = "observed_unique_train_sentences"
    else:
        train_fraction = Fraction(1, 1) - validation_fraction
        estimated_fraction = (
            full_documents
            * train_fraction
            * Fraction(observed_sentences, observed_documents)
        )
        estimated = max(1, _round_fraction_half_up(estimated_fraction))
        basis = "pinned_full_documents_times_bootstrap_train_sentences_per_document"
        formula = (
            "round_half_up(pinned_full_source_documents * (1-validation_fraction) "
            "* observed_unique_train_sentences / observed_train_documents)"
        )
    return estimated, {
        "estimator_id": AVAILABILITY_ESTIMATOR_ID,
        "status": "exact" if complete_source_scan else "estimated",
        "basis": basis,
        "formula": formula,
        "pinned_full_source_documents": full_documents,
        "validation_fraction": float(validation_fraction),
        "observed_train_documents": observed_documents,
        "observed_unique_train_sentences": observed_sentences,
        "complete_source_scan": complete_source_scan,
        "estimated_available_post_split_unique_sentences": estimated,
        "rehydration_used_in_language_level_availability": False,
    }


def _inspect_bootstrap_pool(
    *,
    input_manifest_path: Path,
    language_map_path: Path,
    temporary_root: Path,
    max_sentence_chars: int,
    seed: int,
    test_mode: bool,
    materialize_spools: bool,
) -> tuple[
    tuple[str, ...],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Path],
]:
    languages = _load_languages(language_map_path, test_mode=test_mode)
    train_inputs, validation_paths, report_audit = _parse_inputs(
        input_manifest_path,
        languages=languages,
    )
    document_audit = _DocumentIdAudit(temporary_root)
    input_audit: dict[str, dict[str, Any]] = {}
    spool_paths: dict[str, Path] = {}
    try:
        for index, spec in enumerate(train_inputs):
            spool_path = temporary_root / f"{index:03d}.jsonl"
            input_audit[spec.language] = _spool_language(
                spec,
                spool_path=spool_path,
                document_audit=document_audit,
                max_chars=max_sentence_chars,
                seed=seed,
                materialize_spool=materialize_spools,
            )
            spool_paths[spec.language] = spool_path
        document_audit.validate_unique()
    finally:
        document_audit.close()
    return languages, validation_paths, report_audit, input_audit, spool_paths


def _ceil_fraction(value: Fraction) -> int:
    if value <= 0:
        raise SentenceArtifactError("acquisition target must be positive")
    return (value.numerator + value.denominator - 1) // value.denominator


def build_sentence_acquisition_plan(
    *,
    input_manifest_path: Path,
    language_map_path: Path,
    target_sentences: int,
    alpha: float = PRODUCTION_ALPHA,
    seed: int = 42,
    max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
    safety_margin: float = DEFAULT_ACQUISITION_SAFETY_MARGIN,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Plan a second immutable acquisition large enough for the 100M draw."""

    input_manifest_path = input_manifest_path.resolve()
    language_map_path = language_map_path.resolve()
    if not isinstance(target_sentences, int) or isinstance(target_sentences, bool) or target_sentences <= 0:
        raise SentenceArtifactError("target_sentences must be a positive integer")
    if not test_mode and target_sentences != PRODUCTION_TARGET_SENTENCES:
        raise SentenceArtifactError(
            f"production target_sentences must equal {PRODUCTION_TARGET_SENTENCES}"
        )
    if not math.isclose(float(alpha), PRODUCTION_ALPHA, rel_tol=0.0, abs_tol=0.0):
        raise SentenceArtifactError(f"alpha must equal the frozen production value {PRODUCTION_ALPHA}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SentenceArtifactError("seed must be an integer")
    if not test_mode and seed != 42:
        raise SentenceArtifactError("production tokenizer sentence draw seed must equal 42")
    safety = Fraction(str(safety_margin))
    if safety < 1:
        raise SentenceArtifactError("acquisition safety_margin must be at least one")

    scratch_parent = input_manifest_path.parent
    with tempfile.TemporaryDirectory(
        prefix="tokenizer-sentence-acquisition-plan-",
        dir=scratch_parent,
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        languages, validation_paths, report_audit, input_audit, _ = _inspect_bootstrap_pool(
            input_manifest_path=input_manifest_path,
            language_map_path=language_map_path,
            temporary_root=temporary_root,
            max_sentence_chars=max_sentence_chars,
            seed=seed,
            test_mode=test_mode,
            materialize_spools=False,
        )
        availability: dict[str, int] = {}
        availability_audits: dict[str, dict[str, Any]] = {}
        for language in languages:
            availability[language], availability_audits[language] = _sentence_availability(
                language=language,
                input_audit=input_audit[language],
                report_audit=report_audit[language],
            )
        probabilities = alpha_probabilities(availability, alpha=alpha)
        quotas = largest_remainder_quotas(target_sentences, probabilities)

        language_plans: dict[str, Any] = {}
        for language in languages:
            observed_documents = int(input_audit[language]["document_count"])
            observed_slots = int(input_audit[language]["rehydrated_sentence_count"])
            observed_text_bytes = int(input_audit[language]["text_utf8_bytes"])
            raw_availability = report_audit[language]["availability_inputs"]
            validation_fraction = Fraction(str(raw_availability["validation_fraction"]))
            train_fraction = Fraction(1, 1) - validation_fraction
            requested = report_audit[language].get("pilot_requested_settings")
            if not isinstance(requested, dict):
                raise SentenceArtifactError(
                    f"pilot requested settings are missing for {language}"
                )
            try:
                pilot_target_documents = int(requested["target_unique_documents"])
                pilot_target_bytes = int(requested["target_utf8_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SentenceArtifactError(
                    f"pilot requested settings are invalid for {language}"
                ) from exc
            quota = int(quotas[language])
            target_documents = max(
                pilot_target_documents,
                _ceil_fraction(
                    quota
                    * Fraction(observed_documents, observed_slots)
                    * safety
                    / train_fraction
                ),
            )
            target_bytes = max(
                pilot_target_bytes,
                _ceil_fraction(
                    quota
                    * Fraction(observed_text_bytes, observed_slots)
                    * safety
                    / train_fraction
                ),
            )
            complete_source_scan = bool(raw_availability["complete_source_scan"])
            language_plans[language] = {
                "alpha_probability": probabilities[language],
                "planned_sentence_count": quota,
                "sentence_availability": availability_audits[language],
                "pilot": {
                    "bootstrap_report_payload_sha256": report_audit[language][
                        "report_payload_sha256"
                    ],
                    "observed_train_documents": observed_documents,
                    "observed_unique_train_sentences": int(
                        input_audit[language]["unique_sentence_count"]
                    ),
                    "observed_rehydrated_sentence_slots": observed_slots,
                    "observed_train_text_utf8_bytes": observed_text_bytes,
                    "complete_source_scan": complete_source_scan,
                    "validation_paths_not_read": validation_paths[language],
                },
                "acquisition": {
                    "safety_margin": float(safety),
                    "target_unique_source_documents": target_documents,
                    "target_utf8_bytes": target_bytes,
                    "required_rehydrated_sentence_slots": quota,
                    "document_formula": (
                        "ceil(quota * pilot_train_documents / pilot_rehydrated_sentence_slots "
                        "* safety_margin / (1-validation_fraction))"
                    ),
                    "byte_formula": (
                        "ceil(quota * pilot_train_text_utf8_bytes / "
                        "pilot_rehydrated_sentence_slots * safety_margin / "
                        "(1-validation_fraction))"
                    ),
                    "pilot_pool_already_sufficient": observed_slots >= quota,
                    "true_source_exhausted_repetition_allowed": (
                        complete_source_scan and observed_slots < quota
                    ),
                },
            }

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": ACQUISITION_PLAN_TOOL_ID,
        "status": "complete",
        "test_mode": bool(test_mode),
        "target_sentence_count": int(target_sentences),
        "alpha": float(alpha),
        "seed": int(seed),
        "acquisition_safety_margin": float(safety),
        "language_count": len(languages),
        "ordered_languages": list(languages),
        "ordered_languages_sha256": hashlib.sha256(
            "".join(f"{language}\n" for language in languages).encode("utf-8")
        ).hexdigest(),
        "splitter": sentence_splitter_contract(max_chars=max_sentence_chars),
        "mixture": {
            "availability_unit": (
                "estimated_full_post_split_unique_preprocessed_training_sentences"
            ),
            "availability_estimator_id": AVAILABILITY_ESTIMATOR_ID,
            "rehydration_affects_language_level_probability": False,
            "quota_allocator": "alpha_power_largest_remainder_v1",
        },
        "expanded_pool_gate": sentence_slot_gate_contract(
            max_chars=max_sentence_chars
        ),
        "inputs": {
            "pilot_bootstrap_summary": {
                "path": str(input_manifest_path),
                "sha256": sha256_file(input_manifest_path),
            },
            "language_map": {
                "path": str(language_map_path),
                "sha256": sha256_file(language_map_path),
            },
        },
        "languages": language_plans,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return plan


def write_frozen_acquisition_plan(*, output_path: Path, plan: Mapping[str, Any]) -> None:
    output_path = output_path.resolve()
    if not verify_acquisition_plan(plan):
        raise SentenceArtifactError("refusing to write an invalid acquisition plan")
    if output_path.exists():
        existing = _load_object(output_path, label="existing tokenizer acquisition plan")
        if existing != dict(plan):
            raise SentenceArtifactError(
                f"refusing to replace a different frozen acquisition plan: {output_path}"
            )
        return
    _atomic_write_manifest(output_path, plan)


def _load_acquisition_plan_contract(
    *,
    path: Path,
    language_map_path: Path,
    languages: tuple[str, ...],
    target_sentences: int,
    alpha: float,
    seed: int,
    max_sentence_chars: int,
) -> dict[str, Any]:
    path = path.resolve()
    plan = _load_object(path, label="tokenizer sentence acquisition plan")
    if not verify_acquisition_plan(plan):
        raise SentenceArtifactError("tokenizer sentence acquisition plan hash/schema is invalid")
    expected_order_hash = hashlib.sha256(
        "".join(f"{language}\n" for language in languages).encode("utf-8")
    ).hexdigest()
    mismatches: list[str] = []
    if plan.get("target_sentence_count") != target_sentences:
        mismatches.append("target_sentence_count")
    if not math.isclose(float(plan.get("alpha", -1.0)), float(alpha), rel_tol=0.0, abs_tol=0.0):
        mismatches.append("alpha")
    if plan.get("seed") != seed:
        mismatches.append("seed")
    if plan.get("ordered_languages") != list(languages):
        mismatches.append("ordered_languages")
    if plan.get("ordered_languages_sha256") != expected_order_hash:
        mismatches.append("ordered_languages_sha256")
    if plan.get("inputs", {}).get("language_map", {}).get("sha256") != sha256_file(
        language_map_path
    ):
        mismatches.append("language_map_sha256")
    if plan.get("splitter") != sentence_splitter_contract(
        max_chars=max_sentence_chars
    ):
        mismatches.append("splitter")
    raw_languages = plan.get("languages")
    if not isinstance(raw_languages, dict) or set(raw_languages) != set(languages):
        mismatches.append("languages")
    if mismatches:
        raise SentenceArtifactError(
            "tokenizer sentence acquisition plan contract drift: " + ", ".join(mismatches)
        )
    return plan


def _frozen_plan_mixture_and_gate(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    languages: tuple[str, ...],
    report_audit: Mapping[str, Mapping[str, Any]],
    input_audit: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, float],
    dict[str, int],
    dict[str, dict[str, Any]],
]:
    raw_languages = plan["languages"]
    availability: dict[str, int] = {}
    availability_audits: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, float] = {}
    quotas: dict[str, int] = {}
    pool_gate: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for language in languages:
        raw = raw_languages[language]
        if not isinstance(raw, dict):
            raise SentenceArtifactError(f"acquisition plan language entry is invalid: {language}")
        raw_availability = raw.get("sentence_availability")
        raw_pilot = raw.get("pilot")
        raw_target = raw.get("acquisition")
        if not all(isinstance(value, dict) for value in (raw_availability, raw_pilot, raw_target)):
            raise SentenceArtifactError(f"acquisition plan language lineage is invalid: {language}")
        estimated = int(raw_availability["estimated_available_post_split_unique_sentences"])
        probability = float(raw["alpha_probability"])
        quota = int(raw["planned_sentence_count"])
        if estimated <= 0 or not math.isfinite(probability) or probability <= 0 or quota <= 0:
            raise SentenceArtifactError(f"acquisition plan mixture is invalid: {language}")
        availability[language] = estimated
        availability_audits[language] = dict(raw_availability)
        probabilities[language] = probability
        quotas[language] = quota

        target_lineage = report_audit[language].get(
            "tokenizer_sentence_acquisition_target"
        )
        incremental = report_audit[language].get("incremental_extension")
        expected_target = tokenizer_acquisition_target_contract(
            plan=plan,
            plan_file_sha256=sha256_file(plan_path),
            language=language,
        )
        if target_lineage != expected_target:
            raise SentenceArtifactError(
                f"expanded bootstrap was not acquired from the frozen tokenizer plan: {language}"
            )
        if (
            not isinstance(incremental, dict)
            or incremental.get("predecessor_report_payload_sha256")
            != raw_pilot.get("bootstrap_report_payload_sha256")
            or incremental.get("mode") != "verified_full_deterministic_rebuild"
        ):
            raise SentenceArtifactError(
                f"expanded bootstrap is not a verified immutable rebuild of the pilot: {language}"
            )

        actual_slots = int(input_audit[language]["rehydrated_sentence_count"])
        actual_unique_sentences = int(input_audit[language]["unique_sentence_count"])
        true_source_exhausted = bool(
            report_audit[language]["availability_inputs"]["true_source_exhausted"]
        )
        sufficient = actual_slots >= quota
        shortfall_slots = max(0, quota - actual_slots)
        true_exhausted_repetition = not sufficient and true_source_exhausted
        if not sufficient and not true_source_exhausted:
            blockers.append(language)
        gate_contract = expected_target["sentence_slot_gate"]
        expected_report_gate = {
            "counter_id": gate_contract["counter_id"],
            "counted_split": gate_contract["counted_split"],
            "slot_formula": gate_contract["slot_formula"],
            "splitter": gate_contract["splitter"],
            "non_exhausted_stop_rule": gate_contract[
                "non_exhausted_stop_rule"
            ],
            "required_rehydrated_sentence_slots": quota,
            "actual_unique_train_sentences": actual_unique_sentences,
            "actual_rehydrated_sentence_slots": actual_slots,
            "shortfall_slots": shortfall_slots,
            "quota_satisfied": sufficient,
            "true_pinned_source_exhausted": true_source_exhausted,
            "true_source_exhausted_repetition_allowed": (
                true_exhausted_repetition
            ),
            "artificial_pool_cycling_forbidden": True,
        }
        if (
            report_audit[language].get("reported_unique_train_sentences")
            != actual_unique_sentences
            or report_audit[language].get(
                "reported_rehydrated_train_sentence_slots"
            )
            != actual_slots
            or report_audit[language].get(
                "reported_tokenizer_sentence_slot_gate"
            )
            != expected_report_gate
        ):
            raise SentenceArtifactError(
                f"expanded bootstrap tokenizer sentence-slot recount drifted: {language}"
            )
        pool_gate[language] = {
            "status": (
                "sufficient"
                if sufficient
                else (
                    "true_source_exhausted_repetition_required"
                    if true_exhausted_repetition
                    else "insufficient_artificial_cycle_forbidden"
                )
            ),
            "required_rehydrated_sentence_slots": quota,
            "actual_rehydrated_sentence_slots": actual_slots,
            "shortfall_slots": shortfall_slots,
            "complete_source_scan": true_source_exhausted,
            "artificial_cycle_forbidden": True,
            "true_source_exhausted_repetition_allowed": true_exhausted_repetition,
        }
    if not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SentenceArtifactError("acquisition plan probabilities do not sum to one")
    if sum(quotas.values()) != int(plan["target_sentence_count"]):
        raise SentenceArtifactError("acquisition plan sentence quotas do not preserve the total")
    recomputed = alpha_probabilities(availability, alpha=float(plan["alpha"]))
    if any(
        not math.isclose(probabilities[language], recomputed[language], rel_tol=0.0, abs_tol=1e-15)
        for language in languages
    ):
        raise SentenceArtifactError("acquisition plan probabilities disagree with frozen availability")
    if quotas != largest_remainder_quotas(int(plan["target_sentence_count"]), probabilities):
        raise SentenceArtifactError("acquisition plan quotas disagree with frozen probabilities")
    if blockers:
        raise SentenceArtifactError(
            "expanded tokenizer pool would require artificial cycling; rebuild with more "
            "source documents for: " + ", ".join(blockers)
        )
    return availability, availability_audits, probabilities, quotas, pool_gate


def _next_spooled_sentence(handle: BinaryIO, *, spool_path: Path) -> dict[str, Any]:
    raw = handle.readline()
    if not raw:
        handle.seek(0)
        raw = handle.readline()
    if not raw:
        raise AssertionError(f"empty sentence spool: {spool_path}")
    row = json.loads(raw)
    if not isinstance(row, dict):
        raise AssertionError(f"invalid internal sentence spool: {spool_path}")
    return row


def _write_interleaved(
    *,
    output_path: Path,
    spool_paths: Mapping[str, Path],
    sampling_slots: Mapping[str, int],
    quotas: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    handles: dict[str, BinaryIO] = {}
    output_digest = hashlib.sha256()
    output_size = 0
    emitted = {language: 0 for language in quotas}
    repeated = {language: 0 for language in quotas}
    heap: list[tuple[int, str]] = []
    for language, quota in sorted(quotas.items()):
        if quota > 0:
            handles[language] = spool_paths[language].open("rb")
            heapq.heappush(heap, (_VIRTUAL_TIME_SCALE // quota, language))
    try:
        with os.fdopen(fd, "wb") as output:
            while heap:
                _, language = heapq.heappop(heap)
                sentence = _next_spooled_sentence(handles[language], spool_path=spool_paths[language])
                repeat_index = emitted[language] // sampling_slots[language]
                payload = {
                    "doc_id": sentence["doc_id"],
                    "language": language,
                    "repeat_index": repeat_index,
                    "sentence_index": sentence["sentence_index"],
                    "text": sentence["text"],
                }
                encoded = _canonical_line(payload)
                output.write(encoded)
                output_digest.update(encoded)
                output_size += len(encoded)
                emitted[language] += 1
                if repeat_index > 0:
                    repeated[language] += 1
                if emitted[language] < quotas[language]:
                    due = ((emitted[language] + 1) * _VIRTUAL_TIME_SCALE) // quotas[language]
                    heapq.heappush(heap, (due, language))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    finally:
        for handle in handles.values():
            handle.close()
        if temporary.exists():
            temporary.unlink()

    return (
        {
            "path": str(output_path),
            "sha256": output_digest.hexdigest(),
            "size_bytes": output_size,
            "sentence_count": sum(emitted.values()),
        },
        emitted,
        repeated,
    )


def _atomic_write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_sentence_artifact(
    *,
    input_manifest_path: Path,
    language_map_path: Path,
    output_jsonl_path: Path,
    output_manifest_path: Path,
    target_sentences: int,
    acquisition_plan_path: Path | None = None,
    alpha: float = PRODUCTION_ALPHA,
    seed: int = 42,
    max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Build and hash one deterministic tokenizer-training sentence artifact."""

    input_manifest_path = input_manifest_path.resolve()
    language_map_path = language_map_path.resolve()
    output_jsonl_path = output_jsonl_path.resolve()
    output_manifest_path = output_manifest_path.resolve()
    if output_jsonl_path == output_manifest_path:
        raise SentenceArtifactError("output JSONL and manifest paths must differ")
    if output_jsonl_path.exists() or output_manifest_path.exists():
        raise SentenceArtifactError("refusing to overwrite an existing output artifact")
    if not isinstance(target_sentences, int) or isinstance(target_sentences, bool) or target_sentences <= 0:
        raise SentenceArtifactError("target_sentences must be a positive integer")
    if not test_mode and target_sentences != PRODUCTION_TARGET_SENTENCES:
        raise SentenceArtifactError(
            f"production target_sentences must equal {PRODUCTION_TARGET_SENTENCES}"
        )
    if not math.isclose(float(alpha), PRODUCTION_ALPHA, rel_tol=0.0, abs_tol=0.0):
        raise SentenceArtifactError(f"alpha must equal the frozen production value {PRODUCTION_ALPHA}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SentenceArtifactError("seed must be an integer")
    if not test_mode and seed != 42:
        raise SentenceArtifactError("production tokenizer sentence draw seed must equal 42")
    if not test_mode and acquisition_plan_path is None:
        raise SentenceArtifactError(
            "production tokenizer sentence construction requires a frozen two-pass "
            "acquisition plan"
        )

    languages = _load_languages(language_map_path, test_mode=test_mode)
    acquisition_plan: dict[str, Any] | None = None
    resolved_acquisition_plan_path: Path | None = None
    if acquisition_plan_path is not None:
        resolved_acquisition_plan_path = acquisition_plan_path.resolve()
        acquisition_plan = _load_acquisition_plan_contract(
            path=resolved_acquisition_plan_path,
            language_map_path=language_map_path,
            languages=languages,
            target_sentences=target_sentences,
            alpha=alpha,
            seed=seed,
            max_sentence_chars=max_sentence_chars,
        )
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tokenizer-sentence-spool-",
        dir=output_jsonl_path.parent,
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        (
            inspected_languages,
            validation_paths,
            report_audit,
            input_audit,
            spool_paths,
        ) = _inspect_bootstrap_pool(
            input_manifest_path=input_manifest_path,
            language_map_path=language_map_path,
            temporary_root=temporary_root,
            max_sentence_chars=max_sentence_chars,
            seed=seed,
            test_mode=test_mode,
            materialize_spools=True,
        )
        if inspected_languages != languages:
            raise AssertionError("language ordering changed during bootstrap inspection")
        pool_gate: dict[str, dict[str, Any]] = {}
        if acquisition_plan is not None:
            assert resolved_acquisition_plan_path is not None
            (
                availability,
                availability_audits,
                probabilities,
                quotas,
                pool_gate,
            ) = _frozen_plan_mixture_and_gate(
                plan=acquisition_plan,
                plan_path=resolved_acquisition_plan_path,
                languages=languages,
                report_audit=report_audit,
                input_audit=input_audit,
            )
        else:
            availability_audits = {}
            availability = {}
            for language in languages:
                availability[language], availability_audits[language] = _sentence_availability(
                    language=language,
                    input_audit=input_audit[language],
                    report_audit=report_audit[language],
                )
            probabilities = alpha_probabilities(availability, alpha=alpha)
            quotas = largest_remainder_quotas(target_sentences, probabilities)
        sampling_slots = {
            language: int(input_audit[language]["rehydrated_sentence_count"])
            for language in languages
        }
        output_audit, realized, repeated = _write_interleaved(
            output_path=output_jsonl_path,
            spool_paths=spool_paths,
            sampling_slots=sampling_slots,
            quotas=quotas,
        )

    if output_audit["sentence_count"] != target_sentences:
        raise AssertionError("sentence artifact did not contain the exact target total")
    language_report: dict[str, Any] = {}
    for language in languages:
        quota = quotas[language]
        available = availability[language]
        slots = sampling_slots[language]
        if realized[language] != quota or repeated[language] != max(0, quota - slots):
            raise AssertionError(f"realized/repeated accounting failed for {language}")
        language_report[language] = {
            "estimated_available_post_split_unique_sentence_count": available,
            "alpha_probability": probabilities[language],
            "planned_sentence_count": quota,
            "realized_sentence_count": realized[language],
            "repeated_sentence_count": repeated[language],
            "within_language_rehydrated_sampling_slots": slots,
            "weighted_pool_passes_required": math.ceil(quota / slots) if quota else 0,
            "maximum_repeat_index": ((quota - 1) // slots) if quota else None,
            "sentence_availability": availability_audits[language],
            "expanded_pool_gate": pool_gate.get(
                language,
                {
                    "status": "test_mode_without_acquisition_plan",
                    "actual_rehydrated_sentence_slots": slots,
                },
            ),
            "train_input": input_audit[language],
            "declared_validation_paths_not_read": validation_paths[language],
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "status": "complete",
        "test_mode": bool(test_mode),
        "target_sentence_count": target_sentences,
        "realized_sentence_count": output_audit["sentence_count"],
        "language_count": len(languages),
        "alpha": float(alpha),
        "seed": int(seed),
        "algorithms": {
            "splitter": sentence_splitter_contract(max_chars=max_sentence_chars),
            "quota": {
                "id": "alpha_power_largest_remainder_v1",
                "alpha": float(alpha),
                "availability_unit": (
                    "estimated_full_post_split_unique_preprocessed_training_sentences"
                ),
                "availability_estimator_id": AVAILABILITY_ESTIMATOR_ID,
                "rehydration_affects_language_level_probability": False,
                "cycling_when_quota_exceeds_materialized_weighted_pool": True,
                "hard_repetition_cap": None,
            },
            "within_language_draw": {
                "id": WITHIN_LANGUAGE_DRAW_ID,
                "seed": int(seed),
                "ordering": "ascending_sha256(seed, language, doc_id, sentence_index, replica_index)",
                "rehydration": "integer replica slots affect only within-language selection",
                "hash_bucket_count": _SENTENCE_HASH_BUCKET_COUNT,
            },
            "interleaver": {
                "id": INTERLEAVER_ID,
                "virtual_time_scale": _VIRTUAL_TIME_SCALE,
                "tie_break": "lexicographic_language_id",
            },
        },
        "inputs": {
            "bootstrap_train_manifest": {
                "path": str(input_manifest_path),
                "sha256": sha256_file(input_manifest_path),
            },
            "tokenizer_sentence_acquisition_plan": (
                {
                    "path": str(resolved_acquisition_plan_path),
                    "file_sha256": sha256_file(resolved_acquisition_plan_path),
                    "payload_sha256": acquisition_plan["plan_sha256"],
                }
                if acquisition_plan is not None and resolved_acquisition_plan_path is not None
                else None
            ),
            "language_map": {
                "path": str(language_map_path),
                "sha256": sha256_file(language_map_path),
                "ordered_languages_sha256": hashlib.sha256(
                    "".join(f"{language}\n" for language in languages).encode("utf-8")
                ).hexdigest(),
            },
            "validation_files_opened": False,
            "bootstrap_reports": report_audit,
        },
        "output": output_audit,
        "languages": language_report,
        "totals": {
            "estimated_available_post_split_unique_sentence_count": sum(availability.values()),
            "within_language_rehydrated_sampling_slots": sum(sampling_slots.values()),
            "planned_sentence_count": sum(quotas.values()),
            "realized_sentence_count": sum(realized.values()),
            "repeated_sentence_count": sum(repeated.values()),
        },
    }
    report["manifest_sha256"] = canonical_json_sha256(report)
    _atomic_write_manifest(output_manifest_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--language-map", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--acquisition-plan",
        help="Frozen pilot-derived tokenizer sentence acquisition plan (required in production).",
    )
    parser.add_argument("--target-sentences", type=int, default=PRODUCTION_TARGET_SENTENCES)
    parser.add_argument("--alpha", type=float, default=PRODUCTION_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sentence-chars", type=int, default=DEFAULT_MAX_SENTENCE_CHARS)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args(argv)
    if not args.test_mode and args.target_sentences != PRODUCTION_TARGET_SENTENCES:
        parser.error(
            f"--target-sentences must be {PRODUCTION_TARGET_SENTENCES} unless --test-mode is set"
        )
    report = build_sentence_artifact(
        input_manifest_path=Path(args.input_manifest),
        language_map_path=Path(args.language_map),
        output_jsonl_path=Path(args.output_jsonl),
        output_manifest_path=Path(args.output_manifest),
        target_sentences=args.target_sentences,
        acquisition_plan_path=(
            Path(args.acquisition_plan) if args.acquisition_plan else None
        ),
        alpha=args.alpha,
        seed=args.seed,
        max_sentence_chars=args.max_sentence_chars,
        test_mode=args.test_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_SENTENCE_CHARS",
    "DEFAULT_ACQUISITION_SAFETY_MARGIN",
    "ACQUISITION_PLAN_TOOL_ID",
    "INTERLEAVER_ID",
    "PRODUCTION_ALPHA",
    "PRODUCTION_LANGUAGE_COUNT",
    "PRODUCTION_TARGET_SENTENCES",
    "SENTENCE_SLOT_COUNTER_ID",
    "SPLITTER_ID",
    "WITHIN_LANGUAGE_DRAW_ID",
    "SentenceArtifactError",
    "build_sentence_artifact",
    "build_sentence_acquisition_plan",
    "main",
    "sentence_slot_gate_contract",
    "sentence_splitter_contract",
    "split_sentence_like",
    "tokenizer_acquisition_target_contract",
    "verify_acquisition_plan",
    "write_frozen_acquisition_plan",
]
