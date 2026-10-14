#!/usr/bin/env python3
"""Create the frozen tokenizer-aware FineWeb A/B-v2 calibration artifact.

The tool consumes a completed 200-language bootstrap release.  It measures a
deterministic sample of unique *train* documents with the frozen SentencePiece
model, then combines those measurements with the immutable FineWeb2 inventory.
The compact ``languages`` object is intentionally accepted without adaptation
by :mod:`build_fineweb_v2_sampling_plan`; detailed evidence lives separately in
``language_audits``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    LanguageCalibration,
    canonical_json_sha256,
    sha256_file,
)


SCHEMA_VERSION = 1
TOOL_ID = "fineweb_ab_v2_token_calibration_v1"
DEFAULT_SAMPLE_DOCUMENTS = 1_000
DEFAULT_SAMPLE_SEED = 42
DEFAULT_EXPECTED_LANGUAGES = 200
DEFAULT_EXPECTED_VOCAB_SIZE = 64_000
ENGLISH = "eng_Latn"
ENGLISH_UTF8_BYTES_RULE = (
    "derive from the frozen 1000-document tokenizer calibration as "
    "reported_words * calibration_utf8_bytes / calibration_words"
)
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _verify_payload_hash(
    payload: Mapping[str, Any], *, hash_field: str, label: str
) -> str:
    observed = payload.get(hash_field)
    without_hash = {key: value for key, value in payload.items() if key != hash_field}
    expected = canonical_json_sha256(without_hash)
    if observed != expected:
        raise ValueError(f"{label} payload hash mismatch")
    return expected


def _positive_integer(raw: Any, *, label: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _verify_file_entry(*, path: Path, entry: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    expected_sha256 = str(entry.get("sha256", ""))
    if not HEX_64_RE.fullmatch(expected_sha256) or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    if "bytes" in entry and int(entry["bytes"]) != path.stat().st_size:
        raise ValueError(f"{label} byte-size mismatch")


def _source_for_language(source_lock: Mapping[str, Any], language: str) -> dict[str, str]:
    key = "english_corpus" if language == ENGLISH else "non_english_corpus"
    raw = source_lock.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"Source lock is missing {key}")
    dataset = str(raw.get("dataset", ""))
    revision = str(raw.get("dataset_revision", ""))
    if not dataset or not HEX_40_RE.fullmatch(revision):
        raise ValueError(f"Source lock has invalid {key} identity")
    return {"dataset": dataset, "revision": revision, "subset": language}


def _load_inventory(
    *,
    inventory_path: Path,
    source_lock: Mapping[str, Any],
    selected_languages: Sequence[str],
) -> dict[str, dict[str, int]]:
    non_english = source_lock.get("non_english_corpus")
    inventory_lock = (
        non_english.get("language_inventory") if isinstance(non_english, dict) else None
    )
    if not isinstance(inventory_lock, dict):
        raise ValueError("Source lock is missing non-English language inventory metadata")
    expected_hash = str(inventory_lock.get("sha256", ""))
    if not HEX_64_RE.fullmatch(expected_hash) or sha256_file(inventory_path) != expected_hash:
        raise ValueError("Materialized FineWeb2 inventory does not match the source lock")

    # The source lock authenticates the complete upstream inventory, including
    # rows outside this study whose unavailable statistics are represented as
    # ``-``.  Parse only the selected non-English rows: English comes from the
    # separately locked FineWeb corpus and is derived below from its declared
    # availability plus the tokenizer calibration.
    required_subsets = {str(language) for language in selected_languages} - {ENGLISH}
    inventory: dict[str, dict[str, int]] = {}
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subset", "split", "documents", "utf8_bytes", "words"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("FineWeb2 inventory lacks required columns")
        for line_number, row in enumerate(reader, start=2):
            if row.get("split") != "train":
                continue
            subset = str(row.get("subset", ""))
            if subset not in required_subsets:
                continue
            if subset in inventory:
                raise ValueError(f"Duplicate FineWeb2 train inventory row for {subset}")
            inventory[subset] = {
                "documents": _positive_integer(
                    row.get("documents"), label=f"inventory line {line_number} documents"
                ),
                "utf8_bytes": _positive_integer(
                    row.get("utf8_bytes"), label=f"inventory line {line_number} utf8_bytes"
                ),
                "words": _positive_integer(
                    row.get("words"), label=f"inventory line {line_number} words"
                ),
            }
    missing = sorted(required_subsets - set(inventory))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise ValueError(
            "FineWeb2 inventory has no train row for selected language(s): "
            f"{preview}{suffix}"
        )
    return inventory


def _load_sentencepiece(
    tokenizer_model_path: Path, *, expected_vocab_size: int
) -> tuple[Any, dict[str, int]]:
    if not tokenizer_model_path.is_file():
        raise ValueError(f"SentencePiece model is missing: {tokenizer_model_path}")
    try:
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as model_pb2
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("sentencepiece is required for FineWeb token calibration") from exc

    processor = spm.SentencePieceProcessor()
    if not processor.load(str(tokenizer_model_path)):
        raise ValueError(f"Cannot load SentencePiece model: {tokenizer_model_path}")
    ids = {
        "unk_id": int(processor.unk_id()),
        "bos_id": int(processor.bos_id()),
        "eos_id": int(processor.eos_id()),
        "pad_id": int(processor.pad_id()),
    }
    if int(processor.vocab_size()) != int(expected_vocab_size):
        raise ValueError(
            f"SentencePiece vocab size is {processor.vocab_size()}, expected {expected_vocab_size}"
        )
    if ids != {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": 3}:
        raise ValueError(f"Unexpected SentencePiece special-token IDs: {ids}")
    model = model_pb2.ModelProto()
    try:
        model.ParseFromString(tokenizer_model_path.read_bytes())
    except Exception as exc:
        raise ValueError(f"Cannot parse SentencePiece model protobuf: {tokenizer_model_path}") from exc
    if model.trainer_spec.model_type != model_pb2.TrainerSpec.UNIGRAM:
        raise ValueError("Frozen A/B-v2 tokenizer must be a SentencePiece Unigram model")
    return processor, ids


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    return records


def _validated_train_records(
    records: Sequence[dict[str, Any]],
    *,
    language: str,
    source: Mapping[str, str],
    expected_role: str,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        document_id = str(record.get("document_id", ""))
        if not HEX_64_RE.fullmatch(document_id):
            raise ValueError(f"{language} {expected_role} row {index} has invalid document_id")
        if document_id in seen:
            raise ValueError(f"{language} {expected_role} contains duplicate train originals")
        seen.add(document_id)
        if record.get("language") != language or record.get("artifact_role") != expected_role:
            raise ValueError(f"{language} {expected_role} row {index} has wrong language/role")
        text = record.get("text")
        lineage = record.get("lineage")
        if not isinstance(text, str) or not isinstance(lineage, dict):
            raise ValueError(f"{language} {expected_role} row {index} lacks text/lineage")
        if not text:
            raise ValueError(f"{language} {expected_role} row {index} has empty text")
        expected_lineage = {
            "source_dataset": source["dataset"],
            "source_revision": source["revision"],
            "subset": language,
            "split": "train",
        }
        drift = [key for key, value in expected_lineage.items() if lineage.get(key) != value]
        if drift:
            raise ValueError(
                f"{language} {expected_role} row {index} has invalid train lineage: {drift}"
            )
        if lineage.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError(f"{language} {expected_role} row {index} text hash mismatch")
        multiplier = _positive_integer(
            lineage.get("rehydration_multiplier"),
            label=f"{language} {expected_role} row {index} rehydration_multiplier",
        )
        if language == ENGLISH and multiplier != 1:
            raise ValueError("English calibration must not contain rehydrated replicas")
        source_file_hash = str(lineage.get("source_file_sha256", ""))
        if not HEX_64_RE.fullmatch(source_file_hash):
            raise ValueError(f"{language} {expected_role} row {index} source hash is invalid")
        validated.append(record)
    return validated


def _round_fraction_half_up(numerator: int, denominator: int) -> int:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("fraction terms must be positive")
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(remainder * 2 >= denominator)


def _calibrate_language(
    *,
    language: str,
    bootstrap_root: Path,
    summary_report_hash: str,
    manifest_entry: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    source_lock_sha256: str,
    inventory: Mapping[str, Mapping[str, int]],
    processor: Any,
    sample_documents: int,
    sample_seed: int,
    target_validation_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    language_dir = bootstrap_root / language
    report_path = language_dir / "bootstrap_report.json"
    report = _load_object(report_path)
    if report.get("schema_version") != SCHEMA_VERSION or report.get("tool_id") != "fineweb_ab_v2_bootstrap_v1":
        raise ValueError(f"{language} bootstrap report has unsupported schema/tool ID")
    report_payload_sha256 = _verify_payload_hash(
        report, hash_field="report_payload_sha256", label=f"{language} bootstrap report"
    )
    if report_payload_sha256 != summary_report_hash or report.get("status") != "complete":
        raise ValueError(f"{language} bootstrap report does not match the completed summary")
    run_config = report.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"{language} bootstrap report lacks run_config")
    if report.get("run_config_sha256") != canonical_json_sha256(run_config):
        raise ValueError(f"{language} bootstrap run_config hash mismatch")
    if (
        run_config.get("schema_version") != SCHEMA_VERSION
        or run_config.get("tool_id") != "fineweb_ab_v2_bootstrap_v1"
    ):
        raise ValueError(f"{language} bootstrap run_config has unsupported schema/tool ID")
    if run_config.get("language") != language:
        raise ValueError(f"{language} bootstrap report language mismatch")
    if run_config.get("source_lock_sha256") != source_lock_sha256:
        raise ValueError(f"{language} bootstrap used a different source lock")
    source = _source_for_language(source_lock, language)
    raw_source = run_config.get("source")
    if not isinstance(raw_source, dict):
        raise ValueError(f"{language} bootstrap run_config lacks source identity")
    if any(raw_source.get(key) != value for key, value in source.items()):
        raise ValueError(f"{language} bootstrap source identity mismatch")

    counts = report.get("counts")
    outputs = report.get("outputs")
    if not isinstance(counts, dict) or not isinstance(outputs, dict):
        raise ValueError(f"{language} bootstrap report lacks counts/outputs")
    available_train_originals = _positive_integer(
        counts.get("unique_train_documents"),
        label=f"{language} unique_train_documents",
    )
    target_documents = min(int(sample_documents), available_train_originals)
    if target_documents <= 0:
        raise ValueError(f"{language} has no train originals for calibration")

    train_manifest = manifest_entry.get("train")
    if not isinstance(train_manifest, dict):
        raise ValueError(f"Bootstrap manifest lacks {language} train lineage")
    expected_train_relative = f"{language}/train.jsonl"
    if train_manifest.get("path") != expected_train_relative:
        raise ValueError(f"{language} bootstrap manifest train path is not canonical")
    train_path = bootstrap_root / expected_train_relative
    train_output = outputs.get("train.jsonl")
    if not isinstance(train_output, dict):
        raise ValueError(f"{language} report lacks train.jsonl lineage")
    if train_output.get("path") != "train.jsonl":
        raise ValueError(f"{language} report train path is not canonical")
    if train_manifest.get("sha256") != train_output.get("sha256"):
        raise ValueError(f"{language} manifest/report train hash mismatch")
    _verify_file_entry(path=train_path, entry=train_output, label=f"{language} train.jsonl")

    calibration_output = outputs.get("calibration.jsonl")
    if not isinstance(calibration_output, dict):
        raise ValueError(f"{language} report lacks calibration.jsonl lineage")
    if calibration_output.get("path") != "calibration.jsonl":
        raise ValueError(f"{language} calibration output path is not canonical")
    calibration_path = language_dir / "calibration.jsonl"
    _verify_file_entry(
        path=calibration_path,
        entry=calibration_output,
        label=f"{language} calibration.jsonl",
    )
    calibration_records = _validated_train_records(
        _read_records(calibration_path),
        language=language,
        source=source,
        expected_role="calibration",
    )
    expected_calibration_count = _positive_integer(
        counts.get("calibration_documents"),
        label=f"{language} calibration_documents",
    )
    if len(calibration_records) != expected_calibration_count:
        raise ValueError(f"{language} calibration JSONL row count disagrees with report")
    hashes = report.get("hashes")
    calibration_ids_hash = canonical_json_sha256(
        sorted(str(record["document_id"]) for record in calibration_records)
    )
    if (
        not isinstance(hashes, dict)
        or hashes.get("calibration_ids_sha256") != calibration_ids_hash
    ):
        raise ValueError(f"{language} calibration ID hash disagrees with report")

    # The bootstrap calibration file is verified above as an acquisition audit,
    # but it contains the first train records read.  Draw from the complete
    # acquired train pool by a seeded hash instead, so document order inside a
    # selected Parquet shard cannot bias tokenizer-fertility estimates.
    train_records = _validated_train_records(
        _read_records(train_path),
        language=language,
        source=source,
        expected_role="train",
    )
    if len(train_records) != available_train_originals:
        raise ValueError(f"{language} train JSONL row count disagrees with report")
    candidate_records = train_records
    selected_input_path = train_path
    selected_input_role = "train"
    selected_input_sha256 = str(train_output["sha256"])

    if len(candidate_records) < target_documents:
        raise ValueError(
            f"{language} has only {len(candidate_records)} verified train originals; "
            f"{target_documents} are required"
        )
    selected = sorted(
        candidate_records,
        key=lambda record: (
            hashlib.sha256(
                (
                    f"{int(sample_seed)}\0tokenizer-fertility-calibration-v1\0"
                    f"{language}\0{record['document_id']}"
                ).encode("utf-8")
            ).hexdigest(),
            str(record["document_id"]),
        ),
    )[:target_documents]
    selected_ids = [str(record["document_id"]) for record in selected]

    utf8_bytes = 0
    words = 0
    sentencepiece_tokens = 0
    tokens_with_eod = 0
    rehydrated_tokens = 0
    rehydrated_document_equivalents = 0
    multiplier_histogram: Counter[int] = Counter()
    for record in selected:
        text = str(record["text"])
        multiplier = int(record["lineage"]["rehydration_multiplier"])
        token_ids = processor.encode(text, out_type=int, add_bos=False, add_eos=False)
        if any(int(token_id) < 0 or int(token_id) >= int(processor.vocab_size()) for token_id in token_ids):
            raise ValueError(f"{language} tokenizer emitted an invalid piece ID")
        document_tokens = len(token_ids) + 1  # Exactly one Megatron EOD token per document.
        utf8_bytes += len(text.encode("utf-8"))
        words += len(text.split())
        sentencepiece_tokens += len(token_ids)
        tokens_with_eod += document_tokens
        rehydrated_tokens += document_tokens * multiplier
        rehydrated_document_equivalents += multiplier
        multiplier_histogram[multiplier] += 1
    if utf8_bytes <= 0 or words <= 0 or tokens_with_eod <= target_documents:
        raise ValueError(f"{language} calibration sample has invalid byte/word/token totals")

    candidate_validation_fraction = float(
        run_config.get("settings", {}).get("validation_candidate_fraction", 0.0)
    )
    if not 0.0 < candidate_validation_fraction < 1.0:
        raise ValueError(f"{language} bootstrap has invalid validation_fraction")
    if not 0.0 < float(target_validation_fraction) < candidate_validation_fraction:
        raise ValueError(
            f"{language} target validation fraction must be positive and smaller "
            "than the pre-tokenizer candidate reserve"
        )
    if language == ENGLISH:
        english = source_lock.get("english_corpus")
        availability = english.get("availability") if isinstance(english, dict) else None
        if not isinstance(availability, dict):
            raise ValueError("Source lock lacks English availability")
        if availability.get("utf8_bytes_rule") != ENGLISH_UTF8_BYTES_RULE:
            raise ValueError("English UTF-8 byte derivation rule is not the frozen A/B-v2 rule")
        available_documents = _positive_integer(
            availability.get("reported_documents"), label="English reported_documents"
        )
        reported_words = _positive_integer(
            availability.get("reported_words"), label="English reported_words"
        )
        numerator = reported_words * utf8_bytes
        available_utf8_bytes = _round_fraction_half_up(numerator, words)
        availability_audit: dict[str, Any] = {
            "source": "source_lock.english_corpus.availability",
            "reported_documents": available_documents,
            "reported_words": reported_words,
            "utf8_bytes_derivation": {
                "declared_rule": ENGLISH_UTF8_BYTES_RULE,
                "formula": "reported_words * calibration_utf8_bytes / calibration_words",
                "numerator": numerator,
                "denominator": words,
                "exact_fraction": f"{numerator}/{words}",
                "integer_rounding": "nearest_integer_half_up",
                "result": available_utf8_bytes,
            },
        }
    else:
        raw_availability = inventory.get(language)
        if raw_availability is None:
            raise ValueError(f"FineWeb2 inventory has no train row for {language}")
        available_documents = int(raw_availability["documents"])
        available_utf8_bytes = int(raw_availability["utf8_bytes"])
        availability_audit = {
            "source": "pinned_fineweb2_inventory_train_row",
            "documents": available_documents,
            "utf8_bytes": available_utf8_bytes,
            "words": int(raw_availability["words"]),
        }
    if available_train_originals > available_documents:
        raise ValueError(f"{language} bootstrap train count exceeds frozen source availability")

    compact = {
        "available_utf8_bytes": available_utf8_bytes,
        "available_documents": available_documents,
        "calibration_utf8_bytes": utf8_bytes,
        "calibration_tokens": tokens_with_eod,
        "calibration_documents": target_documents,
        "calibration_rehydrated_tokens": rehydrated_tokens,
        # This fraction describes the final model-training availability.  The
        # larger bootstrap fraction is only a tokenizer-clean candidate pool;
        # unused candidates return to the model-training selection pool after
        # exact token counting.
        "validation_fraction": float(target_validation_fraction),
    }
    LanguageCalibration(language=language, **compact).validate()
    audit = {
        "bootstrap_report_path": str(report_path.resolve()),
        "bootstrap_report_file_sha256": sha256_file(report_path),
        "bootstrap_report_payload_sha256": report_payload_sha256,
        "available_train_originals_in_bootstrap": available_train_originals,
        "selection": {
            "policy": "seeded_sha256_smallest_from_all_verified_train_originals_v1",
            "seed": int(sample_seed),
            "requested_documents": int(sample_documents),
            "effective_documents": target_documents,
            "input_role": selected_input_role,
            "input_path": str(selected_input_path.resolve()),
            "input_sha256": selected_input_sha256,
            "selected_document_ids_sha256": canonical_json_sha256(selected_ids),
            "unique_train_originals": len(set(selected_ids)),
        },
        "measurements": {
            "utf8_bytes": utf8_bytes,
            "words": words,
            "word_definition": "Python str.split Unicode-whitespace tokens",
            "documents": target_documents,
            "sentencepiece_tokens_excluding_eod": sentencepiece_tokens,
            "eod_tokens_added": target_documents,
            "tokens_including_one_eod_per_document": tokens_with_eod,
            "rehydrated_tokens_including_eod": rehydrated_tokens,
            "rehydrated_document_equivalents": rehydrated_document_equivalents,
            "rehydration_multiplier_histogram": {
                str(key): value for key, value in sorted(multiplier_histogram.items())
            },
        },
        "availability": availability_audit,
        "validation_contract": {
            "candidate_reserve_fraction": candidate_validation_fraction,
            "target_token_fraction": float(target_validation_fraction),
            "unused_candidates_return_to_model_training_selection_pool": True,
            "candidate_reserve_excluded_from_tokenizer_training": True,
        },
    }
    return compact, audit


def build_token_calibration_artifact(
    *,
    bootstrap_summary_path: Path,
    tokenizer_model_path: Path,
    source_lock_path: Path,
    inventory_path: Path,
    sample_documents: int = DEFAULT_SAMPLE_DOCUMENTS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    expected_language_count: int = DEFAULT_EXPECTED_LANGUAGES,
    expected_vocab_size: int = DEFAULT_EXPECTED_VOCAB_SIZE,
    target_validation_fraction: float = 0.20,
) -> dict[str, Any]:
    """Validate all inputs and return a planner-compatible calibration object."""

    bootstrap_summary_path = bootstrap_summary_path.resolve()
    bootstrap_root = bootstrap_summary_path.parent
    tokenizer_model_path = tokenizer_model_path.resolve()
    source_lock_path = source_lock_path.resolve()
    inventory_path = inventory_path.resolve()
    if int(sample_documents) <= 0:
        raise ValueError("sample_documents must be positive")
    if not isinstance(sample_seed, int) or isinstance(sample_seed, bool):
        raise ValueError("sample_seed must be an integer")
    if int(expected_language_count) <= 0 or int(expected_vocab_size) <= 0:
        raise ValueError("expected language/vocabulary counts must be positive")
    if not 0.0 < float(target_validation_fraction) < 1.0:
        raise ValueError("target_validation_fraction must be strictly between zero and one")

    source_lock = _load_object(source_lock_path)
    if source_lock.get("schema_version") != SCHEMA_VERSION or not source_lock.get("source_lock_id"):
        raise ValueError("Source lock has unsupported schema or no source_lock_id")
    source_lock_sha256 = sha256_file(source_lock_path)

    summary = _load_object(bootstrap_summary_path)
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("tool_id") != "fineweb_ab_v2_bootstrap_v1":
        raise ValueError("Bootstrap summary has unsupported schema/tool ID")
    summary_payload_sha256 = _verify_payload_hash(
        summary, hash_field="summary_sha256", label="bootstrap summary"
    )
    raw_languages = summary.get("languages")
    raw_report_hashes = summary.get("report_payload_sha256")
    if not isinstance(raw_languages, list) or not isinstance(raw_report_hashes, dict):
        raise ValueError("Bootstrap summary lacks languages/report hashes")
    languages = [str(language) for language in raw_languages]
    if len(languages) != int(expected_language_count) or len(languages) != len(set(languages)):
        raise ValueError(
            f"Bootstrap summary must contain exactly {expected_language_count} unique languages"
        )
    if set(raw_report_hashes) != set(languages) or ENGLISH not in languages:
        raise ValueError("Bootstrap summary report coverage/language coverage is incomplete")
    inventory = _load_inventory(
        inventory_path=inventory_path,
        source_lock=source_lock,
        selected_languages=languages,
    )
    processor, special_ids = _load_sentencepiece(
        tokenizer_model_path, expected_vocab_size=int(expected_vocab_size)
    )

    manifest_path = bootstrap_root / "bootstrap_manifest.json"
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Bootstrap manifest has unsupported schema")
    manifest_payload_sha256 = _verify_payload_hash(
        manifest, hash_field="manifest_sha256", label="bootstrap manifest"
    )
    if summary.get("bootstrap_manifest_payload_sha256") != manifest_payload_sha256:
        raise ValueError("Bootstrap summary/manifest payload hashes disagree")
    manifest_languages = manifest.get("languages")
    manifest_provenance = manifest.get("provenance")
    if not isinstance(manifest_languages, dict) or set(manifest_languages) != set(languages):
        raise ValueError("Bootstrap manifest language coverage is incomplete")
    if (
        not isinstance(manifest_provenance, dict)
        or manifest_provenance.get("source_lock_sha256") != source_lock_sha256
    ):
        raise ValueError("Bootstrap manifest used a different source lock")

    compact_languages: dict[str, Any] = {}
    language_audits: dict[str, Any] = {}
    for language in languages:
        expected_report_hash = str(raw_report_hashes[language])
        if not HEX_64_RE.fullmatch(expected_report_hash):
            raise ValueError(f"Bootstrap summary has invalid report hash for {language}")
        compact, audit = _calibrate_language(
            language=language,
            bootstrap_root=bootstrap_root,
            summary_report_hash=expected_report_hash,
            manifest_entry=manifest_languages[language],
            source_lock=source_lock,
            source_lock_sha256=source_lock_sha256,
            inventory=inventory,
            processor=processor,
            sample_documents=int(sample_documents),
            sample_seed=int(sample_seed),
            target_validation_fraction=float(target_validation_fraction),
        )
        compact_languages[language] = compact
        language_audits[language] = audit

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "status": "complete",
        "languages": compact_languages,
        "language_audits": language_audits,
        "provenance": {
            "bootstrap_summary_path": str(bootstrap_summary_path),
            "bootstrap_summary_file_sha256": sha256_file(bootstrap_summary_path),
            "bootstrap_summary_payload_sha256": summary_payload_sha256,
            "bootstrap_manifest_path": str(manifest_path.resolve()),
            "bootstrap_manifest_file_sha256": sha256_file(manifest_path),
            "bootstrap_manifest_payload_sha256": manifest_payload_sha256,
            "source_lock_path": str(source_lock_path),
            "source_lock_sha256": source_lock_sha256,
            "source_lock_id": source_lock.get("source_lock_id"),
            "fineweb2_inventory_path": str(inventory_path),
            "fineweb2_inventory_sha256": sha256_file(inventory_path),
            "tokenizer_model_path": str(tokenizer_model_path),
            "tokenizer_model_sha256": sha256_file(tokenizer_model_path),
            "tokenizer": {
                "implementation": "sentencepiece",
                "model_type": "unigram",
                "vocab_size": int(processor.vocab_size()),
                **special_ids,
                "encoding": "deterministic_one_best_no_bos_no_eos_then_one_eod",
            },
        },
        "audit": {
            "expected_language_count": int(expected_language_count),
            "observed_language_count": len(languages),
            "requested_calibration_documents_per_language": int(sample_documents),
            "calibration_sample_seed": int(sample_seed),
            "target_validation_fraction": float(target_validation_fraction),
            "effective_calibration_documents_total": sum(
                int(value["calibration_documents"]) for value in compact_languages.values()
            ),
            "calibration_tokens_total": sum(
                int(value["calibration_tokens"]) for value in compact_languages.values()
            ),
            "calibration_rehydrated_tokens_total": sum(
                int(value["calibration_rehydrated_tokens"])
                for value in compact_languages.values()
            ),
            "all_samples_unique_train_originals": True,
            "one_eod_per_document": True,
            "coverage_complete": True,
        },
    }
    artifact["artifact_sha256"] = canonical_json_sha256(artifact)
    return artifact


def write_frozen_calibration(*, output_path: Path, artifact: Mapping[str, Any]) -> None:
    """Atomically create, or exactly verify, the immutable calibration file."""

    output_path = output_path.resolve()
    if output_path.exists():
        if _load_object(output_path) != dict(artifact):
            raise RuntimeError(f"Refusing to replace a different frozen calibration: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-summary", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-documents", type=int, default=DEFAULT_SAMPLE_DOCUMENTS)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--expected-language-count", type=int, default=DEFAULT_EXPECTED_LANGUAGES
    )
    parser.add_argument("--expected-vocab-size", type=int, default=DEFAULT_EXPECTED_VOCAB_SIZE)
    parser.add_argument("--target-validation-fraction", type=float, default=0.20)
    args = parser.parse_args()

    artifact = build_token_calibration_artifact(
        bootstrap_summary_path=Path(args.bootstrap_summary),
        tokenizer_model_path=Path(args.tokenizer_model),
        source_lock_path=Path(args.source_lock),
        inventory_path=Path(args.inventory),
        sample_documents=int(args.sample_documents),
        sample_seed=int(args.sample_seed),
        expected_language_count=int(args.expected_language_count),
        expected_vocab_size=int(args.expected_vocab_size),
        target_validation_fraction=float(args.target_validation_fraction),
    )
    output_path = Path(args.output)
    write_frozen_calibration(output_path=output_path, artifact=artifact)
    print(f"[OK] token_calibration={output_path.resolve()} sha256={artifact['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENGLISH_UTF8_BYTES_RULE",
    "build_token_calibration_artifact",
    "write_frozen_calibration",
]
