#!/usr/bin/env python3
"""Finalize the frozen FineWeb A/B-v2 train pool and validation panel.

This tool has no network, Slurm, or training dependency.  It consumes three
already-frozen inputs:

* a bootstrap manifest listing one train and one validation-reserve JSONL per
  language, together with their SHA-256 hashes;
* a SentencePiece model whose SHA-256 is pinned by the sampling plan; and
* the nested sampling plan produced by ``build_fineweb_v2_sampling_plan``.

Bootstrap manifest schema::

    {
      "schema_version": 1,
      "provenance": {"source_manifest_sha256": "..."},
      "languages": {
        "deu_Latn": {
          "train": {"path": "bootstrap/train/deu_Latn.jsonl", "sha256": "..."},
          "validation_reserve": {
            "path": "bootstrap/validation_reserve/deu_Latn.jsonl",
            "sha256": "..."
          }
        }
      }
    }

Paths are resolved relative to the bootstrap manifest.  Every JSONL row must
contain ``language``, ``text``, ``document_id``, and a ``lineage`` object with
``source_dataset``, ``source_revision``, ``subset``, ``source_file_sha256``,
``text_sha256``, ``split``, and ``rehydration_multiplier``.  At least one of
``source_id`` or ``url`` must also be present.  Every row must preserve its
model-training multiplier as ``lineage.train_rehydration_multiplier``; for a
validation candidate this can differ from its current unrehydrated multiplier.
``document_id`` must equal the stable ID derived by
:func:`stable_document_id`; this makes accidental split overlap and source
drift fail closed.

Selection is by an ascending SHA-256 key over the frozen seed, a split-specific
namespace, and the original document ID.  Final validation documents are
selected from the pre-tokenizer candidate reserve *before* train selection.
Every unused candidate is then deterministically reclassified as train-eligible
with its preserved train multiplier.  A train row represents one original
document and declares its total FineWeb2 rehydration multiplier; all replicas
are either selected or rejected together.  Selected validation rows always
remain unrehydrated.  Token counts include one SentencePiece EOS/EOD per
emitted document, matching the Megatron preprocessing contract.  Unigram
counts exclude every SentencePiece special/control/unused ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    DEFAULT_SEED,
    DEFAULT_VALIDATION_CAP_TOKENS,
    DEFAULT_VALIDATION_FRACTION,
    assign_validation_reserve,
    canonical_json_sha256,
    sha256_file,
    stable_document_id,
    validation_target_tokens,
    verify_plan_hash,
)


SCHEMA_VERSION = 1
# The sampling plan already converts every 32B K-specific exposure target into
# one shared physical-pool target per language.  Underfilling that frozen
# target changes the effective number of passes and must therefore request a
# larger immutable acquisition rather than being silently accepted.
DEFAULT_MAX_TRAIN_SHORTFALL_FRACTION = 0.0
TRAIN_SELECTION_NAMESPACE = "physical-train-document-selection-v1"
VALIDATION_SELECTION_NAMESPACE = "validation-panel-document-selection-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SPLITS = ("train", "validation_reserve")


class CorpusFinalizationError(ValueError):
    """Raised when a frozen input violates the finalization contract."""


@dataclass(frozen=True, slots=True)
class BootstrapInput:
    language: str
    split: str
    path: Path
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class BootstrapCoverage:
    language: str
    complete_source_scan: bool
    unique_source_documents: int
    unique_source_utf8_bytes: int
    acquisition_status: str
    validation_candidate_fraction: float
    report_path: Path
    report_payload_sha256: str


@dataclass(frozen=True, slots=True)
class SelectionSummary:
    target_tokens: int
    available_tokens: int
    selected_tokens: int
    selected_original_documents: int
    emitted_documents: int

    @property
    def shortfall_tokens(self) -> int:
        return max(0, self.target_tokens - self.selected_tokens)

    @property
    def overshoot_tokens(self) -> int:
        return max(0, self.selected_tokens - self.target_tokens)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusFinalizationError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusFinalizationError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusFinalizationError(f"{label} must be a JSON object: {path}")
    return payload


def _require_sha256(value: Any, *, label: str) -> str:
    candidate = str(value or "").lower()
    if not _SHA256_RE.fullmatch(candidate):
        raise CorpusFinalizationError(f"{label} must be a lowercase SHA-256 digest")
    return candidate


def _resolve_manifest_path(
    manifest_path: Path,
    value: Any,
    *,
    label: str,
    expected_suffix: str = ".jsonl",
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise CorpusFinalizationError(f"{label}.path must be set")
    path = Path(text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise CorpusFinalizationError(f"{label} JSONL does not exist: {path}")
    if path.suffix != expected_suffix:
        raise CorpusFinalizationError(
            f"{label} must be an uncompressed {expected_suffix} file: {path}"
        )
    return path


def _parse_bootstrap_inputs(
    *,
    manifest_path: Path,
    payload: Mapping[str, Any],
    required_languages: set[str],
) -> tuple[list[BootstrapInput], dict[str, BootstrapCoverage]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CorpusFinalizationError(
            f"bootstrap manifest schema_version must be {SCHEMA_VERSION}"
        )
    expected_manifest_hash = _require_sha256(
        payload.get("manifest_sha256"),
        label="bootstrap manifest.manifest_sha256",
    )
    manifest_without_hash = dict(payload)
    manifest_without_hash.pop("manifest_sha256", None)
    if canonical_json_sha256(manifest_without_hash) != expected_manifest_hash:
        raise CorpusFinalizationError("bootstrap manifest payload SHA-256 mismatch")
    raw_languages = payload.get("languages")
    if not isinstance(raw_languages, dict):
        raise CorpusFinalizationError("bootstrap manifest must contain a languages object")
    provided_languages = set(map(str, raw_languages))
    if provided_languages != required_languages:
        missing = sorted(required_languages - provided_languages)
        unexpected = sorted(provided_languages - required_languages)
        raise CorpusFinalizationError(
            f"bootstrap languages do not match the plan: missing={missing}, unexpected={unexpected}"
        )

    inputs: list[BootstrapInput] = []
    coverage: dict[str, BootstrapCoverage] = {}
    for language in sorted(required_languages):
        if not _LANGUAGE_RE.fullmatch(language):
            raise CorpusFinalizationError(f"unsafe language identifier in plan: {language!r}")
        raw_language = raw_languages.get(language)
        if not isinstance(raw_language, dict):
            raise CorpusFinalizationError(f"bootstrap entry for {language} must be an object")
        raw_acquisition = raw_language.get("acquisition")
        if not isinstance(raw_acquisition, dict):
            raise CorpusFinalizationError(
                f"bootstrap languages.{language}.acquisition must be an object"
            )
        report_path = _resolve_manifest_path(
            manifest_path,
            raw_acquisition.get("report_path"),
            label=f"bootstrap languages.{language}.acquisition.report",
            expected_suffix=".json",
        )
        report = _load_object(report_path, label=f"{language} bootstrap report")
        report_payload_sha256 = _require_sha256(
            raw_acquisition.get("report_payload_sha256"),
            label=f"bootstrap languages.{language}.acquisition.report_payload_sha256",
        )
        report_without_hash = dict(report)
        observed_report_hash = report_without_hash.pop("report_payload_sha256", None)
        if (
            observed_report_hash != report_payload_sha256
            or canonical_json_sha256(report_without_hash) != report_payload_sha256
        ):
            raise CorpusFinalizationError(f"{language}: bootstrap report payload SHA-256 mismatch")
        targets = report.get("targets")
        counts = report.get("counts")
        if not isinstance(targets, dict) or not isinstance(counts, dict):
            raise CorpusFinalizationError(f"{language}: bootstrap report lacks targets/counts")
        run_config = report.get("run_config")
        settings = run_config.get("settings") if isinstance(run_config, dict) else None
        candidate_fraction = (
            settings.get("validation_candidate_fraction")
            if isinstance(settings, dict)
            else None
        )
        if (
            isinstance(candidate_fraction, bool)
            or not isinstance(candidate_fraction, (int, float))
            or not math.isfinite(float(candidate_fraction))
            or not 0.0 < float(candidate_fraction) < 1.0
        ):
            raise CorpusFinalizationError(
                f"{language}: bootstrap report lacks a valid explicit "
                "run_config.settings.validation_candidate_fraction"
            )
        complete_source_scan = bool(raw_acquisition.get("complete_source_scan"))
        unique_source_documents = int(raw_acquisition.get("unique_source_documents", 0))
        unique_source_utf8_bytes = int(raw_acquisition.get("unique_source_utf8_bytes", 0))
        acquisition_status = str(raw_acquisition.get("acquisition_status", ""))
        if (
            complete_source_scan
            != bool(
                targets.get("source_exhausted")
                or targets.get("pinned_inventory_exhausted")
            )
            or unique_source_documents != int(counts.get("unique_source_documents", 0))
            or unique_source_utf8_bytes != int(counts.get("unique_source_utf8_bytes", 0))
            or acquisition_status != str(report.get("acquisition_status", ""))
            or unique_source_documents <= 0
            or unique_source_utf8_bytes <= 0
        ):
            raise CorpusFinalizationError(
                f"{language}: bootstrap acquisition coverage disagrees with its report"
            )
        coverage[language] = BootstrapCoverage(
            language=language,
            complete_source_scan=complete_source_scan,
            unique_source_documents=unique_source_documents,
            unique_source_utf8_bytes=unique_source_utf8_bytes,
            acquisition_status=acquisition_status,
            validation_candidate_fraction=float(candidate_fraction),
            report_path=report_path,
            report_payload_sha256=report_payload_sha256,
        )
        for split in _SPLITS:
            raw_input = raw_language.get(split)
            label = f"bootstrap languages.{language}.{split}"
            if not isinstance(raw_input, dict):
                raise CorpusFinalizationError(f"{label} must be an object")
            inputs.append(
                BootstrapInput(
                    language=language,
                    split=split,
                    path=_resolve_manifest_path(
                        manifest_path,
                        raw_input.get("path"),
                        label=label,
                    ),
                    expected_sha256=_require_sha256(
                        raw_input.get("sha256"),
                        label=f"{label}.sha256",
                    ),
                )
            )
    return inputs, coverage


def _load_sentencepiece(model_path: Path) -> Any:
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise CorpusFinalizationError("corpus finalization requires sentencepiece") from exc
    processor = spm.SentencePieceProcessor()
    if not processor.load(str(model_path)):
        raise CorpusFinalizationError(f"could not load SentencePiece model: {model_path}")
    if int(processor.eos_id()) < 0:
        raise CorpusFinalizationError("SentencePiece model must define EOS for document EOD")
    return processor


def _special_token_ids(processor: Any) -> set[int]:
    special = {
        int(token_id)
        for token_id in (
            processor.unk_id(),
            processor.bos_id(),
            processor.eos_id(),
            processor.pad_id(),
        )
        if int(token_id) >= 0
    }
    for token_id in range(int(processor.get_piece_size())):
        if processor.is_unknown(token_id) or processor.is_control(token_id):
            special.add(token_id)
        is_unused = getattr(processor, "is_unused", None)
        if is_unused is not None and is_unused(token_id):
            special.add(token_id)
    return special


def _encode_document(processor: Any, text: str) -> list[int]:
    # Unigram inference without sampling is deterministic.  Add one EOD just as
    # Megatron's preprocess_data.py does under --append-eod.
    token_ids = list(processor.encode(text, out_type=int, add_bos=False, add_eos=False))
    token_ids.append(int(processor.eos_id()))
    return token_ids


_POOL_PROCESSOR: Any | None = None
_POOL_INPUT_HANDLES: dict[str, BinaryIO] = {}


def _initialize_encoder_worker(model_path: str) -> None:
    global _POOL_PROCESSOR
    _POOL_PROCESSOR = _load_sentencepiece(Path(model_path))


def _pool_index_batch(
    payload: tuple[
        list[tuple[int, int, bytes]],
        str,
        str,
        str,
        int,
        float,
    ],
) -> tuple[list[tuple[Any, ...]], int, int]:
    """Parse, validate, hash, and tokenize one ordered JSONL batch.

    The original implementation performed every operation except SentencePiece
    encoding in the parent process.  On large languages that serialized JSON
    decoding, lineage validation, stable-ID hashing, and selection-key hashing
    onto one CPU.  Returning SQLite-ready rows keeps the database writer single
    owner while moving the CPU-heavy document work into the existing pool.
    """

    (
        records,
        language,
        split,
        path_raw,
        sampling_seed,
        validation_candidate_fraction,
    ) = payload
    if _POOL_PROCESSOR is None:  # pragma: no cover - worker invariant.
        raise RuntimeError("SentencePiece encoder worker is not initialized")
    path = Path(path_raw)
    insert_rows: list[tuple[Any, ...]] = []
    unique_tokens = 0
    physical_tokens = 0
    for line_number, byte_offset, raw_line in records:
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusFinalizationError(
                f"{path}:{line_number}: invalid UTF-8 JSON"
            ) from exc
        (
            document_id,
            source_file_sha256,
            multiplier,
            train_multiplier,
        ) = _validate_document(
            row,
            language=language,
            split=split,
            path=path,
            line_number=line_number,
            sampling_seed=sampling_seed,
            validation_candidate_fraction=validation_candidate_fraction,
        )
        token_count = len(_encode_document(_POOL_PROCESSOR, row["text"]))
        train_selection_key = hashlib.sha256(
            f"{int(sampling_seed)}\0{TRAIN_SELECTION_NAMESPACE}\0{document_id}".encode(
                "utf-8"
            )
        ).hexdigest()
        validation_selection_key = hashlib.sha256(
            f"{int(sampling_seed)}\0{VALIDATION_SELECTION_NAMESPACE}\0{document_id}".encode(
                "utf-8"
            )
        ).hexdigest()
        insert_rows.append(
            (
                document_id,
                train_selection_key,
                validation_selection_key,
                language,
                split,
                path_raw,
                byte_offset,
                len(raw_line),
                token_count,
                multiplier,
                train_multiplier,
                source_file_sha256,
            )
        )
        unique_tokens += token_count
        physical_tokens += token_count * multiplier
    return insert_rows, unique_tokens, physical_tokens


def _pool_emit_batch(
    payload: tuple[list[tuple[Any, ...]], str, str, tuple[int, ...]],
) -> tuple[str, dict[int, int], int, int, int, int]:
    """Read, verify, tokenize, and serialize selected rows in a worker.

    ``Pool.map`` preserves payload order, so concatenating these returned text
    blocks produces the exact same deterministic corpus order as the serial
    implementation.
    """

    indexed_rows, split, tokenizer_sha256, special_ids_raw = payload
    if _POOL_PROCESSOR is None:  # pragma: no cover - worker invariant.
        raise RuntimeError("SentencePiece encoder worker is not initialized")
    special_ids = set(special_ids_raw)
    unigram_counts: Counter[int] = Counter()
    excluded_special_tokens = 0
    emitted_documents = 0
    emitted_tokens = 0
    unique_original_tokens = 0
    output_lines: list[str] = []
    for indexed in indexed_rows:
        (
            input_path,
            byte_offset,
            byte_length,
            indexed_token_count,
            source_split,
            indexed_multiplier,
            indexed_train_multiplier,
        ) = indexed
        handle = _POOL_INPUT_HANDLES.get(input_path)
        if handle is None:
            handle = Path(input_path).open("rb")
            _POOL_INPUT_HANDLES[input_path] = handle
        handle.seek(int(byte_offset))
        raw_line = handle.read(int(byte_length))
        row = json.loads(raw_line.decode("utf-8"))
        if not isinstance(row, dict):
            raise AssertionError("indexed row changed after validation")
        token_ids = _encode_document(_POOL_PROCESSOR, row["text"])
        token_count = len(token_ids)
        if token_count != int(indexed_token_count):
            raise AssertionError("deterministic token count changed between passes")
        source_multiplier = int(row["lineage"]["rehydration_multiplier"])
        if source_multiplier != int(indexed_multiplier):
            raise AssertionError("source multiplier changed between indexing and emission")
        if source_split == "validation_reserve":
            preserved_train_multiplier = int(
                row["lineage"]["train_rehydration_multiplier"]
            )
            if preserved_train_multiplier != int(indexed_train_multiplier):
                raise AssertionError(
                    "candidate train multiplier changed between indexing and emission"
                )
        else:
            preserved_train_multiplier = source_multiplier
        if split == "train":
            multiplier = preserved_train_multiplier
            for token_id in token_ids:
                if token_id in special_ids:
                    excluded_special_tokens += 1
                else:
                    unigram_counts[token_id] += 1
        else:
            if source_split != "validation_reserve" or source_multiplier != 1:
                raise AssertionError("final validation contains a non-candidate document")
            multiplier = 1
        original_document_id = row["document_id"]
        unique_original_tokens += token_count
        for replica_index in range(multiplier):
            emitted = dict(row)
            emitted["original_document_id"] = original_document_id
            emitted["document_id"] = _replica_document_id(
                original_document_id, replica_index
            )
            emitted_lineage = dict(row["lineage"])
            if split == "train" and source_split == "validation_reserve":
                emitted["artifact_role"] = "train"
                emitted_lineage["split"] = "train"
                emitted_lineage["rehydration_multiplier"] = multiplier
                emitted_lineage["validation_candidate_original_split"] = (
                    "validation_reserve"
                )
                emitted_lineage["validation_candidate_reclaimed"] = True
            emitted_lineage["rehydration_replica_index"] = replica_index
            emitted["lineage"] = emitted_lineage
            emitted["ab_v2_tokenization"] = {
                "token_count_including_eod": token_count,
                "tokenizer_model_sha256": tokenizer_sha256,
                "append_eod": True,
            }
            output_lines.append(
                json.dumps(
                    emitted,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            emitted_documents += 1
            emitted_tokens += token_count
    return (
        "".join(output_lines),
        dict(unigram_counts),
        excluded_special_tokens,
        emitted_documents,
        emitted_tokens,
        unique_original_tokens,
    )


def _pool_encode_batch(
    payload: tuple[list[str], bool, tuple[int, ...]],
) -> tuple[list[int], dict[int, int], int]:
    texts, collect_unigrams, special_ids_raw = payload
    if _POOL_PROCESSOR is None:  # pragma: no cover - worker invariant.
        raise RuntimeError("SentencePiece encoder worker is not initialized")
    special_ids = set(special_ids_raw)
    counts: list[int] = []
    unigrams: Counter[int] = Counter()
    excluded = 0
    for text in texts:
        token_ids = _encode_document(_POOL_PROCESSOR, text)
        counts.append(len(token_ids))
        if collect_unigrams:
            for token_id in token_ids:
                if token_id in special_ids:
                    excluded += 1
                else:
                    unigrams[token_id] += 1
    return counts, dict(unigrams), excluded


def _encode_text_batch(
    processor: Any,
    texts: list[str],
    *,
    encoder_pool: Any | None,
    collect_unigrams: bool,
    special_ids: set[int],
) -> tuple[list[int], Counter[int], int]:
    if not texts:
        return [], Counter(), 0
    if encoder_pool is None:
        results = [
            _pool_encode_batch_local(
                processor, texts, collect_unigrams, special_ids
            )
        ]
    else:
        process_count = max(1, int(getattr(encoder_pool, "_processes", 1)))
        chunk_size = max(1, math.ceil(len(texts) / (process_count * 2)))
        payloads = [
            (
                texts[offset : offset + chunk_size],
                collect_unigrams,
                tuple(sorted(special_ids)),
            )
            for offset in range(0, len(texts), chunk_size)
        ]
        results = encoder_pool.map(_pool_encode_batch, payloads)
    counts: list[int] = []
    unigrams: Counter[int] = Counter()
    excluded = 0
    for batch_counts, batch_unigrams, batch_excluded in results:
        counts.extend(batch_counts)
        unigrams.update(batch_unigrams)
        excluded += int(batch_excluded)
    return counts, unigrams, excluded


def _pool_encode_batch_local(
    processor: Any,
    texts: list[str],
    collect_unigrams: bool,
    special_ids: set[int],
) -> tuple[list[int], dict[int, int], int]:
    counts: list[int] = []
    unigrams: Counter[int] = Counter()
    excluded = 0
    for text in texts:
        token_ids = _encode_document(processor, text)
        counts.append(len(token_ids))
        if collect_unigrams:
            for token_id in token_ids:
                if token_id in special_ids:
                    excluded += 1
                else:
                    unigrams[token_id] += 1
    return counts, dict(unigrams), excluded


def _validate_document(
    row: Any,
    *,
    language: str,
    split: str,
    path: Path,
    line_number: int,
    sampling_seed: int,
    validation_candidate_fraction: float,
) -> tuple[str, str, int, int]:
    where = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise CorpusFinalizationError(f"{where}: expected a JSON object")
    if row.get("language") != language:
        raise CorpusFinalizationError(
            f"{where}: language must be {language!r}, got {row.get('language')!r}"
        )
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise CorpusFinalizationError(f"{where}: text must be a non-empty string")
    document_id = str(row.get("document_id") or "")
    if not document_id:
        raise CorpusFinalizationError(f"{where}: document_id is required")
    lineage = row.get("lineage")
    if not isinstance(lineage, dict):
        raise CorpusFinalizationError(f"{where}: lineage must be an object")

    required = (
        "source_dataset",
        "source_revision",
        "subset",
        "source_file_sha256",
        "text_sha256",
        "split",
        "rehydration_multiplier",
    )
    missing = [key for key in required if key not in lineage]
    if missing:
        raise CorpusFinalizationError(f"{where}: lineage is missing {missing}")
    for key in ("source_dataset", "source_revision", "subset"):
        if not isinstance(lineage.get(key), str) or not lineage[key].strip():
            raise CorpusFinalizationError(f"{where}: lineage.{key} must be non-empty")
    source_id = lineage.get("source_id")
    url = lineage.get("url")
    if not str(source_id or "").strip() and not str(url or "").strip():
        raise CorpusFinalizationError(
            f"{where}: lineage requires at least one of source_id or url"
        )
    source_file_sha256 = _require_sha256(
        lineage.get("source_file_sha256"),
        label=f"{where}: lineage.source_file_sha256",
    )
    expected_text_sha256 = _require_sha256(
        lineage.get("text_sha256"),
        label=f"{where}: lineage.text_sha256",
    )
    actual_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual_text_sha256 != expected_text_sha256:
        raise CorpusFinalizationError(
            f"{where}: text SHA-256 mismatch: expected {expected_text_sha256}, "
            f"observed {actual_text_sha256}"
        )
    if lineage.get("split") != split:
        raise CorpusFinalizationError(
            f"{where}: lineage.split must be {split!r}, got {lineage.get('split')!r}"
        )
    multiplier = lineage.get("rehydration_multiplier")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        raise CorpusFinalizationError(
            f"{where}: lineage.rehydration_multiplier must be a positive integer"
        )
    if split == "validation_reserve" and multiplier != 1:
        raise CorpusFinalizationError(
            f"{where}: validation must never be rehydrated (multiplier={multiplier})"
        )
    train_multiplier = lineage.get("train_rehydration_multiplier")
    if (
        isinstance(train_multiplier, bool)
        or not isinstance(train_multiplier, int)
        or train_multiplier < 1
    ):
        raise CorpusFinalizationError(
            f"{where}: lineage.train_rehydration_multiplier must be a positive integer"
        )
    if split == "train" and train_multiplier != multiplier:
        raise CorpusFinalizationError(
            f"{where}: train rehydration multipliers disagree"
        )

    expected_document_id = stable_document_id(
        source_dataset=lineage["source_dataset"],
        source_revision=lineage["source_revision"],
        subset=lineage["subset"],
        source_id=None if source_id is None else str(source_id),
        url=None if url is None else str(url),
        text=text,
    )
    if document_id != expected_document_id:
        raise CorpusFinalizationError(
            f"{where}: document_id does not match its frozen source lineage"
        )
    assigned_to_candidate = assign_validation_reserve(
        document_id=document_id,
        seed=sampling_seed,
        validation_fraction=validation_candidate_fraction,
    )
    if assigned_to_candidate != (split == "validation_reserve"):
        raise CorpusFinalizationError(
            f"{where}: split does not match the declared pre-tokenizer validation "
            f"candidate fraction {validation_candidate_fraction}"
        )
    return document_id, source_file_sha256, multiplier, int(train_multiplier)


def _create_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = FULL;
        PRAGMA temp_store = FILE;
        CREATE TABLE documents (
            document_id TEXT NOT NULL,
            train_selection_key TEXT NOT NULL,
            validation_selection_key TEXT NOT NULL,
            language TEXT NOT NULL,
            split TEXT NOT NULL,
            input_path TEXT NOT NULL,
            byte_offset INTEGER NOT NULL,
            byte_length INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            multiplier INTEGER NOT NULL,
            train_multiplier INTEGER NOT NULL,
            source_file_sha256 TEXT NOT NULL
        );
        """
    )
    return connection


def _create_selection_indexes(connection: sqlite3.Connection) -> None:
    try:
        connection.execute(
            "CREATE UNIQUE INDEX documents_document_id_unique ON documents(document_id)"
        )
    except sqlite3.IntegrityError as exc:
        duplicate = connection.execute(
            """
            SELECT document_id, GROUP_CONCAT(language), GROUP_CONCAT(split), COUNT(*)
            FROM documents
            GROUP BY document_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is None:  # pragma: no cover - SQLite invariant.
            raise CorpusFinalizationError("duplicate document-ID index failure") from exc
        document_id, languages_raw, splits_raw, _count = duplicate
        languages = str(languages_raw).split(",")
        splits = str(splits_raw).split(",")
        if len(set(splits)) > 1:
            raise CorpusFinalizationError(
                "train/validation document-ID overlap: "
                f"{document_id} ({'/'.join(languages)}/{ '/'.join(splits)})"
            ) from exc
        raise CorpusFinalizationError(
            f"duplicate original document_id in bootstrap inputs: {document_id}"
        ) from exc
    connection.executescript(
        """
        CREATE INDEX documents_language_train_selection
            ON documents(language, train_selection_key, document_id);
        CREATE INDEX documents_language_validation_selection
            ON documents(language, split, validation_selection_key, document_id);
        CREATE TABLE selections (
            language TEXT NOT NULL,
            split TEXT NOT NULL,
            selection_rank INTEGER NOT NULL,
            document_id TEXT NOT NULL UNIQUE,
            PRIMARY KEY(language, split, selection_rank),
            FOREIGN KEY(document_id) REFERENCES documents(document_id)
        );
        """
    )


def _index_input(
    connection: sqlite3.Connection,
    input_spec: BootstrapInput,
    *,
    processor: Any,
    sampling_seed: int,
    validation_candidate_fraction: float,
    encoder_pool: Any | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    documents = 0
    unique_tokens = 0
    physical_tokens = 0
    size_bytes = 0
    serial_pending: list[tuple[str, str, int, int, int, int, str]] = []
    parallel_pending: list[tuple[int, int, bytes]] = []

    def flush_pending() -> None:
        nonlocal documents, unique_tokens, physical_tokens
        if not serial_pending and not parallel_pending:
            return
        result_batches: list[tuple[list[tuple[Any, ...]], int, int]] = []
        if encoder_pool is not None:
            process_count = max(1, int(getattr(encoder_pool, "_processes", 1)))
            chunk_size = max(
                1, math.ceil(len(parallel_pending) / (process_count * 2))
            )
            payloads = [
                (
                    parallel_pending[offset : offset + chunk_size],
                    input_spec.language,
                    input_spec.split,
                    str(input_spec.path),
                    sampling_seed,
                    validation_candidate_fraction,
                )
                for offset in range(0, len(parallel_pending), chunk_size)
            ]
            result_batches = encoder_pool.map(_pool_index_batch, payloads)
        else:
            token_counts, _, _ = _encode_text_batch(
                processor,
                [record[6] for record in serial_pending],
                encoder_pool=None,
                collect_unigrams=False,
                special_ids=set(),
            )
            insert_rows: list[tuple[Any, ...]] = []
            serial_unique_tokens = 0
            serial_physical_tokens = 0
            for record, token_count in zip(
                serial_pending, token_counts, strict=True
            ):
                (
                    document_id,
                    source_file_sha256,
                    multiplier,
                    train_multiplier,
                    byte_offset,
                    byte_length,
                    _text,
                ) = record
                train_selection_key = hashlib.sha256(
                    f"{int(sampling_seed)}\0{TRAIN_SELECTION_NAMESPACE}\0{document_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                validation_selection_key = hashlib.sha256(
                    f"{int(sampling_seed)}\0{VALIDATION_SELECTION_NAMESPACE}\0{document_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                insert_rows.append(
                    (
                        document_id,
                        train_selection_key,
                        validation_selection_key,
                        input_spec.language,
                        input_spec.split,
                        str(input_spec.path),
                        byte_offset,
                        byte_length,
                        token_count,
                        multiplier,
                        train_multiplier,
                        source_file_sha256,
                    )
                )
                serial_unique_tokens += int(token_count)
                serial_physical_tokens += int(token_count) * multiplier
            result_batches = [
                (insert_rows, serial_unique_tokens, serial_physical_tokens)
            ]
        for insert_rows, batch_unique_tokens, batch_physical_tokens in result_batches:
            connection.executemany(
                """
                INSERT INTO documents(
                    document_id, train_selection_key, validation_selection_key,
                    language, split, input_path, byte_offset, byte_length,
                    token_count, multiplier, train_multiplier, source_file_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
            documents += len(insert_rows)
            unique_tokens += int(batch_unique_tokens)
            physical_tokens += int(batch_physical_tokens)
        serial_pending.clear()
        parallel_pending.clear()

    with input_spec.path.open("rb") as handle:
        line_number = 0
        while True:
            byte_offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            digest.update(raw_line)
            size_bytes += len(raw_line)
            if not raw_line.strip():
                raise CorpusFinalizationError(
                    f"{input_spec.path}:{line_number}: blank lines are forbidden"
                )
            if encoder_pool is not None:
                parallel_pending.append((line_number, byte_offset, raw_line))
                if len(parallel_pending) >= 8192:
                    flush_pending()
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CorpusFinalizationError(
                    f"{input_spec.path}:{line_number}: invalid UTF-8 JSON"
                ) from exc
            (
                document_id,
                source_file_sha256,
                multiplier,
                train_multiplier,
            ) = _validate_document(
                row,
                language=input_spec.language,
                split=input_spec.split,
                path=input_spec.path,
                line_number=line_number,
                sampling_seed=sampling_seed,
                validation_candidate_fraction=validation_candidate_fraction,
            )
            serial_pending.append(
                (
                    document_id,
                    source_file_sha256,
                    multiplier,
                    train_multiplier,
                    byte_offset,
                    len(raw_line),
                    row["text"],
                )
            )
            if len(serial_pending) >= 4096:
                flush_pending()
        flush_pending()

    observed_sha256 = digest.hexdigest()
    if observed_sha256 != input_spec.expected_sha256:
        raise CorpusFinalizationError(
            f"bootstrap SHA-256 mismatch for {input_spec.path}: expected "
            f"{input_spec.expected_sha256}, observed {observed_sha256}"
        )
    return {
        "path": str(input_spec.path),
        "sha256": observed_sha256,
        "size_bytes": size_bytes,
        "documents": documents,
        "unique_tokens_including_eod": unique_tokens,
        "physical_tokens_including_rehydration_and_eod": physical_tokens,
    }


def _select_documents(
    connection: sqlite3.Connection,
    *,
    language: str,
    split: str,
    target_tokens: int,
) -> SelectionSummary:
    if target_tokens <= 0:
        raise CorpusFinalizationError(f"{language}/{split}: target_tokens must be positive")
    if split == "validation_reserve":
        rows = connection.execute(
            """
            SELECT document_id, token_count, 1 AS effective_multiplier
            FROM documents
            WHERE language = ? AND split = 'validation_reserve'
            ORDER BY validation_selection_key ASC, document_id ASC
            """,
            (language,),
        )
    elif split == "train":
        # Final validation is selected first.  Only candidate documents not
        # present in that selection are eligible for deterministic train
        # selection, using the multiplier preserved before tokenizer training.
        rows = connection.execute(
            """
            SELECT
                d.document_id,
                d.token_count,
                CASE
                    WHEN d.split = 'validation_reserve' THEN d.train_multiplier
                    ELSE d.multiplier
                END AS effective_multiplier
            FROM documents AS d
            WHERE d.language = ?
              AND (
                    d.split = 'train'
                    OR (
                        d.split = 'validation_reserve'
                        AND NOT EXISTS (
                            SELECT 1
                            FROM selections AS selected_validation
                            WHERE selected_validation.document_id = d.document_id
                              AND selected_validation.split = 'validation_reserve'
                        )
                    )
              )
            ORDER BY d.train_selection_key ASC, d.document_id ASC
            """,
            (language,),
        )
    else:
        raise CorpusFinalizationError(f"unsupported final selection split: {split}")
    available_tokens = 0
    selected_tokens = 0
    selected_original_documents = 0
    emitted_documents = 0
    reached_target = False
    for document_id, token_count, multiplier in rows:
        document_tokens = int(token_count) * int(multiplier)
        available_tokens += document_tokens
        if not reached_target:
            connection.execute(
                """
                INSERT INTO selections(language, split, selection_rank, document_id)
                VALUES (?, ?, ?, ?)
                """,
                (language, split, selected_original_documents, document_id),
            )
            selected_tokens += document_tokens
            selected_original_documents += 1
            emitted_documents += int(multiplier)
            reached_target = selected_tokens >= target_tokens
    return SelectionSummary(
        target_tokens=target_tokens,
        available_tokens=available_tokens,
        selected_tokens=selected_tokens,
        selected_original_documents=selected_original_documents,
        emitted_documents=emitted_documents,
    )


def _selection_rows(
    connection: sqlite3.Connection,
    *,
    language: str,
    split: str,
) -> Iterator[tuple[str, int, int, int, str, int, int]]:
    yield from connection.execute(
        """
        SELECT
            d.input_path,
            d.byte_offset,
            d.byte_length,
            d.token_count,
            d.split,
            d.multiplier,
            d.train_multiplier
        FROM selections AS s
        JOIN documents AS d ON d.document_id = s.document_id
        WHERE s.language = ? AND s.split = ?
        ORDER BY s.selection_rank ASC
        """,
        (language, split),
    )


def _validation_candidate_audit(
    connection: sqlite3.Connection,
    *,
    language: str,
    validation_selection: SelectionSummary,
) -> dict[str, Any]:
    candidate = connection.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(token_count), 0),
            COALESCE(SUM(token_count * train_multiplier), 0),
            COALESCE(SUM(train_multiplier), 0)
        FROM documents
        WHERE language = ? AND split = 'validation_reserve'
        """,
        (language,),
    ).fetchone()
    reclaimed = connection.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(d.token_count), 0),
            COALESCE(SUM(d.token_count * d.train_multiplier), 0),
            COALESCE(SUM(d.train_multiplier), 0)
        FROM documents AS d
        WHERE d.language = ?
          AND d.split = 'validation_reserve'
          AND NOT EXISTS (
                SELECT 1
                FROM selections AS selected_validation
                WHERE selected_validation.document_id = d.document_id
                  AND selected_validation.split = 'validation_reserve'
          )
        """,
        (language,),
    ).fetchone()
    selected_reclaimed = connection.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(d.token_count), 0),
            COALESCE(SUM(d.token_count * d.train_multiplier), 0),
            COALESCE(SUM(d.train_multiplier), 0)
        FROM selections AS selected_train
        JOIN documents AS d ON d.document_id = selected_train.document_id
        WHERE selected_train.language = ?
          AND selected_train.split = 'train'
          AND d.split = 'validation_reserve'
        """,
        (language,),
    ).fetchone()
    candidate_documents, candidate_tokens, candidate_train_tokens, candidate_train_copies = (
        int(value) for value in candidate
    )
    reclaimed_documents, reclaimed_tokens, reclaimed_train_tokens, reclaimed_train_copies = (
        int(value) for value in reclaimed
    )
    (
        selected_reclaimed_documents,
        selected_reclaimed_unique_tokens,
        selected_reclaimed_train_tokens,
        selected_reclaimed_train_copies,
    ) = (int(value) for value in selected_reclaimed)
    expected_reclaimed_documents = (
        candidate_documents - validation_selection.selected_original_documents
    )
    if reclaimed_documents != expected_reclaimed_documents:
        raise AssertionError(f"{language}: validation candidate document accounting drift")
    if reclaimed_tokens != candidate_tokens - validation_selection.selected_tokens:
        raise AssertionError(f"{language}: validation candidate token accounting drift")
    return {
        "status": (
            "adequate"
            if validation_selection.shortfall_tokens == 0
            else "insufficient_candidate_tokens"
        ),
        "candidate_adequate": validation_selection.shortfall_tokens == 0,
        "selection_precedes_train_selection": True,
        "candidate_documents": candidate_documents,
        "candidate_unique_tokens_including_eod": candidate_tokens,
        "candidate_train_rehydrated_tokens_if_all_reclaimed": candidate_train_tokens,
        "candidate_train_emitted_documents_if_all_reclaimed": candidate_train_copies,
        "selected_validation_original_documents": (
            validation_selection.selected_original_documents
        ),
        "selected_validation_tokens_including_eod": validation_selection.selected_tokens,
        "unused_candidate_documents_reclassified_to_train": reclaimed_documents,
        "unused_candidate_unique_tokens_reclassified_to_train": reclaimed_tokens,
        "unused_candidate_rehydrated_tokens_available_to_train": reclaimed_train_tokens,
        "unused_candidate_emitted_documents_available_to_train": reclaimed_train_copies,
        "selected_reclaimed_train_original_documents": selected_reclaimed_documents,
        "selected_reclaimed_train_unique_tokens_including_eod": (
            selected_reclaimed_unique_tokens
        ),
        "selected_reclaimed_train_tokens_including_rehydration_and_eod": (
            selected_reclaimed_train_tokens
        ),
        "selected_reclaimed_train_emitted_documents": selected_reclaimed_train_copies,
        "preserved_train_multiplier_field": "lineage.train_rehydration_multiplier",
        "selected_validation_rehydration_multiplier": 1,
    }


def _read_indexed_row(
    handles: dict[str, BinaryIO],
    *,
    input_path: str,
    byte_offset: int,
    byte_length: int,
) -> dict[str, Any]:
    handle = handles.get(input_path)
    if handle is None:
        handle = Path(input_path).open("rb")
        handles[input_path] = handle
    handle.seek(int(byte_offset))
    raw_line = handle.read(int(byte_length))
    row = json.loads(raw_line.decode("utf-8"))
    if not isinstance(row, dict):
        raise AssertionError("indexed row changed after validation")
    return row


def _replica_document_id(original_document_id: str, replica_index: int) -> str:
    if replica_index == 0:
        return original_document_id
    return f"{original_document_id}#rehydration-{replica_index:04d}"


def _write_selection_parallel(
    connection: sqlite3.Connection,
    *,
    language: str,
    split: str,
    output_path: Path,
    tokenizer_sha256: str,
    special_ids: set[int],
    encoder_pool: Any,
) -> tuple[dict[str, Any], Counter[int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unigram_counts: Counter[int] = Counter()
    emitted_documents = 0
    emitted_tokens = 0
    unique_original_tokens = 0
    excluded_special_tokens = 0
    pending: list[tuple[Any, ...]] = []
    process_count = max(1, int(getattr(encoder_pool, "_processes", 1)))

    def emit_pending(output: Any) -> None:
        nonlocal emitted_documents, emitted_tokens, unique_original_tokens
        nonlocal excluded_special_tokens
        if not pending:
            return
        chunk_size = max(1, math.ceil(len(pending) / (process_count * 2)))
        payloads = [
            (
                pending[offset : offset + chunk_size],
                split,
                tokenizer_sha256,
                tuple(sorted(special_ids)),
            )
            for offset in range(0, len(pending), chunk_size)
        ]
        results = encoder_pool.map(_pool_emit_batch, payloads)
        for (
            output_text,
            batch_unigrams,
            batch_excluded,
            batch_documents,
            batch_tokens,
            batch_unique_tokens,
        ) in results:
            output.write(output_text)
            if split == "train":
                unigram_counts.update(batch_unigrams)
                excluded_special_tokens += int(batch_excluded)
            emitted_documents += int(batch_documents)
            emitted_tokens += int(batch_tokens)
            unique_original_tokens += int(batch_unique_tokens)
        pending.clear()

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for indexed in _selection_rows(connection, language=language, split=split):
            pending.append(tuple(indexed))
            if len(pending) >= 4096:
                emit_pending(output)
        emit_pending(output)
    return (
        {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "documents": emitted_documents,
            "tokens_including_eod": emitted_tokens,
            "unique_original_tokens_including_eod": unique_original_tokens,
            "excluded_special_tokens": excluded_special_tokens,
        },
        unigram_counts,
    )


def _write_selection(
    connection: sqlite3.Connection,
    *,
    language: str,
    split: str,
    output_path: Path,
    processor: Any,
    tokenizer_sha256: str,
    special_ids: set[int],
    encoder_pool: Any | None = None,
) -> tuple[dict[str, Any], Counter[int]]:
    if encoder_pool is not None:
        return _write_selection_parallel(
            connection,
            language=language,
            split=split,
            output_path=output_path,
            tokenizer_sha256=tokenizer_sha256,
            special_ids=special_ids,
            encoder_pool=encoder_pool,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles: dict[str, BinaryIO] = {}
    unigram_counts: Counter[int] = Counter()
    emitted_documents = 0
    emitted_tokens = 0
    unique_original_tokens = 0
    excluded_special_tokens = 0
    pending: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def emit_pending(output: Any) -> None:
        nonlocal emitted_documents, emitted_tokens, unique_original_tokens
        nonlocal excluded_special_tokens
        if not pending:
            return
        token_counts, batch_unigrams, batch_excluded = _encode_text_batch(
            processor,
            [row["text"] for _, row in pending],
            encoder_pool=encoder_pool,
            collect_unigrams=split == "train",
            special_ids=special_ids,
        )
        if split == "train":
            unigram_counts.update(batch_unigrams)
            excluded_special_tokens += batch_excluded
        for (indexed, row), token_count in zip(pending, token_counts, strict=True):
            (
                _input_path,
                _byte_offset,
                _byte_length,
                indexed_token_count,
                source_split,
                indexed_multiplier,
                indexed_train_multiplier,
            ) = indexed
            if int(token_count) != int(indexed_token_count):
                raise AssertionError("deterministic token count changed between passes")
            source_multiplier = int(row["lineage"]["rehydration_multiplier"])
            if source_multiplier != int(indexed_multiplier):
                raise AssertionError("source multiplier changed between indexing and emission")
            if source_split == "validation_reserve":
                preserved_train_multiplier = int(
                    row["lineage"]["train_rehydration_multiplier"]
                )
                if preserved_train_multiplier != int(indexed_train_multiplier):
                    raise AssertionError(
                        "candidate train multiplier changed between indexing and emission"
                    )
            else:
                preserved_train_multiplier = source_multiplier
            if split == "train":
                multiplier = preserved_train_multiplier
            else:
                if source_split != "validation_reserve" or source_multiplier != 1:
                    raise AssertionError("final validation contains a non-candidate document")
                multiplier = 1
            original_document_id = row["document_id"]
            unique_original_tokens += int(token_count)
            for replica_index in range(multiplier):
                emitted = dict(row)
                emitted["original_document_id"] = original_document_id
                emitted["document_id"] = _replica_document_id(
                    original_document_id,
                    replica_index,
                )
                emitted_lineage = dict(row["lineage"])
                if split == "train" and source_split == "validation_reserve":
                    emitted["artifact_role"] = "train"
                    emitted_lineage["split"] = "train"
                    emitted_lineage["rehydration_multiplier"] = multiplier
                    emitted_lineage["validation_candidate_original_split"] = (
                        "validation_reserve"
                    )
                    emitted_lineage["validation_candidate_reclaimed"] = True
                emitted_lineage["rehydration_replica_index"] = replica_index
                emitted["lineage"] = emitted_lineage
                emitted["ab_v2_tokenization"] = {
                    "token_count_including_eod": int(token_count),
                    "tokenizer_model_sha256": tokenizer_sha256,
                    "append_eod": True,
                }
                output.write(
                    json.dumps(
                        emitted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                emitted_documents += 1
                emitted_tokens += int(token_count)
        pending.clear()

    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for indexed in _selection_rows(connection, language=language, split=split):
                (
                input_path,
                byte_offset,
                byte_length,
                _indexed_token_count,
                _source_split,
                _indexed_multiplier,
                _indexed_train_multiplier,
                ) = indexed
                row = _read_indexed_row(
                    handles,
                    input_path=input_path,
                    byte_offset=byte_offset,
                    byte_length=byte_length,
                )
                pending.append((tuple(indexed), row))
                if len(pending) >= 4096:
                    emit_pending(output)
            emit_pending(output)
    finally:
        for handle in handles.values():
            handle.close()
    return (
        {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "documents": emitted_documents,
            "tokens_including_eod": emitted_tokens,
            "unique_original_tokens_including_eod": unique_original_tokens,
            "excluded_special_tokens": excluded_special_tokens,
        },
        unigram_counts,
    )


def _source_hashes(
    connection: sqlite3.Connection,
    *,
    language: str,
    split: str,
    selected_only: bool,
) -> list[str]:
    if selected_only:
        rows = connection.execute(
            """
            SELECT DISTINCT d.source_file_sha256
            FROM selections AS s
            JOIN documents AS d ON d.document_id = s.document_id
            WHERE s.language = ? AND s.split = ?
            ORDER BY d.source_file_sha256
            """,
            (language, split),
        )
    else:
        rows = connection.execute(
            """
            SELECT DISTINCT source_file_sha256
            FROM documents
            WHERE language = ? AND split = ?
            ORDER BY source_file_sha256
            """,
            (language, split),
        )
    return [str(row[0]) for row in rows]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _selection_payload(summary: SelectionSummary) -> dict[str, Any]:
    return {
        "target_tokens": summary.target_tokens,
        "available_tokens": summary.available_tokens,
        "selected_tokens": summary.selected_tokens,
        "selected_original_documents": summary.selected_original_documents,
        "emitted_documents": summary.emitted_documents,
        "shortfall_tokens": summary.shortfall_tokens,
        "overshoot_tokens": summary.overshoot_tokens,
        "whole_document_overshoot": True,
        "selection_priority": (
            "ascending_sha256(sampling_seed, split_namespace, original_document_id)"
        ),
    }


def finalize_corpus(
    *,
    bootstrap_manifest_path: Path,
    tokenizer_model_path: Path,
    sampling_plan_path: Path,
    output_dir: Path,
    max_train_shortfall_fraction: float = DEFAULT_MAX_TRAIN_SHORTFALL_FRACTION,
    validation_target_fraction: float = DEFAULT_VALIDATION_FRACTION,
    tokenization_workers: int = 1,
    index_work_root: Path | None = None,
) -> dict[str, Any]:
    """Finalize a corpus release and return its manifest.

    Hard contract failures raise :class:`CorpusFinalizationError` and leave no
    output directory.  Insufficient source volume still emits all audit
    artifacts, but the returned manifest has ``release_status`` set to
    ``needs_incremental_source_docs`` (or the validation equivalent).  The CLI
    converts every non-ready status to exit code 2.
    """

    bootstrap_manifest_path = bootstrap_manifest_path.resolve()
    tokenizer_model_path = tokenizer_model_path.resolve()
    sampling_plan_path = sampling_plan_path.resolve()
    output_dir = output_dir.resolve()
    if index_work_root is not None:
        index_work_root = index_work_root.resolve()
        if not index_work_root.is_dir():
            raise CorpusFinalizationError(
                f"index_work_root must be an existing directory: {index_work_root}"
            )
    if tokenization_workers <= 0 or tokenization_workers > 256:
        raise CorpusFinalizationError("tokenization_workers must be in [1, 256]")
    if not 0.0 <= float(max_train_shortfall_fraction) < 1.0:
        raise CorpusFinalizationError(
            "max_train_shortfall_fraction must be in the half-open interval [0, 1)"
        )
    if (
        not math.isfinite(float(validation_target_fraction))
        or float(validation_target_fraction) != DEFAULT_VALIDATION_FRACTION
    ):
        raise CorpusFinalizationError(
            "validation_target_fraction must equal the frozen value "
            f"{DEFAULT_VALIDATION_FRACTION}"
        )
    if output_dir.exists():
        raise CorpusFinalizationError(f"output directory already exists: {output_dir}")
    if not tokenizer_model_path.is_file():
        raise CorpusFinalizationError(
            f"SentencePiece model does not exist: {tokenizer_model_path}"
        )

    plan = _load_object(sampling_plan_path, label="sampling plan")
    if not verify_plan_hash(plan):
        raise CorpusFinalizationError("sampling plan SHA-256 payload hash mismatch")
    sampling_seed = plan.get("sampling_seed")
    if (
        not isinstance(sampling_seed, int)
        or isinstance(sampling_seed, bool)
        or sampling_seed != DEFAULT_SEED
    ):
        raise CorpusFinalizationError(
            f"sampling plan sampling_seed must equal the frozen value {DEFAULT_SEED}"
        )
    physical_pool = plan.get("physical_pool")
    tiers = plan.get("tiers")
    if not isinstance(physical_pool, dict) or not physical_pool:
        raise CorpusFinalizationError("sampling plan physical_pool must be non-empty")
    if not isinstance(tiers, dict) or not tiers:
        raise CorpusFinalizationError("sampling plan tiers must be non-empty")
    required_languages = set(map(str, physical_pool))

    provenance = plan.get("provenance")
    expected_tokenizer_hash = None
    if isinstance(provenance, dict):
        expected_tokenizer_hash = provenance.get("tokenizer_model_sha256")
    expected_tokenizer_hash = _require_sha256(
        expected_tokenizer_hash,
        label="sampling plan provenance.tokenizer_model_sha256",
    )
    tokenizer_sha256 = sha256_file(tokenizer_model_path)
    if tokenizer_sha256 != expected_tokenizer_hash:
        raise CorpusFinalizationError(
            "SentencePiece model SHA-256 does not match the sampling plan: "
            f"expected {expected_tokenizer_hash}, observed {tokenizer_sha256}"
        )

    bootstrap = _load_object(bootstrap_manifest_path, label="bootstrap manifest")
    bootstrap_inputs, bootstrap_coverage = _parse_bootstrap_inputs(
        manifest_path=bootstrap_manifest_path,
        payload=bootstrap,
        required_languages=required_languages,
    )
    candidate_fractions = {
        coverage.validation_candidate_fraction
        for coverage in bootstrap_coverage.values()
    }
    if len(candidate_fractions) != 1:
        raise CorpusFinalizationError(
            "bootstrap languages do not share one validation candidate fraction"
        )
    validation_candidate_fraction = next(iter(candidate_fractions))
    if validation_candidate_fraction <= float(validation_target_fraction):
        raise CorpusFinalizationError(
            "validation candidate fraction must be strictly larger than the final "
            "validation target fraction"
        )
    processor = _load_sentencepiece(tokenizer_model_path)
    special_ids = _special_token_ids(processor)

    staging = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        raise CorpusFinalizationError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    index_workspace: Path | None = None
    if index_work_root is None:
        index_path = staging / ".document_index.sqlite3"
    else:
        index_workspace = Path(
            tempfile.mkdtemp(prefix="fineweb-finalizer-", dir=index_work_root)
        )
        index_path = index_workspace / "document_index.sqlite3"
    connection: sqlite3.Connection | None = None
    encoder_pool: Any | None = None
    try:
        if tokenization_workers > 1:
            encoder_pool = multiprocessing.get_context("fork").Pool(
                processes=tokenization_workers,
                initializer=_initialize_encoder_worker,
                initargs=(str(tokenizer_model_path),),
            )
        connection = _create_index(index_path)
        bootstrap_audit: dict[str, dict[str, Any]] = {
            language: {} for language in sorted(required_languages)
        }
        indexed_tokens = 0
        total_inputs = len(bootstrap_inputs)
        for input_index, input_spec in enumerate(bootstrap_inputs, start=1):
            indexed = _index_input(
                connection,
                input_spec,
                processor=processor,
                sampling_seed=sampling_seed,
                validation_candidate_fraction=validation_candidate_fraction,
                encoder_pool=encoder_pool,
            )
            bootstrap_audit[input_spec.language][input_spec.split] = indexed
            indexed_tokens += int(indexed["unique_tokens_including_eod"])
            if input_index % 20 == 0 or input_index == total_inputs:
                print(
                    "[PROGRESS] corpus finalizer indexed "
                    f"{input_index}/{total_inputs} language-split inputs "
                    f"({indexed_tokens:,} unique tokens including EOD)",
                    flush=True,
                )
        _create_selection_indexes(connection)
        connection.commit()

        language_results: dict[str, Any] = {}
        train_blockers: list[str] = []
        validation_blockers: list[str] = []
        validation_inventory_blockers: list[str] = []
        validation_prefix_inputs: dict[str, Any] = {}

        validation_cap = int(plan.get("validation_cap_tokens", DEFAULT_VALIDATION_CAP_TOKENS))
        if validation_cap <= 0 or validation_cap > DEFAULT_VALIDATION_CAP_TOKENS:
            raise CorpusFinalizationError(
                "sampling plan validation_cap_tokens must be in [1, 20,000,000]"
            )

        ordered_languages = sorted(required_languages)
        for language_index, language in enumerate(ordered_languages, start=1):
            raw_physical = physical_pool[language]
            if not isinstance(raw_physical, dict):
                raise CorpusFinalizationError(
                    f"sampling plan physical_pool.{language} must be an object"
                )
            train_target = int(raw_physical.get("physical_target_tokens", 0))
            unique_total_row = connection.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM documents WHERE language = ?",
                (language,),
            ).fetchone()
            unique_total_tokens = int(unique_total_row[0])
            if unique_total_tokens <= 0:
                raise CorpusFinalizationError(f"{language}: bootstrap corpus has no tokens")
            observed_documents = int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE language = ?",
                    (language,),
                ).fetchone()[0]
            )
            coverage = bootstrap_coverage[language]
            if observed_documents != coverage.unique_source_documents:
                raise CorpusFinalizationError(
                    f"{language}: indexed documents disagree with bootstrap coverage"
                )
            cap_proof_threshold = math.ceil(
                validation_cap / float(validation_target_fraction)
            )
            validation_inventory_exact = coverage.complete_source_scan
            validation_cap_proven = unique_total_tokens >= cap_proof_threshold
            if validation_inventory_exact:
                validation_target = validation_target_tokens(
                    unique_total_tokens,
                    fraction=float(validation_target_fraction),
                    cap_tokens=validation_cap,
                )
                validation_inventory_basis = "complete_source_scan_exact_U"
            elif validation_cap_proven:
                validation_target = validation_cap
                validation_inventory_basis = "observed_U_lower_bound_proves_20M_cap"
            else:
                # Retain a diagnostic artifact, but never mark the release
                # ready: the true 20% quota is unknown until acquisition is
                # expanded or the source is exhausted.
                validation_target = validation_target_tokens(
                    unique_total_tokens,
                    fraction=float(validation_target_fraction),
                    cap_tokens=validation_cap,
                )
                validation_inventory_basis = "insufficient_observed_U_provisional_only"
                validation_inventory_blockers.append(language)

            # Select the final validation panel before train so no selected
            # candidate can be reclaimed into the model-training pool.
            validation_selection = _select_documents(
                connection,
                language=language,
                split="validation_reserve",
                target_tokens=validation_target,
            )
            train_selection = _select_documents(
                connection,
                language=language,
                split="train",
                target_tokens=train_target,
            )
            candidate_audit = _validation_candidate_audit(
                connection,
                language=language,
                validation_selection=validation_selection,
            )
            candidate_audit.update(
                {
                    "declared_candidate_fraction": validation_candidate_fraction,
                    "final_validation_target_fraction": float(
                        validation_target_fraction
                    ),
                    "observed_candidate_token_fraction": (
                        candidate_audit["candidate_unique_tokens_including_eod"]
                        / unique_total_tokens
                    ),
                }
            )

            train_shortfall_fraction = (
                train_selection.shortfall_tokens / train_selection.target_tokens
            )
            train_status = "ready"
            if train_shortfall_fraction > float(max_train_shortfall_fraction):
                train_status = "needs_incremental_source_docs"
                train_blockers.append(language)
            validation_status = "ready"
            if validation_selection.shortfall_tokens > 0:
                validation_status = "needs_incremental_validation_docs"
                validation_blockers.append(language)

            train_path = staging / "train" / f"{language}.jsonl"
            validation_path = staging / "validation" / f"{language}.jsonl"
            train_file, unigram_counts = _write_selection(
                connection,
                language=language,
                split="train",
                output_path=train_path,
                processor=processor,
                tokenizer_sha256=tokenizer_sha256,
                special_ids=special_ids,
                encoder_pool=encoder_pool,
            )
            validation_file, _ = _write_selection(
                connection,
                language=language,
                split="validation_reserve",
                output_path=validation_path,
                processor=processor,
                tokenizer_sha256=tokenizer_sha256,
                special_ids=special_ids,
                encoder_pool=encoder_pool,
            )
            if train_file["tokens_including_eod"] != train_selection.selected_tokens:
                raise AssertionError(f"{language}: train output token accounting drift")
            if validation_file["tokens_including_eod"] != validation_selection.selected_tokens:
                raise AssertionError(f"{language}: validation output token accounting drift")

            unigram_path = staging / "unigram_counts" / f"{language}.json"
            unigram_payload = {
                "schema_version": SCHEMA_VERSION,
                "language": language,
                "source_split": "train_only",
                "tokenizer_model_sha256": tokenizer_sha256,
                "special_token_ids_excluded": sorted(special_ids),
                "unique_train_tokens_including_special": train_file[
                    "unique_original_tokens_including_eod"
                ],
                "exposure_train_tokens_including_special": train_selection.selected_tokens,
                "excluded_special_token_count": train_file["excluded_special_tokens"],
                "non_special_token_count": sum(unigram_counts.values()),
                "counts_by_token_id": {
                    str(token_id): unigram_counts[token_id]
                    for token_id in sorted(unigram_counts)
                },
            }
            _write_json(unigram_path, unigram_payload)

            exposure_audit: dict[str, Any] = {}
            for tier_name in sorted(tiers):
                raw_tier = tiers[tier_name]
                if not isinstance(raw_tier, dict):
                    raise CorpusFinalizationError(f"sampling plan tier {tier_name} is invalid")
                raw_exposure = raw_tier.get("target_exposure_tokens")
                if not isinstance(raw_exposure, dict):
                    raise CorpusFinalizationError(
                        f"sampling plan tier {tier_name} lacks target_exposure_tokens"
                    )
                if language not in raw_exposure:
                    continue
                exposure_tokens = int(raw_exposure[language])
                physical_tokens = train_selection.selected_tokens
                if exposure_tokens < 0 or physical_tokens <= 0:
                    raise CorpusFinalizationError(
                        f"{tier_name}/{language}: invalid exposure or physical token count"
                    )
                exposure_audit[tier_name] = {
                    "target_exposure_tokens": exposure_tokens,
                    "physical_train_tokens": physical_tokens,
                    "effective_passes": exposure_tokens / physical_tokens,
                    "full_passes": exposure_tokens // physical_tokens,
                    "partial_pass_tokens": exposure_tokens % physical_tokens,
                    "reuses_physical_pool": exposure_tokens > physical_tokens,
                }

            train_payload = _selection_payload(train_selection)
            train_payload.update(
                {
                    "status": train_status,
                    "max_allowed_shortfall_fraction": float(max_train_shortfall_fraction),
                    "shortfall_fraction": train_shortfall_fraction,
                    "output": _relative_artifact(train_path, staging),
                    "selected_source_file_sha256": _source_hashes(
                        connection,
                        language=language,
                        split="train",
                        selected_only=True,
                    ),
                    "all_bootstrap_source_file_sha256": _source_hashes(
                        connection,
                        language=language,
                        split="train",
                        selected_only=False,
                    ),
                    "unigram_counts": _relative_artifact(unigram_path, staging),
                }
            )
            validation_payload = _selection_payload(validation_selection)
            validation_payload.update(
                {
                    "status": validation_status,
                    "target_rule": "min(20,000,000, floor(0.20 * exact_unique_language_tokens))",
                    "target_fraction": float(validation_target_fraction),
                    "candidate_fraction": validation_candidate_fraction,
                    "candidate_adequate": candidate_audit["candidate_adequate"],
                    "inventory_basis": validation_inventory_basis,
                    "complete_source_scan": coverage.complete_source_scan,
                    "observed_unique_tokens_including_eod": unique_total_tokens,
                    "cap_proof_threshold_tokens": cap_proof_threshold,
                    "cap_proven_from_observed_lower_bound": validation_cap_proven,
                    "quota_exact": validation_inventory_exact or validation_cap_proven,
                    "never_rehydrated": True,
                    "output": _relative_artifact(validation_path, staging),
                    "selected_source_file_sha256": _source_hashes(
                        connection,
                        language=language,
                        split="validation_reserve",
                        selected_only=True,
                    ),
                    "all_bootstrap_source_file_sha256": _source_hashes(
                        connection,
                        language=language,
                        split="validation_reserve",
                        selected_only=False,
                    ),
                }
            )
            validation_prefix_inputs[language] = {
                "jsonl": str(validation_path.relative_to(staging)),
                "text_key": "text",
                "prefix_stem": language,
                "expected_documents": validation_selection.emitted_documents,
                "expected_tokens_including_eod": validation_selection.selected_tokens,
                "sha256": validation_file["sha256"],
            }
            language_results[language] = {
                "exact_unique_language_tokens_including_eod": (
                    unique_total_tokens if coverage.complete_source_scan else None
                ),
                "observed_unique_language_tokens_lower_bound_including_eod": unique_total_tokens,
                "source_coverage": {
                    "complete_source_scan": coverage.complete_source_scan,
                    "acquisition_status": coverage.acquisition_status,
                    "report_path": str(coverage.report_path),
                    "report_payload_sha256": coverage.report_payload_sha256,
                },
                "validation_candidate_reserve": candidate_audit,
                "bootstrap": bootstrap_audit[language],
                "train": train_payload,
                "validation": validation_payload,
                "exposure_pass_audit": exposure_audit,
            }
            if language_index % 10 == 0 or language_index == len(ordered_languages):
                print(
                    "[PROGRESS] corpus finalizer selected "
                    f"{language_index}/{len(ordered_languages)} languages",
                    flush=True,
                )

        if encoder_pool is not None:
            encoder_pool.close()
            encoder_pool.join()
            encoder_pool = None

        prefix_path = staging / "validation_prefix_inputs.json"
        _write_json(
            prefix_path,
            {
                "schema_version": SCHEMA_VERSION,
                "tokenizer_model_sha256": tokenizer_sha256,
                "append_eod": True,
                "inputs_by_language": validation_prefix_inputs,
            },
        )
        # End the selection transaction before hashing the release.  Otherwise
        # SQLite's transient rollback journal could be mistaken for a durable
        # corpus artifact while the hash manifest is assembled.
        connection.commit()

        release_status = "ready"
        if train_blockers:
            release_status = "needs_incremental_source_docs"
        elif validation_inventory_blockers:
            release_status = "needs_incremental_validation_inventory"
        elif validation_blockers:
            release_status = "needs_incremental_validation_docs"
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "release_id": "fineweb_ab_v2_finalized",
            "release_status": release_status,
            "ready": release_status == "ready",
            "blocking_languages": {
                "needs_incremental_source_docs": train_blockers,
                "needs_incremental_validation_inventory": validation_inventory_blockers,
                "needs_incremental_validation_docs": validation_blockers,
            },
            "selection_contract": {
                "unit": "whole_original_document",
                "sampling_seed": sampling_seed,
                "priority": (
                    "ascending_sha256(sampling_seed, split_namespace, "
                    "original_document_id)"
                ),
                "selection_namespaces": {
                    "train": TRAIN_SELECTION_NAMESPACE,
                    "validation": VALIDATION_SELECTION_NAMESPACE,
                    "validation_reserve_membership": "validation-reserve-v1",
                },
                "train_rehydration": "all_declared_replicas_are_atomic",
                "validation_candidate_reserve": {
                    "declared_fraction": validation_candidate_fraction,
                    "final_target_fraction": float(validation_target_fraction),
                    "selection_order": "final_validation_before_train",
                    "unused_candidate_policy": (
                        "reclassify_to_train_with_preserved_train_multiplier"
                    ),
                    "preserved_train_multiplier_field": (
                        "lineage.train_rehydration_multiplier"
                    ),
                },
                "validation_rehydration": "forbidden",
                "train_validation_disjointness": {
                    "original_document_id_overlap": 0,
                    "enforcement": (
                        "single SQLite document_id primary key with explicit "
                        "cross-split duplicate rejection"
                    ),
                },
                "token_count": "sentencepiece_ids_plus_one_eos_eod_per_emitted_document",
                "max_train_shortfall_fraction": float(max_train_shortfall_fraction),
            },
            "inputs": {
                "bootstrap_manifest": {
                    "path": str(bootstrap_manifest_path),
                    "sha256": sha256_file(bootstrap_manifest_path),
                    "provenance": bootstrap.get("provenance", {}),
                },
                "sampling_plan": {
                    "path": str(sampling_plan_path),
                    "file_sha256": sha256_file(sampling_plan_path),
                    "payload_sha256": plan["plan_sha256"],
                },
                "tokenizer_model": {
                    "path": str(tokenizer_model_path),
                    "sha256": tokenizer_sha256,
                    "vocabulary_size": int(processor.get_piece_size()),
                    "special_token_ids": sorted(special_ids),
                },
            },
            "validation_prefix_inputs": _relative_artifact(prefix_path, staging),
            "languages": language_results,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        manifest_path = staging / "corpus_manifest.json"
        _write_json(manifest_path, manifest)

        output_files = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and not path.name.startswith(index_path.name)
        }
        hash_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "sha256",
            "corpus_manifest_payload_sha256": manifest["manifest_sha256"],
            "files": output_files,
        }
        hash_manifest["hash_manifest_payload_sha256"] = canonical_json_sha256(
            hash_manifest
        )
        _write_json(staging / "hash_manifest.json", hash_manifest)

        connection.close()
        connection = None
        index_path.unlink()
        if index_workspace is not None:
            index_workspace.rmdir()
            index_workspace = None
        staging.replace(output_dir)
        return manifest
    except Exception:
        if encoder_pool is not None:
            encoder_pool.terminate()
            encoder_pool.join()
        if connection is not None:
            connection.close()
        if index_workspace is not None:
            shutil.rmtree(index_workspace, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-manifest", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--sampling-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-train-shortfall-fraction",
        type=float,
        default=DEFAULT_MAX_TRAIN_SHORTFALL_FRACTION,
    )
    parser.add_argument(
        "--validation-target-fraction",
        type=float,
        required=True,
    )
    parser.add_argument("--tokenization-workers", type=int, default=1)
    parser.add_argument("--index-work-root", type=Path)
    args = parser.parse_args()
    try:
        manifest = finalize_corpus(
            bootstrap_manifest_path=Path(args.bootstrap_manifest),
            tokenizer_model_path=Path(args.tokenizer_model),
            sampling_plan_path=Path(args.sampling_plan),
            output_dir=Path(args.output_dir),
            max_train_shortfall_fraction=float(args.max_train_shortfall_fraction),
            validation_target_fraction=float(args.validation_target_fraction),
            tokenization_workers=int(args.tokenization_workers),
            index_work_root=args.index_work_root,
        )
    except CorpusFinalizationError as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    print(
        f"[{('OK' if manifest['ready'] else 'INCOMPLETE')}] "
        f"release_status={manifest['release_status']} output={Path(args.output_dir).resolve()}"
    )
    return 0 if manifest["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusFinalizationError",
    "DEFAULT_MAX_TRAIN_SHORTFALL_FRACTION",
    "finalize_corpus",
    "main",
]
