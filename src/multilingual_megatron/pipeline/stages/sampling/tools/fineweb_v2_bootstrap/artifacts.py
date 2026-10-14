"""Atomic artifact writing, hashing, and reuse validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.pipeline.tokenizer_sentence_contract import split_sentence_like

from .contracts import OUTPUT_NAMES, SourceIdentity

def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_or_verify_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_object(path) != dict(payload):
            raise RuntimeError(f"Refusing to replace a different frozen artifact: {path}")
        return
    _atomic_write_json(path, payload)


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _load_languages(language_map_path: Path) -> tuple[str, ...]:
    payload = _load_object(language_map_path)
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError(f"Language map is not a versioned v2 map: {language_map_path}")
    languages: list[str] = []
    for key in sorted((key for key in payload if str(key).isdigit()), key=int):
        group = payload[key]
        if not isinstance(group, dict) or not isinstance(group.get("languages"), list):
            raise ValueError(f"Invalid language group {key!r}: {language_map_path}")
        for language in group["languages"]:
            language = str(language)
            if language not in languages:
                languages.append(language)
    if not languages:
        raise ValueError(f"Language map has no languages: {language_map_path}")
    return tuple(languages)


def _load_language_selection(
    *,
    language_map_path: Path,
    language_selection_map_path: Path | None,
) -> tuple[str, ...]:
    """Resolve an optional execution subset without changing corpus identity."""

    identity_languages = _load_languages(language_map_path)
    if language_selection_map_path is None:
        return identity_languages
    selected_languages = _load_languages(language_selection_map_path)
    missing = sorted(set(selected_languages) - set(identity_languages))
    if missing:
        raise ValueError(
            "Language selection map is not a subset of the identity language map: "
            + ", ".join(missing)
        )
    return selected_languages

def _record(
    *,
    text: str,
    stable_id: str,
    original_id: str,
    url: str,
    source: SourceIdentity,
    source_file: str,
    source_file_sha256: str,
    split: str,
    artifact_role: str,
    rehydration_multiplier: int,
    train_rehydration_multiplier: int,
    cluster_size: int,
) -> dict[str, Any]:
    if int(train_rehydration_multiplier) < 1:
        raise ValueError("train_rehydration_multiplier must be positive")
    if split == "train" and int(rehydration_multiplier) != int(
        train_rehydration_multiplier
    ):
        raise ValueError(
            "train records must use their preserved train rehydration multiplier"
        )
    if split == "validation_reserve" and int(rehydration_multiplier) != 1:
        raise ValueError("validation candidates must not be rehydrated")
    return {
        "document_id": stable_id,
        "language": source.subset,
        "text": text,
        "artifact_role": artifact_role,
        "lineage": {
            "source_dataset": source.dataset,
            "source_revision": source.revision,
            "subset": source.subset,
            "source_id": original_id,
            "source_file": source_file,
            "source_file_sha256": source_file_sha256,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "url": url or None,
            "split": split,
            "rehydration_multiplier": int(rehydration_multiplier),
            # The validation file is a pre-tokenizer candidate reserve.  The
            # finalizer selects the exact 20%-token panel with the frozen
            # tokenizer, then returns every unused candidate to the
            # model-training selection pool.
            # Preserve the released FineWeb2 multiplier here so reclamation is
            # exact without exposing a validation candidate to tokenizer
            # training.
            "train_rehydration_multiplier": int(train_rehydration_multiplier),
            "minhash_cluster_size": int(cluster_size),
        },
    }


def _write_jsonl_record(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def _recount_tokenizer_train_sentence_slots(
    *,
    path: Path,
    language: str,
    max_sentence_chars: int,
) -> dict[str, int]:
    """Independently recount exact train slots from a frozen bootstrap JSONL."""

    document_count = 0
    unique_sentence_count = 0
    rehydrated_sentence_slots = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            label = f"{language}:{path.name}:{line_number}"
            if not line.strip():
                raise RuntimeError(f"Cannot merge blank tokenizer train row: {label}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cannot merge invalid tokenizer train JSON: {label}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Cannot merge non-object tokenizer train row: {label}")
            lineage = row.get("lineage")
            if (
                row.get("language") != language
                or row.get("artifact_role") != "train"
                or not isinstance(lineage, dict)
                or lineage.get("subset") != language
                or lineage.get("split") != "train"
            ):
                raise RuntimeError(f"Cannot merge non-train/language row: {label}")
            text = row.get("text")
            multiplier = lineage.get("rehydration_multiplier")
            if (
                not isinstance(text, str)
                or not text.strip()
                or not isinstance(multiplier, int)
                or isinstance(multiplier, bool)
                or multiplier < 1
            ):
                raise RuntimeError(f"Cannot merge invalid tokenizer train row: {label}")
            sentence_count = sum(
                1
                for _ in split_sentence_like(
                    text,
                    max_chars=max_sentence_chars,
                )
            )
            if sentence_count <= 0:
                raise RuntimeError(
                    f"Cannot merge train row with zero sentence slots: {label}"
                )
            document_count += 1
            unique_sentence_count += sentence_count
            rehydrated_sentence_slots += sentence_count * multiplier
    if document_count <= 0:
        raise RuntimeError(f"Cannot merge empty tokenizer train input: {path}")
    return {
        "train_documents": document_count,
        "unique_train_sentences": unique_sentence_count,
        "rehydrated_train_sentence_slots": rehydrated_sentence_slots,
    }


def _validate_reusable_output(
    *,
    language_dir: Path,
    expected_config_hash: str,
) -> dict[str, Any] | None:
    report_path = language_dir / "bootstrap_report.json"
    output_paths = [language_dir / name for name in OUTPUT_NAMES]
    outputs_exist = any(path.exists() for path in output_paths)
    if not report_path.exists():
        if outputs_exist:
            raise RuntimeError(
                f"Refusing partial/unverified existing bootstrap output in {language_dir}"
            )
        return None

    report = _load_object(report_path)
    if (
        report.get("status") != "complete"
        or report.get("run_config_sha256") != expected_config_hash
    ):
        raise RuntimeError(
            f"Refusing to reuse {language_dir}: report/config/source hashes do not match"
        )
    run_config = report.get("run_config")
    if (
        not isinstance(run_config, dict)
        or canonical_json_sha256(run_config) != expected_config_hash
    ):
        raise RuntimeError(f"Refusing to reuse {language_dir}: run_config payload is corrupt")
    report_payload_hash = report.get("report_payload_sha256")
    report_without_hash = {
        key: value for key, value in report.items() if key != "report_payload_sha256"
    }
    if report_payload_hash != canonical_json_sha256(report_without_hash):
        raise RuntimeError(f"Refusing to reuse {language_dir}: report payload hash is corrupt")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError(f"Refusing to reuse {language_dir}: output lineage is missing")
    for name in OUTPUT_NAMES:
        entry = outputs.get(name)
        path = language_dir / name
        if not isinstance(entry, dict) or not path.is_file():
            raise RuntimeError(f"Refusing to reuse {language_dir}: {name} is missing")
        if (
            entry.get("sha256") != sha256_file(path)
            or int(entry.get("bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"Refusing to reuse {language_dir}: {name} hash/size changed")
    return report

def _validate_incremental_predecessor(
    *,
    predecessor_dir: Path,
    current_run_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a frozen predecessor before a larger deterministic rebuild."""

    report_path = predecessor_dir / "bootstrap_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"Incremental predecessor has no report: {predecessor_dir}")
    report = _load_object(report_path)
    if report.get("status") != "complete":
        raise RuntimeError(f"Incremental predecessor is not complete: {predecessor_dir}")
    report_without_hash = {
        key: value for key, value in report.items() if key != "report_payload_sha256"
    }
    if report.get("report_payload_sha256") != canonical_json_sha256(report_without_hash):
        raise RuntimeError(f"Incremental predecessor report hash is corrupt: {predecessor_dir}")
    prior_config = report.get("run_config")
    if not isinstance(prior_config, dict):
        raise RuntimeError(f"Incremental predecessor lacks run_config: {predecessor_dir}")
    if report.get("run_config_sha256") != canonical_json_sha256(prior_config):
        raise RuntimeError(f"Incremental predecessor run_config is corrupt: {predecessor_dir}")

    immutable_keys = (
        "tool_id",
        "language",
        "source",
        "source_lock_sha256",
        "rehydration_weights_sha256",
        "language_map_sha256",
        "source_inventory_sha256",
        "paths_manifest_sha256",
    )
    drift = [
        key for key in immutable_keys if prior_config.get(key) != current_run_config.get(key)
    ]
    prior_settings = prior_config.get("settings")
    current_settings = current_run_config.get("settings")
    if not isinstance(prior_settings, dict) or not isinstance(current_settings, dict):
        drift.append("settings")
    else:
        for key in ("seed", "validation_fraction", "calibration_documents"):
            if prior_settings.get(key) != current_settings.get(key):
                drift.append(f"settings.{key}")
        for key in ("target_unique_documents", "target_utf8_bytes"):
            if int(current_settings.get(key, 0)) < int(prior_settings.get(key, 0)):
                drift.append(f"settings.{key}_decreased")
    if drift:
        raise RuntimeError(
            f"Incremental predecessor is incompatible with the new release: {sorted(set(drift))}"
        )

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError(f"Incremental predecessor lacks output hashes: {predecessor_dir}")
    for name in OUTPUT_NAMES:
        entry = outputs.get(name)
        path = predecessor_dir / name
        if (
            not isinstance(entry, dict)
            or not path.is_file()
            or entry.get("sha256") != sha256_file(path)
            or int(entry.get("bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(
                f"Incremental predecessor output changed or is missing: {path}"
            )
    return report
