"""Pinned source identity, path manifests, and rehydration policy."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    rehydration_multiplier,
    sha256_file,
)

from .artifacts import _atomic_write_json, _atomic_write_text, _load_object, _manifest_hash
from .contracts import (
    IMMUTABLE_REVISION_RE,
    SCHEMA_VERSION,
    TOOL_ID,
    BootstrapSettings,
    SourceAvailability,
    SourceIdentity,
)


_HF_REPO_FILES_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}
_HF_FILE_SHA256_CACHE: dict[tuple[str, str, str], str] = {}

def _fineweb_parquet_adapter(
    reader: Any,
    data: dict[str, Any],
    path: str,
    row: int,
) -> dict[str, Any]:
    """Keep the Parquet shard path separate from FineWeb's WARC ``file_path``."""

    payload = dict(data)
    text = payload.pop(reader.text_key, "")
    source_id = payload.pop(reader.id_key, f"{path}/{row}")
    nested_metadata = payload.pop("metadata", {})
    if isinstance(nested_metadata, str):
        try:
            nested_metadata = json.loads(nested_metadata)
        except json.JSONDecodeError:
            nested_metadata = {"metadata": nested_metadata}
    if not isinstance(nested_metadata, dict):
        nested_metadata = {"metadata": nested_metadata}
    metadata = nested_metadata | payload
    if "file_path" in metadata:
        metadata["fineweb_warc_file_path"] = metadata.pop("file_path")
    metadata["source_parquet_path"] = str(path)
    return {
        "text": text,
        "id": str(source_id),
        "media": [],
        "metadata": metadata,
    }

def _load_source_availability(
    *,
    language: str,
    source: SourceIdentity,
    source_lock: Mapping[str, Any],
    rehydration_weights_path: Path,
    source_inventory_path: Path | None,
) -> SourceAvailability | None:
    if source.is_english:
        return None
    inventory_path = (
        source_inventory_path.resolve()
        if source_inventory_path is not None
        else rehydration_weights_path.parent / "fineweb2-language-distribution.csv"
    )
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"Materialized pinned FineWeb2 inventory is required: {inventory_path}"
        )
    expected_hash = str(
        source_lock.get("non_english_corpus", {})
        .get("language_inventory", {})
        .get("sha256", "")
    )
    observed_hash = sha256_file(inventory_path)
    if expected_hash and observed_hash != expected_hash:
        raise ValueError("Materialized FineWeb2 inventory does not match the source lock")

    matching: list[dict[str, str]] = []
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("subset") == language and row.get("split") == "train":
                matching.append(row)
    if len(matching) != 1:
        raise ValueError(
            f"Pinned FineWeb2 inventory must contain exactly one train row for {language}"
        )
    try:
        documents = int(matching[0]["documents"])
        utf8_bytes = int(matching[0]["utf8_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Pinned inventory row is invalid for {language}") from exc
    if documents <= 0 or utf8_bytes <= 0:
        raise ValueError(f"Pinned inventory availability must be positive for {language}")
    return SourceAvailability(
        documents=documents,
        utf8_bytes=utf8_bytes,
        inventory_path=str(inventory_path),
        inventory_sha256=observed_hash,
    )


def _cap_settings_to_availability(
    requested: BootstrapSettings,
    availability: SourceAvailability | None,
) -> tuple[BootstrapSettings, dict[str, Any]]:
    if availability is None:
        return requested, {
            "inventory_limited": False,
            "requested_settings": asdict(requested),
            "effective_settings": asdict(requested),
        }
    estimated_train_documents = max(
        1,
        math.floor(availability.documents * (1.0 - requested.validation_fraction)),
    )
    effective = BootstrapSettings(
        seed=requested.seed,
        validation_fraction=requested.validation_fraction,
        target_unique_documents=min(
            requested.target_unique_documents,
            availability.documents,
        ),
        target_utf8_bytes=min(requested.target_utf8_bytes, availability.utf8_bytes),
        # Keep the requested calibration ceiling. If the whole pinned source is
        # smaller, acquisition emits every deterministic train document and the
        # post-read requirement becomes min(requested, actual train documents).
        calibration_documents=requested.calibration_documents,
    )
    return effective, {
        "inventory_limited": effective != requested,
        "requested_settings": asdict(requested),
        "effective_settings": asdict(effective),
        "pinned_available_documents": availability.documents,
        "pinned_available_utf8_bytes": availability.utf8_bytes,
        "estimated_available_train_documents_after_reserve": estimated_train_documents,
        "inventory_path": availability.inventory_path,
        "inventory_sha256": availability.inventory_sha256,
    }

def _source_identity(source_lock: Mapping[str, Any], language: str) -> SourceIdentity:
    is_english = language == "eng_Latn"
    source_key = "english_corpus" if is_english else "non_english_corpus"
    source = source_lock.get(source_key)
    if not isinstance(source, dict):
        raise ValueError(f"Source lock is missing {source_key}")
    dataset = str(source.get("dataset", ""))
    revision = str(source.get("dataset_revision", ""))
    if not dataset or not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ValueError(f"{source_key} must pin a dataset and a 40-hex immutable revision")
    return SourceIdentity(
        dataset=dataset,
        revision=revision,
        subset=language,
        is_english=is_english,
    )


def _ordered_paths(paths: Sequence[str], *, seed: int, subset: str) -> list[tuple[str, str]]:
    keyed = [
        (
            hashlib.sha256(f"{int(seed)}\0{subset}\0{path}".encode("utf-8")).hexdigest(),
            str(path),
        )
        for path in paths
    ]
    return [(path, order_hash) for order_hash, path in sorted(keyed)]


def _list_hf_parquet_paths(source: SourceIdentity) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("huggingface_hub is required for production FineWeb access") from exc

    cache_key = (source.dataset, source.revision)
    if cache_key not in _HF_REPO_FILES_CACHE:
        _HF_REPO_FILES_CACHE[cache_key] = tuple(
            list_repo_files(
                source.dataset,
                revision=source.revision,
                repo_type="dataset",
            )
        )
    files = _HF_REPO_FILES_CACHE[cache_key]
    if source.is_english:
        candidates = [
            path for path in files if path.startswith("data/") and path.endswith(".parquet")
        ]
    else:
        prefix = f"data/{source.subset}/train/"
        candidates = [
            path for path in files if path.startswith(prefix) and path.endswith(".parquet")
        ]
    if not candidates:
        raise RuntimeError(
            f"No Parquet files found for {source.dataset}@{source.revision}:{source.subset}"
        )
    return candidates


def _hf_file_sha256(
    source: SourceIdentity,
    paths: Sequence[str],
) -> dict[str, str]:
    """Resolve the pinned LFS SHA-256 for every source Parquet shard."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("huggingface_hub is required for production FineWeb access") from exc

    missing = [
        path
        for path in paths
        if (source.dataset, source.revision, path) not in _HF_FILE_SHA256_CACHE
    ]
    api = HfApi()
    for start in range(0, len(missing), 500):
        chunk = missing[start : start + 500]
        metadata = api.get_paths_info(
            source.dataset,
            chunk,
            revision=source.revision,
            repo_type="dataset",
        )
        by_path = {str(entry.path): entry for entry in metadata}
        for path in chunk:
            entry = by_path.get(path)
            lfs = getattr(entry, "lfs", None)
            digest = str(getattr(lfs, "sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError(
                    f"Pinned source shard has no LFS SHA-256: "
                    f"{source.dataset}@{source.revision}:{path}"
                )
            _HF_FILE_SHA256_CACHE[(source.dataset, source.revision, path)] = digest
    return {
        path: _HF_FILE_SHA256_CACHE[(source.dataset, source.revision, path)]
        for path in paths
    }


def _resolve_local_parquets(
    input_path: Path,
    *,
    language: str,
    multiple_languages: bool,
) -> tuple[Path, list[str]]:
    rendered = Path(str(input_path).replace("{language}", language)).resolve()
    if rendered.is_file():
        if rendered.suffix != ".parquet":
            raise ValueError(f"Local input file must be Parquet: {rendered}")
        return rendered.parent, [rendered.name]
    if not rendered.is_dir():
        raise FileNotFoundError(f"Local input path does not exist: {rendered}")

    candidates = (
        rendered / "data" / language / "train",
        rendered / language / "train",
        rendered / language,
    )
    selected = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if selected is None:
        if multiple_languages and "{language}" not in str(input_path):
            raise ValueError(
                "Multi-language local input must contain per-language directories or use "
                "a {language} path template"
            )
        selected = rendered
    files = sorted(path for path in selected.rglob("*.parquet") if path.is_file())
    if not files:
        raise RuntimeError(f"No local Parquet files found for {language}: {selected}")
    return selected, [path.relative_to(selected).as_posix() for path in files]


def _build_paths_manifest(
    *,
    source: SourceIdentity,
    seed: int,
    local_input_path: Path | None,
    multiple_languages: bool,
) -> tuple[dict[str, Any], str]:
    if local_input_path is None:
        data_folder = source.hf_root
        paths = _list_hf_parquet_paths(source)
        file_entries = [
            {
                "path": path,
                "order_sha256": order_hash,
            }
            for path, order_hash in _ordered_paths(paths, seed=seed, subset=source.subset)
        ]
        mode = "huggingface_pinned_revision"
    else:
        folder, paths = _resolve_local_parquets(
            local_input_path,
            language=source.subset,
            multiple_languages=multiple_languages,
        )
        data_folder = str(folder)
        file_entries = []
        for path, order_hash in _ordered_paths(paths, seed=seed, subset=source.subset):
            local_file = folder / path
            file_entries.append(
                {
                    "path": path,
                    "order_sha256": order_hash,
                    "content_sha256": sha256_file(local_file),
                    "bytes": local_file.stat().st_size,
                }
            )
        mode = "local_fixture"

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "mode": mode,
        "dataset": source.dataset,
        "revision": source.revision,
        "subset": source.subset,
        "seed": int(seed),
        "data_folder": data_folder,
        "files": file_entries,
        "ordered_paths_sha256": canonical_json_sha256([entry["path"] for entry in file_entries]),
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return manifest, data_folder


def _prepare_paths_cache(
    *,
    language_dir: Path,
    source: SourceIdentity,
    seed: int,
    local_input_path: Path | None,
    multiple_languages: bool,
) -> tuple[dict[str, Any], str, Path]:
    manifest_path = language_dir / "paths_manifest.json"
    paths_path = language_dir / "paths.txt"

    # An immutable HF revision makes a valid cached listing safe to reuse.  A
    # local fixture is re-hashed every time so changing test/source bytes cannot
    # silently reuse prior output.
    manifest: dict[str, Any] | None = None
    if local_input_path is None and manifest_path.exists():
        cached = _load_object(manifest_path)
        identity_matches = (
            cached.get("dataset") == source.dataset
            and cached.get("revision") == source.revision
            and cached.get("subset") == source.subset
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("mode") == "huggingface_pinned_revision"
            and cached.get("manifest_sha256") == _manifest_hash(cached)
        )
        if identity_matches and isinstance(cached.get("files"), list) and cached["files"]:
            manifest = cached

    if manifest is None:
        manifest, data_folder = _build_paths_manifest(
            source=source,
            seed=seed,
            local_input_path=local_input_path,
            multiple_languages=multiple_languages,
        )
        _atomic_write_json(manifest_path, manifest)
    else:
        data_folder = str(manifest["data_folder"])

    ordered_paths = [str(entry["path"]) for entry in manifest["files"]]
    _atomic_write_text(paths_path, "".join(f"{path}\n" for path in ordered_paths))
    return manifest, data_folder, paths_path


def _load_protected_paths_cache(
    *,
    language_dir: Path,
    source: SourceIdentity,
    seed: int,
    local_input_path: Path | None,
    multiple_languages: bool,
) -> tuple[dict[str, Any], str, Path]:
    """Load provenance without modifying any already-produced artifact."""

    manifest_path = language_dir / "paths_manifest.json"
    paths_path = language_dir / "paths.txt"
    if not manifest_path.is_file() or not paths_path.is_file():
        raise RuntimeError(
            f"Refusing to reuse {language_dir}: cached source paths do not match or are missing"
        )
    manifest = _load_object(manifest_path)
    expected_mode = (
        "local_fixture" if local_input_path is not None else "huggingface_pinned_revision"
    )
    identity_matches = (
        manifest.get("dataset") == source.dataset
        and manifest.get("revision") == source.revision
        and manifest.get("subset") == source.subset
        and int(manifest.get("seed", -1)) == int(seed)
        and manifest.get("mode") == expected_mode
        and manifest.get("manifest_sha256") == _manifest_hash(manifest)
        and isinstance(manifest.get("files"), list)
        and bool(manifest["files"])
    )
    if not identity_matches:
        raise RuntimeError(
            f"Refusing to reuse {language_dir}: cached source paths do not match the request"
        )

    if local_input_path is not None:
        fresh_manifest, _ = _build_paths_manifest(
            source=source,
            seed=seed,
            local_input_path=local_input_path,
            multiple_languages=multiple_languages,
        )
        if fresh_manifest != manifest:
            raise RuntimeError(
                f"Refusing to reuse {language_dir}: local source hashes do not match"
            )

    expected_paths = "".join(f"{entry['path']}\n" for entry in manifest["files"])
    if paths_path.read_text(encoding="utf-8") != expected_paths:
        raise RuntimeError(
            f"Refusing to reuse {language_dir}: cached source paths do not match their manifest"
        )
    return manifest, str(manifest["data_folder"]), paths_path


def _language_thresholds(
    weights: Mapping[str, Any],
    *,
    language: str,
    is_english: bool,
    rehydration_lock: Mapping[str, Any],
) -> tuple[dict[str, int], str]:
    """Resolve one audited rehydration policy without implicit fallbacks."""

    if is_english:
        return {"1": 1}, "english_fineweb_no_rehydration"
    thresholds = weights.get(language)
    overrides = rehydration_lock.get("identity_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("Source lock rehydration identity_overrides must be an object")
    if isinstance(thresholds, dict):
        if language in overrides:
            raise ValueError(
                f"Identity override for {language} conflicts with pinned upstream weights"
            )
        return _validated_thresholds(thresholds), "pinned_upstream_weights"

    if (
        rehydration_lock.get("missing_weights_policy")
        != "fail_closed_except_explicit_identity_overrides"
    ):
        raise ValueError(
            f"Pinned rehydration weights have no entry for {language}; "
            "source lock does not permit an explicit identity override"
        )
    override = overrides.get(language)
    if not isinstance(override, dict):
        raise ValueError(f"Pinned rehydration weights have no entry for {language}")
    required_keys = {
        "thresholds",
        "upstream_weights_present",
        "upstream_distribution_present",
        "weights_generation_commit",
        "inventory_addition_commit",
        "reason",
    }
    if set(override) != required_keys:
        raise ValueError(
            f"Explicit identity override for {language} has invalid fields"
        )
    for commit_key in ("weights_generation_commit", "inventory_addition_commit"):
        if not isinstance(override[commit_key], str) or not IMMUTABLE_REVISION_RE.fullmatch(
            override[commit_key]
        ):
            raise ValueError(
                f"Explicit identity override for {language} has invalid {commit_key}"
            )
    reason = override.get("reason")
    if (
        override.get("upstream_weights_present") is not False
        or override.get("upstream_distribution_present") is not False
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError(
            f"Explicit identity override for {language} lacks audited upstream status/reason"
        )
    normalized = _validated_thresholds(override.get("thresholds"))
    if normalized != {"1": 1}:
        raise ValueError(
            f"Explicit identity override for {language} must be exactly {{'1': 1}}"
        )
    return normalized, "source_lock_explicit_identity_override"


def _validated_thresholds(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("rehydration thresholds must be a non-empty object")
    normalized: dict[str, int] = {}
    for raw_limit, raw_weight in raw.items():
        if (
            not isinstance(raw_limit, str)
            or re.fullmatch(r"[1-9][0-9]*", raw_limit) is None
            or isinstance(raw_weight, bool)
            or not isinstance(raw_weight, int)
        ):
            raise ValueError("rehydration thresholds must use canonical positive integer fields")
        limit = int(raw_limit)
        weight = int(raw_weight)
        if weight < 1 or weight > 10:
            raise ValueError("up_to_10_reps weights must be between one and ten")
        normalized[str(limit)] = weight
    # Force the piecewise policy validation before any Parquet read.
    rehydration_multiplier(1, normalized)
    return dict(sorted(normalized.items(), key=lambda row: int(row[0])))


def _cluster_size(metadata: Mapping[str, Any], *, is_english: bool, document_id: str) -> int:
    if is_english:
        return 1
    raw_value = metadata.get("minhash_cluster_size")
    if raw_value is None:
        raise ValueError(f"FineWeb2 document {document_id} has no minhash_cluster_size")
    try:
        cluster_size = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"FineWeb2 document {document_id} has invalid minhash_cluster_size={raw_value!r}"
        ) from exc
    if cluster_size < 1:
        raise ValueError(f"FineWeb2 document {document_id} has cluster size below one")
    return cluster_size
