"""Fail-closed validation of immutable K16-to-K12 adoption manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.hashing import (
    canonical_json_sha256,
    sha256_file,
)


ADOPTION_SCHEMA_VERSION = 1
K12_MIGRATION_TOOL_ID = "fineweb_ab_v2_k12_migration_v1"
UPSTREAM_ADOPTION_PATHS = (
    "corpus/source_lock",
    "corpus/bootstrap",
    "corpus/tokenizer_sentence_acquisition_plan.json",
    "corpus/tokenizer_bootstrap",
    "corpus/tokenizer_sentences",
    "corpus/tokenizer/sp_unigram_64k",
    "corpus/token_calibration.json",
)


class K12AdoptionValidationError(RuntimeError):
    """Raised when a completed adoption no longer matches its sealed inputs."""


def _require_under(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise K12AdoptionValidationError(
            f"{label} must be under {root}: {path}"
        ) from error


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise K12AdoptionValidationError(
            f"cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise K12AdoptionValidationError(
            f"{label} must be a JSON object: {path}"
        )
    return payload


def _verify_payload(
    payload: Mapping[str, Any], key: str, *, label: str
) -> str:
    expected = payload.get(key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise K12AdoptionValidationError(f"{label} lacks {key}")
    unhashed = dict(payload)
    unhashed.pop(key, None)
    if canonical_json_sha256(unhashed) != expected:
        raise K12AdoptionValidationError(f"{label} payload hash mismatch")
    return expected


def _adoption_source_files(source: Path) -> list[tuple[str, Path]]:
    if source.is_symlink():
        raise K12AdoptionValidationError(f"adoption source is a symlink: {source}")
    if source.is_file():
        return [(".", source)]
    if not source.is_dir():
        raise K12AdoptionValidationError(
            f"adoption source does not exist: {source}"
        )
    files: list[tuple[str, Path]] = []
    for candidate in sorted(source.rglob("*")):
        if candidate.is_symlink():
            raise K12AdoptionValidationError(
                f"adoption source contains a symlink: {candidate}"
            )
        if candidate.is_file():
            files.append((candidate.relative_to(source).as_posix(), candidate))
    if not files:
        raise K12AdoptionValidationError(
            f"adoption source contains no files: {source}"
        )
    return files


def _sealed_stat(path: Path) -> dict[str, int]:
    observed = path.lstat()
    return {
        "size": int(observed.st_size),
        "inode": int(observed.st_ino),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
        "mode": int(observed.st_mode & 0o7777),
    }


def _verify_predecessor_seal_stats(
    seal: Mapping[str, Any], *, predecessor_root: Path
) -> dict[str, Any]:
    identities: dict[str, dict[str, Any]] = {"files": {}, "directories": {}}
    for kind, key in (("file", "files"), ("directory", "directories")):
        entries = seal.get(key)
        if not isinstance(entries, dict):
            raise K12AdoptionValidationError(f"predecessor seal lacks {key}")
        for relative, raw in entries.items():
            relative_path = Path(str(relative))
            candidate = predecessor_root / relative_path
            if (
                relative_path.is_absolute()
                or not isinstance(raw, dict)
                or candidate.is_symlink()
            ):
                raise K12AdoptionValidationError(
                    f"invalid predecessor sealed {kind}: {relative}"
                )
            _require_under(
                candidate, predecessor_root, label=f"sealed {kind} {relative}"
            )
            exists = candidate.is_file() if kind == "file" else candidate.is_dir()
            observed = _sealed_stat(candidate) if exists else None
            if observed != raw.get("stat"):
                raise K12AdoptionValidationError(
                    f"predecessor sealed {kind} stat drift: {relative}"
                )
            if kind == "file" and raw.get("rehash_on_verify") is True:
                if sha256_file(candidate) != raw.get("sha256"):
                    raise K12AdoptionValidationError(
                        f"predecessor sealed compact file changed: {relative}"
                    )
            identities[key][str(relative)] = observed
    return {
        "file_count": len(identities["files"]),
        "directory_count": len(identities["directories"]),
        "stat_identity_sha256": canonical_json_sha256(identities),
    }


def verify_completed_adoption(manifest_path: Path) -> dict[str, Any]:
    """Verify a completed adoption and every sealed source/destination binding."""

    manifest_path = manifest_path.resolve()
    manifest = _load_object(manifest_path, label="upstream adoption manifest")
    _verify_payload(
        manifest, "adoption_manifest_sha256", label="upstream adoption manifest"
    )
    if (
        manifest.get("schema_version") != ADOPTION_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("tool_id") != K12_MIGRATION_TOOL_ID
    ):
        raise K12AdoptionValidationError(
            "upstream adoption manifest schema/status is not complete"
        )
    if manifest.get("predecessor_inode_sharing") is not False:
        raise K12AdoptionValidationError(
            "upstream adoption does not forbid predecessor inodes"
        )
    predecessor_root = Path(str(manifest.get("predecessor_root", ""))).resolve()
    successor_root = Path(str(manifest.get("successor_root", ""))).resolve()
    canonical_path = successor_root / "corpus/k12_migration/adoption_manifest.json"
    if predecessor_root == successor_root or manifest_path != canonical_path.resolve():
        raise K12AdoptionValidationError(
            "upstream adoption root/path binding is invalid"
        )
    seal_meta = manifest.get("predecessor_seal")
    if not isinstance(seal_meta, dict):
        raise K12AdoptionValidationError(
            "upstream adoption lacks predecessor-seal evidence"
        )
    seal_path = Path(str(seal_meta.get("path", ""))).resolve()
    _require_under(seal_path, predecessor_root, label="predecessor seal")
    seal = _load_object(seal_path, label="predecessor data seal")
    seal_hash = _verify_payload(
        seal, "seal_payload_sha256", label="predecessor data seal"
    )
    if (
        seal.get("status") != "sealed"
        or Path(str(seal.get("release_root", ""))).resolve() != predecessor_root
        or seal_meta.get("file_sha256") != sha256_file(seal_path)
        or seal_meta.get("seal_payload_sha256") != seal_hash
    ):
        raise K12AdoptionValidationError(
            "upstream adoption predecessor-seal binding changed"
        )
    current_seal_check = _verify_predecessor_seal_stats(
        seal, predecessor_root=predecessor_root
    )
    if (
        seal_meta.get("pre_adoption_stat_check") != current_seal_check
        or seal_meta.get("post_adoption_stat_check") != current_seal_check
    ):
        raise K12AdoptionValidationError(
            "predecessor seal changed across upstream adoption"
        )
    adopted = manifest.get("adopted_paths")
    if not isinstance(adopted, dict) or set(adopted) != set(UPSTREAM_ADOPTION_PATHS):
        raise K12AdoptionValidationError(
            "upstream adoption manifest path coverage is invalid"
        )
    for relative, raw in adopted.items():
        if not isinstance(raw, dict):
            raise K12AdoptionValidationError(
                f"invalid adopted path receipt: {relative}"
            )
        destination = Path(str(raw.get("destination", "")))
        source = Path(str(raw.get("source", "")))
        if (
            source.resolve() != (predecessor_root / relative).resolve()
            or destination.resolve() != (successor_root / relative).resolve()
            or raw.get("predecessor_inode_sharing") is not False
        ):
            raise K12AdoptionValidationError(
                f"adopted path root binding is invalid: {relative}"
            )
        files = _adoption_source_files(destination)
        source_files = dict(_adoption_source_files(source))
        destination_hashes = {
            child: sha256_file(path) for child, path in files
        }
        source_hashes = {
            child: sha256_file(path) for child, path in source_files.items()
        }
        if (
            canonical_json_sha256(destination_hashes)
            != raw.get("destination_tree_sha256")
            or canonical_json_sha256(source_hashes) != raw.get("source_tree_sha256")
            or set(destination_hashes) != set(source_hashes)
        ):
            raise K12AdoptionValidationError(
                f"completed adoption changed: {relative}"
            )
        tree_identity = {
            child: {
                "source": source_hashes[child],
                "destination": destination_hashes[child],
            }
            for child in sorted(source_hashes)
        }
        if canonical_json_sha256(tree_identity) != raw.get("tree_identity_sha256"):
            raise K12AdoptionValidationError(
                f"adoption tree identity changed: {relative}"
            )
        recorded_files = raw.get("files")
        if not isinstance(recorded_files, dict) or set(recorded_files) != set(
            source_hashes
        ):
            raise K12AdoptionValidationError(
                f"adoption file receipt coverage changed: {relative}"
            )
        if any(path.stat().st_mode & 0o222 for _, path in files):
            raise K12AdoptionValidationError(
                f"completed adoption became writable: {relative}"
            )
        for child, destination_file in files:
            source_file = source_files.get(child)
            if source_file is None:
                raise K12AdoptionValidationError(
                    f"completed adoption source coverage changed: {relative}/{child}"
                )
            if (source_file.stat().st_dev, source_file.stat().st_ino) == (
                destination_file.stat().st_dev,
                destination_file.stat().st_ino,
            ):
                raise K12AdoptionValidationError(
                    f"completed adoption shares predecessor inode: {relative}/{child}"
                )
            row = recorded_files[child]
            if (
                not isinstance(row, dict)
                or row.get("source_sha256") != source_hashes[child]
                or row.get("destination_sha256") != destination_hashes[child]
                or row.get("same_inode") is not False
                or int(row.get("destination_mode", -1))
                != (destination_file.stat().st_mode & 0o777)
            ):
                raise K12AdoptionValidationError(
                    f"adoption file receipt changed: {relative}/{child}"
                )
    return manifest


__all__ = [
    "ADOPTION_SCHEMA_VERSION",
    "K12_MIGRATION_TOOL_ID",
    "UPSTREAM_ADOPTION_PATHS",
    "K12AdoptionValidationError",
    "verify_completed_adoption",
]
