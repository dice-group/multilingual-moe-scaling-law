"""Immutable predecessor adoption for K12 corpus migration."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.contracts.k12_adoption import (
    K12AdoptionValidationError,
    verify_completed_adoption as verify_adoption_contract,
)
from multilingual_megatron.experiments.scaling.k12_migration.common import (
    K12MigrationError,
    _install_file,
    _load_object,
    _reject_output_inside_inputs,
    _require_under,
    _verify_payload,
    _write_frozen_json,
)
from multilingual_megatron.experiments.scaling.k12_migration.constants import (
    SCHEMA_VERSION,
    TOOL_ID,
    UPSTREAM_ADOPTION_PATHS,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)


def _adoption_source_files(source: Path) -> list[tuple[str, Path]]:
    if source.is_symlink():
        raise K12MigrationError(f"adoption source is a symlink: {source}")
    if source.is_file():
        return [(".", source)]
    if not source.is_dir():
        raise K12MigrationError(f"adoption source does not exist: {source}")
    files: list[tuple[str, Path]] = []
    for candidate in sorted(source.rglob("*")):
        if candidate.is_symlink():
            raise K12MigrationError(f"adoption source contains a symlink: {candidate}")
        if candidate.is_file():
            files.append((candidate.relative_to(source).as_posix(), candidate))
    if not files:
        raise K12MigrationError(f"adoption source contains no files: {source}")
    return files


def _render_rebased_adoption_json(
    *, source: Path, relative_root: str, predecessor_root: Path, successor_root: Path
) -> bytes | None:
    sentence_manifest = (
        "corpus/tokenizer_sentences/tokenizer_sentences_100m.manifest.json"
    )
    tokenizer_result = (
        "corpus/tokenizer/sp_unigram_64k/tokenizer_training_result.json"
    )
    if relative_root not in {sentence_manifest, tokenizer_result}:
        return None
    payload = _load_object(source, label=f"adoption transform {relative_root}")
    predecessor_corpus = predecessor_root / "corpus"
    successor_corpus = successor_root / "corpus"
    if relative_root == sentence_manifest:
        _verify_payload(payload, "manifest_sha256", label="tokenizer sentence manifest")
        output = payload.get("output")
        if not isinstance(output, dict):
            raise K12MigrationError("tokenizer sentence manifest lacks output metadata")
        expected_old = predecessor_corpus / "tokenizer_sentences/tokenizer_sentences_100m.jsonl"
        if Path(str(output.get("path", ""))).resolve() != expected_old.resolve():
            raise K12MigrationError("tokenizer sentence output is not rooted in predecessor")
        output["path"] = str(
            (successor_corpus / "tokenizer_sentences/tokenizer_sentences_100m.jsonl").resolve()
        )
        payload.pop("manifest_sha256", None)
        payload["manifest_sha256"] = canonical_json_sha256(payload)
    else:
        old_sentence = (
            predecessor_corpus / "tokenizer_sentences/tokenizer_sentences_100m.jsonl"
        ).resolve()
        new_sentence = (
            successor_corpus / "tokenizer_sentences/tokenizer_sentences_100m.jsonl"
        ).resolve()
        old_tokenizer_root = (
            predecessor_corpus / "tokenizer/sp_unigram_64k"
        ).resolve()
        new_tokenizer_root = (successor_corpus / "tokenizer/sp_unigram_64k").resolve()
        source_artifact = payload.get("production_v2", {}).get("source_artifact")
        if (
            not isinstance(source_artifact, dict)
            or Path(str(source_artifact.get("path", ""))).resolve() != old_sentence
            or [Path(str(value)).resolve() for value in payload.get("source_files", [])]
            != [old_sentence]
            or Path(str(payload.get("output_dir", ""))).resolve() != old_tokenizer_root
            or Path(str(payload.get("tokenizer_model", ""))).resolve()
            != old_tokenizer_root / "tokenizer.model"
        ):
            raise K12MigrationError("tokenizer training result paths are not predecessor-pinned")
        source_artifact["path"] = str(new_sentence)
        payload["source_files"] = [str(new_sentence)]
        payload["output_dir"] = str(new_tokenizer_root)
        payload["tokenizer_model"] = str(new_tokenizer_root / "tokenizer.model")
        # The prefix is embedded inside tokenizer.model itself and therefore
        # remains the one explicitly declared provenance-only in the adoption
        # manifest.  Rewriting only this JSON field would falsify the model audit.
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_read_only_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, mode & ~0o222)
    os.replace(temporary, path)


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
    """Verify the cheap stat seal before and after every predecessor read."""

    identities: dict[str, dict[str, Any]] = {"files": {}, "directories": {}}
    for kind, key in (("file", "files"), ("directory", "directories")):
        entries = seal.get(key)
        if not isinstance(entries, dict):
            raise K12MigrationError(f"predecessor seal lacks {key}")
        for relative, raw in entries.items():
            relative_path = Path(str(relative))
            candidate = predecessor_root / relative_path
            if (
                relative_path.is_absolute()
                or not isinstance(raw, dict)
                or candidate.is_symlink()
            ):
                raise K12MigrationError(
                    f"invalid predecessor sealed {kind}: {relative}"
                )
            _require_under(
                candidate, predecessor_root, label=f"sealed {kind} {relative}"
            )
            exists = candidate.is_file() if kind == "file" else candidate.is_dir()
            observed = _sealed_stat(candidate) if exists else None
            if observed != raw.get("stat"):
                raise K12MigrationError(
                    f"predecessor sealed {kind} stat drift: {relative}"
                )
            if kind == "file" and raw.get("rehash_on_verify") is True:
                if sha256_file(candidate) != raw.get("sha256"):
                    raise K12MigrationError(
                        f"predecessor sealed compact file changed: {relative}"
                    )
            identities[key][str(relative)] = observed
    return {
        "file_count": len(identities["files"]),
        "directory_count": len(identities["directories"]),
        "stat_identity_sha256": canonical_json_sha256(identities),
    }


def _adopt_path(
    *,
    predecessor_root: Path,
    successor_root: Path,
    relative: str,
) -> dict[str, Any]:
    source = predecessor_root / relative
    destination = successor_root / relative
    source_files = _adoption_source_files(source)
    file_rows: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    destination_hashes: dict[str, str] = {}
    method_counts: dict[str, int] = {}
    total_bytes = 0
    for child_relative, source_file in source_files:
        target = destination if child_relative == "." else destination / child_relative
        full_relative = relative if child_relative == "." else f"{relative}/{child_relative}"
        transformed = _render_rebased_adoption_json(
            source=source_file,
            relative_root=full_relative,
            predecessor_root=predecessor_root,
            successor_root=successor_root,
        )
        source_hash = sha256_file(source_file)
        expected_hash = (
            hashlib.sha256(transformed).hexdigest() if transformed is not None else source_hash
        )
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or sha256_file(target) != expected_hash:
                raise K12MigrationError(f"partial adoption target differs: {target}")
            same_inode = (source_file.stat().st_dev, source_file.stat().st_ino) == (
                target.stat().st_dev,
                target.stat().st_ino,
            )
            if same_inode:
                raise K12MigrationError(
                    f"predecessor inode is shared by adoption target: {target}"
                )
            method = (
                "rebased_copy"
                if transformed is not None
                else "copy_predecessor_then_make_read_only"
            )
            if target.stat().st_mode & 0o222:
                os.chmod(target, target.stat().st_mode & ~0o222)
        elif transformed is not None:
            _atomic_read_only_bytes(
                target, transformed, source_file.stat().st_mode
            )
            method = "rebased_copy"
        else:
            method = _install_file(source_file, target, force_copy=True)
        destination_hash = sha256_file(target)
        if destination_hash != expected_hash or target.stat().st_mode & 0o222:
            raise K12MigrationError(f"adopted artifact verification failed: {target}")
        source_hashes[child_relative] = source_hash
        destination_hashes[child_relative] = destination_hash
        method_counts[method] = method_counts.get(method, 0) + 1
        total_bytes += target.stat().st_size
        file_rows[child_relative] = {
            "source_sha256": source_hash,
            "destination_sha256": destination_hash,
            "size_bytes": target.stat().st_size,
            "method": method,
            "source_mode": source_file.stat().st_mode & 0o777,
            "destination_mode": target.stat().st_mode & 0o777,
            "same_inode": (
                source_file.stat().st_dev,
                source_file.stat().st_ino,
            )
            == (target.stat().st_dev, target.stat().st_ino),
        }
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "method": (
            next(iter(method_counts))
            if len(method_counts) == 1
            else "copy_with_rebased_metadata"
        ),
        "method_counts": dict(sorted(method_counts.items())),
        "file_count": len(file_rows),
        "size_bytes": total_bytes,
        "source_tree_sha256": canonical_json_sha256(source_hashes),
        "destination_tree_sha256": canonical_json_sha256(destination_hashes),
        "tree_identity_sha256": canonical_json_sha256(
            {
                child: {
                    "source": source_hashes[child],
                    "destination": destination_hashes[child],
                }
                for child in sorted(source_hashes)
            }
        ),
        "files": file_rows,
        "predecessor_inode_sharing": False,
    }


def _verify_completed_adoption(manifest_path: Path) -> dict[str, Any]:
    """Preserve the migration-layer exception while sharing one verifier."""

    try:
        return verify_adoption_contract(manifest_path.resolve())
    except K12AdoptionValidationError as error:
        raise K12MigrationError(str(error)) from error


def verify_completed_adoption(manifest_path: Path) -> dict[str, Any]:
    """Compatibility export for callers of the original migration module."""

    return _verify_completed_adoption(manifest_path)


def adopt_upstream(
    *, predecessor_root: Path, successor_root: Path, predecessor_seal_path: Path
) -> dict[str, Any]:
    """Adopt immutable upstream corpus inputs into a distinct pipeline root."""

    predecessor_root = predecessor_root.resolve()
    successor_root = successor_root.resolve()
    predecessor_seal_path = predecessor_seal_path.resolve()
    if predecessor_root.is_symlink() or not predecessor_root.is_dir():
        raise K12MigrationError(f"predecessor root is invalid: {predecessor_root}")
    if predecessor_root == successor_root:
        raise K12MigrationError("upstream adoption requires a distinct successor root")
    _reject_output_inside_inputs(successor_root, [predecessor_root])
    seal = _load_object(predecessor_seal_path, label="predecessor data seal")
    seal_hash = _verify_payload(
        seal, "seal_payload_sha256", label="predecessor data seal"
    )
    if (
        seal.get("status") != "sealed"
        or Path(str(seal.get("release_root", ""))).resolve() != predecessor_root
    ):
        raise K12MigrationError("predecessor seal does not bind the predecessor root")
    pre_adoption_stat_check = _verify_predecessor_seal_stats(
        seal, predecessor_root=predecessor_root
    )
    manifest_path = successor_root / "corpus/k12_migration/adoption_manifest.json"
    if manifest_path.exists():
        completed = _verify_completed_adoption(manifest_path)
        post_adoption_stat_check = _verify_predecessor_seal_stats(
            seal, predecessor_root=predecessor_root
        )
        if (
            Path(str(completed.get("predecessor_root", ""))).resolve()
            != predecessor_root
            or Path(str(completed.get("successor_root", ""))).resolve()
            != successor_root
            or completed.get("predecessor_seal", {}).get("file_sha256")
            != sha256_file(predecessor_seal_path)
            or pre_adoption_stat_check != post_adoption_stat_check
        ):
            raise K12MigrationError("completed adoption belongs to different roots or seal")
        return completed
    adopted_paths = {
        relative: _adopt_path(
            predecessor_root=predecessor_root,
            successor_root=successor_root,
            relative=relative,
        )
        for relative in UPSTREAM_ADOPTION_PATHS
    }
    post_adoption_stat_check = _verify_predecessor_seal_stats(
        seal, predecessor_root=predecessor_root
    )
    if pre_adoption_stat_check != post_adoption_stat_check:
        raise K12MigrationError("predecessor seal changed while adopting upstream inputs")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "status": "complete",
        "predecessor_root": str(predecessor_root),
        "successor_root": str(successor_root),
        "predecessor_seal": {
            "path": str(predecessor_seal_path),
            "file_sha256": sha256_file(predecessor_seal_path),
            "seal_payload_sha256": seal_hash,
            "pre_adoption_stat_check": pre_adoption_stat_check,
            "post_adoption_stat_check": post_adoption_stat_check,
        },
        "embedded_predecessor_paths_are_provenance_only": True,
        "tokenizer_model_prefix_exception": str(
            predecessor_root / "corpus/tokenizer/sp_unigram_64k/tokenizer"
        ),
        "adopted_paths": adopted_paths,
        "resumability": (
            "matching partial files are hash-verified and retained; mismatches fail closed"
        ),
        "predecessor_inode_sharing": False,
    }
    manifest["adoption_manifest_sha256"] = canonical_json_sha256(manifest)
    _write_frozen_json(manifest_path, manifest)
    os.chmod(manifest_path, manifest_path.stat().st_mode & ~0o222)
    return _verify_completed_adoption(manifest_path)
