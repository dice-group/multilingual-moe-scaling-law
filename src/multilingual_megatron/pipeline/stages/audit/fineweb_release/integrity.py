"""Shared integrity, path, and predecessor-seal checks for FineWeb-v2."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.k12_adoption import (
    K12AdoptionValidationError,
    verify_completed_adoption,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V1_PATH_RE = re.compile(r"(?:^|[/_\\-])(?:h100[_-]?)?ab[_-]?v1(?:$|[/_\\-])", re.I)


class ReleaseAuditError(ValueError):
    """Raised when an immutable A/B-v2 data artifact violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseAuditError(message)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseAuditError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseAuditError(f"invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseAuditError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_sha256(value: Any, *, field: str) -> str:
    value = str(value or "")
    if _SHA256_RE.fullmatch(value) is None:
        raise ReleaseAuditError(f"{field} must be a lowercase SHA-256")
    return value


def _verify_payload(payload: Mapping[str, Any], *, key: str, label: str) -> str:
    expected = _require_sha256(payload.get(key), field=f"{label}.{key}")
    candidate = dict(payload)
    candidate.pop(key, None)
    observed = canonical_json_sha256(candidate)
    _require(observed == expected, f"{label} payload hash mismatch")
    return expected


def _resolve_file(base: Path, value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} path is missing")
    path = Path(str(value))
    path = (path if path.is_absolute() else base / path).resolve()
    _require(path.is_file(), f"missing {label}: {path}")
    return path

def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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
) -> None:
    """Ensure K12 adoption has not invalidated the predecessor's stat seal."""

    for kind, entries in (
        ("file", seal.get("files")),
        ("directory", seal.get("directories")),
    ):
        _require(isinstance(entries, dict), f"predecessor seal lacks {kind} entries")
        for relative, raw in entries.items():
            relative_path = Path(str(relative))
            path = predecessor_root / relative_path
            _require(
                not relative_path.is_absolute()
                and _under(path, predecessor_root)
                and not path.is_symlink()
                and isinstance(raw, dict),
                f"invalid predecessor sealed {kind}: {relative}",
            )
            exists = path.is_file() if kind == "file" else path.is_dir()
            _require(
                exists and _sealed_stat(path) == raw.get("stat"),
                f"predecessor sealed {kind} stat drift: {relative}",
            )
            if kind == "file" and raw.get("rehash_on_verify") is True:
                _require(
                    sha256_file(path) == raw.get("sha256"),
                    f"predecessor sealed compact file hash drift: {relative}",
                )


def _load_k12_adoption(release_root: Path) -> dict[str, Any] | None:
    """Validate the one narrow cross-root K16-to-K12 provenance exception."""

    path = release_root / "corpus" / "k12_migration" / "adoption_manifest.json"
    if not path.exists():
        return None
    payload = _load_object(path, label="K12 adoption manifest")
    _verify_payload(
        payload, key="adoption_manifest_sha256", label="K12 adoption manifest"
    )
    try:
        verified = verify_completed_adoption(path)
    except K12AdoptionValidationError as error:
        raise ReleaseAuditError(str(error)) from error
    _require(
        verified == payload,
        "K12 adoption verifier disagrees with the canonical manifest",
    )
    _require(
        payload.get("schema_version") == 1
        and payload.get("tool_id") == "fineweb_ab_v2_k12_migration_v1"
        and payload.get("status") == "complete",
        "K12 adoption manifest schema/status mismatch",
    )
    predecessor_root = Path(str(payload.get("predecessor_root", ""))).resolve()
    successor_root = Path(str(payload.get("successor_root", ""))).resolve()
    _require(successor_root == release_root, "K12 adoption belongs to another successor")
    _require(
        predecessor_root != successor_root
        and predecessor_root.is_dir()
        and payload.get("embedded_predecessor_paths_are_provenance_only") is True
        and payload.get("predecessor_inode_sharing") is False,
        "K12 adoption predecessor/path policy is invalid",
    )
    seal_meta = payload.get("predecessor_seal")
    _require(isinstance(seal_meta, dict), "K12 adoption lacks predecessor seal")
    seal_path = Path(str(seal_meta.get("path", ""))).resolve()
    _require(
        _under(seal_path, predecessor_root)
        and seal_path.is_file()
        and sha256_file(seal_path) == seal_meta.get("file_sha256"),
        "K12 adoption predecessor-seal file identity mismatch",
    )
    seal = _load_object(seal_path, label="K12 adoption predecessor seal")
    seal_payload_hash = _verify_payload(
        seal, key="seal_payload_sha256", label="K12 adoption predecessor seal"
    )
    _require(
        seal.get("status") == "sealed"
        and Path(str(seal.get("release_root", ""))).resolve() == predecessor_root
        and seal_payload_hash == seal_meta.get("seal_payload_sha256"),
        "K12 adoption predecessor seal is not valid for the declared root",
    )
    _verify_predecessor_seal_stats(seal, predecessor_root=predecessor_root)
    prefix = Path(str(payload.get("tokenizer_model_prefix_exception", ""))).resolve()
    expected_prefix = (
        predecessor_root / "corpus" / "tokenizer" / "sp_unigram_64k" / "tokenizer"
    ).resolve()
    _require(
        prefix == expected_prefix,
        "K12 adoption tokenizer model-prefix exception is not the predecessor tokenizer",
    )
    return payload


def _load_k12_adoption_lineage(
    release_root: Path,
) -> tuple[dict[str, Any] | None, tuple[Path, ...]]:
    """Verify every adoption hop and return its ordered provenance roots."""

    current_root = release_root.resolve()
    first: dict[str, Any] | None = None
    roots: list[Path] = []
    seen = {current_root}
    while True:
        adoption = _load_k12_adoption(current_root)
        if adoption is None:
            break
        if first is None:
            first = adoption
        predecessor = Path(str(adoption["predecessor_root"])).resolve()
        _require(predecessor not in seen, "K12 adoption lineage contains a cycle")
        seen.add(predecessor)
        roots.append(predecessor)
        current_root = predecessor
    return first, tuple(roots)


def _assert_fresh_path(
    path: Path,
    *,
    release_root: Path,
    label: str,
    allowed_predecessor_root: Path | None = None,
) -> None:
    rendered = path.resolve().as_posix()
    _require(_V1_PATH_RE.search(rendered) is None, f"{label} reuses a v1 path: {path}")
    marker = "/outputs/pipeline_runs/"
    if marker in rendered:
        _require(
            _under(path, release_root)
            or (
                allowed_predecessor_root is not None
                and _under(path, allowed_predecessor_root)
            ),
            f"{label} points to another pipeline release instead of this v2 root: {path}",
        )


def _resolve_source_artifact_for_audit(
    *,
    source_lock_dir: Path,
    raw: Mapping[str, Any],
    release_root: Path,
    label: str,
    allowed_predecessor_root: Path | None = None,
) -> tuple[Path, bool]:
    """Resolve a source artifact without letting K12 adoption read old data.

    The K12 successor retains the predecessor's immutable source-manifest
    payload, including historical absolute paths.  When such a path is the
    narrow allowed predecessor provenance exception, require an equal-hash
    successor-local mirror and return that mirror for every scientific check.
    """

    artifact_path = _resolve_file(
        source_lock_dir, raw.get("path"), label=label
    )
    rendered = artifact_path.resolve().as_posix()
    cross_release_adoption = (
        allowed_predecessor_root is not None
        and "/outputs/pipeline_runs/" in rendered
        and not _under(artifact_path, release_root)
    )
    if cross_release_adoption:
        _require(
            _V1_PATH_RE.search(rendered) is None,
            f"{label} reuses a v1 path: {artifact_path}",
        )
    else:
        _assert_fresh_path(
            artifact_path,
            release_root=release_root,
            label=label,
            allowed_predecessor_root=allowed_predecessor_root,
        )
    expected_hash = raw.get("sha256")
    _require(
        sha256_file(artifact_path) == expected_hash,
        f"{label} hash mismatch",
    )
    if cross_release_adoption:
        mirror = source_lock_dir / artifact_path.name
        _require(
            mirror.is_file() and not mirror.is_symlink(),
            f"{label} lacks a successor-local mirror",
        )
        _require(
            sha256_file(mirror) == expected_hash,
            f"{label} successor-local mirror hash mismatch",
        )
        return mirror.resolve(), True
    return artifact_path, False


def _cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    path = path.resolve()
    observed = cache.get(path)
    if observed is None:
        observed = sha256_file(path)
        cache[path] = observed
    return observed


def _walk_path_values(value: Any, *, key: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            joined = f"{key}.{child_key}" if key else str(child_key)
            found.extend(_walk_path_values(child, key=joined))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_path_values(child, key=f"{key}[{index}]"))
    elif isinstance(value, str):
        terminal = key.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if (
            "path" in terminal
            or "prefix" in terminal
            or terminal
            in {"input_dir", "output_dir", "source_files", "tokenizer_model"}
        ):
            found.append((key, value))
    return found


_ADOPTED_PROVENANCE_PAYLOADS = frozenset(
    {
        "source_materialization",
        "pilot_bootstrap_manifest",
        "tokenizer_acquisition_plan",
        "tokenizer_bootstrap_manifest",
        "tokenizer_sentence_manifest",
        "tokenizer_training_result",
        "token_calibration",
    }
)


def _is_predecessor_provenance_path(*, label: str, field: str) -> bool:
    """Allow predecessor paths only in explicit immutable-lineage namespaces."""

    if label in _ADOPTED_PROVENANCE_PAYLOADS:
        return True
    components = [part.split("[", 1)[0] for part in field.split(".")]
    if any(
        component == "provenance"
        or component.startswith("provenance_")
        or component.endswith("_provenance")
        for component in components
    ):
        return True
    if label == "tokenization_report" and field.startswith(
        "incremental_tokenization.predecessor."
    ):
        return True
    if label == "corpus_manifest" and re.fullmatch(
        r"languages\.[^.]+\.(?:bootstrap\.(?:train|validation_reserve)\.path|"
        r"source_coverage\.report_path)",
        field,
    ):
        return True
    return False


def _audit_embedded_paths(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    release_root: Path,
    allowed_predecessor_root: Path | None = None,
    allowed_provenance_paths: set[Path] | None = None,
    allowed_provenance_roots: tuple[Path, ...] = (),
) -> int:
    exact_provenance_paths = {
        path.resolve() for path in (allowed_provenance_paths or set())
    }
    checked = 0
    for label, payload in payloads.items():
        for field, value in _walk_path_values(payload):
            if not value or "://" in value:
                continue
            path = Path(value)
            if path.is_absolute():
                if (
                    path.resolve() in exact_provenance_paths
                    and _is_predecessor_provenance_path(label=label, field=field)
                ):
                    _require(
                        _V1_PATH_RE.search(path.resolve().as_posix()) is None,
                        f"{label}.{field} reuses a v1 path: {path}",
                    )
                    checked += 1
                    continue
                if (
                    any(_under(path, root) for root in allowed_provenance_roots)
                    and _is_predecessor_provenance_path(label=label, field=field)
                ):
                    _require(
                        _V1_PATH_RE.search(path.resolve().as_posix()) is None,
                        f"{label}.{field} reuses a v1 path: {path}",
                    )
                    checked += 1
                    continue
                _assert_fresh_path(
                    path,
                    release_root=release_root,
                    label=f"{label}.{field}",
                    allowed_predecessor_root=(
                        allowed_predecessor_root
                        if _is_predecessor_provenance_path(
                            label=label, field=field
                        )
                        else None
                    ),
                )
                checked += 1
            else:
                _require(
                    _V1_PATH_RE.search(value.replace("\\", "/")) is None,
                    f"{label}.{field} contains a stale v1 relative path: {value}",
                )
    return checked


def _verify_relative_artifact(
    *, base: Path, raw: Any, label: str, hash_cache: dict[Path, str]
) -> Path:
    _require(isinstance(raw, dict), f"{label} must be an artifact object")
    path = _resolve_file(base, raw.get("path"), label=label)
    expected_hash = _require_sha256(raw.get("sha256"), field=f"{label}.sha256")
    expected_size = raw.get("size_bytes")
    if expected_size is None:
        expected_size = raw.get("bytes")
    if expected_size is not None:
        _require(path.stat().st_size == expected_size, f"{label} size mismatch")
    observed = _cached_sha256(path, hash_cache)
    _require(observed == expected_hash, f"{label} hash mismatch")
    return path
