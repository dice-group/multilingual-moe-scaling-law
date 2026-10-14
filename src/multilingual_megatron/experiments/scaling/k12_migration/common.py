"""Shared integrity, filesystem, plan, and receipt helpers for K12 migration."""

from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
    verify_plan_hash,
)
from multilingual_megatron.experiments.scaling.k12_migration.constants import (
    EXPECTED_ACQUISITION,
    EXPECTED_INCREASED,
    NEW_TIERS,
    OLD_TIERS,
    REMOVED_FROM_K16,
    SCHEMA_VERSION,
    SHARED_TIERS,
    TOOL_ID,
)


class K12MigrationError(RuntimeError):
    """Raised when an immutable K12 migration contract is not satisfied."""


def _migration_roots(migration: Mapping[str, Any]) -> tuple[Path, Path]:
    adoption = migration["upstream_adoption"]
    return (
        Path(str(adoption["predecessor_root"])).resolve(),
        Path(str(adoption["successor_root"])).resolve(),
    )


def _require_under(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise K12MigrationError(f"{label} must be under {root}: {path}") from error


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise K12MigrationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise K12MigrationError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_without(payload: Mapping[str, Any], key: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(key, None)
    return canonical_json_sha256(unhashed)


def _verify_payload(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    expected = payload.get(key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise K12MigrationError(f"{label} lacks {key}")
    if _canonical_without(payload, key) != expected:
        raise K12MigrationError(f"{label} payload hash mismatch")
    return expected


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise K12MigrationError(f"frozen JSON target is not a regular file: {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise K12MigrationError(f"refusing to replace a different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _receipt(command: str, details: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "command": command,
        "status": "complete",
        **dict(details),
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    return payload


def _verify_plan(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_object(path, label=label)
    if not verify_plan_hash(payload):
        raise K12MigrationError(f"{label} plan hash mismatch: {path}")
    return payload


def _safe_artifact(root: Path, relative: Any, *, label: str) -> Path:
    text = str(relative or "")
    if not text or Path(text).is_absolute():
        raise K12MigrationError(f"{label} must be a non-empty relative path")
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise K12MigrationError(f"{label} escapes its release root: {relative}") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise K12MigrationError(f"{label} is missing or not a regular file: {candidate}")
    return candidate


def _reject_output_inside_inputs(output: Path, inputs: Sequence[Path]) -> None:
    resolved_output = output.resolve()
    for source in inputs:
        resolved_source = source.resolve()
        if resolved_output == resolved_source:
            raise K12MigrationError(f"output must differ from immutable input: {source}")
        try:
            resolved_output.relative_to(resolved_source)
        except ValueError:
            continue
        raise K12MigrationError(
            f"output may not be created inside immutable input {resolved_source}"
        )


def _install_file(source: Path, target: Path, *, force_copy: bool = False) -> str:
    """Install one immutable file without sharing a writable predecessor inode."""

    if source.is_symlink() or not source.is_file():
        raise K12MigrationError(f"clone source is not a regular file: {source}")
    if target.exists() or target.is_symlink():
        raise K12MigrationError(f"clone target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    if force_copy or source_stat.st_mode & 0o222:
        try:
            shutil.copy2(source, target)
            os.chmod(target, target.stat().st_mode & ~0o222)
        except OSError as error:
            raise K12MigrationError(f"cannot copy source {source}: {error}") from error
        method = (
            "copy_predecessor_then_make_read_only"
            if force_copy
            else "copy_writable_source_then_make_read_only"
        )
    else:
        try:
            os.link(source, target)
        except OSError as error:
            raise K12MigrationError(f"cannot hard-link {source} to {target}: {error}") from error
        method = "hardlink_read_only_source"
    target_stat = target.stat()
    if sha256_file(source) != sha256_file(target) or source_stat.st_size != target_stat.st_size:
        raise K12MigrationError(f"clone content verification failed: {target}")
    same_inode = (source_stat.st_dev, source_stat.st_ino) == (
        target_stat.st_dev,
        target_stat.st_ino,
    )
    if method.startswith("hardlink") != same_inode:
        raise K12MigrationError(f"clone inode-policy verification failed: {target}")
    if target_stat.st_mode & 0o222:
        raise K12MigrationError(f"installed successor artifact remains writable: {target}")
    return method


def _tier_languages(plan: Mapping[str, Any], tier: str) -> list[str]:
    tiers = plan.get("tiers")
    raw = tiers.get(tier) if isinstance(tiers, dict) else None
    languages = raw.get("languages") if isinstance(raw, dict) else None
    if not isinstance(languages, list) or not all(
        isinstance(language, str) and language for language in languages
    ):
        raise K12MigrationError(f"sampling plan has invalid {tier}.languages")
    if len(languages) != len(set(languages)):
        raise K12MigrationError(f"sampling plan {tier} contains duplicate languages")
    return list(languages)


def _tier_exposure(plan: Mapping[str, Any], tier: str, language: str) -> int | None:
    raw_tier = plan["tiers"][tier]
    exposure = raw_tier.get("target_exposure_tokens")
    if not isinstance(exposure, dict):
        raise K12MigrationError(f"sampling plan {tier} lacks target_exposure_tokens")
    if language not in exposure:
        return None
    value = exposure[language]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise K12MigrationError(f"sampling plan has invalid {tier}/{language} exposure")
    return value


def _physical_entry(plan: Mapping[str, Any], language: str) -> dict[str, Any]:
    pool = plan.get("physical_pool")
    raw = pool.get(language) if isinstance(pool, dict) else None
    if not isinstance(raw, dict):
        raise K12MigrationError(f"sampling plan lacks physical_pool.{language}")
    for field in ("physical_target_tokens", "validation_target_tokens"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise K12MigrationError(f"sampling plan has invalid {language}.{field}")
    return raw


def _validate_transition_shape(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    old_keys = tuple(sorted((old.get("tiers") or {}).keys()))
    new_keys = tuple(sorted((new.get("tiers") or {}).keys()))
    if old_keys != tuple(sorted(OLD_TIERS)) or new_keys != tuple(sorted(NEW_TIERS)):
        raise K12MigrationError(
            f"expected tier transition {OLD_TIERS} -> {NEW_TIERS}, got {old_keys} -> {new_keys}"
        )
    for tier in OLD_TIERS:
        if len(_tier_languages(old, tier)) != int(tier[1:]):
            raise K12MigrationError(f"predecessor {tier} language count is invalid")
    for tier in NEW_TIERS:
        if len(_tier_languages(new, tier)) != int(tier[1:]):
            raise K12MigrationError(f"successor {tier} language count is invalid")
    for tier in SHARED_TIERS:
        if _tier_languages(old, tier) != _tier_languages(new, tier):
            raise K12MigrationError(f"shared {tier} membership or order changed")
        if old["tiers"][tier] != new["tiers"][tier]:
            raise K12MigrationError(f"shared {tier} payload changed")
    old_k16 = _tier_languages(old, "K16")
    new_k12 = _tier_languages(new, "K12")
    if set(old_k16) - set(new_k12) != REMOVED_FROM_K16:
        raise K12MigrationError(
            "K12 must remove exactly Indonesian, Dutch, Swedish, and Turkish from K16"
        )
    if [language for language in old_k16 if language not in REMOVED_FROM_K16] != new_k12:
        raise K12MigrationError("K12 order must be the inherited K16 order")
    if not set(_tier_languages(new, "K1")).issubset(new_k12):
        raise K12MigrationError("K1 must be nested in K12")
    if not set(new_k12).issubset(_tier_languages(new, "K50")):
        raise K12MigrationError("K12 must be nested in K50")
    if not set(_tier_languages(new, "K50")).issubset(_tier_languages(new, "K200")):
        raise K12MigrationError("K50 must be nested in K200")
    old_pool = old.get("physical_pool")
    new_pool = new.get("physical_pool")
    if not isinstance(old_pool, dict) or not isinstance(new_pool, dict):
        raise K12MigrationError("both plans require physical_pool objects")
    if set(old_pool) != set(new_pool) or set(new_pool) != set(_tier_languages(new, "K200")):
        raise K12MigrationError("physical-pool coverage must remain exactly K200")


def _expected_exposure_audit(
    plan: Mapping[str, Any], language: str, physical_tokens: int
) -> dict[str, Any]:
    if physical_tokens <= 0:
        raise K12MigrationError(f"{language}: physical train token count must be positive")
    audit: dict[str, Any] = {}
    exposure_groups = dict(plan["tiers"])
    composition_arms = plan.get("composition_arms", {})
    if composition_arms:
        if not isinstance(composition_arms, dict):
            raise K12MigrationError("sampling plan composition_arms must be an object")
        exposure_groups.update(
            {f"composition:{name}": payload for name, payload in composition_arms.items()}
        )
    for tier in sorted(exposure_groups):
        exposure = exposure_groups[tier].get("target_exposure_tokens")
        if not isinstance(exposure, dict):
            raise K12MigrationError(f"sampling plan {tier} lacks target_exposure_tokens")
        target = exposure.get(language)
        if target is None:
            continue
        audit[tier] = {
            "target_exposure_tokens": target,
            "physical_train_tokens": physical_tokens,
            "effective_passes": target / physical_tokens,
            "full_passes": target // physical_tokens,
            "partial_pass_tokens": target % physical_tokens,
            "reuses_physical_pool": target > physical_tokens,
        }
    return audit


def _validate_language_against_plan(
    language: str,
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    require_exposure: bool,
) -> None:
    physical = _physical_entry(plan, language)
    train = payload.get("train")
    validation = payload.get("validation")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise K12MigrationError(f"{language}: release payload lacks train/validation")
    if int(train.get("target_tokens", -1)) != int(physical["physical_target_tokens"]):
        raise K12MigrationError(f"{language}: train target disagrees with sampling plan")
    if int(validation.get("target_tokens", -1)) != int(
        physical["validation_target_tokens"]
    ):
        raise K12MigrationError(f"{language}: validation target disagrees with sampling plan")
    selected = int(train.get("selected_tokens", 0))
    if selected <= 0:
        raise K12MigrationError(f"{language}: release has no selected train tokens")
    if require_exposure and payload.get("exposure_pass_audit") != _expected_exposure_audit(
        plan, language, selected
    ):
        raise K12MigrationError(f"{language}: exposure audit disagrees with sampling plan")


def _verify_release(root: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise K12MigrationError(f"{label} is not a regular directory: {root}")
    manifest_path = root / "corpus_manifest.json"
    hash_path = root / "hash_manifest.json"
    manifest = _load_object(manifest_path, label=f"{label} corpus manifest")
    manifest_hash = _verify_payload(manifest, "manifest_sha256", label=f"{label} corpus manifest")
    if manifest.get("ready") is not True or manifest.get("release_status") != "ready":
        raise K12MigrationError(f"{label} is not ready")
    hash_manifest = _load_object(hash_path, label=f"{label} hash manifest")
    _verify_payload(
        hash_manifest, "hash_manifest_payload_sha256", label=f"{label} hash manifest"
    )
    if hash_manifest.get("corpus_manifest_payload_sha256") != manifest_hash:
        raise K12MigrationError(f"{label} corpus/hash-manifest payload mismatch")
    files = hash_manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise K12MigrationError(f"{label} hash manifest has no files")
    for relative, expected in files.items():
        path = _safe_artifact(root, relative, label=f"{label} artifact")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise K12MigrationError(f"{label} artifact hash mismatch: {relative}")
    if files.get("corpus_manifest.json") != sha256_file(manifest_path):
        raise K12MigrationError(f"{label} does not hash corpus_manifest.json")
    languages = manifest.get("languages")
    if not isinstance(languages, dict) or not languages:
        raise K12MigrationError(f"{label} has no language payloads")
    return manifest, hash_manifest


def _verify_bootstrap(root: Path, *, label: str) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "bootstrap_manifest.json"
    manifest = _load_object(manifest_path, label=f"{label} manifest")
    _verify_payload(manifest, "manifest_sha256", label=f"{label} manifest")
    languages = manifest.get("languages")
    if not isinstance(languages, dict) or not languages:
        raise K12MigrationError(f"{label} has no languages")
    for language, raw in languages.items():
        if not isinstance(raw, dict):
            raise K12MigrationError(f"{label}/{language} entry is invalid")
        acquisition = raw.get("acquisition")
        if not isinstance(acquisition, dict):
            raise K12MigrationError(f"{label}/{language} lacks acquisition metadata")
        report_path = _safe_artifact(
            root, acquisition.get("report_path"), label=f"{label}/{language} report"
        )
        report = _load_object(report_path, label=f"{label}/{language} report")
        report_hash = _verify_payload(
            report, "report_payload_sha256", label=f"{label}/{language} report"
        )
        if acquisition.get("report_payload_sha256") != report_hash:
            raise K12MigrationError(f"{label}/{language} report hash metadata mismatch")
        outputs = report.get("outputs")
        if not isinstance(outputs, dict):
            raise K12MigrationError(f"{label}/{language} report lacks outputs")
        for split in ("train", "validation_reserve"):
            entry = raw.get(split)
            output = outputs.get(f"{split}.jsonl")
            if not isinstance(entry, dict) or not isinstance(output, dict):
                raise K12MigrationError(f"{label}/{language}/{split} metadata is invalid")
            artifact = _safe_artifact(
                root, entry.get("path"), label=f"{label}/{language}/{split}"
            )
            expected = entry.get("sha256")
            if expected != output.get("sha256") or sha256_file(artifact) != expected:
                raise K12MigrationError(f"{label}/{language}/{split} hash mismatch")
        for name, output in outputs.items():
            if not isinstance(output, dict):
                raise K12MigrationError(f"{label}/{language}/{name} output is invalid")
            artifact = _safe_artifact(
                report_path.parent,
                output.get("path", name),
                label=f"{label}/{language}/{name}",
            )
            if sha256_file(artifact) != output.get("sha256"):
                raise K12MigrationError(f"{label}/{language}/{name} changed")
    return manifest


def _load_bootstrap_manifest_only(root: Path, *, label: str) -> dict[str, Any]:
    manifest_path = root.resolve() / "bootstrap_manifest.json"
    manifest = _load_object(manifest_path, label=f"{label} manifest")
    _verify_payload(manifest, "manifest_sha256", label=f"{label} manifest")
    if not isinstance(manifest.get("languages"), dict):
        raise K12MigrationError(f"{label} has no language entries")
    return manifest


def _verify_bootstrap_entry(
    root: Path, manifest: Mapping[str, Any], language: str, *, label: str
) -> tuple[dict[str, Any], Path]:
    raw = manifest.get("languages", {}).get(language)
    if not isinstance(raw, dict):
        raise K12MigrationError(f"{label} lacks {language}")
    acquisition = raw.get("acquisition")
    if not isinstance(acquisition, dict):
        raise K12MigrationError(f"{label}/{language} lacks acquisition metadata")
    report_path = _safe_artifact(
        root, acquisition.get("report_path"), label=f"{label}/{language} report"
    )
    report = _load_object(report_path, label=f"{label}/{language} report")
    report_hash = _verify_payload(
        report, "report_payload_sha256", label=f"{label}/{language} report"
    )
    if acquisition.get("report_payload_sha256") != report_hash:
        raise K12MigrationError(f"{label}/{language} report identity mismatch")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise K12MigrationError(f"{label}/{language} report lacks outputs")
    for name, output in outputs.items():
        if not isinstance(output, dict):
            raise K12MigrationError(f"{label}/{language}/{name} output is invalid")
        artifact = _safe_artifact(
            report_path.parent,
            output.get("path", name),
            label=f"{label}/{language}/{name}",
        )
        if sha256_file(artifact) != output.get("sha256"):
            raise K12MigrationError(f"{label}/{language}/{name} changed")
    for split, name in (
        ("train", "train.jsonl"),
        ("validation_reserve", "validation_reserve.jsonl"),
    ):
        entry = raw.get(split)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != outputs.get(name, {}).get("sha256")
        ):
            raise K12MigrationError(f"{label}/{language}/{split} metadata mismatch")
    return copy.deepcopy(raw), report_path.parent


def _load_release_manifests_only(
    root: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = _load_object(root / "corpus_manifest.json", label=f"{label} manifest")
    manifest_hash = _verify_payload(
        manifest, "manifest_sha256", label=f"{label} manifest"
    )
    hashes = _load_object(root / "hash_manifest.json", label=f"{label} hash manifest")
    _verify_payload(
        hashes, "hash_manifest_payload_sha256", label=f"{label} hash manifest"
    )
    if (
        manifest.get("ready") is not True
        or manifest.get("release_status") != "ready"
        or hashes.get("corpus_manifest_payload_sha256") != manifest_hash
        or not isinstance(hashes.get("files"), dict)
    ):
        raise K12MigrationError(f"{label} manifest pair is not a ready release")
    return manifest, hashes


def _install_or_verify(
    source: Path, target: Path, *, force_copy: bool = False
) -> str:
    if not target.exists() and not target.is_symlink():
        return _install_file(source, target, force_copy=force_copy)
    if target.is_symlink() or not target.is_file() or sha256_file(target) != sha256_file(source):
        raise K12MigrationError(f"partial worker target differs: {target}")
    same_inode = (source.stat().st_dev, source.stat().st_ino) == (
        target.stat().st_dev,
        target.stat().st_ino,
    )
    if (force_copy or source.stat().st_mode & 0o222) and same_inode:
        raise K12MigrationError(f"partial worker target shares forbidden inode: {target}")
    if target.stat().st_mode & 0o222:
        os.chmod(target, target.stat().st_mode & ~0o222)
    if same_inode:
        return "hardlink_read_only_source"
    if force_copy:
        return "copy_predecessor_then_make_read_only"
    if source.stat().st_mode & 0o222:
        return "copy_writable_source_then_make_read_only"
    return "verified_existing_copy"


def _publish_verified_worker_file(source: Path, target: Path) -> str:
    """Publish an already hash-verified, read-only worker artifact by inode."""

    if source.is_symlink() or not source.is_file():
        raise K12MigrationError(f"worker artifact is not a regular file: {source}")
    if source.stat().st_mode & 0o222:
        raise K12MigrationError(f"worker artifact is writable: {source}")
    if target.exists() or target.is_symlink():
        raise K12MigrationError(f"publication target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        raise K12MigrationError(
            f"cannot publish verified worker artifact {source}: {error}"
        ) from error
    source_stat = source.stat()
    target_stat = target.stat()
    if (source_stat.st_dev, source_stat.st_ino) != (
        target_stat.st_dev,
        target_stat.st_ino,
    ):
        raise K12MigrationError(f"worker publication did not share inode: {target}")
    return "hardlink_verified_read_only_worker"


def _load_worker_receipt(path: Path, *, command: str) -> dict[str, Any]:
    receipt = _load_object(path, label=f"{command} worker receipt")
    _verify_payload(receipt, "receipt_sha256", label=f"{command} worker receipt")
    if receipt.get("command") != command or receipt.get("status") != "complete":
        raise K12MigrationError(f"invalid {command} worker receipt: {path}")
    return receipt


def _subset_plan(plan: Mapping[str, Any], languages: Sequence[str]) -> dict[str, Any]:
    selected = set(languages)
    result = copy.deepcopy(dict(plan))
    result.pop("plan_sha256", None)
    result["physical_pool"] = {
        language: copy.deepcopy(plan["physical_pool"][language])
        for language in sorted(selected)
    }
    for tier_name, tier in result["tiers"].items():
        tier["languages"] = [language for language in tier["languages"] if language in selected]
        for field in ("availability_tokens", "probabilities", "target_exposure_tokens"):
            values = tier.get(field)
            if isinstance(values, dict):
                tier[field] = {
                    language: values[language]
                    for language in sorted(selected)
                    if language in values
                }
    provenance = result.setdefault("provenance", {})
    provenance["migration_delta"] = {
        "tool_id": TOOL_ID,
        "full_successor_plan_sha256": plan["plan_sha256"],
        "languages": sorted(selected),
    }
    result["plan_sha256"] = canonical_json_sha256(result)
    return result
