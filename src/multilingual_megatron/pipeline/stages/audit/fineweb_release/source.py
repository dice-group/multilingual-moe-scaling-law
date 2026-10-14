"""Language-contract and source-materialization audits for FineWeb-v2."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.ab_v2 import (
    EXPECTED_K,
    language_set_sha256,
    validate_ab_v2_spec,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.integrity import (
    _load_object,
    _require,
    _resolve_file,
    _resolve_source_artifact_for_audit,
    _verify_payload,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)


def _load_language_contract(
    *, repo_root: Path, study_spec_path: Path
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, str]]:
    validate_ab_v2_spec(repo_root=repo_root, spec_path=study_spec_path)
    spec = _load_object(study_spec_path, label="study specification")
    raw_sets = spec.get("language_sets")
    _require(isinstance(raw_sets, dict), "study specification lacks language_sets")
    language_sets: dict[str, list[str]] = {}
    file_hashes: dict[str, str] = {}
    for k in EXPECTED_K:
        tier = f"K{k}"
        entry = raw_sets.get(tier) if isinstance(raw_sets, dict) else None
        _require(isinstance(entry, dict), f"study specification lacks {tier}")
        path = _resolve_file(repo_root, entry.get("path"), label=f"{tier} language map")
        payload = _load_object(path, label=f"{tier} language map")
        group = payload.get("0")
        _require(
            isinstance(group, dict) and isinstance(group.get("languages"), list),
            f"invalid {tier} map",
        )
        languages = [str(value) for value in group["languages"]]
        _require(
            len(languages) == k and len(set(languages)) == k,
            f"invalid {tier} cardinality",
        )
        _require(
            language_set_sha256(languages) == entry.get("sha256"),
            f"{tier} membership hash mismatch",
        )
        language_sets[tier] = languages
        file_hashes[tier] = sha256_file(path)
    return spec, language_sets, file_hashes


def _is_adopted_equal_hash_source_lock(
    *,
    materialized: Path,
    expected: Path,
    materialized_sha256: str,
    adoption_enabled: bool,
) -> bool:
    """Accept a historical source-lock locator only for verified adoption."""

    return (
        materialized != expected
        and adoption_enabled
        and materialized.name == expected.name
        and materialized_sha256 == sha256_file(expected)
    )


def _audit_source_materialization(
    *,
    repo_root: Path,
    release_root: Path,
    spec: Mapping[str, Any],
    allowed_predecessor_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (
        release_root / "corpus" / "source_lock" / "source_materialization_manifest.json"
    )
    manifest = _load_object(path, label="source materialization manifest")
    payload_hash = _verify_payload(
        manifest, key="manifest_sha256", label="source materialization manifest"
    )
    source_lock = _resolve_file(
        repo_root, manifest.get("source_lock_path"), label="source lock"
    )
    source_lock_sha256 = sha256_file(source_lock)
    _require(
        source_lock_sha256 == manifest.get("source_lock_sha256"),
        "source-lock file hash mismatch",
    )
    spec_ref = spec.get("data_policy", {}).get("source_lock", {})
    expected_source_lock = _resolve_file(
        repo_root, spec_ref.get("path"), label="spec source lock"
    )
    adopted_equal_hash_path = _is_adopted_equal_hash_source_lock(
        materialized=source_lock,
        expected=expected_source_lock,
        materialized_sha256=source_lock_sha256,
        adoption_enabled=allowed_predecessor_root is not None,
    )
    _require(
        source_lock == expected_source_lock or adopted_equal_hash_path,
        "materialized source lock differs from the study spec",
    )
    _require(
        manifest.get("source_lock_sha256") == spec_ref.get("sha256"),
        "source-lock hash differs from the study spec",
    )
    source_payload = _load_object(source_lock, label="source lock")
    _require(
        manifest.get("source_lock_id") == "fineweb2_v2_1_1", "unexpected source-lock ID"
    )
    revisions = manifest.get("dataset_revisions")
    _require(isinstance(revisions, dict), "source manifest lacks dataset revisions")
    _require(
        revisions.get("fineweb2")
        == source_payload.get("non_english_corpus", {}).get("dataset_revision")
        and revisions.get("fineweb")
        == source_payload.get("english_corpus", {}).get("dataset_revision"),
        "materialized dataset revisions differ from the source lock",
    )
    artifacts = manifest.get("artifacts")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts) == {"inventory", "rehydration_weights"},
        "source artifact set is incomplete",
    )
    resolved_artifact_paths: dict[str, str] = {}
    predecessor_provenance_artifacts: list[str] = []
    verified_provenance_paths: list[str] = []
    if adopted_equal_hash_path:
        verified_provenance_paths.append(str(source_lock))
    for name, raw in artifacts.items():
        _require(isinstance(raw, dict), f"source artifact {name} is invalid")
        artifact_path, predecessor_provenance = _resolve_source_artifact_for_audit(
            source_lock_dir=path.parent,
            raw=raw,
            release_root=release_root,
            label=f"source artifact {name}",
            allowed_predecessor_root=allowed_predecessor_root,
        )
        resolved_artifact_paths[name] = str(artifact_path)
        if predecessor_provenance:
            predecessor_provenance_artifacts.append(name)
            original = Path(str(raw["path"]))
            if not original.is_absolute():
                original = path.parent / original
            verified_provenance_paths.append(str(original.resolve()))
    return manifest, {
        "manifest_file_sha256": sha256_file(path),
        "manifest_payload_sha256": payload_hash,
        "source_lock_sha256": manifest["source_lock_sha256"],
        "source_lock_path": str(source_lock),
        "expected_source_lock_path": str(expected_source_lock),
        "source_lock_path_exception": (
            "adopted_equal_hash_provenance" if adopted_equal_hash_path else None
        ),
        "artifact_count": len(artifacts),
        "resolved_artifact_paths": resolved_artifact_paths,
        "predecessor_provenance_artifacts": predecessor_provenance_artifacts,
        "verified_provenance_paths": sorted(verified_provenance_paths),
    }

def _audit_materialized_inventory_selection(
    *,
    inventory_path: Path,
    k50_languages: Sequence[str],
    k200_languages: Sequence[str],
) -> dict[str, Any]:
    """Reproduce the K200 top-150 selection from the acquired inventory.

    The committed language audit was derived from a frozen local snapshot, while
    production reads the independently pinned FineWeb2 inventory materialized by
    the source-lock job.  Recomputing the ranking here proves that those two
    immutable inputs induce the same language/script panel.
    """

    _require(len(k50_languages) == 50, "materialized-inventory audit requires K50")
    _require(len(k200_languages) == 200, "materialized-inventory audit requires K200")
    _require(
        list(k200_languages[:50]) == list(k50_languages),
        "K200 does not begin with the exact ordered K50 panel",
    )
    atlas_iso = {value.split("_", 1)[0] for value in k50_languages}
    representative_by_iso: dict[str, dict[str, Any]] = {}
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subset = str(row.get("subset", ""))
            if row.get("split") != "train" or subset.endswith("_removed"):
                continue
            iso = str(row.get("code", ""))
            script = str(row.get("script", ""))
            _require(
                iso and script and subset == f"{iso}_{script}",
                f"invalid inventory row: {subset}",
            )
            try:
                words = int(float(str(row.get("words", ""))))
                documents = int(float(str(row.get("documents", ""))))
            except ValueError:
                # Match the frozen selection procedure: rows without numeric
                # train availability cannot participate in the ranking.
                continue
            if words <= 0 or documents <= 0:
                continue
            candidate = {
                "subset": subset,
                "code": iso,
                "script": script,
                "words": words,
                "documents": documents,
            }
            current = representative_by_iso.get(iso)
            if current is None or (words, documents, subset) > (
                int(current["words"]),
                int(current["documents"]),
                str(current["subset"]),
            ):
                representative_by_iso[iso] = candidate

    ranked = [row for iso, row in representative_by_iso.items() if iso not in atlas_iso]
    ranked.sort(
        key=lambda row: (
            -int(row["documents"]),
            -int(row["words"]),
            str(row["subset"]),
        )
    )
    selected = ranked[:150]
    _require(
        len(selected) == 150,
        "materialized inventory has fewer than 150 K50-external languages",
    )
    selected_subsets = [str(row["subset"]) for row in selected]
    _require(
        list(k200_languages[50:]) == selected_subsets,
        "materialized production inventory does not reproduce the frozen K200 top-150",
    )
    _require(
        len({str(row["code"]) for row in selected}) == 150,
        "materialized top-150 contains more than one script for an ISO language",
    )
    return {
        "inventory_file_sha256": sha256_file(inventory_path),
        "representative_iso_language_count": len(representative_by_iso),
        "excluded_k50_iso_language_count": len(atlas_iso),
        "added_language_count": len(selected),
        "ordered_added_languages_sha256": canonical_json_sha256(selected_subsets),
        "minimum_selected_train_documents": min(
            int(row["documents"]) for row in selected
        ),
        "cutoff_language": selected_subsets[-1],
        "cutoff_train_documents": int(selected[-1]["documents"]),
    }
