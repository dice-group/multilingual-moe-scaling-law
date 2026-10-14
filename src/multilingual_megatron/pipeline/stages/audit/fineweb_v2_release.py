#!/usr/bin/env python3
"""Fail-closed scientific release audit for the multilingual A/B-v2 data.

This is deliberately a cross-stage *audit*, not another data transformer.  It
recomputes the immutable identities emitted by source materialization,
tokenizer acquisition/training, token calibration, sampling/finalization, and
Megatron preprocessing.  Training may start only from a passing report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.ab_v2 import (
    EXPECTED_K,
    language_set_sha256,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    LanguageCalibration,
    POLICY_ID,
    alpha_probabilities,
    assign_validation_reserve,
    canonical_json_sha256,
    largest_remainder_quotas,
    sha256_file,
    verify_plan_hash,
)
from multilingual_megatron.pipeline.tokenizer_sentence_contract import (
    ACQUISITION_PLAN_TOOL_ID,
    SENTENCE_SLOT_COUNTER_ID,
    SPLITTER_ID,
    sentence_slot_gate_contract,
    split_sentence_like,
    tokenizer_acquisition_target_contract,
    verify_acquisition_plan,
)
from multilingual_megatron.pipeline.stages.tokenizer_training.build_sentence_artifact import (
    AVAILABILITY_ESTIMATOR_ID,
    INTERLEAVER_ID,
    WITHIN_LANGUAGE_DRAW_ID,
)
from multilingual_megatron.pipeline.stages.tokenizer_training.sentencepiece_v2 import (
    audit_production_v2_model,
)


from multilingual_megatron.pipeline.stages.audit.fineweb_release.constants import (
    EXPECTED_ALPHA,
    EXPECTED_DATA_SEED,
    EXPECTED_MEGATRON_DTYPE_CODE,
    EXPECTED_TOKENIZER_SENTENCES,
    EXPECTED_TRAIN_TOKENS,
    EXPECTED_VALIDATION_CANDIDATE_FRACTION,
    EXPECTED_VALIDATION_CAP,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VOCAB_SIZE,
    SCHEMA,
    _DTYPE_SIZES,
    _INDEX_HEADER,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.integrity import (
    ReleaseAuditError,
    _V1_PATH_RE,
    _assert_fresh_path,
    _audit_embedded_paths,
    _cached_sha256,
    _is_predecessor_provenance_path,
    _load_k12_adoption_lineage,
    _load_object,
    _require,
    _require_sha256,
    _resolve_file,
    _resolve_source_artifact_for_audit,
    _sealed_stat,
    _under,
    _verify_payload,
    _verify_relative_artifact,
    _verify_predecessor_seal_stats,
    _walk_path_values,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.source import (
    _audit_materialized_inventory_selection,
    _audit_source_materialization,
    _is_adopted_equal_hash_source_lock,
    _load_language_contract,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.retention import (
    _retention_policy,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.tokenization import (
    _audit_index,
    _audit_tokenization,
    _manifest_payload,
    _verify_prefix,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.tokenizer import (
    _audit_acquisition_plan,
    _audit_bootstrap,
    _audit_bootstrap_split_rows,
    _audit_calibration,
    _audit_sentence_artifact,
    _audit_tokenizer,
    _recount_tokenizer_bootstrap_slots,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.corpus import (
    _audit_corpus_release,
    _audit_finalized_split_artifact,
    _audit_sampling_plan,
    _create_original_id_index,
    _register_original_id,
)




def audit_fineweb_v2_release(
    *, repo_root: Path, study_spec_path: Path, release_root: Path
) -> dict[str, Any]:
    """Audit every immutable stage and return a hash-stamped pass/fail report."""

    repo_root = repo_root.resolve()
    study_spec_path = study_spec_path.resolve()
    release_root = release_root.resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    adoption: Mapping[str, Any] | None = None
    payloads_for_path_audit: dict[str, Mapping[str, Any]] = {}
    hash_cache: dict[Path, str] = {}
    try:
        _require(
            _V1_PATH_RE.search(release_root.as_posix()) is None,
            "release root itself is a v1 path",
        )
        spec, language_sets, language_map_hashes = _load_language_contract(
            repo_root=repo_root, study_spec_path=study_spec_path
        )
        adoption, allowed_provenance_roots = _load_k12_adoption_lineage(
            release_root
        )
        allowed_predecessor_root = (
            Path(str(adoption["predecessor_root"])).resolve()
            if adoption is not None
            else None
        )
        checks["k12_adoption"] = (
            {
                "enabled": True,
                "manifest_file_sha256": sha256_file(
                    release_root
                    / "corpus"
                    / "k12_migration"
                    / "adoption_manifest.json"
                ),
                "manifest_payload_sha256": adoption["adoption_manifest_sha256"],
                "predecessor_root": str(allowed_predecessor_root),
                "predecessor_seal_payload_sha256": adoption["predecessor_seal"][
                    "seal_payload_sha256"
                ],
                "embedded_path_exception": "provenance_only",
                "verified_provenance_roots": [
                    str(path) for path in allowed_provenance_roots
                ],
            }
            if adoption is not None
            else {"enabled": False}
        )
        _require(
            spec.get("data_policy", {}).get("fresh_v2_resample_required") is True,
            "study spec does not require a fresh v2 resample",
        )
        _require(
            spec.get("data_policy", {}).get("reuse_v1_sampled_or_tokenized_artifacts")
            is False,
            "study spec permits v1 data reuse",
        )
        checks["study_spec"] = {
            "file_sha256": sha256_file(study_spec_path),
            "language_set_sha256": {
                tier: language_set_sha256(values)
                for tier, values in language_sets.items()
            },
            "nested": all(
                set(language_sets[f"K{lower}"]).issubset(language_sets[f"K{upper}"])
                for lower, upper in zip(EXPECTED_K, EXPECTED_K[1:])
            ),
            "fresh_v2_resample_required": True,
        }

        source_manifest_path = (
            release_root
            / "corpus"
            / "source_lock"
            / "source_materialization_manifest.json"
        )
        source_manifest, checks["source_materialization"] = (
            _audit_source_materialization(
                repo_root=repo_root,
                release_root=release_root,
                spec=spec,
                allowed_predecessor_root=allowed_predecessor_root,
            )
        )
        payloads_for_path_audit["source_materialization"] = source_manifest
        k200 = language_sets["K200"]
        materialized_inventory = Path(
            checks["source_materialization"]["resolved_artifact_paths"]["inventory"]
        )
        checks["materialized_inventory_language_selection"] = (
            _audit_materialized_inventory_selection(
                inventory_path=materialized_inventory,
                k50_languages=language_sets["K50"],
                k200_languages=k200,
            )
        )
        source_lock_hash = str(source_manifest["source_lock_sha256"])
        weights_hash = str(
            source_manifest["artifacts"]["rehydration_weights"]["sha256"]
        )

        pilot_root = release_root / "corpus" / "bootstrap"
        pilot_manifest, pilot_summary, checks["pilot_bootstrap"] = _audit_bootstrap(
            root=pilot_root,
            label="pilot",
            languages=k200,
            source_lock_sha256=source_lock_hash,
            language_map_file_sha256=language_map_hashes["K200"],
            rehydration_weights_sha256=weights_hash,
        )
        payloads_for_path_audit["pilot_bootstrap_manifest"] = pilot_manifest

        acquisition_path = (
            release_root / "corpus" / "tokenizer_sentence_acquisition_plan.json"
        )
        acquisition_plan, checks["tokenizer_acquisition_plan"] = (
            _audit_acquisition_plan(
                path=acquisition_path,
                languages=k200,
                language_map_hash=language_map_hashes["K200"],
                pilot_summary_path=pilot_root / "bootstrap_summary.json",
            )
        )
        payloads_for_path_audit["tokenizer_acquisition_plan"] = acquisition_plan

        tokenizer_bootstrap_root = release_root / "corpus" / "tokenizer_bootstrap"
        (
            tokenizer_bootstrap_manifest,
            tokenizer_bootstrap_summary,
            checks["tokenizer_bootstrap"],
        ) = _audit_bootstrap(
            root=tokenizer_bootstrap_root,
            label="tokenizer acquisition",
            languages=k200,
            source_lock_sha256=source_lock_hash,
            language_map_file_sha256=language_map_hashes["K200"],
            rehydration_weights_sha256=weights_hash,
            expected_acquisition_plan=(
                sha256_file(acquisition_path),
                str(acquisition_plan["plan_sha256"]),
            ),
            tokenizer_acquisition_plan=acquisition_plan,
            require_incremental=True,
        )
        payloads_for_path_audit["tokenizer_bootstrap_manifest"] = (
            tokenizer_bootstrap_manifest
        )

        sentence_root = release_root / "corpus" / "tokenizer_sentences"
        sentence_path = sentence_root / "tokenizer_sentences_100m.jsonl"
        sentence_manifest_path = (
            sentence_root / "tokenizer_sentences_100m.manifest.json"
        )
        sentence_manifest, checks["tokenizer_sentence_artifact"] = (
            _audit_sentence_artifact(
                manifest_path=sentence_manifest_path,
                sentence_path=sentence_path,
                acquisition_path=acquisition_path,
                acquisition_plan=acquisition_plan,
                tokenizer_bootstrap_summary_path=tokenizer_bootstrap_root
                / "bootstrap_summary.json",
                languages=k200,
            )
        )
        payloads_for_path_audit["tokenizer_sentence_manifest"] = sentence_manifest

        tokenizer_root = release_root / "corpus" / "tokenizer" / "sp_unigram_64k"
        tokenizer_result, checks["tokenizer"] = _audit_tokenizer(
            tokenizer_root=tokenizer_root,
            sentence_path=sentence_path,
            sentence_manifest=sentence_manifest,
            expected_model_prefix=(
                allowed_provenance_roots[-1]
                / "corpus/tokenizer/sp_unigram_64k/tokenizer"
                if allowed_provenance_roots
                else None
            ),
        )
        payloads_for_path_audit["tokenizer_training_result"] = tokenizer_result
        tokenizer_path = tokenizer_root / "tokenizer.model"
        tokenizer_hash = sha256_file(tokenizer_path)

        calibration_path = release_root / "corpus" / "token_calibration.json"
        calibration, checks["token_calibration"] = _audit_calibration(
            path=calibration_path,
            languages=k200,
            tokenizer_hash=tokenizer_hash,
            source_manifest=source_manifest,
            pilot_summary_path=pilot_root / "bootstrap_summary.json",
            pilot_manifest_path=pilot_root / "bootstrap_manifest.json",
        )
        payloads_for_path_audit["token_calibration"] = calibration

        sampling_plan_path = release_root / "corpus" / "sampling_plan_32b.json"
        sampling_plan, checks["sampling_plan"] = _audit_sampling_plan(
            path=sampling_plan_path,
            language_sets=language_sets,
            language_map_hashes=language_map_hashes,
            calibration_path=calibration_path,
            calibration=calibration,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            tokenizer_path=tokenizer_path,
        )
        payloads_for_path_audit["sampling_plan"] = sampling_plan

        production_root = release_root / "corpus" / "production_bootstrap"
        production_manifest, production_summary, checks["production_bootstrap"] = (
            _audit_bootstrap(
                root=production_root,
                label="production",
                languages=k200,
                source_lock_sha256=source_lock_hash,
                language_map_file_sha256=language_map_hashes["K200"],
                rehydration_weights_sha256=weights_hash,
                expected_sampling_plan=(
                    sha256_file(sampling_plan_path),
                    str(sampling_plan["plan_sha256"]),
                ),
                require_incremental=True,
            )
        )
        payloads_for_path_audit["production_bootstrap_manifest"] = production_manifest

        corpus_manifest, checks["corpus_release"] = _audit_corpus_release(
            release_dir=release_root / "corpus" / "release_32b",
            languages=k200,
            plan_path=sampling_plan_path,
            plan=sampling_plan,
            production_manifest_path=production_root / "bootstrap_manifest.json",
            production_manifest=production_manifest,
            tokenizer_path=tokenizer_path,
            hash_cache=hash_cache,
        )
        payloads_for_path_audit["corpus_manifest"] = corpus_manifest

        tokenization_report, checks["tokenization"] = _audit_tokenization(
            tokenized_root=release_root / "corpus" / "tokenized",
            release_root=release_root,
            language_sets=language_sets,
            plan=sampling_plan,
            corpus=corpus_manifest,
            tokenizer_hash=tokenizer_hash,
            hash_cache=hash_cache,
        )
        payloads_for_path_audit["tokenization_report"] = tokenization_report
        checks["fresh_v2_paths"] = {
            "absolute_embedded_paths_checked": _audit_embedded_paths(
                payloads_for_path_audit,
                release_root=release_root,
                allowed_predecessor_root=allowed_predecessor_root,
                allowed_provenance_roots=allowed_provenance_roots,
                allowed_provenance_paths={
                    Path(value)
                    for value in checks["source_materialization"].get(
                        "verified_provenance_paths", []
                    )
                },
            ),
            "v1_paths_detected": 0,
            "other_pipeline_release_paths_detected": 0,
            "allowed_predecessor_paths": (
                str(allowed_predecessor_root)
                if allowed_predecessor_root is not None
                else None
            ),
            "allowed_provenance_roots": [
                str(path) for path in allowed_provenance_roots
            ],
        }
    except (ReleaseAuditError, ValueError, KeyError, TypeError, OSError) as exc:
        errors.append(str(exc))
    except (
        Exception
    ) as exc:  # pragma: no cover - fail closed on dependency/runtime drift.
        errors.append(f"unexpected release-audit failure: {type(exc).__name__}: {exc}")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "pass": not errors,
        "errors": errors,
        "release_root": str(release_root),
        "study_spec": {
            "path": str(study_spec_path),
            "sha256": sha256_file(study_spec_path)
            if study_spec_path.is_file()
            else None,
        },
        "checks": checks,
        "scientific_conclusions": {
            "fresh_v2_resample": not errors
            and checks.get("fresh_v2_paths", {}).get("v1_paths_detected") == 0,
            "exact_32b_exposure_each_K": not errors,
            "zero_frozen_physical_target_shortfall": not errors,
            "split_disjoint_and_validation_exact": not errors,
            "shared_64k_unigram_tokenizer": not errors,
            "alpha_0_3_nested_K_mixtures": not errors,
            "megatron_uint16_counts_and_hashes_verified": not errors,
            "lineage_preserving_k12_adoption": not errors
            and checks.get("k12_adoption", {}).get("enabled") is True,
        },
        "retention": _retention_policy(release_root, adoption=adoption),
    }
    report["report_payload_sha256"] = canonical_json_sha256(report)
    return report


def write_release_audit(
    *, repo_root: Path, study_spec_path: Path, release_root: Path, output_path: Path
) -> dict[str, Any]:
    """Run the release audit and atomically write its report, including failures."""

    report = audit_fineweb_v2_release(
        repo_root=repo_root,
        study_spec_path=study_spec_path,
        release_root=release_root,
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--study-spec", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = write_release_audit(
        repo_root=Path(args.repo_root),
        study_spec_path=Path(args.study_spec),
        release_root=Path(args.release_root),
        output_path=Path(args.output),
    )
    if report["pass"]:
        print(f"[OK] A/B-v2 scientific data release audit: {args.output}")
        return 0
    print("[ERROR] A/B-v2 data release audit failed: " + "; ".join(report["errors"]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReleaseAuditError",
    "audit_fineweb_v2_release",
    "main",
    "write_release_audit",
]
