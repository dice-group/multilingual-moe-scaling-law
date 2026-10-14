"""Sampling-plan and finalized corpus audits for FineWeb-v2 releases."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.ab_v2 import language_set_sha256
from multilingual_megatron.pipeline.stages.audit.fineweb_release.constants import (
    EXPECTED_ALPHA,
    EXPECTED_DATA_SEED,
    EXPECTED_TRAIN_TOKENS,
    EXPECTED_VALIDATION_CANDIDATE_FRACTION,
    EXPECTED_VALIDATION_CAP,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VOCAB_SIZE,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.integrity import (
    ReleaseAuditError,
    _cached_sha256,
    _load_object,
    _require,
    _require_sha256,
    _resolve_file,
    _verify_payload,
    _verify_relative_artifact,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    LanguageCalibration,
    POLICY_ID,
    alpha_probabilities,
    assign_validation_reserve,
    largest_remainder_quotas,
    sha256_file,
    verify_plan_hash,
)


def _audit_sampling_plan(
    *,
    path: Path,
    language_sets: Mapping[str, Sequence[str]],
    language_map_hashes: Mapping[str, str],
    calibration_path: Path,
    calibration: Mapping[str, Any],
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    tokenizer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load_object(path, label="sampling plan")
    _require(verify_plan_hash(plan), "sampling-plan payload hash mismatch")
    _require(plan.get("policy_id") == POLICY_ID, "sampling-plan policy ID mismatch")
    _require(
        plan.get("sampling_seed") == EXPECTED_DATA_SEED,
        "sampling-plan document seed mismatch",
    )
    _require(plan.get("alpha") == EXPECTED_ALPHA, "sampling-plan alpha mismatch")
    _require(
        plan.get("train_tokens_per_set") == EXPECTED_TRAIN_TOKENS,
        "sampling-plan target is not 32B per K",
    )
    _require(
        plan.get("validation_cap_tokens") == EXPECTED_VALIDATION_CAP,
        "sampling-plan validation cap mismatch",
    )
    _require(
        plan.get("availability_contract")
        == {
            "validation_rule": "min(validation_cap_tokens, floor(validation_fraction * estimated_unique_tokens))",
            "train_unique_rule": "estimated_unique_tokens - validation_target_tokens",
            "candidate_reserve_affects_availability": False,
        },
        "sampling-plan cap-aware availability contract mismatch",
    )
    tiers = plan.get("tiers")
    _require(
        isinstance(tiers, dict) and set(tiers) == set(language_sets),
        "sampling-plan K tiers mismatch",
    )
    physical = plan.get("physical_pool")
    _require(
        isinstance(physical, dict) and set(physical) == set(language_sets["K200"]),
        "sampling-plan physical-pool coverage mismatch",
    )
    maximum_quota = {language: 0 for language in language_sets["K200"]}
    tier_totals: dict[str, int] = {}
    for tier, expected_languages in language_sets.items():
        raw = tiers[tier]
        _require(isinstance(raw, dict), f"sampling-plan {tier} is invalid")
        _require(
            raw.get("languages") == list(expected_languages),
            f"sampling-plan {tier} membership/order mismatch",
        )
        _require(
            raw.get("language_set_sha256") == language_set_sha256(expected_languages),
            f"sampling-plan {tier} membership hash mismatch",
        )
        _require(
            raw.get("alpha") == EXPECTED_ALPHA
            and raw.get("availability_basis")
            == "estimated_train_tokens_after_fineweb2_rehydration",
            f"sampling-plan {tier} availability contract mismatch",
        )
        probabilities = raw.get("probabilities")
        availability = raw.get("availability_tokens")
        exposures = raw.get("target_exposure_tokens")
        _require(
            isinstance(availability, dict)
            and set(availability) == set(expected_languages),
            f"sampling-plan {tier} availability coverage mismatch",
        )
        _require(
            isinstance(probabilities, dict)
            and set(probabilities) == set(expected_languages),
            f"sampling-plan {tier} probability coverage mismatch",
        )
        _require(
            isinstance(exposures, dict) and set(exposures) == set(expected_languages),
            f"sampling-plan {tier} exposure coverage mismatch",
        )
        _require(
            math.isclose(
                math.fsum(float(value) for value in probabilities.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"sampling-plan {tier} probabilities do not sum to one",
        )
        recomputed_probabilities = alpha_probabilities(
            {language: int(availability[language]) for language in expected_languages},
            alpha=EXPECTED_ALPHA,
        )
        _require(
            all(
                math.isclose(
                    float(probabilities[language]),
                    recomputed_probabilities[language],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for language in expected_languages
            ),
            f"sampling-plan {tier} probabilities are not n_l^0.3",
        )
        _require(
            all(
                int(availability[language])
                == int(physical[language]["estimated_train_rehydrated_tokens"])
                for language in expected_languages
            ),
            f"sampling-plan {tier} availability differs from the shared physical calibration",
        )
        target = sum(int(value) for value in exposures.values())
        _require(
            raw.get("target_train_tokens") == EXPECTED_TRAIN_TOKENS
            and target == EXPECTED_TRAIN_TOKENS,
            f"sampling-plan {tier} exposure does not sum exactly to 32B",
        )
        _require(
            exposures
            == largest_remainder_quotas(
                EXPECTED_TRAIN_TOKENS,
                {key: float(value) for key, value in probabilities.items()},
            ),
            f"sampling-plan {tier} exposure is not largest-remainder exact",
        )
        tier_totals[tier] = target
        for language, quota in exposures.items():
            maximum_quota[language] = max(maximum_quota[language], int(quota))
    composition_arms = plan.get("composition_arms", {})
    _require(
        isinstance(composition_arms, dict),
        "sampling-plan composition_arms must be an object when present",
    )
    for arm_name, raw in sorted(composition_arms.items()):
        _require(isinstance(raw, dict), f"sampling-plan composition arm is invalid: {arm_name}")
        languages = raw.get("languages")
        _require(
            isinstance(languages, list)
            and len(languages) == 12
            and len(languages) == len(set(languages))
            and set(languages).issubset(language_sets["K200"]),
            f"sampling-plan composition arm membership is invalid: {arm_name}",
        )
        _require(
            raw.get("language_set_sha256") == language_set_sha256(languages),
            f"sampling-plan composition arm membership hash mismatch: {arm_name}",
        )
        _require(
            raw.get("alpha") == EXPECTED_ALPHA
            and raw.get("availability_basis")
            == "estimated_train_tokens_after_fineweb2_rehydration",
            f"sampling-plan composition arm availability contract mismatch: {arm_name}",
        )
        availability = raw.get("availability_tokens")
        probabilities = raw.get("probabilities")
        exposures = raw.get("target_exposure_tokens")
        language_keys = set(languages)
        _require(
            isinstance(availability, dict)
            and isinstance(probabilities, dict)
            and isinstance(exposures, dict)
            and set(availability) == language_keys
            and set(probabilities) == language_keys
            and set(exposures) == language_keys,
            f"sampling-plan composition arm coverage mismatch: {arm_name}",
        )
        _require(
            all(
                int(availability[language])
                == int(physical[language]["estimated_train_rehydrated_tokens"])
                for language in languages
            ),
            f"sampling-plan composition arm availability differs: {arm_name}",
        )
        recomputed = alpha_probabilities(
            {language: int(availability[language]) for language in languages},
            alpha=EXPECTED_ALPHA,
        )
        _require(
            math.isclose(
                math.fsum(float(value) for value in probabilities.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and all(
                math.isclose(
                    float(probabilities[language]),
                    recomputed[language],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for language in languages
            ),
            f"sampling-plan composition arm probabilities differ: {arm_name}",
        )
        target = sum(int(value) for value in exposures.values())
        _require(
            raw.get("target_train_tokens") == EXPECTED_TRAIN_TOKENS
            and target == EXPECTED_TRAIN_TOKENS
            and exposures
            == largest_remainder_quotas(EXPECTED_TRAIN_TOKENS, recomputed),
            f"sampling-plan composition arm exposure differs: {arm_name}",
        )
        tier_totals[f"composition:{arm_name}"] = target
        for language, quota in exposures.items():
            maximum_quota[language] = max(maximum_quota[language], int(quota))
    for language, raw in physical.items():
        _require(
            isinstance(raw, dict), f"sampling-plan physical pool is invalid: {language}"
        )
        _require(
            raw.get("requested_max_tier_tokens") == maximum_quota[language],
            f"physical-pool max quota mismatch: {language}",
        )
        raw_calibration = raw.get("calibration")
        _require(
            isinstance(raw_calibration, dict)
            and raw_calibration.get("language") == language,
            f"physical-pool calibration is missing: {language}",
        )
        try:
            language_calibration = LanguageCalibration(**raw_calibration)
            language_calibration.validate()
        except (TypeError, ValueError) as exc:
            raise ReleaseAuditError(
                f"physical-pool calibration is invalid: {language}"
            ) from exc
        estimated_unique = language_calibration.estimated_unique_tokens
        estimated_validation = (
            language_calibration.estimated_validation_unique_tokens_for_cap(
                EXPECTED_VALIDATION_CAP
            )
        )
        estimated_train_unique = (
            language_calibration.estimated_train_unique_tokens_for_cap(
                EXPECTED_VALIDATION_CAP
            )
        )
        estimated_train_rehydrated = (
            language_calibration.estimated_train_rehydrated_tokens_for_cap(
                EXPECTED_VALIDATION_CAP
            )
        )
        estimated_train_documents = (
            language_calibration.estimated_train_documents_for_cap(
                EXPECTED_VALIDATION_CAP
            )
        )
        _require(
            raw.get("estimated_unique_tokens") == estimated_unique
            and raw.get("estimated_validation_unique_tokens")
            == estimated_validation
            and raw.get("validation_target_tokens") == estimated_validation
            and raw.get("estimated_train_unique_tokens") == estimated_train_unique
            and raw.get("estimated_train_rehydrated_tokens")
            == estimated_train_rehydrated
            and raw.get("estimated_train_documents") == estimated_train_documents
            and math.isclose(
                float(raw.get("estimated_train_fraction", -1.0)),
                estimated_train_unique / float(estimated_unique),
                rel_tol=1e-15,
                abs_tol=0.0,
            ),
            f"physical-pool cap-aware availability estimate mismatch: {language}",
        )
        available = int(raw.get("estimated_train_rehydrated_tokens", 0))
        target = int(raw.get("physical_target_tokens", 0))
        _require(
            target == min(maximum_quota[language], available) and target > 0,
            f"physical-pool target mismatch: {language}",
        )
        _require(
            raw.get("training_repeat_required") == (maximum_quota[language] > target),
            f"physical-pool repeat flag mismatch: {language}",
        )

    provenance = plan.get("provenance")
    _require(isinstance(provenance, dict), "sampling plan lacks provenance")
    expected = {
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_payload_sha256": calibration.get("artifact_sha256"),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_manifest_payload_sha256": source_manifest.get("manifest_sha256"),
        "tokenizer_model_sha256": sha256_file(tokenizer_path),
        "language_map_file_sha256": dict(language_map_hashes),
    }
    for field, value in expected.items():
        _require(
            provenance.get(field) == value,
            f"sampling-plan provenance mismatch: {field}",
        )
    return plan, {
        "file_sha256": sha256_file(path),
        "payload_sha256": plan["plan_sha256"],
        "exact_exposure_tokens_by_tier": tier_totals,
        "physical_language_count": len(physical),
        "repeat_required_language_count": sum(
            bool(raw["training_repeat_required"]) for raw in physical.values()
        ),
    }

def _create_original_id_index(path: Path) -> sqlite3.Connection:
    """Create a bounded-memory, disk-backed split-disjointness index."""

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        PRAGMA cache_size = -65536;
        CREATE TABLE original_ids (
            original_document_id TEXT PRIMARY KEY,
            split TEXT NOT NULL CHECK(split IN ('train', 'validation'))
        ) WITHOUT ROWID;
        """
    )
    return connection


def _register_original_id(
    connection: sqlite3.Connection,
    *,
    original_document_id: str,
    split: str,
    label: str,
) -> None:
    """Fail on either a duplicate within one split or train/validation overlap."""

    try:
        connection.execute(
            "INSERT INTO original_ids(original_document_id, split) VALUES (?, ?)",
            (original_document_id, split),
        )
    except sqlite3.IntegrityError as exc:
        previous = connection.execute(
            "SELECT split FROM original_ids WHERE original_document_id = ?",
            (original_document_id,),
        ).fetchone()
        previous_split = str(previous[0]) if previous is not None else "unknown"
        if previous_split != split:
            raise ReleaseAuditError(
                "finalized train/validation original_document_id overlap: "
                f"{original_document_id} ({previous_split} and {split})"
            ) from exc
        raise ReleaseAuditError(
            f"{label} repeats original_document_id in non-atomic groups: "
            f"{original_document_id}"
        ) from exc


def _audit_finalized_split_artifact(
    *,
    base: Path,
    raw: Any,
    label: str,
    hash_cache: dict[Path, str],
    language: str,
    split: str,
    tokenizer_sha256: str,
    original_id_index: sqlite3.Connection,
) -> tuple[Path, dict[str, int]]:
    """Hash and independently recount a finalized split in one streaming pass."""

    _require(split in {"train", "validation"}, f"unsupported finalized split: {split}")
    _require(isinstance(raw, dict), f"{label} must be an artifact object")
    path = _resolve_file(base, raw.get("path"), label=label)
    expected_hash = _require_sha256(raw.get("sha256"), field=f"{label}.sha256")
    expected_size = raw.get("size_bytes", raw.get("bytes"))
    if expected_size is not None:
        _require(path.stat().st_size == expected_size, f"{label} size mismatch")

    digest = hashlib.sha256()
    emitted_documents = 0
    emitted_tokens = 0
    selected_original_documents = 0
    unique_original_tokens = 0
    reclaimed_original_documents = 0
    reclaimed_unique_tokens = 0
    reclaimed_emitted_documents = 0
    reclaimed_emitted_tokens = 0
    current_original: str | None = None
    current_multiplier = 0
    current_token_count = 0
    current_reclaimed = False
    current_replicas = 0

    def finish_group() -> None:
        if current_original is None:
            return
        _require(
            current_replicas == current_multiplier,
            f"{label} has incomplete/non-atomic replicas for {current_original}",
        )

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            where = f"{path}:{line_number}"
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReleaseAuditError(f"invalid finalized split JSON: {where}") from exc
            _require(isinstance(row, dict), f"finalized split row is not an object: {where}")
            lineage = row.get("lineage")
            tokenization = row.get("ab_v2_tokenization")
            original = row.get("original_document_id")
            _require(
                row.get("language") == language
                and isinstance(lineage, dict)
                and isinstance(tokenization, dict)
                and isinstance(original, str)
                and bool(original),
                f"finalized split row lineage/language mismatch: {where}",
            )
            multiplier = lineage.get("rehydration_multiplier")
            train_multiplier = lineage.get("train_rehydration_multiplier")
            replica_index = lineage.get("rehydration_replica_index")
            token_count = tokenization.get("token_count_including_eod")
            _require(
                isinstance(multiplier, int)
                and not isinstance(multiplier, bool)
                and multiplier >= 1
                and isinstance(train_multiplier, int)
                and not isinstance(train_multiplier, bool)
                and train_multiplier >= 1
                and isinstance(replica_index, int)
                and not isinstance(replica_index, bool)
                and replica_index >= 0
                and isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and token_count > 0,
                f"finalized split row multiplier/token metadata is invalid: {where}",
            )
            _require(
                tokenization.get("append_eod") is True
                and tokenization.get("tokenizer_model_sha256") == tokenizer_sha256,
                f"finalized split row tokenizer identity mismatch: {where}",
            )
            is_candidate = assign_validation_reserve(
                document_id=original,
                seed=EXPECTED_DATA_SEED,
                validation_fraction=EXPECTED_VALIDATION_CANDIDATE_FRACTION,
            )
            reclaimed = lineage.get("validation_candidate_reclaimed") is True
            if split == "validation":
                _require(
                    is_candidate
                    and not reclaimed
                    and "validation_candidate_reclaimed" not in lineage
                    and row.get("artifact_role") == "validation_reserve"
                    and lineage.get("split") == "validation_reserve"
                    and multiplier == 1
                    and replica_index == 0,
                    f"final validation row violates the candidate contract: {where}",
                )
            elif reclaimed:
                _require(
                    is_candidate
                    and row.get("artifact_role") == "train"
                    and lineage.get("split") == "train"
                    and lineage.get("validation_candidate_original_split")
                    == "validation_reserve"
                    and multiplier == train_multiplier,
                    f"reclaimed train row violates the candidate contract: {where}",
                )
            else:
                _require(
                    not is_candidate
                    and "validation_candidate_reclaimed" not in lineage
                    and "validation_candidate_original_split" not in lineage
                    and row.get("artifact_role") == "train"
                    and lineage.get("split") == "train"
                    and multiplier == train_multiplier,
                    f"ordinary train row violates candidate membership: {where}",
                )

            if original != current_original:
                finish_group()
                _register_original_id(
                    original_id_index,
                    original_document_id=original,
                    split=split,
                    label=label,
                )
                current_original = original
                current_multiplier = multiplier
                current_token_count = token_count
                current_reclaimed = reclaimed
                current_replicas = 0
                selected_original_documents += 1
                unique_original_tokens += token_count
                if reclaimed:
                    reclaimed_original_documents += 1
                    reclaimed_unique_tokens += token_count
            else:
                _require(
                    multiplier == current_multiplier
                    and token_count == current_token_count
                    and reclaimed == current_reclaimed,
                    f"finalized split replica metadata drifted: {where}",
                )
            _require(
                replica_index == current_replicas,
                f"finalized split replica ordering drifted: {where}",
            )
            expected_document_id = (
                original
                if replica_index == 0
                else f"{original}#rehydration-{replica_index:04d}"
            )
            _require(
                row.get("document_id") == expected_document_id,
                f"finalized split replica document ID drifted: {where}",
            )
            current_replicas += 1
            emitted_documents += 1
            emitted_tokens += token_count
            if reclaimed:
                reclaimed_emitted_documents += 1
                reclaimed_emitted_tokens += token_count
    finish_group()
    observed_hash = digest.hexdigest()
    _require(observed_hash == expected_hash, f"{label} hash mismatch")
    hash_cache[path] = observed_hash
    return path, {
        "emitted_documents": emitted_documents,
        "emitted_tokens": emitted_tokens,
        "selected_original_documents": selected_original_documents,
        "unique_original_tokens": unique_original_tokens,
        "reclaimed_original_documents": reclaimed_original_documents,
        "reclaimed_unique_tokens": reclaimed_unique_tokens,
        "reclaimed_emitted_documents": reclaimed_emitted_documents,
        "reclaimed_emitted_tokens": reclaimed_emitted_tokens,
    }


def _audit_corpus_release(
    *,
    release_dir: Path,
    languages: Sequence[str],
    plan_path: Path,
    plan: Mapping[str, Any],
    production_manifest_path: Path,
    production_manifest: Mapping[str, Any],
    tokenizer_path: Path,
    hash_cache: dict[Path, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = release_dir / "corpus_manifest.json"
    hash_manifest_path = release_dir / "hash_manifest.json"
    manifest = _load_object(manifest_path, label="corpus manifest")
    manifest_hash = _verify_payload(
        manifest, key="manifest_sha256", label="corpus manifest"
    )
    _require(
        manifest.get("release_id") == "fineweb_ab_v2_finalized"
        and manifest.get("ready") is True
        and manifest.get("release_status") == "ready",
        "corpus release is not ready",
    )
    _require(
        all(not values for values in manifest.get("blocking_languages", {}).values()),
        "corpus release still has blocking languages",
    )
    selection = manifest.get("selection_contract")
    _require(isinstance(selection, dict), "corpus release lacks selection contract")
    _require(
        selection.get("max_train_shortfall_fraction") == 0.0,
        "corpus release permits a physical-pool shortfall",
    )
    _require(
        selection.get("train_rehydration") == "all_declared_replicas_are_atomic"
        and selection.get("validation_rehydration") == "forbidden"
        and selection.get("validation_candidate_reserve")
        == {
            "declared_fraction": EXPECTED_VALIDATION_CANDIDATE_FRACTION,
            "final_target_fraction": EXPECTED_VALIDATION_FRACTION,
            "selection_order": "final_validation_before_train",
            "unused_candidate_policy": (
                "reclassify_to_train_with_preserved_train_multiplier"
            ),
            "preserved_train_multiplier_field": (
                "lineage.train_rehydration_multiplier"
            ),
        },
        "corpus validation candidate/reclamation contract mismatch",
    )
    _require(
        selection.get("sampling_seed") == EXPECTED_DATA_SEED
        and selection.get("priority")
        == "ascending_sha256(sampling_seed, split_namespace, original_document_id)"
        and selection.get("selection_namespaces")
        == {
            "train": "physical-train-document-selection-v1",
            "validation": "validation-panel-document-selection-v1",
            "validation_reserve_membership": "validation-reserve-v1",
        },
        "corpus document-selection seed/namespace contract mismatch",
    )
    disjoint = selection.get("train_validation_disjointness")
    _require(
        isinstance(disjoint, dict)
        and disjoint.get("original_document_id_overlap") == 0,
        "corpus release lacks exact split-disjointness proof",
    )
    inputs = manifest.get("inputs", {})
    _require(
        inputs.get("sampling_plan", {}).get("file_sha256") == sha256_file(plan_path),
        "corpus sampling-plan file lineage mismatch",
    )
    _require(
        inputs.get("sampling_plan", {}).get("payload_sha256")
        == plan.get("plan_sha256"),
        "corpus sampling-plan payload lineage mismatch",
    )
    _require(
        inputs.get("bootstrap_manifest", {}).get("sha256")
        == sha256_file(production_manifest_path),
        "corpus production-bootstrap lineage mismatch",
    )
    _require(
        inputs.get("bootstrap_manifest", {}).get("provenance")
        == production_manifest.get("provenance"),
        "corpus production-bootstrap provenance mismatch",
    )
    tokenizer_hash = sha256_file(tokenizer_path)
    _require(
        inputs.get("tokenizer_model", {}).get("sha256") == tokenizer_hash
        and inputs.get("tokenizer_model", {}).get("vocabulary_size")
        == EXPECTED_VOCAB_SIZE,
        "corpus tokenizer identity mismatch",
    )

    raw_languages = manifest.get("languages")
    _require(
        isinstance(raw_languages, dict) and set(raw_languages) == set(languages),
        "corpus language coverage mismatch",
    )
    total_physical = 0
    total_validation = 0
    total_original_ids = 0
    for language in languages:
        raw = raw_languages[language]
        _require(isinstance(raw, dict), f"invalid corpus entry: {language}")
        coverage = raw.get("source_coverage")
        _require(
            isinstance(coverage, dict), f"corpus source coverage missing: {language}"
        )
        bootstrap = raw.get("bootstrap")
        _require(
            isinstance(bootstrap, dict)
            and set(bootstrap) == {"train", "validation_reserve"},
            f"corpus bootstrap audit incomplete: {language}",
        )
        train = raw.get("train")
        validation = raw.get("validation")
        _require(
            isinstance(train, dict) and isinstance(validation, dict),
            f"corpus split entry missing: {language}",
        )
        target = int(plan["physical_pool"][language]["physical_target_tokens"])
        _require(
            train.get("status") == "ready" and train.get("target_tokens") == target,
            f"corpus train target/status mismatch: {language}",
        )
        _require(
            train.get("shortfall_tokens") == 0
            and train.get("shortfall_fraction") == 0.0,
            f"corpus train pool is underfilled: {language}",
        )
        selected = int(train.get("selected_tokens", -1))
        _require(
            selected >= target and train.get("overshoot_tokens") == selected - target,
            f"corpus whole-document train accounting mismatch: {language}",
        )
        _require(
            train.get("max_allowed_shortfall_fraction") == 0.0,
            f"corpus train shortfall tolerance drifted: {language}",
        )
        _verify_relative_artifact(
            base=release_dir,
            raw=train.get("unigram_counts"),
            label=f"corpus unigram {language}",
            hash_cache=hash_cache,
        )
        _require(
            validation.get("status") == "ready"
            and validation.get("quota_exact") is True
            and validation.get("never_rehydrated") is True,
            f"validation is not exact/unrehydrated: {language}",
        )
        _require(
            validation.get("shortfall_tokens") == 0
            and validation.get("target_rule")
            == "min(20,000,000, floor(0.20 * exact_unique_language_tokens))",
            f"validation target contract mismatch: {language}",
        )
        _require(
            validation.get("target_fraction") == EXPECTED_VALIDATION_FRACTION
            and validation.get("candidate_fraction")
            == EXPECTED_VALIDATION_CANDIDATE_FRACTION
            and validation.get("candidate_adequate") is True,
            f"validation candidate/final fraction contract mismatch: {language}",
        )
        validation_target = int(validation.get("target_tokens", -1))
        if bool(validation.get("complete_source_scan")):
            unique = raw.get("exact_unique_language_tokens_including_eod")
            _require(
                isinstance(unique, int) and unique > 0,
                f"complete validation inventory lacks exact U: {language}",
            )
            expected_validation = min(EXPECTED_VALIDATION_CAP, unique // 5)
        else:
            _require(
                validation.get("cap_proven_from_observed_lower_bound") is True,
                f"validation cap is not proven: {language}",
            )
            expected_validation = EXPECTED_VALIDATION_CAP
        _require(
            validation_target == expected_validation,
            f"validation quota is not the ATLAS rule: {language}",
        )
        validation_selected = int(validation.get("selected_tokens", -1))
        _require(
            validation_selected >= validation_target
            and validation.get("overshoot_tokens")
            == validation_selected - validation_target,
            f"validation whole-document accounting mismatch: {language}",
        )
        # Rebuild split membership independently from the finalized JSONL files.
        # SQLite keeps the proof exact without retaining millions of document IDs
        # in RAM.  Its primary key rejects duplicates within a split and any
        # original ID that appears in both train and validation.  Do not place
        # this disposable index under ``corpus``: a prior passing audit seals
        # that tree read-only, while a rerun must remain a pure read of it.
        # ``TemporaryDirectory`` honours TMPDIR, which the Slurm recipe routes
        # to scratch/node-local storage.
        with tempfile.TemporaryDirectory(
            prefix=f".fineweb-v2-split-audit-{language}-",
        ) as temporary_index_dir:
            original_id_index = _create_original_id_index(
                Path(temporary_index_dir) / "original_ids.sqlite3"
            )
            try:
                _train_path, train_recount = _audit_finalized_split_artifact(
                    base=release_dir,
                    raw=train.get("output"),
                    label=f"corpus train {language}",
                    hash_cache=hash_cache,
                    language=language,
                    split="train",
                    tokenizer_sha256=tokenizer_hash,
                    original_id_index=original_id_index,
                )
                _validation_path, validation_recount = (
                    _audit_finalized_split_artifact(
                        base=release_dir,
                        raw=validation.get("output"),
                        label=f"corpus validation {language}",
                        hash_cache=hash_cache,
                        language=language,
                        split="validation",
                        tokenizer_sha256=tokenizer_hash,
                        original_id_index=original_id_index,
                    )
                )
            finally:
                original_id_index.close()
        _require(
            train_recount["emitted_tokens"] == selected
            and train_recount["emitted_documents"] == train.get("emitted_documents")
            and train_recount["selected_original_documents"]
            == train.get("selected_original_documents"),
            f"corpus train output recount mismatch: {language}",
        )
        _require(
            validation_recount["emitted_tokens"] == validation_selected
            and validation_recount["emitted_documents"]
            == validation.get("emitted_documents")
            and validation_recount["selected_original_documents"]
            == validation.get("selected_original_documents"),
            f"corpus validation output recount mismatch: {language}",
        )
        total_original_ids += int(train_recount["selected_original_documents"])
        total_original_ids += int(validation_recount["selected_original_documents"])
        candidate = raw.get("validation_candidate_reserve")
        _require(
            isinstance(candidate, dict)
            and candidate.get("status") == "adequate"
            and candidate.get("candidate_adequate") is True
            and candidate.get("selection_precedes_train_selection") is True
            and candidate.get("declared_candidate_fraction")
            == EXPECTED_VALIDATION_CANDIDATE_FRACTION
            and candidate.get("final_validation_target_fraction")
            == EXPECTED_VALIDATION_FRACTION
            and candidate.get("preserved_train_multiplier_field")
            == "lineage.train_rehydration_multiplier"
            and candidate.get("selected_validation_rehydration_multiplier") == 1,
            f"validation candidate adequacy contract mismatch: {language}",
        )
        candidate_integer_fields = (
            "candidate_documents",
            "candidate_unique_tokens_including_eod",
            "candidate_train_rehydrated_tokens_if_all_reclaimed",
            "candidate_train_emitted_documents_if_all_reclaimed",
            "selected_validation_original_documents",
            "selected_validation_tokens_including_eod",
            "unused_candidate_documents_reclassified_to_train",
            "unused_candidate_unique_tokens_reclassified_to_train",
            "unused_candidate_rehydrated_tokens_available_to_train",
            "unused_candidate_emitted_documents_available_to_train",
            "selected_reclaimed_train_original_documents",
            "selected_reclaimed_train_unique_tokens_including_eod",
            "selected_reclaimed_train_tokens_including_rehydration_and_eod",
            "selected_reclaimed_train_emitted_documents",
        )
        _require(
            all(
                isinstance(candidate.get(field), int)
                and not isinstance(candidate.get(field), bool)
                and int(candidate[field]) >= 0
                for field in candidate_integer_fields
            ),
            f"validation candidate accounting has invalid counters: {language}",
        )
        _require(
            candidate["selected_validation_original_documents"]
            == validation_recount["selected_original_documents"]
            and candidate["selected_validation_tokens_including_eod"]
            == validation_recount["emitted_tokens"]
            and candidate["unused_candidate_documents_reclassified_to_train"]
            == candidate["candidate_documents"]
            - candidate["selected_validation_original_documents"]
            and candidate["unused_candidate_unique_tokens_reclassified_to_train"]
            == candidate["candidate_unique_tokens_including_eod"]
            - candidate["selected_validation_tokens_including_eod"],
            f"validation candidate selection/reclamation accounting mismatch: {language}",
        )
        _require(
            candidate["selected_reclaimed_train_original_documents"]
            == train_recount["reclaimed_original_documents"]
            and candidate["selected_reclaimed_train_unique_tokens_including_eod"]
            == train_recount["reclaimed_unique_tokens"]
            and candidate[
                "selected_reclaimed_train_tokens_including_rehydration_and_eod"
            ]
            == train_recount["reclaimed_emitted_tokens"]
            and candidate["selected_reclaimed_train_emitted_documents"]
            == train_recount["reclaimed_emitted_documents"],
            f"selected reclaimed-train output accounting mismatch: {language}",
        )
        _require(
            candidate["selected_reclaimed_train_original_documents"]
            <= candidate["unused_candidate_documents_reclassified_to_train"]
            and candidate["selected_reclaimed_train_unique_tokens_including_eod"]
            <= candidate["unused_candidate_unique_tokens_reclassified_to_train"]
            and candidate[
                "selected_reclaimed_train_tokens_including_rehydration_and_eod"
            ]
            <= candidate["unused_candidate_rehydrated_tokens_available_to_train"]
            and candidate["selected_reclaimed_train_emitted_documents"]
            <= candidate["unused_candidate_emitted_documents_available_to_train"]
            and candidate["candidate_train_rehydrated_tokens_if_all_reclaimed"]
            >= candidate["unused_candidate_rehydrated_tokens_available_to_train"]
            + candidate["selected_validation_tokens_including_eod"]
            and candidate["candidate_train_emitted_documents_if_all_reclaimed"]
            >= candidate["unused_candidate_emitted_documents_available_to_train"]
            + candidate["selected_validation_original_documents"],
            f"validation candidate availability accounting mismatch: {language}",
        )
        observed_unique = int(
            raw.get("observed_unique_language_tokens_lower_bound_including_eod", 0)
        )
        _require(
            observed_unique > 0
            and isinstance(candidate.get("observed_candidate_token_fraction"), (int, float))
            and not isinstance(candidate.get("observed_candidate_token_fraction"), bool)
            and math.isclose(
                float(candidate["observed_candidate_token_fraction"]),
                candidate["candidate_unique_tokens_including_eod"] / observed_unique,
                rel_tol=1e-15,
                abs_tol=0.0,
            ),
            f"validation candidate observed fraction mismatch: {language}",
        )
        exposure = raw.get("exposure_pass_audit")
        _require(isinstance(exposure, dict), f"exposure-pass audit missing: {language}")
        exposure_groups = dict(plan["tiers"])
        composition_arms = plan.get("composition_arms", {})
        _require(
            isinstance(composition_arms, dict),
            "sampling-plan composition_arms must be an object when present",
        )
        exposure_groups.update(
            {
                f"composition:{name}": payload
                for name, payload in composition_arms.items()
            }
        )
        expected_tiers = {
            tier
            for tier, tier_payload in exposure_groups.items()
            if language in tier_payload["languages"]
        }
        _require(
            set(exposure) == expected_tiers,
            f"exposure-pass tier coverage mismatch: {language}",
        )
        for tier in expected_tiers:
            row = exposure[tier]
            exposure_tokens = int(
                exposure_groups[tier]["target_exposure_tokens"][language]
            )
            _require(
                row.get("target_exposure_tokens") == exposure_tokens
                and row.get("physical_train_tokens") == selected,
                f"exposure/physical linkage mismatch: {tier}/{language}",
            )
            _require(
                math.isclose(
                    float(row.get("effective_passes", -1)),
                    exposure_tokens / selected,
                    rel_tol=1e-15,
                    abs_tol=0.0,
                ),
                f"effective passes mismatch: {tier}/{language}",
            )
            _require(
                row.get("full_passes") == exposure_tokens // selected
                and row.get("partial_pass_tokens") == exposure_tokens % selected,
                f"pass decomposition mismatch: {tier}/{language}",
            )
        total_physical += selected
        total_validation += validation_selected

    hash_manifest = _load_object(hash_manifest_path, label="corpus hash manifest")
    hash_manifest_hash = _verify_payload(
        hash_manifest, key="hash_manifest_payload_sha256", label="corpus hash manifest"
    )
    _require(
        hash_manifest.get("corpus_manifest_payload_sha256") == manifest_hash,
        "corpus/hash-manifest payload mismatch",
    )
    files = hash_manifest.get("files")
    _require(
        isinstance(files, dict)
        and files.get("corpus_manifest.json") == sha256_file(manifest_path),
        "corpus hash manifest lacks the exact corpus manifest",
    )
    for relative, expected_hash in files.items():
        file_path = _resolve_file(
            release_dir, relative, label=f"corpus hash-manifest file {relative}"
        )
        _require(
            _cached_sha256(file_path, hash_cache) == expected_hash,
            f"corpus release file changed: {relative}",
        )
    return manifest, {
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": manifest_hash,
        "hash_manifest_payload_sha256": hash_manifest_hash,
        "language_count": len(languages),
        "physical_train_tokens": total_physical,
        "validation_tokens": total_validation,
        "exact_physical_targets": True,
        "split_overlap": 0,
        "split_original_ids_indexed": total_original_ids,
        "split_disjointness_proof": (
            "independent_disk_backed_sqlite_primary_key_over_original_document_id"
        ),
    }
