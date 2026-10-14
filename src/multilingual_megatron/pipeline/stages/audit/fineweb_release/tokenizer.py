"""Bootstrap, sentence-artifact, tokenizer, and calibration release audits."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.audit.fineweb_release.constants import (
    EXPECTED_ALPHA,
    EXPECTED_DATA_SEED,
    EXPECTED_MEGATRON_DTYPE_CODE,
    EXPECTED_TOKENIZER_SENTENCES,
    EXPECTED_VALIDATION_CANDIDATE_FRACTION,
    EXPECTED_VALIDATION_FRACTION,
    EXPECTED_VOCAB_SIZE,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.integrity import (
    ReleaseAuditError,
    _load_object,
    _require,
    _require_sha256,
    _resolve_file,
    _under,
    _verify_payload,
    _verify_relative_artifact,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    assign_validation_reserve,
    canonical_json_sha256,
    largest_remainder_quotas,
    sha256_file,
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


def _recount_tokenizer_bootstrap_slots(
    *, path: Path, language: str, max_sentence_chars: int
) -> dict[str, int]:
    """Independently recount exact train slots for the scientific release audit."""

    documents = 0
    unique_sentences = 0
    rehydrated_slots = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            label = f"tokenizer acquisition/{language}/train.jsonl:{line_number}"
            _require(bool(line.strip()), f"blank row in {label}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseAuditError(f"invalid JSON row in {label}") from exc
            _require(isinstance(row, dict), f"non-object row in {label}")
            lineage = row.get("lineage")
            _require(
                row.get("language") == language
                and row.get("artifact_role") == "train"
                and isinstance(lineage, dict)
                and lineage.get("subset") == language
                and lineage.get("split") == "train",
                f"non-train/language row in {label}",
            )
            text = row.get("text")
            multiplier = lineage.get("rehydration_multiplier")
            _require(
                isinstance(text, str)
                and bool(text.strip())
                and isinstance(multiplier, int)
                and not isinstance(multiplier, bool)
                and multiplier >= 1,
                f"invalid text or rehydration multiplier in {label}",
            )
            sentence_count = sum(
                1
                for _ in split_sentence_like(
                    text,
                    max_chars=max_sentence_chars,
                )
            )
            _require(sentence_count > 0, f"zero sentence slots in {label}")
            documents += 1
            unique_sentences += sentence_count
            rehydrated_slots += sentence_count * multiplier
    _require(documents > 0, f"empty tokenizer acquisition train file: {language}")
    return {
        "train_documents": documents,
        "unique_train_sentences": unique_sentences,
        "rehydrated_train_sentence_slots": rehydrated_slots,
    }

def _audit_bootstrap_split_rows(
    *,
    path: Path,
    language: str,
    split: str,
    candidate_fraction: float,
) -> dict[str, int]:
    """Independently verify the deterministic candidate/train row contract."""

    if split not in {"train", "validation_reserve"}:
        raise AssertionError(f"unsupported bootstrap split audit: {split}")
    documents = 0
    train_multiplier_total = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            where = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseAuditError(f"invalid JSON in bootstrap split row: {where}") from exc
            _require(isinstance(row, dict), f"bootstrap split row is not an object: {where}")
            document_id = row.get("document_id")
            lineage = row.get("lineage")
            _require(
                isinstance(document_id, str) and bool(document_id),
                f"bootstrap split row lacks document_id: {where}",
            )
            _require(
                row.get("language") == language and isinstance(lineage, dict),
                f"bootstrap split row language/lineage mismatch: {where}",
            )
            _require(
                row.get("artifact_role") == split and lineage.get("split") == split,
                f"bootstrap split row role mismatch: {where}",
            )
            multiplier = lineage.get("rehydration_multiplier")
            train_multiplier = lineage.get("train_rehydration_multiplier")
            _require(
                isinstance(multiplier, int)
                and not isinstance(multiplier, bool)
                and multiplier >= 1,
                f"bootstrap row has invalid rehydration multiplier: {where}",
            )
            _require(
                isinstance(train_multiplier, int)
                and not isinstance(train_multiplier, bool)
                and train_multiplier >= 1,
                f"bootstrap row lacks a positive preserved train multiplier: {where}",
            )
            expected_candidate = assign_validation_reserve(
                document_id=document_id,
                seed=EXPECTED_DATA_SEED,
                validation_fraction=candidate_fraction,
            )
            _require(
                expected_candidate == (split == "validation_reserve"),
                f"bootstrap row violates seed-42 validation candidate membership: {where}",
            )
            if split == "validation_reserve":
                _require(
                    multiplier == 1,
                    f"validation candidate row is rehydrated: {where}",
                )
            else:
                _require(
                    multiplier == train_multiplier,
                    f"train row does not preserve its rehydration multiplier: {where}",
                )
            documents += 1
            train_multiplier_total += train_multiplier
    return {
        "documents": documents,
        "preserved_train_multiplier_total": train_multiplier_total,
    }


def _audit_bootstrap(
    *,
    root: Path,
    label: str,
    languages: Sequence[str],
    source_lock_sha256: str,
    language_map_file_sha256: str,
    rehydration_weights_sha256: str,
    expected_sampling_plan: tuple[str, str] | None = None,
    expected_acquisition_plan: tuple[str, str] | None = None,
    tokenizer_acquisition_plan: Mapping[str, Any] | None = None,
    require_incremental: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary_path = root / "bootstrap_summary.json"
    manifest_path = root / "bootstrap_manifest.json"
    summary = _load_object(summary_path, label=f"{label} bootstrap summary")
    manifest = _load_object(manifest_path, label=f"{label} bootstrap manifest")
    summary_hash = _verify_payload(
        summary, key="summary_sha256", label=f"{label} bootstrap summary"
    )
    manifest_hash = _verify_payload(
        manifest, key="manifest_sha256", label=f"{label} bootstrap manifest"
    )
    _require(
        summary.get("tool_id") == "fineweb_ab_v2_bootstrap_v1",
        f"{label} bootstrap tool ID mismatch",
    )
    _require(
        summary.get("languages") == list(languages),
        f"{label} bootstrap language ordering mismatch",
    )
    _require(
        summary.get("bootstrap_manifest_payload_sha256") == manifest_hash,
        f"{label} summary/manifest mismatch",
    )
    entries = manifest.get("languages")
    _require(
        isinstance(entries, dict) and set(entries) == set(languages),
        f"{label} bootstrap coverage mismatch",
    )
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), f"{label} bootstrap lacks provenance")
    _require(
        provenance.get("source_lock_sha256") == source_lock_sha256,
        f"{label} source-lock lineage mismatch",
    )
    _require(
        provenance.get("language_map_sha256") == language_map_file_sha256,
        f"{label} language-map lineage mismatch",
    )
    _require(
        provenance.get("rehydration_weights_sha256") == rehydration_weights_sha256,
        f"{label} rehydration lineage mismatch",
    )
    sampling_file, sampling_payload = expected_sampling_plan or (None, None)
    acquisition_file, acquisition_payload = expected_acquisition_plan or (None, None)
    _require(
        (expected_acquisition_plan is None)
        == (tokenizer_acquisition_plan is None),
        f"{label} exact acquisition plan/audit inputs are incomplete",
    )
    if tokenizer_acquisition_plan is not None:
        _require(
            verify_acquisition_plan(tokenizer_acquisition_plan),
            f"{label} tokenizer acquisition plan is invalid",
        )
        _require(
            acquisition_payload == tokenizer_acquisition_plan.get("plan_sha256"),
            f"{label} tokenizer acquisition payload identity mismatch",
        )
    _require(
        provenance.get("sampling_plan_file_sha256") == sampling_file,
        f"{label} sampling-plan file lineage mismatch",
    )
    _require(
        provenance.get("sampling_plan_payload_sha256") == sampling_payload,
        f"{label} sampling-plan payload lineage mismatch",
    )
    _require(
        provenance.get("tokenizer_acquisition_plan_file_sha256") == acquisition_file,
        f"{label} acquisition-plan file lineage mismatch",
    )
    _require(
        provenance.get("tokenizer_acquisition_plan_payload_sha256")
        == acquisition_payload,
        f"{label} acquisition-plan payload lineage mismatch",
    )
    if require_incremental:
        _require(
            provenance.get("incremental_mode") == "verified_full_deterministic_rebuild",
            f"{label} is not a verified full rebuild",
        )

    expected_report_hashes = summary.get("report_payload_sha256")
    _require(
        isinstance(expected_report_hashes, dict)
        and set(expected_report_hashes) == set(languages),
        f"{label} report coverage mismatch",
    )
    complete_sources = 0
    retained_split_bytes = 0
    retained_split_hashes: dict[str, dict[str, str]] = {}
    hash_cache: dict[Path, str] = {}
    tokenizer_slot_quota_total = 0
    tokenizer_slot_actual_total = 0
    tokenizer_true_exhausted_languages: list[str] = []
    candidate_documents_total = 0
    candidate_train_multiplier_total = 0
    for language in languages:
        entry = entries[language]
        _require(
            isinstance(entry, dict), f"{label}/{language} manifest entry is invalid"
        )
        acquisition = entry.get("acquisition")
        _require(
            isinstance(acquisition, dict), f"{label}/{language} lacks acquisition proof"
        )
        _require(
            acquisition.get("report_path") == f"{language}/bootstrap_report.json"
            and entry.get("train", {}).get("path") == f"{language}/train.jsonl"
            and entry.get("validation_reserve", {}).get("path")
            == f"{language}/validation_reserve.jsonl",
            f"{label}/{language} uses non-canonical or external bootstrap paths",
        )
        report_path = _resolve_file(
            root, acquisition.get("report_path"), label=f"{label}/{language} report"
        )
        _require(
            _under(report_path, root),
            f"{label}/{language} report escapes its v2 bootstrap root",
        )
        report = _load_object(report_path, label=f"{label}/{language} report")
        report_hash = _verify_payload(
            report, key="report_payload_sha256", label=f"{label}/{language} report"
        )
        _require(
            report_hash
            == expected_report_hashes[language]
            == acquisition.get("report_payload_sha256"),
            f"{label}/{language} report identity mismatch",
        )
        _require(
            report.get("status") == "complete",
            f"{label}/{language} report is incomplete",
        )
        run_config = report.get("run_config")
        settings = run_config.get("settings") if isinstance(run_config, dict) else None
        legacy_fraction = (
            settings.get("validation_fraction") if isinstance(settings, dict) else None
        )
        candidate_fraction = (
            settings.get("validation_candidate_fraction")
            if isinstance(settings, dict)
            else None
        )
        candidate_contract = (
            run_config.get("validation_candidate_contract")
            if isinstance(run_config, dict)
            else None
        )
        _require(
            isinstance(run_config, dict)
            and run_config.get("language") == language
            and isinstance(settings, dict)
            and settings.get("seed") == EXPECTED_DATA_SEED
            and isinstance(legacy_fraction, (int, float))
            and not isinstance(legacy_fraction, bool)
            and isinstance(candidate_fraction, (int, float))
            and not isinstance(candidate_fraction, bool)
            and float(legacy_fraction) == EXPECTED_VALIDATION_CANDIDATE_FRACTION
            and float(candidate_fraction) == EXPECTED_VALIDATION_CANDIDATE_FRACTION
            and float(legacy_fraction) == float(candidate_fraction),
            f"{label}/{language} seed/candidate-fraction contract mismatch",
        )
        _require(
            candidate_contract
            == {
                "membership_namespace": "validation-reserve-v1",
                "excluded_from_tokenizer_training": True,
                "unused_candidates_return_to_model_training_selection_pool": True,
                "preserved_train_rehydration_multiplier": True,
            },
            f"{label}/{language} validation candidate contract mismatch",
        )
        counts = report.get("counts")
        _require(
            isinstance(counts, dict) and counts.get("train_validation_overlap") == 0,
            f"{label}/{language} split overlap is nonzero",
        )
        hashes = report.get("hashes")
        _require(isinstance(hashes, dict), f"{label}/{language} lacks split-ID hashes")
        for name in (
            "unique_source_ids_sha256",
            "unique_train_ids_sha256",
            "unique_validation_ids_sha256",
        ):
            _require_sha256(hashes.get(name), field=f"{label}/{language}.{name}")
        _require(
            hashes["unique_train_ids_sha256"] != hashes["unique_validation_ids_sha256"],
            f"{label}/{language} train/validation ID hashes unexpectedly match",
        )
        outputs = report.get("outputs")
        _require(isinstance(outputs, dict), f"{label}/{language} lacks output lineage")
        retained_split_hashes[language] = {}
        resolved_splits: dict[str, Path] = {}
        for split in ("train.jsonl", "validation_reserve.jsonl", "calibration.jsonl"):
            raw_output = outputs.get(split)
            _require(isinstance(raw_output, dict), f"{label}/{language} lacks {split}")
            if split != "calibration.jsonl":
                manifest_output = entry[
                    "train" if split == "train.jsonl" else "validation_reserve"
                ]
                _require(
                    manifest_output.get("sha256") == raw_output.get("sha256"),
                    f"{label}/{language}/{split} manifest hash mismatch",
                )
            split_path = _verify_relative_artifact(
                base=report_path.parent,
                raw=raw_output,
                label=f"{label}/{language}/{split}",
                hash_cache=hash_cache,
            )
            expected_path = (root / language / split).resolve()
            _require(
                split_path == expected_path and _under(split_path, root),
                f"{label}/{language}/{split} escapes its canonical v2 bootstrap path",
            )
            retained_split_bytes += split_path.stat().st_size
            retained_split_hashes[language][split] = str(raw_output["sha256"])
            resolved_splits[split] = split_path
        train_row_audit = _audit_bootstrap_split_rows(
            path=resolved_splits["train.jsonl"],
            language=language,
            split="train",
            candidate_fraction=float(candidate_fraction),
        )
        candidate_row_audit = _audit_bootstrap_split_rows(
            path=resolved_splits["validation_reserve.jsonl"],
            language=language,
            split="validation_reserve",
            candidate_fraction=float(candidate_fraction),
        )
        candidate_documents_total += candidate_row_audit["documents"]
        candidate_train_multiplier_total += candidate_row_audit[
            "preserved_train_multiplier_total"
        ]
        _require(
            train_row_audit["preserved_train_multiplier_total"]
            >= train_row_audit["documents"],
            f"{label}/{language} train multiplier audit is invalid",
        )
        targets = report.get("targets")
        _require(isinstance(targets, dict), f"{label}/{language} lacks target audit")
        if tokenizer_acquisition_plan is not None:
            try:
                expected_target = tokenizer_acquisition_target_contract(
                    plan=tokenizer_acquisition_plan,
                    plan_file_sha256=str(acquisition_file),
                    language=language,
                )
            except ValueError as exc:
                raise ReleaseAuditError(
                    f"{label}/{language} acquisition target contract is invalid"
                ) from exc
            _require(
                run_config.get("tokenizer_sentence_acquisition_target")
                == expected_target,
                f"{label}/{language} acquisition target lineage mismatch",
            )
            gate_contract = expected_target["sentence_slot_gate"]
            splitter = gate_contract["splitter"]
            recount = _recount_tokenizer_bootstrap_slots(
                path=resolved_splits["train.jsonl"],
                language=language,
                max_sentence_chars=int(splitter["max_sentence_chars"]),
            )
            required_slots = int(
                expected_target["required_rehydrated_sentence_slots"]
            )
            actual_slots = int(recount["rehydrated_train_sentence_slots"])
            shortfall_slots = max(0, required_slots - actual_slots)
            source_exhausted = targets.get("source_exhausted")
            _require(
                isinstance(source_exhausted, bool),
                f"{label}/{language} true-source exhaustion flag is invalid",
            )
            expected_gate = {
                "counter_id": SENTENCE_SLOT_COUNTER_ID,
                "counted_split": gate_contract["counted_split"],
                "slot_formula": gate_contract["slot_formula"],
                "splitter": splitter,
                "non_exhausted_stop_rule": gate_contract[
                    "non_exhausted_stop_rule"
                ],
                "required_rehydrated_sentence_slots": required_slots,
                "actual_unique_train_sentences": int(
                    recount["unique_train_sentences"]
                ),
                "actual_rehydrated_sentence_slots": actual_slots,
                "shortfall_slots": shortfall_slots,
                "quota_satisfied": shortfall_slots == 0,
                "true_pinned_source_exhausted": source_exhausted,
                "true_source_exhausted_repetition_allowed": bool(
                    source_exhausted and shortfall_slots > 0
                ),
                "artificial_pool_cycling_forbidden": True,
            }
            _require(
                counts.get("unique_train_documents")
                == recount["train_documents"],
                f"{label}/{language} exact train-document recount drifted",
            )
            _require(
                counts.get("unique_train_sentences")
                == recount["unique_train_sentences"],
                f"{label}/{language} exact unique-sentence recount drifted",
            )
            _require(
                counts.get("rehydrated_train_sentence_slots") == actual_slots,
                f"{label}/{language} exact rehydrated-slot recount drifted",
            )
            _require(
                targets.get("tokenizer_sentence_slot_gate") == expected_gate,
                f"{label}/{language} exact tokenizer sentence-slot gate drifted",
            )
            _require(
                acquisition.get("tokenizer_sentence_slot_gate") == expected_gate,
                f"{label}/{language} manifest sentence-slot gate drifted",
            )
            _require(
                shortfall_slots == 0 or source_exhausted,
                f"{label}/{language} has a non-exhausted tokenizer slot shortfall",
            )
            tokenizer_slot_quota_total += required_slots
            tokenizer_slot_actual_total += actual_slots
            if source_exhausted:
                tokenizer_true_exhausted_languages.append(language)
        complete = bool(
            targets.get("source_exhausted")
            or targets.get("pinned_inventory_exhausted")
        )
        _require(
            bool(acquisition.get("complete_source_scan")) == complete,
            f"{label}/{language} complete-source proof mismatch",
        )
        complete_sources += int(complete)
    audit = {
        "manifest_payload_sha256": manifest_hash,
        "summary_payload_sha256": summary_hash,
        "language_count": len(languages),
        "complete_source_count": complete_sources,
        "split_overlap_count": 0,
        "retained_split_file_count": 3 * len(languages),
        "retained_split_bytes": retained_split_bytes,
        "retained_split_artifact_set_sha256": canonical_json_sha256(
            retained_split_hashes
        ),
        "retained_split_files_hash_verified": True,
        "validation_candidate_contract": {
            "candidate_fraction": EXPECTED_VALIDATION_CANDIDATE_FRACTION,
            "final_target_fraction": EXPECTED_VALIDATION_FRACTION,
            "seed": EXPECTED_DATA_SEED,
            "candidate_documents": candidate_documents_total,
            "preserved_train_multiplier_total": candidate_train_multiplier_total,
            "rows_independently_verified": True,
        },
    }
    if tokenizer_acquisition_plan is not None:
        audit["tokenizer_sentence_slot_gate"] = {
            "counter_id": SENTENCE_SLOT_COUNTER_ID,
            "required_rehydrated_sentence_slots": tokenizer_slot_quota_total,
            "actual_rehydrated_sentence_slots": tokenizer_slot_actual_total,
            "true_source_exhausted_language_count": len(
                tokenizer_true_exhausted_languages
            ),
            "true_source_exhausted_languages": tokenizer_true_exhausted_languages,
            "non_exhausted_shortfall_count": 0,
            "independent_train_jsonl_recount": True,
        }
    return (
        manifest,
        summary,
        audit,
    )


def _audit_acquisition_plan(
    *,
    path: Path,
    languages: Sequence[str],
    language_map_hash: str,
    pilot_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load_object(path, label="tokenizer acquisition plan")
    _require(
        verify_acquisition_plan(plan),
        "tokenizer acquisition plan hash/schema/status is invalid",
    )
    _require(
        plan.get("tool_id") == ACQUISITION_PLAN_TOOL_ID,
        "tokenizer acquisition tool ID mismatch",
    )
    _require(
        plan.get("test_mode") is False,
        "production tokenizer acquisition plan is in test mode",
    )
    _require(
        plan.get("target_sentence_count") == EXPECTED_TOKENIZER_SENTENCES,
        "tokenizer acquisition target is not 100M",
    )
    _require(
        plan.get("alpha") == EXPECTED_ALPHA and plan.get("seed") == EXPECTED_DATA_SEED,
        "tokenizer acquisition alpha/seed mismatch",
    )
    _require(
        plan.get("acquisition_safety_margin") == 1.1,
        "tokenizer acquisition safety margin mismatch",
    )
    _require(
        plan.get("ordered_languages") == list(languages)
        and plan.get("language_count") == len(languages),
        "tokenizer acquisition languages mismatch",
    )
    inputs = plan.get("inputs", {})
    _require(
        inputs.get("language_map", {}).get("sha256") == language_map_hash,
        "tokenizer acquisition K200 map mismatch",
    )
    _require(
        inputs.get("pilot_bootstrap_summary", {}).get("sha256")
        == sha256_file(pilot_summary_path),
        "tokenizer acquisition pilot-summary mismatch",
    )
    mixture = plan.get("mixture")
    _require(
        isinstance(mixture, dict)
        and mixture.get("availability_estimator_id") == AVAILABILITY_ESTIMATOR_ID
        and mixture.get("rehydration_affects_language_level_probability") is False,
        "tokenizer acquisition language mixture uses the wrong availability rule",
    )
    splitter = plan.get("splitter")
    _require(
        isinstance(splitter, dict)
        and isinstance(splitter.get("max_sentence_chars"), int)
        and not isinstance(splitter.get("max_sentence_chars"), bool),
        "tokenizer acquisition splitter contract is invalid",
    )
    expected_gate = sentence_slot_gate_contract(
        max_chars=int(splitter["max_sentence_chars"])
    )
    gate = plan.get("expanded_pool_gate")
    _require(
        gate == expected_gate
        and gate.get("counter_id") == SENTENCE_SLOT_COUNTER_ID,
        "tokenizer acquisition expanded-pool gate drifted",
    )
    raw_languages = plan.get("languages")
    _require(
        isinstance(raw_languages, dict) and set(raw_languages) == set(languages),
        "tokenizer acquisition quota coverage mismatch",
    )
    probabilities: dict[str, float] = {}
    quotas: dict[str, int] = {}
    for language in languages:
        raw = raw_languages[language]
        _require(
            isinstance(raw, dict), f"tokenizer acquisition entry is invalid: {language}"
        )
        probability = float(raw.get("alpha_probability", 0.0))
        quota = int(raw.get("planned_sentence_count", 0))
        acquisition = raw.get("acquisition")
        availability = raw.get("sentence_availability")
        _require(
            probability > 0 and math.isfinite(probability) and quota > 0,
            f"invalid tokenizer quota: {language}",
        )
        _require(
            isinstance(acquisition, dict)
            and acquisition.get("required_rehydrated_sentence_slots") == quota,
            f"tokenizer acquisition target/quota mismatch: {language}",
        )
        _require(
            isinstance(availability, dict)
            and availability.get("rehydration_used_in_language_level_availability")
            is False,
            f"rehydration leaked into tokenizer q_l: {language}",
        )
        probabilities[language] = probability
        quotas[language] = quota
    _require(
        math.isclose(
            math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "tokenizer acquisition probabilities do not sum to one",
    )
    _require(
        sum(quotas.values()) == EXPECTED_TOKENIZER_SENTENCES,
        "tokenizer sentence quotas do not sum to 100M",
    )
    _require(
        quotas == largest_remainder_quotas(EXPECTED_TOKENIZER_SENTENCES, probabilities),
        "tokenizer sentence quotas are not the frozen largest-remainder allocation",
    )
    return plan, {
        "file_sha256": sha256_file(path),
        "payload_sha256": plan["plan_sha256"],
        "language_count": len(languages),
        "quota_total": sum(quotas.values()),
        "sentence_slot_counter_id": SENTENCE_SLOT_COUNTER_ID,
        "exact_expanded_pool_gate_verified": True,
    }


def _audit_sentence_artifact(
    *,
    manifest_path: Path,
    sentence_path: Path,
    acquisition_path: Path,
    acquisition_plan: Mapping[str, Any],
    tokenizer_bootstrap_summary_path: Path,
    languages: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_object(manifest_path, label="tokenizer sentence manifest")
    manifest_hash = _verify_payload(
        manifest, key="manifest_sha256", label="tokenizer sentence manifest"
    )
    _require(
        manifest.get("tool_id") == "tokenizer_sentence_artifact_v1"
        and manifest.get("status") == "complete"
        and manifest.get("test_mode") is False,
        "tokenizer sentence artifact is not a complete production artifact",
    )
    _require(
        manifest.get("target_sentence_count") == EXPECTED_TOKENIZER_SENTENCES
        and manifest.get("realized_sentence_count") == EXPECTED_TOKENIZER_SENTENCES
        and manifest.get("language_count") == len(languages),
        "tokenizer sentence total/language count mismatch",
    )
    _require(
        manifest.get("alpha") == EXPECTED_ALPHA
        and manifest.get("seed") == EXPECTED_DATA_SEED,
        "tokenizer sentence alpha/seed mismatch",
    )
    algorithms = manifest.get("algorithms", {})
    _require(
        algorithms.get("splitter", {}).get("id") == SPLITTER_ID,
        "tokenizer sentence splitter mismatch",
    )
    _require(
        algorithms.get("within_language_draw", {}).get("id") == WITHIN_LANGUAGE_DRAW_ID,
        "tokenizer within-language draw mismatch",
    )
    _require(
        algorithms.get("interleaver", {}).get("id") == INTERLEAVER_ID,
        "tokenizer interleaver mismatch",
    )
    quota_rule = algorithms.get("quota", {})
    _require(
        quota_rule.get("availability_estimator_id") == AVAILABILITY_ESTIMATOR_ID
        and quota_rule.get("rehydration_affects_language_level_probability") is False,
        "tokenizer sentence q_l contract mismatch",
    )
    inputs = manifest.get("inputs", {})
    plan_input = inputs.get("tokenizer_sentence_acquisition_plan")
    _require(
        isinstance(plan_input, dict)
        and plan_input.get("file_sha256") == sha256_file(acquisition_path)
        and plan_input.get("payload_sha256") == acquisition_plan.get("plan_sha256"),
        "tokenizer sentence acquisition-plan lineage mismatch",
    )
    _require(
        inputs.get("bootstrap_train_manifest", {}).get("sha256")
        == sha256_file(tokenizer_bootstrap_summary_path),
        "tokenizer sentence expanded-bootstrap lineage mismatch",
    )
    _require(
        inputs.get("validation_files_opened") is False,
        "tokenizer training opened validation data",
    )
    output = manifest.get("output")
    _require(
        isinstance(output, dict), "tokenizer sentence manifest lacks output identity"
    )
    _require(
        _resolve_file(
            manifest_path.parent, output.get("path"), label="tokenizer sentence JSONL"
        )
        == sentence_path.resolve(),
        "tokenizer sentence output path mismatch",
    )
    _require(
        sentence_path.stat().st_size == output.get("size_bytes"),
        "tokenizer sentence file size mismatch",
    )
    _require(
        sha256_file(sentence_path) == output.get("sha256"),
        "tokenizer sentence file hash mismatch",
    )
    _require(
        output.get("sentence_count") == EXPECTED_TOKENIZER_SENTENCES,
        "tokenizer sentence output count mismatch",
    )
    raw_languages = manifest.get("languages")
    _require(
        isinstance(raw_languages, dict) and set(raw_languages) == set(languages),
        "tokenizer sentence language coverage mismatch",
    )
    repeated = 0
    for language in languages:
        raw = raw_languages[language]
        gate = raw.get("expanded_pool_gate") if isinstance(raw, dict) else None
        _require(
            isinstance(gate, dict), f"tokenizer sentence pool gate missing: {language}"
        )
        _require(
            gate.get("status")
            in {"sufficient", "true_source_exhausted_repetition_required"},
            f"artificial tokenizer cycling detected: {language}",
        )
        _require(
            gate.get("artificial_cycle_forbidden") is True,
            f"tokenizer cycling gate drifted: {language}",
        )
        _require(
            raw.get("planned_sentence_count") == raw.get("realized_sentence_count"),
            f"tokenizer quota not realized: {language}",
        )
        repeated += int(raw.get("repeated_sentence_count", 0))
    totals = manifest.get("totals", {})
    _require(
        totals.get("planned_sentence_count")
        == totals.get("realized_sentence_count")
        == EXPECTED_TOKENIZER_SENTENCES,
        "tokenizer sentence totals drifted",
    )
    _require(
        totals.get("repeated_sentence_count") == repeated,
        "tokenizer repeated-sentence accounting mismatch",
    )
    return manifest, {
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": manifest_hash,
        "sentence_jsonl_sha256": output["sha256"],
        "sentence_count": EXPECTED_TOKENIZER_SENTENCES,
        "true_source_exhaustion_repeats": repeated,
    }


def _audit_tokenizer(
    *,
    tokenizer_root: Path,
    sentence_path: Path,
    sentence_manifest: Mapping[str, Any],
    expected_model_prefix: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = tokenizer_root / "tokenizer_training_result.json"
    model_path = tokenizer_root / "tokenizer.model"
    result = _load_object(result_path, label="tokenizer training result")
    _require(
        result.get("schema_version") == 2
        and result.get("production_v2", {}).get("enabled") is True,
        "tokenizer is not a production-v2 training result",
    )
    _require(
        result.get("mode") == "train_unigram"
        and result.get("tokenizer_type") == "SentencePieceTokenizer",
        "tokenizer implementation/model type mismatch",
    )
    _require(
        result.get("vocab_size")
        == EXPECTED_VOCAB_SIZE
        == result.get("requested_vocab_size"),
        "tokenizer vocabulary is not 64K",
    )
    source = result.get("production_v2", {}).get("source_artifact")
    _require(
        isinstance(source, dict), "tokenizer result lacks sentence-source identity"
    )
    _require(
        Path(str(source.get("path", ""))).resolve() == sentence_path.resolve(),
        "tokenizer used a different sentence file",
    )
    _require(
        source.get("sha256") == sentence_manifest.get("output", {}).get("sha256"),
        "tokenizer sentence hash lineage mismatch",
    )
    _require(
        source.get("sentence_count") == EXPECTED_TOKENIZER_SENTENCES,
        "tokenizer source sentence count mismatch",
    )
    model_hash = sha256_file(model_path)
    _require(
        result.get("tokenizer_model_sha256") == model_hash,
        "tokenizer model hash mismatch",
    )
    num_threads = int(result.get("num_threads", 0))
    _require(num_threads > 0, "tokenizer result has invalid thread count")
    try:
        effective = audit_production_v2_model(
            model_path,
            expected_vocab_size=EXPECTED_VOCAB_SIZE,
            expected_num_threads=num_threads,
            expected_model_prefix=str(
                (
                    expected_model_prefix
                    if expected_model_prefix is not None
                    else tokenizer_root / "tokenizer"
                ).resolve()
            ),
        )
    except SystemExit as exc:
        raise ReleaseAuditError(
            f"SentencePiece effective-config audit failed: {exc}"
        ) from exc
    _require(
        result.get("effective_config_sha256")
        == effective.get("effective_config_sha256"),
        "tokenizer effective-config hash mismatch",
    )
    return result, {
        "result_file_sha256": sha256_file(result_path),
        "model_sha256": model_hash,
        "effective_config_sha256": effective["effective_config_sha256"],
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "dtype_code_required_by_megatron": EXPECTED_MEGATRON_DTYPE_CODE,
    }


def _audit_calibration(
    *,
    path: Path,
    languages: Sequence[str],
    tokenizer_hash: str,
    source_manifest: Mapping[str, Any],
    pilot_summary_path: Path,
    pilot_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = _load_object(path, label="token calibration")
    artifact_hash = _verify_payload(
        calibration, key="artifact_sha256", label="token calibration"
    )
    _require(
        calibration.get("schema_version") == 1
        and calibration.get("tool_id") == "fineweb_ab_v2_token_calibration_v1"
        and calibration.get("status") == "complete",
        "token calibration schema/status mismatch",
    )
    raw_languages = calibration.get("languages")
    _require(
        isinstance(raw_languages, dict) and set(raw_languages) == set(languages),
        "token calibration language coverage mismatch",
    )
    provenance = calibration.get("provenance")
    _require(isinstance(provenance, dict), "token calibration lacks provenance")
    _require(
        provenance.get("tokenizer_model_sha256") == tokenizer_hash,
        "calibration tokenizer lineage mismatch",
    )
    _require(
        provenance.get("source_lock_sha256")
        == source_manifest.get("source_lock_sha256"),
        "calibration source-lock lineage mismatch",
    )
    inventory = source_manifest.get("artifacts", {}).get("inventory", {})
    _require(
        provenance.get("fineweb2_inventory_sha256") == inventory.get("sha256"),
        "calibration inventory lineage mismatch",
    )
    _require(
        provenance.get("bootstrap_summary_file_sha256")
        == sha256_file(pilot_summary_path),
        "calibration pilot summary lineage mismatch",
    )
    _require(
        provenance.get("bootstrap_manifest_file_sha256")
        == sha256_file(pilot_manifest_path),
        "calibration pilot manifest lineage mismatch",
    )
    tokenizer = provenance.get("tokenizer")
    _require(
        isinstance(tokenizer, dict)
        and tokenizer.get("implementation") == "sentencepiece"
        and tokenizer.get("model_type") == "unigram"
        and tokenizer.get("vocab_size") == EXPECTED_VOCAB_SIZE
        and {
            key: tokenizer.get(key) for key in ("unk_id", "bos_id", "eos_id", "pad_id")
        }
        == {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": 3},
        "calibration tokenizer contract mismatch",
    )
    audit = calibration.get("audit")
    _require(
        isinstance(audit, dict)
        and audit.get("expected_language_count") == len(languages)
        and audit.get("observed_language_count") == len(languages)
        and audit.get("calibration_sample_seed") == EXPECTED_DATA_SEED
        and audit.get("target_validation_fraction")
        == EXPECTED_VALIDATION_FRACTION
        and audit.get("all_samples_unique_train_originals") is True
        and audit.get("coverage_complete") is True,
        "calibration audit contract mismatch",
    )
    language_audits = calibration.get("language_audits")
    _require(
        isinstance(language_audits, dict)
        and set(language_audits) == set(languages),
        "calibration per-language audit coverage mismatch",
    )
    for language, raw in raw_languages.items():
        _require(
            isinstance(raw, dict)
            and int(raw.get("calibration_documents", 0)) > 0
            and int(raw.get("calibration_tokens", 0)) > 0
            and int(raw.get("calibration_rehydrated_tokens", 0)) > 0
            and math.isclose(
                float(raw.get("validation_fraction", -1)),
                EXPECTED_VALIDATION_FRACTION,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            f"invalid token calibration: {language}",
        )
        language_audit = language_audits.get(language)
        _require(
            isinstance(language_audit, dict)
            and language_audit.get("validation_contract")
            == {
                "candidate_reserve_fraction": (
                    EXPECTED_VALIDATION_CANDIDATE_FRACTION
                ),
                "target_token_fraction": EXPECTED_VALIDATION_FRACTION,
                "unused_candidates_return_to_model_training_selection_pool": True,
                "candidate_reserve_excluded_from_tokenizer_training": True,
            },
            f"invalid calibration validation contract: {language}",
        )
    return calibration, {
        "file_sha256": sha256_file(path),
        "payload_sha256": artifact_hash,
        "language_count": len(languages),
        "sample_seed": EXPECTED_DATA_SEED,
    }
