"""Load and fail-closed validate the multilingual MoE A/B-v2 study contract."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from multilingual_megatron.contracts.language_sets import (
    K1_CODES,
    K12_CODES,
    K50_CODES,
    SHORT_CODE_TO_FINEWEB2_SUBSET,
)
from multilingual_megatron.repo import find_repo_root


DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "study_spec_jointmoe_paper_lr_v1.json"
)
EXPECTED_K = (1, 12, 50, 200)
LANGUAGE_ID_RE = re.compile(r"^[a-z]{3}_[A-Za-z]{4}$")


class SpecValidationError(ValueError):
    """Raised when the frozen A/B-v2 contract is inconsistent or incomplete."""


def _nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for field in dotted_path.split("."):
        if not isinstance(value, dict) or field not in value:
            raise SpecValidationError(f"Frozen study contract is missing {dotted_path}")
        value = value[field]
    return value


def _validate_frozen_scientific_contract(spec: dict[str, Any]) -> None:
    """Reject silent drift in every decision needed before A/B-v2 launch."""

    exact: dict[str, Any] = {
        "axes.experts": [1, 4, 16, 32],
        "axes.language_counts": [1, 12, 50, 200],
        "architecture.family": "decoder_only_transformer",
        "architecture.context_length": 2048,
        "architecture.E1_control": "true dense Transformer; no router or MoE auxiliary losses",
        "architecture.active_experts_per_token": 1,
        "architecture.router_probability_order": "softmax before Top-1 selection",
        "architecture.router_precision": "FP32",
        "architecture.activation": "SwiGLU",
        "architecture.normalization": "RMSNorm",
        "architecture.normalization_epsilon": 1e-6,
        "architecture.position_embedding": "RoPE",
        "architecture.rotary_base": 10_000,
        "architecture.linear_bias": False,
        "architecture.qkv_bias": False,
        "architecture.hidden_dropout": 0.0,
        "architecture.attention_dropout": 0.0,
        "architecture.tie_input_output_embeddings": False,
        "architecture.load_balancing_aux_loss_coefficient": 0.01,
        "architecture.router_z_loss_coefficient": 0.001,
        "architecture.expert_capacity_factor": None,
        "architecture.pad_expert_input_to_capacity": False,
        "architecture.shared_experts": 0,
        "architecture.grouped_gemm": True,
        "data_policy.sampling_seed": 42,
        "data_policy.fresh_v2_resample_required": True,
        "data_policy.reuse_v1_sampled_or_tokenized_artifacts": False,
        "data_policy.document_level_sampling": True,
        "data_policy.fineweb2_rehydration.enabled": True,
        "data_policy.fineweb2_rehydration.missing_weights_policy": "fail closed except for source-lock-pinned explicit identity overrides",
        "data_policy.fineweb2_rehydration.explicit_identity_overrides": {
            "hau_Latn": "multiplier 1 because the subset was added after upstream generated its per-language weights and no Hausa filtering-rate distribution was released"
        },
        "data_policy.language_sampling.distribution": "p_l proportional to n_l^alpha",
        "data_policy.language_sampling.alpha": 0.3,
        "data_policy.language_sampling.availability_unit": "estimated post-final-validation rehydration-weighted tokenizer token exposure",
        "data_policy.language_sampling.availability_estimator": {
            "method": "per-language tokenizer-fertility sample scaled by frozen full UTF-8 inventory",
            "calibration_documents_per_language": 1000,
            "seed": 42,
            "final_validation_rule": "min(20,000,000 tokens, 20% of estimated unique language tokens)",
            "estimated_train_token_rule": "estimated unique tokens minus the capped final-validation estimate",
            "rehydration_multiplier": "token-weighted mean over the calibration sample",
            "english_byte_inventory_exception": "reported words times the frozen calibration bytes-per-word ratio",
        },
        "data_policy.language_sampling.realized_token_accounting": "selected and exposure token counts are measured exactly in release artifacts",
        "data_policy.language_sampling.draw_unit": "whole document",
        "data_policy.language_sampling.within_language_document_order": "ascending SHA-256 of sampling_seed, split namespace, and stable original document ID",
        "data_policy.language_sampling.selection_namespaces": {
            "train": "physical-train-document-selection-v1",
            "validation": "validation-panel-document-selection-v1",
            "validation_reserve_membership": "validation-reserve-v1",
        },
        "data_policy.language_sampling.hard_repetition_cap": None,
        "data_policy.physical_release_target_tokens_per_K": 32_000_000_000,
        "tokenizer_policy.implementation": "SentencePiece",
        "tokenizer_policy.model_type": "unigram",
        "tokenizer_policy.vocab_size": 64_000,
        "tokenizer_policy.shared_across_all_K": True,
        "tokenizer_policy.training_sample_sentences": 100_000_000,
        "tokenizer_policy.training_sample_language_alpha": 0.3,
        "tokenizer_policy.training_sample_seed": 42,
        "tokenizer_policy.training_data": "training split only",
        "tokenizer_policy.language_availability_estimator": "pinned full post-reserve document inventory times deterministic pilot unique train sentences per train document; exact for exhausted sources",
        "tokenizer_policy.language_probability_rehydration_rule": "FineWeb2 rehydration does not change tokenizer language probabilities and acts only within language",
        "tokenizer_policy.acquisition_protocol": "pilot, frozen quotas, then immutable per-language rebuild with a 1.10 document/byte planning floor and exact train-side weighted sentence-slot stopping",
        "tokenizer_policy.sentence_slot_counter": "sum over train documents of deterministic sentence count times the pinned within-language rehydration multiplier",
        "tokenizer_policy.non_exhausted_stop_rule": "document, byte, calibration, and required weighted sentence-slot minima must all be met",
        "tokenizer_policy.artificial_pilot_cycling": "forbidden; cycling is allowed only after proven complete pinned-source exhaustion",
        "tokenizer_policy.normalization": "nmt_nfkc",
        "tokenizer_policy.character_coverage": 0.999995,
        "tokenizer_policy.byte_fallback": True,
        "tokenizer_policy.hard_vocab_limit": True,
        "tokenizer_policy.special_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 3},
        "validation_policy.split_before_tokenizer_and_training_sampling": True,
        "validation_policy.pre_tokenizer_candidate_reserve_fraction": 0.3,
        "validation_policy.final_validation_token_fraction": 0.2,
        "validation_policy.candidate_adequacy_gate": "fail before H100 work if the candidate pool cannot reach the exact final-token target for any language",
        "validation_policy.draw_unit": "whole document",
        "validation_policy.per_language_target": "min(20,000,000 tokens, 20% of unique post-filter language tokens)",
        "validation_policy.primary_metric": "per-language ATLAS/Tao unigram-normalized loss",
        "validation_policy.secondary_metric": "raw per-language token NLL",
        "validation_policy.primary_aggregate": "equal-language macro average",
        "optimizer_policy.name": "AdamW",
        "optimizer_policy.beta1": 0.9,
        "optimizer_policy.beta2": 0.95,
        "optimizer_policy.epsilon": 1e-8,
        "optimizer_policy.weight_decay": 0.1,
        "optimizer_policy.weight_decay_schedule": "constant",
        "optimizer_policy.gradient_clip_norm": 1.0,
        "optimizer_policy.minimum_learning_rate": 0.0,
        "training_policy.model_seed": 1234,
        "training_policy.dataset_shuffle_seed": 1234,
        "training_policy.warmup_tokens": 130_000_000,
        "training_policy.schedule": "warmup, stable, then linear decay to zero over the final 20% of D_max",
        "training_policy.deprecated_checkpoint_cooldown_rule": "forbidden: min(5% of D, 256M tokens)",
        "training_policy.checkpoint_horizons_tokens.A": [500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000],
        "training_policy.checkpoint_horizons_tokens.B": [500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000, 16_000_000_000],
    }
    schema_version = spec.get("schema_version")
    if schema_version == 3:
        exact.update(
            {
                "training_policy.replication_training_seeds": [1234, 5678, 91011],
                "training_policy.replication_seed_scope": "each replication seed controls both model RNG and dataset ordering; corpus construction remains fixed at seed 42",
                "training_policy.learning_rate_prescription": {
                    "source": "JointMoE paper arXiv:2502.05172, printed rounded-coefficient law",
                    "formula": "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))",
                    "intercept": 8.39,
                    "active_parameter_coefficient": -0.81,
                    "expert_coefficient": -0.25,
                    "parameter_count_basis": "paper-style active non-embedding parameter count N_act_nonembed",
                    "constant_across_K": True,
                    "released_code_full_precision_coefficients_used": False,
                    "per_cell_selection_allowed": False,
                },
                "training_policy.prescribed_learning_rate_verification": {
                    "cells": ["A/E1/K1", "A/E1/K200", "B/E32/K1", "B/E32/K200"],
                    "total_tokens": 167_772_160,
                    "global_batch_sequences": 32,
                    "total_steps": 2_560,
                    "warmup_actual_tokens": 130_023_424,
                    "warmup_steps": 1_984,
                    "schedule": "linear warmup to each cell's prescribed peak learning rate, then constant peak learning rate",
                    "selection_or_mutation_allowed": False,
                    "fit_observation": False,
                },
                "training_policy.learning_rate_transfer_diagnostic": {
                    "horizon_tokens": 2_000_000_000,
                    "cells": ["B/E1/K1", "B/E1/K200", "B/E32/K1", "B/E32/K200"],
                    "multipliers": [0.5, 1.0, 2.0],
                    "comparison_seed": 1234,
                    "clear_improvement_threshold": "larger than the same cell's three-seed 1x primary-validation loss range",
                    "block_rule": "block if the same neighboring multiplier clearly improves at least three of four cells",
                    "automatic_learning_rate_mutation": False,
                },
                "training_policy.seed_replication_panel": {
                    "horizon_tokens": 2_000_000_000,
                    "cells": [
                        "A/E1/K1",
                        "A/E1/K200",
                        "A/E32/K1",
                        "A/E32/K200",
                        "B/E1/K1",
                        "B/E1/K200",
                        "B/E32/K1",
                        "B/E32/K200",
                    ],
                    "training_seeds": [1234, 5678, 91011],
                    "learning_rate_multiplier": 1.0,
                    "inference_scope": "descriptive A/B uncertainty reference; not a formal confidence interval or a C/D seed-variance estimate",
                },
            }
        )
    elif schema_version == 2:
        exact.update(
            {
                "training_policy.peak_learning_rate": 0.0002,
                "training_policy.learning_rate_rule": "fixed peak learning rate 2e-4 across every N, E, and K cell; no learning-rate calibration or selection",
                "training_policy.learning_rate_limitation": "the common literature-backed rate is controlled across cells but is not claimed to be optimal for every model size or expert count",
                "training_policy.fixed_learning_rate_verification": {
                    "cells": ["A/E1/K1", "A/E1/K200", "B/E32/K1", "B/E32/K200"],
                    "total_tokens": 167_772_160,
                    "global_batch_sequences": 32,
                    "total_steps": 2_560,
                    "warmup_actual_tokens": 130_023_424,
                    "warmup_steps": 1_984,
                    "schedule": "linear warmup to the fixed 2e-4 peak learning rate, then constant peak learning rate",
                    "selection_or_mutation_allowed": False,
                    "fit_observation": False,
                },
            }
        )
    else:
        raise SpecValidationError("Unexpected A/B-v2 schema version")
    drift = [
        f"{path}: expected {expected!r}, observed {_nested_value(spec, path)!r}"
        for path, expected in exact.items()
        if _nested_value(spec, path) != expected
    ]
    if drift:
        raise SpecValidationError("Frozen A/B-v2 scientific contract drift:\n" + "\n".join(drift))

    training_policy = _nested_value(spec, "training_policy")
    if "learning_rate_calibration" in training_policy:
        raise SpecValidationError("Learning-rate calibration is forbidden")
    if schema_version == 3:
        legacy_lr_fields = {
            "peak_learning_rate",
            "learning_rate_rule",
            "learning_rate_limitation",
            "fixed_learning_rate_verification",
        }
        observed_legacy = sorted(legacy_lr_fields.intersection(training_policy))
        if observed_legacy:
            raise SpecValidationError(
                f"Obsolete fixed learning-rate fields remain: {observed_legacy}"
            )

    expected_ladders = [
        {
            "id": "A",
            "num_attention_heads": 8,
            "num_layers": 8,
            "hidden_size": 512,
            "ffn_hidden_size": 1536,
            "N_act": 92_798_976,
            "N_act_nonembed": 27_262_976,
            "D_max_tokens": 8_000_000_000,
        },
        {
            "id": "B",
            "num_attention_heads": 12,
            "num_layers": 12,
            "hidden_size": 768,
            "ffn_hidden_size": 2304,
            "N_act": 190_316_544,
            "N_act_nonembed": 92_012_544,
            "D_max_tokens": 16_000_000_000,
        },
    ]
    if _nested_value(spec, "axes.ladders") != expected_ladders:
        raise SpecValidationError("Frozen A/B-v2 ladder shapes or parameter counts drifted")

    expected_ramp = [
        {"start_tokens": 0, "end_tokens": 500_000_000, "global_batch_sequences": 32},
        {"start_tokens": 500_000_000, "end_tokens": 1_000_000_000, "global_batch_sequences": 64},
        {"start_tokens": 1_000_000_000, "end_tokens": None, "global_batch_sequences": 128},
    ]
    if _nested_value(spec, "training_policy.global_batch_ramp") != expected_ramp:
        raise SpecValidationError("Frozen Joint-MoE global-batch ramp drifted")

    invalid_if = _nested_value(spec, "run_validity_policy.invalid_if")
    required_failure_rules = {
        "target token accounting is incomplete",
        "required full-state checkpoint is missing or corrupt",
        "loss or gradient is non-finite",
        "any required per-language evaluation is missing",
        "the pre-tokenizer validation candidate pool cannot realize the exact final-token target",
        "any token is dropped by MoE routing",
        "configuration or artifact hashes differ from the frozen plan",
    }
    if not isinstance(invalid_if, list) or set(invalid_if) != required_failure_rules:
        raise SpecValidationError("Frozen A/B-v2 run-invalidity rules drifted")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpecValidationError(f"Expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_resolved_json(
    base: dict[str, Any], overlay: dict[str, Any], *, source: Path
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and set(value) == {"__delete__"}:
            if value["__delete__"] is not True:
                raise SpecValidationError(f"Invalid __delete__ directive in {source}")
            merged.pop(key, None)
        elif isinstance(value, dict) and set(value) == {"__replace__"}:
            merged[key] = value["__replace__"]
        elif isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_resolved_json(merged[key], value, source=source)
        else:
            merged[key] = value
    return merged


def _load_resolved_json(path: Path, *, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise SpecValidationError(f"JSON inheritance cycle at {path}")
    payload = _load_json(path)
    inherited = payload.pop("inherits", None)
    if inherited is None:
        return payload
    expected_parent_sha256: str | None = None
    if isinstance(inherited, dict):
        expected_parent_sha256 = inherited.get("spec_sha256")
        inherited = inherited.get("spec_path")
    if not isinstance(inherited, str) or not inherited.strip():
        raise SpecValidationError(f"Invalid JSON inheritance path in {path}")
    parent_path = Path(inherited)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent_path = parent_path.resolve()
    if expected_parent_sha256 is not None and (
        not isinstance(expected_parent_sha256, str)
        or _sha256_file(parent_path) != expected_parent_sha256
    ):
        raise SpecValidationError(
            f"JSON inheritance checksum mismatch: {parent_path}"
        )
    parent = _load_resolved_json(parent_path, seen=(*seen, path))
    return _merge_resolved_json(parent, payload, source=path)


def language_set_sha256(languages: Iterable[str]) -> str:
    """Hash an ordered language set using the definition frozen in the spec."""

    digest = hashlib.sha256()
    for language in languages:
        digest.update(str(language).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _language_list(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = _load_json(path)
    group = payload.get("0")
    if not isinstance(group, dict) or not isinstance(group.get("languages"), list):
        raise SpecValidationError(f"Language map must contain 0.languages: {path}")
    languages = group["languages"]
    if not all(isinstance(value, str) and value for value in languages):
        raise SpecValidationError(f"Language map contains an invalid language ID: {path}")
    return payload, list(languages)


def _inventory_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            subset = str(raw.get("subset", "")).strip()
            if raw.get("split") != "train" or not subset or subset.endswith("_removed"):
                continue
            try:
                documents = int(float(raw.get("documents") or 0))
                words = int(float(raw.get("words") or 0))
            except (TypeError, ValueError):
                continue
            rows.append({**raw, "subset": subset, "documents": documents, "words": words})
    return rows


def _atlas_subsets(source_lock: dict[str, Any], codes: Iterable[str]) -> list[str]:
    resolutions = source_lock.get("atlas_mapping_resolutions", {})
    subsets: list[str] = []
    for code in codes:
        subset = SHORT_CODE_TO_FINEWEB2_SUBSET.get(code)
        if subset is None:
            resolution = resolutions.get(code, {}) if isinstance(resolutions, dict) else {}
            subset = resolution.get("subset") if isinstance(resolution, dict) else None
        if not isinstance(subset, str) or not subset:
            raise SpecValidationError(
                f"ATLAS language {code!r} has no explicit source mapping; refusing silent substitution"
            )
        subsets.append(subset)
    return subsets


def _load_source_lock(root: Path, data_policy: dict[str, Any]) -> dict[str, Any]:
    source_ref = data_policy.get("source_lock")
    if not isinstance(source_ref, dict) or not isinstance(source_ref.get("path"), str):
        raise SpecValidationError("A pinned data_policy.source_lock is required")
    source_path = root / source_ref["path"]
    if _sha256_file(source_path) != source_ref.get("sha256"):
        raise SpecValidationError("Pinned source-lock checksum mismatch")
    source_lock = _load_json(source_path)
    if source_lock.get("source_lock_id") != "fineweb2_v2_1_1":
        raise SpecValidationError("Unexpected FineWeb2 source-lock ID")
    return source_lock


def _validate_audit(
    *,
    audit_path: Path,
    inventory_rows: list[dict[str, Any]],
    atlas_k50: list[str],
) -> list[str]:
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != 150:
        raise SpecValidationError(f"Availability audit has {len(audit_rows)} rows, expected 150")

    atlas_iso = {value.split("_", 1)[0] for value in atlas_k50}
    representative_by_iso: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        iso = str(row.get("code", ""))
        current = representative_by_iso.get(iso)
        candidate_key = (int(row["words"]), int(row["documents"]), str(row["subset"]))
        if current is None:
            representative_by_iso[iso] = row
            continue
        current_key = (int(current["words"]), int(current["documents"]), str(current["subset"]))
        if candidate_key > current_key:
            representative_by_iso[iso] = row

    expected = [row for iso, row in representative_by_iso.items() if iso not in atlas_iso]
    expected.sort(key=lambda row: (-int(row["documents"]), -int(row["words"]), str(row["subset"])))
    expected_subsets = [str(row["subset"]) for row in expected[:150]]
    audit_subsets = [str(row.get("subset", "")) for row in audit_rows]
    if audit_subsets != expected_subsets:
        raise SpecValidationError("Top-150 availability audit is not reproducible from the frozen inventory")
    if len({str(row.get("code", "")) for row in audit_rows}) != 150:
        raise SpecValidationError("Top-150 availability audit contains more than one script for an ISO language")
    return audit_subsets


def validate_ab_v2_spec(
    *,
    repo_root: Path | None = None,
    spec_path: Path | None = None,
    available_source_subsets: set[str] | None = None,
) -> dict[str, Any]:
    """Validate exact membership, provenance, hashes, nesting, and source resolution.

    ``available_source_subsets`` is an explicit test/acquisition override. When
    supplied, it is treated as the complete source catalog; no pinned resolution
    entries are added implicitly.
    """

    root = (repo_root or find_repo_root(Path(__file__))).resolve()
    resolved_spec_path = spec_path or root / DEFAULT_SPEC
    if not resolved_spec_path.is_absolute():
        resolved_spec_path = root / resolved_spec_path
    spec = _load_resolved_json(resolved_spec_path)
    if (
        spec.get("schema_version") not in {2, 3}
        or spec.get("study_id") != "multilingual_moe_ab_v2"
    ):
        raise SpecValidationError("Unexpected A/B-v2 schema_version or study_id")
    _validate_frozen_scientific_contract(spec)

    data_policy = spec.get("data_policy")
    if not isinstance(data_policy, dict):
        raise SpecValidationError("study_spec.data_policy must be an object")
    source_lock = _load_source_lock(root, data_policy)

    language_specs = spec.get("language_sets")
    if not isinstance(language_specs, dict):
        raise SpecValidationError("study_spec.language_sets must be an object")

    loaded: dict[int, list[str]] = {}
    for k in EXPECTED_K:
        entry = language_specs.get(f"K{k}")
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SpecValidationError(f"Missing language-set declaration K{k}")
        path = root / entry["path"]
        payload, languages = _language_list(path)
        if len(languages) != k:
            raise SpecValidationError(f"K{k} contains {len(languages)} languages")
        if any(not LANGUAGE_ID_RE.fullmatch(language) for language in languages):
            raise SpecValidationError(f"K{k} contains a malformed language-script ID")
        if len(set(languages)) != k:
            raise SpecValidationError(f"K{k} contains duplicate language-script IDs")
        iso_codes = [language.split("_", 1)[0] for language in languages]
        if len(set(iso_codes)) != k:
            raise SpecValidationError(f"K{k} contains more than one script for an ISO language")
        digest = language_set_sha256(languages)
        if digest != entry.get("sha256") or digest != payload.get("language_set_sha256"):
            raise SpecValidationError(f"K{k} ordered membership hash mismatch")
        loaded[k] = languages

    for lower, upper in zip(EXPECTED_K, EXPECTED_K[1:]):
        if not set(loaded[lower]).issubset(loaded[upper]):
            raise SpecValidationError(f"Language sets are not nested: K{lower} is not a subset of K{upper}")

    atlas_k1 = _atlas_subsets(source_lock, K1_CODES)
    atlas_k12 = _atlas_subsets(source_lock, K12_CODES)
    atlas_k50 = _atlas_subsets(source_lock, K50_CODES)
    if loaded[1] != atlas_k1 or loaded[12] != atlas_k12 or loaded[50] != atlas_k50:
        raise SpecValidationError("K1/K12/K50 do not exactly match the frozen ATLAS constants and mappings")

    selection_inventory = data_policy.get("selection_inventory")
    availability_audit = data_policy.get("availability_audit")
    if not isinstance(selection_inventory, dict) or not isinstance(availability_audit, dict):
        raise SpecValidationError("Frozen selection inventory and availability audit are required")
    inventory_path = root / str(selection_inventory.get("path", ""))
    audit_path = root / str(availability_audit.get("path", ""))
    if _sha256_file(inventory_path) != selection_inventory.get("sha256"):
        raise SpecValidationError("Frozen language-selection inventory checksum mismatch")
    if _sha256_file(audit_path) != availability_audit.get("sha256"):
        raise SpecValidationError("Frozen top-150 availability audit checksum mismatch")
    inventory_rows = _inventory_rows(inventory_path)
    audit_subsets = _validate_audit(
        audit_path=audit_path,
        inventory_rows=inventory_rows,
        atlas_k50=atlas_k50,
    )
    if loaded[200] != atlas_k50 + audit_subsets:
        raise SpecValidationError("K200 is not the exact ATLAS-K50 plus audited top-150 ordered set")

    if available_source_subsets is None:
        available = {str(row["subset"]) for row in inventory_rows}
        resolutions = source_lock.get("atlas_mapping_resolutions", {})
        for resolution in resolutions.values() if isinstance(resolutions, dict) else ():
            if isinstance(resolution, dict) and resolution.get("status") == "verified_in_pinned_release":
                subset = resolution.get("subset")
                if isinstance(subset, str) and subset:
                    available.add(subset)
    else:
        available = set(available_source_subsets)
    missing = sorted(set(loaded[200]) - available)
    if missing:
        raise SpecValidationError(
            "Selected languages have no verified source mapping: "
            + ", ".join(missing)
            + "; refusing silent substitution"
        )

    return {
        "study_id": spec["study_id"],
        "language_counts": {f"K{k}": len(loaded[k]) for k in EXPECTED_K},
        "language_set_sha256": {f"K{k}": language_set_sha256(loaded[k]) for k in EXPECTED_K},
        "source_languages_verified": len(loaded[200]),
    }


def main() -> int:
    result = validate_ab_v2_spec()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
