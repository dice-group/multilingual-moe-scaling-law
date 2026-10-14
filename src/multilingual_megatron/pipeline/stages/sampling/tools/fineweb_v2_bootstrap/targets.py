"""Fail-closed production and tokenizer acquisition target derivation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.pipeline.tokenizer_sentence_contract import (
    ACQUISITION_PLAN_TOOL_ID,
    SENTENCE_SLOT_COUNTER_ID,
    sentence_slot_gate_contract,
    tokenizer_acquisition_target_contract,
    verify_acquisition_plan,
)

from .artifacts import _load_languages, _load_object
from .contracts import SCHEMA_VERSION, BootstrapSettings

def _derive_production_settings(
    *,
    base: BootstrapSettings,
    language: str,
    sampling_plan: Mapping[str, Any],
    safety_margin: float,
) -> tuple[BootstrapSettings, dict[str, Any]]:
    if float(safety_margin) < 1.0:
        raise ValueError("production_safety_margin must be at least one")
    physical_pool = sampling_plan.get("physical_pool")
    if not isinstance(physical_pool, dict) or language not in physical_pool:
        raise ValueError(f"Sampling plan has no physical_pool entry for {language}")
    physical = physical_pool[language]
    if not isinstance(physical, dict):
        raise ValueError(f"Sampling plan physical_pool.{language} must be an object")
    calibration = physical.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(
            f"Sampling plan physical_pool.{language} lacks its tokenizer calibration"
        )
    physical_tokens = int(physical.get("physical_target_tokens", 0))
    planned_documents = int(physical.get("planned_source_documents", 0))
    calibration_bytes = int(calibration.get("calibration_utf8_bytes", 0))
    calibration_unique_tokens = int(calibration.get("calibration_tokens", 0))
    calibration_documents = int(calibration.get("calibration_documents", 0))
    calibration_rehydrated_tokens = int(
        calibration.get("calibration_rehydrated_tokens", 0)
    )
    target_validation_fraction = float(
        calibration.get("validation_fraction", base.validation_fraction)
    )
    estimated_unique_tokens = int(physical.get("estimated_unique_tokens", 0))
    estimated_validation_tokens = int(
        physical.get("estimated_validation_unique_tokens", 0)
    )
    estimated_train_unique_tokens = int(
        physical.get("estimated_train_unique_tokens", 0)
    )
    estimated_train_fraction = float(physical.get("estimated_train_fraction", 0.0))
    validation_cap_tokens = int(sampling_plan.get("validation_cap_tokens", 0))
    if (
        physical_tokens <= 0
        or planned_documents <= 0
        or calibration_bytes <= 0
        or calibration_unique_tokens <= 0
        or calibration_documents <= 0
        or calibration_rehydrated_tokens <= 0
        or estimated_unique_tokens <= 0
        or estimated_validation_tokens <= 0
        or estimated_train_unique_tokens <= 0
        or validation_cap_tokens <= 0
        or estimated_validation_tokens
        != min(
            validation_cap_tokens,
            max(
                1,
                math.floor(
                    estimated_unique_tokens * target_validation_fraction
                ),
            ),
        )
        or estimated_validation_tokens + estimated_train_unique_tokens
        != estimated_unique_tokens
        or not 0.0 < target_validation_fraction < base.validation_fraction < 1.0
        or not 0.0 < estimated_train_fraction < 1.0
        or not math.isclose(
            estimated_train_fraction,
            estimated_train_unique_tokens / float(estimated_unique_tokens),
            rel_tol=1e-15,
            abs_tol=0.0,
        )
    ):
        raise ValueError(f"Sampling plan has invalid production targets for {language}")

    # The plan's physical target applies to the train side after rehydration.
    # Acquisition happens before final validation selection, so compensate for
    # the cap-aware estimated train fraction as well as estimation error. This
    # equals 0.80 below the validation-cap threshold and rises above 0.80 once
    # the fixed 20M-token cap applies.
    train_survival = estimated_train_fraction
    derived_documents = math.ceil(
        planned_documents * float(safety_margin) / train_survival
    )
    derived_bytes = math.ceil(
        physical_tokens
        * calibration_bytes
        / calibration_rehydrated_tokens
        * float(safety_margin)
        / train_survival
    )
    # The ATLAS validation rule needs either a complete source scan or proof
    # that U_l >= cap/fraction, at which point the exact target is the cap.
    # Acquire enough unique (pre-rehydration) source mass to establish that
    # lower bound after tokenization.  A later exact audit still fails closed
    # and requests a larger immutable rebuild if the calibration estimate was
    # optimistic.
    validation_proof_unique_tokens = math.ceil(
        validation_cap_tokens / target_validation_fraction
    )
    validation_proof_documents = math.ceil(
        validation_proof_unique_tokens
        * calibration_documents
        / calibration_unique_tokens
        * float(safety_margin)
    )
    validation_proof_bytes = math.ceil(
        validation_proof_unique_tokens
        * calibration_bytes
        / calibration_unique_tokens
        * float(safety_margin)
    )
    settings = BootstrapSettings(
        seed=base.seed,
        validation_fraction=base.validation_fraction,
        target_unique_documents=max(
            base.target_unique_documents,
            derived_documents,
            validation_proof_documents,
        ),
        target_utf8_bytes=max(
            base.target_utf8_bytes,
            derived_bytes,
            validation_proof_bytes,
        ),
        calibration_documents=base.calibration_documents,
    )
    derivation = {
        "sampling_plan_sha256": sampling_plan["plan_sha256"],
        "physical_target_tokens": physical_tokens,
        "planned_source_documents": planned_documents,
        "calibration_rehydrated_tokens": calibration_rehydrated_tokens,
        "calibration_utf8_bytes": calibration_bytes,
        "validation_candidate_fraction": float(base.validation_fraction),
        "validation_target_fraction": target_validation_fraction,
        "estimated_validation_target_tokens": estimated_validation_tokens,
        "estimated_train_survival_fraction": train_survival,
        "safety_margin": float(safety_margin),
        "derived_target_unique_documents": derived_documents,
        "derived_target_utf8_bytes": derived_bytes,
        "validation_exactness_gate": {
            "rule": "complete_source_scan_or_observed_U_at_least_cap_over_fraction",
            "validation_cap_tokens": validation_cap_tokens,
            "validation_fraction": target_validation_fraction,
            "required_observed_unique_tokens": validation_proof_unique_tokens,
            "derived_target_unique_documents": validation_proof_documents,
            "derived_target_utf8_bytes": validation_proof_bytes,
        },
        "effective_settings": asdict(settings),
    }
    return settings, derivation


def _load_tokenizer_acquisition_plan(path: Path) -> dict[str, Any]:
    payload = _load_object(path.resolve())
    expected = payload.get("plan_sha256")
    candidate = dict(payload)
    candidate.pop("plan_sha256", None)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("tool_id") != ACQUISITION_PLAN_TOOL_ID
        or payload.get("status") != "complete"
        or not isinstance(expected, str)
        or expected != canonical_json_sha256(candidate)
        or not isinstance(payload.get("languages"), dict)
        or not verify_acquisition_plan(payload)
    ):
        raise ValueError("Tokenizer sentence acquisition plan hash/schema is invalid")
    return payload


def _derive_tokenizer_acquisition_settings(
    *,
    base: BootstrapSettings,
    language: str,
    acquisition_plan: Mapping[str, Any],
    acquisition_plan_path: Path,
) -> tuple[BootstrapSettings, dict[str, Any]]:
    if acquisition_plan.get("seed") != base.seed:
        raise ValueError(
            "Tokenizer acquisition plan seed differs from the bootstrap data seed"
        )
    plan_file_sha256 = sha256_file(acquisition_plan_path.resolve())
    try:
        marker = tokenizer_acquisition_target_contract(
            plan=acquisition_plan,
            plan_file_sha256=plan_file_sha256,
            language=language,
        )
        target_documents = int(marker["target_unique_source_documents"])
        target_bytes = int(marker["target_utf8_bytes"])
        required_slots = int(marker["required_rehydrated_sentence_slots"])
        safety_margin = float(marker["safety_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Tokenizer acquisition target is invalid for {language}") from exc
    if (
        target_documents <= 0
        or target_bytes <= 0
        or required_slots <= 0
        or safety_margin < 1.0
    ):
        raise ValueError(f"Tokenizer acquisition target is invalid for {language}")
    settings = BootstrapSettings(
        seed=base.seed,
        validation_fraction=base.validation_fraction,
        target_unique_documents=max(base.target_unique_documents, target_documents),
        target_utf8_bytes=max(base.target_utf8_bytes, target_bytes),
        calibration_documents=base.calibration_documents,
    )
    return settings, marker


def _tokenizer_sentence_slot_gate(
    target: Mapping[str, Any] | None,
) -> tuple[int, int, dict[str, Any]] | None:
    """Validate and expose the frozen exact train-slot acquisition gate."""

    if target is None:
        return None
    gate = target.get("sentence_slot_gate")
    if not isinstance(gate, dict):
        raise ValueError("Tokenizer acquisition target lacks its sentence-slot gate")
    splitter = gate.get("splitter")
    raw_max_chars = splitter.get("max_sentence_chars") if isinstance(splitter, dict) else None
    if (
        not isinstance(raw_max_chars, int)
        or isinstance(raw_max_chars, bool)
        or raw_max_chars <= 0
    ):
        raise ValueError("Tokenizer acquisition sentence splitter is invalid")
    expected_gate = sentence_slot_gate_contract(max_chars=raw_max_chars)
    if gate != expected_gate or gate.get("counter_id") != SENTENCE_SLOT_COUNTER_ID:
        raise ValueError("Tokenizer acquisition sentence-slot contract drifted")
    required_slots = target.get("required_rehydrated_sentence_slots")
    if (
        not isinstance(required_slots, int)
        or isinstance(required_slots, bool)
        or required_slots <= 0
    ):
        raise ValueError("Tokenizer acquisition sentence-slot quota is invalid")
    return int(required_slots), int(raw_max_chars), expected_gate


def _validate_tokenizer_plan_language_map(
    *, acquisition_plan: Mapping[str, Any], language_map_path: Path
) -> None:
    languages = _load_languages(language_map_path.resolve())
    raw_inputs = acquisition_plan.get("inputs")
    raw_map = raw_inputs.get("language_map") if isinstance(raw_inputs, dict) else None
    if (
        acquisition_plan.get("ordered_languages") != list(languages)
        or not isinstance(raw_map, dict)
        or raw_map.get("sha256") != sha256_file(language_map_path.resolve())
    ):
        raise ValueError(
            "Tokenizer acquisition plan language map differs from the bootstrap map"
        )
