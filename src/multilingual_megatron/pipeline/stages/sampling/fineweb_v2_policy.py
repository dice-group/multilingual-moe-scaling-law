"""Pure policy helpers for the frozen FineWeb A/B-v2 corpus release.

This module intentionally contains no network or Slurm code.  It is the single
place where the study turns a tokenizer calibration into language probabilities,
token quotas, validation targets, and deterministic document assignments.
Keeping those calculations pure makes the expensive corpus jobs auditable before
they are submitted.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.hashing import (
    canonical_json_sha256,
    sha256_file,
)


POLICY_ID = "fineweb_ab_v2_alpha03_rehydrated"
DEFAULT_ALPHA = 0.3
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_VALIDATION_CAP_TOKENS = 20_000_000


def stable_document_id(
    *,
    source_dataset: str,
    source_revision: str,
    subset: str,
    source_id: str | None,
    url: str | None,
    text: str,
) -> str:
    """Return a stable ID without trusting any one optional source field."""

    payload = {
        "source_dataset": str(source_dataset),
        "source_revision": str(source_revision),
        "subset": str(subset),
        "source_id": str(source_id or ""),
        "url": str(url or ""),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return canonical_json_sha256(payload)


def deterministic_unit_interval(*, document_id: str, seed: int, namespace: str) -> float:
    digest = hashlib.sha256(
        f"{int(seed)}\0{namespace}\0{document_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_validation_reserve(
    *,
    document_id: str,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> bool:
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    return deterministic_unit_interval(
        document_id=document_id,
        seed=seed,
        namespace="validation-reserve-v1",
    ) < float(validation_fraction)


def rehydration_multiplier(cluster_size: int, thresholds: Mapping[int | str, int]) -> int:
    """Apply FineWeb2's piecewise-constant per-language rehydration table."""

    if int(cluster_size) < 1:
        raise ValueError("minhash_cluster_size must be at least one")
    normalized = sorted((int(limit), int(weight)) for limit, weight in thresholds.items())
    if not normalized or normalized[0][0] != 1:
        raise ValueError("rehydration thresholds must start at cluster size one")
    if any(limit < 1 or weight < 1 for limit, weight in normalized):
        raise ValueError("rehydration limits and weights must be positive")
    selected = normalized[0][1]
    for limit, weight in normalized:
        if int(cluster_size) < limit:
            break
        selected = weight
    return selected


@dataclass(frozen=True, slots=True)
class LanguageCalibration:
    language: str
    available_utf8_bytes: int
    available_documents: int
    calibration_utf8_bytes: int
    calibration_tokens: int
    calibration_documents: int
    calibration_rehydrated_tokens: int
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    exact_train_rehydrated_tokens: int | None = None
    exact_train_rehydrated_validation_cap_tokens: int | None = None

    def validate(self) -> None:
        if not self.language:
            raise ValueError("language must be set")
        for field_name in (
            "available_utf8_bytes",
            "available_documents",
            "calibration_utf8_bytes",
            "calibration_tokens",
            "calibration_documents",
            "calibration_rehydrated_tokens",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{self.language}: {field_name} must be positive")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError(f"{self.language}: invalid validation_fraction")
        exact_values = (
            self.exact_train_rehydrated_tokens,
            self.exact_train_rehydrated_validation_cap_tokens,
        )
        if (exact_values[0] is None) != (exact_values[1] is None) or any(
            value is not None and int(value) <= 0 for value in exact_values
        ):
            raise ValueError(
                f"{self.language}: exact train availability override is incomplete"
            )

    @property
    def tokens_per_byte(self) -> float:
        return self.calibration_tokens / self.calibration_utf8_bytes

    @property
    def estimated_unique_tokens(self) -> int:
        return max(1, round(self.available_utf8_bytes * self.tokens_per_byte))

    @property
    def estimated_train_unique_tokens(self) -> int:
        return self.estimated_train_unique_tokens_for_cap(
            DEFAULT_VALIDATION_CAP_TOKENS
        )

    @property
    def estimated_validation_unique_tokens(self) -> int:
        return self.estimated_validation_unique_tokens_for_cap(
            DEFAULT_VALIDATION_CAP_TOKENS
        )

    @property
    def estimated_train_documents(self) -> int:
        return self.estimated_train_documents_for_cap(DEFAULT_VALIDATION_CAP_TOKENS)

    def estimated_validation_unique_tokens_for_cap(self, cap_tokens: int) -> int:
        return validation_target_tokens(
            self.estimated_unique_tokens,
            fraction=self.validation_fraction,
            cap_tokens=cap_tokens,
        )

    def estimated_train_unique_tokens_for_cap(self, cap_tokens: int) -> int:
        validation_tokens = self.estimated_validation_unique_tokens_for_cap(cap_tokens)
        train_tokens = self.estimated_unique_tokens - validation_tokens
        if train_tokens <= 0:
            raise ValueError(
                f"{self.language}: validation estimate leaves no training tokens"
            )
        return train_tokens

    def estimated_train_fraction_for_cap(self, cap_tokens: int) -> float:
        return self.estimated_train_unique_tokens_for_cap(cap_tokens) / float(
            self.estimated_unique_tokens
        )

    def estimated_train_documents_for_cap(self, cap_tokens: int) -> int:
        return max(
            1,
            math.floor(
                self.available_documents
                * self.estimated_train_fraction_for_cap(cap_tokens)
            ),
        )

    @property
    def rehydration_token_multiplier(self) -> float:
        """Mean released exposure multiplier on the train-only calibration sample."""

        return self.calibration_rehydrated_tokens / self.calibration_tokens

    @property
    def estimated_train_rehydrated_tokens(self) -> int:
        """Post-split availability after FineWeb2 rehydration.

        This is the availability ``n_l`` used by the frozen alpha-smoothed
        language mixture.  Keeping it separate from unique-token availability
        is important: validation and unigram references use unique originals,
        whereas training exposure sees the released rehydration replicas.
        """

        return self.estimated_train_rehydrated_tokens_for_cap(
            DEFAULT_VALIDATION_CAP_TOKENS
        )

    def estimated_train_rehydrated_tokens_for_cap(self, cap_tokens: int) -> int:
        if (
            self.exact_train_rehydrated_tokens is not None
            and int(cap_tokens) == int(self.exact_train_rehydrated_validation_cap_tokens)
        ):
            return int(self.exact_train_rehydrated_tokens)
        return max(
            1,
            round(
                self.estimated_train_unique_tokens_for_cap(cap_tokens)
                * self.rehydration_token_multiplier
            ),
        )

    @property
    def rehydrated_tokens_per_source_document(self) -> float:
        return self.calibration_rehydrated_tokens / self.calibration_documents


def alpha_probabilities(
    availability_tokens: Mapping[str, int],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if not availability_tokens:
        raise ValueError("availability_tokens must not be empty")
    weights: dict[str, float] = {}
    for language, raw_tokens in availability_tokens.items():
        tokens = int(raw_tokens)
        if not language or tokens <= 0:
            raise ValueError("every language must have positive token availability")
        weights[str(language)] = math.pow(float(tokens), float(alpha))
    denominator = math.fsum(weights.values())
    return {language: weights[language] / denominator for language in sorted(weights)}


def largest_remainder_quotas(total: int, probabilities: Mapping[str, float]) -> dict[str, int]:
    """Allocate an exact integer total with deterministic largest remainders."""

    if int(total) <= 0:
        raise ValueError("total must be positive")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    probability_sum = math.fsum(float(value) for value in probabilities.values())
    if probability_sum <= 0.0:
        raise ValueError("probabilities must have positive mass")
    raw = {
        str(language): int(total) * float(probability) / probability_sum
        for language, probability in probabilities.items()
    }
    quotas = {language: math.floor(value) for language, value in raw.items()}
    missing = int(total) - sum(quotas.values())
    order = sorted(raw, key=lambda language: (-(raw[language] - quotas[language]), language))
    for language in order[:missing]:
        quotas[language] += 1
    if sum(quotas.values()) != int(total):
        raise AssertionError("largest-remainder allocation did not preserve the total")
    return quotas


def validation_target_tokens(
    estimated_unique_tokens: int,
    *,
    fraction: float = DEFAULT_VALIDATION_FRACTION,
    cap_tokens: int = DEFAULT_VALIDATION_CAP_TOKENS,
) -> int:
    if int(estimated_unique_tokens) <= 0:
        raise ValueError("estimated_unique_tokens must be positive")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be strictly between zero and one")
    if int(cap_tokens) <= 0:
        raise ValueError("cap_tokens must be positive")
    return min(int(cap_tokens), max(1, math.floor(int(estimated_unique_tokens) * float(fraction))))


def _language_set_hash(languages: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{language}\n" for language in languages).encode("utf-8")).hexdigest()


def build_nested_sampling_plan(
    *,
    calibrations: Mapping[str, LanguageCalibration],
    language_sets: Mapping[int, Sequence[str]],
    train_tokens_per_set: int,
    alpha: float = DEFAULT_ALPHA,
    validation_cap_tokens: int = DEFAULT_VALIDATION_CAP_TOKENS,
    sampling_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build K-specific alpha mixtures and one shared physical document pool.

    The physical pool contains enough text for the largest quota assigned to a
    language by any nested K tier.  Training probabilities remain K-specific,
    so a low-resource language may be cycled by Megatron without physically
    cloning it beyond the declared FineWeb2 rehydration policy.
    """

    if int(train_tokens_per_set) <= 0:
        raise ValueError("train_tokens_per_set must be positive")
    if int(validation_cap_tokens) <= 0:
        raise ValueError("validation_cap_tokens must be positive")
    if not isinstance(sampling_seed, int) or isinstance(sampling_seed, bool):
        raise ValueError("sampling_seed must be an integer")
    if not language_sets:
        raise ValueError("language_sets must not be empty")
    normalized_sets = {int(k): tuple(values) for k, values in language_sets.items()}
    ordered_k = sorted(normalized_sets)
    for lower, upper in zip(ordered_k, ordered_k[1:]):
        if not set(normalized_sets[lower]).issubset(normalized_sets[upper]):
            raise ValueError(f"language sets are not nested: K{lower} is not a subset of K{upper}")
    if any(len(normalized_sets[k]) != k for k in ordered_k):
        raise ValueError("each K tier must contain exactly K languages")

    for calibration in calibrations.values():
        calibration.validate()
    required_languages = set().union(*(set(values) for values in normalized_sets.values()))
    missing = sorted(required_languages - set(calibrations))
    if missing:
        raise ValueError("missing tokenizer calibrations: " + ", ".join(missing))

    estimated_unique = {
        language: calibrations[language].estimated_unique_tokens
        for language in required_languages
    }
    estimated_validation = {
        language: calibrations[language].estimated_validation_unique_tokens_for_cap(
            validation_cap_tokens
        )
        for language in required_languages
    }
    estimated_train_unique = {
        language: calibrations[language].estimated_train_unique_tokens_for_cap(
            validation_cap_tokens
        )
        for language in required_languages
    }
    estimated_train_rehydrated = {
        language: calibrations[language].estimated_train_rehydrated_tokens_for_cap(
            validation_cap_tokens
        )
        for language in required_languages
    }
    estimated_train_documents = {
        language: calibrations[language].estimated_train_documents_for_cap(
            validation_cap_tokens
        )
        for language in required_languages
    }

    tiers: dict[str, Any] = {}
    maximum_quota = {language: 0 for language in required_languages}
    for k in ordered_k:
        languages = normalized_sets[k]
        # FineWeb2 rehydration is applied before the language-level alpha
        # mixture, so n_l is the post-split, post-rehydration token mass.
        availability = {
            language: estimated_train_rehydrated[language]
            for language in languages
        }
        probabilities = alpha_probabilities(availability, alpha=alpha)
        quotas = largest_remainder_quotas(int(train_tokens_per_set), probabilities)
        for language, quota in quotas.items():
            maximum_quota[language] = max(maximum_quota[language], quota)
        tiers[f"K{k}"] = {
            "languages": list(languages),
            "language_set_sha256": _language_set_hash(languages),
            "alpha": float(alpha),
            "availability_basis": "estimated_train_tokens_after_fineweb2_rehydration",
            "availability_tokens": availability,
            "target_train_tokens": int(train_tokens_per_set),
            "probabilities": probabilities,
            "target_exposure_tokens": quotas,
        }

    physical: dict[str, Any] = {}
    for language in sorted(required_languages):
        calibration = calibrations[language]
        requested = int(maximum_quota[language])
        available_unique = int(estimated_train_unique[language])
        available_rehydrated = int(estimated_train_rehydrated[language])
        physical_target = min(requested, available_rehydrated)
        source_documents = min(
            int(estimated_train_documents[language]),
            max(1, math.ceil(physical_target / calibration.rehydrated_tokens_per_source_document)),
        )
        physical[language] = {
            "requested_max_tier_tokens": requested,
            "estimated_unique_tokens": int(estimated_unique[language]),
            "estimated_validation_unique_tokens": int(estimated_validation[language]),
            "estimated_train_unique_tokens": available_unique,
            "estimated_train_rehydrated_tokens": available_rehydrated,
            "estimated_train_fraction": (
                available_unique / float(estimated_unique[language])
            ),
            "estimated_train_documents": int(estimated_train_documents[language]),
            "rehydration_token_multiplier": calibration.rehydration_token_multiplier,
            "physical_target_tokens": physical_target,
            "planned_source_documents": source_documents,
            "training_repeat_required": requested > physical_target,
            "estimated_passes_at_max_quota": requested / physical_target,
            "validation_target_tokens": int(estimated_validation[language]),
            "calibration": asdict(calibration),
        }

    payload = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "sampling_seed": int(sampling_seed),
        "alpha": float(alpha),
        "train_tokens_per_set": int(train_tokens_per_set),
        "validation_cap_tokens": int(validation_cap_tokens),
        "availability_contract": {
            "validation_rule": "min(validation_cap_tokens, floor(validation_fraction * estimated_unique_tokens))",
            "train_unique_rule": "estimated_unique_tokens - validation_target_tokens",
            "candidate_reserve_affects_availability": False,
        },
        "tiers": tiers,
        "physical_pool": physical,
    }
    payload["plan_sha256"] = canonical_json_sha256(payload)
    return payload


def verify_plan_hash(payload: Mapping[str, Any]) -> bool:
    candidate = dict(payload)
    expected = candidate.pop("plan_sha256", None)
    return isinstance(expected, str) and expected == canonical_json_sha256(candidate)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_SEED",
    "DEFAULT_VALIDATION_CAP_TOKENS",
    "DEFAULT_VALIDATION_FRACTION",
    "LanguageCalibration",
    "POLICY_ID",
    "alpha_probabilities",
    "assign_validation_reserve",
    "build_nested_sampling_plan",
    "canonical_json_sha256",
    "deterministic_unit_interval",
    "largest_remainder_quotas",
    "rehydration_multiplier",
    "sha256_file",
    "stable_document_id",
    "validation_target_tokens",
    "verify_plan_hash",
]
