"""Deterministic per-language validation metrics for the scaling-law study.

The training/evaluation backend is responsible for producing an *exact sum* of
token negative log-likelihoods and the corresponding target token IDs.  This
module deliberately does not consume batch means: averaging batch means would
silently bias languages whose final batch is shorter.

The adjusted metric follows the predeclared ATLAS/Tao form

    -mean(log(p_model(token) / p_unigram(token))).

The language unigram is built only from training tokens.  Every non-special
vocabulary ID that was absent from those tokens receives pseudo-count one.
Special-token targets are rejected because their NLL cannot be removed from an
already-summed value inside this module; callers must mask them before summing.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


UNIGRAM_SCHEMA = "multilingual_validation_unigram/v1"
LANGUAGE_LOSS_SCHEMA = "multilingual_validation_language_loss/v1"
VALIDATION_REPORT_SCHEMA = "multilingual_validation_report/v1"
PSEUDO_COUNT = 1
ENGLISH_LANGUAGE = "eng_Latn"
FIXED_COHORT_NAMES = ("english", "k12", "k50")
LEGACY_FIXED_COHORT_NAMES = ("english", "k16", "k50")


class MissingRequiredLanguageError(ValueError):
    """Raised when a required per-language validation result is absent."""

    def __init__(self, missing_languages: Iterable[str]) -> None:
        self.missing_languages = tuple(sorted(set(missing_languages)))
        joined = ", ".join(self.missing_languages)
        super().__init__(f"missing required validation languages: {joined}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _language(value: str, *, field_name: str = "language") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _integer(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class UnigramReference:
    """Frozen smoothed unigram counts for one language.

    ``counts`` contains positive counts only for observed, non-special IDs.
    The normalization denominator is

    ``sum(observed counts) + number of unseen non-special vocabulary IDs``.

    This is equivalent to replacing every zero count by the fixed pseudo-count
    one and normalizing over the full non-special vocabulary.
    """

    language: str
    vocab_size: int
    special_token_ids: tuple[int, ...]
    counts: Mapping[int, int] = field(repr=False, compare=False)
    observed_token_count: int
    smoothed_token_count: int
    observed_type_count: int
    eligible_type_count: int
    counts_sha256: str

    @classmethod
    def from_counts(
        cls,
        *,
        language: str,
        vocab_size: int,
        counts: Mapping[int, int],
        special_token_ids: Iterable[int] = (),
    ) -> "UnigramReference":
        language = _language(language)
        vocab_size = _integer(vocab_size, field_name="vocab_size", minimum=1)
        if not isinstance(counts, Mapping):
            raise ValueError("counts must be a mapping from token ID to count")

        special_list = list(special_token_ids)
        special_set: set[int] = set()
        for token_id in special_list:
            parsed = _integer(token_id, field_name="special token ID")
            if parsed >= vocab_size:
                raise ValueError(f"special token ID {parsed} is outside vocabulary")
            special_set.add(parsed)
        if len(special_set) != len(special_list):
            raise ValueError("special_token_ids must not contain duplicates")
        if len(special_set) >= vocab_size:
            raise ValueError("the vocabulary must contain at least one non-special ID")

        normalized: dict[int, int] = {}
        for token_id, count in counts.items():
            parsed_id = _integer(token_id, field_name="unigram token ID")
            parsed_count = _integer(count, field_name=f"count for token ID {parsed_id}", minimum=1)
            if parsed_id >= vocab_size:
                raise ValueError(f"unigram token ID {parsed_id} is outside vocabulary")
            if parsed_id in special_set:
                raise ValueError(f"special token ID {parsed_id} must be excluded from unigram counts")
            normalized[parsed_id] = parsed_count

        sorted_counts = tuple(sorted(normalized.items()))
        observed_token_count = sum(count for _, count in sorted_counts)
        eligible_type_count = vocab_size - len(special_set)
        observed_type_count = len(sorted_counts)
        smoothed_token_count = observed_token_count + eligible_type_count - observed_type_count
        if smoothed_token_count <= 0:  # Defensive; eligible_type_count is positive.
            raise ValueError("smoothed unigram denominator must be positive")

        hash_payload = {
            "schema": UNIGRAM_SCHEMA,
            "language": language,
            "vocab_size": vocab_size,
            "special_token_ids": sorted(special_set),
            "pseudo_count_for_unseen_ids": PSEUDO_COUNT,
            "counts": [[token_id, count] for token_id, count in sorted_counts],
        }
        return cls(
            language=language,
            vocab_size=vocab_size,
            special_token_ids=tuple(sorted(special_set)),
            counts=MappingProxyType(dict(sorted_counts)),
            observed_token_count=observed_token_count,
            smoothed_token_count=smoothed_token_count,
            observed_type_count=observed_type_count,
            eligible_type_count=eligible_type_count,
            counts_sha256=_sha256(hash_payload),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema": UNIGRAM_SCHEMA,
            "language": self.language,
            "vocab_size": self.vocab_size,
            "special_token_ids": list(self.special_token_ids),
            "pseudo_count_for_unseen_ids": PSEUDO_COUNT,
            "observed_token_count": self.observed_token_count,
            "smoothed_token_count": self.smoothed_token_count,
            "observed_type_count": self.observed_type_count,
            "eligible_type_count": self.eligible_type_count,
            "counts_sha256": self.counts_sha256,
        }


@dataclass(frozen=True, slots=True)
class LanguageLoss:
    """Exact per-language sufficient statistics and derived losses."""

    language: str
    token_count: int
    nll_sum: float
    unigram_log_probability_sum: float
    adjusted_nll_sum: float
    raw_nll: float
    atlas_tao_adjusted_loss: float
    unseen_target_token_count: int
    target_token_ids_sha256: str
    unigram_counts_sha256: str
    input_sha256: str
    vocab_size: int
    special_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LANGUAGE_LOSS_SCHEMA,
            "language": self.language,
            "token_count": self.token_count,
            "nll_sum": self.nll_sum,
            "unigram_log_probability_sum": self.unigram_log_probability_sum,
            "adjusted_nll_sum": self.adjusted_nll_sum,
            "raw_nll": self.raw_nll,
            "atlas_tao_unigram_adjusted_loss": self.atlas_tao_adjusted_loss,
            "unseen_target_token_count": self.unseen_target_token_count,
            "target_token_ids_sha256": self.target_token_ids_sha256,
            "unigram_counts_sha256": self.unigram_counts_sha256,
            "input_sha256": self.input_sha256,
            "vocab_size": self.vocab_size,
            "special_token_ids": list(self.special_token_ids),
        }


def compute_language_losses(
    *,
    language: str,
    nll_sum: float,
    target_token_ids: Iterable[int],
    unigram: UnigramReference,
) -> LanguageLoss:
    """Compute raw and unigram-adjusted losses from exact sufficient statistics.

    ``nll_sum`` must be the sum over exactly the targets in
    ``target_token_ids`` after the caller has removed padding, masked targets,
    and all configured special-token targets.
    """

    language = _language(language)
    if language != unigram.language:
        raise ValueError(
            f"language {language!r} does not match unigram language {unigram.language!r}"
        )
    nll_sum = _finite_float(nll_sum, field_name="nll_sum")
    if nll_sum < 0:
        raise ValueError("nll_sum must be non-negative")

    special_set = set(unigram.special_token_ids)
    frequencies: dict[int, int] = {}
    token_hasher = hashlib.sha256()
    token_hasher.update(b"multilingual_validation_target_ids/v1\0")
    token_count = 0
    for token_id in target_token_ids:
        parsed = _integer(token_id, field_name="target token ID")
        if parsed >= unigram.vocab_size:
            raise ValueError(f"target token ID {parsed} is outside vocabulary")
        if parsed in special_set:
            raise ValueError(
                f"target token ID {parsed} is special; mask it before summing NLL"
            )
        frequencies[parsed] = frequencies.get(parsed, 0) + 1
        token_hasher.update(struct.pack(">Q", parsed))
        token_count += 1
    if token_count == 0:
        raise ValueError("target_token_ids must contain at least one evaluated token")

    return _finalize_language_loss(
        language=language,
        nll_sum=nll_sum,
        frequencies=frequencies,
        token_count=token_count,
        target_hash=token_hasher.hexdigest(),
        unigram=unigram,
    )


def compute_language_losses_from_array(
    *,
    language: str,
    nll_sum: float,
    target_token_ids: np.ndarray,
    unigram: UnigramReference,
) -> LanguageLoss:
    """Vectorized equivalent of :func:`compute_language_losses`.

    The evaluator already has language targets in indexed NumPy arrays.  This
    path retains the exact per-token big-endian hash and the same sorted
    frequency reduction, while avoiding a Python loop over every validation
    token.  Callers must still remove special targets before invoking it.
    """

    language = _language(language)
    if language != unigram.language:
        raise ValueError(
            f"language {language!r} does not match unigram language {unigram.language!r}"
        )
    nll_sum = _finite_float(nll_sum, field_name="nll_sum")
    if nll_sum < 0:
        raise ValueError("nll_sum must be non-negative")
    target_ids = np.asarray(target_token_ids)
    if target_ids.ndim != 1 or not np.issubdtype(target_ids.dtype, np.integer):
        raise ValueError("target_token_ids must be a one-dimensional integer array")
    if target_ids.size == 0:
        raise ValueError("target_token_ids must contain at least one evaluated token")
    if int(target_ids.min()) < 0 or int(target_ids.max()) >= unigram.vocab_size:
        raise ValueError("target token ID is outside vocabulary")
    if unigram.special_token_ids and np.isin(
        target_ids, np.asarray(unigram.special_token_ids, dtype=target_ids.dtype)
    ).any():
        raise ValueError("target token ID is special; mask it before summing NLL")

    unique_ids, counts = np.unique(target_ids, return_counts=True)
    frequencies = {
        int(token_id): int(frequency)
        for token_id, frequency in zip(unique_ids.tolist(), counts.tolist(), strict=True)
    }
    token_hasher = hashlib.sha256()
    token_hasher.update(b"multilingual_validation_target_ids/v1\0")
    token_hasher.update(np.asarray(target_ids, dtype=">u8").tobytes(order="C"))
    return _finalize_language_loss(
        language=language,
        nll_sum=nll_sum,
        frequencies=frequencies,
        token_count=int(target_ids.size),
        target_hash=token_hasher.hexdigest(),
        unigram=unigram,
    )


def _finalize_language_loss(
    *,
    language: str,
    nll_sum: float,
    frequencies: Mapping[int, int],
    token_count: int,
    target_hash: str,
    unigram: UnigramReference,
) -> LanguageLoss:
    """Derive the frozen raw and adjusted metrics from verified statistics."""

    if token_count <= 0:
        raise ValueError("target_token_ids must contain at least one evaluated token")
    log_denominator = math.log(unigram.smoothed_token_count)
    log_probability_terms: list[float] = []
    unseen_target_token_count = 0
    for token_id in sorted(frequencies):
        frequency = frequencies[token_id]
        observed_count = unigram.counts.get(token_id)
        effective_count = observed_count if observed_count is not None else PSEUDO_COUNT
        if observed_count is None:
            unseen_target_token_count += frequency
        log_probability_terms.append(frequency * (math.log(effective_count) - log_denominator))
    unigram_log_probability_sum = math.fsum(log_probability_terms)
    adjusted_nll_sum = math.fsum((nll_sum, unigram_log_probability_sum))
    raw_nll = nll_sum / token_count
    adjusted_loss = adjusted_nll_sum / token_count
    input_hash = _sha256(
        {
            "schema": LANGUAGE_LOSS_SCHEMA,
            "language": language,
            "nll_sum_hex": nll_sum.hex(),
            "token_count": token_count,
            "target_token_ids_sha256": target_hash,
            "unigram_counts_sha256": unigram.counts_sha256,
        }
    )
    return LanguageLoss(
        language=language,
        token_count=token_count,
        nll_sum=nll_sum,
        unigram_log_probability_sum=unigram_log_probability_sum,
        adjusted_nll_sum=adjusted_nll_sum,
        raw_nll=raw_nll,
        atlas_tao_adjusted_loss=adjusted_loss,
        unseen_target_token_count=unseen_target_token_count,
        target_token_ids_sha256=target_hash,
        unigram_counts_sha256=unigram.counts_sha256,
        input_sha256=input_hash,
        vocab_size=unigram.vocab_size,
        special_token_ids=unigram.special_token_ids,
    )


def _unique_languages(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field_name} must be an iterable of language IDs, not a string")
    normalized = [_language(value, field_name=field_name) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def validate_fixed_cohorts(cohorts: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    """Validate the production English/K12/K50 or historical K16 cohorts."""

    if not isinstance(cohorts, Mapping):
        raise ValueError("cohorts must be a mapping")
    names = tuple(cohorts)
    if set(names) == set(FIXED_COHORT_NAMES):
        cohort_names = FIXED_COHORT_NAMES
        anchor_name = "k12"
        anchor_size = 12
    elif set(names) == set(LEGACY_FIXED_COHORT_NAMES):
        cohort_names = LEGACY_FIXED_COHORT_NAMES
        anchor_name = "k16"
        anchor_size = 16
    else:
        raise ValueError(
            "fixed cohorts must contain exactly english/k12/k50 "
            "(or historical english/k16/k50)"
        )

    canonical = {
        name: _unique_languages(cohorts[name], field_name=f"cohorts[{name!r}]")
        for name in cohort_names
    }
    expected_sizes = {"english": 1, anchor_name: anchor_size, "k50": 50}
    for name, expected_size in expected_sizes.items():
        if len(canonical[name]) != expected_size:
            raise ValueError(f"cohort {name!r} must contain exactly {expected_size} languages")
    if canonical["english"] != (ENGLISH_LANGUAGE,):
        raise ValueError(f"English cohort must contain only {ENGLISH_LANGUAGE}")
    if not set(canonical["english"]).issubset(canonical[anchor_name]):
        raise ValueError(f"English cohort must be a subset of {anchor_name.upper()}")
    if not set(canonical[anchor_name]).issubset(canonical["k50"]):
        raise ValueError(f"{anchor_name.upper()} cohort must be a subset of K50")
    return canonical


def _aggregate(records: Sequence[LanguageLoss]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: record.language)
    if not ordered:
        return {
            "status": "empty",
            "language_count": 0,
            "token_count": 0,
            "languages": [],
            "equal_language_atlas_tao_adjusted_loss": None,
            "equal_language_raw_nll": None,
            "token_micro_atlas_tao_adjusted_loss": None,
            "token_micro_raw_nll": None,
        }
    language_count = len(ordered)
    token_count = sum(record.token_count for record in ordered)
    return {
        "status": "complete",
        "language_count": language_count,
        "token_count": token_count,
        "languages": [record.language for record in ordered],
        "equal_language_atlas_tao_adjusted_loss": math.fsum(
            record.atlas_tao_adjusted_loss for record in ordered
        )
        / language_count,
        "equal_language_raw_nll": math.fsum(record.raw_nll for record in ordered)
        / language_count,
        "token_micro_atlas_tao_adjusted_loss": math.fsum(
            record.adjusted_nll_sum for record in ordered
        )
        / token_count,
        "token_micro_raw_nll": math.fsum(record.nll_sum for record in ordered) / token_count,
    }


def build_validation_report(
    *,
    run_id: str,
    observations: Iterable[LanguageLoss],
    trained_languages: Iterable[str],
    required_languages: Iterable[str],
    cohorts: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """Build the deterministic study report and enforce language coverage.

    Every trained language must be required and observed.  Required but
    untrained languages are explicitly labelled ``zero_shot`` and are excluded
    from the primary aggregate.  This makes K=1 mean "trained English" rather
    than a mixture of English and zero-shot transfer.
    """

    run_id = _language(run_id, field_name="run_id")
    trained = _unique_languages(trained_languages, field_name="trained_languages")
    required = _unique_languages(required_languages, field_name="required_languages")
    trained_set = set(trained)
    required_set = set(required)
    if not trained_set:
        raise ValueError("trained_languages must not be empty")
    not_required = sorted(trained_set - required_set)
    if not_required:
        raise ValueError(
            "all trained languages must be required for validation: " + ", ".join(not_required)
        )
    fixed_cohorts = validate_fixed_cohorts(cohorts)

    by_language: dict[str, LanguageLoss] = {}
    for observation in observations:
        if not isinstance(observation, LanguageLoss):
            raise ValueError("observations must contain LanguageLoss values")
        if observation.language in by_language:
            raise ValueError(f"duplicate observation for language {observation.language}")
        by_language[observation.language] = observation

    missing = required_set - set(by_language)
    if missing:
        raise MissingRequiredLanguageError(missing)

    # The study uses one shared tokenizer.  Vocabulary/special-ID drift between
    # languages would make the adjusted losses incomparable, so fail closed.
    vocab_specs = {
        (observation.vocab_size, observation.special_token_ids)
        for observation in by_language.values()
    }
    if len(vocab_specs) > 1:
        raise ValueError("all language observations must use the same vocabulary and special IDs")

    trained_records = [by_language[language] for language in trained]
    zero_shot_languages = tuple(sorted(set(by_language) - trained_set))
    zero_shot_records = [by_language[language] for language in zero_shot_languages]
    language_payload: list[dict[str, Any]] = []
    for language in sorted(by_language):
        payload = by_language[language].to_dict()
        payload["role"] = "trained" if language in trained_set else "zero_shot"
        payload["required"] = language in required_set
        language_payload.append(payload)

    trained_aggregate = _aggregate(trained_records)
    zero_shot_aggregate = _aggregate(zero_shot_records)
    cohort_payload: dict[str, Any] = {}
    for name in fixed_cohorts:
        members = fixed_cohorts[name]
        missing_trained = sorted(set(members) - trained_set)
        membership_hash = _sha256({"name": name, "languages": list(members)})
        if missing_trained:
            cohort_payload[name] = {
                "status": "not_applicable",
                "reason": "not_all_cohort_languages_were_trained",
                "languages": list(members),
                "missing_trained_languages": missing_trained,
                "membership_sha256": membership_hash,
            }
            continue
        aggregate = _aggregate([by_language[language] for language in members])
        aggregate["membership_sha256"] = membership_hash
        cohort_payload[name] = aggregate

    report: dict[str, Any] = {
        "schema": VALIDATION_REPORT_SCHEMA,
        "run_id": run_id,
        "status": "complete",
        "metric_spec": {
            "primary": "trained_equal_language_atlas_tao_unigram_adjusted_loss",
            "secondary": "raw_token_nll",
            "evaluated_target_domain": "non_special_validation_tokens",
            "special_target_policy": (
                "exclude every frozen special-token ID from both model-NLL and "
                "unigram sufficient statistics"
            ),
            "eod_context_policy": (
                "retain EOD in the packed input/context stream but exclude EOD as a scored target"
            ),
            "final_partial_sequence_policy": (
                "right-pad to sequence length, mask padding, and score every real non-special target"
            ),
            "adjusted_formula": "-mean(log(p_model(token)/p_unigram_language(token)))",
            "unigram_source": "unique_training_only_non_special_tokens",
            "pseudo_count_for_unseen_ids": PSEUDO_COUNT,
            "unigram_denominator": "sum(max(train_count(token),1)) over non-special vocabulary IDs",
            "summation_input": "exact_nll_sum_and_corresponding_non_special_target_token_ids",
        },
        "coverage": {
            "trained_languages": list(trained),
            "required_languages": list(required),
            "observed_languages": sorted(by_language),
            "zero_shot_languages": list(zero_shot_languages),
            "missing_required_languages": [],
        },
        "languages": language_payload,
        "aggregates": {
            "primary": {
                "name": "trained_equal_language_atlas_tao_unigram_adjusted_loss",
                "value": trained_aggregate["equal_language_atlas_tao_adjusted_loss"],
                "language_count": trained_aggregate["language_count"],
                "languages": trained_aggregate["languages"],
            },
            "secondary": {
                "name": "trained_equal_language_raw_nll",
                "value": trained_aggregate["equal_language_raw_nll"],
                "language_count": trained_aggregate["language_count"],
                "languages": trained_aggregate["languages"],
                "target_domain": "non_special_validation_tokens",
            },
            "trained_languages": trained_aggregate,
            "zero_shot_languages": zero_shot_aggregate,
            "fixed_common_cohorts": cohort_payload,
        },
    }
    report["report_sha256"] = _sha256(report)
    return report


def verify_report_hash(report: Mapping[str, Any]) -> bool:
    """Return whether a report's content matches its embedded SHA-256."""

    if not isinstance(report, Mapping):
        return False
    expected = report.get("report_sha256")
    if not isinstance(expected, str):
        return False
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    try:
        actual = _sha256(unhashed)
    except (TypeError, ValueError):
        return False
    return expected == actual


__all__ = [
    "ENGLISH_LANGUAGE",
    "FIXED_COHORT_NAMES",
    "LANGUAGE_LOSS_SCHEMA",
    "LanguageLoss",
    "MissingRequiredLanguageError",
    "PSEUDO_COUNT",
    "UNIGRAM_SCHEMA",
    "UnigramReference",
    "VALIDATION_REPORT_SCHEMA",
    "build_validation_report",
    "compute_language_losses",
    "validate_fixed_cohorts",
    "verify_report_hash",
]
