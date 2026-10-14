"""Neutral contract for tokenizer sentence acquisition and exact slot counting.

Sampling and tokenizer-training stages both consume this contract.  Keeping the
splitter, gate, and acquisition-plan validation here prevents either stage from
importing implementation details owned by the other.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

try:
    import regex as unicode_regex
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("tokenizer sentence splitting requires the 'regex' package") from exc


ACQUISITION_PLAN_TOOL_ID = "tokenizer_sentence_acquisition_plan_v2"
ACQUISITION_PLAN_SCHEMA_VERSION = 1
SPLITTER_ID = "unicode_sentence_terminal_close_v1"
SENTENCE_SLOT_COUNTER_ID = (
    "train_split_sentence_like_times_rehydration_multiplier_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUNDARY_RE = unicode_regex.compile(
    r"(?V1)\p{Sentence_Terminal}+(?:\p{Close_Punctuation}|\p{Sentence_Break=Close})*"
)


class TokenizerSentenceContractError(ValueError):
    """Raised when a tokenizer sentence contract is invalid."""


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_sentence_like(text: str, *, max_chars: int) -> Iterable[str]:
    """Split on Unicode sentence terminals, then hard-chunk by code points."""

    if not isinstance(text, str) or not text.strip():
        raise TokenizerSentenceContractError(
            "document text must be a non-blank string"
        )
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars <= 0
    ):
        raise TokenizerSentenceContractError("max_chars must be a positive integer")

    boundaries: list[int] = []
    for match in _BOUNDARY_RE.finditer(text):
        matched = match.group(0)
        if (
            matched == "."
            and match.start() > 0
            and match.end() < len(text)
            and text[match.start() - 1].isdecimal()
            and text[match.end()].isdecimal()
        ):
            continue
        boundaries.append(match.end())

    start = 0
    raw_segments: list[str] = []
    for end in boundaries:
        segment = text[start:end].strip()
        if segment:
            raw_segments.append(segment)
        start = end
    remainder = text[start:].strip()
    if remainder:
        raw_segments.append(remainder)

    for segment in raw_segments:
        for offset in range(0, len(segment), max_chars):
            chunk = segment[offset : offset + max_chars].strip()
            if chunk:
                yield chunk


def sentence_splitter_contract(*, max_chars: int) -> dict[str, Any]:
    """Return the complete frozen sentence-splitting contract."""

    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars <= 0
    ):
        raise TokenizerSentenceContractError("max_chars must be a positive integer")
    return {
        "id": SPLITTER_ID,
        "regex_pattern": _BOUNDARY_RE.pattern,
        "regex_version": str(unicode_regex.__version__),
        "max_sentence_chars": int(max_chars),
        "length_unit": "Unicode code_points",
        "outer_whitespace_trimmed": True,
        "long_segment_policy": "hard_code_point_chunks",
        "decimal_period_exception": True,
    }


def sentence_slot_gate_contract(*, max_chars: int) -> dict[str, Any]:
    """Return the exact non-exhausted tokenizer-pool acquisition rule."""

    return {
        "counter_id": SENTENCE_SLOT_COUNTER_ID,
        "counted_split": "train",
        "slot_formula": (
            "sum(len(split_sentence_like(text,max_sentence_chars)) "
            "* rehydration_multiplier)"
        ),
        "splitter": sentence_splitter_contract(max_chars=max_chars),
        "non_exhausted_stop_rule": (
            "document_byte_calibration_minima_and_required_sentence_slots"
        ),
        "artificial_pool_cycling_forbidden": True,
        "true_source_exhausted_repetition_allowed": True,
    }


def verify_acquisition_plan(payload: Mapping[str, Any]) -> bool:
    """Verify the immutable v2 tokenizer sentence acquisition plan."""

    if (
        payload.get("schema_version") != ACQUISITION_PLAN_SCHEMA_VERSION
        or payload.get("tool_id") != ACQUISITION_PLAN_TOOL_ID
        or payload.get("status") != "complete"
    ):
        return False
    expected = payload.get("plan_sha256")
    candidate = dict(payload)
    candidate.pop("plan_sha256", None)
    if not isinstance(expected, str) or expected != _canonical_json_sha256(candidate):
        return False
    splitter = payload.get("splitter")
    if not isinstance(splitter, dict):
        return False
    raw_max_chars = splitter.get("max_sentence_chars")
    if (
        not isinstance(raw_max_chars, int)
        or isinstance(raw_max_chars, bool)
        or raw_max_chars <= 0
    ):
        return False
    try:
        expected_splitter = sentence_splitter_contract(max_chars=raw_max_chars)
        expected_gate = sentence_slot_gate_contract(max_chars=raw_max_chars)
    except (TypeError, ValueError):
        return False
    if splitter != expected_splitter or payload.get("expanded_pool_gate") != expected_gate:
        return False
    raw_languages = payload.get("languages")
    if not isinstance(raw_languages, dict) or not raw_languages:
        return False
    try:
        for raw in raw_languages.values():
            if not isinstance(raw, dict):
                return False
            acquisition = raw["acquisition"]
            quota = raw["planned_sentence_count"]
            if (
                not isinstance(acquisition, dict)
                or not isinstance(quota, int)
                or isinstance(quota, bool)
                or quota <= 0
                or acquisition.get("required_rehydrated_sentence_slots") != quota
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def tokenizer_acquisition_target_contract(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    language: str,
) -> dict[str, Any]:
    """Derive the exact bootstrap lineage marker for one frozen quota."""

    if not verify_acquisition_plan(plan):
        raise TokenizerSentenceContractError(
            "tokenizer sentence acquisition plan is invalid"
        )
    if _SHA256_RE.fullmatch(str(plan_file_sha256)) is None:
        raise TokenizerSentenceContractError(
            "tokenizer acquisition plan file hash is invalid"
        )
    raw_languages = plan.get("languages")
    raw = raw_languages.get(language) if isinstance(raw_languages, dict) else None
    acquisition = raw.get("acquisition") if isinstance(raw, dict) else None
    if not isinstance(acquisition, dict):
        raise TokenizerSentenceContractError(
            f"tokenizer acquisition plan has no target for {language}"
        )
    quota = int(raw["planned_sentence_count"])
    if acquisition.get("required_rehydrated_sentence_slots") != quota:
        raise TokenizerSentenceContractError(
            f"tokenizer acquisition target/quota mismatch for {language}"
        )
    return {
        "plan_payload_sha256": plan["plan_sha256"],
        "plan_file_sha256": str(plan_file_sha256),
        "required_rehydrated_sentence_slots": quota,
        "target_unique_source_documents": int(
            acquisition["target_unique_source_documents"]
        ),
        "target_utf8_bytes": int(acquisition["target_utf8_bytes"]),
        "safety_margin": float(acquisition["safety_margin"]),
        "sentence_slot_gate": dict(plan["expanded_pool_gate"]),
    }


__all__ = [
    "ACQUISITION_PLAN_SCHEMA_VERSION",
    "ACQUISITION_PLAN_TOOL_ID",
    "SENTENCE_SLOT_COUNTER_ID",
    "SPLITTER_ID",
    "TokenizerSentenceContractError",
    "sentence_slot_gate_contract",
    "sentence_splitter_contract",
    "split_sentence_like",
    "tokenizer_acquisition_target_contract",
    "verify_acquisition_plan",
]
