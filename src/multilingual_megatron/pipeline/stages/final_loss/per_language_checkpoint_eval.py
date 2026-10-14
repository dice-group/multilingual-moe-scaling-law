"""Load-once, fail-closed per-language validation for A/B-v2 checkpoints.

The normal Megatron validation blend reports one aggregate.  The scaling-law
study instead needs one exact token-NLL numerator and denominator per language.
This module keeps that study policy in the orchestration repository:

* one checkpoint/model setup per evaluation job;
* one deterministic, unshuffled token stream per validation language;
* special targets masked before both NLL summation and ATLAS/Tao adjustment;
* a complete language-coverage check before GPU setup; and
* artifacts directly consumed by :mod:`validation_metrics`.

Only the small ``return_loss_sufficient_statistics`` hook lives in Bridge.  No
vendored Megatron-LM source is modified.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch

from .validation_metrics import (
    LanguageLoss,
    UnigramReference,
    build_validation_report,
    compute_language_losses_from_array,
    validate_fixed_cohorts,
)


EVALUATION_SPEC_SCHEMA = "multilingual_per_language_checkpoint_eval/v1"
EVALUATION_MANIFEST_SCHEMA = "multilingual_per_language_checkpoint_eval_result/v1"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _resolve_path(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _unique_languages(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    languages: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"{field} contains an invalid language ID")
        languages.append(item)
    if len(languages) != len(set(languages)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(sorted(languages))


def _verify_file_identity(path: Path, *, sha256: str, size_bytes: int | None, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise ValueError(
            f"{label} size mismatch for {path}: expected {size_bytes}, got {path.stat().st_size}"
        )
    actual = _sha256_file(path)
    if actual != sha256:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: expected {sha256}, got {actual}")


@dataclass(frozen=True, slots=True)
class UnigramDescriptor:
    language: str
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    source_path: Path
    source_sha256: str
    run_id: str
    recipe_name: str
    recipe_overrides: tuple[str, ...]
    model_contract: Mapping[str, Any]
    output_dir: Path
    data_manifest_path: Path
    data_manifest_sha256: str
    corpus_manifest_path: Path
    corpus_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_iteration_path: Path
    checkpoint_step: int
    sequence_length: int
    eval_micro_batch_size: int
    require_moe_routing_diagnostics: bool
    context_token_id: int
    trained_languages: tuple[str, ...]
    required_languages: tuple[str, ...]
    cohorts: Mapping[str, tuple[str, ...]]
    validation_prefixes: Mapping[str, tuple[str, ...]]
    unigrams: Mapping[str, UnigramDescriptor]
    vocab_size: int
    special_token_ids: tuple[int, ...]
    tokenizer_type: str
    tokenizer_model_path: Path
    tokenizer_model_sha256: str
    collect_routing_diagnostics: bool = True
    extra_padding_batches: int = 0
    diagnostic_languages: tuple[str, ...] = ()


def _artifact_entry_path(
    entry: Any,
    *,
    root: Path,
    field: str,
) -> tuple[Path, str, int | None]:
    if not isinstance(entry, dict):
        raise ValueError(f"{field} must be an artifact object")
    path = _resolve_path(root, entry.get("path"), field=f"{field}.path")
    sha256 = _require_sha256(entry.get("sha256"), field=f"{field}.sha256")
    size_raw = entry.get("size_bytes")
    size_bytes = None
    if size_raw is not None:
        if not isinstance(size_raw, int) or isinstance(size_raw, bool) or size_raw < 0:
            raise ValueError(f"{field}.size_bytes must be a non-negative integer")
        size_bytes = size_raw
    return path, sha256, size_bytes


def _load_unigram_reference(
    descriptor: UnigramDescriptor,
    *,
    vocab_size: int,
    tokenizer_model_sha256: str | None = None,
) -> UnigramReference:
    payload = _load_json(descriptor.path, label=f"{descriptor.language} unigram")
    if _sha256_file(descriptor.path) != descriptor.sha256:
        raise ValueError(f"unigram SHA-256 changed after preflight: {descriptor.path}")
    if payload.get("language") != descriptor.language:
        raise ValueError(f"unigram language mismatch in {descriptor.path}")
    if (
        tokenizer_model_sha256 is not None
        and payload.get("tokenizer_model_sha256") != tokenizer_model_sha256
    ):
        raise ValueError(f"unigram tokenizer SHA-256 mismatch in {descriptor.path}")
    raw_counts = payload.get("counts_by_token_id")
    if not isinstance(raw_counts, dict):
        raise ValueError(f"unigram counts_by_token_id must be an object: {descriptor.path}")
    counts: dict[int, int] = {}
    for raw_id, raw_count in raw_counts.items():
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid unigram token ID {raw_id!r}: {descriptor.path}") from exc
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count <= 0:
            raise ValueError(f"invalid unigram count for token {raw_id}: {descriptor.path}")
        counts[token_id] = raw_count
    special = payload.get("special_token_ids_excluded")
    if not isinstance(special, list):
        raise ValueError(f"unigram special_token_ids_excluded must be a list: {descriptor.path}")
    return UnigramReference.from_counts(
        language=descriptor.language,
        vocab_size=vocab_size,
        counts=counts,
        special_token_ids=special,
    )


@lru_cache(maxsize=8)
def load_evaluation_spec(path: Path) -> EvaluationSpec:
    """Validate every immutable CPU-side input before model/GPU setup."""

    source_path = path.resolve()
    raw = _load_json(source_path, label="per-language evaluation spec")
    if raw.get("schema") != EVALUATION_SPEC_SCHEMA:
        raise ValueError(f"unsupported evaluation spec schema: {raw.get('schema')!r}")
    source_sha256 = _sha256_file(source_path)
    base = source_path.parent

    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
        raise ValueError("run_id must be a non-empty string without surrounding whitespace")
    output_dir = _resolve_path(base, raw.get("output_dir"), field="output_dir")

    model_recipe = raw.get("model_recipe")
    if not isinstance(model_recipe, dict):
        raise ValueError("model_recipe must be an object")
    recipe_name = model_recipe.get("name")
    if not isinstance(recipe_name, str) or not recipe_name or recipe_name != recipe_name.strip():
        raise ValueError("model_recipe.name must be a non-empty recipe name")
    raw_overrides = model_recipe.get("overrides", [])
    if not isinstance(raw_overrides, list) or any(
        not isinstance(item, str) or "=" not in item or not item.strip()
        for item in raw_overrides
    ):
        raise ValueError("model_recipe.overrides must be a list of key=value strings")
    recipe_overrides = tuple(str(item).strip() for item in raw_overrides)
    if len(recipe_overrides) != len(set(recipe_overrides)):
        raise ValueError("model_recipe.overrides must not contain duplicates")
    allowed_override_prefixes = ("model.", "mixed_precision=")
    if any(
        not item.lstrip("+").startswith(allowed_override_prefixes)
        for item in recipe_overrides
    ):
        raise ValueError("model_recipe.overrides may only describe the model/evaluation precision")
    model_contract = model_recipe.get("expected_model")
    if not isinstance(model_contract, dict):
        raise ValueError("model_recipe.expected_model must be an object")
    required_model_fields = {
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "seq_length",
        "vocab_size",
        "num_moe_experts",
    }
    missing_model_fields = sorted(required_model_fields - set(model_contract))
    if missing_model_fields:
        raise ValueError(
            "model_recipe.expected_model lacks: " + ", ".join(missing_model_fields)
        )
    expected_experts = model_contract.get("num_moe_experts")
    if expected_experts is not None and (
        not isinstance(expected_experts, int)
        or isinstance(expected_experts, bool)
        or expected_experts <= 1
    ):
        raise ValueError("expected_model.num_moe_experts must be null (dense) or an integer > 1")
    if expected_experts is not None:
        required_moe_fields = {
            "moe_router_topk",
            "moe_expert_capacity_factor",
            "moe_pad_expert_input_to_capacity",
        }
        missing_moe_fields = sorted(required_moe_fields - set(model_contract))
        if missing_moe_fields:
            raise ValueError(
                "MoE expected_model lacks: " + ", ".join(missing_moe_fields)
            )

    sequence_length = raw.get("sequence_length")
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    context_token_id = raw.get("context_token_id")
    if not isinstance(context_token_id, int) or isinstance(context_token_id, bool) or context_token_id < 0:
        raise ValueError("context_token_id must be a non-negative integer")

    trained = _unique_languages(raw.get("trained_languages"), field="trained_languages")
    required = _unique_languages(raw.get("required_languages"), field="required_languages")
    if not set(trained).issubset(required):
        raise ValueError("trained_languages must be a subset of required_languages")
    diagnostic = (_unique_languages(raw["diagnostic_languages"], field="diagnostic_languages")
                  if "diagnostic_languages" in raw else ())
    if not set(diagnostic).issubset(required):
        raise ValueError("diagnostic_languages must be a subset of required_languages")
    evaluated = diagnostic or required
    raw_cohorts = raw.get("cohorts")
    cohorts = validate_fixed_cohorts(raw_cohorts)

    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be an object")
    checkpoint_path = _resolve_path(base, checkpoint.get("path"), field="checkpoint.path")
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint root directory: {checkpoint_path}")
    checkpoint_step = checkpoint.get("step")
    if not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool) or checkpoint_step < 0:
        raise ValueError("checkpoint.step must be a non-negative integer")
    checkpoint_pointer = checkpoint_path / "latest_checkpointed_iteration.txt"
    if not checkpoint_pointer.is_file():
        raise ValueError(
            "checkpoint.path must be a checkpoint ROOT containing "
            "latest_checkpointed_iteration.txt and iter_* directories"
        )
    try:
        latest_step = int(checkpoint_pointer.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint tracker: {checkpoint_pointer}") from exc
    if latest_step != checkpoint_step:
        raise ValueError(
            f"checkpoint root tracker points to step {latest_step}, expected {checkpoint_step}"
        )
    checkpoint_candidates = (
        checkpoint_path / f"iter_{checkpoint_step:07d}",
        checkpoint_path / f"iter_{checkpoint_step:010d}",
        checkpoint_path / f"iter_{checkpoint_step}",
    )
    checkpoint_iteration_path = next(
        (candidate for candidate in checkpoint_candidates if candidate.is_dir()),
        checkpoint_candidates[0],
    )
    if not checkpoint_iteration_path.is_dir():
        raise ValueError(
            f"checkpoint root lacks an iter_* directory for step {checkpoint_step}: {checkpoint_path}"
        )

    eval_micro_batch_size = raw.get("eval_micro_batch_size", 64)
    if (
        not isinstance(eval_micro_batch_size, int)
        or isinstance(eval_micro_batch_size, bool)
        or eval_micro_batch_size <= 0
    ):
        raise ValueError("eval_micro_batch_size must be a positive integer")
    require_moe_routing_diagnostics = raw.get("require_moe_routing_diagnostics", True)
    if not isinstance(require_moe_routing_diagnostics, bool):
        raise ValueError("require_moe_routing_diagnostics must be boolean")
    diagnostic_options = raw.get("routing_diagnostics", {})
    if not isinstance(diagnostic_options, dict) or set(diagnostic_options) - {"enabled", "extra_padding_batches"}:
        raise ValueError("unknown routing diagnostic options")
    collect_routing = diagnostic_options.get("enabled", True)
    extra_padding = diagnostic_options.get("extra_padding_batches", 0)
    if not isinstance(collect_routing, bool) or (require_moe_routing_diagnostics and not collect_routing):
        raise ValueError("required routing diagnostics cannot be disabled")
    if type(extra_padding) is not int or extra_padding < 0:
        raise ValueError("extra_padding_batches must be a non-negative integer")

    data_entry = raw.get("validation_data_manifest")
    if not isinstance(data_entry, dict):
        raise ValueError("validation_data_manifest must be an object")
    data_manifest_path = _resolve_path(base, data_entry.get("path"), field="validation_data_manifest.path")
    data_manifest_sha256 = _require_sha256(
        data_entry.get("sha256"), field="validation_data_manifest.sha256"
    )
    _verify_file_identity(
        data_manifest_path,
        sha256=data_manifest_sha256,
        size_bytes=None,
        label="validation data manifest",
    )
    data_manifest = _load_json(data_manifest_path, label="validation data manifest")
    tokenizer_type = data_manifest.get("tokenizer_type")
    if tokenizer_type not in {"SentencePieceTokenizer", "GPTSentencePieceTokenizer"}:
        raise ValueError(
            "A/B-v2 validation requires a SentencePiece tokenizer type in the data manifest"
        )
    splits = data_manifest.get("splits")
    validation = splits.get("validation") if isinstance(splits, dict) else None
    if not isinstance(validation, dict):
        raise ValueError("validation data manifest lacks splits.validation")
    by_language = validation.get("prefixes_by_language")
    artifacts_by_language = validation.get("prefix_artifacts_by_language")
    if not isinstance(by_language, dict) or not isinstance(artifacts_by_language, dict):
        raise ValueError(
            "validation data manifest must contain prefixes_by_language and "
            "prefix_artifacts_by_language"
        )

    validation_prefixes: dict[str, tuple[str, ...]] = {}
    assigned_prefixes: dict[str, str] = {}
    for language in evaluated:
        prefixes = by_language.get(language)
        artifacts = artifacts_by_language.get(language)
        if not isinstance(prefixes, list) or not prefixes:
            raise ValueError(f"missing validation prefixes for required language {language}")
        if not isinstance(artifacts, list) or len(artifacts) != len(prefixes):
            raise ValueError(f"validation prefix artifact coverage mismatch for {language}")
        normalized = tuple(sorted(str(item) for item in prefixes))
        artifact_by_prefix = {
            str(item.get("prefix")): item for item in artifacts if isinstance(item, dict)
        }
        if set(artifact_by_prefix) != set(normalized):
            raise ValueError(f"validation prefix artifact identities mismatch for {language}")
        resolved_prefixes: list[str] = []
        for raw_prefix in normalized:
            prefix_path = _resolve_path(data_manifest_path.parent, raw_prefix, field="validation prefix")
            prefix = str(prefix_path)
            previous = assigned_prefixes.get(prefix)
            if previous is not None and previous != language:
                raise ValueError(f"validation prefix assigned to both {previous} and {language}: {prefix}")
            assigned_prefixes[prefix] = language
            artifact = artifact_by_prefix[raw_prefix]
            for suffix, hash_key, size_key in (
                (".bin", "bin_sha256", "bin_size_bytes"),
                (".idx", "idx_sha256", "idx_size_bytes"),
            ):
                expected_hash = _require_sha256(
                    artifact.get(hash_key), field=f"{language} {raw_prefix} {hash_key}"
                )
                expected_size = artifact.get(size_key)
                if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
                    raise ValueError(f"{language} {raw_prefix} {size_key} must be non-negative")
                _verify_file_identity(
                    Path(prefix + suffix),
                    sha256=expected_hash,
                    size_bytes=expected_size,
                    label=f"{language} validation {suffix[1:]}",
                )
            resolved_prefixes.append(prefix)
        validation_prefixes[language] = tuple(resolved_prefixes)

    corpus_entry = raw.get("corpus_manifest")
    if not isinstance(corpus_entry, dict):
        raise ValueError("corpus_manifest must be an object")
    corpus_manifest_path = _resolve_path(base, corpus_entry.get("path"), field="corpus_manifest.path")
    corpus_manifest_sha256 = _require_sha256(corpus_entry.get("sha256"), field="corpus_manifest.sha256")
    _verify_file_identity(
        corpus_manifest_path,
        sha256=corpus_manifest_sha256,
        size_bytes=None,
        label="corpus manifest",
    )
    corpus = _load_json(corpus_manifest_path, label="corpus manifest")
    if corpus.get("ready") is not True or corpus.get("release_status") != "ready":
        raise ValueError("corpus manifest is not a ready release")
    tokenizer = corpus.get("inputs", {}).get("tokenizer_model")
    if not isinstance(tokenizer, dict):
        raise ValueError("corpus manifest lacks inputs.tokenizer_model")
    vocab_size = tokenizer.get("vocabulary_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("corpus tokenizer vocabulary_size must be positive")
    special_raw = tokenizer.get("special_token_ids")
    if not isinstance(special_raw, list):
        raise ValueError("corpus tokenizer special_token_ids must be a list")
    special_token_ids = tuple(sorted(int(item) for item in special_raw))
    if len(special_token_ids) != len(set(special_token_ids)):
        raise ValueError("corpus tokenizer special_token_ids contains duplicates")
    if context_token_id not in special_token_ids:
        raise ValueError("context_token_id must be one of the frozen special token IDs")
    if context_token_id >= vocab_size or any(item < 0 or item >= vocab_size for item in special_token_ids):
        raise ValueError("corpus tokenizer special token ID is outside vocabulary")
    tokenizer_model_path = _resolve_path(
        corpus_manifest_path.parent, tokenizer.get("path"), field="corpus tokenizer path"
    )
    tokenizer_model_sha256 = _require_sha256(tokenizer.get("sha256"), field="corpus tokenizer sha256")
    _verify_file_identity(
        tokenizer_model_path,
        sha256=tokenizer_model_sha256,
        size_bytes=None,
        label="SentencePiece tokenizer model",
    )

    manifest_tokenizer_path = _resolve_path(
        data_manifest_path.parent,
        data_manifest.get("tokenizer_model"),
        field="validation data tokenizer_model",
    )
    if manifest_tokenizer_path != tokenizer_model_path:
        raise ValueError("validation data and corpus manifests reference different tokenizer models")

    language_entries = corpus.get("languages")
    if not isinstance(language_entries, dict):
        raise ValueError("corpus manifest languages must be an object")
    unigrams: dict[str, UnigramDescriptor] = {}
    for language in evaluated:
        language_entry = language_entries.get(language)
        train = language_entry.get("train") if isinstance(language_entry, dict) else None
        unigram_entry = train.get("unigram_counts") if isinstance(train, dict) else None
        unigram_path, unigram_sha, unigram_size = _artifact_entry_path(
            unigram_entry,
            root=corpus_manifest_path.parent,
            field=f"languages.{language}.train.unigram_counts",
        )
        _verify_file_identity(
            unigram_path,
            sha256=unigram_sha,
            size_bytes=unigram_size,
            label=f"{language} training unigram",
        )
        descriptor = UnigramDescriptor(language=language, path=unigram_path, sha256=unigram_sha)
        reference = _load_unigram_reference(
            descriptor,
            vocab_size=vocab_size,
            tokenizer_model_sha256=tokenizer_model_sha256,
        )
        if reference.special_token_ids != special_token_ids:
            raise ValueError(f"special token IDs drift in {language} unigram")
        unigrams[language] = descriptor

    return EvaluationSpec(
        source_path=source_path,
        source_sha256=source_sha256,
        run_id=run_id,
        recipe_name=recipe_name,
        recipe_overrides=recipe_overrides,
        model_contract=dict(model_contract),
        output_dir=output_dir,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_sha256,
        corpus_manifest_path=corpus_manifest_path,
        corpus_manifest_sha256=corpus_manifest_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_iteration_path=checkpoint_iteration_path.resolve(),
        checkpoint_step=checkpoint_step,
        sequence_length=sequence_length,
        eval_micro_batch_size=eval_micro_batch_size,
        require_moe_routing_diagnostics=require_moe_routing_diagnostics,
        context_token_id=context_token_id,
        trained_languages=trained,
        required_languages=required,
        cohorts=cohorts,
        validation_prefixes=validation_prefixes,
        unigrams=unigrams,
        vocab_size=vocab_size,
        special_token_ids=special_token_ids,
        tokenizer_type=tokenizer_type,
        tokenizer_model_path=tokenizer_model_path,
        tokenizer_model_sha256=tokenizer_model_sha256,
        collect_routing_diagnostics=collect_routing,
        extra_padding_batches=extra_padding,
        diagnostic_languages=diagnostic,
    )


def verify_evaluation_model_contract(config: Any, spec: EvaluationSpec) -> None:
    """Fail before checkpoint loading if the recipe/overrides changed its shape."""

    model = config.model
    mismatches: list[str] = []
    for field, expected in spec.model_contract.items():
        actual = getattr(model, field, None)
        if actual != expected:
            mismatches.append(f"{field}:{actual!r}!={expected!r}")
    if mismatches:
        raise ValueError("evaluation model contract mismatch: " + "; ".join(mismatches))


def configure_checkpoint_evaluation(config: Any, spec: EvaluationSpec) -> Any:
    """Apply only the frozen, evaluation-safe runtime mutations to a recipe."""

    verify_evaluation_model_contract(config, spec)
    config.checkpoint.load = str(spec.checkpoint_path)
    config.checkpoint.ckpt_step = spec.checkpoint_step
    config.checkpoint.load_optim = False
    config.checkpoint.load_rng = False
    config.checkpoint.async_save = False
    config.checkpoint.save = None
    config.validation.skip_train = True
    config.validation.eval_interval = 0
    config.validation.eval_iters = 0
    config.tokenizer.tokenizer_type = spec.tokenizer_type
    config.tokenizer.tokenizer_model = str(spec.tokenizer_model_path)
    return config


class SequenceSource(Protocol):
    @property
    def sequence_lengths(self) -> Sequence[int]: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Sequence[int]: ...

    def get(self, index: int, offset: int = 0, length: int | None = None) -> Sequence[int]: ...


@dataclass(slots=True)
class _SourceView:
    source: SequenceSource
    cumulative_lengths: np.ndarray
    total_tokens: int

    @classmethod
    def build(cls, source: SequenceSource) -> "_SourceView":
        lengths = np.asarray(source.sequence_lengths, dtype=np.int64)
        if lengths.ndim != 1 or len(lengths) != len(source):
            raise ValueError("indexed prefix sequence length metadata is invalid")
        if np.any(lengths < 0):
            raise ValueError("indexed prefix contains a negative sequence length")
        cumulative = np.cumsum(lengths, dtype=np.int64)
        return cls(
            source=source,
            cumulative_lengths=cumulative,
            total_tokens=int(cumulative[-1]) if len(cumulative) else 0,
        )

    def read(self, start: int, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise IndexError("token range is outside indexed prefix")
        pieces: list[np.ndarray] = []
        remaining = count
        position = start
        while remaining:
            document = bisect.bisect_right(self.cumulative_lengths, position)
            previous_end = int(self.cumulative_lengths[document - 1]) if document else 0
            offset = position - previous_end
            available = int(self.cumulative_lengths[document]) - position
            take = min(remaining, available)
            piece = np.asarray(self.source.get(document, offset=offset, length=take), dtype=np.int64)
            if piece.size != take:
                raise ValueError("indexed prefix returned an unexpected token count")
            pieces.append(piece)
            position += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


class TokenStream:
    """A deterministic concatenation of all documents in manifest prefix order."""

    def __init__(self, sources: Sequence[SequenceSource]) -> None:
        if not sources:
            raise ValueError("at least one indexed validation prefix is required")
        self.views = tuple(_SourceView.build(source) for source in sources)
        totals = np.asarray([view.total_tokens for view in self.views], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(totals, dtype=np.int64)
        self.total_tokens = int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) else 0
        if self.total_tokens <= 0:
            raise ValueError("validation token stream is empty")

    def read(self, start: int, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise IndexError("token range is outside validation stream")
        pieces: list[np.ndarray] = []
        remaining = count
        position = start
        while remaining:
            source_index = bisect.bisect_right(self.cumulative_lengths, position)
            previous_end = int(self.cumulative_lengths[source_index - 1]) if source_index else 0
            local_start = position - previous_end
            available = int(self.cumulative_lengths[source_index]) - position
            take = min(remaining, available)
            pieces.append(self.views[source_index].read(local_start, take))
            position += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def iter_tokens(self) -> Iterable[int]:
        for view in self.views:
            for document in range(len(view.source)):
                values = view.source[document]
                for token_id in values:
                    yield int(token_id)


class ExactLanguageDataset(torch.utils.data.Dataset):
    """Fixed-length samples that score every non-special stream target once."""

    def __init__(
        self,
        stream: TokenStream,
        *,
        sequence_length: int,
        context_token_id: int,
        special_token_ids: Iterable[int],
        sample_multiple: int = 1,
    ) -> None:
        if sequence_length <= 0 or sample_multiple <= 0:
            raise ValueError("sequence_length and sample_multiple must be positive")
        self.stream = stream
        self.sequence_length = int(sequence_length)
        self.context_token_id = int(context_token_id)
        self.special_token_ids = frozenset(int(item) for item in special_token_ids)
        self.natural_sample_count = math.ceil(stream.total_tokens / self.sequence_length)
        self.sample_count = (
            math.ceil(self.natural_sample_count / sample_multiple) * sample_multiple
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        tokens = np.zeros(self.sequence_length, dtype=np.int64)
        labels = np.zeros(self.sequence_length, dtype=np.int64)
        loss_mask = np.zeros(self.sequence_length, dtype=np.float32)
        input_valid_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        position_ids = np.arange(self.sequence_length, dtype=np.int64)
        if index < self.natural_sample_count:
            start = index * self.sequence_length
            count = min(self.sequence_length, self.stream.total_tokens - start)
            # Input validity follows constructed positions, including the initial
            # context and real special tokens. It is independent of target scoring.
            input_valid_mask[:count] = True
            targets = self.stream.read(start, count)
            labels[:count] = targets
            tokens[0] = self.context_token_id if start == 0 else int(self.stream.read(start - 1, 1)[0])
            if count > 1:
                tokens[1:count] = targets[:-1]
            loss_mask[:count] = np.asarray(
                [0.0 if int(item) in self.special_token_ids else 1.0 for item in targets],
                dtype=np.float32,
            )
        return {
            "tokens": torch.from_numpy(tokens),
            "labels": torch.from_numpy(labels),
            "loss_mask": torch.from_numpy(loss_mask),
            "position_ids": torch.from_numpy(position_ids),
            "input_valid_mask": torch.from_numpy(input_valid_mask),
            "sample_id": torch.tensor(index, dtype=torch.int64),
        }

    def iter_non_special_targets(self) -> Iterable[int]:
        for token_id in self.stream.iter_tokens():
            if token_id not in self.special_token_ids:
                yield token_id

    def non_special_target_ids(self) -> np.ndarray:
        """Materialize the exact scored targets for vectorized metric reduction."""

        targets = self.stream.read(0, self.stream.total_tokens)
        if not self.special_token_ids:
            return targets
        special = np.asarray(tuple(sorted(self.special_token_ids)), dtype=targets.dtype)
        return targets[~np.isin(targets, special)]


def _open_indexed_sources(prefixes: Sequence[str]) -> list[SequenceSource]:
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    return [IndexedDataset(prefix, multimodal=False, mmap=True) for prefix in prefixes]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distributed_loss_stats(total_loss_dict: Any) -> tuple[float, int]:
    """Select a pipeline-last rank and broadcast its already-DP-reduced stats."""

    import torch.distributed as dist

    local = None
    if isinstance(total_loss_dict, dict) and "lm loss" in total_loss_dict:
        value = total_loss_dict["lm loss"]
        if isinstance(value, torch.Tensor) and value.numel() == 2:
            local = value.detach().to(dtype=torch.float64).view(2)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        if local is None:
            raise RuntimeError("Bridge evaluation returned no lm loss sufficient statistics")
        return float(local[0].item()), int(round(float(local[1].item())))

    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    candidate = torch.tensor([rank if local is not None else world], dtype=torch.int64, device=device)
    dist.all_reduce(candidate, op=dist.ReduceOp.MIN)
    source_rank = int(candidate.item())
    if source_rank >= world:
        raise RuntimeError("no pipeline rank produced lm loss sufficient statistics")
    payload = local.to(device=device) if rank == source_rank else torch.zeros(2, dtype=torch.float64, device=device)
    dist.broadcast(payload, src=source_rank)
    return float(payload[0].item()), int(round(float(payload[1].item())))


def diagnostic_manifest(spec: EvaluationSpec, languages: Mapping[str, Any], timing: Mapping[str, float]) -> dict[str, Any]:
    """A subset receipt is deliberately not a complete scientific evaluation."""
    if not spec.diagnostic_languages or set(languages) != set(spec.diagnostic_languages):
        raise ValueError("incomplete diagnostic language coverage")
    result = {
        "schema": "multilingual_checkpoint_evaluation_diagnostic/v1", "status": "complete_diagnostic",
        "run_id": spec.run_id, "spec": {"path": str(spec.source_path), "sha256": spec.source_sha256},
        "checkpoint": {"root": str(spec.checkpoint_path), "step": spec.checkpoint_step},
        "model_contract": dict(spec.model_contract),
        "validation_data_manifest": {"path": str(spec.data_manifest_path), "sha256": spec.data_manifest_sha256},
        "corpus_manifest": {"path": str(spec.corpus_manifest_path), "sha256": spec.corpus_manifest_sha256},
        "coverage": {"trained_languages": list(spec.trained_languages),
                     "full_required_languages": list(spec.required_languages),
                     "evaluated_languages": sorted(languages), "complete_full_evaluation": False},
        "languages": dict(languages), "timing": dict(timing),
    }
    result["result_sha256"] = _payload_sha256(result)
    return result


def run_per_language_checkpoint_evaluation(config: Any, forward_step_func: Any, spec_path: Path) -> None:
    """Load the requested checkpoint once and evaluate every required language."""

    # Import Bridge lazily so CPU-only spec and dataset tests do not require its
    # optional training dependencies.
    import torch.distributed as dist
    from megatron.core.rerun_state_machine import RerunDataIterator
    from megatron.bridge.data.samplers import build_pretraining_data_loader
    from megatron.bridge.training.config import runtime_config_update
    from megatron.bridge.training.eval import evaluate
    from multilingual_megatron.pipeline.backends.megatron_bridge.checkpoint_inference import setup_bridge_inference_only
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training.train import _finish_train
    from megatron.bridge.training.pretrain import _maybe_destroy_process_group

    spec = load_evaluation_spec(spec_path)
    configure_checkpoint_evaluation(config, spec)
    runtime_config_update(config)
    evaluation_started = time.perf_counter()

    state = GlobalState()
    state.cfg = config

    def _empty_dataset_provider(_sample_counts: Sequence[int], _dataset_config: Any):
        return None, None, None

    should_destroy_process_group = not dist.is_initialized()
    # No backward pass: fused accumulation otherwise expects absent main-grad buffers.
    config.model.gradient_accumulation_fusion = False
    setup_output = setup_bridge_inference_only(state, _empty_dataset_provider)
    print("[EVAL_INFERENCE_ONLY] optimizer=None scheduler=None wrap_with_ddp=False gradient_accumulation_fusion=False", flush=True)
    state = setup_output.state
    model = setup_output.model
    checkpoint_manager = setup_output.checkpoint_manager
    pg_collection = setup_output.pg_collection
    rank = dist.get_rank() if dist.is_initialized() else 0
    num_moe_experts = int(getattr(config.model, "num_moe_experts", 0) or 0)
    is_moe = num_moe_experts > 1 and spec.collect_routing_diagnostics

    try:
        if is_moe:
            from .moe_validation_diagnostics import install_moe_validation_diagnostics
            from .routing_batch_context import validate_routing_topology, wrap_routing_forward_step
            from megatron.bridge.training.utils.train_utils import prepare_forward_step_func

            validate_routing_topology(config.model)
            if not install_moe_validation_diagnostics() and spec.require_moe_routing_diagnostics:
                raise RuntimeError("failed to install required MoE routing diagnostics")
            forward_step_func = wrap_routing_forward_step(
                prepare_forward_step_func(forward_step_func, state),
                expected_layers=int(config.model.num_layers),
            )
        if int(state.train_state.step) != spec.checkpoint_step:
            raise RuntimeError(
                f"loaded checkpoint step {state.train_state.step}, expected {spec.checkpoint_step}"
            )
        if int(config.model.seq_length) != spec.sequence_length:
            raise RuntimeError(
                f"model sequence length {config.model.seq_length}, expected {spec.sequence_length}"
            )

        dp_size = int(pg_collection.dp.size())
        dp_rank = int(pg_collection.dp.rank())
        config.validation.eval_micro_batch_size = spec.eval_micro_batch_size
        config.validation.eval_global_batch_size = spec.eval_micro_batch_size * dp_size
        observations: list[LanguageLoss] = []
        per_language_manifest: dict[str, Any] = {}
        if rank == 0:
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            (spec.output_dir / ".validation_complete").unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - evaluation_started
        total_forward_seconds = 0.0

        for language in spec.diagnostic_languages or spec.required_languages:
            if is_moe:
                from .moe_validation_diagnostics import reset_moe_validation_diagnostics

                reset_moe_validation_diagnostics()
            sources = _open_indexed_sources(spec.validation_prefixes[language])
            stream = TokenStream(sources)
            dataset = ExactLanguageDataset(
                stream,
                sequence_length=spec.sequence_length,
                context_token_id=spec.context_token_id,
                special_token_ids=spec.special_token_ids,
                sample_multiple=spec.eval_micro_batch_size * dp_size,
            )
            dataset.sample_count += spec.extra_padding_batches * spec.eval_micro_batch_size * dp_size
            eval_iters = len(dataset) // (spec.eval_micro_batch_size * dp_size)
            config.validation.eval_iters = eval_iters
            loader = build_pretraining_data_loader(
                dataset,
                consumed_samples=0,
                dataloader_type="single",
                micro_batch_size=spec.eval_micro_batch_size,
                num_workers=0,
                data_sharding=False,
                pin_memory=True,
                persistent_workers=False,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                drop_last=True,
                global_batch_size=spec.eval_micro_batch_size * dp_size,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_started = time.perf_counter()
            total_loss_dict, _non_loss, timelimit = evaluate(
                state,
                forward_step_func,
                RerunDataIterator(iter(loader)),
                model,
                None,
                config.model,
                verbose=True,
                pg_collection=pg_collection,
                return_loss_sufficient_statistics=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started
            total_forward_seconds += forward_seconds
            if timelimit:
                raise RuntimeError(f"time limit reached while evaluating {language}")
            nll_sum, backend_token_count = _distributed_loss_stats(total_loss_dict)
            if is_moe:
                from .moe_validation_diagnostics import collect_moe_validation_diagnostics

                routing_diagnostics = collect_moe_validation_diagnostics(
                    expected_num_layers=int(getattr(config.model, "num_layers", 0) or 0),
                    expected_num_experts=num_moe_experts,
                    expected_sample_count=len(dataset),
                    expected_valid_tokens=stream.total_tokens,
                    sequence_length=spec.sequence_length,
                )
                routing_failure = ""
                if spec.require_moe_routing_diagnostics and not routing_diagnostics.get(
                    "evidence_available", False
                ):
                    routing_failure = (
                        "missing required MoE dropped-token evidence: "
                        f"{routing_diagnostics.get('reason') or routing_diagnostics.get('evidence_errors')}"
                    )
                dropped_assignments = (
                    routing_diagnostics.get("totals", {}).get("dropped_assignments")
                )
                if isinstance(dropped_assignments, int) and dropped_assignments > 0:
                    routing_failure = f"observed {dropped_assignments} dropped MoE assignments"
                if routing_failure:
                    if rank == 0:
                        _atomic_json(
                            spec.output_dir / "failures" / f"{language}.routing.json",
                            {
                                "status": "invalid",
                                "run_id": spec.run_id,
                                "language": language,
                                "reason": routing_failure,
                                "moe_routing": routing_diagnostics,
                            },
                        )
                    raise RuntimeError(f"{language}: {routing_failure}")
            else:
                routing_diagnostics = {
                    "schema": "multilingual_moe_routing_diagnostics/v1",
                    "applicable": False,
                    "evidence_available": True,
                    "reason": "dense model or diagnostics explicitly disabled",
                }
            routing_diagnostics["validation_stream"] = {
                "real_tokens_including_special": stream.total_tokens,
                "initial_context_positions": 1,
                "corpus_derived_input_positions": stream.total_tokens - 1,
                "valid_input_positions_per_layer": stream.total_tokens,
                "natural_samples": dataset.natural_sample_count,
                "total_samples": len(dataset),
                "forwarded_tokens_per_layer": len(dataset) * spec.sequence_length,
                "padding_tokens_per_layer": len(dataset) * spec.sequence_length - stream.total_tokens,
            }

            if rank == 0:
                unigram = _load_unigram_reference(
                    spec.unigrams[language],
                    vocab_size=spec.vocab_size,
                    tokenizer_model_sha256=spec.tokenizer_model_sha256,
                )
                observation = compute_language_losses_from_array(
                    language=language,
                    nll_sum=nll_sum,
                    target_token_ids=dataset.non_special_target_ids(),
                    unigram=unigram,
                )
                if observation.token_count != backend_token_count:
                    raise RuntimeError(
                        f"{language} token denominator mismatch: backend={backend_token_count}, "
                        f"target_stream={observation.token_count}"
                    )
                language_path = spec.output_dir / "languages" / f"{language}.json"
                language_payload = observation.to_dict()
                language_payload["backend"] = {
                    "name": "megatron_bridge",
                    "loss_statistic": "float64_accumulated_sum_and_non_special_token_count",
                    "backend_token_count": backend_token_count,
                    "forward_seconds": forward_seconds,
                    "scored_tokens_per_second": backend_token_count / forward_seconds,
                    "checkpoint_step": spec.checkpoint_step,
                    "sequence_length": spec.sequence_length,
                    "eval_micro_batch_size": spec.eval_micro_batch_size,
                    "stream_tokens_including_special": stream.total_tokens,
                    "padded_samples": len(dataset) - dataset.natural_sample_count,
                    "validation_prefixes": list(spec.validation_prefixes[language]),
                }
                language_payload["moe_routing"] = routing_diagnostics
                _atomic_json(language_path, language_payload)
                observations.append(observation)
                per_language_manifest[language] = {
                    "path": str(language_path.resolve()),
                    "sha256": _sha256_file(language_path),
                    "token_count": observation.token_count,
                    "nll_sum": observation.nll_sum,
                    "forward_seconds": forward_seconds,
                    "scored_tokens_per_second": backend_token_count / forward_seconds,
                }
            if dist.is_initialized():
                dist.barrier()
            del loader, dataset, stream, sources

        if spec.diagnostic_languages:
            if rank == 0:
                result = diagnostic_manifest(spec, per_language_manifest, {
                    "setup_seconds": setup_seconds, "forward_seconds": total_forward_seconds,
                    "internal_total_seconds": time.perf_counter()-evaluation_started})
                _atomic_json(spec.output_dir / "diagnostic_manifest.json", result)
                (spec.output_dir / ".diagnostic_complete").write_text("ok\n", encoding="utf-8")
            if dist.is_initialized():
                dist.barrier()
            return

        if rank == 0:
            report = build_validation_report(
                run_id=spec.run_id,
                observations=observations,
                trained_languages=spec.trained_languages,
                required_languages=spec.required_languages,
                cohorts=spec.cohorts,
            )
            report_path = spec.output_dir / "validation_report.json"
            _atomic_json(report_path, report)
            result = {
                "schema": EVALUATION_MANIFEST_SCHEMA,
                "status": "complete",
                "run_id": spec.run_id,
                "spec": {"path": str(spec.source_path), "sha256": spec.source_sha256},
                "checkpoint": {
                    "root": str(spec.checkpoint_path),
                    "iteration_path": str(spec.checkpoint_iteration_path),
                    "step": spec.checkpoint_step,
                },
                "validation_data_manifest": {
                    "path": str(spec.data_manifest_path),
                    "sha256": spec.data_manifest_sha256,
                },
                "corpus_manifest": {
                    "path": str(spec.corpus_manifest_path),
                    "sha256": spec.corpus_manifest_sha256,
                },
                "tokenizer_model": {
                    "path": str(spec.tokenizer_model_path),
                    "sha256": spec.tokenizer_model_sha256,
                    "vocab_size": spec.vocab_size,
                    "special_token_ids": list(spec.special_token_ids),
                },
                "coverage": {
                    "trained_languages": list(spec.trained_languages),
                    "required_languages": list(spec.required_languages),
                    "evaluated_languages": sorted(per_language_manifest),
                    "missing_required_languages": [],
                },
                "moe_routing_diagnostics": {
                    "model_is_moe": is_moe,
                    "num_experts": num_moe_experts,
                    "required_for_moe": spec.require_moe_routing_diagnostics,
                },
                "timing": {
                    "setup_seconds": setup_seconds,
                    "forward_seconds": total_forward_seconds,
                    "internal_total_seconds": time.perf_counter() - evaluation_started,
                    "setup_definition": "runtime config, distributed/model setup, and checkpoint load before first forward",
                    "forward_definition": "CUDA-synchronized evaluate calls only",
                },
                "languages": per_language_manifest,
                "validation_report": {
                    "path": str(report_path.resolve()),
                    "sha256": _sha256_file(report_path),
                    "report_sha256": report["report_sha256"],
                },
            }
            result["result_sha256"] = _payload_sha256(result)
            _atomic_json(spec.output_dir / "evaluation_manifest.json", result)
            (spec.output_dir / ".validation_complete").write_text("ok\n", encoding="utf-8")
        if dist.is_initialized():
            dist.barrier()
    finally:
        try:
            _finish_train(state, checkpoint_manager)
        finally:
            _maybe_destroy_process_group(should_destroy_process_group)


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA",
    "EVALUATION_SPEC_SCHEMA",
    "EvaluationSpec",
    "ExactLanguageDataset",
    "TokenStream",
    "configure_checkpoint_evaluation",
    "load_evaluation_spec",
    "run_per_language_checkpoint_evaluation",
    "verify_evaluation_model_contract",
]
