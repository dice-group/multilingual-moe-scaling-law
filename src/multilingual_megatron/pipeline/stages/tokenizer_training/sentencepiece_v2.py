"""Fail-closed SentencePiece Unigram contract for the A/B-v2 study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


PRODUCTION_V2_CHARACTER_COVERAGE = 0.999995
PRODUCTION_V2_NORMALIZATION_RULE = "nmt_nfkc"
PRODUCTION_V2_META_PIECES = ("<unk>", "<s>", "</s>", "<pad>")
PRODUCTION_V2_META_IDS = (0, 1, 2, 3)
PRODUCTION_V2_BYTE_PIECE_COUNT = 256
# Built-in ``nmt_nfkc`` map embedded by the pinned SentencePiece 0.2.1 runtime.
# Auditing the bytes prevents a model from merely labelling a different map
# ``nmt_nfkc`` while silently changing normalization.
PRODUCTION_V2_NMT_NFKC_CHARS_MAP_SIZE = 240_007
PRODUCTION_V2_NMT_NFKC_CHARS_MAP_SHA256 = (
    "79372c41389c2b9b29bc171017ab5400e352debd686b02670a42bec709015074"
)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def inspect_sentence_jsonl(
    path: Path,
    *,
    text_key: str,
    expected_sha256: str,
    expected_sentence_count: int,
) -> dict[str, Any]:
    """Validate and hash the frozen one-sentence-per-line training artifact."""

    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"production_v2 sentence JSONL does not exist: {path}")
    if path.suffix != ".jsonl":
        raise SystemExit(f"production_v2 sentence artifact must end in .jsonl: {path}")
    if not text_key:
        raise SystemExit("production_v2 text_key must be non-empty")

    digest = hashlib.sha256()
    sentence_count = 0
    size_bytes = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            size_bytes += len(raw_line)
            try:
                text_line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SystemExit(
                    f"production_v2 sentence JSONL is not UTF-8 at line {line_number}: {path}"
                ) from exc
            if not text_line.strip():
                raise SystemExit(
                    f"production_v2 sentence JSONL contains a blank line at {line_number}: {path}"
                )
            try:
                row = json.loads(text_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"production_v2 sentence JSONL has invalid JSON at line {line_number}: {path}"
                ) from exc
            if not isinstance(row, dict):
                raise SystemExit(
                    f"production_v2 sentence JSONL line {line_number} is not an object: {path}"
                )
            text = row.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(
                    f"production_v2 sentence JSONL line {line_number} lacks non-empty {text_key!r}: {path}"
                )
            sentence_count += 1

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "production_v2 sentence JSONL SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256} ({path})"
        )
    if sentence_count != expected_sentence_count:
        raise SystemExit(
            "production_v2 sentence count mismatch: "
            f"expected {expected_sentence_count}, got {sentence_count} ({path})"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": size_bytes,
        "sentence_count": sentence_count,
        "text_key": text_key,
        "ordering": "physical_jsonl_line_order",
        "strict_jsonl": True,
    }


def iter_sentence_jsonl(path: Path, *, text_key: str) -> Iterable[str]:
    """Yield a previously audited sentence artifact without reordering it."""

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            row = json.loads(line)
            yield row[text_key]


def _protobuf_value(message: Any, field: Any) -> Any:
    value = getattr(message, field.name)
    if field.is_repeated:
        if field.cpp_type == field.CPPTYPE_ENUM:
            return [field.enum_type.values_by_number[int(item)].name for item in value]
        return list(value)
    if field.cpp_type == field.CPPTYPE_ENUM:
        return field.enum_type.values_by_number[int(value)].name
    if field.type == field.TYPE_BYTES:
        data = bytes(value)
        return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return value


def protobuf_effective_spec(message: Any) -> dict[str, Any]:
    """Serialize every protobuf field using its effective value, including defaults."""

    return {
        field.name: _protobuf_value(message, field)
        for field in message.DESCRIPTOR.fields
    }


def _expected_trainer_spec(
    *,
    model_prefix: str,
    vocab_size: int,
    num_threads: int,
) -> dict[str, Any]:
    # Keep this exhaustive.  If a future SentencePiece protobuf adds a field,
    # the field-set comparison below fails until its intended value is reviewed.
    return {
        "input": [],
        "input_format": "",
        "model_prefix": model_prefix,
        "model_type": "UNIGRAM",
        "vocab_size": vocab_size,
        "accept_language": [],
        "self_test_sample_size": 0,
        "enable_differential_privacy": False,
        "differential_privacy_noise_level": 0.0,
        "differential_privacy_clipping_threshold": 0,
        "character_coverage": PRODUCTION_V2_CHARACTER_COVERAGE,
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "mining_sentence_size": 0,
        "training_sentence_size": 0,
        "seed_sentencepiece_size": 1_000_000,
        "shrinking_factor": 0.75,
        "max_sentence_length": 262_144,
        "num_threads": num_threads,
        "num_sub_iterations": 2,
        "max_sentencepiece_length": 16,
        "split_by_unicode_script": True,
        "split_by_number": True,
        "split_by_whitespace": True,
        "treat_whitespace_as_suffix": False,
        "allow_whitespace_only_pieces": False,
        "split_digits": False,
        "pretokenization_delimiter": "",
        "control_symbols": [],
        "user_defined_symbols": [],
        "required_chars": "",
        "byte_fallback": True,
        "vocabulary_output_piece_score": True,
        "hard_vocab_limit": True,
        "use_all_vocab": False,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "pad_piece": "<pad>",
        "unk_surface": " ⁇ ",
        "train_extremely_large_corpus": True,
    }


def _different(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return not (
            isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=5e-8)
        )
    return actual != expected


def audit_production_v2_model(
    model_path: Path,
    *,
    expected_vocab_size: int,
    expected_num_threads: int,
    expected_model_prefix: str,
) -> dict[str, Any]:
    """Parse ``tokenizer.model`` and fail unless the complete v2 contract holds.

    ``expected_vocab_size`` is parameterized solely so unit tests can build a
    viable tiny fixture.  Production config and the CLI require 64,000.
    """

    try:
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2 as model_pb2
    except ModuleNotFoundError as exc:
        raise SystemExit("production_v2 audit requires sentencepiece") from exc

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise SystemExit(f"production_v2 tokenizer model is missing: {model_path}")
    model_bytes = model_path.read_bytes()
    model = model_pb2.ModelProto()
    try:
        model.ParseFromString(model_bytes)
    except Exception as exc:
        raise SystemExit(f"could not parse SentencePiece model protobuf: {model_path}") from exc

    trainer = protobuf_effective_spec(model.trainer_spec)
    expected_trainer = _expected_trainer_spec(
        model_prefix=expected_model_prefix,
        vocab_size=expected_vocab_size,
        num_threads=expected_num_threads,
    )
    if set(trainer) != set(expected_trainer):
        missing = sorted(set(expected_trainer) - set(trainer))
        unexpected = sorted(set(trainer) - set(expected_trainer))
        raise SystemExit(
            "production_v2 TrainerSpec field set changed; review SentencePiece protobuf: "
            f"missing={missing}, unexpected={unexpected}"
        )
    trainer_mismatches = {
        name: {"expected": expected_trainer[name], "actual": trainer[name]}
        for name in sorted(expected_trainer)
        if _different(trainer[name], expected_trainer[name])
    }
    if trainer_mismatches:
        raise SystemExit(
            "production_v2 TrainerSpec mismatch: "
            + json.dumps(trainer_mismatches, ensure_ascii=False, sort_keys=True)
        )

    normalizer = protobuf_effective_spec(model.normalizer_spec)
    expected_normalizer = {
        "name": PRODUCTION_V2_NORMALIZATION_RULE,
        "add_dummy_prefix": True,
        "remove_extra_whitespaces": True,
        "escape_whitespaces": True,
        "normalization_rule_tsv": "",
    }
    expected_normalizer_fields = set(expected_normalizer) | {"precompiled_charsmap"}
    if set(normalizer) != expected_normalizer_fields:
        raise SystemExit(
            "production_v2 NormalizerSpec field set changed; review SentencePiece protobuf: "
            f"actual={sorted(normalizer)}"
        )
    normalizer_mismatches = {
        name: {"expected": expected, "actual": normalizer.get(name)}
        for name, expected in expected_normalizer.items()
        if _different(normalizer.get(name), expected)
    }
    charsmap = normalizer.get("precompiled_charsmap")
    expected_charsmap = {
        "size_bytes": PRODUCTION_V2_NMT_NFKC_CHARS_MAP_SIZE,
        "sha256": PRODUCTION_V2_NMT_NFKC_CHARS_MAP_SHA256,
    }
    if charsmap != expected_charsmap:
        normalizer_mismatches["precompiled_charsmap"] = {
            "expected": expected_charsmap,
            "actual": charsmap,
        }
    if normalizer_mismatches:
        raise SystemExit(
            "production_v2 NormalizerSpec mismatch: "
            + json.dumps(normalizer_mismatches, ensure_ascii=False, sort_keys=True)
        )

    if len(model.pieces) != expected_vocab_size:
        raise SystemExit(
            f"production_v2 piece count mismatch: expected {expected_vocab_size}, got {len(model.pieces)}"
        )
    expected_meta_types = (model_pb2.ModelProto.SentencePiece.UNKNOWN,) + (
        model_pb2.ModelProto.SentencePiece.CONTROL,
    ) * 3
    for token_id, (piece_text, piece_type) in enumerate(
        zip(PRODUCTION_V2_META_PIECES, expected_meta_types, strict=True)
    ):
        piece = model.pieces[token_id]
        if piece.piece != piece_text or piece.type != piece_type:
            raise SystemExit(
                "production_v2 meta-piece mismatch at ID "
                f"{token_id}: expected ({piece_text!r}, {piece_type}), "
                f"got ({piece.piece!r}, {piece.type})"
            )

    byte_pieces = [piece for piece in model.pieces if piece.type == model_pb2.ModelProto.SentencePiece.BYTE]
    expected_byte_pieces = [f"<0x{value:02X}>" for value in range(PRODUCTION_V2_BYTE_PIECE_COUNT)]
    if [piece.piece for piece in byte_pieces] != expected_byte_pieces:
        raise SystemExit(
            "production_v2 must contain exactly the ordered 256 SentencePiece byte-fallback pieces"
        )
    forbidden_piece_types = {
        model_pb2.ModelProto.SentencePiece.USER_DEFINED,
        model_pb2.ModelProto.SentencePiece.UNUSED,
    }
    if any(piece.type in forbidden_piece_types for piece in model.pieces):
        raise SystemExit("production_v2 model contains user-defined or unused pieces")

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if int(processor.get_piece_size()) != expected_vocab_size:
        raise SystemExit("SentencePiece processor and protobuf disagree on piece count")
    for token_id, piece_text in enumerate(PRODUCTION_V2_META_PIECES):
        if processor.id_to_piece(token_id) != piece_text:
            raise SystemExit(f"SentencePiece processor meta ID {token_id} does not match {piece_text}")

    piece_type_counts: dict[str, int] = {}
    type_descriptor = model_pb2.ModelProto.SentencePiece.Type
    for piece in model.pieces:
        type_name = type_descriptor.Name(piece.type)
        piece_type_counts[type_name] = piece_type_counts.get(type_name, 0) + 1
    effective: dict[str, Any] = {
        "schema": "production_sentencepiece_unigram_v2/v1",
        "sentencepiece_version": str(spm.__version__),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_size_bytes": len(model_bytes),
        "piece_count": len(model.pieces),
        "piece_type_counts": dict(sorted(piece_type_counts.items())),
        "trainer_spec": trainer,
        "normalizer_spec": normalizer,
        "inference": {
            "segmentation": "one_best",
            "enable_sampling": False,
            "nbest_size": 0,
            "alpha": 0.0,
        },
    }
    effective["effective_config_sha256"] = _sha256_payload(effective)
    return effective


def production_v2_trainer_kwargs(*, vocab_size: int, num_threads: int) -> dict[str, Any]:
    """Return the explicit trainer arguments shared by production and tests."""

    return {
        "model_type": "unigram",
        "vocab_size": vocab_size,
        "character_coverage": PRODUCTION_V2_CHARACTER_COVERAGE,
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "hard_vocab_limit": True,
        "train_extremely_large_corpus": True,
        "max_sentence_length": 262_144,
        "num_threads": max(1, int(num_threads)),
        "normalization_rule_name": PRODUCTION_V2_NORMALIZATION_RULE,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "pad_piece": "<pad>",
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "byte_fallback": True,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        "split_by_number": True,
        "split_digits": False,
        "treat_whitespace_as_suffix": False,
        "allow_whitespace_only_pieces": False,
        "max_sentencepiece_length": 16,
        "vocabulary_output_piece_score": True,
        "use_all_vocab": False,
        "user_defined_symbols": [],
        "control_symbols": [],
    }


__all__ = [
    "PRODUCTION_V2_BYTE_PIECE_COUNT",
    "PRODUCTION_V2_CHARACTER_COVERAGE",
    "PRODUCTION_V2_META_IDS",
    "PRODUCTION_V2_META_PIECES",
    "PRODUCTION_V2_NMT_NFKC_CHARS_MAP_SHA256",
    "PRODUCTION_V2_NMT_NFKC_CHARS_MAP_SIZE",
    "PRODUCTION_V2_NORMALIZATION_RULE",
    "audit_production_v2_model",
    "inspect_sentence_jsonl",
    "iter_sentence_jsonl",
    "production_v2_trainer_kwargs",
    "protobuf_effective_spec",
]
