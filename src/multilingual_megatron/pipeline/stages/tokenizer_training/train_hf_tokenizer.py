#!/usr/bin/env python3
"""Train a Hugging Face-compatible tokenizer from frozen JSONL input.

The pipeline uses this as the tokenizer-training bridge before Megatron GPT
preprocessing.  Legacy recipes read sampled document shards.  The explicit
production-v2 recipe instead consumes the separately built, train-only,
one-sentence-per-line artifact and verifies its frozen hash before training.

The default study path trains tokenizer recipes from scratch. Legacy modes that
copy or retrain from an existing Hugging Face tokenizer are kept only for
explicit compatibility tests.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .config import (
    PRODUCTION_V2_NORMALIZER,
    PRODUCTION_V2_SENTENCE_COUNT,
    PRODUCTION_V2_VOCAB_SIZE,
)
from .sentencepiece_v2 import (
    audit_production_v2_model,
    inspect_sentence_jsonl,
    iter_sentence_jsonl,
    production_v2_trainer_kwargs,
)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _discover_jsonl(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.rglob("*.jsonl"))
    paths.extend(sorted(input_dir.rglob("*.jsonl.gz")))
    return paths


def _extract_text(raw: str, text_key: str) -> str:
    text = ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        value = obj.get(text_key)
        if isinstance(value, str):
            text = value.strip()
    else:
        text = raw
    return text


def _count_texts(path: Path, text_key: str) -> int:
    count = 0
    with _open_text(path) as handle:
        for line in handle:
            raw = line.strip()
            if raw and _extract_text(raw, text_key):
                count += 1
    return count


def _iter_texts(
    input_dir: Path,
    text_key: str,
    max_samples: int | None,
    sample_fraction: float | None,
) -> Iterable[str]:
    seen = 0
    for path in _discover_jsonl(input_dir):
        per_file_remaining: int | None = None
        if sample_fraction is not None:
            total = _count_texts(path, text_key)
            per_file_remaining = max(1, math.ceil(total * sample_fraction)) if total > 0 else 0
        with _open_text(path) as handle:
            for line in handle:
                if per_file_remaining is not None and per_file_remaining <= 0:
                    break
                raw = line.strip()
                if not raw:
                    continue
                text = _extract_text(raw, text_key)
                if not text:
                    continue
                yield text
                if per_file_remaining is not None:
                    per_file_remaining -= 1
                seen += 1
                if max_samples is not None and seen >= max_samples:
                    return


def _iter_text_segments(
    input_dir: Path,
    text_key: str,
    max_samples: int | None,
    sample_fraction: float | None,
    *,
    max_chars: int,
) -> Iterable[str]:
    for text in _iter_texts(input_dir, text_key, max_samples, sample_fraction):
        if len(text) <= max_chars:
            yield text
            continue
        for start in range(0, len(text), max_chars):
            segment = text[start : start + max_chars].strip()
            if segment:
                yield segment


def _load_base_tokenizer(
    *,
    base_model: str,
    fallback_base_model: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[object, str, bool]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return tokenizer, base_model, False
    except Exception:
        if not fallback_base_model or fallback_base_model == base_model:
            raise
    tokenizer = AutoTokenizer.from_pretrained(
        fallback_base_model,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return tokenizer, fallback_base_model, True


def _special_tokens(args: argparse.Namespace) -> list[str]:
    tokens = []
    for value in (args.unk_token, args.bos_token, args.eos_token, args.pad_token):
        if value and value not in tokens:
            tokens.append(value)
    for value in args.special_token or []:
        if value and value not in tokens:
            tokens.append(value)
    return tokens


def _normalizer(kind: str):
    if kind == "none":
        return None
    if kind == "nfkc":
        return normalizers.NFKC()
    if kind == "nfc":
        return normalizers.NFC()
    raise SystemExit(f"Unsupported tokenizer normalizer: {kind}")


def _configure_pretokenizer(tokenizer: Tokenizer, kind: str) -> None:
    if kind == "byte_level":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return
    if kind == "metaspace":
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        tokenizer.decoder = decoders.Metaspace()
        return
    if kind == "whitespace":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return
    raise SystemExit(f"Unsupported tokenizer pre_tokenizer: {kind}")


def _sentencepiece_normalizer(kind: str) -> str:
    if kind == "none":
        return "identity"
    if kind == "nfkc":
        return "nfkc"
    if kind == "nmt_nfkc":
        return "nmt_nfkc"
    if kind == "nfc":
        raise SystemExit("SentencePiece tokenizer training does not support normalizer=nfc")
    raise SystemExit(f"Unsupported tokenizer normalizer: {kind}")


def _validate_production_v2_args(args: argparse.Namespace) -> None:
    expected: dict[str, tuple[Any, Any]] = {
        "mode": (args.mode, "train_unigram"),
        "vocab_size": (int(args.vocab_size), PRODUCTION_V2_VOCAB_SIZE),
        "normalizer": (args.normalizer, PRODUCTION_V2_NORMALIZER),
        "pre_tokenizer": (args.pre_tokenizer, "metaspace"),
        "unk_token": (args.unk_token, "<unk>"),
        "bos_token": (args.bos_token, "<s>"),
        "eos_token": (args.eos_token, "</s>"),
        "pad_token": (args.pad_token, "<pad>"),
        "max_samples": (int(args.max_samples), 0),
        "max_sample_fraction": (float(args.max_sample_fraction), 0.0),
        "expected_sentence_count": (
            int(args.production_v2_expected_sentence_count),
            PRODUCTION_V2_SENTENCE_COUNT,
        ),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if args.special_token:
        mismatches.append("--special-token is forbidden")
    if mismatches:
        raise SystemExit("production_v2 tokenizer argument mismatch: " + "; ".join(mismatches))


def _train_sentencepiece_unigram(
    args: argparse.Namespace,
    input_dir: Path | None,
    output_dir: Path,
) -> tuple[int, Path, dict[str, Any] | None]:
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mode=train_unigram requires sentencepiece. Install it in the pipeline environment."
        ) from exc

    if str(args.pre_tokenizer) != "metaspace":
        raise SystemExit("mode=train_unigram expects --pre-tokenizer metaspace")

    production_v2 = bool(str(args.production_v2_sentence_jsonl).strip())
    model_prefix = output_dir / "tokenizer"
    model_path = output_dir / "tokenizer.model"
    special_tokens = _special_tokens(args)
    user_defined_symbols = [
        token
        for token in (args.special_token or [])
        if token and token not in {args.unk_token, args.bos_token, args.eos_token, args.pad_token}
    ]

    production_details: dict[str, Any] | None = None
    if production_v2:
        _validate_production_v2_args(args)
        sentence_jsonl = Path(args.production_v2_sentence_jsonl).resolve()
        source_artifact = inspect_sentence_jsonl(
            sentence_jsonl,
            text_key=str(args.text_key),
            expected_sha256=str(args.production_v2_sentence_jsonl_sha256),
            expected_sentence_count=int(args.production_v2_expected_sentence_count),
        )
        texts = iter(
            tqdm(
                iter_sentence_jsonl(sentence_jsonl, text_key=str(args.text_key)),
                total=int(args.production_v2_expected_sentence_count),
                desc="tokenizer_train",
                unit="sentences",
            )
        )
        trainer_kwargs = production_v2_trainer_kwargs(
            vocab_size=int(args.vocab_size),
            num_threads=int(args.num_threads),
        )
        spm.SentencePieceTrainer.train(
            sentence_iterator=texts,
            model_prefix=str(model_prefix),
            **trainer_kwargs,
        )
        effective_config = audit_production_v2_model(
            model_path,
            expected_vocab_size=int(args.vocab_size),
            expected_num_threads=max(1, int(args.num_threads)),
            expected_model_prefix=str(model_prefix),
        )
        production_details = {
            "enabled": True,
            "source_artifact": source_artifact,
            "effective_sentencepiece_config": effective_config,
            "tokenizer_model_sha256": effective_config["model_sha256"],
            "effective_config_sha256": effective_config["effective_config_sha256"],
        }
    else:
        if input_dir is None:
            raise SystemExit("mode=train_unigram requires --input-dir unless production_v2 is enabled")
        max_samples = int(args.max_samples)
        max_samples_value = max_samples if max_samples > 0 else None
        sample_fraction = float(args.max_sample_fraction)
        sample_fraction_value = sample_fraction if sample_fraction > 0.0 else None
        texts = iter(
            tqdm(
                _iter_text_segments(
                    input_dir,
                    str(args.text_key),
                    max_samples_value,
                    sample_fraction_value,
                    max_chars=60000,
                ),
                desc="tokenizer_train",
                unit="segments",
            )
        )
        spm.SentencePieceTrainer.train(
            sentence_iterator=texts,
            model_prefix=str(model_prefix),
            model_type="unigram",
            vocab_size=int(args.vocab_size),
            character_coverage=0.99995,
            input_sentence_size=0,
            shuffle_input_sentence=True,
            hard_vocab_limit=False,
            train_extremely_large_corpus=True,
            max_sentence_length=262144,
            num_threads=max(1, int(args.num_threads)),
            normalization_rule_name=_sentencepiece_normalizer(str(args.normalizer)),
            unk_piece=str(args.unk_token),
            bos_piece=str(args.bos_token),
            eos_piece=str(args.eos_token),
            pad_piece=str(args.pad_token),
            unk_id=0 if args.unk_token else -1,
            bos_id=1 if args.bos_token else -1,
            eos_id=2 if args.eos_token else -1,
            pad_id=3 if args.pad_token else -1,
            user_defined_symbols=user_defined_symbols,
        )
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    tokenizer_config = {
        "tokenizer_type": "SentencePieceTokenizer",
        "model_type": "unigram",
        "normalizer": str(args.normalizer),
        "pre_tokenizer": str(args.pre_tokenizer),
        "special_tokens": special_tokens,
        "production_v2": production_v2,
    }
    if production_details is not None:
        tokenizer_config["model_sha256"] = production_details["tokenizer_model_sha256"]
        tokenizer_config["effective_config_sha256"] = production_details["effective_config_sha256"]
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return int(processor.get_piece_size()), model_path, production_details


def _train_scratch_tokenizer(
    args: argparse.Namespace,
    input_dir: Path | None,
    output_dir: Path,
) -> tuple[int, Path, dict[str, Any] | None]:
    max_samples = int(args.max_samples)
    max_samples_value = max_samples if max_samples > 0 else None
    sample_fraction = float(args.max_sample_fraction)
    sample_fraction_value = sample_fraction if sample_fraction > 0.0 else None
    special_tokens = _special_tokens(args)

    if args.mode == "train_bpe":
        norm = _normalizer(str(args.normalizer))
        tokenizer = Tokenizer(models.BPE(unk_token=args.unk_token))
        if norm is not None:
            tokenizer.normalizer = norm
        _configure_pretokenizer(tokenizer, str(args.pre_tokenizer))
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet() if str(args.pre_tokenizer) == "byte_level" else []
        trainer = trainers.BpeTrainer(
            vocab_size=int(args.vocab_size),
            min_frequency=int(args.min_frequency),
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
        )
    elif args.mode == "train_unigram":
        return _train_sentencepiece_unigram(args, input_dir, output_dir)
    else:
        raise SystemExit(f"Unsupported scratch tokenizer mode: {args.mode}")

    if input_dir is None:
        raise SystemExit("mode=train_bpe requires --input-dir")
    texts = _iter_texts(input_dir, str(args.text_key), max_samples_value, sample_fraction_value)
    tokenizer.train_from_iterator(tqdm(texts, desc="tokenizer_train", unit="docs"), trainer=trainer)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=args.unk_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        additional_special_tokens=[t for t in (args.special_token or []) if t not in {args.unk_token, args.bos_token, args.eos_token, args.pad_token}],
    )
    hf_tokenizer.save_pretrained(output_dir)
    return int(len(hf_tokenizer)), output_dir, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--fallback-base-model", default="")
    parser.add_argument("--mode", choices=["train_bpe", "train_unigram", "train_new", "copy_base"], default="train_bpe")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-sample-fraction", type=float, default=0.0)
    parser.add_argument("--normalizer", choices=["none", "nfkc", "nmt_nfkc", "nfc"], default="nfkc")
    parser.add_argument("--pre-tokenizer", choices=["byte_level", "metaspace", "whitespace"], default="byte_level")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--unk-token", default="<unk>")
    parser.add_argument("--bos-token", default="<s>")
    parser.add_argument("--eos-token", default="</s>")
    parser.add_argument("--pad-token", default="<pad>")
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--production-v2-sentence-jsonl", default="")
    parser.add_argument("--production-v2-sentence-jsonl-sha256", default="")
    parser.add_argument(
        "--production-v2-expected-sentence-count",
        type=int,
        default=PRODUCTION_V2_SENTENCE_COUNT,
    )
    args = parser.parse_args()
    args.num_threads = max(1, int(args.num_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)

    production_v2 = bool(str(args.production_v2_sentence_jsonl).strip())
    if production_v2:
        _validate_production_v2_args(args)
    input_dir = Path(args.input_dir).resolve() if str(args.input_dir).strip() else None
    output_dir = Path(args.output_dir).resolve()
    jsonl_paths: list[Path] = []
    if not production_v2:
        if input_dir is None or not input_dir.is_dir():
            raise SystemExit(f"tokenizer_train: missing input directory: {input_dir}")
        jsonl_paths = _discover_jsonl(input_dir)
        if not jsonl_paths:
            raise SystemExit(f"tokenizer_train: no .jsonl or .jsonl.gz files found under {input_dir}")

    max_samples = int(args.max_samples)
    max_sample_fraction = float(args.max_sample_fraction)
    if max_sample_fraction < 0.0 or max_sample_fraction > 1.0:
        raise SystemExit("--max-sample-fraction must be in [0.0, 1.0]")
    max_samples_value = max_samples if max_samples > 0 else None
    max_sample_fraction_value = max_sample_fraction if max_sample_fraction > 0.0 else None
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_model = ""
    used_fallback = False
    tokenizer_model_path = output_dir
    actual_vocab_size = int(args.vocab_size)
    output_tokenizer_type = "HuggingFaceTokenizer"
    production_details: dict[str, Any] | None = None
    if args.mode in {"train_bpe", "train_unigram"}:
        actual_vocab_size, tokenizer_model_path, production_details = _train_scratch_tokenizer(
            args,
            input_dir,
            output_dir,
        )
        if args.mode == "train_bpe":
            saved_tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
            actual_vocab_size = int(len(saved_tokenizer))
        else:
            output_tokenizer_type = "SentencePieceTokenizer"
    else:
        if not str(args.base_model).strip():
            raise SystemExit(f"mode={args.mode} requires --base-model")
        base_tokenizer, resolved_base_model, used_fallback = _load_base_tokenizer(
            base_model=str(args.base_model),
            fallback_base_model=str(args.fallback_base_model).strip(),
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
        if args.mode == "copy_base":
            base_tokenizer.save_pretrained(output_dir)
        else:
            if not hasattr(base_tokenizer, "train_new_from_iterator"):
                raise SystemExit(
                    f"Tokenizer class {base_tokenizer.__class__.__name__} does not support "
                    "train_new_from_iterator. Use a scratch mode or mode=copy_base only for "
                    "explicit compatibility tests."
                )
            assert input_dir is not None
            texts = _iter_texts(input_dir, str(args.text_key), max_samples_value, max_sample_fraction_value)
            new_tokenizer = base_tokenizer.train_new_from_iterator(
                tqdm(texts, desc="tokenizer_train", unit="docs"),
                vocab_size=int(args.vocab_size),
            )
            new_tokenizer.save_pretrained(output_dir)
        saved_tokenizer = AutoTokenizer.from_pretrained(
            output_dir,
            use_fast=True,
            trust_remote_code=bool(args.trust_remote_code),
        )
        actual_vocab_size = int(len(saved_tokenizer))
    result = {
        "schema_version": 2 if production_details is not None else 1,
        "input_dir": str(input_dir) if input_dir is not None else "",
        "output_dir": str(output_dir),
        "tokenizer_model": str(tokenizer_model_path),
        "tokenizer_type": output_tokenizer_type,
        "base_model": str(args.base_model),
        "resolved_base_model": resolved_base_model,
        "fallback_base_model": str(args.fallback_base_model).strip(),
        "used_fallback": used_fallback,
        "mode": str(args.mode),
        "normalizer": str(args.normalizer),
        "pre_tokenizer": str(args.pre_tokenizer),
        "min_frequency": int(args.min_frequency),
        "special_tokens": _special_tokens(args),
        "unk_token": str(args.unk_token),
        "bos_token": str(args.bos_token),
        "eos_token": str(args.eos_token),
        "pad_token": str(args.pad_token),
        "requested_vocab_size": int(args.vocab_size),
        "vocab_size": actual_vocab_size,
        "text_key": str(args.text_key),
        "max_samples": max_samples_value,
        "max_sample_fraction": max_sample_fraction_value,
        "source_files": [str(path) for path in jsonl_paths],
        "num_threads": int(args.num_threads),
        "production_v2": (
            {
                "enabled": True,
                "source_artifact": production_details["source_artifact"],
            }
            if production_details is not None
            else None
        ),
    }
    if production_details is not None:
        result["source_files"] = [production_details["source_artifact"]["path"]]
        result["source_artifact_sha256"] = production_details["source_artifact"]["sha256"]
        result["tokenizer_model_sha256"] = production_details["tokenizer_model_sha256"]
        result["effective_sentencepiece_config"] = production_details[
            "effective_sentencepiece_config"
        ]
        result["effective_config_sha256"] = production_details["effective_config_sha256"]
    (output_dir / "tokenizer_training_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] trained tokenizer at {tokenizer_model_path} vocab_size={actual_vocab_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
