from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any



SUPPORTED_TOKENIZER_MODES = {"train_bpe", "train_unigram", "train_new", "copy_base"}
SCRATCH_TOKENIZER_MODES = {"train_bpe", "train_unigram"}
PRODUCTION_V2_VOCAB_SIZE = 64_000
PRODUCTION_V2_SENTENCE_COUNT = 100_000_000
PRODUCTION_V2_NORMALIZER = "nmt_nfkc"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_present(raw: dict[str, Any], *, canonical: tuple[str, ...], aliases: tuple[str, ...], default: Any) -> Any:
    value = _nested_get(raw, canonical)
    if value is not None:
        return value
    for alias in aliases:
        if alias in raw:
            return raw[alias]
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _tokenization_model_name(tokenize_cfg: dict[str, Any]) -> str:
    nested = _nested_get(tokenize_cfg, ("tokenizer", "model"))
    if nested is not None:
        return str(nested).strip()
    return str(tokenize_cfg.get("model_name", "")).strip()


@dataclass(frozen=True, slots=True)
class ProductionSentencePieceV2Spec:
    sentence_jsonl_path: str
    sentence_jsonl_sha256: str
    expected_sentence_count: int




@dataclass(frozen=True, slots=True)
class TokenizerTrainingSpec:
    id: str
    mode: str
    vocab_size: int
    model_vocab_size: int
    base_model: str
    fallback_base_model: str
    allow_fallback: bool
    trust_remote_code: bool
    local_files_only: bool
    normalizer: str
    pre_tokenizer: str
    min_frequency: int
    unk_token: str
    bos_token: str
    eos_token: str
    pad_token: str
    special_tokens: tuple[str, ...]
    text_key: str
    max_samples: int
    max_sample_fraction: float
    tokenizer_type: str
    family: str
    production_v2: ProductionSentencePieceV2Spec | None


@dataclass(frozen=True, slots=True)
class TokenizerTrainingConfig:
    enabled: bool
    python_executable: str
    parallel_slurm_jobs: bool
    num_threads: int
    specs: tuple[TokenizerTrainingSpec, ...]


def resolve_tokenizer_training_config(
    raw_cfg: dict[str, Any],
    *,
    tokenize_cfg: dict[str, Any],
) -> TokenizerTrainingConfig:
    """Normalize tokenizer-training config while preserving legacy flat profiles."""

    enabled = _as_bool(raw_cfg.get("enabled", False))
    python_executable = str(
        _first_present(
            raw_cfg,
            canonical=("execution", "python_executable"),
            aliases=("python_executable",),
            default="python3",
        )
    ).strip() or "python3"
    parallel_slurm_jobs = _as_bool(
        _first_present(
            raw_cfg,
            canonical=("execution", "parallel_slurm_jobs"),
            aliases=("parallel_slurm_jobs",),
            default=False,
        )
    )
    num_threads = max(
        1,
        _as_int(
            _first_present(raw_cfg, canonical=("execution", "num_threads"), aliases=("num_threads",), default=1),
            1,
        ),
    )

    default_mode = str(raw_cfg.get("mode", "train_bpe")).strip() or "train_bpe"
    default_vocab_size = _as_int(raw_cfg.get("vocab_size"), 32000)
    default_model_vocab_size = _as_int(raw_cfg.get("model_vocab_size"), default_vocab_size)
    default_text_key = str(
        _first_present(raw_cfg, canonical=("corpus", "text_key"), aliases=("text_key",), default="text")
    ).strip() or "text"
    default_max_samples = max(
        0,
        _as_int(
            _first_present(raw_cfg, canonical=("corpus", "max_samples"), aliases=("max_samples",), default=0),
            0,
        ),
    )
    default_max_sample_fraction = _as_float(
        _first_present(
            raw_cfg,
            canonical=("corpus", "max_sample_fraction"),
            aliases=("max_sample_fraction",),
            default=0.0,
        ),
        0.0,
    )
    if not 0.0 <= default_max_sample_fraction <= 1.0:
        raise SystemExit("tokenizer_training.max_sample_fraction must be in [0.0, 1.0]")
    if default_max_samples > 0 and default_max_sample_fraction > 0.0:
        raise SystemExit("tokenizer_training sets both max_samples and max_sample_fraction; choose one")

    default_base_model = str(raw_cfg.get("base_model", _tokenization_model_name(tokenize_cfg))).strip()
    default_fallback = str(raw_cfg.get("fallback_base_model", "")).strip()
    default_allow_fallback = _as_bool(raw_cfg.get("allow_fallback", False))
    default_trust_remote_code = _as_bool(raw_cfg.get("trust_remote_code", False))
    default_local_files_only = _as_bool(raw_cfg.get("local_files_only", False))
    default_normalizer = str(raw_cfg.get("normalizer", "nfkc")).strip() or "nfkc"
    default_pre_tokenizer = str(raw_cfg.get("pre_tokenizer", "byte_level")).strip() or "byte_level"
    default_min_frequency = max(1, _as_int(raw_cfg.get("min_frequency"), 2))
    default_special_tokens = tuple(_as_list(raw_cfg.get("special_tokens", [])))
    default_tokenizer_type = str(tokenize_cfg.get("tokenizer_type", "HuggingFaceTokenizer"))

    raw_specs = raw_cfg.get("tokenizers", [])
    if not isinstance(raw_specs, list) or not raw_specs:
        raw_specs = [
            {
                "id": "default",
                "base_model": default_base_model,
                "vocab_size": default_vocab_size,
                "mode": default_mode,
            }
        ]

    specs: list[TokenizerTrainingSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise SystemExit(f"tokenizer_training.tokenizers[{index}] must be an object")
        tokenizer_id = str(raw.get("id", "")).strip()
        if not tokenizer_id:
            raise SystemExit(f"tokenizer_training.tokenizers[{index}].id must be set")
        if tokenizer_id in seen:
            raise SystemExit(f"duplicate tokenizer id in tokenizer_training.tokenizers: {tokenizer_id}")
        seen.add(tokenizer_id)

        raw_production_v2 = raw.get("production_v2")
        if raw_production_v2 is None:
            raw_production_v2 = raw_cfg.get("production_v2")
        if raw_production_v2 is not None and not isinstance(raw_production_v2, dict):
            raise SystemExit(f"tokenizer '{tokenizer_id}' production_v2 must be an object")
        production_v2_enabled = bool(
            isinstance(raw_production_v2, dict)
            and _as_bool(raw_production_v2.get("enabled", False))
        )


        mode_default = "train_unigram" if production_v2_enabled else default_mode
        mode = str(raw.get("mode", mode_default)).strip() or mode_default
        if mode not in SUPPORTED_TOKENIZER_MODES:
            raise SystemExit(f"Unsupported tokenizer_training mode for '{tokenizer_id}': {mode}")
        base_model = str(raw.get("base_model", default_base_model)).strip()
        if mode not in SCRATCH_TOKENIZER_MODES and not base_model:
            raise SystemExit(f"tokenizer '{tokenizer_id}' mode={mode} requires base_model")

        vocab_default = PRODUCTION_V2_VOCAB_SIZE if production_v2_enabled else default_vocab_size
        model_vocab_default = PRODUCTION_V2_VOCAB_SIZE if production_v2_enabled else default_model_vocab_size
        vocab_size = _as_int(raw.get("vocab_size"), vocab_default)
        model_vocab_size = _as_int(raw.get("model_vocab_size"), _as_int(raw.get("vocab_size"), model_vocab_default))
        max_samples = max(0, _as_int(raw.get("max_samples"), 0 if production_v2_enabled else default_max_samples))
        max_sample_fraction = _as_float(
            raw.get("max_sample_fraction"),
            0.0 if production_v2_enabled else default_max_sample_fraction,
        )
        if not 0.0 <= max_sample_fraction <= 1.0:
            raise SystemExit(f"tokenizer '{tokenizer_id}' max_sample_fraction must be in [0.0, 1.0]")
        if max_samples > 0 and max_sample_fraction > 0.0:
            raise SystemExit(f"tokenizer '{tokenizer_id}' sets both max_samples and max_sample_fraction; choose one")

        production_v2: ProductionSentencePieceV2Spec | None = None
        normalizer_default = PRODUCTION_V2_NORMALIZER if production_v2_enabled else default_normalizer
        pre_tokenizer_default = "metaspace" if production_v2_enabled else default_pre_tokenizer
        special_tokens_default: tuple[str, ...] = () if production_v2_enabled else default_special_tokens
        unk_token = str(raw.get("unk_token", raw_cfg.get("unk_token", "<unk>")))
        bos_token = str(raw.get("bos_token", raw_cfg.get("bos_token", "<s>")))
        eos_token = str(raw.get("eos_token", raw_cfg.get("eos_token", "</s>")))
        pad_token = str(raw.get("pad_token", raw_cfg.get("pad_token", "<pad>")))
        special_tokens = tuple(_as_list(raw.get("special_tokens", special_tokens_default)))
        normalizer = str(raw.get("normalizer", normalizer_default)).strip() or normalizer_default
        pre_tokenizer = str(raw.get("pre_tokenizer", pre_tokenizer_default)).strip() or pre_tokenizer_default

        if production_v2_enabled:
            assert isinstance(raw_production_v2, dict)
            sentence_jsonl_path = str(
                raw_production_v2.get(
                    "sentence_jsonl_path",
                    raw_production_v2.get("sentence_jsonl", ""),
                )
            ).strip()
            sentence_jsonl_sha256 = str(raw_production_v2.get("sentence_jsonl_sha256", "")).strip().lower()
            expected_sentence_count = _as_int(
                raw_production_v2.get("expected_sentence_count"),
                PRODUCTION_V2_SENTENCE_COUNT,
            )
            if not sentence_jsonl_path:
                raise SystemExit(f"tokenizer '{tokenizer_id}' production_v2.sentence_jsonl_path must be set")
            if _SHA256_RE.fullmatch(sentence_jsonl_sha256) is None:
                raise SystemExit(
                    f"tokenizer '{tokenizer_id}' production_v2.sentence_jsonl_sha256 "
                    "must be a 64-character SHA-256"
                )
            expected = {
                "mode": (mode, "train_unigram"),
                "vocab_size": (vocab_size, PRODUCTION_V2_VOCAB_SIZE),
                "model_vocab_size": (model_vocab_size, PRODUCTION_V2_VOCAB_SIZE),
                "normalizer": (normalizer, PRODUCTION_V2_NORMALIZER),
                "pre_tokenizer": (pre_tokenizer, "metaspace"),
                "unk_token": (unk_token, "<unk>"),
                "bos_token": (bos_token, "<s>"),
                "eos_token": (eos_token, "</s>"),
                "pad_token": (pad_token, "<pad>"),
                "expected_sentence_count": (expected_sentence_count, PRODUCTION_V2_SENTENCE_COUNT),
            }
            mismatches = [
                f"{name}={actual!r} (expected {wanted!r})"
                for name, (actual, wanted) in expected.items()
                if actual != wanted
            ]
            if max_samples != 0:
                mismatches.append(f"max_samples={max_samples!r} (expected 0)")
            if max_sample_fraction != 0.0:
                mismatches.append(f"max_sample_fraction={max_sample_fraction!r} (expected 0.0)")
            if special_tokens:
                mismatches.append("special_tokens must be empty; production v2 has exactly four meta tokens")
            if mismatches:
                raise SystemExit(
                    f"tokenizer '{tokenizer_id}' production_v2 configuration mismatch: "
                    + "; ".join(mismatches)
                )
            production_v2 = ProductionSentencePieceV2Spec(
                sentence_jsonl_path=sentence_jsonl_path,
                sentence_jsonl_sha256=sentence_jsonl_sha256,
                expected_sentence_count=expected_sentence_count,
            )


        specs.append(
            TokenizerTrainingSpec(
                id=tokenizer_id,
                mode=mode,
                vocab_size=vocab_size,
                model_vocab_size=model_vocab_size,
                base_model=base_model,
                fallback_base_model=str(raw.get("fallback_base_model", default_fallback)).strip(),
                allow_fallback=_as_bool(raw.get("allow_fallback", default_allow_fallback)),
                trust_remote_code=default_trust_remote_code or _as_bool(raw.get("trust_remote_code", False)),
                local_files_only=default_local_files_only or _as_bool(raw.get("local_files_only", False)),
                normalizer=normalizer,
                pre_tokenizer=pre_tokenizer,
                min_frequency=max(1, _as_int(raw.get("min_frequency"), default_min_frequency)),
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                special_tokens=special_tokens,
                text_key=str(raw.get("text_key", default_text_key)).strip() or default_text_key,
                max_samples=max_samples,
                max_sample_fraction=max_sample_fraction,
                tokenizer_type=str(raw.get("tokenizer_type", default_tokenizer_type)),
                family=str(raw.get("family", "")),
                production_v2=production_v2,
            )
        )

    return TokenizerTrainingConfig(
        enabled=enabled,
        python_executable=python_executable,
        parallel_slurm_jobs=parallel_slurm_jobs,
        num_threads=num_threads,
        specs=tuple(specs),
    )
