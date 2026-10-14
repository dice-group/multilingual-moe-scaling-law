from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_with_alias(
    *,
    out: dict[str, Any],
    raw: dict[str, Any],
    canonical: tuple[str, ...],
    aliases: tuple[str, ...],
    default: Any = None,
) -> Any:
    canonical_value = _nested_get(raw, canonical)
    alias_values = [(alias, raw[alias]) for alias in aliases if alias in raw]
    alias_value = alias_values[0][1] if alias_values else None
    if canonical_value is None and alias_value is None:
        value = default
    elif canonical_value is None:
        value = alias_value
    else:
        value = canonical_value
        if alias_value is not None and alias_value != canonical_value:
            dotted = ".".join(canonical)
            alias_names = ", ".join(alias for alias, _value in alias_values)
            raise SystemExit(
                f"tokenization config conflict for {dotted}: canonical value and aliases "
                f"({alias_names}) disagree"
            )
    current = out
    for key in canonical[:-1]:
        current = current.setdefault(key, {})
    current[canonical[-1]] = value
    return value


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


@dataclass(frozen=True, slots=True)
class MegatronTokenizationConfig:
    backend: str
    output_dir: Path
    tokenizer_type: str
    tokenizer_model: str
    trust_remote_code: bool
    append_eod: bool
    mode: str
    array_ranks: int
    array_concurrency: int
    launcher: str
    python_executable: str
    workers: int
    workers_per_prefix: int
    partitions_per_prefix: int
    language_jobs: int
    sample_fraction: float
    max_samples: int
    split_enabled: bool
    train_fraction: float
    validation_fraction: float
    split_seed: int
    blend_weight_policy: str
    skip_existing: bool
    keep_work_dir: bool
    validation_enabled: bool
    parallel_check_ranks: int

    @property
    def array_spec(self) -> str:
        base = f"0-{max(0, self.array_ranks - 1)}"
        if self.array_concurrency > 0:
            return f"{base}%{self.array_concurrency}"
        return base

    @property
    def uses_megatron_array(self) -> bool:
        return self.backend == "megatron_gpt" and self.mode in {"slurm_array", "local_array"}


def resolve_megatron_tokenization_config(
    raw_cfg: dict[str, Any],
    *,
    output_dir: Path,
    require_tokenizer_model: bool = True,
) -> MegatronTokenizationConfig:
    """Normalize tokenization config while preserving current flat-key profiles."""

    normalized: dict[str, Any] = {}
    backend = str(raw_cfg.get("backend", "megatron_gpt")).strip().lower()

    tokenizer_model = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("tokenizer", "model"),
        aliases=("model_name",),
        default="",
    )
    tokenizer_type = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("tokenizer", "type"),
        aliases=("tokenizer_type",),
        default="HuggingFaceTokenizer",
    )
    trust_remote_code = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("tokenizer", "trust_remote_code"),
        aliases=("trust_remote_code",),
        default=False,
    )
    append_eod = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("tokenizer", "append_eod"),
        aliases=("append_eod",),
        default=True,
    )

    explicit_mode = _nested_get(raw_cfg, ("execution", "mode"))
    array_enabled = raw_cfg.get("megatron_slurm_array_enabled", raw_cfg.get("slurm_array_enabled"))
    if explicit_mode is None:
        mode = "slurm_array" if _as_bool(array_enabled) else "single"
    else:
        mode = str(explicit_mode).strip().lower()
    if mode == "array":
        mode = "slurm_array"
    if mode not in {"single", "slurm_array", "local_array"}:
        raise SystemExit(f"Unsupported tokenization.execution.mode: {mode!r}")

    array_ranks = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "array_ranks"),
        aliases=("megatron_slurm_array_ranks", "slurm_array_ranks"),
        default=1,
    )
    array_concurrency = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "array_concurrency"),
        aliases=("megatron_slurm_array_concurrency", "slurm_array_concurrency"),
        default=0,
    )
    launcher = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "launcher"),
        aliases=("launcher",),
        default="python",
    )
    python_executable = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "python_executable"),
        aliases=("python_executable",),
        default="python3",
    )
    workers = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "total_cpus"),
        aliases=("workers", "num_proc"),
        default=4,
    )
    workers_per_prefix = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "workers_per_prefix"),
        aliases=("workers_per_language",),
        default=0,
    )
    partitions_per_prefix = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "partitions_per_prefix"),
        aliases=("partitions_per_language",),
        default=0,
    )
    language_jobs = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("execution", "language_jobs"),
        aliases=("language_jobs",),
        default=0,
    )

    sample_fraction = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("sampling", "sample_fraction"),
        aliases=("sample_fraction",),
        default=1.0,
    )
    max_samples = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("sampling", "max_samples"),
        aliases=("max_samples",),
        default=0,
    )

    split_enabled = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("split", "enabled"),
        aliases=("physical_split_enabled",),
        default=False,
    )
    train_fraction = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("split", "train_fraction"),
        aliases=("train_fraction",),
        default=0.98,
    )
    validation_fraction = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("split", "validation_fraction"),
        aliases=("valid_fraction", "validation_fraction", "eval_fraction"),
        default=0.02,
    )
    split_seed = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("split", "seed"),
        aliases=("split_seed", "eval_seed"),
        default=42,
    )

    blend_weight_policy = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("output", "blend_weight_policy"),
        aliases=("blend_weight_policy",),
        default="equal",
    )
    skip_existing = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("output", "skip_existing"),
        aliases=("skip_if_output_exists",),
        default=False,
    )
    keep_work_dir = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("output", "keep_work_dir"),
        aliases=("keep_work_dir",),
        default=False,
    )
    validation_enabled = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("validation", "enabled"),
        aliases=("validation_enabled",),
        default=True,
    )
    parallel_check_ranks = _set_with_alias(
        out=normalized,
        raw=raw_cfg,
        canonical=("validation", "parallel_check_ranks"),
        aliases=("check_parallel_ranks",),
        default=0,
    )

    if backend == "megatron_gpt" and require_tokenizer_model and not str(tokenizer_model).strip():
        raise SystemExit("tokenization.tokenizer.model or tokenization.model_name must be set")

    resolved_validation_fraction = _as_float(validation_fraction, 0.02)
    resolved_train_fraction = _as_float(train_fraction, 1.0 - resolved_validation_fraction)
    if _as_bool(split_enabled):
        total = resolved_train_fraction + resolved_validation_fraction
        if total <= 0:
            raise SystemExit("tokenization split fractions must sum to a positive value")
        resolved_train_fraction = resolved_train_fraction / total
        resolved_validation_fraction = resolved_validation_fraction / total
    else:
        resolved_train_fraction = 1.0
        resolved_validation_fraction = 0.0

    return MegatronTokenizationConfig(
        backend=backend,
        output_dir=output_dir,
        tokenizer_type=str(tokenizer_type),
        tokenizer_model=str(tokenizer_model),
        trust_remote_code=_as_bool(trust_remote_code),
        append_eod=_as_bool(append_eod),
        mode=mode,
        array_ranks=max(1, _as_int(array_ranks, 1)),
        array_concurrency=max(0, _as_int(array_concurrency, 0)),
        launcher=str(launcher),
        python_executable=str(python_executable),
        workers=max(1, _as_int(workers, 4)),
        workers_per_prefix=max(0, _as_int(workers_per_prefix, 0)),
        partitions_per_prefix=max(0, _as_int(partitions_per_prefix, 0)),
        language_jobs=max(0, _as_int(language_jobs, 0)),
        sample_fraction=max(0.0, min(1.0, _as_float(sample_fraction, 1.0))),
        max_samples=max(0, _as_int(max_samples, 0)),
        split_enabled=_as_bool(split_enabled),
        train_fraction=resolved_train_fraction,
        validation_fraction=resolved_validation_fraction,
        split_seed=_as_int(split_seed, 42),
        blend_weight_policy=str(blend_weight_policy),
        skip_existing=_as_bool(skip_existing),
        keep_work_dir=_as_bool(keep_work_dir),
        validation_enabled=_as_bool(validation_enabled),
        parallel_check_ranks=max(0, _as_int(parallel_check_ranks, 0)),
    )
