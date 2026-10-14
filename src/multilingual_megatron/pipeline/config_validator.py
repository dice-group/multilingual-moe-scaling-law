import argparse
from pathlib import Path
from typing import Any

from .config_loader import load_pipeline_config as _shared_load_pipeline_config

"""validator for proof pipeline config JSON, pointer JSON, or complete dirs"""

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "defaults",
    "profiles",
)

REQUIRED_MERGED_PROFILE_KEYS = (
    "output_root",
    "language_map_path",
    "sample",
    "tokenization",
    "training",
    "summary",
    "benchmark",
)

ALLOWED_RUN_KINDS = {"megatron_bridge", "bridge"}
ALLOWED_SAMPLE_SOURCE_TYPES = {"hf", "json", "fineweb_datatrove", "corpus_blend"}
ALLOWED_FLORES_MODES = {"both", "into_en", "from_en"}
ALLOWED_CHECKPOINT_CONVERSION_MODES = {"manifest", "command", "bridge_hf_export"}
ALLOWED_BENCHMARK_BACKENDS = {"", "bridge_artifacts", "training_artifacts", "bridge_hf_lm_eval", "external"}
ALLOWED_PROFILE_EXTRA_KEYS = {"run_naming", "logs_dir", "runtime", "execution", "extends", "stage_executors"}
ALLOWED_TRAINING_CALIBRATION_MODES = {"calibrate_then_train", "calibrate_only"}


def _deep_merge(base: Any, override: Any) -> Any:
    def _should_merge_runs_by_id(override_runs: list[Any]) -> bool:
        if not isinstance(override_runs, list) or not override_runs:
            return False
        if any(not isinstance(item, dict) or not str(item.get("id", "")).strip() for item in override_runs):
            return False
        return any("kind" not in item for item in override_runs)

    def _merge_runs_by_id(base_runs: list[Any], override_runs: list[Any]) -> list[Any]:
        if any(not isinstance(item, dict) or not str(item.get("id", "")).strip() for item in override_runs):
            return override_runs
        if any(not isinstance(item, dict) or not str(item.get("id", "")).strip() for item in base_runs):
            return override_runs

        override_by_id = {str(item["id"]).strip(): item for item in override_runs}
        merged_runs: list[Any] = []
        used_ids: set[str] = set()

        for base_item in base_runs:
            run_id = str(base_item["id"]).strip()
            if run_id in override_by_id:
                merged_runs.append(_deep_merge(base_item, override_by_id[run_id]))
                used_ids.add(run_id)
            else:
                merged_runs.append(base_item)

        for override_item in override_runs:
            run_id = str(override_item["id"]).strip()
            if run_id in used_ids:
                continue
            merged_runs.append(override_item)
        return merged_runs

    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                if (
                    key == "runs"
                    and isinstance(merged[key], list)
                    and isinstance(value, list)
                    and _should_merge_runs_by_id(value)
                ):
                    merged[key] = _merge_runs_by_id(merged[key], value)
                else:
                    merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_non_empty_string(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")
        return ""
    return value.strip()


def _check_bool(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{path} must be a boolean")
        return False
    return value


def _check_dict(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _check_list(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list")
        return []
    return value


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _load_proof_pipeline_payload(path: Path) -> dict[str, Any]:
    return _shared_load_pipeline_config(path)


def _validate_training_runs(training: dict[str, Any], path: str, errors: list[str]) -> set[str]:
    run_ids: set[str] = set()
    runs = training.get("runs")
    if not isinstance(runs, list):
        errors.append(f"{path}.runs must be a list")
        return run_ids
    if not runs and bool(training.get("enabled", True)):
        errors.append(f"{path}.runs must not be empty when {path}.enabled is true")
        return run_ids

    for index, run in enumerate(runs):
        run_path = f"{path}.runs[{index}]"
        if not isinstance(run, dict):
            errors.append(f"{run_path} must be an object")
            continue
        run_id = run.get("id")
        kind = run.get("kind")
        run_id_value = _check_non_empty_string(run_id, f"{run_path}.id", errors)
        kind_value = _check_non_empty_string(kind, f"{run_path}.kind", errors)
        if run_id_value:
            if run_id_value in run_ids:
                errors.append(f"duplicate training run id '{run_id_value}' in {path}.runs")
            run_ids.add(run_id_value)
        if kind_value and kind_value not in ALLOWED_RUN_KINDS:
            errors.append(
                f"{run_path}.kind must be one of {sorted(ALLOWED_RUN_KINDS)} (got '{kind_value}')"
            )
        if "calibration" in run:
            _validate_training_calibration(run, run_path, errors)
    return run_ids


def _validate_training_calibration(training: dict[str, Any], path: str, errors: list[str]) -> None:
    calibration = training.get("calibration", {})
    if calibration is None:
        return
    calibration = _check_dict(calibration, f"{path}.calibration", errors)
    if not calibration:
        return
    if "enabled" in calibration:
        _check_bool(calibration.get("enabled"), f"{path}.calibration.enabled", errors)
    if "profile_probes" in calibration:
        _check_bool(calibration.get("profile_probes"), f"{path}.calibration.profile_probes", errors)
    mode = str(calibration.get("mode", "calibrate_then_train")).strip()
    if mode not in ALLOWED_TRAINING_CALIBRATION_MODES:
        errors.append(
            f"{path}.calibration.mode must be one of {sorted(ALLOWED_TRAINING_CALIBRATION_MODES)} "
            f"(got '{mode}')"
        )
    candidates = calibration.get("candidate_micro_batch_sizes", [])
    if candidates is not None:
        candidates = _check_list(candidates, f"{path}.calibration.candidate_micro_batch_sizes", errors)
        for index, value in enumerate(candidates):
            if not _is_int(value) or int(value) <= 0:
                errors.append(f"{path}.calibration.candidate_micro_batch_sizes[{index}] must be a positive integer")
    for key in ("max_micro_batch_size", "probe_train_iters", "gpu_monitor_interval_seconds"):
        value = calibration.get(key)
        if value is not None and (not _is_int(value) or int(value) <= 0):
            errors.append(f"{path}.calibration.{key} must be a positive integer")
    warmup_iters = calibration.get("warmup_iters")
    if warmup_iters is not None and (not _is_int(warmup_iters) or int(warmup_iters) < 0):
        errors.append(f"{path}.calibration.warmup_iters must be a non-negative integer")
    probe_train_iters = calibration.get("probe_train_iters")
    if _is_int(warmup_iters) and _is_int(probe_train_iters) and int(warmup_iters) >= int(probe_train_iters):
        errors.append(f"{path}.calibration.warmup_iters must be smaller than probe_train_iters")
    for key in ("memory_headroom_fraction", "throughput_drop_threshold"):
        value = calibration.get(key)
        if value is not None and (not _is_number(value) or float(value) < 0.0 or float(value) >= 1.0):
            errors.append(f"{path}.calibration.{key} must be a number in [0.0, 1.0)")
    for key in ("gpu_memory_total_gb", "peak_tflops_per_gpu"):
        value = calibration.get(key)
        if value is not None and (not _is_number(value) or float(value) < 0.0):
            errors.append(f"{path}.calibration.{key} must be a non-negative number")
    for key in ("require_global_batch_divisibility", "stop_after_first_oom"):
        if key in calibration:
            _check_bool(calibration.get(key), f"{path}.calibration.{key}", errors)


def _validate_profile(profile_name: str, merged_profile: dict[str, Any], errors: list[str]):
    prefix = f"profiles.{profile_name}"

    for key in REQUIRED_MERGED_PROFILE_KEYS:
        if key not in merged_profile:
            errors.append(f"{prefix} is missing merged required key '{key}'")

    _check_non_empty_string(merged_profile.get("output_root"), f"{prefix}.output_root", errors)
    _check_non_empty_string(merged_profile.get("language_map_path"), f"{prefix}.language_map_path", errors)

    sample = _check_dict(merged_profile.get("sample"), f"{prefix}.sample", errors)
    tokenizer_training = merged_profile.get("tokenizer_training", {})
    if tokenizer_training is not None:
        tokenizer_training = _check_dict(tokenizer_training, f"{prefix}.tokenizer_training", errors)
    tokenization = _check_dict(merged_profile.get("tokenization"), f"{prefix}.tokenization", errors)
    training = _check_dict(merged_profile.get("training"), f"{prefix}.training", errors)
    summary = _check_dict(merged_profile.get("summary"), f"{prefix}.summary", errors)
    benchmark = _check_dict(merged_profile.get("benchmark"), f"{prefix}.benchmark", errors)
    checkpoint_conversion = merged_profile.get("checkpoint_conversion", {})
    if checkpoint_conversion is not None:
        checkpoint_conversion = _check_dict(checkpoint_conversion, f"{prefix}.checkpoint_conversion", errors)
    final_loss = merged_profile.get("final_loss", {})
    if final_loss is not None:
        final_loss = _check_dict(final_loss, f"{prefix}.final_loss", errors)
    runtime = merged_profile.get("runtime", {})
    if runtime is not None:
        runtime_cfg = _check_dict(runtime, f"{prefix}.runtime", errors)
        runtime_env = runtime_cfg.get("env", {})
        if runtime_env is not None:
            _check_dict(runtime_env, f"{prefix}.runtime.env", errors)

    source_type = _check_non_empty_string(sample.get("source_type"), f"{prefix}.sample.source_type", errors)
    if source_type and source_type not in ALLOWED_SAMPLE_SOURCE_TYPES:
        errors.append(
            f"{prefix}.sample.source_type must be one of {sorted(ALLOWED_SAMPLE_SOURCE_TYPES)} "
            f"(got '{source_type}')"
        )
    _check_non_empty_string(sample.get("output_dir"), f"{prefix}.sample.output_dir", errors)

    tokenizer_training_enabled = isinstance(tokenizer_training, dict) and bool(tokenizer_training.get("enabled", False))
    _check_non_empty_string(tokenization.get("output_dir"), f"{prefix}.tokenization.output_dir", errors)
    tokenization_model = tokenization.get("model_name", _nested_get(tokenization, ("tokenizer", "model")))
    if not tokenizer_training_enabled:
        _check_non_empty_string(tokenization_model, f"{prefix}.tokenization.model_name", errors)
    if tokenizer_training_enabled:
        _check_non_empty_string(tokenizer_training.get("output_dir"), f"{prefix}.tokenizer_training.output_dir", errors)
        execution = tokenizer_training.get("execution", {})
        if execution is not None:
            execution = _check_dict(execution, f"{prefix}.tokenizer_training.execution", errors)
            num_threads = execution.get("num_threads", tokenizer_training.get("num_threads"))
        else:
            num_threads = tokenizer_training.get("num_threads")
        if num_threads is not None and (not _is_int(num_threads) or int(num_threads) <= 0):
            errors.append(f"{prefix}.tokenizer_training.num_threads must be a positive integer when set")
        corpus = tokenizer_training.get("corpus", {})
        if corpus is not None:
            corpus = _check_dict(corpus, f"{prefix}.tokenizer_training.corpus", errors)
        max_samples = tokenizer_training.get("max_samples", corpus.get("max_samples", 0) if isinstance(corpus, dict) else 0)
        max_sample_fraction = tokenizer_training.get(
            "max_sample_fraction",
            corpus.get("max_sample_fraction", 0.0) if isinstance(corpus, dict) else 0.0,
        )
        if max_samples is not None and max_samples != 0 and (not _is_int(max_samples) or int(max_samples) < 0):
            errors.append(f"{prefix}.tokenizer_training.max_samples must be a non-negative integer")
        if max_sample_fraction is not None and (
            not _is_number(max_sample_fraction) or float(max_sample_fraction) < 0.0 or float(max_sample_fraction) > 1.0
        ):
            errors.append(f"{prefix}.tokenizer_training.max_sample_fraction must be a number in [0.0, 1.0]")
        if (
            max_samples is not None
            and _is_int(max_samples)
            and int(max_samples) > 0
            and max_sample_fraction is not None
            and _is_number(max_sample_fraction)
            and float(max_sample_fraction) > 0.0
        ):
            errors.append(f"{prefix}.tokenizer_training cannot set both max_samples and max_sample_fraction")
        specs = _check_list(tokenizer_training.get("tokenizers", []), f"{prefix}.tokenizer_training.tokenizers", errors)
        if not specs:
            errors.append(f"{prefix}.tokenizer_training.tokenizers must not be empty when enabled")
        seen_tokenizers: set[str] = set()
        for index, spec in enumerate(specs):
            spec_path = f"{prefix}.tokenizer_training.tokenizers[{index}]"
            if not isinstance(spec, dict):
                errors.append(f"{spec_path} must be an object")
                continue
            tokenizer_id = _check_non_empty_string(spec.get("id"), f"{spec_path}.id", errors)
            production_v2 = spec.get("production_v2")
            production_v2_enabled = isinstance(production_v2, dict) and bool(production_v2.get("enabled", False))
            if production_v2 is not None and not isinstance(production_v2, dict):
                errors.append(f"{spec_path}.production_v2 must be an object when set")
            mode_default = "train_unigram" if production_v2_enabled else tokenizer_training.get("mode", "train_bpe")
            mode = str(spec.get("mode", mode_default)).strip()
            if mode not in {"train_bpe", "train_unigram", "train_new", "copy_base"}:
                errors.append(f"{spec_path}.mode must be one of ['copy_base', 'train_bpe', 'train_new', 'train_unigram']")
            if mode in {"train_new", "copy_base"} and not str(
                spec.get("base_model", tokenizer_training.get("base_model", tokenization_model or ""))
            ).strip():
                errors.append(f"{spec_path}.base_model or {prefix}.tokenizer_training.base_model must be set for mode={mode}")
            spec_max_samples = spec.get("max_samples", max_samples)
            if spec_max_samples is not None and spec_max_samples != 0 and (
                not _is_int(spec_max_samples) or int(spec_max_samples) < 0
            ):
                errors.append(f"{spec_path}.max_samples must be a non-negative integer")
            spec_fraction = spec.get("max_sample_fraction")
            if spec_fraction is not None and (
                not _is_number(spec_fraction) or float(spec_fraction) < 0.0 or float(spec_fraction) > 1.0
            ):
                errors.append(f"{spec_path}.max_sample_fraction must be a number in [0.0, 1.0]")
            if (
                spec_max_samples is not None
                and _is_int(spec_max_samples)
                and int(spec_max_samples) > 0
                and spec_fraction is not None
                and _is_number(spec_fraction)
                and float(spec_fraction) > 0.0
            ):
                errors.append(f"{spec_path} cannot set both max_samples and max_sample_fraction")
            vocab_default = 64_000 if production_v2_enabled else tokenizer_training.get("vocab_size")
            vocab_size = spec.get("vocab_size", vocab_default)
            if not _is_int(vocab_size) or int(vocab_size) <= 0:
                errors.append(f"{spec_path}.vocab_size must be a positive integer")
            model_vocab_size = spec.get(
                "model_vocab_size",
                64_000 if production_v2_enabled else tokenizer_training.get("model_vocab_size", vocab_size),
            )
            if model_vocab_size is not None and (not _is_int(model_vocab_size) or int(model_vocab_size) <= 0):
                errors.append(f"{spec_path}.model_vocab_size must be a positive integer when set")
            if production_v2_enabled:
                assert isinstance(production_v2, dict)
                sentence_path = production_v2.get(
                    "sentence_jsonl_path",
                    production_v2.get("sentence_jsonl"),
                )
                _check_non_empty_string(
                    sentence_path,
                    f"{spec_path}.production_v2.sentence_jsonl_path",
                    errors,
                )
                sentence_hash = str(production_v2.get("sentence_jsonl_sha256", "")).strip().lower()
                if len(sentence_hash) != 64 or any(char not in "0123456789abcdef" for char in sentence_hash):
                    errors.append(
                        f"{spec_path}.production_v2.sentence_jsonl_sha256 must be a 64-character SHA-256"
                    )
                expected_count = production_v2.get("expected_sentence_count", 100_000_000)
                if not _is_int(expected_count) or int(expected_count) != 100_000_000:
                    errors.append(
                        f"{spec_path}.production_v2.expected_sentence_count must equal 100000000"
                    )
                fixed_values = {
                    "mode": (mode, "train_unigram"),
                    "vocab_size": (vocab_size, 64_000),
                    "model_vocab_size": (model_vocab_size, 64_000),
                    "normalizer": (spec.get("normalizer", "nmt_nfkc"), "nmt_nfkc"),
                    "pre_tokenizer": (spec.get("pre_tokenizer", "metaspace"), "metaspace"),
                }
                for field, (actual, expected) in fixed_values.items():
                    if actual != expected:
                        errors.append(
                            f"{spec_path}.{field} must equal {expected!r} for production_v2"
                        )
            if tokenizer_id:
                if tokenizer_id in seen_tokenizers:
                    errors.append(f"duplicate tokenizer id '{tokenizer_id}' in {prefix}.tokenizer_training.tokenizers")
                seen_tokenizers.add(tokenizer_id)

    training_enabled = _check_bool(training.get("enabled", True), f"{prefix}.training.enabled", errors)
    _check_non_empty_string(training.get("output_dir"), f"{prefix}.training.output_dir", errors)
    backend = str(training.get("backend", "megatron_bridge")).strip().lower()
    if backend not in {"megatron_bridge", "bridge"}:
        errors.append(f"{prefix}.training.backend must be megatron_bridge")
    _check_non_empty_string(training.get("bridge_root"), f"{prefix}.training.bridge_root", errors)
    _check_non_empty_string(training.get("recipe"), f"{prefix}.training.recipe", errors)
    run_ids = _validate_training_runs(training, f"{prefix}.training", errors)
    _validate_training_calibration(training, f"{prefix}.training", errors)

    _check_non_empty_string(summary.get("summary_csv"), f"{prefix}.summary.summary_csv", errors)
    _check_non_empty_string(summary.get("gate_json"), f"{prefix}.summary.gate_json", errors)
    baseline_id = ""
    if summary.get("baseline_id") is not None:
        baseline_id = _check_non_empty_string(summary.get("baseline_id"), f"{prefix}.summary.baseline_id", errors)
    if summary.get("primary_metric") is not None:
        _check_non_empty_string(summary.get("primary_metric"), f"{prefix}.summary.primary_metric", errors)
    min_rel_impr = summary.get("min_relative_improvement")
    if not _is_number(min_rel_impr):
        errors.append(f"{prefix}.summary.min_relative_improvement must be numeric")

    if baseline_id and run_ids and baseline_id not in run_ids:
        errors.append(
            f"{prefix}.summary.baseline_id '{baseline_id}' is not present in {prefix}.training.runs ids"
        )

    if isinstance(checkpoint_conversion, dict):
        checkpoint_enabled = checkpoint_conversion.get("enabled", False)
        if checkpoint_enabled is not None:
            _check_bool(checkpoint_enabled, f"{prefix}.checkpoint_conversion.enabled", errors)
        mode = str(checkpoint_conversion.get("mode", "manifest")).strip().lower()
        if mode not in ALLOWED_CHECKPOINT_CONVERSION_MODES:
            errors.append(
                f"{prefix}.checkpoint_conversion.mode must be one of {sorted(ALLOWED_CHECKPOINT_CONVERSION_MODES)} "
                f"(got '{mode}')"
            )
        _check_non_empty_string(
            checkpoint_conversion.get("output_dir"),
            f"{prefix}.checkpoint_conversion.output_dir",
            errors,
        )
        if mode == "command":
            _check_non_empty_string(checkpoint_conversion.get("command"), f"{prefix}.checkpoint_conversion.command", errors)
        if mode == "bridge_hf_export":
            _check_non_empty_string(
                checkpoint_conversion.get("hf_reference_model"),
                f"{prefix}.checkpoint_conversion.hf_reference_model",
                errors,
            )
            conversion_script = str(checkpoint_conversion.get("conversion_script", "")).strip()
            conversion_module = str(checkpoint_conversion.get("conversion_module", "")).strip()
            if not conversion_script and not conversion_module:
                errors.append(
                    f"{prefix}.checkpoint_conversion requires conversion_module or conversion_script "
                    "when mode='bridge_hf_export'"
                )

    if isinstance(final_loss, dict):
        final_loss_enabled = final_loss.get("enabled", False)
        if final_loss_enabled is not None:
            _check_bool(final_loss_enabled, f"{prefix}.final_loss.enabled", errors)
        _check_non_empty_string(final_loss.get("output_dir"), f"{prefix}.final_loss.output_dir", errors)

    benchmark_enabled = _check_bool(benchmark.get("enabled", False), f"{prefix}.benchmark.enabled", errors)
    benchmark_backend = str(benchmark.get("backend", "")).strip().lower()
    if benchmark_backend not in ALLOWED_BENCHMARK_BACKENDS:
        errors.append(
            f"{prefix}.benchmark.backend must be one of {sorted(ALLOWED_BENCHMARK_BACKENDS)} "
            f"(got '{benchmark_backend}')"
        )
    if benchmark_enabled and benchmark_backend not in {"bridge_artifacts", "training_artifacts"}:
        _check_non_empty_string(benchmark.get("tasks_file"), f"{prefix}.benchmark.tasks_file", errors)
    _check_non_empty_string(benchmark.get("output_dir"), f"{prefix}.benchmark.output_dir", errors)
    if bool(benchmark.get("include_flores", True)):
        flores_mode = _check_non_empty_string(benchmark.get("flores_mode"), f"{prefix}.benchmark.flores_mode", errors)
        if flores_mode and flores_mode not in ALLOWED_FLORES_MODES:
            errors.append(
                f"{prefix}.benchmark.flores_mode must be one of {sorted(ALLOWED_FLORES_MODES)} "
                f"(got '{flores_mode}')"
            )

    if benchmark_enabled and not training_enabled:
        errors.append(f"{prefix} stage dependency failed: benchmark.enabled requires training.enabled")
    if benchmark_enabled and not run_ids:
        errors.append(f"{prefix} stage dependency failed: benchmark.enabled requires training.runs")


def validate_proof_pipeline_config(payload: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ["proof config must be a JSON object"]

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in payload:
            errors.append(f"missing required top-level key: {key}")

    schema_version = payload.get("schema_version")
    if not _is_int(schema_version) or int(schema_version) <= 0:
        errors.append("schema_version must be a positive integer")

    defaults = payload.get("defaults")
    profiles = payload.get("profiles")
    if not isinstance(defaults, dict) or not defaults:
        errors.append("defaults must be a non-empty object")
        defaults = {}
    if not isinstance(profiles, dict) or not profiles:
        errors.append("profiles must be a non-empty object")
        profiles = {}

    if not defaults or not profiles:
        return errors

    allowed_profile_keys = set(defaults.keys()) | ALLOWED_PROFILE_EXTRA_KEYS
    resolved_profile_overrides: dict[str, dict[str, Any]] = {}
    failed_profiles: set[str] = set()

    def _resolve_profile_override(name: str, chain: list[str]) -> dict[str, Any] | None:
        if name in resolved_profile_overrides:
            return resolved_profile_overrides[name]
        if name in failed_profiles:
            return None
        if name in chain:
            errors.append(f"profiles.{name} has inheritance cycle: {' -> '.join(chain + [name])}")
            failed_profiles.add(name)
            return None

        profile_override_raw = profiles.get(name)
        if not isinstance(profile_override_raw, dict):
            errors.append(f"profiles.{name} must be an object")
            failed_profiles.add(name)
            return None

        extends_raw = profile_override_raw.get("extends")
        parent_name = ""
        if extends_raw is not None:
            if not isinstance(extends_raw, str) or not extends_raw.strip():
                errors.append(f"profiles.{name}.extends must be a non-empty string")
                failed_profiles.add(name)
                return None
            parent_name = extends_raw.strip()
            if parent_name not in profiles:
                errors.append(f"profiles.{name}.extends references unknown profile '{parent_name}'")
                failed_profiles.add(name)
                return None

        own_override = dict(profile_override_raw)
        own_override.pop("extends", None)
        parent_override: dict[str, Any] = {}
        if parent_name:
            parent_resolved = _resolve_profile_override(parent_name, chain + [name])
            if parent_resolved is None:
                failed_profiles.add(name)
                return None
            parent_override = parent_resolved

        merged_override = _deep_merge(parent_override, own_override)
        if not isinstance(merged_override, dict):
            errors.append(f"profiles.{name} did not merge to an object")
            failed_profiles.add(name)
            return None
        resolved_profile_overrides[name] = merged_override
        return merged_override

    for profile_name, profile_override in sorted(profiles.items()):
        profile_path = f"profiles.{profile_name}"
        if not isinstance(profile_override, dict):
            errors.append(f"{profile_path} must be an object")
            continue

        unknown = sorted(set(profile_override.keys()) - allowed_profile_keys)
        if unknown:
            errors.append(f"{profile_path} has unknown top-level keys: {unknown}")

        for key, value in profile_override.items():
            if key == "extends":
                continue
            default_value = defaults.get(key)
            if isinstance(default_value, dict) and not isinstance(value, dict):
                errors.append(
                    f"{profile_path}.{key} must be an object to merge with defaults.{key}"
                )

        resolved_override = _resolve_profile_override(profile_name, [])
        if resolved_override is None:
            continue
        merged_profile = _deep_merge(defaults, resolved_override)
        if not isinstance(merged_profile, dict):
            errors.append(f"{profile_path} did not merge to an object")
            continue

        _validate_profile(profile_name, merged_profile, errors)

    return errors


def validate_proof_pipeline_config_file(path: Path) -> list[str]:
    try:
        payload = _load_proof_pipeline_payload(path)
    except Exception as exc:
        return [f"invalid proof config ({path}): {exc}"]
    return validate_proof_pipeline_config(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate proof pipeline profile config.")
    parser.add_argument(
        "--config",
        default="configs/proof_pipeline",
        help="Path to proof pipeline bundle directory or config JSON.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    errors = validate_proof_pipeline_config_file(config_path)
    if errors:
        print("[FAIL] Proof pipeline config validation failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print(f"[OK] Proof pipeline config valid: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
