"""Native Megatron scoring for frozen lm-eval loglikelihood requests.

The scaling Switch-MoE checkpoints do not have a validated Hugging Face export.
This module therefore keeps prompt construction and metric aggregation in the
pinned lm-evaluation-harness environment, while scoring the resulting frozen
``loglikelihood`` requests with the original Megatron model and checkpoint.

Multiple-choice/reference likelihood and explicitly frozen document token
windows are supported. Generation and unfrozen rolling requests fail closed.
"""


import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from multilingual_megatron.contracts.hashing import (
    canonical_json_sha256 as _payload_sha256,
    sha256_file as _sha256_file,
)
from .native_megatron_batching import AutoBatchPolicy


SCORE_SPEC_SCHEMA = "stage1_native_megatron_benchmark_score_spec/v1"
REQUEST_MANIFEST_SCHEMA = "stage1_lm_eval_frozen_requests/v1"
SCORE_RESULT_SCHEMA = "stage1_native_megatron_benchmark_scores/v1"

RAW_NLL_ABS_TOLERANCE = 1.0e-6
NLL_SUM_RELATIVE_TOLERANCE = 1.0e-7
# Catch score drift without demanding bitwise equality from BF16 MoE kernels.
REPLICA_SCORE_SUM_RELATIVE_TOLERANCE = 1.0e-4
EARLY_REPLICA_REQUESTS = 200
EARLY_REPLICA_SCORE_ABS_TOLERANCE = 1.0e-2


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


def _resolve_path(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _verify_payload_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    expected = _require_sha256(payload.get(field), field=f"{label}.{field}")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    actual = _payload_sha256(unhashed)
    if actual != expected:
        raise ValueError(f"{label} payload hash mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True, slots=True)
class NativeBenchmarkScoreSpec:
    source_path: Path
    source_sha256: str
    run_id: str
    output_dir: Path
    request_manifest_path: Path
    request_manifest_sha256: str
    recipe_name: str
    recipe_overrides: tuple[str, ...]
    model_contract: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_iteration_path: Path
    checkpoint_step: int
    training_run_config_path: Path
    training_run_config_sha256: str
    tokenizer_type: str
    tokenizer_model_path: Path
    tokenizer_model_sha256: str | None
    tokenizer_files_sha256: Mapping[str, str]
    tokenizer_vocab_size: int
    context_token_id: int
    pad_token_id: int
    sequence_length: int
    micro_batch_size: int
    expected_world_size: int
    max_gpu_hours: float
    parity_requests: int
    require_moe_routing_diagnostics: bool
    expected_moe_layers: int | None
    required_gpu_name_substring: str
    moe_reduction_dtype: str = "model"
    batching: AutoBatchPolicy | None = None
    minimum_sequence_width: int = 1
    sequence_width_multiple: int = 1
    moe_compute_dtype: str = "model"


def load_native_benchmark_score_spec(
    path: Path,
    *,
    request_manifest_cache: MutableMapping[
        Path, tuple[str, Mapping[str, Any]]
    ]
    | None = None,
) -> NativeBenchmarkScoreSpec:
    """Load and validate every immutable CPU-side scoring input.

    Campaign preparation may pass a process-local cache so a shared, large
    frozen request manifest is hashed and semantically validated once. Runtime
    callers omit the cache and therefore retain the full fail-closed check for
    every independently scheduled benchmark job.
    """

    source_path = path.resolve()
    payload = _load_json(source_path, label="native benchmark score spec")
    if payload.get("schema") != SCORE_SPEC_SCHEMA:
        raise ValueError(f"unsupported score spec schema: {payload.get('schema')!r}")
    _verify_payload_hash(payload, field="spec_sha256", label="score spec")
    source_sha256 = _sha256_file(source_path)
    base = source_path.parent

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip() or run_id != run_id.strip():
        raise ValueError("run_id must be a non-empty trimmed string")

    output_dir = _resolve_path(base, payload.get("output_dir"), field="output_dir")
    request = payload.get("request_manifest")
    if not isinstance(request, dict):
        raise ValueError("request_manifest must be an object")
    request_path = _resolve_path(base, request.get("path"), field="request_manifest.path")
    request_sha256 = _require_sha256(
        request.get("sha256"), field="request_manifest.sha256"
    )
    cached_request = (
        request_manifest_cache.get(request_path)
        if request_manifest_cache is not None
        else None
    )
    if cached_request is not None:
        cached_sha256, request_payload = cached_request
        if cached_sha256 != request_sha256:
            raise ValueError(
                "score specs disagree on the frozen request manifest hash: "
                f"{cached_sha256} != {request_sha256}"
            )
    else:
        actual_request_sha256 = _sha256_file(request_path)
        if actual_request_sha256 != request_sha256:
            raise ValueError(
                "request manifest file hash mismatch: "
                f"expected {request_sha256}, got {actual_request_sha256}"
            )
        request_payload = _load_json(request_path, label="frozen request manifest")
        if request_payload.get("schema") != REQUEST_MANIFEST_SCHEMA:
            raise ValueError(
                "unsupported request manifest schema: "
                f"{request_payload.get('schema')!r}"
            )
        _verify_payload_hash(
            request_payload, field="manifest_sha256", label="request manifest"
        )
        tasks = request_payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("frozen request manifest contains no task metadata")
        if any(
            not isinstance(item, dict) or item.get("output_type") not in {"multiple_choice", "loglikelihood", "frozen_document_loglikelihood"}
            for item in tasks
        ):
            raise ValueError("native scorer only accepts multiple-choice or reference-loglikelihood tasks")
        reference_tasks = {item["task"] for item in tasks if item["output_type"] in {"loglikelihood", "frozen_document_loglikelihood"}}
        window_tasks = {item["task"] for item in tasks if item["output_type"] == "frozen_document_loglikelihood"}
        if window_tasks and not isinstance(payload.get("token_cache"), dict):
            raise ValueError("frozen document windows require a bound token cache")
        if any(item.get("metric") != "bits_per_byte" for item in tasks if item["task"] in reference_tasks):
            raise ValueError("reference-loglikelihood tasks require bits_per_byte")
        requests = request_payload.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("frozen request manifest contains no requests")
        if any(
            not isinstance(item, dict)
            or item.get("request_type") not in {"loglikelihood", "token_window_loglikelihood"}
            or not isinstance(item.get("request_id"), str)
            or len(item["request_id"]) != 64
            or not isinstance(item.get("context"), str)
            or not isinstance(item.get("continuation"), str)
            or not item["continuation"]
            for item in requests
        ):
            raise ValueError("native scorer only accepts frozen loglikelihood requests")
        request_ids = [item["request_id"] for item in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("frozen request manifest contains duplicate request IDs")
        task_names = {item.get("task") for item in tasks}
        if any(item.get("task") not in task_names for item in requests):
            raise ValueError(
                "frozen requests reference task names absent from task metadata"
            )
        for item in requests:
            if (item["task"] in window_tasks) != (item["request_type"] == "token_window_loglikelihood"):
                raise ValueError("document-window request/task mismatch")
            if item["task"] in window_tasks:
                window = item.get("token_window")
                if (not isinstance(window, dict) or any(not isinstance(window.get(k), int) or isinstance(window.get(k), bool)
                        for k in ("context_start", "target_start", "target_end", "document_tokens"))):
                    raise ValueError("invalid document-window metadata")
                if not 0 <= window["context_start"] <= window["target_start"] < window["target_end"] <= window["document_tokens"]:
                    raise ValueError("invalid document-window interval")
                _require_sha256(window.get("window_token_ids_sha256"), field="window_token_ids_sha256")
                _require_sha256(window.get("document_token_ids_sha256"), field="document_token_ids_sha256")
            if item["task"] in reference_tasks and item.get("target_utf8_bytes") != len(item["continuation"].encode("utf-8")):
                raise ValueError("reference target byte count does not match scored text")
        if window_tasks:
            from collections import defaultdict
            documents = defaultdict(list)
            for item in requests:
                if item["task"] in window_tasks:
                    documents[item["task"], item["doc_id"]].append(item)
            for windows in documents.values():
                windows.sort(key=lambda item: item["token_window"]["target_start"])
                cursor = 0
                identity = windows[0]["token_window"]["document_token_ids_sha256"]
                text = windows[0]["continuation"]
                count = windows[0]["token_window"]["document_tokens"]
                for item in windows:
                    window = item["token_window"]
                    if (window["target_start"] != cursor or window["document_tokens"] != count
                            or window["document_token_ids_sha256"] != identity or item["continuation"] != text):
                        raise ValueError("document windows have a gap, overlap or source mismatch")
                    cursor = window["target_end"]
                if cursor != count:
                    raise ValueError("document windows do not cover all targets")
        if request_manifest_cache is not None:
            request_manifest_cache[request_path] = (
                request_sha256,
                request_payload,
            )

    recipe = payload.get("model_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("model_recipe must be an object")
    recipe_name = recipe.get("name")
    overrides = recipe.get("overrides")
    expected_model = recipe.get("expected_model")
    if not isinstance(recipe_name, str) or not recipe_name:
        raise ValueError("model_recipe.name must be a non-empty string")
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("model_recipe.overrides must be a list of strings")
    if not isinstance(expected_model, dict) or not expected_model:
        raise ValueError("model_recipe.expected_model must be a non-empty object")

    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be an object")
    checkpoint_path = _resolve_path(base, checkpoint.get("path"), field="checkpoint.path")
    checkpoint_step = checkpoint.get("step")
    if not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool) or checkpoint_step <= 0:
        raise ValueError("checkpoint.step must be a positive integer")
    checkpoint_iteration_path = _resolve_path(
        base, checkpoint.get("iteration_path"), field="checkpoint.iteration_path"
    )
    expected_iteration = checkpoint_path / f"iter_{checkpoint_step:07d}"
    if checkpoint_iteration_path != expected_iteration.resolve():
        raise ValueError(
            "checkpoint.iteration_path does not match checkpoint.path/checkpoint.step: "
            f"{checkpoint_iteration_path} != {expected_iteration.resolve()}"
        )
    if not checkpoint_iteration_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint iteration: {checkpoint_iteration_path}")
    training_run_config = checkpoint.get("training_run_config")
    if not isinstance(training_run_config, dict):
        raise ValueError("checkpoint.training_run_config must be an object")
    training_run_config_path = _resolve_path(
        base,
        training_run_config.get("path"),
        field="checkpoint.training_run_config.path",
    )
    expected_training_run_config_path = checkpoint_iteration_path / "run_config.yaml"
    if training_run_config_path != expected_training_run_config_path.resolve():
        raise ValueError(
            "checkpoint training run config path does not belong to the frozen iteration"
        )
    training_run_config_sha256 = _require_sha256(
        training_run_config.get("sha256"),
        field="checkpoint.training_run_config.sha256",
    )
    if _sha256_file(training_run_config_path) != training_run_config_sha256:
        raise ValueError(
            f"checkpoint training run config SHA-256 mismatch: {training_run_config_path}"
        )

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer must be an object")
    tokenizer_type = tokenizer.get("type")
    if tokenizer_type not in {"SentencePieceTokenizer", "HuggingFaceTokenizer"}:
        raise ValueError(
            "native scorer supports SentencePieceTokenizer and HuggingFaceTokenizer"
        )
    tokenizer_path = _resolve_path(base, tokenizer.get("model"), field="tokenizer.model")
    tokenizer_sha256: str | None = None
    tokenizer_files_sha256: dict[str, str] = {}
    if tokenizer_path.is_file():
        tokenizer_sha256 = _require_sha256(
            tokenizer.get("sha256"), field="tokenizer.sha256"
        )
        if _sha256_file(tokenizer_path) != tokenizer_sha256:
            raise ValueError(f"tokenizer SHA-256 mismatch: {tokenizer_path}")
    elif tokenizer_path.is_dir():
        files = tokenizer.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("directory tokenizers require a non-empty tokenizer.files map")
        for relative, expected in sorted(files.items()):
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError("tokenizer.files keys must be non-empty relative paths")
            artifact = (tokenizer_path / relative).resolve()
            try:
                artifact.relative_to(tokenizer_path)
            except ValueError as exc:
                raise ValueError(f"tokenizer artifact escapes tokenizer directory: {relative}") from exc
            if not artifact.is_file():
                raise FileNotFoundError(f"missing tokenizer artifact: {artifact}")
            digest = _require_sha256(expected, field=f"tokenizer.files[{relative!r}]")
            if _sha256_file(artifact) != digest:
                raise ValueError(f"tokenizer artifact SHA-256 mismatch: {artifact}")
            tokenizer_files_sha256[relative] = digest
    else:
        raise FileNotFoundError(f"missing tokenizer model path: {tokenizer_path}")
    vocab_size = tokenizer.get("vocab_size")
    context_token_id = tokenizer.get("context_token_id")
    pad_token_id = tokenizer.get("pad_token_id")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("tokenizer.vocab_size must be positive")
    for field, value in (
        ("tokenizer.context_token_id", context_token_id),
        ("tokenizer.pad_token_id", pad_token_id),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < vocab_size:
            raise ValueError(f"{field} must be a valid token ID")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    sequence_length = evaluation.get("sequence_length")
    minimum_sequence_width = evaluation.get("minimum_sequence_width", 1)
    sequence_width_multiple = evaluation.get("sequence_width_multiple", 1)
    micro_batch_size = evaluation.get("micro_batch_size")
    expected_world_size = evaluation.get("expected_world_size")
    parity_requests = evaluation.get("parity_requests")
    max_gpu_hours = evaluation.get("max_gpu_hours")
    required_gpu_name_substring = evaluation.get("required_gpu_name_substring")
    expected_moe_layers = evaluation.get("expected_moe_layers")
    batching = AutoBatchPolicy.from_mapping(evaluation.get("batching"))
    if batching is not None and expected_model.get("moe_expert_capacity_factor") is not None:
        raise ValueError("automatic batching requires dropless MoE routing")
    moe_reduction_dtype = evaluation.get("moe_reduction_dtype", "model")
    if moe_reduction_dtype not in ("model", "float64"):
        raise ValueError("evaluation.moe_reduction_dtype must be model or float64")
    moe_compute_dtype = evaluation.get("moe_compute_dtype", "model")
    if moe_compute_dtype not in ("model", "float64"):
        raise ValueError("evaluation.moe_compute_dtype must be model or float64")
    if moe_compute_dtype == "float64" and expected_world_size != 1:
        raise ValueError("FP64 expert evaluation requires single-GPU scoring")
    for field, value in (
        ("evaluation.sequence_length", sequence_length),
        ("evaluation.minimum_sequence_width", minimum_sequence_width),
        ("evaluation.sequence_width_multiple", sequence_width_multiple),
        ("evaluation.micro_batch_size", micro_batch_size),
        ("evaluation.expected_world_size", expected_world_size),
        ("evaluation.parity_requests", parity_requests),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(max_gpu_hours, (int, float)) or isinstance(max_gpu_hours, bool) or max_gpu_hours <= 0:
        raise ValueError("evaluation.max_gpu_hours must be positive")
    if (
        not isinstance(required_gpu_name_substring, str)
        or not required_gpu_name_substring.strip()
    ):
        raise ValueError("evaluation.required_gpu_name_substring must be non-empty")
    if expected_moe_layers is not None and (
        not isinstance(expected_moe_layers, int)
        or isinstance(expected_moe_layers, bool)
        or expected_moe_layers <= 0
    ):
        raise ValueError("evaluation.expected_moe_layers must be a positive integer or null")
    if expected_model.get("seq_length") != sequence_length:
        raise ValueError("evaluation sequence length does not match expected model contract")
    if sequence_length % sequence_width_multiple:
        raise ValueError("evaluation.sequence_width_multiple must divide sequence_length")
    if minimum_sequence_width > sequence_length:
        raise ValueError("minimum scoring width exceeds the model sequence length")
    # Native tokenizers can have fewer IDs than the padded embedding/output
    # vocabulary. Keep both identities in the spec: model
    # construction checks the embedding size, tokenization checks the ID range.
    model_vocab_size = expected_model.get("vocab_size")
    if (
        not isinstance(model_vocab_size, int)
        or isinstance(model_vocab_size, bool)
        or model_vocab_size < vocab_size
    ):
        raise ValueError("model vocabulary cannot represent every tokenizer ID")

    return NativeBenchmarkScoreSpec(
        source_path=source_path,
        source_sha256=source_sha256,
        run_id=run_id,
        output_dir=output_dir,
        request_manifest_path=request_path,
        request_manifest_sha256=request_sha256,
        recipe_name=recipe_name,
        recipe_overrides=tuple(overrides),
        model_contract=dict(expected_model),
        checkpoint_path=checkpoint_path,
        checkpoint_iteration_path=checkpoint_iteration_path,
        checkpoint_step=checkpoint_step,
        training_run_config_path=training_run_config_path,
        training_run_config_sha256=training_run_config_sha256,
        tokenizer_type=tokenizer_type,
        tokenizer_model_path=tokenizer_path,
        tokenizer_model_sha256=tokenizer_sha256,
        tokenizer_files_sha256=tokenizer_files_sha256,
        tokenizer_vocab_size=vocab_size,
        context_token_id=context_token_id,
        pad_token_id=pad_token_id,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        expected_world_size=expected_world_size,
        max_gpu_hours=float(max_gpu_hours),
        parity_requests=parity_requests,
        require_moe_routing_diagnostics=bool(
            evaluation.get("require_moe_routing_diagnostics", False)
        ),
        expected_moe_layers=expected_moe_layers,
        required_gpu_name_substring=required_gpu_name_substring.strip(),
        moe_reduction_dtype=moe_reduction_dtype,
        batching=batching,
        minimum_sequence_width=minimum_sequence_width,
        sequence_width_multiple=sequence_width_multiple,
        moe_compute_dtype=moe_compute_dtype,
    )



def run_native_benchmark_scoring(config: Any, spec_path: Path) -> None:
    """Load the distributed runner lazily to keep contracts importable on CPU."""

    from .native_megatron_runner import run_native_benchmark_scoring as run

    run(config, spec_path)
