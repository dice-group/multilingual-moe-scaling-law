"""Tensor scoring, parity, replica, and inference setup primitives."""

import json
import os
from copy import copy
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence
from types import MethodType

import torch

from multilingual_megatron.pipeline.backends.megatron_bridge.checkpoint_inference import (
    setup_bridge_inference_only as _setup_bridge_inference_only,
)

from .native_megatron_lm_eval import (
    EARLY_REPLICA_REQUESTS, EARLY_REPLICA_SCORE_ABS_TOLERANCE,
    NLL_SUM_RELATIVE_TOLERANCE, RAW_NLL_ABS_TOLERANCE,
    REPLICA_SCORE_SUM_RELATIVE_TOLERANCE, NativeBenchmarkScoreSpec,
    _payload_sha256,
)


def _fp64_expert_forward(experts, hidden, counts, probabilities):
    """Evaluate unchanged SwiGLU weights in FP64; retain the model output dtype."""
    sizes = counts.tolist()
    if (len(sizes) != experts.num_local_experts or any(n < 0 for n in sizes)
            or sum(sizes) != hidden.shape[0] or probabilities.numel() != hidden.shape[0]):
        raise ValueError("expert token counts or probabilities do not match dispatched inputs")
    outputs, offset = [], 0
    for index, count in enumerate(sizes):
        count = int(count)
        if count:
            first = getattr(experts.linear_fc1, f"weight{index}")
            second = getattr(experts.linear_fc2, f"weight{index}")
            gate, value = torch.nn.functional.linear(
                hidden[offset:offset + count].double(), first.double()
            ).chunk(2, dim=-1)
            activation = torch.nn.functional.silu(gate) * value
            activation *= probabilities[offset:offset + count].double().reshape(count, 1)
            outputs.append(torch.nn.functional.linear(activation, second.double()).to(hidden.dtype))
        offset += count
    return (torch.cat(outputs, dim=0) if outputs else hidden.clone()), None


@contextmanager
def expert_compute_precision(model, dtype: str):
    """Opt-in FP64 router/expert arithmetic for local, bias-free Top-1 SwiGLU.

    Stored parameters, expert assignments and probabilities retain the model's
    definitions. Only evaluation arithmetic changes. FP64 temporaries are
    allocated per expert, and every instance override is restored on exit.
    """
    if dtype == "model":
        yield
        return
    if dtype != "float64":
        raise ValueError("unsupported expert compute dtype")
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    layers = [m for m in model.modules() if isinstance(m, MoELayer)]
    if not layers:
        raise ValueError("FP64 expert evaluation requires MoE layers")
    for layer in layers:
        cfg, experts = layer.config, layer.experts
        if (layer.training or not isinstance(experts, TEGroupedMLP)
                or any(getattr(cfg, key) != 1 for key in (
                    "tensor_model_parallel_size", "expert_tensor_parallel_size",
                    "expert_model_parallel_size", "pipeline_model_parallel_size",
                    "context_parallel_size", "moe_router_topk"))
                or cfg.moe_expert_capacity_factor is not None or cfg.add_bias_linear
                or cfg.fp8 or cfg.fp4 or not cfg.gated_linear_unit
                or cfg.moe_apply_probs_on_input or cfg.moe_single_grouped_weight
                or cfg.moe_mlp_glu_interleave_size is not None
                or cfg.activation_func_clamp_value is not None or cfg.glu_linear_offset != 0
                or experts.activation_func is not torch.nn.functional.silu
                or layer.router.bias is not None):
            raise ValueError("FP64 expert evaluation supports only local, dropless, bias-free Top-1 SwiGLU")
    restored = []
    try:
        for layer in layers:
            experts, router = layer.experts, layer.router
            restored.append((experts, experts.__dict__.get("forward"), router, router.config))
            experts.forward = MethodType(_fp64_expert_forward, experts)
            router.config = copy(router.config)
            router.config.moe_router_dtype = "fp64"
        yield
    finally:
        for experts, forward, router, config in reversed(restored):
            if forward is None:
                del experts.forward
            else:
                experts.forward = forward
            router.config = config


@contextmanager
def expert_reduction_precision(dtype: str):
    """Optionally reduce all-gather expert outputs in FP64 during scoring."""
    if dtype == "model":
        yield
        return
    if dtype != "float64":
        raise ValueError(f"unsupported expert reduction dtype: {dtype}")
    dispatcher = import_module("megatron.core.transformer.moe.token_dispatcher")
    original = dispatcher.MoEAllGatherTokenDispatcher.token_combine

    def token_combine(self, hidden_states):
        if self.tp_size > 1 or self.ep_size > 1:
            # Reduce-scatter can sum identical contributions in different orders
            # for different destinations. BF16 rounding can amplify the resulting
            # FP32 differences downstream. Preserve the incoming activation dtype.
            hidden_states = dispatcher.reduce_scatter_to_sequence_parallel_region(
                hidden_states.double(), group=self.tp_ep_group
            ).to(hidden_states.dtype)
        return hidden_states

    dispatcher.MoEAllGatherTokenDispatcher.token_combine = token_combine
    try:
        yield
    finally:
        dispatcher.MoEAllGatherTokenDispatcher.token_combine = original


def verify_native_benchmark_model_contract(config: Any, spec: NativeBenchmarkScoreSpec) -> None:
    mismatches: list[str] = []
    for field, expected in spec.model_contract.items():
        actual = getattr(config.model, field, None)
        if actual != expected:
            mismatches.append(f"{field}:{actual!r}!={expected!r}")
    if mismatches:
        raise ValueError("native benchmark model contract mismatch: " + "; ".join(mismatches))


def configure_native_benchmark_scoring(config: Any, spec: NativeBenchmarkScoreSpec) -> Any:
    """Apply the frozen checkpoint and evaluation-only runtime settings."""

    verify_native_benchmark_model_contract(config, spec)
    # JSON specifications name the backend; MCore requires its enum to check
    # that the NVTE environment agrees with that choice.
    if isinstance(getattr(config.model, "attention_backend", None), str):
        from megatron.core.transformer.enums import AttnBackend

        config.model.attention_backend = AttnBackend[config.model.attention_backend]
    if spec.moe_reduction_dtype == "float64" and getattr(config.model, "moe_token_dispatcher_type", None) != "allgather":
        raise ValueError("FP64 expert reduction requires all-gather token dispatch")
    model_parallel_size = (
        int(getattr(config.model, "tensor_model_parallel_size", 1) or 1)
        * int(getattr(config.model, "pipeline_model_parallel_size", 1) or 1)
        * int(getattr(config.model, "context_parallel_size", 1) or 1)
    )
    if spec.expected_world_size % model_parallel_size:
        raise ValueError(
            "native benchmark world size is not divisible by tensor/pipeline/context "
            f"model parallelism: {spec.expected_world_size} % {model_parallel_size}"
        )
    data_parallel_size = spec.expected_world_size // model_parallel_size
    # The inference-only model has no main-grad buffers required by the op-fuser.
    config.model.gradient_accumulation_fusion = False
    # Training graph scopes do not describe this variable-length inference loop.
    config.model.cuda_graph_impl = "none"
    config.model.cuda_graph_scope = []
    config.checkpoint.load = str(spec.checkpoint_path)
    config.checkpoint.ckpt_step = spec.checkpoint_step
    config.checkpoint.load_optim = False
    config.checkpoint.load_rng = False
    config.checkpoint.async_save = False
    config.checkpoint.save = None
    config.validation.skip_train = True
    config.validation.eval_interval = 0
    config.validation.eval_iters = 0
    config.validation.eval_micro_batch_size = spec.micro_batch_size
    # Bridge otherwise inherits the training evaluation global batch. That value
    # can be smaller than an early readout's sealed micro batch and is irrelevant
    # to the scorer's direct forward loop. Freeze the smallest valid evaluation
    # global batch for the launch topology before runtime_config_update validates it.
    scoring_global_batch_size = spec.micro_batch_size * data_parallel_size
    config.validation.eval_global_batch_size = scoring_global_batch_size
    config.train.micro_batch_size = spec.micro_batch_size
    # Megatron initializes its training microbatch calculator even when Bridge
    # skips training. Keep that unused calculator internally consistent too.
    config.train.global_batch_size = scoring_global_batch_size
    config.tokenizer.tokenizer_type = spec.tokenizer_type
    config.tokenizer.tokenizer_model = str(spec.tokenizer_model_path)
    config.tokenizer.vocab_size = spec.tokenizer_vocab_size
    return config


def encode_lm_eval_pair(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    context_token_id: int,
) -> tuple[list[int], list[int]]:
    """Match :class:`lm_eval.api.model.TemplateLM` causal pair encoding."""

    if not isinstance(context, str) or not isinstance(continuation, str):
        raise TypeError("context and continuation must be strings")
    if context == "":
        continuation_ids = list(tokenizer.tokenize(continuation))
        if not continuation_ids:
            raise ValueError("continuation tokenized to an empty sequence")
        if continuation_ids[0] == context_token_id:
            return continuation_ids[:1], continuation_ids[1:]
        return [context_token_id], continuation_ids

    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    whole_ids = list(tokenizer.tokenize(context + continuation))
    context_ids = list(tokenizer.tokenize(context))
    continuation_ids = whole_ids[len(context_ids) :]
    if not context_ids:
        context_ids = [context_token_id]
    if not continuation_ids:
        raise ValueError("continuation tokenized to an empty sequence")
    return context_ids, continuation_ids


@dataclass(frozen=True, slots=True)
class ScoredSequence:
    request_index: int
    request_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    continuation_start: int
    continuation_tokens: int
    truncated_tokens: int


def prepare_scored_sequence(
    *,
    request_index: int,
    request_id: str,
    context_ids: Sequence[int],
    continuation_ids: Sequence[int],
    max_length: int,
) -> ScoredSequence:
    """Build the same left-truncated causal window used by lm-eval HFLM."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not context_ids or not continuation_ids:
        raise ValueError("context and continuation token sequences must be non-empty")
    if len(continuation_ids) > max_length:
        raise ValueError(
            f"continuation contains {len(continuation_ids)} tokens, exceeding max_length={max_length}"
        )
    combined = list(context_ids) + list(continuation_ids)
    truncated_tokens = max(0, len(combined) - (max_length + 1))
    window = combined[-(max_length + 1) :]
    input_ids = window[:-1]
    labels = window[1:]
    continuation_start = len(labels) - len(continuation_ids)
    if continuation_start < 0 or not input_ids:
        raise ValueError("invalid causal scoring window")
    return ScoredSequence(
        request_index=request_index,
        request_id=request_id,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        continuation_start=continuation_start,
        continuation_tokens=len(continuation_ids),
        truncated_tokens=truncated_tokens,
    )


def build_score_batch(
    sequences: Sequence[ScoredSequence],
    *,
    pad_token_id: int,
    minimum_sequence_width: int = 1,
    sequence_width_multiple: int = 1,
) -> dict[str, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot build an empty score batch")
    batch_size = len(sequences)
    if not isinstance(minimum_sequence_width, int) or isinstance(minimum_sequence_width, bool) or minimum_sequence_width < 1:
        raise ValueError("minimum scoring width must be a positive integer")
    if not isinstance(sequence_width_multiple, int) or isinstance(sequence_width_multiple, bool) or sequence_width_multiple < 1:
        raise ValueError("scoring width multiple must be a positive integer")
    # Keep short inputs on the validated kernel path without adding targets.
    width = max(minimum_sequence_width, max(len(item.input_ids) for item in sequences))
    width = ((width + sequence_width_multiple - 1) // sequence_width_multiple) * sequence_width_multiple
    tokens = torch.full((batch_size, width), int(pad_token_id), dtype=torch.int64)
    labels = torch.zeros((batch_size, width), dtype=torch.int64)
    loss_mask = torch.zeros((batch_size, width), dtype=torch.float32)
    input_valid_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    position_ids = torch.arange(width, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    for row, item in enumerate(sequences):
        length = len(item.input_ids)
        input_valid_mask[row, :length] = True
        tokens[row, :length] = torch.tensor(item.input_ids, dtype=torch.int64)
        labels[row, :length] = torch.tensor(item.labels, dtype=torch.int64)
        start = item.continuation_start
        loss_mask[row, start:length] = 1.0
        if int(loss_mask[row].sum().item()) != item.continuation_tokens:
            raise RuntimeError(f"continuation mask mismatch for {item.request_id}")
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "input_valid_mask": input_valid_mask,
        "sample_id": torch.tensor([item.request_index for item in sequences], dtype=torch.int64),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Megatron model returned unsupported output type: {type(output)}")
    return output


def _score_cuda_batch(model: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = batch["tokens"].to(device=device, non_blocking=True)
    labels = batch["labels"].to(device=device, non_blocking=True)
    position_ids = batch["position_ids"].to(device=device, non_blocking=True)
    with torch.no_grad():
        output = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
        )
    return _tensor_output(output)


def _parity_gate(
    *,
    state: Any,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    direct_losses: torch.Tensor,
) -> dict[str, Any]:
    """Compare native extraction with the production Bridge GPT forward step."""

    from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step

    with torch.no_grad():
        reference_output, _loss_function = gpt_forward_step(
            state, iter([dict(batch)]), model
        )
    reference_losses = _tensor_output(reference_output)
    device = direct_losses.device
    mask = batch["loss_mask"].to(device=device, dtype=torch.float64)
    direct_sum = float((direct_losses.to(torch.float64) * mask).sum().item())
    reference_sum = float((reference_losses.to(torch.float64) * mask).sum().item())
    token_count = int(mask.sum().item())
    reference_token_count = int(batch["loss_mask"].sum().item())
    direct_raw_nll = direct_sum / token_count
    reference_raw_nll = reference_sum / reference_token_count
    raw_abs_difference = abs(direct_raw_nll - reference_raw_nll)
    denominator = max(abs(reference_sum), torch.finfo(torch.float64).tiny)
    sum_relative_difference = abs(direct_sum - reference_sum) / denominator
    passed = (
        token_count == reference_token_count
        and raw_abs_difference <= RAW_NLL_ABS_TOLERANCE
        and sum_relative_difference <= NLL_SUM_RELATIVE_TOLERANCE
    )
    result = {
        "passed": passed,
        "token_count": token_count,
        "reference_token_count": reference_token_count,
        "native_nll_sum": direct_sum,
        "reference_nll_sum": reference_sum,
        "native_raw_nll": direct_raw_nll,
        "reference_raw_nll": reference_raw_nll,
        "raw_nll_abs_difference": raw_abs_difference,
        "raw_nll_abs_tolerance": RAW_NLL_ABS_TOLERANCE,
        "nll_sum_relative_difference": sum_relative_difference,
        "nll_sum_relative_tolerance": NLL_SUM_RELATIVE_TOLERANCE,
        "reference_path": "megatron.bridge.training.gpt_step.forward_step",
    }
    if not passed:
        raise RuntimeError(f"native benchmark parity gate failed: {result}")
    return result
def _choice_prediction_vector(
    score_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, int]]:
    """Return raw and two length-normalized choices in stable document order."""

    if len(score_rows) != len(request_rows):
        raise RuntimeError("replica gate score/request count mismatch")
    score_by_id = {str(item["request_id"]): item for item in score_rows}
    if len(score_by_id) != len(score_rows):
        raise RuntimeError("replica gate received duplicate score request IDs")
    request_ids = {str(item["request_id"]) for item in request_rows}
    if set(score_by_id) != request_ids:
        raise RuntimeError("replica gate score coverage does not match frozen requests")

    grouped: dict[tuple[str, int], list[tuple[int, float, int, int]]] = {}
    for request in request_rows:
        request_id = str(request["request_id"])
        continuation = str(request["continuation"])
        continuation_length = len(continuation)
        stripped_length = len(continuation.lstrip())
        if continuation_length < 1 or stripped_length < 1:
            raise RuntimeError(f"empty continuation in replica gate: {request_id}")
        key = (str(request["task"]), int(request["doc_id"]))
        grouped.setdefault(key, []).append(
            (
                int(request["choice_index"]),
                float(score_by_id[request_id]["loglikelihood"]),
                continuation_length,
                stripped_length,
            )
        )

    predictions: list[tuple[int, int, int]] = []
    for key in sorted(grouped):
        choices = sorted(grouped[key], key=lambda item: item[0])
        choice_indices = [item[0] for item in choices]
        if len(choice_indices) != len(set(choice_indices)):
            raise RuntimeError(f"duplicate choice indices in replica gate document {key}")
        raw_choice = max(choices, key=lambda item: item[1])[0]
        normalized_choice = max(choices, key=lambda item: item[1] / item[2])[0]
        stripped_normalized_choice = max(choices, key=lambda item: item[1] / item[3])[0]
        predictions.append((raw_choice, normalized_choice, stripped_normalized_choice))
    return predictions


def _write_replica_diagnostic(
    *,
    output_dir: Path,
    rank: int,
    request_manifest_sha256: str,
    score_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist rank-local scores before replica gates."""

    payload: dict[str, Any] = {
        "schema": "stage1_native_megatron_replica_diagnostic/v1",
        "status": "diagnostic_only",
        "rank": rank,
        "request_manifest_sha256": request_manifest_sha256,
        "scores": [
            {
                "request_id": str(item["request_id"]),
                "loglikelihood": float(item["loglikelihood"]),
                "scored_tokens": int(item["scored_tokens"]),
            }
            for item in score_rows
        ],
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    path = output_dir / "replica_diagnostics" / f"rank_{rank}.json"
    _atomic_json(path, payload)
    return path


def _distributed_request_score_spread(score_values: Sequence[float]) -> dict[str, float]:
    """Measure rank-wise spread for aligned request scores."""

    import torch.distributed as dist

    if not score_values:
        raise RuntimeError("cannot measure replica spread for an empty score batch")
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return {"max": 0.0, "mean": 0.0}
    device = torch.device("cuda", torch.cuda.current_device())
    minima = torch.tensor(score_values, dtype=torch.float64, device=device)
    maxima = minima.clone()
    dist.all_reduce(minima, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    spreads = maxima - minima
    return {
        "max": float(spreads.max().item()),
        "mean": float(spreads.mean().item()),
    }


def _distributed_replica_gate(
    score_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    *, reference_tasks: set[str] | None = None,
) -> dict[str, Any]:
    import torch.distributed as dist

    score_values = [float(item["loglikelihood"]) for item in score_rows]
    score_sum = sum(score_values)
    request_count = len(score_values)
    reference_tasks = reference_tasks or set()
    score_by_id = {row["request_id"]: row for row in score_rows}
    if (len(score_by_id) != len(score_rows) or len(score_rows) != len(request_rows)
            or set(score_by_id) != {row["request_id"] for row in request_rows}):
        raise RuntimeError("replica gate score coverage does not match requests")
    choice_requests = [r for r in request_rows if r["task"] not in reference_tasks]
    predictions = _choice_prediction_vector([score_by_id[r["request_id"]] for r in choice_requests], choice_requests)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return {
            "passed": True,
            "world_size": 1,
            "score_sum_min": score_sum,
            "score_sum_max": score_sum,
            "score_sum_abs_spread": 0.0,
            "score_sum_relative_spread": 0.0,
            "score_sum_relative_tolerance": REPLICA_SCORE_SUM_RELATIVE_TOLERANCE,
            "request_score_abs_spread_max": 0.0,
            "request_score_abs_spread_mean": 0.0,
            "raw_choice_prediction_mismatches": 0,
            "continuation_length_normalized_prediction_mismatches": 0,
            "stripped_continuation_length_normalized_prediction_mismatches": 0,
            "documents": len(predictions),
            "request_count_min": request_count,
            "request_count_max": request_count,
        }
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.tensor([score_sum, float(request_count)], dtype=torch.float64, device=device)
    minima = values.clone()
    maxima = values.clone()
    dist.all_reduce(minima, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)

    score_minima = torch.tensor(score_values, dtype=torch.float64, device=device)
    score_maxima = score_minima.clone()
    dist.all_reduce(score_minima, op=dist.ReduceOp.MIN)
    dist.all_reduce(score_maxima, op=dist.ReduceOp.MAX)
    request_spreads = score_maxima - score_minima

    prediction_minima = torch.tensor(predictions, dtype=torch.int64, device=device).reshape(-1, 3)
    prediction_maxima = prediction_minima.clone()
    if predictions:
        dist.all_reduce(prediction_minima, op=dist.ReduceOp.MIN)
        dist.all_reduce(prediction_maxima, op=dist.ReduceOp.MAX)
    prediction_mismatches = prediction_minima != prediction_maxima

    score_sum_min = float(minima[0].item())
    score_sum_max = float(maxima[0].item())
    score_sum_abs_spread = abs(score_sum_max - score_sum_min)
    denominator = max(
        abs(score_sum_min), abs(score_sum_max), torch.finfo(torch.float64).tiny
    )
    score_sum_relative_spread = score_sum_abs_spread / denominator
    score_sum_tolerance = denominator * REPLICA_SCORE_SUM_RELATIVE_TOLERANCE
    request_score_abs_spread_max = float(request_spreads.max().item())
    request_score_abs_spread_mean = float(request_spreads.mean().item())
    raw_prediction_mismatches = int(prediction_mismatches[:, 0].sum().item())
    normalized_prediction_mismatches = int(prediction_mismatches[:, 1].sum().item())
    stripped_normalized_prediction_mismatches = int(
        prediction_mismatches[:, 2].sum().item()
    )
    request_count_min = int(round(float(minima[1].item())))
    request_count_max = int(round(float(maxima[1].item())))
    passed = (
        score_sum_relative_spread <= REPLICA_SCORE_SUM_RELATIVE_TOLERANCE
        and raw_prediction_mismatches == 0
        and normalized_prediction_mismatches == 0
        and stripped_normalized_prediction_mismatches == 0
        and request_count_min == request_count_max
        and (not reference_tasks or request_score_abs_spread_max <= EARLY_REPLICA_SCORE_ABS_TOLERANCE)
    )
    result = {
        "passed": passed,
        "world_size": dist.get_world_size(),
        "score_sum_min": score_sum_min,
        "score_sum_max": score_sum_max,
        "score_sum_abs_spread": score_sum_abs_spread,
        "score_sum_relative_spread": score_sum_relative_spread,
        "score_sum_tolerance": score_sum_tolerance,
        "score_sum_relative_tolerance": REPLICA_SCORE_SUM_RELATIVE_TOLERANCE,
        "request_score_abs_spread_max": request_score_abs_spread_max,
        "request_score_abs_spread_mean": request_score_abs_spread_mean,
        "request_score_abs_spread_gt_1e-3": int((request_spreads > 1.0e-3).sum().item()),
        "request_score_abs_spread_gt_1e-2": int((request_spreads > 1.0e-2).sum().item()),
        "raw_choice_prediction_mismatches": raw_prediction_mismatches,
        "continuation_length_normalized_prediction_mismatches": normalized_prediction_mismatches,
        "stripped_continuation_length_normalized_prediction_mismatches": (
            stripped_normalized_prediction_mismatches
        ),
        "documents": len(predictions),
        "request_count_min": request_count_min,
        "request_count_max": request_count_max,
    }
    if not passed:
        raise RuntimeError(f"expert-parallel score replica gate failed: {result}")
    return result
