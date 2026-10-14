"""Distributed native Megatron benchmark scoring runner."""

import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import torch

from ..final_loss.main_head import main_prediction_only
from .native_megatron_batching import calibrate_native_batches, score_batch_ranges

from .native_megatron_lm_eval import (
    EARLY_REPLICA_REQUESTS,
    EARLY_REPLICA_SCORE_ABS_TOLERANCE,
    SCORE_RESULT_SCHEMA,
    _load_json,
    _payload_sha256,
    load_native_benchmark_score_spec,
)
from .native_megatron_scoring import (
    ScoredSequence,
    _atomic_json,
    _distributed_replica_gate,
    _distributed_request_score_spread,
    _parity_gate,
    _score_cuda_batch,
    _setup_bridge_inference_only,
    _write_replica_diagnostic,
    build_score_batch,
    configure_native_benchmark_scoring,
    encode_lm_eval_pair,
    expert_reduction_precision,
    expert_compute_precision,
    prepare_scored_sequence,
)


def run_native_benchmark_scoring(config: Any, spec_path: Path) -> None:
    """Load one checkpoint once and score every frozen lm-eval request."""

    import torch.distributed as dist
    from megatron.bridge.training.config import runtime_config_update
    from megatron.bridge.training.pretrain import _maybe_destroy_process_group
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training.train import _finish_train

    spec = load_native_benchmark_score_spec(spec_path)
    configure_native_benchmark_scoring(config, spec)
    runtime_config_update(config)
    total_started = time.perf_counter()

    state = GlobalState()
    state.cfg = config

    def _empty_dataset_provider(_sample_counts: Sequence[int], _dataset_config: Any):
        return None, None, None

    should_destroy_process_group = not dist.is_initialized()
    setup_started = time.perf_counter()
    setup_output = _setup_bridge_inference_only(state, _empty_dataset_provider)
    state = setup_output.state
    models = setup_output.model
    checkpoint_manager = setup_output.checkpoint_manager
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    evaluation_scope = ExitStack()
    try:
        if world_size != spec.expected_world_size:
            raise RuntimeError(
                f"world size {world_size} does not match frozen expected_world_size "
                f"{spec.expected_world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("native benchmark scoring requires CUDA")
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        if spec.required_gpu_name_substring.lower() not in gpu_name.lower():
            raise RuntimeError(
                f"GPU {gpu_name!r} does not match required hardware "
                f"{spec.required_gpu_name_substring!r}"
            )
        if not isinstance(models, list) or len(models) != 1:
            raise RuntimeError("native benchmark requires exactly one PP/VP model chunk")
        model = models[0]
        model.eval()
        evaluation_scope.enter_context(main_prediction_only(models))
        evaluation_scope.enter_context(expert_reduction_precision(
            spec.moe_reduction_dtype
        ))
        evaluation_scope.enter_context(expert_compute_precision(model, spec.moe_compute_dtype))

        if int(state.train_state.step) != spec.checkpoint_step:
            raise RuntimeError(
                f"loaded checkpoint step {state.train_state.step}, expected {spec.checkpoint_step}"
            )
        # Return unused checkpoint-loading buffers to CUDA before libraries
        # allocate their inference workspaces. Live model tensors are preserved.
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"[native-benchmark] rank={rank}; checkpoint loaded; "
            f"allocated_bytes={torch.cuda.memory_allocated()}; "
            f"reserved_bytes={torch.cuda.memory_reserved()}; "
            f"free_bytes={free_bytes}; total_bytes={total_bytes}",
            flush=True,
        )
        setup_seconds = time.perf_counter() - setup_started
        request_manifest = _load_json(
            spec.request_manifest_path, label="frozen request manifest"
        )
        request_rows = request_manifest["requests"]
        reference_tasks = {item["task"] for item in request_manifest["tasks"]
                           if item["output_type"] in {"loglikelihood", "frozen_document_loglikelihood"}}
        tokenizer = state.tokenizer
        actual_vocab = int(getattr(tokenizer, "vocab_size", 0))
        if actual_vocab != spec.tokenizer_vocab_size:
            raise RuntimeError(
                f"runtime tokenizer vocab {actual_vocab} != frozen vocab {spec.tokenizer_vocab_size}"
            )
        actual_pad = int(getattr(tokenizer, "pad_id"))
        if actual_pad != spec.pad_token_id:
            raise RuntimeError(
                f"runtime tokenizer pad ID {actual_pad} != frozen pad ID {spec.pad_token_id}"
            )
        actual_eod = int(getattr(tokenizer, "eod"))
        if actual_eod != spec.context_token_id:
            raise RuntimeError(
                "runtime tokenizer EOD ID "
                f"{actual_eod} != frozen empty-context ID {spec.context_token_id}"
            )

        sequences: list[ScoredSequence] = []
        cache_reference = _load_json(spec.source_path, label="score spec").get("token_cache")
        if cache_reference is not None:
            from .native_token_cache import load_cached_sequences
            sequences = load_cached_sequences(cache_reference, spec, request_rows)
        for index, request in enumerate(request_rows if cache_reference is None else []):
            context_ids, continuation_ids = encode_lm_eval_pair(
                tokenizer,
                request["context"],
                request["continuation"],
                context_token_id=spec.context_token_id,
            )
            sequences.append(
                prepare_scored_sequence(
                    request_index=index,
                    request_id=request["request_id"],
                    context_ids=context_ids,
                    continuation_ids=continuation_ids,
                    max_length=spec.sequence_length,
                )
            )
        sequences.sort(key=lambda item: (len(item.input_ids), item.request_index))
        if any(item.truncated_tokens for item in sequences
               if request_rows[item.request_index]["task"] in reference_tasks):
            raise RuntimeError("reference likelihood requires untruncated source/target text")

        batching = {"mode": "fixed", "micro_batch_size": spec.micro_batch_size}
        if spec.batching is not None:
            batching = calibrate_native_batches(
                model, sequences, spec, total_started=total_started
            )
        batch_ranges = score_batch_ranges(
            len(sequences), spec.micro_batch_size, batching.get("buckets")
        )

        num_moe_experts = int(getattr(config.model, "num_moe_experts", 0) or 0)
        is_moe = num_moe_experts > 1
        from ..final_loss.moe_validation_diagnostics import prepare_moe_validation_diagnostics

        router_layers = prepare_moe_validation_diagnostics(
            config.model, required=spec.require_moe_routing_diagnostics
        )
        if router_layers is not None:
            from multilingual_megatron.pipeline.stages.final_loss.routing_batch_context import (
                CURRENT_BATCH, RoutingBatch,
            )

        score_started = time.perf_counter()
        score_rows: list[dict[str, Any] | None] = [None] * len(sequences)
        parity: dict[str, Any] | None = None
        total_scored_tokens = 0
        early_replica_checked = 0
        early_replica_spread_max = 0.0
        early_replica_spread_sum = 0.0
        early_replica_gate: dict[str, Any] | None = None
        batch_count = (
            sum(math.ceil((bucket["stop"] - bucket["start"]) / bucket["batch_size"])
                for bucket in batching["buckets"])
            if "buckets" in batching else math.ceil(len(sequences) / spec.micro_batch_size)
        )
        batch_timings = []
        for batch_index, (start, stop) in enumerate(batch_ranges, start=1):
            chunk = sequences[start:stop]
            cpu_batch = build_score_batch(chunk, pad_token_id=spec.pad_token_id,
                                          minimum_sequence_width=spec.minimum_sequence_width,
                                          sequence_width_multiple=getattr(spec, "sequence_width_multiple", 1))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_started = time.perf_counter()
            if router_layers is not None:
                context_token = CURRENT_BATCH.set(RoutingBatch(
                    cpu_batch["input_valid_mask"], tuple(int(x) for x in cpu_batch["sample_id"].tolist())
                ))
                try:
                    losses = _score_cuda_batch(model, cpu_batch)
                finally:
                    CURRENT_BATCH.reset(context_token)
            else:
                losses = _score_cuda_batch(model, cpu_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_timings.append({"requests": len(chunk), "width": cpu_batch["tokens"].shape[1],
                                  "seconds": time.perf_counter() - forward_started})
            if parity is None:
                parity_count = min(spec.parity_requests, len(chunk))
                parity_batch = build_score_batch(
                    chunk[:parity_count], pad_token_id=spec.pad_token_id,
                    minimum_sequence_width=spec.minimum_sequence_width,
                    sequence_width_multiple=spec.sequence_width_multiple,
                )
                parity_direct = _score_cuda_batch(model, parity_batch)
                parity = _parity_gate(
                    state=state,
                    model=model,
                    batch=parity_batch,
                    direct_losses=parity_direct,
                )
                parity["request_ids"] = [item.request_id for item in chunk[:parity_count]]
                del parity_direct, parity_batch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mask = cpu_batch["loss_mask"].to(device=losses.device, dtype=torch.float64)
            per_request_nll = (losses.to(torch.float64) * mask).sum(dim=1).cpu().tolist()
            batch_loglikelihoods = [-float(value) for value in per_request_nll]
            if early_replica_checked < min(EARLY_REPLICA_REQUESTS, len(sequences)):
                remaining = min(EARLY_REPLICA_REQUESTS, len(sequences)) - early_replica_checked
                checked_values = batch_loglikelihoods[:remaining]
                spread = _distributed_request_score_spread(checked_values)
                early_replica_checked += len(checked_values)
                early_replica_spread_max = max(
                    early_replica_spread_max, spread["max"]
                )
                early_replica_spread_sum += spread["mean"] * len(checked_values)
                if early_replica_checked == min(EARLY_REPLICA_REQUESTS, len(sequences)):
                    early_replica_gate = {
                        "passed": (
                            early_replica_spread_max
                            <= EARLY_REPLICA_SCORE_ABS_TOLERANCE
                        ),
                        "requests": early_replica_checked,
                        "request_score_abs_spread_max": early_replica_spread_max,
                        "request_score_abs_spread_mean": (
                            early_replica_spread_sum / early_replica_checked
                        ),
                        "request_score_abs_tolerance": (
                            EARLY_REPLICA_SCORE_ABS_TOLERANCE
                        ),
                    }
                    if not early_replica_gate["passed"]:
                        raise RuntimeError(
                            "early expert-parallel replica gate failed: "
                            f"{early_replica_gate}"
                        )
            for item, nll_sum in zip(chunk, per_request_nll, strict=True):
                score_rows[item.request_index] = {
                    "request_id": item.request_id,
                    "loglikelihood": -float(nll_sum),
                    "is_greedy": False,
                    "scored_tokens": item.continuation_tokens,
                    "input_tokens": len(item.input_ids),
                    "truncated_tokens": item.truncated_tokens,
                }
                total_scored_tokens += item.continuation_tokens
            elapsed = time.perf_counter() - total_started
            gpu_hours = elapsed * world_size / 3600.0
            if gpu_hours > spec.max_gpu_hours:
                raise RuntimeError(
                    f"pilot GPU-hour cap exceeded: {gpu_hours:.4f} > {spec.max_gpu_hours:.4f}"
                )
            if rank == 0 and (batch_index == 1 or batch_index % 100 == 0 or batch_index == batch_count):
                print(
                    f"[native-benchmark] batch {batch_index}/{batch_count}; "
                    f"requests={min(start + len(chunk), len(sequences))}/{len(sequences)}; "
                    f"gpu_hours={gpu_hours:.4f}",
                    flush=True,
                )
            del losses, mask, cpu_batch

        if any(item is None for item in score_rows):
            raise RuntimeError("native scorer did not produce every requested score")
        concrete_scores = [item for item in score_rows if item is not None]
        scoring_seconds = time.perf_counter() - score_started
        _write_replica_diagnostic(
            output_dir=spec.output_dir,
            rank=rank,
            request_manifest_sha256=spec.request_manifest_sha256,
            score_rows=concrete_scores,
        )
        replica_gate = _distributed_replica_gate(concrete_scores, request_rows, reference_tasks=reference_tasks)

        if router_layers is not None:
            from multilingual_megatron.pipeline.stages.final_loss.moe_validation_diagnostics import (
                collect_moe_validation_diagnostics,
            )

            expected_moe_layers = spec.expected_moe_layers
            routing = collect_moe_validation_diagnostics(
                expected_num_layers=expected_moe_layers,
                expected_layer_indices=router_layers,
                expected_num_experts=num_moe_experts,
                expected_sample_count=len(sequences),
                expected_valid_tokens=sum(len(item.input_ids) for item in sequences),
                ownership_mode="replicated",
            )
        else:
            routing = {
                "schema": "multilingual_moe_routing_diagnostics/v1",
                "applicable": is_moe,
                "evidence_available": not is_moe,
                "reason": "not requested" if is_moe else "dense model",
            }

        if rank == 0:
            _atomic_json(
                spec.output_dir / "validation_diagnostics.json",
                {
                    "schema": "stage1_native_megatron_validation_diagnostics/v1",
                    "status": "diagnostic_only",
                    "parity_gate": parity,
                    "early_replica_gate": early_replica_gate,
                    "replica_gate": replica_gate,
                    "moe_routing": routing,
                },
            )

        if router_layers is not None:
            dropped = routing.get("totals", {}).get("dropped_assignments")
            if spec.require_moe_routing_diagnostics and not routing.get(
                "evidence_available", False
            ):
                raise RuntimeError(f"missing required MoE routing evidence: {routing}")
            if isinstance(dropped, int) and dropped > 0:
                raise RuntimeError(f"observed {dropped} dropped expert assignments")

        peak_vram = 0
        if torch.cuda.is_available():
            peak = torch.tensor(
                [torch.cuda.max_memory_allocated()],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            if dist.is_initialized():
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            peak_vram = int(peak.item())
        total_seconds = time.perf_counter() - total_started
        gpu_hours = total_seconds * world_size / 3600.0
        if rank == 0:
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            result: dict[str, Any] = {
                "schema": SCORE_RESULT_SCHEMA,
                "status": "complete",
                "run_id": spec.run_id,
                "score_spec": {
                    "path": str(spec.source_path),
                    "sha256": spec.source_sha256,
                },
                "request_manifest": {
                    "path": str(spec.request_manifest_path),
                    "sha256": spec.request_manifest_sha256,
                    "manifest_sha256": request_manifest["manifest_sha256"],
                },
                "checkpoint": {
                    "path": str(spec.checkpoint_path),
                    "iteration_path": str(spec.checkpoint_iteration_path),
                    "step": spec.checkpoint_step,
                    "training_run_config": {
                        "path": str(spec.training_run_config_path),
                        "sha256": spec.training_run_config_sha256,
                    },
                },
                "tokenizer": {
                    "type": spec.tokenizer_type,
                    "model": str(spec.tokenizer_model_path),
                    "sha256": spec.tokenizer_model_sha256,
                    "files": dict(spec.tokenizer_files_sha256),
                    "vocab_size": spec.tokenizer_vocab_size,
                    "context_token_id": spec.context_token_id,
                    "pad_token_id": spec.pad_token_id,
                },
                "model": {
                    "recipe": spec.recipe_name,
                    "expected_contract": dict(spec.model_contract),
                    "world_size": world_size,
                    "num_moe_experts": num_moe_experts,
                    "gpu_name": gpu_name,
                    "inference_only_setup": {
                        "ddp_wrapper": False,
                        "optimizer_created": False,
                        "prediction_head": "main_next_token",
                        "cuda_graph_impl": config.model.cuda_graph_impl,
                        "moe_reduction_dtype": spec.moe_reduction_dtype,
                        "moe_compute_dtype": spec.moe_compute_dtype,
                    },
                    "greedy_flag_policy": (
                        "false placeholder; frozen tasks exclude exact_match and selected "
                        "multiple-choice accuracy and reference bits-per-byte metrics consume loglikelihoods only"
                    ),
                },
                "parity_gate": parity,
                "early_replica_gate": early_replica_gate,
                "replica_gate": replica_gate,
                "moe_routing": routing,
                "batching": batching,
                "timing": {
                    "forward_batches": batch_timings,
                    "calibration_seconds": batching.get("calibration_seconds", 0.0),
                    "setup_seconds": setup_seconds,
                    "scoring_seconds": scoring_seconds,
                    "total_seconds": total_seconds,
                    "gpu_count": world_size,
                    "gpu_hours": gpu_hours,
                    "max_gpu_hours": spec.max_gpu_hours,
                    "requests_per_second": len(concrete_scores) / scoring_seconds,
                    "scored_tokens_per_second": total_scored_tokens / scoring_seconds,
                    "peak_vram_bytes": max(peak_vram, batching.get("peak_allocated_bytes", 0)),
                },
                "counts": {
                    "requests": len(concrete_scores),
                    "scored_tokens": total_scored_tokens,
                    "truncated_requests": sum(
                        int(item["truncated_tokens"] > 0) for item in concrete_scores
                    ),
                },
                "scores": concrete_scores,
            }
            unhashed = dict(result)
            result["result_sha256"] = _payload_sha256(unhashed)
            _atomic_json(spec.output_dir / "scores.json", result)
            (spec.output_dir / ".scoring_complete").write_text("ok\n", encoding="utf-8")
        if dist.is_initialized():
            dist.barrier()
    finally:
        evaluation_scope.close()
        # A rank-local scoring error must reach torchrun immediately. Collective
        # shutdown here would wait forever for peers still inside the forward pass.
        if sys.exc_info()[0] is None:
            try:
                _finish_train(state, checkpoint_manager)
            finally:
                _maybe_destroy_process_group(should_destroy_process_group)


__all__ = ["run_native_benchmark_scoring"]
