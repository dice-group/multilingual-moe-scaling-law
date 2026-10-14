"""Opt-in runtime patches for study-local Megatron Bridge runs."""

from __future__ import annotations

import os
from typing import Any


def install_bridge_cp_rank_compat_patch() -> bool:
    """Patch Bridge GPT batch slicing for newer Megatron-Core signatures.

    This is only for temporary override-checkout experiments. Megatron-Core
    core_r0.18 requires an explicit ``is_hybrid_cp`` argument, while the
    currently vendored Bridge calls the older signature. Keep this opt-in so
    production runs on the pinned submodule are unaffected.
    """

    if str(os.environ.get("PROOF_ENABLE_BRIDGE_CP_RANK_COMPAT", "")).strip() != "1":
        return False

    try:
        import inspect

        import megatron.bridge.training.gpt_step as gpt_step
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_mcore_compat] could not import gpt_step: {exc}", flush=True)
        return False

    current = getattr(gpt_step, "get_batch_on_this_cp_rank", None)
    if current is None:
        print("[study_mcore_compat] gpt_step has no get_batch_on_this_cp_rank", flush=True)
        return False
    if getattr(current, "_study_bridge_cp_rank_compat", False):
        return True

    try:
        signature = inspect.signature(current)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get("is_hybrid_cp")
    if parameter is None or parameter.default is not inspect.Parameter.empty:
        return False

    original_get_batch_on_this_cp_rank = current

    def _get_batch_on_this_cp_rank_compat(batch: Any, *args: Any, **kwargs: Any):
        kwargs.setdefault("is_hybrid_cp", False)
        return original_get_batch_on_this_cp_rank(batch, *args, **kwargs)

    _get_batch_on_this_cp_rank_compat._study_bridge_cp_rank_compat = True  # type: ignore[attr-defined]
    _get_batch_on_this_cp_rank_compat._study_original = original_get_batch_on_this_cp_rank  # type: ignore[attr-defined]
    gpt_step.get_batch_on_this_cp_rank = _get_batch_on_this_cp_rank_compat
    print(
        "[study_mcore_compat] installed get_batch_on_this_cp_rank is_hybrid_cp=False wrapper",
        flush=True,
    )
    return True


def install_ddp_default_stream_alignment_patch() -> bool:
    """Construct Bridge DDP on the active CUDA stream.

    Bridge's unimodal ``_ddp_wrap`` constructs DDP on a dedicated stream.
    PyTorch retains the resulting ``AccumulateGrad`` nodes, so later work on
    the active stream emits a mismatch warning and makes scoped graph capture
    unsafe.  Keep the workaround opt-in and process-local, intercepting only
    the ``torch.cuda.Stream()`` call made directly by ``_ddp_wrap``.
    """

    if str(os.environ.get("PROOF_ALIGN_DDP_INIT_TO_ACTIVE_STREAM", "")).strip() != "1":
        return False

    try:
        import inspect

        import torch
        import megatron.bridge.models.common.unimodal as unimodal
        import megatron.bridge.models.model_provider as model_provider
    except Exception as exc:  # pragma: no cover - depends on Megatron runtime.
        print(f"[study_ddp_stream] could not install alignment patch: {exc}", flush=True)
        return False

    current = unimodal._ddp_wrap
    if getattr(current, "_study_ddp_default_stream_alignment", False):
        return True

    original_get_model = current
    original_code = getattr(original_get_model, "__code__", None)
    if original_code is None:
        print("[study_ddp_stream] Bridge _ddp_wrap has no Python code object", flush=True)
        return False

    def _get_model_with_aligned_ddp_stream(*args: Any, **kwargs: Any):
        original_stream_constructor = torch.cuda.Stream

        def _stream_constructor(*stream_args: Any, **stream_kwargs: Any):
            caller = inspect.currentframe().f_back
            if caller is not None and caller.f_code is original_code:
                return torch.cuda.current_stream()
            return original_stream_constructor(*stream_args, **stream_kwargs)

        torch.cuda.Stream = _stream_constructor
        try:
            return original_get_model(*args, **kwargs)
        finally:
            torch.cuda.Stream = original_stream_constructor

    _get_model_with_aligned_ddp_stream._study_ddp_default_stream_alignment = True  # type: ignore[attr-defined]
    _get_model_with_aligned_ddp_stream._study_original = original_get_model  # type: ignore[attr-defined]
    if model_provider._ddp_wrap is not original_get_model:
        print(
            "[study_ddp_stream] native provider _ddp_wrap alias does not match source",
            flush=True,
        )
        return False
    unimodal._ddp_wrap = _get_model_with_aligned_ddp_stream
    model_provider._ddp_wrap = _get_model_with_aligned_ddp_stream
    print("[study_ddp_stream] aligning DDP initialization to the active CUDA stream", flush=True)
    return True


def install_moe_allgather_ep1_topk1_fastpath() -> bool:
    """Install a narrow EP1/top-k=1 MoE allgather dispatcher fast path.

    This is intentionally opt-in and process-local. It avoids editing the
    Megatron-LM submodule while letting us measure whether removing one
    dynamic scalar extraction from the all-local dropless path improves the
    current H100 EP1 E4 bottleneck.
    """

    if str(os.environ.get("PROOF_ENABLE_MOE_EP1_TOPK1_FASTPATH", "")).strip() != "1":
        return False

    try:
        from megatron.core.transformer.moe import token_dispatcher
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        if str(os.environ.get("PROOF_MOE_EP1_TOPK1_FASTPATH_VERBOSE", "")).strip() == "1":
            print(f"[study_moe_fastpath] could not import token_dispatcher: {exc}", flush=True)
        return False

    dispatcher_cls = token_dispatcher.MoEAllGatherTokenDispatcher
    current = dispatcher_cls.dispatch_postprocess
    if getattr(current, "_study_ep1_topk1_fastpath", False):
        return True

    original_dispatch_postprocess = current

    def _supports_fastpath(self: Any, hidden_states: Any, probs: Any) -> bool:
        config = getattr(self, "config", None)
        if config is None:
            return False
        if int(getattr(self, "tp_size", 1)) != 1 or int(getattr(self, "ep_size", 1)) != 1:
            return False
        if int(getattr(config, "moe_router_topk", 0)) != 1:
            return False
        if getattr(config, "moe_expert_capacity_factor", None) is not None:
            return False
        if bool(getattr(config, "moe_pad_expert_input_to_capacity", False)):
            return False
        if getattr(self, "routing_map", None) is None:
            return False
        local_indices = list(getattr(self, "local_expert_indices", []) or [])
        if not local_indices:
            return False
        num_moe_experts = int(getattr(config, "num_moe_experts", len(local_indices)))
        if int(getattr(self, "num_local_experts", len(local_indices))) != num_moe_experts:
            return False
        if local_indices[0] != 0 or local_indices[-1] != num_moe_experts - 1:
            return False
        if getattr(hidden_states, "shape", None) is None or getattr(probs, "shape", None) is None:
            return False
        if hidden_states.shape[0] != self.routing_map.shape[0]:
            return False
        return probs.shape[0] == self.routing_map.shape[0]

    def _dispatch_postprocess_fastpath(self: Any, hidden_states: Any, probs: Any):
        if not _supports_fastpath(self, hidden_states, probs):
            return original_dispatch_postprocess(self, hidden_states, probs)

        self.hidden_shape_before_permute = hidden_states.shape
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()
        num_out_tokens = int(hidden_states.shape[0])

        (
            permuted_local_hidden_states,
            _,
            self.reversed_local_input_permutation_mapping,
            _,
            _,
        ) = token_dispatcher.permute(
            hidden_states,
            self.local_map,
            num_out_tokens=num_out_tokens,
            fused=self.config.moe_permute_fusion,
        )

        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs

    _dispatch_postprocess_fastpath._study_ep1_topk1_fastpath = True  # type: ignore[attr-defined]
    _dispatch_postprocess_fastpath._study_original = original_dispatch_postprocess  # type: ignore[attr-defined]
    dispatcher_cls.dispatch_postprocess = _dispatch_postprocess_fastpath
    if str(os.environ.get("PROOF_MOE_EP1_TOPK1_FASTPATH_VERBOSE", "")).strip() == "1":
        print("[study_moe_fastpath] installed EP1 top-k=1 dispatch_postprocess patch", flush=True)
    return True


def install_moe_alltoall_sync_profiler() -> bool:
    """Install an opt-in profiler for MoE all-to-all DtoH synchronization.

    This is diagnostic only. It times the study-relevant Python wall-clock cost
    of the dispatcher metadata synchronization points without editing the
    Megatron-LM submodule. Keep it opt-in because it adds lightweight Python
    timing work to every MoE all-to-all dispatcher call.
    """

    if str(os.environ.get("PROOF_PROFILE_MOE_A2A_SYNC", "")).strip() != "1":
        return False

    try:
        import atexit
        import json
        import time
        from collections import defaultdict

        from megatron.core.transformer.moe import token_dispatcher
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_moe_a2a_sync] could not install profiler: {exc}", flush=True)
        return False

    dispatcher_cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    current = dispatcher_cls._maybe_dtoh_and_synchronize
    if getattr(current, "_study_a2a_sync_profiler", False):
        return True

    original_maybe_dtoh = current
    raw_points = str(os.environ.get("PROOF_PROFILE_MOE_A2A_SYNC_POINTS", "")).strip()
    profile_points = {item.strip() for item in raw_points.split(",") if item.strip()}
    stats: dict[tuple[str, str, str, int, int, int], dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "total_ms": 0.0, "max_ms": 0.0}
    )

    def _rank() -> int:
        for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            raw = os.environ.get(name)
            if raw is not None:
                try:
                    return int(raw)
                except ValueError:
                    continue
        return -1

    def _profiled_maybe_dtoh(self: Any, point: str, tokens_per_expert: Any = None):
        dtoh_point = str(getattr(self, "cuda_dtoh_point", ""))
        sync_point = str(getattr(self, "cuda_sync_point", ""))
        ep_size = int(getattr(self, "ep_size", 0) or 0)
        tp_size = int(getattr(self, "tp_size", 0) or 0)
        num_local_experts = int(getattr(self, "num_local_experts", 0) or 0)
        measure = point == dtoh_point or point == sync_point
        if profile_points and point not in profile_points:
            measure = False
        if not measure:
            return original_maybe_dtoh(self, point, tokens_per_expert)

        start = time.perf_counter()
        result = original_maybe_dtoh(self, point, tokens_per_expert)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        key = (str(point), dtoh_point, sync_point, ep_size, tp_size, num_local_experts)
        bucket = stats[key]
        bucket["calls"] += 1.0
        bucket["total_ms"] += elapsed_ms
        bucket["max_ms"] = max(bucket["max_ms"], elapsed_ms)
        return result

    def _emit_summary() -> None:
        rank = _rank()
        for (point, dtoh_point, sync_point, ep_size, tp_size, num_local_experts), bucket in sorted(
            stats.items()
        ):
            calls = int(bucket["calls"])
            if calls <= 0:
                continue
            total_ms = float(bucket["total_ms"])
            payload = {
                "rank": rank,
                "point": point,
                "cuda_dtoh_point": dtoh_point,
                "cuda_sync_point": sync_point,
                "ep_size": ep_size,
                "tp_size": tp_size,
                "num_local_experts": num_local_experts,
                "calls": calls,
                "total_ms": round(total_ms, 3),
                "mean_ms": round(total_ms / calls, 6),
                "max_ms": round(float(bucket["max_ms"]), 6),
            }
            print(f"[study_moe_a2a_sync] {json.dumps(payload, sort_keys=True)}", flush=True)

    _profiled_maybe_dtoh._study_a2a_sync_profiler = True  # type: ignore[attr-defined]
    _profiled_maybe_dtoh._study_original = original_maybe_dtoh  # type: ignore[attr-defined]
    dispatcher_cls._maybe_dtoh_and_synchronize = _profiled_maybe_dtoh
    atexit.register(_emit_summary)
    point_note = ",".join(sorted(profile_points)) if profile_points else "dtoh_or_sync_points"
    print(f"[study_moe_a2a_sync] installed MoE alltoall sync profiler points={point_note}", flush=True)
    return True


def install_moe_alltoall_nccl_stream_patch() -> bool:
    """Force the standard MoE all-to-all dispatcher onto its async NCCL path.

    This is an opt-in systems probe. It only toggles the dispatcher's existing
    ``use_nccl_stream`` flag after construction; it does not alter routing,
    capacity, token dropping, or padding semantics.
    """

    if str(os.environ.get("PROOF_ENABLE_MOE_A2A_NCCL_STREAM", "")).strip() != "1":
        return False

    try:
        from megatron.core.transformer.moe import token_dispatcher
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_moe_a2a_nccl_stream] could not import token_dispatcher: {exc}", flush=True)
        return False

    dispatcher_cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    current = dispatcher_cls.__init__
    if getattr(current, "_study_a2a_nccl_stream_patch", False):
        return True

    original_init = current

    def _init_with_nccl_stream(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.use_nccl_stream = True

    _init_with_nccl_stream._study_a2a_nccl_stream_patch = True  # type: ignore[attr-defined]
    _init_with_nccl_stream._study_original = original_init  # type: ignore[attr-defined]
    dispatcher_cls.__init__ = _init_with_nccl_stream
    print("[study_moe_a2a_nccl_stream] forcing MoE alltoall use_nccl_stream=True", flush=True)
    return True


def install_moe_alltoall_batched_dispatch_patch() -> bool:
    """Launch the token/probability dispatch all-to-alls before waiting.

    The standard all-to-all dispatcher sends hidden states and routing
    probabilities through two consecutive collectives and waits after each one.
    This opt-in probe preserves split sizes, routing, probability precision, and
    autograd semantics, but launches both forward dispatch collectives first and
    then waits on both handles. It targets host/collective scheduling overhead
    without changing the scientific recipe.
    """

    if str(os.environ.get("PROOF_ENABLE_MOE_A2A_BATCHED_DISPATCH", "")).strip() != "1":
        return False

    try:
        import torch

        from megatron.core.transformer.moe import token_dispatcher
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_moe_a2a_batched_dispatch] could not import token_dispatcher: {exc}", flush=True)
        return False

    dispatcher_cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    current = dispatcher_cls.token_dispatch
    if getattr(current, "_study_a2a_batched_dispatch_patch", False):
        return True

    original_token_dispatch = current

    def _world_size(group: Any) -> int:
        size = getattr(group, "size", None)
        if callable(size):
            return int(size())
        return int(torch.distributed.get_world_size(group=group))

    def _output_like(input_tensor: Any, output_split_sizes: Any) -> Any:
        if output_split_sizes is None:
            return torch.empty_like(input_tensor)
        return input_tensor.new_empty(
            size=[sum(output_split_sizes)] + list(input_tensor.size()[1:]),
            dtype=input_tensor.dtype,
            device=torch.cuda.current_device(),
        )

    class _BatchedDispatchAllToAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx, group, tokens, probs, output_split_sizes, input_split_sizes):
            ctx.group = group
            ctx.output_split_sizes = output_split_sizes
            ctx.input_split_sizes = input_split_sizes

            if _world_size(group) == 1:
                return tokens, probs

            tokens = tokens.contiguous()
            probs = probs.contiguous()
            output_tokens = _output_like(tokens, output_split_sizes)
            output_probs = _output_like(probs, output_split_sizes)
            token_handle = torch.distributed.all_to_all_single(
                output_tokens,
                tokens,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
            prob_handle = torch.distributed.all_to_all_single(
                output_probs,
                probs,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
            token_handle.wait()
            prob_handle.wait()
            return output_tokens, output_probs

        @staticmethod
        def backward(ctx, grad_tokens, grad_probs):
            if _world_size(ctx.group) == 1:
                return None, grad_tokens, grad_probs, None, None

            grad_tokens = grad_tokens.contiguous()
            grad_probs = grad_probs.contiguous()
            output_grad_tokens = _output_like(grad_tokens, ctx.input_split_sizes)
            output_grad_probs = _output_like(grad_probs, ctx.input_split_sizes)
            token_handle = torch.distributed.all_to_all_single(
                output_grad_tokens,
                grad_tokens,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
                async_op=True,
            )
            prob_handle = torch.distributed.all_to_all_single(
                output_grad_probs,
                grad_probs,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
                async_op=True,
            )
            token_handle.wait()
            prob_handle.wait()
            return None, output_grad_tokens, output_grad_probs, None, None

    def _token_dispatch_batched(self: Any, permutated_local_input_tokens: Any, permuted_probs: Any):
        if getattr(self, "shared_experts", None) is not None:
            return original_token_dispatch(self, permutated_local_input_tokens, permuted_probs)

        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )
        return _BatchedDispatchAllToAll.apply(
            self.ep_group,
            permutated_local_input_tokens,
            permuted_probs,
            self.output_splits,
            self.input_splits,
        )

    _token_dispatch_batched._study_a2a_batched_dispatch_patch = True  # type: ignore[attr-defined]
    _token_dispatch_batched._study_original = original_token_dispatch  # type: ignore[attr-defined]
    dispatcher_cls.token_dispatch = _token_dispatch_batched
    print(
        "[study_moe_a2a_batched_dispatch] batching hidden/probability dispatch alltoalls",
        flush=True,
    )
    return True


def install_moe_alltoall_cpu_split_list_patch() -> bool:
    """Convert MoE all-to-all CPU split metadata to Python lists.

    The standard dropless all-to-all path copies dynamic split tensors to CPU
    before calling ``torch.distributed.all_to_all_single``. This opt-in probe
    keeps the same synchronization and routing semantics, but normalizes the
    small CPU split arrays to plain integer lists before the collective.
    """

    if str(os.environ.get("PROOF_ENABLE_MOE_A2A_CPU_SPLIT_LISTS", "")).strip() != "1":
        return False

    try:
        from megatron.core.transformer.moe import token_dispatcher
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_moe_a2a_split_lists] could not import token_dispatcher: {exc}", flush=True)
        return False

    dispatcher_cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    current = dispatcher_cls._maybe_dtoh_and_synchronize
    if getattr(current, "_study_a2a_cpu_split_list_patch", False):
        return True

    original_maybe_dtoh = current

    def _to_int_list(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, tuple):
            return [int(item) for item in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            listed = tolist()
            if isinstance(listed, list):
                return [int(item) for item in listed]
            return int(listed)
        return value

    def _maybe_dtoh_with_split_lists(self: Any, point: str, tokens_per_expert: Any = None):
        result = original_maybe_dtoh(self, point, tokens_per_expert)
        if point == getattr(self, "cuda_sync_point", ""):
            self.input_splits = _to_int_list(getattr(self, "input_splits", None))
            self.output_splits = _to_int_list(getattr(self, "output_splits", None))
            self.output_splits_tp = _to_int_list(getattr(self, "output_splits_tp", None))
        return result

    _maybe_dtoh_with_split_lists._study_a2a_cpu_split_list_patch = True  # type: ignore[attr-defined]
    _maybe_dtoh_with_split_lists._study_original = original_maybe_dtoh  # type: ignore[attr-defined]
    dispatcher_cls._maybe_dtoh_and_synchronize = _maybe_dtoh_with_split_lists
    print("[study_moe_a2a_split_lists] converting MoE alltoall CPU splits to lists", flush=True)
    return True


def install_moe_alltoall_padded_equal_splits_patch() -> bool:
    """Pad MoE EP all-to-all-v chunks to equal sizes before communication.

    This opt-in systems probe keeps token routing, expert assignment, token
    dropping, and expert computation unchanged. It pads only the communication
    buffers to test whether variable-size all-to-all is the remaining wall for
    EP-heavy D/E32 rows, then removes the padding immediately after the
    collective.
    """

    if str(os.environ.get("PROOF_ENABLE_MOE_A2A_PADDED_EQUAL_SPLITS", "")).strip() != "1":
        return False

    try:
        import atexit
        import json
        from collections import defaultdict

        import torch

        from megatron.core.transformer.moe import token_dispatcher
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_moe_a2a_padded_equal] could not import token_dispatcher: {exc}", flush=True)
        return False

    dispatcher_cls = token_dispatcher.MoEAlltoAllTokenDispatcher
    current = dispatcher_cls.token_dispatch
    if getattr(current, "_study_a2a_padded_equal_splits_patch", False):
        return True

    original_token_dispatch = current
    original_token_combine = dispatcher_cls.token_combine
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0.0,
            "input_rows": 0.0,
            "output_rows": 0.0,
            "padded_rows": 0.0,
            "max_chunk": 0.0,
        }
    )

    def _world_size(group: Any) -> int:
        size = getattr(group, "size", None)
        if callable(size):
            return int(size())
        return int(torch.distributed.get_world_size(group=group))

    def _rank() -> int:
        for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            raw = os.environ.get(name)
            if raw is not None:
                try:
                    return int(raw)
                except ValueError:
                    continue
        return -1

    def _to_int_list(value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, tuple):
            return [int(item) for item in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            listed = tolist()
            if isinstance(listed, list):
                return [int(item) for item in listed]
            return [int(listed)]
        return [int(value)]

    def _global_chunk_size(input_tensor: Any, group: Any, input_splits: list[int]) -> int:
        local_max = max(input_splits) if input_splits else 0
        static_raw = str(os.environ.get("PROOF_MOE_A2A_PADDED_EQUAL_MIN_CHUNK", "")).strip()
        if static_raw:
            local_max = max(local_max, int(static_raw))
        if _world_size(group) <= 1:
            return local_max
        max_tensor = torch.tensor([local_max], dtype=torch.long, device=input_tensor.device)
        torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX, group=group)
        return int(max_tensor.item())

    def _pad_chunks(input_tensor: Any, input_splits: list[int], chunk: int) -> Any:
        trailing_shape = list(input_tensor.shape[1:])
        chunks = list(input_tensor.split(input_splits, dim=0))
        padded_chunks = []
        for split_size, chunk_tensor in zip(input_splits, chunks):
            pad_rows = chunk - int(split_size)
            if pad_rows < 0:
                raise RuntimeError(
                    f"padded alltoall chunk {chunk} is smaller than split {split_size}"
                )
            if pad_rows:
                padding = input_tensor.new_zeros([pad_rows] + trailing_shape)
                chunk_tensor = torch.cat((chunk_tensor, padding), dim=0)
            padded_chunks.append(chunk_tensor)
        return torch.cat(padded_chunks, dim=0)

    def _unpad_chunks(output_tensor: Any, output_splits: list[int], chunk: int) -> Any:
        chunks = output_tensor.split(chunk, dim=0)
        return torch.cat(
            [chunk_tensor[: int(split_size)] for split_size, chunk_tensor in zip(output_splits, chunks)],
            dim=0,
        )

    def _record(direction: str, input_splits: list[int], output_splits: list[int], chunk: int) -> None:
        bucket = stats[direction]
        input_rows = float(sum(input_splits))
        output_rows = float(sum(output_splits))
        padded_rows = float(chunk * len(input_splits))
        bucket["calls"] += 1.0
        bucket["input_rows"] += input_rows
        bucket["output_rows"] += output_rows
        bucket["padded_rows"] += padded_rows
        bucket["max_chunk"] = max(bucket["max_chunk"], float(chunk))

    def _padded_equal_all_to_all(
        group: Any,
        input_tensor: Any,
        output_splits: Any,
        input_splits: Any,
        direction: str,
        chunk: int | None = None,
    ) -> tuple[Any, int]:
        input_split_list = _to_int_list(input_splits)
        output_split_list = _to_int_list(output_splits)
        world_size = _world_size(group)
        if (
            world_size <= 1
            or input_split_list is None
            or output_split_list is None
            or len(input_split_list) != world_size
            or len(output_split_list) != world_size
        ):
            return (
                token_dispatcher.all_to_all(
                    group,
                    input_tensor,
                    output_splits,
                    input_splits,
                    use_nccl_stream=False,
                ),
                0,
            )

        if chunk is None:
            chunk = _global_chunk_size(input_tensor, group, input_split_list)
        packed_input = _pad_chunks(input_tensor.contiguous(), input_split_list, chunk)
        packed_output = torch.empty_like(packed_input)
        torch.distributed.all_to_all_single(packed_output, packed_input, group=group)
        output = _unpad_chunks(packed_output, output_split_list, chunk)
        _record(direction, input_split_list, output_split_list, chunk)
        return output, chunk

    class _PaddedEqualAllToAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx, group, input_tensor, output_splits, input_splits, direction):
            output, chunk = _padded_equal_all_to_all(
                group, input_tensor, output_splits, input_splits, str(direction)
            )
            ctx.group = group
            ctx.output_splits = output_splits
            ctx.input_splits = input_splits
            ctx.chunk = chunk
            ctx.direction = str(direction)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            grad_input, _ = _padded_equal_all_to_all(
                ctx.group,
                grad_output,
                ctx.input_splits,
                ctx.output_splits,
                f"{ctx.direction}_backward",
                chunk=ctx.chunk or None,
            )
            return None, grad_input, None, None, None

    def _token_dispatch_padded_equal(
        self: Any, permutated_local_input_tokens: Any, permuted_probs: Any
    ):
        if getattr(self, "shared_experts", None) is not None:
            return original_token_dispatch(self, permutated_local_input_tokens, permuted_probs)

        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )
        global_input_tokens = _PaddedEqualAllToAll.apply(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            "dispatch_tokens",
        )
        global_probs = _PaddedEqualAllToAll.apply(
            self.ep_group,
            permuted_probs,
            self.output_splits,
            self.input_splits,
            "dispatch_probs",
        )
        return global_input_tokens, global_probs

    def _token_combine_padded_equal(
        self: Any,
        hidden_states: Any,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        if getattr(self, "shared_experts", None) is not None:
            return original_token_combine(self, hidden_states, async_finish, allocate_on_comm_stream)
        return _PaddedEqualAllToAll.apply(
            self.ep_group,
            hidden_states,
            self.input_splits,
            self.output_splits,
            "combine_tokens",
        )

    def _emit_summary() -> None:
        rank = _rank()
        for direction, bucket in sorted(stats.items()):
            calls = int(bucket["calls"])
            if calls <= 0:
                continue
            input_rows = float(bucket["input_rows"])
            padded_rows = float(bucket["padded_rows"])
            payload = {
                "rank": rank,
                "direction": direction,
                "calls": calls,
                "input_rows": int(input_rows),
                "output_rows": int(bucket["output_rows"]),
                "padded_rows": int(padded_rows),
                "padding_overhead_pct": round(
                    ((padded_rows / input_rows) - 1.0) * 100.0 if input_rows else 0.0, 3
                ),
                "max_chunk": int(bucket["max_chunk"]),
            }
            print(f"[study_moe_a2a_padded_equal] {json.dumps(payload, sort_keys=True)}", flush=True)

    _token_dispatch_padded_equal._study_a2a_padded_equal_splits_patch = True  # type: ignore[attr-defined]
    _token_dispatch_padded_equal._study_original = original_token_dispatch  # type: ignore[attr-defined]
    _token_combine_padded_equal._study_a2a_padded_equal_splits_patch = True  # type: ignore[attr-defined]
    _token_combine_padded_equal._study_original = original_token_combine  # type: ignore[attr-defined]
    dispatcher_cls.token_dispatch = _token_dispatch_padded_equal
    dispatcher_cls.token_combine = _token_combine_padded_equal
    atexit.register(_emit_summary)
    print("[study_moe_a2a_padded_equal] padding MoE alltoall-v chunks to equal splits", flush=True)
    return True


def install_te_grouped_mlp_inspector() -> bool:
    """Install an opt-in model-construction inspector for TEGroupedMLP modules."""

    if str(os.environ.get("PROOF_INSPECT_TE_GROUPED_MLP", "")).strip() != "1":
        return False

    try:
        import inspect
        import importlib.metadata as importlib_metadata

        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from transformer_engine.pytorch.ops import ScaledSwiGLU
    except Exception as exc:  # pragma: no cover - depends on Bridge runtime.
        print(f"[study_te_grouped_mlp] could not install inspector: {exc}", flush=True)
        return False

    current = GPTModelProvider.provide
    if getattr(current, "_study_te_grouped_mlp_inspector", False):
        return True

    require_fused = str(os.environ.get("PROOF_REQUIRE_TE_GROUPED_MLP_FUSED", "")).strip() == "1"
    require_moe_act_recompute = (
        str(os.environ.get("PROOF_REQUIRE_TE_MOE_ACT_RECOMPUTE", "")).strip() == "1"
    )
    require_layernorm_recompute = (
        str(os.environ.get("PROOF_REQUIRE_LAYERNORM_RECOMPUTE", "")).strip() == "1"
    )
    activation_recompute_api = (
        "activation_recompute_in_mlp" in inspect.signature(ScaledSwiGLU).parameters
    )
    original_provide = current

    def _version(package: str) -> str:
        try:
            return importlib_metadata.version(package)
        except Exception:
            return "unknown"

    def _provide_with_inspection(self: Any, *args: Any, **kwargs: Any):
        model = original_provide(self, *args, **kwargs)
        te_version = _version("transformer-engine")
        mbridge_version = _version("megatron-bridge")
        print(
            "[study_te_grouped_mlp] "
            f"transformer_engine={te_version} megatron_bridge={mbridge_version} "
            f"use_transformer_engine_op_fuser={getattr(self, 'use_transformer_engine_op_fuser', None)} "
            f"use_te_activation_func={getattr(self, 'use_te_activation_func', None)} "
            f"moe_grouped_gemm={getattr(self, 'moe_grouped_gemm', None)} "
            f"num_moe_experts={getattr(self, 'num_moe_experts', None)}",
            flush=True,
        )

        grouped_mlp_count = 0
        fused_true_count = 0
        fused_attr_absent_count = 0
        activation_recompute_true_count = 0
        layernorm_layer_count = 0
        layernorm_recompute_true_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "recompute_input_layernorm") and hasattr(
                module, "recompute_pre_mlp_layernorm"
            ):
                layernorm_layer_count += 1
                input_recompute = bool(module.recompute_input_layernorm)
                pre_mlp_recompute = bool(module.recompute_pre_mlp_layernorm)
                input_class = module.input_layernorm.__class__.__name__
                pre_mlp_class = module.pre_mlp_layernorm.__class__.__name__
                input_required = input_class != "IdentityOp"
                pre_mlp_required = pre_mlp_class != "IdentityOp"
                enabled = (not input_required or input_recompute) and (
                    not pre_mlp_required or pre_mlp_recompute
                )
                if enabled:
                    layernorm_recompute_true_count += 1
                print(
                    "[study_layernorm_recompute] "
                    f"module={name} layernorm_recompute={enabled} "
                    "available_layernorm_paths_recomputed="
                    f"{enabled} input_layernorm={input_recompute} "
                    f"input_layernorm_class={input_class} "
                    f"pre_mlp_layernorm={pre_mlp_recompute} "
                    f"pre_mlp_layernorm_class={pre_mlp_class}",
                    flush=True,
                )
            if module.__class__.__name__ != "TEGroupedMLP":
                continue
            grouped_mlp_count += 1
            fused_attr = getattr(module, "_with_fused_impl", "<absent>")
            fused_forward = hasattr(module, "_fused_forward")
            if fused_attr == "<absent>":
                fused_attr_absent_count += 1
            if fused_attr is True:
                fused_true_count += 1
            activation_recompute = bool(getattr(module, "activation_recompute", False))
            if activation_recompute:
                activation_recompute_true_count += 1
            print(
                "[study_te_grouped_mlp] "
                f"module={name} class={module.__class__.__name__} "
                f"with_fused_impl={fused_attr} has_fused_forward={fused_forward} "
                f"activation_recompute={activation_recompute} "
                f"activation_recompute_api={activation_recompute_api} "
                f"linear_fc1={module.linear_fc1.__class__.__name__} "
                f"linear_fc2={module.linear_fc2.__class__.__name__} "
                f"activation={getattr(getattr(module, 'activation_func', None), '__class__', type(None)).__name__}",
                flush=True,
            )

        print(
            "[study_te_grouped_mlp] "
            f"summary count={grouped_mlp_count} fused_true={fused_true_count} "
            f"fused_attr_absent={fused_attr_absent_count} "
            f"activation_recompute_true={activation_recompute_true_count} "
            f"activation_recompute_api={activation_recompute_api}",
            flush=True,
        )
        print(
            "[study_layernorm_recompute] "
            f"summary count={layernorm_layer_count} "
            f"enabled={layernorm_recompute_true_count}",
            flush=True,
        )
        if require_fused and (grouped_mlp_count == 0 or fused_true_count != grouped_mlp_count):
            raise RuntimeError(
                "PROOF_REQUIRE_TE_GROUPED_MLP_FUSED=1 but not all TEGroupedMLP modules "
                "reported _with_fused_impl=True."
            )
        if require_moe_act_recompute and (
            not activation_recompute_api
            or grouped_mlp_count == 0
            or activation_recompute_true_count != grouped_mlp_count
        ):
            raise RuntimeError(
                "PROOF_REQUIRE_TE_MOE_ACT_RECOMPUTE=1 but the Transformer Engine API "
                "or one or more TEGroupedMLP modules did not enable fused moe_act recomputation."
            )
        if require_layernorm_recompute and (
            layernorm_layer_count == 0
            or layernorm_recompute_true_count != layernorm_layer_count
        ):
            raise RuntimeError(
                "PROOF_REQUIRE_LAYERNORM_RECOMPUTE=1 but one or more TransformerLayer "
                "modules did not enable every materialized LayerNorm recomputation path."
            )
        return model

    _provide_with_inspection._study_te_grouped_mlp_inspector = True  # type: ignore[attr-defined]
    _provide_with_inspection._study_original = original_provide  # type: ignore[attr-defined]
    GPTModelProvider.provide = _provide_with_inspection
    print("[study_te_grouped_mlp] installed TEGroupedMLP inspector", flush=True)
    return True


def install_mtp_full_recompute_padding_mask_compat_patch() -> bool:
    """Bridge the pinned MTP full-recompute signature mismatch, opt-in only.

    The pinned ``MultiTokenPredictionLayer.forward`` forwards ``padding_mask``
    into ``_checkpointed_forward`` when full recomputation is enabled, but the
    helper has no corresponding parameter.  Its non-recompute path also omits
    that argument when calling ``_proj_and_transformer_layer``.  This wrapper
    therefore only drops that already-unused keyword for the exact older
    helper signature.  It refuses to patch newer runtimes that accept the
    argument and does not alter any upstream source file.
    """

    if str(os.environ.get("PROOF_ENABLE_MTP_FULL_RECOMPUTE_PADDING_MASK_COMPAT", "")).strip() != "1":
        return False

    try:
        import inspect

        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
    except Exception as exc:  # pragma: no cover - depends on the cluster runtime.
        print(f"[study_mtp_recompute_compat] could not import MTP layer: {exc}", flush=True)
        return False

    current = MultiTokenPredictionLayer._checkpointed_forward
    if getattr(current, "_study_mtp_full_recompute_padding_mask_compat", False):
        return True
    try:
        parameters = inspect.signature(current).parameters
    except (TypeError, ValueError):
        return False
    if "padding_mask" in parameters:
        return False

    original_checkpointed_forward = current

    def _checkpointed_forward_compat(self: Any, *args: Any, **kwargs: Any):
        # The MTP non-recompute call path already does not forward this value
        # to ``_proj_and_transformer_layer``.  Match that established behavior
        # only for full recompute, where the pinned caller currently errors.
        if "padding_mask" in kwargs:
            config = getattr(self, "config", None)
            if getattr(config, "recompute_granularity", None) != "full":
                raise RuntimeError(
                    "MTP padding-mask compatibility wrapper was invoked outside full recompute"
                )
            kwargs.pop("padding_mask")
        return original_checkpointed_forward(self, *args, **kwargs)

    _checkpointed_forward_compat._study_mtp_full_recompute_padding_mask_compat = True  # type: ignore[attr-defined]
    _checkpointed_forward_compat._study_original = original_checkpointed_forward  # type: ignore[attr-defined]
    MultiTokenPredictionLayer._checkpointed_forward = _checkpointed_forward_compat
    print(
        "[study_mtp_recompute_compat] installed full-recompute padding_mask signature wrapper",
        flush=True,
    )
    return True


def install_fused_te_moe_act_checkpoint_patch() -> bool:
    """Checkpoint scaled activation output in the narrow fused-TE H100 MoE path.

    Megatron Core's non-fused ``TEGroupedMLP`` path already uses
    ``CheckpointWithoutOutput`` to release the activation between FC1 and FC2.
    The current Transformer Engine operation-fuser path bypasses that mechanism
    on H100 BF16.  This study-local, opt-in shim splits the existing three TE
    operations into FC1 / scaled activation / FC2 containers and applies the
    same MCore checkpoint around only that activation.  It supports the exact
    TE scaled operations constructed by MCore (SwiGLU, clamped QGeGLU, or
    weighted squared ReLU), retains the real TE GroupedLinear ops, and does not
    edit Megatron-LM.

    This is deliberately fail-closed.  It is valid only for the configuration
    exercised by the campaign: fused BF16, TP/ETP1, no FP8/FP4, no activation
    offload/paged stash, and no delayed expert weight-gradient computation.
    """

    if str(os.environ.get("PROOF_ENABLE_FUSED_TE_MOE_ACT_CHECKPOINT", "")).strip() != "1":
        return False

    try:
        import torch

        from megatron.core import tensor_parallel
        from megatron.core.transformer.moe.experts import TEGroupedMLP
    except Exception as exc:  # pragma: no cover - depends on the cluster runtime.
        print(f"[study_te_fused_moe_act] could not install checkpoint patch: {exc}", flush=True)
        return False

    current = TEGroupedMLP._fused_forward
    if getattr(current, "_study_fused_te_moe_act_checkpoint", False):
        return True

    def _unsupported_reason(module: Any, hidden_states: Any) -> str | None:
        config = getattr(module, "config", None)
        if config is None:
            return "missing configuration"
        if not bool(getattr(module, "_with_fused_impl", False)):
            return "TEGroupedMLP is not using the fused implementation"
        if not bool(getattr(module, "activation_recompute", False)):
            return "moe_act activation recomputation is not enabled"
        if bool(getattr(config, "fp8", False)) or bool(getattr(config, "fp4", False)):
            return "FP8/FP4 is not supported by this BF16-only checkpoint shim"
        if bool(getattr(config, "moe_paged_stash", False)):
            return "paged MoE stash is not supported"
        if bool(getattr(config, "moe_apply_probs_on_input", False)):
            return "pre-multiplied router probabilities are not supported"
        if bool(getattr(module, "offload_expert_fc1", False)) or bool(
            getattr(module, "offload_moe_act", False)
        ):
            return "expert activation offload is not supported"
        if bool(getattr(getattr(module, "linear_fc1", None), "delay_wgrad_compute", False)) or bool(
            getattr(getattr(module, "linear_fc2", None), "delay_wgrad_compute", False)
        ):
            return "delayed expert weight-gradient computation is not supported"
        try:
            if int(module.tp_group.size()) != 1:
                return "TP/ETP must be exactly one"
        except Exception:
            return "could not verify the TP/ETP group size"
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.dtype != torch.bfloat16:
            return "the fused checkpoint shim only supports BF16 hidden states"
        return None

    def _require_tensor(value: Any, operation: str) -> Any:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "fused MoE activation checkpoint expected a tensor from "
                f"{operation}, received {type(value).__name__}"
            )
        return value

    def _split_ops(module: Any) -> tuple[Any, Any, Any]:
        cached = getattr(module, "_study_fused_moe_act_checkpoint_ops", None)
        if cached is not None:
            return cached
        if getattr(module, "_fused_ops", None) is not None:
            raise RuntimeError(
                "fused MoE activation checkpoint must be installed before the first fused "
                "TEGroupedMLP forward; refusing to mix cached fused operation containers"
            )

        # _make_fused_ops constructs TE operation shells backed by the original
        # GroupedLinear weights.  The temporary full container is intentionally
        # not retained; the split containers below are held in a tuple, matching
        # MCore's own convention of avoiding duplicate module registration.
        full_ops = module._make_fused_ops()
        if len(full_ops) != 3:
            raise RuntimeError(
                "fused MoE activation checkpoint expected exactly FC1, activation, and FC2 ops"
            )
        fc1_ops = full_ops[0:1]
        activation_ops = full_ops[1:2]
        fc2_ops = full_ops[2:3]
        activation_op = activation_ops[0]
        supported_activation_names = {
            "ScaledSwiGLU",
            "ScaledClampedQGeGLU",
            "ScaledSReLU",
        }
        if activation_op.__class__.__name__ not in supported_activation_names:
            raise RuntimeError(
                "fused MoE activation checkpoint received unsupported TE activation "
                f"{activation_op.__class__.__name__}"
            )
        # This shim owns output-discarding recomputation.  A split TE activation
        # is no longer inside TE's joint GroupedMLP fused op, whose basic path
        # rejects activation_recompute_in_mlp=True.
        if hasattr(activation_op, "activation_recompute_in_mlp"):
            activation_op.activation_recompute_in_mlp = False
        # The original full container emulates DDP pre-forward hooks.  Since it
        # is not executed after splitting, run that exact hook once before FC1.
        fc1_ops.register_forward_pre_hook(module._make_fused_impl_pre_forward_hook())
        module._study_fused_moe_act_checkpoint_ops = (fc1_ops, activation_ops, fc2_ops)
        return module._study_fused_moe_act_checkpoint_ops

    def _fused_forward_with_moe_act_checkpoint(
        self: Any,
        permuted_local_hidden_states: Any,
        tokens_per_expert: Any,
        permuted_probs: Any,
        output_buffer: Any = None,
        grad_input_buffer: Any = None,
    ) -> Any:
        if output_buffer is not None or grad_input_buffer is not None:
            raise RuntimeError(
                "fused MoE activation checkpoint does not support NCCL-EP zero-copy buffers"
            )
        reason = _unsupported_reason(self, permuted_local_hidden_states)
        if reason is not None:
            raise RuntimeError(f"fused MoE activation checkpoint is unsupported: {reason}")

        fc1_ops, activation_ops, fc2_ops = _split_ops(self)
        fc1_output = _require_tensor(
            fc1_ops(permuted_local_hidden_states, tokens_per_expert), "TE GroupedLinear FC1"
        )

        activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
        self.activation_checkpoint = activation_checkpoint

        def _activation_forward(fc1_input: Any, probs: Any) -> Any:
            return _require_tensor(activation_ops(fc1_input, probs), "TE ScaledSwiGLU")

        activation_output = _require_tensor(
            activation_checkpoint.checkpoint(_activation_forward, fc1_output, permuted_probs),
            "checkpointed TE ScaledSwiGLU",
        )
        output = _require_tensor(
            fc2_ops(activation_output, tokens_per_expert), "TE GroupedLinear FC2"
        )
        # FC2 is the direct consumer that saves activation-output views for its
        # backward pass, exactly the contract required by CheckpointWithoutOutput.
        activation_checkpoint.discard_output_and_register_recompute(output)

        if not bool(getattr(self, "_study_fused_moe_act_checkpoint_announced", False)):
            print(
                "[study_te_fused_moe_act] checkpoint_active=True "
                "with_fused_impl=True activation_recompute=True bf16=True tp_size=1 "
                "delay_wgrad_compute=False",
                flush=True,
            )
            self._study_fused_moe_act_checkpoint_announced = True
        return output

    _fused_forward_with_moe_act_checkpoint._study_fused_te_moe_act_checkpoint = True  # type: ignore[attr-defined]
    _fused_forward_with_moe_act_checkpoint._study_original = current  # type: ignore[attr-defined]
    TEGroupedMLP._fused_forward = _fused_forward_with_moe_act_checkpoint
    print(
        "[study_te_fused_moe_act] installed BF16 TP1 fused MoE activation checkpoint patch",
        flush=True,
    )
    return True


def install_p2p_batch_sync_disabled_guard() -> bool:
    """Fail closed unless the live PP communicator disables the legacy device sync.

    Megatron Core documents ``batch_p2p_sync`` as a workaround for an older
    PyTorch ``batch_isend_irecv`` race. The promoted PP4/EP2 run uses PyTorch
    2.11 and its non-interleaved schedule otherwise performs a device-wide
    ``torch.cuda.synchronize()`` after every batched pipeline exchange. This
    guard makes an opt-in configuration experiment auditable without changing
    Megatron-LM source or communication behavior itself.
    """

    if str(os.environ.get("PROOF_REQUIRE_P2P_BATCH_SYNC_DISABLED", "")).strip() != "1":
        return False

    try:
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
    except Exception as exc:  # pragma: no cover - depends on the cluster runtime.
        print(f"[study_p2p_runtime] could not install P2P guard: {exc}", flush=True)
        return False

    current = P2PCommunicator._communicate
    if getattr(current, "_study_p2p_batch_sync_disabled_guard", False):
        return True

    original_communicate = current
    announced = False

    def _communicate_with_guard(self: Any, *args: Any, **kwargs: Any):
        nonlocal announced
        config = getattr(self, "config", None)
        batch_p2p_comm = bool(getattr(config, "batch_p2p_comm", False))
        batch_p2p_sync = bool(getattr(config, "batch_p2p_sync", True))
        overlap_p2p_comm = bool(getattr(config, "overlap_p2p_comm", False))
        if not batch_p2p_comm or batch_p2p_sync or overlap_p2p_comm:
            raise RuntimeError(
                "PROOF_REQUIRE_P2P_BATCH_SYNC_DISABLED=1 requires the live P2P communicator "
                "to use batch_p2p_comm=True, batch_p2p_sync=False, and overlap_p2p_comm=False; "
                f"observed batch_p2p_comm={batch_p2p_comm}, "
                f"batch_p2p_sync={batch_p2p_sync}, overlap_p2p_comm={overlap_p2p_comm}"
            )
        if not announced:
            print(
                "[study_p2p_runtime] batch_p2p_comm=True "
                "batch_p2p_sync=False overlap_p2p_comm=False",
                flush=True,
            )
            announced = True
        return original_communicate(self, *args, **kwargs)

    _communicate_with_guard._study_p2p_batch_sync_disabled_guard = True  # type: ignore[attr-defined]
    _communicate_with_guard._study_original = original_communicate  # type: ignore[attr-defined]
    P2PCommunicator._communicate = _communicate_with_guard
    print("[study_p2p_runtime] installed P2P batch-sync-disabled guard", flush=True)
    return True
