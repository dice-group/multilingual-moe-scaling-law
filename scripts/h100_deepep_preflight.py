#!/usr/bin/env python
from __future__ import annotations

import json
import os
import socket
import sys
import traceback

import torch
import torch.distributed as dist


def _as_int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _as_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _record_hybridep_availability(result: dict[str, object]) -> None:
    """Record whether the installed deep_ep package has MCore's HybridEP symbol."""

    try:
        from deep_ep import HybridEPBuffer  # noqa: F401

        result["deep_ep_has_hybridep"] = True
    except Exception as exc:
        result["deep_ep_has_hybridep"] = False
        result["hybridep_import_error"] = f"{type(exc).__name__}: {exc}"
        if os.environ.get("DEEPEP_PREFLIGHT_REQUIRE_HYBRIDEP", "").strip() == "1":
            raise


def _ensure_nvrx_version_for_mcore_import(result: dict[str, object]) -> None:
    """Patch missing NVRx version metadata for this diagnostic import only."""

    try:
        import nvidia_resiliency_ext as nvrx
    except Exception as exc:
        result["nvrx_importable"] = False
        result["nvrx_import_error"] = f"{type(exc).__name__}: {exc}"
        return

    result["nvrx_importable"] = True
    if hasattr(nvrx, "__version__"):
        result["nvrx_version"] = str(getattr(nvrx, "__version__"))
        result["nvrx_version_shimmed"] = False
        return
    # Megatron-Core imports dist-checkpointing while importing fused_a2a and
    # assumes this attribute exists. The preflight does not use checkpointing;
    # set metadata locally so the dispatcher JIT path can be tested.
    nvrx.__version__ = os.environ.get("DEEPEP_PREFLIGHT_NVRX_VERSION_SHIM", "999.0.0")
    result["nvrx_version"] = str(nvrx.__version__)
    result["nvrx_version_shimmed"] = True


def _build_top1_routing(
    *,
    num_tokens: int,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic top-1 routing tensors in Megatron-Core layout."""

    expert_ids = torch.arange(num_tokens, device=device, dtype=torch.long) % num_experts
    routing_map = torch.zeros(num_tokens, num_experts, device=device, dtype=torch.bool)
    routing_map.scatter_(1, expert_ids.view(-1, 1), True)
    probs = torch.zeros(num_tokens, num_experts, device=device, dtype=torch.float32)
    probs.scatter_(1, expert_ids.view(-1, 1), 1.0)
    return routing_map, probs


def _run_hybridep_jit_preflight(
    *,
    result: dict[str, object],
    device: torch.device,
    num_tokens: int,
    hidden: int,
) -> None:
    """Exercise the MCore HybridEP dispatch/combine JIT path without training."""

    if not _as_bool_env("DEEPEP_PREFLIGHT_HYBRIDEP_JIT"):
        result["hybridep_jit_requested"] = False
        return
    result["hybridep_jit_requested"] = True
    if not result.get("deep_ep_has_hybridep"):
        raise RuntimeError("DEEPEP_PREFLIGHT_HYBRIDEP_JIT=1 requires HybridEPBuffer")

    _ensure_nvrx_version_for_mcore_import(result)
    from megatron.core.transformer.moe import fused_a2a

    if fused_a2a.hybrid_ep_dispatch is None or fused_a2a.hybrid_ep_combine is None:
        raise RuntimeError("Megatron-Core HybridEP dispatch/combine functions are unavailable")

    local_experts = _as_int_env("DEEPEP_PREFLIGHT_LOCAL_EXPERTS", 16)
    num_experts = int(dist.get_world_size()) * local_experts
    fused = _as_bool_env("DEEPEP_PREFLIGHT_HYBRIDEP_FUSED")
    num_sms = _as_int_env("DEEPEP_PREFLIGHT_HYBRIDEP_NUM_SMS", 16)
    num_sms_preprocessing = _as_int_env("DEEPEP_PREFLIGHT_HYBRIDEP_PREPROCESS_SMS", 108)

    fused_a2a.reset_hybrid_ep_buffer()
    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    routing_map, probs = _build_top1_routing(
        num_tokens=num_tokens,
        num_experts=num_experts,
        device=device,
    )
    dispatched, dispatched_probs, _scaling, tokens_per_expert, handle = fused_a2a.hybrid_ep_dispatch(
        x=x,
        routing_map=routing_map,
        probs=probs,
        group=dist.group.WORLD,
        num_local_experts=local_experts,
        num_sms_dispatch_api=num_sms,
        num_sms_combine_api=num_sms,
        fused=fused,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=num_sms_preprocessing,
    )
    torch.cuda.synchronize(device)
    num_permuted_tokens = tokens_per_expert.to(torch.int64).sum()
    combined = fused_a2a.hybrid_ep_combine(
        dispatched,
        handle,
        num_permuted_tokens=num_permuted_tokens,
        pad_multiple=None,
        fused=fused,
    )
    torch.cuda.synchronize(device)
    max_abs_diff = float((combined.float() - x.float()).abs().max().item())
    tolerance = float(os.environ.get("DEEPEP_PREFLIGHT_HYBRIDEP_TOLERANCE", "0.0"))
    if max_abs_diff > tolerance:
        raise RuntimeError(
            "HybridEP dispatch/combine roundtrip mismatch: "
            f"max_abs_diff={max_abs_diff} tolerance={tolerance}"
        )
    result.update(
        {
            "hybridep_jit_status": "ok",
            "hybridep_jit_fused": fused,
            "hybridep_jit_local_experts": local_experts,
            "hybridep_jit_num_experts": num_experts,
            "hybridep_jit_num_sms": num_sms,
            "hybridep_jit_num_sms_preprocessing": num_sms_preprocessing,
            "hybridep_jit_dispatched_shape": list(dispatched.shape),
            "hybridep_jit_dispatched_probs_shape": (
                list(dispatched_probs.shape) if dispatched_probs is not None else None
            ),
            "hybridep_jit_combined_shape": list(combined.shape),
            "hybridep_jit_max_abs_diff": max_abs_diff,
            "hybridep_jit_tolerance": tolerance,
            "hybridep_jit_tokens_per_expert": [int(x) for x in tokens_per_expert.cpu()],
        }
    )
    fused_a2a.reset_hybrid_ep_buffer()


def main() -> int:
    result: dict[str, object] = {
        "host": socket.gethostname(),
        "rank": _as_int_env("RANK"),
        "local_rank": _as_int_env("LOCAL_RANK"),
        "world_size": _as_int_env("WORLD_SIZE", 1),
    }
    try:
        import deep_ep
        from deep_ep import Buffer

        _record_hybridep_availability(result)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        local_rank = int(result["local_rank"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        result.update(
            {
                "rank": rank,
                "world_size": world_size,
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "deep_ep_file": getattr(deep_ep, "__file__", ""),
                "deep_ep_sm90_compiled": bool(Buffer.is_sm90_compiled()),
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(device)),
            }
        )

        num_tokens = int(os.environ.get("DEEPEP_PREFLIGHT_TOKENS", "128"))
        hidden = int(os.environ.get("DEEPEP_PREFLIGHT_HIDDEN", "768"))
        num_experts = world_size
        num_nvl_bytes = int(os.environ.get("DEEPEP_PREFLIGHT_NVL_BYTES", str(256 * 1024 * 1024)))

        buffer = Buffer(dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
        x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
        topk_idx = (torch.arange(num_tokens, device=device, dtype=torch.long) % num_experts).view(num_tokens, 1)
        topk_weights = torch.ones(num_tokens, 1, device=device, dtype=torch.float32)

        layout = buffer.get_dispatch_layout(topk_idx, num_experts=num_experts)
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _event = layout
        recv_x, recv_topk_idx, recv_topk_weights, counts, handle, _event = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        torch.cuda.synchronize(device)
        combined_x, combined_weights, _event = buffer.combine(
            recv_x,
            handle,
            topk_weights=recv_topk_weights,
        )
        torch.cuda.synchronize(device)

        result.update(
            {
                "status": "ok",
                "num_tokens": num_tokens,
                "hidden": hidden,
                "num_experts": num_experts,
                "num_nvl_bytes": num_nvl_bytes,
                "recv_tokens": int(recv_x.shape[0]),
                "combined_shape": list(combined_x.shape),
                "combined_weights_shape": list(combined_weights.shape) if combined_weights is not None else None,
                "local_expert_counts": [int(x) for x in counts],
            }
        )
        _run_hybridep_jit_preflight(
            result=result,
            device=device,
            num_tokens=num_tokens,
            hidden=hidden,
        )
        print(json.dumps(result, sort_keys=True))
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
