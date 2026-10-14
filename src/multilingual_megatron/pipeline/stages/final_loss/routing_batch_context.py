"""Explicit diagnostic metadata for the synchronous, load-once forward path.

The iterator proxy observes the batch before Bridge selects its model inputs.
Nothing here is passed to the model or used to compute loss or routing.
"""

from __future__ import annotations

from contextvars import ContextVar
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RoutingBatch:
    input_valid_mask: torch.Tensor  # CPU [batch, sequence]; True means real input
    sample_ids: tuple[int, ...]
    observed_layers: set[int] = field(default_factory=set)
    _device_mask: torch.Tensor | None = None

    def flat_mask(self, logits: torch.Tensor) -> torch.Tensor:
        batch, sequence = self.input_valid_mask.shape
        # Router logits are [sequence, batch, experts], then flattened by MCore.
        if tuple(logits.shape[:-1]) != (sequence, batch):
            raise ValueError(f"unsupported router layout {tuple(logits.shape)} for B={batch}, S={sequence}")
        if self._device_mask is None:
            self._device_mask = self.input_valid_mask.to(logits.device, non_blocking=True).T.contiguous().view(-1)
        return self._device_mask


CURRENT_BATCH: ContextVar[RoutingBatch | None] = ContextVar("routing_validation_batch", default=None)


def validate_routing_topology(model_config: Any) -> None:
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        if int(getattr(model_config, key, 1) or 1) != 1:
            raise ValueError(f"routing v2 does not support {key} != 1")
    if getattr(model_config, "sequence_parallel", False):
        raise ValueError("routing v2 does not support sequence parallelism")
    scope = getattr(model_config, "cuda_graph_scope", None)
    if scope not in (None, [], (), "attn", ["attn"], ("attn",)):
        raise ValueError(f"routing v2 requires no graphs or attention-only graphs, got {scope!r}")


def expected_router_layers(model_config: Any) -> frozenset[int]:
    """Return router IDs for the Stage 1 all-MoE decoder."""
    layers = int(model_config.num_layers)
    pattern = getattr(model_config, "hybrid_override_pattern", None) or getattr(
        model_config, "hybrid_layer_pattern", None
    )
    if pattern is not None:
        raise ValueError("Stage 1 routing diagnostics do not support hybrid decoders")
    frequency = getattr(model_config, "moe_layer_freq", 1)
    if isinstance(frequency, int) and not isinstance(frequency, bool) and frequency == 1:
        return frozenset(range(layers))
    if isinstance(frequency, (list, tuple)) and len(frequency) == layers and all(
        value == 1 for value in frequency
    ):
        return frozenset(range(layers))
    raise ValueError("Stage 1 routing diagnostics require an expert layer in every decoder block")


def wrap_routing_forward_step(forward_step: Any, *, expected_layers: int | Collection[int]) -> Any:
    """Wrap a state-bound Bridge callable; reject deferred execution and reuse."""
    expected = set(range(expected_layers)) if isinstance(expected_layers, int) else set(expected_layers)

    def wrapped(data_iterator, model, return_schedule_plan=False):
        if return_schedule_plan:
            raise ValueError("routing v2 requires synchronous forward execution")
        batches: list[RoutingBatch] = []

        class IteratorProxy:
            def __iter__(self):
                return self

            def __next__(self):
                if batches:
                    raise ValueError("multiple batches consumed by a single diagnostic forward")
                batch = next(data_iterator)
                mask = batch["input_valid_mask"]
                ids = tuple(int(x) for x in batch["sample_id"].tolist())
                if mask.dtype != torch.bool or mask.ndim != 2 or len(ids) != mask.shape[0]:
                    raise ValueError("invalid input-validity metadata")
                if len(set(ids)) != len(ids):
                    raise ValueError("duplicate sample IDs within a batch")
                context = RoutingBatch(mask, ids)
                batches.append(context)
                CURRENT_BATCH.set(context)
                return batch

        token = CURRENT_BATCH.set(None)
        try:
            result = forward_step(IteratorProxy(), model, return_schedule_plan=False)
            if len(batches) != 1 or batches[0].observed_layers != expected:
                raise ValueError("missing or unexpected router layers in diagnostic forward")
            return result
        finally:
            CURRENT_BATCH.reset(token)

    return wrapped
