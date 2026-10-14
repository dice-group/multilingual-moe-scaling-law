"""Padding-aware, same-forward routing evidence for exact-language validation.

Only the explicitly wrapped synchronous evaluator activates the hook. Counts
use integer arithmetic and rank reconciliation uses disjoint sample ownership,
never similarity of expert histograms. Expert IDs are local to each layer.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from .routing_batch_context import CURRENT_BATCH

DIAGNOSTIC_SCHEMA = "multilingual_moe_routing_diagnostics/v2"
BLOCK_SEQUENCES = 128


@dataclass
class _LayerRecord:
    counts: torch.Tensor  # [valid, padding, all] x expert
    positions: torch.Tensor  # valid, padding, all
    score_sums: torch.Tensor
    z_loss_sum: torch.Tensor
    topk: int
    capacity_factor: float | None
    pad_to_capacity: bool
    aux_loss_coefficient: float
    z_loss_coefficient: float
    sample_ids: set[int] = field(default_factory=set)
    blocks: dict[int, torch.Tensor] = field(default_factory=dict)
    trace: dict[int, str] = field(default_factory=dict)


_RECORDS: dict[int, _LayerRecord] = {}
_ERRORS: list[str] = []
_INSTALLED = False


def _scores(logits: torch.Tensor, score_function: str) -> torch.Tensor:
    values = logits.float()
    if score_function == "softmax":
        return torch.softmax(values, dim=-1)
    if score_function == "sigmoid":
        values = torch.sigmoid(values)
    elif score_function == "sqrtsoftplus":
        values = torch.nn.functional.softplus(values).sqrt()
    else:
        raise ValueError(f"unsupported router score function: {score_function}")
    return values / (values.sum(dim=-1, keepdim=True) + 1.0e-20)


def _record_router_call(router: Any, logits: torch.Tensor, routing_map: torch.Tensor, padding_mask: Any) -> None:
    batch = CURRENT_BATCH.get()
    if batch is None:
        raise ValueError("missing explicit routing batch context")
    layer_index = int(router.layer_number) - 1
    if layer_index < 0 or layer_index in batch.observed_layers:
        raise ValueError(f"duplicate or invalid router layer {layer_index}")
    batch.observed_layers.add(layer_index)
    experts = int(router.config.num_moe_experts)
    topk = int(router.topk)
    valid = batch.flat_mask(logits)
    flat_logits = logits.reshape(-1, experts)
    final_map = routing_map.reshape(-1, experts).bool()
    if flat_logits.shape != final_map.shape or len(valid) != len(final_map):
        raise ValueError("router/mask shapes differ")
    # Deliberately do not interpret the model's optional padding_mask. Its
    # convention is framework-specific; the dataset owns diagnostic validity.
    device = logits.device
    record = _RECORDS.get(layer_index)
    if record is None:
        record = _LayerRecord(
            counts=torch.zeros((3, experts), dtype=torch.int64, device=device),
            positions=torch.zeros(3, dtype=torch.int64, device=device),
            score_sums=torch.zeros(experts, dtype=torch.float64, device=device),
            z_loss_sum=torch.zeros((), dtype=torch.float64, device=device),
            topk=topk,
            capacity_factor=getattr(router.config, "moe_expert_capacity_factor", None),
            pad_to_capacity=bool(getattr(router.config, "moe_pad_expert_input_to_capacity", False)),
            aux_loss_coefficient=float(getattr(router.config, "moe_aux_loss_coeff", 0.0) or 0),
            z_loss_coefficient=float(getattr(router.config, "moe_z_loss_coeff", 0.0) or 0),
        )
        _RECORDS[layer_index] = record
    if record.topk != topk or record.counts.shape != (3, experts):
        raise ValueError("router shape changed within a language")
    if record.sample_ids.intersection(batch.sample_ids):
        raise ValueError("sample observed more than once in a layer")
    record.sample_ids.update(batch.sample_ids)
    valid_map = final_map & valid[:, None]
    counts = valid_map.sum(dim=0, dtype=torch.int64)
    all_counts = final_map.sum(dim=0, dtype=torch.int64)
    n = valid.sum(dtype=torch.int64)
    record.counts.add_(torch.stack((counts, all_counts - counts, all_counts)))
    record.positions.add_(torch.stack((n, valid.numel() - n, torch.full_like(n, valid.numel()))))
    # Additive sufficient statistics define one corpus-level auxiliary metric,
    # independent of how the corpus was divided into microbatches.
    scores = _scores(flat_logits, str(getattr(router, "score_function", "softmax")))
    record.score_sums.add_((scores * valid[:, None]).sum(dim=0, dtype=torch.float64))
    record.z_loss_sum.add_((torch.logsumexp(flat_logits.float(), dim=-1).square() * valid).sum(dtype=torch.float64))
    # Stable sequence blocks remain aligned when batch size or DP size changes.
    per_sample = valid_map.view(logits.shape[0], logits.shape[1], experts).sum(0, dtype=torch.int64)
    groups: dict[int, list[int]] = {}
    for column, sample_id in enumerate(batch.sample_ids):
        groups.setdefault(sample_id // BLOCK_SEQUENCES, []).append(column)
        if sample_id < 2:  # bounded exact same-forward trace for pilot comparisons
            sample_map = final_map.view(logits.shape[0], logits.shape[1], experts)[:, column]
            sample_valid = valid.view(logits.shape[:2])[:, column]
            record.trace[sample_id] = hashlib.sha256(
                sample_map[sample_valid].to(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()
    for block, columns in groups.items():
        value = per_sample[columns].sum(0)
        if block not in record.blocks:
            record.blocks[block] = torch.zeros_like(value)
        record.blocks[block].add_(value)


def install_moe_validation_diagnostics() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    from megatron.core.transformer.moe import router as router_module
    original = router_module.TopKRouter.routing
    if getattr(original, "_validation_diagnostics_v2", False):
        _INSTALLED = True
        return True

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # No effects on ordinary training or unrelated forward passes.
        if CURRENT_BATCH.get() is not None:
            try:
                logits = args[0] if args else kwargs["logits"]
                padding = kwargs.get("padding_mask", args[1] if len(args) > 1 else None)
                _record_router_call(self, logits, result[1], padding)
            except Exception as exc:
                _ERRORS.append(repr(exc))
        return result

    wrapped._validation_diagnostics_v2 = True
    router_module.TopKRouter.routing = wrapped
    _INSTALLED = True
    return True


def reset_moe_validation_diagnostics() -> None:
    _RECORDS.clear()
    _ERRORS.clear()


def prepare_moe_validation_diagnostics(model_config: Any, *, required: bool) -> frozenset[int] | None:
    """Install instrumentation only when requested; loss scoring is independent."""
    if not required or int(getattr(model_config, "num_moe_experts", 0) or 0) <= 1:
        return None
    from .routing_batch_context import expected_router_layers, validate_routing_topology

    validate_routing_topology(model_config)
    layers = expected_router_layers(model_config)
    if not layers or not install_moe_validation_diagnostics():
        raise RuntimeError("failed to install required MoE routing diagnostics")
    reset_moe_validation_diagnostics()
    return layers


def _imbalance(counts: torch.Tensor) -> dict[str, float | None]:
    values = counts.double()
    mean = float(values.mean())
    if mean <= 0:
        return dict.fromkeys(("coefficient_of_variation", "max_to_mean", "min_to_mean", "normalized_entropy"))
    p = values / values.sum()
    positive = p[p > 0]
    return {
        "coefficient_of_variation": float(values.std(unbiased=False) / mean),
        "max_to_mean": float(values.max() / mean),
        "min_to_mean": float(values.min() / mean),
        "normalized_entropy": float(-(positive * positive.log()).sum() / math.log(len(values))),
    }


def _routing_snapshot() -> dict[str, Any]:
    layers = []
    for index, record in sorted(_RECORDS.items()):
        counts = record.counts.cpu().tolist()
        positions = record.positions.cpu().tolist()
        layers.append({
            "layer_index": index, "topk": record.topk,
            "capacity_factor": record.capacity_factor, "pad_to_capacity": record.pad_to_capacity,
            "expert_token_counts": counts[0], "padding_expert_token_counts": counts[1],
            "all_expert_token_counts": counts[2], "forwarded_tokens": positions[0],
            "padding_positions": positions[1], "all_positions": positions[2],
            "expected_assignments": positions[0] * record.topk, "kept_assignments": sum(counts[0]),
            "score_sums": record.score_sums.cpu().tolist(), "z_loss_sum": float(record.z_loss_sum),
            "aux_loss_coefficient": record.aux_loss_coefficient, "z_loss_coefficient": record.z_loss_coefficient,
            "sample_ids": sorted(record.sample_ids),
            "blocks": {str(k): v.cpu().tolist() for k, v in sorted(record.blocks.items())},
            "trace": {str(k): v for k, v in record.trace.items()},
        })
    return {"layers": layers, "hook_errors": list(_ERRORS)}


def reconcile_snapshots(snapshots: list[dict], *, expected_sample_count: int | None = None,
                        ownership_mode: str = "disjoint") -> tuple[list[dict], list[str]]:
    """Sum disjoint rank-owned samples under the supported TP=PP=CP=1 topology."""
    errors: list[str] = []
    if not snapshots or not snapshots[0].get("layers"):
        return [], ["no router calls observed"]
    structure_keys = ("layer_index", "topk", "capacity_factor", "pad_to_capacity", "aux_loss_coefficient", "z_loss_coefficient")
    reference = snapshots[0]["layers"]
    structure = lambda rows: [(tuple(row[k] for k in structure_keys), len(row["expert_token_counts"])) for row in rows]
    for rank, snapshot in enumerate(snapshots):
        errors.extend(f"rank_{rank}:hook_error:{e}" for e in snapshot.get("hook_errors", []))
        rows = snapshot.get("layers", [])
        if structure(rows) != structure(reference):
            errors.append(f"rank_{rank}:routing_structure_differs_from_rank_0")
            continue
        for row in rows:
            ids = row["sample_ids"]
            if len(set(ids)) != len(ids) or ids != rows[0]["sample_ids"]:
                errors.append(f"rank_{rank}:inconsistent_sample_ownership")
            counts, pad, all_counts = (row[k] for k in ("expert_token_counts", "padding_expert_token_counts", "all_expert_token_counts"))
            if (any(x < 0 for x in counts + pad + all_counts)
                or [a + b for a, b in zip(counts, pad)] != all_counts
                or sum(counts) != row["kept_assignments"]
                or row["expected_assignments"] != row["forwarded_tokens"] * row["topk"]
                or row["kept_assignments"] > row["expected_assignments"]
                or row["forwarded_tokens"] + row["padding_positions"] != row["all_positions"]
                or [sum(v[i] for v in row["blocks"].values()) for i in range(len(counts))] != counts):
                errors.append(f"rank_{rank}:invalid_routing_count_relationship")
    if errors:
        return [], errors
    if ownership_mode == "replicated":
        # Native benchmark requests are explicitly broadcast to every rank.
        # Replication must be declared by the caller and then verified.
        keys = ("sample_ids", "expert_token_counts", "padding_expert_token_counts", "all_expert_token_counts", "blocks", "trace")
        for rank, snapshot in enumerate(snapshots[1:], 1):
            if any(any(a[key] != b[key] for key in keys) for a, b in zip(reference, snapshot["layers"], strict=True)):
                errors.append(f"rank_{rank}:declared_replicated_routing_differs")
        if errors:
            return [], errors
        snapshots = snapshots[:1]
    elif ownership_mode != "disjoint":
        raise ValueError(f"unknown sample ownership mode: {ownership_mode}")
    seen: set[int] = set()
    for rank, snapshot in enumerate(snapshots):
        ids = set(snapshot["layers"][0]["sample_ids"])
        if seen.intersection(ids):
            errors.append(f"rank_{rank}:overlapping_sample_ownership")
        seen.update(ids)
    if expected_sample_count is not None and seen != set(range(expected_sample_count)):
        errors.append("missing_or_unexpected_sample_ids")
    if errors:
        return [], errors
    result = []
    vectors = ("expert_token_counts", "padding_expert_token_counts", "all_expert_token_counts", "score_sums")
    scalars = ("forwarded_tokens", "padding_positions", "all_positions", "expected_assignments", "kept_assignments", "z_loss_sum")
    for i, template in enumerate(reference):
        rows = [s["layers"][i] for s in snapshots]
        row = {k: template[k] for k in structure_keys}
        row.update({k: [sum(r[k][j] for r in rows) for j in range(len(template[k]))] for k in vectors})
        row.update({k: sum(r[k] for r in rows) for k in scalars})
        blocks: dict[str, list[int]] = {}
        for r in rows:
            for block, counts in r["blocks"].items():
                blocks[block] = [a + b for a, b in zip(blocks.get(block, [0] * len(counts)), counts)]
        row["blocks"] = blocks
        row["trace"] = {k: v for r in rows for k, v in r["trace"].items()}
        result.append(row)
    return result, []


def collect_moe_validation_diagnostics(*, expected_num_layers=None, expected_num_experts=None,
                                       expected_sample_count=None, expected_valid_tokens=None,
                                       sequence_length=None, ownership_mode="disjoint",
                                       expected_layer_indices=None) -> dict[str, Any]:
    world = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
    local = _routing_snapshot()
    snapshots = [local]
    if world > 1:
        snapshots = [None] * world
        torch.distributed.all_gather_object(snapshots, local)
    rows, errors = reconcile_snapshots(snapshots, expected_sample_count=expected_sample_count, ownership_mode=ownership_mode)
    if expected_num_layers is not None and len(rows) != expected_num_layers:
        errors.append("missing_or_unexpected_layers")
    if expected_layer_indices is not None and {row["layer_index"] for row in rows} != set(expected_layer_indices):
        errors.append("missing_or_unexpected_layer_indices")
    for row in rows:
        counts = row["expert_token_counts"]
        n = row["forwarded_tokens"]
        if expected_num_experts is not None and len(counts) != expected_num_experts:
            errors.append("expert_count_mismatch")
        if expected_valid_tokens is not None and n != expected_valid_tokens:
            errors.append(f"layer_{row['layer_index']}:valid_token_count_mismatch:{n}!={expected_valid_tokens}")
        if sequence_length is not None and row["all_positions"] != expected_sample_count * sequence_length:
            errors.append("all_position_count_mismatch")
        if row["pad_to_capacity"]:
            errors.append("pad_to_capacity_not_supported")
        row["dropped_assignments"] = row["expected_assignments"] - row["kept_assignments"]
        row["dropped_assignment_rate"] = row["dropped_assignments"] / row["expected_assignments"] if n else None
        row["imbalance"] = _imbalance(torch.tensor(counts))
        row["router_z_loss_unweighted"] = row["z_loss_sum"] / n if n else None
        aux = len(counts) * sum(p * c for p, c in zip(row["score_sums"], counts)) / (n * n * row["topk"]) if n else None
        row["switch_aux_loss_unweighted"] = aux
        row["switch_aux_loss_weighted"] = aux * row["aux_loss_coefficient"] if aux is not None else None
        row["router_z_loss_weighted"] = row["router_z_loss_unweighted"] * row["z_loss_coefficient"] if n else None
    totals = {k: sum(r[k] for r in rows) for k in ("expected_assignments", "kept_assignments", "dropped_assignments")}
    totals["dropped_assignment_rate"] = totals["dropped_assignments"] / totals["expected_assignments"] if totals["expected_assignments"] else None
    result = {
        "schema": DIAGNOSTIC_SCHEMA, "applicable": True, "evidence_available": not errors,
        "evidence_errors": errors, "world_size": world, "layers": rows, "totals": totals,
        "rank_reconciliation": {"mode": ownership_mode + "_sample_ids", "world_size": world,
            "sample_counts_by_rank": [len(s["layers"][0]["sample_ids"]) if s["layers"] else 0 for s in snapshots]},
        "block_sequences": BLOCK_SEQUENCES,
        "validity_definition": "constructed input positions, including initial context and real special tokens; excludes synthetic padding",
        "aux_z_note": "Corpus-level diagnostic from additive valid-position statistics, not the training auxiliary loss.",
        "expert_identity_scope": "within one checkpoint and layer only",
    }
    reset_moe_validation_diagnostics()
    return result
