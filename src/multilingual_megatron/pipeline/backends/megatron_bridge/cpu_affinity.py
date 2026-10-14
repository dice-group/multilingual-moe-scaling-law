"""Optional local-rank CPU affinity helpers for Bridge training jobs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


def _cpu_ranges(cpus: Iterable[int]) -> str:
    ordered = sorted(set(int(cpu) for cpu in cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _split_by_count(cpus: list[int], groups: int) -> list[list[int]]:
    groups = max(1, int(groups))
    if groups == 1 or len(cpus) <= 1:
        return [list(cpus)]
    base, remainder = divmod(len(cpus), groups)
    chunks: list[list[int]] = []
    offset = 0
    for group_idx in range(groups):
        size = base + (1 if group_idx < remainder else 0)
        if size <= 0:
            chunks.append([])
            continue
        chunks.append(cpus[offset : offset + size])
        offset += size
    return [chunk for chunk in chunks if chunk]


def select_local_rank_cpus(
    *,
    allowed_cpus: Iterable[int],
    local_rank: int,
    local_world_size: int,
    mode: str,
) -> list[int]:
    """Select a deterministic CPU subset for a local torchrun rank.

    ``socket`` maps 4-GPU H100 jobs as ranks 0/1 to the first half of the
    current CPU set and ranks 2/3 to the second half. For 2-GPU jobs using
    visible devices 0/1, both ranks stay on the first half because those GPUs
    are colocated in the observed H100 topology. ``rank`` splits the available
    CPU set into one chunk per local rank.
    """

    cpus = sorted(set(int(cpu) for cpu in allowed_cpus))
    if not cpus:
        return []
    local_world_size = max(1, int(local_world_size))
    local_rank = max(0, min(int(local_rank), local_world_size - 1))
    normalized = str(mode).strip().lower()
    if normalized in {"socket", "numa", "numa_socket"}:
        halves = _split_by_count(cpus, 2)
        if len(halves) == 1:
            return halves[0]
        socket_idx = 0 if local_world_size <= 2 else min(1, (local_rank * 2) // local_world_size)
        return halves[socket_idx]
    if normalized in {"rank", "local_rank", "chunk", "rank_chunk"}:
        chunks = _split_by_count(cpus, local_world_size)
        return chunks[min(local_rank, len(chunks) - 1)]
    raise ValueError(f"Unsupported PROOF_BIND_LOCAL_RANK_CPUS mode: {mode!r}")


def maybe_bind_local_rank_cpus() -> None:
    """Optionally bind this torchrun process to a local-rank CPU subset."""

    mode = str(os.environ.get("PROOF_BIND_LOCAL_RANK_CPUS", "")).strip()
    if not mode or mode.lower() in {"0", "false", "none", "off", "no"}:
        return
    strict = str(os.environ.get("PROOF_BIND_LOCAL_RANK_CPUS_STRICT", "")).strip() == "1"
    try:
        local_rank = _int_env("LOCAL_RANK", _int_env("SLURM_LOCALID", 0))
        local_world_size = _int_env("LOCAL_WORLD_SIZE", _int_env("SLURM_NTASKS_PER_NODE", 1))
        before = sorted(os.sched_getaffinity(0))
        selected = select_local_rank_cpus(
            allowed_cpus=before,
            local_rank=local_rank,
            local_world_size=local_world_size,
            mode=mode,
        )
        if selected:
            os.sched_setaffinity(0, set(selected))
        after = sorted(os.sched_getaffinity(0))
        payload = {
            "mode": mode,
            "local_rank": local_rank,
            "local_world_size": local_world_size,
            "before": _cpu_ranges(before),
            "selected": _cpu_ranges(selected),
            "after": _cpu_ranges(after),
        }
        print(f"[BRIDGE_CPU_AFFINITY] {json.dumps(payload, sort_keys=True)}", flush=True)
    except Exception as exc:  # pragma: no cover - platform/runtime dependent.
        payload = {"mode": mode, "error": repr(exc)}
        print(f"[BRIDGE_CPU_AFFINITY] {json.dumps(payload, sort_keys=True)}", flush=True)
        if strict:
            raise
