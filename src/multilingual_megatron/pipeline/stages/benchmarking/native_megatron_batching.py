"""Opt-in, measured batch selection for native likelihood scoring.

An OOM terminates the torchrun attempt. In particular, never retry a forward
on just one rank while its peers may still be inside a model collective.
"""

import math
import time
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class AutoBatchPolicy:
    max_batch_size: int = 512
    memory_fraction: float = 0.8
    warmup_batches: int = 1
    timed_batches: int = 3
    max_calibration_seconds: float = 300.0
    minimum_speedup: float = 1.02
    score_abs_tolerance: float = 0.01

    @classmethod
    def from_mapping(cls, value: Any) -> "AutoBatchPolicy | None":
        if value is None:
            return None
        if not isinstance(value, dict) or value.get("mode") != "auto":
            raise ValueError("evaluation.batching must be an object with mode=auto")
        fields = set(cls.__dataclass_fields__)
        if set(value) - fields - {"mode"}:
            raise ValueError("unknown evaluation.batching fields")
        policy = cls(**{key: val for key, val in value.items() if key != "mode"})
        for field in ("max_batch_size", "warmup_batches", "timed_batches"):
            val = getattr(policy, field)
            if not isinstance(val, int) or isinstance(val, bool) or val < 1:
                raise ValueError(f"batching.{field} must be a positive integer")
        if policy.timed_batches < 3:
            raise ValueError("batching requires at least three timed batches")
        for field in ("memory_fraction", "max_calibration_seconds", "minimum_speedup", "score_abs_tolerance"):
            val = getattr(policy, field)
            if isinstance(val, bool) or not isinstance(val, (float, int)) or not math.isfinite(val):
                raise ValueError(f"batching.{field} must be finite and numeric")
        if not 0 < policy.memory_fraction < 1:
            raise ValueError("batching.memory_fraction must be between zero and one")
        if policy.max_calibration_seconds <= 0 or policy.minimum_speedup < 1:
            raise ValueError("invalid batching calibration duration or minimum speedup")
        if not 0 <= policy.score_abs_tolerance <= 0.01:
            raise ValueError("batching score tolerance must be between zero and 0.01")
        return policy


def length_buckets(lengths: Iterable[int]) -> list[tuple[int, int, int]]:
    """Return (upper length, start, stop) for sorted, nonempty requests."""
    buckets: list[tuple[int, int, int]] = []
    previous = 0
    for index, length in enumerate(lengths):
        if length < 1 or length < previous:
            raise ValueError("batching requires positive, sorted sequence lengths")
        previous = length
        upper = max(64, 1 << (length - 1).bit_length())
        if buckets and buckets[-1][0] == upper:
            buckets[-1] = (upper, buckets[-1][1], index + 1)
        else:
            buckets.append((upper, index, index + 1))
    return buckets


def select_batch_size(
    policy: AutoBatchPolicy,
    request_count: int,
    probe: Callable[[int], Mapping[str, Any]],
    record: Callable[[Mapping[str, Any]], None],
    *, initial_batch_size: int = 1,
) -> int:
    """Grow cautiously; select throughput, not the largest fitting batch."""
    maximum = min(policy.max_batch_size, request_count)
    if maximum < 1:
        raise ValueError("cannot calibrate an empty bucket")
    if initial_batch_size < 1:
        raise ValueError("initial batch size must be positive")
    candidate, selected, best_rate = min(initial_batch_size, maximum), 0, 0.0
    while candidate <= maximum:
        record({"batch_size": candidate, "status": "started"})
        measured = dict(probe(candidate))
        for field in ("seconds", "peak_memory_fraction", "projected_double_memory_fraction", "score_abs_difference"):
            val = measured.get(field)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise RuntimeError(f"invalid batch probe measurement: {field}")
        if measured["seconds"] <= 0:
            raise RuntimeError("batch probe duration must be positive")
        memory_ok = measured["peak_memory_fraction"] <= policy.memory_fraction
        parity_ok = measured["score_abs_difference"] <= policy.score_abs_tolerance
        rate = candidate / measured["seconds"]
        accepted = memory_ok and parity_ok and (not selected or rate >= best_rate * policy.minimum_speedup)
        record({"batch_size": candidate, "status": "measured", **measured,
                "requests_per_second": rate, "memory_ok": memory_ok,
                "parity_ok": parity_ok, "selected": accepted})
        if not parity_ok:
            if selected:
                # The rejected probe is never used for evaluation. Keep the
                # already measured smaller batch; do not relax score tolerance.
                break
            raise RuntimeError("automatic batching changed request scores beyond tolerance")
        if accepted:
            selected, best_rate = candidate, rate
        if not memory_ok or measured["projected_double_memory_fraction"] > policy.memory_fraction:
            break
        candidate *= 2
    if not selected:
        raise RuntimeError("no batch fits the requested memory headroom")
    return selected


def score_batch_ranges(
    count: int, fixed_batch_size: int, buckets: Sequence[Mapping[str, int]] | None = None,
) -> Iterator[tuple[int, int]]:
    """Preserve exact request coverage, including short tails of each bucket."""
    if fixed_batch_size < 1 or count < 0:
        raise ValueError("invalid scoring batch size or request count")
    if buckets is None:
        buckets = [{"start": 0, "stop": count, "batch_size": fixed_batch_size}]
    cursor = 0
    # Validate the entire schedule before yielding a forward to any rank.
    for bucket in buckets:
        start, stop, batch = bucket["start"], bucket["stop"], bucket["batch_size"]
        if start != cursor or not start <= stop <= count or batch < 1:
            raise ValueError("batch schedule has a gap, overlap, or invalid bound")
        cursor = stop
    if cursor != count:
        raise ValueError("batch schedule does not cover every request")
    for bucket in buckets:
        for start in range(bucket["start"], bucket["stop"], bucket["batch_size"]):
            yield start, min(start + bucket["batch_size"], bucket["stop"])


def calibrate_native_batches(
    model: Any, sequences: Sequence[Any], spec: Any, *, total_started: float,
) -> dict[str, Any]:
    """Calibrate before routing hooks are installed; reduce decisions across ranks."""
    import torch
    import torch.distributed as dist

    from .native_megatron_scoring import _atomic_json, _score_cuda_batch, build_score_batch

    policy = spec.batching
    started = time.perf_counter()
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    device = torch.cuda.current_device()
    receipt: dict[str, Any] = {
        "mode": "auto", "status": "calibrating", "policy": asdict(policy),
        "buckets": [], "trials": [], "peak_allocated_bytes": 0,
        "comparison": "same leading requests scored at configured microbatch versus candidate batch",
        "reference_micro_batch_size": spec.micro_batch_size,
        "oom_policy": "abort whole torchrun attempt; no rank-local retry",
    }

    def maximum(values: Sequence[float]) -> list[float]:
        tensor = torch.tensor(values, device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.cpu().tolist()

    def check_budget() -> None:
        elapsed, total = maximum([time.perf_counter() - started, time.perf_counter() - total_started])
        if elapsed > policy.max_calibration_seconds or total * world / 3600 > spec.max_gpu_hours:
            raise RuntimeError("automatic batching calibration time/GPU-hour cap exceeded")

    receipt["peak_allocated_bytes"] = int(maximum([torch.cuda.max_memory_allocated()])[0])

    def save() -> None:
        if rank == 0:
            _atomic_json(spec.output_dir / "batching_calibration.json", receipt)

    def score(chunk: Sequence[Any]) -> tuple[list[float], float]:
        check_budget()
        batch = build_score_batch(chunk, pad_token_id=spec.pad_token_id,
                                  minimum_sequence_width=spec.minimum_sequence_width,
                                  sequence_width_multiple=getattr(spec, "sequence_width_multiple", 1))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        losses = _score_cuda_batch(model, batch)
        torch.cuda.synchronize()
        duration = time.perf_counter() - tick
        mask = batch["loss_mask"].to(device=losses.device, dtype=torch.float64)
        values = -(losses.double() * mask).sum(dim=1)
        invalid = maximum([float(not bool(torch.isfinite(values).all()))])[0]
        if invalid:
            raise RuntimeError("non-finite scores during batch calibration")
        result = values.cpu().tolist()
        del losses, mask, values, batch
        check_budget()
        return result, maximum([duration])[0]

    try:
        for upper, start, stop in length_buckets(len(item.input_ids) for item in sequences):
            # Probe the longest real rows; their padding width bounds the bucket.
            pool = list(reversed(sequences[max(start, stop - policy.max_batch_size):stop]))
            baseline_size = min(spec.micro_batch_size, len(pool))
            references = []
            reference_count = min(spec.parity_requests, len(pool))
            for offset in range(0, reference_count, baseline_size):
                values, _duration = score(pool[offset:offset + baseline_size])
                references.extend(values[:reference_count - offset])

            def record(trial: Mapping[str, Any]) -> None:
                receipt["trials"].append({"upper_length": upper, **trial})
                save()

            def probe(batch_size: int) -> Mapping[str, Any]:
                chunk = pool[:batch_size]
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                free, total = torch.cuda.mem_get_info()
                external = max(0, total - free - torch.cuda.memory_reserved())
                persistent = torch.cuda.memory_allocated() + external
                durations, difference = [], 0.0
                for repeat in range(policy.warmup_batches + policy.timed_batches):
                    values, duration = score(chunk)
                    difference = max(difference, max(
                        abs(value - reference)
                        for value, reference in zip(values, references)
                    ))
                    if repeat >= policy.warmup_batches:
                        durations.append(duration)
                # Small candidates must check the same reference rows as large
                # candidates, not just an easier prefix of the parity sample.
                for offset in range(batch_size, len(references), batch_size):
                    values, _duration = score(pool[offset:min(offset + batch_size, len(references))])
                    difference = max(difference, max(
                        abs(value - reference)
                        for value, reference in zip(values, references[offset:])
                    ))
                peak_allocated = torch.cuda.max_memory_allocated()
                peak = max(torch.cuda.max_memory_reserved(), peak_allocated) + external
                # A guarded linear projection only decides whether to try the
                # next batch. Real peak telemetry decides candidate acceptance.
                projected = persistent + 2.3 * max(0, peak - persistent)
                peak_fraction, projected_fraction, difference, all_rank_peak = maximum([
                    peak / total, projected / total, difference, peak_allocated,
                ])
                receipt["peak_allocated_bytes"] = max(receipt["peak_allocated_bytes"], int(all_rank_peak))
                return {"seconds": median(durations), "timed_seconds": durations,
                        "peak_memory_fraction": peak_fraction,
                        "projected_double_memory_fraction": projected_fraction,
                        "score_abs_difference": difference,
                        "parity_requests": len(references)}

            batch_size = select_batch_size(
                policy, len(pool), probe, record, initial_batch_size=baseline_size
            )
            receipt["buckets"].append({"upper_length": upper, "start": start, "stop": stop,
                                       "batch_size": batch_size,
                                       "probe_request_ids": [row.request_id for row in pool[:batch_size]]})
            if rank == 0:
                print(f"[native-benchmark] auto batch: length<={upper}, batch={batch_size}", flush=True)
        receipt["status"] = "calibrated"
        receipt["calibration_seconds"] = maximum([time.perf_counter() - started])[0]
        save()
        # Preserve calibration's high-water mark in its receipt; scoring gets
        # independent peak telemetry and the final result reports their maximum.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return receipt
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
