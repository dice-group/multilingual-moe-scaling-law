"""Disjoint, cost-balanced CPU shards; ordering uses runtime, never heldout loss."""
from collections import defaultdict
import heapq


def allocated_core_hours(allocation):
    return allocation["cores"]*allocation["minutes"]*allocation.get("jobs", 1)/60


def concurrent_cores(allocation):
    return allocation["cores"]*allocation.get("parallel_jobs", allocation.get("jobs", 1))


def partition(task_count, panel, inherited, jobs):
    if jobs < 1 or task_count < 1 or task_count % len(panel):
        raise ValueError("invalid shard dimensions")
    completed = {row["task"] for row in inherited}
    if len(completed) != len(inherited) or any(t < 0 or t >= task_count for t in completed):
        raise ValueError("duplicate or out-of-range inherited task")
    times = defaultdict(list)
    for row in inherited:
        spec = row["spec"]
        times[(spec["form"], spec["parameter_count"], spec["signed"])].append(row["runtime_seconds"])
    means = {key: sum(values)/len(values) for key, values in times.items()}
    costs = {}
    for task in range(task_count):
        if task in completed:
            continue
        spec = panel[task % len(panel)]
        key = (spec["form"], spec["parameter_count"], spec["signed"])
        costs[task] = max(1., means.get(key, 60.))
    heap = [(0., i) for i in range(jobs)]
    heapq.heapify(heap)
    shards = [[] for _ in range(jobs)]
    loads = [0.]*jobs
    for task in sorted(costs, key=lambda t: (-costs[t], t)):
        load, index = heapq.heappop(heap)
        shards[index].append(task)
        loads[index] = load+costs[task]
        heapq.heappush(heap, (loads[index], index))
    flat = [task for shard in shards for task in shard]
    if len(set(flat)) != len(flat) or set(flat) | completed != set(range(task_count)) or set(flat) & completed:
        raise ValueError("shards duplicate, omit, or recompute a task")
    return dict(tasks=shards, estimated_cpu_seconds=loads,
                method="deterministic longest-estimated-runtime first; prior per-form/count/sign means; no heldout scores")


def array_index(allocation, environ):
    jobs = allocation.get("jobs", 1)
    if jobs == 1:
        return None
    expected = {"SLURM_ARRAY_TASK_COUNT": str(jobs), "SLURM_ARRAY_TASK_MIN": "0",
                "SLURM_ARRAY_TASK_MAX": str(jobs-1)}
    if not environ.get("SLURM_ARRAY_JOB_ID") or any(environ.get(k) != v for k, v in expected.items()):
        raise ValueError("requires the exact frozen Slurm array dimensions")
    index = int(environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < jobs:
        raise ValueError("array index out of range")
    return index
