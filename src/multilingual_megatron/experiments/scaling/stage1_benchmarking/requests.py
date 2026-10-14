"""Frozen request manifests for pinned Stage-1 lm-eval tasks."""

import hashlib
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    BenchmarkProfile, REQUEST_MANIFEST_SCHEMA, TASK_FRAGMENT_SCHEMA, _atomic_json,
    _benchmark_dataset_pin, _load_json, _payload_sha256, _set_hf_offline_cache,
    _verify_payload_hash,
)
from .tasks import TaskDefinition, build_task_definitions

def _load_leaf_task(task_manager: Any, definition: TaskDefinition, seed: int) -> Any:
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import get_task_dict

    if isinstance(definition.task_input, Mapping):
        task = ConfigurableTask(config=dict(definition.task_input))
    else:
        task_dict = get_task_dict([definition.task_input], task_manager=task_manager)
        outputs = get_task_list(task_dict)
        if len(outputs) != 1:
            raise ValueError(
                f"expected one leaf for {definition.task_name}, got {len(outputs)}"
            )
        task = outputs[0].task
    task.set_config(key="num_fewshot", value=0)
    task.set_fewshot_seed(seed=seed)
    return task


def _stable_sample(population: int, count: int, *, seed: int, label: str) -> list[int]:
    if population < count:
        raise ValueError(f"{label}: population {population} is smaller than sample {count}")
    derived = int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16], 16)
    return sorted(random.Random(derived).sample(range(population), count))


def allocate_stratified_counts(
    capacities: Mapping[str, int], total: int
) -> dict[str, int]:
    """Allocate an equal-stratum sample, redistributing any shortfall."""

    if total <= 0 or not capacities:
        raise ValueError("stratified allocation requires positive total and capacities")
    names = sorted(capacities)
    if any(capacities[name] < 0 for name in names) or sum(capacities.values()) < total:
        raise ValueError("stratified capacities cannot cover requested total")
    base, remainder = divmod(total, len(names))
    allocation = {
        name: min(capacities[name], base + int(index < remainder))
        for index, name in enumerate(names)
    }
    remaining = total - sum(allocation.values())
    while remaining:
        progressed = False
        for name in names:
            if allocation[name] < capacities[name]:
                allocation[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("failed to redistribute stratified sample shortfall")
    return allocation


def _dataset_identity(task: Any) -> dict[str, Any]:
    docs = task.eval_docs
    cache_files = getattr(docs, "cache_files", []) or []
    config = task.config
    return {
        "dataset_path": config.dataset_path,
        "dataset_name": config.dataset_name,
        "dataset_kwargs": config.dataset_kwargs,
        "split": config.test_split or config.validation_split,
        "rows": len(docs),
        "fingerprint": getattr(docs, "_fingerprint", None),
        "cache_files": [str(item.get("filename")) for item in cache_files],
    }


def _validate_replay_task(
    definition: TaskDefinition,
    task: Any,
    frozen: Mapping[str, Any],
) -> None:
    """Fail closed if an lm-eval task no longer matches its frozen contract."""

    expected_definition = {
        "benchmark": definition.benchmark,
        "category": definition.category,
        "language": definition.language,
        "stratum": definition.stratum,
        "task": definition.task_name,
    }
    actual_definition = {field: frozen.get(field) for field in expected_definition}
    if actual_definition != expected_definition:
        raise ValueError(
            f"frozen task metadata drift for {definition.task_name}: "
            f"expected {expected_definition}, got {actual_definition}"
        )
    if task.config.task != definition.task_name:
        raise ValueError(
            f"replay task-name drift: expected {definition.task_name!r}, "
            f"got {task.config.task!r}"
        )
    if task.OUTPUT_TYPE != frozen.get("output_type"):
        raise ValueError(
            f"replay output-type drift for {definition.task_name}: "
            f"expected {frozen.get('output_type')!r}, got {task.OUTPUT_TYPE!r}"
        )
    actual_metrics = sorted(task._metric_fn_list)
    if actual_metrics != frozen.get("metrics"):
        raise ValueError(
            f"replay metric drift for {definition.task_name}: "
            f"expected {frozen.get('metrics')!r}, got {actual_metrics!r}"
        )
    expected_dataset = frozen.get("dataset")
    if not isinstance(expected_dataset, Mapping):
        raise ValueError(f"missing frozen dataset identity for {definition.task_name}")
    actual_dataset = _dataset_identity(task)
    identity_fields = (
        "dataset_path",
        "dataset_name",
        "dataset_kwargs",
        "split",
        "rows",
    )
    # Custom tasks use pinned revisions and exact replay requests instead.
    if not isinstance(definition.task_input, Mapping):
        identity_fields += ("fingerprint",)
    drift = {
        field: {"expected": expected_dataset.get(field), "actual": actual_dataset.get(field)}
        for field in identity_fields
        if expected_dataset.get(field) != actual_dataset.get(field)
    }
    if drift:
        raise ValueError(f"replay dataset drift for {definition.task_name}: {drift}")


def _request_id(task_name: str, doc_id: int, choice_index: int, args: Sequence[str]) -> str:
    return _payload_sha256(
        {
            "task": task_name,
            "doc_id": doc_id,
            "choice_index": choice_index,
            "args": list(args),
        }
    )


def _freeze_task(
    definition: TaskDefinition,
    task: Any,
    sample_ids: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if task.OUTPUT_TYPE != "multiple_choice":
        raise ValueError(
            f"{definition.task_name} has unsupported output type {task.OUTPUT_TYPE!r}; "
            "the pilot is multiple-choice only"
        )
    if task.config.task != definition.task_name:
        raise ValueError(
            f"task-name drift: expected {definition.task_name!r}, got {task.config.task!r}"
        )
    metric_names = sorted(task._metric_fn_list)
    if "exact_match" in metric_names:
        raise ValueError(
            f"{definition.task_name} requests exact_match, which requires a greedy-token flag"
        )
    task.build_all_requests(
        samples=list(sample_ids),
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
    )
    requests: list[dict[str, Any]] = []
    for instance in task.instances:
        if instance.request_type != "loglikelihood":
            raise ValueError(
                f"{definition.task_name} emitted unsupported {instance.request_type} request"
            )
        if not isinstance(instance.args, tuple) or len(instance.args) != 2:
            raise ValueError(f"invalid loglikelihood arguments in {definition.task_name}")
        context, continuation = instance.args
        if not isinstance(context, str) or not isinstance(continuation, str):
            raise ValueError(f"non-text request in {definition.task_name}")
        source_doc_id = int(sample_ids[int(instance.doc_id)])
        request_id = _request_id(
            definition.task_name,
            int(instance.doc_id),
            int(instance.idx),
            (context, continuation),
        )
        requests.append(
            {
                "request_id": request_id,
                "request_type": "loglikelihood",
                "benchmark": definition.benchmark,
                "category": definition.category,
                "language": definition.language,
                "stratum": definition.stratum,
                "task": definition.task_name,
                "doc_id": int(instance.doc_id),
                "source_doc_id": source_doc_id,
                "source_id": instance.doc.get("source_id")
                if isinstance(instance.doc, dict)
                else None,
                "choice_index": int(instance.idx),
                "context": context,
                "continuation": continuation,
            }
        )
    expected_docs = len(sample_ids)
    observed_docs = len({item["doc_id"] for item in requests})
    if observed_docs != expected_docs:
        raise RuntimeError(
            f"{definition.task_name}: expected {expected_docs} documents, got {observed_docs}"
        )
    metadata = {
        "benchmark": definition.benchmark,
        "category": definition.category,
        "language": definition.language,
        "stratum": definition.stratum,
        "task": definition.task_name,
        "output_type": task.OUTPUT_TYPE,
        "metrics": metric_names,
        "sample_ids": list(sample_ids),
        "documents": expected_docs,
        "requests": len(requests),
        "dataset": _dataset_identity(task),
    }
    return metadata, requests


def _task_fragment_path(fragment_dir: Path, task_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("._") or "task"
    suffix = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:12]
    return fragment_dir / f"{slug}.{suffix}.json"


def _load_task_fragment(
    path: Path,
    *,
    profile: BenchmarkProfile,
    definition: TaskDefinition,
    expected_sample_ids: Sequence[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not path.is_file():
        return None
    payload = _load_json(path, label="frozen task fragment")
    if payload.get("schema") != TASK_FRAGMENT_SCHEMA:
        raise ValueError(f"unsupported task-fragment schema: {path}")
    _verify_payload_hash(payload, field="fragment_sha256", label="task fragment")
    if payload.get("profile_sha256") != profile.source_sha256:
        raise ValueError(f"task fragment profile drift: {path}")
    if payload.get("harness_commit") != profile.payload["harness_commit"]:
        raise ValueError(f"task fragment harness drift: {path}")
    metadata = payload.get("task")
    requests = payload.get("requests")
    if not isinstance(metadata, dict) or not isinstance(requests, list) or not requests:
        raise ValueError(f"invalid frozen task fragment payload: {path}")
    expected_identity = {
        "benchmark": definition.benchmark,
        "category": definition.category,
        "language": definition.language,
        "stratum": definition.stratum,
        "task": definition.task_name,
    }
    if any(metadata.get(field) != value for field, value in expected_identity.items()):
        raise ValueError(f"task fragment identity drift: {path}")
    if metadata.get("output_type") != "multiple_choice":
        raise ValueError(f"task fragment is not multiple-choice: {path}")
    benchmark_entry = next(
        entry
        for entry in profile.benchmarks
        if str(entry["id"]) == definition.benchmark
    )
    dataset_pin = _benchmark_dataset_pin(benchmark_entry)
    if dataset_pin is not None:
        dataset_kwargs = metadata.get("dataset", {}).get("dataset_kwargs", {})
        if not isinstance(dataset_kwargs, dict) or dataset_kwargs.get(
            "revision"
        ) != dataset_pin["revision"]:
            return None
    if expected_sample_ids is not None and metadata.get("sample_ids") != list(
        expected_sample_ids
    ):
        raise ValueError(f"task fragment sample selection drift: {path}")
    if int(metadata.get("requests", -1)) != len(requests):
        raise ValueError(f"task fragment request count mismatch: {path}")
    if any(
        not isinstance(item, dict) or item.get("task") != definition.task_name
        for item in requests
    ):
        raise ValueError(f"task fragment request identity mismatch: {path}")
    if len({item.get("request_id") for item in requests}) != len(requests):
        raise ValueError(f"task fragment contains duplicate request IDs: {path}")
    return dict(metadata), [dict(item) for item in requests]


def _write_task_fragment(
    path: Path,
    *,
    profile: BenchmarkProfile,
    metadata: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "schema": TASK_FRAGMENT_SCHEMA,
        "profile_sha256": profile.source_sha256,
        "harness_commit": profile.payload["harness_commit"],
        "task": dict(metadata),
        "requests": [dict(item) for item in requests],
    }
    payload["fragment_sha256"] = _payload_sha256(payload)
    _atomic_json(path, payload)


def freeze_requests(profile: BenchmarkProfile, output_path: Path) -> Path:
    _set_hf_offline_cache(profile.hf_home)
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    definitions = build_task_definitions(profile, manager)
    output_path = output_path.resolve()
    fragment_dir = output_path.with_suffix(".parts")
    fragment_dir.mkdir(parents=True, exist_ok=True)
    benchmark_entries = {str(item["id"]): item for item in profile.benchmarks}
    sampling_modes = {
        benchmark: str(entry.get("sampling"))
        for benchmark, entry in benchmark_entries.items()
    }
    unsupported_sampling = sorted(
        benchmark
        for benchmark, mode in sampling_modes.items()
        if mode not in {"parallel_shared_ids", "equal_stratified_domains"}
    )
    if unsupported_sampling:
        raise ValueError(
            f"unsupported benchmark sampling modes: {unsupported_sampling}"
        )
    parallel_benchmarks = [
        benchmark
        for benchmark in benchmark_entries
        if sampling_modes[benchmark] == "parallel_shared_ids"
    ]
    stratified_benchmarks = [
        benchmark
        for benchmark in benchmark_entries
        if sampling_modes[benchmark] == "equal_stratified_domains"
    ]
    shared_populations = {
        benchmark: int(
            benchmark_entries[benchmark].get(
                "shared_population", profile.shared_population
            )
        )
        for benchmark in parallel_benchmarks
    }
    too_small = {
        benchmark: population
        for benchmark, population in shared_populations.items()
        if population < profile.examples_per_language
    }
    if too_small:
        raise ValueError(
            "benchmark shared populations do not cover examples_per_language: "
            f"{too_small}"
        )
    shared_ids = {
        benchmark: _stable_sample(
            shared_populations[benchmark],
            profile.examples_per_language,
            seed=profile.seed,
            label=benchmark,
        )
        for benchmark in parallel_benchmarks
    }

    task_metadata: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    completed = 0
    total_tasks = len(definitions)
    for benchmark in parallel_benchmarks:
        selected = sorted(
            (item for item in definitions if item.benchmark == benchmark),
            key=lambda item: item.task_name,
        )
        for definition in selected:
            fragment_path = _task_fragment_path(fragment_dir, definition.task_name)
            frozen = _load_task_fragment(
                fragment_path,
                profile=profile,
                definition=definition,
                expected_sample_ids=shared_ids[benchmark],
            )
            reused = frozen is not None
            if frozen is None:
                task = _load_leaf_task(manager, definition, profile.seed)
                shared_population = shared_populations[benchmark]
                if len(task.eval_docs) < shared_population:
                    raise ValueError(
                        f"{definition.task_name} has {len(task.eval_docs)} docs, below frozen "
                        f"shared population {shared_population}"
                    )
                metadata, rows = _freeze_task(definition, task, shared_ids[benchmark])
                _write_task_fragment(
                    fragment_path,
                    profile=profile,
                    metadata=metadata,
                    requests=rows,
                )
            else:
                metadata, rows = frozen
            task_metadata.append(metadata)
            request_rows.extend(rows)
            completed += 1
            action = "reuse" if reused else "freeze"
            print(f"[{action}] {completed}/{total_tasks} {definition.task_name}", flush=True)

    for benchmark in stratified_benchmarks:
        languages = sorted(
            {item.language for item in definitions if item.benchmark == benchmark}
        )
        for language in languages:
            language_defs = sorted(
                (
                    item
                    for item in definitions
                    if item.benchmark == benchmark and item.language == language
                ),
                key=lambda item: item.task_name,
            )
            loaded: list[tuple[TaskDefinition, Any | None, dict[str, Any] | None, list[dict[str, Any]] | None]] = []
            for definition in language_defs:
                fragment_path = _task_fragment_path(fragment_dir, definition.task_name)
                frozen = _load_task_fragment(
                    fragment_path,
                    profile=profile,
                    definition=definition,
                    expected_sample_ids=None,
                )
                if frozen is None:
                    loaded.append(
                        (
                            definition,
                            _load_leaf_task(manager, definition, profile.seed),
                            None,
                            None,
                        )
                    )
                else:
                    metadata, rows = frozen
                    loaded.append((definition, None, metadata, rows))
            capacities = {
                definition.task_name: (
                    len(task.eval_docs)
                    if task is not None
                    else int(metadata["dataset"]["rows"])
                )
                for definition, task, metadata, _rows in loaded
            }
            allocation = allocate_stratified_counts(
                capacities, profile.examples_per_language
            )
            for definition, task, metadata, rows in loaded:
                count = allocation[definition.task_name]
                sample_ids = _stable_sample(
                    capacities[definition.task_name],
                    count,
                    seed=profile.seed,
                    label=f"{benchmark}:{language}:{definition.stratum}",
                )
                reused = metadata is not None and rows is not None
                fragment_path = _task_fragment_path(fragment_dir, definition.task_name)
                if reused:
                    frozen = _load_task_fragment(
                        fragment_path,
                        profile=profile,
                        definition=definition,
                        expected_sample_ids=sample_ids,
                    )
                    if frozen is None:
                        raise RuntimeError(f"task fragment disappeared: {fragment_path}")
                    metadata, rows = frozen
                else:
                    if task is None:
                        raise RuntimeError(f"missing loaded task: {definition.task_name}")
                    metadata, rows = _freeze_task(definition, task, sample_ids)
                    _write_task_fragment(
                        fragment_path,
                        profile=profile,
                        metadata=metadata,
                        requests=rows,
                    )
                task_metadata.append(metadata)
                request_rows.extend(rows)
                completed += 1
                action = "reuse" if reused else "freeze"
                print(f"[{action}] {completed}/{total_tasks} {definition.task_name}", flush=True)

    if len({row["request_id"] for row in request_rows}) != len(request_rows):
        raise RuntimeError("frozen request IDs are not unique")
    language_counts: dict[str, dict[str, int]] = {}
    for benchmark in [item["id"] for item in profile.benchmarks]:
        by_language: dict[str, set[tuple[str, int]]] = {}
        for row in request_rows:
            if row["benchmark"] == benchmark:
                by_language.setdefault(row["language"], set()).add(
                    (row["task"], row["source_doc_id"])
                )
        counts = {language: len(ids) for language, ids in by_language.items()}
        if set(counts.values()) != {profile.examples_per_language}:
            raise RuntimeError(f"{benchmark} document coverage mismatch: {counts}")
        language_counts[benchmark] = counts

    manifest: dict[str, Any] = {
        "schema": REQUEST_MANIFEST_SCHEMA,
        "pilot_id": profile.benchmark_id,
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "harness": {
            "path": str(profile.harness_root),
            "commit": profile.payload["harness_commit"],
        },
        "sampling": {
            "seed": profile.seed,
            "examples_per_language": profile.examples_per_language,
            "shared_population": profile.shared_population,
            "default_shared_population": profile.shared_population,
            "shared_population_by_benchmark": shared_populations,
            "parallel_shared_ids": shared_ids,
            "stratified_policy": "equal allocation across sorted available strata with shortfall redistribution",
        },
        "dataset_pins": {
            benchmark: pin
            for benchmark, entry in benchmark_entries.items()
            if (pin := _benchmark_dataset_pin(entry)) is not None
        },
        "coverage": {
            "benchmarks": len(profile.benchmarks),
            "languages": {
                benchmark: len(counts) for benchmark, counts in language_counts.items()
            },
            "documents_per_language": language_counts,
            "leaf_tasks": len(task_metadata),
            "requests": len(request_rows),
        },
        "tasks": task_metadata,
        "requests": request_rows,
    }
    manifest["manifest_sha256"] = _payload_sha256(manifest)
    _atomic_json(output_path, manifest)
    return output_path
