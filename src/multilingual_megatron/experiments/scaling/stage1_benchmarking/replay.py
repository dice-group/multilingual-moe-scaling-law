"""Replay native scores through the pinned lm-evaluation-harness metrics."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    BenchmarkProfile,
    METRICS_SCHEMA,
    REQUEST_MANIFEST_SCHEMA,
    SCORE_RESULT_SCHEMA,
    _atomic_json,
    _load_json,
    _payload_sha256,
    _set_hf_offline_cache,
    _sha256_file,
    _verify_payload_hash,
)
from .requests import (
    _load_leaf_task,
    _request_id,
    _validate_replay_task,
)
from .tasks import build_task_definitions


def _score_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path, label="native benchmark scores")
    if payload.get("schema") != SCORE_RESULT_SCHEMA or payload.get("status") != "complete":
        raise ValueError(f"incomplete native score result: {path}")
    _verify_payload_hash(payload, field="result_sha256", label="native score result")
    if not payload.get("parity_gate", {}).get("passed"):
        raise ValueError(f"native score parity gate did not pass: {path}")
    if not payload.get("replica_gate", {}).get("passed"):
        raise ValueError(f"native score replica gate did not pass: {path}")
    return payload


def aggregate_scores(
    profile: BenchmarkProfile,
    request_manifest_path: Path,
    scores_path: Path,
    output_dir: Path,
) -> Path:
    _set_hf_offline_cache(profile.hf_home)
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable

    request_path = request_manifest_path.resolve()
    manifest = _load_json(request_path, label="frozen request manifest")
    if manifest.get("schema") != REQUEST_MANIFEST_SCHEMA:
        raise ValueError("invalid frozen request manifest schema")
    if manifest.get("pilot_id") != profile.request_manifest_pilot_id:
        raise ValueError(
            "request manifest pilot_id does not match the declared request contract"
        )
    _verify_payload_hash(manifest, field="manifest_sha256", label="request manifest")
    scores = _score_payload(scores_path.resolve())
    if scores["request_manifest"]["sha256"] != _sha256_file(request_path):
        raise ValueError("score result references a different frozen request manifest")
    expected_rows = {item["request_id"]: item for item in manifest["requests"]}
    score_map = {item["request_id"]: item for item in scores["scores"]}
    if len(expected_rows) != len(manifest["requests"]):
        raise ValueError("frozen request manifest contains duplicate request IDs")
    if len(score_map) != len(scores["scores"]):
        raise ValueError("native score result contains duplicate request IDs")
    if set(score_map) != set(expected_rows):
        raise ValueError("native score coverage does not exactly match frozen requests")

    class ReplayLM(LM):
        def __init__(self) -> None:
            super().__init__()
            self.used: set[str] = set()

        def loglikelihood(self, requests: Sequence[Any]) -> list[tuple[float, bool]]:
            output: list[tuple[float, bool]] = []
            for request in requests:
                request_id = _request_id(
                    str(request.task_name),
                    int(request.doc_id),
                    int(request.idx),
                    request.args,
                )
                frozen = expected_rows.get(request_id)
                if frozen is None or list(request.args) != [
                    frozen["context"],
                    frozen["continuation"],
                ]:
                    raise ValueError(
                        f"lm-eval replay request drift for {request.task_name}/{request.doc_id}/{request.idx}"
                    )
                row = score_map[request_id]
                self.used.add(request_id)
                output.append((float(row["loglikelihood"]), bool(row["is_greedy"])))
            return output

        def loglikelihood_rolling(self, requests: Sequence[Any]) -> list[float]:
            raise NotImplementedError("pilot supports loglikelihood tasks only")

        def generate_until(self, requests: Sequence[Any]) -> list[str]:
            raise NotImplementedError("pilot supports loglikelihood tasks only")

    manager = TaskManager()
    definitions = build_task_definitions(profile, manager)
    frozen_tasks = {item["task"]: item for item in manifest["tasks"]}
    if len(frozen_tasks) != len(manifest["tasks"]):
        raise ValueError("frozen request manifest contains duplicate task names")
    definition_names = {item.task_name for item in definitions}
    if set(frozen_tasks) != definition_names:
        missing = sorted(definition_names - set(frozen_tasks))
        unexpected = sorted(set(frozen_tasks) - definition_names)
        raise ValueError(
            "lm-eval task coverage drift: "
            f"missing frozen tasks={missing[:10]}, unexpected frozen tasks={unexpected[:10]}"
        )
    replay = ReplayLM()
    output_dir = output_dir.resolve()
    raw_dir = output_dir / "lm_eval_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Mapping[str, Any]] = {}
    for benchmark in [item["id"] for item in profile.benchmarks]:
        selected = [item for item in definitions if item.benchmark == benchmark]
        task_inputs = []
        for definition in selected:
            task = _load_leaf_task(manager, definition, profile.seed)
            _validate_replay_task(definition, task, frozen_tasks[definition.task_name])
            if definition.task_name not in manager.task_index:
                manager.task_index[definition.task_name] = {
                    "type": "task",
                    "yaml_path": str(profile.source_path),
                }
            task_inputs.append(task)
        sample_map = {
            item["task"]: item["sample_ids"]
            for item in manifest["tasks"]
            if item["benchmark"] == benchmark
        }
        result = simple_evaluate(
            model=replay,
            tasks=task_inputs,
            num_fewshot=0,
            batch_size=1,
            samples=sample_map,
            bootstrap_iters=0,
            log_samples=False,
            task_manager=manager,
            verbosity="INFO",
            random_seed=profile.seed,
            numpy_random_seed=profile.seed,
            torch_random_seed=profile.seed,
            fewshot_random_seed=profile.seed,
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"lm-eval returned no result for {benchmark}")
        raw_payload = dict(result)
        raw_payload.pop("samples", None)
        serializable_raw = json.loads(
            json.dumps(
                raw_payload,
                allow_nan=False,
                default=handle_non_serializable,
                ensure_ascii=False,
            )
        )
        _atomic_json(raw_dir / f"{benchmark}.json", serializable_raw)
        for task_name, metrics in result.get("results", {}).items():
            all_results[task_name] = metrics

    if replay.used != set(expected_rows):
        missing = sorted(set(expected_rows) - replay.used)
        raise RuntimeError(f"lm-eval replay did not consume every native score: {missing[:10]}")

    selected_counts = {item["task"]: int(item["documents"]) for item in manifest["tasks"]}
    benchmark_rows: list[dict[str, Any]] = []
    for benchmark_cfg in profile.benchmarks:
        benchmark = benchmark_cfg["id"]
        language_metrics: list[dict[str, Any]] = []
        languages = sorted(
            {item.language for item in definitions if item.benchmark == benchmark}
        )
        for language in languages:
            tasks = [
                item.task_name
                for item in definitions
                if item.benchmark == benchmark and item.language == language
            ]
            total = sum(selected_counts[name] for name in tasks)
            metric_row: dict[str, Any] = {
                "language": language,
                "documents": total,
                "leaf_tasks": tasks,
            }
            for metric in ("acc,none", "acc_norm,none"):
                values = [
                    (float(all_results[name][metric]), selected_counts[name])
                    for name in tasks
                    if metric in all_results.get(name, {})
                ]
                if values:
                    metric_row[metric.split(",", 1)[0]] = sum(
                        value * count for value, count in values
                    ) / sum(count for _, count in values)
            language_metrics.append(metric_row)
        benchmark_row: dict[str, Any] = {
            "benchmark": benchmark,
            "category": benchmark_cfg["category"],
            "languages": len(language_metrics),
            "documents": sum(item["documents"] for item in language_metrics),
            "per_language": language_metrics,
        }
        for metric in ("acc", "acc_norm"):
            values = [item[metric] for item in language_metrics if metric in item]
            if values:
                benchmark_row[f"macro_{metric}"] = sum(values) / len(values)
        benchmark_rows.append(benchmark_row)

    metrics: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "status": "complete",
        "pilot_id": profile.benchmark_id,
        "run_id": scores["run_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "request_manifest": {
            "path": str(request_path),
            "sha256": _sha256_file(request_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "scores": {"path": str(scores_path.resolve()), "result_sha256": scores["result_sha256"]},
        "harness": {
            "path": str(profile.harness_root),
            "commit": profile.payload["harness_commit"],
            "version": lm_eval.__version__,
        },
        "timing": scores["timing"],
        "checkpoint": scores["checkpoint"],
        "model": scores["model"],
        "counts": scores["counts"],
        "parity_gate": scores["parity_gate"],
        "replica_gate": scores["replica_gate"],
        "moe_routing": scores["moe_routing"],
        "benchmarks": benchmark_rows,
    }
    metrics["metrics_sha256"] = _payload_sha256(metrics)
    metrics_path = output_dir / "metrics.json"
    _atomic_json(metrics_path, metrics)
    return metrics_path


__all__ = ["aggregate_scores"]
