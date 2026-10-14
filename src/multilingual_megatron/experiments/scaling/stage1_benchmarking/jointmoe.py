"""Prepare and summarize a profile-driven Stage-1 JointMoE benchmark campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.stage1_benchmarking.api import (
    METRICS_SCHEMA,
    BenchmarkProfile,
    _atomic_json,
    _load_json,
    _payload_sha256,
    _sha256_file,
    _verify_payload_hash,
    aggregate_scores,
    build_score_specs,
    freeze_requests,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
    BENCHMARK_ORDER,
    CampaignProfile,
    load_campaign_profile,
)
from multilingual_megatron.experiments.scaling.stage1_benchmarking.runtime_estimate import (
    write_runtime_estimate,
)
from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import (
    load_native_benchmark_score_spec,
)
from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_native_benchmark import (
    build_native_benchmark_command,
)

PREPARATION_SCHEMA = "stage1_jointmoe_benchmark_preparation/v1"
REPORT_SCHEMA = "stage1_jointmoe_benchmark_report/v1"
TABLE_MANIFEST_SCHEMA = "stage1_jointmoe_benchmark_table_manifest/v1"
STATUS_SCHEMA = "stage1_jointmoe_benchmark_status/v1"
PREFLIGHT_SCHEMA = "stage1_jointmoe_benchmark_preflight/v1"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_structured_rows(
    root: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"structured table {name} has no rows")
    fieldnames = list(rows[0])
    expected_fields = set(fieldnames)
    if any(set(row) != expected_fields for row in rows):
        raise ValueError(f"structured table {name} has inconsistent columns")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row[key]) for key in fieldnames})
    csv_path = root / f"{name}.csv"
    _atomic_text(csv_path, csv_buffer.getvalue())

    jsonl_path = root / f"{name}.jsonl"
    _atomic_text(
        jsonl_path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )
    return {
        "rows": len(rows),
        "columns": fieldnames,
        "csv": {"path": str(csv_path), "sha256": _sha256_file(csv_path)},
        "jsonl": {"path": str(jsonl_path), "sha256": _sha256_file(jsonl_path)},
    }


def _model_table_row(model: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = model.get("checkpoint", {})
    topology = model.get("a40_topology", {})
    return {
        "model_id": str(model["id"]),
        "trajectory_id": str(model["trajectory_id"]),
        "readout_id": str(model["readout_id"]),
        "ladder": str(model["ladder"]),
        "experts": int(model["experts"]),
        "languages": int(model["languages"]),
        "seed": int(model["seed"]),
        "source_cohort": str(model["source_cohort"]),
        "replication_role": str(model["replication_role"]),
        "request_set": str(model["request_set"]),
        "nominal_tokens": int(model["nominal_tokens"]),
        "actual_tokens": int(checkpoint.get("tokens_seen", model.get("tokens_seen", 0))),
        "checkpoint_step": int(checkpoint.get("step", model.get("checkpoint_step", 0))),
        "readout_index": int(model["readout_index"]),
        "readout_count": int(model["readout_count"]),
        "is_final_readout": bool(model["is_final_readout"]),
        "active_parameters": int(model["active_parameters"]),
        "configured_total_parameters": int(model["configured_total_parameters"]),
        "peak_lr": float(model["peak_lr"]),
        "world_size": int(topology.get("world_size", model.get("expert_parallel_size", 0))),
        "checkpoint_path": str(checkpoint.get("path", "")),
    }


def _pilot_profile(
    profile: CampaignProfile,
    request_set: str,
    models: Sequence[Mapping[str, Any]],
) -> BenchmarkProfile:
    if request_set not in {"english", "multilingual"}:
        raise ValueError(f"unknown request set: {request_set}")
    request_contract = profile.payload["request_sets"][request_set]
    sampling = request_contract["sampling"]
    benchmarks = request_contract["benchmarks"]
    request_pilot_id = request_contract["pilot_id"]

    checkpoints: list[dict[str, Any]] = []
    for model in models:
        experts = int(model["experts"])
        topology = profile.topology(model)
        _training_manifest, evaluation_spec, _run_id = profile.training_paths(model)
        checkpoints.append(
            {
                "id": model["id"],
                "active_parameters": int(model["active_parameters"]),
                "experts": experts,
                "languages": int(model["languages"]),
                "tokens_seen": int(model["tokens_seen"]),
                "evaluation_spec": str(evaluation_spec),
                "gpus": int(topology["nodes"]) * int(topology["gpus_per_node"]),
                "micro_batch_size": int(topology["micro_batch_size"]),
                "max_gpu_hours": float(topology["max_gpu_hours"]),
                "training_model_override_replacements": dict(
                    model["training_model_override_replacements"]
                ),
            }
        )
    payload = {
        "pilot_id": request_pilot_id,
        "request_manifest_pilot_id": request_pilot_id,
        "harness_commit": profile.payload["harness_commit"],
        "bridge_python": str(profile.bridge_python),
        "tokenizer": profile.payload["tokenizer"],
        "benchmarks": benchmarks,
        "checkpoints": checkpoints,
        "parity": profile.payload["parity"],
        "runtime": {"required_gpu_name_substring": "A40"},
    }
    return BenchmarkProfile(
        source_path=profile.source_path,
        source_sha256=profile.source_sha256,
        payload=payload,
        repo_root=profile.repo_root,
        harness_root=profile.harness_root,
        bridge_root=profile.bridge_root,
        output_root=profile.output_root,
        hf_home=profile.hf_home,
        seed=int(sampling["seed"]),
        examples_per_language=int(sampling["examples_per_language"]),
        shared_population=int(sampling["shared_population"]),
    )


def _format_override(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def _apply_a40_model_overrides(
    score_spec: dict[str, Any], replacements: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    overrides = list(score_spec["model_recipe"]["overrides"])
    receipt: dict[str, dict[str, Any]] = {}
    for field, value in replacements.items():
        prefix = f"model.{field}="
        indices = [index for index, item in enumerate(overrides) if item.startswith(prefix)]
        if len(indices) != 1:
            raise ValueError(f"expected one source override for {prefix}, got {len(indices)}")
        index = indices[0]
        old = overrides[index].split("=", 1)[1]
        new = _format_override(value)
        if old == new:
            raise ValueError(f"unnecessary A40 replacement for model.{field}")
        overrides[index] = prefix + new
        score_spec["model_recipe"]["expected_model"][field] = value
        receipt[field] = {
            "source_evaluation_value": old,
            "a40_evaluation_value": value,
        }
    score_spec["model_recipe"]["overrides"] = overrides
    return receipt


def prepare(profile_path: Path) -> Path:
    profile = load_campaign_profile(profile_path)
    profile.output_root.mkdir(parents=True, exist_ok=True)

    models_by_request_set = {
        request_set: [
            item
            for item in profile.payload["models"]
            if item["request_set"] == request_set
        ]
        for request_set in profile.payload["request_sets"]
    }
    request_manifests: dict[str, Path] = {}
    for request_set, models in models_by_request_set.items():
        if not models:
            continue
        request_manifests[request_set] = freeze_requests(
            _pilot_profile(profile, request_set, models),
            profile.request_manifest(request_set),
        )

    written_specs: list[Path] = []
    for request_set, models in models_by_request_set.items():
        if not models:
            continue
        pilot = _pilot_profile(profile, request_set, models)
        written_specs.extend(
            build_score_specs(
                pilot,
                request_manifests[request_set],
                profile.output_root / "specs" / request_set,
            )
        )

    prepared_models: list[dict[str, Any]] = []
    request_manifest_cache: dict[Path, tuple[str, Mapping[str, Any]]] = {}
    for model in profile.payload["models"]:
        model_id = str(model["id"])
        experts = int(model["experts"])
        request_set = profile.request_set_name(model_id)
        spec_path = profile.score_spec(model_id)
        if spec_path not in written_specs:
            raise RuntimeError(f"missing generated score spec for {model_id}")
        score_spec = copy.deepcopy(_load_json(spec_path, label=f"{model_id} score spec"))
        replacement_receipt = _apply_a40_model_overrides(
            score_spec, profile.topology(model)["a40_model_overrides"]
        )
        training_manifest, evaluation_spec, run_id = profile.training_paths(model)
        score_spec["source_training_manifest"] = {
            "path": str(training_manifest),
            "sha256": _sha256_file(training_manifest),
            "run_id": run_id,
        }
        score_spec["learning_rate_contract"] = {
            "name": "jointmoe_x1",
            "peak_lr": float(model["peak_lr"]),
            "formula": profile.payload["learning_rate_contract"]["formula"],
        }
        score_spec["request_set"] = request_set
        score_spec["model_recipe"]["a40_runtime_replacements"] = replacement_receipt
        score_spec.pop("spec_sha256", None)
        score_spec["spec_sha256"] = _payload_sha256(score_spec)
        _atomic_json(spec_path, score_spec)
        validated = load_native_benchmark_score_spec(
            spec_path,
            request_manifest_cache=request_manifest_cache,
        )
        prepared_models.append(
            {
                "id": model_id,
                "ladder": str(model["ladder"]),
                "experts": experts,
                "languages": int(model["languages"]),
                "seed": int(model["seed"]),
                "trajectory_id": str(model["trajectory_id"]),
                "readout_id": str(model["readout_id"]),
                "source_cohort": str(model["source_cohort"]),
                "replication_role": str(model["replication_role"]),
                "timing_calibration": bool(model["timing_calibration"]),
                "request_set": request_set,
                "nominal_tokens": int(model["nominal_tokens"]),
                "readout_index": int(model["readout_index"]),
                "readout_count": int(model["readout_count"]),
                "is_final_readout": bool(model["is_final_readout"]),
                "training_manifest": {
                    "path": str(training_manifest),
                    "sha256": _sha256_file(training_manifest),
                },
                "evaluation_spec": {
                    "path": str(evaluation_spec),
                    "sha256": _sha256_file(evaluation_spec),
                },
                "evaluation_manifest": {
                    "path": str(model["evaluation_manifest"]),
                    "sha256": str(model["evaluation_manifest_sha256"]),
                },
                "score_spec": {
                    "path": str(spec_path),
                    "sha256": _sha256_file(spec_path),
                    "spec_sha256": score_spec["spec_sha256"],
                },
                "checkpoint": {
                    "path": str(validated.checkpoint_iteration_path),
                    "step": validated.checkpoint_step,
                    "tokens_seen": int(model["tokens_seen"]),
                },
                "active_parameters": int(model["active_parameters"]),
                "configured_total_parameters": int(model["configured_total_parameters"]),
                "peak_lr": float(model["peak_lr"]),
                "a40_topology": {
                    "nodes": int(profile.topology(model)["nodes"]),
                    "gpus_per_node": int(profile.topology(model)["gpus_per_node"]),
                    "world_size": validated.expected_world_size,
                    "micro_batch_size": int(profile.topology(model)["micro_batch_size"]),
                },
                "a40_runtime_replacements": replacement_receipt,
                "max_gpu_hours": validated.max_gpu_hours,
            }
        )

    request_receipts: dict[str, Any] = {}
    for request_set, manifest_path in request_manifests.items():
        manifest = _load_json(manifest_path, label=f"{request_set} request manifest")
        _verify_payload_hash(
            manifest, field="manifest_sha256", label=f"{request_set} request manifest"
        )
        request_receipts[request_set] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "requests": int(manifest["coverage"]["requests"]),
            "leaf_tasks": int(manifest["coverage"]["leaf_tasks"]),
            "languages": manifest["coverage"]["languages"],
        }

    receipt: dict[str, Any] = {
        "schema": PREPARATION_SCHEMA,
        "status": "validated_not_submitted",
        "campaign_id": profile.payload["campaign_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "source_release": {
            "name": profile.payload["source"]["release_name"],
            "manifest": profile.payload["source"]["release_manifest"],
            "sha256": profile.payload["source"]["release_manifest_sha256"],
        },
        "source_artifacts": profile.payload["resolved_source_artifacts"],
        "selection": profile.payload["selection"],
        "learning_rate_contract": profile.payload["learning_rate_contract"],
        "request_sets": request_receipts,
        "models": prepared_models,
        "budget": profile.payload["budget"],
        "runtime_estimation": {
            **profile.payload["runtime_estimation"],
            "calibration_model_ids": list(profile.calibration_model_ids),
            "calibration_models": len(profile.calibration_model_ids),
            "remaining_models": len(profile.model_ids) - len(profile.calibration_model_ids),
        },
        "submitted": False,
    }
    inventory_root = profile.output_root / "analysis_inputs"
    inventory_table = _write_structured_rows(
        inventory_root,
        "model_inventory",
        [_model_table_row(model) for model in prepared_models],
    )
    inventory_manifest: dict[str, Any] = {
        "schema": TABLE_MANIFEST_SCHEMA,
        "status": "prepared",
        "campaign_id": profile.payload["campaign_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "tables": {"model_inventory": inventory_table},
    }
    inventory_manifest["manifest_sha256"] = _payload_sha256(inventory_manifest)
    inventory_manifest_path = inventory_root / "manifest.json"
    _atomic_json(inventory_manifest_path, inventory_manifest)
    receipt["analysis_inputs"] = {
        "manifest": str(inventory_manifest_path),
        "sha256": _sha256_file(inventory_manifest_path),
        "manifest_sha256": inventory_manifest["manifest_sha256"],
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    receipt_path = profile.output_root / "preparation_receipt.json"
    _atomic_json(receipt_path, receipt)
    return receipt_path


def aggregate(profile_path: Path, model_id: str) -> Path:
    profile = load_campaign_profile(profile_path)
    model = profile.model(model_id)
    request_set = profile.request_set_name(model_id)
    pilot = _pilot_profile(profile, request_set, [model])
    return aggregate_scores(
        pilot,
        profile.request_manifest(request_set),
        profile.scores(model_id),
        profile.metrics_dir(model_id),
    )


def preflight(profile_path: Path) -> Path:
    profile = load_campaign_profile(profile_path)
    preparation_path = profile.output_root / "preparation_receipt.json"
    preparation = _load_json(preparation_path, label="benchmark preparation receipt")
    if (
        preparation.get("campaign_id") != profile.payload["campaign_id"]
        or preparation.get("profile", {}).get("sha256") != profile.source_sha256
        or preparation.get("status") != "validated_not_submitted"
    ):
        raise ValueError("preparation receipt does not match the campaign profile")
    prepared_by_id = {
        str(model["id"]): model for model in preparation.get("models", [])
    }
    if set(prepared_by_id) != set(profile.model_ids):
        raise ValueError("preparation receipt model inventory differs from profile")

    request_manifest_cache: dict[Path, tuple[str, Mapping[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for model_id in profile.model_ids:
        model = profile.model(model_id)
        prepared = prepared_by_id[model_id]
        spec_path = profile.score_spec(model_id)
        if _sha256_file(spec_path) != prepared["score_spec"]["sha256"]:
            raise ValueError(f"prepared score-spec hash drift for {model_id}")
        spec = load_native_benchmark_score_spec(
            spec_path,
            request_manifest_cache=request_manifest_cache,
        )
        topology = profile.topology(model)
        command_args = argparse.Namespace(
            nnodes=int(topology["nodes"]),
            nproc_per_node=int(topology["gpus_per_node"]),
            launcher="python",
            python_executable=str(profile.bridge_python),
        )
        command = build_native_benchmark_command(command_args, spec)
        rows.append(
            {
                "model_id": model_id,
                "readout_id": str(model["readout_id"]),
                "nominal_tokens": int(model["nominal_tokens"]),
                "request_set": str(model["request_set"]),
                "score_spec_sha256": spec.source_sha256,
                "checkpoint_step": spec.checkpoint_step,
                "world_size": spec.expected_world_size,
                "nodes": int(topology["nodes"]),
                "gpus_per_node": int(topology["gpus_per_node"]),
                "command_sha256": _payload_sha256({"command": command}),
            }
        )

    request_manifests = {
        str(path): {"sha256": sha256}
        for path, (sha256, _payload) in sorted(
            request_manifest_cache.items(), key=lambda item: str(item[0])
        )
    }
    runtime_source_relatives = (
        "scripts/slurm/run_stage1_jointmoe_benchmark_job.sh",
        "scripts/slurm/submit_stage1_jointmoe_benchmark.sh",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/jointmoe.py",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/api.py",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/campaign_profile.py",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/campaign_models.py",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/contracts.py",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/replay.py",
        "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/tasks.py",
        "src/multilingual_megatron/pipeline/backends/megatron_bridge/run_megatron_bridge_native_benchmark.py",
        "src/multilingual_megatron/pipeline/backends/megatron_bridge/run_study_bridge_recipe.py",
        "src/multilingual_megatron/pipeline/stages/benchmarking/native_megatron_lm_eval.py",
        "src/multilingual_megatron/pipeline/stages/benchmarking/native_megatron_runner.py",
        "src/multilingual_megatron/pipeline/stages/benchmarking/native_megatron_scoring.py",
    )
    runtime_sources: dict[str, dict[str, str]] = {}
    for relative in runtime_source_relatives:
        source = (profile.repo_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing benchmark runtime source: {source}")
        runtime_sources[relative] = {
            "path": str(source),
            "sha256": _sha256_file(source),
        }
    counts_by_world_size = Counter(row["world_size"] for row in rows)
    payload: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "validated_not_submitted",
        "campaign_id": profile.payload["campaign_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "preparation_receipt": {
            "path": str(preparation_path),
            "sha256": _sha256_file(preparation_path),
        },
        "counts": {
            "models": len(rows),
            "request_manifests": len(request_manifests),
            "by_world_size": {
                str(world_size): count
                for world_size, count in sorted(counts_by_world_size.items())
            },
        },
        "request_manifests": request_manifests,
        "runtime_sources": runtime_sources,
        "models": rows,
        "submitted": False,
    }
    payload["receipt_sha256"] = _payload_sha256(payload)
    output = profile.output_root / "preflight_receipt.json"
    _atomic_json(output, payload)
    return output


def report(profile_path: Path) -> Path:
    profile = load_campaign_profile(profile_path)
    preparation_path = profile.output_root / "preparation_receipt.json"
    preparation = _load_json(preparation_path, label="benchmark preparation receipt")
    prepared_by_id = {
        str(model["id"]): model for model in preparation.get("models", [])
    }
    if set(prepared_by_id) != set(profile.model_ids):
        raise ValueError("preparation receipt model inventory differs from profile")

    rows: list[dict[str, Any]] = []
    model_table_rows: list[dict[str, Any]] = []
    benchmark_table_rows: list[dict[str, Any]] = []
    language_table_rows: list[dict[str, Any]] = []
    for model_id in profile.model_ids:
        metrics_path = profile.metrics_dir(model_id) / "metrics.json"
        metrics = _load_json(metrics_path, label=f"{model_id} metrics")
        if metrics.get("schema") != METRICS_SCHEMA or metrics.get("status") != "complete":
            raise ValueError(f"incomplete metrics for {model_id}")
        model = profile.model(model_id)
        prepared_model = prepared_by_id[model_id]
        runtime_path = profile.runtime_receipt(model_id)
        runtime = _load_json(runtime_path, label=f"{model_id} runtime receipt")
        if (
            runtime.get("schema") != "stage1_jointmoe_benchmark_runtime/v1"
            or runtime.get("status") != "complete"
            or runtime.get("campaign_id") != profile.payload["campaign_id"]
            or runtime.get("model_id") != model_id
            or runtime.get("profile", {}).get("sha256") != profile.source_sha256
        ):
            raise ValueError(f"incomplete or mismatched runtime receipt for {model_id}")
        runtime_metrics = runtime.get("metrics", {})
        if (
            runtime_metrics.get("path") != str(metrics_path)
            or runtime_metrics.get("sha256") != _sha256_file(metrics_path)
        ):
            raise ValueError(f"runtime receipt does not seal metrics for {model_id}")

        benchmarks = metrics.get("benchmarks")
        if not isinstance(benchmarks, list) or tuple(
            item.get("benchmark") for item in benchmarks
        ) != BENCHMARK_ORDER:
            raise ValueError(f"benchmark metric order drift for {model_id}")
        equal_task_macro_acc = sum(float(item["macro_acc"]) for item in benchmarks) / len(
            benchmarks
        )
        equal_task_macro_acc_norm = sum(
            float(item["macro_acc_norm"]) for item in benchmarks
        ) / len(benchmarks)
        base_row = _model_table_row(prepared_model)
        model_table_rows.append(
            {
                **base_row,
                "equal_task_macro_acc": equal_task_macro_acc,
                "equal_task_macro_acc_norm": equal_task_macro_acc_norm,
                "score_path": str(runtime["score_result"]["path"]),
                "score_sha256": str(runtime["score_result"]["sha256"]),
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256_file(metrics_path),
                "runtime_receipt_path": str(runtime_path),
                "runtime_receipt_sha256": _sha256_file(runtime_path),
                "slurm_job_id": str(runtime["slurm"]["job_id"]),
                "job_wall_seconds": int(runtime["timing"]["job_wall_seconds"]),
                "gpu_count": int(runtime["timing"]["gpu_count"]),
                "gpu_hours": float(runtime["timing"]["gpu_hours"]),
            }
        )
        task_base = {
            key: base_row[key]
            for key in (
                "model_id",
                "trajectory_id",
                "readout_id",
                "ladder",
                "experts",
                "languages",
                "seed",
                "source_cohort",
                "request_set",
                "nominal_tokens",
                "actual_tokens",
                "checkpoint_step",
                "readout_index",
                "readout_count",
                "is_final_readout",
            )
        }
        for benchmark in benchmarks:
            benchmark_table_rows.append(
                {
                    **task_base,
                    "benchmark": str(benchmark["benchmark"]),
                    "category": str(benchmark["category"]),
                    "documents": int(benchmark["documents"]),
                    "evaluated_languages": int(benchmark["languages"]),
                    "macro_acc": float(benchmark["macro_acc"]),
                    "macro_acc_norm": float(benchmark["macro_acc_norm"]),
                }
            )
            per_language = benchmark.get("per_language")
            if not isinstance(per_language, list):
                raise ValueError(f"missing per-language metrics for {model_id}")
            for language in per_language:
                language_table_rows.append(
                    {
                        **task_base,
                        "benchmark": str(benchmark["benchmark"]),
                        "category": str(benchmark["category"]),
                        "language": str(language["language"]),
                        "documents": int(language["documents"]),
                        "acc": float(language["acc"]),
                        "acc_norm": float(language["acc_norm"]),
                        "leaf_tasks": list(language["leaf_tasks"]),
                    }
                )
        rows.append(
            {
                "model_id": model_id,
                "ladder": str(model["ladder"]),
                "experts": int(model["experts"]),
                "languages": int(model["languages"]),
                "seed": int(model["seed"]),
                "trajectory_id": str(model["trajectory_id"]),
                "readout_id": str(model["readout_id"]),
                "nominal_tokens": int(model["nominal_tokens"]),
                "actual_tokens": int(model["tokens_seen"]),
                "checkpoint_step": int(model["checkpoint_step"]),
                "readout_index": int(model["readout_index"]),
                "readout_count": int(model["readout_count"]),
                "is_final_readout": bool(model["is_final_readout"]),
                "source_cohort": str(model["source_cohort"]),
                "replication_role": str(model["replication_role"]),
                "request_set": profile.request_set_name(model_id),
                "peak_lr": float(model["peak_lr"]),
                "equal_task_macro_acc": equal_task_macro_acc,
                "equal_task_macro_acc_norm": equal_task_macro_acc_norm,
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256_file(metrics_path),
                "runtime_receipt_path": str(runtime_path),
                "runtime_receipt_sha256": _sha256_file(runtime_path),
                "benchmarks": benchmarks,
            }
        )

    table_root = profile.output_root / "analysis_tables"
    tables = {
        "model_readouts": _write_structured_rows(
            table_root, "model_readouts", model_table_rows
        ),
        "benchmark_metrics": _write_structured_rows(
            table_root, "benchmark_metrics", benchmark_table_rows
        ),
        "language_metrics": _write_structured_rows(
            table_root, "language_metrics", language_table_rows
        ),
    }
    table_manifest: dict[str, Any] = {
        "schema": TABLE_MANIFEST_SCHEMA,
        "status": "complete",
        "campaign_id": profile.payload["campaign_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "tables": tables,
        "aggregation": {
            "task_macro": "equal weight over benchmark languages",
            "campaign_composite": "equal weight over the six task macro metrics",
            "primary_metric": "acc_norm",
        },
    }
    table_manifest["manifest_sha256"] = _payload_sha256(table_manifest)
    table_manifest_path = table_root / "manifest.json"
    _atomic_json(table_manifest_path, table_manifest)

    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "campaign_id": profile.payload["campaign_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "models": rows,
        "analysis_tables": {
            "manifest": str(table_manifest_path),
            "sha256": _sha256_file(table_manifest_path),
            "manifest_sha256": table_manifest["manifest_sha256"],
        },
        "comparison_policy": {
            "within_k_across_e": (
                "compare only entries sharing the same frozen request manifest and "
                "independent training seed"
            ),
            "k1": (
                "English-only; do not compare its multilingual macro average to K>=12"
            ),
            "cross_k": (
                "report English and frozen common-language intersections before "
                "interpreting K scaling"
            ),
            "seed_scope": "seed-general inference is limited to the declared replicated B cohort",
        },
    }
    payload["report_sha256"] = _payload_sha256(payload)
    report_path = profile.output_root / "comparison_report.json"
    _atomic_json(report_path, payload)
    return report_path


def status(profile_path: Path) -> Path:
    profile = load_campaign_profile(profile_path)
    complete: list[str] = []
    partial: list[str] = []
    pending: list[str] = []
    for model_id in profile.model_ids:
        score_exists = profile.scores(model_id).is_file()
        metrics_exists = (profile.metrics_dir(model_id) / "metrics.json").is_file()
        runtime_exists = profile.runtime_receipt(model_id).is_file()
        if score_exists and metrics_exists and runtime_exists:
            complete.append(model_id)
        elif score_exists or metrics_exists or runtime_exists:
            partial.append(model_id)
        else:
            pending.append(model_id)
    payload: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "campaign_id": profile.payload["campaign_id"],
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "counts": {
            "expected": len(profile.model_ids),
            "complete": len(complete),
            "partial": len(partial),
            "pending": len(pending),
        },
        "ready_for_report": len(complete) == len(profile.model_ids),
        "complete_model_ids": complete,
        "partial_model_ids": partial,
        "pending_model_ids": pending,
    }
    payload["status_sha256"] = _payload_sha256(payload)
    output = profile.output_root / "campaign_status.json"
    _atomic_json(output, payload)
    return output


def analyze(profile_path: Path) -> Path:
    """Delegate completed-campaign analysis to result_analysis."""

    from result_analysis.downstream_evaluations.campaign.analysis import (
        write_scientific_analysis_bundle,
    )

    return write_scientific_analysis_bundle(profile_path)


def visualize(profile_path: Path) -> Path:
    """Render the optional, deterministic benchmark-scaling figure bundle."""

    from result_analysis.downstream_evaluations.campaign.analysis_plots import (
        write_benchmark_scaling_visualizations,
    )

    return write_benchmark_scaling_visualizations(profile_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("prepare")
    subparsers.add_parser("preflight")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--model-id", required=True)
    subparsers.add_parser("estimate-runtime")
    subparsers.add_parser("status")
    subparsers.add_parser("report")
    subparsers.add_parser("analyze")
    subparsers.add_parser("visualize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile_path = Path(args.profile)
    if args.command == "validate":
        output: Path = load_campaign_profile(profile_path).source_path
    elif args.command == "prepare":
        output = prepare(profile_path)
    elif args.command == "preflight":
        output = preflight(profile_path)
    elif args.command == "aggregate":
        output = aggregate(profile_path, args.model_id)
    elif args.command == "estimate-runtime":
        output = write_runtime_estimate(load_campaign_profile(profile_path))
    elif args.command == "status":
        output = status(profile_path)
    elif args.command == "analyze":
        output = analyze(profile_path)
    elif args.command == "visualize":
        output = visualize(profile_path)
    else:
        output = report(profile_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ORDER",
    "aggregate",
    "analyze",
    "load_campaign_profile",
    "preflight",
    "prepare",
    "report",
    "status",
    "visualize",
    "write_runtime_estimate",
]
