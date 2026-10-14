"""Large-table builders used by the Stage 1 A--D release orchestrator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    EXPECTED_COUNTS,
    Stage1ReleaseError,
    _json_cell,
    _load_json,
    _sha256,
)
from multilingual_megatron.experiments.scaling.stage1_release.training import aggregate_job_training
from multilingual_megatron.experiments.scaling.stage1_release.training import parse_megatron_training_log
from multilingual_megatron.experiments.scaling.stage1_release.training import parse_nvidia_smi_csv
from multilingual_megatron.experiments.scaling.stage1_release.validation import Stage1ValidationTables
from multilingual_megatron.experiments.scaling.stage1_release.exposure import (
    _exact_exposures,
    _resolve_training_inputs,
)


def build_training_tables(
    training_jobs: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for job in training_jobs:
        training_manifest_path = Path(str(job["training_manifest_path"])).resolve()
        training_manifest = _load_json(training_manifest_path)
        metadata = training_manifest.get("metadata", {})
        sequence_length = int(metadata.get("seq_length", 0) or 0)
        if sequence_length <= 0:
            raise Stage1ReleaseError(
                f"Missing sequence length: {training_manifest_path}"
            )
        segment_rows = metadata.get("segments", [])
        if not isinstance(segment_rows, list) or not segment_rows:
            raise Stage1ReleaseError(
                f"Training manifest has no segments: {training_manifest_path}"
            )
        job_metrics: list[dict[str, Any]] = []
        for raw_segment in segment_rows:
            segment_index = int(raw_segment["segment_index"])
            segment_id = f"{job['training_job_id']}:segment:{segment_index}"
            log_path = Path(str(raw_segment["log_path"])).resolve()
            if not log_path.is_file():
                raise Stage1ReleaseError(f"Missing training log: {log_path}")
            lineage = {
                "trajectory_id": job["trajectory_id"],
                "ladder": job["ladder"],
                "experts": job["experts"],
                "language_count": job["language_count"],
                "training_job_id": job["training_job_id"],
                "segment_id": segment_id,
                "segment_index": segment_index,
                "segment_kind": job["job_role"],
                "branch_parent_job_id": job["parent_training_job_id"],
                "source_path": str(log_path),
            }
            parsed = parse_megatron_training_log(
                log_path,
                sequence_length=sequence_length,
                lineage=lineage,
                timestamp_timezone="Europe/Berlin",
            )
            metrics.extend(parsed)
            job_metrics.extend(parsed)
            log_sha = _sha256(log_path)
            segments.append(
                {
                    "segment_id": segment_id,
                    "training_job_id": job["training_job_id"],
                    "trajectory_id": job["trajectory_id"],
                    "segment_index": segment_index,
                    "segment_kind": job["job_role"],
                    "parent_training_job_id": job["parent_training_job_id"],
                    "actual_start_tokens": raw_segment.get("actual_start_tokens"),
                    "actual_end_tokens": raw_segment.get("actual_end_tokens"),
                    "target_start_tokens": raw_segment.get("target_start_tokens"),
                    "target_end_tokens": raw_segment.get("target_end_tokens"),
                    "global_batch_size": raw_segment.get("global_batch_size"),
                    "micro_batch_size": raw_segment.get("micro_batch_size"),
                    "segment_train_iters": raw_segment.get("segment_train_iters"),
                    "log_path": str(log_path),
                    "log_sha256": log_sha,
                    "provenance_class": "observed",
                }
            )
            artifacts.append(
                {
                    "artifact_type": "training_log",
                    "object_id": segment_id,
                    "path": str(log_path),
                    "sha256": log_sha,
                    "size_bytes": log_path.stat().st_size,
                    "selected": True,
                    "provenance_class": "observed",
                }
            )
        run_root = Path(str(job["run_root"]))
        gpu_path = run_root / "logs" / f"{job['training_job_id']}.gpu.csv"
        if not gpu_path.is_file():
            raise Stage1ReleaseError(f"Missing GPU telemetry: {gpu_path}")
        gpu_lineage = {
            "trajectory_id": job["trajectory_id"],
            "ladder": job["ladder"],
            "experts": job["experts"],
            "language_count": job["language_count"],
            "training_job_id": job["training_job_id"],
            "segment_kind": job["job_role"],
            "branch_parent_job_id": job["parent_training_job_id"],
            "source_path": str(gpu_path),
        }
        parsed_gpu = parse_nvidia_smi_csv(
            gpu_path,
            lineage=gpu_lineage,
            timestamp_timezone="Europe/Berlin",
        )
        gpu_samples.extend(parsed_gpu)
        artifacts.append(
            {
                "artifact_type": "gpu_telemetry_csv",
                "object_id": job["training_job_id"],
                "path": str(gpu_path),
                "sha256": _sha256(gpu_path),
                "size_bytes": gpu_path.stat().st_size,
                "selected": True,
                "provenance_class": "observed",
            }
        )
        log_interval = int(
            training_manifest.get("run_cfg", {})
            .get("train", {})
            .get("log_interval", 10)
        )
        expected_metric_rows = sum(
            max(1, int(raw.get("segment_train_iters", 0)) // log_interval)
            for raw in segment_rows
        )
        summary = aggregate_job_training(
            job_metrics,
            parsed_gpu,
            gpu_count=None,
            walltime_seconds=float(job["end_to_end_seconds"]),
            expected_logged_metric_rows=expected_metric_rows,
            expected_gpu_sample_interval_seconds=10.0,
            lineage={
                "training_job_id": job["training_job_id"],
                "trajectory_id": job["trajectory_id"],
                "ladder": job["ladder"],
                "experts": job["experts"],
                "language_count": job["language_count"],
                "segment_kind": job["job_role"],
                "branch_parent_job_id": job["parent_training_job_id"],
            },
        )
        summary["lineage_records_json"] = _json_cell(summary.pop("lineage_records", []))
        summary["parameter_counts_json"] = _json_cell(
            summary.pop("parameter_counts", [])
        )
        summary["telemetry_gpu_uuids_json"] = _json_cell(
            summary.pop("telemetry_gpu_uuids", [])
        )
        summaries.append(summary)
        artifacts.append(
            {
                "artifact_type": "training_manifest",
                "object_id": job["training_job_id"],
                "path": str(training_manifest_path),
                "sha256": _sha256(training_manifest_path),
                "size_bytes": training_manifest_path.stat().st_size,
                "selected": True,
                "provenance_class": "observed",
            }
        )
    return {
        "training_segments": segments,
        "training_metrics": metrics,
        "gpu_samples": gpu_samples,
        "training_summary": summaries,
        "training_artifacts": artifacts,
    }


def build_data_exposure_rows(
    *,
    validation: Stage1ValidationTables,
    corpus_manifest_path: Path,
) -> list[dict[str, Any]]:
    corpus = _load_json(corpus_manifest_path)
    inventories = corpus["languages"]
    language_role = {
        (str(row["readout_id"]), str(row["language"])): str(row["metric_role"])
        for row in validation.validation_language
    }
    exposure_cache: dict[tuple[str, int], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for readout in validation.readouts:
        evaluation_manifest_path = Path(
            str(readout["evaluation_manifest_path"])
        ).resolve()
        (
            training_manifest_path,
            index_path,
            description_path,
            training,
        ) = _resolve_training_inputs(evaluation_manifest_path)
        sequence_length = int(training["metadata"]["seq_length"])
        actual_tokens = int(readout["actual_tokens"])
        if actual_tokens % sequence_length:
            raise Stage1ReleaseError(
                f"Unaligned actual token count: {readout['readout_id']}"
            )
        end_samples = actual_tokens // sequence_length
        tier_manifest_path = Path(training["run_cfg"]["data_manifest"]).resolve()
        tier_manifest = _load_json(tier_manifest_path)
        if index_path is None or description_path is None:
            tier_languages = [str(value) for value in tier_manifest["languages"]]
            if int(readout["language_count"]) != 1 or len(tier_languages) != 1:
                raise Stage1ReleaseError(
                    f"Missing multilingual blend index: {readout['readout_id']}"
                )
            exposures = {tier_languages[0]: float(actual_tokens)}
        else:
            key = (str(index_path.resolve()), end_samples)
            if key not in exposure_cache:
                exposure_cache[key] = _exact_exposures(
                    index_path=index_path,
                    description_path=description_path,
                    tier_manifest_path=tier_manifest_path,
                    end_samples=end_samples,
                    sequence_length=sequence_length,
                )
            exposures = exposure_cache[key]
        if len(exposures) != int(readout["language_count"]):
            raise Stage1ReleaseError(
                f"Exposure count mismatch: {readout['readout_id']}"
            )
        for language, tokens in sorted(exposures.items()):
            inventory = inventories[language]
            exact_unique = inventory.get("exact_unique_language_tokens_including_eod")
            lower_unique = inventory.get(
                "observed_unique_language_tokens_lower_bound_including_eod"
            )
            rows.append(
                {
                    "readout_id": readout["readout_id"],
                    "trajectory_id": readout["trajectory_id"],
                    "ladder": readout["ladder"],
                    "experts": readout["experts"],
                    "language_count": readout["language_count"],
                    "nominal_tokens": readout["nominal_tokens"],
                    "actual_tokens": actual_tokens,
                    "language": language,
                    "metric_role": language_role.get(
                        (str(readout["readout_id"]), language)
                    ),
                    "realized_self_exposure_tokens": float(tokens),
                    "exact_unique_language_tokens_including_eod": exact_unique,
                    "unique_language_tokens_lower_bound_including_eod": lower_unique,
                    "effective_exposure_passes": (
                        float(tokens) / float(exact_unique)
                        if exact_unique not in (None, 0)
                        else None
                    ),
                    "training_manifest_path": str(training_manifest_path),
                    "tier_manifest_path": str(tier_manifest_path),
                    "tier_manifest_sha256": _sha256(tier_manifest_path),
                    "blend_index_path": (
                        None if index_path is None else str(index_path.resolve())
                    ),
                    "blend_description_path": (
                        None
                        if description_path is None
                        else str(description_path.resolve())
                    ),
                    "corpus_manifest_path": str(corpus_manifest_path.resolve()),
                    "corpus_manifest_sha256": _sha256(corpus_manifest_path),
                    "provenance_class": "observed",
                }
            )
    if len(rows) != EXPECTED_COUNTS["trained_language"]:
        raise Stage1ReleaseError(
            f"Expected {EXPECTED_COUNTS['trained_language']} exposure rows, "
            f"got {len(rows)}"
        )
    return rows


def build_routing_rows(
    validation_language: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language_row in validation_language:
        path = Path(str(language_row["language_artifact_path"])).resolve()
        payload = _load_json(path)
        routing = payload.get("moe_routing", {})
        if not isinstance(routing, Mapping) or not routing.get("applicable"):
            continue
        layers = routing.get("layers", [])
        if not isinstance(layers, list):
            raise Stage1ReleaseError(f"Malformed routing layers: {path}")
        for layer in layers:
            imbalance = layer.get("imbalance", {})
            rows.append(
                {
                    "readout_id": language_row["readout_id"],
                    "trajectory_id": language_row["trajectory_id"],
                    "ladder": language_row["ladder"],
                    "experts": language_row["experts"],
                    "language_count": language_row["language_count"],
                    "nominal_tokens": language_row["nominal_tokens"],
                    "language": language_row["language"],
                    "metric_role": language_row["metric_role"],
                    "layer_index": layer.get("layer_index"),
                    "topk": layer.get("topk"),
                    "expected_assignments": layer.get("expected_assignments"),
                    "kept_assignments": layer.get("kept_assignments"),
                    "dropped_assignments": layer.get("dropped_assignments"),
                    "dropped_assignment_rate": layer.get("dropped_assignment_rate"),
                    "forwarded_tokens": layer.get("forwarded_tokens"),
                    "coefficient_of_variation": imbalance.get(
                        "coefficient_of_variation"
                    ),
                    "max_to_mean": imbalance.get("max_to_mean"),
                    "min_to_mean": imbalance.get("min_to_mean"),
                    "normalized_entropy": imbalance.get("normalized_entropy"),
                    "switch_aux_loss_unweighted": layer.get(
                        "switch_aux_loss_unweighted"
                    ),
                    "switch_aux_loss_weighted": layer.get("switch_aux_loss_weighted"),
                    "router_z_loss_unweighted": layer.get("router_z_loss_unweighted"),
                    "router_z_loss_weighted": layer.get("router_z_loss_weighted"),
                    "expert_token_counts_json": _json_cell(
                        layer.get("expert_token_counts", [])
                    ),
                    "language_artifact_path": str(path),
                    "language_artifact_sha256": language_row[
                        "language_artifact_sha256"
                    ],
                    "provenance_class": "observed",
                }
            )
    return rows


__all__ = [
    "build_data_exposure_rows",
    "build_routing_rows",
    "build_training_tables",
]
