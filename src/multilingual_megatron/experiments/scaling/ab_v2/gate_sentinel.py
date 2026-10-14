"""Sentinel integration and conservative campaign-cost gate.

The sentinel is intentionally its own boundary: it checks real training and
evaluation evidence, then projects campaign cost from that evidence.  It does
not schedule, submit, or mutate the campaign.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .common import (
    DEFAULT_RUNTIME,
    DEFAULT_SPEC,
    EXPERTS,
    LANGUAGE_COUNTS,
    SENTINEL_INTEGRATION_LANGUAGE,
    Paths,
    WorkflowError,
    _abs,
    _atomic_json,
    _file_sha256,
    _is_h100_resource,
    _job,
    _load_object,
    _paths_for_manifest,
    _resource_gpu_count,
)
from .estimates import (
    _performance_budget,
    _planned_checkpoint_count_for_job,
    _scheduled_training_tokens_per_second,
    _training_work_segments_for_job,
    _validate_training_throughput_schedule,
    _validation_tokens_by_k,
)
from .gate_evaluation import _validation_report_from_eval
from .training_health import _parse_sentinel_health
from .variant import active_policy


def _measured_sentinel_projection(
    manifest: Mapping[str, Any],
    sentinel_payloads: Mapping[str, Mapping[str, Any]],
    integration_results: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Project worst-case GPU hours from measured sentinel throughput."""

    factor = float(runtime.get("sentinel_throughput_safety_factor", 0.0))
    if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise WorkflowError("sentinel_throughput_safety_factor must be in (0, 1]")
    budget = _performance_budget(runtime)
    noncheckpoint_seconds = float(
        budget["training_noncheckpoint_overhead_seconds_per_job"]
    )
    checkpoint_budget = budget["checkpoint_overhead_seconds"]
    if not math.isfinite(noncheckpoint_seconds) or noncheckpoint_seconds < 0:
        raise WorkflowError("training non-checkpoint overhead must be finite and non-negative")
    if isinstance(manifest.get("output_root"), str):
        paths = _paths_for_manifest(manifest)
        validation_tokens_by_k, validation_language_counts, _english_tokens, validation_sha256 = (
            _validation_tokens_by_k(paths)
        )
    else:
        validation_tokens_by_k = {
            k: (200 if k == 1 else k) * 20_000_000 for k in LANGUAGE_COUNTS
        }
        validation_language_counts = {
            k: (200 if k == 1 else k) for k in LANGUAGE_COUNTS
        }
        validation_sha256 = "unmaterialized_validation_cap_upper_bound"
    measured: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    measured_by_probe: dict[str, float] = {}
    for source_job, payload in sentinel_payloads.items():
        arguments = payload.get("arguments")
        health = payload.get("sentinel_health")
        if not isinstance(arguments, dict) or not isinstance(health, dict):
            raise WorkflowError(f"{source_job} lacks arguments or sentinel health")
        ladder = str(arguments.get("ladder", ""))
        experts = int(arguments.get("experts", 0) or 0)
        metrics = health.get("metrics")
        if not isinstance(metrics, dict):
            raise WorkflowError(f"{source_job} lacks measured metrics")
        try:
            tokens_per_second = float(metrics["median_tokens_per_second"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"{source_job} lacks measured median token throughput"
            ) from exc
        if not math.isfinite(tokens_per_second) or tokens_per_second <= 0:
            raise WorkflowError(f"{source_job} has invalid measured token throughput")
        measured[(ladder, experts)].append(tokens_per_second)
        measured_by_probe[source_job] = tokens_per_second

    has_scheduled_throughput = isinstance(
        runtime.get("training_throughput_tokens_per_second_by_global_batch_size"),
        Mapping,
    ) and not bool(runtime.get("probe_only_unmeasured"))
    if has_scheduled_throughput:
        _validate_training_throughput_schedule(runtime)
    conservative: dict[tuple[str, int, int], float] = {}
    conservative_checkpoint_overhead: dict[tuple[str, int], float] = {}
    for ladder in active_policy().ladders:
        ladder_values = [
            value
            for (cell_ladder, _), values in measured.items()
            if cell_ladder == ladder
            for value in values
        ]
        if not ladder_values:
            raise WorkflowError(f"no measured sentinel throughput for ladder {ladder}")
        slowest = min(ladder_values)
        try:
            budgeted_checkpoint_seconds = float(checkpoint_budget[ladder])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"runtime lacks a checkpoint-overhead allowance for ladder {ladder}"
            ) from exc
        if not math.isfinite(budgeted_checkpoint_seconds) or budgeted_checkpoint_seconds < 0:
            raise WorkflowError(f"invalid checkpoint-overhead allowance for ladder {ladder}")
        for experts in EXPERTS:
            for global_batch_size in (32, 64, 128):
                direct = (
                    _scheduled_training_tokens_per_second(
                        runtime,
                        ladder=ladder,
                        experts=experts,
                        global_batch_size=global_batch_size,
                    )
                    if has_scheduled_throughput
                    else (
                        min(measured[(ladder, experts)])
                        if measured.get((ladder, experts))
                        else slowest
                    )
                )
                conservative[(ladder, experts, global_batch_size)] = factor * direct
            conservative_checkpoint_overhead[(ladder, experts)] = (
                budgeted_checkpoint_seconds / factor
            )

    base_training_hours = 0.0
    training_rows = 0
    for row in manifest["jobs"]:
        if not _is_h100_resource(row.get("resource")) or int(row.get("gpu_tokens", 0) or 0) <= 0:
            continue
        gpu_count = _resource_gpu_count(runtime, str(row["resource"]))
        arguments = row.get("arguments", {})
        ladder = str(arguments.get("ladder", ""))
        experts = int(arguments.get("experts", 0) or 0)
        segments = _training_work_segments_for_job(row, runtime)
        if not segments:
            raise WorkflowError(f"cannot map training work to {row['job_id']}")
        checkpoint_count = _planned_checkpoint_count_for_job(row, runtime)
        if checkpoint_count <= 0:
            raise WorkflowError(
                f"cannot count planned full-state checkpoints for {row['job_id']}"
            )
        forward_seconds = 0.0
        for segment in segments:
            global_batch_size = int(segment["global_batch_size"])
            tps = conservative.get((ladder, experts, global_batch_size))
            if tps is None:
                raise WorkflowError(
                    f"cannot map GBS{global_batch_size} throughput to {row['job_id']}"
                )
            forward_seconds += int(segment["tokens"]) / tps
        hours = (
            forward_seconds
            + checkpoint_count * conservative_checkpoint_overhead[(ladder, experts)]
        ) / 3600.0 * gpu_count
        training_rows += 1
        base_training_hours += hours
    control_plane_hours = sum(
        _resource_gpu_count(runtime, str(row["resource"]))
        * noncheckpoint_seconds
        / factor
        / 3600.0
        for row in manifest["jobs"]
        if _is_h100_resource(row.get("resource"))
        and int(row.get("gpu_tokens", 0) or 0) > 0
    )
    base_training_hours += control_plane_hours
    evaluation_throughput = runtime.get("evaluation_throughput_tokens_per_second")
    if not isinstance(evaluation_throughput, Mapping):
        raise WorkflowError("runtime lacks per-cell evaluator throughput evidence")
    measured_validation_tps: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    measured_setup_seconds: defaultdict[str, list[float]] = defaultdict(list)
    measured_residual_per_language_seconds: defaultdict[str, list[float]] = defaultdict(
        list
    )
    for eval_job_id, result in integration_results.items():
        source = _job(manifest, eval_job_id)
        ladder = str(source["arguments"]["ladder"])
        experts = int(source["arguments"].get("experts", 0) or 0)
        if experts not in EXPERTS:
            raise WorkflowError(
                f"{eval_job_id} lacks a supported expert count for evaluator timing"
            )
        measured_tps = float(result["tokens_per_second"])
        if not math.isfinite(measured_tps) or measured_tps <= 0:
            raise WorkflowError(f"{eval_job_id} has invalid evaluator throughput")
        measured_validation_tps[(ladder, experts)].append(measured_tps)
        measured_setup_seconds[ladder].append(float(result["setup_seconds"]))
        measured_residual_per_language_seconds[ladder].append(
            float(result["residual_per_language_seconds"])
        )
    conservative_validation_tps: dict[tuple[str, int], float] = {}
    conservative_setup_seconds: dict[str, float] = {}
    conservative_residual_per_language_seconds: dict[str, float] = {}
    for ladder in active_policy().ladders:
        if not measured_setup_seconds[ladder]:
            raise WorkflowError(
                f"no measured evaluator integration timing for ladder {ladder}"
            )
        try:
            configured_ladder_tps = evaluation_throughput[ladder]
            budgeted_setup_seconds = float(budget["evaluation_setup_seconds"][ladder])
            budgeted_residual_seconds = float(
                budget["evaluation_residual_seconds_per_language"][ladder]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"runtime lacks evaluator budget evidence for ladder {ladder}"
            ) from exc
        if not isinstance(configured_ladder_tps, Mapping):
            raise WorkflowError(
                f"runtime lacks per-expert evaluator throughput evidence for {ladder}"
            )
        if not math.isfinite(budgeted_setup_seconds) or budgeted_setup_seconds < 0:
            raise WorkflowError(f"invalid evaluator setup allowance for ladder {ladder}")
        if not math.isfinite(budgeted_residual_seconds) or budgeted_residual_seconds < 0:
            raise WorkflowError(
                f"invalid evaluator residual allowance for ladder {ladder}"
            )
        for experts in EXPERTS:
            try:
                configured_tps = float(configured_ladder_tps[str(experts)])
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkflowError(
                    f"runtime lacks evaluator throughput evidence for {ladder}/E{experts}"
                ) from exc
            if not math.isfinite(configured_tps) or configured_tps <= 0:
                raise WorkflowError(
                    f"invalid evaluator throughput allowance for {ladder}/E{experts}"
                )
            direct = measured_validation_tps.get((ladder, experts), [])
            conservative_validation_tps[(ladder, experts)] = factor * min(
                [configured_tps, *direct]
            )
        conservative_setup_seconds[ladder] = (
            max(budgeted_setup_seconds, max(measured_setup_seconds[ladder])) / factor
        )
        conservative_residual_per_language_seconds[ladder] = (
            max(
                budgeted_residual_seconds,
                max(measured_residual_per_language_seconds[ladder]),
            )
            / factor
        )

    base_validation_hours = 0.0
    validation_rows = 0
    for row in manifest["jobs"]:
        if row.get("action") != "per_language_eval":
            continue
        gpu_count = _resource_gpu_count(runtime, str(row["resource"]))
        arguments = row.get("arguments", {})
        ladder = str(arguments.get("ladder", ""))
        experts = int(arguments.get("experts", 0) or 0)
        if (ladder, experts) not in conservative_validation_tps:
            raise WorkflowError(
                f"cannot map measured evaluator throughput to {row['job_id']}"
            )
        if arguments.get("purpose") == "sentinel_integration":
            tokens = int(integration_results[str(row["job_id"])]["token_count"])
            language_count = 1
        else:
            tokens = validation_tokens_by_k[int(arguments["K"])]
            language_count = validation_language_counts[int(arguments["K"])]
        hours = (
            tokens / conservative_validation_tps[(ladder, experts)]
            + conservative_setup_seconds[ladder]
            + language_count * conservative_residual_per_language_seconds[ladder]
        ) / 3600.0 * gpu_count
        validation_rows += 1
        base_validation_hours += hours
    validation_upper = base_validation_hours
    total = base_training_hours + base_validation_hours
    raw_limit = runtime.get("gpu_hour_limit")
    limit = float(raw_limit) if raw_limit is not None else None
    return {
        "schema_version": 3,
        "method": "long_preflight_batch_schedule_tps_with_exact_sealed_validation_tokens",
        "mapping_rule": (
            "when the runtime supplies a long-preflight batch-rate matrix, charge every "
            "training segment at its matching GBS32/64/128 TPS; short sentinels remain "
            "a functional health gate and are recorded but are never extrapolated across "
            "the production token budget. Without that matrix, use the measured cell TPS "
            "when present or the slowest sentinel in the same ladder for compatibility. "
            "Validation uses the lower of each "
            "cell's fresh full-English TPS and its preflight benchmark TPS, plus the "
            "larger of the fresh or preflight setup/residual allowances; every forward TPS "
            "is multiplied by the safety factor, while fixed control overhead is divided by it"
        ),
        "safety_factor": factor,
        "measured_tokens_per_second": {
            f"{ladder}/E{experts}": sorted(values)
            for (ladder, experts), values in sorted(measured.items())
        },
        "measured_training_tokens_per_second_by_probe": dict(
            sorted(measured_by_probe.items())
        ),
        "checkpoint_overhead_evidence": dict(checkpoint_budget),
        "conservative_checkpoint_overhead_seconds": {
            f"{ladder}/E{experts}": value
            for (ladder, experts), value in sorted(
                conservative_checkpoint_overhead.items()
            )
        },
        "training_throughput_source": (
            "long_preflight_batch_schedule"
            if has_scheduled_throughput
            else "short_sentinel_compatibility_fallback"
        ),
        "conservative_tokens_per_second_by_global_batch_size": {
            f"GBS{global_batch_size}/{ladder}/E{experts}": value
            for (ladder, experts, global_batch_size), value in sorted(conservative.items())
        },
        "training_jobs_counted": training_rows,
        "training_control_plane_gpu_hours": control_plane_hours,
        "validation_jobs_counted": validation_rows,
        "base_training_gpu_hours": base_training_hours,
        "measured_validation_tokens_per_second": {
            f"{ladder}/E{experts}": sorted(values)
            for (ladder, experts), values in sorted(measured_validation_tps.items())
        },
        "conservative_validation_tokens_per_second": {
            f"{ladder}/E{experts}": value
            for (ladder, experts), value in sorted(conservative_validation_tps.items())
        },
        "measured_validation_setup_seconds": {
            ladder: sorted(values) for ladder, values in measured_setup_seconds.items()
        },
        "conservative_validation_setup_seconds": conservative_setup_seconds,
        "measured_validation_residual_per_language_seconds": {
            ladder: sorted(values)
            for ladder, values in measured_residual_per_language_seconds.items()
        },
        "conservative_validation_residual_per_language_seconds": (
            conservative_residual_per_language_seconds
        ),
        "base_validation_gpu_hours": base_validation_hours,
        "validation_exact_token_manifest_sha256": validation_sha256,
        "validation_exact_tokens_by_K": {
            str(k): value for k, value in sorted(validation_tokens_by_k.items())
        },
        "validation_upper_bound_gpu_hours": validation_upper,
        "projected_worst_case_gpu_hours": total,
        "gpu_hour_limit": limit,
        "budget_policy": (
            "hard_cap" if limit is not None else "informational_no_hard_cap"
        ),
        "status": "go" if limit is None or total <= limit else "hold",
    }


def _validate_sentinel_evaluation(path: Path, *, expect_moe: bool) -> dict[str, Any]:
    """Verify the dense/MoE real-prefix evaluator integration artifact."""

    payload, report, report_path = _validation_report_from_eval(path)
    errors: list[str] = []
    if payload.get("status") != "complete" or report.get("status") != "complete":
        errors.append("evaluation or report is not complete")
    expected_languages = [SENTINEL_INTEGRATION_LANGUAGE]
    coverage = payload.get("coverage", {})
    for key in ("trained_languages", "required_languages", "evaluated_languages"):
        if coverage.get(key) != expected_languages:
            errors.append(f"coverage.{key} is not exact English-only coverage")
    languages = payload.get("languages")
    descriptor = (
        languages.get(SENTINEL_INTEGRATION_LANGUAGE)
        if isinstance(languages, dict)
        else None
    )
    language_payload: dict[str, Any] = {}
    language_path = Path()
    if not isinstance(descriptor, dict):
        errors.append("evaluation manifest lacks the English language artifact")
    else:
        language_path = Path(str(descriptor.get("path", "")))
        if not language_path.is_absolute():
            language_path = path.parent / language_path
        language_path = language_path.resolve()
        if not language_path.is_file() or _file_sha256(language_path) != descriptor.get(
            "sha256"
        ):
            errors.append("English language artifact hash mismatch")
        else:
            language_payload = _load_object(language_path)

    if language_payload:
        try:
            token_count = int(language_payload["token_count"])
            backend_count = int(language_payload["backend"]["backend_token_count"])
            descriptor_count = int(descriptor["token_count"])
            nll_sum = float(language_payload["nll_sum"])
            raw_nll = float(language_payload["raw_nll"])
            adjusted_sum = float(language_payload["adjusted_nll_sum"])
            adjusted = float(language_payload["atlas_tao_unigram_adjusted_loss"])
            forward_seconds = float(language_payload["backend"]["forward_seconds"])
            tokens_per_second = float(
                language_payload["backend"]["scored_tokens_per_second"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid English sufficient statistics: {exc}")
        else:
            if (
                token_count <= 0
                or token_count != backend_count
                or token_count != descriptor_count
            ):
                errors.append(
                    "English token denominator is not exact across backend/language/manifest"
                )
            if (
                not math.isfinite(forward_seconds)
                or forward_seconds <= 0
                or not math.isfinite(tokens_per_second)
                or tokens_per_second <= 0
                or not math.isclose(
                    tokens_per_second,
                    token_count / forward_seconds,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                errors.append("English measured forward timing/throughput is invalid")
            for label, value in (
                ("nll_sum", nll_sum),
                ("raw_nll", raw_nll),
                ("adjusted_nll_sum", adjusted_sum),
                ("adjusted_loss", adjusted),
            ):
                if not math.isfinite(value):
                    errors.append(f"non-finite English {label}")
            if token_count > 0 and not math.isclose(
                raw_nll, nll_sum / token_count, rel_tol=1e-12, abs_tol=1e-12
            ):
                errors.append("raw NLL does not use the exact token denominator")
            if token_count > 0 and not math.isclose(
                adjusted, adjusted_sum / token_count, rel_tol=1e-12, abs_tol=1e-12
            ):
                errors.append("adjusted loss does not use the exact token denominator")

        routing = language_payload.get("moe_routing")
        if not isinstance(routing, dict):
            errors.append("English language artifact lacks routing diagnostics")
        elif expect_moe:
            if (
                routing.get("applicable") is not True
                or routing.get("evidence_available") is not True
            ):
                errors.append("MoE routing evidence is unavailable")
            dropped = routing.get("totals", {}).get("dropped_assignments")
            if not isinstance(dropped, int) or dropped != 0:
                errors.append(f"MoE dropped assignments is not exact zero: {dropped!r}")
        elif (
            routing.get("applicable") is not False
            or routing.get("evidence_available") is not True
        ):
            errors.append("dense routing diagnostic marker is invalid")

    aggregates = report.get("aggregates", {})
    for name in ("primary", "secondary"):
        aggregate = aggregates.get(name) if isinstance(aggregates, dict) else None
        try:
            value = float(aggregate["value"])
        except (KeyError, TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            errors.append(f"report {name} metric is non-finite")
        if (
            not isinstance(aggregate, dict)
            or aggregate.get("language_count") != 1
            or aggregate.get("languages") != expected_languages
        ):
            errors.append(f"report {name} aggregate is not exact English-only coverage")
    timing = payload.get("timing")
    try:
        setup_seconds = float(timing["setup_seconds"])
        total_forward_seconds = float(timing["forward_seconds"])
        internal_total_seconds = float(timing["internal_total_seconds"])
    except (KeyError, TypeError, ValueError):
        setup_seconds = total_forward_seconds = internal_total_seconds = math.nan
    if (
        not all(
            math.isfinite(value) and value > 0
            for value in (setup_seconds, total_forward_seconds, internal_total_seconds)
        )
        or internal_total_seconds < setup_seconds + total_forward_seconds
    ):
        errors.append("evaluation setup/forward timing is invalid")
    elif language_payload and not math.isclose(
        total_forward_seconds,
        float(language_payload["backend"]["forward_seconds"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        errors.append(
            "one-language forward timing differs between manifest and language artifact"
        )
    residual_per_language_seconds = (
        internal_total_seconds - setup_seconds - total_forward_seconds
    )
    if (
        not math.isfinite(residual_per_language_seconds)
        or residual_per_language_seconds < 0
    ):
        errors.append("evaluation residual per-language overhead is invalid")
    if errors:
        raise WorkflowError("; ".join(errors))
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "validation_report": {
            "path": str(report_path),
            "sha256": _file_sha256(report_path),
        },
        "language_artifact": {
            "path": str(language_path),
            "sha256": _file_sha256(language_path),
        },
        "raw_nll": float(aggregates["secondary"]["value"]),
        "atlas_tao_adjusted_loss": float(aggregates["primary"]["value"]),
        "token_count": int(language_payload["token_count"]),
        "setup_seconds": setup_seconds,
        "forward_seconds": total_forward_seconds,
        "tokens_per_second": float(
            language_payload["backend"]["scored_tokens_per_second"]
        ),
        "internal_total_seconds": internal_total_seconds,
        "residual_per_language_seconds": residual_per_language_seconds,
        "moe": expect_moe,
    }


def _run_sentinel_gate(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    paths: Paths,
    runtime: Mapping[str, Any] | None = None,
) -> None:
    from multilingual_megatron.pipeline.stages.training.checkpoint_state import (
        inspect_full_state_checkpoint,
    )

    runtime = runtime or _load_object(_abs(DEFAULT_RUNTIME))
    spec = _load_object(Path(str(manifest.get("spec_path", _abs(DEFAULT_SPEC)))))
    execution_policy = spec.get("execution_policy", {})
    default_long_cv_limit = float(
        execution_policy.get("long_probe_step_time_cv_limit_pct", 5.0)
    )
    long_cv_overrides = execution_policy.get(
        "long_probe_step_time_cv_limit_pct_overrides", {}
    )
    long_p99_overrides = execution_policy.get(
        "long_probe_p99_to_median_limit_overrides", {}
    )
    if (
        not math.isfinite(default_long_cv_limit)
        or default_long_cv_limit <= 0
        or not isinstance(long_cv_overrides, Mapping)
        or not isinstance(long_p99_overrides, Mapping)
    ):
        raise WorkflowError("sentinel long-probe stability policy is invalid")
    errors: list[str] = []
    results: dict[str, Any] = {}
    sentinel_payloads: dict[str, dict[str, Any]] = {}
    for source_job in job["arguments"]["sentinel_jobs"]:
        path = paths.jobs / source_job / "result.json"
        payload = _load_object(path)
        sentinel_payloads[source_job] = payload
        results[source_job] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "status": payload.get("status"),
        }
        if payload.get("status") != "complete":
            errors.append(f"{source_job} is not complete")
        expected_arguments = _job(manifest, source_job)["arguments"]
        if payload.get("arguments") != expected_arguments:
            errors.append(f"{source_job} result arguments differ from the frozen probe")
        health = payload.get("sentinel_health")
        if not isinstance(health, dict):
            errors.append(f"{source_job} lacks sentinel health")
        else:
            log_path = Path(str(health.get("log_path", "")))
            arguments = payload.get("arguments", {})
            reparsed = _parse_sentinel_health(
                log_path,
                require_moe=int(arguments.get("experts", 0) or 0) > 1
                if isinstance(arguments, dict)
                else False,
                throughput_window_size=(
                    32
                    if str(expected_arguments.get("sentinel_kind", "")).startswith(
                        "long_gbs"
                    )
                    else 12
                ),
            )
            # The Slurm controller owns the GPU monitor and the pipeline-owned
            # training log may be nested several directories below it. Rebind
            # the controller CSV explicitly during the independent gate pass
            # so GPU evidence is not lost when the gate runs in a CPU job.
            from multilingual_megatron.pipeline.stages.training.metrics import (
                gpu_metrics_for_log,
            )

            controller_gpu_metrics = gpu_metrics_for_log(
                paths.root / "logs" / f"{source_job}.slurm.log"
            )
            if controller_gpu_metrics:
                reparsed.setdefault("metrics", {}).update(controller_gpu_metrics)
            results[source_job]["sentinel_health"] = reparsed
            if health.get("log_sha256") != reparsed.get("log_sha256"):
                errors.append(f"{source_job} training log hash changed")
            if reparsed.get("status") != "ok":
                errors.append(f"{source_job} health failed: {reparsed.get('errors')}")
            observed_metrics = reparsed.get("metrics", {})
            observed_gbs = observed_metrics.get("global_batch_size")
            if observed_gbs != expected_arguments.get("global_batch_size"):
                errors.append(
                    f"{source_job} observed GBS {observed_gbs!r}, expected "
                    f"{expected_arguments.get('global_batch_size')!r}"
                )
            if str(expected_arguments.get("sentinel_kind", "")).startswith(
                "long_gbs"
            ):
                try:
                    steps_counted = int(observed_metrics.get("steps_counted", 0))
                    step_cv_pct = float(
                        observed_metrics.get("step_time_cv_pct", math.inf)
                    )
                    median_step_ms = float(
                        observed_metrics.get("median_step_ms", math.inf)
                    )
                    p99_step_ms = float(
                        observed_metrics.get("p99_step_ms", math.inf)
                    )
                    max_reserved_gb = float(
                        observed_metrics.get("max_reserved_gb", math.inf)
                    )
                    max_vram_used_pct = float(
                        observed_metrics.get("max_vram_used_pct", math.inf)
                    )
                except (TypeError, ValueError):
                    steps_counted = 0
                    step_cv_pct = math.inf
                    median_step_ms = math.inf
                    p99_step_ms = math.inf
                    max_reserved_gb = math.inf
                    max_vram_used_pct = math.inf
                if steps_counted < 32:
                    errors.append(f"{source_job} has fewer than 32 timed iterations")
                cv_limit = float(
                    long_cv_overrides.get(source_job, default_long_cv_limit)
                )
                if (
                    not math.isfinite(cv_limit)
                    or cv_limit <= 0
                    or not math.isfinite(step_cv_pct)
                    or step_cv_pct >= cv_limit
                ):
                    errors.append(
                        f"{source_job} step-time CV is not below {cv_limit:g}%"
                    )
                p99_to_median_limit = long_p99_overrides.get(source_job)
                if p99_to_median_limit is not None:
                    p99_to_median_limit = float(p99_to_median_limit)
                    ratio = (
                        p99_step_ms / median_step_ms
                        if math.isfinite(median_step_ms) and median_step_ms > 0
                        else math.inf
                    )
                    if (
                        not math.isfinite(p99_to_median_limit)
                        or p99_to_median_limit <= 1.0
                        or not math.isfinite(ratio)
                        or ratio >= p99_to_median_limit
                    ):
                        errors.append(
                            f"{source_job} p99/median step-time ratio is not below "
                            f"{p99_to_median_limit:g}"
                        )
                if not math.isfinite(max_reserved_gb) or max_reserved_gb > 84.0:
                    errors.append(f"{source_job} reserved memory exceeds 84 GB")
                if not math.isfinite(max_vram_used_pct) or max_vram_used_pct > 90.0:
                    errors.append(f"{source_job} observed VRAM exceeds 90%")
        checkpoint_text = str(payload.get("final_checkpoint_root", "")).strip()
        step = int(payload.get("final_checkpoint_step", 0) or 0)
        samples = int(payload.get("expected_consumed_samples", 0) or 0)
        if not checkpoint_text or step <= 0 or samples <= 0:
            errors.append(
                f"{source_job} lacks exact checkpoint root/step/sample metadata"
            )
            continue
        state = inspect_full_state_checkpoint(
            checkpoint_root=Path(checkpoint_text),
            expected_step=step,
            expected_consumed_samples=samples,
        )
        results[source_job]["checkpoint_state"] = state.to_dict()
        if not state.valid_full_state:
            errors.append(
                f"{source_job} has invalid full-state checkpoint: {state.errors}"
            )

    integration_results: dict[str, Any] = {}
    for eval_job_id in job["arguments"].get("integration_evaluation_jobs", []):
        source = _job(manifest, eval_job_id)
        path = paths.jobs / eval_job_id / "evaluation_manifest.json"
        try:
            integration_results[eval_job_id] = _validate_sentinel_evaluation(
                path,
                expect_moe=int(source["arguments"]["experts"]) > 1,
            )
        except (OSError, WorkflowError, ValueError) as exc:
            errors.append(f"{eval_job_id} integration validation failed: {exc}")

    projection: dict[str, Any] = {}
    try:
        projection = _measured_sentinel_projection(
            manifest,
            sentinel_payloads,
            integration_results,
            runtime,
        )
        if projection["status"] != "go":
            errors.append(
                "measured sentinel GPU-hour projection exceeds the frozen limit: "
                f"{projection['projected_worst_case_gpu_hours']} > {projection['gpu_hour_limit']}"
            )
    except (KeyError, TypeError, ValueError, WorkflowError) as exc:
        errors.append(f"could not compute measured sentinel GPU-hour projection: {exc}")

    report = {
        "schema_version": 1,
        "status": "failed" if errors else "ok",
        "errors": errors,
        "sentinels": results,
        "integration_evaluations": integration_results,
        "throughput_projection": projection,
    }
    _atomic_json(paths.control / "sentinel_gate.json", report)
    if errors:
        raise WorkflowError("sentinel gate failed: " + "; ".join(errors))
