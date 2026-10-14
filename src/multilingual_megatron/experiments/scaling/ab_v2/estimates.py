"""Own A/B-v2 batch, throughput, GPU-hour, and storage estimates."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .artifact_data import _languages_from_map
from .common import (
    EXPERTS,
    LANGUAGE_COUNTS,
    Paths,
    SENTINEL_INTEGRATION_LANGUAGE,
    WorkflowError,
    _abs,
    _file_sha256,
    _is_h100_resource,
    _language_map,
    _load_object,
    _payload_sha256,
    _resource_gpu_count,
)
from .variant import active_policy


def _validation_tokens_by_k(
    paths: Paths,
) -> tuple[dict[int, int], dict[int, int], int, str]:
    """Read exact scored-token panels from the sealed validation manifest."""

    validation_manifest = paths.tokenized / "validation_megatron_data_manifest.json"
    if not validation_manifest.is_file():
        # Planning a truly fresh corpus necessarily precedes tokenization. It
        # retains the historical cap-only estimate; production sealed-corpus
        # reuse is required to resolve this to exact materialized tokens.
        tokens = {k: (200 if k == 1 else k) * 20_000_000 for k in LANGUAGE_COUNTS}
        counts = {k: (200 if k == 1 else k) for k in LANGUAGE_COUNTS}
        return tokens, counts, 20_000_000, "unmaterialized_validation_cap_upper_bound"
    payload = _load_object(validation_manifest)
    try:
        descriptors = payload["splits"]["validation"]["prefix_artifacts_by_language"]
    except (KeyError, TypeError) as exc:
        raise WorkflowError("sealed validation manifest lacks language prefix descriptors") from exc
    if not isinstance(descriptors, Mapping):
        raise WorkflowError("sealed validation language descriptors must be an object")

    language_tokens: dict[str, int] = {}
    for language, raw_rows in descriptors.items():
        if not isinstance(raw_rows, list) or not raw_rows:
            raise WorkflowError(f"sealed validation set has no prefix rows for {language}")
        try:
            language_tokens[str(language)] = sum(
                int(row["tokens_including_eod"]) for row in raw_rows
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"sealed validation prefix rows lack exact token counts for {language}"
            ) from exc

    tokens_by_k: dict[int, int] = {}
    language_counts: dict[int, int] = {}
    for k in LANGUAGE_COUNTS:
        languages = _languages_from_map(_abs(_language_map(200 if k == 1 else k)))
        missing = [language for language in languages if language not in language_tokens]
        if missing:
            raise WorkflowError(
                f"sealed validation data is missing K{k} languages: {missing[:4]}"
            )
        tokens_by_k[k] = sum(language_tokens[language] for language in languages)
        language_counts[k] = len(languages)
    try:
        english_tokens = language_tokens[SENTINEL_INTEGRATION_LANGUAGE]
    except KeyError as exc:
        raise WorkflowError("sealed validation set lacks the English integration panel") from exc
    return tokens_by_k, language_counts, english_tokens, _file_sha256(validation_manifest)


def _performance_budget(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    budget = runtime.get("performance_budget")
    if not isinstance(budget, Mapping):
        # Compatibility for focused pure-function tests.  build_manifest
        # requires the explicit evidence-backed object for real launches.
        return {
            "training_noncheckpoint_overhead_seconds_per_job": 0.0,
            "checkpoint_overhead_seconds": {"A": 0.0, "B": 0.0},
            "evaluation_setup_seconds": {"A": 0.0, "B": 0.0},
            "evaluation_residual_seconds_per_language": {"A": 0.0, "B": 0.0},
        }
    required = {
        "training_noncheckpoint_overhead_seconds_per_job",
        "checkpoint_overhead_seconds",
        "evaluation_setup_seconds",
        "evaluation_residual_seconds_per_language",
    }
    if set(budget) < required:
        raise WorkflowError("performance_budget lacks a required component")
    return budget


def _training_data_parallel_size(
    runtime: Mapping[str, Any], *, ladder: str, experts: int
) -> int:
    """Return the explicit effective DP degree for one execution cell.

    A/B uses a single GPU for every cell.  Later ladders can introduce
    expert-parallel ranks, where the global batch still spans the ordinary
    data-parallel group.  Keeping this in the shared ramp machinery makes the
    planned MBS/GA, segment receipt, and cost evidence agree with Megatron.
    """

    configured = runtime.get("training", {}).get(
        "data_parallel_size_by_ladder_and_experts"
    )
    if configured is None:
        return 1
    if not isinstance(configured, Mapping):
        raise WorkflowError(
            "training.data_parallel_size_by_ladder_and_experts must be an object"
        )
    try:
        value = int(configured[ladder][str(experts)])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            f"runtime lacks data-parallel degree for {ladder}/E{experts}"
        ) from exc
    if value <= 0:
        raise WorkflowError(f"invalid data-parallel degree for {ladder}/E{experts}")
    return value


def _batch_layout(
    runtime: Mapping[str, Any],
    *,
    ladder: str,
    experts: int,
    global_batch_size: int,
) -> tuple[int, int, int]:
    """Resolve the exact MBS/DP/GA tuple for one global-batch segment."""

    from multilingual_megatron.pipeline.stages.training.batch_policy import (
        derive_segment_micro_batch_size,
    )

    try:
        plateau_mbs = int(runtime["training"]["micro_batch_size"][ladder][str(experts)])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(f"runtime lacks plateau MBS for {ladder}/E{experts}") from exc
    data_parallel_size = _training_data_parallel_size(
        runtime, ladder=ladder, experts=experts
    )
    micro_batch_size = derive_segment_micro_batch_size(
        plateau_micro_batch_size=plateau_mbs,
        global_batch_size=global_batch_size,
        data_parallel_size=data_parallel_size,
    )
    divisor = micro_batch_size * data_parallel_size
    if global_batch_size % divisor:
        raise WorkflowError(
            f"GBS {global_batch_size} is not divisible by MBS {micro_batch_size} and DP {data_parallel_size}"
        )
    return micro_batch_size, data_parallel_size, global_batch_size // divisor


def _canonical_batch_ramp_policy(
    *, horizons: Sequence[int], data_parallel_size: int = 1
) -> dict[str, Any]:
    """Return the one production batch policy used by jobs and cost gates.

    Keeping this policy construction in one place prevents a budget estimate
    from silently treating the first 1B-token ramp as a GBS128-only run.
    """

    return {
        "enabled": True,
        "mode": "token_ramp",
        "planning_method": "joint_moe_full_state_wsd_v2",
        "target_global_batch_size": 128,
        "schedule": [
            {"start_tokens": 0, "end_tokens": 500_000_000, "global_batch_size": 32},
            {
                "start_tokens": 500_000_000,
                "end_tokens": 1_000_000_000,
                "global_batch_size": 64,
            },
            {"start_tokens": 1_000_000_000, "end_tokens": 0, "global_batch_size": 128},
        ],
        "warmup_tokens": 130_000_000,
        "decay_fraction": 0.2,
        "dataset_horizon_tokens": int(max(horizons)),
        "checkpoint_horizons_tokens": [int(value) for value in horizons],
        "plan_role": "trunk",
        "seq_length": 2048,
        "data_parallel_size": int(data_parallel_size),
    }


def _job_horizons(
    arguments: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[int]:
    try:
        ladder = str(arguments["ladder"])
        raw = arguments.get(
            "checkpoint_horizons_tokens",
            runtime["training"]["checkpoint_horizons_tokens"][ladder],
        )
        horizons = [int(value) for value in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError("training job lacks checkpoint horizons") from exc
    if (
        not horizons
        or horizons != sorted(set(horizons))
        or any(value <= 0 for value in horizons)
    ):
        raise WorkflowError(f"invalid checkpoint horizons for {ladder}: {horizons}")
    return horizons


def _training_work_segments_for_job(
    row: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, int]]:
    """Split an H100 training job into the exact planned batch-rate segments."""

    tokens = int(row.get("gpu_tokens", 0) or 0)
    if tokens <= 0:
        return []
    action = str(row.get("action", ""))
    arguments = row.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise WorkflowError(f"{row.get('job_id')} has non-object arguments")
    if action != "production_train":
        global_batch_size = int(arguments.get("global_batch_size", 128))
        if global_batch_size <= 0:
            raise WorkflowError(f"{row.get('job_id')} has an invalid global batch size")
        return [{"global_batch_size": global_batch_size, "tokens": tokens}]

    try:
        ladder = str(arguments["ladder"])
        experts = int(arguments["experts"])
        role = str(arguments["role"])
        d_tokens = int(arguments["D"])
        horizons = _job_horizons(arguments, runtime)
        plateau_mbs = int(runtime["training"]["micro_batch_size"][ladder][str(experts)])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            f"{row.get('job_id')} lacks the production batch-ramp contract"
        ) from exc
    if role not in {"trunk", "branch"}:
        raise WorkflowError(f"{row.get('job_id')} has unsupported production role {role!r}")
    data_parallel_size = _training_data_parallel_size(
        runtime, ladder=ladder, experts=experts
    )

    from multilingual_megatron.pipeline.stages.training.batch_policy import (
        build_joint_moe_full_state_wsd_v2_plan,
    )

    plan = build_joint_moe_full_state_wsd_v2_plan(
        policy=_canonical_batch_ramp_policy(
            horizons=horizons, data_parallel_size=data_parallel_size
        ),
        plateau_micro_batch_size=plateau_mbs,
        data_parallel_size=data_parallel_size,
    )
    if role == "trunk":
        source_segments = plan["trunk"]["segments"]
    else:
        branch = next(
            (
                item
                for item in plan["branches"]
                if int(item["target_tokens"]) == d_tokens
            ),
            None,
        )
        if branch is None:
            raise WorkflowError(
                f"{row.get('job_id')} branch D={d_tokens} is absent from its endpoint plan"
            )
        source_segments = branch["segments"]
    segments = [
        {
            "global_batch_size": int(segment["global_batch_size"]),
            "tokens": int(segment["target_end_tokens"])
            - int(segment["target_start_tokens"]),
        }
        for segment in source_segments
    ]
    if any(segment["tokens"] <= 0 for segment in segments) or sum(
        segment["tokens"] for segment in segments
    ) != tokens:
        raise WorkflowError(
            f"{row.get('job_id')} batch segments do not exactly cover declared gpu_tokens"
        )
    return segments


def _scheduled_training_tokens_per_second(
    runtime: Mapping[str, Any], *, ladder: str, experts: int, global_batch_size: int
) -> float:
    """Read one measured throughput value from the frozen batch-rate matrix."""

    matrix = runtime.get("training_throughput_tokens_per_second_by_global_batch_size")
    if not isinstance(matrix, Mapping):
        raise WorkflowError(
            "production runtime lacks measured per-global-batch training throughput"
        )
    try:
        value = float(matrix[str(int(global_batch_size))][ladder][str(int(experts))])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError(
            "runtime lacks measured training throughput for "
            f"GBS{global_batch_size} {ladder}/E{experts}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise WorkflowError(
            f"invalid measured training throughput for GBS{global_batch_size} {ladder}/E{experts}"
        )
    return value


def _validate_training_throughput_schedule(runtime: Mapping[str, Any]) -> None:
    """Fail closed unless every rate used by the frozen production ramp exists."""

    for global_batch_size in (32, 64, 128):
        for ladder in active_policy().ladders:
            for experts in EXPERTS:
                _scheduled_training_tokens_per_second(
                    runtime,
                    ladder=ladder,
                    experts=experts,
                    global_batch_size=global_batch_size,
                )


def _verify_training_throughput_evidence(runtime: Mapping[str, Any]) -> None:
    """Bind the frozen schedule-rate matrix to complete long-run artifacts."""

    active_cells = {
        (ladder, experts)
        for ladder in active_policy().ladders
        for experts in EXPERTS
    }

    evidence = _performance_budget(runtime).get("training_throughput_evidence")
    if not isinstance(evidence, Mapping):
        raise WorkflowError("performance budget lacks training throughput evidence")

    def load_report(*, label: str) -> Mapping[str, Any]:
        descriptor = evidence.get(label)
        if not isinstance(descriptor, Mapping):
            raise WorkflowError(f"training throughput evidence lacks {label}")
        try:
            report_path = _abs(str(descriptor["report"]))
            expected_file_hash = str(descriptor["file_sha256"])
            expected_payload_hash = str(descriptor["payload_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"{label} throughput evidence lacks report identities") from exc
        if not report_path.is_file() or _file_sha256(report_path) != expected_file_hash:
            raise WorkflowError(f"{label} long-throughput report identity mismatch")
        report = _load_object(report_path)
        if (
            report.get("status") != "complete"
            or report.get("payload_sha256") != expected_payload_hash
        ):
            raise WorkflowError(f"{label} long-throughput report is not complete and pinned")
        candidate = dict(report)
        candidate.pop("payload_sha256", None)
        if _payload_sha256(candidate) != expected_payload_hash:
            raise WorkflowError(f"{label} long-throughput report payload hash mismatch")
        return report

    gbs32_report = load_report(label="GBS32")
    gbs128_report = load_report(label="GBS128")
    gbs64_evidence = evidence.get("GBS64")
    if not isinstance(gbs64_evidence, Mapping):
        raise WorkflowError("training throughput evidence lacks GBS64 rationale")
    # A later ladder may measure every production batch point directly.  Keep
    # that stronger evidence path distinct from A/B's deliberately
    # conservative GBS64 rule, which only benchmarked B/E32 at GA=2.
    if "report" in gbs64_evidence:
        gbs64_report = load_report(label="GBS64")

        def direct_rows_by_cell(
            report: Mapping[str, Any], *, global_batch_size: int
        ) -> dict[tuple[str, int], float]:
            values: dict[tuple[str, int], float] = {}
            rows = report.get("rows")
            if not isinstance(rows, list):
                raise WorkflowError("long-throughput report lacks row records")
            for row in rows:
                if not isinstance(row, Mapping) or int(row.get("GBS", 0) or 0) != global_batch_size:
                    continue
                try:
                    key = (str(row["ladder"]), int(row["experts"]))
                    value = float(row["median_tokens_per_second"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise WorkflowError("long-throughput report contains an invalid cell") from exc
                if row.get("status") != "complete" or not math.isfinite(value) or value <= 0:
                    raise WorkflowError("long-throughput report contains a non-complete cell")
                if key in active_cells:
                    values[key] = value
            return values

        expected_cells = active_cells
        reports = {
            32: direct_rows_by_cell(gbs32_report, global_batch_size=32),
            64: direct_rows_by_cell(gbs64_report, global_batch_size=64),
            128: direct_rows_by_cell(gbs128_report, global_batch_size=128),
        }
        if any(set(rows) != expected_cells for rows in reports.values()):
            raise WorkflowError("direct long-throughput reports do not cover every production cell")
        for global_batch_size, rows in reports.items():
            for ladder, experts in sorted(expected_cells):
                measured = _scheduled_training_tokens_per_second(
                    runtime,
                    ladder=ladder,
                    experts=experts,
                    global_batch_size=global_batch_size,
                )
                if not math.isclose(
                    measured, rows[(ladder, experts)], rel_tol=0.0, abs_tol=1e-9
                ):
                    raise WorkflowError(
                        f"GBS{global_batch_size} rate does not match direct long evidence "
                        f"for {ladder}/E{experts}"
                    )
        return
    if _abs(str(gbs64_evidence.get("B/E32_report", ""))) != _abs(
        str(evidence["GBS128"].get("report", ""))
    ):
        raise WorkflowError("GBS64 B/E32 evidence must reference the pinned long report")

    def rows_by_cell(report: Mapping[str, Any], *, global_batch_size: int) -> dict[tuple[str, int], float]:
        values: dict[tuple[str, int], float] = {}
        rows = report.get("rows")
        if not isinstance(rows, list):
            raise WorkflowError("long-throughput report lacks row records")
        for row in rows:
            if not isinstance(row, Mapping) or int(row.get("GBS", 0) or 0) != global_batch_size:
                continue
            try:
                key = (str(row["ladder"]), int(row["experts"]))
                value = float(row["median_tokens_per_second"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WorkflowError("long-throughput report contains an invalid cell") from exc
            if row.get("status") != "complete" or not math.isfinite(value) or value <= 0:
                raise WorkflowError("long-throughput report contains a non-complete cell")
            if key in active_cells:
                values[key] = value
        return values

    gbs32_cells = rows_by_cell(gbs32_report, global_batch_size=32)
    gbs128_cells = rows_by_cell(gbs128_report, global_batch_size=128)
    gbs64_cells = rows_by_cell(gbs128_report, global_batch_size=64)
    expected_cells = active_cells
    if set(gbs32_cells) != expected_cells or set(gbs128_cells) != expected_cells:
        raise WorkflowError("long-throughput reports do not cover all production cells")
    if set(gbs64_cells) != {("B", 32)}:
        raise WorkflowError("pinned GBS64 report must contain only the direct B/E32 probe")

    for ladder, experts in sorted(expected_cells):
        measured32 = _scheduled_training_tokens_per_second(
            runtime, ladder=ladder, experts=experts, global_batch_size=32
        )
        if not math.isclose(measured32, gbs32_cells[(ladder, experts)], rel_tol=0.0, abs_tol=1e-9):
            raise WorkflowError(f"GBS32 rate does not match its long evidence for {ladder}/E{experts}")
        rate128 = _scheduled_training_tokens_per_second(
            runtime, ladder=ladder, experts=experts, global_batch_size=128
        )
        rate64 = _scheduled_training_tokens_per_second(
            runtime, ladder=ladder, experts=experts, global_batch_size=64
        )
        if rate128 > gbs128_cells[(ladder, experts)]:
            raise WorkflowError(f"GBS128 rate exceeds its long evidence for {ladder}/E{experts}")
        if (ladder, experts) == ("B", 32):
            if not math.isclose(rate64, gbs64_cells[(ladder, experts)], rel_tol=0.0, abs_tol=1e-9):
                raise WorkflowError("GBS64 B/E32 rate does not match its direct long evidence")
        elif rate64 > rate128:
            raise WorkflowError(f"GBS64 rate is not conservative for {ladder}/E{experts}")


def _gpu_estimate(
    jobs: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any], paths: Paths
) -> dict[str, Any]:
    """Fail closed on a measured, exact-token, end-to-end H100 forecast."""

    evaluation_throughput = runtime.get("evaluation_throughput_tokens_per_second")
    if not isinstance(evaluation_throughput, Mapping):
        raise WorkflowError("runtime lacks evaluator throughput evidence")
    budget = _performance_budget(runtime)
    checkpoint_overhead = budget["checkpoint_overhead_seconds"]
    evaluation_setup = budget["evaluation_setup_seconds"]
    evaluation_residual = budget["evaluation_residual_seconds_per_language"]
    noncheckpoint_seconds = float(
        budget["training_noncheckpoint_overhead_seconds_per_job"]
    )
    if not math.isfinite(noncheckpoint_seconds) or noncheckpoint_seconds < 0:
        raise WorkflowError("training non-checkpoint overhead must be finite and non-negative")
    validation_tokens_by_k, validation_language_counts, english_tokens, validation_sha256 = (
        _validation_tokens_by_k(paths)
    )

    by_component: defaultdict[str, float] = defaultdict(float)
    rows: list[dict[str, Any]] = []
    training_jobs = 0
    for row in jobs:
        if not _is_h100_resource(row["resource"]) or int(row.get("gpu_tokens", 0)) <= 0:
            continue
        gpu_count = _resource_gpu_count(runtime, str(row["resource"]))
        arguments = row.get("arguments", {})
        ladder = str(arguments.get("ladder", ""))
        experts = str(arguments.get("experts", ""))
        if ladder not in checkpoint_overhead:
            raise WorkflowError(f"missing checkpoint overhead for {ladder}")
        segments = _training_work_segments_for_job(row, runtime)
        if not segments:
            raise WorkflowError(f"missing training work segments for {row['job_id']}")
        segment_estimates: list[dict[str, Any]] = []
        forward_hours = 0.0
        for segment in segments:
            global_batch_size = int(segment["global_batch_size"])
            tokens_per_second = _scheduled_training_tokens_per_second(
                runtime,
                ladder=ladder,
                experts=int(experts),
                global_batch_size=global_batch_size,
            )
            wall_hours = int(segment["tokens"]) / tokens_per_second / 3600.0
            segment_hours = wall_hours * gpu_count
            forward_hours += segment_hours
            segment_estimates.append(
                {
                    "global_batch_size": global_batch_size,
                    "tokens": int(segment["tokens"]),
                    "aggregate_tokens_per_second": tokens_per_second,
                    "gpu_count": gpu_count,
                    "wall_hours": wall_hours,
                    "forward_gpu_hours": segment_hours,
                }
            )
        checkpoint_count = _planned_checkpoint_count_for_job(row, runtime)
        checkpoint_hours = (
            checkpoint_count * float(checkpoint_overhead[ladder]) / 3600.0 * gpu_count
        )
        control_hours = noncheckpoint_seconds / 3600.0 * gpu_count
        by_component["training_forward"] += forward_hours
        by_component["training_checkpoint"] += checkpoint_hours
        by_component["training_control_plane"] += control_hours
        training_jobs += 1
        rows.append(
            {
                "job_id": row["job_id"],
                "stage": row["stage"],
                "resource": row["resource"],
                "gpu_count": gpu_count,
                "tokens": sum(segment["tokens"] for segment in segments),
                "batch_rate_segments": segment_estimates,
                "checkpoint_count": checkpoint_count,
                "forward_gpu_hours": forward_hours,
                "checkpoint_gpu_hours": checkpoint_hours,
                "control_plane_gpu_hours": control_hours,
                "predicted_gpu_hours": (
                    forward_hours + checkpoint_hours + control_hours
                ),
            }
        )

    production_eval_tokens = 0
    diagnostic_eval_tokens = 0
    sentinel_eval_tokens = 0
    evaluation_jobs = 0
    for row in jobs:
        if row.get("action") != "per_language_eval":
            continue
        arguments = row.get("arguments", {})
        ladder = str(arguments.get("ladder", ""))
        experts = str(arguments.get("experts", ""))
        if ladder not in evaluation_throughput or experts not in evaluation_throughput[ladder]:
            raise WorkflowError(f"missing evaluator throughput for {row['job_id']}")
        if ladder not in evaluation_setup or ladder not in evaluation_residual:
            raise WorkflowError(f"missing evaluator overhead evidence for {ladder}")
        integration = arguments.get("purpose") == "sentinel_integration"
        gpu_count = _resource_gpu_count(runtime, str(row["resource"]))
        k = int(arguments.get("K", 0) or 0)
        tokens = english_tokens if integration else validation_tokens_by_k[k]
        language_count = 1 if integration else validation_language_counts[k]
        forward_hours = (
            tokens
            / float(evaluation_throughput[ladder][experts])
            / 3600.0
            * gpu_count
        )
        overhead_hours = (
            float(evaluation_setup[ladder])
            + language_count * float(evaluation_residual[ladder])
        ) / 3600.0 * gpu_count
        by_component["validation_forward"] += forward_hours
        by_component["validation_overhead"] += overhead_hours
        rows.append(
            {
                "job_id": row["job_id"],
                "stage": row["stage"],
                "resource": row["resource"],
                "action": "per_language_eval",
                "gpu_count": gpu_count,
                "tokens": tokens,
                "language_count": language_count,
                "aggregate_tokens_per_second": float(
                    evaluation_throughput[ladder][experts]
                ),
                "forward_gpu_hours": forward_hours,
                "evaluation_overhead_gpu_hours": overhead_hours,
                "predicted_gpu_hours": forward_hours + overhead_hours,
            }
        )
        if integration:
            sentinel_eval_tokens += tokens
        elif arguments.get("purpose") == "lr_diagnostic":
            diagnostic_eval_tokens += tokens
        else:
            production_eval_tokens += tokens
        evaluation_jobs += 1

    base_hours = sum(by_component.values())
    raw_limit = runtime.get("gpu_hour_limit")
    limit = float(raw_limit) if raw_limit is not None else None
    return {
        "schema_version": 3,
        "method": "sealed_exact_validation_tokens_plus_batch-schedule-aware_verified_h100_end_to_end_overheads",
        "role": "informational ordinary-queue forecast; measured sentinel gate recomputes it",
        "base_estimated_gpu_hours": base_hours,
        "worst_case_estimated_gpu_hours": base_hours,
        "gpu_hour_limit": limit,
        "budget_policy": (
            "hard_cap" if limit is not None else "informational_no_hard_cap"
        ),
        "go_no_go": "go" if limit is None or base_hours <= limit else "hold",
        "by_component_gpu_hours": dict(sorted(by_component.items())),
        "training_jobs_counted": training_jobs,
        "evaluation_jobs_counted": evaluation_jobs,
        "validation_exact_token_manifest_sha256": validation_sha256,
        "validation_exact_tokens_by_K": {
            str(k): value for k, value in sorted(validation_tokens_by_k.items())
        },
        "validation_language_counts_by_K": {
            str(k): value for k, value in sorted(validation_language_counts.items())
        },
        "validation_tokens": {
            "production": production_eval_tokens,
            "lr_diagnostics": diagnostic_eval_tokens,
            "sentinel_integration": sentinel_eval_tokens,
        },
        "job_estimates": rows,
    }


def _model_parameter_upper_bound(ladder: str, experts: int) -> int:
    shape = {
        "A": (8, 512, 1536),
        "B": (12, 768, 2304),
        "C": (16, 1024, 3072),
        "D": (20, 1280, 3840),
    }.get(ladder)
    if shape is None or experts not in EXPERTS:
        raise WorkflowError(
            f"unknown model cell for storage estimate: {ladder}/E{experts}"
        )
    layers, hidden, ffn = shape
    # Untied input/output embeddings, attention projections, SwiGLU expert
    # matrices, router weights, and a small explicit norm/bias allowance.
    embeddings = 2 * 64_000 * hidden
    per_layer = 4 * hidden * hidden + max(1, experts) * 3 * hidden * ffn
    router_and_norms = layers * (hidden * max(1, experts) + 8 * hidden)
    return int(embeddings + layers * per_layer + router_and_norms)


def _full_state_checkpoint_count(
    runtime: Mapping[str, Any],
    *,
    ladder: str,
    experts: int,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Count real segment-end checkpoints from the production endpoint plan."""

    from multilingual_megatron.pipeline.stages.training.batch_policy import (
        build_joint_moe_full_state_wsd_v2_plan,
    )

    data_parallel_size = _training_data_parallel_size(
        runtime, ladder=ladder, experts=experts
    )
    policy = _canonical_batch_ramp_policy(
        horizons=horizons, data_parallel_size=data_parallel_size
    )
    plan = build_joint_moe_full_state_wsd_v2_plan(
        policy=policy,
        plateau_micro_batch_size=int(
            runtime["training"]["micro_batch_size"][ladder][str(experts)]
        ),
        data_parallel_size=data_parallel_size,
    )
    trunk = len(plan["trunk"]["segments"])
    branches = {
        str(int(row["target_tokens"])): len(row["segments"]) for row in plan["branches"]
    }
    return {
        "trunk": trunk,
        "branches": branches,
        "total": trunk + sum(branches.values()),
    }


def _expected_wsd_branch_endpoint(
    runtime: Mapping[str, Any],
    *,
    ladder: str,
    experts: int,
    nominal_horizon_tokens: int,
) -> dict[str, int]:
    """Resolve the one exact batch-aligned endpoint for a nominal WSD horizon."""

    from multilingual_megatron.pipeline.stages.training.batch_policy import (
        build_joint_moe_full_state_wsd_v2_plan,
    )

    nominal_horizon_tokens = int(nominal_horizon_tokens)
    data_parallel_size = _training_data_parallel_size(
        runtime, ladder=ladder, experts=experts
    )
    plan = build_joint_moe_full_state_wsd_v2_plan(
        policy=_canonical_batch_ramp_policy(
            horizons=[nominal_horizon_tokens],
            data_parallel_size=data_parallel_size,
        ),
        plateau_micro_batch_size=int(
            runtime["training"]["micro_batch_size"][ladder][str(experts)]
        ),
        data_parallel_size=data_parallel_size,
    )
    branches = [
        row
        for row in plan.get("branches", [])
        if int(row.get("target_tokens", 0)) == nominal_horizon_tokens
    ]
    if len(branches) != 1:
        raise WorkflowError(
            f"could not resolve one WSD endpoint for {ladder}/E{experts}/D{nominal_horizon_tokens}"
        )
    branch = branches[0]
    endpoint = {
        "nominal_horizon_tokens": nominal_horizon_tokens,
        "actual_end_tokens": int(branch["actual_end_tokens"]),
        "expected_consumed_samples": int(branch["expected_consumed_samples"]),
        "expected_end_step": int(branch["expected_end_step"]),
    }
    if endpoint["actual_end_tokens"] < nominal_horizon_tokens:
        raise WorkflowError("batch-aligned WSD endpoint undershoots its nominal horizon")
    return endpoint


def _planned_checkpoint_count_for_job(
    row: Mapping[str, Any], runtime: Mapping[str, Any]
) -> int:
    action = str(row.get("action", ""))
    arguments = row.get("arguments", {})
    if action in {
        "training_sentinel",
        "fixed_lr_schedule_probe",
        "prescribed_lr_schedule_probe",
    }:
        return 1
    if action == "production_train":
        ladder = str(arguments["ladder"])
        horizons = _job_horizons(arguments, runtime)
        plan = _full_state_checkpoint_count(
            runtime,
            ladder=ladder,
            experts=int(arguments["experts"]),
            horizons=horizons,
        )
        if arguments.get("role") == "trunk":
            return int(plan["trunk"])
        return int(plan["branches"][str(int(arguments["D"]))])
    return 0


from .estimate_storage import _storage_estimate
