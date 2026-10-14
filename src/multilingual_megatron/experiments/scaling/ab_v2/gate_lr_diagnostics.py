"""Analyze the frozen A/B seed panel and LR-neighbor diagnostic evidence."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import _verified_receipt_evidence
from .common import (
    PRIMARY_METRIC,
    PRIMARY_TRAINING_SEED,
    REPLICATION_TRAINING_SEEDS,
    Paths,
    WorkflowError,
    _atomic_json,
    _file_sha256,
    _job,
    _load_object,
    _payload_sha256,
)
from .estimates import _expected_wsd_branch_endpoint
from .gate_evaluation import (
    _cohort_metric_from_eval,
    _metric_from_eval,
    _secondary_metric_from_eval,
    _validate_full_evaluation_artifacts,
)
from .training_health import _training_health_for_segment_plan
from .training_profile import _peak_lr_for_job
from .variant import active_policy


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise WorkflowError("cannot summarize empty or non-finite diagnostic values")
    observed = [float(value) for value in values]
    return {
        "count": len(observed),
        "mean": statistics.fmean(observed),
        "sample_standard_deviation": statistics.stdev(observed)
        if len(observed) > 1
        else 0.0,
        "minimum": min(observed),
        "maximum": max(observed),
        "range": max(observed) - min(observed),
    }


def _analyze_lr_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]], *, systematic_block_count: int
) -> dict[str, Any]:
    """Apply the frozen descriptive seed panel and 3-of-4 transfer rule."""

    by_identity: dict[tuple[str, int, int, int, float], Mapping[str, Any]] = {}
    for row in rows:
        identity = (
            str(row["ladder"]),
            int(row["experts"]),
            int(row["K"]),
            int(row["model_seed"]),
            float(row["lr_multiplier"]),
        )
        if identity in by_identity:
            raise WorkflowError(f"duplicate LR diagnostic identity: {identity}")
        by_identity[identity] = row

    expected_seed = {
        (ladder, experts, k, seed, 1.0)
        for ladder in active_policy().ladders
        for experts in (1, 32)
        for k in (1, 200)
        for seed in REPLICATION_TRAINING_SEEDS
    }
    expected_neighbors = {
        ("B", experts, k, PRIMARY_TRAINING_SEED, multiplier)
        for experts in (1, 32)
        for k in (1, 200)
        for multiplier in (0.5, 2.0)
    }
    expected = expected_seed | expected_neighbors
    if set(by_identity) != expected:
        raise WorkflowError(
            "LR diagnostic identities differ from the frozen 24-seed-plus-8-neighbor panel"
        )

    seed_cells: dict[str, Any] = {}
    for ladder in active_policy().ladders:
        for experts in (1, 32):
            for k in (1, 200):
                cell = f"{ladder}/E{experts}/K{k}"
                selected = [
                    by_identity[(ladder, experts, k, seed, 1.0)]
                    for seed in REPLICATION_TRAINING_SEEDS
                ]
                seed_cells[cell] = {
                    "primary_validation": _describe(
                        [float(row[PRIMARY_METRIC]) for row in selected]
                    ),
                    "english_common_cohort": _describe(
                        [float(row["english_common_cohort_loss"]) for row in selected]
                    ),
                    "by_seed": {
                        str(int(row["model_seed"])): {
                            PRIMARY_METRIC: float(row[PRIMARY_METRIC]),
                            "english_common_cohort_loss": float(
                                row["english_common_cohort_loss"]
                            ),
                        }
                        for row in selected
                    },
                }

    expert_effects: list[dict[str, Any]] = []
    language_effects: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for ladder in active_policy().ladders:
        for seed in REPLICATION_TRAINING_SEEDS:
            for k in (1, 200):
                dense = by_identity[(ladder, 1, k, seed, 1.0)]
                moe = by_identity[(ladder, 32, k, seed, 1.0)]
                expert_effects.append(
                    {
                        "ladder": ladder,
                        "K": k,
                        "seed": seed,
                        "E32_minus_E1_primary_loss": float(moe[PRIMARY_METRIC])
                        - float(dense[PRIMARY_METRIC]),
                    }
                )
            for experts in (1, 32):
                k1 = by_identity[(ladder, experts, 1, seed, 1.0)]
                k200 = by_identity[(ladder, experts, 200, seed, 1.0)]
                language_effects.append(
                    {
                        "ladder": ladder,
                        "experts": experts,
                        "seed": seed,
                        "K200_minus_K1_english_loss": float(
                            k200["english_common_cohort_loss"]
                        )
                        - float(k1["english_common_cohort_loss"]),
                    }
                )
            k1_effect = (
                float(by_identity[(ladder, 32, 1, seed, 1.0)]["english_common_cohort_loss"])
                - float(by_identity[(ladder, 1, 1, seed, 1.0)]["english_common_cohort_loss"])
            )
            k200_effect = (
                float(by_identity[(ladder, 32, 200, seed, 1.0)]["english_common_cohort_loss"])
                - float(by_identity[(ladder, 1, 200, seed, 1.0)]["english_common_cohort_loss"])
            )
            interactions.append(
                {
                    "ladder": ladder,
                    "seed": seed,
                    "expert_by_K_english_difference_in_differences": k200_effect
                    - k1_effect,
                }
            )

    lr_cells: dict[str, Any] = {}
    direction_counts = {"0.5": 0, "2.0": 0}
    for experts in (1, 32):
        for k in (1, 200):
            cell = f"B/E{experts}/K{k}"
            baseline = by_identity[("B", experts, k, PRIMARY_TRAINING_SEED, 1.0)]
            seed_range = float(seed_cells[cell]["primary_validation"]["range"])
            candidates: dict[str, Any] = {}
            for multiplier in (0.5, 2.0):
                neighbor = by_identity[
                    ("B", experts, k, PRIMARY_TRAINING_SEED, multiplier)
                ]
                improvement = float(baseline[PRIMARY_METRIC]) - float(
                    neighbor[PRIMARY_METRIC]
                )
                clear = improvement > seed_range
                direction_counts[str(multiplier)] += int(clear)
                candidates[str(multiplier)] = {
                    "primary_validation_loss": float(neighbor[PRIMARY_METRIC]),
                    "improvement_over_1x": improvement,
                    "clear_beyond_seed_range": clear,
                }
            lr_cells[cell] = {
                "baseline_1x_primary_validation_loss": float(
                    baseline[PRIMARY_METRIC]
                ),
                "three_seed_1x_range": seed_range,
                "neighbors": candidates,
            }
    blocked_directions = [
        multiplier
        for multiplier, count in direction_counts.items()
        if count >= systematic_block_count
    ]
    return {
        "seed_uncertainty": {
            "interpretation": (
                "descriptive A/B reference only; not a formal confidence interval or "
                "a C/D seed-variance estimate"
            ),
            "cells": seed_cells,
            "paired_expert_effects": expert_effects,
            "paired_language_effects_on_english": language_effects,
            "expert_by_K_interactions_on_english": interactions,
        },
        "learning_rate_transfer": {
            "comparison": "seed-1234 neighbors against seed-1234 1x",
            "clear_improvement_threshold": "strictly larger than the cell's three-seed 1x range",
            "systematic_block_count": systematic_block_count,
            "clear_cell_counts_by_multiplier": direction_counts,
            "blocked_directions": blocked_directions,
            "cells": lr_cells,
            "automatic_learning_rate_mutation": False,
        },
        "production_decision": "block" if blocked_directions else "ready",
    }


def _analyze_lr_confirmation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tie_absolute_tolerance: float,
    minimum_x1_cell_wins_or_ties: int,
    expected_ladder: str = "B",
) -> dict[str, Any]:
    """Apply the preregistered robust x1 decision rule to the 9x6 panel."""

    if not math.isfinite(tie_absolute_tolerance) or tie_absolute_tolerance < 0:
        raise WorkflowError("LR-confirmation tie tolerance is invalid")
    candidates = ("x025", "x050", "x100", "x200", "x400", "fixed2e4")
    expected = {
        (experts, k, candidate)
        for experts in (1, 16, 32)
        for k in (1, 12, 200)
        for candidate in candidates
    }
    by_identity: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        if (
            row.get("diagnostic_protocol")
            != "stage1_jointmoe_lr_confirmation_v1"
            or str(row.get("ladder")) != expected_ladder
            or int(row.get("model_seed", 0)) != PRIMARY_TRAINING_SEED
        ):
            raise WorkflowError("LR-confirmation row lineage drifted")
        identity = (
            int(row["experts"]),
            int(row["K"]),
            str(row["lr_candidate_id"]),
        )
        if identity in by_identity:
            raise WorkflowError(f"duplicate LR-confirmation identity: {identity}")
        by_identity[identity] = row
    if set(by_identity) != expected:
        raise WorkflowError("LR-confirmation identities differ from the frozen 9x6 panel")

    cells: dict[str, Any] = {}
    regrets_by_candidate: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    losses_by_candidate: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    wins_or_ties = {candidate: 0 for candidate in candidates}
    boundary_cells: list[str] = []
    for experts in (1, 16, 32):
        for k in (1, 12, 200):
            cell_name = f"{expected_ladder}/E{experts}/K{k}"
            losses = {
                candidate: float(by_identity[(experts, k, candidate)][PRIMARY_METRIC])
                for candidate in candidates
            }
            raw_nll = {
                candidate: float(by_identity[(experts, k, candidate)]["raw_nll"])
                for candidate in candidates
            }
            best_loss = min(losses.values())
            best_or_tied = sorted(
                candidate
                for candidate, loss in losses.items()
                if math.isclose(
                    loss,
                    best_loss,
                    rel_tol=0.0,
                    abs_tol=tie_absolute_tolerance,
                )
            )
            ranks = {
                candidate: 1
                + sum(
                    other_loss < loss - tie_absolute_tolerance
                    for other_loss in losses.values()
                )
                for candidate, loss in losses.items()
            }
            regrets = {
                candidate: loss - best_loss for candidate, loss in losses.items()
            }
            for candidate in candidates:
                losses_by_candidate[candidate].append(losses[candidate])
                regrets_by_candidate[candidate].append(regrets[candidate])
                wins_or_ties[candidate] += int(candidate in best_or_tied)
            if "x025" in best_or_tied or "x400" in best_or_tied:
                boundary_cells.append(cell_name)
            cells[cell_name] = {
                "primary_loss": losses,
                "raw_nll": raw_nll,
                "regret": regrets,
                "rank": ranks,
                "best_or_tied_candidates": best_or_tied,
            }

    aggregates: dict[str, Any] = {}
    for candidate in candidates:
        candidate_regrets = regrets_by_candidate[candidate]
        aggregates[candidate] = {
            "equal_cell_mean_primary_loss": statistics.fmean(
                losses_by_candidate[candidate]
            ),
            "equal_cell_mean_regret": statistics.fmean(candidate_regrets),
            "median_regret": statistics.median(candidate_regrets),
            "maximum_regret": max(candidate_regrets),
            "wins_or_ties": wins_or_ties[candidate],
        }
    x1_mean_regret = float(aggregates["x100"]["equal_cell_mean_regret"])
    x1_strictly_lowest_mean_regret = all(
        x1_mean_regret + tie_absolute_tolerance
        < float(aggregates[candidate]["equal_cell_mean_regret"])
        for candidate in candidates
        if candidate != "x100"
    )
    x1_beats_fixed = (
        float(aggregates["x100"]["equal_cell_mean_primary_loss"])
        + tie_absolute_tolerance
        < float(aggregates["fixed2e4"]["equal_cell_mean_primary_loss"])
    )
    x1_has_required_cell_support = (
        wins_or_ties["x100"] >= minimum_x1_cell_wins_or_ties
    )
    checks = {
        "x1_strictly_lowest_equal_cell_mean_regret": x1_strictly_lowest_mean_regret,
        "x1_wins_or_ties_at_least_required_cells": x1_has_required_cell_support,
        "x1_beats_fixed_2e4_in_equal_cell_mean_primary_loss": x1_beats_fixed,
        "no_outer_boundary_candidate_is_best_or_tied": not boundary_cells,
    }
    if boundary_cells:
        decision = "inconclusive_boundary"
    elif all(checks.values()):
        decision = "confirmed"
    else:
        decision = "not_confirmed"
    return {
        "protocol": "stage1_jointmoe_lr_confirmation_v1",
        "ladder": expected_ladder,
        "decision": decision,
        "primary_metric": PRIMARY_METRIC,
        "lower_is_better": True,
        "negative_primary_values_are_valid": True,
        "tie_absolute_tolerance": tie_absolute_tolerance,
        "minimum_x1_cell_wins_or_ties": minimum_x1_cell_wins_or_ties,
        "checks": checks,
        "boundary_cells": boundary_cells,
        "candidate_aggregates": aggregates,
        "cells": cells,
        "automatic_learning_rate_mutation": False,
        "manual_production_authorization_required": True,
        "production_decision": "manual_stop",
    }


def _segment_plan_matches_nominal_endpoint(
    plan_payload: Mapping[str, Any],
    *,
    nominal_horizon: int,
    expected_endpoint: Mapping[str, int],
) -> bool:
    """Validate either a branch execution plan or the legacy combined plan.

    LR-confirmation results point at the branch-owned execution plan.  That
    plan records its exact endpoint in top-level fields plus ``segments``;
    combined trajectory plans instead record the same endpoint in
    ``dataset_horizon_tokens`` plus ``branches``.  Both representations are
    deterministic descriptions of the same nominal-horizon contract.
    """

    if plan_payload.get("plan_role") == "branch":
        segments = plan_payload.get("segments", [])
        if not isinstance(segments, list) or not segments:
            return False
        final_segment = segments[-1]
        if not isinstance(final_segment, Mapping):
            return False
        return (
            int(plan_payload.get("target_token_budget_tokens", 0))
            == nominal_horizon
            and int(plan_payload.get("branch_target_tokens", 0))
            == nominal_horizon
            and int(plan_payload.get("expected_tokens", 0))
            == int(expected_endpoint["actual_end_tokens"])
            and int(plan_payload.get("expected_consumed_samples", 0))
            == int(expected_endpoint["expected_consumed_samples"])
            and int(plan_payload.get("expected_end_step", 0))
            == int(expected_endpoint["expected_end_step"])
            and int(plan_payload.get("final_cumulative_step", 0))
            == int(expected_endpoint["expected_end_step"])
            and int(final_segment.get("target_end_tokens", 0))
            == nominal_horizon
            and int(final_segment.get("actual_end_tokens", 0))
            == int(expected_endpoint["actual_end_tokens"])
            and int(final_segment.get("expected_consumed_samples", 0))
            == int(expected_endpoint["expected_consumed_samples"])
            and int(final_segment.get("expected_end_step", 0))
            == int(expected_endpoint["expected_end_step"])
        )

    branches = [
        row
        for row in plan_payload.get("branches", [])
        if isinstance(row, Mapping)
        and int(row.get("target_tokens", 0)) == nominal_horizon
    ]
    return (
        int(plan_payload.get("dataset_horizon_tokens", 0)) == nominal_horizon
        and len(branches) == 1
        and int(branches[0].get("actual_end_tokens", 0))
        == int(expected_endpoint["actual_end_tokens"])
        and int(branches[0].get("expected_consumed_samples", 0))
        == int(expected_endpoint["expected_consumed_samples"])
        and int(branches[0].get("expected_end_step", 0))
        == int(expected_endpoint["expected_end_step"])
    )


def _run_lr_diagnostic_gate(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    paths: Paths,
    runtime: Mapping[str, Any],
) -> None:
    from multilingual_megatron.pipeline.stages.training.checkpoint_state import (
        inspect_full_state_checkpoint,
    )

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    for eval_job_id in job["arguments"]["evaluation_jobs"]:
        source = _job(manifest, str(eval_job_id))
        args = source.get("arguments", {})
        training_job_id = str(args.get("training_job_id", ""))
        training_job = _job(manifest, training_job_id)
        training_args = training_job.get("arguments", {})
        eval_path = paths.jobs / str(eval_job_id) / "evaluation_manifest.json"
        result_path = paths.jobs / training_job_id / "result.json"
        try:
            evidence[str(eval_job_id)] = _verified_receipt_evidence(
                manifest, str(eval_job_id)
            )
            evidence[training_job_id] = _verified_receipt_evidence(
                manifest, training_job_id
            )
            artifacts = _validate_full_evaluation_artifacts(eval_path)
            primary = _metric_from_eval(eval_path)
            raw_nll = _secondary_metric_from_eval(eval_path)
            english = _cohort_metric_from_eval(eval_path, "english")
            result = _load_object(result_path)
            if (
                result.get("status") != "complete"
                or result.get("arguments") != training_args
                or training_args.get("purpose") != "lr_diagnostic"
                or int(training_args.get("D", 0))
                != int(job["arguments"]["horizon_tokens"])
                or result.get("training_health", {}).get("status") != "ok"
            ):
                raise WorkflowError("training result does not match the diagnostic contract")
            if not math.isclose(
                float(result.get("peak_lr", math.nan)),
                _peak_lr_for_job(manifest, training_args),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise WorkflowError("training result peak LR lineage is invalid")
            segment_plan = Path(str(result.get("segment_plan", "")))
            if (
                not segment_plan.is_file()
                or result.get("segment_plan_sha256") != _file_sha256(segment_plan)
            ):
                raise WorkflowError("training segment-plan identity is invalid")
            nominal_horizon = int(job["arguments"]["horizon_tokens"])
            expected_endpoint = _expected_wsd_branch_endpoint(
                runtime,
                ladder=str(training_args["ladder"]),
                experts=int(training_args["experts"]),
                nominal_horizon_tokens=nominal_horizon,
            )
            if training_args.get("expected_endpoint") != expected_endpoint:
                raise WorkflowError("training expected-endpoint contract drifted")
            if (
                int(result.get("actual_end_tokens", 0))
                != expected_endpoint["actual_end_tokens"]
                or int(result.get("expected_consumed_samples", 0))
                != expected_endpoint["expected_consumed_samples"]
                or int(result.get("final_checkpoint_step", 0))
                != expected_endpoint["expected_end_step"]
            ):
                raise WorkflowError(
                    "training result differs from its exact batch-aligned endpoint"
                )
            plan_payload = _load_object(segment_plan)
            if not _segment_plan_matches_nominal_endpoint(
                plan_payload,
                nominal_horizon=nominal_horizon,
                expected_endpoint=expected_endpoint,
            ):
                raise WorkflowError("segment plan differs from the nominal-2B contract")
            reparsed_health = _training_health_for_segment_plan(
                segment_plan, require_moe=int(training_args["experts"]) > 1
            )
            checkpoint = inspect_full_state_checkpoint(
                checkpoint_root=Path(str(result["final_checkpoint_root"])),
                expected_step=int(result["final_checkpoint_step"]),
                expected_consumed_samples=int(result["expected_consumed_samples"]),
            )
            if not checkpoint.valid_full_state:
                raise WorkflowError(
                    f"diagnostic final checkpoint is invalid: {checkpoint.errors}"
                )
            for key in (
                "trajectory_id",
                "diagnostic_roles",
                "ladder",
                "experts",
                "K",
                "D",
                "model_seed",
                "dataset_random_seed",
                "expected_endpoint",
                "base_peak_lr",
                "lr_multiplier",
                "peak_lr",
                "diagnostic_protocol",
                "lr_candidate_id",
                "lr_candidate_kind",
                "jointmoe_factor",
                "fixed_control_peak_lr",
            ):
                if args.get(key) != training_args.get(key):
                    raise WorkflowError(f"evaluation/training {key} lineage differs")
            row = {
                "evaluation_job_id": str(eval_job_id),
                "training_job_id": training_job_id,
                "trajectory_id": str(args["trajectory_id"]),
                "diagnostic_roles": list(args["diagnostic_roles"]),
                "ladder": str(args["ladder"]),
                "experts": int(args["experts"]),
                "K": int(args["K"]),
                "D": int(args["D"]),
                "model_seed": int(args["model_seed"]),
                "dataset_random_seed": int(args["dataset_random_seed"]),
                "base_peak_lr": float(args["base_peak_lr"]),
                "lr_multiplier": float(args["lr_multiplier"]),
                "peak_lr": float(args["peak_lr"]),
                "diagnostic_protocol": args.get("diagnostic_protocol"),
                "lr_candidate_id": args.get("lr_candidate_id"),
                "lr_candidate_kind": args.get("lr_candidate_kind"),
                "jointmoe_factor": args.get("jointmoe_factor"),
                "fixed_control_peak_lr": args.get("fixed_control_peak_lr"),
                "nominal_horizon_tokens": nominal_horizon,
                "actual_end_tokens": expected_endpoint["actual_end_tokens"],
                "token_overshoot": expected_endpoint["actual_end_tokens"] - nominal_horizon,
                PRIMARY_METRIC: primary,
                "raw_nll": raw_nll,
                "english_common_cohort_loss": english,
                "evaluation_artifacts": artifacts,
                "training_health_sha256": _payload_sha256(reparsed_health),
                "checkpoint_state": checkpoint.to_dict(),
            }
            rows.append(row)
        except (KeyError, TypeError, ValueError, WorkflowError) as exc:
            errors.append(f"{eval_job_id}: {exc}")

    analysis: dict[str, Any] = {}
    if not errors:
        try:
            if (
                job["arguments"].get("analysis_protocol")
                == "stage1_jointmoe_lr_confirmation_v1"
            ):
                analysis = _analyze_lr_confirmation_rows(
                    rows,
                    tie_absolute_tolerance=float(
                        job["arguments"]["tie_absolute_tolerance"]
                    ),
                    minimum_x1_cell_wins_or_ties=int(
                        job["arguments"]["minimum_x1_cell_wins_or_ties"]
                    ),
                    expected_ladder=str(job["arguments"].get("ladder", "B")),
                )
            else:
                analysis = _analyze_lr_diagnostic_rows(
                    rows,
                    systematic_block_count=int(
                        job["arguments"]["systematic_block_count"]
                    ),
                )
        except (KeyError, TypeError, ValueError, WorkflowError) as exc:
            errors.append(f"diagnostic analysis: {exc}")
    confirmation_protocol = (
        job["arguments"].get("analysis_protocol")
        == "stage1_jointmoe_lr_confirmation_v1"
    )
    blocked = analysis.get("production_decision") == "block"
    report_status = (
        "failed"
        if errors
        else (
            str(analysis.get("decision"))
            if confirmation_protocol
            else ("blocked" if blocked else "ready")
        )
    )
    report = {
        "schema_version": 1,
        "status": report_status,
        "errors": errors,
        "expected_trajectories": int(
            job["arguments"].get("expected_trajectories", 32)
        ),
        "observed_trajectories": len(rows),
        "token_endpoint_contract": {
            "interpretation": (
                "nominal 2B-token horizon with exact deterministic whole-batch "
                "endpoint; no nominal undershoot"
            ),
            "nominal_horizon_tokens": int(job["arguments"]["horizon_tokens"]),
            "observed_actual_end_tokens": sorted(
                {int(row["actual_end_tokens"]) for row in rows}
            ),
            "observed_token_overshoots": sorted(
                {int(row["token_overshoot"]) for row in rows}
            ),
        },
        "rows": rows,
        "analysis": analysis,
        "verified_dependency_evidence": evidence,
        "automatic_learning_rate_mutation": False,
    }
    if confirmation_protocol and not errors:
        decision = str(analysis.get("decision"))
        report["operational_response"] = {
            "automatic_slurm_hold_or_cancel": False,
            "boundary_extension_submitted": False,
            "additional_seeds_submitted": False,
            "alert_severity": (
                "none"
                if decision == "confirmed"
                else ("warning" if decision == "inconclusive_boundary" else "blocking")
            ),
            "message": {
                "confirmed": "x1 confirmation criteria passed; report the bounded 2B evidence",
                "inconclusive_boundary": (
                    "outer-boundary winner observed; disclose and review manually without extension"
                ),
                "not_confirmed": (
                    "x1 confirmation criteria failed; manually pause any new LR-dependent decision"
                ),
            }.get(decision, "unexpected confirmation decision; review manually"),
        }
    report["gate_sha256"] = _payload_sha256(report)
    gate_path = (
        paths.control / "lr_confirmation_gate.json"
        if confirmation_protocol
        else paths.control / "lr_diagnostic_gate.json"
    )
    _atomic_json(gate_path, report)
    if errors:
        raise WorkflowError("LR diagnostic gate failed: " + "; ".join(errors))
    if blocked and not confirmation_protocol:
        directions = analysis["learning_rate_transfer"]["blocked_directions"]
        raise WorkflowError(
            f"LR transfer gate blocked production for scientific review: {directions}"
        )
