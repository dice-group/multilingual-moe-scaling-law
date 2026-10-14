"""Construct the bounded-smoke and storage-safe production DAG."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2.common import (
    Paths,
    WorkflowError,
    _rel_or_abs,
)
from multilingual_megatron.experiments.scaling.ab_v2.estimates import _batch_layout

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    DESIGN_ID,
    EXPERTS,
    HORIZONS,
    SEEDS,
    SMOKE_EXPERTS,
    SMOKE_GLOBAL_BATCH_SIZE,
    SMOKE_MAX_STEP_TIME_CV_PCT,
    SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS,
    SMOKE_TOKENS,
)
from multilingual_megatron.experiments.scaling.language_composition.training.production_smoke_evidence import (
    EVIDENCE_GATE_FILENAME,
    EVIDENCE_GATE_JOB_ID,
)


ARMS = ("compact", "diverse")
WAVES = tuple((arm, seed) for seed in SEEDS for arm in ARMS)


def _output(paths: Paths, job_id: str, filename: str) -> list[str]:
    return [_rel_or_abs(paths.jobs / job_id / filename)]


def _training_contract(
    template_by_id: Mapping[str, Mapping[str, Any]], *, experts: int, seed: int
) -> dict[str, Any]:
    template = template_by_id[f"prod_trunk_b_e{experts}_k012"]
    source = template.get("arguments")
    if not isinstance(source, Mapping):
        raise WorkflowError(f"production template lacks B/E{experts}/K12 arguments")
    result = {
        key: deepcopy(source[key])
        for key in (
            "base_peak_lr",
            "learning_rate_policy",
            "lr_multiplier",
            "peak_lr",
        )
    }
    result.update({"model_seed": seed, "dataset_random_seed": seed})
    return result


def build_jobs(
    *,
    template_jobs: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    output_root: Path,
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    template_by_id = {str(row["job_id"]): row for row in template_jobs}
    paths = Paths(output_root.resolve())
    jobs: list[dict[str, Any]] = []
    for job_id in (
        "storage_preflight",
        "data_reuse_gate",
        "data_megatron_reader_smoke",
    ):
        try:
            jobs.append(deepcopy(dict(template_by_id[job_id])))
        except KeyError as exc:
            raise WorkflowError(f"production template lacks {job_id}") from exc

    smoke_ids: list[str] = []
    composition = runtime["language_composition"]
    smoke = composition["smoke"]
    if (
        int(smoke["tokens"]) != SMOKE_TOKENS
        or int(smoke["global_batch_size"]) != SMOKE_GLOBAL_BATCH_SIZE
        or int(smoke["seed"]) != 1234
        or tuple(smoke["experts"]) != SMOKE_EXPERTS
        or int(smoke["steady_state_window_observations"])
        != SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS
        or float(smoke["maximum_step_time_cv_pct"])
        != SMOKE_MAX_STEP_TIME_CV_PCT
        or float(smoke["minimum_reference_throughput_fraction"]) != 0.85
        or smoke["reference_median_tokens_per_second"]
        != {"1": 329_906.87, "32": 203_559.56}
    ):
        raise WorkflowError("language-composition smoke runtime drifted")
    production_smoke_evidence = composition.get("production_smoke_evidence")
    if isinstance(production_smoke_evidence, Mapping):
        jobs.append(
            {
                "job_id": EVIDENCE_GATE_JOB_ID,
                "stage": "sentinel",
                "action": EVIDENCE_GATE_JOB_ID,
                "resource": "cpu",
                "dependencies": ["data_megatron_reader_smoke"],
                "array": None,
                "arguments": dict(production_smoke_evidence),
                "outputs": [
                    _rel_or_abs(paths.control / EVIDENCE_GATE_FILENAME)
                ],
                "gpu_tokens": 0,
            }
        )
        authorization_dependency = EVIDENCE_GATE_JOB_ID
    else:
        for arm in ARMS:
            for experts in SMOKE_EXPERTS:
                job_id = f"smoke_{arm}_b_e{experts}_s1234"
                micro, data_parallel, accumulation = _batch_layout(
                    runtime,
                    ladder="B",
                    experts=experts,
                    global_batch_size=SMOKE_GLOBAL_BATCH_SIZE,
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "stage": "sentinel",
                        "action": "training_sentinel",
                        "resource": "h100",
                        "dependencies": ["data_megatron_reader_smoke"],
                        "array": None,
                        "arguments": {
                            "ladder": "B",
                            "experts": experts,
                            "K": 12,
                            "tokens": SMOKE_TOKENS,
                            "sentinel_kind": "language_composition_long_real_data",
                            "global_batch_size": SMOKE_GLOBAL_BATCH_SIZE,
                            "micro_batch_size": micro,
                            "data_parallel_size": data_parallel,
                            "gradient_accumulation_steps": accumulation,
                            "composition_arm": arm,
                            "language_composition_design_id": DESIGN_ID,
                            **_training_contract(
                                template_by_id, experts=experts, seed=1234
                            ),
                        },
                        "outputs": _output(paths, job_id, "result.json"),
                        "gpu_tokens": SMOKE_TOKENS,
                    }
                )
                smoke_ids.append(job_id)
        jobs.append(
            {
                "job_id": "language_composition_smoke_gate",
                "stage": "sentinel",
                "action": "language_composition_smoke_gate",
                "resource": "cpu",
                "dependencies": smoke_ids,
                "array": None,
                "arguments": {
                    "smoke_jobs": smoke_ids,
                    "minimum_reference_throughput_fraction": smoke[
                        "minimum_reference_throughput_fraction"
                    ],
                    "reference_median_tokens_per_second": dict(
                        smoke["reference_median_tokens_per_second"]
                    ),
                    "steady_state_window_observations": smoke[
                        "steady_state_window_observations"
                    ],
                    "maximum_step_time_cv_pct": smoke[
                        "maximum_step_time_cv_pct"
                    ],
                },
                "outputs": [
                    _rel_or_abs(
                        paths.control / "language_composition_smoke_gate.json"
                    )
                ],
                "gpu_tokens": 0,
            }
        )
        jobs.append(
            {
                "job_id": "retention_prune_sentinels",
                "stage": "sentinel",
                "action": "checkpoint_prune",
                "resource": "cpu",
                "dependencies": ["language_composition_smoke_gate"],
                "array": None,
                "arguments": {
                    "source_jobs": smoke_ids,
                    "evidence_jobs": ["language_composition_smoke_gate"],
                    "retention_scope": (
                        "four language-composition real-data smoke checkpoints"
                    ),
                },
                "outputs": [
                    _rel_or_abs(
                        paths.control / "prune" / "retention_prune_sentinels.json"
                    )
                ],
                "gpu_tokens": 0,
            }
        )
        authorization_dependency = "retention_prune_sentinels"
    jobs.append(
        {
            "job_id": "language_composition_authorization_gate",
            "stage": "production",
            "action": "language_composition_authorization_gate",
            "resource": "cpu",
            "dependencies": [authorization_dependency],
            "array": None,
            "arguments": {
                "authorization_file": runtime["language_composition"][
                    "production_authorization_file"
                ],
                "authorized_scope": "18 new trajectories and 162 K196 evaluations",
            },
            "outputs": [
                _rel_or_abs(
                    paths.control / "language_composition_authorization_gate.json"
                )
            ],
            "gpu_tokens": 0,
        }
    )

    all_new_evaluations: list[str] = []
    cleanup_by_wave: list[list[str]] = []
    previous_wave_gate = ["language_composition_authorization_gate"]
    for wave_index, (arm, seed) in enumerate(WAVES, start=1):
        wave_cleanups: list[str] = []
        for experts in EXPERTS:
            trunk_id = f"lc_trunk_{arm}_b_e{experts}_s{seed}"
            training_contract = _training_contract(
                template_by_id, experts=experts, seed=seed
            )
            jobs.append(
                {
                    "job_id": trunk_id,
                    "stage": "production",
                    "action": "production_train",
                    "resource": "h100",
                    "dependencies": list(previous_wave_gate),
                    "array": None,
                    "arguments": {
                        "role": "trunk",
                        "ladder": "B",
                        "experts": experts,
                        "K": 12,
                        "D": HORIZONS[-1],
                        "composition_arm": arm,
                        "language_composition_design_id": DESIGN_ID,
                        "wave_index": wave_index,
                        **training_contract,
                    },
                    "outputs": _output(paths, trunk_id, "result.json"),
                    "gpu_tokens": int(0.8 * HORIZONS[-1]),
                }
            )
            evaluations: list[str] = []
            smaller_branches: list[str] = []
            terminal_branch = ""
            for horizon in HORIZONS:
                branch_id = (
                    f"lc_branch_{arm}_b_e{experts}_d{horizon}_s{seed}"
                )
                eval_id = f"lc_eval_{arm}_b_e{experts}_d{horizon}_s{seed}"
                jobs.append(
                    {
                        "job_id": branch_id,
                        "stage": "production",
                        "action": "production_train",
                        "resource": "h100",
                        "dependencies": [trunk_id],
                        "array": None,
                        "arguments": {
                            "role": "branch",
                            "parent_job_id": trunk_id,
                            "ladder": "B",
                            "experts": experts,
                            "K": 12,
                            "D": horizon,
                            "composition_arm": arm,
                            "language_composition_design_id": DESIGN_ID,
                            "wave_index": wave_index,
                            **training_contract,
                        },
                        "outputs": _output(paths, branch_id, "result.json"),
                        "gpu_tokens": int(0.2 * horizon),
                    }
                )
                jobs.append(
                    {
                        "job_id": eval_id,
                        "stage": "validation",
                        "action": "per_language_eval",
                        "resource": "h100",
                        "dependencies": [branch_id],
                        "array": None,
                        "arguments": {
                            "training_job_id": branch_id,
                            "ladder": "B",
                            "experts": experts,
                            "K": 12,
                            "D": horizon,
                            "purpose": "language_composition",
                            "model_seed": seed,
                            "dataset_random_seed": seed,
                            "composition_arm": arm,
                            "language_composition_design_id": DESIGN_ID,
                        },
                        "outputs": _output(paths, eval_id, "evaluation_manifest.json"),
                        "gpu_tokens": 0,
                    }
                )
                evaluations.append(eval_id)
                all_new_evaluations.append(eval_id)
                if horizon == HORIZONS[-1]:
                    terminal_branch = branch_id
                else:
                    smaller_branches.append(branch_id)
            cleanup_id = f"retention_prune_{arm}_b_e{experts}_s{seed}"
            jobs.append(
                {
                    "job_id": cleanup_id,
                    "stage": "production",
                    "action": "checkpoint_prune",
                    "resource": "cpu",
                    "dependencies": evaluations,
                    "array": None,
                    "arguments": {
                        "source_jobs": [trunk_id, *smaller_branches],
                        "retained_jobs": [terminal_branch],
                        "evidence_jobs": evaluations,
                        "production_wave": {
                            "index": wave_index,
                            "arm": arm,
                            "seed": seed,
                            "experts": experts,
                        },
                        "retention_scope": (
                            "after six K196 readouts, remove the trajectory trunk and "
                            "five nonterminal branches; retain only the 16B branch"
                        ),
                    },
                    "outputs": [
                        _rel_or_abs(paths.control / "prune" / f"{cleanup_id}.json")
                    ],
                    "gpu_tokens": 0,
                }
            )
            wave_cleanups.append(cleanup_id)
        cleanup_by_wave.append(wave_cleanups)
        previous_wave_gate = wave_cleanups

    reference_evaluations: list[str] = []
    for reference in references:
        seed = int(reference["seed"])
        experts = int(reference["experts"])
        horizon = int(reference["horizon_tokens"])
        eval_id = f"lc_eval_reference_b_e{experts}_d{horizon}_s{seed}"
        jobs.append(
            {
                "job_id": eval_id,
                "stage": "validation",
                "action": "per_language_eval",
                "resource": "h100",
                "dependencies": ["language_composition_authorization_gate"],
                "array": None,
                "arguments": {
                    "training_job_id": str(reference["source_job_id"]),
                    "external_training_result": dict(reference["result"]),
                    "source_training_receipt": dict(reference["receipt"]),
                    "ladder": "B",
                    "experts": experts,
                    "K": 12,
                    "D": horizon,
                    "purpose": "language_composition_reference_reevaluation",
                    "model_seed": seed,
                    "dataset_random_seed": seed,
                    "composition_arm": "reference",
                    "language_composition_design_id": DESIGN_ID,
                },
                "outputs": _output(paths, eval_id, "evaluation_manifest.json"),
                "gpu_tokens": 0,
            }
        )
        reference_evaluations.append(eval_id)

    jobs.append(
        {
            "job_id": "language_composition_summary",
            "stage": "summary",
            "action": "language_composition_summary",
            "resource": "cpu",
            "dependencies": [*cleanup_by_wave[-1], *reference_evaluations],
            "array": None,
            "arguments": {
                "new_evaluation_jobs": all_new_evaluations,
                "reference_evaluation_jobs": reference_evaluations,
                "primary_horizon_tokens": HORIZONS[-1],
                "primary_cohort": "unseen_177",
            },
            "outputs": [
                _rel_or_abs(paths.control / "language_composition_summary.json")
            ],
            "gpu_tokens": 0,
        }
    )
    validate_topology(jobs)
    return jobs


def validate_topology(jobs: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(row["job_id"]): row for row in jobs}
    reuses_smoke_evidence = EVIDENCE_GATE_JOB_ID in by_id
    expected_counts = {
        "smokes": 0 if reuses_smoke_evidence else 4,
        "trunks": 18,
        "branches": 108,
        "new_evaluations": 108,
        "reference_evaluations": 54,
        "trajectory_cleanups": 18,
    }
    observed = {
        "smokes": sum(job_id.startswith("smoke_") for job_id in by_id),
        "trunks": sum(job_id.startswith("lc_trunk_") for job_id in by_id),
        "branches": sum(job_id.startswith("lc_branch_") for job_id in by_id),
        "new_evaluations": sum(
            job_id.startswith(("lc_eval_compact_", "lc_eval_diverse_"))
            for job_id in by_id
        ),
        "reference_evaluations": sum(
            job_id.startswith("lc_eval_reference_") for job_id in by_id
        ),
        "trajectory_cleanups": sum(
            job_id.startswith(("retention_prune_compact_", "retention_prune_diverse_"))
            for job_id in by_id
        ),
    }
    expected_job_count = 312 if reuses_smoke_evidence else 317
    if observed != expected_counts or len(jobs) != expected_job_count:
        raise WorkflowError(
            f"language-composition topology cardinality drifted: {observed}"
        )
    authorization = by_id.get("language_composition_authorization_gate", {})
    if reuses_smoke_evidence:
        evidence_gate = by_id[EVIDENCE_GATE_JOB_ID]
        if (
            evidence_gate.get("action") != EVIDENCE_GATE_JOB_ID
            or evidence_gate.get("resource") != "cpu"
            or evidence_gate.get("dependencies") != ["data_megatron_reader_smoke"]
            or "language_composition_smoke_gate" in by_id
            or "retention_prune_sentinels" in by_id
            or authorization.get("dependencies") != [EVIDENCE_GATE_JOB_ID]
        ):
            raise WorkflowError(
                "language-composition reused smoke-evidence topology drifted"
            )
    elif authorization.get("dependencies") != ["retention_prune_sentinels"]:
        raise WorkflowError("language-composition smoke dependency drifted")
    if any(
        row.get("action") == "production_train"
        and row.get("arguments", {}).get("composition_arm") == "reference"
        for row in jobs
    ):
        raise WorkflowError("language-composition workflow retrains the reference arm")
    previous_gate = ["language_composition_authorization_gate"]
    for wave_index, (arm, seed) in enumerate(WAVES, start=1):
        current_cleanups: list[str] = []
        for experts in EXPERTS:
            trunk = f"lc_trunk_{arm}_b_e{experts}_s{seed}"
            if by_id.get(trunk, {}).get("dependencies") != previous_gate:
                raise WorkflowError(f"language-composition wave dependency drifted: {trunk}")
            evaluations = [
                f"lc_eval_{arm}_b_e{experts}_d{horizon}_s{seed}"
                for horizon in HORIZONS
            ]
            cleanup = f"retention_prune_{arm}_b_e{experts}_s{seed}"
            current_cleanups.append(cleanup)
            cleanup_args = by_id.get(cleanup, {}).get("arguments", {})
            if (
                by_id.get(cleanup, {}).get("dependencies") != evaluations
                or cleanup_args.get("evidence_jobs") != evaluations
                or cleanup_args.get("production_wave")
                != {
                    "index": wave_index,
                    "arm": arm,
                    "seed": seed,
                    "experts": experts,
                }
                or cleanup_args.get("retained_jobs")
                != [f"lc_branch_{arm}_b_e{experts}_d{HORIZONS[-1]}_s{seed}"]
            ):
                raise WorkflowError(
                    f"language-composition cleanup contract drifted: {cleanup}"
                )
            for horizon in HORIZONS:
                branch = f"lc_branch_{arm}_b_e{experts}_d{horizon}_s{seed}"
                evaluation = f"lc_eval_{arm}_b_e{experts}_d{horizon}_s{seed}"
                if by_id.get(branch, {}).get("dependencies") != [trunk] or by_id.get(
                    evaluation, {}
                ).get("dependencies") != [branch]:
                    raise WorkflowError(
                        f"language-composition trajectory dependency drifted: {branch}"
                    )
        previous_gate = current_cleanups
    summary = by_id.get("language_composition_summary", {})
    if (
        summary.get("action") != "language_composition_summary"
        or set(summary.get("dependencies", ()))
        != set(previous_gate)
        | {
            job_id for job_id in by_id if job_id.startswith("lc_eval_reference_")
        }
    ):
        raise WorkflowError("language-composition summary dependencies drifted")
