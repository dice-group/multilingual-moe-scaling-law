"""Sentinel-job construction for Stage 1 workflows."""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    EXPERTS,
    PRIMARY_TRAINING_SEED,
    SENTINEL_INTEGRATION_LANGUAGE,
    Paths,
    WorkflowError,
)
from .estimates import _training_data_parallel_size
from .job_core import _add_job, _training_contract_arguments
from .variant import active_policy


def _sentinel_jobs(
    jobs: list[dict[str, Any]],
    *,
    data_gate: str,
    runtime: Mapping[str, Any],
    paths: Paths,
    lr_prescription: Mapping[str, Any] | None,
) -> str:
    policy = active_policy()
    sentinels: list[str] = []
    sentinel_by_probe: dict[tuple[str, int, str], str] = {}
    for raw in runtime["training"]["sentinels"]:
        ladder = str(raw["ladder"])
        experts = int(raw["experts"])
        k = int(raw["language_count"])
        tokens = int(raw["tokens"])
        kind = str(raw["kind"])
        global_batch_size = int(raw["global_batch_size"])
        plateau_mbs = int(runtime["training"]["micro_batch_size"][ladder][str(experts)])
        data_parallel_size = _training_data_parallel_size(
            runtime, ladder=ladder, experts=experts
        )
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            derive_segment_micro_batch_size,
        )

        micro_batch_size = derive_segment_micro_batch_size(
            plateau_micro_batch_size=plateau_mbs,
            global_batch_size=global_batch_size,
            data_parallel_size=data_parallel_size,
        )
        if global_batch_size % (micro_batch_size * data_parallel_size):
            raise WorkflowError(
                f"invalid sentinel GBS/MBS/DP for {kind}/{ladder}/E{experts}"
            )
        sentinel_id = _add_job(
            jobs,
            job_id=(
                f"sentinel_{kind}_{ladder.lower()}_e{experts}_k{k:03d}"
                f"_gbs{global_batch_size}"
            ),
            stage="sentinel",
            action="training_sentinel",
            resource="h100",
            dependencies=(data_gate,),
            arguments={
                "ladder": ladder,
                "experts": experts,
                "K": k,
                "tokens": tokens,
                "sentinel_kind": kind,
                "global_batch_size": global_batch_size,
                "micro_batch_size": micro_batch_size,
                "data_parallel_size": data_parallel_size,
                "gradient_accumulation_steps": global_batch_size
                // (micro_batch_size * data_parallel_size),
                **_training_contract_arguments(
                    lr_prescription,
                    ladder=ladder,
                    experts=experts,
                    training_seed=PRIMARY_TRAINING_SEED,
                ),
            },
            outputs=(
                paths.jobs
                / (
                    f"sentinel_{kind}_{ladder.lower()}_e{experts}_k{k:03d}"
                    f"_gbs{global_batch_size}"
                )
                / "result.json",
            ),
            gpu_tokens=tokens,
        )
        sentinels.append(sentinel_id)
        sentinel_by_probe[(ladder, experts, kind)] = sentinel_id

    raw_integration_cells = runtime["training"].get(
        "integration_sentinel_cells", (("A", 1), ("A", 32), ("B", 32))
    )
    if not isinstance(raw_integration_cells, (list, tuple)):
        raise WorkflowError("training.integration_sentinel_cells must be a sequence")
    integration_cells: list[tuple[str, int]] = []
    for cell in raw_integration_cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise WorkflowError(
                "each evaluator integration sentinel cell must be [ladder, experts]"
            )
        ladder, experts = str(cell[0]), int(cell[1])
        if ladder not in policy.ladders or experts not in EXPERTS:
            raise WorkflowError(f"invalid evaluator integration sentinel cell {cell!r}")
        integration_cells.append((ladder, experts))
    if len(integration_cells) != len(set(integration_cells)):
        raise WorkflowError("evaluator integration sentinel cells must be unique")

    integration_jobs: list[str] = []
    for ladder, experts in integration_cells:
        preferred_kind = "early_correctness" if experts == 1 else "plateau_memory_perf"
        training_job_id = sentinel_by_probe.get((ladder, experts, preferred_kind))
        if training_job_id is None:
            raise WorkflowError(
                f"missing {ladder}/E{experts} training sentinel for evaluator integration"
            )
        eval_id = f"sentinel_eval_{ladder.lower()}_e{experts}_eng"
        integration_jobs.append(
            _add_job(
                jobs,
                job_id=eval_id,
                stage="sentinel",
                action="per_language_eval",
                resource="h100",
                dependencies=(training_job_id,),
                arguments={
                    "training_job_id": training_job_id,
                    "ladder": ladder,
                    "experts": experts,
                    "K": 50,
                    "purpose": "sentinel_integration",
                    "integration_languages": [SENTINEL_INTEGRATION_LANGUAGE],
                    "model_seed": PRIMARY_TRAINING_SEED,
                    "dataset_random_seed": PRIMARY_TRAINING_SEED,
                },
                outputs=(paths.jobs / eval_id / "evaluation_manifest.json",),
            )
        )
    gate_id = _add_job(
        jobs,
        job_id="sentinel_gate",
        stage="sentinel",
        action="sentinel_gate",
        resource="cpu",
        dependencies=tuple([*sentinels, *integration_jobs]),
        arguments={
            "sentinel_jobs": sentinels,
            "integration_evaluation_jobs": integration_jobs,
        },
        outputs=(paths.control / "sentinel_gate.json",),
    )
    if bool(runtime.get("storage", {}).get("retain_all_checkpoints", False)):
        return gate_id
    return _add_job(
        jobs,
        job_id="retention_prune_sentinels",
        stage="retention",
        action="checkpoint_prune",
        resource="cpu",
        dependencies=(gate_id,),
        arguments={
            "source_jobs": sentinels,
            "evidence_jobs": [gate_id, *integration_jobs],
            "retention_scope": "sentinel_checkpoints_after_gate",
        },
        outputs=(paths.control / "prune" / "retention_prune_sentinels.json",),
    )
