"""Stage registry for the multilingual pretraining pipeline."""

from __future__ import annotations

from .stages.analysis_report.summary_stage import stage as summary_stage
from .stages.audit.audit_stage import stage as audit_stage
from .stages.benchmarking.baseline_eval_stage import stage as baseline_eval_stage
from .stages.benchmarking.benchmark_stage import stage as benchmark_stage
from .stages.checkpoint_conversion.checkpoint_conversion_stage import stage as checkpoint_conversion_stage
from .stages.cooldown_readout.cooldown_readout_stage import stage as cooldown_readout_stage
from .stages.final_loss.final_loss_stage import stage as final_loss_stage
from .stages.sampling.sample_stage import stage as sampling_stage
from .stages.tokenization.tokenize_stage import stage as tokenization_stage
from .stages.tokenizer_training.tokenizer_train_stage import stage as tokenizer_training_stage
from .stages.training.train_stage import stage as training_stage
from .types import PipelineStage, StagePlan, StageResult


STAGE_ORDER = [
    "sample",
    "tokenizer_train",
    "tokenize",
    "audit",
    "train",
    "cooldown_readout",
    "checkpoint_conversion",
    "final_loss",
    "benchmark",
    "baseline_eval",
    "summary",
]

_STAGES: dict[str, PipelineStage] = {
    sampling_stage.name: sampling_stage,
    tokenizer_training_stage.name: tokenizer_training_stage,
    tokenization_stage.name: tokenization_stage,
    audit_stage.name: audit_stage,
    training_stage.name: training_stage,
    cooldown_readout_stage.name: cooldown_readout_stage,
    checkpoint_conversion_stage.name: checkpoint_conversion_stage,
    final_loss_stage.name: final_loss_stage,
    benchmark_stage.name: benchmark_stage,
    baseline_eval_stage.name: baseline_eval_stage,
    summary_stage.name: summary_stage,
}


def get_stage(name: str) -> PipelineStage:
    try:
        return _STAGES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline stage: {name}") from exc


def stage_handlers() -> dict[str, PipelineStage]:
    return dict(_STAGES)


def run_stage(stage: PipelineStage, ctx: object) -> StageResult:
    plan = stage.plan(ctx)
    result = stage.run(ctx, plan)
    stage.validate(ctx, result)
    return result
