from __future__ import annotations

from typing import Any

from ..runtime import _missing_modules, _save_stage_state, _should_skip_stage, _stage_signature
from ...checks import check_train
from ...context import PipelineContext
from ...types import ArtifactSpec, StagePlan, StageResult
from .calibration import (
    calibration_enabled_for_any_run,
    resolve_training_calibration_settings,
    run_training_calibration,
    write_training_calibrations_manifest,
)
from .config import resolve_training_stage_config
from .planner import (
    TRAINING_CALIBRATION_NAME,
    TRAINING_CALIBRATION_TSV_NAME,
    TRAINING_CALIBRATIONS_NAME,
    TRAINING_MANIFEST_NAME,
    TRAINING_RUN_MANIFEST_NAME,
    TrainingRunPlan,
    build_training_run_plan,
    core_run_artifacts_exist,
    run_is_complete,
    write_model_tokenizer_lineage,
    write_training_command_plan,
    write_training_manifest,
    write_training_run_manifest,
)


def _run_training_plan(ctx: PipelineContext, plan: TrainingRunPlan) -> None:
    ctx.executor.run(plan.cmd, plan.env, ctx.args.dry_run, plan.log_path)


def _write_success_artifacts(
    ctx: PipelineContext,
    *,
    plan: TrainingRunPlan,
    resolved_model_path: str,
) -> None:
    write_model_tokenizer_lineage(
        plan=plan,
        model_name_or_path=resolved_model_path,
        tokenized_path=ctx.tokenized_dir,
    )
    write_training_run_manifest(
        plan=plan,
        status="complete",
        model_name_or_path=resolved_model_path,
        tokenized_path=ctx.tokenized_dir,
    )


def _training_expected_artifacts(
    *,
    ctx: PipelineContext,
    plans: list[TrainingRunPlan],
    runs: list[dict[str, Any]],
    common: dict[str, Any],
) -> list[Any]:
    expected: list[Any] = []
    calibration_enabled = False
    for plan, run in zip(plans, runs):
        settings = resolve_training_calibration_settings(common=common, run_cfg=run)
        if settings.enabled:
            calibration_enabled = True
            expected.append(plan.output_dir / TRAINING_CALIBRATION_NAME)
            expected.append(plan.output_dir / TRAINING_CALIBRATION_TSV_NAME)
            if settings.mode == "calibrate_then_train":
                expected.extend(plan.expected_artifacts)
        else:
            expected.extend(plan.expected_artifacts)
    expected.append(ctx.train_root / TRAINING_MANIFEST_NAME)
    if calibration_enabled:
        expected.append(ctx.train_root / TRAINING_CALIBRATIONS_NAME)
    return expected


def stage_train(ctx: PipelineContext) -> None:
    if "train" not in ctx.active_stages or not bool(ctx.train_cfg.get("enabled", True)):
        return

    backend = str(ctx.train_cfg.get("backend", "")).strip().lower()
    required_modules = ["datasets", "transformers", "torch"] if backend not in {"megatron_bridge", "bridge"} else []
    missing = _missing_modules(required_modules)
    if missing:
        raise SystemExit(f"Missing Python modules for train stage: {missing}")

    train_cfg_for_stage, resolved_model_path, runs = resolve_training_stage_config(ctx)
    ctx.train_root.mkdir(parents=True, exist_ok=True)
    plans = [build_training_run_plan(ctx, run=run, common=train_cfg_for_stage) for run in runs]
    calibration_enabled = calibration_enabled_for_any_run(train_cfg_for_stage, runs)
    signature = _stage_signature(
        {
            "stage": "train",
            "cfg": train_cfg_for_stage,
            "tokenized_dir": str(ctx.tokenized_dir),
        }
    )
    expected = _training_expected_artifacts(ctx=ctx, plans=plans, runs=runs, common=train_cfg_for_stage)
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="train",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] train stage (cache hit)")
        return

    manifest_plans = list(plans)
    final_runs = list(runs)
    if calibration_enabled:
        calibration_results = []
        final_runs = []
        calibration_only_plans: list[TrainingRunPlan] = []
        failed_calibrations: list[str] = []
        for run, base_plan in zip(runs, plans):
            settings = resolve_training_calibration_settings(common=train_cfg_for_stage, run_cfg=run)
            if not settings.enabled:
                final_runs.append(run)
                continue
            result = run_training_calibration(ctx, run_cfg=run, common=train_cfg_for_stage, base_plan=base_plan)
            calibration_results.append(result)
            if ctx.args.dry_run:
                continue
            if settings.mode == "calibrate_only":
                write_training_run_manifest(
                    plan=base_plan,
                    status="calibration_only",
                    model_name_or_path=resolved_model_path,
                    tokenized_path=ctx.tokenized_dir,
                )
                calibration_only_plans.append(base_plan)
                continue
            if result.selected_run_cfg is None:
                write_training_run_manifest(
                    plan=base_plan,
                    status="calibration_failed",
                    model_name_or_path=resolved_model_path,
                    tokenized_path=ctx.tokenized_dir,
                )
                failed_calibrations.append(result.run_id)
                continue
            final_runs.append(result.selected_run_cfg)

        calibrations_manifest = write_training_calibrations_manifest(
            train_root=ctx.train_root,
            results=calibration_results,
        )
        if ctx.args.dry_run:
            return
        if failed_calibrations:
            write_training_manifest(train_root=ctx.train_root, plans=plans)
            raise SystemExit(f"training calibration failed to select a candidate for runs: {failed_calibrations}")
        if not final_runs:
            manifest_path = write_training_manifest(train_root=ctx.train_root, plans=plans)
            _save_stage_state(
                ctx.state_path,
                "train",
                signature,
                [str(path) for path in expected] + [str(manifest_path), str(calibrations_manifest)],
            )
            return
        plans = [build_training_run_plan(ctx, run=run, common=train_cfg_for_stage) for run in final_runs]
        manifest_plans = list(plans) + calibration_only_plans

    pending: list[TrainingRunPlan] = []
    for plan in plans:
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        write_training_command_plan(plan)
        if (
            not calibration_enabled
            and "train" not in ctx.force_stages
            and not ctx.args.dry_run
            and run_is_complete(plan)
        ):
            ctx.logger.info("[SKIP] train run %s (already complete)", plan.run_id)
            continue
        pending.append(plan)

    if pending and hasattr(ctx.executor, "run_batch") and len(pending) > 1:
        jobs = [(plan.cmd, plan.env, plan.log_path) for plan in pending]
        ctx.logger.info("[BATCH] Submitting %d training runs via %s", len(jobs), type(ctx.executor).__name__)
        ctx.executor.run_batch(jobs, dry_run=ctx.args.dry_run)
    else:
        for plan in pending:
            _run_training_plan(ctx, plan)

    if not ctx.args.dry_run:
        for plan in pending:
            if core_run_artifacts_exist(plan):
                _write_success_artifacts(ctx, plan=plan, resolved_model_path=resolved_model_path)
            else:
                write_training_run_manifest(
                    plan=plan,
                    status="missing_artifacts",
                    model_name_or_path=resolved_model_path,
                    tokenized_path=ctx.tokenized_dir,
                )
        for plan in manifest_plans:
            if plan not in pending and run_is_complete(plan):
                _write_success_artifacts(ctx, plan=plan, resolved_model_path=resolved_model_path)
        manifest_path = write_training_manifest(train_root=ctx.train_root, plans=manifest_plans)
        _save_stage_state(ctx.state_path, "train", signature, [str(path) for path in expected] + [str(manifest_path)])
        check_train(ctx.train_root, final_runs, dry_run=ctx.args.dry_run)


class TrainingStage:
    name = "train"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = getattr(ctx, "train_cfg", {})
        enabled = bool(cfg.get("enabled", True)) if isinstance(cfg, dict) else True
        train_root = getattr(ctx, "train_root", None)
        artifacts = ()
        if enabled and train_root is not None:
            artifacts = (
                ArtifactSpec(train_root / TRAINING_MANIFEST_NAME, required=True, description="Aggregate training manifest"),
            )
        return StagePlan(stage=self.name, enabled=enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_train(ctx)
        train_root = getattr(ctx, "train_root", None)
        artifacts = (train_root / TRAINING_MANIFEST_NAME,) if train_root is not None else ()
        return StageResult(stage=self.name, artifacts=artifacts)

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = TrainingStage()

__all__ = ["TrainingStage", "stage", "stage_train"]
