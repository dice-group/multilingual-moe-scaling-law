from __future__ import annotations

from typing import Any

from ..common import ensure_lm_eval_layout
from ..runtime import _abs, _run_stage_cmd, _save_stage_state, _should_skip_stage, _stage_signature
from ...context import PipelineContext
from ...types import ArtifactSpec, StagePlan, StageResult


def build_baseline_eval_command(ctx: PipelineContext) -> list[str]:
    models = ctx.baseline_eval_cfg.get(
        "models", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]
    )
    cmd = [
        "python",
        "-m",
        "multilingual_megatron.pipeline.stages.benchmarking.tools.run_base_model_eval_cache",
        "--models",
        ",".join((str(x) for x in models)),
        "--tasks-file",
        str(_abs(ctx.repo_root, str(ctx.baseline_eval_cfg.get("tasks_file", "configs/lm_eval/lm_eval_tasks.txt")))),
        "--output-root",
        str(ctx.baseline_cache_root),
        "--batch-size",
        str(ctx.baseline_eval_cfg.get("batch_size", "auto")),
        "--lm-eval-bin",
        str(ctx.baseline_eval_cfg.get("lm_eval_bin", "lm_eval")),
        "--seed",
        str(int(ctx.baseline_eval_cfg.get("seed", 42))),
        "--extra-args",
        str(ctx.baseline_eval_cfg.get("extra_args", "")),
    ]
    task_prefixes = ctx.baseline_eval_cfg.get("task_prefixes", [])
    if isinstance(task_prefixes, list):
        for prefix in task_prefixes:
            value = str(prefix).strip()
            if value:
                cmd.extend(["--task-prefix", value])
    single_manifest = str(ctx.baseline_eval_cfg.get("precomputed_manifest", "")).strip()
    if single_manifest:
        cmd.extend(["--precomputed-manifest", str(_abs(ctx.repo_root, single_manifest))])
    manifest_list = ctx.baseline_eval_cfg.get("precomputed_manifests", [])
    if isinstance(manifest_list, list):
        for manifest in manifest_list:
            value = str(manifest).strip()
            if value:
                cmd.extend(["--precomputed-manifest", str(_abs(ctx.repo_root, value))])
    if bool(ctx.baseline_eval_cfg.get("force", False)):
        cmd.append("--force")
    return cmd


def stage_baseline_eval(ctx: PipelineContext) -> None:
    if "baseline_eval" in ctx.active_stages and bool(ctx.baseline_eval_cfg.get("enabled", False)):
        signature = _stage_signature({"stage": "baseline_eval", "cfg": ctx.baseline_eval_cfg})
        baseline_manifest = ctx.baseline_cache_root / "manifest.json"
        expected = [baseline_manifest]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="baseline_eval",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] baseline_eval stage (cache hit)")
        else:
            ensure_lm_eval_layout(ctx.repo_root, ctx.logger)
            cmd = build_baseline_eval_command(ctx)
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="baseline_eval", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "baseline_eval", signature, [str(baseline_manifest)])
            ctx.logger.info("[OK] baseline_manifest=%s", baseline_manifest)


class BaselineEvalStage:
    name = "baseline_eval"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = getattr(ctx, "baseline_eval_cfg", {})
        enabled = bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else False
        baseline_root = getattr(ctx, "baseline_cache_root", None)
        artifacts = ()
        if enabled and baseline_root is not None:
            artifacts = (
                ArtifactSpec(baseline_root / "manifest.json", required=True, description="Baseline eval cache manifest"),
            )
        return StagePlan(stage=self.name, enabled=enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_baseline_eval(ctx)
        baseline_root = getattr(ctx, "baseline_cache_root", None)
        artifacts = (baseline_root / "manifest.json",) if baseline_root is not None else ()
        return StageResult(stage=self.name, artifacts=artifacts)

    def validate(self, ctx: Any, result: StageResult) -> None:
        if result.status == "skipped" or bool(getattr(getattr(ctx, "args", object()), "dry_run", False)):
            return
        for artifact in result.artifacts:
            if not artifact.exists():
                raise SystemExit(f"baseline_eval expected artifact missing: {artifact}")


stage = BaselineEvalStage()

__all__ = ["BaselineEvalStage", "build_baseline_eval_command", "stage", "stage_baseline_eval"]
