from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..runtime import _dump_json, _load_json, _save_stage_state, _should_skip_stage, _stage_signature
from ...context import PipelineContext
from ...types import ArtifactSpec, StagePlan, StageResult


SUMMARY_DEFAULT_PRIMARY_METRIC = "final_loss"


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _load_optional_json(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.exists() else {}


def _summary_metric_value(row: dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    value = row.get(metric)
    return _as_number(value)


def _load_final_loss_row(ctx: PipelineContext, run_id: str, train_dir: Path) -> dict[str, Any]:
    final_loss_cfg = getattr(ctx, "profile", {}).get("final_loss", {})
    output_dir = ""
    if isinstance(final_loss_cfg, dict):
        output_dir = str(final_loss_cfg.get("output_dir", "")).strip()
    candidates: list[Path] = []
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = ctx.repo_root / path
        candidates.append(path / run_id / "final_loss.json")
    candidates.append(train_dir / "final_loss.json")
    for path in candidates:
        if path.exists():
            return _load_optional_json(path)
    return {}


def _benchmark_rows_by_run(ctx: PipelineContext) -> dict[str, dict[str, Any]]:
    summary_path = ctx.benchmark_root / "benchmark_summary.json"
    payload = _load_optional_json(summary_path)
    rows = payload.get("runs", [])
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("run_id", "")).strip():
            result[str(row["run_id"])] = row
    return result


def _baseline_manifest_path(ctx: PipelineContext) -> str:
    path = ctx.baseline_cache_root / "manifest.json"
    return str(path.resolve()) if path.exists() else ""


def _build_summary_rows(ctx: PipelineContext) -> list[dict[str, Any]]:
    runs = ctx.train_cfg.get("runs", [])
    if not isinstance(runs, list):
        raise SystemExit("summary requires training.runs to be a list")
    benchmark_rows = _benchmark_rows_by_run(ctx)
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = str(run["id"])
        train_dir = ctx.train_root / run_id
        train_results = _load_optional_json(train_dir / "train_results.json")
        trainer_state = _load_optional_json(train_dir / "trainer_state.json")
        final_loss = _load_final_loss_row(ctx, run_id, train_dir)
        benchmark = benchmark_rows.get(run_id, {})
        metrics = benchmark.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}
        final_loss_value = final_loss.get("final_loss", train_results.get("final_loss", train_results.get("train_loss")))
        row = {
            "run_id": run_id,
            "kind": run.get("kind"),
            "status": train_results.get("status"),
            "backend": train_results.get("backend"),
            "recipe": train_results.get("recipe"),
            "dataset": train_results.get("dataset"),
            "train_loss": train_results.get("train_loss"),
            "final_loss": final_loss_value,
            "final_loss_source": final_loss.get("source", train_results.get("final_loss_source")),
            "skipped_iterations": final_loss.get("skipped_iterations", train_results.get("skipped_iterations")),
            "nan_iterations": final_loss.get("nan_iterations", train_results.get("nan_iterations")),
            "train_runtime": train_results.get("train_runtime"),
            "num_input_tokens_seen": train_results.get("num_input_tokens_seen"),
            "global_step": final_loss.get("global_step", trainer_state.get("global_step")),
            "training_complete": (train_dir / ".training_complete").exists(),
            "benchmark_backend": metrics.get("backend", benchmark.get("backend")),
            "benchmark_tasks_seen": benchmark.get("num_tasks_seen"),
            "benchmark_tasks_expected": benchmark.get("num_tasks_expected"),
            "benchmark_missing_tasks": len(benchmark.get("missing_tasks", [])) if isinstance(benchmark.get("missing_tasks"), list) else "",
        }
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_gate(ctx: PipelineContext, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric = str(ctx.summary_cfg.get("primary_metric", SUMMARY_DEFAULT_PRIMARY_METRIC)).strip() or SUMMARY_DEFAULT_PRIMARY_METRIC
    baseline_id = str(ctx.summary_cfg.get("baseline_id", rows[0]["run_id"] if rows else "")).strip()
    min_rel_impr = float(ctx.summary_cfg.get("min_relative_improvement", 0.0))
    lower_is_better = bool(ctx.summary_cfg.get("lower_is_better", metric.endswith("loss") or metric.endswith("runtime")))
    baseline_row = next((row for row in rows if row["run_id"] == baseline_id), None)
    baseline_val = _summary_metric_value(baseline_row, metric) if baseline_row else None
    comparisons: list[dict[str, Any]] = []
    passing_runs: list[str] = []
    for row in rows:
        if row["run_id"] == baseline_id:
            continue
        value = _summary_metric_value(row, metric)
        rel = None
        passes_gate = False
        if value is not None and baseline_val is not None and baseline_val != 0:
            if lower_is_better:
                rel = (baseline_val - value) / abs(baseline_val)
            else:
                rel = (value - baseline_val) / abs(baseline_val)
            passes_gate = rel >= min_rel_impr
        if passes_gate:
            passing_runs.append(str(row["run_id"]))
        comparisons.append(
            {
                "run_id": row["run_id"],
                "metric_value": value,
                "baseline_value": baseline_val,
                "relative_improvement": rel,
                "passes_gate": passes_gate,
            }
        )
    return {
        "profile": ctx.args.profile,
        "primary_metric": metric,
        "lower_is_better": lower_is_better,
        "baseline_id": baseline_id,
        "baseline_value": baseline_val,
        "min_relative_improvement": min_rel_impr,
        "passing_runs": passing_runs,
        "comparisons": comparisons,
        "baseline_manifest": _baseline_manifest_path(ctx),
    }


def stage_summary(ctx: PipelineContext) -> None:
    if "summary" not in ctx.active_stages:
        return
    ctx.summary_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.gate_path.parent.mkdir(parents=True, exist_ok=True)
    signature = _stage_signature(
        {
            "stage": "summary",
            "cfg": ctx.summary_cfg,
            "train_cfg": ctx.train_cfg,
            "train_root": str(ctx.train_root),
            "benchmark_root": str(ctx.benchmark_root),
        }
    )
    expected = [ctx.summary_path, ctx.gate_path]
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="summary",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] summary stage (cache hit)")
        return

    rows = _build_summary_rows(ctx)
    gate = _build_gate(ctx, rows)
    if not ctx.args.dry_run:
        _write_summary_csv(ctx.summary_path, rows)
        _dump_json(ctx.gate_path, gate)
        ctx.lineage_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _dump_json(
            ctx.lineage_manifest_path,
            {
                "tokenized_path": str(ctx.tokenized_dir.resolve()) if ctx.tokenized_dir.exists() else "",
                "train_root": str(ctx.train_root.resolve()) if ctx.train_root.exists() else "",
                "benchmark_root": str(ctx.benchmark_root.resolve()) if ctx.benchmark_root.exists() else "",
                "baseline_cache_root": str(ctx.baseline_cache_root.resolve()) if ctx.baseline_cache_root.exists() else "",
            },
        )
        _save_stage_state(ctx.state_path, "summary", signature, [str(ctx.summary_path), str(ctx.gate_path)])
    ctx.logger.info("[OK] summary_csv=%s", ctx.summary_path)
    ctx.logger.info("[OK] gate_json=%s", ctx.gate_path)


class SummaryStage:
    name = "summary"

    def plan(self, ctx: Any) -> StagePlan:
        artifacts = ()
        summary_path = getattr(ctx, "summary_path", None)
        gate_path = getattr(ctx, "gate_path", None)
        if summary_path is not None and gate_path is not None:
            artifacts = (
                ArtifactSpec(summary_path, required=True, description="Pipeline summary CSV"),
                ArtifactSpec(gate_path, required=True, description="Pipeline gate JSON"),
            )
        return StagePlan(stage=self.name, enabled=True, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_summary(ctx)
        summary_path = getattr(ctx, "summary_path", None)
        gate_path = getattr(ctx, "gate_path", None)
        artifacts = tuple(path for path in (summary_path, gate_path) if path is not None)
        return StageResult(stage=self.name, artifacts=artifacts)

    def validate(self, ctx: Any, result: StageResult) -> None:
        if result.status == "skipped" or bool(getattr(getattr(ctx, "args", object()), "dry_run", False)):
            return
        for artifact in result.artifacts:
            if not artifact.exists():
                raise SystemExit(f"summary expected artifact missing: {artifact}")


stage = SummaryStage()

__all__ = ["SUMMARY_DEFAULT_PRIMARY_METRIC", "SummaryStage", "stage", "stage_summary"]
