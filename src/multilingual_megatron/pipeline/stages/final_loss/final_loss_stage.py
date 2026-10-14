from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime import (
    _dump_json,
    _load_json,
    _log_path_component,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
    _structured_logs_enabled,
)
from ...context import PipelineContext
from ...types import ArtifactSpec, StagePlan, StageResult


FINAL_LOSS_MANIFEST = "final_loss.json"
FINAL_LOSS_INDEX = "final_losses.json"

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_ITERATION_RE = re.compile(r"iteration\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)", re.IGNORECASE)
_LM_LOSS_RE = re.compile(r"\blm loss:\s*(?P<value>" + _FLOAT_RE + r")", re.IGNORECASE)
_SKIPPED_RE = re.compile(r"number of skipped iterations:\s*(?P<value>\d+)", re.IGNORECASE)
_NAN_RE = re.compile(r"number of nan iterations:\s*(?P<value>\d+)", re.IGNORECASE)
_CONSUMED_RE = re.compile(r"consumed samples:\s*(?P<value>\d+)", re.IGNORECASE)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass(frozen=True, slots=True)
class FinalLossConfig:
    enabled: bool
    output_dir: Path
    require_training_complete: bool
    allow_missing: bool
    inject_train_results: bool
    log_globs: tuple[str, ...]


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def resolve_final_loss_config(ctx: PipelineContext) -> FinalLossConfig:
    raw_cfg = getattr(ctx, "profile", {}).get("final_loss", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    repo_root = Path(getattr(ctx, "repo_root", Path.cwd()))
    output_root = Path(getattr(ctx, "output_root", repo_root / "outputs" / "pipeline_runs" / "default"))
    output_dir = _resolve_path(repo_root, str(raw_cfg.get("output_dir", output_root / "final_loss")).strip())
    raw_globs = raw_cfg.get("log_globs", [])
    if isinstance(raw_globs, str):
        log_globs = (raw_globs,)
    elif isinstance(raw_globs, list):
        log_globs = tuple(str(item).strip() for item in raw_globs if str(item).strip())
    else:
        log_globs = ()
    return FinalLossConfig(
        enabled=_as_bool(raw_cfg.get("enabled", False)),
        output_dir=output_dir,
        require_training_complete=_as_bool(raw_cfg.get("require_training_complete", True)),
        allow_missing=_as_bool(raw_cfg.get("allow_missing", False)),
        inject_train_results=_as_bool(raw_cfg.get("inject_train_results", True)),
        log_globs=log_globs,
    )


def _selected_runs(ctx: PipelineContext) -> list[dict[str, Any]]:
    runs = ctx.train_cfg.get("runs", [])
    if not isinstance(runs, list):
        raise SystemExit("final_loss requires training.runs to be a list")
    selected_ids = {str(item) for item in getattr(ctx.args, "run_id", []) if str(item).strip()}
    if not selected_ids:
        return [dict(run) for run in runs if isinstance(run, dict)]
    return [dict(run) for run in runs if isinstance(run, dict) and str(run.get("id", "")) in selected_ids]


def _metric_from_line(line: str) -> dict[str, Any] | None:
    loss_match = _LM_LOSS_RE.search(line)
    if loss_match is None:
        return None
    step_match = _ITERATION_RE.search(line)
    skipped_match = _SKIPPED_RE.search(line)
    nan_match = _NAN_RE.search(line)
    consumed_match = _CONSUMED_RE.search(line)
    return {
        "final_loss": float(loss_match.group("value")),
        "global_step": int(step_match.group("step")) if step_match else None,
        "total_iterations": int(step_match.group("total")) if step_match else None,
        "consumed_samples": int(consumed_match.group("value")) if consumed_match else None,
        "skipped_iterations": int(skipped_match.group("value")) if skipped_match else None,
        "nan_iterations": int(nan_match.group("value")) if nan_match else None,
    }


def _candidate_log_paths(ctx: PipelineContext, cfg: FinalLossConfig, run_id: str) -> list[Path]:
    paths: list[Path] = []
    run_manifest = ctx.train_root / run_id / "training_run_manifest.json"
    if run_manifest.exists():
        try:
            manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
        if isinstance(metadata, dict):
            for segment in metadata.get("segments", []):
                if isinstance(segment, dict) and str(segment.get("log_path", "")).strip():
                    paths.append(Path(str(segment["log_path"])))
    run_component = _log_path_component(run_id)
    env = getattr(ctx, "env", {})
    logs_base_raw = str(env.get("PROOF_PIPELINE_LOGS_BASE_ROOT", "")).strip() if isinstance(env, dict) else ""
    logs_base_root = Path(logs_base_raw) if logs_base_raw else None
    structured_patterns: list[str] = []
    if isinstance(env, dict) and _structured_logs_enabled(env):
        structured_patterns.extend(
            [
                str(ctx.logs_root / "stages" / "train" / run_component / "*.log"),
                str(ctx.logs_root / "stages" / "train" / run_component / "*.slurm.log"),
                str(ctx.logs_root / "pipeline.log"),
            ]
        )
        if logs_base_root is not None:
            structured_patterns.extend(
                [
                    str(logs_base_root / "attempts" / "*" / "stages" / "train" / run_component / "*.log"),
                    str(logs_base_root / "attempts" / "*" / "stages" / "train" / run_component / "*.slurm.log"),
                ]
            )
    default_patterns = [
        *structured_patterns,
        str(ctx.logs_root / "train_runs" / run_id / "*.train.log"),
        str(ctx.logs_root / "train_runs" / run_id / "*.train.slurm.log"),
        str(ctx.logs_root / f"train*.{run_id}*.log"),
        str(ctx.logs_root / "pipeline_run_*.log"),
    ]
    for pattern in (*default_patterns, *cfg.log_globs):
        pattern_path = Path(pattern)
        if not pattern_path.is_absolute():
            pattern_path = ctx.repo_root / pattern_path
        paths.extend(Path(path) for path in sorted(glob.glob(str(pattern_path))))
    unique = {path.resolve(): path for path in paths if path.exists() and path.is_file()}
    return sorted(unique.values(), key=lambda path: (path.stat().st_mtime, str(path)))


def _extract_from_logs(ctx: PipelineContext, cfg: FinalLossConfig, run_id: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_source = ""
    for path in _candidate_log_paths(ctx, cfg, run_id):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            parsed = _metric_from_line(line)
            if parsed is None:
                continue
            parsed["source"] = "log"
            parsed["source_path"] = str(path.resolve())
            parsed["source_line"] = line.strip()
            current_step = parsed.get("global_step")
            latest_step = latest.get("global_step") if latest else None
            if latest is None or (
                isinstance(current_step, int)
                and (not isinstance(latest_step, int) or current_step >= latest_step)
            ):
                latest = parsed
                latest_source = str(path.resolve())
    if latest is not None:
        latest["source_path"] = latest_source
    return latest


def _latest_history_metric(history: Any, keys: tuple[str, ...]) -> tuple[float | None, int | None]:
    if not isinstance(history, list):
        return None, None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        step = entry.get("step", entry.get("global_step"))
        step_value = int(step) if isinstance(step, int) and not isinstance(step, bool) else None
        for key, value in entry.items():
            key_text = str(key)
            if key_text in keys or any(key_text.endswith("/" + suffix) for suffix in keys):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value), step_value
    return None, None


def _extract_from_artifacts(train_dir: Path) -> dict[str, Any] | None:
    results_path = train_dir / "train_results.json"
    state_path = train_dir / "trainer_state.json"
    results = _load_json(results_path) if results_path.exists() else {}
    trainer_state = _load_json(state_path) if state_path.exists() else {}
    loss, step = _latest_history_metric(
        trainer_state.get("log_history"),
        ("lm loss", "lm_loss", "loss", "train_loss", "eval_loss"),
    )
    if loss is None:
        for key in ("train_loss", "final_loss", "eval_loss"):
            value = results.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                loss = float(value)
                break
    global_step = trainer_state.get("global_step", step)
    if not isinstance(global_step, int) or isinstance(global_step, bool):
        global_step = step
    if loss is None and global_step is None:
        return None
    return {
        "final_loss": loss,
        "global_step": global_step,
        "total_iterations": None,
        "consumed_samples": None,
        "skipped_iterations": None,
        "nan_iterations": None,
        "source": "training_artifacts",
        "source_path": str(state_path.resolve() if state_path.exists() else results_path.resolve()),
        "source_line": "",
    }


def _inject_train_results(train_dir: Path, manifest_path: Path, metric: dict[str, Any]) -> None:
    results_path = train_dir / "train_results.json"
    payload = _load_json(results_path) if results_path.exists() else {}
    payload["train_loss"] = metric.get("final_loss")
    payload["final_loss"] = metric.get("final_loss")
    payload["final_loss_source"] = metric.get("source")
    payload["final_loss_manifest"] = str(manifest_path.resolve())
    skipped = metric.get("skipped_iterations")
    nan_iterations = metric.get("nan_iterations")
    if skipped is not None:
        payload["skipped_iterations"] = skipped
    if nan_iterations is not None:
        payload["nan_iterations"] = nan_iterations
    _dump_json(results_path, payload)


def _write_run_manifest(
    *,
    run_id: str,
    train_dir: Path,
    output_dir: Path,
    metric: dict[str, Any],
    status: str,
) -> Path:
    path = output_dir / run_id / FINAL_LOSS_MANIFEST
    payload = {
        "schema_version": 1,
        "stage": "final_loss",
        "run_id": run_id,
        "status": status,
        "train_dir": str(train_dir.resolve()),
        "final_loss": metric.get("final_loss"),
        "global_step": metric.get("global_step"),
        "total_iterations": metric.get("total_iterations"),
        "consumed_samples": metric.get("consumed_samples"),
        "skipped_iterations": metric.get("skipped_iterations"),
        "nan_iterations": metric.get("nan_iterations"),
        "source": metric.get("source"),
        "source_path": metric.get("source_path"),
        "source_line": metric.get("source_line"),
    }
    _dump_json(path, payload)
    return path


def stage_final_loss(ctx: PipelineContext) -> None:
    if "final_loss" not in ctx.active_stages:
        return
    cfg = resolve_final_loss_config(ctx)
    if not cfg.enabled:
        return

    runs = _selected_runs(ctx)
    expected = [cfg.output_dir / str(run["id"]) / FINAL_LOSS_MANIFEST for run in runs]
    expected.append(cfg.output_dir / FINAL_LOSS_INDEX)
    env_for_signature = getattr(ctx, "env", {})
    logs_signature_root = (
        env_for_signature.get("PROOF_PIPELINE_LOGS_BASE_ROOT", ctx.logs_root)
        if isinstance(env_for_signature, dict)
        else ctx.logs_root
    )
    signature = _stage_signature(
        {
            "stage": "final_loss",
            "cfg": getattr(ctx, "profile", {}).get("final_loss", {}),
            "runs": [run.get("id") for run in runs],
            "train_root": str(ctx.train_root),
            "logs_root": str(logs_signature_root),
        }
    )
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="final_loss",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] final_loss stage (cache hit)")
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    index_runs: dict[str, Any] = {}
    for run in runs:
        run_id = str(run["id"])
        train_dir = ctx.train_root / run_id
        done_path = train_dir / ".training_complete"
        if cfg.require_training_complete and not done_path.exists() and not ctx.args.dry_run:
            raise SystemExit(f"final_loss requires completed training for run '{run_id}': {done_path}")
        if ctx.args.dry_run:
            index_runs[run_id] = {"status": "dry_run", "final_loss": None}
            continue

        metric = _extract_from_logs(ctx, cfg, run_id) or _extract_from_artifacts(train_dir)
        if metric is None or metric.get("final_loss") is None:
            if not cfg.allow_missing:
                raise SystemExit(f"final_loss could not resolve a loss metric for run '{run_id}'")
            metric = {
                "final_loss": None,
                "global_step": None,
                "total_iterations": None,
                "consumed_samples": None,
                "skipped_iterations": None,
                "nan_iterations": None,
                "source": "missing",
                "source_path": "",
                "source_line": "",
            }
        status = "metric_extracted" if metric.get("final_loss") is not None else "metric_missing"
        manifest_path = _write_run_manifest(
            run_id=run_id,
            train_dir=train_dir,
            output_dir=cfg.output_dir,
            metric=metric,
            status=status,
        )
        if cfg.inject_train_results:
            _inject_train_results(train_dir, manifest_path, metric)
        index_runs[run_id] = {
            "status": status,
            "final_loss": metric.get("final_loss"),
            "global_step": metric.get("global_step"),
            "manifest": str(manifest_path.resolve()),
            "source": metric.get("source"),
        }

    if not ctx.args.dry_run:
        _dump_json(
            cfg.output_dir / FINAL_LOSS_INDEX,
            {
                "schema_version": 1,
                "stage": "final_loss",
                "output_dir": str(cfg.output_dir.resolve()),
                "runs": index_runs,
            },
        )
        _save_stage_state(ctx.state_path, "final_loss", signature, [str(path) for path in expected])
    ctx.logger.info("[OK] final_losses=%s", cfg.output_dir / FINAL_LOSS_INDEX)


class FinalLossStage:
    name = "final_loss"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = resolve_final_loss_config(ctx)
        artifacts = ()
        if cfg.enabled:
            artifacts = (
                ArtifactSpec(cfg.output_dir / FINAL_LOSS_INDEX, required=True, description="Final-loss index"),
            )
        return StagePlan(stage=self.name, enabled=cfg.enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_final_loss(ctx)
        cfg = resolve_final_loss_config(ctx)
        return StageResult(stage=self.name, artifacts=(cfg.output_dir / FINAL_LOSS_INDEX,))

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = FinalLossStage()

__all__ = [
    "FINAL_LOSS_INDEX",
    "FINAL_LOSS_MANIFEST",
    "FinalLossConfig",
    "FinalLossStage",
    "resolve_final_loss_config",
    "stage",
    "stage_final_loss",
]
