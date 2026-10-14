from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime import _dump_json, _run_stage_cmd, _save_stage_state, _should_skip_stage, _stage_signature
from ...context import PipelineContext
from ...types import ArtifactSpec, StagePlan, StageResult


CHECKPOINT_EXPORT_MANIFEST = "checkpoint_export_manifest.json"
CHECKPOINT_EXPORT_INDEX = "checkpoint_exports.json"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@dataclass(frozen=True, slots=True)
class CheckpointConversionConfig:
    enabled: bool
    output_dir: Path
    mode: str
    command: str
    require_training_complete: bool
    hf_reference_model: str
    hf_export_subdir: str
    conversion_module: str
    conversion_script: Path | None
    trust_remote_code: bool
    strict: bool


def resolve_checkpoint_conversion_config(ctx: PipelineContext) -> CheckpointConversionConfig:
    raw_cfg = getattr(ctx, "profile", {}).get("checkpoint_conversion", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    repo_root = Path(getattr(ctx, "repo_root", Path.cwd()))
    output_root = Path(getattr(ctx, "output_root", repo_root / "outputs" / "pipeline_runs" / "default"))
    output_dir_raw = str(raw_cfg.get("output_dir", output_root / "checkpoint_exports")).strip()
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    mode = str(raw_cfg.get("mode", "manifest")).strip().lower() or "manifest"
    if mode not in {"manifest", "command", "bridge_hf_export"}:
        raise SystemExit(
            "checkpoint_conversion.mode must be 'manifest', 'command', or 'bridge_hf_export' "
            f"(got {mode!r})"
        )
    command = str(raw_cfg.get("command", "")).strip()
    if mode == "command" and not command:
        raise SystemExit("checkpoint_conversion.mode=command requires checkpoint_conversion.command")
    hf_reference_model = str(
        raw_cfg.get(
            "hf_reference_model",
            getattr(ctx, "tokenize_cfg", {}).get("model_name", ""),
        )
    ).strip()
    if mode == "bridge_hf_export" and not hf_reference_model:
        raise SystemExit("checkpoint_conversion.mode=bridge_hf_export requires checkpoint_conversion.hf_reference_model")
    conversion_module = str(
        raw_cfg.get(
            "conversion_module",
            "multilingual_megatron.pipeline.backends.megatron_bridge.run_bridge_hf_export",
        )
    ).strip()
    conversion_script_raw = str(raw_cfg.get("conversion_script", "")).strip()
    conversion_script = Path(conversion_script_raw) if conversion_script_raw else None
    if conversion_script is not None and not conversion_script.is_absolute():
        conversion_script = repo_root / conversion_script
    return CheckpointConversionConfig(
        enabled=_as_bool(raw_cfg.get("enabled", False)),
        output_dir=output_dir,
        mode=mode,
        command=command,
        require_training_complete=_as_bool(raw_cfg.get("require_training_complete", True)),
        hf_reference_model=hf_reference_model,
        hf_export_subdir=str(raw_cfg.get("hf_export_subdir", "hf")).strip() or "hf",
        conversion_module=conversion_module,
        conversion_script=conversion_script,
        trust_remote_code=_as_bool(raw_cfg.get("trust_remote_code", False)),
        strict=_as_bool(raw_cfg.get("strict", True)),
    )


def _selected_runs(ctx: PipelineContext) -> list[dict[str, Any]]:
    runs = ctx.train_cfg.get("runs", [])
    if not isinstance(runs, list):
        raise SystemExit("checkpoint_conversion requires training.runs to be a list")
    selected_ids = {str(item) for item in getattr(ctx.args, "run_id", []) if str(item).strip()}
    if not selected_ids:
        return [dict(run) for run in runs if isinstance(run, dict)]
    return [dict(run) for run in runs if isinstance(run, dict) and str(run.get("id", "")) in selected_ids]


def _latest_checkpoint_path(train_dir: Path) -> Path:
    run_manifest = train_dir / "training_run_manifest.json"
    if run_manifest.exists():
        try:
            manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
        if isinstance(metadata, dict):
            final_checkpoint_root = str(metadata.get("final_checkpoint_root", "")).strip()
            final_checkpoint_step = metadata.get("final_checkpoint_step")
            if final_checkpoint_root and final_checkpoint_step not in (None, ""):
                step = int(final_checkpoint_step)
                root = Path(final_checkpoint_root)
                for candidate in (
                    root / f"iter_{step:07d}",
                    root / f"iter_{step:010d}",
                    root / f"iter_{step}",
                ):
                    if candidate.exists():
                        return candidate
                return root / f"iter_{step:07d}"
    checkpoint_dir = train_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return checkpoint_dir
    if (checkpoint_dir / "run_config.yaml").exists():
        return checkpoint_dir

    def _iteration_key(path: Path) -> tuple[int, str]:
        suffix = path.name.replace("iter_", "", 1)
        return (int(suffix) if suffix.isdigit() else -1, path.name)

    iter_dirs = sorted(
        [path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.startswith("iter_")],
        key=_iteration_key,
    )
    if iter_dirs:
        return iter_dirs[-1]
    candidates = sorted([path for path in checkpoint_dir.iterdir() if path.is_dir()])
    return candidates[-1] if candidates else checkpoint_dir


def _export_status(latest_checkpoint: Path) -> str:
    return "checkpoint_exported" if latest_checkpoint.exists() else "metadata_exported"


def _hf_export_status(hf_export_dir: Path) -> str:
    return "hf_exported" if hf_export_dir.exists() and any(hf_export_dir.iterdir()) else "hf_export_missing"


def _format_command(template: str, *, run_id: str, train_dir: Path, export_dir: Path, latest_checkpoint: Path) -> list[str]:
    formatted = template.format(
        run_id=run_id,
        train_dir=str(train_dir),
        export_dir=str(export_dir),
        latest_checkpoint=str(latest_checkpoint),
    )
    return shlex.split(formatted)


def _bridge_hf_export_command(cfg: CheckpointConversionConfig, *, latest_checkpoint: Path, hf_export_dir: Path) -> list[str]:
    cmd = [
        "python",
    ]
    if cfg.conversion_script is not None:
        cmd.append(str(cfg.conversion_script))
    else:
        cmd.extend(["-m", cfg.conversion_module])
    cmd.extend(
        [
            "export",
            "--hf-model",
            cfg.hf_reference_model,
            "--megatron-path",
            str(latest_checkpoint),
            "--hf-path",
            str(hf_export_dir),
        ]
    )
    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    if not cfg.strict:
        cmd.append("--not-strict")
    return cmd


def _write_export_manifest(
    *,
    cfg: CheckpointConversionConfig,
    run_id: str,
    run_cfg: dict[str, Any],
    train_dir: Path,
    export_dir: Path,
    latest_checkpoint: Path,
    status: str,
    hf_export_dir: Path | None = None,
    command: list[str] | None = None,
) -> Path:
    manifest_path = export_dir / CHECKPOINT_EXPORT_MANIFEST
    source_checkpoint_dir = latest_checkpoint.parent if latest_checkpoint.name.startswith("iter_") else train_dir / "checkpoints"
    payload = {
        "schema_version": 1,
        "stage": "checkpoint_conversion",
        "run_id": run_id,
        "status": status,
        "mode": cfg.mode,
        "source_train_dir": str(train_dir.resolve()),
        "source_checkpoint_dir": str(source_checkpoint_dir.resolve()),
        "checkpoint_dir_exists": source_checkpoint_dir.exists(),
        "latest_checkpoint": str(latest_checkpoint.resolve()) if latest_checkpoint.exists() else None,
        "latest_checkpoint_exists": latest_checkpoint.exists(),
        "export_dir": str(export_dir.resolve()),
        "hf_export_dir": str(hf_export_dir.resolve()) if hf_export_dir is not None else "",
        "hf_export_exists": bool(hf_export_dir is not None and hf_export_dir.exists() and any(hf_export_dir.iterdir())),
        "hf_reference_model": cfg.hf_reference_model,
        "export_command": command or [],
        "train_results": str((train_dir / "train_results.json").resolve()),
        "trainer_state": str((train_dir / "trainer_state.json").resolve()),
        "training_run_manifest": str((train_dir / "training_run_manifest.json").resolve()),
        "recipe": run_cfg.get("recipe", ""),
        "backend": run_cfg.get("backend", "megatron_bridge"),
    }
    _dump_json(manifest_path, payload)
    return manifest_path


def stage_checkpoint_conversion(ctx: PipelineContext) -> None:
    if "checkpoint_conversion" not in ctx.active_stages:
        return
    cfg = resolve_checkpoint_conversion_config(ctx)
    if not cfg.enabled:
        return

    runs = _selected_runs(ctx)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    expected = [cfg.output_dir / str(run["id"]) / CHECKPOINT_EXPORT_MANIFEST for run in runs]
    expected.append(cfg.output_dir / CHECKPOINT_EXPORT_INDEX)
    signature = _stage_signature(
        {
            "stage": "checkpoint_conversion",
            "cfg": getattr(ctx, "profile", {}).get("checkpoint_conversion", {}),
            "runs": [run.get("id") for run in runs],
            "train_root": str(ctx.train_root),
        }
    )
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="checkpoint_conversion",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] checkpoint_conversion stage (cache hit)")
        return

    index_runs: dict[str, Any] = {}
    for run in runs:
        run_id = str(run["id"])
        train_dir = ctx.train_root / run_id
        done_path = train_dir / ".training_complete"
        if cfg.require_training_complete and not done_path.exists() and not ctx.args.dry_run:
            raise SystemExit(f"checkpoint_conversion requires completed training for run '{run_id}': {done_path}")
        export_dir = cfg.output_dir / run_id
        export_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint = _latest_checkpoint_path(train_dir)
        status = _export_status(latest_checkpoint)
        command: list[str] | None = None
        hf_export_dir: Path | None = None
        if cfg.mode == "command":
            command = _format_command(
                cfg.command,
                run_id=run_id,
                train_dir=train_dir,
                export_dir=export_dir,
                latest_checkpoint=latest_checkpoint,
            )
            _run_stage_cmd(
                cmd=command,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="checkpoint_conversion",
                suffix=run_id,
                executor=ctx.executor,
            )
        elif cfg.mode == "bridge_hf_export":
            if not latest_checkpoint.exists() and not ctx.args.dry_run:
                raise SystemExit(f"bridge_hf_export requires a saved checkpoint for run '{run_id}': {latest_checkpoint}")
            if cfg.conversion_script is not None and not cfg.conversion_script.exists() and not ctx.args.dry_run:
                raise SystemExit(f"Bridge conversion script not found: {cfg.conversion_script}")
            hf_export_dir = export_dir / cfg.hf_export_subdir
            command = _bridge_hf_export_command(cfg, latest_checkpoint=latest_checkpoint, hf_export_dir=hf_export_dir)
            _run_stage_cmd(
                cmd=command,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="checkpoint_conversion",
                suffix=f"{run_id}.bridge_hf_export",
                executor=ctx.executor,
            )
            if not ctx.args.dry_run:
                status = _hf_export_status(hf_export_dir)
                if status != "hf_exported":
                    raise SystemExit(f"Bridge HF export did not create artifacts for run '{run_id}': {hf_export_dir}")
        if not ctx.args.dry_run:
            manifest_path = _write_export_manifest(
                cfg=cfg,
                run_id=run_id,
                run_cfg=run,
                train_dir=train_dir,
                export_dir=export_dir,
                latest_checkpoint=latest_checkpoint,
                status=status,
                hf_export_dir=hf_export_dir,
                command=command,
            )
            index_runs[run_id] = {
                "status": status,
                "export_manifest": str(manifest_path.resolve()),
                "export_dir": str(export_dir.resolve()),
                "latest_checkpoint_exists": latest_checkpoint.exists(),
                "hf_export_dir": str(hf_export_dir.resolve()) if hf_export_dir is not None else "",
                "hf_export_exists": bool(hf_export_dir is not None and hf_export_dir.exists() and any(hf_export_dir.iterdir())),
            }

    if not ctx.args.dry_run:
        _dump_json(
            cfg.output_dir / CHECKPOINT_EXPORT_INDEX,
            {
                "schema_version": 1,
                "stage": "checkpoint_conversion",
                "output_dir": str(cfg.output_dir.resolve()),
                "runs": index_runs,
            },
        )
        _save_stage_state(ctx.state_path, "checkpoint_conversion", signature, [str(path) for path in expected])
    ctx.logger.info("[OK] checkpoint_exports=%s", cfg.output_dir / CHECKPOINT_EXPORT_INDEX)


class CheckpointConversionStage:
    name = "checkpoint_conversion"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = resolve_checkpoint_conversion_config(ctx)
        artifacts = ()
        if cfg.enabled:
            artifacts = (
                ArtifactSpec(cfg.output_dir / CHECKPOINT_EXPORT_INDEX, required=True, description="Checkpoint export index"),
            )
        return StagePlan(stage=self.name, enabled=cfg.enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_checkpoint_conversion(ctx)
        cfg = resolve_checkpoint_conversion_config(ctx)
        return StageResult(stage=self.name, artifacts=(cfg.output_dir / CHECKPOINT_EXPORT_INDEX,))

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = CheckpointConversionStage()

__all__ = [
    "CHECKPOINT_EXPORT_INDEX",
    "CheckpointConversionConfig",
    "CheckpointConversionStage",
    "resolve_checkpoint_conversion_config",
    "stage",
    "stage_checkpoint_conversion",
]
