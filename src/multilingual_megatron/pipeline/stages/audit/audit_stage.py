from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime import (
    _dump_json,
    _load_json,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ...context import PipelineContext
from ...tokenizer_manifest import load_tokenized_manifest_index
from ...types import ArtifactSpec, StagePlan, StageResult
from .config import AuditConfig, resolve_audit_config


def _prefix_path(prefix: str, suffix: str) -> Path:
    path = Path(f"{prefix}{suffix}")
    return path


def _audit_megatron_manifest(manifest_path: Path, *, require_prefix_files: bool = True) -> dict[str, object]:
    if not manifest_path.exists():
        raise SystemExit(f"audit: missing Megatron data manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("format") != "megatron_gpt":
        raise SystemExit(f"audit: unsupported manifest format: {manifest.get('format')!r}")

    prefixes = [str(item) for item in manifest.get("prefixes", []) if str(item).strip()]
    missing: list[str] = []
    if require_prefix_files:
        for prefix in prefixes:
            for suffix in (".bin", ".idx"):
                candidate = _prefix_path(prefix, suffix)
                if not candidate.exists():
                    missing.append(str(candidate))

    payload = {
        "schema_version": 1,
        "stage": "audit",
        "backend": "megatron_gpt",
        "pass": not missing,
        "manifest": str(manifest_path),
        "languages": manifest.get("languages", []),
        "num_prefixes": len(prefixes),
        "require_prefix_files": require_prefix_files,
        "documents_per_language": manifest.get("documents_per_language", {}),
        "missing_files": missing,
    }
    if missing:
        raise SystemExit(f"audit: missing Megatron bin/idx files: {missing[:5]}")
    return payload


def _audit_megatron_gpt(ctx: PipelineContext, cfg: AuditConfig) -> dict[str, object]:
    index = load_tokenized_manifest_index(ctx.tokenized_dir)
    if index is None:
        if ctx.args.dry_run and not (ctx.tokenized_dir / "megatron_data_manifest.json").exists():
            return {
                "stage": "audit",
                "backend": "megatron_gpt",
                "pass": True,
                "dry_run": True,
                "manifest": str(ctx.tokenized_dir / "megatron_data_manifest.json"),
                "require_prefix_files": cfg.require_prefix_files,
            }
        return _audit_megatron_manifest(
            ctx.tokenized_dir / "megatron_data_manifest.json",
            require_prefix_files=cfg.require_prefix_files,
        )

    tokenizers = index.get("tokenizers", {})
    if not isinstance(tokenizers, dict) or not tokenizers:
        raise SystemExit(f"audit: invalid tokenized manifest index under {ctx.tokenized_dir}")

    tokenizer_payloads: dict[str, object] = {}
    for tokenizer_id, entry in sorted(tokenizers.items()):
        if not isinstance(entry, dict):
            raise SystemExit(f"audit: invalid tokenized manifest entry for {tokenizer_id}")
        data_manifest = Path(str(entry.get("data_manifest", "")))
        tokenizer_payloads[str(tokenizer_id)] = {
            **_audit_megatron_manifest(data_manifest, require_prefix_files=cfg.require_prefix_files),
            "tokenizer_model": entry.get("tokenizer_model", ""),
            "vocab_size": entry.get("vocab_size", 0),
        }

    return {
        "schema_version": 1,
        "stage": "audit",
        "backend": "megatron_gpt",
        "pass": True,
        "manifest_index": str(ctx.tokenized_dir / "tokenized_manifests.json"),
        "tokenizer_count": len(tokenizer_payloads),
        "require_prefix_files": cfg.require_prefix_files,
        "tokenizers": tokenizer_payloads,
    }


def stage_audit(ctx: PipelineContext) -> None:
    if "audit" not in ctx.active_stages:
        return

    cfg = resolve_audit_config(ctx)
    if not cfg.enabled:
        return
    if cfg.backend != "megatron_gpt":
        raise SystemExit(f"audit: unsupported tokenization backend in this repo: {cfg.backend!r}")

    signature = _stage_signature(
        {
            "stage": "audit",
            "cfg": ctx.audit_cfg,
            "tokenizer_training_cfg": ctx.tokenizer_cfg,
            "tokenized_dir": str(ctx.tokenized_dir),
            "backend": cfg.backend,
            "output_path": str(cfg.output_path),
            "require_prefix_files": cfg.require_prefix_files,
        }
    )
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="audit",
        signature=signature,
        expected_artifacts=[cfg.output_path],
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] audit stage (cache hit)")
        return

    payload = _audit_megatron_gpt(ctx, cfg)
    if not ctx.args.dry_run:
        _dump_json(cfg.output_path, payload)
        _save_stage_state(ctx.state_path, "audit", signature, [str(cfg.output_path)])
    ctx.logger.info("[OK] tokenization_audit=%s", cfg.output_path)


class AuditStage:
    name = "audit"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = resolve_audit_config(ctx)
        artifacts = ()
        if cfg.enabled:
            artifacts = (
                ArtifactSpec(cfg.output_path, required=True, description="Tokenization audit report"),
            )
        return StagePlan(stage=self.name, enabled=cfg.enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_audit(ctx)
        cfg = resolve_audit_config(ctx)
        return StageResult(stage=self.name, artifacts=(cfg.output_path,))

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = AuditStage()

__all__ = ["AuditStage", "stage", "stage_audit"]
