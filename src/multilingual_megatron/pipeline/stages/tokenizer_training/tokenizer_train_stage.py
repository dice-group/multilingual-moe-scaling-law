from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..runtime import (
    _load_json,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_log_path,
    _stage_signature,
)
from ...context import PipelineContext
from ...tokenizer_manifest import tokenizer_manifest_path
from ...types import ArtifactSpec, StagePlan, StageResult
from .config import SCRATCH_TOKENIZER_MODES, TokenizerTrainingConfig, TokenizerTrainingSpec, resolve_tokenizer_training_config
from .sentencepiece_v2 import audit_production_v2_model


def _read_training_result(path: Path) -> dict[str, Any]:
    result_path = path / "tokenizer_training_result.json"
    if not result_path.exists():
        raise SystemExit(f"tokenizer_train: missing tokenizer result: {result_path}")
    return _load_json(result_path)




def _result_matches_request(
    result: dict[str, Any],
    *,
    spec: TokenizerTrainingSpec,
    num_threads: int,
    repo_root: Path,
) -> bool:
    expected_max_samples = spec.max_samples if spec.max_samples > 0 else None
    expected_max_sample_fraction = spec.max_sample_fraction if spec.max_sample_fraction > 0.0 else None
    if str(result.get("mode", "")) != spec.mode:
        return False
    if int(result.get("requested_vocab_size", 0) or 0) != spec.vocab_size:
        return False
    if str(result.get("text_key", "")) != spec.text_key:
        return False
    if result.get("max_samples") != expected_max_samples:
        return False
    if result.get("max_sample_fraction") != expected_max_sample_fraction:
        return False
    if int(result.get("num_threads", num_threads) or num_threads) != int(num_threads):
        return False
    expected_fields = {
        "normalizer": spec.normalizer,
        "pre_tokenizer": spec.pre_tokenizer,
        "unk_token": spec.unk_token,
        "bos_token": spec.bos_token,
        "eos_token": spec.eos_token,
        "pad_token": spec.pad_token,
    }
    for key, expected in expected_fields.items():
        if expected is not None and str(result.get(key, expected)) != str(expected):
            return False
    tokenizer_model = str(result.get("tokenizer_model", result.get("output_dir", ""))).strip()
    if not tokenizer_model or not Path(tokenizer_model).exists():
        return False
    if spec.production_v2 is None:
        return result.get("production_v2") in (None, False)

    production = result.get("production_v2")
    if not isinstance(production, dict) or production.get("enabled") is not True:
        return False
    source = production.get("source_artifact")
    if not isinstance(source, dict):
        return False
    configured_source = Path(spec.production_v2.sentence_jsonl_path)
    if not configured_source.is_absolute():
        configured_source = repo_root / configured_source
    try:
        result_source = Path(str(source.get("path", ""))).resolve()
    except (OSError, RuntimeError):
        return False
    if result_source != configured_source.resolve():
        return False
    if str(source.get("sha256", "")) != spec.production_v2.sentence_jsonl_sha256:
        return False
    if int(source.get("sentence_count", 0) or 0) != spec.production_v2.expected_sentence_count:
        return False

    model_path = Path(tokenizer_model)
    expected_model_hash = str(result.get("tokenizer_model_sha256", ""))
    if not expected_model_hash:
        return False
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != expected_model_hash:
        return False
    try:
        effective = audit_production_v2_model(
            model_path,
            expected_vocab_size=spec.vocab_size,
            expected_num_threads=num_threads,
            expected_model_prefix=str(model_path.with_suffix("")),
        )
    except SystemExit:
        return False
    return (
        effective.get("model_sha256") == expected_model_hash
        and effective.get("effective_config_sha256") == result.get("effective_config_sha256")
        and effective == result.get("effective_sentencepiece_config")
    )


def _build_tokenizer_train_cmd(
    ctx: PipelineContext,
    *,
    cfg: TokenizerTrainingConfig,
    spec: TokenizerTrainingSpec,
    output_dir: Path,
) -> list[str]:
    cmd = [
        cfg.python_executable,
        "-m",
        "multilingual_megatron.pipeline.stages.tokenizer_training.train_hf_tokenizer",
        "--output-dir",
        str(output_dir),
        "--mode",
        spec.mode,
        "--vocab-size",
        str(spec.vocab_size),
        "--text-key",
        spec.text_key,
        "--max-samples",
        str(spec.max_samples),
        "--num-threads",
        str(cfg.num_threads),
        "--normalizer",
        spec.normalizer,
        "--pre-tokenizer",
        spec.pre_tokenizer,
        "--min-frequency",
        str(spec.min_frequency),
        "--unk-token",
        spec.unk_token,
        "--bos-token",
        spec.bos_token,
        "--eos-token",
        spec.eos_token,
        "--pad-token",
        spec.pad_token,
    ]
    if spec.production_v2 is None:
        cmd.extend(["--input-dir", str(ctx.sample_shard_dir)])
    else:
        sentence_jsonl = Path(spec.production_v2.sentence_jsonl_path)
        if not sentence_jsonl.is_absolute():
            sentence_jsonl = ctx.repo_root / sentence_jsonl
        cmd.extend(
            [
                "--production-v2-sentence-jsonl",
                str(sentence_jsonl.resolve()),
                "--production-v2-sentence-jsonl-sha256",
                spec.production_v2.sentence_jsonl_sha256,
                "--production-v2-expected-sentence-count",
                str(spec.production_v2.expected_sentence_count),
            ]
        )
    if spec.max_sample_fraction > 0.0:
        cmd.extend(["--max-sample-fraction", str(spec.max_sample_fraction)])
    if spec.base_model and spec.mode not in SCRATCH_TOKENIZER_MODES:
        cmd.extend(["--base-model", spec.base_model])
    if spec.fallback_base_model and spec.allow_fallback:
        cmd.extend(["--fallback-base-model", spec.fallback_base_model])
    if spec.trust_remote_code:
        cmd.append("--trust-remote-code")
    if spec.local_files_only:
        cmd.append("--local-files-only")
    for token in spec.special_tokens:
        if token:
            cmd.extend(["--special-token", token])
    return cmd


def _record_manifest_entry(
    ctx: PipelineContext,
    *,
    trained: dict[str, Any],
    spec: TokenizerTrainingSpec,
    result: dict[str, Any],
) -> None:
    resolved_base_model = str(result.get("resolved_base_model", spec.base_model)).strip()
    used_fallback = bool(result.get("used_fallback", False))
    if spec.mode not in SCRATCH_TOKENIZER_MODES and (used_fallback or resolved_base_model != spec.base_model) and not spec.allow_fallback:
        raise SystemExit(
            f"tokenizer '{spec.id}' resolved to '{resolved_base_model}' "
            f"instead of configured base_model '{spec.base_model}'. "
            "Set allow_fallback=true only for explicit plumbing smokes."
        )
    actual_vocab_size = int(result.get("vocab_size", spec.vocab_size))
    source_files = result.get("source_files", [])
    if not isinstance(source_files, list):
        source_files = []
    result_path = ctx.tokenizers_dir / spec.id / "tokenizer_training_result.json"
    manifest_entry = {
        "id": spec.id,
        "path": str(Path(str(result.get("tokenizer_model", ctx.tokenizers_dir / spec.id))).resolve()),
        "result_path": str(result_path.resolve()),
        "tokenizer_type": str(result.get("tokenizer_type", spec.tokenizer_type)),
        "family": spec.family,
        "mode": spec.mode,
        "normalizer": str(result.get("normalizer", spec.normalizer)),
        "pre_tokenizer": str(result.get("pre_tokenizer", spec.pre_tokenizer)),
        "special_tokens": result.get("special_tokens", list(spec.special_tokens)),
        "base_model": spec.base_model,
        "resolved_base_model": resolved_base_model,
        "fallback_base_model": spec.fallback_base_model,
        "allow_fallback": spec.allow_fallback,
        "used_fallback": used_fallback,
        "trust_remote_code": spec.trust_remote_code,
        "requested_vocab_size": spec.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "vocab_size": spec.model_vocab_size,
        "text_key": spec.text_key,
        "max_samples": spec.max_samples if spec.max_samples > 0 else None,
        "max_sample_fraction": spec.max_sample_fraction if spec.max_sample_fraction > 0.0 else None,
        "source_file_count": len(source_files),
        "source_files": source_files,
    }
    if spec.production_v2 is not None:
        production = result.get("production_v2")
        effective = result.get("effective_sentencepiece_config")
        if not isinstance(production, dict) or not isinstance(effective, dict):
            raise SystemExit(f"tokenizer '{spec.id}' is missing audited production_v2 metadata")
        model_sha256 = str(result.get("tokenizer_model_sha256", ""))
        effective_sha256 = str(result.get("effective_config_sha256", ""))
        if not model_sha256 or not effective_sha256:
            raise SystemExit(f"tokenizer '{spec.id}' is missing production_v2 hashes")
        if actual_vocab_size != spec.vocab_size or actual_vocab_size != spec.model_vocab_size:
            raise SystemExit(
                f"tokenizer '{spec.id}' production_v2 vocabulary mismatch: "
                f"actual={actual_vocab_size}, tokenizer={spec.vocab_size}, model={spec.model_vocab_size}"
            )
        manifest_entry.update(
            {
                "production_v2": production,
                "tokenizer_model_sha256": model_sha256,
                "effective_config_sha256": effective_sha256,
                "effective_sentencepiece_config": effective,
                "source_artifact_sha256": str(result.get("source_artifact_sha256", "")),
            }
        )
    trained[spec.id] = manifest_entry


def stage_tokenizer_train(ctx: PipelineContext) -> None:
    if "tokenizer_train" not in ctx.active_stages:
        return

    cfg = resolve_tokenizer_training_config(ctx.tokenizer_cfg, tokenize_cfg=ctx.tokenize_cfg)
    if not cfg.enabled:
        return

    manifest_path = tokenizer_manifest_path(ctx)
    signature = _stage_signature(
        {
            "stage": "tokenizer_train",
            "cfg": ctx.tokenizer_cfg,
            "resolved": {
                "num_threads": cfg.num_threads,
                "tokenizers": [asdict(spec) for spec in cfg.specs],
            },
            "sample_manifest": str(ctx.sampled_manifest),
            "sample_shard_dir": str(ctx.sample_shard_dir),
        }
    )
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="tokenizer_train",
        signature=signature,
        expected_artifacts=[manifest_path],
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] tokenizer_train stage (cache hit)")
        return

    trained: dict[str, Any] = {}
    pending_jobs: list[tuple[list[str], dict[str, str], Path]] = []
    pending_specs: list[TokenizerTrainingSpec] = []

    for spec in cfg.specs:
        output_dir = ctx.tokenizers_dir / spec.id
        result: dict[str, Any] | None = None
        result_path = output_dir / "tokenizer_training_result.json"
        if not ctx.args.dry_run and result_path.exists():
            result = _read_training_result(output_dir)
            if _result_matches_request(
                result,
                spec=spec,
                num_threads=cfg.num_threads,
                repo_root=ctx.repo_root,
            ):
                ctx.logger.info("[SKIP] tokenizer_train.%s (existing artifact)", spec.id)
            else:
                result = None

        if result is None:
            cmd = _build_tokenizer_train_cmd(ctx, cfg=cfg, spec=spec, output_dir=output_dir)
            if cfg.parallel_slurm_jobs:
                pending_jobs.append((cmd, ctx.env, _stage_log_path(ctx.logs_root, ctx.env, "tokenizer_train", spec.id)))
                pending_specs.append(spec)
                continue
            _run_stage_cmd(
                cmd=cmd,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="tokenizer_train",
                suffix=spec.id,
                executor=ctx.executor,
            )
            if not ctx.args.dry_run:
                result = _read_training_result(output_dir)

        if not ctx.args.dry_run and result is not None:
            _record_manifest_entry(ctx, trained=trained, spec=spec, result=result)

    if pending_jobs:
        ctx.executor.run_batch(pending_jobs, ctx.args.dry_run)
        if not ctx.args.dry_run:
            for spec in pending_specs:
                result = _read_training_result(ctx.tokenizers_dir / spec.id)
                _record_manifest_entry(ctx, trained=trained, spec=spec, result=result)

    if not ctx.args.dry_run:
        manifest = {
            "schema_version": (
                2 if any("production_v2" in entry for entry in trained.values())
                else 1
            ),
            "stage": "tokenizer_train",
            "sample_shard_dir": str(ctx.sample_shard_dir.resolve()),
            "tokenizers": trained,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _save_stage_state(ctx.state_path, "tokenizer_train", signature, [str(manifest_path)])
    ctx.logger.info("[OK] tokenizer_manifest=%s", manifest_path)


class TokenizerTrainingStage:
    name = "tokenizer_train"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = getattr(ctx, "tokenizer_cfg", {})
        enabled = bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else False
        artifacts = ()
        if enabled:
            artifacts = (
                ArtifactSpec(
                    tokenizer_manifest_path(ctx),
                    required=True,
                    description="Tokenizer-training manifest",
                ),
            )
        return StagePlan(stage=self.name, enabled=enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_tokenizer_train(ctx)
        return StageResult(stage=self.name, artifacts=(tokenizer_manifest_path(ctx),))

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = TokenizerTrainingStage()

__all__ = ["TokenizerTrainingStage", "stage", "stage_tokenizer_train"]
