from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..runtime import (
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_log_path,
    _stage_signature,
)
from ...context import PipelineContext
from ...executor import SlurmExecutor
from ...tokenizer_manifest import (
    load_tokenizer_manifest,
    tokenizer_training_enabled,
    tokenized_manifests_path,
)
from ...types import ArtifactSpec, StagePlan, StageResult
from .config import MegatronTokenizationConfig, resolve_megatron_tokenization_config


def _module_cmd(module: str, python_executable: str = "python") -> list[str]:
    return [python_executable, "-m", module]


def _resolve_tokenizer_model(ctx: PipelineContext, tokenizer_type: str, tokenizer_model: str) -> str:
    """Resolve repo-local tokenizer files before spawning workers with different cwd."""

    value = str(tokenizer_model).strip()
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = ctx.repo_root / path
    if candidate.exists():
        return str(candidate.resolve())
    local_hint = value.startswith((".", "configs/", "outputs/")) or value.endswith((".model", ".json", ".vocab"))
    if str(tokenizer_type).strip() == "SentencePieceTokenizer" and local_hint:
        raise SystemExit(f"tokenization tokenizer_model not found: {value} (looked under {ctx.repo_root})")
    return value


def _resolve_languages_for_tokenize(ctx: PipelineContext) -> list[str]:
    token_cfg = ctx.tokenize_cfg
    english_key = str(token_cfg.get("language_rates_english_key", "english")).strip() or "english"
    include_english = bool(token_cfg.get("include_english_key", True))
    explicit = token_cfg.get("languages")
    if isinstance(explicit, list) and explicit:
        base = [str(item).strip() for item in explicit if str(item).strip()]
        extras = _resolve_language_extras(token_cfg)
        if include_english:
            return sorted(set(base + extras + [english_key]))  # preserve legacy behavior by default
        return sorted(set(base + extras))

    csv_path = str(token_cfg.get("languages_csv", "")).strip()
    if csv_path:
        column = str(token_cfg.get("languages_csv_column", "subset")).strip() or "subset"
        split_value = str(token_cfg.get("languages_csv_split", "train")).strip()
        split_column = str(token_cfg.get("languages_csv_split_column", "split")).strip() or "split"
        languages: set[str] = set()
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV is missing a header: {csv_path}")
            if column not in reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV missing column '{column}'. Found: {reader.fieldnames}")
            if split_value and split_column not in reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV missing split column '{split_column}'. Found: {reader.fieldnames}")
            for row in reader:
                if split_value:
                    if str(row.get(split_column, "")).strip() != split_value:
                        continue
                value = str(row.get(column, "")).strip()
                if value:
                    languages.add(value)
        if not languages:
            raise SystemExit(f"No languages resolved from {csv_path} (column={column}, split={split_value})")
        extras = _resolve_language_extras(token_cfg)
        resolved = set(languages).union(extras)
        if include_english:
            resolved.add(english_key)
        return sorted(resolved)

    explicit = ctx.sample_cfg.get("languages")
    if isinstance(explicit, list) and explicit:
        base = [str(item).strip() for item in explicit if str(item).strip()]
        extras = _resolve_language_extras(token_cfg)
        if include_english:
            return sorted(set(base + extras + [english_key]))
        return sorted(set(base + extras))

    map_languages = _languages_from_language_map(ctx.language_map_path)
    if map_languages:
        extras = _resolve_language_extras(token_cfg)
        if include_english and english_key in map_languages:
            return sorted(set(map_languages + extras))
        if include_english and english_key not in {"english", ""}:
            return sorted(set(map_languages + extras + [english_key]))
        return sorted(set(map_languages + extras))

    return []


def _languages_from_language_map(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    languages: set[str] = set()
    if isinstance(payload, dict):
        if all(isinstance(value, str) or value is None for value in payload.values()):
            languages.update(str(key) for key, value in payload.items() if value is not None)
        else:
            for entry in payload.values():
                if not isinstance(entry, dict):
                    continue
                raw_languages = entry.get("languages") or entry.get("language") or []
                if isinstance(raw_languages, str):
                    languages.add(raw_languages)
                elif isinstance(raw_languages, list):
                    languages.update(str(item) for item in raw_languages if str(item).strip())
                metadata = entry.get("metadata")
                if isinstance(metadata, dict):
                    languages.update(str(key) for key in metadata.keys() if str(key).strip())
                subgroups = entry.get("subgroups")
                if isinstance(subgroups, dict):
                    for members in subgroups.values():
                        if isinstance(members, list):
                            languages.update(str(item) for item in members if str(item).strip())
    return sorted(languages)


def _resolve_language_extras(token_cfg: dict) -> list[str]:
    extras = token_cfg.get("languages_extra")
    if extras is None:
        return []
    if isinstance(extras, str):
        return [item.strip() for item in extras.split(",") if item.strip()]
    if isinstance(extras, list):
        return [str(item).strip() for item in extras if str(item).strip()]
    return []


def _megatron_preprocess_cmd(
    ctx: PipelineContext,
    *,
    output_dir: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    trust_remote_code: bool = False,
) -> list[str]:
    python_executable = str(ctx.tokenize_cfg.get("python_executable", ctx.env.get("PYTHON_EXECUTABLE", "python3")))
    resolved_tokenizer_model = _resolve_tokenizer_model(ctx, tokenizer_type, tokenizer_model)
    cmd = [
        *_module_cmd("multilingual_megatron.pipeline.stages.tokenization.tools.preprocess_megatron_gpt", python_executable),
        "--bridge-root",
        str(ctx.repo_root / "external" / "Megatron-Bridge"),
        "--shard-dir",
        str(ctx.sample_shard_dir),
        "--output-dir",
        str(output_dir),
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-model",
        resolved_tokenizer_model,
        "--workers",
        str(int(ctx.tokenize_cfg.get("workers", ctx.tokenize_cfg.get("num_proc", 4)))),
        "--log-interval",
        str(int(ctx.tokenize_cfg.get("log_interval", 1000))),
        "--blend-weight-policy",
        str(ctx.tokenize_cfg.get("blend_weight_policy", "equal")),
        "--launcher",
        str(ctx.tokenize_cfg.get("launcher", "python")),
        "--python-executable",
        python_executable,
    ]
    if int(ctx.tokenize_cfg.get("workers_per_language", 0) or 0) > 0:
        cmd.extend(["--workers-per-language", str(int(ctx.tokenize_cfg.get("workers_per_language", 0) or 0))])
    if int(ctx.tokenize_cfg.get("partitions_per_language", 0) or 0) > 0:
        cmd.extend(["--partitions-per-language", str(int(ctx.tokenize_cfg.get("partitions_per_language", 0) or 0))])
    if int(ctx.tokenize_cfg.get("language_jobs", 0) or 0) > 0:
        cmd.extend(["--language-jobs", str(int(ctx.tokenize_cfg.get("language_jobs", 0) or 0))])
    if bool(ctx.tokenize_cfg.get("append_eod", True)):
        cmd.append("--append-eod")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if "tokenize" in ctx.force_stages:
        cmd.append("--force")
    return cmd


def _tokenization_languages_csv(ctx: PipelineContext) -> str:
    languages = _resolve_languages_for_tokenize(ctx)
    return ",".join(languages)


def _megatron_array_common_cmd(ctx: PipelineContext, cfg: MegatronTokenizationConfig) -> list[str]:
    return [
        cfg.python_executable,
        str(ctx.repo_root / "src" / "multilingual_megatron" / "pipeline" / "stages" / "tokenization" / "megatron_distributed.py"),
    ]


def _megatron_plan_cmd(
    ctx: PipelineContext,
    *,
    cfg: MegatronTokenizationConfig,
    plan_dir: Path,
    output_dir: Path,
) -> list[str]:
    cmd = _megatron_array_common_cmd(ctx, cfg)
    cmd.extend(
        [
            "plan",
            "--shard-dir",
            str(ctx.sample_shard_dir),
            "--output-dir",
            str(output_dir),
            "--plan-dir",
            str(plan_dir),
            "--languages",
            _tokenization_languages_csv(ctx),
            "--array-ranks",
            str(cfg.array_ranks),
            "--sample-fraction",
            str(cfg.sample_fraction),
            "--train-fraction",
            str(cfg.train_fraction),
            "--validation-fraction",
            str(cfg.validation_fraction),
        ]
    )
    return cmd


def _megatron_worker_cmd(
    ctx: PipelineContext,
    *,
    cfg: MegatronTokenizationConfig,
    plan_dir: Path,
    ranks_dir: Path,
    include_rank_arg: bool,
    rank: int = 0,
) -> list[str]:
    resolved_tokenizer_model = _resolve_tokenizer_model(ctx, cfg.tokenizer_type, cfg.tokenizer_model)
    cmd = _megatron_array_common_cmd(ctx, cfg)
    cmd.extend(
        [
            "worker",
            "--repo-root",
            str(ctx.repo_root),
            "--bridge-root",
            str(ctx.repo_root / "external" / "Megatron-Bridge"),
            "--plan-dir",
            str(plan_dir),
            "--ranks-dir",
            str(ranks_dir),
            "--tokenizer-type",
            cfg.tokenizer_type,
            "--tokenizer-model",
            resolved_tokenizer_model,
            "--workers",
            str(cfg.workers),
            "--workers-per-language",
            str(cfg.workers_per_prefix),
            "--partitions-per-language",
            str(cfg.partitions_per_prefix),
            "--language-jobs",
            str(cfg.language_jobs),
            "--blend-weight-policy",
            cfg.blend_weight_policy,
            "--launcher",
            cfg.launcher,
            "--python-executable",
            cfg.python_executable,
            "--sample-fraction",
            str(cfg.sample_fraction),
            "--max-samples",
            str(cfg.max_samples),
            "--train-fraction",
            str(cfg.train_fraction),
            "--split-seed",
            str(cfg.split_seed),
            "--force",
        ]
    )
    if include_rank_arg:
        cmd.extend(["--rank", str(rank)])
    if cfg.append_eod:
        cmd.append("--append-eod")
    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    if cfg.keep_work_dir:
        cmd.append("--keep-work-dir")
    return cmd


def _megatron_merge_cmd(
    ctx: PipelineContext,
    *,
    cfg: MegatronTokenizationConfig,
    ranks_dir: Path,
    output_dir: Path,
) -> list[str]:
    resolved_tokenizer_model = _resolve_tokenizer_model(ctx, cfg.tokenizer_type, cfg.tokenizer_model)
    cmd = _megatron_array_common_cmd(ctx, cfg)
    cmd.extend(
        [
            "merge",
            "--ranks-dir",
            str(ranks_dir),
            "--output-dir",
            str(output_dir),
            "--tokenizer-type",
            cfg.tokenizer_type,
            "--tokenizer-model",
            resolved_tokenizer_model,
            "--blend-weight-policy",
            cfg.blend_weight_policy,
        ]
    )
    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    if cfg.validation_enabled:
        cmd.append("--check-prefix-files")
    return cmd


def _stage_tokenize_megatron_array(
    ctx: PipelineContext,
    cfg: MegatronTokenizationConfig,
    signature: str,
    *,
    output_dir: Path | None = None,
    tokenizer_id: str = "",
    use_stage_cache: bool = True,
    save_stage_state: bool = True,
) -> None:
    output_dir = output_dir or ctx.tokenized_dir
    artifacts = [output_dir / "megatron_data_manifest.json", output_dir / "state.json"]
    if use_stage_cache:
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="tokenize",
            signature=signature,
            expected_artifacts=artifacts,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] tokenize stage (cache hit)")
            return

    if cfg.skip_existing and all(path.exists() for path in artifacts):
        ctx.logger.info("[SKIP] tokenize stage (output exists)")
        return

    plan_dir = output_dir.parent / f"{output_dir.name}_megatron_plan"
    ranks_dir = output_dir.parent / f"{output_dir.name}_megatron_ranks"
    logs_root = ctx.logs_root / "stages" / "tokenize" / "megatron"
    if tokenizer_id:
        logs_root = logs_root / tokenizer_id
    plan_logs_root = logs_root / "plan"
    array_logs_root = logs_root / "array"
    merge_logs_root = logs_root / "merge"
    for path in (plan_logs_root, array_logs_root, merge_logs_root):
        path.mkdir(parents=True, exist_ok=True)

    plan_cmd = _megatron_plan_cmd(ctx, cfg=cfg, plan_dir=plan_dir, output_dir=output_dir)
    worker_cmd = _megatron_worker_cmd(
        ctx,
        cfg=cfg,
        plan_dir=plan_dir,
        ranks_dir=ranks_dir,
        include_rank_arg=False,
    )
    merge_cmd = _megatron_merge_cmd(ctx, cfg=cfg, ranks_dir=ranks_dir, output_dir=output_dir)

    if isinstance(ctx.executor, SlurmExecutor) and cfg.mode == "slurm_array":
        lang_count = max(1, int(getattr(ctx, "lang_count", 0) or len(_resolve_languages_for_tokenize(ctx)) or 1))
        plan_log = _stage_log_path(plan_logs_root, ctx.env, "tokenize", "megatron_plan")
        plan_job_id = ctx.executor.submit(
            plan_cmd,
            ctx.env,
            ctx.args.dry_run,
            plan_log,
            job_name=f"tokp{lang_count}l",
        )
        array_log = _stage_log_path(array_logs_root, ctx.env, "tokenize", "megatron_array")
        array_job_id = ctx.executor.submit_array(
            worker_cmd,
            ctx.env,
            ctx.args.dry_run,
            array_log,
            array=cfg.array_spec,
            dependency=f"afterok:{plan_job_id}" if plan_job_id else None,
            slurm_output=str(array_logs_root / "tokenize_megatron_array_%A_%a.log"),
            job_name=f"tok{lang_count}lmg",
        )
        merge_log = _stage_log_path(merge_logs_root, ctx.env, "tokenize", "megatron_merge")
        merge_job_id = ctx.executor.submit(
            merge_cmd,
            ctx.env,
            ctx.args.dry_run,
            merge_log,
            dependency=f"afterok:{array_job_id}" if array_job_id else None,
            slurm_output=str(merge_logs_root / "tokenize_megatron_merge_%A.log"),
            job_name=f"tokm{lang_count}lmg",
        )
        if merge_job_id:
            ctx.executor.wait_for_jobs([(merge_job_id, merge_log)], ctx.args.dry_run)
    else:
        _run_stage_cmd(
            cmd=plan_cmd,
            env=ctx.env,
            dry_run=ctx.args.dry_run,
            logs_root=plan_logs_root,
            stage="tokenize",
            suffix="megatron_plan",
            executor=ctx.executor,
        )
        for rank in range(cfg.array_ranks):
            rank_cmd = _megatron_worker_cmd(
                ctx,
                cfg=cfg,
                plan_dir=plan_dir,
                ranks_dir=ranks_dir,
                include_rank_arg=True,
                rank=rank,
            )
            _run_stage_cmd(
                cmd=rank_cmd,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=array_logs_root,
                stage="tokenize",
                suffix=f"megatron_rank_{rank:05d}",
                executor=ctx.executor,
            )
        _run_stage_cmd(
            cmd=merge_cmd,
            env=ctx.env,
            dry_run=ctx.args.dry_run,
            logs_root=merge_logs_root,
            stage="tokenize",
            suffix="megatron_merge",
            executor=ctx.executor,
        )

    if not ctx.args.dry_run:
        report_path = ctx.tokenized_dir / "tokenization_report.json"
        if cfg.validation_enabled and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not bool(report.get("pass", False)):
                raise SystemExit(f"Megatron tokenization validation failed: {report_path}")
        if save_stage_state:
            _save_stage_state(ctx.state_path, "tokenize", signature, [str(output_dir)])


def _stage_tokenize_megatron_multi(ctx: PipelineContext, cfg: MegatronTokenizationConfig, signature: str) -> None:
    try:
        tokenizer_manifest = load_tokenizer_manifest(ctx)
        tokenizers = tokenizer_manifest.get("tokenizers", {})
    except SystemExit:
        if not ctx.args.dry_run:
            raise
        tokenizers = {}
        for raw in ctx.tokenizer_cfg.get("tokenizers", []):
            if not isinstance(raw, dict):
                continue
            tokenizer_id = str(raw.get("id", "")).strip()
            if not tokenizer_id:
                continue
            tokenizers[tokenizer_id] = {
                "path": str((ctx.tokenizers_dir / tokenizer_id).resolve()),
                "tokenizer_type": str(raw.get("tokenizer_type", ctx.tokenize_cfg.get("tokenizer_type", "HuggingFaceTokenizer"))),
                "vocab_size": int(raw.get("vocab_size", ctx.tokenizer_cfg.get("vocab_size", 0)) or 0),
            }
    if not isinstance(tokenizers, dict) or not tokenizers:
        raise SystemExit("tokenizer_train manifest contains no tokenizers")
    configured_ids = [
        str(item.get("id", "")).strip()
        for item in ctx.tokenizer_cfg.get("tokenizers", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    if configured_ids:
        requested = set(configured_ids)
        tokenizers = {key: value for key, value in tokenizers.items() if str(key) in requested}
        missing = sorted(requested.difference(str(key) for key in tokenizers.keys()))
        if missing:
            raise SystemExit(f"tokenize: tokenizer manifest missing requested ids: {missing}")

    index_path = tokenized_manifests_path(ctx)
    artifacts = [index_path]
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="tokenize",
        signature=signature,
        expected_artifacts=artifacts,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] tokenize stage (cache hit)")
        return

    resolved: dict[str, dict[str, object]] = {}
    pending_jobs: list[tuple[list[str], dict[str, str], Path]] = []
    pending_entries: list[tuple[str, str, str, bool, Path, dict[str, object]]] = []
    parallel_jobs = bool(ctx.tokenize_cfg.get("parallel_slurm_jobs", False))
    for tokenizer_id, raw_entry in sorted(tokenizers.items()):
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"Invalid tokenizer manifest entry for {tokenizer_id}")
        tokenizer_model = str(raw_entry.get("path", "")).strip()
        if not tokenizer_model:
            raise SystemExit(f"Tokenizer manifest entry '{tokenizer_id}' is missing path")
        tokenizer_type = str(raw_entry.get("tokenizer_type", ctx.tokenize_cfg.get("tokenizer_type", "HuggingFaceTokenizer")))
        trust_remote_code = bool(raw_entry.get("trust_remote_code", False))
        output_dir = ctx.tokenized_dir / str(tokenizer_id)
        if cfg.uses_megatron_array:
            tokenizer_cfg = replace(
                cfg,
                output_dir=output_dir,
                tokenizer_type=tokenizer_type,
                tokenizer_model=tokenizer_model,
                trust_remote_code=trust_remote_code,
            )
            tokenizer_signature = _stage_signature(
                {
                    "stage": "tokenize",
                    "parent_signature": signature,
                    "tokenizer_id": str(tokenizer_id),
                    "tokenizer_entry": raw_entry,
                }
            )
            _stage_tokenize_megatron_array(
                ctx,
                tokenizer_cfg,
                tokenizer_signature,
                output_dir=output_dir,
                tokenizer_id=str(tokenizer_id),
                use_stage_cache=False,
                save_stage_state=False,
            )
            pending_entries.append(
                (str(tokenizer_id), tokenizer_type, tokenizer_model, trust_remote_code, output_dir, raw_entry)
            )
            continue

        cmd = _megatron_preprocess_cmd(
            ctx,
            output_dir=output_dir,
            tokenizer_type=tokenizer_type,
            tokenizer_model=tokenizer_model,
            trust_remote_code=trust_remote_code,
        )
        if parallel_jobs:
            pending_jobs.append((cmd, ctx.env, _stage_log_path(ctx.logs_root, ctx.env, "tokenize", str(tokenizer_id))))
            pending_entries.append(
                (str(tokenizer_id), tokenizer_type, tokenizer_model, trust_remote_code, output_dir, raw_entry)
            )
        else:
            _run_stage_cmd(
                cmd=cmd,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="tokenize",
                suffix=str(tokenizer_id),
                executor=ctx.executor,
            )
            pending_entries.append(
                (str(tokenizer_id), tokenizer_type, tokenizer_model, trust_remote_code, output_dir, raw_entry)
            )

    if pending_jobs:
        ctx.executor.run_batch(pending_jobs, ctx.args.dry_run)

    if not ctx.args.dry_run:
        for tokenizer_id, tokenizer_type, tokenizer_model, trust_remote_code, output_dir, raw_entry in pending_entries:
            data_manifest = output_dir / "megatron_data_manifest.json"
            if not data_manifest.exists():
                raise SystemExit(f"tokenize: missing Megatron manifest for tokenizer {tokenizer_id}: {data_manifest}")
            resolved[tokenizer_id] = {
                "tokenizer_id": tokenizer_id,
                "tokenizer_type": tokenizer_type,
                "tokenizer_model": tokenizer_model,
                "trust_remote_code": trust_remote_code,
                "vocab_size": int(raw_entry.get("vocab_size", 0) or 0),
                "tokenized_dir": str(output_dir.resolve()),
                "data_manifest": str(data_manifest.resolve()),
                "tokenizer_training": raw_entry,
            }

    if not ctx.args.dry_run:
        index = {
            "schema_version": 1,
            "stage": "tokenize",
            "backend": "megatron_gpt",
            "sample_shard_dir": str(ctx.sample_shard_dir.resolve()),
            "tokenizers": resolved,
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (ctx.tokenized_dir / "state.json").write_text(
            json.dumps({"stage": "tokenize", "pass": True, "tokenizer_count": len(resolved)}, indent=2) + "\n",
            encoding="utf-8",
        )
        _save_stage_state(ctx.state_path, "tokenize", signature, [str(ctx.tokenized_dir)])


def stage_tokenize(ctx: PipelineContext) -> None:
    if "tokenize" in ctx.active_stages and bool(ctx.tokenize_cfg.get("enabled", True)):
        backend = str(ctx.tokenize_cfg.get("backend", "megatron_gpt")).strip().lower()
        if backend != "megatron_gpt":
            raise SystemExit(
                f"Unsupported tokenization backend '{backend}'. "
                "This repo now owns tokenization through backend='megatron_gpt'."
            )
        training_enabled = tokenizer_training_enabled(ctx)
        megatron_cfg = resolve_megatron_tokenization_config(
            ctx.tokenize_cfg,
            output_dir=ctx.tokenized_dir,
            require_tokenizer_model=not training_enabled,
        )
        tokenize_model_name = megatron_cfg.tokenizer_model
        signature = _stage_signature(
            {
                "stage": "tokenize",
                "cfg": ctx.tokenize_cfg,
                "tokenizer_training_cfg": ctx.tokenizer_cfg if training_enabled else {},
                "sample_manifest": str(ctx.sampled_manifest),
                "tokenizer_model_name": tokenize_model_name,
            }
        )
        if backend == "megatron_gpt" and megatron_cfg.uses_megatron_array:
            if training_enabled:
                _stage_tokenize_megatron_multi(ctx, megatron_cfg, signature)
                return
            _stage_tokenize_megatron_array(ctx, megatron_cfg, signature)
            return
        if backend == "megatron_gpt" and training_enabled:
            _stage_tokenize_megatron_multi(ctx, megatron_cfg, signature)
            return
        artifacts = [ctx.tokenized_dir / "dataset_info.json", ctx.tokenized_dir / "state.json"]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="tokenize",
            signature=signature,
            expected_artifacts=artifacts,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] tokenize stage (cache hit)")
        else:
            cmd = _megatron_preprocess_cmd(
                ctx,
                output_dir=ctx.tokenized_dir,
                tokenizer_type=megatron_cfg.tokenizer_type,
                tokenizer_model=tokenize_model_name,
                trust_remote_code=megatron_cfg.trust_remote_code,
            )
            _run_stage_cmd(
                cmd=cmd,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="tokenize",
                executor=ctx.executor,
            )
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "tokenize", signature, [str(ctx.tokenized_dir)])


class TokenizationStage:
    name = "tokenize"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = getattr(ctx, "tokenize_cfg", {})
        enabled = bool(cfg.get("enabled", True)) if isinstance(cfg, dict) else True
        tokenized_dir = getattr(ctx, "tokenized_dir", None)
        artifacts = ()
        if tokenized_dir is not None:
            artifacts = (
                ArtifactSpec(tokenized_dir / "megatron_data_manifest.json", required=False, description="Megatron GPT manifest"),
                ArtifactSpec(tokenized_dir / "tokenized_manifests.json", required=False, description="Multi-tokenizer manifest index"),
                ArtifactSpec(tokenized_dir / "state.json", required=True, description="Tokenization state"),
            )
        return StagePlan(stage=self.name, enabled=enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_tokenize(ctx)
        tokenized_dir = getattr(ctx, "tokenized_dir", None)
        artifacts = (tokenized_dir,) if tokenized_dir is not None else ()
        return StageResult(stage=self.name, artifacts=artifacts)

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = TokenizationStage()

__all__ = [
    "TokenizationStage",
    "_stage_tokenize_megatron_array",
    "_stage_tokenize_megatron_multi",
    "stage",
    "stage_tokenize",
]
