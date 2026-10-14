import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .runtime import (
    _abs,
    _compute_pipeline_code_signature,
    _load_pipeline_config,
    _model_slug,
    _resolve_profile,
    _resolve_profile_languages,
    _sanitize_slug,
    build_executor,
)
from .context import PipelineContext
from .executor import SlurmExecutor
from .training import build_training_cmd, describe_training_run, resolve_training_stage_config
from .stages.training.calibration import calibration_enabled_for_any_run
from .tokenizer_manifest import resolve_run_tokenizer_id

from .registry import STAGE_ORDER, get_stage, run_stage

STAGES = STAGE_ORDER


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_run_ids(raw_values: list[str]) -> list[str]:
    """Enable running one or more selected training run IDs from a profile."""
    run_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            run_id = item.strip()
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            run_ids.append(run_id)
    return run_ids


def _configure_temp_env(
    *,
    repo_root: Path,
    env: dict[str, str],
    profile: dict[str, Any],
    tokenize_cfg: dict[str, Any],
) -> None:
    raw_tmp_dir = str(env.get("TMPDIR") or env.get("TMP_DIR") or "").strip()
    if not raw_tmp_dir:
        runtime_cfg = profile.get("runtime", {})
        runtime_cfg = runtime_cfg if isinstance(runtime_cfg, dict) else {}
        for cfg in (tokenize_cfg, runtime_cfg, profile):
            if not isinstance(cfg, dict):
                continue
            raw_tmp_dir = str(cfg.get("tmp_dir") or cfg.get("temp_dir") or "").strip()
            if raw_tmp_dir:
                break
    if not raw_tmp_dir:
        return

    tmp_dir = _abs(repo_root, raw_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir_str = str(tmp_dir)
    env["TMPDIR"] = tmp_dir_str
    env["TMP_DIR"] = tmp_dir_str
    env["TMP"] = tmp_dir_str
    env["TEMP"] = tmp_dir_str


def _apply_profile_env(env: dict[str, str], profile: dict[str, Any]) -> None:
    runtime_cfg = profile.get("runtime", {})
    runtime_cfg = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    env_cfg = runtime_cfg.get("env", {})
    if not isinstance(env_cfg, dict):
        return
    for key, value in env_cfg.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        value_text = str(value)
        for env_key, env_value in env.items():
            value_text = value_text.replace(f"${{{env_key}}}", env_value)
        env[key_text] = value_text


def _stage_profile_cfg(profile: dict[str, Any], stage: str) -> dict[str, Any]:
    """Map pipeline stage names to their profile config blocks."""
    stage_to_profile_key = {
        "sample": "sample",
        "tokenizer_train": "tokenizer_training",
        "tokenize": "tokenization",
        "audit": "audit",
        "train": "training",
        "cooldown_readout": "cooldown_readout",
        "checkpoint_conversion": "checkpoint_conversion",
        "final_loss": "final_loss",
        "benchmark": "benchmark",
        "baseline_eval": "baseline_eval",
        "summary": "summary",
    }
    key = stage_to_profile_key.get(stage, "")
    cfg = profile.get(key, {})
    return cfg if isinstance(cfg, dict) else {}


def _resolve_stage_cfg(profile: dict[str, Any], stage: str) -> dict[str, Any]:
    cfg = _stage_profile_cfg(profile, stage)
    executor_type_override = str(profile.get("_executor_type_override", "")).strip()
    if executor_type_override:
        merged = dict(cfg) if isinstance(cfg, dict) else {}
        stage_executor = dict(merged.get("executor", {})) if isinstance(merged.get("executor"), dict) else {}
        stage_executor["type"] = executor_type_override
        merged["executor"] = stage_executor
        return merged
    stage_executors = profile.get("stage_executors", {})
    if isinstance(stage_executors, dict):
        stage_exec = stage_executors.get(stage)
        if isinstance(stage_exec, dict):
            merged = dict(cfg) if isinstance(cfg, dict) else {}
            exec_override = dict(stage_exec)
            exec_override.update(merged.get("executor", {}) if isinstance(merged.get("executor"), dict) else {})
            merged["executor"] = exec_override
            return merged
    return cfg


def _is_unconfigured_architecture_slot(profile: dict[str, Any], stage: str) -> bool:
    return stage in {"cooldown_readout", "checkpoint_conversion", "final_loss"} and not _stage_profile_cfg(profile, stage)


def _build_language_map_from_csv(
    *,
    csv_path: Path,
    output_path: Path,
    subset_column: str = "subset",
    family_column: str = "family",
    split_column: str = "split",
    split_value: str = "train",
    extra_languages: dict[str, str] | None = None,
) -> Path:
    rows: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"language_map_csv is missing a header: {csv_path}")
        if subset_column not in reader.fieldnames:
            raise SystemExit(f"language_map_csv missing column '{subset_column}'. Found: {reader.fieldnames}")
        if family_column not in reader.fieldnames:
            raise SystemExit(f"language_map_csv missing column '{family_column}'. Found: {reader.fieldnames}")
        if split_value and split_column not in reader.fieldnames:
            raise SystemExit(f"language_map_csv missing split column '{split_column}'. Found: {reader.fieldnames}")

        for row in reader:
            if split_value and str(row.get(split_column, "")).strip() != split_value:
                continue
            subset = str(row.get(subset_column, "")).strip()
            if not subset:
                continue
            family = str(row.get(family_column, "")).strip() or "unknown"
            rows.setdefault(subset, family)

    if extra_languages:
        for lang, family in extra_languages.items():
            if not str(lang).strip():
                continue
            rows.setdefault(str(lang).strip(), str(family).strip() or "unknown")

    if not rows:
        raise SystemExit(f"No languages resolved from language_map_csv: {csv_path}")

    grouped: dict[str, list[str]] = {}
    for lang, family in rows.items():
        grouped.setdefault(family, []).append(lang)

    payload: dict[str, dict[str, Any]] = {}
    for idx, family in enumerate(sorted(grouped.keys())):
        payload[str(idx)] = {
            "group": family,
            "languages": sorted(grouped[family]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _make_attempt_id(raw_attempt_id: str, run_timestamp: str) -> str:
    raw = str(raw_attempt_id or "").strip()
    if raw:
        attempt_id = _sanitize_slug(raw)
    else:
        attempt_id = f"{run_timestamp}-{os.getpid()}"
    if not attempt_id:
        raise SystemExit("--attempt-id resolved to an empty value")
    return attempt_id


def _configure_pipeline_logger(logs_root: Path) -> logging.Logger:
    logger = logging.getLogger("proof_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    run_log_path = logs_root / "pipeline.log"
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multilingual proof pipeline.")
    parser.add_argument("--profile", default="local_mini", help="Profile key in proof config.")
    parser.add_argument("--config", default="configs/pipeline", help="Pipeline config JSON, config-bundle pointer JSON, or config bundle directory.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit.")
    parser.add_argument("--force-stage", action="append", default=[], choices=STAGES, help="Force rerun of a stage even if cached (e.g tokenization)")
    parser.add_argument("--start-at", choices=STAGES, default="sample")
    parser.add_argument("--stop-after", choices=STAGES, default="summary")
    parser.add_argument("--run-id", action="append", default=[], help="Run only selected training run ids")
    parser.add_argument("--submit-chain", action="store_true", help="Submit stage-wise Slurm jobs with dependencies.")
    parser.add_argument("--attempt-id", default="", help="Optional human-readable log attempt id. Defaults to timestamp plus process id.")
    parser.add_argument(
        "--group-by-tokenizer",
        action="store_true",
        help="On local runs, execute tokenizer_train/tokenize/train one tokenizer group at a time.",
    )
    parser.add_argument("--executor-type", choices=["local", "slurm"], default="", help="Override executor type from profile.")
    parser.add_argument("--cuda-visible-devices", default="", help="Override CUDA_VISIBLE_DEVICES for pipeline subprocesses.",)
    return parser.parse_args()


def _build_context(args: argparse.Namespace) -> PipelineContext:
    # main paths / configs
    repo_root = _repo_root()
    config = _load_pipeline_config(_abs(repo_root, args.config))
    profile = dict(_resolve_profile(config, args.profile))
    run_ids = _parse_run_ids(args.run_id)
    args.run_id = run_ids
    force_stages = set(args.force_stage)

    executor_cfg_raw = profile.get("executor", {})
    executor_cfg = dict(executor_cfg_raw) if isinstance(executor_cfg_raw, dict) else {}
    executor_overridden = False
    if args.executor_type:
        executor_cfg["type"] = args.executor_type
        profile["_executor_type_override"] = args.executor_type
        executor_overridden = True
    if str(args.cuda_visible_devices).strip():
        executor_cfg["cuda_visible_devices"] = str(args.cuda_visible_devices).strip()
        executor_overridden = True
    if executor_overridden:
        profile["executor"] = executor_cfg

    stage_start_idx = STAGES.index(args.start_at)
    stage_stop_idx = STAGES.index(args.stop_after)
    if stage_stop_idx < stage_start_idx:
        raise SystemExit("--stop-after must come after --start-at")
    active_stages = STAGES[stage_start_idx: stage_stop_idx + 1]

    output_root = _abs(repo_root, profile.get("output_root", f"outputs/proof_runs/{args.profile}"))
    state_path = output_root / "state" / "pipeline_state.json"
    output_root.mkdir(parents=True, exist_ok=True)

    # stage wise config resolution
    language_map_csv = profile.get("language_map_csv")
    if isinstance(language_map_csv, str) and language_map_csv.strip():
        csv_path = _abs(repo_root, language_map_csv)
        extra_raw = profile.get("language_map_extra_languages", {})
        extra: dict[str, str] = {}
        if isinstance(extra_raw, dict):
            extra = {str(k): str(v) for k, v in extra_raw.items() if str(k).strip()}
        elif isinstance(extra_raw, list):
            fallback_family = str(profile.get("language_map_extra_family", "unknown")).strip() or "unknown"
            for item in extra_raw:
                if str(item).strip():
                    extra[str(item).strip()] = fallback_family
        output_map_path = output_root / "language_map_from_csv.json"
        language_map_path = _build_language_map_from_csv(
            csv_path=csv_path,
            output_path=output_map_path,
            subset_column=str(profile.get("language_map_csv_subset_column", "subset")).strip() or "subset",
            family_column=str(profile.get("language_map_csv_family_column", "family")).strip() or "family",
            split_column=str(profile.get("language_map_csv_split_column", "split")).strip() or "split",
            split_value=str(profile.get("language_map_csv_split", "train")).strip(),
            extra_languages=extra,
        )
    else:
        language_map_path = _abs(repo_root, profile["language_map_path"])
    sample_cfg = profile.get("sample", {})
    tokenizer_cfg = profile.get("tokenizer_training", {})
    tokenize_cfg = profile.get("tokenization", {})
    audit_cfg = profile.get("audit", {})
    train_cfg = profile.get("training", {})
    if run_ids:
        runs = train_cfg.get("runs", []) if isinstance(train_cfg, dict) else []
        if not isinstance(runs, list):
            raise SystemExit("profile.training.runs must be a list to use --run-id")
        selected_run_ids = set(run_ids)
        selected = [run for run in runs if str(run.get("id", "")) in selected_run_ids]
        selected_ids = {str(run.get("id", "")) for run in selected}
        missing = [run_id for run_id in run_ids if run_id not in selected_ids]
        if missing:
            raise SystemExit(f"Unknown --run-id values for profile '{args.profile}': {missing}")
        train_cfg = dict(train_cfg)
        train_cfg["runs"] = selected
        profile["training"] = train_cfg
    summary_cfg = profile.get("summary", {})
    benchmark_cfg = profile.get("benchmark", {})
    baseline_eval_cfg = profile.get("baseline_eval", {})
    wandb_cfg = profile.get("wandb", {})
    run_naming_cfg = profile.get("run_naming", {})
    run_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    attempt_id = _make_attempt_id(getattr(args, "attempt_id", ""), run_timestamp)

    env = os.environ.copy()
    # Preserve virtualenv interpreter paths; resolving symlinks here can strip
    # the venv context and cause stage subprocesses to miss installed packages.
    python_exec = Path(sys.executable)
    env["PYTHON_EXECUTABLE"] = str(python_exec)
    env["PATH"] = f"{python_exec.parent}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = (
        f"{repo_root / 'src'}:"
        f"{repo_root}:"
        f"{repo_root / 'external' / 'Megatron-Bridge' / 'src'}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    env["PROOF_PIPELINE_REPO_ROOT"] = str(repo_root)
    env["PROOF_PIPELINE_RUN_TIMESTAMP"] = run_timestamp
    env["PROOF_PIPELINE_ATTEMPT_ID"] = attempt_id
    env["PROOF_PIPELINE_CODE_SIG"] = _compute_pipeline_code_signature(repo_root)
    _apply_profile_env(env, profile)
    temp_cfg: dict[str, Any] = {}
    if isinstance(tokenizer_cfg, dict):
        temp_cfg.update(tokenizer_cfg)
    if isinstance(tokenize_cfg, dict):
        temp_cfg.update(tokenize_cfg)
    _configure_temp_env(
        repo_root=repo_root,
        env=env,
        profile=profile,
        tokenize_cfg=temp_cfg,
    )
    os.environ["PROOF_PIPELINE_CODE_SIG"] = env["PROOF_PIPELINE_CODE_SIG"]
    executor_cfg = profile.get("executor", {})
    if isinstance(executor_cfg, dict):
        cuda_visible_devices = str(executor_cfg.get("cuda_visible_devices", "")).strip()
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if run_ids:
        env["PROOF_PIPELINE_SELECTED_RUN_IDS"] = ",".join(run_ids)
    env.setdefault("WANDB_MODE", str(profile.get("wandb_mode", "disabled")))
    if bool(wandb_cfg.get("enabled", False)):
        project = str(wandb_cfg.get("project", "")).strip()
        if project:
            env["WANDB_PROJECT"] = project
        entity = str(wandb_cfg.get("entity", "")).strip()
        if entity:
            env["WANDB_ENTITY"] = entity
        env["WANDB_MODE"] = str(wandb_cfg.get("mode", env.get("WANDB_MODE", "online")))

    # paths and dirs
    sampled_dir = _abs(repo_root, sample_cfg.get("output_dir", str(output_root / "sampled")))
    sampled_manifest = sampled_dir / "manifest.json"
    shard_subdir = str(sample_cfg.get("shard_subdir", "")).strip().strip("/")
    sample_shard_dir = sampled_dir if not shard_subdir else (sampled_dir / shard_subdir)
    tokenizers_dir = (
        _abs(repo_root, tokenizer_cfg.get("output_dir", str(output_root / "tokenizers")))
        if isinstance(tokenizer_cfg, dict)
        else output_root / "tokenizers"
    )
    tokenized_dir = _abs(repo_root, tokenize_cfg.get("output_dir", str(output_root / "tokenized")))
    train_root = _abs(repo_root, train_cfg.get("output_dir", str(output_root / "runs")))
    summary_path = _abs(repo_root, summary_cfg.get("summary_csv", str(output_root / "summary.csv")))
    gate_path = _abs(repo_root, summary_cfg.get("gate_json", str(output_root / "gate.json")))
    benchmark_root = _abs(repo_root, benchmark_cfg.get("output_dir", str(output_root / "benchmarks")))
    baseline_cache_root = _abs(repo_root, baseline_eval_cfg.get("output_root", "outputs/baselines"))
    lineage_manifest_path = output_root / "lineage" / "run_lineage.json"

    logs_base_root = _abs(repo_root, profile.get("logs_dir", str(output_root / "logs")))
    logs_root = logs_base_root / "attempts" / attempt_id
    logs_base_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    env["PROOF_PIPELINE_LOGS_BASE_ROOT"] = str(logs_base_root)
    env["PROOF_PIPELINE_ATTEMPT_LOGS_ROOT"] = str(logs_root)
    env["PROOF_PIPELINE_STRUCTURED_LOGS"] = "1"

    naming_languages = _resolve_profile_languages(
        sample_cfg=sample_cfg,
        sample_manifest_path=sampled_manifest,
        sampled_dir=sampled_dir,
    )
    lang_count = (
        len(naming_languages)
        if naming_languages
        else int(run_naming_cfg.get("language_count_hint", len(sample_cfg.get("languages", []))))
    )
    model_slug = _model_slug(str(train_cfg.get("model_name_or_path", "model")))
    exp_tag = _sanitize_slug(str(run_naming_cfg.get("experiment_tag", "proof")))
    base_run_label = f"{model_slug}-{lang_count}langs-{exp_tag}-{run_timestamp}"
    logger = _configure_pipeline_logger(logs_root)

    return PipelineContext(
        args=args,
        repo_root=repo_root,
        profile=profile,
        force_stages=force_stages,
        active_stages=active_stages,
        output_root=output_root,
        state_path=state_path,
        language_map_path=language_map_path,
        sample_cfg=sample_cfg,
        tokenizer_cfg=tokenizer_cfg if isinstance(tokenizer_cfg, dict) else {},
        tokenize_cfg=tokenize_cfg,
        audit_cfg=audit_cfg,
        train_cfg=train_cfg,
        summary_cfg=summary_cfg,
        benchmark_cfg=benchmark_cfg,
        baseline_eval_cfg=baseline_eval_cfg,
        wandb_cfg=wandb_cfg,
        env=env,
        logger=logger,
        sampled_dir=sampled_dir,
        sampled_manifest=sampled_manifest,
        shard_subdir=shard_subdir,
        sample_shard_dir=sample_shard_dir,
        tokenizers_dir=tokenizers_dir,
        tokenized_dir=tokenized_dir,
        train_root=train_root,
        summary_path=summary_path,
        gate_path=gate_path,
        benchmark_root=benchmark_root,
        baseline_cache_root=baseline_cache_root,
        executor=build_executor(
            profile,
            _resolve_stage_cfg(profile, active_stages[0]) if len(active_stages) == 1 else None,
        ),
        lineage_manifest_path=lineage_manifest_path,
        logs_root=logs_root,
        run_naming_cfg=run_naming_cfg,
        run_timestamp=run_timestamp,
        naming_languages=naming_languages,
        lang_count=lang_count,
        model_slug=model_slug,
        exp_tag=exp_tag,
        base_run_label=base_run_label,
    )


def _submit_chain(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    config = _load_pipeline_config(_abs(repo_root, args.config))
    profile = dict(_resolve_profile(config, args.profile))
    run_ids = _parse_run_ids(args.run_id)
    chain_env = os.environ.copy()
    chain_env["PYTHONPATH"] = (
        f"{repo_root / 'src'}:"
        f"{repo_root}:"
        f"{repo_root / 'external' / 'Megatron-Bridge' / 'src'}:"
        f"{chain_env.get('PYTHONPATH', '')}"
    )
    chain_env["PROOF_PIPELINE_REPO_ROOT"] = str(repo_root)

    executor_cfg_raw = profile.get("executor", {})
    executor_cfg = dict(executor_cfg_raw) if isinstance(executor_cfg_raw, dict) else {}
    executor_overridden = False
    if args.executor_type:
        executor_cfg["type"] = args.executor_type
        profile["_executor_type_override"] = args.executor_type
        executor_overridden = True
    if str(args.cuda_visible_devices).strip():
        executor_cfg["cuda_visible_devices"] = str(args.cuda_visible_devices).strip()
        executor_overridden = True
    if executor_overridden:
        profile["executor"] = executor_cfg

    stage_start_idx = STAGES.index(args.start_at)
    stage_stop_idx = STAGES.index(args.stop_after)
    if stage_stop_idx < stage_start_idx:
        raise SystemExit("--stop-after must come after --start-at")
    active_stages = STAGES[stage_start_idx: stage_stop_idx + 1]

    output_root = _abs(repo_root, profile.get("output_root", f"outputs/proof_runs/{args.profile}"))
    chain_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    chain_attempt_id = _make_attempt_id(getattr(args, "attempt_id", ""), chain_timestamp)
    logs_base_root = _abs(repo_root, profile.get("logs_dir", str(output_root / "logs")))
    logs_base_root.mkdir(parents=True, exist_ok=True)
    chain_root = logs_base_root / "attempts" / chain_attempt_id / "chain"
    chain_logs_root = chain_root / "logs"
    chain_sbatch_root = chain_root / "sbatch"
    chain_slurm_root = chain_root / "slurm"
    chain_logs_root.mkdir(parents=True, exist_ok=True)
    chain_sbatch_root.mkdir(parents=True, exist_ok=True)
    chain_slurm_root.mkdir(parents=True, exist_ok=True)

    sample_cfg = profile.get("sample", {}) if isinstance(profile.get("sample", {}), dict) else {}
    sampled_dir = _abs(repo_root, sample_cfg.get("output_dir", str(output_root / "sampled")))
    sampled_manifest = sampled_dir / "sampling_manifest.json"
    naming_languages = _resolve_profile_languages(
        sample_cfg=sample_cfg,
        sample_manifest_path=sampled_manifest,
        sampled_dir=sampled_dir,
    )
    run_naming_cfg = profile.get("run_naming", {}) if isinstance(profile.get("run_naming", {}), dict) else {}
    lang_count = (
        len(naming_languages)
        if naming_languages
        else int(run_naming_cfg.get("language_count_hint", len(sample_cfg.get("languages", [])) or 0) or 0)
    )
    lang_tag = f"{max(1, lang_count)}l" if lang_count else "xl"
    job_name_prefix = _sanitize_slug(str(run_naming_cfg.get("job_name_prefix", "")).strip()) if run_naming_cfg else ""

    def _prefixed_job_name(name: str) -> str:
        return f"{job_name_prefix}_{name}" if job_name_prefix else name

    stage_short = {
        "sample": "samp",
        "tokenizer_train": "toktr",
        "tokenize": "tok",
        "audit": "audit",
        "summary": "sum",
        "train": "train",
        "cooldown_readout": "cool",
        "checkpoint_conversion": "ckpt",
        "final_loss": "floss",
        "benchmark": "bench",
        "baseline_eval": "beval",
    }

    dependency_ids: list[str] = []
    submitted: list[str] = []
    last_stage_job_ids_by_run: dict[str, str] = {}
    stage_job_ids_by_stage: dict[str, list[str]] = {}

    for stage in active_stages:
        stage_cfg = _resolve_stage_cfg(profile, stage)
        if isinstance(stage_cfg, dict) and stage_cfg.get("enabled") is False:
            continue
        if stage in {"cooldown_readout", "checkpoint_conversion", "final_loss"} and not stage_cfg:
            continue
        executor = build_executor(profile, stage_cfg)
        if not isinstance(executor, SlurmExecutor):
            raise SystemExit(f"submit-chain requires Slurm executor for stage '{stage}'")
        stage_tag = stage_short.get(stage, stage)
        dependency, benchmark_depends_directly_on_train = _chain_dependency_for_stage(
            stage=stage,
            dependency_ids=dependency_ids,
            stage_job_ids_by_stage=stage_job_ids_by_stage,
        )
        stage_job_ids: list[str] = []

        if stage == "train":
            train_args = argparse.Namespace(**vars(args))
            train_args.submit_chain = False
            train_args.attempt_id = _sanitize_slug(f"{chain_attempt_id}-train")
            train_ctx = _build_context(train_args)
            train_cfg_for_stage, _resolved_model_path, runs = resolve_training_stage_config(train_ctx)
            calibration_enabled = calibration_enabled_for_any_run(train_cfg_for_stage, runs)

            for run in runs:
                run_id, run_name_label, output_dir, log_path = describe_training_run(
                    train_ctx,
                    run=run,
                    train_cfg_for_stage=train_cfg_for_stage,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                slurm_output = chain_slurm_root / f"chain_{stage}.{run_id}.{chain_timestamp}.slurm.log"
                if calibration_enabled:
                    child_attempt_id = _sanitize_slug(f"{chain_attempt_id}-{stage}-{run_id}")
                    cmd = [
                        "python",
                        "-m",
                        "multilingual_megatron.pipeline",
                        "--config",
                        args.config,
                        "--profile",
                        args.profile,
                        "--start-at",
                        stage,
                        "--stop-after",
                        stage,
                        "--force-stage",
                        stage,
                        "--executor-type",
                        "local",
                        "--run-id",
                        run_id,
                        "--attempt-id",
                        child_attempt_id,
                    ]
                    if args.dry_run:
                        cmd.append("--dry-run")
                    run_env = chain_env
                    log_path = chain_logs_root / f"chain_{stage}.{run_id}.{chain_timestamp}.log"
                else:
                    cmd, run_env = build_training_cmd(
                        repo_root=train_ctx.repo_root,
                        env=train_ctx.env,
                        output_dir=output_dir,
                        run_cfg=run,
                        common=train_cfg_for_stage,
                        language_map_path=train_ctx.language_map_path,
                        tokenized_path=train_ctx.tokenized_dir,
                        run_name=run_name_label,
                    )
                job_id = executor.submit(
                    cmd,
                    run_env,
                    args.dry_run,
                    log_path,
                    dependency=dependency,
                    job_name=_prefixed_job_name(f"{stage_tag}{lang_tag}_{run_id}"),
                    script_dir=chain_sbatch_root,
                    slurm_output=slurm_output,
                )
                if job_id:
                    stage_job_ids.append(job_id)
                    submitted.append(job_id)
                    last_stage_job_ids_by_run[run_id] = job_id
            dependency_ids = stage_job_ids if stage_job_ids else dependency_ids
            if stage_job_ids:
                stage_job_ids_by_stage[stage] = list(stage_job_ids)
            continue

        if stage in {"train", "benchmark"} and len(run_ids) > 1:
            for run_id in run_ids:
                child_attempt_id = _sanitize_slug(f"{chain_attempt_id}-{stage}-{run_id}")
                cmd = [
                    "python",
                    "-m",
                    "multilingual_megatron.pipeline",
                    "--config",
                    args.config,
                    "--profile",
                    args.profile,
                    "--start-at",
                    stage,
                    "--stop-after",
                    stage,
                    "--force-stage",
                    stage,
                    "--executor-type",
                    "local",
                    "--run-id",
                    run_id,
                    "--attempt-id",
                    child_attempt_id,
                ]
                if args.dry_run:
                    cmd.append("--dry-run")
                log_path = chain_logs_root / f"chain_{stage}.{run_id}.{chain_timestamp}.log"
                slurm_output = chain_slurm_root / f"chain_{stage}.{run_id}.{chain_timestamp}.slurm.log"
                job_name = _prefixed_job_name(f"{stage_tag}{lang_tag}_{run_id}")
                job_dependency = dependency
                if stage == "benchmark":
                    prior_job = last_stage_job_ids_by_run.get(run_id)
                    if benchmark_depends_directly_on_train and prior_job:
                        job_dependency = f"afterany:{prior_job}" # use afterany dependency to allow benchmark to run even after training is aborted due to time limit
                job_id = executor.submit(
                    cmd,
                    chain_env,
                    args.dry_run,
                    log_path,
                    dependency=job_dependency,
                    job_name=job_name,
                    script_dir=chain_sbatch_root,
                    slurm_output=slurm_output,
                )
                if job_id:
                    stage_job_ids.append(job_id)
                    submitted.append(job_id)
                    last_stage_job_ids_by_run[run_id] = job_id
        else:
            child_attempt_id = _sanitize_slug(f"{chain_attempt_id}-{stage}")
            nested_stage_executor = (
                stage in {"tokenizer_train", "tokenize"}
                and isinstance(stage_cfg, dict)
                and bool(stage_cfg.get("parallel_slurm_jobs", False))
            )
            cmd = [
                "python",
                "-m",
                "multilingual_megatron.pipeline",
                "--config",
                args.config,
                "--profile",
                args.profile,
                "--start-at",
                stage,
                "--stop-after",
                stage,
                "--force-stage",
                stage,
                "--attempt-id",
                child_attempt_id,
            ]
            if not nested_stage_executor:
                cmd.extend(["--executor-type", "local"])
            if args.dry_run:
                cmd.append("--dry-run")
            for run_id in run_ids:
                cmd.extend(["--run-id", run_id])
            log_path = chain_logs_root / f"chain_{stage}.{chain_timestamp}.log"
            slurm_output = chain_slurm_root / f"chain_{stage}.{chain_timestamp}.slurm.log"
            if stage in {"train", "benchmark", "baseline_eval"}:
                if len(run_ids) == 1:
                    approach = run_ids[0]
                elif run_ids:
                    approach = "multi"
                else:
                    approach = "all"
            else:
                approach = "all"
            job_name = _prefixed_job_name(f"{stage_tag}{lang_tag}_{approach}")
            job_id = executor.submit(
                cmd,
                chain_env,
                args.dry_run,
                log_path,
                dependency=dependency,
                job_name=job_name,
                script_dir=chain_sbatch_root,
                slurm_output=slurm_output,
            )
            if job_id:
                stage_job_ids.append(job_id)
                submitted.append(job_id)

        dependency_ids = stage_job_ids if stage_job_ids else dependency_ids
        if stage_job_ids:
            stage_job_ids_by_stage[stage] = list(stage_job_ids)

    if submitted:
        print(f"Submitted {len(submitted)} job(s): {' -> '.join(submitted)}")
    else:
        print("No stages submitted (all disabled or dry-run).")
    return 0


def _chain_dependency_for_stage(
    *,
    stage: str,
    dependency_ids: list[str],
    stage_job_ids_by_stage: dict[str, list[str]],
) -> tuple[str | None, bool]:
    dependency_source_ids = list(dependency_ids)
    train_job_ids = stage_job_ids_by_stage.get("train", [])
    benchmark_depends_directly_on_train = (
        stage == "benchmark"
        and bool(train_job_ids)
        and dependency_source_ids == train_job_ids
    )
    dependency_mode = "afterany" if benchmark_depends_directly_on_train else "afterok"
    dependency = f"{dependency_mode}:{':'.join(dependency_source_ids)}" if dependency_source_ids else None
    return dependency, benchmark_depends_directly_on_train


def _json_load_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tokenizer_group_specs(ctx: PipelineContext) -> list[dict[str, Any]]:
    specs = ctx.tokenizer_cfg.get("tokenizers", [])
    if not isinstance(specs, list):
        raise SystemExit("tokenizer_training.tokenizers must be a list for --group-by-tokenizer")
    grouped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in specs:
        if not isinstance(raw, dict):
            continue
        tokenizer_id = str(raw.get("id", "")).strip()
        if not tokenizer_id or tokenizer_id in seen:
            continue
        seen.add(tokenizer_id)
        grouped.append(dict(raw))
    if not grouped:
        raise SystemExit("--group-by-tokenizer requires configured tokenizer_training.tokenizers")
    return grouped


def _runs_by_tokenizer(ctx: PipelineContext) -> dict[str, list[dict[str, Any]]]:
    runs = ctx.train_cfg.get("runs", [])
    if not isinstance(runs, list):
        raise SystemExit("training.runs must be a list for --group-by-tokenizer")
    mapped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        tokenizer_id = resolve_run_tokenizer_id(ctx.train_cfg, run)
        if tokenizer_id:
            mapped.setdefault(tokenizer_id, []).append(dict(run))
    return mapped


def _local_grouped_enabled(ctx: PipelineContext) -> bool:
    if bool(getattr(ctx.args, "group_by_tokenizer", False)):
        return True
    execution_cfg = ctx.profile.get("execution", {})
    if not isinstance(execution_cfg, dict):
        return False
    return bool(execution_cfg.get("local_group_by_tokenizer", False))


def _group_ctx_for_tokenizer(
    ctx: PipelineContext,
    *,
    tokenizer_spec: dict[str, Any],
    runs: list[dict[str, Any]],
    active_stages: list[str],
) -> PipelineContext:
    tokenizer_cfg = dict(ctx.tokenizer_cfg)
    tokenizer_cfg["tokenizers"] = [dict(tokenizer_spec)]
    train_cfg = dict(ctx.train_cfg)
    train_cfg["runs"] = [dict(run) for run in runs]
    return replace(ctx, tokenizer_cfg=tokenizer_cfg, train_cfg=train_cfg, active_stages=active_stages)


def _merge_tokenizer_manifest(ctx: PipelineContext, aggregate: dict[str, Any]) -> None:
    manifest_path = ctx.tokenizers_dir / "tokenizer_manifest.json"
    current = _json_load_if_exists(manifest_path)
    current_tokenizers = current.get("tokenizers", {})
    if not isinstance(current_tokenizers, dict):
        return
    aggregate.setdefault("schema_version", current.get("schema_version", 1))
    aggregate.setdefault("stage", current.get("stage", "tokenizer_train"))
    aggregate.setdefault("tokenizers", {})
    aggregate["tokenizers"].update(current_tokenizers)


def _merge_tokenized_manifest(ctx: PipelineContext, aggregate: dict[str, Any]) -> None:
    manifest_path = ctx.tokenized_dir / "tokenized_manifests.json"
    current = _json_load_if_exists(manifest_path)
    current_tokenizers = current.get("tokenizers", {})
    if not isinstance(current_tokenizers, dict):
        return
    for key in ("schema_version", "stage", "backend", "sample_shard_dir"):
        if key in current:
            aggregate.setdefault(key, current[key])
    aggregate.setdefault("tokenizers", {})
    aggregate["tokenizers"].update(current_tokenizers)


def main() -> int:
    args = _parse_args()
    if args.list_profiles:
        repo_root = _repo_root()
        config = _load_pipeline_config(_abs(repo_root, args.config))
        profiles = sorted(config.get("profiles", {}).keys())
        print(f"Available profiles ({len(profiles)}):")
        for name in profiles:
            raw = config["profiles"][name]
            extends = raw.get("extends", "")
            suffix = f"  (extends: {extends})" if extends else ""
            print(f"  {name}{suffix}")
        return 0
    if args.submit_chain:
        return _submit_chain(args)
    ctx = _build_context(args)
    run_started_perf = time.perf_counter()
    run_started_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_events_path = ctx.logs_root / "pipeline.jsonl"
    run_summary_path = ctx.logs_root / "summary.json"
    stage_results: list[dict[str, Any]] = []

    def _emit(event: str, *, stage: str | None = None, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "profile": ctx.args.profile,
            "attempt_id": ctx.env.get("PROOF_PIPELINE_ATTEMPT_ID", ""),
            "event": event,
        }
        if stage is not None:
            payload["stage"] = stage
        payload.update(fields)
        run_events_path.parent.mkdir(parents=True, exist_ok=True)
        with run_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    _emit(
        "run_start",
        dry_run=bool(ctx.args.dry_run),
        start_at=ctx.args.start_at,
        stop_after=ctx.args.stop_after,
        active_stages=ctx.active_stages,
        run_ids=ctx.args.run_id,
        logs_root=str(ctx.logs_root.resolve()),
        group_by_tokenizer=_local_grouped_enabled(ctx),
    )

    def _run_stage(stage_ctx: PipelineContext, stage: str, *, label: str | None = None) -> None:
        try:
            stage_impl = get_stage(stage)
        except KeyError:
            raise SystemExit(f"No stage handler configured for: {stage}")
        stage_label = label or stage
        stage_cfg = _resolve_stage_cfg(stage_ctx.profile, stage)
        stage_ctx.executor = build_executor(stage_ctx.profile, stage_cfg)
        stage_started_perf = time.perf_counter()
        stage_ctx.logger.info("[STAGE_START] %s", stage_label)
        _emit("stage_start", stage=stage_label)
        try:
            result = run_stage(stage_impl, stage_ctx)
        except Exception as exc:
            duration_s = time.perf_counter() - stage_started_perf
            stage_results.append({"stage": stage_label, "status": "failed", "duration_seconds": round(duration_s, 3)})
            stage_ctx.logger.error(
                "[STAGE_FAIL] %s duration_seconds=%.3f error=%s: %s",
                stage_label,
                duration_s,
                type(exc).__name__,
                exc,
            )
            _emit(
                "stage_fail",
                stage=stage_label,
                duration_seconds=round(duration_s, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        duration_s = time.perf_counter() - stage_started_perf
        status = result.status or "ok"
        stage_results.append({"stage": stage_label, "status": status, "duration_seconds": round(duration_s, 3)})
        if status == "skipped":
            reason = result.metadata.get("reason", "") if isinstance(result.metadata, dict) else ""
            stage_ctx.logger.info("[STAGE_SKIP] %s duration_seconds=%.3f reason=%s", stage_label, duration_s, reason)
            _emit("stage_skip", stage=stage_label, duration_seconds=round(duration_s, 3), reason=reason)
        else:
            stage_ctx.logger.info("[STAGE_DONE] %s duration_seconds=%.3f", stage_label, duration_s)
            _emit("stage_done", stage=stage_label, duration_seconds=round(duration_s, 3))

    def _run_tokenizer_grouped(grouped_stages: list[str]) -> None:
        group_specs = _tokenizer_group_specs(ctx)
        runs_by_tokenizer = _runs_by_tokenizer(ctx)
        needed_ids = (
            set(runs_by_tokenizer)
            if ("train" in grouped_stages or ctx.args.run_id)
            else {str(spec["id"]) for spec in group_specs}
        )
        tokenizer_manifest: dict[str, Any] = {}
        tokenized_manifest: dict[str, Any] = {}

        for spec in group_specs:
            tokenizer_id = str(spec["id"])
            if tokenizer_id not in needed_ids:
                continue
            runs = runs_by_tokenizer.get(tokenizer_id, [])
            group_ctx = _group_ctx_for_tokenizer(
                ctx,
                tokenizer_spec=spec,
                runs=runs,
                active_stages=grouped_stages,
            )
            ctx.logger.info(
                "[TOKENIZER_GROUP_START] %s runs=%s",
                tokenizer_id,
                ",".join(str(run.get("id", "")) for run in runs) or "-",
            )
            if "tokenizer_train" in grouped_stages:
                _run_stage(
                    replace(group_ctx, active_stages=["tokenizer_train"]),
                    "tokenizer_train",
                    label=f"tokenizer_train.{tokenizer_id}",
                )
                if not ctx.args.dry_run:
                    _merge_tokenizer_manifest(ctx, tokenizer_manifest)
            if "tokenize" in grouped_stages:
                _run_stage(
                    replace(group_ctx, active_stages=["tokenize"]),
                    "tokenize",
                    label=f"tokenize.{tokenizer_id}",
                )
                if not ctx.args.dry_run:
                    _merge_tokenized_manifest(ctx, tokenized_manifest)
            if "audit" in grouped_stages:
                _run_stage(
                    replace(group_ctx, active_stages=["audit"]),
                    "audit",
                    label=f"audit.{tokenizer_id}",
                )
            if "train" in grouped_stages and runs:
                _run_stage(
                    replace(group_ctx, active_stages=["train"]),
                    "train",
                    label=f"train.{tokenizer_id}",
                )
            ctx.logger.info("[TOKENIZER_GROUP_DONE] %s", tokenizer_id)

        if not ctx.args.dry_run and tokenizer_manifest.get("tokenizers"):
            _json_dump(ctx.tokenizers_dir / "tokenizer_manifest.json", tokenizer_manifest)
        if not ctx.args.dry_run and tokenized_manifest.get("tokenizers"):
            _json_dump(ctx.tokenized_dir / "tokenized_manifests.json", tokenized_manifest)

    grouped_mode = _local_grouped_enabled(ctx)
    grouped_stage_names = {"tokenizer_train", "tokenize", "audit", "train"}
    grouped_done = False
    for stage in ctx.active_stages:
        if len(ctx.active_stages) > 1 and _is_unconfigured_architecture_slot(ctx.profile, stage):
            continue
        if grouped_mode and stage in grouped_stage_names:
            if grouped_done:
                continue
            grouped_stages = [name for name in ctx.active_stages if name in grouped_stage_names]
            if "tokenize" in grouped_stages and "tokenizer_train" not in grouped_stages:
                raise SystemExit("--group-by-tokenizer start-at=tokenize requires tokenizer_train in the active stage range")
            _run_tokenizer_grouped(grouped_stages)
            grouped_done = True
            continue
        _run_stage(ctx, stage)

    total_duration_s = time.perf_counter() - run_started_perf
    run_summary: dict[str, Any] = {
        "profile": ctx.args.profile,
        "attempt_id": ctx.env.get("PROOF_PIPELINE_ATTEMPT_ID", ""),
        "run_started_utc": run_started_utc,
        "run_finished_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dry_run": bool(ctx.args.dry_run),
        "start_at": ctx.args.start_at,
        "stop_after": ctx.args.stop_after,
        "total_duration_seconds": round(total_duration_s, 3),
        "stages": stage_results,
        "logs_root": str(ctx.logs_root.resolve()),
        "events_jsonl": str(run_events_path.resolve()),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
    _emit("run_done", total_duration_seconds=round(total_duration_s, 3), summary_json=str(run_summary_path.resolve()))
    ctx.logger.info("[RUN_SUMMARY] %s", run_summary_path)
    ctx.logger.info("[DONE] profile=%s", ctx.args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
