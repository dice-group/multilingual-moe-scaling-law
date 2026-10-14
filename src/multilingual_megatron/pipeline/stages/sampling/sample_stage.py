from __future__ import annotations

import csv

from ..runtime import (
    _abs,
    _dump_json,
    _missing_modules,
    _resolve_profile_languages,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ...checks import check_sample
from ...context import PipelineContext
from ...types import FunctionStage


def _module_cmd(module: str) -> list[str]:
    return ["python", "-m", module]


def _resolve_data_files(repo_root, raw: object) -> str:
    files = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not files:
        return ""
    return ",".join(str(_abs(repo_root, item)) for item in files)


def _missing_fineweb_outputs(plan_csv, shard_dir) -> list[str]:
    if not plan_csv.exists():
        return []
    existing = {
        path.parent.name
        for path in list(shard_dir.rglob("*.jsonl")) + list(shard_dir.rglob("*.jsonl.gz"))
        if path.stat().st_size > 0
    }
    missing: list[str] = []
    with plan_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            subset = str(row.get("subset", "")).strip()
            if subset and subset not in existing:
                missing.append(subset)
    return missing


def stage_sample(ctx: PipelineContext) -> None:
    if "sample" in ctx.active_stages and bool(ctx.sample_cfg.get("enabled", True)):
        missing = _missing_modules(["datasets"])
        if missing:
            raise SystemExit(f"Missing Python modules for sample stage: {missing}")
        signature = _stage_signature({"stage": "sample", "cfg": ctx.sample_cfg, "language_map_path": str(ctx.language_map_path)})
        artifacts = [ctx.sampled_manifest]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="sample",
            signature=signature,
            expected_artifacts=artifacts,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] sample stage (cache hit)")
        else:
            source_type = str(ctx.sample_cfg.get("source_type", "hf"))
            if source_type == "corpus_blend":
                source_inventory = _abs(
                    ctx.repo_root,
                    str(ctx.sample_cfg.get("source_inventory", "configs/data_sources/source_inventory.json")),
                )
                corpus_blend = _abs(ctx.repo_root, str(ctx.sample_cfg.get("corpus_blend", "")))
                if not str(corpus_blend).strip():
                    raise SystemExit("sample.corpus_blend is required when sample.source_type='corpus_blend'")

                availability_json = ctx.sampled_dir / "source_availability.json"
                availability_tsv = ctx.sampled_dir / "source_availability.tsv"
                corpus_manifest = ctx.sampled_dir / "corpus_manifest.json"
                sampling_plan_tsv = ctx.sampled_dir / "corpus_sampling_plan.tsv"
                audit_json = ctx.sampled_dir / "corpus_audit.json"
                audit_tsv = ctx.sampled_dir / "corpus_audit.tsv"
                langs = ",".join(ctx.sample_cfg.get("languages", []))

                source_ids = str(ctx.sample_cfg.get("source_ids", "")).strip()

                availability_cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.build_source_availability"),
                    "--source-inventory",
                    str(source_inventory),
                    "--corpus-blend",
                    str(corpus_blend),
                    "--output-json",
                    str(availability_json),
                    "--output-tsv",
                    str(availability_tsv),
                    "--repo-root",
                    str(ctx.repo_root),
                ]
                if source_ids:
                    availability_cmd.extend(["--source-ids", source_ids])
                _run_stage_cmd(
                    cmd=availability_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="sample",
                    suffix="availability",
                    executor=ctx.executor,
                )

                manifest_cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.build_corpus_manifest"),
                    "--source-inventory",
                    str(source_inventory),
                    "--corpus-blend",
                    str(corpus_blend),
                    "--availability-json",
                    str(availability_json),
                    "--language-map",
                    str(ctx.language_map_path),
                    "--output-json",
                    str(corpus_manifest),
                    "--output-plan-tsv",
                    str(sampling_plan_tsv),
                    "--repo-root",
                    str(ctx.repo_root),
                ]
                if langs:
                    manifest_cmd.extend(["--languages", langs])
                _run_stage_cmd(
                    cmd=manifest_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="sample",
                    suffix="manifest",
                    executor=ctx.executor,
                )

                sample_cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.sample_corpus_release"),
                    "--source-inventory",
                    str(source_inventory),
                    "--corpus-manifest",
                    str(corpus_manifest),
                    "--output-dir",
                    str(ctx.sampled_dir),
                    "--repo-root",
                    str(ctx.repo_root),
                ]
                if bool(ctx.sample_cfg.get("fail_on_missing", True)):
                    sample_cmd.append("--fail-on-missing")
                if bool(ctx.sample_cfg.get("fail_on_shortfall", True)):
                    sample_cmd.append("--fail-on-shortfall")
                _run_stage_cmd(
                    cmd=sample_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="sample",
                    suffix="release",
                    executor=ctx.executor,
                )

                audit_cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.audit_corpus_release"),
                    "--manifest",
                    str(ctx.sampled_manifest),
                    "--output-json",
                    str(audit_json),
                    "--output-tsv",
                    str(audit_tsv),
                    "--repo-root",
                    str(ctx.repo_root),
                    "--fail-on-empty",
                ]
                _run_stage_cmd(
                    cmd=audit_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="sample",
                    suffix="audit",
                    executor=ctx.executor,
                )
            elif source_type == "fineweb_datatrove":
                missing = _missing_modules(["pandas", "datatrove"])
                if missing and (not ctx.args.dry_run):
                    raise SystemExit(f"Missing Python modules for datatrove planning: {missing}")
                availability_csv = _abs(
                    ctx.repo_root,
                    str(ctx.sample_cfg.get("availability_csv", "configs/data_sources/fineweb2-language-distribution.csv")),
                )
                plan_csv = ctx.sampled_dir / "sampling_plan.csv"
                report_json = ctx.sampled_dir / "sampling_report.json"
                availability_tsv = ctx.sampled_dir / "language_availability.tsv"
                langs = ",".join(ctx.sample_cfg.get("languages", []))
                plan_cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.build_fineweb_sampling_plan"),
                    "--availability-csv",
                    str(availability_csv),
                    "--output-plan-csv",
                    str(plan_csv),
                    "--output-report-json",
                    str(report_json),
                    "--output-availability-tsv",
                    str(availability_tsv),
                    "--languages",
                    langs,
                    "--language-map",
                    str(ctx.language_map_path),
                    "--max-languages",
                    str(int(ctx.sample_cfg.get("max_languages", 0))),
                    "--seed",
                    str(int(ctx.sample_cfg.get("seed", 42))),
                    "--target-docs-per-language",
                    str(int(ctx.sample_cfg.get("target_docs_per_language", 12000))),
                    "--fairness-policy",
                    str(ctx.sample_cfg.get("fairness_policy", "equal_documents")),
                    "--alpha",
                    str(float(ctx.sample_cfg.get("alpha", 0.3))),
                    "--split",
                    str(ctx.sample_cfg.get("availability_split", "train")),
                ]
                max_total_docs = int(ctx.sample_cfg.get("max_total_documents", 0))
                if max_total_docs > 0:
                    plan_cmd.extend(["--max-total-docs", str(max_total_docs)])
                if bool(ctx.sample_cfg.get("auto_adjust_target_to_min_available", True)):
                    plan_cmd.append("--auto-adjust-target-to-min-available")
                if bool(ctx.sample_cfg.get("allow_upsample", False)):
                    plan_cmd.append("--allow-upsample")
                if bool(ctx.sample_cfg.get("fail_on_missing", True)):
                    plan_cmd.append("--fail-on-missing")
                _run_stage_cmd(
                    cmd=plan_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="sample",
                    suffix="plan",
                    executor=ctx.executor,
                )
                datatrove_cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.datatrove_sampler"),
                    str(plan_csv),
                    str(ctx.sampled_dir),
                    "--num-proc",
                    str(int(ctx.sample_cfg.get("num_proc", 12))),
                    "--logging-dir",
                    str(ctx.logs_root / "stages" / "sample" / "datatrove_internal"),
                ]
                max_sample = int(ctx.sample_cfg.get("max_sample_per_language", 0))
                if max_sample > 0:
                    datatrove_cmd.extend(["--max-sample", str(max_sample)])
                datatrove_env = dict(ctx.env)
                if bool(ctx.sample_cfg.get("datatrove_internal_slurm_enabled", False)):
                    datatrove_env["SLURM_ACTIVE"] = "true"
                missing_outputs = [] if ctx.args.dry_run else _missing_fineweb_outputs(plan_csv, ctx.sample_shard_dir)
                if (not ctx.args.dry_run) and not missing_outputs:
                    ctx.logger.info("[SKIP] datatrove sampling (all planned FineWeb shard files already exist)")
                else:
                    if missing_outputs:
                        ctx.logger.info("[INFO] datatrove missing FineWeb subsets=%s", ",".join(missing_outputs))
                    _run_stage_cmd(
                        cmd=datatrove_cmd,
                        env=datatrove_env,
                        dry_run=ctx.args.dry_run,
                        logs_root=ctx.logs_root,
                        stage="sample",
                        suffix="datatrove",
                        executor=ctx.executor,
                    )
                upsample_cap_factor = float(ctx.sample_cfg.get("upsample_cap_factor", 1.0))
                if bool(ctx.sample_cfg.get("allow_upsample", False)) and upsample_cap_factor > 1.0:
                    upsample_report_json = ctx.sampled_dir / "upsample_report.json"
                    upsample_report_tsv = ctx.sampled_dir / "upsample_report.tsv"
                    upsample_cmd = [
                        *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.apply_fineweb_upsample_cap"),
                        "--plan-csv",
                        str(plan_csv),
                        "--samples-root",
                        str(ctx.sample_shard_dir),
                        "--cap-factor",
                        str(upsample_cap_factor),
                        "--seed",
                        str(int(ctx.sample_cfg.get("seed", 42))),
                        "--report-json",
                        str(upsample_report_json),
                        "--report-tsv",
                        str(upsample_report_tsv),
                    ]
                    _run_stage_cmd(
                        cmd=upsample_cmd,
                        env=ctx.env,
                        dry_run=ctx.args.dry_run,
                        logs_root=ctx.logs_root,
                        stage="sample",
                        suffix="upsample",
                        executor=ctx.executor,
                    )
                if not ctx.args.dry_run:
                    data_files = [
                        str(path.resolve())
                        for path in (
                            sorted(ctx.sample_shard_dir.rglob("*.jsonl"))
                            + sorted(ctx.sample_shard_dir.rglob("*.jsonl.gz"))
                        )
                    ]
                    upsample_report_json = ctx.sampled_dir / "upsample_report.json"
                    upsample_report_tsv = ctx.sampled_dir / "upsample_report.tsv"
                    manifest_payload = {
                        "source_type": source_type,
                        "shard_dir": str(ctx.sample_shard_dir.resolve()),
                        "sampling_plan_csv": str(plan_csv.resolve()),
                        "sampling_report_json": str(report_json.resolve()),
                        "availability_tsv": str(availability_tsv.resolve()),
                        "upsample_report_json": str(upsample_report_json.resolve()) if upsample_report_json.exists() else "",
                        "upsample_report_tsv": str(upsample_report_tsv.resolve()) if upsample_report_tsv.exists() else "",
                        "data_files": data_files,
                    }
                    _dump_json(ctx.sampled_manifest, manifest_payload)
            else:
                cmd = [
                    *_module_cmd("multilingual_megatron.pipeline.stages.sampling.tools.sample_multilingual_corpus"),
                    "--output-dir",
                    str(ctx.sampled_dir),
                    "--source-type",
                    source_type,
                    "--dataset-name",
                    str(ctx.sample_cfg.get("dataset_name", "c4")),
                    "--dataset-config",
                    str(ctx.sample_cfg.get("dataset_config", "multilingual")),
                    "--dataset-split",
                    str(ctx.sample_cfg.get("dataset_split", "train")),
                    "--data-files",
                    _resolve_data_files(ctx.repo_root, ctx.sample_cfg.get("data_files", "")),
                    "--language-columns",
                    str(ctx.sample_cfg.get("language_columns", "lang,language,subset")),
                    "--text-column",
                    str(ctx.sample_cfg.get("text_column", "text")),
                    "--languages",
                    ",".join(ctx.sample_cfg.get("languages", [])),
                    "--language-map",
                    str(ctx.language_map_path),
                    "--max-languages",
                    str(int(ctx.sample_cfg.get("max_languages", 0))),
                    "--sampling-mode",
                    str(ctx.sample_cfg.get("sampling_mode", "equal_documents")),
                    "--per-language-target",
                    str(int(ctx.sample_cfg.get("per_language_target", 200))),
                    "--total-target",
                    str(int(ctx.sample_cfg.get("total_target", 2400))),
                    "--alpha",
                    str(float(ctx.sample_cfg.get("alpha", 0.3))),
                    "--min-per-language",
                    str(int(ctx.sample_cfg.get("min_per_language", 50))),
                    "--max-per-language",
                    str(int(ctx.sample_cfg.get("max_per_language", 100000))),
                    "--max-scan",
                    str(int(ctx.sample_cfg.get("max_scan", 250000))),
                    "--reservoir-cap-factor",
                    str(float(ctx.sample_cfg.get("reservoir_cap_factor", 1.5))),
                    "--seed",
                    str(int(ctx.sample_cfg.get("seed", 42))),
                ]
                if bool(ctx.sample_cfg.get("allow_upsample", True)):
                    cmd.append("--allow-upsample")
                if bool(ctx.sample_cfg.get("fail_on_missing", True)):
                    cmd.append("--fail-on-missing")
                _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="sample", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "sample", signature, [str(ctx.sampled_manifest)])
            check_sample(ctx.sampled_dir, ctx.sample_shard_dir, dry_run=ctx.args.dry_run)


stage = FunctionStage("sample", stage_sample, enabled_key="sample_cfg")

__all__ = ["stage", "stage_sample"]
