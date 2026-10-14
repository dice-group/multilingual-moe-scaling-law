from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..common import ensure_lm_eval_layout, lm_eval_harness_root
from ..runtime import (
    _abs,
    _build_run_label,
    _dump_json,
    _load_json,
    _resolve_profile_languages,
    _run_stage_cmd,
    _save_stage_state,
    _select_tasks_for_languages,
    _should_skip_stage,
    _stage_signature,
)
from ...checks import check_benchmark
from ...context import PipelineContext
from ...types import ArtifactSpec, StagePlan, StageResult


def _read_selected_tasks_file(path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _write_selected_tasks_file(path: Path, tasks: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tasks) + ("\n" if tasks else ""), encoding="utf-8")


def _extract_tasks_from_payload(payload: object) -> set[str]:
    tasks: set[str] = set()
    if not isinstance(payload, dict):
        return tasks
    task_name = payload.get("task")
    if isinstance(task_name, str):
        tasks.add(task_name)
    results = payload.get("results")
    if isinstance(results, dict):
        tasks.update(task for task in results if isinstance(task, str) and not task.startswith("_"))
    return tasks


def _load_results_payloads_from_dir(path: Path) -> list[dict[str, object]]:
    candidates = sorted(path.rglob("results_*.json"))
    payloads: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _collect_tasks_from_output_dir(path: Path) -> tuple[list[Path], set[str]]:
    files = sorted(path.glob("*_lm_eval.jsonl"))
    files.extend(sorted(path.glob("no_language_ids.json")))
    files.extend(sorted(path.glob("with_language_ids_*.json")))
    task_seen: set[str] = set()
    for file in files:
        if file.is_dir():
            for payload in _load_results_payloads_from_dir(file):
                task_seen.update(_extract_tasks_from_payload(payload))
            continue
        if file.suffix == ".jsonl":
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_seen.update(_extract_tasks_from_payload(payload))
            continue
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        task_seen.update(_extract_tasks_from_payload(payload))
    return files, task_seen


def _explicit_benchmark_tasks(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("tasks", [])
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _limit_tasks(tasks: list[str], cfg: dict[str, Any]) -> list[str]:
    raw_limit = cfg.get("task_limit", cfg.get("max_tasks", None))
    if raw_limit is None:
        return tasks
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return tasks
    if limit <= 0:
        return tasks
    return tasks[:limit]


def _resolve_checkpoint_export_index(ctx: PipelineContext) -> Path:
    raw = str(ctx.benchmark_cfg.get("checkpoint_exports_index", "")).strip()
    if raw:
        return _abs(ctx.repo_root, raw)
    conversion_cfg = getattr(ctx, "profile", {}).get("checkpoint_conversion", {})
    if isinstance(conversion_cfg, dict):
        output_dir = str(conversion_cfg.get("output_dir", "")).strip()
        if output_dir:
            return _abs(ctx.repo_root, output_dir) / "checkpoint_exports.json"
    return ctx.output_root / "checkpoint_exports" / "checkpoint_exports.json"


def _resolve_export_manifest_for_run(ctx: PipelineContext, index_payload: dict[str, Any], run_id: str) -> Path:
    runs = index_payload.get("runs", {})
    if not isinstance(runs, dict):
        raise SystemExit(f"checkpoint export index has no runs object: {_resolve_checkpoint_export_index(ctx)}")
    run_info = runs.get(run_id)
    if not isinstance(run_info, dict):
        raise SystemExit(f"checkpoint export index has no entry for run '{run_id}': {_resolve_checkpoint_export_index(ctx)}")
    manifest = str(run_info.get("export_manifest", "")).strip()
    if not manifest:
        raise SystemExit(f"checkpoint export index entry for run '{run_id}' has no export_manifest")
    return Path(manifest)


def _resolve_eval_script(ctx: PipelineContext) -> Path:
    raw = str(ctx.benchmark_cfg.get("eval_script", "")).strip()
    if raw:
        return _abs(ctx.repo_root, raw)
    raise SystemExit(
        "Legacy benchmark backend requires benchmark.eval_script. "
        "Use benchmark.backend=bridge_artifacts for smoke summaries or "
        "benchmark.backend=bridge_hf_lm_eval for HF-export lm-eval runs."
    )


def _extract_test_data_file(config_path: Path) -> str | None:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("test:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _resolve_lm_eval_relative_path(include_path: Path, rel_path: Path) -> Path | None:
    if not rel_path.parts or rel_path.parts[0] != "lm_eval":
        return None
    current = include_path
    while current.name != "lm_eval" and current.parent != current:
        current = current.parent
    if current.name != "lm_eval":
        return None
    return current.parent / rel_path


def _validate_flores_task_data_paths(repo_root: Path, include_path: Path, selected_tasks: list[str]) -> None:
    for task in selected_tasks:
        if not task.startswith("flores_en_perplexity_"):
            continue
        task_yaml = include_path / f"{task}.yaml"
        if not task_yaml.exists():
            raise SystemExit(f"Missing FLORES task config: {task_yaml}")
        data_file = _extract_test_data_file(task_yaml)
        if not data_file:
            raise SystemExit(f"FLORES task config missing dataset test path: {task_yaml}")
        data_path = Path(data_file)
        if not data_path.is_absolute():
            resolved = _resolve_lm_eval_relative_path(include_path, data_path)
            data_path = resolved if resolved is not None else (repo_root / data_path)
        if not data_path.exists():
            raise SystemExit(
                f"FLORES dataset path not found for task '{task}': {data_path}. "
                "Regenerate/repair flores_en_perplexity YAML configs."
            )


def _validate_task_yaml_presence(include_path: Path, selected_tasks: list[str], prefix: str, label: str) -> None:
    for task in selected_tasks:
        if not task.startswith(prefix):
            continue
        task_yaml = include_path / f"{task}.yaml"
        if not task_yaml.exists():
            raise SystemExit(f"Missing {label} task config: {task_yaml}")


def _ensure_external_task_configs(repo_root: Path) -> None:
    harness_root = lm_eval_harness_root(repo_root)
    if harness_root is None:
        raise SystemExit("Missing external lm-evaluation-harness submodule under external/lm-evaluation-harness")
    generator = harness_root / "lm_eval" / "tasks" / "generate_local_task_configs.py"
    if not generator.exists():
        raise SystemExit(f"Missing external task generator: {generator}")
    subprocess.run([sys.executable, str(generator)], check=True, cwd=repo_root)


def _default_lm_eval_include_dirs(repo_root: Path, selected_tasks: list[str]) -> list[Path]:
    harness_root = lm_eval_harness_root(repo_root)
    if harness_root is None:
        return []
    task_root = harness_root / "lm_eval" / "tasks"
    include_dirs: list[Path] = []
    if any(task.startswith("flores_en_perplexity_") for task in selected_tasks):
        include_dirs.append(task_root / "flores_en_perplexity" / "configs")
    if any(task.startswith("sib200_") for task in selected_tasks):
        include_dirs.append(task_root / "sib200" / "configs")
    if any(task.startswith("taxi1500_") for task in selected_tasks):
        include_dirs.append(task_root / "taxi1500" / "configs")
    return include_dirs


def _resolve_benchmark_languages(ctx: PipelineContext) -> list[str]:
    langs = _resolve_profile_languages(
        sample_cfg=ctx.sample_cfg,
        sample_manifest_path=ctx.sampled_manifest,
        sampled_dir=ctx.sampled_dir,
    )
    if langs:
        return langs

    check_path = ctx.tokenized_dir / "_check.json"
    if check_path.exists():
        try:
            payload = json.loads(check_path.read_text(encoding="utf-8"))
            samples = payload.get("samples_per_language", {})
            if isinstance(samples, dict) and samples:
                return sorted({str(k) for k in samples.keys() if str(k).strip()})
        except Exception:
            pass

    token_cfg = ctx.tokenize_cfg
    csv_path = str(token_cfg.get("languages_csv", "")).strip()
    if csv_path:
        column = str(token_cfg.get("languages_csv_column", "subset")).strip() or "subset"
        split_value = str(token_cfg.get("languages_csv_split", "train")).strip()
        split_column = str(token_cfg.get("languages_csv_split_column", "split")).strip() or "split"
        languages: set[str] = set()
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and column in reader.fieldnames:
                    for row in reader:
                        if split_value:
                            if str(row.get(split_column, "")).strip() != split_value:
                                continue
                        value = str(row.get(column, "")).strip()
                        if value:
                            languages.add(value)
        except Exception:
            languages = set()
        extras = token_cfg.get("languages_extra", [])
        if isinstance(extras, list):
            for item in extras:
                if isinstance(item, str) and item.strip():
                    languages.add(item.strip())
        if languages:
            return sorted(languages)

    return []


def _stage_bridge_artifact_benchmark(
    ctx: PipelineContext,
    *,
    runs: list[dict[str, object]],
    signature: str,
    benchmark_summary_path: Path,
    expected: list[Path],
) -> None:
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="benchmark",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] benchmark stage (cache hit)")
        return

    ctx.benchmark_root.mkdir(parents=True, exist_ok=True)
    benchmark_rows: list[dict[str, object]] = []
    eval_limit = str(ctx.benchmark_cfg.get("eval_limit", ctx.benchmark_cfg.get("limit", ctx.env.get("LM_EVAL_LIMIT", "")))).strip()
    for run in runs:
        run_id = str(run["id"])
        train_dir = ctx.train_root / run_id
        out_dir = ctx.benchmark_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if ctx.args.dry_run:
            benchmark_rows.append(
                {
                    "run_id": run_id,
                    "evaluated_checkpoint": str(train_dir.resolve()),
                    "output_dir": str(out_dir.resolve()),
                    "num_tasks_expected": 1,
                    "num_tasks_seen": 1,
                    "missing_tasks": [],
                    "metrics": {"run_id": run_id, "dry_run": True},
                    "eval_limit": eval_limit,
                }
            )
            continue
        results_path = train_dir / "train_results.json"
        state_path = train_dir / "trainer_state.json"
        done_path = train_dir / ".training_complete"
        checkpoint_dir = train_dir / "checkpoints"
        if not results_path.exists():
            raise SystemExit(f"Benchmark requires train_results.json for run '{run_id}': {results_path}")
        train_results = _load_json(results_path)
        trainer_state = _load_json(state_path) if state_path.exists() else {}
        metrics = {
            "run_id": run_id,
            "backend": train_results.get("backend"),
            "recipe": train_results.get("recipe"),
            "dataset": train_results.get("dataset"),
            "train_runtime": train_results.get("train_runtime"),
            "train_loss": train_results.get("train_loss"),
            "global_step": trainer_state.get("global_step"),
            "num_input_tokens_seen": train_results.get("num_input_tokens_seen"),
            "training_complete": done_path.exists(),
            "checkpoint_dir_exists": checkpoint_dir.exists(),
            "eval_limit": eval_limit,
        }
        _dump_json(out_dir / "metrics.json", metrics)
        benchmark_rows.append(
            {
                "run_id": run_id,
                "evaluated_checkpoint": str(train_dir.resolve()),
                "output_dir": str(out_dir.resolve()),
                "num_tasks_expected": 1,
                "num_tasks_seen": 1 if done_path.exists() else 0,
                "missing_tasks": [] if done_path.exists() else ["training_complete"],
                "metrics": metrics,
            }
        )

    if not ctx.args.dry_run:
        _dump_json(
            benchmark_summary_path,
            {
                "backend": "bridge_artifacts",
                "num_selected_tasks": 1,
                "eval_limit": eval_limit,
                "tasks_file": "",
                "selected_tasks_file": "",
                "selected_tasks_summary": "",
                "runs": benchmark_rows,
            },
        )
        _save_stage_state(ctx.state_path, "benchmark", signature, [str(path) for path in expected])
        check_benchmark(ctx.benchmark_root, benchmark_summary_path, dry_run=ctx.args.dry_run)


def _select_bridge_hf_tasks(ctx: PipelineContext, tasks_file: Path) -> tuple[list[str], dict[str, Any]]:
    explicit = _explicit_benchmark_tasks(ctx.benchmark_cfg)
    if explicit:
        selected = explicit
        mode = "explicit"
    else:
        all_tasks = _read_selected_tasks_file(tasks_file)
        if bool(ctx.benchmark_cfg.get("select_tasks_by_language", True)):
            langs = _resolve_benchmark_languages(ctx)
            if not langs and bool(ctx.benchmark_cfg.get("fail_if_no_languages", False)):
                raise SystemExit("Benchmark stage could not resolve trained language list.")
            if langs:
                selected = _select_tasks_for_languages(
                    tasks_file=tasks_file,
                    languages=langs,
                    include_belebele=bool(ctx.benchmark_cfg.get("include_belebele", True)),
                    include_sib200=bool(ctx.benchmark_cfg.get("include_sib200", True)),
                    include_taxi1500=bool(ctx.benchmark_cfg.get("include_taxi1500", True)),
                    include_flores=bool(ctx.benchmark_cfg.get("include_flores", True)),
                    flores_mode=str(ctx.benchmark_cfg.get("flores_mode", "both")),
                )
                mode = "language_filtered"
            else:
                selected = all_tasks
                mode = "all_tasks_no_languages"
        else:
            selected = all_tasks
            mode = "all_tasks"
    selected = _limit_tasks(selected, ctx.benchmark_cfg)
    summary = {
        "mode": mode,
        "num_selected_tasks": len(selected),
        "tasks": selected,
    }
    return selected, summary


def _stage_bridge_hf_lm_eval_benchmark(
    ctx: PipelineContext,
    *,
    runs: list[dict[str, object]],
    signature: str,
    selected_tasks_file: Path,
    selected_tasks_summary: Path,
    benchmark_summary_path: Path,
    expected: list[Path],
) -> None:
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="benchmark",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] benchmark stage (cache hit)")
        return

    ctx.benchmark_root.mkdir(parents=True, exist_ok=True)
    tasks_file = _abs(ctx.repo_root, str(ctx.benchmark_cfg.get("tasks_file", "configs/lm_eval/lm_eval_tasks.txt")))
    if not tasks_file.exists() and not ctx.args.dry_run:
        raise SystemExit(f"Benchmark tasks file not found: {tasks_file}")
    selected_tasks, task_summary = _select_bridge_hf_tasks(ctx, tasks_file)
    if not selected_tasks and bool(ctx.benchmark_cfg.get("fail_if_no_tasks", True)) and not ctx.args.dry_run:
        raise SystemExit("Benchmark stage selected zero tasks.")
    if not ctx.args.dry_run:
        _write_selected_tasks_file(selected_tasks_file, selected_tasks)
        _dump_json(selected_tasks_summary, task_summary)

    checkpoint_index_path = _resolve_checkpoint_export_index(ctx)
    checkpoint_index = _load_json(checkpoint_index_path) if checkpoint_index_path.exists() else {}
    if not checkpoint_index and not ctx.args.dry_run:
        raise SystemExit(f"Bridge HF benchmark requires checkpoint export index: {checkpoint_index_path}")

    eval_limit = str(ctx.benchmark_cfg.get("eval_limit", ctx.benchmark_cfg.get("limit", ""))).strip()
    batch_size = str(ctx.benchmark_cfg.get("batch_size", "auto"))
    tokenizer = str(ctx.benchmark_cfg.get("tokenizer", ctx.benchmark_cfg.get("hf_tokenizer", ""))).strip()
    lm_eval_bin = str(ctx.benchmark_cfg.get("lm_eval_bin", "lm_eval"))
    seed = str(int(ctx.benchmark_cfg.get("seed", 42)))
    extra_args = str(ctx.benchmark_cfg.get("extra_args", "")).strip()
    include_paths = ctx.benchmark_cfg.get("include_paths", ctx.benchmark_cfg.get("include_path", []))
    if isinstance(include_paths, str):
        include_paths = [include_paths] if include_paths.strip() else []
    if not isinstance(include_paths, list):
        include_paths = []

    benchmark_rows: list[dict[str, object]] = []
    for run in runs:
        run_id = str(run["id"])
        out_dir = ctx.benchmark_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if ctx.args.dry_run:
            benchmark_rows.append(
                {
                    "run_id": run_id,
                    "evaluated_checkpoint": "",
                    "output_dir": str(out_dir.resolve()),
                    "num_tasks_expected": len(selected_tasks),
                    "num_tasks_seen": len(selected_tasks),
                    "missing_tasks": [],
                    "metrics": {"run_id": run_id, "dry_run": True},
                }
            )
            continue

        export_manifest = _resolve_export_manifest_for_run(ctx, checkpoint_index, run_id)
        export_payload = _load_json(export_manifest)
        hf_export_dir = Path(str(export_payload.get("hf_export_dir", "")))
        if str(export_payload.get("status", "")) != "hf_exported" or not hf_export_dir.exists():
            raise SystemExit(f"Run '{run_id}' does not have a completed HF export: {export_manifest}")
        cmd = [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.benchmarking.tools.run_bridge_hf_lm_eval",
            "--checkpoint-export-manifest",
            str(export_manifest),
            "--tasks-file",
            str(selected_tasks_file),
            "--output-dir",
            str(out_dir),
            "--batch-size",
            batch_size,
            "--tokenizer",
            tokenizer,
            "--lm-eval-bin",
            lm_eval_bin,
            "--seed",
            seed,
            "--repo-root",
            str(ctx.repo_root),
        ]
        if eval_limit:
            cmd.extend(["--limit", eval_limit])
        for include_path in include_paths:
            value = str(include_path).strip()
            if value:
                cmd.extend(["--include-path", str(_abs(ctx.repo_root, value))])
        if extra_args:
            cmd.extend(["--extra-args", extra_args])
        _run_stage_cmd(
            cmd=cmd,
            env=ctx.env,
            dry_run=ctx.args.dry_run,
            logs_root=ctx.logs_root,
            stage="benchmark",
            suffix=f"{run_id}.bridge_hf_lm_eval",
            executor=ctx.executor,
        )
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Bridge HF lm-eval did not write metrics: {metrics_path}")
        metrics = _load_json(metrics_path)
        benchmark_rows.append(
            {
                "run_id": run_id,
                "evaluated_checkpoint": str(hf_export_dir.resolve()),
                "checkpoint_export_manifest": str(export_manifest.resolve()),
                "output_dir": str(out_dir.resolve()),
                "num_tasks_expected": metrics.get("num_tasks_expected", len(selected_tasks)),
                "num_tasks_seen": metrics.get("num_tasks_seen", 0),
                "missing_tasks": metrics.get("missing_tasks", selected_tasks),
                "metrics": metrics,
            }
        )

    if not ctx.args.dry_run:
        _dump_json(
            benchmark_summary_path,
            {
                "backend": "bridge_hf_lm_eval",
                "tasks_file": str(tasks_file.resolve()),
                "selected_tasks_file": str(selected_tasks_file.resolve()),
                "selected_tasks_summary": str(selected_tasks_summary.resolve()),
                "checkpoint_exports_index": str(checkpoint_index_path.resolve()),
                "num_selected_tasks": len(selected_tasks),
                "eval_limit": eval_limit,
                "runs": benchmark_rows,
            },
        )
        _save_stage_state(ctx.state_path, "benchmark", signature, [str(path) for path in expected])
        check_benchmark(ctx.benchmark_root, benchmark_summary_path, dry_run=ctx.args.dry_run)


def stage_benchmark(ctx: PipelineContext) -> None:
    if "benchmark" in ctx.active_stages and bool(ctx.benchmark_cfg.get("enabled", False)):
        resolved_eval_tokenizer = str(ctx.benchmark_cfg.get("tokenizer", "")).strip()
        if not resolved_eval_tokenizer:
            resolved_eval_tokenizer = str(ctx.tokenize_cfg.get("model_name", ""))
        signature = _stage_signature(
            {
                "stage": "benchmark",
                "cfg": ctx.benchmark_cfg,
                "resolved_eval_tokenizer": resolved_eval_tokenizer,
                "sample_manifest": str(ctx.sampled_manifest),
                "train_runs": [run.get("id") for run in ctx.train_cfg.get("runs", [])],
            }
        )
        selected_tasks_file = ctx.benchmark_root / "selected_tasks.txt"
        selected_tasks_summary = ctx.benchmark_root / "selected_tasks_summary.json"
        if ctx.args.run_id and len(ctx.args.run_id) == 1:
            run_id = str(ctx.args.run_id[0])
            selected_tasks_file = ctx.benchmark_root / f"selected_tasks.{run_id}.txt"
            selected_tasks_summary = ctx.benchmark_root / f"selected_tasks_summary.{run_id}.json"
        benchmark_summary_path = ctx.benchmark_root / "benchmark_summary.json"
        all_runs = ctx.train_cfg.get("runs", [])
        if ctx.args.run_id:
            run_id_set = {str(rid) for rid in ctx.args.run_id}
            runs = [run for run in all_runs if str(run.get("id")) in run_id_set]
        else:
            runs = all_runs
        expected = [selected_tasks_file, selected_tasks_summary, benchmark_summary_path]
        expected.extend([ctx.benchmark_root / str(run["id"]) for run in runs])
        benchmark_backend = str(ctx.benchmark_cfg.get("backend", "")).strip().lower()
        if benchmark_backend in {"bridge_artifacts", "training_artifacts"}:
            artifact_expected = [benchmark_summary_path]
            artifact_expected.extend([ctx.benchmark_root / str(run["id"]) / "metrics.json" for run in runs])
            _stage_bridge_artifact_benchmark(
                ctx,
                runs=runs,
                signature=signature,
                benchmark_summary_path=benchmark_summary_path,
                expected=artifact_expected,
            )
            return
        if benchmark_backend == "bridge_hf_lm_eval":
            bridge_hf_expected = [selected_tasks_file, selected_tasks_summary, benchmark_summary_path]
            bridge_hf_expected.extend([ctx.benchmark_root / str(run["id"]) / "metrics.json" for run in runs])
            _stage_bridge_hf_lm_eval_benchmark(
                ctx,
                runs=runs,
                signature=signature,
                selected_tasks_file=selected_tasks_file,
                selected_tasks_summary=selected_tasks_summary,
                benchmark_summary_path=benchmark_summary_path,
                expected=bridge_hf_expected,
            )
            return
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="benchmark",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] benchmark stage (cache hit)")
        else:
            ensure_lm_eval_layout(ctx.repo_root, ctx.logger)
            tasks_file = _abs(ctx.repo_root, str(ctx.benchmark_cfg["tasks_file"]))
            langs = _resolve_benchmark_languages(ctx)
            print(f"[INFO] Resolved benchmark languages, Eval is only performed in those languages: {langs}")
            if not langs:
                raise SystemExit("Benchmark stage could not resolve trained language list.")
            ctx.benchmark_root.mkdir(parents=True, exist_ok=True)
            select_cmd = [
                "python",
                "-m",
                "multilingual_megatron.pipeline.stages.benchmarking.tools.select_multilingual_eval_tasks",
                "--tasks-file",
                str(tasks_file),
                "--languages",
                ",".join(langs),
                "--output-file",
                str(selected_tasks_file),
                "--summary-json",
                str(selected_tasks_summary),
            ]
            if bool(ctx.benchmark_cfg.get("include_belebele", True)):
                select_cmd.append("--include-belebele")
            if bool(ctx.benchmark_cfg.get("include_sib200", True)):
                select_cmd.append("--include-sib200")
            if bool(ctx.benchmark_cfg.get("include_taxi1500", True)):
                select_cmd.append("--include-taxi1500")
            if bool(ctx.benchmark_cfg.get("include_flores", True)):
                select_cmd.append("--include-flores")
                select_cmd.extend(["--flores-mode", str(ctx.benchmark_cfg.get("flores_mode", "both"))])
            _run_stage_cmd(
                cmd=select_cmd,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="benchmark",
                suffix="task_select",
                executor=ctx.executor,
            )
            if ctx.args.dry_run:
                selected_tasks = _select_tasks_for_languages(
                    tasks_file=tasks_file,
                    languages=langs,
                    include_belebele=bool(ctx.benchmark_cfg.get("include_belebele", True)),
                    include_sib200=bool(ctx.benchmark_cfg.get("include_sib200", True)),
                    include_taxi1500=bool(ctx.benchmark_cfg.get("include_taxi1500", True)),
                    include_flores=bool(ctx.benchmark_cfg.get("include_flores", True)),
                    flores_mode=str(ctx.benchmark_cfg.get("flores_mode", "both")),
                )
            else:
                selected_tasks = _read_selected_tasks_file(selected_tasks_file)
            if not ctx.args.dry_run and (not selected_tasks) and bool(ctx.benchmark_cfg.get("fail_if_no_tasks", True)):
                raise SystemExit("Benchmark stage selected zero tasks.")
            tasks_csv = ",".join(selected_tasks) if selected_tasks else "belebele_eng_Latn"
            lm_eval_bin = str(ctx.benchmark_cfg.get("lm_eval_bin", "lm_eval"))
            if not ctx.args.dry_run and (not bool(ctx.benchmark_cfg.get("allow_missing_lm_eval", False))):
                if shutil.which(lm_eval_bin) is None:
                    raise SystemExit(f"Benchmark stage requires '{lm_eval_bin}' in PATH.")
            eval_script = _resolve_eval_script(ctx)
            if not eval_script.exists():
                raise SystemExit(f"Benchmark eval script not found: {eval_script}")
            tokenizer = resolved_eval_tokenizer
            batch_size = str(ctx.benchmark_cfg.get("batch_size", "auto"))
            lang_mode = str(ctx.benchmark_cfg.get("lang_mode", "")).strip()
            extra_args = str(ctx.benchmark_cfg.get("extra_args", "")).strip()
            if any(task.startswith(("sib200_", "taxi1500_")) for task in selected_tasks):
                _ensure_external_task_configs(ctx.repo_root)
            selected_flores_tasks = [task for task in selected_tasks if task.startswith("flores_en_perplexity_")]
            include_dirs = _default_lm_eval_include_dirs(ctx.repo_root, selected_tasks)
            for include_dir in include_dirs:
                if include_dir.parent.name == "flores_en_perplexity" and selected_flores_tasks:
                    _validate_flores_task_data_paths(ctx.repo_root, include_dir, selected_flores_tasks)
                elif include_dir.parent.name == "sib200":
                    _validate_task_yaml_presence(include_dir, selected_tasks, "sib200_", "SIB200")
                elif include_dir.parent.name == "taxi1500":
                    _validate_task_yaml_presence(include_dir, selected_tasks, "taxi1500_", "Taxi1500")

            parsed_extra_args = shlex.split(extra_args) if extra_args else []
            if "--include-path" in parsed_extra_args:
                idx = parsed_extra_args.index("--include-path")
                parsed_extra_args[idx] = "--include_path"
            if include_dirs and "--include_path" not in parsed_extra_args:
                parsed_extra_args.extend(["--include_path", ":".join(str(path) for path in include_dirs)])
            extra_args = shlex.join(parsed_extra_args) if parsed_extra_args else ""
            default_eval_project = str(ctx.wandb_cfg.get("project", "proof_eval")) if bool(ctx.wandb_cfg.get("enabled", False)) else "proof_eval"
            wandb_project = str(ctx.benchmark_cfg.get("wandb_project", default_eval_project))
            wandb_prefix_root = str(
                ctx.benchmark_cfg.get(
                    "wandb_prefix",
                    str(ctx.wandb_cfg.get("eval_prefix", "eval/benchmark")) if bool(ctx.wandb_cfg.get("enabled", False)) else "eval/benchmark",
                )
            )
            wandb_mode = str(
                ctx.benchmark_cfg.get(
                    "wandb_mode",
                    str(ctx.wandb_cfg.get("mode", "disabled")) if bool(ctx.wandb_cfg.get("enabled", False)) else "disabled",
                )
            )
            benchmark_rows: list[dict[str, object]] = []
            run_name_by_id: dict[str, str] = {
                str(run["id"]): _build_run_label(
                    base_run_label=ctx.base_run_label,
                    run=run,
                    default_num_experts=int(ctx.train_cfg.get("num_experts", 0)),
                )
                for run in runs
            }
            for run in runs:
                run_id = str(run["id"])
                run_name_label = run_name_by_id.get(run_id, run_id)
                checkpoint_dir = ctx.train_root / run_id
                if not checkpoint_dir.exists():
                    raise SystemExit(f"Benchmark checkpoint directory missing: {checkpoint_dir}")
                evaluated_checkpoint = checkpoint_dir
                out_dir = ctx.benchmark_root / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "bash",
                    str(eval_script),
                    "--checkpoint",
                    str(evaluated_checkpoint),
                    "--tokenizer",
                    tokenizer,
                    "--tasks",
                    tasks_csv,
                    "--batch-size",
                    batch_size,
                    "--output-dir",
                    str(out_dir),
                    "--wandb-project",
                    wandb_project,
                    "--wandb-prefix",
                    f"{wandb_prefix_root}/{run_name_label}",
                    "--wandb-mode",
                    wandb_mode,
                ]
                if lang_mode:
                    cmd.extend(["--lang-mode", lang_mode])
                if extra_args:
                    cmd.extend(["--extra-args", extra_args])
                _run_stage_cmd(
                    cmd=cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="benchmark",
                    suffix=run_name_label,
                    executor=ctx.executor,
                )
                if not ctx.args.dry_run:
                    files, task_seen = _collect_tasks_from_output_dir(out_dir)
                    benchmark_rows.append(
                        {
                            "run_id": run_id,
                            "evaluated_checkpoint": str(evaluated_checkpoint.resolve()),
                            "evaluated_tokenizer": tokenizer,
                            "output_dir": str(out_dir.resolve()),
                            "num_output_files": len(files),
                            "num_tasks_expected": len(selected_tasks),
                            "num_tasks_seen": len(task_seen),
                            "missing_tasks": sorted(set(selected_tasks) - task_seen),
                        }
                    )
            if not ctx.args.dry_run:
                _dump_json(
                    benchmark_summary_path,
                    {
                        "tasks_file": str(tasks_file.resolve()),
                        "selected_tasks_file": str(selected_tasks_file.resolve()),
                        "selected_tasks_summary": str(selected_tasks_summary.resolve()),
                        "resolved_eval_tokenizer": tokenizer,
                        "lm_eval_limit": str(ctx.env.get("LM_EVAL_LIMIT", "")).strip(),
                        "num_selected_tasks": len(selected_tasks),
                        "runs": benchmark_rows,
                    },
                )
                _save_stage_state(ctx.state_path, "benchmark", signature, [str(path) for path in expected])
                check_benchmark(ctx.benchmark_root, benchmark_summary_path, dry_run=ctx.args.dry_run)


class BenchmarkStage:
    name = "benchmark"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = getattr(ctx, "benchmark_cfg", {})
        enabled = bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else False
        benchmark_root = getattr(ctx, "benchmark_root", None)
        artifacts = ()
        if enabled and benchmark_root is not None:
            artifacts = (
                ArtifactSpec(benchmark_root / "benchmark_summary.json", required=True, description="Benchmark summary"),
            )
        return StagePlan(stage=self.name, enabled=enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped", metadata={"reason": "disabled"})
        stage_benchmark(ctx)
        benchmark_root = getattr(ctx, "benchmark_root", None)
        artifacts = (benchmark_root / "benchmark_summary.json",) if benchmark_root is not None else ()
        return StageResult(stage=self.name, artifacts=artifacts)

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx, result


stage = BenchmarkStage()

__all__ = ["BenchmarkStage", "stage", "stage_benchmark"]
