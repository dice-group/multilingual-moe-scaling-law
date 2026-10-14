#!/usr/bin/env python3
"""Build and cache base-model lm-eval results for reuse in comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def _model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return _sanitize_slug(value)


def _read_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        tasks.append(raw)
    return tasks


def _pick_metrics(task: str, metrics: dict[str, Any]) -> list[tuple[str, float]]:
    if not isinstance(metrics, dict):
        return []
    preferred: list[str]
    if task.startswith("belebele_"):
        preferred = ["acc_norm,none", "acc,none"]
    elif task.startswith("flores_en_perplexity_"):
        preferred = ["word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none", "perplexity,none"]
    else:
        preferred = ["word_perplexity,none", "byte_perplexity,none", "perplexity,none"]
    selected: list[tuple[str, float]] = []
    for key in preferred:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            selected.append((key, float(val)))
    if selected:
        if task.startswith("belebele_"):
            return selected[:1]
        return selected
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            return [(str(key), float(val))]
    return []


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_from_manifest(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _load_precomputed_sources(manifest_paths: list[Path]) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for path in manifest_paths:
        if not path.exists():
            raise FileNotFoundError(f"Precomputed manifest not found: {path}")
        payload = _read_json(path)
        model_slug = str(payload.get("model_slug", "")).strip()
        model = str(payload.get("model", "")).strip()
        metrics_long_csv = str(payload.get("metrics_long_csv", "")).strip()
        metrics_summary_csv = str(payload.get("metrics_summary_csv", "")).strip()
        raw_output_json = str(payload.get("raw_output_json", "")).strip()
        if not model_slug:
            raise ValueError(f"Precomputed manifest missing model_slug: {path}")
        chosen_csv = metrics_summary_csv or metrics_long_csv
        if not chosen_csv:
            raise ValueError(f"Precomputed manifest missing both metrics_summary_csv and metrics_long_csv: {path}")
        resolved_csv = _resolve_from_manifest(path, chosen_csv)
        sources[model_slug] = {
            "manifest_path": str(path.resolve()),
            "model": model,
            "model_slug": model_slug,
            "metrics_csv": str(resolved_csv),
            "metrics_summary_csv": metrics_summary_csv,
            "metrics_long_csv": metrics_long_csv,
            "raw_output_json": raw_output_json,
        }
    return sources


def _collect_metrics_rows_from_csv(
    *,
    csv_path: Path,
    model: str,
    model_slug: str,
    requested_tasks: list[str],
) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Precomputed CSV not found: {csv_path}")
    requested = {task.strip() for task in requested_tasks if task.strip()}
    group_prefixes = tuple(f"{task}_" for task in requested if "_" not in task)

    def include_task(task_name: str) -> bool:
        if task_name in requested:
            return True
        for prefix in group_prefixes:
            if task_name.startswith(prefix):
                return True
        return False

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = str(row.get("task", "")).strip()
            metric = str(row.get("metric", "")).strip()
            raw_value = row.get("value")
            if not task or not metric:
                continue
            if not include_task(task):
                continue
            try:
                value = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is None:
                continue
            payload = {
                "run_id": f"baseline::{model_slug}",
                "task": task,
                "metric": metric,
                "value": value,
                "wandb_project": "",
                "wandb_prefix": "",
                "model": model,
                "model_slug": model_slug,
            }
            rows.append(payload)
    return rows


def _load_existing_metrics_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        task = str(payload.get("task", "")).strip()
        metric = str(payload.get("metric", "")).strip()
        value = payload.get("value")
        if not task or not metric or not isinstance(value, (int, float)):
            continue
        rows[(task, metric)] = payload
    return rows


def _merge_metrics_rows(
    existing_rows: dict[tuple[str, str], dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = dict(existing_rows)
    for row in new_rows:
        task = str(row.get("task", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = row.get("value")
        if not task or not metric or not isinstance(value, (int, float)):
            continue
        merged[(task, metric)] = row
    return [merged[key] for key in sorted(merged.keys())]


def _write_metrics_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def _load_lm_eval_results_payload(output_path: Path) -> tuple[dict[str, Any], Path]:
    if output_path.is_file():
        payload_path = output_path
    elif output_path.is_dir():
        candidates = sorted(output_path.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No results_*.json files found under {output_path}")
        payload_path = candidates[-1]
    else:
        stem = output_path.stem
        candidates = sorted(output_path.parent.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"lm-eval output path not found: {output_path}")
        payload_path = candidates[-1]
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"lm-eval payload at {payload_path} is not a JSON object")
    return payload, payload_path


def _cache_entry_matches_request(
    *,
    manifest_payload: dict[str, Any] | None,
    model_slug: str,
    metrics_jsonl: Path,
    tasks_file: Path,
    seed: int,
    batch_size: str,
) -> bool:
    if not manifest_payload:
        return False
    rows = manifest_payload.get("rows", [])
    if not isinstance(rows, list):
        return False
    expected_metrics = str(metrics_jsonl.resolve())
    expected_tasks_file = str(tasks_file.resolve())
    expected_batch_size = str(batch_size)
    expected_seed = int(seed)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("model_slug", "")).strip() != model_slug:
            continue
        if str(row.get("metrics_long_jsonl", "")).strip() != expected_metrics:
            continue
        row_tasks_file = str(row.get("tasks_file", "")).strip()
        row_batch_size = str(row.get("batch_size", "")).strip()
        row_seed = row.get("seed")
        if row_tasks_file or row_batch_size or row_seed is not None:
            if row_tasks_file != expected_tasks_file:
                continue
            if row_batch_size != expected_batch_size:
                continue
            if int(row_seed) != expected_seed:
                continue
            return True
        if (
            str(manifest_payload.get("tasks_file", "")) == expected_tasks_file
            and str(manifest_payload.get("batch_size", "")) == expected_batch_size
            and int(manifest_payload.get("seed", -1)) == expected_seed
        ):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache base-model lm-eval results.")
    parser.add_argument("--models", required=True, help="Comma-separated HF model IDs.")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--lm-eval-bin", default="lm_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-args", default="")
    parser.add_argument(
        "--task-prefix",
        action="append",
        default=[],
        help="Restrict the tasks file to entries whose names start with one of these prefixes. Can be provided multiple times.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--precomputed-manifest",
        action="append",
        default=[],
        help="Path to a precomputed baseline manifest (from run_llama31_8b_full_baseline_eval.py). Can be provided multiple times.",
    )
    args = parser.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if not models:
        raise SystemExit("No models provided.")
    tasks_file = Path(args.tasks_file)
    tasks = _read_tasks(tasks_file)
    task_prefixes = [str(prefix).strip() for prefix in args.task_prefix if str(prefix).strip()]
    if task_prefixes:
        tasks = [task for task in tasks if any(task.startswith(prefix) for prefix in task_prefixes)]
    if not tasks:
        raise SystemExit(f"No tasks found in {tasks_file}")
    tasks_csv = ",".join(tasks)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_root / "manifest.json"
    output_manifest_payload: dict[str, Any] | None = None
    if output_manifest_path.exists():
        try:
            output_manifest_payload = _read_json(output_manifest_path)
        except (OSError, json.JSONDecodeError, ValueError):
            output_manifest_payload = None
    precomputed_manifests = [Path(path).resolve() for path in args.precomputed_manifest]
    precomputed_by_slug = _load_precomputed_sources(precomputed_manifests) if precomputed_manifests else {}
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    manifest_rows: list[dict[str, Any]] = []
    for model in models:
        slug = _model_slug(model)
        model_dir = output_root / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        raw_json = model_dir / "lm_eval_output.json"
        metrics_jsonl = model_dir / "metrics_long.jsonl"

        if (
            metrics_jsonl.exists()
            and not args.force
            and not args.dry_run
            and _cache_entry_matches_request(
                manifest_payload=output_manifest_payload,
                model_slug=slug,
                metrics_jsonl=metrics_jsonl,
                tasks_file=tasks_file,
                seed=args.seed,
                batch_size=str(args.batch_size),
            )
        ):
            print(f"[SKIP] cached baseline exists for {model} at {metrics_jsonl}")
            manifest_rows.append(
                {
                    "model": model,
                    "model_slug": slug,
                    "raw_output_json": str(raw_json.resolve()),
                    "metrics_long_jsonl": str(metrics_jsonl.resolve()),
                    "num_tasks": None,
                }
            )
            continue

        precomputed = precomputed_by_slug.get(slug)
        existing_rows = _load_existing_metrics_jsonl(metrics_jsonl) if not args.dry_run else {}
        if precomputed and not args.force and not args.dry_run:
            csv_path = Path(precomputed["metrics_csv"]).resolve()
            new_rows = _collect_metrics_rows_from_csv(
                csv_path=csv_path,
                model=model,
                model_slug=slug,
                requested_tasks=tasks,
            )
            merged_rows = _merge_metrics_rows(existing_rows, new_rows)
            _write_metrics_jsonl(metrics_jsonl, merged_rows)
            row_count = len(new_rows)
            print(
                f"[SKIP] using precomputed baseline CSV for {model}: {csv_path} -> {metrics_jsonl} "
                f"({row_count} requested rows, {len(merged_rows)} cached rows total)"
            )
            raw_output_value = str(precomputed.get("raw_output_json", "")).strip() or str(raw_json.resolve())
            manifest_rows.append(
                {
                    "model": model,
                    "model_slug": slug,
                    "raw_output_json": raw_output_value,
                    "metrics_long_jsonl": str(metrics_jsonl.resolve()),
                    "num_tasks": row_count,
                    "source": "precomputed_csv",
                    "source_manifest": str(precomputed.get("manifest_path", "")),
                    "tasks_file": str(tasks_file.resolve()),
                    "batch_size": str(args.batch_size),
                    "seed": int(args.seed),
                }
            )
            continue

        cmd = [
            args.lm_eval_bin,
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},tokenizer={model}",
            "--tasks",
            tasks_csv,
            "--batch_size",
            str(args.batch_size),
            "--output_path",
            str(raw_json),
            "--seed",
            str(args.seed),
        ]
        if args.extra_args.strip():
            cmd.extend(shlex.split(args.extra_args))
        _run_cmd(cmd, env=env, dry_run=args.dry_run)

        task_count = None
        if not args.dry_run:
            payload, payload_path = _load_lm_eval_results_payload(raw_json)
            results = payload.get("results", {})
            rows: list[dict[str, Any]] = []
            if isinstance(results, dict):
                for task, metrics in results.items():
                    picked_metrics = _pick_metrics(str(task), metrics if isinstance(metrics, dict) else {})
                    if not picked_metrics:
                        continue
                    for metric_name, metric_value in picked_metrics:
                        rows.append(
                            {
                                "run_id": f"baseline::{slug}",
                                "task": str(task),
                                "metric": metric_name,
                                "value": metric_value,
                                "wandb_project": "",
                                "wandb_prefix": "",
                                "model": model,
                                "model_slug": slug,
                            }
                        )
            merged_rows = _merge_metrics_rows(existing_rows, rows)
            _write_metrics_jsonl(metrics_jsonl, merged_rows)
            task_count = len(rows)
            raw_output_value = str(payload_path.resolve())
        else:
            raw_output_value = str(raw_json.resolve())

        manifest_rows.append(
            {
                "model": model,
                "model_slug": slug,
                "raw_output_json": raw_output_value,
                "metrics_long_jsonl": str(metrics_jsonl.resolve()),
                "num_tasks": task_count,
                "source": "lm_eval",
                "tasks_file": str(tasks_file.resolve()),
                "batch_size": str(args.batch_size),
                "seed": int(args.seed),
            }
        )

    existing_manifest_rows = output_manifest_payload.get("rows", []) if isinstance(output_manifest_payload, dict) else []
    merged_manifest_rows: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for row in existing_manifest_rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("model_slug", "")).strip(),
            str(row.get("metrics_long_jsonl", "")).strip(),
            str(row.get("tasks_file", "")).strip(),
            str(row.get("batch_size", "")).strip(),
            int(row.get("seed", -1)),
        )
        merged_manifest_rows[key] = row
    for row in manifest_rows:
        key = (
            str(row.get("model_slug", "")).strip(),
            str(row.get("metrics_long_jsonl", "")).strip(),
            str(row.get("tasks_file", "")).strip(),
            str(row.get("batch_size", "")).strip(),
            int(row.get("seed", -1)),
        )
        merged_manifest_rows[key] = row

    manifest = {
        "tasks_file": str(tasks_file.resolve()),
        "num_tasks": len(tasks),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "rows": list(merged_manifest_rows.values()),
    }
    manifest_path = output_root / "manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] baseline manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
