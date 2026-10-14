#!/usr/bin/env python3
"""Run lm-eval against a Bridge checkpoint exported to Hugging Face format."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.common import lm_eval_harness_root
from multilingual_megatron.repo import find_repo_root


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            tasks.append(value)
    return tasks


def _pick_metrics(task: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(metrics, dict):
        return []
    if task.startswith("belebele_"):
        preferred = ["acc_norm,none", "acc,none"]
    elif task.startswith("flores_en_perplexity_"):
        preferred = ["word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none", "perplexity,none"]
    else:
        preferred = ["acc_norm,none", "acc,none", "word_perplexity,none", "perplexity,none"]
    rows: list[dict[str, Any]] = []
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({"metric": key, "value": float(value)})
    if rows:
        return rows[:1] if task.startswith("belebele_") else rows
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [{"metric": str(key), "value": float(value)}]
    return []


def _load_lm_eval_results(path: Path) -> tuple[dict[str, Any], Path]:
    if path.is_file():
        payload_path = path
    elif path.is_dir():
        candidates = sorted(path.rglob("results_*.json"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            candidates = sorted(path.rglob("*.json"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No lm-eval JSON results found under {path}")
        payload_path = candidates[-1]
    else:
        candidates = sorted(path.parent.glob(f"{path.stem}*.json"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"lm-eval output path not found: {path}")
        payload_path = candidates[-1]
    payload = _read_json(payload_path)
    return payload, payload_path


def _local_harness_parent(repo_root: Path) -> Path | None:
    return lm_eval_harness_root(repo_root)


def _ensure_lm_eval_metadata_shim(output_dir: Path) -> Path:
    shim_root = output_dir / ".lm_eval_metadata"
    dist_info = shim_root / "lm_eval-0.4.10.dev0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lm_eval\nVersion: 0.4.10.dev0\n",
        encoding="utf-8",
    )
    return shim_root


def _lm_eval_command(args: argparse.Namespace, env: dict[str, str]) -> list[str]:
    if shutil.which(args.lm_eval_bin):
        return [args.lm_eval_bin]
    repo_root = Path(args.repo_root).resolve()
    harness_parent = _local_harness_parent(repo_root)
    if harness_parent is None:
        raise SystemExit(f"lm-eval binary not found and local harness is unavailable: {args.lm_eval_bin}")
    metadata_shim = _ensure_lm_eval_metadata_shim(Path(args.output_dir))
    env["PYTHONPATH"] = f"{harness_parent}:{metadata_shim}:{env.get('PYTHONPATH', '')}"
    return [sys.executable, "-m", "lm_eval"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Bridge HF export with lm-eval.")
    parser.add_argument("--checkpoint-export-manifest", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--lm-eval-bin", default="lm_eval")
    parser.add_argument("--limit", default="")
    parser.add_argument("--include-path", action="append", default=[])
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", default=str(find_repo_root(Path(__file__))))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    export_manifest = _read_json(Path(args.checkpoint_export_manifest))
    hf_export_dir = Path(str(export_manifest.get("hf_export_dir", ""))).resolve()
    tokenizer = str(args.tokenizer).strip() or str(hf_export_dir)
    run_id = str(export_manifest.get("run_id", hf_export_dir.name)).strip() or hf_export_dir.name
    if not hf_export_dir.exists() and not args.dry_run:
        raise SystemExit(f"HF export directory does not exist: {hf_export_dir}")

    tasks_file = Path(args.tasks_file).resolve()
    tasks = _read_tasks(tasks_file)
    if not tasks:
        raise SystemExit(f"No tasks found in {tasks_file}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "lm_eval_output.json"
    metrics_path = output_dir / "metrics.json"

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    cmd = _lm_eval_command(args, env)
    cmd.extend(
        [
            "--model",
            "hf",
            "--model_args",
            f"pretrained={hf_export_dir},tokenizer={tokenizer}",
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            str(args.batch_size),
            "--output_path",
            str(raw_output),
            "--seed",
            str(args.seed),
        ]
    )
    if str(args.limit).strip():
        cmd.extend(["--limit", str(args.limit).strip()])
    include_paths = [str(Path(path).resolve()) for path in args.include_path if str(path).strip()]
    if include_paths:
        cmd.extend(["--include_path", ":".join(include_paths)])
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))

    print(f"[CMD] {' '.join(cmd)}", flush=True)
    if not args.dry_run:
        subprocess.run(cmd, env=env, check=True)
        payload, payload_path = _load_lm_eval_results(raw_output)
        results = payload.get("results", {})
    else:
        payload_path = raw_output
        results = {}

    metric_rows: list[dict[str, Any]] = []
    if isinstance(results, dict):
        for task, task_metrics in results.items():
            for picked in _pick_metrics(str(task), task_metrics if isinstance(task_metrics, dict) else {}):
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "task": str(task),
                        "metric": picked["metric"],
                        "value": picked["value"],
                    }
                )
    seen_tasks = sorted({row["task"] for row in metric_rows})
    missing_tasks = sorted(set(tasks) - set(seen_tasks))
    metrics = {
        "schema_version": 1,
        "backend": "bridge_hf_lm_eval",
        "run_id": run_id,
        "hf_export_dir": str(hf_export_dir),
        "tokenizer": tokenizer,
        "checkpoint_export_manifest": str(Path(args.checkpoint_export_manifest).resolve()),
        "tasks_file": str(tasks_file),
        "tasks": tasks,
        "num_tasks_expected": len(tasks),
        "num_tasks_seen": len(seen_tasks),
        "missing_tasks": missing_tasks,
        "raw_output_json": str(payload_path.resolve()),
        "metrics": metric_rows,
        "limit": str(args.limit).strip(),
    }
    if not args.dry_run:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] bridge_hf_lm_eval metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
