#!/usr/bin/env python3
"""Verify FLORES metrics present in the baseline cache and compare word perplexity to a reference CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def _model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return _sanitize_slug(value)


def _load_cache_rows(path: Path) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            continue
        task = str(payload.get("task", "")).strip()
        metric = str(payload.get("metric", "")).strip()
        value = payload.get("value")
        if task.startswith("flores_en_perplexity_") and metric and isinstance(value, (int, float)):
            rows[(task, metric)] = float(value)
    return rows


def _load_reference_rows(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = str(row.get("task", "")).strip()
            metric = str(row.get("metric", "")).strip()
            raw_value = row.get("value")
            if not task.startswith("flores_en_perplexity_") or metric != "word_perplexity,none":
                continue
            try:
                value = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None:
                rows[task] = value
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify FLORES baseline cache coverage and word perplexity stability.")
    parser.add_argument("--baseline-cache-root", default="outputs/baselines")
    parser.add_argument("--model-name-or-path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--reference-csv",
        default="configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    slug = _model_slug(args.model_name_or_path)
    metrics_path = Path(args.baseline_cache_root) / slug / "metrics_long.jsonl"
    cache_rows = _load_cache_rows(metrics_path)
    reference_rows = _load_reference_rows(Path(args.reference_csv))

    task_names = sorted({task for task, _ in cache_rows} | set(reference_rows))
    metrics_of_interest = ("word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none")
    coverage = {metric: 0 for metric in metrics_of_interest}
    missing_by_metric = {metric: [] for metric in metrics_of_interest}
    for task in task_names:
        for metric in metrics_of_interest:
            if (task, metric) in cache_rows:
                coverage[metric] += 1
            else:
                missing_by_metric[metric].append(task)

    compared = 0
    exact_matches = 0
    mismatches: list[tuple[str, float, float, float]] = []
    for task, old_value in sorted(reference_rows.items()):
        new_value = cache_rows.get((task, "word_perplexity,none"))
        if new_value is None:
            continue
        compared += 1
        diff = abs(new_value - old_value)
        if diff <= args.tolerance:
            exact_matches += 1
        else:
            mismatches.append((task, old_value, new_value, diff))

    print(f"[INFO] cache_file={metrics_path}")
    print(f"[INFO] flores_tasks_reference={len(reference_rows)}")
    print(f"[INFO] flores_tasks_cache={len({task for task, _ in cache_rows})}")
    for metric in metrics_of_interest:
        print(f"[INFO] cache_coverage[{metric}]={coverage[metric]}")

    print(f"[INFO] word_ppl_compared={compared}")
    print(f"[INFO] word_ppl_exact_matches={exact_matches}")
    if compared:
        max_diff = max((diff for _, _, _, diff in mismatches), default=0.0)
        print(f"[INFO] word_ppl_max_abs_diff={max_diff}")

    for metric in ("byte_perplexity,none", "bits_per_byte,none"):
        if missing_by_metric[metric]:
            print(f"[WARN] missing_{metric}={len(missing_by_metric[metric])}")
            print(f"[WARN] sample_missing_{metric}={missing_by_metric[metric][:5]}")

    if mismatches:
        print(f"[WARN] word_ppl_mismatches={len(mismatches)}")
        for task, old_value, new_value, diff in mismatches[:10]:
            print(
                "[WARN] mismatch "
                f"task={task} old={old_value:.12f} new={new_value:.12f} diff={diff:.12g}"
            )

    complete_metrics = all(not missing_by_metric[metric] for metric in metrics_of_interest)
    stable_word_ppl = compared == len(reference_rows) and not mismatches
    if complete_metrics and stable_word_ppl:
        print("[OK] FLORES cache contains word/byte/bit metrics and word perplexity matches the tracked baseline.")
        return 0

    if not complete_metrics:
        print("[FAIL] FLORES byte/bit cache coverage is incomplete.")
        return 1
    if not stable_word_ppl:
        print("[FAIL] FLORES word perplexity differs from the tracked baseline.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
