#!/usr/bin/env python3
"""Apply capped upsampling (with replacement) to sampled FineWeb JSONL shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def _stable_lang_seed(base_seed: int, language: str) -> int:
    digest = hashlib.sha256(language.encode("utf-8")).hexdigest()[:8]
    return base_seed + int(digest, 16)


def _read_plan_rows(plan_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with plan_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_jsonl_lines(dir_path: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(dir_path.glob("*.jsonl")):
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    return lines


def _rewrite_single_jsonl(dir_path: Path, lines: list[str]) -> Path:
    for path in sorted(dir_path.glob("*.jsonl")):
        path.unlink()
    out_path = dir_path / "part_000.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsample sampled FineWeb shards with cap.")
    parser.add_argument("--plan-csv", required=True, help="Sampling plan CSV with target documents.")
    parser.add_argument("--samples-root", required=True, help="Root folder containing <lang>/*.jsonl.")
    parser.add_argument("--cap-factor", type=float, required=True, help="Maximum multiplicative upsample factor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--report-json", required=True, help="Output report JSON path.")
    parser.add_argument("--report-tsv", required=True, help="Output report TSV path.")
    args = parser.parse_args()

    if args.cap_factor <= 1.0:
        raise SystemExit("--cap-factor must be > 1.0 for capped upsampling.")

    plan_csv = Path(args.plan_csv)
    samples_root = Path(args.samples_root)
    report_json = Path(args.report_json)
    report_tsv = Path(args.report_tsv)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_plan_rows(plan_csv)
    report_rows: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0

    for row in rows:
        lang = str(row.get("subset", "")).strip()
        if not lang:
            continue
        target = int(float(row.get("target_documents", row.get("documents", 0)) or 0))
        planned_available = int(float(row.get("available_documents", 0) or 0))

        lang_dir = samples_root / lang
        if not lang_dir.exists():
            report_rows.append(
                {
                    "language": lang,
                    "planned_target": target,
                    "planned_available": planned_available,
                    "before_count": 0,
                    "cap_max_count": 0,
                    "after_count": 0,
                    "added_by_upsample": 0,
                    "status": "missing_dir",
                }
            )
            continue

        lines = _read_jsonl_lines(lang_dir)
        before_count = len(lines)
        total_before += before_count

        if before_count == 0:
            report_rows.append(
                {
                    "language": lang,
                    "planned_target": target,
                    "planned_available": planned_available,
                    "before_count": 0,
                    "cap_max_count": 0,
                    "after_count": 0,
                    "added_by_upsample": 0,
                    "status": "no_data",
                }
            )
            continue

        cap_max_count = int(math.floor(before_count * args.cap_factor))
        desired_count = min(target, cap_max_count)
        if desired_count <= before_count:
            after_count = before_count
            added = 0
            status = "kept_original"
        else:
            rng = random.Random(_stable_lang_seed(args.seed, lang))
            while len(lines) < desired_count:
                lines.append(lines[rng.randrange(before_count)])
            _rewrite_single_jsonl(lang_dir, lines)
            after_count = len(lines)
            added = after_count - before_count
            status = "upsampled_capped"

        total_after += after_count
        report_rows.append(
            {
                "language": lang,
                "planned_target": target,
                "planned_available": planned_available,
                "before_count": before_count,
                "cap_max_count": cap_max_count,
                "after_count": after_count,
                "added_by_upsample": added,
                "status": status,
            }
        )

    report_payload = {
        "plan_csv": str(plan_csv.resolve()),
        "samples_root": str(samples_root.resolve()),
        "cap_factor": float(args.cap_factor),
        "seed": int(args.seed),
        "num_languages": len(report_rows),
        "total_before": int(total_before),
        "total_after": int(total_after),
        "total_added_by_upsample": int(total_after - total_before),
        "rows": report_rows,
    }
    report_json.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    with report_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "language\tplanned_target\tplanned_available\tbefore_count\tcap_max_count\tafter_count\tadded_by_upsample\tstatus\n"
        )
        for row in report_rows:
            handle.write(
                f"{row['language']}\t{row['planned_target']}\t{row['planned_available']}\t"
                f"{row['before_count']}\t{row['cap_max_count']}\t{row['after_count']}\t"
                f"{row['added_by_upsample']}\t{row['status']}\n"
            )

    print(f"[OK] cap_factor={args.cap_factor}")
    print(f"[OK] total_before={total_before} total_after={total_after} added={total_after - total_before}")
    print(f"[OK] report_json={report_json}")
    print(f"[OK] report_tsv={report_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
