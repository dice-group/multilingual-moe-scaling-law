#!/usr/bin/env python3
"""Select lm-eval tasks for trained languages from a task catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _task_lang_from_belebele(task: str) -> str | None:
    if not task.startswith("belebele_"):
        return None
    return task.replace("belebele_", "", 1)


def _task_lang_from_sib200(task: str) -> str | None:
    if not task.startswith("sib200_"):
        return None
    return task.replace("sib200_", "", 1)


def _task_lang_from_taxi1500(task: str) -> str | None:
    if not task.startswith("taxi1500_"):
        return None
    return task.replace("taxi1500_", "", 1)


def _task_langs_from_flores(task: str) -> tuple[str, str] | None:
    prefix = "flores_en_perplexity_"
    if not task.startswith(prefix):
        return None
    rest = task[len(prefix) :]
    if "-" not in rest:
        return None
    src, tgt = rest.split("-", 1)
    return src, tgt


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter lm-eval tasks for trained languages.")
    parser.add_argument("--tasks-file", required=True, help="Task catalog file.")
    parser.add_argument("--languages", required=True, help="Comma-separated trained language ids.")
    parser.add_argument("--output-file", required=True, help="Output selected task file.")
    parser.add_argument("--summary-json", required=True, help="Output summary JSON.")
    parser.add_argument("--include-belebele", action="store_true")
    parser.add_argument("--include-sib200", action="store_true")
    parser.add_argument("--include-taxi1500", action="store_true")
    parser.add_argument("--include-flores", action="store_true")
    parser.add_argument(
        "--flores-mode",
        choices=["both", "into_en", "from_en"],
        default="both",
        help="FLORES direction filter relative to eng_Latn.",
    )
    args = parser.parse_args()

    task_lines = _read_lines(Path(args.tasks_file))
    languages = sorted({item.strip() for item in args.languages.split(",") if item.strip()})
    lang_set = set(languages)

    selected: list[str] = []
    for task in task_lines:
        if args.include_belebele:
            lang = _task_lang_from_belebele(task)
            if lang is not None and lang in lang_set:
                selected.append(task)
                continue

        if args.include_sib200:
            lang = _task_lang_from_sib200(task)
            if lang is not None and lang in lang_set:
                selected.append(task)
                continue

        if args.include_taxi1500:
            lang = _task_lang_from_taxi1500(task)
            if lang is not None and lang in lang_set:
                selected.append(task)
                continue

        if args.include_flores:
            pair = _task_langs_from_flores(task)
            if pair is not None:
                src, tgt = pair
                include = False
                if args.flores_mode == "both":
                    include = src in lang_set or tgt in lang_set
                elif args.flores_mode == "into_en":
                    include = src in lang_set and tgt == "eng_Latn"
                elif args.flores_mode == "from_en":
                    include = src == "eng_Latn" and tgt in lang_set
                if include:
                    selected.append(task)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    summary = {
        "languages": languages,
        "num_languages": len(languages),
        "tasks_file": str(Path(args.tasks_file).resolve()),
        "output_file": str(output_path.resolve()),
        "include_belebele": bool(args.include_belebele),
        "include_sib200": bool(args.include_sib200),
        "include_taxi1500": bool(args.include_taxi1500),
        "include_flores": bool(args.include_flores),
        "flores_mode": args.flores_mode,
        "num_tasks": len(selected),
        "num_belebele": len([t for t in selected if t.startswith("belebele_")]),
        "num_sib200": len([t for t in selected if t.startswith("sib200_")]),
        "num_taxi1500": len([t for t in selected if t.startswith("taxi1500_")]),
        "num_flores": len([t for t in selected if t.startswith("flores_en_perplexity_")]),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] selected_tasks={len(selected)}")
    print(f"[OK] output_file={output_path}")
    print(f"[OK] summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
