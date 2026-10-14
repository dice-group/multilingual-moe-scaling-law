#!/usr/bin/env python3
"""Validate generated multilingual Switch-MoE scaling-law artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import DEFAULT_STUDY_DIR, validate_artifacts


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", default=str(DEFAULT_STUDY_DIR))
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    language_sets_path = study_dir / "language_sets.json"
    run_configs_path = study_dir / "run_configs.jsonl"
    language_sets = json.loads(language_sets_path.read_text(encoding="utf-8"))
    runs = _load_jsonl(run_configs_path)
    errors = validate_artifacts(language_sets, runs)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALID runs={len(runs)} checkpoint_observations={sum(len(run['checkpoints']) for run in runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
