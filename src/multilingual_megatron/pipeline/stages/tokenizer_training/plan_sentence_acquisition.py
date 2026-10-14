#!/usr/bin/env python3
"""Plan the second, quota-sized FineWeb acquisition for tokenizer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_sentence_artifact import (
    DEFAULT_ACQUISITION_SAFETY_MARGIN,
    DEFAULT_MAX_SENTENCE_CHARS,
    PRODUCTION_ALPHA,
    PRODUCTION_TARGET_SENTENCES,
    build_sentence_acquisition_plan,
    write_frozen_acquisition_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-bootstrap-summary", required=True)
    parser.add_argument("--language-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-sentences", type=int, default=PRODUCTION_TARGET_SENTENCES)
    parser.add_argument("--alpha", type=float, default=PRODUCTION_ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-sentence-chars", type=int, default=DEFAULT_MAX_SENTENCE_CHARS
    )
    parser.add_argument(
        "--acquisition-safety-margin",
        type=float,
        default=DEFAULT_ACQUISITION_SAFETY_MARGIN,
    )
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args(argv)

    plan = build_sentence_acquisition_plan(
        input_manifest_path=Path(args.pilot_bootstrap_summary),
        language_map_path=Path(args.language_map),
        target_sentences=int(args.target_sentences),
        alpha=float(args.alpha),
        seed=int(args.seed),
        max_sentence_chars=int(args.max_sentence_chars),
        safety_margin=float(args.acquisition_safety_margin),
        test_mode=bool(args.test_mode),
    )
    output = Path(args.output)
    write_frozen_acquisition_plan(output_path=output, plan=plan)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output.resolve()),
                "plan_sha256": plan["plan_sha256"],
                "language_count": plan["language_count"],
                "target_sentence_count": plan["target_sentence_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
