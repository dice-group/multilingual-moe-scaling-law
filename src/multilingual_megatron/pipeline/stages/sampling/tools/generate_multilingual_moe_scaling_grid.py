#!/usr/bin/env python3
"""Generate the multilingual Switch-MoE scaling-law study artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (
    DEFAULT_FINEWEB2_CSV,
    DEFAULT_STUDY_DIR,
    build_corpus_manifest,
    build_language_sets,
    build_run_configs,
    build_study_spec,
    checkpoint_rows,
    read_fineweb2_inventory,
    run_matrix_rows,
    validate_artifacts,
    write_csv,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_STUDY_DIR))
    parser.add_argument("--fineweb2-csv", default=str(DEFAULT_FINEWEB2_CSV))
    parser.add_argument("--fail-on-validation-error", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    language_sets = build_language_sets(read_fineweb2_inventory(Path(args.fineweb2_csv)))
    corpus_manifest = build_corpus_manifest(language_sets)
    runs = build_run_configs(language_sets)
    errors = validate_artifacts(language_sets, runs)
    if errors and args.fail_on_validation_error:
        raise SystemExit("Generated invalid artifacts:\n" + "\n".join(f"- {error}" for error in errors))

    write_json(output_dir / "study_spec.json", build_study_spec())
    write_json(output_dir / "language_sets.json", language_sets)
    write_json(output_dir / "corpus_manifest.json", corpus_manifest)
    write_jsonl(output_dir / "run_configs.jsonl", runs)
    write_csv(output_dir / "run_matrix.csv", run_matrix_rows(runs))
    write_csv(output_dir / "checkpoint_observations.csv", checkpoint_rows(runs))
    write_json(
        output_dir / "validation_summary.json",
        {
            "valid": not errors,
            "errors": errors,
            "num_runs": len(runs),
            "num_checkpoint_observations": sum(len(run["checkpoints"]) for run in runs),
            "outputs": [
                "study_spec.json",
                "language_sets.json",
                "corpus_manifest.json",
                "run_configs.jsonl",
                "run_matrix.csv",
                "checkpoint_observations.csv",
            ],
        },
    )
    print(f"Wrote multilingual MoE scaling artifacts to {output_dir}")
    print(f"runs={len(runs)} checkpoint_observations={sum(len(run['checkpoints']) for run in runs)} valid={not errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
