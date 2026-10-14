"""Build a bounded, auditable FineWeb A/B-v2 bootstrap release.

The bootstrap is deliberately smaller than the final corpus release.  It gives
the study a deterministic source sample for tokenizer calibration, byte/token
estimation, validation reservation, and rehydration audits before expensive
sampling jobs are launched.

Production mode lists Parquet files from immutable Hugging Face revisions.
``--input-path`` provides an equivalent local-fixture mode for tests and offline
audits.  In both modes DataTrove's :class:`ParquetReader` performs the reads.

Train JSONL stores one row per original source document and declares the exact
FineWeb2 rehydration multiplier in its lineage.  The corpus finalizer emits all
declared replicas atomically.  This avoids duplicating a production-scale pool
before exact tokenizer-aware quota selection while retaining the released
``up_to_10_reps`` semantics.

``validation_reserve.jsonl`` is an oversized deterministic candidate pool,
not the final panel.  It is withheld from tokenizer training.  Once the final
tokenizer exists, the corpus finalizer selects the exact ATLAS 20%-token target
from this pool and returns every unused candidate to the model-training
selection pool with its
preserved FineWeb2 multiplier.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    sha256_file,
    verify_plan_hash,
)

from .artifacts import _load_language_selection, _load_object
from .contracts import BootstrapSettings
from .language import bootstrap_language
from .release import bootstrap_release, merge_bootstrap_release, preflight_paths
from .targets import (
    _derive_production_settings,
    _derive_tokenizer_acquisition_settings,
    _load_tokenizer_acquisition_plan,
    _validate_tokenizer_plan_language_map,
)

def run(
    *,
    preflight_paths_fn=preflight_paths,
    merge_bootstrap_release_fn=merge_bootstrap_release,
    bootstrap_language_fn=bootstrap_language,
    bootstrap_release_fn=bootstrap_release,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--rehydration-weights", required=True)
    parser.add_argument(
        "--source-inventory",
        help=(
            "Materialized pinned FineWeb2 language-distribution CSV. Defaults to "
            "fineweb2-language-distribution.csv beside --rehydration-weights."
        ),
    )
    parser.add_argument("--language-map", required=True)
    parser.add_argument(
        "--language-selection-map",
        help=(
            "Optional versioned-v2 execution subset of --language-map. It controls "
            "preflight, array-index, and merge coverage without changing the immutable "
            "corpus language-map identity."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--language",
        default="all",
        help="One language subset from the map, or 'all' (default).",
    )
    parser.add_argument(
        "--language-index",
        type=int,
        nargs="?",
        const=-1,
        help=(
            "Run exactly one zero-based language-map index and write only its language "
            "directory. With no value or a negative value, read SLURM_ARRAY_TASK_ID."
        ),
    )
    parser.add_argument(
        "--merge-summary",
        action="store_true",
        help=(
            "Do not acquire data. Hash-verify completed per-language tasks and create "
            "bootstrap_manifest.json/bootstrap_summary.json once after the array."
        ),
    )
    parser.add_argument(
        "--metadata-preflight",
        action="store_true",
        help=(
            "List and freeze ordered Parquet paths for the selected/all languages "
            "without reading Parquet. Run once before a language Slurm array."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.30,
        help=(
            "Pre-tokenizer whole-document candidate-reserve fraction. This must "
            "exceed the final token target; unused candidates return to the "
            "model-training selection pool."
        ),
    )
    parser.add_argument("--target-unique-documents", type=int, default=10_000)
    parser.add_argument("--target-utf8-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--calibration-documents", type=int, default=1_000)
    parser.add_argument(
        "--sampling-plan",
        help=(
            "Optional frozen v2 sampling plan. Per-language acquisition targets are "
            "derived from its physical token quotas and tokenizer calibration."
        ),
    )
    parser.add_argument(
        "--tokenizer-acquisition-plan",
        help=(
            "Frozen pilot-derived tokenizer sentence acquisition plan. Per-language "
            "document/byte targets are rebuilt into --output-dir."
        ),
    )
    parser.add_argument(
        "--production-safety-margin",
        type=float,
        default=1.10,
        help="Multiplicative source-volume margin used with --sampling-plan (default: 1.10).",
    )
    parser.add_argument(
        "--incremental-from",
        help=(
            "Optional prior frozen bootstrap root. It is hash-verified and the larger "
            "release is rebuilt deterministically into --output-dir; prior files are "
            "never appended to or modified."
        ),
    )
    parser.add_argument(
        "--input-path",
        help=(
            "Optional local Parquet file/root for offline fixtures. Multi-language roots "
            "must have per-language directories or contain a {language} placeholder."
        ),
    )
    args = parser.parse_args()

    base_settings = BootstrapSettings(
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        target_unique_documents=args.target_unique_documents,
        target_utf8_bytes=args.target_utf8_bytes,
        calibration_documents=args.calibration_documents,
    )
    source_lock_path = Path(args.source_lock)
    weights_path = Path(args.rehydration_weights)
    source_inventory_path = (
        Path(args.source_inventory) if args.source_inventory else None
    )
    language_map_path = Path(args.language_map)
    language_selection_map_path = (
        Path(args.language_selection_map) if args.language_selection_map else None
    )
    output_root = Path(args.output_dir)
    sampling_plan_path = Path(args.sampling_plan) if args.sampling_plan else None
    tokenizer_acquisition_plan_path = (
        Path(args.tokenizer_acquisition_plan)
        if args.tokenizer_acquisition_plan
        else None
    )
    if sampling_plan_path is not None and tokenizer_acquisition_plan_path is not None:
        parser.error("--sampling-plan and --tokenizer-acquisition-plan are mutually exclusive")
    incremental_root = Path(args.incremental_from) if args.incremental_from else None

    if args.metadata_preflight:
        if args.merge_summary or args.language_index is not None:
            parser.error(
                "--metadata-preflight is mutually exclusive with merge/array-task modes"
            )
        map_languages = _load_language_selection(
            language_map_path=language_map_path.resolve(),
            language_selection_map_path=(
                language_selection_map_path.resolve()
                if language_selection_map_path is not None
                else None
            ),
        )
        preflight_languages = (
            map_languages if args.language == "all" else (str(args.language),)
        )
        if sampling_plan_path is not None:
            sampling_plan = _load_object(sampling_plan_path.resolve())
            if not verify_plan_hash(sampling_plan):
                raise ValueError("Sampling plan SHA-256 payload hash mismatch")
            for language in preflight_languages:
                _derive_production_settings(
                    base=base_settings,
                    language=language,
                    sampling_plan=sampling_plan,
                    safety_margin=args.production_safety_margin,
                )
        if tokenizer_acquisition_plan_path is not None:
            acquisition_plan = _load_tokenizer_acquisition_plan(
                tokenizer_acquisition_plan_path.resolve()
            )
            _validate_tokenizer_plan_language_map(
                acquisition_plan=acquisition_plan,
                language_map_path=language_map_path,
            )
            for language in preflight_languages:
                _derive_tokenizer_acquisition_settings(
                    base=base_settings,
                    language=language,
                    acquisition_plan=acquisition_plan,
                    acquisition_plan_path=tokenizer_acquisition_plan_path.resolve(),
                )
        summary = preflight_paths_fn(
            source_lock_path=source_lock_path,
            rehydration_weights_path=weights_path,
            source_inventory_path=(
                source_inventory_path
                if source_inventory_path is not None
                else weights_path.parent / "fineweb2-language-distribution.csv"
            ),
            language_map_path=language_map_path,
            output_root=output_root,
            seed=args.seed,
            selected_languages=preflight_languages,
            language_selection_map_path=language_selection_map_path,
            local_input_path=Path(args.input_path) if args.input_path else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.merge_summary:
        if args.language_index is not None:
            parser.error("--merge-summary and --language-index are mutually exclusive")
        map_languages = _load_language_selection(
            language_map_path=language_map_path.resolve(),
            language_selection_map_path=(
                language_selection_map_path.resolve()
                if language_selection_map_path is not None
                else None
            ),
        )
        merge_languages = (
            map_languages if args.language == "all" else (str(args.language),)
        )
        summary = merge_bootstrap_release_fn(
            source_lock_path=source_lock_path,
            rehydration_weights_path=weights_path,
            language_map_path=language_map_path,
            output_root=output_root,
            selected_languages=merge_languages,
            language_selection_map_path=language_selection_map_path,
            source_inventory_path=source_inventory_path,
            sampling_plan_path=sampling_plan_path,
            tokenizer_acquisition_plan_path=tokenizer_acquisition_plan_path,
            incremental_from_root=incremental_root,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.language_index is not None:
        if args.language != "all":
            parser.error("--language-index cannot be combined with --language")
        index = int(args.language_index)
        if index < 0:
            raw_array_index = os.getenv("SLURM_ARRAY_TASK_ID")
            if raw_array_index is None:
                parser.error(
                    "negative/default --language-index requires SLURM_ARRAY_TASK_ID"
                )
            index = int(raw_array_index)
        map_languages = _load_language_selection(
            language_map_path=language_map_path.resolve(),
            language_selection_map_path=(
                language_selection_map_path.resolve()
                if language_selection_map_path is not None
                else None
            ),
        )
        if not 0 <= index < len(map_languages):
            parser.error(
                f"language index {index} is outside [0, {len(map_languages) - 1}]"
            )
        language = map_languages[index]
        language_settings = base_settings
        production_target = None
        tokenizer_target = None
        if sampling_plan_path is not None:
            sampling_plan = _load_object(sampling_plan_path.resolve())
            if not verify_plan_hash(sampling_plan):
                raise ValueError("Sampling plan SHA-256 payload hash mismatch")
            language_settings, production_target = _derive_production_settings(
                base=base_settings,
                language=language,
                sampling_plan=sampling_plan,
                safety_margin=args.production_safety_margin,
            )
            production_target["sampling_plan_file_sha256"] = sha256_file(
                sampling_plan_path.resolve()
            )
        if tokenizer_acquisition_plan_path is not None:
            acquisition_plan = _load_tokenizer_acquisition_plan(
                tokenizer_acquisition_plan_path.resolve()
            )
            _validate_tokenizer_plan_language_map(
                acquisition_plan=acquisition_plan,
                language_map_path=language_map_path,
            )
            language_settings, tokenizer_target = _derive_tokenizer_acquisition_settings(
                base=base_settings,
                language=language,
                acquisition_plan=acquisition_plan,
                acquisition_plan_path=tokenizer_acquisition_plan_path.resolve(),
            )
        report = bootstrap_language_fn(
            language=language,
            source_lock_path=source_lock_path,
            rehydration_weights_path=weights_path,
            language_map_path=language_map_path,
            output_root=output_root,
            settings=language_settings,
            local_input_path=Path(args.input_path) if args.input_path else None,
            source_inventory_path=source_inventory_path,
            multiple_languages=False,
            production_target=production_target,
            tokenizer_sentence_acquisition_target=tokenizer_target,
            incremental_from_root=incremental_root,
            language_selection_map_path=language_selection_map_path,
        )
        print(
            json.dumps(
                {
                    "language_index": index,
                    "language": language,
                    "report_payload_sha256": report["report_payload_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    summary = bootstrap_release_fn(
        source_lock_path=Path(args.source_lock),
        rehydration_weights_path=Path(args.rehydration_weights),
        language_map_path=Path(args.language_map),
        output_root=Path(args.output_dir),
        settings=base_settings,
        selected_language=args.language,
        local_input_path=Path(args.input_path) if args.input_path else None,
        source_inventory_path=source_inventory_path,
        sampling_plan_path=sampling_plan_path,
        tokenizer_acquisition_plan_path=tokenizer_acquisition_plan_path,
        production_safety_margin=args.production_safety_margin,
        incremental_from_root=incremental_root,
        language_selection_map_path=language_selection_map_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
