#!/usr/bin/env python3
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

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    assign_validation_reserve,
    canonical_json_sha256,
    rehydration_multiplier,
    sha256_file,
    stable_document_id,
    verify_plan_hash,
)
from multilingual_megatron.pipeline.tokenizer_sentence_contract import (
    ACQUISITION_PLAN_TOOL_ID,
    SENTENCE_SLOT_COUNTER_ID,
    sentence_slot_gate_contract,
    split_sentence_like,
    tokenizer_acquisition_target_contract,
    verify_acquisition_plan,
)

from .fineweb_v2_bootstrap.artifacts import (
    _atomic_write_json,
    _atomic_write_text,
    _load_language_selection,
    _load_languages,
    _load_object,
    _manifest_hash,
    _record,
    _recount_tokenizer_train_sentence_slots,
    _validate_incremental_predecessor,
    _validate_reusable_output,
    _write_jsonl_record,
    _write_or_verify_frozen_json,
)
from .fineweb_v2_bootstrap.cli import run as _run_cli
from .fineweb_v2_bootstrap.contracts import (
    IMMUTABLE_REVISION_RE,
    OUTPUT_NAMES,
    SCHEMA_VERSION,
    TOOL_ID,
    BootstrapSettings,
    SourceAvailability,
    SourceIdentity,
)
from .fineweb_v2_bootstrap.language import bootstrap_language
from .fineweb_v2_bootstrap.release import (
    _preflight_rehydration_coverage,
    bootstrap_release,
    merge_bootstrap_release,
    preflight_paths,
)
from .fineweb_v2_bootstrap.sources import (
    _HF_FILE_SHA256_CACHE,
    _HF_REPO_FILES_CACHE,
    _build_paths_manifest,
    _cap_settings_to_availability,
    _cluster_size,
    _fineweb_parquet_adapter,
    _hf_file_sha256,
    _language_thresholds,
    _list_hf_parquet_paths,
    _load_protected_paths_cache,
    _load_source_availability,
    _ordered_paths,
    _prepare_paths_cache,
    _resolve_local_parquets,
    _source_identity,
    _validated_thresholds,
)
from .fineweb_v2_bootstrap.targets import (
    _derive_production_settings,
    _derive_tokenizer_acquisition_settings,
    _load_tokenizer_acquisition_plan,
    _tokenizer_sentence_slot_gate,
    _validate_tokenizer_plan_language_map,
)


def main() -> int:
    """Run the legacy CLI with facade-level dependencies for patch compatibility."""

    return _run_cli(
        preflight_paths_fn=preflight_paths,
        merge_bootstrap_release_fn=merge_bootstrap_release,
        bootstrap_language_fn=bootstrap_language,
        bootstrap_release_fn=bootstrap_release,
    )


_LEGACY_DEFINITIONS = (
    BootstrapSettings,
    SourceIdentity,
    SourceAvailability,
    _fineweb_parquet_adapter,
    _load_source_availability,
    _cap_settings_to_availability,
    _derive_production_settings,
    _load_tokenizer_acquisition_plan,
    _derive_tokenizer_acquisition_settings,
    _tokenizer_sentence_slot_gate,
    _validate_tokenizer_plan_language_map,
    _load_object,
    _atomic_write_text,
    _atomic_write_json,
    _write_or_verify_frozen_json,
    _manifest_hash,
    _load_languages,
    _load_language_selection,
    _source_identity,
    _ordered_paths,
    _list_hf_parquet_paths,
    _hf_file_sha256,
    _resolve_local_parquets,
    _build_paths_manifest,
    _prepare_paths_cache,
    _load_protected_paths_cache,
    _language_thresholds,
    _validated_thresholds,
    _cluster_size,
    _record,
    _write_jsonl_record,
    _recount_tokenizer_train_sentence_slots,
    _validate_reusable_output,
    _validate_incremental_predecessor,
    bootstrap_language,
    _preflight_rehydration_coverage,
    preflight_paths,
    merge_bootstrap_release,
    bootstrap_release,
)

# Keep introspection and serialized references compatible with the historical
# module even though implementation ownership moved into stage-local modules.
for _definition in _LEGACY_DEFINITIONS:
    _definition.__module__ = __name__


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapSettings",
    "bootstrap_language",
    "bootstrap_release",
    "merge_bootstrap_release",
    "preflight_paths",
    "main",
]
