"""Release preflight, merge verification, and orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
    verify_plan_hash,
)

from .artifacts import (
    _load_language_selection,
    _load_languages,
    _load_object,
    _manifest_hash,
    _recount_tokenizer_train_sentence_slots,
    _validate_reusable_output,
    _write_or_verify_frozen_json,
)
from .contracts import OUTPUT_NAMES, SCHEMA_VERSION, TOOL_ID, BootstrapSettings
from .language import bootstrap_language
from .sources import (
    _language_thresholds,
    _load_protected_paths_cache,
    _load_source_availability,
    _prepare_paths_cache,
    _source_identity,
)
from .targets import (
    _derive_production_settings,
    _derive_tokenizer_acquisition_settings,
    _load_tokenizer_acquisition_plan,
    _validate_tokenizer_plan_language_map,
)

def _preflight_rehydration_coverage(
    *,
    source_lock: Mapping[str, Any],
    source_lock_path: Path,
    rehydration_weights_path: Path,
    source_inventory_path: Path,
    languages: Sequence[str],
) -> dict[str, Any]:
    """Validate compact corpus policy inputs before any HF or output access."""

    rehydration_lock = source_lock.get("non_english_corpus", {}).get(
        "rehydration", {}
    )
    if (
        not isinstance(rehydration_lock, dict)
        or rehydration_lock.get("enabled") is not True
        or rehydration_lock.get("weights_policy") != "up_to_10_reps"
        or rehydration_lock.get("missing_weights_policy")
        != "fail_closed_except_explicit_identity_overrides"
    ):
        raise ValueError(
            "Source lock must enable the fail-closed FineWeb2 up_to_10_reps policy"
        )
    weights_path = rehydration_weights_path.resolve()
    inventory_path = source_inventory_path.resolve()
    weights_sha256 = sha256_file(weights_path)
    expected_weights_sha256 = str(rehydration_lock.get("weights_sha256", ""))
    if weights_sha256 != expected_weights_sha256:
        raise ValueError(
            "Materialized FineWeb2 rehydration weights do not match the pinned source lock"
        )
    expected_inventory_sha256 = str(
        source_lock.get("non_english_corpus", {})
        .get("language_inventory", {})
        .get("sha256", "")
    )
    inventory_sha256 = sha256_file(inventory_path)
    if not expected_inventory_sha256 or inventory_sha256 != expected_inventory_sha256:
        raise ValueError("Materialized FineWeb2 inventory does not match the source lock")

    weights = _load_object(weights_path)
    overrides = rehydration_lock.get("identity_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("Source lock rehydration identity_overrides must be an object")
    for override_language in sorted(overrides):
        if override_language == "eng_Latn":
            raise ValueError("English cannot have a FineWeb2 identity override")
        _language_thresholds(
            weights,
            language=str(override_language),
            is_english=False,
            rehydration_lock=rehydration_lock,
        )

    coverage: dict[str, Any] = {}
    for language in languages:
        source = _source_identity(source_lock, language)
        thresholds, policy_source = _language_thresholds(
            weights,
            language=language,
            is_english=source.is_english,
            rehydration_lock=rehydration_lock,
        )
        availability = _load_source_availability(
            language=language,
            source=source,
            source_lock=source_lock,
            rehydration_weights_path=weights_path,
            source_inventory_path=inventory_path,
        )
        coverage[language] = {
            "policy_source": policy_source,
            "thresholds_sha256": canonical_json_sha256(thresholds),
            "inventory_documents": (
                None if availability is None else int(availability.documents)
            ),
            "inventory_utf8_bytes": (
                None if availability is None else int(availability.utf8_bytes)
            ),
        }
    return {
        "source_lock_sha256": sha256_file(source_lock_path),
        "rehydration_weights_sha256": weights_sha256,
        "source_inventory_sha256": inventory_sha256,
        "selected_language_count": len(languages),
        "selected_non_english_language_count": sum(
            language != "eng_Latn" for language in languages
        ),
        "languages": coverage,
    }


def preflight_paths(
    *,
    source_lock_path: Path,
    rehydration_weights_path: Path,
    source_inventory_path: Path,
    language_map_path: Path,
    output_root: Path,
    seed: int,
    selected_languages: Sequence[str] | None = None,
    language_selection_map_path: Path | None = None,
    local_input_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze ordered path manifests without opening any Parquet data."""

    source_lock_path = source_lock_path.resolve()
    language_map_path = language_map_path.resolve()
    output_root = output_root.resolve()
    source_lock = _load_object(source_lock_path)
    map_languages = _load_languages(language_map_path)
    selection_languages = _load_language_selection(
        language_map_path=language_map_path,
        language_selection_map_path=(
            language_selection_map_path.resolve()
            if language_selection_map_path is not None
            else None
        ),
    )
    languages = (
        tuple(selected_languages)
        if selected_languages is not None
        else selection_languages
    )
    if (
        not languages
        or not set(languages).issubset(map_languages)
        or not set(languages).issubset(selection_languages)
    ):
        raise ValueError("Preflight languages must be a non-empty subset of the language map")
    resolved_local_input = local_input_path.resolve() if local_input_path is not None else None
    rehydration_coverage = _preflight_rehydration_coverage(
        source_lock=source_lock,
        source_lock_path=source_lock_path,
        rehydration_weights_path=rehydration_weights_path,
        source_inventory_path=source_inventory_path,
        languages=languages,
    )

    manifests: dict[str, str] = {}
    for language in languages:
        source = _source_identity(source_lock, language)
        if resolved_local_input is None:
            expected_dataset = (
                "HuggingFaceFW/fineweb"
                if source.is_english
                else "HuggingFaceFW/fineweb-2"
            )
            if source.dataset != expected_dataset:
                raise ValueError(
                    f"Production {language} source must be the pinned {expected_dataset} dataset"
                )
        language_dir = output_root / language
        language_dir.mkdir(parents=True, exist_ok=True)
        protected = (language_dir / "bootstrap_report.json").exists() or any(
            (language_dir / name).exists() for name in OUTPUT_NAMES
        )
        if protected:
            manifest, _, _ = _load_protected_paths_cache(
                language_dir=language_dir,
                source=source,
                seed=seed,
                local_input_path=resolved_local_input,
                multiple_languages=len(languages) > 1,
            )
        else:
            manifest, _, _ = _prepare_paths_cache(
                language_dir=language_dir,
                source=source,
                seed=seed,
                local_input_path=resolved_local_input,
                multiple_languages=len(languages) > 1,
            )
        manifests[language] = str(manifest["manifest_sha256"])

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "mode": "metadata_preflight_no_parquet_reads",
        "source_lock_sha256": sha256_file(source_lock_path),
        "rehydration_weights_sha256": rehydration_coverage[
            "rehydration_weights_sha256"
        ],
        "source_inventory_sha256": rehydration_coverage[
            "source_inventory_sha256"
        ],
        "rehydration_coverage": rehydration_coverage,
        "language_map_sha256": sha256_file(language_map_path),
        **(
            {
                "language_selection_map_sha256": sha256_file(
                    language_selection_map_path.resolve()
                )
            }
            if language_selection_map_path is not None
            else {}
        ),
        "seed": int(seed),
        "languages": list(languages),
        "paths_manifest_sha256": manifests,
        "lfs_hash_policy": "resolved_lazily_only_for_parquet_shards_actually_read",
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _write_or_verify_frozen_json(output_root / "paths_preflight_summary.json", summary)
    return summary


def merge_bootstrap_release(
    *,
    source_lock_path: Path,
    rehydration_weights_path: Path,
    language_map_path: Path,
    output_root: Path,
    selected_languages: Sequence[str] | None = None,
    language_selection_map_path: Path | None = None,
    source_inventory_path: Path | None = None,
    sampling_plan_path: Path | None = None,
    tokenizer_acquisition_plan_path: Path | None = None,
    incremental_from_root: Path | None = None,
) -> dict[str, Any]:
    """Hash-verify completed language tasks and create shared release metadata."""

    source_lock_path = source_lock_path.resolve()
    rehydration_weights_path = rehydration_weights_path.resolve()
    language_map_path = language_map_path.resolve()
    output_root = output_root.resolve()
    map_languages = _load_languages(language_map_path)
    resolved_selection_map = (
        language_selection_map_path.resolve()
        if language_selection_map_path is not None
        else None
    )
    selection_languages = _load_language_selection(
        language_map_path=language_map_path,
        language_selection_map_path=resolved_selection_map,
    )
    languages = (
        tuple(selected_languages)
        if selected_languages is not None
        else selection_languages
    )
    if (
        not languages
        or not set(languages).issubset(map_languages)
        or not set(languages).issubset(selection_languages)
    ):
        raise ValueError("Merge languages must be a non-empty subset of the language map")

    expected_source_lock_hash = sha256_file(source_lock_path)
    expected_weights_hash = sha256_file(rehydration_weights_path)
    expected_language_map_hash = sha256_file(language_map_path)
    resolved_inventory = (
        source_inventory_path.resolve()
        if source_inventory_path is not None
        else rehydration_weights_path.parent / "fineweb2-language-distribution.csv"
    )
    inventory_hash = sha256_file(resolved_inventory) if resolved_inventory.is_file() else None
    reports: dict[str, Any] = {}
    for language in languages:
        language_dir = output_root / language
        report_path = language_dir / "bootstrap_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"Missing completed language report: {report_path}")
        candidate = _load_object(report_path)
        expected_config_hash = str(candidate.get("run_config_sha256", ""))
        report = _validate_reusable_output(
            language_dir=language_dir,
            expected_config_hash=expected_config_hash,
        )
        if report is None:
            raise AssertionError("completed report unexpectedly failed reuse validation")
        config = report["run_config"]
        expected = {
            "language": language,
            "source_lock_sha256": expected_source_lock_hash,
            "rehydration_weights_sha256": expected_weights_hash,
            "language_map_sha256": expected_language_map_hash,
            # Legacy reports omit this key, which is equivalent to no selector.
            # Reports created with a selector must never be merged without it.
            "language_selection_map_sha256": (
                sha256_file(resolved_selection_map)
                if resolved_selection_map is not None
                else None
            ),
            "source_inventory_sha256": inventory_hash,
        }
        drift = [key for key, value in expected.items() if config.get(key) != value]
        if drift:
            raise RuntimeError(
                f"Cannot merge {language}: frozen input hashes differ for {drift}"
            )
        paths_manifest = _load_object(language_dir / "paths_manifest.json")
        if (
            paths_manifest.get("manifest_sha256") != _manifest_hash(paths_manifest)
            or paths_manifest.get("manifest_sha256")
            != config.get("paths_manifest_sha256")
        ):
            raise RuntimeError(f"Cannot merge {language}: path manifest hash mismatch")
        reports[language] = report

    sampling_plan: dict[str, Any] | None = None
    if sampling_plan_path is not None and tokenizer_acquisition_plan_path is not None:
        raise ValueError(
            "Sampling-plan and tokenizer-acquisition-plan modes are mutually exclusive"
        )
    if sampling_plan_path is not None:
        sampling_plan_path = sampling_plan_path.resolve()
        sampling_plan = _load_object(sampling_plan_path)
        if not verify_plan_hash(sampling_plan):
            raise ValueError("Sampling plan SHA-256 payload hash mismatch")
        for language, report in reports.items():
            target = report["run_config"].get("production_target")
            if not isinstance(target, dict) or target.get("sampling_plan_sha256") != sampling_plan[
                "plan_sha256"
            ]:
                raise RuntimeError(
                    f"Cannot merge {language}: it was not acquired for this sampling plan"
                )

    tokenizer_acquisition_plan: dict[str, Any] | None = None
    tokenizer_slot_recounts: dict[str, dict[str, Any]] = {}
    if tokenizer_acquisition_plan_path is not None:
        tokenizer_acquisition_plan_path = tokenizer_acquisition_plan_path.resolve()
        tokenizer_acquisition_plan = _load_tokenizer_acquisition_plan(
            tokenizer_acquisition_plan_path
        )
        _validate_tokenizer_plan_language_map(
            acquisition_plan=tokenizer_acquisition_plan,
            language_map_path=language_map_path,
        )
        for language, report in reports.items():
            _, expected_target = _derive_tokenizer_acquisition_settings(
                base=BootstrapSettings(),
                language=language,
                acquisition_plan=tokenizer_acquisition_plan,
                acquisition_plan_path=tokenizer_acquisition_plan_path,
            )
            if report["run_config"].get("tokenizer_sentence_acquisition_target") != expected_target:
                raise RuntimeError(
                    f"Cannot merge {language}: it was not acquired from this tokenizer plan"
                )
            gate_contract = expected_target["sentence_slot_gate"]
            splitter = gate_contract["splitter"]
            recount = _recount_tokenizer_train_sentence_slots(
                path=output_root / language / "train.jsonl",
                language=language,
                max_sentence_chars=int(splitter["max_sentence_chars"]),
            )
            required_slots = int(
                expected_target["required_rehydrated_sentence_slots"]
            )
            actual_slots = int(recount["rehydrated_train_sentence_slots"])
            shortfall_slots = max(0, required_slots - actual_slots)
            targets = report.get("targets")
            counts = report.get("counts")
            if not isinstance(targets, dict) or not isinstance(counts, dict):
                raise RuntimeError(
                    f"Cannot merge {language}: tokenizer slot accounting is missing"
                )
            true_source_exhausted = bool(targets.get("source_exhausted"))
            expected_audit = {
                "counter_id": gate_contract["counter_id"],
                "counted_split": gate_contract["counted_split"],
                "slot_formula": gate_contract["slot_formula"],
                "splitter": splitter,
                "non_exhausted_stop_rule": gate_contract[
                    "non_exhausted_stop_rule"
                ],
                "required_rehydrated_sentence_slots": required_slots,
                "actual_unique_train_sentences": int(
                    recount["unique_train_sentences"]
                ),
                "actual_rehydrated_sentence_slots": actual_slots,
                "shortfall_slots": shortfall_slots,
                "quota_satisfied": shortfall_slots == 0,
                "true_pinned_source_exhausted": true_source_exhausted,
                "true_source_exhausted_repetition_allowed": bool(
                    true_source_exhausted and shortfall_slots > 0
                ),
                "artificial_pool_cycling_forbidden": True,
            }
            drift = []
            reported_train_documents = counts.get("unique_train_documents")
            if (
                not isinstance(reported_train_documents, int)
                or isinstance(reported_train_documents, bool)
                or reported_train_documents != int(recount["train_documents"])
            ):
                drift.append("unique_train_documents")
            reported_unique_sentences = counts.get("unique_train_sentences")
            if (
                not isinstance(reported_unique_sentences, int)
                or isinstance(reported_unique_sentences, bool)
                or reported_unique_sentences
                != int(recount["unique_train_sentences"])
            ):
                drift.append("unique_train_sentences")
            reported_slots = counts.get("rehydrated_train_sentence_slots")
            if (
                not isinstance(reported_slots, int)
                or isinstance(reported_slots, bool)
                or reported_slots != actual_slots
            ):
                drift.append("rehydrated_train_sentence_slots")
            if targets.get("tokenizer_sentence_slot_gate") != expected_audit:
                drift.append("tokenizer_sentence_slot_gate")
            if drift:
                raise RuntimeError(
                    f"Cannot merge {language}: exact tokenizer slot recount drifted for {drift}"
                )
            if shortfall_slots and not true_source_exhausted:
                raise RuntimeError(
                    f"Cannot merge {language}: non-exhausted tokenizer pool is "
                    f"{shortfall_slots} sentence slots short"
                )
            tokenizer_slot_recounts[language] = expected_audit

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "tool_id": TOOL_ID,
            "source_lock_sha256": expected_source_lock_hash,
            "rehydration_weights_sha256": expected_weights_hash,
            "language_map_sha256": expected_language_map_hash,
            **(
                {
                    "language_selection_map_sha256": sha256_file(
                        resolved_selection_map
                    )
                }
                if resolved_selection_map is not None
                else {}
            ),
            "sampling_plan_file_sha256": (
                sha256_file(sampling_plan_path) if sampling_plan_path is not None else None
            ),
            "sampling_plan_payload_sha256": (
                sampling_plan["plan_sha256"] if sampling_plan is not None else None
            ),
            "tokenizer_acquisition_plan_file_sha256": (
                sha256_file(tokenizer_acquisition_plan_path)
                if tokenizer_acquisition_plan_path is not None
                else None
            ),
            "tokenizer_acquisition_plan_payload_sha256": (
                tokenizer_acquisition_plan["plan_sha256"]
                if tokenizer_acquisition_plan is not None
                else None
            ),
            "incremental_mode": (
                "verified_full_deterministic_rebuild"
                if incremental_from_root is not None
                else None
            ),
        },
        "languages": {
            language: {
                "rehydration": {
                    "policy_source": report["run_config"][
                        "rehydration_policy_source"
                    ],
                    "thresholds_sha256": report["run_config"][
                        "rehydration_thresholds_sha256"
                    ],
                    "identity_override": bool(
                        report["rehydration"]["identity_override"]
                    ),
                },
                "acquisition": {
                    "report_path": f"{language}/bootstrap_report.json",
                    "report_payload_sha256": report["report_payload_sha256"],
                    "acquisition_status": report["acquisition_status"],
                    "complete_source_scan": bool(
                        report["targets"]["source_exhausted"]
                        or report["targets"]["pinned_inventory_exhausted"]
                    ),
                    "unique_source_documents": int(
                        report["counts"]["unique_source_documents"]
                    ),
                    "unique_source_utf8_bytes": int(
                        report["counts"]["unique_source_utf8_bytes"]
                    ),
                }
                | (
                    {
                        "tokenizer_sentence_slot_gate": tokenizer_slot_recounts[
                            language
                        ]
                    }
                    if language in tokenizer_slot_recounts
                    else {}
                ),
                "train": {
                    "path": f"{language}/train.jsonl",
                    "sha256": report["outputs"]["train.jsonl"]["sha256"],
                },
                "validation_reserve": {
                    "path": f"{language}/validation_reserve.jsonl",
                    "sha256": report["outputs"]["validation_reserve.jsonl"]["sha256"],
                },
            }
            for language, report in reports.items()
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_or_verify_frozen_json(output_root / "bootstrap_manifest.json", manifest)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "languages": list(languages),
        "report_payload_sha256": {
            language: report["report_payload_sha256"] for language, report in reports.items()
        },
        "bootstrap_manifest_payload_sha256": manifest["manifest_sha256"],
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _write_or_verify_frozen_json(output_root / "bootstrap_summary.json", summary)
    return summary


def bootstrap_release(
    *,
    source_lock_path: Path,
    rehydration_weights_path: Path,
    language_map_path: Path,
    output_root: Path,
    settings: BootstrapSettings,
    selected_language: str,
    local_input_path: Path | None = None,
    source_inventory_path: Path | None = None,
    sampling_plan_path: Path | None = None,
    tokenizer_acquisition_plan_path: Path | None = None,
    production_safety_margin: float = 1.10,
    incremental_from_root: Path | None = None,
    language_selection_map_path: Path | None = None,
) -> dict[str, Any]:
    languages = _load_language_selection(
        language_map_path=language_map_path.resolve(),
        language_selection_map_path=(
            language_selection_map_path.resolve()
            if language_selection_map_path is not None
            else None
        ),
    )
    if selected_language == "all":
        selected = languages
    elif selected_language in languages:
        selected = (selected_language,)
    else:
        raise ValueError(f"Language {selected_language!r} is not present in the language map")

    sampling_plan: dict[str, Any] | None = None
    tokenizer_acquisition_plan: dict[str, Any] | None = None
    if sampling_plan_path is not None and tokenizer_acquisition_plan_path is not None:
        raise ValueError(
            "Sampling-plan and tokenizer-acquisition-plan modes are mutually exclusive"
        )
    if sampling_plan_path is not None:
        sampling_plan_path = sampling_plan_path.resolve()
        sampling_plan = _load_object(sampling_plan_path)
        if not verify_plan_hash(sampling_plan):
            raise ValueError("Sampling plan SHA-256 payload hash mismatch")
    if tokenizer_acquisition_plan_path is not None:
        tokenizer_acquisition_plan_path = tokenizer_acquisition_plan_path.resolve()
        tokenizer_acquisition_plan = _load_tokenizer_acquisition_plan(
            tokenizer_acquisition_plan_path
        )
        _validate_tokenizer_plan_language_map(
            acquisition_plan=tokenizer_acquisition_plan,
            language_map_path=language_map_path,
        )

    reports: dict[str, Any] = {}
    for language in selected:
        language_settings = settings
        production_target = None
        tokenizer_target = None
        if sampling_plan is not None:
            language_settings, production_target = _derive_production_settings(
                base=settings,
                language=language,
                sampling_plan=sampling_plan,
                safety_margin=production_safety_margin,
            )
            production_target["sampling_plan_file_sha256"] = sha256_file(
                sampling_plan_path
            )
        if tokenizer_acquisition_plan is not None:
            assert tokenizer_acquisition_plan_path is not None
            language_settings, tokenizer_target = _derive_tokenizer_acquisition_settings(
                base=settings,
                language=language,
                acquisition_plan=tokenizer_acquisition_plan,
                acquisition_plan_path=tokenizer_acquisition_plan_path,
            )
        reports[language] = bootstrap_language(
            language=language,
            source_lock_path=source_lock_path,
            rehydration_weights_path=rehydration_weights_path,
            language_map_path=language_map_path,
            output_root=output_root,
            settings=language_settings,
            local_input_path=local_input_path,
            source_inventory_path=source_inventory_path,
            multiple_languages=len(selected) > 1,
            production_target=production_target,
            tokenizer_sentence_acquisition_target=tokenizer_target,
            incremental_from_root=incremental_from_root,
            language_selection_map_path=language_selection_map_path,
        )
    return merge_bootstrap_release(
        source_lock_path=source_lock_path,
        rehydration_weights_path=rehydration_weights_path,
        language_map_path=language_map_path,
        output_root=output_root,
        selected_languages=selected,
        source_inventory_path=source_inventory_path,
        sampling_plan_path=sampling_plan_path,
        tokenizer_acquisition_plan_path=tokenizer_acquisition_plan_path,
        incremental_from_root=incremental_from_root,
        language_selection_map_path=language_selection_map_path,
    )
