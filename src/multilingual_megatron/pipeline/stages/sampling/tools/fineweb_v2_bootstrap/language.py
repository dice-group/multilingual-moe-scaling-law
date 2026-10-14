"""Deterministic per-language FineWeb v2 bootstrap acquisition."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    assign_validation_reserve,
    canonical_json_sha256,
    rehydration_multiplier,
    sha256_file,
    stable_document_id,
)
from multilingual_megatron.pipeline.tokenizer_sentence_contract import split_sentence_like

from .artifacts import (
    _atomic_write_json,
    _load_language_selection,
    _load_object,
    _record,
    _validate_incremental_predecessor,
    _validate_reusable_output,
    _write_jsonl_record,
)
from .contracts import OUTPUT_NAMES, SCHEMA_VERSION, TOOL_ID, BootstrapSettings
from .sources import (
    _cap_settings_to_availability,
    _cluster_size,
    _fineweb_parquet_adapter,
    _hf_file_sha256,
    _language_thresholds,
    _load_protected_paths_cache,
    _load_source_availability,
    _prepare_paths_cache,
    _source_identity,
)
from .targets import _tokenizer_sentence_slot_gate

def bootstrap_language(
    *,
    language: str,
    source_lock_path: Path,
    rehydration_weights_path: Path,
    language_map_path: Path,
    output_root: Path,
    settings: BootstrapSettings,
    local_input_path: Path | None = None,
    source_inventory_path: Path | None = None,
    multiple_languages: bool = False,
    production_target: Mapping[str, Any] | None = None,
    tokenizer_sentence_acquisition_target: Mapping[str, Any] | None = None,
    incremental_from_root: Path | None = None,
    language_selection_map_path: Path | None = None,
) -> dict[str, Any]:
    """Create or hash-verify one language bootstrap."""

    requested_settings = settings
    requested_settings.validate()
    source_lock_path = source_lock_path.resolve()
    rehydration_weights_path = rehydration_weights_path.resolve()
    language_map_path = language_map_path.resolve()
    resolved_selection_map = (
        language_selection_map_path.resolve()
        if language_selection_map_path is not None
        else None
    )
    output_root = output_root.resolve()

    languages = _load_language_selection(
        language_map_path=language_map_path,
        language_selection_map_path=resolved_selection_map,
    )
    if language not in languages:
        raise ValueError(f"Language {language!r} is not present in {language_map_path}")
    source_lock = _load_object(source_lock_path)
    weights = _load_object(rehydration_weights_path)
    source = _source_identity(source_lock, language)
    if local_input_path is None:
        expected_dataset = (
            "HuggingFaceFW/fineweb" if source.is_english else "HuggingFaceFW/fineweb-2"
        )
        if source.dataset != expected_dataset:
            raise ValueError(
                f"Production {language} source must be the pinned {expected_dataset} dataset"
            )

    rehydration_lock = source_lock.get("non_english_corpus", {}).get("rehydration", {})
    if (
        not isinstance(rehydration_lock, dict)
        or rehydration_lock.get("enabled") is not True
        or rehydration_lock.get("weights_policy") != "up_to_10_reps"
        or rehydration_lock.get("missing_weights_policy")
        != "fail_closed_except_explicit_identity_overrides"
        or not isinstance(rehydration_lock.get("identity_overrides"), dict)
    ):
        raise ValueError(
            "Source lock must enable the fail-closed FineWeb2 up_to_10_reps policy"
        )
    expected_weights_hash = str(rehydration_lock.get("weights_sha256", ""))
    observed_weights_hash = sha256_file(rehydration_weights_path)
    if observed_weights_hash != expected_weights_hash:
        raise ValueError(
            "Materialized FineWeb2 rehydration weights do not match the pinned source lock"
        )
    thresholds, rehydration_policy_source = _language_thresholds(
        weights,
        language=language,
        is_english=source.is_english,
        rehydration_lock=rehydration_lock,
    )
    availability = _load_source_availability(
        language=language,
        source=source,
        source_lock=source_lock,
        rehydration_weights_path=rehydration_weights_path,
        source_inventory_path=source_inventory_path,
    )
    settings, availability_policy = _cap_settings_to_availability(
        requested_settings,
        availability,
    )
    # A corrected production plan may lower one proxy target while raising or
    # preserving the exact token requirement.  Incremental recovery must never
    # discard already acquired documents or bytes, so retain the predecessor's
    # larger effective bounds before rebuilding into the new immutable root.
    if incremental_from_root is not None:
        predecessor_report_path = (
            incremental_from_root.resolve() / language / "bootstrap_report.json"
        )
        predecessor_report = _load_object(predecessor_report_path)
        predecessor_config = predecessor_report.get("run_config")
        predecessor_settings = (
            predecessor_config.get("settings")
            if isinstance(predecessor_config, dict)
            else None
        )
        if not isinstance(predecessor_settings, dict):
            raise ValueError(
                f"incremental predecessor lacks settings: {predecessor_report_path}"
            )
        settings = BootstrapSettings(
            seed=settings.seed,
            validation_fraction=settings.validation_fraction,
            target_unique_documents=max(
                settings.target_unique_documents,
                int(predecessor_settings.get("target_unique_documents", 0)),
            ),
            target_utf8_bytes=max(
                settings.target_utf8_bytes,
                int(predecessor_settings.get("target_utf8_bytes", 0)),
            ),
            calibration_documents=settings.calibration_documents,
        )
    settings.validate()
    tokenizer_slot_gate = _tokenizer_sentence_slot_gate(
        tokenizer_sentence_acquisition_target
    )
    resolved_inventory_path = (
        source_inventory_path.resolve()
        if source_inventory_path is not None
        else rehydration_weights_path.parent / "fineweb2-language-distribution.csv"
    )
    source_inventory_sha256 = (
        sha256_file(resolved_inventory_path) if resolved_inventory_path.is_file() else None
    )

    language_dir = output_root / language
    language_dir.mkdir(parents=True, exist_ok=True)
    resolved_local_input = local_input_path.resolve() if local_input_path is not None else None
    has_protected_artifacts = (language_dir / "bootstrap_report.json").exists() or any(
        (language_dir / name).exists() for name in OUTPUT_NAMES
    )
    if has_protected_artifacts:
        paths_manifest, data_folder, paths_file = _load_protected_paths_cache(
            language_dir=language_dir,
            source=source,
            seed=settings.seed,
            local_input_path=resolved_local_input,
            multiple_languages=multiple_languages,
        )
    else:
        paths_manifest, data_folder, paths_file = _prepare_paths_cache(
            language_dir=language_dir,
            source=source,
            seed=settings.seed,
            local_input_path=resolved_local_input,
            multiple_languages=multiple_languages,
        )

    settings_payload = asdict(settings)
    settings_payload["validation_candidate_fraction"] = float(
        settings.validation_fraction
    )
    run_config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": TOOL_ID,
        "language": language,
        "source": asdict(source),
        "source_lock_sha256": sha256_file(source_lock_path),
        "rehydration_weights_sha256": observed_weights_hash,
        "rehydration_policy_source": rehydration_policy_source,
        "rehydration_thresholds_sha256": canonical_json_sha256(thresholds),
        "language_map_sha256": sha256_file(language_map_path),
        **(
            {"language_selection_map_sha256": sha256_file(resolved_selection_map)}
            if resolved_selection_map is not None
            else {}
        ),
        "source_inventory_sha256": source_inventory_sha256,
        "paths_manifest_sha256": str(paths_manifest["manifest_sha256"]),
        "settings": settings_payload,
        "validation_candidate_contract": {
            "membership_namespace": "validation-reserve-v1",
            "excluded_from_tokenizer_training": True,
            "unused_candidates_return_to_model_training_selection_pool": True,
            "preserved_train_rehydration_multiplier": True,
        },
        "availability_policy": availability_policy,
    }
    if production_target is not None:
        run_config["production_target"] = dict(production_target)
    if tokenizer_sentence_acquisition_target is not None:
        run_config["tokenizer_sentence_acquisition_target"] = dict(
            tokenizer_sentence_acquisition_target
        )
    if incremental_from_root is not None:
        predecessor_root = incremental_from_root.resolve()
        if predecessor_root == output_root:
            raise ValueError("incremental_from_root must differ from the new output root")
        predecessor = _validate_incremental_predecessor(
            predecessor_dir=predecessor_root / language,
            current_run_config=run_config,
        )
        run_config["incremental_extension"] = {
            "predecessor_root": str(predecessor_root),
            "predecessor_report_payload_sha256": predecessor["report_payload_sha256"],
            "mode": "verified_full_deterministic_rebuild",
            "reason": (
                "completed hash-addressed outputs are immutable; larger targets rebuild "
                "into a new root instead of appending in place"
            ),
        }
    run_config_hash = canonical_json_sha256(run_config)
    reusable = _validate_reusable_output(
        language_dir=language_dir,
        expected_config_hash=run_config_hash,
    )
    if reusable is not None:
        return reusable

    try:
        from datatrove.pipeline.readers import ParquetReader
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("datatrove is required to read FineWeb Parquet inputs") from exc

    temporary_paths: dict[str, Path] = {}
    handles: dict[str, Any] = {}
    try:
        for name in OUTPUT_NAMES:
            fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=language_dir)
            temporary_paths[name] = Path(temporary_name)
            handles[name] = os.fdopen(fd, "w", encoding="utf-8")

        reader = ParquetReader(
            data_folder,
            paths_file=str(paths_file),
            read_metadata=True,
            file_progress=False,
            doc_progress=False,
            shuffle_files=False,
            adapter=_fineweb_parquet_adapter,
        )
        source_file_hashes = {
            str(entry["path"]): str(entry["content_sha256"])
            for entry in paths_manifest["files"]
            if "content_sha256" in entry
        }
        pinned_source_files = {
            str(entry["path"]) for entry in paths_manifest["files"]
        }
        read_source_file_hashes: dict[str, str] = {}
        seen: set[str] = set()
        train_ids: set[str] = set()
        validation_ids: set[str] = set()
        calibration_ids: set[str] = set()
        replica_histogram: Counter[int] = Counter()
        validation_candidate_train_replica_histogram: Counter[int] = Counter()
        cluster_histogram: Counter[int] = Counter()
        duplicate_source_documents = 0
        source_utf8_bytes = 0
        logical_train_records = 0
        train_unique_sentence_count = 0
        train_rehydrated_sentence_slots = 0
        raw_source_documents_read = 0

        source_exhausted = True
        for document in reader.run(rank=0, world_size=1):
            raw_source_documents_read += 1
            text = str(document.text)
            original_id = str(document.id)
            metadata = document.metadata if isinstance(document.metadata, dict) else {}
            url = str(metadata.get("url", "") or "")
            stable_id = stable_document_id(
                source_dataset=source.dataset,
                source_revision=source.revision,
                subset=source.subset,
                source_id=original_id,
                url=url,
                text=text,
            )
            if stable_id in seen:
                duplicate_source_documents += 1
                continue
            seen.add(stable_id)
            source_utf8_bytes += len(text.encode("utf-8"))
            cluster_size = _cluster_size(
                metadata,
                is_english=source.is_english,
                document_id=stable_id,
            )
            cluster_histogram[cluster_size] += 1
            source_file = str(metadata.get("source_parquet_path", "") or "")
            source_file_sha256 = source_file_hashes.get(source_file)
            if source_file_sha256 is None:
                if source_file not in pinned_source_files:
                    raise RuntimeError(
                        f"{language}: DataTrove returned an unpinned source path {source_file!r}"
                    )
                if resolved_local_input is not None:
                    raise RuntimeError(
                        f"{language}: local source path lacks its strict content hash"
                    )
                source_file_sha256 = _hf_file_sha256(source, [source_file])[source_file]
                source_file_hashes[source_file] = source_file_sha256
            read_source_file_hashes[source_file] = source_file_sha256

            # Freeze an oversized candidate reserve before tokenizer training.
            # The exact ATLAS 20%-token panel is selected later with the final
            # tokenizer; unused candidates return to the model-training
            # selection pool with the
            # multiplier preserved below.
            replicas = (
                1
                if source.is_english
                else rehydration_multiplier(cluster_size, thresholds)
            )
            is_validation = assign_validation_reserve(
                document_id=stable_id,
                seed=settings.seed,
                validation_fraction=settings.validation_fraction,
            )
            if is_validation:
                validation_ids.add(stable_id)
                validation_candidate_train_replica_histogram[replicas] += 1
                _write_jsonl_record(
                    handles["validation_reserve.jsonl"],
                    _record(
                        text=text,
                        stable_id=stable_id,
                        original_id=original_id,
                        url=url,
                        source=source,
                        source_file=source_file,
                        source_file_sha256=source_file_sha256,
                        split="validation_reserve",
                        artifact_role="validation_reserve",
                        rehydration_multiplier=1,
                        train_rehydration_multiplier=replicas,
                        cluster_size=cluster_size,
                    ),
                )
            else:
                train_ids.add(stable_id)
                if tokenizer_slot_gate is not None:
                    _, max_sentence_chars, _ = tokenizer_slot_gate
                    document_sentence_count = sum(
                        1
                        for _ in split_sentence_like(
                            text,
                            max_chars=max_sentence_chars,
                        )
                    )
                    if document_sentence_count <= 0:
                        raise RuntimeError(
                            f"{language}: frozen sentence splitter produced no train slots"
                        )
                    train_unique_sentence_count += document_sentence_count
                    train_rehydrated_sentence_slots += (
                        document_sentence_count * replicas
                    )
                replica_histogram[replicas] += 1
                _write_jsonl_record(
                    handles["train.jsonl"],
                    _record(
                        text=text,
                        stable_id=stable_id,
                        original_id=original_id,
                        url=url,
                        source=source,
                        source_file=source_file,
                        source_file_sha256=source_file_sha256,
                        split="train",
                        artifact_role="train",
                        rehydration_multiplier=replicas,
                        train_rehydration_multiplier=replicas,
                        cluster_size=cluster_size,
                    ),
                )
                logical_train_records += replicas
                if len(calibration_ids) < settings.calibration_documents:
                    calibration_ids.add(stable_id)
                    _write_jsonl_record(
                        handles["calibration.jsonl"],
                        _record(
                            text=text,
                            stable_id=stable_id,
                            original_id=original_id,
                            url=url,
                            source=source,
                            source_file=source_file,
                            source_file_sha256=source_file_sha256,
                            split="train",
                            artifact_role="calibration",
                            rehydration_multiplier=replicas,
                            train_rehydration_multiplier=replicas,
                            cluster_size=cluster_size,
                        ),
                    )

            sentence_slot_quota_met = (
                tokenizer_slot_gate is None
                or train_rehydrated_sentence_slots >= tokenizer_slot_gate[0]
            )
            targets_met = (
                len(seen) >= settings.target_unique_documents
                and source_utf8_bytes >= settings.target_utf8_bytes
                and len(calibration_ids) >= settings.calibration_documents
                and sentence_slot_quota_met
            )
            if targets_met:
                source_exhausted = False
                break

        required_calibration_documents = min(
            settings.calibration_documents,
            len(train_ids),
        )
        if len(calibration_ids) != required_calibration_documents:
            raise RuntimeError(
                f"{language}: calibration selection drift: observed={len(calibration_ids)}, "
                f"required={required_calibration_documents}"
            )
        if not train_ids:
            raise RuntimeError(
                f"{language}: source produced no unique train documents after validation reserve"
            )
        overlap = train_ids & validation_ids
        if overlap:
            raise AssertionError(f"{language}: train/validation stable-ID overlap detected")
        if not calibration_ids.issubset(train_ids):
            raise AssertionError(
                f"{language}: calibration is not a subset of unique train documents"
            )
        pinned_inventory_exhausted = bool(
            availability is not None
            and (
                len(seen) >= availability.documents
                or source_utf8_bytes >= availability.utf8_bytes
            )
        )
        tokenizer_sentence_slot_audit: dict[str, Any] | None = None
        if tokenizer_slot_gate is not None:
            required_slots, _, gate_contract = tokenizer_slot_gate
            shortfall_slots = max(
                0,
                required_slots - train_rehydrated_sentence_slots,
            )
            if shortfall_slots and not source_exhausted:
                raise RuntimeError(
                    f"{language}: non-exhausted tokenizer acquisition stopped with "
                    f"a {shortfall_slots}-slot shortfall"
                )
            tokenizer_sentence_slot_audit = {
                "counter_id": gate_contract["counter_id"],
                "counted_split": gate_contract["counted_split"],
                "slot_formula": gate_contract["slot_formula"],
                "splitter": gate_contract["splitter"],
                "non_exhausted_stop_rule": gate_contract[
                    "non_exhausted_stop_rule"
                ],
                "required_rehydrated_sentence_slots": required_slots,
                "actual_unique_train_sentences": train_unique_sentence_count,
                "actual_rehydrated_sentence_slots": train_rehydrated_sentence_slots,
                "shortfall_slots": shortfall_slots,
                "quota_satisfied": shortfall_slots == 0,
                "true_pinned_source_exhausted": bool(source_exhausted),
                "true_source_exhausted_repetition_allowed": bool(
                    source_exhausted and shortfall_slots > 0
                ),
                "artificial_pool_cycling_forbidden": True,
            }

        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        for name in OUTPUT_NAMES:
            os.replace(temporary_paths[name], language_dir / name)

        output_lineage = {
            name: {
                "path": name,
                "sha256": sha256_file(language_dir / name),
                "bytes": (language_dir / name).stat().st_size,
            }
            for name in OUTPUT_NAMES
        }
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tool_id": TOOL_ID,
            "status": "complete",
            "acquisition_status": (
                "source_exhausted"
                if source_exhausted
                else (
                    "pinned_inventory_exhausted"
                    if pinned_inventory_exhausted
                    else "targets_reached"
                )
            ),
            "run_config": run_config,
            "run_config_sha256": run_config_hash,
            "lineage": {
                "source_lock_path": str(source_lock_path),
                "rehydration_weights_path": str(rehydration_weights_path),
                "language_map_path": str(language_map_path),
                "paths_manifest": "paths_manifest.json",
                "paths_manifest_sha256": paths_manifest["manifest_sha256"],
                "ordered_paths_sha256": paths_manifest["ordered_paths_sha256"],
                "read_source_file_sha256": dict(sorted(read_source_file_hashes.items())),
            },
            "counts": {
                "unique_source_documents": len(seen),
                "raw_source_documents_read": raw_source_documents_read,
                "unique_source_utf8_bytes": source_utf8_bytes,
                "duplicate_source_documents_skipped": duplicate_source_documents,
                "unique_train_documents": len(train_ids),
                "unique_validation_documents": len(validation_ids),
                "unique_validation_candidate_documents": len(validation_ids),
                "declared_train_documents_with_rehydration": logical_train_records,
                "emitted_validation_documents": len(validation_ids),
                "calibration_documents": len(calibration_ids),
                "train_validation_overlap": len(overlap),
            },
            "targets": {
                "requested": availability_policy["requested_settings"],
                "effective_after_inventory_cap": availability_policy[
                    "effective_settings"
                ],
                "inventory_limited": availability_policy["inventory_limited"],
                "source_exhausted": source_exhausted,
                "pinned_inventory_exhausted": pinned_inventory_exhausted,
                "unique_document_shortfall": max(
                    0,
                    settings.target_unique_documents - len(seen),
                ),
                "utf8_byte_shortfall": max(
                    0,
                    settings.target_utf8_bytes - source_utf8_bytes,
                ),
                "calibration_requirement": min(
                    requested_settings.calibration_documents,
                    len(train_ids),
                ),
            },
            "hashes": {
                "unique_source_ids_sha256": canonical_json_sha256(sorted(seen)),
                "unique_train_ids_sha256": canonical_json_sha256(sorted(train_ids)),
                "unique_validation_ids_sha256": canonical_json_sha256(sorted(validation_ids)),
                "unique_validation_candidate_ids_sha256": canonical_json_sha256(
                    sorted(validation_ids)
                ),
                "calibration_ids_sha256": canonical_json_sha256(sorted(calibration_ids)),
            },
            "rehydration": {
                "enabled": rehydration_policy_source == "pinned_upstream_weights",
                "identity_override": (
                    rehydration_policy_source
                    == "source_lock_explicit_identity_override"
                ),
                "policy_source": rehydration_policy_source,
                "thresholds_sha256": canonical_json_sha256(thresholds),
                "replica_count_histogram": {
                    str(key): value for key, value in sorted(replica_histogram.items())
                },
                "validation_candidate_train_replica_count_histogram": {
                    str(key): value
                    for key, value in sorted(
                        validation_candidate_train_replica_histogram.items()
                    )
                },
                "cluster_size_histogram": {
                    str(key): value for key, value in sorted(cluster_histogram.items())
                },
            },
            "outputs": output_lineage,
        }
        if tokenizer_sentence_slot_audit is not None:
            report["counts"]["unique_train_sentences"] = train_unique_sentence_count
            report["counts"][
                "rehydrated_train_sentence_slots"
            ] = train_rehydrated_sentence_slots
            report["targets"][
                "tokenizer_sentence_slot_gate"
            ] = tokenizer_sentence_slot_audit
        report["report_payload_sha256"] = canonical_json_sha256(report)
        _atomic_write_json(language_dir / "bootstrap_report.json", report)
        return report
    finally:
        for handle in handles.values():
            handle.close()
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()
