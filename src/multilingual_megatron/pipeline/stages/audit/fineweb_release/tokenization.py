"""Megatron prefix and tokenization-manifest audits for FineWeb-v2."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.audit.fineweb_release.constants import (
    EXPECTED_MEGATRON_DTYPE_CODE,
    _DTYPE_SIZES,
    _INDEX_HEADER,
)
from multilingual_megatron.pipeline.stages.audit.fineweb_release.integrity import (
    _assert_fresh_path,
    _cached_sha256,
    _load_object,
    _require,
    _require_sha256,
    _resolve_file,
    _verify_payload,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import sha256_file


def _audit_index(prefix: Path) -> dict[str, int]:
    idx_path = Path(str(prefix) + ".idx")
    bin_path = Path(str(prefix) + ".bin")
    _require(
        idx_path.is_file() and bin_path.is_file(), f"missing Megatron prefix: {prefix}"
    )
    with idx_path.open("rb") as handle:
        _require(
            handle.read(9) == _INDEX_HEADER,
            f"invalid Megatron index header: {idx_path}",
        )
        version_raw, code_raw, counts_raw = (
            handle.read(8),
            handle.read(1),
            handle.read(16),
        )
        _require(
            len(version_raw) == 8 and len(code_raw) == 1 and len(counts_raw) == 16,
            f"truncated Megatron index: {idx_path}",
        )
        version = struct.unpack("<Q", version_raw)[0]
        dtype_code = struct.unpack("<B", code_raw)[0]
        sequences, document_index_count = struct.unpack("<QQ", counts_raw)
        _require(
            version == 1 and dtype_code in _DTYPE_SIZES and document_index_count >= 1,
            f"unsupported Megatron index metadata: {idx_path}",
        )
        remaining = int(sequences)
        tokens = 0
        while remaining:
            count = min(remaining, 1_000_000)
            raw = handle.read(count * 4)
            _require(
                len(raw) == count * 4,
                f"truncated Megatron sequence lengths: {idx_path}",
            )
            tokens += sum(value[0] for value in struct.iter_unpack("<i", raw))
            remaining -= count
    _require(
        bin_path.stat().st_size == tokens * _DTYPE_SIZES[dtype_code],
        f"Megatron bin size/count mismatch: {prefix}",
    )
    return {
        "sequences": int(sequences),
        "documents": int(document_index_count - 1),
        "tokens_including_eod": int(tokens),
        "dtype_code": int(dtype_code),
    }

def _verify_prefix(
    raw: Mapping[str, Any], *, hash_cache: dict[Path, str], label: str
) -> dict[str, int]:
    prefix = Path(str(raw.get("prefix", ""))).resolve()
    audit = _audit_index(prefix)
    for field, value in audit.items():
        _require(raw.get(field) == value, f"{label}.{field} mismatch")
    _require(
        audit["dtype_code"] == EXPECTED_MEGATRON_DTYPE_CODE,
        f"{label} dtype is not uint16 for the 64K tokenizer",
    )
    _require(
        audit["sequences"] == audit["documents"],
        f"{label} is not one sequence per document",
    )
    for suffix in ("bin", "idx"):
        path = Path(str(prefix) + f".{suffix}")
        _require(
            raw.get(f"{suffix}_size_bytes") == path.stat().st_size,
            f"{label}.{suffix} size mismatch",
        )
        expected = _require_sha256(
            raw.get(f"{suffix}_sha256"), field=f"{label}.{suffix}_sha256"
        )
        _require(
            _cached_sha256(path, hash_cache) == expected,
            f"{label}.{suffix} hash mismatch",
        )
    return audit


def _manifest_payload(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload = _load_object(path, label=label)
    return payload, _verify_payload(payload, key="manifest_payload_sha256", label=label)


def _audit_tokenization(
    *,
    tokenized_root: Path,
    release_root: Path,
    language_sets: Mapping[str, Sequence[str]],
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
    tokenizer_hash: str,
    hash_cache: dict[Path, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = tokenized_root / "tokenization_report.json"
    report = _load_object(report_path, label="tokenization report")
    report_hash = _verify_payload(
        report, key="report_payload_sha256", label="tokenization report"
    )
    _require(
        report.get("schema") == "fineweb_ab_v2_presplit_tokenization_merge/v1"
        and report.get("pass") is True,
        "tokenization merge did not pass",
    )
    _require(
        report.get("language_count") == len(language_sets["K200"])
        and report.get("item_count") == 2 * len(language_sets["K200"]),
        "tokenization coverage count mismatch",
    )
    contract = report.get("scientific_contract")
    _require(
        isinstance(contract, dict)
        and contract.get("random_resplit") is False
        and contract.get("physical_K200_pool_tokenized_once") is True
        and contract.get("tier_mixtures_from_sampling_plan_probabilities") is True
        and contract.get("validation_is_frozen_and_unrehydrated") is True,
        "tokenization scientific contract mismatch",
    )
    exact = report.get("exact_counts")
    _require(
        isinstance(exact, dict) and set(exact) == {"train", "validation"},
        "tokenization exact-count table missing",
    )
    for split in ("train", "validation"):
        for field in ("documents_per_language", "tokens_per_language"):
            values = exact[split].get(field)
            _require(
                isinstance(values, dict) and set(values) == set(language_sets["K200"]),
                f"tokenization exact-count coverage mismatch: {split}/{field}",
            )

    prefix_cache: dict[str, dict[str, int]] = {}
    tier_audit: dict[str, Any] = {}
    embedded_validation_payload: dict[str, Any] | None = None
    for tier, languages in language_sets.items():
        manifest_path = tokenized_root / "tiers" / tier / "megatron_data_manifest.json"
        manifest, payload_hash = _manifest_payload(
            manifest_path, label=f"tokenized {tier} manifest"
        )
        descriptor = report.get("tiers", {}).get(tier)
        _require(
            isinstance(descriptor, dict)
            and descriptor.get("sha256") == sha256_file(manifest_path)
            and descriptor.get("payload_sha256") == payload_hash,
            f"tokenization report/{tier} manifest identity mismatch",
        )
        _require(
            manifest.get("schema") == "fineweb_ab_v2_megatron_data/v1"
            and manifest.get("tier") == tier,
            f"tokenized {tier} manifest identity mismatch",
        )
        _require(
            manifest.get("languages") == list(languages),
            f"tokenized {tier} language membership mismatch",
        )
        _require(
            manifest.get("tokenizer_model_sha256") == tokenizer_hash,
            f"tokenized {tier} tokenizer mismatch",
        )
        lineage = manifest.get("lineage")
        _require(
            isinstance(lineage, dict)
            and lineage.get("corpus_manifest_payload_sha256")
            == corpus.get("manifest_sha256")
            and lineage.get("sampling_plan_payload_sha256") == plan.get("plan_sha256")
            and lineage.get("tokenizer_model_sha256") == tokenizer_hash
            and lineage.get("source_manifest_payload_sha256")
            == plan.get("provenance", {}).get("source_manifest_payload_sha256"),
            f"tokenized {tier} core lineage mismatch",
        )
        raw_plan = plan["tiers"][tier]
        for field in (
            "alpha",
            "target_train_tokens",
            "target_exposure_tokens",
            "language_probabilities",
        ):
            expected = (
                raw_plan["probabilities"]
                if field == "language_probabilities"
                else raw_plan[field]
            )
            _require(
                manifest.get(field) == expected,
                f"tokenized {tier} differs from sampling plan: {field}",
            )
        train = manifest.get("splits", {}).get("train")
        _require(isinstance(train, dict), f"tokenized {tier} lacks train split")
        _require(
            train.get("documents_per_language")
            == {
                language: exact["train"]["documents_per_language"][language]
                for language in languages
            },
            f"tokenized {tier} train document counts mismatch",
        )
        _require(
            train.get("tokens_per_language")
            == {
                language: exact["train"]["tokens_per_language"][language]
                for language in languages
            },
            f"tokenized {tier} train token counts mismatch",
        )
        artifacts_by_language = train.get("prefix_artifacts_by_language")
        _require(
            isinstance(artifacts_by_language, dict)
            and set(artifacts_by_language) == set(languages),
            f"tokenized {tier} prefix coverage mismatch",
        )
        expected_prefixes: list[str] = []
        expected_weights: list[float] = []
        expected_prefixes_by_language: dict[str, list[str]] = {}
        for language in languages:
            rows = artifacts_by_language[language]
            _require(
                isinstance(rows, list) and bool(rows),
                f"tokenized prefix list is empty: {tier}/{language}",
            )
            language_prefixes = [str(row.get("prefix")) for row in rows]
            expected_prefixes.extend(language_prefixes)
            expected_prefixes_by_language[language] = language_prefixes
            expected_weights.extend(
                [float(raw_plan["probabilities"][language]) / len(rows)] * len(rows)
            )
            observed_documents = observed_tokens = 0
            for index, raw in enumerate(rows):
                _require(
                    isinstance(raw, dict),
                    f"invalid prefix artifact: {tier}/{language}/{index}",
                )
                key = str(raw.get("prefix"))
                _assert_fresh_path(
                    Path(key),
                    release_root=release_root,
                    label=f"{tier}/{language} Megatron prefix",
                )
                audit = _verify_prefix(
                    raw,
                    hash_cache=hash_cache,
                    label=f"{tier}/{language}/{index}",
                )
                previous = prefix_cache.setdefault(key, audit)
                _require(
                    previous == audit,
                    f"duplicate prefix metadata drift: {tier}/{language}",
                )
                observed_documents += audit["documents"]
                observed_tokens += audit["tokens_including_eod"]
            _require(
                observed_documents
                == exact["train"]["documents_per_language"][language],
                f"tokenized train document count mismatch: {tier}/{language}",
            )
            _require(
                observed_tokens == exact["train"]["tokens_per_language"][language],
                f"tokenized train token count mismatch: {tier}/{language}",
            )
        _require(
            train.get("prefixes") == expected_prefixes,
            f"tokenized {tier} flattened prefix order mismatch",
        )
        _require(
            train.get("prefixes_by_language") == expected_prefixes_by_language,
            f"tokenized {tier} per-language prefix map mismatch",
        )
        observed_weights = train.get("weights")
        _require(
            isinstance(observed_weights, list)
            and len(observed_weights) == len(expected_weights)
            and all(
                math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-18)
                for observed, expected in zip(observed_weights, expected_weights)
            )
            and math.isclose(
                math.fsum(float(value) for value in observed_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"tokenized {tier} physical-prefix blend weights mismatch",
        )
        for field in (
            "prefixes",
            "weights",
            "prefixes_by_language",
            "documents_per_language",
            "tokens_per_language",
        ):
            _require(
                manifest.get(field) == train.get(field),
                f"tokenized {tier} top-level training field differs from split: {field}",
            )
        tier_validation = manifest.get("splits", {}).get("validation")
        _require(
            isinstance(tier_validation, dict),
            f"tokenized {tier} lacks embedded validation",
        )
        if embedded_validation_payload is None:
            embedded_validation_payload = tier_validation
        else:
            _require(
                tier_validation == embedded_validation_payload,
                f"tokenized {tier} uses a different validation panel",
            )
        tier_audit[tier] = {
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": payload_hash,
            "language_count": len(languages),
            "exact_exposure_tokens": sum(
                int(value) for value in manifest["target_exposure_tokens"].values()
            ),
        }

    validation_path = tokenized_root / "validation_megatron_data_manifest.json"
    validation, validation_hash = _manifest_payload(
        validation_path, label="global validation manifest"
    )
    descriptor = report.get("global_validation_manifest")
    _require(
        isinstance(descriptor, dict)
        and descriptor.get("sha256") == sha256_file(validation_path)
        and descriptor.get("payload_sha256") == validation_hash,
        "global validation/report identity mismatch",
    )
    _require(
        validation.get("purpose") == "global_K200_per_language_validation"
        and validation.get("languages") == list(language_sets["K200"]),
        "global validation language panel mismatch",
    )
    _require(
        validation.get("tokenizer_model_sha256") == tokenizer_hash,
        "global validation tokenizer mismatch",
    )
    split = validation.get("splits", {}).get("validation")
    _require(isinstance(split, dict), "global validation split missing")
    _require(
        split == embedded_validation_payload,
        "global and tier-embedded validation panels differ",
    )
    _require(
        split.get("documents_per_language")
        == exact["validation"]["documents_per_language"],
        "global validation document counts mismatch",
    )
    _require(
        split.get("tokens_per_language") == exact["validation"]["tokens_per_language"],
        "global validation token counts mismatch",
    )
    artifacts_by_language = split.get("prefix_artifacts_by_language")
    _require(
        isinstance(artifacts_by_language, dict)
        and set(artifacts_by_language) == set(language_sets["K200"]),
        "global validation prefix coverage mismatch",
    )
    expected_validation_prefixes: list[str] = []
    expected_validation_weights: list[float] = []
    expected_validation_by_language: dict[str, list[str]] = {}
    for language in language_sets["K200"]:
        rows = artifacts_by_language[language]
        _require(
            isinstance(rows, list) and bool(rows),
            f"validation prefix list is empty: {language}",
        )
        language_prefixes = [str(row.get("prefix")) for row in rows]
        expected_validation_prefixes.extend(language_prefixes)
        expected_validation_by_language[language] = language_prefixes
        expected_validation_weights.extend(
            [1.0 / len(language_sets["K200"]) / len(rows)] * len(rows)
        )
        observed_documents = observed_tokens = 0
        for index, raw in enumerate(rows):
            _require(
                isinstance(raw, dict), f"invalid validation prefix: {language}/{index}"
            )
            key = str(raw.get("prefix"))
            _assert_fresh_path(
                Path(key),
                release_root=release_root,
                label=f"validation/{language} Megatron prefix",
            )
            audit = _verify_prefix(
                raw,
                hash_cache=hash_cache,
                label=f"validation/{language}/{index}",
            )
            previous = prefix_cache.setdefault(key, audit)
            _require(
                previous == audit,
                f"duplicate validation prefix metadata drift: {language}",
            )
            observed_documents += audit["documents"]
            observed_tokens += audit["tokens_including_eod"]
        _require(
            observed_documents
            == exact["validation"]["documents_per_language"][language],
            f"tokenized validation document count mismatch: {language}",
        )
        _require(
            observed_tokens == exact["validation"]["tokens_per_language"][language],
            f"tokenized validation token count mismatch: {language}",
        )
    _require(
        split.get("prefixes") == expected_validation_prefixes,
        "global validation flattened prefix order mismatch",
    )
    _require(
        split.get("prefixes_by_language") == expected_validation_by_language,
        "global validation per-language prefix map mismatch",
    )
    validation_weights = split.get("weights")
    _require(
        isinstance(validation_weights, list)
        and len(validation_weights) == len(expected_validation_weights)
        and all(
            math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-18)
            for observed, expected in zip(
                validation_weights, expected_validation_weights
            )
        )
        and math.isclose(
            math.fsum(float(value) for value in validation_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "global validation equal-language prefix weights mismatch",
    )

    for language in language_sets["K200"]:
        corpus_language = corpus["languages"][language]
        _require(
            exact["train"]["documents_per_language"][language]
            == corpus_language["train"]["emitted_documents"],
            f"corpus/tokenized train document mismatch: {language}",
        )
        _require(
            exact["train"]["tokens_per_language"][language]
            == corpus_language["train"]["selected_tokens"],
            f"corpus/tokenized train token mismatch: {language}",
        )
        _require(
            exact["validation"]["documents_per_language"][language]
            == corpus_language["validation"]["emitted_documents"],
            f"corpus/tokenized validation document mismatch: {language}",
        )
        _require(
            exact["validation"]["tokens_per_language"][language]
            == corpus_language["validation"]["selected_tokens"],
            f"corpus/tokenized validation token mismatch: {language}",
        )

    maximum = tokenized_root / "megatron_data_manifest.json"
    _require(
        sha256_file(maximum)
        == sha256_file(
            tokenized_root / "tiers" / "K200" / "megatron_data_manifest.json"
        ),
        "default tokenized manifest is not exact K200",
    )
    for raw in report.get("rank_results", []):
        _require(isinstance(raw, dict), "invalid tokenization rank-result descriptor")
        rank_path = _resolve_file(
            tokenized_root, raw.get("path"), label="tokenization rank result"
        )
        _assert_fresh_path(
            rank_path, release_root=release_root, label="tokenization rank result"
        )
        _require(
            sha256_file(rank_path) == raw.get("sha256"),
            f"tokenization rank result changed: {rank_path}",
        )
        rank_payload = _load_object(rank_path, label="tokenization rank result")
        _require(
            _verify_payload(
                rank_payload,
                key="result_payload_sha256",
                label="tokenization rank result",
            )
            == raw.get("payload_sha256"),
            f"tokenization rank result payload mismatch: {rank_path}",
        )
    return report, {
        "report_file_sha256": sha256_file(report_path),
        "report_payload_sha256": report_hash,
        "tiers": tier_audit,
        "global_validation_payload_sha256": validation_hash,
        "unique_prefix_count": len(prefix_cache),
        "dtype_code": EXPECTED_MEGATRON_DTYPE_CODE,
        "all_prefix_hashes_and_counts_verified": True,
    }
