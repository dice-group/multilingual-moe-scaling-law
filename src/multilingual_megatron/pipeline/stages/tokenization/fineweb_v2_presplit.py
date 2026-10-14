#!/usr/bin/env python3
"""Tokenize the frozen FineWeb A/B-v2 train and validation pools exactly once.

The generic tokenization pipeline can create a train/validation split from a
directory of JSONL files.  That behaviour is intentionally *not* used here:
the A/B-v2 corpus finalizer has already made and audited the split.  This tool
therefore accepts only the finalizer's immutable ``corpus_manifest.json`` and
performs three explicit operations:

``plan``
    Verify the ready corpus, tokenizer, sampling plan, and every source JSONL,
    then assign the 400 (language, split) inputs to deterministic array ranks.
``worker``
    Tokenize the assigned source files without sampling or splitting and audit
    the emitted Megatron ``bin/idx`` prefixes against the exact document and
    token counts frozen in the corpus manifest.
``merge``
    Verify every worker artifact and write one training manifest per nested K
    tier plus a single K200 validation manifest used for per-language loss.

Only physical prefixes are written.  K-specific exposure is represented by
the alpha-smoothed weights from the sampling plan; when preprocessing emits
multiple partitions, one language's probability is divided equally among its
partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
    verify_plan_hash,
)


SCHEMA_VERSION = 1
PLAN_SCHEMA = "fineweb_ab_v2_presplit_tokenization_plan/v1"
RANK_PLAN_SCHEMA = "fineweb_ab_v2_presplit_tokenization_rank/v1"
RANK_RESULT_SCHEMA = "fineweb_ab_v2_presplit_tokenization_result/v1"
MANIFEST_SCHEMA = "fineweb_ab_v2_megatron_data/v1"
_SPLITS = ("train", "validation")
_DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}
_INDEX_HEADER = b"MMIDIDX\x00\x00"


class PresplitTokenizationError(ValueError):
    """Raised when a frozen input or emitted prefix violates the contract."""


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PresplitTokenizationError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PresplitTokenizationError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PresplitTokenizationError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise PresplitTokenizationError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _resolve_file(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PresplitTokenizationError(f"{field} must be a non-empty path")
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_file():
        raise PresplitTokenizationError(f"missing {field}: {path}")
    return path


def _verified_file(
    base: Path,
    artifact: Any,
    *,
    field: str,
    expected_suffix: str | None = None,
) -> tuple[Path, str, int]:
    if not isinstance(artifact, dict):
        raise PresplitTokenizationError(f"{field} must be an artifact object")
    path = _resolve_file(base, artifact.get("path"), field=f"{field}.path")
    if expected_suffix is not None and path.suffix != expected_suffix:
        raise PresplitTokenizationError(f"{field}.path must end in {expected_suffix}: {path}")
    expected_hash = _require_sha256(artifact.get("sha256"), field=f"{field}.sha256")
    expected_size = artifact.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise PresplitTokenizationError(f"{field}.size_bytes must be a non-negative integer")
    if path.stat().st_size != expected_size:
        raise PresplitTokenizationError(
            f"{field} size mismatch: expected {expected_size}, observed {path.stat().st_size}"
        )
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise PresplitTokenizationError(
            f"{field} SHA-256 mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    return path, expected_hash, expected_size


def _verify_payload_hash(payload: Mapping[str, Any], *, key: str, label: str) -> str:
    expected = _require_sha256(payload.get(key), field=f"{label}.{key}")
    candidate = dict(payload)
    candidate.pop(key, None)
    observed = canonical_json_sha256(candidate)
    if observed != expected:
        raise PresplitTokenizationError(
            f"{label} payload SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return expected


def _tier_sort_key(name: str) -> int:
    if not name.startswith("K") or not name[1:].isdigit():
        raise PresplitTokenizationError(f"invalid sampling-plan tier name: {name!r}")
    return int(name[1:])


def _parse_tiers(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_tiers = plan.get("tiers")
    if not isinstance(raw_tiers, dict) or not raw_tiers:
        raise PresplitTokenizationError("sampling plan must contain non-empty tiers")
    tiers: dict[str, dict[str, Any]] = {}
    previous: set[str] = set()
    for tier_name in sorted(map(str, raw_tiers), key=_tier_sort_key):
        raw = raw_tiers[tier_name]
        if not isinstance(raw, dict):
            raise PresplitTokenizationError(f"sampling plan {tier_name} must be an object")
        languages_raw = raw.get("languages")
        probabilities_raw = raw.get("probabilities")
        exposure_raw = raw.get("target_exposure_tokens")
        if not isinstance(languages_raw, list) or not languages_raw:
            raise PresplitTokenizationError(f"sampling plan {tier_name}.languages must be non-empty")
        languages = [str(value) for value in languages_raw]
        if len(languages) != _tier_sort_key(tier_name) or len(languages) != len(set(languages)):
            raise PresplitTokenizationError(
                f"sampling plan {tier_name} language cardinality or uniqueness is invalid"
            )
        if previous and not previous.issubset(languages):
            raise PresplitTokenizationError(f"sampling plan tiers are not nested at {tier_name}")
        previous = set(languages)
        if not isinstance(probabilities_raw, dict) or set(map(str, probabilities_raw)) != set(languages):
            raise PresplitTokenizationError(f"sampling plan {tier_name}.probabilities coverage mismatch")
        probabilities: dict[str, float] = {}
        for language in languages:
            value = probabilities_raw.get(language)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise PresplitTokenizationError(
                    f"sampling plan {tier_name}.probabilities.{language} must be finite and positive"
                )
            probabilities[language] = float(value)
        if not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise PresplitTokenizationError(f"sampling plan {tier_name} probabilities do not sum to one")
        if not isinstance(exposure_raw, dict) or set(map(str, exposure_raw)) != set(languages):
            raise PresplitTokenizationError(
                f"sampling plan {tier_name}.target_exposure_tokens coverage mismatch"
            )
        exposures: dict[str, int] = {}
        for language in languages:
            value = exposure_raw.get(language)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise PresplitTokenizationError(
                    f"sampling plan {tier_name}.target_exposure_tokens.{language} must be non-negative"
                )
            exposures[language] = value
        tiers[tier_name] = {
            "languages": languages,
            "probabilities": probabilities,
            "target_exposure_tokens": exposures,
            "target_train_tokens": int(raw.get("target_train_tokens", sum(exposures.values()))),
            "alpha": float(raw.get("alpha", plan.get("alpha", 0.0))),
        }
    return tiers


def _source_items(
    *,
    corpus_path: Path,
    corpus: Mapping[str, Any],
    expected_languages: set[str],
    verification_workers: int = 1,
) -> list[dict[str, Any]]:
    raw_languages = corpus.get("languages")
    if not isinstance(raw_languages, dict) or set(map(str, raw_languages)) != expected_languages:
        raise PresplitTokenizationError("corpus and maximum-tier language sets differ")
    if not 1 <= verification_workers <= 64:
        raise PresplitTokenizationError(
            "verification_workers must be between 1 and 64"
        )

    def verify_item(pair: tuple[str, str]) -> dict[str, Any]:
        language, split = pair
        raw_language = raw_languages.get(language)
        if not isinstance(raw_language, dict):
            raise PresplitTokenizationError(f"corpus languages.{language} must be an object")
        raw_split = raw_language.get(split)
        if not isinstance(raw_split, dict) or raw_split.get("status") != "ready":
            raise PresplitTokenizationError(f"corpus {language}/{split} is not ready")
        source, source_hash, source_size = _verified_file(
            corpus_path.parent,
            raw_split.get("output"),
            field=f"corpus languages.{language}.{split}.output",
            expected_suffix=".jsonl",
        )
        documents = raw_split.get("emitted_documents")
        tokens = raw_split.get("selected_tokens")
        if not isinstance(documents, int) or isinstance(documents, bool) or documents <= 0:
            raise PresplitTokenizationError(f"corpus {language}/{split} emitted_documents is invalid")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise PresplitTokenizationError(f"corpus {language}/{split} selected_tokens is invalid")
        return {
            "language": language,
            "split": split,
            "source_path": str(source),
            "source_sha256": source_hash,
            "source_size_bytes": source_size,
            "expected_documents": documents,
            "expected_tokens_including_eod": tokens,
        }

    pairs = [
        (language, split)
        for language in sorted(expected_languages)
        for split in _SPLITS
    ]
    if verification_workers == 1:
        return [verify_item(pair) for pair in pairs]
    with ThreadPoolExecutor(max_workers=verification_workers) as executor:
        return list(executor.map(verify_item, pairs))


def _assign_items(items: Sequence[dict[str, Any]], ranks: int) -> list[list[dict[str, Any]]]:
    if ranks <= 0:
        raise PresplitTokenizationError("array_ranks must be positive")
    if ranks > len(items):
        raise PresplitTokenizationError(
            f"array_ranks ({ranks}) exceeds tokenization work items ({len(items)})"
        )
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
    loads = [0] * ranks
    for item in sorted(
        items,
        key=lambda row: (
            -int(row["expected_tokens_including_eod"]),
            str(row["language"]),
            str(row["split"]),
        ),
    ):
        rank = min(range(ranks), key=lambda index: (loads[index], index))
        assigned[rank].append(item)
        loads[rank] += int(item["expected_tokens_including_eod"])
    return assigned



def _resolve_output_tokenizer(
    *,
    selection_tokenizer_model_path: Path,
    selection_tokenizer_model_sha256: str,
) -> dict[str, Any]:
    """Resolve a production tokenizer while retaining corpus-selection lineage."""

    return {
        "tokenizer_type": "SentencePieceTokenizer",
        "tokenizer_model_path": str(selection_tokenizer_model_path),
        "tokenizer_model_file_path": str(selection_tokenizer_model_path),
        "tokenizer_model_sha256": selection_tokenizer_model_sha256,
        "token_count_policy": "match_selection_tokenizer",
    }


def build_tokenization_plan(
    *,
    corpus_manifest_path: Path,
    sampling_plan_path: Path,
    tokenizer_model_path: Path,
    plan_dir: Path,
    array_ranks: int,
    verification_workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """Verify all immutable inputs and write deterministic Slurm-array plans."""

    corpus_manifest_path = corpus_manifest_path.resolve()
    sampling_plan_path = sampling_plan_path.resolve()
    tokenizer_model_path = tokenizer_model_path.resolve()
    plan_dir = plan_dir.resolve()
    corpus = _load_object(corpus_manifest_path, label="corpus manifest")
    corpus_payload_hash = _verify_payload_hash(
        corpus, key="manifest_sha256", label="corpus manifest"
    )
    if corpus.get("ready") is not True or corpus.get("release_status") != "ready":
        raise PresplitTokenizationError("corpus manifest is not a ready release")

    sampling_plan = _load_object(sampling_plan_path, label="sampling plan")
    if not verify_plan_hash(sampling_plan):
        raise PresplitTokenizationError("sampling plan payload SHA-256 mismatch")
    sampling_plan_hash = _require_sha256(
        sampling_plan.get("plan_sha256"), field="sampling plan.plan_sha256"
    )
    tiers = _parse_tiers(sampling_plan)
    maximum_tier = max(tiers, key=_tier_sort_key)
    expected_languages = set(tiers[maximum_tier]["languages"])
    physical_pool = sampling_plan.get("physical_pool")
    if not isinstance(physical_pool, dict) or set(map(str, physical_pool)) != expected_languages:
        raise PresplitTokenizationError("sampling-plan physical pool differs from maximum K tier")

    corpus_plan = corpus.get("inputs", {}).get("sampling_plan")
    if not isinstance(corpus_plan, dict) or corpus_plan.get("payload_sha256") != sampling_plan_hash:
        raise PresplitTokenizationError("corpus manifest was not finalized from this sampling plan")
    if sha256_file(sampling_plan_path) != corpus_plan.get("file_sha256"):
        raise PresplitTokenizationError("sampling-plan file identity differs from the corpus manifest")

    corpus_tokenizer = corpus.get("inputs", {}).get("tokenizer_model")
    if not isinstance(corpus_tokenizer, dict):
        raise PresplitTokenizationError("corpus manifest lacks tokenizer identity")
    tokenizer_hash = _require_sha256(
        corpus_tokenizer.get("sha256"), field="corpus tokenizer_model.sha256"
    )
    if not tokenizer_model_path.is_file() or sha256_file(tokenizer_model_path) != tokenizer_hash:
        raise PresplitTokenizationError("tokenizer model identity differs from the corpus manifest")
    plan_tokenizer_hash = sampling_plan.get("provenance", {}).get("tokenizer_model_sha256")
    if plan_tokenizer_hash != tokenizer_hash:
        raise PresplitTokenizationError("sampling plan and corpus use different tokenizer models")
    output_tokenizer = _resolve_output_tokenizer(
        selection_tokenizer_model_path=tokenizer_model_path,
        selection_tokenizer_model_sha256=tokenizer_hash,
    )

    items = _source_items(
        corpus_path=corpus_manifest_path,
        corpus=corpus,
        expected_languages=expected_languages,
        verification_workers=int(verification_workers),
    )
    assigned = _assign_items(items, int(array_ranks))
    sampling_provenance = sampling_plan.get("provenance")
    if not isinstance(sampling_provenance, dict):
        raise PresplitTokenizationError("sampling plan lacks provenance")
    source_manifest_hash = _require_sha256(
        sampling_provenance.get("source_manifest_payload_sha256"),
        field="sampling plan provenance.source_manifest_payload_sha256",
    )
    core_lineage = {
        "corpus_manifest_path": str(corpus_manifest_path),
        "corpus_manifest_file_sha256": sha256_file(corpus_manifest_path),
        "corpus_manifest_payload_sha256": corpus_payload_hash,
        "sampling_plan_path": str(sampling_plan_path),
        "sampling_plan_file_sha256": sha256_file(sampling_plan_path),
        "sampling_plan_payload_sha256": sampling_plan_hash,
        "selection_tokenizer_model_path": str(tokenizer_model_path),
        "selection_tokenizer_model_sha256": tokenizer_hash,
        **output_tokenizer,
        "source_manifest_payload_sha256": source_manifest_hash,
        "sampling_plan_provenance": sampling_provenance,
    }
    core_lineage_sha256 = canonical_json_sha256(core_lineage)

    if plan_dir.exists() and any(plan_dir.iterdir()):
        if not force:
            raise PresplitTokenizationError(f"plan directory is not empty: {plan_dir}")
        shutil.rmtree(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    rank_files: dict[str, dict[str, Any]] = {}
    for rank, rank_items in enumerate(assigned):
        rank_payload = {
            "schema": RANK_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "rank": rank,
            "array_ranks": int(array_ranks),
            "core_lineage_sha256": core_lineage_sha256,
            "items": sorted(rank_items, key=lambda row: (row["split"], row["language"])),
        }
        rank_payload["rank_plan_payload_sha256"] = canonical_json_sha256(rank_payload)
        rank_path = plan_dir / f"rank_{rank:05d}.json"
        _write_json(rank_path, rank_payload)
        rank_files[str(rank)] = {
            "path": rank_path.name,
            "sha256": sha256_file(rank_path),
            "payload_sha256": rank_payload["rank_plan_payload_sha256"],
            "item_count": len(rank_items),
            "expected_tokens_including_eod": sum(
                int(item["expected_tokens_including_eod"]) for item in rank_items
            ),
        }
    summary: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "array_ranks": int(array_ranks),
        "item_count": len(items),
        "language_count": len(expected_languages),
        "splits": list(_SPLITS),
        "maximum_tier": maximum_tier,
        "languages": sorted(expected_languages),
        "tiers": tiers,
        "core_lineage": core_lineage,
        "core_lineage_sha256": core_lineage_sha256,
        "rank_files": rank_files,
        "contract": {
            "input_unit": "finalized_whole_document_jsonl",
            "split_policy": "already_frozen_by_corpus_finalizer_no_resplit",
            "physical_pool": "maximum_K_pool_tokenized_once",
            "append_eod": True,
            "token_count": "sum_of_megatron_index_sequence_lengths",
            "token_count_policy": output_tokenizer["token_count_policy"],
            "input_sha256_verification_workers": int(verification_workers),
        },
    }
    summary["plan_payload_sha256"] = canonical_json_sha256(summary)
    _write_json(plan_dir / "summary.json", summary)
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _audit_index(prefix: Path) -> dict[str, int]:
    idx_path = Path(str(prefix) + ".idx")
    bin_path = Path(str(prefix) + ".bin")
    if not idx_path.is_file() or not bin_path.is_file():
        raise PresplitTokenizationError(f"missing Megatron bin/idx for prefix: {prefix}")
    with idx_path.open("rb") as handle:
        if handle.read(9) != _INDEX_HEADER:
            raise PresplitTokenizationError(f"invalid Megatron index header: {idx_path}")
        version_raw = handle.read(8)
        code_raw = handle.read(1)
        counts_raw = handle.read(16)
        if len(version_raw) != 8 or len(code_raw) != 1 or len(counts_raw) != 16:
            raise PresplitTokenizationError(f"truncated Megatron index header: {idx_path}")
        version = struct.unpack("<Q", version_raw)[0]
        dtype_code = struct.unpack("<B", code_raw)[0]
        sequence_count, document_index_count = struct.unpack("<QQ", counts_raw)
        if version != 1 or dtype_code not in _DTYPE_SIZES or document_index_count < 1:
            raise PresplitTokenizationError(f"unsupported Megatron index metadata: {idx_path}")
        remaining = int(sequence_count)
        token_count = 0
        while remaining:
            count = min(remaining, 1_000_000)
            raw = handle.read(count * 4)
            if len(raw) != count * 4:
                raise PresplitTokenizationError(f"truncated sequence lengths: {idx_path}")
            token_count += sum(value[0] for value in struct.iter_unpack("<i", raw))
            remaining -= count
    expected_bin_size = token_count * _DTYPE_SIZES[dtype_code]
    if bin_path.stat().st_size != expected_bin_size:
        raise PresplitTokenizationError(
            f"Megatron bin size mismatch for {prefix}: expected {expected_bin_size}, "
            f"observed {bin_path.stat().st_size}"
        )
    return {
        "sequences": int(sequence_count),
        "documents": int(document_index_count - 1),
        "tokens_including_eod": int(token_count),
        "dtype_code": int(dtype_code),
    }


def _artifact_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _artifact(
    prefix: Path,
    *,
    cache: dict[
        tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    bin_path = Path(str(prefix) + ".bin")
    idx_path = Path(str(prefix) + ".idx")
    cache_key = (_artifact_identity(bin_path), _artifact_identity(idx_path))
    if cache is not None and cache_key in cache:
        return {"prefix": str(prefix.resolve()), **cache[cache_key]}
    audit = _audit_index(prefix)
    fields = {
        "bin_sha256": sha256_file(bin_path),
        "bin_size_bytes": bin_path.stat().st_size,
        "idx_sha256": sha256_file(idx_path),
        "idx_size_bytes": idx_path.stat().st_size,
        **audit,
    }
    if cache is not None:
        cache[cache_key] = fields
    return {"prefix": str(prefix.resolve()), **fields}


PreprocessRunner = Callable[[dict[str, Any], Path, dict[str, Any]], Sequence[Path]]


def _default_preprocess_runner(
    item: dict[str, Any],
    output_prefix: Path,
    settings: dict[str, Any],
) -> Sequence[Path]:
    bridge_root = Path(settings["bridge_root"])
    preprocess = bridge_root / "3rdparty" / "Megatron-LM" / "tools" / "preprocess_data.py"
    if not preprocess.is_file():
        raise PresplitTokenizationError(f"missing Megatron preprocess_data.py: {preprocess}")
    partitions = int(settings["partitions_per_language"])
    workers = int(settings["workers"])
    if partitions <= 0 or workers <= 0 or workers % partitions:
        raise PresplitTokenizationError("workers must be positive and divisible by partitions")
    if int(item["expected_documents"]) < partitions:
        # Avoid declaring empty Megatron datasets for tiny sentinels.  Select
        # the largest worker-compatible partition count that can receive at
        # least one whole document; production languages are far above this.
        partitions = max(
            candidate
            for candidate in range(1, partitions + 1)
            if workers % candidate == 0
            and candidate <= int(item["expected_documents"])
        )
    work_dir = Path(settings["work_dir"]) / str(item["split"]) / str(item["language"])
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.jsonl"
    if input_path.exists() or input_path.is_symlink():
        input_path.unlink()
    input_path.symlink_to(Path(item["source_path"]))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(settings["python_executable"]),
        "-m",
        "multilingual_megatron.pipeline.stages.tokenization.tools.run_megatron_preprocess",
        "--script",
        str(preprocess),
        "--input",
        str(input_path),
        "--output-prefix",
        str(output_prefix),
        "--tokenizer-type",
        str(settings["tokenizer_type"]),
        "--tokenizer-model",
        str(settings["tokenizer_model"]),
        "--workers",
        str(workers),
        "--partitions",
        str(partitions),
        "--log-interval",
        str(settings["log_interval"]),
        "--append-eod",
    ]
    if partitions > 1:
        cmd.append("--keep-sequential-samples")
    env = dict(os.environ)
    orchestration_src = Path(__file__).resolve().parents[4]
    pythonpath = [
        str(orchestration_src),
        str(bridge_root / "3rdparty" / "Megatron-LM"),
        str(bridge_root / "src"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    subprocess.run(cmd, check=True, cwd=str(bridge_root), env=env)
    if partitions == 1:
        return [Path(str(output_prefix) + "_text_document")]
    prefixes = [Path(f"{output_prefix}_{index}_text_document") for index in range(partitions)]
    # preprocess_data.py also merges all partitions.  The partition prefixes
    # are the declared physical artifacts, so remove only this redundant copy.
    merged = Path(str(output_prefix) + "_text_document")
    Path(str(merged) + ".bin").unlink(missing_ok=True)
    Path(str(merged) + ".idx").unlink(missing_ok=True)
    return prefixes


def _verify_plan_summary(plan_dir: Path) -> dict[str, Any]:
    summary = _load_object(plan_dir / "summary.json", label="tokenization plan summary")
    if summary.get("schema") != PLAN_SCHEMA:
        raise PresplitTokenizationError("unsupported tokenization plan summary schema")
    _verify_payload_hash(summary, key="plan_payload_sha256", label="tokenization plan summary")
    core = summary.get("core_lineage")
    if not isinstance(core, dict) or canonical_json_sha256(core) != summary.get("core_lineage_sha256"):
        raise PresplitTokenizationError("tokenization core lineage hash mismatch")
    for path_field, hash_field in (
        ("corpus_manifest_path", "corpus_manifest_file_sha256"),
        ("sampling_plan_path", "sampling_plan_file_sha256"),
        ("selection_tokenizer_model_path", "selection_tokenizer_model_sha256"),
        ("tokenizer_model_file_path", "tokenizer_model_sha256"),
    ):
        path = _resolve_file(plan_dir, core.get(path_field), field=f"core_lineage.{path_field}")
        expected = _require_sha256(core.get(hash_field), field=f"core_lineage.{hash_field}")
        if sha256_file(path) != expected:
            raise PresplitTokenizationError(f"tokenization input changed after planning: {path}")
    for path_field, hash_field in (
        ("output_tokenizer_manifest_path", "output_tokenizer_manifest_file_sha256"),
        ("output_tokenizer_result_path", "output_tokenizer_result_file_sha256"),
    ):
        if path_field not in core:
            continue
        path = _resolve_file(plan_dir, core.get(path_field), field=f"core_lineage.{path_field}")
        if sha256_file(path) != core.get(hash_field):
            raise PresplitTokenizationError(f"tokenization input changed after planning: {path}")
    return summary


def run_tokenization_worker(
    *,
    plan_dir: Path,
    ranks_dir: Path,
    rank: int,
    bridge_root: Path,
    workers: int,
    partitions_per_language: int,
    tokenizer_type: str | None = None,
    python_executable: str = sys.executable,
    log_interval: int = 1000,
    force: bool = False,
    keep_work_dir: bool = False,
    reuse_tokenized_root: Path | None = None,
    reuse_corpus_manifest_path: Path | None = None,
    preprocess_runner: PreprocessRunner | None = None,
) -> dict[str, Any]:
    """Tokenize exactly the frozen files assigned to one array rank."""

    plan_dir = plan_dir.resolve()
    ranks_dir = ranks_dir.resolve()
    summary = _verify_plan_summary(plan_dir)
    if rank < 0 or rank >= int(summary["array_ranks"]):
        raise PresplitTokenizationError(f"rank {rank} is outside the planned array")
    rank_entry = summary.get("rank_files", {}).get(str(rank))
    if not isinstance(rank_entry, dict):
        raise PresplitTokenizationError(f"missing rank {rank} entry in plan summary")
    rank_plan_path = _resolve_file(plan_dir, rank_entry.get("path"), field=f"rank_files.{rank}.path")
    if sha256_file(rank_plan_path) != rank_entry.get("sha256"):
        raise PresplitTokenizationError(f"rank {rank} plan file SHA-256 mismatch")
    rank_plan = _load_object(rank_plan_path, label=f"rank {rank} plan")
    if rank_plan.get("schema") != RANK_PLAN_SCHEMA or rank_plan.get("rank") != rank:
        raise PresplitTokenizationError(f"rank {rank} plan identity mismatch")
    rank_plan_payload_hash = _verify_payload_hash(
        rank_plan, key="rank_plan_payload_sha256", label=f"rank {rank} plan"
    )
    if rank_plan_payload_hash != rank_entry.get("payload_sha256"):
        raise PresplitTokenizationError(f"rank {rank} payload identity mismatch")
    if rank_plan.get("core_lineage_sha256") != summary.get("core_lineage_sha256"):
        raise PresplitTokenizationError(f"rank {rank} lineage differs from plan summary")

    core = summary["core_lineage"]
    planned_tokenizer_type = str(core["tokenizer_type"])
    if tokenizer_type is not None and tokenizer_type != planned_tokenizer_type:
        raise PresplitTokenizationError(
            f"worker tokenizer type {tokenizer_type!r} differs from planned {planned_tokenizer_type!r}"
        )
    tokenizer_type = planned_tokenizer_type
    tokenizer_model = Path(core["tokenizer_model_path"])
    rank_root = ranks_dir / f"rank_{rank:05d}"
    if rank_root.exists():
        if not force:
            raise PresplitTokenizationError(f"rank output already exists: {rank_root}")
        shutil.rmtree(rank_root)
    rank_root.mkdir(parents=True)
    settings = {
        "bridge_root": str(bridge_root.resolve()),
        "tokenizer_model": str(tokenizer_model),
        "tokenizer_type": tokenizer_type,
        "workers": int(workers),
        "partitions_per_language": int(partitions_per_language),
        "python_executable": python_executable,
        "log_interval": int(log_interval),
        "work_dir": str(rank_root / "_work"),
    }
    runner = preprocess_runner or _default_preprocess_runner
    items = rank_plan.get("items")
    if not isinstance(items, list):
        raise PresplitTokenizationError(f"rank {rank} plan items must be a list")
    if (reuse_tokenized_root is None) != (reuse_corpus_manifest_path is None):
        raise PresplitTokenizationError(
            "reuse_tokenized_root and reuse_corpus_manifest_path must be provided together"
        )
    reuse, reuse_lineage = (
        _reuse_catalog(
            tokenized_root=reuse_tokenized_root,
            corpus_manifest_path=reuse_corpus_manifest_path,
            tokenizer_model_sha256=str(core["tokenizer_model_sha256"]),
            requested_keys={
                (str(item.get("language")), str(item.get("split")))
                for item in items
                if isinstance(item, dict)
            },
        )
        if reuse_tokenized_root is not None and reuse_corpus_manifest_path is not None
        else ({}, None)
    )
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise PresplitTokenizationError(f"rank {rank} contains an invalid work item")
        source = Path(str(item.get("source_path", "")))
        if not source.is_file() or source.stat().st_size != item.get("source_size_bytes"):
            raise PresplitTokenizationError(f"rank {rank} source identity changed: {source}")
        if sha256_file(source) != item.get("source_sha256"):
            raise PresplitTokenizationError(f"rank {rank} source SHA-256 changed: {source}")
        output_prefix = rank_root / "megatron" / str(item["split"]) / str(item["language"])
        reusable = reuse.get((str(item["language"]), str(item["split"])))
        immutable_fields = (
            "source_sha256",
            "source_size_bytes",
            "expected_documents",
            "expected_tokens_including_eod",
        )
        can_reuse = reusable is not None and all(
            reusable.get(field) == item.get(field) for field in immutable_fields
        )
        if can_reuse:
            prefixes = _copy_reused_prefixes(
                rows=reusable["prefix_artifacts"], output_prefix=output_prefix
            )
        else:
            prefixes = [Path(value) for value in runner(item, output_prefix, settings)]
        if not prefixes or len(prefixes) != len(set(map(str, prefixes))):
            raise PresplitTokenizationError(
                f"{item['language']}/{item['split']} preprocessing returned no unique prefixes"
            )
        artifacts = [_artifact(prefix) for prefix in prefixes]
        if any(row["sequences"] != row["documents"] for row in artifacts):
            raise PresplitTokenizationError(
                f"{item['language']}/{item['split']} must emit one Megatron sequence per document"
            )
        documents = sum(int(row["documents"]) for row in artifacts)
        tokens = sum(int(row["tokens_including_eod"]) for row in artifacts)
        if documents != int(item["expected_documents"]):
            raise PresplitTokenizationError(
                f"{item['language']}/{item['split']} document count mismatch: "
                f"expected {item['expected_documents']}, observed {documents}"
            )
        if (
            core["token_count_policy"] == "match_selection_tokenizer"
            and tokens != int(item["expected_tokens_including_eod"])
        ):
            raise PresplitTokenizationError(
                f"{item['language']}/{item['split']} token count mismatch: "
                f"expected {item['expected_tokens_including_eod']}, observed {tokens}"
            )
        if tokens < documents:
            raise PresplitTokenizationError(
                f"{item['language']}/{item['split']} emitted fewer tokens than documents"
            )
        result_item = {
            **item,
            "prefix_artifacts": artifacts,
            "observed_tokens_including_eod": tokens,
            "reused_predecessor_prefix": bool(can_reuse),
        }
        if can_reuse:
            result_item["predecessor_prefix_artifacts"] = reusable[
                "prefix_artifacts"
            ]
        results.append(result_item)
    result: dict[str, Any] = {
        "schema": RANK_RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "rank_plan_file_sha256": rank_entry["sha256"],
        "rank_plan_payload_sha256": rank_plan_payload_hash,
        "core_lineage_sha256": summary["core_lineage_sha256"],
        "preprocessing": {
            "tokenizer_type": tokenizer_type,
            "tokenizer_model_sha256": core["tokenizer_model_sha256"],
            "token_count_policy": core["token_count_policy"],
            "append_eod": True,
            "random_sampling": False,
            "split_assignment": "preserved_from_finalized_corpus",
            "workers": int(workers),
            "partitions_per_language": int(partitions_per_language),
            "reuse_enabled": reuse_lineage is not None,
            "reuse_lineage": reuse_lineage,
        },
        "items": results,
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    _write_json(rank_root / "result.json", result)
    if not keep_work_dir:
        shutil.rmtree(rank_root / "_work", ignore_errors=True)
    return result


def _verify_prefix_artifact(
    raw: Any,
    *,
    field: str,
    cache: dict[
        tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PresplitTokenizationError(f"{field} must be an artifact object")
    prefix = Path(str(raw.get("prefix", ""))).resolve()
    observed = _artifact(prefix, cache=cache)
    for key in (
        "bin_sha256",
        "bin_size_bytes",
        "idx_sha256",
        "idx_size_bytes",
        "sequences",
        "documents",
        "tokens_including_eod",
        "dtype_code",
    ):
        if observed[key] != raw.get(key):
            raise PresplitTokenizationError(f"{field}.{key} changed after worker completion")
    return observed


def _reuse_catalog(
    *,
    tokenized_root: Path,
    corpus_manifest_path: Path,
    tokenizer_model_sha256: str,
    requested_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Load hash-verified predecessor prefixes keyed by language and split."""

    tokenized_root = tokenized_root.resolve()
    corpus_manifest_path = corpus_manifest_path.resolve()
    corpus = _load_object(corpus_manifest_path, label="reuse corpus manifest")
    _verify_payload_hash(corpus, key="manifest_sha256", label="reuse corpus manifest")
    if corpus.get("ready") is not True or corpus.get("release_status") != "ready":
        raise PresplitTokenizationError("reuse corpus manifest is not ready")
    manifest_paths = {
        "train": tokenized_root / "megatron_data_manifest.json",
        "validation": tokenized_root / "validation_megatron_data_manifest.json",
    }
    manifests = {
        split: _load_object(path, label=f"reuse {split} manifest")
        for split, path in manifest_paths.items()
    }
    lineages = [manifest.get("lineage") for manifest in manifests.values()]
    if not all(isinstance(lineage, dict) for lineage in lineages) or lineages[0] != lineages[1]:
        raise PresplitTokenizationError(
            "reuse train and validation manifests do not share exact lineage"
        )
    lineage = dict(lineages[0])
    corpus_file_hash = sha256_file(corpus_manifest_path)
    corpus_payload_hash = str(corpus["manifest_sha256"])
    if (
        lineage.get("corpus_manifest_file_sha256") != corpus_file_hash
        or lineage.get("corpus_manifest_payload_sha256") != corpus_payload_hash
    ):
        raise PresplitTokenizationError(
            "reuse tokenized manifests do not belong to the supplied corpus manifest"
        )
    corpus_inputs = corpus.get("inputs")
    corpus_plan = (
        corpus_inputs.get("sampling_plan") if isinstance(corpus_inputs, dict) else None
    )
    if not isinstance(corpus_plan, dict):
        raise PresplitTokenizationError("reuse corpus lacks sampling-plan lineage")
    sampling_plan_path = _resolve_file(
        corpus_manifest_path.parent,
        corpus_plan.get("path"),
        field="reuse corpus inputs.sampling_plan.path",
    )
    sampling_plan = _load_object(sampling_plan_path, label="reuse sampling plan")
    if not verify_plan_hash(sampling_plan):
        raise PresplitTokenizationError("reuse sampling plan payload hash mismatch")
    sampling_file_hash = sha256_file(sampling_plan_path)
    sampling_payload_hash = str(sampling_plan.get("plan_sha256", ""))
    declared_sampling_file_hash = corpus_plan.get(
        "file_sha256", corpus_plan.get("sha256")
    )
    if (
        declared_sampling_file_hash != sampling_file_hash
        or corpus_plan.get("payload_sha256") != sampling_payload_hash
        or lineage.get("sampling_plan_file_sha256") != sampling_file_hash
        or lineage.get("sampling_plan_payload_sha256") != sampling_payload_hash
    ):
        raise PresplitTokenizationError(
            "reuse corpus and tokenized manifests disagree on the sampling plan"
        )
    if lineage.get("tokenizer_model_sha256") != tokenizer_model_sha256:
        raise PresplitTokenizationError(
            "reuse tokenized lineage was built with another tokenizer"
        )
    corpus_languages = set(map(str, corpus.get("languages", {})))
    if not corpus_languages:
        raise PresplitTokenizationError("reuse corpus has no languages")
    for split, manifest in manifests.items():
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise PresplitTokenizationError(f"unsupported reuse {split} manifest schema")
        if set(map(str, manifest.get("languages", []))) != corpus_languages:
            raise PresplitTokenizationError(
                f"reuse {split} manifest language coverage differs from its corpus"
            )
    provenance: dict[str, Any] = {
        "schema": "fineweb_ab_v2_presplit_reuse_lineage/v1",
        "predecessor_tokenized_root": str(tokenized_root),
        "trusted_predecessor_artifact_root": str(tokenized_root.parent),
        "predecessor_corpus_manifest": {
            "path": str(corpus_manifest_path),
            "file_sha256": corpus_file_hash,
            "payload_sha256": corpus_payload_hash,
        },
        "predecessor_sampling_plan": {
            "path": str(sampling_plan_path),
            "file_sha256": sampling_file_hash,
            "payload_sha256": sampling_payload_hash,
        },
        "predecessor_manifests": {
            split: {
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "payload_sha256": str(manifests[split]["manifest_payload_sha256"]),
            }
            for split, path in manifest_paths.items()
        },
        "tokenizer_model_sha256": tokenizer_model_sha256,
    }
    provenance["reuse_lineage_sha256"] = canonical_json_sha256(provenance)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for split, manifest in manifests.items():
        _verify_payload_hash(
            manifest,
            key="manifest_payload_sha256",
            label=f"reuse {split} manifest",
        )
        if manifest.get("tokenizer_model_sha256") != tokenizer_model_sha256:
            raise PresplitTokenizationError(
                f"reuse {split} manifest was built with another tokenizer"
            )
        split_payload = manifest.get("splits", {}).get(split)
        raw_artifacts = (
            split_payload.get("prefix_artifacts_by_language")
            if isinstance(split_payload, dict)
            else None
        )
        if not isinstance(raw_artifacts, dict):
            raise PresplitTokenizationError(
                f"reuse {split} manifest lacks per-language prefix artifacts"
            )
        for language, rows in raw_artifacts.items():
            if (str(language), split) not in requested_keys:
                continue
            corpus_split = corpus.get("languages", {}).get(language, {}).get(split)
            if not isinstance(corpus_split, dict) or not isinstance(rows, list) or not rows:
                raise PresplitTokenizationError(
                    f"reuse lineage is incomplete for {language}/{split}"
                )
            output = corpus_split.get("output")
            if not isinstance(output, dict):
                raise PresplitTokenizationError(
                    f"reuse corpus lacks output identity for {language}/{split}"
                )
            source = _resolve_file(
                corpus_manifest_path.parent,
                output.get("path"),
                field=f"reuse corpus {language}/{split}.output.path",
            )
            artifacts = [
                _verify_prefix_artifact(row, field=f"reuse {split} manifest {language}")
                for row in rows
            ]
            for artifact in artifacts:
                prefix = Path(str(artifact["prefix"])).resolve()
                try:
                    prefix.relative_to(tokenized_root.parent)
                except ValueError as error:
                    raise PresplitTokenizationError(
                        f"reuse prefix escapes predecessor artifact root: {prefix}"
                    ) from error
                for suffix in ("bin", "idx"):
                    file_path = Path(f"{prefix}.{suffix}")
                    if file_path.is_symlink() or file_path.stat().st_mode & 0o222:
                        raise PresplitTokenizationError(
                            f"reuse prefix must be an immutable regular file: {file_path}"
                        )
            result[(str(language), split)] = {
                "source_path": str(source),
                "source_sha256": _require_sha256(
                    output.get("sha256"),
                    field=f"reuse corpus {language}/{split}.output.sha256",
                ),
                "source_size_bytes": int(output.get("size_bytes", -1)),
                "expected_documents": int(corpus_split.get("emitted_documents", -1)),
                "expected_tokens_including_eod": int(corpus_split.get("selected_tokens", -1)),
                "prefix_artifacts": artifacts,
            }
    return result, provenance


def _copy_reused_prefixes(
    *, rows: Sequence[Mapping[str, Any]], output_prefix: Path
) -> list[Path]:
    """Copy verified prefixes without changing the sealed predecessor inode."""

    targets: list[Path] = []
    for index, raw in enumerate(rows):
        source_prefix = Path(str(raw["prefix"]))
        target = output_prefix if len(rows) == 1 else Path(f"{output_prefix}_{index:03d}")
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("bin", "idx"):
            source = Path(f"{source_prefix}.{suffix}")
            destination = Path(f"{target}.{suffix}")
            if destination.exists() or destination.is_symlink():
                raise PresplitTokenizationError(f"reuse target already exists: {destination}")
            source_stat = source.stat()
            temporary = destination.with_name(f".{destination.name}.partial")
            try:
                shutil.copy2(source, temporary)
                os.chmod(temporary, temporary.stat().st_mode & ~0o222)
                os.replace(temporary, destination)
            except OSError as error:
                temporary.unlink(missing_ok=True)
                raise PresplitTokenizationError(
                    f"could not copy reused prefix {source} -> {destination}: {error}"
                ) from error
            after = source.stat()
            if _artifact_identity(source) != (
                int(source_stat.st_dev),
                int(source_stat.st_ino),
                int(source_stat.st_size),
                int(source_stat.st_mtime_ns),
                int(source_stat.st_ctime_ns),
            ) or (after.st_dev, after.st_ino) == (
                destination.stat().st_dev,
                destination.stat().st_ino,
            ):
                raise PresplitTokenizationError(
                    f"copy reuse changed or hard-linked its sealed predecessor: {source}"
                )
        observed = _artifact(target)
        for key in (
            "bin_sha256",
            "bin_size_bytes",
            "idx_sha256",
            "idx_size_bytes",
            "sequences",
            "documents",
            "tokens_including_eod",
            "dtype_code",
        ):
            if observed[key] != raw.get(key):
                raise PresplitTokenizationError(
                    f"copied reuse prefix changed field {key}: {source_prefix}"
                )
        targets.append(target)
    return targets


def _split_payload(
    *,
    languages: Sequence[str],
    artifacts: Mapping[str, Sequence[dict[str, Any]]],
    documents: Mapping[str, int],
    tokens: Mapping[str, int],
    language_weights: Mapping[str, float] | None,
) -> dict[str, Any]:
    prefixes: list[str] = []
    weights: list[float] | None = [] if language_weights is not None else None
    prefixes_by_language: dict[str, list[str]] = {}
    artifacts_by_language: dict[str, list[dict[str, Any]]] = {}
    for language in languages:
        rows = sorted(artifacts[language], key=lambda row: str(row["prefix"]))
        language_prefixes = [str(row["prefix"]) for row in rows]
        prefixes.extend(language_prefixes)
        prefixes_by_language[language] = language_prefixes
        artifacts_by_language[language] = rows
        if weights is not None:
            probability = float(language_weights[language])
            weights.extend([probability / len(rows)] * len(rows))
    if weights is not None and not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("partitioned blend weights no longer sum to one")
    return {
        "prefixes": prefixes,
        "weights": weights,
        "prefixes_by_language": prefixes_by_language,
        "prefix_artifacts_by_language": artifacts_by_language,
        "documents_per_language": {language: int(documents[language]) for language in languages},
        "tokens_per_language": {language: int(tokens[language]) for language in languages},
        "total_documents": sum(int(documents[language]) for language in languages),
        "total_tokens_including_eod": sum(int(tokens[language]) for language in languages),
    }


def _manifest_with_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_payload_sha256"] = canonical_json_sha256(result)
    return result


def merge_tokenization_results(
    *,
    plan_dir: Path,
    ranks_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify all ranks and emit K-specific and global validation manifests."""

    plan_dir = plan_dir.resolve()
    ranks_dir = ranks_dir.resolve()
    output_dir = output_dir.resolve()
    summary = _verify_plan_summary(plan_dir)
    if output_dir.exists():
        raise PresplitTokenizationError(f"merge output already exists: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        raise PresplitTokenizationError(f"merge staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
            split: defaultdict(list) for split in _SPLITS
        }
        documents: dict[str, dict[str, int]] = {split: {} for split in _SPLITS}
        tokens: dict[str, dict[str, int]] = {split: {} for split in _SPLITS}
        seen_items: set[tuple[str, str]] = set()
        seen_prefixes: set[str] = set()
        reused_items: set[tuple[str, str]] = set()
        reuse_lineage: dict[str, Any] | None = None
        artifact_cache: dict[
            tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]],
            dict[str, Any],
        ] = {}
        result_files: list[dict[str, Any]] = []
        for rank in range(int(summary["array_ranks"])):
            result_path = ranks_dir / f"rank_{rank:05d}" / "result.json"
            result = _load_object(result_path, label=f"rank {rank} result")
            if result.get("schema") != RANK_RESULT_SCHEMA or result.get("rank") != rank:
                raise PresplitTokenizationError(f"rank {rank} result identity mismatch")
            result_hash = _verify_payload_hash(
                result, key="result_payload_sha256", label=f"rank {rank} result"
            )
            rank_entry = summary["rank_files"][str(rank)]
            if (
                result.get("rank_plan_file_sha256") != rank_entry["sha256"]
                or result.get("rank_plan_payload_sha256") != rank_entry["payload_sha256"]
                or result.get("core_lineage_sha256") != summary["core_lineage_sha256"]
            ):
                raise PresplitTokenizationError(f"rank {rank} result lineage mismatch")
            preprocessing = result.get("preprocessing")
            if (
                not isinstance(preprocessing, dict)
                or preprocessing.get("tokenizer_type")
                != summary["core_lineage"]["tokenizer_type"]
                or preprocessing.get("tokenizer_model_sha256")
                != summary["core_lineage"]["tokenizer_model_sha256"]
                or preprocessing.get("token_count_policy")
                != summary["core_lineage"]["token_count_policy"]
                or preprocessing.get("append_eod") is not True
                or preprocessing.get("random_sampling") is not False
                or preprocessing.get("split_assignment")
                != "preserved_from_finalized_corpus"
            ):
                raise PresplitTokenizationError(f"rank {rank} preprocessing contract mismatch")
            rank_reuse_enabled = preprocessing.get("reuse_enabled")
            rank_reuse_lineage = preprocessing.get("reuse_lineage")
            if not isinstance(rank_reuse_enabled, bool):
                raise PresplitTokenizationError(
                    f"rank {rank} lacks an explicit reuse-enabled marker"
                )
            if rank_reuse_enabled:
                if not isinstance(rank_reuse_lineage, dict):
                    raise PresplitTokenizationError(
                        f"rank {rank} lacks predecessor reuse lineage"
                    )
                expected_reuse_hash = _require_sha256(
                    rank_reuse_lineage.get("reuse_lineage_sha256"),
                    field=f"rank {rank} reuse_lineage_sha256",
                )
                reuse_candidate = dict(rank_reuse_lineage)
                reuse_candidate.pop("reuse_lineage_sha256", None)
                if canonical_json_sha256(reuse_candidate) != expected_reuse_hash:
                    raise PresplitTokenizationError(
                        f"rank {rank} predecessor reuse lineage hash mismatch"
                    )
                if reuse_lineage is None:
                    reuse_lineage = dict(rank_reuse_lineage)
                elif reuse_lineage != rank_reuse_lineage:
                    raise PresplitTokenizationError(
                        "tokenization ranks use different predecessors"
                    )
            elif rank_reuse_lineage is not None:
                raise PresplitTokenizationError(
                    f"rank {rank} declares reuse lineage while reuse is disabled"
                )
            rank_plan_path = _resolve_file(
                plan_dir,
                rank_entry.get("path"),
                field=f"rank_files.{rank}.path",
            )
            if sha256_file(rank_plan_path) != rank_entry["sha256"]:
                raise PresplitTokenizationError(f"rank {rank} plan changed before merge")
            rank_plan = _load_object(rank_plan_path, label=f"rank {rank} plan")
            _verify_payload_hash(
                rank_plan,
                key="rank_plan_payload_sha256",
                label=f"rank {rank} plan",
            )
            expected_items = rank_plan.get("items")
            if not isinstance(expected_items, list):
                raise PresplitTokenizationError(f"rank {rank} plan items are invalid")
            expected_by_key = {
                (str(item.get("language")), str(item.get("split"))): item
                for item in expected_items
                if isinstance(item, dict)
            }
            if len(expected_by_key) != len(expected_items):
                raise PresplitTokenizationError(f"rank {rank} plan contains duplicate items")
            raw_items = result.get("items")
            if not isinstance(raw_items, list) or len(raw_items) != rank_entry["item_count"]:
                raise PresplitTokenizationError(f"rank {rank} result item count mismatch")
            for item_index, item in enumerate(raw_items):
                if not isinstance(item, dict):
                    raise PresplitTokenizationError(f"rank {rank} item {item_index} is invalid")
                language, split = str(item.get("language")), str(item.get("split"))
                key = (language, split)
                expected_item = expected_by_key.get(key)
                if split not in _SPLITS or expected_item is None or key in seen_items:
                    raise PresplitTokenizationError(f"duplicate or invalid tokenization item: {key}")
                for immutable_field in (
                    "source_path",
                    "source_sha256",
                    "source_size_bytes",
                    "expected_documents",
                    "expected_tokens_including_eod",
                ):
                    if item.get(immutable_field) != expected_item.get(immutable_field):
                        raise PresplitTokenizationError(
                            f"rank {rank} item {key} changed immutable field {immutable_field}"
                        )
                seen_items.add(key)
                source = Path(str(item.get("source_path", "")))
                if (
                    not source.is_file()
                    or source.stat().st_size != item.get("source_size_bytes")
                    or sha256_file(source) != item.get("source_sha256")
                ):
                    raise PresplitTokenizationError(f"source changed before merge: {source}")
                raw_artifacts = item.get("prefix_artifacts")
                if not isinstance(raw_artifacts, list) or not raw_artifacts:
                    raise PresplitTokenizationError(f"missing prefix artifacts for {language}/{split}")
                rows = [
                    _verify_prefix_artifact(
                        raw,
                        field=f"rank {rank} item {item_index}",
                        cache=artifact_cache,
                    )
                    for raw in raw_artifacts
                ]
                reused = item.get("reused_predecessor_prefix")
                if not isinstance(reused, bool):
                    raise PresplitTokenizationError(
                        f"rank {rank} item {key} lacks an explicit reuse marker"
                    )
                predecessor_rows = item.get("predecessor_prefix_artifacts")
                if reused:
                    if not rank_reuse_enabled or not isinstance(predecessor_rows, list):
                        raise PresplitTokenizationError(
                            f"rank {rank} item {key} has incomplete reuse evidence"
                        )
                    verified_predecessor = [
                        _verify_prefix_artifact(
                            raw,
                            field=f"rank {rank} predecessor item {item_index}",
                            cache=artifact_cache,
                        )
                        for raw in predecessor_rows
                    ]
                    if len(verified_predecessor) != len(rows):
                        raise PresplitTokenizationError(
                            f"rank {rank} item {key} predecessor partition count differs"
                        )
                    for current, predecessor in zip(rows, verified_predecessor, strict=True):
                        for field in (
                            "bin_sha256",
                            "bin_size_bytes",
                            "idx_sha256",
                            "idx_size_bytes",
                            "sequences",
                            "documents",
                            "tokens_including_eod",
                            "dtype_code",
                        ):
                            if current[field] != predecessor[field]:
                                raise PresplitTokenizationError(
                                    f"rank {rank} item {key} reused artifact changed {field}"
                                )
                        for suffix in ("bin", "idx"):
                            current_path = Path(f"{current['prefix']}.{suffix}")
                            predecessor_path = Path(f"{predecessor['prefix']}.{suffix}")
                            current_stat = current_path.stat()
                            predecessor_stat = predecessor_path.stat()
                            if (
                                current_stat.st_dev,
                                current_stat.st_ino,
                            ) == (
                                predecessor_stat.st_dev,
                                predecessor_stat.st_ino,
                            ):
                                raise PresplitTokenizationError(
                                    f"rank {rank} item {key} shares its sealed predecessor inode"
                                )
                    reused_items.add(key)
                elif predecessor_rows is not None:
                    raise PresplitTokenizationError(
                        f"rank {rank} item {key} has predecessor artifacts but was rebuilt"
                    )
                item_prefixes = {str(row["prefix"]) for row in rows}
                if len(item_prefixes) != len(rows) or seen_prefixes.intersection(item_prefixes):
                    raise PresplitTokenizationError(
                        f"Megatron prefix is reused across tokenization items: {key}"
                    )
                seen_prefixes.update(item_prefixes)
                observed_documents = sum(int(row["documents"]) for row in rows)
                observed_tokens = sum(int(row["tokens_including_eod"]) for row in rows)
                if observed_documents != int(item.get("expected_documents", -1)):
                    raise PresplitTokenizationError(f"document drift for {language}/{split}")
                if (
                    summary["core_lineage"]["token_count_policy"] == "match_selection_tokenizer"
                    and observed_tokens != int(item.get("expected_tokens_including_eod", -1))
                ):
                    raise PresplitTokenizationError(f"token drift for {language}/{split}")
                if observed_tokens != int(item.get("observed_tokens_including_eod", -1)):
                    raise PresplitTokenizationError(f"reported output-token drift for {language}/{split}")
                by_split[split][language] = rows
                documents[split][language] = observed_documents
                tokens[split][language] = observed_tokens
            result_files.append(
                {
                    "path": str(result_path),
                    "sha256": sha256_file(result_path),
                    "payload_sha256": result_hash,
                }
            )

        tiers = summary["tiers"]
        maximum_tier = str(summary["maximum_tier"])
        languages = list(tiers[maximum_tier]["languages"])
        if set(languages) != set(summary["languages"]):
            raise PresplitTokenizationError(
                "maximum-tier language membership differs from the physical tokenization pool"
            )
        expected_items = {(language, split) for language in languages for split in _SPLITS}
        if seen_items != expected_items:
            raise PresplitTokenizationError(
                f"tokenization result coverage mismatch: missing={sorted(expected_items - seen_items)}, "
                f"unexpected={sorted(seen_items - expected_items)}"
            )
        reuse_summary = {
            "enabled": reuse_lineage is not None,
            "predecessor": reuse_lineage,
            "materialization": "verified_content_copy_distinct_inode",
            "sealed_predecessor_inode_mutation": False,
            "reused_item_count": len(reused_items),
            "rebuilt_item_count": len(seen_items - reused_items),
            "reused_items_by_split": {
                split: sorted(language for language, item_split in reused_items if item_split == split)
                for split in _SPLITS
            },
            "rebuilt_items_by_split": {
                split: sorted(
                    language
                    for language, item_split in seen_items - reused_items
                    if item_split == split
                )
                for split in _SPLITS
            },
        }
        core = summary["core_lineage"]
        common = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "format": "megatron_gpt",
            "stage": "tokenize",
            "tokenizer_type": core["tokenizer_type"],
            "tokenizer_model": core["tokenizer_model_path"],
            "tokenizer_model_sha256": core["tokenizer_model_sha256"],
            "append_eod": True,
            "lineage": {
                **core,
                "tokenization_plan_payload_sha256": summary["plan_payload_sha256"],
                "tokenization_core_lineage_sha256": summary["core_lineage_sha256"],
            },
            "incremental_tokenization": reuse_summary,
        }
        validation = _split_payload(
            languages=languages,
            artifacts=by_split["validation"],
            documents=documents["validation"],
            tokens=tokens["validation"],
            language_weights={language: 1.0 / len(languages) for language in languages},
        )
        validation_manifest = _manifest_with_hash(
            {
                **common,
                "purpose": "global_K200_per_language_validation",
                "languages": languages,
                "prefixes": [],
                "weights": None,
                "prefixes_by_language": {},
                "documents_per_language": {},
                "tokens_per_language": {},
                "splits": {"train": None, "validation": validation, "test": None},
            }
        )
        validation_path = staging / "validation_megatron_data_manifest.json"
        _write_json(validation_path, validation_manifest)

        tier_artifacts: dict[str, dict[str, Any]] = {}
        for tier_name in sorted(tiers, key=_tier_sort_key):
            tier = tiers[tier_name]
            tier_languages = list(tier["languages"])
            train = _split_payload(
                languages=tier_languages,
                artifacts=by_split["train"],
                documents=documents["train"],
                tokens=tokens["train"],
                language_weights=tier["probabilities"],
            )
            manifest = _manifest_with_hash(
                {
                    **common,
                    "purpose": "nested_K_training_with_global_validation",
                    "tier": tier_name,
                    "languages": tier_languages,
                    "alpha": tier["alpha"],
                    "target_train_tokens": tier["target_train_tokens"],
                    "target_exposure_tokens": tier["target_exposure_tokens"],
                    "language_probabilities": tier["probabilities"],
                    "prefixes": train["prefixes"],
                    "weights": train["weights"],
                    "prefixes_by_language": train["prefixes_by_language"],
                    "documents_per_language": train["documents_per_language"],
                    "tokens_per_language": train["tokens_per_language"],
                    "splits": {"train": train, "validation": validation, "test": None},
                }
            )
            path = staging / "tiers" / tier_name / "megatron_data_manifest.json"
            _write_json(path, manifest)
            tier_artifacts[tier_name] = {
                "path": str(path.relative_to(staging)),
                "sha256": sha256_file(path),
                "payload_sha256": manifest["manifest_payload_sha256"],
                "language_count": len(tier_languages),
                "train_prefix_count": len(train["prefixes"]),
            }

        maximum_path = staging / tier_artifacts[maximum_tier]["path"]
        shutil.copyfile(maximum_path, staging / "megatron_data_manifest.json")
        shutil.copyfile(maximum_path, staging / "dataset_info.json")
        report: dict[str, Any] = {
            "schema": "fineweb_ab_v2_presplit_tokenization_merge/v1",
            "schema_version": SCHEMA_VERSION,
            "pass": True,
            "language_count": len(languages),
            "item_count": len(seen_items),
            "maximum_tier": maximum_tier,
            "tiers": tier_artifacts,
            "global_validation_manifest": {
                "path": str(validation_path.relative_to(staging)),
                "sha256": sha256_file(validation_path),
                "payload_sha256": validation_manifest["manifest_payload_sha256"],
            },
            "rank_results": result_files,
            "incremental_tokenization": reuse_summary,
            "exact_counts": {
                split: {
                    "documents_per_language": documents[split],
                    "tokens_per_language": tokens[split],
                }
                for split in _SPLITS
            },
            "scientific_contract": {
                "random_resplit": False,
                "physical_K200_pool_tokenized_once": True,
                "tier_mixtures_from_sampling_plan_probabilities": True,
                "validation_is_frozen_and_unrehydrated": True,
                "selection_tokenizer_lineage_preserved": True,
                "output_token_counts_recomputed": (
                    core["token_count_policy"] == "recompute_for_output_tokenizer"
                ),
            },
        }
        report["report_payload_sha256"] = canonical_json_sha256(report)
        _write_json(staging / "tokenization_report.json", report)
        _write_json(
            staging / "state.json",
            {
                "stage": "tokenize",
                "pass": True,
                "report_payload_sha256": report["report_payload_sha256"],
            },
        )
        staging.replace(output_dir)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="verify inputs and build array plans")
    plan.add_argument("--corpus-manifest", required=True)
    plan.add_argument("--sampling-plan", required=True)
    plan.add_argument("--tokenizer-model", required=True)
    plan.add_argument("--plan-dir", required=True)
    plan.add_argument("--array-ranks", required=True, type=int)
    plan.add_argument("--verification-workers", type=int, default=1)
    plan.add_argument("--force", action="store_true")

    worker = subparsers.add_parser("worker", help="tokenize one array rank")
    worker.add_argument("--plan-dir", required=True)
    worker.add_argument("--ranks-dir", required=True)
    worker.add_argument("--rank", type=int, default=None)
    worker.add_argument("--bridge-root", required=True)
    worker.add_argument("--workers", type=int, default=16)
    worker.add_argument("--partitions-per-language", type=int, default=1)
    worker.add_argument("--tokenizer-type")
    worker.add_argument("--python-executable", default=sys.executable)
    worker.add_argument("--log-interval", type=int, default=1000)
    worker.add_argument("--force", action="store_true")
    worker.add_argument("--keep-work-dir", action="store_true")
    worker.add_argument("--reuse-tokenized-root")
    worker.add_argument("--reuse-corpus-manifest")

    merge = subparsers.add_parser("merge", help="verify ranks and write nested-K manifests")
    merge.add_argument("--plan-dir", required=True)
    merge.add_argument("--ranks-dir", required=True)
    merge.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        summary = build_tokenization_plan(
            corpus_manifest_path=Path(args.corpus_manifest),
            sampling_plan_path=Path(args.sampling_plan),
            tokenizer_model_path=Path(args.tokenizer_model),
            plan_dir=Path(args.plan_dir),
            array_ranks=int(args.array_ranks),
            verification_workers=int(args.verification_workers),
            force=bool(args.force),
        )
        print(
            f"[OK] presplit plan ranks={summary['array_ranks']} items={summary['item_count']} "
            f"sha256={summary['plan_payload_sha256']}",
            flush=True,
        )
        return 0
    if args.command == "worker":
        rank = int(args.rank if args.rank is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        result = run_tokenization_worker(
            plan_dir=Path(args.plan_dir),
            ranks_dir=Path(args.ranks_dir),
            rank=rank,
            bridge_root=Path(args.bridge_root),
            workers=int(args.workers),
            partitions_per_language=int(args.partitions_per_language),
            tokenizer_type=(str(args.tokenizer_type) if args.tokenizer_type else None),
            python_executable=str(args.python_executable),
            log_interval=int(args.log_interval),
            force=bool(args.force),
            keep_work_dir=bool(args.keep_work_dir),
            reuse_tokenized_root=(
                Path(args.reuse_tokenized_root) if args.reuse_tokenized_root else None
            ),
            reuse_corpus_manifest_path=(
                Path(args.reuse_corpus_manifest) if args.reuse_corpus_manifest else None
            ),
        )
        print(
            f"[OK] presplit tokenization rank={rank} items={len(result['items'])} "
            f"sha256={result['result_payload_sha256']}",
            flush=True,
        )
        return 0
    report = merge_tokenization_results(
        plan_dir=Path(args.plan_dir),
        ranks_dir=Path(args.ranks_dir),
        output_dir=Path(args.output_dir),
    )
    print(
        f"[OK] presplit merge languages={report['language_count']} "
        f"sha256={report['report_payload_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
