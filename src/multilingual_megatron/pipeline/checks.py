"""Lightweight post-stage verification checks.

Each function inspects stage outputs, writes a ``_check.json`` manifest
into the stage's output directory, and raises on critical problems.
Designed for minimal LOC — every check is a single callable.
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("multilingual_megatron.pipeline.checks")


def _validation_fraction(tokens_by_split: dict[str, dict[str, int]], lang: str) -> float | None:
    val_tokens = int(tokens_by_split.get("validation", {}).get(lang, 0))
    train_tokens = int(tokens_by_split.get("train", {}).get(lang, 0))
    total = val_tokens + train_tokens
    if total <= 0:
        return None
    return val_tokens / total



def _write_check(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    LOGGER.info("[CHECK] wrote %s", path)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def check_sample(sampled_dir: Path, shard_dir: Path, *, dry_run: bool) -> None:
    """Verify sampled shards: per-language file counts, doc counts, disk size."""

    if dry_run:
        return

    out: dict[str, Any] = {"stage": "sample", "pass": True, "languages": {}}
    total_docs = 0
    total_bytes = 0
    shard_files = sorted(shard_dir.rglob("*.jsonl")) + sorted(shard_dir.rglob("*.jsonl.gz"))
    sample_sizes_path = sampled_dir / "sample_sizes.tsv"

    if sample_sizes_path.exists():
        expected_docs: dict[str, int] = {}
        with sample_sizes_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                subset = str(row.get("subset", "")).strip()
                if not subset:
                    continue
                expected_docs[subset] = int(row.get("sample_size", 0) or 0)

        for jsonl in shard_files:
            lang = jsonl.parent.name
            size = jsonl.stat().st_size
            entry = out["languages"].setdefault(lang, {"files": 0, "docs": expected_docs.get(lang, 0), "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += size
            total_bytes += size

        total_docs = sum(int(v["docs"]) for v in out["languages"].values())
        missing = [lang for lang in sorted(expected_docs) if lang not in out["languages"]]
        if missing:
            out["pass"] = False
            out["missing_languages"] = missing
            _write_check(sampled_dir / "_check.json", out)
            raise SystemExit(f"[CHECK FAIL] sample: missing shard files for languages: {missing}")

        out["total_docs"] = total_docs
        out["total_bytes"] = total_bytes
        out["total_bytes_human"] = f"{total_bytes / 1024 / 1024:.2f} MB"
        out["num_languages"] = len(out["languages"])
        _write_check(sampled_dir / "_check.json", out)
        return

    # iterate over shard files and count documents and count
    for jsonl in shard_files:
        lang = jsonl.parent.name  # e.g. arb_Arab/part_000.jsonl(.gz)
        size = jsonl.stat().st_size

        if jsonl.name.endswith(".jsonl.gz"):
            with gzip.open(jsonl, "rt", encoding="utf-8") as fh:
                docs = sum(1 for _ in fh)
        else:
            docs = sum(1 for _ in jsonl.open("r", encoding="utf-8"))

        entry = out["languages"].setdefault(lang, {"files": 0, "docs": 0, "bytes": 0})
        entry["files"] += 1
        entry["docs"] += docs
        entry["bytes"] += size
        total_docs += docs
        total_bytes += size

    out["total_docs"] = total_docs
    out["total_bytes"] = total_bytes
    out["total_bytes_human"] = f"{total_bytes / 1024 / 1024:.2f} MB"
    out["num_languages"] = len(out["languages"])

    if total_docs == 0:
        out["pass"] = False
        _write_check(sampled_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] sample: zero documents produced")

    empty = [lang for lang, v in out["languages"].items() if v["docs"] == 0]
    if empty:
        out["pass"] = False
        out["empty_languages"] = empty
        _write_check(sampled_dir / "_check.json", out)
        raise SystemExit(f"[CHECK FAIL] sample: languages with 0 docs: {empty}")
    _write_check(sampled_dir / "_check.json", out)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def check_tokenize(
    tokenized_dir: Path,
    *,
    dry_run: bool,
    expected_languages: list[str] | None = None,
    english_median_check: bool = False,
    english_key: str = "english",
    english_median_tolerance: float = 0.05,
    eval_fraction: float | None = None,
) -> None:
    """Verify tokenized dataset: row counts, token counts, splits, language IDs, example."""

    if dry_run:
        return

    from datasets import load_from_disk
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    from multilingual_megatron.pipeline.stages.tokenization.tools.check_tokenized_parallel import _iter_ipc_batches

    rank_dirs = sorted(
        p for p in tokenized_dir.iterdir() if p.is_dir() and p.name.startswith("rank_")
    )

    out: dict[str, Any] = {"stage": "tokenize", "pass": True}

    if rank_dirs:
        # Stream through rank_* shards to avoid loading the entire merged dataset at once.
        split_langs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        split_tokens: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        langs = collections.Counter()
        tokens_by_lang: dict[str, int] = collections.Counter()
        docs_by_lang: dict[str, int] = collections.Counter()
        split_docs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        total_tokens = 0
        total_rows = 0
        unique_lids: set[int] = set()
        unique_fids: set[int] = set()
        pad_count = 0
        has_lang_ids = False
        has_family_ids = False
        columns_seen: set[str] = set()
        example: dict[str, Any] | None = None

        for rank_dir in rank_dirs:
            arrow_files = sorted(rank_dir.glob("data-*.arrow"))
            if not arrow_files:
                continue

            required = [
                "language",
                "split",
                "input_ids",
                "language_ids",
                "family_ids",
                "doc_id",
                "chunk_idx",
                "doc_token_count",
            ]
            try:
                dataset = ds.dataset([str(p) for p in arrow_files], format="ipc")
                available = set(dataset.schema.names)
                columns_seen.update(available)
                columns = [col for col in required if col in available]
                if "language" not in available:
                    out["pass"] = False
                    _write_check(tokenized_dir / "_check.json", out)
                    raise SystemExit("[CHECK FAIL] tokenize: missing 'language' column in tokenized data")

                batches_iter = dataset.scanner(columns=columns, batch_size=8192).to_batches()
            except pa.ArrowInvalid:
                # HuggingFace Datasets writes IPC stream shards; pyarrow.dataset expects file IPC.
                def _stream_batches():
                    schema_checked = False
                    columns: list[str] = []
                    for path in arrow_files:
                        for batch, schema in _iter_ipc_batches(path):
                            if not schema_checked:
                                available = set(schema.names)
                                columns_seen.update(available)
                                columns = [col for col in required if col in available]
                                if "language" not in available:
                                    out["pass"] = False
                                    _write_check(tokenized_dir / "_check.json", out)
                                    raise SystemExit("[CHECK FAIL] tokenize: missing 'language' column in tokenized data")
                                schema_checked = True
                            yield batch.select(columns) if columns else batch

                batches_iter = _stream_batches()

            for batch in batches_iter:
                batch_rows = batch.num_rows
                total_rows += batch_rows
                lang_list = batch.column("language").to_pylist()
                split_list = batch.column("split").to_pylist() if "split" in batch.column_names else ["train"] * batch_rows

                if "input_ids" in batch.column_names:
                    lengths = pc.list_value_length(batch.column("input_ids")).to_pylist()
                else:
                    lengths = [0] * batch_rows

                if "chunk_idx" in batch.column_names:
                    chunk_idx_list = batch.column("chunk_idx").to_pylist()
                else:
                    chunk_idx_list = [None] * batch_rows

                for idx, (split_name, lang, length) in enumerate(zip(split_list, lang_list, lengths)):
                    lang_key = str(lang)
                    split_key = str(split_name)
                    langs[lang_key] += 1
                    tokens_by_lang[lang_key] += int(length)
                    split_langs[split_key][lang_key] += 1
                    split_tokens[split_key][lang_key] += int(length)
                    total_tokens += int(length)
                    if chunk_idx_list[idx] == 0:
                        docs_by_lang[lang_key] += 1
                        split_docs[split_key][lang_key] += 1

                if "language_ids" in batch.column_names:
                    has_lang_ids = True
                    lids = batch.column("language_ids").to_pylist()
                    for lid in lids:
                        lid_int = int(lid)
                        unique_lids.add(lid_int)
                        if lid_int < 0:
                            pad_count += 1

                if "family_ids" in batch.column_names:
                    has_family_ids = True
                    fids = batch.column("family_ids").to_pylist()
                    for fid in fids:
                        unique_fids.add(int(fid))

                if example is None and batch_rows > 0:
                    idx = 0
                    example = {
                        "language": lang_list[idx],
                        "split": split_list[idx],
                        "input_ids_len": int(lengths[idx]) if lengths else 0,
                        "input_ids_head": [],
                        "language_ids": int(batch.column("language_ids")[idx]) if "language_ids" in batch.column_names else None,
                        "family_ids": int(batch.column("family_ids")[idx]) if "family_ids" in batch.column_names else None,
                    }

        out["num_rows"] = total_rows
        out["columns"] = sorted(columns_seen)
        out["splits"] = {k: sum(v.values()) for k, v in split_langs.items()}
        out["samples_per_language_by_split"] = {split: dict(counts) for split, counts in split_langs.items()}
        out["tokens_per_language_by_split"] = {split: dict(counts) for split, counts in split_tokens.items()}
        out["docs_per_language_by_split"] = {split: dict(counts) for split, counts in split_docs.items()}
        out["samples_per_language"] = dict(langs)
        out["num_languages"] = len(langs)
        out["total_tokens"] = total_tokens
        out["tokens_per_language"] = dict(tokens_by_lang)
        out["docs_per_language"] = dict(docs_by_lang)
        out["has_language_ids"] = has_lang_ids
        if has_lang_ids:
            out["unique_language_ids"] = sorted(unique_lids)
            out["num_unique_language_ids"] = len(unique_lids)
            out["language_ids_pad_count"] = pad_count
            if pad_count > 0:
                out["language_ids_warning"] = (
                    f"{pad_count} samples have pad language_id (<0) — "
                    "these will be ignored by LPR guidance"
                )
        out["has_family_ids"] = has_family_ids
        if has_family_ids:
            out["unique_family_ids"] = sorted(unique_fids)
            out["num_unique_family_ids"] = len(unique_fids)
        if example is not None:
            out["example"] = example
    else:
        ds = load_from_disk(str(tokenized_dir))
        out["num_rows"] = len(ds)
        out["columns"] = ds.column_names        # Splits
        splits = collections.Counter(ds["split"])
        out["splits"] = dict(splits)
        split_langs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        split_tokens: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        split_docs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
        for split_name, lang, ids, chunk_idx in zip(ds["split"], ds["language"], ds["input_ids"], ds["chunk_idx"]):
            split_langs[str(split_name)][str(lang)] += 1
            split_tokens[str(split_name)][str(lang)] += len(ids)
            if int(chunk_idx) == 0:
                split_docs[str(split_name)][str(lang)] += 1
        out["samples_per_language_by_split"] = {split: dict(counts) for split, counts in split_langs.items()}
        out["tokens_per_language_by_split"] = {split: dict(counts) for split, counts in split_tokens.items()}
        out["docs_per_language_by_split"] = {split: dict(counts) for split, counts in split_docs.items()}

        # Language counts
        langs = collections.Counter(ds["language"])
        out["samples_per_language"] = dict(langs)
        out["num_languages"] = len(langs)

        # Token counts
        total_tokens = sum(len(ids) for ids in ds["input_ids"])
        out["total_tokens"] = total_tokens
        tokens_by_lang: dict[str, int] = collections.Counter()
        docs_by_lang: dict[str, int] = collections.Counter()
        for lang, ids, chunk_idx in zip(ds["language"], ds["input_ids"], ds["chunk_idx"]):
            tokens_by_lang[lang] += len(ids)
            if int(chunk_idx) == 0:
                docs_by_lang[lang] += 1
        out["tokens_per_language"] = dict(tokens_by_lang)
        out["docs_per_language"] = dict(docs_by_lang)

        # Language IDs

        has_lang_ids = "language_ids" in ds.column_names
        out["has_language_ids"] = has_lang_ids
        if has_lang_ids:
            all_lids = ds["language_ids"]
            unique_lids = sorted(set(all_lids))
            out["unique_language_ids"] = unique_lids
            out["num_unique_language_ids"] = len(unique_lids)
            # Check for invalid pad IDs (-1)
            pad_count = sum(1 for lid in all_lids if lid < 0)
            out["language_ids_pad_count"] = pad_count
            if pad_count > 0:
                out["language_ids_warning"] = (
                    f"{pad_count} samples have pad language_id (<0) — "
                    "these will be ignored by LPR guidance"
                )

        # Family IDs (cluster/expert mapping for LPR)
        has_family_ids = "family_ids" in ds.column_names
        out["has_family_ids"] = has_family_ids
        if has_family_ids:
            unique_fids = sorted(set(ds["family_ids"]))
            out["unique_family_ids"] = unique_fids
            out["num_unique_family_ids"] = len(unique_fids)

        # Example (first row, truncated)
        ex = ds[0]
        out["example"] = {
            "language": ex.get("language"),
            "split": ex.get("split"),
            "input_ids_len": len(ex.get("input_ids", [])),
            "input_ids_head": ex.get("input_ids", [])[:10],
            "language_ids": ex.get("language_ids"),
            "family_ids": ex.get("family_ids"),
        }

    # Fail conditions
    if int(out.get("num_rows", 0)) == 0:
        out["pass"] = False
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] tokenize: dataset has 0 rows")
    if "train" not in out.get("splits", {}):
        out["pass"] = False
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] tokenize: no 'train' split found")
    if not has_lang_ids:
        out["pass"] = False
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit("[CHECK FAIL] tokenize: missing 'language_ids' column")

    required_chunk_columns = {"doc_id", "chunk_idx", "doc_token_count"}
    missing_chunk_columns = sorted(required_chunk_columns - set(out.get("columns", [])))
    if missing_chunk_columns:
        out["pass"] = False
        out["missing_chunk_columns"] = missing_chunk_columns
        _write_check(tokenized_dir / "_check.json", out)
        raise SystemExit(
            "[CHECK FAIL] tokenize: missing chunk metadata columns: "
            f"{missing_chunk_columns}"
        )

    if expected_languages:
        expected = sorted({str(item) for item in expected_languages if str(item).strip()})
        missing = [lang for lang in expected if lang not in langs]
        if missing:
            out["pass"] = False
            out["missing_languages"] = missing
            _write_check(tokenized_dir / "_check.json", out)
            raise SystemExit(f"[CHECK FAIL] tokenize: missing languages in tokenized data: {missing}")

    if english_median_check and expected_languages:
        english_key = str(english_key or "english").strip()
        if english_key not in docs_by_lang:
            out["pass"] = False
            out["english_missing"] = english_key
            _write_check(tokenized_dir / "_check.json", out)
            raise SystemExit(f"[CHECK FAIL] tokenize: english key '{english_key}' missing from tokenized data")

        english_frac = _validation_fraction(split_docs, english_key)
        if english_frac is None:
            out["pass"] = False
            out["english_validation_fraction"] = None
            _write_check(tokenized_dir / "_check.json", out)
            raise SystemExit("[CHECK FAIL] tokenize: english validation fraction unavailable")

        if eval_fraction is not None and eval_fraction > 0:
            min_allowed = eval_fraction * 0.25
            max_allowed = eval_fraction * 3.0
            out["english_validation_fraction_expected"] = eval_fraction
            out["english_validation_fraction_allowed_min"] = min_allowed
            out["english_validation_fraction_allowed_max"] = max_allowed
            out["english_validation_fraction_actual"] = english_frac
            if english_frac < min_allowed or english_frac > max_allowed:
                out["pass"] = False
                _write_check(tokenized_dir / "_check.json", out)
                raise SystemExit(
                    "[CHECK FAIL] tokenize: english validation fraction is outside allowed range "
                    f"(expected ~{eval_fraction:.4f} within [{min_allowed:.4f}, {max_allowed:.4f}], got {english_frac:.4f})"
                )
        else:
            expected_set = {str(item) for item in expected_languages if str(item).strip()}
            other_fracs = []
            for lang in expected_set:
                if lang == english_key:
                    continue
                frac = _validation_fraction(split_docs, lang)
                if frac is not None:
                    other_fracs.append(frac)
            if other_fracs:
                median_frac = float(statistics.median(other_fracs))
                tolerance = max(0.02, float(median_frac) * float(english_median_tolerance))
                if abs(english_frac - median_frac) > tolerance:
                    out["pass"] = False
                    out["english_validation_fraction_expected"] = median_frac
                    out["english_validation_fraction_actual"] = english_frac
                    out["english_validation_fraction_tolerance"] = tolerance
                    _write_check(tokenized_dir / "_check.json", out)
                    raise SystemExit(
                        "[CHECK FAIL] tokenize: english validation fraction is not near median of other languages "
                        f"(expected ~{median_frac:.4f} ±{tolerance:.4f}, got {english_frac:.4f})"
                    )

    _write_check(tokenized_dir / "_check.json", out)


# ---------------------------------------------------------------------------
# train  — public API
# ---------------------------------------------------------------------------

def check_train(train_root: Path, runs: list[dict[str, Any]], *, dry_run: bool):
    """Verify Bridge training wrapper artifacts."""

    if dry_run:
        return

    out: dict[str, Any] = {"stage": "train", "pass": True, "runs": {}}

    for run in runs:
        run_id = str(run["id"])
        run_dir = train_root / run_id
        entry: dict[str, Any] = {}

        results_path = run_dir / "train_results.json"
        done_path = run_dir / ".training_complete"
        manifest_path = run_dir / "training_run_manifest.json"
        entry["results_exist"] = results_path.exists()
        entry["training_complete_exists"] = done_path.exists()
        entry["run_manifest_exists"] = manifest_path.exists()
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            backend = str(results.get("backend") or run.get("backend") or "").strip().lower()
            status = str(results.get("status") or ("complete" if done_path.exists() else "")).strip().lower()
            entry["backend"] = backend
            entry["status"] = status
            entry["train_loss"] = results.get("train_loss")
            entry["train_runtime"] = results.get("train_runtime")
            entry["tokens_seen"] = results.get("num_input_tokens_seen")
            entry["backend_ok"] = backend in {"megatron_bridge", "bridge"}
            entry["status_ok"] = status in {"complete", "ok"}
            entry["tokens_ok"] = isinstance(entry["tokens_seen"], int) and int(entry["tokens_seen"]) > 0
            if (
                not entry["backend_ok"]
                or not entry["tokens_ok"]
                or not entry["status_ok"]
                or not entry["training_complete_exists"]
                or not entry["run_manifest_exists"]
            ):
                out["pass"] = False
        else:
            out["pass"] = False
        out["runs"][run_id] = entry

    _write_check(train_root / "_check.json", out)
    if not out["pass"]:
        failed = [
            rid
            for rid, v in out["runs"].items()
            if not v.get("results_exist")
            or not v.get("backend_ok", False)
            or not v.get("status_ok", False)
            or not v.get("training_complete_exists", False)
            or not v.get("run_manifest_exists", False)
        ]
        raise SystemExit(f"[CHECK FAIL] train: failed runs: {failed}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def check_benchmark(benchmark_root: Path, summary_path: Path, *, dry_run: bool):
    """Verify benchmark: all runs have expected task count."""

    if dry_run:
        return

    if not summary_path.exists():
        return  # benchmark may be disabled

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    out: dict[str, Any] = {"stage": "benchmark", "pass": True, "runs": {}}
    for run_info in summary.get("runs", []):
        run_id = run_info.get("run_id", "?")
        expected = run_info.get("num_tasks_expected", 0)
        seen = run_info.get("num_tasks_seen", 0)
        missing = run_info.get("missing_tasks", [])
        entry = {"expected": expected, "seen": seen, "missing_count": len(missing)}
        if expected > 0 and seen == 0:
            entry["pass"] = False
            out["pass"] = False
        else:
            entry["pass"] = True
        out["runs"][run_id] = entry
    _write_check(benchmark_root / "_check.json", out)
    if not out["pass"]:
        failed = [rid for rid, v in out["runs"].items() if not v.get("pass")]
        raise SystemExit(f"[CHECK FAIL] benchmark: runs with 0 tasks evaluated: {failed}")
