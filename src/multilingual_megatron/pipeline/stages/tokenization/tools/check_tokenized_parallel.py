import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.ipc as ipc

"""
I Seperated checks for data into parallel execution to match slurm requirements for tokenized dataset checks for rank_* datasets.
Made this since tokenization verificaiton is importatn for me and we wand to keep the pipeline modulare and maintinable

The script is designed to run as a SLURM array:
- mode=worker: each task scans a subset of rank_* dirs and writes a partial JSON.
- mode=merge: merges partial JSONs, runs validations, and writes _check.json.
"""


def _discover_rank_dirs(tokenized_dir: Path) -> list[Path]:
    return sorted(p for p in tokenized_dir.iterdir() if p.is_dir() and p.name.startswith("rank_"))


def _assign_ranks(rank_dirs: list[Path], rank: int, world_size: int) -> list[Path]:
    return [d for i, d in enumerate(rank_dirs) if i % world_size == rank]


def _iter_ipc_batches(path: Path):
    source = pa.memory_map(str(path))
    try:
        reader = ipc.open_file(source)
        for idx in range(reader.num_record_batches):
            yield reader.get_batch(idx), reader.schema
        return
    except pa.ArrowInvalid as exc:
        first_err = exc
    source = pa.memory_map(str(path))
    try:
        reader = ipc.open_stream(source)
        for batch in reader:
            yield batch, reader.schema
    except pa.ArrowInvalid as exc:
        raise RuntimeError(f"Failed to read IPC data from {path}: {exc}") from first_err


def _scan_rank(rank_dir: Path) -> dict[str, Any]:
    arrow_files = sorted(rank_dir.glob("data-*.arrow"))
    if not arrow_files:
        return {
            "rank": rank_dir.name,
            "num_rows": 0,
            "columns": [],
            "splits": {},
            "samples_per_language": {},
            "tokens_per_language": {},
        "docs_per_language": {},
            "samples_per_language_by_split": {},
            "tokens_per_language_by_split": {},
        "docs_per_language_by_split": {},
            "has_language_ids": False,
            "language_ids_pad_count": 0,
            "has_family_ids": False,
            "unique_language_ids": [],
            "unique_family_ids": [],
            "example": None,
        }

    available: set[str] = set()
    columns: list[str] = []
    schema_checked = False

    split_langs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    split_tokens: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    split_docs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
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
    example = None

    try:
        dataset = ds.dataset([str(p) for p in arrow_files], format="ipc")
        available = set(dataset.schema.names)
        has_lang_ids = "language_ids" in available
        has_family_ids = "family_ids" in available
        columns = [
            c
            for c in ["language", "split", "input_ids", "language_ids", "family_ids", "doc_id", "chunk_idx", "doc_token_count"]
            if c in available
        ]
        if "language" not in available:
            raise RuntimeError(f"Missing 'language' column in {rank_dir}")
        scanner = dataset.scanner(columns=columns, batch_size=8192)
        batches_iter = scanner.to_batches()
    except pa.ArrowInvalid:
        # Fallback for IPC stream-formatted shards.
        def _stream_batches():
            nonlocal available, columns, schema_checked, has_lang_ids, has_family_ids
            for path in arrow_files:
                for batch, schema in _iter_ipc_batches(path):
                    if not schema_checked:
                        available = set(schema.names)
                        has_lang_ids = "language_ids" in available
                        has_family_ids = "family_ids" in available
                        columns = [
                            c
                            for c in [
                                "language",
                                "split",
                                "input_ids",
                                "language_ids",
                                "family_ids",
                                "doc_id",
                                "chunk_idx",
                                "doc_token_count",
                            ]
                            if c in available
                        ]
                        if "language" not in available:
                            raise RuntimeError(f"Missing 'language' column in {rank_dir}")
                        schema_checked = True
                    if columns:
                        batch = batch.select(columns)
                    yield batch

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

        if has_lang_ids:
            lids = batch.column("language_ids").to_pylist()
            for lid in lids:
                lid_int = int(lid)
                unique_lids.add(lid_int)
                if lid_int < 0:
                    pad_count += 1

        if has_family_ids:
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
                "language_ids": int(batch.column("language_ids")[idx]) if has_lang_ids else None,
                "family_ids": int(batch.column("family_ids")[idx]) if has_family_ids else None,
            }

    return {
        "rank": rank_dir.name,
        "num_rows": total_rows,
        "columns": sorted(available),
        "splits": {k: sum(v.values()) for k, v in split_langs.items()},
        "samples_per_language": dict(langs),
        "tokens_per_language": dict(tokens_by_lang),
        "docs_per_language": dict(docs_by_lang),
        "samples_per_language_by_split": {split: dict(counts) for split, counts in split_langs.items()},
        "tokens_per_language_by_split": {split: dict(counts) for split, counts in split_tokens.items()},
        "docs_per_language_by_split": {split: dict(counts) for split, counts in split_docs.items()},
        "has_language_ids": has_lang_ids,
        "language_ids_pad_count": pad_count,
        "has_family_ids": has_family_ids,
        "unique_language_ids": sorted(unique_lids),
        "unique_family_ids": sorted(unique_fids),
        "example": example,
    }


def _merge_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stage": "tokenize",
        "pass": True,
        "num_rows": 0,
        "columns": [],
        "splits": {},
        "samples_per_language": {},
        "tokens_per_language": {},
        "docs_per_language": {},
        "samples_per_language_by_split": {},
        "tokens_per_language_by_split": {},
        "docs_per_language_by_split": {},
        "has_language_ids": False,
        "language_ids_pad_count": 0,
        "has_family_ids": False,
        "unique_language_ids": [],
        "unique_family_ids": [],
        "example": None,
    }
    columns: set[str] = set()
    splits = collections.Counter()
    langs = collections.Counter()
    tokens_by_lang = collections.Counter()
    docs_by_lang = collections.Counter()
    split_langs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    split_tokens: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    split_docs: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    unique_lids: set[int] = set()
    unique_fids: set[int] = set()

    for part in parts:
        out["num_rows"] += int(part.get("num_rows", 0))
        columns.update(part.get("columns", []))
        splits.update(part.get("splits", {}))
        langs.update(part.get("samples_per_language", {}))
        tokens_by_lang.update(part.get("tokens_per_language", {}))
        docs_by_lang.update(part.get("docs_per_language", {}))
        for split, counts in part.get("samples_per_language_by_split", {}).items():
            split_langs[split].update(counts)
        for split, counts in part.get("tokens_per_language_by_split", {}).items():
            split_tokens[split].update(counts)
        for split, counts in part.get("docs_per_language_by_split", {}).items():
            split_docs[split].update(counts)
        if part.get("has_language_ids"):
            out["has_language_ids"] = True
            unique_lids.update(part.get("unique_language_ids", []))
            out["language_ids_pad_count"] += int(part.get("language_ids_pad_count", 0))
        if part.get("has_family_ids"):
            out["has_family_ids"] = True
            unique_fids.update(part.get("unique_family_ids", []))
        if out["example"] is None and part.get("example"):
            out["example"] = part.get("example")

    out["columns"] = sorted(columns)
    out["splits"] = dict(splits)
    out["samples_per_language"] = dict(langs)
    out["tokens_per_language"] = dict(tokens_by_lang)
    out["docs_per_language"] = dict(docs_by_lang)
    out["samples_per_language_by_split"] = {k: dict(v) for k, v in split_langs.items()}
    out["tokens_per_language_by_split"] = {k: dict(v) for k, v in split_tokens.items()}
    out["docs_per_language_by_split"] = {k: dict(v) for k, v in split_docs.items()}
    out["unique_language_ids"] = sorted(unique_lids) if out["has_language_ids"] else []
    out["num_unique_language_ids"] = len(out["unique_language_ids"]) if out["has_language_ids"] else 0
    out["unique_family_ids"] = sorted(unique_fids) if out["has_family_ids"] else []
    out["num_unique_family_ids"] = len(out["unique_family_ids"]) if out["has_family_ids"] else 0
    out["num_languages"] = len(langs)
    out["total_tokens"] = int(sum(tokens_by_lang.values()))
    return out


def _validation_fraction(tokens_by_split: dict[str, dict[str, int]], lang: str) -> float | None:
    val_tokens = int(tokens_by_split.get("validation", {}).get(lang, 0))
    train_tokens = int(tokens_by_split.get("train", {}).get(lang, 0))
    total = val_tokens + train_tokens
    if total <= 0:
        return None
    return val_tokens / total


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel tokenized dataset check")
    ap.add_argument("--tokenized-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["worker", "merge"], required=True)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--world-size", type=int, default=None)
    ap.add_argument("--expected-languages", default="")
    ap.add_argument("--english-key", default="english")
    ap.add_argument("--english-median-check", action="store_true")
    ap.add_argument("--english-median-tolerance", type=float, default=0.05)
    ap.add_argument("--eval-fraction", type=float, default=None)
    args = ap.parse_args()

    tokenized_dir = Path(args.tokenized_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "worker":
        rank = args.rank
        world_size = args.world_size
        if rank is None:
            rank = int(os.getenv("SLURM_ARRAY_TASK_ID") or 0)
        if world_size is None:
            world_size = int(os.getenv("SLURM_ARRAY_TASK_COUNT") or 1)
        rank_dirs = _discover_rank_dirs(tokenized_dir)
        assigned = _assign_ranks(rank_dirs, rank, world_size)
        parts = [_scan_rank(rd) for rd in assigned]
        out = _merge_parts(parts) if parts else {
            "rank": f"part_{rank}",
            "num_rows": 0,
            "columns": [],
            "splits": {},
            "samples_per_language": {},
            "tokens_per_language": {},
        "docs_per_language": {},
            "samples_per_language_by_split": {},
            "tokens_per_language_by_split": {},
        "docs_per_language_by_split": {},
            "has_language_ids": False,
            "language_ids_pad_count": 0,
            "has_family_ids": False,
            "unique_language_ids": [],
            "unique_family_ids": [],
            "example": None,
        }
        part_path = output_dir / f"part_{rank:05d}.json"
        part_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OK] wrote {part_path}")
        return 0

    parts = sorted(output_dir.glob("part_*.json"))
    if not parts:
        raise SystemExit(f"No part_*.json found in {output_dir}")
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in parts]
    out = _merge_parts(payloads)

    required_chunk_columns = {"doc_id", "chunk_idx", "doc_token_count"}
    missing_chunk_columns = sorted(required_chunk_columns - set(out.get("columns", [])))
    if missing_chunk_columns:
        out["pass"] = False
        out["missing_chunk_columns"] = missing_chunk_columns
        (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit(
            "[CHECK FAIL] tokenize: missing chunk metadata columns: "
            f"{missing_chunk_columns}"
        )

    expected = [x.strip() for x in args.expected_languages.split(",") if x.strip()]
    if expected:
        missing = [lang for lang in sorted(set(expected)) if lang not in out.get("samples_per_language", {})]
        if missing:
            out["pass"] = False
            out["missing_languages"] = missing
            (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
            raise SystemExit(f"[CHECK FAIL] tokenize: missing languages in tokenized data: {missing}")

    if args.english_median_check and expected:
        english_key = args.english_key.strip() or "english"
        docs_by_split = out.get("docs_per_language_by_split", {})
        if english_key not in out.get("docs_per_language", {}):
            out["pass"] = False
            out["english_missing"] = english_key
            (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
            raise SystemExit(f"[CHECK FAIL] tokenize: english key '{english_key}' missing from tokenized data")

        english_frac = _validation_fraction(docs_by_split, english_key)
        if english_frac is None:
            out["pass"] = False
            out["english_validation_fraction"] = None
            (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
            raise SystemExit("[CHECK FAIL] tokenize: english validation fraction unavailable")

        eval_fraction = args.eval_fraction if args.eval_fraction and args.eval_fraction > 0 else None
        if eval_fraction is not None:
            min_allowed = eval_fraction * 0.25
            max_allowed = eval_fraction * 3.0
            out["english_validation_fraction_expected"] = eval_fraction
            out["english_validation_fraction_allowed_min"] = min_allowed
            out["english_validation_fraction_allowed_max"] = max_allowed
            out["english_validation_fraction_actual"] = english_frac
            if english_frac < min_allowed or english_frac > max_allowed:
                out["pass"] = False
                (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
                raise SystemExit(
                    "[CHECK FAIL] tokenize: english validation fraction is outside allowed range "
                    f"(expected ~{eval_fraction:.4f} within [{min_allowed:.4f}, {max_allowed:.4f}], got {english_frac:.4f})"
                )
        else:
            other_fracs = []
            for lang in expected:
                if lang == english_key:
                    continue
                frac = _validation_fraction(docs_by_split, lang)
                if frac is not None:
                    other_fracs.append(frac)
            if other_fracs:
                import statistics

                median_frac = float(statistics.median(other_fracs))
                tolerance = max(0.02, float(median_frac) * float(args.english_median_tolerance))
                if abs(english_frac - median_frac) > tolerance:
                    out["pass"] = False
                    out["english_validation_fraction_expected"] = median_frac
                    out["english_validation_fraction_actual"] = english_frac
                    out["english_validation_fraction_tolerance"] = tolerance
                    (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
                    raise SystemExit(
                        "[CHECK FAIL] tokenize: english validation fraction is not near median of other languages "
                        f"(expected ~{median_frac:.4f} ±{tolerance:.4f}, got {english_frac:.4f})"
                    )

    (tokenized_dir / "_check.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {tokenized_dir / '_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
