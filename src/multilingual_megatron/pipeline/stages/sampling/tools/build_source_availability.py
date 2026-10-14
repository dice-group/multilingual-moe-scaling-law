#!/usr/bin/env python3
"""Build source/language availability tables for corpus blend sampling."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import glob
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _first_present(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def _iter_jsonl(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def _source_items(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = inventory.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("source inventory must contain object key 'sources'")
    out: dict[str, dict[str, Any]] = {}
    for source_id, cfg in sources.items():
        if isinstance(source_id, str) and isinstance(cfg, dict):
            out[source_id] = cfg
    if not out:
        raise ValueError("source inventory contains no sources")
    return out


def _resolve_source_files(repo_root: Path, source_id: str, cfg: dict[str, Any]) -> list[Path]:
    files_raw = cfg.get("files")
    globs_raw = cfg.get("file_globs")
    files: list[Path] = []
    if isinstance(files_raw, list):
        files.extend(_repo_path(repo_root, str(item)) for item in files_raw)
    if isinstance(globs_raw, list):
        for item in globs_raw:
            pattern_path = _repo_path(repo_root, str(item))
            files.extend(Path(path) for path in glob.glob(str(pattern_path), recursive=True))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"source '{source_id}' requires non-empty files or file_globs")
    return files


def _scan_local_jsonl(repo_root: Path, source_id: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    language_column = str(cfg.get("language_column", "language")).strip() or "language"
    text_column = str(cfg.get("text_column", "text")).strip() or "text"
    counts: dict[str, dict[str, int]] = {}

    for path in _resolve_source_files(repo_root, source_id, cfg):
        if not path.exists():
            raise FileNotFoundError(f"Missing local_jsonl file for source '{source_id}': {path}")
        for row in _iter_jsonl(path):
            language = str(row.get(language_column, "")).strip()
            text = row.get(text_column)
            if not language or not isinstance(text, str) or not text.strip():
                continue
            item = counts.setdefault(language, {"documents": 0, "chars": 0})
            item["documents"] += 1
            item["chars"] += len(text)

    rows: list[dict[str, Any]] = []
    for language in sorted(counts):
        rows.append(
            {
                "source_id": source_id,
                "source_type": str(cfg.get("type", "")),
                "domain": str(cfg.get("domain", "")),
                "role": str(cfg.get("role", "")),
                "language": language,
                "available_documents": counts[language]["documents"],
                "available_chars": counts[language]["chars"],
                "available_tokens": 0,
                "availability_source": "local_scan",
            }
        )
    return rows


def _availability_file_rows(repo_root: Path, source_id: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_path = cfg.get("availability_file") or cfg.get("availability")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return []
    path = _repo_path(repo_root, raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing availability file for source '{source_id}': {path}")

    suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".json.gz") else path.suffix
    raw_rows: list[dict[str, Any]] = []
    if suffix == ".json":
        payload = _load_json(path)
        items = payload.get("rows", payload.get("data", []))
        if not isinstance(items, list):
            raise ValueError(f"availability JSON must contain rows/data list: {path}")
        raw_rows = [item for item in items if isinstance(item, dict)]
    else:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            raw_rows = [dict(row) for row in reader]

    language_column = str(cfg.get("availability_language_column", cfg.get("language_column", ""))).strip()
    documents_column = str(cfg.get("availability_documents_column", "documents")).strip() or "documents"
    chars_column = str(cfg.get("availability_chars_column", "")).strip()
    tokens_column = str(cfg.get("availability_tokens_column", "tokens")).strip() or "tokens"
    bytes_column = str(cfg.get("availability_bytes_column", "available_bytes")).strip()
    source_id_column = str(cfg.get("availability_source_id_column", "source_id")).strip()
    split_column = str(cfg.get("availability_split_column", "split")).strip()
    split_value = str(cfg.get("availability_split", "")).strip()

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if source_id_column and source_id_column in raw:
            row_source_id = str(raw.get(source_id_column, "")).strip()
            if row_source_id and row_source_id != source_id:
                continue
        if split_value and split_column and str(raw.get(split_column, "")).strip() != split_value:
            continue
        language = (
            str(raw.get(language_column, "")).strip()
            if language_column
            else str(_first_present(raw, ["language", "lang", "subset", "language_code"])).strip()
        )
        if not language:
            continue
        documents = _as_int(_first_present(raw, [documents_column, "documents", "doc_count", "count", "num_docs"]))
        chars = _as_int(_first_present(raw, [chars_column, "chars", "characters"], 0)) if chars_column else 0
        tokens = _as_int(_first_present(raw, [tokens_column, "tokens", "num_tokens", "words"], 0))
        bytes_ = _as_int(_first_present(raw, [bytes_column, "bytes", "size_bytes"], 0)) if bytes_column else 0
        rows.append(
            {
                "source_id": source_id,
                "source_type": str(cfg.get("type", "")),
                "domain": str(cfg.get("domain", "")),
                "role": str(cfg.get("role", "")),
                "language": language,
                "available_documents": documents,
                "available_chars": chars,
                "available_tokens": tokens,
                "available_bytes": bytes_,
                "availability_source": str(path),
            }
        )
    return rows


def _manual_availability(source_id: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    availability = cfg.get("manual_availability")
    if availability is None:
        availability = cfg.get("availability_rows")
    if not isinstance(availability, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in availability:
        if not isinstance(item, dict):
            continue
        language = str(item.get("language", "")).strip()
        if not language:
            continue
        rows.append(
            {
                "source_id": source_id,
                "source_type": str(cfg.get("type", "")),
                "domain": str(cfg.get("domain", "")),
                "role": str(cfg.get("role", "")),
                "language": language,
                "available_documents": int(item.get("available_documents", item.get("documents", 0)) or 0),
                "available_chars": int(item.get("available_chars", item.get("chars", 0)) or 0),
                "available_tokens": int(item.get("available_tokens", item.get("tokens", 0)) or 0),
                "available_bytes": int(item.get("available_bytes", item.get("bytes", 0)) or 0),
                "availability_source": "manual",
            }
        )
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id",
        "source_type",
        "domain",
        "role",
        "language",
        "available_documents",
        "available_chars",
        "available_tokens",
        "available_bytes",
        "availability_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source availability for corpus blend sampling.")
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--corpus-blend", default="", help="Optional blend file; used to select source ids.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-ids", default="", help="Optional comma-separated source ids to scan.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    inventory_path = _repo_path(repo_root, args.source_inventory)
    inventory = _load_json(inventory_path)
    sources = _source_items(inventory)
    selected = [item.strip() for item in args.source_ids.split(",") if item.strip()]
    if not selected and str(args.corpus_blend).strip():
        blend_path = _repo_path(repo_root, str(args.corpus_blend))
        blend = _load_json(blend_path)
        blend_sources = blend.get("sources", [])
        if not isinstance(blend_sources, list):
            raise SystemExit(f"corpus blend sources must be a list: {blend_path}")
        for entry in blend_sources:
            if isinstance(entry, dict):
                source_id = str(entry.get("source_id", "")).strip()
                if source_id:
                    selected.append(source_id)
    selected_ids = set(selected) if selected else set(sources)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for source_id in sorted(selected_ids):
        cfg = sources.get(source_id)
        if cfg is None:
            raise SystemExit(f"Unknown source_id in --source-ids: {source_id}")
        source_type = str(cfg.get("type", "")).strip()
        manual_rows = _manual_availability(source_id, cfg)
        if manual_rows:
            rows.extend(manual_rows)
            continue
        file_rows = _availability_file_rows(repo_root, source_id, cfg)
        if file_rows:
            rows.extend(file_rows)
            continue
        if source_type in {"local_jsonl", "external_manifest", "jsonl_manifest", "hf_jsonl_export"} and (
            isinstance(cfg.get("files"), list) or isinstance(cfg.get("file_globs"), list)
        ):
            rows.extend(_scan_local_jsonl(repo_root, source_id, cfg))
            continue
        skipped.append({"source_id": source_id, "reason": f"no availability adapter for type '{source_type}'"})

    rows = sorted(rows, key=lambda row: (str(row["source_id"]), str(row["language"])))
    output_json = _repo_path(repo_root, args.output_json)
    output_tsv = _repo_path(repo_root, args.output_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_inventory": str(inventory_path),
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "sources_with_rows": len({str(row["source_id"]) for row in rows}),
            "languages_with_rows": len({str(row["language"]) for row in rows}),
            "rows": len(rows),
            "skipped_sources": len(skipped),
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(output_tsv, rows)
    print(f"[OK] availability_json={output_json}")
    print(f"[OK] availability_tsv={output_tsv}")
    print(f"[OK] rows={len(rows)} skipped_sources={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
