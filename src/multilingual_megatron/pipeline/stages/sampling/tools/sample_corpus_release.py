#!/usr/bin/env python3
"""Materialize a sampled corpus release from a corpus blend manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
import re
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


def _source_configs(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = inventory.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("source inventory must contain object key 'sources'")
    return {str(key): value for key, value in sources.items() if isinstance(value, dict)}


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


def _load_local_pool(
    *,
    repo_root: Path,
    source_id: str,
    cfg: dict[str, Any],
    languages: set[str],
) -> dict[str, list[dict[str, str]]]:
    language_column = str(cfg.get("language_column", "language")).strip() or "language"
    text_column = str(cfg.get("text_column", "text")).strip() or "text"
    pool: dict[str, list[dict[str, str]]] = {language: [] for language in languages}
    for path in _resolve_source_files(repo_root, source_id, cfg):
        if not path.exists():
            raise FileNotFoundError(f"Missing local_jsonl file for source '{source_id}': {path}")
        for row_idx, row in enumerate(_iter_jsonl(path)):
            language = str(row.get(language_column, "")).strip()
            text = row.get(text_column)
            if language not in languages or not isinstance(text, str) or not text.strip():
                continue
            doc_id = str(row.get("document_id") or row.get("id") or f"{path.name}:{row_idx}")
            pool[language].append({"language": language, "text": text.strip(), "document_id": doc_id})
    return pool


def _write_report_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "language",
        "source_id",
        "domain",
        "target_documents",
        "available_documents",
        "emitted_documents",
        "upsampled_documents",
        "dedup_dropped_documents",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _dedup_key(text: str, mode: str) -> str:
    if mode == "exact_text":
        return text
    if mode == "normalized_text":
        return re.sub(r"\s+", " ", text.strip().lower())
    return ""


def _sample_rows(
    *,
    pool: list[dict[str, str]],
    target: int,
    allow_upsample: bool,
    dedup_mode: str,
    seen_dedup_keys: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, str]], int, int]:
    if target <= 0 or not pool:
        return [], 0, 0
    if dedup_mode in {"exact_text", "normalized_text"}:
        candidates = list(pool)
        rng.shuffle(candidates)
        emitted: list[dict[str, str]] = []
        dedup_dropped = 0
        for item in candidates:
            key = _dedup_key(item["text"], dedup_mode)
            if key in seen_dedup_keys:
                dedup_dropped += 1
                continue
            seen_dedup_keys.add(key)
            emitted.append(item)
            if len(emitted) >= target:
                break
        return emitted, 0, dedup_dropped

    if len(pool) >= target:
        return rng.sample(pool, target), 0, 0
    emitted = list(pool)
    upsampled = 0
    if allow_upsample:
        while len(emitted) < target:
            emitted.append(pool[rng.randint(0, len(pool) - 1)])
            upsampled += 1
    return emitted, upsampled, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample a concrete corpus release.")
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--fail-on-shortfall", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    inventory_path = _repo_path(repo_root, args.source_inventory)
    manifest_path = _repo_path(repo_root, args.corpus_manifest)
    output_dir = _repo_path(repo_root, args.output_dir)
    inventory = _load_json(inventory_path)
    manifest = _load_json(manifest_path)
    sources = _source_configs(inventory)
    plan = manifest.get("plan")
    if not isinstance(plan, list) or not plan:
        raise SystemExit("corpus manifest contains no plan rows")

    languages = {str(row.get("language", "")).strip() for row in plan if isinstance(row, dict)}
    languages.discard("")
    source_ids = {str(row.get("source_id", "")).strip() for row in plan if isinstance(row, dict)}
    source_ids.discard("")
    allow_upsample = bool((manifest.get("sampling") or {}).get("allow_upsample", False))
    dedup_cfg = (manifest.get("sampling") or {}).get("dedup", {})
    dedup_cfg = dedup_cfg if isinstance(dedup_cfg, dict) else {}
    dedup_mode = str(dedup_cfg.get("mode", "report_only")).strip()
    seed = int((manifest.get("sampling") or {}).get("seed", 42) or 42)
    rng = random.Random(seed)
    seen_dedup_keys: set[str] = set()

    pools: dict[str, dict[str, list[dict[str, str]]]] = {}
    for source_id in sorted(source_ids):
        cfg = sources.get(source_id)
        if cfg is None:
            raise SystemExit(f"Unknown source_id in manifest: {source_id}")
        source_type = str(cfg.get("type", "")).strip()
        if source_type not in {"local_jsonl", "external_manifest", "jsonl_manifest", "hf_jsonl_export"}:
            raise SystemExit(
                f"Sampling adapter for source '{source_id}' type '{source_type}' is not implemented yet. "
                "Use manual availability for planning only, or add a source adapter."
            )
        pools[source_id] = _load_local_pool(
            repo_root=repo_root,
            source_id=source_id,
            cfg=cfg,
            languages=languages,
        )

    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            elif child.name in {"manifest.json", "sampling_report.json", "sampling_report.tsv"}:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    emitted_by_language: dict[str, list[dict[str, str]]] = {language: [] for language in sorted(languages)}
    report_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    shortfalls: list[str] = []
    blend_id = str(manifest.get("blend_id", ""))

    for row in plan:
        if not isinstance(row, dict):
            continue
        language = str(row.get("language", "")).strip()
        source_id = str(row.get("source_id", "")).strip()
        target = int(row.get("target_documents", 0) or 0)
        if not language or not source_id or target <= 0:
            continue
        cfg = sources[source_id]
        pool = list(pools[source_id].get(language, []))
        available = len(pool)
        sampled, upsampled, dedup_dropped = _sample_rows(
            pool=pool,
            target=target,
            allow_upsample=allow_upsample,
            dedup_mode=dedup_mode,
            seen_dedup_keys=seen_dedup_keys,
            rng=rng,
        )
        if target > 0 and not sampled:
            missing.append(f"{source_id}:{language}")
        if len(sampled) < target:
            shortfalls.append(f"{source_id}:{language}:{len(sampled)}/{target}")

        for idx, item in enumerate(sampled):
            out = {
                "language": language,
                "text": item["text"],
                "source_id": source_id,
                "source_domain": str(cfg.get("domain", "")),
                "source_role": str(cfg.get("role", "")),
                "source_document_id": item.get("document_id", ""),
                "corpus_blend": blend_id,
                "sample_index": str(idx),
            }
            emitted_by_language[language].append(out)

        status = "ok"
        if not sampled:
            status = "missing"
        elif len(sampled) < target:
            status = "shortfall"
        elif dedup_dropped > 0:
            status = "dedup_filtered"
        elif upsampled > 0:
            status = "upsample"
        report_rows.append(
            {
                "language": language,
                "source_id": source_id,
                "domain": str(cfg.get("domain", "")),
                "target_documents": target,
                "available_documents": available,
                "emitted_documents": len(sampled),
                "upsampled_documents": upsampled,
                "dedup_dropped_documents": dedup_dropped,
                "status": status,
            }
        )

    data_files: list[str] = []
    for language in sorted(emitted_by_language):
        lang_dir = output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        shard_path = lang_dir / "part_000.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for row in emitted_by_language[language]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        data_files.append(str(shard_path.resolve()))

    report_json = output_dir / "sampling_report.json"
    report_tsv = output_dir / "sampling_report.tsv"
    release_manifest = output_dir / "manifest.json"
    summary = {
        "languages": len(emitted_by_language),
        "sources": len(source_ids),
        "target_documents": sum(int(row.get("target_documents", 0) or 0) for row in report_rows),
        "emitted_documents": sum(int(row.get("emitted_documents", 0) or 0) for row in report_rows),
        "upsampled_documents": sum(int(row.get("upsampled_documents", 0) or 0) for row in report_rows),
        "dedup_dropped_documents": sum(int(row.get("dedup_dropped_documents", 0) or 0) for row in report_rows),
        "missing_plan_cells": missing,
        "shortfall_plan_cells": shortfalls,
    }
    report_payload = {
        "schema_version": 1,
        "blend_id": blend_id,
        "corpus_manifest": str(manifest_path),
        "rows": report_rows,
        "summary": summary,
    }
    report_json.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report_tsv(report_tsv, report_rows)
    release_payload = {
        "schema_version": 1,
        "source_type": "corpus_blend",
        "blend_id": blend_id,
        "source_inventory": str(inventory_path),
        "corpus_manifest": str(manifest_path),
        "sampling_report_json": str(report_json.resolve()),
        "sampling_report_tsv": str(report_tsv.resolve()),
        "shard_dir": str(output_dir.resolve()),
        "data_files": data_files,
        "sampling": manifest.get("sampling", {}),
        "summary": summary,
    }
    release_manifest.write_text(json.dumps(release_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.fail_on_missing and missing:
        raise SystemExit(f"Missing sampled cells: {missing}")
    if args.fail_on_shortfall and shortfalls:
        raise SystemExit(f"Sampled cells below target: {shortfalls}")
    print(f"[OK] output_dir={output_dir}")
    print(f"[OK] release_manifest={release_manifest}")
    print(f"[OK] emitted_documents={summary['emitted_documents']} upsampled={summary['upsampled_documents']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
