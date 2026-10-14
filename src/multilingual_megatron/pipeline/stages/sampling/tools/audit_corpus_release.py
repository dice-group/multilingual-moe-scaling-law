#!/usr/bin/env python3
"""Audit sampled corpus release shards by language, source, and domain."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
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


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["language", "source_id", "source_domain", "documents", "chars"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a sampled corpus release.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fail-on-empty", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = _repo_path(repo_root, args.manifest)
    manifest = _load_json(manifest_path)
    files_raw = manifest.get("data_files")
    if not isinstance(files_raw, list) or not files_raw:
        raise SystemExit(f"manifest has no data_files: {manifest_path}")

    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    total_docs = 0
    total_chars = 0
    empty_files: list[str] = []
    for raw_file in files_raw:
        path = Path(str(raw_file))
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise SystemExit(f"Missing data file listed in manifest: {path}")
        file_docs = 0
        for row in _iter_jsonl(path):
            language = str(row.get("language", path.parent.name)).strip() or path.parent.name
            source_id = str(row.get("source_id", "unknown")).strip() or "unknown"
            domain = str(row.get("source_domain", row.get("domain", ""))).strip()
            text = row.get("text", "")
            chars = len(text) if isinstance(text, str) else 0
            item = counts.setdefault((language, source_id, domain), {"documents": 0, "chars": 0})
            item["documents"] += 1
            item["chars"] += chars
            total_docs += 1
            total_chars += chars
            file_docs += 1
        if file_docs == 0:
            empty_files.append(str(path))

    rows: list[dict[str, Any]] = []
    for (language, source_id, domain), values in sorted(counts.items()):
        rows.append(
            {
                "language": language,
                "source_id": source_id,
                "source_domain": domain,
                "documents": values["documents"],
                "chars": values["chars"],
            }
        )

    sampling_report_path = str(manifest.get("sampling_report_json", "")).strip()
    sampling_status_counts: dict[str, int] = {}
    sampling_summary: dict[str, Any] = {}
    if sampling_report_path:
        report_path = Path(sampling_report_path)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        if report_path.exists():
            report = _load_json(report_path)
            report_rows = report.get("rows", [])
            if isinstance(report_rows, list):
                for item in report_rows:
                    if isinstance(item, dict):
                        status = str(item.get("status", "")).strip() or "unknown"
                        sampling_status_counts[status] = sampling_status_counts.get(status, 0) + 1
            summary = report.get("summary", {})
            if isinstance(summary, dict):
                sampling_summary = summary

    bad_statuses = {
        status: count
        for status, count in sampling_status_counts.items()
        if status in {"missing", "shortfall"}
    }
    expected_docs = int(sampling_summary.get("emitted_documents", total_docs) or 0)
    pass_check = total_docs > 0 and not empty_files and not bad_statuses and expected_docs == total_docs
    output_json = _repo_path(repo_root, args.output_json)
    output_tsv = _repo_path(repo_root, args.output_tsv)
    payload = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "pass": pass_check,
        "rows": rows,
        "summary": {
            "documents": total_docs,
            "chars": total_chars,
            "languages": len({row["language"] for row in rows}),
            "sources": len({row["source_id"] for row in rows}),
            "domains": len({row["source_domain"] for row in rows}),
            "empty_files": empty_files,
            "dedup_mode": ((manifest.get("sampling") or {}).get("dedup") or {}).get("mode", "not_recorded"),
            "decontamination_mode": ((manifest.get("sampling") or {}).get("decontamination") or {}).get("mode", "not_recorded"),
            "sampling_status_counts": sampling_status_counts,
            "sampling_summary": sampling_summary,
            "bad_sampling_status_counts": bad_statuses,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(output_tsv, rows)
    if args.fail_on_empty and not pass_check:
        raise SystemExit(
            f"Corpus audit failed: total_docs={total_docs} empty_files={empty_files} "
            f"bad_sampling_status_counts={bad_statuses} expected_docs={expected_docs}"
        )
    print(f"[OK] audit_json={output_json}")
    print(f"[OK] documents={total_docs} languages={payload['summary']['languages']} sources={payload['summary']['sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
