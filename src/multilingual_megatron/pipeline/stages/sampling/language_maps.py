#!/usr/bin/env python3
"""Generate frozen scaling language maps from the 200-language grouping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TIERS = {
    "k001": 1,
    "k016": 16,
    "k050": 50,
    "k200": 200,
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _language_rows(source_map: dict[str, Any], anchor_language: str) -> list[dict[str, str]]:
    by_language: dict[str, dict[str, str]] = {}
    for group_id, raw_group in source_map.items():
        if not isinstance(raw_group, dict):
            continue
        metadata = raw_group.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        for language in raw_group.get("languages", []):
            if not isinstance(language, str) or not language.strip():
                continue
            lang = language.strip()
            raw_meta = metadata.get(lang, {})
            raw_meta = raw_meta if isinstance(raw_meta, dict) else {}
            by_language[lang] = {
                "language": lang,
                "source_group": str(group_id),
                "group": str(raw_group.get("group", "")),
                "family": str(raw_meta.get("family", raw_group.get("group", ""))),
                "name": str(raw_meta.get("name", "")),
                "resource_availability": str(raw_meta.get("resource_availability", "")),
            }
    if anchor_language not in by_language:
        raise ValueError(f"Anchor language {anchor_language!r} is not present in source map.")
    ordered = [by_language[anchor_language]]
    ordered.extend(by_language[lang] for lang in sorted(by_language) if lang != anchor_language)
    return ordered


def _map_payload(rows: list[dict[str, str]], *, tier: str, source_map: Path, anchor_language: str) -> dict[str, Any]:
    languages = [row["language"] for row in rows]
    metadata = {
        row["language"]: {
            "name": row["name"],
            "family": row["family"],
            "resource_availability": row["resource_availability"],
            "subset": row["language"],
        }
        for row in rows
    }
    return {
        "0": {
            "group": tier,
            "languages": languages,
            "metadata": metadata,
            "selection_policy": {
                "source_map": str(source_map),
                "anchor_language": anchor_language,
                "ordering": "anchor first, then lexicographic language code",
            },
        }
    }


def generate_language_maps(
    *,
    source_map_path: Path,
    output_dir: Path,
    language_sets_tsv: Path,
    anchor_language: str = "eng_Latn",
) -> dict[str, Path]:
    source_map = _load_json(source_map_path)
    rows = _language_rows(source_map, anchor_language)
    if len(rows) < max(TIERS.values()):
        raise ValueError(f"Need at least {max(TIERS.values())} languages, found {len(rows)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    selected_by_tier: dict[str, set[str]] = {}
    for tier, count in TIERS.items():
        selected = rows[:count]
        selected_by_tier[tier] = {row["language"] for row in selected}
        output_path = output_dir / f"scaling_fw2_{tier}.json"
        output_path.write_text(
            json.dumps(
                _map_payload(selected, tier=tier, source_map=source_map_path, anchor_language=anchor_language),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written[tier] = output_path

    language_sets_tsv.parent.mkdir(parents=True, exist_ok=True)
    with language_sets_tsv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["language", "name", "family", "resource_availability", "source_group", *TIERS.keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for tier in TIERS:
                payload[tier] = "1" if row["language"] in selected_by_tier[tier] else "0"
            writer.writerow({field: payload.get(field, "") for field in fieldnames})
    written["language_sets"] = language_sets_tsv
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate frozen scaling FineWeb2 language maps.")
    parser.add_argument("--source-map", default="configs/language_groupings/200_languages_grouping.json")
    parser.add_argument("--output-dir", default="configs/language_groupings")
    parser.add_argument("--language-sets-tsv", default="configs/language_groupings/scaling_fw2_language_sets.tsv")
    parser.add_argument("--anchor-language", default="eng_Latn")
    args = parser.parse_args()

    written = generate_language_maps(
        source_map_path=Path(args.source_map),
        output_dir=Path(args.output_dir),
        language_sets_tsv=Path(args.language_sets_tsv),
        anchor_language=str(args.anchor_language).strip() or "eng_Latn",
    )
    for key, path in sorted(written.items()):
        print(f"[OK] {key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
