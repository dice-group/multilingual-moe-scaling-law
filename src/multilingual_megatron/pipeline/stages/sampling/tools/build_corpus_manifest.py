#!/usr/bin/env python3
"""Build a concrete per-source/per-language corpus sampling manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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


def _load_languages_from_map(path: Path) -> list[str]:
    payload = _load_json(path)
    languages: list[str] = []
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        items = value.get("languages")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and item.strip():
                languages.append(item.strip())
    unique = sorted(set(languages))
    if not unique:
        raise ValueError(f"No languages found in language map: {path}")
    return unique


def _resolve_languages(blend: dict[str, Any], language_map: Path, explicit: str, seed: int) -> list[str]:
    if explicit.strip():
        return sorted({item.strip() for item in explicit.split(",") if item.strip()})
    policy = blend.get("language_policy", {})
    policy = policy if isinstance(policy, dict) else {}
    raw_policy_languages = policy.get("languages")
    if isinstance(raw_policy_languages, list) and raw_policy_languages:
        return sorted({str(item).strip() for item in raw_policy_languages if str(item).strip()})

    languages = _load_languages_from_map(language_map)
    max_languages = int(policy.get("max_languages", 0) or 0)
    if max_languages > 0 and max_languages < len(languages):
        rng = random.Random(seed)
        languages = sorted(rng.sample(languages, max_languages))
    return languages


def _allocate_int(total: int, keys: list[str], weights: dict[str, float], *, minimum: int = 0) -> dict[str, int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    if not keys:
        return {}
    base_min = minimum * len(keys)
    if base_min > total:
        raise ValueError(f"minimum allocation {minimum} across {len(keys)} keys exceeds total {total}")
    remaining = total - base_min
    weight_sum = sum(max(0.0, weights.get(key, 0.0)) for key in keys)
    if weight_sum <= 0.0:
        weights = {key: 1.0 for key in keys}
        weight_sum = float(len(keys))

    raw = {key: (max(0.0, weights[key]) / weight_sum) * remaining for key in keys}
    out = {key: minimum + int(math.floor(raw[key])) for key in keys}
    leftover = total - sum(out.values())
    order = sorted(keys, key=lambda key: (raw[key] - math.floor(raw[key]), key), reverse=True)
    idx = 0
    while leftover > 0:
        out[order[idx % len(order)]] += 1
        leftover -= 1
        idx += 1
    return out


def _source_configs(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = inventory.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("source inventory must contain object key 'sources'")
    return {str(key): value for key, value in sources.items() if isinstance(value, dict)}


def _availability_index(availability: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = availability.get("rows")
    if not isinstance(rows, list):
        raise ValueError("availability JSON must contain list key 'rows'")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id", "")).strip()
        language = str(row.get("language", "")).strip()
        if not source_id or not language:
            continue
        index[(source_id, language)] = row
    return index


def _availability_by_language(availability_idx: dict[tuple[str, str], dict[str, Any]], source_ids: list[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    source_set = set(source_ids)
    for (source_id, language), row in availability_idx.items():
        if source_id not in source_set:
            continue
        totals[language] = totals.get(language, 0) + int(row.get("available_documents", 0) or 0)
    return totals


def _language_allocation_weights(
    *,
    languages: list[str],
    policy: dict[str, Any],
    availability_totals: dict[str, int],
) -> dict[str, float]:
    allocation = str(policy.get("allocation", "equal_documents")).strip()
    if allocation == "equal_documents":
        return {language: 1.0 for language in languages}
    if allocation == "alpha_smoothing":
        alpha = float(policy.get("alpha", 0.3) or 0.3)
        return {language: math.pow(float(max(1, availability_totals.get(language, 0))), alpha) for language in languages}
    if allocation == "availability_proportional":
        return {language: float(max(1, availability_totals.get(language, 0))) for language in languages}
    if allocation == "explicit_weights":
        raw_weights = policy.get("weights", {})
        if not isinstance(raw_weights, dict):
            raise SystemExit("language_policy.weights must be an object for explicit_weights allocation")
        return {language: float(raw_weights.get(language, 0.0) or 0.0) for language in languages}
    raise SystemExit(
        "language_policy.allocation must be one of "
        "equal_documents, alpha_smoothing, availability_proportional, explicit_weights"
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    source_documents: dict[str, int] = {}
    domain_documents: dict[str, int] = {}
    language_documents: dict[str, int] = {}
    for row in rows:
        target = int(row.get("target_documents", 0) or 0)
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        source_id = str(row.get("source_id", ""))
        domain = str(row.get("domain", ""))
        language = str(row.get("language", ""))
        source_documents[source_id] = source_documents.get(source_id, 0) + target
        domain_documents[domain] = domain_documents.get(domain, 0) + target
        language_documents[language] = language_documents.get(language, 0) + target
    planned = sum(int(row["target_documents"]) for row in rows)
    return {
        "planned_documents": planned,
        "requested_documents": sum(int(row.get("requested_documents", row["target_documents"])) for row in rows),
        "available_documents_for_plan": sum(int(row["available_documents"]) for row in rows),
        "status_counts": status_counts,
        "source_documents": dict(sorted(source_documents.items())),
        "domain_documents": dict(sorted(domain_documents.items())),
        "language_documents": dict(sorted(language_documents.items())),
    }


def _write_plan_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "language",
        "source_id",
        "domain",
        "role",
        "requested_documents",
        "target_documents",
        "available_documents",
        "available_tokens",
        "sampling_ratio_vs_available",
        "upsample_needed",
        "capped_documents",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corpus blend manifest.")
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--corpus-blend", required=True)
    parser.add_argument("--availability-json", required=True)
    parser.add_argument("--language-map", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-plan-tsv", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--languages", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    inventory_path = _repo_path(repo_root, args.source_inventory)
    blend_path = _repo_path(repo_root, args.corpus_blend)
    availability_path = _repo_path(repo_root, args.availability_json)
    language_map_path = _repo_path(repo_root, args.language_map)
    inventory = _load_json(inventory_path)
    blend = _load_json(blend_path)
    availability = _load_json(availability_path)

    sampling_cfg = blend.get("sampling", {})
    sampling_cfg = sampling_cfg if isinstance(sampling_cfg, dict) else {}
    seed = int(sampling_cfg.get("seed", 42) or 42)
    allow_upsample = bool(sampling_cfg.get("allow_upsample", False))
    max_upsample_factor = float(sampling_cfg.get("max_upsample_factor", 0.0) or 0.0)
    languages = _resolve_languages(blend, language_map_path, args.languages, seed)
    language_policy = blend.get("language_policy", {})
    language_policy = language_policy if isinstance(language_policy, dict) else {}

    target_cfg = blend.get("target", {})
    target_cfg = target_cfg if isinstance(target_cfg, dict) else {}
    unit = str(target_cfg.get("unit", "documents")).strip()
    if unit != "documents":
        raise SystemExit("Only target.unit='documents' is implemented in the first corpus_blend version")
    total_documents = int(target_cfg.get("total", 0) or 0)
    if total_documents <= 0:
        raise SystemExit("corpus blend target.total must be positive")

    source_entries = blend.get("sources")
    if not isinstance(source_entries, list) or not source_entries:
        raise SystemExit("corpus blend requires non-empty sources list")
    source_ids: list[str] = []
    source_weights: dict[str, float] = {}
    for entry in source_entries:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id", "")).strip()
        if not source_id:
            continue
        source_ids.append(source_id)
        source_weights[source_id] = float(entry.get("weight", 1.0) or 0.0)
    if not source_ids:
        raise SystemExit("corpus blend sources did not resolve any source_id values")

    sources = _source_configs(inventory)
    missing_sources = sorted(set(source_ids) - set(sources))
    if missing_sources:
        raise SystemExit(f"corpus blend references unknown sources: {missing_sources}")

    availability_idx = _availability_index(availability)
    availability_totals = _availability_by_language(availability_idx, source_ids)
    per_language_min = int(target_cfg.get("min_documents_per_language", 0) or 0)
    language_weights = _language_allocation_weights(
        languages=languages,
        policy=language_policy,
        availability_totals=availability_totals,
    )
    lang_targets = _allocate_int(total_documents, languages, language_weights, minimum=per_language_min)

    rows: list[dict[str, Any]] = []
    for language in languages:
        source_targets = _allocate_int(lang_targets[language], source_ids, source_weights)
        for source_id in source_ids:
            requested_documents = int(source_targets[source_id])
            target_documents = requested_documents
            cfg = sources[source_id]
            availability_row = availability_idx.get((source_id, language), {})
            available_documents = int(availability_row.get("available_documents", 0) or 0)
            available_tokens = int(availability_row.get("available_tokens", 0) or 0)
            capped_documents = 0
            if max_upsample_factor > 0.0 and available_documents > 0:
                max_allowed = int(math.floor(available_documents * max_upsample_factor))
                max_allowed = max(available_documents, max_allowed)
                if target_documents > max_allowed:
                    capped_documents = target_documents - max_allowed
                    target_documents = max_allowed
            if target_documents <= 0:
                status = "zero_target"
            elif available_documents <= 0:
                status = "missing"
            elif capped_documents > 0:
                status = "capped"
            elif target_documents > available_documents:
                status = "upsample" if allow_upsample else "shortfall"
            else:
                status = "ok"
            ratio = 0.0 if available_documents <= 0 else target_documents / available_documents
            rows.append(
                {
                    "language": language,
                    "source_id": source_id,
                    "source_type": str(cfg.get("type", "")),
                    "domain": str(cfg.get("domain", "")),
                    "role": str(cfg.get("role", "")),
                    "requested_documents": requested_documents,
                    "target_documents": target_documents,
                    "available_documents": available_documents,
                    "available_tokens": available_tokens,
                    "sampling_ratio_vs_available": round(ratio, 6),
                    "upsample_needed": max(0, target_documents - available_documents),
                    "capped_documents": capped_documents,
                    "status": status,
                }
            )

    row_summary = _summarize_rows(rows)

    output_json = _repo_path(repo_root, args.output_json)
    output_tsv = _repo_path(repo_root, args.output_plan_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "blend_id": str(blend.get("blend_id", blend_path.stem)),
        "source_inventory": str(inventory_path),
        "corpus_blend": str(blend_path),
        "availability_json": str(availability_path),
        "language_map": str(language_map_path),
        "languages": languages,
        "target": {
            "unit": "documents",
            "total": total_documents,
            "per_language": lang_targets,
        },
        "language_policy": language_policy,
        "sampling": {
            "allow_upsample": allow_upsample,
            "max_upsample_factor": max_upsample_factor,
            "seed": seed,
            "dedup": sampling_cfg.get("dedup", {"mode": "report_only"}),
            "decontamination": sampling_cfg.get("decontamination", {"mode": "report_only"}),
        },
        "plan": rows,
        "summary": {
            "languages": len(languages),
            "sources": len(source_ids),
            **row_summary,
        },
    }
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_plan_tsv(output_tsv, rows)
    print(f"[OK] corpus_manifest={output_json}")
    print(f"[OK] sampling_plan_tsv={output_tsv}")
    print(f"[OK] planned_documents={manifest['summary']['planned_documents']} status_counts={row_summary['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
