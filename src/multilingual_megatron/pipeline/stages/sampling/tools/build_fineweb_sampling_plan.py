#!/usr/bin/env python3
"""Build a fair per-language FineWeb sampling plan from availability stats."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _load_languages_from_map(language_map_path: Path) -> list[str]:
    payload = json.loads(language_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {language_map_path}")
    langs: list[str] = []
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        for lang in value.get("languages", []):
            if isinstance(lang, str):
                langs.append(lang)
    out = sorted(set(langs))
    if not out:
        raise ValueError(f"No languages found in language map: {language_map_path}")
    return out


def _resolve_languages(explicit: str, language_map_path: str, max_languages: int, seed: int) -> list[str]:
    if explicit.strip():
        langs = [item.strip() for item in explicit.split(",") if item.strip()]
        if not langs:
            raise ValueError("No languages parsed from --languages")
        return sorted(set(langs))
    if not language_map_path:
        raise ValueError("Provide --languages or --language-map")
    langs = _load_languages_from_map(Path(language_map_path))
    if max_languages > 0 and max_languages < len(langs):
        rng = random.Random(seed)
        langs = sorted(rng.sample(langs, max_languages))
    return langs


def _allocate_weighted_targets(
        available: dict[str, int],
        weights: dict[str, float],
        total_target: int,
        *,
        allow_upsample: bool,
) -> dict[str, int]:
    if total_target <= 0:
        raise ValueError("max_total_docs must be > 0 for weighted sampling policies.")

    positive = {lang: int(max(0, count)) for lang, count in available.items() if int(max(0, count)) > 0}
    if not positive:
        raise ValueError("None of the selected languages have positive document availability.")

    capacity_total = sum(positive.values())
    effective_total = int(total_target) if allow_upsample else int(min(total_target, capacity_total))
    if effective_total <= 0:
        return {lang: 0 for lang in available}

    total_weight = sum(float(max(0.0, weights.get(lang, 0.0))) for lang in positive)
    if total_weight > 0:
        raw = {
            lang: (effective_total * float(max(0.0, weights.get(lang, 0.0))) / float(total_weight))
            for lang in positive
        }
    else:
        raise ValueError("None of the selected languages have positive document availability.")

    targets = {lang: int(raw[lang]) for lang in positive}
    assigned = sum(targets.values())

    # largest remainder to match the exact requested total.
    order = sorted(
        positive.keys(),
        key=lambda lang: (raw[lang] - targets[lang], float(max(0.0, weights.get(lang, 0.0)))),
        reverse=True,
    )
    idx = 0
    guard = 0
    max_steps = max(10000, effective_total * 10)

    # cycle through sorted order with biggest remainder, add +1 until total docs is reached,
    while assigned < effective_total and order:
        lang = order[idx % len(order)]
        if allow_upsample or targets[lang] < positive[lang]: # only add when upsampled is allowed or current targe is still below available docs
            targets[lang] += 1
            assigned += 1
        idx += 1
        guard += 1
        if guard > max_steps:
            raise RuntimeError("Failed to allocate weighted targets within safety bound.")

    out = {lang: 0 for lang in available}
    out.update(targets)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fair FineWeb sampling plan.")
    parser.add_argument("--availability-csv", required=True, help="FineWeb availability CSV path.")
    parser.add_argument("--output-plan-csv", required=True, help="Output plan CSV path.")
    parser.add_argument("--output-report-json", required=True, help="Output summary JSON path.")
    parser.add_argument("--output-availability-tsv", required=True, help="Output availability TSV path.")
    parser.add_argument("--languages", default="", help="Explicit comma-separated languages.")
    parser.add_argument("--language-map", default="", help="Language map path for language selection.")
    parser.add_argument("--max-languages", type=int, default=0, help="Randomly sample N languages from map.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for max-languages selection.")
    parser.add_argument("--target-docs-per-language", type=int, default=10000)
    parser.add_argument(
        "--fairness-policy",
        choices=["equal_documents", "alpha_smoothing"],
        default="equal_documents",
        help="Sampling policy for selected languages.",
    )
    parser.add_argument(
        "--max-total-docs",
        type=int,
        default=0,
        help=(
            "Global cap of planned documents across all selected languages. "
            "Used by alpha_smoothing; if omitted there, falls back to "
            "target_docs_per_language * num_languages."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Smoothing exponent for alpha_smoothing policy (lower => more upweighting of lower-resource languages).",
    )
    parser.add_argument(
        "--auto-adjust-target-to-min-available",
        action="store_true",
        help="Set target to min available docs among selected languages.",
    )
    parser.add_argument(
        "--allow-upsample",
        action="store_true",
        help="Keep target above availability and mark upsample requirement.",
    )
    parser.add_argument("--split", default="train", help="Availability split to use.")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail when any selected language has zero available documents.",
    )
    args = parser.parse_args()

    languages = _resolve_languages(args.languages, args.language_map, args.max_languages, args.seed)
    df = pd.read_csv(args.availability_csv)
    required = {"subset", "split", "documents"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required availability columns: {missing}")

    df = df[df["split"] == args.split].copy()
    df["documents"] = pd.to_numeric(df["documents"], errors="coerce").fillna(0).astype(int)
    docs_map = df.groupby("subset")["documents"].max().to_dict()

    available = {lang: int(docs_map.get(lang, 0)) for lang in languages}
    positive = [count for count in available.values() if count > 0]
    if not positive:
        raise SystemExit("None of the selected languages have positive document availability.")

    target_docs = int(args.target_docs_per_language)
    if args.auto_adjust_target_to_min_available and args.fairness_policy == "equal_documents":
        target_docs = min(target_docs, min(positive))

    requested_total_docs = int(args.max_total_docs)
    effective_total_docs = 0
    if args.fairness_policy == "alpha_smoothing":
        if args.alpha <= 0:
            raise SystemExit("alpha must be > 0 when fairness_policy=alpha_smoothing")
        if requested_total_docs <= 0:
            requested_total_docs = target_docs * len(languages)
        #
        # Match existing smoothing behavior: use language token mass (words) for weights; fall back to documents.
        words_map: dict[str, float] = {}
        if "words" in df.columns:
            words_df = df[["subset", "words"]].copy()
            words_df["words"] = pd.to_numeric(words_df["words"], errors="coerce").fillna(0.0)
            words_map = words_df.groupby("subset")["words"].max().to_dict()
        alpha_weights = {}
        for lang in languages:
            avail_docs = int(available.get(lang, 0))
            if avail_docs <= 0:
                raise ValueError(f"No documents available in language: {lang}")
            mass = float(words_map.get(lang, 0.0))
            if mass <= 0.0:
                mass = float(avail_docs)
            alpha_weights[lang] = float(mass ** float(args.alpha))
        targets = _allocate_weighted_targets(
            available=available,
            weights=alpha_weights,
            total_target=requested_total_docs,
            allow_upsample=bool(args.allow_upsample),
        )
        effective_total_docs = int(sum(targets.values()))

    rows = []
    total_planned = 0
    total_available = 0
    total_upsample = 0
    missing_langs = []
    for lang in languages:
        avail = int(available[lang])
        total_available += avail
        target = target_docs if args.fairness_policy == "equal_documents" else int(targets.get(lang, 0))
        if avail <= 0:
            sampled = 0
            upsample = target if args.allow_upsample else 0
            status = "missing"
            missing_langs.append(lang)
        else:
            if args.allow_upsample:
                sampled = target
                upsample = max(0, target - avail)
                status = "ok_with_upsample" if upsample > 0 else "ok"
            else:
                sampled = min(target, avail)
                upsample = 0
                status = "capped_by_availability" if sampled < target else "ok"
        total_planned += sampled
        total_upsample += upsample
        ratio = (sampled / avail) if avail > 0 else None
        rows.append(
            {
                "subset": lang,
                "documents": int(sampled),
                "available_documents": int(avail),
                "target_documents": int(target),
                "upsample_needed": int(upsample),
                "sampling_ratio_vs_available": ratio,
                "status": status,
            }
        )

    plan_path = Path(args.output_plan_csv)
    report_path = Path(args.output_report_json)
    tsv_path = Path(args.output_availability_tsv)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    plan_df = pd.DataFrame(rows)
    plan_df.to_csv(plan_path, index=False)
    plan_df.to_csv(tsv_path, index=False, sep="\t")

    report = {
        "languages": languages,
        "num_languages": len(languages),
        "split": args.split,
        "fairness_policy": args.fairness_policy,
        "alpha": float(args.alpha) if args.fairness_policy == "alpha_smoothing" else 0.0,
        "target_docs_per_language": target_docs,
        "max_total_docs_requested": int(requested_total_docs) if args.fairness_policy == "alpha_smoothing" else 0,
        "max_total_docs_effective": int(effective_total_docs) if args.fairness_policy == "alpha_smoothing" else 0,
        "total_planned_documents": total_planned,
        "total_available_documents": total_available,
        "total_upsample_needed": total_upsample,
        "allow_upsample": bool(args.allow_upsample),
        "missing_languages": missing_langs,
        "plan_csv": str(plan_path.resolve()),
        "availability_tsv": str(tsv_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK] plan_csv={plan_path}")
    print(f"[OK] report_json={report_path}")
    print(f"[OK] availability_tsv={tsv_path}")
    print(f"[OK] languages={len(languages)} total_planned_docs={total_planned}")
    if missing_langs:
        print(f"[WARN] missing_languages={','.join(missing_langs)}")
        if args.fail_on_missing:
            raise SystemExit(f"Missing languages in availability CSV: {missing_langs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
