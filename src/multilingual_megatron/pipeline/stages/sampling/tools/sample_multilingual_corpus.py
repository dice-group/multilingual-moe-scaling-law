#!/usr/bin/env python3
"""Sample a multilingual corpus with explicit language balancing controls."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any


def _load_languages_from_map(language_map_path: Path) -> list[str]:
    payload = json.loads(language_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {language_map_path}")
    langs: list[str] = []
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        items = value.get("languages")
        if not isinstance(items, list):
            continue
        for lang in items:
            if isinstance(lang, str):
                langs.append(lang)
    unique = sorted(set(langs))
    if not unique:
        raise ValueError(f"No languages found in language map: {language_map_path}")
    return unique


def _resolve_languages(
    explicit: str,
    language_map: str,
    max_languages: int,
    seed: int,
) -> list[str]:
    if explicit.strip():
        langs = [item.strip() for item in explicit.split(",") if item.strip()]
        if not langs:
            raise ValueError("No languages parsed from --languages.")
        return sorted(set(langs))

    if not language_map:
        raise ValueError("Provide --languages or --language-map.")
    all_langs = _load_languages_from_map(Path(language_map))
    if max_languages <= 0 or max_languages >= len(all_langs):
        return all_langs
    rng = random.Random(seed)
    return sorted(rng.sample(all_langs, max_languages))


def _read_stream(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("datasets package is required: pip install datasets") from exc

    if args.source_type == "hf":
        return load_dataset(
            args.dataset_name,
            args.dataset_config if args.dataset_config else None,
            split=args.dataset_split,
            streaming=True,
        )
    if args.source_type == "json":
        files = [item.strip() for item in args.data_files.split(",") if item.strip()]
        if not files:
            raise SystemExit("--data-files is required for --source-type json")
        return load_dataset("json", data_files=files, split="train", streaming=True)
    raise SystemExit(f"Unsupported source type: {args.source_type}")


def _extract_language(example: dict[str, Any], candidates: list[str]) -> str | None:
    for key in candidates:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_text(example: dict[str, Any], text_column: str) -> str | None:
    value = example.get(text_column)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _apply_reservoir(
    buf: list[dict[str, str]],
    sample: dict[str, str],
    seen_count: int,
    cap: int,
    rng: random.Random,
) -> None:
    if len(buf) < cap:
        buf.append(sample)
        return
    replace_idx = rng.randint(0, seen_count - 1)
    if replace_idx < cap:
        buf[replace_idx] = sample


def _allocate_alpha_targets(
    seen_counts: dict[str, int],
    total_target: int,
    alpha: float,
    min_per_language: int,
    max_per_language: int,
) -> dict[str, int]:
    weights = {lang: math.pow(float(max(1, count)), alpha) for lang, count in seen_counts.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Unable to compute alpha-smoothing targets: total weight is 0.")

    raw = {lang: (weights[lang] / total_weight) * total_target for lang in seen_counts}
    targets = {lang: max(min_per_language, int(raw[lang])) for lang in seen_counts}

    diff = total_target - sum(targets.values())
    order = sorted(seen_counts.keys(), key=lambda lang: raw[lang] - int(raw[lang]), reverse=True)
    idx = 0
    while diff > 0:
        lang = order[idx % len(order)]
        if targets[lang] < max_per_language:
            targets[lang] += 1
            diff -= 1
        idx += 1
        if idx > total_target * 8:
            raise ValueError("Could not finish alpha target allocation (positive diff).")

    idx = 0
    while diff < 0:
        lang = order[idx % len(order)]
        if targets[lang] > min_per_language:
            targets[lang] -= 1
            diff += 1
        idx += 1
        if idx > total_target * 8:
            raise ValueError("Could not finish alpha target allocation (negative diff).")

    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample multilingual corpus with balance controls.")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-language JSONL shards.")
    parser.add_argument("--source-type", choices=["hf", "json"], default="hf")
    parser.add_argument("--dataset-name", default="c4")
    parser.add_argument("--dataset-config", default="multilingual")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--data-files", default="", help="Comma-separated JSONL files when source-type=json.")
    parser.add_argument("--language-columns", default="lang,language,subset", help="Comma-separated language column candidates.")
    parser.add_argument("--text-column", default="text", help="Text column name.")
    parser.add_argument("--languages", default="", help="Explicit languages (comma-separated).")
    parser.add_argument("--language-map", default="", help="Language map JSON for language selection.")
    parser.add_argument("--max-languages", type=int, default=0, help="Randomly choose this many languages from map.")
    parser.add_argument("--sampling-mode", choices=["equal_documents", "alpha_smoothing"], default="equal_documents")
    parser.add_argument("--per-language-target", type=int, default=200, help="Target docs/language for equal_documents mode.")
    parser.add_argument("--total-target", type=int, default=2400, help="Total docs for alpha_smoothing mode.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha for alpha_smoothing mode.")
    parser.add_argument("--min-per-language", type=int, default=50, help="Min docs/language for alpha mode.")
    parser.add_argument("--max-per-language", type=int, default=100000, help="Max docs/language for alpha mode.")
    parser.add_argument("--max-scan", type=int, default=250000, help="Maximum source rows to scan.")
    parser.add_argument("--reservoir-cap-factor", type=float, default=1.5, help="Per-language reservoir cap multiplier.")
    parser.add_argument("--allow-upsample", action="store_true", help="Upsample with replacement when target > available.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Fail if any selected language has zero collected samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected_langs = _resolve_languages(args.languages, args.language_map, args.max_languages, args.seed)
    lang_set = set(selected_langs)
    lang_columns = [item.strip() for item in args.language_columns.split(",") if item.strip()]
    if not lang_columns:
        raise SystemExit("No language columns configured.")

    if args.sampling_mode == "equal_documents":
        base_cap = max(1, args.per_language_target)
    else:
        # Use average target for alpha mode before precise allocation.
        avg = max(1, int(args.total_target / max(1, len(selected_langs))))
        base_cap = avg
    reservoir_cap = max(1, int(base_cap * args.reservoir_cap_factor))

    rng = random.Random(args.seed)
    stream = _read_stream(args)

    seen_total = 0
    seen_per_lang = {lang: 0 for lang in selected_langs}
    kept_per_lang = {lang: [] for lang in selected_langs}

    for example in stream:
        seen_total += 1
        lang = _extract_language(example, lang_columns)
        if lang not in lang_set:
            if seen_total >= args.max_scan:
                break
            continue
        text = _extract_text(example, args.text_column)
        if text is None:
            if seen_total >= args.max_scan:
                break
            continue
        seen_per_lang[lang] += 1
        sample = {"language": lang, "text": text}
        _apply_reservoir(
            kept_per_lang[lang],
            sample,
            seen_per_lang[lang],
            reservoir_cap,
            rng,
        )
        if seen_total >= args.max_scan:
            break

    if args.sampling_mode == "equal_documents":
        targets = {lang: args.per_language_target for lang in selected_langs}
    else:
        targets = _allocate_alpha_targets(
            seen_counts=seen_per_lang,
            total_target=args.total_target,
            alpha=args.alpha,
            min_per_language=args.min_per_language,
            max_per_language=args.max_per_language,
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files: list[str] = []
    availability_rows: list[dict[str, Any]] = []
    missing_langs: list[str] = []

    for lang in selected_langs:
        target = int(targets[lang])
        pool = kept_per_lang[lang]
        available = len(pool)
        emitted: list[dict[str, str]]
        upsampled = 0
        if available >= target:
            emitted = rng.sample(pool, target)
        elif available > 0 and args.allow_upsample:
            emitted = list(pool)
            while len(emitted) < target:
                emitted.append(pool[rng.randint(0, available - 1)])
                upsampled += 1
        else:
            emitted = list(pool)

        if len(emitted) == 0:
            missing_langs.append(lang)

        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        shard_path = lang_dir / "part_000.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for row in emitted:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        data_files.append(str(shard_path.resolve()))

        availability_rows.append(
            {
                "language": lang,
                "seen_in_scan": int(seen_per_lang[lang]),
                "kept_in_pool": int(available),
                "target": int(target),
                "emitted": int(len(emitted)),
                "shortfall_before_upsample": int(max(0, target - available)),
                "upsampled": int(upsampled),
            }
        )

    manifest = {
        "source": {
            "type": args.source_type,
            "dataset_name": args.dataset_name if args.source_type == "hf" else None,
            "dataset_config": args.dataset_config if args.source_type == "hf" else None,
            "dataset_split": args.dataset_split if args.source_type == "hf" else None,
            "data_files": args.data_files if args.source_type == "json" else None,
            "max_scan": args.max_scan,
        },
        "selection": {
            "languages": selected_langs,
            "sampling_mode": args.sampling_mode,
            "allow_upsample": args.allow_upsample,
            "seed": args.seed,
        },
        "targets": targets,
        "availability": availability_rows,
        "data_files": data_files,
        "seen_total": seen_total,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report_tsv = output_dir / "language_availability.tsv"
    with report_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "language\tseen_in_scan\tkept_in_pool\ttarget\temitted\tshortfall_before_upsample\tupsampled\n"
        )
        for row in availability_rows:
            handle.write(
                f"{row['language']}\t{row['seen_in_scan']}\t{row['kept_in_pool']}\t"
                f"{row['target']}\t{row['emitted']}\t{row['shortfall_before_upsample']}\t"
                f"{row['upsampled']}\n"
            )

    if args.fail_on_missing and missing_langs:
        raise SystemExit(f"Missing languages with zero sampled rows: {missing_langs}")

    print(f"[OK] output_dir={output_dir}")
    print(f"[OK] languages={len(selected_langs)} seen_total={seen_total}")
    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] availability_tsv={report_tsv}")
    print(f"[OK] DATA_FILES_CSV={','.join(data_files)}")
    if missing_langs:
        print(f"[WARN] languages with zero emitted samples: {','.join(missing_langs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
