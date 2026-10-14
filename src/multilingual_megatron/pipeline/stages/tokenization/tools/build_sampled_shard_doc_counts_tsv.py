import argparse
import gzip
from pathlib import Path


def _count_nonempty_lines(path: Path) -> int:
    opener = gzip.open if path.name.endswith(".gz") else open
    count = 0
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a shard_doc_counts.tsv from sampled shard outputs.")
    parser.add_argument("--shard_dir", type=Path, required=True, help="Directory containing <lang>/*.jsonl(.gz) shards.")
    parser.add_argument("--output_tsv", type=Path, required=True, help="Path to write the TSV (language, shard, doc_count).")
    parser.add_argument("--languages", type=str, default="", help="Optional comma-separated language directory allowlist.")
    args = parser.parse_args()

    shard_dir = args.shard_dir
    if not shard_dir.exists():
        raise SystemExit(f"Missing shard_dir: {shard_dir}")

    allow = {lang.strip() for lang in str(args.languages).split(",") if lang.strip()} or None

    rows: list[tuple[str, str, int]] = []
    for language_dir in sorted(p for p in shard_dir.iterdir() if p.is_dir()):
        language = language_dir.name
        if allow is not None and language not in allow:
            continue
        for shard_path in sorted(language_dir.iterdir()):
            if not shard_path.is_file():
                continue
            if not (shard_path.name.endswith(".jsonl") or shard_path.name.endswith(".jsonl.gz")):
                continue
            rel = shard_path.relative_to(shard_dir).as_posix()
            rows.append((language, rel, _count_nonempty_lines(shard_path)))

    if not rows:
        raise SystemExit(f"No shard files found under {shard_dir}")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8") as handle:
        handle.write("language\tshard\tdoc_count\n")
        for language, rel, doc_count in rows:
            handle.write(f"{language}\t{rel}\t{doc_count}\n")

    print(f"[OK] wrote {len(rows)} shard rows to {args.output_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

