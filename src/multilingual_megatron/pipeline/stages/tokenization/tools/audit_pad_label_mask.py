#!/usr/bin/env python3
"""Audit tokenized dataset padding and label masking invariants."""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_split(dataset_path: Path, split: str):
    try:
        from datasets import concatenate_datasets, load_from_disk
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit("datasets package is required: pip install datasets") from exc

    ds = load_from_disk(str(dataset_path))
    if isinstance(ds, dict):
        if split == "all":
            return concatenate_datasets([ds[key] for key in sorted(ds.keys())])
        if split not in ds:
            raise SystemExit(f"Split '{split}' not found in {dataset_path}")
        return ds[split]
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pad-label mask and language metadata.")
    parser.add_argument("--dataset-path", required=True, help="Path to Dataset or DatasetDict.")
    parser.add_argument("--split", default="train", help="Dataset split for DatasetDict input.")
    parser.add_argument("--pad-label-id", type=int, default=-100, help="Expected label at pad positions.")
    parser.add_argument("--max-rows", type=int, default=256, help="Rows to scan (0 means full split).")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on any violation.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {dataset_path}")

    ds = _load_split(dataset_path, args.split)
    required_columns = ["input_ids", "attention_mask", "labels", "language_ids", "family_ids"]
    missing_columns = [col for col in required_columns if col not in ds.column_names]
    if missing_columns:
        raise SystemExit(f"Missing required columns: {missing_columns}")

    limit = len(ds) if args.max_rows <= 0 else min(len(ds), args.max_rows)
    violations = 0
    all_pad_language = 0
    nontrivial_masks = 0

    for idx in range(limit):
        row = ds[idx]
        input_ids = row["input_ids"]
        attn = row["attention_mask"]
        labels = row["labels"]

        if not (len(input_ids) == len(attn) == len(labels)):
            violations += 1
            continue

        saw_mask_zero = False
        for token_id, mask, label in zip(input_ids, attn, labels):
            if int(mask) == 0:
                saw_mask_zero = True
                if int(label) != int(args.pad_label_id):
                    violations += 1
            elif int(mask) == 1 and int(label) != int(token_id):
                # Labels should mirror token ids at attended positions for this pipeline.
                violations += 1

        if saw_mask_zero:
            nontrivial_masks += 1

        if int(row["language_ids"]) < 0 or int(row["family_ids"]) < 0:
            all_pad_language += 1

    print(f"[AUDIT] dataset={dataset_path} split={args.split} rows_scanned={limit}")
    print(f"[AUDIT] nontrivial_attention_masks={nontrivial_masks}")
    print(f"[AUDIT] invalid_language_metadata_rows={all_pad_language}")
    print(f"[AUDIT] violations={violations}")

    if args.strict and (violations > 0 or all_pad_language == limit):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
