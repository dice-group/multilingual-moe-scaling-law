#!/usr/bin/env python3
"""Materialize and verify the external metadata in a FineWeb A/B-v2 lock."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    canonical_json_sha256,
    sha256_file,
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _download_verified(*, url: str, expected_sha256: str, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and sha256_file(output_path) == expected_sha256:
        return {
            "path": str(output_path.resolve()),
            "url": url,
            "sha256": expected_sha256,
        }

    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        observed = sha256_file(partial)
        if observed != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {url}: expected {expected_sha256}, observed {observed}"
            )
        partial.replace(output_path)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "path": str(output_path.resolve()),
        "url": url,
        "sha256": expected_sha256,
    }


def materialize_source_lock(*, source_lock_path: Path, output_dir: Path) -> dict[str, Any]:
    source_lock = _load_object(source_lock_path)
    non_english = source_lock.get("non_english_corpus")
    if not isinstance(non_english, dict):
        raise ValueError("source lock is missing non_english_corpus")
    inventory = non_english.get("language_inventory")
    rehydration = non_english.get("rehydration")
    if not isinstance(inventory, dict) or not isinstance(rehydration, dict):
        raise ValueError("source lock is missing inventory or rehydration metadata")

    required = {
        "inventory": (
            str(inventory.get("url", "")),
            str(inventory.get("sha256", "")),
            output_dir / "fineweb2-language-distribution.csv",
        ),
        "rehydration_weights": (
            str(rehydration.get("weights_url", "")),
            str(rehydration.get("weights_sha256", "")),
            output_dir / "fineweb2-up-to-10-reps.json",
        ),
    }
    artifacts: dict[str, Any] = {}
    for name, (url, expected_sha256, output_path) in required.items():
        if not url.startswith(("https://", "file://")) or len(expected_sha256) != 64:
            raise ValueError(f"source lock has an invalid {name} URL or SHA-256")
        artifacts[name] = _download_verified(
            url=url,
            expected_sha256=expected_sha256,
            output_path=output_path,
        )

    source_lock_sha256 = sha256_file(source_lock_path)
    manifest = {
        "schema_version": 1,
        "source_lock_id": source_lock.get("source_lock_id"),
        "source_lock_path": str(source_lock_path.resolve()),
        "source_lock_sha256": source_lock_sha256,
        "dataset_revisions": {
            "fineweb2": str(non_english.get("dataset_revision", "")),
            "fineweb": str(source_lock.get("english_corpus", {}).get("dataset_revision", "")),
        },
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = materialize_source_lock(
        source_lock_path=Path(args.source_lock).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
