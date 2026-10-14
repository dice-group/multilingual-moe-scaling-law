"""Open every A/B-v2 train and validation prefix with Megatron's real reader."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


class DatasetSmokeError(RuntimeError):
    """Raised when a Megatron indexed dataset cannot be consumed."""


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetSmokeError(f"manifest is not an object: {path}")
    return payload


def _payload_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_smoke(tokenized_root: Path, output: Path) -> dict[str, Any]:
    # Some cluster images expose the NVRx namespace package without version
    # metadata. Megatron only needs a valid package version during import here.
    try:
        import nvidia_resiliency_ext as nvrx

        if not hasattr(nvrx, "__version__"):
            nvrx.__version__ = "0.6.0"
    except ModuleNotFoundError:
        pass

    from megatron.core.datasets.indexed_dataset import IndexedDataset

    tokenized_root = tokenized_root.resolve()
    report = _load(tokenized_root / "tokenization_report.json")
    tiers = report.get("tiers")
    if report.get("pass") is not True or not isinstance(tiers, dict):
        raise DatasetSmokeError("tokenization report is not passing or has no tiers")

    manifests: list[tuple[str, Path, str]] = []
    for tier in ("K1", "K12", "K50", "K200"):
        raw = tiers.get(tier)
        if not isinstance(raw, dict):
            raise DatasetSmokeError(f"tokenization report lacks {tier}")
        manifests.append((tier, tokenized_root / str(raw["path"]), "train"))
    manifests.append(
        (
            "global_validation",
            tokenized_root / "validation_megatron_data_manifest.json",
            "validation",
        )
    )

    opened: dict[str, dict[str, Any]] = {}
    manifest_results: dict[str, dict[str, Any]] = {}
    for label, manifest_path, split in manifests:
        manifest = _load(manifest_path)
        split_payload = manifest.get("splits", {}).get(split)
        if not isinstance(split_payload, dict):
            raise DatasetSmokeError(f"{manifest_path} lacks the {split} split")
        prefixes = split_payload.get("prefixes")
        if not isinstance(prefixes, list) or not prefixes:
            raise DatasetSmokeError(f"{manifest_path} has no {split} prefixes")
        languages = manifest.get("languages")
        if not isinstance(languages, list) or len(languages) != len(prefixes):
            raise DatasetSmokeError(f"{manifest_path} language/prefix coverage differs")

        document_count = 0
        for raw_prefix in prefixes:
            prefix = str(raw_prefix)
            cached = opened.get(prefix)
            if cached is None:
                if not IndexedDataset.exists(prefix):
                    raise DatasetSmokeError(f"Megatron dataset does not exist: {prefix}")
                dataset = IndexedDataset(prefix, multimodal=False, mmap=True)
                sequences = len(dataset)
                if sequences <= 0:
                    raise DatasetSmokeError(f"Megatron dataset is empty: {prefix}")
                first = dataset.get(0, length=1)
                last = dataset.get(sequences - 1, length=1)
                if getattr(first, "size", 0) != 1 or getattr(last, "size", 0) != 1:
                    raise DatasetSmokeError(f"Megatron could not read boundary tokens: {prefix}")
                cached = {
                    "sequences": sequences,
                    "dtype": str(dataset.index.dtype),
                    "first_token": int(first[0]),
                    "last_token": int(last[0]),
                }
                opened[prefix] = cached
                # ``first`` and ``last`` reference the dataset's mmap-backed
                # buffer.  Release them before destroying the dataset so its
                # destructor can close the mmap and file descriptor for every
                # one of the 400 train/validation prefixes.
                del first
                del last
                del dataset
                gc.collect()
            document_count += int(cached["sequences"])

        expected_documents = split_payload.get("total_documents")
        if expected_documents is not None and document_count != int(expected_documents):
            raise DatasetSmokeError(
                f"{label} document count differs: {document_count} != {expected_documents}"
            )
        manifest_results[label] = {
            "manifest": str(manifest_path),
            "language_count": len(languages),
            "prefix_count": len(prefixes),
            "documents_opened": document_count,
            "pass": True,
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "pass": True,
        "reader": "megatron.core.datasets.indexed_dataset.IndexedDataset",
        "boundary_reads_per_unique_prefix": 2,
        "unique_prefixes_opened": len(opened),
        "manifests": manifest_results,
    }
    result["report_payload_sha256"] = _payload_hash(result)
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.tokenized_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
