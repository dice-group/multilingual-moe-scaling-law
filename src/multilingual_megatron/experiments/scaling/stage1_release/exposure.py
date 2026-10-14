"""Reconstruct exact language exposure from retained training blend indices."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class Stage1DataError(ValueError):
    """Raised when an input or result violates the analysis contract."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage1DataError(f"Expected JSON object: {path}")
    return payload


def _language_by_prefix(manifest: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for language, prefixes in manifest["prefixes_by_language"].items():
        for prefix in prefixes:
            resolved = str(Path(prefix).resolve())
            if resolved in mapping:
                raise Stage1DataError(f"Duplicate tier-manifest prefix: {resolved}")
            mapping[resolved] = str(language)
    return mapping


def _resolve_training_inputs(
    evaluation_manifest_path: Path,
) -> tuple[Path, Path | None, Path | None, dict[str, Any]]:
    evaluation = _load_json(evaluation_manifest_path)
    checkpoint_root = Path(evaluation["checkpoint"]["root"]).resolve()
    run_root = checkpoint_root.parents[2]
    training_manifest_path = run_root / "training_run_manifest.json"
    training = _load_json(training_manifest_path)
    if training.get("status") != "complete":
        raise Stage1DataError(f"Incomplete training run: {training_manifest_path}")
    cache_root = checkpoint_root.parent / "dataset_cache"
    indices = sorted(cache_root.glob("*-BlendedDataset-train-dataset_index.npy"))
    descriptions = sorted(cache_root.glob("*-BlendedDataset-train-description.txt"))
    if not indices and not descriptions:
        return training_manifest_path, None, None, training
    if len(indices) != 1 or len(descriptions) != 1:
        raise Stage1DataError(f"Expected one blend index/description under {cache_root}")
    return training_manifest_path, indices[0], descriptions[0], training


def _exact_exposures(
    *,
    index_path: Path,
    description_path: Path,
    tier_manifest_path: Path,
    end_samples: int,
    sequence_length: int,
) -> dict[str, float]:
    import numpy as np

    description = _load_json(description_path)
    prefix_to_language = _language_by_prefix(_load_json(tier_manifest_path))
    dataset_paths = [
        str(Path(item["dataset_path"]).resolve()) for item in description["datasets"]
    ]
    if set(dataset_paths) != set(prefix_to_language):
        missing = sorted(set(prefix_to_language) - set(dataset_paths))
        extra = sorted(set(dataset_paths) - set(prefix_to_language))
        raise Stage1DataError(
            f"Blend/tier mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    dataset_index = np.load(index_path, mmap_mode="r")
    if dataset_index.ndim != 1 or not 0 < end_samples <= dataset_index.size:
        raise Stage1DataError(
            f"Invalid cutoff {end_samples} for blend shape {dataset_index.shape}: {index_path}"
        )
    selected = np.asarray(dataset_index[:end_samples], dtype=np.int64)
    if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= len(dataset_paths)):
        raise Stage1DataError(f"Invalid dataset ordinal in {index_path}")
    counts = np.bincount(selected, minlength=len(dataset_paths))
    exposures = {
        prefix_to_language[prefix]: float(counts[index] * sequence_length)
        for index, prefix in enumerate(dataset_paths)
    }
    if int(sum(exposures.values())) != end_samples * sequence_length:
        raise Stage1DataError(f"Exposure sum mismatch for {index_path}")
    return exposures
