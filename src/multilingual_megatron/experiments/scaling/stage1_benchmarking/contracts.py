"""Freeze, score, and replay Stage-1 multiple-choice benchmarks.

The workflow intentionally separates three environments:

1. ``freeze`` uses the pinned lm-evaluation-harness to select examples and
   materialize exact text loglikelihood requests.
2. ``score-specs`` freezes the native Megatron checkpoint contracts consumed by
   :mod:`native_megatron_lm_eval` under ``torchrun``.
3. ``aggregate`` replays native scores through lm-eval's task metric code.

This module never exports the custom scaling Switch-MoE model to Hugging Face.
"""


import ast
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.contracts.hashing import (
    canonical_json_sha256 as _payload_sha256,
    sha256_file as _sha256_file,
)

from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import (
    NLL_SUM_RELATIVE_TOLERANCE,
    RAW_NLL_ABS_TOLERANCE,
    REQUEST_MANIFEST_SCHEMA,
    SCORE_RESULT_SCHEMA,
    SCORE_SPEC_SCHEMA,
)


PROFILE_SCHEMA = "stage1_lm_eval_benchmark_profile/v1"
METRICS_SCHEMA = "stage1_lm_eval_benchmark_metrics/v1"
TASK_FRAGMENT_SCHEMA = "stage1_lm_eval_frozen_task_fragment/v1"

OKAPI_DATASET = "jon-tow/okapi_hellaswag"
OKAPI_REVISION = "15a37c9ed3daa92a360f5b0420933a808ca3ee9c"
HELLASWAG_DATASET = "Rowan/hellaswag"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
COPA_DATASET = "aps/super_glue"
COPA_REVISION = "3de24cf8022e94f4ee4b9d55a6f539891524d646"
XNLI_DATASET = "xnli"
XNLI_REVISION = "b8dd5d7af51114dbda02c0e3f6133f332186418e"
BELEBELE_DATASET = "facebook/belebele"
BELEBELE_REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
XSTORYCLOZE_DATASET = "juletxara/xstory_cloze"
XSTORYCLOZE_REVISION = "c4c2d88a1ec8b37fe22166d2a610f272726724b6"
XCOPA_DATASET = "xcopa"
XCOPA_REVISION = "042f78955ba48e6404616762fa6e05e839c3907a"
UHURA_ARC_EASY_DATASET = "masakhane/uhura-arc-easy"
UHURA_ARC_EASY_REVISION = "561efa15460cba0965212b77f09bf6a7f57939a9"
GLOBAL_MMLU_DATASET = "CohereForAI/Global-MMLU-Lite"
GLOBAL_MMLU_REVISION = "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b"

PINNED_BENCHMARK_REVISIONS = {
    "belebele_cloze": BELEBELE_REVISION,
    "global_mmlu_cloze": GLOBAL_MMLU_REVISION,
    "okapi_hellaswag": OKAPI_REVISION,
    "uhura_arc_easy": UHURA_ARC_EASY_REVISION,
    "xcopa": XCOPA_REVISION,
    "xnli": XNLI_REVISION,
    "xstorycloze": XSTORYCLOZE_REVISION,
}


def _benchmark_dataset_pin(entry: Mapping[str, Any]) -> dict[str, str] | None:
    """Return the exact dataset identity selected by a benchmark entry."""

    benchmark = str(entry["id"])
    if benchmark == "okapi_hellaswag" and entry.get("variant") == "original_english":
        return {"repo_id": HELLASWAG_DATASET, "revision": HELLASWAG_REVISION}
    if benchmark == "xcopa" and entry.get("variant") == "original_english":
        return {"repo_id": COPA_DATASET, "revision": COPA_REVISION}
    revision = PINNED_BENCHMARK_REVISIONS.get(benchmark)
    if revision is None:
        return None
    repo_ids = {
        "belebele_cloze": BELEBELE_DATASET,
        "global_mmlu_cloze": GLOBAL_MMLU_DATASET,
        "okapi_hellaswag": OKAPI_DATASET,
        "uhura_arc_easy": UHURA_ARC_EASY_DATASET,
        "xcopa": XCOPA_DATASET,
        "xnli": XNLI_DATASET,
        "xstorycloze": XSTORYCLOZE_DATASET,
    }
    return {"repo_id": repo_ids[benchmark], "revision": revision}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _set_hf_offline_cache(hf_home: Path) -> None:
    """Use only the campaign's pre-populated, pinned dataset cache."""

    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return payload


def _resolve(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _verify_payload_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label}.{field} is not a SHA-256 digest")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    actual = _payload_sha256(unhashed)
    if actual != expected:
        raise ValueError(f"{label} payload hash mismatch: expected {expected}, got {actual}")


def _parse_model_override_value(raw: str) -> Any:
    lower = raw.lower()
    if lower == "null":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _format_model_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    return repr(value)


def reconcile_training_model_overrides(
    source_overrides: Sequence[str],
    training_model: Mapping[str, Any],
    declared_replacements: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Make explicitly declared corrections and prove all model overrides match training."""

    overrides = list(source_overrides)
    model_indices: dict[str, int] = {}
    parsed_values: dict[str, Any] = {}
    for index, override in enumerate(overrides):
        if not override.startswith("model."):
            continue
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError(f"invalid model override: {override!r}")
        field = key.removeprefix("model.")
        if field in model_indices:
            raise ValueError(f"duplicate model override field: {field}")
        model_indices[field] = index
        parsed_values[field] = _parse_model_override_value(raw)

    replacement_receipt: dict[str, dict[str, Any]] = {}
    for field, declared_training_value in declared_replacements.items():
        if not isinstance(field, str) or not field:
            raise ValueError("training model replacement fields must be non-empty strings")
        if field not in model_indices:
            raise ValueError(f"declared replacement has no source override: model.{field}")
        if field not in training_model:
            raise ValueError(f"training run config lacks declared model field: {field}")
        actual_training_value = training_model[field]
        if actual_training_value != declared_training_value:
            raise ValueError(
                f"declared training value drift for model.{field}: "
                f"{declared_training_value!r} != {actual_training_value!r}"
            )
        source_value = parsed_values[field]
        if source_value == actual_training_value:
            raise ValueError(
                f"declared replacement for model.{field} is unnecessary; source already matches"
            )
        overrides[model_indices[field]] = (
            f"model.{field}={_format_model_override_value(actual_training_value)}"
        )
        parsed_values[field] = actual_training_value
        replacement_receipt[field] = {
            "source_evaluation_value": source_value,
            "training_run_value": actual_training_value,
        }

    mismatches: list[str] = []
    for field, expected in parsed_values.items():
        if field not in training_model:
            mismatches.append(f"model.{field}: absent from training run config")
            continue
        actual = training_model[field]
        if actual != expected:
            mismatches.append(f"model.{field}: score={expected!r}, training={actual!r}")
    if mismatches:
        raise ValueError(
            "undeclared score/training model override mismatch: " + "; ".join(mismatches)
        )
    return overrides, replacement_receipt


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    source_path: Path
    source_sha256: str
    payload: Mapping[str, Any]
    repo_root: Path
    harness_root: Path
    bridge_root: Path
    output_root: Path
    hf_home: Path
    seed: int
    examples_per_language: int
    shared_population: int

    @property
    def benchmark_id(self) -> str:
        return str(self.payload["pilot_id"])

    @property
    def benchmarks(self) -> Sequence[Mapping[str, Any]]:
        return self.payload["benchmarks"]

    @property
    def checkpoints(self) -> Sequence[Mapping[str, Any]]:
        return self.payload["checkpoints"]

    @property
    def request_manifest_pilot_id(self) -> str:
        """Legacy manifest identifier of the frozen prompt contract."""

        return str(self.payload.get("request_manifest_pilot_id", self.benchmark_id))
