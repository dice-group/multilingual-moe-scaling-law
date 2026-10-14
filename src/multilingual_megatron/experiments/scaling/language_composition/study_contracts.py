"""Small frozen constants shared by data and training orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__)).resolve()
DESIGN_ID = "stage1_language_composition_v1"
SELECTION_IDENTITY = (
    "20671b754c795dc37fbaca5608a19c1175477a6d2c12b4f8a398ee991dc30319"
)
COMPACT_SET_SHA256 = (
    "552d8faf4ce09a5999741c1632c9685560d3a53b7336202d081abb43d634d5f9"
)
DIVERSE_SET_SHA256 = (
    "4491080d0787499774ebe9a7c0132459b43c5c3de6c6430853da19297bba5591"
)
SELECTION_MANIFEST = Path(
    "result_analysis/language_composition_selection_v2/selection_manifest.json"
)
ARM_GROUPINGS = {
    "compact": Path("configs/language_groupings/scaling_fw2_v2_k012_compact.json"),
    "diverse": Path("configs/language_groupings/scaling_fw2_v2_k012_diverse.json"),
}
K200_GROUPING = Path("configs/language_groupings/scaling_fw2_v2_k200.json")
REFERENCE_GROUPING = Path("configs/language_groupings/scaling_fw2_v2_k012.json")
EXCLUDED_LANGUAGES = ("hyw_Armn", "epo_Latn", "ido_Latn", "srd_Latn")
EXPERTS = (1, 16, 32)
SEEDS = (1234, 5678, 91011)
SMOKE_EXPERTS = (1, 32)
SMOKE_TOKENS = 134_217_728
SMOKE_GLOBAL_BATCH_SIZE = 32
SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS = 12
SMOKE_MAX_STEP_TIME_CV_PCT = 5.0
HORIZONS = (
    500_000_000,
    1_000_000_000,
    2_000_000_000,
    4_000_000_000,
    8_000_000_000,
    16_000_000_000,
)


class CompositionStudyError(RuntimeError):
    """Raised when a frozen language-composition contract cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionStudyError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CompositionStudyError(f"{label} must be a JSON object: {path}")
    return payload


def language_set_sha256(languages: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{language}\n" for language in languages).encode("utf-8")
    ).hexdigest()


def grouping_languages(path: Path, *, expected_count: int) -> tuple[str, ...]:
    payload = read_object(path.resolve(), label="language grouping")
    try:
        raw = payload["0"]["languages"]
    except (KeyError, TypeError) as exc:
        raise CompositionStudyError(f"grouping lacks 0.languages: {path}") from exc
    if (
        not isinstance(raw, list)
        or len(raw) != expected_count
        or len(raw) != len(set(raw))
        or not all(isinstance(value, str) and value for value in raw)
    ):
        raise CompositionStudyError(
            f"grouping must contain {expected_count} unique language IDs: {path}"
        )
    languages = tuple(raw)
    declared = payload.get("language_set_sha256")
    observed = language_set_sha256(languages)
    if declared is not None and declared != observed:
        raise CompositionStudyError(f"grouping language hash drifted: {path}")
    return languages


def verify_selection_authority() -> dict[str, Any]:
    manifest_path = (REPO_ROOT / SELECTION_MANIFEST).resolve()
    manifest = read_object(manifest_path, label="selection manifest")
    expected_receipt = manifest.get("receipt_sha256")
    candidate = dict(manifest)
    candidate.pop("receipt_sha256", None)
    if canonical_sha256(candidate) != expected_receipt:
        raise CompositionStudyError("selection manifest self-hash mismatch")
    if (
        manifest.get("status") != "passed"
        or manifest.get("selection_identity_sha256") != SELECTION_IDENTITY
    ):
        raise CompositionStudyError("selection authority is not the frozen passed result")
    observed: dict[str, tuple[str, ...]] = {}
    expected_sets = {
        "compact": COMPACT_SET_SHA256,
        "diverse": DIVERSE_SET_SHA256,
    }
    for arm, relative in ARM_GROUPINGS.items():
        path = (REPO_ROOT / relative).resolve()
        languages = grouping_languages(path, expected_count=12)
        if language_set_sha256(languages) != expected_sets[arm]:
            raise CompositionStudyError(f"{arm} grouping differs from the promoted set")
        file_record = manifest.get("files", {}).get(f"{arm}_grouping")
        if (
            not isinstance(file_record, Mapping)
            or file_record.get("path") != relative.as_posix()
            or file_record.get("sha256") != sha256_file(path)
        ):
            raise CompositionStudyError(f"{arm} grouping is not bound by the receipt")
        observed[arm] = languages
    return {
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "receipt_sha256": expected_receipt,
        "selection_identity_sha256": SELECTION_IDENTITY,
        "arms": {arm: list(values) for arm, values in observed.items()},
    }
