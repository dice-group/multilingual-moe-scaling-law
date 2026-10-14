"""Loss-blind language-composition exposure components for exploratory M6.

The builder partitions every readout's non-target exposure into disjoint
same-family, same-script, and remainder components.  Language family labels
come from an explicit, caller-bound FineWeb2 inventory; scripts are derived
from language identifiers so a language without an inventory row remains
usable without inventing a family label.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeAlias

import numpy as np

from .data import canonical_json_sha256

COMPONENT_COLUMNS = ("D_target", "D_family", "D_script", "D_other")
COMPONENT_CONTENT_COLUMNS = (
    "trajectory_id",
    "readout_id",
    "language",
    "D_t",
    "D_family",
    "D_script",
    "D_other",
    "D_total",
)
RECONSTRUCTION_RTOL = 1e-12
RECONSTRUCTION_ATOL = 1e-6

LanguageMetadata: TypeAlias = dict[str, dict[str, str | None]]
CompositionSets: TypeAlias = dict[str, dict[str, tuple[str, ...]]]


class M6CompositionError(RuntimeError):
    """Raised when metadata or exposure rows cannot form M6 components."""


def _language_script(language: str) -> str:
    if not isinstance(language, str) or not language or language.endswith("_removed"):
        raise M6CompositionError(f"invalid language identifier: {language!r}")
    base, separator, script = language.rpartition("_")
    if not separator or not base or not script:
        raise M6CompositionError(
            f"language identifier lacks a script suffix: {language!r}"
        )
    return script


def _optional_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    if label.lower() in {"", "-", "nan", "none", "null"}:
        return None
    return label


def _languages(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise M6CompositionError("languages must be non-empty string identifiers")
        _language_script(value)
        result.append(value)
    unique = tuple(sorted(set(result)))
    if not unique:
        raise M6CompositionError("at least one language is required")
    return unique


def load_language_metadata(
    inventory_path: Path | str,
    languages: Iterable[Any],
) -> LanguageMetadata:
    """Load exact requested-language metadata from a FineWeb2 inventory.

    Only ``split=train`` rows with exact subset identifiers are eligible.
    Family is taken from the inventory and remains ``None`` when absent.
    Script is always derived from the language identifier and any populated
    inventory script is used as a consistency check.  Consequently, a
    requested language absent from the inventory has a script but no family.
    """
    requested = _languages(languages)
    requested_set = set(requested)
    path = Path(inventory_path)
    found: LanguageMetadata = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"subset", "split", "script", "family"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise M6CompositionError(
                    "language inventory lacks subset/split/script/family columns"
                )
            for row in reader:
                language = str(row.get("subset", "")).strip()
                if (
                    language not in requested_set
                    or language.endswith("_removed")
                    or str(row.get("split", "")).strip() != "train"
                ):
                    continue
                if language in found:
                    raise M6CompositionError(
                        f"language inventory duplicates train row for {language}"
                    )
                script = _language_script(language)
                inventory_script = _optional_label(row.get("script"))
                if inventory_script is not None and inventory_script != script:
                    raise M6CompositionError(
                        f"inventory script disagrees with {language}: "
                        f"{inventory_script!r} != {script!r}"
                    )
                found[language] = {
                    "script": script,
                    "family": _optional_label(row.get("family")),
                }
    except OSError as error:
        raise M6CompositionError(f"cannot read language inventory {path}: {error}") from error

    # Preserve requested languages missing from the pinned inventory (notably
    # hau_Latn) without making an external or inferred family assignment.
    return {
        language: found.get(
            language,
            {"script": _language_script(language), "family": None},
        )
        for language in requested
    }


def _normalized_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
    languages: Iterable[Any] | None = None,
) -> LanguageMetadata:
    selected = _languages(metadata if languages is None else languages)
    normalized: LanguageMetadata = {}
    for language in selected:
        value = metadata.get(language)
        if not isinstance(value, Mapping) or set(value) != {"script", "family"}:
            raise M6CompositionError(
                f"metadata for {language} must contain exactly script and family"
            )
        script = _optional_label(value["script"])
        if script is None:
            raise M6CompositionError(f"metadata lacks a script for {language}")
        expected_script = _language_script(language)
        if script != expected_script:
            raise M6CompositionError(
                f"metadata script disagrees with {language}: "
                f"{script!r} != {expected_script!r}"
            )
        normalized[language] = {
            "script": script,
            "family": _optional_label(value["family"]),
        }
    return normalized


def language_metadata_payload(
    metadata: Mapping[str, Mapping[str, Any]],
) -> LanguageMetadata:
    """Return a validated, language-sorted, canonical-JSON metadata mapping."""
    return _normalized_metadata(metadata)


def language_metadata_sha256(metadata: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_json_sha256(language_metadata_payload(metadata))


def composition_sets(
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    languages: Iterable[Any] | None = None,
) -> CompositionSets:
    """Return deterministic disjoint family/script source sets per target."""
    normalized = _normalized_metadata(metadata, languages)
    language_ids = tuple(normalized)
    result: CompositionSets = {}
    for target in language_ids:
        target_family = normalized[target]["family"]
        target_script = normalized[target]["script"]
        family = tuple(
            source
            for source in language_ids
            if source != target
            and target_family is not None
            and normalized[source]["family"] == target_family
        )
        family_set = set(family)
        script = tuple(
            source
            for source in language_ids
            if source != target
            and source not in family_set
            and normalized[source]["script"] == target_script
        )
        result[target] = {"family": family, "script": script}
    return result


def composition_sets_payload(
    memberships: Mapping[str, Mapping[str, Iterable[Any]]],
) -> dict[str, dict[str, list[str]]]:
    """Return canonical-JSON membership content with stable target/source order."""
    payload: dict[str, dict[str, list[str]]] = {}
    for target in sorted(memberships):
        if not isinstance(target, str) or not target:
            raise M6CompositionError("composition targets must be non-empty strings")
        value = memberships[target]
        if not isinstance(value, Mapping) or set(value) != {"family", "script"}:
            raise M6CompositionError(
                f"composition set for {target} must contain family and script"
            )
        family_values = list(value["family"])
        script_values = list(value["script"])
        if any(not isinstance(item, str) or not item for item in family_values + script_values):
            raise M6CompositionError("composition sources must be non-empty strings")
        family = sorted(family_values)
        script = sorted(script_values)
        if (
            len(family) != len(set(family))
            or len(script) != len(set(script))
            or target in family
            or target in script
            or set(family) & set(script)
        ):
            raise M6CompositionError(
                f"composition sources for {target} are duplicated or overlapping"
            )
        payload[target] = {"family": family, "script": script}
    if not payload:
        raise M6CompositionError("composition membership mapping is empty")
    return payload


def composition_sets_sha256(
    memberships: Mapping[str, Mapping[str, Iterable[Any]]],
) -> str:
    return canonical_json_sha256(composition_sets_payload(memberships))


def _validated_fit_exposures(frame: Any) -> tuple[str, ...]:
    required = {"readout_id", "language", "K", "D_total", "D_t"}
    if not required <= set(frame.columns):
        raise M6CompositionError("fit rows lack readout/language exposure columns")
    if len(frame) == 0:
        raise M6CompositionError("fit rows are empty")
    if frame.duplicated(["readout_id", "language"]).any():
        raise M6CompositionError("fit rows duplicate readout/language exposure keys")
    languages = _languages(frame["language"].tolist())
    if any(
        not isinstance(value, str) or not value
        for value in frame["readout_id"].tolist()
    ):
        raise M6CompositionError("readout identifiers must be non-empty strings")
    try:
        numeric = frame[["K", "D_total", "D_t"]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise M6CompositionError("fit exposures must be numeric") from error
    k = numeric[:, 0]
    totals = numeric[:, 1]
    target = numeric[:, 2]
    if (
        not np.isfinite(numeric).all()
        or np.any(k <= 0.0)
        or np.any(k != np.floor(k))
        or np.any(totals <= 0.0)
        or np.any(target <= 0.0)
    ):
        raise M6CompositionError(
            "K must be a positive integer and fit exposures finite and positive"
        )
    for readout_id, group in frame.groupby("readout_id", sort=False):
        group_k = group["K"].to_numpy(dtype=np.float64)
        group_total = group["D_total"].to_numpy(dtype=np.float64)
        declared_k = int(group_k[0])
        if np.any(group_k != group_k[0]) or len(group) != declared_k:
            raise M6CompositionError(
                f"readout {readout_id} must expose exactly its declared K rows"
            )
        if not np.allclose(
            group_total,
            group_total[0],
            rtol=RECONSTRUCTION_RTOL,
            atol=RECONSTRUCTION_ATOL,
        ):
            raise M6CompositionError(f"D_total changes within readout {readout_id}")
        if not math.isclose(
            float(group["D_t"].sum()),
            float(group_total[0]),
            rel_tol=RECONSTRUCTION_RTOL,
            abs_tol=RECONSTRUCTION_ATOL,
        ):
            raise M6CompositionError(
                f"source exposures do not sum to D_total within readout {readout_id}"
            )
    return languages


def composition_exposure_components(
    frame: Any,
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, CompositionSets]:
    """Partition every fit row into target/family/script/remainder exposure.

    The source sets depend only on language metadata.  Sources absent from a
    particular readout contribute zero, matching the loss-blind construction
    used by :mod:`atlas_exposure`.
    """
    import pandas as pd

    languages = _validated_fit_exposures(frame)
    memberships = composition_sets(metadata, languages=languages)
    exposure_lookup = {
        (str(readout), str(language)): float(exposure)
        for readout, language, exposure in frame[
            ["readout_id", "language", "D_t"]
        ].itertuples(index=False, name=None)
    }
    family = np.asarray(
        [
            sum(
                exposure_lookup.get((str(row.readout_id), source), 0.0)
                for source in memberships[str(row.language)]["family"]
            )
            for row in frame[["readout_id", "language"]].itertuples(index=False)
        ],
        dtype=np.float64,
    )
    script = np.asarray(
        [
            sum(
                exposure_lookup.get((str(row.readout_id), source), 0.0)
                for source in memberships[str(row.language)]["script"]
            )
            for row in frame[["readout_id", "language"]].itertuples(index=False)
        ],
        dtype=np.float64,
    )
    target = frame["D_t"].to_numpy(dtype=np.float64)
    total = frame["D_total"].to_numpy(dtype=np.float64)
    other = total - target - family - script
    tolerance = RECONSTRUCTION_ATOL + np.abs(total) * RECONSTRUCTION_RTOL
    if (
        not all(np.isfinite(item).all() for item in (target, family, script, other))
        or np.any(target < -tolerance)
        or np.any(family < -tolerance)
        or np.any(script < -tolerance)
        or np.any(other < -tolerance)
    ):
        raise M6CompositionError(
            "derived composition exposures must be finite and non-negative"
        )
    components = pd.DataFrame(
        {
            "D_target": np.maximum(target, 0.0),
            "D_family": np.maximum(family, 0.0),
            "D_script": np.maximum(script, 0.0),
            "D_other": np.maximum(other, 0.0),
        },
        index=frame.index,
    )
    reconstructed = components.sum(axis=1).to_numpy(dtype=np.float64)
    if not np.allclose(
        reconstructed,
        total,
        rtol=RECONSTRUCTION_RTOL,
        atol=RECONSTRUCTION_ATOL,
    ):
        raise M6CompositionError(
            "D_t + D_family + D_script + D_other does not reconstruct D_total"
        )
    return components, memberships


def build_composition_components(
    frame: Any,
    inventory_path: Path | str,
) -> tuple[Any, LanguageMetadata, CompositionSets]:
    """Facade helper loading bound metadata before deriving M6 components."""
    metadata = load_language_metadata(inventory_path, frame["language"].tolist())
    components, memberships = composition_exposure_components(frame, metadata)
    return components, metadata, memberships


def component_content_payload(frame: Any) -> dict[str, Any]:
    """Return order-independent canonical content for an M6 component table."""
    required = set(COMPONENT_CONTENT_COLUMNS)
    if not required <= set(frame.columns):
        raise M6CompositionError("M6 component table lacks canonical content columns")
    selected = frame.loc[:, list(COMPONENT_CONTENT_COLUMNS)]
    for column in COMPONENT_CONTENT_COLUMNS[:3]:
        if any(
            not isinstance(value, str) or not value
            for value in selected[column].tolist()
        ):
            raise M6CompositionError(
                "M6 component row identifiers must be non-empty strings"
            )
    ordered = selected.sort_values(
        ["trajectory_id", "readout_id", "language"], kind="mergesort"
    )
    if ordered.duplicated(["trajectory_id", "readout_id", "language"]).any():
        raise M6CompositionError("M6 component table duplicates canonical row keys")
    numeric_columns = COMPONENT_CONTENT_COLUMNS[3:]
    try:
        numeric = ordered.loc[:, list(numeric_columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise M6CompositionError("M6 component content must be numeric") from error
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric[:, 1:4] < 0.0)
        or np.any(numeric[:, 0] <= 0.0)
        or np.any(numeric[:, 4] <= 0.0)
    ):
        raise M6CompositionError(
            "M6 component content must be finite with non-negative components"
        )
    reconstructed = numeric[:, 0] + numeric[:, 1] + numeric[:, 2] + numeric[:, 3]
    if not np.allclose(
        reconstructed,
        numeric[:, 4],
        rtol=RECONSTRUCTION_RTOL,
        atol=RECONSTRUCTION_ATOL,
    ):
        raise M6CompositionError("M6 component content does not reconstruct D_total")
    rows = [
        [
            str(row[0]),
            str(row[1]),
            str(row[2]),
            *(float(value) for value in row[3:]),
        ]
        for row in ordered.itertuples(index=False, name=None)
    ]
    return {"columns": list(COMPONENT_CONTENT_COLUMNS), "rows": rows}


def component_content_sha256(frame: Any) -> str:
    return canonical_json_sha256(component_content_payload(frame))
