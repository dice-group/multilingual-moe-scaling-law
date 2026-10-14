"""URIEL+ distance construction for the frozen K196 candidate frame."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from multilingual_megatron.experiments.scaling.language_composition.selection.common import PRIMARY_AXES, SelectionError, SelectionInputs


GENETIC_DISTANCE_MAX = 32.123024


@dataclass(frozen=True)
class DistanceBundle:
    """Four normalized matrices and their equal-weight composite."""

    languages: tuple[str, ...]
    matrices: dict[str, np.ndarray]
    composite: np.ndarray
    raw_maxima: dict[str, float]
    matrix_sha256: dict[str, str]


def validate_distance_matrix(
    matrix: np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (size, size):
        raise SelectionError(
            f"{name} distance matrix has shape {values.shape}, expected {(size, size)}"
        )
    if not np.isfinite(values).all():
        raise SelectionError(f"{name} distance matrix contains a non-finite value")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0.0):
        raise SelectionError(f"{name} distance matrix is not symmetric")
    if not np.allclose(np.diag(values), 0.0, atol=1e-12, rtol=0.0):
        raise SelectionError(f"{name} distance matrix has a nonzero diagonal")
    if values.min() < -1e-12 or values.max() > 1.0 + 1e-12:
        raise SelectionError(f"{name} distance matrix is outside [0, 1]")
    return np.clip(values, 0.0, 1.0)


def angular_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return angular distances divided by pi, with range [0, 1]."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SelectionError("Angular-distance input must be a finite 2D matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        raise SelectionError("Angular distance is undefined for a zero vector")
    cosine = (values @ values.T) / np.outer(norms, norms)
    matrix = np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi
    np.fill_diagonal(matrix, 0.0)
    return matrix


def normalize_distance_matrix(
    matrix: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, float]:
    """Divide a complete candidate-frame matrix by its observed maximum."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise SelectionError(f"{name} distance matrix must be square")
    if not np.isfinite(values).all() or values.min() < -1e-12:
        raise SelectionError(f"{name} distance matrix must be finite and nonnegative")
    maximum = float(values.max())
    if maximum <= 0.0:
        raise SelectionError(f"{name} distance matrix has no positive distance")
    normalized = validate_distance_matrix(
        values / maximum,
        size=values.shape[0],
        name=name,
    )
    return normalized, maximum


def equal_weight_composite(
    matrices: Mapping[str, np.ndarray],
    *,
    axes: Sequence[str] = PRIMARY_AXES,
) -> np.ndarray:
    """Average exactly one normalized matrix per frozen modality."""

    if tuple(axes) != PRIMARY_AXES or set(matrices) != set(PRIMARY_AXES):
        raise SelectionError(f"Composite requires exactly {PRIMARY_AXES}")
    size = np.asarray(matrices[PRIMARY_AXES[0]]).shape[0]
    validated = [
        validate_distance_matrix(matrices[axis], size=size, name=axis)
        for axis in PRIMARY_AXES
    ]
    return validate_distance_matrix(
        np.mean(validated, axis=0),
        size=size,
        name="composite",
    )


def mean_pairwise_distance(matrix: np.ndarray, indices: Sequence[int]) -> float:
    selected = tuple(indices)
    if len(selected) < 2 or len(selected) != len(set(selected)):
        raise SelectionError("A selected set needs at least two unique indices")
    values = [
        float(matrix[first, second])
        for offset, first in enumerate(selected)
        for second in selected[offset + 1 :]
    ]
    return float(np.mean(values))


def _posterior_pair(model: Mapping[str, Any], observation: np.ndarray) -> np.ndarray:
    try:
        prior = np.asarray(model["P_z"], dtype=float)
        conditional = model["P_x_given_z"]
    except (KeyError, TypeError) as exc:
        raise SelectionError("Malformed latent-islands model") from exc
    if prior.shape != (2,):
        raise SelectionError("Latent-islands prior must have two states")
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.log(prior)
        for index, value in enumerate(observation.astype(int)):
            if value not in (0, 1):
                raise SelectionError(
                    "URIEL+ feature is not binary after the released cast"
                )
            likelihood = np.asarray(conditional[index], dtype=float)
            if likelihood.shape != (2, 2):
                raise SelectionError("Malformed latent-islands likelihood table")
            score += np.log(likelihood[:, value])
        posterior = np.exp(score - logsumexp(score))
    if posterior.shape != (2,) or not np.isfinite(posterior).all():
        raise SelectionError("Non-finite latent-island posterior")
    return posterior


def _typology_matrix(inputs: SelectionInputs) -> np.ndarray:
    try:
        frame = pd.read_csv(
            inputs.artifact_paths["urielplus_softimpute"], index_col=0
        )
        with inputs.artifact_paths["latent_islands"].open("rb") as handle:
            islands = pickle.load(handle)  # noqa: S301 - artifact is hash-pinned
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        raise SelectionError(f"Cannot load pinned typology artifacts: {exc}") from exc
    if frame.shape[1] != 800 or not frame.index.is_unique:
        raise SelectionError("Expected a unique-index URIEL+ table with 800 features")
    if not isinstance(islands, list) or len(islands) != 325:
        raise SelectionError("Expected the released 325 latent islands")
    codes = [
        inputs.glottocode_by_language[language]
        for language in inputs.eligible_languages
    ]
    missing = sorted(set(codes) - set(frame.index))
    if missing:
        raise SelectionError(f"Typology coverage is incomplete: {missing}")

    vectors: list[np.ndarray] = []
    for code in codes:
        row = frame.loc[code].to_numpy(dtype=float)
        if not np.isfinite(row).all():
            raise SelectionError(f"Non-finite URIEL+ typology row: {code}")
        posteriors: list[np.ndarray] = []
        for island in islands:
            try:
                indices = np.asarray(island["indices"], dtype=int)
                model = island["model"]
            except (KeyError, TypeError) as exc:
                raise SelectionError("Malformed latent-islands entry") from exc
            posteriors.append(_posterior_pair(model, row[indices]))
        vectors.append(np.concatenate(posteriors))
    return angular_distance_matrix(np.asarray(vectors))


def _genealogy_matrix(inputs: SelectionInputs) -> np.ndarray:
    try:
        frame = pd.read_csv(
            inputs.artifact_paths["genetic_embeddings"], index_col=0
        )
    except (OSError, ValueError) as exc:
        raise SelectionError(f"Cannot load pinned genetic embeddings: {exc}") from exc
    if not frame.index.is_unique:
        raise SelectionError("Genetic embedding index is not unique")
    codes = [
        inputs.glottocode_by_language[language]
        for language in inputs.eligible_languages
    ]
    missing = sorted(set(codes) - set(frame.index))
    if missing:
        raise SelectionError(f"Genealogy coverage is incomplete: {missing}")
    vectors = frame.loc[codes].to_numpy(dtype=float)
    if not np.isfinite(vectors).all():
        raise SelectionError("Genetic embeddings contain a non-finite value")
    count = len(codes)
    matrix = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(first + 1, count):
            left, right = vectors[first], vectors[second]
            lorentz = -left[0] * right[0] + np.dot(left[1:], right[1:])
            argument = max(-lorentz, 1.0 + np.finfo(float).eps)
            value = float(np.arccosh(argument)) / GENETIC_DISTANCE_MAX
            matrix[first, second] = matrix[second, first] = value
    return matrix


def _geography_matrix(inputs: SelectionInputs) -> np.ndarray:
    try:
        with np.load(
            inputs.artifact_paths["urielplus_geography"], allow_pickle=False
        ) as artifact:
            language_codes = [str(code) for code in artifact["langs"]]
            data = np.asarray(artifact["data"], dtype=float)
    except (OSError, ValueError, KeyError) as exc:
        raise SelectionError(f"Cannot load pinned geography artifact: {exc}") from exc
    if data.ndim != 3 or data.shape[1:] != (299, 1):
        raise SelectionError(f"Unexpected URIEL+ geography shape: {data.shape}")
    if len(language_codes) != len(set(language_codes)):
        raise SelectionError("URIEL+ geography language codes are duplicated")
    row_by_iso = {code: index for index, code in enumerate(language_codes)}
    isos = [language.split("_", 1)[0] for language in inputs.eligible_languages]
    missing = sorted(set(isos) - set(row_by_iso))
    if missing:
        raise SelectionError(f"Geography coverage is incomplete: {missing}")
    vectors = np.asarray([data[row_by_iso[iso], :, 0] for iso in isos])
    return angular_distance_matrix(vectors)


def _script_matrix(languages: Sequence[str]) -> np.ndarray:
    scripts: list[str] = []
    for language in languages:
        parts = language.split("_", 1)
        if len(parts) != 2 or not parts[1]:
            raise SelectionError(f"Language identifier lacks a script suffix: {language}")
        scripts.append(parts[1])
    values = np.asarray(scripts)
    return (values[:, None] != values[None, :]).astype(float)


def _matrix_digest(languages: Sequence[str], name: str, matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    for language in languages:
        digest.update(language.encode("utf-8") + b"\n")
    digest.update(name.encode("utf-8") + b"\n")
    digest.update(np.asarray(matrix, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def build_distance_bundle(inputs: SelectionInputs) -> DistanceBundle:
    raw = {
        "typology": _typology_matrix(inputs),
        "genealogy": _genealogy_matrix(inputs),
        "geography": _geography_matrix(inputs),
        "script": _script_matrix(inputs.eligible_languages),
    }
    matrices: dict[str, np.ndarray] = {}
    raw_maxima: dict[str, float] = {}
    for axis in PRIMARY_AXES:
        matrices[axis], raw_maxima[axis] = normalize_distance_matrix(
            raw[axis], name=axis
        )
    composite = equal_weight_composite(matrices)
    matrix_hashes = {
        axis: _matrix_digest(inputs.eligible_languages, axis, matrices[axis])
        for axis in PRIMARY_AXES
    }
    matrix_hashes["composite"] = _matrix_digest(
        inputs.eligible_languages, "composite", composite
    )
    return DistanceBundle(
        languages=inputs.eligible_languages,
        matrices=matrices,
        composite=composite,
        raw_maxima=raw_maxima,
        matrix_sha256=matrix_hashes,
    )
