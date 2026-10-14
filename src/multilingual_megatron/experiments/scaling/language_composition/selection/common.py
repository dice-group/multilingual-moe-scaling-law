"""Frozen inputs and shared contracts for K12 language-composition selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.repo import find_repo_root


SPEC_SCHEMA = "language_composition_selection/v2"
RECEIPT_SCHEMA = "language_composition_selection_receipt/v2"
AVAILABILITY_SCHEMA = "language_composition_availability/v1"
PRIMARY_AXES = ("typology", "genealogy", "geography", "script")
EXTERNAL_ARTIFACTS = frozenset(
    {
        "urielplus_softimpute",
        "latent_islands",
        "genetic_embeddings",
        "urielplus_geography",
        "glottolog_languages",
    }
)
DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "language_composition_selection_v2.json"
)


class SelectionError(ValueError):
    """Raised when the frozen selection contract cannot be proven."""


@dataclass(frozen=True)
class ResourceStats:
    """Alpha-smoothed mixture statistics for one K12 arm."""

    english_probability: float
    effective_language_count: float
    probabilities: dict[str, float]
    target_exposure_tokens: dict[str, float]
    availability_tokens: dict[str, int]
    maximum_exposure_to_availability_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionInputs:
    """Hash-verified repository, corpus, mapping, and external inputs."""

    repo_root: Path
    spec_path: Path
    spec_sha256: str
    spec: dict[str, Any]
    source_languages: tuple[str, ...]
    eligible_languages: tuple[str, ...]
    reference_languages: tuple[str, ...]
    glottocode_by_language: dict[str, str]
    glottocode_mapping_sha256: str
    availability_tokens: dict[str, int]
    availability_projection_path: Path
    availability_projection_sha256: str
    source_sampling_plan_sha256: str
    source_sampling_plan_reverified: bool
    artifact_paths: dict[str, Path]
    artifact_sha256: dict[str, str]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectionError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SelectionError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any], *, omit: Sequence[str] = ()) -> str:
    omitted = set(omit)
    encoded = json.dumps(
        {key: value for key, value in payload.items() if key not in omitted},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def language_set_sha256(languages: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for language in languages:
        digest.update(language.encode("utf-8") + b"\n")
    return digest.hexdigest()


def render_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("receipt_sha256", None)
    sealed["receipt_sha256"] = canonical_sha256(sealed)
    return sealed


def verify_receipt(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise SelectionError(f"Expected receipt schema {RECEIPT_SCHEMA}")
    expected = payload.get("receipt_sha256")
    observed = canonical_sha256(payload, omit=("receipt_sha256",))
    if not isinstance(expected, str) or expected != observed:
        raise SelectionError("Selection receipt self-hash mismatch")


def write_immutable(path: Path, content: bytes) -> None:
    """Create an artifact, or prove an existing artifact is byte-identical."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise SelectionError(f"Refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectionError(f"{label} must be an object")
    return value


def _group_languages(path: Path, definition: Mapping[str, Any]) -> tuple[str, ...]:
    expected_file = definition.get("file_sha256")
    expected_set = definition.get("language_set_sha256")
    if not isinstance(expected_file, str) or sha256_file(path) != expected_file:
        raise SelectionError(f"Frozen grouping file hash drifted: {path}")
    payload = read_json(path)
    try:
        languages = payload["0"]["languages"]
    except (KeyError, TypeError) as exc:
        raise SelectionError(f"Language grouping lacks 0.languages: {path}") from exc
    if not isinstance(languages, list) or not all(
        isinstance(language, str) for language in languages
    ):
        raise SelectionError(f"Invalid language list in {path}")
    if len(languages) != len(set(languages)):
        raise SelectionError(f"Duplicate language in {path}")
    observed_set = language_set_sha256(languages)
    if not isinstance(expected_set, str) or observed_set != expected_set:
        raise SelectionError(f"Frozen language order drifted: {path}")
    declared_set = payload.get("language_set_sha256")
    if declared_set is not None and declared_set != observed_set:
        raise SelectionError(f"Grouping's declared language hash drifted: {path}")
    return tuple(languages)


def mixture_stats(
    languages: Sequence[str],
    availability_tokens: Mapping[str, int],
    *,
    alpha: float,
    total_tokens: int,
    required_language: str,
) -> ResourceStats:
    if len(languages) != len(set(languages)):
        raise SelectionError("Cannot score a language set with duplicates")
    if required_language not in languages:
        raise SelectionError(f"Required language {required_language} is absent")
    try:
        availability = {
            language: int(availability_tokens[language]) for language in languages
        }
    except KeyError as exc:
        raise SelectionError(f"Missing availability for {exc.args[0]}") from exc
    if any(value <= 0 for value in availability.values()):
        raise SelectionError("All availability values must be positive")
    weights = {language: value**alpha for language, value in availability.items()}
    weight_sum = math.fsum(weights.values())
    probabilities = {
        language: weights[language] / weight_sum for language in languages
    }
    effective = 1.0 / math.fsum(value * value for value in probabilities.values())
    targets = {
        language: total_tokens * probabilities[language] for language in languages
    }
    ratios = {
        language: targets[language] / availability[language]
        for language in languages
    }
    return ResourceStats(
        english_probability=probabilities[required_language],
        effective_language_count=effective,
        probabilities=probabilities,
        target_exposure_tokens=targets,
        availability_tokens=availability,
        maximum_exposure_to_availability_ratio=max(ratios.values()),
    )


def resource_violations(
    stats: ResourceStats,
    *,
    reference: ResourceStats,
    english_tolerance: float,
    effective_language_count_tolerance: float,
    numerical_tolerance: float = 1e-9,
) -> list[str]:
    violations: list[str] = []
    if (
        abs(stats.english_probability - reference.english_probability)
        > english_tolerance + numerical_tolerance
    ):
        violations.append("english_probability")
    if (
        abs(stats.effective_language_count - reference.effective_language_count)
        > effective_language_count_tolerance + numerical_tolerance
    ):
        violations.append("effective_language_count")
    if stats.maximum_exposure_to_availability_ratio > 1.0 + numerical_tolerance:
        violations.append("additional_corpus_cycling")
    return violations


def _resolve_artifacts(
    spec: Mapping[str, Any],
    *,
    artifact_dir: Path,
    overrides: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, str]]:
    definitions = _require_mapping(spec.get("external_artifacts"), "external_artifacts")
    if set(definitions) != EXTERNAL_ARTIFACTS:
        raise SelectionError(
            f"External artifact contract differs from {sorted(EXTERNAL_ARTIFACTS)}"
        )
    unknown = sorted(set(overrides) - set(definitions))
    if unknown:
        raise SelectionError(f"Unknown artifact overrides: {unknown}")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, raw_definition in definitions.items():
        definition = _require_mapping(
            raw_definition, f"external_artifacts.{name}"
        )
        filename = definition.get("filename")
        expected = definition.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise SelectionError(f"Artifact {name} lacks filename or sha256")
        path = overrides.get(name, artifact_dir / filename).resolve()
        if not path.is_file():
            raise SelectionError(f"Required artifact {name} is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise SelectionError(
                f"Artifact hash mismatch for {name}: expected {expected}, got {observed}"
            )
        paths[name] = path
        hashes[name] = observed
    return paths, hashes


def _map_eligible_languages(
    source_languages: Sequence[str],
    spec: Mapping[str, Any],
    glottolog_path: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    try:
        with glottolog_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SelectionError(f"Cannot read Glottolog language table: {exc}") from exc
    codes_by_iso: dict[str, list[str]] = {}
    row_by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = str(row.get("Glottocode", ""))
        iso = str(row.get("ISO639P3code", ""))
        level = str(row.get("Level", ""))
        if code:
            if code in row_by_code:
                raise SelectionError(f"Duplicate Glottocode in pinned table: {code}")
            row_by_code[code] = row
        if iso and code and level in {"language", "dialect"}:
            codes_by_iso.setdefault(iso, []).append(code)

    eligibility = _require_mapping(spec.get("eligibility"), "eligibility")
    raw_excluded = eligibility.get("exclude_before_metric_construction")
    if not isinstance(raw_excluded, list) or not all(
        isinstance(language, str) for language in raw_excluded
    ):
        raise SelectionError("eligibility exclusions must be a language list")
    if len(raw_excluded) != len(set(raw_excluded)) or not set(raw_excluded) <= set(
        source_languages
    ):
        raise SelectionError("eligibility exclusions are duplicated or outside K200")
    excluded = set(raw_excluded)
    exceptions = _require_mapping(spec.get("mapping_exceptions", {}), "mapping_exceptions")

    eligible: list[str] = []
    mapping: dict[str, str] = {}
    for language in source_languages:
        if language in excluded:
            continue
        iso = language.split("_", 1)[0]
        expected_exception_iso: str | None = None
        if language in exceptions:
            exception = _require_mapping(
                exceptions[language], f"mapping_exceptions.{language}"
            )
            glottocode = exception.get("glottocode")
            candidates = [glottocode] if isinstance(glottocode, str) else []
            raw_exception_iso = exception.get("glottolog_iso")
            if isinstance(raw_exception_iso, str):
                expected_exception_iso = raw_exception_iso
        else:
            candidates = codes_by_iso.get(iso, [])
        if len(candidates) != 1 or candidates[0] not in row_by_code:
            raise SelectionError(
                f"No unambiguous Glottolog mapping for {language}: {candidates}"
            )
        if language in exceptions:
            observed_iso = row_by_code[candidates[0]].get("ISO639P3code")
            if expected_exception_iso is None or observed_iso != expected_exception_iso:
                raise SelectionError(
                    f"Mapping exception ISO drift for {language}: {observed_iso}"
                )
        eligible.append(language)
        mapping[language] = candidates[0]

    expected_size = int(eligibility.get("eligible_frame_size", -1))
    expected_hash = eligibility.get("eligible_language_set_sha256")
    if len(eligible) != expected_size:
        raise SelectionError(
            f"Expected K{expected_size} eligible frame, found K{len(eligible)}"
        )
    if not isinstance(expected_hash, str) or language_set_sha256(eligible) != expected_hash:
        raise SelectionError("Frozen eligible-language order drifted")
    return tuple(eligible), mapping


def _load_availability(
    spec: Mapping[str, Any],
    *,
    root: Path,
    source_languages: Sequence[str],
    source_language_set_sha256: str,
    sampling_plan_override: Path | None,
) -> tuple[dict[str, int], Path, str, str, bool]:
    definition = _require_mapping(
        spec.get("availability_projection"), "availability_projection"
    )
    projection_path = (root / str(definition.get("path", ""))).resolve()
    if not projection_path.is_relative_to(root):
        raise SelectionError("Availability projection must be repository-owned")
    expected_projection_hash = definition.get("file_sha256")
    observed_projection_hash = sha256_file(projection_path)
    if (
        not isinstance(expected_projection_hash, str)
        or observed_projection_hash != expected_projection_hash
    ):
        raise SelectionError("Frozen availability-projection hash drifted")
    projection = read_json(projection_path)
    if projection.get("schema") != AVAILABILITY_SCHEMA:
        raise SelectionError(f"Expected availability schema {AVAILABILITY_SCHEMA}")
    if projection.get("source_language_set_sha256") != source_language_set_sha256:
        raise SelectionError("Availability projection uses a different K200 frame")
    raw_availability = _require_mapping(
        projection.get("availability_tokens"), "availability_tokens"
    )
    if set(raw_availability) != set(source_languages):
        raise SelectionError("Availability projection keys differ from the K200 frame")
    try:
        availability = {
            language: int(raw_availability[language]) for language in source_languages
        }
    except (TypeError, ValueError) as exc:
        raise SelectionError("Availability projection contains a non-integer") from exc
    if any(value <= 0 for value in availability.values()):
        raise SelectionError("Availability projection must be positive")

    source_plan = _require_mapping(
        projection.get("source_sampling_plan"), "source_sampling_plan"
    )
    expected_source_hash = source_plan.get("sha256")
    if not isinstance(expected_source_hash, str):
        raise SelectionError("Availability projection lacks its source-plan hash")
    default_source_path = (root / str(source_plan.get("path", ""))).resolve()
    source_path = sampling_plan_override
    if source_path is None and default_source_path.is_file():
        source_path = default_source_path
    reverified = source_path is not None
    if source_path is not None:
        source_path = source_path.resolve()
        if sha256_file(source_path) != expected_source_hash:
            raise SelectionError("Frozen source sampling-plan hash drifted")
        source_payload = read_json(source_path)
        try:
            observed_availability = {
                language: int(
                    source_payload["physical_pool"][language][
                        "estimated_train_rehydrated_tokens"
                    ]
                )
                for language in source_languages
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionError("Source plan lacks the K200 availability vector") from exc
        if observed_availability != availability:
            raise SelectionError("Availability projection drifted from its source plan")
    return (
        availability,
        projection_path,
        observed_projection_hash,
        expected_source_hash,
        reverified,
    )


def load_inputs(
    spec_path: Path = DEFAULT_SPEC,
    *,
    artifact_dir: Path,
    repo_root: Path | None = None,
    sampling_plan: Path | None = None,
    artifact_overrides: Mapping[str, Path] | None = None,
) -> SelectionInputs:
    """Load every authority through explicit paths and fail on any drift."""

    root = (repo_root or find_repo_root(Path(__file__))).resolve()
    resolved_spec = spec_path if spec_path.is_absolute() else root / spec_path
    resolved_spec = resolved_spec.resolve()
    if not resolved_spec.is_relative_to(root):
        raise SelectionError("Selection spec must be repository-owned")
    spec = read_json(resolved_spec)
    if spec.get("schema") != SPEC_SCHEMA:
        raise SelectionError(f"Expected spec schema {SPEC_SCHEMA}: {resolved_spec}")

    metric_policy = _require_mapping(spec.get("metric_policy"), "metric_policy")
    if tuple(metric_policy.get("primary_axes", ())) != PRIMARY_AXES:
        raise SelectionError(f"Main selection requires exactly {PRIMARY_AXES}")
    policy = _require_mapping(spec.get("selection_policy"), "selection_policy")
    if int(policy.get("set_size", -1)) != 12:
        raise SelectionError("Main selection requires K12 arms")
    if policy.get("no_additional_corpus_cycling") is not True:
        raise SelectionError("Main selection must prohibit additional corpus cycling")
    promotion = _require_mapping(spec.get("promotion_policy"), "promotion_policy")
    if promotion.get("directional_gates") != ["composite", *PRIMARY_AXES]:
        raise SelectionError("Promotion must fail closed on the composite and all axes")
    expected_memberships = _require_mapping(
        promotion.get("expected_language_set_sha256"),
        "promotion_policy.expected_language_set_sha256",
    )
    if set(expected_memberships) != {"compact", "diverse"} or any(
        not isinstance(digest, str) or len(digest) != 64
        for digest in expected_memberships.values()
    ):
        raise SelectionError("Promotion requires both frozen membership hashes")

    source_definition = _require_mapping(spec.get("source_frame"), "source_frame")
    reference_definition = _require_mapping(spec.get("reference_arm"), "reference_arm")
    source_path = (root / str(source_definition.get("path", ""))).resolve()
    reference_path = (root / str(reference_definition.get("path", ""))).resolve()
    if not source_path.is_relative_to(root) or not reference_path.is_relative_to(root):
        raise SelectionError("Language grouping authorities must be repository-owned")
    source = _group_languages(source_path, source_definition)
    reference = _group_languages(reference_path, reference_definition)
    eligibility = _require_mapping(spec.get("eligibility"), "eligibility")
    if len(source) != int(eligibility.get("source_frame_size", -1)):
        raise SelectionError("Frozen source-frame size drifted")
    required_language = str(policy.get("required_language", ""))
    if len(reference) != 12 or required_language not in reference:
        raise SelectionError("Reference must be K12 and contain the required language")

    paths, artifact_hashes = _resolve_artifacts(
        spec,
        artifact_dir=artifact_dir.resolve(),
        overrides={
            name: path.resolve()
            for name, path in (artifact_overrides or {}).items()
        },
    )
    eligible, mapping = _map_eligible_languages(
        source, spec, paths["glottolog_languages"]
    )
    if not set(reference) <= set(eligible):
        raise SelectionError("Reference arm is not a subset of the eligible K196 frame")

    (
        source_availability,
        availability_projection_path,
        availability_projection_sha256,
        source_sampling_plan_sha256,
        source_sampling_plan_reverified,
    ) = _load_availability(
        spec,
        root=root,
        source_languages=source,
        source_language_set_sha256=language_set_sha256(source),
        sampling_plan_override=(sampling_plan.resolve() if sampling_plan else None),
    )
    availability = {
        language: source_availability[language] for language in eligible
    }

    alpha = float(policy.get("alpha", math.nan))
    total_tokens = int(policy.get("total_tokens", -1))
    english_tolerance = float(
        policy.get("english_probability_absolute_tolerance", math.nan)
    )
    effective_tolerance = float(
        policy.get("effective_language_count_absolute_tolerance", math.nan)
    )
    if (
        not math.isfinite(alpha)
        or alpha <= 0.0
        or total_tokens <= 0
        or not 0.0 < english_tolerance < 1.0
        or not math.isfinite(effective_tolerance)
        or effective_tolerance <= 0.0
    ):
        raise SelectionError("Invalid sampling or resource-control policy")
    reference_stats = mixture_stats(
        reference,
        availability,
        alpha=alpha,
        total_tokens=total_tokens,
        required_language=required_language,
    )
    if resource_violations(
        reference_stats,
        reference=reference_stats,
        english_tolerance=english_tolerance,
        effective_language_count_tolerance=effective_tolerance,
    ):
        raise SelectionError("Frozen reference arm requires additional corpus cycling")
    if reference_stats.english_probability <= english_tolerance:
        raise SelectionError("English tolerance makes the lower probability bound invalid")

    return SelectionInputs(
        repo_root=root,
        spec_path=resolved_spec,
        spec_sha256=sha256_file(resolved_spec),
        spec=spec,
        source_languages=source,
        eligible_languages=eligible,
        reference_languages=reference,
        glottocode_by_language=mapping,
        glottocode_mapping_sha256=canonical_sha256(mapping),
        availability_tokens=availability,
        availability_projection_path=availability_projection_path,
        availability_projection_sha256=availability_projection_sha256,
        source_sampling_plan_sha256=source_sampling_plan_sha256,
        source_sampling_plan_reverified=source_sampling_plan_reverified,
        artifact_paths=paths,
        artifact_sha256=artifact_hashes,
    )
