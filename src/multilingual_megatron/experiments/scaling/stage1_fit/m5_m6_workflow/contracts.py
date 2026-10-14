"""Scientific constants, source validation, and immutable inputs for M5/M6."""

from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..data import (
    FIT_ROW_COLUMNS,
    FitRow,
    fit_rows_content_sha256,
    load_fit_rows_snapshot,
    sha256_file,
)
from ..execution import (
    load_manifest,
    read_json,
    result_path,
    write_json_once,
)
from ..fit import (
    EXPLORATORY_COMPOSITION_MODELS,
    M5_M6_GAMMA_N_SCALE,
    M5_M6_PHI_ENDPOINT_SCALE,
    M5_M6_PSI_ENDPOINT_SCALE,
    M5_M6_REGULARIZATION_SCHEMA,
    StartPoint,
    outer_parameter_names,
)
from ..interaction import (
    PRIMARY_FOLDS,
    _official_report,
    _reduced_exploratory,
    _seal,
    _sealed,
    _wave_index,
)
from ..m6_composition import (
    component_content_sha256,
    composition_exposure_components,
    composition_sets_payload,
    composition_sets_sha256,
    language_metadata_payload,
    language_metadata_sha256,
    load_language_metadata,
)
from ..workflow_source import bind_source_bundle

PROFILE_SCHEMA = "stage1_m5_m6_profile/v1"


COMPONENT_SCHEMA = "stage1_m5_m6_composition_components/v1"


INDEX_SCHEMA = "stage1_m5_m6_screen_index/v1"


AUDIT_INDEX_SCHEMA = "stage1_m5_m6_screen_audit_index/v1"


AUDIT_SCHEMA = "stage1_m5_m6_screen_audit/v1"


REPORT_SCHEMA = "stage1_m5_m6_screen_report/v1"


MODELS = EXPLORATORY_COMPOSITION_MODELS


MODEL_ORDER = ("M2", "M5", "M6")


COMPONENT_COLUMNS = ("D_family", "D_script", "D_other")


ANCHOR_COUNT = 8


RIDGE_LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


M6_SEEDS = (0.0, 0.03, 0.1)


CHUNK_SIZE = 2


EQUAL_FOLD_M2_GATE = 0.05066808


K200_M1_GATE = 0.05441795


MINIMUM_EQUIVALENT_BASIN_STARTS = 3


class M5M6Error(RuntimeError):
    """Raised when the exploratory M5/M6 campaign loses identity."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _runner_source_binding() -> dict[str, str]:
    from .. import m5_m6 as facade

    entrypoint = Path(str(facade.__file__)).resolve()
    return bind_source_bundle(
        entrypoint=entrypoint,
        source_root=entrypoint.parent,
        expected_sources=facade.SOURCE_BUNDLE,
        error_type=M5M6Error,
    )


def _resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (_repository_root() / path).resolve()


def _binding(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise M5M6Error(f"profile input {name} must contain path and sha256")
    path = _resolve_path(value["path"])
    expected = str(value["sha256"])
    if not path.is_file() or sha256_file(path) != expected:
        raise M5M6Error(f"profile input {name} is missing or changed: {path}")
    return {"path": str(path), "file_sha256": expected}


def load_profile(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M5M6Error(f"cannot read profile {path}: {error}") from error
    required = {
        "schema_version", "output_root", "mode", "chunk_size", "blas_threads",
        "memory", "time_limit", "array_concurrency", "partition", "account",
        "reservation", "anchor_count", "ridge_lambdas", "m6_seeds",
        "regularization_scales", "screen_gates", "inputs",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema_version") != PROFILE_SCHEMA:
        raise M5M6Error("M5/M6 profile keys or schema changed")
    if value.get("mode") != "slurm_array" or int(value["chunk_size"]) != CHUNK_SIZE \
            or int(value["blas_threads"]) != 1 or int(value["array_concurrency"]) < 1:
        raise M5M6Error("M5/M6 requires chunk-size-two CPU arrays and one BLAS thread")
    if int(value["anchor_count"]) != ANCHOR_COUNT \
            or tuple(float(item) for item in value["ridge_lambdas"]) != RIDGE_LAMBDAS \
            or tuple(float(item) for item in value["m6_seeds"]) != M6_SEEDS:
        raise M5M6Error("M5/M6 screen grid changed")
    scales = value.get("regularization_scales")
    expected_scales = {
        "phi_endpoint_scale": M5_M6_PHI_ENDPOINT_SCALE,
        "psi_endpoint_scale": M5_M6_PSI_ENDPOINT_SCALE,
        "gamma_n_scale": M5_M6_GAMMA_N_SCALE,
    }
    if scales != expected_scales:
        raise M5M6Error("M5/M6 regularization scales changed")
    gates = value.get("screen_gates")
    if gates != {
        "equal_fold_rmse_below_m2": EQUAL_FOLD_M2_GATE,
        "k200_rmse_below_m1": K200_M1_GATE,
    }:
        raise M5M6Error("M5/M6 screen gates changed")
    expected_inputs = {
        "input_receipt", "holdouts", "candidate_report",
        "interaction_screen_index", "interaction_screen_audit", "language_metadata",
    }
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise M5M6Error("M5/M6 source inventory changed")
    return {
        **value,
        "output_root": str(_resolve_path(value["output_root"])),
        "inputs": {name: _binding(item, name=name) for name, item in inputs.items()},
        "profile_path": str(path.resolve()),
        "profile_file_sha256": sha256_file(path),
    }


def _fit_rows(frame: Any) -> tuple[FitRow, ...]:
    return tuple(FitRow(
        trajectory_id=str(item.trajectory_id), readout_id=str(item.readout_id),
        language=str(item.language), N_act=float(item.N_act),
        N_act_nonembedding=float(item.N_act_nonembedding), E=int(item.E), K=int(item.K),
        D_total=float(item.D_total), D_t=float(item.D_t), adjusted_loss=float(item.adjusted_loss),
        raw_nll=float(item.raw_nll), validation_token_count=int(item.validation_token_count),
        unigram_nll=float(item.unigram_nll), ladder=str(item.ladder), model_seed=int(item.model_seed),
    ) for item in frame[list(FIT_ROW_COLUMNS)].itertuples(index=False))


def _validate_sources(profile: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    receipt = read_json(Path(str(profile["inputs"]["input_receipt"]["path"])))
    frame = load_fit_rows_snapshot(receipt)
    input_sha256 = str(receipt.get("fit_rows_content_sha256"))
    if fit_rows_content_sha256(_fit_rows(frame)) != input_sha256:
        raise M5M6Error("source fit rows differ from their receipt")
    holdouts = _sealed(read_json(Path(str(profile["inputs"]["holdouts"]["path"]))),
        "manifest_sha256")
    official = _official_report(Path(str(profile["inputs"]["candidate_report"]["path"])))
    if holdouts.get("input_sha256") != input_sha256 or official.get("input_sha256") != input_sha256:
        raise M5M6Error("receipt, holdouts, and official report disagree")
    screen_path = Path(str(profile["inputs"]["interaction_screen_index"]["path"]))
    screen = _wave_index(screen_path)
    if screen.get("wave") != "screen" or set(screen.get("manifests", {}).get("M2-KN", {})) != set(PRIMARY_FOLDS):
        raise M5M6Error("M5/M6 requires the complete exact-fold M2-KN screen")
    if screen.get("input_receipt", {}).get("file_sha256") != profile["inputs"]["input_receipt"]["file_sha256"] \
            or screen.get("holdouts", {}).get("file_sha256") != profile["inputs"]["holdouts"]["file_sha256"]:
        raise M5M6Error("M2-KN screen belongs to different fit rows or memberships")
    screen_audit = _sealed(read_json(Path(str(profile["inputs"]["interaction_screen_audit"]["path"]))),
        "audit_sha256")
    m2_kn_audits = [item for item in screen_audit.get("items", []) if item.get("model") == "M2-KN"]
    if screen_audit.get("schema_version") != "stage1_m2_k_interaction_audit/v1" \
            or screen_audit.get("status") != "complete" \
            or screen_audit.get("screen_index_sha256") != screen["index_sha256"] \
            or {str(item.get("fold")) for item in m2_kn_audits} != set(PRIMARY_FOLDS) \
            or any(item.get("status") != "complete" for item in m2_kn_audits):
        raise M5M6Error("exact-fold M2-KN anchors lack the completed numerical audit")
    return {"receipt": receipt, "holdouts": holdouts, "official": official,
        "screen": screen, "screen_audit": screen_audit}, frame


def _write_component_parquet(path: Path, frame: Any, *, content_sha256: str) -> str:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if component_content_sha256(pd.read_parquet(path)) != content_sha256:
            raise M5M6Error(f"refusing to replace a different component snapshot: {path}")
        return sha256_file(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if component_content_sha256(pd.read_parquet(path)) != content_sha256:
                raise M5M6Error(f"immutable component snapshot collision: {path}")
        return sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(profile_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Freeze loss-blind family/script/remainder exposure before fitting."""
    profile = load_profile(profile_path)
    sources, frame = _validate_sources(profile)
    languages = tuple(sorted(set(frame["language"].astype(str))))
    metadata_path = Path(str(profile["inputs"]["language_metadata"]["path"]))
    metadata = load_language_metadata(metadata_path, languages)
    components, sets = composition_exposure_components(frame, metadata)
    expected_component_columns = ("D_target", *COMPONENT_COLUMNS)
    if tuple(components.columns) != expected_component_columns or not components.index.equals(frame.index):
        raise M5M6Error("composition builder returned a misaligned component table")
    if not np.array_equal(components["D_target"].to_numpy(dtype=np.float64),
            frame["D_t"].to_numpy(dtype=np.float64)):
        raise M5M6Error("composition builder changed target exposure")
    derived = frame.copy()
    for name in COMPONENT_COLUMNS:
        derived[name] = components[name].to_numpy(dtype=np.float64)
    reconstructed = derived["D_t"].to_numpy(dtype=np.float64) \
        + sum(derived[name].to_numpy(dtype=np.float64) for name in COMPONENT_COLUMNS)
    if not np.allclose(reconstructed, derived["D_total"].to_numpy(dtype=np.float64),
            rtol=1e-12, atol=1e-6):
        raise M5M6Error("composition snapshot does not reconstruct D_total")
    content_sha256 = component_content_sha256(derived)
    metadata_payload = language_metadata_payload(metadata)
    metadata_sha256 = language_metadata_sha256(metadata)
    sets_payload = composition_sets_payload(sets)
    sets_sha256 = composition_sets_sha256(sets)
    output_root = Path(str(profile["output_root"]))
    snapshot_path = output_root / "input" / "m5_m6_components.parquet"
    result = {
        "written": not dry_run, "rows": int(len(derived)),
        "trajectories": int(derived["trajectory_id"].nunique()),
        "readouts": int(derived["readout_id"].nunique()),
        "languages": int(derived["language"].nunique()),
        "component_content_sha256": content_sha256,
        "language_metadata_sha256": metadata_sha256,
        "composition_sets_sha256": sets_sha256,
        "source_input_sha256": sources["receipt"]["fit_rows_content_sha256"],
    }
    if dry_run:
        return result
    file_sha256 = _write_component_parquet(snapshot_path, derived, content_sha256=content_sha256)
    builder_path = Path(composition_exposure_components.__code__.co_filename).resolve()
    receipt = _seal({
        "schema_version": COMPONENT_SCHEMA, "exploratory": True,
        "profile": {"path": profile["profile_path"], "file_sha256": profile["profile_file_sha256"]},
        "runner": _runner_source_binding(),
        "component_builder": {"path": str(builder_path), "file_sha256": sha256_file(builder_path)},
        "sources": profile["inputs"],
        "source_input_sha256": sources["receipt"]["fit_rows_content_sha256"],
        "snapshot": {"path": str(snapshot_path.resolve()), "file_sha256": file_sha256,
            "component_content_sha256": content_sha256, "columns": list(COMPONENT_COLUMNS)},
        "language_metadata": metadata_payload,
        "language_metadata_sha256": metadata_sha256,
        "composition_sets": sets_payload,
        "composition_sets_sha256": sets_sha256,
        "claim_boundary": "Loss-blind immutable exposure derivation; no held-out loss enters component construction.",
    }, "receipt_sha256")
    receipt_path = output_root / "input" / "component_receipt.json"
    write_json_once(receipt_path, receipt)
    return {**result, "receipt_path": str(receipt_path), "receipt_sha256": receipt["receipt_sha256"]}


def _component_receipt(path: Path, profile: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = _sealed(read_json(path), "receipt_sha256")
    if receipt.get("schema_version") != COMPONENT_SCHEMA or receipt.get("exploratory") is not True:
        raise M5M6Error("unknown M5/M6 component receipt")
    if receipt.get("profile") != {"path": profile["profile_path"],
            "file_sha256": profile["profile_file_sha256"]} or receipt.get("sources") != profile["inputs"]:
        raise M5M6Error("component receipt belongs to a different profile")
    builder_path = Path(composition_exposure_components.__code__.co_filename).resolve()
    expected_sources = {
        "runner": _runner_source_binding(),
        "component_builder": {
            "path": str(builder_path), "file_sha256": sha256_file(builder_path)},
    }
    for name, expected_binding in expected_sources.items():
        if receipt.get(name) != expected_binding:
            raise M5M6Error(f"component receipt {name} source changed")
    snapshot = receipt.get("snapshot", {})
    snapshot_path = Path(str(snapshot.get("path")))
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != snapshot.get("file_sha256"):
        raise M5M6Error("component snapshot is missing or changed")
    return receipt


def _lambda_key(value: float) -> str:
    if value not in RIDGE_LAMBDAS:
        raise M5M6Error(f"unregistered ridge lambda: {value}")
    return format(value, ".12g")


def _historical_m2kn_result(screen: Mapping[str, Any], fold: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    path = Path(str(screen["manifests"]["M2-KN"][fold]))
    manifest = load_manifest(path, require_current_provenance=False)
    result, bound_manifest = _reduced_exploratory(result_path(manifest))
    if bound_manifest.get("fit", {}).get("model") != "M2-KN" \
            or bound_manifest.get("fit", {}).get("fold_id") != fold:
        raise M5M6Error("historical M2-KN anchor has the wrong identity")
    return result, bound_manifest, path


def _retained_m2kn_anchors(result: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    usable = [item for item in result.get("attempts", [])
        if item.get("success") and isinstance(item.get("coordinates"), list)
        and isinstance(item.get("objective"), (int, float))]
    usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
    anchors = tuple(np.asarray(item["coordinates"], dtype=np.float64) for item in usable[:ANCHOR_COUNT])
    if len(anchors) != ANCHOR_COUNT or any(item.shape != (len(outer_parameter_names("M5")),)
            or not np.isfinite(item).all() for item in anchors):
        raise M5M6Error("M2-KN screen does not provide eight usable exact-fold anchors")
    return anchors


def composition_starts(model: str, anchor_result: Mapping[str, Any], *, namespace: str) -> tuple[StartPoint, ...]:
    """Build the locked 8-start M5 or 216-start M6 design."""
    if model not in MODELS:
        raise M5M6Error(f"unknown composition model: {model}")
    anchors = _retained_m2kn_anchors(anchor_result)
    starts: list[StartPoint] = []
    if model == "M5":
        for anchor_index, anchor in enumerate(anchors):
            starts.append(StartPoint(f"{namespace}:a{anchor_index}:zero-extension", None,
                anchor.copy(), "m2_kn_anchor"))
    else:
        for anchor_index, anchor in enumerate(anchors):
            for family_index, script_index, other_index in itertools.product(range(3), repeat=3):
                extension = np.asarray((M6_SEEDS[family_index], M6_SEEDS[script_index],
                    M6_SEEDS[other_index]), dtype=np.float64)
                starts.append(StartPoint(
                    f"{namespace}:a{anchor_index}:f{family_index}-s{script_index}-o{other_index}",
                    None, np.r_[anchor, extension].astype(np.float64), "m2_kn_composition_seed"))
    expected = ANCHOR_COUNT if model == "M5" else ANCHOR_COUNT * 27
    if len(starts) != expected or len({item.start_id for item in starts}) != expected:
        raise M5M6Error("composition start construction changed")
    return tuple(starts)


def _regularization(strength: float, *, m2_reference_objective: float) -> dict[str, float | str]:
    if strength not in RIDGE_LAMBDAS or not math.isfinite(m2_reference_objective) \
            or m2_reference_objective <= 0.0:
        raise M5M6Error("invalid ridge strength or M2 reference objective")
    return {
        "schema_version": M5_M6_REGULARIZATION_SCHEMA,
        "lambda": float(strength),
        "m2_reference_objective": float(m2_reference_objective),
        "phi_endpoint_scale": M5_M6_PHI_ENDPOINT_SCALE,
        "psi_endpoint_scale": M5_M6_PSI_ENDPOINT_SCALE,
        "gamma_n_scale": M5_M6_GAMMA_N_SCALE,
    }


def _bind_composition(manifest: Mapping[str, Any], *, receipt_path: Path,
        receipt: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    unsigned["m5_m6_composition"] = {
        "schema_version": COMPONENT_SCHEMA,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "component_content_sha256": receipt["snapshot"]["component_content_sha256"],
        "composition_sets_sha256": receipt["composition_sets_sha256"],
        "review_only": True,
    }
    return _seal(unsigned, "manifest_sha256")


def _manifest_kwargs(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blas_threads": int(profile["blas_threads"]), "partition": profile.get("partition"),
        "account": profile.get("account"), "reservation": profile.get("reservation"),
        "memory": str(profile["memory"]), "time_limit": str(profile["time_limit"]),
        "concurrency": int(profile["array_concurrency"]),
    }
