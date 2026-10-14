"""Manifest planning and execution-state operations for the M5/M6 screen."""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
)

import numpy as np

from ..campaign import scientific_result
from ..data import sha256_file
from ..execution import (
    bind_membership,
    build_manifest,
    load_manifest,
    read_json,
    reduce,
    render_slurm,
    result_path,
    write_json_once,
    write_manifest,
    write_text_once,
)
from ..fit import OuterBound, StartPoint, initial_bounds
from ..interaction import (
    PRIMARY_FOLDS,
    RECEIPT_SCHEMA,
    RESULT_SCHEMA,
    _official_m2_paths,
    _seal,
    _sealed,
)
from .contracts import (
    AUDIT_INDEX_SCHEMA,
    INDEX_SCHEMA,
    M6_SEEDS,
    MINIMUM_EQUIVALENT_BASIN_STARTS,
    MODEL_ORDER,
    MODELS,
    M5M6Error,
    RIDGE_LAMBDAS,
    _bind_composition,
    _component_receipt,
    _historical_m2kn_result,
    _lambda_key,
    _manifest_kwargs,
    _regularization,
    _validate_sources,
    composition_starts,
    load_profile,
)

def _write_index(output_root: Path, stage: str, payload: Mapping[str, Any], *,
        schema: str = INDEX_SCHEMA) -> Path:
    value = _seal({"schema_version": schema, "stage": stage, "exploratory": True,
        **dict(payload)}, "index_sha256")
    path = output_root / "m5_m6" / f"{stage}-{value['index_sha256'][:12]}.json"
    write_json_once(path, value)
    return path


def _index(path: Path, *, schemas: Sequence[str] = (INDEX_SCHEMA, AUDIT_INDEX_SCHEMA)) -> Mapping[str, Any]:
    value = _sealed(read_json(path), "index_sha256")
    if value.get("schema_version") not in set(schemas) or value.get("exploratory") is not True:
        raise M5M6Error("unknown M5/M6 campaign index")
    return value


def plan_screen(profile_path: Path, *, component_receipt_path: Path | None = None,
        dry_run: bool = False) -> dict[str, Any]:
    """Plan all 42 independent model/fold/lambda manifests in one wave."""
    profile = load_profile(profile_path)
    sources, _ = _validate_sources(profile)
    output_root = Path(str(profile["output_root"]))
    receipt_path = component_receipt_path or output_root / "input" / "component_receipt.json"
    receipt = _component_receipt(receipt_path, profile)
    holdouts_path = Path(str(profile["inputs"]["holdouts"]["path"]))
    candidate_path = Path(str(profile["inputs"]["candidate_report"]["path"]))
    official_paths = _official_m2_paths(sources["official"])
    manifests: dict[str, dict[str, dict[str, str]]] = {
        model: {_lambda_key(value): {} for value in RIDGE_LAMBDAS} for model in MODELS}
    starts_total = 0
    tasks_total = 0
    for fold in PRIMARY_FOLDS:
        membership = bind_membership(holdouts_path, kind="primary", membership_id=fold)
        official_path = official_paths[fold]
        official = scientific_result(official_path)
        identity = official.get("fit_identity", {})
        objective = float(official.get("selected", {}).get("objective", math.nan))
        if identity.get("model") != "M2" or identity.get("fold_id") != fold \
                or not math.isfinite(objective) or objective <= 0.0:
            raise M5M6Error("official M2 ridge reference has the wrong identity or objective")
        anchor_result, anchor_manifest, anchor_manifest_path = _historical_m2kn_result(sources["screen"], fold)
        if anchor_manifest.get("membership", {}).get("membership_sha256") != membership["membership_sha256"]:
            raise M5M6Error("M2-KN anchor and official fold membership differ")
        scale_parent = [Path(str(item["path"])) for item in official.get("parents", [])
            if item.get("kind") == "scale_m0_ols"]
        if len(scale_parent) != 1:
            raise M5M6Error("official M2 reference lacks exactly one M0 scale parent")
        for model in MODELS:
            starts = composition_starts(model, anchor_result,
                namespace=f"m5-m6:screen:{fold}:{model}")
            for strength in RIDGE_LAMBDAS:
                key = _lambda_key(strength)
                regularization = _regularization(strength, m2_reference_objective=objective)
                safe_key = key.replace(".", "p")
                manifest = build_manifest(
                    output_root=output_root,
                    fit_rows_path=Path(str(receipt["snapshot"]["path"])),
                    fit_rows_content_sha256=str(receipt["source_input_sha256"]),
                    phase=f"m5-m6-screen-{fold}-{model.lower()}-lambda-{safe_key}",
                    model=model, fold_id=fold,
                    objective=str(identity["objective"]), delta=official.get("frozen_delta"),
                    chunk_size=int(profile["chunk_size"]), bounds=initial_bounds(model),
                    parent_results={"anchor_m2_kn": result_path(anchor_manifest),
                        "official_m2_reference": official_path, "scale_m0_ols": scale_parent[0]},
                    selection_report=candidate_path, membership=membership,
                    explicit_starts=starts, regularization=regularization,
                    review_only=True, action="m5_m6_screen", **_manifest_kwargs(profile),
                )
                manifest = _bind_composition(manifest, receipt_path=receipt_path, receipt=receipt)
                starts_total += len(starts)
                tasks_total += len(manifest["tasks"])
                if not dry_run:
                    manifests[model][key][fold] = str(write_manifest(manifest))
    expected = {"manifests": 2 * 3 * 7, "starts": 4704, "tasks": 2352}
    if {"manifests": 42, "starts": starts_total, "tasks": tasks_total} != expected:
        raise M5M6Error("M5/M6 screen workload changed")
    result = {**expected, "written": not dry_run, "submitted": False, "manifests_by_model": {
        "M5": 21, "M6": 21}, "manifests": manifests}
    if not dry_run:
        path = _write_index(output_root, "screen", {
            "output_root": str(output_root),
            "profile": {"path": profile["profile_path"], "file_sha256": profile["profile_file_sha256"]},
            "component_receipt": {"path": str(receipt_path.resolve()),
                "file_sha256": sha256_file(receipt_path), "receipt_sha256": receipt["receipt_sha256"]},
            "official_candidate_report": profile["inputs"]["candidate_report"],
            "holdouts": profile["inputs"]["holdouts"],
            "interaction_screen_index": profile["inputs"]["interaction_screen_index"],
            "model_order": list(MODEL_ORDER), "ridge_lambdas": list(RIDGE_LAMBDAS),
            "m6_seeds": list(M6_SEEDS), "manifests": manifests, "expected": expected,
            "screen_gates": profile["screen_gates"],
            "claim_boundary": "Retrospective development screen on already-inspected primary folds.",
        })
        result["index_path"] = str(path)
    return result


def _iter_manifest_items(index: Mapping[str, Any]) -> Iterable[tuple[str, str, str, Path]]:
    manifests = index.get("manifests", {})
    if not isinstance(manifests, Mapping):
        raise M5M6Error("index manifest inventory is malformed")
    for model, lambda_items in manifests.items():
        if not isinstance(lambda_items, Mapping):
            raise M5M6Error("index model inventory is malformed")
        for lambda_key, fold_items in lambda_items.items():
            if not isinstance(fold_items, Mapping):
                raise M5M6Error("index lambda inventory is malformed")
            for fold, value in fold_items.items():
                yield str(model), str(lambda_key), str(fold), Path(str(value))


def emit_slurm(index_path: Path) -> dict[str, Any]:
    index = _index(index_path)
    scripts: dict[str, str] = {}
    root = Path(str(index["output_root"]))
    for model, lambda_key, fold, manifest_path in _iter_manifest_items(index):
        manifest = load_manifest(manifest_path)
        script, _ = render_slurm(manifest_path, manifest)
        safe_lambda = lambda_key.replace(".", "p")
        path = root / "slurm" / (
            f"m5-m6-{index['stage']}-{model}-{fold}-lambda-{safe_lambda}-"
            f"{manifest['manifest_sha256'][:12]}.sbatch")
        write_text_once(path, script)
        scripts[f"{model}/lambda={lambda_key}/{fold}"] = str(path)
    return {"index_sha256": index["index_sha256"], "count": len(scripts),
        "scripts": scripts, "submitted": False}


def status(index_path: Path) -> dict[str, Any]:
    index = _index(index_path)
    rows = []
    for model, lambda_key, fold, manifest_path in _iter_manifest_items(index):
        manifest = load_manifest(manifest_path)
        stem = manifest["manifest_sha256"][:20]
        chunk_root = Path(str(index["output_root"])) / "chunks" / stem
        present = sum((chunk_root / f"task-{item['task_index']:06d}.json").exists()
            for item in manifest["tasks"])
        rows.append({"key": f"{model}/lambda={lambda_key}/{fold}",
            "tasks": len(manifest["tasks"]), "chunks": present,
            "result": result_path(manifest).exists()})
    return {"stage": index["stage"],
        "complete": all(item["chunks"] == item["tasks"] for item in rows), "rows": rows}


def reduce_index(index_path: Path) -> dict[str, Any]:
    index = _index(index_path)
    paths: dict[str, dict[str, dict[str, str]]] = {}
    for model, lambda_key, fold, manifest_path in _iter_manifest_items(index):
        manifest = load_manifest(manifest_path)
        result = reduce(manifest)
        if result.get("status") != "reduced" or not isinstance(result.get("selected"), Mapping):
            raise M5M6Error(f"{model}/lambda={lambda_key}/{fold} produced no selected fit")
        paths.setdefault(model, {}).setdefault(lambda_key, {})[fold] = str(result_path(manifest))
    return {"stage": index["stage"], "count": sum(
        len(folds) for lambdas in paths.values() for folds in lambdas.values()), "results": paths}


def _reduced_current_exploratory(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load a reduced campaign result against its exact current source bytes."""
    result = _sealed(read_json(path), "result_sha256")
    if result.get("schema_version") != RESULT_SCHEMA or not result.get("review_only"):
        raise M5M6Error("result is not an exploratory reduced fit")
    receipt = _sealed(read_json(Path(str(result["receipt_path"]))), "receipt_sha256")
    if receipt.get("schema_version") != RECEIPT_SCHEMA \
            or receipt.get("manifest_sha256") != result.get("manifest_sha256"):
        raise M5M6Error("exploratory completion receipt differs")
    manifest = load_manifest(Path(str(result["manifest_path"])))
    if manifest.get("review_only") is not True \
            or manifest["manifest_sha256"] != result["manifest_sha256"] \
            or manifest.get("membership") != result.get("membership") \
            or result.get("status") != "reduced" \
            or not isinstance(result.get("selected"), Mapping):
        raise M5M6Error("exploratory result does not match its manifest")
    return result, manifest


def _campaign_result(manifest_path: Path, index: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = load_manifest(manifest_path)
    result, bound_manifest = _reduced_current_exploratory(result_path(manifest))
    binding = bound_manifest.get("m5_m6_composition", {})
    if binding.get("review_only") is not True \
            or binding.get("receipt_sha256") != index.get("component_receipt", {}).get("receipt_sha256"):
        raise M5M6Error("M5/M6 result lost its composition binding")
    return result, bound_manifest


def plan_audits(screen_index_path: Path) -> dict[str, Any]:
    """Plan three-start basin and one-start direct checks for all 42 cells."""
    screen = _index(screen_index_path, schemas=(INDEX_SCHEMA,))
    if screen.get("stage") != "screen":
        raise M5M6Error("audit planning requires the screen index")
    root = Path(str(screen["output_root"]))
    planned: dict[str, dict[str, dict[str, str]]] = {}
    starts_total = 0
    tasks_total = 0
    for model, lambda_key, fold, manifest_path in _iter_manifest_items(screen):
        result, base = _campaign_result(manifest_path, screen)
        usable = [item for item in result.get("attempts", [])
            if item.get("success") and isinstance(item.get("coordinates"), list)]
        usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
        if len(usable) < MINIMUM_EQUIVALENT_BASIN_STARTS:
            raise M5M6Error(f"{model}/lambda={lambda_key}/{fold} lacks three successful starts")
        coordinates = tuple(np.asarray(item["coordinates"], dtype=np.float64)
            for item in usable[:MINIMUM_EQUIVALENT_BASIN_STARTS])
        parents = {str(item["kind"]): Path(str(item["path"])) for item in base.get("parents", [])}
        parents["grid"] = result_path(base)
        selection = base.get("selection_report")
        selection_path = None if not isinstance(selection, Mapping) else Path(str(selection["path"]))
        planned.setdefault(model, {}).setdefault(lambda_key, {})
        for action, starts in (
            ("basin", tuple(StartPoint(f"m5-m6-audit:{model}:{lambda_key}:{fold}:basin:{number}",
                None, value, "basin") for number, value in enumerate(coordinates))),
            ("direct", (StartPoint(f"m5-m6-audit:{model}:{lambda_key}:{fold}:direct", None,
                np.asarray(result["selected"]["coordinates"], dtype=np.float64), "direct"),)),
        ):
            manifest = build_manifest(
                output_root=root, fit_rows_path=Path(str(base["input"]["fit_rows_path"])),
                fit_rows_content_sha256=str(base["input"]["fit_rows_content_sha256"]),
                phase=f"{base['phase']}-{action}", model=model, fold_id=fold,
                objective=str(base["fit"]["objective"]), delta=base["fit"]["delta"],
                chunk_size=int(base["execution"]["chunk_size"]),
                bounds=tuple(OuterBound(**item) for item in base["fit"]["bounds"]),
                parent_results=parents, selection_report=selection_path,
                membership=base["membership"], explicit_starts=starts,
                regularization=base["fit"]["regularization"], review_only=True, action=action,
                blas_threads=int(base["execution"]["blas_threads"]),
                partition=base["execution"].get("partition"), account=base["execution"].get("account"),
                reservation=base["execution"].get("reservation"), memory=str(base["execution"]["memory"]),
                time_limit=str(base["execution"]["time_limit"]),
                concurrency=int(base["execution"]["concurrency"]),
            )
            unsigned = dict(manifest)
            unsigned.pop("manifest_sha256", None)
            unsigned["m5_m6_composition"] = dict(base["m5_m6_composition"])
            manifest = _seal(unsigned, "manifest_sha256")
            planned[model][lambda_key][f"{fold}:{action}"] = str(write_manifest(manifest))
            starts_total += len(starts)
            tasks_total += len(manifest["tasks"])
    expected = {"manifests": 84, "starts": 168, "tasks": 126}
    if {"manifests": 84, "starts": starts_total, "tasks": tasks_total} != expected:
        raise M5M6Error("M5/M6 audit workload changed")
    path = _write_index(root, "screen-audit", {
        "output_root": str(root), "profile": screen["profile"],
        "component_receipt": screen["component_receipt"],
        "screen_index": {"path": str(screen_index_path.resolve()),
            "index_sha256": screen["index_sha256"]},
        "manifests": planned, "expected": expected,
    }, schema=AUDIT_INDEX_SCHEMA)
    return {"index_path": str(path), **expected, "manifests": planned, "submitted": False}
