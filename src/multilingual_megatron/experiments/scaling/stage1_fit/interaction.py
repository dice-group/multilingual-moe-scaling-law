"""Bounded exploratory M2-by-language-count interaction campaign.

This sidecar deliberately cannot alter the registered M0/M1/M2 candidate
selection.  It reuses the frozen Stage 1 rows, memberships, Huber scales,
executor, and official audited M2 fits while marking every new fit review-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .campaign import CampaignError, _record_prediction, _records, scientific_result
from .data import canonical_json_sha256, load_fit_rows_snapshot, sha256_file
from .execution import (
    ExecutionError,
    bind_membership,
    build_manifest,
    load_manifest,
    read_json,
    reduce,
    render_slurm,
    result_path,
    validate_manifest,
    write_json_once,
    write_manifest,
    write_text_once,
)
from .fit import (
    EXPLORATORY_INTERACTION_MODELS,
    OBJECTIVE_RELATIVE_TOLERANCE,
    PREDICTION_ABSOLUTE_TOLERANCE,
    SharedParameters,
    StartPoint,
    deterministic_local_perturbations,
    effective_exponents,
    equivalence_check,
    initial_bounds,
    multilingual_k_coordinate,
    outer_parameter_names,
    predict,
    predictor_terms,
)
from .selection import (
    PAIRED_BOOTSTRAP_REPLICATES,
    select_model_paired_one_standard_error,
    trajectory_balanced_r2,
    trajectory_balanced_rmse,
    trajectory_mean_squared_errors,
)

PRIMARY_FOLDS = ("largest_N_act", "E32", "K200_common_languages")
MODEL_ORDER = ("M2",) + EXPLORATORY_INTERACTION_MODELS
WAVE_SCHEMA = "stage1_m2_k_interaction_wave/v1"
AUDIT_INDEX_SCHEMA = "stage1_m2_k_interaction_audit_index/v1"
AUDIT_SCHEMA = "stage1_m2_k_interaction_audit/v1"
REPORT_SCHEMA = "stage1_m2_k_interaction_report/v1"
SUBMISSION_SCHEMA = "stage1_m2_k_interaction_submission/v1"
OFFICIAL_REPORT_SCHEMA = "stage1_fit_candidate_report/v5"
RESULT_SCHEMA = "stage1_fit_reduced/v2"
RECEIPT_SCHEMA = "stage1_fit_complete/v2"


class InteractionError(RuntimeError):
    """Raised when the exploratory sidecar loses a frozen binding or gate."""


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return {**unsigned, field: canonical_json_sha256(unsigned)}


def _sealed(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    unsigned = dict(payload)
    digest = unsigned.pop(field, None)
    if not isinstance(digest, str) or digest != canonical_json_sha256(unsigned):
        raise InteractionError(f"invalid {field}")
    return payload


def _profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    required = {"output_root", "chunk_size", "blas_threads", "memory", "time_limit", "array_concurrency"}
    if not isinstance(value, dict) or not required <= set(value):
        raise InteractionError("interaction execution profile is incomplete")
    if int(value["chunk_size"]) < 1 or int(value["blas_threads"]) != 1:
        raise InteractionError("interaction profile requires positive chunks and one BLAS thread")
    return value


def _official_report(path: Path) -> Mapping[str, Any]:
    report = _sealed(read_json(path), "report_sha256")
    if report.get("schema_version") != OFFICIAL_REPORT_SCHEMA:
        raise InteractionError("official candidate report schema changed")
    if report.get("selection", {}).get("selected_model") != "M2":
        raise InteractionError("interaction campaign requires the frozen selected M2 law")
    return report


def _official_m2_paths(report: Mapping[str, Any]) -> dict[str, Path]:
    paths = {
        str(item["fold_id"]): Path(str(item["path"]))
        for item in report.get("source_results", [])
        if item.get("role") == "primary" and item.get("model") == "M2"
    }
    if set(paths) != set(PRIMARY_FOLDS):
        raise InteractionError("official report does not bind exactly three M2 primary folds")
    return paths


def _result_parent(result: Mapping[str, Any], kind: str) -> Path:
    matches = [Path(str(item["path"])) for item in result.get("parents", []) if item.get("kind") == kind]
    if len(matches) != 1:
        raise InteractionError(f"result does not bind exactly one {kind} parent")
    return matches[0]


def _retained_m2_anchors(result: Mapping[str, Any], *, count: int) -> tuple[np.ndarray, ...]:
    selected = result.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("coordinates"), list):
        raise InteractionError("official M2 result lacks selected coordinates")
    manifest = load_manifest(Path(str(result["manifest_path"])), require_current_provenance=False)
    bounds = initial_bounds("M2")
    widths = np.asarray([bound.upper - bound.lower for bound in bounds], dtype=np.float64)
    anchors: list[np.ndarray] = [np.asarray(selected["coordinates"], dtype=np.float64)]
    usable = [
        item for item in result.get("attempts", [])
        if item.get("success") and isinstance(item.get("objective"), (int, float))
        and isinstance(item.get("coordinates"), list)
    ]
    usable.sort(key=lambda item: (float(item["objective"]), str(item.get("start_id", ""))))
    for item in usable:
        if len(anchors) == count:
            break
        candidate = np.asarray(item["coordinates"], dtype=np.float64)
        if candidate.shape != anchors[0].shape or not np.isfinite(candidate).all():
            continue
        if all(float(np.max(np.abs((candidate - existing) / widths))) > 1e-6 for existing in anchors):
            anchors.append(candidate)
    if len(anchors) < count:
        for point in deterministic_local_perturbations(anchors[0], bounds, relative_step=0.02):
            anchors.append(np.asarray(point.values, dtype=np.float64))
            if len(anchors) == count:
                break
    if len(anchors) != count or any(value.shape != (10,) for value in anchors):
        raise InteractionError("could not construct the requested deterministic M2 anchors")
    if manifest["fit"]["model"] != "M2":
        raise InteractionError("anchor manifest is not M2")
    return tuple(anchors)


def _gamma_seeds(anchor: np.ndarray, *, effect: str) -> tuple[float, float, float]:
    endpoints = anchor[:2] if effect == "n" else anchor[2:4]
    feasible_lower = max(-1.0, 0.011 - float(np.min(endpoints)))
    values = (0.25 * feasible_lower, 0.0, 0.25)
    if not values[0] < values[1] < values[2]:
        raise InteractionError("gamma seed interval collapsed")
    return values


def interaction_starts(model: str, result: Mapping[str, Any], *, wave: str) -> tuple[StartPoint, ...]:
    if model not in EXPLORATORY_INTERACTION_MODELS or wave not in {"smoke", "screen"}:
        raise InteractionError("unknown interaction model or wave")
    anchor_count = 1 if wave == "smoke" else 8
    anchors = _retained_m2_anchors(result, count=anchor_count)
    starts: list[StartPoint] = []
    for anchor_index, anchor in enumerate(anchors):
        n_values = _gamma_seeds(anchor, effect="n") if model in {"M2-KN", "M2-KND"} else (0.0,)
        d_values = _gamma_seeds(anchor, effect="d") if model in {"M2-KD", "M2-KND"} else (0.0,)
        for n_index, gamma_n in enumerate(n_values):
            for d_index, gamma_d in enumerate(d_values):
                suffix = []
                values = anchor.tolist()
                if model in {"M2-KN", "M2-KND"}:
                    values.append(gamma_n); suffix.append(f"n{n_index}")
                if model in {"M2-KD", "M2-KND"}:
                    values.append(gamma_d); suffix.append(f"d{d_index}")
                starts.append(StartPoint(
                    start_id=f"interaction:{wave}:a{anchor_index}:{'-'.join(suffix)}",
                    start_index=None,
                    values=np.asarray(values, dtype=np.float64),
                    kind=f"interaction_{wave}",
                ))
    expected = {"M2-KN": 3, "M2-KD": 3, "M2-KND": 9}[model] * anchor_count
    if len(starts) != expected or len({item.start_id for item in starts}) != expected:
        raise InteractionError("interaction start construction changed")
    return tuple(starts)


def _phase(wave: str, fold: str, model: str) -> str:
    return f"interaction-{wave}-{fold}-{model.lower().replace('-', '')}"


def _write_wave_index(output_root: Path, wave: str, payload: Mapping[str, Any]) -> Path:
    index = _seal({"schema_version": WAVE_SCHEMA, "wave": wave, **dict(payload)}, "index_sha256")
    path = output_root / "interaction" / f"{wave}_index.json"
    write_json_once(path, index)
    return path


def plan_wave(*, wave: str, profile_path: Path, input_receipt_path: Path, holdouts_path: Path,
    candidate_report_path: Path, dry_run: bool = False) -> dict[str, Any]:
    """Plan all nine smoke or screen manifests without scheduler side effects."""
    profile = _profile(profile_path)
    output_root = Path(str(profile["output_root"]))
    receipt = read_json(input_receipt_path)
    report = _official_report(candidate_report_path)
    if report.get("input_sha256") != receipt.get("fit_rows_content_sha256"):
        raise InteractionError("official report and input receipt differ")
    holdouts = _sealed(read_json(holdouts_path), "manifest_sha256")
    if holdouts.get("input_sha256") != receipt.get("fit_rows_content_sha256"):
        raise InteractionError("holdouts and input receipt differ")
    official_paths = _official_m2_paths(report)
    planned: dict[str, dict[str, str]] = {model: {} for model in EXPLORATORY_INTERACTION_MODELS}
    totals = {"starts": 0, "tasks": 0}
    for fold in PRIMARY_FOLDS:
        membership = bind_membership(holdouts_path, kind="primary", membership_id=fold)
        anchor_path = official_paths[fold]
        anchor = scientific_result(anchor_path)
        identity = anchor.get("fit_identity", {})
        if identity.get("model") != "M2" or identity.get("fold_id") != fold:
            raise InteractionError("official M2 source has the wrong identity")
        scale_path = _result_parent(anchor, "scale_m0_ols")
        for model in EXPLORATORY_INTERACTION_MODELS:
            starts = interaction_starts(model, anchor, wave=wave)
            manifest = build_manifest(
                output_root=output_root,
                fit_rows_path=Path(str(receipt["fit_rows_path"])),
                fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]),
                phase=_phase(wave, fold, model),
                model=model,
                fold_id=fold,
                objective=str(identity["objective"]),
                delta=anchor.get("frozen_delta"),
                chunk_size=int(profile["chunk_size"]),
                bounds=initial_bounds(model),
                parent_results={"anchor_m2": anchor_path, "scale_m0_ols": scale_path},
                selection_report=candidate_report_path,
                membership=membership,
                explicit_starts=starts,
                review_only=True,
                action=f"interaction_{wave}",
                blas_threads=int(profile["blas_threads"]),
                partition=profile.get("partition"),
                account=profile.get("account"),
                reservation=profile.get("reservation"),
                memory=str(profile["memory"]),
                time_limit=str(profile["time_limit"]),
                concurrency=int(profile["array_concurrency"]),
            )
            totals["starts"] += int(manifest["fit"]["start_count"])
            totals["tasks"] += len(manifest["tasks"])
            if not dry_run:
                planned[model][fold] = str(write_manifest(manifest))
    expected = {"smoke": {"starts": 45, "tasks": 9}, "screen": {"starts": 360, "tasks": 30}}[wave]
    if totals != expected:
        raise InteractionError(f"{wave} workload changed: {totals} != {expected}")
    result: dict[str, Any] = {"wave": wave, **totals, "written": not dry_run, "manifests": planned}
    if not dry_run:
        path = _write_wave_index(output_root, wave, {
            "exploratory": True,
            "output_root": str(output_root.resolve()),
            "profile": {"path": str(profile_path.resolve()), "file_sha256": sha256_file(profile_path)},
            "input_receipt": {"path": str(input_receipt_path.resolve()), "file_sha256": sha256_file(input_receipt_path)},
            "holdouts": {"path": str(holdouts_path.resolve()), "file_sha256": sha256_file(holdouts_path), "manifest_sha256": holdouts["manifest_sha256"]},
            "official_candidate_report": {"path": str(candidate_report_path.resolve()), "file_sha256": sha256_file(candidate_report_path), "report_sha256": report["report_sha256"]},
            "official_m2_results": {fold: str(path.resolve()) for fold, path in official_paths.items()},
            "manifests": planned,
            "expected": totals,
        })
        result["index_path"] = str(path)
    return result


def _wave_index(path: Path) -> Mapping[str, Any]:
    index = _sealed(read_json(path), "index_sha256")
    if index.get("schema_version") not in {WAVE_SCHEMA, AUDIT_INDEX_SCHEMA}:
        raise InteractionError("unknown interaction index schema")
    return index


def emit_slurm(index_path: Path) -> dict[str, Any]:
    index = _wave_index(index_path)
    scripts: dict[str, str] = {}
    for model, folds in index["manifests"].items():
        for fold, manifest_value in folds.items():
            manifest_path = Path(str(manifest_value))
            manifest = load_manifest(manifest_path)
            script, _ = render_slurm(manifest_path, manifest)
            key = f"{model}/{fold}"
            output = Path(str(index["output_root"])) / "slurm" / f"{index['wave']}-{model}-{fold}-{manifest['manifest_sha256'][:12]}.sbatch"
            write_text_once(output, script)
            scripts[key] = str(output)
    return {"index_sha256": index["index_sha256"], "scripts": scripts, "count": len(scripts), "submitted": False}


def record_submission(index_path: Path, jobs: Sequence[str], *, attempt: int) -> dict[str, Any]:
    index = _wave_index(index_path)
    expected = {f"{model}/{fold}" for model, folds in index["manifests"].items() for fold in folds}
    parsed: dict[str, str] = {}
    for item in jobs:
        key, separator, job_id = item.partition("=")
        if separator != "=" or not job_id.isdigit() or key in parsed:
            raise InteractionError(f"malformed submission binding: {item}")
        parsed[key] = job_id
    if set(parsed) != expected or attempt < 1:
        raise InteractionError("submission bindings do not exactly cover the index")
    payload = _seal({
        "schema_version": SUBMISSION_SCHEMA,
        "index_path": str(index_path.resolve()),
        "index_sha256": index["index_sha256"],
        "wave": index["wave"],
        "attempt": attempt,
        "jobs": parsed,
    }, "submission_sha256")
    path = Path(str(index["output_root"])) / "submissions" / f"{index['wave']}-attempt-{attempt:02d}.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def reduce_index(index_path: Path) -> dict[str, Any]:
    index = _wave_index(index_path)
    paths: dict[str, dict[str, str]] = {}
    for model, folds in index["manifests"].items():
        paths[model] = {}
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value)))
            result = reduce(manifest)
            paths[model][fold] = str(result_path(manifest))
            if result.get("status") != "reduced" or not result.get("selected"):
                raise InteractionError(f"{model}/{fold} produced no selected fit")
    return {"wave": index["wave"], "results": paths, "count": sum(len(value) for value in paths.values())}


def status_index(index_path: Path) -> dict[str, Any]:
    index = _wave_index(index_path)
    rows = []
    for model, folds in index["manifests"].items():
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value)))
            root = Path(str(index["output_root"]))
            stem = manifest["manifest_sha256"][:20]
            chunk_root = root / "chunks" / stem
            present = sum((chunk_root / f"task-{task['task_index']:06d}.json").exists() for task in manifest["tasks"])
            rows.append({"key": f"{model}/{fold}", "tasks": len(manifest["tasks"]), "chunks": present,
                "result": (root / "results" / f"{stem}.json").exists()})
    return {"wave": index["wave"], "complete": all(row["chunks"] == row["tasks"] for row in rows), "rows": rows}


def _reduced_exploratory(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = _sealed(read_json(path), "result_sha256")
    if result.get("schema_version") != RESULT_SCHEMA or not result.get("review_only"):
        raise InteractionError("result is not an exploratory reduced fit")
    receipt = _sealed(read_json(Path(str(result["receipt_path"]))), "receipt_sha256")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("manifest_sha256") != result.get("manifest_sha256"):
        raise InteractionError("exploratory completion receipt differs")
    manifest = load_manifest(Path(str(result["manifest_path"])), require_current_provenance=False)
    if (manifest.get("review_only") is not True or manifest["manifest_sha256"] != result["manifest_sha256"]
            or manifest.get("membership") != result.get("membership")
            or result.get("status") != "reduced" or not isinstance(result.get("selected"), Mapping)):
        raise InteractionError("exploratory result does not match its manifest")
    return result, manifest


def smoke_report(index_path: Path) -> dict[str, Any]:
    index = _wave_index(index_path)
    if index.get("wave") != "smoke":
        raise InteractionError("smoke report requires the smoke index")
    rows = []
    for model, folds in index["manifests"].items():
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value)))
            result, _ = _reduced_exploratory(result_path(manifest))
            successes = sum(bool(item.get("success")) for item in result.get("attempts", []))
            rows.append({"model": model, "fold": fold, "starts": manifest["fit"]["start_count"],
                "successful_starts": successes, "selected_objective": result["selected"]["objective"],
                "boundary_contacts": result["selected"].get("boundary_contacts", [])})
    healthy = all(row["successful_starts"] >= 1 and not row["boundary_contacts"] for row in rows)
    payload = _seal({"schema_version": "stage1_m2_k_interaction_smoke/v1", "status": "complete" if healthy else "failed",
        "index_sha256": index["index_sha256"], "rows": rows}, "report_sha256")
    path = Path(str(index["output_root"])) / "reports" / "smoke_readiness.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def plan_audits(screen_index_path: Path) -> dict[str, Any]:
    screen = _wave_index(screen_index_path)
    if screen.get("wave") != "screen":
        raise InteractionError("audit planning requires the screen index")
    output_root = Path(str(screen["output_root"]))
    planned: dict[str, dict[str, str]] = {}
    totals = {"starts": 0, "tasks": 0}
    for model, folds in screen["manifests"].items():
        planned[model] = {}
        for fold, manifest_value in folds.items():
            base_path = Path(str(manifest_value)); base = load_manifest(base_path)
            result, _ = _reduced_exploratory(result_path(base))
            usable = [item for item in result.get("attempts", []) if item.get("success") and item.get("coordinates")]
            usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
            if len(usable) < 3:
                raise InteractionError(f"{model}/{fold} lacks three successful screen starts")
            parents = {str(item["kind"]): Path(str(item["path"])) for item in base.get("parents", [])}
            parents["grid"] = result_path(base)
            selection = None if base.get("selection_report") is None else Path(str(base["selection_report"]["path"]))
            for action, starts in (
                ("basin", tuple(StartPoint(f"basin:{index}", None, np.asarray(item["coordinates"], dtype=np.float64), "basin") for index, item in enumerate(usable[:3]))),
                ("direct", (StartPoint("direct:best", None, np.asarray(result["selected"]["coordinates"], dtype=np.float64), "direct"),)),
            ):
                manifest = build_manifest(
                    output_root=output_root,
                    fit_rows_path=Path(str(base["input"]["fit_rows_path"])),
                    fit_rows_content_sha256=str(base["input"]["fit_rows_content_sha256"]),
                    phase=f"{base['phase']}-{action}", model=model, fold_id=fold,
                    objective=str(base["fit"]["objective"]), delta=base["fit"]["delta"],
                    chunk_size=int(base["execution"]["chunk_size"]), bounds=initial_bounds(model),
                    parent_results=parents, selection_report=selection, membership=base["membership"],
                    explicit_starts=starts, review_only=True, action=action,
                    blas_threads=int(base["execution"]["blas_threads"]), partition=base["execution"].get("partition"),
                    account=base["execution"].get("account"), reservation=base["execution"].get("reservation"),
                    memory=str(base["execution"]["memory"]), time_limit=str(base["execution"]["time_limit"]),
                    concurrency=int(base["execution"]["concurrency"]),
                )
                path = write_manifest(manifest)
                planned[model][f"{fold}:{action}"] = str(path)
                totals["starts"] += int(manifest["fit"]["start_count"]); totals["tasks"] += len(manifest["tasks"])
    if totals != {"starts": 36, "tasks": 18}:
        raise InteractionError(f"screen audit workload changed: {totals}")
    index = _seal({"schema_version": AUDIT_INDEX_SCHEMA, "wave": "screen-audit", "exploratory": True,
        "output_root": str(output_root), "screen_index": {"path": str(screen_index_path.resolve()), "index_sha256": screen["index_sha256"]},
        "manifests": planned, "expected": totals}, "index_sha256")
    path = output_root / "interaction" / "screen_audit_index.json"; write_json_once(path, index)
    return {"index_path": str(path), **totals, "manifests": planned}


def _test_metrics(frame: Any, membership: Mapping[str, Any], selected: Mapping[str, Any], model: str) -> dict[str, Any]:
    rows = frame[frame["trajectory_id"].isin(membership["test_trajectory_ids"])].copy()
    if membership["test_readout_ids"]:
        rows = rows[rows["readout_id"].isin(membership["test_readout_ids"])]
    if membership["score_languages"]:
        rows = rows[rows["language"].isin(membership["score_languages"])]
    coefficients = {str(item["language"]): item for item in selected["language_coefficients"]}
    if rows.empty or not set(rows["language"].astype(str)) <= set(coefficients):
        raise InteractionError("held-out rows are empty or lack nuisance coefficients")
    nuisance = [coefficients[str(language)] for language in rows["language"]]
    prediction = predict(model, SharedParameters(**selected["shared_parameters"]), rows["N_act"], rows["D_t"],
        rows["E"], rows["K"], [item["L_inf"] for item in nuisance], [item["A"] for item in nuisance],
        [item["B"] for item in nuisance])
    errors = trajectory_mean_squared_errors(rows["adjusted_loss"], prediction, rows["trajectory_id"])
    return {"trajectory_mse": errors,
        "trajectory_balanced_rmse": trajectory_balanced_rmse(rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "trajectory_balanced_r2": trajectory_balanced_r2(rows["adjusted_loss"], prediction, rows["trajectory_id"]),
        "rows": int(len(rows)), "trajectories": int(rows["trajectory_id"].nunique())}


def audit_screen(screen_index_path: Path, audit_index_path: Path) -> dict[str, Any]:
    screen = _wave_index(screen_index_path); audit_index = _wave_index(audit_index_path)
    if screen.get("wave") != "screen" or audit_index.get("wave") != "screen-audit":
        raise InteractionError("screen audit indices have the wrong wave")
    if audit_index.get("screen_index", {}).get("index_sha256") != screen["index_sha256"]:
        raise InteractionError("audit index does not bind the screen")
    official = _official_report(Path(str(screen["official_candidate_report"]["path"])))
    official_paths = _official_m2_paths(official)
    items = []
    for model, folds in screen["manifests"].items():
        for fold, base_value in folds.items():
            base_manifest = load_manifest(Path(str(base_value)))
            base_result, _ = _reduced_exploratory(result_path(base_manifest))
            basin_manifest = load_manifest(Path(str(audit_index["manifests"][model][f"{fold}:basin"])))
            direct_manifest = load_manifest(Path(str(audit_index["manifests"][model][f"{fold}:direct"])))
            basin_result, _ = _reduced_exploratory(result_path(basin_manifest))
            direct_result, _ = _reduced_exploratory(result_path(direct_manifest))
            retained = base_result["selected"]
            retained_prediction = _record_prediction(retained, manifest=base_manifest)
            equivalent = 0
            for record in _records(basin_manifest):
                if not record.get("success") or not isinstance(record.get("objective"), (int, float)):
                    continue
                check = equivalence_check(float(retained["objective"]), float(record["objective"]), retained_prediction,
                    _record_prediction(record, manifest=basin_manifest), objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
                    prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)
                equivalent += int(check.equivalent)
            direct = direct_result.get("selected", {}).get("direct_equivalence", {})
            contacts = retained.get("boundary_contacts", [])
            parameters = SharedParameters(**retained["shared_parameters"])
            alpha, beta, _, _ = effective_exponents(parameters, np.asarray([1.0, 4.0, 16.0, 32.0])[:, None],
                np.asarray([1.0, 12.0, 50.0, 200.0])[None, :], model=model)
            official_m2 = scientific_result(official_paths[fold])["selected"]
            m2_parameters = SharedParameters(**official_m2["shared_parameters"])
            nested = SharedParameters(**{**m2_parameters.as_dict(), "gamma_n": 0.0, "gamma_d": 0.0})
            predictors = (np.asarray([1e8, 2e8, 4e8, 8e8]), np.asarray([1e8, 5e8, 2e9, 8e9]),
                np.asarray([1.0, 4.0, 16.0, 32.0]), np.asarray([1.0, 12.0, 50.0, 200.0]))
            nested_ok = all(np.array_equal(first, second) for first, second in zip(
                predictor_terms("M2", m2_parameters, *predictors), predictor_terms(model, nested, *predictors)))
            complete = equivalent >= 3 and bool(direct.get("resolved")) and not contacts and nested_ok \
                and bool(np.all(alpha > 0.0)) and bool(np.all(beta > 0.0))
            items.append({"model": model, "fold": fold, "status": "complete" if complete else "failed",
                "equivalent_basin_starts": equivalent, "direct_equivalence": direct, "boundary_contacts": contacts,
                "exact_m2_nesting": nested_ok, "minimum_alpha": float(np.min(alpha)), "minimum_beta": float(np.min(beta)),
                "base_result_sha256": base_result["result_sha256"], "basin_result_sha256": basin_result["result_sha256"],
                "direct_result_sha256": direct_result["result_sha256"]})
    status = "complete" if all(item["status"] == "complete" for item in items) else "failed"
    payload = _seal({"schema_version": AUDIT_SCHEMA, "status": status, "screen_index_sha256": screen["index_sha256"],
        "audit_index_sha256": audit_index["index_sha256"], "items": items}, "audit_sha256")
    path = Path(str(screen["output_root"])) / "reports" / "screen_numerical_audit.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def build_report(screen_index_path: Path, audit_path: Path) -> dict[str, Any]:
    screen = _wave_index(screen_index_path); audit = _sealed(read_json(audit_path), "audit_sha256")
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != "complete" or audit.get("screen_index_sha256") != screen["index_sha256"]:
        raise InteractionError("screen numerical audit is incomplete or mismatched")
    official_path = Path(str(screen["official_candidate_report"]["path"])); official = _official_report(official_path)
    receipt_path = Path(str(screen["input_receipt"]["path"])); receipt = read_json(receipt_path)
    frame = load_fit_rows_snapshot(receipt)
    errors: dict[str, dict[str, dict[str, float]]] = {"M2": {}}
    metrics: dict[str, dict[str, Any]] = {"M2": {}}
    sources = []
    for fold in PRIMARY_FOLDS:
        metric = official["primary_metrics"]["M2"][fold]
        errors["M2"][fold] = {str(key): float(value) for key, value in metric["trajectory_mse"].items()}
        metrics["M2"][fold] = metric
    for model, folds in screen["manifests"].items():
        errors[model] = {}; metrics[model] = {}
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value)))
            result, _ = _reduced_exploratory(result_path(manifest))
            metric = _test_metrics(frame, manifest["membership"], result["selected"], model)
            errors[model][fold] = metric["trajectory_mse"]; metrics[model][fold] = metric
            sources.append({"model": model, "fold": fold, "result_path": str(result_path(manifest)),
                "result_sha256": result["result_sha256"], "selected_coordinates": result["selected"]["coordinates"],
                "shared_parameters": result["selected"]["shared_parameters"]})
    selection = select_model_paired_one_standard_error(errors, model_order=MODEL_ORDER,
        fold_order=PRIMARY_FOLDS, replicates=PAIRED_BOOTSTRAP_REPLICATES)
    advance = selection.selected_model != "M2"
    payload = _seal({"schema_version": REPORT_SCHEMA, "status": "complete", "exploratory": True,
        "screen_index_sha256": screen["index_sha256"], "audit_sha256": audit["audit_sha256"],
        "official_candidate_report": screen["official_candidate_report"], "model_order": list(MODEL_ORDER),
        "primary_metrics": metrics, "selection": selection.to_dict(), "advance_nested_confirmation": advance,
        "source_results": sources,
        "claim_boundary": "Post-hoc same-grid model development; registered M2 remains authoritative pending prospective evidence."},
        "report_sha256")
    root = Path(str(screen["output_root"])); json_path = root / "reports" / "screen_report.json"
    write_json_once(json_path, payload)
    lines = ["# Exploratory M2 K-interaction screen", "", f"Selected by paired one-SE rule: **{selection.selected_model}**",
        "", f"Observed best: **{selection.observed_best_model}**", "",
        "This is post-hoc same-grid model development. It does not replace the registered M2 result.", "",
        "| Model | Equal-fold RMSE |", "|---|---:|"]
    for model in MODEL_ORDER:
        lines.append(f"| {model} | {selection.observed_aggregate_rmse[model]:.8f} |")
    markdown_path = root / "reports" / "screen_report.md"; write_text_once(markdown_path, "\n".join(lines) + "\n")
    return {"path": str(json_path), "markdown_path": str(markdown_path), **payload}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-wave"); plan.add_argument("--wave", choices=("smoke", "screen"), required=True)
    plan.add_argument("--profile", type=Path, required=True); plan.add_argument("--input-receipt", type=Path, required=True)
    plan.add_argument("--holdouts", type=Path, required=True); plan.add_argument("--candidate-report", type=Path, required=True)
    plan.add_argument("--dry-run", action="store_true")
    emit = commands.add_parser("emit-slurm"); emit.add_argument("--index", type=Path, required=True)
    submit = commands.add_parser("record-submission"); submit.add_argument("--index", type=Path, required=True)
    submit.add_argument("--job", action="append", default=[]); submit.add_argument("--attempt", type=int, default=1)
    status = commands.add_parser("status"); status.add_argument("--index", type=Path, required=True)
    reduce_p = commands.add_parser("reduce-index"); reduce_p.add_argument("--index", type=Path, required=True)
    smoke = commands.add_parser("smoke-report"); smoke.add_argument("--index", type=Path, required=True)
    audits = commands.add_parser("plan-audits"); audits.add_argument("--screen-index", type=Path, required=True)
    audit = commands.add_parser("audit-screen"); audit.add_argument("--screen-index", type=Path, required=True); audit.add_argument("--audit-index", type=Path, required=True)
    report = commands.add_parser("report"); report.add_argument("--screen-index", type=Path, required=True); report.add_argument("--audit", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan-wave":
        result = plan_wave(wave=args.wave, profile_path=args.profile, input_receipt_path=args.input_receipt,
            holdouts_path=args.holdouts, candidate_report_path=args.candidate_report, dry_run=args.dry_run)
    elif args.command == "emit-slurm": result = emit_slurm(args.index)
    elif args.command == "record-submission": result = record_submission(args.index, args.job, attempt=args.attempt)
    elif args.command == "status": result = status_index(args.index)
    elif args.command == "reduce-index": result = reduce_index(args.index)
    elif args.command == "smoke-report": result = smoke_report(args.index)
    elif args.command == "plan-audits": result = plan_audits(args.screen_index)
    elif args.command == "audit-screen": result = audit_screen(args.screen_index, args.audit_index)
    else: result = build_report(args.screen_index, args.audit)
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
