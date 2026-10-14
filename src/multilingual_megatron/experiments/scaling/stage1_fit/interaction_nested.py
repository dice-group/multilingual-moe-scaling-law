"""Leakage-controlled nested ladder check for the exploratory M2 K family.

The candidate family was designed after inspecting the original primary
holdouts.  This module therefore keeps the registered M2 result authoritative
and estimates selection optimism with four outer ladder holdouts.  For every
outer ladder, model selection is repeated using only the other three ladders.
All artifacts are immutable, signed, and permanently review-only.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .campaign import _record_prediction, _records, scientific_result
from .data import canonical_json_sha256, load_fit_rows_snapshot, sha256_file
from .execution import (
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
from .fit import (
    OBJECTIVE_RELATIVE_TOLERANCE,
    PREDICTION_ABSOLUTE_TOLERANCE,
    SharedParameters,
    StartPoint,
    effective_exponents,
    equivalence_check,
    initial_bounds,
)
from .interaction import (
    MODEL_ORDER,
    PRIMARY_FOLDS,
    REPORT_SCHEMA,
    SUBMISSION_SCHEMA,
    _gamma_seeds,
    _official_m2_paths,
    _official_report,
    _profile,
    _reduced_exploratory,
    _retained_m2_anchors,
    _seal,
    _sealed,
    _test_metrics,
    _wave_index,
)
from .selection import (
    PAIRED_BOOTSTRAP_REPLICATES,
    select_model_paired_one_standard_error,
)

LADDERS = ("A", "B", "C", "D")
INNER_AXES = ("largest_remaining", "E32", "K200_common_languages")
AXIS_TO_PRIMARY = {
    "largest_remaining": "largest_N_act",
    "E32": "E32",
    "K200_common_languages": "K200_common_languages",
}
NESTED_MODELS = MODEL_ORDER
ANCHOR_COUNT = 8
BOOTSTRAP_SEED = 20260828

MEMBERSHIP_SCHEMA = "stage1_m2_k_interaction_nested_memberships/v1"
SCALE_INDEX_SCHEMA = "stage1_m2_k_interaction_nested_scale_index/v1"
INNER_INDEX_SCHEMA = "stage1_m2_k_interaction_nested_inner_index/v1"
INNER_REPORT_SCHEMA = "stage1_m2_k_interaction_nested_inner_report/v1"
OUTER_INDEX_SCHEMA = "stage1_m2_k_interaction_nested_outer_index/v1"
OUTER_AUDIT_INDEX_SCHEMA = "stage1_m2_k_interaction_nested_outer_audit_index/v1"
OUTER_AUDIT_SCHEMA = "stage1_m2_k_interaction_nested_outer_audit/v1"
FINAL_REPORT_SCHEMA = "stage1_m2_k_interaction_nested_report/v1"
NESTED_INDEX_SCHEMAS = {
    SCALE_INDEX_SCHEMA,
    INNER_INDEX_SCHEMA,
    OUTER_INDEX_SCHEMA,
    OUTER_AUDIT_INDEX_SCHEMA,
}


class NestedInteractionError(RuntimeError):
    """Raised when a frozen nested-CV binding or gate changes."""


def _screen_report(path: Path, *, screen_index_sha256: str | None = None) -> Mapping[str, Any]:
    report = _sealed(read_json(path), "report_sha256")
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "complete":
        raise NestedInteractionError("interaction screen report is incomplete")
    if screen_index_sha256 is not None and report.get("screen_index_sha256") != screen_index_sha256:
        raise NestedInteractionError("interaction screen report does not bind the screen index")
    selection = report.get("selection", {})
    if not report.get("advance_nested_confirmation") or selection.get("selected_model") == "M2":
        raise NestedInteractionError("nested confirmation requires a non-M2 screen selection")
    if selection.get("selected_model") not in NESTED_MODELS:
        raise NestedInteractionError("screen selected an unknown interaction form")
    return report


def _nested_index(path: Path, *, schema: str | None = None) -> Mapping[str, Any]:
    index = _sealed(read_json(path), "index_sha256")
    if index.get("schema_version") not in NESTED_INDEX_SCHEMAS:
        raise NestedInteractionError("unknown nested interaction index schema")
    if schema is not None and index.get("schema_version") != schema:
        raise NestedInteractionError("nested interaction index has the wrong stage")
    if index.get("exploratory") is not True:
        raise NestedInteractionError("nested interaction index lost review-only status")
    return index


def _trajectory_parts(trajectory_id: str) -> tuple[str, str, str]:
    parts = tuple(str(trajectory_id).split(":"))
    if len(parts) != 3 or parts[0] not in LADDERS or parts[1] not in {"E1", "E4", "E16", "E32"} \
            or parts[2] not in {"K1", "K12", "K50", "K200"}:
        raise NestedInteractionError(f"unexpected Stage 1 trajectory identifier: {trajectory_id}")
    return parts  # type: ignore[return-value]


def _membership_payload(holdouts: Mapping[str, Any], *, input_sha256: str) -> dict[str, Any]:
    folds = {str(item["fold_id"]): item for item in holdouts.get("primary_folds", [])}
    if set(folds) != set(PRIMARY_FOLDS):
        raise NestedInteractionError("source holdouts do not contain the three primary folds")
    all_ids = sorted(set(folds["largest_N_act"]["train_trajectory_ids"])
        | set(folds["largest_N_act"]["test_trajectory_ids"]))
    expected = {
        f"{ladder}:E{experts}:K{languages}"
        for ladder in LADDERS for experts in (1, 4, 16, 32) for languages in (1, 12, 50, 200)
    }
    if set(all_ids) != expected or len(all_ids) != 64:
        raise NestedInteractionError("nested protocol requires the complete 4x4x4 trajectory grid")
    all_languages = sorted(set(str(item) for item in folds["largest_N_act"]["score_languages"]))
    if all_languages != sorted(set(str(item) for item in folds["E32"]["score_languages"])):
        raise NestedInteractionError("largest-N and E32 source folds disagree on scored languages")
    common_languages = sorted(set(str(item) for item in folds["K200_common_languages"]["score_languages"]))
    if not common_languages or not set(common_languages) <= set(all_languages):
        raise NestedInteractionError("K200 common-language population is malformed")

    items: dict[str, Any] = {}

    def add(membership_id: str, *, role: str, outer: str, axis: str | None,
            train: Sequence[str], test: Sequence[str], score: Sequence[str]) -> None:
        train_ids = sorted(set(str(item) for item in train)); test_ids = sorted(set(str(item) for item in test))
        if not train_ids or not test_ids or set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(all_ids):
            raise NestedInteractionError(f"{membership_id} is not a complete disjoint partition")
        item = {
            "fold_id": membership_id,
            "role": role,
            "outer_ladder": outer,
            "inner_axis": axis,
            "train_trajectory_ids": train_ids,
            "test_trajectory_ids": test_ids,
            "score_languages": sorted(set(str(value) for value in score)),
            "train_readout_ids": [],
            "test_readout_ids": [],
        }
        item["content_sha256"] = canonical_json_sha256(item)
        items[membership_id] = item

    for outer in LADDERS:
        outer_test = [item for item in all_ids if _trajectory_parts(item)[0] == outer]
        outer_train = [item for item in all_ids if _trajectory_parts(item)[0] != outer]
        add(f"outer_{outer}", role="outer_ladder_holdout", outer=outer, axis=None,
            train=outer_train, test=outer_test, score=all_languages)
        remaining = tuple(item for item in LADDERS if item != outer)
        largest = max(remaining, key=LADDERS.index)
        inner_tests = {
            "largest_remaining": [item for item in outer_train if _trajectory_parts(item)[0] == largest],
            "E32": [item for item in outer_train if _trajectory_parts(item)[1] == "E32"],
            "K200_common_languages": [item for item in outer_train if _trajectory_parts(item)[2] == "K200"],
        }
        for axis in INNER_AXES:
            test = inner_tests[axis]
            train = [item for item in outer_train if item not in set(test)]
            # The outer ladder is absent from both inner partitions by design.
            full_train = sorted(set(train) | set(test))
            if full_train != sorted(outer_train):
                raise NestedInteractionError(f"inner split {outer}/{axis} escapes its outer training set")
            membership_id = f"outer_{outer}__{axis}"
            score = common_languages if axis == "K200_common_languages" else all_languages
            # ``add`` expects a partition of the global population, so bind the
            # outer ladder to neither inner side explicitly below.
            train_ids = sorted(set(train)); test_ids = sorted(set(test))
            if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(outer_train):
                raise NestedInteractionError(f"{membership_id} is not a partition of outer training data")
            item = {
                "fold_id": membership_id,
                "role": "inner_model_selection_holdout",
                "outer_ladder": outer,
                "inner_axis": axis,
                "largest_remaining_ladder": largest if axis == "largest_remaining" else None,
                "train_trajectory_ids": train_ids,
                "test_trajectory_ids": test_ids,
                "score_languages": sorted(set(str(value) for value in score)),
                "train_readout_ids": [],
                "test_readout_ids": [],
            }
            item["content_sha256"] = canonical_json_sha256(item)
            items[membership_id] = item

    if len(items) != 16:
        raise NestedInteractionError("nested membership count changed")
    for outer in LADDERS:
        if len(items[f"outer_{outer}"]["train_trajectory_ids"]) != 48 \
                or len(items[f"outer_{outer}"]["test_trajectory_ids"]) != 16:
            raise NestedInteractionError("outer membership cardinality changed")
    payload = {
        "schema_version": MEMBERSHIP_SCHEMA,
        "input_sha256": input_sha256,
        "protocol": {
            "outer_ladders": list(LADDERS),
            "inner_axes": list(INNER_AXES),
            "selection_models": list(NESTED_MODELS),
            "unit": "whole training trajectory",
            "outer_test_access": "after inner selection and outer-train refit only",
        },
        "source_holdouts": {
            "manifest_sha256": holdouts["manifest_sha256"],
        },
        **items,
    }
    return _seal(payload, "manifest_sha256")


def _write_index(output_root: Path, name: str, payload: Mapping[str, Any]) -> Path:
    index = _seal(dict(payload), "index_sha256")
    path = output_root / "interaction" / f"{name}_index.json"
    write_json_once(path, index)
    return path


def emit_slurm(index_path: Path) -> dict[str, Any]:
    index = _nested_index(index_path)
    scripts: dict[str, str] = {}
    for model, folds in index["manifests"].items():
        for fold, manifest_value in folds.items():
            manifest_path = Path(str(manifest_value)); manifest = load_manifest(manifest_path)
            script, _ = render_slurm(manifest_path, manifest)
            key = f"{model}/{fold}"
            safe = key.replace("/", "-").replace(":", "-")
            output = Path(str(index["output_root"])) / "slurm" / \
                f"{index['wave']}-{safe}-{manifest['manifest_sha256'][:12]}.sbatch"
            write_text_once(output, script); scripts[key] = str(output)
    return {"index_sha256": index["index_sha256"], "scripts": scripts,
        "count": len(scripts), "submitted": False}


def record_submission(index_path: Path, jobs: Sequence[str], *, attempt: int) -> dict[str, Any]:
    index = _nested_index(index_path)
    expected = {f"{model}/{fold}" for model, folds in index["manifests"].items() for fold in folds}
    parsed: dict[str, str] = {}
    for item in jobs:
        key, separator, job_id = item.partition("=")
        if separator != "=" or not job_id.isdigit() or key in parsed:
            raise NestedInteractionError(f"malformed submission binding: {item}")
        parsed[key] = job_id
    if set(parsed) != expected or attempt < 1:
        raise NestedInteractionError("submission bindings do not exactly cover the nested index")
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


def status_index(index_path: Path) -> dict[str, Any]:
    index = _nested_index(index_path); rows = []
    for model, folds in index["manifests"].items():
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value)))
            stem = manifest["manifest_sha256"][:20]
            chunk_root = Path(str(index["output_root"])) / "chunks" / stem
            present = sum((chunk_root / f"task-{task['task_index']:06d}.json").exists()
                for task in manifest["tasks"])
            rows.append({"key": f"{model}/{fold}", "tasks": len(manifest["tasks"]), "chunks": present,
                "result": (Path(str(index["output_root"])) / "results" / f"{stem}.json").exists()})
    return {"wave": index["wave"], "complete": all(row["tasks"] == row["chunks"] for row in rows),
        "rows": rows}


def reduce_index(index_path: Path) -> dict[str, Any]:
    index = _nested_index(index_path); paths: dict[str, dict[str, str]] = {}
    for model, folds in index["manifests"].items():
        paths[model] = {}
        for fold, manifest_value in folds.items():
            manifest = load_manifest(Path(str(manifest_value))); result = reduce(manifest)
            if result.get("status") != "reduced" or not result.get("selected"):
                raise NestedInteractionError(f"{model}/{fold} produced no selected fit")
            paths[model][fold] = str(result_path(manifest))
    return {"wave": index["wave"], "results": paths,
        "count": sum(len(value) for value in paths.values())}


def plan_scales(*, screen_index_path: Path, screen_report_path: Path, dry_run: bool = False) -> dict[str, Any]:
    """Freeze 16 nested memberships and plan exact 81-start M0-OLS scales."""
    screen = _wave_index(screen_index_path)
    if screen.get("wave") != "screen":
        raise NestedInteractionError("nested scales require the completed representative screen")
    screen_report = _screen_report(screen_report_path, screen_index_sha256=str(screen["index_sha256"]))
    profile_path = Path(str(screen["profile"]["path"])); profile = _profile(profile_path)
    if int(profile["chunk_size"]) != 12:
        raise NestedInteractionError("nested protocol freezes chunk size 12")
    output_root = Path(str(screen["output_root"])); receipt_path = Path(str(screen["input_receipt"]["path"]))
    holdouts_path = Path(str(screen["holdouts"]["path"])); receipt = read_json(receipt_path)
    holdouts = _sealed(read_json(holdouts_path), "manifest_sha256")
    if receipt.get("fit_rows_content_sha256") != holdouts.get("input_sha256"):
        raise NestedInteractionError("nested inputs do not bind the frozen fit rows")
    memberships = _membership_payload(holdouts, input_sha256=str(receipt["fit_rows_content_sha256"]))
    membership_path = output_root / "input" / "nested_ladder_memberships.json"
    if not dry_run:
        write_json_once(membership_path, memberships)

    planned: dict[str, dict[str, str]] = {"M0": {}}
    total_starts = 0; total_tasks = 0
    membership_ids = [f"outer_{outer}" for outer in LADDERS] + [
        f"outer_{outer}__{axis}" for outer in LADDERS for axis in INNER_AXES
    ]
    for membership_id in membership_ids:
        if dry_run:
            item = memberships[membership_id]
            membership = {
                "kind": f"compute:{membership_id}",
                "train_trajectory_ids": item["train_trajectory_ids"],
                "test_trajectory_ids": item["test_trajectory_ids"],
                "score_languages": item["score_languages"],
                "train_readout_ids": [], "test_readout_ids": [],
                "membership_sha256": "dry-run",
                "path": str(membership_path.resolve()), "file_sha256": "dry-run", "embedded_sha256": "dry-run",
            }
        else:
            membership = bind_membership(membership_path, kind="compute", membership_id=membership_id)
        manifest = build_manifest(
            output_root=output_root,
            fit_rows_path=Path(str(receipt["fit_rows_path"])),
            fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]),
            phase=f"interaction-nested-scale-{membership_id}",
            model="M0", fold_id=membership_id, objective="ols",
            chunk_size=int(profile["chunk_size"]), membership=membership,
            review_only=True, action="nested_scale",
            selection_report=screen_report_path,
            blas_threads=int(profile["blas_threads"]), partition=profile.get("partition"),
            account=profile.get("account"), reservation=profile.get("reservation"),
            memory=str(profile["memory"]), time_limit=str(profile["time_limit"]),
            concurrency=int(profile["array_concurrency"]),
        )
        total_starts += int(manifest["fit"]["start_count"]); total_tasks += len(manifest["tasks"])
        if not dry_run:
            planned["M0"][membership_id] = str(write_manifest(manifest))
    expected = {"starts": 16 * 81, "tasks": 16 * math.ceil(81 / 12)}
    if {"starts": total_starts, "tasks": total_tasks} != expected:
        raise NestedInteractionError("nested M0-OLS workload changed")
    result: dict[str, Any] = {**expected, "wave": "nested-scale", "written": not dry_run,
        "memberships": str(membership_path), "manifests": planned}
    if not dry_run:
        path = _write_index(output_root, "nested_scale", {
            "schema_version": SCALE_INDEX_SCHEMA, "wave": "nested-scale", "exploratory": True,
            "output_root": str(output_root),
            "screen_index": {"path": str(screen_index_path.resolve()), "index_sha256": screen["index_sha256"]},
            "screen_report": {"path": str(screen_report_path.resolve()), "file_sha256": sha256_file(screen_report_path),
                "report_sha256": screen_report["report_sha256"]},
            "profile": {"path": str(profile_path.resolve()), "file_sha256": sha256_file(profile_path)},
            "input_receipt": {"path": str(receipt_path.resolve()), "file_sha256": sha256_file(receipt_path)},
            "source_holdouts": {"path": str(holdouts_path.resolve()), "file_sha256": sha256_file(holdouts_path),
                "manifest_sha256": holdouts["manifest_sha256"]},
            "memberships": {"path": str(membership_path.resolve()), "file_sha256": sha256_file(membership_path),
                "manifest_sha256": memberships["manifest_sha256"]},
            "manifests": planned, "expected": expected,
        })
        result["index_path"] = str(path)
    return result


def _scale_objective(scale: Mapping[str, Any], *, membership: Mapping[str, Any], fold_id: str) -> tuple[str, float | None]:
    identity = scale.get("fit_identity", {})
    if identity.get("model") != "M0" or identity.get("objective") != "ols" or identity.get("fold_id") != fold_id:
        raise NestedInteractionError(f"{fold_id} scale has the wrong fit identity")
    if scale.get("membership", {}).get("membership_sha256") != membership.get("membership_sha256"):
        raise NestedInteractionError(f"{fold_id} scale uses a different membership")
    adaptive = scale.get("adaptive_huber_scale")
    if not isinstance(adaptive, Mapping) or not isinstance(adaptive.get("use_ols"), bool):
        raise NestedInteractionError(f"{fold_id} scale lacks the adaptive Huber decision")
    if adaptive["use_ols"]:
        return "ols", None
    delta = adaptive.get("delta")
    if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)) or float(delta) <= 0.0:
        raise NestedInteractionError(f"{fold_id} scale has an invalid Huber delta")
    return "huber", float(delta)


def _anchored_starts(model: str, anchor_result: Mapping[str, Any], *, namespace: str) -> tuple[StartPoint, ...]:
    anchors = _retained_m2_anchors(anchor_result, count=ANCHOR_COUNT)
    starts: list[StartPoint] = []
    for anchor_index, anchor in enumerate(anchors):
        n_values = _gamma_seeds(anchor, effect="n") if model in {"M2-KN", "M2-KND"} else (0.0,)
        d_values = _gamma_seeds(anchor, effect="d") if model in {"M2-KD", "M2-KND"} else (0.0,)
        for n_index, gamma_n in enumerate(n_values):
            for d_index, gamma_d in enumerate(d_values):
                values = anchor.tolist(); suffix: list[str] = []
                if model in {"M2-KN", "M2-KND"}:
                    values.append(gamma_n); suffix.append(f"n{n_index}")
                if model in {"M2-KD", "M2-KND"}:
                    values.append(gamma_d); suffix.append(f"d{d_index}")
                if model == "M2":
                    suffix.append("m2")
                starts.append(StartPoint(
                    start_id=f"{namespace}:a{anchor_index}:{'-'.join(suffix)}",
                    start_index=None, values=np.asarray(values, dtype=np.float64), kind="nested_retained_basin",
                ))
    expected = {"M2": 8, "M2-KN": 24, "M2-KD": 24, "M2-KND": 72}[model]
    if len(starts) != expected or len({item.start_id for item in starts}) != expected:
        raise NestedInteractionError("nested retained-basin start protocol changed")
    return tuple(starts)


def _load_scale(scale_index: Mapping[str, Any], membership_id: str) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    manifest_path = Path(str(scale_index["manifests"]["M0"][membership_id]))
    manifest = load_manifest(manifest_path)
    result, _ = _reduced_exploratory(result_path(manifest))
    return result_path(manifest), result, manifest


def plan_inner(*, scale_index_path: Path, dry_run: bool = False) -> dict[str, Any]:
    """Plan four-model inner selection fits for all 4x3 nested folds."""
    scales = _nested_index(scale_index_path, schema=SCALE_INDEX_SCHEMA)
    screen = _wave_index(Path(str(scales["screen_index"]["path"])))
    screen_report_path = Path(str(scales["screen_report"]["path"]))
    screen_report = _screen_report(screen_report_path, screen_index_sha256=str(screen["index_sha256"]))
    official_path = Path(str(screen_report["official_candidate_report"]["path"]))
    official = _official_report(official_path); anchor_paths = _official_m2_paths(official)
    receipt_path = Path(str(scales["input_receipt"]["path"])); receipt = read_json(receipt_path)
    membership_path = Path(str(scales["memberships"]["path"])); profile = _profile(Path(str(scales["profile"]["path"])))
    output_root = Path(str(scales["output_root"])); planned: dict[str, dict[str, str]] = {model: {} for model in NESTED_MODELS}
    total_starts = 0; total_tasks = 0
    for outer in LADDERS:
        for axis in INNER_AXES:
            membership_id = f"outer_{outer}__{axis}"
            membership = bind_membership(membership_path, kind="compute", membership_id=membership_id)
            scale_path, scale, _ = _load_scale(scales, membership_id)
            objective, delta = _scale_objective(scale, membership=membership, fold_id=membership_id)
            anchor_path = anchor_paths[AXIS_TO_PRIMARY[axis]]; anchor = scientific_result(anchor_path)
            for model in NESTED_MODELS:
                starts = _anchored_starts(model, anchor, namespace=f"nested-inner:{outer}:{axis}:{model}")
                manifest = build_manifest(
                    output_root=output_root, fit_rows_path=Path(str(receipt["fit_rows_path"])),
                    fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]),
                    phase=f"interaction-nested-inner-{outer}-{axis}-{model.lower().replace('-', '')}",
                    model=model, fold_id=membership_id, objective=objective, delta=delta,
                    chunk_size=int(profile["chunk_size"]), bounds=initial_bounds(model),
                    parent_results={"anchor_m2": anchor_path, "scale_m0_ols": scale_path},
                    selection_report=screen_report_path, membership=membership, explicit_starts=starts,
                    review_only=True, action="nested_inner", blas_threads=int(profile["blas_threads"]),
                    partition=profile.get("partition"), account=profile.get("account"),
                    reservation=profile.get("reservation"), memory=str(profile["memory"]),
                    time_limit=str(profile["time_limit"]), concurrency=int(profile["array_concurrency"]),
                )
                total_starts += int(manifest["fit"]["start_count"]); total_tasks += len(manifest["tasks"])
                if not dry_run:
                    planned[model][membership_id] = str(write_manifest(manifest))
    expected = {"starts": 12 * (8 + 24 + 24 + 72), "tasks": 12 * (1 + 2 + 2 + 6)}
    if {"starts": total_starts, "tasks": total_tasks} != expected:
        raise NestedInteractionError("nested inner workload changed")
    result: dict[str, Any] = {**expected, "wave": "nested-inner", "written": not dry_run, "manifests": planned}
    if not dry_run:
        path = _write_index(output_root, "nested_inner", {
            "schema_version": INNER_INDEX_SCHEMA, "wave": "nested-inner", "exploratory": True,
            "output_root": str(output_root), "scale_index": {"path": str(scale_index_path.resolve()),
                "index_sha256": scales["index_sha256"]}, "screen_index": scales["screen_index"],
            "screen_report": scales["screen_report"], "input_receipt": scales["input_receipt"],
            "memberships": scales["memberships"], "profile": scales["profile"],
            "official_candidate_report": {"path": str(official_path.resolve()), "file_sha256": sha256_file(official_path),
                "report_sha256": official["report_sha256"]},
            "anchor_results": {key: str(value.resolve()) for key, value in anchor_paths.items()},
            "model_order": list(NESTED_MODELS), "outer_ladders": list(LADDERS),
            "inner_axes": list(INNER_AXES), "manifests": planned, "expected": expected,
        })
        result["index_path"] = str(path)
    return result


def _numerical_health(result: Mapping[str, Any], model: str) -> dict[str, Any]:
    selected = result.get("selected", {}); successes = sum(bool(item.get("success")) for item in result.get("attempts", []))
    contacts = selected.get("boundary_contacts", []) if isinstance(selected, Mapping) else ["missing_selected"]
    minimum_alpha = math.nan; minimum_beta = math.nan; positive = False
    if isinstance(selected, Mapping) and isinstance(selected.get("shared_parameters"), Mapping):
        parameters = SharedParameters(**selected["shared_parameters"])
        alpha, beta, _, _ = effective_exponents(parameters,
            np.asarray([1.0, 4.0, 16.0, 32.0])[:, None],
            np.asarray([1.0, 12.0, 50.0, 200.0])[None, :], model=model)
        minimum_alpha = float(np.min(alpha)); minimum_beta = float(np.min(beta))
        positive = minimum_alpha > 0.0 and minimum_beta > 0.0
    healthy = successes >= ANCHOR_COUNT and not contacts and positive
    return {"status": "complete" if healthy else "failed", "successful_starts": successes,
        "boundary_contacts": contacts, "minimum_alpha": minimum_alpha, "minimum_beta": minimum_beta}


def build_inner_report(inner_index_path: Path) -> dict[str, Any]:
    """Select a model inside each outer training population without outer scores."""
    index = _nested_index(inner_index_path, schema=INNER_INDEX_SCHEMA)
    receipt = read_json(Path(str(index["input_receipt"]["path"])))
    frame = load_fit_rows_snapshot(receipt)
    outer_reports: dict[str, Any] = {}; sources = []; healthy = True
    for outer in LADDERS:
        errors: dict[str, dict[str, dict[str, float]]] = {model: {} for model in NESTED_MODELS}
        metrics: dict[str, dict[str, Any]] = {model: {} for model in NESTED_MODELS}
        numerical: dict[str, dict[str, Any]] = {model: {} for model in NESTED_MODELS}
        for model in NESTED_MODELS:
            for axis in INNER_AXES:
                membership_id = f"outer_{outer}__{axis}"
                manifest = load_manifest(Path(str(index["manifests"][model][membership_id])))
                result, _ = _reduced_exploratory(result_path(manifest))
                metric = _test_metrics(frame, manifest["membership"], result["selected"], model)
                check = _numerical_health(result, model); healthy = healthy and check["status"] == "complete"
                errors[model][axis] = metric["trajectory_mse"]
                metrics[model][axis] = metric; numerical[model][axis] = check
                sources.append({"outer_ladder": outer, "inner_axis": axis, "model": model,
                    "result_path": str(result_path(manifest)), "result_sha256": result["result_sha256"]})
        selection = select_model_paired_one_standard_error(errors, model_order=NESTED_MODELS,
            fold_order=INNER_AXES, replicates=PAIRED_BOOTSTRAP_REPLICATES)
        outer_reports[outer] = {"selection": selection.to_dict(), "metrics": metrics, "numerical": numerical,
            "outer_scores_accessed": False}
    status = "complete" if healthy else "failed"
    screen_report = _screen_report(Path(str(index["screen_report"]["path"])),
        screen_index_sha256=str(index["screen_index"]["index_sha256"]))
    payload = _seal({
        "schema_version": INNER_REPORT_SCHEMA, "status": status, "exploratory": True,
        "inner_index_sha256": index["index_sha256"],
        "screen_selected_model": screen_report["selection"]["selected_model"],
        "outer_ladders": outer_reports, "source_results": sources,
        "selection_protocol": "paired 10000-bootstrap whole-trajectory one-standard-error rule inside each outer training set",
        "claim_boundary": "Nested resampling sensitivity analysis on the same 64 trajectories; no outer score was used for inner selection.",
    }, "report_sha256")
    path = Path(str(index["output_root"])) / "reports" / "nested_inner_selection.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def plan_outer(*, scale_index_path: Path, inner_report_path: Path, dry_run: bool = False) -> dict[str, Any]:
    """Refit M2 and each inner-selected form on complete outer training data."""
    scales = _nested_index(scale_index_path, schema=SCALE_INDEX_SCHEMA)
    inner = _sealed(read_json(inner_report_path), "report_sha256")
    if inner.get("schema_version") != INNER_REPORT_SCHEMA or inner.get("status") != "complete":
        raise NestedInteractionError("outer fits require a healthy inner selection report")
    # Bind through the report's embedded inner-index hash and the on-disk index
    # discovered beside the scale index; this avoids accepting an arbitrary
    # same-schema report.
    candidate_inner_index = Path(str(scales["output_root"])) / "interaction" / "nested_inner_index.json"
    nested_inner = _nested_index(candidate_inner_index, schema=INNER_INDEX_SCHEMA)
    if inner.get("inner_index_sha256") != nested_inner["index_sha256"] \
            or nested_inner.get("scale_index", {}).get("index_sha256") != scales["index_sha256"]:
        raise NestedInteractionError("inner selection report does not bind the nested scale campaign")
    screen = _wave_index(Path(str(scales["screen_index"]["path"])))
    screen_report_path = Path(str(scales["screen_report"]["path"])); screen_report = _screen_report(
        screen_report_path, screen_index_sha256=str(screen["index_sha256"]))
    official = _official_report(Path(str(screen_report["official_candidate_report"]["path"])))
    anchor_path = _official_m2_paths(official)["largest_N_act"]; anchor = scientific_result(anchor_path)
    receipt = read_json(Path(str(scales["input_receipt"]["path"])))
    membership_path = Path(str(scales["memberships"]["path"])); profile = _profile(Path(str(scales["profile"]["path"])))
    output_root = Path(str(scales["output_root"])); planned: dict[str, dict[str, str]] = {model: {} for model in NESTED_MODELS}
    total_starts = 0; total_tasks = 0; selected_models: dict[str, str] = {}
    for outer in LADDERS:
        membership_id = f"outer_{outer}"; membership = bind_membership(membership_path, kind="compute", membership_id=membership_id)
        scale_path, scale, _ = _load_scale(scales, membership_id)
        objective, delta = _scale_objective(scale, membership=membership, fold_id=membership_id)
        selected_model = str(inner["outer_ladders"][outer]["selection"]["selected_model"])
        if selected_model not in NESTED_MODELS:
            raise NestedInteractionError(f"inner selection for outer {outer} is unknown")
        selected_models[outer] = selected_model
        models = ("M2",) if selected_model == "M2" else ("M2", selected_model)
        for model in models:
            starts = _anchored_starts(model, anchor, namespace=f"nested-outer:{outer}:{model}")
            manifest = build_manifest(
                output_root=output_root, fit_rows_path=Path(str(receipt["fit_rows_path"])),
                fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]),
                phase=f"interaction-nested-outer-{outer}-{model.lower().replace('-', '')}",
                model=model, fold_id=membership_id, objective=objective, delta=delta,
                chunk_size=int(profile["chunk_size"]), bounds=initial_bounds(model),
                parent_results={"anchor_m2": anchor_path, "scale_m0_ols": scale_path},
                selection_report=inner_report_path, membership=membership, explicit_starts=starts,
                review_only=True, action="nested_outer", blas_threads=int(profile["blas_threads"]),
                partition=profile.get("partition"), account=profile.get("account"),
                reservation=profile.get("reservation"), memory=str(profile["memory"]),
                time_limit=str(profile["time_limit"]), concurrency=int(profile["array_concurrency"]),
            )
            total_starts += int(manifest["fit"]["start_count"]); total_tasks += len(manifest["tasks"])
            if not dry_run:
                planned[model][outer] = str(write_manifest(manifest))
    expected = {"starts": total_starts, "tasks": total_tasks}
    result: dict[str, Any] = {**expected, "wave": "nested-outer", "written": not dry_run,
        "selected_models": selected_models, "manifests": planned}
    if not dry_run:
        path = _write_index(output_root, "nested_outer", {
            "schema_version": OUTER_INDEX_SCHEMA, "wave": "nested-outer", "exploratory": True,
            "output_root": str(output_root), "scale_index": {"path": str(scale_index_path.resolve()),
                "index_sha256": scales["index_sha256"]}, "screen_index": scales["screen_index"],
            "screen_report": scales["screen_report"], "input_receipt": scales["input_receipt"],
            "memberships": scales["memberships"], "profile": scales["profile"],
            "inner_report": {"path": str(inner_report_path.resolve()), "file_sha256": sha256_file(inner_report_path),
                "report_sha256": inner["report_sha256"]}, "selected_models": selected_models,
            "anchor_result": str(anchor_path.resolve()), "manifests": planned, "expected": expected,
        })
        result["index_path"] = str(path)
    return result


def plan_outer_audits(outer_index_path: Path) -> dict[str, Any]:
    outer = _nested_index(outer_index_path, schema=OUTER_INDEX_SCHEMA)
    output_root = Path(str(outer["output_root"])); planned: dict[str, dict[str, str]] = {}
    total_starts = 0; total_tasks = 0
    for model, folds in outer["manifests"].items():
        if not folds:
            continue
        planned[model] = {}
        for ladder, manifest_value in folds.items():
            base = load_manifest(Path(str(manifest_value))); result, _ = _reduced_exploratory(result_path(base))
            usable = [item for item in result.get("attempts", []) if item.get("success") and item.get("coordinates")]
            usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
            if len(usable) < 3:
                raise NestedInteractionError(f"outer {ladder}/{model} lacks three successful starts")
            parents = {str(item["kind"]): Path(str(item["path"])) for item in base.get("parents", [])}
            parents["grid"] = result_path(base)
            selection = Path(str(base["selection_report"]["path"]))
            for action, starts in (
                ("basin", tuple(StartPoint(f"nested-outer-basin:{index}", None,
                    np.asarray(item["coordinates"], dtype=np.float64), "basin") for index, item in enumerate(usable[:3]))),
                ("direct", (StartPoint("nested-outer-direct:best", None,
                    np.asarray(result["selected"]["coordinates"], dtype=np.float64), "direct"),)),
            ):
                manifest = build_manifest(
                    output_root=output_root, fit_rows_path=Path(str(base["input"]["fit_rows_path"])),
                    fit_rows_content_sha256=str(base["input"]["fit_rows_content_sha256"]),
                    phase=f"{base['phase']}-{action}", model=model, fold_id=str(base["fit"]["fold_id"]),
                    objective=str(base["fit"]["objective"]), delta=base["fit"]["delta"],
                    chunk_size=int(base["execution"]["chunk_size"]), bounds=initial_bounds(model),
                    parent_results=parents, selection_report=selection, membership=base["membership"],
                    explicit_starts=starts, review_only=True, action=action,
                    blas_threads=int(base["execution"]["blas_threads"]), partition=base["execution"].get("partition"),
                    account=base["execution"].get("account"), reservation=base["execution"].get("reservation"),
                    memory=str(base["execution"]["memory"]), time_limit=str(base["execution"]["time_limit"]),
                    concurrency=int(base["execution"]["concurrency"]),
                )
                planned[model][f"{ladder}:{action}"] = str(write_manifest(manifest))
                total_starts += int(manifest["fit"]["start_count"]); total_tasks += len(manifest["tasks"])
    expected_base = sum(len(folds) for folds in outer["manifests"].values())
    if total_starts != expected_base * 4 or total_tasks != expected_base * 2:
        raise NestedInteractionError("nested outer audit workload changed")
    expected = {"starts": total_starts, "tasks": total_tasks}
    path = _write_index(output_root, "nested_outer_audit", {
        "schema_version": OUTER_AUDIT_INDEX_SCHEMA, "wave": "nested-outer-audit", "exploratory": True,
        "output_root": str(output_root), "outer_index": {"path": str(outer_index_path.resolve()),
            "index_sha256": outer["index_sha256"]}, "manifests": planned, "expected": expected,
    })
    return {"index_path": str(path), **expected, "manifests": planned}


def audit_outer(outer_index_path: Path, audit_index_path: Path) -> dict[str, Any]:
    outer = _nested_index(outer_index_path, schema=OUTER_INDEX_SCHEMA)
    audits = _nested_index(audit_index_path, schema=OUTER_AUDIT_INDEX_SCHEMA)
    if audits.get("outer_index", {}).get("index_sha256") != outer["index_sha256"]:
        raise NestedInteractionError("outer numerical audit does not bind the outer fits")
    items = []
    for model, folds in outer["manifests"].items():
        for ladder, manifest_value in folds.items():
            base = load_manifest(Path(str(manifest_value))); result, _ = _reduced_exploratory(result_path(base))
            basin = load_manifest(Path(str(audits["manifests"][model][f"{ladder}:basin"])))
            direct = load_manifest(Path(str(audits["manifests"][model][f"{ladder}:direct"])))
            basin_result, _ = _reduced_exploratory(result_path(basin))
            direct_result, _ = _reduced_exploratory(result_path(direct))
            retained = result["selected"]; retained_prediction = _record_prediction(retained, manifest=base)
            equivalent = 0
            for record in _records(basin):
                if not record.get("success") or not isinstance(record.get("objective"), (int, float)):
                    continue
                check = equivalence_check(float(retained["objective"]), float(record["objective"]),
                    retained_prediction, _record_prediction(record, manifest=basin),
                    objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
                    prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)
                equivalent += int(check.equivalent)
            direct_check = direct_result.get("selected", {}).get("direct_equivalence", {})
            health = _numerical_health(result, model)
            complete = equivalent >= 3 and bool(direct_check.get("resolved")) and health["status"] == "complete"
            items.append({"outer_ladder": ladder, "model": model,
                "status": "complete" if complete else "failed", "equivalent_basin_starts": equivalent,
                "direct_equivalence": direct_check, **{key: health[key] for key in (
                    "successful_starts", "boundary_contacts", "minimum_alpha", "minimum_beta")},
                "base_result_sha256": result["result_sha256"],
                "basin_result_sha256": basin_result["result_sha256"],
                "direct_result_sha256": direct_result["result_sha256"]})
    status = "complete" if all(item["status"] == "complete" for item in items) else "failed"
    payload = _seal({
        "schema_version": OUTER_AUDIT_SCHEMA, "status": status,
        "outer_index_sha256": outer["index_sha256"], "audit_index_sha256": audits["index_sha256"],
        "items": items,
    }, "audit_sha256")
    path = Path(str(outer["output_root"])) / "reports" / "nested_outer_numerical_audit.json"
    write_json_once(path, payload)
    return {"path": str(path), **payload}


def _paired_outer_comparison(baseline: Mapping[str, float], procedure: Mapping[str, float],
        *, replicates: int = PAIRED_BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(procedure))
    if set(keys) != set(baseline) or set(keys) != set(procedure) or not keys:
        raise NestedInteractionError("outer error maps do not cover the same trajectories")
    first = np.asarray([float(baseline[key]) for key in keys], dtype=np.float64)
    second = np.asarray([float(procedure[key]) for key in keys], dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all() or np.any(first < 0.0) or np.any(second < 0.0):
        raise NestedInteractionError("outer trajectory errors are invalid")
    baseline_rmse = float(np.sqrt(np.mean(first))); procedure_rmse = float(np.sqrt(np.mean(second)))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=replicates)
    baseline_boot = np.sqrt((counts @ first) / len(keys)); procedure_boot = np.sqrt((counts @ second) / len(keys))
    differences = procedure_boot - baseline_boot
    percentiles = np.percentile(differences, [2.5, 10.0, 50.0, 90.0, 97.5])
    return {
        "trajectories": len(keys), "bootstrap_replicates": replicates, "seed": seed,
        "sampling": "paired multinomial bootstrap over outer-held-out whole trajectories",
        "m2_rmse": baseline_rmse, "nested_selected_procedure_rmse": procedure_rmse,
        "difference_procedure_minus_m2": procedure_rmse - baseline_rmse,
        "relative_difference": procedure_rmse / baseline_rmse - 1.0,
        "difference_percentiles_2_5_10_50_90_97_5": percentiles.tolist(),
        "probability_procedure_lower": float(np.mean(differences < 0.0)),
    }


def build_final_report(outer_index_path: Path, audit_path: Path) -> dict[str, Any]:
    outer = _nested_index(outer_index_path, schema=OUTER_INDEX_SCHEMA)
    audit = _sealed(read_json(audit_path), "audit_sha256")
    if audit.get("schema_version") != OUTER_AUDIT_SCHEMA or audit.get("status") != "complete" \
            or audit.get("outer_index_sha256") != outer["index_sha256"]:
        raise NestedInteractionError("final report requires a complete bound outer numerical audit")
    inner = _sealed(read_json(Path(str(outer["inner_report"]["path"]))), "report_sha256")
    if inner.get("report_sha256") != outer["inner_report"]["report_sha256"]:
        raise NestedInteractionError("outer index and inner selection report differ")
    receipt = read_json(Path(str(outer["input_receipt"]["path"]))); frame = load_fit_rows_snapshot(receipt)
    baseline_errors: dict[str, float] = {}; procedure_errors: dict[str, float] = {}
    outer_metrics: dict[str, Any] = {}; sources = []
    for ladder in LADDERS:
        selected_model = str(outer["selected_models"][ladder])
        m2_manifest = load_manifest(Path(str(outer["manifests"]["M2"][ladder])))
        m2_result, _ = _reduced_exploratory(result_path(m2_manifest))
        m2_metric = _test_metrics(frame, m2_manifest["membership"], m2_result["selected"], "M2")
        if selected_model == "M2":
            selected_result = m2_result; selected_metric = m2_metric; selected_path = result_path(m2_manifest)
        else:
            selected_manifest = load_manifest(Path(str(outer["manifests"][selected_model][ladder])))
            selected_result, _ = _reduced_exploratory(result_path(selected_manifest))
            selected_metric = _test_metrics(frame, selected_manifest["membership"], selected_result["selected"], selected_model)
            selected_path = result_path(selected_manifest)
        if set(baseline_errors) & set(m2_metric["trajectory_mse"]):
            raise NestedInteractionError("outer ladder test trajectories overlap")
        baseline_errors.update({str(key): float(value) for key, value in m2_metric["trajectory_mse"].items()})
        procedure_errors.update({str(key): float(value) for key, value in selected_metric["trajectory_mse"].items()})
        outer_metrics[ladder] = {"selected_model": selected_model, "M2": m2_metric,
            "selected": selected_metric, "selected_minus_m2_rmse":
                float(selected_metric["trajectory_balanced_rmse"] - m2_metric["trajectory_balanced_rmse"])}
        sources.extend([
            {"outer_ladder": ladder, "role": "M2", "model": "M2", "path": str(result_path(m2_manifest)),
                "result_sha256": m2_result["result_sha256"]},
            {"outer_ladder": ladder, "role": "inner_selected", "model": selected_model,
                "path": str(selected_path), "result_sha256": selected_result["result_sha256"]},
        ])
    if len(baseline_errors) != 64 or len(procedure_errors) != 64:
        raise NestedInteractionError("final nested comparison does not cover all 64 trajectories exactly once")
    comparison = _paired_outer_comparison(baseline_errors, procedure_errors)
    screen = _screen_report(Path(str(outer["screen_report"]["path"])),
        screen_index_sha256=str(outer["screen_index"]["index_sha256"]))
    screen_model = str(screen["selection"]["selected_model"])
    counts = Counter(str(value) for value in outer["selected_models"].values())
    ladder_wins = sum(float(item["selected_minus_m2_rmse"]) < 0.0 for item in outer_metrics.values())
    interval = comparison["difference_percentiles_2_5_10_50_90_97_5"]
    advance = counts.get(screen_model, 0) >= 3 and comparison["difference_procedure_minus_m2"] < 0.0 \
        and float(interval[3]) < 0.0
    decision = "advance_screen_form_to_prospective_panel" if advance else "retain_registered_m2"
    payload = _seal({
        "schema_version": FINAL_REPORT_SCHEMA, "status": "complete", "exploratory": True,
        "outer_index_sha256": outer["index_sha256"], "audit_sha256": audit["audit_sha256"],
        "screen_selected_model": screen_model, "inner_selection_counts": dict(sorted(counts.items())),
        "outer_metrics": outer_metrics, "paired_comparison": comparison,
        "outer_ladder_wins_for_selected_procedure": ladder_wins,
        "decision": decision,
        "decision_rule": "Advance only if the screen-selected form is chosen in at least 3 of 4 inner loops, the observed outer aggregate improves on M2, and the paired 10th-to-90th percentile interval is wholly below zero.",
        "source_results": sources,
        "claim_boundary": "Same-dataset nested resampling can diagnose selection optimism but cannot prospectively validate a post-hoc law; registered M2 remains authoritative.",
    }, "report_sha256")
    root = Path(str(outer["output_root"])); json_path = root / "reports" / "nested_final_report.json"
    write_json_once(json_path, payload)
    lines = ["# Nested ladder check for the exploratory M2 K family", "",
        f"Decision: **{decision}**", "", f"Screen-selected form: **{screen_model}**", "",
        f"M2 outer RMSE: `{comparison['m2_rmse']:.8f}`", f"Nested-selected outer RMSE: `{comparison['nested_selected_procedure_rmse']:.8f}`",
        f"Difference (procedure - M2): `{comparison['difference_procedure_minus_m2']:.8f}`", "",
        "| Outer ladder | Inner-selected model | Selected RMSE - M2 RMSE |", "|---|---|---:|"]
    for ladder in LADDERS:
        lines.append(f"| {ladder} | {outer_metrics[ladder]['selected_model']} | {outer_metrics[ladder]['selected_minus_m2_rmse']:.8f} |")
    lines.extend(["", payload["claim_boundary"]])
    markdown_path = root / "reports" / "nested_final_report.md"
    write_text_once(markdown_path, "\n".join(lines) + "\n")
    return {"path": str(json_path), "markdown_path": str(markdown_path), **payload}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); commands = root.add_subparsers(dest="command", required=True)
    scales = commands.add_parser("plan-scales"); scales.add_argument("--screen-index", type=Path, required=True)
    scales.add_argument("--screen-report", type=Path, required=True); scales.add_argument("--dry-run", action="store_true")
    inner = commands.add_parser("plan-inner"); inner.add_argument("--scale-index", type=Path, required=True)
    inner.add_argument("--dry-run", action="store_true")
    inner_report = commands.add_parser("inner-report"); inner_report.add_argument("--inner-index", type=Path, required=True)
    outer = commands.add_parser("plan-outer"); outer.add_argument("--scale-index", type=Path, required=True)
    outer.add_argument("--inner-report", type=Path, required=True); outer.add_argument("--dry-run", action="store_true")
    outer_audits = commands.add_parser("plan-outer-audits"); outer_audits.add_argument("--outer-index", type=Path, required=True)
    audit = commands.add_parser("audit-outer"); audit.add_argument("--outer-index", type=Path, required=True)
    audit.add_argument("--audit-index", type=Path, required=True)
    final = commands.add_parser("final-report"); final.add_argument("--outer-index", type=Path, required=True)
    final.add_argument("--audit", type=Path, required=True)
    emit = commands.add_parser("emit-slurm"); emit.add_argument("--index", type=Path, required=True)
    submit = commands.add_parser("record-submission"); submit.add_argument("--index", type=Path, required=True)
    submit.add_argument("--job", action="append", default=[]); submit.add_argument("--attempt", type=int, default=1)
    status = commands.add_parser("status"); status.add_argument("--index", type=Path, required=True)
    reduce_p = commands.add_parser("reduce-index"); reduce_p.add_argument("--index", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "plan-scales":
        result = plan_scales(screen_index_path=args.screen_index, screen_report_path=args.screen_report, dry_run=args.dry_run)
    elif args.command == "plan-inner": result = plan_inner(scale_index_path=args.scale_index, dry_run=args.dry_run)
    elif args.command == "inner-report": result = build_inner_report(args.inner_index)
    elif args.command == "plan-outer": result = plan_outer(scale_index_path=args.scale_index,
        inner_report_path=args.inner_report, dry_run=args.dry_run)
    elif args.command == "plan-outer-audits": result = plan_outer_audits(args.outer_index)
    elif args.command == "audit-outer": result = audit_outer(args.outer_index, args.audit_index)
    elif args.command == "final-report": result = build_final_report(args.outer_index, args.audit)
    elif args.command == "emit-slurm": result = emit_slurm(args.index)
    elif args.command == "record-submission": result = record_submission(args.index, args.job, attempt=args.attempt)
    elif args.command == "status": result = status_index(args.index)
    else: result = reduce_index(args.index)
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
