"""Small Stage 1 campaign planner and the required post-grid checks.

This module deliberately has no scheduler state.  A caller supplies immutable
completed result paths; the planner emits the next immutable manifests.  The
same manifests are consumed locally or by the Slurm-array worker in
``execution.py``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import FitRow, canonical_json_sha256, load_fit_rows_snapshot, sha256_file
from .execution import (
    ExecutionError, build_manifest, chunk_path, load_manifest, load_problem,
    read_json, result_path, run_serial, validate_chunk, write_json_once,
    write_manifest,
)
from .fit import (
    LITERATURE_MODELS, OBJECTIVE_RELATIVE_TOLERANCE, PREDICTION_ABSOLUTE_TOLERANCE, StartPoint,
    d16_adaptive_starts, d16_adaptive_tiers, deterministic_local_perturbations,
    equivalence_check, expand_contacted_bounds, nested_warm_start, predict, start_count,
)
from .selection import LARGEST_COMPUTE_VIEW, build_generalization_holdout_manifest

PRIMARY_FOLDS = ("largest_N_act", "E32", "K200_common_languages")
STAGES = ("m0_ols", "m0_huber", "m1_huber", "m2_huber")
STAGE_MODEL = {"m0_ols": "M0", "m0_huber": "M0", "m1_huber": "M1", "m2_huber": "M2"}
STAGE_OBJECTIVE = {"m0_ols": "ols", "m0_huber": "huber", "m1_huber": "huber", "m2_huber": "huber"}
TARGET_STAGE = {"M0": "m0_huber", "M1": "m1_huber", "M2": "m2_huber"}
AUDIT_SCHEMA = "stage1_fit_grid_audit/v1"
CANDIDATE_REPORT_SCHEMA = "stage1_fit_candidate_report/v5"
GENERALIZATION_MODELS = ("C0", "M0", "J0", "M1")


class CampaignError(RuntimeError):
    pass


def _sealed(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    unsigned = dict(payload); digest = unsigned.pop(field, None)
    if not isinstance(digest, str) or digest != canonical_json_sha256(unsigned):
        raise CampaignError(f"invalid {field}")
    return payload


def _reduced(path: Path) -> Mapping[str, Any]:
    result = _sealed(read_json(path), "result_sha256")
    if result.get("schema_version") != "stage1_fit_reduced/v2":
        raise CampaignError(f"not a reduced fit result: {path}")
    return result


def _completion_receipt(result: Mapping[str, Any]) -> None:
    path = result.get("receipt_path")
    if not isinstance(path, str):
        raise CampaignError("reduced result lacks a completion receipt")
    receipt = _sealed(read_json(Path(path)), "receipt_sha256")
    if receipt.get("schema_version") != "stage1_fit_complete/v2" or receipt.get("manifest_sha256") != result["manifest_sha256"]:
        raise CampaignError("completion receipt does not bind the reduced result")
    if receipt["receipt_sha256"] != result.get("receipt_sha256"):
        raise CampaignError("completion receipt hash differs")


def audit_path(result: Mapping[str, Any], *, output_root: Path) -> Path:
    return output_root / "audits" / f"{str(result['manifest_sha256'])[:20]}.json"


def scientific_result(path: Path) -> Mapping[str, Any]:
    """Return only a non-review result that passed its immutable audit chain."""
    result = _reduced(path)
    _completion_receipt(result)
    if result.get("review_only") or result.get("status") != "reduced" or not result.get("selected"):
        raise CampaignError(f"result is not a complete definitive grid: {path}")
    audit = _sealed(read_json(audit_path(result, output_root=path.parent.parent)), "audit_sha256")
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise CampaignError(f"post-grid audit schema changed: {path}")
    if audit.get("manifest_sha256") != result["manifest_sha256"] or audit.get("result_sha256") != result["result_sha256"]:
        raise CampaignError(f"audit does not bind result: {path}")
    if audit.get("status") != "complete":
        try:
            from .amendment import AmendmentError, validate_accepted_amendment
            validate_accepted_amendment(result, audit, output_root=path.parent.parent)
        except (AmendmentError, ExecutionError, OSError, KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"post-grid checks are incomplete: {path}: {error}") from error
    manifest_path = result.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise CampaignError(f"result lacks its immutable manifest: {path}")
    try:
        try:
            manifest = load_manifest(Path(manifest_path))
        except ExecutionError:
            manifest = load_manifest(Path(manifest_path), require_current_provenance=False)
    except ExecutionError as error:
        raise CampaignError(f"result manifest is no longer valid: {error}") from error
    identity = {key: manifest["fit"][key] for key in
        ("model", "fold_id", "replicate_id", "objective", "bound_version")}
    if (manifest["manifest_sha256"] != result["manifest_sha256"]
            or manifest["input"]["fit_rows_content_sha256"] != result.get("fit_rows_content_sha256")
            or manifest.get("membership") != result.get("membership")
            or manifest.get("review_only") != result.get("review_only")
            or identity != result.get("fit_identity")
            or manifest.get("parents") != result.get("parents")
            or manifest.get("selection_report") != result.get("selection_report")):
        raise CampaignError(f"reduced result does not match its immutable manifest: {path}")
    return result


def _input_receipt(path: Path) -> Mapping[str, Any]:
    receipt = read_json(path)
    needed = {"fit_rows_path", "fit_rows_content_sha256"}
    if not needed <= set(receipt):
        raise CampaignError("input receipt is incomplete")
    return receipt


def _check_membership_input(path: Path, receipt: Mapping[str, Any]) -> None:
    if read_json(path).get("input_sha256") != receipt["fit_rows_content_sha256"]:
        raise CampaignError("membership manifest belongs to different fit rows")


def _selected_model(path: Path) -> str:
    report = read_json(path); unsigned = dict(report); digest = unsigned.pop("report_sha256", None)
    if report.get("schema_version") != CANDIDATE_REPORT_SCHEMA or digest != canonical_json_sha256(unsigned):
        raise CampaignError("candidate report is not sealed")
    model = report.get("selection", {}).get("selected_model")
    if model not in TARGET_STAGE:
        raise CampaignError("candidate report has no registered selected model")
    return str(model)


def _profile(value: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(value)
    if "array_concurrency" not in normalized and "concurrency" in normalized:
        normalized["array_concurrency"] = normalized["concurrency"]
    needed = {"output_root", "chunk_size", "blas_threads", "memory", "time_limit", "array_concurrency"}
    if not needed <= set(normalized):
        raise CampaignError("execution profile is incomplete")
    return normalized


def _membership_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return first.get("membership_sha256") == second.get("membership_sha256") and first.get("kind") == second.get("kind")


def _validate_parent(result: Mapping[str, Any], *, receipt: Mapping[str, Any], membership: Mapping[str, Any],
    model: str | None = None, objective: str | None = None, fold_id: str | None = None) -> None:
    identity = result.get("fit_identity", {})
    if result.get("fit_rows_content_sha256") != receipt["fit_rows_content_sha256"]:
        raise CampaignError("parent uses different fit rows")
    if not isinstance(result.get("membership"), Mapping) or not _membership_equal(result["membership"], membership):
        raise CampaignError("parent uses different frozen membership")
    if model is not None and identity.get("model") != model:
        raise CampaignError("parent model differs")
    if objective is not None and identity.get("objective") != objective:
        raise CampaignError("parent objective differs")
    if fold_id is not None and identity.get("fold_id") != fold_id:
        raise CampaignError("parent fold differs")


def _delta(ols: Mapping[str, Any], *, receipt: Mapping[str, Any], membership: Mapping[str, Any], fold_id: str) -> tuple[str, float | None]:
    _validate_parent(ols, receipt=receipt, membership=membership, model="M0", objective="ols", fold_id=fold_id)
    scale = ols.get("adaptive_huber_scale")
    if not isinstance(scale, Mapping) or not isinstance(scale.get("use_ols"), bool):
        raise CampaignError("validated M0-OLS parent lacks frozen adaptive Huber delta")
    if scale["use_ols"]:
        return ("ols", None)
    delta = scale.get("delta")
    if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)) or float(delta) <= 0.0:
        raise CampaignError("M0-OLS parent has invalid frozen adaptive Huber delta")
    return ("huber", float(delta))


def _parent_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    return {str(item["kind"]): Path(str(item["path"])) for item in manifest.get("parents", [])}


def _make_manifest(*, receipt: Mapping[str, Any], profile: Mapping[str, Any], phase: str, model: str,
    fold_id: str, objective: str, membership: Mapping[str, Any], delta: float | None = None,
    parents: Mapping[str, Path] | None = None, bounds: Sequence[Any] | None = None,
    bound_version: int = 0, replicate_id: int | None = None, action: str = "grid",
    explicit_starts: Sequence[StartPoint] | None = None, selection_report: Path | None = None,
    persist: bool = True) -> Path | Mapping[str, Any]:
    profile = _profile(profile)
    manifest = build_manifest(
        output_root=Path(str(profile["output_root"])), fit_rows_path=Path(str(receipt["fit_rows_path"])),
        fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]), phase=phase, model=model,
        fold_id=fold_id, objective=objective, delta=delta, chunk_size=int(profile["chunk_size"]),
        bounds=bounds, bound_version=bound_version, replicate_id=replicate_id,
        parent_results=parents, selection_report=selection_report, membership=membership, explicit_starts=explicit_starts,
        action=action, blas_threads=int(profile["blas_threads"]), partition=profile.get("partition"),
        account=profile.get("account"), reservation=profile.get("reservation"), memory=str(profile["memory"]),
        time_limit=str(profile["time_limit"]), concurrency=int(profile["array_concurrency"]),
    )
    return write_manifest(manifest) if persist else manifest


def _stage_manifest(*, stage: str, scope: str, fold_id: str, membership: Mapping[str, Any], receipt: Mapping[str, Any],
    profile: Mapping[str, Any], completed: Mapping[str, Path], replicate_id: int | None = None,
    selection_report: Path | None = None) -> Path:
    index = STAGES.index(stage)
    parents: dict[str, Path] = {}
    if stage != "m0_ols":
        ols = scientific_result(completed[f"{scope}/m0_ols"])
        ols_path = completed[f"{scope}/m0_ols"]
        objective, delta = _delta(ols, receipt=receipt, membership=membership, fold_id=fold_id)
        parents["scale_m0_ols"] = ols_path
        if stage == "m0_huber":
            # The initial robust grid inherits the M0-OLS bounds.  A later audit
            # may expand only the robust grid through its bound-repeat parent.
            parent_manifest = _manifest_for_result(ols_path, ols)
            bounds = tuple(_bounds(parent_manifest))
        else:
            bounds = None
            warm_stage = "m0_huber" if stage == "m1_huber" else "m1_huber"
            warm_path = completed[f"{scope}/{warm_stage}"]
            warm = scientific_result(warm_path)
            _validate_parent(warm, receipt=receipt, membership=membership, fold_id=fold_id)
            parents["warm"] = warm_path
        return _make_manifest(receipt=receipt, profile=profile, phase=f"{scope}-{stage}", model=STAGE_MODEL[stage],
            fold_id=fold_id, objective=objective, membership=membership, delta=delta, parents=parents,
            bounds=bounds, replicate_id=replicate_id, selection_report=selection_report)
    return _make_manifest(receipt=receipt, profile=profile, phase=f"{scope}-{stage}", model="M0", fold_id=fold_id,
        objective="ols", membership=membership, replicate_id=replicate_id, selection_report=selection_report)


def _manifest_for_result(path: Path, result: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest_path = result.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise CampaignError(f"result lacks its immutable manifest path: {path}")
    manifest = load_manifest(Path(manifest_path))
    if manifest["manifest_sha256"] != result["manifest_sha256"]:
        raise CampaignError("result manifest identity differs")
    return manifest


def _bounds(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
    from .fit import OuterBound
    return tuple(OuterBound(**item) for item in manifest["fit"]["bounds"])


def _validate_all_data_reference(path: Path, *, receipt: Mapping[str, Any], candidate_report_path: Path,
    model: str, objective: str | None = None) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Bind one D16 anchor to the selected, audited all-data campaign."""
    result = scientific_result(path)
    identity = result.get("fit_identity", {})
    if (result.get("fit_rows_content_sha256") != receipt["fit_rows_content_sha256"]
            or result.get("membership", {}).get("kind") != "all"
            or identity.get("model") != model or identity.get("fold_id") != "all_data"
            or (objective is not None and identity.get("objective") != objective)):
        raise CampaignError("D16 all-data anchor has the wrong input, model, objective, or membership")
    candidate = read_json(candidate_report_path)
    binding = result.get("selection_report")
    if (not isinstance(binding, Mapping)
            or binding.get("path") != str(candidate_report_path.resolve())
            or binding.get("file_sha256") != sha256_file(candidate_report_path)
            or binding.get("report_sha256") != candidate.get("report_sha256")):
        raise CampaignError("D16 all-data anchor is not bound to the selected candidate report")
    return result, _manifest_for_result(path, result)


def _reference_coordinates(result: Mapping[str, Any]) -> tuple[list[float], tuple[list[float], ...]]:
    selected = result.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("coordinates"), list):
        raise CampaignError("D16 all-data anchor lacks optimizer coordinates")
    attempts = [item for item in result.get("attempts", []) if isinstance(item, Mapping)
        and item.get("success") and isinstance(item.get("objective"), (int, float))
        and isinstance(item.get("coordinates"), list)]
    attempts.sort(key=lambda item: (float(item["objective"]), str(item.get("start_id", ""))))
    return list(selected["coordinates"]), tuple(list(item["coordinates"]) for item in attempts[:3])


def _reference_basin_starts(result: Mapping[str, Any], *, model: str, bounds: Sequence[Any], target: int,
    coordinates: tuple[list[float], tuple[list[float], ...]] | None = None) -> tuple[StartPoint, ...]:
    center, basin = _reference_coordinates(result) if coordinates is None else coordinates
    try:
        return d16_adaptive_starts(model, center, basin, bounds, target=target)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"cannot construct frozen D16 starts: {error}") from error


def _d16_initial_manifest(*, scope: str, stage: str, membership: Mapping[str, Any], receipt: Mapping[str, Any],
    profile: Mapping[str, Any], completed: Mapping[str, Path], candidate_report_path: Path, replicate_id: int,
    reference_bundle: tuple[Path, Mapping[str, Any], Mapping[str, Any], tuple[list[float], tuple[list[float], ...]]]) -> Path:
    model = STAGE_MODEL[stage]
    reference_path, reference, reference_manifest, reference_coordinates = reference_bundle
    parents: dict[str, Path] = {"full_data": reference_path}
    if stage == "m0_ols":
        objective, delta = "ols", None
        bounds = _bounds(reference_manifest)
    else:
        scale_path = completed[f"{scope}/m0_ols"]
        scale = scientific_result(scale_path)
        objective, delta = _delta(scale, receipt=receipt, membership=membership, fold_id="d16")
        parents["scale_m0_ols"] = scale_path
        bounds = _bounds(_manifest_for_result(scale_path, scale)) if model == "M0" else _bounds(reference_manifest)
    tier = d16_adaptive_tiers(model)[0]
    starts = _reference_basin_starts(reference, model=model, bounds=bounds, target=tier,
        coordinates=reference_coordinates)
    return _make_manifest(receipt=receipt, profile=profile,
        phase=f"{scope}-{stage}-adaptive-{tier:03d}", model=model, fold_id="d16", objective=objective,
        membership=membership, delta=delta, parents=parents, bounds=bounds, replicate_id=replicate_id,
        action="d16_adaptive", explicit_starts=starts, selection_report=candidate_report_path)


def _plan_stage(*, scope_memberships: Mapping[str, Mapping[str, Any]], receipt: Mapping[str, Any], profile: Mapping[str, Any],
    completed: Mapping[str, Path], target_stage: str, replicate_ids: Mapping[str, int | None] | None = None,
    selection_report: Path | None = None) -> dict[str, Path]:
    upto = STAGES.index(target_stage)
    for stage in STAGES[:upto + 1]:
        missing = [scope for scope in sorted(scope_memberships) if f"{scope}/{stage}" not in completed]
        if not missing:
            continue
        for scope in missing:
            for previous in STAGES[:STAGES.index(stage)]:
                key = f"{scope}/{previous}"
                if key not in completed:
                    raise CampaignError(f"missing prerequisite {key}")
                scientific_result(completed[key])
        return {f"{scope}/{stage}": _stage_manifest(stage=stage, scope=scope,
            fold_id="d16" if scope.startswith("d16:") else scope.rsplit(":", 1)[-1],
            membership=scope_memberships[scope], receipt=receipt, profile=profile, completed=completed,
            replicate_id=None if replicate_ids is None else replicate_ids[scope], selection_report=selection_report) for scope in missing}
    return {}


def plan_candidate_campaign(*, input_receipt_path: Path, holdouts_path: Path, profile: Mapping[str, Any],
    completed: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Plan the next exact candidate stage for all three primary folds."""
    from .execution import bind_membership
    receipt = _input_receipt(input_receipt_path)
    _check_membership_input(holdouts_path, receipt)
    memberships = {fold: bind_membership(holdouts_path, kind="primary", membership_id=fold) for fold in PRIMARY_FOLDS}
    return _plan_stage(scope_memberships=memberships, receipt=receipt, profile=profile, completed=dict(completed or {}),
        target_stage="m2_huber")


def plan_selected_all_data(*, candidate_report_path: Path, input_receipt_path: Path, holdouts_path: Path,
    profile: Mapping[str, Any], completed: Mapping[str, Path] | None = None) -> dict[str, Path]:
    from .execution import bind_membership
    selected_model = _selected_model(candidate_report_path)
    receipt = _input_receipt(input_receipt_path); _check_membership_input(holdouts_path, receipt)
    return _plan_stage(scope_memberships={"all_data": bind_membership(holdouts_path, kind="all")},
        receipt=receipt, profile=profile, completed=dict(completed or {}), target_stage=TARGET_STAGE[selected_model],
        selection_report=candidate_report_path)


def plan_d16_refits(*, candidate_report_path: Path, input_receipt_path: Path, resampling_path: Path,
    profile: Mapping[str, Any], completed: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Plan D16 estimator-stability refits from audited all-data anchors.

    D16 does not repeat model discovery.  It recomputes the M0 residual scale
    on each subset, then refits only the already selected law from the frozen
    all-data basin.  ``audit_grid`` escalates an unresolved 32-start fit through
    128 and 512 starts and finally the complete Cartesian grid.
    """
    from .execution import bind_membership
    selected_model = _selected_model(candidate_report_path)
    payload = read_json(resampling_path)
    receipt = _input_receipt(input_receipt_path); _check_membership_input(resampling_path, receipt)
    subsets = payload.get("subsets")
    if not isinstance(subsets, list) or len(subsets) != 100:
        raise CampaignError("D16 requires exactly 100 frozen subsets")
    scopes = {f"d16:{int(item['replicate_id']):03d}": bind_membership(resampling_path, kind="d16", membership_id=int(item["replicate_id"])) for item in subsets}
    ids = {scope: int(scope.rsplit(":", 1)[1]) for scope in scopes}
    done = dict(completed or {})
    target_stage = TARGET_STAGE[selected_model]
    for required in ("all_data/m0_ols", f"all_data/{target_stage}"):
        if required not in done:
            raise CampaignError(f"D16 requires the audited selected all-data result {required}")
    scale_reference, scale_manifest = _validate_all_data_reference(done["all_data/m0_ols"], receipt=receipt,
        candidate_report_path=candidate_report_path, model="M0", objective="ols")
    target_reference, target_manifest = _validate_all_data_reference(done[f"all_data/{target_stage}"], receipt=receipt,
        candidate_report_path=candidate_report_path, model=selected_model)
    scale_bundle = (done["all_data/m0_ols"], scale_reference, scale_manifest, _reference_coordinates(scale_reference))
    target_bundle = (done[f"all_data/{target_stage}"], target_reference, target_manifest,
        _reference_coordinates(target_reference))
    missing_scale = [scope for scope in sorted(scopes) if f"{scope}/m0_ols" not in done]
    if missing_scale:
        return {f"{scope}/m0_ols": _d16_initial_manifest(scope=scope, stage="m0_ols", membership=scopes[scope],
            receipt=receipt, profile=profile, completed=done, candidate_report_path=candidate_report_path,
            replicate_id=ids[scope], reference_bundle=scale_bundle) for scope in missing_scale}
    for scope in sorted(scopes):
        scale = scientific_result(done[f"{scope}/m0_ols"])
        _validate_parent(scale, receipt=receipt, membership=scopes[scope], model="M0", objective="ols", fold_id="d16")
    missing_target = [scope for scope in sorted(scopes) if f"{scope}/{target_stage}" not in done]
    return {f"{scope}/{target_stage}": _d16_initial_manifest(scope=scope, stage=target_stage,
        membership=scopes[scope], receipt=receipt, profile=profile, completed=done,
        candidate_report_path=candidate_report_path, replicate_id=ids[scope],
        reference_bundle=target_bundle) for scope in missing_target}


def plan_secondary_diagnostic(*, diagnostic: str, candidate_report_path: Path, input_receipt_path: Path, holdouts_path: Path,
    profile: Mapping[str, Any], completed: Mapping[str, Path] | None = None) -> dict[str, Path]:
    from .execution import bind_membership
    kinds = {"random": ("random", "random_13_of_64"), "lowest": ("lowest", "lowest_terminal_macro_loss_30"),
        "terminal": ("terminal", None)}
    if diagnostic not in kinds:
        raise CampaignError("unknown secondary diagnostic")
    selected_model = _selected_model(candidate_report_path)
    kind, identifier = kinds[diagnostic]
    scope = f"secondary:{diagnostic}"
    receipt = _input_receipt(input_receipt_path); _check_membership_input(holdouts_path, receipt)
    target = "m2_huber" if diagnostic == "lowest" else TARGET_STAGE[selected_model]
    return _plan_stage(scope_memberships={scope: bind_membership(holdouts_path, kind=kind, membership_id=identifier)},
        receipt=receipt, profile=profile, completed=dict(completed or {}), target_stage=target,
        selection_report=candidate_report_path)


def plan_literature_comparison(*, candidate_report_path: Path, input_receipt_path: Path, holdouts_path: Path,
    profile: Mapping[str, Any], completed: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Plan only the missing C0/J0/JH fits on the frozen D17 literature views."""
    from .execution import bind_membership
    if _selected_model(candidate_report_path) != "M1":
        raise CampaignError("the frozen literature comparison currently requires selected model M1")
    receipt = _input_receipt(input_receipt_path); _check_membership_input(holdouts_path, receipt)
    memberships = {fold: bind_membership(holdouts_path, kind="primary", membership_id=fold)
        for fold in PRIMARY_FOLDS}
    memberships.update({
        "lowest": bind_membership(holdouts_path, kind="lowest", membership_id="lowest_terminal_macro_loss_30"),
        "all_data": bind_membership(holdouts_path, kind="all"),
    })
    scale_keys = {**{fold: f"{fold}/m0_ols" for fold in PRIMARY_FOLDS},
        "lowest": "secondary:lowest/m0_ols", "all_data": "all_data/m0_ols"}
    candidate = read_json(candidate_report_path)
    expected_selection = {"path": str(candidate_report_path.resolve()),
        "file_sha256": sha256_file(candidate_report_path), "report_sha256": candidate["report_sha256"]}
    done = dict(completed or {}); preflight: dict[str, tuple[Path, str, float | None]] = {}
    for view, membership in memberships.items():
        scale_key = scale_keys[view]
        if scale_key not in done:
            raise CampaignError(f"literature comparison requires prerequisite {scale_key}")
        scale_path = done[scale_key]; scale = scientific_result(scale_path)
        objective, delta = _delta(scale, receipt=receipt, membership=membership, fold_id=view)
        if view in {"lowest", "all_data"} and scale.get("selection_report") != expected_selection:
            raise CampaignError(f"{scale_key} is not bound to the selected candidate report")
        preflight[view] = (scale_path, objective, delta)
        for model in LITERATURE_MODELS:
            key = f"d17:{view}/{model}"
            if key in done:
                result = scientific_result(done[key])
                _validate_parent(result, receipt=receipt, membership=membership, model=model,
                    objective=objective, fold_id=view)
                expected_scale = {"kind": "scale_m0_ols", "path": str(scale_path.resolve()),
                    "file_sha256": sha256_file(scale_path), "result_sha256": scale["result_sha256"]}
                if [item for item in result.get("parents", []) if isinstance(item, Mapping)
                        and item.get("kind") == "scale_m0_ols"] != [expected_scale]:
                    raise CampaignError(f"{key} does not use the frozen M0-OLS scale parent")
                if result.get("selection_report") != expected_selection:
                    raise CampaignError(f"{key} is not bound to the selected candidate report")
    payloads: dict[str, Mapping[str, Any]] = {}
    for view, membership in memberships.items():
        scale_path, objective, delta = preflight[view]
        for model in LITERATURE_MODELS:
            key = f"d17:{view}/{model}"
            if key in done:
                continue
            payload = _make_manifest(receipt=receipt, profile=profile, phase=key, model=model,
                fold_id=view, objective=objective, membership=membership, delta=delta,
                parents={"scale_m0_ols": scale_path}, selection_report=candidate_report_path, persist=False)
            if not isinstance(payload, Mapping):
                raise AssertionError("non-persistent manifest planning returned a path")
            payloads[key] = payload
    return {key: write_manifest(payload) for key, payload in payloads.items()}


def _generalization_memberships(*, holdouts_path: Path, generalization_path: Path,
    receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    from .execution import bind_membership
    _check_membership_input(holdouts_path, receipt); _check_membership_input(generalization_path, receipt)
    holdouts = read_json(holdouts_path); supplement = _sealed(read_json(generalization_path), 'manifest_sha256')
    source = {'path': str(holdouts_path.resolve()), 'file_sha256': sha256_file(holdouts_path), 'manifest_sha256': holdouts.get('manifest_sha256')}
    frame = load_fit_rows_snapshot(receipt); rows = tuple(FitRow(**record) for record in frame.to_dict('records'))
    expected = build_generalization_holdout_manifest(rows, input_sha256=str(receipt['fit_rows_content_sha256']), source_holdouts=source)
    if supplement != expected: raise CampaignError('generalization holdout differs from the recomputed frozen split')
    return {'random_13_of_64': bind_membership(holdouts_path, kind='random', membership_id='random_13_of_64'),
        'terminal_max_exposure': bind_membership(holdouts_path, kind='terminal'),
        LARGEST_COMPUTE_VIEW: bind_membership(generalization_path, kind='compute', membership_id=LARGEST_COMPUTE_VIEW)}


def _generalization_result(path: Path, *, receipt: Mapping[str, Any], membership: Mapping[str, Any],
    candidate_binding: Mapping[str, Any], model: str, fold_id: str, objective: str, delta: float | None,
    parents: Mapping[str, tuple[Path, Mapping[str, Any]]]) -> Mapping[str, Any]:
    result = scientific_result(path)
    _validate_parent(result, receipt=receipt, membership=membership, model=model, objective=objective, fold_id=fold_id)
    if result.get('frozen_delta') != delta: raise CampaignError('generalization result uses a different frozen Huber delta')
    if result.get('selection_report') != candidate_binding: raise CampaignError('generalization result is not bound to the selected candidate report')
    for kind, (parent_path, parent) in parents.items():
        expected = {'kind': kind, 'path': str(parent_path.resolve()), 'file_sha256': sha256_file(parent_path), 'result_sha256': parent['result_sha256']}
        if [item for item in result.get('parents', []) if item.get('kind') == kind] != [expected]:
            raise CampaignError(f'generalization result has the wrong {kind} parent')
    return result


def plan_generalization_comparison(*, candidate_report_path: Path, input_receipt_path: Path,
    holdouts_path: Path, generalization_holdouts_path: Path, profile: Mapping[str, Any],
    completed: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Plan the three-wave, post-selection C0/M0/J0/M1 battery."""
    if _selected_model(candidate_report_path) != 'M1':
        raise CampaignError('generalization comparison requires the frozen selected M1 law')
    receipt = _input_receipt(input_receipt_path)
    memberships = _generalization_memberships(holdouts_path=holdouts_path, generalization_path=generalization_holdouts_path, receipt=receipt)
    candidate = read_json(candidate_report_path)
    input_binding = {'path': str(input_receipt_path.resolve()), 'file_sha256': sha256_file(input_receipt_path)}
    if (candidate.get('input_sha256') != receipt['fit_rows_content_sha256'] or candidate.get('input_receipt') != input_binding
            or candidate.get('holdout_manifest_sha256') != read_json(holdouts_path).get('manifest_sha256')): raise CampaignError('generalization comparison differs from the selected candidate inputs')
    binding = {'path': str(candidate_report_path.resolve()), 'file_sha256': sha256_file(candidate_report_path), 'report_sha256': candidate['report_sha256']}
    done = dict(completed or {}); scales: dict[str, tuple[Path, Mapping[str, Any]]] = {}; payloads = {}
    scale_keys = {f'generalization:{view}/m0_ols' for view in memberships}; form_keys = {f'generalization:{view}/{model}' for view in memberships for model in GENERALIZATION_MODELS}
    present = {key for key in done if key.startswith('generalization:')}
    if not present <= scale_keys | form_keys: raise CampaignError('completed index contains an unknown generalization stage')
    if (scale_keys - set(done)) and (present - scale_keys): raise CampaignError('generalization result precedes a complete M0-OLS wave')
    for view, membership in memberships.items():
        key = f'generalization:{view}/m0_ols'
        if key not in done:
            payloads[key] = _make_manifest(receipt=receipt, profile=profile, phase=key, model='M0', fold_id=view,
                objective='ols', membership=membership, selection_report=candidate_report_path, persist=False)
        else:
            scale = _generalization_result(done[key], receipt=receipt, membership=membership, candidate_binding=binding, model='M0', fold_id=view, objective='ols', delta=None, parents={})
            scales[view] = (done[key], scale)
    if payloads: return {key: write_manifest(payload) for key, payload in payloads.items()}
    objectives = {view: _delta(scale, receipt=receipt, membership=memberships[view], fold_id=view) for view, (_, scale) in scales.items()}
    for view, membership in memberships.items():
        scale_path, scale = scales[view]; objective, delta = objectives[view]
        for model in ('C0', 'M0', 'J0'):
            key = f'generalization:{view}/{model}'
            if key in done:
                _generalization_result(done[key], receipt=receipt, membership=membership, candidate_binding=binding,
                    model=model, fold_id=view, objective=objective, delta=delta, parents={'scale_m0_ols': (scale_path, scale)})
            else:
                payloads[key] = _make_manifest(receipt=receipt, profile=profile, phase=key, model=model, fold_id=view,
                    objective=objective, membership=membership, delta=delta, parents={'scale_m0_ols': scale_path}, selection_report=candidate_report_path, persist=False)
    if payloads:
        if any(f'generalization:{view}/M1' in done for view in memberships): raise CampaignError('M1 generalization result precedes a complete baseline wave')
        return {key: write_manifest(payload) for key, payload in payloads.items()}
    for view, membership in memberships.items():
        key = f'generalization:{view}/M1'; scale_path, scale = scales[view]
        warm_path = done[f'generalization:{view}/M0']
        warm = _generalization_result(warm_path, receipt=receipt, membership=membership, candidate_binding=binding,
            model='M0', fold_id=view, objective=objectives[view][0], delta=objectives[view][1], parents={'scale_m0_ols': (scale_path, scale)})
        if key in done:
            _generalization_result(done[key], receipt=receipt, membership=membership, candidate_binding=binding,
                model='M1', fold_id=view, objective=objectives[view][0], delta=objectives[view][1], parents={'scale_m0_ols': (scale_path, scale), 'warm': (warm_path, warm)})
        else:
            payloads[key] = _make_manifest(receipt=receipt, profile=profile, phase=key, model='M1', fold_id=view,
                objective=objectives[view][0], membership=membership, delta=objectives[view][1],
                parents={'scale_m0_ols': scale_path, 'warm': warm_path}, selection_report=candidate_report_path, persist=False)
    return {key: write_manifest(payload) for key, payload in payloads.items()}


def _records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for task in manifest["tasks"]:
        chunk = read_json(chunk_path(manifest, int(task["task_index"])))
        validate_chunk(chunk, manifest, task)
        records.extend(chunk["records"])
    return records


def _record_prediction(record: Mapping[str, Any], *, manifest: Mapping[str, Any]) -> np.ndarray:
    selected = record
    if not {"shared_parameters", "language_coefficients"} <= set(selected):
        raise CampaignError("derived start lacks complete coefficients")
    from .fit import SharedParameters, effective_exposure_data
    problem = load_problem(manifest); shared = SharedParameters(**selected["shared_parameters"])
    by_language = {str(item["language"]): item for item in selected["language_coefficients"]}
    nuisance = [by_language[str(language)] for language in problem.language_ids]
    effective_d = effective_exposure_data(manifest["fit"]["model"], shared, problem.D,
        problem.D_transfer, problem.D_other)
    return predict(manifest["fit"]["model"], shared, problem.N, effective_d, problem.E, problem.K,
        [item["L_inf"] for item in nuisance], [item["A"] for item in nuisance], [item["B"] for item in nuisance])


def _equivalent_to_retained(records: Sequence[Mapping[str, Any]], *, manifest: Mapping[str, Any],
    retained: Mapping[str, Any], retained_manifest: Mapping[str, Any]) -> bool:
    """Require three derived starts to return to the grid's retained basin."""
    usable = [record for record in records if record.get("success") and isinstance(record.get("objective"), (int, float))]
    if len(usable) < 3:
        return False
    if not isinstance(retained.get("objective"), (int, float)):
        raise CampaignError("retained grid optimum lacks a finite objective")
    prediction = _record_prediction(retained, manifest=retained_manifest)
    count = 0
    for record in usable:
        check = equivalence_check(float(retained["objective"]), float(record["objective"]), prediction,
            _record_prediction(record, manifest=manifest), objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
            prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE)
        count += int(check.equivalent)
    return count >= 3


def _derived(*, base: Mapping[str, Any], manifest_path: Path, action: str, starts: Sequence[StartPoint]) -> tuple[Path, Mapping[str, Any]]:
    parents = _parent_paths(base)
    parents["grid"] = result_path(base)
    manifest = _make_manifest(receipt={"fit_rows_path": base["input"]["fit_rows_path"],
        "fit_rows_content_sha256": base["input"]["fit_rows_content_sha256"]}, profile=base["execution"],
        phase=f"{base['phase']}-{action}", model=base["fit"]["model"], fold_id=base["fit"]["fold_id"],
        objective=base["fit"]["objective"], membership=base["membership"], delta=base["fit"]["delta"], parents=parents,
        bounds=_bounds(base), bound_version=base["fit"]["bound_version"], replicate_id=base["fit"]["replicate_id"],
        action=action, explicit_starts=starts,
        selection_report=None if base["selection_report"] is None else Path(base["selection_report"]["path"]))
    derived_manifest = load_manifest(manifest)
    return (manifest, run_serial(derived_manifest))


def _audit_payload(*, manifest: Mapping[str, Any], result: Mapping[str, Any], status: str, actions: Sequence[Mapping[str, Any]],
    reason: str | None = None, followup: Path | None = None) -> Mapping[str, Any]:
    unsigned = {"schema_version": AUDIT_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
        "result_sha256": result["result_sha256"], "status": status, "actions": list(actions), "reason": reason,
        "followup_manifest": None if followup is None else str(followup)}
    return {**unsigned, "audit_sha256": canonical_json_sha256(unsigned)}


def _validate_scale_binding(manifest: Mapping[str, Any]) -> None:
    """Require every robust-stage delta to come from its exact M0-OLS parent."""
    if manifest["fit"]["model"] == "M0" and manifest["fit"]["objective"] == "ols":
        return
    parent = _parent_paths(manifest).get("scale_m0_ols")
    if parent is None:
        raise CampaignError("non-OLS stage lacks its M0-OLS scale parent")
    source = scientific_result(parent)
    receipt = {"fit_rows_content_sha256": manifest["input"]["fit_rows_content_sha256"]}
    _validate_parent(source, receipt=receipt, membership=manifest["membership"], model="M0", objective="ols",
        fold_id=manifest["fit"]["fold_id"])
    scale = source.get("adaptive_huber_scale")
    if not isinstance(scale, Mapping) or not isinstance(scale.get("use_ols"), bool):
        raise CampaignError("M0-OLS scale parent is invalid")
    expected_objective = "ols" if scale["use_ols"] else "huber"
    expected_delta = None if scale["use_ols"] else scale.get("delta")
    if manifest["fit"]["objective"] != expected_objective or manifest["fit"]["delta"] != expected_delta:
        raise CampaignError("stage does not use its parent’s frozen Huber delta")
    if manifest["fit"]["model"] == "M0" and source.get("fit_bounds") != manifest["fit"]["bounds"]:
        repeat = _parent_paths(manifest).get("bound_repeat")
        adaptive = _parent_paths(manifest).get("adaptive_parent")
        if repeat is None and adaptive is None:
            raise CampaignError("initial M0 Huber stage does not use its M0-OLS parent bounds")
        previous = _reduced(repeat if repeat is not None else adaptive)
        _validate_parent(previous, receipt=receipt, membership=manifest["membership"], model="M0",
            objective=manifest["fit"]["objective"], fold_id=manifest["fit"]["fold_id"])
        expected_version = manifest["fit"]["bound_version"] - 1 if repeat is not None else manifest["fit"]["bound_version"]
        if previous.get("fit_identity", {}).get("bound_version") != expected_version:
            raise CampaignError("M0 Huber follow-up has a non-contiguous bound version")
        if adaptive is not None and previous.get("fit_bounds") != manifest["fit"]["bounds"]:
            raise CampaignError("M0 Huber adaptive escalation changed bounds")


def _d16_followup(*, manifest: Mapping[str, Any], result: Mapping[str, Any], tier: int,
    bounds: Sequence[Any], bound_repeat: bool) -> Path:
    """Freeze one adaptive escalation or same-tier bound repeat."""
    parents = _parent_paths(manifest)
    parents.pop("grid", None)
    if bound_repeat:
        parents["bound_repeat"] = result_path(manifest)
        phase = f"{manifest['phase']}-bounds-{manifest['fit']['bound_version'] + 1}"
        bound_version = int(manifest["fit"]["bound_version"]) + 1
    else:
        parents.pop("bound_repeat", None)
        parents["adaptive_parent"] = result_path(manifest)
        phase = f"{manifest['phase']}-escalate-{tier:06d}"
        bound_version = int(manifest["fit"]["bound_version"])
    full_data_path = parents.get("full_data")
    if full_data_path is None:
        raise CampaignError("D16 follow-up lost its all-data anchor")
    reference = scientific_result(full_data_path)
    full = d16_adaptive_tiers(manifest["fit"]["model"])[-1]
    explicit = None if tier == full else _reference_basin_starts(reference,
        model=manifest["fit"]["model"], bounds=bounds, target=tier)
    return _make_manifest(receipt={"fit_rows_path": manifest["input"]["fit_rows_path"],
        "fit_rows_content_sha256": manifest["input"]["fit_rows_content_sha256"]}, profile=manifest["execution"],
        phase=phase, model=manifest["fit"]["model"], fold_id="d16", objective=manifest["fit"]["objective"],
        membership=manifest["membership"], delta=manifest["fit"]["delta"], parents=parents, bounds=bounds,
        bound_version=bound_version, replicate_id=manifest["fit"]["replicate_id"],
        action="d16_full_grid" if tier == full else "d16_adaptive", explicit_starts=explicit,
        selection_report=Path(manifest["selection_report"]["path"]))


def _d16_numerical_checks(*, manifest_path: Path, manifest: Mapping[str, Any],
    result: Mapping[str, Any]) -> tuple[bool, list[Mapping[str, Any]], str | None]:
    """Run basin reproducibility and independent projected/direct checks."""
    selected = result["selected"]
    usable = [record for record in _records(manifest) if record.get("success") and record.get("coordinates")]
    usable.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
    actions: list[Mapping[str, Any]] = []
    basin_ok = False
    if len(usable) >= 3:
        starts = tuple(StartPoint(f"basin:{index}", None, np.asarray(item["coordinates"], dtype=np.float64), "basin")
            for index, item in enumerate(usable[:3]))
        basin_path, _ = _derived(base=manifest, manifest_path=manifest_path, action="basin", starts=starts)
        basin_manifest = load_manifest(basin_path)
        basin_ok = _equivalent_to_retained(_records(basin_manifest), manifest=basin_manifest,
            retained=selected, retained_manifest=manifest)
        actions.append({"action": "basin", "resolved": basin_ok, "manifest": str(basin_path)})
    else:
        actions.append({"action": "basin", "resolved": False, "successful_starts": len(usable)})
    if not basin_ok:
        return False, actions, "adaptive basin unresolved"
    direct_start = StartPoint("direct:best", None, np.asarray(selected["coordinates"], dtype=np.float64), "direct")
    direct_path, direct_result = _derived(base=manifest, manifest_path=manifest_path,
        action="direct", starts=(direct_start,))
    direct_record = direct_result.get("selected")
    resolved = bool(direct_record and direct_record.get("direct_equivalence", {}).get("resolved"))
    actions.append({"action": "direct", "resolved": resolved, "manifest": str(direct_path)})
    return resolved, actions, None if resolved else "projected/direct equivalence unresolved"


def _validate_d16_lineage(manifest: Mapping[str, Any]) -> None:
    parents = _parent_paths(manifest)
    full_path = parents.get("full_data")
    if full_path is None:
        raise CampaignError("D16 manifest lacks its all-data anchor")
    full = scientific_result(full_path); identity = full.get("fit_identity", {})
    if (full.get("fit_rows_content_sha256") != manifest["input"]["fit_rows_content_sha256"]
            or full.get("membership", {}).get("kind") != "all"
            or identity.get("model") != manifest["fit"]["model"]
            or identity.get("fold_id") != "all_data"
            or full.get("selection_report") != manifest.get("selection_report")):
        raise CampaignError("D16 all-data anchor changed scientific identity")
    adaptive_path = parents.get("adaptive_parent")
    if adaptive_path is not None:
        previous = _reduced(adaptive_path); _completion_receipt(previous)
        previous_identity = previous.get("fit_identity", {})
        if (previous.get("fit_rows_content_sha256") != manifest["input"]["fit_rows_content_sha256"]
                or previous.get("membership") != manifest.get("membership")
                or previous_identity.get("model") != manifest["fit"]["model"]
                or previous_identity.get("replicate_id") != manifest["fit"]["replicate_id"]
                or previous.get("selection_report") != manifest.get("selection_report")):
            raise CampaignError("D16 adaptive parent changed scientific identity")


def _audit_d16(manifest_path: Path, manifest: Mapping[str, Any], result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Audit one selected-estimator tier and escalate only on an audit failure."""
    if result.get("review_only"):
        raise CampaignError("review-only result cannot enter D16")
    _validate_d16_lineage(manifest)
    _validate_scale_binding(manifest)
    tiers = d16_adaptive_tiers(manifest["fit"]["model"])
    tier = int(manifest["fit"]["start_count"])
    if tier not in tiers:
        raise CampaignError("D16 manifest uses an unregistered adaptive tier")
    actions: list[Mapping[str, Any]] = [{"action": "adaptive_tier", "start_count": tier,
        "full_grid": manifest["fit"]["start_mode"] == "grid"}]
    selected = result.get("selected")
    if result.get("status") != "reduced" or not isinstance(selected, Mapping):
        reason = "adaptive tier produced no successful optimizer start"
        if tier != tiers[-1]:
            next_tier = tiers[tiers.index(tier) + 1]
            followup = _d16_followup(manifest=manifest, result=result, tier=next_tier,
                bounds=_bounds(manifest), bound_repeat=False)
            actions.append({"action": "escalate", "from": tier, "to": next_tier, "reason": reason})
            payload = _audit_payload(manifest=manifest, result=result, status="needs_escalation", actions=actions,
                reason=reason, followup=followup)
        else:
            payload = _audit_payload(manifest=manifest, result=result, status="failed", actions=actions,
                reason=f"full-grid D16 fallback failed: {reason}")
        write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload)
        return payload
    contacts = selected.get("boundary_contacts", [])
    if contacts:
        from .fit import BoundaryContact
        expansion = expand_contacted_bounds(_bounds(manifest), tuple(BoundaryContact(**item) for item in contacts))
        actions.append({"action": "boundary", "contacts": contacts, "nonidentified": list(expansion.nonidentified)})
        if expansion.nonidentified:
            payload = _audit_payload(manifest=manifest, result=result, status="failed", actions=actions,
                reason="non-identification after frozen bound-expansion limit")
        else:
            followup = _d16_followup(manifest=manifest, result=result, tier=tier,
                bounds=expansion.bounds, bound_repeat=True)
            payload = _audit_payload(manifest=manifest, result=result, status="needs_bound_repeat", actions=actions,
                followup=followup)
        write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload)
        return payload
    resolved, checks, reason = _d16_numerical_checks(manifest_path=manifest_path, manifest=manifest, result=result)
    actions.extend(checks)
    if resolved:
        payload = _audit_payload(manifest=manifest, result=result, status="complete", actions=actions)
    elif tier != tiers[-1]:
        next_tier = tiers[tiers.index(tier) + 1]
        followup = _d16_followup(manifest=manifest, result=result, tier=next_tier,
            bounds=_bounds(manifest), bound_repeat=False)
        actions.append({"action": "escalate", "from": tier, "to": next_tier, "reason": reason})
        payload = _audit_payload(manifest=manifest, result=result, status="needs_escalation", actions=actions,
            reason=reason, followup=followup)
    else:
        payload = _audit_payload(manifest=manifest, result=result, status="failed", actions=actions,
            reason=f"full-grid D16 fallback failed: {reason}")
    write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload)
    return payload


def audit_grid(manifest_path: Path) -> Mapping[str, Any]:
    """Run mandatory numerical checks and emit any required follow-up."""
    manifest = load_manifest(manifest_path)
    result = _reduced(result_path(manifest))
    _completion_receipt(result)
    if (str(manifest.get("membership", {}).get("kind", "")).startswith("d16:")
            and manifest["fit"].get("action") in {"d16_adaptive", "d16_full_grid"}):
        return _audit_d16(manifest_path, manifest, result)
    if result.get("review_only") or result.get("status") != "reduced" or not result.get("selected"):
        raise CampaignError("only a complete definitive reduced grid may be audited")
    if manifest["fit"]["start_mode"] != "grid" or manifest["fit"]["start_count"] != start_count(manifest["fit"]["model"]):
        raise CampaignError("partial grid cannot be scientifically audited")
    _validate_scale_binding(manifest)
    selected = result["selected"]; actions: list[Mapping[str, Any]] = []
    contacts = selected.get("boundary_contacts", [])
    if contacts:
        from .fit import BoundaryContact
        expansion = expand_contacted_bounds(_bounds(manifest), tuple(BoundaryContact(**item) for item in contacts))
        actions.append({"action": "boundary", "contacts": contacts, "nonidentified": list(expansion.nonidentified)})
        if expansion.nonidentified:
            payload = _audit_payload(manifest=manifest, result=result, status="failed", actions=actions,
                reason="non-identification after frozen bound-expansion limit")
        else:
            parents = _parent_paths(manifest); parents["bound_repeat"] = result_path(manifest)
            followup = _make_manifest(receipt={"fit_rows_path": manifest["input"]["fit_rows_path"],
                "fit_rows_content_sha256": manifest["input"]["fit_rows_content_sha256"]}, profile=manifest["execution"],
                phase=f"{manifest['phase']}-bounds-{manifest['fit']['bound_version'] + 1}", model=manifest["fit"]["model"],
                fold_id=manifest["fit"]["fold_id"], objective=manifest["fit"]["objective"], membership=manifest["membership"],
                delta=manifest["fit"]["delta"], parents=parents, bounds=expansion.bounds,
                bound_version=int(manifest["fit"]["bound_version"]) + 1, replicate_id=manifest["fit"]["replicate_id"],
                selection_report=None if manifest["selection_report"] is None
                    else Path(manifest["selection_report"]["path"]))
            payload = _audit_payload(manifest=manifest, result=result, status="needs_bound_repeat", actions=actions,
                followup=followup)
        write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload)
        return payload
    if manifest["fit"]["model"] in {"M1", "M2"}:
        parent = _parent_paths(manifest).get("warm")
        if parent is None:
            raise CampaignError("nested model grid lacks frozen warm parent")
        warm_source = scientific_result(parent)["selected"]
        from .fit import SharedParameters
        warm = nested_warm_start(manifest["fit"]["model"], SharedParameters(**warm_source["shared_parameters"]))
        warm_path, _ = _derived(base=manifest, manifest_path=manifest_path, action="warm", starts=(warm,))
        try:
            warm_records = _records(load_manifest(warm_path))
            warm_record = next((item for item in warm_records if item.get("success")), None)
            if warm_record is None:
                equivalent = False
            else:
                equivalent = equivalence_check(float(selected["objective"]), float(warm_record["objective"]),
                    _record_prediction(selected, manifest=manifest),
                    _record_prediction(warm_record, manifest=load_manifest(warm_path)),
                    objective_tolerance=OBJECTIVE_RELATIVE_TOLERANCE,
                    prediction_tolerance=PREDICTION_ABSOLUTE_TOLERANCE).equivalent
        except (CampaignError, ExecutionError, KeyError, TypeError, ValueError):
            equivalent = False
        actions.append({"action": "warm", "resolved": equivalent})
        if not equivalent:
            payload = _audit_payload(manifest=manifest, result=result, status="failed", actions=actions, reason="nested warm start disagrees")
            write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload); return payload
    grid_records = [record for record in _records(manifest) if record.get("success") and record.get("coordinates")]
    grid_records.sort(key=lambda item: (float(item["objective"]), str(item["start_id"])))
    starts = tuple(StartPoint(str(item["start_id"]), None, np.asarray(item["coordinates"], dtype=np.float64), "basin") for item in grid_records[:3])
    basin_ok = False
    if len(starts) == 3:
        basin_path, _ = _derived(base=manifest, manifest_path=manifest_path, action="basin", starts=starts)
        basin_manifest = load_manifest(basin_path)
        basin_ok = _equivalent_to_retained(_records(basin_manifest), manifest=basin_manifest,
            retained=selected, retained_manifest=manifest)
        actions.append({"action": "basin", "resolved": basin_ok, "manifest": str(basin_path)})
    if not basin_ok:
        center = np.asarray(selected["coordinates"], dtype=np.float64)
        local_starts = (StartPoint("local:center", None, center, "local"),) + deterministic_local_perturbations(center, _bounds(manifest))
        local_path, _ = _derived(base=manifest, manifest_path=manifest_path, action="local", starts=local_starts)
        local_manifest = load_manifest(local_path)
        basin_ok = _equivalent_to_retained(_records(local_manifest), manifest=local_manifest,
            retained=selected, retained_manifest=manifest)
        actions.append({"action": "local", "resolved": basin_ok, "manifest": str(local_path)})
    if not basin_ok:
        payload = _audit_payload(manifest=manifest, result=result, status="failed", actions=actions, reason="best basin unresolved")
        write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload); return payload
    direct_start = StartPoint("direct:best", None, np.asarray(selected["coordinates"], dtype=np.float64), "direct")
    direct_path, direct_result = _derived(base=manifest, manifest_path=manifest_path, action="direct", starts=(direct_start,))
    direct_record = direct_result.get("selected")
    resolved = bool(direct_record and direct_record.get("direct_equivalence", {}).get("resolved"))
    actions.append({"action": "direct", "resolved": resolved, "manifest": str(direct_path)})
    payload = _audit_payload(manifest=manifest, result=result, status="complete" if resolved else "failed", actions=actions,
        reason=None if resolved else "projected/direct equivalence unresolved")
    write_json_once(audit_path(result, output_root=Path(manifest["execution"]["output_root"])), payload)
    return payload
