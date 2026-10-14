"""Create each fit wave only after its exact scientific parents pass audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..campaign import _bounds, _delta, _manifest_for_result, scientific_result
from ..data import sha256_file
from ..execution import (
    bind_membership,
    build_manifest,
    load_manifest,
    read_json,
    result_path,
    write_json_once,
    write_manifest,
)
from .contracts import (
    BASELINES,
    FOLDS,
    INDEX_SCHEMA,
    BaselineRefitError,
    index_path,
    job_by_id,
    seal,
    verify_seal,
)


def _receipt(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = read_json(Path(str(workflow["inputs"]["input_receipt"]["path"])))
    needed = {"fit_rows_path", "fit_rows_content_sha256"}
    if not needed <= set(receipt):
        raise BaselineRefitError("input receipt is incomplete")
    holdouts = read_json(Path(str(workflow["inputs"]["holdouts"]["path"])))
    if holdouts.get("input_sha256") != receipt["fit_rows_content_sha256"]:
        raise BaselineRefitError("holdouts and fit rows have different identities")
    return receipt


def _membership(workflow: Mapping[str, Any], fold: str) -> Mapping[str, Any]:
    return bind_membership(
        Path(str(workflow["inputs"]["holdouts"]["path"])),
        kind="primary",
        membership_id=fold,
    )


def _manifest(
    workflow: Mapping[str, Any],
    *,
    phase: str,
    model: str,
    fold: str,
    objective: str,
    delta: float | None = None,
    parents: Mapping[str, Path] | None = None,
    bounds: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    receipt = _receipt(workflow)
    profile = workflow["profile"]
    return build_manifest(
        output_root=Path(str(workflow["output_root"])),
        fit_rows_path=Path(str(receipt["fit_rows_path"])),
        fit_rows_content_sha256=str(receipt["fit_rows_content_sha256"]),
        phase=phase,
        model=model,
        fold_id=fold,
        objective=objective,
        delta=delta,
        chunk_size=int(profile["chunk_size"]),
        parent_results=parents,
        membership=_membership(workflow, fold),
        bounds=bounds,
        blas_threads=int(profile["blas_threads"]),
        partition=profile.get("partition"),
        account=profile.get("account"),
        reservation=profile.get("reservation"),
        memory=str(profile["memory"]),
        time_limit=str(profile["m2_time_limit"] if model == "M2" else profile["time_limit"]),
        concurrency=int(profile["array_concurrency"]),
    )


def build_initial_manifests(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    provisional = {
        "output_root": profile["output_root"],
        "profile": profile,
        "inputs": profile["inputs"],
    }
    return {
        f"grid.m0_ols.{fold}": _manifest(
            provisional,
            phase=f"baseline-refit-m0-ols-{fold}",
            model="M0",
            fold=fold,
            objective="ols",
        )
        for fold in FOLDS
    }


def _index_payload(
    workflow: Mapping[str, Any],
    *,
    name: str,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = {}
    for job_id, manifest in sorted(manifests.items()):
        expected = job_by_id(workflow, job_id)
        if expected["kind"] != "array" or expected["manifest_index"] != name \
                or expected["model"] != manifest["fit"]["model"] \
                or expected["fold"] != manifest["fit"]["fold_id"] \
                or expected["task_count"] != len(manifest["tasks"]):
            raise BaselineRefitError(f"manifest does not match logical job {job_id}")
        path = write_manifest(manifest)
        bindings[job_id] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "manifest_sha256": manifest["manifest_sha256"],
        }
    unsigned = {
        "schema_version": INDEX_SCHEMA,
        "workflow_sha256": workflow["workflow_sha256"],
        "name": name,
        "manifests": bindings,
    }
    return seal(unsigned, "index_sha256")


def write_index(
    workflow: Mapping[str, Any],
    *,
    name: str,
    manifests: Mapping[str, Mapping[str, Any]],
) -> Path:
    payload = _index_payload(workflow, name=name, manifests=manifests)
    path = index_path(workflow, name)
    write_json_once(path, payload)
    return path


def load_index(workflow: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    path = index_path(workflow, name)
    payload = read_json(path)
    verify_seal(payload, "index_sha256", label=f"{name} index")
    if payload.get("schema_version") != INDEX_SCHEMA \
            or payload.get("workflow_sha256") != workflow["workflow_sha256"] \
            or payload.get("name") != name:
        raise BaselineRefitError(f"{name} index belongs to another workflow")
    return payload


def manifest_path_for_job(workflow: Mapping[str, Any], job_id: str) -> Path:
    job = job_by_id(workflow, job_id)
    if job["kind"] != "array" or not job["manifest_index"]:
        raise BaselineRefitError(f"{job_id} is not an array job")
    index = load_index(workflow, str(job["manifest_index"]))
    binding = index.get("manifests", {}).get(job_id)
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "file_sha256", "manifest_sha256",
    }:
        raise BaselineRefitError(f"{job_id} is absent from its manifest index")
    path = Path(str(binding["path"]))
    if not path.is_file() or sha256_file(path) != binding["file_sha256"]:
        raise BaselineRefitError(f"manifest changed for {job_id}")
    manifest = load_manifest(path)
    if manifest["manifest_sha256"] != binding["manifest_sha256"] \
            or manifest["fit"]["model"] != job["model"] \
            or manifest["fit"]["fold_id"] != job["fold"] \
            or len(manifest["tasks"]) != job["task_count"]:
        raise BaselineRefitError(f"manifest identity changed for {job_id}")
    return path


def result_for_job(workflow: Mapping[str, Any], job_id: str) -> Path:
    manifest = load_manifest(manifest_path_for_job(workflow, job_id))
    return result_path(manifest)


def plan_robust(workflow: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    receipt = _receipt(workflow)
    for fold in FOLDS:
        membership = _membership(workflow, fold)
        scale_job = f"grid.m0_ols.{fold}"
        scale_path = result_for_job(workflow, scale_job)
        scale = scientific_result(scale_path)
        objective, delta = _delta(scale, receipt=receipt, membership=membership, fold_id=fold)
        scale_manifest = _manifest_for_result(scale_path, scale)
        for model in BASELINES:
            job_id = f"grid.robust.{model}.{fold}"
            manifests[job_id] = _manifest(
                workflow,
                phase=f"baseline-refit-robust-{model}-{fold}",
                model=model,
                fold=fold,
                objective=objective,
                delta=delta,
                parents={"scale_m0_ols": scale_path},
                bounds=_bounds(scale_manifest) if model == "M0" else None,
            )
    return manifests


def plan_m1(workflow: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    receipt = _receipt(workflow)
    for fold in FOLDS:
        membership = _membership(workflow, fold)
        scale_path = result_for_job(workflow, f"grid.m0_ols.{fold}")
        scale = scientific_result(scale_path)
        objective, delta = _delta(scale, receipt=receipt, membership=membership, fold_id=fold)
        warm_path = result_for_job(workflow, f"grid.robust.M0.{fold}")
        scientific_result(warm_path)
        manifests[f"grid.m1.{fold}"] = _manifest(
            workflow,
            phase=f"baseline-refit-m1-{fold}",
            model="M1",
            fold=fold,
            objective=objective,
            delta=delta,
            parents={"scale_m0_ols": scale_path, "warm": warm_path},
        )
    return manifests


def plan_m2(workflow: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    receipt = _receipt(workflow)
    for fold in FOLDS:
        membership = _membership(workflow, fold)
        scale_path = result_for_job(workflow, f"grid.m0_ols.{fold}")
        scale = scientific_result(scale_path)
        objective, delta = _delta(scale, receipt=receipt, membership=membership, fold_id=fold)
        warm_path = result_for_job(workflow, f"grid.m1.{fold}")
        scientific_result(warm_path)
        manifests[f"grid.m2.{fold}"] = _manifest(
            workflow,
            phase=f"baseline-refit-m2-{fold}",
            model="M2",
            fold=fold,
            objective=objective,
            delta=delta,
            parents={"scale_m0_ols": scale_path, "warm": warm_path},
        )
    return manifests
