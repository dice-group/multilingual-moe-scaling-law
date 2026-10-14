"""Execution-bound provenance for the Stage 1 A--D analytical release."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    Stage1ReleaseError,
    _canonical_bytes,
    _json_cell,
    _load_json,
    _sha256,
)


def _resolve_input(path_text: str, *, repo_root: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _embedded_manifest_sha256(payload: Mapping[str, Any]) -> str:
    hash_input = dict(payload)
    hash_input.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(hash_input)).hexdigest()


def build_launch_provenance(
    *,
    repo_root: Path,
    primary_roots: Mapping[str, Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Normalize execution snapshots and verify their sealed launch contracts."""

    launches: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    bridge_commits: set[str] = set()
    for family in ("AB", "C", "D"):
        launch_path = (
            primary_roots[family] / "control" / "launch_manifest.json"
        ).resolve()
        launch = _load_json(launch_path)
        embedded = str(launch.get("manifest_sha256", ""))
        recomputed = _embedded_manifest_sha256(launch)
        if embedded != recomputed:
            raise Stage1ReleaseError(
                f"Launch manifest embedded hash mismatch: {launch_path}"
            )
        git = launch.get("git_provenance", {})
        orchestration = (
            git.get("orchestration_repo", {}) if isinstance(git, Mapping) else {}
        )
        bridge = (
            git.get("megatron_bridge_submodule", {}) if isinstance(git, Mapping) else {}
        )
        orchestration_commit = str(orchestration.get("commit", ""))
        bridge_commit = str(bridge.get("commit", ""))
        if len(orchestration_commit) != 40 or len(bridge_commit) != 40:
            raise Stage1ReleaseError(
                f"Incomplete execution commit provenance: {launch_path}"
            )
        bridge_commits.add(bridge_commit)
        fixed_inputs = launch.get("fixed_input_sha256", {})
        if not isinstance(fixed_inputs, Mapping) or not fixed_inputs:
            raise Stage1ReleaseError(
                f"Launch manifest has no fixed inputs: {launch_path}"
            )
        data_materialization = launch.get("data_materialization", {})
        sealed_corpus = (
            data_materialization.get("sealed_corpus", {})
            if isinstance(data_materialization, Mapping)
            else {}
        )
        data_seal_path = _resolve_input(
            str(sealed_corpus.get("data_seal", "")),
            repo_root=repo_root,
        )
        data_seal_expected = str(sealed_corpus.get("data_seal_sha256", ""))
        data_seal_observed = (
            _sha256(data_seal_path) if data_seal_path.is_file() else None
        )
        if data_seal_observed != data_seal_expected:
            raise Stage1ReleaseError(
                f"{family} sealed corpus identity is unavailable or changed: "
                f"{data_seal_path}"
            )
        profile = launch.get("resolved_training_profile_contract", {})
        if (
            not isinstance(profile, Mapping)
            or profile.get("cuda_preflight") is not True
        ):
            raise Stage1ReleaseError(
                f"{family} launch lacks required CUDA preflight contract"
            )
        role_counts = launch.get("role_counts", {})
        launches.append(
            {
                "execution_id": f"stage1_{family.lower()}_production",
                "source_family": family,
                "study_id": launch.get("study_id"),
                "run_root": str(primary_roots[family].resolve()),
                "launch_manifest_path": str(launch_path),
                "launch_manifest_file_sha256": _sha256(launch_path),
                "launch_manifest_payload_sha256": embedded,
                "launch_manifest_hash_verified": True,
                "launch_spec_path": launch.get("spec_path"),
                "runtime_path": launch.get("runtime_path"),
                "orchestration_root_at_execution": orchestration.get("root"),
                "orchestration_commit": orchestration_commit,
                "orchestration_dirty": orchestration.get("dirty"),
                "orchestration_status_sha256": orchestration.get("status_sha256"),
                "orchestration_tracked_diff_sha256": orchestration.get(
                    "tracked_diff_sha256"
                ),
                "bridge_root_at_execution": bridge.get("root"),
                "bridge_commit": bridge_commit,
                "bridge_dirty": bridge.get("dirty"),
                "bridge_status_sha256": bridge.get("status_sha256"),
                "bridge_tracked_diff_sha256": bridge.get("tracked_diff_sha256"),
                "cuda_preflight_required": profile.get("cuda_preflight"),
                "expected_python_executable": profile.get("expected_python_executable"),
                "slurm_export_mode": profile.get("slurm_export_mode"),
                "data_materialization_mode": data_materialization.get("mode"),
                "data_seal_path": str(data_seal_path),
                "data_seal_sha256": data_seal_observed,
                "production_trunks_declared": role_counts.get("production_trunks"),
                "production_branches_declared": role_counts.get("production_branches"),
                "production_evaluations_declared": role_counts.get(
                    "production_evaluations"
                ),
                "git_provenance_json": _json_cell(git),
                "training_profile_contract_json": _json_cell(profile),
                "data_materialization_json": _json_cell(data_materialization),
                "walltime_contract_json": _json_cell(launch.get("walltime_contract")),
                "checkpoint_resume_policy_json": _json_cell(
                    launch.get("checkpoint_resume_policy", {})
                ),
                "provenance_class": "observed",
            }
        )
        for declared_path, expected_hash in sorted(fixed_inputs.items()):
            resolved = _resolve_input(str(declared_path), repo_root=repo_root)
            exists = resolved.is_file()
            observed_hash = _sha256(resolved) if exists else None
            inputs.append(
                {
                    "execution_id": f"stage1_{family.lower()}_production",
                    "source_family": family,
                    "declared_path": str(declared_path),
                    "resolved_path": str(resolved),
                    "expected_sha256": str(expected_hash),
                    "currently_available": exists,
                    "current_sha256": observed_hash,
                    "current_hash_matches_execution": (
                        observed_hash == str(expected_hash)
                    ),
                    "evidence_semantics": (
                        "Expected hash is execution-bound. Current availability "
                        "or equality is a preservation check, not permission to "
                        "substitute a later file."
                    ),
                    "provenance_class": "observed",
                }
            )
        checks.append(
            {
                "check_id": f"launch_manifest_{family.lower()}_seal",
                "observed": embedded,
                "expected": recomputed,
                "status": "pass",
                "severity": "error",
                "details": (
                    f"{family} embedded launch-manifest identity and required "
                    "data seal were independently recomputed."
                ),
                "provenance_class": "derived",
            }
        )
    if len(bridge_commits) != 1:
        raise Stage1ReleaseError(
            f"A-D executions used inconsistent Bridge commits: {bridge_commits}"
        )
    return launches, inputs, checks


def append_analysis_protocol_states(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add the declared confirmatory sequence, including pending stages."""

    result = [dict(row) for row in rows]
    freeze_receipts = [
        row
        for row in result
        if row.get("artifact_role") == "predeclared_freeze_receipt"
        and row.get("exists") is True
        and row.get("sha256")
        and row.get("size_bytes") is not None
    ]
    if len(freeze_receipts) != 1:
        raise Stage1ReleaseError(
            "The frozen A/B analysis sequence requires exactly one hash-sealed "
            f"freeze receipt, found {len(freeze_receipts)}"
        )
    frozen_ab = freeze_receipts[0]
    result.extend(
        [
            {
                "analysis_artifact_id": "frozen_ab_model_selection",
                "artifact_role": "frozen_ab_fit_and_model_selection",
                "path": frozen_ab["path"],
                "exists": True,
                "size_bytes": frozen_ab["size_bytes"],
                "sha256": frozen_ab["sha256"],
                "alias_of_analysis_artifact_id": frozen_ab["analysis_artifact_id"],
                "status": "observed_frozen",
                "scope_disclosure": (
                    "A/B-only fit and frozen M1 selection; historical A/B is "
                    "not C/D evidence."
                ),
                "sequence_order": 1,
                "depends_on": None,
                "provenance_class": "observed",
            },
            {
                "analysis_artifact_id": "frozen_cd_m1_score",
                "artifact_role": "untouched_cd_confirmatory_score",
                "path": None,
                "exists": False,
                "size_bytes": None,
                "sha256": None,
                "status": "pending",
                "scope_disclosure": (
                    "Must score frozen A/B-selected M1 on untouched C/D before "
                    "any A-D refit is interpreted."
                ),
                "sequence_order": 2,
                "depends_on": "frozen_ab_model_selection",
                "provenance_class": "planned",
            },
            {
                "analysis_artifact_id": "declared_ad_refit",
                "artifact_role": "post_confirmatory_all_ladder_refit",
                "path": None,
                "exists": False,
                "size_bytes": None,
                "sha256": None,
                "status": "locked_pending_frozen_cd_score",
                "scope_disclosure": (
                    "A-D refit is permitted only after the frozen C/D score is "
                    "sealed; it must remain separate from confirmatory scoring."
                ),
                "sequence_order": 3,
                "depends_on": "frozen_cd_m1_score",
                "provenance_class": "planned",
            },
        ]
    )
    return result


__all__ = [
    "append_analysis_protocol_states",
    "build_launch_provenance",
]
