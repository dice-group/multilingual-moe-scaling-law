"""Auditable orchestration and immutable receipts for K12 selection."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy
import pandas
import scipy

from multilingual_megatron.experiments.scaling.language_composition.selection.common import (
    RECEIPT_SCHEMA,
    SelectionError,
    SelectionInputs,
    canonical_sha256,
    language_set_sha256,
    read_json,
    render_json,
    seal_receipt,
    sha256_file,
    verify_receipt,
    write_immutable,
)
from multilingual_megatron.experiments.scaling.language_composition.selection.distances import DistanceBundle, build_distance_bundle
from multilingual_megatron.experiments.scaling.language_composition.selection.optimization import (
    MIP_GAP_NUMERICAL_TOLERANCE,
    SolverSettings,
    select_language_sets,
)


@dataclass(frozen=True)
class SelectionComputation:
    inputs: SelectionInputs
    distances: DistanceBundle
    selection: dict[str, Any]
    solver_settings: SolverSettings
    scientific_payload: dict[str, Any]
    selection_identity_sha256: str


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SelectionError(f"Repository authority is outside the repository: {path}") from exc


def _recorded_path(raw_path: Any, root: Path) -> Path:
    if not isinstance(raw_path, str):
        raise SelectionError("Recorded artifact path must be a string")
    relative = Path(raw_path)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise SelectionError(f"Unsafe recorded artifact path: {raw_path!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SelectionError(f"Recorded artifact resolves outside the repository: {raw_path}")
    return resolved


def _implementation_provenance(inputs: SelectionInputs) -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    source_hashes = {
        _relative(path, inputs.repo_root): sha256_file(path)
        for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name)
    }
    return {
        "source_sha256": source_hashes,
        "aggregate_sha256": canonical_sha256(source_hashes),
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
        },
    }


def _scientific_payload(
    inputs: SelectionInputs,
    distances: DistanceBundle,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    eligibility = inputs.spec["eligibility"]
    return {
        "candidate_frame": {
            "source_count": len(inputs.source_languages),
            "eligible_count": len(inputs.eligible_languages),
            "eligible_language_set_sha256": language_set_sha256(
                inputs.eligible_languages
            ),
            "availability_projection_sha256": (
                inputs.availability_projection_sha256
            ),
            "excluded_before_metric_construction": eligibility[
                "exclude_before_metric_construction"
            ],
        },
        "distance_contract": {
            "normalization": "each complete K196 matrix divided by its observed maximum",
            "composite": "equal arithmetic mean of four normalized axes",
            "raw_maxima": distances.raw_maxima,
            "matrix_sha256": distances.matrix_sha256,
        },
        "selection_policy": inputs.spec["selection_policy"],
        "promotion_policy": inputs.spec["promotion_policy"],
        "selection": {
            key: value
            for key, value in selection.items()
            if key != "solver_trace"
        },
    }


def compute_selection(
    inputs: SelectionInputs,
    *,
    solver_settings: SolverSettings = SolverSettings(),
) -> SelectionComputation:
    distances = build_distance_bundle(inputs)
    selection = select_language_sets(
        spec=inputs.spec,
        languages=inputs.eligible_languages,
        reference_languages=inputs.reference_languages,
        availability=inputs.availability_tokens,
        bundle=distances,
        settings=solver_settings,
    )
    scientific_payload = _scientific_payload(inputs, distances, selection)
    return SelectionComputation(
        inputs=inputs,
        distances=distances,
        selection=selection,
        solver_settings=solver_settings,
        scientific_payload=scientific_payload,
        selection_identity_sha256=canonical_sha256(scientific_payload),
    )


def _grouping_payload(
    computation: SelectionComputation,
    *,
    arm: str,
    manifest_path: Path,
) -> dict[str, Any]:
    inputs = computation.inputs
    record = computation.selection[arm]
    policy = inputs.spec["selection_policy"]
    return {
        "schema_version": 2,
        "language_set_id": f"scaling_fw2_v2_k012_{arm}",
        "language_set_sha256": record["language_set_sha256"],
        "0": {
            "group": f"k012_{arm}_v2",
            "languages": record["languages"],
            "selection_policy": {
                "definition": (
                    "Exact resource-feasible K12 "
                    f"{'minimum' if arm == 'compact' else 'maximum'} of the "
                    "frozen equal-weight URIEL+ four-axis composite"
                ),
                "method_id": "urielplus_k196_four_axis_minsum_maxsum_v2",
                "candidate_frame": "frozen K196 order derived from scaling_fw2_v2_k200",
                "required_language": policy["required_language"],
                "alpha": policy["alpha"],
                "english_probability_absolute_tolerance": policy[
                    "english_probability_absolute_tolerance"
                ],
                "effective_language_count_absolute_tolerance": policy[
                    "effective_language_count_absolute_tolerance"
                ],
                "selection_identity_sha256": computation.selection_identity_sha256,
                "selection_manifest": _relative(manifest_path, inputs.repo_root),
                "manual_substitutions": "prohibited",
            },
        },
    }


def _base_receipt(computation: SelectionComputation) -> dict[str, Any]:
    inputs = computation.inputs
    selection = computation.selection
    return {
        "schema": RECEIPT_SCHEMA,
        "study_id": inputs.spec["study_id"],
        "status": selection["status"],
        "selection_identity_sha256": computation.selection_identity_sha256,
        **computation.scientific_payload,
        "solver_certificate": {
            "interface": "scipy.optimize.milp",
            "backend": "HiGHS",
            "mip_relative_gap_required": 0.0,
            "observed_gap_numerical_tolerance": MIP_GAP_NUMERICAL_TOLERANCE,
            "independent_arm_solves": True,
            "settings": {
                "arm_workers": computation.solver_settings.arm_workers,
                "solver_threads": computation.solver_settings.solver_threads,
                "show_progress": computation.solver_settings.show_progress,
            },
            "trace": selection["solver_trace"],
        },
        "inputs": {
            "selection_spec": {
                "path": _relative(inputs.spec_path, inputs.repo_root),
                "sha256": inputs.spec_sha256,
            },
            "source_frame": inputs.spec["source_frame"],
            "reference_arm": inputs.spec["reference_arm"],
            "availability_projection": {
                **inputs.spec["availability_projection"],
                "source_sampling_plan_sha256": (
                    inputs.source_sampling_plan_sha256
                ),
            },
            "external_source_lock": inputs.spec["external_source_lock"],
            "external_artifacts": {
                name: {
                    "filename": inputs.spec["external_artifacts"][name]["filename"],
                    "sha256": digest,
                }
                for name, digest in sorted(inputs.artifact_sha256.items())
            },
            "glottocode_mapping_sha256": inputs.glottocode_mapping_sha256,
            "implementation": _implementation_provenance(inputs),
        },
        "runtime_audit": {
            "source_sampling_plan_reverified": (
                inputs.source_sampling_plan_reverified
            )
        },
        "claim_boundary": inputs.spec["claim_boundary"],
    }


def _preflight_artifacts(files: Mapping[Path, bytes]) -> None:
    for path, content in files.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise SelectionError(f"Refusing to overwrite non-identical artifact: {path}")


def select_and_write(
    inputs: SelectionInputs,
    *,
    output_dir: Path,
    grouping_dir: Path,
    solver_settings: SolverSettings = SolverSettings(),
) -> dict[str, Any]:
    """Solve once and publish only a passed, internally sealed result."""

    output_dir = output_dir.resolve()
    grouping_dir = grouping_dir.resolve()
    if not output_dir.is_relative_to(inputs.repo_root) or not grouping_dir.is_relative_to(
        inputs.repo_root
    ):
        raise SelectionError("Selection receipts and groupings must be repository-owned")
    manifest_path = output_dir / "selection_manifest.json"
    computation = compute_selection(inputs, solver_settings=solver_settings)
    receipt = _base_receipt(computation)
    if computation.selection["status"] != "passed":
        receipt["files"] = {}
        sealed = seal_receipt(receipt)
        write_immutable(manifest_path, render_json(sealed))
        return sealed

    grouping_paths = {
        "compact_grouping": grouping_dir / "scaling_fw2_v2_k012_compact.json",
        "diverse_grouping": grouping_dir / "scaling_fw2_v2_k012_diverse.json",
    }
    grouping_content = {
        name: render_json(
            _grouping_payload(
                computation,
                arm=name.removesuffix("_grouping"),
                manifest_path=manifest_path,
            )
        )
        for name in grouping_paths
    }
    receipt["files"] = {
        name: {
            "path": _relative(path, inputs.repo_root),
            "sha256": hashlib.sha256(grouping_content[name]).hexdigest(),
        }
        for name, path in grouping_paths.items()
    }
    sealed = seal_receipt(receipt)
    files = {
        grouping_paths[name]: content for name, content in grouping_content.items()
    }
    files[manifest_path] = render_json(sealed)
    _preflight_artifacts(files)
    for path, content in files.items():
        write_immutable(path, content)
    return sealed


def verify_selection(
    manifest_path: Path,
    inputs: SelectionInputs,
    *,
    solver_settings: SolverSettings = SolverSettings(),
) -> dict[str, Any]:
    """Rebuild matrices, independently re-solve both arms, and verify files."""

    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    verify_receipt(manifest)
    if manifest.get("status") != "passed":
        raise SelectionError("Cannot verify a blocked selection")
    computation = compute_selection(inputs, solver_settings=solver_settings)
    if computation.selection["status"] != "passed":
        raise SelectionError("Current replay fails a promotion gate")
    if computation.selection_identity_sha256 != manifest.get(
        "selection_identity_sha256"
    ):
        raise SelectionError("Independent replay changed the selection identity")
    if computation.scientific_payload != {
        key: manifest[key] for key in computation.scientific_payload
    }:
        raise SelectionError("Independent replay changed the scientific payload")
    current_receipt = _base_receipt(computation)
    for name, current_value in current_receipt["inputs"].items():
        if name == "implementation":
            continue
        if current_value != manifest.get("inputs", {}).get(name):
            raise SelectionError(f"Independent replay changed input authority: {name}")
    if current_receipt["claim_boundary"] != manifest.get("claim_boundary"):
        raise SelectionError("Independent replay changed the claim boundary")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SelectionError("Passed receipt lacks a file ledger")
    for arm in ("compact", "diverse"):
        name = f"{arm}_grouping"
        record = files.get(name)
        if not isinstance(record, dict):
            raise SelectionError(f"Passed receipt lacks {name}")
        path = _recorded_path(record.get("path"), inputs.repo_root)
        expected = render_json(
            _grouping_payload(
                computation,
                arm=arm,
                manifest_path=manifest_path,
            )
        )
        if not path.is_file() or path.read_bytes() != expected:
            raise SelectionError(f"Grouping replay mismatch: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise SelectionError(f"Grouping hash mismatch: {name}")

    current_implementation = current_receipt["inputs"]["implementation"]
    recorded_implementation = manifest["inputs"]["implementation"]
    observed_gaps = [
        trace["mip_gap"] for trace in computation.selection["solver_trace"]
    ]
    return {
        "status": "verified",
        "selection_identity_sha256": computation.selection_identity_sha256,
        "receipt_sha256": manifest["receipt_sha256"],
        "all_five_directional_gates_passed": True,
        "both_mip_gaps_zero": all(gap == 0.0 for gap in observed_gaps),
        "both_mip_gaps_within_numerical_tolerance": all(
            abs(gap) <= MIP_GAP_NUMERICAL_TOLERANCE for gap in observed_gaps
        ),
        "groupings_verified": 2,
        "source_sampling_plan_reverified": (
            inputs.source_sampling_plan_reverified
        ),
        "implementation_matches_receipt": (
            current_implementation == recorded_implementation
        ),
    }
