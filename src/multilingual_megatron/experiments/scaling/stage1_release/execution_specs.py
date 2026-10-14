"""Resolve the exact study specifications sealed into production launches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    Stage1ReleaseError,
    _json_cell,
    _load_json,
    _sha256,
)
from multilingual_megatron.experiments.scaling.ab_v2.common import _load_object


def _resolve_declared(path_text: str, *, repo_root: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fixed_hash_for_path(
    fixed_inputs: Mapping[str, Any],
    *,
    path: Path,
    repo_root: Path,
) -> str | None:
    matches = {
        str(digest)
        for declared, digest in fixed_inputs.items()
        if _resolve_declared(str(declared), repo_root=repo_root) == path.resolve()
    }
    if len(matches) > 1:
        raise Stage1ReleaseError(f"Conflicting fixed-input hashes for {path}")
    return next(iter(matches), None)


def _verify_inheritance_chain(
    leaf: Path,
    *,
    fixed_inputs: Mapping[str, Any],
    repo_root: Path,
) -> None:
    current = leaf.resolve()
    seen: set[Path] = set()
    while True:
        if current in seen:
            raise Stage1ReleaseError(f"Study-spec inheritance cycle at {current}")
        seen.add(current)
        payload = _load_json(current)
        fixed_hash = _fixed_hash_for_path(
            fixed_inputs, path=current, repo_root=repo_root
        )
        if fixed_hash != _sha256(current):
            raise Stage1ReleaseError(
                f"Execution fixed inputs do not bind study spec {current}"
            )
        inherited = payload.get("inherits")
        if not isinstance(inherited, Mapping):
            return
        parent = (current.parent / str(inherited.get("spec_path", ""))).resolve()
        expected = str(inherited.get("spec_sha256", ""))
        if not parent.is_file() or _sha256(parent) != expected:
            raise Stage1ReleaseError(f"Study-spec parent identity changed: {parent}")
        current = parent


def build_execution_spec_rows(
    *,
    repo_root: Path,
    primary_roots: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load specs by launch-manifest hash, not by mutable current config path.

    Overlay inheritance is resolved only after every leaf and parent file is
    matched to the fixed-input hashes sealed into the production launch.
    """

    rows: list[dict[str, Any]] = []
    ladder_specs: dict[str, dict[str, Any]] = {}
    family_ladders = {"AB": ("A", "B"), "C": ("C",), "D": ("D",)}
    for family, ladders in family_ladders.items():
        launch_path = (
            primary_roots[family] / "control" / "launch_manifest.json"
        ).resolve()
        launch = _load_json(launch_path)
        declared_path = _resolve_declared(
            str(launch.get("spec_path", "")), repo_root=repo_root
        )
        declared_hashes = launch.get("fixed_input_sha256", {})
        if not isinstance(declared_hashes, Mapping):
            raise Stage1ReleaseError(
                f"Launch manifest has malformed fixed inputs: {launch_path}"
            )
        expected_hash = _fixed_hash_for_path(
            declared_hashes,
            path=declared_path,
            repo_root=repo_root,
        )
        if (
            expected_hash is None
            or not declared_path.is_file()
            or _sha256(declared_path) != expected_hash
        ):
            raise Stage1ReleaseError(
                f"No available {family} leaf study spec matches its production launch"
            )
        _verify_inheritance_chain(
            declared_path,
            fixed_inputs=declared_hashes,
            repo_root=repo_root,
        )
        spec_path = declared_path
        spec = _load_object(spec_path)
        declared_ladders = {
            str(item["id"]): dict(item) for item in spec["axes"]["ladders"]
        }
        if set(declared_ladders) != set(ladders):
            raise Stage1ReleaseError(
                f"{family} execution spec ladders differ: "
                f"{sorted(declared_ladders)} != {list(ladders)}"
            )
        for ladder in ladders:
            ladder_specs[ladder] = {
                "family": family,
                "path": spec_path,
                "spec": spec,
                "ladder": declared_ladders[ladder],
            }
            training_policy = spec.get("training_policy", {})
            prescription = training_policy.get("learning_rate_prescription")
            rows.append(
                {
                    "protocol_id": (
                        "stage1_ab_v2"
                        if family == "AB"
                        else f"stage1_{family.lower()}_v1"
                    ),
                    "study_id": spec["study_id"],
                    "ladder": ladder,
                    "spec_path": str(spec_path),
                    "spec_sha256": expected_hash,
                    "launch_manifest_path": str(launch_path),
                    "launch_manifest_sha256": _sha256(launch_path),
                    "launch_declared_spec_path": str(declared_path),
                    "execution_authority": "production_launch_manifest",
                    "status": "observed_complete",
                    "model_seed": training_policy.get("model_seed", 1234),
                    "dataset_shuffle_seed": training_policy.get(
                        "dataset_shuffle_seed", 1234
                    ),
                    "peak_learning_rate": training_policy.get("peak_learning_rate"),
                    "learning_rate_mode": (
                        "prescribed_per_model_configuration"
                        if isinstance(prescription, Mapping)
                        else "fixed_global"
                    ),
                    "learning_rate_prescription_json": (
                        _json_cell(prescription)
                        if isinstance(prescription, Mapping)
                        else None
                    ),
                    "warmup_tokens": training_policy.get("warmup_tokens"),
                    "schedule": training_policy.get("schedule"),
                    "provenance_class": "configured",
                }
            )
    if set(ladder_specs) != {"A", "B", "C", "D"}:
        raise Stage1ReleaseError(
            f"Unexpected execution ladder specs: {sorted(ladder_specs)}"
        )
    return rows, ladder_specs


__all__ = ["build_execution_spec_rows"]
