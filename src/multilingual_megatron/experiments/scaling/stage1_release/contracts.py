#!/usr/bin/env python3
"""Build the publication-grade Stage 1 A--D analytical release.

Raw training/evaluation artifacts remain immutable.  This builder creates a
versioned, hash-sealed derived release with typed Parquet tables, a compact
Excel workbook, exact source lineage, a raw Slurm-accounting receipt, and
explicit NeurIPS-style completeness disclosures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "multilingual_megatron_stage1_abcd_release/v1"
RELEASE_NAME = "stage1_abcd_v1"
BUILD_SPEC_SCHEMA = "multilingual_megatron_stage1_release_build_spec/v1"
PROVENANCE_CLASSES = ("observed", "configured", "planned", "derived")
EXPECTED_COUNTS = {
    "trajectories": 64,
    "readouts": 400,
    "validation_language": 46_200,
    "trained_language": 26_300,
    "zero_shot_language": 19_900,
}
LADDER_COUNTS = {"A": 80, "B": 96, "C": 112, "D": 112}
PRIMARY_ROOT_NAMES = {
    "AB": "h100_ab_v2_k12_fixed_lr_optimized_v3",
    "C": "h100_c_v1_full_study_phase1_20260723",
    "D": "h100_d_v1_full_study_phase1_20260723",
}
SPEC_PATHS = {
    "AB": Path("configs/scaling_law/multilingual_moe_ab_v2/study_spec.json"),
    "C": Path("configs/scaling_law/multilingual_moe_c_v1/study_spec.json"),
    "D": Path("configs/scaling_law/multilingual_moe_d_v1/study_spec.json"),
}
READOUT_ID = re.compile(
    r"^prod_eval_([abcd])_e(1|4|16|32)_k(001|012|050|200)_d([1-9][0-9]*)$"
)


@dataclass(frozen=True)
class ReleaseBuildSpec:
    """Versioned source binding for one immutable Stage-1 release build."""

    release_name: str
    primary_root_names: Mapping[str, str]
    corpus_manifest: Path
    cd_source: str
    cd_ledger: Path | None
    analysis_registry_paths: tuple[Path, ...]
    seed_confirmation_summary: Path | None
    expected_validation_candidate_count: int
    expected_duplicate_candidate_count: int
    scientific_scope: str


def _relative_path(value: Any, *, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise Stage1ReleaseError(f"{field} must be a non-empty repository-relative path")
    return path


def historical_release_build_spec() -> ReleaseBuildSpec:
    """Return the source contract used by the sealed historical release."""

    return ReleaseBuildSpec(
        release_name=RELEASE_NAME,
        primary_root_names=dict(PRIMARY_ROOT_NAMES),
        corpus_manifest=Path(
            "outputs/pipeline_runs/h100_ab_v2_k12_fixed_lr_attempt3/"
            "corpus/release_32b/corpus_manifest.json"
        ),
        cd_source="cross_root_ledger",
        cd_ledger=Path("result_analysis/stage1_cd_analysis/cross_root_result_ledger.json"),
        analysis_registry_paths=(
            Path("result_analysis/stage1_cd_20260723/freeze_receipt.json"),
            Path("result_analysis/stage1_cd_20260723/simulation_receipt.json"),
            Path("result_analysis/stage1_cd_20260723/protocol_amendment.json"),
            Path("result_analysis/stage1_cd_20260723/frozen_launched_d_study_spec.json"),
            Path("result_analysis/stage1_cd_20260723/analysis_first_pause_cancellation_manifest_20260730.json"),
            Path("result_analysis/stage1_cd_20260723/analysis_first_pause_cancellation_receipt_20260730.json"),
            Path("result_analysis/stage1_cd_analysis/analysis_summary.json"),
            Path("result_analysis/stage1_cd_analysis/cross_root_result_ledger.json"),
            Path("result_analysis/stage1_cd_result_analysis.ipynb"),
            Path("result_analysis/fit_input_validation_loss_joined.csv"),
        ),
        seed_confirmation_summary=None,
        expected_validation_candidate_count=435,
        expected_duplicate_candidate_count=35,
        scientific_scope="Historical fixed-learning-rate Stage-1 A-D release.",
    )


def load_release_build_spec(path: Path) -> ReleaseBuildSpec:
    """Load a strict build specification without interpreting source outcomes."""

    payload = _load_json(path.resolve())
    expected_keys = {
        "schema",
        "release_name",
        "primary_root_names",
        "corpus_manifest",
        "cd_source",
        "cd_ledger",
        "analysis_registry_paths",
        "seed_confirmation_summary",
        "expected_validation_candidate_count",
        "expected_duplicate_candidate_count",
        "scientific_scope",
    }
    if set(payload) != expected_keys or payload.get("schema") != BUILD_SPEC_SCHEMA:
        raise Stage1ReleaseError("release build-spec schema or fields changed")
    release_name = str(payload["release_name"])
    if not re.fullmatch(r"stage1_[a-z0-9_]+_v[0-9]+", release_name):
        raise Stage1ReleaseError(f"invalid release name: {release_name!r}")
    roots = payload["primary_root_names"]
    if not isinstance(roots, Mapping) or set(roots) != {"AB", "C", "D"}:
        raise Stage1ReleaseError("primary_root_names must contain exactly AB, C, and D")
    root_names = {str(key): str(value) for key, value in roots.items()}
    if any(Path(value).name != value or value in {"", ".", ".."} for value in root_names.values()):
        raise Stage1ReleaseError("primary root values must be pipeline-run directory names")
    cd_source = str(payload["cd_source"])
    if cd_source not in {"cross_root_ledger", "direct_final_summaries"}:
        raise Stage1ReleaseError(f"unsupported C/D source mode: {cd_source}")
    ledger_raw = payload["cd_ledger"]
    ledger = None if ledger_raw is None else _relative_path(ledger_raw, field="cd_ledger")
    if (cd_source == "cross_root_ledger") != (ledger is not None):
        raise Stage1ReleaseError("cd_ledger must be present only for cross_root_ledger mode")
    registry = payload["analysis_registry_paths"]
    if not isinstance(registry, list):
        raise Stage1ReleaseError("analysis_registry_paths must be a list")
    seed_raw = payload["seed_confirmation_summary"]
    seed = None if seed_raw is None else _relative_path(
        seed_raw, field="seed_confirmation_summary"
    )
    scope = str(payload["scientific_scope"]).strip()
    if not scope:
        raise Stage1ReleaseError("scientific_scope must be non-empty")
    expected_candidates = int(payload["expected_validation_candidate_count"])
    expected_duplicates = int(payload["expected_duplicate_candidate_count"])
    if expected_candidates <= 0:
        raise Stage1ReleaseError("expected_validation_candidate_count must be positive")
    if not 0 <= expected_duplicates < expected_candidates:
        raise Stage1ReleaseError(
            "expected_duplicate_candidate_count must be non-negative and smaller "
            "than expected_validation_candidate_count"
        )
    return ReleaseBuildSpec(
        release_name=release_name,
        primary_root_names=root_names,
        corpus_manifest=_relative_path(payload["corpus_manifest"], field="corpus_manifest"),
        cd_source=cd_source,
        cd_ledger=ledger,
        analysis_registry_paths=tuple(
            _relative_path(value, field="analysis_registry_paths") for value in registry
        ),
        seed_confirmation_summary=seed,
        expected_validation_candidate_count=expected_candidates,
        expected_duplicate_candidate_count=expected_duplicates,
        scientific_scope=scope,
    )


class Stage1ReleaseError(RuntimeError):
    """Raised when the A--D release cannot be built without ambiguity."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ReleaseError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage1ReleaseError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_cell(value: Any) -> str:
    return _canonical_bytes(value).decode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _portable(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _require_analysis_stack() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise Stage1ReleaseError(
            "The release builder requires pandas and pyarrow; run it in the "
            "study analysis environment."
        ) from exc
    return pd, pyarrow


def _primary_roots(
    root: Path,
    root_names: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    pipeline = root / "outputs" / "pipeline_runs"
    names = PRIMARY_ROOT_NAMES if root_names is None else root_names
    result = {
        key: (pipeline / name).resolve() for key, name in names.items()
    }
    missing = [str(path) for path in result.values() if not path.is_dir()]
    if missing:
        raise Stage1ReleaseError("Missing primary result roots:\n" + "\n".join(missing))
    return result


def _source_registry(
    *,
    root: Path,
    primary_roots: Mapping[str, Path],
    ledger_path: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    ledger = _load_json(ledger_path)
    run_roots = {path.resolve() for path in primary_roots.values()}
    for record in ledger.get("records", []):
        for candidate in record.get("candidates", []):
            path = Path(str(candidate.get("path", ""))).resolve()
            parts = path.parts
            if "pipeline_runs" not in parts:
                continue
            index = parts.index("pipeline_runs")
            if index + 1 < len(parts):
                run_roots.add(Path(*parts[: index + 2]))
    rows: list[dict[str, Any]] = []
    for path in sorted(run_roots):
        rows.append(
            {
                "source_root_id": path.name,
                "path": str(path),
                "portable_path": _portable(path, root),
                "role": (
                    "primary_ab"
                    if path == primary_roots["AB"]
                    else "primary_c"
                    if path == primary_roots["C"]
                    else "primary_d"
                    if path == primary_roots["D"]
                    else "validation_recovery_or_offload"
                ),
                "exists": path.is_dir(),
                "provenance_class": "observed",
            }
        )
    return rows, sorted(run_roots)


def _spec_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ladder_specs: dict[str, dict[str, Any]] = {}
    for family, relative in SPEC_PATHS.items():
        path = (root / relative).resolve()
        spec = _load_json(path)
        for ladder in spec["axes"]["ladders"]:
            ladder_id = str(ladder["id"])
            ladder_specs[ladder_id] = {
                "family": family,
                "path": path,
                "spec": spec,
                "ladder": dict(ladder),
            }
            rows.append(
                {
                    "protocol_id": "stage1_ab_v2"
                    if family == "AB"
                    else f"stage1_{family.lower()}_v1",
                    "study_id": spec["study_id"],
                    "ladder": ladder_id,
                    "spec_path": str(path),
                    "spec_sha256": _sha256(path),
                    "status": "observed_complete",
                    "model_seed": spec.get("training_policy", {}).get(
                        "model_seed", 1234
                    ),
                    "dataset_shuffle_seed": spec.get("training_policy", {}).get(
                        "dataset_shuffle_seed", 1234
                    ),
                    "peak_learning_rate": spec.get("training_policy", {}).get(
                        "peak_learning_rate"
                    ),
                    "warmup_tokens": spec.get("training_policy", {}).get(
                        "warmup_tokens"
                    ),
                    "schedule": spec.get("training_policy", {}).get("schedule"),
                    "provenance_class": "configured",
                }
            )
    if set(ladder_specs) != {"A", "B", "C", "D"}:
        raise Stage1ReleaseError(f"Unexpected ladder specs: {sorted(ladder_specs)}")
    return rows, ladder_specs


def _readout_eval_model(readout: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(readout["evaluation_manifest_path"]))
    spec_path = manifest_path.parent / "evaluation_spec.json"
    spec = _load_json(spec_path)
    recipe = spec.get("model_recipe", {})
    expected = recipe.get("expected_model", {})
    if not isinstance(expected, Mapping):
        expected = {}
    return {
        "recipe": recipe.get("name"),
        "expected": dict(expected),
        "overrides": list(recipe.get("overrides", [])),
        "evaluation_spec_path": str(spec_path),
        "evaluation_spec_sha256": _sha256(spec_path),
        "sequence_length": spec.get("sequence_length"),
    }


def _model_and_trajectory_rows(
    *,
    readouts: Sequence[Mapping[str, Any]],
    ladder_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    representatives: dict[tuple[str, int], Mapping[str, Any]] = {}
    for readout in readouts:
        key = (str(readout["ladder"]), int(readout["experts"]))
        representatives.setdefault(key, readout)
    model_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for ladder in ("A", "B", "C", "D"):
        context = ladder_specs[ladder]
        declared = context["ladder"]
        spec = context["spec"]
        for experts in (1, 4, 16, 32):
            model = _readout_eval_model(representatives[(ladder, experts)])
            expected = model["expected"]
            model_id = f"{ladder}:E{experts}"
            model_rows.append(
                {
                    "model_config_id": model_id,
                    "ladder": ladder,
                    "experts": experts,
                    "architecture_type": "dense_control"
                    if experts == 1
                    else "sparse_moe",
                    "recipe": model["recipe"],
                    "num_layers": declared["num_layers"],
                    "hidden_size": declared["hidden_size"],
                    "ffn_hidden_size": declared["ffn_hidden_size"],
                    "num_attention_heads": declared["num_attention_heads"],
                    "sequence_length": model["sequence_length"]
                    or expected.get("seq_length"),
                    "vocab_size": expected.get("vocab_size"),
                    "num_moe_experts_runtime": expected.get("num_moe_experts"),
                    "moe_router_topk": expected.get("moe_router_topk"),
                    "moe_dispatcher": expected.get("moe_token_dispatcher_type"),
                    "moe_capacity_factor": expected.get("moe_expert_capacity_factor"),
                    "dropless": expected.get("moe_expert_capacity_factor") is None,
                    "declared_active_parameters": declared["N_act"],
                    "declared_active_nonembedding_parameters": declared.get(
                        "N_act_nonembed"
                    ),
                    "runtime_total_parameters": None,
                    "runtime_total_parameters_status": (
                        "to_be_derived_from_training_log_events"
                    ),
                    "evaluation_spec_path": model["evaluation_spec_path"],
                    "evaluation_spec_sha256": model["evaluation_spec_sha256"],
                    "resolved_overrides_json": _json_cell(model["overrides"]),
                    "provenance_class": "configured",
                }
            )
            for language_count in (1, 12, 50, 200):
                trajectory_rows.append(
                    {
                        "trajectory_id": f"{ladder}:E{experts}:K{language_count}",
                        "protocol_id": (
                            "stage1_ab_v2"
                            if context["family"] == "AB"
                            else f"stage1_{ladder.lower()}_v1"
                        ),
                        "ladder": ladder,
                        "experts": experts,
                        "language_count": language_count,
                        "model_config_id": model_id,
                        "model_seed": spec.get("training_policy", {}).get(
                            "model_seed", 1234
                        ),
                        "dataset_shuffle_seed": spec.get("training_policy", {}).get(
                            "dataset_shuffle_seed", 1234
                        ),
                        "replicate_id": 1,
                        "independent_seed_replicates": 1,
                        "replication_disclosure": (
                            "One configured seed; checkpoint and language rows "
                            "are repeated measures, not independent replicates."
                        ),
                        "status": "observed_complete",
                        "provenance_class": "configured",
                    }
                )
    return model_rows, trajectory_rows


def _production_jobs(
    primary_roots: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    validity: list[dict[str, Any]] = []
    for family, run_root in primary_roots.items():
        manifest_path = run_root / "control" / "launch_manifest.json"
        manifest = _load_json(manifest_path)
        for job in manifest.get("jobs", []):
            arguments = job.get("arguments", {})
            if job.get("action") != "production_train":
                continue
            logical_id = str(job["job_id"])
            result_path = run_root / "jobs" / logical_id / "result.json"
            if not result_path.is_file():
                validity.append(
                    {
                        "object_type": "training_job",
                        "object_id": logical_id,
                        "issue_type": "missing_result",
                        "severity": "error",
                        "verdict": "invalid",
                        "evidence_path": str(result_path),
                        "provenance_class": "observed",
                    }
                )
                continue
            result = _load_json(result_path)
            ladder = str(arguments["ladder"])
            experts = int(arguments["experts"])
            language_count = int(arguments["K"])
            role = str(arguments.get("role", result.get("role", "")))
            jobs.append(
                {
                    "training_job_id": logical_id,
                    "trajectory_id": f"{ladder}:E{experts}:K{language_count}",
                    "run_root": str(run_root),
                    "source_family": family,
                    "ladder": ladder,
                    "experts": experts,
                    "language_count": language_count,
                    "job_role": role,
                    "parent_training_job_id": arguments.get("parent_job_id"),
                    "target_tokens": int(arguments.get("D", 0)),
                    "actual_end_tokens": int(result.get("actual_end_tokens", 0)),
                    "expected_consumed_samples": int(
                        result.get("expected_consumed_samples", 0)
                    ),
                    "end_to_end_seconds": float(result.get("end_to_end_seconds", 0.0)),
                    "status": result.get("status"),
                    "result_path": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "training_manifest_path": str(result.get("training_manifest", "")),
                    "training_manifest_sha256": result.get("training_manifest_sha256"),
                    "final_checkpoint_root": result.get("final_checkpoint_root"),
                    "final_checkpoint_step": result.get("final_checkpoint_step"),
                    "peak_learning_rate": result.get("peak_lr"),
                    "resource": job.get("resource"),
                    "predicted_wall_hours": job.get("predicted_wall_hours"),
                    "planned_gpu_tokens": job.get("gpu_tokens"),
                    "provenance_class": "observed",
                }
            )
    jobs.sort(key=lambda row: row["training_job_id"])
    expected = 208 + 128 + 128
    if len(jobs) != expected:
        raise Stage1ReleaseError(
            f"Expected {expected} production training jobs, got {len(jobs)}"
        )
    return jobs, validity
