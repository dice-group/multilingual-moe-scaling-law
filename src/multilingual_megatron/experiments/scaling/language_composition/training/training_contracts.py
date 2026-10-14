"""Frozen spec, runtime, source-checkpoint, and data-profile contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from multilingual_megatron.experiments.scaling.ab_v2.common import (
    WorkflowError,
    _file_sha256,
    _load_object,
    _payload_sha256,
)
from multilingual_megatron.experiments.scaling.stage1_training.ab_production import validate_production_spec

from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    ARM_GROUPINGS,
    COMPACT_SET_SHA256,
    DESIGN_ID,
    DIVERSE_SET_SHA256,
    EXPERTS,
    HORIZONS,
    REFERENCE_GROUPING,
    REPO_ROOT,
    SEEDS,
    SELECTION_IDENTITY,
    SMOKE_EXPERTS,
    SMOKE_GLOBAL_BATCH_SIZE,
    SMOKE_MAX_STEP_TIME_CV_PCT,
    SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS,
    SMOKE_TOKENS,
    grouping_languages,
    read_object,
    verify_selection_authority,
)
from multilingual_megatron.experiments.scaling.language_composition.data.data_artifacts import (
    ARM_MANIFEST_RELATIVE,
    COHORTS_RELATIVE,
    K196_VALIDATION_RELATIVE,
)
from multilingual_megatron.experiments.scaling.language_composition.training.production_smoke_evidence import validate_evidence_config


DEFAULT_SPEC = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "study_spec_language_composition_v1.json"
)
DEFAULT_RUNTIME = Path(
    "configs/scaling_law/multilingual_moe_ab_v2/"
    "workflow_runtime_language_composition_v1_attempt3.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/pipeline_runs/stage1_language_composition_v1_attempt3"
)
RUNTIME_ATTEMPTS = {
    "stage1_language_composition_v1_attempt2": {
        "runtime_filename": "workflow_runtime_language_composition_v1.json",
        "output_root": "outputs/pipeline_runs/stage1_language_composition_v1_attempt2",
        "workflow_job_prefix": "lcomp1a2",
        "production_authorization_file": (
            "outputs/pipeline_runs/stage1_language_composition_v1_attempt2/control/"
            "production_authorization.json"
        ),
    },
    "stage1_language_composition_v1_attempt3": {
        "runtime_filename": "workflow_runtime_language_composition_v1_attempt3.json",
        "output_root": "outputs/pipeline_runs/stage1_language_composition_v1_attempt3",
        "workflow_job_prefix": "lcomp1a3",
        "production_authorization_file": (
            "outputs/pipeline_runs/stage1_language_composition_v1_attempt3/control/"
            "production_authorization.json"
        ),
    },
    "stage1_language_composition_v1_release1": {
        "runtime_filename": "workflow_runtime_language_composition_v1_release1.json",
        "output_root": "outputs/pipeline_runs/stage1_language_composition_v1_release1",
        "workflow_job_prefix": "lcomp1r1",
        "production_authorization_file": (
            "outputs/pipeline_runs/stage1_language_composition_v1_release1/control/"
            "production_authorization.json"
        ),
    },
}
PARENT_SPEC_SHA256 = (
    "788105aff6f93fa2f0e4a501c864f216c2ac783da2d5e1b0bcbb9fe7a4863585"
)
PARENT_RUNTIME_SHA256 = (
    "bfd62239c6d542f776071a66c8918b836653786377853848ad48020aea021f9c"
)


def _raw_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"{label} must be a JSON object: {path}")
    return payload


def validate_study_spec(
    *, repo_root: Path, spec_path: Path, **_kwargs: Any
) -> dict[str, Any]:
    if repo_root.resolve() != REPO_ROOT:
        raise WorkflowError("language-composition spec resolved under another repository")
    spec_path = spec_path.resolve()
    raw = _raw_object(spec_path, label="language-composition spec")
    if set(raw) != {
        "inherits",
        "status_date",
        "scope",
        "language_composition_design",
    }:
        raise WorkflowError("language-composition spec overlay keys drifted")
    expected_parent = {
        "spec_path": "study_spec_jointmoe_x1_production_v1.json",
        "spec_sha256": PARENT_SPEC_SHA256,
    }
    inherited = raw.get("inherits")
    if isinstance(inherited, dict) and "historical_spec_sha256" in inherited:
        expected_parent["historical_spec_sha256"] = inherited["historical_spec_sha256"]
    if inherited != expected_parent:
        raise WorkflowError("language-composition parent spec binding drifted")
    parent = (spec_path.parent / expected_parent["spec_path"]).resolve()
    if _file_sha256(parent) != PARENT_SPEC_SHA256:
        raise WorkflowError("language-composition parent spec changed")
    validate_production_spec(repo_root=repo_root, spec_path=parent)
    if (
        raw.get("status_date") != "2026-08-31"
        or raw.get("scope")
        != "Predeclared Stage-1 K12 language-composition sensitivity side study"
    ):
        raise WorkflowError("language-composition scope metadata drifted")
    design = raw.get("language_composition_design")
    if not isinstance(design, Mapping):
        raise WorkflowError("language-composition design is missing")
    required_keys = {
        "status",
        "design_id",
        "selection",
        "arms",
        "new_training_arms",
        "experts",
        "seeds",
        "checkpoint_horizons_tokens",
        "trajectory_counts",
        "new_job_counts",
        "data_contract",
        "evaluation_contract",
        "analysis_contract",
        "execution_contract",
        "claim_boundary",
    }
    if set(design) != required_keys:
        raise WorkflowError("language-composition design keys drifted")
    selection = design.get("selection")
    if (
        design.get("status") != "predeclared_before_new_training"
        or design.get("design_id") != DESIGN_ID
        or design.get("arms") != ["reference", "compact", "diverse"]
        or design.get("new_training_arms") != ["compact", "diverse"]
        or tuple(design.get("experts", ())) != EXPERTS
        or tuple(design.get("seeds", ())) != SEEDS
        or tuple(design.get("checkpoint_horizons_tokens", ())) != HORIZONS
        or not isinstance(selection, Mapping)
        or set(selection)
        != {
            "method",
            "selection_identity_sha256",
            "compact_language_set_sha256",
            "diverse_language_set_sha256",
            "sampling_alpha",
            "english_probability_absolute_tolerance",
            "effective_language_count_absolute_tolerance",
            "geography_included",
        }
        or selection.get("method")
        != (
            "equal-weight mean across the frozen typology, genealogy, geography, "
            "and script distance matrices"
        )
        or selection.get("selection_identity_sha256") != SELECTION_IDENTITY
        or selection.get("compact_language_set_sha256") != COMPACT_SET_SHA256
        or selection.get("diverse_language_set_sha256") != DIVERSE_SET_SHA256
        or selection.get("sampling_alpha") != 0.3
        or selection.get("english_probability_absolute_tolerance") != 0.005
        or selection.get("effective_language_count_absolute_tolerance") != 0.25
        or selection.get("geography_included") is not True
    ):
        raise WorkflowError("language-composition scientific axes drifted")
    if design.get("trajectory_counts") != {
        "reused_reference": 9,
        "new_compact": 9,
        "new_diverse": 9,
        "complete": 27,
        "readouts_per_trajectory": 6,
    } or design.get("new_job_counts") != {
        "training_trunks": 18,
        "cooldown_branches": 108,
        "new_arm_evaluations": 108,
        "reference_reevaluations": 54,
        "trajectory_cleanups": 18,
        "production_gpu_jobs": 288,
        "bounded_smoke_gpu_jobs": 4,
    }:
        raise WorkflowError("language-composition job cardinalities drifted")
    if design.get("data_contract") != {
        "release": (
            "one common physical K200 successor release shared by compact and diverse"
        ),
        "physical_capacity": (
            "elementwise maximum across reference, compact, and diverse exposure requirements"
        ),
        "canonical_nested_tiers_unchanged": [1, 12, 50, 200],
        "sampling_policy": (
            "document-level alpha=0.3 sampling with exact largest-remainder 32B "
            "exposure quotas"
        ),
    } or design.get("claim_boundary") != (
        "This is a three-seed B-ladder sensitivity study of the frozen K12 "
        "compositions, not a language-level significance test, a full-grid "
        "invariance claim, or a new LR-selection experiment."
    ):
        raise WorkflowError("language-composition data or claim contract drifted")
    evaluation = design.get("evaluation_contract")
    analysis = design.get("analysis_contract")
    execution = design.get("execution_contract")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("eligible_frame") != "K196"
        or evaluation.get("excluded_languages")
        != ["hyw_Armn", "epo_Latn", "ido_Latn", "srd_Latn"]
        or evaluation.get("primary_cohort") != "unseen_177"
        or evaluation.get("primary_metric")
        != "equal-language mean ATLAS-Tao unigram-adjusted held-out loss"
        or evaluation.get("secondary_cohorts")
        != [
            "reference_only",
            "compact_only",
            "diverse_only",
            "reference_compact",
            "reference_diverse",
            "compact_diverse",
            "all_three",
            "k196",
        ]
        or not isinstance(analysis, Mapping)
        or analysis.get("moe_gain")
        != "loss(E1)-loss(E32) within an arm, seed, and horizon"
        or analysis.get("primary_contrast")
        != "moe_gain(diverse)-moe_gain(compact) at 16B on unseen_177"
        or analysis.get("replication_unit") != "independent training seed"
        or analysis.get("reporting")
        != "three seedwise contrasts plus mean, sample standard deviation, and range"
        or analysis.get("nested_readouts")
        != "descriptive trajectory readouts only"
        or "no significance" not in str(analysis.get("significance_policy", ""))
        or not isinstance(execution, Mapping)
        or execution.get("wave_order")
        != [
            "compact/1234",
            "diverse/1234",
            "compact/5678",
            "diverse/5678",
            "compact/91011",
            "diverse/91011",
        ]
        or execution.get("production_authorization")
        != "explicit hash-bound authorization file required after four real-data smokes"
    ):
        raise WorkflowError("language-composition analysis contract drifted")
    verify_selection_authority()
    resolved = _load_object(spec_path)
    prescription = resolved.get("training_policy", {}).get(
        "learning_rate_prescription", {}
    )
    if (
        prescription.get("formula")
        != "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))"
        or prescription.get("constant_across_K") is not True
        or prescription.get("per_cell_selection_allowed") is not False
    ):
        raise WorkflowError("language-composition parent no longer resolves JointMoE x1")
    return resolved


def validate_runtime(runtime_path: Path, *, require_data: bool = True) -> dict[str, Any]:
    runtime_path = runtime_path.resolve()
    raw = _raw_object(runtime_path, label="language-composition runtime")
    if set(raw) != {
        "inherits",
        "output_root",
        "workflow_attempt_id",
        "workflow_job_prefix",
        "data",
        "performance_budget",
        "storage",
        "language_composition",
    }:
        raise WorkflowError("language-composition runtime overlay keys drifted")
    expected_parent = {
        "spec_path": "workflow_runtime_seed_confirmation_v1.json",
        "spec_sha256": PARENT_RUNTIME_SHA256,
    }
    inherited = raw.get("inherits")
    if isinstance(inherited, dict) and "historical_spec_sha256" in inherited:
        expected_parent["historical_spec_sha256"] = inherited["historical_spec_sha256"]
    if inherited != expected_parent:
        raise WorkflowError("language-composition parent runtime binding drifted")
    parent = (runtime_path.parent / expected_parent["spec_path"]).resolve()
    if _file_sha256(parent) != PARENT_RUNTIME_SHA256:
        raise WorkflowError("language-composition parent runtime changed")
    resolved = _load_object(runtime_path)
    attempt_id = str(resolved.get("workflow_attempt_id", ""))
    attempt = RUNTIME_ATTEMPTS.get(attempt_id)
    if (
        attempt is None
        or runtime_path.name != attempt["runtime_filename"]
        or resolved.get("output_root") != attempt["output_root"]
        or resolved.get("workflow_job_prefix") != attempt["workflow_job_prefix"]
        or resolved.get("gpu_hour_limit") is not None
        or resolved.get("seeds")
        != {
            "data": 42,
            "primary_training": 1234,
            "replication_training": list(SEEDS),
        }
        or tuple(resolved.get("training", {}).get("checkpoint_horizons_tokens", {}).get("B", ()))
        != HORIZONS
    ):
        raise WorkflowError("language-composition inherited runtime axes drifted")
    data = resolved.get("data")
    if not isinstance(data, Mapping) or data.get("mode") != "reuse_sealed_k12":
        raise WorkflowError("language-composition runtime must reuse one sealed corpus")
    sealed_corpus_root = data.get("sealed_corpus_root")
    if not isinstance(sealed_corpus_root, str) or not sealed_corpus_root.strip():
        raise WorkflowError(
            "language-composition runtime must name a sealed corpus root"
        )
    data_root = Path(sealed_corpus_root).resolve()
    if require_data and not (
        (data_root / "control/data_artifact_seal.json").is_file()
        and (data_root / "control/launch_manifest.json").is_file()
        and (data_root / "corpus/tokenized" / K196_VALIDATION_RELATIVE).is_file()
    ):
        raise WorkflowError("language-composition data workflow is not sealed and complete")
    storage = resolved.get("storage")
    if not isinstance(storage, Mapping) or {
        "enforcement": storage.get("enforcement"),
        "retain_all_checkpoints": storage.get("retain_all_checkpoints"),
        "production_launch_policy": storage.get("production_launch_policy"),
        "composition_non_checkpoint_reserve_bytes": storage.get(
            "composition_non_checkpoint_reserve_bytes"
        ),
        "composition_final_free_headroom_bytes": storage.get(
            "composition_final_free_headroom_bytes"
        ),
        "composition_required_available_inodes": storage.get(
            "composition_required_available_inodes"
        ),
    } != {
        "enforcement": "fail_closed_wave_bound",
        "retain_all_checkpoints": False,
        "production_launch_policy": "one_arm_seed_wave_with_three_expert_trajectories",
        "composition_non_checkpoint_reserve_bytes": 256 * 1024**3,
        "composition_final_free_headroom_bytes": 1024**4,
        "composition_required_available_inodes": 200_000,
    }:
        raise WorkflowError("language-composition storage runtime drifted")
    composition = resolved.get("language_composition")
    required_composition_keys = {
        "reference_sources",
        "smoke",
        "production_authorization_file",
    }
    if attempt_id == "stage1_language_composition_v1_attempt3":
        required_composition_keys.add("diagnostic_source_manifest")
    if attempt_id == "stage1_language_composition_v1_release1":
        required_composition_keys.add("production_smoke_evidence")
    if not isinstance(composition, Mapping) or set(composition) != required_composition_keys:
        raise WorkflowError("language-composition runtime contract is incomplete")
    diagnostic_source_manifest = composition.get("diagnostic_source_manifest")
    if attempt_id == "stage1_language_composition_v1_attempt3" and (
        not isinstance(diagnostic_source_manifest, str) or not diagnostic_source_manifest
    ):
        raise WorkflowError("language-composition diagnostic source is not configured")
    production_smoke_evidence = composition.get("production_smoke_evidence")
    if attempt_id == "stage1_language_composition_v1_release1":
        validate_evidence_config(
            production_smoke_evidence, require_files=require_data
        )
    smoke = composition.get("smoke")
    if smoke != {
        "tokens": SMOKE_TOKENS,
        "global_batch_size": SMOKE_GLOBAL_BATCH_SIZE,
        "seed": 1234,
        "experts": list(SMOKE_EXPERTS),
        "steady_state_window_observations": (
            SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS
        ),
        "maximum_step_time_cv_pct": SMOKE_MAX_STEP_TIME_CV_PCT,
        "minimum_reference_throughput_fraction": 0.85,
        "reference_median_tokens_per_second": {
            "1": 329_906.87,
            "32": 203_559.56,
        },
    } or composition.get("production_authorization_file") != attempt[
        "production_authorization_file"
    ]:
        raise WorkflowError("language-composition smoke or authorization runtime drifted")
    references = composition.get("reference_sources")
    if not isinstance(references, Mapping) or set(references) != {
        "primary_seed_root",
        "primary_seed_launch_file_sha256",
        "primary_seed_manifest_sha256",
        "replication_seed_root",
        "replication_seed_launch_file_sha256",
        "replication_seed_manifest_sha256",
    }:
        raise WorkflowError("language-composition reference-source contract drifted")
    return resolved


def _verify_source_job(
    *, root: Path, launch: Mapping[str, Any], job_id: str, seed: int, experts: int, horizon: int
) -> dict[str, Any]:
    result_path = root / "jobs" / job_id / "result.json"
    receipt_path = root / "control" / "receipts" / f"{job_id}.json"
    result = read_object(result_path, label="reference training result")
    receipt = read_object(receipt_path, label="reference training receipt")
    receipt_candidate = dict(receipt)
    embedded_receipt_hash = receipt_candidate.pop("receipt_sha256", None)
    result_hash = _file_sha256(result_path)
    if (
        embedded_receipt_hash != _payload_sha256(receipt_candidate)
        or receipt.get("status") != "complete"
        or receipt.get("job_id") != job_id
        or receipt.get("manifest_sha256") != launch.get("manifest_sha256")
        or result_hash not in set(receipt.get("output_sha256", {}).values())
    ):
        raise WorkflowError(f"reference receipt is not hash-valid for {job_id}")
    arguments = result.get("arguments")
    if (
        result.get("status") != "complete"
        or result.get("job_id") != job_id
        or not isinstance(arguments, Mapping)
        or arguments.get("ladder") != "B"
        or int(arguments.get("experts", 0)) != experts
        or int(arguments.get("K", 0)) != 12
        or int(arguments.get("D", 0)) != horizon
        or int(arguments.get("model_seed", 0)) != seed
        or int(arguments.get("dataset_random_seed", 0)) != seed
        or arguments.get("lr_multiplier") != 1.0
        or arguments.get("role") != "branch"
    ):
        raise WorkflowError(f"reference result contract drifted for {job_id}")
    fixed_files = [result_path.resolve(), receipt_path.resolve()]
    for path_key, hash_key in (
        ("training_manifest", "training_manifest_sha256"),
        ("segment_plan", "segment_plan_sha256"),
    ):
        artifact = Path(str(result.get(path_key, ""))).resolve()
        if not artifact.is_file() or _file_sha256(artifact) != result.get(hash_key):
            raise WorkflowError(f"reference {path_key} changed for {job_id}")
        fixed_files.append(artifact)
    checkpoint_root = Path(str(result.get("final_checkpoint_root", ""))).resolve()
    step = int(result.get("final_checkpoint_step", 0))
    latest = checkpoint_root / "latest_checkpointed_iteration.txt"
    checkpoint = checkpoint_root / f"iter_{step:07d}"
    if (
        step <= 0
        or not latest.is_file()
        or latest.read_text(encoding="utf-8").strip() != str(step)
        or not checkpoint.is_dir()
        or not (checkpoint / "common.pt").is_file()
        or not (checkpoint / "train_state.pt").is_file()
    ):
        raise WorkflowError(f"reference checkpoint is incomplete for {job_id}")
    inventory_paths = [latest, *sorted(checkpoint.rglob("*"))]
    inventory: dict[str, Any] = {}
    total_bytes = 0
    for path in inventory_paths:
        if path.is_symlink():
            raise WorkflowError(f"reference checkpoint contains a symlink: {path}")
        if not path.is_file():
            continue
        stat = path.stat()
        total_bytes += int(stat.st_size)
        inventory[path.relative_to(checkpoint_root).as_posix()] = {
            "size": int(stat.st_size),
            "inode": int(stat.st_ino),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "mode": int(stat.st_mode & 0o7777),
        }
    if len(inventory) < 3:
        raise WorkflowError(f"reference checkpoint inventory is incomplete for {job_id}")
    fixed_files.append(latest.resolve())
    return {
        "source_job_id": job_id,
        "seed": seed,
        "experts": experts,
        "horizon_tokens": horizon,
        "result": {"path": str(result_path.resolve()), "sha256": result_hash},
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": _file_sha256(receipt_path),
            "payload_sha256": embedded_receipt_hash,
        },
        "checkpoint": {
            "root": str(checkpoint_root),
            "step": step,
            "iteration": checkpoint.relative_to(checkpoint_root).as_posix(),
            "file_count": len(inventory),
            "total_bytes": total_bytes,
            "stat_inventory_sha256": _payload_sha256(inventory),
        },
        "fixed_files": [str(path) for path in fixed_files],
    }


def reference_results(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = runtime.get("language_composition", {}).get("reference_sources")
    if not isinstance(source, Mapping):
        raise WorkflowError("language-composition runtime lacks reference sources")
    roots = {
        1234: (
            Path(str(source["primary_seed_root"])).resolve(),
            str(source["primary_seed_launch_file_sha256"]),
            str(source["primary_seed_manifest_sha256"]),
        ),
        5678: (
            Path(str(source["replication_seed_root"])).resolve(),
            str(source["replication_seed_launch_file_sha256"]),
            str(source["replication_seed_manifest_sha256"]),
        ),
        91011: (
            Path(str(source["replication_seed_root"])).resolve(),
            str(source["replication_seed_launch_file_sha256"]),
            str(source["replication_seed_manifest_sha256"]),
        ),
    }
    launch_by_root: dict[Path, dict[str, Any]] = {}
    for root, file_hash, payload_hash in set(roots.values()):
        launch_path = root / "control/launch_manifest.json"
        launch = read_object(launch_path, label="reference launch manifest")
        if (
            _file_sha256(launch_path) != file_hash
            or launch.get("manifest_sha256") != payload_hash
        ):
            raise WorkflowError(f"reference launch identity changed: {root}")
        launch_by_root[root] = launch
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        root = roots[seed][0]
        launch = launch_by_root[root]
        for experts in EXPERTS:
            for horizon in HORIZONS:
                job_id = (
                    f"prod_branch_b_e{experts}_k012_d{horizon}"
                    if seed == 1234
                    else f"seed_branch_b_e{experts}_k012_d{horizon}_s{seed}"
                )
                results.append(
                    _verify_source_job(
                        root=root,
                        launch=launch,
                        job_id=job_id,
                        seed=seed,
                        experts=experts,
                        horizon=horizon,
                    )
                )
    if len(results) != 54:
        raise WorkflowError("reference result inventory is not 54 readouts")
    return results


def arm_languages(arm: str) -> list[str]:
    path = REFERENCE_GROUPING if arm == "reference" else ARM_GROUPINGS.get(arm)
    if path is None:
        raise WorkflowError(f"unknown composition arm {arm!r}")
    return list(grouping_languages((REPO_ROOT / path).resolve(), expected_count=12))


def evaluator_compatibility_k50(
    *, trained_languages: list[str], canonical_k50: list[str]
) -> list[str]:
    """Return a deterministic 50-language superset required by the evaluator.

    Study results are recomputed from all K196 per-language rows.  This cohort
    exists only because the stable evaluator schema requires english/K12/K50
    nesting, while compact contains Danish outside the historical K50.
    """

    if (
        len(trained_languages) != 12
        or len(set(trained_languages)) != 12
        or len(canonical_k50) != 50
        or len(set(canonical_k50)) != 50
        or "eng_Latn" not in trained_languages
    ):
        raise WorkflowError("invalid evaluator compatibility-cohort inputs")
    missing = [
        language for language in trained_languages if language not in canonical_k50
    ]
    removable = [
        language
        for language in reversed(canonical_k50)
        if language not in trained_languages
    ]
    if len(removable) < len(missing):
        raise WorkflowError("cannot construct evaluator K50 compatibility superset")
    removed = set(removable[: len(missing)])
    result = [language for language in canonical_k50 if language not in removed]
    result.extend(missing)
    if len(result) != 50 or not set(trained_languages).issubset(result):
        raise WorkflowError("evaluator K50 compatibility cohort is invalid")
    return result


def profile_transform(
    profile: dict[str, Any], *, paths: Any, arguments: Mapping[str, Any], **_kwargs: Any
) -> dict[str, Any]:
    arm = str(arguments.get("composition_arm", ""))
    if arm not in ARM_MANIFEST_RELATIVE:
        raise WorkflowError("new training job lacks compact/diverse composition_arm")
    manifest = paths.tokenized / ARM_MANIFEST_RELATIVE[arm]
    grouping = (REPO_ROOT / ARM_GROUPINGS[arm]).resolve()
    if not manifest.is_file() or not grouping.is_file():
        raise WorkflowError(f"composition training inputs are missing for {arm}")
    transformed = dict(profile)
    transformed["language_map_path"] = str(grouping)
    transformed["tokenization"] = {
        **dict(transformed["tokenization"]),
        "output_dir": str(manifest.parent),
    }
    training = dict(transformed["training"])
    runs = [dict(value) for value in training["runs"]]
    if len(runs) != 1:
        raise WorkflowError("composition training profile must contain one run")
    runs[0]["data_manifest"] = str(manifest)
    runs[0]["data_manifest_sha256"] = _file_sha256(manifest)
    training["runs"] = runs
    transformed["training"] = training
    return transformed


def evaluation_spec_transform(
    spec: dict[str, Any], *, arguments: Mapping[str, Any], paths: Any, **_kwargs: Any
) -> dict[str, Any]:
    arm = str(arguments.get("composition_arm", ""))
    trained = arm_languages(arm)
    cohorts = read_object(paths.tokenized / COHORTS_RELATIVE, label="composition cohorts")
    required = list(cohorts["cohorts"]["k196"]["languages"])
    validation = paths.tokenized / K196_VALIDATION_RELATIVE
    corpus = paths.release / "corpus_manifest.json"
    if not validation.is_file() or not corpus.is_file():
        raise WorkflowError("composition evaluation inputs are missing")
    transformed = dict(spec)
    transformed["trained_languages"] = trained
    transformed["required_languages"] = required
    # Preserve the evaluator's stable fixed-cohort schema.  Study-specific
    # cohorts are computed from hash-validated per-language rows at summary.
    transformed["cohorts"] = {
        "english": ["eng_Latn"],
        "k12": trained,
        "k50": evaluator_compatibility_k50(
            trained_languages=trained,
            canonical_k50=list(spec["cohorts"]["k50"]),
        ),
    }
    transformed["validation_data_manifest"] = {
        "path": str(validation),
        "sha256": _file_sha256(validation),
    }
    transformed["corpus_manifest"] = {
        "path": str(corpus),
        "sha256": _file_sha256(corpus),
    }
    transformed["language_composition"] = {
        "design_id": DESIGN_ID,
        "arm": arm,
        "eligible_frame": "K196",
        "primary_cohort": "unseen_177",
        "cohort_manifest": {
            "path": str((paths.tokenized / COHORTS_RELATIVE).resolve()),
            "sha256": _file_sha256(paths.tokenized / COHORTS_RELATIVE),
        },
        "evaluator_k50_policy": (
            "schema-compatibility superset only; all reported cohorts are recomputed "
            "from K196 per-language rows"
        ),
    }
    return transformed
