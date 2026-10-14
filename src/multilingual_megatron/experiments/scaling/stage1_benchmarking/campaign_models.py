"""Resolve Stage-1 benchmark checkpoint cohorts from sealed evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from multilingual_megatron.experiments.scaling.stage1_benchmarking.api import (
    _load_json,
    _sha256_file,
)


def _require_sha(path: Path, expected: Any, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} SHA-256 is invalid")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: expected {expected}, got {actual}")


def _release_table(
    release_manifest: Mapping[str, Any],
    release_dir: Path,
    name: str,
) -> tuple[Path, list[dict[str, Any]]]:
    inventory = release_manifest.get("table_inventory", {}).get(name)
    if not isinstance(inventory, dict):
        raise ValueError(f"release does not declare the {name} table")
    relative_path = inventory.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"release {name} table path is invalid")
    path = (release_dir / relative_path).resolve()
    _require_sha(path, inventory.get("sha256"), label=f"release {name} table")

    import pyarrow.parquet as parquet

    rows = parquet.read_table(path).to_pylist()
    if len(rows) != int(inventory.get("rows", -1)):
        raise ValueError(f"release {name} row-count drift")
    return path, rows


def _positive_ints(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique positive integers")
    return result


def _final_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...],
    token_field: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    finals: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        previous = finals.get(key)
        if previous is None or int(row[token_field]) > int(previous[token_field]):
            finals[key] = row
        elif int(row[token_field]) == int(previous[token_field]):
            raise ValueError(f"duplicate final readout for {key}")
    return finals


def _model_id(
    ladder: str,
    experts: int,
    languages: int,
    seed: int,
    *,
    nominal_tokens: int | None,
) -> str:
    readout = "final" if nominal_tokens is None else f"d{nominal_tokens}"
    return f"{ladder.lower()}_jointmoe_x1_e{experts}_k{languages:03d}_{readout}_s{seed}"


def _declared_schedule(value: Any, *, field: str) -> tuple[int, ...]:
    schedule = _positive_ints(value, field=field)
    if tuple(sorted(schedule)) != schedule:
        raise ValueError(f"{field} must be strictly increasing")
    return schedule


def _load_evaluation_spec(
    evaluation_manifest_path: Path,
    evaluation_manifest_sha256: str,
    *,
    label: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    _require_sha(
        evaluation_manifest_path,
        evaluation_manifest_sha256,
        label=f"{label} evaluation manifest",
    )
    manifest = _load_json(
        evaluation_manifest_path, label=f"{label} evaluation manifest"
    )
    if manifest.get("status") != "complete":
        raise ValueError(f"{label} evaluation manifest is incomplete")
    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"{label} evaluation manifest does not seal its spec")
    spec_path = Path(str(spec.get("path"))).resolve()
    spec_sha256 = str(spec.get("sha256", ""))
    _require_sha(spec_path, spec_sha256, label=f"{label} evaluation spec")
    spec_payload = _load_json(spec_path, label=f"{label} evaluation spec")
    return spec_path, spec_sha256, spec_payload


def _resolved_model(
    *,
    ladder: str,
    experts: int,
    languages: int,
    seed: int,
    dataset_shuffle_seed: int,
    nominal_tokens: int,
    actual_tokens: int,
    checkpoint_step: int,
    trajectory_id: str,
    readout_id: str,
    training_manifest_path: Path,
    training_manifest_sha256: str,
    evaluation_manifest_path: Path,
    evaluation_manifest_sha256: str,
    model_config: Mapping[str, Any],
    source_cohort: str,
    replication_role: str,
    timing_calibration: bool,
    learning_rate_contract: Mapping[str, Any],
    model_id_nominal_tokens: int | None,
    readout_index: int,
    readout_count: int,
) -> dict[str, Any]:
    label = _model_id(
        ladder,
        experts,
        languages,
        seed,
        nominal_tokens=model_id_nominal_tokens,
    )
    training_manifest_path = training_manifest_path.resolve()
    _require_sha(
        training_manifest_path,
        training_manifest_sha256,
        label=f"{label} training manifest",
    )
    training_manifest = _load_json(
        training_manifest_path, label=f"{label} training manifest"
    )
    metadata = training_manifest.get("metadata")
    if training_manifest.get("status") != "complete" or not isinstance(metadata, dict):
        raise ValueError(f"{label} training manifest is incomplete")
    evaluation_spec_path, evaluation_spec_sha256, evaluation_spec = _load_evaluation_spec(
        evaluation_manifest_path.resolve(),
        evaluation_manifest_sha256,
        label=label,
    )

    active_nonembedding = int(model_config["declared_active_nonembedding_parameters"])
    tolerance = float(learning_rate_contract["relative_tolerance"])
    peak_lr = math.exp(
        8.39 - 0.81 * math.log(active_nonembedding) - 0.25 * math.log(experts)
    )
    if tolerance <= 0:
        raise ValueError("learning-rate relative tolerance must be positive")
    model_recipe = evaluation_spec.get("model_recipe", {})
    expected_model = model_recipe.get("expected_model")
    if not isinstance(expected_model, dict):
        raise ValueError(f"{label} evaluation spec has no model contract")
    expert_parallel_size = int(expected_model.get("expert_model_parallel_size") or 1)
    source_overrides = model_recipe.get("overrides")
    if not isinstance(source_overrides, list):
        raise ValueError(f"{label} evaluation spec has no override contract")
    cuda_graph_overrides = [
        item
        for item in source_overrides
        if isinstance(item, str) and item.startswith("model.cuda_graph_scope=")
    ]
    if cuda_graph_overrides not in ([], ["model.cuda_graph_scope=attn"]):
        raise ValueError(
            f"{label} has an unsupported source cuda_graph_scope override: "
            f"{cuda_graph_overrides!r}"
        )

    return {
        "id": label,
        "ladder": ladder,
        "experts": experts,
        "languages": languages,
        "seed": seed,
        "dataset_shuffle_seed": dataset_shuffle_seed,
        "trajectory_id": trajectory_id,
        "readout_id": readout_id,
        "source_cohort": source_cohort,
        "replication_role": replication_role,
        "timing_calibration": timing_calibration,
        "request_set": "english" if languages == 1 else "multilingual",
        "nominal_tokens": nominal_tokens,
        "tokens_seen": actual_tokens,
        "checkpoint_step": checkpoint_step,
        "readout_index": readout_index,
        "readout_count": readout_count,
        "is_final_readout": readout_index == readout_count,
        "run_id": str(training_manifest["run_id"]),
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": training_manifest_sha256,
        "evaluation_manifest": str(evaluation_manifest_path.resolve()),
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "evaluation_spec": str(evaluation_spec_path),
        "evaluation_spec_sha256": evaluation_spec_sha256,
        "micro_batch_size": int(metadata["micro_batch_size"]),
        "expert_parallel_size": expert_parallel_size,
        "moe_dispatcher": expected_model.get("moe_token_dispatcher_type"),
        # Some C-ladder final-loss specs enabled attention CUDA graphs after
        # training. This is a runtime optimization, not learned model state.
        # The score-spec validator proves the saved training config is None
        # before applying this explicit reconciliation.
        "training_model_override_replacements": (
            {"cuda_graph_scope": None} if cuda_graph_overrides else {}
        ),
        "recipe": str(model_config["recipe"]),
        "active_parameters": int(model_config["declared_active_parameters"]),
        "active_nonembedding_parameters": active_nonembedding,
        "configured_total_parameters": int(
            model_config["configured_design_total_parameters"]
        ),
        "peak_lr": peak_lr,
    }


def resolve_checkpoint_models(
    payload: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    release_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve exactly the declared canonical and replicated checkpoint readouts."""

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    checkpoint_policy = selection.get("checkpoint_policy")
    if checkpoint_policy not in {"final_only", "all_declared_readouts"}:
        raise ValueError(
            "selection.checkpoint_policy must be final_only or all_declared_readouts"
        )
    all_readouts = checkpoint_policy == "all_declared_readouts"
    primary = selection.get("canonical_release")
    replicated = selection.get("b_seed_replication")
    calibration = payload.get("runtime_estimation")
    if not isinstance(primary, dict) or not isinstance(replicated, dict):
        raise ValueError("selection must declare canonical and B-seed cohorts")
    if not isinstance(calibration, dict):
        raise ValueError("runtime_estimation must be an object")

    ladders = tuple(str(item) for item in primary.get("ladders", []))
    if ladders != ("A", "B", "C", "D"):
        raise ValueError("canonical benchmark ladders must be A, B, C, D")
    experts = _positive_ints(primary.get("experts"), field="canonical experts")
    languages = _positive_ints(primary.get("languages"), field="canonical languages")
    primary_seed = int(primary.get("seed", 0))
    if experts != (1, 4, 16, 32) or languages != (1, 12, 50, 200):
        raise ValueError("canonical benchmark must retain the complete E x K grid")
    if primary_seed <= 0:
        raise ValueError("canonical seed must be positive")

    declared_primary_schedules: dict[str, tuple[int, ...]] = {}
    if all_readouts:
        raw_schedules = primary.get("nominal_tokens_by_ladder")
        if not isinstance(raw_schedules, dict) or set(raw_schedules) != set(ladders):
            raise ValueError(
                "all-readout canonical selection must declare every ladder schedule"
            )
        declared_primary_schedules = {
            ladder: _declared_schedule(
                raw_schedules[ladder],
                field=f"canonical {ladder} nominal tokens",
            )
            for ladder in ladders
        }

    replicated_experts = _positive_ints(
        replicated.get("experts"), field="B replication experts"
    )
    replicated_languages = _positive_ints(
        replicated.get("languages"), field="B replication languages"
    )
    replicated_seeds = _positive_ints(
        replicated.get("additional_seeds"), field="B replication seeds"
    )
    if replicated.get("ladder") != "B":
        raise ValueError("seed replication is defined only for ladder B")
    if primary_seed in replicated_seeds:
        raise ValueError("additional B seeds must not repeat the canonical seed")
    declared_replicated_schedule: tuple[int, ...] = ()
    if all_readouts:
        declared_replicated_schedule = _declared_schedule(
            replicated.get("nominal_tokens"),
            field="B replication nominal tokens",
        )

    release_dir = release_manifest_path.parent
    table_paths: dict[str, Path] = {}
    table_rows: dict[str, list[dict[str, Any]]] = {}
    for table_name in ("readouts", "trajectories", "model_configs"):
        table_paths[table_name], table_rows[table_name] = _release_table(
            release_manifest, release_dir, table_name
        )

    trajectory_by_id = {
        str(row["trajectory_id"]): row for row in table_rows["trajectories"]
    }
    config_by_cell = {
        (str(row["ladder"]), int(row["experts"])): row
        for row in table_rows["model_configs"]
    }
    release_schedules = {
        ladder: tuple(
            sorted(
                {
                    int(row["nominal_tokens"])
                    for row in table_rows["readouts"]
                    if str(row["ladder"]) == ladder
                }
            )
        )
        for ladder in ladders
    }
    if all_readouts and release_schedules != declared_primary_schedules:
        raise ValueError(
            "declared canonical readout schedules differ from the sealed release"
        )
    if all_readouts:
        selected_primary_rows: Sequence[Mapping[str, Any]] = table_rows["readouts"]
        expected_primary = {
            (ladder, expert_count, language_count, primary_seed, nominal_tokens)
            for ladder in ladders
            for expert_count in experts
            for language_count in languages
            for nominal_tokens in declared_primary_schedules[ladder]
        }
    else:
        selected_primary_rows = list(
            _final_rows(
                table_rows["readouts"],
                key_fields=("trajectory_id",),
                token_field="actual_tokens",
            ).values()
        )
        expected_primary = {
            (ladder, expert_count, language_count, primary_seed)
            for ladder in ladders
            for expert_count in experts
            for language_count in languages
        }
    observed_primary: set[tuple[Any, ...]] = set()
    models: list[dict[str, Any]] = []
    calibration_ladder = str(calibration.get("ladder", ""))
    calibration_languages = {
        int(calibration.get("english_language_count", 0)),
        int(calibration.get("multilingual_language_count", 0)),
    }
    for row in selected_primary_rows:
        trajectory_id = str(row["trajectory_id"])
        trajectory = trajectory_by_id.get(trajectory_id)
        if trajectory is None:
            raise ValueError(f"missing trajectory row for {trajectory_id}")
        design_cell = (
            str(row["ladder"]),
            int(row["experts"]),
            int(row["language_count"]),
            int(trajectory["model_seed"]),
        )
        nominal_tokens = int(row["nominal_tokens"])
        cell = design_cell + ((nominal_tokens,) if all_readouts else ())
        if cell not in expected_primary:
            continue
        observed_primary.add(cell)
        ladder, expert_count, language_count, seed = design_cell
        if str(trajectory["status"]) != "observed_complete":
            raise ValueError(f"canonical trajectory is incomplete: {trajectory_id}")
        if int(trajectory["dataset_shuffle_seed"]) != seed:
            raise ValueError(f"canonical seed mismatch: {trajectory_id}")
        model_config = config_by_cell.get((ladder, expert_count))
        if model_config is None:
            raise ValueError(f"missing model configuration for {ladder}/E{expert_count}")
        schedule = release_schedules[ladder]
        readout_index = schedule.index(nominal_tokens) + 1
        is_final_readout = readout_index == len(schedule)
        models.append(
            _resolved_model(
                ladder=ladder,
                experts=expert_count,
                languages=language_count,
                seed=seed,
                dataset_shuffle_seed=int(trajectory["dataset_shuffle_seed"]),
                nominal_tokens=nominal_tokens,
                actual_tokens=int(row["actual_tokens"]),
                checkpoint_step=int(row["checkpoint_step"]),
                trajectory_id=trajectory_id,
                readout_id=str(row["readout_id"]),
                training_manifest_path=Path(str(row["training_manifest_path"])),
                training_manifest_sha256=str(row["training_manifest_sha256"]),
                evaluation_manifest_path=Path(str(row["evaluation_manifest_path"])),
                evaluation_manifest_sha256=str(row["evaluation_manifest_sha256"]),
                model_config=model_config,
                source_cohort="canonical_release",
                replication_role="primary_release_seed",
                timing_calibration=(
                    ladder == calibration_ladder
                    and language_count in calibration_languages
                    and is_final_readout
                ),
                learning_rate_contract=payload["learning_rate_contract"],
                model_id_nominal_tokens=nominal_tokens if all_readouts else None,
                readout_index=readout_index,
                readout_count=len(schedule),
            )
        )
    if observed_primary != expected_primary:
        missing = sorted(expected_primary - observed_primary)
        raise ValueError(f"canonical final-checkpoint cohort is incomplete: {missing}")

    source_contract = release_manifest.get("source_contract")
    if not isinstance(source_contract, dict):
        raise ValueError("release source contract is missing")
    seed_summary_path = Path(
        str(source_contract.get("seed_confirmation_summary_path", ""))
    ).resolve()
    seed_summary_sha256 = str(
        source_contract.get("seed_confirmation_summary_sha256", "")
    )
    _require_sha(
        seed_summary_path,
        seed_summary_sha256,
        label="B seed-confirmation summary",
    )
    seed_summary = _load_json(seed_summary_path, label="B seed-confirmation summary")
    if (
        seed_summary.get("status") != "complete"
        or int(seed_summary.get("expected_results", -1))
        != int(seed_summary.get("observed_results", -2))
    ):
        raise ValueError("B seed-confirmation summary is incomplete")
    seed_rows = seed_summary.get("results")
    if not isinstance(seed_rows, list) or any(not isinstance(row, dict) for row in seed_rows):
        raise ValueError("B seed-confirmation results are invalid")
    observed_seed_schedule = tuple(
        sorted(
            {
                int(row["nominal_D_tokens"])
                for row in seed_rows
                if str(row["ladder"]) == "B"
                and int(row["experts"]) in replicated_experts
                and int(row["K"]) in replicated_languages
                and int(row["model_seed"]) in replicated_seeds
            }
        )
    )
    if all_readouts and observed_seed_schedule != declared_replicated_schedule:
        raise ValueError(
            "declared B replicated readout schedule differs from sealed seed evidence"
        )
    if all_readouts:
        selected_additional_rows: Sequence[Mapping[str, Any]] = seed_rows
        expected_additional = {
            ("B", expert_count, language_count, seed, nominal_tokens)
            for expert_count in replicated_experts
            for language_count in replicated_languages
            for seed in replicated_seeds
            for nominal_tokens in declared_replicated_schedule
        }
    else:
        selected_additional_rows = list(
            _final_rows(
                seed_rows,
                key_fields=("ladder", "experts", "K", "model_seed"),
                token_field="actual_end_tokens",
            ).values()
        )
        expected_additional = {
            ("B", expert_count, language_count, seed)
            for expert_count in replicated_experts
            for language_count in replicated_languages
            for seed in replicated_seeds
        }
    observed_additional: set[tuple[Any, ...]] = set()
    seed_root = seed_summary_path.parents[1]
    for row in selected_additional_rows:
        design_cell = (
            str(row["ladder"]),
            int(row["experts"]),
            int(row["K"]),
            int(row["model_seed"]),
        )
        nominal_tokens = int(row["nominal_D_tokens"])
        cell = design_cell + ((nominal_tokens,) if all_readouts else ())
        if cell not in expected_additional:
            continue
        observed_additional.add(cell)
        ladder, expert_count, language_count, seed = design_cell
        if row.get("replication_role") != replicated.get("replication_role"):
            raise ValueError(f"B replication-role drift for {cell}")
        training_job_id = str(row["training_job_id"])
        training_result_path = seed_root / "jobs" / training_job_id / "result.json"
        training_result = _load_json(
            training_result_path, label=f"{cell} training result"
        )
        if training_result.get("status") != "complete":
            raise ValueError(f"B replicated training result is incomplete: {cell}")
        training_manifest_path = Path(str(training_result["training_manifest"]))
        training_manifest_sha256 = str(training_result["training_manifest_sha256"])
        evaluation_manifest_path = (
            seed_root / "jobs" / str(row["job_id"]) / "evaluation_manifest.json"
        )
        model_config = config_by_cell.get((ladder, expert_count))
        if model_config is None:
            raise ValueError(f"missing model configuration for {ladder}/E{expert_count}")
        readout_index = observed_seed_schedule.index(nominal_tokens) + 1
        models.append(
            _resolved_model(
                ladder=ladder,
                experts=expert_count,
                languages=language_count,
                seed=seed,
                dataset_shuffle_seed=int(row["dataset_random_seed"]),
                nominal_tokens=nominal_tokens,
                actual_tokens=int(row["actual_end_tokens"]),
                checkpoint_step=int(training_result["final_checkpoint_step"]),
                trajectory_id=f"B:E{expert_count}:K{language_count}:S{seed}",
                readout_id=str(row["job_id"]),
                training_manifest_path=training_manifest_path,
                training_manifest_sha256=training_manifest_sha256,
                evaluation_manifest_path=evaluation_manifest_path,
                evaluation_manifest_sha256=str(row["sha256"]),
                model_config=model_config,
                source_cohort="b_seed_confirmation",
                replication_role=str(row["replication_role"]),
                timing_calibration=False,
                learning_rate_contract=payload["learning_rate_contract"],
                model_id_nominal_tokens=nominal_tokens if all_readouts else None,
                readout_index=readout_index,
                readout_count=len(observed_seed_schedule),
            )
        )
    if observed_additional != expected_additional:
        missing = sorted(expected_additional - observed_additional)
        raise ValueError(f"B replicated final-checkpoint cohort is incomplete: {missing}")

    ladder_order = {ladder: index for index, ladder in enumerate(ladders)}
    models.sort(
        key=lambda model: (
            ladder_order[str(model["ladder"])],
            int(model["experts"]),
            int(model["languages"]),
            int(model["seed"]),
            int(model["nominal_tokens"]),
        )
    )
    model_ids = [str(model["id"]) for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("resolved final-checkpoint model IDs are not unique")

    source_artifacts = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in table_paths.items()
    }
    source_artifacts["seed_confirmation_summary"] = {
        "path": str(seed_summary_path),
        "sha256": seed_summary_sha256,
    }
    return models, source_artifacts


def resolve_final_checkpoint_models(
    payload: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    release_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Backward-compatible final-only resolver entry point."""

    if payload.get("selection", {}).get("checkpoint_policy") != "final_only":
        raise ValueError("resolve_final_checkpoint_models requires final_only selection")
    return resolve_checkpoint_models(payload, release_manifest, release_manifest_path)


__all__ = ["resolve_checkpoint_models", "resolve_final_checkpoint_models"]
