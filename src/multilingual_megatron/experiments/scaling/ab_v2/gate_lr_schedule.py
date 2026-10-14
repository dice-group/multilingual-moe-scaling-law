"""Verify the pre-production learning-rate warmup schedule probes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .artifacts import _verified_receipt_evidence
from .common import (
    FIXED_LR_SCHEDULE_PROBE_TOKENS,
    FIXED_LR_SCHEDULE_WARMUP_STEPS,
    FIXED_LR_SCHEDULE_WARMUP_TOKENS,
    FIXED_PEAK_LEARNING_RATE,
    PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS,
    PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS,
    PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS,
    PRIMARY_TRAINING_SEED,
    Paths,
    WorkflowError,
    _atomic_json,
    _file_sha256,
    _job,
    _load_object,
    _payload_sha256,
    _workflow_profile_name,
)
from .estimates import _batch_layout
from .jobs import _training_contract_arguments
from .training_health import _parse_sentinel_health
from .training_profile import _peak_lr_for_job
from .variant import active_policy


def _run_lr_schedule_gate(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    paths: Paths,
    *,
    prescribed: bool,
) -> None:
    """Gate four architecture/K extreme schedule probes.

    These runs are deliberately not fit observations or LR selection inputs.
    They verify the selected LR policy and exact early production warmup schedule
    on both K extremes before production is allowed to start. The false mode
    preserves execution of the immutable historical fixed-LR manifests.
    """

    from multilingual_megatron.pipeline.stages.training.checkpoint_state import (
        inspect_full_state_checkpoint,
    )

    runtime = _load_object(Path(str(manifest["runtime_path"])))
    expected_cells = set(
        active_policy().prescribed_lr_schedule_probe_cells
        if prescribed
        else active_policy().fixed_lr_schedule_probe_cells
    )
    probe_tokens = (
        PRESCRIBED_LR_SCHEDULE_PROBE_TOKENS
        if prescribed
        else FIXED_LR_SCHEDULE_PROBE_TOKENS
    )
    warmup_tokens = (
        PRESCRIBED_LR_SCHEDULE_WARMUP_TOKENS
        if prescribed
        else FIXED_LR_SCHEDULE_WARMUP_TOKENS
    )
    warmup_steps = (
        PRESCRIBED_LR_SCHEDULE_WARMUP_STEPS
        if prescribed
        else FIXED_LR_SCHEDULE_WARMUP_STEPS
    )
    result_role = (
        "prescribed_lr_schedule_verification"
        if prescribed
        else "fixed_lr_schedule_verification"
    )
    observed_cells: set[tuple[str, int, int]] = set()
    errors: list[str] = []
    results: dict[str, Any] = {}
    for source_job_id in job["arguments"]["probe_jobs"]:
        source = _job(manifest, source_job_id)
        args = source.get("arguments", {})
        cell = (
            str(args.get("ladder", "")),
            int(args.get("experts", 0)),
            int(args.get("K", 0)),
        )
        observed_cells.add(cell)
        path = paths.jobs / source_job_id / "result.json"
        payload = _load_object(path)
        item: dict[str, Any] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "status": payload.get("status"),
            "cell": list(cell),
        }
        results[source_job_id] = item
        try:
            _verified_receipt_evidence(manifest, source_job_id)
        except WorkflowError as exc:
            errors.append(f"{source_job_id} receipt: {exc}")
        if payload.get("status") != "complete" or payload.get("arguments") != args:
            errors.append(f"{source_job_id} is not a complete exact frozen probe")
        try:
            expected_mbs, expected_dp, expected_ga = _batch_layout(
                runtime,
                ladder=cell[0],
                experts=cell[1],
                global_batch_size=32,
            )
        except WorkflowError as exc:
            errors.append(f"{source_job_id} batch layout: {exc}")
            expected_mbs, expected_dp, expected_ga = -1, -1, -1
        lr_args = (
            _training_contract_arguments(
                manifest["learning_rate_prescription"],
                ladder=cell[0],
                experts=cell[1],
                training_seed=PRIMARY_TRAINING_SEED,
            )
            if prescribed
            else {
                "peak_lr": FIXED_PEAK_LEARNING_RATE,
                "learning_rate_policy": "fixed_across_all_cells",
            }
        )
        expected_args = {
            **lr_args,
            "tokens": probe_tokens,
            "fit_observation": False,
            "constant_after_warmup": True,
            "warmup_tokens": warmup_tokens,
            "warmup_steps": warmup_steps,
            "total_steps": 2_560,
            "global_batch_size": 32,
            "micro_batch_size": expected_mbs,
            "data_parallel_size": expected_dp,
            "gradient_accumulation_steps": expected_ga,
            "data_tier": f"K{cell[2]}",
        }
        for key, expected in expected_args.items():
            if args.get(key) != expected:
                errors.append(
                    f"{source_job_id} {key} differs from stress protocol: "
                    f"{args.get(key)!r} != {expected!r}"
                )
        if (
            payload.get("fit_observation") is not False
            or payload.get("role") != result_role
            or payload.get("constant_after_warmup") is not True
            or payload.get("warmup_steps") != warmup_steps
            or payload.get("warmup_tokens") != warmup_tokens
        ):
            errors.append(
                f"{source_job_id} result could be mistaken for a fit observation"
            )

        try:
            expected_peak_lr = _peak_lr_for_job(manifest, args)
            observed_peak_lr = float(payload.get("peak_lr", math.nan))
        except (TypeError, ValueError, WorkflowError) as exc:
            errors.append(f"{source_job_id} LR lineage failed: {exc}")
            expected_peak_lr = math.nan
        else:
            if not math.isclose(
                observed_peak_lr, expected_peak_lr, rel_tol=0.0, abs_tol=0.0
            ):
                errors.append(
                    f"{source_job_id} did not use its manifest peak LR"
                )

        tier_manifest = (
            paths.tokenized / "tiers" / f"K{cell[2]}" / "megatron_data_manifest.json"
        ).resolve()
        if (
            not tier_manifest.is_file()
            or payload.get("data_tier") != f"K{cell[2]}"
            or payload.get("data_manifest") != str(tier_manifest)
            or payload.get("data_manifest_sha256") != _file_sha256(tier_manifest)
        ):
            errors.append(f"{source_job_id} data tier/hash lineage is invalid")

        config_path = Path(str(payload.get("resolved_pipeline_config", "")))
        if not config_path.is_file() or payload.get(
            "resolved_pipeline_config_sha256"
        ) != _file_sha256(config_path):
            errors.append(f"{source_job_id} resolved config identity is invalid")
        else:
            config = _load_object(config_path)
            profile = config.get("profiles", {}).get(
                _workflow_profile_name(runtime), {}
            )
            training = profile.get("training", {}) if isinstance(profile, dict) else {}
            train = training.get("train", {}) if isinstance(training, dict) else {}
            runs = training.get("runs", []) if isinstance(training, dict) else []
            run = runs[0] if isinstance(runs, list) and len(runs) == 1 else {}
            overrides = (
                training.get("overrides", []) if isinstance(training, dict) else []
            )
            required_overrides = {
                "scheduler.lr_decay_style=constant",
                "scheduler.lr_warmup_fraction=null",
                "scheduler.lr_warmup_samples=0",
                f"scheduler.lr_warmup_iters={warmup_steps}",
            }
            if (
                train.get("global_batch_size") != 32
                # The fixed-LR probes deliberately hold the *global* batch at
                # 32, but retain each architecture's validated physical
                # micro-batch/gradient-accumulation layout.  Requiring 32 here
                # would reject valid MoE probes such as C/E32 (MBS=8, GA=2).
                or train.get("micro_batch_size") != expected_mbs
                or train.get("train_iters") != 2_560
                or train.get("batch_policy") != {"enabled": False}
                or not required_overrides.issubset(set(overrides))
                or run.get("data_manifest") != str(tier_manifest)
                or run.get("data_manifest_sha256") != _file_sha256(tier_manifest)
                or training.get("python_executable")
                != runtime["bridge_python_executable"]
                or training.get("launch", {}).get("preflight_cuda") is not True
            ):
                errors.append(
                    f"{source_job_id} resolved early-schedule/data contract is invalid"
                )

        health = payload.get("sentinel_health")
        if not isinstance(health, dict):
            errors.append(f"{source_job_id} lacks training health")
        else:
            reparsed = _parse_sentinel_health(
                Path(str(health.get("log_path", ""))),
                require_moe=cell[1] > 1,
                expected_learning_rate=expected_peak_lr,
                allow_linear_warmup=True,
            )
            item["training_health"] = reparsed
            if (
                health.get("log_sha256") != reparsed.get("log_sha256")
                or reparsed.get("status") != "ok"
                or reparsed.get("metrics", {}).get("global_batch_size") != 32
            ):
                errors.append(f"{source_job_id} training health/log identity failed")

        checkpoint_root_text = str(payload.get("final_checkpoint_root", "")).strip()
        step = int(payload.get("final_checkpoint_step", 0) or 0)
        samples = int(payload.get("expected_consumed_samples", 0) or 0)
        if not checkpoint_root_text or step <= 0 or samples <= 0:
            errors.append(
                f"{source_job_id} lacks exact checkpoint root/step/sample metadata"
            )
            continue
        checkpoint_root = Path(checkpoint_root_text)
        state = inspect_full_state_checkpoint(
            checkpoint_root=checkpoint_root,
            expected_step=step,
            expected_consumed_samples=samples,
        )
        item["checkpoint_state"] = state.to_dict()
        if (
            step != 2_560
            or samples != 81_920
            or payload.get("actual_end_tokens") != probe_tokens
            or not state.valid_full_state
        ):
            errors.append(f"{source_job_id} exact full-state endpoint is invalid")

    if observed_cells != expected_cells:
        errors.append(
            f"schedule-probe cells differ from frozen cross: {sorted(observed_cells)} != "
            f"{sorted(expected_cells)}"
        )
    gate = {
        "schema_version": 1,
        "status": "failed" if errors else "ready",
        "fit_observation": False,
        "selection_mutation_allowed": False,
        "errors": errors,
        "results": results,
    }
    gate["gate_sha256"] = _payload_sha256(gate)
    gate_name = (
        "prescribed_lr_schedule_gate.json"
        if prescribed
        else "fixed_lr_schedule_gate.json"
    )
    _atomic_json(paths.control / gate_name, gate)
    if errors:
        policy = "prescribed-LR" if prescribed else "fixed-LR"
        raise WorkflowError(
            f"{policy} K-extreme schedule gate failed: " + "; ".join(errors)
        )


def _run_prescribed_lr_schedule_gate(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    _run_lr_schedule_gate(manifest, job, paths, prescribed=True)


def _run_fixed_lr_schedule_gate(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    _run_lr_schedule_gate(manifest, job, paths, prescribed=False)
