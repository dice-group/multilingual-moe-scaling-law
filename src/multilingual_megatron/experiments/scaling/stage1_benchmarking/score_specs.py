"""Native Megatron score-spec construction for frozen benchmark requests."""

from pathlib import Path
from typing import Any

from .contracts import (
    BenchmarkProfile, REQUEST_MANIFEST_SCHEMA, SCORE_SPEC_SCHEMA, _atomic_json,
    _load_json, _payload_sha256, _resolve, _sha256_file, _verify_payload_hash,
    reconcile_training_model_overrides,
)

def build_score_specs(
    profile: BenchmarkProfile,
    request_manifest_path: Path,
    output_dir: Path,
) -> list[Path]:
    request_path = request_manifest_path.resolve()
    request_payload = _load_json(request_path, label="frozen request manifest")
    if request_payload.get("schema") != REQUEST_MANIFEST_SCHEMA:
        raise ValueError("invalid frozen request manifest schema")
    _verify_payload_hash(request_payload, field="manifest_sha256", label="request manifest")
    if request_payload.get("pilot_id") != profile.request_manifest_pilot_id:
        raise ValueError(
            "request manifest pilot_id does not match the declared request contract"
        )
    request_file_sha = _sha256_file(request_path)
    tokenizer_cfg = profile.payload["tokenizer"]
    tokenizer_model = _resolve(
        profile.source_path.parent,
        tokenizer_cfg["model"],
        field="tokenizer.model",
    )
    tokenizer_sha = _sha256_file(tokenizer_model)
    written: list[Path] = []
    launch_rows: list[dict[str, Any]] = []
    for checkpoint in profile.checkpoints:
        evaluation_spec_path = _resolve(
            profile.source_path.parent,
            checkpoint["evaluation_spec"],
            field=f"checkpoints.{checkpoint['id']}.evaluation_spec",
        )
        evaluation_spec = _load_json(
            evaluation_spec_path, label=f"{checkpoint['id']} final-loss evaluation spec"
        )
        recipe = evaluation_spec.get("model_recipe")
        frozen_checkpoint = evaluation_spec.get("checkpoint")
        if not isinstance(recipe, dict) or not isinstance(frozen_checkpoint, dict):
            raise ValueError(f"invalid final-loss evaluation spec: {evaluation_spec_path}")
        step = int(frozen_checkpoint["step"])
        checkpoint_root = Path(frozen_checkpoint["path"]).resolve()
        iteration_path = checkpoint_root / f"iter_{step:07d}"
        if not iteration_path.is_dir():
            raise FileNotFoundError(f"missing checkpoint iteration: {iteration_path}")
        checkpoint_tokenizer = iteration_path / "tokenizer" / "tokenizer.model"
        if not checkpoint_tokenizer.is_file():
            raise FileNotFoundError(f"checkpoint tokenizer is missing: {checkpoint_tokenizer}")
        if _sha256_file(checkpoint_tokenizer) != tokenizer_sha:
            raise ValueError(f"checkpoint tokenizer drift: {checkpoint_tokenizer}")
        expected_model = recipe.get("expected_model")
        if not isinstance(expected_model, dict):
            raise ValueError(f"missing expected model contract: {evaluation_spec_path}")
        training_run_config_path = iteration_path / "run_config.yaml"
        if not training_run_config_path.is_file():
            raise FileNotFoundError(
                f"checkpoint training run config is missing: {training_run_config_path}"
            )
        import yaml

        training_run_config = yaml.safe_load(
            training_run_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(training_run_config, dict) or not isinstance(
            training_run_config.get("model"), dict
        ):
            raise ValueError(f"invalid checkpoint run config: {training_run_config_path}")
        declared_replacements = checkpoint.get(
            "training_model_override_replacements", {}
        )
        if not isinstance(declared_replacements, dict):
            raise ValueError(
                f"checkpoints.{checkpoint['id']}.training_model_override_replacements "
                "must be an object"
            )
        reconciled_overrides, replacement_receipt = reconcile_training_model_overrides(
            recipe.get("overrides", []),
            training_run_config["model"],
            declared_replacements,
        )
        expected_model = dict(expected_model)
        expected_model.update(declared_replacements)
        if int(expected_model.get("num_moe_experts") or 1) != int(checkpoint["experts"]):
            raise ValueError(f"expert count mismatch for {checkpoint['id']}")
        run_output = profile.output_root / "scores" / checkpoint["id"]
        score_spec: dict[str, Any] = {
            "schema": SCORE_SPEC_SCHEMA,
            "run_id": f"{profile.benchmark_id}_{checkpoint['id']}",
            "output_dir": str(run_output),
            "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
            "source_final_loss_evaluation_spec": {
                "path": str(evaluation_spec_path),
                "sha256": _sha256_file(evaluation_spec_path),
            },
            "request_manifest": {"path": str(request_path), "sha256": request_file_sha},
            "model_recipe": {
                "name": recipe["name"],
                "overrides": reconciled_overrides,
                "expected_model": expected_model,
                "training_contract_replacements": replacement_receipt,
            },
            "checkpoint": {
                "path": str(checkpoint_root),
                "iteration_path": str(iteration_path.resolve()),
                "step": step,
                "tokens_seen": int(checkpoint["tokens_seen"]),
                "training_run_config": {
                    "path": str(training_run_config_path.resolve()),
                    "sha256": _sha256_file(training_run_config_path),
                },
            },
            "tokenizer": {
                "type": tokenizer_cfg["type"],
                "model": str(checkpoint_tokenizer.resolve()),
                "sha256": tokenizer_sha,
                "vocab_size": int(tokenizer_cfg["vocab_size"]),
                "context_token_id": int(tokenizer_cfg["context_token_id"]),
                "pad_token_id": int(tokenizer_cfg["pad_token_id"]),
            },
            "evaluation": {
                "sequence_length": int(evaluation_spec["sequence_length"]),
                "micro_batch_size": int(checkpoint["micro_batch_size"]),
                "expected_world_size": int(checkpoint["gpus"]),
                "max_gpu_hours": float(checkpoint["max_gpu_hours"]),
                "parity_requests": int(profile.payload["parity"]["requests"]),
                "require_moe_routing_diagnostics": bool(checkpoint["experts"] > 1),
                "required_gpu_name_substring": str(
                    profile.payload["runtime"]["required_gpu_name_substring"]
                ),
            },
        }
        score_spec["spec_sha256"] = _payload_sha256(score_spec)
        spec_path = output_dir.resolve() / f"{checkpoint['id']}.score_spec.json"
        _atomic_json(spec_path, score_spec)
        written.append(spec_path)
        launch_rows.append(
            {
                "checkpoint_id": checkpoint["id"],
                "holder_job_id": None,
                "allocated_nodes": None,
                "node_subset": None,
                "gpu_count": int(checkpoint["gpus"]),
                "score_spec": str(spec_path),
                "command": [
                    str(profile.payload["bridge_python"]),
                    "-m",
                    "multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_native_benchmark",
                    "--bridge-root",
                    str(profile.bridge_root),
                    "--spec",
                    str(spec_path),
                    "--nproc-per-node",
                    str(checkpoint["gpus"]),
                    "--python-executable",
                    str(profile.payload["bridge_python"]),
                    "--preflight-cuda",
                ],
            }
        )
    launch_plan = {
        "schema": "stage1_lm_eval_benchmark_launch_plan/v1",
        "pilot_id": profile.benchmark_id,
        "profile": {"path": str(profile.source_path), "sha256": profile.source_sha256},
        "request_manifest": {"path": str(request_path), "sha256": request_file_sha},
        "jobs": launch_rows,
        "constraint": "populate holder_job_id, allocated_nodes, and node_subset from live allocation evidence before launch",
    }
    launch_plan["plan_sha256"] = _payload_sha256(launch_plan)
    _atomic_json(output_dir.resolve() / "launch_plan.json", launch_plan)
    return written
