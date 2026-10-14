"""Dispatch allocation-local A/B-v2 data and training jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import (
    _array_task_output_hashes,
    _declared_output_hashes,
    _has_completed_job_evidence,
    _quarantine_retry_owned_outputs,
    _record_source_drift_advisory,
    _receipt_path,
    _run_storage_preflight,
    _verified_receipt_evidence,
    _write_receipt,
)
from .artifact_data import (
    _guard_sealed_corpus_retry,
    _run_data_audit,
    _run_megatron_data_smoke,
    _run_reuse_sealed_data_verify,
    _verify_workflow_data,
)
from .common import (
    LANGUAGE_COUNTS,
    Paths,
    WorkflowError,
    _abs,
    _file_sha256,
    _is_h100_resource,
    _job,
    _language_map,
    _load_object,
    _paths_for_manifest,
    _run,
    _validate_recovery_job_authorization,
)
from .contract import (
    validate_manifest,
)
from .gate_lr_diagnostics import _run_lr_diagnostic_gate
from .gate_lr_schedule import (
    _run_fixed_lr_schedule_gate,
    _run_prescribed_lr_schedule_gate,
)
from .gate_production_lr_evidence import _run_production_lr_evidence_gate
from .gate_retention import _run_checkpoint_prune
from .gate_sentinel import _run_sentinel_gate
from .gate_summary import _run_final_summary
from .training import (
    _run_training_job,
    _run_validation_job,
)
from .variant import active_variant


def _bootstrap_base_args(
    paths: Paths, runtime: Mapping[str, Any], *, output_dir: Path
) -> list[str]:
    data = runtime["data"]
    return [
        "python",
        "-m",
        "multilingual_megatron.pipeline.stages.sampling.tools.bootstrap_fineweb_v2",
        "--source-lock",
        str(_abs("configs/data_sources/fineweb2_v2_1_1_source_lock.json")),
        "--rehydration-weights",
        str(paths.source_lock / "fineweb2-up-to-10-reps.json"),
        "--source-inventory",
        str(paths.source_lock / "fineweb2-language-distribution.csv"),
        "--language-map",
        str(_abs(_language_map(200))),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(runtime["seeds"]["data"]),
        "--validation-fraction",
        str(data["validation_candidate_fraction"]),
        "--target-unique-documents",
        str(data["bootstrap_target_unique_documents"]),
        "--target-utf8-bytes",
        str(data["bootstrap_target_utf8_bytes"]),
        "--calibration-documents",
        str(data["calibration_documents_per_language"]),
    ]


def _tokenizer_bootstrap_args(paths: Paths, runtime: Mapping[str, Any]) -> list[str]:
    return [
        *_bootstrap_base_args(paths, runtime, output_dir=paths.tokenizer_bootstrap),
        "--tokenizer-acquisition-plan",
        str(paths.tokenizer_acquisition_plan),
        "--incremental-from",
        str(paths.bootstrap),
    ]


def _direct_command(
    job: Mapping[str, Any], paths: Paths, runtime: Mapping[str, Any]
) -> list[str] | None:
    action = str(job["action"])
    data = runtime["data"]
    if action == "source_lock":
        return [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.sampling.tools.materialize_fineweb_v2_source_lock",
            "--source-lock",
            str(_abs("configs/data_sources/fineweb2_v2_1_1_source_lock.json")),
            "--output-dir",
            str(paths.source_lock),
        ]
    if action == "bootstrap_metadata_preflight":
        return [
            *_bootstrap_base_args(paths, runtime, output_dir=paths.bootstrap),
            "--metadata-preflight",
        ]
    if action == "bootstrap_worker":
        return [
            *_bootstrap_base_args(paths, runtime, output_dir=paths.bootstrap),
            "--language-index",
        ]
    if action == "bootstrap_merge":
        return [
            *_bootstrap_base_args(paths, runtime, output_dir=paths.bootstrap),
            "--merge-summary",
        ]
    if action == "tokenizer_acquisition_plan":
        return [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.tokenizer_training.plan_sentence_acquisition",
            "--pilot-bootstrap-summary",
            str(paths.bootstrap / "bootstrap_summary.json"),
            "--language-map",
            str(_abs(_language_map(200))),
            "--output",
            str(paths.tokenizer_acquisition_plan),
            "--target-sentences",
            str(data["tokenizer_training_sentences"]),
            "--alpha",
            str(data["alpha"]),
            "--seed",
            str(runtime["seeds"]["data"]),
            "--acquisition-safety-margin",
            "1.10",
        ]
    if action == "tokenizer_bootstrap_metadata_preflight":
        return [*_tokenizer_bootstrap_args(paths, runtime), "--metadata-preflight"]
    if action == "tokenizer_bootstrap_worker":
        return [*_tokenizer_bootstrap_args(paths, runtime), "--language-index"]
    if action == "tokenizer_bootstrap_merge":
        return [*_tokenizer_bootstrap_args(paths, runtime), "--merge-summary"]
    if action == "tokenizer_sentences":
        return [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.tokenizer_training.build_sentence_artifact",
            "--input-manifest",
            str(paths.tokenizer_bootstrap / "bootstrap_summary.json"),
            "--language-map",
            str(_abs(_language_map(200))),
            "--output-jsonl",
            str(paths.sentence_root / "tokenizer_sentences_100m.jsonl"),
            "--output-manifest",
            str(paths.sentence_root / "tokenizer_sentences_100m.manifest.json"),
            "--target-sentences",
            str(data["tokenizer_training_sentences"]),
            "--alpha",
            str(data["alpha"]),
            "--seed",
            str(runtime["seeds"]["data"]),
            "--acquisition-plan",
            str(paths.tokenizer_acquisition_plan),
        ]
    if action == "token_calibration":
        return [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.sampling.tools.calibrate_fineweb_v2_tokens",
            "--bootstrap-summary",
            str(paths.bootstrap / "bootstrap_summary.json"),
            "--tokenizer-model",
            str(paths.tokenizer / "tokenizer.model"),
            "--source-lock",
            str(_abs("configs/data_sources/fineweb2_v2_1_1_source_lock.json")),
            "--inventory",
            str(paths.source_lock / "fineweb2-language-distribution.csv"),
            "--output",
            str(paths.calibration),
            "--sample-documents",
            str(data["calibration_documents_per_language"]),
            "--sample-seed",
            str(runtime["seeds"]["data"]),
            "--expected-language-count",
            "200",
            "--expected-vocab-size",
            "64000",
            "--target-validation-fraction",
            str(data["validation_target_fraction"]),
        ]
    if action == "sampling_plan":
        cmd = [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.sampling.tools.build_fineweb_v2_sampling_plan",
            "--calibration",
            str(paths.calibration),
        ]
        for k in LANGUAGE_COUNTS:
            cmd.extend(["--language-map", f"K{k}={_abs(_language_map(k))}"])
        cmd.extend(
            [
                "--source-manifest",
                str(paths.source_lock / "source_materialization_manifest.json"),
                "--tokenizer-model",
                str(paths.tokenizer / "tokenizer.model"),
                "--target-train-tokens",
                str(data["target_train_tokens_per_K"]),
                "--alpha",
                str(data["alpha"]),
                "--sampling-seed",
                str(runtime["seeds"]["data"]),
                "--validation-cap-tokens",
                str(data["validation_cap_tokens_per_language"]),
                "--output",
                str(paths.sampling_plan),
            ]
        )
        return cmd
    if action == "production_metadata_preflight":
        return [
            *_bootstrap_base_args(
                paths, runtime, output_dir=paths.production_bootstrap
            ),
            "--metadata-preflight",
        ]
    if action in {"production_worker", "production_merge"}:
        cmd = [
            *_bootstrap_base_args(
                paths, runtime, output_dir=paths.production_bootstrap
            ),
            "--sampling-plan",
            str(paths.sampling_plan),
            "--production-safety-margin",
            str(data["production_safety_margin"]),
            "--incremental-from",
            str(paths.bootstrap),
        ]
        cmd.append(
            "--language-index" if action == "production_worker" else "--merge-summary"
        )
        return cmd
    if action == "finalize_corpus":
        return [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.sampling.tools.finalize_fineweb_v2_corpus",
            "--bootstrap-manifest",
            str(paths.production_bootstrap / "bootstrap_manifest.json"),
            "--tokenizer-model",
            str(paths.tokenizer / "tokenizer.model"),
            "--sampling-plan",
            str(paths.sampling_plan),
            "--output-dir",
            str(paths.release),
            "--max-train-shortfall-fraction",
            "0.0",
            "--validation-target-fraction",
            str(data["validation_target_fraction"]),
        ]
    return None


def _run_tokenizer(paths: Paths, runtime: Mapping[str, Any]) -> None:
    manifest = _load_object(
        paths.sentence_root / "tokenizer_sentences_100m.manifest.json"
    )
    source = paths.sentence_root / "tokenizer_sentences_100m.jsonl"
    digest = _file_sha256(source)
    declared = str(
        manifest.get("output", {}).get(
            "sha256", manifest.get("sentence_jsonl_sha256", "")
        )
    )
    if declared and digest != declared:
        raise WorkflowError(
            "tokenizer sentence artifact hash differs from its manifest"
        )
    _run(
        [
            "python",
            "-m",
            "multilingual_megatron.pipeline.stages.tokenizer_training.train_hf_tokenizer",
            "--output-dir",
            str(paths.tokenizer),
            "--mode",
            "train_unigram",
            "--vocab-size",
            "64000",
            "--text-key",
            "text",
            "--max-samples",
            "0",
            "--normalizer",
            "nmt_nfkc",
            "--pre-tokenizer",
            "metaspace",
            "--min-frequency",
            "2",
            "--unk-token",
            "<unk>",
            "--bos-token",
            "<s>",
            "--eos-token",
            "</s>",
            "--pad-token",
            "<pad>",
            "--num-threads",
            str(runtime["data"]["tokenizer_threads"]),
            "--production-v2-sentence-jsonl",
            str(source),
            "--production-v2-sentence-jsonl-sha256",
            digest,
            "--production-v2-expected-sentence-count",
            str(runtime["data"]["tokenizer_training_sentences"]),
        ]
    )


def execute_job(
    manifest_path: Path,
    job_id: str,
    *,
    validate_manifest_fn: Callable[..., dict[str, Any]] = validate_manifest,
) -> None:
    manifest = _load_object(manifest_path)
    validation = validate_manifest_fn(manifest)
    _record_source_drift_advisory(manifest, validation, job_id=job_id)
    _validate_recovery_job_authorization(manifest, job_id)
    job = _job(manifest, job_id)
    task = os.getenv("SLURM_ARRAY_TASK_ID") if job.get("array") else None
    own_receipt = _receipt_path(manifest, job_id, task)
    if own_receipt.is_file():
        # Retried arrays are submitted as a whole. A task whose own receipt is
        # still hash-valid is already complete and must not redo or quarantine
        # its large output merely because a sibling failed.
        _verified_receipt_evidence(manifest, job_id, array_task=task)
        return
    paths = _paths_for_manifest(manifest)
    _guard_sealed_corpus_retry(job, paths)
    for dependency in job.get("dependencies", []):
        dependency_job = _job(manifest, str(dependency))
        if not _has_completed_job_evidence(manifest, {"jobs": {}}, dependency_job):
            raise WorkflowError(
                f"job {job_id} lacks complete hash-valid dependency evidence for {dependency}"
            )
    if os.getenv("PROOF_AB_V2_RETRY_FAILED") == "1":
        _quarantine_retry_owned_outputs(manifest, job)
    runtime = _load_object(Path(str(manifest["runtime_path"])))
    if _is_h100_resource(job["resource"]):
        _verify_workflow_data(manifest, paths)
        allocation = os.getenv("SLURM_JOB_ID", "local")
        os.environ["TRITON_CACHE_DIR"] = str(
            paths.root / "runtime" / "triton" / job_id / allocation
        )
        os.environ["TRITON_HOME"] = str(
            paths.root / "runtime" / "triton_home" / job_id / allocation
        )
        Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["TRITON_HOME"]).mkdir(parents=True, exist_ok=True)

    direct = _direct_command(job, paths, runtime)
    if direct is not None:
        _run(direct)
    elif job["action"] == "tokenizer_train":
        _run_tokenizer(paths, runtime)
    elif job["action"] in {
        "presplit_tokenize_plan",
        "presplit_tokenize_worker",
        "presplit_tokenize_merge",
    }:
        _run_presplit_tokenization(job, paths, runtime)
    elif job["action"] in {
        "training_sentinel",
        "fixed_lr_schedule_probe",
        "prescribed_lr_schedule_probe",
        "production_train",
    }:
        _run_training_job(manifest, job, paths, runtime)
    elif job["action"] == "per_language_eval":
        _run_validation_job(manifest, job, paths, runtime)
    elif job["action"] == "data_audit":
        _run_data_audit(manifest, paths)
    elif job["action"] == "reuse_sealed_data_verify":
        _run_reuse_sealed_data_verify(manifest, paths)
    elif job["action"] == "megatron_data_smoke":
        _run_megatron_data_smoke(manifest, paths, runtime)
    elif job["action"] == "storage_preflight":
        _run_storage_preflight(manifest, paths, runtime)
    elif job["action"] == "checkpoint_prune":
        _run_checkpoint_prune(manifest, job, paths)
    elif job["action"] == "sentinel_gate":
        _run_sentinel_gate(manifest, job, paths, runtime)
    elif job["action"] == "fixed_lr_schedule_gate":
        _run_fixed_lr_schedule_gate(manifest, job, paths)
    elif job["action"] == "prescribed_lr_schedule_gate":
        _run_prescribed_lr_schedule_gate(manifest, job, paths)
    elif job["action"] == "lr_diagnostic_gate":
        _run_lr_diagnostic_gate(manifest, job, paths, runtime)
    elif job["action"] == "production_lr_evidence_gate":
        _run_production_lr_evidence_gate(manifest, job, paths)
    elif job["action"] == "final_summary":
        _run_final_summary(manifest, job, paths)
    else:
        handler = active_variant().custom_action_handler
        if handler is None or handler(manifest, job, paths, runtime) is not True:
            raise WorkflowError(f"unimplemented A/B-v2 workflow action: {job['action']}")

    if _is_h100_resource(job["resource"]):
        # Close the preflight-to-use window before any successful H100 receipt
        # can become dependency evidence for a downstream job.
        _verify_workflow_data(manifest, paths)
    output_hashes = _declared_output_hashes(manifest, job)
    output_hashes.update(_array_task_output_hashes(job, paths, task))
    _write_receipt(
        manifest=manifest, job=job, status="complete", output_hashes=output_hashes
    )


def _run_presplit_tokenization(
    job: Mapping[str, Any], paths: Paths, runtime: Mapping[str, Any]
) -> None:
    """Call the stage-owned pre-split tokenizer without duplicating its logic."""

    module = "multilingual_megatron.pipeline.stages.tokenization.fineweb_v2_presplit"
    data = runtime["data"]
    if job["action"] == "presplit_tokenize_plan":
        _run(
            [
                "python",
                "-m",
                module,
                "plan",
                "--corpus-manifest",
                str(paths.release / "corpus_manifest.json"),
                "--sampling-plan",
                str(paths.sampling_plan),
                "--tokenizer-model",
                str(paths.tokenizer / "tokenizer.model"),
                "--plan-dir",
                str(paths.tokenization_work / "plan"),
                "--array-ranks",
                str(data["tokenization_array_ranks"]),
                "--verification-workers",
                str(data["tokenization_plan_verification_workers"]),
            ]
        )
    elif job["action"] == "presplit_tokenize_worker":
        _run(
            [
                "python",
                "-m",
                module,
                "worker",
                "--plan-dir",
                str(paths.tokenization_work / "plan"),
                "--ranks-dir",
                str(paths.tokenization_work / "ranks"),
                "--bridge-root",
                str(_abs("external/Megatron-Bridge")),
                "--python-executable",
                str(runtime["bridge_python_executable"]),
                "--workers",
                str(data["tokenization_workers_per_rank"]),
                "--partitions-per-language",
                str(data["tokenization_partitions_per_language"]),
            ]
        )
    else:
        _run(
            [
                "python",
                "-m",
                module,
                "merge",
                "--plan-dir",
                str(paths.tokenization_work / "plan"),
                "--ranks-dir",
                str(paths.tokenization_work / "ranks"),
                "--output-dir",
                str(paths.tokenized),
            ]
        )
