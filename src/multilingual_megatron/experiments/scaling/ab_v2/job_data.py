"""Construct the CPU corpus, tokenizer, and tokenization side of the DAG."""

from __future__ import annotations

from typing import Any, Mapping

from .common import DATA_ARTIFACT_SEAL_NAME, LANGUAGE_COUNTS, Paths
from .jobs import _add_job


def _data_jobs(
    jobs: list[dict[str, Any]], paths: Paths, runtime: Mapping[str, Any]
) -> str:
    data = runtime["data"]
    storage = _add_job(
        jobs,
        job_id="storage_preflight",
        stage="storage",
        action="storage_preflight",
        resource="cpu",
        outputs=(paths.control / "storage_preflight.json",),
    )
    if data.get("mode") in {"adopt_k12_migration", "reuse_sealed_k12"}:
        audit = _add_job(
            jobs,
            job_id=(
                "data_reuse_gate"
                if data.get("mode") == "reuse_sealed_k12"
                else "data_audit_gate"
            ),
            stage="data",
            action=(
                "reuse_sealed_data_verify"
                if data.get("mode") == "reuse_sealed_k12"
                else "data_audit"
            ),
            resource="cpu",
            dependencies=(storage,),
            outputs=(
                (
                    paths.control / "data_reuse_evidence.json"
                    if data.get("mode") == "reuse_sealed_k12"
                    else paths.control / "data_audit.json"
                ),
                *(
                    ()
                    if data.get("mode") == "reuse_sealed_k12"
                    else (paths.control / DATA_ARTIFACT_SEAL_NAME,)
                ),
            ),
        )
        return _add_megatron_data_smoke(jobs, paths, dependency=audit)
    source = _add_job(
        jobs,
        job_id="data_source_lock",
        stage="data",
        action="source_lock",
        resource="cpu",
        dependencies=(storage,),
        outputs=(paths.source_lock / "source_materialization_manifest.json",),
    )
    preflight = _add_job(
        jobs,
        job_id="data_metadata_preflight",
        stage="data",
        action="bootstrap_metadata_preflight",
        resource="cpu",
        dependencies=(source,),
        outputs=(paths.bootstrap / "paths_preflight_summary.json",),
    )
    bootstrap_array = _add_job(
        jobs,
        job_id="data_bootstrap_workers",
        stage="data",
        action="bootstrap_worker",
        resource="cpu",
        dependencies=(preflight,),
        array=str(data["language_array"]),
    )
    bootstrap_merge = _add_job(
        jobs,
        job_id="data_bootstrap_merge",
        stage="data",
        action="bootstrap_merge",
        resource="cpu",
        dependencies=(bootstrap_array,),
        outputs=(
            paths.bootstrap / "bootstrap_manifest.json",
            paths.bootstrap / "bootstrap_summary.json",
        ),
    )
    acquisition_plan = _add_job(
        jobs,
        job_id="data_tokenizer_acquisition_plan",
        stage="data",
        action="tokenizer_acquisition_plan",
        resource="cpu_largemem",
        dependencies=(bootstrap_merge,),
        outputs=(paths.tokenizer_acquisition_plan,),
    )
    tokenizer_preflight = _add_job(
        jobs,
        job_id="data_tokenizer_metadata_preflight",
        stage="data",
        action="tokenizer_bootstrap_metadata_preflight",
        resource="cpu",
        dependencies=(acquisition_plan,),
        outputs=(paths.tokenizer_bootstrap / "paths_preflight_summary.json",),
    )
    tokenizer_bootstrap_workers = _add_job(
        jobs,
        job_id="data_tokenizer_bootstrap_workers",
        stage="data",
        action="tokenizer_bootstrap_worker",
        resource="cpu_streaming",
        dependencies=(tokenizer_preflight,),
        array=str(data["language_array"]),
    )
    tokenizer_bootstrap_merge = _add_job(
        jobs,
        job_id="data_tokenizer_bootstrap_merge",
        stage="data",
        action="tokenizer_bootstrap_merge",
        resource="cpu_streaming",
        dependencies=(tokenizer_bootstrap_workers,),
        outputs=(
            paths.tokenizer_bootstrap / "bootstrap_manifest.json",
            paths.tokenizer_bootstrap / "bootstrap_summary.json",
        ),
    )
    sentence = _add_job(
        jobs,
        job_id="data_tokenizer_sentences",
        stage="data",
        action="tokenizer_sentences",
        resource="cpu_largemem",
        dependencies=(tokenizer_bootstrap_merge,),
        outputs=(
            paths.sentence_root / "tokenizer_sentences_100m.jsonl",
            paths.sentence_root / "tokenizer_sentences_100m.manifest.json",
        ),
    )
    tokenizer = _add_job(
        jobs,
        job_id="data_tokenizer_train",
        stage="data",
        action="tokenizer_train",
        resource="cpu_tokenizer",
        dependencies=(sentence,),
        outputs=(
            paths.tokenizer / "tokenizer.model",
            paths.tokenizer / "tokenizer_training_result.json",
        ),
    )
    calibration = _add_job(
        jobs,
        job_id="data_token_calibration",
        stage="data",
        action="token_calibration",
        resource="cpu_largemem",
        dependencies=(tokenizer,),
        outputs=(paths.calibration,),
    )
    plan = _add_job(
        jobs,
        job_id="data_sampling_plan",
        stage="data",
        action="sampling_plan",
        resource="cpu",
        dependencies=(calibration,),
        outputs=(paths.sampling_plan,),
    )
    production_preflight = _add_job(
        jobs,
        job_id="data_production_metadata_preflight",
        stage="data",
        action="production_metadata_preflight",
        resource="cpu",
        dependencies=(plan,),
        outputs=(paths.production_bootstrap / "paths_preflight_summary.json",),
    )
    production_workers = _add_job(
        jobs,
        job_id="data_production_workers",
        stage="data",
        action="production_worker",
        resource="cpu_streaming",
        dependencies=(production_preflight,),
        array=str(data["language_array"]),
    )
    production_merge = _add_job(
        jobs,
        job_id="data_production_merge",
        stage="data",
        action="production_merge",
        resource="cpu",
        dependencies=(production_workers,),
        outputs=(
            paths.production_bootstrap / "bootstrap_manifest.json",
            paths.production_bootstrap / "bootstrap_summary.json",
        ),
    )
    finalizer = _add_job(
        jobs,
        job_id="data_finalize",
        stage="data",
        action="finalize_corpus",
        resource="cpu_largemem",
        dependencies=(production_merge,),
        outputs=(
            paths.release / "corpus_manifest.json",
            paths.release / "hash_manifest.json",
        ),
    )
    token_plan = _add_job(
        jobs,
        job_id="data_tokenize_plan",
        stage="data",
        action="presplit_tokenize_plan",
        resource="cpu_largemem",
        dependencies=(finalizer,),
        outputs=(paths.tokenization_work / "plan" / "summary.json",),
    )
    token_workers = _add_job(
        jobs,
        job_id="data_tokenize_workers",
        stage="data",
        action="presplit_tokenize_worker",
        resource="cpu_largemem",
        dependencies=(token_plan,),
        array=str(data["tokenization_array"]),
    )
    token_merge = _add_job(
        jobs,
        job_id="data_tokenize_merge",
        stage="data",
        action="presplit_tokenize_merge",
        resource="cpu_largemem",
        dependencies=(token_workers,),
        outputs=(
            paths.tokenized / "megatron_data_manifest.json",
            paths.tokenized / "validation_megatron_data_manifest.json",
            paths.tokenized / "tokenization_report.json",
            *(
                paths.tokenized / "tiers" / f"K{k}" / "megatron_data_manifest.json"
                for k in LANGUAGE_COUNTS
            ),
        ),
    )
    gate_id = _add_job(
        jobs,
        job_id="data_audit_gate",
        stage="data",
        action="data_audit",
        resource="cpu",
        dependencies=(token_merge,),
        outputs=(
            paths.control / "data_audit.json",
            paths.control / DATA_ARTIFACT_SEAL_NAME,
        ),
    )
    return _add_megatron_data_smoke(jobs, paths, dependency=gate_id)


def _add_megatron_data_smoke(
    jobs: list[dict[str, Any]], paths: Paths, *, dependency: str
) -> str:
    return _add_job(
        jobs,
        job_id="data_megatron_reader_smoke",
        stage="data",
        action="megatron_data_smoke",
        resource="cpu_largemem",
        dependencies=(dependency,),
        outputs=(paths.control / "megatron_data_smoke.json",),
    )
