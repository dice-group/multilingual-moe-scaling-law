"""Final aggregation gate for completed evaluation trajectories."""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    PRIMARY_METRIC,
    Paths,
    WorkflowError,
    _abs,
    _atomic_json,
    _file_sha256,
    _job,
    _load_object,
)
from .gate_evaluation import _metric_from_eval, _secondary_metric_from_eval


def _run_final_summary(
    manifest: Mapping[str, Any], job: Mapping[str, Any], paths: Paths
) -> None:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for eval_job_id in job["arguments"]["evaluation_jobs"]:
        path = paths.jobs / eval_job_id / "evaluation_manifest.json"
        try:
            metric = _metric_from_eval(path)
            secondary_metric = _secondary_metric_from_eval(path)
        except WorkflowError as exc:
            errors.append(f"{eval_job_id}: {exc}")
            continue
        source = _job(manifest, eval_job_id)
        training_job_id = str(source["arguments"]["training_job_id"])
        training_result_path = paths.jobs / training_job_id / "result.json"
        training_result = _load_object(training_result_path)
        rows.append(
            {
                "job_id": eval_job_id,
                **source["arguments"],
                "nominal_D_tokens": int(source["arguments"]["D"]),
                "actual_end_tokens": int(
                    training_result.get("actual_end_tokens", 0) or 0
                ),
                "expected_consumed_samples": int(
                    training_result.get("expected_consumed_samples", 0) or 0
                ),
                PRIMARY_METRIC: metric,
                "raw_token_nll": secondary_metric,
                "sha256": _file_sha256(path),
            }
        )
    expected_results = len(job["arguments"]["evaluation_jobs"])
    report = {
        "schema_version": 1,
        "status": "failed" if errors else "complete",
        "expected_results": expected_results,
        "observed_results": len(rows),
        "errors": errors,
        "results": rows,
    }
    declared_outputs = list(job.get("outputs", []))
    output_path = (
        _abs(str(declared_outputs[0]))
        if declared_outputs
        else paths.control / "final_summary.json"
    )
    _atomic_json(output_path, report)
    if errors or len(rows) != expected_results:
        raise WorkflowError(
            f"final summary incomplete: {len(rows)}/{expected_results}; errors={len(errors)}"
        )
