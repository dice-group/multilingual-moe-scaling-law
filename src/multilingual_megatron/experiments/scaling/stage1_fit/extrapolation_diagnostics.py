"""Read-only-source diagnostics for already inspected scaling-law holdouts.

No fitting, selection, scheduling, or amendment of historical evidence occurs.
New outputs are create-only and bind the exact analysis source bytes, including
uncommitted bytes. They are diagnostic evidence, never a formal fitting receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import execution, fit, report as fit_report
from .baseline_refit_workflow import recovery
from .baseline_refit_workflow.contracts import FOLDS, MODELS, repository_root, seal, verify_seal
from .data import sha256_file
from .m5_m6_workflow import analysis as composition_analysis, campaign as composition_campaign

SCHEMA = "stage1_extrapolation_diagnostics/v1"
KEYS = ["trajectory_id", "readout_id", "language"]
SCORE_TOLERANCE = 1e-12


def bound_file(value: dict, *, root: Path | None = None) -> Path:
    path = Path(value["path"])
    path = path if path.is_absolute() else (root or repository_root()) / path
    if not path.is_file() or sha256_file(path) != value["file_sha256"]:
        raise ValueError(f"diagnostic input missing or changed: {path}")
    return path.resolve()


def sealed_json(path: Path, key: str) -> dict:
    value = execution.read_json(path)
    verify_seal(value, key, label=str(path))
    return value


def load_profile(path: Path) -> tuple[dict, dict[str, Path]]:
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != "stage1_extrapolation_diagnostics_profile/v1":
        raise ValueError("unknown diagnostics profile")
    required = {"recovery_plan", "comparison", "m6_index", "m6_report", "m6_audit", "release_manifest"}
    if set(profile["inputs"]) != required or profile["bootstrap_replicates"] < 100:
        raise ValueError("incomplete diagnostic profile")
    paths = {key: bound_file(value) for key, value in profile["inputs"].items()}
    output = Path(profile["output_root"])
    output = (repository_root() / output).resolve()
    # This CLI owns one new diagnostic directory, not historical result roots.
    if output.parent != repository_root() / "result_analysis" or not output.name.startswith("stage1_extrapolation_diagnostics_"):
        raise ValueError("output must be a separate stage1_extrapolation_diagnostics_* root")
    profile["output_root"] = str(output)
    return profile, paths


def aligned_components(base: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Prove exact base-row identity before accepting composition features."""
    if base.duplicated(KEYS).any() or components.duplicated(KEYS).any():
        raise ValueError("duplicate component/base row identity")
    left = base.sort_values(KEYS).reset_index(drop=True)
    right = components.sort_values(KEYS).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right[list(left.columns)], check_dtype=False, check_exact=True)
    except (AssertionError, KeyError) as error:
        raise ValueError("composition alignment or base values changed") from error
    exposures = right[["D_t", "D_family", "D_script", "D_other"]]
    if not np.isfinite(exposures.to_numpy()).all() or (exposures < 0).any().any() \
            or not np.allclose(exposures.sum(axis=1), right.D_total, rtol=1e-12, atol=1e-6):
        raise ValueError("composition components do not reconstruct total exposure")
    return right


def predict_rows(frame: pd.DataFrame, membership: dict, selected: dict, model: str, split: str) -> pd.DataFrame:
    # score_languages constrains evaluation, not the actual training objective.
    member = dict(membership)
    if split == "train":
        member["score_languages"] = []
    if model != "M6":
        rows, prediction = fit_report._prediction(frame, member, selected, model, split=split)
    else:
        rows = frame[frame.trajectory_id.isin(member[f"{split}_trajectory_ids"])].copy()
        if member[f"{split}_readout_ids"]:
            rows = rows[rows.readout_id.isin(member[f"{split}_readout_ids"])]
        if member["score_languages"]:
            rows = rows[rows.language.isin(member["score_languages"])]
        coefficients = {v["language"]: v for v in selected["language_coefficients"]}
        if rows.empty or not set(rows.language) <= set(coefficients):
            raise ValueError("M6 membership empty or lacks language coefficients")
        nuisance = [coefficients[v] for v in rows.language]
        prediction = fit.predict(model, fit.SharedParameters(**selected["shared_parameters"]),
            rows.N_act, rows.D_t, rows.E, rows.K,
            [v["L_inf"] for v in nuisance], [v["A"] for v in nuisance], [v["B"] for v in nuisance],
            **{name: rows[name] for name in composition_analysis.COMPONENT_COLUMNS})
    rows["prediction"] = prediction
    if not np.isfinite(rows[["prediction", "adjusted_loss"]].to_numpy()).all():
        raise ValueError("non-finite predictions or observations")
    return rows


def annotate(rows: pd.DataFrame, *, model: str, fold: str, split: str, delta: float) -> pd.DataFrame:
    rows = rows.copy()
    rows["model"], rows["fold"], rows["split"] = model, fold, split
    rows["residual"] = rows.prediction - rows.adjusted_loss
    rows["squared_error"] = rows.residual ** 2
    rows["weight"] = fit.trajectory_weights(tuple(rows.trajectory_id))
    rows["delta"] = delta
    rows["huber_influence_ratio"] = np.minimum(1., delta / np.maximum(np.abs(rows.residual), np.finfo(float).tiny))
    rows["huber_loss"] = fit.huber_loss(rows.residual, delta)
    rows["total_tokens_per_active_parameter"] = rows.D_total / rows.N_act
    rows["target_tokens_per_active_parameter"] = rows.D_t / rows.N_act
    rows["total_tokens_per_nonembedding_parameter"] = rows.D_total / rows.N_act_nonembedding
    return rows


def reproduce_score(rows: pd.DataFrame, expected: dict) -> float:
    from .selection import trajectory_mean_squared_errors
    errors = trajectory_mean_squared_errors(rows.adjusted_loss, rows.prediction, rows.trajectory_id)
    if set(errors) != set(expected["trajectory_mse"]) or len(rows) != expected["rows"]:
        raise ValueError("scored membership differs from accepted report")
    difference = max(abs(errors[key] - expected["trajectory_mse"][key]) for key in errors)
    rmse = float(np.sqrt(np.mean(list(errors.values()))))
    difference = max(difference, abs(rmse - expected["trajectory_balanced_rmse"]))
    if difference > SCORE_TOLERANCE:
        raise ValueError(f"accepted score reproduction failed: {difference}")
    return difference


def historical_source(profile: dict, component_receipt: dict) -> dict:
    commit = profile["preserved_m6_source_commit"]
    core_hash = execution.source_sha256_at_commit(commit)
    if core_hash != execution.source_sha256():
        raise ValueError("current predictor/fitter differs from preserved M6 source")
    files = {}
    for key in ("runner", "component_builder"):
        binding = component_receipt[key]
        relative = str(Path(binding["path"]).relative_to(repository_root()))
        content = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repository_root(),
                                 check=True, capture_output=True).stdout
        digest = hashlib.sha256(content).hexdigest()
        files[key] = {"path": relative, "receipt_sha256": binding["file_sha256"],
            "preserved_sha256": digest, "receipt_matches_preserved": digest == binding["file_sha256"]}
    if not files["component_builder"]["receipt_matches_preserved"]:
        raise ValueError("historical composition builder does not match its receipt")
    return {"preserved_commit": commit, "core_source_sha256": core_hash,
        "live_predictor_core_byte_identical": True, "files": files,
        "disclosure": "The component receipt's old runner hash is not the retained runner snapshot. "
        "This diagnostics reader verifies the exact predictor/core bytes, component builder, immutable "
        "inputs, reduced results and numerical audit; it does not certify original orchestration replay "
        "or waive the historical nested-amendment source check."}


def validate_m6_audit(audit: dict, result: dict, fold: str) -> dict:
    matching = [v for v in audit["items"] if (v["model"], v["lambda_key"], v["fold"]) == ("M6", "0.01", fold)]
    if len(matching) != 1 or matching[0]["status"] != "healthy" or matching[0]["base_result_sha256"] != result["result_sha256"]:
        raise ValueError("M6 lacks an exact healthy numerical audit")
    return matching[0]


def collect_predictions(profile: dict, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict, dict]:
    print("Verifying accepted baseline recovery and M6 input lineage", flush=True)
    final = recovery.verify(paths["recovery_plan"])
    if Path(final["reports"]["comparison"]["path"]).resolve() != paths["comparison"]:
        raise ValueError("comparison is not the accepted recovery report")
    plan, _ = recovery.load_plan(paths["recovery_plan"], live=False)
    loader = recovery.amended_result_loader(plan)
    comparison = sealed_json(paths["comparison"], "report_sha256")
    base, receipt, _ = fit_report._snapshot(bound_file(comparison["input_receipt"]))
    if paths["release_manifest"] != Path(receipt["archive_manifest_path"]).resolve():
        raise ValueError("training evidence comes from another release")
    index = composition_campaign._index(paths["m6_index"])
    screen = sealed_json(paths["m6_report"], "report_sha256")
    audit = sealed_json(paths["m6_audit"], "audit_sha256")
    if screen["screen_index_sha256"] != index["index_sha256"] or audit["screen_index_sha256"] != index["index_sha256"] \
            or screen["audit_sha256"] != audit["audit_sha256"] or audit["status"] != "complete":
        raise ValueError("M6 numerical audit/index/report mismatch")
    cr = sealed_json(Path(index["component_receipt"]["path"]), "receipt_sha256")
    for value in cr["sources"].values():
        bound_file(value)
    bound_file(cr["snapshot"])
    if cr["source_input_sha256"] != receipt["fit_rows_content_sha256"]:
        raise ValueError("M6 and baseline fit inputs differ")
    source = historical_source(profile, cr)
    frame = aligned_components(base, composition_analysis._component_frame(index))
    expected_memberships = {}
    inputs, checks, frames = [], [], []
    for item in comparison["source_results"]:
        result = loader(bound_file(item))
        model, fold = item["model"], item["fold"]
        manifest = execution.load_manifest(Path(result["manifest_path"]))
        if model == "M2":
            expected_memberships[fold] = result["membership"]
        audit_evidence = item["audit"] if "audit_amendment" not in result else {
            "original_preserved_audit": item["audit"], "accepted_refit": result["audit_amendment"]}
        inputs.append((model, fold, result, manifest, comparison["primary_metrics"][model][fold], audit_evidence))
    if {(m, f) for m, f, *_ in inputs} != {(m, f) for m in MODELS for f in FOLDS}:
        raise ValueError("incomplete baseline cell matrix")
    for fold in FOLDS:
        result, manifest = composition_campaign._campaign_result(Path(index["manifests"]["M6"]["0.01"][fold]), index)
        matching = validate_m6_audit(audit, result, fold)
        if manifest["membership"] != expected_memberships[fold] or manifest["provenance"]["source_sha256"] != source["core_source_sha256"]:
            raise ValueError("M6 membership or historical predictor source mismatch")
        inputs.append(("M6", fold, result, manifest, screen["candidate_metrics"]["M6@lambda=0.01"][fold], matching))
    for model, fold, result, manifest, expected, numerical_audit in inputs:
        if manifest["membership"] != expected_memberships[fold]:
            raise ValueError("baseline primary memberships differ")
        train_objective = None
        for split in ("train", "test"):
            rows = annotate(predict_rows(frame, manifest["membership"], result["selected"], model, split),
                model=model, fold=fold, split=split, delta=manifest["fit"]["delta"])
            if split == "test":
                difference = reproduce_score(rows, expected)
            else:
                train_objective = float(np.dot(rows.weight, rows.huber_loss))
                if model == "M6":
                    penalty, _ = fit.regularization_objective_and_gradient(model, result["selected"]["coordinates"],
                        fit.M5M6Regularization.from_mapping(manifest["fit"]["regularization"]))
                    train_objective += penalty
                if abs(train_objective - result["selected"]["objective"]) > SCORE_TOLERANCE:
                    raise ValueError(f"training objective reproduction failed: {model}/{fold}")
            frames.append(rows)
        checks.append({"model": model, "fold": fold, "result_sha256": result["result_sha256"],
            "maximum_score_difference": difference, "training_objective": train_objective,
            "recorded_objective": result["selected"]["objective"], "numerical_audit": numerical_audit})
        print(f"Reproduced {model}/{fold}: max score difference {difference:.3g}", flush=True)
    evidence = {"status": "complete", "score_tolerance": SCORE_TOLERANCE, "reproduction": checks,
        "m6_historical_source": source, "input_sha256": receipt["fit_rows_content_sha256"],
        "selected_m6_lambda": 0.01, "new_fits": 0}
    return pd.concat(frames, ignore_index=True), evidence, receipt


def release_table(manifest_path: Path, name: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    archive = json.loads(manifest_path.read_text())
    item = archive["table_inventory"].get(name)
    if item is None:
        raise ValueError(f"release has no table: {name}")
    path = bound_file({"path": item["path"], "file_sha256": item["sha256"]}, root=manifest_path.parent)
    return pd.read_parquet(path, columns=columns)


def prefix_and_branch(by_job: dict, empty: pd.DataFrame, *, job_id: str,
        parent_id: str | None, branch_start: float, readout_end: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    own = by_job.get(job_id, empty)
    own = own[own.consumed_tokens <= readout_end]
    prefix = by_job.get(str(parent_id), empty) if pd.notna(parent_id) and parent_id else empty
    prefix = prefix[prefix.consumed_tokens <= min(float(branch_start), readout_end)] if pd.notna(branch_start) else empty
    return prefix, own


def training_evidence(manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    """Use release-bound parsed logs; never include a parent's future tokens."""
    readouts = release_table(manifest_path, "readouts")
    archive = json.loads(manifest_path.read_text())
    unavailable = []

    def optional(name: str, columns: list[str]) -> pd.DataFrame:
        entry = archive["table_inventory"].get(name)
        if entry is None or not (manifest_path.parent / entry["path"]).is_file():
            unavailable.append(name)
            return pd.DataFrame(columns=columns)
        return release_table(manifest_path, name, columns=columns)

    configs = optional("training_configs", ["training_job_id", "warmup_tokens", "decay_fraction", "schedule_mode"]).set_index("training_job_id")
    jobs = optional("training_jobs", ["training_job_id", "parent_training_job_id", "peak_learning_rate"]).set_index("training_job_id")
    segments = optional("training_segments", ["training_job_id", "actual_start_tokens"])
    summary = optional("training_summary", ["training_job_id", "training_metric_coverage_fraction"]).set_index("training_job_id")
    if any(not table.index.is_unique for table in (configs, jobs, summary)):
        raise ValueError("ambiguous training-job metadata")
    columns = ["training_job_id", "consumed_tokens", "learning_rate", "lm_loss", "grad_norm", "skipped_iterations", "nan_iterations"]
    metrics = optional("training_metrics", columns)
    metrics = metrics[metrics.consumed_tokens.notna() & metrics.learning_rate.notna()]
    by_job = {str(k): v for k, v in metrics.groupby("training_job_id", sort=False)}
    starts = segments.groupby("training_job_id").actual_start_tokens.min()
    records = []
    for r in readouts.itertuples(index=False):
        job = jobs.loc[r.training_job_id] if r.training_job_id in jobs.index else None
        cfg = configs.loc[r.training_job_id] if r.training_job_id in configs.index else None
        parent_id = None if job is None else job.parent_training_job_id
        prefix, own = prefix_and_branch(by_job, metrics.iloc[:0], job_id=r.training_job_id,
            parent_id=parent_id, branch_start=starts.get(r.training_job_id, np.nan), readout_end=r.actual_tokens)
        observed = pd.concat([prefix, own], ignore_index=True)
        warmup = np.nan if cfg is None else pd.to_numeric(cfg.warmup_tokens, errors="coerce")
        item = {"readout_id": r.readout_id, "trajectory_id": r.trajectory_id, "ladder": r.ladder,
            "E": r.experts, "K": r.language_count, "D_total": r.actual_tokens,
            "training_job_id": r.training_job_id, "parent_training_job_id": parent_id,
            "configured_peak_lr": np.nan if job is None else job.peak_learning_rate,
            "warmup_tokens": warmup, "warmup_fraction": warmup / r.actual_tokens,
            "configured_decay_fraction": np.nan if cfg is None else pd.to_numeric(cfg.decay_fraction, errors="coerce"),
            "schedule_mode": None if cfg is None else cfg.schedule_mode,
            "branch_start_tokens": starts.get(r.training_job_id, np.nan),
            "observed_prefix_rows": len(prefix), "observed_branch_rows": len(own),
            "observed_peak_lr": observed.learning_rate.max(), "observed_max_grad_norm": observed.grad_norm.max(),
            "observed_max_skipped_counter": observed.skipped_iterations.max(), "observed_max_nan_counter": observed.nan_iterations.max(),
            "observed_skipped_counter_rows": int(observed.skipped_iterations.notna().sum()),
            "observed_nan_counter_rows": int(observed.nan_iterations.notna().sum()),
            "branch_log_coverage": summary.loc[r.training_job_id].training_metric_coverage_fraction if r.training_job_id in summary.index else np.nan,
            "telemetry_status": "present" if len(own) else "missing",
            "configuration_status": "present" if cfg is not None and job is not None else "missing"}
        records.append(item)
    result = pd.DataFrame(records)
    return result, {"readouts": len(result), "unavailable_tables": unavailable, "missing_telemetry": int(result.telemetry_status.eq("missing").sum()),
        "missing_configuration": int(result.configuration_status.eq("missing").sum()),
        "source": "release-bound training configs, segments, summaries and parsed training metrics",
        "scope": "Observed prefix stops at branch fork; branch stops at readout. Counters are maxima, not sums. "
                 "Missing counter observations are not evidence of zero anomalies. No LR intervention or new seed is available."}


def run(profile_path: Path, *, dry_run: bool = False) -> dict:
    from result_analysis.scaling_law_fit import residuals as analysis
    profile, paths = load_profile(profile_path)
    output = Path(profile["output_root"])
    if output.exists() and not dry_run:
        raise ValueError(f"create-only diagnostic root already exists: {output}")
    predictions, evidence, _ = collect_predictions(profile, paths)
    if dry_run:
        return {"status": "validated_dry_run", "output_created": False, "prediction_rows": len(predictions),
            "reproduced_cells": len(evidence.get("reproduction", [])), "new_fits": 0,
            "maximum_score_difference": max((v["maximum_score_difference"] for v in evidence.get("reproduction", [])), default=0.)}
    print("Reading release-bound training telemetry and calculating residual panels", flush=True)
    telemetry, coverage = training_evidence(paths["release_manifest"])
    tables, statistical_summary = analysis.analyze(predictions, replicates=profile["bootstrap_replicates"], seed=profile["bootstrap_seed"])
    source_paths = [Path(__file__), Path(analysis.__file__)]
    sources = {str(p.relative_to(repository_root())): sha256_file(p) for p in source_paths}
    sources.update({str(Path(fit.__file__).parent / name): sha256_file(Path(fit.__file__).parent / name)
                   for name in execution.SOURCE_FILES})
    report = {"schema_version": SCHEMA, "status": "complete", "review_only": True,
        "selects_model": False, "new_fits": 0, "profile": {"path": str(profile_path.resolve()), "file_sha256": sha256_file(profile_path)},
        "inputs": profile["inputs"], "analysis_sources": sources, "evidence": evidence,
        "training_coverage": coverage, "statistics": statistical_summary,
        "claim_boundary": "Post-hoc diagnostics on already inspected holdouts. Whole trajectories are resampling units; "
                          "intervals are conditional on fitted coefficients and do not account for model search or seed uncertainty."}
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "residuals.parquet", index=False)
    telemetry.to_csv(output / "training_evidence.csv", index=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    analysis.write_plots(output, predictions, tables)
    execution.write_text_once(output / "metrics.md", analysis.markdown(report, tables))
    report["outputs"] = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    execution.write_json_once(output / "report.json", seal(report, "report_sha256"))
    return {"status": "complete", "report": str(output / "report.json"), "new_fits": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.profile, dry_run=args.dry_run), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
