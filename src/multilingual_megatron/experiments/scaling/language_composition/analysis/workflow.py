"""Prepare, analyze, verify and package the completed composition side study.

The preparation freezes the specification and existing coefficients before new
prediction scores are computed. Numerical replay uses only portable inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from . import analysis_models as models, analysis_statistics as stats
from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_data import Reader, extract_exposures, extract_losses, sha256, write_json
from multilingual_megatron.experiments.scaling.language_composition.study_contracts import HORIZONS, SEEDS

from multilingual_megatron.repo import find_repo_root

REPO = find_repo_root(Path(__file__))
DEFAULT_SPEC = "configs/stage1_fit/language_composition_analysis_v1.json"
DEFAULT_OUTPUT = "result_analysis/stage1_language_composition_v1"


def inventory(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def prepare(spec_path, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    reader = Reader(REPO)
    spec = reader.read(spec_path)
    if spec["schema"] != "language_composition_analysis/v1":
        raise ValueError("unknown analysis specification")
    selected = models.frozen_models(spec, reader.read)
    write_json(inputs / "spec.json", spec)
    write_json(inputs / "models.json", selected)
    write_json(output / "freeze.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": sha256(inputs / "spec.json"),
        "models_sha256": sha256(inputs / "models.json"),
        "status": spec["prediction_analysis_status"],
        "scores_computed": False,
    })
    manifest = reader.read(spec["study_manifest"])
    if manifest["manifest_sha256"] != spec["study_manifest_sha256"]:
        raise ValueError("wrong study release")
    root = reader.path(manifest["output_root"])
    summary = reader.read(root / "control/language_composition_summary.json")
    if summary["status"] != "complete" or len(summary["evaluation_evidence"]) != 162:
        raise ValueError("study summary is not complete")
    from multilingual_megatron.experiments.scaling.ab_v2.artifacts import _verified_receipt_evidence
    from multilingual_megatron.experiments.scaling.ab_v2.variant import variant_scope
    from multilingual_megatron.experiments.scaling.language_composition.training.training_workflow import VARIANT
    with variant_scope(VARIANT):
        _verified_receipt_evidence(manifest, "language_composition_summary")
    cohort_path = next(p for p in manifest["fixed_input_sha256"] if p.endswith("composition_cohorts.json"))
    cohort_payload = reader.read(cohort_path, manifest["fixed_input_sha256"][cohort_path])
    cohorts = {k: v["languages"] for k, v in cohort_payload["cohorts"].items()}
    if sorted(cohorts["all_three"]) != spec["shared_languages"] or len(cohorts["unseen_177"]) != 177:
        raise ValueError("frozen cohort membership mismatch")
    write_json(inputs / "cohorts.json", cohorts)
    write_json(inputs / "original_summary.json", summary)
    losses, parity = extract_losses(manifest, summary, reader, spec)
    stats.validate_losses(losses)
    losses.to_parquet(inputs / "losses.parquet", index=False)
    parity.to_parquet(inputs / "reference_parity.parquet", index=False)
    metadata = reader.read(spec["inputs"]["component_receipt"])["language_metadata"]
    exposures, support, stream_checks = extract_exposures(manifest, reader, metadata)
    exposures.to_parquet(inputs / "exposures.parquet", index=False)
    write_json(inputs / "composition_support.json", support)
    write_json(inputs / "stream_checks.json", stream_checks)
    history = pd.read_parquet(reader.bind(spec["inputs"]["historical_predictions"]))
    history = history[history.model.isin(spec["models"])].copy()
    history.to_parquet(inputs / "historical_predictions.parquet", index=False)
    validation = pd.read_parquet(reader.bind(spec["inputs"]["historical_validation"]))
    validation = validation[validation.language.isin(spec["shared_languages"])]
    normalization = []
    for language in spec["shared_languages"]:
        old = validation[validation.language.eq(language)]
        current = losses[losses.language.eq(language)]
        for key, new_key in (("target_token_ids_sha256", "target_hash"), ("token_count", "token_count")):
            if old[key].nunique() != 1 or old[key].iloc[0] != current[new_key].iloc[0]:
                raise ValueError("old/new target identities differ")
        old_u = -old.unigram_log_probability_sum / old.token_count
        if old_u.nunique() != 1 or old.unigram_counts_sha256.nunique() != 1:
            raise ValueError("ambiguous historical loss coordinate")
        normalization.append({"language": language, "old_unigram_nll": float(old_u.iloc[0]),
            "new_unigram_nll": float(current.unigram_nll.iloc[0]),
            "old_unigram_hash": old.unigram_counts_sha256.iloc[0],
            "new_unigram_hash": current.unigram_hash.iloc[0]})
    write_json(inputs / "normalization.json", normalization)
    # The observed B recipe supplies the same active-parameter coordinate used in fitting.
    architecture = history[history.ladder.eq("B")][["E", "N_act"]].drop_duplicates()
    if architecture.E.duplicated().any():
        raise ValueError("ambiguous historical B active-parameter accounting")
    architecture.to_csv(inputs / "architecture.csv", index=False)
    write_json(inputs / "manifest.json", {"files": inventory(inputs), "source_bindings": reader.bindings,
        "scope": "portable sufficient statistics; original routing counters are excluded"})
    print(f"Prepared and frozen: {output}", flush=True)


def load_inputs(output):
    inputs = output / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(inputs / name) != digest:
            raise ValueError(f"portable input changed: {name}")
    freeze = json.loads((output / "freeze.json").read_text())
    for name, key in (("spec.json", "spec_sha256"), ("models.json", "models_sha256")):
        if sha256(inputs / name) != freeze[key]:
            raise ValueError("analysis specification or coefficients changed after freezing")
    return inputs, json.loads((inputs / "spec.json").read_text())


def compute(output):
    inputs, spec = load_inputs(output)
    losses = pd.read_parquet(inputs / "losses.parquet")
    cohorts = json.loads((inputs / "cohorts.json").read_text())
    cells, empirical = stats.empirical(losses, cohorts)
    original = json.loads((inputs / "original_summary.json").read_text())
    for horizon in HORIZONS:
        block = empirical[empirical.cohort.eq("unseen_177") & empirical.horizon.eq(horizon)].sort_values("seed")
        expected = original["descriptive_all_horizons"][str(horizon)]
        for row, previous in zip(block.to_dict("records"), expected["seedwise"], strict=True):
            if row["seed"] != previous["seed"]:
                raise ValueError("original empirical seed ordering changed")
            for key in ("reference_moe_gain", "compact_moe_gain", "diverse_moe_gain", "paired_diverse_minus_compact_gain"):
                if abs(row[key] - previous[key]) > spec["summary_tolerance"]:
                    raise ValueError("original empirical summary did not reproduce")
        measured = stats.describe(block.paired_diverse_minus_compact_gain)
        for key in ("mean", "sample_standard_deviation", "range"):
            if not np.allclose(measured[key], expected["paired_contrast_summary"][key], rtol=0, atol=spec["summary_tolerance"]):
                raise ValueError("original seed summary did not reproduce")
    records = json.loads((inputs / "models.json").read_text())
    historical = pd.read_parquet(inputs / "historical_predictions.parquet")
    replay = []
    for record in records:
        old = historical[historical.model.eq(record["model"]) & historical.fold.eq(record["fold"])]
        delta = float(np.max(np.abs(models.predict(old, record) - old.prediction)))
        if delta > spec["historical_prediction_tolerance"]:
            raise ValueError(f"historical prediction replay failed: {record['model']}/{record['fold']}: {delta}")
        replay.append({"model": record["model"], "fold": record["fold"], "rows": len(old), "maximum_error": delta})
    exposure = pd.read_parquet(inputs / "exposures.parquet")
    if len(exposure) != 1296 or exposure.D_script.ne(0).any():
        raise ValueError("unexpected exposure coverage or script support; interpretation must be reviewed")
    scoring = losses[losses.arm.isin(["compact", "diverse"]) & losses.language.isin(spec["shared_languages"])].copy()
    scoring = scoring.merge(exposure, on=stats.CELL + ["language"], validate="one_to_one", how="left")
    if scoring.D_t.isna().any():
        raise ValueError("missing observed exposure")
    normal = pd.DataFrame(json.loads((inputs / "normalization.json").read_text()))
    scoring = scoring.merge(normal[["language", "old_unigram_nll"]], on="language", validate="many_to_one")
    scoring = scoring.merge(pd.read_csv(inputs / "architecture.csv"), on="E", validate="many_to_one")
    predicted = []
    for record in records:
        rows = scoring.copy()
        rows["prediction_old_adjusted"] = models.predict(rows, record)
        rows["prediction_raw"] = rows.prediction_old_adjusted + rows.old_unigram_nll
        rows["prediction"] = rows.prediction_raw - rows.unigram_nll
        rows["model"], rows["fold"] = record["model"], record["fold"]
        predicted.append(rows)
    predictions = pd.concat(predicted, ignore_index=True)
    metrics, seeds, contrasts, differences = stats.prediction_metrics(predictions, spec)
    primary = seeds[seeds.horizon.eq(spec["primary_horizon"])]
    difference_seeds = differences.groupby(["seed", "horizon"]).full_minus_coverage.mean().reset_index()
    primary_differences = difference_seeds[difference_seeds.horizon.eq(spec["primary_horizon"])].sort_values("seed")
    report = {"schema": "language_composition_analysis_result/v1", "status": "complete",
        "loss_observations": len(losses), "evaluation_count": len(losses[stats.CELL].drop_duplicates()),
        "prediction_observations_per_fit": len(scoring), "frozen_fit_count": len(records),
        "historical_replay": replay,
        "original_primary": original["primary"],
        "primary_predictive_scores": {name: stats.describe(block.sort_values("seed").contrast_rmse)
                                      for name, block in primary.groupby("model")},
        "full_minus_coverage_seedwise": primary_differences.to_dict("records"),
        "full_minus_coverage_summary": stats.describe(primary_differences.full_minus_coverage),
        "interpretation": spec["prediction_analysis_status"],
        "independent_unit": "three model-seed blocks; historical fits are overlapping sensitivity analyses",
        "script_support": "zero D_script for all 24 new-arm trained target/arm pairs; script coefficient untested"}
    return report, {"empirical_cells": cells, "empirical_contrasts": empirical, "predictions": predictions,
        "fit_metrics": metrics, "seed_metrics": seeds, "prediction_contrasts": contrasts,
        "paired_fit_differences": differences, "paired_seed_differences": difference_seeds}


def analyze(output):
    if (output / "completion.json").exists():
        raise ValueError("completed analysis is immutable; use verify")
    report, frames = compute(output)
    derived = output / "derived"
    derived.mkdir(exist_ok=True)
    for name, frame in frames.items():
        if name in ("predictions", "prediction_contrasts"):
            frame.to_parquet(derived / f"{name}.parquet", index=False)
        else:
            frame.to_csv(derived / f"{name}.csv", index=False, float_format="%.17g")
    write_json(derived / "report.json", report)
    from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_reporting import export
    export(output, report, frames)
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "pyarrow", "PyYAML")}
    # Bind the executed package dependency closure, not unrelated concurrent work.
    loaded = {Path(module.__file__).resolve() for module in list(sys.modules.values())
              if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")}
    source = {str(p.relative_to(REPO)): sha256(p) for p in sorted(loaded)
              if p.is_relative_to(REPO / "src/multilingual_megatron")}
    write_json(output / "completion.json", {"schema": "language_composition_analysis_completion/v1",
        "status": "complete", "inputs_manifest_sha256": sha256(output / "inputs/manifest.json"),
        "freeze_sha256": sha256(output / "freeze.json"), "derived": inventory(derived),
        "source_files": source, "packages": packages})
    print(json.dumps(report["full_minus_coverage_summary"], indent=2), flush=True)


def verify(output):
    completion = json.loads((output / "completion.json").read_text())
    if sha256(output / "inputs/manifest.json") != completion["inputs_manifest_sha256"] or sha256(output / "freeze.json") != completion["freeze_sha256"]:
        raise ValueError("frozen preparation changed")
    for name, digest in completion["derived"].items():
        if sha256(output / "derived" / name) != digest:
            raise ValueError(f"derived artifact changed: {name}")
    report, frames = compute(output)
    if report != json.loads((output / "derived/report.json").read_text()):
        raise ValueError("numerical report replay mismatch")
    for name, frame in frames.items():
        path = output / "derived" / name
        old = pd.read_parquet(path.with_suffix(".parquet")) if name in ("predictions", "prediction_contrasts") else pd.read_csv(path.with_suffix(".csv"), float_precision="round_trip")
        # CSV does not serialize the optional name of the columns Index.
        frame = frame.reset_index(drop=True).rename_axis(None, axis="columns")
        pd.testing.assert_frame_equal(frame, old, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    print(f"Verified all results and {sum(r['rows'] for r in report['historical_replay']):,} historical predictions", flush=True)


def package(output, destination):
    verify(output)
    completion = json.loads((output / "completion.json").read_text())
    if destination.exists():
        raise ValueError("refusing to overwrite a replay archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(output)))
        for relative, digest in completion["source_files"].items():
            source = REPO / relative
            if sha256(source) != digest:
                raise ValueError(f"analysis source changed before packaging: {relative}")
            archive.write(source, relative)
        archive.writestr("requirements.txt", "\n".join(f"{k}=={v}" for k, v in completion["packages"].items()) + "\n")
        archive.writestr("REPLAY.md", "# Composition analysis replay\n\n"
            "Install requirements.txt, then from this extracted directory run:\n\n"
            "    PYTHONPATH=src python -m multilingual_megatron.experiments.scaling.language_composition.analysis.workflow verify --output .\n\n"
            "This replays frozen predictions and statistics without training, optimizer refits, original corpora or cluster access. "
            "Input source paths record provenance; replay uses the included sufficient statistics.\n")
    print(f"Replay archive: {destination}; sha256={sha256(destination)}", flush=True)


def export_paper(output, paper):
    """Copy only verified, deterministic manuscript assets; never overwrite drift."""
    verify(output)
    import shutil
    for path in sorted((output / "derived").iterdir()):
        if path.suffix not in {".pdf", ".tex"}:
            continue
        category = "figures" if path.suffix == ".pdf" else "tables"
        destination = paper / category / "language_composition_v1" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(path):
            raise ValueError(f"paper asset already differs: {destination}")
        shutil.copyfile(path, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "analyze", "verify", "package", "export-paper"))
    parser.add_argument("--spec", type=Path, default=Path(DEFAULT_SPEC))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--paper", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(args.spec, output)
    elif args.action == "analyze":
        analyze(output)
    elif args.action == "verify":
        verify(output)
    elif args.action == "export-paper":
        if args.paper is None:
            parser.error("export-paper requires --paper")
        export_paper(output, args.paper.resolve())
    elif args.archive is None:
        parser.error("package requires --archive")
    else:
        package(output, args.archive.resolve())


if __name__ == "__main__":
    main()
