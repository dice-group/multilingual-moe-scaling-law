"""Export measured Stage 1 results as portable, ordinary parquet tables.

Run with --config configs/result_analysis/stage1_data_package.json.
The exporter reads sealed observations; it never runs training or model scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from multilingual_megatron.experiments.scaling.stage1_release.exposure import (
    _exact_exposures, _resolve_training_inputs,
)

KEYS = ["trajectory_id", "readout_id", "language"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


class Export:
    def __init__(self, config):
        self.config = config
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=False)
        self.sources, self.tables = {}, {}
        self.benchmark_revision = None
        self.fit_revision = None

    def source(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.sources:
            self.sources[key] = digest(path)
        if expected and self.sources[key] != expected:
            raise ValueError(f"Source checksum mismatch: {path}")
        return path

    def json(self, path, expected=None):
        return json.loads(self.source(path, expected).read_text())

    def read(self, path):
        path = self.source(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)

    def write(self, name, frame, grain, keys=None):
        if keys and frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} keys: {keys}")
        path = self.output / f"{name}.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, path, compression="zstd")
        # Numerical round trips are exact; the source values are never rounded.
        if not table.equals(pq.read_table(path)):
            raise ValueError(f"Parquet round-trip changed values: {name}")
        self.tables[name] = {"rows": len(frame), "bytes": path.stat().st_size,
                            "sha256": digest(path), "grain": grain, "keys": keys,
                            "columns": {c: str(t) for c, t in frame.dtypes.items()}}
        print(f"{name}: {len(frame):,} rows, {path.stat().st_size / 2**20:.2f} MiB", flush=True)


def join_columns(left, right, keys):
    """Add only new columns, enforcing a many-to-one join and complete coverage."""
    cols = keys + [c for c in right if c not in left]
    result = left.merge(right[cols], on=keys, how="left", validate="many_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError(f"Missing join keys: {keys}")
    return result.drop(columns="_merge")


def portable(frame):
    """Drop deployment locations, retaining numerical semantics and identity hashes."""
    return frame.drop(columns=[c for c in frame if c.endswith("_path") or c.endswith("_paths")
        or c in {"checkpoint_root", "language_artifact", "lineage_records_json",
                 "parameter_counts_json", "telemetry_gpu_uuids_json", "corpus_inventory_json",
                 "runtime_rank_local_parameter_observations", "runtime_rounded_global_parameter_observations"}],
        errors="ignore")


def main_grid(export):
    root = Path(export.config["release"]) / "tables"
    read = lambda name: export.read(root / f"{name}.parquet")
    grid = export.read(Path(export.config["evidence"]) / "scoring/grid.parquet")
    val, readouts, trajectories = read("validation_language"), read("readouts"), read("trajectories")
    geometry = grid[["ladder", "E", "N_act", "N_act_nonembedding", "N_total"]].drop_duplicates()
    frame = val.rename(columns={"experts": "E", "language_count": "K", "actual_tokens": "D_total",
                               "atlas_tao_unigram_adjusted_loss": "adjusted_loss"})
    frame = join_columns(frame, trajectories[["trajectory_id", "model_seed", "dataset_shuffle_seed"]], ["trajectory_id"])
    frame = join_columns(frame, geometry, ["ladder", "E"])
    features = [c for c in grid if c not in frame or c in KEYS]
    frame = frame.merge(grid[features], on=KEYS, how="left", validate="one_to_one")
    frame["unigram_nll"] = -frame.unigram_log_probability_sum / frame.token_count
    frame["validation_token_count"] = frame.token_count
    frame["experiment"] = "main_grid"
    frame = join_columns(frame, readouts[["readout_id", "training_job_id", "checkpoint_step"]], ["readout_id"])
    attribution = read("readout_compute_attribution")
    frame = join_columns(frame, attribution, ["readout_id"])
    summary, costs = read("training_summary"), read("unique_physical_job_compute")
    runs = join_columns(summary, costs, ["training_job_id"])
    # MFU and throughput describe the measured training job (usually cooldown).
    # Readout GPU-hours separately include an attributed shared trunk prefix.
    run_columns = ["training_job_id", "gpu_count", "mean_mfu_pct_of_configured_peak",
                   "peak_tflops_per_gpu", "mean_tokens_per_second", "median_step_ms",
                   "mfu_semantics", "throughput_semantics", "max_training_allocated_gib"]
    frame = join_columns(frame, runs[run_columns], ["training_job_id"])
    np.testing.assert_allclose(frame.raw_nll, frame.nll_sum/frame.token_count, rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame.adjusted_loss, frame.raw_nll-frame.unigram_nll, rtol=0, atol=1e-12)
    observed = frame.loc[frame.metric_role.eq("trained")].set_index(KEYS).sort_index()
    expected = grid.set_index(KEYS).sort_index()
    pd.testing.assert_index_equal(observed.index, expected.index, exact=False)
    for c in ["raw_nll", "adjusted_loss", "D_t", "D_total", "N_act", "N_total", "U_t"]:
        np.testing.assert_allclose(observed[c], expected[c], rtol=0, atol=1e-12)
    if len(frame) != 46200 or len(observed) != 26300 or frame.readout_id.nunique() != 400:
        raise ValueError("Primary grid population changed")
    export.write("main_grid", portable(frame), "one checkpoint x validation language", KEYS)
    export.write("training_runs", portable(runs), "one physical training job; additive GPU-hour totals", ["training_job_id"])
    steps = read("training_metrics")
    steps = steps.loc[steps.lm_loss.notna()].copy()
    # Keep repeated observations from restarts identifiable; do not silently choose a log.
    steps["source_id"] = pd.factorize(steps.source_path, sort=True)[0]
    steps["repeated_job_segment_iteration"] = steps.duplicated(["training_job_id", "segment_id", "iteration"], keep=False)
    export.write("training_steps", portable(steps).dropna(axis=1, how="all"),
                 "one recorded training iteration, including identifiable restart duplicates")
    export.write("model_configs", portable(read("model_configs")), "one architecture x expert count", ["model_config_id"])
    export.write("languages", portable(read("languages")), "one main-corpus language", ["language"])
    export.write("language_membership", portable(read("language_tier_membership")), "one nested language-set membership")
    export.write("hardware", portable(read("hardware_contract")), "recorded MFU denominator and hardware contract")
    return frame, geometry


def seed_study(export, main, geometry):
    root = Path(export.config["seed_study"])
    summary = export.json(root / "control/seed_confirmation_summary.json")
    if summary["status"] != "complete" or summary["errors"] or len(summary["results"]) != 108:
        raise ValueError("Incomplete seed study")
    rows = []
    for number, result in enumerate(summary["results"]):
        ep = root / "jobs" / result["job_id"] / "evaluation_manifest.json"
        ev = export.json(ep)
        if ev["status"] != "complete":
            raise ValueError(f"Incomplete seed readout: {ep}")
        tp, index, description, training = _resolve_training_inputs(ep)
        export.source(tp)
        actual = result["actual_end_tokens"]
        seq = training["metadata"]["seq_length"]
        tier = Path(training["run_cfg"]["data_manifest"])
        export.source(tier)
        if result["K"] == 1:
            exposures = {next(iter(export.json(tier)["prefixes_by_language"])): actual}
        else:
            export.source(index); export.source(description)
            exposures = _exact_exposures(index_path=index, description_path=description,
                tier_manifest_path=tier, end_samples=actual//seq, sequence_length=seq)
        for language, item in ev["languages"].items():
            p = Path(item["path"])
            if not p.is_absolute(): p = ep.parent / p
            raw = export.json(p, item["sha256"])
            rows.append(dict(experiment="seed_comparison", ladder="B", E=result["experts"], K=result["K"],
                model_seed=result["model_seed"], dataset_shuffle_seed=result["dataset_random_seed"],
                trajectory_id=f"seed_B_E{result['experts']}_K{result['K']}_s{result['model_seed']}",
                readout_id=result["job_id"], training_job_id=result["training_job_id"],
                checkpoint_step=ev["checkpoint"]["step"], nominal_tokens=result["D"], D_total=actual,
                language=language, metric_role="trained" if language in exposures else "zero_shot",
                D_t=exposures.get(language), token_count=raw["token_count"],
                validation_token_count=raw["token_count"], nll_sum=raw["nll_sum"], raw_nll=raw["raw_nll"],
                adjusted_loss=raw["atlas_tao_unigram_adjusted_loss"],
                unigram_nll=-raw["unigram_log_probability_sum"]/raw["token_count"],
                unigram_log_probability_sum=raw["unigram_log_probability_sum"],
                target_token_ids_sha256=raw["target_token_ids_sha256"],
                unigram_counts_sha256=raw["unigram_counts_sha256"], source_sha256=export.sources[str(p.resolve())]))
        if (number+1) % 18 == 0: print(f"seed readouts: {number+1}/108", flush=True)
    extra = join_columns(pd.DataFrame(rows), geometry, ["ladder", "E"])
    original = main[main.ladder.eq("B") & main.E.isin([1, 16, 32]) & main.K.isin([1, 12, 200])].copy()
    original["experiment"] = "seed_comparison"
    original["reused_main_grid"] = True
    extra["reused_main_grid"] = False
    cols = list(extra.columns)
    result = pd.concat([original.reindex(columns=cols), extra], ignore_index=True)
    if result.readout_id.nunique() != 162:
        raise ValueError("Expected 162 seed-study readouts")
    export.write("seed_comparison", result, "one seed-study checkpoint x language; seed 1234 reuses main-grid observations", KEYS)


def composition(export, geometry):
    root = Path(export.config["composition"])
    losses, exposure = export.read(root/"losses.parquet"), export.read(root/"exposures.parquet")
    frame = losses.merge(exposure, on=["arm", "seed", "E", "horizon", "language"], how="left", validate="one_to_one")
    # All evaluated languages are retained; only the trained 12 have exposure rows.
    frame["metric_role"] = np.where(frame.D_t.notna(), "trained", "untrained")
    totals = exposure[["arm", "seed", "E", "horizon", "D_total", "K", "dataset_seed"]].drop_duplicates()
    # The reference arm re-evaluates existing B/K12 training trajectories.
    # Its exposure is therefore supplied by the main/seed observations, not by
    # the new-mixture exposure file (which contains compact/diverse only).
    seed = pd.read_parquet(export.output / "seed_comparison.parquet").query("K == 12")
    reference = seed.rename(columns={"model_seed": "seed", "nominal_tokens": "horizon",
                                    "dataset_shuffle_seed": "dataset_seed"})
    reference["arm"] = "reference"
    refkeys = ["arm", "seed", "E", "horizon", "language"]
    refexposure = reference.loc[reference.metric_role.eq("trained"), refkeys+["D_t"]]
    frame = frame.merge(refexposure.rename(columns={"D_t": "reference_D_t"}), on=refkeys,
                        how="left", validate="one_to_one")
    frame["D_t"] = frame.D_t.fillna(frame.reference_D_t)
    frame = frame.drop(columns="reference_D_t")
    frame["metric_role"] = np.where(frame.D_t.notna(), "trained", "untrained")
    totals = pd.concat([totals, reference[totals.columns].drop_duplicates()], ignore_index=True)
    frame = frame.drop(columns=["D_total", "K", "dataset_seed"]).merge(totals,
        on=["arm", "seed", "E", "horizon"], how="left", validate="many_to_one")
    if len(frame) != len(losses) or frame.D_total.isna().any():
        raise ValueError("Composition observations or token horizons lost")
    frame = frame.rename(columns={"seed": "model_seed", "job_id": "readout_id", "horizon": "nominal_tokens"})
    frame["ladder"] = "B"
    frame["trajectory_id"] = [f"composition_{a}_E{e}_s{s}" for a,e,s in zip(frame.arm,frame.E,frame.model_seed)]
    frame["experiment"] = "language_composition"
    frame = join_columns(frame, geometry, ["ladder", "E"])
    export.write("language_composition", frame, "one composition arm x seed x checkpoint x evaluated language", KEYS)
    export.write("composition_normalization", pd.DataFrame(export.json(root/"normalization.json")),
                 "recorded old/new unigram reference per common language", ["language"])


def fitting(export):
    root = Path(export.config["evidence"])
    export.write("large_model", export.read(root/"scoring/endpoint.parquet"), "one L26 endpoint x language", KEYS)
    predictions = []
    for campaign, file in [("final_law", "law/report/predictions.parquet"),
        ("baselines", "baselines/predictions.parquet"), ("coverage_ablation", "coverage/predictions.parquet"),
        ("expert_ablation", "expert/predictions.parquet")]:
        df = export.read(root/file); df.insert(0, "fit_campaign", campaign); predictions.append(df)
    export.write("fit_predictions", pd.concat(predictions, ignore_index=True),
                 "one fitted model x fold x evaluated observation; campaigns retain separate identities")
    statuses = []
    for campaign, directory in [("baselines", "baselines"), ("coverage_ablation", "coverage"),
                                ("expert_ablation", "expert")]:
        status = export.read(root/directory/"metrics.csv")
        status = status.drop(columns=["rmse", "mae", "bias", "max_abs_error", "rn_beats"])
        status.insert(0, "fit_campaign", campaign)
        statuses.append(status)
    parameters = []
    def add(model, fold, selected, source):
        for i, value in enumerate(selected["coordinates"]):
            parameters.append(dict(model=model, fold=fold, scope="shared", language=None,
                parameter=f"coordinate_{i}", coordinate_index=i, value=float(value), source=source))
        for row in selected["language_coefficients"]:
            for name in ("A", "B", "L_inf", "floor_fraction"):
                if name in row:
                    parameters.append(dict(model=model, fold=fold, scope="language", language=row["language"],
                        parameter=name, coordinate_index=None, value=float(row[name]), source=source))
    for path in sorted((root/"law/results").glob("*.json")):
        result = export.json(path)
        if not result["complete"] or not result["certified"]:
            raise ValueError(f"Uncertified final fit: {path}")
        add(result["form"], result["fold"], result["selected"], str(path.relative_to(root)))
        statuses.append(pd.DataFrame([dict(fit_campaign="final_law", model=result["form"],
            fold=result["fold"], case="L26" if result["fold"] == "all_grid" else "grid",
            certified=result["certified"], complete=result["complete"],
            role="paper law" if result["form"] == "RN14-stored-reuse" else "historical inner selection")]))
    for fold, result in export.json(root/"allocation/frozen_coefficients.json").items():
        add("M6-ND", fold, result, "allocation/frozen_coefficients.json")
    export.write("fit_parameters", pd.DataFrame(parameters), "one saved RN13/RN14/M6-ND parameter")
    export.write("fit_status", pd.concat(statuses, ignore_index=True),
                 "recorded fitting completeness/certification and population counts; no error summaries",
                 ["fit_campaign", "model", "fold", "case"])
    # Explicit folds preserve membership without loading a historical result table.
    folds = export.json(root/"study/primary_fits/folds.json")
    memberships = []
    for fold, entry in folds.items():
        member = entry["membership"]
        for key, values in member.items():
            if isinstance(values, list):
                memberships.extend(dict(fold=fold, membership_kind=key, member=str(v)) for v in values)
    export.write("fit_membership", pd.DataFrame(memberships), "one recorded fit-fold membership item")
    if export.config.get("fit_update"):
        from .refit_package import export_refit
        export_refit(export, export.output)


def fit_readme(revision):
    if not revision:
        return ""
    return (f"Fitting revision: {revision['revision']}. The {revision['certified_cells']} certified cells "
        f"cover {revision['models']} models. Select fit_revision == '{revision['revision']}', "
        f"fit_campaign == '{revision['fit_campaign']}' and "
        f"arm == '{revision['default_arm']}' for the adopted fits. SLSQP arms are explicit controls; "
        "fit_revision == 'historical' retains previous results. RN13's historical selection is unchanged. "
        "Filter one revision/arm before scoring or using coefficients.\n\n"
        "fit_parameters now includes every refitted shared/native-language coordinate and local amplitude, "
        "with bounds, donor identities and native normalization metadata. fit_status carries the numerical "
        "protocol and certification; fit_membership includes all eight folds and exact readout membership.\n\n")


def apply_fit_metadata(manifest, readme, export):
    manifest["tables"].update(export.tables)
    manifest["bytes"] = sum(t["bytes"] for t in manifest["tables"].values())
    manifest["fit_revision"] = export.fit_revision
    limitation = "Saved coefficients cover all 30 refitted laws and optimizer controls; historical coefficients keep their original scope."
    manifest["limitations"] = [limitation if s.startswith("Saved fit parameters cover") else s for s in manifest["limitations"]]
    readme = readme.replace("Saved fit parameters cover RN13, RN14 and M6-ND; other comparator predictions are retained for rescoring.", limitation)
    readme = readme.replace("Choose fit_campaign explicitly before scoring.",
        "Choose one fit_revision, fit_campaign and arm before scoring.")
    readme = readme.replace("M6-ND supplies the historical allocation predictor.",
        "M6-ND is the separate allocation predictor.")
    for name, record in export.tables.items():
        readme = "\n".join(f"| {name}.parquet | {record['rows']:,} | {record['grain']} |" if line.startswith(f"| {name}.parquet |")
                           else line for line in readme.splitlines()) + "\n"
    readme = readme.split("Total parquet bytes:")[0].split("Fitting revision:")[0]
    return readme + fit_readme(export.fit_revision) + f"Total parquet bytes: {manifest['bytes']:,}.\n"


def refresh_fits(config, output):
    """Refresh the four fit tables from an explicitly reviewed release seal."""
    from . import refit_package
    if not config.get("fit_update"):
        raise ValueError("A pinned fit_update release seal is required")
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["tables"].items():
        if digest(output / f"{name}.parquet") != record["sha256"]:
            raise ValueError(f"Package checksum mismatch: {name}")
    previous = manifest.get("fit_revision", {})
    if previous.get("revision") == config["fit_update"]["revision"] and previous.get("release_seal") != config["fit_update"]:
        raise ValueError("Use a new fit revision for a different release seal")
    if (previous.get("release_seal") == config["fit_update"]
            and previous.get("exporter_sha256") == digest(refit_package.__file__)):
        print("Fitting revision is already current", flush=True)
        return
    with TemporaryDirectory(prefix=".fit-update-", dir=output.parent) as tmp:
        export = Export({**config, "output": str(Path(tmp) / "tables")})
        refit_package.export_refit(export, output)
        manifest["previous_fit_manifest_sha256"] = digest(manifest_path)
        readme = apply_fit_metadata(manifest, ((output / "README.md").read_text() if (output / "README.md").exists() else "Total parquet bytes: 0.\n"), export)
        (export.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        (export.output / "README.md").write_text(readme)
        for name in (*[f"{n}.parquet" for n in export.tables], "README.md", "manifest.json"):
            (export.output / name).replace(output / name)
    print(f"Updated {output}: {export.fit_revision['revision']}", flush=True)


def benchmark(export):
    update = export.config.get("benchmark_update")
    if update:
        receipt = export.json(update["path"], update["sha256"])
        payload = {k: v for k, v in receipt.items() if k != "manifest_sha256"}
        if receipt["manifest_sha256"] != canonical_json_sha256(payload) or receipt["status"] != "passed":
            raise ValueError("Benchmark update requires a verified passing receipt")
        source = receipt["table"]
        frame = export.read(export.source(source["path"], source["sha256"]))
        observed = receipt["observed"]
        if (len(frame) != observed["language_rows"] or frame.model_id.nunique() != observed["models"]
                or frame.benchmark.nunique() != observed["tasks"]
                or not frame.benchmark_revision.eq(receipt["revision"]).all()
                or not np.isfinite(frame.primary).all() or not frame.samples.gt(0).all()):
            raise ValueError("Benchmark update population or metrics differ from the audit")
        export.benchmark_revision = {k: receipt[k] for k in
            ("revision", "observed", "limitations", "campaign", "collection", "metric_sources", "comparison_audit")}
        export.benchmark_revision.update(receipt=update, table_source=source,
            previous_benchmarks_sha256=receipt["previous_benchmarks"]["sha256"],
            exporter_sha256=digest(Path(__file__)))
    else:
        root = Path(export.config["evidence"])
        frame = export.read(root/"benchmark/language_metrics_with_validation.csv")
    export.write("benchmarks", portable(frame), "one checkpoint x benchmark x evaluation language",
                 ["model_id", "benchmark", "language"])


def benchmark_readme(revision):
    if not revision:
        return ""
    return (f"Benchmark revision: {revision['revision']}. Evaluation hardware, precision, batch, padding, "
            "recovery status and receipt hashes are retained in the evaluation_* columns. "
            "evaluation_launcher_gpu_hours describes one successful evaluation job; deduplicate "
            "evaluation_job_id before summing and account for failed attempts separately.\n\n"
            + "\n".join(f"- {s}" for s in revision["limitations"]) + "\n\n")


def refresh_benchmarks(config, output):
    """Apply a reviewed benchmark replacement to an existing portable package."""
    if not config.get("benchmark_update"):
        raise ValueError("A pinned benchmark_update receipt is required")
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["tables"].items():
        if digest(output / f"{name}.parquet") != record["sha256"]:
            raise ValueError(f"Package checksum mismatch: {name}")
    with TemporaryDirectory(prefix=".benchmark-update-", dir=output.parent) as tmp:
        export = Export({**config, "output": str(Path(tmp) / "tables")})
        benchmark(export)
        revision = export.benchmark_revision
        current = manifest["tables"]["benchmarks"]["sha256"]
        if current == export.tables["benchmarks"]["sha256"]:
            print("Benchmark revision is already current", flush=True)
            return
        if current != revision["previous_benchmarks_sha256"]:
            raise ValueError("Audit baseline differs from the package being updated")
        # Observational values may change; scientific identities and denominators may not.
        keys = ["model_id", "benchmark", "language"]
        before = pd.read_parquet(output / "benchmarks.parquet").set_index(keys).sort_index()
        after = pd.read_parquet(export.output / "benchmarks.parquet").set_index(keys).sort_index()
        pd.testing.assert_index_equal(before.index, after.index)
        if not set(before.columns) <= set(after.columns):
            raise ValueError("Benchmark replacement drops existing columns")
        identity = ["E", "K", "ladder", "seed", "samples", "corpus_language", "panel",
                    "D_total", "D_t", "N_act", "N_total", "readout_id", "trained",
                    "in_common12", "is_english", "in_common12_non_english",
                    "validation_nll", "validation_adjusted_loss"]
        pd.testing.assert_frame_equal(before[identity], after[identity], check_exact=True)
        previous_manifest = digest(manifest_path)
        manifest["tables"]["benchmarks"] = export.tables["benchmarks"]
        manifest["bytes"] = sum(t["bytes"] for t in manifest["tables"].values())
        manifest["benchmark_revision"] = revision
        manifest["previous_manifest_sha256"] = previous_manifest
        readme = ((output / "README.md").read_text() if (output / "README.md").exists() else "Total parquet bytes: 0.\n").split("Total parquet bytes:")[0]
        fit_notes = "Fitting revision:" + readme.split("Fitting revision:", 1)[1] if "Fitting revision:" in readme else ""
        readme = readme.split("Benchmark revision:")[0].split("Fitting revision:")[0]
        readme += benchmark_readme(revision) + fit_notes + f"Total parquet bytes: {manifest['bytes']:,}.\n"
        # Prepare every file before replacing the reviewed observation table.
        (export.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        (export.output / "README.md").write_text(readme)
        for name in ("benchmarks.parquet", "README.md", "manifest.json"):
            (export.output / name).replace(output / name)
    print(f"Updated {output}: {revision['revision']}", flush=True)


def routing(export):
    frame = export.read(export.config["routing"])
    counts, streams = [], []
    for (model, language, path), group in frame.groupby(["id", "language", "language_artifact"], sort=False):
        raw = export.json(path, group.language_artifact_sha256.iloc[0])
        metadata = group.iloc[0][["id", "ladder", "experts", "languages", "seed", "tokens_seen"]].to_dict()
        streams.append({**metadata, "language": language, **{k: raw[k] for k in
            ["token_count", "nll_sum", "raw_nll", "atlas_tao_unigram_adjusted_loss",
             "unigram_log_probability_sum", "target_token_ids_sha256", "unigram_counts_sha256"]}})
        by_layer = {r.layer_index:r for r in group.itertuples()}
        for layer in raw["moe_routing"]["layers"]:
            index = layer["layer_index"]
            original = by_layer[index]
            blocks = sorted((int(k), v) for k,v in layer["blocks"].items())
            np.testing.assert_array_equal(np.sum([v for _,v in blocks], axis=0), original.expert_token_counts)
            counts.append({**metadata, "language": language, "layer_index": index,
                "expert_token_counts": list(original.expert_token_counts),
                "padding_expert_token_counts": list(original.padding_expert_token_counts),
                "block_ids": [k for k,_ in blocks], "block_expert_counts": [v for _,v in blocks],
                "router_z_loss": original.router_z_loss, "switch_aux_loss": original.switch_aux_loss,
                "source_sha256": group.language_artifact_sha256.iloc[0]})
        if len(streams) % 500 == 0: print(f"routing streams: {len(streams)}/4008", flush=True)
    export.write("routing_counts", pd.DataFrame(counts), "one checkpoint x language x layer; real-token and bootstrap-block expert counts",
                 ["id", "language", "layer_index"])
    export.write("routing_losses", pd.DataFrame(streams), "one corrected routing checkpoint x language loss", ["id", "language"])


def lr_study(export):
    root = Path(export.config["lr_study"])
    rows = [export.read(root/file).drop(columns=["rank", "loss_regret"], errors="ignore")
            for file in ("h100_results.csv", "a100_reference.csv")]
    export.write("learning_rate", portable(pd.concat(rows, ignore_index=True)),
                 "one cluster x seed x configuration x learning-rate candidate")


def finish(export):
    # Verify the selected upstream tables against their existing seals, in
    # addition to recording their checksums and checking the numeric joins.
    bound_sources = 0
    for key in ("release", "evidence"):
        root = Path(export.config[key])
        name = "release_manifest.json" if key == "release" else "manifest.json"
        upstream = export.json(root/name)
        if key == "release":
            entries = {str((root/"tables"/f"{table}.parquet").resolve()): record["sha256"]
                       for table, record in upstream["table_inventory"].items()}
        else:
            entries = {str((root/table).resolve()): record["sha256"] for table,record in upstream["files"].items()}
        for path, expected in entries.items():
            if path in export.sources:
                if export.sources[path] != expected:
                    raise ValueError(f"Upstream sealed table changed: {path}")
                bound_sources += 1
    total = sum(v["bytes"] for v in export.tables.values())
    manifest = {"schema": "stage1-observations/v1", "status": "complete", "tables": export.tables,
        "upstream_sealed_tables_verified": bound_sources,
        "exporter_sha256": digest(Path(__file__)),
        "source_sha256": export.sources, "bytes": total,
        "scope": "Measured observations and saved fitted predictions/parameters; no analysis contrasts or plots.",
        "limitations": ["Benchmark rows retain language-level scores and counts; paired-example/bootstrap analyses require original item scores.",
            "Saved fit parameters cover RN13, RN14 and M6-ND; other comparator predictions are retained for rescoring.",
            "Training steps and physical cost summaries cover the main A-D grid."]}
    if export.benchmark_revision:
        manifest["benchmark_revision"] = export.benchmark_revision
    (export.output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    lines = ["# Stage 1 numerical data", "", "Read any table directly with pandas.read_parquet. No logs, checkpoints or corpus text are needed.", "",
        "```python", 'from pathlib import Path', 'import pandas as pd', 'data = Path("PATH_TO_THIS_DIRECTORY")',
        'grid = pd.read_parquet(data / "main_grid.parquet")', 'trained = grid.query("metric_role == \'trained\'")', '```', "",
        "| File | Rows | Meaning |", "|---|---:|---|"]
    lines.extend(f"| {name}.parquet | {v['rows']:,} | {v['grain']} |" for name,v in export.tables.items())
    lines += ["", "Losses are nats/token. adjusted_loss = raw_nll - unigram_nll. D_total and D_t are realized tokens; N_act and N_total are parameter counts.",
        "MFU is a percentage of the recorded configured hardware peak. In main_grid, MFU/throughput describe the checkpoint's physical branch job. Logical readout GPU-hours include an attributed shared trunk prefix and MUST NOT be added across readouts. Sum training_runs physical job costs instead.",
        "Main-grid K1 includes untrained validation languages; their training D_t is missing. Filter metric_role == 'trained' for the original 26,300 fitting rows.",
        "Seed 1234 in seed_comparison reuses main_grid and is marked reused_main_grid. Do not count it twice. Composition uses its own arm, corpus/normalization and evaluation identity. Checkpoints and languages within a trajectory are repeated measurements.",
        "routing_counts excludes synthetic padding from expert_token_counts and keeps padding separately; block_expert_counts supports evaluation bootstrap recomputation.",
        "fit_predictions contains overlapping campaigns. Choose fit_campaign explicitly before scoring. RN14-stored-reuse is the paper law; RN13-psi-tied is the historical inner selection; M6-ND supplies the historical allocation predictor.",
        "", "Scope limits:", *[f"- {s}" for s in manifest["limitations"]], "",
        "manifest.json records table schemas and SHA-256 checksums for the exported tables and source files. Source locations are provenance only; loading the package never accesses them.",
        "", benchmark_readme(export.benchmark_revision), f"Total parquet bytes: {total:,}.", ""]
    readme = "\n".join(lines)
    if export.fit_revision:
        readme = apply_fit_metadata(manifest, readme, export)
        (export.output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    (export.output/"README.md").write_text(readme)
    print(f"Complete: {export.output.resolve()} ({total/2**20:.2f} MiB)", flush=True)


def compact(source, output):
    """Keep observation tables, removing bulky optional routing/step detail."""
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    # Validate the complete input before creating an output directory.
    for name, record in manifest["tables"].items():
        if digest(source / f"{name}.parquet") != record["sha256"]:
            raise ValueError(f"Input checksum mismatch: {name}")
    export = Export({"output": str(output)})
    for name, record in manifest["tables"].items():
        frame = pd.read_parquet(source / f"{name}.parquet")
        grain = record["grain"]
        if name == "routing_counts":
            frame = frame.drop(columns=["block_ids", "block_expert_counts"])
            grain = "one checkpoint x language x layer; real-token and padding expert counts"
        elif name == "training_steps":
            frame = frame[["training_job_id", "segment_id", "source_id", "iteration",
                           "consumed_tokens", "lm_loss", "learning_rate",
                           "repeated_job_segment_iteration"]]
        export.write(name, frame, grain, record.get("keys"))
    compact_manifest = {k: v for k, v in manifest.items()
                        if k not in {"source_sha256", "tables", "bytes", "exporter_sha256"}}
    compact_manifest.update(tables=export.tables,
        bytes=sum(v["bytes"] for v in export.tables.values()),
        variant="compact", parent_manifest_sha256=digest(source / "manifest.json"),
        exporter_sha256=digest(Path(__file__)))
    compact_manifest["limitations"] = manifest["limitations"] + [
        "Routing sequence-block counts are omitted; routing bootstrap intervals require the full package.",
        "Training iterations retain LM loss, learning rate and tokens; detailed step telemetry is omitted.",
        "Detailed source-file provenance is in the full package manifest, identified by parent_manifest_sha256."]
    (export.output / "manifest.json").write_text(json.dumps(compact_manifest, indent=2)+"\n")
    readme = ((source / "README.md").read_text() if (source / "README.md").exists() else "Total parquet bytes: 0.\n")
    readme = readme.replace("# Stage 1 numerical data", "# Stage 1 numerical data (compact)")
    readme = readme.replace("; block_expert_counts supports evaluation bootstrap recomputation.",
                           "; sequence-block bootstrap counts are omitted in this compact package.")
    readme = readme.replace("real-token and bootstrap-block expert counts", "real-token and padding expert counts")
    readme = readme.replace("for the exported tables and source files", "for the exported tables; the full source manifest is identified by its hash")
    readme = readme[:readme.index("Total parquet bytes:")]
    readme += "Compact scope:\n" + "\n".join(f"- {s}" for s in compact_manifest["limitations"][-3:])
    readme += f"\n\nTotal parquet bytes: {compact_manifest['bytes']:,}.\n"
    (export.output / "README.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--compact-from", type=Path)
    parser.add_argument("--output", type=Path)
    updates = parser.add_mutually_exclusive_group()
    updates.add_argument("--update-benchmarks", type=Path,
                        help="refresh benchmark observations in this existing package using --config")
    updates.add_argument("--update-fits", type=Path,
                        help="refresh fitted observations in this existing package using --config")
    args = parser.parse_args()
    if args.update_benchmarks or args.update_fits:
        if not args.config or args.output:
            parser.error("updates require --config and cannot be combined with --output")
        refresh = refresh_fits if args.update_fits else refresh_benchmarks
        refresh(json.loads(args.config.read_text()), args.update_fits or args.update_benchmarks)
        return
    if args.compact_from:
        if not args.output:
            parser.error("--compact-from requires --output")
        compact(args.compact_from, args.output)
        return
    config = json.loads(args.config.read_text())
    if args.output: config["output"] = str(args.output)
    export = Export(config)
    grid, geometry = main_grid(export)
    seed_study(export, grid, geometry)
    composition(export, geometry)
    fitting(export)
    benchmark(export)
    routing(export)
    lr_study(export)
    finish(export)


if __name__ == "__main__":
    main()
