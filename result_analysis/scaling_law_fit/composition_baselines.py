"""Prediction-only published-baseline comparison on the sealed K12 study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import scoring
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs, numerics
from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_statistics import prediction_metrics

FOLDS = ("largest_N_act", "E32", "K200_common_languages")
NATIVE = inputs.NATIVE
SHARED = ("RN14-stored-reuse", "M5-ND", "M6-ND")
KEYS = ["trajectory_id", "readout_id", "language"]
PAIR = ["arm", "seed", "E", "horizon", "language"]


def identify(rows):
    rows = rows.copy()
    rows["trajectory_id"] = [f"composition:{a}:s{s}:e{e}" for a, s, e in
                             rows[["arm", "seed", "E"]].itertuples(index=False, name=None)]
    rows["readout_id"] = rows.trajectory_id + ":d" + rows.horizon.astype(str)
    if rows.duplicated(KEYS).any():
        raise ValueError("duplicate composition observation")
    return rows


class Evaluation:
    def __init__(self, args):
        self.args, self.bindings, self.checks = args, {}, {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"input hash mismatch: {path}")
        if str(path) in self.bindings and self.bindings[str(path)] != digest:
            raise ValueError(f"input changed during evaluation: {path}")
        self.bindings[str(path)] = digest
        return path

    def read(self, path, expected=None):
        path = self.bind(path, expected)
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())

    def bound(self, root, entry):
        return self.read(inputs.local_path(root, entry), entry["sha256"])

    def inventories(self, original, exposures):
        records = []
        for path, digest in original["source_bindings"].items():
            if Path(path).name != "megatron_data_manifest.json":
                continue
            manifest = self.read(path, digest)
            lineage = manifest["lineage"]
            arm = lineage["language_composition_arm"]["arm"]
            members = set(exposures.loc[exposures.arm.eq(arm), "language"])
            if len(members) != 12 or members != set(manifest["prefixes_by_language"]):
                raise ValueError("composition membership mismatch")
            corpus_path = Path(lineage["corpus_manifest_path"])
            corpus = self.read(corpus_path, lineage["corpus_manifest_file_sha256"])
            hashes = self.read(corpus_path.parent / "hash_manifest.json")["files"]
            self.bind(corpus_path, hashes["corpus_manifest.json"])
            for language in sorted(members):
                item = corpus["languages"][language]["train"]
                binding = item["unigram_counts"]
                path = corpus_path.parent / binding["path"]
                count = self.read(path, binding["sha256"])
                self.bind(path, hashes[binding["path"]])
                assert count["language"] == language and count["source_split"] == "train_only"
                assert count["tokenizer_model_sha256"] == corpus["inputs"]["tokenizer_model"]["sha256"]
                unique = count["unique_train_tokens_including_special"]
                assert 0 < unique <= count["exposure_train_tokens_including_special"] == item["selected_tokens"]
                assert unique == count["non_special_token_count"] + count["excluded_special_token_count"]
                records.append(dict(arm=arm, language=language, K=12, U_t=unique, source=str(path)))
        inventory = pd.DataFrame(records)
        assert len(inventory) == 24 and not inventory.duplicated(["arm", "language"]).any()
        return inventory

    def run(self):
        a = self.args
        a.output.mkdir(parents=True, exist_ok=False)
        # Lock protocol and executable before reading any additional predictions.
        self.bind(__file__)
        self.bind(a.protocol)
        (a.output / "protocol.md").write_text(a.protocol.read_text())
        plan = inputs.read_sealed(self.bind(a.refit_root / "plan.json"))
        report = inputs.read_sealed(self.bind(a.fit_report))
        assert report["plan_sha256"] == plan["sha256"]
        assert report["all_cells_certified"] and report["all_scoring_complete"]
        outputs = {Path(e["path"]).name: e for e in report["outputs"]}
        fits = self.bound(a.refit_root, outputs["coefficients.json"])["fits"]
        saved_grid = self.bound(a.refit_root, outputs["predictions.parquet"])
        manifest_path = inputs.local_path(a.refit_root, plan["input"])
        manifest = inputs.read_sealed(self.bind(manifest_path))
        grid = self.bound(manifest_path.parent, manifest["scoring"]["grid.parquet"])
        contract = self.bound(manifest_path.parent, manifest["scoring"]["exposure_contract.json"])
        metadata = contract["metadata"]
        grid_inventory = {(x["language"], x["K"]): x["U"] for x in contract["inventories"]}
        original = self.read(a.composition_root / "inputs/manifest.json")
        def composition_file(name):
            return self.read(a.composition_root / "inputs" / name, original["files"][name])
        spec = composition_file("spec.json")
        exposures = identify(composition_file("exposures.parquet"))
        support = composition_file("composition_support.json")
        for item in support:
            assert all(metadata[item["language"]][k] == item[k] for k in ("family", "script"))
        inventory = self.inventories(original, exposures)
        completion = self.read(a.replay_root / "completion.json")
        assert completion["complete"]
        def replay_file(name):
            return self.read(a.replay_root / name, completion["outputs"][name])
        rows = identify(replay_file("composition_inputs.parquet"))
        previous = replay_file("composition_predictions.parquet")
        assert len(rows) == 756 and set(rows.K) == {12}
        matched = rows.merge(exposures, on=PAIR, suffixes=("", "_source"), validate="one_to_one")
        assert len(matched) == len(rows)
        for column in ("D_t", "D_total", "D_family", "D_script", "D_other"):
            np.testing.assert_array_equal(matched[column], matched[column + "_source"])
        matched = rows.merge(inventory, on=["arm", "language", "K"], suffixes=("", "_source"), validate="many_to_one")
        np.testing.assert_array_equal(matched.U_t, matched.U_t_source)
        covariates = ["readout_id", "language", "K", "D_t", "D_total"]
        predictions, parity = [], []
        for name in (*NATIVE, *SHARED):
            for fold in FOLDS:
                cell, = [f for f in fits if (f["model"], f["fold"], f["arm"]) == (name, fold, "lbfgsb_box")]
                result = cell["result"]
                assert result["certified"] and result["complete"]
                p = rows.copy()
                if name in NATIVE:
                    reference = saved_grid[(saved_grid.model == name) & (saved_grid.fold == fold) &
                        (saved_grid.arm == "lbfgsb_box") & (saved_grid.case == "grid") &
                        saved_grid.language.isin(spec["shared_languages"])]
                    test = grid.merge(reference[KEYS], on=KEYS, validate="one_to_one").reset_index(drop=True)
                    assert len(test) == len(reference) and len(test) > 0
                    replay = scoring.predict_result(result, test, grid[covariates], grid_inventory, metadata)
                    compare = replay.merge(reference[KEYS + ["prediction"]], on=KEYS, suffixes=("", "_saved"), validate="one_to_one")
                    np.testing.assert_allclose(compare.prediction, compare.prediction_saved, rtol=0, atol=1e-10)
                    parity.append(dict(model=name, fold=fold, panel="grid", rows=len(compare), max_gap=float(abs(compare.prediction-compare.prediction_saved).max())))
                    normals = {g["language"]: g["unigram_nll"] for g in result["groups"]}
                    np.testing.assert_allclose(p.language.map(normals), p.old_unigram_nll, rtol=0, atol=1e-10)
                    p["prediction"] = np.nan
                    for arm, block in p.groupby("arm"):
                        inv = inventory[inventory.arm.eq(arm)]
                        lookup = {(r.language, int(r.K)): r.U_t for r in inv.itertuples()}
                        source = exposures[exposures.arm.eq(arm)][covariates]
                        values = scoring.predict_result(result, block, source, lookup, metadata)
                        p.loc[block.index, "prediction"] = values.prediction
                else:
                    regularization = dict(result["regularization"])
                    regularization["lambda"] = regularization.pop("strength")
                    model = numerics.make_model(name, p, None, [], dict(regularization=regularization, delta=result["delta"]), "lbfgsb_box")
                    p["prediction"] = numerics.predict(model, result["selected"])
                    independent = numerics.independent(model, result["selected"], p, [])
                    np.testing.assert_allclose(p.prediction, independent, rtol=0, atol=1e-10)
                p["prediction"] += p.old_unigram_nll - p.unigram_nll
                if not np.isfinite(p.prediction).all():
                    raise ValueError(f"nonfinite predictions: {name}/{fold}")
                p["model"], p["fold"] = name, fold
                if name in SHARED:
                    ref = previous[(previous.model == name) & (previous.fold == fold)]
                    compare = p.merge(ref[PAIR + ["prediction"]], on=PAIR, suffixes=("", "_saved"), validate="one_to_one")
                    assert len(compare) == len(p)
                    np.testing.assert_allclose(compare.prediction, compare.prediction_saved, rtol=0, atol=1e-10)
                    parity.append(dict(model=name, fold=fold, panel="composition", rows=len(compare), max_gap=float(abs(compare.prediction-compare.prediction_saved).max())))
                predictions.append(p)
                print(f"Verified {name} / {fold}: {len(p)} predictions", flush=True)
        predicted = pd.concat(predictions, ignore_index=True)
        spec = spec | dict(models=list((*NATIVE, *SHARED)), folds=list(FOLDS))
        metrics, seeds, contrasts, differences = prediction_metrics(predicted, spec)
        columns = ["absolute_rmse", "contrast_rmse", "interaction_rmse"]
        horizon = seeds.groupby(["model", "horizon"], as_index=False)[columns].mean()
        primary = horizon[horizon.horizon.eq(16_000_000_000)].drop(columns="horizon")
        for name, table in dict(predictions=predicted, contrasts=contrasts, evaluation_inputs=rows).items():
            table.to_parquet(a.output / f"{name}.parquet", index=False)
        for name, table in dict(primary=primary, horizon_metrics=horizon, seed_metrics=seeds, fit_metrics=metrics,
                                internal_contrast_differences=differences, inventories=inventory, parity=pd.DataFrame(parity)).items():
            table.to_csv(a.output / f"{name}.csv", index=False)
        (a.output / "evaluation_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        # Bind imported project code actually used, and detect input mutation.
        import sys
        for module in list(sys.modules.values()):
            path = getattr(module, "__file__", None)
            if path and "multilingual_megatron" in path and path.endswith(".py"):
                self.bind(path)
        for path, digest in list(self.bindings.items()):
            self.bind(path, digest)
        output_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()}
        receipt = dict(complete=True, refits=0, training_runs=0, neural_evaluations=0,
            inputs=self.bindings, outputs=output_hashes, models=spec["models"], folds=list(FOLDS),
            observations_per_fit=len(rows), predictions=len(predicted), parity_checks=parity)
        (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(primary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("refit-root", "fit-report", "composition-root", "replay-root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    Evaluation(parser.parse_args()).run()


if __name__ == "__main__":
    main()
