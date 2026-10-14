"""Portable fitting-input contracts; no optimizer runs or cluster submissions."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import data
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs as refit
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import numerics


def fixture(root):
    source = root / "source"
    source.mkdir()
    metadata = {f"l{i}_S{i}": dict(family=None, script=f"S{i}") for i in range(4)}
    grid = pd.DataFrame([
        dict(trajectory_id=f"t{t}", readout_id=f"t{t}-r{r}", language=language,
             N_act=1e8*(t+1), N_total=4e8*(t+1), E=4., K=4.,
             D_t=1e9*(r+1), D_total=4e9*(r+1), U_t=1.5e9,
             adjusted_loss=-3.+.01*t+.03*r, raw_nll=3.+.01*t+.03*r, unigram_nll=6.,
             D_family=0., D_script=0., D_other=3e9*(r+1))
        for t in range(4) for r in range(2) for language in metadata])
    endpoint = grid[grid.readout_id.eq("t0-r1")].copy()
    endpoint["trajectory_id"], endpoint["readout_id"] = "large", "large-r1"
    endpoint["D_total"] = 68e9
    covariates = grid.drop(columns=list(data.RESPONSES & set(grid)))
    inventories = {(language, 4): 1.5e9 for language in metadata}
    memberships, training, calibrations, native, expected = {}, {}, {}, [], {}
    regularization = dict(schema_version="stage1_m5_m6_ridge/v1", **{"lambda": .01},
                          m2_reference_objective=.001, gamma_n_scale=.1,
                          phi_endpoint_scale=.05, psi_endpoint_scale=.05)
    for i, fold in enumerate(refit.FOLDS):
        test = grid[grid.trajectory_id.eq(f"t{i}")] if fold != "all_grid" else grid.iloc[:0]
        train = grid[~grid.trajectory_id.isin(test.trajectory_id)].reset_index(drop=True)
        memberships[fold] = data.membership(train, test)
        training[fold] = data.write_training(source, fold, train, covariates, inventories, metadata)
        calibrations[fold] = dict(delta=.12, regularization=regularization, anchor=[.1]*15)
        expected[fold] = dict(rows=len(train), configurations=train.trajectory_id.nunique(),
                              languages=4, test_rows=len(test))
        if fold != "all_grid":
            _, _, groups = data.read_training(training[fold])
            for j, group in enumerate(groups):
                packet = group["train"] | {"delta": .13+.01*j}
                support = {k: v for k, v in packet.items() if k not in {"delta", "weights", "y"}}
                native.append(group | dict(fold=fold, train=packet, support=support))
    native_path = source / "packets.json.gz"
    native_path.write_bytes(gzip.compress(json.dumps(native).encode()))
    grid.to_parquet(source / "grid.parquet", index=False)
    endpoint.to_parquet(source / "endpoint.parquet", index=False)
    io.write_json_once(source / "exposure_contract.json", dict(metadata=metadata,
        inventories=[dict(language=language, K=k, U=u) for (language, k), u in inventories.items()]))
    search = refit.sealed(dict(inputs=[refit.binding(native_path)], training=training,
        memberships=memberships, native_specs=[s.payload() for s in refit.panel()],
        scoring=[refit.binding(source / n) for n in ("grid.parquet", "endpoint.parquet", "exposure_contract.json")],
        calibration={f: calibrations[f] for f in refit.PRIMARY}, protocol={}, environment={}))
    io.write_json_once(source / "search.json", search)
    coverage = refit.sealed(dict(parent_sha256=search["sha256"], training=training,
        memberships=memberships, calibrations=calibrations, rules={}, environment={}))
    io.write_json_once(source / "coverage.json", coverage)
    profile = root / "profile.json"
    io.write_json_once(profile, dict(search_plan="source/search.json", coverage_plan="source/coverage.json",
        output_root="bundle", expected_inventory=expected))
    return profile, grid, native


class PaperRefitTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.profile, self.grid, self.native = fixture(self.root)
        self.path = self.root / "bundle/manifest.json"

    def prepare(self):
        # Input preparation may never launch a numerical worker or scheduler.
        with patch("subprocess.run", side_effect=AssertionError("unexpected external command")):
            return refit.prepare(self.profile)

    def test_panel_and_exact_row_weight_calibration_parity(self):
        result = self.prepare()
        self.assertEqual((result["models"], result["fits"]), (8, 32))
        self.assertEqual(len({(c["model"], c["fold"]) for c in refit.cells()}), 32)
        for fold in refit.FOLDS:
            rows, sources, packets, cal = refit.read_training(self.path, fold)
            old, old_sources, _ = data.read_training(io.read_json(self.root / "source/search.json")["training"][fold])
            pd.testing.assert_frame_equal(rows, old, check_exact=True)
            for name, values in sources.__dict__.items():
                np.testing.assert_array_equal(values, old_sources.__dict__[name])
            self.assertEqual(cal["delta"], .12)
            if fold != "all_grid":
                originals = {p["language"]: p for p in self.native if p["fold"] == fold}
                for p in packets:
                    for name in ("train", "support", "donors", "unigram_nll"):
                        self.assertEqual(p[name], originals[p["language"]][name])
                    self.assertNotEqual(p["train"]["delta"], cal["delta"])
            else:
                self.assertTrue(all(p["train"]["delta"] == cal["delta"] for p in packets))
                self.assertTrue(all("y" not in p["support"] for p in packets))

    def test_portable_without_source_campaigns_or_evaluation_reads(self):
        self.prepare()
        (self.root / "source").rename(self.root / "unavailable_source")
        moved = self.root / "relocated"
        (self.root / "bundle").rename(moved)
        path = moved / "manifest.json"
        self.assertEqual(refit.verify_bundle(path)["fits"], 32)
        # An optimizer can load its packet even when evaluation files are absent.
        (moved / "scoring").rename(moved / "unavailable_scoring")
        (moved / "provenance.json").rename(moved / "unavailable_provenance.json")
        for fold in refit.FOLDS:
            rows, _, _, _ = refit.read_training(path, fold)
            self.assertFalse(rows.empty)

    def test_native_objective_and_domain_parity(self):
        self.prepare()
        specifications = [s for s in refit.panel() if s.name in refit.NATIVE]
        for fold in refit.FOLDS:
            _, _, packets, cal = refit.read_training(self.path, fold)
            if fold == "all_grid":
                search = io.read_json(self.root / "source/search.json")
                _, _, originals = data.read_training(search["training"][fold])
                originals = [p | {"train": p["train"] | {"delta": cal["delta"]}, "support": p["train"]}
                             for p in originals]
            else:
                originals = [p for p in self.native if p["fold"] == fold]
            for new, old in zip(packets, originals):
                for spec in specifications:
                    coordinates = numerics.starts(spec)[0]
                    for a, b in zip(numerics.objective(spec, coordinates, new["train"]),
                                    numerics.objective(spec, coordinates, old["train"])):
                        np.testing.assert_array_equal(a, b)
                    for a, b in zip(numerics.m.domain(spec, coordinates, new["support"]),
                                    numerics.m.domain(spec, coordinates, old["support"])):
                        np.testing.assert_array_equal(a, b)

    def test_create_only_and_source_hash_tampering(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "create-only"):
            self.prepare()
        entry = io.read_json(self.root / "source/search.json")["training"][refit.PRIMARY[0]]
        packet = io.read_json(Path(entry["path"]))
        source = Path(packet["files"][0]["path"])
        source.write_bytes(source.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed bound input"):
            refit.prepare(self.profile, output=self.root / "second_bundle")
        self.assertFalse((self.root / "second_bundle").exists())

    def test_bundle_hash_and_external_path_fail_closed(self):
        self.prepare()
        rows = self.root / f"bundle/training/{refit.PRIMARY[0]}/rows.parquet"
        rows.write_bytes(rows.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed bound input"):
            refit.read_training(self.path, refit.PRIMARY[0])
        for path in ("../profile.json", str(self.profile)):
            with self.assertRaisesRegex(ValueError, "internal relative paths"):
                refit.local_path(self.root / "bundle", dict(path=path, sha256="unused"))

    def test_native_packets_reject_missing_duplicate_or_heldout_responses(self):
        fold = refit.PRIMARY[0]
        selected = [p for p in self.native if p["fold"] == fold]
        rows, _, _ = data.read_training(io.read_json(self.root / "source/search.json")["training"][fold])
        for invalid in (selected[:-1], [*selected, selected[0]]):
            with self.assertRaisesRegex(ValueError, "exactly the fitted languages"):
                refit.native_groups(rows, invalid)
        invalid = deepcopy(selected)
        invalid[0]["support"]["y"] = [3.]
        with self.assertRaisesRegex(ValueError, "covariates only"):
            refit.native_groups(rows, invalid)
        invalid = deepcopy(selected)
        invalid[0]["train"]["weights"][0] *= 2
        with self.assertRaises(AssertionError):
            refit.native_groups(rows, invalid)

    def test_split_checks_reject_large_run_overlap_and_changed_membership(self):
        search = io.read_json(self.root / "source/search.json")
        fold = refit.PRIMARY[0]
        rows, _, _ = data.read_training(search["training"][fold])
        member = search["memberships"][fold]
        with self.assertRaisesRegex(ValueError, "larger-model evaluation entered"):
            refit.validate_split(rows, self.grid, rows.iloc[:1], member, fold)
        endpoint = pd.read_parquet(self.root / "source/endpoint.parquet")
        with self.assertRaisesRegex(ValueError, "whole configurations"):
            refit.validate_split(rows, self.grid, endpoint,
                member | {"test_trajectory_ids": member["train_trajectory_ids"][:1]}, fold)
        with self.assertRaises(AssertionError):
            refit.validate_split(rows.iloc[:-1], self.grid, endpoint, member, fold)


if __name__ == "__main__":
    unittest.main()
