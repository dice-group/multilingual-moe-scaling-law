"""Guard numerical export joins against lost or duplicated observations."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from result_analysis.study_setup.data_package import Export, digest, join_columns, refresh_benchmarks, refresh_fits
from result_analysis.scaling_law_fit import analysis as laws


class DataPackageJoins(unittest.TestCase):
    def test_keeps_observations_and_refuses_missing_or_duplicate_metadata(self):
        observations = pd.DataFrame({"id": ["a", "a", "b"], "loss": [1., 2., 3.]})
        metadata = pd.DataFrame({"id": ["a", "b"], "gpu_hours": [4., 5.]})
        result = join_columns(observations, metadata, ["id"])
        self.assertEqual(result.loss.tolist(), [1., 2., 3.])
        self.assertEqual(result.gpu_hours.tolist(), [4., 4., 5.])
        with self.assertRaisesRegex(ValueError, "Missing join"):
            join_columns(observations, metadata.iloc[:1], ["id"])
        with self.assertRaises(pd.errors.MergeError):
            join_columns(observations, pd.concat([metadata, metadata]), ["id"])


class BenchmarkRefresh(unittest.TestCase):
    def fixture(self, root, samples=10):
        package = Export({"output": str(root / "package")})
        old = pd.DataFrame([dict(model_id="m", benchmark="task", language="en", E=1, K=12,
            ladder="B", seed=1234, samples=10, corpus_language="eng_Latn", panel="original_complete",
            D_total=1000, D_t=100, N_act=10, N_total=10, readout_id="r", trained=True,
            in_common12=True, is_english=True, in_common12_non_english=False,
            validation_nll=3., validation_adjusted_loss=-1., primary=.4)])
        package.write("benchmarks", old, "language observations", ["model_id", "benchmark", "language"])
        package.write("other_study", pd.DataFrame({"value": [7.25]}), "other observations")
        manifest = dict(tables=package.tables, bytes=sum(t["bytes"] for t in package.tables.values()))
        (package.output/"manifest.json").write_text(json.dumps(manifest))
        (package.output/"README.md").write_text("Test package\n\nTotal parquet bytes: 0.\n")
        updated = old.assign(primary=.6, samples=samples, benchmark_revision="test")
        source = root/"source.parquet"
        updated.to_parquet(source, index=False, compression="zstd")
        receipt = dict(status="passed", revision="test", table=dict(path=str(source), sha256=digest(source)),
            previous_benchmarks=dict(sha256=package.tables["benchmarks"]["sha256"]),
            observed=dict(language_rows=1, models=1, tasks=1), limitations=[],
            campaign={}, collection={}, metric_sources={}, comparison_audit={})
        receipt["manifest_sha256"] = canonical_json_sha256(receipt)
        path = root/"receipt.json"
        path.write_text(json.dumps(receipt))
        config = dict(benchmark_update=dict(path=str(path), sha256=digest(path)))
        return package.output, config, updated

    def test_refresh_keeps_other_studies_and_can_be_repeated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, config, updated = self.fixture(root)
            readme = package / "README.md"
            readme.write_text(readme.read_text().replace("Total parquet bytes:",
                "Fitting revision: test-fit.\n\nTotal parquet bytes:"))
            other_hash = digest(package/"other_study.parquet")
            refresh_benchmarks(config, package)
            pd.testing.assert_frame_equal(pd.read_parquet(package/"benchmarks.parquet"), updated)
            self.assertEqual(digest(package/"other_study.parquet"), other_hash)
            self.assertIn("Fitting revision: test-fit.", readme.read_text())
            manifest_hash = digest(package/"manifest.json")
            refresh_benchmarks(config, package)
            self.assertEqual(digest(package/"manifest.json"), manifest_hash)

    def test_changed_sample_denominator_leaves_package_untouched(self):
        with TemporaryDirectory() as tmp:
            package, config, _ = self.fixture(Path(tmp), samples=9)
            before = {p.name: digest(p) for p in package.iterdir()}
            with self.assertRaises(AssertionError):
                refresh_benchmarks(config, package)
            self.assertEqual({p.name: digest(p) for p in package.iterdir()}, before)

    def test_changed_source_checksum_leaves_package_untouched(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, config, _ = self.fixture(root)
            before = {p.name: digest(p) for p in package.iterdir()}
            (root/"source.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                refresh_benchmarks(config, package)
            self.assertEqual({p.name: digest(p) for p in package.iterdir()}, before)


class FittingRefresh(unittest.TestCase):
    def test_fit_inputs_keep_fitting_taxonomy_and_unknown_family_semantics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = pd.DataFrame(dict(language=["x", "y", "foreign"], K=[2, 2, 1],
                U_t=[100., 200., None], metric_role=["trained", "trained", "zero_shot"]))
            endpoint = pd.DataFrame(dict(language=["x", "y"], K=[2, 2], U_t=[100., 200.],
                family=["fitting-family", None], script=["Latn", "Latn"]))
            grid.to_parquet(root / "main_grid.parquet", index=False)
            endpoint.to_parquet(root / "large_model.parquet", index=False)
            observations, _, inventories, metadata = laws.load_fit_inputs(root)
            self.assertEqual(observations.language.tolist(), ["x", "y"])
            self.assertEqual(metadata["x"]["family"], "fitting-family")
            self.assertIsNone(metadata["y"]["family"])
            self.assertEqual(inventories, {("x", 2): 100., ("y", 2): 200.})
            endpoint.loc[0, "U_t"] = 101.
            endpoint.to_parquet(root / "large_model.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "Inconsistent fitting inventory"):
                laws.load_fit_inputs(root)

    def test_adopted_predictions_do_not_mix_optimizer_controls_or_history(self):
        with TemporaryDirectory() as tmp:
            data = Path(tmp)
            row = dict(model="law", fold="fold", case="grid", trajectory_id="t",
                       readout_id="r", language="x", adjusted_loss=1.)
            predictions = pd.DataFrame([
                row | dict(prediction=1.1, fit_revision="new", fit_campaign="paper_refit", arm="lbfgsb_box"),
                row | dict(prediction=1.2, fit_revision="new", fit_campaign="paper_refit", arm="slsqp_box"),
                row | dict(prediction=1.3, fit_revision="historical", fit_campaign="final_law", arm="historical"),
                row | dict(prediction=1.4, fit_revision="historical", fit_campaign="baselines", arm="historical"),
            ])
            (data / "manifest.json").write_text(json.dumps(dict(fit_revision=dict(
                revision="new", default_arm="lbfgsb_box", fit_campaign="paper_refit"))))
            predictions.to_parquet(data / "fit_predictions.parquet", index=False)
            self.assertEqual(laws.load_predictions(data).prediction.tolist(), [1.1])
            self.assertEqual(laws.load_predictions(data, arm="slsqp_box").prediction.tolist(), [1.2])
            self.assertEqual(laws.load_predictions(data, revision="historical").prediction.tolist(), [1.3])
            self.assertEqual(laws.load_predictions(data, revision="historical", campaign="baselines").prediction.tolist(), [1.4])
            with self.assertRaisesRegex(ValueError, "Empty or ambiguous"):
                laws.load_predictions(data, arm="missing")
            pd.concat([predictions, predictions.iloc[:1]]).to_parquet(data / "fit_predictions.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                laws.load_predictions(data)

    def test_coefficient_round_trip_preserves_native_and_shared_coordinates(self):
        from result_analysis.study_setup.refit_package import parameters
        from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import model
        spec = model.Spec("native", "chinchilla")
        native = dict(model="native", fold="f", arm="lbfgsb_box", receipt="native.json", result=dict(groups=[
            dict(language="x", spec=spec.payload(), selected=dict(coordinates=[.1, 2., 3., .4, .5]),
                 donors=["a", "b", "c"], training_minimum=2., unigram_nll=6.)]))
        shared = dict(model="shared", fold="f", arm="lbfgsb_box", receipt="shared.json", result=dict(
            selected=dict(coordinates=[.2, -.5], language_coefficients=[dict(language="x", A=2., B=3., L_inf=-4.)]),
            bounds=[dict(name="a", lower=0., upper=1.), dict(name="b", lower=-1., upper=1.)]))
        rows = parameters(native, {"x": {"train": {"delta": .2}}}, "new", "native-hash")
        rows += parameters(shared, {}, "new", "shared-hash")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "parameters.parquet"
            pd.DataFrame(rows).sample(frac=1, random_state=9).to_parquet(path, index=False)
            frame = pd.read_parquet(path)
            expected_native = {k: v for k, v in native["result"]["groups"][0].items()}
            self.assertEqual(laws.parameter_record(frame, "native", "f", revision="new")["groups"], [expected_native])
            self.assertEqual(laws.parameter_record(frame, "shared", "f", revision="new")["selected"], shared["result"]["selected"])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                laws.parameter_record(frame[~(frame.model.eq("shared") & frame.coordinate_index.eq(0))],
                                      "shared", "f", revision="new")

    def test_unbound_refit_seal_leaves_package_untouched(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Export({"output": str(root / "package")})
            package.write("measurement", pd.DataFrame({"loss": [1.25]}), "observations")
            (package.output / "manifest.json").write_text(json.dumps(dict(tables=package.tables)))
            (package.output / "README.md").write_text("test")
            seal = root / "seal.json"
            seal.write_text("{}")
            before = {p.name: digest(p) for p in package.output.iterdir()}
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                refresh_fits(dict(fit_update=dict(path=str(seal), sha256="wrong", revision="new")), package.output)
            self.assertEqual({p.name: digest(p) for p in package.output.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
