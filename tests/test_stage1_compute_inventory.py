"""Study inventory and measured compute checks on a synthetic grid."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from result_analysis.evidence import Evidence, read_result, sha256
from result_analysis.compute_allocation.inventory import run


def fixture_tables():
    """Match the declared grid shape while keeping all values explicitly synthetic."""
    rows, geometry, readouts, attribution, training = [], [], [], [], []
    ratios = {1: 1., 4: 1.17, 16: 1.44, 32: 1.69}
    mfu = {1: 35.7, 4: 30., 16: 25., 32: 22.}
    for size, ladder in enumerate("ABCD"):
        n = 5e7 * (size + 1)
        for e in (1, 4, 16, 32):
            geometry.append(dict(ladder=ladder, E=e, N_act=n, N_total=n*(1+e)))
            for k in (1, 12, 50, 200):
                trajectory = f"synthetic-{ladder}-E{e}-K{k}"
                training.append(dict(training_job_id=trajectory,
                    mean_mfu_pct_of_configured_peak=mfu[e], mean_tokens_per_second_per_gpu=1000/ratios[e]))
                for step in range(7 if ladder == "D" else 6):
                    tokens = (step + 1) * 5e8
                    readout = f"{trajectory}-{step}"
                    readouts.append(dict(readout_id=readout, ladder=ladder, experts=e,
                        language_count=k, nominal_tokens=tokens, training_job_id=trajectory))
                    attribution.append(dict(readout_id=readout,
                        logical_readout_attributed_productive_gpu_hours=(step+1)*(size+1)*ratios[e]))
                    target, other = tokens/k, tokens-tokens/k
                    for language in range(k):
                        loss = 3. - .05*step - .001*e + .0001*language
                        rows.append(dict(trajectory_id=trajectory, readout_id=readout,
                            language=f"synthetic-{language}", ladder=ladder, E=e, K=k, N_act=n,
                            D_total=tokens, D_t=target, D_family=.5*other, D_script=.25*other,
                            D_other=.25*other, raw_nll=loss, unigram_nll=5., adjusted_loss=loss-5.))
    return {
        "study/primary_fits/fit_rows.parquet": pd.DataFrame(rows),
        "scoring/grid.parquet": pd.DataFrame(geometry),
        "release/tables/readouts.parquet": pd.DataFrame(readouts),
        "release/tables/readout_compute_attribution.parquet": pd.DataFrame(attribution),
        "release/tables/training_summary.parquet": pd.DataFrame(training),
        "release/tables/unique_physical_study_compute.parquet": pd.DataFrame({"synthetic_runs": [64]}),
        "study/paper_exports/language_tier_metadata.csv": pd.DataFrame({"language": [f"synthetic-{i}" for i in range(200)]}),
    }


def write_bundle(root, tables):
    files = {}
    for name, frame in tables.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        files[name] = {"sha256": sha256(path)}
    (root / "manifest.json").write_text(json.dumps({
        "schema": "stage1-paper-inputs/v1", "paper_commit": "synthetic-fixture",
        "analysis": {"version": "stage1-paper-v2"}, "files": files}))
    return Evidence(root)


class StudyInventoryTests(unittest.TestCase):
    def test_grid_compute_costs_and_receipt(self):
        tables = fixture_tables()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = write_bundle(root / "inputs", tables)
            output = root / "setup"
            claims = run(evidence, output)
            self.assertEqual(json.loads(json.dumps(claims)), read_result(output, evidence))
            self.assertEqual([claims[k] for k in ("training_trajectories", "dependent_endpoints",
                             "language_readouts", "architecture_cells", "checks_passed")],
                             [64, 400, 26300, 16, 9])
            endpoints = pd.read_csv(output / "compute_overview.csv").set_index("readout_id")
            rows = tables["study/primary_fits/fit_rows.parquet"]
            means = rows.groupby("readout_id").raw_nll.mean().reindex(endpoints.index)
            np.testing.assert_allclose(endpoints.mean_raw_nll, means, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(endpoints.FLOPs, 6*endpoints.N_act*endpoints.D_total)
            np.testing.assert_array_equal(endpoints.languages, endpoints.K)
            np.testing.assert_allclose([claims["physical_gpu_hour_ratios"][e] for e in (4, 16, 32)],
                                       [1.17, 1.44, 1.69], rtol=0, atol=1e-14)
            self.assertTrue(pd.read_csv(output / "checks.csv").passed.all())
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(set(receipt["inputs"]), set(tables))
            self.assertFalse(receipt["new_training"])
            self.assertFalse(receipt["new_law_fit"])
            self.assertIn("nested", receipt["independent_unit"])
            self.assertTrue((output / "compute_overview.png").is_file())
            self.assertTrue((output / "compute_overview.pdf").is_file())
            with self.assertRaises(FileExistsError):
                run(evidence, output)

    def test_duplicate_observations_fail_closed(self):
        tables = fixture_tables()
        name = "study/primary_fits/fit_rows.parquet"
        tables[name] = pd.concat([tables[name], tables[name].iloc[:1]], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "unique observed rows"):
                run(write_bundle(root / "inputs", tables), root / "setup")
            self.assertFalse((root / "setup/receipt.json").exists())

    def test_loss_and_exposure_identities_fail_closed(self):
        for column, message in (("adjusted_loss", "unigram-adjusted"), ("D_t", "exposure components")):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                tables = fixture_tables()
                tables["study/primary_fits/fit_rows.parquet"].loc[0, column] += 1.
                root = Path(directory)
                with self.assertRaisesRegex(ValueError, message):
                    run(write_bundle(root / "inputs", tables), root / "setup")
                self.assertFalse((root / "setup/receipt.json").exists())

    def test_missing_physical_comparison_fails_closed(self):
        tables = fixture_tables()
        name = "release/tables/readout_compute_attribution.parquet"
        tables[name] = tables[name].iloc[1:]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "100 matched physical-cost comparisons"):
                run(write_bundle(root / "inputs", tables), root / "setup")
            self.assertFalse((root / "setup/receipt.json").exists())

    def test_profile_and_cpu_command_are_standalone(self):
        root = Path(__file__).resolve().parents[1]
        profile = json.loads((root / "configs/result_analysis/stage1_paper_v2.json").read_text())
        self.assertEqual(profile["analysis"]["version"], "stage1-paper-v2")
        for name in fixture_tables():
            self.assertRegex(profile["files"][name]["sha256"], r"^[0-9a-f]{64}$")
        result = subprocess.run([sys.executable, "-m", "result_analysis.compute_allocation.inventory", "--help"],
                                cwd=root, check=True, text=True, capture_output=True)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
