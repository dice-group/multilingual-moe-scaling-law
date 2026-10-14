"""Scientific invariants of the frozen composition analysis."""
import itertools
import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.language_composition.analysis import analysis_statistics as s
from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_data import check_prefix
from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_models import predict
from multilingual_megatron.experiments.scaling.language_composition.study_contracts import EXPERTS, HORIZONS, SEEDS


class AnalysisTests(unittest.TestCase):
    def test_training_provenance_exists(self):
        from multilingual_megatron.experiments.scaling.language_composition.training import training_workflow as training
        for path in training.VARIANT.provenance_files:
            self.assertTrue(path.is_file(), path)
        self.assertIn(Path(training.__file__), training.VARIANT.provenance_files)

    def test_replay_import_without_training_dependencies(self):
        subprocess.run([sys.executable, "-c", "import sys; sys.modules['tokenizers'] = None; "
            "from multilingual_megatron.experiments.scaling.language_composition.analysis import workflow"], check=True)

    def test_reader_rejects_changed_binding(self):
        from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_data import Reader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Reader(directory).read(path, "0" * 64)

    def panel(self):
        rows = []
        for model, fold, arm, seed, expert, horizon in itertools.product(
                ("M6-ND", "M5-ND"), ("f1", "f2", "f3"), ("compact", "diverse"), SEEDS, EXPERTS, HORIZONS):
            # Full has constant 1/2/3-nat contrast errors by fit; coverage doubles them.
            value = float(arm == "diverse")
            error = int(fold[-1]) * (1 if model == "M6-ND" else 2)
            rows.append({"model": model, "fold": fold, "arm": arm, "seed": seed, "E": expert,
                "horizon": horizon, "language": "eng_Latn", "adjusted_loss": value,
                "prediction": value + (error if arm == "diverse" else 0)})
        return pd.DataFrame(rows)

    def spec(self):
        return {"models": ["M6-ND", "M5-ND"], "folds": ["f1", "f2", "f3"], "shared_languages": ["eng_Latn"]}

    def test_mean_of_fit_scores_not_pooled_rmse(self):
        _, seeds, _, paired = s.prediction_metrics(self.panel(), self.spec())
        self.assertTrue(seeds[seeds.model.eq("M6-ND")].contrast_rmse.eq(2).all())
        self.assertFalse(np.isclose(2, np.sqrt((1 + 4 + 9) / 3)))
        self.assertEqual(paired.groupby(["seed", "horizon"]).full_minus_coverage.mean().iloc[0], -2)
        self.assertTrue(seeds[seeds.model.eq("zero_shift")].contrast_rmse.eq(1).all())

    def test_missing_or_duplicate_pairs_fail(self):
        rows = self.panel()
        with self.assertRaisesRegex(ValueError, "unpaired"):
            s.prediction_metrics(rows.iloc[1:], self.spec())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            s.prediction_metrics(pd.concat([rows, rows.iloc[:1]]), self.spec())

    def test_offsets_cancel_from_contrasts(self):
        rows = self.panel().query("model == 'M6-ND' and fold == 'f1'").copy()
        before = s.arm_contrasts(rows, ["prediction", "adjusted_loss"])
        rows["prediction"] -= 1.25
        rows["adjusted_loss"] -= 1.25
        pd.testing.assert_frame_equal(before, s.arm_contrasts(rows, ["prediction", "adjusted_loss"]))
        # old adjusted + old unigram - new unigram has the required sign.
        self.assertEqual(-4.0 + 7.0 - 8.0, -5.0)

    def test_moe_interaction_sign(self):
        rows = self.panel().query("model == 'M6-ND' and fold == 'f1'").copy()
        rows["adjusted_loss"] = np.where(rows.arm.eq("diverse"), np.where(rows.E.eq(1), 2., 1.), 0.)
        differences = s.arm_contrasts(rows, ["adjusted_loss"])
        table = differences.pivot(index=["seed", "horizon", "language"], columns="E", values="adjusted_loss")
        self.assertTrue((table[1] - table[32]).eq(1).all())

    def test_seed_summary_requires_three(self):
        self.assertEqual(s.describe([1, 2, 3])["sample_standard_deviation"], 1)
        with self.assertRaises(ValueError):
            s.describe([1] * 9)

    def test_prefix_continuity_and_bounds(self):
        reference = np.array([0, 1, 1, 0, 2])
        check_prefix(reference, np.array([0, 1, 1, 0]), 4)
        with self.assertRaises(ValueError):
            check_prefix(reference, np.array([1, 1, 0, 2]), 4)
        with self.assertRaises(ValueError):
            check_prefix(reference, reference, 6)

    def test_no_zero_exposure_or_new_coefficients(self):
        row = pd.DataFrame({"D_t": [0.], "language": ["eng_Latn"]})
        with self.assertRaisesRegex(ValueError, "positive"):
            predict(row, {"model": "M6-ND", "selected": {}})
        row["D_t"] = 10.
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            predict(row, {"model": "M6-ND", "selected": {"language_coefficients": []}})


if __name__ == "__main__":
    unittest.main()
