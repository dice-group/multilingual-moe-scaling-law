import unittest

import numpy as np
import pandas as pd

from result_analysis.stage1_l26_e32.prediction_comparison import endpoint_packet
from result_analysis.stage1_l26_e32.prediction_comparison import shared_predict
from result_analysis.stage1_l26_e32.prediction_comparison import summarize
from multilingual_megatron.experiments.scaling.stage1_fit.fit import SharedParameters
from dataclasses import asdict


class PredictionComparisonTest(unittest.TestCase):
    def test_size_duration_multiplier_only_changes_data_term(self):
        rows = pd.DataFrame({"language": ["a"], "N_act": [4e8], "D_t": [1e9],
                             "E": [1], "K": [1], "D_family": [0], "D_script": [0], "D_other": [0]})
        selected = {"shared_parameters": asdict(SharedParameters(alpha_0=.5, beta_0=.5, alpha_1=0., beta_1=0.)),
                    "language_coefficients": [{"language": "a", "L_inf": -5., "A": 2., "B": 3.}], "zeta": 0.}
        np.testing.assert_allclose(shared_predict(selected, rows), [-1.])
        selected["zeta"] = .5
        np.testing.assert_allclose(shared_predict(selected, rows), [2.])

    def test_packet_preserves_inventory_donors_and_training_floor(self):
        rows = pd.DataFrame({"language": list("abcde"), "D_t": [10, 20, 30, 15, 25],
                             "D_total": 100, "N_act": 1000, "E": 32, "K": 200})
        packet = {"language": "a", "donors": list("bcd"),
                  "support": {"K": [200, 200], "U": [[1, 2, 3, 4, 5]]*2},
                  "train": {"training_minimum": 2.5}}
        result = endpoint_packet(packet, rows, dict(zip("abcde", "xyxxz")), 5000)
        self.assertEqual(result["D"], [[10, 20, 30, 15, 25]])
        np.testing.assert_array_equal(result["U"], [[1, 2, 3, 4, 5]])
        self.assertEqual(result["family_share"], [.55])
        self.assertEqual(result["training_minimum"], 2.5)
        packet["support"]["U"][1] = [9, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, "inventory"):
            endpoint_packet(packet, rows, dict(zip("abcde", "xyxxz")), 5000)

    def test_incomplete_or_invalid_models_cannot_get_a_score(self):
        rows = pd.DataFrame({"fold": ["f"]*5, "model": ["good", "good", "invalid", "invalid", "missing"],
                             "language": ["a", "b", "a", "b", "a"],
                             "observed": [1.]*5, "prediction": [0., 2., 1., np.nan, 1.]})
        scores = summarize(rows, {"f": ["a", "b"]}).set_index("model")
        self.assertEqual(scores.loc["good", "rmse"], 1.)
        self.assertEqual(scores.loc["good", "bias"], 0.)
        self.assertTrue(scores.loc["good", "complete"])
        for model in ("invalid", "missing"):
            self.assertFalse(scores.loc[model, "complete"])
            self.assertTrue(np.isnan(scores.loc[model, "rmse"]))


if __name__ == "__main__":
    unittest.main()
