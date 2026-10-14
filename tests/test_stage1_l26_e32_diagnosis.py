from copy import deepcopy
from dataclasses import asdict
import unittest

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.fit import SharedParameters
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow.model import Spec, predict
from result_analysis.stage1_l26_e32.diagnosis import error_decomposition
from result_analysis.stage1_l26_e32.diagnosis import matched_experts
from result_analysis.stage1_l26_e32.diagnosis import native_terms
from result_analysis.stage1_l26_e32.diagnosis import shared_terms
from result_analysis.stage1_l26_e32.evidence import audit_resume
from result_analysis.stage1_l26_e32.evidence import require_equal_stream


class DiagnosisTest(unittest.TestCase):
    def test_mean_bias_is_not_residual_variance(self):
        result = error_decomposition([0, 0], [1, 3])
        self.assertEqual(result["bias"], 2)
        self.assertEqual(result["squared_bias"], 4)
        self.assertEqual(result["residual_variance"], 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            error_decomposition([0, 0], [1, np.nan])

    def test_terms_reconstruct_and_active_atlas_has_no_expert_coordinate(self):
        spec = Spec("fixture", "full")
        x = [.5, 2, 3, .5, .3, 1, .2, .2, .2, .1]
        packet = {"N": [1e8], "D": [[1e9, 2e9, 3e9, 4e9, 5e9]],
                  "U": [[1e8]*5], "E": [1], "K": [200], "training_minimum": 2.}
        a = predict(spec, x, packet)[0]
        np.testing.assert_array_equal(a, predict(spec, x, dict(packet, E=[32]))[0])
        np.testing.assert_allclose(native_terms(spec, x, packet, 5).sum(axis=1), a-5)
        rows = pd.DataFrame({"language": ["a"], "N_act": [4e8], "D_t": [1e9],
                             "E": [1], "K": [1], "D_family": [0], "D_script": [0], "D_other": [0]})
        selected = {"shared_parameters": asdict(SharedParameters(.5, 0., .5, 0.)), "zeta": .5,
                    "language_coefficients": [{"language": "a", "L_inf": -5, "A": 2, "B": 3}]}
        np.testing.assert_allclose(shared_terms(selected, rows), [[-5, 1, 6]])

    def test_exact_expert_pairs_reject_exposure_changes_and_duplicates(self):
        rows = pd.DataFrame({"fold": ["f"]*2, "model": ["m"]*2, "ladder": ["D"]*2,
            "K": [200]*2, "language": ["a"]*2, "N_act": [1e8]*2, "D_total": [100]*2,
            "D_t": [10]*2, "D_family": [20]*2, "D_script": [30]*2, "D_other": [40]*2,
            "E": [1, 32], "adjusted_loss": [-3, -4], "prediction": [-3.5, -3.5]})
        _, summary = matched_experts(rows)
        self.assertEqual(summary.observed_gain.iloc[0], 1.)
        self.assertEqual(summary.predicted_gain.iloc[0], 0.)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            matched_experts(pd.concat([rows, rows]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            matched_experts(rows.iloc[:1])
        rows.loc[1, "D_t"] = 11
        with self.assertRaisesRegex(ValueError, "exposure"):
            matched_experts(rows)

    def test_validation_and_resume_missing_evidence_fails_closed(self):
        language = {k: 1 for k in ("target_token_ids_sha256", "unigram_counts_sha256",
                    "special_token_ids", "token_count", "vocab_size", "unigram_log_probability_sum")}
        require_equal_stream(language, deepcopy(language))
        with self.assertRaisesRegex(ValueError, "missing or different"):
            require_equal_stream(language, {**language, "target_token_ids_sha256": 2})
        expected = {"step": 217233, "consumed_train_samples": 26829184, "scheduler_num_steps": 26829184}
        receipts = [dict(rank=i, status="verified_before_first_update", expected=expected,
                        observed=expected, checkpoint_path="/valid/iter_0217233") for i in range(8)]
        audit_resume(receipts)
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            audit_resume(receipts[:-1])
        receipts[0] = dict(receipts[0], observed=dict(expected, step=0))
        with self.assertRaisesRegex(ValueError, "wrong restored"):
            audit_resume(receipts)


if __name__ == "__main__":
    unittest.main()
