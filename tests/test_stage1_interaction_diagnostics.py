from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from result_analysis.scaling_law_fit import interaction_diagnostics as diagnostics


def _canonical_fixture() -> pd.DataFrame:
    active_sizes = {
        "A": 100_000_000.0,
        "B": 200_000_000.0,
        "C": 400_000_000.0,
        "D": 800_000_000.0,
    }
    horizons = {"A": 5, "B": 6, "C": 7, "D": 7}
    rows = []
    for ladder in diagnostics.LADDERS:
        n_act = active_sizes[ladder]
        for experts in diagnostics.EXPERT_LEVELS:
            for languages in diagnostics.LANGUAGE_COUNT_LEVELS:
                trajectory = f"{ladder}:E{experts}:K{languages}"
                for index in range(horizons[ladder]):
                    d_total = float(500_000_000 * (2**index))
                    d_t = d_total / float(languages)
                    loss = (
                        5.0
                        - 0.2 * math.log(n_act)
                        - 0.1 * math.log(d_t)
                        - 0.02 * math.log(experts)
                        + 0.01 * math.log(languages)
                    )
                    rows.append(
                        {
                            "trajectory_id": trajectory,
                            "readout_id": f"{trajectory}:D{index}",
                            "language": "eng_Latn",
                            "N_act": n_act,
                            "E": experts,
                            "K": languages,
                            "D_total": d_total,
                            "D_t": d_t,
                            "adjusted_loss": loss,
                            "raw_nll": loss + 7.0,
                            "ladder": ladder,
                        }
                    )
    frame = pd.DataFrame(rows)
    assert frame["trajectory_id"].nunique() == 64
    assert frame["readout_id"].nunique() == 400
    return frame


class Stage1InteractionDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _canonical_fixture()

    def test_expert_contrasts_are_exact_and_have_expected_gain(self) -> None:
        contrasts = diagnostics.expert_contrasts(self.frame)
        self.assertEqual(len(contrasts), 400)
        selected = contrasts.query("E_low == 1 and E_high == 32")
        expected = 0.02 * math.log(32.0)
        np.testing.assert_allclose(selected["expert_gain"], expected, atol=1e-12)
        np.testing.assert_allclose(selected["expert_gain_per_log_e"], 0.02, atol=1e-12)

    def test_log_interpolation_never_extrapolates(self) -> None:
        curve = pd.DataFrame(
            {"D_t": [10.0, 100.0], "adjusted_loss": [2.0, 1.0]}
        )
        matched = diagnostics._log_interpolation(curve, math.sqrt(1000.0))
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertEqual(matched[1], "interpolated")
        self.assertAlmostEqual(matched[0], 1.5)
        self.assertIsNone(diagnostics._log_interpolation(curve, 9.0))
        self.assertIsNone(diagnostics._log_interpolation(curve, 101.0))

    def test_language_count_contrasts_are_in_support(self) -> None:
        contrasts = diagnostics.language_count_contrasts(self.frame)
        self.assertGreater(len(contrasts), 0)
        self.assertTrue(
            contrasts["D_t"].ge(contrasts["reference_lower_D_t"]).all()
        )
        self.assertTrue(
            contrasts["D_t"].le(contrasts["reference_upper_D_t"]).all()
        )
        selected = contrasts.query("K_low == 12 and K_high == 200")
        expected = 0.01 * math.log(200.0 / 12.0)
        np.testing.assert_allclose(
            selected["language_count_penalty"], expected, atol=1e-12
        )
        np.testing.assert_allclose(selected["penalty_per_log_k"], 0.01, atol=1e-12)

    def test_local_data_slope_recovers_fixture_exponent(self) -> None:
        slopes = diagnostics.local_data_slopes(self.frame)
        np.testing.assert_allclose(slopes["data_gain_per_log_D_t"], 0.1, atol=1e-12)

    def test_active_size_contrast_recovers_fixture_exponent(self) -> None:
        contrasts = diagnostics.active_size_contrasts(self.frame)
        np.testing.assert_allclose(
            contrasts["active_size_gain_per_log_N"], 0.2, atol=1e-12
        )

    def test_validation_rejects_active_size_drift_across_experts(self) -> None:
        broken = self.frame.copy()
        broken.loc[
            broken["trajectory_id"].eq("A:E32:K200"), "N_act"
        ] *= 1.01
        with self.assertRaises(diagnostics.InteractionDiagnosticError):
            diagnostics.expert_contrasts(broken)


if __name__ == "__main__":
    unittest.main()
