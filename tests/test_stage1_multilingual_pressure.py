"""Focused tests for the maintained Stage 1 pressure derivations."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from result_analysis.compute_allocation.matched_compute import (
    allocation_contrasts,
    interpolate_fixed_compute,
    select_allocation_pairs,
)
from result_analysis.capacity_response.tables import (
    load_primary,
    matched_pressure,
    summarize,
    validate_primary_support,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "result_analysis/stage1_abcd_jointmoe_x1_v1/release_manifest.json"


class MatchedPressureUnitTests(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trajectory_id": "K12",
                    "readout_id": "K12-1",
                    "ladder": "A",
                    "E": 1,
                    "K": 12,
                    "D_total": 10.0,
                    "D_t": 1.0,
                    "language": "eng",
                    "adjusted_loss": 4.0,
                    "raw_nll": 5.0,
                    "N_act": 100.0,
                },
                {
                    "trajectory_id": "K12",
                    "readout_id": "K12-2",
                    "ladder": "A",
                    "E": 1,
                    "K": 12,
                    "D_total": 40.0,
                    "D_t": 4.0,
                    "language": "eng",
                    "adjusted_loss": 2.0,
                    "raw_nll": 3.0,
                    "N_act": 100.0,
                },
                {
                    "trajectory_id": "K200",
                    "readout_id": "K200-in",
                    "ladder": "A",
                    "E": 1,
                    "K": 200,
                    "D_total": 20.0,
                    "D_t": 2.0,
                    "language": "eng",
                    "adjusted_loss": 3.5,
                    "raw_nll": 4.5,
                    "N_act": 100.0,
                },
                {
                    "trajectory_id": "K200",
                    "readout_id": "K200-out",
                    "ladder": "A",
                    "E": 1,
                    "K": 200,
                    "D_total": 80.0,
                    "D_t": 8.0,
                    "language": "eng",
                    "adjusted_loss": 2.0,
                    "raw_nll": 3.0,
                    "N_act": 100.0,
                },
            ]
        )

    def test_interpolates_reference_in_log_exposure_without_extrapolation(self) -> None:
        points = matched_pressure(self._frame())
        self.assertEqual(len(points), 1)
        self.assertEqual(points.iloc[0].target_readout_id, "K200-in")
        self.assertAlmostEqual(points.iloc[0].reference_loss, 3.0)
        self.assertAlmostEqual(points.iloc[0].pressure, 0.5)

    def test_duplicate_reference_exposure_fails(self) -> None:
        frame = self._frame()
        duplicate = frame.iloc[[0]].assign(readout_id="K12-duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate reference target exposure"):
            matched_pressure(pd.concat([frame, duplicate], ignore_index=True))

    def test_summary_weights_ladders_equally(self) -> None:
        points = pd.DataFrame(
            {
                "ladder": ["A", "A", "B"],
                "N_act": [100.0, 100.0, 200.0],
                "E": [1, 1, 1],
                "pressure": [0.0, 2.0, 3.0],
            }
        )
        by_ladder, summary = summarize(points, expected_experts=(1,))
        self.assertEqual(len(by_ladder), 2)
        self.assertAlmostEqual(summary.iloc[0].pressure, 2.0)


class AllocationUnitTests(unittest.TestCase):
    @staticmethod
    def _configs() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ladder": "A", "E": 4, "N_act": 25.0, "N_total": 100.0, "R": 4.0},
                {"ladder": "B", "E": 1, "N_act": 100.0, "N_total": 105.0, "R": 1.05},
                {"ladder": "C", "E": 1, "N_act": 300.0, "N_total": 300.0, "R": 1.0},
            ]
        )

    def test_pair_selection_uses_geometry_only(self) -> None:
        pairs = select_allocation_pairs(self._configs())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0].higher_R, "A/E4")
        self.assertEqual(pairs.iloc[0].lower_R, "B/E1")
        self.assertAlmostEqual(pairs.iloc[0].N_total_mismatch_pct, 5.0)

    def test_contrast_does_not_require_positive_amplification(self) -> None:
        pairs = select_allocation_pairs(self._configs())
        advantages = {12: 0.10, 50: 0.08, 200: 0.05}
        rows = []
        for budget in (2.0, 4.0):
            for coverage, advantage in advantages.items():
                rows.extend(
                    [
                        {
                            "config": "A/E4",
                            "F_EFLOPs": budget,
                            "K": coverage,
                            "adjusted_loss": 1.0,
                            "raw_nll": 2.0,
                        },
                        {
                            "config": "B/E1",
                            "F_EFLOPs": budget,
                            "K": coverage,
                            "adjusted_loss": 1.0 + advantage,
                            "raw_nll": 2.0 + advantage,
                        },
                    ]
                )
        contrasts = allocation_contrasts(pd.DataFrame(rows), pairs)
        self.assertEqual(len(contrasts), 2)
        self.assertTrue(np.allclose(contrasts.K200_minus_K12, -0.05))


@unittest.skipUnless(RELEASE.exists(), "sealed Stage 1 release is unavailable")
class Stage1PressureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_primary(ROOT)

    def test_primary_result_matches_sealed_reference(self) -> None:
        points = matched_pressure(self.frame)
        validate_primary_support(points)
        _, summary = summarize(points)
        pressure = summary.set_index("E").pressure
        self.assertAlmostEqual(pressure.loc[1], 0.07955117501993697, places=12)
        self.assertAlmostEqual(pressure.loc[32], 0.0135370765661889, places=12)
        attenuation = 1 - pressure.loc[32] / pressure.loc[1]
        self.assertAlmostEqual(attenuation, 0.8298318464460611, places=12)

    def test_empirical_allocation_matches_paper_audit(self) -> None:
        pairs = select_allocation_pairs(self.frame)
        self.assertEqual(
            pairs.pair.to_list(),
            ["C/E32 vs D/E16", "A/E16 vs B/E4", "A/E16 vs C/E1", "B/E4 vs C/E1"],
        )
        fixed = interpolate_fixed_compute(self.frame)
        contrasts = allocation_contrasts(fixed, pairs)
        self.assertEqual(len(fixed), 96)
        self.assertEqual(len(contrasts), 8)
        self.assertAlmostEqual(contrasts.K200_minus_K12.min(), 0.012042055322960099, places=12)
        self.assertAlmostEqual(contrasts.K200_minus_K12.max(), 0.05460464012829114, places=12)


if __name__ == "__main__":
    unittest.main()
