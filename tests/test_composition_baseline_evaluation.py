"""Changed mixtures must preserve source identity and frozen donor semantics."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.scaling_law_fit.composition_baselines import identify
from multilingual_megatron.experiments.scaling.stage1_fit.law_search.data import donor_covariates


class CompositionBaselineTests(unittest.TestCase):
    def test_arm_seed_expert_and_horizon_do_not_alias(self):
        rows = pd.DataFrame([
            dict(arm=arm, seed=seed, E=expert, horizon=horizon, language="target")
            for arm in ("compact", "diverse") for seed in (1, 2)
            for expert in (1, 16) for horizon in (10, 20)])
        panel = identify(rows)
        self.assertEqual(panel.readout_id.nunique(), 16)
        self.assertEqual(panel.trajectory_id.nunique(), 8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            identify(pd.concat([rows, rows.iloc[:1]]))

    def test_absent_frozen_donors_and_arm_specific_inventory(self):
        metadata = {l: dict(family="f", script="s") for l in ("target", "d1", "d2", "d3", "other")}
        rows = identify(pd.DataFrame([dict(arm="compact", seed=1, E=16, horizon=10,
            language="target", D_t=2., D_total=10., K=12, N_act=100., N_total=300.)]))
        source = pd.DataFrame(dict(readout_id=[rows.readout_id.iloc[0]] * 3,
            language=["target", "d1", "other"], D_t=[2., 3., 5.]))
        inventory = {("target", 12): 1., ("d1", 12): 2., ("other", 12): 4.}
        first = donor_covariates(rows, source, inventory, metadata, "target", ["d1", "d2", "d3"])
        second = donor_covariates(rows, source, inventory | {("other", 12): 40.}, metadata, "target", ["d1", "d2", "d3"])
        np.testing.assert_array_equal(first["D"], [[2., 3., 0., 0., 5.]])
        np.testing.assert_array_equal(first["U"], [[1., 2., 0., 0., 4.]])
        np.testing.assert_array_equal(second["U"], [[1., 2., 0., 0., 40.]])
        # No donor reselection: a present remainder language stays in remainder.
        self.assertEqual(np.sum(first["D"]), 10.)
        with self.assertRaisesRegex(ValueError, "covariates only"):
            donor_covariates(rows, source.assign(adjusted_loss=0.), inventory, metadata, "target", ["d1", "d2", "d3"])


if __name__ == "__main__":
    unittest.main()
