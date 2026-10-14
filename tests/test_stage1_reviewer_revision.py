"""Checks that reviewer diagnostics preserve scientific scoring populations."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.scaling_law_fit.diagnostics import (
    KEYS, composition_diagnostics, matched_rows,
)


class ReviewerDiagnosticsTest(unittest.TestCase):
    def test_missing_extra_and_duplicate_predictions_are_rejected(self):
        left = pd.DataFrame([['fold', 'a', 'r1', 'x'], ['fold', 'b', 'r2', 'y']], columns=KEYS)
        for right in [left.iloc[:1], pd.concat([left, left.iloc[:1]]),
                      left.assign(language=['x', 'z'])]:
            with self.assertRaises(ValueError):
                matched_rows(left, right)

    def test_long_trajectories_do_not_dominate_attribution(self):
        observations = [('a', 'r1', 'x', 1., 0.),
                        ('b', 'r2', 'y', 0., 1.), ('b', 'r3', 'y', 0., 1.),
                        ('b', 'r4', 'y', 0., 1.)]
        rows = []
        for trajectory, readout, language, e5, e6 in observations:
            for model, error in [('M5-ND', e5), ('M6-ND', e6)]:
                rows.append(dict(fold='testfold', trajectory_id=trajectory, readout_id=readout,
                    language=language, model=model, split='test', squared_error=error))
        meta = pd.DataFrame(dict(tier=['K200']*2, language=['x','y'], name=['X','Y'],
            family=['A','B'], production_script=['Latn','Latn'], resource_availability=['UNKNOWN','unknown']))
        result = composition_diagnostics(pd.DataFrame(rows), meta)
        overall = result.loc[result.dimension.eq('all')].iloc[0]
        np.testing.assert_allclose([overall.rmse_m5, overall.rmse_m6], np.sqrt(.5))
        resource = result.loc[result.dimension.eq('resource_availability')]
        self.assertEqual(resource.group.tolist(), ['UNKNOWN'])
        np.testing.assert_allclose(resource.scoring_weight, 1.)
        language = result.loc[result.dimension.eq('language')]
        np.testing.assert_allclose(language.fold_mse_contribution, [-.5, .5])


if __name__ == '__main__':
    unittest.main()
