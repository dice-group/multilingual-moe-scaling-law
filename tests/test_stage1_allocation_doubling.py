"""Support, architecture mapping and weighting for frozen-law diagnostics."""
import unittest
import numpy as np
import pandas as pd
from result_analysis.compute_allocation import doubling as d


class DoublingTest(unittest.TestCase):
    def test_regimes(self):
        self.assertEqual(d.regime(np.array([4.,6.,10.,11.]),10.).tolist(),
                         ['before','crossing','repeated','repeated'])

    def test_architecture_mapping(self):
        geometry = pd.DataFrame(dict(ladder=['A','B','C'],E=[4,4,4],
                                     N_act=[1.,2.,4.],N_total=[2.,5.,12.]))
        total,status,_ = d.stored_size(geometry,4,3.)
        self.assertEqual(total,8.5)
        self.assertEqual(status,'interpolated')
        self.assertEqual(d.stored_size(geometry,4,2.)[:2],(5.,'observed'))
        self.assertEqual(d.stored_size(geometry,4,8.)[:2],(26.,'extrapolated'))
        self.assertNotEqual(total,4*3.)

    def test_support_and_no_observation_extrapolation(self):
        g=pd.DataFrame(dict(D_t=[1.,2.,4.],adjusted_loss=[3.,2.,1.]))
        self.assertEqual(d.exposure_status(g,2.),'observed')
        self.assertEqual(d.exposure_status(g,3.),'interpolated')
        self.assertEqual(d.combined_status(['interpolated','extrapolated']),'extrapolated')
        for method in ('linear_log','pchip_log'):
            self.assertTrue(np.isnan(d.interpolate_loss(g,8.,method)))
            self.assertAlmostEqual(d.interpolate_loss(g,2.,method),2.)

    def test_configuration_balanced_weights(self):
        frame=pd.DataFrame(dict(K=[12]*3,trajectory_id=['A','B','B'],
            readout_id=['a','b','c'],language=['en']*3,gain=[0.,1.,1.]))
        summary=d.aggregate(frame,['K'],['gain'])
        self.assertEqual(summary.gain.iloc[0],.5)


if __name__ == '__main__':
    unittest.main()
