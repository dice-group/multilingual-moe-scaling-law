import unittest

import numpy as np

from result_analysis.compute_allocation.policy_validation import interpolate_exposure


class JointStoryTest(unittest.TestCase):
    def test_cumulative_exposure_conservation_and_zero_components(self):
        tokens = np.array([10., 20., 50.])
        components = np.array([[2., 0., 3., 5.], [5., 0., 4., 11.], [10., 0., 12., 28.]])
        np.testing.assert_array_equal(interpolate_exposure(tokens, components, tokens), components)
        query = np.array([11., 15., 37., 50.])
        result = interpolate_exposure(tokens, components, query)
        np.testing.assert_allclose(result.sum(axis=1), query)
        np.testing.assert_array_equal(result[:, 1], 0.)
        self.assertTrue(np.all(result >= 0))

    def test_unsupported_queries_and_invalid_accounting_rejected(self):
        for query in ([5.], [21.]):
            with self.assertRaisesRegex(ValueError, "support"):
                interpolate_exposure([10., 20.], [[10., 0., 0., 0.], [20., 0., 0., 0.]], query)
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            interpolate_exposure([10., 20.], [[9., 0., 0., 0.], [20., 0., 0., 0.]], [15.])



if __name__ == "__main__":
    unittest.main()
