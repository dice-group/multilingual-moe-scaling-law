from dataclasses import replace
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from tests.test_stage1_m6_size_duration import fixture
from multilingual_megatron.experiments.scaling.stage1_fit import fit, size_duration_ablations as a
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as m


class AblationTest(unittest.TestCase):
    def test_exact_nested_forms_and_j0_zero_penalty(self):
        problem, x, bounds = fixture()
        for name, parent, count in (("J0-ND", "J0", 6), ("M5-ND", "M5", 11)):
            z = a.restrict(x, name)
            z[-1] = 0.
            actual = m.terms(z, problem)
            expected = fit.predictor_terms_and_jacobian(parent, z[:count], problem.N, problem.D, problem.E, problem.K)
            for i in (0, 1):
                np.testing.assert_allclose(actual[i], expected[i], atol=1e-14, rtol=1e-14)
            if parent == "J0":
                self.assertEqual(m.penalty(z, problem)[0], 0.)

    def test_free_coordinate_basis_and_profile_gradients(self):
        problem, x, bounds = fixture()
        for name in a.FIXED:
            z = a.restrict(x, name)
            free = [i for i in range(15) if i not in a.FIXED[name]]
            def expand(y):
                v = z.copy()
                v[free] = y
                return v
            profile = m.Profile(problem, .05)
            numeric = approx_derivative(lambda y: profile.evaluate(expand(y))[0]["objective"], z[free], abs_step=1e-6)
            analytic = profile.evaluate(z)[1][free]
            np.testing.assert_allclose(numeric.ravel(), analytic, atol=1e-8, rtol=1e-5)
            self.assertEqual(len(free), a.PROTOCOL["free_shared_coordinates"][name])

    def test_direct_solver_retains_fixed_coordinates(self):
        problem, x, bounds = fixture()
        z = a.restrict(x, "J0-ND")
        cap, data = m.terms(z, problem)[:2]
        problem = replace(problem, y=-3+.3*cap+.7*data)
        bounds = a.restricted_bounds(bounds, "J0-ND")
        profile = m.Profile(problem, .05)
        selected = profile.evaluate(z)[0] | {"success": True}
        direct = m.Direct(profile, selected, bounds).run()
        self.assertTrue(direct["success"])
        np.testing.assert_array_equal(np.asarray(direct["coordinates"])[list(a.FIXED["J0-ND"])], 0.)

    def test_nonzero_ablated_coordinate_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            a.health(None, {"coordinates": [1.]*15}, None, "J0-ND")


if __name__ == "__main__":
    unittest.main()
