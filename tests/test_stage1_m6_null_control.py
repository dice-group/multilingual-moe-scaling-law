from types import SimpleNamespace
from dataclasses import replace
import unittest

from tests.test_stage1_m6_size_duration import fixture
from multilingual_megatron.experiments.scaling.stage1_fit import m6_null_control as c
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as m


class NullControlTest(unittest.TestCase):
    def test_fixed_null_is_never_an_optimized_fifteenth_parameter(self):
        problem, x, bounds = fixture()
        old_data = m.terms(x, problem)[1]
        x[-1] = 0.
        problem = replace(problem, y=problem.y+.7*(m.terms(x, problem)[1]-old_data))
        # Keep this solver fixture inside the positive-exponent domain; real
        # controls use the original full manifest bounds, not these test boxes.
        local = tuple(SimpleNamespace(name=b.name, lower=max(b.lower, v-.04), upper=min(b.upper, v+.04), kind=b.kind)
            for b, v in zip(bounds[:14], x[:14]))
        fixed = (*local, SimpleNamespace(name="zeta", lower=0., upper=0., kind="fixed"))
        profile = m.Profile(problem, .05)
        selected = profile.run(x, fixed)
        self.assertTrue(selected["success"])
        self.assertEqual(selected["coordinates"][14], 0.)
        direct = m.Direct(profile, selected, fixed).run()
        self.assertTrue(direct["success"])
        self.assertEqual(direct["coordinates"][14], 0.)
        self.assertTrue(m.equivalence(selected, direct, problem)["equivalent"])
        health = c.health(profile, selected, fixed)
        self.assertEqual(health["required_rank"], 14)
        self.assertEqual(health["shared_prediction_jacobian_rank"], 14)
        self.assertNotIn("zeta", [v["name"] for v in health["rejected_boundary_contacts"]])

    def test_nonzero_zeta_rejected_even_if_small(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            c.health(None, {"coordinates": [0.]*14+[1e-15]}, None)


if __name__ == "__main__":
    unittest.main()
