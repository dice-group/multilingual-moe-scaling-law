"""Small non-paper panels exercise the reusable scientific entrypoints."""
import unittest

import numpy as np
import pandas as pd

from result_analysis.capacity_response import analysis as capacity
from result_analysis.scaling_law_fit import analysis as laws
from result_analysis.compute_allocation import analysis as allocation
from result_analysis.language_preservation import analysis as preservation
from result_analysis.downstream_evaluations import analysis as downstream


def observations(gap=.1):
    rows = []
    for k in (2, 5):
        for i, tokens in enumerate(np.geomspace(1e8, 1e10, 8)):
            loss = 5-np.log(tokens/1e8)+(gap if k == 5 else 0)
            rows.append(dict(ladder="custom", E=2, K=k, language="x", N_act=1e8,
                D_total=tokens, D_t=tokens, raw_nll=loss+2, adjusted_loss=loss,
                trajectory_id=f"k{k}", readout_id=str(i)))
    return pd.DataFrame(rows)


def geometry():
    return pd.DataFrame([dict(ladder="custom", E=2, N_act=1e8, N_total=2e8)])


def coverage_observations():
    return observations().rename(columns={"E": "expert_count", "K": "language_coverage",
        "N_act": "active_parameters", "D_total": "total_tokens", "D_t": "target_exposure"})


class ResearchWorkflows(unittest.TestCase):
    def test_capacity_custom_coverage(self):
        answer = capacity.analyze(coverage_observations(), reference_coverage=2, target_coverage=5,
                                  reference_expert_count=2)
        self.assertAlmostEqual(answer["summary"].coverage_gap.iloc[0], .1)
        self.assertAlmostEqual(answer["summary"].gap_reduction.iloc[0], 0.)
        self.assertEqual(len(answer["matched_points"]), 8)

    def test_capacity_rejects_mixed_trajectories(self):
        rows = coverage_observations()
        extra = rows.iloc[[0]].assign(trajectory_id="another-seed")
        with self.assertRaisesRegex(ValueError, "one trajectory"):
            capacity.analyze(pd.concat([rows, extra]), reference_coverage=2, target_coverage=5)

    def test_coverage_gap_checks_offset_geometry_and_zero_reference(self):
        rows = coverage_observations()
        settings = dict(reference_coverage=2, target_coverage=5)
        inconsistent = rows.copy()
        inconsistent.loc[inconsistent.language_coverage.eq(5), "raw_nll"] += 1.
        with self.assertRaisesRegex(ValueError, "raw coverage gaps"):
            capacity.analyze(inconsistent, **settings)
        inconsistent = rows.copy()
        inconsistent.loc[inconsistent.language_coverage.eq(5), "active_parameters"] *= 2
        with self.assertRaisesRegex(ValueError, "equal active parameters"):
            capacity.analyze(inconsistent, **settings)
        zero = rows.copy()
        zero["adjusted_loss"] = 1.
        zero["raw_nll"] = 3.
        with self.assertRaisesRegex(ValueError, "nonzero reference"):
            capacity.analyze(zero, reference_expert_count=2, **settings)

    def test_laws_custom_folds_and_two_language_endpoint(self):
        rows = pd.DataFrame([dict(model=m, fold=f, trajectory_id=t, readout_id=i,
            language="x", adjusted_loss=1., prediction=1.+error)
            for m, error in (("new", .1), ("control", .2)) for f in ("a", "b")
            for t, count in (("short", 1), ("long", 3)) for i in range(count)])
        end = pd.DataFrame([dict(model=m, language=l, adjusted_loss=2., prediction=2.+e)
            for m, e in (("new", .3), ("control", .4)) for l in ("x", "y")])
        scores = laws.analyze(rows, folds=("a", "b"), endpoint_predictions=end,
                              endpoint_languages=("x", "y"))
        self.assertAlmostEqual(scores.loc["new", "mean_rmse"], .1)
        self.assertAlmostEqual(scores.loc["new", "endpoint_rmse"], .3)
        self.assertTrue(laws.analyze(rows, folds=("a", "missing")).mean_rmse.isna().all())
        with self.assertRaisesRegex(ValueError, "population"):
            laws.analyze(rows.iloc[1:], folds=("a", "b"))
        with self.assertRaisesRegex(ValueError, "cohort"):
            laws.analyze(rows, folds=("a",), endpoint_predictions=end.iloc[1:],
                         endpoint_languages=("x", "y"))

    def test_configuration_balancing_not_row_weighting(self):
        rows = pd.DataFrame([dict(model="m", fold="f", trajectory_id=t,
            readout_id=i, language="x", adjusted_loss=0., prediction=e)
            for t, count, e in (("short", 1, 0.), ("long", 3, 2.)) for i in range(count)])
        self.assertAlmostEqual(laws.analyze(rows, folds=("f",)).loc["m", "f"], np.sqrt(2))

    def test_allocation_custom_cap_and_no_switch(self):
        answer = allocation.analyze(observations(), geometry(), languages=("x",),
            coverages=(2, 5), cap=3e8, penalty=1e-6, methods=("linear_log",), grid_points=9)
        self.assertTrue(answer["switches"].empty)
        self.assertIn("F", answer["switches"])
        self.assertEqual(len(answer["loss_curves"]), 18)
        self.assertEqual(set(answer["loss_curves"].K), {2, 5})
        with self.assertRaisesRegex(ValueError, "geometry"):
            allocation.analyze(observations(), geometry().assign(N_act=2e8),
                languages=("x",), coverages=(2, 5), cap=3e8, penalty=1e-6)

    def test_preservation_new_expansion_and_unreached_status(self):
        settings = dict(languages=("x",), base_k=2, target_k=5, cap=3e8,
                        baseline_f=.6, penalty=1e-6)
        answer = preservation.analyze(observations(), geometry(), **settings)
        self.assertEqual(set(answer["policies"].base_K), {2})
        self.assertEqual(set(answer["policies"].target_K), {5})
        np.testing.assert_allclose(answer["language_deltas"].loss_delta, 0., atol=1e-9)
        unavailable = preservation.analyze(observations(gap=100.), geometry(), **settings)
        self.assertTrue(unavailable["policies"].empty)
        self.assertTrue(unavailable["summary"].empty)
        self.assertEqual(set(unavailable["candidates"].status), {"unreached"})

    def test_downstream_custom_metrics_without_training_losses(self):
        frame = pd.DataFrame([dict(model_id=f"k{k}e{e}", ladder="new", E=e, K=k,
            seed=7, benchmark="custom", language=l, corpus_language=l, panel="test",
            score=(e/8)*(k/5)+(1 if l == "y" else 0))
            for k in (2, 5) for e in (2, 8) for l in ("x", "y")])
        settings = dict(languages=("x", "y"), coverages=(2, 5), coverage_pair=(2, 5),
            expert_pairs=((2, 8),), metric_directions={("custom", "score"): 1})
        answer = downstream.analyze(frame, **settings)
        self.assertAlmostEqual(answer["coverage_interactions"].interaction.iloc[0], .45)
        self.assertTrue(answer["expert_gains"].validation_nll_reduction.isna().all())
        self.assertIn("gain_K5", answer["coverage_interactions"])
        with self.assertRaisesRegex(ValueError, "unpaired"):
            downstream.analyze(frame.iloc[1:], **settings)
        frame.loc[0, "score"] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            downstream.analyze(frame, **settings)


if __name__ == "__main__":
    unittest.main()
