from dataclasses import replace
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import expert_coordinate_side_study as s
from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as m
from tests.test_stage1_m6_size_duration import fixture


class ExpertCoordinateTest(unittest.TestCase):
    def test_published_mapping_and_exact_endpoints(self):
        problem, x, _ = fixture()
        self.assertIs(s.coordinate_problem(problem, "raw"), problem)
        with self.assertRaises(ValueError):
            s.coordinate_problem(problem, "other")
        original = problem.E.copy()
        mapped = s.coordinate_problem(problem, "jointmoe_fixed")
        np.testing.assert_array_equal(original, problem.E)
        self.assertEqual(m.penalty(x, problem)[0], m.penalty(x, mapped)[0])
        np.testing.assert_array_equal(m.penalty(x, problem)[1], m.penalty(x, mapped)[1])
        mask = np.isin(problem.E, [1, 32])
        np.testing.assert_array_equal(mapped.E[mask], problem.E[mask])
        np.testing.assert_allclose(np.log(mapped.E), fit.expert_log_coordinate("JH", original), atol=1e-15)
        np.testing.assert_allclose(np.log(s.coordinate_problem(replace(problem, E=np.resize([1, 4, 16, 32], len(original))),
            "jointmoe_fixed").E[:4])/np.log(32), [0, .330722, .770838, 1], atol=1e-6)
        for name in ("N", "D", "K", "D_family", "D_script", "D_other", "weights", "y"):
            np.testing.assert_array_equal(getattr(problem, name), getattr(mapped, name))
        for left, right in zip(m.terms(x, problem), m.terms(x, mapped)):
            np.testing.assert_array_equal(left[mask], right[mask])

    def test_failed_better_minimum_cannot_be_discarded(self):
        selected = {"objective": .1}
        self.assertFalse(s.known_minimum_ok(selected, [selected, {"objective": .09, "success": False}])["passed"])
        self.assertTrue(s.known_minimum_ok(selected, [selected, {"objective": .09, "feasible": False}])["passed"])

    def test_no_optimization_before_allocation_check(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), patch.object(s, "cell_profile") as profile:
            plan = {"output_root": directory, "cells": [{"starts": [{}]}]}
            with self.assertRaises(ValueError):
                s.attempt(plan, 0, 0)
            profile.assert_not_called()
            with patch.object(s, "load", return_value=plan), patch.object(s, "receipt", return_value={"audit_tasks": [{}]}):
                with self.assertRaises(ValueError):
                    s.audit(Path(directory)/"plan.json", 0)
            profile.assert_not_called()

    def test_plan_pairs_and_output_isolation(self):
        cells = []
        for name in s.PROTOCOL["models"]:
            for fold in s.p.FOLDS:
                for coordinate in s.PROTOCOL["coordinates"]:
                    cells.append(dict(cell_id=len(cells), model=name, fold=fold, coordinate=coordinate,
                        starts=[{"start_id": str(i)} for i in range(48)], bounds=[]))
        plan = dict(schema_version=s.SCHEMA, protocol=s.PROTOCOL, cells=cells, repository_root="/repo",
            output_root="/repo/result_analysis/stage1_expert_coordinate_side_study_v1", protected=[{"path": "/repo/result_analysis/main/report.json"}])
        s.validate_plan(plan)
        broken = copy.deepcopy(plan)
        broken["cells"][1]["starts"][0]["coordinates"] = [0]
        with self.assertRaises(ValueError):
            s.validate_plan(broken)
        broken = copy.deepcopy(plan)
        broken["output_root"] = "/repo/result_analysis/main"
        with self.assertRaises(ValueError):
            s.validate_plan(broken)

    def test_policy_raw_parity_and_physical_accounting(self):
        from multilingual_megatron.experiments.scaling.stage1_fit import expert_coordinate_reporting as reporting
        from result_analysis.compute_allocation import policy_validation as story
        problem, x, _ = fixture()
        selected = m.Profile(problem, .05).evaluate(x)[0]
        records = []
        for ladder, experts, n, stored in (("A", 4, 5e7, 1e8), ("B", 16, 1e8, 2e8)):
            for language in ("a", "b"):
                for tokens in (1e8, 2e8, 4e8, 8e8):
                    records.append(dict(ladder=ladder, E=experts, N_act=n, N_total=stored,
                        language=language, D_total=tokens, D_t=tokens/200, D_family=tokens*.3,
                        D_script=tokens*.2, D_other=tokens*.495, K=200,
                        adjusted_loss=-3-.1*np.log(tokens/1e8)-.01*experts))
        rows = pd.DataFrame(records)
        original = rows.copy(deep=True)
        cohorts = {"common12": ["a"], "incumbent50": ["a", "b"]}
        plan = dict(decision_inputs={}, cohorts=cohorts, smoothing_penalty=1.)
        expected, decisions = story.decision_analysis(rows, {"M6-ND": selected}, cohorts, 1.)
        with patch.object(reporting.p.diagnosis, "bound_file", return_value="fixture"), patch.object(reporting.pd, "read_parquet", return_value=rows):
            actual, choices = reporting.policy_metrics(plan, {"M6-ND": ("raw", selected)})
            transformed, _ = reporting.policy_metrics(plan, {"M6-ND": ("jointmoe_fixed", selected)})
        pd.testing.assert_frame_equal(actual, expected[actual.columns])
        pd.testing.assert_frame_equal(choices, decisions[choices.columns])
        pd.testing.assert_frame_equal(rows, original)
        pd.testing.assert_series_equal(actual.candidate_count, transformed.candidate_count)
        pd.testing.assert_series_equal(actual.cap, transformed.cap)


if __name__ == "__main__":
    unittest.main()
