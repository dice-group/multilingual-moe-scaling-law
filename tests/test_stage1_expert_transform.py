"""Matched expert-coordinate ablation: equations, gradients and campaign gates."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import (
    campaign, contracts as c, expert_transform as et, numerics, replay, reporting,
)
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.model import (
    Model, Reduction, starts, validate,
)
from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as legacy
from tests.test_stage1_law_improvement_round2 import fixture
from tests.test_stage1_law_search import source_fixture


def profile():
    return json.loads(Path("configs/stage1_fit/expert_transform_v1.json").read_text())


class ExpertTransformTest(unittest.TestCase):
    def test_parallel_amendment_only_changes_execution(self):
        p = json.loads(Path("configs/stage1_fit/expert_transform_parallel_v1.json").read_text())
        rules = c.rules_for(p)
        c.validate_profile(p)
        for key, value in c.EXPERT_RULES.items():
            if key not in ("schema", "protocol", "starts_per_shard"):
                self.assertEqual(rules[key], value)
        tasks = c.tasks(p)
        self.assertEqual(len(tasks), 1568)
        self.assertEqual(c.reservation(p), 12548.)
        for cell in c.cells(rules):
            group = [t for t in tasks if t["stage"] == "starts" and t["cell"] == cell["id"]]
            self.assertEqual([t["start_indices"] for t in group], [[i] for i in range(48)])
        with self.assertRaises(ValueError): c.validate_profile(p | {"concurrency": 1537})
        with self.assertRaises(ValueError): c.validate_profile(profile() | {"concurrency": 1536})
        plan = dict(profile=p, tasks=tasks)
        script = campaign.script(Path("/tmp/parallel-fixture/plan.json"), plan, "starts")
        self.assertIn("#SBATCH --array=0-1535%1536", script)

    def test_literal_paper_equation_endpoints_and_monotonicity(self):
        e = np.array([1., 4., 16., 32., 128., 1e8])
        for s in (1/16, 1., 16.):
            for r in (0., .05, .5, .95):
                with self.subTest(s=s, r=r):
                    got, h, _, _ = et.coordinates(e, np.log(s), r)
                    expected = 1/(1/(e-1+s/(1-r))+r/s)
                    np.testing.assert_allclose(np.exp(got), expected, rtol=2e-14)
                    self.assertEqual(h[0], 0.)
                    self.assertEqual(h[3], 1.)
                    self.assertTrue(np.all(np.diff(got) > 0))
                    self.assertTrue(np.all((h[:4] >= 0)&(h[:4] <= 1)))
        loge, h, _, _ = et.coordinates(e, 0., 0.)
        np.testing.assert_array_equal(loge, np.log(e))
        np.testing.assert_array_equal(h, np.log(e)/np.log(32.))
        for bad in ([0.], [np.nan], [np.inf]):
            with self.assertRaises(ValueError): et.coordinates(bad, 0., 0.)

    def test_raw_null_counts_starts_and_stored_metadata(self):
        anchor = source_fixture()[2][:15]
        for raw, transformed in c.EXPERT_PAIRS.items():
            model, x, rows = fixture(transformed)
            reference, _, _ = fixture(raw)
            self.assertEqual(len(Reduction(model).groups), c.EXPERT_FORMS[transformed])
            self.assertEqual(model.penalty(x)[0], 0.)
            x[17:] = 0.
            for a, b in zip(model.terms(x)[:2], reference.terms(x)[:2]):
                np.testing.assert_array_equal(a, b)
            initial = starts(model, anchor)
            np.testing.assert_array_equal(initial, starts(model, anchor))
            np.testing.assert_array_equal(initial[:, :17], starts(reference, anchor)[:, :17])
            np.testing.assert_array_equal(initial[:12, 17:], 0.)
            for row in initial: validate(model, {"coordinates": row})
            if "stored" in transformed:
                with self.assertRaises(ValueError): Model(transformed, model.problem, model.inventory)

    def test_all_free_and_tied_gradients_and_independent_replay(self):
        for name in c.EXPERT_PAIRS.values():
            model, x, rows = fixture(name)
            for shape in ((.27, .3), (-1.8, .8), (1.8, .02)):
                x[17:] = shape
                cap, data, cj, dj = model.terms(x)
                record = dict(coordinates=x, language_coefficients=[dict(language="x", L_inf=-5., A=.3, B=.4)])
                np.testing.assert_allclose(legacy.predict(model, record), replay.predict(name, record, rows),
                                           atol=1e-11, rtol=0)
                reduction = Reduction(model)
                z = reduction.pack(x)
                for j, group in enumerate(reduction.groups):
                    with self.subTest(name=name, shape=shape, group=group):
                        step = np.eye(len(z))[j]*1e-6
                        plus, minus = model.terms(reduction.expand(z+step)), model.terms(reduction.expand(z-step))
                        for i, jac in ((0, cj), (1, dj)):
                            np.testing.assert_allclose((plus[i]-minus[i])/2e-6, jac[:, group].sum(axis=1),
                                                       atol=2e-7, rtol=2e-5)
            x[17:] = 0.
            base = model.terms(x)
            plus = x.copy(); plus[18] = 1e-8
            for i in (0, 1):
                np.testing.assert_allclose((model.terms(plus)[i]-base[i])/1e-8,
                                           base[i+2][:, 18], atol=2e-5, rtol=2e-5)

    def test_task_budget_selection_is_primary_pair_only(self):
        p = profile(); rules = c.rules_for(p)
        c.validate_profile(p)
        self.assertEqual(len(c.cells(rules)), 32)
        self.assertEqual(len(c.tasks(p)), 416)
        self.assertEqual(c.reservation(p), 3332.)
        records = [dict(form=n, fold=f, certified=True, complete=True, rmse=1.)
                   for n in c.EXPERT_FORMS for f in c.INNER]
        self.assertEqual(c.choose(records, rules)["chosen"]["form"], "RN13-psi-tied")
        for row in records:
            if row["form"] == "RN16-stored-reuse-ehat": row["rmse"] = .01
        self.assertEqual(c.choose(records, rules)["chosen"]["form"], "RN13-psi-tied")
        for row in records:
            if row["form"] == "RN15-psi-tied-ehat": row["rmse"] = .95
        self.assertEqual(c.choose(records, rules)["chosen"]["form"], "RN15-psi-tied-ehat")
        records[0]["certified"] = False
        with self.assertRaises(ValueError): c.choose(records, rules)

    def test_zero_ridge_and_shape_diagnostics(self):
        model, x, rows = fixture("RN15-psi-tied-ehat")
        rows["unigram_nll"] = -model.problem.physical_floors
        rows["adjusted_loss"] = model.problem.y
        rows["trajectory_id"] = [f"t{i}" for i in range(len(rows))]
        original = model.problem.regularization.as_dict() | {"lambda": .2}
        plan = dict(profile=profile(), training={"all_grid": {}},
                    calibrations={"all_grid": dict(regularization=original, delta=.15, anchor=[])})
        with patch.object(campaign.data, "read_training", return_value=(rows, None, None)):
            fitted, _, cal = campaign.make_model(plan, "all_grid", model.name)
        self.assertEqual(cal["regularization"]["lambda"], 0.)
        self.assertEqual(cal["calibration_regularization"], original)
        record = {"coordinates": x}
        fit = legacy.Profile(fitted, None)
        diagnostics = numerics.geometry(fit, record)
        self.assertAlmostEqual(diagnostics["expert_transform"]["E_start"], np.exp(x[17]))
        self.assertIn("contacts", diagnostics)

    def test_full_synthetic_selection_report_and_matched_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"results").mkdir()
            base, _, rows = fixture("RN13-psi-tied")
            rows["trajectory_id"] = [f"t{i}" for i in range(len(rows))]
            rows["readout_id"] = [f"r{i}" for i in range(len(rows))]
            rows["adjusted_loss"] = base.problem.y
            rows["unigram_nll"] = -base.problem.physical_floors
            rows["D_total"] = rows.D_t+rows.D_family+rows.D_script+rows.D_other
            rows.to_parquet(root/"rows.parquet", index=False)
            entry = campaign.binding(root/"rows.parquet")
            pd.DataFrame([dict(model="published", category="published_form_adaptation",
                fold=fold, case=case, certified=True, complete=True, rmse=1.)
                for fold in (*c.PRIMARY, "all_grid") for case in
                (["L26"] if fold == "all_grid" else ["grid", "L26"])]).to_csv(root/"baseline.csv", index=False)
            members = {f: dict(train_trajectory_ids=[], test_trajectory_ids=rows.trajectory_id.tolist(),
                train_readout_ids=[], test_readout_ids=[], score_languages=["x"]) for f in c.PRIMARY}
            plan = dict(sha256="fixture", profile=profile(), tasks=c.tasks(profile()), memberships=members,
                inner_scoring={f: [entry] for f in c.INNER},
                scoring={"grid.parquet": entry, "endpoint.parquet": entry},
                baseline_metrics=campaign.binding(root/"baseline.csv"), reserved_core_hours=3332.)
            for task in (t for t in plan["tasks"] if t["stage"] == "certify"):
                model, x, _ = fixture(task["form"])
                x[17:] = 0.  # exact paired null
                result = dict(certified=True, complete=True, plan_sha256="fixture",
                    task_sha256=campaign.canonical_json_sha256(task), dependencies=[],
                    selected=dict(coordinates=x.tolist(), language_coefficients=[dict(language="x", L_inf=-5., A=.3, B=.4)]),
                    bounds=[asdict(b) for b in model.bounds], equalities=list(model.equalities),
                    regularization=model.problem.regularization.as_dict(), geometry={})
                io.write_json_once(campaign.result_path(root, task), result)
            with patch.object(reporting, "load", return_value=plan), patch.object(reporting, "cpu_guard"):
                self.assertTrue(reporting.select(root/"plan.json"))
                self.assertTrue(reporting.report(root/"plan.json"))
            paired = pd.read_csv(root/"report/paired_ablation.csv")
            self.assertEqual(len(paired), 24)
            self.assertTrue(paired.complete.all())
            np.testing.assert_allclose(paired.delta_rmse, 0., atol=1e-12)
            self.assertTrue(io.read_json(root/"selection.json")["frozen_before_outer_scoring"])


if __name__ == "__main__":
    unittest.main()
