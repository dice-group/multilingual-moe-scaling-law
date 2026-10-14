"""Round-3 literal equations, nested nulls and workflow contracts; no study fits."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as legacy
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import campaign, contracts as c, diagnostics, replay, reporting
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.model import Model, Reduction, starts, validate
from tests.test_stage1_coverage_laws import profile as base_profile
from tests.test_stage1_law_improvement_round2 import fixture
from tests.test_stage1_law_search import source_fixture


NEW = {"RN13-psi-tied-stored": 13, "Cap-burden-stored-13": 13,
       "Cap-burden-stored-gamma14": 14}
FORMS = NEW | {"RN14-stored-reuse": 14, "RN13-psi-tied": 13,
               "Cap-burden-13": 13, "RN14-no-ridge": 14}


def profile():
    return base_profile() | dict(protocol="law_improvement_round3_v1", max_reserved_core_hours=3000)


def oracle(name, x, rows):
    """Independent literal round-3 algebra, without production equation helpers."""
    n, m, e, k, d, u = (rows[key].to_numpy() for key in
                         ("N_act", "N_total", "E", "K", "D_t", "U_t"))
    n, m = n/1e8, m/1e8
    h = np.log(e)/np.log(32.)
    g = np.maximum(np.log(k/12.), 0.)/np.log(200./12.)
    alpha, beta = x[0]*(1-h)+x[1]*h, x[2]*(1-h)+x[3]*h
    psi = np.full(len(n), x[8]) if name == "RN13-psi-tied-stored" else x[8]*(1-h)+x[9]*h
    if name.startswith("Cap-burden"):
        capacity = e**x[4]*n**(-alpha)+x[17]*np.log(k)*n**(-x[0])*e**(-x[18])
        if name == "Cap-burden-stored-gamma14":
            capacity *= n**(-x[10]*g)
    else:
        capacity = e**x[4]*k**(x[6]*(1-h)+x[7]*h)*n**(-alpha-x[10]*g)
    rate = x[15]*m**x[16]
    useful = d.copy()
    mask = (d > u) & (rate > 0)
    useful[mask] = u[mask]*(1-np.expm1(-rate[mask]*(d[mask]/u[mask]-1))/rate[mask])
    data = e**x[5]*k**psi*n**x[14]*(useful/1e9)**(-beta)
    return capacity, data


class LawImprovementRound3Test(unittest.TestCase):
    def test_counts_zero_ridge_and_independent_replay(self):
        self.assertEqual(c.ROUND3_FORMS, FORMS)
        for name, count in FORMS.items():
            with self.subTest(name=name):
                model, x, rows = fixture(name)
                validate(model, {"coordinates": x})
                self.assertEqual(len(Reduction(model).groups), count)
                self.assertEqual(model.penalty(x)[0], 0.)
                np.testing.assert_array_equal(model.penalty(x)[1], np.zeros(19))
                np.testing.assert_array_equal(x[11:14], 0.)
                if name not in NEW:
                    continue
                cap, data, *_ = model.terms(x)
                for actual, expected in zip((cap, data), oracle(name, x, rows)):
                    self.assertTrue((actual > 0).all())
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                record = dict(coordinates=x, language_coefficients=[dict(language="x", L_inf=-5., A=.3, B=.4)])
                expected = -5.+.3*cap+.4*data
                np.testing.assert_allclose(legacy.predict(model, record), expected, atol=1e-12, rtol=0)
                np.testing.assert_allclose(replay.predict(name, record, rows), expected, atol=1e-12, rtol=0)

    def test_all_free_and_tied_derivatives(self):
        for name in NEW:
            model, x, _ = fixture(name)
            if name.endswith("gamma14"):
                x[10] = .17
            _, _, cj, dj = model.terms(x)
            reduction = Reduction(model)
            z = reduction.pack(x)
            for j, group in enumerate(reduction.groups):
                with self.subTest(name=name, group=group):
                    step = np.eye(len(z))[j]*1e-6
                    plus, minus = model.terms(reduction.expand(z+step)), model.terms(reduction.expand(z-step))
                    for index, jac in ((0, cj), (1, dj)):
                        np.testing.assert_allclose((plus[index]-minus[index])/2e-6,
                            jac[:, group].sum(axis=1), atol=2e-8, rtol=5e-6)
                    numerical = (.3*(plus[0]-minus[0])+.4*(plus[1]-minus[1]))/2e-6
                    np.testing.assert_allclose(numerical, ((.3*cj+.4*dj)@reduction.matrix)[:, j],
                                               atol=2e-8, rtol=5e-6)

    def test_gamma_zero_exactly_nests_combined13(self):
        model, x, rows = fixture("Cap-burden-stored-gamma14")
        null, _, _ = fixture("Cap-burden-stored-13")
        x[10] = 0.
        for a, b in zip(model.terms(x)[:2], null.terms(x)[:2]):
            np.testing.assert_array_equal(a, b)
        x[10] = .23
        cap, term, cj, _ = model.terms(x)
        g = np.maximum(np.log(rows.K/12.), 0.)/np.log(200./12.)
        z = x.copy(); z[10] = 0.
        np.testing.assert_allclose(cap, null.terms(z)[0]*(rows.N_act/1e8)**(-.23*g), atol=1e-12)
        np.testing.assert_array_equal(term, null.terms(z)[1])
        np.testing.assert_array_equal(cj[rows.K <= 12, 10], 0.)
        np.testing.assert_array_equal(cap[rows.K == 1], null.terms(z)[0][rows.K == 1])
        x[17] = 0.
        np.testing.assert_array_equal(model.terms(x)[2][:, 18], 0.)
        for alpha1, alpha32, gamma in ((.1, .4, -.2), (.4, .1, -.2)):
            bad = x.copy(); bad[0], bad[1], bad[10] = alpha1, alpha32, gamma
            with self.assertRaisesRegex(ValueError, "positivity"):
                validate(model, {"coordinates": bad})

    def test_stored_and_active_exact_nulls_and_tied_psi(self):
        for name, active in (("RN13-psi-tied-stored", "RN13-psi-tied"),
                             ("Cap-burden-stored-13", "Cap-burden-13")):
            model, x, rows = fixture(name)
            reference, _, _ = fixture(active)
            equal = Model(name, model.problem, model.inventory, stored_parameters=model.problem.N)
            for a, b in zip(equal.terms(x)[:2], reference.terms(x)[:2]):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            for index in (15, 16):
                null = x.copy(); null[index] = 0.
                for a, b in zip(model.terms(null)[:2], reference.terms(null)[:2]):
                    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            bogus = Model(name, model.problem, model.inventory,
                          stored_parameters=model.problem.N*model.problem.E)
            self.assertGreater(np.max(abs(model.terms(x)[1]-bogus.terms(x)[1])), 1e-4)
        tied, x, _ = fixture("RN13-psi-tied-stored")
        full, _, _ = fixture("RN14-stored-reuse")
        for a, b in zip(tied.terms(x)[:2], full.terms(x)[:2]):
            np.testing.assert_array_equal(a, b)

    def test_required_stored_metadata_and_zero_rate_boundary(self):
        for name in NEW:
            model, x, rows = fixture(name)
            for bad in (None, model.problem.N[:-1], model.problem.N*.9,
                        model.problem.N*np.nan, model.problem.N*np.inf):
                with self.assertRaises(ValueError):
                    Model(name, model.problem, model.inventory, stored_parameters=bad)
            np.testing.assert_array_equal(model.terms(x)[3][rows.D_t <= rows.U_t, 15:17], 0.)
            null = x.copy(); null[15] = 0.
            plus = null.copy(); plus[15] = 1e-7
            current = model.terms(null)
            np.testing.assert_array_equal(current[3][:, 16], 0.)
            np.testing.assert_allclose((model.terms(plus)[1]-current[1])/1e-7,
                                       current[3][:, 15], atol=2e-6, rtol=1e-5)
            for a, b in zip(current[:2], oracle(name, null, rows)):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)

    def test_starts_reproducible_valid_and_gamma_null_anchors(self):
        _, _, anchor, _ = source_fixture()
        anchor = anchor[:15]
        for name in FORMS:
            model, _, _ = fixture(name)
            first, second = starts(model, anchor), starts(model, anchor)
            self.assertEqual(first.shape, (48, 19))
            np.testing.assert_array_equal(first, second)
            for x in first:
                validate(model, {"coordinates": x})
            if name.endswith("gamma14"):
                np.testing.assert_array_equal(first[:12, 10], 0.)
                self.assertGreater(np.max(abs(first[12:, 10])), 0.)
                null, _, _ = fixture("Cap-burden-stored-13")
                for x, y in zip(first[:12], starts(null, anchor)[:12]):
                    np.testing.assert_array_equal(x, y)

    def test_task_graph_budget_and_all_procedures_eligible_only_this_round(self):
        p = profile(); rules = c.rules_for(p)
        c.validate_profile(p)
        self.assertEqual(len(c.cells(rules)), 56)
        tasks = c.tasks(p)
        self.assertEqual(sum(t["stage"] == "starts" for t in tasks), 672)
        self.assertEqual(len(tasks), 728)
        self.assertEqual(c.reservation(p), 2914.)
        self.assertEqual(rules["eligible"], list(FORMS))
        self.assertTrue(rules["diagnostics_after_report"])
        for cell in c.cells(rules):
            self.assertEqual([i for t in tasks if t["stage"] == "starts" and t["cell"] == cell["id"]
                              for i in t["start_indices"]], list(range(48)))
        self.assertNotIn("eligible", c.rules_for({"protocol": "law_improvement_round2_v1"}))
        self.assertNotIn("diagnostics_after_report", c.rules_for())
        rows = [dict(form=name, fold=fold, certified=True, complete=True, rmse=1.)
                for name in FORMS for fold in c.INNER]
        self.assertEqual(c.choose(rows, rules)["chosen"]["form"], "RN13-psi-tied-stored")
        for row in rows:
            if row["form"] == "RN13-psi-tied": row["rmse"] = .95
        decision = c.choose(rows, rules)
        self.assertEqual(decision["chosen"]["form"], "RN13-psi-tied")
        for row in rows:
            if row["fold"] == "inner_terminal": row["rmse"] = 1e12
        self.assertEqual(c.choose(rows, rules), decision)
        for invalid in (list(FORMS)+[list(FORMS)[0]], ["unknown"], list(NEW)):
            with self.assertRaises(ValueError): c.choose(rows, rules | {"eligible": invalid})

    def test_prior_panel_keeps_failed_cases_and_distinct_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"metrics.csv"
            rows = pd.DataFrame([dict(model=name, role="candidate", category="new_candidate",
                        fold="all_grid", case="L26", certified=name != "failed", complete=name != "failed",
                        rmse=.2 if name != "failed" else None) for name in ("good", "failed")])
            rows.to_csv(path, index=False)
            entry = dict(metrics=campaign.binding(path), forms={"good": 13, "failed": 14}, prefix="round2::")
            result = reporting.prior_campaign_panel(entry)
            self.assertEqual(len(result), 14)
            self.assertEqual(set(result.model), {"round2::good", "round2::failed"})
            self.assertTrue(result.role.eq("historical").all())
            self.assertEqual(result.certified.sum(), 1)
            with self.assertRaises(ValueError):
                reporting.prior_campaign_panel(entry | {"forms": {"missing": 13}})
            pd.concat([rows, rows.iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                reporting.prior_campaign_panel(entry | {"metrics": campaign.binding(path)})

    def test_zero_ridge_receipt_preserves_calibration(self):
        model, _, rows = fixture("RN14-no-ridge")
        rows["unigram_nll"] = -model.problem.physical_floors
        rows["adjusted_loss"] = model.problem.y
        rows["trajectory_id"] = [f"t{i}" for i in range(len(rows))]
        original = model.problem.regularization.as_dict() | {"lambda": .2}
        plan = dict(profile=profile(), training={"all_grid": {}},
                    calibrations={"all_grid": dict(regularization=original, delta=.15, anchor=[])})
        with patch.object(campaign.data, "read_training", return_value=(rows, None, None)):
            fitted, _, calibration = campaign.make_model(plan, "all_grid", "RN13-psi-tied-stored")
        self.assertEqual(calibration["regularization"]["lambda"], 0.)
        self.assertEqual(calibration["calibration_regularization"], original)
        self.assertEqual(original["lambda"], .2)
        self.assertEqual(fitted.problem.regularization.as_dict()["lambda"], 0.)

    def test_synthetic_selection_report_and_auto_diagnostic(self):
        """Real file/replay/scoring pipeline; diagnostics mocked, no study optimization."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"results").mkdir()
            base, _, rows = fixture("RN14-no-ridge")
            rows["trajectory_id"] = [f"t{i}" for i in range(len(rows))]
            rows["readout_id"] = [f"r{i}" for i in range(len(rows))]
            rows["adjusted_loss"] = base.problem.y
            rows["unigram_nll"] = -base.problem.physical_floors
            rows["D_total"] = rows.D_t+rows.D_family+rows.D_script+rows.D_other
            rows.to_parquet(root/"rows.parquet", index=False)
            row_entry = campaign.binding(root/"rows.parquet")
            pd.DataFrame([dict(model="published", category="published_form_adaptation", role="baseline",
                fold=fold, case=case, certified=True, complete=True, rmse=1.)
                for fold in (*c.PRIMARY, "all_grid") for case in
                (["L26"] if fold == "all_grid" else ["grid", "L26"])]).to_csv(root/"baseline.csv", index=False)
            members = {f: dict(train_trajectory_ids=[], test_trajectory_ids=rows.trajectory_id.tolist(),
                train_readout_ids=[], test_readout_ids=[], score_languages=["x"]) for f in c.PRIMARY}
            plan = dict(sha256="fixture", profile=profile(), tasks=c.tasks(profile()), memberships=members,
                inner_scoring={f: [row_entry] for f in c.INNER},
                scoring={"grid.parquet": row_entry, "endpoint.parquet": row_entry},
                baseline_metrics=campaign.binding(root/"baseline.csv"), reserved_core_hours=2914.)
            for task in (t for t in plan["tasks"] if t["stage"] == "certify"):
                model, x, _ = fixture(task["form"])
                # Common exact null makes all predictions equal; deterministic selection is then testable.
                x[6:11] = 0.; x[15] = 0.; x[17:19] = 0.
                result = dict(certified=True, complete=True, plan_sha256="fixture",
                    task_sha256=campaign.canonical_json_sha256(task), dependencies=[],
                    selected=dict(coordinates=x.tolist(), language_coefficients=[dict(language="x", L_inf=-5., A=.3, B=.4)]),
                    bounds=[asdict(b) for b in model.bounds], equalities=list(model.equalities),
                    regularization=model.problem.regularization.as_dict(), geometry={})
                io.write_json_once(campaign.result_path(root, task), result)
            with patch.object(reporting, "load", return_value=plan), patch.object(reporting, "cpu_guard"), \
                    patch.object(diagnostics, "diagnose", return_value=True) as diagnostic:
                self.assertTrue(reporting.select(root/"plan.json"))
                self.assertTrue(reporting.report(root/"plan.json"))
                diagnostic.assert_called_once_with(root/"plan.json")
            summary = pd.read_csv(root/"report/comparison.csv")
            self.assertEqual(set(summary.model), set(FORMS)|{"published"})
            self.assertTrue(summary.mean_primary_rmse.notna().all())
            receipt = io.read_json(root/"report/report.json")
            self.assertEqual(receipt["chosen"]["form"], "RN13-psi-tied-stored")
            self.assertEqual(receipt["post_report_diagnostics"], "diagnostics/receipt.json")
            self.assertFalse(receipt["new_significance_test"])


if __name__ == "__main__":
    unittest.main()
