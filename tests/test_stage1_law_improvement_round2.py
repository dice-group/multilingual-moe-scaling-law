"""Independent round-2 equations, derivatives and contracts; no study fits/jobs."""
from dataclasses import replace
import itertools
import unittest

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as legacy
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws import contracts, diagnostics, replay
from multilingual_megatron.experiments.scaling.stage1_fit.coverage_laws.model import Model, Reduction, starts, validate
from tests.test_stage1_law_search import source_fixture


FORMS = {"Cap-burden-13": 13, "RN14-stored-data": 14, "RN14-psi-tied-zetaE": 14,
         "Cap-burden-chi0": 12, "RN14-stored-reuse": 14, "RN13-psi-tied": 13,
         "RN14-no-ridge": 14}


def fixture(name):
    p, _, x, _ = source_fixture()
    axes = np.array(list(itertools.product([1e8, 2e8, 8e8], [1., 4., 16., 32.], [1., 12., 50., 200.])))
    n, e, k = axes.T
    size = len(n)
    inventory = np.linspace(1e9, 1.3e9, size)
    ratio = np.tile([.1, .8, 1., 1.01, 2., 10.], size//6)
    p = replace(p, N=n, E=e, K=k, D=inventory*ratio, y=np.linspace(-2.8, -4., size),
                weights=np.full(size, 1/size), language_ids=("x",)*size,
                physical_floors=np.full(size, -6.), D_family=np.full(size, 2e9),
                D_script=np.full(size, 3e9), D_other=np.full(size, 4e9))
    stored = n*(1+(e-1)*(.25+.1*np.log1p(n/1e8)))
    model = Model(name, p, inventory, stored_parameters=stored)
    x = np.r_[x, .27, .7]
    x = Reduction(model).expand(Reduction(model).pack(x))
    rows = pd.DataFrame(dict(language=p.language_ids, N_act=n, N_total=stored, E=e, K=k,
                            D_t=p.D, U_t=inventory, D_family=p.D_family,
                            D_script=p.D_script, D_other=p.D_other))
    return model, x, rows


def oracle(name, x, rows):
    """Literal protocol algebra, independent of production equation helpers."""
    n, m, e, k, d, u = (rows[c].to_numpy() for c in ("N_act", "N_total", "E", "K", "D_t", "U_t"))
    n, m = n/1e8, m/1e8
    h = np.log(e)/np.log(32)
    alpha = x[0]*(1-h)+x[1]*h
    beta = x[2]*(1-h)+x[3]*h
    phi, psi = x[6]*(1-h)+x[7]*h, x[8]*(1-h)+x[9]*h
    if name.startswith("Cap-burden"):
        capacity = e**x[4]*n**(-alpha)+x[17]*np.log(k)*n**(-x[0])*e**(-x[18])
    else:
        gamma = x[10]*np.maximum(np.log(k/12), 0)/np.log(200/12)
        capacity = e**x[4]*k**phi*n**(-alpha-gamma)
    repeat_size = m if name in ("RN14-stored-data", "RN14-stored-reuse") else n
    rate = x[15]*repeat_size**x[16]
    effective = d.copy()
    mask = (d > u) & (rate > 0)
    effective[mask] = u[mask]*(1+(1-np.exp(-rate[mask]*(d[mask]/u[mask]-1)))/rate[mask])
    if name == "RN14-stored-data":
        size_factor = m**x[14]
    elif name == "RN14-psi-tied-zetaE":
        size_factor = n**(x[14]*(1-h)+x[17]*h)
    else:
        size_factor = n**x[14]
    data = e**x[5]*k**psi*size_factor*(effective/1e9)**(-beta)
    return capacity, data


def contrast_fixture():
    records = []
    for e, k in itertools.product((1, 32), (12, 200)):
        points = [2., 5., 10., 25., 40., 70.] if (e, k) == (32, 200) else [1., 4., 10., 20., 40., 80.]
        slope = {(1, 200): .3, (32, 200): .1, (1, 12): .2, (32, 12): .05}[e, k]
        for index, exposure in enumerate(points):
            capacity = 2. if (e, k) == (1, 200) else 1.
            data = slope*np.log(exposure)
            prediction = -5.+capacity+data
            records.append(dict(trajectory_id=f"D:E{e}:K{k}", readout_id=f"r{index}",
                language="x", E=e, K=k, D_t=exposure, U_t=10., floor=-5., capacity=capacity,
                data=data, prediction=prediction,
                adjusted_loss=prediction-(.2 if (e, k) == (1, 200) else 0.),
                repetition_correction=0. if exposure <= 10 else np.log(exposure/10)))
    return pd.DataFrame(records)


class LawImprovementRound2Test(unittest.TestCase):
    def test_expert_coverage_strict_boundary_and_no_extrapolation(self):
        rows = contrast_fixture()
        pre = pd.DataFrame(diagnostics.expert_coverage(rows, "pre_repeat"))
        repeated = pd.DataFrame(diagnostics.expert_coverage(rows, "repeated"))
        self.assertEqual(len(pre), 5); self.assertEqual(len(repeated), 5)
        self.assertAlmostEqual(pre.D_t.min(), 2.)
        self.assertAlmostEqual(pre.D_t.max(), 10.)
        self.assertAlmostEqual(repeated.D_t.min(), 25.)
        self.assertAlmostEqual(repeated.D_t.max(), 70.)
        np.testing.assert_allclose(pre.prediction, 1.+.05*np.log(pre.D_t), atol=1e-12)
        np.testing.assert_allclose(pre.adjusted_loss, .8+.05*np.log(pre.D_t), atol=1e-12)
        np.testing.assert_array_equal(pre.repetition_correction, 0.)
        changed = rows.copy()
        changed.loc[changed.D_t > changed.U_t, list(diagnostics.FIELDS)] += 1000.
        pd.testing.assert_frame_equal(pre, pd.DataFrame(diagnostics.expert_coverage(changed, "pre_repeat")))
        changed = rows.copy()
        changed.loc[changed.D_t <= changed.U_t, list(diagnostics.FIELDS)] += 1000.
        pd.testing.assert_frame_equal(repeated, pd.DataFrame(diagnostics.expert_coverage(changed, "repeated")))
        # Disjoint actual exposure ranges yield no invented/extrapolated points.
        changed = rows.copy()
        changed.loc[(changed.E == 32)&(changed.K == 200), "D_t"] += 1000.
        self.assertEqual(diagnostics.expert_coverage(changed, "all_support"), [])
        self.assertEqual(diagnostics.expert_coverage(rows[~((rows.E == 32)&(rows.K == 200))], "all_support"), [])

    def test_expert_coverage_rejects_ambiguous_inputs(self):
        rows = contrast_fixture()
        with self.assertRaises(ValueError):
            diagnostics.expert_coverage(pd.concat([rows, rows.iloc[[0]]]), "all_support")
        changed = rows.copy(); changed.loc[0, "U_t"] = 11.
        with self.assertRaises(ValueError):
            diagnostics.expert_coverage(changed, "all_support")
        with self.assertRaises(ValueError):
            diagnostics.expert_coverage(rows, "unknown")

    def test_size_duration_exact_two_postpass_anchors_and_components(self):
        records = []
        for ladder, exposures in (("C", [10., 20., 40., 70.]), ("D", [5., 10., 20., 40., 80.])):
            for index, exposure in enumerate(exposures):
                capacity = 2. if ladder == "C" else 1.
                prediction = (.4 if ladder == "C" else .15)*np.log(exposure)
                observed = (.5 if ladder == "C" else .2)*np.log(exposure)
                records.append(dict(trajectory_id=f"{ladder}:E32:K12", readout_id=f"r{index}",
                    E=32, K=12, language="x", D_t=exposure, U_t=10., adjusted_loss=observed,
                    prediction=prediction, floor=-4., capacity=capacity, data=prediction+4-capacity,
                    repetition_correction=0. if exposure <= 10 else .1*np.log(exposure/10)))
        rows = pd.DataFrame(records)
        result = diagnostics.size_duration(rows)
        self.assertEqual(len(result), 1)
        out = result[0]
        self.assertEqual(out["anchors"], 2)
        self.assertEqual(out["early_D_t"], 20.); self.assertEqual(out["late_D_t"], 40.)
        self.assertAlmostEqual(out["change_adjusted_loss"], .3*np.log(2), places=12)
        self.assertAlmostEqual(out["change_prediction"], .25*np.log(2), places=12)
        self.assertAlmostEqual(out["change_capacity"], 0., places=12)
        self.assertAlmostEqual(out["change_data"], out["change_prediction"], places=12)
        # One exact post-pass anchor, even with other bracketing nodes, is insufficient.
        changed = rows.copy()
        changed.loc[(changed.trajectory_id.str.startswith("D:"))&(changed.D_t == 40.), "D_t"] = 41.
        self.assertEqual(diagnostics.size_duration(changed), [])
        # Inventory-boundary nodes are excluded, not counted as post-pass anchors.
        changed = rows.copy(); changed["U_t"] = 20.
        self.assertEqual(diagnostics.size_duration(changed), [])
        with self.assertRaises(ValueError):
            diagnostics.size_duration(pd.concat([rows, rows[rows.D_t == 20].iloc[[0]]]))
        for mask in (rows.trajectory_id.str.startswith("D:"),
                     rows.trajectory_id.str.startswith("D:")&(rows.D_t == 40.)):
            changed = rows.copy(); changed.loc[mask, "U_t"] = 11.
            with self.assertRaises(ValueError):
                diagnostics.size_duration(changed)

    def test_round2_task_graph_budget_and_preserved_default_protocol(self):
        profile = dict(protocol="law_improvement_round2_v1", concurrency=256,
                       start_minutes=240, certify_minutes=240, analysis_minutes=60,
                       max_reserved_core_hours=3000, account="fixture", partition="cpu",
                       memory="4G", workspace_root="/tmp/round2 fixture", python_executable="/env/bin/python")
        rules = contracts.rules_for(profile)
        self.assertEqual(contracts.ROUND2_FORMS, FORMS)
        contracts.validate_profile(profile)
        tasks = contracts.tasks(profile)
        self.assertEqual(len(contracts.cells(rules)), 56)
        self.assertEqual(len(tasks), 728)
        self.assertEqual(sum(t["stage"] == "starts" for t in tasks), 672)
        self.assertEqual(sum(t["stage"] == "certify" for t in tasks), 56)
        self.assertEqual(contracts.reservation(profile), 2914.)
        for cell in contracts.cells(rules):
            starts_for_cell = [i for t in tasks if t["stage"] == "starts" and t["cell"] == cell["id"]
                               for i in t["start_indices"]]
            self.assertEqual(starts_for_cell, list(range(48)))
        legacy_profile = {key: val for key, val in profile.items() if key != "protocol"}
        self.assertEqual(len(contracts.cells()), 48)
        self.assertEqual(len(contracts.tasks(legacy_profile)), 624)
        self.assertEqual(contracts.reservation(legacy_profile), 2498.)
        with self.assertRaises(ValueError):
            contracts.rules_for(profile | {"protocol": "unknown"})

    def test_selector_reference_and_nulls_are_frozen(self):
        rules = contracts.rules_for({"protocol": "law_improvement_round2_v1"})
        rows = [dict(form=name, fold=fold, certified=True, complete=True, rmse=1.)
                for name in FORMS for fold in contracts.INNER]
        decision = contracts.choose(rows, rules)
        self.assertEqual(decision["chosen"]["form"], "Cap-burden-13")
        for row in rows:
            if row["form"] in ("Cap-burden-chi0", "RN14-stored-reuse", "RN13-psi-tied"):
                row["rmse"] = 1e-12
            if row["fold"] == "inner_terminal":
                row["rmse"] = 1000. if row["form"] == "Cap-burden-13" else 1e-12
        self.assertEqual(contracts.choose(rows, rules), decision)
        for row in rows:
            if row["form"] in ("Cap-burden-13", "RN14-stored-data", "RN14-psi-tied-zetaE"):
                row["rmse"] = 1.03
        self.assertEqual(contracts.choose(rows, rules)["chosen"]["form"], "RN14-no-ridge")
        with self.assertRaises(ValueError):
            contracts.choose(rows+[rows[0]], rules)

    def test_nominal_counts_positive_basis_and_independent_equations(self):
        for name, count in FORMS.items():
            with self.subTest(name=name):
                model, x, rows = fixture(name)
                validate(model, {"coordinates": x})
                self.assertEqual(len(Reduction(model).groups), count)
                cap, data, *_ = model.terms(x)
                for actual, expected in zip((cap, data), oracle(name, x, rows)):
                    self.assertTrue(np.all(actual > 0))
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
                record = dict(coordinates=x, language_coefficients=[dict(language="x", L_inf=-5., A=.3, B=.4)])
                expected = -5.+.3*cap+.4*data
                np.testing.assert_allclose(legacy.predict(model, record), expected, atol=1e-12, rtol=0)
                np.testing.assert_allclose(replay.predict(name, record, rows), expected, atol=1e-12, rtol=0)
                value, derivative = model.penalty(x)
                self.assertEqual(value, 0.)
                np.testing.assert_array_equal(derivative, np.zeros(19))

    def test_all_free_term_derivatives_and_reduced_prediction_jacobian(self):
        for name in FORMS:
            with self.subTest(name=name):
                model, x, _ = fixture(name)
                cap, data, cj, dj = model.terms(x)
                reduction = Reduction(model)
                matrix = reduction.matrix
                packed = reduction.pack(x)
                for j, group in enumerate(reduction.groups):
                    direction = np.eye(len(packed))[j]*1e-6
                    plus = model.terms(reduction.expand(packed+direction))
                    minus = model.terms(reduction.expand(packed-direction))
                    for index, jac in ((0, cj), (1, dj)):
                        numerical = (plus[index]-minus[index])/2e-6
                        np.testing.assert_allclose(numerical, jac[:, group].sum(axis=1), atol=2e-8, rtol=5e-6)
                    numerical = (.3*(plus[0]-minus[0])+.4*(plus[1]-minus[1]))/2e-6
                    np.testing.assert_allclose(numerical, ((.3*cj+.4*dj)@matrix)[:, j], atol=2e-8, rtol=5e-6)

    def test_burden_anchors_null_and_exact_contrast(self):
        model, x, rows = fixture("Cap-burden-13")
        null, _, _ = fixture("Cap-burden-chi0")
        x[18] = 0.
        for left, right in zip(model.terms(x)[:2], null.terms(x)[:2]):
            np.testing.assert_array_equal(left, right)
        x[17], x[18] = .23, 1.1
        cap = model.terms(x)[0]
        before = x.copy(); before[17] = 0.
        np.testing.assert_array_equal(cap[rows.K == 1], model.terms(before)[0][rows.K == 1])
        np.testing.assert_array_equal(model.terms(before)[2][:, 18], 0.)
        frame = rows.assign(capacity=cap)
        for n, f in frame.groupby("N_act"):
            f = f.set_index(["E", "K"])
            contrast = f.loc[(1., 200.), "capacity"]-f.loc[(32., 200.), "capacity"]-f.loc[(1., 12.), "capacity"]+f.loc[(32., 12.), "capacity"]
            expected = x[17]*np.log(200/12)*(n/1e8)**(-x[0])*(1-32**(-x[18]))
            self.assertAlmostEqual(contrast, expected, places=12)

    def test_stored_size_equal_active_nests_reference(self):
        reference, x, rows = fixture("RN14-no-ridge")
        for name in ("RN14-stored-data", "RN14-stored-reuse"):
            model = Model(name, reference.problem, reference.inventory, stored_parameters=reference.problem.N)
            for a, b in zip(model.terms(x)[:2], reference.terms(x)[:2]):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            record = dict(coordinates=x, language_coefficients=[dict(language="x", L_inf=-5., A=.3, B=.4)])
            np.testing.assert_allclose(replay.predict(name, record, rows.assign(N_total=rows.N_act)),
                                       replay.predict("RN14-no-ridge", record, rows), atol=1e-12, rtol=0)

    def test_stored_data_zeta_zero_nests_stored_reuse(self):
        data, x, _ = fixture("RN14-stored-data")
        reuse, _, _ = fixture("RN14-stored-reuse")
        x[14] = 0.
        for a, b in zip(data.terms(x)[:2], reuse.terms(x)[:2]):
            np.testing.assert_array_equal(a, b)

    def test_stored_coordinate_is_not_expert_times_active_or_constant_shift(self):
        actual, x, rows = fixture("RN14-stored-data")
        bogus = Model("RN14-stored-data", actual.problem, actual.inventory,
                      stored_parameters=actual.problem.N*actual.problem.E)
        self.assertGreater(np.max(abs(actual.terms(x)[1]-bogus.terms(x)[1])), .01)
        reference, _, _ = fixture("RN14-no-ridge")
        x[15] = 0.  # Separate multiplier identifiability from repetition.
        ratio = actual.terms(x)[1]/reference.terms(x)[1]
        ratio_by_e = pd.DataFrame(dict(E=rows.E, ratio=ratio)).groupby("E").ratio
        self.assertGreater(ratio_by_e.max().loc[32]-ratio_by_e.min().loc[32], .001)
        np.testing.assert_array_equal(ratio[rows.E == 1], 1.)

    def test_stored_metadata_is_required_aligned_finite_and_physical(self):
        reference, _, _ = fixture("RN14-no-ridge")
        for bad in (None, reference.problem.N[:-1], reference.problem.N*.5,
                    reference.problem.N*np.nan, reference.problem.N*np.inf):
            with self.assertRaises(ValueError):
                Model("RN14-stored-data", reference.problem, reference.inventory, stored_parameters=bad)

    def test_psi_tied_zeta_endpoints_and_exact_constant_zeta_null(self):
        model, x, rows = fixture("RN14-psi-tied-zetaE")
        null, _, _ = fixture("RN13-psi-tied")
        x[17] = x[14]
        for a, b in zip(model.terms(x)[:2], null.terms(x)[:2]):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        x[17] = .33
        _, term, _, jac = model.terms(x)
        logn = np.log(rows.N_act/1e8)
        np.testing.assert_array_equal(jac[rows.E == 1, 17], 0.)
        np.testing.assert_array_equal(jac[rows.E == 32, 14], 0.)
        np.testing.assert_allclose(jac[rows.E == 1, 14], term[rows.E == 1]*logn[rows.E == 1], atol=1e-12)
        np.testing.assert_allclose(jac[rows.E == 32, 17], term[rows.E == 32]*logn[rows.E == 32], atol=1e-12)

    def test_exposure_boundary_and_zero_rate(self):
        for name in FORMS:
            model, x, rows = fixture(name)
            _, _, _, derivative = model.terms(x)
            before_repeat = rows.D_t <= rows.U_t
            np.testing.assert_array_equal(derivative[before_repeat, 15:17], 0.)
            null = x.copy(); null[15] = 0.
            plus = null.copy(); plus[15] = 1e-7
            current = model.terms(null)
            numerical = (model.terms(plus)[1]-current[1])/1e-7
            np.testing.assert_allclose(numerical, current[3][:, 15], atol=2e-6, rtol=1e-5)
            for actual, expected in zip(current[:2], oracle(name, null, rows)):
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(current[3][:, 16], 0.)

    def test_predetermined_starts_are_valid_reproducible_and_nonmutating(self):
        _, _, anchor, _ = source_fixture()
        anchor = anchor[:15]
        before = anchor.copy()
        for name in FORMS:
            model, _, _ = fixture(name)
            left, right = starts(model, anchor), starts(model, anchor)
            self.assertEqual(left.shape, (48, 19))
            np.testing.assert_array_equal(left, right)
            for vector in left:
                validate(model, {"coordinates": vector})
        np.testing.assert_array_equal(before, anchor)

    def test_synthetic_burden_variable_projection_and_direct_audit(self):
        """Optimize only a synthetic two-coordinate example, never study rows."""
        model, x, _ = fixture("Cap-burden-13")
        cap, data, *_ = model.terms(x)
        model.problem = replace(model.problem, y=-5.+.3*cap+.4*data)
        model.bounds = [replace(bound, lower=float(value), upper=float(value))
                        for bound, value in zip(model.bounds, x)]
        model.bounds[17] = replace(model.bounds[17], lower=.01, upper=.9)
        model.bounds[18] = replace(model.bounds[18], lower=.01, upper=1.5)
        initial = x.copy(); initial[17:] = [.31, .5]
        profile = legacy.Profile(model, .15)
        fitted = profile.run(initial)
        self.assertTrue(fitted["success"], fitted)
        audited = legacy.Direct(profile, fitted).run()
        self.assertTrue(audited["success"], audited)
        np.testing.assert_allclose(legacy.predict(model, fitted), model.problem.y, atol=1e-5, rtol=0)
        np.testing.assert_allclose(legacy.predict(model, fitted), legacy.predict(model, audited), atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
