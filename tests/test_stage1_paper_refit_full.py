"""Scientific domain, derivative, inventory and sealing tests for the full refit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.law_search import model as legacy
from multilingual_megatron.experiments.scaling.stage1_fit.repetition_model import Bound
from multilingual_megatron.experiments.scaling.stage1_fit.nd_coupling.model import Reduction
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs, numerics, protocol, runner
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import full_report, seal
from multilingual_megatron.experiments.scaling.stage1_fit import execution as io
from tests.test_stage1_paper_refit import fixture
from tests.test_stage1_paper_refit_execution import runtime_profile


class FullRefitTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        profile, _, _ = fixture(self.root)
        inputs.prepare(profile)
        self.rows, self.sources, self.packets, self.cal = inputs.read_training(self.root / "bundle/manifest.json", "all_grid")

    def model(self, name="RN14-stored-reuse", arm="lbfgsb_box"):
        return numerics.make_model(name, self.rows, self.sources, self.packets, self.cal, arm)

    def test_default_domain_is_preserved_but_relaxed_model_accepts_negative_exponents(self):
        model = self.model()
        x = numerics.starts(model, self.cal)[0]
        x[0:2] = .1; x[10] = -.5
        # Explicitly exercise K200, beyond this small fixture's K4.
        from dataclasses import replace
        model.problem = replace(model.problem, K=np.full(len(self.rows), 200.))
        cap, _, jac, _ = model.terms(x)
        self.assertTrue(np.isfinite(cap).all())
        numeric = approx_derivative(lambda z: model.terms(np.r_[z, x[2:]])[0], x[:2])
        np.testing.assert_allclose(jac[:, :2], numeric, atol=2e-6, rtol=2e-5)
        model.require_positive_exponents = True
        with self.assertRaisesRegex(FloatingPointError, "non-positive"):
            model.terms(x)

    def test_stored_coverage_control_is_exact_nested_restriction(self):
        parent = self.model()
        control = self.model("RN12-stored-common-K")
        self.assertTrue(control.stored_form)
        self.assertEqual(control.equalities, ((6, 7), (8, 9)))
        self.assertEqual(len(Reduction(parent).bounds)-len(Reduction(control).bounds), 2)
        expected = numerics.starts(parent, self.cal).copy()
        for a, b in control.equalities:
            expected[:, a] = expected[:, b] = (expected[:, a]+expected[:, b])/2
        np.testing.assert_array_equal(numerics.starts(control, self.cal), expected)
        for x in expected[:3]:
            for a, b in zip(parent.terms(x), control.terms(x)):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)

    def test_all_models_replay_and_analytic_derivatives(self):
        for name in (*protocol.SHARED, *protocol.ADDITIONAL_ABLATIONS):
            with self.subTest(model=name):
                model = self.model(name)
                x = numerics.starts(model, self.cal)[5]
                cap, term, cj, dj = model.terms(x)
                red = Reduction(model)
                numerical = approx_derivative(lambda z: np.r_[model.terms(red.expand(z))[0], model.terms(red.expand(z))[1]], red.pack(x), bounds=np.asarray(red.bounds).T)
                np.testing.assert_allclose(np.vstack((cj, dj)) @ red.matrix, numerical, atol=5e-6, rtol=2e-4)
                record = dict(coordinates=x.tolist(), language_coefficients=[
                    dict(language=l, L_inf=-5., A=.3, B=.4) for l in sorted(self.rows.language.unique())])
                np.testing.assert_allclose(numerics.predict(model, record),
                    numerics.independent(model, record, self.rows, self.packets), atol=1e-10, rtol=0)

    def test_identical_starts_all_arms_and_relaxed_domain_is_explored(self):
        models = [self.model(arm=a) for a in protocol.POLICIES]
        seeds = [numerics.starts(m, self.cal) for m in models]
        for s in seeds[1:]:
            np.testing.assert_array_equal(s, seeds[0])
        # This fixture has no K>12 and fixes gamma to zero; a real supported
        # gamma domain is necessary for the full-box seeds to cross zero.
        model = models[0]
        model.bounds[10] = Bound("gamma_n", -1., 1., "signed")
        s = numerics.starts(model, self.cal)
        self.assertTrue(np.any(np.minimum(s[:, 0], s[:, 1])+s[:, 10] < 0))

    def test_box_optimizer_finds_negative_optimum_and_sqp_retains_constraint(self):
        bounds = [Bound("alpha_e1", .01, 2., "positive"), Bound("gamma_n", -1., 1., "signed")]
        def objective(x):
            residual = x-np.array([.2, -.8])
            return 1e4*float(residual@residual), 2e4*residual
        for arm in protocol.POLICIES:
            model = SimpleNamespace(equalities=(), solver_policy=arm,
                constraint_matrix=lambda size, arm=arm: np.array([[1., 1.]]) if arm == "slsqp_positive" else None)
            result = numerics.optimize(objective, [.5, 0.], bounds, model)
            self.assertTrue(result.success)
            if arm == "slsqp_positive":
                self.assertGreaterEqual(result.x.sum(), 1e-6-1e-10)
            else:
                np.testing.assert_allclose(result.x, [.2, -.8], atol=1e-7)

    def test_ties_are_eliminated_and_not_ignored(self):
        bounds = [Bound("x", -1., 1., "signed")]*2
        model = SimpleNamespace(equalities=((0, 1),), solver_policy="lbfgsb_box", constraint_matrix=lambda n: None)
        def objective(x):
            r = x-np.array([.2, .8]); return float(r@r), 2*r
        result = numerics.optimize(objective, [0., 0.], bounds, model)
        np.testing.assert_allclose(result.x, [.5, .5], atol=1e-7)

    def test_audit_recovery_cannot_loosen_stopping_tolerance(self):
        bounds = [Bound("x", -1., 1., "signed")]
        model = SimpleNamespace(equalities=(), solver_policy="slsqp_box", audit_ftol=1e-6,
                                constraint_matrix=lambda n: None)
        with self.assertRaisesRegex(ValueError, "only tighten"):
            numerics.optimize(lambda x: (float(x@x), 2*x), [.5], bounds, model)
        model.audit_ftol = 1e-14
        result = numerics.optimize(lambda x: (float(x@x), 2*x), [.5], bounds, model)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, 0., atol=1e-10)

    def test_powered_joint_gradient_and_physical_coefficients(self):
        model = self.model("JointMoE-Skaling-pooled-active-Dt")
        x = numerics.starts(model, self.cal)[0]; x[17] = .8
        record = dict(coordinates=x.tolist(), language_coefficients=[
            dict(language=l, L_inf=-4., A=.3, B=.4) for l in sorted(self.rows.language.unique())])
        direct = numerics.PoweredDirect(legacy.Profile(model, self.cal["delta"]), record)
        value, grad, pred, _ = direct.evaluate(np.asarray(direct.initial))
        red = Reduction(SimpleNamespace(bounds=direct.bounds, equalities=model.equalities))
        numeric = approx_derivative(lambda z: direct.evaluate(red.expand(z))[0], red.pack(direct.initial), bounds=np.asarray(red.bounds).T).ravel()
        np.testing.assert_allclose(grad @ red.matrix, numeric, atol=2e-6, rtol=2e-4)
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(pred, numerics.independent(model, record, self.rows, self.packets), atol=1e-12)

    def test_allocation_replays_refit_regularization_serialization(self):
        from dataclasses import asdict
        from result_analysis.compute_allocation.doubling import refit_predictions
        model = self.model(); x = numerics.starts(model, self.cal)[0]
        record = dict(coordinates=x.tolist(), language_coefficients=[
            dict(language=l, L_inf=-4., A=.3, B=.4) for l in sorted(self.rows.language.unique())])
        result = dict(selected=record, regularization=asdict(model.problem.regularization))
        actual = refit_predictions(result, model.name, self.rows).prediction
        np.testing.assert_allclose(actual, numerics.independent(model, record, self.rows, self.packets), atol=1e-10)

    def test_complete_inventory_and_paired_control_tasks(self):
        profile = runtime_profile() | dict(full_paper_policy="lbfgsb_box", paired_controls=True)
        tasks = runner.tasks_for(protocol.MODELS, {f: ["x", "y"] for f in protocol.FOLDS}, {}, profile)
        cells = [t for t in tasks if t["stage"] == "certify"]
        self.assertEqual(len(protocol.cells()), 144)
        self.assertEqual(len(cells), 160)
        self.assertEqual(len(set(t["cell"] for t in cells)), 160)
        for cell in cells:
            source = [tasks[i] for i in cell["dependencies"]]
            if cell["model"] not in protocol.NATIVE:
                self.assertEqual([i for t in source for i in t["start_indices"]], list(range(48)))

    def test_report_preserves_missing_cells_and_exports_all_grid_predictions(self):
        model = self.model()
        x = numerics.starts(model, self.cal)[0]
        selected = dict(coordinates=x.tolist(), language_coefficients=[
            dict(language=l, L_inf=-4., A=.3, B=.4) for l in sorted(self.rows.language.unique())])
        tasks = [dict(id=i, stage="certify", model=model.name, fold=f, arm="lbfgsb_box")
                 for i, f in enumerate(inputs.FOLDS)]
        plan = dict(sha256="fixture", tasks=tasks, models=[model.name], reference_predictions=None,
                    input=inputs.local_binding(self.root / "bundle/manifest.json", self.root))
        (self.root / "results").mkdir()
        io.write_json_once(self.root / "results/00003.json", {})
        receipt = dict(certified=True, complete=True, selected=selected,
                       effective_exponents={}, positive_effective_exponents=True)
        with patch.object(runner, "load", return_value=plan), patch.object(runner, "receipt", return_value=receipt):
            result = full_report.report(self.root / "plan.json")
        self.assertEqual(result["certified_fit_cells"], 1)
        self.assertFalse(result["all_cells_certified"])
        self.assertFalse(result["paper_ready"])
        self.assertTrue((self.root / "report/predictions.parquet").exists())
        self.assertTrue(result["pending_paper_adoption_gates"])

    def test_seal_rejects_incomplete_and_requires_bound_downstream_review(self):
        output = self.root / "report"; output.mkdir()
        plan = dict(sha256="plan", models=list(protocol.MODELS),
                    profile=dict(full_paper_policy="lbfgsb_box"), tasks=[])
        summary = inputs.sealed(dict(plan_sha256="plan", all_cells_certified=False, all_scoring_complete=False))
        io.write_json_once(output / "report.json", summary)
        with patch.object(runner, "load", return_value=plan):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                seal.seal(self.root / "plan.json", self.root / "missing_review.json")
        diagnostics = [dict(model="RN14-stored-reuse", arm="lbfgsb_box", positive_effective_exponents=True)
                       for _ in protocol.folds("RN14-stored-reuse")]
        io.write_json_once(output / "fit_diagnostics.json", dict(fits=diagnostics))
        evidence = inputs.local_binding(output / "fit_diagnostics.json", self.root)
        summary = inputs.sealed(dict(plan_sha256="plan", all_cells_certified=True,
                                    all_scoring_complete=True, outputs=[evidence]))
        (output / "report.json").unlink()
        io.write_json_once(output / "report.json", summary)
        review = dict(report_sha256=summary["sha256"], story_unchanged=True, gates={})
        review_path = self.root / "review.json"; io.write_json_once(review_path, review)
        with patch.object(runner, "load", return_value=plan):
            with self.assertRaisesRegex(ValueError, "all downstream"):
                seal.seal(self.root / "plan.json", review_path)
            review["gates"] = {g: dict(passed=True, rationale="fixture", evidence=[evidence]) for g in seal.GATES}
            review_path.unlink()
            io.write_json_once(review_path, review)
            result = seal.seal(self.root / "plan.json", review_path)
            self.assertEqual(result["report_sha256"], summary["sha256"])
            self.assertFalse(result["manuscript_automatically_changed"])


if __name__ == "__main__":
    unittest.main()
