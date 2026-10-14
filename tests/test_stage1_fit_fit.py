from __future__ import annotations
import math
import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_exposure import exposure_components
from multilingual_megatron.experiments.scaling.stage1_fit.fit import (
    ATLAS_TRANSFER_TAU_SCALE, EXPLORATORY_EXPOSURE_MODELS, PRIMARY_MODELS,
    SharedParameters, effective_exposure_data, outer_parameter_names)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from multilingual_megatron.experiments.scaling.stage1_fit.fit import (data_column, endpoints_from_line,
    expert_log_coordinate, line_from_endpoints, outer_from_shared, predict, predictor_terms,
    predictor_terms_and_jacobian, shared_from_outer, transformed_experts)
from multilingual_megatron.experiments.scaling.stage1_fit.fit import adaptive_huber_scale, huber_loss, huber_score, trajectory_weights, weighted_mad, weighted_mad_scale, weighted_median, weighted_objective_and_gradient
from multilingual_megatron.experiments.scaling.stage1_fit.fit import DIRECT_PENALTY_OBJECTIVE, BoundaryContact, LanguageProblem, OuterBound, ProjectedOptimizationError, ProjectedProblem, ProjectedStartResult, _direct_objective_gradient, best_basin, boundary_contacts, d16_adaptive_starts, d16_adaptive_tiers, deterministic_local_perturbations, direct_joint_check, equivalence_check, expand_contacted_bounds, floor_is_saturated, initial_bounds, iter_starts, nested_warm_start, run_projected_start, solve_language, solve_language_reference, start_at, start_count, weighted_objective

class Stage1FitModelsObjectiveTest(unittest.TestCase):

    def test_hand_calculated_m0_m1_m2_predictions(self) -> None:
        N = np.asarray([400000000.0])
        D = np.asarray([2000000000.0])
        E = np.asarray([4.0])
        K = np.asarray([16.0])
        floor = np.asarray([-4.0])
        A = np.asarray([0.7])
        B = np.asarray([0.3])
        cases = {'M0': np.asarray([0.3, 0.2, 0.1, -0.05]), 'M1': np.asarray([0.3, 0.4, 0.2, 0.16, -0.1, 0.06, 0.1, -0.05]), 'M2': np.asarray([0.3, 0.4, 0.2, 0.16, -0.1, 0.06, 0.1, -0.02, -0.05, 0.03])}
        for model, outer in cases.items():
            with self.subTest(model=model):
                p = shared_from_outer(model, outer)
                log_e = math.log(4.0)
                alpha = p.alpha_0 + p.alpha_1 * log_e
                beta = p.beta_0 + p.beta_1 * log_e
                phi = p.phi_0 + p.phi_1 * log_e
                psi = p.psi_0 + p.psi_1 * log_e
                expected_capacity = math.exp(p.delta * log_e + phi * math.log(16.0) - alpha * math.log(4.0))
                expected_data = math.exp(p.omega * log_e + psi * math.log(16.0) - beta * math.log(2.0))
                expected = -4.0 + 0.7 * expected_capacity + 0.3 * expected_data
                actual = predict(model, p, N, D, E, K, floor, A, B)
                self.assertEqual(actual.dtype, np.dtype(np.float64))
                self.assertAlmostEqual(float(actual[0]), expected, places=13)

    def test_literature_forms_are_nested_and_published_transform_is_fixed(self) -> None:
        predictors = (np.asarray([1.0e8, 4.0e8]), np.asarray([1.0e9, 3.0e9]),
            np.asarray([1.0, 32.0]), np.asarray([12.0, 200.0]))
        c0 = shared_from_outer('C0', np.asarray([.3, .2]))
        m0 = shared_from_outer('M0', np.asarray([.3, .2, 0.0, 0.0]))
        for first, second in zip(predictor_terms('C0', c0, *predictors),
                predictor_terms('M0', m0, *predictors)):
            np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-15)
        joint = np.asarray([.3, .4, .2, .16, -.1, .06])
        j0 = shared_from_outer('J0', joint)
        m1 = shared_from_outer('M1', np.r_[joint, 0.0, 0.0])
        for first, second in zip(predictor_terms('J0', j0, *predictors),
                predictor_terms('M1', m1, *predictors)):
            np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-15)
        transformed = transformed_experts(np.asarray([1.0, 4.0, 16.0, 32.0]))
        self.assertAlmostEqual(float(transformed[0]), 2.0732, places=12)
        self.assertTrue(np.all(np.diff(transformed) > 0.0))
        self.assertTrue(np.all(transformed < 290.4521))
        np.testing.assert_allclose(expert_log_coordinate('JH', [1.0, 32.0]),
            [0.0, math.log(32.0)], rtol=0.0, atol=5e-15)
        self.assertEqual(data_column('JH'), 'D_total')
        self.assertEqual(data_column('J0'), 'D_t')

    def test_endpoint_round_trip_and_outer_round_trip(self) -> None:
        intercept, slope = line_from_endpoints(0.11, 0.47)
        first, last = endpoints_from_line(intercept, slope)
        self.assertAlmostEqual(first, 0.11)
        self.assertAlmostEqual(last, 0.47)
        original = np.asarray([0.05, 0.5, 0.25, 0.05, -0.5, 0.5, -0.5, 0.0, 0.5, -0.5])
        restored = outer_from_shared('M2', shared_from_outer('M2', original))
        np.testing.assert_allclose(restored, original, rtol=0.0, atol=1e-15)

    def test_predictor_term_outer_jacobian_matches_finite_difference(self) -> None:
        outer = np.asarray([0.24, 0.34, 0.19, 0.23, -0.07, 0.04, 0.06, -0.03, -0.02, 0.05])
        predictors = (np.asarray([100000000.0, 300000000.0, 600000000.0], dtype=np.float32), np.asarray([200000000.0, 1000000000.0, 5000000000.0], dtype=np.float32), np.asarray([1, 4, 32], dtype=np.float32), np.asarray([1, 12, 200], dtype=np.float32))
        capacity, data, capacity_jac, data_jac = predictor_terms_and_jacobian('M2', outer, *predictors)
        self.assertEqual(capacity.dtype, np.dtype(np.float64))
        self.assertEqual(capacity_jac.dtype, np.dtype(np.float64))
        epsilon = 1e-06
        for column in range(outer.size):
            plus = outer.copy()
            minus = outer.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            cap_plus, data_plus = predictor_terms('M2', shared_from_outer('M2', plus), *predictors)
            cap_minus, data_minus = predictor_terms('M2', shared_from_outer('M2', minus), *predictors)
            np.testing.assert_allclose(capacity_jac[:, column], (cap_plus - cap_minus) / (2.0 * epsilon), rtol=2e-08, atol=2e-10)
            np.testing.assert_allclose(data_jac[:, column], (data_plus - data_minus) / (2.0 * epsilon), rtol=2e-08, atol=2e-10)

    def test_literature_form_jacobians_match_finite_difference(self) -> None:
        predictors = (np.asarray([1.0e8, 3.0e8, 6.0e8]), np.asarray([2.0e8, 1.0e9, 5.0e9]),
            np.asarray([1.0, 4.0, 32.0]), np.asarray([1.0, 12.0, 200.0]))
        cases = {'C0': np.asarray([.24, .19]),
            'J0': np.asarray([.24, .34, .19, .23, -.07, .04]),
            'JH': np.asarray([.24, .34, .19, .23, -.07, .04])}
        epsilon = 1e-6
        for model, outer in cases.items():
            _, _, capacity_jac, data_jac = predictor_terms_and_jacobian(model, outer, *predictors)
            for column in range(outer.size):
                plus, minus = outer.copy(), outer.copy(); plus[column] += epsilon; minus[column] -= epsilon
                cap_plus, data_plus = predictor_terms(model, shared_from_outer(model, plus), *predictors)
                cap_minus, data_minus = predictor_terms(model, shared_from_outer(model, minus), *predictors)
                np.testing.assert_allclose(capacity_jac[:, column], (cap_plus-cap_minus)/(2*epsilon),
                    rtol=3e-8, atol=3e-10)
                np.testing.assert_allclose(data_jac[:, column], (data_plus-data_minus)/(2*epsilon),
                    rtol=3e-8, atol=3e-10)

    def test_exact_trajectory_weights_have_equal_totals(self) -> None:
        identifiers = ['t0', 't0', 't1', 't1', 't1', 't1', 't2']
        weights = trajectory_weights(identifiers)
        self.assertEqual(weights.dtype, np.dtype(np.float64))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        totals = {name: float(weights[np.asarray(identifiers) == name].sum()) for name in set(identifiers)}
        for total in totals.values():
            self.assertAlmostEqual(total, 1.0 / 3.0)

    def test_weighted_median_mad_and_zero_scale_fallback(self) -> None:
        self.assertEqual(weighted_median([10.0, 0.0], [0.5, 0.5]), 0.0)
        values = np.asarray([0.0, 1.0, 2.0, 100.0])
        weights = np.asarray([0.2, 0.4, 0.3, 0.1])
        self.assertEqual(weighted_median(values, weights), 1.0)
        self.assertEqual(weighted_mad(values, weights), 1.0)
        self.assertAlmostEqual(weighted_mad_scale(values, weights), 1.4826)
        scale = adaptive_huber_scale([2.0, 2.0, 2.0], [0.2, 0.3, 0.5])
        self.assertTrue(scale.use_ols)
        self.assertIsNone(scale.delta)
        self.assertEqual(scale.scale, 0.0)

    def test_huber_values_scores_and_analytic_gradient(self) -> None:
        residual = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected_loss = np.asarray([2.5, 0.5, 0.0, 0.5, 2.5])
        expected_score = np.asarray([-1.0, -1.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(huber_loss(residual, 1.0), expected_loss)
        np.testing.assert_allclose(huber_score(residual, 1.0), expected_score)
        parameters = np.asarray([0.3, -0.4])
        design = np.asarray([[1.0, 0.2], [1.0, -0.7], [1.0, 1.5], [1.0, 2.0]])
        y = np.asarray([0.1, -0.2, 0.9, 3.0])
        weights = np.asarray([0.1, 0.2, 0.3, 0.4])

        def evaluate(values: np.ndarray) -> tuple[float, np.ndarray]:
            return weighted_objective_and_gradient(design @ values - y, design, weights, 0.6)
        objective, gradient = evaluate(parameters)
        self.assertTrue(math.isfinite(objective))
        epsilon = 1e-07
        numerical = np.empty(2)
        for index in range(2):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (evaluate(plus)[0] - evaluate(minus)[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-08, atol=2e-09)

    def test_numeric_zero_is_not_a_huber_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            huber_loss([1.0], 0.0)
TRUE_OUTER = {'C0': np.asarray([0.3, 0.22]),
    'J0': np.asarray([0.28, 0.36, 0.22, 0.19, -0.1, 0.04]),
    'JH': np.asarray([0.28, 0.36, 0.22, 0.19, -0.1, 0.04]),
    'M0': np.asarray([0.3, 0.22, 0.06, -0.04]),
    'M1': np.asarray([0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.06, -0.04]),
    'M2': np.asarray([0.28, 0.36, 0.22, 0.19, -0.1, 0.04, 0.08, -0.02, -0.04, 0.03])}

def _synthetic_problem(model: str, *, noise: float=0.0) -> ProjectedProblem:
    rows = list(itertools.product((100000000.0, 250000000.0, 600000000.0), (200000000.0, 1000000000.0, 5000000000.0), (1.0, 4.0, 32.0), (1.0, 12.0, 200.0)))
    base = np.asarray(rows, dtype=np.float64)
    N = np.tile(base[:, 0], 2)
    D = np.tile(base[:, 1], 2)
    E = np.tile(base[:, 2], 2)
    K = np.tile(base[:, 3], 2)
    languages = tuple(['deu_Latn'] * len(rows) + ['eng_Latn'] * len(rows))
    shared = shared_from_outer(model, TRUE_OUTER[model])
    capacity, data = predictor_terms(model, shared, N, D, E, K)
    floor = np.asarray([-5.0] * len(rows) + [-5.7] * len(rows))
    coefficient_a = np.asarray([0.65] * len(rows) + [0.48] * len(rows))
    coefficient_b = np.asarray([0.31] * len(rows) + [0.42] * len(rows))
    y = floor + coefficient_a * capacity + coefficient_b * data
    if noise:
        y = y + noise * np.sin(np.arange(y.size, dtype=np.float64) * 0.71)
    trajectory_ids = [f'n{N[index]}:e{E[index]}:k{K[index]}' for index in range(y.size)]
    return ProjectedProblem.from_arrays(y=y, weights=trajectory_weights(trajectory_ids), N=N, D=D, E=E, K=K, language_ids=languages, physical_floors=np.full(y.size, -8.0))

def _result(identifier: str, objective: float | None, prediction: np.ndarray | None, *, success: bool=True) -> ProjectedStartResult:
    return ProjectedStartResult(model='M0', start_id=identifier, start_index=None, success=success, status=0 if success else 1, message='test', objective=objective, coordinates=np.asarray([0.2, 0.2, 0.0, 0.0]), shared_parameters=None, prediction=prediction, profile=None, iterations=1, function_evaluations=1, boundary_contacts=())

class Stage1FitOptimizerTest(unittest.TestCase):

    def test_exact_start_counts_and_lexicographic_indexing(self) -> None:
        self.assertEqual(start_count('C0'), 9)
        self.assertEqual(start_count('J0'), 729)
        self.assertEqual(start_count('JH'), 729)
        self.assertEqual(start_count('M0'), 81)
        self.assertEqual(start_count('M1'), 6561)
        self.assertEqual(start_count('M2'), 59049)
        np.testing.assert_array_equal(start_at('M0', 0).values, [0.05, 0.05, -0.5, -0.5])
        np.testing.assert_array_equal(start_at('M0', 1).values, [0.05, 0.05, -0.5, 0.0])
        np.testing.assert_array_equal(start_at('M0', 80).values, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual([item.start_index for item in iter_starts('M2', 37, 40)], [37, 38, 39])
        with self.assertRaises(IndexError):
            start_at('M2', 59049)
        self.assertEqual(start_at('M0', 1).start_id, 'grid:000001')
        warm = nested_warm_start('M1', shared_from_outer('M0', TRUE_OUTER['M0']))
        self.assertEqual(warm.start_id, 'warm:M0')
        self.assertEqual(warm.start_index, 6561)

    def test_d16_adaptive_starts_are_deterministic_nested_and_end_in_full_grid(self) -> None:
        center = np.asarray([0.3, 0.4, 0.2, 0.16, -0.1, 0.06, 0.1, -0.02, -0.05, 0.03])
        bounds = initial_bounds('M2')
        starts32 = d16_adaptive_starts('M2', center, (), bounds, target=32)
        starts128 = d16_adaptive_starts('M2', center, (), bounds, target=128)
        self.assertEqual(d16_adaptive_tiers('M0'), (32, 81))
        self.assertEqual(d16_adaptive_tiers('M2'), (32, 128, 512, 59049))
        self.assertEqual(len(starts32), 32)
        self.assertEqual([item.start_id for item in starts32], [item.start_id for item in starts128[:32]])
        for first, second in zip(starts32, starts128):
            np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(starts32[0].values, center)

    def test_inner_huber_matches_independent_reference_and_applies_huber_once(self) -> None:
        capacity = np.asarray([0.4, 0.7, 1.0, 1.5, 2.2, 3.0])
        data = np.asarray([2.7, 1.9, 1.3, 0.9, 0.6, 0.4])
        y = -3.2 + 0.8 * capacity + 0.35 * data
        y[-1] += 1.5
        weights = np.asarray([0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
        problem = LanguageProblem(y, capacity, data, weights, -6.0, float(y.min()))
        fitted = solve_language(problem, 0.2)
        reference = solve_language_reference(problem, 0.2)
        self.assertTrue(fitted.success)
        self.assertTrue(reference.success)
        self.assertGreater(fitted.A, 0.0)
        self.assertGreater(fitted.B, 0.0)
        self.assertGreater(fitted.L_inf, -6.0)
        self.assertLess(fitted.L_inf, float(y.min()))
        self.assertAlmostEqual(fitted.objective, reference.objective, places=9)
        np.testing.assert_allclose([fitted.L_inf, fitted.A, fitted.B], [reference.L_inf, reference.A, reference.B], rtol=2e-05, atol=2e-06)
        expected_once = weighted_objective(fitted.residual, weights, 0.2)
        incorrectly_twice = float(np.sum(huber_loss(np.sqrt(weights) * fitted.residual, 0.2)))
        self.assertAlmostEqual(fitted.objective, expected_once, places=14)
        self.assertNotAlmostEqual(fitted.objective, incorrectly_twice, places=8)

    def test_synthetic_projected_recovery_for_each_candidate(self) -> None:
        for model in ('C0', 'J0', 'JH', 'M0', 'M1', 'M2'):
            with self.subTest(model=model):
                problem = _synthetic_problem(model)
                result = run_projected_start(model, TRUE_OUTER[model], problem, delta=None, maximum_iterations=25)
                self.assertTrue(result.success, result.message)
                self.assertIsNotNone(result.prediction)
                assert result.prediction is not None
                self.assertLess(float(np.max(np.abs(result.prediction - problem.y))), 2e-07)
                self.assertEqual(result.prediction.dtype, np.dtype(np.float64))

    def test_projected_and_direct_joint_fits_are_equivalent(self) -> None:
        problem = _synthetic_problem('M1', noise=0.0005)
        projected = run_projected_start('M1', TRUE_OUTER['M1'], problem, delta=0.01, maximum_iterations=250)
        self.assertTrue(projected.success, projected.message)
        check = direct_joint_check(projected, problem, delta=0.01, maximum_iterations=500)
        self.assertTrue(check.direct.success, check.direct.message)
        self.assertTrue(check.resolved, check.equivalence)
        self.assertLessEqual(check.equivalence.relative_objective_difference, 1e-06)
        self.assertLessEqual(check.equivalence.maximum_absolute_prediction_difference, 0.0001)

    def test_direct_joint_extreme_log_coefficients_return_finite_penalty(self) -> None:
        problem = _synthetic_problem('M0', noise=0.0)
        languages = tuple(sorted(set(problem.language_ids)))
        values = list(TRUE_OUTER['M0'])
        for _ in languages:
            values.extend((0.0, 1000.0, 0.0))
        objective, gradient, prediction, coefficients = _direct_objective_gradient(np.asarray(values, dtype=np.float64), model='M0', problem=problem, languages=languages, delta=0.01)
        self.assertEqual(objective, DIRECT_PENALTY_OBJECTIVE)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(coefficients, {})

    def test_boundary_expansion_saturation_and_nonidentification_paths(self) -> None:
        bounds = initial_bounds('M0')
        values = np.asarray([0.01001, 1.9999, -0.99995, 0.0])
        contacts = boundary_contacts(values, bounds)
        self.assertEqual({(item.name, item.side) for item in contacts}, {('alpha_e1', 'lower'), ('beta_e1', 'upper'), ('phi_e1', 'lower')})
        expanded = expand_contacted_bounds(bounds, contacts)
        by_name = {item.name: item for item in expanded.bounds}
        self.assertAlmostEqual(by_name['alpha_e1'].lower, 0.001)
        self.assertAlmostEqual(by_name['beta_e1'].upper, 4.0)
        self.assertEqual((by_name['phi_e1'].lower, by_name['phi_e1'].upper), (-2.0, 2.0))
        exhausted = tuple((OuterBound(item.name, item.lower, item.upper, item.kind, 3) if item.name == 'phi_e1' else item for item in expanded.bounds))
        decision = expand_contacted_bounds(exhausted, [BoundaryContact('phi_e1', 'lower', -2.0, 0.0, 4.0)])
        self.assertEqual(decision.nonidentified, ('phi_e1',))
        self.assertFalse(floor_is_saturated(0.5))
        self.assertTrue(floor_is_saturated(0.009))
        self.assertTrue(floor_is_saturated(0.991))
        local = deterministic_local_perturbations(values, bounds)
        self.assertEqual(len(local), 2 * len(bounds))
        self.assertEqual(len({item.start_id for item in local}), len(local))

    def test_basin_and_failure_paths_are_deterministic(self) -> None:
        prediction = np.asarray([1.0, 2.0, 3.0])
        results = [_result('b', 1.0, prediction + 5e-06), _result('a', 1.0 - 2e-07, prediction), _result('c', 1.0 + 2e-07, prediction - 5e-06), _result('failed', None, None, success=False)]
        basin = best_basin(results)
        self.assertTrue(basin.resolved)
        self.assertEqual(basin.best.start_id, 'a')
        self.assertEqual([item.start_id for item in basin.members], ['a', 'b', 'c'])
        comparison = equivalence_check(1.0, 1.1, prediction, prediction)
        self.assertFalse(comparison.equivalent)
        with self.assertRaises(ProjectedOptimizationError):
            best_basin([_result('failed', None, None, success=False)])

    def test_invalid_inner_iteration_limit_and_bad_problem_fail_closed(self) -> None:
        problem = LanguageProblem(np.asarray([-1.0, -1.2]), np.asarray([1.0, 0.8]), np.asarray([0.6, 1.1]), np.asarray([0.5, 0.5]), -3.0, -1.2)
        with self.assertRaisesRegex(ValueError, 'positive'):
            solve_language(problem, None, maximum_iterations=0)
        with self.assertRaisesRegex(ValueError, 'physical floor'):
            ProjectedProblem.from_arrays(y=[-2.0, -2.1], weights=[0.5, 0.5], N=[100000000.0, 200000000.0], D=[1000000000.0, 2000000000.0], E=[1, 1], K=[1, 1], language_ids=['eng', 'eng'], physical_floors=[-1.0, -1.0])


def _exposure_frame() -> pd.DataFrame:
    records = []
    mixtures = {
        "r2": {"a": 80.0, "b": 20.0},
        "r5": {"a": 40.0, "b": 25.0, "c": 15.0, "d": 12.0, "e": 8.0},
    }
    for readout, exposures in mixtures.items():
        for language, exposure in exposures.items():
            records.append({
                "readout_id": readout,
                "language": language,
                "K": len(exposures),
                "D_total": 100.0,
                "D_t": exposure,
                "adjusted_loss": -1.0,
            })
    return pd.DataFrame.from_records(records)


class Stage1TransferExposureTest(unittest.TestCase):

    def test_components_reconstruct_total_with_loss_blind_top_three(self) -> None:
        frame = _exposure_frame()
        components, transfer_sets = exposure_components(frame)
        np.testing.assert_allclose(
            components.sum(axis=1), frame["D_total"], rtol=0.0, atol=0.0
        )
        self.assertEqual(transfer_sets["a"], ("b", "c", "d"))
        row = frame.index[
            (frame["readout_id"] == "r5") & (frame["language"] == "e")
        ][0]
        self.assertEqual(components.loc[row, "D_target"], 8.0)
        self.assertEqual(components.loc[row, "D_transfer"], 80.0)
        self.assertEqual(components.loc[row, "D_other"], 12.0)

        changed = frame.copy()
        changed["adjusted_loss"] = np.arange(len(changed), dtype=np.float64) * 1000.0
        _, changed_sets = exposure_components(changed)
        self.assertEqual(transfer_sets, changed_sets)


    def test_models_are_exploratory_and_have_only_one_or_two_new_coordinates(self) -> None:
        self.assertEqual(PRIMARY_MODELS, ("M0", "M1", "M2"))
        self.assertEqual(EXPLORATORY_EXPOSURE_MODELS, ("M3", "M4"))
        self.assertEqual(len(outer_parameter_names("M2")), 10)
        self.assertEqual(len(outer_parameter_names("M3")), 11)
        self.assertEqual(len(outer_parameter_names("M4")), 12)


    def test_registered_parameter_serialization_remains_unchanged(self) -> None:
        serialized = SharedParameters(.2, 0.0, .3, 0.0,
            tau_transfer=.003, tau_other=.006).as_dict()
        self.assertNotIn("tau_transfer", serialized)
        self.assertNotIn("tau_other", serialized)


    def test_exact_m2_m3_m4_nesting(self) -> None:
        target = np.asarray([10.0, 30.0])
        transfer = np.asarray([20.0, 5.0])
        other = np.asarray([70.0, 65.0])
        m2 = SharedParameters(.2, 0.0, .3, 0.0)
        np.testing.assert_array_equal(effective_exposure_data("M2", m2, target), target)
        zero_m4 = SharedParameters(.2, 0.0, .3, 0.0, tau_transfer=0.0, tau_other=0.0)
        np.testing.assert_array_equal(
            effective_exposure_data("M4", zero_m4, target, transfer, other), target)
        common = SharedParameters(.2, 0.0, .3, 0.0, tau_transfer=.003, tau_other=.003)
        np.testing.assert_array_equal(
            effective_exposure_data("M3", common, target, transfer, other),
            effective_exposure_data("M4", common, target, transfer, other),
        )


    def test_m3_m4_data_jacobians_match_finite_differences(self) -> None:
        base = np.asarray([.24, .34, .19, .23, -.07, .04, .06, -.03, -.02, .05])
        N = np.asarray([1.0e8, 3.0e8, 6.0e8])
        target = np.asarray([2.0e8, 1.0e9, 5.0e9])
        transfer = np.asarray([3.0e8, 8.0e8, 2.0e9])
        other = np.asarray([5.0e8, 2.0e9, 7.0e9])
        E = np.asarray([1.0, 4.0, 32.0])
        K = np.asarray([1.0, 12.0, 200.0])
        for model, outer in (
            ("M3", np.r_[base, .1]),
            ("M4", np.r_[base, .2, .1]),
        ):
            with self.subTest(model=model):
                _, _, _, data_jac = predictor_terms_and_jacobian(
                    model, outer, N, target, E, K,
                    D_transfer=transfer, D_other=other,
                )
                epsilon = 1e-6
                for column in range(10, outer.size):
                    plus, minus = outer.copy(), outer.copy()
                    plus[column] += epsilon
                    minus[column] -= epsilon
                    plus_shared = shared_from_outer(model, plus)
                    minus_shared = shared_from_outer(model, minus)
                    plus_d = effective_exposure_data(model, plus_shared, target, transfer, other)
                    minus_d = effective_exposure_data(model, minus_shared, target, transfer, other)
                    plus_term = predictor_terms(model, plus_shared, N, plus_d, E, K)[1]
                    minus_term = predictor_terms(model, minus_shared, N, minus_d, E, K)[1]
                    numerical = (plus_term - minus_term) / (2.0 * epsilon)
                    np.testing.assert_allclose(data_jac[:, column], numerical, rtol=2e-7, atol=2e-10)


    def test_tau_bounds_keep_zero_and_expand_the_upper_cap_once(self) -> None:
        bounds = initial_bounds("M3")
        self.assertEqual((bounds[-1].lower, bounds[-1].upper, bounds[-1].kind),
            (0.0, 1.0, "nonnegative"))
        lower = boundary_contacts(np.r_[np.asarray([.2, .3, .2, .3, 0, 0, 0, 0, 0, 0]), 0.0], bounds)
        lower_tau = [item for item in lower if item.name == "tau_other_u"]
        unchanged = expand_contacted_bounds(bounds, lower_tau)
        self.assertEqual(unchanged.bounds[-1], bounds[-1])
        upper = boundary_contacts(np.r_[np.asarray([.2, .3, .2, .3, 0, 0, 0, 0, 0, 0]), 1.0], bounds)
        expanded = expand_contacted_bounds(bounds, [item for item in upper if item.name == "tau_other_u"])
        self.assertAlmostEqual(expanded.bounds[-1].upper * ATLAS_TRANSFER_TAU_SCALE, 0.1)
        second = expand_contacted_bounds(expanded.bounds,
            [type(upper[-1])("tau_other_u", "upper", expanded.bounds[-1].upper, 0.0,
                expanded.bounds[-1].upper)])
        self.assertEqual(second.nonidentified, ("tau_other_u",))


    def test_projected_solver_accepts_component_exposures(self) -> None:
        base = np.asarray([.28, .36, .22, .19, -.1, .04, .08, -.02, -.04, .03])
        outer = np.r_[base, .2, .1]
        N = np.tile(np.asarray([1.0e8, 2.5e8, 6.0e8, 1.0e8, 2.5e8, 6.0e8]), 2)
        target = np.tile(np.asarray([2.0e8, 1.0e9, 5.0e9, 4.0e8, 2.0e9, 8.0e9]), 2)
        transfer = np.tile(np.asarray([3.0e8, 8.0e8, 2.0e9, 4.0e8, 1.0e9, 3.0e9]), 2)
        other = np.tile(np.asarray([5.0e8, 2.0e9, 7.0e9, 8.0e8, 3.0e9, 9.0e9]), 2)
        E = np.tile(np.asarray([1.0, 4.0, 32.0, 1.0, 4.0, 32.0]), 2)
        K = np.tile(np.asarray([1.0, 12.0, 200.0, 12.0, 200.0, 1.0]), 2)
        languages = tuple(["deu_Latn"] * 6 + ["eng_Latn"] * 6)
        shared = shared_from_outer("M4", outer)
        effective = effective_exposure_data("M4", shared, target, transfer, other)
        capacity, data = predictor_terms("M4", shared, N, effective, E, K)
        floor = np.asarray([-5.0] * 6 + [-5.7] * 6)
        a = np.asarray([.65] * 6 + [.48] * 6)
        b = np.asarray([.31] * 6 + [.42] * 6)
        y = floor + a * capacity + b * data
        problem = ProjectedProblem.from_arrays(
            y=y, weights=trajectory_weights([f"t{index // 2}" for index in range(y.size)]),
            N=N, D=target, D_transfer=transfer, D_other=other, E=E, K=K,
            language_ids=languages, physical_floors=np.full(y.size, -8.0),
        )
        result = run_projected_start("M4", outer, problem, delta=None, maximum_iterations=50)
        self.assertTrue(result.success, result.message)
        self.assertLess(float(np.max(np.abs(result.prediction - y))), 2e-7)
