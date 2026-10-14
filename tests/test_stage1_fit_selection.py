from __future__ import annotations
import sys
import unittest
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from multilingual_megatron.experiments.scaling.stage1_fit.data import FitRow, canonical_json_sha256
from multilingual_megatron.experiments.scaling.stage1_fit.selection import Stage1FitSelectionError, aggregate_equal_fold_rmse, build_generalization_holdout_manifest, build_holdout_manifest, select_model_paired_one_standard_error, trajectory_balanced_r2, trajectory_balanced_rmse
from multilingual_megatron.experiments.scaling.stage1_fit.selection import Stage1FitUncertaintyError, build_resampling_manifest, generate_uncertainty_subsets

def _row(trajectory_id: str, readout: int, *, n_act: float, experts: int, language_count: int, language: str, loss: float) -> FitRow:
    return FitRow(trajectory_id=trajectory_id, readout_id=f'{trajectory_id}:r{readout}', language=language, N_act=n_act, N_act_nonembedding=0.8 * n_act, E=experts, K=language_count, D_total=float(readout * 1000), D_t=float(readout * 100), adjusted_loss=loss, raw_nll=loss + 7.0, validation_token_count=10, unigram_nll=7.0, ladder='A' if n_act == 100.0 else 'B', model_seed=1234)

def _holdout_rows() -> list[FitRow]:
    configurations = (('t0', 100.0, 1, 50, ('eng',)), ('t1', 100.0, 32, 50, ('eng',)), ('t2', 200.0, 1, 50, ('eng',)), ('t3', 100.0, 1, 200, ('eng', 'new_only')))
    rows: list[FitRow] = []
    for index, (trajectory, n_act, experts, languages, members) in enumerate(configurations):
        for readout in (1, 2):
            for language_index, language in enumerate(members):
                rows.append(_row(trajectory, readout, n_act=n_act, experts=experts, language_count=languages, language=language, loss=1.0 + index + language_index * 0.1 - readout * 0.01))
    return rows

class Stage1FitSelectionTest(unittest.TestCase):

    def test_folds_keep_trajectories_whole_and_exclude_unseen_k200_languages(self) -> None:
        manifest = build_holdout_manifest(_holdout_rows(), input_sha256='a' * 64, random_count=1, lowest_loss_count=2, require_canonical_population=False)
        folds = {fold.fold_id: fold for fold in manifest.primary_folds}
        self.assertEqual(folds['largest_N_act'].test_trajectory_ids, ('t2',))
        self.assertEqual(folds['E32'].test_trajectory_ids, ('t1',))
        self.assertEqual(folds['K200_common_languages'].test_trajectory_ids, ('t3',))
        self.assertEqual(folds['K200_common_languages'].score_languages, ('eng',))
        self.assertEqual(folds['K200_common_languages'].excluded_test_languages, ('new_only',))
        for fold in manifest.primary_folds:
            self.assertFalse(set(fold.train_trajectory_ids) & set(fold.test_trajectory_ids))
        self.assertEqual(len(manifest.terminal_continuation.test_readout_ids), 4)
        self.assertEqual(len(manifest.lowest_loss_extrapolation.test_trajectory_ids), 2)
        self.assertIn('Secondary only', manifest.lowest_loss_definition)

    def test_logical_compute_split_is_exact_and_tie_safe(self) -> None:
        rows = [_row(trajectory, readout, n_act=n_act, experts=1, language_count=12,
            language='eng', loss=1.0) for trajectory, n_act in (('t0', 100.0), ('t1', 200.0))
            for readout in range(1, 6)]
        source = {'path': '/frozen/holdouts.json', 'file_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64}
        manifest = build_generalization_holdout_manifest(rows, input_sha256='c' * 64,
            source_holdouts=source, require_canonical_population=False)
        split = manifest['largest_logical_compute_top20']
        self.assertEqual((len(split['train_readout_ids']), len(split['test_readout_ids'])), (8, 2))
        self.assertEqual(manifest['logical_compute']['threshold_flops'], 4_800_000)
        self.assertEqual(split['test_readout_ids'], ['t1:r4', 't1:r5'])
        self.assertFalse(set(split['train_readout_ids']) & set(split['test_readout_ids']))
        self.assertEqual(set(split['train_readout_ids']) | set(split['test_readout_ids']), {row.readout_id for row in rows})
        self.assertEqual(split['score_languages'], ['eng'])
        self.assertTrue(set(split['test_trajectory_ids']) <= set(split['train_trajectory_ids']))
        tied = [FitRow(**{**row.to_dict(), 'D_total': 4000.0}) if row.readout_id == 't1:r3' else row
            for row in rows]
        with self.assertRaisesRegex(Stage1FitSelectionError, 'tie crosses'):
            build_generalization_holdout_manifest(tied, input_sha256='c' * 64,
                source_holdouts=source, require_canonical_population=False)

    def test_trajectory_balancing_and_equal_fold_aggregation(self) -> None:
        observed = [0.0, 0.0, 0.0, 0.0]
        predicted = [1.0, 1.0, 1.0, 3.0]
        trajectory_ids = ['many', 'many', 'many', 'single']
        self.assertAlmostEqual(trajectory_balanced_rmse(observed, predicted, trajectory_ids), 5.0 ** 0.5)
        self.assertIsNone(trajectory_balanced_r2(observed, predicted, trajectory_ids))
        self.assertEqual(aggregate_equal_fold_rmse({'small': 1.0, 'large': 3.0}), 2.0)

    def test_paired_one_standard_error_selects_simplest_practical_tie(self) -> None:
        fold_errors = {'M0': {fold: {'a': 0.81, 'b': 0.81, 'c': 1.21, 'd': 1.21} for fold in ('f1', 'f2', 'f3')}, 'M1': {fold: {name: 1.0 for name in ('a', 'b', 'c', 'd')} for fold in ('f1', 'f2', 'f3')}, 'M2': {fold: {name: 1.44 for name in ('a', 'b', 'c', 'd')} for fold in ('f1', 'f2', 'f3')}}
        first = select_model_paired_one_standard_error(fold_errors)
        second = select_model_paired_one_standard_error(fold_errors)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.observed_best_model, 'M1')
        self.assertEqual(first.selected_model, 'M0')
        self.assertIn('M0', first.eligible_models)
        self.assertNotIn('M2', first.eligible_models)
        self.assertEqual(first.bootstrap_replicates, 10000)

    def test_overlapping_folds_reuse_one_union_bootstrap_multiplicity(self) -> None:
        import math
        import numpy as np
        errors = {
            'M0': {'f1': {'a': .1, 'b': .4, 'c': .9}, 'f2': {'b': .4, 'c': .9, 'd': 1.6}},
            'M1': {'f1': {'a': .2, 'b': .3, 'c': .8}, 'f2': {'b': .3, 'c': .8, 'd': 1.4}},
        }
        replicates = 50; seed = 123
        actual = select_model_paired_one_standard_error(errors, replicates=replicates, seed=seed)
        generator = np.random.Generator(np.random.PCG64(seed)); population = ('a', 'b', 'c', 'd')
        folds = {'f1': ('a', 'b', 'c'), 'f2': ('b', 'c', 'd')}; index = {name: i for i, name in enumerate(population)}
        values = {model: [] for model in errors}
        for _ in range(replicates):
            while True:
                counts = np.bincount(generator.integers(0, 4, size=4), minlength=4)
                masses = {fold: sum(counts[index[name]] for name in names) for fold, names in folds.items()}
                if all(masses.values()): break
            for model in errors:
                score = sum(math.sqrt(sum(errors[model][fold][name] * counts[index[name]] for name in names) / masses[fold])
                    for fold, names in folds.items()) / 2.0
                values[model].append(score)
        best = actual.observed_best_model
        for model in errors:
            expected = float(np.std(np.asarray(values[model]) - np.asarray(values[best]), ddof=1))
            self.assertAlmostEqual(actual.paired_standard_error_vs_best[model], expected)
        self.assertEqual(actual.bootstrap_population_trajectories, 4)
        self.assertIn('shared multiplicities', actual.bootstrap_sampling)

    def test_canonical_pcg64_membership_hashes(self) -> None:
        trajectories = sorted((f'{ladder}:E{experts}:K{languages}' for ladder in 'ABCD' for experts in (1, 4, 16, 32) for languages in (1, 12, 50, 200)))
        uncertainty = generate_uncertainty_subsets(trajectories)
        self.assertEqual(len(uncertainty), 100)
        self.assertTrue(all((len(item.trajectory_ids) == 51 for item in uncertainty)))
        self.assertEqual(uncertainty[0].membership_sha256, '9881354b679e6dd883d77e361d302bbbfe345ea0af46f1e119e51dd36d087494')
        import numpy as np
        generator = np.random.Generator(np.random.PCG64(20260803))
        indices = generator.choice(len(trajectories), size=13, replace=False)
        random_membership = sorted((trajectories[int(index)] for index in indices))
        self.assertEqual(canonical_json_sha256(random_membership), '64bfc21970562c00c48e960b25fd100e2b3fc49adefa606253462f910d33b8d7')
        manifest = build_resampling_manifest(trajectories, input_sha256='b' * 64)
        self.assertEqual(manifest.bit_generator, 'PCG64')
        self.assertEqual(manifest.subset_size, 51)
