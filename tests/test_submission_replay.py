"""CPU checks for the evidence actually delivered in the anonymous archive."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from result_analysis.scaling_law_fit.submission_replay import (
    ROOT, align, fold_rows, run, scores, verify_manifest,
)


class SubmissionReplayTests(unittest.TestCase):
    def test_delivered_study_specs_validate_without_private_campaigns(self):
        # Validate study definitions, not historical launch authorizations/receipts.
        from multilingual_megatron.experiments.scaling.stage1_training import seed_confirmation as seed
        from multilingual_megatron.experiments.scaling.stage1_training import lr_confirmation as lr
        from multilingual_megatron.experiments.scaling.language_composition.training.training_contracts import validate_study_spec
        repo = ROOT.parent
        seed.validate_seed_confirmation_spec(repo_root=repo, spec_path=repo/seed.DEFAULT_SPEC)
        lr.validate_stage1_lr_confirmation_spec(repo_root=repo, spec_path=repo/lr.DEFAULT_SPEC)
        validate_study_spec(repo_root=repo, spec_path=repo/'configs/scaling_law/multilingual_moe_ab_v2/study_spec_language_composition_v1.json')

    def test_delivered_laws_reproduce_predictions_and_paper_scores(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)/'replay'
            receipt = run(ROOT/'stage1_selected_law_v1', ROOT/'stage1_jointmoe_token_contrast_v1', output)
            self.assertEqual(receipt['selected_metric_cells'], 7)
            self.assertEqual(receipt['jointmoe_metric_cells'], 8)
            self.assertEqual(receipt['composition_cells'], 108)
            self.assertLess(receipt['selected_prediction_max_abs_error'], 1e-10)
            self.assertLess(receipt['jointmoe_prediction_max_abs_error'], 1e-10)
            selected = pd.read_csv(output/'selected_scores.csv')
            primary = selected[selected.case.eq('grid')].set_index('fold')
            for fold, expected in [('largest_N_act', .029213878306493465),
                                   ('E32', .02575730536617431),
                                   ('K200_common_languages', .04460994894944229)]:
                self.assertAlmostEqual(primary.loc[fold, 'rmse'], expected, places=12)
            self.assertAlmostEqual(selected[selected.fold.eq('all_grid')].rmse.item(),
                                   .17297775272574958, places=12)
            baselines = pd.read_csv(output/'jointmoe_scores.csv')
            mean_dt = baselines[baselines.token.eq('Dt') & ~baselines.fold.eq('all_grid')].rmse.mean()
            self.assertAlmostEqual(mean_dt, .04616913048181922, places=12)
            self.assertAlmostEqual(1-primary.rmse.mean()/mean_dt, .281041, places=5)

    def test_historical_parquet_checksums_still_match(self):
        root = ROOT/'stage1_data_v1'
        manifest = json.loads((root/'manifest.json').read_text())
        for name, entry in manifest['tables'].items():
            self.assertEqual(hashlib.sha256((root/(name+'.parquet')).read_bytes()).hexdigest(),
                             entry['sha256'], name)

    def test_manifest_rejects_changed_or_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data.csv'
            data.write_text('loss\n0.1\n')
            manifest = {'files': {'data.csv': hashlib.sha256(data.read_bytes()).hexdigest()}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            verify_manifest(root)
            data.write_text('loss\n0.2\n')
            with self.assertRaisesRegex(ValueError, 'file hash mismatch'):
                verify_manifest(root)
            data.unlink()
            with self.assertRaisesRegex(ValueError, 'file hash mismatch'):
                verify_manifest(root)

    def test_manifest_cannot_escape_delivery_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps({'files': {'../escape': '0'*64}}))
            with self.assertRaisesRegex(ValueError, 'unsafe evidence path'):
                verify_manifest(root)

    @staticmethod
    def observations():
        return pd.DataFrame({'trajectory_id': ['a', 'b', 'b'], 'readout_id': ['r0', 'r1', 'r2'],
                             'language': ['en', 'fr', 'fr'], 'adjusted_loss': [1., 2., 3.],
                             'prediction': [2., 2., 3.]})

    def test_alignment_rejects_missing_extra_duplicate_and_changed_observations(self):
        saved = self.observations()
        measured = saved.drop(columns='prediction')
        self.assertEqual(len(align(saved, measured)), 3)
        for bad in (measured.iloc[:-1], pd.concat([measured, measured.iloc[:1]])):
            with self.assertRaises(ValueError):
                align(saved, bad)
        changed = measured.copy()
        changed.loc[0, 'adjusted_loss'] += .01
        with self.assertRaises(AssertionError):
            align(saved, changed)

    def test_score_weights_trajectories_equally_and_endpoint_languages_equally(self):
        rows = self.observations()
        self.assertAlmostEqual(scores(rows)['rmse'], np.sqrt(.5))
        with self.assertRaisesRegex(ValueError, 'one row per language'):
            scores(rows, endpoint=True)
        rows.loc[2, 'language'] = 'de'
        self.assertAlmostEqual(scores(rows, endpoint=True)['rmse'], np.sqrt(1/3))

    def test_membership_rejects_overlap_and_limits_coverage_holdout_languages(self):
        rows = self.observations()
        membership = dict(train_trajectory_ids=['a'], test_trajectory_ids=['b'],
                          test_readout_ids=['r1'], score_languages=['fr'])
        self.assertEqual(fold_rows(rows, rows, membership, 'grid').readout_id.tolist(), ['r1'])
        membership['train_trajectory_ids'].append('b')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            fold_rows(rows, rows, membership, 'grid')

    def test_output_must_not_modify_evidence(self):
        with self.assertRaisesRegex(ValueError, 'overlaps evidence'):
            run(ROOT/'stage1_selected_law_v1', ROOT/'stage1_jointmoe_token_contrast_v1',
                ROOT/'stage1_selected_law_v1/new-output')


if __name__ == '__main__':
    unittest.main()
