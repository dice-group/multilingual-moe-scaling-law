from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from multilingual_megatron.experiments.scaling.stage1_fit.data import EXPECTED_RELEASE_MANIFEST_FILE_SHA256, EXPECTED_RELEASE_MANIFEST_INTEGRITY_SHA256, FitRow, fit_rows_content_sha256, sha256_file
from multilingual_megatron.experiments.scaling.stage1_fit.data import (DEFAULT_ARCHIVED_MANIFEST_PATH,
    DEFAULT_RELEASE_ROOT, Stage1FitDataError, assemble_fit_frame, build_release_contract,
    load_canonical_fit_data, load_release_contract, write_fit_data_snapshot,
    write_release_contract)

def _tiny_tables() -> dict[str, pd.DataFrame]:
    trajectories = pd.DataFrame([{'trajectory_id': 't1', 'ladder': 'A', 'experts': 1, 'language_count': 1, 'model_config_id': 'm1', 'model_seed': 7, 'status': 'observed_complete', 'provenance_class': 'configured'}, {'trajectory_id': 't2', 'ladder': 'B', 'experts': 4, 'language_count': 1, 'model_config_id': 'm2', 'model_seed': 7, 'status': 'observed_complete', 'provenance_class': 'configured'}])
    model_configs = pd.DataFrame([{'model_config_id': 'm1', 'ladder': 'A', 'experts': 1, 'declared_active_parameters': 100, 'declared_active_nonembedding_parameters': 80, 'provenance_class': 'configured'}, {'model_config_id': 'm2', 'ladder': 'B', 'experts': 4, 'declared_active_parameters': 200, 'declared_active_nonembedding_parameters': 160, 'provenance_class': 'configured'}])
    readouts = pd.DataFrame([{'readout_id': 'r1', 'trajectory_id': 't1', 'ladder': 'A', 'experts': 1, 'language_count': 1, 'actual_tokens': 1000, 'trained_language_count': 1, 'primary_metric_name': 'trained_equal_language_atlas_tao_unigram_adjusted_loss', 'provenance_class': 'observed'}, {'readout_id': 'r2', 'trajectory_id': 't2', 'ladder': 'B', 'experts': 4, 'language_count': 1, 'actual_tokens': 2000, 'trained_language_count': 1, 'primary_metric_name': 'trained_equal_language_atlas_tao_unigram_adjusted_loss', 'provenance_class': 'observed'}])
    validation = pd.DataFrame([{'readout_id': readout, 'trajectory_id': trajectory, 'ladder': ladder, 'experts': experts, 'language_count': 1, 'actual_tokens': tokens, 'language': 'eng_Latn', 'metric_role': 'trained', 'token_count': 10, 'nll_sum': raw * 10, 'raw_nll': raw, 'adjusted_nll_sum': (raw - 7.0) * 10, 'atlas_tao_unigram_adjusted_loss': raw - 7.0, 'unigram_log_probability_sum': -70.0, 'vocab_size': 64000, 'special_token_ids_json': '[0,1]', 'validation_input_identity_sha256': 'v' * 64, 'target_token_ids_sha256': 't' * 64, 'unigram_counts_sha256': 'u' * 64, 'provenance_class': 'observed'} for readout, trajectory, ladder, experts, tokens, raw in (('r1', 't1', 'A', 1, 1000, 3.0), ('r2', 't2', 'B', 4, 2000, 3.2))])
    exposure = pd.DataFrame([{'readout_id': readout, 'trajectory_id': trajectory, 'ladder': ladder, 'experts': experts, 'language_count': 1, 'actual_tokens': tokens, 'language': 'eng_Latn', 'metric_role': 'trained', 'realized_self_exposure_tokens': exposure_tokens, 'provenance_class': 'observed'} for readout, trajectory, ladder, experts, tokens, exposure_tokens in (('r1', 't1', 'A', 1, 1000, 997.0), ('r2', 't2', 'B', 4, 2000, 1876.0))])
    return {'trajectories': trajectories, 'readouts': readouts, 'model_configs': model_configs, 'validation_language': validation, 'data_exposure': exposure}

def _as_rows(frame: pd.DataFrame) -> list[FitRow]:
    return [FitRow(**record) for record in frame.to_dict('records')]

class Stage1FitDataTest(unittest.TestCase):

    def test_exact_mapping_and_adjusted_loss_derivation(self) -> None:
        frame = assemble_fit_frame(**_tiny_tables(), strict_counts=False)
        self.assertEqual(list(frame['N_act']), [100.0, 200.0])
        self.assertEqual(list(frame['D_total']), [1000.0, 2000.0])
        self.assertEqual(list(frame['D_t']), [997.0, 1876.0])
        self.assertEqual(list(frame['unigram_nll']), [7.0, 7.0])
        self.assertEqual(list(frame['adjusted_loss']), [-4.0, -3.8])
        self.assertEqual(list(frame['model_seed']), [7, 7])

    def test_duplicate_missing_and_bad_response_fail_closed(self) -> None:
        duplicate = _tiny_tables()
        duplicate['validation_language'] = pd.concat([duplicate['validation_language'], duplicate['validation_language'].iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(Stage1FitDataError, 'duplicate trained validation'):
            assemble_fit_frame(**duplicate, strict_counts=False)
        missing = _tiny_tables()
        missing['data_exposure'] = missing['data_exposure'].iloc[:1].copy()
        with self.assertRaisesRegex(Stage1FitDataError, 'coverage mismatch'):
            assemble_fit_frame(**missing, strict_counts=False)
        invalid = _tiny_tables()
        invalid['validation_language'].loc[0, 'atlas_tao_unigram_adjusted_loss'] += 0.25
        with self.assertRaisesRegex(Stage1FitDataError, 'adjusted-loss'):
            assemble_fit_frame(**invalid, strict_counts=False)

    def test_canonical_content_hash_is_order_independent_and_sensitive(self) -> None:
        frame = assemble_fit_frame(**_tiny_tables(), strict_counts=False)
        rows = _as_rows(frame)
        reverse = list(reversed(rows))
        self.assertEqual(fit_rows_content_sha256(rows), fit_rows_content_sha256(reverse))
        changed = list(rows)
        changed[0] = FitRow(**{**changed[0].to_dict(), 'D_t': changed[0].D_t + 1.0})
        self.assertNotEqual(fit_rows_content_sha256(rows), fit_rows_content_sha256(changed))

    def test_snapshot_write_is_atomic_receipted_and_no_overwrite(self) -> None:
        prepared = load_canonical_fit_data()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / 'fit'
            receipt = write_fit_data_snapshot(prepared, output_root)
            receipt_path = output_root / 'input' / 'input_receipt.json'
            payload = json.loads(receipt_path.read_text(encoding='utf-8'))
            self.assertEqual(payload, receipt.to_dict())
            self.assertEqual(payload['fit_rows'], 26300)
            self.assertEqual(payload['fit_rows_content_sha256'], prepared.content_sha256)
            self.assertEqual(sha256_file(Path(payload['fit_rows_path'])), payload['fit_rows_file_sha256'])
            with self.assertRaisesRegex(Stage1FitDataError, 'overwrite'):
                write_fit_data_snapshot(prepared, output_root)

    def test_read_only_canonical_release_planning_smoke(self) -> None:
        manifest_before = sha256_file(DEFAULT_ARCHIVED_MANIFEST_PATH)
        data = load_canonical_fit_data(DEFAULT_RELEASE_ROOT)
        manifest_after = sha256_file(DEFAULT_ARCHIVED_MANIFEST_PATH)
        self.assertEqual((data.trajectory_count, data.readout_count, len(data.rows)), (64, 400, 26300))
        self.assertEqual(data.archive_manifest_file_sha256, EXPECTED_RELEASE_MANIFEST_FILE_SHA256)
        self.assertEqual(data.release_manifest_integrity_sha256, EXPECTED_RELEASE_MANIFEST_INTEGRITY_SHA256)
        self.assertEqual(manifest_before, manifest_after)
        self.assertEqual(len(data.source_tables), 5)
        self.assertTrue(all((item.sha256 == item.manifest_sha256 for item in data.source_tables)))

    def test_explicit_release_contract_round_trip_and_tamper_rejection(self) -> None:
        contract = build_release_contract(DEFAULT_RELEASE_ROOT, DEFAULT_ARCHIVED_MANIFEST_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            path = write_release_contract(contract, Path(temporary) / 'release_contract.json')
            self.assertEqual(load_release_contract(path), contract)
            prepared = load_canonical_fit_data(
                DEFAULT_RELEASE_ROOT,
                archive_manifest_path=DEFAULT_ARCHIVED_MANIFEST_PATH,
                release_contract_path=path,
            )
            self.assertEqual(prepared.release_contract_sha256, contract.contract_sha256)
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['release_id'] = 'tampered'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(Stage1FitDataError, 'integrity mismatch'):
                load_release_contract(path)
