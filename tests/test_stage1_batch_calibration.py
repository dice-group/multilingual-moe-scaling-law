"""Calibration cannot silently select a batch untested on an expert setting."""
import unittest

from multilingual_megatron.experiments.scaling.stage1_benchmarking.batch_calibration import (
    common_candidates, select_long_documents,
)


class BatchCalibrationTest(unittest.TestCase):
    def test_longest_selection_preserves_complete_documents_and_source_order(self):
        rows = [dict(task='x', doc_id=i) for i in (1, 2, 1, 3)]
        self.assertEqual(select_long_documents(rows, [2048, 100, 31, 1800], 1), [0, 2])

    def test_selection_uses_common_measured_sizes_and_rejects_incomplete_grid(self):
        receipts = {(a,e): dict(status='calibrated',trials=[
            dict(status='measured',upper_length=2048,batch_size=b,memory_ok=True,parity_ok=True)
            for b in (4,8,16)]) for a in 'ABCD' for e in (1,4,16,32)}
        receipts['D',32]['trials'][-1]['memory_ok'] = False
        receipts['B',16]['trials'][-1]['parity_ok'] = False
        self.assertEqual(common_candidates(receipts),dict(A=16,B=8,C=16,D=8))
        del receipts['A',1]
        with self.assertRaisesRegex(ValueError,'all 16'):
            common_candidates(receipts)

    def test_failed_calibration_cannot_be_accepted_from_partial_trials(self):
        receipts = {(a,e): dict(status='failed',trials=[]) for a in 'ABCD' for e in (1,4,16,32)}
        with self.assertRaisesRegex(ValueError,'incomplete'):
            common_candidates(receipts)


if __name__ == '__main__':
    unittest.main()
