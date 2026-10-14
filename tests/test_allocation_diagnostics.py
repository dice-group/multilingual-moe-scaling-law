"""Allocation comparisons must remain within measured compute support."""
import unittest

import pandas as pd

from result_analysis.compute_allocation.diagnostics import allocation_margins


class AllocationDiagnosticsTest(unittest.TestCase):
    def test_allocation_margins_reject_unsupported_budget(self):
        frame = pd.DataFrame(dict(cap=[400000000]*2, method=['smooth']*2,
            K=[12]*2, config=['A/E16']*2, F=[1.,3.], loss=[2.,1.]))
        with self.assertRaisesRegex(ValueError, 'outside support'):
            allocation_margins(frame)


if __name__ == '__main__':
    unittest.main()
