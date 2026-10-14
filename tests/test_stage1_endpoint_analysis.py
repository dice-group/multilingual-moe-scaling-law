"""Scientific arithmetic for paired endpoint diagnostics; no GPU dependencies."""
import unittest
import numpy as np

from result_analysis.downstream_evaluations.endpoints.example_diagnostics import (
    paired_bootstrap, observations, f1_scores,
)
from result_analysis.downstream_evaluations.endpoints.endpoint_diagnostics import orientation, scale


class EndpointAnalysisTest(unittest.TestCase):
    def test_strict_minimal_pair_tie_is_not_a_correct_answer(self):
        task={'documents':[dict(conditional=['a','b'],target=0,char_lengths=[2,2],byte_lengths=[2,2],token_lengths=[1,1])]}
        scores={k:dict(loglikelihood=-2.,scored_tokens=1) for k in ['a','b']}
        value=observations(task,scores,'multiblimp')
        self.assertEqual(value['primary'].item(),0)
        self.assertEqual(value['margin'].item(),0)

    def test_pmi_can_reverse_raw_prediction(self):
        task={'documents':[dict(conditional=['a','b'],unconditional=['u','v'],target=1,char_lengths=[2,2],token_lengths=[1,1])]}
        scores={k:dict(loglikelihood=v,scored_tokens=1) for k,v in zip(['a','b','u','v'],[-1,-2,-1,-5])}
        value=observations(task,scores,'x_csqa')
        self.assertEqual(value['accuracy_raw'].item(),0)
        self.assertEqual(value['primary'].item(),1)

    def test_loss_windows_remain_one_document_and_tokens_are_checked(self):
        task={'documents':[dict(conditional=['a','b'],token_lengths=[2,1],target_tokens=3,utf8_bytes=8)]}
        scores={'a':dict(loglikelihood=-4,scored_tokens=2),'b':dict(loglikelihood=-3,scored_tokens=1)}
        value=observations(task,scores,'ntrex_mono')
        np.testing.assert_equal(value['nll'],[7]); np.testing.assert_equal(value['tokens'],[3])
        scores['b']['scored_tokens']=2
        with self.assertRaises(AssertionError): observations(task,scores,'ntrex_mono')

    def test_paired_bootstrap_cancels_shared_checkpoint_noise(self):
        num=np.array([[0,1,2,3],[10,11,12,13]],float)
        point,limits,blocks=paired_bootstrap(num,np.ones_like(num),['x','y'])
        np.testing.assert_allclose(point,[1,1,0]);np.testing.assert_allclose(limits,[[1,1,0],[1,1,0]])
        self.assertEqual(blocks,2)

    def test_bootstrap_uses_ratio_of_sums_and_groups_source_ids(self):
        num=np.array([[0,1,0,3],[0,2,0,6]],float); den=np.array([[1]*4,[2]*4])
        point,limits,blocks=paired_bootstrap(num,den,['same','same'])
        np.testing.assert_allclose(point,[1,3,2]);np.testing.assert_allclose(limits,[[1,3,2],[1,3,2]])
        self.assertEqual(blocks,1)

    def test_majority_predictions_can_have_low_macro_f1(self):
        cm,macro,weighted=f1_scores(np.array([0,0,0,1]),np.array([0,0,0,0]),2)
        self.assertEqual(cm.sum(),4);self.assertAlmostEqual(macro,3/7);self.assertAlmostEqual(weighted,9/14)

    def test_loss_accuracy_units_and_orientation_are_distinct(self):
        self.assertEqual(orientation('flores_mono','primary'),-1)
        self.assertEqual(scale('flores_mono','primary'),1)
        self.assertEqual(orientation('multiblimp','primary'),1)
        self.assertEqual(scale('multiblimp','primary'),100)


if __name__=='__main__': unittest.main()
