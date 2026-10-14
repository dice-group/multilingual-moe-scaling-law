"""Paper release guards: complete matrix and immutable scientific bindings."""
import copy
import unittest
from multilingual_megatron.experiments.scaling.stage1_benchmarking.h100_release import batch_for, check_gate, check_spec, validate_matrix


class H100ReleaseTest(unittest.TestCase):
    def test_model_specific_policy_reports_drift_without_claiming_invariance(self):
        cells={(a,e,200) for a in 'ABCD' for e in (1,4,16,32)}|{(a,e,12) for a in 'ABCD' for e in (1,32)}
        rows=[dict(ladder=a,E=e,K=k,passed=True,max_score_difference=2,max_per_token_difference=.1,decision_changes=1) for a,e,k in cells]
        gate=dict(complete=True,passed=True,comparisons=rows,
                  acceptance_policy='model_specific_memory_safe_with_reported_sensitivity',
                  runtime_checks_passed=True,capacity_checks_passed=True,
                  metric_differences=[dict(absolute_difference=.01)]*672,
                  paired_contrasts=[dict(absolute_difference=.02)]*112)
        check_gate(gate)
        with self.assertRaises(ValueError):check_gate(dict(gate,capacity_checks_passed=False))
        with self.assertRaises(ValueError):check_gate(dict(gate,metric_differences=[]))
        with self.assertRaises(ValueError):check_gate(dict(gate,acceptance_policy='strict'))
        self.assertEqual(batch_for({'D_e1':128,'D_e32':64},'D',32),64)
        with self.assertRaises(ValueError):batch_for({'D_e1':128},'D',32)

    def test_gate_requires_all_architectures_and_coverage_endpoints(self):
        cells={(a,e,200) for a in 'ABCD' for e in (1,4,16,32)}|{(a,e,12) for a in 'ABCD' for e in (1,32)}
        rows=[dict(ladder=a,E=e,K=k,passed=True,max_score_difference=0,max_per_token_difference=0,decision_changes=0) for a,e,k in cells]
        gate=dict(complete=True,passed=True,comparisons=rows)
        check_gate(gate)
        for key,value in (('max_score_difference',1),('max_per_token_difference',.01),('decision_changes',1)):
            bad=copy.deepcopy(gate);bad['comparisons'][0][key]=value
            with self.assertRaisesRegex(ValueError,'numerical gate'):check_gate(bad)
        with self.assertRaisesRegex(ValueError,'numerical gate'):check_gate(dict(gate,comparisons=rows[:-1]))

    def test_requires_every_model_task_pair_including_seed_repeats(self):
        ids=[(a,e,k,1234) for a in 'ABCD' for e in (1,4,16,32) for k in (1,12,50,200)]
        ids += [('B',e,k,s) for e in (1,16,32) for k in (1,12,200) for s in (5678,91011)]
        jobs=[dict(id=f'{a}_{e}_{k}_{s}_{b}',benchmark=str(b),ladder=a,E=e,K=k,seed=s) for a,e,k,s in ids for b in range(14)]
        validate_matrix(jobs)
        for bad in (jobs[:-1],jobs[:-1]+jobs[:1]):
            with self.assertRaises(ValueError):validate_matrix(bad)

    def test_hardware_batch_and_all_scientific_bindings_are_checked(self):
        original={key:{'identity':key} for key in ('checkpoint','tokenizer','model_recipe','request_manifest','token_cache')}
        original['evaluation']=dict(required_gpu_name_substring='A40',expected_world_size=1,micro_batch_size=64,sequence_length=2048)
        target=copy.deepcopy(original)
        target['evaluation'].update(required_gpu_name_substring='H100',micro_batch_size=128)
        check_spec(target,original,128)
        for key in ('checkpoint','tokenizer','model_recipe','request_manifest','token_cache'):
            bad=copy.deepcopy(target);bad[key]['identity']='changed'
            with self.assertRaisesRegex(ValueError,'scientific binding'):check_spec(bad,original,128)
        for change in (dict(required_gpu_name_substring='A40'),dict(micro_batch_size=64),dict(batching={'mode':'auto'}),dict(expected_world_size=2),dict(minimum_sequence_width=2048)):
            bad=copy.deepcopy(target);bad['evaluation'].update(change)
            with self.assertRaisesRegex(ValueError,'execution contract'):check_spec(bad,original,128)
        bad=copy.deepcopy(target);bad['evaluation']['sequence_length']=4096
        with self.assertRaisesRegex(ValueError,'scientific evaluation'):check_spec(bad,original,128)


if __name__=='__main__':unittest.main()
