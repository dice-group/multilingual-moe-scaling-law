"""Likelihood coverage, multilingual aggregation and unattended submission."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multilingual_megatron.experiments.scaling.stage1_benchmarking.extension_data import aggregate, classification_metrics, classification_prompt
from multilingual_megatron.experiments.scaling.stage1_benchmarking.extension_sources import choose_demonstrations, dominant_script
from multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign import cohort_summary
from multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_queue import submit_all
from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import document_windows, token_window_sha256, TokenCacheWriter, load_cached_sequences
from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import prepare_scored_sequence, build_score_batch


class ExtensionBenchmarksTest(unittest.TestCase):
    def test_document_windows_score_each_target_once_including_first_and_tail(self):
        for length in (1, 2047, 2048, 2049, 4096, 5000):
            original=list(range(4,4+length))
            for stride in (1,1024,2048):
                recovered=[]
                for i,(context,targets,window) in enumerate(document_windows(original,context_token_id=2,max_length=2048,stride=stride)):
                    sequence=prepare_scored_sequence(request_index=i,request_id=str(i),context_ids=context,continuation_ids=targets,max_length=2048)
                    self.assertEqual(sequence.truncated_tokens,0)
                    self.assertEqual(list(sequence.labels[sequence.continuation_start:]),targets)
                    self.assertEqual(targets,original[window['target_start']:window['target_end']])
                    recovered.extend(targets)
                self.assertEqual(recovered,original)

    def test_cache_rejects_wrong_window_offsets_and_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            context,targets,window=next(document_windows([4,5,6],context_token_id=2,max_length=4,stride=2))
            sequence=prepare_scored_sequence(request_index=0,request_id='r',context_ids=context,continuation_ids=targets,max_length=4)
            writer=TokenCacheWriter(Path(temp)/'cache');writer.append(sequence)
            reference=writer.finish(request_sha256='a'*64,tokenizer={'sha256':'b'*64,'context_token_id':2},sequence_length=4)
            spec=SimpleNamespace(request_manifest_sha256='a'*64,tokenizer_model_sha256='b'*64,sequence_length=4,context_token_id=2,tokenizer_vocab_size=100)
            row=dict(request_id='r',request_type='token_window_loglikelihood',token_window=dict(window,window_token_ids_sha256=token_window_sha256(context+targets)))
            loaded=load_cached_sequences(reference,spec,[row])
            self.assertEqual(int(build_score_batch(loaded,pad_token_id=3)['loss_mask'].sum()),3)
            for key,value in (('target_start',1),('window_token_ids_sha256','0'*64)):
                bad=copy.deepcopy(row);bad['token_window'][key]=value
                with self.assertRaises(ValueError):load_cached_sequences(reference,spec,[bad])

    def test_corpus_loss_counts_document_bytes_once_across_windows(self):
        docs=[dict(conditional=['a','b'],token_lengths=[2,1],target_tokens=3,utf8_bytes=4,unigram_log_probability_sum=-5.),
              dict(conditional=['c'],token_lengths=[2],target_tokens=2,utf8_bytes=8,unigram_log_probability_sum=-3.)]
        task=dict(benchmark='ntrex_mono',language='en',corpus_language='eng_Latn',documents=docs,primary_metric='bits_per_byte',unigram_counts_sha256='u')
        manifest=dict(requests=[dict(request_id=x) for x in 'abc'],tasks=[task])
        scores=[dict(request_id=x,loglikelihood=-v,scored_tokens=t,truncated_tokens=0) for x,v,t in [('a',2.,2),('b',3.,1),('c',7.,2)]]
        row=aggregate(manifest,scores)[0]
        self.assertEqual((row['scored_tokens'],row['utf8_bytes'],row['nll_sum']),(5,12,12.))
        self.assertAlmostEqual(row['bits_per_byte'],1/math.log(2))
        self.assertAlmostEqual(row['byte_perplexity'],2**row['bits_per_byte'])
        self.assertAlmostEqual(row['unigram_adjusted_loss'],.8)
        for invalid in (scores[:-1],scores+[scores[0]]):
            with self.assertRaises(ValueError):aggregate(manifest,invalid)
        changed=copy.deepcopy(scores);changed[1]['scored_tokens']=2
        with self.assertRaises(ValueError):aggregate(manifest,changed)

    def test_macro_f1_and_majority_are_distinct_from_accuracy_and_uniform(self):
        docs=[];scores={}
        for i,target in enumerate([0,0,0,1]):
            ids=[f'{i}a',f'{i}b',f'{i}c']
            docs.append(dict(conditional=ids,token_lengths=[1,1,1],char_lengths=[1,1,1],byte_lengths=[1,1,1],target=target,classes=3))
            scores.update({rid:dict(loglikelihood=-float(j+1),scored_tokens=1) for j,rid in enumerate(ids)})
        row=classification_metrics(docs,scores)
        self.assertEqual(row['acc'],.75)
        self.assertEqual(row['majority_baseline'],.75)
        self.assertAlmostEqual(row['chance'],1/3)
        self.assertAlmostEqual(row['macro_f1'],(6/7)/3)
        self.assertAlmostEqual(row['weighted_f1'],(6/7)*.75)

    def test_study_cohorts_collapse_regional_variants_and_keep_missing_loss_null(self):
        values=[dict(group='all',language=lang,corpus_language=corpus,primary_metric='bits_per_byte',bits_per_byte=loss,
                     margin=None,samples=2,in_K200=True,in_common12=True,trained_at_K=False)
                for lang,corpus,loss in [('por','por_Latn',1.),('por-BR','por_Latn',3.),('deu','deu_Latn',8.)]]
        study=cohort_summary({},'ntrex_mono','common12',values,.1)
        self.assertEqual((study['languages'],study['language_variants'],study['language_macro_primary']),(2,3,5.))
        self.assertIsNone(study['language_macro_accuracy'])
        empty=cohort_summary({},'ntrex_mono','trained_at_K',values,.1)
        self.assertIsNone(empty['language_macro_primary'])

    def test_demos_are_deterministic_label_diverse_and_prompt_records_english(self):
        rows=[dict(id=str(i),text='training '+str(i),target=i%6) for i in range(30)]
        demos=choose_demonstrations(rows)
        self.assertEqual(demos,choose_demonstrations(list(reversed(rows))))
        self.assertEqual(len({d['target'] for d in demos}),3)
        prompt=classification_prompt('taxi1500','test verse',demos)
        self.assertEqual(prompt.count('Verse:'),4)
        self.assertTrue(prompt.endswith('Verse: test verse\nTopic:'))
        self.assertEqual(dominant_script('Ово је српски текст')[0],'Cyrl')
        self.assertEqual(dominant_script('Ovo je srpski tekst')[0],'Latn')

    def test_one_submission_queues_gate_then_independent_cells_then_collection(self):
        module='multilingual_megatron.experiments.scaling.stage1_benchmarking.'
        jobs=[dict(id=f's{i}',phase='smoke',E=i+1,sbatch_script={'path':f'/s{i}'}) for i in range(2)]
        jobs += [dict(id=f'e{i}',phase='full',E=1,requests=10,walltime_minutes=10,sbatch_script={'path':f'/e{i}'}) for i in range(2)]
        campaign=dict(profile_payload={'extension_protocol':True,'slurm':{'bin':'/mock'}},evaluation_phase='full',jobs=jobs,
                      queue_controls={'gate':{'path':'/gate'},'monitor':{'path':'/monitor'}},collection_job={'path':'/collect'})
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{},clear=True), \
             patch(module+'candidate_campaign.validate'), patch(module+'candidate_queue.checked'), \
             patch(module+'candidate_queue.subprocess.check_output',side_effect=[str(i) for i in range(101,108)]) as call:
            path=Path(temp)/'campaign.json';path.write_text('{}')
            result=submit_all(path,campaign)
            commands=[x.args[0] for x in call.call_args_list]
            self.assertEqual(result['submitted_evaluation_jobs'],2)
            self.assertIn('--dependency=afterok:101:102',commands[2])
            for command in commands[3:5]:
                self.assertIn('--dependency=afterok:103',command)
                self.assertIn('--kill-on-invalid-dep=yes',command)
            self.assertIn('--dependency=afterany:104:105',commands[5])
            self.assertEqual(commands[6][-1],'/monitor')
            journal=json.loads((Path(temp)/'submission_full.json').read_text())
            self.assertEqual(journal['status'],'submitted')
            with self.assertRaises(ValueError):submit_all(path,campaign)

    def test_completed_startup_is_reused_with_measured_limits_without_new_gpu_checks(self):
        module='multilingual_megatron.experiments.scaling.stage1_benchmarking.'
        jobs=[dict(id='startup',phase='smoke',E=1,sbatch_script={'path':'/startup'}),
              dict(id='full',phase='full',E=1,requests=10,walltime_minutes=180,sbatch_script={'path':'/full'})]
        campaign=dict(profile_payload={'extension_protocol':True,'slurm':{'bin':'/mock'}},evaluation_phase='full',jobs=jobs,
                      queue_controls={'gate':{'path':'/gate'},'monitor':{'path':'/monitor'}},collection_job={'path':'/collect'})
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{},clear=True), \
             patch(module+'candidate_campaign.validate'), patch(module+'candidate_queue.checked'), \
             patch(module+'candidate_queue.finish_gate'), \
             patch(module+'candidate_campaign.result_for',return_value={'status':'complete'}) as completed, \
             patch(module+'candidate_queue.subprocess.check_output',side_effect=['201','202','203']) as call:
            root=Path(temp);path=root/'campaign.json';path.write_text('{}')
            (root/'submission_smoke.json').write_text(json.dumps({'jobs':[{'id':'startup','slurm_job_id':'101'}]}))
            (root/'a40_budget.json').write_text(json.dumps({'jobs':[{'id':'full','central_seconds':1200,'high_seconds':1800}]}))
            completed.return_value=None
            with self.assertRaisesRegex(ValueError,'not all completed'):submit_all(path,campaign)
            self.assertFalse((root/'submission_full.json').exists())
            call.assert_not_called()
            completed.return_value={'status':'complete'}
            result=submit_all(path,campaign)
            commands=[x.args[0] for x in call.call_args_list]
            self.assertEqual(len(commands),3)
            self.assertIn('--time=00:40:00',commands[0])
            self.assertEqual(commands[0][-1],'/full')
            self.assertFalse(any(arg.startswith('--dependency=') for arg in commands[0]))
            self.assertIn('--dependency=afterany:201',commands[1])
            self.assertIsNone(result['gate'])


if __name__=='__main__':
    unittest.main()
