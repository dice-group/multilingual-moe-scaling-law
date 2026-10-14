"""Scientific coverage, normalization and token-cache contracts."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_data import aggregate_metrics, render_document, freeze_language
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign import validate_matrix, submit
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_monitor import collection_has_coverage, inspect
from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import prepare_scored_sequence, build_score_batch
from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import TokenCacheWriter, combine_token_caches, load_cached_sequences


class FullSuiteTest(unittest.TestCase):
    def test_candidate_monitor_checks_pilot_matrix_and_cohorts_without_counting_smoke(self):
        jobs = [dict(id=f'm{i}__b', model_id=f'm{i}', benchmark='b', phase='pilot') for i in range(2)]
        inputs = {'b': dict(languages=2, documents=10)}
        metrics = dict(status='complete', incomplete_jobs=[], benchmark_summary=[
            dict(model_id=j['model_id'], benchmark='b', cohort=cohort, languages=2, samples=10)
            for j in jobs for cohort in ('all_supported_K200', 'common12', 'trained_at_K')], metrics=[
            dict(model_id=j['model_id'], benchmark='b', language=language, group=group)
            for j in jobs for language, group in (('en', 'all'), ('de', 'all'), ('en', 'category'))])
        self.assertTrue(collection_has_coverage(metrics, jobs, inputs, candidate=True))
        for field in ('benchmark_summary', 'metrics'):
            bad = copy.deepcopy(metrics)
            bad[field][0] = bad[field][1]
            self.assertFalse(collection_has_coverage(bad, jobs, inputs, candidate=True))
        bad = copy.deepcopy(metrics)
        bad['benchmark_summary'][0]['samples'] -= 1
        self.assertFalse(collection_has_coverage(bad, jobs, inputs, candidate=True))
        regular = copy.deepcopy(metrics)
        regular['metrics'] = [r for r in regular['metrics'] if r['group'] == 'all']
        regular['benchmark_summary'] = [{k: v for k, v in r.items() if k != 'cohort'}
            for r in regular['benchmark_summary'] if r['cohort'] == 'all_supported_K200']
        self.assertTrue(collection_has_coverage(regular, jobs, inputs))
        campaign = dict(schema='stage1_candidate_campaign/v1', inputs=inputs,
                        jobs=jobs + [dict(id='startup', phase='smoke')],
                        profile_payload={'slurm': {'bin': '/mock'}})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'monitoring').mkdir()
            submission = dict(jobs=[dict(id=j['id'], slurm_job_id=str(i)) for i, j in enumerate(jobs, 1)],
                              collection_job={'slurm_job_id': '3'})
            (root / 'submission_pilot.json').write_text(json.dumps(submission))
            for job in jobs:
                directory = root / 'scores' / job['id']
                directory.mkdir(parents=True)
                (directory / 'scores.json').write_text('{}')
            metrics['manifest_sha256'] = canonical_json_sha256(metrics)
            (root / 'metrics.json').write_text(json.dumps(metrics))
            accounting = '\n'.join(f'{i}|COMPLETED|60|cpu=1|0:0' for i in range(1, 4))
            with patch('multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_monitor.subprocess.check_output',
                       side_effect=[accounting, '']):
                self.assertTrue(inspect(root / 'campaign.json', campaign))
            status = json.loads((root / 'monitoring/status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['submitted_gpu_jobs'], 2)
            self.assertEqual(status['failure_count'], 0)
            campaign['evaluation_phase'] = 'full'
            for job in jobs:
                job['phase'] = 'full'
            submission['jobs'][0]['reused'] = True
            (root / 'submission_full.json').write_text(json.dumps(submission))
            with patch('multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_monitor.subprocess.check_output',
                       side_effect=[accounting, '']):
                self.assertTrue(inspect(root / 'campaign.json', campaign))
            status = json.loads((root / 'monitoring/status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual((status['submitted_gpu_jobs'], status['reused_gpu_jobs'], status['evaluation_jobs']), (1, 1, 2))

    def test_queue_matrix_has_one_independent_job_per_model_and_benchmark(self):
        inputs = {f'b{i}': {} for i in range(6)}
        jobs = [dict(id=f'm{m}__{b}', model_id=f'm{m}', benchmark=b,
                     sbatch_script={'path': f'/prepared/m{m}__{b}.sbatch'}) for m in range(82) for b in inputs]
        validate_matrix(jobs, inputs)
        for wrong in (jobs[:-1], jobs[:-1] + jobs[:1]):
            with self.assertRaises(ValueError):
                validate_matrix(wrong, inputs)
        calls = []
        def fake_slurm(command, **kwargs):
            calls.append(command)
            if command[0].endswith('/squeue'):
                return '\n'.join(str(i) for i in range(1000, 1492))
            return str(1000 + sum(c[0].endswith('/sbatch') for c in calls) - 1)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'SLURM_JOB_ID': ''}), \
             patch('multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign.validate'), \
             patch('multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign.subprocess.check_output', side_effect=fake_slurm):
            campaign = {'jobs': jobs, 'profile_payload': {'slurm': {'bin': '/mock'}}, 'collection_job': {'path': '/prepared/collect.sbatch'}}
            result = submit(Path(temp) / 'campaign.json', campaign)
            self.assertEqual(result['submitted_gpu_jobs'], 492)
            for command in calls[:492]:
                self.assertFalse(any(x.startswith('--dependency') for x in command))
            self.assertTrue(calls[-1][-2].startswith('--dependency=afterany:'))
            self.assertEqual(calls[-1][-1], '/prepared/collect.sbatch')
            self.assertEqual(len(calls[-1][-2].split(':')) - 1, 492)
            with self.assertRaisesRegex(ValueError, 'journal already exists'):
                submit(Path(temp) / 'campaign.json', campaign)
            self.assertEqual(len(calls), 494)

    def test_complete_freeze_excludes_only_approved_invalid_source_and_roundtrips_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = dict(activity_label='test', ctx_a='one', ctx_b='two', endings=['first', 'second'], label='0')
            bad = {**good, 'endings': ['', 'second']}
            source = root / 'source.json'
            source.write_text(json.dumps([good, bad]))
            item = dict(benchmark='okapi_hellaswag', language='en', source=str(source), source_sha256=sha256_file(source),
                        full_documents=2, approved_exclusions=[dict(doc_id=1, reason='empty or invalid option', document_sha256=canonical_json_sha256(bad))])
            token = dict(model='unused', sha256='a' * 64, context_token_id=2)
            with patch('sentencepiece.SentencePieceProcessor', return_value=SimpleNamespace(encode=lambda text: [4 + ord(x) % 30 for x in text])), \
                 patch('multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_data.load_xcopa', return_value=None):
                receipt = freeze_language((item, str(root / 'output'), '.', token, 2048))
            self.assertEqual((receipt['documents'], receipt['requests'], len(receipt['exclusions'])), (1, 2, 1))
            self.assertEqual(canonical_json_sha256(receipt), canonical_json_sha256(json.loads(json.dumps(receipt))))
            self.assertEqual([x['doc_id'] for x in json.loads(Path(receipt['path']).read_text())['tasks'][0]['documents']], [0])

    def test_metrics_use_original_choice_lengths_and_stable_ties(self):
        manifest = {'tasks': [{'task': 't', 'benchmark': 'b', 'language': 'l', 'documents': [
            {'doc_id': 9, 'target': 1, 'choice_lengths': [2, 8]},
            {'doc_id': 10, 'target': 0, 'choice_lengths': [2, 2]}]}],
            'requests': [dict(request_id=str(i), task='t', doc_id=doc, choice_index=choice)
                         for i, (doc, choice) in enumerate([(9, 0), (9, 1), (10, 0), (10, 1)])]}
        scores = [dict(request_id=str(i), loglikelihood=value) for i, value in enumerate([-2., -4., -1., -1.])]
        metric = aggregate_metrics(manifest, scores)[0]
        self.assertEqual(metric['samples'], 2)
        self.assertEqual(metric['acc'], .5)
        self.assertEqual(metric['acc_norm'], 1)
        for invalid in (scores[:-1], scores + scores[:1], [{**scores[0], 'loglikelihood': float('nan')}] + scores[1:]):
            with self.assertRaises(ValueError):
                aggregate_metrics(manifest, invalid)

    def test_arc_gold_uses_actual_labels_and_mmlu_rejects_missing_options(self):
        doc = {'question': ' x ', 'choices': {'text': ['red', 'blue', 'green'], 'label': ['1', '2', '3']}, 'answerKey': '2'}
        self.assertEqual(render_document('uhura_arc_easy', 'en', doc, None), ('Question: x\nAnswer:', ['red', 'blue', 'green'], 1))
        doc = dict(question='test', option_a=None, option_b='b', option_c='c', option_d='d', answer='A')
        with self.assertRaisesRegex(ValueError, 'empty or invalid option'):
            render_document('global_mmlu_full', 'en', doc, None)

    def test_token_cache_preserves_exact_windows_masks_order_and_hashes(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'request_id': str(i)} for i in range(3)]
            original = [prepare_scored_sequence(request_index=i, request_id=str(i),
                        context_ids=[4] * (i + 2), continuation_ids=[5, 6], max_length=4) for i in range(3)]
            token = {'sha256': 'a' * 64, 'context_token_id': 2}
            refs = []
            for i, parts in enumerate([original[:1], original[1:]]):
                writer = TokenCacheWriter(root / str(i))
                for sequence in parts:
                    writer.append(sequence)
                refs.append(writer.finish(request_sha256='b' * 64, tokenizer=token, sequence_length=4))
            merged = combine_token_caches(refs, root / 'merged', request_sha256='c' * 64, tokenizer=token, sequence_length=4)
            spec = SimpleNamespace(request_manifest_sha256='c' * 64, tokenizer_model_sha256='a' * 64,
                                   sequence_length=4, context_token_id=2, tokenizer_vocab_size=10)
            loaded = load_cached_sequences(merged, spec, rows)
            a, b = build_score_batch(original, pad_token_id=3), build_score_batch(loaded, pad_token_id=3)
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertEqual([s.truncated_tokens for s in original], [s.truncated_tokens for s in loaded])
            wrong = copy.copy(spec)
            wrong.request_manifest_sha256 = 'd' * 64
            with self.assertRaisesRegex(ValueError, 'request_sha256'):
                load_cached_sequences(merged, wrong, rows)
            (root / 'merged/tokens.i32').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'array drift'):
                load_cached_sequences(merged, spec, rows)


if __name__ == '__main__':
    unittest.main()
