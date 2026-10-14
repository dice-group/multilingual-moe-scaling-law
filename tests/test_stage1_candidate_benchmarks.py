"""Scientific boundaries of the supplementary minimal-pair/PMI pilot."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_data import aggregate, freeze_language, render
from multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign import cohort_summary, estimate_budget, select_models, submit, validate_matrix
from multilingual_megatron.experiments.scaling.stage1_benchmarking.full_suite_campaign import write_job


class CandidateBenchmarksTest(unittest.TestCase):
    def test_benchmark_without_k1_language_has_missing_not_zero_trained_average(self):
        values = [dict(group='all', primary_metric='acc', acc=.6, margin=.1, samples=100,
                       in_common12=True, trained_at_K=False)]
        identity = dict(model_id='english_only', K=1)
        empty = cohort_summary(identity, 'xcomps', 'trained_at_K', values, .1)
        self.assertEqual((empty['languages'], empty['samples']), (0, 0))
        self.assertIsNone(empty['language_macro_accuracy'])
        self.assertIsNone(empty['language_macro_margin'])
        self.assertEqual(empty['cohort_status'], 'not_applicable_no_languages')
        self.assertEqual(empty['primary_metric'], 'acc')
        for cohort in ('common12', 'all_supported_K200'):
            row = cohort_summary(identity, 'xcomps', cohort, values, .1)
            self.assertEqual((row['languages'], row['samples'], row['language_macro_accuracy']), (1, 100, .6))
            self.assertEqual(row['cohort_status'], 'available')

    def test_full_matrix_retains_all_82_final_models_including_seed_repeats(self):
        identities = [(a, e, k, 1234) for a in 'ABCD' for e in (1, 4, 16, 32) for k in (1, 12, 50, 200)]
        identities += [('B', e, k, s) for e in (1, 16, 32) for k in (1, 12, 200) for s in (5678, 91011)]
        inputs = dict.fromkeys(('xcopa', 'b1', 'b2', 'b3', 'b4', 'b5'), {})
        source = dict(inputs=inputs, jobs=[dict(id=f'{a}_{e}_{k}_{s}__{b}', model_id=f'{a}_{e}_{k}_{s}',
            ladder=a, E=e, K=k, seed=s, benchmark=b) for a, e, k, s in identities for b in inputs])
        models = select_models(source, dict(all_final=True, expected_models=82))
        self.assertEqual(len(models), 82)
        jobs = [dict(m, id=m['model_id']+'__'+b, benchmark=b, phase='full')
                for m in models for b in ('multiblimp', 'x_csqa', 'xcomps')]
        jobs += [dict(m, id=m['model_id']+'__smoke', benchmark='smoke', phase='smoke')
                 for m in models if m['K'] == 200 and m['seed'] == 1234]
        campaign = dict(evaluation_phase='full', jobs=jobs)
        evaluation, smoke = validate_matrix(campaign, models)
        self.assertEqual((len(evaluation), len(smoke)), (246, 16))
        for bad_jobs in (jobs[:-1], jobs[:-1]+jobs[:1], jobs[:1]+jobs[2:]):
            with self.assertRaises(ValueError):
                validate_matrix(dict(evaluation_phase='full', jobs=bad_jobs), models)
        wrong = copy.deepcopy(campaign)
        wrong['jobs'][0]['seed'] = 91011
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            validate_matrix(wrong, models)

    def test_expansion_reuses_completed_jobs_and_does_not_submit_them_again(self):
        jobs = [dict(id='existing', phase='full', reused=True, original_slurm_job_id='77',
                     origin_campaign={'path': '/old/campaign.json'}, walltime_minutes=20, requests=10, E=1),
                dict(id='new', phase='full', walltime_minutes=20, requests=10, E=1,
                     sbatch_script={'path': '/new/job.sbatch'})]
        campaign = dict(evaluation_phase='full', jobs=jobs, profile_payload={'slurm': {'bin': '/mock'}},
                        collection_job={'path': '/new/collect.sbatch'})
        module = 'multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign'
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True), \
             patch(module+'.validate'), patch(module+'.smoke_gate', return_value={'passed': True}), \
             patch(module+'.estimate_budget', return_value={'jobs': []}), \
             patch(module+'.result_for', return_value={'status': 'complete'}), \
             patch(module+'.subprocess.check_output', side_effect=['100', '100\n', '101']) as call:
            (Path(temp)/'campaign.json').write_text(json.dumps(campaign))
            result = submit(Path(temp)/'campaign.json', campaign, 'full')
            self.assertEqual((result['submitted_jobs'], result['reused_jobs']), (1, 1))
            self.assertEqual(result['job_ids'], ['100'])
            self.assertEqual(call.call_count, 3)
            self.assertEqual(call.call_args_list[0].args[0][-1], '/new/job.sbatch')
            self.assertIn('--dependency=afterany:100', call.call_args_list[-1].args[0])
            journal = json.loads((Path(temp)/'submission_full.json').read_text())
            self.assertEqual([r['slurm_job_id'] for r in journal['jobs']], ['77', '100'])
            with self.assertRaisesRegex(ValueError, 'already has a submission journal'):
                submit(Path(temp)/'campaign.json', campaign, 'full')
            self.assertEqual(call.call_count, 3)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True), \
             patch(module+'.validate'), patch(module+'.smoke_gate', return_value={'passed': True}), \
             patch(module+'.estimate_budget', return_value={'jobs': []}), \
             patch(module+'.result_for', return_value=None), patch(module+'.subprocess.check_output') as call:
            (Path(temp)/'campaign.json').write_text(json.dumps(campaign))
            with self.assertRaisesRegex(ValueError, 'not yet complete'):
                submit(Path(temp)/'campaign.json', campaign, 'full')
            self.assertFalse((Path(temp)/'submission_full.json').exists())
            call.assert_not_called()

    def test_budget_weights_all_request_lengths_and_excludes_reused_cost(self):
        jobs = [dict(id='startup', phase='smoke', ladder='D', E=32)]
        jobs += [dict(id=name, phase='full', ladder='D', E=32, benchmark='xcomps', reused=reused)
                 for name, reused in (('old', True), ('new', False))]
        campaign = dict(evaluation_phase='full', jobs=jobs, data_manifests={'full': {}})
        gate = dict(results=[dict(id='startup', timing={'forward_batches': [
            dict(width=8, requests=8, seconds=value) for value in (999., 999., .08)]})])
        module = 'multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign'
        with patch(module+'.checked'), patch(module+'.read', return_value={'languages': [
                dict(benchmark='xcomps', width_histogram={'8':100})]}):
            budget = estimate_budget(campaign, gate)
        self.assertEqual((budget['new_jobs'], budget['reused_jobs']), (1, 1))
        self.assertAlmostEqual(budget['new_gpu_hours']['central'], 121/3600)

    def test_minimal_pair_ties_cannot_become_perfect_accuracy(self):
        manifest = {"requests": [{"request_id": "a"}, {"request_id": "b"}], "tasks": [{
            "benchmark": "multiblimp", "language": "eng", "corpus_language": "eng_Latn", "primary_metric": "acc",
            "documents": [{"conditional": ["a", "b"], "unconditional": [], "token_lengths": [3, 3],
                           "byte_lengths": [8, 8], "target": 0, "group": "agreement"}]}]}
        scores = [dict(request_id=x, loglikelihood=-4., scored_tokens=3) for x in ("a", "b")]
        metric = aggregate(manifest, scores)[0]
        self.assertEqual((metric["acc"], metric["tie"], metric["margin"]), (0., 1., 0.))
        scores[1]["loglikelihood"] = -5.
        self.assertEqual(aggregate(manifest, scores)[0]["acc"], 1.)
        with self.assertRaisesRegex(ValueError, "token counts"):
            aggregate(manifest, [{**scores[0], "scored_tokens": 2}, scores[1]])
        for bad in (scores[:1], scores + scores[:1], [{**scores[0], "loglikelihood": float("nan")}, scores[1]]):
            with self.assertRaises(ValueError):
                aggregate(manifest, bad)

    def test_pmi_subtracts_matched_unconditional_option_scores(self):
        conditional, unconditional = list("abcde"), list("vwxyz")
        manifest = {"requests": [{"request_id": x} for x in conditional + unconditional], "tasks": [{
            "benchmark": "x_csqa", "language": "en", "corpus_language": "eng_Latn", "primary_metric": "acc_pmi",
            "documents": [{"conditional": conditional, "unconditional": unconditional,
                           "token_lengths": [1]*5, "byte_lengths": [2]*5, "target": 1, "group": "all"}]}]}
        values = [-1., -2., -4., -5., -6., -1., -8., -4., -5., -6.]
        scores = [dict(request_id=x, loglikelihood=v, scored_tokens=1) for x, v in zip(conditional + unconditional, values)]
        metric = aggregate(manifest, scores)[0]
        self.assertEqual((metric["acc"], metric["acc_pmi"], metric["margin_pmi"]), (0., 1., 6.))

    def test_frozen_minimal_pairs_keep_exact_text_and_all_scored_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "xcomps.json"
            source.write_text(json.dumps({"pair-7": dict(acceptable_sent="猫は眠る。", unacceptable_sent="石は眠る。",
                acceptable_concept="猫", property="眠る", negative_sample_type="random")}))
            item = dict(path=str(source), sha256=sha256_file(source), benchmark="xcomps", language="ja", corpus_language="jpn_Jpan")
            tokenizer = dict(model="unused", context_token_id=2, pad_token_id=3, sha256="a"*64)
            with patch("sentencepiece.SentencePieceProcessor", return_value=SimpleNamespace(encode=lambda x: [4+ord(c)%20 for c in x])):
                receipt = freeze_language((item, str(root / "out"), tokenizer, 2048))
            full = json.loads(Path(receipt["full"]["path"]).read_text())
            self.assertIsNone(full["sampling"])
            self.assertEqual([r["context"] for r in full["requests"]], ["", ""])
            self.assertEqual(full["requests"][0]["continuation"], "猫は眠る。")
            self.assertEqual(full["tasks"][0]["documents"][0]["source_id"], "pair-7")
            self.assertEqual(full["tasks"][0]["documents"][0]["token_lengths"], [5, 5])
            self.assertEqual(receipt["full"]["requests"], 2)
            self.assertEqual(canonical_json_sha256(receipt), canonical_json_sha256(json.loads(json.dumps(receipt))))

    def test_missing_labels_and_identical_pairs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            render("multiblimp", dict(sen="same", wrong_sen="same"))
        with self.assertRaisesRegex(ValueError, "answer label"):
            render("x_csqa", dict(question=dict(stem="question", choices=dict(label=list("ABCDE"), text=list("abcde"))), answerKey=""))

    def test_pilot_submission_requires_completed_smoke_before_sbatch(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True), \
             patch("multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign.validate"), \
             patch("multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign.smoke_gate", side_effect=ValueError("smoke missing")), \
             patch("subprocess.check_output") as call:
            with self.assertRaisesRegex(ValueError, "smoke missing"):
                submit(Path(temp) / "campaign.json", {}, "pilot")
            call.assert_not_called()

    def test_shared_launcher_routes_to_candidate_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = dict(id="test", walltime_minutes=20, memory="8G")
            profile = dict(module_init="/modules.sh", modules=["cuda"], environment={}, python="/env/bin/python",
                slurm=dict(account="test", partition="gpu_a40", gpu_type="a40", cpus_per_task=2, bin="/slurm"))
            run, sbatch = write_job(root, job, profile, root, module="example.candidate", job_prefix="candidate-")
            self.assertIn("example.candidate run", Path(run["path"]).read_text())
            self.assertIn("--gres=gpu:a40:1", Path(sbatch["path"]).read_text())


if __name__ == "__main__":
    unittest.main()
