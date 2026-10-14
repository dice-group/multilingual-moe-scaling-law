"""Frozen panel, publication gates, paired statistics and scheduler lifecycle."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multilingual_megatron.experiments.scaling.stage1_routing_repair.campaign import (
    identity, load_campaign, validate_panel, verify, write,
)
from multilingual_megatron.experiments.scaling.stage1_routing_repair.scheduler import build_graph, submit, dependency_text, executor_for, status, definite_rejection
from multilingual_megatron.experiments.scaling.stage1_routing_repair.execution import loss_parity, retry_allowed, validate_attempt
from multilingual_megatron.experiments.scaling.stage1_routing_repair.analysis import collect, entropy, mutual_information, js_matrix, bootstrap_probabilities
from multilingual_megatron.experiments.scaling.stage1_routing_repair.recovery import controls_gate, validate_control_task, require_controls, freeze_recovery_sources


def models():
    tokens = {'A':8000372736,'B':16000483328,'C':32000442368,'D':32000442368}
    cells = [(l,e,k,1234) for l in 'ABCD' for e in (4,16,32) for k in (1,12,50,200)]
    cells += [('B',e,k,s) for e in (16,32) for k in (1,12,200) for s in (5678,91011)]
    return [dict(id=f'{l}_{e}_{k}_{s}',ladder=l,experts=e,languages=k,seed=s,tokens_seen=tokens[l],is_final_readout=True) for l,e,k,s in cells]


def campaign(root):
    tasks = [dict(id=m['id'],kind='production',model=m,gpus=1,time_minutes=30) for m in models()]
    for ladder in 'BCD':
        tasks.append(dict(id=f'pilot_{ladder.lower()}',kind='pilot',model=dict(ladder=ladder),gpus=1,time_minutes=120))
    resources = dict(partition='test',gres='none',cpus_per_task=8,mem='64G',time='01:00:00',modules=[])
    return dict(root=str(root),tasks=tasks,runtime=dict(resources={'cpu':resources,'h100':resources},
        python='/fake/python',environment={},account='test',qos='test',retries={'allowed_scheduler_states':['NODE_FAIL','BOOT_FAIL','PREEMPTED']}))


def control_fixture(root):
    """Real manifest/hash fixtures for the diagnostic gate, without a GPU load."""
    tolerance = {'raw_nll_abs_tolerance':1e-6,'nll_sum_relative_tolerance':1e-7}
    routing = {'schema':'multilingual_moe_routing_diagnostics/v2','evidence_available':True,'evidence_errors':[],
        'layers':[{'layer_index':0,'topk':1,'expert_token_counts':[2,1], 'padding_expert_token_counts':[0,1],
            'all_expert_token_counts':[2,2],'forwarded_tokens':3,'dropped_assignments':0,
            'blocks':{'0':[2,1]},'trace':{'0':'fixed'}}]}
    def artifact(label, spec, nll):
        spec = copy.deepcopy(spec)
        spec.update(run_id=label,output_dir=str(root/label))
        path = root/'specs'/f'{label}.json'
        write(path,spec)
        entry = identity(path)
        row = root/label/'x.json'
        write(row,{'nll_sum':nll,'token_count':10,'backend':{'stream_tokens_including_special':3},'moe_routing':routing})
        write(root/label/'evaluation_manifest.json',{'status':'complete','spec':entry,
            'checkpoint':{'step':7},'languages':{'x':identity(row)}})
        (root/label/'.validation_complete').touch()
        return entry
    spec = {'checkpoint':{'step':7},'required_languages':['x'],'eval_micro_batch_size':4,
        'routing_diagnostics':{'enabled':False,'extra_padding_batches':0},'require_moe_routing_diagnostics':False,
        'model_recipe':{'expected_model':{'num_layers':1}}}
    old_specs = {'baseline':artifact('baseline',spec,20.)}
    spec['eval_micro_batch_size'] = 2
    spec['routing_diagnostics']['enabled'] = spec['require_moe_routing_diagnostics'] = True
    old_specs['alternate_batch'] = artifact('alternate',spec,20.01)
    task = {'id':'pilot_c','gpus':2,'languages':['x'],
        'model':{'experts':2,'checkpoint_step':7,'evaluation_manifest':str(root/'baseline/evaluation_manifest.json')}}
    old = {**task,'specs':{'1':old_specs}}
    entries = {}
    for label in ('original_disabled','half_disabled','half_enabled_repeat'):
        original = json.loads(Path(old_specs['baseline' if label=='original_disabled' else 'alternate_batch']['path']).read_text())
        if label == 'half_disabled':
            original['routing_diagnostics']['enabled'] = original['require_moe_routing_diagnostics'] = False
        entries[label] = artifact(label,original,20. if label=='original_disabled' else 20.01)
    task.update(old_specs=old_specs,specs=entries)
    write(root/'receipts/pilot_c.json',{'execution_status':'complete','task':'pilot_c'})
    c = {'runtime':{'parity':tolerance}}
    return c, {'root':str(root)}, task, {'tasks':[old],'runtime':c['runtime']}, artifact


class RoutingCampaignTests(unittest.TestCase):
    def test_recovery_controls_precede_parallel_pilots_and_respect_retry_count(self):
        c = campaign(Path('/tmp/example'))
        c['runtime'].update(gate_policy='fixed_execution_v3',recovery={'wait_for_jobs':['99']})
        nodes = {n['id']:n for n in build_graph(c)}
        self.assertEqual(len(nodes),256)
        self.assertEqual(dependency_text(nodes['controls'],{'seal':{'job_id':'100'}}),'afterok:100,afterany:99')
        for pilot in ('pilot_b','pilot_c','pilot_d'):
            self.assertEqual(nodes[f'{pilot}.a1']['dependencies'],[('afterok','controls')])
        c['runtime']['retries']['maximum_attempts'] = 1
        self.assertEqual(len(build_graph(c)),130)
        self.assertFalse(any(n['id'].endswith('.a2') for n in build_graph(c)))

    def test_recovery_resource_profile_is_used_and_gpu_mismatch_rejected(self):
        c = campaign(Path('/tmp/example'))
        c['runtime']['task_resources'] = {'D/4':{'gres':'gpu:h100:4','mem':'300G','cpus_per_task':48}}
        c['runtime']['cpu_time_limits'] = {'seal':'00:45:00'}
        c['tasks'][-1]['gpus'] = 4
        executor = executor_for(c,{'action':'run','task':'pilot_d','id':'pilot_d.a1'})
        self.assertEqual((executor.mem,executor.cpus_per_task,executor.gres),('300G',48,'gpu:h100:4'))
        cpu = executor_for(c,{'action':'seal','task':None,'id':'seal'})
        self.assertEqual(cpu.time_limit,'00:45:00')
        c['runtime']['task_resources']['D/4']['gres'] = 'gpu:h100:2'
        with self.assertRaises(ValueError):
            executor_for(c,{'action':'run','task':'pilot_d','id':'pilot_d.a1'})

    def test_fixed_execution_policy_keeps_legacy_four_variant_gate(self):
        c = {'runtime':{'gate_policy':'fixed_execution_v3','parity':{'raw_nll_abs_tolerance':1e-6,'nll_sum_relative_tolerance':1e-7}}}
        task = {'id':'pilot_c','kind':'pilot','languages':['x'],
            'specs':{'1':{name:{} for name in ('baseline','corrected','extra_padding')}}}
        row = {'nll_sum':20.,'token_count':10,'moe_routing':{'layers':[{'layer_index':0,'expert_token_counts':[1,2],'trace':{'0':'fixed'}}]}}
        target = 'multilingual_megatron.experiments.scaling.stage1_routing_repair.execution.validate_variant'
        with patch(target,return_value=({'x':row},{},{})):
            self.assertEqual(validate_attempt(c,task,1)['status'],'complete')
            del c['runtime']['gate_policy']
            with self.assertRaisesRegex(ValueError,'variant'):
                validate_attempt(c,task,1)
            c['runtime']['gate_policy'] = 'fixed_execution_v3'
            changed = copy.deepcopy(row)
            changed['nll_sum'] += .001
            with patch(target,side_effect=[({'x':row},{},{}),({'x':row},{},{}),({'x':changed},{},{})]):
                with self.assertRaisesRegex(ValueError,'loss parity'):
                    validate_attempt(c,task,1)
            changed = copy.deepcopy(row)
            changed['moe_routing']['layers'][0]['trace']['0'] = 'changed'
            with patch(target,side_effect=[({'x':row},{},{}),({'x':row},{},{}),({'x':changed},{},{})]):
                with self.assertRaisesRegex(ValueError,'real routing changed'):
                    validate_attempt(c,task,1)

    def test_control_reader_recomputes_fixed_batch_parity_and_rejects_changed_routing(self):
        with tempfile.TemporaryDirectory() as temp:
            c,d,t,b,artifact = control_fixture(Path(temp))
            report, evidence = validate_control_task(c,d,t,b)
            self.assertTrue(report['batch_sensitivity']['x']['exceeds_original_tolerances'])
            self.assertTrue(evidence)
            # A success receipt cannot conceal a changed diagnostics-off loss.
            spec = json.loads(Path(t['specs']['half_disabled']['path']).read_text())
            t['specs']['half_disabled'] = artifact('half_disabled',spec,20.02)
            with self.assertRaisesRegex(ValueError,'loss parity'):
                validate_control_task(c,d,t,b)
            t['specs']['half_disabled'] = artifact('half_disabled',spec,20.01)
            row_path = Path(temp)/'half_enabled_repeat/x.json'
            row = json.loads(row_path.read_text())
            row['moe_routing']['layers'][0]['trace']['0'] = 'changed'
            write(row_path,row)
            manifest_path = row_path.parent/'evaluation_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['languages']['x'] = identity(row_path)
            write(manifest_path,manifest)
            with self.assertRaisesRegex(ValueError,'routing repeat changed'):
                validate_control_task(c,d,t,b)

    def test_control_missing_or_stale_seal_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            c = {'root':temp}
            with self.assertRaises(OSError):
                require_controls(c)
            root = Path(temp)
            write(root/'controls_complete.json',{'status':'complete','campaign_sha256':'old'})
            (root/'controls_complete.sha256').write_text(identity(root/'controls_complete.json')['sha256'])
            write(root/'campaign.json',{})
            with self.assertRaisesRegex(ValueError,'have not passed'):
                require_controls(c)

    def test_cpu_controls_gate_cannot_publish_when_d_receipt_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c,d,t,b,_ = control_fixture(root)
            c.update(root=temp,source={'files':[]})
            c['runtime'].update(environment={},python='/python',bridge_commit='frozen')
            b.update(root=str(root/'old'),source={'files':[]})
            b['tasks'].append({**b['tasks'][0],'id':'pilot_d'})
            write(root/'base.json',b)
            write(root/'runner.py',{})
            write(root/'old_input_seal.json',{})
            d.update(tasks=[t,{**t,'id':'pilot_d'}],runtime=copy.deepcopy(c['runtime']),
                source_campaign=identity(root/'base.json'),runner=identity(root/'runner.py'),
                input_seal=identity(root/'old_input_seal.json'))
            write(root/'diagnosis.json',d)
            c['runtime']['recovery'] = {'base_campaign':identity(root/'base.json'),
                'control_manifest':identity(root/'diagnosis.json'),'collector_test':identity(root/'runner.py')}
            write(root/'campaign.json',c)
            write(root/'input_seal.json',{'status':'complete','campaign_sha256':identity(root/'campaign.json')['sha256']})
            with patch('subprocess.run') as replay:
                with self.assertRaises(FileNotFoundError):
                    controls_gate(c)
                replay.assert_not_called()
            self.assertFalse((root/'controls_complete.json').exists())

    def test_source_recovery_keeps_frozen_execution_despite_workspace_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = 'src/engine.py'
            write(root/'old/source'/name,{'engine':'evaluated'})
            write(root/'workspace'/name,{'engine':'unrelated change'})
            write(root/'old/campaign.json',{'root':str(root/'old'),'source':{'files':[identity(root/'old/source'/name)]}})
            recovery = {'base_campaign':identity(root/'old/campaign.json'),'source_overlays':[]}
            freeze_recovery_sources(root/'workspace',root/'new/source',recovery)
            self.assertEqual((root/'new/source'/name).read_bytes(),(root/'old/source'/name).read_bytes())
            recovery['source_overlays'] = [name]
            with self.assertRaisesRegex(ValueError,'cannot change evaluation code'):
                freeze_recovery_sources(root/'workspace',root/'other/source',recovery)

    def test_collection_reports_corrupt_receipts_and_stale_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = {'id':'x','kind':'production','languages':['eng']}
            write(root/'results/x.json',{'status':'complete','task_id':'x','receipt':{'path':str(root/'absent'),'sha256':'0'*64}})
            (root/'scheduler_status.psv').write_text('99|job|COMPLETED|60|gres/gpu=2|0:0\nmalformed\n')
            write(root/'scheduler_status.json',{'status':'query_failed','error':'sacct unavailable'})
            result = collect({'root':str(root),'tasks':[task],'runtime':{}})
            self.assertEqual(result['status'],'incomplete')
            report = json.loads((root/'completeness.json').read_text())
            self.assertEqual(report['models'][0]['status'],'invalid')
            accounting = json.loads((root/'runtime_accounting.json').read_text())
            self.assertTrue(accounting['snapshot_is_stale_or_unverified'])
            self.assertFalse(accounting['scheduler_accounting_available'])
            self.assertTrue((root/'analysis/README.md').exists())

    def test_scheduler_uses_profile_binary_and_records_query_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = campaign(root)
            c['runtime']['slurm_commands'] = {'sacct':'/site/slurm/sacct'}
            write(root/'submission.json',{'jobs':{'seal':{'job_id':'99'}}})
            with patch('subprocess.check_output',return_value='99|job|COMPLETED|10||0:0\n') as query:
                status(c)
                self.assertEqual(query.call_args.args[0][0],'/site/slurm/sacct')
            with patch('subprocess.check_output',side_effect=FileNotFoundError('missing')):
                with self.assertRaises(FileNotFoundError):
                    status(c)
            self.assertEqual(json.loads((root/'scheduler_status.json').read_text())['status'],'query_failed')
            self.assertTrue((root/'scheduler_status.psv').exists())

    def test_submission_rejection_is_distinct_from_ambiguous_timeout(self):
        self.assertTrue(definite_rejection(subprocess.CalledProcessError(1,['sbatch'],stderr='Invalid account')))
        self.assertFalse(definite_rejection(subprocess.CalledProcessError(1,['sbatch'],stderr='Socket timed out on send/recv operation')))

    def test_exact_final_panel(self):
        panel = models()
        validate_panel(panel)
        self.assertEqual(sum(m['languages'] for m in panel),4008)
        for altered in (panel[:-1],panel+[panel[0]]):
            with self.assertRaises(ValueError):
                validate_panel(altered)
        panel[0]['tokens_seen'] -= 1
        with self.assertRaises(ValueError):
            validate_panel(panel)

    def test_dag_retries_and_failure_collection(self):
        graph = build_graph(campaign(Path('/tmp/example')))
        self.assertEqual(len(graph),255)
        by_id = {n['id']:n for n in graph}
        a3 = by_id['pilot_b.a3']
        self.assertEqual(a3['dependencies'],[('afterok','seal'),('afternotok','pilot_b.a1'),('afternotok','pilot_b.a2')])
        for pilot in ('pilot_b','pilot_c','pilot_d'):
            self.assertEqual(by_id[f'{pilot}.a1']['dependencies'],[('afterok','seal')])
            self.assertEqual(by_id[f'{pilot}.a3']['dependencies'],[
                ('afterok','seal'),('afternotok',f'{pilot}.a1'),('afternotok',f'{pilot}.a2')])
        self.assertEqual(by_id['pilots']['dependencies'],[
            ('afterany','pilot_b.complete'),('afterany','pilot_c.complete'),('afterany','pilot_d.complete')])
        self.assertTrue(all(kind=='afterany' for kind,_ in by_id['collect']['dependencies']))
        self.assertEqual(len(by_id['collect']['dependencies']),61)
        registered = set()
        for node in graph:
            self.assertTrue(all(key in registered for _,key in node['dependencies']))
            registered.add(node['id'])

    def test_success_cancels_both_later_attempts(self):
        graph = {n['id']:n for n in build_graph(campaign(Path('/tmp/example')))}
        def eligible(key,states):
            return all((states[p]=='COMPLETED' if kind=='afterok' else states[p]!='COMPLETED' if kind=='afternotok' else True) for kind,p in graph[key]['dependencies'])
        states = {'seal':'COMPLETED','pilot_b.a1':'COMPLETED','pilot_b.a2':'CANCELLED'}
        self.assertFalse(eligible('pilot_b.a2',states))
        self.assertFalse(eligible('pilot_b.a3',states))
        states.update({'pilot_b.a1':'NODE_FAIL','pilot_b.a2':'COMPLETED'})
        self.assertFalse(eligible('pilot_b.a3',states))
        states['pilot_b.a2']='NODE_FAIL'
        self.assertTrue(eligible('pilot_b.a3',states))

    def test_submission_is_idempotent_and_root_released_last(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = campaign(root)
            write(root/'campaign.json',c)
            calls = []
            def fake_submit(executor,cmd,env,dry_run,log_path,**kwargs):
                calls.append((cmd,kwargs,executor.extra_sbatch_flags))
                return str(1000+len(calls))
            with patch.dict('os.environ',{},clear=True), patch('multilingual_megatron.pipeline.executor.SlurmExecutor.submit',new=fake_submit), patch('subprocess.check_output',return_value='Reason=JobHeldUser'), patch('subprocess.run') as release:
                first = submit(c)
                second = submit(c)
            self.assertEqual(len(calls),255)
            self.assertEqual(first,second)
            release.assert_called_once()
            self.assertIn('--hold',calls[0][2])
            self.assertTrue(all('--no-requeue' in flags and '--kill-on-invalid-dep=yes' in flags for _,_,flags in calls))
            self.assertTrue(first['released'])

    def test_scientific_failure_and_unknown_state_do_not_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = campaign(root)
            write(root/'submission.json',{'jobs':{'x.a1':{'job_id':'99'}}})
            path = root/'attempts/x/1/receipt.json'
            for status in ('invalid','complete','skipped'):
                write(path,{'status':status})
                self.assertFalse(retry_allowed(c,'x',2))
            write(path,{'status':'runtime_failure'})
            with patch('multilingual_megatron.experiments.scaling.stage1_routing_repair.execution.scheduler_state',return_value='FAILED'):
                self.assertFalse(retry_allowed(c,'x',2))
            with patch('multilingual_megatron.experiments.scaling.stage1_routing_repair.execution.scheduler_state',return_value='NODE_FAIL'):
                self.assertTrue(retry_allowed(c,'x',2))

    def test_loss_parity_denominators_and_tolerances(self):
        tolerance = {'raw_nll_abs_tolerance':1e-6,'nll_sum_relative_tolerance':1e-7}
        reference = {'nll_sum':100.,'token_count':50}
        loss_parity(reference,reference,tolerance)
        for row in ({'nll_sum':100.,'token_count':49},{'nll_sum':101.,'token_count':50},{'nll_sum':float('nan'),'token_count':50}):
            with self.assertRaises(ValueError):
                loss_parity(row,reference,tolerance)

    def test_equal_language_mi_and_js(self):
        separated = np.eye(2)
        self.assertAlmostEqual(float(mutual_information(separated)),np.log(2))
        self.assertEqual(float(mutual_information(np.array([[.9,.1],[.9,.1]]))),0.)
        matrix = js_matrix(separated)
        np.testing.assert_allclose(matrix,matrix.T)
        np.testing.assert_allclose(np.diag(matrix),0.)
        self.assertAlmostEqual(matrix[0,1],np.log(2))

    def test_bootstrap_pairing_and_padding_invariance(self):
        a = {'blocks':{'0':[8,2],'1':[2,8]}}
        b = copy.deepcopy(a)
        b['blocks']['2']=[0,0]
        p = bootstrap_probabilities(a,'eng_Latn',1000,20260907)
        q = bootstrap_probabilities(b,'eng_Latn',1000,20260907)
        np.testing.assert_array_equal(p,q)
        self.assertEqual(p.shape,(1000,2))

    def test_sealed_artifact_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'artifact'
            path.write_text('changed')
            with self.assertRaises(ValueError):
                verify({'path':str(path),'sha256':'0'*64})
