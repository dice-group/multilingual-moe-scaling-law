from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.execution import (
    ExecutionError, build_manifest, chunk_path, execute_task, read_json, reduce,
    render_slurm, run_serial, source_sha256, validate_manifest, write_json_once, write_manifest,
)
from multilingual_megatron.experiments.scaling.stage1_fit.data import canonical_json_sha256
from multilingual_megatron.experiments.scaling.stage1_fit.fit import (DirectCheckResult, DirectFitResult,
    EquivalenceResult, SharedParameters)
from multilingual_megatron.experiments.scaling.stage1_fit.report import execution_summary

class Stage1FitExecutionSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        clean = patch('multilingual_megatron.experiments.scaling.stage1_fit.execution._repository_provenance',
            return_value=('f' * 40, False))
        committed = patch(
            'multilingual_megatron.experiments.scaling.stage1_fit.execution.source_sha256_at_commit',
            return_value=source_sha256(),
        )
        clean.start(); committed.start(); self.addCleanup(clean.stop); self.addCleanup(committed.stop)

    def test_dirty_checkout_is_recorded_but_only_fitter_source_drift_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows = root / 'fit_rows.parquet'
            pd.DataFrame({'sentinel': [1]}).to_parquet(fit_rows, index=False)
            clean_manifest = build_manifest(output_root=root / 'clean', fit_rows_path=fit_rows,
                fit_rows_content_sha256='a' * 64, phase='definitive', model='M0',
                fold_id='synthetic', objective='ols', chunk_size=81, membership={})
            dirty_provenance = {**clean_manifest['provenance'], 'git_dirty': True}
            with patch('multilingual_megatron.experiments.scaling.stage1_fit.execution.runtime_provenance',
                    return_value=dirty_provenance):
                dirty_manifest = build_manifest(output_root=root / 'output', fit_rows_path=fit_rows,
                    fit_rows_content_sha256='a' * 64, phase='definitive', model='M0',
                    fold_id='synthetic', objective='ols', chunk_size=81, membership={})
            self.assertTrue(dirty_manifest['provenance']['git_dirty'])
            validate_manifest(dirty_manifest, verify_files=False,
                require_current_provenance=False)
            advanced_head = {
                **dirty_provenance,
                'git_commit': 'e' * 40,
                'git_dirty': False,
            }
            with patch(
                'multilingual_megatron.experiments.scaling.stage1_fit.execution.runtime_provenance',
                return_value=advanced_head,
            ):
                validate_manifest(dirty_manifest, verify_files=False)
            with patch(
                'multilingual_megatron.experiments.scaling.stage1_fit.execution.runtime_provenance',
                return_value=dirty_provenance,
            ), patch(
                'multilingual_megatron.experiments.scaling.stage1_fit.execution.source_sha256_at_commit',
                return_value='0' * 64,
            ), self.assertRaisesRegex(ExecutionError, 'fitter source bytes'):
                build_manifest(output_root=root / 'drifted', fit_rows_path=fit_rows,
                    fit_rows_content_sha256='a' * 64, phase='definitive', model='M0',
                    fold_id='synthetic', objective='ols', chunk_size=81, membership={})
            self.assertFalse((root / 'output').exists())

    def test_local_and_array_resume_reduce_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fit_rows=root/'fit_rows.parquet'
            pd.DataFrame({'sentinel':[1]}).to_parquet(fit_rows,index=False)
            manifest=build_manifest(output_root=root/'output',fit_rows_path=fit_rows,
                fit_rows_content_sha256='a'*64,phase='candidate_search',model='M0',
                fold_id='synthetic',objective='ols',chunk_size=2,start_total=5,
                review_only=True, blas_threads=1,partition='cpu',concurrency=2)
            manifest_path=write_manifest(manifest)
            self.assertEqual(set(manifest['provenance']), {
                'source_sha256', 'python', 'numpy', 'scipy', 'pandas', 'pyarrow', 'platform', 'float_dtype',
                'git_commit', 'git_dirty'})
            tampered=dict(manifest); tampered_provenance=dict(manifest['provenance']); tampered_provenance['scipy']='changed'
            tampered['provenance']=tampered_provenance; tampered.pop('manifest_sha256')
            tampered['manifest_sha256']=canonical_json_sha256(tampered)
            with self.assertRaisesRegex(ExecutionError, 'numerical environment'):
                validate_manifest(tampered, verify_files=False)
            def runner(_manifest, index):
                return {'start_index':index,'start_id':f'grid:{index:06d}','success':True,
                        'status':0,'message':'ok','objective':float(abs(index-2)),
                        'parameters':{'x':float(index)}}
            for task in reversed(manifest['tasks']):
                self.assertEqual(execute_task(manifest,task['task_index'],runner=runner)['status'],'written')
            self.assertEqual(execute_task(manifest,0,runner=runner)['status'],'skipped_hash_valid')
            array_result=reduce(manifest)
            serial_result=run_serial(manifest,runner=runner)
            self.assertEqual(array_result,serial_result)
            self.assertEqual(array_result['selected']['start_index'],2)
            summary=execution_summary(manifest_path)
            self.assertEqual((summary['attempted'],summary['selected_start']),(5,'grid:000002'))
            script,launch=render_slurm(manifest_path,manifest)
            self.assertIn('SLURM_ARRAY_TASK_ID',script)
            self.assertIn('OPENBLAS_NUM_THREADS=1',script)
            attempt_root = root/'output'/'logs'/'attempts'/manifest['manifest_sha256'][:20]
            self.assertIn(f'#SBATCH --output={attempt_root}/slurm-%A_%a.out',script)
            self.assertTrue(attempt_root.is_dir())
            self.assertTrue(launch.startswith('sbatch '))
            missing=chunk_path(manifest,1); missing.unlink()
            with self.assertRaisesRegex(ExecutionError,'missing chunk'):
                reduce(manifest)
            missing.write_text('{}\n')
            with self.assertRaisesRegex(ExecutionError,'canonical|chunk'):
                reduce(manifest)
            write_json_once(root/'output'/'receipts'/'finalization.json', {'finished': True})
            with self.assertRaisesRegex(ExecutionError, 'finalized output root'):
                execute_task(manifest,0,runner=runner)

    def test_direct_fit_is_persisted_and_review_profile_fits_time_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fit_rows=root/'fit_rows.parquet'
            pd.DataFrame({'sentinel':[1]}).to_parquet(fit_rows,index=False)
            manifest=build_manifest(output_root=root/'output',fit_rows_path=fit_rows,
                fit_rows_content_sha256='a'*64,phase='direct',model='M0',fold_id='synthetic',
                objective='ols',chunk_size=1,start_total=1,action='direct',review_only=True,
                blas_threads=1,partition='cpu',concurrency=1)
            shared=SharedParameters(.2,0.0,.3,0.0)
            coefficient=SimpleNamespace(language='eng',L_inf=1.0,A=.1,B=.2,
                floor_fraction=.5,floor_saturated=False)
            projected=SimpleNamespace(start_index=0,start_id='grid:000000',success=True,
                status=0,message='ok',objective=.1,coordinates=np.asarray([.2,.3,0.0,0.0]),
                shared_parameters=shared,prediction=np.asarray([1.2]),
                profile=SimpleNamespace(coefficients=(coefficient,)),iterations=2,function_evaluations=3,
                boundary_contacts=())
            direct=DirectFitResult(success=True,status=0,message='ok',objective=.1,
                prediction=np.asarray([1.2]),outer_coordinates=np.asarray([.2,.3,0.0,0.0]),
                shared_parameters=shared,language_coefficients={'eng': {'L_inf':1.0,'A':.1,'B':.2,'floor_fraction':.5}},
                iterations=4,function_evaluations=5)
            check=DirectCheckResult(direct=direct,
                equivalence=EquivalenceResult(True,0.0,0.0),retried=False,resolved=True)
            with patch('multilingual_megatron.experiments.scaling.stage1_fit.execution.load_problem',return_value=object()), \
                 patch('multilingual_megatron.experiments.scaling.stage1_fit.execution.run_projected_start',return_value=projected), \
                 patch('multilingual_megatron.experiments.scaling.stage1_fit.execution.direct_joint_check',return_value=check):
                execute_task(manifest,0)
            record=read_json(chunk_path(manifest,0))['records'][0]
            self.assertTrue({'objective','outer_coordinates','shared_parameters','language_coefficients','predictions'}
                <= set(record['direct_fit']))
            self.assertEqual(record['direct_fit']['predictions'],[1.2])

if __name__=='__main__': unittest.main()
