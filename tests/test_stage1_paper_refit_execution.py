"""Numerical delegation, snapshot isolation, Slurm safety, and scoring fixtures."""
from copy import deepcopy
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import execution as io, fit
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import (
    inputs, backend, runner, report as reporting)
from tests.test_stage1_paper_refit import fixture


def runtime_profile():
    return dict(inputs="bundle/manifest.json", output_root="run", python_executable=sys.executable,
        account="fixture", partition="cpu", memory="2G", grid_chunk_size=12, starts_per_shard=4,
        max_array_size=20000, max_concurrent_cores=256,
        concurrency=dict(grid=128, native=96, repetition=29, shared=3),
        minutes=dict(grid=180, native=120, repetition=240, shared=720, certify=720, report=60),
        max_reserved_core_hours=60000)


class PaperRefitExecutionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.profile, _, _ = fixture(self.root)
        inputs.prepare(self.profile)
        self.runtime = self.root / "runtime.json"
        io.write_json_once(self.runtime, runtime_profile())
        self.plan_path = self.root / "run/plan.json"

    def prepare(self, models=("RN14-stored-reuse",)):
        runner.prepare(self.runtime, models=models)
        return runner.load(self.plan_path)

    def test_original_primary_grid_counts_and_global_parallelism(self):
        contracts = {f"{f}--{m}": dict(fit=dict(start_count=fit.start_count(m)))
                     for f in inputs.PRIMARY for m in ("J0", "M2")}
        contracts.update({f"{f}--M5-ND": dict(starts=[{}]*48) for f in inputs.PRIMARY})
        tasks = runner.tasks_for(inputs.MODELS, {f: ["x", "y"] for f in inputs.FOLDS}, contracts, runtime_profile())
        for fold in inputs.PRIMARY:
            for model, count in (("J0", 729), ("M2", 59049), ("M5-ND", 48), ("RN14-stored-reuse", 48)):
                indices = [i for t in tasks if t["stage"] == "fit" and t["model"] == model and t["fold"] == fold
                           for i in t["start_indices"]]
                self.assertEqual(indices, list(range(count)))
        arrays = runner.arrays_for(tasks, runtime_profile())
        self.assertLessEqual(sum(a["concurrency"] for a in arrays if a["name"] != "certify"), 256)
        self.assertEqual(sum(t["stage"] == "certify" for t in tasks), 32)
        with self.assertRaisesRegex(ValueError, "array limit"):
            runner.arrays_for(tasks, runtime_profile() | {"max_array_size": 10})

    def test_snapshot_scripts_dry_run_and_source_tampering(self):
        plan = self.prepare()
        source_names = {Path(entry["path"]).name for entry in plan["source"]}
        self.assertIn("inputs.py", source_names)
        self.assertIn("paper_refit_workflow/runner.py", {entry["path"].split("source/paper_fit_snapshot/")[-1] for entry in plan["source"]})
        self.assertIn("shared_numerics.py", source_names)
        self.assertFalse({"repetition_campaign.py", "law_search.py", "m6_nested.py"} & source_names)
        for name in ("repetition", "certify", "report"):
            script = runner.script(self.plan_path, plan, name)
            self.assertIn("--cpus-per-task=1", script)
            self.assertNotIn("--gres", script)
            self.assertIn("paper_fit_snapshot.paper_refit", script)
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as call:
            self.assertFalse(runner.submit(self.plan_path, dry_run=True)["submitted"])
            self.assertTrue(all(c.args[0][0] == "bash" for c in call.call_args_list))
        self.assertFalse((self.root / "run/submission").exists())
        bound = self.root / "run" / plan["source"][0]["path"]
        bound.write_bytes(bound.read_bytes()+b"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "changed bound input"):
            runner.load(self.plan_path)

    def test_snapshot_imports_and_prediction_replay_without_live_package(self):
        self.prepare()
        # Exercise the copied backend rather than merely inspecting import strings.
        code = '''
import importlib.abc
import json
from pathlib import Path
import sys
import numpy as np
class BlockLivePackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "multilingual_megatron" or fullname.startswith("multilingual_megatron."):
            raise ImportError("snapshot attempted to import live repository: " + fullname)
sys.meta_path.insert(0, BlockLivePackage())
from paper_fit_snapshot.paper_refit_workflow import runner, backend, inputs
plan_path = Path(sys.argv[1])
plan = runner.load(plan_path)
rows, _, _, cal = inputs.read_training(plan_path.parent / "inputs/manifest.json", "all_grid")
for name in inputs.SHARED:
    model = (backend.repetition.Model(name, backend.problem(rows, cal["regularization"]), rows.U_t, stored_parameters=rows.N_total)
             if name == "RN14-stored-reuse" else backend.shared.Model(name, backend.problem(rows, cal["regularization"]), rows.U_t))
    x = (backend.repetition.starts(model, cal["anchor"])[0]
         if name == "RN14-stored-reuse" else backend.shared.starts(model)[0])
    selected = dict(coordinates=x.tolist(), language_coefficients=[dict(language=l, L_inf=-5., A=.3, B=.4) for l in sorted(rows.language.unique())])
    prediction, gap = backend.predict_shared(name, selected, rows, cal["regularization"])
    assert np.isfinite(prediction).all() and gap <= 1e-10
print(json.dumps({"models": list(inputs.SHARED), "pending": runner.status(plan_path)["counts"]["pending"]}))
'''
        env = os.environ | {"PYTHONPATH": str(self.root / "run/source"), "PYTHONDONTWRITEBYTECODE": "1"}
        completed = subprocess.run([sys.executable, "-B", "-c", code, str(self.plan_path)],
            cwd=self.root, env=env, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(completed.stdout)["models"], list(inputs.SHARED))

    def test_submission_dependencies_and_no_duplicate_or_ambiguous_retries(self):
        self.prepare((inputs.NATIVE[0], "RN14-stored-reuse"))
        commands = []

        def submit(command, **kwargs):
            if command[0] == "sbatch":
                commands.append(command)
                self.assertNotIn("SBATCH_GRES", kwargs["env"])
            return subprocess.CompletedProcess(command, 0, stdout=f"{100+len(commands)}\n", stderr="")

        with patch.object(runner.subprocess, "run", side_effect=submit), patch.dict(os.environ, {"SBATCH_GRES": "gpu:1"}, clear=True):
            result = runner.submit(self.plan_path)
            self.assertEqual(len(result["jobs"]), 4)
            self.assertIn("--dependency=afterany:101:102", commands[2])
            self.assertIn("--dependency=afterany:103", commands[3])
            with self.assertRaises(FileExistsError):
                runner.submit(self.plan_path)
            self.assertEqual(len(commands), 4)

    def test_ambiguous_submission_preserves_intent(self):
        self.prepare()
        with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="unknown\n")), \
                patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                runner.submit(self.plan_path)
            self.assertTrue((self.root / "run/submission/repetition-intent.json").exists())
            with self.assertRaises(FileExistsError):
                runner.submit(self.plan_path)

    def test_cpu_and_nested_submission_guards(self):
        plan = self.prepare()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Slurm CPU"):
                runner.cpu_guard(self.plan_path, plan)
        with patch.dict(os.environ, dict(SLURM_JOB_ID="fixture", SLURM_CPUS_PER_TASK="1", SLURM_GPUS="1"), clear=True):
            with self.assertRaisesRegex(ValueError, "GPU"):
                runner.cpu_guard(self.plan_path, plan)
            with self.assertRaisesRegex(ValueError, "nested allocation"):
                runner.submit(self.plan_path)
        with patch.dict(os.environ, dict(SLURM_JOB_ID="fixture", SLURM_CPUS_PER_TASK="1", SLURM_JOB_GPUS="0"), clear=True):
            with self.assertRaisesRegex(ValueError, "GPU"):
                runner.cpu_guard(self.plan_path, plan)
        with patch.dict(os.environ, dict(SLURM_JOB_ID="fixture", SLURM_CPUS_PER_TASK="1", **dict.fromkeys(runner.THREADS, "1")), clear=True):
            with self.assertRaisesRegex(ValueError, "source snapshot"):
                runner.cpu_guard(self.plan_path, plan)

    def test_worker_inputs_do_not_read_evaluation_files_and_claim_once(self):
        plan = self.prepare()
        evidence = self.root / "run/inputs"
        (evidence / "scoring").rename(evidence / "unavailable_scoring")

        def execute(task, path, *args):
            rows, _, _, _ = inputs.read_training(path, task["fold"])
            self.assertFalse(rows.empty)
            return dict(complete=True)

        with patch.object(runner, "cpu_guard"), patch.object(backend, "execute", side_effect=execute), \
                patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}):
            self.assertTrue(runner.worker(self.plan_path, "repetition", 0)["complete"])
            with self.assertRaises(FileExistsError):
                runner.worker(self.plan_path, "repetition", 0)
        self.assertEqual(runner.status(self.plan_path)["counts"]["complete"], 1)
        self.assertFalse(runner.receipt(self.plan_path, plan, 0)["heldout_scores_computed"])

    def test_timeout_is_explicit_and_not_certified(self):
        self.prepare()
        with patch.object(runner, "cpu_guard"), patch.object(backend, "execute", side_effect=TimeoutError("fixture")), \
                patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}):
            result = runner.worker(self.plan_path, "repetition", 0)
        self.assertFalse(result["complete"])
        self.assertFalse(result["certified"])
        self.assertIn("TimeoutError", result["error"])
        self.assertEqual(runner.status(self.plan_path)["counts"]["incomplete"], 1)

    def test_native_delegation_preserves_per_language_calibration(self):
        path = self.root / "bundle/manifest.json"
        for fold in ("largest_N_act", "all_grid"):
            _, _, groups, cal = inputs.read_training(path, fold)
            task = dict(model=inputs.NATIVE[0], fold=fold, stage="fit", language=groups[0]["language"])
            failed = dict(certified=False, selected=None)
            with patch.object(backend.numerics, "fit_group", return_value=failed) as initial, \
                    patch.object(backend.recovery_numerics, "recover", return_value=failed) as recovery:
                result = backend.execute(task, path, None, float("inf"), self.root)
            self.assertFalse(result["certified"])
            self.assertEqual(initial.call_args.args[1:3], (groups[0]["train"], groups[0]["support"]))
            self.assertEqual(recovery.call_args.args[1:3], (groups[0]["train"], groups[0]["support"]))
            self.assertEqual(groups[0]["train"]["delta"] == cal["delta"], fold == "all_grid")

    def test_grid_coverage_rejects_missing_duplicate_and_changed_best(self):
        record = dict(start_index=0, start_id="grid:0", success=True, objective=1., coordinates=[.1])
        shard = (dict(start_indices=[0]), dict(complete=True, attempts=[record], best=[record]))
        self.assertEqual(len(backend.collect_grid([shard], 1)[0]), 1)
        for shards in ([], [shard, shard]):
            with self.assertRaisesRegex(ValueError, "start coverage"):
                backend.collect_grid(shards, 1)
        bad = deepcopy(shard)
        bad[1]["best"][0] = record | {"objective": .5}
        with self.assertRaisesRegex(ValueError, "best coefficient"):
            backend.collect_grid([bad], 1)

    def test_duration_certification_reuses_three_basin_and_one_direct_audit(self):
        attempts = [dict(start_id=str(i), coordinates=[i], success=True, objective=float(i)) for i in (3, 1, 2, 4)]
        profile = Mock()
        with patch.object(backend.ablation.m, "Direct") as direct, \
                patch.object(backend.ablation.m, "equivalence", return_value=dict(equivalent=True)), \
                patch.object(backend.ablation, "health", return_value=dict(healthy=True)):
            result = backend.ablation.certify_attempts(profile, (), "M5-ND", attempts)
            self.assertEqual(result["status"], "healthy")
            self.assertEqual(result["selected"]["start_id"], "1")
            self.assertEqual([c.args[0] for c in profile.run.call_args_list], [[1], [2], [3]])
            direct.assert_called_once_with(profile, attempts[1], ())
            self.assertEqual([a["kind"] for a in result["audits"]], ["basin", "basin", "basin", "direct"])

    def test_all_grid_certification_receipt_is_resealed_not_nested(self):
        self.prepare()
        rows, _, _, cal = inputs.read_training(self.root / "bundle/manifest.json", "all_grid")
        model = backend.shared.Model("J0", backend.problem(rows, cal["regularization"]), rows.U_t)
        fitted = inputs.sealed(dict(certified=True, complete=True, model="J0", task={"stage": "fit"},
            plan_sha256="prior", job_id="1", elapsed_seconds=1.,
            selected=dict(coordinates=backend.shared.starts(model)[0].tolist(),
                language_coefficients=[dict(language=l, L_inf=-5., A=.3, B=.4) for l in sorted(rows.language.unique())])))
        result = backend.execute(dict(model="J0", fold="all_grid", stage="certify"),
            self.root / "bundle/manifest.json", None, float("inf"), self.root, [({}, fitted)])
        self.assertTrue(result["certified"])
        self.assertFalse({"sha256", "task", "plan_sha256", "job_id", "elapsed_seconds"} & set(result))
        io.write_json_once(self.root / "certificate.json", inputs.sealed(result))
        self.assertTrue(inputs.read_sealed(self.root / "certificate.json")["certified"])

    def test_shared_equation_replay_all_main_forms(self):
        rows, _, _, cal = inputs.read_training(self.root / "bundle/manifest.json", "all_grid")
        for name in inputs.SHARED:
            if name == "RN14-stored-reuse":
                model = backend.repetition.Model(name, backend.problem(rows, cal["regularization"]), rows.U_t, stored_parameters=rows.N_total)
                x = backend.repetition.starts(model, cal["anchor"])[0]
            else:
                model = backend.shared.Model(name, backend.problem(rows, cal["regularization"]), rows.U_t)
                x = backend.shared.starts(model)[0]
            selected = dict(coordinates=x.tolist(), language_coefficients=[dict(language=l, L_inf=-5., A=.3, B=.4) for l in sorted(rows.language.unique())])
            prediction, gap = backend.predict_shared(name, selected, rows, cal["regularization"])
            self.assertTrue(np.isfinite(prediction).all())
            self.assertLessEqual(gap, 1e-10)

    def test_scoring_only_certified_complete_cells_and_no_available_case_mean(self):
        plan = self.prepare()
        cells = [t for t in plan["tasks"] if t["stage"] == "certify"]
        for task in cells[:2]:
            rows, _, _, cal = inputs.read_training(self.root / "bundle/manifest.json", task["fold"])
            model = backend.repetition.Model(task["model"], backend.problem(rows, cal["regularization"]), rows.U_t, stored_parameters=rows.N_total)
            selected = dict(coordinates=backend.repetition.starts(model, cal["anchor"])[0].tolist(),
                language_coefficients=[dict(language=l, L_inf=-5., A=.3, B=.4) for l in sorted(rows.language.unique())])
            result = dict(plan_sha256=plan["sha256"], task=task, certified=True, complete=True, selected=selected)
            io.write_json_once(self.root / f"run/results/{task['id']:05d}.json", inputs.sealed(result))
        report = reporting.report(self.plan_path)
        self.assertEqual(report["complete_cells"], 2)
        self.assertFalse(report["all_cells_certified"])
        table = pd.read_csv(self.root / "run/report/comparison.csv")
        self.assertTrue(table.mean_primary_rmse.isna().all())
        self.assertTrue(table.L26_all_grid.isna().all())
        with self.assertRaisesRegex(ValueError, "create-only"):
            reporting.report(self.plan_path)

    def test_missing_results_produce_no_scores(self):
        self.prepare()
        result = reporting.report(self.plan_path)
        self.assertEqual(result["complete_cells"], 0)
        self.assertFalse(result["all_cells_certified"])
        table = pd.read_csv(self.root / "run/report/comparison.csv", index_col="model")
        self.assertTrue(table.isna().all().all())
        self.assertFalse((self.root / "run/report/predictions.parquet").exists())


if __name__ == "__main__":
    unittest.main()
