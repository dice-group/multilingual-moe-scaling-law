from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import cooled
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import collect
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import execution_env
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import identity
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import read
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import verify_files
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import write
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation import PROFILE
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import allocation
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import attempt_logs
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import render_script
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import submission_command
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import step_command
from multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation_launch import worker_command


class EvaluationTest(unittest.TestCase):
    def launch_fixture(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read(root / PROFILE)
        return {"output": "/tmp/evaluation with spaces", "attempt_id": "fixture_01",
                "profile": cfg, "runtime": read(root / cfg["runtime"]),
                "spec": {"path": "/tmp/frozen spec.json"}}

    def test_submission_and_workers_share_validated_profile_allocation(self):
        plan = self.launch_fixture()
        spec = {"sequence_length": 2048, "model_recipe": {"name": "frozen_recipe",
                "overrides": ["model.pipeline_model_parallel_size=4", "mixed_precision=bf16_mixed"]}}
        for nodes, gpus in ((2, 4), (4, 2)):
            plan["profile"].update(nodes=nodes, gpus_per_node=gpus)
            plan["runtime"]["hardware"].update(nodes=nodes, gpus_per_node=gpus)
            batch = submission_command(plan, "launch.sh")
            step = step_command("plan.json", plan)
            worker = worker_command(plan, spec, "0", "master", "123")
            self.assertIn(f"--nodes={nodes}", batch)
            self.assertIn(f"--gres=gpu:h100:{gpus}", batch)
            self.assertIn(f"--ntasks={nodes}", step)
            self.assertIn(f"--nodes={nodes}", step)
            self.assertIn(f"--nnodes={nodes}", worker)
            self.assertIn(f"--nproc_per_node={gpus}", worker)
            self.assertEqual(worker[-2:], spec["model_recipe"]["overrides"])
        with self.assertRaisesRegex(ValueError, "node rank"):
            worker_command(plan, spec, "4", "master", "123")
        for change in ({"nodes": 1}, {"nodes": True}, {"gpus_per_node": 0}):
            with self.assertRaises(ValueError):
                allocation({**plan["profile"], **change}, plan["runtime"])
        plan["runtime"]["hardware"]["world_size"] = 16
        with self.assertRaisesRegex(ValueError, "runtime hardware"):
            allocation(plan["profile"], plan["runtime"])

    def test_launch_logs_and_scripts_preserve_stable_artifact_paths(self):
        plan = self.launch_fixture()
        logs = attempt_logs(plan["output"], plan["attempt_id"])
        self.assertEqual(logs, Path(plan["output"])/"logs/attempts/fixture_01")
        self.assertIn(f"--output={logs}/evaluation-%j.log", submission_command(plan, "launch.sh"))
        plan_path = Path(plan["output"])/"plan.json"
        for action in ("run", "preflight"):
            script = render_script(plan_path, plan, action)
            subprocess.run(["bash", "-n"], input=script, text=True, check=True)
            argv = shlex.split(script.splitlines()[-1])
            self.assertEqual(argv[-3:], [action, "--plan", str(plan_path)])
        for invalid in ("../escape", "/absolute", "", "a/b"):
            with self.assertRaises(ValueError):
                attempt_logs(plan["output"], invalid)

    def test_preflight_and_workers_use_explicit_production_compatibility(self):
        profile = read(Path(__file__).resolve().parents[1] / PROFILE)
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {
            "PROOF_ENABLE_BRIDGE_OPTIONAL_DEP_SHIMS": "0", "WANDB_MODE": "online"
        }):
            plan = {"output": tmp, "profile": profile,
                    "runtime": {"module_runtime": {"conda_env": "/unused"}}}
            backend = "multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training"
            with patch(backend + "._prepare_pybind11_shim", return_value=None), patch(backend + "._prepend_python_env_cuda_libs"):
                env = execution_env(plan, Path(tmp) / "worker")
            self.assertEqual(env["PROOF_ENABLE_BRIDGE_OPTIONAL_DEP_SHIMS"], "1")
            self.assertEqual(env["PROOF_REQUIRE_REAL_MODELOPT"], "1")
            self.assertEqual(env["WANDB_MODE"], "disabled")
            self.assertIn(str(Path(tmp) / "source/src"), env["PYTHONPATH"])

    def test_only_complete_cooled_endpoint_is_eligible(self):
        audit = SimpleNamespace(valid_full_state=True, train_state_step=12, consumed_train_samples=384)
        scheduler = {"lr_decay_style": "WSD", "num_steps": 384, "lr_decay_steps": 384, "min_lr": 0.0}
        self.assertTrue(cooled(audit, scheduler, 12, 384, True))
        self.assertFalse(cooled(audit, scheduler, 12, 384, False))
        self.assertFalse(cooled(audit, {**scheduler, "lr_decay_steps": 768}, 12, 384, True))
        self.assertFalse(cooled(audit, scheduler, 13, 416, True))
        audit.valid_full_state = False
        self.assertFalse(cooled(audit, scheduler, 12, 384, True))

    def test_source_or_input_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            write(p, {"value": 1})
            entry = identity(p)
            verify_files([entry])
            write(p, {"value": 2})
            with self.assertRaisesRegex(ValueError, "changed input"):
                verify_files([entry])

    def test_collection_requires_exact_coverage_and_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write(out / "spec.json", {"required_languages": ["eng_Latn"]})
            write(out / "exposure.json", [{"language": "eng_Latn", "D_t": 2048}])
            plan = {"output": str(out), "inputs": [], "spec": identity(out / "spec.json"),
                    "expected_step": 1, "expected_samples": 1, "N_act": 100,
                    "N_act_nonembedding": 80, "E": 32, "K": 1, "model_seed": 1234}
            write(out / "plan.json", plan)
            loss = {"raw_nll": 2.0, "atlas_tao_unigram_adjusted_loss": -1.0,
                    "nll_sum": 4.0, "token_count": 2, "unigram_log_probability_sum": -6.0,
                    "backend": {"backend_token_count": 2}}
            write(out / "language.json", loss)
            write(out / "report.json", {})
            manifest = {"status": "complete", "checkpoint": {"step": 1},
                        "spec": plan["spec"], "languages": {"eng_Latn": identity(out / "language.json")},
                        "validation_report": identity(out / "report.json")}
            write(out / "final/evaluation_manifest.json", manifest)
            collect(plan)
            self.assertEqual(read(out / "release.json")["rows"], 1)
            write(out / "final/evaluation_manifest.json", {**manifest, "languages": {}})
            with self.assertRaisesRegex(ValueError, "coverage"):
                collect(plan)
            loss["backend"]["backend_token_count"] = 3
            write(out / "language.json", loss)
            manifest["languages"]["eng_Latn"] = identity(out / "language.json")
            write(out / "final/evaluation_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "denominator"):
                collect(plan)


if __name__ == "__main__":
    unittest.main()
