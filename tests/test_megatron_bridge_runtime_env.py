from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class MegatronBridgeRuntimeEnvTest(unittest.TestCase):
    def test_benchmark_single_node_isolates_rendezvous_multi_node_preserves_endpoint(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge import run_megatron_bridge_native_benchmark as launcher

        args = SimpleNamespace(nnodes=1, nproc_per_node=1)
        spec = SimpleNamespace(expected_world_size=1, recipe_name="fixture",
                               sequence_length=16, source_path=Path("spec.json"), recipe_overrides=[])
        with patch.object(launcher, "_launcher_python_cmd", return_value=[sys.executable]), \
                patch.dict("os.environ", {"MASTER_ADDR": "occupied-host", "MASTER_PORT": "39730"}), \
                patch.object(launcher, "_resolve_master_port", side_effect=AssertionError("must not probe a port")):
            command = launcher.build_native_benchmark_command(args, spec)
            self.assertIn("--standalone", command)
            self.assertFalse(any(x.startswith(("--master_", "--node_rank")) for x in command))
        args.nnodes = 2
        spec.expected_world_size = 2
        with patch.object(launcher, "_launcher_python_cmd", return_value=[sys.executable]), \
                patch.object(launcher, "_resolve_master_addr", return_value="node0"), \
                patch.object(launcher, "_resolve_master_port", return_value="24001"), \
                patch.object(launcher, "_node_rank_arg", return_value="1"):
            command = launcher.build_native_benchmark_command(args, spec)
            self.assertNotIn("--standalone", command)
            for arg in ("--master_addr=node0", "--master_port=24001", "--node_rank=1"):
                self.assertIn(arg, command)

    def test_deterministic_benchmark_configures_mamba_before_worker_import(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge import run_megatron_bridge_native_benchmark as launcher

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict("os.environ", {"MAMBA_DETERMINISTIC": "0", "TRITON_CACHE_AUTOTUNING": "1"}, clear=True), \
                patch.object(launcher, "_prepare_pybind11_shim", return_value=None), \
                patch.object(launcher, "_prepend_bridge_venv_cuda_libs"), \
                patch.object(launcher, "_prepend_python_env_cuda_libs"):
            for deterministic in (False, True):
                env = launcher._runtime_environment(
                    bridge_root=Path(tmp), output_dir=Path(tmp),
                    python_executable=sys.executable, deterministic_mode=deterministic,
                )
                self.assertEqual(env["MAMBA_DETERMINISTIC"], "1" if deterministic else "0")
                self.assertEqual(env["TRITON_CACHE_AUTOTUNING"], "0" if deterministic else "1")

    def test_snapshot_bootstrap_preserves_mcore_override_from_launcher(self) -> None:
        import ast
        from importlib.machinery import PathFinder

        # Execute the runner's actual bootstrap without importing CUDA engines.
        source = SRC_ROOT / "multilingual_megatron/pipeline/backends/megatron_bridge/run_study_bridge_recipe.py"
        nodes = ast.parse(source.read_text()).body
        start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_REPO_ROOT" for t in node.targets))
        end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.ImportFrom)
                   and node.module == "cpu_affinity")
        bootstrap = compile(ast.Module(body=nodes[start:end], type_ignores=[]), str(source), "exec")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = root / "engine"
            bundled = bridge / "3rdparty/Megatron-LM"
            override = root / "selected_mcore"
            snapshot = root / "snapshot"
            (bridge / "src").mkdir(parents=True)
            (snapshot / "external").mkdir(parents=True)
            (snapshot / "external/Megatron-Bridge").symlink_to(bridge, target_is_directory=True)
            for engine in (bundled, override):
                (engine / "megatron/core").mkdir(parents=True)
                (engine / "megatron/core/__init__.py").write_text("")
            for selected in (override, None):
                with self.subTest(override=selected):
                    paths = [str(snapshot), str(bridge / "src")]
                    if selected:
                        paths.append(str(selected))
                    paths.append(str(bundled))
                    scope = {"Path": Path, "__file__": str(source),
                             "find_repo_root": lambda anchor: snapshot,
                             "os": SimpleNamespace(environ={"PROOF_MCORE_OVERRIDE_PATH": str(selected)} if selected else {}),
                             "sys": SimpleNamespace(path=paths)}
                    exec(bootstrap, scope)
                    spec = PathFinder.find_spec("megatron.core", [str(Path(p) / "megatron") for p in paths])
                    self.assertEqual(Path(spec.origin), (selected or bundled) / "megatron/core/__init__.py")

    def test_multinode_nsys_reports_are_host_unique(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _with_nsys_profile,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nsys = root / "nsys"
            nsys.touch()
            args = SimpleNamespace(
                nsys_profile=True,
                nsys_bin=str(nsys),
                nsys_output_dir=str(root / "reports"),
                nsys_trace="cuda,nvtx",
                nsys_extra_arg=[],
                nnodes=2,
            )
            command = _with_nsys_profile(["python", "train.py"], args, root)

        self.assertIn(str(root / "reports" / "profile_%h"), command)

    def test_srun_uses_absolute_bash_under_clean_slurm_path(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _srun_cmd,
        )

        command = _srun_cmd(2, ["--exclusive"], "python -c 'print(1)'")

        self.assertEqual(
            command,
            [
                "srun",
                "--nodes",
                "2",
                "--ntasks",
                "2",
                "--ntasks-per-node",
                "1",
                "--exclusive",
                "/bin/bash",
                "-c",
                "python -c 'print(1)'",
            ],
        )

    def test_python_env_cuda_libs_prepend_active_env_paths(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _prepend_python_env_cuda_libs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "bridge-train-a40-cu126"
            python_executable = prefix / "bin" / "python"
            python_executable.parent.mkdir(parents=True)
            site = prefix / "lib" / "python3.12" / "site-packages"
            legacy_site = prefix / "lib" / "python3.1" / "site-packages"
            cuda_runtime_lib = site / "nvidia" / "cuda_runtime" / "lib"
            cuda_runtime_include = site / "nvidia" / "cuda_runtime" / "include"
            legacy_cuda_runtime_lib = legacy_site / "nvidia" / "cuda_runtime" / "lib"
            te_lib = site / "transformer_engine" / "wheel_lib"
            torch_lib = site / "torch" / "lib"
            prefix_lib = prefix / "lib"
            for path in (cuda_runtime_lib, cuda_runtime_include, legacy_cuda_runtime_lib, te_lib, torch_lib, prefix_lib):
                path.mkdir(parents=True, exist_ok=True)
            (cuda_runtime_include / "cuda_runtime.h").write_text("/* fixture */\n", encoding="utf-8")

            env = {
                "PROOF_PREFER_PYTHON_ENV_CUDA_LIBS": "1",
                "LD_LIBRARY_PATH": "/opt/cluster/CUDA/13.0.0/lib64",
            }
            lib_dirs = _prepend_python_env_cuda_libs(env, str(python_executable))

        ld_entries = env["LD_LIBRARY_PATH"].split(":")
        self.assertEqual(ld_entries[0], str(cuda_runtime_lib))
        self.assertNotIn(str(legacy_cuda_runtime_lib), ld_entries)
        self.assertIn(str(te_lib), ld_entries)
        self.assertIn(str(torch_lib), ld_entries)
        self.assertEqual(ld_entries[-1], "/opt/cluster/CUDA/13.0.0/lib64")
        self.assertEqual(env["NVTE_CUDA_INCLUDE_PATH"], str(cuda_runtime_include))
        self.assertEqual(env["PROOF_PYTHON_ENV_PREFIX"], str(prefix))
        self.assertIn(cuda_runtime_lib, lib_dirs)

    def test_runtime_env_exports_dynamic_cuda_te_and_network_keys(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
            _export_runtime_env,
        )

        exports = _export_runtime_env(
            {
                "PATH": "/bin",
                "NVTE_FUSED_ATTN": "1",
                "NCCL_DEBUG": "INFO",
                "DEEP_EP_DEVICE_TO_HCA_MAPPING": "0:mlx5_0:1",
                "FI_PROVIDER": "efa",
                "NVSHMEM_DIR": "/opt/nvshmem",
                "NVSHMEM_ENABLE_NIC_PE_MAPPING": "1",
                "PROOF_PREFER_PYTHON_ENV_CUDA_LIBS": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "IGNORED_KEY": "value",
                "NVTE_EMPTY": "",
            }
        )

        self.assertIn("export NVTE_FUSED_ATTN=1", exports)
        self.assertIn("export NCCL_DEBUG=INFO", exports)
        self.assertIn("export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1", exports)
        self.assertIn("export FI_PROVIDER=efa", exports)
        self.assertIn("export NVSHMEM_DIR=/opt/nvshmem", exports)
        self.assertIn("export NVSHMEM_ENABLE_NIC_PE_MAPPING=1", exports)
        self.assertIn("export PROOF_PREFER_PYTHON_ENV_CUDA_LIBS=1", exports)
        self.assertIn("export CUDA_DEVICE_MAX_CONNECTIONS=1", exports)
        self.assertNotIn("IGNORED_KEY", exports)
        self.assertNotIn("NVTE_EMPTY", exports)

    def test_slurm_executor_exports_deepep_and_nvshmem_keys(self) -> None:
        from multilingual_megatron.pipeline.executor import SlurmExecutor

        executor = SlurmExecutor(conda_base="none", conda_env="none")

        script = executor._build_job_script(
            ["python", "-c", "print('ok')"],
            {
                "DEEP_EP_DEVICE_TO_HCA_MAPPING": "0:mlx5_0:1",
                "NVSHMEM_DIR": "/opt/nvshmem",
                "NVSHMEM_ENABLE_NIC_PE_MAPPING": "1",
            },
            "env-test",
            "env-test.log",
        )

        self.assertIn('export DEEP_EP_DEVICE_TO_HCA_MAPPING="0:mlx5_0:1"', script)
        self.assertIn('export NVSHMEM_DIR="/opt/nvshmem"', script)
        self.assertIn('export NVSHMEM_ENABLE_NIC_PE_MAPPING="1"', script)

    def test_slurm_executor_defaults_are_site_neutral(self) -> None:
        from multilingual_megatron.pipeline.executor import SlurmExecutor

        script = SlurmExecutor()._build_job_script(
            ["/bin/true"], {}, "portable-defaults", "portable-defaults.log"
        )

        self.assertNotIn("#SBATCH --partition=", script)
        self.assertNotIn("#SBATCH --gres=", script)
        self.assertNotIn("#SBATCH --mem=", script)
        self.assertNotIn("module load", script)
        self.assertNotIn("CONDA_BASE=", script)
        self.assertNotIn("/opt/software", script)

    def test_gpu_monitor_preserves_dotted_attempt_directories(self) -> None:
        from multilingual_megatron.pipeline.executor import SlurmExecutor

        for attempt in (1, 2):
            log = f'/tmp/routing.v3/logs/attempts/pilot_c.a{attempt}/slurm.log'
            script = SlurmExecutor()._build_job_script(['/bin/true'], {}, 'monitor', log)
            self.assertIn(f'GPU_MONITOR_CSV="/tmp/routing.v3/logs/attempts/pilot_c.a{attempt}/slurm.gpu.csv"', script)

    def test_slurm_nil_export_bootstraps_a_deterministic_system_path(self) -> None:
        from multilingual_megatron.pipeline.executor import SlurmExecutor

        clean_executor = SlurmExecutor(
            conda_base="none",
            conda_env="none",
            modules=[],
            extra_sbatch_flags=["--export=NIL"],
        )
        inherited_executor = SlurmExecutor(
            conda_base="none",
            conda_env="none",
            modules=[],
        )

        clean_script = clean_executor._build_job_script(
            ["/bin/true"], {}, "clean-env-test", "clean-env-test.log"
        )
        inherited_script = inherited_executor._build_job_script(
            ["/bin/true"], {}, "inherited-env-test", "inherited-env-test.log"
        )

        self.assertIn('export PATH="/usr/local/bin:/usr/bin:/bin"', clean_script)
        self.assertNotIn(
            'export PATH="/usr/local/bin:/usr/bin:/bin"', inherited_script
        )

    def test_submitted_sbatch_script_is_written_owner_only(self) -> None:
        from multilingual_megatron.pipeline.executor import SlurmExecutor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "private.log"
            with patch(
                "multilingual_megatron.pipeline.executor.subprocess.run",
                return_value=SimpleNamespace(stdout="12345\n"),
            ):
                job_id = SlurmExecutor().submit(
                    ["/bin/true"],
                    {"WANDB_API_KEY": "test-secret"},
                    False,
                    log_path,
                    script_dir=root,
                )

            script_path = root / "private.sbatch.sh"
            self.assertEqual(job_id, "12345")
            self.assertEqual(script_path.stat().st_mode & 0o777, 0o600)
            self.assertIn("test-secret", script_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
