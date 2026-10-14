#!/usr/bin/env python3
"""Launch native Megatron scoring for a frozen lm-eval request manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from multilingual_megatron.repo import find_repo_root
from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import (
    NativeBenchmarkScoreSpec,
    load_native_benchmark_score_spec,
)

from .run_megatron_bridge_training import (
    _launcher_python_cmd,
    _node_rank_arg,
    _prepend_bridge_venv_cuda_libs,
    _prepend_python_env_cuda_libs,
    _prepare_pybind11_shim,
    _resolve_master_addr,
    _resolve_master_port,
    _srun_cmd,
    _with_runtime_env,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--launcher", choices=("python", "uv"), default="python")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--srun-arg", action="append", default=[])
    parser.add_argument("--preflight-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_native_benchmark_command(
    args: argparse.Namespace,
    spec: NativeBenchmarkScoreSpec,
) -> list[str]:
    nnodes = int(args.nnodes)
    nproc_per_node = int(args.nproc_per_node)
    if nnodes <= 0 or nproc_per_node <= 0:
        raise ValueError("nnodes and nproc-per-node must be positive")
    if nnodes * nproc_per_node != spec.expected_world_size:
        raise ValueError(
            f"launch world size {nnodes * nproc_per_node} does not match frozen "
            f"expected_world_size {spec.expected_world_size}"
        )
    command = _launcher_python_cmd(args)
    command.extend(
        [
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
        ]
    )
    if nnodes == 1:
        # Let torchrun bind and retain an OS-assigned rendezvous port. Job-ID
        # arithmetic can collide with other services or ephemeral sockets;
        # probing a free port and closing it also leaves a race before launch.
        command.append("--standalone")
    else:
        master_addr = _resolve_master_addr()
        if master_addr:
            command.append(f"--master_addr={master_addr}")
        command.append(f"--master_port={_resolve_master_port()}")
        command.append(f"--node_rank={_node_rank_arg()}")
    command.extend(
        [
            "--module",
            "multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe",
            "--recipe",
            spec.recipe_name,
            "--dataset",
            "llm-pretrain",
            "--seq_length",
            str(spec.sequence_length),
            "--native-benchmark-score-spec",
            str(spec.source_path),
        ]
    )
    command.extend(spec.recipe_overrides)
    return command


def _shell_join_distributed_command(command: list[str]) -> str:
    """Quote a launch command while leaving Slurm's per-node rank expandable."""

    return " ".join(
        item if item.startswith("--node_rank=${") else shlex.quote(item)
        for item in command
    )


def _runtime_environment(
    *,
    bridge_root: Path,
    output_dir: Path,
    python_executable: str,
    deterministic_mode: bool = False,
) -> dict[str, str]:
    repo_root = find_repo_root(Path(__file__))
    env = dict(os.environ)
    if deterministic_mode:
        # Mamba chooses Triton configurations at import time, before Bridge
        # applies torch.use_deterministic_algorithms. Select its supported
        # deterministic kernels before starting the worker processes. Cached
        # autotuning would instead permit rank-specific kernel choices.
        env["MAMBA_DETERMINISTIC"] = "1"
        env["TRITON_CACHE_AUTOTUNING"] = "0"
    env["NEMO_EXPERIMENTS_DIR"] = str(output_dir / "nemo_experiments")
    if env.get("PROOF_ISOLATE_TRITON_CACHE", "").strip() != "1":
        env.setdefault("TRITON_CACHE_DIR", str(output_dir / "triton_cache"))
    shim_root = _prepare_pybind11_shim(output_dir, env)
    pythonpath = [
        str(repo_root / "src"),
        str(repo_root),
        str(bridge_root / "src"),
    ]
    mcore_override = env.get("PROOF_MCORE_OVERRIDE_PATH", "").strip()
    if mcore_override:
        pythonpath.append(str(Path(mcore_override).expanduser().resolve()))
    pythonpath.append(str(bridge_root / "3rdparty" / "Megatron-LM"))
    if shim_root is not None:
        pythonpath.insert(0, str(shim_root))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    _prepend_bridge_venv_cuda_libs(env, bridge_root)
    _prepend_python_env_cuda_libs(env, python_executable)
    return env


def _run_preflight(args: argparse.Namespace, bridge_root: Path, env: dict[str, str]) -> None:
    command = [
        str(args.python_executable),
        "-c",
        (
            "import torch; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "print(torch.cuda.get_device_name(torch.cuda.current_device()))"
        ),
    ]
    if int(args.nnodes) > 1:
        inner = " ".join(shlex.quote(item) for item in command)
        inner = _with_runtime_env(f"cd {shlex.quote(str(bridge_root))} && {inner}", env)
        command = _srun_cmd(int(args.nnodes), list(args.srun_arg), inner)
    subprocess.run(command, cwd=str(bridge_root), env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_native_benchmark_score_spec(Path(args.spec))
    bridge_root = Path(args.bridge_root).resolve()
    if not (bridge_root / "src" / "megatron" / "bridge").is_dir():
        raise SystemExit(f"invalid Megatron-Bridge root: {bridge_root}")
    try:
        command = build_native_benchmark_command(args, spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = spec.output_dir / ".scoring_complete"
    if not args.dry_run:
        complete_marker.unlink(missing_ok=True)
    env = _runtime_environment(
        bridge_root=bridge_root,
        output_dir=spec.output_dir,
        python_executable=str(args.python_executable),
        deterministic_mode=bool(spec.model_contract.get("deterministic_mode", False)),
    )
    invocation = {
        "schema": "megatron_bridge_native_benchmark_invocation/v1",
        "cmd": command,
        "cwd": str(bridge_root),
        "spec": {"path": str(spec.source_path), "sha256": spec.source_sha256},
        "checkpoint_root": str(spec.checkpoint_path),
        "checkpoint_step": spec.checkpoint_step,
        "recipe": spec.recipe_name,
        "request_manifest": str(spec.request_manifest_path),
        "nnodes": int(args.nnodes),
        "nproc_per_node": int(args.nproc_per_node),
        "max_gpu_hours": spec.max_gpu_hours,
        "dry_run": bool(args.dry_run),
        "deterministic_kernel_environment": {
            key: env[key] for key in ("MAMBA_DETERMINISTIC", "TRITON_CACHE_AUTOTUNING")
            if key in env
        },
    }
    (spec.output_dir / "bridge_benchmark_command.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        return 0
    if args.preflight_cuda:
        _run_preflight(args, bridge_root, env)

    started = time.time()
    log_path = spec.output_dir / "native_benchmark.log"
    with log_path.open("w", encoding="utf-8") as log:
        if int(args.nnodes) > 1:
            inner = _shell_join_distributed_command(command)
            inner = _with_runtime_env(
                f"cd {shlex.quote(str(bridge_root))} && {inner}", env
            )
            launch = _srun_cmd(int(args.nnodes), list(args.srun_arg), inner)
        else:
            launch = command
        completed = subprocess.run(
            launch,
            cwd=str(bridge_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    scores_path = spec.output_dir / "scores.json"
    status = (
        "complete"
        if completed.returncode == 0 and complete_marker.is_file() and scores_path.is_file()
        else "failed"
    )
    result = {
        "schema": "megatron_bridge_native_benchmark_launcher_result/v1",
        "status": status,
        "return_code": int(completed.returncode),
        "runtime_seconds": time.time() - started,
        "log_path": str(log_path),
        "scores": str(scores_path),
    }
    (spec.output_dir / "launcher_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status != "complete":
        raise SystemExit(
            f"native benchmark failed (return_code={completed.returncode}); see {log_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
