#!/usr/bin/env python3
"""Production launcher for one load-once per-language Bridge evaluation.

This wrapper mirrors the training launch environment but writes validation-
specific completion artifacts.  It is also the GPU sentinel entrypoint: point
``--spec`` at a one-language fixture and the exact production recipe/checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from multilingual_megatron.repo import find_repo_root
from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import (
    load_evaluation_spec,
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
    parser.add_argument(
        "--recipe",
        default="",
        help="Optional assertion; the executable recipe is frozen in model_recipe.name inside --spec.",
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--launcher", choices=("python", "uv"), default="python")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--srun-arg", action="append", default=[])
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--preflight-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", help="Invocation log directory; artifacts stay under spec.output_dir")
    return parser


def build_validation_command(
    args: argparse.Namespace,
    *,
    sequence_length: int,
    recipe_name: str,
    recipe_overrides: tuple[str, ...],
) -> list[str]:
    nnodes = max(1, int(args.nnodes))
    cmd = _launcher_python_cmd(args)
    cmd.extend(
        [
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={int(args.nproc_per_node)}",
        ]
    )
    master_addr = _resolve_master_addr() if nnodes > 1 else str(os.environ.get("MASTER_ADDR", "")).strip()
    master_port = _resolve_master_port()
    if master_addr:
        cmd.append(f"--master_addr={master_addr}")
    cmd.append(f"--master_port={master_port}")
    if nnodes > 1:
        cmd.append(f"--node_rank={_node_rank_arg()}")
    cmd.extend(
        [
            "--module",
            "multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe",
            "--recipe",
            recipe_name,
            "--dataset",
            "llm-pretrain",
            "--seq_length",
            str(sequence_length),
            "--per-language-validation-spec",
            str(Path(args.spec).resolve()),
        ]
    )
    cmd.extend(recipe_overrides)
    cmd.extend(str(item) for item in args.extra_override if str(item).strip())
    return cmd


def _runtime_environment(
    *,
    bridge_root: Path,
    output_dir: Path,
    python_executable: str,
) -> dict[str, str]:
    repo_root = find_repo_root(Path(__file__))
    env = dict(os.environ)
    env["NEMO_EXPERIMENTS_DIR"] = str(output_dir / "nemo_experiments")
    env.setdefault("TRITON_CACHE_DIR", str(output_dir / "triton_cache"))
    shim_root = _prepare_pybind11_shim(output_dir, env)
    pythonpath = [
        str(repo_root / "src"),
        str(repo_root),
        str(bridge_root / "src"),
        str(bridge_root / "3rdparty" / "Megatron-LM"),
    ]
    if shim_root is not None:
        pythonpath.insert(0, str(shim_root))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    _prepend_bridge_venv_cuda_libs(env, bridge_root)
    _prepend_python_env_cuda_libs(env, python_executable)
    return env


def _run_preflight(
    *,
    args: argparse.Namespace,
    bridge_root: Path,
    env: dict[str, str],
) -> None:
    command = [
        str(args.python_executable),
        "-c",
        (
            "import torch; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "assert torch.cuda.device_count() >= 1, 'no CUDA device'; "
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
    spec = load_evaluation_spec(Path(args.spec))
    bridge_root = Path(args.bridge_root).resolve()
    if not (bridge_root / "src" / "megatron" / "bridge").is_dir():
        raise SystemExit(f"invalid Megatron-Bridge root: {bridge_root}")
    if int(args.nproc_per_node) <= 0 or int(args.nnodes) <= 0:
        raise SystemExit("nproc-per-node and nnodes must be positive")
    if args.recipe and str(args.recipe) != spec.recipe_name:
        raise SystemExit(
            f"--recipe {args.recipe!r} does not match frozen spec recipe {spec.recipe_name!r}"
        )
    if args.extra_override:
        raise SystemExit(
            "ad-hoc --extra-override values are forbidden for production validation; "
            "freeze them in model_recipe.overrides"
        )

    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = output_dir / ".validation_complete"
    if not args.dry_run:
        complete_marker.unlink(missing_ok=True)
    command = build_validation_command(
        args,
        sequence_length=spec.sequence_length,
        recipe_name=spec.recipe_name,
        recipe_overrides=spec.recipe_overrides,
    )
    env = _runtime_environment(
        bridge_root=bridge_root,
        output_dir=output_dir,
        python_executable=str(args.python_executable),
    )
    invocation = {
        "schema": "megatron_bridge_per_language_validation_invocation/v1",
        "cmd": command,
        "cwd": str(bridge_root),
        "spec": {"path": str(spec.source_path), "sha256": spec.source_sha256},
        "checkpoint_root": str(spec.checkpoint_path),
        "checkpoint_step": spec.checkpoint_step,
        "recipe": spec.recipe_name,
        "recipe_overrides": list(spec.recipe_overrides),
        "eval_micro_batch_size": spec.eval_micro_batch_size,
        "nnodes": int(args.nnodes),
        "nproc_per_node": int(args.nproc_per_node),
        "dry_run": bool(args.dry_run),
    }
    (output_dir / "bridge_validation_command.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        return 0
    if args.preflight_cuda:
        _run_preflight(args=args, bridge_root=bridge_root, env=env)

    started = time.time()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "bridge_validation.log"
    with log_path.open("w", encoding="utf-8") as log:
        if int(args.nnodes) > 1:
            inner = " ".join(shlex.quote(item) for item in command)
            inner = _with_runtime_env(f"cd {shlex.quote(str(bridge_root))} && {inner}", env)
            launch = _srun_cmd(int(args.nnodes), list(args.srun_arg), inner)
            completed = subprocess.run(launch, cwd=str(bridge_root), env=env, stdout=log, stderr=subprocess.STDOUT)
        else:
            completed = subprocess.run(command, cwd=str(bridge_root), env=env, stdout=log, stderr=subprocess.STDOUT)
    result_path = output_dir / "evaluation_manifest.json"
    status = (
        "complete"
        if completed.returncode == 0 and complete_marker.is_file() and result_path.is_file()
        else "failed"
    )
    result = {
        "schema": "megatron_bridge_per_language_validation_launcher_result/v1",
        "status": status,
        "return_code": int(completed.returncode),
        "runtime_seconds": time.time() - started,
        "log_path": str(log_path),
        "evaluation_manifest": str(result_path),
    }
    (output_dir / "bridge_validation_launcher_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status != "complete":
        raise SystemExit(
            f"per-language Bridge validation failed (return_code={completed.returncode}); see {log_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
