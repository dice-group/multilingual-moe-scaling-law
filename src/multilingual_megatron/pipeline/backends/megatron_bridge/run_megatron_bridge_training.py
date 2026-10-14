#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a Megatron Bridge recipe and emit pipeline-friendly artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

from multilingual_megatron.repo import find_repo_root
from multilingual_megatron.pipeline.stages.training.checkpoint_state import (
    checkpoint_path_for_step,
    inspect_full_state_checkpoint,
    write_checkpoint_audit,
)


_PYTHON_ENV_CUDA_PACKAGES = (
    "cuda_runtime",
    "cuda_cupti",
    "cuda_nvrtc",
    "cublas",
    "cudnn",
    "nccl",
    "cusparselt",
    "cusparse",
    "cufft",
    "curand",
    "cusolver",
    "cufile",
    "nvjitlink",
    "nvshmem",
    "nvtx",
)
_RUNTIME_EXPORT_KEYS = {
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PYTHON_EXEC",
    "PYTHON_EXECUTABLE",
    "NSYS_BIN",
    "BRIDGE_TRAIN_PYTHON",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NEMO_EXPERIMENTS_DIR",
    "TRITON_CACHE_DIR",
    "TRITON_HOME",
    "PROOF_TRITON_CACHE_PARENT",
}
_RUNTIME_EXPORT_PREFIXES = (
    "BRIDGE_",
    "FI_",
    "NCCL_",
    "DEEP_EP_",
    "NVSHMEM_",
    "NVTE_",
    "OMPI_",
    "PMI_",
    "PMIX_",
    "PROOF_",
    "PYTORCH_",
    "UCX_",
)


def _dedupe_paths(entries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        value = str(entry).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _prepend_ld_library_path(env: dict[str, str], lib_dirs: list[Path]) -> None:
    entries = [str(path) for path in lib_dirs if path.is_dir()]
    existing = str(env.get("LD_LIBRARY_PATH", "")).strip()
    if existing:
        entries.extend(existing.split(":"))
    entries = _dedupe_paths(entries)
    if entries:
        env["LD_LIBRARY_PATH"] = ":".join(entries)


def _python_prefix_for_executable(python_executable: str, env: dict[str, str]) -> Path | None:
    executable = str(python_executable).strip()
    if executable:
        path = Path(executable)
        if path.name.startswith("python") and path.parent.name == "bin":
            return path.parent.parent
    conda_prefix = str(env.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        return Path(conda_prefix)
    if Path(sys.executable).name.startswith("python") and Path(sys.executable).parent.name == "bin":
        return Path(sys.executable).parent.parent
    return None


def _site_root_version(path: Path) -> tuple[int, ...]:
    raw = path.parent.name.removeprefix("python")
    parts: list[int] = []
    for item in raw.split("."):
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _python_env_site_roots(prefix: Path) -> list[Path]:
    roots = [
        path
        for path in sorted((prefix / "lib").glob("python*/site-packages"))
        if path.is_dir()
    ]
    if not roots:
        return []
    highest = max(_site_root_version(path) for path in roots)
    return [path for path in roots if _site_root_version(path) == highest]


def _prepend_python_env_cuda_libs(env: dict[str, str], python_executable: str) -> list[Path]:
    """Prefer CUDA and TE libraries bundled with the configured Python env."""
    if str(env.get("PROOF_PREFER_PYTHON_ENV_CUDA_LIBS", "")).strip() != "1":
        return []

    prefix = _python_prefix_for_executable(python_executable, env)
    if prefix is None:
        return []

    site_roots = _python_env_site_roots(prefix)
    lib_dirs: list[Path] = []
    for site_root in site_roots:
        for package in _PYTHON_ENV_CUDA_PACKAGES:
            lib_dirs.append(site_root / "nvidia" / package / "lib")
        lib_dirs.append(site_root / "transformer_engine" / "wheel_lib")
        lib_dirs.append(site_root / "torch" / "lib")
        cuda_include = site_root / "nvidia" / "cuda_runtime" / "include"
        if (cuda_include / "cuda_runtime.h").exists():
            env.setdefault("NVTE_CUDA_INCLUDE_PATH", str(cuda_include))
    lib_dirs.append(prefix / "lib")

    _prepend_ld_library_path(env, lib_dirs)
    env["PROOF_PYTHON_ENV_PREFIX"] = str(prefix)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    return [path for path in lib_dirs if path.is_dir()]


def _prepend_bridge_venv_cuda_libs(env: dict[str, str], bridge_root: Path) -> None:
    """Prefer CUDA libraries bundled with the Bridge venv."""
    if str(env.get("BRIDGE_PREFER_VENV_CUDA_LIBS", "")).strip() != "1":
        return

    lib_root = bridge_root / ".venv" / "lib"
    lib_dirs = [
        path
        for path in sorted(lib_root.glob("python*/site-packages/nvidia/*/lib"))
        if path.is_dir()
    ]
    lib_dirs.extend(
        path
        for path in sorted(lib_root.glob("python*/site-packages/transformer_engine/wheel_lib"))
        if path.is_dir()
    )
    _prepend_ld_library_path(env, lib_dirs)


def _prepare_pybind11_shim(output_dir: Path, env: dict[str, str]) -> Path | None:
    """Provide the small pybind11/python-config surface Megatron needs.

    The H100 ``study`` environment has torch and the pybind11 headers bundled
    under torch, but it does not install the pybind11 Python package. Megatron
    Core's dataset helper Makefile shells out to ``python3 -m pybind11
    --includes`` and ``python3-config --extension-suffix``; this runtime shim
    keeps that compile step on the active interpreter without mutating the
    environment.
    """
    if str(env.get("PROOF_ENABLE_PYBIND11_SHIM", "")).strip() != "1":
        return None

    shim_root = output_dir / "runtime_shims"
    bin_dir = shim_root / "bin"
    package_dir = shim_root / "pybind11"
    bin_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)
    init_py = package_dir / "__init__.py"
    main_py = package_dir / "__main__.py"
    python3 = bin_dir / "python3"
    python_config = bin_dir / "python3-config"
    init_py.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def get_include(user=False):",
                "    import torch",
                "    return str(Path(torch.__file__).resolve().parent / 'include')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    main_py.write_text(
        "\n".join(
            [
                "import sys",
                "import sysconfig",
                "from . import get_include",
                "",
                "if '--includes' in sys.argv:",
                "    print(f\"-I{sysconfig.get_paths()['include']} -I{get_include()}\")",
                "else:",
                "    raise SystemExit('unsupported pybind11 shim invocation')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    python_config.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import sys",
                "import sysconfig",
                "",
                "args = set(sys.argv[1:])",
                "if '--extension-suffix' in args:",
                "    print(sysconfig.get_config_var('EXT_SUFFIX') or '')",
                "elif '--includes' in args:",
                "    include = sysconfig.get_paths().get('include', '')",
                "    platinclude = sysconfig.get_paths().get('platinclude', include)",
                "    print(' '.join(f'-I{path}' for path in dict.fromkeys([include, platinclude]) if path))",
                "else:",
                "    raise SystemExit('unsupported python3-config shim invocation')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    python3.write_text(
        "\n".join(
            [
                "#!/usr/bin/env sh",
                f"exec {shlex.quote(sys.executable)} \"$@\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    python3.chmod(0o755)
    python_config.chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return shim_root


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing Megatron data manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "megatron_gpt":
        raise SystemExit(f"Unsupported manifest format in {path}: {payload.get('format')!r}")
    return payload


def _resolve_master_port() -> str:
    explicit = str(os.environ.get("MASTER_PORT", "")).strip()
    if explicit:
        return explicit
    if str(os.environ.get("SLURM_NNODES", "1")).strip() in {"", "1"}:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return str(sock.getsockname()[1])
    raw_job_id = str(os.environ.get("SLURM_JOB_ID", "")).strip()
    if raw_job_id.isdigit():
        return str(20000 + (int(raw_job_id) % 40000))
    return str(20000 + (os.getpid() % 40000))


def _resolve_master_addr() -> str:
    explicit = str(os.environ.get("MASTER_ADDR", "")).strip()
    if explicit:
        return explicit
    nodelist = str(os.environ.get("SLURM_JOB_NODELIST", "")).strip()
    if nodelist:
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                check=False,
            )
            first = (result.stdout or "").strip().splitlines()[0]
            if first:
                return first
        except Exception:
            pass
    return socket.gethostname()


def _blend_value(manifest: dict) -> list[object]:
    blend = _optional_blend_value(manifest)
    if blend is None:
        raise SystemExit("Megatron data manifest has no prefixes.")
    return blend


def _optional_blend_value(manifest: dict | None) -> list[object] | None:
    if not isinstance(manifest, dict):
        return None
    prefixes = [str(item) for item in manifest.get("prefixes", []) if str(item).strip()]
    if not prefixes:
        return None
    if len(prefixes) == 1:
        return [prefixes, None]
    weights = manifest.get("weights")
    if isinstance(weights, list) and weights:
        return [prefixes, [float(item) for item in weights]]
    return [prefixes, None]


def _manifest_blend_per_split(
    manifest: dict,
    *,
    include_validation: bool,
    include_test: bool,
) -> list[object | None] | None:
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        return None
    train_blend = _optional_blend_value(splits.get("train"))
    if train_blend is None:
        return None
    return [
        train_blend,
        _optional_blend_value(splits.get("validation")) if include_validation else None,
        _optional_blend_value(splits.get("test")) if include_test else None,
    ]


def _doc_counts(payload: dict) -> dict[str, int]:
    raw = payload.get("documents_per_language", {})
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for language, value in raw.items():
        try:
            counts[str(language)] = int(value)
        except (TypeError, ValueError):
            counts[str(language)] = 0
    return counts


def _write_validation_coverage(output_dir: Path, manifest_path: Path, manifest: dict) -> Path:
    splits = manifest.get("splits", {})
    splits = splits if isinstance(splits, dict) else {}
    train_split = splits.get("train") if isinstance(splits.get("train"), dict) else manifest
    valid_split = splits.get("validation") if isinstance(splits.get("validation"), dict) else {}
    train_docs = _doc_counts(train_split)
    valid_docs = _doc_counts(valid_split)
    languages = sorted(set(train_docs) | set(valid_docs) | set(manifest.get("languages", [])))
    coverage = {
        language: {
            "train_docs": int(train_docs.get(language, 0)),
            "validation_docs": int(valid_docs.get(language, 0)),
        }
        for language in languages
    }
    skipped = [
        language
        for language, row in coverage.items()
        if row["train_docs"] > 0 and row["validation_docs"] == 0
    ]

    payload = {
        "schema_version": 1,
        "kind": "validation_coverage",
        "source_manifest": str(manifest_path),
        "train_prefixes": len(train_split.get("prefixes", [])),
        "validation_prefixes": len(valid_split.get("prefixes", [])),
        "skipped_validation_languages": skipped,
        "languages": coverage,
    }
    path = output_dir / "validation_coverage.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _dataset_overrides_for_manifest(
    manifest_path: Path,
    manifest: dict,
    output_dir: Path,
    *,
    eval_iters: int,
) -> tuple[dict[str, object], bool]:
    validation_requested = int(eval_iters) > 0
    test_requested = int(eval_iters) > 0
    blend_per_split = _manifest_blend_per_split(
        manifest,
        include_validation=validation_requested,
        include_test=test_requested,
    )
    if validation_requested and blend_per_split is not None and blend_per_split[1] is None:
        raise SystemExit(
            "Validation was requested, but the Megatron data manifest has no usable "
            f"physical validation prefixes: {manifest_path}"
        )
    if blend_per_split is not None:
        json_overrides: dict[str, object] = {
            "dataset.blend": None,
            "dataset.blend_per_split": blend_per_split,
            "dataset.split": None,
            "dataset.path_to_cache": str(output_dir / "dataset_cache"),
        }
        uses_physical_splits = True
    else:
        json_overrides = {
            "dataset.blend": _blend_value(manifest),
            "dataset.path_to_cache": str(output_dir / "dataset_cache"),
        }
        uses_physical_splits = False

    tokenizer_type = str(manifest.get("tokenizer_type", "")).strip()
    tokenizer_model = str(manifest.get("tokenizer_model", "")).strip()
    if tokenizer_type:
        json_overrides["tokenizer.tokenizer_type"] = tokenizer_type
    if tokenizer_model:
        json_overrides["tokenizer.tokenizer_model"] = tokenizer_model
    return json_overrides, uses_physical_splits


def _node_rank_arg() -> str:
    return "${SLURM_NODEID:-${SLURM_PROCID:-0}}"


def _srun_cmd(nnodes: int, extra_args: list[str], inner_cmd: str) -> list[str]:
    return [
        "srun",
        "--nodes",
        str(nnodes),
        "--ntasks",
        str(nnodes),
        "--ntasks-per-node",
        "1",
        *extra_args,
        "/bin/bash",
        "-c",
        inner_cmd,
    ]


def _export_runtime_env(env: dict[str, str]) -> str:
    exports = []
    keys = sorted(
        key
        for key in env
        if key in _RUNTIME_EXPORT_KEYS or any(key.startswith(prefix) for prefix in _RUNTIME_EXPORT_PREFIXES)
    )
    for key in keys:
        value = str(env.get(key, "")).strip()
        if value:
            exports.append(f"export {key}={shlex.quote(value)}")
    return "; ".join(exports)


def _with_runtime_env(inner_cmd: str, env: dict[str, str]) -> str:
    exports = _export_runtime_env(env)
    if not exports:
        return inner_cmd
    return f"{exports}; {inner_cmd}"


def _with_nsys_profile(cmd: list[str], args: argparse.Namespace, output_dir: Path) -> list[str]:
    if not bool(args.nsys_profile):
        return cmd
    requested_bin = str(args.nsys_bin or os.environ.get("NSYS_BIN", "")).strip()
    nsys_bin = requested_bin or shutil.which("nsys")
    if not nsys_bin:
        raise FileNotFoundError(
            "Nsight Systems profiling requested, but no nsys binary was found. "
            "Set launch.nsys_bin or NSYS_BIN to a working nsys executable on the target node."
        )
    if requested_bin and not Path(requested_bin).exists() and shutil.which(requested_bin) is None:
        raise FileNotFoundError(f"Configured nsys binary does not exist or is not executable in PATH: {requested_bin}")
    nsys_dir = Path(args.nsys_output_dir).resolve() if args.nsys_output_dir else output_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    # One launcher runs on each node for multi-node training.  Nsight expands
    # ``%h`` to the host name, preventing those launchers from racing on the
    # same shared-filesystem report path.
    report_base = nsys_dir / ("profile_%h" if int(args.nnodes) > 1 else "profile")
    nsys_cmd = [
        str(nsys_bin),
        "profile",
        "-s",
        "none",
        "-t",
        str(args.nsys_trace or "cuda,nvtx,osrt,cublas,cudnn"),
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o",
        str(report_base),
    ]
    nsys_cmd.extend(str(item) for item in args.nsys_extra_arg if str(item).strip())
    return nsys_cmd + cmd


def _write_wrapper_artifacts(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    status: str,
    runtime: float,
    error: str = "",
    return_code: int | None = None,
    global_step: int | None = None,
    consumed_train_samples: int | None = None,
    mark_training_complete: bool = True,
) -> None:
    expected_consumed_samples = int(getattr(args, "expected_consumed_samples", 0) or 0)
    expected_end_step = int(getattr(args, "expected_end_step", 0) or 0)
    actual_consumed_samples = int(consumed_train_samples or expected_consumed_samples or 0)
    expected_tokens = (
        actual_consumed_samples * int(args.seq_length)
        if actual_consumed_samples > 0
        else int(args.train_iters) * int(args.global_batch_size) * int(args.seq_length)
    )
    train_results = {
        "backend": "megatron_bridge",
        "recipe": args.recipe,
        "dataset": args.dataset,
        "status": status,
        "train_runtime": runtime,
        "train_loss": None,
        "num_input_tokens_seen": expected_tokens,
    }
    if error:
        train_results["error"] = error
    if return_code is not None:
        train_results["return_code"] = int(return_code)
    (output_dir / "train_results.json").write_text(json.dumps(train_results, indent=2) + "\n", encoding="utf-8")
    trainer_state = {
        "backend": "megatron_bridge",
        "status": status,
        "log_history": [],
        "global_step": (
            int(global_step)
            if global_step is not None
            else (expected_end_step or int(args.train_iters))
            if status == "complete"
            else 0
        ),
    }
    if error:
        trainer_state["error"] = error
    (output_dir / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2) + "\n", encoding="utf-8")
    if status == "complete" and mark_training_complete:
        (output_dir / ".training_complete").write_text("ok\n", encoding="utf-8")


def _checkpoint_root_from_overrides(args: argparse.Namespace, output_dir: Path) -> Path:
    raw = str(output_dir / "checkpoints")
    for override in args.extra_override:
        value = str(override)
        if value.startswith("checkpoint.save="):
            raw = value.split("=", 1)[1]
    return Path(raw).resolve()


def _replace_extra_override(args: argparse.Namespace, key: str, value: str) -> None:
    """Set a runner override once, removing any stale occurrence of the key."""

    retained: list[str] = []
    for item in args.extra_override:
        raw = str(item)
        if raw.split("=", 1)[0] != key:
            retained.append(raw)
    retained.append(f"{key}={value}")
    args.extra_override = retained


def _normalize_checkpoint_save_override(args: argparse.Namespace, output_dir: Path) -> None:
    """Bind a checkpoint-save override to the wrapper's filesystem namespace.

    The training command runs with ``bridge_root`` as its working directory,
    while the wrapper audits paths relative to its caller.  Leaving a relative
    ``checkpoint.save`` override unnormalised therefore lets the trainer and
    auditor write/read different roots.  Preserve an explicit ``null`` value,
    but canonicalize every real path before constructing the engine command.
    """

    raw = str(output_dir / "checkpoints")
    for override in args.extra_override:
        value = str(override)
        if value.startswith("checkpoint.save="):
            raw = value.split("=", 1)[1]
    if raw.strip().lower() in {"", "null", "none"}:
        return
    _replace_extra_override(args, "checkpoint.save", str(Path(raw).expanduser().resolve()))


def _set_full_state_resume_overrides(
    args: argparse.Namespace,
    *,
    checkpoint_root: Path,
    checkpoint_step: int,
) -> None:
    """Inject the only permitted continuation semantics into the engine command.

    Bridge's root-level tracker can name a later non-persistent checkpoint even
    when ``checkpoint.ckpt_step`` is present in an ordinary override list.  A
    continuation must instead pass the immutable ``iter_*`` directory itself.
    This makes the selected full state (rather than the mutable root tracker)
    the engine's only load candidate.
    """

    selected_checkpoint = checkpoint_path_for_step(checkpoint_root, checkpoint_step)
    if not selected_checkpoint.is_dir():
        raise FileNotFoundError(
            f"selected continuation checkpoint directory is missing: {selected_checkpoint}"
        )

    for key, value in (
        ("checkpoint.load", str(selected_checkpoint.resolve())),
        # The direct iteration directory controls selection.  Clearing this
        # auxiliary option avoids any later configuration path re-resolving
        # the mutable root-level checkpoint tracker.
        ("checkpoint.ckpt_step", "null"),
        ("checkpoint.load_optim", "true"),
        ("checkpoint.load_rng", "true"),
        ("checkpoint.finetune", "false"),
        ("checkpoint.exit_on_missing_checkpoint", "true"),
        # The continuation's frozen schedule (possibly a new longer WSD
        # horizon) owns its hyperparameters; model/optimizer/RNG state remain
        # exactly those of the selected full-state checkpoint.
        ("scheduler.override_opt_param_scheduler", "true"),
        ("scheduler.use_checkpoint_opt_param_scheduler", "false"),
    ):
        _replace_extra_override(args, key, value)




def _launcher_python_cmd(args: argparse.Namespace) -> list[str]:
    if args.launcher == "uv":
        return ["uv", "run", "--no-sync", "python"]
    return [str(args.python_executable)]


def _run_cuda_preflight(
    *,
    args: argparse.Namespace,
    bridge_root: Path,
    output_dir: Path,
    env: dict[str, str],
    nnodes: int,
    srun_args: list[str],
) -> None:
    script_path = output_dir / "cuda_preflight.py"
    script_path.write_text(
        """
import json
import os
import socket
import sys
from pathlib import Path


def _loaded_library_paths() -> list[str]:
    tokens = (
        "libcuda",
        "libcudart",
        "libcudnn",
        "libnccl",
        "libtorch",
        "transformer_engine",
    )
    paths: list[str] = []
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return paths
    for line in lines:
        raw = line.rsplit(" ", 1)[-1].strip()
        if raw.startswith("/") and any(token in raw for token in tokens):
            paths.append(raw)
    return sorted(set(paths))


payload = {
    "host": socket.gethostname(),
    "argv": sys.argv,
    "python": sys.executable,
    "SLURM_PROCID": os.environ.get("SLURM_PROCID", ""),
    "SLURM_NODEID": os.environ.get("SLURM_NODEID", ""),
    "SLURM_LOCALID": os.environ.get("SLURM_LOCALID", ""),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "NVTE_CUDA_INCLUDE_PATH": os.environ.get("NVTE_CUDA_INCLUDE_PATH", ""),
    "MASTER_ADDR": os.environ.get("MASTER_ADDR", ""),
    "MASTER_PORT": os.environ.get("MASTER_PORT", ""),
}
try:
    import torch

    payload["torch_version"] = torch.__version__
    payload["torch_cuda_version"] = getattr(torch.version, "cuda", "")
    payload["torch_cudnn_version"] = torch.backends.cudnn.version()
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        current = int(torch.cuda.current_device())
        payload["current_device"] = current
        payload["device_name"] = torch.cuda.get_device_name(current)
        payload["device_capability"] = torch.cuda.get_device_capability(current)
except Exception as exc:
    payload["torch_exception"] = repr(exc)

try:
    import transformer_engine
    import transformer_engine.pytorch

    payload["transformer_engine_version"] = getattr(transformer_engine, "__version__", "")
    payload["transformer_engine_pytorch_import"] = "ok"
except Exception as exc:
    payload["transformer_engine_exception"] = repr(exc)

payload["loaded_libraries"] = _loaded_library_paths()
payload["bad_cuda13_runtime"] = any(
    "libcudart.so.13" in path or ("libcudart" in path and "/CUDA/13" in path)
    for path in payload["loaded_libraries"]
)
print(json.dumps(payload, sort_keys=True), flush=True)
if not payload.get("cuda_available") or int(payload.get("cuda_device_count") or 0) < 1:
    raise SystemExit(42)
if payload.get("bad_cuda13_runtime"):
    raise SystemExit(43)
if payload.get("transformer_engine_pytorch_import") != "ok":
    raise SystemExit(44)
""".lstrip(),
        encoding="utf-8",
    )
    preflight_python = str(args.python_executable) if args.python_executable else sys.executable
    preflight_cmd = [preflight_python, str(script_path)]
    (output_dir / "cuda_preflight_command.json").write_text(
        json.dumps({"cmd": preflight_cmd, "nnodes": nnodes, "srun_args": srun_args}, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = output_dir / "cuda_preflight.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        if nnodes > 1:
            inner = " ".join(shlex.quote(item) for item in preflight_cmd)
            inner = f"cd {shlex.quote(str(bridge_root))} && {inner}"
            inner = _with_runtime_env(inner, env)
            subprocess.run(
                _srun_cmd(nnodes, srun_args, inner),
                check=True,
                cwd=str(bridge_root),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        else:
            subprocess.run(
                preflight_cmd,
                check=True,
                cwd=str(bridge_root),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dataset", default="llm-pretrain-mock")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--hf-path", default="")
    parser.add_argument("--launcher", choices=["uv", "python"], default="python")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--preflight-cuda", action="store_true")
    parser.add_argument("--srun-arg", action="append", default=[])
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--nsys-profile", action="store_true")
    parser.add_argument("--nsys-bin", default="")
    parser.add_argument("--nsys-output-dir", default="")
    parser.add_argument("--nsys-trace", default="cuda,nvtx,osrt,cublas,cudnn")
    parser.add_argument("--nsys-extra-arg", action="append", default=[])
    parser.add_argument("--expected-end-step", type=int, default=0)
    parser.add_argument("--expected-consumed-samples", type=int, default=0)
    parser.add_argument("--require-full-state-checkpoint", action="store_true")
    parser.add_argument("--resume-checkpoint-root", default="", help="Checkpoint root for exact full-state resume.")
    parser.add_argument("--resume-checkpoint-step", type=int, default=0, help="Exact checkpoint step to resume.")
    parser.add_argument("--resume-checkpoint-consumed-samples", type=int, default=0, help="Cumulative samples at the resume step.")
    args = parser.parse_args()

    bridge_root = Path(args.bridge_root).resolve()
    repo_root = find_repo_root(Path(__file__))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A previous interrupted or obsolete attempt must never make this attempt
    # look complete.  Verified completed segments are skipped by segment_runner
    # before this wrapper is invoked.
    (output_dir / ".training_complete").unlink(missing_ok=True)
    (output_dir / "checkpoint_state_audit.json").unlink(missing_ok=True)

    expected_resume = None
    if args.resume_checkpoint_step and not args.resume_checkpoint_root:
        raise SystemExit("--resume-checkpoint-step requires --resume-checkpoint-root")
    if args.resume_checkpoint_consumed_samples and not args.resume_checkpoint_root:
        raise SystemExit(
            "--resume-checkpoint-consumed-samples requires --resume-checkpoint-root"
        )
    if args.resume_checkpoint_root:
        if int(args.resume_checkpoint_step) <= 0:
            raise SystemExit("--resume-checkpoint-root requires positive --resume-checkpoint-step")
        resume_root = Path(args.resume_checkpoint_root).expanduser().resolve()
        resume_step = int(args.resume_checkpoint_step)
        resume_samples = int(args.resume_checkpoint_consumed_samples or 0)
        if resume_samples <= 0:
            resume_samples = resume_step * int(args.global_batch_size)
        resume_state = inspect_full_state_checkpoint(
            checkpoint_root=resume_root,
            checkpoint_step=resume_step,
            expected_step=resume_step,
            expected_consumed_samples=resume_samples,
        )
        write_checkpoint_audit(output_dir / "requested_resume_checkpoint_audit.json", resume_state)
        if not resume_state.valid_full_state:
            raise SystemExit(f"invalid requested full-state resume: {resume_state.errors}")
        expected_resume = (resume_step, resume_samples)
        _set_full_state_resume_overrides(
            args,
            checkpoint_root=resume_root,
            checkpoint_step=resume_step,
        )

    _normalize_checkpoint_save_override(args, output_dir)

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
            str(args.recipe),
            "--dataset",
            str(args.dataset),
            "--seq_length",
            str(int(args.seq_length)),
        ]
    )
    if args.hf_path:
        cmd.extend(["--hf_path", str(args.hf_path)])
    if expected_resume is not None:
        cmd.extend([
            "--expected-resume-step", str(expected_resume[0]),
            "--expected-resume-consumed-samples", str(expected_resume[1]),
            "--resume-state-receipt", str(output_dir / "restored_resume_state.json"),
        ])
    cmd.extend(
        [
            f"train.train_iters={int(args.train_iters)}",
            f"train.global_batch_size={int(args.global_batch_size)}",
            f"train.micro_batch_size={int(args.micro_batch_size)}",
            f"validation.eval_interval={int(args.eval_interval)}",
            f"validation.eval_iters={int(args.eval_iters)}",
            "scheduler.lr_warmup_iters=1",
            f"scheduler.lr_decay_iters={max(1, int(args.train_iters))}",
            f"logger.log_interval={int(args.log_interval)}",
            f"checkpoint.save={output_dir / 'checkpoints'}",
            f"checkpoint.save_interval={max(1, int(args.save_interval or args.train_iters))}",
        ]
    )
    if str(os.environ.get("BRIDGE_LOG_THROUGHPUT", "")).strip() in {"1", "true", "True", "yes"}:
        cmd.append("logger.log_throughput=true")
        cmd.append("logger.log_throughput_to_tensorboard=false")
        window_size = str(os.environ.get("BRIDGE_THROUGHPUT_WINDOW_SIZE", "")).strip()
        if window_size:
            cmd.append(f"logger.throughput_window_size={int(window_size)}")
    if args.dataset == "llm-pretrain":
        manifest_path = Path(args.data_manifest).resolve() if args.data_manifest else output_dir / "missing_manifest.json"
        manifest = _load_manifest(manifest_path)
        json_overrides, uses_physical_splits = _dataset_overrides_for_manifest(
            manifest_path,
            manifest,
            output_dir,
            eval_iters=int(args.eval_iters),
        )
        if uses_physical_splits:
            _write_validation_coverage(output_dir, manifest_path, manifest)
        overrides_path = output_dir / "bridge_overrides.json"
        overrides_path.write_text(json.dumps(json_overrides, indent=2) + "\n", encoding="utf-8")
        cmd.extend(["--overrides-json", str(overrides_path)])
    cmd.extend(str(item) for item in args.extra_override)
    cmd = _with_nsys_profile(cmd, args, output_dir)

    env = dict(os.environ)
    env["NEMO_EXPERIMENTS_DIR"] = str(output_dir / "nemo_experiments")
    pybind11_shim_root = _prepare_pybind11_shim(output_dir, env)
    pythonpath_entries = [
        str(repo_root / "src"),
        str(repo_root),
        str(bridge_root / "src"),
    ]
    mcore_override = str(env.get("PROOF_MCORE_OVERRIDE_PATH", "")).strip()
    if mcore_override:
        pythonpath_entries.append(str(Path(mcore_override).expanduser().resolve()))
    pythonpath_entries.append(str(bridge_root / "3rdparty" / "Megatron-LM"))
    if pybind11_shim_root is not None:
        pythonpath_entries.insert(0, str(pybind11_shim_root))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    _prepend_bridge_venv_cuda_libs(env, bridge_root)
    python_env_lib_dirs = _prepend_python_env_cuda_libs(env, str(args.python_executable))
    srun_args = [str(item) for item in args.srun_arg if str(item).strip()]

    started = time.time()
    expected_tokens = int(args.train_iters) * int(args.global_batch_size) * int(args.seq_length)
    (output_dir / "bridge_command.json").write_text(
        json.dumps({"cmd": cmd, "cwd": str(bridge_root)}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "bridge_invocation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cmd": cmd,
                "cwd": str(bridge_root),
                "dataset": args.dataset,
                "recipe": args.recipe,
                "seq_length": int(args.seq_length),
                "train_iters": int(args.train_iters),
                "global_batch_size": int(args.global_batch_size),
                "micro_batch_size": int(args.micro_batch_size),
                "expected_tokens": expected_tokens,
                "nnodes": nnodes,
                "nproc_per_node": int(args.nproc_per_node),
                "nsys_profile": bool(args.nsys_profile),
                "nsys_bin": str(args.nsys_bin or os.environ.get("NSYS_BIN", "")),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_environment.json").write_text(
        json.dumps(
            {
                "python_executable": str(args.python_executable),
                "python_env_prefix": env.get("PROOF_PYTHON_ENV_PREFIX", ""),
                "python_env_cuda_lib_dirs": [str(path) for path in python_env_lib_dirs],
                "ld_library_path": env.get("LD_LIBRARY_PATH", ""),
                "nvte_cuda_include_path": env.get("NVTE_CUDA_INCLUDE_PATH", ""),
                "runtime_exports": _export_runtime_env(env).split("; ") if _export_runtime_env(env) else [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        if args.preflight_cuda:
            _run_cuda_preflight(
                args=args,
                bridge_root=bridge_root,
                output_dir=output_dir,
                env=env,
                nnodes=nnodes,
                srun_args=srun_args,
            )
        if nnodes > 1:
            env["MASTER_ADDR"] = master_addr
            env["MASTER_PORT"] = master_port
            inner_cmd = " ".join(
                item if item.startswith("--node_rank=${") else shlex.quote(item)
                for item in cmd
            )
            inner_cmd = f"cd {shlex.quote(str(bridge_root))} && {inner_cmd}"
            inner_cmd = _with_runtime_env(inner_cmd, env)
            srun_cmd = _srun_cmd(nnodes, srun_args, inner_cmd)
            subprocess.run(srun_cmd, check=True, cwd=str(bridge_root), env=env)
        else:
            subprocess.run(cmd, check=True, cwd=str(bridge_root), env=env)
    except subprocess.CalledProcessError as exc:
        runtime = time.time() - started
        _write_wrapper_artifacts(
            output_dir=output_dir,
            args=args,
            status="failed",
            runtime=runtime,
            error=str(exc),
            return_code=exc.returncode,
        )
        raise
    except Exception as exc:
        runtime = time.time() - started
        _write_wrapper_artifacts(
            output_dir=output_dir,
            args=args,
            status="failed",
            runtime=runtime,
            error=repr(exc),
        )
        raise
    runtime = time.time() - started
    if bool(args.require_full_state_checkpoint):
        if int(args.expected_end_step) <= 0 or int(args.expected_consumed_samples) <= 0:
            raise SystemExit(
                "--require-full-state-checkpoint requires positive --expected-end-step "
                "and --expected-consumed-samples"
            )
        checkpoint_state = inspect_full_state_checkpoint(
            checkpoint_root=_checkpoint_root_from_overrides(args, output_dir),
            expected_step=int(args.expected_end_step),
            expected_consumed_samples=int(args.expected_consumed_samples),
        )
        write_checkpoint_audit(output_dir / "checkpoint_state_audit.json", checkpoint_state)
        if not checkpoint_state.valid_full_state:
            error = "invalid full-state checkpoint: " + ", ".join(checkpoint_state.errors)
            _write_wrapper_artifacts(
                output_dir=output_dir,
                args=args,
                status="invalid",
                runtime=runtime,
                error=error,
                return_code=2,
            )
            raise SystemExit(error)
    _write_wrapper_artifacts(output_dir=output_dir, args=args, status="complete", runtime=runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
