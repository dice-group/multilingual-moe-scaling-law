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

"""Bridge pretraining and checkpoint-validation runner for study-local recipes."""

from __future__ import annotations

import argparse
import ast
import faulthandler
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import typing
import types
from collections.abc import Callable
from pathlib import Path


def _isolate_triton_cache_for_local_rank() -> None:
    """Avoid concurrent Triton cache writes on shared cluster home filesystems."""
    if os.environ.get("PROOF_ISOLATE_TRITON_CACHE", "").strip() != "1":
        return
    if os.environ.get("TRITON_CACHE_DIR"):
        return
    job_id = os.environ.get("SLURM_JOB_ID", "no_job")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "megatron_bridge_triton" / f"job_{job_id}" / f"local_rank_{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


_isolate_triton_cache_for_local_rank()

from multilingual_megatron.repo import find_repo_root

_REPO_ROOT = find_repo_root(Path(__file__))
_MCORE_OVERRIDE_PATH = os.environ.get("PROOF_MCORE_OVERRIDE_PATH")
_PATH_BOOTSTRAP = [
    _REPO_ROOT,
    _REPO_ROOT / "external" / "Megatron-Bridge" / "src",
]
if _MCORE_OVERRIDE_PATH:
    _PATH_BOOTSTRAP.append(Path(_MCORE_OVERRIDE_PATH).expanduser().resolve())
_PATH_BOOTSTRAP.append(_REPO_ROOT / "external" / "Megatron-Bridge" / "3rdparty" / "Megatron-LM")
for _path in reversed(_PATH_BOOTSTRAP):
    if _path.exists():
        # Snapshot symlinks and launcher PYTHONPATH entries can name the same
        # engine differently. Reapply priority even for an existing entry so
        # the selected MCore override always precedes the bundled version.
        _entry = str(_path.resolve())
        while _entry in sys.path:
            sys.path.remove(_entry)
        sys.path.insert(0, _entry)

from .cpu_affinity import maybe_bind_local_rank_cpus
from .megatron_training_import_shim import (
    install_bridge_mamba_model_import_shim,
    install_bridge_mimo_model_import_shim,
    install_bridge_packed_sequence_specs_import_shim,
    install_bridge_pretraining_dataset_import_shims,
    install_megatron_training_config_import_shim,
    install_mcore_cuda_graphs_import_shim,
)
from .runtime_patches import (
    install_bridge_cp_rank_compat_patch,
    install_ddp_default_stream_alignment_patch,
    install_fused_te_moe_act_checkpoint_patch,
    install_mtp_full_recompute_padding_mask_compat_patch,
    install_moe_allgather_ep1_topk1_fastpath,
    install_moe_alltoall_batched_dispatch_patch,
    install_moe_alltoall_cpu_split_list_patch,
    install_moe_alltoall_nccl_stream_patch,
    install_moe_alltoall_padded_equal_splits_patch,
    install_moe_alltoall_sync_profiler,
    install_p2p_batch_sync_disabled_guard,
    install_te_grouped_mlp_inspector,
)

if not hasattr(typing, "override"):
    try:
        from typing_extensions import override as _typing_override
    except ImportError:  # pragma: no cover - only needed on older fallback envs.
        def _typing_override(func):
            return func

    typing.override = _typing_override

_ENABLE_OPTIONAL_DEP_SHIMS = str(os.environ.get("PROOF_ENABLE_BRIDGE_OPTIONAL_DEP_SHIMS", "")).strip() == "1"
_REQUIRE_REAL_MODELOPT = str(os.environ.get("PROOF_REQUIRE_REAL_MODELOPT", "")).strip() == "1"
_TRACE_IMPORTS = str(os.environ.get("PROOF_TRACE_BRIDGE_IMPORTS", "")).strip() == "1"


def _trace_import(message: str) -> None:
    if _TRACE_IMPORTS:
        print(f"[BRIDGE_IMPORT] {message}", flush=True)


def _cancel_import_trace_timer() -> None:
    if _TRACE_IMPORTS:
        faulthandler.cancel_dump_traceback_later()


if _TRACE_IMPORTS:
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)

maybe_bind_local_rank_cpus()

_REAL_NVRX_AVAILABLE = False
_NVRX_MODE = "not_requested"
if _ENABLE_OPTIONAL_DEP_SHIMS and importlib.util.find_spec("nvidia_resiliency_ext") is not None:
    nvrx_module = importlib.import_module("nvidia_resiliency_ext")
    if not hasattr(nvrx_module, "__version__"):
        try:
            nvrx_module.__version__ = importlib.metadata.version("nvidia-resiliency-ext")
        except importlib.metadata.PackageNotFoundError:
            nvrx_module.__version__ = importlib.metadata.version("nvidia_resiliency_ext")
    # MCore 0.15 requires NVRx >=0.6 for async checkpointing, while the
    # production environment intentionally uses synchronous saves and carries
    # NVRx 0.4 for in-process resilience.  Keep the real in-process API and
    # make only the unsupported async-checkpoint modules unavailable so MCore
    # selects its normal synchronous strategy.  Never claim a false package
    # version or replace the real CallWrapper.
    from packaging.version import Version as _PackageVersion

    if _PackageVersion(str(nvrx_module.__version__)) < _PackageVersion("0.6.0"):
        importlib.import_module("nvidia_resiliency_ext.inprocess")
        for _name in (
            "nvidia_resiliency_ext.checkpointing.async_ckpt.core",
            "nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader",
            "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async",
            "nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver",
        ):
            sys.modules[_name] = None
        _NVRX_MODE = (
            f"real_inprocess_async_disabled_incompatible_{nvrx_module.__version__}"
        )
    else:
        importlib.import_module("nvidia_resiliency_ext.inprocess")
        _NVRX_MODE = f"real_{nvrx_module.__version__}"
    _REAL_NVRX_AVAILABLE = True

_REAL_MODELOPT_AVAILABLE = False
_MODELOPT_MODE = "not_requested"
if _ENABLE_OPTIONAL_DEP_SHIMS and importlib.util.find_spec("modelopt") is not None:
    # Never shadow a real ModelOpt install: its checkpoint hooks are part of
    # full-state production correctness.  An installed-but-broken package is a
    # hard import error rather than a reason to inject no-op state helpers.
    importlib.import_module("modelopt.torch.distill")
    importlib.import_module("modelopt.torch.opt.plugins")
    _REAL_MODELOPT_AVAILABLE = True
    _MODELOPT_MODE = f"real_{importlib.metadata.version('nvidia-modelopt')}"

if _REQUIRE_REAL_MODELOPT and not _REAL_MODELOPT_AVAILABLE:
    raise ImportError(
        "PROOF_REQUIRE_REAL_MODELOPT=1 but a working real nvidia-modelopt package "
        "was not imported; refusing to install no-op checkpoint hooks"
    )

if _ENABLE_OPTIONAL_DEP_SHIMS and not _REAL_MODELOPT_AVAILABLE and "modelopt.torch.distill" not in sys.modules:
    modelopt_module = types.ModuleType("modelopt")
    modelopt_torch_module = types.ModuleType("modelopt.torch")
    modelopt_distill_module = types.ModuleType("modelopt.torch.distill")
    modelopt_plugins_module = types.ModuleType("modelopt.torch.distill.plugins")
    modelopt_megatron_module = types.ModuleType("modelopt.torch.distill.plugins.megatron")
    modelopt_opt_module = types.ModuleType("modelopt.torch.opt")
    modelopt_opt_plugins_module = types.ModuleType("modelopt.torch.opt.plugins")
    modelopt_torch_module.__path__ = []
    modelopt_distill_module.__path__ = []
    modelopt_plugins_module.__path__ = []
    modelopt_opt_module.__path__ = []

    class _UnavailableDistillationModel:
        pass

    class _UnavailableDistillationConfig:
        pass

    def _no_tensor_shape_adjust(*args, **kwargs):
        return None

    def _no_modelopt_state(*args, **kwargs):
        return None

    modelopt_distill_module.DistillationModel = _UnavailableDistillationModel
    modelopt_megatron_module.DistillationConfig = _UnavailableDistillationConfig
    modelopt_megatron_module.get_tensor_shapes_adjust_fn_for_distillation = _no_tensor_shape_adjust
    modelopt_plugins_module.megatron = modelopt_megatron_module
    modelopt_distill_module.plugins = modelopt_plugins_module
    modelopt_opt_plugins_module.restore_modelopt_state = _no_modelopt_state
    modelopt_opt_plugins_module.save_modelopt_state = _no_modelopt_state
    modelopt_opt_plugins_module.save_sharded_modelopt_state = _no_modelopt_state
    modelopt_opt_module.plugins = modelopt_opt_plugins_module
    modelopt_torch_module.distill = modelopt_distill_module
    modelopt_torch_module.opt = modelopt_opt_module
    modelopt_module.torch = modelopt_torch_module
    sys.modules.setdefault("modelopt", modelopt_module)
    sys.modules.setdefault("modelopt.torch", modelopt_torch_module)
    sys.modules.setdefault("modelopt.torch.distill", modelopt_distill_module)
    sys.modules.setdefault("modelopt.torch.distill.plugins", modelopt_plugins_module)
    sys.modules.setdefault("modelopt.torch.distill.plugins.megatron", modelopt_megatron_module)
    sys.modules.setdefault("modelopt.torch.opt", modelopt_opt_module)
    sys.modules.setdefault("modelopt.torch.opt.plugins", modelopt_opt_plugins_module)
    _MODELOPT_MODE = "disabled_missing_package_shim"

if _ENABLE_OPTIONAL_DEP_SHIMS:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        tensorboard_module = types.ModuleType("torch.utils.tensorboard")
        tensorboard_writer_module = types.ModuleType("torch.utils.tensorboard.writer")

        class _NoOpSummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def add_text(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        tensorboard_writer_module.SummaryWriter = _NoOpSummaryWriter
        tensorboard_module.SummaryWriter = _NoOpSummaryWriter
        tensorboard_module.writer = tensorboard_writer_module
        sys.modules.setdefault("torch.utils.tensorboard", tensorboard_module)
        sys.modules.setdefault("torch.utils.tensorboard.writer", tensorboard_writer_module)

if _ENABLE_OPTIONAL_DEP_SHIMS and not _REAL_NVRX_AVAILABLE and "nvidia_resiliency_ext.inprocess" not in sys.modules:
    if importlib.util.find_spec("nvidia_resiliency_ext") is None:
        nvrx_module = types.ModuleType("nvidia_resiliency_ext")
        nvrx_module.__version__ = "0.0.0"
        nvrx_inprocess_module = types.ModuleType("nvidia_resiliency_ext.inprocess")

        class _UnavailableCallWrapper:
            iteration = 0

        nvrx_inprocess_module.CallWrapper = _UnavailableCallWrapper
        nvrx_module.inprocess = nvrx_inprocess_module
        sys.modules.setdefault("nvidia_resiliency_ext", nvrx_module)
        sys.modules.setdefault("nvidia_resiliency_ext.inprocess", nvrx_inprocess_module)
        _NVRX_MODE = "disabled_missing_package_shim"

_OPTIONAL_DEPENDENCY_PROVENANCE = {
    "compatibility_mode_enabled": _ENABLE_OPTIONAL_DEP_SHIMS,
    "real_modelopt_required": _REQUIRE_REAL_MODELOPT,
    "modelopt_mode": _MODELOPT_MODE,
    "modelopt_file": (
        str(getattr(sys.modules.get("modelopt"), "__file__", ""))
        if _REAL_MODELOPT_AVAILABLE
        else ""
    ),
    "nvrx_mode": _NVRX_MODE,
    "nvrx_inprocess_available": bool(
        _REAL_NVRX_AVAILABLE
        and sys.modules.get("nvidia_resiliency_ext.inprocess") is not None
    ),
    "nvrx_async_checkpoint_available": bool(
        _REAL_NVRX_AVAILABLE
        and sys.modules.get("nvidia_resiliency_ext.checkpointing.async_ckpt.core")
        is not None
    ),
    "checkpoint_async_save_required": False,
    "nvrx_inprocess_restart_enabled": False,
    "nvrx_straggler_enabled": False,
}
if str(os.environ.get("RANK", "0")).strip() in {"", "0"}:
    print(
        "[bridge_optional_dependencies] "
        + json.dumps(_OPTIONAL_DEPENDENCY_PROVENANCE, sort_keys=True),
        flush=True,
    )

if str(os.environ.get("PROOF_DISABLE_MEGATRON_JIT_FUSER", "")).strip() == "1":
    import torch

    def _noop_compile(model=None, *args, **kwargs):
        if model is None:
            return lambda fn: fn
        return model

    torch.compile = _noop_compile
    from megatron.core.jit import disable_jit_fuser

    disable_jit_fuser()

_trace_import("install startup shims begin")
install_megatron_training_config_import_shim()
install_bridge_packed_sequence_specs_import_shim()
install_bridge_pretraining_dataset_import_shims()
install_bridge_mamba_model_import_shim()
install_bridge_mimo_model_import_shim()
install_mcore_cuda_graphs_import_shim()
_trace_import("install startup shims done")

_trace_import("import megatron.bridge.training.config begin")
from megatron.bridge.training.config import ConfigContainer, GPTDatasetConfig, MockGPTDatasetConfig
_trace_import("import megatron.bridge.training.config done")
_trace_import("import megatron.bridge.training.mixed_precision begin")
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
_trace_import("import megatron.bridge.training.mixed_precision done")
_trace_import("import megatron.bridge.training.gpt_step begin")
from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step
_trace_import("import megatron.bridge.training.gpt_step done")
_trace_import("import megatron.bridge.training.pretrain begin")
from megatron.bridge.training.pretrain import pretrain
_trace_import("import megatron.bridge.training.pretrain done")
_trace_import("import multilingual_megatron.pipeline.backends.megatron_bridge.study_bridge_recipes begin")
from . import study_bridge_recipes
_trace_import("import multilingual_megatron.pipeline.backends.megatron_bridge.study_bridge_recipes done")
_cancel_import_trace_timer()


DATASET_TYPES = ["llm-pretrain", "llm-pretrain-mock"]
STEP_FUNCTIONS: dict[str, Callable] = {"gpt_step": gpt_forward_step}


def _patch_cuda_bool_argsort() -> None:
    if str(os.environ.get("PROOF_ENABLE_CUDA_BOOL_ARGSORT_PATCH", "")).strip() != "1":
        return
    try:
        import torch
    except ImportError:
        return

    if getattr(torch.Tensor.argsort, "_study_bool_cuda_patch", False):
        return

    original_argsort = torch.Tensor.argsort

    def _argsort(self, *args, **kwargs):
        # Megatron Core MoE dispatch sorts a boolean CUDA routing map. This
        # PyTorch build rejects bool CUDA sort, while int8 sort is equivalent
        # for the 0/1 routing map and keeps the compatibility patch process-local.
        if self.dtype is torch.bool and self.is_cuda:
            return original_argsort(self.to(dtype=torch.int8), *args, **kwargs)
        return original_argsort(self, *args, **kwargs)

    _argsort._study_bool_cuda_patch = True
    torch.Tensor.argsort = _argsort


def _resolve_seq_length(config: ConfigContainer, seq_length: int | None) -> int:
    if seq_length is not None:
        return seq_length
    if hasattr(config, "model") and config.model is not None and hasattr(config.model, "seq_length"):
        return int(config.model.seq_length)
    return 4096


def _apply_dataset_override(
    config: ConfigContainer,
    dataset_type: str,
    seq_length: int | None,
) -> ConfigContainer:
    resolved_seq_length = _resolve_seq_length(config, seq_length)
    common_kwargs = {
        "seq_length": resolved_seq_length,
        "random_seed": 1234,
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "num_dataset_builder_threads": int(os.environ.get("PROOF_DATASET_BUILDER_THREADS", "16")),
        "split": "9999,8,2",
        "data_sharding": True,
        "dataloader_type": "single",
        "skip_getting_attention_mask_from_dataset": True,
    }
    if dataset_type == "llm-pretrain":
        config.dataset = GPTDatasetConfig(blend=None, blend_per_split=None, **common_kwargs)
    elif dataset_type == "llm-pretrain-mock":
        config.dataset = MockGPTDatasetConfig(**common_kwargs)
    else:
        raise ValueError(f"Unsupported dataset type for this runner: {dataset_type}")

    if seq_length is not None and hasattr(config, "model") and config.model is not None:
        config.model.seq_length = seq_length
    return config


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dataset", choices=DATASET_TYPES, default="llm-pretrain")
    parser.add_argument("--step_func", choices=sorted(STEP_FUNCTIONS), default="gpt_step")
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--hf_path", default=None)
    parser.add_argument("--peft_scheme", default=None)
    parser.add_argument("--packed_sequence", action="store_true", default=False)
    parser.add_argument("--overrides-json", action="append", default=[])
    parser.add_argument("--expected-resume-step", type=int, default=0)
    parser.add_argument("--expected-resume-consumed-samples", type=int, default=0)
    parser.add_argument("--resume-state-receipt", default="")
    parser.add_argument(
        "--per-language-validation-spec",
        default="",
        help=(
            "Run the load-once A/B-v2 per-language checkpoint evaluator instead "
            "of training. The JSON spec freezes checkpoint, language coverage, "
            "validation prefixes, tokenizer, and unigram references."
        ),
    )
    parser.add_argument(
        "--native-benchmark-score-spec",
        default="",
        help=(
            "Run inference-only native Megatron scoring for a frozen lm-eval "
            "request manifest instead of training."
        ),
    )
    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def _recipe_builder(recipe_name: str) -> Callable:
    if hasattr(study_bridge_recipes, recipe_name):
        return getattr(study_bridge_recipes, recipe_name)
    import megatron.bridge.recipes as bridge_recipes

    if hasattr(bridge_recipes, recipe_name):
        return getattr(bridge_recipes, recipe_name)
    raise AttributeError(
        f"Recipe '{recipe_name}' not found in study_bridge_recipes or megatron.bridge.recipes."
    )


def _dtype_from_string(raw: str):
    try:
        import torch
    except ImportError:
        return raw

    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "torch.float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
    }
    return aliases.get(raw.lower(), raw)


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_unquoted_list(value: str) -> list[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return [value]
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip("'\"") for item in _split_top_level(inner)]


def _parse_blend_value(raw: str) -> tuple[list[str], list[float] | None]:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        groups = _split_top_level(raw[1:-1].strip()) if raw.startswith("[") and raw.endswith("]") else []
        if not groups:
            raise ValueError(f"Invalid dataset.blend override: {raw}")
        prefixes = _parse_unquoted_list(groups[0])
        weights = None
        if len(groups) > 1 and groups[1].lower() != "null":
            weights = [float(item) for item in _parse_unquoted_list(groups[1])]
        return prefixes, weights

    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(f"Invalid dataset.blend override: {raw}")
    prefixes = [str(item) for item in parsed[0]]
    weights = None
    if len(parsed) > 1 and parsed[1] is not None:
        weights = [float(item) for item in parsed[1]]
    return prefixes, weights


def _parse_override_value(key: str, raw: str):
    lower = raw.lower()
    if lower == "null":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if key == "dataset.blend":
        return _parse_blend_value(raw)
    if key == "dataset.split":
        return raw
    if (
        key.endswith("params_dtype")
        or key.endswith("pipeline_dtype")
        or key
        in {
            "optimizer.main_grads_dtype",
            "optimizer.main_params_dtype",
            "optimizer.exp_avg_dtype",
            "optimizer.exp_avg_sq_dtype",
        }
    ):
        return _dtype_from_string(raw)
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _set_nested_attr(config: ConfigContainer, key: str, value) -> None:
    target = config
    fields = key.lstrip("+").split(".")
    for field in fields[:-1]:
        if isinstance(target, dict):
            target = target[field]
        else:
            if field == "mixed_precision" and isinstance(getattr(target, field, None), str):
                setattr(target, field, get_mixed_precision_config(getattr(target, field)))
            target = getattr(target, field)
    leaf = fields[-1]
    if isinstance(target, dict):
        target[leaf] = value
    else:
        setattr(target, leaf, value)


def _apply_cli_overrides(config: ConfigContainer, cli_overrides: list[str] | None) -> ConfigContainer:
    if not cli_overrides:
        return config
    for override in cli_overrides:
        if "=" not in override:
            raise ValueError(f"Unsupported override without '=': {override}")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        _set_nested_attr(config, key, _parse_override_value(key, raw_value.strip()))
    return config


def _normalize_json_override(key: str, value):
    if key == "dataset.blend" and isinstance(value, list) and value:
        prefixes = [str(item) for item in value[0]]
        weights = None
        if len(value) > 1 and value[1] is not None:
            weights = [float(item) for item in value[1]]
        return prefixes, weights
    if isinstance(value, str) and (key.endswith("params_dtype") or key.endswith("pipeline_dtype")):
        return _dtype_from_string(value)
    return value


def _apply_json_overrides(config: ConfigContainer, override_paths: list[str] | None) -> ConfigContainer:
    if not override_paths:
        return config
    for raw_path in override_paths:
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        with open(path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError(f"JSON override file must contain an object: {path}")
        for key, value in overrides.items():
            _set_nested_attr(config, str(key), _normalize_json_override(str(key), value))
    return config


def _load_recipe(
    recipe_name: str,
    peft_scheme: str | None,
    packed_sequence: bool,
    seq_length: int | None,
    hf_path: str | None,
) -> ConfigContainer:
    config_builder = _recipe_builder(recipe_name)
    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {}
        if "peft" in params or has_var_keyword:
            kwargs["peft"] = peft_scheme
        if packed_sequence and ("packed_sequence" in params or has_var_keyword):
            kwargs["packed_sequence"] = packed_sequence
        if seq_length is not None and ("seq_length" in params or has_var_keyword):
            kwargs["seq_length"] = seq_length
        if hf_path is not None and ("hf_path" in params or has_var_keyword):
            kwargs["hf_path"] = hf_path
        return config_builder(**kwargs)
    except TypeError:
        return config_builder()


def main() -> None:
    args, cli_overrides = _parse_args()
    evaluation_modes = (
        args.per_language_validation_spec,
        args.native_benchmark_score_spec,
    )
    if sum(bool(value) for value in evaluation_modes) > 1:
        raise ValueError("checkpoint evaluation modes are mutually exclusive")
    if args.per_language_validation_spec:
        from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import (
            load_evaluation_spec,
        )

        evaluation_spec = load_evaluation_spec(Path(args.per_language_validation_spec))
        if args.recipe != evaluation_spec.recipe_name:
            raise ValueError(
                f"evaluation recipe {args.recipe!r} does not match frozen spec "
                f"{evaluation_spec.recipe_name!r}"
            )
        if tuple(cli_overrides) != evaluation_spec.recipe_overrides:
            raise ValueError(
                "evaluation CLI overrides must exactly match model_recipe.overrides "
                "in the frozen spec"
            )
    if args.native_benchmark_score_spec:
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import (
            load_native_benchmark_score_spec,
        )

        benchmark_spec = load_native_benchmark_score_spec(
            Path(args.native_benchmark_score_spec)
        )
        if args.recipe != benchmark_spec.recipe_name:
            raise ValueError(
                f"benchmark recipe {args.recipe!r} does not match frozen spec "
                f"{benchmark_spec.recipe_name!r}"
            )
        if tuple(cli_overrides) != benchmark_spec.recipe_overrides:
            raise ValueError(
                "benchmark CLI overrides must exactly match model_recipe.overrides "
                "in the frozen spec"
            )
    config = _load_recipe(
        args.recipe,
        peft_scheme=args.peft_scheme,
        packed_sequence=args.packed_sequence,
        seq_length=args.seq_length,
        hf_path=args.hf_path,
    )
    config = _apply_dataset_override(config, args.dataset, args.seq_length)
    config = _apply_cli_overrides(config, cli_overrides or None)
    config = _apply_json_overrides(config, args.overrides_json or None)
    has_model = hasattr(config, "model") and config.model is not None
    has_dataset = hasattr(config, "dataset") and config.dataset is not None
    if has_model and has_dataset:
        if hasattr(config.dataset, "seq_length") and config.model.seq_length != config.dataset.seq_length:
            config.model.seq_length = config.dataset.seq_length
    _patch_cuda_bool_argsort()
    install_bridge_cp_rank_compat_patch()
    install_ddp_default_stream_alignment_patch()
    install_moe_allgather_ep1_topk1_fastpath()
    install_moe_alltoall_batched_dispatch_patch()
    install_moe_alltoall_padded_equal_splits_patch()
    install_moe_alltoall_nccl_stream_patch()
    install_moe_alltoall_cpu_split_list_patch()
    install_moe_alltoall_sync_profiler()
    install_mtp_full_recompute_padding_mask_compat_patch()
    install_fused_te_moe_act_checkpoint_patch()
    install_p2p_batch_sync_disabled_guard()
    install_te_grouped_mlp_inspector()
    if args.per_language_validation_spec:
        from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import (
            run_per_language_checkpoint_evaluation,
        )

        run_per_language_checkpoint_evaluation(
            config=config,
            forward_step_func=STEP_FUNCTIONS[args.step_func],
            spec_path=Path(args.per_language_validation_spec),
        )
    elif args.native_benchmark_score_spec:
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_runner import (
            run_native_benchmark_scoring,
        )

        run_native_benchmark_scoring(
            config=config,
            spec_path=Path(args.native_benchmark_score_spec),
        )
    else:
        callbacks = None
        if args.expected_resume_step or args.expected_resume_consumed_samples or args.resume_state_receipt:
            from functools import partial
            from megatron.bridge.training.callbacks import CallbackManager
            from .resume_guard import verify_restored_resume

            if not (args.expected_resume_step > 0 and args.expected_resume_consumed_samples > 0 and args.resume_state_receipt):
                raise ValueError("resume guard requires positive step, sample count, and receipt path")
            callbacks = CallbackManager()
            callbacks.register("on_train_start", partial(
                verify_restored_resume,
                expected_step=args.expected_resume_step,
                expected_samples=args.expected_resume_consumed_samples,
                receipt_path=Path(args.resume_state_receipt),
            ))
        pretrain(config=config, forward_step_func=STEP_FUNCTIONS[args.step_func], callbacks=callbacks)


if __name__ == "__main__":
    main()
