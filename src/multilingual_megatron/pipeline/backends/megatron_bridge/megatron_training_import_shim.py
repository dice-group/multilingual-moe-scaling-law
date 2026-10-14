"""Import helpers for Megatron-LM training submodules.

Megatron-LM's ``megatron.training`` package currently imports the full legacy
training stack from its ``__init__``. Bridge config code only needs
``megatron.training.config``, so importing the parent package eagerly can turn a
cheap dataclass import into a heavyweight startup path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def install_megatron_training_config_import_shim() -> None:
    """Allow ``megatron.training.config`` without executing training ``__init__``.

    The shim keeps the real Megatron-LM ``megatron/training`` directory on the
    package search path, so submodule imports still resolve to upstream files.
    If code later requests convenience exports from ``megatron.training`` itself,
    the real ``__init__`` is loaded lazily at that point.
    """
    if str(os.environ.get("PROOF_DISABLE_MEGATRON_TRAINING_IMPORT_SHIM", "")).strip() == "1":
        return
    if "megatron.training" in sys.modules:
        return

    training_paths = _find_megatron_training_paths()
    if not training_paths:
        return

    import megatron

    module = types.ModuleType("megatron.training")
    module.__file__ = str(training_paths[0] / "__init__.py")
    module.__package__ = "megatron.training"
    module.__path__ = [str(path) for path in training_paths]
    module.__loader__ = None
    spec = importlib.util.spec_from_loader(
        "megatron.training",
        loader=None,
        is_package=True,
    )
    if spec is not None:
        spec.submodule_search_locations = list(module.__path__)
        module.__spec__ = spec

    module.__getattr__ = _lazy_real_training_attr(module, training_paths)  # type: ignore[attr-defined]
    sys.modules["megatron.training"] = module
    setattr(megatron, "training", module)

    _install_lightweight_training_config_package(training_paths)
    _install_lightweight_training_utils_module()


def _find_megatron_training_paths() -> list[Path]:
    import megatron

    paths: list[Path] = []
    for raw_path in getattr(megatron, "__path__", []):
        candidate = Path(raw_path) / "training"
        if (candidate / "config" / "__init__.py").exists():
            paths.append(candidate)
    return paths


def _lazy_real_training_attr(module: types.ModuleType, training_paths: list[Path]):
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if _has_real_training_child(name, training_paths):
            raise AttributeError(name)
        _load_real_training_init(module, training_paths)
        try:
            return module.__dict__[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    return __getattr__


def _has_real_training_child(name: str, training_paths: list[Path]) -> bool:
    for training_path in training_paths:
        package_init = training_path / name / "__init__.py"
        module_file = training_path / f"{name}.py"
        if package_init.exists() or module_file.exists():
            return True
    return False


def _install_lightweight_training_config_package(training_paths: list[Path]) -> None:
    if "megatron.training.config" in sys.modules:
        return

    config_paths = [path / "config" for path in training_paths if (path / "config" / "__init__.py").exists()]
    if not config_paths:
        return

    config_module = types.ModuleType("megatron.training.config")
    config_module.__file__ = str(config_paths[0] / "__init__.py")
    config_module.__package__ = "megatron.training.config"
    config_module.__path__ = [str(path) for path in config_paths]
    config_module.__loader__ = None
    spec = importlib.util.spec_from_loader(
        "megatron.training.config",
        loader=None,
        is_package=True,
    )
    if spec is not None:
        spec.submodule_search_locations = list(config_module.__path__)
        config_module.__spec__ = spec

    sys.modules["megatron.training.config"] = config_module
    sys.modules["megatron.training"].config = config_module  # type: ignore[attr-defined]

    from megatron.training.config.common_config import DistributedInitConfig, ProfilingConfig, RNGConfig
    from megatron.training.config.resilience_config import RerunStateMachineConfig, StragglerDetectionConfig
    from megatron.training.config.training_config import (
        CheckpointConfig,
        LoggerConfig,
        SchedulerConfig,
        TrainingConfig,
        ValidationConfig,
    )

    config_module.RNGConfig = RNGConfig
    config_module.ProfilingConfig = ProfilingConfig
    config_module.DistributedInitConfig = DistributedInitConfig
    config_module.TrainingConfig = TrainingConfig
    config_module.ValidationConfig = ValidationConfig
    config_module.SchedulerConfig = SchedulerConfig
    config_module.LoggerConfig = LoggerConfig
    config_module.CheckpointConfig = CheckpointConfig
    config_module.RerunStateMachineConfig = RerunStateMachineConfig
    config_module.StragglerDetectionConfig = StragglerDetectionConfig
    config_module.__all__ = [
        "RNGConfig",
        "ProfilingConfig",
        "DistributedInitConfig",
        "TrainingConfig",
        "ValidationConfig",
        "SchedulerConfig",
        "LoggerConfig",
        "CheckpointConfig",
        "RerunStateMachineConfig",
        "StragglerDetectionConfig",
    ]
    config_module.__getattr__ = _lazy_real_training_config_attr(config_module, config_paths)  # type: ignore[attr-defined]


def _install_lightweight_training_utils_module() -> None:
    module_name = "megatron.training.utils"
    if module_name in sys.modules:
        return

    utils_module = types.ModuleType(module_name)
    utils_module.__package__ = "megatron.training"

    def has_nvrx_installed() -> bool:
        if "nvidia_resiliency_ext" in sys.modules or "nvidia_resiliency_ext.inprocess" in sys.modules:
            return True
        try:
            return importlib.util.find_spec("nvidia_resiliency_ext") is not None
        except (ImportError, ValueError):
            return False

    def has_nvrx_checkpointing_async_support() -> bool:
        return False

    def print_rank_0(*args, **kwargs) -> None:
        if str(os.environ.get("RANK", "0")).strip() in {"", "0"}:
            print(*args, **kwargs)

    def print_rank_last(*args, **kwargs) -> None:
        print_rank_0(*args, **kwargs)

    def is_last_rank() -> bool:
        return str(os.environ.get("RANK", "0")).strip() in {"", "0"}

    utils_module.has_nvrx_installed = has_nvrx_installed
    utils_module.has_nvrx_checkpointing_async_support = has_nvrx_checkpointing_async_support
    utils_module.print_rank_0 = print_rank_0
    utils_module.print_rank_last = print_rank_last
    utils_module.is_last_rank = is_last_rank
    utils_module.__all__ = [
        "has_nvrx_installed",
        "has_nvrx_checkpointing_async_support",
        "print_rank_0",
        "print_rank_last",
        "is_last_rank",
    ]
    sys.modules[module_name] = utils_module
    sys.modules["megatron.training"].utils = utils_module  # type: ignore[attr-defined]


def _lazy_real_training_config_attr(config_module: types.ModuleType, config_paths: list[Path]):
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        _load_real_training_config_init(config_module, config_paths)
        try:
            return config_module.__dict__[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    return __getattr__


def _load_real_training_init(module: types.ModuleType, training_paths: list[Path]) -> None:
    if bool(module.__dict__.get("_study_real_init_loaded", False)):
        return
    if bool(module.__dict__.get("_study_real_init_loading", False)):
        return

    init_path = training_paths[0] / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "megatron.training",
        init_path,
        submodule_search_locations=[str(path) for path in training_paths],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Megatron training package from {init_path}")

    module.__dict__["_study_real_init_loading"] = True
    try:
        module.__spec__ = spec
        module.__loader__ = spec.loader
        spec.loader.exec_module(module)
        module.__dict__["_study_real_init_loaded"] = True
    finally:
        module.__dict__.pop("_study_real_init_loading", None)


def _load_real_training_config_init(config_module: types.ModuleType, config_paths: list[Path]) -> None:
    if bool(config_module.__dict__.get("_study_real_init_loaded", False)):
        return
    if bool(config_module.__dict__.get("_study_real_init_loading", False)):
        return

    init_path = config_paths[0] / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "megatron.training.config",
        init_path,
        submodule_search_locations=[str(path) for path in config_paths],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Megatron training config package from {init_path}")

    config_module.__dict__["_study_real_init_loading"] = True
    try:
        config_module.__spec__ = spec
        config_module.__loader__ = spec.loader
        spec.loader.exec_module(config_module)
        config_module.__dict__["_study_real_init_loaded"] = True
    finally:
        config_module.__dict__.pop("_study_real_init_loading", None)


def install_bridge_packed_sequence_specs_import_shim() -> None:
    """Expose ``PackedSequenceSpecs`` without importing SFT dataset stacks."""
    if str(os.environ.get("PROOF_DISABLE_BRIDGE_PACKED_SEQUENCE_IMPORT_SHIM", "")).strip() == "1":
        return
    module_name = "megatron.bridge.data.datasets.packed_sequence"
    if module_name in sys.modules:
        return

    import megatron.bridge.data.datasets as datasets_package

    module_path = _find_bridge_packed_sequence_module()
    if module_path is None:
        return

    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = "megatron.bridge.data.datasets"
    module.__loader__ = None
    spec = importlib.util.spec_from_loader(module_name, loader=None, is_package=False)
    if spec is not None:
        module.__spec__ = spec

    module.PackedSequenceSpecs = _packed_sequence_specs_class()
    module.__all__ = ["PackedSequenceSpecs"]
    module.__getattr__ = _lazy_real_packed_sequence_attr(module, module_path)  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    setattr(datasets_package, "packed_sequence", module)


def install_bridge_pretraining_dataset_import_shims() -> None:
    """Avoid unused finetuning/HF dataset imports for GPT pretraining runs."""
    if str(os.environ.get("PROOF_DISABLE_BRIDGE_PRETRAINING_DATASET_IMPORT_SHIMS", "")).strip() == "1":
        return

    import megatron.bridge.data.builders as builders_package
    import megatron.bridge.data.datasets as datasets_package

    _install_placeholder_module(
        "megatron.bridge.data.builders.finetuning_dataset",
        builders_package,
        "finetuning_dataset",
        {
            "FinetuningDatasetBuilder": _unavailable_class(
                "FinetuningDatasetBuilder",
                "Finetuning dataset support is not imported in this GPT-pretraining startup-shimmed run.",
            )
        },
    )
    _install_placeholder_module(
        "megatron.bridge.data.builders.hf_dataset",
        builders_package,
        "hf_dataset",
        {
            "HFDatasetBuilder": _unavailable_class(
                "HFDatasetBuilder",
                "Hugging Face dataset support is not imported in this GPT-pretraining startup-shimmed run.",
            ),
            "HFDatasetConfig": type("HFDatasetConfig", (), {"__module__": "megatron.bridge.data.builders.hf_dataset"}),
        },
    )
    _install_placeholder_module(
        "megatron.bridge.data.datasets.fim_dataset",
        datasets_package,
        "fim_dataset",
        {
            "GPTFIMDataset": _unavailable_class(
                "GPTFIMDataset",
                "FIM dataset support is not imported in this GPT-pretraining startup-shimmed run.",
            )
        },
    )


def install_bridge_mamba_model_import_shim() -> None:
    """Avoid importing optional Mamba/Transformers stacks for non-Mamba runs."""
    if str(os.environ.get("PROOF_DISABLE_BRIDGE_MAMBA_IMPORT_SHIM", "")).strip() == "1":
        return

    package_name = "megatron.bridge.models.mamba"
    builder_name = f"{package_name}.mamba_builder"
    provider_name = f"{package_name}.mamba_provider"
    if builder_name in sys.modules and provider_name in sys.modules:
        return

    models_package = _ensure_bridge_models_package()
    if models_package is None:
        return

    package_path = _find_bridge_mamba_package(models_package)
    if package_path is None:
        return

    package = types.ModuleType(package_name)
    package.__file__ = str(package_path / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(package_path)]
    package.__loader__ = None
    package_spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    if package_spec is not None:
        package_spec.submodule_search_locations = list(package.__path__)
        package.__spec__ = package_spec

    builder = types.ModuleType(builder_name)
    builder.__file__ = str(package_path / "mamba_builder.py")
    builder.__package__ = package_name
    builder.__loader__ = None
    builder.MambaModelConfig = type("MambaModelConfig", (), {"__module__": builder_name})
    builder.__all__ = ["MambaModelConfig"]

    provider = types.ModuleType(provider_name)
    provider.__file__ = str(package_path / "mamba_provider.py")
    provider.__package__ = package_name
    provider.__loader__ = None
    provider.MambaModelProvider = type("MambaModelProvider", (), {"__module__": provider_name})
    provider.__all__ = ["MambaModelProvider"]

    package.MambaModelConfig = builder.MambaModelConfig
    package.MambaModelProvider = provider.MambaModelProvider
    package.__all__ = ["MambaModelConfig", "MambaModelProvider"]

    sys.modules[package_name] = package
    sys.modules[builder_name] = builder
    sys.modules[provider_name] = provider
    setattr(models_package, "mamba", package)


def install_bridge_mimo_model_import_shim() -> None:
    """Avoid importing optional MIMO/distributed stacks for non-MIMO runs."""
    if str(os.environ.get("PROOF_DISABLE_BRIDGE_MIMO_IMPORT_SHIM", "")).strip() == "1":
        return

    package_name = "megatron.bridge.models.megatron_mimo"
    provider_name = f"{package_name}.megatron_mimo_provider"
    if provider_name in sys.modules:
        return

    models_package = _ensure_bridge_models_package()
    if models_package is None:
        return

    package_path = _find_bridge_mimo_package(models_package)
    if package_path is None:
        return

    package = types.ModuleType(package_name)
    package.__file__ = str(package_path / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(package_path)]
    package.__loader__ = None
    package_spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    if package_spec is not None:
        package_spec.submodule_search_locations = list(package.__path__)
        package.__spec__ = package_spec

    provider = types.ModuleType(provider_name)
    provider.__file__ = str(package_path / "megatron_mimo_provider.py")
    provider.__package__ = package_name
    provider.__loader__ = None
    provider.MegatronMIMOProvider = type("MegatronMIMOProvider", (), {"__module__": provider_name})
    provider.__all__ = ["MegatronMIMOProvider"]

    package.MegatronMIMOProvider = provider.MegatronMIMOProvider
    package.__all__ = ["MegatronMIMOProvider"]

    sys.modules[package_name] = package
    sys.modules[provider_name] = provider
    setattr(models_package, "megatron_mimo", package)


def install_mcore_cuda_graphs_import_shim() -> None:
    """Avoid Transformer Engine graph imports when CUDA graphs are disabled."""
    if str(os.environ.get("PROOF_DISABLE_MCORE_CUDA_GRAPHS_IMPORT_SHIM", "")).strip() == "1":
        return

    module_name = "megatron.core.transformer.cuda_graphs"
    if module_name in sys.modules:
        return

    module_path = _find_mcore_cuda_graphs_module()
    if module_path is None:
        return

    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = "megatron.core.transformer"
    module.__loader__ = None
    spec = importlib.util.spec_from_loader(module_name, loader=None, is_package=False)
    if spec is not None:
        module.__spec__ = spec

    class _UnavailableTECudaGraphHelper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Transformer Engine CUDA graphs are not available in this startup-shimmed run.")

    class CudaGraphManager:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def create_cudagraphs(self, *args, **kwargs) -> None:
            return None

        def delete_cuda_graphs(self, *args, **kwargs) -> None:
            return None

        def cuda_graph_set_manual_hooks(self, *args, **kwargs) -> None:
            return None

        def graphs_created(self) -> bool:
            return False

    def is_graph_capturing() -> bool:
        return False

    def is_graph_warmup() -> bool:
        return False

    @contextmanager
    def graph_capture():
        yield

    def create_cudagraphs(*args, **kwargs) -> None:
        return None

    def delete_cuda_graphs(*args, **kwargs) -> None:
        return None

    def set_current_microbatch(*args, **kwargs) -> None:
        return None

    def get_vision_cuda_graph_seq_length(config) -> int | None:
        return getattr(config, "seq_length", None)

    module.CudaGraphManager = CudaGraphManager
    module.TECudaGraphHelper = _UnavailableTECudaGraphHelper
    module.VisionTECudaGraphHelper = _UnavailableTECudaGraphHelper
    module.is_graph_capturing = is_graph_capturing
    module.is_graph_warmup = is_graph_warmup
    module.graph_capture = graph_capture
    module.create_cudagraphs = create_cudagraphs
    module.delete_cuda_graphs = delete_cuda_graphs
    module.set_current_microbatch = set_current_microbatch
    module.get_vision_cuda_graph_seq_length = get_vision_cuda_graph_seq_length
    module.__all__ = [
        "CudaGraphManager",
        "TECudaGraphHelper",
        "VisionTECudaGraphHelper",
        "is_graph_capturing",
        "is_graph_warmup",
        "graph_capture",
        "create_cudagraphs",
        "delete_cuda_graphs",
        "set_current_microbatch",
        "get_vision_cuda_graph_seq_length",
    ]

    sys.modules[module_name] = module


def _install_placeholder_module(
    module_name: str,
    parent_package: types.ModuleType,
    parent_attr: str,
    exports: dict[str, object],
) -> None:
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = None
    spec = importlib.util.spec_from_loader(module_name, loader=None, is_package=False)
    if spec is not None:
        module.__spec__ = spec
    for name, value in exports.items():
        setattr(module, name, value)
    module.__all__ = sorted(exports)
    sys.modules[module_name] = module
    setattr(parent_package, parent_attr, module)


def _unavailable_class(name: str, message: str):
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(message)

    _Unavailable.__name__ = name
    _Unavailable.__qualname__ = name
    return _Unavailable


def _ensure_bridge_models_package() -> types.ModuleType | None:
    module_name = "megatron.bridge.models"
    if module_name in sys.modules:
        return sys.modules[module_name]

    import megatron.bridge as bridge_package

    models_paths: list[Path] = []
    for raw_path in getattr(bridge_package, "__path__", []):
        candidate = Path(raw_path) / "models"
        if (candidate / "__init__.py").exists():
            models_paths.append(candidate)
    if not models_paths:
        return None

    module = types.ModuleType(module_name)
    module.__file__ = str(models_paths[0] / "__init__.py")
    module.__package__ = module_name
    module.__path__ = [str(path) for path in models_paths]
    module.__loader__ = None
    spec = importlib.util.spec_from_loader(module_name, loader=None, is_package=True)
    if spec is not None:
        spec.submodule_search_locations = list(module.__path__)
        module.__spec__ = spec

    module.GPTModelProvider = type("GPTModelProvider", (), {"__module__": module_name})
    module.T5ModelProvider = type("T5ModelProvider", (), {"__module__": module_name})
    module.__all__ = ["GPTModelProvider", "T5ModelProvider"]
    sys.modules[module_name] = module
    setattr(bridge_package, "models", module)
    return module


def _find_bridge_mamba_package(models_package: types.ModuleType) -> Path | None:
    for raw_path in getattr(models_package, "__path__", []):
        candidate = Path(raw_path) / "mamba"
        if (candidate / "__init__.py").exists():
            return candidate
    return None


def _find_bridge_mimo_package(models_package: types.ModuleType) -> Path | None:
    for raw_path in getattr(models_package, "__path__", []):
        candidate = Path(raw_path) / "megatron_mimo"
        if (candidate / "__init__.py").exists():
            return candidate
    return None


def _find_mcore_cuda_graphs_module() -> Path | None:
    import megatron

    for raw_path in getattr(megatron, "__path__", []):
        candidate = Path(raw_path) / "core" / "transformer" / "cuda_graphs.py"
        if candidate.exists():
            return candidate
    return None


def _packed_sequence_specs_class():
    @dataclass
    class PackedSequenceSpecs:
        packed_sequence_size: int = -1
        tokenizer_model_name: str | None = None
        num_tokenizer_workers: int = -1
        packed_train_data_path: str | None = None
        packed_val_data_path: str | None = None
        packed_metadata_path: str | None = None
        pad_cu_seqlens: bool = False
        pad_seq_to_mult: int | None = 1

    PackedSequenceSpecs.__module__ = "megatron.bridge.data.datasets.packed_sequence"
    return PackedSequenceSpecs


def _find_bridge_packed_sequence_module() -> Path | None:
    import megatron.bridge.data.datasets as datasets_package

    for raw_path in getattr(datasets_package, "__path__", []):
        candidate = Path(raw_path) / "packed_sequence.py"
        if candidate.exists():
            return candidate
    return None


def _lazy_real_packed_sequence_attr(module: types.ModuleType, module_path: Path):
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        _load_real_packed_sequence_module(module, module_path)
        try:
            return module.__dict__[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    return __getattr__


def _load_real_packed_sequence_module(module: types.ModuleType, module_path: Path) -> None:
    if bool(module.__dict__.get("_study_real_init_loaded", False)):
        return
    if bool(module.__dict__.get("_study_real_init_loading", False)):
        return

    spec = importlib.util.spec_from_file_location(module.__name__, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Bridge packed sequence module from {module_path}")

    module.__dict__["_study_real_init_loading"] = True
    try:
        module.__spec__ = spec
        module.__loader__ = spec.loader
        spec.loader.exec_module(module)
        module.__dict__["_study_real_init_loaded"] = True
    finally:
        module.__dict__.pop("_study_real_init_loading", None)
