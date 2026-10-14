"""Shared evaluation-only Bridge setup; training setup is restored on exit."""
from typing import Any


def setup_bridge_inference_only(state: Any, dataset_provider: Any) -> Any:
    """Run Bridge setup without training-only DDP buffers or optimizer copies.

    The pinned Bridge setup has no inference-only switch. Temporarily replace
    its training hooks while preserving its checkpoint loader, then restore them
    even on failure. Use only in a dedicated evaluation process, not concurrently
    with training setup in the same process.
    """

    import importlib

    setup_module = importlib.import_module("megatron.bridge.training.setup")
    cfg = state.cfg
    if (not cfg.validation.skip_train or cfg.checkpoint.load_optim or cfg.checkpoint.load_rng
            or cfg.checkpoint.save is not None):
        raise RuntimeError("inference setup requires skip_train, no optimizer/RNG loading and no saving")
    if cfg.peft is not None:
        raise RuntimeError("checkpoint inference setup does not support PEFT")
    if cfg.dist.use_megatron_fsdp or cfg.dist.use_torch_fsdp2:
        raise RuntimeError("checkpoint inference setup does not support FSDP")

    original_build_model = setup_module._build_distributed_model
    original_setup_optimizer = setup_module.setup_optimizer
    original_update_model_config_funcs = setup_module._update_model_config_funcs

    def _build_inference_model(config: Any, pg_collection: Any) -> list[Any]:
        model_config = config.model
        if isinstance(model_config, setup_module.ModelConfig):
            builder = model_config.get_builder_cls()(model_config)
            return builder.build_distributed_models(
                pg_collection=pg_collection,
                ddp_config=None,
                overlap_param_gather_with_optimizer_step=False,
                use_megatron_fsdp=False,
                use_torch_fsdp2=False,
                wrap_with_ddp=False,
                data_parallel_random_init=config.rng.data_parallel_random_init,
            )
        return model_config.provide_distributed_model(
            ddp_config=None,
            overlap_param_gather_with_optimizer_step=False,
            use_megatron_fsdp=False,
            use_torch_fsdp2=False,
            wrap_with_ddp=False,
            data_parallel_random_init=config.rng.data_parallel_random_init,
            pg_collection=pg_collection,
        )

    def _skip_optimizer(**_kwargs: Any) -> tuple[None, None]:
        return None, None

    def _skip_training_model_config_hooks(*_args: Any, **_kwargs: Any) -> None:
        return None

    setup_module._build_distributed_model = _build_inference_model
    setup_module.setup_optimizer = _skip_optimizer
    setup_module._update_model_config_funcs = _skip_training_model_config_hooks
    try:
        setup_output = setup_module.setup(state, dataset_provider)
    finally:
        setup_module._build_distributed_model = original_build_model
        setup_module.setup_optimizer = original_setup_optimizer
        setup_module._update_model_config_funcs = original_update_model_config_funcs
    if setup_output.optimizer is not None or setup_output.scheduler is not None:
        raise RuntimeError("inference-only Bridge setup unexpectedly created optimizer state")
    return setup_output
