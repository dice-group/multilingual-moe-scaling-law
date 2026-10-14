# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Stage 1 dense and Switch-MoE recipe builders."""

from __future__ import annotations

import inspect

import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.config import ConfigContainer

from .common import _kv_channels


def _pilot_local_layer_spec(config, vp_stage=None):
    if str(getattr(config, "transformer_impl", "")).strip() == "transformer_engine":
        signature = inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters
        kwargs = {}
        if "use_te_op_fuser" in signature:
            kwargs["use_te_op_fuser"] = getattr(config, "use_transformer_engine_op_fuser", False)
        if "use_te_activation_func" in signature:
            kwargs["use_te_activation_func"] = getattr(config, "use_te_activation_func", False)
        if "dense_grouped_gemm" in signature:
            kwargs["dense_grouped_gemm"] = getattr(config, "dense_grouped_gemm", False)
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=getattr(config, "num_moe_experts", None),
            moe_grouped_gemm=getattr(config, "moe_grouped_gemm", False),
            qk_layernorm=getattr(config, "qk_layernorm", False),
            **kwargs,
        )
    return get_gpt_layer_local_spec(
        num_experts=getattr(config, "num_moe_experts", None),
        moe_grouped_gemm=getattr(config, "moe_grouped_gemm", False),
        qk_layernorm=getattr(config, "qk_layernorm", False),
        normalization="RMSNorm",
    )


def _finish_scaling_switch_config(cfg: ConfigContainer) -> ConfigContainer:
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.mixed_precision = "bf16_mixed"
    cfg.model.bf16 = True
    cfg.optimizer.bf16 = True
    cfg.optimizer.use_distributed_optimizer = False
    # The optimizer builder defaults to AdamW weight_decay=0.1 but its
    # scheduler defaults to a constant 0.033 and overwrites every param-group
    # value on the first scheduler step.  Freeze both sides of the A/B-v2
    # contract explicitly so the effective runtime value is 0.1.
    cfg.optimizer.optimizer = "adam"
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.decoupled_weight_decay = True
    cfg.optimizer.clip_grad = 1.0
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    return cfg


def _scaling_switch_base_config(spec: dict[str, int]) -> ConfigContainer:
    cfg = vanilla_gpt_pretrain_config()
    cfg.model = GPTModelProvider(
        num_layers=spec["num_layers"],
        hidden_size=spec["hidden_size"],
        ffn_hidden_size=spec["ffn_hidden_size"],
        num_attention_heads=spec["num_attention_heads"],
        num_query_groups=spec.get("num_query_groups", spec["num_attention_heads"]),
        kv_channels=_kv_channels(spec),
        attention_backend="auto",
        vocab_size=spec.get("vocab_size", 64_000),
        seq_length=spec.get("seq_length", 2048),
        position_embedding_type="rope",
        rotary_base=10000,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        share_embeddings_and_output_weights=False,
        should_pad_vocab=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        init_method_std=0.02,
        transformer_impl="transformer_engine",
        cuda_graph_impl="none",
        cross_entropy_loss_fusion=True,
        transformer_layer_spec=_pilot_local_layer_spec,
    )
    return _finish_scaling_switch_config(cfg)


def _scaling_dense_config(spec: dict[str, int]) -> ConfigContainer:
    return _scaling_switch_base_config(spec)


def _scaling_switch_moe_config(spec: dict[str, int]) -> ConfigContainer:
    cfg = _scaling_switch_base_config(spec)
    cfg.model.num_moe_experts = spec.get("scaling_num_experts", spec.get("scaling_default_experts", 4))
    cfg.model.moe_ffn_hidden_size = spec["ffn_hidden_size"]
    cfg.model.moe_layer_freq = 1
    cfg.model.moe_router_topk = 1
    cfg.model.moe_router_pre_softmax = True
    cfg.model.moe_router_load_balancing_type = "aux_loss"
    cfg.model.moe_aux_loss_coeff = 0.01
    cfg.model.moe_z_loss_coeff = 0.001
    cfg.model.moe_token_dispatcher_type = "allgather"
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_force_load_balancing = False
    return cfg
