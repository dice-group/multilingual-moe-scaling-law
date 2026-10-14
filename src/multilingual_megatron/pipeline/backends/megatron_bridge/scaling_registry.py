"""Stage 1 dense and Switch-MoE recipe geometry."""
from __future__ import annotations

ARCHITECTURE_FAMILIES = {
    "scaling_dense": {
        "arch_key": "scaling_switch",
        "tokenizer_id": "multilingual_sp_unigram_64k_k200_s03",
        "builder": "scaling_dense"
    },
    "scaling_switch_moe": {
        "arch_key": "scaling_switch",
        "tokenizer_id": "multilingual_sp_unigram_64k_k200_s03",
        "builder": "scaling_switch_moe"
    }
}

SCALE_SPECS: dict[str, dict[str, int]] = {
    "scaling_a": {
        "target_active_params": 92798976,
        "num_layers": 8,
        "hidden_size": 512,
        "ffn_hidden_size": 1536,
        "num_attention_heads": 8,
        "num_query_groups": 8,
        "vocab_size": 64000,
        "seq_length": 2048,
        "scaling_default_experts": 4
    },
    "scaling_b": {
        "target_active_params": 190316544,
        "num_layers": 12,
        "hidden_size": 768,
        "ffn_hidden_size": 2304,
        "num_attention_heads": 12,
        "num_query_groups": 12,
        "vocab_size": 64000,
        "seq_length": 2048,
        "scaling_default_experts": 4
    },
    "scaling_c": {
        "target_active_params": 349175808,
        "num_layers": 16,
        "hidden_size": 1024,
        "ffn_hidden_size": 3072,
        "num_attention_heads": 16,
        "num_query_groups": 16,
        "vocab_size": 64000,
        "seq_length": 2048,
        "scaling_default_experts": 4
    },
    "scaling_d": {
        "target_active_params": 589824000,
        "num_layers": 20,
        "hidden_size": 1280,
        "ffn_hidden_size": 3840,
        "num_attention_heads": 20,
        "num_query_groups": 20,
        "vocab_size": 64000,
        "seq_length": 2048,
        "scaling_default_experts": 4
    },
    "scaling_l26": {
        "target_active_params": 1148878848,
        "num_layers": 26,
        "hidden_size": 1664,
        "ffn_hidden_size": 4992,
        "num_attention_heads": 26,
        "num_query_groups": 26,
        "vocab_size": 64000,
        "seq_length": 2048,
        "scaling_default_experts": 32
    }
}


def scale_spec(scale_label: str, arch_key: str) -> dict[str, int]:
    if arch_key != "scaling_switch":
        raise KeyError(f"Unknown architecture key: {arch_key}")
    return dict(SCALE_SPECS[scale_label])


def recipe_name(family: str, scale_label: str) -> str:
    return f"{family}_scale_{scale_label}_pretrain_config"
