#!/usr/bin/env python3
"""Shared definitions for the multilingual Switch-MoE scaling-law study."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multilingual_megatron.contracts.language_sets import (
    K1_CODES,
    K4_CODES,
    K12_CODES,
    K16_CODES,
    K50_CODES,
    SHORT_CODE_TO_FINEWEB2_SUBSET,
)
from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__))
DEFAULT_STUDY_DIR = REPO_ROOT / "configs/scaling_law/multilingual_moe_v1"
DEFAULT_FINEWEB2_CSV = REPO_ROOT / "configs/data_sources/fineweb2-language-distribution.csv"
LANGUAGE_GROUPING_DIR = REPO_ROOT / "configs/language_groupings"

STUDY_ID = "multilingual_moe_v1"
STATUS_DATE = "2026-06-02"
VOCAB_SIZE = 64_000
SEQ_LENGTH = 2_048
E_CORE = (1, 4, 16, 32)
K_CORE = (1, 16, 50, 200)
MOE_MULTIPLIERS = {1: 1.00, 4: 1.10, 16: 1.25, 32: 1.35}
# H100 SXM BF16 Tensor Core peak without sparsity. Keep historical 756 TFLOP/s
# figures in old run notes only; current H100 planning uses the SXM denominator.
H100_PEAK_TFLOPS_PER_GPU = 989.5
H100_ASSUMED_MFU = 0.35
H100_EFFECTIVE_FLOPS_PER_GPU_HOUR = H100_PEAK_TFLOPS_PER_GPU * 1e12 * H100_ASSUMED_MFU * 3600
ROUTING_GATE = {
    "max_dropped_assignment_rate": 0.05,
    "require_zero_kept_tokens": True,
    "min_capacity_factor": 1.0,
}


@dataclass(frozen=True)
class Ladder:
    label: str
    num_attention_heads: int
    num_layers: int
    hidden_size: int
    token_checkpoints: tuple[int, ...]

    @property
    def ffn_hidden_size(self) -> int:
        return 3 * self.hidden_size

    @property
    def n_act(self) -> int:
        return 2 * self.hidden_size * VOCAB_SIZE + 13 * self.num_layers * self.hidden_size**2

    @property
    def n_act_nonembed(self) -> int:
        return 13 * self.num_layers * self.hidden_size**2

    def n_total(self, experts: int) -> int:
        return 2 * self.hidden_size * VOCAB_SIZE + (4 + 9 * experts) * self.num_layers * self.hidden_size**2


LADDERS = (
    Ladder("A", 8, 8, 512, (500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000)),
    Ladder(
        "B",
        12,
        12,
        768,
        (500_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000, 16_000_000_000),
    ),
    Ladder(
        "C",
        16,
        16,
        1024,
        (
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
            8_000_000_000,
            16_000_000_000,
            32_000_000_000,
        ),
    ),
    Ladder(
        "D",
        20,
        20,
        1280,
        (
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
            8_000_000_000,
            16_000_000_000,
            32_000_000_000,
        ),
    ),
)


def read_fineweb2_inventory(path: Path = DEFAULT_FINEWEB2_CSV) -> dict[str, dict[str, Any]]:
    def parse_int(value: Any) -> int:
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subset = str(row.get("subset", "")).strip()
            if not subset or subset.endswith("_removed") or row.get("split") != "train":
                continue
            row = dict(row)
            row["documents"] = parse_int(row.get("documents"))
            row["words"] = parse_int(row.get("words"))
            rows[subset] = row
    return rows


def _language_entry(short_code: str, inventory: dict[str, dict[str, Any]], *, source: str, rank: int | None = None) -> dict[str, Any]:
    subset = SHORT_CODE_TO_FINEWEB2_SUBSET.get(short_code)
    row = inventory.get(subset or "", {})
    return {
        "id": short_code,
        "short_code": short_code,
        "fineweb2_subset": subset,
        "source": source,
        "k100_added_rank": rank,
        "name": row.get("name", ""),
        "family": row.get("family", ""),
        "script": row.get("script", ""),
        "fineweb2_train_documents": int(row.get("documents") or 0),
        "fineweb2_train_words": int(row.get("words") or 0),
        "resource_availability": row.get("resource_availability", ""),
        "fallback_sources": [] if subset else ["wikimedia/wikipedia", "native_source_or_manual_inventory"],
    }


def _load_language_grouping(k: int) -> dict[str, Any]:
    path = LANGUAGE_GROUPING_DIR / f"scaling_fw2_k{k:03d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    group = payload.get("0")
    if not isinstance(group, dict):
        raise ValueError(f"{path} does not contain grouping key '0'")
    languages = group.get("languages")
    if not isinstance(languages, list):
        raise ValueError(f"{path} grouping key '0' has no language list")
    if len(languages) != k:
        raise ValueError(f"{path} has {len(languages)} languages, expected K={k}")
    return group


def _language_entry_from_subset(
    subset: str,
    inventory: dict[str, dict[str, Any]],
    *,
    metadata: dict[str, Any],
    source: str,
    rank: int,
) -> dict[str, Any]:
    row = inventory.get(subset, {})
    code = str(row.get("code") or subset.split("_", 1)[0])
    script = row.get("script") or metadata.get("script", "")
    if not script and "_" in subset:
        script = subset.split("_", 1)[1]
    return {
        "id": subset,
        "short_code": code,
        "fineweb2_subset": subset,
        "source": source,
        "k200_rank": rank,
        "name": row.get("name") or metadata.get("name", ""),
        "family": row.get("family") or metadata.get("family", ""),
        "script": script,
        "fineweb2_train_documents": int(row.get("documents") or 0),
        "fineweb2_train_words": int(row.get("words") or 0),
        "resource_availability": row.get("resource_availability") or metadata.get("resource_availability", ""),
        "fallback_sources": [] if subset in inventory else ["wikimedia/wikipedia", "native_source_or_manual_inventory"],
    }


def build_language_sets(inventory: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    inventory = inventory or read_fineweb2_inventory()
    by_k: dict[int, list[dict[str, Any]]] = {}
    grouping_sources: dict[int, str] = {}
    for k in K_CORE:
        group = _load_language_grouping(k)
        languages = [str(item) for item in group["languages"]]
        metadata = group.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_map = group.get("selection_policy", {}).get("source_map", "") if isinstance(group.get("selection_policy"), dict) else ""
        grouping_sources[k] = str(source_map)
        by_k[k] = [
            _language_entry_from_subset(
                subset,
                inventory,
                metadata=metadata.get(subset, {}) if isinstance(metadata.get(subset, {}), dict) else {},
                source=f"scaling_fw2_k{k:03d}",
                rank=rank,
            )
            for rank, subset in enumerate(languages, start=1)
        ]
    for k, entries in by_k.items():
        if len(entries) != k:
            raise ValueError(f"Language set K={k} has {len(entries)} entries")
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status_date": STATUS_DATE,
        "source_inventory": str(DEFAULT_FINEWEB2_CSV.relative_to(REPO_ROOT)),
        "language_groupings": {
            f"K{k}": str((LANGUAGE_GROUPING_DIR / f"scaling_fw2_k{k:03d}.json").relative_to(REPO_ROOT))
            for k in K_CORE
        },
        "grouping_source_maps": {f"K{k}": source for k, source in grouping_sources.items()},
        "notes": [
            "K sets are frozen from configs/language_groupings/scaling_fw2_k*.json.",
            "K=200 is the largest real-study language set; smaller K sets are fixed subsets for controlled scaling comparisons.",
            "FineWeb2 subsets missing from the local inventory are retained and require Wikipedia or native-source fallback data before real training.",
        ],
        "sets": {f"K{k}": {"language_set_id": f"{STUDY_ID}_K{k}", "K": k, "languages": entries} for k, entries in by_k.items()},
    }


def learning_rate(n_act_nonembed: int, experts: int) -> float:
    return math.exp(8.39 - 0.81 * math.log(float(n_act_nonembed)) - 0.25 * math.log(float(experts)))


def train_flops(n_act: int, tokens: int) -> int:
    return int(6 * int(n_act) * int(tokens))


def batch_schedule() -> list[dict[str, int]]:
    return [
        {"start_tokens": 0, "end_tokens": 500_000_000, "global_batch_tokens": 65_536, "global_batch_sequences": 32},
        {
            "start_tokens": 500_000_000,
            "end_tokens": 1_000_000_000,
            "global_batch_tokens": 131_072,
            "global_batch_sequences": 64,
        },
        {"start_tokens": 1_000_000_000, "end_tokens": 0, "global_batch_tokens": 262_144, "global_batch_sequences": 128},
    ]


def h100_plan_for_run(ladder: Ladder, experts: int) -> dict[str, Any]:
    if experts == 1:
        return {"stage": "h100_shape_smoke", "gpus": 1, "tensor_parallel": 1, "expert_parallel": 1, "notes": "dense baseline/control"}
    if experts == 4:
        return {
            "stage": "h100_shape_smoke",
            "gpus": 1,
            "tensor_parallel": 1,
            "expert_parallel": 1,
            "notes": "single-H100 Switch-MoE baseline before EP tuning",
        }
    if experts == 16:
        if ladder.label in {"A", "B"}:
            return {
                "stage": "h100_shape_smoke",
                "gpus": 1,
                "tensor_parallel": 1,
                "expert_parallel": 1,
                "notes": "try one H100 first; promote to EP2 if memory or MFU gate fails",
            }
        return {
            "stage": "h100_shape_smoke",
            "gpus": 2,
            "tensor_parallel": 1,
            "expert_parallel": 2,
            "notes": "two-H100 expert parallel candidate for larger active scales",
        }
    if ladder.label == "A":
        return {
            "stage": "h100_shape_smoke",
            "gpus": 1,
            "tensor_parallel": 1,
            "expert_parallel": 1,
            "notes": "try one H100 first; route diagnostics decide whether EP2 is needed",
        }
    if ladder.label in {"B", "C"}:
        return {
            "stage": "h100_shape_smoke",
            "gpus": 2,
            "tensor_parallel": 1,
            "expert_parallel": 2,
            "notes": "two-H100 EP2 candidate for E32",
        }
    return {
        "stage": "h100_shape_smoke",
        "gpus": 4,
        "tensor_parallel": 1,
        "expert_parallel": 4,
        "notes": "four-H100 EP4 candidate for largest E32 shape",
    }


def build_run_configs(language_sets: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    language_sets = language_sets or build_language_sets()
    runs: list[dict[str, Any]] = []
    for ladder in LADDERS:
        for experts in E_CORE:
            for k in K_CORE:
                checkpoints = [
                    {
                        "D_checkpoint": tokens,
                        "F_checkpoint": train_flops(ladder.n_act, tokens),
                    }
                    for tokens in ladder.token_checkpoints
                ]
                run_id = f"{STUDY_ID}_L{ladder.label}_E{experts}_K{k}"
                runs.append(
                    {
                        "schema_version": 1,
                        "study_id": STUDY_ID,
                        "run_id": run_id,
                        "ladder": ladder.label,
                        "E": experts,
                        "K": k,
                        "language_set_id": language_sets["sets"][f"K{k}"]["language_set_id"],
                        "corpus_manifest_id": f"{STUDY_ID}_hq_first_v0",
                        "architecture": {
                            "family": "decoder_only_transformer",
                            "moe_type": "switch_moe_ffn",
                            "routing": "top1",
                            "active_experts_per_token": 1,
                            "replace_every_ffn": True,
                            "expert_granularity": 1,
                            "num_attention_heads": ladder.num_attention_heads,
                            "num_layers": ladder.num_layers,
                            "hidden_size": ladder.hidden_size,
                            "ffn_hidden_size": ladder.ffn_hidden_size,
                            "vocab_size": VOCAB_SIZE,
                            "seq_length": SEQ_LENGTH,
                            "num_moe_experts": experts,
                            "expert_hidden_size": ladder.ffn_hidden_size,
                            "activation": "swiglu",
                            "router_z_loss": 0.001,
                            "load_balancing_loss": 0.01,
                            "init_method": "truncated_normal_reduced_scale_0.1",
                            "precision": "bf16_mixed",
                            "high_precision_modules": ["attention", "rope", "router"],
                        },
                        "counts": {
                            "N_total": ladder.n_total(experts),
                            "N_act": ladder.n_act,
                            "N_act_nonembed": ladder.n_act_nonembed,
                        },
                        "training": {
                            "tokenizer": "64k_multilingual_sentencepiece",
                            "data_policy": "fixed_hq_first_corpus",
                            "language_sampling": "uniform",
                            "per_language_tokens": "D_tot / K",
                            "peak_lr": learning_rate(ladder.n_act_nonembed, experts),
                            "lr_rule": "exp(8.39 - 0.81 ln N_act_nonembed - 0.25 ln E)",
                            "warmup_tokens": 130_000_000,
                            "decay_fraction": 0.20,
                            "batch_schedule": batch_schedule(),
                            "D_max": ladder.token_checkpoints[-1],
                            "token_checkpoints": list(ladder.token_checkpoints),
                        },
                        "checkpoints": checkpoints,
                        "routing_gate": dict(ROUTING_GATE),
                        "h100_rollout": h100_plan_for_run(ladder, experts),
                    }
                )
    return runs


def build_corpus_manifest(language_sets: dict[str, Any]) -> dict[str, Any]:
    languages = language_sets["sets"]["K200"]["languages"]
    return {
        "schema_version": 1,
        "corpus_manifest_id": f"{STUDY_ID}_hq_first_v0",
        "study_id": STUDY_ID,
        "status": "frozen_for_config_generation",
        "unit": "tokens",
        "target_quota_policy": "uniform_per_active_language_at_runtime",
        "target_quota_expression": "D_t = D_tot / K",
        "sources_priority": [
            {"source_id": "native_high_quality", "priority": 1, "role": "native_hq_if_available"},
            {"source_id": "epfml/FineWeb2-HQ", "priority": 2, "role": "hq_web_for_supported_languages"},
            {"source_id": "wikimedia/wikipedia", "priority": 3, "role": "curated_native_floor"},
            {"source_id": "HuggingFaceFW/fineweb-2", "priority": 4, "role": "coverage_backbone"},
            {"source_id": "capability_anchors", "priority": 5, "role": "english_zh_ja_ar_capability_mix"},
        ],
        "fixed_policy": {
            "data_is_not_an_experimental_axis": True,
            "language_sampling": "uniform over active K",
            "tokenizer": "64k multilingual SentencePiece",
            "dedup_filters": "freeze before training; report manifest hash",
            "quality_filters": "freeze before training; do not vary in phase 1",
        },
        "languages": [
            {
                "language_id": row["id"],
                "short_code": row["short_code"],
                "fineweb2_subset": row["fineweb2_subset"],
                "source_selection": "priority_order",
                "split": {"train": "source_default_train", "validation": "heldout_per_language"},
                "filters": ["fixed_quality_filter_v0", "fixed_dedup_filter_v0"],
                "target_quota": "runtime_uniform_D_tot_over_K",
                "fallback_sources": row["fallback_sources"],
            }
            for row in languages
        ],
    }


def build_study_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status_date": STATUS_DATE,
        "fixed_variables": {
            "architecture_family": "decoder-only Transformer",
            "moe_type": "Switch MoE feed-forward layers",
            "routing": "top-1",
            "active_experts": "one expert active per token",
            "moe_placement": "replace every FFN with MoE FFN",
            "expert_granularity": 1,
            "expert_hidden_size": "3 * d_model, SwiGLU",
            "router_z_loss": 0.001,
            "load_balancing_loss": 0.01,
            "precision": "BF16 mixed precision",
            "context_length": SEQ_LENGTH,
            "tokenizer": "64k multilingual SentencePiece",
            "language_sampling": "uniform over active K",
            "language_counts": list(K_CORE),
            "target_hardware": "H100",
        },
        "ladders": [
            {
                "ladder": ladder.label,
                "num_attention_heads": ladder.num_attention_heads,
                "num_layers": ladder.num_layers,
                "hidden_size": ladder.hidden_size,
                "N_act": ladder.n_act,
                "N_act_nonembed": ladder.n_act_nonembed,
                "token_checkpoints": list(ladder.token_checkpoints),
            }
            for ladder in LADDERS
        ],
        "experts": list(E_CORE),
        "language_counts": list(K_CORE),
        "routing_gate": dict(ROUTING_GATE),
        "parameter_counting": {
            "N_total": "2*d_model*d_vocab + (4+9E)*N_blocks*d_model^2",
            "N_act": "2*d_model*d_vocab + 13*N_blocks*d_model^2",
            "N_act_nonembed": "13*N_blocks*d_model^2",
            "F_train": "6*N_act*D",
            "d_vocab": VOCAB_SIZE,
        },
        "h100_optimization_policy": {
            "strategy": "shape-first local H100 optimization",
            "target_hardware": "H100",
            "gpus_per_run": "1-4",
            "mfu_gate_pct": 35.0,
            "default": "optimize one systems config per (ladder,E); K variants reuse the locked shape unless corpus throughput changes materially",
            "routing_gate": dict(ROUTING_GATE),
        },
    }


def run_matrix_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        counts = run["counts"]
        training = run["training"]
        rows.append(
            {
                "run_id": run["run_id"],
                "ladder": run["ladder"],
                "E": run["E"],
                "K": run["K"],
                "language_set_id": run["language_set_id"],
                "corpus_manifest_id": run["corpus_manifest_id"],
                "N_act": counts["N_act"],
                "N_total": counts["N_total"],
                "N_act_nonembed": counts["N_act_nonembed"],
                "peak_lr": f"{training['peak_lr']:.10g}",
                "D_max": training["D_max"],
                "num_checkpoints": len(run["checkpoints"]),
                "h100_gpus": run["h100_rollout"]["gpus"],
                "h100_tensor_parallel": run["h100_rollout"]["tensor_parallel"],
                "h100_expert_parallel": run["h100_rollout"]["expert_parallel"],
                "max_dropped_assignment_rate": run["routing_gate"]["max_dropped_assignment_rate"],
                "require_zero_kept_tokens": run["routing_gate"]["require_zero_kept_tokens"],
                "min_capacity_factor": run["routing_gate"]["min_capacity_factor"],
            }
        )
    return rows


def checkpoint_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for ckpt in run["checkpoints"]:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "ladder": run["ladder"],
                    "E": run["E"],
                    "K": run["K"],
                    "N_act": run["counts"]["N_act"],
                    "D_checkpoint": ckpt["D_checkpoint"],
                    "F_checkpoint": ckpt["F_checkpoint"],
                    "language_set_id": run["language_set_id"],
                    "corpus_manifest_id": run["corpus_manifest_id"],
                }
            )
    return rows


def estimate_gpu_hours(*, n_act: int, tokens: int, experts: int, effective_flops_per_gpu_hour: float) -> float:
    return train_flops(n_act, tokens) * MOE_MULTIPLIERS[int(experts)] / float(effective_flops_per_gpu_hour)


def validate_artifacts(language_sets: dict[str, Any], runs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_checkpoints = sum(len(ladder.token_checkpoints) for ladder in LADDERS) * len(E_CORE) * len(K_CORE)
    if len(runs) != len(LADDERS) * len(E_CORE) * len(K_CORE):
        errors.append(f"expected {len(LADDERS) * len(E_CORE) * len(K_CORE)} runs, found {len(runs)}")
    if sum(len(run["checkpoints"]) for run in runs) != expected_checkpoints:
        errors.append(f"expected {expected_checkpoints} checkpoint observations")
    for k in K_CORE:
        found = len(language_sets["sets"][f"K{k}"]["languages"])
        if found != k:
            errors.append(f"K{k} has {found} languages")
    ladder_by_label = {ladder.label: ladder for ladder in LADDERS}
    for run in runs:
        ladder = ladder_by_label[str(run["ladder"])]
        experts = int(run["E"])
        if run["counts"]["N_act"] != ladder.n_act:
            errors.append(f"{run['run_id']}: bad N_act")
        if run["counts"]["N_total"] != ladder.n_total(experts):
            errors.append(f"{run['run_id']}: bad N_total")
        expected_lr = learning_rate(ladder.n_act_nonembed, experts)
        if not math.isclose(float(run["training"]["peak_lr"]), expected_lr, rel_tol=1e-12):
            errors.append(f"{run['run_id']}: bad peak_lr")
        for ckpt in run["checkpoints"]:
            if ckpt["F_checkpoint"] != train_flops(ladder.n_act, ckpt["D_checkpoint"]):
                errors.append(f"{run['run_id']}: bad F_checkpoint at {ckpt['D_checkpoint']}")
    return errors


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
