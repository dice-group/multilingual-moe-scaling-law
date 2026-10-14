#!/usr/bin/env python3
"""Context objects for proof pipeline stage execution."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .executor import Executor


@dataclass(slots=True)
class PipelineContext:
    args: argparse.Namespace
    repo_root: Path
    profile: dict[str, Any]
    force_stages: set[str]
    active_stages: list[str]

    output_root: Path
    state_path: Path

    language_map_path: Path
    sample_cfg: dict[str, Any]
    tokenizer_cfg: dict[str, Any]
    tokenize_cfg: dict[str, Any]
    audit_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    summary_cfg: dict[str, Any]
    benchmark_cfg: dict[str, Any]
    baseline_eval_cfg: dict[str, Any]
    wandb_cfg: dict[str, Any]

    env: dict[str, str]
    logger: logging.Logger
    executor: Executor

    sampled_dir: Path
    sampled_manifest: Path
    shard_subdir: str
    sample_shard_dir: Path
    tokenizers_dir: Path
    tokenized_dir: Path
    train_root: Path
    summary_path: Path
    gate_path: Path
    benchmark_root: Path
    baseline_cache_root: Path
    lineage_manifest_path: Path
    logs_root: Path

    run_naming_cfg: dict[str, Any]
    run_timestamp: str
    naming_languages: list[str]
    lang_count: int
    model_slug: str
    exp_tag: str
    base_run_label: str
