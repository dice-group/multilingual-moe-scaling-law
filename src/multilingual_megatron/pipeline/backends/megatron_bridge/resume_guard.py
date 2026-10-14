"""Verify restored training state before a continuation's first update."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def verify_restored_resume(
    context: Any, *, expected_step: int, expected_samples: int, receipt_path: Path
) -> None:
    """Called on every rank by Bridge's on_train_start event, before updates."""
    if expected_step <= 0 or expected_samples <= 0:
        raise ValueError("full-state resume requires positive step and sample expectations")
    cfg = context.state.cfg.checkpoint
    observed = {
        "step": int(context.state.train_state.step),
        "consumed_train_samples": int(context.state.train_state.consumed_train_samples),
        "scheduler_num_steps": getattr(context.scheduler, "num_steps", None),
    }
    expected = {
        "step": expected_step,
        "consumed_train_samples": expected_samples,
        "scheduler_num_steps": expected_samples,
    }
    if observed != expected:
        raise RuntimeError(f"RESUME_STATE_MISMATCH: expected={expected}, observed={observed}")
    load_path = Path(str(cfg.load))
    if not load_path.is_absolute() or not load_path.is_dir():
        raise RuntimeError(f"resume requires an existing absolute checkpoint path: {load_path}")
    if not cfg.load_optim or not cfg.load_rng or cfg.finetune or context.optimizer is None:
        raise RuntimeError("resume requires restored optimizer/RNG and finetune=false")
    rank = int(os.environ.get("RANK", "0"))
    payload = {
        "schema": "restored_resume_state/v1",
        "status": "verified_before_first_update",
        "rank": rank,
        "checkpoint_path": str(load_path),
        "expected": expected,
        "observed": observed,
    }
    # Keep rank-specific evidence; one successful rank is not an all-rank proof.
    rank_receipt = receipt_path.with_name(f"{receipt_path.stem}.rank{rank}.json")
    rank_receipt.parent.mkdir(parents=True, exist_ok=True)
    rank_receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[RESUME_STATE_VERIFIED] rank={rank} step={expected_step} samples={expected_samples}", flush=True)
