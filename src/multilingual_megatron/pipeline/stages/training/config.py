from __future__ import annotations

from typing import Any


BRIDGE_BACKENDS = {"megatron_bridge", "bridge"}


def as_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def is_bridge_backend(common: dict[str, Any], run_cfg: dict[str, Any]) -> bool:
    backend = str(run_cfg.get("backend", common.get("backend", "megatron_bridge"))).strip().lower()
    return backend in BRIDGE_BACKENDS


def resolve_train_iters(
    *,
    train: dict[str, Any],
    common: dict[str, Any],
    global_batch_size: int,
    seq_length: int,
) -> int:
    token_budget = as_positive_int(
        train.get(
            "token_budget_tokens",
            train.get("token_budget", common.get("token_budget_tokens", common.get("token_budget"))),
        )
    )
    if token_budget is not None:
        tokens_per_iter = max(1, int(global_batch_size) * int(seq_length))
        return max(1, (token_budget + tokens_per_iter - 1) // tokens_per_iter)
    return int(train.get("train_iters", train.get("max_steps", common.get("train_iters", 10))) or 10)


def validate_training_runs(common: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, run in enumerate(runs):
        run_id = str(run.get("id", "")).strip()
        if not run_id:
            raise SystemExit(f"training.runs[{index}].id must be set")
        if run_id in seen:
            raise SystemExit(f"duplicate training run id: {run_id}")
        seen.add(run_id)
        if not is_bridge_backend(common, run):
            kind = str(run.get("kind", ""))
            raise SystemExit(f"Unsupported training run for Megatron study: {kind!r}")


def resolve_training_stage_config(ctx: Any) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    train_cfg_for_stage = dict(ctx.train_cfg)
    runs = [dict(run) for run in train_cfg_for_stage.get("runs", [])]
    if not runs and bool(train_cfg_for_stage.get("enabled", True)):
        raise SystemExit("training.enabled=true requires at least one entry in training.runs")
    validate_training_runs(train_cfg_for_stage, runs)
    return train_cfg_for_stage, str(train_cfg_for_stage.get("recipe", "")), runs
