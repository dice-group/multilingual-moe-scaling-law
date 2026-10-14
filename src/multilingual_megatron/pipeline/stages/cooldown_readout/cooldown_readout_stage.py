from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...context import PipelineContext
from ...runtime import _abs, _deep_merge, _dump_json, _save_stage_state, _should_skip_stage, _stage_signature
from ...types import ArtifactSpec, StagePlan, StageResult
from ..training.config import resolve_training_stage_config
from ..training.planner import (
    TRAINING_RUN_MANIFEST_NAME,
    build_training_run_plan,
    core_run_artifacts_exist,
    write_training_command_plan,
    write_training_run_manifest,
)
from ..training.batch_policy import (
    cumulative_step_for_tokens,
    segment_for_token_position,
    tokens_for_cumulative_step,
)


COOLDOWN_PLAN = "checkpoint_readout_plan.json"
COOLDOWN_PLAN_CSV = "checkpoint_readout_plan.csv"
COOLDOWN_MANIFEST = "cooldown_readout_manifest.json"
COOLDOWN_MANIFEST_CSV = "cooldown_readout_manifest.csv"
FIT_OBSERVATIONS_CSV = "checkpoint_observations_for_fit.csv"
_READOUT_EVAL_OVERRIDE_KEYS = {"validation.eval_interval", "validation.eval_iters"}
_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_ITERATION_RE = re.compile(r"iteration\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)", re.IGNORECASE)
_LM_LOSS_RE = re.compile(r"\blm loss:\s*(?P<value>" + _FLOAT_RE + r")", re.IGNORECASE)
_SKIPPED_RE = re.compile(r"number of skipped iterations:\s*(?P<value>\d+)", re.IGNORECASE)
_NAN_RE = re.compile(r"number of nan iterations:\s*(?P<value>\d+)", re.IGNORECASE)
_CONSUMED_RE = re.compile(r"consumed samples:\s*(?P<value>\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CooldownReadoutConfig:
    enabled: bool
    output_dir: Path
    horizons_tokens_by_scale: dict[str, list[int]]
    cooldown_fraction: float
    cooldown_max_tokens: int
    eval_iters: int
    require_training_complete: bool
    require_checkpoint: bool
    run_forks: bool
    strict: bool


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int_list(value: Any) -> list[int]:
    out: list[int] = []
    if value is None:
        return out
    raw_items = value if isinstance(value, list) else [value]
    for item in raw_items:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            out.append(parsed)
    return out


def _steps_for_tokens(tokens: int, *, global_batch_size: int, seq_length: int) -> int:
    tokens_per_step = max(1, int(global_batch_size) * int(seq_length))
    return max(1, (int(tokens) + tokens_per_step - 1) // tokens_per_step)


def _token_label(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    return str(tokens)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def resolve_cooldown_readout_config(ctx: PipelineContext) -> CooldownReadoutConfig:
    raw_cfg = getattr(ctx, "profile", {}).get("cooldown_readout", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    repo_root = Path(getattr(ctx, "repo_root", Path.cwd()))
    output_root = Path(getattr(ctx, "output_root", repo_root / "outputs" / "pipeline_runs" / "default"))
    output_dir_raw = str(raw_cfg.get("output_dir", output_root / "cooldown_readouts")).strip()
    output_dir = _abs(repo_root, output_dir_raw)
    raw_horizons = raw_cfg.get("horizons_tokens_by_scale", {})
    horizons: dict[str, list[int]] = {}
    if isinstance(raw_horizons, dict):
        for key, value in raw_horizons.items():
            values = sorted(set(_as_int_list(value)))
            if values:
                horizons[str(key)] = values
    return CooldownReadoutConfig(
        enabled=_as_bool(raw_cfg.get("enabled", False)),
        output_dir=output_dir,
        horizons_tokens_by_scale=horizons,
        cooldown_fraction=float(raw_cfg.get("cooldown_fraction", 0.05) or 0.05),
        cooldown_max_tokens=int(raw_cfg.get("cooldown_max_tokens", 256_000_000) or 256_000_000),
        eval_iters=int(raw_cfg.get("eval_iters", 10) or 10),
        require_training_complete=_as_bool(raw_cfg.get("require_training_complete", True), default=True),
        require_checkpoint=_as_bool(raw_cfg.get("require_checkpoint", True), default=True),
        run_forks=_as_bool(raw_cfg.get("run_forks", True), default=True),
        strict=_as_bool(raw_cfg.get("strict", True), default=True),
    )


def _selected_runs(ctx: PipelineContext, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_ids = {str(item) for item in getattr(ctx.args, "run_id", []) if str(item).strip()}
    if not selected_ids:
        return [dict(run) for run in runs]
    return [dict(run) for run in runs if str(run.get("id", "")) in selected_ids]


def _json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scale_metadata(run_cfg: dict[str, Any], train_dir: Path) -> dict[str, Any]:
    scaling = run_cfg.get("scaling", {})
    if isinstance(scaling, dict) and scaling.get("scale_label"):
        return dict(scaling)
    manifest = _json_if_exists(train_dir / TRAINING_RUN_MANIFEST_NAME)
    run_manifest_cfg = manifest.get("run_cfg", {})
    if isinstance(run_manifest_cfg, dict):
        scaling = run_manifest_cfg.get("scaling", {})
        if isinstance(scaling, dict) and scaling.get("scale_label"):
            return dict(scaling)
    recipe = str(run_cfg.get("recipe", manifest.get("recipe", "")))
    if "scaling_a" in recipe:
        return {"scale_label": "scaling_a"}
    if "scaling_b" in recipe:
        return {"scale_label": "scaling_b"}
    return {}


def _actual_batch_from_training_manifest(
    *,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    train_dir: Path,
) -> tuple[int, int, int, int, int]:
    train_cfg = _deep_merge(common.get("train", {}), run_cfg.get("train", {}))
    seq_length = int(run_cfg.get("seq_length", common.get("seq_length", train_cfg.get("seq_length", 2048))))
    gbs = int(train_cfg.get("global_batch_size", train_cfg.get("global_batch", 1)))
    mbs = int(train_cfg.get("micro_batch_size", train_cfg.get("micro_batch", 1)))
    train_iters = int(train_cfg.get("train_iters", 0) or 0)
    save_interval = int(train_cfg.get("save_interval", train_cfg.get("save_interval_iters", 0)) or 0)
    manifest = _json_if_exists(train_dir / TRAINING_RUN_MANIFEST_NAME)
    run_manifest_cfg = manifest.get("run_cfg", {})
    if isinstance(run_manifest_cfg, dict):
        manifest_train_cfg = run_manifest_cfg.get("train", {})
        if isinstance(manifest_train_cfg, dict):
            gbs = int(manifest_train_cfg.get("global_batch_size", manifest_train_cfg.get("global_batch", gbs)) or gbs)
            mbs = int(manifest_train_cfg.get("micro_batch_size", manifest_train_cfg.get("micro_batch", mbs)) or mbs)
            seq_length = int(run_manifest_cfg.get("seq_length", manifest_train_cfg.get("seq_length", seq_length)) or seq_length)
            train_iters = int(manifest_train_cfg.get("train_iters", train_iters) or train_iters)
            save_interval = int(
                manifest_train_cfg.get("save_interval", manifest_train_cfg.get("save_interval_iters", save_interval)) or save_interval
            )
    metadata = manifest.get("metadata", {})
    if isinstance(metadata, dict):
        gbs = int(metadata.get("global_batch_size", gbs) or gbs)
        mbs = int(metadata.get("micro_batch_size", mbs) or mbs)
        seq_length = int(metadata.get("seq_length", seq_length) or seq_length)
        train_iters = int(metadata.get("train_iters", train_iters) or train_iters)
        save_interval = int(metadata.get("save_interval", metadata.get("save_interval_iters", save_interval)) or save_interval)
    return gbs, mbs, seq_length, train_iters, save_interval


def _checkpoint_dir_for_step(checkpoint_root: Path, step: int) -> Path:
    candidates = [
        checkpoint_root / f"iter_{step:07d}",
        checkpoint_root / f"iter_{step:010d}",
        checkpoint_root / f"iter_{step}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _checkpoint_step_from_dir(path: Path) -> int | None:
    match = re.fullmatch(r"iter_(\d+)", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _existing_checkpoint_steps(checkpoint_root: Path) -> list[int]:
    if not checkpoint_root.exists():
        return []
    steps: list[int] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        step = _checkpoint_step_from_dir(child)
        if step is not None:
            steps.append(step)
    return sorted(set(steps))


def _segmented_training_metadata(train_dir: Path) -> dict[str, Any] | None:
    manifest = _json_if_exists(train_dir / TRAINING_RUN_MANIFEST_NAME)
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict) or not metadata.get("segmented_training"):
        return None
    segments = metadata.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return None
    usable_segments = [dict(segment) for segment in segments if isinstance(segment, dict)]
    if not usable_segments:
        return None
    return {
        "segments": usable_segments,
        "target_global_batch_size": int(metadata.get("global_batch_size", usable_segments[-1].get("global_batch_size", 1)) or 1),
        "micro_batch_size": int(metadata.get("micro_batch_size", usable_segments[-1].get("micro_batch_size", 1)) or 1),
        "seq_length": int(metadata.get("seq_length", usable_segments[-1].get("seq_length", 2048)) or 2048),
        "train_iters": int(metadata.get("train_iters", usable_segments[-1].get("cumulative_end_step", 0)) or 0),
        "final_checkpoint_root": str(metadata.get("final_checkpoint_root", "")),
        "final_checkpoint_step": int(metadata.get("final_checkpoint_step", usable_segments[-1].get("cumulative_end_step", 0)) or 0),
    }


def _segment_checkpoint_root(segment: dict[str, Any]) -> Path:
    output_dir = str(segment.get("output_dir", "")).strip()
    return Path(output_dir) / "checkpoints" if output_dir else Path("checkpoints")


def _resolve_segmented_checkpoint_for_horizon(
    *,
    segmented: dict[str, Any],
    horizon_tokens: int,
    final_horizon: bool,
) -> dict[str, Any]:
    segments = list(segmented["segments"])
    nominal_step = cumulative_step_for_tokens(segments, int(horizon_tokens))
    candidates: list[tuple[str, int]] = []
    if final_horizon and int(segmented.get("final_checkpoint_step", 0) or 0) > 0:
        candidates.append(("final_train_iters", int(segmented["final_checkpoint_step"])))
    candidates.append(("nominal", nominal_step))

    deduped: list[tuple[str, int]] = []
    seen: set[int] = set()
    for source, step in candidates:
        if step <= 0 or step in seen:
            continue
        seen.add(step)
        deduped.append((source, step))

    roots = [(_segment_checkpoint_root(segment), segment) for segment in segments]
    for source, step in deduped:
        segment = segment_for_token_position(segments, int(horizon_tokens))
        root = _segment_checkpoint_root(segment)
        path = _checkpoint_dir_for_step(root, step)
        if path.exists():
            actual_tokens = tokens_for_cumulative_step(segments, step)
            return {
                "checkpoint_step": step,
                "checkpoint_path": path,
                "checkpoint_resolution": source,
                "checkpoint_exists": True,
                "nominal_checkpoint_step": nominal_step,
                "target_checkpoint_tokens": int(horizon_tokens),
                "actual_checkpoint_tokens": actual_tokens,
                "checkpoint_token_delta": actual_tokens - int(horizon_tokens),
            }

    existing: list[tuple[int, Path]] = []
    for root, _segment in roots:
        for step in _existing_checkpoint_steps(root):
            existing.append((step, _checkpoint_dir_for_step(root, step)))
    if existing:
        best_step, best_path = min(
            existing,
            key=lambda item: (
                abs(tokens_for_cumulative_step(segments, item[0]) - int(horizon_tokens)),
                0 if tokens_for_cumulative_step(segments, item[0]) >= int(horizon_tokens) else 1,
                item[0],
            ),
        )
        actual_tokens = tokens_for_cumulative_step(segments, best_step)
        return {
            "checkpoint_step": best_step,
            "checkpoint_path": best_path,
            "checkpoint_resolution": "nearest_existing",
            "checkpoint_exists": True,
            "nominal_checkpoint_step": nominal_step,
            "target_checkpoint_tokens": int(horizon_tokens),
            "actual_checkpoint_tokens": actual_tokens,
            "checkpoint_token_delta": actual_tokens - int(horizon_tokens),
        }

    source, step = deduped[0]
    planned_segment = segment_for_token_position(segments, int(horizon_tokens))
    path = _checkpoint_dir_for_step(_segment_checkpoint_root(planned_segment), step)
    actual_tokens = tokens_for_cumulative_step(segments, step)
    return {
        "checkpoint_step": step,
        "checkpoint_path": path,
        "checkpoint_resolution": f"{source}_planned",
        "checkpoint_exists": False,
        "nominal_checkpoint_step": nominal_step,
        "target_checkpoint_tokens": int(horizon_tokens),
        "actual_checkpoint_tokens": actual_tokens,
        "checkpoint_token_delta": actual_tokens - int(horizon_tokens),
    }


def _resolve_checkpoint_for_horizon(
    *,
    checkpoint_root: Path,
    horizon_tokens: int,
    global_batch_size: int,
    seq_length: int,
    save_interval: int,
    base_interval_tokens: int,
    base_train_iters: int,
    final_horizon: bool,
) -> dict[str, Any]:
    tokens_per_step = max(1, int(global_batch_size) * int(seq_length))
    nominal_step = _steps_for_tokens(horizon_tokens, global_batch_size=global_batch_size, seq_length=seq_length)
    candidates: list[tuple[str, int]] = []
    if final_horizon and base_train_iters > 0:
        candidates.append(("final_train_iters", int(base_train_iters)))
    if save_interval > 0 and base_interval_tokens > 0:
        checkpoint_index = max(1, round(int(horizon_tokens) / int(base_interval_tokens)))
        candidates.append(("save_interval", int(save_interval) * checkpoint_index))
    candidates.append(("nominal", nominal_step))

    deduped: list[tuple[str, int]] = []
    seen: set[int] = set()
    for source, step in candidates:
        if step <= 0 or step in seen:
            continue
        seen.add(step)
        deduped.append((source, step))

    for source, step in deduped:
        path = _checkpoint_dir_for_step(checkpoint_root, step)
        if path.exists():
            actual_tokens = step * tokens_per_step
            return {
                "checkpoint_step": step,
                "checkpoint_path": path,
                "checkpoint_resolution": source,
                "checkpoint_exists": True,
                "nominal_checkpoint_step": nominal_step,
                "target_checkpoint_tokens": int(horizon_tokens),
                "actual_checkpoint_tokens": actual_tokens,
                "checkpoint_token_delta": actual_tokens - int(horizon_tokens),
            }

    existing_steps = _existing_checkpoint_steps(checkpoint_root)
    if existing_steps:
        best_step = min(
            existing_steps,
            key=lambda step: (
                abs(step * tokens_per_step - int(horizon_tokens)),
                0 if step * tokens_per_step >= int(horizon_tokens) else 1,
                step,
            ),
        )
        actual_tokens = best_step * tokens_per_step
        return {
            "checkpoint_step": best_step,
            "checkpoint_path": _checkpoint_dir_for_step(checkpoint_root, best_step),
            "checkpoint_resolution": "nearest_existing",
            "checkpoint_exists": True,
            "nominal_checkpoint_step": nominal_step,
            "target_checkpoint_tokens": int(horizon_tokens),
            "actual_checkpoint_tokens": actual_tokens,
            "checkpoint_token_delta": actual_tokens - int(horizon_tokens),
        }

    source, step = deduped[0]
    actual_tokens = step * tokens_per_step
    return {
        "checkpoint_step": step,
        "checkpoint_path": _checkpoint_dir_for_step(checkpoint_root, step),
        "checkpoint_resolution": f"{source}_planned",
        "checkpoint_exists": False,
        "nominal_checkpoint_step": nominal_step,
        "target_checkpoint_tokens": int(horizon_tokens),
        "actual_checkpoint_tokens": actual_tokens,
        "checkpoint_token_delta": actual_tokens - int(horizon_tokens),
    }


def _override_key(override: str) -> str:
    key = str(override).split("=", 1)[0].strip()
    while key.startswith("+"):
        key = key[1:]
    return key


def _without_override_keys(overrides: list[Any], keys: set[str]) -> list[str]:
    return [str(item) for item in overrides if _override_key(str(item)) not in keys]


def _metric_from_log_line(line: str) -> dict[str, Any] | None:
    loss_match = _LM_LOSS_RE.search(line)
    if loss_match is None:
        return None
    step_match = _ITERATION_RE.search(line)
    skipped_match = _SKIPPED_RE.search(line)
    nan_match = _NAN_RE.search(line)
    consumed_match = _CONSUMED_RE.search(line)
    return {
        "loss": float(loss_match.group("value")),
        "global_step": int(step_match.group("step")) if step_match else None,
        "total_iterations": int(step_match.group("total")) if step_match else None,
        "consumed_samples": int(consumed_match.group("value")) if consumed_match else None,
        "skipped_iterations": int(skipped_match.group("value")) if skipped_match else None,
        "nan_iterations": int(nan_match.group("value")) if nan_match else None,
    }


def _metric_from_log(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists() or not log_path.is_file():
        return None
    latest: dict[str, Any] | None = None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line_number, line in enumerate(lines, start=1):
        parsed = _metric_from_log_line(line)
        if parsed is None:
            continue
        parsed["source"] = "log_lm_loss"
        parsed["source_path"] = str(log_path.resolve())
        parsed["source_line"] = line.strip()
        parsed["source_line_number"] = line_number
        current_step = parsed.get("global_step")
        latest_step = latest.get("global_step") if latest else None
        if latest is None or (
            isinstance(current_step, int)
            and (not isinstance(latest_step, int) or current_step >= latest_step)
        ):
            latest = parsed
    return latest


def _metric_from_readout_artifacts(output_dir: Path, log_path: Path | None = None) -> dict[str, Any]:
    payload = _json_if_exists(output_dir / "train_results.json")
    for key in ("eval_loss", "final_loss", "train_loss"):
        value = payload.get(key)
        if value is None or value == "":
            continue
        try:
            return {"loss": float(value), "source": key, "source_path": str((output_dir / "train_results.json").resolve())}
        except (TypeError, ValueError):
            continue
    if log_path is not None:
        parsed = _metric_from_log(log_path)
        if parsed is not None:
            return parsed
    return {"loss": None, "source": "", "source_path": "", "source_line": ""}


def _infer_language_count(ctx: PipelineContext, cfg: CooldownReadoutConfig) -> int | str:
    raw_cfg = getattr(ctx, "profile", {}).get("cooldown_readout", {})
    if isinstance(raw_cfg, dict):
        explicit = raw_cfg.get("k", raw_cfg.get("language_count", ""))
        if str(explicit).strip():
            try:
                return int(explicit)
            except (TypeError, ValueError):
                return str(explicit)
    candidates = [
        str(getattr(ctx, "output_root", "")),
        str(cfg.output_dir),
        str(getattr(ctx, "profile", {}).get("language_map_path", "")),
    ]
    for candidate in candidates:
        match = re.search(r"k(\d{3})", candidate)
        if match:
            return int(match.group(1))
    return ""


def _strict_failure_rows(rows: list[dict[str, Any]], *, allow_dry_run: bool) -> list[dict[str, Any]]:
    allowed = {"complete"}
    if allow_dry_run:
        allowed.add("dry_run")
    return [
        row for row in rows
        if str(row.get("status", "")).strip() not in allowed
    ]


def _raise_for_strict_failures(
    *,
    cfg: CooldownReadoutConfig,
    rows: list[dict[str, Any]],
    dry_run: bool,
    manifest_path: Path,
) -> None:
    if not cfg.strict or dry_run:
        return
    bad = _strict_failure_rows(rows, allow_dry_run=dry_run)
    if not bad:
        return
    preview = [
        "{run}/{target}: {status} {reason}".format(
            run=row.get("base_run_id", ""),
            target=row.get("target_checkpoint_tokens", row.get("checkpoint_tokens", "")),
            status=row.get("status", ""),
            reason=row.get("failure_reason", ""),
        )
        for row in bad[:8]
    ]
    more = "" if len(bad) <= 8 else f"\n- ... +{len(bad) - 8} more"
    raise SystemExit(
        "cooldown_readout strict validation failed; readout rows are not complete. "
        f"manifest={manifest_path}\n- " + "\n- ".join(preview) + more
    )


def _build_readout_rows(
    *,
    ctx: PipelineContext,
    cfg: CooldownReadoutConfig,
    common: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_cfg in runs:
        base_run_id = str(run_cfg["id"])
        train_dir = ctx.train_root / base_run_id
        scaling = _scale_metadata(run_cfg, train_dir)
        scale_label = str(scaling.get("scale_label", "")).strip()
        horizons = cfg.horizons_tokens_by_scale.get(scale_label, [])
        if not horizons:
            rows.append(
                {
                    "base_run_id": base_run_id,
                    "status": "skipped",
                    "failure_reason": f"no horizons configured for scale_label={scale_label!r}",
                }
            )
            continue
        gbs, mbs, seq_length, base_train_iters, manifest_save_interval = _actual_batch_from_training_manifest(
            run_cfg=run_cfg,
            common=common,
            train_dir=train_dir,
        )
        segmented = _segmented_training_metadata(train_dir)
        if segmented is not None:
            seq_length = int(segmented["seq_length"])
            base_train_iters = int(segmented["train_iters"])
        save_interval = int(scaling.get("save_interval", manifest_save_interval) or manifest_save_interval)
        checkpoint_root = train_dir / "checkpoints"
        training_complete = (train_dir / ".training_complete").exists()
        base_interval_tokens = min(horizons) if horizons else 0
        final_target_tokens = max(horizons) if horizons else 0
        for horizon_tokens in horizons:
            cooldown_tokens = min(int(horizon_tokens * cfg.cooldown_fraction), cfg.cooldown_max_tokens)
            segment_index = ""
            if segmented is not None:
                horizon_segment = segment_for_token_position(list(segmented["segments"]), int(horizon_tokens))
                gbs = int(horizon_segment["global_batch_size"])
                mbs = int(horizon_segment["micro_batch_size"])
                segment_index = int(horizon_segment["segment_index"])
            cooldown_iters = _steps_for_tokens(cooldown_tokens, global_batch_size=gbs, seq_length=seq_length)
            label = _token_label(horizon_tokens)
            readout_run_id = f"{base_run_id}_cooldown_{label}"
            readout_output_dir = cfg.output_dir / base_run_id / label
            if segmented is not None:
                checkpoint = _resolve_segmented_checkpoint_for_horizon(
                    segmented=segmented,
                    horizon_tokens=horizon_tokens,
                    final_horizon=int(horizon_tokens) == int(final_target_tokens),
                )
            else:
                checkpoint = _resolve_checkpoint_for_horizon(
                    checkpoint_root=checkpoint_root,
                    horizon_tokens=horizon_tokens,
                    global_batch_size=gbs,
                    seq_length=seq_length,
                    save_interval=save_interval,
                    base_interval_tokens=base_interval_tokens,
                    base_train_iters=base_train_iters,
                    final_horizon=int(horizon_tokens) == int(final_target_tokens),
                )
            checkpoint_step = int(checkpoint["checkpoint_step"])
            checkpoint_path = Path(checkpoint["checkpoint_path"])
            status = "planned"
            failure_reason = ""
            if cfg.require_training_complete and not training_complete and not ctx.args.dry_run:
                status = "blocked"
                failure_reason = f"training is not complete: {train_dir / '.training_complete'}"
            elif cfg.strict and checkpoint["checkpoint_resolution"] == "nearest_existing" and not ctx.args.dry_run:
                status = "blocked"
                failure_reason = (
                    "planned checkpoint is missing and nearest_existing fallback is not accepted in strict mode: "
                    f"{checkpoint_path}"
                )
            elif cfg.require_checkpoint and not checkpoint_path.exists() and not ctx.args.dry_run:
                status = "blocked"
                failure_reason = f"checkpoint is missing: {checkpoint_path}"
            rows.append(
                {
                    "base_run_id": base_run_id,
                    "readout_run_id": readout_run_id,
                    "status": status,
                    "failure_reason": failure_reason,
                    "scale_label": scale_label,
                    "family": scaling.get("family", ""),
                    "variant_label": scaling.get("variant_label", ""),
                    "num_experts": scaling.get("num_experts", ""),
                    "num_gpus": scaling.get("num_gpus", ""),
                    "target_checkpoint_tokens": checkpoint["target_checkpoint_tokens"],
                    "checkpoint_tokens": checkpoint["actual_checkpoint_tokens"],
                    "checkpoint_token_delta": checkpoint["checkpoint_token_delta"],
                    "checkpoint_step": checkpoint_step,
                    "nominal_checkpoint_step": checkpoint["nominal_checkpoint_step"],
                    "checkpoint_resolution": checkpoint["checkpoint_resolution"],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_exists": checkpoint["checkpoint_exists"],
                    "cooldown_tokens": cooldown_tokens,
                    "cooldown_iters": cooldown_iters,
                    "global_batch_size": gbs,
                    "micro_batch_size": mbs,
                    "seq_length": seq_length,
                    "base_train_iters": base_train_iters,
                    "save_interval": save_interval,
                    "batch_policy": "segmented" if segmented is not None else "",
                    "segment_index": segment_index,
                    "output_dir": str(readout_output_dir),
                    "log_path": str(ctx.logs_root / "cooldown_readout" / base_run_id / f"{label}.log"),
                }
            )
    return rows


def _readout_run_cfg(
    *,
    base_run_cfg: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    out = dict(base_run_cfg)
    out["id"] = str(row["readout_run_id"])
    train_cfg = _deep_merge(out.get("train", {}), {})
    train_cfg["global_batch_size"] = int(row["global_batch_size"])
    train_cfg["global_batch"] = int(row["global_batch_size"])
    train_cfg["micro_batch_size"] = int(row["micro_batch_size"])
    train_cfg["micro_batch"] = int(row["micro_batch_size"])
    train_cfg["train_iters"] = int(row["cooldown_iters"])
    train_cfg["token_budget_tokens"] = int(row["cooldown_tokens"])
    train_cfg["save_interval"] = int(row["cooldown_iters"])
    train_cfg["save_interval_iters"] = int(row["cooldown_iters"])
    train_cfg["batch_policy"] = {"enabled": False}
    out["train"] = train_cfg
    out["eval"] = {
        "eval_interval": int(row["cooldown_iters"]),
        "eval_iters": int(row.get("eval_iters", 10) or 10),
    }
    inherited_overrides = list(out.get("overrides", [])) if isinstance(out.get("overrides", []), list) else []
    overrides = _without_override_keys(inherited_overrides, _READOUT_EVAL_OVERRIDE_KEYS)
    overrides.extend(
        [
            f"checkpoint.load={Path(str(row['checkpoint_path'])).parent}",
            f"checkpoint.ckpt_step={int(row['checkpoint_step'])}",
            "checkpoint.finetune=true",
            "checkpoint.load_optim=false",
            "checkpoint.load_rng=false",
            "checkpoint.load_main_params_from_ckpt=true",
            "checkpoint.exit_on_missing_checkpoint=true",
            f"checkpoint.save={Path(str(row['output_dir'])) / 'checkpoints'}",
            f"checkpoint.save_interval={int(row['cooldown_iters'])}",
            "checkpoint.async_save=true",
            "checkpoint.async_strategy=mcore",
            f"validation.eval_interval={int(row['cooldown_iters'])}",
            f"validation.eval_iters={int(row.get('eval_iters', 10) or 10)}",
        ]
    )
    out["overrides"] = overrides
    return out


def stage_cooldown_readout(ctx: PipelineContext) -> None:
    if "cooldown_readout" not in ctx.active_stages:
        return
    cfg = resolve_cooldown_readout_config(ctx)
    if not cfg.enabled:
        return

    common, resolved_model_path, all_runs = resolve_training_stage_config(ctx)
    runs = _selected_runs(ctx, all_runs)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = cfg.output_dir / COOLDOWN_PLAN
    manifest_path = cfg.output_dir / COOLDOWN_MANIFEST
    observations_path = cfg.output_dir / FIT_OBSERVATIONS_CSV
    signature = _stage_signature(
        {
            "stage": "cooldown_readout",
            "cfg": getattr(ctx, "profile", {}).get("cooldown_readout", {}),
            "runs": runs,
        }
    )
    expected = [plan_path, manifest_path, observations_path]
    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="cooldown_readout",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        cached = _json_if_exists(manifest_path)
        cached_rows = cached.get("rows", [])
        if isinstance(cached_rows, list):
            _raise_for_strict_failures(
                cfg=cfg,
                rows=[row for row in cached_rows if isinstance(row, dict)],
                dry_run=ctx.args.dry_run,
                manifest_path=manifest_path,
            )
        ctx.logger.info("[SKIP] cooldown_readout stage (cache hit)")
        return

    rows = _build_readout_rows(ctx=ctx, cfg=cfg, common=common, runs=runs)
    for row in rows:
        row["eval_iters"] = cfg.eval_iters
    _dump_json(plan_path, {"schema_version": 1, "stage": "cooldown_readout", "rows": rows})
    _write_csv(cfg.output_dir / COOLDOWN_PLAN_CSV, rows)

    pending: list[tuple[dict[str, Any], Any]] = []
    runs_by_id = {str(run["id"]): dict(run) for run in runs}
    for row in rows:
        if row.get("status") != "planned":
            continue
        readout_run_cfg = _readout_run_cfg(base_run_cfg=runs_by_id[str(row["base_run_id"])], row=row)
        plan = build_training_run_plan(
            ctx,
            run=readout_run_cfg,
            common=common,
            output_dir=Path(str(row["output_dir"])),
            log_path=Path(str(row["log_path"])),
        )
        Path(str(row["output_dir"])).mkdir(parents=True, exist_ok=True)
        command_plan_path = Path(str(row["output_dir"])) / "training_command_plan.json"
        row["command_plan"] = str(command_plan_path)
        if ctx.args.dry_run:
            write_training_command_plan(plan)
            row["status"] = "dry_run"
            continue
        if core_run_artifacts_exist(plan):
            if not (plan.output_dir / TRAINING_RUN_MANIFEST_NAME).exists():
                write_training_run_manifest(
                    plan=plan,
                    status="complete",
                    model_name_or_path=resolved_model_path,
                    tokenized_path=ctx.tokenized_dir,
                )
            metric = _metric_from_readout_artifacts(plan.output_dir, plan.log_path)
            row["status"] = "complete"
            row["loss"] = metric["loss"] if metric.get("loss") is not None else ""
            row["loss_source"] = metric.get("source", "")
            row["loss_source_path"] = metric.get("source_path", "")
            row["loss_source_line"] = metric.get("source_line", "")
            continue
        write_training_command_plan(plan)
        if not cfg.run_forks:
            row["status"] = "planned_not_run"
            continue
        pending.append((row, plan))

    if pending:
        jobs = [(plan.cmd, plan.env, plan.log_path) for _row, plan in pending]
        ctx.executor.run_batch(jobs, dry_run=False)

    for row, plan in pending:
        if core_run_artifacts_exist(plan):
            write_training_run_manifest(
                plan=plan,
                status="complete",
                model_name_or_path=resolved_model_path,
                tokenized_path=ctx.tokenized_dir,
            )
            metric = _metric_from_readout_artifacts(plan.output_dir, plan.log_path)
            row["status"] = "complete"
            row["loss"] = metric["loss"] if metric.get("loss") is not None else ""
            row["loss_source"] = metric.get("source", "")
            row["loss_source_path"] = metric.get("source_path", "")
            row["loss_source_line"] = metric.get("source_line", "")
        else:
            row["status"] = "missing_artifacts"
            row["failure_reason"] = "readout run did not produce core training artifacts"

    _dump_json(plan_path, {"schema_version": 1, "stage": "cooldown_readout", "rows": rows})
    _write_csv(cfg.output_dir / COOLDOWN_PLAN_CSV, rows)
    _dump_json(manifest_path, {"schema_version": 1, "stage": "cooldown_readout", "rows": rows})
    _write_csv(cfg.output_dir / COOLDOWN_MANIFEST_CSV, rows)
    observation_rows = [
        {
            "K": _infer_language_count(ctx, cfg),
            "E": row.get("num_experts", ""),
            "ladder": "A" if row.get("scale_label") == "scaling_a" else ("B" if row.get("scale_label") == "scaling_b" else ""),
            "family": row.get("family", ""),
            "scale_label": row.get("scale_label", ""),
            "D_checkpoint": row.get("checkpoint_tokens", ""),
            "D_checkpoint_target": row.get("target_checkpoint_tokens", ""),
            "checkpoint_step": row.get("checkpoint_step", ""),
            "checkpoint_resolution": row.get("checkpoint_resolution", ""),
            "checkpoint_token_delta": row.get("checkpoint_token_delta", ""),
            "cooldown_tokens": row.get("cooldown_tokens", ""),
            "loss": row.get("loss", ""),
            "loss_source": row.get("loss_source", ""),
            "loss_source_path": row.get("loss_source_path", ""),
            "loss_source_line": row.get("loss_source_line", ""),
            "source_run_id": row.get("base_run_id", ""),
            "readout_run_id": row.get("readout_run_id", ""),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "status": row.get("status", ""),
            "failure_reason": row.get("failure_reason", ""),
        }
        for row in rows
    ]
    _write_csv(observations_path, observation_rows)
    _raise_for_strict_failures(cfg=cfg, rows=rows, dry_run=ctx.args.dry_run, manifest_path=manifest_path)
    if not ctx.args.dry_run:
        _save_stage_state(ctx.state_path, "cooldown_readout", signature, [str(path) for path in expected])
    ctx.logger.info("[OK] cooldown_readout=%s", manifest_path)


class CooldownReadoutStage:
    name = "cooldown_readout"

    def plan(self, ctx: Any) -> StagePlan:
        cfg = resolve_cooldown_readout_config(ctx)
        artifacts = ()
        if cfg.enabled:
            artifacts = (
                ArtifactSpec(cfg.output_dir / COOLDOWN_PLAN, description="checkpoint readout plan"),
                ArtifactSpec(cfg.output_dir / COOLDOWN_MANIFEST, description="cooldown readout manifest"),
                ArtifactSpec(cfg.output_dir / FIT_OBSERVATIONS_CSV, description="fit-ready checkpoint observations"),
            )
        return StagePlan(stage=self.name, enabled=cfg.enabled, expected_artifacts=artifacts)

    def run(self, ctx: Any, plan: StagePlan) -> StageResult:
        if not plan.enabled:
            return StageResult(stage=self.name, status="skipped")
        stage_cooldown_readout(ctx)
        cfg = resolve_cooldown_readout_config(ctx)
        return StageResult(
            stage=self.name,
            artifacts=(cfg.output_dir / COOLDOWN_PLAN, cfg.output_dir / COOLDOWN_MANIFEST, cfg.output_dir / FIT_OBSERVATIONS_CSV),
        )

    def validate(self, ctx: Any, result: StageResult) -> None:
        del ctx
        missing = [path for path in result.artifacts if not path.exists()]
        if missing:
            raise SystemExit(f"cooldown_readout missing expected artifacts: {missing}")


stage = CooldownReadoutStage()


__all__ = [
    "CooldownReadoutStage",
    "resolve_cooldown_readout_config",
    "stage",
    "stage_cooldown_readout",
]
