"""Fail-closed inspection of Megatron full-state checkpoints."""

from __future__ import annotations

import json
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CheckpointState:
    checkpoint_root: str
    checkpoint_path: str
    # ``latest_step`` is the root tracker value.  ``selected_step`` is the
    # checkpoint that was actually inspected; ordinarily they are identical,
    # but continuation validation may deliberately inspect an earlier,
    # immutable branch point while the parent run has since advanced.
    latest_step: int | None
    selected_step: int | None
    train_state_step: int | None
    consumed_train_samples: int | None
    scheduler_num_steps: int | None
    has_model_state: bool
    has_optimizer_state: bool
    has_rng_state: bool
    has_scheduler_state: bool
    valid_full_state: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_step(checkpoint_root: Path) -> int | None:
    pointer = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not pointer.is_file():
        return None
    try:
        return int(pointer.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def checkpoint_path_for_step(checkpoint_root: Path, step: int) -> Path:
    candidates = (
        checkpoint_root / f"iter_{int(step):07d}",
        checkpoint_root / f"iter_{int(step):010d}",
        checkpoint_root / f"iter_{int(step)}",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _load_torch_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {}


def _distributed_metadata(checkpoint_path: Path) -> Any:
    metadata_path = checkpoint_path / ".metadata"
    if not metadata_path.is_file():
        return None
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(str(checkpoint_path)).read_metadata()


def _distributed_metadata_keys(checkpoint_path: Path) -> set[str]:
    metadata = _distributed_metadata(checkpoint_path)
    if metadata is None:
        return set()
    return {str(key) for key in metadata.state_dict_metadata}


def _load_distributed_common_state(checkpoint_path: Path) -> dict[str, Any]:
    """Load Megatron's replicated common state from a Torch DCP checkpoint."""

    metadata = _distributed_metadata(checkpoint_path)
    if metadata is None:
        return {}
    common_keys = {
        str(key) for key in metadata.state_dict_metadata if str(key).startswith("common_state/")
    }
    if not common_keys:
        return {}

    import torch

    payloads: list[dict[str, Any]] = []
    for index, storage in metadata.storage_data.items():
        if str(index.fqn) not in common_keys:
            continue
        storage_path = checkpoint_path / storage.relative_path
        with storage_path.open("rb") as handle:
            handle.seek(storage.offset)
            raw = handle.read(storage.length)
        if len(raw) != storage.length:
            raise OSError(
                f"short distributed common-state read: {len(raw)} != {storage.length}"
            )
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payloads.append(payload)
        elif isinstance(payload, list):
            payloads.extend(item for item in payload if isinstance(item, dict))

    if not payloads:
        return {}
    first = payloads[0]
    if any(payload != first for payload in payloads[1:]):
        raise ValueError("distributed common-state shards disagree")
    return first


def inspect_full_state_checkpoint(
    *,
    checkpoint_root: Path,
    expected_step: int | None = None,
    expected_consumed_samples: int | None = None,
    checkpoint_step: int | None = None,
) -> CheckpointState:
    """Inspect a full-state checkpoint without silently changing its target.

    With the default ``checkpoint_step=None`` this preserves the historical
    behaviour: inspect the checkpoint named by the root's latest pointer and
    reject an ``expected_step`` that differs from that pointer.  Supplying an
    explicit ``checkpoint_step`` instead selects that immutable iteration
    directly.  This is required for scientifically valid continuation checks:
    a later checkpoint in the same source root must not make an intended
    pre-cooldown branch point uninspectable.
    """

    checkpoint_root = checkpoint_root.resolve()
    errors: list[str] = []
    latest_step = _latest_step(checkpoint_root)
    selected_step = int(checkpoint_step) if checkpoint_step is not None else latest_step
    if checkpoint_step is not None and selected_step < 0:
        errors.append(f"invalid_selected_checkpoint_step:{selected_step}")
    if selected_step is None:
        errors.append("missing_or_invalid_latest_checkpoint_pointer")
        checkpoint_path = checkpoint_path_for_step(checkpoint_root, int(expected_step or 0))
    else:
        checkpoint_path = checkpoint_path_for_step(checkpoint_root, selected_step)
    if (
        checkpoint_step is None
        and latest_step is not None
        and expected_step is not None
        and latest_step != int(expected_step)
    ):
        errors.append(f"latest_step_mismatch:{latest_step}!={int(expected_step)}")
    if not checkpoint_path.is_dir():
        errors.append("checkpoint_directory_missing")

    train_state: dict[str, Any] = {}
    common: dict[str, Any] = {}
    metadata_keys: set[str] = set()
    if checkpoint_path.is_dir():
        try:
            train_state = _load_torch_dict(checkpoint_path / "train_state.pt")
        except Exception as exc:
            errors.append(f"train_state_unreadable:{type(exc).__name__}")
        try:
            common = _load_torch_dict(checkpoint_path / "common.pt")
        except Exception as exc:
            errors.append(f"common_state_unreadable:{type(exc).__name__}")
        if not common:
            try:
                common = _load_distributed_common_state(checkpoint_path)
            except Exception as exc:
                errors.append(f"distributed_common_state_unreadable:{type(exc).__name__}")
        try:
            metadata_keys = _distributed_metadata_keys(checkpoint_path)
        except Exception as exc:
            errors.append(f"distributed_metadata_unreadable:{type(exc).__name__}")

    train_state_step = _as_int(train_state.get("step"))
    consumed_samples = _as_int(train_state.get("consumed_train_samples"))
    if train_state_step is None:
        errors.append("train_state_step_missing")
    elif selected_step is not None and train_state_step != selected_step:
        errors.append(f"train_state_step_mismatch:{train_state_step}!={selected_step}")
    if expected_step is not None and train_state_step != int(expected_step):
        errors.append(f"expected_train_state_step_mismatch:{train_state_step}!={int(expected_step)}")
    if consumed_samples is None:
        errors.append("consumed_train_samples_missing")
    elif expected_consumed_samples is not None and consumed_samples != int(expected_consumed_samples):
        errors.append(
            f"consumed_train_samples_mismatch:{consumed_samples}!={int(expected_consumed_samples)}"
        )

    has_model = any(
        key.startswith(("decoder.", "embedding.", "output_layer."))
        for key in metadata_keys
    )
    # Megatron stores the ordinary optimizer state under ``optimizer.*``.
    # With the distributed optimizer enabled it instead nests the sharded
    # states below entries such as ``chained_0.optimizer.distributed.*``.
    # Require the explicit optimizer namespace in either representation; do
    # not accept a merely non-empty ``common[\"optimizer\"]`` marker.
    has_optimizer_metadata = any(
        key.startswith("optimizer.") or ".optimizer." in key for key in metadata_keys
    )
    has_optimizer = (
        isinstance(common.get("optimizer"), dict)
        and bool(common["optimizer"])
        and has_optimizer_metadata
    )
    has_rng = any(key.startswith("rng_state/") for key in metadata_keys)
    scheduler_state = common.get("opt_param_scheduler")
    has_scheduler = isinstance(scheduler_state, dict) and bool(scheduler_state)
    scheduler_num_steps = (
        _as_int(scheduler_state.get("num_steps")) if isinstance(scheduler_state, dict) else None
    )
    if has_scheduler and scheduler_num_steps is None:
        errors.append("scheduler_num_steps_missing")
    elif (
        scheduler_num_steps is not None
        and expected_consumed_samples is not None
        and scheduler_num_steps != int(expected_consumed_samples)
    ):
        errors.append(
            f"scheduler_num_steps_mismatch:{scheduler_num_steps}!={int(expected_consumed_samples)}"
        )
    for present, error in (
        (has_model, "model_state_missing"),
        (has_optimizer, "optimizer_state_missing"),
        (has_rng, "rng_state_missing"),
        (has_scheduler, "scheduler_state_missing"),
    ):
        if not present:
            errors.append(error)

    return CheckpointState(
        checkpoint_root=str(checkpoint_root),
        checkpoint_path=str(checkpoint_path.resolve()),
        latest_step=latest_step,
        selected_step=selected_step,
        train_state_step=train_state_step,
        consumed_train_samples=consumed_samples,
        scheduler_num_steps=scheduler_num_steps,
        has_model_state=has_model,
        has_optimizer_state=has_optimizer,
        has_rng_state=has_rng,
        has_scheduler_state=has_scheduler,
        valid_full_state=not errors,
        errors=tuple(errors),
    )


def write_checkpoint_audit(path: Path, state: CheckpointState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CheckpointState",
    "checkpoint_path_for_step",
    "inspect_full_state_checkpoint",
    "write_checkpoint_audit",
]
