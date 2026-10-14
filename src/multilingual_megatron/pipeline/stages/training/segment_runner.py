from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .checkpoint_state import inspect_full_state_checkpoint, write_checkpoint_audit


TRAINING_RUN_MANIFEST_NAME = "training_run_manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_extra_override(cmd: list[str], key: str, value: str) -> list[str]:
    updated: list[str] = []
    index = 0
    while index < len(cmd):
        if cmd[index] == "--extra-override" and index + 1 < len(cmd):
            override = cmd[index + 1]
            if override.split("=", 1)[0] == key:
                index += 2
                continue
        updated.append(cmd[index])
        index += 1
    updated.extend(["--extra-override", f"{key}={value}"])
    return updated


def _v2_checkpoint_state(segment: dict[str, Any]):
    if not (
        segment.get("fixed_dataset_sample_horizon")
        or segment.get("require_full_state_checkpoint")
    ):
        return None
    output_dir = Path(str(segment["output_dir"]))
    return inspect_full_state_checkpoint(
        checkpoint_root=output_dir / "checkpoints",
        expected_step=int(segment["expected_end_step"]),
        expected_consumed_samples=int(segment["expected_consumed_samples"]),
    )


def _completed_result(segment: dict[str, Any], *, status: str, runtime_seconds: float = 0.0) -> dict[str, Any]:
    output_dir = Path(str(segment["output_dir"]))
    log_path = Path(str(segment["log_path"]))
    return {
        "segment_index": int(segment["segment_index"]),
        "status": status,
        "return_code": 0,
        "runtime_seconds": runtime_seconds,
        "output_dir": str(output_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "training_complete": (output_dir / ".training_complete").exists(),
        "train_results": str((output_dir / "train_results.json").resolve()),
        "trainer_state": str((output_dir / "trainer_state.json").resolve()),
        "checkpoint_root": str((output_dir / "checkpoints").resolve()),
        "cumulative_end_step": int(segment["cumulative_end_step"]),
        "actual_end_tokens": int(segment["actual_end_tokens"]),
        "expected_consumed_samples": int(segment.get("expected_consumed_samples", 0) or 0),
    }


def _run_segment(segment: dict[str, Any], *, env: dict[str, str], cwd: str | None) -> dict[str, Any]:
    cmd = [str(item) for item in segment["cmd"]]
    log_path = Path(str(segment["log_path"]))
    output_dir = Path(str(segment["output_dir"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_checkpoint = segment.get("initial_checkpoint")
    resume_state = None
    if isinstance(initial_checkpoint, dict):
        parent_state = inspect_full_state_checkpoint(
            checkpoint_root=Path(str(initial_checkpoint.get("checkpoint_root", ""))),
            checkpoint_step=int(initial_checkpoint.get("expected_step", 0) or 0),
            expected_step=int(initial_checkpoint.get("expected_step", 0) or 0),
            expected_consumed_samples=int(
                initial_checkpoint.get("expected_consumed_samples", 0) or 0
            ),
        )
        write_checkpoint_audit(output_dir / "parent_checkpoint_state_audit.json", parent_state)
        if not parent_state.valid_full_state:
            raise RuntimeError(
                f"Segment {segment['segment_index']} has an invalid branch parent checkpoint: "
                f"{parent_state.errors}"
            )
        resume_state = parent_state

    checkpoint_state = _v2_checkpoint_state(segment)
    if checkpoint_state is not None and checkpoint_state.valid_full_state:
        write_checkpoint_audit(output_dir / "checkpoint_state_audit.json", checkpoint_state)
        if not (output_dir / ".training_complete").exists():
            (output_dir / ".training_complete").write_text("recovered_from_verified_checkpoint\n", encoding="utf-8")
        result = _completed_result(segment, status="complete")
        result["execution"] = "verified_checkpoint_skip"
        return result

    if checkpoint_state is not None and checkpoint_state.latest_step is not None:
        latest_step = int(checkpoint_state.latest_step)
        start_step = int(segment["cumulative_start_step"])
        end_step = int(segment["expected_end_step"])
        if latest_step > end_step:
            raise RuntimeError(
                f"Segment {segment['segment_index']} checkpoint step {latest_step} exceeds expected endpoint {end_step}"
            )
        if start_step < latest_step < end_step:
            partial = inspect_full_state_checkpoint(
                checkpoint_root=output_dir / "checkpoints",
                expected_step=latest_step,
            )
            write_checkpoint_audit(output_dir / "checkpoint_state_resume_audit.json", partial)
            if not partial.valid_full_state:
                raise RuntimeError(
                    f"Segment {segment['segment_index']} has an invalid partial checkpoint: {partial.errors}"
                )
            resume_state = partial
            for key, value in (
                ("checkpoint.load", str(output_dir / "checkpoints")),
                ("checkpoint.ckpt_step", str(latest_step)),
                ("checkpoint.load_optim", "true"),
                ("checkpoint.load_rng", "true"),
                ("checkpoint.finetune", "false"),
                ("checkpoint.exit_on_missing_checkpoint", "true"),
            ):
                cmd = _replace_extra_override(cmd, key, value)

    if resume_state is not None:
        # Bind execution to the same full state we audited. The wrapper pins
        # the absolute iter_* directory and installs a pre-update state guard.
        cmd.extend([
            "--resume-checkpoint-root", str(Path(resume_state.checkpoint_root).resolve()),
            "--resume-checkpoint-step", str(resume_state.selected_step),
            "--resume-checkpoint-consumed-samples", str(resume_state.consumed_train_samples),
        ])

    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[SEGMENT] start index={segment['segment_index']} cmd={' '.join(cmd)}\n")
        handle.flush()
        completed = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[SEGMENT] end index={segment['segment_index']} return_code={completed.returncode}\n")
    runtime_seconds = time.time() - started
    if (
        segment.get("fixed_dataset_sample_horizon")
        or segment.get("require_full_state_checkpoint")
    ):
        verified = _v2_checkpoint_state(segment)
        if verified is not None:
            write_checkpoint_audit(output_dir / "checkpoint_state_audit.json", verified)
        status = (
            "complete"
            if completed.returncode == 0
            and verified is not None
            and verified.valid_full_state
            and (output_dir / ".training_complete").exists()
            else "failed"
        )
    else:
        status = "complete" if completed.returncode == 0 and (output_dir / ".training_complete").exists() else "failed"
    result = _completed_result(segment, status=status, runtime_seconds=runtime_seconds)
    result["return_code"] = int(completed.returncode)
    return result


def _latest_checkpoint_path(checkpoint_root: Path, step: int) -> Path:
    candidates = [
        checkpoint_root / f"iter_{step:07d}",
        checkpoint_root / f"iter_{step:010d}",
        checkpoint_root / f"iter_{step}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parent_manifest_metadata(plan: dict[str, Any], final_result: dict[str, Any], final_step: int) -> dict[str, Any]:
    policy = plan.get("policy", {})
    segments = plan.get("segments", [])
    if not isinstance(policy, dict):
        policy = {}
    if not isinstance(segments, list):
        segments = []
    seq_length = int(policy.get("seq_length", final_result.get("seq_length", 2048)) or 2048)
    return {
        "segmented_training": True,
        "seq_length": seq_length,
        "train_iters": int(final_step),
        "global_batch_size": int(policy.get("target_global_batch_size", plan.get("global_batch_size", 1)) or 1),
        "micro_batch_size": int(plan.get("plateau_micro_batch_size", final_result.get("micro_batch_size", 1)) or 1),
        "expected_tokens": int(plan.get("expected_tokens", 0) or 0),
        "target_token_budget_tokens": int(plan.get("target_token_budget_tokens", 0) or 0),
        "batch_policy": {
            "mode": policy.get("mode", "token_ramp"),
            "planning_method": policy.get("planning_method", ""),
            "target_global_batch_size": int(policy.get("target_global_batch_size", 0) or 0),
            "mbs_policy": policy.get("mbs_policy", ""),
            "schedule": policy.get("schedule", []),
        },
        "segments": [
            {
                key: value
                for key, value in segment.items()
                if key
                in {
                    "segment_index",
                    "target_start_tokens",
                    "target_end_tokens",
                    "actual_start_tokens",
                    "actual_end_tokens",
                    "actual_start_samples",
                    "actual_end_samples",
                    "global_batch_size",
                    "micro_batch_size",
                    "tokens_per_step",
                    "segment_train_iters",
                    "cumulative_start_step",
                    "cumulative_end_step",
                    "expected_end_step",
                    "expected_consumed_samples",
                    "fixed_dataset_sample_horizon",
                    "output_dir",
                    "log_path",
                }
            }
            for segment in segments
            if isinstance(segment, dict)
        ],
        "final_checkpoint_root": str(Path(str(final_result["checkpoint_root"])).resolve()),
        "final_checkpoint_step": int(final_step),
    }


def _write_parent_artifacts(*, output_dir: Path, plan: dict[str, Any], results: list[dict[str, Any]], runtime: float) -> None:
    segments = list(plan["segments"])
    final_segment = segments[-1]
    final_result = results[-1]
    final_output_dir = Path(str(final_result["output_dir"]))
    final_step = int(final_segment["cumulative_end_step"])
    final_checkpoint = _latest_checkpoint_path(Path(str(final_result["checkpoint_root"])), final_step)

    final_train_results = _load_json(final_output_dir / "train_results.json")
    final_trainer_state = _load_json(final_output_dir / "trainer_state.json")
    metadata = _parent_manifest_metadata(plan, final_result, final_step)
    segment_summary = {
        "schema_version": 2 if plan.get("planning_method") else 1,
        "kind": "segmented_training_results",
        "status": "complete",
        "segments": results,
        "final_checkpoint_root": str(Path(str(final_result["checkpoint_root"])).resolve()),
        "final_checkpoint_path": str(final_checkpoint.resolve()) if final_checkpoint.exists() else str(final_checkpoint),
        "final_checkpoint_exists": final_checkpoint.exists(),
        "final_checkpoint_step": final_step,
        "expected_tokens": int(plan["expected_tokens"]),
        "target_token_budget_tokens": int(plan["target_token_budget_tokens"]),
        "runtime_seconds": runtime,
    }
    _dump_json(output_dir / "segment_results.json", segment_summary)

    final_train_results.update(
        {
            "status": "complete",
            "segmented_training": True,
            "segment_results": str((output_dir / "segment_results.json").resolve()),
            "num_input_tokens_seen": int(
                plan.get("expected_consumed_samples", 0) * plan.get("policy", {}).get("seq_length", 0)
                or plan["expected_tokens"]
            ),
            "target_token_budget_tokens": int(plan["target_token_budget_tokens"]),
            "final_checkpoint_root": segment_summary["final_checkpoint_root"],
            "final_checkpoint_path": segment_summary["final_checkpoint_path"],
            "final_checkpoint_step": final_step,
        }
    )
    _dump_json(output_dir / "train_results.json", final_train_results)

    final_trainer_state.update(
        {
            "status": "complete",
            "segmented_training": True,
            "global_step": final_step,
            "segment_results": str((output_dir / "segment_results.json").resolve()),
        }
    )
    _dump_json(output_dir / "trainer_state.json", final_trainer_state)
    _dump_json(
        output_dir / TRAINING_RUN_MANIFEST_NAME,
        {
            "schema_version": 2 if plan.get("planning_method") else 1,
            "stage": "train",
            "run_id": str(plan.get("run_id", "")),
            "run_name": str(plan.get("run_name", "")),
            "status": "complete",
            "output_dir": str(output_dir.resolve()),
            "metadata": metadata,
            "artifacts": {
                "train_results": str((output_dir / "train_results.json").resolve()),
                "trainer_state": str((output_dir / "trainer_state.json").resolve()),
                "training_complete": str((output_dir / ".training_complete").resolve()),
                "segment_results": str((output_dir / "segment_results.json").resolve()),
                "command_plan": str((output_dir / "training_command_plan.json").resolve()),
            },
        },
    )
    (output_dir / ".training_complete").write_text("ok\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-iters", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--seq-length", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.segment_plan).resolve()
    plan = _load_json(plan_path)
    if not plan or not isinstance(plan.get("segments"), list):
        raise SystemExit(f"Invalid segmented training plan: {plan_path}")

    env = dict(os.environ)
    cwd = str(plan.get("cwd", "")).strip() or None
    started = time.time()
    results: list[dict[str, Any]] = []
    for segment in plan["segments"]:
        try:
            result = _run_segment(segment, env=env, cwd=cwd)
        except Exception as exc:
            result = _completed_result(segment, status="invalid")
            result["return_code"] = 2
            result["error"] = repr(exc)
        results.append(result)
        _dump_json(
            output_dir / "segment_results.partial.json",
            {"schema_version": 2 if plan.get("planning_method") else 1, "segments": results},
        )
        if result["status"] != "complete":
            _dump_json(
                output_dir / "segment_results.json",
                {
                    "schema_version": 2 if plan.get("planning_method") else 1,
                    "kind": "segmented_training_results",
                    "status": "failed",
                    "segments": results,
                    "failed_segment_index": result["segment_index"],
                    "runtime_seconds": time.time() - started,
                },
            )
            raise SystemExit(int(result["return_code"]) or 1)

    _write_parent_artifacts(output_dir=output_dir, plan=plan, results=results, runtime=time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
