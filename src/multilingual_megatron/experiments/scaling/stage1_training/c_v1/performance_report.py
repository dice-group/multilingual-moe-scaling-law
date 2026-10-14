#!/usr/bin/env python3
"""Seal direct, long-run C-ladder throughput evidence.

The C workflow deliberately consumes only selected, completed 134M-token
probes.  This small tool turns the generic matrix CSV into a hash-pinned,
fail-closed report that the production workflow can use for its exact
GBS32/64/128 GPU-hour forecast.  It validates the final-window measurements,
memory margin, and training-health lines rather than trusting a scheduler
exit code alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from multilingual_megatron.experiments.scaling.ab_v2.common import (
    REPO_ROOT,
    _payload_sha256,
)


SCHEMA_VERSION = 1
PROBE_TOKENS = 134_217_728
EXPECTED_EXECUTION = {
    1: {"micro_batch_size": 16, "num_gpus": 1},
    4: {"micro_batch_size": 32, "num_gpus": 1},
    16: {"micro_batch_size": 16, "num_gpus": 1},
    32: {"micro_batch_size": 8, "num_gpus": 2},
}
LOSS_RE = re.compile(r"\blm loss:\s*([^\s|]+)", re.IGNORECASE)
SKIPPED_RE = re.compile(r"\bnumber of skipped iterations:\s*(\d+)", re.IGNORECASE)
NAN_RE = re.compile(r"\bnumber of nan iterations:\s*(\d+)", re.IGNORECASE)
DROP_RE = re.compile(r"\bdropped_assignment_rate:\s*([^\s|]+)", re.IGNORECASE)
ZERO_KEPT_RE = re.compile(r"\btokens_with_zero_kept_experts:\s*(\d+)", re.IGNORECASE)
ITERATION_PROGRESS_RE = re.compile(
    r"\biteration\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE
)


class CPerformanceReportError(ValueError):
    """Raised when a C direct-throughput artifact is incomplete or unsafe."""


def _abs(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise CPerformanceReportError(f"probe row lacks numeric {key}") from exc
    if not math.isfinite(value):
        raise CPerformanceReportError(f"probe row has non-finite {key}")
    return value


def _int(row: Mapping[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CPerformanceReportError(f"probe row lacks integer {key}") from exc


def _health(
    log_path: Path,
    *,
    require_moe_routing: bool,
    expected_train_iters: int,
) -> dict[str, int | float]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CPerformanceReportError(f"cannot read probe log {log_path}: {exc}") from exc
    skipped = [int(value) for value in SKIPPED_RE.findall(text)]
    nans = [int(value) for value in NAN_RE.findall(text)]
    drops = [float(value) for value in DROP_RE.findall(text)]
    zero_kept = [int(value) for value in ZERO_KEPT_RE.findall(text)]
    progress = [
        (int(current), int(total))
        for current, total in ITERATION_PROGRESS_RE.findall(text)
    ]
    losses: list[float] = []
    for raw in LOSS_RE.findall(text):
        try:
            losses.append(float(raw))
        except ValueError as exc:
            raise CPerformanceReportError(f"probe log has malformed lm loss {raw!r}") from exc
    if not skipped or not nans or not losses:
        raise CPerformanceReportError(f"probe log lacks required health metrics: {log_path}")
    if any(value != 0 for value in skipped):
        raise CPerformanceReportError(f"probe has skipped optimizer steps: {log_path}")
    if any(value != 0 for value in nans) or any(not math.isfinite(value) for value in losses):
        raise CPerformanceReportError(f"probe has non-finite loss or gradient state: {log_path}")
    if not any(
        current == expected_train_iters and total == expected_train_iters
        for current, total in progress
    ):
        raise CPerformanceReportError(
            "probe did not log its exact planned final iteration "
            f"{expected_train_iters}: {log_path}"
        )
    if require_moe_routing:
        if not drops or not zero_kept:
            raise CPerformanceReportError(f"MoE probe lacks routing-health metrics: {log_path}")
        if any(not math.isfinite(value) or value != 0.0 for value in drops):
            raise CPerformanceReportError(f"probe dropped routed MoE assignments: {log_path}")
        if any(value != 0 for value in zero_kept):
            raise CPerformanceReportError(f"probe routed tokens to zero experts: {log_path}")
    return {
        "skipped_iterations_max": max(skipped),
        "nan_iterations_max": max(nans),
        "dropped_assignment_rate_max": max(drops) if drops else 0.0,
        "tokens_with_zero_kept_experts_max": max(zero_kept) if zero_kept else 0,
        "finite_loss_lines": len(losses),
        "completed_train_iters": expected_train_iters,
    }


def _read_rows(paths: Iterable[Path]) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows: list[dict[str, str]] = []
    identities: dict[str, str] = {}
    for path in paths:
        resolved = _abs(path)
        if not resolved.is_file():
            raise CPerformanceReportError(f"missing matrix results CSV: {resolved}")
        with resolved.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
        identities[str(resolved)] = _sha256(resolved)
    if not rows:
        raise CPerformanceReportError("no probe rows supplied")
    return rows, identities


def build_report(*, input_csvs: Sequence[Path], global_batch_size: int) -> dict[str, Any]:
    """Return the pinned report for the four selected C execution rows."""

    if global_batch_size not in {32, 64, 128}:
        raise CPerformanceReportError("C direct evidence must be GBS32, GBS64, or GBS128")
    rows, identities = _read_rows(input_csvs)
    selected: list[dict[str, Any]] = []
    for experts, layout in EXPECTED_EXECUTION.items():
        candidates = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("scale_label") == "scaling_c"
            and "te_attn_graph" in str(row.get("variant_label", ""))
            and _int(row, "num_experts") == experts
            and _int(row, "global_batch_size") == global_batch_size
            and _int(row, "micro_batch_size") == int(layout["micro_batch_size"])
            and _int(row, "num_gpus") == int(layout["num_gpus"])
        ]
        if len(candidates) != 1:
            raise CPerformanceReportError(
                f"expected exactly one selected C/E{experts}/GBS{global_batch_size} row, "
                f"found {len(candidates)}"
            )
        row = candidates[0]
        if _int(row, "token_budget_tokens") != PROBE_TOKENS:
            raise CPerformanceReportError("direct C throughput probe has the wrong token budget")
        if _int(row, "warmup_iters") != 64 or _int(row, "throughput_window_size") < 32:
            raise CPerformanceReportError("direct C throughput probe lacks the required long window")
        if _int(row, "steps_counted") < 32:
            raise CPerformanceReportError("direct C throughput probe has too few measured steps")
        max_vram = _float(row, "max_vram_used_pct")
        max_reserved = _float(row, "max_reserved_gb")
        if max_vram > 90.0 or max_reserved > 90.0:
            raise CPerformanceReportError(
                "selected C throughput probe lacks the required H100 memory headroom"
            )
        log_path = Path(str(row.get("log_path", "")))
        if not log_path.is_absolute():
            log_path = _abs(log_path)
        expected_train_iters, remainder = divmod(
            PROBE_TOKENS, global_batch_size * 2048
        )
        if remainder:
            raise CPerformanceReportError("C probe token budget is not iteration-aligned")
        health = _health(
            log_path,
            require_moe_routing=experts > 1,
            expected_train_iters=expected_train_iters,
        )
        median = _float(row, "median_tokens_per_second")
        mean = _float(row, "mean_tokens_per_second")
        mfu = _float(row, "mean_mfu_pct_of_peak")
        megatron_mfu = _float(row, "megatron_mfu_pct_of_peak")
        if median <= 0 or mean <= 0 or mfu <= 0 or megatron_mfu <= 0:
            raise CPerformanceReportError("selected C throughput probe lacks positive performance metrics")
        data_parallel = _int(row, "dp")
        expected_ga = global_batch_size // (int(layout["micro_batch_size"]) * data_parallel)
        if global_batch_size % (int(layout["micro_batch_size"]) * data_parallel):
            raise CPerformanceReportError("C batch configuration has a non-integral accumulation factor")
        selected.append(
            {
                "id": f"c_e{experts}_gbs{global_batch_size}",
                "status": "complete",
                "ladder": "C",
                "experts": experts,
                "GBS": global_batch_size,
                "MBS": int(layout["micro_batch_size"]),
                "GA": expected_ga,
                "num_gpus": int(layout["num_gpus"]),
                "median_tokens_per_second": median,
                "mean_tokens_per_second": mean,
                "mean_mfu_pct_of_peak": mfu,
                "megatron_mfu_pct_of_peak": megatron_mfu,
                "max_reserved_gb": max_reserved,
                "max_vram_used_pct": max_vram,
                "steps_counted": _int(row, "steps_counted"),
                "log_path": str(log_path),
                "health": health,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "c_v1_direct_long_throughput_evidence",
        "global_batch_size": global_batch_size,
        "probe_tokens": PROBE_TOKENS,
        "selected_execution": {
            "cuda_graph_impl": "transformer_engine",
            "cuda_graph_scope": "attn",
            "cuda_graph_warmup_steps": 5,
            "NCCL_GRAPH_REGISTER": "0",
            "E32": "EP2 all-to-all with overlap, delayed wgrad, and distributed optimizer",
        },
        "source_csv_sha256": identities,
        "rows": selected,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def write_report(*, input_csvs: Sequence[Path], global_batch_size: int, output: Path) -> Path:
    payload = build_report(input_csvs=input_csvs, global_batch_size=global_batch_size)
    output = _abs(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, choices=(32, 64, 128), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(
            write_report(
                input_csvs=args.input_csv,
                global_batch_size=args.global_batch_size,
                output=args.output,
            )
        )
    except CPerformanceReportError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
