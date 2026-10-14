#!/usr/bin/env python3
"""Materialize a confirmation panel and its preregistered paired statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy.stats import friedmanchisquare, wilcoxon


PRIMARY_METRIC = "equal_language_atlas_tao_adjusted_loss"
FACTORS = ("x025", "x050", "x100", "x200", "x400")
CONTROL = "fixed2e4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * raw[key]))
        adjusted[key] = running
    return adjusted


def analyze(gate_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = list(gate.get("rows", []))
    analysis = gate.get("analysis", {})
    ladder = str(analysis.get("ladder") or rows[0].get("ladder", "")) if rows else ""
    if gate.get("errors") or len(rows) != 54 or ladder not in {"A", "B"}:
        raise ValueError("gate is not a complete A/B 54-trajectory confirmation panel")
    tidy: list[dict[str, Any]] = []
    by_cell: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        candidate = str(row["lr_candidate_id"])
        identity = (int(row["experts"]), int(row["K"]))
        by_cell.setdefault(identity, {})[candidate] = float(row[PRIMARY_METRIC])
        tidy.append(
            {
                "ladder": ladder,
                "experts": identity[0],
                "K": identity[1],
                "cell": f"{ladder}/E{identity[0]}/K{identity[1]}",
                "candidate": candidate,
                "jointmoe_factor": "" if candidate == CONTROL else row["jointmoe_factor"],
                "peak_lr": row["peak_lr"],
                "primary_loss": row[PRIMARY_METRIC],
                "raw_nll": row["raw_nll"],
                "actual_end_tokens": row["actual_end_tokens"],
                "model_seed": row["model_seed"],
                "evaluation_job_id": row["evaluation_job_id"],
                "training_job_id": row["training_job_id"],
            }
        )
    expected_candidates = set((*FACTORS, CONTROL))
    if len(by_cell) != 9 or any(set(values) != expected_candidates for values in by_cell.values()):
        raise ValueError("panel identities differ from the frozen 9x6 contract")
    ordered_cells = sorted(by_cell)
    vectors = {candidate: [by_cell[cell][candidate] for cell in ordered_cells] for candidate in FACTORS}
    friedman = friedmanchisquare(*(vectors[candidate] for candidate in FACTORS))
    raw_p: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    for candidate in FACTORS:
        if candidate == "x100":
            continue
        differences = [
            vectors["x100"][index] - vectors[candidate][index]
            for index in range(len(ordered_cells))
        ]
        if any(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=0.0) for value in differences):
            raise ValueError("exact Wilcoxon protocol requires nonzero paired differences")
        result = wilcoxon(
            vectors["x100"], vectors[candidate], alternative="two-sided", method="exact"
        )
        raw_p[candidate] = float(result.pvalue)
        comparisons[candidate] = {
            "statistic": float(result.statistic),
            "raw_two_sided_p": float(result.pvalue),
            "x1_lower_loss_cells": sum(value < 0 for value in differences),
            "x1_higher_loss_cells": sum(value > 0 for value in differences),
            "median_x1_minus_candidate_loss": sorted(differences)[len(differences) // 2],
        }
    adjusted = _holm(raw_p)
    for candidate, value in adjusted.items():
        comparisons[candidate]["holm_adjusted_p"] = value
        comparisons[candidate]["holm_significant_at_0p05"] = value < 0.05
    control_vector = [by_cell[cell][CONTROL] for cell in ordered_cells]
    control_result = wilcoxon(
        vectors["x100"], control_vector, alternative="two-sided", method="exact"
    )
    report = {
        "schema": "stage1_lr_confirmation_panel_statistics/v1",
        "status": "complete",
        "ladder": ladder,
        "source_gate": {"path": str(gate_path.resolve()), "sha256": _sha256(gate_path)},
        "design": {
            "blocks": 9,
            "jointmoe_factors": list(FACTORS),
            "fixed_control_excluded_from_friedman": True,
            "seed_count": 1,
            "inference_limit": (
                "Configuration-block evidence at one seed; not a seed-population or "
                "long-horizon significance claim."
            ),
        },
        "preregistered_decision": analysis.get("decision"),
        "friedman_jointmoe_factors": {
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
            "significant_at_0p05": float(friedman.pvalue) < 0.05,
        },
        "exact_two_sided_wilcoxon_x1_vs_jointmoe_factors_holm": comparisons,
        "secondary_x1_vs_fixed2e4_exact_two_sided_wilcoxon": {
            "statistic": float(control_result.statistic),
            "p_value": float(control_result.pvalue),
            "x1_lower_loss_cells": sum(
                left < right for left, right in zip(vectors["x100"], control_vector, strict=True)
            ),
        },
    }
    tidy.sort(key=lambda row: (row["experts"], row["K"], (*FACTORS, CONTROL).index(row["candidate"])))
    return tidy, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tidy, report = analyze(args.gate.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "panel_tidy.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(tidy[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(tidy)
    (args.output_dir / "panel_statistics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
