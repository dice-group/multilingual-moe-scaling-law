#!/usr/bin/env python3
"""Materialize and compare the Stage-1 H100 and A100 LR screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIMARY_METRIC = "equal_language_atlas_tao_adjusted_loss"
CANDIDATES = ("x025", "x050", "x100", "x200", "x400", "fixed2e4")
A100_POLICY_MAP = {
    "jointmoe_x025": "x025",
    "jointmoe_x050": "x050",
    "jointmoe_x100": "x100",
    "jointmoe_x200": "x200",
    "jointmoe_x400": "x400",
    "historical_fixed_2e-4": "fixed2e4",
}
A100_CSV_PATH = "result_analysis/stage1_lr_seed_v1_a100_screen_results.csv"
A100_RUNTIME_PATH = (
    "configs/scaling_law/stage1_lr_seed_v1/workflow_runtime_a100.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["experts"]),
        int(row["K"]),
        CANDIDATES.index(str(row["candidate"])),
    )


def _load_h100_rows(gate_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("errors") or int(gate.get("observed_trajectories", 0)) != 54:
        raise ValueError("H100 gate is not a complete 54-trajectory result")
    analysis = gate.get("analysis", {})
    rows: list[dict[str, Any]] = []
    campaign_root = gate_path.parent.parent
    for raw in gate["rows"]:
        candidate = str(raw["lr_candidate_id"])
        cell = f'B/E{int(raw["experts"])}/K{int(raw["K"])}'
        cell_analysis = analysis["cells"][cell]
        training_result_path = (
            campaign_root / "jobs" / str(raw["training_job_id"]) / "result.json"
        )
        training_result = json.loads(
            training_result_path.read_text(encoding="utf-8")
        )
        final_health = training_result["training_health"]["segments"][-1]
        rows.append(
            {
                "cluster": "H100",
                "hardware": "H100",
                "seed": int(raw["model_seed"]),
                "cell": cell,
                "ladder": str(raw["ladder"]),
                "experts": int(raw["experts"]),
                "K": int(raw["K"]),
                "candidate": candidate,
                "lr_factor": "" if candidate == "fixed2e4" else raw["lr_multiplier"],
                "base_peak_lr": raw["base_peak_lr"],
                "peak_lr": raw["peak_lr"],
                "primary_validation_loss": raw[PRIMARY_METRIC],
                "raw_nll": raw["raw_nll"],
                "english_common_cohort_loss": raw["english_common_cohort_loss"],
                "rank": int(cell_analysis["rank"][candidate]),
                "loss_regret": cell_analysis["regret"][candidate],
                "actual_end_tokens": int(raw["actual_end_tokens"]),
                "training_health_status": training_result["training_health"]["status"],
                "last_finite_training_loss": final_health["last_finite_lm_loss"],
                "last_finite_gradient_norm": final_health["last_finite_gradient_norm"],
                "learning_rate_min": final_health["learning_rate_min"],
                "learning_rate_max": final_health["learning_rate_max"],
                "max_nan_iterations": final_health["max_nan_iterations"],
                "max_skipped_iterations": final_health["max_skipped_iterations"],
                "evaluation_job_id": str(raw["evaluation_job_id"]),
                "training_job_id": str(raw["training_job_id"]),
                "training_result_path": str(training_result_path.resolve()),
            }
        )
    rows.sort(key=_candidate_sort_key)
    return gate, rows


def _load_a100_rows(repo: Path, ref: str) -> tuple[str, str, list[dict[str, Any]]]:
    content = _git_output(repo, "show", f"{ref}:{A100_CSV_PATH}")
    commit = _git_output(repo, "rev-parse", ref).strip()
    blob = _git_output(repo, "rev-parse", f"{ref}:{A100_CSV_PATH}").strip()
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(io.StringIO(content)):
        candidate = A100_POLICY_MAP[str(raw["policy_id"])]
        experts = int(raw["experts"])
        k_value = int(raw["K"])
        rows.append(
            {
                "cluster": "A100",
                "hardware": "A100-40GB",
                "seed": int(raw["model_seed"]),
                "cell": f"B/E{experts}/K{k_value}",
                "ladder": str(raw["ladder"]),
                "experts": experts,
                "K": k_value,
                "candidate": candidate,
                "lr_factor": raw["lr_factor"],
                "base_peak_lr": raw["joint_moe_base_lr"],
                "peak_lr": float(raw["peak_lr"]),
                "primary_validation_loss": float(raw["primary_validation_loss"]),
                "rank": int(raw["rank"]),
                "loss_regret": float(raw["loss_regret"]),
                "allocated_gpus": int(raw["allocated_gpus"]),
                "task_id": str(raw["task_id"]),
                "source_collection_report_sha256": str(
                    raw["source_collection_report_sha256"]
                ),
            }
        )
    rows.sort(key=_candidate_sort_key)
    return commit, blob, rows


def _validate_alignment(
    h100_rows: Sequence[Mapping[str, Any]],
    a100_rows: Sequence[Mapping[str, Any]],
) -> None:
    h100 = {
        (int(row["experts"]), int(row["K"]), str(row["candidate"]))
        for row in h100_rows
    }
    a100 = {
        (int(row["experts"]), int(row["K"]), str(row["candidate"]))
        for row in a100_rows
    }
    expected = {
        (experts, k_value, candidate)
        for experts in (1, 16, 32)
        for k_value in (1, 12, 200)
        for candidate in CANDIDATES
    }
    if h100 != expected or a100 != expected:
        raise ValueError("H100 and A100 grids do not match the frozen 9x6 panel")
    for left, right in zip(h100_rows, a100_rows, strict=True):
        identity = (left["experts"], left["K"], left["candidate"])
        if identity != (right["experts"], right["K"], right["candidate"]):
            raise ValueError(f"row alignment drifted at {identity}")
        if not math.isclose(
            float(left["peak_lr"]),
            float(right["peak_lr"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"peak LR differs between clusters at {identity}")


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal nontrivial vectors")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x_value - left_mean) * (y_value - right_mean)
        for x_value, y_value in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator


def _candidate_summary(
    rows: Sequence[Mapping[str, Any]], cluster: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate"])].append(row)
    output: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        selected = grouped[candidate]
        output.append(
            {
                "cluster": cluster,
                "candidate": candidate,
                "equal_cell_mean_primary_loss": statistics.fmean(
                    float(row["primary_validation_loss"]) for row in selected
                ),
                "equal_cell_mean_regret": statistics.fmean(
                    float(row["loss_regret"]) for row in selected
                ),
                "median_regret": statistics.median(
                    float(row["loss_regret"]) for row in selected
                ),
                "maximum_regret": max(float(row["loss_regret"]) for row in selected),
                "mean_rank": statistics.fmean(float(row["rank"]) for row in selected),
                "cell_wins": sum(int(row["rank"]) == 1 for row in selected),
                "top_two_cells": sum(int(row["rank"]) <= 2 for row in selected),
            }
        )
    by_regret = sorted(output, key=lambda row: (row["equal_cell_mean_regret"], row["candidate"]))
    aggregate_rank = {str(row["candidate"]): index + 1 for index, row in enumerate(by_regret)}
    for row in output:
        row["aggregate_regret_rank"] = aggregate_rank[str(row["candidate"])]
    return output


def _winner_by_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    winners: dict[str, str] = {}
    for row in rows:
        if int(row["rank"]) == 1:
            cell = str(row["cell"])
            if cell in winners:
                raise ValueError(f"unexpected tied rank-1 winner in {cell}")
            winners[cell] = str(row["candidate"])
    return winners


def _materialize(
    *, repo: Path, gate_path: Path, a100_ref: str, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate, h100_rows = _load_h100_rows(gate_path)
    a100_commit, a100_blob, a100_rows = _load_a100_rows(repo, a100_ref)
    _validate_alignment(h100_rows, a100_rows)
    h100_data_evidence_path = gate_path.parent / "data_reuse_evidence.json"
    h100_data_evidence = json.loads(
        h100_data_evidence_path.read_text(encoding="utf-8")
    )
    a100_runtime_content = _git_output(
        repo, "show", f"{a100_ref}:{A100_RUNTIME_PATH}"
    )
    a100_runtime = json.loads(a100_runtime_content)
    h100_data_seal = h100_data_evidence["reused_corpus"]["source_data_seal"]
    a100_data = a100_runtime["data"]
    a100_source_seal = a100_data["content_copy"][
        "source_data_seal_payload_sha256"
    ]
    if h100_data_seal["seal_payload_sha256"] != a100_source_seal:
        raise ValueError("H100 and A100 corpus seal identities differ")

    h100_path = output_dir / "h100_results.csv"
    a100_path = output_dir / "a100_reference.csv"
    _write_csv(h100_path, h100_rows)
    _write_csv(a100_path, a100_rows)

    comparisons: list[dict[str, Any]] = []
    for h100, a100 in zip(h100_rows, a100_rows, strict=True):
        comparisons.append(
            {
                "cell": h100["cell"],
                "experts": h100["experts"],
                "K": h100["K"],
                "candidate": h100["candidate"],
                "peak_lr": h100["peak_lr"],
                "h100_seed": h100["seed"],
                "h100_primary_validation_loss": h100["primary_validation_loss"],
                "h100_rank": h100["rank"],
                "h100_regret": h100["loss_regret"],
                "a100_seed": a100["seed"],
                "a100_primary_validation_loss": a100["primary_validation_loss"],
                "a100_rank": a100["rank"],
                "a100_regret": a100["loss_regret"],
                "rank_delta_h100_minus_a100": int(h100["rank"])
                - int(a100["rank"]),
            }
        )
    comparison_path = output_dir / "cross_cluster_comparison.csv"
    _write_csv(comparison_path, comparisons)

    boundary_rows = [
        {
            key: row[key]
            for key in (
                "candidate",
                "peak_lr",
                "primary_validation_loss",
                "raw_nll",
                "rank",
                "loss_regret",
                "training_health_status",
                "last_finite_training_loss",
                "last_finite_gradient_norm",
                "learning_rate_min",
                "learning_rate_max",
                "max_nan_iterations",
                "max_skipped_iterations",
                "training_result_path",
            )
        }
        for row in h100_rows
        if row["cell"] == "B/E1/K1"
    ]
    boundary_path = output_dir / "boundary_e1_k1_diagnostics.csv"
    _write_csv(boundary_path, boundary_rows)

    candidate_rows = _candidate_summary(h100_rows, "H100") + _candidate_summary(
        a100_rows, "A100"
    )
    candidate_path = output_dir / "candidate_summary.csv"
    _write_csv(candidate_path, candidate_rows)

    h100_winners = _winner_by_cell(h100_rows)
    a100_winners = _winner_by_cell(a100_rows)
    cell_rows: list[dict[str, Any]] = []
    cell_spearman: dict[str, float] = {}
    for experts in (1, 16, 32):
        for k_value in (1, 12, 200):
            cell = f"B/E{experts}/K{k_value}"
            selected = [row for row in comparisons if row["cell"] == cell]
            correlation = _pearson(
                [float(row["h100_rank"]) for row in selected],
                [float(row["a100_rank"]) for row in selected],
            )
            cell_spearman[cell] = correlation
            x1 = next(row for row in selected if row["candidate"] == "x100")
            cell_rows.append(
                {
                    "cell": cell,
                    "h100_winner": h100_winners[cell],
                    "a100_winner": a100_winners[cell],
                    "exact_winner_agreement": h100_winners[cell] == a100_winners[cell],
                    "h100_x1_rank": x1["h100_rank"],
                    "a100_x1_rank": x1["a100_rank"],
                    "h100_x1_regret": x1["h100_regret"],
                    "a100_x1_regret": x1["a100_regret"],
                    "within_cell_rank_spearman": correlation,
                }
            )
    cell_path = output_dir / "cell_winners.csv"
    _write_csv(cell_path, cell_rows)

    h100_summary = [row for row in candidate_rows if row["cluster"] == "H100"]
    a100_summary = [row for row in candidate_rows if row["cluster"] == "A100"]
    exact_winner_agreements = sum(bool(row["exact_winner_agreement"]) for row in cell_rows)
    pooled_rank_spearman = _pearson(
        [float(row["h100_rank"]) for row in comparisons],
        [float(row["a100_rank"]) for row in comparisons],
    )
    x1_h100 = next(row for row in h100_summary if row["candidate"] == "x100")
    x1_a100 = next(row for row in a100_summary if row["candidate"] == "x100")
    boundary_by_candidate = {
        str(row["candidate"]): row for row in boundary_rows
    }
    boundary_ordered = sorted(
        boundary_rows, key=lambda row: float(row["primary_validation_loss"])
    )
    fixed_pairs = [row for row in comparisons if row["candidate"] in {"x100", "fixed2e4"}]
    fixed_by_cell: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in fixed_pairs:
        fixed_by_cell[str(row["cell"])][str(row["candidate"])] = row
    summary = {
        "schema": "stage1_lr_confirmation_cross_cluster_analysis/v1",
        "status": "complete",
        "scientific_conclusion": {
            "aggregate_leader_h100": min(
                h100_summary,
                key=lambda row: (row["equal_cell_mean_regret"], row["candidate"]),
            )["candidate"],
            "aggregate_leader_a100": min(
                a100_summary,
                key=lambda row: (row["equal_cell_mean_regret"], row["candidate"]),
            )["candidate"],
            "h100_preregistered_decision": gate["analysis"]["decision"],
            "h100_boundary_cells": gate["analysis"]["boundary_cells"],
            "interpretation": (
                "JointMoE x1 is the equal-cell aggregate leader on both clusters, "
                "but the H100 preregistered decision is inconclusive because x4 "
                "wins the E1/K1 outer-boundary cell."
            ),
        },
        "h100_boundary_diagnostic": {
            "cell": "B/E1/K1",
            "winner": boundary_ordered[0]["candidate"],
            "runner_up": boundary_ordered[1]["candidate"],
            "winner_margin_over_runner_up": float(
                boundary_ordered[1]["primary_validation_loss"]
            )
            - float(boundary_ordered[0]["primary_validation_loss"]),
            "x1_regret": boundary_by_candidate["x100"]["loss_regret"],
            "raw_nll_agrees_with_primary_winner": min(
                boundary_rows, key=lambda row: float(row["raw_nll"])
            )["candidate"]
            == boundary_ordered[0]["candidate"],
            "all_training_health_ok": all(
                row["training_health_status"] == "ok"
                and row["max_nan_iterations"] == 0
                and row["max_skipped_iterations"] == 0
                for row in boundary_rows
            ),
        },
        "agreement": {
            "exact_cell_winner_agreements": exact_winner_agreements,
            "cell_count": 9,
            "exact_cell_winner_agreement_fraction": exact_winner_agreements / 9,
            "pooled_within_cell_rank_spearman": pooled_rank_spearman,
            "mean_cell_rank_spearman": statistics.fmean(cell_spearman.values()),
            "minimum_cell_rank_spearman": min(cell_spearman.values()),
            "maximum_cell_rank_spearman": max(cell_spearman.values()),
            "cell_rank_spearman": cell_spearman,
            "h100_winner_counts": dict(sorted(Counter(h100_winners.values()).items())),
            "a100_winner_counts": dict(sorted(Counter(a100_winners.values()).items())),
            "x1_h100_wins": x1_h100["cell_wins"],
            "x1_a100_wins": x1_a100["cell_wins"],
            "x1_h100_top_two_cells": x1_h100["top_two_cells"],
            "x1_a100_top_two_cells": x1_a100["top_two_cells"],
            "x1_beats_fixed_h100_cells": sum(
                values["x100"]["h100_primary_validation_loss"]
                < values["fixed2e4"]["h100_primary_validation_loss"]
                for values in fixed_by_cell.values()
            ),
            "x1_beats_fixed_a100_cells": sum(
                values["x100"]["a100_primary_validation_loss"]
                < values["fixed2e4"]["a100_primary_validation_loss"]
                for values in fixed_by_cell.values()
            ),
        },
        "scope": {
            "matched": [
                "B architecture and E/K cells",
                "2B nominal WSD endpoint",
                "five JointMoE factors plus fixed 2e-4 control",
                "tokenizer, corpus, validation data, and primary metric",
            ],
            "intentionally_different": {
                "h100_seed": 1234,
                "a100_seed": 1729,
                "hardware_topology": "H100 versus A100 8x A100-40GB TP4/EP2/DP2",
            },
            "inference_limit": (
                "Cross-cluster agreement is a robustness check with seed and hardware "
                "jointly changed; it does not isolate a hardware effect or estimate seed variance."
            ),
        },
        "sources": {
            "h100_gate": {
                "path": str(gate_path.resolve()),
                "sha256": _sha256(gate_path),
                "gate_sha256": gate["gate_sha256"],
            },
            "a100_reference": {
                "git_ref": a100_ref,
                "commit": a100_commit,
                "blob": a100_blob,
                "path": A100_CSV_PATH,
                "source_collection_report_sha256": a100_rows[0][
                    "source_collection_report_sha256"
                ],
            },
            "matched_data_identity": {
                "status": "exact_source_seal_payload_match",
                "source_data_seal_payload_sha256": a100_source_seal,
                "h100_data_reuse_evidence_path": str(
                    h100_data_evidence_path.resolve()
                ),
                "h100_data_reuse_evidence_sha256": _sha256(
                    h100_data_evidence_path
                ),
                "a100_runtime_path": A100_RUNTIME_PATH,
                "a100_runtime_blob": _git_output(
                    repo, "rev-parse", f"{a100_ref}:{A100_RUNTIME_PATH}"
                ).strip(),
            },
        },
    }
    summary_path = output_dir / "comparison_summary.json"
    _write_json(summary_path, summary)

    candidate_lookup = {
        (str(row["cluster"]), str(row["candidate"])): row for row in candidate_rows
    }
    markdown = [
        "# Stage-1 JointMoE LR confirmation: H100 and A100 comparison",
        "",
        "## Result",
        "",
        "JointMoE x1 is the equal-cell aggregate leader on both clusters. The H100",
        "preregistered decision is nevertheless **inconclusive**, because x4 wins the",
        "outer-boundary B/E1/K1 cell. The result therefore supports x1 as the strongest",
        "global candidate, but does not satisfy the frozen rule needed to claim full",
        "confirmation without a targeted boundary follow-up.",
        "",
        "## Candidate aggregates",
        "",
        "| Candidate | H100 mean regret | H100 mean rank | H100 wins | A100 mean regret | A100 mean rank | A100 wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        h100 = candidate_lookup[("H100", candidate)]
        a100 = candidate_lookup[("A100", candidate)]
        markdown.append(
            f"| {candidate} | {h100['equal_cell_mean_regret']:.9f} | "
            f"{h100['mean_rank']:.3f} | {h100['cell_wins']} | "
            f"{a100['equal_cell_mean_regret']:.9f} | {a100['mean_rank']:.3f} | "
            f"{a100['cell_wins']} |"
        )
    markdown.extend(
        [
            "",
            "## Cell winners",
            "",
            "| Cell | H100 seed 1234 | A100 seed 1729 | Same winner | H100 x1 regret |",
            "|---|---|---|:---:|---:|",
        ]
    )
    for row in cell_rows:
        markdown.append(
            f"| {row['cell']} | {row['h100_winner']} | {row['a100_winner']} | "
            f"{'yes' if row['exact_winner_agreement'] else 'no'} | "
            f"{float(row['h100_x1_regret']):.9f} |"
        )
    markdown.extend(
        [
            "",
            "## Agreement and limits",
            "",
            f"- Exact cell winner agreement: {exact_winner_agreements}/9.",
            f"- Pooled within-cell rank Spearman: {pooled_rank_spearman:.3f}.",
            f"- x1 is top-two in {x1_h100['top_two_cells']}/9 H100 cells and "
            f"{x1_a100['top_two_cells']}/9 A100 cells.",
            "- x1 beats the historical fixed 2e-4 control in all nine cells on both clusters.",
            "- Both campaigns resolve to the same sealed 1,012-file corpus payload.",
            "- Seed and hardware/topology changed together, so this comparison cannot isolate a hardware effect.",
            "- The primary adjusted loss may be negative; lower remains better by construction.",
            "",
            "## Files",
            "",
            "- `STATISTICAL_ANALYSIS.md`: Friedman and exact paired Wilcoxon results,",
            "  including the two-sided and one-sided protocols and why they differ.",
            "- `h100_results.csv`: tidy 54-row H100 table.",
            "- `boundary_e1_k1_diagnostics.csv`: health and metric audit for the boundary cell.",
            "- `a100_reference.csv`: portable 54-row A100 reference table.",
            "- `cross_cluster_comparison.csv`: aligned 54-row comparison.",
            "- `candidate_summary.csv`: candidate-level aggregate diagnostics.",
            "- `cell_winners.csv`: nine cell-level winner comparisons.",
            "- `comparison_summary.json`: machine-readable conclusion and provenance.",
            "- `analysis_manifest.json`: SHA-256 inventory of this analysis package.",
            "",
        ]
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(markdown), encoding="utf-8")

    artifact_names = (
        "README.md",
        "STATISTICAL_ANALYSIS.md",
        "boundary_e1_k1_diagnostics.csv",
        "candidate_summary.csv",
        "cell_winners.csv",
        "comparison_summary.json",
        "cross_cluster_comparison.csv",
        "h100_results.csv",
        "a100_reference.csv",
    )
    manifest = {
        "schema": "stage1_lr_confirmation_analysis_manifest/v1",
        "status": "complete",
        "generator": Path(__file__).resolve().relative_to(repo).as_posix(),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "artifacts": {
            name: _sha256(output_dir / name) for name in artifact_names
        },
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h100-gate",
        type=Path,
        default=Path(
            "outputs/pipeline_runs/stage1_lr_confirmation_h100_v1/control/"
            "lr_confirmation_gate.json"
        ),
    )
    parser.add_argument("--a100-ref", default="origin/stage1/lr-calibration-v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result_analysis/stage1_lr_confirmation_h100_v1"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    summary = _materialize(
        repo=repo,
        gate_path=(repo / args.h100_gate).resolve(),
        a100_ref=args.a100_ref,
        output_dir=(repo / args.output_dir).resolve(),
    )
    print(json.dumps(summary["scientific_conclusion"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
