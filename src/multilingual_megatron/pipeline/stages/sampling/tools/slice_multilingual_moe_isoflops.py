#!/usr/bin/env python3
"""Derive compute-optimal multilingual MoE choices from fitted isoFLOP slices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.tools.fit_multilingual_moe_law import predict, transformed_expert_count
from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (
    E_CORE,
    K_CORE,
    STUDY_ID,
    write_csv,
    write_json,
)


def _read_run_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _predict_loss(params: dict[str, float], *, n_act: float, tokens: float, experts: float, languages: float) -> float:
    import numpy as np

    x = np.array([[n_act, tokens, transformed_expert_count(experts), languages]], dtype=float)
    return float(predict(x, params, ablated=False)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-json", required=True)
    parser.add_argument("--run-matrix", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--flops",
        default="1e18,3e18,1e19,3e19,1e20",
        help="Comma-separated isoFLOP values. Each slice uses D=F/(6*N_act).",
    )
    parser.add_argument("--experts", default=",".join(str(item) for item in E_CORE))
    parser.add_argument("--languages", default=",".join(str(item) for item in K_CORE))
    args = parser.parse_args()

    fit = json.loads(Path(args.fit_json).read_text(encoding="utf-8"))
    params = fit["models"]["full"]["params"]
    run_rows = _read_run_matrix(Path(args.run_matrix))
    n_values = sorted({float(row["N_act"]) for row in run_rows})
    experts = _parse_floats(args.experts)
    languages = _parse_floats(args.languages)
    flops_values = _parse_floats(args.flops)

    all_rows = []
    winners = []
    for flops in flops_values:
        slice_rows = []
        for n_act in n_values:
            tokens = flops / (6.0 * n_act)
            for e in experts:
                for k in languages:
                    loss = _predict_loss(params, n_act=n_act, tokens=tokens, experts=e, languages=k)
                    row = {
                        "F": f"{flops:.8e}",
                        "N_act": f"{n_act:.0f}",
                        "D_tot": f"{tokens:.8e}",
                        "E": f"{e:.0f}",
                        "K": f"{k:.0f}",
                        "predicted_loss": f"{loss:.10g}",
                    }
                    slice_rows.append(row)
                    all_rows.append(row)
        winner = min(slice_rows, key=lambda row: float(row["predicted_loss"]))
        winners.append(winner)

    atlas_rows = []
    for base_k in (1.0, 4.0, 16.0, 50.0):
        for target_k in (4.0, 16.0, 50.0, 100.0):
            if target_k <= base_k:
                continue
            r = target_k / base_k
            atlas_rows.append(
                {
                    "base_K": int(base_k),
                    "target_K": int(target_k),
                    "r": r,
                    "dense_atlas_multiplier_r_pow_0_97": math.pow(r, 0.97),
                }
            )

    write_csv(Path(args.output_csv), all_rows)
    write_json(
        Path(args.output_json),
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "fit_json": args.fit_json,
            "run_matrix": args.run_matrix,
            "winner_by_isoflop": winners,
            "atlas_dense_reference": atlas_rows,
            "notes": [
                "D_tot is derived as F/(6*N_act).",
                "The winner search is restricted to observed ladder N_act values and requested E,K grids.",
            ],
        },
    )
    print(f"Wrote isoFLOP rows to {args.output_csv}")
    print(f"Wrote isoFLOP summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
