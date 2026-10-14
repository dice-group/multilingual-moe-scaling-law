#!/usr/bin/env python3
"""Fit the multilingual MoE scaling law and the no-interaction ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import STUDY_ID, write_json


PARAMS_FULL = (
    "c",
    "log_A",
    "delta",
    "phi_0",
    "phi_1",
    "alpha",
    "gamma",
    "log_B",
    "omega",
    "psi_0",
    "psi_1",
    "beta",
    "zeta",
)
PARAMS_ABLATED = tuple(name for name in PARAMS_FULL if name not in {"phi_1", "psi_1"})


def transformed_expert_count(experts: float, e_start: float = 1.0, e_max: float = 32.0) -> float:
    inverse = 1.0 / (experts - 1.0 + (1.0 / e_start - 1.0 / e_max) ** -1.0) + 1.0 / e_max
    return 1.0 / inverse


def _read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            raw_rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    for row in raw_rows:
        loss_value = row.get("loss", row.get("validation_loss", row.get("val_loss")))
        d_value = row.get("D_checkpoint", row.get("D_tot", row.get("tokens", row.get("num_input_tokens_seen"))))
        if loss_value in (None, "") or d_value in (None, ""):
            continue
        rows.append(
            {
                "loss": float(loss_value),
                "N_act": float(row["N_act"]),
                "D_tot": float(d_value),
                "E": float(row["E"]),
                "K": float(row["K"]),
            }
        )
    if not rows:
        raise SystemExit(f"No fit rows with loss and N_act/D/E/K found in {path}")
    return rows


def _filter_holdout(rows: list[dict[str, float]], holdout: str) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    if holdout == "none":
        return rows, []
    if holdout == "largest_D":
        largest_by_group: dict[tuple[float, float, float], float] = {}
        for row in rows:
            key = (row["N_act"], row["E"], row["K"])
            largest_by_group[key] = max(largest_by_group.get(key, 0.0), row["D_tot"])
        test = [row for row in rows if row["D_tot"] == largest_by_group[(row["N_act"], row["E"], row["K"])]]
    elif holdout == "ladder_D":
        max_n = max(row["N_act"] for row in rows)
        test = [row for row in rows if row["N_act"] == max_n]
    elif holdout == "E32":
        test = [row for row in rows if row["E"] == 32.0]
    elif holdout == "K200":
        test = [row for row in rows if row["K"] == 200.0]
    elif holdout == "E32_K200":
        test = [row for row in rows if row["E"] == 32.0 and row["K"] == 200.0]
    else:
        raise SystemExit(f"Unknown holdout {holdout!r}")
    test_ids = {id(row) for row in test}
    train = [row for row in rows if id(row) not in test_ids]
    if not train:
        raise SystemExit(f"Holdout {holdout!r} leaves no training rows")
    return train, test


def _as_arrays(rows: Iterable[dict[str, float]]):
    import numpy as np

    rows = list(rows)
    x = np.array(
        [[row["N_act"], row["D_tot"], transformed_expert_count(row["E"]), row["K"]] for row in rows],
        dtype=float,
    )
    y = np.array([row["loss"] for row in rows], dtype=float)
    return x, y


def predict(x, params: dict[str, float], *, ablated: bool = False):
    import numpy as np

    n_act = x[:, 0]
    d_tot = x[:, 1]
    e_hat = x[:, 2]
    k = x[:, 3]
    phi_1 = 0.0 if ablated else params["phi_1"]
    psi_1 = 0.0 if ablated else params["psi_1"]
    term_n = (
        math.exp(params["log_A"])
        * np.power(e_hat, params["delta"])
        * np.power(k, params["phi_0"] + phi_1 * np.log(e_hat))
        * np.power(n_act, params["alpha"] + params["gamma"] * np.log(e_hat))
    )
    term_d = (
        math.exp(params["log_B"])
        * np.power(e_hat, params["omega"])
        * np.power(k, params["psi_0"] + psi_1 * np.log(e_hat))
        * np.power(d_tot / k, params["beta"] + params["zeta"] * np.log(e_hat))
    )
    return params["c"] + term_n + term_d


def _initial_params(names: tuple[str, ...], y_mean: float) -> list[float]:
    seeds = {
        "c": max(0.0, y_mean * 0.4),
        "log_A": math.log(max(1e-12, y_mean)),
        "delta": -0.05,
        "phi_0": 0.05,
        "phi_1": -0.01,
        "alpha": -0.08,
        "gamma": 0.0,
        "log_B": math.log(max(1e-12, y_mean)),
        "omega": -0.05,
        "psi_0": 0.05,
        "psi_1": -0.01,
        "beta": -0.08,
        "zeta": 0.0,
    }
    return [seeds[name] for name in names]


def _fit_one(x_train, y_train, names: tuple[str, ...], *, ablated: bool) -> dict[str, float]:
    import numpy as np
    from scipy.optimize import least_squares

    def pack(values):
        params = {name: float(value) for name, value in zip(names, values)}
        if ablated:
            params["phi_1"] = 0.0
            params["psi_1"] = 0.0
        return params

    def residual(values):
        pred = predict(x_train, pack(values), ablated=ablated)
        return pred - y_train

    result = least_squares(
        residual,
        np.array(_initial_params(names, float(np.mean(y_train))), dtype=float),
        max_nfev=20000,
    )
    params = pack(result.x)
    params["_optimizer_success"] = bool(result.success)
    params["_optimizer_message"] = str(result.message)
    params["_train_rmse"] = rmse(y_train, predict(x_train, params, ablated=ablated))
    return params


def rmse(y_true, y_pred) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def r2(y_true, y_pred) -> float:
    import numpy as np

    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.square(y_pred - y_true))) / denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV or JSONL with loss,N_act,D_checkpoint,E,K columns.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--holdout",
        choices=["none", "largest_D", "ladder_D", "E32", "K200", "E32_K200"],
        default="E32_K200",
    )
    args = parser.parse_args()

    rows = _read_rows(Path(args.input))
    train_rows, test_rows = _filter_holdout(rows, args.holdout)
    x_train, y_train = _as_arrays(train_rows)
    full = _fit_one(x_train, y_train, PARAMS_FULL, ablated=False)
    ablated = _fit_one(x_train, y_train, PARAMS_ABLATED, ablated=True)
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "input": args.input,
        "holdout": args.holdout,
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "expert_transform": {"E_start": 1.0, "E_max": 32.0},
        "models": {
            "full": {"params": full},
            "ablated_no_EK_interaction": {"params": ablated},
        },
    }
    for name, params, is_ablated in (("full", full, False), ("ablated_no_EK_interaction", ablated, True)):
        payload["models"][name]["train"] = {
            "rmse": rmse(y_train, predict(x_train, params, ablated=is_ablated)),
            "r2": r2(y_train, predict(x_train, params, ablated=is_ablated)),
        }
        if test_rows:
            x_test, y_test = _as_arrays(test_rows)
            payload["models"][name]["test"] = {
                "rmse": rmse(y_test, predict(x_test, params, ablated=is_ablated)),
                "r2": r2(y_test, predict(x_test, params, ablated=is_ablated)),
            }
    payload["interpretation"] = {
        "main_hypothesis": "phi_1 < 0",
        "full_phi_1": full.get("phi_1"),
        "full_psi_1": full.get("psi_1"),
        "full_beats_ablation_on_holdout": (
            bool(test_rows)
            and payload["models"]["full"]["test"]["rmse"]
            < payload["models"]["ablated_no_EK_interaction"]["test"]["rmse"]
        ),
    }
    write_json(Path(args.output_json), payload)
    print(f"Wrote fit report to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
