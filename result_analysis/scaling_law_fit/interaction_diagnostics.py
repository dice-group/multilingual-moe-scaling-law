"""Exploratory interaction diagnostics for the clean Stage 1 JointMoE-x1 grid.

This module never refits or changes the registered M0/M1/M2 candidates.  It
builds transparent empirical contrasts that can motivate a later, separately
validated model-development protocol:

* exact expert contrasts at fixed active size, K, D_t, and language;
* in-support language-count contrasts at matched realized D_t;
* local data-exposure slopes within each trajectory and language;
* exact active-size contrasts at fixed E, K, D_t, and language; and
* signed out-of-fold M1/M2 residual maps on the three frozen primary folds.

All results are exploratory.  Trajectories are the independent design units;
languages and checkpoint readouts remain nested measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.campaign import (
    scientific_result,
)
from multilingual_megatron.experiments.scaling.stage1_fit.data import (
    canonical_json_sha256,
    load_fit_rows_snapshot,
    sha256_file,
)
from multilingual_megatron.experiments.scaling.stage1_fit.execution import (
    bind_membership,
    read_json,
)
from multilingual_megatron.experiments.scaling.stage1_fit.fit import (
    SharedParameters,
    data_column,
    effective_exponents,
    predict,
)


SCHEMA_VERSION = "stage1_jointmoe_x1_interaction_diagnostics/v1"
REPORT_SCHEMA = "stage1_fit_candidate_report/v5"
EXPERT_LEVELS = (1, 4, 16, 32)
LANGUAGE_COUNT_LEVELS = (1, 12, 50, 200)
LADDERS = ("A", "B", "C", "D")
EXPERT_PAIRS = ((1, 4), (4, 16), (16, 32), (1, 32))
LANGUAGE_COUNT_PAIRS = ((1, 12), (12, 50), (50, 200), (12, 200))
LADDER_PAIRS = (("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"))
PRIMARY_FOLDS = ("largest_N_act", "E32", "K200_common_languages")


class InteractionDiagnosticError(RuntimeError):
    """Raised when an input or contrast violates the analysis contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise InteractionDiagnosticError(f"{context} columns missing: {missing}")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "trajectory_id",
        "readout_id",
        "language",
        "N_act",
        "E",
        "K",
        "D_total",
        "D_t",
        "adjusted_loss",
        "raw_nll",
        "ladder",
    )
    _require_columns(frame, required, "fit rows")
    result = frame.loc[:, required].copy()
    numeric = result[["N_act", "E", "K", "D_total", "D_t", "adjusted_loss"]]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise InteractionDiagnosticError("fit rows contain non-finite values")
    if (result[["N_act", "E", "K", "D_total", "D_t"]] <= 0).any().any():
        raise InteractionDiagnosticError("fit predictors must be strictly positive")
    if tuple(sorted(result["E"].unique())) != EXPERT_LEVELS:
        raise InteractionDiagnosticError("expert grid changed")
    if tuple(sorted(result["K"].unique())) != LANGUAGE_COUNT_LEVELS:
        raise InteractionDiagnosticError("language-count grid changed")
    if tuple(sorted(result["ladder"].unique())) != LADDERS:
        raise InteractionDiagnosticError("ladder grid changed")
    if result["trajectory_id"].nunique() != 64 or result["readout_id"].nunique() != 400:
        raise InteractionDiagnosticError("canonical trajectory/readout counts changed")
    active_sizes = result[["ladder", "E", "N_act"]].drop_duplicates()
    if active_sizes.duplicated(["ladder", "E"]).any():
        raise InteractionDiagnosticError("one ladder/expert cell has multiple active sizes")
    if not active_sizes.groupby("ladder")["N_act"].nunique().eq(1).all():
        raise InteractionDiagnosticError("N_act is not fixed across E within ladder")
    return result.sort_values(
        ["ladder", "E", "K", "D_total", "language"]
    ).reset_index(drop=True)


def expert_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exact expert gains; positive values mean the higher E is better."""

    data = _validate_frame(frame)
    keys = ["ladder", "N_act", "K", "D_total", "D_t", "language"]
    outputs: list[pd.DataFrame] = []
    for low, high in EXPERT_PAIRS:
        lower = data.loc[data["E"].eq(low), keys + ["adjusted_loss"]].rename(
            columns={"adjusted_loss": "loss_low"}
        )
        higher = data.loc[data["E"].eq(high), keys + ["adjusted_loss"]].rename(
            columns={"adjusted_loss": "loss_high"}
        )
        matched = lower.merge(higher, on=keys, how="inner", validate="one_to_one")
        if matched.empty:
            raise InteractionDiagnosticError(f"expert contrast E{low}->E{high} is empty")
        matched.insert(2, "E_low", low)
        matched.insert(3, "E_high", high)
        matched["expert_gain"] = matched["loss_low"] - matched["loss_high"]
        matched["expert_gain_per_log_e"] = matched["expert_gain"] / math.log(
            high / low
        )
        outputs.append(matched)
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(
        ["E_low", "E_high", "ladder", "K", "D_total", "language"]
    ).reset_index(drop=True)


def _log_interpolation(
    curve: pd.DataFrame, target: float
) -> tuple[float, str, float, float, float] | None:
    ordered = curve.sort_values("D_t").drop_duplicates("D_t")
    exposure = ordered["D_t"].to_numpy(dtype=np.float64)
    loss = ordered["adjusted_loss"].to_numpy(dtype=np.float64)
    if target < exposure[0] or target > exposure[-1]:
        return None
    exact = np.flatnonzero(exposure == target)
    if exact.size:
        index = int(exact[0])
        return float(loss[index]), "exact", target, target, 0.0
    upper_index = int(np.searchsorted(exposure, target, side="right"))
    if upper_index == 0 or upper_index == len(exposure):
        return None
    lower_index = upper_index - 1
    lower = float(exposure[lower_index])
    upper = float(exposure[upper_index])
    weight = (math.log(target) - math.log(lower)) / (math.log(upper) - math.log(lower))
    estimate = float(loss[lower_index] + weight * (loss[upper_index] - loss[lower_index]))
    return estimate, "interpolated", lower, upper, float(weight)


def language_count_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Match higher-K rows to lower-K curves at identical realized D_t."""

    data = _validate_frame(frame)
    rows: list[dict[str, Any]] = []
    for low, high in LANGUAGE_COUNT_PAIRS:
        references = {
            key: group
            for key, group in data.loc[data["K"].eq(low)].groupby(
                ["ladder", "E", "language"], observed=True, sort=True
            )
        }
        targets = data.loc[data["K"].eq(high)]
        for target in targets.itertuples(index=False):
            curve = references.get((target.ladder, target.E, target.language))
            if curve is None:
                continue
            interpolation = _log_interpolation(curve, float(target.D_t))
            if interpolation is None:
                continue
            estimate, kind, lower, upper, weight = interpolation
            rows.append(
                {
                    "ladder": target.ladder,
                    "N_act": float(target.N_act),
                    "E": int(target.E),
                    "K_low": low,
                    "K_high": high,
                    "D_total": float(target.D_total),
                    "D_t": float(target.D_t),
                    "language": target.language,
                    "loss_low_matched": estimate,
                    "loss_high": float(target.adjusted_loss),
                    "language_count_penalty": float(target.adjusted_loss - estimate),
                    "penalty_per_log_k": float(
                        (target.adjusted_loss - estimate) / math.log(high / low)
                    ),
                    "match_kind": kind,
                    "reference_lower_D_t": lower,
                    "reference_upper_D_t": upper,
                    "interpolation_weight": weight,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise InteractionDiagnosticError("all language-count contrasts are empty")
    if (
        (result["D_t"] < result["reference_lower_D_t"])
        | (result["D_t"] > result["reference_upper_D_t"])
    ).any():
        raise InteractionDiagnosticError("language-count matching extrapolated")
    return result.sort_values(
        ["K_low", "K_high", "ladder", "E", "D_total", "language"]
    ).reset_index(drop=True)


def local_data_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return adjacent-checkpoint loss improvement per unit log D_t."""

    data = _validate_frame(frame)
    rows: list[dict[str, Any]] = []
    groups = data.groupby(
        ["trajectory_id", "ladder", "N_act", "E", "K", "language"],
        observed=True,
        sort=True,
    )
    for (trajectory, ladder, n_act, experts, languages, language), group in groups:
        ordered = group.sort_values("D_t").drop_duplicates("D_t")
        exposure = ordered["D_t"].to_numpy(dtype=np.float64)
        loss = ordered["adjusted_loss"].to_numpy(dtype=np.float64)
        if len(ordered) < 2 or np.any(np.diff(exposure) <= 0):
            raise InteractionDiagnosticError(f"invalid D_t curve: {trajectory}/{language}")
        for index in range(len(ordered) - 1):
            denominator = math.log(exposure[index + 1] / exposure[index])
            rows.append(
                {
                    "trajectory_id": trajectory,
                    "ladder": ladder,
                    "N_act": float(n_act),
                    "E": int(experts),
                    "K": int(languages),
                    "language": language,
                    "interval_index": index,
                    "D_t_low": float(exposure[index]),
                    "D_t_high": float(exposure[index + 1]),
                    "D_t_geometric_midpoint": float(
                        math.sqrt(exposure[index] * exposure[index + 1])
                    ),
                    "data_gain_per_log_D_t": float(
                        -(loss[index + 1] - loss[index]) / denominator
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["trajectory_id", "language", "interval_index"]
    ).reset_index(drop=True)


def active_size_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exact ladder gains normalized by the active-size log ratio."""

    data = _validate_frame(frame)
    sizes = (
        data[["ladder", "N_act"]]
        .drop_duplicates()
        .set_index("ladder")["N_act"]
        .to_dict()
    )
    keys = ["E", "K", "D_total", "D_t", "language"]
    outputs: list[pd.DataFrame] = []
    for low, high in LADDER_PAIRS:
        lower = data.loc[data["ladder"].eq(low), keys + ["adjusted_loss"]].rename(
            columns={"adjusted_loss": "loss_low_N"}
        )
        higher = data.loc[data["ladder"].eq(high), keys + ["adjusted_loss"]].rename(
            columns={"adjusted_loss": "loss_high_N"}
        )
        matched = lower.merge(higher, on=keys, how="inner", validate="one_to_one")
        if matched.empty:
            raise InteractionDiagnosticError(f"active-size contrast {low}->{high} is empty")
        matched.insert(0, "ladder_low", low)
        matched.insert(1, "ladder_high", high)
        matched.insert(2, "N_act_low", float(sizes[low]))
        matched.insert(3, "N_act_high", float(sizes[high]))
        matched["active_size_gain"] = matched["loss_low_N"] - matched["loss_high_N"]
        matched["active_size_gain_per_log_N"] = matched["active_size_gain"] / math.log(
            sizes[high] / sizes[low]
        )
        outputs.append(matched)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["ladder_low", "ladder_high", "E", "K", "D_total", "language"]
    ).reset_index(drop=True)


def _cell_then_horizon_mean(
    frame: pd.DataFrame,
    *,
    cell_keys: Sequence[str],
    value: str,
) -> pd.DataFrame:
    readout_keys = [*cell_keys, "D_total"]
    readouts = (
        frame.groupby(readout_keys, observed=True, as_index=False)
        .agg(
            value=(value, "mean"),
            geometric_D_t=("D_t", lambda values: float(np.exp(np.log(values).mean()))),
            languages=("language", "nunique"),
        )
        .sort_values(readout_keys)
    )
    rows: list[dict[str, Any]] = []
    for key, group in readouts.groupby(list(cell_keys), observed=True, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        ordered = group.sort_values("D_total")
        row = dict(zip(cell_keys, key_values))
        row.update(
            {
                "mean": float(ordered["value"].mean()),
                "first": float(ordered.iloc[0]["value"]),
                "terminal": float(ordered.iloc[-1]["value"]),
                "horizons": int(len(ordered)),
                "minimum_languages_per_horizon": int(ordered["languages"].min()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(cell_keys)).reset_index(drop=True)


def summarize_contrasts(
    experts: pd.DataFrame,
    language_counts: pd.DataFrame,
    data_slopes: pd.DataFrame,
    active_sizes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    expert_summary = _cell_then_horizon_mean(
        experts,
        cell_keys=("ladder", "N_act", "K", "E_low", "E_high"),
        value="expert_gain",
    )
    language_summary = _cell_then_horizon_mean(
        language_counts,
        cell_keys=("ladder", "N_act", "E", "K_low", "K_high"),
        value="language_count_penalty",
    )
    interval = (
        data_slopes.groupby(
            ["trajectory_id", "ladder", "N_act", "E", "K", "interval_index"],
            observed=True,
            as_index=False,
        )
        .agg(
            data_gain_per_log_D_t=("data_gain_per_log_D_t", "mean"),
            languages=("language", "nunique"),
        )
        .sort_values(["trajectory_id", "interval_index"])
    )
    data_rows: list[dict[str, Any]] = []
    for key, group in interval.groupby(
        ["trajectory_id", "ladder", "N_act", "E", "K"],
        observed=True,
        sort=True,
    ):
        ordered = group.sort_values("interval_index")
        data_rows.append(
            {
                "trajectory_id": key[0],
                "ladder": key[1],
                "N_act": key[2],
                "E": key[3],
                "K": key[4],
                "mean": float(ordered["data_gain_per_log_D_t"].mean()),
                "early": float(ordered.iloc[0]["data_gain_per_log_D_t"]),
                "late": float(ordered.iloc[-1]["data_gain_per_log_D_t"]),
                "late_over_early": float(
                    ordered.iloc[-1]["data_gain_per_log_D_t"]
                    / ordered.iloc[0]["data_gain_per_log_D_t"]
                ),
                "intervals": int(len(ordered)),
                "languages": int(ordered["languages"].min()),
            }
        )
    data_summary = pd.DataFrame(data_rows).sort_values(
        ["ladder", "E", "K"]
    ).reset_index(drop=True)
    active_summary = _cell_then_horizon_mean(
        active_sizes.rename(columns={"active_size_gain_per_log_N": "value"}),
        cell_keys=("ladder_low", "ladder_high", "E", "K"),
        value="value",
    )
    return {
        "expert_summary": expert_summary,
        "language_count_summary": language_summary,
        "data_slope_summary": data_summary,
        "active_size_summary": active_summary,
    }


def _validated_candidate_report(path: Path) -> Mapping[str, Any]:
    report = read_json(path)
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256", None)
    if report.get("schema_version") != REPORT_SCHEMA:
        raise InteractionDiagnosticError("candidate report schema changed")
    if digest != canonical_json_sha256(unsigned):
        raise InteractionDiagnosticError("candidate report hash changed")
    return report


def _prediction_rows(
    frame: pd.DataFrame,
    membership: Mapping[str, Any],
    result: Mapping[str, Any],
    model: str,
) -> pd.DataFrame:
    rows = frame[frame["trajectory_id"].isin(membership["test_trajectory_ids"])].copy()
    if membership["test_readout_ids"]:
        rows = rows[rows["readout_id"].isin(membership["test_readout_ids"])]
    if membership["score_languages"]:
        rows = rows[rows["language"].isin(membership["score_languages"])]
    coefficients = {
        item["language"]: item for item in result["selected"]["language_coefficients"]
    }
    if not set(rows["language"]).issubset(coefficients):
        raise InteractionDiagnosticError("held-out language lacks nuisance coefficients")
    nuisance = [coefficients[str(language)] for language in rows["language"]]
    selected = result["selected"]
    predictions = predict(
        model,
        SharedParameters(**selected["shared_parameters"]),
        rows["N_act"],
        rows[data_column(model)],
        rows["E"],
        rows["K"],
        [item["L_inf"] for item in nuisance],
        [item["A"] for item in nuisance],
        [item["B"] for item in nuisance],
    )
    keep = [
        "trajectory_id",
        "readout_id",
        "language",
        "ladder",
        "N_act",
        "E",
        "K",
        "D_total",
        "D_t",
        "adjusted_loss",
    ]
    output = rows[keep].copy()
    output[f"{model}_prediction"] = predictions
    output[f"{model}_residual"] = predictions - rows["adjusted_loss"].to_numpy()
    output[f"{model}_squared_error"] = output[f"{model}_residual"] ** 2
    return output


def fold_residuals(
    frame: pd.DataFrame,
    *,
    fit_root: Path,
    candidate_report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return signed M1/M2 out-of-fold residual rows and cell summaries."""

    report = _validated_candidate_report(candidate_report_path)
    source_paths: dict[tuple[str, str], Path] = {}
    for item in report["source_results"]:
        if item.get("role") == "primary" and item.get("model") in {"M1", "M2"}:
            source_paths[(str(item["fold_id"]), str(item["model"]))] = Path(item["path"])
    expected = {(fold, model) for fold in PRIMARY_FOLDS for model in ("M1", "M2")}
    if set(source_paths) != expected:
        raise InteractionDiagnosticError("candidate report lacks M1/M2 primary sources")
    holdouts = fit_root / "input" / "holdouts.json"
    outputs: list[pd.DataFrame] = []
    for fold in PRIMARY_FOLDS:
        membership = bind_membership(holdouts, kind="primary", membership_id=fold)
        model_rows = []
        for model in ("M1", "M2"):
            result = scientific_result(source_paths[(fold, model)])
            model_rows.append(_prediction_rows(frame, membership, result, model))
        keys = [
            "trajectory_id",
            "readout_id",
            "language",
            "ladder",
            "N_act",
            "E",
            "K",
            "D_total",
            "D_t",
            "adjusted_loss",
        ]
        merged = model_rows[0].merge(model_rows[1], on=keys, validate="one_to_one")
        merged.insert(0, "fold_id", fold)
        merged["M2_minus_M1_squared_error"] = (
            merged["M2_squared_error"] - merged["M1_squared_error"]
        )
        for model in ("M1", "M2"):
            expected = report["primary_metrics"][model][fold]
            observed = merged.groupby("trajectory_id")[
                f"{model}_squared_error"
            ].mean()
            expected_mse = pd.Series(expected["trajectory_mse"], dtype=np.float64)
            if (
                len(merged) != expected["rows"]
                or observed.size != expected["trajectories"]
                or set(observed.index) != set(expected_mse.index)
                or not np.allclose(
                    observed.sort_index().to_numpy(),
                    expected_mse.sort_index().to_numpy(),
                    rtol=1e-12,
                    atol=1e-14,
                )
            ):
                raise InteractionDiagnosticError(
                    f"recomputed {fold}/{model} metrics differ from the sealed report"
                )
        outputs.append(merged)
    residuals = pd.concat(outputs, ignore_index=True).sort_values(
        ["fold_id", "trajectory_id", "D_total", "language"]
    ).reset_index(drop=True)
    cells = (
        residuals.groupby(
            ["fold_id", "trajectory_id", "ladder", "N_act", "E", "K", "D_total"],
            observed=True,
            as_index=False,
        )
        .agg(
            M1_bias=("M1_residual", "mean"),
            M2_bias=("M2_residual", "mean"),
            M1_mse=("M1_squared_error", "mean"),
            M2_mse=("M2_squared_error", "mean"),
            M2_minus_M1_mse=("M2_minus_M1_squared_error", "mean"),
            languages=("language", "nunique"),
        )
        .sort_values(["fold_id", "trajectory_id", "D_total"])
    )
    cells["horizon_index"] = (
        cells.groupby(["fold_id", "trajectory_id"], observed=True).cumcount() + 1
    )
    return residuals, cells.reset_index(drop=True)


def parameter_endpoints(candidate_report_path: Path) -> pd.DataFrame:
    report = _validated_candidate_report(candidate_report_path)
    rows: list[dict[str, Any]] = []
    for item in report["source_results"]:
        model = str(item.get("model"))
        if item.get("role") != "primary" or model not in {"M1", "M2"}:
            continue
        result = scientific_result(Path(item["path"]))
        shared = SharedParameters(**result["selected"]["shared_parameters"])
        for experts in EXPERT_LEVELS:
            alpha, beta, phi, psi = effective_exponents(shared, experts, model=model)
            rows.append(
                {
                    "fold_id": item["fold_id"],
                    "model": model,
                    "E": experts,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "phi": float(phi),
                    "psi": float(psi),
                    "delta": shared.delta,
                    "omega": shared.omega,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "fold_id", "E"]).reset_index(
        drop=True
    )


def joint_summaries(
    language_counts: pd.DataFrame,
    experts: pd.DataFrame,
    residual_cells: pd.DataFrame,
    summaries: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    k_rows = language_counts.loc[
        language_counts["K_low"].eq(12) & language_counts["K_high"].eq(200)
    ]
    k_readouts = (
        k_rows.groupby(["ladder", "N_act", "E", "D_total"], as_index=False)
        .agg(
            penalty=("language_count_penalty", "mean"),
            geometric_D_t=("D_t", lambda values: float(np.exp(np.log(values).mean()))),
            languages=("language", "nunique"),
        )
        .sort_values(["ladder", "E", "D_total"])
    )
    attenuation = k_readouts.pivot_table(
        index=["ladder", "N_act", "D_total"], columns="E", values="penalty"
    ).reset_index()
    if 1 not in attenuation or 32 not in attenuation:
        raise InteractionDiagnosticError("E1/E32 K-penalty attenuation is unavailable")
    attenuation["E1_minus_E32_penalty"] = attenuation[1] - attenuation[32]
    attenuation = attenuation.rename(columns={1: "penalty_E1", 32: "penalty_E32"})

    trend_rows: list[dict[str, Any]] = []
    for (ladder, n_act, expert), group in k_readouts.groupby(
        ["ladder", "N_act", "E"], observed=True, sort=True
    ):
        ordered = group.sort_values("geometric_D_t")
        slope = float(
            np.polyfit(np.log(ordered["geometric_D_t"]), ordered["penalty"], 1)[0]
        )
        trend_rows.append(
            {
                "ladder": ladder,
                "N_act": n_act,
                "E": expert,
                "horizons": len(ordered),
                "first_penalty": float(ordered.iloc[0]["penalty"]),
                "terminal_penalty": float(ordered.iloc[-1]["penalty"]),
                "penalty_slope_per_log_D_t": slope,
            }
        )
    k_trends = pd.DataFrame(trend_rows)

    e_rows = experts.loc[experts["E_low"].eq(1) & experts["E_high"].eq(32)]
    e_readouts = (
        e_rows.groupby(["ladder", "N_act", "K", "D_total"], as_index=False)
        .agg(
            expert_gain=("expert_gain", "mean"),
            geometric_D_t=("D_t", lambda values: float(np.exp(np.log(values).mean()))),
            languages=("language", "nunique"),
        )
        .sort_values(["ladder", "K", "D_total"])
    )
    expert_trends: list[dict[str, Any]] = []
    for (ladder, n_act, languages), group in e_readouts.groupby(
        ["ladder", "N_act", "K"], observed=True, sort=True
    ):
        ordered = group.sort_values("geometric_D_t")
        expert_trends.append(
            {
                "ladder": ladder,
                "N_act": n_act,
                "K": languages,
                "horizons": len(ordered),
                "first_gain": float(ordered.iloc[0]["expert_gain"]),
                "terminal_gain": float(ordered.iloc[-1]["expert_gain"]),
                "gain_slope_per_log_D_t": float(
                    np.polyfit(
                        np.log(ordered["geometric_D_t"]), ordered["expert_gain"], 1
                    )[0]
                ),
            }
        )
    residual_summary = (
        residual_cells.groupby(["fold_id", "ladder", "E", "K"], as_index=False)
        .agg(
            M1_bias=("M1_bias", "mean"),
            M2_bias=("M2_bias", "mean"),
            M1_mse=("M1_mse", "mean"),
            M2_mse=("M2_mse", "mean"),
            M2_minus_M1_mse=("M2_minus_M1_mse", "mean"),
            horizons=("D_total", "nunique"),
        )
        .sort_values(["fold_id", "ladder", "E", "K"])
    )

    data_by_k = summaries["data_slope_summary"].pivot(
        index=["ladder", "E"], columns="K", values="mean"
    )
    expert_by_pair = summaries["expert_summary"].pivot(
        index=["ladder", "K"], columns=["E_low", "E_high"], values="mean"
    )
    active_adjacent = summaries["active_size_summary"].loc[
        ~(
            summaries["active_size_summary"]["ladder_low"].eq("A")
            & summaries["active_size_summary"]["ladder_high"].eq("D")
        )
    ]
    active_by_k = active_adjacent.pivot(
        index=["ladder_low", "ladder_high", "E"], columns="K", values="mean"
    )

    def concordance_row(
        pattern: str, evidence_unit: str, support: pd.Series
    ) -> dict[str, Any]:
        observed = support.dropna().astype(bool)
        return {
            "pattern": pattern,
            "evidence_unit": evidence_unit,
            "supporting_cells": int(observed.sum()),
            "total_cells": int(len(observed)),
            "support_fraction": float(observed.mean()),
        }

    concordance = pd.DataFrame(
        [
            concordance_row(
                "K200-minus-K12 penalty grows with D_t",
                "ladder-by-E trend",
                k_trends["penalty_slope_per_log_D_t"].gt(0.0),
            ),
            concordance_row(
                "E32 attenuates the K200-minus-K12 penalty versus E1",
                "matched ladder-by-horizon readout",
                attenuation["E1_minus_E32_penalty"].gt(0.0),
            ),
            concordance_row(
                "local D_t gain is larger at K200 than K12",
                "ladder-by-E trajectory cell",
                data_by_k[200].gt(data_by_k[12]),
            ),
            concordance_row(
                "E16-to-E32 gain is smaller than E4-to-E16",
                "ladder-by-K cell",
                expert_by_pair[(16, 32)].lt(expert_by_pair[(4, 16)]),
            ),
            concordance_row(
                "E16-to-E32 gain is smaller than E1-to-E4",
                "ladder-by-K cell",
                expert_by_pair[(16, 32)].lt(expert_by_pair[(1, 4)]),
            ),
            concordance_row(
                "adjacent-ladder N_act gain is larger at K200 than K12",
                "adjacent-ladder-by-E cell",
                active_by_k[200].gt(active_by_k[12]),
            ),
        ]
    )
    return {
        "k_penalty_readouts": k_readouts,
        "k_penalty_attenuation": attenuation,
        "k_penalty_D_t_trends": k_trends,
        "expert_gain_readouts": e_readouts,
        "expert_gain_D_t_trends": pd.DataFrame(expert_trends),
        "fold_residual_summary": residual_summary,
        "pattern_concordance": concordance,
    }


def _heatmap(
    ax: Any,
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    title: str,
    xlabel: str,
    ylabel: str,
    annotation_format: str = ".3f",
) -> None:
    matrix = frame.pivot(index=index, columns=columns, values=values)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), [str(value) for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [str(value) for value in matrix.index])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            ax.text(
                column,
                row,
                format(matrix.iloc[row, column], annotation_format),
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_figures(
    summaries: Mapping[str, pd.DataFrame],
    joint: Mapping[str, pd.DataFrame],
    residual_cells: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    expert = summaries["expert_summary"].query("E_low == 1 and E_high == 32")
    k_penalty = summaries["language_count_summary"].query(
        "K_low == 12 and K_high == 200"
    )
    data_slope = (
        summaries["data_slope_summary"]
        .groupby(["E", "K"], as_index=False)["mean"]
        .mean()
    )
    active = (
        summaries["active_size_summary"]
        .query("ladder_low != 'A' or ladder_high != 'D'")
        .groupby(["E", "K"], as_index=False)["mean"]
        .mean()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    _heatmap(
        axes[0, 0],
        expert,
        index="ladder",
        columns="K",
        values="mean",
        title="E1 to E32 gain at fixed N_act and D_t",
        xlabel="K",
        ylabel="ladder",
    )
    _heatmap(
        axes[0, 1],
        k_penalty,
        index="ladder",
        columns="E",
        values="mean",
        title="K12 to K200 penalty at matched D_t",
        xlabel="E",
        ylabel="ladder",
    )
    _heatmap(
        axes[1, 0],
        data_slope,
        index="E",
        columns="K",
        values="mean",
        title="Local data gain per log D_t",
        xlabel="K",
        ylabel="E",
    )
    _heatmap(
        axes[1, 1],
        active,
        index="E",
        columns="K",
        values="mean",
        title="Adjacent-ladder gain per log N_act",
        xlabel="K",
        ylabel="E",
    )
    fig.suptitle("Separate Stage 1 interaction diagnostics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    separate_path = figures / "separate_effects.png"
    fig.savefig(separate_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    k_readouts = joint["k_penalty_readouts"]
    for ax, ladder in zip(axes.flat, LADDERS):
        for expert, group in k_readouts.loc[k_readouts["ladder"].eq(ladder)].groupby(
            "E", observed=True
        ):
            ordered = group.sort_values("geometric_D_t")
            ax.plot(
                ordered["geometric_D_t"],
                ordered["penalty"],
                marker="o",
                label=f"E={expert}",
            )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set(title=f"ladder {ladder}", xlabel="geometric mean D_t", ylabel="K200 - K12")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Matched-D_t multilingual penalty changes over exposure", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    joint_path = figures / "joint_k_penalty_over_D_t.png"
    fig.savefig(joint_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    largest = (
        residual_cells.loc[residual_cells["fold_id"].eq("largest_N_act")]
        .groupby(["E", "K"], as_index=False)["M2_minus_M1_mse"]
        .mean()
    )
    e32 = (
        residual_cells.loc[residual_cells["fold_id"].eq("E32")]
        .groupby(["ladder", "K"], as_index=False)["M2_minus_M1_mse"]
        .mean()
    )
    k200 = (
        residual_cells.loc[
            residual_cells["fold_id"].eq("K200_common_languages")
        ]
        .groupby(["ladder", "E"], as_index=False)["M2_minus_M1_mse"]
        .mean()
    )
    _heatmap(
        axes[0, 0],
        largest,
        index="E",
        columns="K",
        values="M2_minus_M1_mse",
        title="largest-N fold: M2 minus M1 MSE",
        xlabel="K",
        ylabel="E",
        annotation_format=".1e",
    )
    _heatmap(
        axes[0, 1],
        e32,
        index="ladder",
        columns="K",
        values="M2_minus_M1_mse",
        title="E32 fold: M2 minus M1 MSE",
        xlabel="K",
        ylabel="ladder",
        annotation_format=".1e",
    )
    _heatmap(
        axes[1, 0],
        k200,
        index="ladder",
        columns="E",
        values="M2_minus_M1_mse",
        title="K200 fold: M2 minus M1 MSE",
        xlabel="E",
        ylabel="ladder",
        annotation_format=".1e",
    )
    horizon = residual_cells.loc[
        residual_cells["fold_id"].eq("K200_common_languages")
    ]
    for expert, group in horizon.groupby("E", observed=True):
        values = group.groupby("horizon_index")["M2_minus_M1_mse"].mean()
        axes[1, 1].plot(values.index, values.values, marker="o", label=f"E={expert}")
    axes[1, 1].axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    axes[1, 1].set(
        title="K200 fold over checkpoint horizon",
        xlabel="horizon index",
        ylabel="M2 minus M1 MSE",
    )
    axes[1, 1].legend(frameon=False, ncol=2)
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("Out-of-fold structural residual diagnostics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    residual_path = figures / "fold_residual_diagnostics.png"
    fig.savefig(residual_path, dpi=180)
    plt.close(fig)
    return {
        "separate_effects": separate_path,
        "joint_k_penalty_over_D_t": joint_path,
        "fold_residual_diagnostics": residual_path,
    }


def _markdown_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["floating"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    *,
    candidate_report: Mapping[str, Any],
    summaries: Mapping[str, pd.DataFrame],
    joint: Mapping[str, pd.DataFrame],
    parameter_table: pd.DataFrame,
    counts: Mapping[str, int],
) -> str:
    selection = candidate_report["selection"]
    expert = summaries["expert_summary"].query("E_low == 1 and E_high == 32")
    expert_table = expert.pivot(index="ladder", columns="K", values="mean").reset_index()
    k_penalty = summaries["language_count_summary"].query(
        "K_low == 12 and K_high == 200"
    )
    k_table = k_penalty.pivot(index="ladder", columns="E", values="mean").reset_index()
    attenuation = (
        joint["k_penalty_attenuation"]
        .groupby("ladder")["E1_minus_E32_penalty"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    k200_residual = joint["fold_residual_summary"].loc[
        joint["fold_residual_summary"]["fold_id"].eq("K200_common_languages")
    ]
    k200_by_ladder = (
        k200_residual.groupby("ladder", as_index=False)["M2_minus_M1_mse"].mean()
    )
    endpoint = parameter_table.loc[
        parameter_table["E"].isin([1, 32]),
        ["model", "fold_id", "E", "alpha", "beta", "phi", "psi"],
    ]
    return f"""# Stage 1 JointMoE-x1 interaction diagnostics

Status: exploratory model development. The sealed M2 candidate selection is
unchanged and remains the authoritative registered result.

## Analysis contract

- Input: 64 trajectories, 400 readouts, and 26,300 trained-language rows from
  the clean JointMoE-x1 release.
- Independent design unit: training trajectory. Checkpoints and languages are
  nested observations, not independent replications.
- E contrasts are exact at fixed ladder/N_act, K, D_t, and language.
- K contrasts use only shared languages and log-linear interpolation strictly
  inside observed D_t support; unmatched endpoints are censored.
- K1 is English-only and is never interpreted as the same multilingual
  estimand as K12/K50/K200.
- New functional forms motivated here are post-selection and must remain
  exploratory until nested or fresh validation is completed.

Raw diagnostic rows: {counts['expert_contrasts']:,} expert contrasts,
{counts['language_count_contrasts']:,} matched-K contrasts,
{counts['local_data_slopes']:,} local D_t slopes, and
{counts['active_size_contrasts']:,} active-size contrasts.

## 1. Expert effects at fixed active size

Mean E1-to-E32 adjusted-loss gain, balancing languages within each readout and
then horizons within each ladder/K cell:

{_markdown_table(expert_table)}

Expert benefit is positive throughout this aggregate table, generally larger
at broader K, and smaller for the larger C/D ladders. Separate adjacent-E
tables in the artifact bundle show pronounced diminishing returns after E16.

## 2. Language-count effects at matched D_t

Mean K200-minus-K12 adjusted-loss penalty:

{_markdown_table(k_table)}

Within this grid, larger E and N_act are associated with attenuated
multilingual pressure. The sign and magnitude also change over D_t;
checkpoint averaging therefore hides a clear descriptive K-by-data pattern.

E1-to-E32 attenuation of the K12-to-K200 penalty over matched horizons:

{_markdown_table(attenuation)}

Descriptive sign concordance across the designed cells:

{_markdown_table(joint['pattern_concordance'])}

These are consistency counts, not p-values: their cells share trajectories,
languages, and/or checkpoints and therefore are not independent replications.

## 3. Joint interaction and M2 residual signature

The frozen selection still chooses **{selection['selected_model']}** with
aggregate RMSE {selection['observed_aggregate_rmse']['M2']:.6f}, versus
{selection['observed_aggregate_rmse']['M1']:.6f} for M1.

On the K200 fold, M2-minus-M1 MSE by ladder is:

{_markdown_table(k200_by_ladder, digits=6)}

M2 loses on A/B but wins on C/D. The horizon-level residual table and figure
show a second crossover: for high E, M2 tends to improve early K200 checkpoints
and lose at later checkpoints. Conversely, M2 improves nearly all multilingual
cells on the largest-N holdout. This is evidence for real joint structure, but
also evidence that a single E-dependent K amplitude does not capture every
N_act and D_t regime equally well.

## 4. Structural interpretation

M2 permits alpha(E), beta(E), phi(E), and psi(E). It therefore represents
E-by-N_act, E-by-D_t, and E-by-K effects, but it does not explicitly permit the
N_act or D_t exponents to vary with K. The empirical signatures to test before
registering a successor are therefore:

1. K-dependent data scaling, beta(E,K), motivated by the horizon crossover;
2. K-dependent active-size scaling, alpha(E,K), motivated by attenuation of
   K pressure as N_act increases;
3. nonlinear or saturating E/K coordinates, motivated by diminishing E16-to-E32
   gains and non-constant K12-to-K50 versus K50-to-K200 increments; and
4. only then, bounded triple interactions if simpler nested terms fail.

The endpoint table records fold sensitivity of the fitted exponents; optimizer
audits establish numerical solutions, not parameter identification:

{_markdown_table(endpoint, digits=4)}

## 5. Recommended nested candidate family

Freeze the next exploratory family before fitting it. Let
`n = log(N_act / 1e8)` and `d = log(D_t / 1e9)`, and define a multilingual
coordinate `u_M(K)` that is zero for K1 and K12 and equals
`log(K / 12) / log(200 / 12)` for K50/K200. The zero at K1 prevents the new
interaction from using the English-only cell as evidence for a multilingual
K slope.

| candidate | exact change to M2 | added shared parameters |
|---|---|---|
| M2-KN | `alpha(E,K) = alpha(E) + gamma_N * u_M(K)` | 1 |
| M2-KD | `beta(E,K) = beta(E) + gamma_D * u_M(K)` | 1 |
| M2-KND | both changes together | 2 |

All three nest M2 exactly when the new coefficients are zero. Enforce the
registered `0.01 <= alpha(E,K), beta(E,K) <= 2` bounds at the E1/E32 and
K12/K200 corners. Fit both new coefficients as signed parameters; the
descriptive direction must not become a sign constraint. Do not add E-by-K
slopes, triple interactions, or saturation parameters in this round.
Saturation is plausible descriptively, but four E levels and only one interior
multilingual K level do not identify it cleanly enough to stack on top of the
two supported interactions.

The minimum exploratory comparison is balanced nested grouped validation:
four outer folds and three inner folds, with whole trajectories indivisible.
Selection and uncertainty must use trajectory-balanced metrics; languages and
checkpoints remain nested. Report paired M2 deltas, fold dispersion, targeted
residual-trend reduction, parameter/bound stability, and worst-regime
regressions.

Because this family was designed after inspecting all 64 trajectories and the
three primary holdouts, even nested same-grid validation remains exploratory.
Promotion over M2 requires a prospectively sealed fresh panel spanning
multiple N_act, E, and multilingual K regimes and independent seeds, with its
size and success threshold fixed before observing its losses.

## Decision boundary

Do not replace M2 or tune a new law against the same three primary holdouts.
Use this package to define a small nested candidate family, freeze it, and
evaluate it with nested whole-trajectory validation or genuinely untouched
data. Report any same-grid improvement as exploratory.
"""


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise InteractionDiagnosticError(f"unsupported table format: {path}")


def _git_state(repository: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InteractionDiagnosticError(f"cannot record Git provenance: {error}") from error
    return {
        "commit": commit,
        "dirty_worktree": dirty,
        "dirty_status_policy": "recorded_not_gating",
    }


def run_diagnostics(*, fit_root: Path, output_dir: Path) -> dict[str, Any]:
    fit_root = fit_root.resolve()
    output_dir = output_dir.resolve()
    receipt_path = fit_root / "input" / "input_receipt.json"
    candidate_path = fit_root / "reports" / "candidate_selection.json"
    receipt = read_json(receipt_path)
    frame = _validate_frame(load_fit_rows_snapshot(receipt))
    candidate = _validated_candidate_report(candidate_path)

    experts = expert_contrasts(frame)
    language_counts = language_count_contrasts(frame)
    data_slopes = local_data_slopes(frame)
    active_sizes = active_size_contrasts(frame)
    summaries = summarize_contrasts(experts, language_counts, data_slopes, active_sizes)
    residuals, residual_cells = fold_residuals(
        frame, fit_root=fit_root, candidate_report_path=candidate_path
    )
    parameters = parameter_endpoints(candidate_path)
    joint = joint_summaries(language_counts, experts, residual_cells, summaries)

    tables: dict[str, pd.DataFrame] = {
        "expert_contrasts": experts,
        "language_count_contrasts": language_counts,
        "local_data_slopes": data_slopes,
        "active_size_contrasts": active_sizes,
        "fold_residuals": residuals,
        "fold_residual_cells": residual_cells,
        "parameter_endpoints": parameters,
        **summaries,
        **joint,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, table in tables.items():
        extension = ".parquet" if name in {
            "expert_contrasts",
            "language_count_contrasts",
            "local_data_slopes",
            "active_size_contrasts",
            "fold_residuals",
        } else ".csv"
        path = output_dir / "tables" / f"{name}{extension}"
        _write_table(path, table)
        artifacts[str(path.relative_to(output_dir))] = {
            "sha256": _sha256(path),
            "rows": len(table),
        }
    figures = make_figures(summaries, joint, residual_cells, output_dir)
    for path in figures.values():
        artifacts[str(path.relative_to(output_dir))] = {"sha256": _sha256(path)}

    counts = {name: len(table) for name, table in tables.items()}
    report = build_report(
        candidate_report=candidate,
        summaries=summaries,
        joint=joint,
        parameter_table=parameters,
        counts=counts,
    )
    report_path = output_dir / "interaction_diagnostics_report.md"
    report_path.write_text(report, encoding="utf-8")
    artifacts[str(report_path.relative_to(output_dir))] = {"sha256": _sha256(report_path)}

    repository = Path(__file__).resolve().parents[2]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "exploratory_model_development",
        "git_provenance": _git_state(repository),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "inputs": {
            "fit_root": str(fit_root),
            "fit_rows_content_sha256": receipt["fit_rows_content_sha256"],
            "input_receipt_path": str(receipt_path),
            "input_receipt_file_sha256": sha256_file(receipt_path),
            "candidate_report_path": str(candidate_path),
            "candidate_report_file_sha256": sha256_file(candidate_path),
            "candidate_report_sha256": candidate["report_sha256"],
            "holdouts_path": str(fit_root / "input" / "holdouts.json"),
            "holdouts_file_sha256": sha256_file(fit_root / "input" / "holdouts.json"),
        },
        "analysis_contract": {
            "independent_unit": "training_trajectory",
            "nested_measurements": ["checkpoint_readout", "language"],
            "expert_contrasts": "exact_fixed_N_act_K_D_t_language",
            "language_count_contrasts": "shared_language_log_D_t_interpolation_in_support_only",
            "K1_scope": "English_only",
            "registered_M2_unchanged": True,
            "new_model_claim_status": "exploratory_until_nested_or_fresh_validation",
        },
        "counts": counts,
        "artifacts": artifacts,
    }
    manifest = {**unsigned, "manifest_sha256": canonical_json_sha256(unsigned)}
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_diagnostics(fit_root=args.fit_root, output_dir=args.output)
    print(
        json.dumps(
            {
                "manifest": str(args.output / "analysis_manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "tables": sum(
                    path.startswith("tables/") for path in manifest["artifacts"]
                ),
                "figures": sum(
                    path.startswith("figures/") for path in manifest["artifacts"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
