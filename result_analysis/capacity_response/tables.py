"""Load Stage-1 tables and adapt their column names for existing report commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from . import matched_exposure

import numpy as np
import pandas as pd


EXPERTS = (1, 4, 16, 32)
COVERAGE_CONTRASTS = ((1, 12), (1, 50), (1, 200), (12, 50), (12, 200), (50, 200))
RELEASE = Path("result_analysis/stage1_abcd_jointmoe_x1_v1")
SEED_SIDECAR = Path(__file__).resolve().parent / "reference_data/seed_attenuation_by_seed.csv"


def load_release_table(root: Path, name: str, columns: Sequence[str]) -> pd.DataFrame:
    """Load one manifest-verified table from the sealed release."""
    manifest = _release_manifest(root)
    entry = manifest["table_inventory"][name]
    path = root / RELEASE / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"hash mismatch for sealed table: {path}")
    frame = pd.read_parquet(path, columns=list(columns))
    if len(frame) != entry["rows"]:
        raise ValueError(f"row-count mismatch for sealed table: {path}")
    return frame


def load_primary(root: Path) -> pd.DataFrame:
    """Load trained-language loss, exposure, and model geometry."""
    validation = load_release_table(
        root,
        "validation_language",
        (
            "trajectory_id",
            "readout_id",
            "ladder",
            "experts",
            "language_count",
            "actual_tokens",
            "language",
            "metric_role",
            "raw_nll",
            "atlas_tao_unigram_adjusted_loss",
        ),
    ).query("metric_role == 'trained'")
    exposure = load_release_table(
        root,
        "data_exposure",
        ("trajectory_id", "readout_id", "language", "realized_self_exposure_tokens"),
    )
    configs = load_release_table(
        root,
        "model_configs",
        (
            "ladder",
            "experts",
            "declared_active_parameters",
            "configured_design_total_parameters",
        ),
    ).rename(
        columns={
            "experts": "E",
            "declared_active_parameters": "N_act",
            "configured_design_total_parameters": "N_total",
        }
    )

    frame = (
        validation.merge(
            exposure,
            on=["trajectory_id", "readout_id", "language"],
            validate="one_to_one",
        )
        .rename(
            columns={
                "experts": "E",
                "language_count": "K",
                "actual_tokens": "D_total",
                "realized_self_exposure_tokens": "D_t",
                "atlas_tao_unigram_adjusted_loss": "adjusted_loss",
            }
        )
        .merge(configs, on=["ladder", "E"], validate="many_to_one")
    )
    frame["R"] = frame.N_total / frame.N_act
    numeric = ["D_total", "D_t", "raw_nll", "adjusted_loss", "N_act", "N_total", "R"]
    if len(frame) != 26_300 or frame.trajectory_id.nunique() != 64:
        raise ValueError("unexpected Stage 1 trained-language inventory")
    if frame.readout_id.nunique() != 400:
        raise ValueError("unexpected Stage 1 readout inventory")
    if (
        not np.isfinite(frame[numeric].to_numpy(float)).all()
        or not (frame[["D_total", "D_t"]] > 0).all().all()
    ):
        raise ValueError("non-finite or non-positive Stage 1 inputs")
    return frame


def validate_primary_support(points: pd.DataFrame) -> None:
    """Validate the exact inventory used for the reported coverage comparison."""
    expected = {expert: 204 for expert in EXPERTS}
    if len(points) != 816 or points.groupby("E").size().to_dict() != expected:
        raise ValueError("unexpected K12-to-K200 matched-exposure support")
    if points.ladder.nunique() != 4 or points.language.nunique() != 12:
        raise ValueError("unexpected primary ladder or language support")
    trajectories = set(points.reference_trajectory_id) | set(points.target_trajectory_id)
    if len(trajectories) != 32:
        raise ValueError("unexpected primary source-trajectory support")


def expert_gain(frame: pd.DataFrame, languages: set[str], coverages: Sequence[int]) -> pd.DataFrame:
    """Return descriptive E1-minus-E32 gains for one fixed language cohort."""
    selected = frame.loc[frame.language.isin(languages) & frame.K.isin(coverages)]
    losses = (
        selected.groupby(["ladder", "K", "E", "D_total"], observed=True)
        .adjusted_loss.mean()
        .unstack("E")
    )
    if 1 not in losses or 32 not in losses:
        raise ValueError("missing E1 or E32 support")
    gains = (losses[1] - losses[32]).rename("gain").reset_index()
    active_sizes = selected.groupby("ladder", observed=True).N_act.median().rename("N_act")
    return gains.merge(active_sizes, on="ladder", validate="many_to_one")




def load_seed_sidecar(root: Path) -> pd.DataFrame:
    """Load the frozen confirmation shipped with this analysis.

    ``root`` remains accepted for caller compatibility; the fixed-hash sidecar
    is independent of the separately supplied primary-release location.
    """
    path = SEED_SIDECAR
    if sha256_file(path) != SEED_SIDECAR_SHA256:
        raise ValueError(f"unexpected seed sidecar: {path}")
    seed = pd.read_csv(path)
    required = {"seed", "E1_pressure", "E16_pressure", "E32_pressure", "relative_attenuation"}
    if not required.issubset(seed.columns) or len(seed) != 3 or seed.seed.nunique() != 3:
        raise ValueError(f"unexpected seed inventory: {path}")
    expected = 1 - seed.E32_pressure / seed.E1_pressure
    if not np.allclose(seed.relative_attenuation, expected, atol=1e-12):
        raise ValueError(f"inconsistent seed-derived quantities: {path}")
    return seed.sort_values("seed").reset_index(drop=True)



# Translate persisted table names only at the report boundary.
COLUMNS = {"E": "expert_count", "K": "language_coverage", "N_act": "active_parameters",
           "N_total": "stored_parameters", "D_total": "total_tokens", "D_t": "target_exposure",
           "pressure": "coverage_gap", "reference_K": "reference_coverage", "target_K": "target_coverage"}


def matched_pressure(frame, reference_k=12, target_k=200):
    """Use the coverage-gap estimator while preserving report column names."""
    return matched_exposure.coverage_gap(frame.rename(columns=COLUMNS), reference_k, target_k).rename(
        columns={v: k for k, v in COLUMNS.items()})


def summarize(points, expected_experts=EXPERTS):
    """Keep persisted report schemas unchanged when aggregating coverage gaps."""
    tables = matched_exposure.summarize(points.rename(columns=COLUMNS), expected_experts)
    return tuple(table.rename(columns={v: k for k, v in COLUMNS.items()}) for table in tables)

RELEASE_MANIFEST_SHA256 = "807b2476ff15c71f1164b92b4c8c12136c7ca9f610ce85acc24ea4a99723e806"

RELEASE_INTEGRITY_ID = "50a54525dae4f3102d0ff6811c52b38b74a82d9c5d5759c8891f1bf9c63ffc81"

SEED_SIDECAR_SHA256 = "20da102777a37095b572786094f7fa20f9774ee35b631356edd7a2ffe28c1000"

def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _release_manifest(root: Path) -> dict[str, Any]:
    path = root / RELEASE / "release_manifest.json"
    if sha256_file(path) != RELEASE_MANIFEST_SHA256:
        raise ValueError(f"unexpected release manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_integrity_sha256") != RELEASE_INTEGRITY_ID:
        raise ValueError(f"unexpected release identity: {path}")
    return manifest
