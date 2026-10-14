"""Loss-blind exposure-component construction for exploratory M3/M4 fits."""
from __future__ import annotations

from typing import Any

import numpy as np

TRANSFER_COUNT = 3


class AtlasExposureError(RuntimeError):
    """Raised when multilingual exposure components cannot be constructed."""


def exposure_components(
    frame: Any,
    *,
    transfer_count: int = TRANSFER_COUNT,
) -> tuple[Any, dict[str, tuple[str, ...]]]:
    """Derive exact top-co-sampled and remainder exposure for every target row.

    Source ranking is loss-blind. For each target, each source receives its
    mean normalized exposure over the target's observed K levels, with zero
    exposure when that source is absent from a narrower mixture. Ties are
    broken by language id.
    """
    import pandas as pd

    required = {"readout_id", "language", "K", "D_total", "D_t"}
    if not required <= set(frame.columns) or transfer_count < 1:
        raise AtlasExposureError(
            "fit rows lack exposure columns or transfer_count is invalid"
        )
    if frame.duplicated(["readout_id", "language"]).any():
        raise AtlasExposureError("fit rows duplicate readout/language exposure keys")
    numeric = frame[["K", "D_total", "D_t"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric[:, 1:] <= 0.0):
        raise AtlasExposureError("fit exposures must be finite and positive")
    sizes = frame.groupby("readout_id", sort=False).size()
    declared_k = frame.groupby("readout_id", sort=False)["K"].first().astype(int)
    if not np.array_equal(
        sizes.to_numpy(dtype=np.int64), declared_k.to_numpy(dtype=np.int64)
    ):
        raise AtlasExposureError("each readout must expose exactly K target-language rows")
    totals = frame.groupby("readout_id", sort=False)["D_total"].first()
    summed = frame.groupby("readout_id", sort=False)["D_t"].sum()
    if not np.allclose(
        summed.to_numpy(), totals.to_numpy(), rtol=1e-12, atol=1e-6
    ):
        raise AtlasExposureError("source exposures do not sum to D_total within readout")

    shares = frame.assign(_share=frame["D_t"] / frame["D_total"])
    mean_share = shares.groupby(["K", "language"], sort=True)["_share"].mean().to_dict()
    languages = sorted(str(value) for value in frame["language"].unique())
    observed_k = {
        language: sorted(
            int(value)
            for value in frame.loc[frame["language"] == language, "K"].unique()
        )
        for language in languages
    }
    transfer_sets: dict[str, tuple[str, ...]] = {}
    for target in languages:
        levels = observed_k[target]
        ranked = []
        for source in languages:
            if source == target:
                continue
            score = sum(
                float(mean_share.get((level, source), 0.0)) for level in levels
            ) / len(levels)
            if score > 0.0:
                ranked.append((source, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        transfer_sets[target] = tuple(
            source for source, _ in ranked[:transfer_count]
        )

    exposure_lookup = {
        (str(readout), str(language)): float(exposure)
        for readout, language, exposure in frame[
            ["readout_id", "language", "D_t"]
        ].itertuples(index=False)
    }
    transfer = np.asarray(
        [
            sum(
                exposure_lookup.get((str(row.readout_id), source), 0.0)
                for source in transfer_sets[str(row.language)]
            )
            for row in frame[["readout_id", "language"]].itertuples(index=False)
        ],
        dtype=np.float64,
    )
    target = frame["D_t"].to_numpy(dtype=np.float64)
    other = frame["D_total"].to_numpy(dtype=np.float64) - target
    remainder = other - transfer
    tolerance = np.maximum(
        1.0, frame["D_total"].to_numpy(dtype=np.float64) * 1e-12
    )
    if np.any(transfer < -tolerance) or np.any(remainder < -tolerance):
        raise AtlasExposureError("derived transfer or remainder exposure is negative")
    components = pd.DataFrame(
        {
            "D_target": target,
            "D_transfer": np.maximum(transfer, 0.0),
            "D_other": np.maximum(remainder, 0.0),
        },
        index=frame.index,
    )
    reconstructed = components.sum(axis=1).to_numpy(dtype=np.float64)
    if not np.allclose(
        reconstructed,
        frame["D_total"].to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=1e-6,
    ):
        raise AtlasExposureError("effective-exposure components do not reconstruct D_total")
    return components, transfer_sets
