#!/usr/bin/env python3
"""Generate the publication-oriented Stage 1 A--D analysis notebook.

The generated notebook is deliberately a release consumer: every scientific
input is read from ``<release-dir>/tables/*.parquet``.  It never follows source
paths embedded in those tables back into mutable run directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = ROOT / "result_analysis" / "stage1_abcd_v1"


def _md(text: str) -> Any:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def _code(text: str) -> Any:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def _cells(release_dir: Path) -> list[Any]:
    release_hint = repr(str(release_dir.resolve()))
    return [
        _md(
            """
# Stage 1 A–D: publication analysis of the sealed release

This notebook is the canonical, reproducible analysis consumer for the Stage 1
A–D multilingual pretraining release. It reads **only typed Parquet tables**
under `tables/`; it does not read training logs, validation JSON, checkpoints,
Slurm state, or other mutable source artifacts.

The analysis follows a NeurIPS-style reporting boundary:

- **Observed:** completed training, checkpoint readouts, per-language
  validation, telemetry, and snapshotted accounting included in the release.
- **Configured:** architectures, data tiers, schedules, and seeds recorded by
  the experiment specifications.
- **Derived:** aggregates and figures computed below from released Parquet rows.
- **Not claimed:** full-grid independent-seed uncertainty, causal architecture
  effects, or facility-wide energy/carbon estimates.

Within the primary 64-trajectory panel, checkpoint and language measurements
are repeated observations from one configured model seed. They are not
independent experimental replicates; any separately released seed-confirmation
sidecar has its own narrower scope and is not pooled into this analysis.
"""
        ),
        _code(
            f"""
from __future__ import annotations

from pathlib import Path
import json
import math
import platform
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import Markdown, display

RELEASE_HINT = Path({release_hint})
cwd = Path.cwd().resolve()
release_candidates = [
    RELEASE_HINT,
    cwd,
    cwd / "result_analysis" / "stage1_abcd_v1",
]
RELEASE_DIR = next(
    (path for path in release_candidates if (path / "tables").is_dir()),
    RELEASE_HINT,
).resolve()
TABLE_DIR = RELEASE_DIR / "tables"
if not TABLE_DIR.is_dir():
    raise RuntimeError(
        f"Release table directory does not exist: {{TABLE_DIR}}. "
        "Generate the central A-D release before executing this notebook."
    )

release_metadata_path = RELEASE_DIR / "release_manifest.json"
RELEASE_METADATA = (
    json.loads(release_metadata_path.read_text(encoding="utf-8"))
    if release_metadata_path.is_file()
    else {{}}
)
ANALYSIS_BOUNDARY = str(
    RELEASE_METADATA.get(
        "analysis_boundary",
        "The release is analysis-ready; follow its declared protocol before "
        "making confirmatory claims.",
    )
)
ANALYSIS_NEXT_ACTION = str(
    RELEASE_METADATA.get(
        "analysis_next_action",
        "Follow the release-specific analysis protocol and preserve all "
        "selection decisions in immutable receipts.",
    )
)

ANALYSIS_DIR = RELEASE_DIR.parent / f"{{RELEASE_DIR.name}}_analysis_outputs"
FIGURE_DIR = ANALYSIS_DIR / "figures"
TABLE_OUTPUT_DIR = ANALYSIS_DIR / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({{
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
}})
LADDER_COLORS = {{
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
    "D": "#CC79A7",
}}
K_COLORS = {{
    1: "#0072B2",
    12: "#E69F00",
    50: "#009E73",
    200: "#CC79A7",
}}

CORE_TABLES = (
    "trajectories",
    "readouts",
    "validation_aggregates",
    "validation_language",
    "validation_scope",
    "training_metrics",
    "training_summary",
    "gpu_samples",
)
OPTIONAL_TABLES = (
    "compute_summary",
    "readout_compute_attribution",
    "unique_physical_study_compute",
    "physical_compute_categories",
    "routing_layers",
    "release_summary",
    "readout_training_bindings",
    "qa_issues",
)

def table_path(name):
    return TABLE_DIR / f"{{name}}.parquet"

def parquet_columns(name):
    path = table_path(name)
    if not path.is_file():
        return []
    return pq.ParquetFile(path).schema_arrow.names

def parquet_row_count(name):
    path = table_path(name)
    if not path.is_file():
        return None
    return pq.ParquetFile(path).metadata.num_rows

def load_table(name, *, columns=None, filters=None, required=True):
    path = table_path(name)
    if not path.is_file():
        if required:
            raise RuntimeError(f"Missing required release table: {{path}}")
        return pd.DataFrame()
    available = set(parquet_columns(name))
    selected = None
    if columns is not None:
        missing = sorted(set(columns) - available)
        if missing:
            raise RuntimeError(
                f"{{name}}.parquet is missing required columns: {{missing}}"
            )
        selected = list(columns)
    return pd.read_parquet(path, columns=selected, filters=filters)

missing_core = [name for name in CORE_TABLES if not table_path(name).is_file()]
if missing_core:
    raise RuntimeError(f"Missing canonical core tables: {{missing_core}}")

print("Release:", RELEASE_DIR)
print("Python:", sys.executable)
print("Versions:", {{
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
}})
"""
        ),
        _md(
            """
## 1. Load the analytical contract and fail closed

The exact canonical expectations are checked before analysis. Raw event tables
are checked through Parquet metadata first, so the completeness audit does not
depend on loading millions of rows into memory. Optional accounting and routing
tables are reported transparently rather than silently imputed.
"""
        ),
        _code(
            """
trajectories = load_table("trajectories")
readouts = load_table("readouts")
validation_aggregates = load_table("validation_aggregates")
validation_language = load_table("validation_language")
validation_scope = load_table("validation_scope")
training_summary = load_table("training_summary")
compute_summary = load_table("compute_summary", required=False)
readout_compute_attribution = load_table(
    "readout_compute_attribution", required=False
)
unique_physical_study_compute = load_table(
    "unique_physical_study_compute", required=False
)
physical_compute_categories = load_table(
    "physical_compute_categories", required=False
)
release_summary = load_table("release_summary", required=False)
readout_training_bindings = load_table(
    "readout_training_bindings", required=False
)
qa_issues = load_table("qa_issues", required=False)

required_columns = {
    "trajectories": {
        "trajectory_id", "ladder", "experts", "language_count",
        "independent_seed_replicates",
    },
    "readouts": {
        "readout_id", "trajectory_id", "training_job_id", "ladder", "experts",
        "language_count", "nominal_tokens", "actual_tokens",
        "primary_metric_name", "primary_metric_value",
    },
    "validation_aggregates": {
        "readout_id", "aggregate_kind", "cohort", "metric_name", "metric_value",
        "equal_language_adjusted_loss",
    },
    "validation_language": {
        "readout_id", "trajectory_id", "ladder", "experts", "language_count",
        "nominal_tokens", "language", "metric_role",
        "atlas_tao_unigram_adjusted_loss", "raw_nll", "token_count",
    },
    "validation_scope": {
        "validation_scope_id", "ladder", "language_count", "readout_count",
        "trained_languages_per_readout", "zero_shot_languages_per_readout",
        "scope_status",
    },
    "training_summary": {
        "training_job_id", "trajectory_id", "ladder", "experts",
        "language_count", "mean_tokens_per_second",
        "mean_gpu_utilization_pct", "board_energy_estimate_kwh",
    },
}
loaded = {
    "trajectories": trajectories,
    "readouts": readouts,
    "validation_aggregates": validation_aggregates,
    "validation_language": validation_language,
    "validation_scope": validation_scope,
    "training_summary": training_summary,
}
for name, columns in required_columns.items():
    missing = sorted(columns - set(loaded[name].columns))
    if missing:
        raise RuntimeError(f"{name} is missing canonical columns: {missing}")

expected = {
    "trajectories": 64,
    "readouts": 400,
    "validation_language": 46_200,
    "validation_scope": 16,
    "trained_language_rows": 26_300,
    "zero_shot_language_rows": 19_900,
}
observed = {
    "trajectories": len(trajectories),
    "readouts": len(readouts),
    "validation_language": len(validation_language),
    "validation_scope": len(validation_scope),
    "trained_language_rows": int(
        validation_language["metric_role"].eq("trained").sum()
    ),
    "zero_shot_language_rows": int(
        validation_language["metric_role"].eq("zero_shot").sum()
    ),
}
if observed != expected:
    raise RuntimeError(
        "Canonical release completeness failure: "
        f"expected={expected}, observed={observed}"
    )

if set(readouts["ladder"]) != {"A", "B", "C", "D"}:
    raise RuntimeError(f"Unexpected ladder set: {sorted(readouts['ladder'].unique())}")
if readouts["readout_id"].duplicated().any():
    raise RuntimeError("Duplicate canonical readout IDs")
if trajectories["trajectory_id"].duplicated().any():
    raise RuntimeError("Duplicate trajectory IDs")
if not trajectories["independent_seed_replicates"].eq(1).all():
    raise RuntimeError("Unexpected replicate metadata; inspect the release contract")

ladder_expected = {"A": 80, "B": 96, "C": 112, "D": 112}
ladder_observed = readouts.groupby("ladder").size().to_dict()
if ladder_observed != ladder_expected:
    raise RuntimeError(
        f"Ladder readout mismatch: expected={ladder_expected}, "
        f"observed={ladder_observed}"
    )

event_rows = {
    "training_metrics": parquet_row_count("training_metrics"),
    "gpu_samples": parquet_row_count("gpu_samples"),
}
if any(value is None or value <= 0 for value in event_rows.values()):
    raise RuntimeError(f"Empty required event table: {event_rows}")

completeness = pd.DataFrame([
    {
        "component": key,
        "expected_rows": value,
        "observed_rows": observed[key],
        "complete": value == observed[key],
    }
    for key, value in expected.items()
])
display(completeness)
display(
    readouts.groupby(["ladder", "experts", "language_count"])
    .agg(readouts=("readout_id", "size"),
         min_tokens=("actual_tokens", "min"),
         max_tokens=("actual_tokens", "max"))
    .reset_index()
)
display(validation_scope.sort_values(["ladder", "language_count"]))
print("Raw event rows:", event_rows)
print("Optional tables:", {
    name: parquet_row_count(name) for name in OPTIONAL_TABLES
})

completeness.to_csv(TABLE_OUTPUT_DIR / "01_completeness.csv", index=False)
validation_scope.to_csv(
    TABLE_OUTPUT_DIR / "01_validation_scope.csv", index=False
)
"""
        ),
        _code(
            """
fig, ax = plt.subplots(figsize=(8.2, 3.8))
positions = np.arange(len(completeness))
expected_values = completeness["expected_rows"].to_numpy()
observed_values = completeness["observed_rows"].to_numpy()
ax.barh(positions, expected_values, color="#d9d9d9", label="expected")
ax.barh(positions, observed_values, color="#0072B2", alpha=0.85, label="observed")
ax.set_yticks(positions, completeness["component"])
ax.set_xlabel("Rows (log scale)")
ax.set_xscale("log")
ax.set_title("Canonical release completeness")
ax.legend(frameon=False, loc="lower right")
for position, value in zip(positions, observed_values, strict=True):
    ax.text(value * 1.03, position, f"{value:,}", va="center", fontsize=8)
fig.savefig(FIGURE_DIR / "01_release_completeness.png")
plt.show()
"""
        ),
        _md(
            """
## 2. Metric integrity and matched descriptive contrasts

The declared primary value in each readout is checked against its normalized
`declared_metric/primary` aggregate. A/B and C/D differences are then computed
only for matched expert count, language tier, and nominal token readout.
Negative right-minus-left deltas indicate lower loss for B or D. These are
descriptive paired checkpoint contrasts, not independent-seed hypothesis tests.
"""
        ),
        _code(
            """
declared_primary = (
    validation_aggregates.loc[
        validation_aggregates["aggregate_kind"].eq("declared_metric")
        & validation_aggregates["cohort"].eq("primary"),
        ["readout_id", "metric_name", "metric_value"],
    ]
    .rename(columns={
        "metric_name": "aggregate_metric_name",
        "metric_value": "aggregate_metric_value",
    })
)
integrity = readouts[
    ["readout_id", "primary_metric_name", "primary_metric_value"]
].merge(declared_primary, on="readout_id", how="left", validate="one_to_one")
if integrity["aggregate_metric_value"].isna().any():
    raise RuntimeError("Missing declared primary aggregate for a canonical readout")
if not integrity["primary_metric_name"].eq(
    integrity["aggregate_metric_name"]
).all():
    raise RuntimeError("Primary metric names disagree with aggregate rows")
max_metric_delta = (
    integrity["primary_metric_value"]
    - integrity["aggregate_metric_value"]
).abs().max()
if not np.isfinite(max_metric_delta) or max_metric_delta > 1e-12:
    raise RuntimeError(f"Primary aggregate mismatch: {max_metric_delta}")
print(f"Primary aggregate audit: 400/400 exact; max |delta|={max_metric_delta:.3g}")

role_map = {"trained_languages": "trained", "zero_shot_languages": "zero_shot"}
role_aggregates = validation_aggregates.loc[
    validation_aggregates["aggregate_kind"].eq("metric_role")
    & validation_aggregates["status"].eq("complete")
].copy()
role_aggregates["metric_role"] = role_aggregates["cohort"].map(role_map)
role_recomputed = (
    validation_language.groupby(["readout_id", "metric_role"], observed=True)
    .agg(
        recomputed_equal_adjusted=("atlas_tao_unigram_adjusted_loss", "mean"),
        recomputed_equal_raw=("raw_nll", "mean"),
        adjusted_nll_sum=("adjusted_nll_sum", "sum"),
        nll_sum=("nll_sum", "sum"),
        token_count=("token_count", "sum"),
    )
    .reset_index()
)
role_recomputed["recomputed_micro_adjusted"] = (
    role_recomputed["adjusted_nll_sum"] / role_recomputed["token_count"]
)
role_recomputed["recomputed_micro_raw"] = (
    role_recomputed["nll_sum"] / role_recomputed["token_count"]
)
role_audit = role_aggregates.merge(
    role_recomputed, on=["readout_id", "metric_role"],
    how="left", validate="one_to_one",
)
role_checks = {
    "equal_language_adjusted_loss": "recomputed_equal_adjusted",
    "equal_language_raw_nll": "recomputed_equal_raw",
    "token_micro_adjusted_loss": "recomputed_micro_adjusted",
    "token_micro_raw_nll": "recomputed_micro_raw",
}
role_deltas = {
    released: (
        pd.to_numeric(role_audit[released], errors="coerce")
        - pd.to_numeric(role_audit[recomputed], errors="coerce")
    ).abs().max()
    for released, recomputed in role_checks.items()
}
if any(not np.isfinite(value) or value > 1e-10 for value in role_deltas.values()):
    raise RuntimeError(f"Per-role aggregate reconstruction failed: {role_deltas}")
print(
    f"Metric-role aggregate reconstruction: {len(role_audit):,} complete rows; "
    f"max deltas={role_deltas}"
)

def paired_contrast(frame, left, right):
    matched = (
        frame.loc[frame["ladder"].isin([left, right])]
        .pivot(
            index=["experts", "language_count", "nominal_tokens"],
            columns="ladder",
            values="primary_metric_value",
        )
        .dropna(subset=[left, right])
        .reset_index()
    )
    matched["comparison"] = f"{right} - {left}"
    matched["loss_delta"] = matched[right] - matched[left]
    return matched

ab = paired_contrast(readouts, "A", "B")
cd = paired_contrast(readouts, "C", "D")
paired = pd.concat([ab, cd], ignore_index=True)
paired_summary = (
    paired.groupby("comparison")["loss_delta"]
    .agg(
        matched_readouts="size",
        mean_delta="mean",
        median_delta="median",
        min_delta="min",
        max_delta="max",
    )
    .reset_index()
)
paired_summary["right_lower_loss_fraction"] = paired.groupby(
    "comparison"
)["loss_delta"].apply(lambda values: float((values < 0).mean())).to_numpy()
display(paired_summary)
paired.to_csv(TABLE_OUTPUT_DIR / "02_matched_ladder_contrasts.csv", index=False)
paired_summary.to_csv(
    TABLE_OUTPUT_DIR / "02_matched_ladder_contrast_summary.csv", index=False
)
"""
        ),
        _md(
            """
## 3. Validation learning curves across every A–D / E / K trajectory

Each panel fixes ladder and expert count; colors distinguish K. The x-axis uses
actual consumed tokens, while nominal token readouts remain in the exported
table. Lines connect checkpoints within the same configured trajectory only.
"""
        ),
        _code(
            """
curve_data = readouts.copy()
curve_data["actual_tokens_B"] = curve_data["actual_tokens"] / 1e9
curve_data["nominal_tokens_B"] = curve_data["nominal_tokens"] / 1e9

fig, axes = plt.subplots(
    4, 4, figsize=(15.5, 13.0), sharex=True, sharey=True, constrained_layout=True
)
for row_index, ladder in enumerate(("A", "B", "C", "D")):
    for column_index, experts in enumerate((1, 4, 16, 32)):
        ax = axes[row_index, column_index]
        panel = curve_data.loc[
            curve_data["ladder"].eq(ladder)
            & curve_data["experts"].eq(experts)
        ]
        for language_count in (1, 12, 50, 200):
            series = panel.loc[
                panel["language_count"].eq(language_count)
            ].sort_values("actual_tokens")
            if series.empty:
                continue
            ax.plot(
                series["actual_tokens_B"],
                series["primary_metric_value"],
                marker="o",
                markersize=2.8,
                linewidth=1.25,
                color=K_COLORS[language_count],
                label=f"K={language_count}",
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"Ladder {ladder}, E={experts}")
        if row_index == 3:
            ax.set_xlabel("Actual consumed tokens (billions)")
        if column_index == 0:
            ax.set_ylabel("Declared primary validation loss")
        if row_index == 0 and column_index == 3:
            ax.legend(frameon=False, title="Language tier")

fig.suptitle(
    "Stage 1 A–D validation trajectories (all 64 configured trajectories)",
    fontsize=14,
    y=1.015,
)
fig.savefig(FIGURE_DIR / "03_validation_curves_all_trajectories.png")
plt.show()

curve_data[[
    "readout_id", "trajectory_id", "ladder", "experts", "language_count",
    "nominal_tokens", "actual_tokens", "primary_metric_name",
    "primary_metric_value",
]].to_csv(TABLE_OUTPUT_DIR / "03_validation_curves.csv", index=False)
"""
        ),
        _md(
            """
## 4. Trained-language versus zero-shot behavior

Per-language adjusted losses are first averaged equally within each readout and
metric role. The figure shows the distribution at the final checkpoint of each
trajectory. Only K1 has zero-shot rows: each K1 readout contains one trained
and 199 zero-shot languages. K12, K50, and K200 are trained-tier-only. To avoid
a scope-confounded comparison, both plotted roles are restricted to the same 16
final K1 readouts (four expert configurations per ladder).
"""
        ),
        _code(
            """
role_by_readout = (
    validation_language.groupby(
        [
            "readout_id", "trajectory_id", "ladder", "experts",
            "language_count", "nominal_tokens", "metric_role",
        ],
        observed=True,
    )
    .agg(
        equal_language_adjusted_loss=(
            "atlas_tao_unigram_adjusted_loss", "mean"
        ),
        equal_language_raw_nll=("raw_nll", "mean"),
        languages=("language", "nunique"),
        tokens=("token_count", "sum"),
    )
    .reset_index()
)
final_readout_ids = set(
    readouts.sort_values(["trajectory_id", "actual_tokens"])
    .groupby("trajectory_id", observed=True)
    .tail(1)["readout_id"]
)
final_roles = role_by_readout.loc[
    role_by_readout["readout_id"].isin(final_readout_ids)
    & role_by_readout["language_count"].eq(1)
].copy()

roles = ("trained", "zero_shot")
fig, ax = plt.subplots(figsize=(10.0, 4.8))
positions = []
values = []
labels = []
colors = []
position = 0.0
for ladder in ("A", "B", "C", "D"):
    for role in roles:
        sample = final_roles.loc[
            final_roles["ladder"].eq(ladder)
            & final_roles["metric_role"].eq(role),
            "equal_language_adjusted_loss",
        ].dropna().to_numpy()
        if sample.size:
            positions.append(position)
            values.append(sample)
            labels.append(
                f"{ladder}\\n{role.replace('_', '-')}\\n(K1 only)"
            )
            colors.append(LADDER_COLORS[ladder])
        position += 1.0
    position += 0.45
boxes = ax.boxplot(
    values, positions=positions, widths=0.65, patch_artist=True,
    showfliers=False, medianprops={"color": "black"},
)
for patch, color in zip(boxes["boxes"], colors, strict=True):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
ax.set_xticks(positions, labels)
ax.set_ylabel("Equal-language adjusted loss")
ax.set_title(
    "K1 only: 1 trained language vs 199 zero-shot languages"
)
fig.savefig(FIGURE_DIR / "04_trained_vs_zero_shot.png")
plt.show()

role_summary = (
    final_roles.groupby(["ladder", "metric_role"], observed=True)
    .agg(
        trajectories=("readout_id", "size"),
        mean_loss=("equal_language_adjusted_loss", "mean"),
        median_loss=("equal_language_adjusted_loss", "median"),
        min_loss=("equal_language_adjusted_loss", "min"),
        max_loss=("equal_language_adjusted_loss", "max"),
    )
    .reset_index()
)
role_summary["language_tier_scope"] = "K1_only"
k1_paired = (
    final_roles.pivot(
        index=["readout_id", "trajectory_id", "ladder", "experts"],
        columns="metric_role",
        values="equal_language_adjusted_loss",
    )
    .reset_index()
)
if len(k1_paired) != 16 or k1_paired[["trained", "zero_shot"]].isna().any().any():
    raise RuntimeError("K1 trained/zero-shot pairing is incomplete")
k1_paired["zero_shot_minus_trained"] = (
    k1_paired["zero_shot"] - k1_paired["trained"]
)
display(role_summary)
display(k1_paired)
role_by_readout.to_csv(
    TABLE_OUTPUT_DIR / "04_metric_role_by_readout.csv", index=False
)
role_summary.to_csv(
    TABLE_OUTPUT_DIR / "04_final_metric_role_summary.csv", index=False
)
k1_paired.to_csv(
    TABLE_OUTPUT_DIR / "04_k1_trained_zero_shot_paired.csv", index=False
)
"""
        ),
        _md(
            """
## 5. Training-loss trajectories with trunk/branch lineage retained

Only `training_iteration` events are loaded. Curves are aggregated by the
released trajectory ID and consumed-token coordinate; duplicate observations at
the same coordinate are represented by their median. This prevents individual
log attempts from being treated as independent runs. The underlying event rows
retain `training_job_id`, `segment_kind`, and parent lineage for deeper audits.
"""
        ),
        _code(
            """
training_columns = [
    "event_type", "training_job_id", "trajectory_id", "ladder", "experts",
    "language_count", "segment_kind", "consumed_tokens", "lm_loss",
]
training_events = load_table(
    "training_metrics",
    columns=training_columns,
    filters=[("event_type", "==", "training_iteration")],
)
training_events["consumed_tokens"] = pd.to_numeric(
    training_events["consumed_tokens"], errors="coerce"
)
training_events["lm_loss"] = pd.to_numeric(
    training_events["lm_loss"], errors="coerce"
)
training_events = training_events.dropna(subset=["consumed_tokens", "lm_loss"])
if training_events.empty:
    raise RuntimeError("No finite training-iteration losses in the release")

loss_curves = (
    training_events.groupby(
        [
            "trajectory_id", "ladder", "experts", "language_count",
            "consumed_tokens",
        ],
        observed=True,
    )["lm_loss"]
    .median()
    .rename("median_lm_loss")
    .reset_index()
)

def thin_curve(frame, maximum_points=220):
    ordered = frame.sort_values("consumed_tokens")
    if len(ordered) <= maximum_points:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, maximum_points).round().astype(int)
    return ordered.iloc[np.unique(indices)]

fig, axes = plt.subplots(
    4, 4, figsize=(15.5, 13.0), sharex=True, sharey=True, constrained_layout=True
)
for row_index, ladder in enumerate(("A", "B", "C", "D")):
    for column_index, experts in enumerate((1, 4, 16, 32)):
        ax = axes[row_index, column_index]
        panel = loss_curves.loc[
            loss_curves["ladder"].eq(ladder)
            & loss_curves["experts"].eq(experts)
        ]
        for language_count in (1, 12, 50, 200):
            series = thin_curve(
                panel.loc[panel["language_count"].eq(language_count)]
            )
            if series.empty:
                continue
            ax.plot(
                series["consumed_tokens"] / 1e9,
                series["median_lm_loss"],
                linewidth=1.0,
                color=K_COLORS[language_count],
                label=f"K={language_count}",
            )
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_title(f"Ladder {ladder}, E={experts}")
        if row_index == 3:
            ax.set_xlabel("Logged consumed tokens (billions)")
        if column_index == 0:
            ax.set_ylabel("Megatron LM loss")
        if row_index == 0 and column_index == 3:
            ax.legend(frameon=False, title="Language tier")
fig.suptitle(
    "Training loss by released trajectory (median at duplicate token coordinates)",
    fontsize=14,
    y=1.015,
)
fig.savefig(FIGURE_DIR / "05_training_loss_all_trajectories.png")
plt.show()

training_inventory = (
    training_events.groupby(
        ["ladder", "experts", "language_count", "segment_kind"],
        observed=True,
    )
    .agg(
        metric_rows=("lm_loss", "size"),
        jobs=("training_job_id", "nunique"),
        min_consumed_tokens=("consumed_tokens", "min"),
        max_consumed_tokens=("consumed_tokens", "max"),
    )
    .reset_index()
)
display(training_inventory)
training_inventory.to_csv(
    TABLE_OUTPUT_DIR / "05_training_metric_inventory.csv", index=False
)
"""
        ),
        _md(
            """
## 6. Throughput, MFU, utilization, and observed board energy

Every point summarized here is one unique production training job. Throughput
is job-level tokens/s. MFU is shown only when the release contains a configured
hardware-peak denominator; otherwise the panel explicitly falls back to the
logged model TFLOP/s/GPU. GPU utilization comes from sampled `nvidia-smi`.
Board energy is trapezoidal integration of observed GPU board power only—it
excludes host, networking, storage, cooling, and facility overhead.
"""
        ),
        _code(
            """
if training_summary["training_job_id"].duplicated().any():
    raise RuntimeError("training_summary must have one row per training job")

def first_finite_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return column
    return None

throughput_column = first_finite_column(
    training_summary,
    ["mean_tokens_per_second", "median_tokens_per_second"],
)
mfu_column = first_finite_column(
    training_summary,
    ["mean_mfu_pct_of_configured_peak", "mfu_pct"],
)
tflops_column = first_finite_column(
    training_summary,
    ["mean_model_tflops_per_gpu"],
)
utilization_column = first_finite_column(
    training_summary,
    ["mean_gpu_utilization_pct"],
)
energy_column = first_finite_column(
    training_summary,
    ["board_energy_estimate_kwh"],
)

metric_panels = [
    (throughput_column, "Observed throughput", "Tokens/s"),
    (
        mfu_column or tflops_column,
        "MFU (% configured peak)" if mfu_column else
        "MFU unavailable; model TFLOP/s/GPU shown",
        "% of configured peak" if mfu_column else "Model TFLOP/s/GPU",
    ),
    (utilization_column, "Mean sampled GPU utilization", "GPU utilization (%)"),
    (energy_column, "Observed GPU board-energy estimate", "kWh per job"),
]

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
for ax, (column, title, ylabel) in zip(axes.flat, metric_panels, strict=True):
    if column is None:
        ax.text(
            0.5, 0.5, "Not computable from released fields",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_title(title)
        ax.set_axis_off()
        continue
    grouped = (
        training_summary.assign(
            _metric=pd.to_numeric(training_summary[column], errors="coerce")
        )
        .dropna(subset=["_metric"])
        .groupby(["ladder", "experts"], observed=True)["_metric"]
        .median()
        .reset_index()
    )
    for ladder in ("A", "B", "C", "D"):
        series = grouped.loc[grouped["ladder"].eq(ladder)].sort_values("experts")
        if not series.empty:
            ax.plot(
                series["experts"], series["_metric"], marker="o",
                color=LADDER_COLORS[ladder], label=f"Ladder {ladder}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 32], ["1", "4", "16", "32"])
    ax.set_xlabel("Experts (E)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
axes[0, 0].legend(frameon=False, ncol=2)
fig.savefig(FIGURE_DIR / "06_efficiency_utilization_energy.png")
plt.show()

efficiency_summary = (
    training_summary.groupby(["ladder", "experts"], observed=True)
    .agg(
        jobs=("training_job_id", "nunique"),
        median_tokens_per_second=("mean_tokens_per_second", "median"),
        median_model_tflops_per_gpu=("mean_model_tflops_per_gpu", "median"),
        median_mfu_pct=(
            "mean_mfu_pct_of_configured_peak", "median"
        ) if "mean_mfu_pct_of_configured_peak" in training_summary else
        ("training_job_id", lambda values: np.nan),
        median_gpu_utilization_pct=("mean_gpu_utilization_pct", "median"),
        observed_board_energy_kwh=("board_energy_estimate_kwh", "sum"),
    )
    .reset_index()
)
display(efficiency_summary)
efficiency_summary.to_csv(
    TABLE_OUTPUT_DIR / "06_efficiency_summary.csv", index=False
)
"""
        ),
        _md(
            """
## 7. Compute–performance view

The final checkpoint of each trajectory is joined first to the dedicated
per-readout attribution ledger. That estimate includes the relevant shared trunk
prefix plus the branch and is valid for comparing individual readouts, but it
**must never be summed as physical study compute**. The separate unique-physical
ledger counts each production job once. If attribution is unavailable, the plot
falls back to terminal-job allocation with an explicit shared-trunk warning.
Missing accounting is never imputed from requests.
"""
        ),
        _code(
            """
final_readouts = (
    readouts.sort_values(["trajectory_id", "actual_tokens"])
    .groupby("trajectory_id", observed=True)
    .tail(1)
    .copy()
)

attributed_candidates = [
    "logical_readout_attributed_productive_gpu_hours",
    "logical_attributed_gpu_hours_including_shared_trunk",
    "attributed_gpu_hours_including_shared_trunk",
]
physical_job_candidates = [
    "allocated_gpu_hours_including_retries",
    "allocated_gpu_hours", "gpu_hours", "total_gpu_hours",
    "elapsed_gpu_hours", "job_gpu_hours",
]
attributed_column = first_finite_column(
    readout_compute_attribution, attributed_candidates
)
physical_job_column = first_finite_column(compute_summary, physical_job_candidates)
if attributed_column is not None:
    attribution_unique = readout_compute_attribution[
        ["readout_id", attributed_column, "physical_aggregation_allowed"]
    ].drop_duplicates("readout_id")
    if len(attribution_unique) != len(readout_compute_attribution):
        raise RuntimeError("Duplicate readout compute-attribution rows")
    if attribution_unique["physical_aggregation_allowed"].fillna(True).any():
        raise RuntimeError("Logical readout attribution was marked physically additive")
    performance = final_readouts.merge(
        attribution_unique, on="readout_id", how="left", validate="one_to_one"
    )
    x_column = attributed_column
    x_label = "Logically attributed productive GPU-hours (shared trunk included)"
    compute_semantics = "logical_readout_attributed_productive_gpu_hours"
elif physical_job_column is not None:
    compute_unique = compute_summary[
        ["training_job_id", physical_job_column]
    ].drop_duplicates("training_job_id")
    performance = final_readouts.merge(
        compute_unique, on="training_job_id", how="left", validate="many_to_one"
    )
    x_column = physical_job_column
    x_label = (
        "Terminal-job allocated GPU-hours (shared trunk excluded for "
        "WSD branches)"
    )
    compute_semantics = "terminal_job_physical_gpu_hours"
else:
    fallback = training_summary[["training_job_id", "walltime_seconds"]].copy()
    fallback["released_wallclock_hours"] = (
        pd.to_numeric(fallback["walltime_seconds"], errors="coerce") / 3600.0
    )
    performance = final_readouts.merge(
        fallback[["training_job_id", "released_wallclock_hours"]],
        on="training_job_id", how="left", validate="many_to_one",
    )
    x_column = "released_wallclock_hours"
    x_label = (
        "Terminal-job wall-clock hours (GPU-hours and shared-trunk "
        "attribution unavailable)"
    )
    compute_semantics = "terminal_job_wallclock_fallback"

performance[x_column] = pd.to_numeric(performance[x_column], errors="coerce")
plot_performance = performance.dropna(
    subset=[x_column, "primary_metric_value"]
)
fig, ax = plt.subplots(figsize=(8.8, 5.4))
if plot_performance.empty:
    ax.text(
        0.5, 0.5,
        "No matched compute/performance rows in the released accounting snapshot",
        transform=ax.transAxes, ha="center", va="center",
    )
    ax.set_axis_off()
else:
    for ladder in ("A", "B", "C", "D"):
        sample = plot_performance.loc[plot_performance["ladder"].eq(ladder)]
        ax.scatter(
            sample[x_column], sample["primary_metric_value"],
            s=22 + 5 * np.sqrt(sample["experts"]),
            alpha=0.72, color=LADDER_COLORS[ladder], label=f"Ladder {ladder}",
        )
    if (plot_performance[x_column] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Final declared primary validation loss")
    ax.set_title(
        "Final validation versus "
        + (
            "logically attributed compute"
            if attributed_column
            else "terminal-job execution compute"
        )
    )
    ax.legend(frameon=False, ncol=2)
fig.savefig(FIGURE_DIR / "07_compute_performance.png")
plt.show()

performance["compute_axis_semantics"] = compute_semantics
display(
    performance.groupby("ladder", observed=True)
    .agg(
        trajectories=("trajectory_id", "nunique"),
        compute_rows=(x_column, "count"),
        median_compute=(x_column, "median"),
        median_final_loss=("primary_metric_value", "median"),
    )
    .reset_index()
)
if not unique_physical_study_compute.empty:
    display(Markdown(
        "**Unique physical study compute (additive):** each production training "
        "job counted once; logical readout attribution is excluded."
    ))
    display(unique_physical_study_compute)
    unique_physical_study_compute.to_csv(
        TABLE_OUTPUT_DIR / "07_unique_physical_study_compute.csv", index=False
    )
if not physical_compute_categories.empty:
    display(physical_compute_categories)
    physical_compute_categories.to_csv(
        TABLE_OUTPUT_DIR / "07_physical_compute_categories.csv", index=False
    )
performance.to_csv(
    TABLE_OUTPUT_DIR / "07_compute_performance.csv", index=False
)
"""
        ),
        _md(
            """
## 8. Optional routing-balance analysis

Routing is applicable only to sparse-MoE readouts. If the optional routing table
is present, the analysis uses final trajectory checkpoints and reports
coefficient of variation and dropped-assignment rate. If it is absent, the
notebook records the omission without failing the core analysis.
"""
        ),
        _code(
            """
routing_path = table_path("routing_layers")
if not routing_path.is_file() or parquet_row_count("routing_layers") == 0:
    display(Markdown(
        "**Routing table unavailable:** core A-D analysis remains valid, but "
        "routing-balance plots cannot be produced from this release."
    ))
    routing_summary = pd.DataFrame()
else:
    routing_required = [
        "readout_id", "ladder", "experts", "language_count",
        "coefficient_of_variation", "dropped_assignment_rate",
    ]
    routing = load_table("routing_layers", columns=routing_required)
    routing = routing.loc[routing["readout_id"].isin(final_readout_ids)].copy()
    for column in ("coefficient_of_variation", "dropped_assignment_rate"):
        routing[column] = pd.to_numeric(routing[column], errors="coerce")
    routing_summary = (
        routing.groupby(
            ["readout_id", "ladder", "experts", "language_count"],
            observed=True,
        )
        .agg(
            mean_routing_cv=("coefficient_of_variation", "mean"),
            max_routing_cv=("coefficient_of_variation", "max"),
            mean_dropped_assignment_rate=("dropped_assignment_rate", "mean"),
            max_dropped_assignment_rate=("dropped_assignment_rate", "max"),
            layer_language_rows=("readout_id", "size"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    routing_panels = [
        ("mean_routing_cv", "Mean routing coefficient of variation"),
        ("mean_dropped_assignment_rate", "Mean dropped-assignment rate"),
    ]
    for ax, (column, title) in zip(axes, routing_panels, strict=True):
        grouped = (
            routing_summary.groupby(["ladder", "experts"], observed=True)[column]
            .median()
            .reset_index()
        )
        for ladder in ("A", "B", "C", "D"):
            series = grouped.loc[grouped["ladder"].eq(ladder)].sort_values(
                "experts"
            )
            if not series.empty:
                ax.plot(
                    series["experts"], series[column], marker="o",
                    color=LADDER_COLORS[ladder], label=f"Ladder {ladder}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 16, 32], ["4", "16", "32"])
        ax.set_xlabel("Experts (E)")
        ax.set_title(title)
    axes[0].legend(frameon=False, ncol=2)
    fig.savefig(FIGURE_DIR / "08_routing_balance.png")
    plt.show()
    display(
        routing_summary.groupby(["ladder", "experts"], observed=True)
        .agg(
            readouts=("readout_id", "nunique"),
            median_mean_cv=("mean_routing_cv", "median"),
            median_drop_rate=("mean_dropped_assignment_rate", "median"),
        )
        .reset_index()
    )
    routing_summary.to_csv(
        TABLE_OUTPUT_DIR / "08_routing_summary.csv", index=False
    )
"""
        ),
        _md(
            """
## 9. Coverage, quality flags, and study limitations

The following table is deliberately part of the analytical result. It prevents
coverage limits from being separated from attractive plots during paper
writing.
"""
        ),
        _code(
            """
coverage_columns = [
    column for column in (
        "training_metric_coverage_fraction",
        "gpu_sample_count_coverage_fraction",
        "board_energy_coverage_fraction",
    )
    if column in training_summary.columns
]
coverage_summary_rows = []
for column in coverage_columns:
    values = pd.to_numeric(training_summary[column], errors="coerce")
    coverage_summary_rows.append({
        "measure": column,
        "jobs_with_measure": int(values.notna().sum()),
        "median_coverage": values.median(),
        "minimum_coverage": values.min(),
        "complete_jobs_ge_0_99": int(values.ge(0.99).sum()),
    })
coverage_summary = pd.DataFrame(coverage_summary_rows)
display(coverage_summary)
display(Markdown("**Validation scope by ladder and trained-language tier:**"))
display(validation_scope.sort_values(["ladder", "language_count"]))

if readout_training_bindings.empty:
    checkpoint_absent = None
    checkpoint_absent_by_ladder = {}
else:
    absent_mask = readout_training_bindings[
        "checkpoint_preservation_status"
    ].eq("absent_now")
    checkpoint_absent = int(absent_mask.sum())
    checkpoint_absent_by_ladder = (
        readout_training_bindings.loc[absent_mask]
        .groupby("ladder", observed=True)
        .size()
        .to_dict()
    )

limitations = pd.DataFrame([
    {
        "topic": "Independent replication",
        "status": "disclosed limitation",
        "interpretation": (
            "The primary 64-trajectory panel uses one configured model seed "
            "per cell. Checkpoints, languages, layers, and telemetry samples "
            "are repeated measures. Any seed-confirmation sidecar has a "
            "separate, narrower analysis scope."
        ),
    },
    {
        "topic": "Validation scope by tier",
        "status": "zero-shot outcomes available only for K1",
        "interpretation": (
            "Each K1 readout has 1 trained plus 199 zero-shot languages. "
            "K12/K50/K200 evaluate trained languages only, so cross-tier "
            "zero-shot comparisons are unavailable."
        ),
    },
    {
        "topic": "Checkpoint preservation",
        "status": (
            f"{checkpoint_absent} readout checkpoint directories absent now"
            if checkpoint_absent is not None
            else "binding table unavailable"
        ),
        "interpretation": (
            f"Current missing directories by ladder: {checkpoint_absent_by_ladder}. "
            "Released evaluation results and hashes remain analyzable, but new "
            "checkpoint-level evaluation requires restoration or retraining."
        ),
    },
    {
        "topic": "Matched ladder contrasts",
        "status": "descriptive only",
        "interpretation": (
            "A/B and C/D deltas match E, K, and token readout, but do not "
            "supply independent-seed uncertainty or causal identification."
        ),
    },
    {
        "topic": "Training lineage",
        "status": "preserved",
        "interpretation": (
            "WSD trunks and branches remain distinct jobs/segments; shared "
            "trunk work must not be counted once per downstream readout."
        ),
    },
    {
        "topic": "Compute accounting",
        "status": (
            "logical shared-trunk attribution available"
            if attributed_column is not None
            else "terminal-job accounting only"
            if physical_job_column is not None
            else "missing or unusable; terminal wall-clock fallback shown"
        ),
        "interpretation": (
            "Scheduler values are historical allocation records, not direct "
            "measurements of useful floating-point work. Terminal-job values "
            "exclude shared WSD trunks unless an attributed field is present."
        ),
    },
    {
        "topic": "MFU",
        "status": "available" if mfu_column else "not computable",
        "interpretation": (
            "MFU requires a declared hardware peak denominator. Logged model "
            "TFLOP/s/GPU is shown when the denominator is absent."
        ),
    },
    {
        "topic": "Energy",
        "status": "board-only observed-interval estimate",
        "interpretation": (
            "Power integration excludes telemetry gaps and all host, network, "
            "storage, cooling, and facility overhead; it is not carbon impact."
        ),
    },
    {
        "topic": "Optional routing",
        "status": (
            "analyzed" if not routing_summary.empty else "not released"
        ),
        "interpretation": (
            "Routing rows are layer/language repeated measures and are "
            "summarized descriptively at final trajectory checkpoints."
        ),
    },
])
display(limitations)

if not qa_issues.empty:
    qa_public = qa_issues.copy()
    display(Markdown(
        f"**Released QA issue rows:** {len(qa_public):,}. "
        "Review severity and verdict columns before confirmatory use."
    ))
    display(qa_public.head(50))
else:
    display(Markdown(
        "**Released QA issue table:** absent or empty. This does not replace "
        "the explicit coverage and design limitations above."
    ))

coverage_summary.to_csv(
    TABLE_OUTPUT_DIR / "09_coverage_summary.csv", index=False
)
limitations.to_csv(TABLE_OUTPUT_DIR / "09_limitations.csv", index=False)
"""
        ),
        _md(
            """
## 10. Reproducible output index

All generated figures and compact tables live under
`stage1_abcd_analysis_outputs/`. They are derived products, not additional raw
evidence. The central release Parquet tables remain the source of truth.
"""
        ),
        _code(
            """
output_rows = []
for path in sorted(ANALYSIS_DIR.rglob("*")):
    if path.is_file():
        output_rows.append({
            "path": str(path.relative_to(ANALYSIS_DIR)),
            "bytes": path.stat().st_size,
        })
output_index = pd.DataFrame(output_rows)
display(output_index)

analysis_summary = pd.DataFrame([
    {"measure": "trajectories", "value": len(trajectories)},
    {"measure": "readouts", "value": len(readouts)},
    {"measure": "validation_language_rows", "value": len(validation_language)},
    {"measure": "validation_scope_rows", "value": len(validation_scope)},
    {
        "measure": "checkpoint_directories_absent_now",
        "value": checkpoint_absent if checkpoint_absent is not None else -1,
    },
    {"measure": "training_metric_rows", "value": event_rows["training_metrics"]},
    {"measure": "gpu_sample_rows", "value": event_rows["gpu_samples"]},
    {"measure": "matched_A_B_readouts", "value": len(ab)},
    {"measure": "matched_C_D_readouts", "value": len(cd)},
    {
        "measure": "routing_rows_available",
        "value": parquet_row_count("routing_layers") or 0,
    },
])
analysis_summary.to_csv(
    TABLE_OUTPUT_DIR / "10_analysis_summary.csv", index=False
)
display(analysis_summary)

print("Figures:", FIGURE_DIR)
print("Tables:", TABLE_OUTPUT_DIR)
print("Analysis boundary:", ANALYSIS_BOUNDARY)
print("Next action:", ANALYSIS_NEXT_ACTION)
"""
        ),
    ]


def build_notebook(release_dir: Path, output: Path) -> None:
    notebook = {
        "cells": _cells(release_dir),
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (study)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "stage1_release_consumer": {
            "schema": "stage1_abcd_analysis_notebook/v1",
            "input_contract": "release Parquet tables only",
            "release_dir_hint": str(release_dir.resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


_KERNEL_EXECUTOR = r"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty
import sys

from jupyter_client import KernelManager


def _cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _append_output(outputs, message_type, content):
    if message_type == "stream":
        name = content.get("name", "stdout")
        text = content.get("text", "")
        if (
            outputs
            and outputs[-1].get("output_type") == "stream"
            and outputs[-1].get("name") == name
        ):
            outputs[-1]["text"] = str(outputs[-1].get("text", "")) + str(text)
        else:
            outputs.append({"output_type": "stream", "name": name, "text": text})
    elif message_type in {"display_data", "execute_result"}:
        output = {
            "output_type": message_type,
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
        }
        if message_type == "execute_result":
            output["execution_count"] = content.get("execution_count")
        outputs.append(output)
    elif message_type == "error":
        outputs.append({
            "output_type": "error",
            "ename": content.get("ename", "Error"),
            "evalue": content.get("evalue", ""),
            "traceback": content.get("traceback", []),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("working_directory", type=Path)
    parser.add_argument("timeout", type=int)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(args.working_directory.resolve()))
    client = manager.client()
    client.start_channels()
    failure = None
    try:
        client.wait_for_ready(timeout=min(args.timeout, 120))
        for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = _cell_source(cell)
            if not source.strip():
                continue
            message_id = client.execute(
                source,
                allow_stdin=False,
                stop_on_error=True,
                store_history=True,
            )
            outputs = []
            execution_count = None
            try:
                while True:
                    message = client.get_iopub_msg(timeout=args.timeout)
                    if message.get("parent_header", {}).get("msg_id") != message_id:
                        continue
                    message_type = message.get("header", {}).get("msg_type")
                    content = message.get("content", {})
                    if message_type == "execute_input":
                        execution_count = content.get("execution_count")
                    elif message_type == "clear_output":
                        outputs = []
                    elif message_type in {"stream", "display_data", "execute_result", "error"}:
                        _append_output(outputs, message_type, content)
                        if message_type == "error":
                            error_name = content.get("ename", "Error")
                            error_value = content.get("evalue", "")
                            failure = RuntimeError(
                                f"Notebook cell {cell_index} failed: "
                                f"{error_name}: {error_value}"
                            )
                    elif message_type == "status" and content.get("execution_state") == "idle":
                        break
            except Empty:
                failure = TimeoutError(
                    f"Notebook cell {cell_index} exceeded {args.timeout} seconds"
                )
                outputs.append({
                    "output_type": "error",
                    "ename": "CellTimeoutError",
                    "evalue": str(failure),
                    "traceback": [],
                })
            cell["execution_count"] = execution_count
            cell["outputs"] = outputs
            if failure is not None:
                break
    except BaseException as exc:
        if failure is None:
            failure = exc
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    if failure is not None:
        raise failure


if __name__ == "__main__":
    sys.exit(main())
"""


def execute_notebook(
    *,
    release_dir: Path,
    output: Path,
    conda_env: str,
    timeout_seconds: int,
) -> None:
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("Cannot execute notebook: conda is not on PATH")
    with tempfile.TemporaryDirectory(prefix="stage1-ad-notebook-") as temporary:
        runner = Path(temporary) / "execute_notebook.py"
        runner.write_text(_KERNEL_EXECUTOR, encoding="utf-8")
        command = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            str(runner),
            str(output.resolve()),
            str(release_dir.resolve()),
            str(timeout_seconds),
        ]
        subprocess.run(command, cwd=release_dir, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="Canonical release directory (default: result_analysis/stage1_abcd_v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Notebook path (default: <release-dir>/stage1_abcd_analysis.ipynb)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated notebook with Jupyter in the conda environment",
    )
    parser.add_argument(
        "--conda-env",
        default="study",
        help="Conda environment used by --execute (default: study)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell execution timeout in seconds (default: 3600)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release_dir = args.release_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else release_dir / "stage1_abcd_analysis.ipynb"
    )
    build_notebook(release_dir, output)
    print(f"Wrote {output}")
    if args.execute:
        if not (release_dir / "tables").is_dir():
            raise RuntimeError(
                f"Cannot execute before the release exists: {release_dir / 'tables'}"
            )
        execute_notebook(
            release_dir=release_dir,
            output=output,
            conda_env=args.conda_env,
            timeout_seconds=args.timeout,
        )
        print(f"Executed {output} in conda environment {args.conda_env!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
