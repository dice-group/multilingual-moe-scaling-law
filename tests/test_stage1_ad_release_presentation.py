from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_release.presentation import build_neurips_checklist
from multilingual_megatron.experiments.scaling.stage1_release.presentation import build_table_catalog
from multilingual_megatron.experiments.scaling.stage1_release.presentation import render_release_readme
from multilingual_megatron.experiments.scaling.stage1_release.presentation import write_excel_workbook


FINAL_TABLE_PRIMARY_KEYS = {
    "launch_provenance": ["execution_id"],
    "execution_inputs": ["execution_id", "declared_path"],
    "hardware_contract": ["hardware_contract_id"],
    "languages": ["language"],
    "language_tier_membership": ["language_count", "language"],
    "language_metadata_sources": ["language_metadata_source_id"],
    "validation_cohorts": ["validation_cohort_id"],
    "readout_compute_attribution": ["readout_id"],
    "unique_physical_job_compute": ["physical_compute_unit_id"],
    "unique_physical_study_compute": ["physical_compute_scope_id"],
    "physical_compute_categories": ["workload_category"],
}

FINAL_PARQUET_ONLY_PRIMARY_KEYS = {
    "production_artifact_verifications": [
        "training_job_id",
        "artifact_type",
        "path",
    ],
    "readout_training_bindings": ["readout_id"],
}


def _release_metadata() -> dict[str, object]:
    return {
        "release_name": "stage1_abcd_test",
        "schema": "stage1/test",
        "status": "complete",
        "built_at_utc": "2026-07-30T00:00:00Z",
        "confirmatory_analysis_status": (
            "pending_frozen_cd_m1_score_then_locked_ad_refit"
        ),
        "independent_trajectories": 64,
        "configured_seed_replicates_per_trajectory": 1,
    }


def _table_counts() -> dict[str, int]:
    return {
        "trajectories": 64,
        "readouts": 400,
        "launch_provenance": 3,
        "languages": 200,
        "validation_cohorts": 7,
        "validation_language": 46_200,
        "data_exposure": 26_300,
        "compute_summary": 464,
        "readout_compute_attribution": 400,
        "unique_physical_job_compute": 464,
        "unique_physical_study_compute": 1,
        "physical_compute_categories": 8,
    }


class Stage1ADReleasePresentationTest(unittest.TestCase):
    def test_catalog_declares_every_final_analysis_table(self) -> None:
        catalog = {row["table_name"]: row for row in build_table_catalog()}

        self.assertTrue(FINAL_TABLE_PRIMARY_KEYS.keys() <= catalog.keys())
        for table_name, primary_key in FINAL_TABLE_PRIMARY_KEYS.items():
            with self.subTest(table=table_name):
                row = catalog[table_name]
                self.assertEqual(row["primary_key"], primary_key)
                self.assertNotEqual(row["category"], "uncatalogued")
                self.assertTrue(row["excel_default"])

        self.assertTrue(FINAL_PARQUET_ONLY_PRIMARY_KEYS.keys() <= catalog.keys())
        for table_name, primary_key in FINAL_PARQUET_ONLY_PRIMARY_KEYS.items():
            with self.subTest(table=table_name):
                row = catalog[table_name]
                self.assertEqual(row["primary_key"], primary_key)
                self.assertNotEqual(row["category"], "uncatalogued")
                self.assertFalse(row["excel_default"])

        attribution = catalog["readout_compute_attribution"]
        self.assertIn("never sum", attribution["caveats"])
        physical = catalog["unique_physical_study_compute"]
        self.assertIn("additive", physical["caveats"])

    def test_readme_and_checklist_preserve_scientific_scope(self) -> None:
        metadata = _release_metadata()
        counts = _table_counts()
        catalog = build_table_catalog()
        checklist = build_neurips_checklist(metadata, counts)
        by_id = {row["checklist_id"]: row for row in checklist}

        self.assertEqual(
            by_id["analysis_status"]["release_support"],
            metadata["confirmatory_analysis_status"],
        )
        self.assertEqual(by_id["compute"]["release_support"], "documented")
        self.assertEqual(by_id["datasets"]["release_support"], "documented")

        readme = render_release_readme(
            metadata,
            counts,
            catalog=catalog,
            checklist=checklist,
        )
        self.assertIn("analysis-ready data release", readme)
        self.assertIn("does not mean confirmatory-complete", readme)
        self.assertIn("64 full training trajectories", readme)
        self.assertIn("one configured seed", readme)
        self.assertIn("Adjusted loss may be negative", readme)
        self.assertIn("do not exponentiate", readme)
        self.assertIn("deliberately non-additive", readme)
        self.assertIn("source of truth", readme)
        self.assertIn("2026-07-30T00:00:00Z", readme)
        self.assertIn(
            "pending_frozen_cd_m1_score_then_locked_ad_refit",
            readme,
        )
        self.assertIn("training-horizon comparisons.", readme)
        self.assertNotIn("{validation_rows}", readme)
        self.assertIn("copied under `code/`", readme)
        self.assertIn(
            "`tables/training_metrics.parquet`",
            readme,
        )

    def test_workbook_omits_whole_table_instead_of_truncating(self) -> None:
        launch = pd.DataFrame(
            [
                {
                    "execution_id": "stage1_ab_production",
                    "source_family": "AB",
                    "optional_note": "",
                }
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "training_job_id": "job-1",
                    "segment_id": "segment-1",
                    "iteration": iteration,
                }
                for iteration in (1, 2)
            ]
        )
        tables = {
            "launch_provenance": launch,
            "training_metrics": metrics,
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "release.xlsx"
            receipt = write_excel_workbook(
                path,
                tables,
                _release_metadata(),
                include_tables=list(tables),
                max_data_rows_per_sheet=1,
            )

            coverage = {row["table_name"]: row for row in receipt["coverage"]}
            self.assertEqual(receipt["canonical_table_count"], 2)
            self.assertEqual(receipt["excel_coverage_record_count"], 2)
            self.assertTrue(receipt["no_silent_row_truncation"])
            self.assertTrue(receipt["no_silent_column_truncation"])
            self.assertEqual(
                coverage["launch_provenance"]["status"],
                "included_complete",
            )
            self.assertTrue(
                coverage["launch_provenance"]["complete_row_and_column_coverage"]
            )
            transformations = json.loads(
                coverage["launch_provenance"]["transformations"]
            )
            self.assertEqual(transformations["empty_string_to_blank"], 1)
            self.assertEqual(
                coverage["training_metrics"]["status"],
                "omitted_excel_row_limit",
            )
            self.assertEqual(
                coverage["training_metrics"]["canonical_parquet_path"],
                "tables/training_metrics.parquet",
            )
            self.assertEqual(coverage["training_metrics"]["excel_rows"], 0)
            self.assertIn("LaunchProvenance", receipt["sheets"])
            self.assertNotIn("TrainingMetrics", receipt["sheets"])

    def test_workbook_detects_overlong_inferred_string_cells(self) -> None:
        tables = {
            "model_configs": pd.DataFrame(
                {
                    "model_config_id": pd.Series(["model-1"], dtype="str"),
                    "resolved_config_json": pd.Series(["x" * 32_768], dtype="str"),
                }
            )
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            receipt = write_excel_workbook(
                Path(raw_directory) / "release.xlsx",
                tables,
                _release_metadata(),
                include_tables=["model_configs"],
            )

        coverage = receipt["coverage"][0]
        self.assertEqual(coverage["status"], "omitted_excel_cell_limit")
        self.assertFalse(coverage["complete_row_and_column_coverage"])
        self.assertTrue(coverage["omitted_from_excel"])
        self.assertEqual(coverage["excel_rows"], 0)
        self.assertNotIn("ModelConfigs", receipt["sheets"])


if __name__ == "__main__":
    unittest.main()
