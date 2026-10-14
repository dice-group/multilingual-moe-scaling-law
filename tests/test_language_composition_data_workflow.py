from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from multilingual_megatron.experiments.scaling.language_composition.data import (
    data_artifacts,
    data_plan,
    data_workflow,
)
from multilingual_megatron.experiments.scaling.k12_migration import workflow as k12_migration_workflow
from multilingual_megatron.experiments.scaling.k12_migration import (
    release as migration_release,
)
from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_training import (
    _manifest_blend_per_split,
)


def _split(languages: list[str]) -> dict[str, object]:
    prefixes = [f"/sealed/{language}" for language in languages]
    return {
        "prefixes": prefixes,
        "prefixes_by_language": {
            language: [prefix] for language, prefix in zip(languages, prefixes, strict=True)
        },
        "prefix_artifacts_by_language": {
            language: [{"prefix": prefix, "tokens_including_eod": index + 10}]
            for index, (language, prefix) in enumerate(
                zip(languages, prefixes, strict=True)
            )
        },
        "weights": [1.0 / len(languages)] * len(languages),
        "documents_per_language": {
            language: index + 1 for index, language in enumerate(languages)
        },
        "tokens_per_language": {
            language: index + 10 for index, language in enumerate(languages)
        },
        "total_documents": sum(range(1, len(languages) + 1)),
        "total_tokens_including_eod": sum(range(10, 10 + len(languages))),
    }


class LanguageCompositionDataWorkflowTest(unittest.TestCase):
    def test_release_refreshes_changed_language_audit_from_full_plan(self) -> None:
        payload = {
            "train": {"selected_tokens": 80},
            "exposure_pass_audit": {
                "K200": {"stale": True},
            },
        }
        plan = {
            "tiers": {
                "K200": {"target_exposure_tokens": {"x_Latn": 40}},
            },
            "composition_arms": {
                "compact": {
                    "target_exposure_tokens": {"x_Latn": 100},
                }
            },
        }
        migration_release._refresh_exposure_pass_audit(
            payload, plan=plan, language="x_Latn"
        )
        self.assertEqual(
            set(payload["exposure_pass_audit"]),
            {"K200", "composition:compact"},
        )
        self.assertEqual(
            payload["exposure_pass_audit"]["composition:compact"],
            {
                "target_exposure_tokens": 100,
                "physical_train_tokens": 80,
                "effective_passes": 1.25,
                "full_passes": 1,
                "partial_pass_tokens": 20,
                "reuses_physical_pool": True,
            },
        )

    def test_adopted_provenance_paths_are_rebased_to_successor(self) -> None:
        plan = {
            "provenance": {
                "calibration_path": "/predecessor/calibration.json",
                "calibration_sha256": "a" * 64,
                "source_manifest_path": "/predecessor/source.json",
                "source_manifest_file_sha256": "b" * 64,
                "tokenizer_model_path": "/predecessor/tokenizer.model",
                "tokenizer_model_sha256": "c" * 64,
            }
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_plan._rebase_adopted_provenance_paths(
                plan,
                calibration_path=root / "corpus/token_calibration.json",
                source_manifest_path=root
                / "corpus/source_lock/source_materialization_manifest.json",
                tokenizer_model_path=root
                / "corpus/tokenizer/sp_unigram_64k/tokenizer.model",
            )
            provenance = plan["provenance"]
            self.assertEqual(
                provenance["calibration_path"],
                str((root / "corpus/token_calibration.json").resolve()),
            )
            self.assertEqual(
                provenance["source_manifest_path"],
                str(
                    (
                        root
                        / "corpus/source_lock/source_materialization_manifest.json"
                    ).resolve()
                ),
            )
            self.assertEqual(
                provenance["tokenizer_model_path"],
                str(
                    (
                        root / "corpus/tokenizer/sp_unigram_64k/tokenizer.model"
                    ).resolve()
                ),
            )
        self.assertEqual(provenance["calibration_sha256"], "a" * 64)
        self.assertEqual(provenance["source_manifest_file_sha256"], "b" * 64)
        self.assertEqual(provenance["tokenizer_model_sha256"], "c" * 64)

    def test_acquisition_margin_preserves_target_and_adds_source_capacity(self) -> None:
        row = {
            "estimated_train_rehydrated_tokens": 10_000,
            "estimated_train_documents": 1_000,
            "requested_max_tier_tokens": 1_000,
            "physical_target_tokens": 1_000,
            "calibration": {
                "language": "x_Latn",
                "available_utf8_bytes": 10_000,
                "available_documents": 1_000,
                "calibration_utf8_bytes": 1_000,
                "calibration_tokens": 1_000,
                "calibration_documents": 100,
                "calibration_rehydrated_tokens": 1_000,
                "validation_fraction": 0.3,
            },
        }
        result = data_plan._physical_with_requested(
            row,
            5_000,
            source_document_margin=data_plan.ACQUISITION_SOURCE_DOCUMENT_MARGIN,
        )
        self.assertEqual(result["physical_target_tokens"], 5_000)
        self.assertEqual(result["planned_source_documents"], 550)
        self.assertEqual(result["acquisition_source_document_margin"], 1.10)

    def test_migration_engine_uses_manifest_owned_dynamic_counts(self) -> None:
        manifest = {
            "contract": {
                "acquisition_languages": ["arb_Arab", "dan_Latn"],
                "finalizer_workers": 16,
            }
        }
        self.assertEqual(
            k12_migration_workflow._manifest_acquisition_languages(manifest),
            ("arb_Arab", "dan_Latn"),
        )
        self.assertEqual(
            k12_migration_workflow._manifest_finalizer_workers(manifest), 16
        )

    def test_cpu_dag_has_exact_dynamic_array_cardinalities(self) -> None:
        jobs = data_workflow._jobs()
        by_id = {row["job_id"]: row for row in jobs}
        self.assertEqual(by_id["acquisition_workers"]["array"], "0-7%8")
        self.assertEqual(by_id["acquisition_workers"]["array_count"], 8)
        self.assertEqual(by_id["finalizer_workers"]["array"], "0-15%16")
        self.assertEqual(by_id["finalizer_workers"]["array_count"], 16)
        self.assertEqual(sum(int(row["array_count"]) for row in jobs), 309)

    def test_manifest_fixed_inputs_cover_the_migration_package(self) -> None:
        runtime = (data_workflow.REPO_ROOT / data_workflow.DEFAULT_RUNTIME).resolve()
        fixed = set(data_workflow._fixed_additions(runtime))
        migration_package = (
            data_workflow.REPO_ROOT
            / "src/multilingual_megatron/experiments/scaling/k12_migration"
        )
        expected = set(migration_package.glob("*.py"))
        self.assertTrue(expected)
        self.assertTrue(expected <= fixed)
        audit_package = (
            data_workflow.REPO_ROOT
            / "src/multilingual_megatron/pipeline/stages/audit/fineweb_release"
        )
        expected_audit = set(audit_package.glob("*.py"))
        self.assertTrue(expected_audit)
        self.assertTrue(expected_audit <= fixed)
        self.assertIn(
            (
                data_workflow.REPO_ROOT
                / "src/multilingual_megatron/pipeline/stages/audit/fineweb_v2_release.py"
            ).resolve(),
            fixed,
        )
        self.assertIn(
            (data_workflow.REPO_ROOT / data_workflow.DATA_AUDIT_SPEC).resolve(),
            fixed,
        )
        self.assertIn(
            (
                data_workflow.REPO_ROOT
                / "src/multilingual_megatron/experiments/scaling/ab_v2/artifact_data.py"
            ).resolve(),
            fixed,
        )
        self.assertIn(
            (
                data_workflow.REPO_ROOT
                / "src/multilingual_megatron/experiments/scaling/k12_migration/data_smoke.py"
            ).resolve(),
            fixed,
        )
        self.assertIn(
            (
                data_workflow.REPO_ROOT
                / "external/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets/indexed_dataset.py"
            ).resolve(),
            fixed,
        )

    def test_split_audit_uses_node_local_tmpdir(self) -> None:
        runtime_path = (
            data_workflow.REPO_ROOT / data_workflow.DEFAULT_RUNTIME
        ).resolve()
        runtime = data_workflow._load_inherited_object(runtime_path)
        self.assertEqual(runtime["data"]["runtime_env"]["TMPDIR"], "/dev/shm")
        self.assertEqual(
            k12_migration_workflow._job_environment({"runtime": runtime})["TMPDIR"],
            "/dev/shm",
        )

    def test_cohorts_are_cwd_independent_and_exact(self) -> None:
        previous = Path.cwd()
        with TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                payload = data_artifacts.build_cohorts()
            finally:
                os.chdir(previous)
        cohorts = payload["cohorts"]
        self.assertEqual(cohorts["k196"]["language_count"], 196)
        self.assertEqual(cohorts["unseen_177"]["language_count"], 177)
        self.assertEqual(
            {name: cohorts[name]["language_count"] for name in (
                "reference_only",
                "compact_only",
                "diverse_only",
                "reference_compact",
                "reference_diverse",
                "compact_diverse",
                "all_three",
            )},
            {
                "reference_only": 2,
                "compact_only": 4,
                "diverse_only": 3,
                "reference_compact": 1,
                "reference_diverse": 2,
                "compact_diverse": 0,
                "all_three": 7,
            },
        )

    def test_arm_and_validation_manifests_filter_the_materialized_split(self) -> None:
        languages = ["a_Latn", "b_Latn", "c_Latn"]
        train = {
            "languages": languages,
            "prefixes": [f"/sealed/{language}" for language in languages],
            "prefixes_by_language": {
                language: [f"/sealed/{language}"] for language in languages
            },
            "weights": [0.2, 0.3, 0.5],
            "documents_per_language": {language: 1 for language in languages},
            "tokens_per_language": {},
            "splits": {"train": _split(languages)},
            "lineage": {"sampling_plan_payload_sha256": "p" * 64},
            "manifest_payload_sha256": "old",
        }
        arm = data_artifacts._arm_manifest(
            train,
            arm="compact",
            arm_plan={
                "languages": languages[:2],
                "probabilities": {"a_Latn": 0.4, "b_Latn": 0.6},
                "target_exposure_tokens": {"a_Latn": 40, "b_Latn": 60},
                "target_train_tokens": 100,
                "alpha": 0.3,
                "language_set_sha256": "s" * 64,
            },
        )
        self.assertEqual(arm["languages"], languages[:2])
        self.assertEqual(len(arm["prefixes"]), 2)
        self.assertEqual(len(arm["splits"]["train"]["prefixes"]), 2)
        self.assertEqual(arm["weights"], [0.4, 0.6])
        self.assertEqual(arm["splits"]["train"]["weights"], [0.4, 0.6])
        self.assertEqual(
            _manifest_blend_per_split(
                arm, include_validation=False, include_test=False
            )[0][1],
            [0.4, 0.6],
        )

        validation = {
            **train,
            "prefixes": [],
            "prefixes_by_language": {},
            "weights": [],
            "documents_per_language": {},
            "splits": {"validation": _split(languages)},
        }
        filtered = data_artifacts._validation_manifest(validation, languages[:2])
        self.assertEqual(filtered["prefixes"], [])
        self.assertEqual(filtered["prefixes_by_language"], {})
        self.assertEqual(len(filtered["splits"]["validation"]["prefixes"]), 2)
        self.assertEqual(
            set(filtered["splits"]["validation"]["prefix_artifacts_by_language"]),
            set(languages[:2]),
        )


if __name__ == "__main__":
    unittest.main()
