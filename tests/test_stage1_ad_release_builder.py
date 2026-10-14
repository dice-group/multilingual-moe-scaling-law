from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multilingual_megatron.experiments.scaling.stage1_release.builder import (  # noqa: E402
    _assign_qa_issue_ids,
    _canonical_qa_evidence_cell,
    _jointmoe_learning_rate_qa,
    _jointmoe_scope_disclosures,
    _validation_scope_rows,
)
from multilingual_megatron.experiments.scaling.stage1_release import (  # noqa: E402
    builder as release_builder,
)
from multilingual_megatron.experiments.scaling.stage1_release.contracts import (  # noqa: E402
    Stage1ReleaseError,
    load_release_build_spec,
)


class Stage1ADReleaseBuilderTest(unittest.TestCase):
    def test_compatibility_facade_preserves_cli_contract(self) -> None:
        args = release_builder._parser().parse_args(
            [
                "--repo-root",
                "/tmp/stage1-release-root",
                "--output",
                "release",
                "--slurm-missing-policy",
                "error",
            ]
        )

        self.assertEqual(args.repo_root, Path("/tmp/stage1-release-root"))
        self.assertEqual(args.output, Path("release"))
        self.assertEqual(args.slurm_missing_policy, "error")
        self.assertTrue(callable(release_builder.build_release))

    def test_jointmoe_release_build_spec_is_strict_and_versioned(self) -> None:
        path = (
            REPO_ROOT
            / "configs/scaling_law/stage1_jointmoe_analysis_v1/release_spec.json"
        )
        spec = load_release_build_spec(path)
        self.assertEqual(spec.release_name, "stage1_abcd_jointmoe_x1_v1")
        self.assertEqual(spec.cd_source, "direct_final_summaries")
        self.assertEqual(set(spec.primary_root_names), {"AB", "C", "D"})
        self.assertIsNotNone(spec.seed_confirmation_summary)
        self.assertEqual(spec.expected_validation_candidate_count, 400)
        self.assertEqual(spec.expected_duplicate_candidate_count, 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.json"
            invalid.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage1ReleaseError, "schema or fields"):
                load_release_build_spec(invalid)

    def test_jointmoe_disclosures_bound_lr_and_seed_evidence_scope(self) -> None:
        disclosures = {
            row["disclosure_id"]: row for row in _jointmoe_scope_disclosures()
        }

        self.assertEqual(
            set(disclosures),
            {"learning_rate_evidence_scope", "seed_confirmation_scope"},
        )
        lr_text = disclosures["learning_rate_evidence_scope"]
        self.assertIn("B-ladder", lr_text["statement"])
        self.assertIn("inconclusive", lr_text["statement"])
        self.assertIn("not as a per-cell", lr_text["analysis_consequence"])
        seed_text = disclosures["seed_confirmation_scope"]
        self.assertIn("two additional independent training seeds", seed_text["statement"])
        self.assertIn("do not pool", seed_text["analysis_consequence"])

    def test_jointmoe_learning_rate_contract_is_exact_and_k_invariant(self) -> None:
        prescription = {
            "intercept": 8.39,
            "active_parameter_coefficient": -0.81,
            "expert_coefficient": -0.25,
            "constant_across_K": True,
            "parameter_count_basis": (
                "paper-style active non-embedding parameter count N_act_nonembed"
            ),
            "formula": "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))",
        }
        studies = [
            {
                "learning_rate_mode": "prescribed_per_model_configuration",
                "learning_rate_prescription_json": json.dumps(
                    prescription, sort_keys=True
                ),
            }
            for _ in "ABCD"
        ]
        active_parameters = {
            "A": 27_262_976,
            "B": 92_012_544,
            "C": 218_103_808,
            "D": 425_984_000,
        }
        models = []
        jobs = []
        for ladder, nonembedding in active_parameters.items():
            for experts in (1, 4, 16, 32):
                models.append(
                    {
                        "ladder": ladder,
                        "experts": experts,
                        "declared_active_nonembedding_parameters": nonembedding,
                    }
                )
                learning_rate = math.exp(
                    8.39 - 0.81 * math.log(nonembedding) - 0.25 * math.log(experts)
                )
                for language_count in (1, 200):
                    jobs.append(
                        {
                            "training_job_id": (
                                f"{ladder}:E{experts}:K{language_count}"
                            ),
                            "trajectory_id": (
                                f"{ladder}:E{experts}:K{language_count}"
                            ),
                            "ladder": ladder,
                            "experts": experts,
                            "language_count": language_count,
                            "peak_learning_rate": learning_rate,
                        }
                    )
        metrics = [
            {
                "training_job_id": job["training_job_id"],
                "learning_rate": job["peak_learning_rate"],
            }
            for job in jobs
        ]

        qa = _jointmoe_learning_rate_qa(
            studies=studies,
            model_configs=models,
            training_jobs=jobs,
            training_metrics=metrics,
        )
        self.assertEqual(qa["status"], "pass")
        self.assertEqual(len(qa["observed"]), 16)

        jobs[0] = {**jobs[0], "peak_learning_rate": jobs[0]["peak_learning_rate"] / 2}
        with self.assertRaisesRegex(Stage1ReleaseError, "not constant across K/jobs"):
            _jointmoe_learning_rate_qa(
                studies=studies,
                model_configs=models,
                training_jobs=jobs,
                training_metrics=metrics,
            )

    def test_qa_evidence_has_one_stable_json_string_type(self) -> None:
        values = [
            64,
            "pass",
            {"A": 80, "B": 96},
            ["trained", "zero_shot"],
            '{"already":"json"}',
        ]

        cells = [_canonical_qa_evidence_cell(value) for value in values]

        self.assertTrue(all(isinstance(value, str) for value in cells))
        self.assertEqual(
            [json.loads(value) for value in cells], values[:-1] + [{"already": "json"}]
        )

    def test_null_qa_evidence_remains_null(self) -> None:
        self.assertIsNone(_canonical_qa_evidence_cell(None))

    def test_validation_scope_makes_zero_shot_boundary_explicit(self) -> None:
        readouts = []
        observations = []
        for ladder in "ABCD":
            for language_count in (1, 12, 50, 200):
                readout_id = f"{ladder}-K{language_count}"
                readouts.append(
                    {
                        "readout_id": readout_id,
                        "ladder": ladder,
                        "language_count": language_count,
                    }
                )
                validation_count = 200 if language_count == 1 else language_count
                for index in range(validation_count):
                    observations.append(
                        {
                            "readout_id": readout_id,
                            "language": f"lang-{index:03d}",
                            "metric_role": (
                                "trained" if index < language_count else "zero_shot"
                            ),
                        }
                    )

        rows = _validation_scope_rows(readouts, observations)
        by_id = {row["validation_scope_id"]: row for row in rows}

        self.assertEqual(len(rows), 16)
        self.assertEqual(by_id["A:K1"]["trained_languages_per_readout"], 1)
        self.assertEqual(by_id["A:K1"]["zero_shot_languages_per_readout"], 199)
        self.assertEqual(by_id["D:K200"]["zero_shot_languages_per_readout"], 0)
        self.assertEqual(
            by_id["D:K200"]["scope_status"],
            "trained_tier_only_no_zero_shot",
        )

    def test_qa_issue_ids_are_stable_unique_and_ignore_build_time(self) -> None:
        first = _assign_qa_issue_ids(
            [
                {"check_id": "same", "status": "pass", "checked_at_utc": "t1"},
                {"check_id": "same", "status": "pass", "checked_at_utc": "t1"},
            ]
        )
        rebuilt = _assign_qa_issue_ids(
            [
                {"check_id": "same", "status": "pass", "checked_at_utc": "t2"},
                {"check_id": "same", "status": "pass", "checked_at_utc": "t2"},
            ]
        )

        self.assertEqual(
            [row["qa_issue_id"] for row in first],
            [row["qa_issue_id"] for row in rebuilt],
        )
        self.assertEqual(len({row["qa_issue_id"] for row in first}), 2)


if __name__ == "__main__":
    unittest.main()
