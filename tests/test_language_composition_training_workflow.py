from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from multilingual_megatron.experiments.scaling.ab_v2.common import (
    Paths,
    WorkflowError,
    _atomic_json,
    _file_sha256,
    _payload_sha256,
)
from multilingual_megatron.experiments.scaling.ab_v2.training import (
    _resolved_profile_data_manifest,
)
from multilingual_megatron.experiments.scaling.language_composition import study_contracts as contracts
from multilingual_megatron.experiments.scaling.language_composition.training import (
    production_smoke_evidence,
    training_actions,
    training_contracts,
    training_jobs,
)
from multilingual_megatron.experiments.scaling.language_composition.study_contracts import (
    EXPERTS,
    HORIZONS,
    REPO_ROOT,
    SEEDS,
    SMOKE_GLOBAL_BATCH_SIZE,
    SMOKE_MAX_STEP_TIME_CV_PCT,
    SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS,
    SMOKE_TOKENS,
)
from multilingual_megatron.pipeline.stages.final_loss.validation_metrics import (
    validate_fixed_cohorts,
)
from multilingual_megatron.experiments.scaling.language_composition.training.training_actions import (
    AUTHORIZATION_SCHEMA,
    _cohort_means,
    _run_smoke_gate,
    storage_estimate,
    verify_authorization,
)


def _template_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = [
        {
            "job_id": "storage_preflight",
            "stage": "storage",
            "action": "storage_preflight",
            "resource": "cpu",
            "dependencies": [],
            "array": None,
            "arguments": {},
            "outputs": ["unused/storage.json"],
            "gpu_tokens": 0,
        },
        {
            "job_id": "data_reuse_gate",
            "stage": "data",
            "action": "reuse_sealed_data_verify",
            "resource": "cpu",
            "dependencies": ["storage_preflight"],
            "array": None,
            "arguments": {},
            "outputs": ["unused/reuse.json"],
            "gpu_tokens": 0,
        },
        {
            "job_id": "data_megatron_reader_smoke",
            "stage": "data",
            "action": "megatron_data_smoke",
            "resource": "cpu_largemem",
            "dependencies": ["data_reuse_gate"],
            "array": None,
            "arguments": {},
            "outputs": ["unused/reader.json"],
            "gpu_tokens": 0,
        },
    ]
    for experts in EXPERTS:
        jobs.append(
            {
                "job_id": f"prod_trunk_b_e{experts}_k012",
                "arguments": {
                    "base_peak_lr": 0.001 / experts,
                    "learning_rate_policy": "jointmoe_paper_rounded_coefficients_k_invariant",
                    "lr_multiplier": 1.0,
                    "peak_lr": 0.001 / experts,
                },
            }
        )
    return jobs


def _references() -> list[dict[str, object]]:
    return [
        {
            "source_job_id": f"source_e{experts}_d{horizon}_s{seed}",
            "seed": seed,
            "experts": experts,
            "horizon_tokens": horizon,
            "result": {"path": "/source/result.json", "sha256": "a" * 64},
            "receipt": {"path": "/source/receipt.json", "sha256": "b" * 64},
        }
        for seed in SEEDS
        for experts in EXPERTS
        for horizon in HORIZONS
    ]


class LanguageCompositionTrainingWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime_path = (
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/"
            "workflow_runtime_language_composition_v1_attempt3.json"
        )
        cls.spec_path = (
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/"
            "study_spec_language_composition_v1.json"
        )
        cls.runtime = training_contracts.validate_runtime(
            cls.runtime_path, require_data=False
        )

    def test_downstream_study_reuses_the_sealed_selection_authority(self) -> None:
        authority = contracts.verify_selection_authority()

        self.assertEqual(
            authority["selection_identity_sha256"], contracts.SELECTION_IDENTITY
        )
        for arm in ("compact", "diverse"):
            self.assertEqual(
                authority["arms"][arm], training_contracts.arm_languages(arm)
            )

    def test_spec_and_runtime_freeze_the_scientific_axes(self) -> None:
        resolved = training_contracts.validate_study_spec(
            repo_root=REPO_ROOT, spec_path=self.spec_path
        )
        self.assertEqual(
            resolved["training_policy"]["learning_rate_prescription"]["formula"],
            "exp(8.39 - 0.81 * ln(N_act_nonembed) - 0.25 * ln(E))",
        )
        self.assertEqual(
            self.runtime["seeds"]["replication_training"], list(SEEDS)
        )
        self.assertEqual(
            resolved["language_composition_design"]["selection"]["method"],
            "equal-weight mean across the frozen typology, genealogy, geography, "
            "and script distance matrices",
        )
        self.assertEqual(
            resolved["language_composition_design"]["evaluation_contract"][
                "primary_cohort"
            ],
            "unseen_177",
        )
        self.assertEqual(
            tuple(self.runtime["training"]["checkpoint_horizons_tokens"]["B"]),
            HORIZONS,
        )

    def test_runtime_data_root_is_owned_by_the_profile(self) -> None:
        with patch.object(training_contracts, "REPO_ROOT", Path("/different/checkout")):
            resolved = training_contracts.validate_runtime(
                self.runtime_path, require_data=False
            )

        self.assertEqual(
            resolved["data"]["sealed_corpus_root"],
            self.runtime["data"]["sealed_corpus_root"],
        )

    def test_attempt2_runtime_remains_explicitly_valid(self) -> None:
        attempt2 = (
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/"
            "workflow_runtime_language_composition_v1.json"
        )
        resolved = training_contracts.validate_runtime(
            attempt2, require_data=False
        )
        self.assertEqual(
            resolved["workflow_attempt_id"],
            "stage1_language_composition_v1_attempt2",
        )

    def test_attempt3_diagnostic_source_is_runtime_owned(self) -> None:
        source = self.runtime["language_composition"]["diagnostic_source_manifest"]
        self.assertEqual(
            source,
            "/workspace/language_composition/"
            "outputs/pipeline_runs/stage1_language_composition_v1_attempt2/"
            "control/launch_manifest.json",
        )

    def test_release_runtime_discloses_the_exact_attempt3_boundary(self) -> None:
        release_path = (
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/"
            "workflow_runtime_language_composition_v1_release1.json"
        )
        release = training_contracts.validate_runtime(
            release_path, require_data=False
        )
        evidence = release["language_composition"]["production_smoke_evidence"]
        self.assertEqual(
            evidence["decision"],
            production_smoke_evidence.DECISION,
        )
        self.assertEqual(
            tuple(evidence["waived_errors"]),
            production_smoke_evidence.ALLOWED_WAIVED_ERRORS,
        )

    def test_release_dag_reuses_smokes_without_new_gpu_sentinels(self) -> None:
        release_path = (
            REPO_ROOT
            / "configs/scaling_law/multilingual_moe_ab_v2/"
            "workflow_runtime_language_composition_v1_release1.json"
        )
        release = training_contracts.validate_runtime(
            release_path, require_data=False
        )
        jobs = training_jobs.build_jobs(
            template_jobs=_template_jobs(),
            runtime=release,
            output_root=REPO_ROOT
            / "outputs/pipeline_runs/test_language_composition_release",
            references=_references(),
        )
        training_jobs.validate_topology(jobs)
        self.assertEqual(len(jobs), 312)
        by_id = {row["job_id"]: row for row in jobs}
        self.assertFalse(any(job_id.startswith("smoke_") for job_id in by_id))
        self.assertEqual(
            by_id["language_composition_authorization_gate"]["dependencies"],
            [production_smoke_evidence.EVIDENCE_GATE_JOB_ID],
        )
        self.assertEqual(
            by_id[production_smoke_evidence.EVIDENCE_GATE_JOB_ID]["dependencies"],
            ["data_megatron_reader_smoke"],
        )

    def test_exact_dag_has_six_storage_safe_waves(self) -> None:
        jobs = training_jobs.build_jobs(
            template_jobs=_template_jobs(),
            runtime=self.runtime,
            output_root=REPO_ROOT / "outputs/pipeline_runs/test_language_composition",
            references=_references(),
        )
        training_jobs.validate_topology(jobs)
        self.assertEqual(len(jobs), 317)
        by_id = {row["job_id"]: row for row in jobs}
        self.assertEqual(
            by_id["lc_trunk_diverse_b_e1_s1234"]["dependencies"],
            [
                "retention_prune_compact_b_e1_s1234",
                "retention_prune_compact_b_e16_s1234",
                "retention_prune_compact_b_e32_s1234",
            ],
        )
        self.assertEqual(
            by_id["lc_trunk_compact_b_e1_s5678"]["dependencies"],
            [
                "retention_prune_diverse_b_e1_s1234",
                "retention_prune_diverse_b_e16_s1234",
                "retention_prune_diverse_b_e32_s1234",
            ],
        )
        self.assertFalse(
            any(
                row.get("action") == "production_train"
                and row.get("arguments", {}).get("composition_arm") == "reference"
                for row in jobs
            )
        )

    def test_topology_fails_closed_on_cross_wave_leak(self) -> None:
        jobs = training_jobs.build_jobs(
            template_jobs=_template_jobs(),
            runtime=self.runtime,
            output_root=REPO_ROOT / "outputs/pipeline_runs/test_language_composition",
            references=_references(),
        )
        broken = copy.deepcopy(jobs)
        by_id = {row["job_id"]: row for row in broken}
        by_id["lc_trunk_diverse_b_e1_s1234"]["dependencies"] = [
            "language_composition_authorization_gate"
        ]
        with self.assertRaisesRegex(WorkflowError, "wave dependency"):
            training_jobs.validate_topology(broken)

    def test_every_arm_satisfies_the_stable_evaluator_cohort_schema(self) -> None:
        canonical_k50 = training_contracts.grouping_languages(
            REPO_ROOT / "configs/language_groupings/scaling_fw2_v2_k050.json",
            expected_count=50,
        )
        compatibility_by_arm = {}
        for arm in ("reference", "compact", "diverse"):
            trained = training_contracts.arm_languages(arm)
            compatibility = training_contracts.evaluator_compatibility_k50(
                trained_languages=trained,
                canonical_k50=list(canonical_k50),
            )
            compatibility_by_arm[arm] = compatibility
            validated = validate_fixed_cohorts(
                {
                    "english": ["eng_Latn"],
                    "k12": trained,
                    "k50": compatibility,
                }
            )
            self.assertTrue(set(trained).issubset(validated["k50"]))
        self.assertIn("dan_Latn", compatibility_by_arm["compact"])

    def test_storage_bound_accounts_for_wave_and_retained_terminals(self) -> None:
        estimate = storage_estimate(self.runtime)
        per_set = sum(estimate["per_checkpoint_bytes"].values())
        self.assertEqual(
            estimate["maximum_current_wave_checkpoint_bytes"], 14 * per_set
        )
        self.assertEqual(
            estimate["maximum_prior_terminal_checkpoint_bytes"], 5 * per_set
        )
        self.assertEqual(
            estimate["retained_terminal_checkpoint_bytes"], 6 * per_set
        )
        self.assertGreater(
            estimate["required_available_bytes_at_preflight"], 3 * 1024**4
        )

    def test_production_authorization_is_exactly_hash_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = Paths(Path(temporary))
            paths.control.mkdir(parents=True)
            smoke_path = paths.control / "language_composition_smoke_gate.json"
            smoke = {"status": "passed", "payload_sha256": "s" * 64}
            _atomic_json(smoke_path, smoke)
            manifest = {
                "manifest_sha256": "m" * 64,
                "data_materialization": {
                    "sealed_corpus": {"data_seal_sha256": "d" * 64}
                },
            }
            authorization_path = paths.control / "authorization.json"
            authorization = {
                "schema": AUTHORIZATION_SCHEMA,
                "decision": "authorized",
                "design_id": "stage1_language_composition_v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "data_seal_sha256": "d" * 64,
                "smoke_gate_sha256": _file_sha256(smoke_path),
                "smoke_gate_payload_sha256": smoke["payload_sha256"],
                "authorized_scope": "18 new trajectories and 162 K196 evaluations",
                "authorized_by": "test-reviewer",
                "authorized_at_utc": "2026-08-31T00:00:00Z",
            }
            authorization["authorization_sha256"] = _payload_sha256(authorization)
            _atomic_json(authorization_path, authorization)
            runtime = {
                "language_composition": {
                    "production_authorization_file": str(authorization_path)
                }
            }
            verified = verify_authorization(manifest, paths, runtime)
            self.assertEqual(verified["authorized_by"], "test-reviewer")
            authorization["authorized_scope"] = "broader scope"
            _atomic_json(authorization_path, authorization)
            with self.assertRaisesRegex(WorkflowError, "absent, invalid, or out of scope"):
                verify_authorization(manifest, paths, runtime)

    def test_disclosed_boundary_authorization_is_exactly_hash_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = Paths(Path(temporary))
            paths.control.mkdir(parents=True)
            evidence_path = (
                paths.control
                / production_smoke_evidence.EVIDENCE_GATE_FILENAME
            )
            evidence_gate = {
                "status": production_smoke_evidence.DECISION,
                "decision": production_smoke_evidence.DECISION,
                "payload_sha256": "e" * 64,
            }
            _atomic_json(evidence_path, evidence_gate)
            manifest = {
                "manifest_sha256": "m" * 64,
                "data_materialization": {
                    "sealed_corpus": {"data_seal_sha256": "d" * 64}
                },
            }
            authorization_path = paths.control / "authorization.json"
            smoke_evidence = {
                "source_manifest_payload_sha256": "s" * 64,
                "source_smoke_gate_file_sha256": "f" * 64,
                "source_smoke_gate_payload_sha256": "g" * 64,
                "waived_errors": list(
                    production_smoke_evidence.ALLOWED_WAIVED_ERRORS
                ),
                "rationale": "explicit test boundary",
                "authorized_by": "test-reviewer",
            }
            authorization = {
                "schema": production_smoke_evidence.AUTHORIZATION_SCHEMA,
                "decision": production_smoke_evidence.DECISION,
                "design_id": "stage1_language_composition_v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "data_seal_sha256": "d" * 64,
                "smoke_evidence_gate_sha256": _file_sha256(evidence_path),
                "smoke_evidence_gate_payload_sha256": "e" * 64,
                "source_manifest_payload_sha256": "s" * 64,
                "source_smoke_gate_sha256": "f" * 64,
                "source_smoke_gate_payload_sha256": "g" * 64,
                "waived_errors": smoke_evidence["waived_errors"],
                "waiver_rationale": smoke_evidence["rationale"],
                "authorized_scope": (
                    "18 new trajectories and 162 K196 evaluations"
                ),
                "authorized_by": "test-reviewer",
                "authorized_at_utc": "2026-09-04T00:00:00Z",
            }
            authorization["authorization_sha256"] = _payload_sha256(
                authorization
            )
            _atomic_json(authorization_path, authorization)
            runtime = {
                "language_composition": {
                    "production_authorization_file": str(authorization_path),
                    "production_smoke_evidence": smoke_evidence,
                }
            }
            verified = verify_authorization(manifest, paths, runtime)
            self.assertEqual(
                verified["decision"], production_smoke_evidence.DECISION
            )
            authorization["waived_errors"] = []
            _atomic_json(authorization_path, authorization)
            with self.assertRaisesRegex(
                WorkflowError, "absent, invalid, or out of scope"
            ):
                verify_authorization(manifest, paths, runtime)

    def test_malformed_e32_smoke_fails_closed_without_attribute_error(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = Paths(Path(temporary))
            result_path = paths.jobs / "bad_e32" / "result.json"
            result_path.parent.mkdir(parents=True)
            _atomic_json(
                result_path,
                {
                    "status": "failed",
                    "arguments": {
                        "experts": 32,
                        "composition_arm": "compact",
                        "tokens": SMOKE_TOKENS,
                        "sentinel_kind": "language_composition_long_real_data",
                        "global_batch_size": SMOKE_GLOBAL_BATCH_SIZE,
                    },
                    "sentinel_health": None,
                },
            )
            job = {
                "arguments": {
                    "minimum_reference_throughput_fraction": 0.85,
                    "reference_median_tokens_per_second": {"32": 203_559.56},
                    "steady_state_window_observations": (
                        SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS
                    ),
                    "maximum_step_time_cv_pct": SMOKE_MAX_STEP_TIME_CV_PCT,
                    "smoke_jobs": ["bad_e32"],
                }
            }
            with patch.object(
                training_actions,
                "_verified_receipt_evidence",
                return_value={"status": "complete"},
            ):
                with self.assertRaisesRegex(WorkflowError, "smoke gate failed"):
                    _run_smoke_gate(
                        {"manifest_sha256": "m" * 64}, job, paths
                    )

    def test_smoke_gate_requires_a_complete_stable_late_window(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = Paths(Path(temporary))
            data_manifest = paths.tokenized / "composition_arms/compact/megatron_data_manifest.json"
            data_manifest.parent.mkdir(parents=True)
            _atomic_json(data_manifest, {"arm": "compact"})
            result_path = paths.jobs / "good_e1" / "result.json"
            result_path.parent.mkdir(parents=True)
            result = {
                "status": "complete",
                "arguments": {
                    "experts": 1,
                    "composition_arm": "compact",
                    "tokens": SMOKE_TOKENS,
                    "sentinel_kind": "language_composition_long_real_data",
                    "global_batch_size": SMOKE_GLOBAL_BATCH_SIZE,
                },
                "actual_end_tokens": SMOKE_TOKENS,
                "checkpoint_state": {"valid_full_state": True},
                "data_manifest": str(data_manifest.resolve()),
                "data_manifest_sha256": _file_sha256(data_manifest),
                "sentinel_health": {
                    "status": "ok",
                    "errors": [],
                    "metrics": {
                        "finite_loss_and_gradients": True,
                        "lm_loss_lines": 200,
                        "grad_norm_lines": 200,
                        "max_skipped_iterations": 0,
                        "max_nan_iterations": 0,
                        "nonfinite_lm_loss_count": 0,
                        "nonfinite_grad_norm_count": 0,
                        "steps_counted": SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS,
                        "step_time_cv_pct": 4.9,
                        "median_tokens_per_second": 300_000.0,
                    },
                },
            }
            _atomic_json(result_path, result)
            job = {
                "arguments": {
                    "minimum_reference_throughput_fraction": 0.85,
                    "reference_median_tokens_per_second": {"1": 329_906.87},
                    "steady_state_window_observations": (
                        SMOKE_STEADY_STATE_WINDOW_OBSERVATIONS
                    ),
                    "maximum_step_time_cv_pct": SMOKE_MAX_STEP_TIME_CV_PCT,
                    "smoke_jobs": ["good_e1"],
                }
            }
            with patch.object(
                training_actions,
                "_verified_receipt_evidence",
                return_value={"status": "complete"},
            ):
                _run_smoke_gate({"manifest_sha256": "m" * 64}, job, paths)
                report = training_contracts.read_object(
                    paths.control / "language_composition_smoke_gate.json",
                    label="smoke gate",
                )
                self.assertEqual(report["status"], "passed")
                self.assertEqual(
                    report["schema"], "stage1_language_composition_smoke_gate/v2"
                )

                result["sentinel_health"]["metrics"]["step_time_cv_pct"] = 5.0
                _atomic_json(result_path, result)
                with self.assertRaisesRegex(
                    WorkflowError, "steady-state stability"
                ):
                    _run_smoke_gate(
                        {"manifest_sha256": "m" * 64}, job, paths
                    )

    def test_empty_predeclared_cohort_is_explicitly_not_applicable(self) -> None:
        means = _cohort_means(
            {"eng_Latn": 1.5},
            {"english": ["eng_Latn"], "compact_diverse": []},
        )
        self.assertEqual(means["english"], 1.5)
        self.assertIsNone(means["compact_diverse"])
        with self.assertRaisesRegex(WorkflowError, "lacks cohort"):
            _cohort_means({}, {"english": ["eng_Latn"]})

    def test_training_result_resolves_the_manifest_actually_consumed(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "arm_manifest.json"
            _atomic_json(manifest, {"arm": "compact"})
            config = {
                "profiles": {
                    "ab_v2_job": {
                        "training": {
                            "runs": [
                                {
                                    "data_manifest": str(manifest),
                                    "data_manifest_sha256": _file_sha256(manifest),
                                }
                            ]
                        }
                    }
                }
            }
            self.assertEqual(
                _resolved_profile_data_manifest(config, {}), manifest.resolve()
            )
            _atomic_json(manifest, {"arm": "diverse"})
            with self.assertRaisesRegex(WorkflowError, "identity changed"):
                _resolved_profile_data_manifest(config, {})


if __name__ == "__main__":
    unittest.main()
