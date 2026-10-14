from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.experiments.scaling.stage1_release.validation import Stage1ValidationError
from multilingual_megatron.experiments.scaling.stage1_release.validation import build_stage1_ad_validation
from multilingual_megatron.experiments.scaling.stage1_release.validation import semantic_sufficient_statistics_sha256


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(payload)
    sealed[field] = _payload_sha256(payload)
    return sealed


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _language_row(
    language: str,
    *,
    role: str,
    nll_sum: float,
) -> dict[str, Any]:
    token_count = 2
    unigram_sum = -2.0
    adjusted_sum = nll_sum + unigram_sum
    return {
        "adjusted_nll_sum": adjusted_sum,
        "atlas_tao_unigram_adjusted_loss": adjusted_sum / token_count,
        "input_sha256": _hash(f"{language}:input"),
        "language": language,
        "nll_sum": nll_sum,
        "raw_nll": nll_sum / token_count,
        "required": True,
        "role": role,
        "schema": "multilingual_validation_language_loss/v1",
        "special_token_ids": [0, 1],
        "target_token_ids_sha256": _hash(f"{language}:targets"),
        "token_count": token_count,
        "unigram_counts_sha256": _hash(f"{language}:counts"),
        "unigram_log_probability_sum": unigram_sum,
        "unseen_target_token_count": 0,
        "vocab_size": 16,
    }


def _role_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "equal_language_atlas_tao_adjusted_loss": None,
            "equal_language_raw_nll": None,
            "language_count": 0,
            "languages": [],
            "status": "empty",
            "token_count": 0,
            "token_micro_atlas_tao_adjusted_loss": None,
            "token_micro_raw_nll": None,
        }
    token_count = sum(row["token_count"] for row in rows)
    return {
        "equal_language_atlas_tao_adjusted_loss": sum(
            row["atlas_tao_unigram_adjusted_loss"] for row in rows
        )
        / len(rows),
        "equal_language_raw_nll": sum(row["raw_nll"] for row in rows) / len(rows),
        "language_count": len(rows),
        "languages": sorted(row["language"] for row in rows),
        "status": "complete",
        "token_count": token_count,
        "token_micro_atlas_tao_adjusted_loss": sum(
            row["adjusted_nll_sum"] for row in rows
        )
        / token_count,
        "token_micro_raw_nll": sum(row["nll_sum"] for row in rows) / token_count,
    }


def _make_evaluation(
    *,
    evaluation_root: Path,
    readout_id: str,
    checkpoint_root: Path,
    checkpoint_step: int,
    semantic_delta: float = 0.0,
    runtime_label: str,
) -> dict[str, Any]:
    job_root = evaluation_root / "jobs" / readout_id
    languages_root = job_root / "languages"
    languages_root.mkdir(parents=True)

    english = _language_row("eng_Latn", role="trained", nll_sum=4.0 + semantic_delta)
    french = _language_row("fra_Latn", role="zero_shot", nll_sum=8.0)
    report_rows = [english, french]
    language_descriptors: dict[str, Any] = {}
    for row in report_rows:
        artifact = {
            key: value for key, value in row.items() if key not in {"required", "role"}
        }
        path = languages_root / f"{row['language']}.json"
        _write_json(path, artifact)
        language_descriptors[row["language"]] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "token_count": row["token_count"],
            "nll_sum": row["nll_sum"],
        }

    trained = _role_aggregate([english])
    zero_shot = _role_aggregate([french])
    report = {
        "aggregates": {
            "fixed_common_cohorts": {
                "english": {
                    **trained,
                    "membership_sha256": _hash("english-membership"),
                },
                "k12": {
                    "languages": ["eng_Latn", "fra_Latn"],
                    "membership_sha256": _hash("k12-membership"),
                    "missing_trained_languages": ["fra_Latn"],
                    "reason": "not_all_cohort_languages_were_trained",
                    "status": "not_applicable",
                },
            },
            "primary": {
                "language_count": 1,
                "languages": ["eng_Latn"],
                "name": "trained_equal_language_atlas_tao_unigram_adjusted_loss",
                "value": trained["equal_language_atlas_tao_adjusted_loss"],
            },
            "secondary": {
                "language_count": 1,
                "languages": ["eng_Latn"],
                "name": "trained_equal_language_raw_nll",
                "target_domain": "non_special_validation_tokens",
                "value": trained["equal_language_raw_nll"],
            },
            "trained_languages": trained,
            "zero_shot_languages": zero_shot,
        },
        "coverage": {
            "missing_required_languages": [],
            "observed_languages": ["eng_Latn", "fra_Latn"],
            "required_languages": ["eng_Latn", "fra_Latn"],
            "trained_languages": ["eng_Latn"],
            "zero_shot_languages": ["fra_Latn"],
        },
        "languages": report_rows,
        "metric_spec": {
            "adjusted_formula": "nll_sum_plus_unigram_log_probability_sum",
            "primary": "trained_equal_language_atlas_tao_unigram_adjusted_loss",
            "secondary": "raw_token_nll",
        },
        "run_id": f"{readout_id}:{runtime_label}",
        "schema": "multilingual_validation_report/v1",
        "status": "complete",
    }
    report = _seal(report, "report_sha256")
    report_path = job_root / "validation_report.json"
    _write_json(report_path, report)

    spec = {
        "checkpoint": {"path": str(checkpoint_root), "step": checkpoint_step},
        "output_dir": str(job_root),
        "required_languages": ["eng_Latn", "fra_Latn"],
        "schema": "multilingual_per_language_checkpoint_eval/v1",
        "trained_languages": ["eng_Latn"],
    }
    spec_path = job_root / "evaluation_spec.json"
    _write_json(spec_path, spec)

    manifest = {
        "checkpoint": {
            "iteration_path": str(checkpoint_root / f"iter_{checkpoint_step:07d}"),
            "root": str(checkpoint_root),
            "step": checkpoint_step,
        },
        "coverage": {
            "evaluated_languages": ["eng_Latn", "fra_Latn"],
            "missing_required_languages": [],
            "required_languages": ["eng_Latn", "fra_Latn"],
            "trained_languages": ["eng_Latn"],
        },
        "languages": language_descriptors,
        "run_id": f"{readout_id}:{runtime_label}",
        "schema": "multilingual_per_language_checkpoint_eval_result/v1",
        "spec": {"path": str(spec_path), "sha256": _file_sha256(spec_path)},
        "status": "complete",
        "timing": {"forward_seconds": 1.0 if runtime_label == "first" else 2.0},
        "validation_report": {
            "path": str(report_path),
            "report_sha256": report["report_sha256"],
            "sha256": _file_sha256(report_path),
        },
    }
    manifest = _seal(manifest, "result_sha256")
    manifest_path = job_root / "evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    (job_root / ".validation_complete").write_text("complete\n")
    return {
        "manifest": manifest_path,
        "manifest_sha256": _file_sha256(manifest_path),
        "result_sha256": manifest["result_sha256"],
        "report": report_path,
        "report_sha256": _file_sha256(report_path),
        "semantic_sha256": semantic_sufficient_statistics_sha256(report),
    }


def _make_training(
    *,
    root: Path,
    training_job_id: str,
    checkpoint_step: int,
    actual_tokens: int,
    expected_samples: int,
) -> Path:
    checkpoint_root = (
        root
        / "jobs"
        / training_job_id
        / "runs"
        / "run"
        / "segments"
        / "00"
        / "checkpoints"
    )
    checkpoint_root.mkdir(parents=True)
    training_manifest = {
        "metadata": {
            "segments": [
                {
                    "actual_end_tokens": actual_tokens,
                    "cumulative_end_step": checkpoint_step,
                    "expected_consumed_samples": expected_samples,
                }
            ]
        },
        "status": "complete",
    }
    _write_json(
        checkpoint_root.parents[2] / "training_run_manifest.json",
        training_manifest,
    )
    return checkpoint_root


def _build_fixture(
    root: Path,
    *,
    duplicate_delta: float = 0.0,
) -> dict[str, Path]:
    ab_root = root / "ab"
    ab_readout = "prod_eval_a_e1_k001_d500000000"
    ab_training_id = "prod_branch_a_e1_k001_d500000000"
    ab_checkpoint = _make_training(
        root=ab_root,
        training_job_id=ab_training_id,
        checkpoint_step=10,
        actual_tokens=500_039_680,
        expected_samples=244_160,
    )
    ab_evaluation = _make_evaluation(
        evaluation_root=ab_root,
        readout_id=ab_readout,
        checkpoint_root=ab_checkpoint,
        checkpoint_step=10,
        runtime_label="first",
    )
    ab_report = json.loads(ab_evaluation["report"].read_text())
    summary = {
        "errors": [],
        "expected_results": 1,
        "observed_results": 1,
        "results": [
            {
                "D": 500_000_000,
                "K": 1,
                "actual_end_tokens": 500_039_680,
                "equal_language_atlas_tao_adjusted_loss": ab_report["aggregates"][
                    "primary"
                ]["value"],
                "expected_consumed_samples": 244_160,
                "experts": 1,
                "job_id": ab_readout,
                "ladder": "A",
                "nominal_D_tokens": 500_000_000,
                "purpose": "production",
                "raw_token_nll": ab_report["aggregates"]["secondary"]["value"],
                "sha256": ab_evaluation["manifest_sha256"],
                "training_job_id": ab_training_id,
            }
        ],
        "schema_version": 1,
        "status": "complete",
    }
    _write_json(ab_root / "control" / "final_summary.json", summary)

    cd_readout = "prod_eval_c_e4_k012_d500000000"
    cd_training_id = "prod_branch_c_e4_k012_d500000000"
    training_root = root / "cd_training"
    cd_checkpoint = _make_training(
        root=training_root,
        training_job_id=cd_training_id,
        checkpoint_step=20,
        actual_tokens=500_170_752,
        expected_samples=244_224,
    )
    canonical = _make_evaluation(
        evaluation_root=root / "cd_canonical",
        readout_id=cd_readout,
        checkpoint_root=cd_checkpoint,
        checkpoint_step=20,
        runtime_label="first",
    )
    recovery = _make_evaluation(
        evaluation_root=root / "cd_recovery",
        readout_id=cd_readout,
        checkpoint_root=cd_checkpoint,
        checkpoint_step=20,
        semantic_delta=duplicate_delta,
        runtime_label="recovery",
    )

    def candidate(evaluation: dict[str, Any]) -> dict[str, Any]:
        return {
            "details": {
                "checkpoint_root": str(cd_checkpoint),
                "checkpoint_step": 20,
                "manifest": str(evaluation["manifest"]),
                "manifest_file_sha256": evaluation["manifest_sha256"],
                "required_language_count": 2,
                "result_sha256": evaluation["result_sha256"],
                "scored_token_count": 4,
                "spec": str(evaluation["manifest"].parent / "evaluation_spec.json"),
                "spec_sha256": _file_sha256(
                    evaluation["manifest"].parent / "evaluation_spec.json"
                ),
                "trained_language_count": 1,
            },
            "path": str(evaluation["manifest"]),
            "valid": True,
        }

    ledger = {
        "created_at_utc": "2026-07-30T00:00:00+00:00",
        "expected_results": 1,
        "ladders": {"C": 1},
        "records": [
            {
                "candidate_count": 2,
                "candidates": [candidate(canonical), candidate(recovery)],
                "canonical_id": cd_readout,
                "ladder": "C",
                "selected_manifest": str(canonical["manifest"]),
                "selection_reason": "valid_canonical_root",
                "valid_candidate_count": 2,
            }
        ],
        "schema": "stage1_cd_cross_root_analysis_ledger/v1",
        "selected_results": 1,
        "selection_policy": [
            "prefer hash-valid canonical result-root artifact",
            "otherwise select lexicographically first hash-valid recovery artifact",
        ],
    }
    ledger = _seal(ledger, "ledger_sha256")
    ledger_path = root / "cross_root_result_ledger.json"
    _write_json(ledger_path, ledger)
    return {
        "ab_root": ab_root,
        "cd_ledger": ledger_path,
        "cd_selected_report": canonical["report"],
    }


class Stage1AdValidationTest(unittest.TestCase):
    def test_builds_readouts_languages_aggregates_and_duplicate_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _build_fixture(Path(temporary))

            tables = build_stage1_ad_validation(
                ab_result_root=paths["ab_root"],
                cd_ledger_path=paths["cd_ledger"],
                expected_ladder_counts={"A": 1, "B": 0, "C": 1, "D": 0},
            )

        self.assertEqual(len(tables.readouts), 2)
        self.assertEqual(
            [row["ladder"] for row in tables.readouts],
            ["A", "C"],
        )
        self.assertEqual(len(tables.validation_language), 4)
        self.assertEqual(
            {row["metric_role"] for row in tables.validation_language},
            {"trained", "zero_shot"},
        )
        self.assertTrue(
            all(
                row["source_path"]
                and row["source_sha256"]
                and row["provenance_class"] == "observed"
                for row in (
                    *tables.readouts,
                    *tables.validation_aggregates,
                    *tables.validation_language,
                    *tables.validation_candidates,
                    *tables.artifacts,
                    *tables.provenance,
                )
            )
        )
        c_readout = next(row for row in tables.readouts if row["ladder"] == "C")
        self.assertEqual(c_readout["candidate_count"], 2)
        self.assertEqual(c_readout["valid_candidate_count"], 2)
        self.assertEqual(c_readout["selection_reason"], "valid_canonical_root")
        self.assertEqual(c_readout["actual_tokens"], 500_170_752)
        self.assertEqual(
            c_readout["training_job_id"],
            "prod_branch_c_e4_k012_d500000000",
        )
        c_candidates = [
            row for row in tables.validation_candidates if row["ladder"] == "C"
        ]
        self.assertEqual(len(c_candidates), 2)
        self.assertEqual(sum(bool(row["selected"]) for row in c_candidates), 1)
        self.assertTrue(
            all(row["semantically_matches_selected"] for row in c_candidates)
        )
        self.assertEqual(
            len({row["semantic_sufficient_statistics_sha256"] for row in c_candidates}),
            1,
        )
        aggregate_kinds = {
            (row["aggregate_kind"], row["cohort"])
            for row in tables.validation_aggregates
            if row["ladder"] == "C"
        }
        self.assertTrue(
            {
                ("metric_role", "trained_languages"),
                ("metric_role", "zero_shot_languages"),
                ("fixed_common_cohort", "english"),
                ("fixed_common_cohort", "k12"),
                ("declared_metric", "primary"),
                ("declared_metric", "secondary"),
            }.issubset(aggregate_kinds)
        )
        self.assertIn(
            "validation_language",
            {row["artifact_type"] for row in tables.artifacts},
        )
        self.assertIn(
            "cd_cross_root_ledger_selects_evaluation",
            {row["relation"] for row in tables.provenance},
        )

    def test_rejects_semantically_different_valid_cd_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _build_fixture(Path(temporary), duplicate_delta=0.5)

            with self.assertRaisesRegex(
                Stage1ValidationError,
                "not semantically identical",
            ):
                build_stage1_ad_validation(
                    ab_result_root=paths["ab_root"],
                    cd_ledger_path=paths["cd_ledger"],
                    expected_ladder_counts={
                        "A": 1,
                        "B": 0,
                        "C": 1,
                        "D": 0,
                    },
                )

    def test_rejects_report_changed_after_manifest_was_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _build_fixture(Path(temporary))
            report_path = paths["cd_selected_report"]
            report = json.loads(report_path.read_text())
            report["run_id"] = "tampered"
            _write_json(report_path, report)

            with self.assertRaisesRegex(
                Stage1ValidationError,
                "file hash mismatch",
            ):
                build_stage1_ad_validation(
                    ab_result_root=paths["ab_root"],
                    cd_ledger_path=paths["cd_ledger"],
                    expected_ladder_counts={
                        "A": 1,
                        "B": 0,
                        "C": 1,
                        "D": 0,
                    },
                )

    def test_semantic_identity_ignores_run_id_but_not_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _build_fixture(Path(temporary))
            report = json.loads(paths["cd_selected_report"].read_text())
            renamed = dict(report)
            renamed["run_id"] = "different-runtime-run-id"
            changed = json.loads(json.dumps(report))
            changed["languages"][0]["nll_sum"] += 1.0

            self.assertEqual(
                semantic_sufficient_statistics_sha256(report),
                semantic_sufficient_statistics_sha256(renamed),
            )
            self.assertNotEqual(
                semantic_sufficient_statistics_sha256(report),
                semantic_sufficient_statistics_sha256(changed),
            )


if __name__ == "__main__":
    unittest.main()
