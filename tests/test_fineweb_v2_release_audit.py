from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multilingual_megatron.pipeline.stages.audit.fineweb_v2_release import (
    EXPECTED_TRAIN_TOKENS,
    ReleaseAuditError,
    _audit_bootstrap,
    _audit_corpus_release,
    _audit_embedded_paths,
    _audit_materialized_inventory_selection,
    _resolve_source_artifact_for_audit,
    _audit_sampling_plan,
    _is_adopted_equal_hash_source_lock,
    _load_k12_adoption_lineage,
    _sealed_stat,
    _verify_predecessor_seal_stats,
    _verify_prefix,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (
    assign_validation_reserve,
    canonical_json_sha256,
    sha256_file,
    stable_document_id,
)
from multilingual_megatron.pipeline.stages.sampling.tools.build_fineweb_v2_sampling_plan import (
    build_plan_artifact,
)
from multilingual_megatron.pipeline.tokenizer_sentence_contract import (
    ACQUISITION_PLAN_TOOL_ID,
    sentence_slot_gate_contract,
    sentence_splitter_contract,
    tokenizer_acquisition_target_contract,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class FineWebV2ReleaseAuditTest(unittest.TestCase):
    def test_adoption_lineage_returns_every_verified_predecessor_root(self) -> None:
        release = Path("/release").resolve()
        predecessor = Path("/predecessor").resolve()
        ancestor = Path("/ancestor").resolve()
        with patch(
            "multilingual_megatron.pipeline.stages.audit.fineweb_release."
            "integrity._load_k12_adoption",
            side_effect=[
                {"predecessor_root": str(predecessor)},
                {"predecessor_root": str(ancestor)},
                None,
            ],
        ):
            adoption, roots = _load_k12_adoption_lineage(release)
        self.assertEqual(adoption, {"predecessor_root": str(predecessor)})
        self.assertEqual(roots, (predecessor, ancestor))

    def test_adopted_source_lock_allows_only_equal_hash_provenance_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "current" / "source-lock.json"
            materialized = root / "predecessor" / "source-lock.json"
            expected.parent.mkdir()
            materialized.parent.mkdir()
            expected.write_text("same bytes\n", encoding="utf-8")
            materialized.write_text("same bytes\n", encoding="utf-8")
            observed = sha256_file(materialized)
            self.assertTrue(
                _is_adopted_equal_hash_source_lock(
                    materialized=materialized,
                    expected=expected,
                    materialized_sha256=observed,
                    adoption_enabled=True,
                )
            )
            self.assertFalse(
                _is_adopted_equal_hash_source_lock(
                    materialized=materialized,
                    expected=expected,
                    materialized_sha256=observed,
                    adoption_enabled=False,
                )
            )
            expected.write_text("different\n", encoding="utf-8")
            self.assertFalse(
                _is_adopted_equal_hash_source_lock(
                    materialized=materialized,
                    expected=expected,
                    materialized_sha256=observed,
                    adoption_enabled=True,
                )
            )

    def test_k12_source_artifact_uses_equal_hash_successor_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs" / "pipeline_runs"
            predecessor = root / "h100_ab_v2_attempt7"
            lineage_ancestor = root / "h100_ab_v2_attempt6"
            successor = root / "h100_ab_v2_k12_fixed_lr_attempt3"
            old = lineage_ancestor / "corpus" / "source_lock" / "inventory.csv"
            mirror = successor / "corpus" / "source_lock" / "inventory.csv"
            old.parent.mkdir(parents=True)
            mirror.parent.mkdir(parents=True)
            old.write_text("language,documents\\neng_Latn,1\\n", encoding="utf-8")
            mirror.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
            raw = {"path": str(old), "sha256": sha256_file(old)}
            resolved, predecessor_provenance = _resolve_source_artifact_for_audit(
                source_lock_dir=mirror.parent,
                raw=raw,
                release_root=successor,
                label="source artifact inventory",
                allowed_predecessor_root=predecessor,
            )
            self.assertEqual(resolved, mirror.resolve())
            self.assertTrue(predecessor_provenance)
            mirror.write_text("different\\n", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseAuditError, "mirror hash mismatch"):
                _resolve_source_artifact_for_audit(
                    source_lock_dir=mirror.parent,
                    raw=raw,
                    release_root=successor,
                    label="source artifact inventory",
                    allowed_predecessor_root=predecessor,
                )

    def test_embedded_path_audit_accepts_only_exact_verified_lineage_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs" / "pipeline_runs"
            predecessor = root / "predecessor"
            ancestor_path = root / "ancestor" / "corpus" / "source.json"
            successor = root / "successor"
            ancestor_path.parent.mkdir(parents=True)
            ancestor_path.write_text("{}\n", encoding="utf-8")
            payloads = {
                "source_materialization": {
                    "source_lock_path": str(ancestor_path),
                }
            }
            with self.assertRaisesRegex(
                ReleaseAuditError, "points to another pipeline release"
            ):
                _audit_embedded_paths(
                    payloads,
                    release_root=successor,
                    allowed_predecessor_root=predecessor,
                )
            self.assertEqual(
                _audit_embedded_paths(
                    payloads,
                    release_root=successor,
                    allowed_predecessor_root=predecessor,
                    allowed_provenance_roots=(root / "ancestor",),
                ),
                1,
            )
            self.assertEqual(
                _audit_embedded_paths(
                    payloads,
                    release_root=successor,
                    allowed_predecessor_root=predecessor,
                    allowed_provenance_paths={ancestor_path},
                ),
                1,
            )

    def _bootstrap_row(
        self,
        *,
        language: str,
        split: str,
        source_id: str,
        text: str,
        train_multiplier: int = 1,
    ) -> dict:
        expected_candidate = split == "validation_reserve"
        for suffix in range(10_000):
            candidate_source_id = f"{source_id}-{suffix}"
            document_id = stable_document_id(
                source_dataset="HuggingFaceFW/fineweb-2",
                source_revision="fixture-revision",
                subset=language,
                source_id=candidate_source_id,
                url=None,
                text=text,
            )
            if assign_validation_reserve(
                document_id=document_id,
                seed=42,
                validation_fraction=0.30,
            ) == expected_candidate:
                break
        else:  # pragma: no cover
            raise AssertionError("could not construct deterministic candidate fixture")
        return {
            "artifact_role": split,
            "document_id": document_id,
            "language": language,
            "text": text,
            "lineage": {
                "source_dataset": "HuggingFaceFW/fineweb-2",
                "source_revision": "fixture-revision",
                "subset": language,
                "source_id": candidate_source_id,
                "url": None,
                "split": split,
                "rehydration_multiplier": (
                    1 if split == "validation_reserve" else train_multiplier
                ),
                "train_rehydration_multiplier": train_multiplier,
            },
        }

    def _bootstrap_fixture(self, root: Path) -> tuple[Path, Path]:
        language = "eng_Latn"
        language_root = root / language
        train_path = language_root / "train.jsonl"
        validation_path = language_root / "validation_reserve.jsonl"
        calibration_path = language_root / "calibration.jsonl"
        train_path.parent.mkdir(parents=True)
        train_path.write_text(
            json.dumps(
                self._bootstrap_row(
                    language=language,
                    split="train",
                    source_id="training",
                    text="training",
                    train_multiplier=2,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validation_path.write_text(
            json.dumps(
                self._bootstrap_row(
                    language=language,
                    split="validation_reserve",
                    source_id="validation",
                    text="validation",
                    train_multiplier=3,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        calibration_path.write_text('{"text":"calibration"}\n', encoding="utf-8")
        outputs = {
            "train.jsonl": {
                "path": "train.jsonl",
                "sha256": sha256_file(train_path),
                "bytes": train_path.stat().st_size,
            },
            "validation_reserve.jsonl": {
                "path": "validation_reserve.jsonl",
                "sha256": sha256_file(validation_path),
                "bytes": validation_path.stat().st_size,
            },
            "calibration.jsonl": {
                "path": "calibration.jsonl",
                "sha256": sha256_file(calibration_path),
                "bytes": calibration_path.stat().st_size,
            },
        }
        report = {
            "schema_version": 1,
            "tool_id": "fineweb_ab_v2_bootstrap_v1",
            "status": "complete",
            "acquisition_status": "targets_reached",
            "run_config": {
                "language": language,
                "settings": {
                    "seed": 42,
                    "validation_fraction": 0.30,
                    "validation_candidate_fraction": 0.30,
                },
                "validation_candidate_contract": {
                    "membership_namespace": "validation-reserve-v1",
                    "excluded_from_tokenizer_training": True,
                    "unused_candidates_return_to_model_training_selection_pool": True,
                    "preserved_train_rehydration_multiplier": True,
                },
            },
            "counts": {"train_validation_overlap": 0},
            "hashes": {
                "unique_source_ids_sha256": "1" * 64,
                "unique_train_ids_sha256": "2" * 64,
                "unique_validation_ids_sha256": "3" * 64,
            },
            "outputs": outputs,
            "targets": {"source_exhausted": False, "pinned_inventory_exhausted": False},
        }
        report["report_payload_sha256"] = canonical_json_sha256(report)
        report_path = language_root / "bootstrap_report.json"
        _write(report_path, report)
        manifest = {
            "schema_version": 1,
            "provenance": {
                "source_lock_sha256": "a" * 64,
                "language_map_sha256": "b" * 64,
                "rehydration_weights_sha256": "c" * 64,
                "sampling_plan_file_sha256": None,
                "sampling_plan_payload_sha256": None,
                "tokenizer_acquisition_plan_file_sha256": None,
                "tokenizer_acquisition_plan_payload_sha256": None,
                "incremental_mode": None,
            },
            "languages": {
                language: {
                    "acquisition": {
                        "report_path": f"{language}/bootstrap_report.json",
                        "report_payload_sha256": report["report_payload_sha256"],
                        "complete_source_scan": False,
                    },
                    "train": {
                        "path": f"{language}/train.jsonl",
                        "sha256": outputs["train.jsonl"]["sha256"],
                    },
                    "validation_reserve": {
                        "path": f"{language}/validation_reserve.jsonl",
                        "sha256": outputs["validation_reserve.jsonl"]["sha256"],
                    },
                }
            },
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        _write(root / "bootstrap_manifest.json", manifest)
        summary = {
            "schema_version": 1,
            "tool_id": "fineweb_ab_v2_bootstrap_v1",
            "languages": [language],
            "bootstrap_manifest_payload_sha256": manifest["manifest_sha256"],
            "report_payload_sha256": {
                language: report["report_payload_sha256"],
            },
        }
        summary["summary_sha256"] = canonical_json_sha256(summary)
        _write(root / "bootstrap_summary.json", summary)
        return train_path, validation_path

    def _resign_bootstrap_fixture(
        self,
        root: Path,
        *,
        report: dict,
        manifest: dict,
        summary: dict,
    ) -> None:
        language = "eng_Latn"
        report.pop("report_payload_sha256", None)
        report["report_payload_sha256"] = canonical_json_sha256(report)
        _write(root / language / "bootstrap_report.json", report)
        manifest["languages"][language]["acquisition"][
            "report_payload_sha256"
        ] = report["report_payload_sha256"]
        manifest["languages"][language]["train"]["sha256"] = report["outputs"][
            "train.jsonl"
        ]["sha256"]
        manifest["languages"][language]["validation_reserve"]["sha256"] = report[
            "outputs"
        ]["validation_reserve.jsonl"]["sha256"]
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        _write(root / "bootstrap_manifest.json", manifest)
        summary["report_payload_sha256"][language] = report[
            "report_payload_sha256"
        ]
        summary["bootstrap_manifest_payload_sha256"] = manifest["manifest_sha256"]
        summary.pop("summary_sha256", None)
        summary["summary_sha256"] = canonical_json_sha256(summary)
        _write(root / "bootstrap_summary.json", summary)

    def _exact_slot_fixture(self, root: Path) -> tuple[Path, dict]:
        train_path, _ = self._bootstrap_fixture(root)
        train_row = json.loads(train_path.read_text(encoding="utf-8"))
        train_row.update(
            {
                "artifact_role": "train",
                "language": "eng_Latn",
                "text": "First sentence. Second sentence.",
            }
        )
        train_row["lineage"].update(
            {
                "rehydration_multiplier": 2,
                "train_rehydration_multiplier": 2,
                "split": "train",
                "subset": "eng_Latn",
            }
        )
        train_path.write_text(
            json.dumps(train_row, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        splitter = sentence_splitter_contract(max_chars=60_000)
        plan = {
            "schema_version": 1,
            "tool_id": ACQUISITION_PLAN_TOOL_ID,
            "status": "complete",
            "splitter": splitter,
            "expanded_pool_gate": sentence_slot_gate_contract(max_chars=60_000),
            "languages": {
                "eng_Latn": {
                    "planned_sentence_count": 4,
                    "acquisition": {
                        "required_rehydrated_sentence_slots": 4,
                        "target_unique_source_documents": 1,
                        "target_utf8_bytes": 1,
                        "safety_margin": 1.1,
                    },
                }
            },
        }
        plan["plan_sha256"] = canonical_json_sha256(plan)
        plan_path = root / "tokenizer_sentence_acquisition_plan.json"
        _write(plan_path, plan)
        plan_file_sha256 = sha256_file(plan_path)
        target = tokenizer_acquisition_target_contract(
            plan=plan,
            plan_file_sha256=plan_file_sha256,
            language="eng_Latn",
        )
        gate_contract = plan["expanded_pool_gate"]
        gate = {
            "counter_id": gate_contract["counter_id"],
            "counted_split": gate_contract["counted_split"],
            "slot_formula": gate_contract["slot_formula"],
            "splitter": splitter,
            "non_exhausted_stop_rule": gate_contract["non_exhausted_stop_rule"],
            "required_rehydrated_sentence_slots": 4,
            "actual_unique_train_sentences": 2,
            "actual_rehydrated_sentence_slots": 4,
            "shortfall_slots": 0,
            "quota_satisfied": True,
            "true_pinned_source_exhausted": False,
            "true_source_exhausted_repetition_allowed": False,
            "artificial_pool_cycling_forbidden": True,
        }
        report = json.loads(
            (root / "eng_Latn" / "bootstrap_report.json").read_text(
                encoding="utf-8"
            )
        )
        report["run_config"]["tokenizer_sentence_acquisition_target"] = target
        report["counts"].update(
            {
                "unique_train_documents": 1,
                "unique_train_sentences": 2,
                "rehydrated_train_sentence_slots": 4,
            }
        )
        report["targets"]["tokenizer_sentence_slot_gate"] = gate
        report["outputs"]["train.jsonl"].update(
            {"sha256": sha256_file(train_path), "bytes": train_path.stat().st_size}
        )
        manifest = json.loads(
            (root / "bootstrap_manifest.json").read_text(encoding="utf-8")
        )
        manifest["provenance"].update(
            {
                "tokenizer_acquisition_plan_file_sha256": plan_file_sha256,
                "tokenizer_acquisition_plan_payload_sha256": plan["plan_sha256"],
                "incremental_mode": "verified_full_deterministic_rebuild",
            }
        )
        manifest["languages"]["eng_Latn"]["acquisition"][
            "tokenizer_sentence_slot_gate"
        ] = gate
        summary = json.loads(
            (root / "bootstrap_summary.json").read_text(encoding="utf-8")
        )
        self._resign_bootstrap_fixture(
            root,
            report=report,
            manifest=manifest,
            summary=summary,
        )
        return plan_path, plan

    def test_bootstrap_audit_hashes_retained_split_files_and_rejects_corruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, validation_path = self._bootstrap_fixture(root)
            _manifest, _summary, audit = _audit_bootstrap(
                root=root,
                label="fixture",
                languages=["eng_Latn"],
                source_lock_sha256="a" * 64,
                language_map_file_sha256="b" * 64,
                rehydration_weights_sha256="c" * 64,
            )
            self.assertTrue(audit["retained_split_files_hash_verified"])
            self.assertEqual(audit["retained_split_file_count"], 3)
            self.assertEqual(
                audit["validation_candidate_contract"]["candidate_fraction"], 0.30
            )
            self.assertEqual(
                audit["validation_candidate_contract"]["final_target_fraction"],
                0.20,
            )
            self.assertTrue(
                audit["validation_candidate_contract"]["rows_independently_verified"]
            )

            original = train_path.read_bytes()
            train_path.write_bytes(original + b"corrupt")
            with self.assertRaisesRegex(ReleaseAuditError, "size mismatch"):
                _audit_bootstrap(
                    root=root,
                    label="fixture",
                    languages=["eng_Latn"],
                    source_lock_sha256="a" * 64,
                    language_map_file_sha256="b" * 64,
                    rehydration_weights_sha256="c" * 64,
                )
            train_path.write_bytes(original)
            validation_path.unlink()
            with self.assertRaisesRegex(
                ReleaseAuditError, "missing fixture/eng_Latn/validation"
            ):
                _audit_bootstrap(
                    root=root,
                    label="fixture",
                    languages=["eng_Latn"],
                    source_lock_sha256="a" * 64,
                    language_map_file_sha256="b" * 64,
                    rehydration_weights_sha256="c" * 64,
                )

    def test_bootstrap_audit_requires_explicit_candidate_fraction_and_legacy_agreement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._bootstrap_fixture(root)
            report = json.loads(
                (root / "eng_Latn" / "bootstrap_report.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (root / "bootstrap_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (root / "bootstrap_summary.json").read_text(encoding="utf-8")
            )
            report["run_config"]["settings"]["validation_fraction"] = 0.20
            self._resign_bootstrap_fixture(
                root, report=report, manifest=manifest, summary=summary
            )
            with self.assertRaisesRegex(ReleaseAuditError, "candidate-fraction"):
                _audit_bootstrap(
                    root=root,
                    label="fixture",
                    languages=["eng_Latn"],
                    source_lock_sha256="a" * 64,
                    language_map_file_sha256="b" * 64,
                    rehydration_weights_sha256="c" * 64,
                )

    def test_bootstrap_audit_recounts_candidate_membership_and_multipliers(self) -> None:
        mutations = {
            "rehydrated": (
                lambda row, _train: row["lineage"].__setitem__(
                    "rehydration_multiplier", 2
                ),
                "candidate row is rehydrated",
            ),
            "missing preserved multiplier": (
                lambda row, _train: row["lineage"].pop(
                    "train_rehydration_multiplier"
                ),
                "positive preserved train multiplier",
            ),
            "wrong membership": (
                lambda row, train: row.__setitem__(
                    "document_id", train["document_id"]
                ),
                "seed-42 validation candidate membership",
            ),
        }
        for name, (mutate, expected_error) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                train_path, validation_path = self._bootstrap_fixture(root)
                train = json.loads(train_path.read_text(encoding="utf-8"))
                candidate = json.loads(validation_path.read_text(encoding="utf-8"))
                mutate(candidate, train)
                validation_path.write_text(
                    json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8"
                )
                report = json.loads(
                    (root / "eng_Latn" / "bootstrap_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                report["outputs"]["validation_reserve.jsonl"].update(
                    {
                        "sha256": sha256_file(validation_path),
                        "bytes": validation_path.stat().st_size,
                    }
                )
                manifest = json.loads(
                    (root / "bootstrap_manifest.json").read_text(encoding="utf-8")
                )
                summary = json.loads(
                    (root / "bootstrap_summary.json").read_text(encoding="utf-8")
                )
                self._resign_bootstrap_fixture(
                    root, report=report, manifest=manifest, summary=summary
                )
                with self.assertRaisesRegex(ReleaseAuditError, expected_error):
                    _audit_bootstrap(
                        root=root,
                        label="fixture",
                        languages=["eng_Latn"],
                        source_lock_sha256="a" * 64,
                        language_map_file_sha256="b" * 64,
                        rehydration_weights_sha256="c" * 64,
                    )

    def test_tokenizer_bootstrap_audit_recounts_slots_and_gates_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path, plan = self._exact_slot_fixture(root)

            def audit() -> dict:
                _manifest, _summary, result = _audit_bootstrap(
                    root=root,
                    label="tokenizer acquisition",
                    languages=["eng_Latn"],
                    source_lock_sha256="a" * 64,
                    language_map_file_sha256="b" * 64,
                    rehydration_weights_sha256="c" * 64,
                    expected_acquisition_plan=(
                        sha256_file(plan_path),
                        plan["plan_sha256"],
                    ),
                    tokenizer_acquisition_plan=plan,
                    require_incremental=True,
                )
                return result

            gate_audit = audit()["tokenizer_sentence_slot_gate"]
            self.assertEqual(gate_audit["required_rehydrated_sentence_slots"], 4)
            self.assertEqual(gate_audit["actual_rehydrated_sentence_slots"], 4)
            self.assertTrue(gate_audit["independent_train_jsonl_recount"])

            # Re-sign a self-consistent but false report/manifest count.  The
            # release audit must use the train JSONL, not trust those counters.
            report_path = root / "eng_Latn" / "bootstrap_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (root / "bootstrap_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (root / "bootstrap_summary.json").read_text(encoding="utf-8")
            )
            report["counts"]["rehydrated_train_sentence_slots"] = 5
            report["targets"]["tokenizer_sentence_slot_gate"][
                "actual_rehydrated_sentence_slots"
            ] = 5
            manifest["languages"]["eng_Latn"]["acquisition"][
                "tokenizer_sentence_slot_gate"
            ]["actual_rehydrated_sentence_slots"] = 5
            self._resign_bootstrap_fixture(
                root,
                report=report,
                manifest=manifest,
                summary=summary,
            )
            with self.assertRaisesRegex(
                ReleaseAuditError, "exact rehydrated-slot recount drifted"
            ):
                audit()

            # Restore exact observed accounting, then raise the frozen quota.
            # A shortfall is invalid until true pinned-source exhaustion is
            # explicitly and consistently proven.
            plan["languages"]["eng_Latn"]["planned_sentence_count"] = 5
            plan["languages"]["eng_Latn"]["acquisition"][
                "required_rehydrated_sentence_slots"
            ] = 5
            plan.pop("plan_sha256")
            plan["plan_sha256"] = canonical_json_sha256(plan)
            _write(plan_path, plan)
            target = tokenizer_acquisition_target_contract(
                plan=plan,
                plan_file_sha256=sha256_file(plan_path),
                language="eng_Latn",
            )
            report["run_config"]["tokenizer_sentence_acquisition_target"] = target
            report["counts"]["rehydrated_train_sentence_slots"] = 4
            report_gate = report["targets"]["tokenizer_sentence_slot_gate"]
            report_gate.update(
                {
                    "required_rehydrated_sentence_slots": 5,
                    "actual_rehydrated_sentence_slots": 4,
                    "shortfall_slots": 1,
                    "quota_satisfied": False,
                }
            )
            manifest["provenance"].update(
                {
                    "tokenizer_acquisition_plan_file_sha256": sha256_file(plan_path),
                    "tokenizer_acquisition_plan_payload_sha256": plan["plan_sha256"],
                }
            )
            manifest["languages"]["eng_Latn"]["acquisition"][
                "tokenizer_sentence_slot_gate"
            ] = dict(report_gate)
            self._resign_bootstrap_fixture(
                root,
                report=report,
                manifest=manifest,
                summary=summary,
            )
            with self.assertRaisesRegex(
                ReleaseAuditError, "non-exhausted tokenizer slot shortfall"
            ):
                audit()

            report["targets"]["source_exhausted"] = True
            report["acquisition_status"] = "source_exhausted"
            report_gate.update(
                {
                    "true_pinned_source_exhausted": True,
                    "true_source_exhausted_repetition_allowed": True,
                }
            )
            manifest["languages"]["eng_Latn"]["acquisition"].update(
                {
                    "complete_source_scan": True,
                    "tokenizer_sentence_slot_gate": dict(report_gate),
                }
            )
            self._resign_bootstrap_fixture(
                root,
                report=report,
                manifest=manifest,
                summary=summary,
            )
            exhausted = audit()["tokenizer_sentence_slot_gate"]
            self.assertEqual(exhausted["true_source_exhausted_languages"], ["eng_Latn"])

    def test_materialized_inventory_reproduces_frozen_k200_top150(self) -> None:
        k50 = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/language_groupings/scaling_fw2_v2_k050.json"
            ).read_text(encoding="utf-8")
        )["0"]["languages"]
        k200 = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/language_groupings/scaling_fw2_v2_k200.json"
            ).read_text(encoding="utf-8")
        )["0"]["languages"]
        audit = _audit_materialized_inventory_selection(
            inventory_path=Path(__file__).resolve().parents[1]
            / "configs/data_sources/fineweb2-language-distribution.csv",
            k50_languages=k50,
            k200_languages=k200,
        )
        self.assertEqual(audit["added_language_count"], 150)
        self.assertEqual(audit["minimum_selected_train_documents"], 9003)
        self.assertEqual(audit["cutoff_language"], "zea_Latn")

    def _sampling_fixture(
        self, root: Path
    ) -> tuple[
        Path, dict[str, list[str]], dict[str, str], Path, dict, Path, dict, Path
    ]:
        tokenizer = root / "tokenizer.model"
        tokenizer.write_bytes(b"tokenizer")
        inventory = root / "inventory.csv"
        inventory.write_text("subset,split\n", encoding="utf-8")
        weights = root / "weights.json"
        weights.write_text("{}\n", encoding="utf-8")
        source_manifest = {
            "schema_version": 1,
            "source_lock_sha256": "a" * 64,
            "artifacts": {
                "inventory": {"path": str(inventory), "sha256": sha256_file(inventory)},
                "rehydration_weights": {
                    "path": str(weights),
                    "sha256": sha256_file(weights),
                },
            },
        }
        source_manifest["manifest_sha256"] = canonical_json_sha256(source_manifest)
        source_manifest_path = root / "source.json"
        _write(source_manifest_path, source_manifest)
        compact = {
            "available_utf8_bytes": 100_000,
            "available_documents": 1_000,
            "calibration_utf8_bytes": 1_000,
            "calibration_tokens": 500,
            "calibration_documents": 10,
            "calibration_rehydrated_tokens": 750,
            "validation_fraction": 0.2,
        }
        calibration = {
            "schema_version": 1,
            "tool_id": "fineweb_ab_v2_token_calibration_v1",
            "status": "complete",
            "languages": {"eng_Latn": compact, "deu_Latn": compact},
            "language_audits": {},
            "provenance": {
                "tokenizer_model_sha256": sha256_file(tokenizer),
                "source_lock_sha256": "a" * 64,
                "fineweb2_inventory_sha256": sha256_file(inventory),
            },
            "audit": {},
        }
        calibration["artifact_sha256"] = canonical_json_sha256(calibration)
        calibration_path = root / "calibration.json"
        _write(calibration_path, calibration)
        map1 = root / "k1.json"
        map2 = root / "k2.json"
        _write(map1, {"0": {"languages": ["eng_Latn"]}})
        _write(map2, {"0": {"languages": ["eng_Latn", "deu_Latn"]}})
        plan = build_plan_artifact(
            calibration_path=calibration_path,
            language_map_args=[f"K1={map1}", f"K2={map2}"],
            source_manifest_path=source_manifest_path,
            tokenizer_model_path=tokenizer,
            target_train_tokens=EXPECTED_TRAIN_TOKENS,
            alpha=0.3,
            validation_cap_tokens=20_000_000,
        )
        plan["tiers"]["K200"] = plan["tiers"].pop("K2")
        plan["provenance"]["language_map_file_sha256"]["K200"] = plan["provenance"][
            "language_map_file_sha256"
        ].pop("K2")
        plan.pop("plan_sha256")
        plan["plan_sha256"] = canonical_json_sha256(plan)
        plan_path = root / "plan.json"
        _write(plan_path, plan)
        return (
            plan_path,
            {"K1": ["eng_Latn"], "K200": ["eng_Latn", "deu_Latn"]},
            {"K1": sha256_file(map1), "K200": sha256_file(map2)},
            calibration_path,
            calibration,
            source_manifest_path,
            source_manifest,
            tokenizer,
        )

    def test_sampling_audit_proves_exact_32b_exposure_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._sampling_fixture(Path(tmp))
            plan, audit = _audit_sampling_plan(
                path=fixture[0],
                language_sets=fixture[1],
                language_map_hashes=fixture[2],
                calibration_path=fixture[3],
                calibration=fixture[4],
                source_manifest_path=fixture[5],
                source_manifest=fixture[6],
                tokenizer_path=fixture[7],
            )
            self.assertEqual(
                audit["exact_exposure_tokens_by_tier"],
                {"K1": EXPECTED_TRAIN_TOKENS, "K200": EXPECTED_TRAIN_TOKENS},
            )
            plan["tiers"]["K200"]["target_exposure_tokens"]["deu_Latn"] -= 1
            plan.pop("plan_sha256")
            plan["plan_sha256"] = canonical_json_sha256(plan)
            _write(fixture[0], plan)
            with self.assertRaisesRegex(ReleaseAuditError, "exactly to 32B"):
                _audit_sampling_plan(
                    path=fixture[0],
                    language_sets=fixture[1],
                    language_map_hashes=fixture[2],
                    calibration_path=fixture[3],
                    calibration=fixture[4],
                    source_manifest_path=fixture[5],
                    source_manifest=fixture[6],
                    tokenizer_path=fixture[7],
                )

    def _resign_corpus_fixture(self, release_dir: Path, manifest: dict) -> None:
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        manifest_path = release_dir / "corpus_manifest.json"
        _write(manifest_path, manifest)
        hash_manifest = {
            "schema_version": 1,
            "corpus_manifest_payload_sha256": manifest["manifest_sha256"],
            "files": {"corpus_manifest.json": sha256_file(manifest_path)},
        }
        hash_manifest["hash_manifest_payload_sha256"] = canonical_json_sha256(
            hash_manifest
        )
        _write(release_dir / "hash_manifest.json", hash_manifest)

    def _corpus_candidate_fixture(self, root: Path) -> tuple:
        language = "eng_Latn"
        release_dir = root / "release"
        release_dir.mkdir(parents=True)
        tokenizer_path = root / "tokenizer.model"
        tokenizer_path.write_bytes(b"frozen-tokenizer")
        tokenizer_hash = sha256_file(tokenizer_path)
        plan = {
            "plan_sha256": "d" * 64,
            "physical_pool": {language: {"physical_target_tokens": 4}},
            "tiers": {
                "K1": {
                    "languages": [language],
                    "target_exposure_tokens": {language: 8},
                }
            },
        }
        plan_path = root / "sampling_plan.json"
        _write(plan_path, plan)
        production_manifest = {"provenance": {"fixture": True}}
        production_manifest_path = root / "production_manifest.json"
        _write(production_manifest_path, production_manifest)

        ordinary = self._bootstrap_row(
            language=language,
            split="train",
            source_id="ordinary-train",
            text="ordinary train",
            train_multiplier=1,
        )
        selected_validation = self._bootstrap_row(
            language=language,
            split="validation_reserve",
            source_id="selected-validation",
            text="selected validation",
            train_multiplier=1,
        )
        reclaimed = self._bootstrap_row(
            language=language,
            split="validation_reserve",
            source_id="reclaimed-candidate",
            text="reclaimed candidate",
            train_multiplier=2,
        )

        def emitted(
            source: dict, *, replica: int, token_count: int, reclaimed_row: bool
        ) -> dict:
            row = json.loads(json.dumps(source))
            original = row["document_id"]
            row["original_document_id"] = original
            row["document_id"] = (
                original
                if replica == 0
                else f"{original}#rehydration-{replica:04d}"
            )
            row["lineage"]["rehydration_replica_index"] = replica
            if reclaimed_row:
                row["artifact_role"] = "train"
                row["lineage"].update(
                    {
                        "split": "train",
                        "rehydration_multiplier": row["lineage"][
                            "train_rehydration_multiplier"
                        ],
                        "validation_candidate_original_split": "validation_reserve",
                        "validation_candidate_reclaimed": True,
                    }
                )
            row["ab_v2_tokenization"] = {
                "token_count_including_eod": token_count,
                "tokenizer_model_sha256": tokenizer_hash,
                "append_eod": True,
            }
            return row

        train_rows = [
            emitted(ordinary, replica=0, token_count=1, reclaimed_row=False),
            emitted(reclaimed, replica=0, token_count=2, reclaimed_row=True),
            emitted(reclaimed, replica=1, token_count=2, reclaimed_row=True),
        ]
        validation_rows = [
            emitted(
                selected_validation,
                replica=0,
                token_count=2,
                reclaimed_row=False,
            )
        ]
        train_path = release_dir / "train" / f"{language}.jsonl"
        validation_path = release_dir / "validation" / f"{language}.jsonl"
        train_path.parent.mkdir(parents=True)
        validation_path.parent.mkdir(parents=True)
        train_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
            encoding="utf-8",
        )
        validation_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in validation_rows
            ),
            encoding="utf-8",
        )
        unigram_path = release_dir / "unigram_counts" / f"{language}.json"
        _write(unigram_path, {})

        def artifact(path: Path) -> dict:
            return {
                "path": str(path.relative_to(release_dir)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

        candidate_audit = {
            "status": "adequate",
            "candidate_adequate": True,
            "selection_precedes_train_selection": True,
            "candidate_documents": 2,
            "candidate_unique_tokens_including_eod": 4,
            "candidate_train_rehydrated_tokens_if_all_reclaimed": 6,
            "candidate_train_emitted_documents_if_all_reclaimed": 3,
            "selected_validation_original_documents": 1,
            "selected_validation_tokens_including_eod": 2,
            "unused_candidate_documents_reclassified_to_train": 1,
            "unused_candidate_unique_tokens_reclassified_to_train": 2,
            "unused_candidate_rehydrated_tokens_available_to_train": 4,
            "unused_candidate_emitted_documents_available_to_train": 2,
            "selected_reclaimed_train_original_documents": 1,
            "selected_reclaimed_train_unique_tokens_including_eod": 2,
            "selected_reclaimed_train_tokens_including_rehydration_and_eod": 4,
            "selected_reclaimed_train_emitted_documents": 2,
            "preserved_train_multiplier_field": (
                "lineage.train_rehydration_multiplier"
            ),
            "selected_validation_rehydration_multiplier": 1,
            "declared_candidate_fraction": 0.30,
            "final_validation_target_fraction": 0.20,
            "observed_candidate_token_fraction": 0.4,
        }
        manifest = {
            "schema_version": 1,
            "release_id": "fineweb_ab_v2_finalized",
            "release_status": "ready",
            "ready": True,
            "blocking_languages": {
                "needs_incremental_source_docs": [],
                "needs_incremental_validation_inventory": [],
                "needs_incremental_validation_docs": [],
            },
            "selection_contract": {
                "unit": "whole_original_document",
                "sampling_seed": 42,
                "priority": (
                    "ascending_sha256(sampling_seed, split_namespace, "
                    "original_document_id)"
                ),
                "selection_namespaces": {
                    "train": "physical-train-document-selection-v1",
                    "validation": "validation-panel-document-selection-v1",
                    "validation_reserve_membership": "validation-reserve-v1",
                },
                "train_rehydration": "all_declared_replicas_are_atomic",
                "validation_candidate_reserve": {
                    "declared_fraction": 0.30,
                    "final_target_fraction": 0.20,
                    "selection_order": "final_validation_before_train",
                    "unused_candidate_policy": (
                        "reclassify_to_train_with_preserved_train_multiplier"
                    ),
                    "preserved_train_multiplier_field": (
                        "lineage.train_rehydration_multiplier"
                    ),
                },
                "validation_rehydration": "forbidden",
                "train_validation_disjointness": {
                    "original_document_id_overlap": 0
                },
                "max_train_shortfall_fraction": 0.0,
            },
            "inputs": {
                "sampling_plan": {
                    "file_sha256": sha256_file(plan_path),
                    "payload_sha256": plan["plan_sha256"],
                },
                "bootstrap_manifest": {
                    "sha256": sha256_file(production_manifest_path),
                    "provenance": production_manifest["provenance"],
                },
                "tokenizer_model": {
                    "sha256": tokenizer_hash,
                    "vocabulary_size": 64_000,
                },
            },
            "languages": {
                language: {
                    "exact_unique_language_tokens_including_eod": 10,
                    "observed_unique_language_tokens_lower_bound_including_eod": 10,
                    "source_coverage": {"complete_source_scan": True},
                    "bootstrap": {"train": {}, "validation_reserve": {}},
                    "validation_candidate_reserve": candidate_audit,
                    "train": {
                        "status": "ready",
                        "target_tokens": 4,
                        "available_tokens": 5,
                        "selected_tokens": 5,
                        "selected_original_documents": 2,
                        "emitted_documents": 3,
                        "shortfall_tokens": 0,
                        "overshoot_tokens": 1,
                        "shortfall_fraction": 0.0,
                        "max_allowed_shortfall_fraction": 0.0,
                        "output": artifact(train_path),
                        "unigram_counts": artifact(unigram_path),
                    },
                    "validation": {
                        "status": "ready",
                        "target_tokens": 2,
                        "available_tokens": 4,
                        "selected_tokens": 2,
                        "selected_original_documents": 1,
                        "emitted_documents": 1,
                        "shortfall_tokens": 0,
                        "overshoot_tokens": 0,
                        "target_rule": (
                            "min(20,000,000, floor(0.20 * "
                            "exact_unique_language_tokens))"
                        ),
                        "target_fraction": 0.20,
                        "candidate_fraction": 0.30,
                        "candidate_adequate": True,
                        "quota_exact": True,
                        "never_rehydrated": True,
                        "complete_source_scan": True,
                        "output": artifact(validation_path),
                    },
                    "exposure_pass_audit": {
                        "K1": {
                            "target_exposure_tokens": 8,
                            "physical_train_tokens": 5,
                            "effective_passes": 1.6,
                            "full_passes": 1,
                            "partial_pass_tokens": 3,
                        }
                    },
                }
            },
        }
        self._resign_corpus_fixture(release_dir, manifest)
        return (
            release_dir,
            plan_path,
            plan,
            production_manifest_path,
            production_manifest,
            tokenizer_path,
            manifest,
        )

    def test_corpus_audit_proves_candidate_reclamation_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._corpus_candidate_fixture(Path(tmp))

            def audit() -> dict:
                _manifest, result = _audit_corpus_release(
                    release_dir=fixture[0],
                    languages=["eng_Latn"],
                    plan_path=fixture[1],
                    plan=fixture[2],
                    production_manifest_path=fixture[3],
                    production_manifest=fixture[4],
                    tokenizer_path=fixture[5],
                    hash_cache={},
                )
                return result

            result = audit()
            self.assertTrue(result["exact_physical_targets"])
            self.assertEqual(result["split_overlap"], 0)
            self.assertEqual(result["split_original_ids_indexed"], 3)
            self.assertEqual(
                result["split_disjointness_proof"],
                "independent_disk_backed_sqlite_primary_key_over_original_document_id",
            )
            self.assertEqual(
                list(fixture[0].parent.glob(".fineweb-v2-split-audit-*")), []
            )
            manifest = fixture[6]
            candidate = manifest["languages"]["eng_Latn"][
                "validation_candidate_reserve"
            ]
            candidate[
                "selected_reclaimed_train_tokens_including_rehydration_and_eod"
            ] += 1
            self._resign_corpus_fixture(fixture[0], manifest)
            with self.assertRaisesRegex(
                ReleaseAuditError, "selected reclaimed-train output accounting"
            ):
                audit()

    def test_corpus_audit_independently_rejects_original_id_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._corpus_candidate_fixture(Path(tmp))
            release_dir = fixture[0]
            manifest = fixture[6]
            language = "eng_Latn"
            train_path = release_dir / "train" / f"{language}.jsonl"
            validation_path = release_dir / "validation" / f"{language}.jsonl"
            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]

            # Replace the selected validation document with the exact original
            # candidate already reclaimed into train.  All aggregate counters
            # remain plausible, so only an independent ID-set proof catches it.
            overlapping_validation = json.loads(json.dumps(train_rows[1]))
            original = overlapping_validation["original_document_id"]
            overlapping_validation["artifact_role"] = "validation_reserve"
            overlapping_validation["document_id"] = original
            lineage = overlapping_validation["lineage"]
            lineage["split"] = "validation_reserve"
            lineage["rehydration_multiplier"] = 1
            lineage["rehydration_replica_index"] = 0
            lineage.pop("validation_candidate_original_split", None)
            lineage.pop("validation_candidate_reclaimed", None)
            validation_path.write_text(
                json.dumps(overlapping_validation, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output = manifest["languages"][language]["validation"]["output"]
            output["sha256"] = sha256_file(validation_path)
            output["size_bytes"] = validation_path.stat().st_size
            self._resign_corpus_fixture(release_dir, manifest)

            with self.assertRaisesRegex(
                ReleaseAuditError,
                "train/validation original_document_id overlap",
            ):
                _audit_corpus_release(
                    release_dir=fixture[0],
                    languages=[language],
                    plan_path=fixture[1],
                    plan=fixture[2],
                    production_manifest_path=fixture[3],
                    production_manifest=fixture[4],
                    tokenizer_path=fixture[5],
                    hash_cache={},
                )
            self.assertEqual(
                list(release_dir.parent.glob(".fineweb-v2-split-audit-*")), []
            )

    def test_corpus_audit_rejects_candidate_contract_or_inadequacy(self) -> None:
        for mutation, expected in (
            (
                lambda manifest: manifest["selection_contract"][
                    "validation_candidate_reserve"
                ].__setitem__("declared_fraction", 0.20),
                "candidate/reclamation contract",
            ),
            (
                lambda manifest: manifest["languages"]["eng_Latn"][
                    "validation_candidate_reserve"
                ].__setitem__("candidate_adequate", False),
                "candidate adequacy contract",
            ),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                fixture = self._corpus_candidate_fixture(Path(tmp))
                mutation(fixture[6])
                self._resign_corpus_fixture(fixture[0], fixture[6])
                with self.assertRaisesRegex(ReleaseAuditError, expected):
                    _audit_corpus_release(
                        release_dir=fixture[0],
                        languages=["eng_Latn"],
                        plan_path=fixture[1],
                        plan=fixture[2],
                        production_manifest_path=fixture[3],
                        production_manifest=fixture[4],
                        tokenizer_path=fixture[5],
                        hash_cache={},
                    )

    def test_prefix_audit_requires_uint16_and_exact_counts_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "language"
            lengths = [3, 5]
            idx = Path(str(prefix) + ".idx")
            idx.write_bytes(
                b"MMIDIDX\x00\x00"
                + struct.pack("<Q", 1)
                + struct.pack("<B", 8)
                + struct.pack("<QQ", len(lengths), len(lengths) + 1)
                + b"".join(struct.pack("<i", value) for value in lengths)
            )
            binary = Path(str(prefix) + ".bin")
            binary.write_bytes(b"\x00" * (sum(lengths) * 2))
            raw = {
                "prefix": str(prefix),
                "bin_sha256": sha256_file(binary),
                "bin_size_bytes": binary.stat().st_size,
                "idx_sha256": sha256_file(idx),
                "idx_size_bytes": idx.stat().st_size,
                "sequences": 2,
                "documents": 2,
                "tokens_including_eod": 8,
                "dtype_code": 8,
            }
            self.assertEqual(
                _verify_prefix(raw, hash_cache={}, label="fixture"),
                {
                    "sequences": 2,
                    "documents": 2,
                    "tokens_including_eod": 8,
                    "dtype_code": 8,
                },
            )
            raw["bin_sha256"] = "0" * 64
            with self.assertRaisesRegex(ReleaseAuditError, "bin hash mismatch"):
                _verify_prefix(raw, hash_cache={}, label="fixture")

    def test_path_audit_rejects_v1_or_another_pipeline_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs" / "pipeline_runs" / "h100_ab_v2"
            root.mkdir(parents=True)
            old = root.parent / "h100_ab_v1" / "corpus" / "tokenized"
            with self.assertRaisesRegex(ReleaseAuditError, "v1 path"):
                _audit_embedded_paths(
                    {"artifact": {"input_path": str(old)}},
                    release_root=root,
                )
            other = root.parent / "h100_ab_v3" / "corpus" / "tokenized"
            with self.assertRaisesRegex(ReleaseAuditError, "another pipeline release"):
                _audit_embedded_paths(
                    {"artifact": {"input_path": str(other)}},
                    release_root=root,
                )
            self.assertEqual(
                _audit_embedded_paths(
                    {"artifact": {"provenance_path": str(other)}},
                    release_root=root,
                    allowed_predecessor_root=root.parent / "h100_ab_v3",
                ),
                1,
            )
            with self.assertRaisesRegex(ReleaseAuditError, "another pipeline release"):
                _audit_embedded_paths(
                    {"artifact": {"input_path": str(other)}},
                    release_root=root,
                    allowed_predecessor_root=root.parent / "h100_ab_v3",
                )

    def test_predecessor_seal_rejects_inode_metadata_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "predecessor"
            artifact = root / "corpus" / "tokenized" / "fixture.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"immutable")
            artifact.chmod(0o444)
            seal = {
                "files": {
                    "corpus/tokenized/fixture.bin": {
                        "stat": _sealed_stat(artifact),
                        "sha256": sha256_file(artifact),
                        "rehash_on_verify": False,
                    }
                },
                "directories": {
                    "corpus/tokenized": {"stat": _sealed_stat(artifact.parent)}
                },
            }
            _verify_predecessor_seal_stats(seal, predecessor_root=root)
            artifact.chmod(0o400)
            with self.assertRaisesRegex(ReleaseAuditError, "stat drift"):
                _verify_predecessor_seal_stats(seal, predecessor_root=root)


if __name__ == "__main__":
    unittest.main()
