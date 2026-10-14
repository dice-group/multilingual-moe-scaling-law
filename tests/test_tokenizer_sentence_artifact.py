from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.sampling.tools.bootstrap_fineweb_v2 import (  # noqa: E402
    BootstrapSettings,
    bootstrap_release,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    canonical_json_sha256,
)
from multilingual_megatron.pipeline.stages.tokenizer_training.build_sentence_artifact import (  # noqa: E402
    INTERLEAVER_ID,
    SPLITTER_ID,
    SentenceArtifactError,
    build_sentence_acquisition_plan,
    build_sentence_artifact,
    main,
    split_sentence_like,
    verify_acquisition_plan,
    write_frozen_acquisition_plan,
)
from multilingual_megatron.pipeline.stages.tokenizer_training.sentencepiece_v2 import (  # noqa: E402
    inspect_sentence_jsonl,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build_bootstrap_fixture(
    root: Path,
    *,
    documents_per_language: int = 10,
    text_factory: Callable[[str, int], str] | None = None,
    validation_fraction: float = 1e-12,
) -> tuple[Path, Path]:
    languages = ("eng_Latn", "deu_Latn")
    weights_path = root / "materialized" / "fineweb2-up-to-10-reps.json"
    _write_json(weights_path, {"deu_Latn": {"1": 2}})
    inventory_path = weights_path.parent / "fineweb2-language-distribution.csv"
    inventory_path.write_text(
        "subset,split,documents,utf8_bytes\n"
        f"deu_Latn,train,{documents_per_language},{documents_per_language * 100000}\n",
        encoding="utf-8",
    )
    source_lock_path = root / "source-lock.json"
    _write_json(
        source_lock_path,
        {
            "schema_version": 1,
            "source_lock_id": "sentence-fixture",
            "non_english_corpus": {
                "dataset": "fixture/fineweb-2",
                "dataset_revision": "a" * 40,
                "language_inventory": {"sha256": _sha256(inventory_path)},
                "rehydration": {
                    "enabled": True,
                    "identity_overrides": {},
                    "missing_weights_policy": (
                        "fail_closed_except_explicit_identity_overrides"
                    ),
                    "weights_policy": "up_to_10_reps",
                    "weights_sha256": _sha256(weights_path),
                },
            },
            "english_corpus": {
                "dataset": "fixture/fineweb",
                "dataset_revision": "b" * 40,
                "subset": "eng_Latn",
                "availability": {"reported_documents": documents_per_language},
            },
        },
    )
    language_map_path = root / "language-map.json"
    _write_json(
        language_map_path,
        {
            "schema_version": 2,
            "language_set_id": "sentence-fixture-v2",
            "0": {"group": "fixture", "languages": list(languages)},
        },
    )
    for language in languages:
        rows = [
            {
                "id": f"{language}-doc-{index:03d}",
                "text": (
                    text_factory(language, index)
                    if text_factory is not None
                    else (
                        f"{language} sentence {index}. "
                        + (
                            "日本語の二文目です。"
                            if language == "eng_Latn"
                            else "Zweiter Satz!"
                        )
                    )
                ),
                "url": f"https://example.test/{language}/{index}",
                **({} if language == "eng_Latn" else {"minhash_cluster_size": 1}),
            }
            for index in range(documents_per_language)
        ]
        parquet_path = root / "parquet" / language / "part-000.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    bootstrap_root = root / "bootstrap"
    bootstrap_release(
        source_lock_path=source_lock_path,
        rehydration_weights_path=weights_path,
        language_map_path=language_map_path,
        output_root=bootstrap_root,
        settings=BootstrapSettings(
            seed=42,
            validation_fraction=validation_fraction,
            target_unique_documents=10,
            target_utf8_bytes=1,
            calibration_documents=2,
        ),
        selected_language="all",
        local_input_path=root / "parquet",
    )
    return bootstrap_root / "bootstrap_summary.json", language_map_path


def _refresh_bootstrap_hashes(root: Path, language: str) -> None:
    report_path = root / "bootstrap" / language / "bootstrap_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train_path = root / "bootstrap" / language / "train.jsonl"
    report["outputs"]["train.jsonl"]["sha256"] = _sha256(train_path)
    report["outputs"]["train.jsonl"]["bytes"] = train_path.stat().st_size
    report.pop("report_payload_sha256")
    report["report_payload_sha256"] = canonical_json_sha256(report)
    _write_json(report_path, report)
    summary_path = root / "bootstrap" / "bootstrap_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["report_payload_sha256"][language] = report["report_payload_sha256"]
    summary.pop("summary_sha256")
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _write_json(summary_path, summary)


class TokenizerSentenceArtifactTest(unittest.TestCase):
    def test_unicode_splitter_uses_terminal_close_and_bounded_chunks(self) -> None:
        text = "Hello. “Next?” 日本語。次！ 3.14 stays. abcdefghijk"
        self.assertEqual(
            list(split_sentence_like(text, max_chars=8)),
            ["Hello.", "“Next?”", "日本語。", "次！", "3.14 sta", "ys.", "abcdefgh", "ijk"],
        )

    def test_tokenizer_planning_rejects_candidate_fraction_alias_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap_summary, language_map = _build_bootstrap_fixture(root)
            report_path = root / "bootstrap" / "eng_Latn" / "bootstrap_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["run_config"]["settings"]["validation_fraction"] = 0.20
            report_path.write_text(
                json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
            )
            _refresh_bootstrap_hashes(root, "eng_Latn")

            with self.assertRaisesRegex(
                SentenceArtifactError, "validation candidate contract"
            ):
                build_sentence_acquisition_plan(
                    input_manifest_path=bootstrap_summary,
                    language_map_path=language_map,
                    target_sentences=20,
                    alpha=0.3,
                    seed=42,
                    test_mode=True,
                )

    def test_bootstrap_to_sentence_artifact_is_exact_deterministic_and_never_reads_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap_summary, language_map = _build_bootstrap_fixture(root)

            # Deliberately corrupt validation after bootstrap.  The tokenizer
            # builder may inspect only its declared path, never its contents.
            for language in ("eng_Latn", "deu_Latn"):
                (root / "bootstrap" / language / "validation_reserve.jsonl").write_bytes(b"not-json\n")

            reports = [
                build_sentence_artifact(
                    input_manifest_path=bootstrap_summary,
                    language_map_path=language_map,
                    output_jsonl_path=root / "sentences-a.jsonl",
                    output_manifest_path=root / "sentences-a.manifest.json",
                    target_sentences=25,
                    alpha=0.3,
                    max_sentence_chars=1_000,
                    test_mode=True,
                )
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--input-manifest",
                            str(bootstrap_summary),
                            "--language-map",
                            str(language_map),
                            "--output-jsonl",
                            str(root / "sentences-b.jsonl"),
                            "--output-manifest",
                            str(root / "sentences-b.manifest.json"),
                            "--target-sentences",
                            "25",
                            "--max-sentence-chars",
                            "1000",
                            "--test-mode",
                        ]
                    ),
                    0,
                )
            reports.append(
                json.loads((root / "sentences-b.manifest.json").read_text(encoding="utf-8"))
            )
            third = build_sentence_artifact(
                input_manifest_path=bootstrap_summary,
                language_map_path=language_map,
                output_jsonl_path=root / "sentences-seed-43.jsonl",
                output_manifest_path=root / "sentences-seed-43.manifest.json",
                target_sentences=25,
                alpha=0.3,
                seed=43,
                max_sentence_chars=1_000,
                test_mode=True,
            )

            first = reports[0]
            second = reports[1]
            self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])
            unhashed = dict(first)
            embedded_manifest_hash = unhashed.pop("manifest_sha256")
            self.assertEqual(embedded_manifest_hash, canonical_json_sha256(unhashed))
            self.assertEqual(
                (root / "sentences-a.jsonl").read_bytes(),
                (root / "sentences-b.jsonl").read_bytes(),
            )
            self.assertEqual(first["realized_sentence_count"], 25)
            self.assertEqual(first["totals"]["planned_sentence_count"], 25)
            self.assertEqual(first["totals"]["realized_sentence_count"], 25)
            self.assertEqual(first["inputs"]["validation_files_opened"], False)
            self.assertEqual(first["algorithms"]["splitter"]["id"], SPLITTER_ID)
            self.assertEqual(first["algorithms"]["interleaver"]["id"], INTERLEAVER_ID)
            self.assertEqual(set(first["inputs"]["bootstrap_reports"]), {"eng_Latn", "deu_Latn"})
            self.assertEqual(
                first["languages"]["deu_Latn"]["within_language_rehydrated_sampling_slots"],
                2 * first["languages"]["deu_Latn"]["train_input"]["unique_sentence_count"],
            )
            self.assertEqual(
                first["languages"]["eng_Latn"][
                    "estimated_available_post_split_unique_sentence_count"
                ],
                first["languages"]["deu_Latn"][
                    "estimated_available_post_split_unique_sentence_count"
                ],
            )
            self.assertEqual(first["languages"]["eng_Latn"]["alpha_probability"], 0.5)
            self.assertEqual(first["languages"]["deu_Latn"]["alpha_probability"], 0.5)
            self.assertEqual(first["languages"]["deu_Latn"]["planned_sentence_count"], 13)
            self.assertEqual(first["languages"]["eng_Latn"]["planned_sentence_count"], 12)
            self.assertFalse(
                first["languages"]["deu_Latn"]["sentence_availability"][
                    "rehydration_used_in_language_level_availability"
                ]
            )
            self.assertNotEqual(first["output"]["sha256"], third["output"]["sha256"])

            rows = _read_jsonl(root / "sentences-a.jsonl")
            self.assertEqual(len(rows), 25)
            self.assertTrue(
                all(
                    set(row) == {"doc_id", "language", "repeat_index", "sentence_index", "text"}
                    for row in rows
                )
            )
            for language, language_report in first["languages"].items():
                planned = language_report["planned_sentence_count"]
                available = language_report["within_language_rehydrated_sampling_slots"]
                self.assertEqual(language_report["realized_sentence_count"], planned)
                self.assertEqual(language_report["repeated_sentence_count"], max(0, planned - available))
                self.assertEqual(sum(row["language"] == language for row in rows), planned)

            inspect_sentence_jsonl(
                root / "sentences-a.jsonl",
                text_key="text",
                expected_sha256=first["output"]["sha256"],
                expected_sentence_count=25,
            )

    def test_two_pass_plan_expands_pool_and_forbids_artificial_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot_summary, language_map = _build_bootstrap_fixture(
                root, documents_per_language=100
            )
            plan = build_sentence_acquisition_plan(
                input_manifest_path=pilot_summary,
                language_map_path=language_map,
                target_sentences=200,
                alpha=0.3,
                seed=42,
                safety_margin=1.10,
                test_mode=True,
            )
            self.assertTrue(verify_acquisition_plan(plan))
            plan_path = root / "tokenizer-acquisition-plan.json"
            write_frozen_acquisition_plan(output_path=plan_path, plan=plan)
            self.assertTrue(
                all(
                    row["acquisition"]["target_unique_source_documents"] >= 10
                    for row in plan["languages"].values()
                )
            )

            expanded_root = root / "expanded-bootstrap"
            bootstrap_release(
                source_lock_path=root / "source-lock.json",
                rehydration_weights_path=root
                / "materialized"
                / "fineweb2-up-to-10-reps.json",
                language_map_path=language_map,
                output_root=expanded_root,
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=1e-12,
                    target_unique_documents=10,
                    target_utf8_bytes=1,
                    calibration_documents=2,
                ),
                selected_language="all",
                local_input_path=root / "parquet",
                tokenizer_acquisition_plan_path=plan_path,
                incremental_from_root=root / "bootstrap",
            )
            report = build_sentence_artifact(
                input_manifest_path=expanded_root / "bootstrap_summary.json",
                language_map_path=language_map,
                output_jsonl_path=root / "expanded-sentences.jsonl",
                output_manifest_path=root / "expanded-sentences.manifest.json",
                target_sentences=200,
                acquisition_plan_path=plan_path,
                alpha=0.3,
                seed=42,
                test_mode=True,
            )
            self.assertEqual(report["realized_sentence_count"], 200)
            self.assertTrue(
                all(
                    row["expanded_pool_gate"]["status"] == "sufficient"
                    for row in report["languages"].values()
                )
            )
            self.assertEqual(
                sum(row["repeated_sentence_count"] for row in report["languages"].values()),
                0,
            )

            with self.assertRaisesRegex(
                SentenceArtifactError, "was not acquired from the frozen tokenizer plan"
            ):
                build_sentence_artifact(
                    input_manifest_path=pilot_summary,
                    language_map_path=language_map,
                    output_jsonl_path=root / "pilot-must-not-cycle.jsonl",
                    output_manifest_path=root / "pilot-must-not-cycle.manifest.json",
                    target_sentences=200,
                    acquisition_plan_path=plan_path,
                    alpha=0.3,
                    seed=42,
                    test_mode=True,
                )

    def test_exact_slot_gate_streams_past_optimistic_document_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def variable_density(language: str, index: int) -> str:
                sentence_count = 10 if index < 10 else 1
                return " ".join(
                    f"{language} document {index} sentence {sentence}."
                    for sentence in range(sentence_count)
                )

            pilot_summary, language_map = _build_bootstrap_fixture(
                root,
                documents_per_language=300,
                text_factory=variable_density,
                validation_fraction=0.20,
            )
            plan = build_sentence_acquisition_plan(
                input_manifest_path=pilot_summary,
                language_map_path=language_map,
                target_sentences=400,
                alpha=0.3,
                seed=42,
                safety_margin=1.10,
                test_mode=True,
            )
            self.assertTrue(verify_acquisition_plan(plan))
            stale_estimate = plan["languages"]["eng_Latn"]["acquisition"][
                "target_unique_source_documents"
            ]
            self.assertLess(stale_estimate, 100)
            plan_path = root / "tokenizer-acquisition-plan.json"
            write_frozen_acquisition_plan(output_path=plan_path, plan=plan)

            expanded_root = root / "expanded-bootstrap"
            bootstrap_release(
                source_lock_path=root / "source-lock.json",
                rehydration_weights_path=root
                / "materialized"
                / "fineweb2-up-to-10-reps.json",
                language_map_path=language_map,
                output_root=expanded_root,
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=0.20,
                    target_unique_documents=10,
                    target_utf8_bytes=1,
                    calibration_documents=2,
                ),
                selected_language="all",
                local_input_path=root / "parquet",
                tokenizer_acquisition_plan_path=plan_path,
                incremental_from_root=root / "bootstrap",
            )
            english_report_path = expanded_root / "eng_Latn" / "bootstrap_report.json"
            english_report = json.loads(english_report_path.read_text(encoding="utf-8"))
            gate = english_report["targets"]["tokenizer_sentence_slot_gate"]
            self.assertGreater(
                english_report["counts"]["unique_source_documents"],
                stale_estimate,
            )
            self.assertGreaterEqual(
                gate["actual_rehydrated_sentence_slots"],
                gate["required_rehydrated_sentence_slots"],
            )
            self.assertEqual(gate["shortfall_slots"], 0)
            self.assertTrue(gate["quota_satisfied"])
            self.assertFalse(gate["true_pinned_source_exhausted"])
            self.assertEqual(
                gate["counter_id"],
                "train_split_sentence_like_times_rehydration_multiplier_v1",
            )
            train_slots = 0
            for row in _read_jsonl(expanded_root / "eng_Latn" / "train.jsonl"):
                train_slots += len(
                    list(split_sentence_like(row["text"], max_chars=60_000))
                ) * int(row["lineage"]["rehydration_multiplier"])
            validation_slots = sum(
                len(list(split_sentence_like(row["text"], max_chars=60_000)))
                for row in _read_jsonl(
                    expanded_root / "eng_Latn" / "validation_reserve.jsonl"
                )
            )
            self.assertGreater(validation_slots, 0)
            self.assertEqual(gate["actual_rehydrated_sentence_slots"], train_slots)
            self.assertNotEqual(
                gate["actual_rehydrated_sentence_slots"],
                train_slots + validation_slots,
            )

            # Re-sign internally self-consistent but false report accounting.
            english_report["counts"]["unique_train_sentences"] += 1
            english_report["targets"]["tokenizer_sentence_slot_gate"][
                "actual_unique_train_sentences"
            ] += 1
            english_report.pop("report_payload_sha256")
            english_report["report_payload_sha256"] = canonical_json_sha256(
                english_report
            )
            _write_json(english_report_path, english_report)
            summary_path = expanded_root / "bootstrap_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["report_payload_sha256"]["eng_Latn"] = english_report[
                "report_payload_sha256"
            ]
            summary.pop("summary_sha256")
            summary["summary_sha256"] = canonical_json_sha256(summary)
            _write_json(summary_path, summary)

            with self.assertRaisesRegex(RuntimeError, "exact tokenizer slot recount drifted"):
                from multilingual_megatron.pipeline.stages.sampling.tools.bootstrap_fineweb_v2 import (
                    merge_bootstrap_release,
                )

                merge_bootstrap_release(
                    source_lock_path=root / "source-lock.json",
                    rehydration_weights_path=root
                    / "materialized"
                    / "fineweb2-up-to-10-reps.json",
                    language_map_path=language_map,
                    output_root=expanded_root,
                    tokenizer_acquisition_plan_path=plan_path,
                )

            with self.assertRaisesRegex(
                SentenceArtifactError,
                "sentence-slot recount drifted",
            ):
                build_sentence_artifact(
                    input_manifest_path=summary_path,
                    language_map_path=language_map,
                    output_jsonl_path=root / "tampered-sentences.jsonl",
                    output_manifest_path=root / "tampered-sentences.manifest.json",
                    target_sentences=400,
                    acquisition_plan_path=plan_path,
                    alpha=0.3,
                    seed=42,
                    test_mode=True,
                )

            stale_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            stale_plan["expanded_pool_gate"].pop("counter_id")
            stale_plan.pop("plan_sha256")
            stale_plan["plan_sha256"] = canonical_json_sha256(stale_plan)
            self.assertFalse(verify_acquisition_plan(stale_plan))

    def test_true_source_exhaustion_allows_declared_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot_summary, language_map = _build_bootstrap_fixture(root)
            plan = build_sentence_acquisition_plan(
                input_manifest_path=pilot_summary,
                language_map_path=language_map,
                target_sentences=200,
                test_mode=True,
            )
            plan_path = root / "tokenizer-acquisition-plan.json"
            write_frozen_acquisition_plan(output_path=plan_path, plan=plan)
            expanded_root = root / "exhausted-bootstrap"
            bootstrap_release(
                source_lock_path=root / "source-lock.json",
                rehydration_weights_path=root
                / "materialized"
                / "fineweb2-up-to-10-reps.json",
                language_map_path=language_map,
                output_root=expanded_root,
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=1e-12,
                    target_unique_documents=10,
                    target_utf8_bytes=1,
                    calibration_documents=2,
                ),
                selected_language="all",
                local_input_path=root / "parquet",
                tokenizer_acquisition_plan_path=plan_path,
                incremental_from_root=root / "bootstrap",
            )
            report = build_sentence_artifact(
                input_manifest_path=expanded_root / "bootstrap_summary.json",
                language_map_path=language_map,
                output_jsonl_path=root / "source-exhausted-sentences.jsonl",
                output_manifest_path=root / "source-exhausted-sentences.manifest.json",
                target_sentences=200,
                acquisition_plan_path=plan_path,
                test_mode=True,
            )
            statuses = {
                row["expanded_pool_gate"]["status"]
                for row in report["languages"].values()
            }
            self.assertEqual(statuses, {"true_source_exhausted_repetition_required"})
            self.assertGreater(
                sum(row["repeated_sentence_count"] for row in report["languages"].values()),
                0,
            )

    def test_duplicate_doc_ids_and_train_validation_path_overlap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap_summary, language_map = _build_bootstrap_fixture(root)
            german_train = root / "bootstrap" / "deu_Latn" / "train.jsonl"
            english_row = _read_jsonl(root / "bootstrap" / "eng_Latn" / "train.jsonl")[0]
            german_rows = _read_jsonl(german_train)
            german_rows[0]["document_id"] = english_row["document_id"]
            german_train.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in german_rows),
                encoding="utf-8",
            )
            _refresh_bootstrap_hashes(root, "deu_Latn")

            with self.assertRaisesRegex(SentenceArtifactError, "overlapping/duplicate doc_id"):
                build_sentence_artifact(
                    input_manifest_path=bootstrap_summary,
                    language_map_path=language_map,
                    output_jsonl_path=root / "sentences.jsonl",
                    output_manifest_path=root / "sentences.manifest.json",
                    target_sentences=10,
                    test_mode=True,
                )

    def test_missing_blank_and_train_validation_path_overlap_fail_closed(self) -> None:
        for failure in ("missing", "blank", "path_overlap"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bootstrap_summary, language_map = _build_bootstrap_fixture(root)
                english_train = root / "bootstrap" / "eng_Latn" / "train.jsonl"
                if failure == "missing":
                    english_train.unlink()
                    expected = "does not exist"
                elif failure == "blank":
                    english_train.write_bytes(english_train.read_bytes() + b"\n")
                    _refresh_bootstrap_hashes(root, "eng_Latn")
                    expected = "blank input line"
                else:
                    report_path = root / "bootstrap" / "eng_Latn" / "bootstrap_report.json"
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report["outputs"]["validation_reserve.jsonl"]["path"] = "train.jsonl"
                    report.pop("report_payload_sha256")
                    report["report_payload_sha256"] = canonical_json_sha256(report)
                    _write_json(report_path, report)
                    summary = json.loads(bootstrap_summary.read_text(encoding="utf-8"))
                    summary["report_payload_sha256"]["eng_Latn"] = report[
                        "report_payload_sha256"
                    ]
                    summary.pop("summary_sha256")
                    summary["summary_sha256"] = canonical_json_sha256(summary)
                    _write_json(bootstrap_summary, summary)
                    expected = "train/validation path overlap"

                with self.assertRaisesRegex(SentenceArtifactError, expected):
                    build_sentence_artifact(
                        input_manifest_path=bootstrap_summary,
                        language_map_path=language_map,
                        output_jsonl_path=root / "sentences.jsonl",
                        output_manifest_path=root / "sentences.manifest.json",
                        target_sentences=10,
                        test_mode=True,
                    )

    def test_small_cli_target_requires_test_mode(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "--input-manifest",
                    "unused.json",
                    "--language-map",
                    "unused-map.json",
                    "--output-jsonl",
                    "unused-output.jsonl",
                    "--output-manifest",
                    "unused-manifest.json",
                    "--target-sentences",
                    "25",
                ]
            )


if __name__ == "__main__":
    unittest.main()
