from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    LanguageCalibration,
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.pipeline.stages.sampling.tools.calibrate_fineweb_v2_tokens import (  # noqa: E402
    ENGLISH_UTF8_BYTES_RULE,
    build_token_calibration_artifact,
    write_frozen_calibration,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class FineWebV2TokenCalibrationTest(unittest.TestCase):
    def _tokenizer(self, root: Path) -> Path:
        try:
            import sentencepiece as spm
        except ModuleNotFoundError:
            raise unittest.SkipTest("sentencepiece is not installed")
        training_text = root / "tokenizer.txt"
        training_text.write_text(
            ("English multilingual calibration text. Deutsche Kalibrierung.\n" * 200),
            encoding="utf-8",
        )
        prefix = root / "tokenizer"
        spm.SentencePieceTrainer.train(
            input=str(training_text),
            model_prefix=str(prefix),
            model_type="unigram",
            vocab_size=64,
            character_coverage=1.0,
            hard_vocab_limit=False,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            minloglevel=2,
        )
        return prefix.with_suffix(".model")

    def _release(
        self,
        root: Path,
        *,
        calibration_count: int = 3,
        invalid_split_language: str | None = None,
        include_german_inventory: bool = True,
        german_words: object = 8_000,
        extra_inventory_rows: list[dict[str, object]] | None = None,
    ) -> tuple[Path, Path, Path, Path, dict[str, list[dict]]]:
        inventory_path = root / "fineweb2-language-distribution.csv"
        with inventory_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("subset", "split", "code", "script", "words", "documents", "utf8_bytes"),
            )
            writer.writeheader()
            if include_german_inventory:
                writer.writerow(
                    {
                        "subset": "deu_Latn",
                        "split": "train",
                        "code": "deu",
                        "script": "Latn",
                        "words": german_words,
                        "documents": 100,
                        "utf8_bytes": 50_000,
                    }
                )
            writer.writerow(
                {
                    "subset": "deu_Latn",
                    "split": "test",
                    "code": "deu",
                    "script": "Latn",
                    "words": 80,
                    "documents": 2,
                    "utf8_bytes": 500,
                }
            )
            for row in extra_inventory_rows or []:
                writer.writerow(row)

        source_lock_path = root / "source-lock.json"
        source_lock = {
            "schema_version": 1,
            "source_lock_id": "test-v2",
            "non_english_corpus": {
                "dataset": "fixture/fineweb-2",
                "dataset_revision": "a" * 40,
                "language_inventory": {
                    "url": "file://fixture",
                    "sha256": sha256_file(inventory_path),
                },
            },
            "english_corpus": {
                "dataset": "fixture/fineweb",
                "dataset_revision": "b" * 40,
                "subset": "eng_Latn",
                "availability": {
                    "reported_words": 1_001,
                    "reported_documents": 777,
                    "reported_parquet_bytes": 123,
                    "utf8_bytes_rule": ENGLISH_UTF8_BYTES_RULE,
                    "source": "fixture",
                },
            },
        }
        _write_json(source_lock_path, source_lock)
        source_lock_hash = sha256_file(source_lock_path)

        bootstrap_root = root / "bootstrap"
        languages = ["eng_Latn", "deu_Latn"]
        all_train: dict[str, list[dict]] = {}
        report_hashes: dict[str, str] = {}
        manifest_languages: dict[str, dict] = {}
        for language in languages:
            source_dataset = (
                source_lock["english_corpus"]["dataset"]
                if language == "eng_Latn"
                else source_lock["non_english_corpus"]["dataset"]
            )
            source_revision = (
                source_lock["english_corpus"]["dataset_revision"]
                if language == "eng_Latn"
                else source_lock["non_english_corpus"]["dataset_revision"]
            )
            train: list[dict] = []
            for index, text in enumerate(
                (
                    f"{language} first calibration document",
                    f"{language} second multilingual sample",
                    f"{language} third token measurement",
                )
            ):
                document_id = hashlib.sha256(f"{language}-{index}".encode()).hexdigest()
                train.append(
                    {
                        "document_id": document_id,
                        "language": language,
                        "text": text,
                        "artifact_role": "train",
                        "lineage": {
                            "source_dataset": source_dataset,
                            "source_revision": source_revision,
                            "subset": language,
                            "source_id": f"source-{index}",
                            "source_file": "part-000.parquet",
                            "source_file_sha256": "c" * 64,
                            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "url": None,
                            "split": (
                                "validation_reserve"
                                if invalid_split_language == language and index == 0
                                else "train"
                            ),
                            "rehydration_multiplier": (
                                1 if language == "eng_Latn" else (2 + index)
                            ),
                            "minhash_cluster_size": 1 + index,
                        },
                    }
                )
            all_train[language] = train
            language_dir = bootstrap_root / language
            train_path = language_dir / "train.jsonl"
            calibration_path = language_dir / "calibration.jsonl"
            validation_path = language_dir / "validation_reserve.jsonl"
            _write_jsonl(train_path, train)
            calibration_rows = []
            for record in train[:calibration_count]:
                copy = json.loads(json.dumps(record))
                copy["artifact_role"] = "calibration"
                calibration_rows.append(copy)
            _write_jsonl(calibration_path, calibration_rows)
            _write_jsonl(validation_path, [])
            outputs = {
                path.name: {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (train_path, calibration_path, validation_path)
            }
            run_config = {
                "schema_version": 1,
                "tool_id": "fineweb_ab_v2_bootstrap_v1",
                "language": language,
                "source": {
                    "dataset": source_dataset,
                    "revision": source_revision,
                    "subset": language,
                    "is_english": language == "eng_Latn",
                },
                "source_lock_sha256": source_lock_hash,
                "settings": {
                    "validation_fraction": 0.30,
                    "validation_candidate_fraction": 0.30,
                },
            }
            report = {
                "schema_version": 1,
                "tool_id": "fineweb_ab_v2_bootstrap_v1",
                "status": "complete",
                "run_config": run_config,
                "run_config_sha256": canonical_json_sha256(run_config),
                "counts": {
                    "unique_train_documents": len(train),
                    "calibration_documents": len(calibration_rows),
                },
                "hashes": {
                    "calibration_ids_sha256": canonical_json_sha256(
                        sorted(row["document_id"] for row in calibration_rows)
                    )
                },
                "outputs": outputs,
            }
            report["report_payload_sha256"] = canonical_json_sha256(report)
            report_path = language_dir / "bootstrap_report.json"
            _write_json(report_path, report)
            report_hashes[language] = report["report_payload_sha256"]
            manifest_languages[language] = {
                "train": {
                    "path": f"{language}/train.jsonl",
                    "sha256": outputs["train.jsonl"]["sha256"],
                },
                "validation_reserve": {
                    "path": f"{language}/validation_reserve.jsonl",
                    "sha256": outputs["validation_reserve.jsonl"]["sha256"],
                },
            }

        manifest = {
            "schema_version": 1,
            "provenance": {"source_lock_sha256": source_lock_hash},
            "languages": manifest_languages,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        _write_json(bootstrap_root / "bootstrap_manifest.json", manifest)
        summary = {
            "schema_version": 1,
            "tool_id": "fineweb_ab_v2_bootstrap_v1",
            "languages": languages,
            "report_payload_sha256": report_hashes,
            "bootstrap_manifest_payload_sha256": manifest["manifest_sha256"],
        }
        summary["summary_sha256"] = canonical_json_sha256(summary)
        summary_path = bootstrap_root / "bootstrap_summary.json"
        _write_json(summary_path, summary)
        tokenizer_path = self._tokenizer(root)
        return summary_path, tokenizer_path, source_lock_path, inventory_path, all_train

    def _build(self, fixture: tuple[Path, Path, Path, Path, object], **kwargs: object) -> dict:
        summary, tokenizer, source_lock, inventory, _ = fixture
        import sentencepiece as spm

        fixture_vocab_size = spm.SentencePieceProcessor(model_file=str(tokenizer)).vocab_size()
        target_validation_fraction = float(
            kwargs.pop("target_validation_fraction", 0.20)
        )
        return build_token_calibration_artifact(
            bootstrap_summary_path=summary,
            tokenizer_model_path=tokenizer,
            source_lock_path=source_lock,
            inventory_path=inventory,
            sample_documents=3,
            expected_language_count=2,
            expected_vocab_size=fixture_vocab_size,
            target_validation_fraction=target_validation_fraction,
            **kwargs,
        )

    def test_exact_counts_eod_rehydration_and_planner_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root)
            artifact = self._build(fixture)
            _, tokenizer_path, _, _, all_train = fixture

            import sentencepiece as spm

            processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            german = all_train["deu_Latn"]
            expected_tokens = sum(len(processor.encode(row["text"], out_type=int)) + 1 for row in german)
            expected_rehydrated = sum(
                (len(processor.encode(row["text"], out_type=int)) + 1)
                * row["lineage"]["rehydration_multiplier"]
                for row in german
            )
            compact = artifact["languages"]["deu_Latn"]
            self.assertEqual(compact["available_documents"], 100)
            self.assertEqual(compact["available_utf8_bytes"], 50_000)
            self.assertEqual(compact["calibration_documents"], 3)
            self.assertEqual(compact["calibration_tokens"], expected_tokens)
            self.assertEqual(compact["calibration_rehydrated_tokens"], expected_rehydrated)
            self.assertEqual(compact["validation_fraction"], 0.20)
            self.assertEqual(
                artifact["language_audits"]["deu_Latn"]["validation_contract"],
                {
                    "candidate_reserve_fraction": 0.30,
                    "target_token_fraction": 0.20,
                    "unused_candidates_return_to_model_training_selection_pool": True,
                    "candidate_reserve_excluded_from_tokenizer_training": True,
                },
            )
            self.assertEqual(
                artifact["language_audits"]["deu_Latn"]["measurements"]["eod_tokens_added"],
                3,
            )
            # No audit-only fields leak into the exact planner dataclass schema.
            LanguageCalibration(language="deu_Latn", **compact).validate()
            without_hash = dict(artifact)
            without_hash.pop("artifact_sha256")
            self.assertEqual(artifact["artifact_sha256"], canonical_json_sha256(without_hash))

    def test_english_bytes_use_declared_words_ratio_with_exact_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root)
            artifact = self._build(fixture)
            english = artifact["languages"]["eng_Latn"]
            audit = artifact["language_audits"]["eng_Latn"]
            calibration_bytes = english["calibration_utf8_bytes"]
            calibration_words = audit["measurements"]["words"]
            numerator = 1_001 * calibration_bytes
            quotient, remainder = divmod(numerator, calibration_words)
            expected = quotient + int(remainder * 2 >= calibration_words)
            self.assertEqual(english["available_documents"], 777)
            self.assertEqual(english["available_utf8_bytes"], expected)
            derivation = audit["availability"]["utf8_bytes_derivation"]
            self.assertEqual(derivation["numerator"], numerator)
            self.assertEqual(derivation["denominator"], calibration_words)
            self.assertEqual(derivation["declared_rule"], ENGLISH_UTF8_BYTES_RULE)

    def test_falls_back_to_full_train_when_calibration_jsonl_is_short(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root, calibration_count=1)
            artifact = self._build(fixture)
            for language in ("eng_Latn", "deu_Latn"):
                self.assertEqual(
                    artifact["language_audits"][language]["selection"]["input_role"],
                    "train",
                )
                self.assertEqual(artifact["languages"][language]["calibration_documents"], 3)

    def test_invalid_split_fails_closed_even_when_all_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root, invalid_split_language="deu_Latn")
            with self.assertRaisesRegex(ValueError, "invalid train lineage"):
                self._build(fixture)

    def test_candidate_reserve_must_exceed_target_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root)
            with self.assertRaisesRegex(ValueError, "smaller than the pre-tokenizer"):
                self._build(fixture, target_validation_fraction=0.30)

    def test_missing_inventory_coverage_and_frozen_output_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root, include_german_inventory=False)
            with self.assertRaisesRegex(ValueError, "no usable train rows|no train row"):
                self._build(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._build(self._release(root))
            output = root / "calibration.json"
            write_frozen_calibration(output_path=output, artifact=artifact)
            write_frozen_calibration(output_path=output, artifact=artifact)
            changed = json.loads(json.dumps(artifact))
            changed["status"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "different frozen calibration"):
                write_frozen_calibration(output_path=output, artifact=changed)

    def test_ignores_unselected_and_separately_locked_english_inventory_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(
                root,
                extra_inventory_rows=[
                    {
                        "subset": "und_Thai",
                        "split": "train",
                        "code": "und",
                        "script": "Thai",
                        "words": "-",
                        "documents": 251_256,
                        "utf8_bytes": 880_357_119,
                    },
                    {
                        "subset": "eng_Latn",
                        "split": "train",
                        "code": "eng",
                        "script": "Latn",
                        "words": 18_500,
                        "documents": 25_900,
                        "utf8_bytes": "-",
                    },
                ],
            )
            artifact = self._build(fixture)
            self.assertEqual(set(artifact["languages"]), {"eng_Latn", "deu_Latn"})
            self.assertEqual(
                artifact["language_audits"]["eng_Latn"]["availability"]["source"],
                "source_lock.english_corpus.availability",
            )

    def test_selected_non_english_inventory_metrics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._release(root, german_words="-")
            with self.assertRaisesRegex(
                ValueError, "inventory line 2 words must be a positive integer"
            ):
                self._build(fixture)


if __name__ == "__main__":
    unittest.main()
