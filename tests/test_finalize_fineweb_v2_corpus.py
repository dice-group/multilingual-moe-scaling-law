from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    assign_validation_reserve,
    canonical_json_sha256,
    sha256_file,
    stable_document_id,
    validation_target_tokens,
)
from multilingual_megatron.pipeline.stages.sampling.tools.finalize_fineweb_v2_corpus import (  # noqa: E402
    CorpusFinalizationError,
    TRAIN_SELECTION_NAMESPACE,
    VALIDATION_SELECTION_NAMESPACE,
    finalize_corpus,
)
from multilingual_megatron.pipeline.stages.sampling.tools.parallel_finalize_fineweb_v2 import (  # noqa: E402
    merge as merge_parallel_finalizer,
    prepare as prepare_parallel_finalizer,
    run_worker as run_parallel_finalizer_worker,
)


LANGUAGE = "deu_Latn"
SOURCE_FILE_SHA256 = hashlib.sha256(b"frozen-source-shard").hexdigest()
VALIDATION_CANDIDATE_FRACTION = 0.30


class FinalizeFineWebV2CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import sentencepiece as spm
        except ModuleNotFoundError:
            raise unittest.SkipTest("sentencepiece is not installed")

        cls._tokenizer_tmp = tempfile.TemporaryDirectory()
        tokenizer_root = Path(cls._tokenizer_tmp.name)
        training_text = tokenizer_root / "tokenizer-training.txt"
        training_text.write_text(
            "\n".join(
                [
                    "alpha beta gamma delta epsilon zeta eta theta",
                    "ein kleiner deutscher testsatz mit mehreren woertern",
                    "validation reserve document text for deterministic tokenization",
                    "multilingual mixture of experts scaling law corpus",
                    "whole documents retain their stable source lineage",
                ]
                * 50
            )
            + "\n",
            encoding="utf-8",
        )
        model_prefix = tokenizer_root / "fixture"
        spm.SentencePieceTrainer.train(
            input=str(training_text),
            model_prefix=str(model_prefix),
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
        cls.tokenizer_model = model_prefix.with_suffix(".model")
        cls.processor = spm.SentencePieceProcessor(model_file=str(cls.tokenizer_model))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tokenizer_tmp.cleanup()

    def _row(
        self,
        *,
        source_id: str,
        text: str,
        split: str,
        multiplier: int = 1,
        train_multiplier: int | None = None,
    ) -> dict[str, Any]:
        candidate = split == "validation_reserve"
        source_id_base = source_id
        for suffix in range(10_000):
            source_id = f"{source_id_base}-{suffix}"
            document_id = stable_document_id(
                source_dataset="HuggingFaceFW/fineweb-2",
                source_revision="frozen-fixture-revision",
                subset=LANGUAGE,
                source_id=source_id,
                url=None,
                text=text,
            )
            if assign_validation_reserve(
                document_id=document_id,
                seed=42,
                validation_fraction=VALIDATION_CANDIDATE_FRACTION,
            ) == candidate:
                break
        else:  # pragma: no cover - impossible except after a hash-contract bug.
            raise AssertionError("could not construct a fixture with the requested split")
        lineage = {
            "source_dataset": "HuggingFaceFW/fineweb-2",
            "source_revision": "frozen-fixture-revision",
            "subset": LANGUAGE,
            "source_id": source_id,
            "url": None,
            "source_file_sha256": SOURCE_FILE_SHA256,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "split": split,
            "rehydration_multiplier": multiplier,
            "train_rehydration_multiplier": int(
                multiplier if train_multiplier is None else train_multiplier
            ),
        }
        return {
            "document_id": document_id,
            "language": LANGUAGE,
            "text": text,
            "lineage": lineage,
        }

    def _document_tokens(self, row: dict[str, Any]) -> list[int]:
        return [
            *self.processor.encode(row["text"], out_type=int),
            int(self.processor.eos_id()),
        ]

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _write_inputs(
        self,
        root: Path,
        *,
        train_rows: list[dict[str, Any]],
        validation_rows: list[dict[str, Any]],
        physical_target_tokens: int,
        validation_cap_tokens: int = 1,
        exposure_tokens: int | None = None,
        complete_source_scan: bool = True,
    ) -> tuple[Path, Path]:
        train_path = root / "bootstrap" / "train" / f"{LANGUAGE}.jsonl"
        validation_path = (
            root / "bootstrap" / "validation_reserve" / f"{LANGUAGE}.jsonl"
        )
        self._write_jsonl(train_path, train_rows)
        self._write_jsonl(validation_path, validation_rows)

        report_path = root / "bootstrap" / "reports" / LANGUAGE / "bootstrap_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        all_rows = train_rows + validation_rows
        report: dict[str, Any] = {
            "schema_version": 1,
            "tool_id": "fineweb_ab_v2_bootstrap_v1",
            "status": "complete",
            "acquisition_status": (
                "source_exhausted" if complete_source_scan else "targets_reached"
            ),
            "targets": {"source_exhausted": complete_source_scan},
            "counts": {
                "unique_source_documents": len(all_rows),
                "unique_source_utf8_bytes": sum(
                    len(str(row["text"]).encode("utf-8")) for row in all_rows
                ),
            },
            "run_config": {
                "settings": {
                    "validation_candidate_fraction": VALIDATION_CANDIDATE_FRACTION,
                }
            },
        }
        report["report_payload_sha256"] = canonical_json_sha256(report)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        bootstrap: dict[str, Any] = {
            "schema_version": 1,
            "provenance": {
                "source_manifest_sha256": hashlib.sha256(
                    b"source-materialization-manifest"
                ).hexdigest()
            },
            "languages": {
                LANGUAGE: {
                    "acquisition": {
                        "report_path": str(report_path.relative_to(root)),
                        "report_payload_sha256": report["report_payload_sha256"],
                        "acquisition_status": report["acquisition_status"],
                        "complete_source_scan": complete_source_scan,
                        "unique_source_documents": report["counts"]["unique_source_documents"],
                        "unique_source_utf8_bytes": report["counts"]["unique_source_utf8_bytes"],
                    },
                    "train": {
                        "path": str(train_path.relative_to(root)),
                        "sha256": sha256_file(train_path),
                    },
                    "validation_reserve": {
                        "path": str(validation_path.relative_to(root)),
                        "sha256": sha256_file(validation_path),
                    },
                }
            },
        }
        bootstrap["manifest_sha256"] = canonical_json_sha256(bootstrap)
        bootstrap_path = root / "bootstrap_manifest.json"
        bootstrap_path.write_text(
            json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        plan: dict[str, Any] = {
            "schema_version": 1,
            "policy_id": "fineweb_ab_v2_alpha03_rehydrated",
            "sampling_seed": 42,
            "alpha": 0.3,
            "train_tokens_per_set": int(
                exposure_tokens
                if exposure_tokens is not None
                else physical_target_tokens
            ),
            "validation_cap_tokens": validation_cap_tokens,
            "tiers": {
                "K1": {
                    "languages": [LANGUAGE],
                    "target_exposure_tokens": {
                        LANGUAGE: int(
                            exposure_tokens
                            if exposure_tokens is not None
                            else physical_target_tokens
                        )
                    },
                }
            },
            "physical_pool": {
                LANGUAGE: {
                    "physical_target_tokens": physical_target_tokens,
                }
            },
            "provenance": {
                "tokenizer_model_sha256": sha256_file(self.tokenizer_model),
                "source_manifest_payload_sha256": bootstrap["provenance"][
                    "source_manifest_sha256"
                ],
            },
        }
        plan["plan_sha256"] = canonical_json_sha256(plan)
        plan_path = root / "sampling_plan.json"
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return bootstrap_path, plan_path

    def _finalize(
        self,
        root: Path,
        *,
        train_rows: list[dict[str, Any]],
        validation_rows: list[dict[str, Any]],
        physical_target_tokens: int,
        validation_cap_tokens: int = 1,
        exposure_tokens: int | None = None,
        complete_source_scan: bool = True,
    ) -> tuple[dict[str, Any], Path]:
        bootstrap_path, plan_path = self._write_inputs(
            root,
            train_rows=train_rows,
            validation_rows=validation_rows,
            physical_target_tokens=physical_target_tokens,
            validation_cap_tokens=validation_cap_tokens,
            exposure_tokens=exposure_tokens,
            complete_source_scan=complete_source_scan,
        )
        output_dir = root / "finalized"
        manifest = finalize_corpus(
            bootstrap_manifest_path=bootstrap_path,
            tokenizer_model_path=self.tokenizer_model,
            sampling_plan_path=plan_path,
            output_dir=output_dir,
        )
        return manifest, output_dir

    def test_parallel_wrapper_matches_monolithic_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_rows = [
                self._row(
                    source_id=f"parallel-train-{index}",
                    text=f"alpha beta gamma delta {index}",
                    split="train",
                )
                for index in range(4)
            ]
            validation_rows = [
                self._row(
                    source_id=f"parallel-validation-{index}",
                    text=f"validation reserve document text {index}",
                    split="validation_reserve",
                )
                for index in range(3)
            ]
            bootstrap_path, plan_path = self._write_inputs(
                root,
                train_rows=train_rows,
                validation_rows=validation_rows,
                physical_target_tokens=1,
                validation_cap_tokens=1,
            )
            monolithic_dir = root / "monolithic"
            monolithic = finalize_corpus(
                bootstrap_manifest_path=bootstrap_path,
                tokenizer_model_path=self.tokenizer_model,
                sampling_plan_path=plan_path,
                output_dir=monolithic_dir,
            )
            work_root = root / "parallel-work"
            prepare_parallel_finalizer(
                bootstrap_manifest_path=bootstrap_path,
                sampling_plan_path=plan_path,
                work_root=work_root,
                workers=1,
            )
            (root / "local-index").mkdir()
            run_parallel_finalizer_worker(
                work_root=work_root,
                tokenizer_model_path=self.tokenizer_model,
                rank=0,
                workers=1,
                tokenization_workers=2,
                index_work_root=root / "local-index",
            )
            parallel_dir = root / "parallel"
            parallel = merge_parallel_finalizer(
                work_root=work_root,
                output_dir=parallel_dir,
                tokenizer_model_path=self.tokenizer_model,
            )

            self.assertEqual(parallel["languages"], monolithic["languages"])
            self.assertEqual(
                parallel["selection_contract"], monolithic["selection_contract"]
            )
            for relative in (
                f"train/{LANGUAGE}.jsonl",
                f"validation/{LANGUAGE}.jsonl",
                f"unigram_counts/{LANGUAGE}.json",
                "validation_prefix_inputs.json",
            ):
                self.assertEqual(
                    (parallel_dir / relative).read_bytes(),
                    (monolithic_dir / relative).read_bytes(),
                )
            hashes = json.loads(
                (parallel_dir / "hash_manifest.json").read_text(encoding="utf-8")
            )
            expected = hashes.pop("hash_manifest_payload_sha256")
            self.assertEqual(expected, canonical_json_sha256(hashes))
            self.assertEqual(
                hashes["files"]["corpus_manifest.json"],
                sha256_file(parallel_dir / "corpus_manifest.json"),
            )
            self.assertEqual(list((root / "local-index").iterdir()), [])

    def test_whole_document_selection_reports_exact_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_rows = [
                self._row(
                    source_id="train-a",
                    text="alpha beta gamma",
                    split="train",
                ),
                self._row(
                    source_id="train-b",
                    text="delta epsilon zeta eta theta",
                    split="train",
                ),
                self._row(
                    source_id="train-c",
                    text="ein kleiner deutscher testsatz",
                    split="train",
                ),
            ]
            ordered = sorted(
                train_rows,
                key=lambda row: hashlib.sha256(
                    f"42\0{TRAIN_SELECTION_NAMESPACE}\0{row['document_id']}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            first_tokens = len(self._document_tokens(ordered[0]))
            second_tokens = len(self._document_tokens(ordered[1]))
            target = first_tokens + 1
            validation_rows = [
                self._row(
                    source_id="validation-a",
                    text="validation reserve document",
                    split="validation_reserve",
                )
            ]

            manifest, output_dir = self._finalize(
                root,
                train_rows=list(reversed(train_rows)),
                validation_rows=validation_rows,
                physical_target_tokens=target,
            )

            self.assertEqual(manifest["selection_contract"]["sampling_seed"], 42)
            self.assertEqual(
                manifest["selection_contract"]["selection_namespaces"]["train"],
                TRAIN_SELECTION_NAMESPACE,
            )
            train = manifest["languages"][LANGUAGE]["train"]
            self.assertEqual(train["selected_original_documents"], 2)
            self.assertEqual(train["selected_tokens"], first_tokens + second_tokens)
            self.assertEqual(train["overshoot_tokens"], second_tokens - 1)
            emitted = [
                json.loads(line)
                for line in (output_dir / "train" / f"{LANGUAGE}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["original_document_id"] for row in emitted],
                [ordered[0]["document_id"], ordered[1]["document_id"]],
            )

    def test_train_validation_document_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="overlap",
                text="same original document",
                split="train",
            )
            validation = json.loads(json.dumps(train))
            validation["lineage"]["split"] = "validation_reserve"
            validation["lineage"]["train_rehydration_multiplier"] = 1
            bootstrap_path, plan_path = self._write_inputs(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=1,
            )

            with self.assertRaisesRegex(
                CorpusFinalizationError,
                "split does not match|train/validation document-ID overlap",
            ):
                finalize_corpus(
                    bootstrap_manifest_path=bootstrap_path,
                    tokenizer_model_path=self.tokenizer_model,
                    sampling_plan_path=plan_path,
                    output_dir=root / "finalized",
                )
            self.assertFalse((root / "finalized").exists())

    def test_rehydration_replicas_are_selected_and_emitted_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="rehydrated",
                text="alpha beta gamma delta",
                split="train",
                multiplier=3,
            )
            per_replica_tokens = len(self._document_tokens(train))
            validation = self._row(
                source_id="validation",
                text="validation reserve document",
                split="validation_reserve",
            )
            manifest, output_dir = self._finalize(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=per_replica_tokens + 1,
                exposure_tokens=per_replica_tokens * 6,
            )

            train_audit = manifest["languages"][LANGUAGE]["train"]
            self.assertEqual(train_audit["selected_original_documents"], 1)
            self.assertEqual(train_audit["emitted_documents"], 3)
            self.assertEqual(train_audit["selected_tokens"], per_replica_tokens * 3)
            rows = [
                json.loads(line)
                for line in (output_dir / "train" / f"{LANGUAGE}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["lineage"]["rehydration_replica_index"] for row in rows],
                [0, 1, 2],
            )
            self.assertEqual({row["original_document_id"] for row in rows}, {train["document_id"]})
            exposure = manifest["languages"][LANGUAGE]["exposure_pass_audit"]["K1"]
            self.assertEqual(exposure["effective_passes"], 2.0)
            self.assertTrue(exposure["reuses_physical_pool"])

    def test_validation_target_uses_fraction_and_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_rows = [
                self._row(
                    source_id="train-long",
                    text="alpha beta gamma delta epsilon " * 8,
                    split="train",
                )
            ]
            validation_rows = [
                self._row(
                    source_id=f"validation-{index}",
                    text="validation reserve document text " * 8,
                    split="validation_reserve",
                )
                for index in range(3)
            ]
            manifest, _ = self._finalize(
                root,
                train_rows=train_rows,
                validation_rows=validation_rows,
                physical_target_tokens=1,
                validation_cap_tokens=7,
            )
            language = manifest["languages"][LANGUAGE]
            self.assertGreater(language["exact_unique_language_tokens_including_eod"], 35)
            self.assertEqual(language["validation"]["target_tokens"], 7)
            self.assertGreaterEqual(language["validation"]["selected_tokens"], 7)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_rows = [
                self._row(
                    source_id="train-short",
                    text="alpha beta",
                    split="train",
                )
            ]
            validation_rows = [
                self._row(
                    source_id="validation-long",
                    text="validation reserve document text " * 12,
                    split="validation_reserve",
                )
            ]
            manifest, _ = self._finalize(
                root,
                train_rows=train_rows,
                validation_rows=validation_rows,
                physical_target_tokens=1,
                validation_cap_tokens=1000,
            )
            language = manifest["languages"][LANGUAGE]
            expected = validation_target_tokens(
                language["exact_unique_language_tokens_including_eod"],
                cap_tokens=1000,
            )
            self.assertEqual(language["validation"]["target_tokens"], expected)

    def test_incomplete_source_below_cap_proof_is_never_release_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="train-incomplete",
                text="alpha beta gamma delta",
                split="train",
            )
            validation = self._row(
                source_id="validation-incomplete",
                text="validation reserve document",
                split="validation_reserve",
            )

            manifest, _ = self._finalize(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=1,
                validation_cap_tokens=10_000,
                complete_source_scan=False,
            )

            self.assertFalse(manifest["ready"])
            self.assertEqual(
                manifest["release_status"],
                "needs_incremental_validation_inventory",
            )
            self.assertEqual(
                manifest["blocking_languages"][
                    "needs_incremental_validation_inventory"
                ],
                [LANGUAGE],
            )
            validation_audit = manifest["languages"][LANGUAGE]["validation"]
            self.assertFalse(validation_audit["quota_exact"])
            self.assertEqual(
                validation_audit["inventory_basis"],
                "insufficient_observed_U_provisional_only",
            )

    def test_train_only_unigram_counts_exclude_special_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="unigrams",
                text="alpha beta gamma delta",
                split="train",
                multiplier=2,
            )
            validation = self._row(
                source_id="validation",
                text="validation reserve document",
                split="validation_reserve",
            )
            manifest, output_dir = self._finalize(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=1,
            )
            unigram = json.loads(
                (output_dir / "unigram_counts" / f"{LANGUAGE}.json").read_text(
                    encoding="utf-8"
                )
            )
            special = set(unigram["special_token_ids_excluded"])
            expected: Counter[int] = Counter()
            excluded = 0
            # Rehydration affects training exposure, but the vocabulary-
            # insensitive unigram baseline is built from unique originals.
            for token_id in self._document_tokens(train):
                if token_id in special:
                    excluded += 1
                else:
                    expected[token_id] += 1
            self.assertEqual(
                unigram["counts_by_token_id"],
                {str(token_id): count for token_id, count in sorted(expected.items())},
            )
            self.assertEqual(unigram["excluded_special_token_count"], excluded)
            self.assertEqual(
                unigram["unique_train_tokens_including_special"],
                len(self._document_tokens(train)),
            )
            self.assertEqual(
                unigram["exposure_train_tokens_including_special"],
                2 * len(self._document_tokens(train)),
            )
            self.assertNotIn(str(self.processor.eos_id()), unigram["counts_by_token_id"])
            self.assertEqual(
                manifest["languages"][LANGUAGE]["train"]["unigram_counts"]["sha256"],
                sha256_file(output_dir / "unigram_counts" / f"{LANGUAGE}.json"),
            )

    def test_material_train_shortfall_emits_non_ready_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="only-train-document",
                text="alpha beta gamma",
                split="train",
            )
            available = len(self._document_tokens(train))
            validation = self._row(
                source_id="validation",
                text="validation reserve document",
                split="validation_reserve",
            )
            manifest, output_dir = self._finalize(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=available * 2,
            )

            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["release_status"], "needs_incremental_source_docs")
            self.assertEqual(
                manifest["blocking_languages"]["needs_incremental_source_docs"],
                [LANGUAGE],
            )
            train_audit = manifest["languages"][LANGUAGE]["train"]
            self.assertEqual(train_audit["status"], "needs_incremental_source_docs")
            self.assertEqual(train_audit["shortfall_tokens"], available)
            self.assertEqual(train_audit["shortfall_fraction"], 0.5)
            self.assertTrue((output_dir / "corpus_manifest.json").is_file())
            self.assertTrue((output_dir / "hash_manifest.json").is_file())
            hashes = json.loads(
                (output_dir / "hash_manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotIn(".document_index.sqlite3", hashes["files"])
            for relative_path, expected_sha256 in hashes["files"].items():
                artifact = output_dir / relative_path
                self.assertTrue(artifact.is_file(), relative_path)
                self.assertEqual(sha256_file(artifact), expected_sha256)

    def test_even_sub_five_percent_train_shortfall_is_not_ready(self) -> None:
        """A frozen physical-pool target may not be silently underfilled."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_rows = [
                self._row(
                    source_id=f"train-{index}",
                    text="alpha beta gamma delta epsilon zeta eta theta",
                    split="train",
                )
                for index in range(20)
            ]
            available = sum(len(self._document_tokens(row)) for row in train_rows)
            validation = self._row(
                source_id="validation",
                text="validation reserve document",
                split="validation_reserve",
            )
            manifest, _ = self._finalize(
                root,
                train_rows=train_rows,
                validation_rows=[validation],
                physical_target_tokens=available + 1,
            )

            train_audit = manifest["languages"][LANGUAGE]["train"]
            self.assertLess(train_audit["shortfall_fraction"], 0.05)
            self.assertEqual(train_audit["max_allowed_shortfall_fraction"], 0.0)
            self.assertEqual(train_audit["shortfall_tokens"], 1)
            self.assertEqual(train_audit["status"], "needs_incremental_source_docs")
            self.assertFalse(manifest["ready"])

    def test_unused_validation_candidate_is_reclaimed_with_preserved_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="ordinary-train",
                text="alpha beta gamma delta",
                split="train",
            )
            candidates = [
                self._row(
                    source_id=f"candidate-{index}",
                    text=f"validation reserve candidate text number {index}",
                    split="validation_reserve",
                    train_multiplier=3,
                )
                for index in range(2)
            ]
            selected_validation, reclaimed = sorted(
                candidates,
                key=lambda row: hashlib.sha256(
                    f"42\0{VALIDATION_SELECTION_NAMESPACE}\0{row['document_id']}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            train_target = len(self._document_tokens(train)) + 3 * len(
                self._document_tokens(reclaimed)
            )

            manifest, output_dir = self._finalize(
                root,
                train_rows=[train],
                validation_rows=list(reversed(candidates)),
                physical_target_tokens=train_target,
                validation_cap_tokens=1,
            )

            self.assertTrue(manifest["ready"])
            contract = manifest["selection_contract"]["validation_candidate_reserve"]
            self.assertEqual(contract["declared_fraction"], 0.30)
            self.assertEqual(contract["final_target_fraction"], 0.20)
            audit = manifest["languages"][LANGUAGE]["validation_candidate_reserve"]
            self.assertTrue(audit["candidate_adequate"])
            self.assertEqual(audit["candidate_documents"], 2)
            self.assertEqual(audit["selected_validation_original_documents"], 1)
            self.assertEqual(
                audit["unused_candidate_documents_reclassified_to_train"], 1
            )
            self.assertEqual(audit["selected_reclaimed_train_original_documents"], 1)
            self.assertEqual(audit["selected_reclaimed_train_emitted_documents"], 3)

            validation_output = [
                json.loads(line)
                for line in (output_dir / "validation" / f"{LANGUAGE}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                {row["original_document_id"] for row in validation_output},
                {selected_validation["document_id"]},
            )
            self.assertEqual(
                {row["lineage"]["rehydration_multiplier"] for row in validation_output},
                {1},
            )
            train_output = [
                json.loads(line)
                for line in (output_dir / "train" / f"{LANGUAGE}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            reclaimed_output = [
                row
                for row in train_output
                if row["original_document_id"] == reclaimed["document_id"]
            ]
            self.assertEqual(len(reclaimed_output), 3)
            self.assertEqual(
                {row["lineage"]["rehydration_multiplier"] for row in reclaimed_output},
                {3},
            )
            self.assertTrue(
                all(
                    row["lineage"]["validation_candidate_reclaimed"]
                    for row in reclaimed_output
                )
            )
            self.assertTrue(
                {row["original_document_id"] for row in train_output}.isdisjoint(
                    {row["original_document_id"] for row in validation_output}
                )
            )

    def test_insufficient_validation_candidate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="large-train",
                text="alpha beta gamma delta epsilon " * 1_000,
                split="train",
            )
            candidate = self._row(
                source_id="tiny-candidate",
                text="alpha",
                split="validation_reserve",
                train_multiplier=2,
            )
            manifest, _ = self._finalize(
                root,
                train_rows=[train],
                validation_rows=[candidate],
                physical_target_tokens=1,
                validation_cap_tokens=1_000,
            )

            self.assertFalse(manifest["ready"])
            self.assertEqual(
                manifest["release_status"], "needs_incremental_validation_docs"
            )
            validation = manifest["languages"][LANGUAGE]["validation"]
            self.assertGreater(validation["shortfall_tokens"], 0)
            candidate_audit = manifest["languages"][LANGUAGE][
                "validation_candidate_reserve"
            ]
            self.assertFalse(candidate_audit["candidate_adequate"])
            self.assertEqual(
                candidate_audit["status"], "insufficient_candidate_tokens"
            )
            self.assertEqual(
                candidate_audit["unused_candidate_documents_reclassified_to_train"],
                0,
            )

    def test_validation_candidate_requires_preserved_train_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="train",
                text="alpha beta gamma",
                split="train",
            )
            candidate = self._row(
                source_id="candidate",
                text="validation reserve document",
                split="validation_reserve",
            )
            del candidate["lineage"]["train_rehydration_multiplier"]
            bootstrap_path, plan_path = self._write_inputs(
                root,
                train_rows=[train],
                validation_rows=[candidate],
                physical_target_tokens=1,
            )
            with self.assertRaisesRegex(
                CorpusFinalizationError, "train_rehydration_multiplier"
            ):
                finalize_corpus(
                    bootstrap_manifest_path=bootstrap_path,
                    tokenizer_model_path=self.tokenizer_model,
                    sampling_plan_path=plan_path,
                    output_dir=root / "finalized",
                )

    def test_invalid_lineage_and_validation_rehydration_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="bad-lineage",
                text="alpha beta gamma",
                split="train",
            )
            del train["lineage"]["source_file_sha256"]
            validation = self._row(
                source_id="validation",
                text="validation reserve document",
                split="validation_reserve",
            )
            bootstrap_path, plan_path = self._write_inputs(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=1,
            )
            with self.assertRaisesRegex(CorpusFinalizationError, "lineage is missing"):
                finalize_corpus(
                    bootstrap_manifest_path=bootstrap_path,
                    tokenizer_model_path=self.tokenizer_model,
                    sampling_plan_path=plan_path,
                    output_dir=root / "finalized",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._row(
                source_id="train",
                text="alpha beta gamma",
                split="train",
            )
            validation = self._row(
                source_id="rehydrated-validation",
                text="validation reserve document",
                split="validation_reserve",
                multiplier=2,
            )
            bootstrap_path, plan_path = self._write_inputs(
                root,
                train_rows=[train],
                validation_rows=[validation],
                physical_target_tokens=1,
            )
            with self.assertRaisesRegex(CorpusFinalizationError, "validation must never"):
                finalize_corpus(
                    bootstrap_manifest_path=bootstrap_path,
                    tokenizer_model_path=self.tokenizer_model,
                    sampling_plan_path=plan_path,
                    output_dir=root / "finalized",
                )


if __name__ == "__main__":
    unittest.main()
