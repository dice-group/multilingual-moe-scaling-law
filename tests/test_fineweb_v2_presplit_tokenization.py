from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.pipeline.stages.tokenization import (  # noqa: E402
    fineweb_v2_presplit,
)
from multilingual_megatron.pipeline.stages.tokenization.fineweb_v2_presplit import (  # noqa: E402
    PresplitTokenizationError,
    build_tokenization_plan,
    merge_tokenization_results,
    run_tokenization_worker,
)


LANGUAGES = ("eng_Latn", "deu_Latn")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, *, language: str, split: str, documents: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "document_id": f"{language}-{split}-{index}",
                    "language": language,
                    "text": f"{language} frozen {split} document {index}",
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(documents)
        ),
        encoding="utf-8",
    )


def _write_index(prefix: Path, *, documents: int, tokens: int) -> None:
    if documents <= 0 or tokens < documents:
        raise AssertionError("fixture requires at least one token per document")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lengths = [tokens // documents] * documents
    for index in range(tokens % documents):
        lengths[index] += 1
    pointers: list[int] = []
    offset = 0
    for length in lengths:
        pointers.append(offset)
        offset += length * 2
    idx_path = Path(str(prefix) + ".idx")
    with idx_path.open("wb") as handle:
        handle.write(b"MMIDIDX\x00\x00")
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 3))  # uint16 token IDs
        handle.write(struct.pack("<Q", documents))
        handle.write(struct.pack("<Q", documents + 1))
        handle.write(struct.pack(f"<{documents}i", *lengths))
        handle.write(struct.pack(f"<{documents}q", *pointers))
        handle.write(struct.pack(f"<{documents + 1}q", *range(documents + 1)))
    Path(str(prefix) + ".bin").write_bytes(b"\x00" * (tokens * 2))


class FineWebV2PresplitTokenizationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        tokenizer = root / "tokenizer" / "tokenizer.model"
        tokenizer.parent.mkdir(parents=True)
        tokenizer.write_bytes(b"frozen-sentencepiece-fixture")

        source_specs = {
            "eng_Latn": {"train": (4, 20), "validation": (2, 10)},
            "deu_Latn": {"train": (4, 24), "validation": (2, 12)},
        }
        source_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
        for language, splits in source_specs.items():
            source_artifacts[language] = {}
            for split, (documents, _tokens) in splits.items():
                path = root / "corpus" / split / f"{language}.jsonl"
                _write_jsonl(path, language=language, split=split, documents=documents)
                source_artifacts[language][split] = {
                    "path": str(path.relative_to(root / "corpus")),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }

        plan: dict[str, Any] = {
            "schema_version": 1,
            "policy_id": "fixture",
            "alpha": 0.3,
            "train_tokens_per_set": 100,
            "validation_cap_tokens": 20_000_000,
            "tiers": {
                "K1": {
                    "languages": ["eng_Latn"],
                    "alpha": 0.3,
                    "target_train_tokens": 100,
                    "probabilities": {"eng_Latn": 1.0},
                    "target_exposure_tokens": {"eng_Latn": 100},
                },
                "K2": {
                    "languages": ["eng_Latn", "deu_Latn"],
                    "alpha": 0.3,
                    "target_train_tokens": 100,
                    "probabilities": {"eng_Latn": 0.25, "deu_Latn": 0.75},
                    "target_exposure_tokens": {"eng_Latn": 25, "deu_Latn": 75},
                },
            },
            "physical_pool": {language: {"physical_target_tokens": 20} for language in LANGUAGES},
            "provenance": {
                "tokenizer_model_sha256": sha256_file(tokenizer),
                "source_manifest_payload_sha256": hashlib.sha256(b"source-lock").hexdigest(),
            },
        }
        plan["plan_sha256"] = canonical_json_sha256(plan)
        plan_path = root / "sampling_plan.json"
        _write_json(plan_path, plan)

        corpus_languages: dict[str, Any] = {}
        for language, splits in source_specs.items():
            corpus_languages[language] = {}
            for split, (documents, tokens) in splits.items():
                corpus_languages[language][split] = {
                    "status": "ready",
                    "emitted_documents": documents,
                    "selected_tokens": tokens,
                    "output": source_artifacts[language][split],
                }
        corpus: dict[str, Any] = {
            "schema_version": 1,
            "release_id": "fixture",
            "release_status": "ready",
            "ready": True,
            "inputs": {
                "sampling_plan": {
                    "path": str(plan_path),
                    "file_sha256": sha256_file(plan_path),
                    "payload_sha256": plan["plan_sha256"],
                },
                "tokenizer_model": {
                    "path": str(tokenizer),
                    "sha256": sha256_file(tokenizer),
                    "vocabulary_size": 64,
                    "special_token_ids": [0, 1, 2, 3],
                },
            },
            "languages": corpus_languages,
        }
        corpus["manifest_sha256"] = canonical_json_sha256(corpus)
        corpus_path = root / "corpus" / "corpus_manifest.json"
        _write_json(corpus_path, corpus)
        return corpus_path, plan_path, tokenizer

    @staticmethod
    def _fake_runner(item: dict[str, Any], output_prefix: Path, settings: dict[str, Any]):
        partitions = int(settings["partitions_per_language"])
        documents = int(item["expected_documents"])
        tokens = int(item["expected_tokens_including_eod"])
        doc_counts = [documents // partitions] * partitions
        token_counts = [tokens // partitions] * partitions
        for index in range(documents % partitions):
            doc_counts[index] += 1
        for index in range(tokens % partitions):
            token_counts[index] += 1
        prefixes: list[Path] = []
        for index in range(partitions):
            prefix = Path(f"{output_prefix}_{index}_text_document")
            _write_index(prefix, documents=doc_counts[index], tokens=token_counts[index])
            prefixes.append(prefix)
        return prefixes

    def test_end_to_end_preserves_splits_and_builds_nested_weighted_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, sampling_plan, tokenizer = self._fixture(root)
            plan_dir = root / "tokenization_plan"
            ranks_dir = root / "tokenization_ranks"
            output_dir = root / "tokenized"

            summary = build_tokenization_plan(
                corpus_manifest_path=corpus,
                sampling_plan_path=sampling_plan,
                tokenizer_model_path=tokenizer,
                plan_dir=plan_dir,
                array_ranks=2,
                verification_workers=2,
            )
            self.assertEqual(summary["item_count"], 4)
            self.assertEqual(
                summary["contract"]["input_sha256_verification_workers"], 2
            )
            planned_items = []
            for rank in range(2):
                rank_plan = json.loads(
                    (plan_dir / f"rank_{rank:05d}.json").read_text(encoding="utf-8")
                )
                planned_items.extend(rank_plan["items"])
            self.assertEqual(
                {(row["language"], row["split"]) for row in planned_items},
                {(language, split) for language in LANGUAGES for split in ("train", "validation")},
            )
            self.assertTrue(
                all(Path(row["source_path"]).name == f"{row['language']}.jsonl" for row in planned_items)
            )

            for rank in range(2):
                run_tokenization_worker(
                    plan_dir=plan_dir,
                    ranks_dir=ranks_dir,
                    rank=rank,
                    bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                    workers=2,
                    partitions_per_language=2,
                    preprocess_runner=self._fake_runner,
                )
            report = merge_tokenization_results(
                plan_dir=plan_dir,
                ranks_dir=ranks_dir,
                output_dir=output_dir,
            )

            self.assertTrue(report["pass"])
            self.assertTrue(report["scientific_contract"]["physical_K200_pool_tokenized_once"])
            self.assertFalse(report["scientific_contract"]["random_resplit"])
            k2 = json.loads(
                (output_dir / "tiers" / "K2" / "megatron_data_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            train = k2["splits"]["train"]
            self.assertEqual(train["documents_per_language"], {"deu_Latn": 4, "eng_Latn": 4})
            self.assertEqual(train["tokens_per_language"], {"deu_Latn": 24, "eng_Latn": 20})
            weight_by_prefix = dict(zip(train["prefixes"], train["weights"]))
            for prefix in train["prefixes_by_language"]["eng_Latn"]:
                self.assertAlmostEqual(weight_by_prefix[prefix], 0.125)
            for prefix in train["prefixes_by_language"]["deu_Latn"]:
                self.assertAlmostEqual(weight_by_prefix[prefix], 0.375)
            self.assertAlmostEqual(sum(train["weights"]), 1.0)

            k1 = json.loads(
                (output_dir / "tiers" / "K1" / "megatron_data_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(k1["languages"], ["eng_Latn"])
            self.assertEqual(set(k1["splits"]["validation"]["prefixes_by_language"]), set(LANGUAGES))
            validation = json.loads(
                (output_dir / "validation_megatron_data_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validation["languages"], ["eng_Latn", "deu_Latn"])
            self.assertEqual(
                validation["splits"]["validation"]["prefixes"],
                [
                    *validation["splits"]["validation"]["prefixes_by_language"]["eng_Latn"],
                    *validation["splits"]["validation"]["prefixes_by_language"]["deu_Latn"],
                ],
            )
            self.assertEqual(
                set(validation["splits"]["validation"]["prefix_artifacts_by_language"]),
                set(LANGUAGES),
            )




    def test_worker_fails_if_a_frozen_source_changes_after_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, sampling_plan, tokenizer = self._fixture(root)
            plan_dir = root / "tokenization_plan"
            build_tokenization_plan(
                corpus_manifest_path=corpus,
                sampling_plan_path=sampling_plan,
                tokenizer_model_path=tokenizer,
                plan_dir=plan_dir,
                array_ranks=1,
            )
            rank_plan = json.loads((plan_dir / "rank_00000.json").read_text(encoding="utf-8"))
            source = Path(rank_plan["items"][0]["source_path"])
            source.write_text(source.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

            with self.assertRaisesRegex(PresplitTokenizationError, "source identity changed"):
                run_tokenization_worker(
                    plan_dir=plan_dir,
                    ranks_dir=root / "ranks",
                    rank=0,
                    bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                    workers=2,
                    partitions_per_language=2,
                    preprocess_runner=self._fake_runner,
                )

    def test_worker_rejects_token_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, sampling_plan, tokenizer = self._fixture(root)
            plan_dir = root / "tokenization_plan"
            build_tokenization_plan(
                corpus_manifest_path=corpus,
                sampling_plan_path=sampling_plan,
                tokenizer_model_path=tokenizer,
                plan_dir=plan_dir,
                array_ranks=1,
            )

            def wrong_runner(item: dict[str, Any], output_prefix: Path, settings: dict[str, Any]):
                prefix = Path(f"{output_prefix}_text_document")
                _write_index(
                    prefix,
                    documents=int(item["expected_documents"]),
                    tokens=int(item["expected_tokens_including_eod"]) + 1,
                )
                return [prefix]

            with self.assertRaisesRegex(PresplitTokenizationError, "token count mismatch"):
                run_tokenization_worker(
                    plan_dir=plan_dir,
                    ranks_dir=root / "ranks",
                    rank=0,
                    bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                    workers=1,
                    partitions_per_language=1,
                    preprocess_runner=wrong_runner,
                )

    def test_worker_copies_unchanged_predecessor_prefixes_without_seal_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, sampling_plan, tokenizer = self._fixture(root)
            first_plan = root / "first_plan"
            first_ranks = root / "first_ranks"
            build_tokenization_plan(
                corpus_manifest_path=corpus,
                sampling_plan_path=sampling_plan,
                tokenizer_model_path=tokenizer,
                plan_dir=first_plan,
                array_ranks=1,
            )
            run_tokenization_worker(
                plan_dir=first_plan,
                ranks_dir=first_ranks,
                rank=0,
                bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                workers=2,
                partitions_per_language=2,
                preprocess_runner=self._fake_runner,
            )
            predecessor = root / "first_tokenized"
            merge_tokenization_results(
                plan_dir=first_plan,
                ranks_dir=first_ranks,
                output_dir=predecessor,
            )

            old_manifest = json.loads(
                (predecessor / "megatron_data_manifest.json").read_text(encoding="utf-8")
            )
            second_plan = root / "second_plan"
            second_ranks = root / "second_ranks"
            build_tokenization_plan(
                corpus_manifest_path=corpus,
                sampling_plan_path=sampling_plan,
                tokenizer_model_path=tokenizer,
                plan_dir=second_plan,
                array_ranks=1,
            )

            altered_corpus = json.loads(corpus.read_text(encoding="utf-8"))
            altered_corpus.pop("manifest_sha256")
            altered_corpus["release_id"] = "unrelated-self-consistent-corpus"
            altered_corpus["manifest_sha256"] = canonical_json_sha256(altered_corpus)
            altered_corpus_path = root / "altered_corpus_manifest.json"
            _write_json(altered_corpus_path, altered_corpus)
            with self.assertRaisesRegex(
                PresplitTokenizationError, "do not belong to the supplied corpus"
            ):
                run_tokenization_worker(
                    plan_dir=second_plan,
                    ranks_dir=root / "lineage_bad_ranks",
                    rank=0,
                    bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                    workers=2,
                    partitions_per_language=2,
                    reuse_tokenized_root=predecessor,
                    reuse_corpus_manifest_path=altered_corpus_path,
                    preprocess_runner=self._fake_runner,
                )
            with self.assertRaisesRegex(
                PresplitTokenizationError, "immutable regular file"
            ):
                run_tokenization_worker(
                    plan_dir=second_plan,
                    ranks_dir=root / "writable_bad_ranks",
                    rank=0,
                    bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                    workers=2,
                    partitions_per_language=2,
                    reuse_tokenized_root=predecessor,
                    reuse_corpus_manifest_path=corpus,
                    preprocess_runner=self._fake_runner,
                )
            for split in ("train", "validation"):
                payload = old_manifest["splits"][split]
                for artifacts in payload["prefix_artifacts_by_language"].values():
                    for artifact in artifacts:
                        for suffix in ("bin", "idx"):
                            Path(f"{artifact['prefix']}.{suffix}").chmod(0o444)

            def forbidden_runner(*_args: Any, **_kwargs: Any):
                raise AssertionError("unchanged prefixes must be reused")

            result = run_tokenization_worker(
                plan_dir=second_plan,
                ranks_dir=second_ranks,
                rank=0,
                bridge_root=REPO_ROOT / "external" / "Megatron-Bridge",
                workers=2,
                partitions_per_language=2,
                reuse_tokenized_root=predecessor,
                reuse_corpus_manifest_path=corpus,
                preprocess_runner=forbidden_runner,
            )
            self.assertTrue(result["preprocessing"]["reuse_enabled"])
            self.assertTrue(
                all(row["reused_predecessor_prefix"] for row in result["items"])
            )
            old_prefix = Path(
                old_manifest["splits"]["train"]["prefixes_by_language"]["eng_Latn"][0]
            )
            old_stat = Path(f"{old_prefix}.bin").stat()
            new_item = next(
                row
                for row in result["items"]
                if row["language"] == "eng_Latn" and row["split"] == "train"
            )
            new_prefix = Path(new_item["prefix_artifacts"][0]["prefix"])
            self.assertNotEqual(
                Path(f"{old_prefix}.bin").stat().st_ino,
                Path(f"{new_prefix}.bin").stat().st_ino,
            )
            old_stat_after = Path(f"{old_prefix}.bin").stat()
            self.assertEqual(old_stat.st_ino, old_stat_after.st_ino)
            self.assertEqual(old_stat.st_ctime_ns, old_stat_after.st_ctime_ns)
            real_sha256_file = fineweb_v2_presplit.sha256_file
            inode_hash_calls: Counter[tuple[int, int]] = Counter()

            def counting_sha256_file(path: Path) -> str:
                path = Path(path)
                if path.suffix in {".bin", ".idx"}:
                    stat = path.stat()
                    inode_hash_calls[(int(stat.st_dev), int(stat.st_ino))] += 1
                return real_sha256_file(path)

            with patch.object(
                fineweb_v2_presplit,
                "sha256_file",
                side_effect=counting_sha256_file,
            ):
                report = merge_tokenization_results(
                    plan_dir=second_plan,
                    ranks_dir=second_ranks,
                    output_dir=root / "second_tokenized",
                )
            self.assertTrue(inode_hash_calls)
            self.assertTrue(all(count == 1 for count in inode_hash_calls.values()))
            reuse = report["incremental_tokenization"]
            self.assertEqual(
                reuse["materialization"], "verified_content_copy_distinct_inode"
            )
            self.assertFalse(reuse["sealed_predecessor_inode_mutation"])
            self.assertEqual(reuse["reused_item_count"], 4)
            self.assertEqual(reuse["rebuilt_item_count"], 0)
            self.assertEqual(
                reuse["predecessor"]["predecessor_corpus_manifest"]["file_sha256"],
                sha256_file(corpus),
            )


if __name__ == "__main__":
    unittest.main()
