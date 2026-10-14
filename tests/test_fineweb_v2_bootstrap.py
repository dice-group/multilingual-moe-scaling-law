from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.sampling.tools import (  # noqa: E402
    bootstrap_fineweb_v2 as bootstrap,
)
from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)


BootstrapSettings = bootstrap.BootstrapSettings
bootstrap_language = bootstrap.bootstrap_language
bootstrap_release = bootstrap.bootstrap_release


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FineWebV2BootstrapTest(unittest.TestCase):
    def _fixture_config(
        self,
        root: Path,
        *,
        languages: tuple[str, ...] = ("deu_Latn",),
        thresholds: dict[str, dict[str, int]] | None = None,
        identity_overrides: dict[str, dict[str, object]] | None = None,
        inventory_documents: int = 1_000,
        inventory_utf8_bytes: int = 1_000_000,
    ) -> tuple[Path, Path, Path]:
        weights_path = root / "materialized" / "fineweb2-up-to-10-reps.json"
        _write_json(
            weights_path,
            (
                thresholds
                if thresholds is not None
                else {"deu_Latn": {"1": 1, "2": 3, "5": 7, "10": 2}}
            ),
        )
        inventory_path = weights_path.parent / "fineweb2-language-distribution.csv"
        inventory_rows = [
            f"{language},train,{inventory_documents},{inventory_utf8_bytes}"
            for language in languages
            if language != "eng_Latn"
        ]
        inventory_path.write_text(
            "subset,split,documents,utf8_bytes\n" + "\n".join(inventory_rows) + "\n",
            encoding="utf-8",
        )
        source_lock_path = root / "source-lock.json"
        _write_json(
            source_lock_path,
            {
                "schema_version": 1,
                "source_lock_id": "fixture",
                "non_english_corpus": {
                    "dataset": "fixture/fineweb-2",
                    "dataset_revision": "a" * 40,
                    "language_inventory": {"sha256": _sha256(inventory_path)},
                    "rehydration": {
                        "enabled": True,
                        "weights_policy": "up_to_10_reps",
                        "weights_sha256": _sha256(weights_path),
                        "missing_weights_policy": (
                            "fail_closed_except_explicit_identity_overrides"
                        ),
                        "identity_overrides": identity_overrides or {},
                    },
                },
                "english_corpus": {
                    "dataset": "fixture/fineweb",
                    "dataset_revision": "b" * 40,
                    "subset": "eng_Latn",
                },
            },
        )
        language_map_path = root / "language-map.json"
        _write_json(
            language_map_path,
            {
                "schema_version": 2,
                "language_set_id": "fixture-v2",
                "0": {"group": "fixture", "languages": list(languages)},
            },
        )
        return source_lock_path, weights_path, language_map_path

    @staticmethod
    def _use_production_dataset_ids(source_lock_path: Path) -> None:
        payload = json.loads(source_lock_path.read_text(encoding="utf-8"))
        payload["non_english_corpus"]["dataset"] = "HuggingFaceFW/fineweb-2"
        payload["english_corpus"]["dataset"] = "HuggingFaceFW/fineweb"
        _write_json(source_lock_path, payload)

    def _write_parquet(self, root: Path, language: str, rows: list[dict]) -> Path:
        output = root / "input" / language / "part-000.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), output)
        return output

    @staticmethod
    def _small_settings(*, documents: int, seed: int = 42) -> BootstrapSettings:
        return BootstrapSettings(
            seed=seed,
            validation_fraction=0.20,
            target_unique_documents=documents,
            target_utf8_bytes=1,
            calibration_documents=5,
        )

    @staticmethod
    def _train_fixture_tokenizer(root: Path, rows: list[dict]) -> Path:
        try:
            import sentencepiece as spm
        except ModuleNotFoundError:
            raise unittest.SkipTest("sentencepiece is not installed")
        text_path = root / "tokenizer-training.txt"
        text_path.write_text(
            "\n".join(str(row["text"]) for row in rows) * 20 + "\n",
            encoding="utf-8",
        )
        prefix = root / "fixture-tokenizer"
        spm.SentencePieceTrainer.train(
            input=str(text_path),
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

    def test_path_order_is_seeded_sha256_order_not_listing_order(self) -> None:
        paths = [
            "data/lang/train/c.parquet",
            "data/lang/train/a.parquet",
            "data/lang/train/b.parquet",
        ]
        forward = bootstrap._ordered_paths(paths, seed=19, subset="lang_Latn")
        reverse = bootstrap._ordered_paths(list(reversed(paths)), seed=19, subset="lang_Latn")
        self.assertEqual(forward, reverse)
        self.assertEqual([order_hash for _, order_hash in forward], sorted(x[1] for x in forward))
        self.assertNotEqual(
            [path for path, _ in forward],
            [path for path, _ in bootstrap._ordered_paths(paths, seed=20, subset="lang_Latn")],
        )

    def test_metadata_preflight_lists_each_dataset_once_and_never_resolves_lfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "fineweb2-up-to-10-reps.json"
            _write_json(
                weights,
                {
                    "deu_Latn": {"1": 1},
                    "fra_Latn": {"1": 1},
                },
            )
            inventory = root / "fineweb2-language-distribution.csv"
            inventory.write_text(
                "subset,split,documents,utf8_bytes\n"
                "deu_Latn,train,100,1000\n"
                "fra_Latn,train,100,1000\n",
                encoding="utf-8",
            )
            source_lock = root / "source-lock.json"
            _write_json(
                source_lock,
                {
                    "schema_version": 1,
                    "non_english_corpus": {
                        "dataset": "HuggingFaceFW/fineweb-2",
                        "dataset_revision": "a" * 40,
                        "language_inventory": {"sha256": _sha256(inventory)},
                        "rehydration": {
                            "enabled": True,
                            "weights_policy": "up_to_10_reps",
                            "weights_sha256": _sha256(weights),
                            "missing_weights_policy": (
                                "fail_closed_except_explicit_identity_overrides"
                            ),
                            "identity_overrides": {},
                        },
                    },
                    "english_corpus": {
                        "dataset": "HuggingFaceFW/fineweb",
                        "dataset_revision": "b" * 40,
                    },
                },
            )
            language_map = root / "language-map.json"
            _write_json(
                language_map,
                {
                    "schema_version": 2,
                    "0": {
                        "group": "fixture",
                        "languages": ["eng_Latn", "deu_Latn", "fra_Latn"],
                    },
                },
            )

            def fake_list(repo_id: str, **_: object) -> list[str]:
                if repo_id == "HuggingFaceFW/fineweb":
                    return ["data/CC-MAIN-2024/000.parquet"]
                return [
                    "data/deu_Latn/train/000.parquet",
                    "data/fra_Latn/train/000.parquet",
                ]

            bootstrap._HF_REPO_FILES_CACHE.clear()
            with (
                patch("huggingface_hub.list_repo_files", side_effect=fake_list) as listing,
                patch.object(
                    bootstrap,
                    "_hf_file_sha256",
                    side_effect=AssertionError("preflight must not resolve LFS hashes"),
                ),
            ):
                first = bootstrap.preflight_paths(
                    source_lock_path=source_lock,
                    rehydration_weights_path=weights,
                    source_inventory_path=inventory,
                    language_map_path=language_map,
                    output_root=root / "preflight",
                    seed=42,
                )
                second = bootstrap.preflight_paths(
                    source_lock_path=source_lock,
                    rehydration_weights_path=weights,
                    source_inventory_path=inventory,
                    language_map_path=language_map,
                    output_root=root / "preflight",
                    seed=42,
                )
            self.assertEqual(listing.call_count, 2)
            self.assertEqual(first, second)
            self.assertEqual(len(first["paths_manifest_sha256"]), 3)
            self.assertEqual(
                first["rehydration_coverage"]["selected_non_english_language_count"],
                2,
            )
            self.assertEqual(
                first["rehydration_coverage"]["languages"]["eng_Latn"][
                    "policy_source"
                ],
                "english_fineweb_no_rehydration",
            )
            self.assertFalse((root / "preflight" / "eng_Latn" / "train.jsonl").exists())
            bootstrap._HF_REPO_FILES_CACHE.clear()

    def test_metadata_preflight_rejects_undeclared_missing_weight_before_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("eng_Latn", "deu_Latn", "hau_Latn"),
                thresholds={"deu_Latn": {"1": 1}},
            )
            self._use_production_dataset_ids(lock)
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            output = root / "preflight"
            with patch("huggingface_hub.list_repo_files") as listing:
                with self.assertRaisesRegex(ValueError, "no entry for hau_Latn"):
                    bootstrap.preflight_paths(
                        source_lock_path=lock,
                        rehydration_weights_path=weights,
                        source_inventory_path=inventory,
                        language_map_path=language_map,
                        output_root=output,
                        seed=42,
                    )
            listing.assert_not_called()
            self.assertFalse(output.exists())

    def test_metadata_preflight_accepts_only_audited_explicit_identity_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            override = {
                "thresholds": {"1": 1},
                "upstream_weights_present": False,
                "upstream_distribution_present": False,
                "weights_generation_commit": "a" * 40,
                "inventory_addition_commit": "b" * 40,
                "reason": "Pinned upstream release omits Hausa weights and distribution.",
            }
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("eng_Latn", "deu_Latn", "hau_Latn"),
                thresholds={"deu_Latn": {"1": 1}},
                identity_overrides={"hau_Latn": override},
            )
            self._use_production_dataset_ids(lock)
            inventory = weights.parent / "fineweb2-language-distribution.csv"

            def fake_list(repo_id: str, **_: object) -> list[str]:
                if repo_id == "HuggingFaceFW/fineweb":
                    return ["data/CC-MAIN-2024/000.parquet"]
                return [
                    "data/deu_Latn/train/000.parquet",
                    "data/hau_Latn/train/000.parquet",
                ]

            bootstrap._HF_REPO_FILES_CACHE.clear()
            with patch("huggingface_hub.list_repo_files", side_effect=fake_list):
                summary = bootstrap.preflight_paths(
                    source_lock_path=lock,
                    rehydration_weights_path=weights,
                    source_inventory_path=inventory,
                    language_map_path=language_map,
                    output_root=root / "preflight",
                    seed=42,
                )
            hau = summary["rehydration_coverage"]["languages"]["hau_Latn"]
            self.assertEqual(
                hau["policy_source"], "source_lock_explicit_identity_override"
            )
            self.assertEqual(
                hau["thresholds_sha256"], bootstrap.canonical_json_sha256({"1": 1})
            )
            bootstrap._HF_REPO_FILES_CACHE.clear()

    def test_explicit_identity_override_requires_both_commit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("hau_Latn",),
                thresholds={},
                identity_overrides={
                    "hau_Latn": {
                        "thresholds": {"1": 1},
                        "upstream_weights_present": False,
                        "upstream_distribution_present": False,
                        "reason": "Missing immutable commit provenance fixture.",
                    }
                },
            )
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            with self.assertRaisesRegex(ValueError, "invalid fields"):
                bootstrap.preflight_paths(
                    source_lock_path=lock,
                    rehydration_weights_path=weights,
                    source_inventory_path=inventory,
                    language_map_path=language_map,
                    output_root=root / "preflight",
                    seed=42,
                    local_input_path=root / "input",
                )

    def test_metadata_preflight_rejects_malformed_thresholds_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                thresholds={"deu_Latn": {"2": 1}},
            )
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            with patch("huggingface_hub.list_repo_files") as listing:
                with self.assertRaisesRegex(ValueError, "start at cluster size one"):
                    bootstrap.preflight_paths(
                        source_lock_path=lock,
                        rehydration_weights_path=weights,
                        source_inventory_path=inventory,
                        language_map_path=language_map,
                        output_root=root / "bad-thresholds",
                        seed=42,
                        local_input_path=root / "input",
                    )
            listing.assert_not_called()

            lock, weights, language_map = self._fixture_config(
                root / "bad-override",
                languages=("hau_Latn",),
                thresholds={},
                identity_overrides={
                    "hau_Latn": {
                        "thresholds": {"1": 2},
                        "upstream_weights_present": False,
                        "upstream_distribution_present": False,
                        "weights_generation_commit": "a" * 40,
                        "inventory_addition_commit": "b" * 40,
                        "reason": "Invalid non-identity test fixture.",
                    }
                },
            )
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                bootstrap.preflight_paths(
                    source_lock_path=lock,
                    rehydration_weights_path=weights,
                    source_inventory_path=inventory,
                    language_map_path=language_map,
                    output_root=root / "bad-override-output",
                    seed=42,
                    local_input_path=root / "input",
                )

            lock, weights, language_map = self._fixture_config(root / "drift")
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            weights.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match the pinned source lock"):
                bootstrap.preflight_paths(
                    source_lock_path=lock,
                    rehydration_weights_path=weights,
                    source_inventory_path=inventory,
                    language_map_path=language_map,
                    output_root=root / "hash-drift",
                    seed=42,
                    local_input_path=root / "input",
                )

    def test_metadata_preflight_rejects_missing_inventory_row_before_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("deu_Latn", "hau_Latn"),
                thresholds={
                    "deu_Latn": {"1": 1},
                    "hau_Latn": {"1": 1},
                },
            )
            self._use_production_dataset_ids(lock)
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            inventory.write_text(
                "subset,split,documents,utf8_bytes\n"
                "deu_Latn,train,100,1000\n",
                encoding="utf-8",
            )
            source_lock = json.loads(lock.read_text(encoding="utf-8"))
            source_lock["non_english_corpus"]["language_inventory"]["sha256"] = (
                _sha256(inventory)
            )
            _write_json(lock, source_lock)
            output = root / "preflight"
            with patch("huggingface_hub.list_repo_files") as listing:
                with self.assertRaisesRegex(
                    ValueError, "exactly one train row for hau_Latn"
                ):
                    bootstrap.preflight_paths(
                        source_lock_path=lock,
                        rehydration_weights_path=weights,
                        source_inventory_path=inventory,
                        language_map_path=language_map,
                        output_root=output,
                        seed=42,
                    )
            listing.assert_not_called()
            self.assertFalse(output.exists())

    def test_metadata_preflight_cli_forwards_weights_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(root)
            inventory = weights.parent / "fineweb2-language-distribution.csv"
            argv = [
                "bootstrap_fineweb_v2",
                "--source-lock",
                str(lock),
                "--rehydration-weights",
                str(weights),
                "--source-inventory",
                str(inventory),
                "--language-map",
                str(language_map),
                "--output-dir",
                str(root / "output"),
                "--metadata-preflight",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    bootstrap, "preflight_paths", return_value={"status": "ok"}
                ) as preflight,
            ):
                self.assertEqual(bootstrap.main(), 0)
            kwargs = preflight.call_args.kwargs
            self.assertEqual(kwargs["rehydration_weights_path"], weights)
            self.assertEqual(kwargs["source_inventory_path"], inventory)

    def test_language_selection_map_must_be_a_subset_of_identity_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, language_map = self._fixture_config(root)
            selection_map = root / "selection-map.json"
            _write_json(
                selection_map,
                {
                    "schema_version": 2,
                    "language_set_id": "invalid-selection",
                    "0": {"languages": ["hau_Latn"]},
                },
            )
            with self.assertRaisesRegex(ValueError, "not a subset"):
                bootstrap._load_language_selection(
                    language_map_path=language_map,
                    language_selection_map_path=selection_map,
                )

    def test_language_index_uses_selection_order_but_preserves_identity_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root, languages=("eng_Latn", "deu_Latn")
            )
            selection_map = root / "selection-map.json"
            _write_json(
                selection_map,
                {
                    "schema_version": 2,
                    "language_set_id": "ordered-selection",
                    "0": {"languages": ["deu_Latn", "eng_Latn"]},
                },
            )
            argv = [
                "bootstrap_fineweb_v2",
                "--source-lock",
                str(lock),
                "--rehydration-weights",
                str(weights),
                "--language-map",
                str(language_map),
                "--language-selection-map",
                str(selection_map),
                "--output-dir",
                str(root / "output"),
                "--language-index",
                "0",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    bootstrap,
                    "bootstrap_language",
                    return_value={"report_payload_sha256": "a" * 64},
                ) as worker,
            ):
                self.assertEqual(bootstrap.main(), 0)
            kwargs = worker.call_args.kwargs
            self.assertEqual(kwargs["language"], "deu_Latn")
            self.assertEqual(kwargs["language_map_path"], language_map)
            self.assertEqual(kwargs["language_selection_map_path"], selection_map)

    def test_exact_language_thresholds_are_applied_after_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(root)
            rows = [
                {
                    "id": f"doc-{cluster}",
                    "text": f"cluster size {cluster}",
                    "url": f"https://example.test/{cluster}",
                    "minhash_cluster_size": cluster,
                }
                for cluster in (1, 2, 4, 5, 12)
            ]
            self._write_parquet(root, "deu_Latn", rows)
            report = bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "output",
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=1e-12,
                    target_unique_documents=len(rows),
                    target_utf8_bytes=1,
                    calibration_documents=2,
                ),
                local_input_path=root / "input",
            )

            train = _read_jsonl(root / "output" / "deu_Latn" / "train.jsonl")
            by_original_id = {
                record["lineage"]["source_id"]: record["lineage"][
                    "rehydration_multiplier"
                ]
                for record in train
            }
            self.assertEqual(
                by_original_id,
                {"doc-1": 1, "doc-2": 3, "doc-4": 3, "doc-5": 7, "doc-12": 2},
            )
            self.assertEqual(report["counts"]["unique_validation_documents"], 0)
            self.assertEqual(
                report["counts"]["declared_train_documents_with_rehydration"], 16
            )
            self.assertEqual(
                report["run_config"]["rehydration_policy_source"],
                "pinned_upstream_weights",
            )
            self.assertEqual(
                report["rehydration"]["policy_source"], "pinned_upstream_weights"
            )
            self.assertTrue(report["rehydration"]["enabled"])
            self.assertFalse(report["rehydration"]["identity_override"])
            for record in train:
                self.assertEqual(record["lineage"]["split"], "train")
                self.assertEqual(record["lineage"]["source_revision"], "a" * 40)
                self.assertEqual(record["language"], "deu_Latn")
                self.assertEqual(len(record["lineage"]["source_file_sha256"]), 64)

    def test_bootstrap_report_records_explicit_identity_override_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("hau_Latn",),
                thresholds={},
                identity_overrides={
                    "hau_Latn": {
                        "thresholds": {"1": 1},
                        "upstream_weights_present": False,
                        "upstream_distribution_present": False,
                        "weights_generation_commit": "a" * 40,
                        "inventory_addition_commit": "b" * 40,
                        "reason": "Pinned upstream release omits Hausa policy artifacts.",
                    }
                },
            )
            self._write_parquet(
                root,
                "hau_Latn",
                [
                    {
                        "id": "hau-1",
                        "text": "Hausa identity override fixture",
                        "url": "https://example.test/hau",
                        "minhash_cluster_size": 9,
                    }
                ],
            )
            report = bootstrap_language(
                language="hau_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "output",
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=1e-12,
                    target_unique_documents=1,
                    target_utf8_bytes=1,
                    calibration_documents=1,
                ),
                local_input_path=root / "input",
            )
            self.assertEqual(
                report["run_config"]["rehydration_policy_source"],
                "source_lock_explicit_identity_override",
            )
            self.assertEqual(
                report["rehydration"]["policy_source"],
                "source_lock_explicit_identity_override",
            )
            self.assertFalse(report["rehydration"]["enabled"])
            self.assertTrue(report["rehydration"]["identity_override"])
            train = _read_jsonl(root / "output" / "hau_Latn" / "train.jsonl")
            self.assertEqual(train[0]["lineage"]["rehydration_multiplier"], 1)
            bootstrap.merge_bootstrap_release(
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "output",
                selected_languages=("hau_Latn",),
            )
            manifest = json.loads(
                (root / "output" / "bootstrap_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["languages"]["hau_Latn"]["rehydration"]["policy_source"],
                "source_lock_explicit_identity_override",
            )
            self.assertTrue(
                manifest["languages"]["hau_Latn"]["rehydration"][
                    "identity_override"
                ]
            )

    def test_split_is_disjoint_and_release_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(root)
            rows = [
                {
                    "id": f"doc-{index:03d}",
                    "text": f"Deterministic multilingual document {index}",
                    "url": f"https://example.test/{index}",
                    "minhash_cluster_size": 1 + index % 12,
                }
                for index in range(200)
            ]
            self._write_parquet(root, "deu_Latn", rows)
            settings = BootstrapSettings(
                seed=1729,
                validation_fraction=0.20,
                target_unique_documents=len(rows),
                target_utf8_bytes=1,
                calibration_documents=20,
            )

            reports = []
            for output_name in ("output-a", "output-b"):
                reports.append(
                    bootstrap_language(
                        language="deu_Latn",
                        source_lock_path=lock,
                        rehydration_weights_path=weights,
                        language_map_path=language_map,
                        output_root=root / output_name,
                        settings=settings,
                        local_input_path=root / "input",
                    )
                )

            first_dir = root / "output-a" / "deu_Latn"
            second_dir = root / "output-b" / "deu_Latn"
            for name in (
                "train.jsonl",
                "validation_reserve.jsonl",
                "calibration.jsonl",
            ):
                self.assertEqual((first_dir / name).read_bytes(), (second_dir / name).read_bytes())
            self.assertEqual(reports[0]["run_config_sha256"], reports[1]["run_config_sha256"])
            self.assertEqual(reports[0]["hashes"], reports[1]["hashes"])

            train_ids = {
                row["document_id"]
                for row in _read_jsonl(first_dir / "train.jsonl")
            }
            validation = _read_jsonl(first_dir / "validation_reserve.jsonl")
            validation_ids = {row["document_id"] for row in validation}
            calibration_ids = {
                row["document_id"]
                for row in _read_jsonl(first_dir / "calibration.jsonl")
            }
            self.assertTrue(validation_ids)
            self.assertFalse(train_ids & validation_ids)
            self.assertTrue(calibration_ids.issubset(train_ids))
            self.assertEqual(len(calibration_ids), 20)
            self.assertTrue(
                all(row["lineage"]["rehydration_multiplier"] == 1 for row in validation)
            )
            self.assertTrue(
                all(
                    row["lineage"]["train_rehydration_multiplier"] >= 1
                    for row in validation
                )
            )
            self.assertTrue(
                all(
                    row["lineage"]["train_rehydration_multiplier"]
                    == row["lineage"]["rehydration_multiplier"]
                    for row in _read_jsonl(first_dir / "train.jsonl")
                )
            )
            self.assertEqual(
                reports[0]["run_config"]["settings"][
                    "validation_candidate_fraction"
                ],
                0.20,
            )

    def test_existing_output_requires_exact_config_and_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(root)
            self._write_parquet(
                root,
                "deu_Latn",
                [
                    {
                        "id": f"doc-{index}",
                        "text": "x" * (index + 1),
                        "url": "",
                        "minhash_cluster_size": 1,
                    }
                    for index in range(20)
                ],
            )
            settings = BootstrapSettings(
                seed=7,
                validation_fraction=1e-12,
                target_unique_documents=20,
                target_utf8_bytes=1,
                calibration_documents=5,
            )
            kwargs = {
                "language": "deu_Latn",
                "source_lock_path": lock,
                "rehydration_weights_path": weights,
                "language_map_path": language_map,
                "output_root": root / "output",
                "settings": settings,
                "local_input_path": root / "input",
            }
            first = bootstrap_language(**kwargs)
            reused = bootstrap_language(**kwargs)
            self.assertEqual(first["report_payload_sha256"], reused["report_payload_sha256"])

            with self.assertRaisesRegex(RuntimeError, "do not match"):
                bootstrap_language(**(kwargs | {"settings": BootstrapSettings(
                    seed=8,
                    validation_fraction=1e-12,
                    target_unique_documents=20,
                    target_utf8_bytes=1,
                    calibration_documents=5,
                )}))

            train_path = root / "output" / "deu_Latn" / "train.jsonl"
            train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash/size changed"):
                bootstrap_language(**kwargs)

    def test_all_mode_uses_per_language_local_directories_and_english_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("eng_Latn", "deu_Latn"),
            )
            for language in ("eng_Latn", "deu_Latn"):
                rows = [
                    {
                        "id": f"{language}-{index}",
                        "text": f"{language} fixture {index}",
                        "url": "",
                        **({} if language == "eng_Latn" else {"minhash_cluster_size": 12}),
                    }
                    for index in range(20)
                ]
                self._write_parquet(root, language, rows)

            summary = bootstrap_release(
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "output",
                settings=BootstrapSettings(
                    seed=3,
                    validation_fraction=1e-12,
                    target_unique_documents=20,
                    target_utf8_bytes=1,
                    calibration_documents=5,
                ),
                selected_language="all",
                local_input_path=root / "input",
            )
            self.assertEqual(summary["languages"], ["eng_Latn", "deu_Latn"])
            english_report = json.loads(
                (root / "output" / "eng_Latn" / "bootstrap_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(english_report["rehydration"]["enabled"])
            self.assertEqual(
                english_report["counts"]["unique_train_documents"],
                english_report["counts"]["declared_train_documents_with_rehydration"],
            )
            bootstrap_manifest = json.loads(
                (root / "output" / "bootstrap_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                set(bootstrap_manifest["languages"]), {"eng_Latn", "deu_Latn"}
            )
            self.assertEqual(
                bootstrap_manifest["languages"]["eng_Latn"]["rehydration"][
                    "policy_source"
                ],
                "english_fineweb_no_rehydration",
            )
            self.assertEqual(
                bootstrap_manifest["languages"]["deu_Latn"]["rehydration"][
                    "policy_source"
                ],
                "pinned_upstream_weights",
            )
            self.assertFalse(
                bootstrap_manifest["languages"]["deu_Latn"]["rehydration"][
                    "identity_override"
                ]
            )

    def test_sampling_plan_derives_production_source_targets(self) -> None:
        base = BootstrapSettings(
            seed=42,
            validation_fraction=0.30,
            target_unique_documents=1,
            target_utf8_bytes=1,
            calibration_documents=1,
        )
        plan = {
            "plan_sha256": "f" * 64,
            "validation_cap_tokens": 10,
            "physical_pool": {
                "deu_Latn": {
                    "physical_target_tokens": 8_000,
                    "planned_source_documents": 100,
                    "estimated_unique_tokens": 100,
                    "estimated_validation_unique_tokens": 10,
                    "estimated_train_unique_tokens": 90,
                    "estimated_train_fraction": 0.90,
                    "calibration": {
                        "calibration_utf8_bytes": 4_000,
                        "calibration_tokens": 2_000,
                        "calibration_documents": 100,
                        "calibration_rehydrated_tokens": 2_000,
                        "validation_fraction": 0.20,
                    },
                }
            },
        }
        settings, derivation = bootstrap._derive_production_settings(
            base=base,
            language="deu_Latn",
            sampling_plan=plan,
            safety_margin=1.10,
        )
        self.assertEqual(settings.target_unique_documents, 123)
        self.assertEqual(settings.target_utf8_bytes, 19_556)
        self.assertEqual(derivation["physical_target_tokens"], 8_000)
        self.assertEqual(derivation["validation_candidate_fraction"], 0.30)
        self.assertEqual(derivation["validation_target_fraction"], 0.20)
        self.assertEqual(derivation["estimated_train_survival_fraction"], 0.90)

        with self.assertRaisesRegex(ValueError, "invalid production targets"):
            bootstrap._derive_production_settings(
                base=BootstrapSettings(
                    seed=42,
                    validation_fraction=0.20,
                    target_unique_documents=1,
                    target_utf8_bytes=1,
                    calibration_documents=1,
                ),
                language="deu_Latn",
                sampling_plan=plan,
                safety_margin=1.10,
            )

        drifted_plan = json.loads(json.dumps(plan))
        drifted = drifted_plan["physical_pool"]["deu_Latn"]
        drifted["estimated_validation_unique_tokens"] = 20
        drifted["estimated_train_unique_tokens"] = 80
        drifted["estimated_train_fraction"] = 0.80
        with self.assertRaisesRegex(ValueError, "invalid production targets"):
            bootstrap._derive_production_settings(
                base=base,
                language="deu_Latn",
                sampling_plan=drifted_plan,
                safety_margin=1.10,
            )

    def test_low_inventory_caps_generic_targets_and_uses_all_train_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                inventory_documents=9,
                inventory_utf8_bytes=100,
            )
            self._write_parquet(
                root,
                "deu_Latn",
                [
                    {
                        "id": f"low-{index}",
                        "text": f"low availability document {index}",
                        "url": "",
                        "minhash_cluster_size": 1,
                    }
                    for index in range(9)
                ],
            )
            report = bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "output",
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=0.20,
                    target_unique_documents=10_000,
                    target_utf8_bytes=64 * 1024 * 1024,
                    calibration_documents=1_000,
                ),
                local_input_path=root / "input",
            )
            self.assertTrue(report["targets"]["inventory_limited"])
            self.assertEqual(
                report["targets"]["effective_after_inventory_cap"][
                    "target_unique_documents"
                ],
                9,
            )
            self.assertEqual(
                report["targets"]["effective_after_inventory_cap"][
                    "target_utf8_bytes"
                ],
                100,
            )
            self.assertTrue(report["targets"]["source_exhausted"])
            self.assertTrue(report["targets"]["pinned_inventory_exhausted"])
            self.assertEqual(
                report["counts"]["calibration_documents"],
                report["counts"]["unique_train_documents"],
            )

    def test_incremental_extension_rebuilds_new_root_and_preserves_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(root)
            rows = [
                {
                    "id": f"doc-{index}",
                    "text": f"incremental source document {index}",
                    "url": "",
                    "minhash_cluster_size": 1,
                }
                for index in range(50)
            ]
            self._write_parquet(root, "deu_Latn", rows)
            predecessor_root = root / "bootstrap-v1"
            bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=predecessor_root,
                settings=self._small_settings(documents=20),
                local_input_path=root / "input",
            )
            predecessor_hashes = {
                path.name: _sha256(path)
                for path in (predecessor_root / "deu_Latn").glob("*.jsonl")
            }

            expanded = bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "bootstrap-v2",
                settings=self._small_settings(documents=40),
                local_input_path=root / "input",
                incremental_from_root=predecessor_root,
            )
            self.assertEqual(expanded["counts"]["unique_source_documents"], 40)
            self.assertEqual(
                expanded["run_config"]["incremental_extension"]["mode"],
                "verified_full_deterministic_rebuild",
            )
            self.assertEqual(
                predecessor_hashes,
                {
                    path.name: _sha256(path)
                    for path in (predecessor_root / "deu_Latn").glob("*.jsonl")
                },
            )
            retained = bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=root / "bootstrap-retained",
                settings=self._small_settings(documents=10),
                local_input_path=root / "input",
                incremental_from_root=predecessor_root,
            )
            self.assertEqual(retained["counts"]["unique_source_documents"], 20)
            self.assertEqual(
                retained["run_config"]["settings"]["target_unique_documents"],
                20,
            )
            self.assertNotIn(
                "language_selection_map_sha256", retained["run_config"]
            )

    def test_incremental_selection_keeps_identity_map_hash_and_binds_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("deu_Latn", "fra_Latn"),
                thresholds={
                    "deu_Latn": {"1": 1},
                    "fra_Latn": {"1": 1},
                },
            )
            selection_map = root / "selection-map.json"
            _write_json(
                selection_map,
                {
                    "schema_version": 2,
                    "language_set_id": "incremental-selection",
                    "0": {"languages": ["deu_Latn"]},
                },
            )
            self._write_parquet(
                root,
                "deu_Latn",
                [
                    {
                        "id": f"doc-{index}",
                        "text": f"incremental selected document {index}",
                        "url": "",
                        "minhash_cluster_size": 1,
                    }
                    for index in range(50)
                ],
            )
            predecessor_root = root / "bootstrap-v1"
            bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=predecessor_root,
                settings=self._small_settings(documents=20),
                local_input_path=root / "input",
            )

            successor_root = root / "bootstrap-v2"
            report = bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                language_selection_map_path=selection_map,
                output_root=successor_root,
                settings=self._small_settings(documents=40),
                local_input_path=root / "input",
                incremental_from_root=predecessor_root,
            )
            self.assertEqual(
                report["run_config"]["language_map_sha256"], _sha256(language_map)
            )
            self.assertEqual(
                report["run_config"]["language_selection_map_sha256"],
                _sha256(selection_map),
            )
            with self.assertRaisesRegex(
                RuntimeError, "language_selection_map_sha256"
            ):
                bootstrap.merge_bootstrap_release(
                    source_lock_path=lock,
                    rehydration_weights_path=weights,
                    language_map_path=language_map,
                    output_root=successor_root,
                    source_inventory_path=(
                        weights.parent / "fineweb2-language-distribution.csv"
                    ),
                    incremental_from_root=predecessor_root,
                )
            summary = bootstrap.merge_bootstrap_release(
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                language_selection_map_path=selection_map,
                output_root=successor_root,
                source_inventory_path=(
                    weights.parent / "fineweb2-language-distribution.csv"
                ),
                incremental_from_root=predecessor_root,
            )
            self.assertEqual(summary["languages"], ["deu_Latn"])
            manifest = json.loads(
                (successor_root / "bootstrap_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["provenance"]["language_map_sha256"], _sha256(language_map)
            )
            self.assertEqual(
                manifest["provenance"]["language_selection_map_sha256"],
                _sha256(selection_map),
            )

    def test_incremental_still_rejects_selector_used_as_identity_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("deu_Latn", "fra_Latn"),
                thresholds={
                    "deu_Latn": {"1": 1},
                    "fra_Latn": {"1": 1},
                },
            )
            subset_identity = root / "subset-identity.json"
            _write_json(
                subset_identity,
                {
                    "schema_version": 2,
                    "language_set_id": "wrong-identity",
                    "0": {"languages": ["deu_Latn"]},
                },
            )
            self._write_parquet(
                root,
                "deu_Latn",
                [
                    {
                        "id": f"doc-{index}",
                        "text": f"identity mismatch document {index}",
                        "url": "",
                        "minhash_cluster_size": 1,
                    }
                    for index in range(30)
                ],
            )
            predecessor_root = root / "bootstrap-v1"
            bootstrap_language(
                language="deu_Latn",
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=predecessor_root,
                settings=self._small_settings(documents=20),
                local_input_path=root / "input",
            )
            with self.assertRaisesRegex(RuntimeError, "language_map_sha256"):
                bootstrap_language(
                    language="deu_Latn",
                    source_lock_path=lock,
                    rehydration_weights_path=weights,
                    language_map_path=subset_identity,
                    output_root=root / "bootstrap-v2",
                    settings=self._small_settings(documents=25),
                    local_input_path=root / "input",
                    incremental_from_root=predecessor_root,
                )

    def test_array_tasks_only_write_language_dirs_then_merge_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(
                root,
                languages=("eng_Latn", "deu_Latn"),
            )
            for language in ("eng_Latn", "deu_Latn"):
                self._write_parquet(
                    root,
                    language,
                    [
                        {
                            "id": f"{language}-{index}",
                            "text": f"{language} array document {index}",
                            "url": "",
                            **(
                                {}
                                if language == "eng_Latn"
                                else {"minhash_cluster_size": 1}
                            ),
                        }
                        for index in range(20)
                    ],
                )
            output = root / "array-output"
            common = [
                "bootstrap_fineweb_v2",
                "--source-lock",
                str(lock),
                "--rehydration-weights",
                str(weights),
                "--language-map",
                str(language_map),
                "--output-dir",
                str(output),
                "--input-path",
                str(root / "input"),
                "--target-unique-documents",
                "20",
                "--target-utf8-bytes",
                "1",
                "--calibration-documents",
                "5",
            ]
            with patch.object(sys, "argv", common + ["--language-index", "0"]):
                self.assertEqual(bootstrap.main(), 0)
            self.assertTrue((output / "eng_Latn" / "bootstrap_report.json").is_file())
            self.assertFalse((output / "deu_Latn").exists())
            self.assertFalse((output / "bootstrap_summary.json").exists())

            with (
                patch.object(sys, "argv", common + ["--language-index"]),
                patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}),
            ):
                self.assertEqual(bootstrap.main(), 0)
            self.assertTrue((output / "deu_Latn" / "bootstrap_report.json").is_file())
            self.assertFalse((output / "bootstrap_summary.json").exists())

            with patch.object(sys, "argv", common + ["--merge-summary"]):
                self.assertEqual(bootstrap.main(), 0)
            self.assertTrue((output / "bootstrap_summary.json").is_file())
            self.assertTrue((output / "bootstrap_manifest.json").is_file())
            manifest = json.loads(
                (output / "bootstrap_manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "language_selection_map_sha256", manifest["provenance"]
            )

    def test_bootstrap_manifest_is_consumed_by_finalizer_end_to_end(self) -> None:
        try:
            from multilingual_megatron.pipeline.stages.sampling.tools import (
                finalize_fineweb_v2_corpus as finalizer,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"finalizer runtime dependencies unavailable: {exc}")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, weights, language_map = self._fixture_config(root)
            rows = [
                {
                    "id": f"doc-{index}",
                    "text": f"alpha beta gamma multilingual document number {index}",
                    "url": f"https://example.test/{index}",
                    "minhash_cluster_size": 2 if index % 2 else 1,
                }
                for index in range(100)
            ]
            self._write_parquet(root, "deu_Latn", rows)
            bootstrap_root = root / "bootstrap"
            bootstrap_release(
                source_lock_path=lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=bootstrap_root,
                settings=BootstrapSettings(
                    seed=42,
                    validation_fraction=0.30,
                    target_unique_documents=100,
                    target_utf8_bytes=1,
                    calibration_documents=5,
                ),
                selected_language="all",
                local_input_path=root / "input",
            )
            tokenizer = self._train_fixture_tokenizer(root, rows)
            plan = {
                "schema_version": 1,
                "policy_id": "fineweb_ab_v2_alpha03_rehydrated",
                "sampling_seed": 42,
                "alpha": 0.3,
                "train_tokens_per_set": 1,
                "validation_cap_tokens": 1,
                "tiers": {
                    "K1": {
                        "languages": ["deu_Latn"],
                        "target_exposure_tokens": {"deu_Latn": 1},
                    }
                },
                "physical_pool": {
                    "deu_Latn": {"physical_target_tokens": 1}
                },
                "provenance": {"tokenizer_model_sha256": sha256_file(tokenizer)},
            }
            plan["plan_sha256"] = canonical_json_sha256(plan)
            plan_path = root / "sampling-plan.json"
            _write_json(plan_path, plan)
            finalized_root = root / "finalized"
            final_manifest = finalizer.finalize_corpus(
                bootstrap_manifest_path=bootstrap_root / "bootstrap_manifest.json",
                tokenizer_model_path=tokenizer,
                sampling_plan_path=plan_path,
                output_dir=finalized_root,
            )
            self.assertTrue(final_manifest["ready"])

            selection_namespace = "physical-train-document-selection-v1"
            original_train = _read_jsonl(
                bootstrap_root / "deu_Latn" / "train.jsonl"
            )
            candidates = _read_jsonl(
                bootstrap_root / "deu_Latn" / "validation_reserve.jsonl"
            )
            final_validation_ids = {
                row["original_document_id"]
                for row in _read_jsonl(
                    finalized_root / "validation" / "deu_Latn.jsonl"
                )
            }
            compact_train = sorted(
                original_train
                + [
                    row
                    for row in candidates
                    if row["document_id"] not in final_validation_ids
                ],
                key=lambda row: hashlib.sha256(
                    f"42\0{selection_namespace}\0{row['document_id']}".encode("utf-8")
                ).hexdigest(),
            )
            selected = compact_train[0]
            emitted = _read_jsonl(finalized_root / "train" / "deu_Latn.jsonl")
            expected_multiplier = selected["lineage"].get(
                "train_rehydration_multiplier",
                selected["lineage"]["rehydration_multiplier"],
            )
            self.assertEqual(
                len(emitted), expected_multiplier
            )
            self.assertEqual(
                {row["original_document_id"] for row in emitted},
                {selected["document_id"]},
            )


if __name__ == "__main__":
    unittest.main()
