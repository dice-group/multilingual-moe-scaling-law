from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TokenizerTrainingPipelineTest(unittest.TestCase):
    def test_training_config_resolves_canonical_execution_and_corpus(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenizer_training.config import (
            resolve_tokenizer_training_config,
        )

        cfg = resolve_tokenizer_training_config(
            {
                "enabled": True,
                "execution": {"python_executable": "python", "num_threads": 12, "parallel_slurm_jobs": True},
                "corpus": {"text_key": "content", "max_sample_fraction": 0.25},
                "vocab_size": 64000,
                "tokenizers": [
                    {
                        "id": "shared_k200",
                        "mode": "train_unigram",
                        "pre_tokenizer": "metaspace",
                        "tokenizer_type": "SentencePieceTokenizer",
                    }
                ],
            },
            tokenize_cfg={"backend": "megatron_gpt"},
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.python_executable, "python")
        self.assertEqual(cfg.num_threads, 12)
        self.assertTrue(cfg.parallel_slurm_jobs)
        self.assertEqual(len(cfg.specs), 1)
        spec = cfg.specs[0]
        self.assertEqual(spec.id, "shared_k200")
        self.assertEqual(spec.mode, "train_unigram")
        self.assertEqual(spec.text_key, "content")
        self.assertEqual(spec.max_sample_fraction, 0.25)
        self.assertEqual(spec.vocab_size, 64000)
        self.assertIsNone(spec.production_v2)

    def test_production_v2_config_is_explicit_and_fail_closed(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenizer_training.config import (
            resolve_tokenizer_training_config,
        )

        source_hash = "a" * 64
        cfg = resolve_tokenizer_training_config(
            {
                "enabled": True,
                "tokenizers": [
                    {
                        "id": "shared_v2",
                        "production_v2": {
                            "enabled": True,
                            "sentence_jsonl_path": "outputs/tokenizer/sentences.jsonl",
                            "sentence_jsonl_sha256": source_hash,
                        },
                    }
                ],
            },
            tokenize_cfg={"backend": "megatron_gpt"},
        )

        spec = cfg.specs[0]
        self.assertEqual(spec.mode, "train_unigram")
        self.assertEqual(spec.vocab_size, 64_000)
        self.assertEqual(spec.model_vocab_size, 64_000)
        self.assertEqual(spec.normalizer, "nmt_nfkc")
        self.assertEqual(spec.pre_tokenizer, "metaspace")
        self.assertEqual((spec.unk_token, spec.bos_token, spec.eos_token, spec.pad_token), (
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
        ))
        self.assertEqual(spec.special_tokens, ())
        self.assertIsNotNone(spec.production_v2)
        assert spec.production_v2 is not None
        self.assertEqual(spec.production_v2.sentence_jsonl_sha256, source_hash)
        self.assertEqual(spec.production_v2.expected_sentence_count, 100_000_000)

        with self.assertRaisesRegex(SystemExit, "production_v2 configuration mismatch"):
            resolve_tokenizer_training_config(
                {
                    "enabled": True,
                    "tokenizers": [
                        {
                            "id": "bad_v2",
                            "vocab_size": 32_000,
                            "production_v2": {
                                "enabled": True,
                                "sentence_jsonl_path": "sentences.jsonl",
                                "sentence_jsonl_sha256": source_hash,
                            },
                        }
                    ],
                },
                tokenize_cfg={},
            )

    def test_production_v2_command_consumes_only_frozen_sentence_artifact(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenizer_training.config import (
            resolve_tokenizer_training_config,
        )
        from multilingual_megatron.pipeline.stages.tokenizer_training.tokenizer_train_stage import (
            _build_tokenizer_train_cmd,
        )

        cfg = resolve_tokenizer_training_config(
            {
                "enabled": True,
                "tokenizers": [
                    {
                        "id": "shared_v2",
                        "production_v2": {
                            "enabled": True,
                            "sentence_jsonl_path": "frozen/sentences.jsonl",
                            "sentence_jsonl_sha256": "b" * 64,
                        },
                    }
                ],
            },
            tokenize_cfg={},
        )
        ctx = SimpleNamespace(repo_root=REPO_ROOT, sample_shard_dir=REPO_ROOT / "must_not_be_used")

        cmd = _build_tokenizer_train_cmd(
            ctx,
            cfg=cfg,
            spec=cfg.specs[0],
            output_dir=REPO_ROOT / "outputs" / "tokenizer_v2",
        )

        self.assertNotIn("--input-dir", cmd)
        self.assertIn("--production-v2-sentence-jsonl", cmd)
        self.assertEqual(cmd[cmd.index("--vocab-size") + 1], "64000")
        self.assertEqual(cmd[cmd.index("--normalizer") + 1], "nmt_nfkc")
        self.assertEqual(cmd[cmd.index("--pre-tokenizer") + 1], "metaspace")
        self.assertEqual(
            cmd[cmd.index("--production-v2-expected-sentence-count") + 1],
            "100000000",
        )

    def test_production_v2_manifest_keeps_model_source_and_effective_config_hashes(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenizer_training.config import (
            resolve_tokenizer_training_config,
        )
        from multilingual_megatron.pipeline.stages.tokenizer_training.tokenizer_train_stage import (
            _record_manifest_entry,
        )

        cfg = resolve_tokenizer_training_config(
            {
                "enabled": True,
                "tokenizers": [
                    {
                        "id": "shared_v2",
                        "production_v2": {
                            "enabled": True,
                            "sentence_jsonl_path": "frozen/sentences.jsonl",
                            "sentence_jsonl_sha256": "c" * 64,
                        },
                    }
                ],
            },
            tokenize_cfg={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained: dict[str, object] = {}
            ctx = SimpleNamespace(tokenizers_dir=root / "tokenizers")
            result = {
                "tokenizer_model": str(root / "tokenizer.model"),
                "tokenizer_type": "SentencePieceTokenizer",
                "vocab_size": 64_000,
                "production_v2": {
                    "enabled": True,
                    "source_artifact": {
                        "path": str(root / "sentences.jsonl"),
                        "sha256": "c" * 64,
                        "sentence_count": 100_000_000,
                    },
                },
                "source_files": [str(root / "sentences.jsonl")],
                "source_artifact_sha256": "c" * 64,
                "tokenizer_model_sha256": "d" * 64,
                "effective_config_sha256": "e" * 64,
                "effective_sentencepiece_config": {"schema": "production_sentencepiece_unigram_v2/v1"},
            }

            _record_manifest_entry(
                ctx,
                trained=trained,
                spec=cfg.specs[0],
                result=result,
            )

            entry = trained["shared_v2"]
            assert isinstance(entry, dict)
            self.assertEqual(entry["tokenizer_model_sha256"], "d" * 64)
            self.assertEqual(entry["source_artifact_sha256"], "c" * 64)
            self.assertEqual(entry["effective_config_sha256"], "e" * 64)
            self.assertEqual(
                entry["effective_sentencepiece_config"]["schema"],
                "production_sentencepiece_unigram_v2/v1",
            )

    def test_production_v2_sentence_jsonl_is_strict_hashed_and_ordered(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenizer_training.sentencepiece_v2 import (
            inspect_sentence_jsonl,
            iter_sentence_jsonl,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentences.jsonl"
            payload = (
                json.dumps({"text": "first"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"text": "δεύτερο"}, ensure_ascii=False)
                + "\n"
            )
            path.write_text(payload, encoding="utf-8")
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            audit = inspect_sentence_jsonl(
                path,
                text_key="text",
                expected_sha256=digest,
                expected_sentence_count=2,
            )

            self.assertEqual(audit["sha256"], digest)
            self.assertEqual(audit["sentence_count"], 2)
            self.assertEqual(list(iter_sentence_jsonl(path, text_key="text")), ["first", "δεύτερο"])
            with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                inspect_sentence_jsonl(
                    path,
                    text_key="text",
                    expected_sha256="0" * 64,
                    expected_sentence_count=2,
                )

    def test_production_v2_model_protobuf_effective_config_is_fully_audited(self) -> None:
        try:
            import sentencepiece as spm
            from sentencepiece import sentencepiece_model_pb2 as model_pb2
        except ModuleNotFoundError:
            self.skipTest("sentencepiece is not installed")

        from multilingual_megatron.pipeline.stages.tokenizer_training.sentencepiece_v2 import (
            audit_production_v2_model,
            production_v2_trainer_kwargs,
        )

        sentences = [
            "This is sentence "
            f"{index} café Ελληνικά русский 日本語 العربية हिंदी "
            f"{chr(0x400 + index % 100)} token_variant_{index % 400}"
            for index in range(5_000)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "tokenizer"
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(sentences),
                model_prefix=str(prefix),
                minloglevel=2,
                **production_v2_trainer_kwargs(vocab_size=512, num_threads=1),
            )
            model_path = prefix.with_suffix(".model")

            effective = audit_production_v2_model(
                model_path,
                expected_vocab_size=512,
                expected_num_threads=1,
                expected_model_prefix=str(prefix),
            )

            trainer = effective["trainer_spec"]
            normalizer = effective["normalizer_spec"]
            self.assertEqual(effective["piece_count"], 512)
            self.assertEqual(effective["piece_type_counts"]["BYTE"], 256)
            self.assertEqual(trainer["model_type"], "UNIGRAM")
            self.assertAlmostEqual(trainer["character_coverage"], 0.999995)
            self.assertFalse(trainer["shuffle_input_sentence"])
            self.assertTrue(trainer["hard_vocab_limit"])
            self.assertTrue(trainer["byte_fallback"])
            self.assertFalse(trainer["split_digits"])
            self.assertEqual((trainer["unk_id"], trainer["bos_id"], trainer["eos_id"], trainer["pad_id"]), (0, 1, 2, 3))
            self.assertEqual(normalizer["name"], "nmt_nfkc")
            self.assertTrue(normalizer["add_dummy_prefix"])
            self.assertTrue(normalizer["escape_whitespaces"])
            self.assertTrue(normalizer["remove_extra_whitespaces"])
            self.assertEqual(effective["inference"]["segmentation"], "one_best")
            self.assertEqual(len(effective["model_sha256"]), 64)
            self.assertEqual(len(effective["effective_config_sha256"]), 64)

            tampered = model_pb2.ModelProto()
            tampered.ParseFromString(model_path.read_bytes())
            tampered.trainer_spec.shuffle_input_sentence = True
            model_path.write_bytes(tampered.SerializeToString())
            with self.assertRaisesRegex(SystemExit, "TrainerSpec mismatch"):
                audit_production_v2_model(
                    model_path,
                    expected_vocab_size=512,
                    expected_num_threads=1,
                    expected_model_prefix=str(prefix),
                )

    def test_training_config_rejects_duplicate_tokenizer_ids(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenizer_training.config import (
            resolve_tokenizer_training_config,
        )

        with self.assertRaises(SystemExit):
            resolve_tokenizer_training_config(
                {
                    "enabled": True,
                    "tokenizers": [{"id": "dup"}, {"id": "dup"}],
                },
                tokenize_cfg={"model_name": "base"},
            )

    def test_manifest_driven_megatron_array_fans_out_per_tokenizer(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.config import (
            resolve_megatron_tokenization_config,
        )
        from multilingual_megatron.pipeline.stages.tokenization.tokenize_stage import _stage_tokenize_megatron_multi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizers_dir = root / "tokenizers"
            tokenizers_dir.mkdir()
            manifest = {
                "schema_version": 1,
                "stage": "tokenizer_train",
                "tokenizers": {
                    "shared": {
                        "path": str(root / "shared.model"),
                        "tokenizer_type": "SentencePieceTokenizer",
                        "trust_remote_code": False,
                        "vocab_size": 64000,
                    },
                    "ablation": {
                        "path": str(root / "ablation"),
                        "tokenizer_type": "HuggingFaceTokenizer",
                        "trust_remote_code": True,
                        "vocab_size": 32000,
                    },
                },
            }
            (tokenizers_dir / "tokenizer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            ctx = SimpleNamespace(
                args=SimpleNamespace(dry_run=True),
                env={},
                executor=None,
                force_stages=set(),
                logger=logging.getLogger("test"),
                logs_root=root / "logs",
                repo_root=REPO_ROOT,
                sample_shard_dir=root / "sampled",
                state_path=root / "state.json",
                tokenize_cfg={
                    "backend": "megatron_gpt",
                    "execution": {"mode": "local_array", "array_ranks": 2},
                    "languages": ["eng_Latn"],
                },
                tokenizer_cfg={
                    "enabled": True,
                    "tokenizers": [{"id": "shared"}, {"id": "ablation"}],
                },
                tokenized_dir=root / "tokenized",
                tokenizers_dir=tokenizers_dir,
                language_map_path=root / "language_map.json",
            )
            cfg = resolve_megatron_tokenization_config(
                ctx.tokenize_cfg,
                output_dir=ctx.tokenized_dir,
                require_tokenizer_model=False,
            )
            calls: list[tuple[str, Path, str, str, bool]] = []

            def fake_array(_ctx, tokenizer_cfg, _signature, **kwargs):
                calls.append(
                    (
                        kwargs["tokenizer_id"],
                        kwargs["output_dir"],
                        tokenizer_cfg.tokenizer_type,
                        tokenizer_cfg.tokenizer_model,
                        tokenizer_cfg.trust_remote_code,
                    )
                )

            with patch(
                "multilingual_megatron.pipeline.stages.tokenization.tokenize_stage._stage_tokenize_megatron_array",
                side_effect=fake_array,
            ):
                _stage_tokenize_megatron_multi(ctx, cfg, "sig")

            self.assertEqual([call[0] for call in calls], ["ablation", "shared"])
            self.assertEqual(calls[0][1], ctx.tokenized_dir / "ablation")
            self.assertEqual(calls[0][2], "HuggingFaceTokenizer")
            self.assertTrue(calls[0][4])
            self.assertEqual(calls[1][1], ctx.tokenized_dir / "shared")
            self.assertEqual(calls[1][2], "SentencePieceTokenizer")







if __name__ == "__main__":
    unittest.main()
