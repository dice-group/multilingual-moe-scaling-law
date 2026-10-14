from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class MegatronTokenizationArrayTest(unittest.TestCase):
    def test_alias_resolution_prefers_workers_over_num_proc(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.config import resolve_megatron_tokenization_config

        cfg = resolve_megatron_tokenization_config(
            {
                "backend": "megatron_gpt",
                "model_name": "tokenizer.model",
                "tokenizer_type": "SentencePieceTokenizer",
                "num_proc": 4,
                "workers": 16,
                "megatron_slurm_array_enabled": True,
                "megatron_slurm_array_ranks": 32,
                "megatron_slurm_array_concurrency": 8,
                "physical_split_enabled": True,
                "train_fraction": 0.9,
                "valid_fraction": 0.1,
            },
            output_dir=Path("out"),
        )

        self.assertEqual(cfg.mode, "slurm_array")
        self.assertEqual(cfg.array_ranks, 32)
        self.assertEqual(cfg.array_spec, "0-31%8")
        self.assertEqual(cfg.workers, 16)
        self.assertEqual(cfg.tokenizer_type, "SentencePieceTokenizer")
        self.assertAlmostEqual(cfg.train_fraction, 0.9)
        self.assertAlmostEqual(cfg.validation_fraction, 0.1)

    def test_canonical_alias_conflict_fails(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.config import resolve_megatron_tokenization_config

        with self.assertRaises(SystemExit):
            resolve_megatron_tokenization_config(
                {
                    "backend": "megatron_gpt",
                    "model_name": "old.model",
                    "tokenizer": {"model": "new.model"},
                },
                output_dir=Path("out"),
            )

    def test_worker_command_resolves_repo_local_sentencepiece_model(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.config import resolve_megatron_tokenization_config
        from multilingual_megatron.pipeline.stages.tokenization.tokenize_stage import _megatron_worker_cmd

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            model = repo_root / "outputs" / "tokenizers" / "tok" / "tokenizer.model"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"spm")
            cfg = resolve_megatron_tokenization_config(
                {
                    "backend": "megatron_gpt",
                    "model_name": "outputs/tokenizers/tok/tokenizer.model",
                    "tokenizer_type": "SentencePieceTokenizer",
                    "megatron_slurm_array_enabled": True,
                },
                output_dir=Path("out"),
            )

            cmd = _megatron_worker_cmd(
                SimpleNamespace(repo_root=repo_root),
                cfg=cfg,
                plan_dir=repo_root / "plan",
                ranks_dir=repo_root / "ranks",
                include_rank_arg=True,
                rank=0,
            )

            model_index = cmd.index("--tokenizer-model") + 1
            self.assertEqual(cmd[model_index], str(model.resolve()))

    def test_missing_repo_local_sentencepiece_model_fails_early(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.config import resolve_megatron_tokenization_config
        from multilingual_megatron.pipeline.stages.tokenization.tokenize_stage import _megatron_worker_cmd

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            cfg = resolve_megatron_tokenization_config(
                {
                    "backend": "megatron_gpt",
                    "model_name": "outputs/tokenizers/missing/tokenizer.model",
                    "tokenizer_type": "SentencePieceTokenizer",
                    "megatron_slurm_array_enabled": True,
                },
                output_dir=Path("out"),
            )

            with self.assertRaises(SystemExit):
                _megatron_worker_cmd(
                    SimpleNamespace(repo_root=repo_root),
                    cfg=cfg,
                    plan_dir=repo_root / "plan",
                    ranks_dir=repo_root / "ranks",
                    include_rank_arg=True,
                    rank=0,
                )

    def test_plan_balances_sources_across_ranks(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.megatron_distributed import _cmd_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "sampled" / "samples"
            for language, count in {"eng_Latn": 3, "deu_Latn": 2}.items():
                lang_dir = shard_dir / language
                lang_dir.mkdir(parents=True)
                with (lang_dir / "00000.jsonl").open("w", encoding="utf-8") as handle:
                    for idx in range(count):
                        handle.write(json.dumps({"text": f"{language} {idx}", "language": language}) + "\n")
            output_dir = root / "tokenized"
            plan_dir = root / "plan"

            _cmd_plan(
                SimpleNamespace(
                    shard_dir=str(shard_dir),
                    output_dir=str(output_dir),
                    plan_dir=str(plan_dir),
                    languages="eng_Latn,deu_Latn",
                    array_ranks=2,
                    sample_fraction=1.0,
                    train_fraction=0.9,
                    validation_fraction=0.1,
                )
            )

            summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["array_ranks"], 2)
            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["document_count"], 5)
            self.assertTrue((plan_dir / "rank_00000.jsonl").exists())
            self.assertTrue((plan_dir / "rank_00001.jsonl").exists())

    def test_merge_writes_bridge_manifest_with_splits(self) -> None:
        from multilingual_megatron.pipeline.stages.tokenization.megatron_distributed import _cmd_merge

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranks_dir = root / "ranks"
            output_dir = root / "tokenized"
            train_prefix = root / "ranks" / "rank_00000" / "megatron" / "train" / "eng_Latn_text_document"
            valid_prefix = root / "ranks" / "rank_00000" / "megatron" / "validation" / "eng_Latn_text_document"
            for prefix in (train_prefix, valid_prefix):
                prefix.parent.mkdir(parents=True, exist_ok=True)
                Path(str(prefix) + ".bin").write_bytes(b"bin")
                Path(str(prefix) + ".idx").write_bytes(b"idx")
            result = {
                "manifests": {
                    "train": {
                        "prefixes_by_language": {"eng_Latn": [str(train_prefix)]},
                        "documents_per_language": {"eng_Latn": 3},
                    },
                    "validation": {
                        "prefixes_by_language": {"eng_Latn": [str(valid_prefix)]},
                        "documents_per_language": {"eng_Latn": 1},
                    },
                }
            }
            result_path = ranks_dir / "rank_00000" / "result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result), encoding="utf-8")

            _cmd_merge(
                SimpleNamespace(
                    ranks_dir=str(ranks_dir),
                    output_dir=str(output_dir),
                    tokenizer_type="SentencePieceTokenizer",
                    tokenizer_model="tokenizer.model",
                    trust_remote_code=False,
                    blend_weight_policy="proportional_documents",
                    check_prefix_files=True,
                )
            )

            manifest = json.loads((output_dir / "megatron_data_manifest.json").read_text(encoding="utf-8"))
            report = json.loads((output_dir / "tokenization_report.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "megatron_gpt")
            self.assertEqual(manifest["prefixes"], [str(train_prefix)])
            self.assertEqual(
                manifest["splits"]["train"]["prefixes_by_language"],
                {"eng_Latn": [str(train_prefix)]},
            )
            self.assertEqual(manifest["splits"]["validation"]["prefixes"], [str(valid_prefix)])
            self.assertEqual(
                manifest["splits"]["validation"]["prefixes_by_language"],
                {"eng_Latn": [str(valid_prefix)]},
            )
            artifact = manifest["splits"]["validation"]["prefix_artifacts_by_language"]["eng_Latn"][0]
            self.assertEqual(artifact["prefix"], str(valid_prefix))
            self.assertEqual(artifact["bin_sha256"], hashlib.sha256(b"bin").hexdigest())
            self.assertEqual(artifact["idx_sha256"], hashlib.sha256(b"idx").hexdigest())
            self.assertTrue(report["pass"])


if __name__ == "__main__":
    unittest.main()
