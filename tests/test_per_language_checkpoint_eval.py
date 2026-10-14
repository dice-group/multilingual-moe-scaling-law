from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import (  # noqa: E402
    EVALUATION_SPEC_SCHEMA,
    ExactLanguageDataset,
    TokenStream,
    configure_checkpoint_evaluation,
    load_evaluation_spec,
    verify_evaluation_model_contract,
)
from multilingual_megatron.pipeline.backends.megatron_bridge.run_megatron_bridge_per_language_validation import (  # noqa: E402,E501
    build_validation_command,
)


class InferenceSetupTest(unittest.TestCase):
    def test_shared_setup_skips_training_allocations_and_restores_hooks(self):
        from multilingual_megatron.pipeline.backends.megatron_bridge.checkpoint_inference import setup_bridge_inference_only
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import _setup_bridge_inference_only
        self.assertIs(_setup_bridge_inference_only, setup_bridge_inference_only)
        for fail in (False, True):
            provider = Mock()
            cfg = SimpleNamespace(peft=None, dist=SimpleNamespace(use_megatron_fsdp=False, use_torch_fsdp2=False),
                validation=SimpleNamespace(skip_train=True), checkpoint=SimpleNamespace(load_optim=False, load_rng=False, save=None),
                rng=SimpleNamespace(data_parallel_random_init=False), model=SimpleNamespace(provide_distributed_model=Mock(return_value=['model'])))
            original = [Mock(side_effect=AssertionError('training allocation called')) for _ in range(3)]
            bridge = SimpleNamespace(ModelConfig=type('ModelConfig', (), {}), _build_distributed_model=original[0],
                setup_optimizer=original[1], _update_model_config_funcs=original[2])
            state = SimpleNamespace(cfg=cfg)
            def setup(given, datasets):
                self.assertIs(given, state)
                self.assertIs(datasets, provider)
                self.assertEqual(bridge._build_distributed_model(cfg, 'groups'), ['model'])
                self.assertEqual(bridge.setup_optimizer(), (None, None))
                bridge._update_model_config_funcs()
                if fail:
                    raise RuntimeError('checkpoint load failed')
                return SimpleNamespace(optimizer=None, scheduler=None)
            bridge.setup = setup
            with patch('importlib.import_module', return_value=bridge):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'checkpoint load failed'):
                        setup_bridge_inference_only(state, provider)
                else:
                    setup_bridge_inference_only(state, provider)
            self.assertFalse(cfg.model.provide_distributed_model.call_args.kwargs['wrap_with_ddp'])
            self.assertIsNone(cfg.model.provide_distributed_model.call_args.kwargs['ddp_config'])
            self.assertEqual([bridge._build_distributed_model, bridge.setup_optimizer, bridge._update_model_config_funcs], original)
            cfg.validation.skip_train = False
            with patch('importlib.import_module', return_value=bridge), self.assertRaisesRegex(RuntimeError, 'skip_train'):
                setup_bridge_inference_only(state, provider)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cohorts() -> dict[str, list[str]]:
    additions = [f"lang_{index:03d}" for index in range(1, 50)]
    return {
        "english": ["eng_Latn"],
        "k12": ["eng_Latn", *additions[:11]],
        "k50": ["eng_Latn", *additions],
    }


class _SequenceSource:
    def __init__(self, documents: list[list[int]]) -> None:
        self.documents = documents
        self.sequence_lengths = [len(document) for document in documents]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> list[int]:
        return self.documents[index]

    def get(self, index: int, offset: int = 0, length: int | None = None) -> list[int]:
        values = self.documents[index]
        return values[offset:] if length is None else values[offset : offset + length]


class PerLanguageCheckpointEvalTest(unittest.TestCase):
    def _spec_fixture(self, root: Path, *, required: list[str] | None = None) -> Path:
        required = required or ["eng_Latn"]
        tokenizer = root / "tokenizer.model"
        tokenizer.write_bytes(b"frozen sentencepiece model")
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("17\n", encoding="utf-8")
        (checkpoint / "iter_0000017").mkdir()

        prefixes: dict[str, list[str]] = {}
        prefix_artifacts: dict[str, list[dict[str, object]]] = {}
        corpus_languages: dict[str, object] = {}
        for index, language in enumerate(required):
            prefix = root / "tokenized" / language
            prefix.parent.mkdir(parents=True, exist_ok=True)
            bin_path = Path(str(prefix) + ".bin")
            idx_path = Path(str(prefix) + ".idx")
            bin_path.write_bytes(f"bin-{language}".encode())
            idx_path.write_bytes(f"idx-{language}".encode())
            prefixes[language] = [str(prefix)]
            prefix_artifacts[language] = [
                {
                    "prefix": str(prefix),
                    "bin_sha256": _sha(bin_path),
                    "bin_size_bytes": bin_path.stat().st_size,
                    "idx_sha256": _sha(idx_path),
                    "idx_size_bytes": idx_path.stat().st_size,
                }
            ]
            unigram = root / "unigram_counts" / f"{language}.json"
            _write_json(
                unigram,
                {
                    "language": language,
                    "tokenizer_model_sha256": _sha(tokenizer),
                    "special_token_ids_excluded": [2, 3],
                    "counts_by_token_id": {"0": index + 2, "1": 1},
                },
            )
            corpus_languages[language] = {
                "train": {
                    "unigram_counts": {
                        "path": str(unigram.relative_to(root)),
                        "sha256": _sha(unigram),
                        "size_bytes": unigram.stat().st_size,
                    }
                }
            }

        data_manifest = root / "megatron_data_manifest.json"
        _write_json(
            data_manifest,
            {
                "format": "megatron_gpt",
                "tokenizer_type": "SentencePieceTokenizer",
                "tokenizer_model": str(tokenizer),
                "splits": {
                    "validation": {
                        "prefixes_by_language": prefixes,
                        "prefix_artifacts_by_language": prefix_artifacts,
                    }
                },
            },
        )
        corpus_manifest = root / "corpus_manifest.json"
        _write_json(
            corpus_manifest,
            {
                "ready": True,
                "release_status": "ready",
                "inputs": {
                    "tokenizer_model": {
                        "path": str(tokenizer.relative_to(root)),
                        "sha256": _sha(tokenizer),
                        "vocabulary_size": 4,
                        "special_token_ids": [2, 3],
                    }
                },
                "languages": corpus_languages,
            },
        )
        spec_path = root / "eval_spec.json"
        _write_json(
            spec_path,
            {
                "schema": EVALUATION_SPEC_SCHEMA,
                "run_id": "ab_v2_test",
                "model_recipe": {
                    "name": "scaling_dense_scale_scaling_a_pretrain_config",
                    "overrides": [],
                    "expected_model": {
                        "num_layers": 8,
                        "hidden_size": 512,
                        "ffn_hidden_size": 1536,
                        "num_attention_heads": 8,
                        "seq_length": 4,
                        "vocab_size": 4,
                        "num_moe_experts": None,
                    },
                },
                "output_dir": "results",
                "sequence_length": 4,
                "eval_micro_batch_size": 32,
                "context_token_id": 2,
                "trained_languages": ["eng_Latn"],
                "required_languages": required,
                "cohorts": _cohorts(),
                "checkpoint": {"path": str(checkpoint), "step": 17},
                "validation_data_manifest": {
                    "path": str(data_manifest),
                    "sha256": _sha(data_manifest),
                },
                "corpus_manifest": {
                    "path": str(corpus_manifest),
                    "sha256": _sha(corpus_manifest),
                },
            },
        )
        return spec_path

    def test_exact_dataset_scores_each_non_special_target_once(self) -> None:
        stream = TokenStream(
            [
                _SequenceSource([[1, 2], [3]]),
                _SequenceSource([[0, 1]]),
            ]
        )
        dataset = ExactLanguageDataset(
            stream,
            sequence_length=3,
            context_token_id=2,
            special_token_ids=[2, 3],
            sample_multiple=4,
        )

        self.assertEqual(stream.read(1, 4).tolist(), [2, 3, 0, 1])
        self.assertEqual(dataset.natural_sample_count, 2)
        self.assertEqual(len(dataset), 4)
        first = dataset[0]
        second = dataset[1]
        padded = dataset[2]
        self.assertEqual(first["tokens"].tolist(), [2, 1, 2])
        self.assertEqual(first["labels"].tolist(), [1, 2, 3])
        self.assertEqual(first["loss_mask"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(second["tokens"].tolist(), [3, 0, 0])
        self.assertEqual(second["labels"].tolist(), [0, 1, 0])
        self.assertEqual(second["loss_mask"].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(padded["loss_mask"].sum().item(), 0.0)
        self.assertEqual(first["input_valid_mask"].tolist(), [True, True, True])
        self.assertEqual(second["input_valid_mask"].tolist(), [True, True, False])
        self.assertEqual(padded["input_valid_mask"].tolist(), [False, False, False])
        self.assertEqual(second["sample_id"].item(), 1)
        self.assertEqual(list(dataset.iter_non_special_targets()), [1, 0, 1])

    def test_spec_preflight_freezes_complete_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = load_evaluation_spec(self._spec_fixture(root))

        self.assertEqual(spec.run_id, "ab_v2_test")
        self.assertEqual(spec.checkpoint_step, 17)
        self.assertEqual(spec.checkpoint_path.name, "checkpoint")
        self.assertEqual(spec.checkpoint_iteration_path.name, "iter_0000017")
        self.assertEqual(spec.eval_micro_batch_size, 32)
        self.assertEqual(spec.trained_languages, ("eng_Latn",))
        self.assertEqual(spec.required_languages, ("eng_Latn",))
        self.assertEqual(spec.special_token_ids, (2, 3))
        self.assertEqual(set(spec.validation_prefixes), {"eng_Latn"})

    def test_explicit_diagnostic_subset_cannot_claim_complete_release(self) -> None:
        from multilingual_megatron.pipeline.stages.final_loss.per_language_checkpoint_eval import diagnostic_manifest
        with tempfile.TemporaryDirectory() as tmp:
            path = self._spec_fixture(Path(tmp))
            raw = json.loads(path.read_text())
            raw["diagnostic_languages"] = ["eng_Latn"]
            path.write_text(json.dumps(raw))
            spec = load_evaluation_spec(path)
            self.assertEqual(spec.diagnostic_languages, ("eng_Latn",))
            result = diagnostic_manifest(spec, {"eng_Latn": {}}, {})
            self.assertEqual(result["status"], "complete_diagnostic")
            self.assertFalse(result["coverage"]["complete_full_evaluation"])
            self.assertNotIn("validation_report", result)
            with self.assertRaisesRegex(ValueError, "incomplete diagnostic"):
                diagnostic_manifest(spec, {}, {})
            raw["diagnostic_languages"] = ["unconfigured_Latn"]
            invalid_path = Path(tmp)/"invalid_spec.json"
            invalid_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "subset"):
                load_evaluation_spec(invalid_path)

    def test_dense_recipe_contract_and_checkpoint_root_are_applied_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = load_evaluation_spec(self._spec_fixture(Path(tmp)))
            model = SimpleNamespace(**spec.model_contract)
            config = SimpleNamespace(
                model=model,
                checkpoint=SimpleNamespace(),
                validation=SimpleNamespace(),
                tokenizer=SimpleNamespace(),
            )

            configure_checkpoint_evaluation(config, spec)

        self.assertIsNone(config.model.num_moe_experts)
        self.assertEqual(config.checkpoint.load, str(spec.checkpoint_path))
        self.assertNotEqual(config.checkpoint.load, str(spec.checkpoint_iteration_path))
        self.assertEqual(config.checkpoint.ckpt_step, 17)
        self.assertIsNone(config.checkpoint.save)
        self.assertFalse(config.checkpoint.load_optim)
        self.assertFalse(config.checkpoint.load_rng)
        self.assertTrue(config.validation.skip_train)

    def test_moe_model_contract_catches_wrong_expert_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = self._spec_fixture(root)
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
            payload["model_recipe"] = {
                "name": "scaling_switch_moe_scale_scaling_b_pretrain_config",
                "overrides": ["model.num_moe_experts=32"],
                "expected_model": {
                    "num_layers": 12,
                    "hidden_size": 768,
                    "ffn_hidden_size": 2304,
                    "num_attention_heads": 12,
                    "seq_length": 2048,
                    "vocab_size": 64000,
                    "num_moe_experts": 32,
                    "moe_router_topk": 1,
                    "moe_expert_capacity_factor": None,
                    "moe_pad_expert_input_to_capacity": False,
                },
            }
            _write_json(spec_path, payload)
            spec = load_evaluation_spec(spec_path)
            model = SimpleNamespace(**spec.model_contract)
            config = SimpleNamespace(model=model)
            verify_evaluation_model_contract(config, spec)
            model.num_moe_experts = 16

            with self.assertRaisesRegex(ValueError, "num_moe_experts:16!=32"):
                verify_evaluation_model_contract(config, spec)

    def test_missing_required_validation_language_fails_before_gpu_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = self._spec_fixture(root)
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
            payload["required_languages"] = ["eng_Latn", "lang_001"]
            _write_json(spec_path, payload)

            with self.assertRaisesRegex(ValueError, "missing validation prefixes.*lang_001"):
                load_evaluation_spec(spec_path)

    def test_tampered_prefix_fails_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = self._spec_fixture(root)
            Path(str(root / "tokenized" / "eng_Latn") + ".bin").write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "(?:size|SHA-256) mismatch"):
                load_evaluation_spec(spec_path)

    def test_production_launcher_command_selects_load_once_evaluator(self) -> None:
        args = SimpleNamespace(
            launcher="python",
            python_executable="python",
            nnodes=1,
            nproc_per_node=1,
            recipe="scaling_dense_scale_scaling_a_pretrain_config",
            spec="/tmp/eval.json",
            extra_override=[],
        )

        with patch.dict("os.environ", {"MASTER_PORT": "23456"}):
            command = build_validation_command(
                args,
                sequence_length=2048,
                recipe_name="scaling_dense_scale_scaling_a_pretrain_config",
                recipe_overrides=(),
            )

        self.assertIn("torch.distributed.run", command)
        self.assertIn("--per-language-validation-spec", command)
        self.assertIn("/tmp/eval.json", command)
        self.assertEqual(command[command.index("--seq_length") + 1], "2048")


if __name__ == "__main__":
    unittest.main()
