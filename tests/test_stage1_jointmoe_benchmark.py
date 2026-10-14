from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
HARNESS_ROOT = REPO_ROOT / "external/lm-evaluation-harness"
for path in (REPO_ROOT, SRC_ROOT, HARNESS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


PROFILE_PATH = REPO_ROOT / "configs/benchmarks/stage1_abcd_jointmoe_final_a40.json"
SLURM_PATH = (
    REPO_ROOT / "configs/benchmarks/stage1_abcd_jointmoe_final_a40_slurm.json"
)


class Stage1JointMoEBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
            load_campaign_profile,
        )

        cls.profile = load_campaign_profile(PROFILE_PATH)
        cls.models = list(cls.profile.payload["models"])

    def test_profile_resolves_only_the_declared_final_checkpoint_cohorts(self) -> None:
        self.assertEqual(len(self.models), 82)
        self.assertEqual(
            Counter(model["source_cohort"] for model in self.models),
            {"canonical_release": 64, "b_seed_confirmation": 18},
        )
        self.assertEqual(
            Counter(model["ladder"] for model in self.models),
            {"A": 16, "B": 34, "C": 16, "D": 16},
        )
        self.assertTrue(all(model["checkpoint_step"] > 0 for model in self.models))
        self.assertTrue(
            all(
                model["tokens_seen"] >= model["nominal_tokens"]
                for model in self.models
            )
        )

        replicated = {
            (model["experts"], model["languages"], model["seed"])
            for model in self.models
            if model["ladder"] == "B"
            and model["experts"] in {1, 16, 32}
            and model["languages"] in {1, 12, 200}
        }
        self.assertEqual(
            replicated,
            {
                (experts, languages, seed)
                for experts in (1, 16, 32)
                for languages in (1, 12, 200)
                for seed in (1234, 5678, 91011)
            },
        )

    def test_lr_topology_and_calibration_contracts_match_sealed_models(self) -> None:
        tolerance = self.profile.payload["learning_rate_contract"]["relative_tolerance"]
        for model in self.models:
            expected_lr = math.exp(
                8.39
                - 0.81 * math.log(model["active_nonembedding_parameters"])
                - 0.25 * math.log(model["experts"])
            )
            self.assertTrue(
                math.isclose(model["peak_lr"], expected_lr, rel_tol=tolerance)
            )
            topology = self.profile.topology(model)
            self.assertEqual(
                topology["nodes"] * topology["gpus_per_node"],
                model["expert_parallel_size"],
            )

        calibration = [model for model in self.models if model["timing_calibration"]]
        self.assertEqual(len(calibration), 8)
        self.assertTrue(
            all(
                model["ladder"] == "D"
                and model["seed"] == 1234
                and model["languages"] in {1, 12}
                for model in calibration
            )
        )
        reconciled = [
            model
            for model in self.models
            if model["training_model_override_replacements"]
        ]
        self.assertEqual(len(reconciled), 16)
        self.assertTrue(all(model["ladder"] == "C" for model in reconciled))
        self.assertTrue(
            all(
                model["training_model_override_replacements"]
                == {"cuda_graph_scope": None}
                for model in reconciled
            )
        )

    def test_runtime_estimate_weights_all_82_models_without_queue_claims(self) -> None:
        from multilingual_megatron.experiments.scaling.stage1_benchmarking.runtime_estimate import (
            RUNTIME_RECEIPT_SCHEMA,
            build_runtime_estimate,
        )

        measured_seconds: dict[tuple[int, str], float] = {}
        receipts = {}
        for model in self.models:
            if not model["timing_calibration"]:
                continue
            key = (model["experts"], model["request_set"])
            request_cost = 50 if model["request_set"] == "multilingual" else 0
            seconds = float(100 + model["experts"] + request_cost)
            measured_seconds[key] = seconds
            gpu_count = model["expert_parallel_size"]
            receipts[model["id"]] = {
                "schema": RUNTIME_RECEIPT_SCHEMA,
                "status": "complete",
                "campaign_id": self.profile.payload["campaign_id"],
                "model_id": model["id"],
                "profile": {"sha256": self.profile.source_sha256},
                "timing": {
                    "job_wall_seconds": seconds,
                    "gpu_count": gpu_count,
                    "gpu_hours": seconds * gpu_count / 3600.0,
                },
            }

        estimate = build_runtime_estimate(
            self.models,
            receipts,
            campaign_id=self.profile.payload["campaign_id"],
            profile_sha256=self.profile.source_sha256,
            policy=self.profile.payload["runtime_estimation"]["policy"],
            safety_factor=self.profile.payload["runtime_estimation"]["safety_factor"],
        )
        expected_seconds = sum(
            measured_seconds[(model["experts"], model["request_set"])]
            for model in self.models
        )
        expected_gpu_hours = sum(
            measured_seconds[(model["experts"], model["request_set"])]
            * model["expert_parallel_size"]
            / 3600.0
            for model in self.models
        )
        self.assertEqual(
            estimate["counts"],
            {"campaign_models": 82, "calibration_models": 8, "remaining_models": 74},
        )
        self.assertAlmostEqual(
            estimate["point_estimate"]["serial_job_hours"],
            expected_seconds / 3600.0,
        )
        self.assertAlmostEqual(
            estimate["point_estimate"]["gpu_hours"], expected_gpu_hours
        )
        self.assertFalse(estimate["method"]["queue_time_included"])

    def test_request_and_a40_slurm_contracts_are_consistent(self) -> None:
        from multilingual_megatron.experiments.scaling.stage1_benchmarking.jointmoe import (
            _pilot_profile,
        )

        payload = self.profile.payload
        english = payload["request_sets"]["english"]
        multilingual = payload["request_sets"]["multilingual"]
        self.assertEqual(
            multilingual["sampling"],
            {"seed": 20260828, "examples_per_language": 100, "shared_population": 500},
        )
        self.assertTrue(
            all(item["languages_expected"] == 1 for item in english["benchmarks"])
        )
        self.assertTrue(
            all(
                model["request_set"]
                == ("english" if model["languages"] == 1 else "multilingual")
                for model in self.models
            )
        )
        for request_set in ("english", "multilingual"):
            model = next(
                item for item in self.models if item["request_set"] == request_set
            )
            pilot = _pilot_profile(self.profile, request_set, [model])
            expected_id = payload["request_sets"][request_set]["pilot_id"]
            self.assertEqual(pilot.benchmark_id, expected_id)
            self.assertEqual(pilot.request_manifest_pilot_id, expected_id)

        slurm = json.loads(SLURM_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(slurm["job_classes"]), {"1", "2", "4"})
        for world_size, shape in slurm["job_classes"].items():
            profile_shape = payload["evaluation_topology_by_world_size"][world_size]
            self.assertEqual(
                (shape["nodes"], shape["gpus_per_node"]),
                (profile_shape["nodes"], profile_shape["gpus_per_node"]),
            )
            self.assertEqual(
                shape["nodes"] * shape["gpus_per_node"], int(world_size)
            )
        self.assertEqual(payload["budget"], {"hardware": "A40", "max_gpu_hours": 408.0})

        runner = (
            REPO_ROOT / "scripts/slurm/run_stage1_jointmoe_benchmark_job.sh"
        ).read_text(encoding="utf-8")
        for runtime_module in (
            "stage1_benchmarking/replay.py",
            "run_study_bridge_recipe.py",
            "native_megatron_runner.py",
            "native_megatron_scoring.py",
        ):
            self.assertIn(runtime_module, runner)

    def test_native_scoring_masks_context_and_left_truncates(self) -> None:
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import (
            build_score_batch,
            prepare_scored_sequence,
        )

        sequence = prepare_scored_sequence(
            request_index=0,
            request_id="request-0",
            context_ids=[10, 11, 12, 13],
            continuation_ids=[20, 21],
            max_length=4,
        )
        self.assertEqual(sequence.input_ids, (11, 12, 13, 20))
        self.assertEqual(sequence.labels, (12, 13, 20, 21))
        batch = build_score_batch([sequence], pad_token_id=3)
        self.assertEqual(batch["loss_mask"].tolist(), [[0.0, 0.0, 1.0, 1.0]])
        self.assertEqual(batch["input_valid_mask"].tolist(), [[True, True, True, True]])
        self.assertEqual(batch["sample_id"].tolist(), [0])

    def test_native_scoring_sets_a_topology_valid_evaluation_global_batch(self) -> None:
        from types import SimpleNamespace

        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import (
            configure_native_benchmark_scoring,
        )

        model = SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            gradient_accumulation_fusion=True,
        )
        config = SimpleNamespace(
            model=model,
            checkpoint=SimpleNamespace(),
            validation=SimpleNamespace(eval_global_batch_size=32),
            train=SimpleNamespace(global_batch_size=32),
            tokenizer=SimpleNamespace(),
        )
        spec = SimpleNamespace(
            model_contract={
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
            expected_world_size=4,
            micro_batch_size=64,
            checkpoint_path=Path("/checkpoint"),
            checkpoint_step=1,
            tokenizer_type="SentencePieceTokenizer",
            tokenizer_model_path=Path("/tokenizer.model"),
            tokenizer_vocab_size=64_000,
            moe_reduction_dtype="model",
        )

        configure_native_benchmark_scoring(config, spec)

        self.assertEqual(config.validation.eval_micro_batch_size, 64)
        self.assertEqual(config.validation.eval_global_batch_size, 256)
        self.assertEqual(config.train.micro_batch_size, 64)
        self.assertEqual(config.train.global_batch_size, 256)

    def test_native_runner_binds_the_early_replica_gate_contract(self) -> None:
        from multilingual_megatron.pipeline.stages.benchmarking import (
            native_megatron_lm_eval,
            native_megatron_runner,
        )

        self.assertEqual(
            native_megatron_runner.EARLY_REPLICA_REQUESTS,
            native_megatron_lm_eval.EARLY_REPLICA_REQUESTS,
        )
        self.assertEqual(
            native_megatron_runner.EARLY_REPLICA_SCORE_ABS_TOLERANCE,
            native_megatron_lm_eval.EARLY_REPLICA_SCORE_ABS_TOLERANCE,
        )

    def test_split_benchmark_modules_bind_runtime_dependencies(self) -> None:
        from multilingual_megatron.experiments.scaling.stage1_benchmarking import (
            contracts,
            replay,
        )
        from multilingual_megatron.pipeline.stages.benchmarking import (
            native_megatron_runner,
        )

        payload = {"status": "complete"}
        self.assertEqual(
            replay._payload_sha256(payload), contracts._payload_sha256(payload)
        )
        for module in (replay, native_megatron_runner):
            self.assertTrue(all(hasattr(module, name) for name in module.__all__))


if __name__ == "__main__":
    unittest.main()
