from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Stage1BatchPolicyTest(unittest.TestCase):
    def test_token_ramp_materializes_segment_steps_and_cumulative_tokens(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            cumulative_step_for_tokens,
            materialize_batch_segments,
            resolve_batch_policy,
            tokens_for_cumulative_step,
        )

        policy = resolve_batch_policy(
            {
                "batch_policy": {
                    "enabled": True,
                    "target_global_batch_size": 128,
                    "schedule": [
                        {"start_tokens": 0, "end_tokens": 512, "global_batch_size": 32},
                        {"start_tokens": 512, "end_tokens": 1024, "global_batch_size": 64},
                        {"start_tokens": 1024, "end_tokens": 0, "global_batch_size": 128},
                    ],
                }
            },
            seq_length=16,
        )

        self.assertIsNotNone(policy)
        segments = materialize_batch_segments(
            policy=policy,
            total_tokens=1600,
            plateau_micro_batch_size=64,
        )

        self.assertEqual([segment["global_batch_size"] for segment in segments], [32, 64, 128])
        self.assertEqual([segment["micro_batch_size"] for segment in segments], [32, 64, 64])
        self.assertEqual([segment["cumulative_end_step"] for segment in segments], [1, 2, 3])
        self.assertEqual(cumulative_step_for_tokens(segments, 900), 2)
        self.assertEqual(tokens_for_cumulative_step(segments, 2), 1536)

    def test_target_gbs_must_match_final_schedule_gbs(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import resolve_batch_policy

        with self.assertRaises(SystemExit):
            resolve_batch_policy(
                {
                    "batch_policy": {
                        "enabled": True,
                        "target_global_batch_size": 64,
                        "schedule": [
                            {"start_tokens": 0, "end_tokens": 100, "global_batch_size": 32},
                            {"start_tokens": 100, "end_tokens": 0, "global_batch_size": 128},
                        ],
                    }
                },
                seq_length=16,
            )

    def test_joint_moe_v2_materializes_trunk_branches_and_fixed_dataset_horizon(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            JOINT_MOE_FULL_STATE_WSD_V2,
            build_joint_moe_full_state_wsd_v2_plan,
            joint_moe_full_state_wsd_v2_scheduler_accounting,
            resolve_batch_policy,
        )

        policy = resolve_batch_policy(
            {
                "batch_policy": {
                    "enabled": True,
                    "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                    "dataset_horizon_tokens": 8_000_000_000,
                    "checkpoint_horizons_tokens": [
                        500_000_000,
                        1_000_000_000,
                        2_000_000_000,
                        4_000_000_000,
                        8_000_000_000,
                    ],
                }
            },
            seq_length=2048,
        )

        self.assertIsNotNone(policy)
        endpoint_plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=policy,
            plateau_micro_batch_size=64,
        )

        self.assertEqual(
            endpoint_plan["stable_checkpoint_tokens"],
            [400_000_000, 800_000_000, 1_600_000_000, 3_200_000_000, 6_400_000_000],
        )
        self.assertEqual(
            [segment["target_end_tokens"] for segment in endpoint_plan["trunk"]["segments"]],
            [400_000_000, 500_000_000, 800_000_000, 1_000_000_000, 1_600_000_000, 3_200_000_000, 6_400_000_000],
        )
        self.assertEqual(
            [segment["global_batch_size"] for segment in endpoint_plan["trunk"]["segments"]],
            [32, 32, 64, 64, 128, 128, 128],
        )
        self.assertEqual(
            [(branch["start_tokens"], branch["target_tokens"]) for branch in endpoint_plan["branches"]],
            [
                (400_000_000, 500_000_000),
                (800_000_000, 1_000_000_000),
                (1_600_000_000, 2_000_000_000),
                (3_200_000_000, 4_000_000_000),
                (6_400_000_000, 8_000_000_000),
            ],
        )
        self.assertEqual(endpoint_plan["fixed_dataset_final_cumulative_step"], 38_149)
        self.assertEqual(endpoint_plan["fixed_dataset_sample_horizon"], 38_149 * 128)

        scheduler = joint_moe_full_state_wsd_v2_scheduler_accounting(
            policy=policy,
            endpoint_plan=endpoint_plan,
        )
        self.assertEqual(scheduler["warmup_samples"], 63_488)
        self.assertEqual(scheduler["decay_start_samples"], 3_125_120)
        self.assertEqual(scheduler["lr_decay_samples"], 3_906_432)
        self.assertEqual(scheduler["wsd_decay_samples"], 781_312)

    def test_joint_moe_v2_rejects_any_noncanonical_batch_ramp(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            JOINT_MOE_FULL_STATE_WSD_V2,
            resolve_batch_policy,
        )

        with self.assertRaises(SystemExit):
            resolve_batch_policy(
                {
                    "batch_policy": {
                        "enabled": True,
                        "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                        "dataset_horizon_tokens": 8_000_000_000,
                        "checkpoint_horizons_tokens": [500_000_000, 8_000_000_000],
                        "schedule": [
                            {"start_tokens": 0, "end_tokens": 500_000_000, "global_batch_size": 16},
                            {"start_tokens": 500_000_000, "end_tokens": 1_000_000_000, "global_batch_size": 64},
                            {"start_tokens": 1_000_000_000, "end_tokens": 0, "global_batch_size": 128},
                        ],
                    }
                },
                seq_length=2048,
            )

    def test_joint_moe_v2_b_horizon_includes_12_8b_trunk_and_16b_branch(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            JOINT_MOE_FULL_STATE_WSD_V2,
            build_joint_moe_full_state_wsd_v2_plan,
            joint_moe_full_state_wsd_v2_scheduler_accounting,
            resolve_batch_policy,
        )

        policy = resolve_batch_policy(
            {
                "batch_policy": {
                    "enabled": True,
                    "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                    "dataset_horizon_tokens": 16_000_000_000,
                    "checkpoint_horizons_tokens": [
                        500_000_000,
                        1_000_000_000,
                        2_000_000_000,
                        4_000_000_000,
                        8_000_000_000,
                        16_000_000_000,
                    ],
                }
            },
            seq_length=2048,
        )
        endpoint_plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=policy,
            plateau_micro_batch_size=32,
        )
        scheduler = joint_moe_full_state_wsd_v2_scheduler_accounting(
            policy=policy,
            endpoint_plan=endpoint_plan,
        )

        self.assertEqual(endpoint_plan["stable_checkpoint_tokens"][-1], 12_800_000_000)
        self.assertEqual(endpoint_plan["branches"][-1]["target_tokens"], 16_000_000_000)
        self.assertEqual(endpoint_plan["fixed_dataset_final_cumulative_step"], 68_667)
        self.assertEqual(endpoint_plan["fixed_dataset_sample_horizon"], 8_789_376)
        self.assertEqual(scheduler["decay_start_samples"], 6_250_112)
        self.assertEqual(scheduler["lr_decay_samples"], 7_812_736)
        self.assertEqual(scheduler["wsd_decay_samples"], 1_562_624)

    def test_joint_moe_v2_materializes_readouts_without_extra_cooldown_branches(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            JOINT_MOE_FULL_STATE_WSD_V2,
            build_joint_moe_full_state_wsd_v2_plan,
            resolve_batch_policy,
        )

        final_target = 79_999_795_200
        readouts = [
            500_000_000,
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
            8_000_000_000,
            16_000_000_000,
            32_000_000_000,
            64_000_000_000,
        ]
        policy = resolve_batch_policy(
            {
                "batch_policy": {
                    "enabled": True,
                    "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                    "dataset_horizon_tokens": final_target,
                    "checkpoint_horizons_tokens": [final_target],
                    "readout_horizons_tokens": readouts,
                }
            },
            seq_length=2048,
        )
        endpoint_plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=policy,
            plateau_micro_batch_size=4,
            data_parallel_size=2,
        )

        self.assertEqual(len(endpoint_plan["branches"]), 1)
        self.assertEqual(endpoint_plan["readout_horizons_tokens"], readouts)
        self.assertEqual(
            [row["actual_tokens"] for row in endpoint_plan["readout_endpoints"]],
            [
                500_039_680,
                1_000_079_360,
                2_000_158_720,
                4_000_317_440,
                8_000_110_592,
                16_000_221_184,
                32_000_180_224,
                64_000_360_448,
            ],
        )
        self.assertEqual(endpoint_plan["readout_endpoints"][-1]["plan_role"], "branch")
        self.assertEqual(endpoint_plan["branches"][0]["actual_end_tokens"], 80_000_057_344)

    def test_checkpoint_splits_do_not_change_the_final_endpoint(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            JOINT_MOE_FULL_STATE_WSD_V2,
            build_joint_moe_full_state_wsd_v2_plan,
            resolve_batch_policy,
        )

        common = {
            "enabled": True,
            "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
            "dataset_horizon_tokens": 8_000_000_000,
            "checkpoint_horizons_tokens": [8_000_000_000],
        }
        unsplit = resolve_batch_policy({"batch_policy": common}, seq_length=2048)
        split = resolve_batch_policy(
            {
                "batch_policy": {
                    **common,
                    "readout_horizons_tokens": [2_000_000_000, 4_000_000_000, 6_000_000_000],
                }
            },
            seq_length=2048,
        )
        self.assertIsNotNone(unsplit)
        self.assertIsNotNone(split)
        unsplit_plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=unsplit,
            plateau_micro_batch_size=8,
            data_parallel_size=2,
        )
        split_plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=split,
            plateau_micro_batch_size=8,
            data_parallel_size=2,
        )
        self.assertEqual(
            split_plan["branches"][-1]["actual_end_tokens"],
            unsplit_plan["branches"][-1]["actual_end_tokens"],
        )
        self.assertEqual(
            split_plan["branches"][-1]["expected_end_step"],
            unsplit_plan["branches"][-1]["expected_end_step"],
        )

    def test_joint_moe_v2_rejects_readout_at_or_beyond_final_horizon(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            JOINT_MOE_FULL_STATE_WSD_V2,
            resolve_batch_policy,
        )

        with self.assertRaisesRegex(SystemExit, "strictly below"):
            resolve_batch_policy(
                {
                    "batch_policy": {
                        "enabled": True,
                        "planning_method": JOINT_MOE_FULL_STATE_WSD_V2,
                        "dataset_horizon_tokens": 8_000_000_000,
                        "checkpoint_horizons_tokens": [8_000_000_000],
                        "readout_horizons_tokens": [8_000_000_000],
                    }
                },
                seq_length=2048,
            )

    def test_segment_runner_writes_parent_training_manifest(self) -> None:
        from multilingual_megatron.pipeline.stages.training.segment_runner import _write_parent_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "run"
            seg0 = output_dir / "segments" / "00_gbs32"
            seg1 = output_dir / "segments" / "01_gbs128"
            (seg1 / "checkpoints" / "iter_0000003").mkdir(parents=True)
            (seg1 / "train_results.json").write_text(json.dumps({"train_loss": 3.0}), encoding="utf-8")
            (seg1 / "trainer_state.json").write_text(json.dumps({"global_step": 3}), encoding="utf-8")
            plan = {
                "run_id": "run_a",
                "run_name": "run-a",
                "target_token_budget_tokens": 4096,
                "expected_tokens": 5120,
                "plateau_micro_batch_size": 64,
                "policy": {
                    "mode": "token_ramp",
                    "target_global_batch_size": 128,
                    "mbs_policy": "cap_per_segment",
                    "seq_length": 16,
                    "schedule": [
                        {"start_tokens": 0, "end_tokens": 1024, "global_batch_size": 32},
                        {"start_tokens": 1024, "end_tokens": 0, "global_batch_size": 128},
                    ],
                },
                "segments": [
                    {
                        "segment_index": 0,
                        "target_start_tokens": 0,
                        "target_end_tokens": 1024,
                        "actual_start_tokens": 0,
                        "actual_end_tokens": 1024,
                        "global_batch_size": 32,
                        "micro_batch_size": 32,
                        "tokens_per_step": 512,
                        "segment_train_iters": 2,
                        "cumulative_start_step": 0,
                        "cumulative_end_step": 2,
                        "output_dir": str(seg0),
                        "log_path": str(seg0.with_suffix(".train.log")),
                    },
                    {
                        "segment_index": 1,
                        "target_start_tokens": 1024,
                        "target_end_tokens": 4096,
                        "actual_start_tokens": 1024,
                        "actual_end_tokens": 5120,
                        "global_batch_size": 128,
                        "micro_batch_size": 64,
                        "tokens_per_step": 2048,
                        "segment_train_iters": 1,
                        "cumulative_start_step": 2,
                        "cumulative_end_step": 3,
                        "output_dir": str(seg1),
                        "log_path": str(seg1.with_suffix(".train.log")),
                    },
                ],
            }
            results = [
                {"output_dir": str(seg0), "checkpoint_root": str(seg0 / "checkpoints")},
                {"output_dir": str(seg1), "checkpoint_root": str(seg1 / "checkpoints")},
            ]

            _write_parent_artifacts(output_dir=output_dir, plan=plan, results=results, runtime=1.0)
            manifest = json.loads((output_dir / "training_run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["metadata"]["segmented_training"])
        self.assertEqual(manifest["metadata"]["global_batch_size"], 128)
        self.assertEqual(manifest["metadata"]["segments"][1]["cumulative_end_step"], 3)
        self.assertEqual(manifest["metadata"]["final_checkpoint_step"], 3)


if __name__ == "__main__":
    unittest.main()
