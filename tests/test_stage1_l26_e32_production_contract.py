from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CONTRACT_PATH = REPO_ROOT / "configs/stage1/production/stage1_l26_e32_k200_law_production_v1.json"
PROFILE_PATH = REPO_ROOT / "configs/pipeline/profiles/stage1_l26_e32_k200_production.json"
ARCHITECTURE_PATH = REPO_ROOT / "configs/stage1/architectures/stage1_l26_e32_v1.json"
SYSTEMS_PATH = REPO_ROOT / "configs/stage1/l26_e32/winner_pp4_ep2_mbs8_v1.json"
JOINTMOE_LR_CONTRACT_PATH = (
    REPO_ROOT
    / "configs/stage1/production/stage1_l26_e32_k200_jointmoe_lr_production_v1.json"
)
JOINTMOE_LR_COOLDOWN_LAUNCH_PATH = (
    REPO_ROOT
    / "configs/stage1/production/stage1_l26_e32_k200_jointmoe_lr_cooldown_launch_v1.json"
)
JOINTMOE_LR_COOLDOWN_LAUNCHER_PATH = (
    REPO_ROOT / "scripts/slurm/submit_stage1_l26_e32_jointmoe_lr_cooldown.sh"
)
JOINTMOE_LR_PROFILE_NAMES = (
    "stage1_l26_e32_k200_jointmoe_lr_smoke_v1",
    "stage1_l26_e32_k200_jointmoe_lr_confirm_r1_v1",
    "stage1_l26_e32_k200_jointmoe_lr_confirm_r2_v1",
    "stage1_l26_e32_k200_jointmoe_lr_ramp_gbs32_v1",
    "stage1_l26_e32_k200_jointmoe_lr_ramp_gbs64_v1",
    "stage1_l26_e32_k200_jointmoe_lr_checkpoint_resume_v1",
    "stage1_l26_e32_k200_jointmoe_lr_trunk_production_v1",
    "stage1_l26_e32_k200_jointmoe_lr_cooldown_production_v1",
)


class Stage1L26E32ProductionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        cls.profile = cls.profiles["stage1_l26_e32_k200_law_trunk_production_v1"]
        cls.cooldown_profile = cls.profiles["stage1_l26_e32_k200_law_cooldown_production_v1"]
        cls.architecture = json.loads(ARCHITECTURE_PATH.read_text(encoding="utf-8"))
        cls.systems = json.loads(SYSTEMS_PATH.read_text(encoding="utf-8"))

    def test_frozen_model_and_systems_authorities_are_preserved(self) -> None:
        model = self.contract["model"]
        architecture = self.architecture["model"]
        self.assertEqual(self.contract["status"], "materialized_prelaunch_pending_calibration_and_confirmation")
        for field in ("layers", "hidden_size", "attention_heads", "ffn_hidden_size", "experts", "top_k"):
            self.assertEqual(model[field], architecture[field])
        self.assertEqual(model["active_parameters"], architecture["active_parameters"])
        self.assertEqual(model["total_parameters"], architecture["total_parameters"])
        self.assertEqual(model["micro_batch_size"], self.systems["systems_configuration"]["micro_batch_size"])
        self.assertEqual(model["global_batch_size"], self.systems["scientific_contract"]["global_batch_size"])

        overrides = set(self.profile["training"]["overrides"])
        for key, value in self.systems["overrides"].items():
            if key == "train.micro_batch_size":
                self.assertEqual(self.profile["training"]["train"]["micro_batch_size"], int(value))
                continue
            self.assertIn(f"{key}={value}", overrides)

    def test_corrected_lr_verification_profiles_are_short_fresh_runs(self) -> None:
        from multilingual_megatron.pipeline.config_loader import (
            load_pipeline_config_bundle,
        )
        from multilingual_megatron.pipeline.runtime import _resolve_profile
        from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (
            learning_rate,
        )

        model = self.architecture["model"]
        active_nonembedding = model["active_parameters"] - (
            2 * model["vocab_size"] * model["hidden_size"]
        )
        expected_lr = learning_rate(active_nonembedding, model["experts"])
        self.assertAlmostEqual(expected_lr, 0.00010017337354820424, places=16)

        config = load_pipeline_config_bundle(REPO_ROOT / "configs/pipeline")
        expected_profiles = {
            "stage1_l26_e32_k200_lr_smoke_v1": 2_097_152,
            "stage1_l26_e32_k200_lr_confirm_r1_v1": 134_217_728,
            "stage1_l26_e32_k200_lr_confirm_r2_v1": 134_217_728,
        }
        output_roots = set()
        for name, token_budget in expected_profiles.items():
            profile = _resolve_profile(config, name)
            output_roots.add(profile["output_root"])
            self.assertNotIn("m2", name)
            self.assertNotIn("m2", profile["output_root"])
            self.assertFalse(profile["training"]["train"]["batch_policy"]["enabled"])
            self.assertEqual(
                profile["training"]["train"]["token_budget_tokens"], token_budget
            )
            self.assertEqual(
                profile["runtime"]["env"]["BRIDGE_MOE_TOKEN_DROP_METRICS"], "1"
            )
            self.assertNotIn(
                "--reservation=study_phase2",
                profile["executor"]["extra_sbatch_flags"],
            )
            run = profile["training"]["runs"][0]
            self.assertEqual(run["id"], "stage1_l26_e32_k200_lr_verification")
            self.assertIn(f"optimizer.lr={expected_lr}", run["overrides"])
        self.assertEqual(len(output_roots), len(expected_profiles))

    def test_k200_manifest_is_reused_without_retokenization_or_resampling(self) -> None:
        data = self.contract["data"]
        manifest_path = REPO_ROOT / data["manifest"]
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(), data["manifest_file_sha256"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["tier"], "K200")
        self.assertEqual(len(manifest["languages"]), data["languages"])
        self.assertEqual(len(manifest["prefixes"]), data["languages"])
        self.assertEqual(len(manifest["weights"]), data["languages"])
        self.assertEqual(manifest["manifest_payload_sha256"], data["manifest_payload_sha256"])
        self.assertEqual(manifest["tokenizer_model_sha256"], data["tokenizer_model_sha256"])
        self.assertEqual(
            sum(int(value) for value in manifest["tokens_per_language"].values()),
            data["physical_tokens_including_eod"],
        )
        self.assertFalse(self.profile["sample"]["enabled"])
        self.assertFalse(self.profile["tokenization"]["enabled"])

    def test_checkpoint_plan_has_exact_quarters_and_one_final_cooldown_branch(self) -> None:
        from multilingual_megatron.pipeline.stages.training.batch_policy import (
            build_joint_moe_full_state_wsd_v2_plan,
            resolve_batch_policy,
        )

        train = self.profile["training"]["train"]
        policy = resolve_batch_policy(train, seq_length=2048)
        self.assertIsNotNone(policy)
        plan = build_joint_moe_full_state_wsd_v2_plan(
            policy=policy,
            plateau_micro_batch_size=8,
            data_parallel_size=2,
        )
        checkpoint_contract = self.contract["checkpoint_contract"]
        expected = checkpoint_contract["scientific_checkpoints"]
        self.assertEqual(plan["checkpoint_horizons_tokens"], checkpoint_contract["checkpoint_horizons_tokens"])
        self.assertEqual(plan["readout_horizons_tokens"], checkpoint_contract["readout_horizons_tokens"])
        self.assertEqual(len(plan["branches"]), 1)
        physical = checkpoint_contract["physical_500_h100_hour_checkpoint"]
        physical_target = physical["profile_readout_horizon_tokens"]
        scientific_readouts = [
            row for row in plan["readout_endpoints"] if row["target_tokens"] != physical_target
        ]
        self.assertEqual(
            [row["actual_tokens"] for row in scientific_readouts],
            [row["executable_consumed_tokens"] for row in expected[:-1]],
        )
        physical_readout = next(
            row for row in plan["readout_endpoints"] if row["target_tokens"] == physical_target
        )
        self.assertEqual(physical_readout["actual_tokens"], physical["executable_consumed_tokens"])
        self.assertEqual(physical_readout["expected_step"], physical["expected_step"])
        final = plan["branches"][-1]
        self.assertEqual(final["actual_end_tokens"], expected[-1]["executable_consumed_tokens"])
        self.assertEqual(final["expected_end_step"], expected[-1]["expected_step"])
        self.assertEqual(final["actual_end_tokens"] % (4 * 128 * 2048), 0)
        self.assertEqual(
            [row["executable_consumed_tokens"] for row in expected[:-1]],
            [final["actual_end_tokens"] // 4, final["actual_end_tokens"] // 2, 3 * final["actual_end_tokens"] // 4],
        )
        self.assertEqual(self.profile["training"]["train"]["token_budget_tokens"], plan["trunk"]["target_end_tokens"])
        cooldown_train = self.cooldown_profile["training"]["train"]
        self.assertEqual(cooldown_train["token_budget_tokens"], final["target_tokens"])
        self.assertEqual(cooldown_train["batch_policy"]["plan_role"], "branch")
        self.assertEqual(cooldown_train["batch_policy"]["branch_target_tokens"], final["target_tokens"])

    def test_exact_500_hour_checkpoint_is_an_explicit_prelaunch_gate(self) -> None:
        physical = self.contract["checkpoint_contract"]["physical_500_h100_hour_checkpoint"]
        self.assertTrue(physical["required"])
        self.assertEqual(physical["status"], "selected_from_complete_calibration_receipt")
        self.assertEqual(physical["profile_readout_horizon_tokens"], 45_881_412_097)
        self.assertEqual(physical["executable_consumed_tokens"], 45_881_753_600)
        self.assertEqual(physical["expected_step"], 182_655)
        self.assertEqual(
            self.contract["cost_forecast"]["allocation_request_walltime"],
            {"trunk": "96:00:00", "cooldown": "48:00:00"},
        )

    def test_submission_profiles_omit_historical_reservations_and_nodes(self) -> None:
        self.assertEqual(set(self.profiles), {
            "stage1_l26_e32_k200_law_trunk_production_v1",
            "stage1_l26_e32_k200_law_cooldown_production_v1",
        })
        for profile in self.profiles.values():
            flags = profile.get("executor", {}).get("extra_sbatch_flags", [])
            self.assertFalse(any(flag.startswith(("--reservation=", "--nodelist="))
                                 for flag in flags))


class Stage1L26E32JointMoELRResubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from multilingual_megatron.pipeline.config_loader import (
            load_pipeline_config_bundle,
        )
        from multilingual_megatron.pipeline.runtime import _resolve_profile

        cls.contract = json.loads(
            JOINTMOE_LR_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        cls.cooldown_launch = json.loads(
            JOINTMOE_LR_COOLDOWN_LAUNCH_PATH.read_text(encoding="utf-8")
        )
        cls.bundle = load_pipeline_config_bundle(REPO_ROOT / "configs/pipeline")
        cls.profiles = {
            name: _resolve_profile(cls.bundle, name)
            for name in JOINTMOE_LR_PROFILE_NAMES
        }
        cls.historical = _resolve_profile(
            cls.bundle,
            "stage1_l26_e32_k200_law_trunk_production_v1",
        )

    @staticmethod
    def _effective_overrides(profile: dict) -> dict[str, str]:
        values: dict[str, str] = {}
        run = profile["training"]["runs"][0]
        for source in (profile["training"]["overrides"], run.get("overrides", [])):
            for item in source:
                key, value = str(item).split("=", 1)
                values[key] = value
        return values

    def test_resubmission_changes_only_lr_identity_and_physical_readout(self) -> None:
        name = "stage1_l26_e32_k200_jointmoe_lr_trunk_production_v1"
        profile = self.profiles[name]
        self.assertEqual(
            profile["training"]["overrides"],
            self.historical["training"]["overrides"],
        )

        historical_train = deepcopy(self.historical["training"]["train"])
        new_train = deepcopy(profile["training"]["train"])
        historical_train["batch_policy"].pop("readout_horizons_tokens")
        new_train["batch_policy"].pop("readout_horizons_tokens")
        self.assertEqual(new_train, historical_train)

        historical_overrides = self._effective_overrides(self.historical)
        new_overrides = self._effective_overrides(profile)
        differences = {
            key
            for key in historical_overrides | new_overrides
            if historical_overrides.get(key) != new_overrides.get(key)
        }
        self.assertEqual(
            differences,
            {
                "optimizer.lr",
                "logger.wandb_project",
                "logger.wandb_exp_name",
                "logger.wandb_save_dir",
            },
        )
        self.assertEqual(new_overrides["optimizer.lr"], "0.00010017337354820424")
        self.assertEqual(new_overrides["logger.wandb_project"], "stage1_scaling_laws")
        self.assertEqual(new_overrides["model.moe_router_fusion"], "true")
        self.assertEqual(new_overrides["rng.seed"], "1234")
        self.assertEqual(new_overrides["model.moe_token_dispatcher_type"], "alltoall")

        old_run = self.historical["training"]["runs"][0]
        new_run = profile["training"]["runs"][0]
        for key in (
            "kind",
            "backend",
            "recipe",
            "data_manifest",
            "data_manifest_sha256",
        ):
            self.assertEqual(new_run[key], old_run[key])
        self.assertNotEqual(new_run["id"], old_run["id"])
        self.assertNotEqual(profile["output_root"], self.historical["output_root"])
        self.assertFalse(profile["sample"]["enabled"])
        self.assertFalse(profile["tokenization"]["enabled"])
        flags = profile["executor"]["extra_sbatch_flags"]
        self.assertFalse(any(flag.startswith("--reservation=") for flag in flags))
        self.assertFalse(any(flag.startswith("--nodelist=") for flag in flags))

    def test_all_preflights_are_fresh_matched_real_data_runs(self) -> None:
        expected = {
            "stage1_l26_e32_k200_jointmoe_lr_smoke_v1": (128, 2_097_152, False),
            "stage1_l26_e32_k200_jointmoe_lr_confirm_r1_v1": (128, 134_217_728, False),
            "stage1_l26_e32_k200_jointmoe_lr_confirm_r2_v1": (128, 134_217_728, False),
            "stage1_l26_e32_k200_jointmoe_lr_ramp_gbs32_v1": (32, 33_554_432, False),
            "stage1_l26_e32_k200_jointmoe_lr_ramp_gbs64_v1": (64, 67_108_864, False),
            "stage1_l26_e32_k200_jointmoe_lr_checkpoint_resume_v1": (128, 134_217_728, True),
        }
        output_roots: set[str] = set()
        run_ids: set[str] = set()
        for name, (gbs, tokens, segmented) in expected.items():
            profile = self.profiles[name]
            train = profile["training"]["train"]
            run = profile["training"]["runs"][0]
            output_roots.add(profile["output_root"])
            run_ids.add(run["id"])
            self.assertEqual(train["global_batch_size"], gbs)
            self.assertEqual(train["micro_batch_size"], 8)
            self.assertEqual(train["token_budget_tokens"], tokens)
            self.assertEqual(bool(train["batch_policy"]["enabled"]), segmented)
            self.assertEqual(
                profile["runtime"]["env"]["BRIDGE_MOE_TOKEN_DROP_METRICS"],
                "1",
            )
            expected_window = "32" if name.endswith("_smoke_v1") else "128"
            self.assertEqual(
                profile["runtime"]["env"]["BRIDGE_THROUGHPUT_WINDOW_SIZE"],
                expected_window,
            )
            self.assertEqual(
                self._effective_overrides(profile)["optimizer.lr"],
                "0.00010017337354820424",
            )
            self.assertEqual(profile["wandb"]["project"], "stage1_scaling_laws")
            self.assertEqual(
                self._effective_overrides(profile)["logger.wandb_project"],
                "stage1_scaling_laws",
            )
            self.assertEqual(
                run["data_manifest_sha256"],
                self.contract["data"]["manifest_file_sha256"],
            )
        self.assertEqual(len(output_roots), len(expected))
        self.assertEqual(len(run_ids), len(expected))

        resume = self.profiles[
            "stage1_l26_e32_k200_jointmoe_lr_checkpoint_resume_v1"
        ]
        policy = resume["training"]["train"]["batch_policy"]
        self.assertTrue(policy["require_full_state_checkpoint"])
        self.assertEqual(
            [row["global_batch_size"] for row in policy["schedule"]],
            [128, 128],
        )

    def test_selected_physical_checkpoint_preserves_the_final_horizon(self) -> None:
        from multilingual_megatron.experiments.scaling.stage1_l26_e32.production import build_plan_from_profile

        trunk = self.profiles[
            "stage1_l26_e32_k200_jointmoe_lr_trunk_production_v1"
        ]
        cooldown = self.profiles[
            "stage1_l26_e32_k200_jointmoe_lr_cooldown_production_v1"
        ]
        for profile in (trunk, cooldown):
            self.assertEqual(profile["wandb"]["project"], "stage1_scaling_laws")
            self.assertEqual(
                self._effective_overrides(profile)["logger.wandb_project"],
                "stage1_scaling_laws",
            )
        plan = build_plan_from_profile(trunk)
        self.assertEqual(
            plan["readout_horizons_tokens"],
            [
                17_170_614_784,
                34_341_308_928,
                43_579_525_633,
                51_512_003_072,
            ],
        )
        self.assertEqual(plan["trunk"]["target_end_tokens"], 54_946_011_544)
        self.assertEqual(plan["branches"][0]["actual_end_tokens"], 68_682_776_576)

        lr_schedule = self.contract["training_contract"]["learning_rate_schedule"]
        self.assertEqual(lr_schedule["warmup_executable_tokens"], 130_023_424)
        self.assertTrue(lr_schedule["warmup_completes_during_gbs32"])
        self.assertEqual(
            lr_schedule["stable_trunk_executable_tokens"],
            54_946_168_832,
        )
        self.assertEqual(
            lr_schedule["linear_decay_final_executable_tokens"],
            68_682_776_576,
        )
        self.assertEqual(lr_schedule["final_lr"], 0.0)
        self.assertEqual(
            [
                row["executable_end_tokens"]
                for row in self.contract["training_contract"]["batch_ramp"]
            ],
            [500_039_680, 1_000_079_360, 68_682_776_576],
        )

        final_segment = plan["trunk"]["segments"][-1]
        expected_parent = (
            "outputs/pipeline_runs/stage1_l26_e32_k200_jointmoe_lr_production_v1/"
            f"trunk/runs/stage1_l26_e32_k200_jointmoe_lr_trunk/segments/"
            f"{final_segment['segment_index']:02d}_gbs128/checkpoints"
        )
        self.assertEqual(
            cooldown["training"]["train"]["batch_policy"]["branch_parent_checkpoint_root"],
            expected_parent,
        )
        physical = self.contract["checkpoint_contract"][
            "physical_500_h100_hour_checkpoint"
        ]
        self.assertTrue(physical["required"])
        self.assertEqual(
            physical["status"],
            "selected_from_complete_calibration_receipt",
        )
        self.assertEqual(physical["profile_readout_horizon_tokens"], 43_579_525_633)
        self.assertEqual(physical["executable_consumed_tokens"], 43_579_867_136)
        self.assertEqual(physical["expected_step"], 173_874)
        self.assertLess(physical["completed_checkpoint_h100_gpu_hours"], 500.0)
        self.assertGreater(
            physical["next_update_completed_checkpoint_h100_gpu_hours"],
            500.0,
        )
        self.assertTrue(physical["historical_anchor_is_not_reused"])
        self.assertTrue(self.contract["release_policy"]["production_authorized"])
        self.assertIn(
            "source-seal",
            self.contract["release_policy"]["launch_condition"].lower(),
        )

    def test_cooldown_retry_is_terminal_parent_gated_and_source_locked(self) -> None:
        policy = self.contract["release_policy"]
        presubmit = policy["cooldown_pre_submission"]
        launch = self.cooldown_launch

        self.assertTrue(policy["trunk_must_precede_cooldown_execution"])
        self.assertTrue(policy["cooldown_queue_submission_may_precede_trunk_completion"])
        self.assertTrue(presubmit["authorized"])
        self.assertEqual(presubmit["required_dependency_type"], "afterok")
        self.assertFalse(presubmit["manual_intermediate_performance_gate_required"])
        self.assertEqual(
            launch["scheduler"]["authorized_presubmission_dependency"],
            "afterok:2758779",
        )
        self.assertIsNone(launch["scheduler"]["effective_dependency"])
        self.assertEqual(
            launch["scheduler"]["parent_gate"]["mode"],
            "terminal_parent_verified_before_submission",
        )
        self.assertTrue(
            launch["runtime_gates"][
                "cooldown_parent_full_state_reaudited_before_first_update"
            ]
        )
        self.assertEqual(launch["trunk"]["stable_parent_step"], 217_233)
        self.assertEqual(launch["trunk"]["stable_parent_consumed_samples"], 26_829_184)
        self.assertEqual(launch["cooldown"]["expected_final_step"], 269_634)
        self.assertEqual(
            launch["cooldown"]["attempt_id"],
            "20260907-jointmoe-lr-cooldown-retry2-verified-resume-v1",
        )
        self.assertEqual(launch["retry_history"]["superseded_job_id"], 2_793_367)
        self.assertTrue(launch["retry_history"]["parent_checkpoint_untouched"])
        self.assertFalse(launch["retry_history"]["excluded_scratch_restart"]["eligible_for_scientific_results"])
        self.assertTrue(launch["runtime_gates"]["engine_load_pinned_to_absolute_audited_iteration"])
        self.assertTrue(launch["runtime_gates"]["restored_step_samples_and_scheduler_verified_on_every_rank_before_first_update"])
        self.assertFalse(
            launch["runtime_gates"]["mutable_checkout_after_submission_is_executed"]
        )
        self.assertNotIn(
            "2758779",
            json.dumps(
                self.profiles[
                    "stage1_l26_e32_k200_jointmoe_lr_cooldown_production_v1"
                ]
            ),
        )

        for item in launch["source_lock"]["files"]:
            path = REPO_ROOT / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                item["sha256"],
                item["path"],
            )

        launcher = JOINTMOE_LR_COOLDOWN_LAUNCHER_PATH.read_text(encoding="utf-8")
        launch_sha = hashlib.sha256(
            JOINTMOE_LR_COOLDOWN_LAUNCH_PATH.read_bytes()
        ).hexdigest()
        self.assertIn(f'expected_launch_contract_sha256="{launch_sha}"', launcher)
        self.assertNotIn("--dependency=", launcher)
        self.assertIn("verify_completed_trunk", launcher)
        self.assertIn("--format=JobIDRaw,State,ExitCode", launcher)
        self.assertIn("--export=WANDB_API_KEY", launcher)
        self.assertNotIn("WANDB_API_KEY=", launcher)
        self.assertIn('if ! mkdir "${submission_guard}"', launcher)
        self.assertIn('scontrol" show job "${job_id}" -o', launcher)
        self.assertIn("logs/attempts/${attempt_id}/slurm", launcher)
        self.assertIn('git archive --format=tar "${source_commit}" -- src configs', launcher)
        self.assertIn('verify_submission_snapshot "${execution_root}"', launcher)
        self.assertIn('cd "${execution_root}"', launcher)
        self.assertIn("immutable_submission_commit_snapshot", launcher)
        self.assertIn('"${observed_nodes}" == "2-2"', launcher)

        with tempfile.TemporaryDirectory() as tmp:
            spooled = Path(tmp) / "slurm_script"
            spooled.write_text(launcher, encoding="utf-8")
            spooled.chmod(0o700)
            env = dict(os.environ)
            env.update(
                {
                    "SLURM_JOB_ID": "999999",
                    "SLURM_SUBMIT_DIR": str(REPO_ROOT),
                }
            )
            completed = subprocess.run(
                [str(spooled), "self-test-spooled-root"],
                cwd=tmp,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.stdout.strip(), str(REPO_ROOT))

    def test_lr_formula_sources_and_inference_scope_are_auditable(self) -> None:
        from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (
            learning_rate,
        )

        change = self.contract["controlled_change"]
        model = self.contract["model"]
        active_nonembedding = (
            model["active_parameters"] - 2 * 64_000 * model["hidden_size"]
        )
        self.assertEqual(active_nonembedding, change["N_act_nonembed"])
        self.assertAlmostEqual(
            learning_rate(active_nonembedding, model["experts"]),
            change["new_value"],
            places=16,
        )
        self.assertEqual(
            self.contract["training_contract"]["independent_experimental_units"],
            1,
        )
        self.assertFalse(
            self.contract["training_contract"]["nested_readouts_are_independent_replicates"]
        )
        self.assertEqual(
            self.contract["observability"]["wandb_project"],
            "stage1_scaling_laws",
        )
        self.assertFalse(
            self.contract["observability"]["offline_fallback_authorized"]
        )

        for key in (
            "architecture",
            "systems",
            "learning_rate_specification",
            "learning_rate_confirmation",
        ):
            authority = self.contract["authorities"][key]
            path = REPO_ROOT / authority["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), authority["sha256"])


if __name__ == "__main__":
    unittest.main()
