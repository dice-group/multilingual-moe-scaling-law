from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
HARNESS_ROOT = REPO_ROOT / "external/lm-evaluation-harness"
for path in (REPO_ROOT, SRC_ROOT, HARNESS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


PROFILE_PATH = (
    REPO_ROOT / "configs/benchmarks/stage1_abcd_jointmoe_all_readouts_a40_v2.json"
)


class Stage1JointMoEAllReadoutsBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from multilingual_megatron.experiments.scaling.stage1_benchmarking.campaign_profile import (
            load_campaign_profile,
        )

        cls.profile = load_campaign_profile(PROFILE_PATH)
        cls.models = list(cls.profile.payload["models"])

    def test_complete_canonical_and_seed_readout_cohorts_are_resolved(self) -> None:
        self.assertEqual(len(self.models), 508)
        self.assertEqual(
            Counter(model["source_cohort"] for model in self.models),
            {"canonical_release": 400, "b_seed_confirmation": 108},
        )
        self.assertEqual(
            Counter(model["ladder"] for model in self.models),
            {"A": 80, "B": 204, "C": 112, "D": 112},
        )
        self.assertEqual(sum(model["is_final_readout"] for model in self.models), 82)
        self.assertEqual(len({model["id"] for model in self.models}), 508)
        self.assertTrue(
            all(f"_d{model['nominal_tokens']}_" in model["id"] for model in self.models)
        )

    def test_each_trajectory_has_the_sealed_token_schedule(self) -> None:
        observed: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)
        for model in self.models:
            observed[
                (
                    str(model["ladder"]),
                    int(model["experts"]),
                    int(model["languages"]),
                    int(model["seed"]),
                )
            ].append(int(model["nominal_tokens"]))

        schedules = self.profile.payload["selection"]["canonical_release"][
            "nominal_tokens_by_ladder"
        ]
        replicated_schedule = self.profile.payload["selection"][
            "b_seed_replication"
        ]["nominal_tokens"]
        self.assertEqual(len(observed), 82)
        for (ladder, _experts, _languages, seed), nominal_tokens in observed.items():
            expected = schedules[ladder] if seed == 1234 else replicated_schedule
            self.assertEqual(nominal_tokens, expected)

    def test_only_terminal_d_readouts_form_the_runtime_calibration(self) -> None:
        calibration = [model for model in self.models if model["timing_calibration"]]
        self.assertEqual(len(calibration), 8)
        self.assertTrue(
            all(
                model["ladder"] == "D"
                and model["languages"] in {1, 12}
                and model["is_final_readout"]
                and model["nominal_tokens"] == 32_000_000_000
                for model in calibration
            )
        )

    def test_budget_and_analysis_metadata_cover_every_readout(self) -> None:
        cap = sum(
            float(self.profile.topology(model)["max_gpu_hours"])
            for model in self.models
        )
        self.assertEqual(cap, 2592.0)
        for model in self.models:
            self.assertGreaterEqual(model["tokens_seen"], model["nominal_tokens"])
            self.assertLessEqual(model["readout_index"], model["readout_count"])
            self.assertEqual(
                model["is_final_readout"],
                model["readout_index"] == model["readout_count"],
            )

    def test_one_command_submission_mode_is_available(self) -> None:
        submitter = (
            REPO_ROOT / "scripts/slurm/submit_stage1_jointmoe_benchmark.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("plan-all|submit-all", submitter)
        self.assertIn("nominal_tokens", submitter)
        self.assertIn("readout_id", submitter)


if __name__ == "__main__":
    unittest.main()
