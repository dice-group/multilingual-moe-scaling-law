from __future__ import annotations

import json
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

from multilingual_megatron.experiments.scaling.stage1_l26_e32.production import PhysicalCheckpointSelectionError
from multilingual_megatron.experiments.scaling.stage1_l26_e32.production import build_plan_from_profile
from multilingual_megatron.experiments.scaling.stage1_l26_e32.production import load_production_profile
from multilingual_megatron.experiments.scaling.stage1_l26_e32.production import main
from multilingual_megatron.experiments.scaling.stage1_l26_e32.production import select_500_h100_hour_checkpoint


PROFILES_PATH = REPO_ROOT / "configs/pipeline/profiles/stage1_l26_e32_k200_production.json"
CONFIG_BUNDLE_PATH = REPO_ROOT / "configs/pipeline"
PROFILE_NAME = "stage1_l26_e32_k200_law_trunk_production_v1"
JOINTMOE_LR_PROFILE_NAME = "stage1_l26_e32_k200_jointmoe_lr_trunk_production_v1"


class Stage1L26E32PhysicalCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        cls.plan = build_plan_from_profile(cls.profiles[PROFILE_NAME])
        cls.receipt = {
            "aggregate_tokens_per_second_by_gbs": {
                "32": 51_230.0,
                "64": 102_460.0,
                "128": 204_920.0703537229,
            },
            "startup_h100_gpu_seconds": 960.0,
            "full_checkpoint_h100_gpu_seconds": 480.0,
        }

    def test_selection_is_the_last_complete_update_plus_checkpoint_under_budget(self) -> None:
        result = select_500_h100_hour_checkpoint(
            endpoint_plan=self.plan,
            calibration_receipt=self.receipt,
        )
        selected = result["selected_checkpoint"]
        following = result["next_full_update_plus_checkpoint"]
        self.assertLessEqual(selected["completed_checkpoint_h100_gpu_hours"], 500.0)
        self.assertGreater(following["completed_checkpoint_h100_gpu_hours"], 500.0)
        self.assertLessEqual(selected["profile_readout_horizon_tokens_to_add"], selected["actual_tokens"])
        self.assertLess(selected["actual_tokens"], self.plan["trunk"]["actual_end_tokens"])
        self.assertGreater(selected["actual_tokens"], 0)
        self.assertIn(selected["checkpoint_kind"], {"additional_physical_readout", "existing_scientific_readout"})
        amended_profile = deepcopy(self.profiles[PROFILE_NAME])
        horizons = amended_profile["training"]["train"]["batch_policy"]["readout_horizons_tokens"]
        horizons.append(selected["profile_readout_horizon_tokens_to_add"])
        horizons.sort()
        amended_plan = build_plan_from_profile(amended_profile)
        emitted = next(
            endpoint
            for endpoint in amended_plan["readout_endpoints"]
            if endpoint["target_tokens"] == selected["profile_readout_horizon_tokens_to_add"]
        )
        self.assertEqual(emitted["actual_tokens"], selected["actual_tokens"])
        self.assertEqual(emitted["expected_step"], selected["step"])

    def test_missing_ramp_rate_fails_closed(self) -> None:
        receipt = dict(self.receipt)
        receipt["aggregate_tokens_per_second_by_gbs"] = {"128": 204_920.0}
        with self.assertRaises(PhysicalCheckpointSelectionError):
            select_500_h100_hour_checkpoint(endpoint_plan=self.plan, calibration_receipt=receipt)

    def test_cli_writes_an_auditable_profile_amendment_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "calibration.json"
            output = root / "selection.json"
            receipt.write_text(json.dumps(self.receipt), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "--profiles",
                        str(PROFILES_PATH),
                        "--profile",
                        PROFILE_NAME,
                        "--calibration-receipt",
                        str(receipt),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "selected_from_complete_calibration_receipt")
            self.assertEqual(result["target_h100_gpu_hours"], 500.0)

    def test_selector_resolves_the_inherited_corrected_lr_profile_from_bundle(self) -> None:
        profile = load_production_profile(
            CONFIG_BUNDLE_PATH,
            profile_name=JOINTMOE_LR_PROFILE_NAME,
        )
        self.assertEqual(
            profile["training"]["train"]["batch_policy"]["readout_horizons_tokens"],
            [
                17_170_614_784,
                34_341_308_928,
                43_579_525_633,
                51_512_003_072,
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "selection.json"
            receipt = Path(temporary) / "calibration.json"
            receipt.write_text(json.dumps(self.receipt), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "--profiles",
                        str(CONFIG_BUNDLE_PATH),
                        "--profile",
                        JOINTMOE_LR_PROFILE_NAME,
                        "--calibration-receipt",
                        str(receipt),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            selection = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(selection["status"], "selected_from_complete_calibration_receipt")


if __name__ == "__main__":
    unittest.main()
