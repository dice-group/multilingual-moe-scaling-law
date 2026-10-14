from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multilingual_megatron.experiments.scaling.ab_v2.estimates import _batch_layout
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.spec import CSpecValidationError
from multilingual_megatron.experiments.scaling.stage1_training.c_v1.spec import validate_c_v1_spec
from multilingual_megatron.repo import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__))
SPEC_PATH = REPO_ROOT / "configs/scaling_law/multilingual_moe_c_v1/study_spec.json"


class CStudySpecTests(unittest.TestCase):
    def test_c_spec_is_currently_valid(self) -> None:
        validate_c_v1_spec(repo_root=REPO_ROOT, spec_path=SPEC_PATH)

    def test_c_spec_rejects_e32_data_parallel_drift(self) -> None:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        payload["inherits"]["spec_path"] = str(
            (REPO_ROOT / "configs/scaling_law/multilingual_moe_ab_v2/study_spec.json").resolve()
        )
        payload["execution_policy"]["E32"]["data_parallel_size"] = 1
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "study_spec.json"
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CSpecValidationError):
                validate_c_v1_spec(repo_root=REPO_ROOT, spec_path=candidate)

    def test_e32_batch_ramp_uses_two_data_parallel_ranks(self) -> None:
        runtime = {
            "training": {
                "micro_batch_size": {"C": {"1": 16, "4": 32, "16": 16, "32": 8}},
                "data_parallel_size_by_ladder_and_experts": {
                    "C": {"1": 1, "4": 1, "16": 1, "32": 2}
                },
            }
        }
        self.assertEqual(
            _batch_layout(runtime, ladder="C", experts=32, global_batch_size=32),
            (8, 2, 2),
        )
        self.assertEqual(
            _batch_layout(runtime, ladder="C", experts=32, global_batch_size=128),
            (8, 2, 8),
        )


if __name__ == "__main__":
    unittest.main()
