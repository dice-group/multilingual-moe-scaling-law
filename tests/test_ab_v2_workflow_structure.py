from __future__ import annotations

import unittest
from pathlib import Path

from multilingual_megatron.experiments.scaling.ab_v2 import common, common_provenance
from multilingual_megatron.experiments.scaling.ab_v2.policy import AB_VARIANT
from multilingual_megatron.experiments.scaling.ab_v2.variant import active_variant


REPO_ROOT = Path(__file__).resolve().parents[1]
SCALING_ROOT = REPO_ROOT / "src/multilingual_megatron/experiments/scaling"


class AbV2WorkflowStructureTest(unittest.TestCase):
    def test_hash_contract_covers_the_complete_workflow_source_bundle(self) -> None:
        expected = {
            (SCALING_ROOT / "ab_v2/workflow.py").resolve(),
            *(path.resolve() for path in (SCALING_ROOT / "ab_v2").rglob("*.py")),
        }
        self.assertEqual(set(common._workflow_source_files()), expected)
        bootstrap_tools = (
            REPO_ROOT
            / "src/multilingual_megatron/pipeline/stages/sampling/tools"
        )
        bootstrap_implementation = bootstrap_tools / "fineweb_v2_bootstrap"
        self.assertEqual(
            common_provenance._fineweb_bootstrap_source_files(),
            (
                (bootstrap_tools / "bootstrap_fineweb_v2.py").resolve(),
                *tuple(
                    path.resolve()
                    for path in sorted(bootstrap_implementation.rglob("*.py"))
                ),
            ),
        )

    def test_c_and_d_imports_do_not_change_the_active_ab_variant(self) -> None:
        expected = AB_VARIANT

        from multilingual_megatron.experiments.scaling.stage1_training.c_v1 import workflow as c_v1_workflow
        from multilingual_megatron.experiments.scaling.stage1_training.d_v1 import workflow as d_v1_workflow

        self.assertIs(active_variant(), expected)

    def test_c_and_d_expose_immutable_variant_contracts(self) -> None:
        from multilingual_megatron.experiments.scaling.stage1_training.c_v1 import workflow as c_v1_workflow
        from multilingual_megatron.experiments.scaling.stage1_training.d_v1 import workflow as d_v1_workflow

        self.assertEqual(c_v1_workflow.C_VARIANT.policy.ladders, ("C",))
        self.assertEqual(d_v1_workflow.D_VARIANT.policy.ladders, ("D",))
        self.assertNotEqual(
            c_v1_workflow.C_VARIANT.manifest_descriptor(),
            d_v1_workflow.D_VARIANT.manifest_descriptor(),
        )


if __name__ == "__main__":
    unittest.main()
