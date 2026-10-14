from __future__ import annotations

import unittest

from multilingual_megatron.experiments.scaling.stage1_release.contracts import (
    Stage1ReleaseError,
)
from multilingual_megatron.experiments.scaling.stage1_release.provenance import append_analysis_protocol_states


class Stage1ADReleaseProvenanceTest(unittest.TestCase):
    def test_protocol_sequence_alias_carries_frozen_receipt_seal(self) -> None:
        rows = [
            {
                "analysis_artifact_id": "freeze_receipt.json",
                "artifact_role": "predeclared_freeze_receipt",
                "path": "/evidence/freeze_receipt.json",
                "exists": True,
                "size_bytes": 123,
                "sha256": "a" * 64,
                "status": "observed",
                "scope_disclosure": "A/B only",
                "provenance_class": "observed",
            }
        ]

        result = append_analysis_protocol_states(rows)
        by_id = {row["analysis_artifact_id"]: row for row in result}
        frozen = by_id["frozen_ab_model_selection"]

        self.assertEqual(frozen["path"], "/evidence/freeze_receipt.json")
        self.assertEqual(frozen["size_bytes"], 123)
        self.assertEqual(frozen["sha256"], "a" * 64)
        self.assertEqual(
            frozen["alias_of_analysis_artifact_id"],
            "freeze_receipt.json",
        )
        self.assertEqual(
            by_id["frozen_cd_m1_score"]["depends_on"],
            "frozen_ab_model_selection",
        )
        self.assertEqual(
            by_id["declared_ad_refit"]["depends_on"],
            "frozen_cd_m1_score",
        )

    def test_protocol_sequence_fails_without_a_sealed_freeze_receipt(self) -> None:
        with self.assertRaisesRegex(Stage1ReleaseError, "freeze receipt"):
            append_analysis_protocol_states([])


if __name__ == "__main__":
    unittest.main()
