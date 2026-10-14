from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from multilingual_megatron.pipeline.stages.sampling.tools.multilingual_moe_scaling import (
    K12_CODES,
    K16_CODES,
    SHORT_CODE_TO_FINEWEB2_SUBSET,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class ScalingLanguageGroupingTest(unittest.TestCase):
    def test_k12_is_exact_atlas_cohort_with_frozen_ordered_hash(self) -> None:
        self.assertEqual(
            K12_CODES,
            ("de", "en", "es", "fr", "hi", "it", "ja", "pl", "pt", "ru", "vi", "zh"),
        )
        self.assertEqual(len(K16_CODES), 16)

        payload = json.loads(
            (
                REPO_ROOT
                / "configs/language_groupings/scaling_fw2_v2_k012.json"
            ).read_text(encoding="utf-8")
        )
        languages = payload["0"]["languages"]
        expected_languages = [SHORT_CODE_TO_FINEWEB2_SUBSET[code] for code in K12_CODES]
        self.assertEqual(languages, expected_languages)

        digest = hashlib.sha256()
        for language in languages:
            digest.update(language.encode("utf-8"))
            digest.update(b"\n")
        self.assertEqual(
            digest.hexdigest(),
            "bbff51cc58457e0e22b228df8bfdc34aa05c91e17acebcd4bd64de981c35839d",
        )
        self.assertEqual(payload["language_set_sha256"], digest.hexdigest())


if __name__ == "__main__":
    unittest.main()
