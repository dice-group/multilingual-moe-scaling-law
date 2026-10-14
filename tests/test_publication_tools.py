"""Integrity checks for the retained anonymous source distribution."""
from __future__ import annotations

import json
import unittest

from multilingual_megatron.devtools.validate_repo_config import main as validate_repo
from multilingual_megatron.devtools.anonymous_provenance import (
    anonymous_source_id, anonymize_metadata,
)


class PublicationToolsTest(unittest.TestCase):
    def test_source_redaction_preserves_public_revisions_and_checksums(self) -> None:
        private_revision, public_revision = "a" * 40, "b" * 40
        original = {"paper_commit": private_revision,
                    "nested": [{"git_commit": private_revision,
                                "dataset_revision": public_revision,
                                "megatron_lm_commit": public_revision}],
                    "sha256": "c" * 64, "loss": 0.03319}
        delivered = anonymize_metadata(original)
        self.assertEqual(original["paper_commit"], private_revision)
        self.assertNotIn(private_revision, json.dumps(delivered))
        self.assertEqual(delivered["paper_commit"], delivered["nested"][0]["git_commit"])
        self.assertNotEqual(anonymous_source_id(private_revision), anonymous_source_id(public_revision))
        self.assertEqual(anonymize_metadata(delivered), delivered)
        self.assertEqual(delivered["nested"][0]["dataset_revision"], public_revision)
        self.assertEqual(delivered["nested"][0]["megatron_lm_commit"], public_revision)
        self.assertEqual(delivered["sha256"], original["sha256"])
        self.assertEqual(delivered["loss"], original["loss"])

    def test_repo_validation_supports_portable_and_full_scopes(self) -> None:
        self.assertEqual(validate_repo(["--scope", "portable"]), 0)
        self.assertEqual(validate_repo(["--scope", "full"]), 0)



if __name__ == "__main__":
    unittest.main()
