from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from multilingual_megatron.devtools.run_test_suite import main as run_suite_main
from multilingual_megatron.devtools.test_suites import validate_suite_manifests


REPO_ROOT = Path(__file__).resolve().parents[1]


class CuratedSuiteTests(unittest.TestCase):
    def test_checked_in_suite_manifests_are_valid_and_disjoint(self) -> None:
        self.assertEqual(validate_suite_manifests(REPO_ROOT), [])

    def test_runner_can_list_a_suite_without_importing_tests(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_suite_main(["core_pipeline", "--list-modules"])
        self.assertEqual(result, 0)
        modules = output.getvalue().splitlines()
        self.assertIn("tests.test_repo_maintenance", modules)
        self.assertEqual(len(modules), len(set(modules)))


if __name__ == "__main__":
    unittest.main()
