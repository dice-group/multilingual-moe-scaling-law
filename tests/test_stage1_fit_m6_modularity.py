from __future__ import annotations

import unittest
from pathlib import Path

from multilingual_megatron.experiments.scaling.stage1_fit import (
    m5_m6,
)
from multilingual_megatron.experiments.scaling.stage1_fit.workflow_source import (
    bind_source_bundle,
)


class Stage1M6ModularityTest(unittest.TestCase):
    def test_every_source_bundle_is_verified_by_its_public_entrypoint(self) -> None:
        cases = (
            (m5_m6, m5_m6._runner_source_binding),
        )
        for module, binder in cases:
            with self.subTest(module=module.__name__):
                binding = binder()
                self.assertEqual(
                    Path(binding["path"]).resolve(),
                    Path(str(module.__file__)).resolve(),
                )
                self.assertEqual(set(binding), {"path", "file_sha256"})

    def test_source_bundle_verification_fails_closed(self) -> None:
        entrypoint = Path(str(m5_m6.__file__)).resolve()
        changed = dict(m5_m6.SOURCE_BUNDLE)
        changed[next(iter(changed))] = "0" * 64
        with self.assertRaises(m5_m6.M5M6Error):
            bind_source_bundle(
                entrypoint=entrypoint,
                source_root=entrypoint.parent,
                expected_sources=changed,
                error_type=m5_m6.M5M6Error,
            )


if __name__ == "__main__":
    unittest.main()
