import json
import tempfile
import unittest
from pathlib import Path

from multilingual_megatron.experiments.scaling.stage1_release.exposure import _exact_exposures, Stage1DataError


class Stage1ExposureTest(unittest.TestCase):
    def test_exact_exposure_uses_blend_ordinals_not_planned_shares(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eng_prefix = root / "eng_Latn"
            fra_prefix = root / "fra_Latn_text_document"
            tier_path = root / "tier.json"
            description_path = root / "description.txt"
            index_path = root / "dataset_index.npy"
            tier_path.write_text(
                json.dumps(
                    {
                        "prefixes_by_language": {
                            "eng_Latn": [str(eng_prefix)],
                            "fra_Latn": [str(fra_prefix)],
                        }
                    }
                ),
                encoding="utf-8",
            )
            description_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"dataset_path": str(fra_prefix)},
                            {"dataset_path": str(eng_prefix)},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            np.save(index_path, np.asarray([0, 1, 1, 0, 1], dtype=np.int16))

            exposure = _exact_exposures(
                index_path=index_path,
                description_path=description_path,
                tier_manifest_path=tier_path,
                end_samples=5,
                sequence_length=4,
            )

            with self.assertRaisesRegex(Stage1DataError, "Invalid cutoff"):
                _exact_exposures(index_path=index_path, description_path=description_path,
                                 tier_manifest_path=tier_path, end_samples=6, sequence_length=4)
            np.save(index_path, np.asarray([0, 2], dtype=np.int16))
            with self.assertRaisesRegex(Stage1DataError, "Invalid dataset ordinal"):
                _exact_exposures(index_path=index_path, description_path=description_path,
                                 tier_manifest_path=tier_path, end_samples=2, sequence_length=4)

        self.assertEqual(exposure, {"fra_Latn": 8.0, "eng_Latn": 12.0})
