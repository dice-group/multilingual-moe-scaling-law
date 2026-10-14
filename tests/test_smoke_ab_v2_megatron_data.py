from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from multilingual_megatron.experiments.scaling.k12_migration.data_smoke import (
    run_smoke,
)


class _Token:
    size = 1

    def __getitem__(self, index: int) -> int:
        if index != 0:
            raise IndexError(index)
        return 7


class _IndexedDataset:
    @staticmethod
    def exists(prefix: str) -> bool:
        return bool(prefix)

    def __init__(self, prefix: str, *, multimodal: bool, mmap: bool) -> None:
        self.prefix = prefix
        self.index = types.SimpleNamespace(dtype="uint16")

    def __len__(self) -> int:
        return 2

    def get(self, index: int, *, length: int) -> _Token:
        if index not in {0, 1} or length != 1:
            raise IndexError(index)
        return _Token()


class MegatronDataSmokeTest(unittest.TestCase):
    def test_opens_all_k_tiers_and_global_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tokenized"
            tiers: dict[str, dict[str, object]] = {}
            for tier in ("K1", "K12", "K50", "K200"):
                manifest_path = root / "tiers" / tier / "megatron_data_manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "languages": ["eng_Latn"],
                            "splits": {
                                "train": {
                                    "prefixes": [str(root / f"{tier}.prefix")],
                                    "total_documents": 2,
                                }
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tiers[tier] = {
                    "path": str(manifest_path.relative_to(root)),
                }
            (root / "tokenization_report.json").write_text(
                json.dumps({"pass": True, "tiers": tiers}) + "\n",
                encoding="utf-8",
            )
            (root / "validation_megatron_data_manifest.json").write_text(
                json.dumps(
                    {
                        "languages": ["eng_Latn"],
                        "splits": {
                            "validation": {
                                "prefixes": [str(root / "validation.prefix")],
                                "total_documents": 2,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            indexed = types.ModuleType("megatron.core.datasets.indexed_dataset")
            indexed.IndexedDataset = _IndexedDataset
            modules = {
                "megatron": types.ModuleType("megatron"),
                "megatron.core": types.ModuleType("megatron.core"),
                "megatron.core.datasets": types.ModuleType("megatron.core.datasets"),
                "megatron.core.datasets.indexed_dataset": indexed,
            }
            output = Path(tmp) / "smoke.json"
            with patch.dict(sys.modules, modules):
                report = run_smoke(root, output)

        self.assertTrue(report["pass"])
        self.assertEqual(report["unique_prefixes_opened"], 5)
        self.assertEqual(
            set(report["manifests"]),
            {"K1", "K12", "K50", "K200", "global_validation"},
        )


if __name__ == "__main__":
    unittest.main()
