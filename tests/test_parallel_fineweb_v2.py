from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from multilingual_megatron.pipeline.stages.sampling.fineweb_v2_policy import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from multilingual_megatron.pipeline.stages.sampling.tools import (  # noqa: E402
    bootstrap_fineweb_v2 as sequential,
)
from multilingual_megatron.pipeline.stages.sampling.tools.parallel_fineweb_v2 import (  # noqa: E402
    merge_fragments,
    run_worker,
)


class ParallelFineWebV2Test(unittest.TestCase):
    def test_parallel_outputs_match_sequential_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "cache"
            paths = []
            for file_index in range(4):
                path = f"data/part_{file_index}.parquet"
                target = cache / path
                target.parent.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "text": f"document {file_index}-{row_index} with enough text",
                        "id": f"id-{file_index}-{row_index}",
                        "metadata": json.dumps({"url": f"https://e/{file_index}/{row_index}"}),
                    }
                    for row_index in range(5)
                ]
                pq.write_table(pa.Table.from_pylist(rows), target)
                paths.append(path)

            revision = "a" * 40
            source_lock = root / "source_lock.json"
            weights = root / "weights.json"
            weights.write_text("{}\n", encoding="utf-8")
            source_lock.write_text(
                json.dumps(
                    {
                        "english_corpus": {
                            "dataset": "HuggingFaceFW/fineweb",
                            "dataset_revision": revision,
                        },
                        "non_english_corpus": {
                            "rehydration": {
                                "enabled": True,
                                "weights_policy": "up_to_10_reps",
                                "missing_weights_policy": "fail_closed_except_explicit_identity_overrides",
                                "identity_overrides": {},
                                "weights_sha256": sha256_file(weights),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            language_map = root / "languages.json"
            language_map.write_text(
                json.dumps({"schema_version": 2, "1": {"languages": ["eng_Latn"]}}),
                encoding="utf-8",
            )
            inventory = root / "inventory.csv"
            inventory.write_text("subset,split,documents,utf8_bytes\n", encoding="utf-8")

            ordered = [path for path, _digest in sequential._ordered_paths(paths, seed=42, subset="eng_Latn")]
            entries = [
                {
                    "index": index,
                    "path": path,
                    "lfs_sha256": sha256_file(cache / path),
                    "lfs_bytes": (cache / path).stat().st_size,
                    "worker": index % 2,
                }
                for index, path in enumerate(ordered)
            ]
            plan = {
                "schema": "ab_v2_parallel_source_cache/v1",
                "dataset": "HuggingFaceFW/fineweb",
                "revision": revision,
                "source_paths_manifest_sha256": "fixture",
                "ordered_paths_sha256": canonical_json_sha256(ordered),
                "file_count": len(entries),
                "workers": 2,
                "entries": entries,
            }
            plan["plan_sha256"] = canonical_json_sha256(plan)
            plan_path = root / "cache_plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            settings = sequential.BootstrapSettings(
                seed=42,
                validation_fraction=0.30,
                target_unique_documents=13,
                target_utf8_bytes=1,
                calibration_documents=2,
            )
            sequential_root = root / "sequential"
            sequential.bootstrap_language(
                language="eng_Latn",
                source_lock_path=source_lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=sequential_root,
                settings=settings,
                local_input_path=cache,
                source_inventory_path=inventory,
            )
            fragments = root / "fragments"
            for rank in range(2):
                run_worker(
                    cache_plan_path=plan_path,
                    cache_root=cache,
                    fragment_root=fragments,
                    source_lock_path=source_lock,
                    rank=rank,
                    world_size=2,
                    seed=42,
                    validation_fraction=0.30,
                )
            parallel_root = root / "parallel"
            merge_fragments(
                cache_plan_path=plan_path,
                cache_root=cache,
                fragment_root=fragments,
                output_root=parallel_root,
                source_lock_path=source_lock,
                rehydration_weights_path=weights,
                source_inventory_path=inventory,
                language_map_path=language_map,
                sampling_plan_path=None,
                world_size=2,
                seed=42,
                validation_fraction=0.30,
                safety_margin=1.10,
                settings_override=settings,
            )
            for name in sequential.OUTPUT_NAMES:
                self.assertEqual(
                    (sequential_root / "eng_Latn" / name).read_bytes(),
                    (parallel_root / "eng_Latn" / name).read_bytes(),
                    name,
                )
            summary = sequential.merge_bootstrap_release(
                source_lock_path=source_lock,
                rehydration_weights_path=weights,
                language_map_path=language_map,
                output_root=parallel_root,
                selected_languages=("eng_Latn",),
                source_inventory_path=inventory,
            )
            self.assertEqual(summary["languages"], ["eng_Latn"])


if __name__ == "__main__":
    unittest.main()
