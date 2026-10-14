from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class StageDataPrepIntegrationTest(unittest.TestCase):
    def test_registry_uses_packaged_stage_implementations(self) -> None:
        from multilingual_megatron.pipeline.registry import get_stage

        stage = get_stage("sample")
        self.assertEqual(stage.name, "sample")
        self.assertTrue(stage._handler.__module__.startswith("multilingual_megatron.pipeline.stages.sampling"))

    def test_packaged_stage_import_exposes_handler(self) -> None:
        from multilingual_megatron.pipeline.stages.sampling import stage_sample

        self.assertTrue(stage_sample.__module__.startswith("multilingual_megatron.pipeline.stages.sampling"))

    def test_stage_packages_use_explicit_implementation_modules(self) -> None:
        stages_root = SRC_ROOT / "multilingual_megatron" / "pipeline" / "stages"
        generic_stage_modules = sorted(stages_root.glob("*/stage.py"))

        self.assertEqual(generic_stage_modules, [])

    def test_tools_packages_are_reserved_for_multi_script_groups(self) -> None:
        stages_root = SRC_ROOT / "multilingual_megatron" / "pipeline" / "stages"
        tool_dirs = sorted({path.parent for path in stages_root.glob("*/tools/*.py")})

        for tool_dir in tool_dirs:
            scripts = sorted(path for path in tool_dir.glob("*.py") if path.name != "__init__.py")
            self.assertGreaterEqual(len(scripts), 3, tool_dir)

    def test_scaling_language_maps_are_frozen_and_resolvable(self) -> None:
        import json

        expected_counts = {
            "scaling_fw2_k001.json": 1,
            "scaling_fw2_k050.json": 50,
            "scaling_fw2_k200.json": 200,
        }
        for file_name, expected_count in expected_counts.items():
            path = REPO_ROOT / "configs" / "language_groupings" / file_name
            payload = json.loads(path.read_text(encoding="utf-8"))
            languages = payload["0"]["languages"]
            self.assertEqual(len(languages), expected_count)
            self.assertEqual(languages[0], "eng_Latn")
            self.assertEqual(len(set(languages)), expected_count)
        self.assertTrue((REPO_ROOT / "configs" / "language_groupings" / "scaling_fw2_language_sets.tsv").exists())

    def test_scaling_data_prep_profiles_resolve_required_inputs(self) -> None:
        from multilingual_megatron.pipeline.runtime import _abs, _load_pipeline_config, _resolve_profile

        config = _load_pipeline_config(REPO_ROOT / "configs" / "pipeline")
        for profile_name in (
            "scaling_fw2_data_prep_base",
            "scaling_fw2_k200_tokenizer_64k_manifest",
            "scaling_fw2_k001_sample_tokenize",
            "scaling_fw2_k050_sample_tokenize",
            "scaling_fw2_k200_sample_tokenize",
        ):
            profile = _resolve_profile(config, profile_name)
            self.assertTrue(_abs(REPO_ROOT, profile["language_map_path"]).exists(), profile_name)
            token_cfg = profile.get("tokenization", {})
            metadata = token_cfg.get("language_metadata")
            if metadata:
                self.assertTrue(_abs(REPO_ROOT, metadata).exists(), profile_name)

    def test_executor_type_override_wins_over_stage_executor(self) -> None:
        from multilingual_megatron.pipeline.cli import _resolve_stage_cfg

        profile = {
            "_executor_type_override": "local",
            "sample": {"enabled": True},
            "stage_executors": {"sample": {"type": "slurm", "partition": "normal"}},
        }
        stage_cfg = _resolve_stage_cfg(profile, "sample")
        self.assertEqual(stage_cfg["executor"]["type"], "local")


if __name__ == "__main__":
    unittest.main()
