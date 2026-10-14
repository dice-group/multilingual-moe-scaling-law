from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Stage1RecipeGeometryTest(unittest.TestCase):
    """Check checkpoint-compatible geometry without the GPU training stack."""

    def test_production_geometries_and_families(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.scaling_registry import (
            ARCHITECTURE_FAMILIES, SCALE_SPECS, scale_spec,
        )

        expected = {
            "scaling_a": (8, 512, 1536, 8, 4),
            "scaling_b": (12, 768, 2304, 12, 4),
            "scaling_c": (16, 1024, 3072, 16, 4),
            "scaling_d": (20, 1280, 3840, 20, 4),
            "scaling_l26": (26, 1664, 4992, 26, 32),
        }
        self.assertEqual(set(SCALE_SPECS), set(expected))
        self.assertEqual(set(ARCHITECTURE_FAMILIES), {"scaling_dense", "scaling_switch_moe"})
        for scale, geometry in expected.items():
            spec = scale_spec(scale, "scaling_switch")
            self.assertEqual(tuple(spec[k] for k in (
                "num_layers", "hidden_size", "ffn_hidden_size",
                "num_attention_heads", "scaling_default_experts",
            )), geometry)
            self.assertEqual(spec["num_query_groups"], spec["num_attention_heads"])
            self.assertEqual((spec["vocab_size"], spec["seq_length"]), (64000, 2048))

    def test_geometry_copies_and_retired_scales(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.scaling_registry import scale_spec

        spec = scale_spec("scaling_a", "scaling_switch")
        spec["hidden_size"] = 1
        self.assertEqual(scale_spec("scaling_a", "scaling_switch")["hidden_size"], 512)
        with self.assertRaises(KeyError):
            scale_spec("100m", "scaling_switch")
        with self.assertRaises(KeyError):
            scale_spec("scaling_a", "generic_moe")


class MegatronBridgeRecipeOverrideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runtime = json.loads(
            (
                REPO_ROOT
                / "configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime.json"
            ).read_text(encoding="utf-8")
        )
        os.environ.update(runtime["training"]["runtime_env"])

    def test_ab_v2_exact_bridge_env_imports_real_optional_dependencies(self) -> None:
        runtime = json.loads(
            (
                REPO_ROOT
                / "configs/scaling_law/multilingual_moe_ab_v2/workflow_runtime.json"
            ).read_text(encoding="utf-8")
        )
        env = dict(os.environ)
        env.update(runtime["training"]["runtime_env"])
        # This subprocess deliberately does not enter Slurm, so the CUDA/cuDNN
        # module paths from the generated job script are unavailable.  Expose
        # the equivalent wheel libraries from the frozen Bridge environment;
        # otherwise importing Transformer Engine tests the login-node linker
        # setup rather than the real ModelOpt/NVRx contract below.
        bridge_prefix = Path(runtime["bridge_python_executable"]).parent.parent
        wheel_library_dirs: list[str] = []
        for site_root in sorted((bridge_prefix / "lib").glob("python*/site-packages")):
            wheel_library_dirs.extend(
                str(path)
                for path in sorted((site_root / "nvidia").glob("*/lib"))
                if path.is_dir()
            )
            for path in (
                site_root / "transformer_engine" / "wheel_lib",
                site_root / "torch" / "lib",
            ):
                if path.is_dir():
                    wheel_library_dirs.append(str(path))
        wheel_library_dirs.append(str(bridge_prefix / "lib"))
        env["LD_LIBRARY_PATH"] = ":".join(wheel_library_dirs)
        env["PYTHONPATH"] = ":".join(
            [
                str(REPO_ROOT / "src"),
                str(REPO_ROOT),
                str(REPO_ROOT / "external/Megatron-Bridge/src"),
                str(REPO_ROOT / "external/Megatron-Bridge/3rdparty/Megatron-LM"),
            ]
        )
        completed = subprocess.run(
            [
                runtime["bridge_python_executable"],
                "-c",
                (
                    "import json; "
                    "import multilingual_megatron.pipeline.backends.megatron_bridge."
                    "run_study_bridge_recipe as r; "
                    "print('PROVENANCE=' + json.dumps(r._OPTIONAL_DEPENDENCY_PROVENANCE, sort_keys=True))"
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        line = next(
            item for item in completed.stdout.splitlines() if item.startswith("PROVENANCE=")
        )
        provenance = json.loads(line.removeprefix("PROVENANCE="))
        self.assertTrue(provenance["modelopt_mode"].startswith("real_"))
        self.assertTrue(Path(provenance["modelopt_file"]).is_file())
        self.assertTrue(provenance["real_modelopt_required"])
        self.assertTrue(provenance["nvrx_inprocess_available"])
        self.assertEqual(
            provenance["nvrx_mode"],
            "real_inprocess_async_disabled_incompatible_0.4.1",
        )
        self.assertFalse(provenance["nvrx_async_checkpoint_available"])
        self.assertFalse(provenance["checkpoint_async_save_required"])
        self.assertFalse(provenance["nvrx_inprocess_restart_enabled"])
        self.assertFalse(provenance["nvrx_straggler_enabled"])

    def test_nested_mixed_precision_override_materializes_recipe(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe import (
            _set_nested_attr,
        )

        class Config:
            mixed_precision = "bf16_mixed"

        config = Config()

        _set_nested_attr(config, "mixed_precision.grad_reduce_in_fp32", False)

        self.assertFalse(config.mixed_precision.grad_reduce_in_fp32)

    def test_ab_v2_dense_and_moe_evaluation_recipes_have_exact_shapes(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe import (
            _apply_cli_overrides,
            _load_recipe,
        )

        dense = _load_recipe(
            "scaling_dense_scale_scaling_a_pretrain_config",
            peft_scheme=None,
            packed_sequence=False,
            seq_length=2048,
            hf_path=None,
        )
        moe = _load_recipe(
            "scaling_switch_moe_scale_scaling_b_pretrain_config",
            peft_scheme=None,
            packed_sequence=False,
            seq_length=2048,
            hf_path=None,
        )
        _apply_cli_overrides(moe, ["model.num_moe_experts=32"])

        self.assertEqual(
            (dense.model.num_layers, dense.model.hidden_size, dense.model.ffn_hidden_size),
            (8, 512, 1536),
        )
        self.assertIsNone(dense.model.num_moe_experts)
        self.assertEqual(
            (moe.model.num_layers, moe.model.hidden_size, moe.model.ffn_hidden_size),
            (12, 768, 2304),
        )
        self.assertEqual(moe.model.num_moe_experts, 32)
        self.assertEqual(moe.model.moe_router_topk, 1)
        self.assertIsNone(moe.model.moe_expert_capacity_factor)
        self.assertFalse(moe.model.moe_pad_expert_input_to_capacity)
        for recipe in (dense, moe):
            self.assertEqual(recipe.optimizer.optimizer, "adam")
            self.assertEqual(recipe.optimizer.adam_beta1, 0.9)
            self.assertEqual(recipe.optimizer.adam_beta2, 0.95)
            self.assertEqual(recipe.optimizer.adam_eps, 1e-8)
            self.assertEqual(recipe.optimizer.weight_decay, 0.1)
            self.assertTrue(recipe.optimizer.decoupled_weight_decay)
            self.assertEqual(recipe.optimizer.clip_grad, 1.0)
            self.assertEqual(recipe.scheduler.start_weight_decay, 0.1)
            self.assertEqual(recipe.scheduler.end_weight_decay, 0.1)
            self.assertEqual(recipe.scheduler.weight_decay_incr_style, "constant")

    def test_recipe_facade_preserves_family_exports_and_geometry(self) -> None:
        from multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe import (
            _load_recipe,
        )
        from multilingual_megatron.pipeline.backends.megatron_bridge import (
            study_bridge_recipes as facade,
        )
        from multilingual_megatron.pipeline.backends.megatron_bridge.scaling_registry import (
            ARCHITECTURE_FAMILIES,
            SCALE_SPECS,
            recipe_name,
        )
        expected_generated = {recipe_name(family, scale)
                              for scale in SCALE_SPECS for family in ARCHITECTURE_FAMILIES}
        self.assertEqual(set(facade.__all__), expected_generated)
        self.assertEqual(len(expected_generated), 10)
        for name in expected_generated:
            recipe = getattr(facade, name)
            self.assertEqual(recipe.__name__, name)
            self.assertEqual(recipe.__module__, facade.__name__)


if __name__ == "__main__":
    unittest.main()
