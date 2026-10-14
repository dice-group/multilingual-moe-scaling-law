from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "scripts" / "h100_deepep_preflight.py"


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("h100_deepep_preflight_for_test", PREFLIGHT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PREFLIGHT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class H100DeepEPPreflightTest(unittest.TestCase):
    def test_as_bool_env_parses_common_truthy_values(self) -> None:
        module = _load_preflight_module()

        with patch.dict("os.environ", {"DEEPEP_PREFLIGHT_HYBRIDEP_JIT": "yes"}, clear=True):
            self.assertTrue(module._as_bool_env("DEEPEP_PREFLIGHT_HYBRIDEP_JIT"))

        with patch.dict("os.environ", {"DEEPEP_PREFLIGHT_HYBRIDEP_JIT": "0"}, clear=True):
            self.assertFalse(module._as_bool_env("DEEPEP_PREFLIGHT_HYBRIDEP_JIT"))

        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(module._as_bool_env("DEEPEP_PREFLIGHT_HYBRIDEP_JIT", default=True))

    def test_build_top1_routing_uses_megatron_core_layout(self) -> None:
        module = _load_preflight_module()

        routing_map, probs = module._build_top1_routing(
            num_tokens=5,
            num_experts=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(routing_map.shape), (5, 3))
        self.assertEqual(tuple(probs.shape), (5, 3))
        self.assertEqual(routing_map.dtype, torch.bool)
        self.assertEqual(probs.dtype, torch.float32)
        self.assertEqual(routing_map.sum(dim=1).tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(probs.sum(dim=1).tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(routing_map.nonzero(as_tuple=False)[:, 1].tolist(), [0, 1, 2, 0, 1])

    def test_hybridep_availability_records_missing_symbol_without_required_gate(self) -> None:
        module = _load_preflight_module()
        fake_deep_ep = types.ModuleType("deep_ep")
        result: dict[str, object] = {}

        with patch.dict(sys.modules, {"deep_ep": fake_deep_ep}), patch.dict(
            "os.environ", {}, clear=True
        ):
            module._record_hybridep_availability(result)

        self.assertFalse(result["deep_ep_has_hybridep"])
        self.assertIn("ImportError", str(result["hybridep_import_error"]))
        self.assertIn("HybridEPBuffer", str(result["hybridep_import_error"]))

    def test_hybridep_availability_records_present_symbol(self) -> None:
        module = _load_preflight_module()
        fake_deep_ep = types.ModuleType("deep_ep")
        fake_deep_ep.HybridEPBuffer = object
        result: dict[str, object] = {}

        with patch.dict(sys.modules, {"deep_ep": fake_deep_ep}), patch.dict(
            "os.environ", {}, clear=True
        ):
            module._record_hybridep_availability(result)

        self.assertTrue(result["deep_ep_has_hybridep"])
        self.assertNotIn("hybridep_import_error", result)

    def test_hybridep_required_gate_raises_when_symbol_is_missing(self) -> None:
        module = _load_preflight_module()
        fake_deep_ep = types.ModuleType("deep_ep")

        with patch.dict(sys.modules, {"deep_ep": fake_deep_ep}), patch.dict(
            "os.environ", {"DEEPEP_PREFLIGHT_REQUIRE_HYBRIDEP": "1"}, clear=True
        ):
            with self.assertRaises(ImportError):
                module._record_hybridep_availability({})

    def test_hybridep_jit_preflight_is_opt_in(self) -> None:
        module = _load_preflight_module()
        result: dict[str, object] = {"deep_ep_has_hybridep": True}

        with patch.dict("os.environ", {}, clear=True):
            module._run_hybridep_jit_preflight(
                result=result,
                device=torch.device("cpu"),
                num_tokens=4,
                hidden=8,
            )

        self.assertFalse(result["hybridep_jit_requested"])
        self.assertNotIn("hybridep_jit_status", result)

    def test_hybridep_jit_preflight_requires_hybridep_symbol(self) -> None:
        module = _load_preflight_module()
        result: dict[str, object] = {"deep_ep_has_hybridep": False}

        with patch.dict("os.environ", {"DEEPEP_PREFLIGHT_HYBRIDEP_JIT": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires HybridEPBuffer"):
                module._run_hybridep_jit_preflight(
                    result=result,
                    device=torch.device("cpu"),
                    num_tokens=4,
                    hidden=8,
                )

    def test_nvrx_version_shim_adds_missing_version_for_mcore_import(self) -> None:
        module = _load_preflight_module()
        fake_nvrx = types.ModuleType("nvidia_resiliency_ext")
        result: dict[str, object] = {}

        with patch.dict(sys.modules, {"nvidia_resiliency_ext": fake_nvrx}), patch.dict(
            "os.environ", {}, clear=True
        ):
            module._ensure_nvrx_version_for_mcore_import(result)

        self.assertTrue(result["nvrx_importable"])
        self.assertTrue(result["nvrx_version_shimmed"])
        self.assertEqual(result["nvrx_version"], "999.0.0")
        self.assertEqual(fake_nvrx.__version__, "999.0.0")

    def test_nvrx_version_shim_keeps_existing_version(self) -> None:
        module = _load_preflight_module()
        fake_nvrx = types.ModuleType("nvidia_resiliency_ext")
        fake_nvrx.__version__ = "0.8.0"
        result: dict[str, object] = {}

        with patch.dict(sys.modules, {"nvidia_resiliency_ext": fake_nvrx}):
            module._ensure_nvrx_version_for_mcore_import(result)

        self.assertTrue(result["nvrx_importable"])
        self.assertFalse(result["nvrx_version_shimmed"])
        self.assertEqual(result["nvrx_version"], "0.8.0")


if __name__ == "__main__":
    unittest.main()
