from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from multilingual_megatron.pipeline.backends.megatron_bridge.runtime_patches import (
    install_bridge_cp_rank_compat_patch,
    install_ddp_default_stream_alignment_patch,
    install_fused_te_moe_act_checkpoint_patch,
    install_mtp_full_recompute_padding_mask_compat_patch,
    install_moe_allgather_ep1_topk1_fastpath,
    install_moe_alltoall_batched_dispatch_patch,
    install_moe_alltoall_cpu_split_list_patch,
    install_moe_alltoall_nccl_stream_patch,
    install_moe_alltoall_padded_equal_splits_patch,
    install_moe_alltoall_sync_profiler,
    install_p2p_batch_sync_disabled_guard,
    install_te_grouped_mlp_inspector,
)


class MegatronBridgeRuntimePatchesTest(unittest.TestCase):
    def test_ddp_default_stream_alignment_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_ddp_default_stream_alignment_patch())

    def test_ddp_default_stream_alignment_patch_changes_only_bridge_ddp_wrap_stream(self) -> None:
        active_stream = object()
        regular_stream = object()

        def fake_ddp_wrap():
            return torch.cuda.Stream()

        unimodal_module = types.ModuleType("megatron.bridge.models.common.unimodal")
        unimodal_module._ddp_wrap = fake_ddp_wrap
        model_provider_module = types.ModuleType("megatron.bridge.models.model_provider")
        model_provider_module._ddp_wrap = fake_ddp_wrap
        common_package = types.ModuleType("megatron.bridge.models.common")
        common_package.unimodal = unimodal_module
        models_package = types.ModuleType("megatron.bridge.models")
        models_package.common = common_package
        models_package.model_provider = model_provider_module
        bridge_package = types.ModuleType("megatron.bridge")
        bridge_package.models = models_package
        megatron_package = types.ModuleType("megatron")
        megatron_package.bridge = bridge_package
        module_map = {
            "megatron": megatron_package,
            "megatron.bridge": bridge_package,
            "megatron.bridge.models": models_package,
            "megatron.bridge.models.common": common_package,
            "megatron.bridge.models.common.unimodal": unimodal_module,
            "megatron.bridge.models.model_provider": model_provider_module,
        }

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ALIGN_DDP_INIT_TO_ACTIVE_STREAM": "1"}, clear=True
        ), patch.object(torch.cuda, "current_stream", return_value=active_stream), patch.object(
            torch.cuda, "Stream", return_value=regular_stream
        ) as stream_constructor:
            self.assertTrue(install_ddp_default_stream_alignment_patch())
            self.assertIs(unimodal_module._ddp_wrap(), active_stream)
            self.assertIs(model_provider_module._ddp_wrap(), active_stream)
            self.assertIs(torch.cuda.Stream(), regular_stream)
            self.assertEqual(stream_constructor.call_count, 1)

    def test_bridge_cp_rank_compat_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_bridge_cp_rank_compat_patch())

    def test_moe_fastpath_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_moe_allgather_ep1_topk1_fastpath())

    def test_moe_alltoall_sync_profiler_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_moe_alltoall_sync_profiler())

    def test_moe_alltoall_nccl_stream_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_moe_alltoall_nccl_stream_patch())

    def test_moe_alltoall_batched_dispatch_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_moe_alltoall_batched_dispatch_patch())

    def test_moe_alltoall_cpu_split_list_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_moe_alltoall_cpu_split_list_patch())

    def test_moe_alltoall_padded_equal_splits_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_moe_alltoall_padded_equal_splits_patch())

    def test_te_grouped_mlp_inspector_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_te_grouped_mlp_inspector())

    def test_fused_te_moe_act_checkpoint_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_fused_te_moe_act_checkpoint_patch())

    def test_mtp_full_recompute_padding_mask_compat_patch_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_mtp_full_recompute_padding_mask_compat_patch())

    def test_mtp_full_recompute_padding_mask_compat_matches_nonrecompute_path(self) -> None:
        class FakeMTP:
            def __init__(self) -> None:
                self.config = SimpleNamespace(recompute_granularity="full")
                self.calls: list[tuple[object, object]] = []

            def _checkpointed_forward(self, hidden_states, attention_mask=None):
                self.calls.append((hidden_states, attention_mask))
                return hidden_states

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.multi_token_prediction",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].multi_token_prediction = module_map[
            "megatron.core.transformer.multi_token_prediction"
        ]
        module_map["megatron.core.transformer.multi_token_prediction"].MultiTokenPredictionLayer = FakeMTP

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_MTP_FULL_RECOMPUTE_PADDING_MASK_COMPAT": "1"}, clear=True
        ):
            self.assertTrue(install_mtp_full_recompute_padding_mask_compat_patch())
            instance = FakeMTP()
            value = object()
            self.assertIs(
                instance._checkpointed_forward(value, attention_mask="attention", padding_mask="unused"),
                value,
            )

        self.assertEqual(instance.calls, [(value, "attention")])

    def test_p2p_batch_sync_disabled_guard_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(install_p2p_batch_sync_disabled_guard())

    def test_p2p_batch_sync_disabled_guard_checks_live_communicator_config(self) -> None:
        calls: list[str] = []

        class FakeP2PCommunicator:
            def __init__(self, config) -> None:
                self.config = config

            def _communicate(self, marker):
                calls.append(marker)
                return marker

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.pipeline_parallel",
            "megatron.core.pipeline_parallel.p2p_communication",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].pipeline_parallel = module_map[
            "megatron.core.pipeline_parallel"
        ]
        module_map["megatron.core.pipeline_parallel"].p2p_communication = module_map[
            "megatron.core.pipeline_parallel.p2p_communication"
        ]
        module_map["megatron.core.pipeline_parallel.p2p_communication"].P2PCommunicator = (
            FakeP2PCommunicator
        )

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_REQUIRE_P2P_BATCH_SYNC_DISABLED": "1"}, clear=True
        ):
            self.assertTrue(install_p2p_batch_sync_disabled_guard())
            healthy = FakeP2PCommunicator(
                SimpleNamespace(
                    batch_p2p_comm=True,
                    batch_p2p_sync=False,
                    overlap_p2p_comm=False,
                )
            )
            self.assertEqual(healthy._communicate("healthy"), "healthy")
            unhealthy = FakeP2PCommunicator(
                SimpleNamespace(
                    batch_p2p_comm=True,
                    batch_p2p_sync=True,
                    overlap_p2p_comm=False,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "batch_p2p_sync=True"):
                unhealthy._communicate("unhealthy")

        self.assertEqual(calls, ["healthy"])

    def test_bridge_cp_rank_compat_patch_adds_is_hybrid_cp_default(self) -> None:
        calls: list[dict[str, object]] = []

        def get_batch_on_this_cp_rank(batch, is_hybrid_cp, cp_group=None):
            calls.append(
                {
                    "batch": batch,
                    "is_hybrid_cp": is_hybrid_cp,
                    "cp_group": cp_group,
                }
            )
            return batch

        module_names = [
            "megatron",
            "megatron.bridge",
            "megatron.bridge.training",
            "megatron.bridge.training.gpt_step",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].bridge = module_map["megatron.bridge"]
        module_map["megatron.bridge"].training = module_map["megatron.bridge.training"]
        module_map["megatron.bridge.training"].gpt_step = module_map[
            "megatron.bridge.training.gpt_step"
        ]
        gpt_step = module_map["megatron.bridge.training.gpt_step"]
        gpt_step.get_batch_on_this_cp_rank = get_batch_on_this_cp_rank

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_BRIDGE_CP_RANK_COMPAT": "1"}, clear=True
        ):
            self.assertTrue(install_bridge_cp_rank_compat_patch())
            output = gpt_step.get_batch_on_this_cp_rank({"tokens": "x"}, cp_group="cp")

        self.assertEqual(output, {"tokens": "x"})
        self.assertEqual(
            calls,
            [{"batch": {"tokens": "x"}, "is_hybrid_cp": False, "cp_group": "cp"}],
        )

    def test_moe_fastpath_uses_static_num_out_tokens_for_ep1_topk1(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeDispatcher:
            def dispatch_postprocess(self, hidden_states, probs):
                return "original", hidden_states, probs

        def fake_permute(tokens, routing_map, num_out_tokens=None, fused=False):
            calls.append(
                {
                    "num_out_tokens": num_out_tokens,
                    "fused": fused,
                    "routing_shape": tuple(routing_map.shape),
                }
            )
            return tokens, None, torch.arange(tokens.shape[0]), None, None

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.token_dispatcher",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].token_dispatcher = module_map[
            "megatron.core.transformer.moe.token_dispatcher"
        ]
        token_dispatcher = module_map["megatron.core.transformer.moe.token_dispatcher"]
        token_dispatcher.MoEAllGatherTokenDispatcher = FakeDispatcher
        token_dispatcher.permute = fake_permute

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_MOE_EP1_TOPK1_FASTPATH": "1"}, clear=True
        ):
            self.assertTrue(install_moe_allgather_ep1_topk1_fastpath())

            instance = FakeDispatcher()
            instance.tp_size = 1
            instance.ep_size = 1
            instance.num_local_experts = 4
            instance.local_expert_indices = [0, 1, 2, 3]
            instance.config = SimpleNamespace(
                moe_router_topk=1,
                moe_expert_capacity_factor=None,
                moe_pad_expert_input_to_capacity=False,
                num_moe_experts=4,
                moe_permute_fusion=True,
            )
            instance.routing_map = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                ],
                dtype=torch.bool,
            )
            hidden_states = torch.randn(5, 3)
            probs = torch.rand(5, 4)

            output, tokens_per_expert, local_probs = instance.dispatch_postprocess(
                hidden_states, probs
            )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(tokens_per_expert.tolist(), [2, 1, 1, 1])
        self.assertEqual(local_probs.shape, (5,))
        self.assertEqual(calls, [{"num_out_tokens": 5, "fused": True, "routing_shape": (5, 4)}])

    def test_moe_alltoall_sync_profiler_installs_with_point_filter(self) -> None:
        calls: list[str] = []

        class FakeDispatcher:
            def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
                calls.append(point)
                return tokens_per_expert

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.token_dispatcher",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].token_dispatcher = module_map[
            "megatron.core.transformer.moe.token_dispatcher"
        ]
        token_dispatcher = module_map["megatron.core.transformer.moe.token_dispatcher"]
        token_dispatcher.MoEAlltoAllTokenDispatcher = FakeDispatcher

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ",
            {
                "PROOF_PROFILE_MOE_A2A_SYNC": "1",
                "PROOF_PROFILE_MOE_A2A_SYNC_POINTS": "before_ep_alltoall",
            },
            clear=True,
        ):
            self.assertTrue(install_moe_alltoall_sync_profiler())
            instance = FakeDispatcher()
            instance.cuda_dtoh_point = "before_permutation_1"
            instance.cuda_sync_point = "before_ep_alltoall"
            instance.ep_size = 2
            instance.tp_size = 1
            instance.num_local_experts = 8
            self.assertIs(instance._maybe_dtoh_and_synchronize("before_permutation_1", "x"), "x")
            self.assertIs(instance._maybe_dtoh_and_synchronize("before_ep_alltoall", "y"), "y")

        self.assertEqual(calls, ["before_permutation_1", "before_ep_alltoall"])

    def test_moe_alltoall_nccl_stream_patch_forces_existing_dispatcher_flag(self) -> None:
        class FakeDispatcher:
            def __init__(self, marker=None):
                self.marker = marker
                self.use_nccl_stream = False

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.token_dispatcher",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].token_dispatcher = module_map[
            "megatron.core.transformer.moe.token_dispatcher"
        ]
        token_dispatcher = module_map["megatron.core.transformer.moe.token_dispatcher"]
        token_dispatcher.MoEAlltoAllTokenDispatcher = FakeDispatcher

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_MOE_A2A_NCCL_STREAM": "1"}, clear=True
        ):
            self.assertTrue(install_moe_alltoall_nccl_stream_patch())
            instance = FakeDispatcher(marker="x")

        self.assertEqual(instance.marker, "x")
        self.assertTrue(instance.use_nccl_stream)

    def test_moe_alltoall_batched_dispatch_launches_both_collectives_before_wait(self) -> None:
        events: list[str] = []

        class FakeGroup:
            def size(self):
                return 4

        class FakeHandle:
            def __init__(self, name: str):
                self.name = name

            def wait(self):
                events.append(f"wait:{self.name}")

        class FakeDistributed:
            def all_to_all_single(
                self,
                output,
                input,
                output_split_sizes=None,
                input_split_sizes=None,
                group=None,
                async_op=False,
            ):
                self_name = "tokens" if input.dim() == 2 else "probs"
                events.append(f"launch:{self_name}:async={async_op}")
                output.copy_(input)
                return FakeHandle(self_name)

        class FakeDispatcher:
            def token_dispatch(self, tokens, probs):
                return "original", tokens, probs

            def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
                events.append(f"sync:{point}")
                return tokens_per_expert

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.token_dispatcher",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].token_dispatcher = module_map[
            "megatron.core.transformer.moe.token_dispatcher"
        ]
        token_dispatcher = module_map["megatron.core.transformer.moe.token_dispatcher"]
        token_dispatcher.MoEAlltoAllTokenDispatcher = FakeDispatcher

        original_distributed = torch.distributed
        torch.distributed = FakeDistributed()  # type: ignore[assignment]
        try:
            with patch.dict(sys.modules, module_map), patch.dict(
                "os.environ", {"PROOF_ENABLE_MOE_A2A_BATCHED_DISPATCH": "1"}, clear=True
            ):
                self.assertTrue(install_moe_alltoall_batched_dispatch_patch())
                instance = FakeDispatcher()
                instance.shared_experts = None
                instance.ep_group = FakeGroup()
                instance.tokens_per_expert = torch.tensor([1, 1, 1, 1])
                instance.input_splits = None
                instance.output_splits = None
                tokens = torch.randn(4, 3, requires_grad=True)
                probs = torch.randn(4, requires_grad=True)

                output_tokens, output_probs = instance.token_dispatch(tokens, probs)
                loss = output_tokens.sum() + output_probs.sum()
                loss.backward()
        finally:
            torch.distributed = original_distributed  # type: ignore[assignment]

        self.assertEqual(
            events[:5],
            [
                "sync:before_ep_alltoall",
                "launch:tokens:async=True",
                "launch:probs:async=True",
                "wait:tokens",
                "wait:probs",
            ],
        )
        self.assertEqual(tokens.grad.shape, tokens.shape)
        self.assertEqual(probs.grad.shape, probs.shape)

    def test_moe_alltoall_cpu_split_list_patch_normalizes_sync_point_splits(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeArray:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        class FakeDispatcher:
            def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
                calls.append((point, tokens_per_expert))
                return tokens_per_expert

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.token_dispatcher",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].token_dispatcher = module_map[
            "megatron.core.transformer.moe.token_dispatcher"
        ]
        token_dispatcher = module_map["megatron.core.transformer.moe.token_dispatcher"]
        token_dispatcher.MoEAlltoAllTokenDispatcher = FakeDispatcher

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_MOE_A2A_CPU_SPLIT_LISTS": "1"}, clear=True
        ):
            self.assertTrue(install_moe_alltoall_cpu_split_list_patch())
            instance = FakeDispatcher()
            instance.cuda_sync_point = "before_ep_alltoall"
            instance.input_splits = FakeArray([1, 2])
            instance.output_splits = FakeArray([3, 4])
            instance.output_splits_tp = (5, 6)

            self.assertIs(
                instance._maybe_dtoh_and_synchronize("before_permutation_1", "unchanged"),
                "unchanged",
            )
            self.assertIsInstance(instance.input_splits, FakeArray)

            self.assertIs(
                instance._maybe_dtoh_and_synchronize("before_ep_alltoall", "converted"),
                "converted",
            )

        self.assertEqual(calls, [("before_permutation_1", "unchanged"), ("before_ep_alltoall", "converted")])
        self.assertEqual(instance.input_splits, [1, 2])
        self.assertEqual(instance.output_splits, [3, 4])
        self.assertEqual(instance.output_splits_tp, [5, 6])

    def test_moe_alltoall_padded_equal_splits_patch_installs_dispatch_and_combine(self) -> None:
        class FakeDispatcher:
            def token_dispatch(self, tokens, probs):
                return "original_dispatch", tokens, probs

            def token_combine(
                self,
                hidden_states,
                async_finish: bool = True,
                allocate_on_comm_stream: bool = True,
            ):
                return "original_combine", hidden_states

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.token_dispatcher",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].token_dispatcher = module_map[
            "megatron.core.transformer.moe.token_dispatcher"
        ]
        token_dispatcher = module_map["megatron.core.transformer.moe.token_dispatcher"]
        token_dispatcher.MoEAlltoAllTokenDispatcher = FakeDispatcher

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_MOE_A2A_PADDED_EQUAL_SPLITS": "1"}, clear=True
        ):
            self.assertTrue(install_moe_alltoall_padded_equal_splits_patch())

        self.assertTrue(
            getattr(FakeDispatcher.token_dispatch, "_study_a2a_padded_equal_splits_patch", False)
        )
        self.assertTrue(
            getattr(FakeDispatcher.token_combine, "_study_a2a_padded_equal_splits_patch", False)
        )

    def test_fused_te_moe_act_checkpoint_splits_ops_and_preserves_gradients(self) -> None:
        class FakeCheckpoint:
            instances: list["FakeCheckpoint"] = []

            def __init__(self) -> None:
                self.discarded = None
                FakeCheckpoint.instances.append(self)

            def checkpoint(self, function, *args):
                return function(*args)

            def discard_output_and_register_recompute(self, output) -> None:
                self.discarded = output

        class FakeGroup:
            def size(self) -> int:
                return 1

        class FakeOp(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def forward(self, value, *extra):
                if self.name == "fc1":
                    self.assert_extra_count(extra, 1)
                    return value * 2
                if self.name == "activation":
                    self.assert_extra_count(extra, 1)
                    return value * extra[0].unsqueeze(-1)
                self.assert_extra_count(extra, 1)
                return value + 1

            @staticmethod
            def assert_extra_count(extra, expected: int) -> None:
                if len(extra) != expected:
                    raise RuntimeError(f"expected {expected} extra inputs, got {len(extra)}")

        class ScaledSReLU(FakeOp):
            def __init__(self) -> None:
                super().__init__("activation")
                self.activation_recompute_in_mlp = True

        class FakeOps(torch.nn.Module):
            def __init__(self, *ops: torch.nn.Module) -> None:
                super().__init__()
                self.ops = torch.nn.ModuleList(ops)

            def __len__(self) -> int:
                return len(self.ops)

            def __getitem__(self, item):
                if isinstance(item, slice):
                    return FakeOps(*list(self.ops)[item])
                return self.ops[item]

            def forward(self, value, *extra):
                if len(self.ops) != 1:
                    raise RuntimeError("test fake only executes split operation containers")
                return self.ops[0](value, *extra)

        class FakeGroupedMLP:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    gated_linear_unit=True,
                    fp8=False,
                    fp4=False,
                    moe_paged_stash=False,
                    moe_apply_probs_on_input=False,
                )
                self._with_fused_impl = True
                self.activation_recompute = True
                self.tp_group = FakeGroup()
                self.linear_fc1 = SimpleNamespace(delay_wgrad_compute=False)
                self.linear_fc2 = SimpleNamespace(delay_wgrad_compute=False)
                self.offload_expert_fc1 = False
                self.offload_moe_act = False
                self._fused_ops = None
                self.make_ops_calls = 0
                self.pre_forward_hook_calls = 0
                self.original_calls = 0

            def _make_fused_ops(self):
                self.make_ops_calls += 1
                return FakeOps(FakeOp("fc1"), ScaledSReLU(), FakeOp("fc2"))

            def _make_fused_impl_pre_forward_hook(self):
                def hook(*_):
                    self.pre_forward_hook_calls += 1

                return hook

            def _fused_forward(self, value, tokens_per_expert, probs):
                self.original_calls += 1
                return value

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.tensor_parallel",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.experts",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].tensor_parallel = module_map["megatron.core.tensor_parallel"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].experts = module_map[
            "megatron.core.transformer.moe.experts"
        ]
        module_map["megatron.core.tensor_parallel"].CheckpointWithoutOutput = FakeCheckpoint
        module_map["megatron.core.transformer.moe.experts"].TEGroupedMLP = FakeGroupedMLP

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_FUSED_TE_MOE_ACT_CHECKPOINT": "1"}, clear=True
        ):
            self.assertTrue(install_fused_te_moe_act_checkpoint_patch())
            instance = FakeGroupedMLP()
            hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
            hidden_states.requires_grad_(True)
            probs = torch.tensor([0.5, 0.25], dtype=torch.bfloat16)
            probs.requires_grad_(True)
            output = instance._fused_forward(hidden_states, torch.tensor([1, 1]), probs)
            output.float().sum().backward()

        expected = hidden_states.detach() * 2 * probs.detach().unsqueeze(-1) + 1
        torch.testing.assert_close(output.float(), expected.float())
        self.assertEqual(instance.make_ops_calls, 1)
        self.assertEqual(instance.pre_forward_hook_calls, 1)
        self.assertEqual(instance.original_calls, 0)
        self.assertEqual(len(FakeCheckpoint.instances), 1)
        self.assertIs(FakeCheckpoint.instances[0].discarded, output)
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(probs.grad)

        with self.assertRaisesRegex(RuntimeError, "zero-copy buffers"):
            instance._fused_forward(
                hidden_states,
                torch.tensor([1, 1]),
                probs,
                torch.empty_like(hidden_states),
                None,
            )

    def test_fused_te_moe_act_checkpoint_fails_closed_for_delayed_wgrad(self) -> None:
        class FakeCheckpoint:
            def checkpoint(self, function, *args):
                return function(*args)

            def discard_output_and_register_recompute(self, output) -> None:
                return None

        class FakeGroupedMLP:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    gated_linear_unit=True,
                    fp8=False,
                    fp4=False,
                    moe_paged_stash=False,
                    moe_apply_probs_on_input=False,
                )
                self._with_fused_impl = True
                self.activation_recompute = True
                self.tp_group = SimpleNamespace(size=lambda: 1)
                self.linear_fc1 = SimpleNamespace(delay_wgrad_compute=True)
                self.linear_fc2 = SimpleNamespace(delay_wgrad_compute=False)
                self.offload_expert_fc1 = False
                self.offload_moe_act = False
                self._fused_ops = None

            def _fused_forward(self, value, tokens_per_expert, probs):
                return value

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.tensor_parallel",
            "megatron.core.transformer",
            "megatron.core.transformer.moe",
            "megatron.core.transformer.moe.experts",
        ]
        module_map = {name: types.ModuleType(name) for name in module_names}
        module_map["megatron"].core = module_map["megatron.core"]
        module_map["megatron.core"].tensor_parallel = module_map["megatron.core.tensor_parallel"]
        module_map["megatron.core"].transformer = module_map["megatron.core.transformer"]
        module_map["megatron.core.transformer"].moe = module_map[
            "megatron.core.transformer.moe"
        ]
        module_map["megatron.core.transformer.moe"].experts = module_map[
            "megatron.core.transformer.moe.experts"
        ]
        module_map["megatron.core.tensor_parallel"].CheckpointWithoutOutput = FakeCheckpoint
        module_map["megatron.core.transformer.moe.experts"].TEGroupedMLP = FakeGroupedMLP

        with patch.dict(sys.modules, module_map), patch.dict(
            "os.environ", {"PROOF_ENABLE_FUSED_TE_MOE_ACT_CHECKPOINT": "1"}, clear=True
        ):
            self.assertTrue(install_fused_te_moe_act_checkpoint_patch())
            with self.assertRaisesRegex(RuntimeError, "delayed expert weight-gradient"):
                FakeGroupedMLP()._fused_forward(
                    torch.ones(1, 1, dtype=torch.bfloat16), torch.tensor([1]), torch.ones(1)
                )


if __name__ == "__main__":
    unittest.main()
