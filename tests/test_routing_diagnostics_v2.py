"""Count conservation, distributed ownership, and observational-hook contracts."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import math
import torch

from multilingual_megatron.pipeline.stages.final_loss.routing_batch_context import (
    CURRENT_BATCH, RoutingBatch, expected_router_layers, validate_routing_topology,
    wrap_routing_forward_step,
)
from multilingual_megatron.pipeline.stages.final_loss.moe_validation_diagnostics import (
    _record_router_call, _routing_snapshot, collect_moe_validation_diagnostics,
    prepare_moe_validation_diagnostics, reconcile_snapshots, reset_moe_validation_diagnostics,
)


def router():
    return SimpleNamespace(layer_number=1, topk=1, score_function="softmax",
        config=SimpleNamespace(num_moe_experts=2, moe_expert_capacity_factor=None))


def observe(mask, ids, choices):
    mask = torch.tensor(mask, dtype=torch.bool)
    choices = torch.tensor(choices)
    logits = torch.nn.functional.one_hot(choices, 2).float() * 3
    route = torch.nn.functional.one_hot(choices, 2).bool().view(-1, 2)
    token = CURRENT_BATCH.set(RoutingBatch(mask, tuple(ids)))
    try:
        _record_router_call(router(), logits, route, None)
    finally:
        CURRENT_BATCH.reset(token)


class RoutingDiagnosticsV2Tests(unittest.TestCase):
    def setUp(self):
        reset_moe_validation_diagnostics()

    def tearDown(self):
        reset_moe_validation_diagnostics()

    def test_sequence_major_partial_and_entire_padding_conservation(self):
        # B=2,S=3; first sample has two real inputs; second is wholly padding.
        observe([[True, True, False], [False, False, False]], [0, 1], [[0, 1], [1, 1], [1, 1]])
        report = collect_moe_validation_diagnostics(expected_num_layers=1, expected_num_experts=2,
            expected_sample_count=2, expected_valid_tokens=2, sequence_length=3)
        assert report["evidence_available"]
        row = report["layers"][0]
        assert row["expert_token_counts"] == [1, 1]
        assert row["padding_expert_token_counts"] == [0, 4]
        assert row["all_expert_token_counts"] == [1, 5]
        assert row["blocks"] == {"0": [1, 1]}


    def test_identical_histograms_on_different_samples_are_summed(self):
        observe([[True, True]], [0], [[0], [1]])
        first = _routing_snapshot()
        second = copy.deepcopy(first)
        second["layers"][0]["sample_ids"] = [1]
        second["layers"][0]["trace"] = {}
        rows, errors = reconcile_snapshots([first, second], expected_sample_count=2)
        assert not errors
        assert rows[0]["expert_token_counts"] == [2, 2]
        _, errors = reconcile_snapshots([first, first], expected_sample_count=2)
        assert "rank_1:overlapping_sample_ownership" in errors
        _, errors = reconcile_snapshots([first], expected_sample_count=2)
        assert "missing_or_unexpected_sample_ids" in errors


    def test_padding_and_batch_partition_invariance(self):
        observe([[True, True]], [0], [[0], [1]])
        base = collect_moe_validation_diagnostics()["layers"][0]
        observe([[True, True], [False, False]], [0, 1], [[0, 1], [1, 1]])
        padded = collect_moe_validation_diagnostics()["layers"][0]
        for key in ("expert_token_counts", "score_sums", "router_z_loss_unweighted", "switch_aux_loss_unweighted"):
            assert torch.allclose(torch.as_tensor(padded[key], dtype=torch.float64), torch.as_tensor(base[key], dtype=torch.float64), rtol=1e-7, atol=1e-9)
        observe([[True, True]], [0], [[0], [1]])
        observe([[False, False]], [1], [[1], [1]])
        separate = collect_moe_validation_diagnostics()["layers"][0]
        for key in ("expert_token_counts", "padding_expert_token_counts", "all_expert_token_counts", "blocks"):
            assert separate[key] == padded[key]


    def test_drop_and_hook_errors_fail_closed(self):
        observe([[True]], [0], [[0]])
        snapshot = _routing_snapshot()
        snapshot["layers"][0]["topk"] = 2
        _, errors = reconcile_snapshots([_routing_snapshot(), snapshot])
        assert "rank_1:routing_structure_differs_from_rank_0" in errors
        snapshot = _routing_snapshot()
        snapshot["hook_errors"] = ["bad mask"]
        assert reconcile_snapshots([snapshot])[1]


    def test_context_lifecycle_and_forward_identity(self):
        batch = {"input_valid_mask": torch.tensor([[True]]), "sample_id": torch.tensor([0]), "tokens": torch.tensor([[0]])}
        output = object()
        def forward(iterator, model, return_schedule_plan=False):
            assert next(iterator) is batch
            _record_router_call(router(), torch.tensor([[[1., 0.]]]), torch.tensor([[True, False]]), None)
            return output
        wrapped = wrap_routing_forward_step(forward, expected_layers=1)
        assert wrapped(iter([batch]), None) is output
        assert CURRENT_BATCH.get() is None
        with self.assertRaisesRegex(ValueError, "synchronous"):
            wrapped(iter([batch]), None, return_schedule_plan=True)
        def fail(iterator, model, return_schedule_plan=False):
            next(iterator)
            raise RuntimeError("forward failure")
        with self.assertRaises(RuntimeError):
            wrap_routing_forward_step(fail, expected_layers=1)(iter([batch]), None)
        assert CURRENT_BATCH.get() is None
        with self.assertRaisesRegex(ValueError, "missing explicit"):
            _record_router_call(router(), torch.tensor([[[1., 0.]]]), torch.tensor([[True, False]]), None)


    def test_unsupported_topology(self):
        validate_routing_topology(SimpleNamespace(cuda_graph_scope=["attn"]))
        for settings in ({"tensor_model_parallel_size": 2}, {"context_parallel_size": 2},
                         {"pipeline_model_parallel_size": 2}, {"sequence_parallel": True}, {"cuda_graph_scope": "full"}):
            with self.assertRaises(ValueError):
                validate_routing_topology(SimpleNamespace(**settings))

    def test_explicit_replication_is_verified(self):
        observe([[True, True]], [0], [[0], [1]])
        snapshot = _routing_snapshot()
        rows, errors = reconcile_snapshots([snapshot, snapshot], expected_sample_count=1, ownership_mode="replicated")
        self.assertFalse(errors)
        self.assertEqual(rows[0]["expert_token_counts"], [1, 1])
        changed = copy.deepcopy(snapshot)
        changed["layers"][0]["trace"] = {}
        self.assertTrue(reconcile_snapshots([snapshot, changed], ownership_mode="replicated")[1])

    def test_actual_dropped_assignment_is_retained(self):
        context = RoutingBatch(torch.tensor([[True, True]]), (0,))
        token = CURRENT_BATCH.set(context)
        try:
            _record_router_call(router(), torch.tensor([[[3., 0.]], [[0., 3.]]]),
                torch.tensor([[True, False], [False, False]]), None)
        finally:
            CURRENT_BATCH.reset(token)
        report = collect_moe_validation_diagnostics(expected_valid_tokens=2)
        self.assertTrue(report["evidence_available"])
        self.assertEqual(report["totals"]["dropped_assignments"], 1)

    def test_duplicate_samples_and_missing_layers_fail(self):
        observe([[True]], [0], [[0]])
        with self.assertRaisesRegex(ValueError, "more than once"):
            observe([[True]], [0], [[0]])
        report = collect_moe_validation_diagnostics(expected_num_layers=2)
        self.assertFalse(report["evidence_available"])

    def test_stage1_decoder_layers(self):
        expected = frozenset(range(8))
        for frequency in (1, [1] * 8):
            config = SimpleNamespace(num_layers=8, moe_layer_freq=frequency)
            self.assertEqual(expected_router_layers(config), expected)
        batch = {"input_valid_mask": torch.tensor([[True]]), "sample_id": torch.tensor([0])}

        def forward(iterator, model, return_schedule_plan=False):
            next(iterator)
            for index in expected:
                layer_router = router()
                layer_router.layer_number = index + 1
                _record_router_call(layer_router, torch.tensor([[[1., 0.]]]),
                                    torch.tensor([[True, False]]), None)
            return "main loss"

        wrapped = wrap_routing_forward_step(forward, expected_layers=expected)
        self.assertEqual(wrapped(iter([batch]), None), "main loss")
        report = collect_moe_validation_diagnostics(expected_layer_indices=expected,
                                                    expected_valid_tokens=1)
        self.assertTrue(report["evidence_available"], report)
        wrapped(iter([batch]), None)
        wrong_layers = collect_moe_validation_diagnostics(expected_layer_indices={0, 1, 2})
        self.assertFalse(wrong_layers["evidence_available"])

    def test_non_stage1_decoder_layouts_are_rejected(self):
        for layout in ({"hybrid_layer_pattern": "MEME"},
                       {"moe_layer_freq": [0, 1, 0, 1]}, {"moe_layer_freq": 2}):
            with self.assertRaises(ValueError):
                expected_router_layers(SimpleNamespace(num_layers=4, **layout))
