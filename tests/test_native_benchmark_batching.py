"""CPU checks for request coverage and bounded native batch selection."""

import unittest
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_batching import (
    AutoBatchPolicy, calibrate_native_batches, length_buckets, score_batch_ranges, select_batch_size,
)


def measurement(seconds=1.0, peak=0.4, projected=0.5, difference=0.0):
    return {"seconds": seconds, "peak_memory_fraction": peak,
            "projected_double_memory_fraction": projected,
            "score_abs_difference": difference}


class NativeBatchingTest(unittest.TestCase):
    def test_minimum_width_preserves_real_tokens_targets_and_routing_mask(self):
        import torch
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import (
            build_score_batch, prepare_scored_sequence,
        )
        sequence = prepare_scored_sequence(request_index=7, request_id="short",
            context_ids=[2, 3], continuation_ids=[4, 5], max_length=128)
        plain = build_score_batch([sequence], pad_token_id=9)
        padded = build_score_batch([sequence], pad_token_id=9, minimum_sequence_width=64)
        width = plain["tokens"].shape[1]
        self.assertEqual(padded["tokens"].shape, (1, 64))
        for name in ("tokens", "labels", "position_ids", "loss_mask", "input_valid_mask"):
            self.assertTrue(torch.equal(plain[name], padded[name][:, :width]))
        self.assertEqual(padded["loss_mask"].sum().item(), sequence.continuation_tokens)
        self.assertFalse(padded["input_valid_mask"][:, width:].any())
        self.assertFalse(padded["loss_mask"][:, width:].any())
        self.assertTrue((padded["tokens"][:, width:] == 9).all())

    def test_width_alignment_preserves_targets_and_masks(self):
        import torch
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import (
            build_score_batch, prepare_scored_sequence,
        )
        for length in (1, 7, 8, 9, 127, 2047, 2048):
            sequence = prepare_scored_sequence(request_index=0, request_id="aligned",
                context_ids=[2] * length, continuation_ids=[4], max_length=2048)
            plain = build_score_batch([sequence], pad_token_id=9)
            padded = build_score_batch([sequence], pad_token_id=9, sequence_width_multiple=8)
            width = plain["tokens"].shape[1]
            self.assertEqual(padded["tokens"].shape[1], ((width + 7) // 8) * 8)
            for name in ("tokens", "labels", "loss_mask", "position_ids", "input_valid_mask"):
                self.assertTrue(torch.equal(plain[name], padded[name][:, :width]))
            self.assertFalse(padded["loss_mask"][:, width:].any())
            self.assertFalse(padded["input_valid_mask"][:, width:].any())
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_score_batch([sequence], pad_token_id=9, sequence_width_multiple=value)

    def test_auto_is_opt_in_and_policy_rejects_unsafe_or_misspelled_values(self):
        self.assertIsNone(AutoBatchPolicy.from_mapping(None))
        self.assertEqual(AutoBatchPolicy.from_mapping({"mode": "auto"}).max_batch_size, 512)
        for fields in (
            {"mode": "fixed"}, {"unknown": 2}, {"max_batch_size": True},
            {"memory_fraction": 1.0}, {"memory_fraction": float("nan")},
            {"max_calibration_seconds": 0}, {"timed_batches": 1},
            {"minimum_speedup": 0.9}, {"score_abs_tolerance": 0.1},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                AutoBatchPolicy.from_mapping({"mode": "auto", **fields})

    def test_lengths_and_short_tail_batches_cover_every_request_once(self):
        lengths = [1, 8, 64, 65, 66, 127, 128, 129, 2048]
        buckets = length_buckets(lengths)
        self.assertEqual(buckets, [(64, 0, 3), (128, 3, 7), (256, 7, 8), (2048, 8, 9)])
        schedule = [{"start": start, "stop": stop, "batch_size": batch}
                    for (_, start, stop), batch in zip(buckets, (2, 3, 1, 1))]
        ranges = list(score_batch_ranges(len(lengths), 100, schedule))
        self.assertEqual([i for a, b in ranges for i in range(a, b)], list(range(len(lengths))))
        self.assertEqual(ranges, [(0, 2), (2, 3), (3, 6), (6, 7), (7, 8), (8, 9)])
        self.assertEqual(list(score_batch_ranges(9, 4)), [(0, 4), (4, 8), (8, 9)])
        with self.assertRaises(ValueError):
            length_buckets([100, 80])

    def test_bad_schedule_fails_before_first_forward(self):
        for schedule in (
            [{"start": 0, "stop": 3, "batch_size": 2}, {"start": 4, "stop": 5, "batch_size": 1}],
            [{"start": 0, "stop": 3, "batch_size": 2}, {"start": 2, "stop": 5, "batch_size": 1}],
            [{"start": 0, "stop": 4, "batch_size": 2}],
            [{"start": 0, "stop": 6, "batch_size": 2}],
        ):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                next(score_batch_ranges(5, 2, schedule))

    def test_fastest_batch_wins_and_memory_projection_stops_growth(self):
        calls, receipts = [], []
        def probe(batch):
            calls.append(batch)
            return measurement(seconds={1: 1.0, 2: 1.1, 4: 2.3, 8: 5.0}[batch],
                               projected=0.9 if batch == 8 else 0.6)
        selected = select_batch_size(AutoBatchPolicy(), 512, probe, receipts.append)
        self.assertEqual(selected, 2)
        self.assertEqual(calls, [1, 2, 4, 8])
        self.assertEqual(sum(row["status"] == "measured" for row in receipts), 4)

    def test_tiny_speedup_keeps_smaller_batch(self):
        selected = select_batch_size(AutoBatchPolicy(max_batch_size=2), 10,
            lambda batch: measurement(seconds=1.0 if batch == 1 else 1.99), lambda row: None)
        self.assertEqual(selected, 1)

    def test_over_limit_probe_reverts_to_previously_measured_batch(self):
        selected = select_batch_size(AutoBatchPolicy(), 100,
            lambda batch: measurement(peak=0.4 if batch == 1 else 0.85), lambda row: None)
        self.assertEqual(selected, 1)
        with self.assertRaisesRegex(RuntimeError, "no batch fits"):
            select_batch_size(AutoBatchPolicy(), 100,
                lambda batch: measurement(peak=0.9), lambda row: None)

    def test_score_drift_and_nonfinite_timing_abort(self):
        for result in (measurement(difference=0.02), measurement(seconds=float("nan")),
                       measurement(seconds=0)):
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                select_batch_size(AutoBatchPolicy(), 16, lambda batch: result, lambda row: None)

    def test_score_drift_rejects_larger_candidate_without_relaxing_tolerance(self):
        trials = []
        selected = select_batch_size(AutoBatchPolicy(), 100,
            lambda batch: measurement(difference=0.03 if batch >= 4 else 0), trials.append)
        self.assertEqual(selected, 2)
        self.assertFalse(trials[-1]["parity_ok"])
        self.assertFalse(trials[-1]["selected"])

    def test_oom_is_not_retried_on_one_rank(self):
        calls = []
        def probe(batch):
            calls.append(batch)
            raise RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            select_batch_size(AutoBatchPolicy(), 100, probe, lambda row: None)
        self.assertEqual(calls, [1])

    def test_small_bucket_never_probes_more_requests_than_it_has(self):
        calls = []
        def probe(batch):
            calls.append(batch)
            return measurement()
        select_batch_size(AutoBatchPolicy(), 3, probe, lambda row: None)
        self.assertEqual(calls, [1, 2])

    def test_calibration_can_start_from_the_configured_fixed_control(self):
        calls = []
        def probe(batch):
            calls.append(batch)
            return measurement()
        selected = select_batch_size(AutoBatchPolicy(max_batch_size=32), 100, probe,
                                     lambda row: None, initial_batch_size=8)
        self.assertEqual(calls, [8, 16, 32])
        self.assertEqual(selected, 32)

    def test_calibration_receipt_and_score_gate_with_cpu_forward(self):
        import json
        import torch
        from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import (
            prepare_scored_sequence,
        )
        sequences = [prepare_scored_sequence(
            request_index=i, request_id=str(i), context_ids=[2] * length,
            continuation_ids=[4, 5], max_length=2048,
        ) for i, length in enumerate((2, 3, 4, 65, 66, 67))]
        for drift in (False, True):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                stack.enter_context(patch("torch.distributed.is_initialized", return_value=False))
                for name, value in {"current_device": torch.device("cpu"), "mem_get_info": (900, 1000),
                                    "memory_reserved": 80, "memory_allocated": 70,
                                    "max_memory_allocated": 100, "max_memory_reserved": 120,
                                    "synchronize": None, "empty_cache": None,
                                    "reset_peak_memory_stats": None}.items():
                    stack.enter_context(patch(f"torch.cuda.{name}", return_value=value))
                def forward(model, batch):
                    value = batch["labels"].float()
                    if drift and len(value) > 1:
                        value = value + 1
                    return value
                stack.enter_context(patch(
                    "multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring._score_cuda_batch",
                    side_effect=forward,
                ))
                spec = SimpleNamespace(batching=AutoBatchPolicy(max_batch_size=2), max_gpu_hours=1,
                                       parity_requests=2, pad_token_id=3, output_dir=Path(folder),
                                       micro_batch_size=1, minimum_sequence_width=64)
                receipt = calibrate_native_batches(None, sequences, spec, total_started=time.perf_counter())
                ranges = list(score_batch_ranges(len(sequences), 1, receipt["buckets"]))
                self.assertEqual([i for a, b in ranges for i in range(a, b)], list(range(6)))
                if drift:
                    self.assertTrue(all(row["batch_size"] == 1 for row in receipt["buckets"]))
                    self.assertTrue(any(row.get("parity_ok") is False for row in receipt["trials"]))
                saved = json.loads((Path(folder) / "batching_calibration.json").read_text())
                self.assertEqual(saved["status"], "calibrated")
                self.assertTrue(saved["trials"])


if __name__ == "__main__":
    unittest.main()
