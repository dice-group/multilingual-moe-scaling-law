from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from result_analysis.scaling_law_fit import residuals as a
from multilingual_megatron.experiments.scaling.stage1_fit import extrapolation_diagnostics as d
from multilingual_megatron.experiments.scaling.stage1_fit import execution, fit
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.contracts import seal
from multilingual_megatron.experiments.scaling.stage1_fit.data import sha256_file


def fixture() -> pd.DataFrame:
    rows = []
    for model, error in [("J0", -.2), ("M2", -.1), ("M6", -.05)]:
        for ladder in ("A", "D"):
            for k in (1, 2):
                for language in range(k):
                    for tokens in (1e9, 2e9, 4e9):
                        observed = -np.log(tokens / 1e9) - (ladder == "D")
                        rows.append({"trajectory_id": f"{ladder}:K{k}", "readout_id": f"{ladder}:K{k}:D{tokens}",
                            "language": str(language), "ladder": ladder, "N_act": 1e8 if ladder == "A" else 1e9,
                            "N_act_nonembedding": 8e7 if ladder == "A" else 8e8, "E": 1, "K": k,
                            "D_total": tokens, "D_t": tokens / k, "adjusted_loss": observed,
                            "prediction": observed + error, "model": model})
    parts = []
    for (model, ladder), group in pd.DataFrame(rows).groupby(["model", "ladder"]):
        parts.append(d.annotate(group, model=model, fold="largest_N_act", split="train" if ladder == "A" else "test", delta=.15))
    return pd.concat(parts, ignore_index=True)


class ResidualAnalysisTest(unittest.TestCase):
    def test_sign_weight_and_influence(self):
        rows = fixture()
        part = rows[rows.model.eq("J0") & rows.split.eq("test")]
        self.assertAlmostEqual(part.weight.sum(), 1.)
        self.assertTrue((part.residual < 0).all())
        self.assertAlmostEqual(a.summary(part)["bias"], -.2)
        self.assertAlmostEqual(a.summary(part)["mean_huber_influence_ratio"], .75)
        by_trajectory = part.groupby("trajectory_id").weight.sum()
        self.assertTrue(np.allclose(by_trajectory, .5))

    def test_score_reproduction_and_support_failure(self):
        rows = fixture(); rows = rows[rows.model.eq("M6") & rows.split.eq("test")]
        expected = {"trajectory_mse": rows.groupby("trajectory_id").squared_error.mean().to_dict(),
            "rows": len(rows), "trajectory_balanced_rmse": .05}
        self.assertLess(d.reproduce_score(rows, expected), 1e-12)
        changed = copy.deepcopy(expected); changed["trajectory_balanced_rmse"] += 2e-12
        with self.assertRaisesRegex(ValueError, "reproduction"):
            d.reproduce_score(rows, changed)
        changed = copy.deepcopy(expected); changed["trajectory_mse"]["unexpected"] = .1
        with self.assertRaisesRegex(ValueError, "membership"):
            d.reproduce_score(rows, changed)

    def test_mse_shares_partition_without_changing_scoring(self):
        groups = a.residual_groups(fixture())
        horizon = groups[groups.axis.eq("D_total")]
        self.assertTrue(np.allclose(horizon.groupby(["model", "split"]).mse_share.sum(), 1.))

    def test_exact_matching_reports_missing_horizon(self):
        frame = fixture()
        frame.loc[frame.ladder.eq("D") & frame.D_total.eq(4e9), "D_total"] += 1
        matches = a.matched_sizes(frame, replicates=100, seed=3)
        self.assertEqual(set(matches[matches.matched_rows.eq(0)].D_total), {4e9, 4e9 + 1})
        self.assertTrue(matches[matches.matched_rows.gt(0)].right_minus_left_bias.abs().lt(1e-12).all())

    def test_duplicate_matching_is_rejected(self):
        frame = fixture()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            a.matched_sizes(pd.concat([frame, frame.iloc[:1]]), replicates=100, seed=3)

    def test_slopes_are_exact_log_token_secants(self):
        slopes = a.learning_slopes(fixture())
        self.assertTrue(np.allclose(slopes.observed_slope, -1.))
        self.assertTrue(np.allclose(slopes.slope_error, 0., atol=1e-12))

    def test_bootstrap_is_deterministic_and_trajectory_based(self):
        frame = fixture()
        frame = frame[frame.split.eq("test")]
        frame = pd.concat([frame.assign(fold=fold) for fold in a.FOLDS], ignore_index=True)
        first = a.pairwise_bootstrap(frame, replicates=100, seed=9)
        second = a.pairwise_bootstrap(pd.concat([frame, frame], ignore_index=True), replicates=100, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_equal_fold_rmse"]["M6"], .05)
        self.assertLess(first["comparisons"][1]["p975"], 0.)


class EvidenceReaderTest(unittest.TestCase):
    def test_binding_and_seal_reject_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.json"
            execution.write_json_once(path, seal({"status": "complete"}, "report_sha256"))
            binding = {"path": str(path), "file_sha256": sha256_file(path)}
            self.assertEqual(d.bound_file(binding), path)
            binding["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "changed"):
                d.bound_file(binding)
            with patch.object(d.execution, "read_json", return_value={"status": "changed", "report_sha256": "0" * 64}):
                with self.assertRaises(RuntimeError):
                    d.sealed_json(path, "report_sha256")

    def test_alignment_accepts_row_permutation_not_value_changes(self):
        base = fixture().iloc[:3][d.KEYS + ["D_t", "D_total"]].copy()
        component = base.assign(D_family=0., D_script=0., D_other=base.D_total-base.D_t)
        self.assertEqual(len(d.aligned_components(base, component.iloc[::-1])), 3)
        component.loc[component.index[0], "D_t"] += 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            d.aligned_components(base, component)

    def test_invalid_exposure_partition_rejected(self):
        base = fixture().iloc[:3][d.KEYS + ["D_t", "D_total"]].copy()
        component = base.assign(D_family=10., D_script=0., D_other=base.D_total-base.D_t)
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            d.aligned_components(base, component)

    def test_failed_duplicate_or_wrong_result_audit_rejected(self):
        item = {"model": "M6", "lambda_key": "0.01", "fold": "largest_N_act", "status": "healthy", "base_result_sha256": "result"}
        result = {"result_sha256": "result"}
        self.assertEqual(d.validate_m6_audit({"items": [item]}, result, "largest_N_act"), item)
        for items in [[dict(item, status="failed")], [item, item], [dict(item, base_result_sha256="other")]]:
            with self.assertRaisesRegex(ValueError, "healthy numerical audit"):
                d.validate_m6_audit({"items": items}, result, "largest_N_act")

    def test_training_keeps_non_scored_languages(self):
        rows = fixture().iloc[:1].copy()
        rows = pd.concat([rows.assign(language="a"), rows.assign(language="b")], ignore_index=True)
        rows["D_family"] = 0.; rows["D_script"] = 0.; rows["D_other"] = 0.
        member = {"train_trajectory_ids": list(rows.trajectory_id.unique()), "train_readout_ids": [], "score_languages": ["a"]}
        selected = {"shared_parameters": fit.SharedParameters(.3, 0., .3, 0.).as_dict(),
            "language_coefficients": [{"language": name, "L_inf": -5., "A": 1., "B": 1.} for name in ["a", "b"]]}
        for model in ["M2", "M6"]:
            self.assertEqual(set(d.predict_rows(rows, member, selected, model, "train").language), {"a", "b"})

    def test_parent_future_and_branch_future_are_excluded(self):
        table = pd.DataFrame({"consumed_tokens": [100., 200., 300., 400.]})
        prefix, branch = d.prefix_and_branch({"parent": table, "branch": table}, table.iloc[:0],
            job_id="branch", parent_id="parent", branch_start=200., readout_end=300.)
        self.assertEqual(list(prefix.consumed_tokens), [100., 200.])
        self.assertEqual(list(branch.consumed_tokens), [100., 200., 300.])
        prefix, _ = d.prefix_and_branch({"parent": table}, table.iloc[:0],
            job_id="missing", parent_id="parent", branch_start=np.nan, readout_end=300.)
        self.assertTrue(prefix.empty)

    def test_missing_telemetry_is_disclosed_not_imputed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "release.json"
            execution.write_json_once(path, {"table_inventory": {}})
            readouts = pd.DataFrame([{"readout_id": "r", "trajectory_id": "t", "ladder": "D", "experts": 1,
                "language_count": 1, "actual_tokens": 1000, "training_job_id": "missing"}])
            with patch.object(d, "release_table", return_value=readouts):
                table, coverage = d.training_evidence(path)
            self.assertEqual(coverage["missing_telemetry"], 1)
            self.assertEqual(coverage["missing_configuration"], 1)
            self.assertTrue(pd.isna(table.iloc[0].observed_max_nan_counter))

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "not_created"
            with patch.object(d, "load_profile", return_value=({"output_root": str(output)}, {})), \
                    patch.object(d, "collect_predictions", return_value=(fixture(), {"status": "complete"}, {})):
                result = d.run(Path(temp) / "profile", dry_run=True)
            self.assertEqual(result["status"], "validated_dry_run")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
