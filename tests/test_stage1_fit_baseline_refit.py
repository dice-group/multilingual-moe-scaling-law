from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.contracts import (
    BASELINES,
    FOLDS,
    BaselineRefitError,
    build_jobs,
    gate_path,
    load_profile,
    seal,
    workflow_path,
)
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.gates import (
    GATES,
    execute_gate,
    execute_job,
)
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.reporting import (
    _paired_baseline_minus_m2,
    build_comparison_report,
    build_reproduction_report,
    strict_baseline_decision,
)
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.scheduler import (
    inspect,
    status,
    submit,
)
from multilingual_megatron.experiments.scaling.stage1_fit.execution import read_json, write_json_once


class BaselineRefitWorkflowTest(unittest.TestCase):
    def _profile(self, root: Path) -> tuple[Path, Path]:
        source = root / "source"
        input_root = source / "input"
        input_root.mkdir(parents=True)
        fit_rows = input_root / "fit_rows.parquet"
        fit_rows.write_bytes(b"sealed-fixture")
        input_sha = "a" * 64
        input_receipt = input_root / "input_receipt.json"
        write_json_once(input_receipt, {
            "fit_rows_path": str(fit_rows),
            "fit_rows_content_sha256": input_sha,
        })
        primary = []
        for index, fold in enumerate(FOLDS):
            primary.append({
                "fold_id": fold,
                "role": "fixture",
                "train_trajectory_ids": [f"train-{index}"],
                "test_trajectory_ids": [f"test-{index}"],
                "score_languages": ["eng"],
                "excluded_test_languages": [],
                "membership_sha256": "fixture",
            })
        holdouts = input_root / "holdouts.json"
        write_json_once(holdouts, seal({
            "input_sha256": input_sha,
            "primary_folds": primary,
        }, "manifest_sha256"))
        historical = source / "candidate_selection.json"
        write_json_once(historical, seal({
            "schema_version": "stage1_fit_candidate_report/v5",
            "selection": {"selected_model": "M2"},
            "source_results": [],
        }, "report_sha256"))
        output = root / "output"
        profile = root / "profile.json"
        profile.write_text(json.dumps({
            "schema_version": "stage1_fit_baseline_refit_profile/v1",
            "output_root": str(output),
            "mode": "slurm_array",
            "python_executable": sys.executable,
            "chunk_size": 12,
            "array_concurrency": 20000,
            "blas_threads": 1,
            "memory": "2G",
            "time_limit": "01:00:00",
            "m2_time_limit": "03:00:00",
            "gate_time_limit": "01:00:00",
            "partition": "normal",
            "account": "fixture",
            "reservation": None,
            "inputs": {
                "input_receipt": {"path": str(input_receipt), "sha256": self._sha(input_receipt)},
                "holdouts": {"path": str(holdouts), "sha256": self._sha(holdouts)},
                "historical_candidate_report": {"path": str(historical), "sha256": self._sha(historical)},
            },
        }, indent=2) + "\n", encoding="utf-8")
        return profile, output

    @staticmethod
    def _sha(path: Path) -> str:
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_inspect_is_read_only_and_reports_exact_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile, output = self._profile(Path(temporary))
            result = inspect(profile)
            self.assertEqual(result["registered_optimizer_starts"], 199_530)
            self.assertEqual(result["array_tasks"], 16_632)
            self.assertEqual(result["array_jobs"], 18)
            self.assertEqual(result["gate_jobs"], 5)
            self.assertEqual(result["slurm_jobs"], 23)
            self.assertEqual(result["largest_array"], 4_921)
            self.assertFalse(result["submitted"])
            self.assertFalse(output.exists())

    def test_static_dag_submits_once_and_partial_rerun_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_path, output = self._profile(Path(temporary))
            calls: list[tuple[Path, str | None]] = []

            def fake_submitter(script: Path, dependency: str | None) -> str:
                calls.append((script, dependency))
                return str(1000 + len(calls) - 1)

            result = submit(profile_path, submitter=fake_submitter, require_committed=False)
            self.assertEqual(result["submitted_jobs"], 23)
            self.assertEqual(result["final_job_id"], "1022")
            self.assertEqual(len(calls), 23)
            self.assertEqual(calls[3][1], "afterany:1000:1001:1002")
            self.assertEqual(calls[4][1], "afterok:1003")
            self.assertEqual(calls[-1][1], "afterany:1019:1020:1021:1014")
            scripts = [path.read_text(encoding="utf-8") for path, _ in calls]
            self.assertTrue(any("#SBATCH --array=0-4920%20000" in text for text in scripts))
            self.assertTrue(all("stage1_fit.baseline_refit execute" in text for text in scripts))
            for script in scripts:
                syntax = subprocess.run(["bash", "-n"], input=script, text=True,
                    capture_output=True, check=False)
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
            self.assertTrue(workflow_path(output).is_file())

            again = submit(profile_path, submitter=fake_submitter, require_committed=False)
            self.assertEqual(again["final_job_id"], "1022")
            self.assertEqual(len(calls), 23)

            with patch(
                "multilingual_megatron.experiments.scaling.stage1_fit."
                "baseline_refit_workflow.scheduler._scheduler_states",
                return_value=({str(value): "PENDING" for value in range(1000, 1023)}, None),
            ):
                observed = status(profile_path)
            self.assertEqual(observed["status"], "running")
            self.assertEqual(observed["submitted_jobs"], 23)
            failed_states = {str(value): "PENDING" for value in range(1000, 1023)}
            failed_states["1000"] = "FAILED"
            with patch(
                "multilingual_megatron.experiments.scaling.stage1_fit."
                "baseline_refit_workflow.scheduler._scheduler_states",
                return_value=(failed_states, None),
            ):
                failed = status(profile_path)
            self.assertEqual(failed["status"], "scheduler_failed")
            self.assertEqual(failed["failed_jobs"], ["grid.m0_ols.largest_N_act"])

    def test_mid_submission_failure_resumes_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_path, _ = self._profile(Path(temporary))
            successful: list[str] = []
            attempts = 0

            def flaky(_script: Path, _dependency: str | None) -> str:
                nonlocal attempts
                attempts += 1
                if attempts == 6:
                    raise BaselineRefitError("scheduler unavailable")
                job_id = str(3000 + len(successful))
                successful.append(job_id)
                return job_id

            with self.assertRaisesRegex(BaselineRefitError, "scheduler unavailable"):
                submit(profile_path, submitter=flaky, require_committed=False)
            self.assertEqual(len(successful), 5)
            completed = submit(profile_path, submitter=flaky, require_committed=False)
            self.assertEqual(completed["submitted_jobs"], 23)
            self.assertEqual(len(successful), 23)
            self.assertEqual(attempts, 24)

    def test_array_worker_resolves_deferred_manifest_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_path, output = self._profile(Path(temporary))
            counter = iter(range(2000, 2023))
            submit(profile_path, submitter=lambda _script, _dependency: str(next(counter)),
                require_committed=False)
            manifest_path = workflow_path(output)
            with patch(
                "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.gates.execute_task",
                side_effect=lambda manifest, index: {
                    "model": manifest["fit"]["model"],
                    "fold": manifest["fit"]["fold_id"],
                    "task_index": index,
                },
            ):
                result = execute_job(manifest_path, "grid.m0_ols.E32", task_index=2)
            self.assertEqual(result, {"model": "M0", "fold": "E32", "task_index": 2})

    def test_submit_refuses_nested_slurm_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile, output = self._profile(Path(temporary))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), self.assertRaisesRegex(
                BaselineRefitError, "nested Slurm submission",
            ):
                submit(profile, submitter=lambda _script, _dependency: "1", require_committed=False)
            self.assertFalse(output.exists())

    def test_gate_receipts_are_complete_or_immutable_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = {"workflow_sha256": "f" * 64, "output_root": str(root)}
            with patch.dict(GATES, {"gate.scale": lambda _workflow: {"ok": True}}):
                complete = execute_gate(workflow, "gate.scale")
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(execute_gate(workflow, "gate.scale"), complete)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = {"workflow_sha256": "e" * 64, "output_root": str(root)}

            def fail(_workflow):
                raise ValueError("missing chunk")

            with patch.dict(GATES, {"gate.scale": fail}), self.assertRaisesRegex(ValueError, "missing chunk"):
                execute_gate(workflow, "gate.scale")
            blocked = read_json(gate_path(workflow, "gate.scale"))
            self.assertEqual(blocked["status"], "blocked")
            with self.assertRaisesRegex(BaselineRefitError, "already blocked"):
                execute_gate(workflow, "gate.scale")

    def test_strict_win_requires_every_fold_and_paired_support(self) -> None:
        fold_wins = {baseline: {fold: True for fold in FOLDS} for baseline in BASELINES}
        pairwise = {baseline: {"paired_support": True} for baseline in BASELINES}
        self.assertTrue(strict_baseline_decision(fold_wins, pairwise))
        fold_wins["J0"]["K200_common_languages"] = False
        self.assertFalse(strict_baseline_decision(fold_wins, pairwise))
        fold_wins["J0"]["K200_common_languages"] = True
        pairwise["C0"]["paired_support"] = False
        self.assertFalse(strict_baseline_decision(fold_wins, pairwise))

    def test_pairing_uses_shared_trajectory_bootstrap(self) -> None:
        errors = {
            "M2": {fold: {"shared": 0.01, f"m2-{fold}": 0.01} for fold in FOLDS},
            "C0": {fold: {"shared": 0.04, f"m2-{fold}": 0.04} for fold in FOLDS},
        }
        result = _paired_baseline_minus_m2(errors, "C0")
        self.assertGreater(result["bootstrap_difference_percentiles_10_90"][0], 0.0)
        self.assertTrue(result["paired_support"])

    def test_profile_produces_exact_job_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile_path, _ = self._profile(Path(temporary))
            profile = load_profile(profile_path)
            jobs = build_jobs(profile)
            self.assertEqual(len(jobs), 23)
            self.assertEqual(sum(job["kind"] == "array" for job in jobs), 18)
            self.assertEqual(jobs[-1]["job_id"], "gate.final")
            self.assertEqual(jobs[-1]["dependency_mode"], "afterany")

    def test_comparison_report_exercises_strict_win_contract(self) -> None:
        workflow = {
            "workflow_sha256": "f" * 64,
            "output_root": "/tmp/baseline-refit-report-fixture",
            "inputs": {
                "input_receipt": {"path": "/tmp/input.json"},
                "holdouts": {"path": "/tmp/holdouts.json"},
            },
            "claim_rule": {"fold_rule": "strict", "paired_rule": "strict"},
        }
        results = {
            model: {fold: Path(f"/{model}-{fold}.json") for fold in FOLDS}
            for model in ("C0", "M0", "J0", "M1", "M2")
        }
        scores = {"C0": .30, "M0": .25, "J0": .20, "M1": .15, "M2": .10}

        def scientific(path: Path):
            model, fold = path.name.removesuffix(".json").split("-", 1)
            return {
                "manifest_sha256": (model + fold).encode().hex().ljust(64, "0")[:64],
                "result_sha256": (fold + model).encode().hex().ljust(64, "1")[:64],
                "fit_identity": {"model": model, "fold_id": fold, "objective": "huber"},
                "frozen_delta": .1,
                "parents": [{"kind": "scale_m0_ols", "path": f"/{fold}-scale.json",
                    "file_sha256": "a" * 64, "result_sha256": "b" * 64}],
                "selected": {
                    "coordinates": [0.1, 0.2],
                    "language_coefficients": [{"language": "eng", "L_inf": 1.0, "A": .1, "B": .1}],
                },
            }

        def metrics(_frame, membership, _selected, model):
            value = scores[model]
            return {
                "trajectory_mse": {f"trajectory-{membership['fold']}": value * value},
                "trajectory_balanced_rmse": value,
                "trajectory_balanced_r2": 1.0 - value,
                "rows": 1,
                "trajectories": 1,
            }

        candidate = {"schema_version": "stage1_fit_candidate_report/v5", "report_sha256": "c" * 64}
        with patch.multiple(
            "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.reporting",
            _snapshot=lambda _path: (
                object(), {"fit_rows_content_sha256": "d" * 64},
                {"path": "/tmp/input.json", "file_sha256": "e" * 64},
            ),
            _holdout=lambda _path, _sha: {"manifest_sha256": "f" * 64},
            bind_membership=lambda _path, **kwargs: {"fold": kwargs["membership_id"]},
            scientific_result=scientific,
            _validate_result=lambda **_kwargs: None,
            _metrics=metrics,
            sha256_file=lambda _path: "1" * 64,
            read_json=lambda _path: candidate,
            verify_seal=lambda *_args, **_kwargs: "c" * 64,
            _report_binding=lambda path: {"path": str(path), "file_sha256": "2" * 64,
                "embedded_sha256": "3" * 64},
        ):
            report = build_comparison_report(
                workflow,
                candidate_report_path=Path("/tmp/candidate.json"),
                reproduction_report_path=Path("/tmp/reproduction.json"),
                results=results,
            )
        self.assertTrue(report["strict_baseline_win"])
        self.assertTrue(all(
            value > 0.0
            for folds in report["fold_percent_improvement_vs_m2"].values()
            for value in folds.values()
        ))

    def test_reproduction_report_checks_predictions_and_selected_model(self) -> None:
        historical_path = Path("/tmp/historical-candidate.json")
        workflow = {
            "workflow_sha256": "f" * 64,
            "inputs": {
                "historical_candidate_report": {"path": str(historical_path)},
                "input_receipt": {"path": "/tmp/input.json"},
                "holdouts": {"path": "/tmp/holdouts.json"},
            },
        }
        fresh = {model: {fold: Path(f"/fresh-{model}-{fold}.json") for fold in FOLDS}
            for model in ("M0", "M1", "M2")}
        sources = [{"model": model, "fold_id": fold,
            "path": f"/old-{model}-{fold}.json", "sha256": f"old-{model}-{fold}"}
            for model in ("M0", "M1", "M2") for fold in FOLDS]
        historical = {
            "schema_version": "stage1_fit_candidate_report/v5",
            "report_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "holdout_manifest_sha256": "c" * 64,
            "selection": {"selected_model": "M2"},
            "source_results": sources,
        }

        def scientific(path: Path):
            name = path.name.removesuffix(".json")
            old = name.startswith("old-")
            model, fold = name.removeprefix("old-").removeprefix("fresh-").split("-", 1)
            return {
                "result_sha256": f"old-{model}-{fold}" if old else f"fresh-{model}-{fold}",
                "fit_identity": {"model": model, "fold_id": fold},
                "parents": [{"kind": "scale_m0_ols"}],
                "selected": {"objective": 1.0, "marker": name},
            }

        rows = pd.DataFrame({"readout_id": ["r"], "language": ["eng"]})
        with patch.multiple(
            "multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.reporting",
            read_json=lambda _path: historical,
            verify_seal=lambda *_args, **_kwargs: "a" * 64,
            _snapshot=lambda _path: (object(), {"fit_rows_content_sha256": "b" * 64}, {}),
            _holdout=lambda _path, _sha: {"manifest_sha256": "c" * 64},
            bind_membership=lambda _path, **_kwargs: {},
            scientific_result=scientific,
            _validate_result=lambda **_kwargs: None,
            _prediction=lambda _frame, _membership, _selected, _model: (rows, np.asarray([1.0])),
            sha256_file=lambda _path: "d" * 64,
        ):
            report = build_reproduction_report(
                workflow,
                fresh_candidate={"selection": {"selected_model": "M2"}},
                fresh_results=fresh,
            )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["checks"]), 9)


if __name__ == "__main__":
    unittest.main()
