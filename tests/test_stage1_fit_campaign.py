from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit.campaign import (CampaignError, _equivalent_to_retained,
    audit_grid, audit_path, plan_candidate_campaign, plan_d16_refits, plan_generalization_comparison, plan_literature_comparison,
    plan_secondary_diagnostic, scientific_result)
from multilingual_megatron.experiments.scaling.stage1_fit.cli import _prepare, parser as cli_parser
from multilingual_megatron.experiments.scaling.stage1_fit.data import (INPUT_RECEIPT_SCHEMA_VERSION, FitRow,
    canonical_json_sha256, fit_rows_content_sha256, sha256_file)
from multilingual_megatron.experiments.scaling.stage1_fit.execution import (
    ExecutionError, bind_membership, build_manifest, load_manifest, load_problem, result_path,
    read_json, run_serial, source_sha256, validate_manifest, write_json_once, write_manifest,
)
from multilingual_megatron.experiments.scaling.stage1_fit.fit import OuterBound, d16_adaptive_starts, initial_bounds
from multilingual_megatron.experiments.scaling.stage1_fit.report import (ReportError, build_candidate_report,
    LOGICAL_COMPUTE_CONTRACT_SCHEMA, _d16_adaptive_history, build_d16_report, build_final_report,
    build_generalization_report, build_literature_comparison_report,
    build_secondary_diagnostics_report, write_final_report, write_secondary_diagnostics_report)
from multilingual_megatron.experiments.scaling.stage1_fit.selection import (LARGEST_COMPUTE_VIEW,
    build_generalization_holdout_manifest, build_holdout_manifest, build_resampling_manifest)


def _rows() -> list[FitRow]:
    configurations = (("t0", 1.0e8, 1, 12), ("t1", 1.1e8, 32, 12), ("t2", 1.2e8, 1, 200), ("t3", 2.0e8, 1, 12))
    rows: list[FitRow] = []
    for number, (trajectory, n_act, experts, languages) in enumerate(configurations):
        for readout in (1, 2, 3, 4, 5):
            rows.append(FitRow(trajectory_id=trajectory, readout_id=f"{trajectory}:r{readout}", language="eng",
                N_act=n_act, N_act_nonembedding=0.8*n_act, E=experts, K=languages,
                D_total=float(readout*1.0e9), D_t=float(readout*1.0e8), adjusted_loss=1.2 + .04*number - .03*readout,
                raw_nll=8.2, validation_token_count=10, unigram_nll=7.0, ladder="A", model_seed=1))
    return rows


class Stage1FitCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        clean = patch("multilingual_megatron.experiments.scaling.stage1_fit.execution._repository_provenance",
            return_value=("f" * 40, False))
        committed = patch(
            "multilingual_megatron.experiments.scaling.stage1_fit.execution.source_sha256_at_commit",
            return_value=source_sha256(),
        )
        clean.start(); committed.start()
        self.addCleanup(clean.stop); self.addCleanup(committed.stop)

    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
        rows = _rows(); fit_rows = root / "fit_rows.parquet"
        pd.DataFrame([item.to_dict() for item in rows]).to_parquet(fit_rows, index=False)
        content = fit_rows_content_sha256(rows)
        holdouts = root / "holdouts.json"; write_json_once(holdouts, build_holdout_manifest(rows, input_sha256=content,
            random_count=1, lowest_loss_count=2, require_canonical_population=False).to_dict())
        receipt = root / "input_receipt.json"; write_json_once(receipt, {
            "schema_version": INPUT_RECEIPT_SCHEMA_VERSION, "fit_rows_path": str(fit_rows),
            "fit_rows_file_sha256": sha256_file(fit_rows), "fit_rows_content_sha256": content,
            "trajectories": 4, "readouts": 20, "fit_rows": len(rows)})
        profile = {"output_root": str(root / "out"), "chunk_size": 7, "blas_threads": 1, "memory": "1G",
            "time_limit": "00:05:00", "array_concurrency": 2}
        return fit_rows, holdouts, {"path": receipt, "content": content}, profile

    def _generalization_fixture(self, root: Path, holdouts: Path, content: str) -> Path:
        source_payload = read_json(holdouts)
        source = {'path': str(holdouts.resolve()), 'file_sha256': sha256_file(holdouts),
            'manifest_sha256': source_payload['manifest_sha256']}
        manifest = build_generalization_holdout_manifest(_rows(), input_sha256=content,
            source_holdouts=source, require_canonical_population=False)
        path = root / 'generalization_holdouts.json'
        write_json_once(path, manifest)
        return path

    def _completion_receipt(self, root: Path, name: str, manifest_sha256: str) -> tuple[Path, str]:
        unsigned = {"schema_version": "stage1_fit_complete/v2", "manifest_sha256": manifest_sha256,
            "chunk_sha256s": [], "start_count": 0, "successful_starts": 0}
        receipt = {**unsigned, "receipt_sha256": canonical_json_sha256(unsigned)}
        path = root / "receipts" / f"{name}.json"; write_json_once(path, receipt)
        return path, receipt["receipt_sha256"]

    def _candidate_report(self, root: Path, model: str = "M0", input_sha256: str | None = None,
        holdout_manifest_sha256: str | None = None, input_receipt: dict[str, str] | None = None) -> Path:
        unsigned = {"schema_version": "stage1_fit_candidate_report/v5", "selection": {"selected_model": model},
            "input_sha256": input_sha256}
        if holdout_manifest_sha256 is not None: unsigned["holdout_manifest_sha256"] = holdout_manifest_sha256
        if input_receipt is not None: unsigned["input_receipt"] = dict(input_receipt)
        report = {**unsigned, "report_sha256": canonical_json_sha256(unsigned)}
        path = root / "candidate.json"; write_json_once(path, report); return path

    def _fake_scientific_m0(self, *, root: Path, fit_rows: Path, receipt: dict[str, object], profile: dict[str, object],
        membership: dict[str, object], fold_id: str, selection_report: Path | None = None) -> Path:
        manifest = build_manifest(output_root=Path(str(profile["output_root"])), fit_rows_path=fit_rows,
            fit_rows_content_sha256=str(receipt["content"]), phase=f"{fold_id}-m0-ols", model="M0", fold_id=fold_id,
            objective="ols", chunk_size=7, membership=membership, blas_threads=1, memory="1G", time_limit="00:05:00",
            concurrency=2, selection_report=selection_report)
        write_manifest(manifest)
        receipt_path, receipt_sha = self._completion_receipt(Path(str(profile["output_root"])), f"fake-{fold_id}", manifest["manifest_sha256"])
        unsigned = {"schema_version": "stage1_fit_reduced/v2", "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(Path(str(profile["output_root"])) / "manifests" / f"{manifest['phase']}-{manifest['manifest_sha256'][:20]}.json"),
            "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha, "fit_rows_content_sha256": receipt["content"],
            "fit_identity": {"model": "M0", "fold_id": fold_id, "replicate_id": None, "objective": "ols", "bound_version": 0},
            "fit_bounds": manifest["fit"]["bounds"], "membership": membership, "review_only": False, "status": "reduced",
            "selected": {"objective": 1.0, "coordinates": [.25, .25, 0.0, 0.0],
                "shared_parameters": {"alpha_0": .25, "alpha_1": 0.0, "beta_0": .25, "beta_1": 0.0,
                "delta": 0.0, "omega": 0.0, "phi_0": 0.0, "phi_1": 0.0, "psi_0": 0.0, "psi_1": 0.0}},
            "attempts": [], "frozen_delta": None,
            "adaptive_huber_scale": {"location": 0.0, "weighted_mad": 0.1, "scale": 0.14826, "delta": 0.1999097, "use_ols": False},
            "parents": manifest["parents"], "selection_report": manifest["selection_report"]}
        result = {**unsigned, "result_sha256": canonical_json_sha256(unsigned)}
        path = result_path(manifest); write_json_once(path, result)
        audit_unsigned = {"schema_version": "stage1_fit_grid_audit/v1", "manifest_sha256": manifest["manifest_sha256"],
            "result_sha256": result["result_sha256"], "status": "complete", "actions": [], "reason": None, "followup_manifest": None}
        write_json_once(audit_path(result, output_root=Path(str(profile["output_root"]))),
            {**audit_unsigned, "audit_sha256": canonical_json_sha256(audit_unsigned)})
        return path

    def _write_reduced(self, manifest: dict[str, object], *, selected: dict[str, object] | None) -> Path:
        root = Path(str(manifest["execution"]["output_root"])); receipt_path, receipt_sha = self._completion_receipt(root,
            f"fake-{manifest['manifest_sha256'][:20]}", manifest["manifest_sha256"])
        unsigned = {"schema_version": "stage1_fit_reduced/v2", "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(root / "manifests" / f"{manifest['phase']}-{manifest['manifest_sha256'][:20]}.json"),
            "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha,
            "fit_rows_content_sha256": manifest["input"]["fit_rows_content_sha256"],
            "fit_identity": {key: manifest["fit"][key] for key in ("model", "fold_id", "replicate_id", "objective", "bound_version")},
            "fit_bounds": manifest["fit"]["bounds"], "membership": manifest["membership"], "review_only": False,
            "status": "reduced" if selected is not None else "failed", "selected": selected,
            "attempts": [], "frozen_delta": manifest["fit"]["delta"],
            "adaptive_huber_scale": None, "parents": manifest["parents"],
            "selection_report": manifest["selection_report"]}
        result = {**unsigned, "result_sha256": canonical_json_sha256(unsigned)}
        path = result_path(manifest); write_json_once(path, result); return path

    def _report_result(self, *, root: Path, name: str, model: str, fold_id: str, membership: dict[str, object], content: str,
        replicate_id: int | None = None, selection_report: Path | None = None,
        bound_repeat: Path | None = None, bound_version: int = 0,
        boundary_contacts: list[dict[str, object]] | None = None, full_data: Path | None = None,
        scale_parent: Path | None = None, warm_parent: Path | None = None, delta: float = .2) -> Path:
        shared = {"alpha_0": .25, "alpha_1": 0.0, "beta_0": .25, "beta_1": 0.0, "delta": 0.0, "omega": 0.0,
            "phi_0": 0.0, "phi_1": 0.0, "psi_0": 0.0, "psi_1": 0.0}
        coordinates = {"C0": [.25, .25],
            "J0": [.25, .25, .25, .25, 0.0, 0.0],
            "JH": [.25, .25, .25, .25, 0.0, 0.0],
            "M0": [.25, .25, 0.0, 0.0],
            "M1": [.25, .25, .25, .25, 0.0, 0.0, 0.0, 0.0],
            "M2": [.25, .25, .25, .25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}[model]
        selected = {"objective": 1.0, "coordinates": coordinates, "shared_parameters": shared, "language_coefficients": [
            {"language": "eng", "L_inf": 1.0, "A": .1, "B": .1, "floor_fraction": .5, "floor_saturated": False}],
            "boundary_contacts": list(boundary_contacts or [])}
        parent_path = scale_parent
        if parent_path is None:
            parent_unsigned = {"fixture": name}
            parent = {**parent_unsigned, "result_sha256": canonical_json_sha256(parent_unsigned)}
            parent_path = root / "parents" / f"{name}.json"; write_json_once(parent_path, parent)
        parents = {"scale_m0_ols": parent_path}
        if warm_parent is not None: parents["warm"] = warm_parent
        if bound_repeat is not None: parents["bound_repeat"] = bound_repeat
        explicit_starts = None; action = "grid"
        if replicate_id is not None:
            if full_data is None: raise ValueError("D16 fixture requires its full-data anchor")
            parents["full_data"] = full_data
            explicit_starts = d16_adaptive_starts(model, coordinates, (), initial_bounds(model), target=32)
            action = "d16_adaptive"
        manifest = build_manifest(output_root=root, fit_rows_path=root / "fit_rows.parquet",
            fit_rows_content_sha256=content, phase=name, model=model, fold_id=fold_id, objective="huber", delta=delta,
            chunk_size=59049, replicate_id=replicate_id, membership=membership, bound_version=bound_version,
            parent_results=parents, selection_report=selection_report,
            explicit_starts=explicit_starts, action=action,
            blas_threads=1, memory="1G", time_limit="00:05:00", concurrency=1)
        manifest_path = write_manifest(manifest)
        receipt_path, receipt_sha = self._completion_receipt(root, f"fake-{name}", manifest["manifest_sha256"])
        unsigned = {"schema_version": "stage1_fit_reduced/v2", "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(manifest_path), "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha, "fit_rows_content_sha256": content,
            "fit_identity": {key: manifest["fit"][key] for key in ("model", "fold_id", "replicate_id", "objective", "bound_version")},
            "fit_bounds": manifest["fit"]["bounds"], "membership": membership, "review_only": False, "status": "reduced", "selected": selected,
            "attempts": [], "frozen_delta": delta, "adaptive_huber_scale": None,
            "selection_report": manifest["selection_report"], "parents": manifest["parents"]}
        result = {**unsigned, "result_sha256": canonical_json_sha256(unsigned)}
        path = result_path(manifest); write_json_once(path, result)
        audit_unsigned = {"schema_version": "stage1_fit_grid_audit/v1", "manifest_sha256": manifest["manifest_sha256"],
            "result_sha256": result["result_sha256"], "status": "complete", "actions": [], "reason": None, "followup_manifest": None}
        write_json_once(audit_path(result, output_root=root), {**audit_unsigned, "audit_sha256": canonical_json_sha256(audit_unsigned)})
        return path

    def test_definitive_short_grid_rejected_and_review_grid_cannot_be_scientific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            membership = bind_membership(holdouts, kind="primary", membership_id="largest_N_act")
            with self.assertRaisesRegex(ExecutionError, "every registered start"):
                build_manifest(output_root=root / "out", fit_rows_path=fit_rows, fit_rows_content_sha256=receipt["content"],
                    phase="bad", model="M0", fold_id="largest_N_act", objective="ols", chunk_size=1, start_total=1,
                    membership=membership)
            review = build_manifest(output_root=root / "out", fit_rows_path=fit_rows, fit_rows_content_sha256=receipt["content"],
                phase="review", model="M0", fold_id="largest_N_act", objective="ols", chunk_size=1, start_total=1,
                membership=membership, review_only=True)
            result = run_serial(review, runner=lambda _m, index: {"start_index": index, "start_id": "grid:000000",
                "success": True, "status": 0, "message": "review", "objective": 1.0})
            self.assertTrue(result["review_only"])
            with self.assertRaisesRegex(CampaignError, "complete definitive"):
                scientific_result(result_path(review))

    def test_memberships_are_bound_and_secondary_terminal_rows_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            primary = bind_membership(holdouts, kind="primary", membership_id="largest_N_act")
            manifest = build_manifest(output_root=root / "out", fit_rows_path=fit_rows, fit_rows_content_sha256=receipt["content"],
                phase="review", model="M0", fold_id="largest_N_act", objective="ols", chunk_size=1, start_total=1,
                membership=primary, review_only=True)
            self.assertEqual(len(load_problem(manifest).y), 15)
            terminal = bind_membership(holdouts, kind="terminal")
            terminal_manifest = build_manifest(output_root=root / "out", fit_rows_path=fit_rows,
                fit_rows_content_sha256=receipt["content"], phase="terminal", model="M0", fold_id="terminal", objective="ols",
                chunk_size=1, start_total=1, membership=terminal, review_only=True)
            self.assertEqual(len(load_problem(terminal_manifest).y), 16)
            joint_manifest = build_manifest(output_root=root / "out", fit_rows_path=fit_rows,
                fit_rows_content_sha256=receipt["content"], phase="joint-published", model="JH",
                fold_id="largest_N_act", objective="ols", chunk_size=1, start_total=1,
                membership=primary, review_only=True)
            joint_problem = load_problem(joint_manifest)
            expected = pd.read_parquet(fit_rows)
            expected = expected[expected["trajectory_id"].isin(primary["train_trajectory_ids"])]
            self.assertEqual(joint_problem.D.tolist(), expected["D_total"].astype(float).tolist())
            holdouts.write_text("{}\n")
            with self.assertRaisesRegex(ExecutionError, "membership file hash changed"):
                validate_manifest(manifest)

    def test_candidate_planner_requires_exact_parent_and_d16_memberships_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            first = plan_candidate_campaign(input_receipt_path=Path(str(receipt["path"])), holdouts_path=holdouts, profile=profile)
            self.assertEqual(set(first), {f"{fold}/m0_ols" for fold in ("largest_N_act", "E32", "K200_common_languages")})
            for path in first.values():
                manifest = load_manifest(path); self.assertEqual(manifest["fit"]["start_count"], 81); self.assertFalse(manifest["review_only"])
            completed = {}
            for fold in ("largest_N_act", "E32", "K200_common_languages"):
                membership = bind_membership(holdouts, kind="primary", membership_id=fold)
                completed[f"{fold}/m0_ols"] = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt,
                    profile=profile, membership=membership, fold_id=fold)
            second = plan_candidate_campaign(input_receipt_path=Path(str(receipt["path"])), holdouts_path=holdouts,
                profile=profile, completed=completed)
            self.assertEqual(set(second), {f"{fold}/m0_huber" for fold in ("largest_N_act", "E32", "K200_common_languages")})
            trajectories = [f"t{index:02d}" for index in range(64)]
            resampling = root / "resampling.json"; write_json_once(resampling,
                build_resampling_manifest(trajectories, input_sha256=receipt["content"]).to_dict())
            first_subset = bind_membership(resampling, kind="d16", membership_id=0)
            second_subset = bind_membership(resampling, kind="d16", membership_id=1)
            self.assertNotEqual(first_subset["membership_sha256"], second_subset["membership_sha256"])
            candidate = self._candidate_report(root, input_sha256=receipt["content"])
            all_membership = bind_membership(holdouts, kind="all")
            full_ols = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt, profile=profile,
                membership=all_membership, fold_id="all_data", selection_report=candidate)
            full_target = self._report_result(root=root, name="all-m0-huber", model="M0", fold_id="all_data",
                membership=all_membership, content=receipt["content"], selection_report=candidate)
            d16_completed = {"all_data/m0_ols": full_ols, "all_data/m0_huber": full_target}
            d16 = plan_d16_refits(candidate_report_path=candidate, input_receipt_path=Path(str(receipt["path"])),
                resampling_path=resampling, profile=profile, completed=d16_completed)
            self.assertEqual(len(d16), 100)
            first_d16 = load_manifest(d16["d16:000/m0_ols"])
            self.assertEqual(first_d16["membership"]["membership_sha256"], first_subset["membership_sha256"])
            self.assertEqual(load_manifest(d16["d16:001/m0_ols"])["membership"]["membership_sha256"], second_subset["membership_sha256"])
            self.assertEqual(first_d16["selection_report"]["path"], str(candidate.resolve()))
            self.assertEqual(first_d16["fit"]["start_mode"], "explicit")
            self.assertEqual(first_d16["fit"]["start_count"], 32)
            self.assertEqual(first_d16["fit"]["action"], "d16_adaptive")
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._plan_stage", return_value={}) as planner:
                plan_secondary_diagnostic(diagnostic="lowest", candidate_report_path=candidate,
                    input_receipt_path=Path(str(receipt["path"])), holdouts_path=holdouts,
                    profile=profile)
            self.assertEqual(planner.call_args.kwargs["target_stage"], "m2_huber")

    @patch('multilingual_megatron.experiments.scaling.stage1_fit.selection.EXPECTED_READOUTS', 20)
    def test_generalization_planner_waves_and_nonselecting_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            supplement = self._generalization_fixture(root, holdouts, str(receipt['content']))
            frozen = read_json(supplement)['logical_compute']
            self.assertEqual((frozen['train_count'], frozen['test_count']), (16, 4))
            self.assertEqual(frozen['threshold_flops'], 3_600_000_000_000_000_000)
            self.assertEqual(frozen['test_fraction'], .2)
            holdout_payload = read_json(holdouts)
            candidate = self._candidate_report(root, model='M1', input_sha256=str(receipt['content']),
                holdout_manifest_sha256=holdout_payload['manifest_sha256'],
                input_receipt={'path': str(Path(str(receipt['path'])).resolve()),
                    'file_sha256': sha256_file(Path(str(receipt['path'])))})
            memberships = {'random_13_of_64': bind_membership(holdouts, kind='random',
                    membership_id='random_13_of_64'),
                'terminal_max_exposure': bind_membership(holdouts, kind='terminal'),
                LARGEST_COMPUTE_VIEW: bind_membership(supplement, kind='compute',
                    membership_id=LARGEST_COMPUTE_VIEW)}
            wrong_unsigned = dict(read_json(candidate)); wrong_unsigned.pop('report_sha256')
            wrong_unsigned['input_receipt'] = {**wrong_unsigned['input_receipt'], 'file_sha256': '0' * 64}
            wrong_candidate = root / 'wrong-candidate.json'
            write_json_once(wrong_candidate, {**wrong_unsigned,
                'report_sha256': canonical_json_sha256(wrong_unsigned)})
            receipt_profile = {**profile, 'output_root': str(root / 'bad-receipt-output')}
            with self.assertRaisesRegex(CampaignError, 'selected candidate inputs'):
                plan_generalization_comparison(candidate_report_path=wrong_candidate,
                    input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                    generalization_holdouts_path=supplement, profile=receipt_profile)
            self.assertFalse((root / 'bad-receipt-output').exists())
            cross_profile = {**profile, 'output_root': str(root / 'bad-cross-wave-output')}
            with self.assertRaisesRegex(CampaignError, 'precedes a complete M0-OLS wave'):
                plan_generalization_comparison(candidate_report_path=candidate,
                    input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                    generalization_holdouts_path=supplement, profile=cross_profile,
                    completed={'generalization:random_13_of_64/M1': root / 'absent.json'})
            self.assertFalse((root / 'bad-cross-wave-output').exists())
            first = plan_generalization_comparison(candidate_report_path=candidate,
                input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                generalization_holdouts_path=supplement, profile=profile)
            self.assertEqual(set(first), {f'generalization:{view}/m0_ols' for view in memberships})
            completed = {}
            for view, membership in memberships.items():
                key = f'generalization:{view}/m0_ols'
                completed[key] = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt,
                    profile=profile, membership=membership, fold_id=view, selection_report=candidate)
            second = plan_generalization_comparison(candidate_report_path=candidate,
                input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                generalization_holdouts_path=supplement, profile=profile, completed=completed)
            self.assertEqual(len(second), 9)
            expected_starts = {'C0': 9, 'M0': 81, 'J0': 729}
            for key, path in second.items():
                self.assertEqual(load_manifest(path)['fit']['start_count'], expected_starts[key.rsplit('/', 1)[1]])
            results = {view: {} for view in memberships}; delta = .1999097
            for view, membership in memberships.items():
                scale = completed[f'generalization:{view}/m0_ols']
                for model in ('C0', 'M0', 'J0'):
                    path = self._report_result(root=root, name=f'generalization-{view}-{model}',
                        model=model, fold_id=view, membership=membership, content=str(receipt['content']),
                        selection_report=candidate, scale_parent=scale, delta=delta)
                    completed[f'generalization:{view}/{model}'] = path; results[view][model] = path
            third = plan_generalization_comparison(candidate_report_path=candidate,
                input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                generalization_holdouts_path=supplement, profile=profile, completed=completed)
            self.assertEqual(set(third), {f'generalization:{view}/M1' for view in memberships})
            for key, path in third.items():
                view = key.removeprefix('generalization:').removesuffix('/M1')
                manifest = load_manifest(path)
                self.assertEqual(manifest['fit']['start_count'], 6_561)
                warm = [item for item in manifest['parents'] if item['kind'] == 'warm']
                self.assertEqual(len(warm), 1)
                self.assertEqual(warm[0]['path'], str(completed[f'generalization:{view}/M0'].resolve()))
            for view, membership in memberships.items():
                m1 = self._report_result(root=root, name=f'generalization-{view}-M1',
                    model='M1', fold_id=view, membership=membership, content=str(receipt['content']),
                    selection_report=candidate, scale_parent=completed[f'generalization:{view}/m0_ols'],
                    warm_parent=completed[f'generalization:{view}/M0'], delta=delta)
                results[view]['M1'] = m1
            report = build_generalization_report(candidate_report_path=candidate,
                input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                generalization_holdouts_path=supplement, results=results)
            self.assertEqual(report['fair_models'], ['C0', 'M0', 'J0', 'M1'])
            self.assertNotIn('selection', report); self.assertNotIn('aggregate', report)
            self.assertEqual(report['views']['random_13_of_64']['evidence_class'],
                'independent_trajectory_holdout')
            self.assertIn('trajectory_balanced_r2',
                report['metrics'][LARGEST_COMPUTE_VIEW]['M1']['heldout'])
            bad_unsigned = dict(read_json(supplement)); bad_unsigned.pop('manifest_sha256')
            split = dict(bad_unsigned[LARGEST_COMPUTE_VIEW])
            train, test = list(split['train_readout_ids']), list(split['test_readout_ids'])
            train[0], test[0] = test[0], train[0]
            split.update({'train_readout_ids': sorted(train), 'test_readout_ids': sorted(test)})
            split_unsigned = dict(split); split_unsigned.pop('membership_sha256')
            split['membership_sha256'] = canonical_json_sha256(split_unsigned)
            bad_unsigned[LARGEST_COMPUTE_VIEW] = split
            bad = root / 'bad-generalization.json'
            write_json_once(bad, {**bad_unsigned, 'manifest_sha256': canonical_json_sha256(bad_unsigned)})
            bad_profile = {**profile, 'output_root': str(root / 'bad-output')}
            with self.assertRaisesRegex(CampaignError, 'recomputed frozen split'):
                plan_generalization_comparison(candidate_report_path=candidate,
                    input_receipt_path=Path(str(receipt['path'])), holdouts_path=holdouts,
                    generalization_holdouts_path=bad, profile=bad_profile)
            self.assertFalse((root / 'bad-output').exists())

    def test_literature_planner_reuses_scales_and_plans_only_three_new_forms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            candidate = self._candidate_report(root, model="M1", input_sha256=receipt["content"])
            completed = {}
            for fold in ("largest_N_act", "E32", "K200_common_languages"):
                membership = bind_membership(holdouts, kind="primary", membership_id=fold)
                completed[f"{fold}/m0_ols"] = self._fake_scientific_m0(root=root, fit_rows=fit_rows,
                    receipt=receipt, profile=profile, membership=membership, fold_id=fold)
            lowest = bind_membership(holdouts, kind="lowest", membership_id="lowest_terminal_macro_loss_30")
            completed["secondary:lowest/m0_ols"] = self._fake_scientific_m0(root=root,
                fit_rows=fit_rows, receipt=receipt, profile=profile, membership=lowest, fold_id="lowest",
                selection_report=candidate)
            all_data = bind_membership(holdouts, kind="all")
            completed["all_data/m0_ols"] = self._fake_scientific_m0(root=root, fit_rows=fit_rows,
                receipt=receipt, profile=profile, membership=all_data, fold_id="all_data",
                selection_report=candidate)
            planned = plan_literature_comparison(candidate_report_path=candidate,
                input_receipt_path=Path(str(receipt["path"])), holdouts_path=holdouts,
                profile=profile, completed=completed)
            self.assertEqual(len(planned), 15)
            self.assertEqual({key.rsplit("/", 1)[1] for key in planned}, {"C0", "J0", "JH"})
            for key, path in planned.items():
                manifest = load_manifest(path)
                expected = {"C0": 9, "J0": 729, "JH": 729}[key.rsplit("/", 1)[1]]
                self.assertEqual(manifest["fit"]["start_count"], expected)
                self.assertEqual(manifest["selection_report"]["path"], str(candidate.resolve()))

    def test_literature_planner_preflights_every_view_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            candidate = self._candidate_report(root, model="M1", input_sha256=receipt["content"])
            completed = {}
            for fold in ("largest_N_act", "E32", "K200_common_languages"):
                membership = bind_membership(holdouts, kind="primary", membership_id=fold)
                completed[f"{fold}/m0_ols"] = self._fake_scientific_m0(root=root, fit_rows=fit_rows,
                    receipt=receipt, profile=profile, membership=membership, fold_id=fold)
            planning_profile = {**profile, "output_root": str(root / "planned")}
            with self.assertRaisesRegex(CampaignError, "secondary:lowest/m0_ols"):
                plan_literature_comparison(candidate_report_path=candidate,
                    input_receipt_path=Path(str(receipt["path"])), holdouts_path=holdouts,
                    profile=planning_profile, completed=completed)
            self.assertFalse((root / "planned").exists())

    def test_literature_report_separates_harmonized_and_published_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, holdouts, receipt, _ = self._fixture(root)
            primary_results = {model: {} for model in ("M0", "M1", "M2")}
            scale_parents = {}
            for fold in ("largest_N_act", "E32", "K200_common_languages", "lowest", "all_data"):
                unsigned = {"fixture": f"scale-{fold}"}
                payload = {**unsigned, "result_sha256": canonical_json_sha256(unsigned)}
                scale_parents[fold] = root / "parents" / f"scale-{fold}.json"
                write_json_once(scale_parents[fold], payload)
            for model in primary_results:
                for fold in ("largest_N_act", "E32", "K200_common_languages"):
                    membership = bind_membership(holdouts, kind="primary", membership_id=fold)
                    primary_results[model][fold] = self._report_result(root=root,
                        name=f"primary-{model}-{fold}", model=model, fold_id=fold,
                        membership=membership, content=receipt["content"], scale_parent=scale_parents[fold])
            candidate = build_candidate_report(input_receipt_path=Path(receipt["path"]),
                holdouts_path=holdouts, results=primary_results)
            candidate["selection"]["selected_model"] = "M1"
            candidate.pop("report_sha256"); candidate["report_sha256"] = canonical_json_sha256(candidate)
            candidate_path = root / "candidate-literature.json"; write_json_once(candidate_path, candidate)
            results = {}
            for view in ("largest_N_act", "E32", "K200_common_languages"):
                membership = bind_membership(holdouts, kind="primary", membership_id=view)
                results[view] = {"M0": primary_results["M0"][view], "M1": primary_results["M1"][view]}
                for model in ("C0", "J0", "JH"):
                    results[view][model] = self._report_result(root=root, name=f"lit-{view}-{model}",
                        model=model, fold_id=view, membership=membership, content=receipt["content"],
                        selection_report=candidate_path, scale_parent=scale_parents[view])
            for view, membership in (("lowest", bind_membership(holdouts, kind="lowest",
                    membership_id="lowest_terminal_macro_loss_30")),
                    ("all_data", bind_membership(holdouts, kind="all"))):
                results[view] = {model: self._report_result(root=root, name=f"lit-{view}-{model}",
                    model=model, fold_id=view, membership=membership, content=receipt["content"],
                    selection_report=candidate_path, scale_parent=scale_parents[view])
                    for model in ("C0", "M0", "J0", "JH", "M1")}
            contract_unsigned = {"schema_version": LOGICAL_COMPUTE_CONTRACT_SCHEMA,
                "logical_compute": {"formula": "F=6*N_act*D_total", "unit": "FLOPs"},
                "physical_compute": "excluded_separate_measured_h100_ledger",
                "admissible_domain": {"label": "test", "N_act": {"minimum": 1.0e8, "maximum": 2.0e8},
                    "D_total": {"minimum": 1.0e9, "maximum": 2.0e9}, "E": [1, 32], "K": [1]},
                "budgets": [{"budget_id": "small", "maximum_flops": 2.4e18},
                    {"budget_id": "large", "maximum_flops": 3.0e18}],
                "prediction_points": [
                    {"point_id": "small", "N_act": 1.0e8, "D_total": 1.0e9, "E": 1, "K": 1,
                        "D_t_by_language": {"eng": 1.0e9}},
                    {"point_id": "large", "N_act": 2.0e8, "D_total": 2.0e9, "E": 32, "K": 1,
                        "D_t_by_language": {"eng": 2.0e9}}]}
            contract = {**contract_unsigned, "contract_sha256": canonical_json_sha256(contract_unsigned)}
            contract_path = root / "logical-compute.json"; write_json_once(contract_path, contract)
            report = build_literature_comparison_report(candidate_report_path=candidate_path,
                input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts,
                logical_compute_contract_path=contract_path, results=results)
            self.assertEqual(report["fair_primary_models"], ["C0", "M0", "J0", "M1"])
            self.assertEqual(report["published_input_sensitivity"], "JH")
            self.assertEqual(report["model_contracts"]["JH"]["data"], "D_total")
            self.assertEqual(report["paired_descriptive_comparison"]["bootstrap_replicates"], 10_000)
            self.assertNotIn("selected_model", report["paired_descriptive_comparison"])
            self.assertIn("training_plus_extrapolation_rmse", report["lowest_loss_metrics"]["M1"])
            self.assertEqual(set(report["all_data_fits_for_d20"]), {"C0", "M0", "J0", "JH", "M1"})
            self.assertEqual(set(report["logical_compute_predictions"]["models"]), {"C0", "M0", "J0", "JH", "M1"})
            self.assertEqual(report["logical_compute_predictions"]["physical_compute_ledger"], "excluded")
            self.assertEqual({item["E"] for item in
                report["logical_compute_predictions"]["models"]["M1"]["optima_by_expert"]}, {1, 32})
            self.assertEqual(report["model_contracts"]["JH"]["coefficient_comparability"],
                "affine-normalized; do not compare directly with published coefficients")
            mismatched = {view: dict(models) for view, models in results.items()}
            membership = bind_membership(holdouts, kind="primary", membership_id="largest_N_act")
            mismatched["largest_N_act"]["C0"] = self._report_result(root=root,
                name="lit-mismatched-scale", model="C0", fold_id="largest_N_act",
                membership=membership, content=receipt["content"], selection_report=candidate_path)
            with self.assertRaisesRegex(ReportError, "fold-specific Huber scale"):
                build_literature_comparison_report(candidate_report_path=candidate_path,
                    input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts,
                    logical_compute_contract_path=contract_path, results=mismatched)

    def test_wrong_m0_parent_membership_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            completed = {}
            for fold in ("largest_N_act", "E32", "K200_common_languages"):
                parent_fold = "E32" if fold == "largest_N_act" else fold
                membership = bind_membership(holdouts, kind="primary", membership_id=parent_fold)
                completed[f"{fold}/m0_ols"] = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt,
                    profile=profile, membership=membership, fold_id=parent_fold)
            with self.assertRaisesRegex(CampaignError, "different frozen membership|parent fold differs"):
                plan_candidate_campaign(input_receipt_path=Path(str(receipt["path"])), holdouts_path=holdouts, profile=profile,
                    completed=completed)

    def test_failed_d16_tier_escalates_to_the_registered_full_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            candidate = self._candidate_report(root, model="M0", input_sha256=receipt["content"])
            all_membership = bind_membership(holdouts, kind="all")
            full_ols = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt, profile=profile,
                membership=all_membership, fold_id="all_data", selection_report=candidate)
            full_target = self._report_result(root=root, name="all-m0-huber-escalation", model="M0",
                fold_id="all_data", membership=all_membership, content=receipt["content"], selection_report=candidate)
            trajectories = [f"t{index:02d}" for index in range(64)]
            resampling = root / "resampling.json"; write_json_once(resampling,
                build_resampling_manifest(trajectories, input_sha256=receipt["content"]).to_dict())
            planned = plan_d16_refits(candidate_report_path=candidate,
                input_receipt_path=Path(str(receipt["path"])), resampling_path=resampling, profile=profile,
                completed={"all_data/m0_ols": full_ols, "all_data/m0_huber": full_target})
            manifest_path = planned["d16:000/m0_ols"]; manifest = load_manifest(manifest_path)
            self._write_reduced(manifest, selected=None)
            audit = audit_grid(manifest_path)
            self.assertEqual(audit["status"], "needs_escalation")
            followup = load_manifest(Path(audit["followup_manifest"]))
            self.assertEqual(followup["fit"]["start_count"], 81)
            self.assertEqual(followup["fit"]["start_mode"], "grid")
            self.assertEqual(followup["fit"]["action"], "d16_full_grid")
            self.assertIn("adaptive_parent", {item["kind"] for item in followup["parents"]})
            followup_result = self._write_reduced(followup, selected={"objective": 1.0,
                "coordinates": [.25, .25, 0.0, 0.0], "boundary_contacts": []})
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._d16_numerical_checks",
                    return_value=(True, [], None)):
                self.assertEqual(audit_grid(Path(followup_result).parent.parent / "manifests" /
                    f"{followup['phase']}-{followup['manifest_sha256'][:20]}.json")["status"], "complete")
            history, _ = _d16_adaptive_history(read_json(followup_result))
            self.assertEqual([item["start_count"] for item in history], [32, 81])

    def test_one_start_real_default_executor_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            membership = bind_membership(holdouts, kind="primary", membership_id="largest_N_act")
            manifest = build_manifest(output_root=root / "out", fit_rows_path=fit_rows, fit_rows_content_sha256=receipt["content"],
                phase="real-smoke", model="M0", fold_id="largest_N_act", objective="ols", chunk_size=1, start_total=1,
                membership=membership, review_only=True, blas_threads=1, memory="1G", time_limit="00:05:00", concurrency=1)
            result = run_serial(manifest)
            self.assertEqual(len(result["attempts"]), 1)
            self.assertIsNotNone(result["selected"])
            self.assertTrue((root / "out" / "chunks").exists())

    def test_audit_enforces_bound_repeats_basin_warm_and_direct_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            membership = bind_membership(holdouts, kind="primary", membership_id="largest_N_act")
            def grid(phase: str, *, model: str = "M0", objective: str = "ols", delta: float | None = None,
                parents: dict[str, Path] | None = None, bounds=None) -> dict[str, object]:
                manifest = build_manifest(output_root=root / "out", fit_rows_path=fit_rows, fit_rows_content_sha256=receipt["content"],
                    phase=phase, model=model, fold_id="largest_N_act", objective=objective, delta=delta, chunk_size=99,
                    membership=membership, parent_results=parents, bounds=bounds, blas_threads=1, memory="1G", time_limit="00:05:00", concurrency=1)
                write_manifest(manifest); return manifest
            contact = {"name": "alpha_e1", "side": "lower", "value": .01, "distance": 0.0, "interval_width": 1.99}
            boundary = grid("boundary")
            self._write_reduced(boundary, selected={"objective": 1.0, "coordinates": [.01, .25, 0.0, 0.0], "boundary_contacts": [contact]})
            self.assertEqual(audit_grid(root / "out" / "manifests" / f"boundary-{boundary['manifest_sha256'][:20]}.json")["status"], "needs_bound_repeat")
            exhausted_bounds = list(initial_bounds("M0")); exhausted_bounds[0] = OuterBound("alpha_e1", .01, 2.0, "positive", 3)
            exhausted = grid("exhausted", bounds=exhausted_bounds)
            self._write_reduced(exhausted, selected={"objective": 1.0, "coordinates": [.01, .25, 0.0, 0.0], "boundary_contacts": [contact]})
            self.assertEqual(audit_grid(root / "out" / "manifests" / f"exhausted-{exhausted['manifest_sha256'][:20]}.json")["status"], "failed")
            basin = grid("basin")
            self._write_reduced(basin, selected={"objective": 1.0, "coordinates": [.25, .25, 0.0, 0.0], "boundary_contacts": []})
            records = [{"success": True, "objective": 1.0, "start_id": f"grid:{i:06d}", "coordinates": [.25, .25, 0.0, 0.0]} for i in range(3)]
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._records", return_value=records), \
                 patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._derived", return_value=(root / "out" / "manifests" / f"basin-{basin['manifest_sha256'][:20]}.json", {})), \
                 patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._equivalent_to_retained", return_value=False):
                self.assertEqual(audit_grid(root / "out" / "manifests" / f"basin-{basin['manifest_sha256'][:20]}.json")["status"], "failed")
            direct = grid("direct")
            self._write_reduced(direct, selected={"objective": 1.0, "coordinates": [.25, .25, 0.0, 0.0], "boundary_contacts": []})
            direct_path = root / "out" / "manifests" / f"direct-{direct['manifest_sha256'][:20]}.json"
            def derived(**kwargs):
                return direct_path, {"selected": {"direct_equivalence": {"resolved": kwargs["action"] != "direct"}}}
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._records", return_value=records), \
                 patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._derived", side_effect=derived), \
                 patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._equivalent_to_retained", return_value=True):
                self.assertEqual(audit_grid(direct_path)["status"], "failed")
            parent = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt, profile=profile,
                membership=membership, fold_id="largest_N_act")
            warm = grid("warm", model="M1", objective="huber", delta=.1999097, parents={"scale_m0_ols": parent, "warm": parent})
            self._write_reduced(warm, selected={"objective": 1.0, "coordinates": [.25, .25, .25, .25, 0.0, 0.0, 0.0, 0.0], "boundary_contacts": []})
            warm_path = root / "out" / "manifests" / f"warm-{warm['manifest_sha256'][:20]}.json"
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._derived", return_value=(warm_path, {"selected": {"objective": 2.0}})):
                self.assertEqual(audit_grid(warm_path)["status"], "failed")

    def test_local_basin_must_match_retained_grid_optimum_and_huber_repeat_can_expand(self) -> None:
        retained = {"success": True, "objective": 1.0, "start_id": "grid:000000"}
        local = [{"success": True, "objective": .5, "start_id": f"local:{index}"} for index in range(3)]
        with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign._record_prediction", return_value=[1.0]):
            self.assertFalse(_equivalent_to_retained(local, manifest={}, retained=retained, retained_manifest={}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            membership = bind_membership(holdouts, kind="primary", membership_id="largest_N_act")
            parent = self._fake_scientific_m0(root=root, fit_rows=fit_rows, receipt=receipt, profile=profile,
                membership=membership, fold_id="largest_N_act")
            candidate = self._candidate_report(root, input_sha256=receipt["content"])
            initial = build_manifest(output_root=root / "out", fit_rows_path=fit_rows,
                fit_rows_content_sha256=receipt["content"], phase="m0-huber", model="M0", fold_id="largest_N_act",
                objective="huber", delta=.1999097, chunk_size=81, membership=membership,
                parent_results={"scale_m0_ols": parent}, selection_report=candidate,
                blas_threads=1, memory="1G", time_limit="00:05:00", concurrency=1)
            initial_path = write_manifest(initial)
            previous = self._write_reduced(initial, selected={"objective": 1.0, "coordinates": [.01, .25, 0.0, 0.0],
                "boundary_contacts": [{"name": "alpha_e1", "side": "lower", "value": .01, "distance": 0.0, "interval_width": 1.99}]})
            first_audit = audit_grid(initial_path)
            followup = load_manifest(Path(first_audit["followup_manifest"]))
            self.assertEqual(followup["selection_report"], initial["selection_report"])
            expanded_bounds = list(initial_bounds("M0")); expanded_bounds[0] = OuterBound("alpha_e1", .001, 2.0, "positive", 1)
            repeat = build_manifest(output_root=root / "out", fit_rows_path=fit_rows,
                fit_rows_content_sha256=receipt["content"], phase="m0-huber-bounds-1", model="M0", fold_id="largest_N_act",
                objective="huber", delta=.1999097, chunk_size=81, membership=membership,
                parent_results={"scale_m0_ols": parent, "bound_repeat": previous}, bounds=expanded_bounds, bound_version=1,
                selection_report=candidate, blas_threads=1, memory="1G", time_limit="00:05:00", concurrency=1)
            repeat_path = write_manifest(repeat)
            contact = {"name": "alpha_e1", "side": "lower", "value": .001, "distance": 0.0, "interval_width": 1.999}
            self._write_reduced(repeat, selected={"objective": 1.0, "coordinates": [.001, .25, 0.0, 0.0], "boundary_contacts": [contact]})
            self.assertEqual(audit_grid(repeat_path)["status"], "needs_bound_repeat")

    def test_candidate_report_has_primary_rmse_r2_and_separate_secondary_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            results = {}
            for model in ("M0", "M1", "M2"):
                results[model] = {}
                for fold in ("largest_N_act", "E32", "K200_common_languages"):
                    membership = bind_membership(holdouts, kind="primary", membership_id=fold)
                    results[model][fold] = self._report_result(root=root, name=f"{model}-{fold}", model=model,
                        fold_id=fold, membership=membership, content=receipt["content"])
            with self.assertRaisesRegex(ReportError, "exactly 10,000"):
                build_candidate_report(input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts,
                    results=results, bootstrap_replicates=20)
            report = build_candidate_report(input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts, results=results)
            self.assertEqual(report["selection"]["bootstrap_replicates"], 10_000)
            self.assertIn("trajectory_balanced_rmse", report["primary_metrics"]["M0"]["largest_N_act"])
            self.assertIn("trajectory_balanced_r2", report["primary_metrics"]["M0"]["largest_N_act"])
            self.assertEqual(report["secondary_diagnostics"]["status"], "publish_as_separate_post_selection_artifact")
            candidate_path = root / "candidate-selection.json"; write_json_once(candidate_path, report)
            selected_model = report["selection"]["selected_model"]
            selected_results = {}
            for name, kind, identifier in (("random", "random", "random_13_of_64"), ("terminal", "terminal", None)):
                membership = bind_membership(holdouts, kind=kind, membership_id=identifier)
                selected_results[name] = self._report_result(root=root, name=f"secondary-{name}", model=selected_model,
                    fold_id=name, membership=membership, content=receipt["content"], selection_report=candidate_path)
            lowest_membership = bind_membership(holdouts, kind="lowest", membership_id="lowest_terminal_macro_loss_30")
            lowest_results = {model: self._report_result(root=root, name=f"secondary-lowest-{model}", model=model,
                fold_id="lowest", membership=lowest_membership, content=receipt["content"], selection_report=candidate_path)
                for model in ("M0", "M1", "M2")}
            secondary = build_secondary_diagnostics_report(candidate_report_path=candidate_path,
                input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts,
                selected_results=selected_results, lowest_results=lowest_results)
            self.assertEqual(set(secondary["lowest_loss_all_candidates"]), {"M0", "M1", "M2"})
            self.assertIn("training_plus_extrapolation_rmse", secondary["lowest_loss_all_candidates"]["M2"])
            secondary_path = write_secondary_diagnostics_report(root, secondary)
            self.assertNotEqual(candidate_path, secondary_path)
            incomplete = {key: value for key, value in results.items() if key != "M2"}
            with self.assertRaisesRegex(ReportError, "incomplete"):
                build_candidate_report(input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts,
                    results=incomplete)
            changed = pd.read_parquet(fit_rows)
            changed.loc[0, "adjusted_loss"] = changed.loc[0, "adjusted_loss"] + .1
            changed.to_parquet(fit_rows, index=False)
            with self.assertRaisesRegex(ReportError, "snapshot file hash changed"):
                build_candidate_report(input_receipt_path=Path(receipt["path"]), holdouts_path=holdouts, results=results)
            with self.assertRaisesRegex(CampaignError, "manifest is no longer valid"):
                scientific_result(results["M0"]["largest_N_act"])
            forged = read_json(Path(receipt["path"])); forged["fit_rows_file_sha256"] = sha256_file(fit_rows)
            forged_path = root / "forged_input_receipt.json"; write_json_once(forged_path, forged)
            with self.assertRaisesRegex(ReportError, "canonical content hash changed"):
                build_candidate_report(input_receipt_path=forged_path, holdouts_path=holdouts, results=results)

    def test_d16_report_summarizes_the_frozen_hundred_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            all_data = self._report_result(root=root, name="all", model="M0", fold_id="all_data",
                membership=bind_membership(holdouts, kind="all"), content=receipt["content"])
            trajectories = [f"t{index:02d}" for index in range(64)]
            resampling = root / "resampling.json"; write_json_once(resampling,
                build_resampling_manifest(trajectories, input_sha256=receipt["content"]).to_dict())
            replicas = {index: self._report_result(root=root, name=f"d16-{index}", model="M0", fold_id="d16",
                replicate_id=index, membership=bind_membership(resampling, kind="d16", membership_id=index),
                content=receipt["content"], full_data=all_data)
                for index in range(100)}
            contact = {"name": "alpha_e1", "side": "lower", "value": .01,
                "distance": 0.0, "interval_width": 1.99}
            membership0 = bind_membership(resampling, kind="d16", membership_id=0)
            previous = self._report_result(root=root, name="d16-0-bound", model="M0", fold_id="d16",
                replicate_id=0, membership=membership0, content=receipt["content"], boundary_contacts=[contact],
                full_data=all_data)
            replicas[0] = self._report_result(root=root, name="d16-0-expanded", model="M0", fold_id="d16",
                replicate_id=0, membership=membership0, content=receipt["content"], bound_repeat=previous,
                bound_version=1, full_data=all_data)
            points = {"stress_test": {"N_act": 1.0e8, "E": 1, "K": 1, "D_t_by_language": {"eng": 1.0e8}}}
            report = build_d16_report(full_result=all_data, resampling_path=resampling,
                replicate_results=replicas, prediction_points=points)
            self.assertEqual(report["convergence_failures"], [])
            self.assertEqual(report["shared_parameter_estimates"]["alpha_0"]["p10"], .25)
            self.assertEqual(report["boundary_activation_replicates"], [0])
            self.assertEqual(report["boundary_frequency"], .01)
            self.assertEqual(report["adaptive_protocol"]["final_tier_counts"], {"32": 100})
            self.assertEqual(report["adaptive_protocol"]["escalated_replicates"], [])
            self.assertEqual(report["derived_prediction_estimates"]["stress_test"]["macro_adjusted_loss"]["p10"],
                report["derived_prediction_estimates"]["stress_test"]["macro_adjusted_loss"]["full_data"])
            other_resampling = root / "other-resampling.json"; write_json_once(other_resampling,
                build_resampling_manifest(trajectories, input_sha256=receipt["content"], seed=20260806).to_dict())
            with self.assertRaisesRegex(ReportError, "frozen D16"):
                build_d16_report(full_result=all_data, resampling_path=other_resampling,
                    replicate_results=replicas, prediction_points=points)

    def test_failed_d16_refit_is_reported_and_finalization_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fit_rows, holdouts, receipt, profile = self._fixture(root)
            candidate = self._candidate_report(root, input_sha256=receipt["content"])
            all_data = self._report_result(root=root, name="all", model="M0", fold_id="all_data",
                membership=bind_membership(holdouts, kind="all"), content=receipt["content"], selection_report=candidate)
            trajectories = [f"t{index:02d}" for index in range(64)]
            resampling = root / "resampling.json"; write_json_once(resampling,
                build_resampling_manifest(trajectories, input_sha256=receipt["content"]).to_dict())
            replicas = {index: self._report_result(root=root, name=f"d16-{index}", model="M0", fold_id="d16",
                replicate_id=index, membership=bind_membership(resampling, kind="d16", membership_id=index),
                content=receipt["content"], selection_report=candidate, full_data=all_data)
                for index in range(100)}
            membership0 = bind_membership(resampling, kind="d16", membership_id=0)
            exhausted = list(initial_bounds("M0")); exhausted[0] = OuterBound("alpha_e1", .01, 2.0, "positive", 3)
            starts = d16_adaptive_starts("M0", [.25, .25, 0.0, 0.0], (), exhausted, target=32)
            failed_manifest = build_manifest(output_root=root, fit_rows_path=fit_rows,
                fit_rows_content_sha256=receipt["content"], phase="d16-terminal-failure", model="M0", fold_id="d16",
                objective="ols", chunk_size=32, replicate_id=0, membership=membership0, bounds=exhausted,
                parent_results={"full_data": all_data}, selection_report=candidate,
                explicit_starts=starts, action="d16_adaptive", blas_threads=1, memory="1G",
                time_limit="00:05:00", concurrency=1)
            failed_manifest_path = write_manifest(failed_manifest)
            contact = {"name": "alpha_e1", "side": "lower", "value": .01,
                "distance": 0.0, "interval_width": 1.99}
            failed_path = self._write_reduced(failed_manifest, selected={"objective": 1.0,
                "coordinates": [.01, .25, 0.0, 0.0], "boundary_contacts": [contact]})
            self.assertEqual(audit_grid(failed_manifest_path)["status"], "failed")
            replicas[0] = failed_path
            points = {"stress_test": {"N_act": 1.0e8, "E": 1, "K": 1, "D_t_by_language": {"eng": 1.0e8}}}
            d16 = build_d16_report(full_result=all_data, resampling_path=resampling,
                replicate_results=replicas, prediction_points=points)
            self.assertEqual(d16["convergence_failures"], [0])
            d16_path = root / "d16.json"; write_json_once(d16_path, d16)
            with self.assertRaisesRegex(ReportError, "complete D16"):
                build_final_report(candidate_report_path=candidate, all_data_result=all_data, d16_report_path=d16_path)
            complete_d16 = dict(d16); complete_d16["convergence_failures"] = []
            complete_d16.pop("report_sha256"); complete_d16["report_sha256"] = canonical_json_sha256(complete_d16)
            complete_path = root / "d16-complete.json"; write_json_once(complete_path, complete_d16)
            with self.assertRaisesRegex(ReportError, "D17 is executable|D17"):
                build_final_report(candidate_report_path=candidate, all_data_result=all_data, d16_report_path=complete_path)
            other_candidate = self._candidate_report(root / "other", input_sha256=receipt["content"])
            with self.assertRaisesRegex(ReportError, "all-data result"):
                build_final_report(candidate_report_path=other_candidate, all_data_result=all_data, d16_report_path=complete_path)
            blocked = {"schema_version": "stage1_fit_final_report/v1", "d16_status": "complete", "d17_status": "blocked"}
            blocked = {**blocked, "report_sha256": canonical_json_sha256(blocked)}
            with self.assertRaisesRegex(ReportError, "requires complete"):
                write_final_report(root, blocked)

    def test_prepare_data_writes_frozen_d16_resampling_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = SimpleNamespace(rows=tuple(_rows()), content_sha256="c" * 64,
                trajectory_count=4, readout_count=12)
            args = SimpleNamespace(release_root=root, archive_manifest=root / "archive.json", output_root=root, dry_run=False)
            holdout = SimpleNamespace(to_dict=lambda: {"kind": "holdouts"})
            resampling = SimpleNamespace(to_dict=lambda: {"kind": "resampling"})
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.cli.prepare_fit_data",
                    return_value=(prepared, object())), \
                 patch("multilingual_megatron.experiments.scaling.stage1_fit.cli.build_holdout_manifest", return_value=holdout), \
                 patch("multilingual_megatron.experiments.scaling.stage1_fit.cli.build_resampling_manifest", return_value=resampling):
                result = _prepare(args)
            self.assertTrue(result["written"])
            self.assertEqual(read_json(root / "input" / "holdouts.json")["kind"], "holdouts")
            self.assertEqual(read_json(root / "input" / "resampling.json")["kind"], "resampling")

    def test_cli_exposes_literature_planning_and_reporting(self) -> None:
        planned = cli_parser().parse_args(["plan-literature", "--profile", "profile.json",
            "--input-receipt", "receipt.json", "--holdouts", "holdouts.json",
            "--candidate-report", "candidate.json"])
        self.assertEqual(planned.command, "plan-literature")
        reported = cli_parser().parse_args(["report", "--literature-index", "literature.json"])
        self.assertEqual(reported.literature_index, Path("literature.json"))
        generalization = cli_parser().parse_args(["plan-generalization", "--profile", "profile.json",
            "--input-receipt", "receipt.json", "--holdouts", "holdouts.json",
            "--generalization-holdouts", "generalization.json", "--candidate-report", "candidate.json"])
        self.assertEqual(generalization.command, "plan-generalization")
        prepared = cli_parser().parse_args(["prepare-generalization", "--input-receipt", "receipt.json",
            "--holdouts", "holdouts.json", "--output-root", "output", "--dry-run"])
        self.assertTrue(prepared.dry_run)
        reported = cli_parser().parse_args(["report", "--generalization-index", "generalization-index.json"])
        self.assertEqual(reported.generalization_index, Path("generalization-index.json"))

if __name__ == "__main__":
    unittest.main()
