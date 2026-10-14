from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multilingual_megatron.experiments.scaling.stage1_fit.amendment import (
    ACCEPTANCE_SCHEMA,
    PROTOCOL,
    RUN_SCHEMA,
    AmendmentError,
    _binding,
    _render_slurm,
    _seal,
    _validate_nested_warm_input,
    _validate_result_manifest_binding,
    accepted_path,
    validate_accepted_amendment,
)
from multilingual_megatron.experiments.scaling.stage1_fit.execution import (
    OPTIONAL_HISTORICAL_SOURCE_FILES,
    SOURCE_FILES,
    read_json,
    source_sha256_at_commit,
    write_json_once,
)
from multilingual_megatron.experiments.scaling.stage1_fit.fit import SharedParameters, nested_warm_start


class Stage1FitAmendmentTest(unittest.TestCase):
    def test_historical_source_hash_recovers_versioned_files(self) -> None:
        contents = {name: f"# historical {name}\n".encode() for name in SOURCE_FILES
                    if name not in OPTIONAL_HISTORICAL_SOURCE_FILES}
        expected = hashlib.sha256()
        for name, content in contents.items():
            expected.update(name.encode())
            expected.update(len(content).to_bytes(8, "big"))
            expected.update(content)

        def git_show(command, **kwargs):
            self.assertEqual(command[:2], ["git", "show"])
            self.assertTrue(command[2].startswith("fixture-revision:"))
            name = command[2].rsplit("/", 1)[1]
            if name not in contents:
                raise subprocess.CalledProcessError(128, command)
            return SimpleNamespace(stdout=contents[name])

        with patch("multilingual_megatron.experiments.scaling.stage1_fit.execution.subprocess.run",
                   side_effect=git_show):
            self.assertEqual(source_sha256_at_commit("fixture-revision"), expected.hexdigest())

    def _chain(self, root: Path, *, resolved: bool) -> tuple[dict[str, object], dict[str, object]]:
        manifest = _seal({"kind": "grid"}, "manifest_sha256")
        manifest_path = root / "manifest.json"; write_json_once(manifest_path, manifest)
        result = _seal({"manifest_sha256": manifest["manifest_sha256"]}, "result_sha256")
        result_path = root / "result.json"; write_json_once(result_path, result)
        failed = _seal({"status": "failed", "reason": "nested warm start disagrees"}, "audit_sha256")
        failed_path = root / "audit.json"; write_json_once(failed_path, failed)
        plan = _seal({"actions": ["strict_warm_retry", "three_start_basin", "projected_direct"],
            "grid_manifest": _binding(manifest_path, "manifest_sha256"),
            "grid_result": _binding(result_path, "result_sha256"),
            "failed_audit": _binding(failed_path, "audit_sha256")}, "plan_sha256")
        plan_path = root / "plan.json"; write_json_once(plan_path, plan)
        actions = [
            {"action": "strict_warm_retry", "resolved": True},
            {"action": "three_start_basin", "resolved": resolved},
            {"action": "projected_direct", "resolved": True},
        ]
        run = _seal({"schema_version": RUN_SCHEMA, "protocol": PROTOCOL,
            "plan": _binding(plan_path, "plan_sha256"), "status": "complete", "reason": None,
            "grid_manifest_sha256": manifest["manifest_sha256"],
            "grid_result_sha256": result["result_sha256"],
            "actions": actions, "provenance": {}}, "run_sha256")
        run_path = root / "run.json"; write_json_once(run_path, run)
        accepted = _seal({"schema_version": ACCEPTANCE_SCHEMA, "protocol": PROTOCOL,
            "grid_manifest": _binding(manifest_path, "manifest_sha256"),
            "grid_result": _binding(result_path, "result_sha256"),
            "failed_audit": _binding(failed_path, "audit_sha256"),
            "run": _binding(run_path, "run_sha256"), "status": "complete"}, "acceptance_sha256")
        path = accepted_path(result, output_root=root)
        write_json_once(path, accepted)
        return result, failed

    def test_acceptance_requires_every_frozen_action_to_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); result, failed = self._chain(root, resolved=True)
            plan = {"actions": ["strict_warm_retry", "three_start_basin", "projected_direct"],
                "grid_manifest": read_json(accepted_path(result, output_root=root))["grid_manifest"],
                "grid_result": read_json(accepted_path(result, output_root=root))["grid_result"],
                "failed_audit": read_json(accepted_path(result, output_root=root))["failed_audit"]}
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.amendment.validate_recorded_provenance"), \
                    patch("multilingual_megatron.experiments.scaling.stage1_fit.amendment._validate_plan", return_value=plan):
                validate_accepted_amendment(result, failed, output_root=root)

    def test_nested_warm_input_is_recomputed_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = {"result_sha256": "s" * 64, "selected": {"shared_parameters":
                SharedParameters(.2, .01, .3, -.02, -.1, .04, .05, 0.0, -.03, 0.0).as_dict()}}
            source_path = root / "source.json"; write_json_once(source_path, source)
            grid_result = {"result_sha256": "g" * 64}; grid_result_path = root / "grid-result.json"
            write_json_once(grid_result_path, grid_result)
            fit = {"model": "M2", "fold_id": "fold", "objective": "huber", "delta": .1,
                "bounds": [], "bound_version": 0, "replicate_id": None, "outer_ftol": 1e-12,
                "outer_gtol": 1e-6, "outer_max_iterations": 10_000, "grid_version": "v1"}
            source_binding = {"kind": "warm", "path": str(source_path.resolve()),
                "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "result_sha256": source["result_sha256"]}
            grid = {"manifest_sha256": "m" * 64, "fit": {**fit, "action": "grid"},
                "parents": [source_binding], "input": {"rows": "x"}, "membership": {"kind": "test"},
                "selection_report": None}
            expected = nested_warm_start("M2", SharedParameters(**source["selected"]["shared_parameters"]))
            warm_fit = {**fit, "action": "warm", "start_mode": "explicit", "start_count": 1,
                "explicit_starts": [{"start_id": expected.start_id, "values": expected.values.tolist(),
                    "kind": expected.kind}]}
            grid_binding = {"kind": "grid", "path": str(grid_result_path.resolve()),
                "file_sha256": hashlib.sha256(grid_result_path.read_bytes()).hexdigest(),
                "result_sha256": grid_result["result_sha256"]}
            warm = {"fit": warm_fit, "parents": [grid_binding, source_binding], "input": grid["input"],
                "membership": grid["membership"], "selection_report": None}
            warm_result = {"selected": {"start_id": expected.start_id},
                "attempts": [{"start_id": expected.start_id}]}
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.campaign.scientific_result",
                    return_value=source):
                _validate_nested_warm_input(grid_manifest=grid, grid_result_path=grid_result_path,
                    grid_result=grid_result, warm_manifest=warm, warm_result=warm_result)
                bad = {**warm, "fit": {**warm_fit, "explicit_starts": [{"start_id": expected.start_id,
                    "values": [expected.values[0] + .01, *expected.values[1:]], "kind": expected.kind}]}}
                with self.assertRaisesRegex(AmendmentError, "derived nested warm start"):
                    _validate_nested_warm_input(grid_manifest=grid, grid_result_path=grid_result_path,
                        grid_result=grid_result, warm_manifest=bad, warm_result=warm_result)

    def test_result_binding_and_profile_owned_interpreter_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest_path = root / "manifest.json"; result_path = root / "result.json"
            manifest = {"manifest_sha256": "m" * 64, "fit": {"model": "M1", "fold_id": "fold",
                "replicate_id": None, "objective": "huber", "bound_version": 0, "bounds": [],
                "start_count": 1}, "input": {"fit_rows_content_sha256": "i" * 64},
                "membership": {"kind": "test"}, "parents": [], "selection_report": None,
                "review_only": False}
            write_json_once(manifest_path, manifest)
            receipt = _seal({"schema_version": "stage1_fit_complete/v2",
                "manifest_sha256": manifest["manifest_sha256"], "chunk_sha256s": ["c" * 64],
                "start_count": 1, "successful_starts": 1}, "receipt_sha256")
            receipt_path = root / "receipt.json"; write_json_once(receipt_path, receipt)
            result = {"manifest_sha256": manifest["manifest_sha256"], "manifest_path": str(manifest_path),
                "fit_rows_content_sha256": "i" * 64, "fit_identity": {key: manifest["fit"][key]
                    for key in ("model", "fold_id", "replicate_id", "objective", "bound_version")},
                "fit_bounds": [], "membership": manifest["membership"], "parents": [],
                "selection_report": None, "review_only": False, "status": "reduced", "selected": {"x": 1},
                "attempts": [{"success": True}], "receipt_path": str(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"]}
            _validate_result_manifest_binding(manifest_path=manifest_path, manifest=manifest,
                result_path_=result_path, result=result)
            with self.assertRaisesRegex(AmendmentError, "does not belong"):
                _validate_result_manifest_binding(manifest_path=manifest_path, manifest=manifest,
                    result_path_=result_path, result={**result, "fit_identity": {}})
        script = _render_slurm(Path("/tmp/plan with spaces.json"), {"execution": {"memory": "2G",
            "time_limit": "00:10:00", "partition": None, "account": None, "reservation": None,
            "python_executable": "/opt/bridge train/bin/python"}})
        self.assertIn("'/opt/bridge train/bin/python'", script)
        self.assertNotIn("/opt/conda", script)
        with self.assertRaisesRegex(AmendmentError, "profile-owned"):
            _render_slurm(Path("plan.json"), {"execution": {"memory": "2G", "time_limit": "00:10:00"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); result, failed = self._chain(root, resolved=False)
            plan = {"actions": ["strict_warm_retry", "three_start_basin", "projected_direct"],
                "grid_manifest": read_json(accepted_path(result, output_root=root))["grid_manifest"],
                "grid_result": read_json(accepted_path(result, output_root=root))["grid_result"],
                "failed_audit": read_json(accepted_path(result, output_root=root))["failed_audit"]}
            with patch("multilingual_megatron.experiments.scaling.stage1_fit.amendment.validate_recorded_provenance"), \
                    patch("multilingual_megatron.experiments.scaling.stage1_fit.amendment._validate_plan", return_value=plan), \
                    self.assertRaisesRegex(AmendmentError, "incomplete or unresolved"):
                validate_accepted_amendment(result, failed, output_root=root)


if __name__ == "__main__":
    unittest.main()
