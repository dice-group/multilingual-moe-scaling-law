"""Fail-closed provenance checks for retained composition-analysis coefficients."""
import copy
import unittest

from multilingual_megatron.experiments.scaling.language_composition.analysis.analysis_models import frozen_models
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.contracts import seal
from multilingual_megatron.experiments.scaling.stage1_fit.data import canonical_json_sha256


class FrozenInputs:
    """Small original-report graph; the reader enforces serialized-content hashes."""

    def __init__(self):
        self.files = {}
        self.fields = {}
        self.spec = {
            "models": ["M6-ND", "M5-ND", "M6", "J0-ND", "J0", "C0", "M0", "M1"],
            "folds": ["largest_N_act", "E32", "K200_common_languages"],
            "shared_languages": ["eng_Latn"],
            "inputs": {"nd_plan": "nd/plan.json", "nd_reduction": "nd/reduction.json",
                       "ablation_root": "ablation", "baseline_report": "baseline/report.json",
                       "component_receipt": "components/receipt.json"},
        }
        self.selected = {"coordinates": [1.0, 2.0], "shared_parameters": {"alpha": 0.5},
                         "language_coefficients": [{"language": "eng_Latn", "L_inf": 1.,
                                                    "A": 2., "B": 3., "diagnostic": 9.}],
                         "optimizer_history": "not part of frozen predictions"}
        receipt = {"language_metadata": {"eng_Latn": {"family": "Indo-European", "script": "Latn"}},
                   "composition_sets": {"K12": ["eng_Latn"]}}
        receipt.update({f"{key}_sha256": canonical_json_sha256(value)
                        for key, value in list(receipt.items())})
        self.put("components/receipt.json", receipt, "receipt_sha256")
        sources = []
        for model in ("C0", "J0", "M0", "M1"):
            for fold in self.spec["folds"]:
                path = f"baseline/{model}-{fold}.json"
                self.put(path, {"status": "reduced", "fit_identity": {"model": model, "fold_id": fold},
                                "selected": self.selected}, "result_sha256")
                sources.append({**self.binding(path), "model": model, "fold": fold,
                                "result_sha256": self.files[path]["result_sha256"]})
        self.put("baseline/report.json", {"status": "complete", "source_results": sources}, "report_sha256")
        component = {"receipt_file_sha256": self.binding("components/receipt.json")["file_sha256"],
                     "receipt_sha256": self.files["components/receipt.json"]["receipt_sha256"],
                     "composition_sets_sha256": receipt["composition_sets_sha256"]}
        self.put("nd/plan.json", {"folds": {fold: {"manifest": {"m5_m6_composition": component},
                                                        "m6_selected": self.selected}
                                               for fold in self.spec["folds"]}})
        self.put("nd/reduction.json", {"plan_sha256": self.files["nd/plan.json"]["sha256"],
                                      "folds": {fold: {"selected": self.selected}
                                                for fold in self.spec["folds"]}})
        self.put("nd/report.json", {"status": "complete", "plan_sha256": self.files["nd/plan.json"]["sha256"],
                                   "reduction_sha256": self.files["nd/reduction.json"]["sha256"]})
        self.put("ablation/plan.json", {"parent": self.binding("nd/plan.json"),
                                        "bindings": [self.binding("baseline/report.json")]})
        for index, (model, fold) in enumerate((model, fold) for model in ("J0-ND", "M5-ND")
                                             for fold in self.spec["folds"]):
            self.put(f"ablation/cell-{index}.json", {
                "model": model, "fold": fold, "status": "healthy", "selected": self.selected,
                "plan_sha256": self.files["ablation/plan.json"]["sha256"]})
        self.put("ablation/report.json", {"status": "complete", "main_report": self.binding("nd/report.json"),
            "plan_sha256": self.files["ablation/plan.json"]["sha256"],
            "cells": [self.binding(f"ablation/cell-{index}.json") for index in range(6)]})

    def put(self, path, value, field="sha256"):
        self.fields[path] = field
        self.files[path] = seal(copy.deepcopy(value), field)

    def update(self, path, **changes):
        value = copy.deepcopy(self.files[path])
        value.pop(self.fields[path])
        value.update(changes)
        self.put(path, value, self.fields[path])

    def binding(self, path):
        return {"path": path, "file_sha256": canonical_json_sha256(self.files[path])}

    def read(self, path, expected=None):
        value = self.files[str(path)]
        if expected is not None and canonical_json_sha256(value) != expected:
            raise ValueError(f"input hash mismatch: {path}")
        return copy.deepcopy(value)


class FrozenBindingTests(unittest.TestCase):
    def setUp(self):
        self.inputs = FrozenInputs()

    def resolve(self):
        return frozen_models(self.inputs.spec, self.inputs.read)

    def test_valid_original_graph_preserves_all_selected_coefficients(self):
        records = self.resolve()
        self.assertEqual(len(records), 24)
        self.assertEqual(records, sorted(records, key=lambda row: (row["model"], row["fold"])))
        expected = {key: self.inputs.selected[key] for key in ("coordinates", "shared_parameters")}
        expected["language_coefficients"] = [{"language": "eng_Latn", "L_inf": 1., "A": 2., "B": 3.}]
        self.assertTrue(all(row["selected"] == expected for row in records))

    def test_unsealed_root_reports_and_reduction_fail(self):
        for path in ("ablation/plan.json", "ablation/report.json", "nd/reduction.json"):
            with self.subTest(path=path):
                self.inputs = FrozenInputs()
                self.inputs.files[path]["new_field"] = "unsealed mutation"
                with self.assertRaisesRegex(ValueError, "invalid.*sha256"):
                    self.resolve()

    def test_resealed_sources_cannot_replace_original_file_bindings(self):
        for path in ("nd/plan.json", "nd/report.json", "ablation/cell-0.json",
                     "baseline/report.json", "baseline/C0-largest_N_act.json", "components/receipt.json"):
            with self.subTest(path=path):
                self.inputs = FrozenInputs()
                self.inputs.update(path, new_field="self-consistent but not the original file")
                with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                    self.resolve()

    def test_resealed_reduction_must_match_completed_original_report(self):
        self.inputs.update("nd/reduction.json", new_field="changed reduction")
        with self.assertRaisesRegex(ValueError, "ND report reduction"):
            self.resolve()

    def test_plan_links_are_checked_even_with_valid_internal_seals(self):
        for path, message in (("ablation/report.json", "ablation report plan"),
                              ("nd/reduction.json", "ND reduction plan")):
            with self.subTest(path=path):
                self.inputs = FrozenInputs()
                self.inputs.update(path, plan_sha256="0" * 64)
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve()

    def test_cell_plan_link_checked_after_rebinding(self):
        self.inputs.update("ablation/cell-0.json", plan_sha256="0" * 64)
        cells = self.inputs.files["ablation/report.json"]["cells"]
        cells[0] = self.inputs.binding("ablation/cell-0.json")
        self.inputs.update("ablation/report.json", cells=cells)
        with self.assertRaisesRegex(ValueError, "ablation cell plan"):
            self.resolve()

    def test_ambiguous_original_cell_binding_fails(self):
        cells = self.inputs.files["ablation/report.json"]["cells"]
        cells[1] = dict(cells[0])
        self.inputs.update("ablation/report.json", cells=cells)
        with self.assertRaisesRegex(ValueError, "missing or ambiguous original binding"):
            self.resolve()

    def test_incomplete_report_fails(self):
        self.inputs.update("ablation/report.json", status="running")
        with self.assertRaisesRegex(ValueError, "incomplete retained ablation report"):
            self.resolve()


if __name__ == "__main__":
    unittest.main()
