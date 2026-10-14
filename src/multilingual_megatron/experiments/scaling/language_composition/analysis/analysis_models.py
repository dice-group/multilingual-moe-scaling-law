"""Adapters for existing fitted laws; no fitting or selection operations."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from multilingual_megatron.experiments.scaling.stage1_fit import fit
from multilingual_megatron.experiments.scaling.stage1_fit.baseline_refit_workflow.contracts import BaselineRefitError, verify_seal
from multilingual_megatron.experiments.scaling.stage1_fit.data import canonical_json_sha256
from multilingual_megatron.experiments.scaling.stage1_fit.m6_size_duration_workflow import model as nd


def _sealed_read(read, path, field="sha256", expected=None):
    value = read(path, expected)
    try:
        verify_seal(value, field, label=str(path))
    except BaselineRefitError as error:
        raise ValueError(str(error)) from error
    return value


def _binding(bindings, path):
    # Match the unique filename, then bind the configured file by CONTENT hash.
    # This permits moving an unchanged input tree without trusting a new file.
    matches = [item for item in bindings if Path(item["path"]).name == Path(path).name]
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous original binding: {path}")
    return matches[0]["file_sha256"]


def _same_hash(actual, expected, label):
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"original binding mismatch: {label}")


def frozen_models(spec, read):
    """Resolve the exact retained fits through their original source records."""
    paths = spec["inputs"]
    ablation_plan = _sealed_read(read, f'{paths["ablation_root"]}/plan.json')
    ablation_report = _sealed_read(read, f'{paths["ablation_root"]}/report.json')
    _same_hash(ablation_report["plan_sha256"], ablation_plan["sha256"], "ablation report plan")
    if ablation_report["status"] != "complete" or len(ablation_report["cells"]) != 6:
        raise ValueError("incomplete retained ablation report")
    plan = _sealed_read(read, paths["nd_plan"], expected=ablation_plan["parent"]["file_sha256"])
    main_binding = ablation_report["main_report"]
    main_report = _sealed_read(read, main_binding["path"], expected=main_binding["file_sha256"])
    if main_report["status"] != "complete":
        raise ValueError("incomplete retained ND report")
    _same_hash(main_report["plan_sha256"], plan["sha256"], "ND report plan")
    reduction = _sealed_read(read, paths["nd_reduction"])
    _same_hash(reduction["plan_sha256"], plan["sha256"], "ND reduction plan")
    _same_hash(reduction["sha256"], main_report["reduction_sha256"], "ND report reduction")
    records = []
    for fold in spec["folds"]:
        component = plan["folds"][fold]["manifest"]["m5_m6_composition"]
        receipt = _sealed_read(read, paths["component_receipt"], "receipt_sha256",
                               component["receipt_file_sha256"])
        _same_hash(receipt["receipt_sha256"], component["receipt_sha256"], "component receipt")
        for key in ("language_metadata", "composition_sets"):
            _same_hash(canonical_json_sha256(receipt[key]), receipt[f"{key}_sha256"], key)
        _same_hash(receipt["composition_sets_sha256"], component["composition_sets_sha256"],
                   "original composition sets")
        records.extend([
            {"model": "M6-ND", "fold": fold,
             "selected": reduction["folds"][fold]["selected"]},
            {"model": "M6", "fold": fold,
             "selected": plan["folds"][fold]["m6_selected"]},
        ])
    for index in range(6):
        path = f'{paths["ablation_root"]}/cell-{index}.json'
        value = _sealed_read(read, path, expected=_binding(ablation_report["cells"], path))
        _same_hash(value["plan_sha256"], ablation_plan["sha256"], "ablation cell plan")
        if value["status"] != "healthy":
            raise ValueError("unhealthy retained ablation")
        records.append({k: value[k] for k in ("model", "fold", "selected")})
    baseline = _sealed_read(read, paths["baseline_report"], "report_sha256",
                            _binding(ablation_plan["bindings"], paths["baseline_report"]))
    if baseline["status"] != "complete":
        raise ValueError("incomplete retained baseline report")
    for source in baseline["source_results"]:
        if source["model"] in {"C0", "J0", "M0", "M1"}:
            value = _sealed_read(read, source["path"], "result_sha256", source["file_sha256"])
            _same_hash(value["result_sha256"], source["result_sha256"], "baseline result")
            identity = value["fit_identity"]
            if value["status"] != "reduced" or (identity["model"], identity["fold_id"]) != \
                    (source["model"], source["fold"]):
                raise ValueError("retained baseline fit identity mismatch")
            records.append({"model": source["model"], "fold": source["fold"],
                            "selected": value["selected"]})
    expected = {(m, f) for m in spec["models"] for f in spec["folds"]}
    if len(records) != len(expected) or {(r["model"], r["fold"]) for r in records} != expected:
        raise ValueError("incomplete or duplicate frozen fit matrix")
    # Preserve every coefficient for complete historical replay, not optimizer histories.
    for record in records:
        selected = record["selected"]
        coefficients = {c["language"]: c for c in selected["language_coefficients"]}
        if not set(spec["shared_languages"]) <= coefficients.keys():
            raise ValueError("shared language lacks a historical coefficient")
        record["selected"] = {k: selected[k] for k in
                              ("coordinates", "shared_parameters") if k in selected}
        record["selected"]["language_coefficients"] = [
            {k: coefficients[language][k] for k in ("language", "L_inf", "A", "B")}
            for language in sorted(coefficients)]
    return sorted(records, key=lambda r: (r["model"], r["fold"]))


def predict(rows, record):
    """Return predictions in the ORIGINAL fitted adjusted-loss coordinate."""
    model, selected = record["model"], record["selected"]
    if rows.empty or not np.isfinite(rows.D_t).all() or (rows.D_t <= 0).any():
        raise ValueError("frozen-law evaluation requires positive target exposure")
    coefficients = {c["language"]: c for c in selected["language_coefficients"]}
    if not set(rows.language) <= coefficients.keys():
        raise ValueError("missing frozen language coefficient")
    if model.endswith("-ND"):
        problem = SimpleNamespace(N=rows.N_act.to_numpy(), D=rows.D_t.to_numpy(),
            E=rows.E.to_numpy(), K=rows.K.to_numpy(), language_ids=tuple(rows.language),
            D_family=rows.D_family.to_numpy(), D_script=rows.D_script.to_numpy(),
            D_other=rows.D_other.to_numpy())
        output = nd.predict(problem, selected)
    else:
        cs = [coefficients[l] for l in rows.language]
        components = {k: rows[k] for k in ("D_family", "D_script", "D_other")} if model == "M6" else {}
        output = fit.predict(model, fit.SharedParameters(**selected["shared_parameters"]),
            rows.N_act, rows[fit.data_column(model)], rows.E, rows.K,
            [c["L_inf"] for c in cs], [c["A"] for c in cs], [c["B"] for c in cs], **components)
    if not np.isfinite(output).all():
        raise ValueError("non-finite frozen prediction")
    return np.asarray(output)
