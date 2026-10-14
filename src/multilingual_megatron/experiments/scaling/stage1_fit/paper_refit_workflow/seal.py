"""Seal a complete, reviewed paper release without discarding adverse outcomes."""
from pathlib import Path

from .. import execution as io
from . import inputs, protocol, runner

GATES = ("solver_recovery", "composition_and_uncertainty", "interaction_diagnostics",
         "allocation_and_regret", "measured_evidence_replay", "manuscript_claim_review")


def seal(path, review_path, *, report_path=None):
    path, review_path = Path(path).resolve(), Path(review_path).resolve()
    plan = runner.load(path); root = path.parent
    if (set(plan["models"]) != set(protocol.MODELS) or plan["profile"].get("full_paper_policy") != "lbfgsb_box"
            or not plan["profile"].get("paired_controls", True)):
        raise ValueError("paper seal requires the entire full-paper inventory")
    report_path = Path(report_path).resolve() if report_path else root / "report/report.json"
    if not report_path.is_relative_to(root):
        raise ValueError("report must be inside the run directory")
    report = inputs.read_sealed(report_path)
    if report["plan_sha256"] != plan["sha256"] or not report["all_cells_certified"] or not report["all_scoring_complete"]:
        raise ValueError("incomplete or uncertified results cannot be sealed as final paper values")
    for entry in report["outputs"]:
        inputs.local_path(root, entry)
    from .recovery import verified_overrides
    overrides = verified_overrides(path, inputs.local_path(root, report["recovery"]) if report.get("recovery") else None)
    diagnostics = io.read_json(report_path.parent / "fit_diagnostics.json")["fits"]
    target = [d for d in diagnostics if d["model"] == "RN14-stored-reuse" and d["arm"] == "lbfgsb_box"]
    if len(target) != len(protocol.folds("RN14-stored-reuse")) or not all(d["positive_effective_exponents"] for d in target):
        raise ValueError("relaxed RN14 did not recover positive effective exponents in every fit")
    review = io.read_json(review_path)
    if review.get("report_sha256") != report["sha256"] or not review.get("story_unchanged"):
        raise ValueError("review must bind this report and explicitly assess the paper claims")
    if set(review.get("gates", {})) != set(GATES):
        raise ValueError("all downstream and claim-review gates are required")
    evidence = []
    for name, gate in review["gates"].items():
        if gate.get("passed") is not True or not gate.get("rationale") or not gate.get("evidence"):
            raise ValueError(f"unresolved paper adoption gate: {name}")
        for item in gate["evidence"]:
            inputs.local_path(root, item)
            evidence.append(item)
    # Task receipts are independently sealed and bound to this exact plan.
    receipts = []
    for task in plan["tasks"]:
        runner.receipt(path, plan, task["id"])
        receipt_path = overrides[task['id']][0] if task['id'] in overrides else root / "results" / f"{task['id']:05d}.json"
        receipts.append(inputs.local_binding(receipt_path, root))
    value = inputs.sealed(dict(schema="stage1_paper_release/v1", plan_sha256=plan["sha256"],
        report_sha256=report["sha256"], report=inputs.local_binding(report_path, root),
        review=inputs.binding(review_path), evidence=evidence,
        receipts=receipts, manuscript_automatically_changed=False,
        scope="reviewed new fitting values; historical outcomes and selection history retained"))
    io.write_json_once(root / "paper_release_seal.json", value)
    return value
