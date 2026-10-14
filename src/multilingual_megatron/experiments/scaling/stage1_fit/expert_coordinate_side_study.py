"""Isolated, paired expert-coordinate sensitivity study; never promotes main results.

The private E proxy changes predictor features only, using the existing JH
coordinate. Physical experts and resource/exposure accounting remain untouched.
Numerics, ridge, bounds, language profiling and certification are reused intact.
"""
from __future__ import annotations

import argparse
from dataclasses import is_dataclass, replace
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from . import execution as core, fit, size_duration_ablations as ab
from .m6_size_duration_workflow import execution as e, model as m, planning as p

MODULE = "multilingual_megatron.experiments.scaling.stage1_fit.expert_coordinate_side_study"
SCHEMA = "stage1_expert_coordinate_side_study/v1"
PROTOCOL = {
    "models": ["M6-ND", "M5-ND", "J0-ND"], "coordinates": ["raw", "jointmoe_fixed"],
    "folds": list(p.FOLDS), "starts_per_cell": 48, "chunk_size": 2,
    "E_start": 2.0732, "E_max": 290.4521, "anchor_experts": [1, 32],
    "transform": "exp(expert_log_coordinate('JH', physical_E)); exact endpoint anchoring",
    "refit": "all free shared and language coefficients on original training rows only",
    "starts": "paired exact parent grid; eight anchors times six zeta starts; no new search",
    "audits_per_cell": 4, "audit_order": "three best successful starts and independent joint from best",
    "gradient_tolerance": 1e-8, "objective_tolerance": 1e-6, "prediction_tolerance": 1e-4,
    "rank_relative_tolerance": 1e-8, "minimum_successful_starts": 3,
    "known_minimum_guard": "include every feasible terminal point, even failed optimizers, and raw parent winner",
    "pilot": "first start, largest-size fold, each model and coordinate; reused in grid",
    "pilot_projected_core_hours_limit": 100., "pilot_safety_factor": 2.,
    "bootstrap_replicates": 10000, "bootstrap_seed": 20260905,
    "score_gate": "only fully certified three-fold model-coordinate variants; all unresolved cells disclosed",
    "evaluation": "all holdouts, paired mean RMSE, horizon/E/K residuals, E32 long tails, K200 decision regret",
    "decision_support": "81 common-support compute points; original 12/50 language cohorts; all three truth interpolants",
    "evidence": "separate post-diagnosis development sensitivity; already inspected holdouts; no main-result promotion",
    "retries": 0, "holdout_selection": False, "new_model_training": False,
}


def coordinate_problem(problem, coordinate):
    """Return an isolated numerical feature view, never a modified physical row."""
    if coordinate == "raw":
        return problem
    if coordinate != "jointmoe_fixed":
        raise ValueError("unknown expert coordinate")
    experts = np.asarray(problem.E, dtype=float)
    if not np.isfinite(experts).all() or np.any(experts < 1):
        raise ValueError("physical expert counts must be finite and at least one")
    proxy = np.exp(fit.expert_log_coordinate("JH", experts))
    proxy = np.where((experts == 1) | (experts == 32), experts, proxy)
    if is_dataclass(problem):
        return replace(problem, E=proxy)
    return SimpleNamespace(**(vars(problem) | {"E": proxy}))


def cell_profile(plan, cell):
    manifest = plan["folds"][cell["fold"]]["manifest"]
    problem = core.load_problem(manifest)
    profile = m.Profile(coordinate_problem(problem, cell["coordinate"]), manifest["fit"]["delta"])
    return profile, tuple(ab.AblationBound(**b) for b in cell["bounds"])


def health(profile, record, bounds, name):
    return (m.numerical_health(profile, record, bounds) if name == "M6-ND"
            else ab.health(profile, record, bounds, name))


def known_minimum_ok(selected, records):
    finite = [v["objective"] for v in records if v.get("feasible", True)
              and np.isfinite(v.get("objective", np.nan))]
    best = min(finite, default=float("inf"))
    relative = (selected["objective"]-best)/max(abs(best), 1e-15)
    return {"best_known_feasible_objective": best, "relative_gap": relative,
            "passed": bool(relative <= PROTOCOL["objective_tolerance"])}


def validate_plan(plan):
    if plan["schema_version"] != SCHEMA or plan["protocol"] != PROTOCOL:
        raise ValueError("side-study protocol changed")
    cells = plan["cells"]
    expected = [(n, f, c) for n in PROTOCOL["models"] for f in p.FOLDS for c in PROTOCOL["coordinates"]]
    if [(v["model"], v["fold"], v["coordinate"]) for v in cells] != expected:
        raise ValueError("side-study cell coverage changed")
    for i, cell in enumerate(cells):
        if cell["cell_id"] != i or len(cell["starts"]) != 48 or len({s["start_id"] for s in cell["starts"]}) != 48:
            raise ValueError("side-study start coverage changed")
        if i % 2 and any(cell[k] != cells[i-1][k] for k in ("starts", "bounds")):
            raise ValueError("raw/transformed pair differs beyond expert coordinate")
    output = Path(plan["output_root"])
    if output.parent != Path(plan["repository_root"])/"result_analysis" or not output.name.startswith("stage1_expert_coordinate_side_study_"):
        raise ValueError("output not owned by side study")
    for binding in plan["protected"]:
        if Path(binding["path"]).is_relative_to(output):
            raise ValueError("protected established evidence overlaps side-study output")


def load(path):
    plan = p.read_sealed(path)
    validate_plan(plan)
    for binding in [*plan["protected"], *plan["snapshot_sources"], plan["archive"], *plan["scripts"]]:
        p.diagnosis.bound_file(binding)
    if p.environment() != plan["environment"]:
        raise ValueError("numerical environment changed")
    expected = Path(plan["runtime_root"])/Path(plan["module_relative_path"])
    if Path(__file__).resolve() != expected:
        raise ValueError("execute the side study using its frozen run.sh")
    return plan


def receipt(plan, relative):
    value = p.read_sealed(Path(plan["output_root"])/relative)
    if value["plan_sha256"] != plan["sha256"]:
        raise ValueError("receipt belongs to another study")
    return value


def prepare(profile_path):
    from .atlas_comparison_workflow.snapshot import freeze
    profile_path = Path(profile_path).resolve()
    config = json.loads(profile_path.read_text())
    ab_root = p.diagnosis.bound_file(config["parent_ablation"]).parent
    parent_ab, parent = ab.load(ab_root/"plan.json")
    root = p.repository_root()
    output = (root/config["output_root"]).resolve()
    if output.parent != root/"result_analysis" or not output.name.startswith("stage1_expert_coordinate_side_study_") or output.exists():
        raise ValueError("preparation requires a new, isolated side-study directory")
    if str(Path(config["python_executable"]).resolve()) != p.environment()["executable"]:
        raise ValueError("use configured bridge-train interpreter")
    parent_reduction = p.read_sealed(Path(parent["output_root"])/"reduction.json")
    main_report = p.read_sealed(Path(parent["output_root"])/"reports/report.json")
    ab_report = p.read_sealed(ab_root/"report.json")
    protected = [*parent["bindings"], *parent["sources"], parent_ab["parent"], config["parent_ablation"],
                 *parent_ab["sources"], *parent_ab["bindings"], p.binding(profile_path),
                 p.binding(Path(parent["output_root"])/"reduction.json"),
                 p.binding(Path(parent["output_root"])/"reports/report.json"),
                 p.binding(ab_root/"report.json"),
                 *main_report["outputs"], ab_report["predictions"], *ab_report["cells"], *config["protected_results"]]
    cells = []
    for name in PROTOCOL["models"]:
        for fold in p.FOLDS:
            if name == "M6-ND":
                starts = [s for task in parent["tasks"] for s in task["starts"] if s["fold"] == fold]
                bounds = parent["folds"][fold]["bounds"]
                winner = parent_reduction["folds"][fold]["selected"]
            else:
                task, = [t for t in parent_ab["tasks"] if (t["model"], t["fold"]) == (name, fold)]
                starts, bounds = task["starts"], task["bounds"]
                old = p.read_sealed(ab_root/f"cell-{task['task_id']}.json")
                if old["status"] != "healthy":
                    raise ValueError("parent ablation not certified")
                winner = old["selected"]
            for coordinate in PROTOCOL["coordinates"]:
                cells.append(dict(cell_id=len(cells), model=name, fold=fold, coordinate=coordinate,
                                  starts=starts, bounds=bounds, raw_parent_selected=winner))
    # Decision inputs use existing, verified physical counts and cohort definitions.
    # This read checks old predictions only; no new heldout prediction is made.
    from result_analysis.compute_allocation import policy_validation as story
    _, _, rows, _, cohorts, smooth, story_bindings = story.load_evidence(ab_root)
    protected += story_bindings
    protected = list({b["path"]: b for b in protected}.values())
    for binding in protected:
        p.diagnosis.bound_file(binding)
    output.mkdir()
    rows.to_parquet(output/"decision_inputs.parquet", index=False)
    extra = [Path(__file__), Path(__file__).with_name("expert_coordinate_reporting.py"),
             root/"tests/test_stage1_expert_coordinate_side_study.py",
             root/"result_analysis/compute_allocation/policy_validation.py"]
    extra += [root/"result_analysis"/name for name in (
        "learning_curves.py", "evidence.py",
        "scaling_law_fit/__init__.py", "scaling_law_fit/residuals.py",
        "capacity_response/__init__.py", "capacity_response/matched_exposure.py", "capacity_response/tables.py",
        "language_preservation/__init__.py", "language_preservation/diagnostics.py",
        "compute_allocation/__init__.py")]
    # A self-contained snapshot permits unrelated checkout edits while jobs run.
    sources = [Path(b["path"]) for b in parent["sources"]] + list(Path(__file__).parent.rglob("*.py")) + extra
    runtime, snapshot_sources, archive = freeze(output, sources, root)
    run = output/"run.sh"
    core.write_text_once(run, "\n".join(["#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(str(runtime))}",
        "export PYTHONPATH=src:.", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        f"exec {shlex.quote(config['python_executable'])} -m {MODULE} --plan {shlex.quote(str(output/'plan.json'))} \"$@\"", ""]))
    scripts = [p.binding(run)]
    (output/"logs").mkdir()
    for action, array in (("pilot", "0-5%6"), ("worker", f"0-431%{config['max_parallel']}"), ("audit", f"0-71%{config['max_parallel']}")):
        script = output/f"{action}.sbatch"
        core.write_text_once(script, "\n".join(["#!/bin/bash", f"#SBATCH --job-name=expert-side-{action}",
            f"#SBATCH --array={array}", f"#SBATCH --account={config['account']}", f"#SBATCH --partition={config['partition']}",
            "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={config['memory']}",
            f"#SBATCH --time={config['time_limit']}", f"#SBATCH --output={output}/logs/%A_%a.out",
            "set -euo pipefail", f"bash {shlex.quote(str(run))} {action} --task \"$SLURM_ARRAY_TASK_ID\"", ""]))
        subprocess.run(["bash", "-n", str(script)], check=True)
        scripts.append(p.binding(script))
    plan = dict(schema_version=SCHEMA, protocol=PROTOCOL, repository_root=str(root), output_root=str(output),
        runtime_root=str(runtime), module_relative_path=str(Path(__file__).relative_to(root)), cells=cells,
        folds=parent["folds"], protected=protected, snapshot_sources=snapshot_sources, archive=archive, scripts=scripts,
        environment=p.environment(), base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        decision_inputs=p.binding(output/"decision_inputs.parquet"), cohorts=cohorts,
        smoothing_penalty=smooth["smoothing_strengths"]["smooth"], historical_results_changed=False,
        raw_parent_metrics=ab_report["statistics"]["metrics"], new_heldout_scores_computed=False)
    validate_plan(plan)
    return p.write_sealed(output/"plan.json", plan)


def attempt(plan, cell_id, index):
    cell = plan["cells"][cell_id]
    path = f"attempts/{cell_id:02d}-{index:02d}.json"
    if (Path(plan["output_root"])/path).exists():
        value = receipt(plan, path)
    else:
        context = e.runtime_context()
        profile, bounds = cell_profile(plan, cell)
        result = e.run_attempt(profile, cell["starts"][index], bounds)
        # Failed terminals can still expose a better feasible training solution.
        x = np.asarray(result.get("coordinates", []))
        feasible = x.shape == (15,) and np.isfinite(x).all() and all(b.lower <= v <= b.upper for v, b in zip(x, bounds))
        result["feasible"] = bool(feasible)
        value = p.write_sealed(Path(plan["output_root"])/path, dict(plan_sha256=plan["sha256"],
            cell_id=cell_id, start_index=index, start=cell["starts"][index], result=result, runtime=context))
    if (value["cell_id"], value["start_index"], value["start"]) != (cell_id, index, cell["starts"][index]):
        raise ValueError("attempt identity changed")
    return value


def worker(plan_path, task_id, pilot=False):
    plan = load(plan_path)
    e.runtime_context()
    if pilot:
        if not 0 <= task_id < 6:
            raise ValueError("invalid pilot task")
        return attempt(plan, (task_id//2)*6+task_id%2, 0)
    if not 0 <= task_id < 432 or receipt(plan, "pilot_gate.json")["status"] != "passed":
        raise ValueError("grid requires valid task and successful training-only pilot")
    return {"attempt_sha256": [attempt(plan, task_id//24, task_id%24*2+j)["sha256"] for j in (0, 1)]}


def pilot_gate(plan_path):
    plan = load(plan_path)
    values = [receipt(plan, f"attempts/{i:02d}-00.json") for i in (0, 1, 6, 7, 12, 13)]
    records = [v["result"] for v in values]
    estimate = max(v["runtime_seconds"] for v in records)*936/3600*PROTOCOL["pilot_safety_factor"]
    passed = all(v["success"] for v in records) and estimate <= PROTOCOL["pilot_projected_core_hours_limit"]
    return p.write_sealed(Path(plan["output_root"])/"pilot_gate.json", dict(plan_sha256=plan["sha256"],
        status="passed" if passed else "blocked", runtime_seconds=[v["runtime_seconds"] for v in records],
        successful=sum(v["success"] for v in records), projected_core_hours=estimate,
        heldout_scores_computed=False, attempt_sha256=[v["sha256"] for v in values]))


def reduce(plan_path):
    plan = load(plan_path)
    if receipt(plan, "pilot_gate.json")["status"] != "passed":
        raise ValueError("pilot has not passed")
    cells, tasks = [], []
    for cell in plan["cells"]:
        receipts = [receipt(plan, f"attempts/{cell['cell_id']:02d}-{i:02d}.json") for i in range(48)]
        for i, value in enumerate(receipts):
            if (value["cell_id"], value["start_index"], value["start"]) != (cell["cell_id"], i, cell["starts"][i]):
                raise ValueError("attempt identity mismatch")
        records = [v["result"] for v in receipts]
        successes = sorted((v for v in records if v["success"] and v["feasible"]), key=lambda v: (v["objective"], v["start_id"]))
        selected = successes[0] if successes else None
        known = records + ([cell["raw_parent_selected"]] if cell["coordinate"] == "raw" else [])
        guard = known_minimum_ok(selected, known) if selected else {"passed": False}
        cells.append(dict(cell_id=cell["cell_id"], selected=selected, attempts_sha256=[v["sha256"] for v in receipts],
                          known_minimum=guard, successful=len(successes)))
        # Four stable task IDs per cell, including explicit unresolved receipts.
        for j in range(4):
            start = successes[j] if len(successes) >= 3 and j < 3 else selected
            tasks.append(dict(task_id=len(tasks), cell_id=cell["cell_id"], kind="direct" if j == 3 else "basin",
                              start=start, eligible=len(successes) >= 3 and guard["passed"]))
    return p.write_sealed(Path(plan["output_root"])/"reduction.json", dict(plan_sha256=plan["sha256"],
        cells=cells, audit_tasks=tasks, heldout_scores_computed=False))


def audit(plan_path, task_id):
    plan = load(plan_path)
    if not 0 <= task_id < 72:
        raise ValueError("invalid audit task")
    reduction = receipt(plan, "reduction.json")
    task = reduction["audit_tasks"][task_id]
    path = Path(plan["output_root"])/f"audits/{task_id:02d}.json"
    if path.exists():
        value = receipt(plan, f"audits/{task_id:02d}.json")
        if value["reduction_sha256"] != reduction["sha256"] or value["task"] != task:
            raise ValueError("audit identity changed")
        return value
    context = e.runtime_context()
    result, equivalent, numerical = {}, {}, {}
    status = "unresolved"
    began = time.monotonic()
    if task["eligible"]:
        cell = plan["cells"][task["cell_id"]]
        profile, bounds = cell_profile(plan, cell)
        try:
            result = m.Direct(profile, task["start"], bounds).run() if task["kind"] == "direct" else profile.run(task["start"]["coordinates"], bounds)
            selected = reduction["cells"][task["cell_id"]]["selected"]
            equivalent = m.equivalence(selected, result, profile.problem)
            numerical = health(profile, result, bounds, cell["model"])
            selected_health = health(profile, selected, bounds, cell["model"])
            status = "healthy" if equivalent["equivalent"] and numerical["healthy"] and selected_health["healthy"] else "unresolved"
            numerical["selected_healthy"] = selected_health["healthy"]
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            result = {"success": False, "message": f"{type(error).__name__}: {error}"}
    return p.write_sealed(path, dict(plan_sha256=plan["sha256"], reduction_sha256=reduction["sha256"], task=task,
        result=result, equivalence=equivalent, health=numerical, status=status, runtime=context,
        runtime_seconds=time.monotonic()-began, heldout_scores_computed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("prepare").add_argument("--profile", type=Path, required=True)
    for name in ("pilot", "worker", "audit"):
        subs.add_parser(name).add_argument("--task", type=int, required=True)
    for name in ("pilot-gate", "reduce", "report", "validate"):
        subs.add_parser(name)
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(args.profile)
    elif args.plan is None:
        parser.error("--plan required")
    elif args.action in ("pilot", "worker"):
        value = worker(args.plan, args.task, args.action == "pilot")
    elif args.action == "audit":
        value = audit(args.plan, args.task)
    elif args.action == "pilot-gate":
        value = pilot_gate(args.plan)
    elif args.action == "reduce":
        value = reduce(args.plan)
    elif args.action == "report":
        from .expert_coordinate_reporting import report
        value = report(args.plan)
    else:
        plan = load(args.plan)
        value = {"status": "verified", "plan_sha256": plan["sha256"], "protected_files": len(plan["protected"])}
    print(json.dumps({k: value[k] for k in ("sha256", "status", "projected_core_hours", "successful", "plan_sha256", "protected_files") if k in value}, indent=2))


if __name__ == "__main__":
    main()
