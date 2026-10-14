"""Adapters to the existing numerical procedures; no scheduler or test losses."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .. import execution as io
from .. import fit
from ..law_search import data
from ..law_search import model as shared
from .. import duration_ablation as ablation
from . import inputs, protocol
from ..atlas_comparison_workflow import model as native, numerics, recovery_numerics
from ..baseline_refit_workflow import recovery_numerics as conditioned
from ..coverage_laws import model as repetition, numerics as repetition_numerics, replay
from ..law_search.replay import native as native_replay
from ..law_search.replay import shared_equation
from ..shared_numerics import fit_cell
from ..m6_size_duration_workflow import model as duration
from ..problems import problem


def primary_problem(rows):
    return fit.ProjectedProblem.from_arrays(y=rows.adjusted_loss, N=rows.N_act,
        D=rows.D_t, E=rows.E, K=rows.K, language_ids=tuple(rows.language),
        weights=fit.trajectory_weights(rows.trajectory_id.tolist()), unigram_nll=rows.unigram_nll)


def import_contracts(comparison_path, ablation_path, input_path):
    """Import bounds and training-only start coordinates once, not fit outcomes."""
    manifest = inputs.load(input_path)
    sources = []

    def read(path, digest=None):
        path = Path(path).resolve()
        entry = inputs.binding(path)
        if digest is not None and entry["sha256"] != digest:
            raise ValueError(f"changed numerical contract: {path}")
        sources.append(entry)
        return json.loads(path.read_text())

    def membership(value, fold):
        for key in ("train_trajectory_ids", "test_trajectory_ids", "score_languages"):
            if value[key] != manifest["memberships"][fold][key]:
                raise ValueError("numerical contract belongs to a different split")

    comparison = read(comparison_path)
    contracts = {}
    for item in comparison["source_results"]:
        name, fold = item["model"], item["fold"]
        if name not in ("J0", "M2") or fold not in inputs.PRIMARY:
            continue
        result = read(item["path"], item["file_sha256"])
        source = read(result["manifest_path"])
        io._verify_seal(source, "manifest_sha256")
        if source["manifest_sha256"] != result["manifest_sha256"]:
            raise ValueError("result and numerical manifest disagree")
        membership(source["membership"], fold)
        rows, _, _, cal = inputs.read_training(input_path, fold)
        old, new = io.load_problem(source), primary_problem(rows)
        for key in ("y", "weights", "N", "D", "E", "K", "physical_floors", "language_ids"):
            np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
        settings = source["fit"]
        if settings["delta"] != cal["delta"] or settings["start_mode"] != "grid" or settings["start_count"] != fit.start_count(name):
            raise ValueError("shared grid or Huber calibration changed")
        contract = dict(backend="primary_grid", fit=settings)
        if name == "M2":
            parent, = [p for p in result["parents"] if p["kind"] == "warm"]
            warm = read(parent["path"], parent["file_sha256"])
            membership(warm["membership"], fold)
            point = fit.nested_warm_start(name, fit.SharedParameters(**warm["selected"]["shared_parameters"]))
            contract["warm"] = point.values.tolist()
        if "numerical_refinement" in result:
            entry = result["numerical_refinement"]["plan"]
            refinement = read(entry["path"], entry["file_sha256"])
            if name != "J0" or fold != "largest_N_act" or len(refinement["starts"]) != 3:
                raise ValueError("unregistered primary numerical refinement")
            contract["conditioned_starts"] = refinement["starts"]
        contracts[f"{fold}--{name}"] = contract
    plan = read(ablation_path)
    io._verify_seal(plan, "sha256")
    if plan["protocol"] != ablation.PROTOCOL:
        raise ValueError("size-data numerical protocol changed")
    parent = read(plan["parent"]["path"], plan["parent"]["file_sha256"])
    for task in plan["tasks"]:
        if task["model"] != "M5-ND":
            continue
        fold = task["fold"]
        source = parent["folds"][fold]["manifest"]
        membership(source["membership"], fold)
        rows, _, _, cal = inputs.read_training(input_path, fold)
        old, new = io.load_problem(source), problem(rows, cal["regularization"])
        for key in ("y", "weights", "N", "D", "E", "K", "physical_floors", "language_ids", "D_family", "D_script", "D_other"):
            np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
        if source["fit"]["delta"] != cal["delta"] or source["fit"]["regularization"] != cal["regularization"] or len(task["starts"]) != 48:
            raise ValueError("size-data calibration or restart count changed")
        contracts[f"{fold}--M5-ND"] = dict(backend="primary_duration", starts=task["starts"], bounds=task["bounds"])
    expected = {f"{fold}--{name}" for fold in inputs.PRIMARY for name in ("J0", "M2", "M5-ND")}
    if set(contracts) != expected:
        raise ValueError("incomplete primary numerical contract panel")
    return contracts, sources


def predict_shared(name, selected, rows, regularization):
    p = problem(rows, regularization)
    model = (repetition.Model(name, p, rows.U_t, stored_parameters=rows.N_total)
             if name == "RN14-stored-reuse" else shared.Model(name, p, rows.U_t))
    prediction = shared.predict(model, selected)
    if name == "RN14-stored-reuse":
        independent = replay.predict(name, selected, rows)
    else:
        x = selected["coordinates"]
        long = name == "M5-ND"
        parameters = fit.shared_from_outer("M6" if long else name, x[:14] if long else x)
        independent = shared_equation(dict(shared_parameters=asdict(parameters), zeta=x[14] if long else 0.,
                                           language_coefficients=selected["language_coefficients"]), rows)
    np.testing.assert_allclose(prediction, independent, atol=1e-10, rtol=0)
    return prediction, float(np.max(abs(prediction-independent)))


def run_grid(task, rows, calibration, contract, sink=None):
    name = task["model"]
    attempts = []
    if name == "M5-ND":
        profiler = duration.Profile(problem(rows, calibration["regularization"]), calibration["delta"])
        bounds = tuple(ablation.AblationBound(**b) for b in contract["bounds"])
    else:
        p = primary_problem(rows)
        bounds = tuple(fit.OuterBound(**b) for b in contract["fit"]["bounds"])
    for index in task["start_indices"]:
        if name == "M5-ND":
            result = duration.run_attempt(profiler, contract["starts"][index], bounds)
            result["start_index"] = index
        else:
            settings = contract["fit"]
            fitted = fit.run_projected_start(name, fit.start_at(name, index), p, delta=calibration["delta"],
                bounds=bounds, ftol=settings["outer_ftol"], gtol=settings["outer_gtol"],
                maximum_iterations=settings["outer_max_iterations"])
            result = io._fit_record(fitted)
        attempts.append(result)
        if sink is not None:
            io.write_json_once(sink / f"start-{index:05d}.json", io._diagnostic(result))
    best = sorted((a for a in attempts if a["success"]), key=lambda a: (a["objective"], a["start_id"]))[:3]
    # Only three full coefficient records per shard are needed for the basin audits.
    return dict(complete=True, attempts=[io._diagnostic(a) for a in attempts], best=best)


def collect_grid(shards, count):
    attempts, best = [], []
    for task, result in shards:
        if not result.get("complete") or [a["start_index"] for a in result["attempts"]] != task["start_indices"]:
            raise ValueError("incomplete grid shard or changed start identities")
        for record in result["attempts"]:
            if record.get("objective") is not None and not np.isfinite(record["objective"]):
                raise ValueError("nonfinite training objective")
            if record.get("success") and record.get("objective") is None:
                raise ValueError("successful attempt has no objective")
        successful = sorted((a for a in result["attempts"] if a["success"]), key=lambda a: (a["objective"], a["start_id"]))[:3]
        if [io._diagnostic(a) for a in result["best"]] != successful:
            raise ValueError("grid shard did not retain its three best coefficient records")
        attempts.extend(result["attempts"])
        best.extend(result["best"])
    if [a["start_index"] for a in attempts] != list(range(count)):
        raise ValueError("grid requires exact, duplicate-free start coverage")
    return attempts, sorted(best, key=lambda a: (a["objective"], a["start_id"]))[:3]


def certify_grid(name, rows, cal, contract, shards):
    count = len(contract["starts"]) if name == "M5-ND" else contract["fit"]["start_count"]
    attempts, top = collect_grid(shards, count)
    if not top:
        return dict(certified=False, complete=True, error="no successful grid start", selected=None)
    if name == "M5-ND":
        profiler = duration.Profile(problem(rows, cal["regularization"]), cal["delta"])
        box = tuple(ablation.AblationBound(**b) for b in contract["bounds"])
        result = ablation.certify_attempts(profiler, box, name, top)
        return result | dict(certified=result["status"] == "healthy", complete=True, initial_starts=count)
    p, settings = primary_problem(rows), contract["fit"]
    box = tuple(fit.OuterBound(**b) for b in settings["bounds"])
    selected, audits = top[0], []

    def run(point):
        return fit.run_projected_start(name, point, p, delta=cal["delta"], bounds=box,
            ftol=settings["outer_ftol"], gtol=settings["outer_gtol"], maximum_iterations=settings["outer_max_iterations"])

    def agrees(record):
        if not record.get("success"):
            return False
        a, _ = predict_shared(name, selected, rows, cal["regularization"])
        b, _ = predict_shared(name, record, rows, cal["regularization"])
        return fit.equivalence_check(selected["objective"], record["objective"], a, b).equivalent

    if selected["boundary_contacts"]:
        return dict(certified=False, complete=True, selected=selected, error="bound-repeat required; no automatic bound changes")
    if name == "M2":
        warm = io._fit_record(run(fit.StartPoint("warm", None, np.array(contract["warm"]), "warm")))
        audits.append(dict(kind="warm", result=warm, resolved=agrees(warm)))
        if not audits[-1]["resolved"]:
            return dict(certified=False, complete=True, selected=selected, audits=audits, error="nested warm start disagrees")
    basins = [io._fit_record(run(fit.StartPoint(a["start_id"], None, np.array(a["coordinates"]), "basin"))) for a in top]
    basin_ok = len(basins) == 3 and all(agrees(a) for a in basins)
    audits.append(dict(kind="basin", results=basins, resolved=basin_ok))
    if not basin_ok:
        points = (fit.StartPoint("local:center", None, np.array(selected["coordinates"]), "local"),
                  *fit.deterministic_local_perturbations(selected["coordinates"], box))
        local = [io._fit_record(run(point)) for point in points]
        basin_ok = sum(agrees(a) for a in local) >= 3
        audits.append(dict(kind="local", results=local, resolved=basin_ok))
    if not basin_ok and "conditioned_starts" in contract:
        profiler = conditioned.J0Profile(p, cal["delta"])
        refined = [profiler.run(fit.StartPoint(a["start_id"], i, np.array(a["values"]), "basin"), box)
                   for i, a in enumerate(contract["conditioned_starts"])]
        chosen = min(refined, key=lambda a: (a[0].objective, a[0].start_id))
        prior = selected
        selected = io._fit_record(chosen[0])
        from scipy.optimize._numdiff import approx_derivative
        numerical = approx_derivative(lambda x: profiler.evaluate(x)[0].objective,
                                     selected["coordinates"], method="3-point", abs_step=1e-6)
        error = float(np.max(abs(numerical-chosen[1])))
        basin_ok = (all(agrees(io._fit_record(a)) and not a.boundary_contacts and np.max(abs(g)) <= 1e-8 for a, g in refined)
                    and selected["objective"] <= prior["objective"]+1e-15 and error <= 1e-8)
        audits.append(dict(kind="conditioned", results=[io._fit_record(a) for a, _ in refined],
                           gradient_check_error=error, resolved=bool(basin_ok)))
        direct_start = chosen[0]
    else:
        direct_start = run(fit.StartPoint("direct:best", None, np.array(selected["coordinates"]), "direct")) if basin_ok else None
    direct_ok = False
    if basin_ok and direct_start.success:
        direct = fit.direct_joint_check(direct_start, p, delta=cal["delta"], bounds=box)
        direct_ok = direct.resolved
        audits.append(dict(kind="direct", resolved=direct_ok, result=io._direct_fit_record(direct.direct, name)))
    return dict(certified=bool(basin_ok and direct_ok), complete=True, selected=selected, audits=audits,
                initial_starts=count, successful_starts=sum(a["success"] for a in attempts))


def execute(task, input_path, contract, deadline, sink, shards=()):
    rows, sources, packets, cal = inputs.read_training(input_path, task["fold"])
    name, stage = task["model"], task["stage"]
    if "arm" in task and name not in protocol.NATIVE:
        from . import numerics as controlled
        return controlled.execute(task, rows, sources, packets, cal, task["arm"], deadline, sink, shards)
    if name == "RN14-stored-reuse":
        cal = cal | dict(calibration_regularization=cal["regularization"], regularization=cal["regularization"] | {"lambda": 0.})
        model = repetition.Model(name, problem(rows, cal["regularization"]), rows.U_t, stored_parameters=rows.N_total)
        if stage == "fit":
            return repetition_numerics.start_shard(model, rows, cal, task["start_indices"], deadline, sink)
        attempts = repetition_numerics.collect_attempts(shards)
        return repetition_numerics.certify(model, rows, cal, attempts, deadline, sink)
    if name in (protocol.NATIVE if "arm" in task else inputs.NATIVE):
        if stage == "fit":
            item, = [p for p in packets if p["language"] == task["language"]]
            spec = native.Spec(**inputs.load(input_path)["native_specs"][name])
            result = numerics.fit_group(spec, item["train"], item["support"])
            if not result["certified"]:
                initial = result
                options = {"joint_method": "L-BFGS-B"} if "arm" in task else {}
                result = recovery_numerics.recover(spec, item["train"], item["support"], initial, **options) | {"initial_fit": initial}
            if result["certified"]:
                prediction = native.predict(spec, result["selected"]["coordinates"], item["train"])[0]
                np.testing.assert_allclose(prediction, native_replay(spec.payload(), result["selected"]["coordinates"], item["train"]), atol=1e-10, rtol=0)
            return result | dict(complete=True, language=item["language"], donors=item["donors"],
                                 unigram_nll=item["unigram_nll"], training_minimum=item["train"]["training_minimum"])
        groups = [r for _, r in shards]
        if sorted(g["language"] for g in groups) != sorted(p["language"] for p in packets):
            raise ValueError("incomplete or duplicated native language cohort")
        return dict(certified=all(g.get("certified") and g.get("complete") for g in groups),
                    complete=True, groups=groups, expected_languages=len(packets))
    if task["fold"] in inputs.PRIMARY:
        if stage == "fit":
            return run_grid(task, rows, cal, contract, sink)
        result = certify_grid(name, rows, cal, contract, shards)
    elif stage == "fit":
        model = data.make_model(name, rows, sources, packets, cal["regularization"], delta=cal["delta"])
        result = fit_cell(model, cal["delta"], deadline, anchor=cal["anchor"] if name == "M5-ND" else None, sink=sink)
        result["complete"] = not result["timed_out"]
    else:
        if len(shards) != 1:
            raise ValueError("expected exactly one adaptive all-grid fit")
        result = {k: v for k, v in shards[0][1].items()
                  if k not in ("sha256", "plan_sha256", "task", "job_id", "elapsed_seconds")}
    if result.get("certified"):
        _, gap = predict_shared(name, result["selected"], rows, cal["regularization"])
        result = result | {"independent_training_replay_max_gap": gap}
    return result | dict(model=name, delta=cal["delta"], regularization=cal["regularization"], heldout_scores_computed=False)
