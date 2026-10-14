"""Restart shards and independent certification reuse the established fitter."""
from dataclasses import asdict
import time

import numpy as np

from .. import execution as io
from ..law_search import model as legacy
from ..shared_numerics import agreement, best_attempt
from ..nd_coupling.model import Reduction
from . import replay
from .contracts import ALL_FORMS, RULES
from .model import starts, validate


def start_shard(model, rows, calibration, indices, deadline, sink):
    seeds = starts(model, calibration["anchor"])
    profile = legacy.Profile(model, calibration["delta"], deadline)
    attempts = []
    for index in indices:
        began = time.monotonic()
        try:
            result = profile.run(seeds[index])
            validate(model, result)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            result = dict(success=False, error=f"{type(error).__name__}: {error}")
        result.update(index=index, initial=seeds[index].tolist(), elapsed_seconds=time.monotonic()-began)
        io.write_json_once(sink / f"start-{index:02d}.json", result)
        attempts.append(result)
        print(dict(start=index, success=result["success"], objective=result.get("objective")), flush=True)
    return dict(complete=True, attempts=attempts, heldout_scores_computed=False)


def collect_attempts(shards):
    attempts = []
    for task, receipt in shards:
        if not receipt.get("complete"):
            raise ValueError(f"incomplete start shard: {task['id']}")
        local = receipt["attempts"]
        if [a["index"] for a in local] != task["start_indices"]:
            raise ValueError("start coverage/order mismatch")
        for record in local:
            if record.get("objective") is not None and not np.isfinite(record["objective"]):
                raise ValueError("nonfinite saved training objective")
            if record.get("success") and (record.get("objective") is None or
                    record.get("kkt") is None or not 0 <= record["kkt"] <= RULES["gradient_tolerance"]):
                raise ValueError("malformed successful start")
        attempts.extend(local)
    if sorted(a["index"] for a in attempts) != list(range(RULES["starts"])):
        raise ValueError("exactly 48 distinct start indices are required")
    return sorted(attempts, key=lambda a: a["index"])


def geometry(profile, selected, *, include_support=False):
    """Local geometry in reduced coordinates, not the unreduced 19-slot space."""
    _, _, prediction, jac = profile.evaluate(selected["coordinates"])
    cap, term, _, _ = profile.model.terms(selected["coordinates"])
    robust = np.ones(len(prediction)) if profile.delta is None else (
        abs(prediction-profile.problem.y) < profile.delta).astype(float)
    weight = np.sqrt(profile.problem.weights*robust)
    reduction = Reduction(profile.model)
    residual = (jac@reduction.matrix)*weight[:, None]
    for _, idx in profile.groups:
        design = np.column_stack((np.ones(len(idx)), cap[idx], term[idx]))*weight[idx, None]
        residual[idx] -= design@np.linalg.lstsq(design, residual[idx], rcond=1e-12)[0]
    widths = np.array([high-low for low, high in reduction.bounds])
    singular = np.linalg.svd(residual*widths, compute_uv=False)
    contacts = [f"{b.name}:{side}" for value, b in zip(selected["coordinates"], profile.model.bounds)
                if b.upper != b.lower for side, limit in (("lower", b.lower), ("upper", b.upper))
                if abs(value-limit) <= 1e-6*max(1., abs(limit))]
    output = dict(free_shared=len(widths), nuisance_adjusted_rank=int(np.sum(
                  singular > (singular[0]*1e-7 if len(singular) else 0))),
                  singular_values=singular.tolist(), contacts=contacts,
                  warning="local rank is not global identifiability or seed uncertainty")
    if profile.model.data_coverage or profile.model.name == "RN14-stored-reuse":
        x = selected["coordinates"]
        gd = float(x[17]) if profile.model.data_coverage else 0.
        output["coverage_exponents"] = dict(gamma_n=float(x[10]), gamma_d=gd,
            difference=float(x[10]-gd),
            alpha_K12=[float(x[0]), float(x[1])], alpha_K200=[float(x[0]+x[10]), float(x[1]+x[10])],
            beta_K12=[float(x[2]), float(x[3])], beta_K200=[float(x[2]+gd), float(x[3]+gd)],
            interpretation="point coefficients, not relative importance or statistical equivalence")
    if profile.model.expert_transform:
        from .expert_transform import coordinates
        x = selected["coordinates"]
        e = np.array([1., 4., 16., 32., 64., 128.])
        loge, h, _, _ = coordinates(e, x[17], x[18])
        start = float(np.exp(x[17]))
        output["expert_transform"] = dict(E_start=start, ratio=float(x[18]),
            E_max=start/x[18] if x[18] > 0 else None,
            infinite_E_max=bool(x[18] == 0), E=e.tolist(), E_hat=np.exp(loge).tolist(),
            h=h.tolist(), beyond_E32="coordinate diagnostics only, no validation observations")
    if profile.model.load_form:
        x, p = selected["coordinates"], profile.problem
        load = x[17]*(p.K-1)/p.E**x[18]
        output["load_support"] = dict(min=float(load.min()), max=float(load.max()),
                                      fraction_between_point1_and10=float(np.mean((load >= .1)&(load <= 10.))))
    if include_support:
        x, model, p = selected["coordinates"], profile.model, profile.problem
        size = model.stored_parameters if model.stored_form else p.N
        rate = x[15]*(size/1e8)**x[16]
        output["repetition_support"] = dict(
            size_coordinate="N_total" if model.stored_form else "N_act",
            lambda_repetition=float(x[15]), rho=float(x[16]),
            rate_min=float(rate.min()), rate_max=float(rate.max()),
            fraction_D_above_U=float(np.mean(p.D > model.inventory)),
            weighting="unweighted training rows; support only, not independent observations")
        if model.burden_form:
            burden = x[17]*np.log(p.K)*np.exp(-x[0]*np.log(p.N/1e8)-x[18]*np.log(p.E))
            if model.name == "Cap-burden-stored-gamma14":
                hinge = np.maximum(np.log(p.K/12.), 0.)/np.log(200./12.)
                burden *= (p.N/1e8)**(-x[10]*hinge)
            fraction = burden/cap
            output["burden_support"] = dict(fraction_min=float(fraction.min()),
                                             fraction_max=float(fraction.max()),
                                             denominator="total capacity basis; same local A cancels")
        if model.name == "RN14-psi-tied-zetaE":
            output["zeta_endpoints"] = dict(E1=float(x[14]), E32=float(x[17]))
    return output


def certify(model, rows, calibration, attempts, deadline, sink):
    profile = legacy.Profile(model, calibration["delta"], deadline)
    best, audit, history = best_attempt(attempts), None, []
    for iteration in range(RULES["polish_attempts"]+1):
        if best is None:
            break  # never hide a lower-objective uncertified start
        audit = legacy.Direct(profile, best).run()
        validate(model, audit)
        history.append(audit)
        io.write_json_once(sink / f"audit-{iteration}.json", audit)
        if agreement(model, best, audit) or iteration == RULES["polish_attempts"]:
            break
        if not audit.get("success") or audit["objective"] >= best["objective"]:
            break
        refined = profile.run(audit["coordinates"])
        refined.update(index=RULES["starts"]+iteration, initial=audit["coordinates"])
        validate(model, refined)
        attempts.append(refined)
        io.write_json_once(sink / f"polish-{iteration}.json", refined)
        best = best_attempt(attempts)
    passed = bool(agreement(model, best, audit))
    gap, diagnostic = None, {}
    if best:
        validate(model, best)
        primary = legacy.predict(model, best)
        independent = replay.predict(model.name, best, rows)
        np.testing.assert_allclose(primary, independent, atol=RULES["replay_tolerance"], rtol=0)
        gap = float(np.max(abs(primary-independent)))
        diagnostic = geometry(profile, best, include_support="calibration_regularization" in calibration)
    return dict(certified=passed, complete=True, selected=best, audit=audit, audit_history=history,
                initial_starts=RULES["starts"], numerical_polishes=len(attempts)-RULES["starts"],
                bounds=[asdict(b) for b in model.bounds], equalities=list(model.equalities),
                nominal_shared=ALL_FORMS[model.name], local_coefficients=3*rows.language.nunique(),
                model=model.name, delta=calibration["delta"], regularization=calibration["regularization"],
                geometry=diagnostic, independent_training_replay_max_gap=gap,
                near_equivalent=[dict(index=a["index"], objective=a["objective"], coordinates=a["coordinates"])
                                 for a in attempts if best and a.get("success") and
                                 abs(a["objective"]-best["objective"]) <= 1e-5*max(abs(best["objective"]), 1e-15)],
                **({"calibration_regularization": calibration["calibration_regularization"]}
                   if "calibration_regularization" in calibration else {}),
                error=None if passed else "48-start and direct-audit certification did not pass",
                heldout_scores_computed=False)
