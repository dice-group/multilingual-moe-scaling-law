"""Population-matched fit scoring with independent prediction checks."""
from dataclasses import asdict
import numpy as np
from . import fit
from .law_search import data
from .law_search import model
from .law_search import replay
from .atlas_comparison_workflow import model as native
KEYS = ["trajectory_id", "readout_id", "language"]


def score(predictions, reference, *, endpoint=False):
    if predictions.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise ValueError("duplicate score identity")
    left = predictions.set_index(KEYS)
    right = reference.set_index(KEYS)
    complete = set(left.index) == set(right.index) and np.isfinite(left.prediction).all()
    result = {"complete": bool(complete), "rows": len(left), "expected_rows": len(right)}
    if not complete:
        return result
    left = left.loc[right.index]
    np.testing.assert_allclose(left.adjusted_loss, right.adjusted_loss, atol=1e-10, rtol=0)
    error = left.prediction.to_numpy()-right.adjusted_loss.to_numpy()
    weights = np.full(len(error), 1/len(error)) if endpoint else fit.trajectory_weights(reference.trajectory_id.tolist())
    return result | {"rmse": float(np.sqrt(weights @ error**2)), "mae": float(weights @ abs(error)),
                     "bias": float(weights @ error), "max_abs_error": float(max(abs(error)))}


def coefficients_predict(result,rows,sources,packets=()):
    fitted = data.make_model(result["model"],rows,sources,packets,result["regularization"],result["tau"],result["delta"])
    selected = result["selected"]
    predicted = model.predict(fitted,selected)
    coefficients = {r["language"]:r for r in selected["language_coefficients"]}
    c = np.array([[coefficients[l][k] for k in ("L_inf","A","B")] for l in rows.language])
    if result["model"] in model.CANDIDATES:
        cap,term = replay.candidate(result["model"],selected["coordinates"],rows,sources)
        independent = c[:,0]+c[:,1]*cap+c[:,2]*term
    elif fitted.spec:
        independent = np.full(len(rows),np.nan)
        for item in packets:
            idx=np.asarray(item["indices"],int)
            cap=replay.native(fitted.spec.payload(),np.r_[0.,1.,0.,selected["coordinates"]],item["train"])
            term=replay.native(fitted.spec.payload(),np.r_[0.,0.,1.,selected["coordinates"]],item["train"])
            independent[idx]=c[idx,0]+c[idx,1]*cap+c[idx,2]*term
    else:
        from .law_search.replay import shared_equation
        long=result["model"] in ("M6","M6-ND","M5-ND","J0-ND")
        x=selected["coordinates"]
        parameters=fit.shared_from_outer("M6" if long else result["model"],x[:14] if long else x)
        independent=shared_equation(dict(shared_parameters=asdict(parameters),
            zeta=float(x[14]) if long else 0.,language_coefficients=selected["language_coefficients"]),rows)
    np.testing.assert_allclose(predicted,independent,rtol=0,atol=1e-10)
    return predicted


def checked_score(predictions,reference,endpoint=False):
    value = score(predictions,reference,endpoint=endpoint)
    if value["complete"]:
        aligned = predictions.set_index(data.KEYS).loc[reference.set_index(data.KEYS).index]
        independent = replay.score(reference.adjusted_loss.to_numpy(),aligned.prediction.to_numpy(),
                                   reference.trajectory_id.to_numpy(),endpoint)
        for key in ("rmse","mae","bias"):
            if abs(value[key]-independent[key]) > 1e-10:
                raise ValueError("independent score replay disagrees")
    return value


def predict_result(result,rows,universe,inventories,metadata,*,selected=None):
    output = rows[data.KEYS+["adjusted_loss"]].copy()
    output["prediction"] = np.nan
    if selected is not None:
        result = result | {"selected":selected}
    if "groups" in result:
        for group in result["groups"]:
            subset = rows[rows.language.eq(group["language"])]
            if subset.empty or not group.get("selected"):
                continue
            packet = data.donor_covariates(subset,universe,inventories,metadata,group["language"],group["donors"])
            packet["training_minimum"] = group["training_minimum"]
            spec = native.Spec(**group["spec"])
            try:
                values = native.predict(spec,group["selected"]["coordinates"],packet)[0]
                independent = replay.native(group["spec"],group["selected"]["coordinates"],packet)
                np.testing.assert_allclose(values,independent,rtol=0,atol=1e-10)
                output.loc[subset.index,"prediction"] = values-group["unigram_nll"]
            except (ValueError,FloatingPointError):
                pass  # Domain-invalid rows remain present and NaN, never clipped.
        return output
    if not result.get("selected"):
        return output
    sources = data.source_arrays(rows,universe,inventories,metadata)
    packets = []
    if result["model"] in {s.name for s in model.POOLED}:
        for item in result["native_metadata"]:
            idx = np.flatnonzero(rows.language.eq(item["language"]))
            if not len(idx):
                continue
            packet = data.donor_covariates(rows.iloc[idx],universe,inventories,metadata,item["language"],item["donors"])
            packet["training_minimum"] = item["training_minimum"]
            packets.append(dict(indices=idx.tolist(),train=packet))
    output["prediction"] = coefficients_predict(result,rows,sources,packets)
    return output
