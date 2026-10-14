"""Independent NumPy equation oracle; never imports the fitting model."""
import numpy as np

from ..law_search.replay import candidate
from ..law_search.replay import saturate


def components(name, record, rows):
    """Return local floor, capacity and data contributions for fixed coefficients."""
    x = np.asarray(record["coordinates"], float)
    if name.startswith("Shared-load"):
        n, e, k = rows.N_act.to_numpy()/1e8, rows.E.to_numpy(), rows.K.to_numpy()
        h = np.log(e)/np.log(32.)
        alpha, beta = x[0]*(1-h)+x[1]*h, x[2]*(1-h)+x[3]*h
        g = 1+x[17]*(k-1)/e**x[18]
        effective = saturate(rows.D_t, rows.U_t, x[15]*n**x[16])/1e9
        cap = e**x[4]*g**x[6]*n**(-alpha)
        term = e**x[5]*g**x[8]*n**x[14]*effective**(-beta)
    elif name.startswith("Cap-burden") or name in (
            "RN14-stored-data", "RN14-stored-reuse", "RN12-stored-common-K", "RN14-psi-tied-zetaE", "RN13-psi-tied",
            "RN13-psi-tied-stored", "RN15-psi-tied-ehat", "RN16-stored-reuse-ehat",
            "RN14-stored-data-exponent", "RN14-stored-shared-exponent", "RN15-stored-separate-exponents"):
        n, e, k = rows.N_act.to_numpy()/1e8, rows.E.to_numpy(), rows.K.to_numpy()
        h = np.log(e)/np.log(32.)
        if name.endswith("-ehat"):
            # Independent literal reciprocal equation, not the fitter's helper.
            start, ratio = np.exp(x[17]), x[18]
            offset, inverse_max = start/(1-ratio), ratio/start
            def transform(count):
                return 1./(1./(count-1.+offset)+inverse_max)
            e = transform(e)
            h = np.log(e/transform(1.))/np.log(transform(32.)/transform(1.))
        alpha, beta = x[0]*(1-h)+x[1]*h, x[2]*(1-h)+x[3]*h
        phi, psi = x[6]*(1-h)+x[7]*h, x[8]*(1-h)+x[9]*h
        hinge = np.maximum(np.log(k/12.), 0.)/np.log(200./12.)
        if name in ("RN14-stored-data-exponent", "RN14-stored-shared-exponent", "RN15-stored-separate-exponents"):
            beta = beta+x[17]*hinge
        cap = e**x[4]*k**phi*n**(-alpha-x[10]*hinge)
        if name.startswith("Cap-burden"):
            cap = e**x[4]*n**(-alpha)+x[17]*np.log(k)*n**(-x[0])*e**(-x[18])
            if name == "Cap-burden-stored-gamma14":
                cap *= n**(-x[10]*hinge)
        stored = name.startswith("RN14-stored") or name in (
            "RN13-psi-tied-stored", "RN12-stored-common-K", "Cap-burden-stored-13", "Cap-burden-stored-gamma14",
            "RN16-stored-reuse-ehat", "RN15-stored-separate-exponents")
        repeat_size = rows.N_total.to_numpy()/1e8 if stored else n
        effective = saturate(rows.D_t, rows.U_t, x[15]*repeat_size**x[16])/1e9
        multiplier = repeat_size**x[14] if name == "RN14-stored-data" else n**x[14]
        if name == "RN14-psi-tied-zetaE":
            multiplier = n**(x[14]*(1-h)+x[17]*h)
        term = e**x[5]*k**psi*multiplier*effective**(-beta)
    else:
        cap, term = candidate("RN-no-composition", x[:17], rows, None)
    coefficients = {c["language"]: c for c in record["language_coefficients"]}
    if len(coefficients) != len(record["language_coefficients"]):
        raise ValueError("duplicate language coefficients")
    c = np.array([[coefficients[l][key] for key in ("L_inf", "A", "B")] for l in rows.language])
    answer = c[:, 0], c[:, 1]*cap, c[:, 2]*term
    if not np.isfinite(answer).all():
        raise ValueError("nonfinite independent prediction")
    return answer


def predict(name, record, rows):
    floor, cap, term = components(name, record, rows)
    return floor+cap+term
