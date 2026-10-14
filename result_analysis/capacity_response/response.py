"""Coverage gaps evaluated at specified points within observed target exposure."""
import numpy as np
import pandas as pd


def interpolate(x, xp, values):
    x, xp = np.asarray(x, float), np.asarray(xp, float)
    if (not np.isfinite(x).all() or not np.isfinite(xp).all() or np.any(x <= 0)
            or np.any(xp <= 0) or np.any(np.diff(xp) <= 0)
            or x.min() < xp.min()*(1-1e-10) or x.max() > xp.max()*(1+1e-10)):
        raise ValueError("interpolation outside ordered observed exposure support")
    return np.interp(np.log(x), np.log(xp), values)


def matched_response(rows, queries):
    """Use query positions, not saved losses, to compute coverage gaps."""
    languages = sorted(rows[rows.K.eq(12)].language.unique())
    groups = {key: group.sort_values("D_t") for key, group in rows.groupby(["ladder", "E", "K", "language"])}
    result = []
    for query in queries.itertuples():
        differences, ratios = [], []
        for language in languages:
            a, b = groups[query.ladder, query.E, 12, language], groups[query.ladder, query.E, 200, language]
            exposure = float(np.exp(interpolate(query.D200, b.D_total, np.log(b.D_t))))
            raw = float(interpolate(exposure, b.D_t, b.raw_nll)-interpolate(exposure, a.D_t, a.raw_nll))
            adjusted = float(interpolate(exposure, b.D_t, b.adjusted_loss)-interpolate(exposure, a.D_t, a.adjusted_loss))
            np.testing.assert_allclose(raw, adjusted, atol=1e-11, rtol=0)
            matched_tokens = float(np.exp(interpolate(exposure, a.D_t, np.log(a.D_total))))
            differences.append(raw)
            ratios.append(query.D200/matched_tokens)
        n = float(b.N_act.iloc[0])
        result.append(dict(ladder=query.ladder, E=query.E, N_act=n, D200=query.D200,
            FLOPs=6*n*query.D200, response=float(np.mean(differences)), language_count=len(languages),
            compute_ratio_min=min(ratios), compute_ratio_max=max(ratios)))
    return pd.DataFrame(result)


def crossovers(response):
    """Locate the coverage-gap sign change by interpolating in log FLOPs."""
    result = {}
    for expert in (1, 32):
        group = response[response.ladder.eq("D") & response.E.eq(expert)].sort_values("FLOPs")
        values = group.response.to_numpy()
        crossing = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0))
        if len(crossing) != 1:
            raise ValueError("expected one supported size-D benefit-to-penalty crossing")
        pair = group.iloc[crossing[0]:crossing[0]+2]
        result[str(expert)] = float(np.exp(np.interp(0, pair.response, np.log(pair.FLOPs))))
    return result
