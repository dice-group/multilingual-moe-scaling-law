"""Search preservation policies and evaluate losses within observed support."""
import numpy as np
import pandas as pd

from result_analysis import learning_curves as curves


def search_policies(rows, geometry, languages, penalty, cap=2.2e9, baseline_f=2., *,
                    base_k=50, target_k=200, objectives=("mean", "all"), require_reached=True):
    """Least-compute feasible restoration across measured recipes and support.

    This searches observed recipe curves, not hypothetical larger architectures.
    Censored/unmeasured continuations prevent a global-minimum claim.
    """
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("unique nonempty language cohort required")
    if not 0 < base_k < target_k or len(languages) > base_k:
        raise ValueError("invalid expansion or incumbent language cohort")
    if not np.isfinite([cap, baseline_f]).all() or min(cap, baseline_f) <= 0:
        raise ValueError("positive finite storage cap and baseline compute required")
    if not objectives or not set(objectives) <= {"mean", "all"} or len(set(objectives)) != len(objectives):
        raise ValueError("unique mean/all objectives required")
    if geometry.duplicated(["ladder", "E"]).any() or not np.isfinite(geometry[["N_act", "N_total"]]).all().all():
        raise ValueError("invalid or duplicate architecture geometry")
    if not geometry[["N_act", "N_total"]].gt(0).all().all():
        raise ValueError("positive architecture sizes required")
    bank = {base_k: [], target_k: []}
    for k in bank:
        for recipe in geometry[geometry.N_total.le(cap)].sort_values(["N_total", "ladder", "E"]).itertuples():
            part = rows[rows.K.eq(k) & rows.ladder.eq(recipe.ladder) & rows.E.eq(recipe.E) & rows.language.isin(languages)]
            pivot = part.pivot(index="D_total", columns="language", values="adjusted_loss").reindex(columns=languages)
            if pivot.empty or pivot.isna().any().any():
                raise ValueError("incomplete preservation recipe")
            tokens = pivot.index.to_numpy(float)
            coefficients = np.stack([curves.LearningCurve.fit(tokens, y, penalty).coefficients for y in pivot.to_numpy().T], axis=1)
            fit = curves.LearningCurve(tokens, pivot.mean(axis=1).to_numpy(), penalty, coefficients)
            bank[k].append(dict(name=f"{recipe.ladder}/E{recipe.E}", n=recipe.N_act, stored=recipe.N_total,
                                fit=fit, fmin=6*recipe.N_act*tokens[0]/1e18, fmax=6*recipe.N_act*tokens[-1]/1e18))
    def values(recipe, f):
        return recipe["fit"](f*1e18/(6*recipe["n"]))
    eligible = [r for r in bank[base_k] if r["fmin"] <= baseline_f <= r["fmax"]]
    if not eligible:
        raise ValueError("no supported baseline at this compute")
    base = min(eligible, key=lambda r: (float(values(r, baseline_f).mean()), r["stored"]))
    reference = values(base, baseline_f)
    selected, candidates = [], []
    for objective in objectives:
        options = []
        for recipe in bank[target_k]:
            def change(f):
                delta = values(recipe, f)-reference
                return float(delta.mean() if objective == "mean" else delta.max())
            compute, status = curves.invert_monotone(change, 0., recipe["fmin"], recipe["fmax"])
            candidates.append(dict(objective=objective, config=recipe["name"], status=status,
                                   feasible_compute_EFLOPs=compute, support_max_EFLOPs=recipe["fmax"]))
            if compute is not None:
                options.append((compute, recipe["stored"], recipe["name"], recipe))
        if not options:
            if require_reached:
                raise ValueError("preservation objective not reached within observed support")
            continue
        compute, _, _, recipe = min(options, key=lambda row: row[:3])
        selected.append(dict(base_K=base_k, target_K=target_k, cap=cap, baseline_F=baseline_f,
                             baseline_config=base["name"], baseline_N_act=base["n"], objective=objective,
                             method="smooth", tolerance=0., config=recipe["name"], N_act=recipe["n"],
                             D_total=compute*1e18/(6*recipe["n"]), compute_multiplier=compute/baseline_f))
    return pd.DataFrame(selected), pd.DataFrame(candidates)


def preservation_at_saved_policies(rows, policies, languages, penalty):
    """Forward-check the frozen feasible policies; do not claim a global optimum."""
    results = []
    for policy in policies.itertuples():
        reference_size, reference_e = policy.baseline_config.split("/E")
        target_size, target_e = policy.config.split("/E")
        for language in languages:
            losses = []
            for size, experts, coverage, tokens in (
                (reference_size, int(reference_e), policy.base_K, policy.baseline_F*1e18/(6*policy.baseline_N_act)),
                (target_size, int(target_e), policy.target_K, policy.D_total),
            ):
                group = rows[rows.ladder.eq(size) & rows.E.eq(experts) & rows.K.eq(coverage)
                             & rows.language.eq(language)].sort_values("D_total")
                fit = curves.LearningCurve.fit(group.D_total.to_numpy(), group.adjusted_loss.to_numpy(), penalty)
                losses.append(float(fit(tokens)))
            multiplier = 6*policy.N_act*policy.D_total/(policy.baseline_F*1e18)
            np.testing.assert_allclose(multiplier, policy.compute_multiplier, atol=1e-12, rtol=0)
            results.append(dict(objective=policy.objective, language=language, reference_loss=losses[0],
                expanded_loss=losses[1], loss_delta=losses[1]-losses[0], compute_multiplier=float(multiplier)))
    return pd.DataFrame(results)
