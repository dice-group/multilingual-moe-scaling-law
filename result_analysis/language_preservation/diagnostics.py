"""Per-language loss and target-exposure costs of expanding language coverage."""
import numpy as np
import pandas as pd


def preservation_diagnostics(frame, policies, metadata, penalty):
    from result_analysis.learning_curves import LearningCurve
    policies = policies.query("base_K == 50 and target_K == 200 and cap == 2200000000 and baseline_F == 2 and method == 'smooth' and tolerance == 0")
    meta = metadata.loc[metadata.tier.eq("K50")].set_index("language")
    records, input_rows = [], []
    for row in policies.itertuples():
        old_ladder, old_e = row.baseline_config.split("/E")
        new_ladder, new_e = row.config.split("/E")
        old_tokens = row.baseline_F*1e18/(6*row.baseline_N_act)
        for language in meta.index:
            values = []
            for role, ladder, experts, tier, tokens in (("reference", old_ladder, int(old_e), 50, old_tokens),
                    ("expanded", new_ladder, int(new_e), 200, row.D_total)):
                group = frame.loc[frame.ladder.eq(ladder) & frame.E.eq(experts) & frame.K.eq(tier) & frame.language.eq(language)].sort_values("D_total")
                curve = LearningCurve.fit(group.D_total.to_numpy(), group.adjusted_loss.to_numpy(), penalty)
                if tokens < group.D_total.min() or tokens > group.D_total.max():
                    raise ValueError("preservation interpolation outside support")
                values.append(float(curve(np.array([tokens]))[0]))
                for point in group.itertuples():
                    input_rows.append(dict(objective=row.objective, role=role, language=language,
                        D_total=point.D_total, adjusted_loss=point.adjusted_loss, query_tokens=tokens))
            records.append(dict(objective=row.objective, language=language, name=meta.loc[language, "name"],
                resource=meta.loc[language, "resource_availability"], family=meta.loc[language, "family"],
                script=meta.loc[language, "production_script"], reference_loss=values[0], expanded_loss=values[1],
                loss_delta=values[1]-values[0], compute_multiplier=row.compute_multiplier,
                config=row.config, unknown_continuations=row.unknown_continuations))
    result = pd.DataFrame(records)
    for row in policies.itertuples():
        delta = result.loc[result.objective.eq(row.objective), "loss_delta"]
        np.testing.assert_allclose([delta.mean(), delta.max()], [row.mean_delta, row.worst_delta], atol=1e-10, rtol=0)
    return result, pd.DataFrame(input_rows)


def dilution_reference(frame, policies):
    """Loss-blind same-recipe cost of restoring each incumbent's exposure."""
    records = []
    base_rows = policies.query("method == 'smooth' and objective == 'mean' and tolerance == 0")
    for row in base_rows.itertuples():
        ladder, expert = row.baseline_config.split("/E")
        recipe = frame.loc[frame.ladder.eq(ladder) & frame.E.eq(int(expert))]
        old, new = recipe.loc[recipe.K.eq(row.base_K)], recipe.loc[recipe.K.eq(row.target_K)]
        tokens = row.baseline_F*1e18/(6*row.baseline_N_act)
        ratios = []
        for language, group in old.groupby("language"):
            group = group.sort_values("D_total")
            target = new.loc[new.language.eq(language)].sort_values("D_total")
            exposure = np.interp(tokens, group.D_total, group.D_t)
            if exposure < target.D_t.min() or exposure > target.D_t.max():
                ratios.append(np.nan)
            else:
                ratios.append(float(np.interp(exposure, target.D_t, target.D_total)/tokens))
        complete = bool(np.isfinite(ratios).all())
        records.append(dict(base_K=row.base_K, target_K=row.target_K, cap=row.cap,
            baseline_F=row.baseline_F, baseline_config=row.baseline_config,
            exposure_multiplier_median=float(np.median(ratios)) if complete else np.nan,
            exposure_multiplier_min=float(np.min(ratios)) if complete else np.nan,
            exposure_multiplier_max=float(np.max(ratios)) if complete else np.nan,
            complete=complete, languages=len(ratios),
            interpretation="same recipe; target exposure preservation, not loss preservation"))
    return pd.DataFrame(records)
