"""Supporting populations kept distinct from the primary matched-exposure claim."""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from result_analysis.capacity_response import tables as matched_exposure


def resource_gains(rows, metadata):
    """Assign resource tiers without using measured model losses.

    Missing labels use the nearest standardized log corpus-profile centroid;
    missing profiles use the middle tier. This is distinct from the explicit
    UNKNOWN group in the separate common50 expansion analysis below.
    """
    features = ["words", "documents", "utf8_bytes"]
    source = metadata[metadata.split.eq("train")].copy()
    source[features] = source[features].apply(pd.to_numeric, errors="coerce")
    source = source[["subset", "resource_availability", *features]].drop_duplicates("subset")
    k200 = rows[rows.K.eq(200)]
    mapping = pd.DataFrame({"language": sorted(k200.language.unique())}).merge(
        source, left_on="language", right_on="subset", how="left", validate="one_to_one")
    mapping["resource_availability"] = mapping.resource_availability.str.upper()
    labelled = mapping.dropna(subset=["resource_availability"])
    profiles = np.log10(labelled[features])
    centroids = profiles.groupby(labelled.resource_availability).mean()
    scales = profiles.std(ddof=0)
    assignments = []
    for row in mapping.itertuples():
        tier = row.resource_availability
        method = "source label"
        if pd.isna(tier):
            profile = pd.Series({f: getattr(row, f) for f in features}, dtype=float)
            usable = profile.notna() & profile.gt(0)
            if usable.any():
                distance = ((centroids.loc[:, usable]-np.log10(profile[usable]))/scales[usable]).pow(2)
                tier, method = str(distance.mean(axis=1).idxmin()), "nearest mean log corpus profile"
            else:
                tier, method = "MEDIUM", "no corpus profile: neutral middle tier"
        assignments.append((tier, method))
    mapping[["tier", "assignment"]] = assignments
    keys = ["N_act", "D_total", "D_t", "language"]
    dense = k200[k200.E.eq(1)][keys+["adjusted_loss"]].rename(columns={"adjusted_loss": "dense_loss"})
    pairs = k200[k200.E.ne(1)].merge(dense, on=keys, validate="many_to_one").merge(
        mapping[["language", "tier"]], on="language", validate="many_to_one")
    pairs["gain"] = pairs.dense_loss-pairs.adjusted_loss
    per_language = pairs.groupby(["tier", "N_act", "E", "language"]).gain.mean()
    gains = per_language.groupby(["tier", "N_act", "E"]).mean().groupby(["tier", "E"]).mean()
    return mapping, gains


def composition_points(observations, queries):
    """Compare interpolation choices at identical query points."""
    curves = {key: g.sort_values("D_t") for key, g in observations.groupby(["arm", "seed", "E", "language"])}
    records = []
    for row in queries.itertuples():
        reference = curves[row.arm, row.seed, row.E, row.language]
        target = curves["K200", row.seed, row.E, row.language]
        target = target[target.horizon.eq(row.horizon)]
        if len(target) != 1:
            raise ValueError("ambiguous composition target horizon")
        t = target.iloc[0]
        if not reference.D_t.min() <= t.D_t <= reference.D_t.max():
            raise ValueError("composition query outside observed support")
        x = reference.D_t.to_numpy(float)
        y = reference.raw_nll.to_numpy(float)
        if row.method == "log_linear_D_t":
            estimate = np.interp(np.log(t.D_t), np.log(x), y)
        elif row.method == "linear_D_t":
            estimate = np.interp(t.D_t, x, y)
        elif row.method == "pchip_log_D_t":
            estimate = PchipInterpolator(np.log(x), y, extrapolate=False)(np.log(t.D_t))
        else:
            raise ValueError(f"unknown interpolation: {row.method}")
        records.append(dict(arm=row.arm, seed=row.seed, E=row.E, language=row.language,
                            horizon=row.horizon, method=row.method, gap=float(t.raw_nll-estimate)))
    return pd.DataFrame(records)


def run(evidence, output, checks, rows):
    # At approximately 8B tokens, every architecture has the same 200 targets.
    native = rows[rows.K.eq(200) & np.isclose(rows.D_total, 8e9, rtol=5e-4)]
    native = native.groupby(["ladder", "E"]).raw_nll.mean().unstack("E")
    checks.require("full 8B native-language grid", native.shape == (4, 4) and native.notna().all().all())
    expert, size = float((native[1]-native[32]).mean()), float((native.loc["A"]-native.loc["D"]).mean())
    checks.close("paper native K200 capacity gains", [expert, size], [.2637, .4911], atol=5.001e-5, source="paper/my_content.tex")
    native.to_csv(output / "native_8B_means.csv")
    tiers, absolute_gains = resource_gains(rows, evidence.table("metadata/fineweb2-language-distribution.csv"))
    checks.require("historical resource-tier membership", tiers.tier.value_counts().to_dict() == {"HIGH": 28, "MEDIUM": 20, "LOW": 152})
    checks.require("loss-blind resource fallback inventory", tiers.assignment.ne("source label").sum() == 19)
    checks.close("paper absolute resource gains", [absolute_gains.loc[t, 32] for t in ("HIGH", "MEDIUM", "LOW")],
                 [.1960, .2010, .2181], atol=5.001e-5, source="paper/my_appendix.tex")
    tiers.to_csv(output / "absolute_resource_tier_assignments.csv", index=False)
    absolute_gains.to_csv(output / "absolute_resource_gains.csv")
    seeds = evidence.table("study/paper_exports/seed_pressure_points.csv")
    summary = seeds.groupby(["seed", "E"]).adjusted_delta.mean().rename("pressure")
    reference = evidence.table("study/paper_exports/seed_pressure_summary.csv").set_index(["seed", "E"]).pressure
    checks.close("three-seed pressure aggregation", summary.sort_index(), reference.sort_index())
    summary.to_csv(output / "seed_pressure_reaggregated.csv")
    wide = summary.unstack("E")
    attenuation = (1-wide[32]/wide[1]).rename("attenuation")
    attenuation.to_csv(output / "seed_attenuation_reaggregated.csv")
    query = evidence.table("study/composition_coverage/matched_points.csv")
    comp = composition_points(evidence.table("study/composition_coverage/observations.csv"), query)
    checks.close("composition interpolation replay", comp.gap, query.gap, atol=2e-11)
    comp.to_csv(output / "composition_matched_points.csv", index=False)
    summary = comp.groupby(["method", "arm", "seed", "E", "horizon"]).gap.mean().groupby(["method", "arm", "E"]).mean().unstack("E")
    summary = summary.rename(columns={1: "E1_gap", 16: "E16_gap", 32: "E32_gap"})
    summary["absolute_reduction"] = summary.E1_gap-summary.E32_gap
    summary["reduction_percent"] = 100*summary.absolute_reduction/summary.E1_gap
    reference = evidence.table("study/composition_coverage/summary.csv").set_index(["method", "arm"])
    checks.close("composition summary", summary.sort_index(), reference[summary.columns].sort_index())
    summary.to_csv(output / "composition_summary.csv")
    points = matched_exposure.matched_pressure(rows, 50, 200)
    reference = evidence.table("study/resource_expansion/matched_gap_points.csv")
    keys = ["ladder", "E", "language", "target_readout_id"]
    paired = points.merge(reference, on=keys, how="outer", validate="one_to_one", indicator=True)
    checks.require("resource expansion membership", paired._merge.eq("both").all())
    checks.close("resource expansion contrasts", paired.pressure, paired.gap, atol=2e-11)
    metadata = evidence.table("study/resource_expansion/cohort_metadata.csv")[["language", "resource_group"]]
    points = points.merge(metadata, on="language", validate="many_to_one")
    points.to_csv(output / "resource_expansion_points.csv", index=False)
    all50 = points.assign(resource_group="ALL50")
    pooled = pd.concat([points, all50]).groupby(["resource_group", "ladder", "E"]).pressure.mean().groupby(["resource_group", "E"]).mean()
    reference = evidence.table("study/resource_expansion/pooled_gaps.csv").set_index(["resource_group", "E"]).gap
    checks.close("equal-size resource summaries", pooled.sort_index(), reference.sort_index())
    pooled.to_csv(output / "resource_expansion_summary.csv")
    # Router MI requires underlying router counts; do not reconstruct it from
    # language means, so these tables are copied rather than recalculated.
    for name in ("language_metrics", "checkpoint_summary", "coverage_contrasts", "loss_associations"):
        evidence.table(f"routing/tables/{name}.csv").to_csv(output / f"retained_routing_{name}.csv", index=False)
    for name in ("release_manifest.json", "analysis_spec.json", "verification/receipt.json"):
        evidence.json(f"routing/{name}")
    return {"native_K200_8B_expert_gain": expert, "native_K200_8B_active_size_gain": size,
            "composition_points_recomputed": len(comp), "resource_points_recomputed": len(points),
            "seed_source": "reaggregate saved matched points; supplementary training evaluation not rerun",
            "routing_source": "sealed diagnostic tables retained, no router-count replay"}
