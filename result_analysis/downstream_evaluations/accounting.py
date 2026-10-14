"""Exact answer-transition accounting, independent of a causal interpretation."""
import pandas as pd


def aggregate(languages, flips, checks):
    mapping = languages[["benchmark", "language", "corpus_language"]].drop_duplicates()
    checks.require("one benchmark-variety mapping", not mapping.duplicated(["benchmark", "language"]).any())
    flips = flips.merge(mapping, on=["benchmark", "language"], how="left", validate="many_to_one")
    checks.require("all decision languages mapped", flips.corpus_language.notna().all())
    for name in ("corrected", "broken"):
        flips[name+"_pp"] = 100*flips[name]/flips.samples
    checks.close("item-level answer-change identity", flips.corrected_pp-flips.broken_pp, flips.net_gain_pp)
    keys = ["benchmark", "ladder", "seed", "K"]
    rates = ["corrected_pp", "broken_pp", "net_gain_pp"]
    macro = flips.groupby(keys+["corpus_language"])[rates].mean().groupby(keys).mean().reset_index()
    metrics = ["primary", "accuracy_raw", "accuracy_token", "accuracy_char", "gold_nll", "wrong_nll", "margin", "margin_token", "macro_f1"]
    keys = ["benchmark", "ladder", "seed", "E", "K"]
    means = languages.groupby(keys+["corpus_language"])[metrics].mean().groupby(keys).mean().reset_index()
    outputs = []
    for (task, ladder, seed), group in means.groupby(["benchmark", "ladder", "seed"]):
        if ladder not in ("B", "D") or task not in set(flips.benchmark):
            continue
        cells = group.set_index(["E", "K"])
        if not all(pair in cells.index for pair in ((1, 12), (32, 12), (1, 200), (32, 200))):
            continue
        row = dict(benchmark=task, ladder=ladder, seed=int(seed))
        for k in (12, 200):
            for metric in metrics:
                a, b = cells.loc[(1, k), metric], cells.loc[(32, k), metric]
                if pd.notna(a) and pd.notna(b):
                    sign = -1 if metric.endswith("nll") else 1
                    factor = 100 if metric in ("primary", "macro_f1") or metric.startswith("accuracy") else 1
                    row[f"{metric}_gain_K{k}"] = sign*factor*(b-a)
            part = macro[(macro.benchmark == task) & (macro.ladder == ladder) & (macro.seed == seed) & (macro.K == k)]
            checks.require(f"paired answer cells {task}/{ladder}/{seed}/{k}", len(part) == 1)
            for rate in rates:
                row[f"{rate}_K{k}"] = float(part.iloc[0][rate])
            checks.close(f"macro gain identity {task}/{ladder}/{seed}/{k}", row[f"primary_gain_K{k}"], row[f"net_gain_pp_K{k}"])
        for metric in metrics:
            if f"{metric}_gain_K12" in row:
                row[metric+"_interaction"] = row[f"{metric}_gain_K200"]-row[f"{metric}_gain_K12"]
        for rate in ("corrected_pp", "broken_pp"):
            row[rate+"_change"] = row[rate+"_K200"]-row[rate+"_K12"]
        checks.close(f"interaction accounting {task}/{ladder}/{seed}", row["primary_interaction"],
                     row["corrected_pp_change"]-row["broken_pp_change"])
        outputs.append(row)
    return pd.DataFrame(outputs), macro, means
