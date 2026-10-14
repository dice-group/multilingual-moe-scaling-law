"""CPU-only answer-loss, margin and token-normalization diagnostics.

Exploratory D/E={1,32}/K={12,200} contrasts on the fixed benchmark-supported
K12 languages. Keep original raw/character accuracy; token accuracy is an
additional sensitivity, motivated by FineWeb2's independently published setup.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from result_analysis.downstream_evaluations.endpoints.analyze import (
    BENCHMARKS, referenced, sealed,
)
from result_analysis.capacity_response.tables import sha256_file


def question_metrics(values, tokens, lengths, gold):
    q = np.arange(len(gold))
    valid = np.isfinite(values)
    result = {"gold_nll_per_token": -values[q, gold] / tokens[q, gold]}
    wrong_nll = np.where(valid, -values / tokens, np.nan)
    wrong_nll[q, gold] = np.nan
    result["wrong_nll_per_token"] = np.nanmean(wrong_nll, axis=1)
    predictions = {}
    for suffix, score in [("raw", values), ("char", values / lengths), ("token", values / tokens)]:
        pred = np.argmax(score, axis=1)
        predictions[suffix] = pred
        result[f"accuracy_{suffix}"] = (pred == gold).astype(float)
        ce = logsumexp(score, axis=1) - score[q, gold]
        result[f"choice_ce_{suffix}"] = ce
        result[f"choice_probability_{suffix}"] = np.exp(-ce)
        wrong = score.copy()
        wrong[q, gold] = -np.inf
        result[f"margin_{suffix}"] = score[q, gold] - wrong.max(axis=1)
    result["raw_char_disagreement"] = (predictions["raw"] != predictions["char"]).astype(float)
    result["char_token_disagreement"] = (predictions["char"] != predictions["token"]).astype(float)
    assert all(np.isfinite(x).all() for x in result.values())
    return result


def main(campaign_root, analysis, output):
    campaign = sealed(campaign_root / "campaign.json")
    verification = json.loads((analysis / "verification.json").read_text())
    membership_path = analysis / "language_membership.csv"
    assert sha256_file(membership_path) == verification["outputs"]["language_membership"]["sha256"]
    membership = pd.read_csv(membership_path).query("in_common12")
    expected_path = analysis / "cohort_metrics.csv"
    assert sha256_file(expected_path) == verification["outputs"]["cohort_metrics"]["sha256"]
    expected = pd.read_csv(expected_path).query('ladder == "D" and seed == 1234 and scope == "common12"')
    expected = expected.set_index(["benchmark", "E", "K"])
    jobs = [j for j in campaign["jobs"] if j["ladder"] == "D" and j["E"] in [1, 32] and j["K"] in [12, 200] and j["seed"] == 1234]
    assert len(jobs) == 24
    first_spec = sealed(referenced(jobs[0]["spec"]), "spec_sha256")
    data_path = referenced(first_spec["full_suite_provenance"]["data_manifest"])
    data = sealed(data_path)
    sources = [{"path": str(p), "sha256": sha256_file(p)} for p in [
        Path(__file__), campaign_root / "campaign.json", data_path, membership_path, expected_path]]
    output.mkdir(parents=True, exist_ok=False)
    language_rows, paired_rows, score_receipts = [], [], []
    for benchmark in BENCHMARKS:
        languages = sorted(membership.loc[membership.benchmark.eq(benchmark), "language"])
        docs, request_index = [], {}
        for language in languages:
            ref = next(r for r in data["languages"] if r["benchmark"] == benchmark and r["language"] == language)
            path = referenced(ref)
            frozen = sealed(path)
            assert len(frozen["tasks"]) == 1
            task = frozen["tasks"][0]
            local_index = {}
            for d in task["documents"]:
                local_index[d["doc_id"]] = len(docs)
                docs.append({"language": language, **d})
            for request in frozen["requests"]:
                rid = request["request_id"]
                assert rid not in request_index
                request_index[rid] = (local_index[request["doc_id"]], request["choice_index"])
            sources.append({"path": str(path), "sha256": ref["sha256"]})
            del frozen
        n, width = len(docs), max(len(d["choice_lengths"]) for d in docs)
        lengths = np.ones((n, width))
        gold = np.array([d["target"] for d in docs])
        lang = np.array([d["language"] for d in docs])
        for i, d in enumerate(docs):
            lengths[i, :len(d["choice_lengths"])] = d["choice_lengths"]
        cells, reference_tokens = {}, None
        for job in sorted((j for j in jobs if j["benchmark"] == benchmark), key=lambda j: (j["K"], j["E"])):
            spec_path = referenced(job["spec"])
            spec = sealed(spec_path, "spec_sha256")
            path = Path(spec["output_dir"]) / "scores.json"
            scores = sealed(path, "result_sha256")
            assert scores["status"] == "complete" and scores["parity_gate"]["passed"] and scores["replica_gate"]["passed"]
            assert scores["score_spec"]["sha256"] == job["spec"]["sha256"]
            assert scores["request_manifest"]["sha256"] == campaign["inputs"][benchmark]["request_manifest"]["sha256"]
            if job["E"] > 1:
                assert scores["moe_routing"]["totals"]["dropped_assignments"] == 0
            values = np.full((n, width), -np.inf)
            tokens = np.ones((n, width), dtype=np.int32)
            truncated = np.zeros(n, dtype=bool)
            count = 0
            for row in scores["scores"]:
                index = request_index.get(row["request_id"])
                if index is None:
                    continue
                assert not np.isfinite(values[index])
                assert row["scored_tokens"] > 0 and np.isfinite(row["loglikelihood"])
                values[index] = row["loglikelihood"]
                tokens[index] = row["scored_tokens"]
                truncated[index[0]] |= row["truncated_tokens"] > 0
                count += 1
            assert count == len(request_index)
            if reference_tokens is None:
                reference_tokens = tokens.copy()
            else:
                np.testing.assert_array_equal(tokens, reference_tokens)
            score_receipts.append({"benchmark": benchmark, "E": job["E"], "K": job["K"],
                "path": str(path), "sha256": sha256_file(path), "spec_sha256": job["spec"]["sha256"],
                "selected_documents": n, "selected_requests": count})
            del scores
            gc.collect()
            metrics = question_metrics(values, tokens, lengths, gold)
            metrics["context_truncated"] = truncated.astype(float)
            cells[(job["E"], job["K"])] = metrics
            for language in languages:
                selected = lang == language
                language_rows.append({"benchmark": benchmark, "E": job["E"], "K": job["K"],
                    "language": language, "samples": int(selected.sum()),
                    **{m: float(v[selected].mean()) for m, v in metrics.items()}})
            print(f"{benchmark} E{job['E']} K{job['K']}: {n:,} questions, scores and tokens verified", flush=True)
        # Pair the exact same questions when counting corrected and broken answers.
        for k in [12, 200]:
            for language in languages:
                selected = lang == language
                for suffix in ["raw", "char", "token"]:
                    a = cells[(1, k)][f"accuracy_{suffix}"][selected].astype(bool)
                    b = cells[(32, k)][f"accuracy_{suffix}"][selected].astype(bool)
                    paired_rows.append({"benchmark": benchmark, "K": k, "language": language, "metric": suffix,
                        "samples": len(a), "corrected": int((~a & b).sum()), "broken": int((a & ~b).sum()),
                        "net_gain_pp": float(100 * (b.mean() - a.mean()))})
        del cells
    language = pd.DataFrame(language_rows)
    metric_names = [c for c in language.columns if c not in ["benchmark", "E", "K", "language", "samples"]]
    macro = language.groupby(["benchmark", "E", "K"])[metric_names].mean()
    for suffix, old in [("raw", "acc"), ("char", "acc_norm")]:
        np.testing.assert_allclose(macro[f"accuracy_{suffix}"], expected.loc[macro.index, old], rtol=0, atol=1e-14)
    gains = macro.xs(32, level="E") - macro.xs(1, level="E")
    # Improvement-positive for losses, margins, probabilities and accuracy.
    lower = [m for m in metric_names if "nll" in m or "choice_ce" in m]
    gains[lower] *= -1
    interaction = gains.xs(200, level="K") - gains.xs(12, level="K")
    tables = {"language_diagnostics": language, "benchmark_diagnostics": macro.reset_index(),
              "expert_gains": gains.reset_index(), "coverage_interactions": interaction.reset_index(),
              "paired_answer_changes": pd.DataFrame(paired_rows)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    receipt = {"status": "verified", "source_inputs": sources, "score_inputs": score_receipts,
        "scope": "D/seed1234, E1/E32 x K12/K200, fixed benchmark-supported K12 languages; exploratory CPU analysis",
        "checks": ["all source payload hashes", "exact option coverage", "fixed token counts across checkpoints",
                   "score-spec/request provenance", "parity and replica gates", "dropless routing", "all 24 raw/character macro results reproduce canonical metrics"],
        "metric_notes": {"gold_nll_per_token": "mean per-question correct-continuation NLL divided by scored token count; not corpus validation NLL",
            "choice_ce": "negative log softmax of option scores; char/token variants are diagnostic distributions, not calibrated model probabilities",
            "margin": "correct option score minus highest distractor score, under each normalization",
            "expert_gains": "E32 improvement at each K; loss gains negate E32-E1",
            "coverage_interactions": "expert gain at K200 minus gain at K12; positive means greater broad-coverage benefit",
            "uncertainty": "descriptive only, no independent-question or training-seed inference"},
        "outputs": {p.name: sha256_file(p) for p in output.iterdir()}}
    (output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(interaction[["gold_nll_per_token", "accuracy_raw", "accuracy_char", "accuracy_token", "margin_token"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign.resolve(), args.analysis.resolve(), args.output.resolve())
