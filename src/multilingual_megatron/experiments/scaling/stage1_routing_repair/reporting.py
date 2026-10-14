"""CPU reporting from an immutable, complete final-checkpoint routing release.

The trained trajectory is the replication unit. Blocks quantify evaluation
uncertainty only; layers and languages are never independent training runs.
"""
from __future__ import annotations

import itertools
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import bootstrap_probabilities, entropy, js_matrix, mutual_information
from .campaign import identity, read, sha, validate_panel, verify, write


SPEC = {
    "schema": "stage1_routing_report_spec/v1",
    "interpretation": "Exploratory, after preliminary inspection; no causal or matched-exposure attenuation claim.",
    "panel": "60 sparse final checkpoints; 4008 trained-language streams; 22 matched dense controls",
    "cohorts": {"primary": "common12", "robustness": "common50", "descriptive": "trained"},
    "language_weights": "uniform within each fixed cohort",
    "layer_reducer": "median; recomputed inside every bootstrap draw",
    "metrics": {
        "mi_nats": "H(mean_language p(expert|language)) - mean_language H(p(expert|language))",
        "effective_expert_fraction": "mean_language exp(H(p(expert|language))) / E",
        "pooled_expert_fraction": "exp(H(mean_language p(expert|language))) / E",
        "language_distinctness": "median_layer mean_other_language JS(p_language, p_other), nats",
    },
    "bootstrap": {"replicates": 1000, "block_sequences": 128, "seed": 20260907,
                  "interval": "95% percentile, conditional on checkpoint and fixed cohort; common cohorts only"},
    "contrasts": "All K pairs within ladder/E/seed/horizon and common cohort; shared sequence-block draws",
    "loss_gain": "dense minus sparse unigram-adjusted NLL; equals raw NLL difference on identical targets",
    "associations": "Within-checkpoint/cohort Spearman rho for breadth and distinctness; no p-values",
    "figures": "Primary seed common12, common50 robustness, individual B seed contrasts, common50/K200 associations",
    "input_positions": "Include real special tokens and initial context; exclude synthetic padding",
    "input_hash_note": "Language input_sha256 includes nll_sum; match target hashes and stream/scoring metadata instead.",
    "execution_note": "Use backend eval_micro_batch_size; model micro_batch_size describes training.",
}
ID_FIELDS = ("id", "ladder", "experts", "languages", "seed", "tokens_seen")
METRICS = ("mi_nats", "effective_expert_fraction", "pooled_expert_fraction")


class InputFiles:
    """Verify and record exactly the evidence read, without modifying it."""

    def __init__(self):
        self.entries = {}

    def add(self, entry):
        entry = {"path": str(entry)} if isinstance(entry, (str, Path)) else entry
        path = str(Path(entry["path"]).resolve())
        actual = self.entries.get(path)
        if actual is None:
            actual = identity(path)
            self.entries[path] = actual
        if entry.get("sha256", actual["sha256"]) != actual["sha256"]:
            raise ValueError(f"sealed artifact changed: {path}")
        return Path(path)

    def json(self, entry):
        return read(self.add(entry))

    def check(self):
        for entry in self.entries.values():
            verify(entry)


def routing_metrics(probabilities):
    """Language axis first, expert axis last; optional bootstrap axis between."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim not in (2, 3) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid routing probabilities")
    if not np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("routing probabilities do not sum to one")
    return {
        "mi_nats": mutual_information(p) if len(p) > 1 else np.full(p.shape[1:-1], np.nan),
        "effective_expert_fraction": np.exp(entropy(p)).mean(axis=0) / p.shape[-1],
        "pooled_expert_fraction": np.exp(entropy(p.mean(axis=0))) / p.shape[-1],
    }


def real_block_layout(layer):
    """Keep block IDs and real position totals; empty batch padding is irrelevant."""
    blocks = sorted((int(k), tuple(v)) for k, v in layer["blocks"].items())
    return tuple((k, sum(v)) for k, v in blocks if sum(v) > 0)


def require_stream_match(left, right):
    for key in ("language", "token_count", "target_token_ids_sha256", "unigram_counts_sha256",
                "unigram_log_probability_sum", "special_token_ids", "vocab_size"):
        if left[key] != right[key]:
            raise ValueError(f"validation stream/scoring mismatch: {key}")
    for key in ("sequence_length", "stream_tokens_including_special", "validation_prefixes", "loss_statistic"):
        if left["backend"][key] != right["backend"][key]:
            raise ValueError(f"validation stream/scoring mismatch: backend.{key}")


def matched_gain(sparse, dense):
    require_stream_match(sparse, dense)
    if sparse["token_count"] <= 0:
        raise ValueError("empty scored-token denominator")
    raw = (dense["nll_sum"] - sparse["nll_sum"]) / sparse["token_count"]
    adjusted = dense["atlas_tao_unigram_adjusted_loss"] - sparse["atlas_tao_unigram_adjusted_loss"]
    if not math.isfinite(raw) or not math.isclose(raw, adjusted, rel_tol=0, abs_tol=1e-12):
        raise ValueError("raw and adjusted loss gains disagree")
    return float(adjusted)


def spearman(x, y):
    from scipy.stats import rankdata

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("association has mismatched or nonfinite observations")
    if len(x) < 3:
        return None, "fewer_than_three_languages"
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, "constant_vector"
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]), None


def summarize_layers(layer_metrics, sampled_metrics=None):
    result = {}
    for key in METRICS:
        values = np.asarray([row[key] for row in layer_metrics])
        result[key] = None if np.isnan(values).all() else float(np.median(values))
        result[key + "_ci_low"] = result[key + "_ci_high"] = None
        if sampled_metrics is not None and result[key] is not None:
            draws = np.median([row[key] for row in sampled_metrics], axis=0)
            result[key + "_ci_low"], result[key + "_ci_high"] = map(float, np.quantile(draws, [.025, .975]))
    if sampled_metrics is None:
        return result, None
    draws = {key: np.median([row[key] for row in sampled_metrics], axis=0) for key in METRICS}
    return result, draws


def _pair_key(model):
    return tuple(model[k] for k in ("ladder", "languages", "seed", "tokens_seen", "dataset_shuffle_seed"))


def _model_manifests(model, inputs):
    for name in ("training_manifest", "evaluation_spec"):
        inputs.add({"path": model[name], "sha256": model[name + "_sha256"]})
    manifest = inputs.json({"path": model["evaluation_manifest"], "sha256": model["evaluation_manifest_sha256"]})
    if manifest["status"] != "complete" or manifest["checkpoint"]["step"] != model["checkpoint_step"]:
        raise ValueError("selected model evaluation is incomplete or at the wrong step")
    return manifest


def _require_unique(frame, keys, label):
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate {label}")


def _load_inputs(campaign, inputs):
    import pyarrow.parquet as pq
    from ..stage1_benchmarking.campaign_models import resolve_checkpoint_models

    root = Path(campaign["root"])
    inputs.add(root / "campaign.sha256")
    inputs.add(root / "campaign.json")
    release = inputs.json(root / "analysis/release_manifest.json")
    if (release["status"] != "complete" or release["gate_policy"] != "fixed_execution_v3"
            or release["campaign_sha256"] != sha(root / "campaign.json")):
        raise ValueError("report requires the complete, campaign-bound V3 release")
    expected_bootstrap = SPEC["bootstrap"]
    if release["bootstrap"] != {"block_sequences": expected_bootstrap["block_sequences"],
                                "bootstrap_replicates": expected_bootstrap["replicates"],
                                "seed": expected_bootstrap["seed"]}:
        raise ValueError("release bootstrap contract differs")
    tables = {}
    for name, entry in release["table_inventory"].items():
        tables[name] = pq.read_table(inputs.add(entry)).to_pandas()
        if len(tables[name]) != entry["rows"]:
            raise ValueError(f"release row-count drift: {name}")
    _require_unique(tables["routing_languages"], ["id", "language", "layer_index"], "routing language/layer")
    _require_unique(tables["cohort_summary"], ["id", "cohort", "layer_index"], "cohort/layer")
    completion = inputs.json(root / "completeness.json")
    if completion["status"] != "complete" or len(completion["results"]) != 60:
        raise ValueError("incomplete routing campaign")
    selected = {Path(e["path"]).stem: e for e in completion["results"]}
    if len(selected) != 60:
        raise ValueError("duplicate selected production result")
    controls_hash = inputs.add(root / "controls_complete.sha256").read_text().strip()
    controls = inputs.json({"path": str(root / "controls_complete.json"), "sha256": controls_hash})
    if controls["status"] != "complete" or controls["campaign_sha256"] != release["campaign_sha256"]:
        raise ValueError("recovery controls are not complete for this campaign")
    for entry in controls["evidence"]:
        inputs.add(entry)
    pilots = inputs.json(root / "pilots_complete.json")
    if pilots["status"] != "complete" or len(pilots["pilots"]) != 3:
        raise ValueError("pilot gate incomplete")
    for entry in pilots["pilots"]:
        pilot = inputs.json(entry)
        if pilot["status"] != "complete":
            raise ValueError("pilot result incomplete")
        inputs.add(pilot["receipt"])
        for evidence in pilot["manifests"].values():
            inputs.add(evidence)
    source_release = inputs.json(campaign["release"])
    models, sources = resolve_checkpoint_models(campaign["runtime"], source_release, Path(campaign["release"]["path"]))
    for entry in sources.values():
        if "path" in entry and "sha256" in entry:
            inputs.add(entry)
    tasks = [t for t in campaign["tasks"] if t["kind"] == "production"]
    validate_panel([t["model"] for t in tasks])
    if {t["id"] for t in tasks} != set(selected):
        raise ValueError("completion does not select the declared panel")
    dense = {}
    for model in models:
        if model["experts"] == 1:
            key = _pair_key(model)
            if key in dense:
                raise ValueError("duplicate dense baseline")
            dense[key] = (model, _model_manifests(model, inputs))
    if len(dense) != 22 or any(_pair_key(t["model"]) not in dense for t in tasks):
        raise ValueError("missing matched dense controls")
    return release, tables, tasks, selected, dense


def _validate_language(row, old, model, spec, expected_layers):
    require_stream_match(row, old)
    if row["token_count"] <= 0 or row["nll_sum"] != old["nll_sum"]:
        raise ValueError("historical scored-token count or loss changed")
    if not math.isfinite(row["nll_sum"]):
        raise ValueError("nonfinite validation loss")
    routing = row["moe_routing"]
    if (routing["schema"] != "multilingual_moe_routing_diagnostics/v2"
            or not routing["evidence_available"] or routing["evidence_errors"]
            or routing["block_sequences"] != SPEC["bootstrap"]["block_sequences"]):
        raise ValueError("invalid corrected routing evidence")
    if [r["layer_index"] for r in routing["layers"]] != list(range(expected_layers)):
        raise ValueError("missing or reordered routing layers")
    if row["backend"]["eval_micro_batch_size"] != spec["eval_micro_batch_size"]:
        raise ValueError("actual evaluator batch differs from spec")
    if row["backend"]["checkpoint_step"] != model["checkpoint_step"]:
        raise ValueError("language evidence at wrong checkpoint step")
    n = row["backend"]["stream_tokens_including_special"]
    layouts = []
    for layer in routing["layers"]:
        counts = np.asarray(layer["expert_token_counts"], dtype=np.int64)
        blocks = np.asarray(list(layer["blocks"].values()), dtype=np.int64)
        if (len(counts) != model["experts"] or layer["topk"] != 1 or (counts < 0).any()
                or sum(counts) != n or layer["forwarded_tokens"] != n or layer["dropped_assignments"]
                or (blocks < 0).any() or not np.array_equal(blocks.sum(axis=0), counts)):
            raise ValueError("valid routing/block count conservation failed")
        if not np.array_equal(counts + layer["padding_expert_token_counts"], layer["all_expert_token_counts"]):
            raise ValueError("valid + padding != all expert assignments")
        if sum(layer["all_expert_token_counts"]) != layer["all_positions"]:
            raise ValueError("all-position count conservation failed")
        layouts.append(real_block_layout(layer))
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError("sequence blocks do not align across layers")
    return layouts[0]


def compute_report(campaign):
    """Return tables, audit and verified input inventory; never write evaluation data."""
    inputs = InputFiles()
    release, original, tasks, selected, dense = _load_inputs(campaign, inputs)
    original_layers = original["routing_languages"].set_index(["id", "language", "layer_index"])
    original_summary = original["cohort_summary"].set_index(["id", "cohort", "layer_index"])
    checkpoint_rows, language_rows, loss_rows, padding_rows = [], [], [], []
    draws_by_group, streams, execution, dense_cache = {}, {}, {}, {}
    valid_total = scored_total = observed_layers = 0
    for index, task in enumerate(sorted(tasks, key=lambda t: t["id"])):
        model = task["model"]
        result = inputs.json(selected[task["id"]])
        receipt = inputs.json(result["receipt"])
        if (result["status"] != "complete" or receipt["status"] != "complete"
                or result["task_id"] != task["id"] or receipt["task_id"] != task["id"]
                or receipt["attempt"] != result["attempt"]):
            raise ValueError("invalid selected production receipt")
        manifest = inputs.json(result["manifests"]["corrected"])
        spec_entry = task["specs"][str(result["attempt"])]["corrected"]
        spec = inputs.json(spec_entry)
        if (manifest["status"] != "complete" or manifest["spec"]["sha256"] != spec_entry["sha256"]
                or manifest["checkpoint"]["step"] != model["checkpoint_step"]
                or set(manifest["languages"]) != set(task["languages"])):
            raise ValueError("corrected manifest does not match selected task")
        old_manifest = _model_manifests(model, inputs)
        dense_model, dense_manifest = dense[_pair_key(model)]
        dense_spec = inputs.json({"path": dense_model["evaluation_spec"], "sha256": dense_model["evaluation_spec_sha256"]})
        if spec["context_token_id"] != dense_spec["context_token_id"]:
            raise ValueError("dense/sparse context token differs")
        for field in ("tokenizer_model", "validation_data_manifest"):
            if manifest[field]["sha256"] != dense_manifest[field]["sha256"]:
                raise ValueError(f"dense/sparse {field} differs")
            inputs.add(manifest[field])
        n_layers = spec["model_recipe"]["expected_model"]["num_layers"]
        meta = {key: model[key] for key in ID_FIELDS}
        meta.update(active_parameters=model["active_parameters"], num_layers=n_layers,
                    training_micro_batch_size=model["micro_batch_size"],
                    eval_micro_batch_size=spec["eval_micro_batch_size"],
                    expert_parallel_size=model["expert_parallel_size"], allocated_gpus=task["gpus"],
                    moe_dispatcher=model["moe_dispatcher"])
        condition = (spec["eval_micro_batch_size"], model["expert_parallel_size"], task["gpus"],
                     model["moe_dispatcher"], spec["sequence_length"], spec["context_token_id"],
                     json.dumps(spec["model_recipe"], sort_keys=True))
        condition_key = (model["ladder"], model["experts"])
        if execution.setdefault(condition_key, condition) != condition:
            raise ValueError("evaluation conditions differ across K/seeds within ladder/E")
        rows, gains, historical_rows = {}, {}, {}
        for language in sorted(task["languages"]):
            row = inputs.json(manifest["languages"][language])
            old = inputs.json(old_manifest["languages"][language])
            layout = _validate_language(row, old, model, spec, n_layers)
            stream = (row["target_token_ids_sha256"], row["token_count"],
                      tuple(row["backend"]["validation_prefixes"]), spec["context_token_id"],
                      spec["sequence_length"], layout)
            if streams.setdefault(language, stream) != stream:
                raise ValueError(f"real sequence blocks/stream do not align across checkpoints: {language}")
            cache_key = (dense_model["id"], language)
            if cache_key not in dense_cache:
                dense_cache[cache_key] = inputs.json(dense_manifest["languages"][language])
            gain = matched_gain(row, dense_cache[cache_key])
            rows[language], gains[language], historical_rows[language] = row, gain, old
            valid_total += row["backend"]["stream_tokens_including_special"]
            scored_total += row["token_count"]
            loss_rows.append({**meta, "language": language, "dense_id": dense_model["id"],
                              "scored_tokens": row["token_count"], "moe_gain_nats": gain,
                              "sparse_raw_nll": row["raw_nll"], "dense_raw_nll": dense_cache[cache_key]["raw_nll"],
                              "target_token_ids_sha256": row["target_token_ids_sha256"]})
            for layer in row["moe_routing"]["layers"]:
                observed_layers += 1
                previous = original_layers.loc[(model["id"], language, layer["layer_index"])]
                for field in ("expert_token_counts", "padding_expert_token_counts", "all_expert_token_counts"):
                    if not np.array_equal(previous[field], layer[field]):
                        raise ValueError("released table differs from raw language counts")
                if not np.array_equal(previous["historical_expert_token_counts"],
                                      old["moe_routing"]["layers"][layer["layer_index"]]["expert_token_counts"]):
                    raise ValueError("released historical counts differ from old artifact")
        cohorts = {"trained": sorted(rows)}
        for size in (12, 50):
            members = campaign["cohorts"][f"k{size}"]
            if len(members) != size or len(set(members)) != size:
                raise ValueError("invalid fixed cohort")
            if set(members).issubset(rows):
                cohorts[f"common{size}"] = sorted(members)
        # One set of draws per real stream, reused across layers and cohorts.
        sampled_by_layer = []
        common = sorted(set().union(*(set(v) for k, v in cohorts.items() if k != "trained")))
        for layer_index in range(n_layers):
            sampled_by_layer.append({lang: bootstrap_probabilities(
                rows[lang]["moe_routing"]["layers"][layer_index], lang,
                SPEC["bootstrap"]["replicates"], SPEC["bootstrap"]["seed"]) for lang in common})
        for cohort, members in cohorts.items():
            per_layer, sampled, old_mi, all_mi = [], [], [], []
            breadth = {lang: [] for lang in members}
            distinctness = {lang: [] for lang in members}
            for layer_index in range(n_layers):
                layers = [rows[lang]["moe_routing"]["layers"][layer_index] for lang in members]
                counts = np.asarray([layer["expert_token_counts"] for layer in layers], dtype=float)
                p = counts / counts.sum(axis=-1, keepdims=True)
                metrics = routing_metrics(p)
                per_layer.append(metrics)
                ref = original_summary.loc[(model["id"], cohort, layer_index)]
                for actual, expected in ((metrics["mi_nats"], ref["language_expert_mi_nats"]),
                                         (metrics["effective_expert_fraction"] * model["experts"], ref["mean_language_effective_experts"])):
                    if not np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True):
                        raise ValueError("reported metric differs from released layer statistic")
                for count_rows, reference, values in (
                        ([layer["all_expert_token_counts"] for layer in layers], ref["new_all_position_mi_nats"], all_mi),
                        ([historical_rows[lang]["moe_routing"]["layers"][layer_index]["expert_token_counts"] for lang in members],
                         ref["historical_v1_mi_nats"], old_mi)):
                    c = np.asarray(count_rows, dtype=float)
                    value = routing_metrics(c / c.sum(axis=-1, keepdims=True))["mi_nats"]
                    if not np.isclose(value, reference, rtol=0, atol=1e-12, equal_nan=True):
                        raise ValueError("released padding/historical metric differs from counts")
                    values.append(value)
                if cohort != "trained":
                    sampled.append(routing_metrics(np.stack([sampled_by_layer[layer_index][lang] for lang in members])))
                    js = js_matrix(p)
                    for i, lang in enumerate(members):
                        breadth[lang].append(float(np.exp(entropy(p[i])) / model["experts"]))
                        distinctness[lang].append(float(js[i].sum() / (len(members) - 1)))
            summary, draws = summarize_layers(per_layer, sampled if sampled else None)
            summary_row = {**meta, "cohort": cohort, "cohort_size": len(members), **summary,
                           "mi_over_log_experts": None if summary["mi_nats"] is None else summary["mi_nats"] / math.log(model["experts"])}
            checkpoint_rows.append(summary_row)
            historical = None if len(members) == 1 else float(np.median(old_mi))
            all_position = None if len(members) == 1 else float(np.median(all_mi))
            padding_rows.append({**meta, "cohort": cohort, "cohort_size": len(members),
                                 "valid_mi_nats": summary["mi_nats"], "all_position_mi_nats": all_position,
                                 "historical_v1_mi_nats": historical,
                                 "valid_minus_all_mi_nats": None if all_position is None else summary["mi_nats"] - all_position,
                                 "historical_minus_all_mi_nats": None if historical is None else historical - all_position})
            if cohort != "trained":
                key = (model["ladder"], model["experts"], model["seed"], model["tokens_seen"], cohort)
                draws_by_group.setdefault(key, {})[model["languages"]] = (summary_row, draws)
                for lang in members:
                    language_rows.append({**meta, "cohort": cohort, "cohort_size": len(members), "language": lang,
                                          "effective_expert_fraction": float(np.median(breadth[lang])),
                                          "mean_pair_js_nats": float(np.median(distinctness[lang])), "moe_gain_nats": gains[lang]})
        print(f"routing report: {index + 1}/60 {model['id']} verified and summarized", flush=True)
    if (valid_total != release["valid_input_positions"] or scored_total != release["scored_target_tokens"]
            or observed_layers != len(original_layers)):
        raise ValueError("campaign token totals or layer-language coverage differ")
    contrasts = []
    for key, by_k in sorted(draws_by_group.items()):
        ladder, experts, seed, tokens, cohort = key
        for low, high in itertools.combinations(sorted(by_k), 2):
            left, left_draws = by_k[low]
            right, right_draws = by_k[high]
            for metric in METRICS:
                delta = right_draws[metric] - left_draws[metric]
                lo, hi = np.quantile(delta, [.025, .975])
                contrasts.append({"ladder": ladder, "experts": experts, "seed": seed, "tokens_seen": tokens,
                                  "cohort": cohort, "K_from": low, "K_to": high, "metric": metric,
                                  "delta": right[metric] - left[metric], "ci_low": float(lo), "ci_high": float(hi)})
    language_frame = pd.DataFrame(language_rows)
    associations = []
    for (model_id, cohort), group in language_frame.groupby(["id", "cohort"], sort=True):
        meta = {key: group.iloc[0][key] for key in ID_FIELDS}
        for predictor in ("effective_expert_fraction", "mean_pair_js_nats"):
            rho, reason = spearman(group[predictor], group["moe_gain_nats"])
            associations.append({**meta, "cohort": cohort, "predictor": predictor, "n_languages": len(group),
                                 "spearman_rho": rho, "undefined_reason": reason})
    tables = {"checkpoint_summary": pd.DataFrame(checkpoint_rows), "coverage_contrasts": pd.DataFrame(contrasts),
              "language_metrics": language_frame, "loss_gains": pd.DataFrame(loss_rows),
              "loss_associations": pd.DataFrame(associations), "padding_impact": pd.DataFrame(padding_rows)}
    audit = {"checkpoints": len(tasks), "dense_controls": len(dense), "language_streams": len(loss_rows),
             "valid_input_positions": valid_total, "scored_target_tokens": scored_total,
             "language_layers": observed_layers, "max_historical_nll_sum_difference": 0.0,
             "stream_and_block_alignment": "verified", "fixed_execution_within_ladder_experts": "verified"}
    _validate_tables(tables)
    return tables, audit, inputs


def _validate_tables(tables):
    expected = {"checkpoint_summary": 132, "coverage_contrasts": 156, "loss_gains": 4008,
                "language_metrics": 1928, "loss_associations": 144, "padding_impact": 132}
    for name, frame in tables.items():
        if len(frame) != expected[name]:
            raise ValueError(f"unexpected reporting coverage: {name} {len(frame)} != {expected[name]}")
    _require_unique(tables["checkpoint_summary"], ["id", "cohort"], "checkpoint/cohort summary")
    _require_unique(tables["loss_gains"], ["id", "language"], "dense/sparse language match")
    _require_unique(tables["language_metrics"], ["id", "cohort", "language"], "language metric")
    _require_unique(tables["coverage_contrasts"], ["ladder", "experts", "seed", "tokens_seen", "cohort", "K_from", "K_to", "metric"], "coverage contrast")
    _require_unique(tables["loss_associations"], ["id", "cohort", "predictor"], "loss association")
    summary = tables["checkpoint_summary"]
    informative = summary[summary.cohort_size > 1]
    if (informative.mi_nats < -1e-12).any() or (informative.mi_nats > np.log(np.minimum(informative.experts, informative.cohort_size)) + 1e-12).any():
        raise ValueError("mutual information outside theoretical bounds")
    if summary.loc[summary.cohort_size == 1, "mi_nats"].notna().any():
        raise ValueError("K=1 language information must remain undefined")
    for column in ("effective_expert_fraction", "pooled_expert_fraction"):
        if ((summary[column] < 1 / summary.experts - 1e-12) | (summary[column] > 1 + 1e-12)).any():
            raise ValueError("expert breadth outside theoretical bounds")


def verify_report(campaign, output, *, replay=False):
    """Read-only artifact verification, optionally replaying all numerical analysis."""
    output = Path(output).resolve()
    manifest = read(output / "release_manifest.json")
    if manifest["schema"] != "stage1_routing_report/v1" or manifest["status"] != "complete":
        raise ValueError("report release is incomplete or unsupported")
    if manifest["campaign_sha256"] != sha(Path(campaign["root"]) / "campaign.json"):
        raise ValueError("report belongs to another campaign")
    for entry in manifest["inputs"] + manifest["code"]:
        verify(entry)
    for name, entry in manifest["outputs"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output):
            raise ValueError("output inventory escapes report directory")
        verify({"path": str(path), "sha256": entry["sha256"]})
    tables = {name: pd.read_parquet(output / f"tables/{name}.parquet") for name in manifest["table_rows"]}
    _validate_tables(tables)
    if read(output / "analysis_spec.json") != SPEC:
        raise ValueError("report specification differs from implementation")
    if replay:
        replayed, audit, _ = compute_report(campaign)
        for name in tables:
            pd.testing.assert_frame_equal(tables[name], replayed[name], check_exact=True)
        if audit != manifest["audit"]:
            raise ValueError("numerical replay audit differs")
    return {"status": "verified", "numerical_replay": replay, "tables": manifest["table_rows"], "output": str(output)}


def report(campaign, output, *, verify_only=False, replay=False):
    from .report_figures import render_figures, write_findings

    output = Path(output).resolve()
    if verify_only:
        return verify_report(campaign, output, replay=replay)
    if replay:
        raise ValueError("--replay requires --verify")
    if output.is_relative_to(Path(campaign["root"]).resolve()):
        raise ValueError("report output must be separate from the sealed campaign")
    from ..stage1_benchmarking import campaign_models
    from ..stage1_benchmarking import api as stage1_benchmark
    code_paths = [Path(__file__), Path(__file__).with_name("report_figures.py"), Path(__file__).with_name("__main__.py"),
                  Path(__file__).with_name("analysis.py"), Path(__file__).with_name("campaign.py"),
                  Path(campaign_models.__file__), Path(stage1_benchmark.__file__)]
    code = [identity(p) for p in code_paths]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "analysis_spec.json", SPEC)
    tables, audit, inputs = compute_report(campaign)
    (output / "tables").mkdir()
    for name, frame in tables.items():
        frame.to_parquet(output / f"tables/{name}.parquet", index=False)
        frame.to_csv(output / f"tables/{name}.csv", index=False, float_format="%.17g")
    render_figures(tables, output)
    write_findings(tables, audit, output)
    inputs.check()
    for entry in code:
        verify(entry)
    import matplotlib
    import pyarrow
    import scipy
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = {"schema": "stage1_routing_report/v1", "status": "complete", "campaign_sha256": sha(Path(campaign["root"]) / "campaign.json"),
                "audit": audit, "table_rows": {name: len(frame) for name, frame in tables.items()},
                "inputs": sorted(inputs.entries.values(), key=lambda e: e["path"]), "code": code,
                "git_commit": commit, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                                         "matplotlib": matplotlib.__version__, "pyarrow": pyarrow.__version__, "scipy": scipy.__version__},
                "outputs": {str(p.relative_to(output)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
                            for p in sorted(output.rglob("*")) if p.is_file()}}
    write(output / "release_manifest.json", manifest)
    return verify_report(campaign, output)
