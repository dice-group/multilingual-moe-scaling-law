"""Pinned data and likelihood metrics for the supplementary Stage 1 pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from .contracts import _atomic_json, _verify_payload_hash
from .full_suite_campaign import checked, read, ref, save, prepare_benchmark_inputs
from .requests import _request_id

BENCHMARKS = ("multiblimp", "x_csqa", "xcomps")
PRIMARY = {"multiblimp": "acc", "x_csqa": "acc_pmi", "xcomps": "acc"}


def download(profile: dict, repo: Path, output: Path) -> dict:
    """Fetch only pinned public files; resume complete downloads with receipts."""
    import requests
    audit = repo / profile["coverage_audit"]
    verification = read(audit / "verification.json")
    mapping = audit / "language_mapping.csv"
    if sha256_file(mapping) != verification["outputs_sha256"][mapping.name]:
        raise ValueError("candidate language inventory changed")
    entries = []
    for row in csv.DictReader(mapping.open()):
        family, code = row["benchmark"], row["dataset_code"]
        if family not in BENCHMARKS or row["in_K200"] != "True":
            continue
        key = "xcsr" if family == "x_csqa" else family
        pin = verification["revisions"][key]
        if family == "xcomps":
            source = f"data/comps_{code}.json"
            url = f"https://raw.githubusercontent.com/{pin['repo']}/{pin['revision']}/{source}"
        else:
            metadata_path = audit / "sources" / f"{key}_api.json"
            if sha256_file(metadata_path) != verification["source_sha256"][metadata_path.name]:
                raise ValueError("pinned dataset metadata changed")
            metadata = read(metadata_path)
            if metadata["sha"] != pin["revision"]:
                raise ValueError("dataset revision mismatch")
            if family == "multiblimp":
                source = f"{code}/data.tsv"
            else:
                matches = [x["rfilename"] for x in metadata["siblings"]
                           if x["rfilename"].startswith(f"X-CSQA-{code}/validation-")]
                if len(matches) != 1:
                    raise ValueError("expected one complete X-CSQA validation shard")
                source = matches[0]
            url = f"https://huggingface.co/datasets/{pin['repo']}/resolve/{pin['revision']}/{source}"
        entries.append(dict(benchmark=family, language=code,
                            corpus_language=row["corpus_language"], mapping_policy=row["mapping_policy"],
                            repo=pin["repo"], revision=pin["revision"], source_path=source, url=url,
                            path=str(output / "sources" / family / (code + Path(source).suffix))))
    if Counter(x["benchmark"] for x in entries) != {"multiblimp": 61, "x_csqa": 16, "xcomps": 17}:
        raise ValueError("unexpected candidate language coverage")

    def fetch(item):
        path = Path(item["path"])
        receipt = path.with_suffix(path.suffix + ".receipt.json")
        if receipt.exists():
            prior = read(receipt)
            if any(prior[k] != v for k, v in item.items()) or sha256_file(path) != prior["sha256"]:
                raise ValueError(f"download receipt drift: {path}")
            return prior
        if path.exists():
            raise ValueError(f"unreceipted download: {path}")
        response = requests.get(item["url"], timeout=180)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        result = {**item, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        _atomic_json(receipt, result)
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        sources = list(pool.map(fetch, entries))
    payload = {"schema": "stage1_candidate_sources/v1", "coverage_audit": ref(audit / "verification.json"),
               "language_mapping": ref(mapping), "sources": sources}
    save(output / "source_inventory.json", payload, "manifest_sha256")
    return {"downloaded_languages": len(sources), "bytes": sum(x["bytes"] for x in sources)}


def documents(item):
    path = checked(item)
    if item["benchmark"] == "multiblimp":
        with path.open() as handle:
            return list(enumerate(csv.DictReader(handle, delimiter="\t")))
    if item["benchmark"] == "xcomps":
        return list(read(path).items())
    import pyarrow.parquet as pq
    return list(enumerate(pq.read_table(path).to_pylist()))


def render(family, row):
    """Keep complete sentences unchanged; X-CSQA uses frozen cloze + PMI."""
    if family == "multiblimp":
        context, choices, target = "", [row["sen"], row["wrong_sen"]], 0
        group = row.get("phenomenon", "unknown")
    elif family == "xcomps":
        context, choices, target = "", [row["acceptable_sent"], row["unacceptable_sent"]], 0
        group = row["negative_sample_type"]
    elif family == "x_csqa":
        question = row["question"]
        choices = question["choices"]["text"]
        labels = question["choices"]["label"]
        if len(choices) != 5 or len(set(labels)) != 5 or row["answerKey"] not in labels:
            raise ValueError("missing/invalid X-CSQA answer label")
        context, target, group = "Question: " + question["stem"].strip() + "\nAnswer:", labels.index(row["answerKey"]), "all"
    else:
        raise ValueError(f"unknown candidate {family}")
    if any(not isinstance(x, str) or not x.strip() for x in choices):
        raise ValueError("empty answer/sentence")
    if family != "x_csqa" and choices[0] == choices[1]:
        raise ValueError("identical minimal pair")
    return context, choices, target, group


def audit_sources(inventory, output):
    """Only mechanical invalidities can be excluded; preserve every source ID."""
    invalid, counts = [], Counter()
    for item in inventory["sources"]:
        rows = documents(item)
        counts[item["benchmark"]] += len(rows)
        for index, (source_id, row) in enumerate(rows):
            try:
                render(item["benchmark"], row)
            except ValueError as exc:
                if str(exc) not in {"empty answer/sentence", "identical minimal pair"}:
                    raise
                invalid.append(dict(benchmark=item["benchmark"], language=item["language"], doc_id=index,
                    source_id=str(row.get("id", source_id)), reason=str(exc),
                    group=row.get("negative_sample_type", row.get("phenomenon", "all")),
                    document_sha256=canonical_json_sha256(row)))
    payload = dict(source_inventory=ref(output / "source_inventory.json"), counts=dict(counts), invalid=invalid,
                   policy="Exclude only empty options/sentences and identical minimal pairs; retain exact source IDs and hashes")
    save(output / "invalid_source_audit.json", payload, "manifest_sha256")
    with (output / "excluded_questions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(invalid[0]) if invalid else ["benchmark", "language", "doc_id", "reason"])
        writer.writeheader()
        writer.writerows(invalid)
    return invalid


def freeze_language(args):
    item, output, tokenizer_cfg, sequence_length = args
    import sentencepiece as spm
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import encode_lm_eval_pair, prepare_scored_sequence
    from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import TokenCacheWriter
    tokenizer = SimpleNamespace(tokenize=spm.SentencePieceProcessor(model_file=tokenizer_cfg["model"]).encode)
    family, code = item["benchmark"], item["language"]
    task = f"stage1_candidate_{family}_{code}"
    requests, docs, sequences = [], [], []
    allowed = {x["doc_id"]: x for x in item.get("approved_exclusions", [])}
    exclusions = []
    for doc_id, (source_id, row) in enumerate(documents(item)):
        try:
            context, choices, target, group = render(family, row)
        except ValueError as exc:
            approved = allowed.get(doc_id)
            if approved is None or approved["document_sha256"] != canonical_json_sha256(row) or approved["reason"] != str(exc):
                raise ValueError(f"{family}/{code}/{source_id}: {exc}") from exc
            exclusions.append(approved)
            continue
        if doc_id in allowed:
            raise ValueError("audited invalidity no longer reproduces")
        doc = dict(doc_id=doc_id, source_id=str(row.get("id", source_id)), target=target, group=group,
                   char_lengths=[len(x) for x in choices], byte_lengths=[len(x.encode()) for x in choices],
                   conditional=[], unconditional=[], token_lengths=[], max_width=0,
                   source_document_sha256=canonical_json_sha256(row))
        if family == "xcomps":
            doc["concept"] = row["acceptable_concept"]
            doc["property"] = row["property"]
        if family == "multiblimp":
            # Preserve sentence/treebank identity for clustered uncertainty.
            doc["metadata"] = row.get("metadata", "")
        for unconditional in ([False, True] if family == "x_csqa" else [False]):
            for i, choice in enumerate(choices):
                ctx = "Answer:" if unconditional else context
                continuation = (" " if family == "x_csqa" else "") + choice
                request_doc_id = doc_id + (1_000_000 if unconditional else 0)
                rid = _request_id(task, request_doc_id, i, (ctx, continuation))
                context_ids, target_ids = encode_lm_eval_pair(tokenizer, ctx, continuation,
                                                            context_token_id=tokenizer_cfg["context_token_id"])
                sequence = prepare_scored_sequence(request_index=len(requests), request_id=rid,
                    context_ids=context_ids, continuation_ids=target_ids, max_length=sequence_length)
                if sequence.truncated_tokens:
                    raise ValueError(f"candidate request would be truncated: {task}/{source_id}")
                requests.append(dict(request_id=rid, task=task, benchmark=family, language=code,
                                     doc_id=request_doc_id, choice_index=i, request_type="loglikelihood",
                                     context=ctx, continuation=continuation))
                sequences.append(sequence)
                doc["unconditional" if unconditional else "conditional"].append(rid)
                if not unconditional:
                    doc["token_lengths"].append(len(target_ids))
                doc["max_width"] = max(doc["max_width"], len(sequence.input_ids))
        docs.append(doc)
    if len(exclusions) != len(allowed):
        raise ValueError("audited exclusion is missing from source")
    if family == "x_csqa" and len(docs) + len(exclusions) != 1000:
        raise ValueError(f"incomplete X-CSQA labeled validation: {code}/{len(docs)}")
    smoke_ids = {d["doc_id"] for d in docs[:2] + docs[-2:] + sorted(docs, key=lambda d: d["max_width"])[-2:]}
    receipts = {}
    for scope in ("full", "smoke"):
        selected = docs if scope == "full" else [d for d in docs if d["doc_id"] in smoke_ids]
        ids = {rid for d in selected for rid in d["conditional"] + d["unconditional"]}
        selected_requests = [r for r in requests if r["request_id"] in ids]
        metadata = dict(task=task, benchmark=family, language=code, corpus_language=item["corpus_language"],
                        output_type="multiple_choice", documents=selected, source=item,
                        exclusions=exclusions if scope == "full" else [],
                        metrics=["acc", "acc_token", "acc_byte", "margin"] + (["acc_pmi"] if family == "x_csqa" else []),
                        primary_metric=PRIMARY[family])
        payload = dict(schema="stage1_lm_eval_frozen_requests/v1", protocol="stage1_candidate_likelihood_v1",
                       sampling=None if scope == "full" else "first/last/longest pairs per language; diagnostic only",
                       tasks=[metadata], requests=selected_requests)
        path = Path(output) / scope / "languages" / (task + ".json")
        request_ref = save(path, payload, "manifest_sha256")
        writer = TokenCacheWriter(Path(output) / scope / "token_cache" / task)
        histogram = Counter()
        count = 0
        for sequence in sequences:
            if sequence.request_id in ids:
                writer.append(replace(sequence, request_index=count))
                histogram[len(sequence.input_ids)] += 1
                count += 1
        token_ref = writer.finish(request_sha256=request_ref["sha256"], tokenizer=tokenizer_cfg, sequence_length=sequence_length)
        receipts[scope] = dict(**request_ref, benchmark=family, language=code, corpus_language=item["corpus_language"],
                               documents=len(selected), requests=len(selected_requests), token_cache=token_ref,
                               width_histogram={str(k): v for k, v in sorted(histogram.items())})
    return receipts


def freeze(profile, repo, output, workers):
    inventory = read(output / "source_inventory.json")
    _verify_payload_hash(inventory, field="manifest_sha256", label="candidate sources")
    tokenizer = dict(profile["tokenizer"])
    tokenizer["model"] = str((repo / tokenizer["model"]).resolve())
    if sha256_file(Path(tokenizer["model"])) != tokenizer["sha256"]:
        raise ValueError("tokenizer drift")
    if (output / "full").exists() or (output / "smoke").exists():
        raise ValueError("refusing to overwrite frozen candidate requests")
    exclusions = audit_sources(inventory, output)
    sources = [{**item, "approved_exclusions": [x for x in exclusions if (x["benchmark"], x["language"]) == (item["benchmark"], item["language"])]}
               for item in inventory["sources"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(freeze_language, [(item, str(output), tokenizer, 2048) for item in sources]))
    for scope in ("full", "smoke"):
        data = dict(schema="stage1_candidate_data/v1", sampling=None if scope == "full" else "diagnostic_subset",
                    source_inventory=ref(output / "source_inventory.json"), tokenizer=tokenizer, sequence_length=2048,
                    languages=[r[scope] for r in receipts], families=[], exclusions=exclusions if scope == "full" else [],
                    exclusion_audit=ref(output / "invalid_source_audit.json"), unsupported_long_targets=[])
        for family in BENCHMARKS:
            fragments = [read(checked(x)) for x in data["languages"] if x["benchmark"] == family]
            payload = dict(schema="stage1_lm_eval_frozen_requests/v1", protocol="stage1_candidate_likelihood_v1",
                           sampling=data["sampling"], tasks=[t for f in fragments for t in f["tasks"]],
                           requests=[r for f in fragments for r in f["requests"]])
            path = output / scope / "families" / (family + ".json")
            entry = save(path, payload, "manifest_sha256")
            data["families"].append(dict(**entry, benchmark=family, documents=sum(len(t["documents"]) for t in payload["tasks"]),
                                         requests=len(payload["requests"]), languages=len(fragments)))
        data["inputs"] = prepare_benchmark_inputs(data, output / scope)
        save(output / scope / "data_manifest.json", data, "manifest_sha256")
    return {scope: [{k: f[k] for k in ("benchmark", "documents", "requests", "languages")}
                    for f in read(output / scope / "data_manifest.json")["families"]] for scope in ("full", "smoke")}


def aggregate(manifest, scores):
    """Strict minimal-pair comparisons avoid gold-first tie inflation."""
    expected = {r["request_id"] for r in manifest["requests"]}
    by_id = {s["request_id"]: s for s in scores}
    if len(by_id) != len(scores) or set(by_id) != expected:
        raise ValueError("score request coverage mismatch")
    if any(not math.isfinite(s["loglikelihood"]) or s["scored_tokens"] < 1 for s in scores):
        raise ValueError("nonfinite score or empty token count")
    metrics = []
    for task in manifest["tasks"]:
        observations = []
        for doc in task["documents"]:
            values = [by_id[r]["loglikelihood"] for r in doc["conditional"]]
            tokens = [by_id[r]["scored_tokens"] for r in doc["conditional"]]
            if tokens != doc["token_lengths"]:
                raise ValueError("scored token counts differ from frozen tokenizer")
            target = doc["target"]
            normalized = [v / n for v, n in zip(values, tokens)]
            byte_normalized = [v / n for v, n in zip(values, doc["byte_lengths"])]
            pair = task["benchmark"] != "x_csqa"
            def correct(v):
                return float(v[0] > v[1]) if pair else float(max(range(len(v)), key=v.__getitem__) == target)
            record = dict(acc=correct(values), acc_token=correct(normalized), acc_byte=correct(byte_normalized),
                          margin=values[target] - max(v for i, v in enumerate(values) if i != target),
                          margin_token=normalized[target] - max(v for i, v in enumerate(normalized) if i != target),
                          tie=float(values.count(max(values)) > 1), group=doc["group"])
            if doc["unconditional"]:
                pmi = [v - by_id[r]["loglikelihood"] for v, r in zip(values, doc["unconditional"])]
                record["acc_pmi"] = correct(pmi)
                record["margin_pmi"] = pmi[target] - max(v for i, v in enumerate(pmi) if i != target)
            observations.append(record)
        for group in ["all"] + sorted({o["group"] for o in observations} - {"all"}):
            selected = observations if group == "all" else [o for o in observations if o["group"] == group]
            degenerate = sum(x["reason"] == "identical minimal pair" and (group == "all" or x["group"] == group)
                             for x in task.get("exclusions", []))
            metrics.append(dict(benchmark=task["benchmark"], language=task["language"], corpus_language=task["corpus_language"],
                                group=group, samples=len(selected), chance=.2 if task["benchmark"] == "x_csqa" else .5,
                                primary_metric=task["primary_metric"],
                                excluded_identical_pairs=degenerate,
                                acc_with_identical_pairs=sum(o["acc"] for o in selected)/(len(selected)+degenerate),
                                **{key: sum(o[key] for o in selected) / len(selected) for key in selected[0] if key != "group"}))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "freeze"])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    profile = read(args.profile)
    repo = (args.profile.resolve().parent / profile["repo_root"]).resolve()
    output = args.output.resolve()
    result = download(profile, repo, output) if args.command == "download" else freeze(profile, repo, output, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
