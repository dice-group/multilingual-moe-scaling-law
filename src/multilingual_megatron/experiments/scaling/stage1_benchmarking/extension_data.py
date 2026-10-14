"""Frozen requests and corpus-weighted metrics for external loss/classification."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from .contracts import _verify_payload_hash
from .extension_sources import FAMILIES, LABELS
from .full_suite_campaign import checked, read, ref, save, prepare_benchmark_inputs

LOSS = {"flores_mono", "ntrex_mono"}
PRIMARY = {"flores_mono": "bits_per_byte", "ntrex_mono": "bits_per_byte", "arc_easy_eu20": "acc_char", "sib200": "acc", "taxi1500": "acc"}


def classification_prompt(family, text, demonstrations):
    if family == "arc_easy_eu20":
        return f"Question: {text}\nAnswer:"
    noun = "text" if family == "sib200" else "Bible verse"
    field = "Text" if family == "sib200" else "Verse"
    labels = LABELS[family]
    prefix = f"Classify the topic of the following {noun} into one of these labels: {', '.join(labels)}.\n\n"
    for row in demonstrations:
        prefix += f"{field}: {row['text']}\nTopic: {labels[row['target']]}\n\n"
    return prefix + f"{field}: {text}\nTopic:"


def unigram_values(reference, tokenizer):
    if reference is None:
        return None, None
    import numpy as np
    from multilingual_megatron.pipeline.stages.final_loss.validation_metrics import UnigramReference
    data = read(checked(reference))
    if data["tokenizer_model_sha256"] != tokenizer["sha256"]:
        raise ValueError("unigram tokenizer mismatch")
    unigram = UnigramReference.from_counts(language=data["language"], vocab_size=tokenizer["vocab_size"],
        counts={int(k):v for k,v in data["counts_by_token_id"].items()}, special_token_ids=data["special_token_ids_excluded"])
    values = np.ones(tokenizer["vocab_size"], dtype=np.float64)
    for token, count in unigram.counts.items():
        values[token] = count
    values = np.log(values / unigram.smoothed_token_count)
    if unigram.counts_sha256 != reference["counts_sha256"]:
        raise ValueError("unigram no longer matches the paper's sealed loss table")
    return values, unigram.counts_sha256


def freeze_language(args):
    item, output, tokenizer_cfg, demonstrations, unigram_ref = args
    import sentencepiece as spm
    import numpy as np
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_scoring import encode_lm_eval_pair, prepare_scored_sequence
    from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import TokenCacheWriter, document_windows, token_window_sha256
    sp = spm.SentencePieceProcessor(model_file=tokenizer_cfg["model"])
    tokenizer = SimpleNamespace(tokenize=sp.encode)
    family, language = item["benchmark"], item["language"]
    task = f"stage1_extension_{family}_{language}"
    normalized = read(checked(item))
    unigram, unigram_sha = unigram_values(unigram_ref, tokenizer_cfg) if family in LOSS else (None, None)
    requests, sequences, docs = [], [], []
    for number, row in enumerate(normalized["documents"]):
        document = dict(doc_id=number, source_id=row["id"], source_document_sha256=canonical_json_sha256(row),
                        conditional=[], token_lengths=[], max_width=0)
        if family in LOSS:
            tokens = sp.encode(row["text"])
            if not tokens or any(t in (0,1,2,3) for t in tokens):
                raise ValueError(f"empty or special target in {task}/{row['id']}")
            document.update(target_tokens=len(tokens), utf8_bytes=len(row["text"].encode("utf-8")),
                sentence_ids=row.get("sentence_ids"), target_token_ids_sha256=token_window_sha256(tokens),
                unigram_log_probability_sum=float(np.sum(unigram[tokens], dtype=np.float64)) if unigram is not None else None)
            windows = document_windows(tokens, context_token_id=tokenizer_cfg["context_token_id"], max_length=2048,
                                       stride=1024 if family == "ntrex_mono" else 2048)
            for window_index, (context_ids, target_ids, window) in enumerate(windows):
                if family == "flores_mono" and window_index:
                    raise ValueError("FLORES sentence exceeds the frozen single-window contract")
                request = dict(task=task, benchmark=family, language=language, doc_id=number, window_index=window_index,
                    request_type="token_window_loglikelihood" if family == "ntrex_mono" else "loglikelihood",
                    context="", continuation=row["text"], target_utf8_bytes=document["utf8_bytes"])
                if family == "ntrex_mono":
                    request["token_window"] = dict(**window, document_token_ids_sha256=document["target_token_ids_sha256"],
                                                   window_token_ids_sha256=token_window_sha256(context_ids + target_ids))
                rid = canonical_json_sha256(request)
                request["request_id"] = rid
                sequence = prepare_scored_sequence(request_index=len(requests), request_id=rid, context_ids=context_ids,
                    continuation_ids=target_ids, max_length=2048)
                requests.append(request)
                sequences.append(sequence)
                document["conditional"].append(rid)
                document["token_lengths"].append(len(target_ids))
                document["max_width"] = max(document["max_width"], len(sequence.input_ids))
            if sum(document["token_lengths"]) != len(tokens):
                raise ValueError("document targets were lost or repeated")
        else:
            choices = row.get("choices", LABELS.get(family))
            context = classification_prompt(family, row["text"], demonstrations.get(family, []))
            document.update(target=row["target"], char_lengths=[len(c) for c in choices],
                            byte_lengths=[len(c.encode("utf-8")) for c in choices], classes=len(choices))
            for index, choice in enumerate(choices):
                request = dict(task=task, benchmark=family, language=language, doc_id=number, choice_index=index,
                               request_type="loglikelihood", context=context, continuation=" " + choice)
                rid = canonical_json_sha256(request)
                request["request_id"] = rid
                context_ids, target_ids = encode_lm_eval_pair(tokenizer, context, request["continuation"], context_token_id=tokenizer_cfg["context_token_id"])
                sequence = prepare_scored_sequence(request_index=len(requests), request_id=rid, context_ids=context_ids,
                    continuation_ids=target_ids, max_length=2048)
                if sequence.truncated_tokens:
                    raise ValueError(f"classification prompt would truncate: {task}/{row['id']}")
                requests.append(request)
                sequences.append(sequence)
                document["conditional"].append(rid)
                document["token_lengths"].append(len(target_ids))
                document["max_width"] = max(document["max_width"], len(sequence.input_ids))
        docs.append(document)
    smoke_ids = {d["doc_id"] for d in docs[:1] + docs[-1:] + sorted(docs, key=lambda d:d["max_width"])[-1:]}
    receipts = {}
    for scope in ("full", "smoke"):
        selected = docs if scope == "full" else [d for d in docs if d["doc_id"] in smoke_ids]
        ids = {r for d in selected for r in d["conditional"]}
        rows = [r for r in requests if r["request_id"] in ids]
        metadata = dict(task=task, benchmark=family, language=language, corpus_language=item["corpus_language"],
            output_type="frozen_document_loglikelihood" if family == "ntrex_mono" else "loglikelihood" if family in LOSS else "multiple_choice",
            metric="bits_per_byte" if family in LOSS else PRIMARY[family], documents=selected, source=item,
            primary_metric=PRIMARY[family], unigram_reference=unigram_ref if family in LOSS else None,
            unigram_counts_sha256=unigram_sha, demonstrations=demonstrations.get(family, []))
        payload = dict(schema="stage1_lm_eval_frozen_requests/v1", protocol="stage1_extension_likelihood_v1",
                       sampling=None if scope == "full" else "first/last/longest documents per variety", tasks=[metadata], requests=rows)
        request_ref = save(Path(output)/scope/"languages"/(task+".json"), payload, "manifest_sha256")
        writer = TokenCacheWriter(Path(output)/scope/"token_cache"/task)
        histogram = Counter()
        for sequence in sequences:
            if sequence.request_id in ids:
                writer.append(replace(sequence, request_index=sum(histogram.values())))
                histogram[len(sequence.input_ids)] += 1
        cache = writer.finish(request_sha256=request_ref["sha256"], tokenizer=tokenizer_cfg, sequence_length=2048)
        receipts[scope] = dict(**request_ref, benchmark=family, language=language, corpus_language=item["corpus_language"],
            documents=len(selected), requests=len(rows), token_cache=cache, width_histogram={str(k):v for k,v in sorted(histogram.items())})
    return receipts


def freeze(profile, repo, output, workers):
    import pyarrow.parquet as pq
    inventory = read(output/"source_inventory.json")
    _verify_payload_hash(inventory, field="manifest_sha256", label="extension sources")
    tokenizer = dict(profile["tokenizer"], model=str(repo/profile["tokenizer"]["model"]))
    if sha256_file(Path(tokenizer["model"])) != tokenizer["sha256"]:
        raise ValueError("tokenizer drift")
    # Match smoothed references to the original paper, not a newly built unigram.
    loss_path = repo/profile["validation_loss_table"]["path"]
    checked(dict(profile["validation_loss_table"], path=str(loss_path)))
    paper_hashes = defaultdict(set)
    for row in pq.read_table(loss_path, columns=["language", "unigram_counts_sha256"]).to_pylist():
        paper_hashes[row["language"]].add(row["unigram_counts_sha256"])
    corpus_path = repo/profile["corpus_manifest"]["path"]
    corpus = read(checked(dict(profile["corpus_manifest"], path=str(corpus_path))))
    unigrams = {}
    for item in inventory["sources"]:
        language = item["corpus_language"]
        if item["benchmark"] not in LOSS or not language or language in unigrams:
            continue
        if len(paper_hashes[language]) != 1:
            raise ValueError(f"paper's unigram is not unique: {language}")
        entry = corpus["languages"][language]["train"]["unigram_counts"]
        source = checked(dict(entry, path=str(corpus_path.parent/entry["path"])))
        dest = output/"unigrams"/(language+".json")
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        unigrams[language] = dict(**ref(dest), counts_sha256=next(iter(paper_hashes[language])))
    if (output/"full").exists() or (output/"smoke").exists():
        raise ValueError("refusing to overwrite frozen extension requests")
    receipts = []
    tasks = [(item, str(output), tokenizer, inventory["demonstrations"], unigrams.get(item["corpus_language"])) for item in inventory["sources"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for receipt in pool.map(freeze_language, tasks):
            receipts.append(receipt)
            if len(receipts)%50==0:
                print(f"Frozen {len(receipts)}/{len(tasks)} language inputs", flush=True)
    exclusions = read(checked(inventory["exclusions"]))["exclusions"]
    for scope in ("full", "smoke"):
        data = dict(schema="stage1_candidate_data/v1", protocol="stage1_extension_likelihood_v1", sampling=None if scope=="full" else "diagnostic_subset",
                    source_inventory=ref(output/"source_inventory.json"), tokenizer=tokenizer, sequence_length=2048,
                    languages=[r[scope] for r in receipts], families=[], exclusions=exclusions if scope=="full" else [],
                    exclusion_audit=inventory["exclusions"], unsupported_long_targets=[], unigram_references=unigrams)
        for family in FAMILIES:
            fragments = [read(checked(x)) for x in data["languages"] if x["benchmark"]==family]
            payload = dict(schema="stage1_lm_eval_frozen_requests/v1", protocol="stage1_extension_likelihood_v1", sampling=data["sampling"],
                           tasks=[t for f in fragments for t in f["tasks"]], requests=[r for f in fragments for r in f["requests"]])
            reference = save(output/scope/"families"/(family+".json"), payload, "manifest_sha256")
            data["families"].append(dict(**reference, benchmark=family, documents=sum(len(t["documents"]) for t in payload["tasks"]),
                                         requests=len(payload["requests"]), languages=len(fragments)))
        data["inputs"] = prepare_benchmark_inputs(data, output/scope)
        save(output/scope/"data_manifest.json", data, "manifest_sha256")
    return read(output/"full/data_manifest.json")["families"]


def classification_metrics(documents, by_id):
    observations, gold, predicted = [], [], []
    for doc in documents:
        scores = [by_id[r]["loglikelihood"] for r in doc["conditional"]]
        tokens = [by_id[r]["scored_tokens"] for r in doc["conditional"]]
        if tokens != doc["token_lengths"]:
            raise ValueError("classification scored-token drift")
        target = doc["target"]
        def prediction(values): return max(range(len(values)), key=values.__getitem__)
        pred = prediction(scores)
        record = dict(acc=float(pred==target), margin=scores[target]-max(x for i,x in enumerate(scores) if i!=target),
                      tie=float(scores.count(max(scores))>1), chance=1/len(scores))
        for name, lengths in (("char",doc["char_lengths"]),("token",tokens),("byte",doc["byte_lengths"])):
            values = [s/n for s,n in zip(scores,lengths,strict=True)]
            record["acc_"+name] = float(prediction(values)==target)
            record["margin_"+name] = values[target]-max(x for i,x in enumerate(values) if i!=target)
        observations.append(record)
        gold.append(target)
        predicted.append(pred)
    classes = max(d["classes"] for d in documents)
    support = Counter(gold)
    f1 = []
    for label in range(classes):
        tp = sum(g==p==label for g,p in zip(gold,predicted,strict=True))
        denominator = support[label] + predicted.count(label)
        f1.append(2*tp/denominator if denominator else 0.)
    return dict(**{k:math.fsum(o[k] for o in observations)/len(observations) for k in observations[0]},
                macro_f1=math.fsum(f1)/classes, weighted_f1=sum(f1[k]*support[k] for k in range(classes))/len(gold),
                majority_baseline=max(support.values())/len(gold), label_counts=json.dumps(dict(support),sort_keys=True))


def aggregate(manifest, scores):
    expected = {r["request_id"] for r in manifest["requests"]}
    by_id = {r["request_id"]:r for r in scores}
    if len(expected)!=len(manifest["requests"]) or len(by_id)!=len(scores) or set(by_id)!=expected:
        raise ValueError("extension score membership mismatch")
    if any(not math.isfinite(r["loglikelihood"]) or r["scored_tokens"]<1 or r.get("truncated_tokens",0) for r in scores):
        raise ValueError("nonfinite, empty or truncated extension score")
    rows = []
    for task in manifest["tasks"]:
        docs, family = task["documents"], task["benchmark"]
        if family in LOSS:
            for doc in docs:
                if [by_id[r]["scored_tokens"] for r in doc["conditional"]] != doc["token_lengths"] or sum(doc["token_lengths"])!=doc["target_tokens"]:
                    raise ValueError("document scored-token coverage changed")
            nll = -math.fsum(by_id[r]["loglikelihood"] for d in docs for r in d["conditional"])
            tokens, byte_count = sum(d["target_tokens"] for d in docs), sum(d["utf8_bytes"] for d in docs)
            offsets = [d["unigram_log_probability_sum"] for d in docs]
            adjusted = (nll+math.fsum(offsets))/tokens if all(x is not None for x in offsets) else None
            metrics = dict(nll_sum=nll, scored_tokens=tokens, utf8_bytes=byte_count, raw_nll=nll/tokens,
                bits_per_byte=nll/byte_count/math.log(2), token_perplexity=math.exp(nll/tokens) if nll/tokens<700 else None,
                byte_perplexity=math.exp(nll/byte_count) if nll/byte_count<700 else None,
                unigram_adjusted_loss=adjusted, unigram_counts_sha256=task["unigram_counts_sha256"],
                word_perplexity_status="not_reported_without_language_appropriate_segmentation", margin=None)
        else:
            metrics = classification_metrics(docs,by_id)
            if family == "arc_easy_eu20":
                for key in ("macro_f1", "weighted_f1", "majority_baseline", "label_counts"):
                    metrics.pop(key)
        rows.append(dict(benchmark=family, language=task["language"], corpus_language=task["corpus_language"], group="all",
                         samples=len(docs), primary_metric=task["primary_metric"], **metrics))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    profile=read(args.profile)
    repo=(args.profile.resolve().parent/profile["repo_root"]).resolve()
    print(json.dumps(freeze(profile,repo,args.output.resolve(),args.workers),indent=2))


if __name__=="__main__":
    main()
