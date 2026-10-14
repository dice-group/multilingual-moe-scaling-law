"""Freeze complete public sources for the second Stage 1 benchmark extension."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import zipfile

from multilingual_megatron.contracts.hashing import canonical_json_sha256, sha256_file
from .full_suite_campaign import checked, read, ref, save

FAMILIES = ("flores_mono", "ntrex_mono", "arc_easy_eu20", "sib200", "taxi1500")
LABELS = {
    "sib200": ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"],
    "taxi1500": ["Recommendation", "Faith", "Description", "Sin", "Grace", "Violence"],
}


def fetch(entry):
    """Resume only receipt-verified downloads; publish files atomically."""
    import requests
    path = Path(entry["path"])
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if receipt.exists():
        old = read(receipt)
        if any(old.get(k) != v for k, v in entry.items()):
            raise ValueError(f"source identity changed: {path}")
        checked(old)
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with requests.get(entry["url"], timeout=(30, 180), stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    temporary.replace(path)
    result = {**entry, **ref(path), "bytes": path.stat().st_size}
    save(receipt, result)
    return result


def download(profile, repo, output):
    pins = profile["sources"]
    entries = []
    for family in ("sib200", "arc_easy_eu20"):
        pin = pins[family]
        metadata = read(checked({"path": str(repo / pin["metadata"]), "sha256": pin["metadata_sha256"]}))
        if metadata["sha"] != pin["revision"]:
            raise ValueError("dataset revision mismatch")
        for item in metadata["siblings"]:
            name = item["rfilename"]
            selected = (name.startswith("data/") and name.endswith("/test.tsv")) if family == "sib200" else (name.startswith("arc_easy_") and name.endswith("_test.jsonl"))
            if family == "sib200" and name == "data/eng_Latn/train.tsv":
                selected = True
            if selected:
                entries.append(dict(benchmark=family, source_path=name,
                    url=f"https://huggingface.co/datasets/{pin['repo']}/resolve/{pin['revision']}/{name}",
                    path=str(output / "sources" / family / name)))
    pin = pins["ntrex_mono"]
    tree = read(checked({"path": str(repo / pin["metadata"]), "sha256": pin["metadata_sha256"]}))
    if tree["sha"] != pin["revision"] or tree.get("truncated"):
        raise ValueError("incomplete NTREX source tree")
    for item in tree["tree"]:
        name = item["path"]
        if (re.fullmatch(r"NTREX-128/newstest2019-ref\.[^.]+\.txt", name)
                or name == "NTREX-128/newstest2019-src.eng.txt"):
            entries.append(dict(benchmark="ntrex_mono", source_path=name,
                url=f"https://raw.githubusercontent.com/{pin['repo']}/{pin['revision']}/{name}",
                path=str(output / "sources/ntrex_mono" / Path(name).name)))
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for item in pool.map(fetch, entries):
            results.append(item)
            if len(results) % 40 == 0:
                print(f"Downloaded/verified {len(results)}/{len(entries)} public files", flush=True)
    local = {}
    for name, source in profile["local_sources"].items():
        src = Path(source["path"])
        if not src.is_absolute():
            src = repo / src
        if source.get("sha256") and sha256_file(src) != source["sha256"]:
            raise ValueError(f"local source drift: {name}")
        dest = output / "sources/local" / (name + src.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = dest.with_suffix(dest.suffix + ".partial")
            shutil.copyfile(src, temporary)
            temporary.replace(dest)
        if sha256_file(src) != sha256_file(dest):
            raise ValueError(f"local source copy mismatch: {name}")
        local[name] = {**ref(dest), "origin": str(src.resolve()), "url": source.get("url")}
    return save(output / "downloads.json", dict(sources=results, local=local), "manifest_sha256")


def tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def choose_demonstrations(rows):
    """Three label-diverse English training examples; no test-dependent choice."""
    chosen, labels = [], set()
    for row in sorted(rows, key=lambda r: hashlib.sha256(("stage1-extension-1234:" + r["id"]).encode()).hexdigest()):
        if row["target"] not in labels and row["text"].strip():
            chosen.append(row)
            labels.add(row["target"])
        if len(chosen) == 3:
            return chosen
    raise ValueError("fewer than three valid training labels for demonstrations")


def dominant_script(text):
    """Audit actual letters, rather than trusting an ISO-only archive filename."""
    import regex
    names = ("Latn", "Cyrl", "Arab", "Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym", "Sinh", "Thai", "Laoo", "Tibt", "Mymr", "Geor", "Armn", "Hebr", "Ethi", "Khmr", "Hang", "Hani", "Hira", "Kana", "Grek", "Cans", "Cher", "Thaa", "Olck", "Tfng")
    counts = Counter()
    for char, count in Counter(text).items():
        if not char.isalpha():
            continue
        for name in names:
            if regex.fullmatch(r"\p{sc=" + name + "}", char):
                counts[name] += count
                break
        else:
            counts["Zzzz"] += count
    if counts["Hira"] + counts["Kana"]:
        counts["Jpan"] = counts.pop("Hira", 0) + counts.pop("Kana", 0) + counts.pop("Hani", 0)
    if not counts:
        return "Zzzz", {}
    return counts.most_common(1)[0][0], dict(counts)


def normalize(profile, repo, output):
    downloads = read(output / "downloads.json")
    local = {k: checked(v) for k, v in downloads["local"].items()}
    mapping = list(csv.DictReader(local["candidate_mapping"].open()))
    mapped = {(r["benchmark"], r["dataset_code"]): r["corpus_language"] for r in mapping}
    training = read(local["training_languages"])
    study = set(training["200"])
    arc_map = {r["dataset_code"]: r["corpus_language"] for r in csv.DictReader(local["extension_mapping"].open()) if r["benchmark"] == "arc_easy_eu20"}
    ntrex_map = {r["path"]: r["corpus_language"] for r in csv.DictReader(local["ntrex_mapping"].open())}
    sources, exclusions, edition_audit = [], [], []
    demo_sets = {}

    def emit(family, language, corpus_language, documents, origin, **extra):
        if not documents:
            exclusions.append(dict(benchmark=family, language=language, source_id="ALL", reason="no available labeled test verses"))
            return
        if len({d["id"] for d in documents}) != len(documents):
            raise ValueError(f"duplicate source IDs: {family}/{language}")
        path = output / "normalized" / family / (language + ".json")
        save(path, dict(documents=documents, **extra))
        sources.append(dict(**ref(path), benchmark=family, language=language,
                            corpus_language=corpus_language, source=origin, documents=len(documents)))

    def exclude(family, language, row, reason):
        exclusions.append(dict(benchmark=family, language=language, source_id=row["id"], reason=reason,
                               document_sha256=canonical_json_sha256(row)))

    with tarfile.open(local["flores_archive"]) as archive:
        for member in archive:
            p = Path(member.name)
            if member.isfile() and p.parent.name == "devtest" and p.suffix == ".devtest":
                texts = archive.extractfile(member).read().decode("utf-8").splitlines()
                if len(texts) != 1012 or any(not t.strip() for t in texts):
                    raise ValueError(f"invalid FLORES source: {p}")
                emit("flores_mono", p.stem, mapped.get(("flores200_external_loss", p.stem), ""),
                     [dict(id=str(i), text=text) for i, text in enumerate(texts)], downloads["local"]["flores_archive"])

    doc_ids = local["ntrex_document_ids"].read_text().splitlines()
    if len(doc_ids) != 1997 or len(set(doc_ids)) != 123:
        raise ValueError("NTREX document inventory mismatch")
    for item in downloads["sources"]:
        family, path = item["benchmark"], checked(item)
        if family == "ntrex_mono":
            language = Path(item["source_path"]).stem.split(".", 1)[1]
            texts = path.read_text().splitlines()
            if len(texts) != len(doc_ids) or any(not t.strip() for t in texts):
                raise ValueError(f"invalid NTREX source: {path}")
            groups = defaultdict(list)
            for i, (doc_id, text) in enumerate(zip(doc_ids, texts, strict=True)):
                groups[doc_id].append((i, text))
            docs = [dict(id=k, text="\n".join(t for _, t in v), sentence_ids=[i for i, _ in v]) for k, v in groups.items()]
            emit(family, language, "eng_Latn" if language == "eng" else ntrex_map.get(item["source_path"], ""), docs, item)
        elif family == "arc_easy_eu20":
            language = path.name[len("arc_easy_"):-len("_test.jsonl")]
            docs = []
            for line in path.read_text().splitlines():
                row = json.loads(line)
                choices, labels = row["choices"]["text"], row["choices"]["label"]
                if len(choices) < 2 or len(choices) != len(labels) or len(set(labels)) != len(labels) or row["answerKey"] not in labels:
                    raise ValueError(f"invalid ARC label structure: {language}/{row['id']}")
                if any(not isinstance(c, str) or not c.strip() for c in choices):
                    exclude(family, language, row, "empty answer option")
                    continue
                docs.append(dict(id=row["id"], text=row["question"], choices=choices, target=labels.index(row["answerKey"])))
            emit(family, language, arc_map[language], docs, item)
        elif family == "sib200":
            language = path.parent.name
            rows = tsv(path)
            docs = [dict(id=str(r.get("index_id", r.get("id", i))), text=r["text"], target=LABELS[family].index(r["category"])) for i, r in enumerate(rows)]
            if path.name == "train.tsv":
                demo_sets[family] = choose_demonstrations(docs)
                continue
            valid = []
            for row in docs:
                if not row["text"].strip():
                    exclude(family, language, row, "empty text")
                else:
                    valid.append(row)
            emit(family, language, mapped.get((family, language), ""), valid, item)

    splits = {}
    for split in ("train", "dev", "test"):
        with local["taxi_" + split].open() as handle:
            splits[split] = [dict(id=r[0], target=LABELS["taxi1500"].index(r[1]), text=r[2]) for r in csv.reader(handle, delimiter="\t") if len(r) >= 3]
    sets = {k: {r["id"] for r in rows} for k, rows in splits.items()}
    if tuple(len(splits[k]) for k in ("train", "dev", "test")) != (860, 106, 111) or any(sets[a] & sets[b] for a,b in (("train","dev"),("train","test"),("dev","test"))):
        raise ValueError("official Taxi split identity changed")
    demo_sets["taxi1500"] = choose_demonstrations(splits["train"])
    union = set.union(*sets.values())
    best = {}
    with zipfile.ZipFile(local["taxi_archive"]) as archive:
        files = [x for x in archive.infolist() if x.filename.endswith(".txt")]
        for index, member in enumerate(files):
            iso_match = re.match(r"([a-z]{3})[-_]", Path(member.filename).name)
            if not iso_match:
                raise ValueError(f"unknown Taxi member name: {member.filename}")
            iso = iso_match[1]
            raw = archive.read(member)  # zipfile verifies CRC; strict UTF-8 below.
            verses, all_ids = {}, set()
            for line in raw.decode("utf-8").splitlines():
                if not line or line.startswith("#") or "\t" not in line:
                    continue
                verse_id, text = line.split("\t", 1)
                verse_id, text = verse_id.strip(), text.strip()
                all_ids.add(verse_id)
                if verse_id in union:
                    if verse_id in verses and verses[verse_id] != text:
                        raise ValueError(f"conflicting Taxi verse ID: {member.filename}/{verse_id}")
                    verses[verse_id] = text
            script, script_counts = dominant_script(" ".join(verses.values()))
            language = iso + "_" + script
            score = (sum(bool(v) for v in verses.values()), len(all_ids), member.filename)
            audit = dict(member=member.filename, language=language, labeled_overlap=score[0], total_verses=len(all_ids),
                         sha256=hashlib.sha256(raw).hexdigest(), script_counts=script_counts)
            edition_audit.append(audit)
            if language not in best or score > best[language][0]:
                best[language] = (score, verses, audit)
            if (index + 1) % 200 == 0:
                print(f"Audited Taxi archive members {index + 1}/{len(files)}", flush=True)
    for language, (_, verses, audit) in sorted(best.items()):
        docs = []
        for row in splits["test"]:
            if not verses.get(row["id"], "").strip():
                exclude("taxi1500", language, row, "missing or empty released test verse")
                continue
            docs.append({**row, "text": verses[row["id"]]})
        # Exact ISO/script matches only; do not equate macro-languages/dialects.
        emit("taxi1500", language, language if language in study else "", docs,
             dict(archive=downloads["local"]["taxi_archive"], selected_edition=audit), expected_test_verses=111)
    if set(demo_sets) != set(LABELS):
        raise ValueError("missing frozen training-only demonstrations")
    for item in sources:
        if item["benchmark"] in LABELS:
            demo_ids = {d["id"] for d in demo_sets[item["benchmark"]]}
            if any(d["id"] in demo_ids for d in read(checked(item))["documents"]):
                raise ValueError("demonstration and test IDs overlap")
    save(output / "taxi_edition_audit.json", dict(editions=edition_audit, selected_languages=len(best)), "manifest_sha256")
    save(output / "exclusions.json", dict(policy="Only unavailable/empty text or empty options; no performance-based filtering", exclusions=exclusions), "manifest_sha256")
    payload = dict(schema="stage1_extension_sources/v1", downloads=ref(output / "downloads.json"), sources=sources,
                   demonstrations=demo_sets, exclusions=ref(output / "exclusions.json"),
                   taxi_edition_audit=ref(output / "taxi_edition_audit.json"), training=downloads["local"]["training_languages"])
    save(output / "source_inventory.json", payload, "manifest_sha256")
    return dict(languages=dict(Counter(x["benchmark"] for x in sources)),
                documents={f:sum(x["documents"] for x in sources if x["benchmark"]==f) for f in FAMILIES}, exclusions=len(exclusions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "normalize"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = read(args.profile)
    repo = (args.profile.resolve().parent / profile["repo_root"]).resolve()
    output = args.output.resolve()
    result = download(profile, repo, output) if args.command == "download" else normalize(profile, repo, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
