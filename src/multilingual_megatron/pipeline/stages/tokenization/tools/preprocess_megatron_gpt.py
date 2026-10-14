#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocess multilingual JSONL shards into Megatron GPT bin/idx prefixes.

This is the bridge between the copied multilingual sampling pipeline and
Megatron Bridge's GPTDatasetConfig contract. It keeps one data prefix per
language so the training stage can build explicit per-language blends.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _extract_language(path: Path, shard_root: Path, obj: object | None) -> str:
    if isinstance(obj, dict):
        for key in ("language", "lang", "subset"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    rel = path.relative_to(shard_root)
    if len(rel.parts) > 1:
        # FineWeb-style samples are stored as samples/<lang>/00000.jsonl.gz.
        # Use the parent directory as the language, not the container name.
        return rel.parts[-2]
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _iter_jsonl(path: Path, shard_root: Path) -> Iterable[tuple[str, str]]:
    with _open_text(path) as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            obj: object | None = None
            text = raw
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                value = obj.get("text")
                if not isinstance(value, str) or not value.strip():
                    continue
                text = value
            language = _extract_language(path, shard_root, obj)
            yield language, text


def _discover_jsonl(shard_root: Path) -> list[Path]:
    paths = sorted(shard_root.rglob("*.jsonl"))
    paths.extend(sorted(shard_root.rglob("*.jsonl.gz")))
    return paths


def _discover_language_sources(shard_root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for source in _discover_jsonl(shard_root):
        rel = source.relative_to(shard_root)
        if len(rel.parts) > 2 and rel.parts[0] == "samples":
            language = rel.parts[1]
        elif len(rel.parts) > 1:
            language = rel.parts[-2]
        else:
            language = _extract_language(source, shard_root, None)
        grouped[language.replace("/", "_")].append(source)
    return {language: sorted(paths) for language, paths in sorted(grouped.items())}


def _read_planned_doc_counts(shard_root: Path) -> dict[str, int]:
    counts_path = shard_root / "sample_sizes.tsv"
    if not counts_path.exists():
        return {}
    counts: dict[str, int] = {}
    with counts_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        try:
            subset_idx = header.index("subset")
            size_idx = header.index("sample_size")
        except ValueError:
            return {}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(subset_idx, size_idx):
                continue
            try:
                counts[parts[subset_idx]] = int(parts[size_idx])
            except ValueError:
                continue
    return counts


def _count_jsonl(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with _open_text(path) as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _weights(languages: list[str], docs: dict[str, int], policy: str) -> list[float] | None:
    if policy == "none":
        return None
    if policy == "equal":
        return [1.0 for _ in languages]
    if policy == "proportional_documents":
        return [float(max(1, docs[language])) for language in languages]
    raise ValueError(f"Unsupported blend weight policy: {policy}")


def _weights_for_prefixes(
    languages: list[str],
    docs: dict[str, int],
    policy: str,
    prefixes_by_language: dict[str, list[str]],
) -> list[float] | None:
    if policy == "none":
        return None
    weights: list[float] = []
    for language in languages:
        partitions = max(1, len(prefixes_by_language[language]))
        if policy == "equal":
            language_weight = 1.0
        elif policy == "proportional_documents":
            language_weight = float(max(1, docs[language]))
        else:
            raise ValueError(f"Unsupported blend weight policy: {policy}")
        weights.extend([language_weight / float(partitions)] * partitions)
    return weights


def _prepare_language_input(language: str, sources: list[Path], language_work_dir: Path) -> Path:
    language_work_dir.mkdir(parents=True, exist_ok=True)
    target = language_work_dir / ("input.jsonl.gz" if len(sources) == 1 and sources[0].suffix == ".gz" else "input.jsonl")
    if target.exists() or target.is_symlink():
        target.unlink()
    if len(sources) == 1:
        try:
            target.symlink_to(sources[0])
            return target
        except OSError:
            pass
    with target.open("w", encoding="utf-8") as out:
        for source in sources:
            with _open_text(source) as handle:
                shutil.copyfileobj(handle, out)
    return target


def _run_language_preprocess(
    *,
    language: str,
    sources: list[Path],
    work_dir: Path,
    output_dir: Path,
    preprocess: Path,
    bridge_root: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    trust_remote_code: bool,
    append_eod: bool,
    workers: int,
    partitions: int,
    log_interval: int,
    launcher: str,
    python_executable: str,
) -> tuple[str, list[str]]:
    language_work_dir = work_dir / language
    if language_work_dir.exists():
        shutil.rmtree(language_work_dir)
    input_path = _prepare_language_input(language, sources, language_work_dir)
    output_prefix = output_dir / language
    final_prefixes = [
        f"{output_prefix}_{idx}_text_document" if partitions > 1 else f"{output_prefix}_text_document"
        for idx in range(partitions)
    ]
    if all(Path(prefix + ".bin").exists() and Path(prefix + ".idx").exists() for prefix in final_prefixes):
        return language, final_prefixes

    if launcher == "uv":
        cmd = ["uv", "run", "--no-sync", "python", str(preprocess)]
    else:
        cmd = [python_executable, str(preprocess)]
    cmd.extend(
        [
            "--input",
            str(input_path),
            "--output-prefix",
            str(output_prefix),
            "--tokenizer-type",
            tokenizer_type,
            "--tokenizer-model",
            tokenizer_model,
            "--workers",
            str(workers),
            "--partitions",
            str(partitions),
            "--log-interval",
            str(log_interval),
        ]
    )
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if append_eod:
        cmd.append("--append-eod")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{bridge_root / '3rdparty' / 'Megatron-LM'}:"
        f"{bridge_root / 'src'}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    subprocess.run(cmd, check=True, cwd=str(bridge_root), env=env)
    return language, final_prefixes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-type", default="HuggingFaceTokenizer")
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--language-jobs", type=int, default=0)
    parser.add_argument("--workers-per-language", type=int, default=0)
    parser.add_argument("--partitions-per-language", type=int, default=0)
    parser.add_argument("--append-eod", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--launcher", choices=["python", "uv"], default="python")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--blend-weight-policy",
        choices=["equal", "proportional_documents", "none"],
        default="equal",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    bridge_root = Path(args.bridge_root).resolve()
    shard_root = Path(args.shard_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    preprocess = bridge_root / "3rdparty" / "Megatron-LM" / "tools" / "preprocess_data.py"
    if not preprocess.exists():
        raise SystemExit(f"Missing Megatron preprocess script: {preprocess}")
    if not shard_root.exists():
        raise SystemExit(f"Missing shard directory: {shard_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "megatron_data_manifest.json"
    if args.force and output_dir.exists():
        for child in output_dir.glob("*"):
            if child.is_file() and child.name != manifest_path.name:
                child.unlink()
            elif child.is_dir() and child.name == "_work":
                shutil.rmtree(child)

    work_dir = output_dir / "_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    language_sources = _discover_language_sources(shard_root)
    if not language_sources:
        raise SystemExit(f"No JSONL documents found under {shard_root}")

    total_workers = max(1, int(args.workers))
    workers_per_language = int(args.workers_per_language or 0)
    if workers_per_language <= 0:
        workers_per_language = 8 if total_workers >= 16 else max(1, min(4, total_workers))
    workers_per_language = max(1, min(total_workers, workers_per_language))
    language_jobs = int(args.language_jobs or 0)
    if language_jobs <= 0:
        language_jobs = max(1, total_workers // workers_per_language)
    partitions = int(args.partitions_per_language or 0)
    if partitions <= 0:
        partitions = min(4, workers_per_language)
    while workers_per_language % partitions != 0 and partitions > 1:
        partitions -= 1
    partitions = max(1, partitions)

    planned_docs = _read_planned_doc_counts(shard_root)
    docs: dict[str, int] = {}
    for language, sources in language_sources.items():
        docs[language] = planned_docs.get(language) or _count_jsonl(sources)

    print(
        "[INFO] preprocessing "
        f"{len(language_sources)} languages with language_jobs={language_jobs} "
        f"workers_per_language={workers_per_language} partitions_per_language={partitions}",
        flush=True,
    )
    prefixes_by_language: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=language_jobs) as pool:
        futures = [
            pool.submit(
                _run_language_preprocess,
                language=language,
                sources=sources,
                work_dir=work_dir,
                output_dir=output_dir,
                preprocess=preprocess,
                bridge_root=bridge_root,
                tokenizer_type=str(args.tokenizer_type),
                tokenizer_model=str(args.tokenizer_model),
                trust_remote_code=bool(args.trust_remote_code),
                append_eod=bool(args.append_eod),
                workers=workers_per_language,
                partitions=partitions,
                log_interval=int(args.log_interval),
                launcher=str(args.launcher),
                python_executable=str(args.python_executable),
            )
            for language, sources in language_sources.items()
        ]
        for future in as_completed(futures):
            language, final_prefixes = future.result()
            prefixes_by_language[language] = final_prefixes
            print(f"[OK] tokenized {language} prefixes={len(final_prefixes)}", flush=True)

    languages = sorted(language_sources)
    prefix_by_language = {language: prefixes_by_language[language] for language in languages}
    prefixes = [prefix for language in languages for prefix in prefixes_by_language[language]]
    weights = _weights_for_prefixes(languages, docs, str(args.blend_weight_policy), prefixes_by_language)
    manifest = {
        "format": "megatron_gpt",
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_model": args.tokenizer_model,
        "trust_remote_code": bool(args.trust_remote_code),
        "blend_weight_policy": args.blend_weight_policy,
        "languages": languages,
        "prefixes": prefixes,
        "weights": weights,
        "prefixes_by_language": prefix_by_language,
        "documents_per_language": docs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dataset_info.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "state.json").write_text(json.dumps({"stage": "tokenize", "pass": True}, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
