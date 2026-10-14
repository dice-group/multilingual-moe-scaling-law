from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _extract_text(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(payload, dict):
        return stripped
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def _discover_sources(shard_dir: Path, languages: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(shard_dir.rglob("*.jsonl")) + sorted(shard_dir.rglob("*.jsonl.gz"))
    for source in paths:
        rel = source.relative_to(shard_dir)
        if len(rel.parts) > 2 and rel.parts[0] == "samples":
            language = rel.parts[1]
        elif len(rel.parts) > 1:
            language = rel.parts[-2]
        else:
            name = source.name
            for suffix in (".jsonl.gz", ".jsonl"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            language = name
        if languages is not None and language not in languages:
            continue
        doc_count = 0
        with _open_text(source) as handle:
            for line in handle:
                if _extract_text(line) is not None:
                    doc_count += 1
        if doc_count <= 0:
            continue
        rows.append(
            {
                "language": language,
                "source": str(source.resolve()),
                "source_rel": str(rel),
                "doc_count": doc_count,
            }
        )
    if not rows:
        wanted = sorted(languages) if languages else "all languages"
        raise SystemExit(f"No tokenization sources found under {shard_dir} for {wanted}")
    return rows


def _assign_rows(rows: list[dict[str, Any]], ranks: int) -> list[list[dict[str, Any]]]:
    rank_rows: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
    rank_counts = [0 for _ in range(ranks)]
    for row in sorted(rows, key=lambda item: (-int(item["doc_count"]), str(item["language"]), str(item["source_rel"]))):
        rank = min(range(ranks), key=lambda idx: (rank_counts[idx], idx))
        rank_rows[rank].append(row)
        rank_counts[rank] += int(row["doc_count"])
    return rank_rows


def _hash_score(*parts: object) -> float:
    key = "\0".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_plan_documents(
    rows: list[dict[str, Any]],
    *,
    sample_fraction: float,
    max_samples: int,
    train_fraction: float,
    split_seed: int,
) -> Iterable[tuple[str, str, str]]:
    emitted = 0
    for row in rows:
        language = str(row["language"])
        source = Path(str(row["source"]))
        source_rel = str(row.get("source_rel", source.name))
        with _open_text(source) as handle:
            for line_idx, line in enumerate(handle):
                text = _extract_text(line)
                if text is None:
                    continue
                if sample_fraction < 1.0 and _hash_score("sample", language, source_rel, line_idx) > sample_fraction:
                    continue
                if max_samples > 0 and emitted >= max_samples:
                    return
                split_score = _hash_score("split", split_seed, language, source_rel, line_idx)
                split = "train" if split_score < train_fraction else "validation"
                yield split, language, text
                emitted += 1


def _write_rank_inputs(
    *,
    rows: list[dict[str, Any]],
    work_dir: Path,
    sample_fraction: float,
    max_samples: int,
    train_fraction: float,
    split_seed: int,
) -> dict[str, dict[str, int]]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    counters: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
    }
    handles: dict[tuple[str, str], Any] = {}
    try:
        for split, language, text in _iter_plan_documents(
            rows,
            sample_fraction=sample_fraction,
            max_samples=max_samples,
            train_fraction=train_fraction,
            split_seed=split_seed,
        ):
            path = work_dir / split / language / "input.jsonl"
            key = (split, language)
            if key not in handles:
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[key] = path.open("w", encoding="utf-8")
            handles[key].write(json.dumps({"text": text, "language": language}, ensure_ascii=False) + "\n")
            counters[split][language] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {split: dict(values) for split, values in counters.items()}


def _run_preprocess(
    *,
    repo_root: Path,
    bridge_root: Path,
    input_dir: Path,
    output_dir: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    trust_remote_code: bool,
    append_eod: bool,
    workers: int,
    workers_per_language: int,
    partitions_per_language: int,
    language_jobs: int,
    blend_weight_policy: str,
    launcher: str,
    python_executable: str,
) -> dict[str, Any] | None:
    if not input_dir.exists() or not any(input_dir.rglob("*.jsonl")):
        return None
    cmd = [
        python_executable,
        "-m",
        "multilingual_megatron.pipeline.stages.tokenization.tools.preprocess_megatron_gpt",
        "--bridge-root",
        str(bridge_root),
        "--shard-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--tokenizer-type",
        tokenizer_type,
        "--tokenizer-model",
        tokenizer_model,
        "--workers",
        str(workers),
        "--log-interval",
        "1000",
        "--blend-weight-policy",
        blend_weight_policy,
        "--launcher",
        launcher,
        "--python-executable",
        python_executable,
        "--force",
    ]
    if workers_per_language > 0:
        cmd.extend(["--workers-per-language", str(workers_per_language)])
    if partitions_per_language > 0:
        cmd.extend(["--partitions-per-language", str(partitions_per_language)])
    if language_jobs > 0:
        cmd.extend(["--language-jobs", str(language_jobs)])
    if append_eod:
        cmd.append("--append-eod")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    manifest_path = output_dir / "megatron_data_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing worker Megatron manifest: {manifest_path}")
    return _read_json(manifest_path)


def _prefix_language_pairs(manifest: dict[str, Any]) -> Iterable[tuple[str, str]]:
    by_language = manifest.get("prefixes_by_language", {})
    if isinstance(by_language, dict):
        for language, prefixes in by_language.items():
            if isinstance(prefixes, list):
                for prefix in prefixes:
                    yield str(language), str(prefix)
        return
    for prefix in manifest.get("prefixes", []):
        yield "", str(prefix)


def _aggregate_weights(prefix_languages: list[tuple[str, str]], docs: dict[str, int], policy: str) -> list[float] | None:
    if policy == "none":
        return None
    counts = defaultdict(int)
    for language, _prefix in prefix_languages:
        counts[language] += 1
    weights: list[float] = []
    for language, _prefix in prefix_languages:
        divisor = max(1, counts[language])
        if policy == "proportional_documents":
            value = float(max(1, docs.get(language, 1))) / float(divisor)
        else:
            value = 1.0 / float(divisor)
        weights.append(value)
    return weights


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prefixes_by_language(prefix_languages: list[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for language, prefix in prefix_languages:
        if language:
            grouped[language].append(prefix)
    return {language: sorted(prefixes) for language, prefixes in sorted(grouped.items())}


def _prefix_artifacts(prefix_languages: list[tuple[str, str]]) -> dict[str, list[dict[str, Any]]]:
    """Freeze validation bin/idx identities once at tokenization merge time."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for language, raw_prefix in prefix_languages:
        if not language:
            continue
        prefix = Path(raw_prefix)
        bin_path = Path(str(prefix) + ".bin")
        idx_path = Path(str(prefix) + ".idx")
        if not bin_path.is_file() or not idx_path.is_file():
            raise SystemExit(f"Missing Megatron prefix files for {prefix}")
        grouped[language].append(
            {
                "prefix": str(prefix),
                "bin_sha256": _sha256_file(bin_path),
                "bin_size_bytes": bin_path.stat().st_size,
                "idx_sha256": _sha256_file(idx_path),
                "idx_size_bytes": idx_path.stat().st_size,
            }
        )
    return {
        language: sorted(rows, key=lambda row: row["prefix"])
        for language, rows in sorted(grouped.items())
    }


def _merge_manifests(
    *,
    result_paths: list[Path],
    output_dir: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    trust_remote_code: bool,
    blend_weight_policy: str,
) -> dict[str, Any]:
    split_prefix_languages: dict[str, list[tuple[str, str]]] = {"train": [], "validation": []}
    split_docs: dict[str, dict[str, int]] = {"train": defaultdict(int), "validation": defaultdict(int)}
    rank_results: list[dict[str, Any]] = []

    for result_path in result_paths:
        result = _read_json(result_path)
        rank_results.append(result)
        for split in ("train", "validation"):
            manifest = result.get("manifests", {}).get(split)
            if not isinstance(manifest, dict):
                continue
            split_prefix_languages[split].extend(_prefix_language_pairs(manifest))
            for language, count in manifest.get("documents_per_language", {}).items():
                split_docs[split][str(language)] += int(count)

    train_pairs = split_prefix_languages["train"]
    validation_pairs = split_prefix_languages["validation"]
    all_languages = sorted({language for language, _prefix in train_pairs + validation_pairs if language})
    train_docs = dict(split_docs["train"])
    validation_docs = dict(split_docs["validation"])
    total_docs = {language: int(train_docs.get(language, 0)) + int(validation_docs.get(language, 0)) for language in all_languages}

    train_prefixes_by_language = _prefixes_by_language(train_pairs)
    validation_prefixes_by_language = _prefixes_by_language(validation_pairs)

    manifest = {
        "format": "megatron_gpt",
        "schema_version": 2,
        "stage": "tokenize",
        "tokenizer_type": tokenizer_type,
        "tokenizer_model": tokenizer_model,
        "trust_remote_code": trust_remote_code,
        "blend_weight_policy": blend_weight_policy,
        "languages": all_languages,
        "prefixes": [prefix for _language, prefix in train_pairs],
        "weights": _aggregate_weights(train_pairs, train_docs, blend_weight_policy),
        "prefixes_by_language": train_prefixes_by_language,
        "documents_per_language": total_docs,
        "splits": {
            "train": {
                "prefixes": [prefix for _language, prefix in train_pairs],
                "weights": _aggregate_weights(train_pairs, train_docs, blend_weight_policy),
                "prefixes_by_language": train_prefixes_by_language,
                "documents_per_language": train_docs,
            },
            "validation": {
                "prefixes": [prefix for _language, prefix in validation_pairs],
                "weights": _aggregate_weights(validation_pairs, validation_docs, blend_weight_policy),
                "prefixes_by_language": validation_prefixes_by_language,
                "prefix_artifacts_by_language": _prefix_artifacts(validation_pairs),
                "documents_per_language": validation_docs,
            },
        },
        "rank_results": [str(path.resolve()) for path in result_paths],
    }
    _write_json(output_dir / "megatron_data_manifest.json", manifest)
    _write_json(output_dir / "dataset_info.json", manifest)
    _write_json(output_dir / "state.json", {"stage": "tokenize", "pass": True, "rank_count": len(result_paths)})
    return manifest


def _check_prefix_files(manifest: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    prefixes: set[str] = set(str(prefix) for prefix in manifest.get("prefixes", []))
    for split in manifest.get("splits", {}).values():
        if isinstance(split, dict):
            prefixes.update(str(prefix) for prefix in split.get("prefixes", []))
    for prefix in sorted(prefixes):
        if not Path(prefix + ".bin").exists():
            missing.append(prefix + ".bin")
        if not Path(prefix + ".idx").exists():
            missing.append(prefix + ".idx")
    return missing


def _cmd_plan(args: argparse.Namespace) -> int:
    shard_dir = Path(args.shard_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    plan_dir = Path(args.plan_dir).resolve()
    languages = {item.strip() for item in str(args.languages or "").split(",") if item.strip()} or None
    rows = _discover_sources(shard_dir, languages)
    rank_rows = _assign_rows(rows, int(args.array_ranks))
    plan_dir.mkdir(parents=True, exist_ok=True)
    for stale in plan_dir.glob("rank_*.jsonl"):
        stale.unlink()
    rank_counts: dict[str, int] = {}
    for rank, assigned in enumerate(rank_rows):
        rank_counts[f"rank_{rank:05d}"] = sum(int(row["doc_count"]) for row in assigned)
        with (plan_dir / f"rank_{rank:05d}.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(assigned, key=lambda item: (str(item["language"]), str(item["source_rel"]))):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "stage": "tokenize",
        "backend": "megatron_gpt",
        "shard_dir": str(shard_dir),
        "output_dir": str(output_dir),
        "plan_dir": str(plan_dir),
        "array_ranks": int(args.array_ranks),
        "languages": sorted({str(row["language"]) for row in rows}),
        "source_count": len(rows),
        "document_count": sum(int(row["doc_count"]) for row in rows),
        "rank_counts": rank_counts,
        "sample_fraction": float(args.sample_fraction),
        "train_fraction": float(args.train_fraction),
        "validation_fraction": float(args.validation_fraction),
    }
    _write_json(output_dir / "tokenization_plan.json", summary)
    _write_json(plan_dir / "summary.json", summary)
    print(f"[OK] planned Megatron tokenization ranks={args.array_ranks} sources={len(rows)} output={plan_dir}", flush=True)
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    rank = int(args.rank if args.rank is not None else os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    plan_path = Path(args.plan_dir).resolve() / f"rank_{rank:05d}.jsonl"
    if not plan_path.exists():
        raise SystemExit(f"Missing rank plan: {plan_path}")
    rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rank_root = Path(args.ranks_dir).resolve() / f"rank_{rank:05d}"
    work_dir = rank_root / "_work_inputs"
    output_root = rank_root / "megatron"
    if rank_root.exists() and bool(args.force):
        shutil.rmtree(rank_root)
    rank_root.mkdir(parents=True, exist_ok=True)
    counters = _write_rank_inputs(
        rows=rows,
        work_dir=work_dir,
        sample_fraction=float(args.sample_fraction),
        max_samples=int(args.max_samples),
        train_fraction=float(args.train_fraction),
        split_seed=int(args.split_seed),
    )
    manifests: dict[str, Any] = {}
    for split in ("train", "validation"):
        manifest = _run_preprocess(
            repo_root=Path(args.repo_root).resolve(),
            bridge_root=Path(args.bridge_root).resolve(),
            input_dir=work_dir / split,
            output_dir=output_root / split,
            tokenizer_type=str(args.tokenizer_type),
            tokenizer_model=str(args.tokenizer_model),
            trust_remote_code=bool(args.trust_remote_code),
            append_eod=bool(args.append_eod),
            workers=int(args.workers),
            workers_per_language=int(args.workers_per_language),
            partitions_per_language=int(args.partitions_per_language),
            language_jobs=int(args.language_jobs),
            blend_weight_policy=str(args.blend_weight_policy),
            launcher=str(args.launcher),
            python_executable=str(args.python_executable),
        )
        if manifest is not None:
            manifests[split] = manifest
    result = {
        "stage": "tokenize",
        "rank": rank,
        "plan": str(plan_path),
        "doc_counts": counters,
        "manifests": manifests,
    }
    result_path = rank_root / "result.json"
    _write_json(result_path, result)
    if not bool(args.keep_work_dir) and work_dir.exists():
        shutil.rmtree(work_dir)
    print(f"[OK] rank={rank} result={result_path}", flush=True)
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    ranks_dir = Path(args.ranks_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_paths = sorted(ranks_dir.glob("rank_*/result.json"))
    if not result_paths:
        raise SystemExit(f"No rank results found under {ranks_dir}")
    manifest = _merge_manifests(
        result_paths=result_paths,
        output_dir=output_dir,
        tokenizer_type=str(args.tokenizer_type),
        tokenizer_model=str(args.tokenizer_model),
        trust_remote_code=bool(args.trust_remote_code),
        blend_weight_policy=str(args.blend_weight_policy),
    )
    missing = _check_prefix_files(manifest) if bool(args.check_prefix_files) else []
    report = {
        "stage": "tokenize",
        "pass": not missing,
        "result_count": len(result_paths),
        "languages": manifest.get("languages", []),
        "num_prefixes": len(manifest.get("prefixes", [])),
        "num_validation_prefixes": len(manifest.get("splits", {}).get("validation", {}).get("prefixes", [])),
        "missing_files": missing,
    }
    _write_json(output_dir / "tokenization_report.json", report)
    if missing:
        raise SystemExit(f"Missing Megatron prefix files: {missing[:5]}")
    print(f"[OK] merged Megatron tokenization manifest={output_dir / 'megatron_data_manifest.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Distributed Megatron GPT tokenization helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--shard-dir", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--plan-dir", required=True)
    plan.add_argument("--languages", default="")
    plan.add_argument("--array-ranks", type=int, required=True)
    plan.add_argument("--sample-fraction", type=float, default=1.0)
    plan.add_argument("--train-fraction", type=float, default=1.0)
    plan.add_argument("--validation-fraction", type=float, default=0.0)
    plan.set_defaults(func=_cmd_plan)

    worker = sub.add_parser("worker")
    worker.add_argument("--repo-root", required=True)
    worker.add_argument("--bridge-root", required=True)
    worker.add_argument("--plan-dir", required=True)
    worker.add_argument("--ranks-dir", required=True)
    worker.add_argument("--rank", type=int, default=None)
    worker.add_argument("--tokenizer-type", required=True)
    worker.add_argument("--tokenizer-model", required=True)
    worker.add_argument("--trust-remote-code", action="store_true")
    worker.add_argument("--append-eod", action="store_true")
    worker.add_argument("--workers", type=int, default=4)
    worker.add_argument("--workers-per-language", type=int, default=0)
    worker.add_argument("--partitions-per-language", type=int, default=0)
    worker.add_argument("--language-jobs", type=int, default=0)
    worker.add_argument("--blend-weight-policy", default="equal")
    worker.add_argument("--launcher", choices=["python", "uv"], default="python")
    worker.add_argument("--python-executable", default=sys.executable)
    worker.add_argument("--sample-fraction", type=float, default=1.0)
    worker.add_argument("--max-samples", type=int, default=0)
    worker.add_argument("--train-fraction", type=float, default=1.0)
    worker.add_argument("--split-seed", type=int, default=42)
    worker.add_argument("--keep-work-dir", action="store_true")
    worker.add_argument("--force", action="store_true")
    worker.set_defaults(func=_cmd_worker)

    merge = sub.add_parser("merge")
    merge.add_argument("--ranks-dir", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--tokenizer-type", required=True)
    merge.add_argument("--tokenizer-model", required=True)
    merge.add_argument("--trust-remote-code", action="store_true")
    merge.add_argument("--blend-weight-policy", default="equal")
    merge.add_argument("--check-prefix-files", action="store_true")
    merge.set_defaults(func=_cmd_merge)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
