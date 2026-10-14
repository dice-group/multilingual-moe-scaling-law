#!/usr/bin/env python3
"""Shared runtime helpers for local pipeline orchestrators.

This module intentionally stays lightweight and dependency-minimal so orchestration
scripts can reuse common logic without duplicating implementation details.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .config_loader import (
    deep_merge as _loader_deep_merge,
    load_json as _loader_load_json,
    load_pipeline_config as _loader_load_pipeline_config,
    load_pipeline_config_bundle as _loader_load_pipeline_config_bundle,
)


LOGGER = logging.getLogger("proof_pipeline")


def _load_json(path: Path) -> dict[str, Any]:
    return _loader_load_json(path)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _deep_merge(base: Any, override: Any) -> Any:
    return _loader_deep_merge(base, override)


def _load_pipeline_config(path: Path) -> dict[str, Any]:
    try:
        return _loader_load_pipeline_config(path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _load_pipeline_config_bundle(bundle_dir: Path) -> dict[str, Any]:
    try:
        return _loader_load_pipeline_config_bundle(bundle_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_profile(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise SystemExit(f"Unknown profile '{profile_name}'. Available: {sorted(profiles.keys())}")

    resolved_cache: dict[str, dict[str, Any]] = {}

    def _resolve_profile_chain(name: str, chain: list[str]) -> dict[str, Any]:
        if name in resolved_cache:
            return resolved_cache[name]
        if name in chain:
            cycle = " -> ".join(chain + [name])
            raise SystemExit(f"Profile inheritance cycle detected: {cycle}")

        override = profiles.get(name)
        if not isinstance(override, dict):
            raise SystemExit(f"Profile '{name}' must be an object.")

        extends_raw = override.get("extends")
        parent_name = ""
        if extends_raw is not None:
            if not isinstance(extends_raw, str) or not extends_raw.strip():
                raise SystemExit(f"Profile '{name}'.extends must be a non-empty string when provided.")
            parent_name = extends_raw.strip()
            if parent_name not in profiles:
                raise SystemExit(f"Profile '{name}'.extends references unknown profile '{parent_name}'.")

        current_override = dict(override)
        current_override.pop("extends", None)
        parent_payload: dict[str, Any] = {}
        if parent_name:
            parent_payload = _resolve_profile_chain(parent_name, chain + [name])

        merged = _deep_merge(parent_payload, current_override)
        if not isinstance(merged, dict):
            raise SystemExit(f"Profile '{name}' did not resolve to an object.")
        resolved_cache[name] = merged
        return merged

    return _deep_merge(defaults, _resolve_profile_chain(profile_name, []))


def _abs(repo_root: Path, maybe_rel: str) -> Path:
    path = Path(maybe_rel)
    return path if path.is_absolute() else (repo_root / path)


def _missing_modules(names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


from .executor import Executor, LocalExecutor, SlurmExecutor, build_executor  # noqa: F401


_DEFAULT_EXECUTOR = LocalExecutor()


def _run_with_tee(
    cmd: list[str],
    env: dict[str, str],
    dry_run: bool,
    log_path: Path,
    executor: Executor | None = None,
) -> None:
    normalized_cmd = list(cmd)
    python_executable = str(env.get("PYTHON_EXECUTABLE", "")).strip()
    runner = executor or _DEFAULT_EXECUTOR
    if normalized_cmd and normalized_cmd[0] in {"python", "python3"}:
        if isinstance(runner, SlurmExecutor):
            normalized_cmd[0] = "python"
        elif python_executable:
            normalized_cmd[0] = python_executable
    runner.run(normalized_cmd, env, dry_run, log_path)


def _run_stage_cmd(
    *,
    cmd: list[str],
    env: dict[str, str],
    dry_run: bool,
    logs_root: Path,
    stage: str,
    suffix: str = "",
    executor: Executor | None = None,
) -> None:
    log_path = _stage_log_path(logs_root, env, stage, suffix)
    _run_with_tee(cmd, env, dry_run, log_path, executor=executor)


def _structured_logs_enabled(env: dict[str, str]) -> bool:
    return str(env.get("PROOF_PIPELINE_STRUCTURED_LOGS", "")).strip().lower() in {"1", "true", "yes", "on"}


def _log_path_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip())
    return cleaned.strip(".-") or "default"


def _stage_log_path(
    logs_root: Path,
    env: dict[str, str],
    stage: str,
    suffix: str = "",
) -> Path:
    """Build a deterministic log path for a stage invocation."""
    if _structured_logs_enabled(env):
        attempt_root = str(env.get("PROOF_PIPELINE_ATTEMPT_LOGS_ROOT", "")).strip()
        if attempt_root and Path(logs_root).resolve() == Path(attempt_root).resolve():
            stage_dir = logs_root / "stages" / _log_path_component(stage)
            if suffix:
                stage_dir = stage_dir / _log_path_component(suffix)
            return stage_dir / f"{_log_path_component(stage)}.log"
        suffix_part = f".{_log_path_component(suffix)}" if suffix else ""
        return logs_root / f"{_log_path_component(stage)}{suffix_part}.log"

    suffix_part = f".{suffix}" if suffix else ""
    run_timestamp = str(env.get("PROOF_PIPELINE_RUN_TIMESTAMP", "")).strip()
    log_name = f"{stage}{suffix_part}.{run_timestamp}.log" if run_timestamp else f"{stage}{suffix_part}.log"
    return logs_root / log_name


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def _model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return _sanitize_slug(value)


def _build_run_label(base_run_label: str, run: dict[str, Any], default_num_experts: int) -> str:
    del default_num_experts
    run_kind = str(run.get("kind", "run"))
    run_id = str(run.get("id", run_kind))
    return _sanitize_slug(f"{base_run_label}-{run_id}")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    return _load_json(path)


def _save_stage_state(
    state_path: Path,
    stage: str,
    signature: str,
    artifacts: list[str],
) -> None:
    state = _load_state(state_path)
    stages = state.setdefault("stages", {})
    stages[stage] = {
        "signature": signature,
        "status": "completed",
        "artifacts": artifacts,
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    _dump_json(state_path, state)


def _stage_signature(payload: dict[str, Any]) -> str:
    code_sig = str(os.environ.get("PROOF_PIPELINE_CODE_SIG", "")).strip()
    if code_sig:
        payload = {"_code_sig": code_sig, **payload}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compute_pipeline_code_signature(repo_root: Path) -> str:
    digest = hashlib.sha256()
    scan_paths: list[Path] = [
        repo_root / "src" / "multilingual_megatron" / "pipeline",
    ]
    for candidate in scan_paths:
        if candidate.is_file():
            digest.update(str(candidate.relative_to(repo_root)).encode("utf-8"))
            digest.update(candidate.read_bytes())
            continue
        if candidate.is_dir():
            for file_path in sorted(candidate.rglob("*.py")):
                digest.update(str(file_path.relative_to(repo_root)).encode("utf-8"))
                digest.update(file_path.read_bytes())
    return digest.hexdigest()[:16]


def _artifacts_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _should_skip_stage(
    *,
    state_path: Path,
    stage: str,
    signature: str,
    expected_artifacts: list[Path],
    force_stages: set[str],
    dry_run: bool,
) -> bool:
    if stage in force_stages or dry_run:
        return False
    state = _load_state(state_path)
    info = state.get("stages", {}).get(stage)
    if not isinstance(info, dict):
        return False
    if info.get("signature") != signature:
        return False
    return _artifacts_exist(expected_artifacts)


def _count_experts(language_map_path: Path) -> int:
    payload = _load_json(language_map_path)
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    raise ValueError(f"Unsupported language_map format in {language_map_path}")


def _latest_metric(history: list[dict[str, Any]], suffix: str) -> float | None:
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if suffix in entry and isinstance(entry[suffix], (int, float)):
            return float(entry[suffix])
        for key, value in entry.items():
            if key.endswith("/" + suffix) and isinstance(value, (int, float)):
                return float(value)
    return None


def _latest_metric_any_suffix(history: list[dict[str, Any]], suffixes: list[str]) -> float | None:
    for suffix in suffixes:
        value = _latest_metric(history, suffix)
        if value is not None:
            return value
    return None


def _read_task_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _select_tasks_for_languages(
    *,
    tasks_file: Path,
    languages: list[str],
    include_belebele: bool,
    include_sib200: bool,
    include_taxi1500: bool,
    include_flores: bool,
    flores_mode: str,
) -> list[str]:
    lang_set = set(languages)
    selected: list[str] = []
    for task in _read_task_lines(tasks_file):
        if include_belebele and task.startswith("belebele_"):
            lang = task.replace("belebele_", "", 1)
            if lang in lang_set:
                selected.append(task)
                continue

        if include_sib200 and task.startswith("sib200_"):
            lang = task.replace("sib200_", "", 1)
            if lang in lang_set:
                selected.append(task)
                continue

        if include_taxi1500 and task.startswith("taxi1500_"):
            lang = task.replace("taxi1500_", "", 1)
            if lang in lang_set:
                selected.append(task)
                continue

        if include_flores and task.startswith("flores_en_perplexity_"):
            rest = task.replace("flores_en_perplexity_", "", 1)
            if "-" not in rest:
                continue
            src, tgt = rest.split("-", 1)
            include = False
            if flores_mode == "both":
                include = src in lang_set or tgt in lang_set
            elif flores_mode == "into_en":
                include = src in lang_set and tgt == "eng_Latn"
            elif flores_mode == "from_en":
                include = src == "eng_Latn" and tgt in lang_set
            if include:
                selected.append(task)
    return selected


def _resolve_profile_languages(
    *,
    sample_cfg: dict[str, Any],
    sample_manifest_path: Path,
    sampled_dir: Path,
) -> list[str]:
    explicit = [item for item in sample_cfg.get("languages", []) if isinstance(item, str) and item.strip()]
    if explicit:
        return sorted(set(explicit))

    if sample_manifest_path.exists():
        try:
            manifest = _load_json(sample_manifest_path)
            selection = manifest.get("selection", {})
            langs = selection.get("languages", [])
            if isinstance(langs, list) and langs:
                return sorted({str(item) for item in langs if str(item).strip()})
        except Exception:
            pass

    plan_csv = sampled_dir / "sampling_plan.csv"
    if plan_csv.exists():
        langs: list[str] = []
        with plan_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                subset = str(row.get("subset", "")).strip()
                if subset:
                    langs.append(subset)
        if langs:
            return sorted(set(langs))
    return []
