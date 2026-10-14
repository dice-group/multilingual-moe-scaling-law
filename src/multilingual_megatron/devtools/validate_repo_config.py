#!/usr/bin/env python3
"""Lightweight repo config validator for CI and local preflight."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from multilingual_megatron.devtools.test_suites import validate_suite_manifests
from multilingual_megatron.repo import find_repo_root


def _require_path(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing path: {path}")


def _require_json_object(path: Path, errors: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - direct diagnostics path
        errors.append(f"invalid json ({path}): {exc}")
        return
    if not isinstance(payload, dict) or not payload:
        errors.append(f"expected non-empty JSON object: {path}")


def _validate_pipeline_config(repo_root: Path, errors: list[str]) -> None:
    config_path = repo_root / "configs" / "pipeline"
    if not config_path.exists():
        errors.append(f"missing path: {config_path}")
        return

    src_root = repo_root / "src"
    for path in (str(src_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from multilingual_megatron.pipeline.config_validator import (  # pylint: disable=import-outside-toplevel
            validate_proof_pipeline_config_file,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        errors.append(f"failed loading pipeline config validator: {exc}")
        return

    for err in validate_proof_pipeline_config_file(config_path):
        errors.append(f"pipeline config: {err}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("portable", "full"),
        default="full",
        help="portable checks tracked repo contracts; full also requires submodules.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = find_repo_root(Path(__file__))
    errors: list[str] = []

    portable_paths = [
        repo_root / "README.md",
        repo_root / "configs" / "pipeline",
        repo_root / "configs" / "pipeline" / "schema.json",
        repo_root / "configs" / "language_groupings" / "bridge_smoke_4lang.json",
        repo_root / "configs" / "data_sources" / "fineweb2-language-distribution.csv",
        repo_root / "configs" / "lm_eval" / "lm_eval_tasks_10_langs_smoke.txt",
        repo_root / "outputs" / "sample_data" / "fixtures" / "multilingual_4lang",
        repo_root / "src" / "multilingual_megatron" / "pipeline",
        repo_root / "src" / "multilingual_megatron" / "pipeline" / "stages",
        repo_root / "src" / "multilingual_megatron" / "pipeline" / "backends" / "megatron_bridge",
    ]
    full_paths = [
        repo_root / "external" / "Megatron-Bridge",
        repo_root / "external" / "lm-evaluation-harness",
    ]

    for path in [*portable_paths, *(full_paths if args.scope == "full" else [])]:
        _require_path(path, errors)

    for tier_file in [
        repo_root / "configs" / "language_groupings" / "200_languages_grouping.json",
    ]:
        if tier_file.exists():
            _require_json_object(tier_file, errors)

    _validate_pipeline_config(repo_root, errors)
    errors.extend(f"test suites: {error}" for error in validate_suite_manifests(repo_root))

    if errors:
        print("[FAIL] Repo config validation failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print(f"[OK] Repo config validation passed (scope={args.scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
