from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import PipelineContext


TOKENIZER_MANIFEST_NAME = "tokenizer_manifest.json"
TOKENIZED_MANIFESTS_NAME = "tokenized_manifests.json"


def tokenizer_training_enabled(ctx: "PipelineContext") -> bool:
    cfg = ctx.tokenizer_cfg
    if not isinstance(cfg, dict):
        return False
    return bool(cfg.get("enabled", False))


def tokenizer_manifest_path(ctx: "PipelineContext") -> Path:
    return ctx.tokenizers_dir / TOKENIZER_MANIFEST_NAME


def tokenized_manifests_path(ctx: "PipelineContext") -> Path:
    return ctx.tokenized_dir / TOKENIZED_MANIFESTS_NAME


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return payload


def load_tokenizer_manifest(ctx: "PipelineContext") -> dict[str, Any]:
    path = tokenizer_manifest_path(ctx)
    if not path.exists():
        raise SystemExit(f"Missing tokenizer training manifest: {path}")
    return load_json(path)


def load_tokenized_manifest_index(tokenized_path: Path) -> dict[str, Any] | None:
    path = tokenized_path / TOKENIZED_MANIFESTS_NAME
    if not path.exists():
        return None
    return load_json(path)


def resolve_run_tokenizer_id(common: dict[str, Any], run_cfg: dict[str, Any]) -> str:
    run_id = str(run_cfg.get("id", "")).strip()
    explicit = str(run_cfg.get("tokenizer_id", "")).strip()
    if explicit:
        return explicit
    mapping = common.get("tokenizer_map", {})
    if isinstance(mapping, dict) and run_id:
        mapped = str(mapping.get(run_id, "")).strip()
        if mapped:
            return mapped
    return str(common.get("tokenizer_id", "")).strip()


def resolve_tokenized_entry(
    tokenized_path: Path,
    common: dict[str, Any],
    run_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    index = load_tokenized_manifest_index(tokenized_path)
    if index is None:
        return None
    tokenizers = index.get("tokenizers", {})
    if not isinstance(tokenizers, dict) or not tokenizers:
        raise SystemExit(f"Invalid tokenized manifest index: {tokenized_path / TOKENIZED_MANIFESTS_NAME}")
    tokenizer_id = resolve_run_tokenizer_id(common, run_cfg)
    if not tokenizer_id:
        if len(tokenizers) == 1:
            tokenizer_id = next(iter(tokenizers.keys()))
        else:
            run_id = str(run_cfg.get("id", "")).strip()
            raise SystemExit(f"Training run '{run_id}' must set tokenizer_id when multiple tokenizers exist")
    entry = tokenizers.get(tokenizer_id)
    if not isinstance(entry, dict):
        raise SystemExit(
            f"Training run '{run_cfg.get('id', '')}' references unknown tokenizer_id "
            f"'{tokenizer_id}'. Available: {sorted(tokenizers.keys())}"
        )
    return entry
