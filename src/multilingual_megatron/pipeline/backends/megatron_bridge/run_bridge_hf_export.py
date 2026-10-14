#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

from multilingual_megatron.repo import find_repo_root


def _repo_root() -> Path:
    return find_repo_root(Path(__file__))


def _ensure_bridge_paths() -> None:
    bridge_root = _repo_root() / "external" / "Megatron-Bridge"
    for path in (
        bridge_root / "src",
        bridge_root / "3rdparty" / "Megatron-LM",
    ):
        if path.exists():
            sys.path.insert(0, str(path))


def _import_auto_bridge():
    _ensure_bridge_paths()
    try:
        from megatron.bridge import AutoBridge  # type: ignore[attr-defined]
    except ImportError:
        from megatron.bridge.models.conversion import AutoBridge
    return AutoBridge


def _register_bridge_modules() -> None:
    for module_name in (
        "megatron.bridge.models.llama",
        "megatron.bridge.models.gemma",
        "megatron.bridge.models.glm",
        "megatron.bridge.models.mistral",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"[WARN] Could not import Bridge model registrations from {module_name}: {exc}", file=sys.stderr)


def _validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Megatron checkpoint path does not exist: {path}")
    if list(path.glob("**/run_config.yaml")):
        return
    raise FileNotFoundError(f"Could not find run_config.yaml under Megatron checkpoint path: {path}")


def export_megatron_to_hf(
    *,
    hf_model: str,
    megatron_path: Path,
    hf_path: Path,
    show_progress: bool,
    strict: bool,
    trust_remote_code: bool,
) -> None:
    AutoBridge = _import_auto_bridge()
    _register_bridge_modules()
    _validate_checkpoint(megatron_path)
    hf_path.mkdir(parents=True, exist_ok=True)
    bridge = AutoBridge.from_auto_config(str(megatron_path), hf_model, trust_remote_code=trust_remote_code)
    bridge.export_ckpt(
        megatron_path=str(megatron_path),
        hf_path=str(hf_path),
        show_progress=show_progress,
        strict=strict,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Megatron Bridge checkpoint to Hugging Face format.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Export Megatron checkpoint to Hugging Face format")
    export_parser.add_argument("--hf-model", required=True)
    export_parser.add_argument("--megatron-path", required=True)
    export_parser.add_argument("--hf-path", required=True)
    export_parser.add_argument("--no-progress", action="store_true")
    export_parser.add_argument("--not-strict", action="store_true")
    export_parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.command != "export":
        raise RuntimeError(f"Unsupported command: {args.command}")

    export_megatron_to_hf(
        hf_model=args.hf_model,
        megatron_path=Path(args.megatron_path),
        hf_path=Path(args.hf_path),
        show_progress=not args.no_progress,
        strict=not args.not_strict,
        trust_remote_code=args.trust_remote_code,
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
