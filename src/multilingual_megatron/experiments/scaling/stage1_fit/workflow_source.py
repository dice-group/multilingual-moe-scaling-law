"""Fail-closed source-bundle bindings for modular Stage 1 workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, TypeVar

from .data import sha256_file


ErrorT = TypeVar("ErrorT", bound=Exception)


def bind_source_bundle(*, entrypoint: Path, source_root: Path,
        expected_sources: Mapping[str, str], error_type: type[ErrorT]) -> dict[str, str]:
    """Verify every implementation file before binding the thin entrypoint."""
    entrypoint = entrypoint.resolve()
    source_root = source_root.resolve()
    if not entrypoint.is_file() or not expected_sources:
        raise error_type("workflow source bundle is missing its entrypoint or inventory")
    for relative, expected_sha256 in sorted(expected_sources.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise error_type(f"workflow source path is not repository-relative: {relative}")
        path = (source_root / relative_path).resolve()
        if not path.is_relative_to(source_root) or not path.is_file() \
                or sha256_file(path) != expected_sha256:
            raise error_type(f"workflow implementation source changed: {path}")
    return {"path": str(entrypoint), "file_sha256": sha256_file(entrypoint)}
