"""Bound artifacts and runtime metadata shared by fitting workflows."""
import ast
import importlib.util
from pathlib import Path
import shutil
import platform
import numpy as np
import pandas as pd
import scipy
from .data import sha256_file


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verify(entry):
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"changed bound input: {path}")
    return path


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}


def validate_output(profile):
    """Reject source-tree writes, including through existing symlinks."""
    output = Path(profile["output_root"]).resolve()
    if "workspace_root" in profile:
        workspace = Path(profile["workspace_root"]).resolve()
        if output == workspace or not output.is_relative_to(workspace):
            raise ValueError("output must be a dedicated directory within workspace_root")
    if "read_only_root" in profile:
        source = Path(profile["read_only_root"]).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output overlaps read-only upstream source")
    return output


def dependency_sources(entrypoints: tuple[str, ...]) -> dict[str, Path]:
    """Resolve local Python imports, including deferred CLI imports, for a snapshot.

    Third-party dependencies remain environment dependencies. Local imports
    use relative paths so the copied package can be imported under a new name.
    Historical snapshot files and hashes are never rewritten.
    """
    root = Path(__file__).resolve().parent
    package = __package__
    modules = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).with_suffix("")
        parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
        modules[".".join((package, *parts))] = path
    pending = [package + "." + name for name in entrypoints]
    selected = {}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        path = modules[name]
        selected[name] = path
        parent = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        while parent.startswith(package):
            if parent in modules and parent not in selected:
                pending.append(parent)
            if parent == package:
                break
            parent = parent.rsplit(".", 1)[0]
        context = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            dependencies = []
            if isinstance(node, ast.ImportFrom):
                target = (importlib.util.resolve_name("." * node.level + (node.module or ""), context)
                          if node.level else node.module or "")
                dependencies = [target, *(target + "." + alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                dependencies = [alias.name for alias in node.names]
            pending.extend(dependency for dependency in dependencies if dependency in modules)
    return {path.relative_to(root).as_posix(): path for path in sorted(selected.values())}


def copy_dependency_snapshot(destination: Path, entrypoints: tuple[str, ...]) -> dict[str, Path]:
    """Copy the source closure into a fresh package; never reuse an old snapshot."""
    sources = dependency_sources(entrypoints)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return sources
