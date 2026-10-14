"""Load and validate the curated unittest suite manifests."""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_SUITES = ("core_pipeline", "stage1_science")
OPTIONAL_SUITES = ("reviewer_cpu",)
RUNNABLE_SUITES = (*DEFAULT_SUITES, *OPTIONAL_SUITES)
_MODULE_PATTERN = re.compile(r"^tests\.test_[a-z0-9_]+$")


def suite_manifest_path(repo_root: Path, suite: str) -> Path:
    if suite not in RUNNABLE_SUITES:
        raise ValueError(f"unknown test suite: {suite}")
    return repo_root / "tests" / "suites" / f"{suite}.txt"


def load_suite_modules(path: Path) -> tuple[str, ...]:
    modules: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        module = raw_line.strip()
        if not module or module.startswith("#"):
            continue
        if not _MODULE_PATTERN.fullmatch(module):
            raise ValueError(f"{path}:{line_number}: invalid unittest module {module!r}")
        if module in seen:
            raise ValueError(f"{path}:{line_number}: duplicate unittest module {module!r}")
        seen.add(module)
        modules.append(module)
    if not modules:
        raise ValueError(f"test suite is empty: {path}")
    return tuple(modules)


def validate_suite_manifests(repo_root: Path) -> list[str]:
    errors: list[str] = []
    memberships: dict[str, str] = {}
    for suite in RUNNABLE_SUITES:
        path = suite_manifest_path(repo_root, suite)
        if not path.is_file():
            errors.append(f"missing test suite manifest: {path}")
            continue
        try:
            modules = load_suite_modules(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        for module in modules:
            module_path = repo_root / Path(*module.split(".")).with_suffix(".py")
            if not module_path.is_file():
                errors.append(f"{suite} references missing test module: {module}")
            previous = memberships.get(module)
            if previous is not None:
                errors.append(f"test module appears in both {previous} and {suite}: {module}")
            memberships[module] = suite
    return errors
