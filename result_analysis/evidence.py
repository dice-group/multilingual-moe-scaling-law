"""Explicit contract for byte-identical paper inputs and derived outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import platform
import time
import sys
from importlib.metadata import version

import pandas as pd


def main():
    """Prepare or verify inputs without dispatching result analyses."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--profile", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.profile, args.output) if args.command == "prepare" else Evidence(args.inputs).verify()
    print(json.dumps(result, indent=2, allow_nan=False))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inside(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"artifact paths must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"path escapes artifact root: {relative}")
    return path


def prepare(profile_path: Path, destination: Path) -> dict:
    """Validate every source before copying; never overwrite an evidence bundle."""
    profile = json.loads(profile_path.read_text())
    if profile["schema"] != "stage1-paper-inputs/v1":
        raise ValueError("unsupported paper-input profile")
    roots = {name: Path(value).resolve() for name, value in profile["roots"].items()}
    sources = {}
    for name, record in profile["files"].items():
        inside(destination, name)
        source = inside(roots[record["root"]], record["path"])
        if sha256(source) != record["sha256"]:
            raise ValueError(f"source hash mismatch: {name}")
        sources[name] = source
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        target = inside(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != profile["files"][name]["sha256"]:
            raise ValueError(f"copy hash mismatch: {name}")
    manifest = {"schema": profile["schema"], "files": profile["files"],
                "profile_sha256": sha256(profile_path), "roots_at_preparation": profile["roots"],
                "paper_commit": profile["paper_commit"], "no_new_training_or_law_fitting": True}
    if "analysis" in profile:
        manifest["analysis"] = profile["analysis"]
    write_json(destination / "manifest.json", manifest)
    return {"files": len(sources), "input_root": str(destination.resolve())}


class Evidence:
    """Read only declared, hash-verified inputs, including after relocation."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != "stage1-paper-inputs/v1":
            raise ValueError("unsupported evidence manifest")
        self.used: dict[str, str] = {}
        self.started = time.monotonic()

    @property
    def analysis(self) -> dict:
        """Versioned presentation contract; never reinterpret a historical selector."""
        return self.manifest.get("analysis", {})

    def require_current(self) -> dict:
        if self.analysis.get("version") != "stage1-paper-v2":
            raise ValueError("this result requires the versioned stage1-paper-v2 evidence bundle")
        return self.analysis

    def path(self, name: str) -> Path:
        record = self.manifest["files"][name]
        path = inside(self.root, name)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        self.used[name] = record["sha256"]
        return path

    def json(self, name: str):
        return json.loads(self.path(name).read_text())

    def table(self, name: str) -> pd.DataFrame:
        path = self.path(name)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)

    def verify(self) -> dict:
        for name in self.manifest["files"]:
            self.path(name)
        return {"verified_files": len(self.used), "paper_commit": self.manifest["paper_commit"]}


def output_directory(path: Path, evidence: Evidence) -> Path:
    output = path.resolve()
    if output == evidence.root or output.is_relative_to(evidence.root) or evidence.root.is_relative_to(output):
        raise ValueError("outputs must be separate from immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    return output


def finish(output: Path, evidence: Evidence, result: str, claims: dict, sources: list[Path]) -> dict:
    write_json(output / "claims.json", claims)
    # Include imported repository dependencies, not only the top-level driver.
    # External libraries are identified by versions below, not copied or hashed.
    root = Path(__file__).resolve().parents[1]
    code = {p.resolve() for p in sources}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(root / directory)
                                           for directory in ("result_analysis", "src/multilingual_megatron")):
                code.add(path)
    receipt = {"result": result, "paper_commit": evidence.manifest["paper_commit"],
               "input_manifest_sha256": sha256(evidence.root / "manifest.json"),
               "inputs": evidence.used, "code": {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha256(p) for p in sorted(code)},
               "outputs": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
               "new_training": False, "new_law_fit": False,
               "python": platform.python_version(),
               "dependencies": {name: version(name) for name in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib")},
               "independent_unit": "training trajectory; languages and readouts are nested",
               "inference": "descriptive replay, not a new significance or confirmatory test"}
    receipt["elapsed_seconds"] = time.monotonic() - evidence.started
    receipt["analysis"] = evidence.analysis
    receipt["reproduction_scope"] = claims.get("reproduction_scope", "see result README and claims")
    write_json(output / "receipt.json", receipt)
    return claims


def read_result(output: Path, evidence: Evidence) -> dict:
    """Verify saved outputs before a notebook displays them; never refit implicitly."""
    receipt = json.loads((output / "receipt.json").read_text())
    if sha256(evidence.root / "manifest.json") != receipt["input_manifest_sha256"]:
        raise ValueError("result belongs to a different evidence bundle")
    for name, expected in receipt["inputs"].items():
        if sha256(evidence.path(name)) != expected:
            raise ValueError(f"changed result input: {name}")
    for name, expected in receipt["outputs"].items():
        if sha256(inside(output, name)) != expected:
            raise ValueError(f"changed derived output: {name}")
    return json.loads((output / "claims.json").read_text())


if __name__ == "__main__":
    main()
