"""Verify immutable inputs and run existing paper analyses into fresh outputs."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import time

from result_analysis.evidence import Evidence, inside, output_directory, sha256, write_json
from multilingual_megatron.devtools.anonymous_provenance import anonymous_source_id


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/result_analysis/paper_reproduction.json"


def validate_inputs(evidence, contract):
    if contract["schema"] != "stage1-paper-reproduction/v1":
        raise ValueError("unsupported reproduction contract")
    if anonymous_source_id(evidence.manifest["paper_commit"]) != anonymous_source_id(contract["paper_commit"]):
        raise ValueError("paper revision differs from reproduction contract")
    if evidence.require_current() != contract["analysis"]:
        raise ValueError("law attribution or scoring membership differs from reproduction contract")
    if len(evidence.manifest["files"]) != contract["input_files"]:
        raise ValueError("input inventory differs from reproduction contract")
    return evidence.verify()


def verify_assets(root, contract):
    records = []
    for asset in contract["assets"]:
        path = inside(root, asset["path"])
        if sha256(path) != asset["sha256"]:
            raise ValueError(f"manuscript asset hash mismatch: {asset['path']}")
        records.append(dict(asset))
    return records


def run(inputs, output, *, mode="smoke", paper_assets=None):
    started = time.monotonic()
    contract = json.loads(CONTRACT.read_text())
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    evidence = Evidence(Path(inputs))
    verified = validate_inputs(evidence, contract)
    assets = []
    if mode == "full":
        default = ROOT / "paper_assets"
        paper_assets = Path(paper_assets) if paper_assets else (
            default if default.is_dir() else ROOT / contract["asset_source"])
        assets = verify_assets(paper_assets, contract)
        target = Path(output).resolve()
        source = paper_assets.resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("outputs must be separate from manuscript assets")
    output = output_directory(Path(output), evidence)
    summary = {"schema": "stage1-paper-reproduction-result/v1", "mode": mode,
               "inputs": verified, "input_manifest_sha256": sha256(evidence.root / "manifest.json"),
               "contract_sha256": sha256(CONTRACT), "analysis": evidence.analysis,
               "scope": contract["replay_scope"], "results": {}}
    for topic in contract[mode]:
        module = "result_analysis." + topic
        if not topic.endswith(".inventory"):
            module += ".reproduce"
        # Keep each topic's input-use receipt and timing independent.
        claims = importlib.import_module(module).run(Evidence(evidence.root), output / topic)
        summary["results"][topic] = {"checks_passed": claims["checks_passed"],
            "receipt_sha256": sha256(output / topic / "receipt.json")}
        print(f"{topic}: {claims['checks_passed']} checks passed", flush=True)
    if assets:
        for asset in assets:
            destination = inside(output / "paper_assets", asset["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(paper_assets, asset["path"]), destination)
        verify_assets(output / "paper_assets", contract)
        write_json(output / "paper_assets/manifest.json", {
            "mode": "verified retained manuscript assets; fresh review renders are in topic directories",
            "paper_commit": contract["paper_commit"], "files": assets})
    summary["verified_assets"] = len(assets)
    summary["checks_passed"] = sum(r["checks_passed"] for r in summary["results"].values())
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory, separate from inputs")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--paper-assets", type=Path, help="retained manuscript asset root; full mode only")
    args = parser.parse_args(argv)
    result = run(args.inputs, args.output, mode=args.mode, paper_assets=args.paper_assets)
    print(f"Verified {result['inputs']['verified_files']} inputs; {result['checks_passed']} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
