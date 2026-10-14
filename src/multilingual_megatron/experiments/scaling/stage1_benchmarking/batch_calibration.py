"""Bounded H100 capacity calibration; outputs are never paper benchmark results.

Use complete long NTREX documents and the native scorer's measured batch probes.
Choose the largest measured memory/parity-passing power of two, rather than the
throughput-selected size. The per-tier minimum is only a candidate for the
subsequent benchmark-level numerical sensitivity gate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import full_suite_campaign as runtime
from .contracts import _verify_payload_hash
from .full_suite_campaign import checked, read, ref, save

MODULE = "multilingual_megatron.experiments.scaling.stage1_benchmarking.batch_calibration"


def select_long_documents(requests, widths, count):
    """Select by input length, retaining every window of selected documents."""
    ranked = sorted(range(len(requests)), key=lambda i: (-int(widths[i]), i))[:count]
    documents = {(requests[i]["task"], requests[i]["doc_id"]) for i in ranked}
    return [i for i, row in enumerate(requests) if (row["task"], row["doc_id"]) in documents]


def common_candidates(receipts):
    """Fail closed on incomplete architecture coverage or failed calibration."""
    expected = {(a, e) for a in "ABCD" for e in (1, 4, 16, 32)}
    if set(receipts) != expected:
        raise ValueError("need all 16 size/expert calibration receipts")
    passing = {}
    for key, receipt in receipts.items():
        if receipt["status"] != "calibrated":
            raise ValueError(f"calibration incomplete: {key}")
        sizes = {row["batch_size"] for row in receipt["trials"]
                 if row["status"] == "measured" and row["upper_length"] == 2048
                 and row["memory_ok"] and row["parity_ok"]}
        if not sizes:
            raise ValueError(f"no passing full-context batch: {key}")
        passing[key] = sizes
    # Intersection requires the exact candidate to have passed for every E.
    return {a: max(set.intersection(*(passing[a, e] for e in (1, 4, 16, 32))))
            for a in "ABCD"}


def prepare(profile_path, output):
    import numpy as np
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import load_native_benchmark_score_spec
    from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import (
        TokenCacheWriter, load_cached_sequences,
    )
    profile_path, output = profile_path.resolve(), output.resolve()
    profile = read(profile_path)
    repo = (profile_path.parent / profile["repo_root"]).resolve()
    if output.exists():
        raise ValueError("use a fresh calibration output directory")
    source_path = repo / profile["source_campaign"]
    source = read(source_path)
    _verify_payload_hash(source, field="manifest_sha256", label="source campaign")
    models = [j for j in source["jobs"] if j["benchmark"] == "ntrex_mono"
              and j["K"] == 200 and j["seed"] == 1234]
    if {(j["ladder"], j["E"]) for j in models} != {(a,e) for a in "ABCD" for e in (1,4,16,32)} or len(models) != 16:
        raise ValueError("source lacks the 16 architecture controls")
    original = read(checked(models[0]["spec"]))
    spec = load_native_benchmark_score_spec(Path(models[0]["spec"]["path"]))
    requests = read(checked(original["request_manifest"]))
    sequences = load_cached_sequences(original["token_cache"], spec, requests["requests"])
    widths = np.asarray([len(s.input_ids) for s in sequences])
    indices = select_long_documents(requests["requests"], widths, profile["calibration"]["max_batch_size"])
    if sum(widths[i] > 1024 for i in indices) < profile["calibration"]["max_batch_size"] or max(widths) != 2048:
        raise ValueError("insufficient full-context calibration requests")
    selected = [requests["requests"][i] for i in indices]
    doc_keys = {(r["task"], r["doc_id"]) for r in selected}
    tasks = []
    for task in requests["tasks"]:
        docs = [d for d in task["documents"] if (task["task"], d["doc_id"]) in doc_keys]
        if docs:
            tasks.append(dict(task, documents=docs))
    output.mkdir(parents=True)
    payload = dict(schema=requests["schema"], protocol=requests["protocol"],
                   sampling={"purpose": "capacity calibration only; not benchmark results",
                             "selection": "longest requests plus complete source document windows",
                             "source_manifest": original["request_manifest"]}, tasks=tasks, requests=selected)
    request_ref = save(output / "data/requests.json", payload, "manifest_sha256")
    writer = TokenCacheWriter(output / "data/token_cache")
    for i in indices:
        writer.append(sequences[i])
    token_ref = writer.finish(request_sha256=request_ref["sha256"], tokenizer=original["tokenizer"], sequence_length=2048)
    jobs = []
    for model in models:
        payload = copy.deepcopy(read(checked(model["spec"])))
        payload.pop("spec_sha256")
        job_id = f"{model['ladder'].lower()}_e{model['E']}"
        payload.update(run_id=output.name + "_" + job_id, output_dir=str(output / "scores" / job_id),
                       request_manifest=request_ref, token_cache=token_ref)
        payload["evaluation"].update(required_gpu_name_substring="H100", expected_world_size=1,
            micro_batch_size=profile["reference_batch"], parity_requests=32,
            minimum_sequence_width=2048, max_gpu_hours=profile["slurm"]["minutes"] / 60,
            batching=dict(mode="auto", **profile["calibration"]))
        payload["calibration_provenance"] = dict(source_spec=model["spec"], source_campaign=ref(source_path),
            claim="capacity probe only; numerical benchmark acceptance pending")
        reference = save(output / "specs" / (job_id + ".json"), payload, "spec_sha256")
        load_native_benchmark_score_spec(Path(reference["path"]))
        job = dict(id=job_id, ladder=model["ladder"], E=model["E"], model_id=model["model_id"],
                   spec=reference, gpus=1, memory=profile["slurm"]["memory"], walltime_minutes=profile["slurm"]["minutes"])
        job["run_script"], job["sbatch_script"] = runtime.write_job(output, job, profile, repo, module=MODULE, job_prefix="s1-h100-batch-")
        jobs.append(job)
    campaign = dict(schema="stage1_h100_batch_calibration/v1", repo=str(repo), profile=ref(profile_path),
                    profile_payload=profile, source_campaign=ref(source_path), jobs=jobs,
                    source_manifest=runtime.freeze_source(repo, output, profile),
                    status="prepared_capacity_probe_only")
    save(output / "campaign.json", campaign, "manifest_sha256")
    return dict(jobs=len(jobs), requests=len(selected), longest_width=int(max(widths)),
                maximum_allocated_gpu_hours=len(jobs)*profile["slurm"]["minutes"]/60)


def collect(path):
    campaign = read(path)
    _verify_payload_hash(campaign, field="manifest_sha256", label="calibration campaign")
    receipts = {}
    missing = []
    for job in campaign["jobs"]:
        directory = Path(read(checked(job["spec"]))["output_dir"])
        target = directory / "batching_calibration.json"
        if not target.exists():
            missing.append(job["id"])
        else:
            receipts[job["ladder"],job["E"]] = read(target)
    result = dict(status="incomplete" if missing else "capacity_measured_numerical_gate_pending", missing=missing,
                  campaign=ref(path), production_accepted=False)
    if not missing:
        result["candidate_batches_by_tier"] = common_candidates(receipts)
        result["boundary"] = "Largest commonly measured passing power of two at 2048 width, within configured cap/headroom. Not proof of the absolute maximum or benchmark-level numerical equivalence."
    save(path.parent / "capacity_summary.json", result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command", required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--profile",type=Path,required=True);prep.add_argument("--output",type=Path,required=True)
    for name in ("run", "collect"):
        p=sub.add_parser(name);p.add_argument("--campaign",type=Path,required=True)
        if name=="run":p.add_argument("--job",required=True)
    args=parser.parse_args()
    if args.command=="prepare":result=prepare(args.profile,args.output)
    elif args.command=="collect":result=collect(args.campaign)
    else:
        campaign=read(args.campaign)
        _verify_payload_hash(campaign,field="manifest_sha256",label="calibration campaign")
        runtime.run(campaign,args.job);result={"completed":args.job}
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
