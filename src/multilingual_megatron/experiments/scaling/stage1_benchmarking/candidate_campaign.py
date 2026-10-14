"""Prepare, reuse, run and collect supplementary A40 benchmark campaigns."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
from statistics import median

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from .contracts import _atomic_json, _verify_payload_hash
from . import full_suite_campaign as runtime
from .full_suite_campaign import checked, read, ref, save
from .candidate_data import BENCHMARKS, aggregate

MODULE = "multilingual_megatron.experiments.scaling.stage1_benchmarking.candidate_campaign"
SCHEMA = "stage1_candidate_campaign/v1"


def benchmark_names(payload):
    profile = payload.get("profile_payload", payload)
    names = tuple(profile.get("benchmarks", BENCHMARKS))
    if profile.get("extension_protocol"):
        from .extension_sources import FAMILIES
        if names != FAMILIES:
            raise ValueError("extension family inventory mismatch")
    elif names != BENCHMARKS:
        raise ValueError("candidate family inventory mismatch")
    return names


def reduce_scores(campaign, manifest, scores):
    if campaign["profile_payload"].get("extension_protocol"):
        from .extension_data import aggregate as aggregate_extension
        return aggregate_extension(manifest, scores)
    return aggregate(manifest, scores)


def load(path):
    campaign = read(path)
    _verify_payload_hash(campaign, field="manifest_sha256", label="candidate campaign")
    if campaign["schema"] != SCHEMA:
        raise ValueError("not a supplementary candidate campaign")
    return campaign


def select_models(source, selection):
    models = [job for job in source["jobs"] if job["benchmark"] == "xcopa"]
    if selection.get("all_final"):
        runtime.validate_matrix(source["jobs"], source["inputs"])
        if len(models) != selection["expected_models"]:
            raise ValueError("unexpected final checkpoint inventory")
    else:
        models = [j for j in models if j["ladder"] in selection["ladders"] and j["E"] in selection["experts"]
                  and j["K"] in selection["languages"] and j["seed"] == selection["seed"]]
        expected = {(a, e, k, selection["seed"]) for a in ("A", "D") for e in (1, 32) for k in (12, 200)}
        if len(models) != 8 or {(j["ladder"], j["E"], j["K"], j["seed"]) for j in models} != expected:
            raise ValueError("expected exactly eight final checkpoint identities")
    if len({job["model_id"] for job in models}) != len(models):
        raise ValueError("duplicate final checkpoint identity")
    return models


def validate_matrix(campaign, models):
    phase = campaign.get("evaluation_phase", "pilot")
    evaluation = [job for job in campaign["jobs"] if job["phase"] == phase]
    smoke = [job for job in campaign["jobs"] if job["phase"] == "smoke"]
    expected = {(model["model_id"], benchmark) for model in models for benchmark in benchmark_names(campaign)}
    if (len(evaluation) != len(expected)
            or {(j["model_id"], j["benchmark"]) for j in evaluation} != expected
            or len({j["id"] for j in campaign["jobs"]}) != len(campaign["jobs"])
            or len(evaluation) + len(smoke) != len(campaign["jobs"])):
        raise ValueError("incomplete final checkpoint x benchmark grid")
    architectures = {(model["ladder"], model["E"]) for model in models}
    if len(smoke) != len(architectures) or {(j["ladder"], j["E"]) for j in smoke} != architectures:
        raise ValueError("startup checks must cover every model architecture")
    identities = {model["model_id"]: model for model in models}
    for job in campaign["jobs"]:
        if any(job[key] != identities[job["model_id"]][key] for key in ("ladder", "E", "K", "seed")):
            raise ValueError("checkpoint identity drift")
    return evaluation, smoke


def verify_reuse(previous, repo, profile):
    """Preserve pilot numerical semantics while allowing orchestration changes."""
    if previous["profile_payload"]["source_campaign"] != profile["source_campaign"]:
        raise ValueError("reuse campaign belongs to a different final checkpoint inventory")
    if previous["profile_payload"]["tokenizer"] != profile["tokenizer"]:
        raise ValueError("reuse tokenizer mismatch")
    for key in ("version", "multiblimp", "xcomps", "x_csqa", "ties"):
        if previous["protocol"][key] != profile["protocol"][key]:
            raise ValueError("reuse scoring protocol mismatch")
    old_root = Path(previous["source_manifest"]["path"]).parent / "source"
    allowed = {"src/multilingual_megatron/experiments/scaling/stage1_benchmarking/candidate_campaign.py",
               "src/multilingual_megatron/experiments/scaling/stage1_benchmarking/full_suite_monitor.py"}
    for entry in read(checked(previous["source_manifest"]))["files"]:
        path = Path(entry["path"])
        if path.is_relative_to(old_root / "src"):
            relative = path.relative_to(old_root)
            if str(relative) not in allowed:
                checked(dict(path=str(repo / relative), sha256=entry["sha256"]))
        elif not path.is_relative_to(old_root):
            checked(entry)


def prepare(profile_path, output):
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import load_native_benchmark_score_spec
    from multilingual_megatron.pipeline.stages.benchmarking.native_token_cache import combine_token_caches
    output, profile_path = output.resolve(), profile_path.resolve()
    profile = read(profile_path)
    benchmarks = benchmark_names(profile)
    repo = (profile_path.parent / profile["repo_root"]).resolve()
    if (output / "campaign.json").exists() or (output / "source").exists():
        raise ValueError("refusing to overwrite a prepared campaign")
    output.mkdir(parents=True, exist_ok=True)
    previous_path = repo / profile["reuse_campaign"] if profile.get("reuse_campaign") else None
    previous = load(previous_path) if previous_path else None
    if previous:
        verify_reuse(previous, repo, profile)
    phase = "full" if profile["selection"].get("all_final") else "pilot"
    data_root = repo / profile["data_root"] if profile.get("data_root") else output / "data"
    data = {}
    for scope in ("full", "smoke"):
        path = data_root / scope / "data_manifest.json"
        data[scope] = read(path)
        _verify_payload_hash(data[scope], field="manifest_sha256", label=f"{scope} candidate data")
        if set(data[scope]["inputs"]) != set(benchmarks):
            raise ValueError("candidate benchmark inventory mismatch")
    allowed_exclusions = ({"empty answer option", "empty text", "missing or empty released test verse", "no available labeled test verses"}
                          if profile.get("extension_protocol") else {"empty answer/sentence", "identical minimal pair"})
    if data["full"]["sampling"] is not None or any(x["reason"] not in allowed_exclusions for x in data["full"]["exclusions"]):
        raise ValueError("evaluation requires all valid source questions")
    checked(data["full"]["exclusion_audit"])
    source_path = repo / profile["source_campaign"]
    source = runtime.load_campaign(source_path)
    models = select_models(source, profile["selection"])
    if previous:
        if previous["inputs"] != data["full"]["inputs"]:
            raise ValueError("reuse requires exactly the frozen pilot questions and token caches")
        smoke_input = previous["smoke_input"]
    else:
        smoke_limit = profile.get("smoke_languages_per_family")
        if smoke_limit:
            smoke_sources = []
            for benchmark in benchmarks:
                languages = [x for x in data["smoke"]["languages"] if x["benchmark"] == benchmark]
                ordered = sorted(languages, key=lambda x: (-max(map(int, x["width_histogram"])), x["language"]))
                selected = [x for x in ordered if x.get("corpus_language") == "eng_Latn"][:1]
                selected += [x for x in ordered if x not in selected][:smoke_limit-len(selected)]
                smoke_sources.extend(selected)
            smoke_fragments = [read(checked(x)) for x in smoke_sources]
            smoke_caches = [x["token_cache"] for x in smoke_sources]
        else:
            smoke_fragments = [read(checked(x["request_manifest"])) for x in data["smoke"]["inputs"].values()]
            smoke_caches = [x["token_cache"] for x in data["smoke"]["inputs"].values()]
        smoke_payload = dict(schema="stage1_lm_eval_frozen_requests/v1", protocol="stage1_candidate_likelihood_v1",
                             sampling={"purpose": "startup diagnostic only", "documents": "first/last/longest",
                                       "languages_per_family": smoke_limit,
                                       "language_selection": "English when available, then longest complete input widths"},
                             tasks=[t for f in smoke_fragments for t in f["tasks"]],
                             requests=[r for f in smoke_fragments for r in f["requests"]])
        smoke_manifest = save(output / "data/smoke/combined.json", smoke_payload, "manifest_sha256")
        smoke_cache = combine_token_caches(smoke_caches,
            output / "data/smoke/combined_token_cache", request_sha256=smoke_manifest["sha256"],
            tokenizer=data["smoke"]["tokenizer"], sequence_length=2048)
        smoke_input = dict(request_manifest=smoke_manifest, token_cache=smoke_cache,
                           documents=sum(len(t["documents"]) for t in smoke_payload["tasks"]), requests=len(smoke_payload["requests"]))
    prior_jobs = {job["id"]: job for job in previous["jobs"]} if previous else {}
    prior_submissions = {}
    if previous:
        for old_phase in ("smoke", previous.get("evaluation_phase", "pilot")):
            journal = read(previous_path.parent / f"submission_{old_phase}.json")
            prior_submissions.update({job["id"]: job for job in journal["jobs"]})
    jobs, cache = [], {}
    for model in sorted(models, key=lambda j: (j["ladder"], j["E"], j["K"], j["seed"])):
        source_spec = checked(model["spec"])
        original = read(source_spec)
        _verify_payload_hash(original, field="spec_sha256", label="original final score spec")
        for benchmark in (*benchmarks, "smoke"):
            if benchmark == "smoke" and (model["K"] != 200 or model["seed"] != 1234):
                continue
            family = smoke_input if benchmark == "smoke" else data["full"]["inputs"][benchmark]
            job_id = model["model_id"] + "__" + benchmark
            if job_id in prior_jobs:
                job = dict(prior_jobs[job_id], phase="smoke" if benchmark == "smoke" else phase,
                           reused=True, origin_campaign=ref(previous_path),
                           original_slurm_job_id=prior_submissions[job_id]["slurm_job_id"])
                directory = Path(read(checked(job["spec"]))["output_dir"])
                (output / "scores").mkdir(exist_ok=True)
                (output / "scores" / job_id).symlink_to(directory, target_is_directory=True)
                jobs.append(job)
                continue
            minutes = profile["slurm"]["smoke_minutes"] if benchmark == "smoke" else profile["slurm"]["minutes_by_ladder"][model["ladder"]]
            spec = json.loads(json.dumps(original))
            spec.pop("spec_sha256")
            spec["run_id"] = output.name + "_" + job_id
            spec["output_dir"] = str(output / "scores" / job_id)
            spec["request_manifest"], spec["token_cache"] = family["request_manifest"], family["token_cache"]
            spec["evaluation"].update(expected_world_size=1, required_gpu_name_substring="A40",
                micro_batch_size=profile["batches"][f"{model['ladder']}_E{model['E']}"], max_gpu_hours=minutes / 60)
            spec["evaluation"].pop("batching", None)
            spec.pop("full_suite_provenance", None)
            spec["candidate_provenance"] = dict(source_final_score_spec=ref(source_spec), source_campaign=ref(source_path),
                                                 profile=ref(profile_path), phase="smoke" if benchmark == "smoke" else phase)
            spec_ref = save(output / "specs" / (job_id + ".json"), spec, "spec_sha256")
            load_native_benchmark_score_spec(Path(spec_ref["path"]), request_manifest_cache=cache)
            job = {k: model[k] for k in ("model_id", "ladder", "E", "K", "seed")}
            job.update(id=job_id, benchmark=benchmark, phase="smoke" if benchmark == "smoke" else phase, gpus=1,
                       memory=profile["slurm"]["memory"], walltime_minutes=minutes, spec=spec_ref,
                       requests=family["requests"], documents=family["documents"])
            job["run_script"], job["sbatch_script"] = runtime.write_job(output, job, profile, repo, module=MODULE, job_prefix="s1-candidate-")
            jobs.append(job)
    campaign = dict(schema=SCHEMA, status="prepared_smoke_pending", evaluation_phase=phase,
                    repo=str(repo), profile=ref(profile_path),
                    profile_payload=profile, source_campaign=ref(source_path),
                    data_manifests={scope: ref(data_root / scope / "data_manifest.json") for scope in data},
                    inputs=data["full"]["inputs"], smoke_input=smoke_input, jobs=jobs,
                    source_manifest=runtime.freeze_source(repo, output, profile),
                    collection_job=runtime.write_collection_job(output, profile, module=MODULE, job_prefix="s1-candidate-"),
                    max_pilot_allocated_gpu_hours=sum(j["walltime_minutes"] / 60 for j in jobs
                                                      if j["phase"] == phase and not j.get("reused")),
                    protocol=profile["protocol"])
    if previous:
        campaign["reuse_campaign"] = ref(previous_path)
    if profile.get("extension_protocol"):
        from .candidate_queue import prepare_controls
        campaign["queue_controls"] = prepare_controls(output, profile)
    save(output / "campaign.json", campaign, "manifest_sha256")
    for command in ("validate", "submit", "collect"):
        script = output / (command + ".sh")
        arguments = ["--phase", "all"] if command == "submit" and profile.get("extension_protocol") else []
        script.write_text("\n".join(["#!/bin/bash", "set -euo pipefail",
            "export PYTHONPATH=" + shlex.quote(str(output / "source/src") + ":" + str(output / "source")),
            "exec " + shlex.join(["conda", "run", "--no-capture-output", "-p", str(Path(profile["python"]).parents[1]),
                                   "python", "-m", MODULE, command, "--campaign", str(output / "campaign.json"), *arguments]) + ' "$@"', ""]))
        subprocess.run(["bash", "-n", str(script)], check=True)
    return validate(campaign)


def validate(campaign):
    from multilingual_megatron.pipeline.stages.benchmarking.native_megatron_lm_eval import load_native_benchmark_score_spec
    for key in ("profile", "source_campaign", "source_manifest", "collection_job"):
        checked(campaign[key])
    models = select_models(runtime.load_campaign(Path(campaign["source_campaign"]["path"])),
                           campaign["profile_payload"]["selection"])
    evaluation, smoke = validate_matrix(campaign, models)
    models = {model["model_id"]: model for model in models}
    prior_jobs = {}
    if campaign.get("reuse_campaign"):
        previous = load(checked(campaign["reuse_campaign"]))
        prior_jobs = {job["id"]: job for job in previous["jobs"]}
        if previous["inputs"] != campaign["inputs"] or previous["smoke_input"] != campaign["smoke_input"]:
            raise ValueError("reuse data contract drift")
    for entry in read(checked(campaign["source_manifest"]))["files"]:
        checked(entry)
    for entry in campaign["data_manifests"].values():
        checked(entry)
    for family in list(campaign["inputs"].values()) + [campaign["smoke_input"]]:
        checked(family["request_manifest"])
        for entry in read(checked(family["token_cache"]))["files"].values():
            checked(entry)
    cache = {}
    for job in campaign["jobs"]:
        for key in ("spec", "run_script", "sbatch_script"):
            checked(job[key])
        payload = read(Path(job["spec"]["path"]))
        family = campaign["smoke_input"] if job["phase"] == "smoke" else campaign["inputs"][job["benchmark"]]
        if (payload["candidate_provenance"]["source_final_score_spec"] != models[job["model_id"]]["spec"]
                or payload["request_manifest"] != family["request_manifest"]
                or payload["token_cache"] != family["token_cache"]
                or (job["documents"], job["requests"]) != (family["documents"], family["requests"])):
            raise ValueError("candidate checkpoint/data binding mismatch")
        if job.get("reused"):
            original = prior_jobs[job["id"]]
            if any(job[key] != original[key] for key in ("spec", "run_script", "sbatch_script", "model_id", "benchmark")):
                raise ValueError("reused job differs from its original")
        spec = load_native_benchmark_score_spec(Path(job["spec"]["path"]), request_manifest_cache=cache)
        if spec.expected_world_size != 1 or spec.required_gpu_name_substring != "A40":
            raise ValueError("candidate campaign must use one A40 per job")
        if payload["evaluation"]["micro_batch_size"] != campaign["profile_payload"]["batches"][f"{job['ladder']}_E{job['E']}"]:
            raise ValueError("candidate batch contract drift")
    return dict(validated_evaluation_jobs=len(evaluation), validated_smoke_jobs=len(smoke),
                reused_evaluation_jobs=sum(bool(j.get("reused")) for j in evaluation),
                max_new_evaluation_allocated_gpu_hours=campaign["max_pilot_allocated_gpu_hours"])


def result_for(job):
    spec = read(checked(job["spec"]))
    path = Path(spec["output_dir"]) / "scores.json"
    if not path.exists():
        return None
    result = read(path)
    _verify_payload_hash(result, field="result_sha256", label="candidate scores")
    if (result["status"] != "complete" or result["score_spec"]["sha256"] != job["spec"]["sha256"]
            or result["request_manifest"]["sha256"] != spec["request_manifest"]["sha256"]
            or not result["parity_gate"]["passed"] or not result["replica_gate"]["passed"]
            or result["counts"]["truncated_requests"]):
        raise ValueError(f"invalid scoring receipt: {job['id']}")
    if job["E"] > 1 and (not result["moe_routing"]["evidence_available"] or result["moe_routing"]["totals"]["dropped_assignments"] != 0):
        raise ValueError("missing dropless MoE evidence")
    return result


def smoke_gate(campaign):
    manifest = read(checked(campaign["smoke_input"]["request_manifest"]))
    observations = []
    for job in campaign["jobs"]:
        if job["phase"] != "smoke":
            continue
        result = result_for(job)
        if result is None:
            raise ValueError(f"A40 smoke result missing: {job['id']}")
        # Full CPU reduction also catches data/score/token binding mistakes.
        reduce_scores(campaign, manifest, result["scores"])
        observations.append(dict(id=job["id"], timing=result["timing"], counts=result["counts"]))
    return dict(passed=True, architectures=len(observations), results=observations)


def estimate_budget(campaign, gate):
    """Weight measured A40 forward timings by every frozen request length."""
    def bucket(width):
        return 1 << (max(1, int(width)) - 1).bit_length()
    data = read(checked(campaign["data_manifests"]["full"]))
    histograms = {benchmark: Counter() for benchmark in benchmark_names(campaign)}
    for language in data["languages"]:
        for width, count in language["width_histogram"].items():
            histograms[language["benchmark"]][bucket(width)] += count
    jobs = {job["id"]: job for job in campaign["jobs"]}
    rates = {}
    for result in gate["results"]:
        job = jobs[result["id"]]
        bins = defaultdict(list)
        for batch in result["timing"]["forward_batches"][2:]:
            bins[bucket(batch["width"])].append(batch["seconds"] / batch["requests"])
        if not bins:
            raise ValueError("startup check has insufficient steady-state timing evidence")
        rates[job["ladder"], job["E"]] = {width: median(values) for width, values in bins.items()}
    projections = []
    for job in campaign["jobs"]:
        if job["phase"] != campaign.get("evaluation_phase", "pilot"):
            continue
        timing = rates[job["ladder"], job["E"]]
        seconds = 120.0
        for width, count in histograms[job["benchmark"]].items():
            nearest = min(timing, key=lambda value: abs(value.bit_length() - width.bit_length()))
            seconds += count * timing[nearest] * max(1.0, width / nearest)
        projections.append(dict(id=job["id"], reused=bool(job.get("reused")), central_seconds=seconds,
                                low_seconds=max(30.0, 0.7 * seconds), high_seconds=1.5 * seconds + 180))
    new = [job for job in projections if not job["reused"]]
    return dict(method="A40 startup forward medians per request by power-of-two width; discard first two batches; "
                       "complete request-length histograms; add 120 seconds startup/job; range 0.7x to 1.5x plus 3 minutes. "
                       "Extrapolation, not a confidence interval or guaranteed wall-clock ETA.",
                new_jobs=len(new), reused_jobs=len(projections)-len(new), jobs=projections,
                new_gpu_hours={key: sum(job[key + "_seconds"] for job in new) / 3600 for key in ("low", "central", "high")})


def submit(path, campaign, phase):
    if phase == "all":
        from .candidate_queue import submit_all
        return submit_all(path, campaign)
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit only from the login host")
    validate(campaign)
    evaluation_phase = campaign.get("evaluation_phase", "pilot")
    if phase not in {"smoke", evaluation_phase}:
        raise ValueError("submission phase does not match campaign")
    estimates = {}
    if phase == evaluation_phase:
        gate = smoke_gate(campaign)
        save(path.parent / "a40_smoke_gate.json", gate)
        budget = estimate_budget(campaign, gate)
        save(path.parent / "a40_budget.json", dict(campaign=ref(path), **budget), "manifest_sha256")
        estimates = {job["id"]: job["central_seconds"] for job in budget["jobs"]}
    journal = path.parent / f"submission_{phase}.json"
    slurm = Path(campaign["profile_payload"]["slurm"]["bin"])
    with (path.parent / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if journal.exists():
            raise ValueError("phase already has a submission journal; inspect before retry")
        records = []
        # Longest measured projections enter first without priority manipulation.
        jobs = sorted((j for j in campaign["jobs"] if j["phase"] == phase),
                      key=lambda j: (-estimates.get(j["id"], j["walltime_minutes"] * 60),
                                     -j["requests"], -j["E"], j["id"]))
        reused = [job for job in jobs if job.get("reused")]
        for job in reused:
            if result_for(job) is None:
                raise ValueError(f"reused result not yet complete: {job['id']}")
            records.append(dict(id=job["id"], slurm_job_id=job["original_slurm_job_id"],
                                reused=True, origin_campaign=job["origin_campaign"]))
        _atomic_json(journal, dict(status="submitting", phase=phase, jobs=records))
        for job in jobs:
            if job.get("reused"):
                continue
            command = [str(slurm / "sbatch"), "--parsable", job["sbatch_script"]["path"]]
            jid = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not jid.isdecimal():
                raise ValueError("invalid Slurm submission response")
            records.append(dict(id=job["id"], slurm_job_id=jid, command=command))
            _atomic_json(journal, dict(status="submitting", jobs=records))
        collection = None
        if phase == evaluation_phase:
            queued = set(subprocess.check_output([str(slurm / "squeue"), "--me", "--noheader", "--format=%A"], text=True).split())
            pending = [r["slurm_job_id"] for r in records if r["slurm_job_id"] in queued]
            command = [str(slurm / "sbatch"), "--parsable"]
            if pending:
                command.append("--dependency=afterany:" + ":".join(pending))
            command.append(campaign["collection_job"]["path"])
            jid = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not jid.isdecimal():
                raise ValueError("invalid collection submission response")
            collection = dict(slurm_job_id=jid, command=command)
        _atomic_json(journal, dict(status="submitted", phase=phase, jobs=records, collection_job=collection))
    return dict(phase=phase, submitted_jobs=len(records)-len(reused), reused_jobs=len(reused),
                job_ids=[r["slurm_job_id"] for r in records if not r.get("reused")], collection=collection)


def run(campaign, job_id):
    job = next(j for j in campaign["jobs"] if j["id"] == job_id)
    if job.get("reused"):
        raise ValueError("reused results must never be relaunched")
    directory = Path(read(checked(job["spec"]))["output_dir"])
    if campaign["profile_payload"].get("extension_protocol") and job["phase"] != "smoke":
        root = directory.parent.parent
        gate = read(root / "a40_smoke_gate.json")
        _verify_payload_hash(gate, field="manifest_sha256", label="extension startup gate")
        if not gate.get("passed") or gate.get("campaign") != ref(root / "campaign.json"):
            raise ValueError("complete startup gate is required for this exact campaign")
    directory.mkdir(parents=True, exist_ok=True)
    telemetry = None
    with (directory / "gpu_telemetry.csv").open("wb") as output, (directory / "gpu_telemetry.stderr").open("wb") as errors:
        try:
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw",
                                         "--format=csv", "-l", "5"], stdout=output, stderr=errors)
            runtime.run(campaign, job_id)
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)


def cohort_summary(identity, benchmark, cohort, values, gpu_hours):
    selected = [v for v in values if v["group"] == "all" and (cohort in {"all_supported_K200", "all_available"}
                or (v.get("in_K200", False) if cohort == "study_K200" else v["in_common12"] if cohort == "common12" else v["trained_at_K"]))]
    primary = next(v["primary_metric"] for v in values if v["group"] == "all")
    # Regional references share one language weight in the study cohorts.
    groups = defaultdict(list)
    for index, value in enumerate(selected):
        key = value.get("language", str(index)) if cohort in {"all_available", "all_supported_K200"} else value.get("corpus_language", value.get("language", str(index)))
        groups[key].append(value)
    def average(key):
        means = [sum(v[key] for v in group if v.get(key) is not None)/sum(v.get(key) is not None for v in group)
                 for group in groups.values() if any(v.get(key) is not None for v in group)]
        return sum(means)/len(means) if means else None
    result = dict(**identity, benchmark=benchmark, cohort=cohort,
                cohort_status="available" if selected else "not_applicable_no_languages",
                languages=len(groups), language_variants=len(selected), samples=sum(v["samples"] for v in selected), primary_metric=primary,
                language_macro_primary=average(primary),
                language_macro_accuracy=average(primary) if primary.startswith("acc") else None,
                language_macro_margin=average("margin"),
                gpu_hours=gpu_hours)
    for key in ("raw_nll", "bits_per_byte", "unigram_adjusted_loss", "macro_f1", "weighted_f1", "majority_baseline"):
        if any(key in v for v in values):
            result["language_macro_"+key] = average(key)
            result[key+"_languages"] = sum(any(v.get(key) is not None for v in group) for group in groups.values())
    return result


def collect(path, campaign):
    rows, summaries, missing = [], [], []
    current, manifest = None, None
    audit = Path(campaign["repo"]) / campaign["profile_payload"]["coverage_audit"]
    training = read(audit / "training_languages.json")
    jobs = [j for j in campaign["jobs"] if j["phase"] == campaign.get("evaluation_phase", "pilot")]
    for job in sorted(jobs, key=lambda j: (j["benchmark"], j["id"])):
        result = result_for(job)
        if result is None:
            missing.append(job["id"])
            continue
        if current != job["benchmark"]:
            current = job["benchmark"]
            manifest = read(checked(campaign["inputs"][current]["request_manifest"]))
        values = reduce_scores(campaign, manifest, result["scores"])
        identity = {k: job[k] for k in ("model_id", "ladder", "E", "K", "seed")}
        for value in values:
            value.update(in_common12=value["corpus_language"] in training["12"],
                         in_K200=value["corpus_language"] in training["200"],
                         trained_at_K=value["corpus_language"] in training[str(job["K"])])
        rows.extend(dict(**identity, **value) for value in values)
        for cohort in campaign["profile_payload"].get("reporting_cohorts", ("all_supported_K200", "common12", "trained_at_K")):
            summaries.append(cohort_summary(identity, job["benchmark"], cohort, values, result["timing"]["gpu_hours"]))
    payload = dict(status="partial" if missing else "complete", campaign=ref(path),
                   metrics=rows, benchmark_summary=summaries, incomplete_jobs=missing,
                   interpretation="Complete labeled questions; fixed common12 cohort for K comparisons; report seed scope per checkpoint")
    save(path.parent / "metrics.json", payload, "manifest_sha256")
    for name, records in (("language_metrics.csv", [r for r in rows if r["group"] == "all"]),
                          ("category_metrics.csv", [r for r in rows if r["group"] != "all"]),
                          ("benchmark_metrics.csv", summaries)):
        with (path.parent / name).open("w") as handle:
            if records:
                writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in records))))
                writer.writeheader()
                writer.writerows(records)
    return dict(status=payload["status"], completed_jobs=len(jobs)-len(missing), missing_jobs=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--profile", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("validate", "submit", "run", "collect", "smoke-gate", "queue-gate"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name == "submit":
            command.add_argument("--phase", choices=("smoke", "pilot", "full", "all"), required=True)
        if name == "run":
            command.add_argument("--job", required=True)
        if name == "collect":
            command.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.profile, args.output)
    else:
        campaign = load(args.campaign)
        if args.command == "validate":
            result = validate(campaign)
        elif args.command == "submit":
            result = submit(args.campaign, campaign, args.phase)
        elif args.command == "run":
            result = run(campaign, args.job)
        elif args.command == "smoke-gate":
            result = smoke_gate(campaign)
        elif args.command == "queue-gate":
            from .candidate_queue import finish_gate
            result = finish_gate(args.campaign, campaign)
        else:
            result = collect(args.campaign, campaign)
    print(json.dumps(result, indent=2))
    if args.command == "collect" and args.require_complete and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
