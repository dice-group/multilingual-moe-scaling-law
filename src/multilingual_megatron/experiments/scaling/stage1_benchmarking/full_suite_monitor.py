"""Observe a submitted full-suite or candidate pilot without changing its jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from multilingual_megatron.contracts.hashing import canonical_json_sha256

FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}
TERMINAL = FAILED | {"COMPLETED"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_payload(path):
    payload = read(path)
    expected = payload.pop("manifest_sha256")
    if canonical_json_sha256(payload) != expected:
        raise ValueError(f"payload hash mismatch: {path}")
    return payload


def tail(path, limit=32768):
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


def collection_has_coverage(metrics, jobs, inputs, *, candidate=False, cohorts=None):
    """Require distinct model/task/language cells and every planned summary."""
    if metrics["status"] != "complete" or metrics["incomplete_jobs"]:
        return False
    pairs = {(job["model_id"], job["benchmark"]) for job in jobs}
    cohorts = cohorts or (("all_supported_K200", "common12", "trained_at_K") if candidate else (None,))
    expected = {(model, benchmark, cohort) for model, benchmark in pairs for cohort in cohorts}
    summaries = metrics["benchmark_summary"]
    observed = {(row["model_id"], row["benchmark"], row.get("cohort")) for row in summaries}
    if observed != expected or len(summaries) != len(expected):
        return False
    rows = [row for row in metrics["metrics"] if not candidate or row["group"] == "all"]
    cells = {(row["model_id"], row["benchmark"], row["language"]) for row in rows}
    counts = Counter((model, benchmark) for model, benchmark, _ in cells)
    if len(cells) != len(rows) or set(counts) != pairs:
        return False
    if any(counts[model, benchmark] != inputs[benchmark]["languages"] for model, benchmark in pairs):
        return False
    for row in summaries:
        if row.get("cohort") in (None, "all_supported_K200", "all_available"):
            family = inputs[row["benchmark"]]
            if row["languages"] != family["languages"] or row["samples"] != family["documents"]:
                return False
    return True


def inspect(campaign_path, campaign):
    """Prefer live states; retain accounting evidence for jobs that left squeue."""
    root = campaign_path.parent
    candidate = campaign.get("schema") == "stage1_candidate_campaign/v1"
    phase = campaign.get("evaluation_phase", "pilot")
    jobs = [job for job in campaign["jobs"] if not candidate or job["phase"] == phase]
    expected_ids = {job["id"] for job in jobs}
    submission = read(root / (f"submission_{phase}.json" if candidate else "submission.json"))
    records = {row["slurm_job_id"]: row["id"] for row in submission["jobs"]}
    reused_ids = {row["slurm_job_id"] for row in submission["jobs"] if row.get("reused")}
    collector = (submission.get("collection_job") or {}).get("slurm_job_id")
    all_ids = list(records) + ([collector] if collector else [])
    binary = Path(campaign["profile_payload"]["slurm"]["bin"])
    commands = {
        "accounting": [str(binary / "sacct"), "--allocations", "--noheader", "--parsable2",
                       "--jobs=" + ",".join(all_ids), "--format=JobIDRaw,State%30,ElapsedRaw,AllocTRES,ExitCode"],
        "queue": [str(binary / "squeue"), "--me", "--noheader", "--format=%A|%T|%M|%R"],
    }
    raw = {key: subprocess.check_output(command, text=True, timeout=45) for key, command in commands.items()}
    states, accounting, queue = {}, {}, {}
    for line in raw["accounting"].splitlines():
        fields = line.split("|")
        if fields[0] in all_ids and len(fields) >= 5:
            states[fields[0]] = fields[1].split()[0].rstrip("+")
            accounting[fields[0]] = {"state": fields[1], "elapsed_seconds": int(fields[2]),
                                     "allocated_resources": fields[3], "exit_code": fields[4]}
    for line in raw["queue"].splitlines():
        fields = line.split("|")
        if fields[0] in all_ids and len(fields) >= 4:
            states[fields[0]] = fields[1]
            queue[fields[0]] = {"state": fields[1], "elapsed": fields[2], "node_or_reason": fields[3]}
    failures, progress = [], []
    if len(records) != len(jobs) or set(records.values()) != expected_ids:
        failures.append({"job_id": "submission", "reason": "submitted jobs differ from the planned evaluation matrix"})
    results_present = 0
    for slurm_id, job_id in records.items():
        state = states.get(slurm_id, "UNKNOWN")
        output = root / "scores" / job_id
        has_scores = (output / "scores.json").is_file()
        results_present += has_scores
        if state in FAILED or (state == "COMPLETED" and not has_scores):
            failures.append({"slurm_job_id": slurm_id, "job_id": job_id, "state": state,
                             "reason": "missing scores" if state == "COMPLETED" else state,
                             "native_log_tail": tail(output / "native_benchmark.log"),
                             "slurm_log_tail": tail(root / "jobs" / job_id / f"slurm-{slurm_id}.log")})
        elif state in {"RUNNING", "COMPLETING"}:
            text = tail(output / "native_benchmark.log")
            markers = [line for line in text.splitlines() if "[native-benchmark] batch " in line]
            progress.append({"slurm_job_id": slurm_id, "job_id": job_id,
                             "latest_batch": markers[-1] if markers else None, **queue.get(slurm_id, {})})
    collection_state = states.get(collector, "NOT_SUBMITTED" if not collector else "UNKNOWN")
    if collection_state in FAILED:
        failures.append({"slurm_job_id": collector, "job_id": "collection", "state": collection_state,
                         "slurm_log_tail": tail(root / f"collect-{collector}.log")})
    complete = False
    if collection_state == "COMPLETED":
        try:
            metrics = checked_payload(root / "metrics.json")
            complete = (not failures
                        and all(states.get(job) == "COMPLETED" for job in records)
                        and collection_has_coverage(metrics, jobs, campaign["inputs"], candidate=candidate,
                            cohorts=campaign["profile_payload"].get("reporting_cohorts")))
            if not complete:
                failures.append({"job_id": "collection", "reason": "final coverage or completion check failed"})
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"job_id": "collection", "reason": str(error)})
    terminal = bool(collector) and all(
        states.get(job) in TERMINAL for job in all_ids)
    status = "complete" if complete else "finished_with_failures" if terminal else "attention_required" if failures else "monitoring"
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": status,
               "submitted_gpu_jobs": len(records)-len(reused_ids), "reused_gpu_jobs": len(reused_ids),
               "evaluation_jobs": len(jobs), "gpu_states": dict(Counter(states.get(job, "UNKNOWN") for job in records)),
               "pending_reasons": dict(Counter(row["node_or_reason"] for job, row in queue.items()
                                               if job in records and row["state"] == "PENDING")),
               "score_files_present": results_present, "collection_job": collector, "collection_state": collection_state,
               "failure_count": len(failures), "progress": progress, "accounting": accounting,
               "observed_gpu_hours": sum(row["elapsed_seconds"] for job, row in accounting.items() if job in records) / 3600,
               "observed_new_gpu_hours": sum(row["elapsed_seconds"] for job, row in accounting.items()
                                             if job in records and job not in reused_ids) / 3600,
               "monitor_pid": os.getpid(), "campaign_path": str(campaign_path)}
    monitoring = root / "monitoring"
    write(monitoring / "status.json", summary)
    write(monitoring / "failures.json", {"timestamp_utc": summary["timestamp_utc"], "failures": failures})
    for key, contents in raw.items():
        (monitoring / (key + ".txt")).write_text(contents)
    brief = {key: value for key, value in summary.items() if key not in {"progress", "accounting"}}
    with (monitoring / "history.jsonl").open("a") as handle:
        handle.write(json.dumps(brief, sort_keys=True) + "\n")
    print(json.dumps(brief, sort_keys=True), flush=True)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 30:
        parser.error("use at least 30 seconds between scheduler queries")
    campaign_path = args.campaign.resolve()
    campaign = checked_payload(campaign_path)
    monitoring = campaign_path.parent / "monitoring"
    monitoring.mkdir(exist_ok=True)
    with (monitoring / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                finished = inspect(campaign_path, campaign)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                # An access/query failure is not evidence that any job stopped.
                write(monitoring / "query_error.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                                        "error": str(error), "monitor_pid": os.getpid()})
                print(f"monitor query failed: {error}", flush=True)
                if not args.watch:
                    raise
                finished = False
            if finished or not args.watch:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
