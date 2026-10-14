#!/usr/bin/env python3
"""Executor abstraction for pipeline command execution.

Provides a protocol for running shell commands either locally (subprocess)
or via Slurm (sbatch + polling).  Stages never know which backend is active —
they just call ``executor.run()`` or ``executor.run_batch()`` with command lists.
"""

from __future__ import annotations

import datetime as dt
import errno
import logging
import os
import re
import signal
import subprocess
import textwrap
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

LOGGER = logging.getLogger("proof_pipeline")


def _write_private_text(path: Path, text: str) -> None:
    """Atomically write a file that may contain submitted environment secrets."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Executor(Protocol):
    """
        Minimal contract every executor backend must satisfy
        We use this because the pipline should be usable within different environments e.g. H100 Cluster and GPU VM
    """

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> None:
        """Execute *cmd* and stream output to *log_path*."""
        ...

    def run_batch(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> None:
        """Execute multiple independent jobs.

        Each element is ``(cmd, env, log_path)``.
        ``LocalExecutor`` runs them sequentially; ``SlurmExecutor`` submits
        all as separate sbatch jobs and waits for all to finish.
        """
        ...


# ---------------------------------------------------------------------------
# Local executor (current behaviour, subprocess.Popen)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LocalExecutor:
    """Run commands as local subprocesses — the default."""

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> None:
        started_at = _utc_stamp()
        started_perf = time.perf_counter()
        LOGGER.info("[CMD] %s", " ".join(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[START_UTC] {started_at}\n")
                fh.write(f"[DRY_RUN CMD] {' '.join(cmd)}\n")
                fh.write("[END] dry_run=true exit_code=0 duration_seconds=0.000\n")
            return

        proc: subprocess.Popen[str] | None = None
        got_signal = False
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        previous_sigusr1_handler = signal.getsignal(signal.SIGUSR1)

        def _forward_signal(signum: int, frame: object) -> None:  # pragma: no cover (signal timing)
            nonlocal got_signal, proc
            got_signal = True
            sig_name = (
                "SIGTERM"
                if signum == signal.SIGTERM
                else ("SIGUSR1" if signum == signal.SIGUSR1 else str(signum))
            )
            if proc is None:
                return
            # Cluster uses SIGUSR1 as a time-limit warning; treat it as SIGTERM when forwarding.
            forward_signum = signal.SIGTERM if signum == signal.SIGUSR1 else signum
            if forward_signum != signum:
                LOGGER.warning("[SIGNAL] Received %s; forwarding SIGTERM to child process.", sig_name)
            else:
                LOGGER.warning("[SIGNAL] Received %s; forwarding to child process.", sig_name)
            try:
                os.killpg(proc.pid, forward_signum)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[START_UTC] {started_at}\n")
            fh.write(f"\n[CMD] {' '.join(cmd)}\n")
            signal.signal(signal.SIGTERM, _forward_signal)
            signal.signal(signal.SIGUSR1, _forward_signal)
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                assert proc.stdout is not None
                while True:
                    try:
                        line = proc.stdout.readline()
                    except OSError as exc:
                        if exc.errno == errno.EINTR:
                            continue
                        raise
                    if line:
                        LOGGER.info("%s", line.rstrip("\n"))
                        fh.write(line)
                        continue
                    if proc.poll() is not None:
                        break
                code = proc.wait()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)
                signal.signal(signal.SIGUSR1, previous_sigusr1_handler)
            duration = time.perf_counter() - started_perf
            fh.write(f"[END] dry_run=false exit_code={code} duration_seconds={duration:.3f}\n")
            if code != 0:
                raise subprocess.CalledProcessError(code, cmd)
            if got_signal:
                LOGGER.warning("[SIGNAL] Signal handled; child exited cleanly.")

    def run_batch(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> None:
        """Run jobs sequentially on a local machine."""
        for cmd, env, log_path in jobs:
            self.run(cmd, env, dry_run, log_path)


# ---------------------------------------------------------------------------
# Slurm executor (sbatch + poll)
# ---------------------------------------------------------------------------

# Portable safety defaults. Site modules, environments, accelerators, memory,
# and partitions belong in named runtime profiles.
_SLURM_DEFAULTS: dict[str, Any] = {
    "partition": "",
    "gres": "",
    "mem": "",
    "time": "01:00:00",
    "cpus_per_task": 1,
    "module_init": "",
    "modules": [],
    "conda_base": "",
    "conda_env": "",
    "poll_interval_seconds": 30,
    "extra_sbatch_flags": [],
}


@dataclass(slots=True)
class SlurmExecutor:
    """Submit commands via ``sbatch`` and poll ``sacct`` until they finish."""

    partition: str = _SLURM_DEFAULTS["partition"]
    gres: str = _SLURM_DEFAULTS["gres"]
    mem: str = _SLURM_DEFAULTS["mem"]
    time_limit: str = _SLURM_DEFAULTS["time"]
    cpus_per_task: int = _SLURM_DEFAULTS["cpus_per_task"]
    module_init: str = _SLURM_DEFAULTS["module_init"]
    modules: list[str] = field(default_factory=lambda: list(_SLURM_DEFAULTS["modules"]))
    conda_base: str = _SLURM_DEFAULTS["conda_base"]
    conda_env: str = _SLURM_DEFAULTS["conda_env"]
    poll_interval: int = _SLURM_DEFAULTS["poll_interval_seconds"]
    extra_sbatch_flags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> None:
        job_ids = self._submit_jobs([(cmd, env, log_path)], dry_run)
        if job_ids:
            self._wait_for_jobs(job_ids, dry_run)

    def run_batch(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> None:
        """Submit all jobs via sbatch, then wait for all to finish."""
        job_ids = self._submit_jobs(jobs, dry_run)
        if job_ids:
            self._wait_for_jobs(job_ids, dry_run)

    # ------------------------------------------------------------------
    # sbatch submission
    # ------------------------------------------------------------------

    def _submit_jobs(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> list[tuple[str, Path]]:
        """Submit each job and return list of (job_id, log_path) tuples."""
        submitted: list[tuple[str, Path]] = []
        for cmd, env, log_path in jobs:
            job_id = self._sbatch_submit(cmd, env, dry_run, log_path)
            if job_id:
                submitted.append((job_id, log_path))
        return submitted

    def submit(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
        *,
        dependency: str | None = None,
        job_name: str | None = None,
        script_dir: Path | None = None,
        slurm_output: str | Path | None = None,
    ) -> str | None:
        """Submit a single job, optionally with a dependency (e.g. afterok:<jobid>)."""
        return self._sbatch_submit(
            cmd,
            env,
            dry_run,
            log_path,
            dependency=dependency,
            job_name=job_name,
            script_dir=script_dir,
            slurm_output=slurm_output,
        )

    def submit_array(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
        *,
        array: str,
        dependency: str | None = None,
        slurm_output: str | Path | None = None,
        job_name: str | None = None,
        script_dir: Path | None = None,
    ) -> str | None:
        """Submit a SLURM array job. Array should be a spec like '0-31'."""
        return self._sbatch_submit(
            cmd,
            env,
            dry_run,
            log_path,
            dependency=dependency,
            array=array,
            slurm_output=slurm_output,
            job_name=job_name,
            script_dir=script_dir,
        )

    def wait_for_jobs(self, jobs: list[tuple[str, Path]], dry_run: bool = False) -> None:
        """Wait for jobs to finish, streaming Slurm logs into the provided paths."""
        self._wait_for_jobs(jobs, dry_run)

    def _sbatch_submit(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
        dependency: str | None = None,
        array: str | None = None,
        slurm_output: str | Path | None = None,
        job_name: str | None = None,
        script_dir: Path | None = None,
    ) -> str | None:
        """Write a job script and submit it.  Returns the Slurm job ID."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = _utc_stamp()

        if dry_run:
            LOGGER.info("[DRY_RUN SBATCH] %s", " ".join(cmd))
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[START_UTC] {started_at}\n")
                fh.write(f"[DRY_RUN SBATCH] {' '.join(cmd)}\n")
                fh.write("[END] dry_run=true exit_code=0 duration_seconds=0.000\n")
            return None

        job_name = _sanitize_job_name(job_name) if job_name else _job_name_from_log(log_path)
        if slurm_output is None:
            slurm_log = str(log_path.with_suffix(".slurm.log"))
        else:
            slurm_log = str(slurm_output)
        script = self._build_job_script(cmd, env, job_name, slurm_log)

        # Write the job script to a temp file next to the log.
        script_root = script_dir if script_dir is not None else log_path.parent
        script_root.mkdir(parents=True, exist_ok=True)
        script_path = script_root / log_path.with_suffix(".sbatch.sh").name
        _write_private_text(script_path, script)

        LOGGER.info("[SBATCH] Submitting %s", script_path)
        try:
            submit_cmd = ["sbatch", "--parsable"]
            if array:
                submit_cmd.append(f"--array={array}")
            if dependency:
                submit_cmd.append(f"--dependency={dependency}")
            submit_cmd.append(str(script_path))
            result = subprocess.run(
                submit_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            err_text = stderr or stdout or "sbatch failed without stdout/stderr."
            LOGGER.error("[SBATCH] Submission failed for %s: %s", script_path, err_text)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[START_UTC] {started_at}\n")
                fh.write(f"[SBATCH_ERROR] script={script_path}\n")
                if stdout:
                    fh.write(f"[SBATCH_STDOUT] {stdout}\n")
                if stderr:
                    fh.write(f"[SBATCH_STDERR] {stderr}\n")
                fh.write(f"[CMD] {' '.join(cmd)}\n")
            raise
        job_id = result.stdout.strip().split(";")[0]
        LOGGER.info("[SBATCH] Submitted job %s for %s", job_id, job_name)

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[START_UTC] {started_at}\n")
            fh.write(f"[SBATCH] job_id={job_id} script={script_path}\n")
            fh.write(f"[CMD] {' '.join(cmd)}\n")

        return job_id

    def _build_job_script(
        self,
        cmd: list[str],
        env: dict[str, str],
        job_name: str,
        slurm_log: str,
    ) -> str:
        """Produce a complete sbatch shell script."""
        extra_sbatch_flags = list(self.extra_sbatch_flags)
        account = (
            os.environ.get("PROOF_PIPELINE_SLURM_ACCOUNT")
            or os.environ.get("SLURM_ACCOUNT")
            or ""
        ).strip()
        qos = (
            os.environ.get("PROOF_PIPELINE_SLURM_QOS")
            or os.environ.get("SLURM_QOS")
            or ""
        ).strip()
        has_account_flag = any(flag.strip().startswith("--account") for flag in extra_sbatch_flags)
        has_qos_flag = any(flag.strip().startswith("--qos") for flag in extra_sbatch_flags)
        if account and not has_account_flag:
            extra_sbatch_flags.append(f"--account={account}")
        if qos and not has_qos_flag:
            extra_sbatch_flags.append(f"--qos={qos}")
        repo_root = str(env.get("PROOF_PIPELINE_REPO_ROOT", "")).strip()

        sbatch_lines = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --output={slurm_log}",
        ]
        partition_value = str(self.partition).strip()
        if partition_value:
            sbatch_lines.insert(1, f"#SBATCH --partition={partition_value}")
        mem_value = str(self.mem).strip()
        if mem_value and mem_value.lower() not in {"0", "none"}:
            sbatch_lines.insert(2, f"#SBATCH --mem={mem_value}")
        gres_value = str(self.gres).strip()
        if gres_value and gres_value.lower() not in {"none", "cpu"}:
            sbatch_lines.insert(2, f"#SBATCH --gres={gres_value}")
        if repo_root:
            sbatch_lines.append(f"#SBATCH --chdir={repo_root}")
        for flag in extra_sbatch_flags:
            sbatch_lines.append(f"#SBATCH {flag}")

        module_lines = ""
        if self.modules:
            module_init = str(self.module_init).strip()
            setup_lines = []
            if module_init:
                setup_lines.extend(
                    [
                    f'MODULE_INIT="{module_init}"',
                    'if [[ ! -r "${MODULE_INIT}" ]]; then',
                    '  echo "[SLURM_ERROR] unreadable Lmod init: ${MODULE_INIT}" >&2',
                    "  exit 1",
                    "fi",
                    "set +u",
                    'source "${MODULE_INIT}"',
                    "set -u",
                    'if ! type module >/dev/null 2>&1; then',
                    '  echo "[SLURM_ERROR] module command unavailable after ${MODULE_INIT}" >&2',
                    "  exit 1",
                    "fi",
                    'if [[ -z "${MODULEPATH:-}" ]]; then',
                    '  echo "[SLURM_ERROR] MODULEPATH unavailable after ${MODULE_INIT}" >&2',
                    "  exit 1",
                    "fi",
                    ]
                )
            setup_lines.extend(
                ["module purge", "module load " + " ".join(self.modules)]
            )
            module_lines = "\n".join(setup_lines)

        # Export key env vars the pipeline needs.
        export_keys = {
            "TMPDIR",
            "TMP_DIR",
            "TMP",
            "TEMP",
            "OMP_NUM_THREADS",
            "LD_LIBRARY_PATH",
            "PATH",
            "PYTHON_EXEC",
        }
        export_prefixes = (
            "PYTHONPATH",
            "PROOF_",
            "WANDB_",
            "CUDA_",
            "HF_",
            "BRIDGE_",
            "FI_",
            "STRAGGLER_",
            "NCCL_",
            "DEEP_EP_",
            "NVSHMEM_",
            "NVTE_",
            "TOKENIZERS_",
            "TRANSFORMERS_",
            "PYTORCH_",
        )
        env_exports = []
        for key in sorted(env):
            # Only export vars the pipeline explicitly sets and Bridge/runtime
            # knobs that must survive the sbatch boundary.
            if key in export_keys or key.startswith(export_prefixes):
                if key == "LD_LIBRARY_PATH":
                    append_ld = str(env.get("PROOF_APPEND_LD_LIBRARY_PATH", "1")).strip().lower()
                    if append_ld in {"0", "false", "no"}:
                        env_exports.append(f'export {key}="{env[key]}"')
                    else:
                        env_exports.append(f'export {key}="{env[key]}:${{{key}:-}}"')
                else:
                    env_exports.append(f'export {key}="{env[key]}"')

        cmd_str = " ".join(_shell_quote(c) for c in cmd)
        # Wrap in a new session so signal forwarding can target the whole subtree reliably.
        cmd_str_for_bash = cmd_str.replace("'", "'\"'\"'")
        script_lines: list[str] = ["#!/bin/bash", *sbatch_lines, "", "set -euo pipefail", ""]
        if "--export=NIL" in extra_sbatch_flags:
            # A clean Slurm export intentionally removes the submit shell's
            # PATH.  Establish only the system bootstrap tools here; Lmod and
            # conda prepend their own exact toolchain paths below.
            script_lines.extend(
                ['export PATH="/usr/local/bin:/usr/bin:/bin"', ""]
            )

        if module_lines:
            script_lines.extend(module_lines.splitlines())
            script_lines.append("")

        conda_base = str(self.conda_base).strip()
        conda_env = str(self.conda_env).strip()
        if conda_base and conda_env and conda_base.lower() != "none" and conda_env.lower() != "none":
            script_lines.extend(
                [
                    "set +u",
                    f'CONDA_BASE="{conda_base}"',
                    f'CONDA_ENV="{conda_env}"',
                    'if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then',
                    '  source "${CONDA_BASE}/etc/profile.d/conda.sh"',
                    '  if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then',
                    '    conda activate "${CONDA_ENV}"',
                    "  fi",
                    "else",
                    '  echo "[SLURM_ERROR] conda.sh not found: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2',
                    "  exit 1",
                    "fi",
                    "set -u",
                ]
            )
        else:
            script_lines.append('echo "[SLURM_INFO] conda disabled for this job script."')
        if env_exports:
            script_lines.extend(env_exports)
        if (
            conda_base
            and conda_env
            and conda_base.lower() != "none"
            and conda_env.lower() != "none"
            and "LD_LIBRARY_PATH" not in env
        ):
            script_lines.extend(
                [
                    'if [[ -n "${CONDA_PREFIX:-}" ]]; then',
                    '  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"',
                    "fi",
                ]
            )
        script_lines.extend(
            [
                'if [[ -n "${PROOF_EXPECTED_PYTHON_EXECUTABLE:-}" ]]; then',
                '  RESOLVED_PYTHON="$(command -v python)"',
                '  if [[ "$(readlink -f "${RESOLVED_PYTHON}")" != "$(readlink -f "${PROOF_EXPECTED_PYTHON_EXECUTABLE}")" ]]; then',
                '    echo "[SLURM_ERROR] wrong Python: ${RESOLVED_PYTHON}; expected ${PROOF_EXPECTED_PYTHON_EXECUTABLE}" >&2',
                "    exit 1",
                "  fi",
                "fi",
            ]
        )
        if repo_root:
            script_lines.extend(
                [
                    'cd "${PROOF_PIPELINE_REPO_ROOT}"',
                    'echo "[SLURM_INFO] cwd=$(pwd)"',
                ]
            )
        script_lines.extend(
            [
                "",
                "export PYTHONUNBUFFERED=1",
                "",
                'echo "[SLURM_START] $(date -u +%Y-%m-%dT%H:%M:%SZ) job=$SLURM_JOB_ID"',
                "EXIT_CODE=0",
                "CHILD_PID=",
                "CHILD_PGID=",
                "GPU_MONITOR_PID=",
                "GPU_MONITOR_CSV=",
                "start_gpu_monitor() {",
                '  local interval="${PROOF_GPU_MONITOR_INTERVAL_SECONDS:-0}"',
                '  if [[ "${interval}" == "0" || "${interval}" == "" ]]; then',
                "    return",
                "  fi",
                "  if ! command -v nvidia-smi >/dev/null 2>&1; then",
                '    echo "[GPU_MONITOR] nvidia-smi not found; skipping"',
                "    return",
                "  fi",
                "  if ! nvidia-smi -L >/dev/null 2>&1; then",
                '    echo "[GPU_MONITOR] no visible NVIDIA GPU; skipping"',
                "    return",
                "  fi",
                f'  GPU_MONITOR_CSV="{Path(slurm_log).with_suffix(".gpu.csv")}"',
                '  export PROOF_GPU_MONITOR_CSV="${GPU_MONITOR_CSV}"',
                '  echo "timestamp,index,uuid,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w" > "${GPU_MONITOR_CSV}"',
                "  (",
                "    while true; do",
                "      nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits >> \"${GPU_MONITOR_CSV}\" 2>/dev/null || true",
                "      sleep \"${interval}\"",
                "    done",
                "  ) &",
                "  GPU_MONITOR_PID=$!",
                '  echo "[GPU_MONITOR] csv=${GPU_MONITOR_CSV} interval=${interval}s pid=${GPU_MONITOR_PID}"',
                "}",
                "stop_gpu_monitor() {",
                '  if [[ -n "${GPU_MONITOR_PID:-}" ]]; then',
                '    kill "${GPU_MONITOR_PID}" 2>/dev/null || true',
                '    wait "${GPU_MONITOR_PID}" 2>/dev/null || true',
                "  fi",
                "}",
                "collect_worker_children() {",
                "  local roots=\"$1\"",
                "  local children=\"\"",
                "  local child=\"\"",
                "  for child in ${roots}; do",
                "    children=\"${children} $(pgrep -P \"${child}\" 2>/dev/null || true)\"",
                "  done",
                "  echo \"${children}\" | xargs 2>/dev/null || true",
                "}",
                "signal_forward() {",
                "  local sig=\"$1\"",
                '  echo "[SLURM_SIGNAL] $(date -u +%Y-%m-%dT%H:%M:%SZ) signal=${sig} forwarding_to_pid=${CHILD_PID:-} forwarding_to_pgid=${CHILD_PGID:-}"',
                '  if [[ -n "${CHILD_PGID:-}" ]]; then',
                '    kill -${sig} -- "-${CHILD_PGID}" 2>/dev/null || true',
                "    return",
                "  fi",
                '  if [[ -n "${CHILD_PID:-}" ]]; then',
                '    kill -${sig} "${CHILD_PID}" 2>/dev/null || true',
                "  fi",
                "}",
                "signal_forward_usr1() {",
                "  local worker_pids=\"\"",
                "  if [[ -n \"${CHILD_PID:-}\" ]]; then",
                "    worker_pids=\"$(collect_worker_children \"${CHILD_PID}\")\"",
                "  fi",
                '  echo "[SLURM_SIGNAL] $(date -u +%Y-%m-%dT%H:%M:%SZ) signal=USR1 forwarding_to_workers=${worker_pids:-none} fallback_pid=${CHILD_PID:-} forwarding_to_pgid=${CHILD_PGID:-}"',
                "  # In distributed accelerate/torchrun jobs, signaling the launcher with TERM",
                "  # starts elastic's forced shutdown timer and can kill ranks while FSDP is saving.",
                "  # Send USR1 to worker descendants instead; the trainer callback handles USR1.",
                "  if [[ -n \"${worker_pids}\" ]]; then",
                "    kill -USR1 ${worker_pids} 2>/dev/null || true",
                "    return",
                "  fi",
                "  # Single-process jobs have no launcher descendants, so signal the trainer directly.",
                '  if [[ -n "${CHILD_PID:-}" ]]; then',
                '    kill -TERM "${CHILD_PID}" 2>/dev/null || true',
                "    return",
                "  fi",
                "  signal_forward TERM",
                "}",
                "trap 'signal_forward TERM' TERM INT",
                "trap 'signal_forward_usr1' USR1",
                "trap 'stop_gpu_monitor; signal_forward TERM' TERM INT",
                # Replace the intermediate shell with the target command so the tracked child PID
                # is the actual Python process. Otherwise a warning signal can hit the shell first
                # and cascade into abrupt teardown before nested checkpoint handlers run.
                "set +e",
                "start_gpu_monitor",
                f"setsid bash -c 'exec {cmd_str_for_bash}' &",
                "CHILD_PID=$!",
                'CHILD_PGID="$(ps -o pgid= \"${CHILD_PID}\" 2>/dev/null | tr -d \" \")"',
                "while true; do",
                '  wait "${CHILD_PID}"',
                "  rc=$?",
                "  if [[ ${rc} -eq 0 ]]; then",
                "    EXIT_CODE=0",
                "    break",
                "  fi",
                "  # wait can be interrupted by trapped signals (USR1/TERM). Retry until the child exits.",
                "  if [[ ${rc} -ge 128 ]]; then",
                "    continue",
                "  fi",
                "  EXIT_CODE=${rc}",
                "  break",
                "done",
                "stop_gpu_monitor",
                'echo "[SLURM_END] $(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=${EXIT_CODE}"',
                'exit "${EXIT_CODE}"',
            ]
        )
        return "\n".join(script_lines) + "\n"

    # ------------------------------------------------------------------
    # polling / waiting
    # ------------------------------------------------------------------

    def _wait_for_jobs(
        self,
        jobs: list[tuple[str, Path]],
        dry_run: bool,
    ) -> None:
        """Poll sacct until all jobs have finished.  Raises on failure."""
        if dry_run or not jobs:
            return

        pending = dict(jobs)  # {job_id: log_path}
        LOGGER.info("[SLURM] Waiting for %d job(s): %s", len(pending), ", ".join(pending))

        while pending:
            time.sleep(self.poll_interval)
            finished = self._poll_jobs(list(pending.keys()))
            for job_id, (state, exit_code) in finished.items():
                log_path = pending.pop(job_id)
                slurm_log = log_path.with_suffix(".slurm.log")
                duration_str = "unknown"

                # Stream Slurm log content back to pipeline log.
                if slurm_log.exists():
                    content = slurm_log.read_text(encoding="utf-8", errors="replace")
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"\n--- slurm output (job {job_id}) ---\n")
                        fh.write(content)
                    for line in content.splitlines():
                        LOGGER.info("[SLURM:%s] %s", job_id, line.rstrip())

                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"[END] slurm_job={job_id} state={state} "
                        f"exit_code={exit_code} duration_seconds={duration_str}\n"
                    )

                if state not in ("COMPLETED",):
                    raise subprocess.CalledProcessError(
                        int(exit_code) if exit_code.isdigit() else 1,
                        [f"slurm-job-{job_id}"],
                    )
                LOGGER.info("[SLURM] Job %s finished: %s", job_id, state)

            if pending:
                LOGGER.info(
                    "[SLURM] Still waiting for %d job(s): %s",
                    len(pending),
                    ", ".join(pending),
                )

    @staticmethod
    def _poll_jobs(job_ids: list[str]) -> dict[str, tuple[str, str]]:
        """Check sacct for finished jobs.  Returns {job_id: (state, exit_code)}."""
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "--format=JobID,State,ExitCode",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
        )
        # Local/simplified Slurm installs often have accounting disabled, which makes sacct unusable.
        if (
            result.returncode != 0
            or "accounting storage is disabled" in (result.stderr or "").lower()
            or not (result.stdout or "").strip()
        ):
            finished: dict[str, tuple[str, str]] = {}
            for jid in job_ids:
                show = subprocess.run(
                    ["scontrol", "show", "job", "-o", str(jid)],
                    capture_output=True,
                    text=True,
                )
                text_out = (show.stdout or "").strip()
                if not text_out:
                    continue
                # Example: "... JobState=COMPLETED ... ExitCode=0:0 ..."
                state_match = re.search(r"\bJobState=(\w+)\b", text_out)
                exit_match = re.search(r"\bExitCode=([0-9]+):", text_out)
                state = state_match.group(1).split()[0] if state_match else ""
                exit_code = exit_match.group(1) if exit_match else "1"
                if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                    finished[str(jid)] = (state, exit_code)
            return finished
        finished: dict[str, tuple[str, str]] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            jid, state, exit_code = parts[0], parts[1].split()[0], parts[2]
            # Skip sub-steps (e.g. "12345.batch").
            if "." in jid:
                continue
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                finished[jid] = (state, exit_code.split(":")[0])
        return finished


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_executor(
    profile: dict[str, Any],
    stage_cfg: dict[str, Any] | None = None,
) -> LocalExecutor | SlurmExecutor:
    """Instantiate the right executor from profile config.

    Reads ``profile["executor"]`` for the base config.  If *stage_cfg*
    contains an ``"executor"`` key, those fields override the profile-level
    settings (e.g. a stage can request more GPUs or memory).

    Returns ``LocalExecutor`` when executor type is ``"local"`` or absent.
    """
    base_cfg: dict[str, Any] = dict(profile.get("executor", {}))
    # Merge stage-level overrides before selecting executor type so a stage can switch local->slurm.
    if stage_cfg:
        stage_exec = stage_cfg.get("executor", {})
        if isinstance(stage_exec, dict):
            base_cfg.update(stage_exec)

    exec_type = base_cfg.pop("type", "local")

    if exec_type == "local":
        return LocalExecutor()

    if exec_type != "slurm":
        raise ValueError(f"Unknown executor type: {exec_type!r}  (expected 'local' or 'slurm')")

    return SlurmExecutor(
        partition=base_cfg.get("partition", _SLURM_DEFAULTS["partition"]),
        gres=base_cfg.get("gres", _SLURM_DEFAULTS["gres"]),
        mem=base_cfg.get("mem", _SLURM_DEFAULTS["mem"]),
        time_limit=base_cfg.get("time", _SLURM_DEFAULTS["time"]),
        cpus_per_task=int(base_cfg.get("cpus_per_task", _SLURM_DEFAULTS["cpus_per_task"])),
        module_init=str(base_cfg.get("module_init", _SLURM_DEFAULTS["module_init"])),
        modules=list(base_cfg.get("modules", _SLURM_DEFAULTS["modules"])),
        conda_base=str(base_cfg.get("conda_base", _SLURM_DEFAULTS["conda_base"])),
        conda_env=str(base_cfg.get("conda_env", _SLURM_DEFAULTS["conda_env"])),
        poll_interval=int(base_cfg.get("poll_interval_seconds", _SLURM_DEFAULTS["poll_interval_seconds"])),
        extra_sbatch_flags=list(base_cfg.get("extra_sbatch_flags", [])),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _job_name_from_log(log_path: Path) -> str:
    """Derive a compact Slurm job name, e.g. ``l8B_10l_hydra``."""
    run_id = log_path.parent.name.strip().lower()
    run_name = log_path.stem
    if run_name.endswith(".train"):
        run_name = run_name[: -len(".train")]

    # Parse language count from labels like "...-10langs-...".
    lang_count = ""
    lang_match = re.search(r"-(\d+)langs-", run_name)
    if lang_match:
        lang_count = lang_match.group(1)

    # Parse model size from model slug prefixes like "l-3-1-8b-..." or "...-1b-...".
    model_tag = "lX"
    model_segment = run_name
    if lang_match:
        model_segment = run_name[: lang_match.start()]
    size_match = re.search(r"-(\d+)b(?:-|$)", model_segment)
    if size_match:
        model_tag = f"l{size_match.group(1)}B"

    approach = run_id or "run"
    if not approach:
        approach = "run"

    if lang_count:
        name = f"{approach}_{lang_count}l_{model_tag}"
    else:
        name = f"{approach}_{model_tag}"

    return _sanitize_job_name(name)


def _sanitize_job_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return sanitized[:64]


def _shell_quote(s: str) -> str:
    """Simple shell quoting — wrap in single quotes if it contains specials."""
    if re.match(r"^[a-zA-Z0-9_./:=@,-]+$", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"
