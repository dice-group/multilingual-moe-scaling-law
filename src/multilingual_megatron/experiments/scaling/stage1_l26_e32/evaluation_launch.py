"""Pure launch construction for the L26/E32 evaluation workflow."""
from pathlib import Path
import re
import shlex

MODULE = "multilingual_megatron.experiments.scaling.stage1_l26_e32.evaluation"
SOURCE = "src/multilingual_megatron/experiments/scaling/stage1_l26_e32/evaluation.py"
LAUNCH_SOURCE = "src/multilingual_megatron/experiments/scaling/stage1_l26_e32/evaluation_launch.py"
EXPOSURE_SOURCE = "src/multilingual_megatron/experiments/scaling/stage1_release/exposure.py"
PACKAGE_SOURCE = "src/multilingual_megatron/experiments/scaling/stage1_l26_e32/__init__.py"
PROFILE = "configs/stage1/production/stage1_l26_e32_evaluation_v1.json"


def allocation(profile, runtime):
    """The requested allocation must match the explicitly recorded hardware."""
    nodes, gpus = profile["nodes"], profile["gpus_per_node"]
    if any(type(n) is not int or n <= 0 for n in (nodes, gpus)):
        raise ValueError("nodes and gpus_per_node must be positive integers")
    hardware = runtime["hardware"]
    if (nodes != hardware["nodes"] or gpus != hardware["gpus_per_node"]
            or nodes * gpus != hardware["world_size"]):
        raise ValueError("profile allocation differs from recorded runtime hardware")
    kind = hardware["gpu_type"].lower()
    if not re.fullmatch(r"[a-z0-9_-]+", kind):
        raise ValueError("invalid runtime GPU type")
    return nodes, gpus, kind


def attempt_logs(output, attempt_id):
    if not isinstance(attempt_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt_id):
        raise ValueError("attempt_id must be one safe path component")
    return Path(output) / "logs" / "attempts" / attempt_id


def render_script(plan_path, plan, action, *, module=MODULE):
    if action not in ("run", "preflight"):
        raise ValueError("unsupported evaluation launch action")
    cfg, runtime = plan["profile"], plan["runtime"]["module_runtime"]
    source = Path(plan["output"]) / "source"
    python = str(Path(runtime["conda_env"]) / "bin/python")
    q = shlex.quote
    return ("#!/usr/bin/env bash\nset -eo pipefail\n"
        + "export PATH=/usr/local/bin:/usr/bin:/bin:" + q(cfg["slurm_bin"]) + "\n"
        + "source " + q(runtime["module_init"]) + "\nmodule purge\nmodule load " + shlex.join(runtime["modules"]) + "\n"
        + "source " + q(cfg["conda_init"]) + "\nconda activate " + q(runtime["conda_env"]) + "\n"
        + "export SLURM_EXPORT_ENV=ALL PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
        + "export PYTHONPATH=" + q(str(source / "src") + ":" + str(source)) + "\n"
        + "cd " + q(str(source)) + "\nexec "
        + shlex.join([python, "-m", module, action, "--plan", str(plan_path)]) + "\n")


def submission_command(plan, script, *, job_name="l26_e32_final_loss"):
    cfg = plan["profile"]
    nodes, gpus, kind = allocation(cfg, plan["runtime"])
    logs = attempt_logs(plan["output"], plan["attempt_id"])
    return [str(Path(cfg["slurm_bin"]) / "sbatch"), "--parsable", "--export=NONE",
        f"--partition={cfg['partition']}", f"--account={cfg['account']}", f"--nodes={nodes}",
        "--ntasks-per-node=1", f"--cpus-per-task={cfg['cpus_per_task']}", f"--mem={cfg['memory']}",
        f"--gres=gpu:{kind}:{gpus}", f"--time={cfg['time_limit']}",
        f"--job-name={job_name}", f"--output={logs}/evaluation-%j.log", str(script)]


def step_command(plan_path, plan, *, module=MODULE):
    nodes, _, _ = allocation(plan["profile"], plan["runtime"])
    python = str(Path(plan["runtime"]["module_runtime"]["conda_env"]) / "bin/python")
    return [str(Path(plan["profile"]["slurm_bin"]) / "srun"), f"--nodes={nodes}",
            f"--ntasks={nodes}", "--ntasks-per-node=1", python, "-m", module,
            "rank", "--plan", str(plan_path)]


def worker_command(plan, spec, node, master, job_id):
    nodes, gpus, _ = allocation(plan["profile"], plan["runtime"])
    if not 0 <= int(node) < nodes:
        raise ValueError("worker node rank is outside the configured allocation")
    python = str(Path(plan["runtime"]["module_runtime"]["conda_env"]) / "bin/python")
    return [python, "-m", "torch.distributed.run", f"--nnodes={nodes}", f"--nproc_per_node={gpus}",
        f"--node_rank={node}", f"--master_addr={master}", f"--master_port={20000 + int(job_id) % 20000}",
        "--module", "multilingual_megatron.pipeline.backends.megatron_bridge.run_study_bridge_recipe",
        "--recipe", spec["model_recipe"]["name"], "--dataset", "llm-pretrain", "--seq_length", str(spec["sequence_length"]),
        "--per-language-validation-spec", plan["spec"]["path"], *spec["model_recipe"]["overrides"]]
