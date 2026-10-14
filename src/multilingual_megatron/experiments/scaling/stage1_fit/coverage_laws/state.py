"""Coverage-fit plan and result validation shared by workers and reporting."""
import os
from pathlib import Path
import sys

from .. import execution as io
from ..law_search import data
from ..data import canonical_json_sha256
from ..artifacts import binding, environment, validate_output, verify
from ..problems import problem
from .contracts import cells, plan_rules, tasks, validate_profile
from .model import Model


def seal(path, payload):
    io.write_json_once(path, payload | {"sha256": canonical_json_sha256(payload)})


def cpu_guard():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("real work requires a single-core CPU Slurm task")
    if any(os.environ.get(k) not in (None, "", "0") for k in
           ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")):
        raise ValueError("GPU work is not authorized")
    if any(os.environ.get(k) != "1" for k in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("all numerical thread counts must equal one")


def load(path, *, all_inputs=False):
    path = Path(path).resolve()
    plan = io.read_json(path)
    unsigned = {k: v for k, v in plan.items() if k != "sha256"}
    if canonical_json_sha256(unsigned) != plan["sha256"]:
        raise ValueError("sealed plan or equations changed")
    rules = plan_rules(plan)
    validate_profile(plan["profile"], rules)
    if plan["tasks"] != tasks(plan["profile"], rules) or plan["cells"] != cells(rules):
        raise ValueError("task graph changed")
    if validate_output(plan["profile"]) != path.parent:
        raise ValueError("plan outside frozen output root")
    if environment() != plan["environment"] or Path(sys.executable).resolve() != Path(plan["profile"]["python_executable"]).resolve():
        raise ValueError("frozen fitting runtime changed")
    current = Path(__file__).resolve().parent.parent
    for entry in plan["source"]:
        verify(entry)
        relative = Path(entry["path"]).relative_to(path.parent / "source" / "stage1_fit")
        if binding(current / relative)["sha256"] != entry["sha256"]:
            raise ValueError("run the frozen source bundle, not changed live code")
    for entry in plan["inputs"]:
        verify(entry)
    if all_inputs:
        for entry in [*plan["scoring"].values(), plan["baseline_metrics"],
                      *(e for group in plan["inner_scoring"].values() for e in group)]:
            verify(entry)
        for entry in plan["training"].values():
            data.read_training(entry)
    return plan


def result_path(root, task):
    return root / "results" / f"{task['stage']}--{task['id']}.json"


def read_result(root, plan, task):
    value = io.read_json(result_path(root, task))
    if value["plan_sha256"] != plan["sha256"] or value["task_sha256"] != canonical_json_sha256(task):
        raise ValueError("foreign task result")
    for entry in value.get("dependencies", []):
        verify(entry)
    return value


def make_model(plan, fold, name):
    rows, _, _ = data.read_training(plan["training"][fold])
    cal = plan["calibrations"][fold]
    if plan_rules(plan).get("protocol") in ("law_improvement_round2_v1", "law_improvement_round3_v1",
                                           "expert_transform_v1", "expert_transform_parallel_v1", "coverage_exponents_v1"):
        # Keep the inherited calibration auditable while describing the actual
        # zero-ridge objective in worker/certification and scoring receipts.
        cal = cal | dict(calibration_regularization=cal["regularization"],
                         regularization=cal["regularization"] | {"lambda": 0.})
    stored = rows.N_total.to_numpy() if "N_total" in rows else None
    return Model(name, problem(rows, cal["regularization"]), rows.U_t.to_numpy(),
                 stored_parameters=stored), rows, cal
