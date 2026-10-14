"""Bound search-plan validation and task prerequisites; no scheduling imports."""
from importlib.metadata import version
from pathlib import Path
import sys

from .. import execution as io
from . import data
from ..data import canonical_json_sha256
from ..artifacts import environment
from ..artifacts import validate_output
from ..artifacts import verify
from .protocol import SCHEMA
from .protocol import PROTOCOL
from .protocol import check_budget


def load_plan(path, *, all_inputs=False):
    plan = io.read_json(Path(path))
    unsigned = dict(plan); digest = unsigned.pop("sha256")
    if digest != canonical_json_sha256(unsigned) or plan["schema"] != SCHEMA or plan["protocol"] != PROTOCOL:
        raise ValueError("changed search protocol or plan")
    validate_output(plan["profile"])
    if Path(plan["profile"]["output_root"]).resolve() != Path(path).resolve().parent:
        raise ValueError("plan moved outside its frozen output root")
    if plan["environment"] != environment() or plan["runtime"] != {
        "python_executable":str(Path(sys.executable).resolve()),"pyarrow":version("pyarrow")}:
        raise ValueError("fitting runtime differs from the frozen package")
    for entry in plan["source"]:
        verify(entry)
    if all_inputs:
        for entry in [*plan["inputs"],*plan["scoring"],*plan["training"].values(),
                      *(e for group in plan["inner_scoring"].values() for e in group)]:
            verify(entry)
        for entry in plan["training"].values():
            data.read_training(entry)
    check_budget(plan["tasks"])
    return plan


def prerequisites(task,root,plan):
    required = []
    if task["kind"] != "calibrate" and task["fold"] not in plan["calibration"]:
        required.append(root/"results"/f"cal-{task['fold']}.json")
    if task.get("model") == "selected":
        required.append(root/"selection.json")
    if task["kind"] in ("rho_profile","bootstrap","no_EK"):
        required.append(root/"results"/"selected-all_grid.json")
    return required
