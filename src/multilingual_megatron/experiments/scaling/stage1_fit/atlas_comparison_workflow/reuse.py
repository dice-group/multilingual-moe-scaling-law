"""Exact inheritance of sealed pilot evidence, retaining unresolved outcomes."""
from pathlib import Path

from .. import atlas_full_pilot as io


def remap_task(task, old_panel, new_panel):
    packet, index = divmod(task, len(old_panel))
    spec = old_panel[index]
    matches = [i for i, value in enumerate(new_panel) if value["name"] == spec["name"]]
    if not matches:
        if spec["form"] != "moe":
            raise ValueError("only constructed MoE variants may be deferred")
        return None
    index = matches[0]
    if new_panel[index] != spec:
        raise ValueError("inherited equation specification changed")
    return packet*len(new_panel)+index


def packet_parity(old, new):
    if len(old) != len(new) or any(a["file_sha256"] != b["file_sha256"] for a, b in zip(old, new)):
        raise ValueError("inherited pilot packets or observation bytes changed")


def budget_protocol_parity(old, new):
    changed = {"version", "pilot_allocated_core_hours", "main_allocated_core_hours",
               "finalize_allocated_core_hours", "local_check_reserve_core_hours",
               "allocations", "maximum_total_core_hours", "max_concurrent_cores", "task_partition", "amendment"}
    if {k: v for k, v in old.items() if k not in changed} != {k: v for k, v in new.items() if k not in changed}:
        raise ValueError("budget amendment changed the scientific protocol")


def budget_receipt(value, old, task, packet, spec, receipt, certified):
    if (value.get("worker_error") or value["plan_sha256"] != old["sha256"]
            or value["task"] != task or value["packet_sha256"] != packet["sha256"]
            or value["spec"] != spec or value.get("heldout_scores_computed") is not False
            or value["certified"] != certified):
        raise ValueError("inherited pilot evidence or certification identity changed")
    # Keep failed certifications as failures. Inheritance must never choose
    # only successful languages or silently turn a prior failure into a pass.
    return ({key: item for key, item in value.items() if key not in ("sha256", "plan_sha256")}
            | dict(inherited_from=dict(plan_sha256=old["sha256"], task=task,
                                       receipt=receipt, certified=certified)))


def prepare_budget(path, protocol, environment, packets, observations, historical, sources, amended):
    old = io.read(path)
    if old["protocol"]["version"] != "atlas_comparison/v7_native" or old["environment"] != environment:
        raise ValueError("parallel inheritance requires the frozen v7 environment")
    budget_protocol_parity(old["protocol"], protocol)
    for bound in [old["archive"], *old["sources"], *old["inputs"], *old["packets"],
                  old["observations"], old["historical_predictions"]]:
        io.verify(bound)
    current = {Path(b["original_path"]): b for b in sources}
    for bound in old["sources"]:
        original = Path(bound["original_path"])
        if original not in amended and current[original]["file_sha256"] != bound["file_sha256"]:
            raise ValueError(f"budget amendment changed fitting/runtime source: {original}")
    packet_parity(old["packets"], packets)
    packet_parity([old["observations"], old["historical_predictions"]], [observations, historical])
    audit_path = Path(old["output_root"])/"audit.json"
    audit = io.read(audit_path)
    completion_path = Path(old["output_root"])/"execution/main_completion.json"
    completion = io.read(completion_path)
    if audit["plan_sha256"] != old["sha256"] or completion["plan_sha256"] != old["sha256"]:
        raise ValueError("inherited pilot audit belongs to another plan")
    coverage = {row["task"]: row for row in audit["coverage"]}
    tasks = sorted(coverage)
    present = {int(p.stem) for p in (Path(old["output_root"])/"results").glob("*.json")}
    if (len(old["pilot_tasks"]) != 146 or not set(old["pilot_tasks"]) <= set(tasks)
            or present != set(tasks) or len(coverage) != len(audit["coverage"])
            or set(tasks) & set(audit["missing_tasks"])
            or set(tasks) | set(audit["missing_tasks"]) != set(range(old["task_count"]))):
        raise ValueError("must inherit every completed fit and all original pilot cases without filtering")
    rows, bindings = [], [io.bind(path), old["archive"], io.bind(audit_path), io.bind(completion_path)]
    for task in tasks:
        checked = coverage[task]
        receipt = checked["receipt"]
        io.verify(receipt)
        packet_index, spec_index = divmod(task, len(protocol["panel"]))
        row = budget_receipt(io.read(receipt["path"]), old, task, io.read(packets[packet_index]["path"]),
            protocol["panel"][spec_index], receipt, checked["certified"])
        rows.append(row)
        bindings.append(receipt)
    return rows, bindings


def write_budget(plan, rows):
    root = Path(plan["output_root"])
    for value in rows:
        io.write(root/f"results/{value['task']:05d}.json", value | dict(plan_sha256=plan["sha256"]))
    io.write(root/"execution/inheritance.json", dict(plan_sha256=plan["sha256"],
        completed_tasks=len(rows), certified_tasks=sum(row["certified"] for row in rows),
        unresolved_tasks=[row["task"] for row in rows if not row["certified"]],
        status="all_completed_v7_fits_inherited_including_uncertified",
        additional_allocated_core_hours=0., heldout_scores_computed=False))
    pilots = [row for row in rows if row["task"] in plan["pilot_tasks"]]
    io.write(root/"execution/pilot_completion.json", dict(plan_sha256=plan["sha256"],
        completed_tasks=len(pilots), certified_tasks=sum(row["certified"] for row in pilots),
        unresolved_tasks=[row["task"] for row in pilots if not row["certified"]],
        status="original_146_corrected_pilot_cases_inherited_not_reexecuted",
        additional_allocated_core_hours=0., heldout_scores_computed=False))


def prepare(path, protocol, environment, packets, observations, historical):
    old = io.read(path)
    if old["protocol"]["version"] != "atlas_comparison/v2" or old["environment"] != environment:
        raise ValueError("pilot inheritance requires the retained v2 environment")
    changed = {"version", "panel", "pilot_allocated_core_hours", "pilot_design", "amendment", "source_mode"}
    for key, value in old["protocol"].items():
        if key not in changed and protocol.get(key) != value:
            raise ValueError(f"pilot scientific protocol changed: {key}")
    # Only orchestration/reporting and their tests change. Verify EVERY other
    # previously bound source, not just the main equation/optimizer modules.
    allowed = {Path(__file__).parent/"campaign.py", Path(__file__).parent/"reporting.py",
               Path.cwd()/"tests/test_stage1_atlas_comparison.py"}
    for bound in old["sources"]:
        if Path(bound["path"]) not in allowed:
            io.verify(bound)
    for bound in [old["archive"], *old["inputs"], *old["packets"], old["observations"], old["historical_predictions"]]:
        io.verify(bound)
    packet_parity(old["packets"], packets)
    packet_parity([old["observations"], old["historical_predictions"]], [observations, historical])
    checked_path = Path(old["output_root"])/"audit.json"
    checked = io.read(checked_path)
    if checked["plan_sha256"] != old["sha256"]:
        raise ValueError("inherited pilot audit belongs to a different plan")
    coverage = {item["task"]: item for item in checked["coverage"]}
    bindings = [io.bind(path), old["archive"], io.bind(checked_path)]
    rows = []
    for original_task in old["pilot_tasks"]:
        task = remap_task(original_task, old["protocol"]["panel"], protocol["panel"])
        if task is None:
            continue
        receipt = coverage[original_task]["receipt"]
        io.verify(receipt)
        value = io.read(receipt["path"])
        packet_index, spec_index = divmod(task, len(protocol["panel"]))
        packet = io.read(packets[packet_index]["path"])
        if (not coverage[original_task]["certified"] or not value["certified"]
                or value["plan_sha256"] != old["sha256"] or value["task"] != original_task
                or value["packet_sha256"] != packet["sha256"]
                or value["spec"] != protocol["panel"][spec_index]
                or value.get("heldout_scores_computed") is not False):
            raise ValueError("inherited pilot has uncertified or mismatched evidence")
        bindings.append(receipt)
        rows.append({key: item for key, item in value.items() if key not in ("sha256", "plan_sha256", "task")}
                    | dict(task=task, inherited_from=dict(plan_sha256=old["sha256"],
                           task=original_task, receipt=receipt)))
    if len(rows) != 63 or len({row["task"] for row in rows}) != 63:
        raise ValueError("expected all 63 predeclared native/control v2 pilot cases")
    return rows, bindings


def write(plan, rows):
    root = Path(plan["output_root"])
    for value in rows:
        io.write(root/f"results/{value['task']:05d}.json", value | dict(plan_sha256=plan["sha256"]))
    io.write(root/"execution/pilot_completion.json", dict(plan_sha256=plan["sha256"],
        completed_tasks=len(rows), status="inherited_certified_v2_not_reexecuted",
        additional_allocated_core_hours=0., heldout_scores_computed=False))
