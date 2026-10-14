"""Create-only, training-only audit refinement for one coverage-exponent cell.

The original 48-start winner is unchanged. Only the independent direct audit's
optimizer stopping tolerance is tightened; physical certification is unchanged.
"""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .. import execution as io
from ..law_search import data
from ..law_search import model as legacy
from ..data import canonical_json_sha256
from ..shared_numerics import agreement
from ..scoring import checked_score
from ..artifacts import binding, verify
from .state import cpu_guard, seal, make_model, read_result, result_path
from .model import validate
from .reporting import predictions, selection_receipt
from . import replay

CELL = "largest_N_act--RN15-stored-separate-exponents"


def prepare(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    plan = io.read_json(parent)
    if canonical_json_sha256({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("modified parent plan")
    for entry in plan["source"] + plan["inputs"]:
        verify(entry)
    task = next(t for t in plan["tasks"] if t["id"] == CELL and t["stage"] == "certify")
    original = read_result(parent.parent, plan, task)
    if original["certified"] or not original["selected"]["success"]:
        raise ValueError("expected the original failed audit of a successful fit")
    output.mkdir()  # create-only, no overwrite or resume
    source = output / "source" / "stage1_fit"
    shutil.copytree(parent.parent / "source" / "stage1_fit", source)
    live = Path(__file__).resolve().parent.parent
    for relative in ("law_search/model.py", "coverage_laws/recovery.py", "coverage_laws/campaign.py"):
        shutil.copy2(live / relative, source / relative)
    seal(output / "plan.json", dict(parent=binding(parent),
        original=binding(result_path(parent.parent, task)), cell=CELL,
        source=[binding(p) for p in sorted(source.rglob("*.py"))],
        ftols=[1e-14, 1e-16], seconds=3600,
        amendment="tighten direct optimizer stop only; retain selected 48-start winner and all acceptance thresholds"))


def run(path):
    cpu_guard()
    path = Path(path).resolve()
    amendment = io.read_json(path)
    if canonical_json_sha256({k: v for k, v in amendment.items() if k != "sha256"}) != amendment["sha256"]:
        raise ValueError("modified amendment")
    for entry in amendment["source"]:
        verify(entry)
    if Path(__file__).resolve() != path.parent / "source/stage1_fit/coverage_laws/recovery.py":
        raise ValueError("execute the frozen recovery source")
    parent = verify(amendment["parent"])
    plan = io.read_json(parent)
    for entry in plan["source"] + plan["inputs"]:
        verify(entry)
    original = io.read_json(verify(amendment["original"]))
    for entry in original["dependencies"]:
        verify(entry)
    selection_receipt(Path(parent).parent, plan)
    sink = path.parent / "attempts"
    sink.mkdir()
    model, rows, cal = make_model(plan, original["fold"], original["form"])
    profile = legacy.Profile(model, cal["delta"], time.monotonic()+amendment["seconds"])
    selected = original["selected"]
    validate(model, selected)
    passed = False
    for i, ftol in enumerate(amendment["ftols"]):
        audit = legacy.Direct(profile, selected).run(ftol=ftol)
        validate(model, audit)
        io.write_json_once(sink / f"audit-{i}.json", audit | dict(ftol=ftol))
        passed = bool(agreement(model, selected, audit))
        print(dict(ftol=ftol, kkt=audit["kkt"], passed=passed), flush=True)
        if passed:
            break
    gap = float(np.max(abs(legacy.predict(model, selected)-replay.predict(model.name, selected, rows))))
    if gap > 1e-10:
        raise ValueError("independent training replay failed")
    result = original | dict(certified=passed, audit=audit,
        recovery_plan_sha256=amendment["sha256"], original_receipt=amendment["original"],
        independent_training_replay_max_gap=gap, heldout_scores_computed=False,
        error=None if passed else "bounded tighter audit still failed")
    io.write_json_once(path.parent / "certification.json", result)
    if not passed:
        return False
    # Certification is persisted before any new holdout response is read.
    metrics, outputs = [], []
    for case, filename in (("grid", "grid.parquet"), ("L26", "endpoint.parquet")):
        score_rows = pd.read_parquet(verify(plan["scoring"][filename]))
        membership = plan["memberships"][original["fold"]]
        score_rows = (data.subset(score_rows, membership, "test") if case == "grid" else
                      score_rows[score_rows.language.isin(membership["score_languages"])].reset_index(drop=True))
        pred = predictions(result, model.name, score_rows)
        metrics.append(dict(model=model.name, fold=original["fold"], case=case,
                            **checked_score(pred, score_rows, endpoint=case == "L26")))
        outputs.append(pred.assign(case=case))
    pd.concat(outputs).to_parquet(path.parent / "predictions.parquet", index=False)
    seal(path.parent / "scores.json", dict(recovery_plan_sha256=amendment["sha256"],
        certification=binding(path.parent / "certification.json"),
        predictions=binding(path.parent / "predictions.parquet"),
        inputs=list(plan["scoring"].values()), metrics=metrics))
    for entry in amendment["source"] + [amendment["original"], amendment["parent"]]:
        verify(entry)
    print(metrics, flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.output:
            parser.error("prepare requires --output")
        prepare(args.plan, args.output)
    else:
        raise SystemExit(0 if run(args.plan) else 1)
