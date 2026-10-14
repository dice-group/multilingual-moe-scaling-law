"""Read-only independent replay of the historical panel; no optimization.

This analysis sidecar has its own source binding and cannot alter a frozen
campaign's fitting code, equations, selector, or output receipts.
"""

from .replay import shared_equation
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .. import execution as io
from . import replay
from .state import load_plan
from .protocol import PRIMARY
from ..artifacts import binding


def reproduce(plan_path):
    began=time.monotonic()
    plan=load_plan(plan_path,all_inputs=True)
    root=Path(plan_path).resolve().parent
    profile=json.loads(Path(plan["profile"]["input_profile"]).read_text())
    shared_path=Path(profile["shared_fits"])/"selected.json"
    literature=Path(profile["literature"])
    shared=json.loads(shared_path.read_text())
    selected=json.loads(gzip.decompress((literature/"selected.json.gz").read_bytes()))
    packets=json.loads(gzip.decompress((literature/"packets.json.gz").read_bytes()))
    packets={(p["fold"],p["language"]):p for p in packets}
    grid=pd.read_parquet(root/"scoring/grid.parquet").set_index(["trajectory_id","readout_id","language"])
    stored=pd.read_parquet(root/"scoring/baseline_grid.parquet")
    keys=["trajectory_id","readout_id","language"]
    reference={key:group.set_index(keys).sort_index() for key,group in stored.groupby(["fold","model"])}
    maximum=0.; rows_replayed=0; models=set()
    for fold in PRIMARY:
        for name,record in shared[fold].items():
            expected=reference[fold,name]
            rows=grid.loc[expected.index].reset_index()
            predicted=shared_equation(record,rows)
            difference=float(np.max(abs(predicted-expected.prediction.to_numpy())))
            if difference>1e-10:
                raise ValueError(f"shared baseline replay mismatch {fold}/{name}: {difference}")
            maximum=max(maximum,difference);rows_replayed+=len(rows);models.add(name)
    for entry in selected:
        fold,name,lang=entry["fold"],entry["model"],entry["language"]
        if name not in {s["name"] for s in plan["native_specs"]}:
            continue
        packet=packets[fold,lang]
        query=packet["test"] | {"training_minimum":packet["train"]["training_minimum"]}
        prediction=replay.native(entry["spec"],entry["coordinates"],query)-packet["unigram_nll"]
        identity=pd.MultiIndex.from_arrays([query["trajectory_ids"],query["readout_ids"],[lang]*len(prediction)],names=keys)
        expected=reference[fold,name].loc[identity]
        difference=float(np.max(abs(prediction-expected.prediction.to_numpy())))
        if difference>1e-10:
            raise ValueError(f"native baseline replay mismatch {fold}/{name}/{lang}: {difference}")
        maximum=max(maximum,difference);rows_replayed+=len(prediction);models.add(name)
    if rows_replayed!=len(stored) or len(models)!=51:
        raise ValueError("historical replay coverage differs from all 51 frozen baselines")
    receipt=dict(plan_sha256=plan["sha256"],complete=True,rows=rows_replayed,models=len(models),
        max_absolute_prediction_difference=maximum,elapsed_seconds=time.monotonic()-began,
        optimization_performed=False,endpoint_scores_computed=False,
        inputs=[binding(shared_path),binding(literature/"selected.json.gz"),binding(literature/"packets.json.gz"),
                binding(root/"scoring/baseline_grid.parquet")],
        source=[binding(Path(__file__)),binding(Path(replay.__file__))])
    io.write_json_once(root/"baseline_replay.json",receipt)
    print({k:v for k,v in receipt.items() if k not in ("inputs","source")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    reproduce(parser.parse_args().plan)
