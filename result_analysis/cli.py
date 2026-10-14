"""Small shared argument contract for independently runnable CPU analyses."""
import argparse
import json
from pathlib import Path

from .evidence import Evidence


def module_main(run, description, from_parquet=None):
    parser = argparse.ArgumentParser(description=description)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", type=Path, help="prepared, hash-bound historical evidence directory")
    if from_parquet is not None:
        inputs.add_argument("--data", type=Path, help="simple Stage 1 parquet directory")
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory; never overwritten")
    args = parser.parse_args()
    if getattr(args, "data", None) is not None:
        data, output = args.data.resolve(), args.output.resolve()
        if output == data or output.is_relative_to(data) or data.is_relative_to(output):
            raise ValueError("write analysis outputs outside the input data directory")
        tables = from_parquet(data)
        output.mkdir(parents=True, exist_ok=False)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=table.index.name is not None)
        result = {"data": str(data), "output": str(output), "tables": {k: len(v) for k,v in tables.items()},
                  "scope": "CPU analysis from numerical observations; no model training or benchmark scoring"}
    else:
        result = run(Evidence(args.inputs), args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
