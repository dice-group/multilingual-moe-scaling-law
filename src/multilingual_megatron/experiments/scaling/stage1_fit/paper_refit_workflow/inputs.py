"""Fixed-law refits with portable inputs and separate CPU execution commands.

Calibration comes from the corresponding training fold, not evaluation losses.
Preparation never submits jobs; only the explicit submit command can do so.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .. import execution as io
from .. import fit
from . import protocol
from ..law_search import data
from ..atlas_comparison_workflow.model import panel
from ..data import canonical_json_sha256, sha256_file
from ..law_search.protocol import PRIMARY
from ..artifacts import binding, verify

SCHEMA = "stage1_paper_refit_inputs/v1"
FOLDS = (*PRIMARY, "all_grid")
NATIVE = ("Chinchilla-target-local", "JointMoE-JH-local",
          "ATLAS-full-active-positive-T100-L10", "ATLAS-full-total-positive-T100-L10")
SHARED = {"J0": 6, "M2": 10, "M5-ND": 12, "RN14-stored-reuse": 14}
MODELS = (*NATIVE, *SHARED)


def cells():
    """Fixed-law reproduction, not a new inner/outer model-selection exercise."""
    return [dict(model=model, fold=fold,
                 coefficient_scope="language_local" if model in NATIVE else "shared_with_language_local_amplitudes",
                 shared_parameters=SHARED.get(model, 0)) for model in MODELS for fold in FOLDS]


def sealed(payload):
    return payload | {"sha256": canonical_json_sha256(payload)}


def read_sealed(path):
    value = io.read_json(Path(path))
    unsigned = dict(value)
    digest = unsigned.pop("sha256", None)
    if digest != canonical_json_sha256(unsigned):
        raise ValueError(f"changed input manifest: {path}")
    return value


def local_binding(path, root):
    return dict(path=str(path.relative_to(root)), sha256=sha256_file(path))


def local_path(root, entry):
    relative = Path(entry["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("bundle data must use internal relative paths")
    return verify(entry | {"path": str(path)})


def source_input(plan, filename):
    matches = {(e["path"], e["sha256"]): e for e in plan["inputs"] if Path(e["path"]).name == filename}
    if len(matches) != 1:
        raise ValueError(f"expected one bound source for {filename}")
    entry = next(iter(matches.values()))
    verify(entry)
    return entry


def validate_split(rows, grid, endpoint, membership, fold):
    if rows.empty or rows.duplicated(data.KEYS).any():
        raise ValueError("empty or duplicate training observations")
    if set(rows.trajectory_id) & set(endpoint.trajectory_id):
        raise ValueError("larger-model evaluation entered training")
    if rows.D_total.max() > 33e9:
        raise ValueError("training exposure exceeds the original grid")
    if fold != "inner_terminal" and set(membership["train_trajectory_ids"]) & set(membership["test_trajectory_ids"]):
        raise ValueError("primary folds must hold out whole configurations")
    expected = data.subset(grid, membership, "train")
    pd.testing.assert_frame_equal(rows, expected, check_exact=True)
    test = data.subset(grid, membership, "test")
    if set(map(tuple, rows[data.KEYS].to_numpy())) & set(map(tuple, test[data.KEYS].to_numpy())):
        raise ValueError("training and scoring rows overlap")
    if not set(membership["score_languages"]) <= set(rows.language):
        raise ValueError("scoring includes a language without fitted coefficients")
    if fold == "all_grid":
        pd.testing.assert_frame_equal(rows, grid, check_exact=True)
        if len(test) or set(endpoint.language) != set(rows.language):
            raise ValueError("all-grid fit and larger-model language cohort disagree")
    elif test.empty:
        raise ValueError("empty primary evaluation split")
    return dict(rows=len(rows), configurations=rows.trajectory_id.nunique(),
                languages=rows.language.nunique(), test_rows=len(test))


def native_groups(rows, packets):
    """Preserve per-language objectives, donor choices, and loss-blind support."""
    languages = [p["language"] for p in packets]
    if len(set(languages)) != len(languages) or set(languages) != set(rows.language):
        raise ValueError("native packets do not cover exactly the fitted languages")
    index = pd.MultiIndex.from_frame(rows[data.KEYS])
    weights = fit.trajectory_weights(rows.trajectory_id.tolist())
    result = []
    for item in sorted(packets, key=lambda p: p["language"]):
        train, support = item["train"], item["support"]
        keys = pd.MultiIndex.from_arrays([train["trajectory_ids"], train["readout_ids"],
                                         [item["language"]] * len(train["y"])])
        indices = index.get_indexer(keys)
        if keys.has_duplicates or np.any(indices < 0) or len(indices) != rows.language.eq(item["language"]).sum():
            raise ValueError("native training membership differs from the shared-law split")
        expected = rows.iloc[indices]
        np.testing.assert_allclose(expected.unigram_nll, item["unigram_nll"], atol=0, rtol=0)
        for name, column in (("y", "raw_nll"), ("N", "N_act"), ("N_total", "N_total"), ("E", "E"), ("K", "K")):
            np.testing.assert_allclose(train[name], expected[column], atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(train["D"])[:, 0], expected.D_t, atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(train["U"])[:, 0], expected.U_t, atol=0, rtol=0)
        np.testing.assert_allclose(np.sum(train["D"], axis=1), expected.D_total, atol=1e-6, rtol=1e-12)
        actual_weights = np.asarray(train["weights"], float)
        if not np.isfinite(actual_weights).all() or np.any(actual_weights <= 0):
            raise ValueError("invalid native objective weights")
        np.testing.assert_allclose(actual_weights/actual_weights.sum(), weights[indices]/weights[indices].sum(),
                                   atol=1e-14, rtol=1e-12)
        if not 0 < train["delta"] < float("inf") or train["training_minimum"] != min(train["y"]):
            raise ValueError("invalid training-only Huber calibration or loss floor")
        if ({"y", "weights", "delta"} | data.RESPONSES) & set(support):
            raise ValueError("native domain support must contain covariates only")
        if support["training_minimum"] != train["training_minimum"]:
            raise ValueError("domain support must use the training loss minimum")
        result.append({k: item[k] for k in ("language", "donors", "train", "support", "unigram_nll")} |
                      {"indices": indices.tolist()})
    return result


def load(path):
    """Read metadata only; fitting must not open evaluation response files."""
    manifest = read_sealed(path)
    full = manifest.get("scope") == protocol.VERSION
    if manifest["schema"] != SCHEMA or manifest["cells"] != (protocol.cells() if full else cells()):
        raise ValueError("fixed main-paper refit inventory changed")
    if manifest["native_specs"] != {s.name: s.payload() for s in panel() if s.name in (protocol.NATIVE if full else NATIVE)}:
        raise ValueError("published-law definitions changed")
    expected_folds = protocol.FOLDS if full else FOLDS
    if set(manifest["training"]) != set(expected_folds) or set(manifest["memberships"]) != set(expected_folds):
        raise ValueError("missing or additional fit folds")
    return manifest


def read_training(path, fold):
    manifest = load(path)
    root = Path(path).resolve().parent
    entry = manifest["training"][fold]
    packet_path = local_path(root, entry["packet"])
    packet = read_sealed(packet_path)
    if packet["fold"] != fold:
        raise ValueError("training packet belongs to a different fold")
    for bound in packet["files"]:
        local_path(packet_path.parent, bound)
    rows, sources, native = data.read_training(binding(packet_path))
    calibration = io.read_json(local_path(root, entry["calibration"]))
    return rows, sources, native, calibration


def verify_bundle(path):
    manifest = load(path)
    root = Path(path).resolve().parent
    scoring = {name: local_path(root, e) for name, e in manifest["scoring"].items()}
    grid, endpoint = (pd.read_parquet(scoring[n]) for n in ("grid.parquet", "endpoint.parquet"))
    local_path(root, manifest["provenance"])
    for fold in manifest["training"]:
        rows, _, native, calibration = read_training(path, fold)
        inventory = validate_split(rows, grid, endpoint, manifest["memberships"][fold], fold)
        if inventory != manifest["training"][fold]["inventory"]:
            raise ValueError("training inventory changed")
        native_groups(rows, native)
        if not 0 < calibration["delta"] < float("inf") or len(calibration["anchor"]) != 15:
            raise ValueError("invalid shared-law training calibration")
        fit.M5M6Regularization.from_mapping(calibration["regularization"])
    return dict(models=len(set(c["model"] for c in manifest["cells"])), fits=len(manifest["cells"]),
                inventory={f: manifest["training"][f]["inventory"] for f in manifest["training"]},
                scope="input preparation only; no coefficients refitted or jobs submitted")


def prepare(profile_path, *, output=None):
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    full = profile.get("scope") == protocol.VERSION
    folds = protocol.FOLDS if full else FOLDS
    native_names = protocol.NATIVE if full else NATIVE
    root = Path(output).resolve() if output is not None else (profile_path.parent / profile["output_root"]).resolve()
    source_paths = [(profile_path.parent / profile[key]).resolve() for key in ("search_plan", "coverage_plan")]
    if root.exists():
        raise ValueError("preparation is create-only; choose a new output directory")
    if any(root.is_relative_to(p.parent) or p.is_relative_to(root) for p in source_paths):
        raise ValueError("output overlaps an input campaign")
    search, coverage = [read_sealed(p) for p in source_paths]
    if coverage["parent_sha256"] != search["sha256"]:
        raise ValueError("coverage laws and baseline inputs belong to different grids")
    if any(coverage[key][fold] != search[key][fold] for key in ("training", "memberships") for fold in folds):
        raise ValueError("coverage laws and baseline split contracts differ")
    specs = {s["name"]: s for s in search["native_specs"] if s["name"] in native_names}
    if specs != {s.name: s.payload() for s in panel() if s.name in native_names}:
        raise ValueError("published-law definitions changed")
    scoring = {Path(e["path"]).name: e for e in search["scoring"]}
    names = ("grid.parquet", "endpoint.parquet", "exposure_contract.json")
    source_scoring = {name: verify(scoring[name]) for name in names}
    grid, endpoint = (pd.read_parquet(source_scoring[n]) for n in names[:2])
    original_native = source_input(search, "packets.json.gz")
    packets = json.loads(gzip.decompress(Path(original_native["path"]).read_bytes()))
    prepared, source_bindings = {}, [binding(profile_path), *(binding(p) for p in source_paths), original_native]
    for fold in folds:
        entry = search["training"][fold]
        rows, _, generated = data.read_training(entry)
        inventory = validate_split(rows, grid, endpoint, search["memberships"][fold], fold)
        if fold in profile["expected_inventory"] and inventory != profile["expected_inventory"][fold]:
            raise ValueError(f"unexpected paper inventory: {fold}: {inventory}")
        calibration = {k: coverage["calibrations"][fold][k] for k in ("delta", "regularization", "anchor")}
        if fold == "all_grid":
            native = [p | {"train": p["train"] | {"delta": calibration["delta"]},
                           "support": {k: v for k, v in p["train"].items() if k not in {"y", "weights"}}}
                      for p in generated]
        elif fold in PRIMARY:
            native = [p for p in packets if p["fold"] == fold]
            for key in ("delta", "regularization", "anchor"):
                if calibration[key] != search["calibration"][fold][key]:
                    raise ValueError("shared-law training calibration changed")
        else:
            native = [p | {"train": p["train"] | {"delta": calibration["delta"]},
                           "support": {k: v for k, v in p["train"].items() if k not in {"y", "weights"}}}
                      for p in generated]
        native = native_groups(rows, native)
        packet_path = verify(entry)
        packet = read_sealed(packet_path)
        files = {Path(e["path"]).name: e | {"path": str(packet_path.parent / e["path"])} for e in packet["files"]}
        prepared[fold] = (inventory, calibration, native, files)
        source_bindings.extend([entry, *files.values()])
    # Do not create an apparently usable bundle until the source populations pass.
    root.mkdir(parents=True)
    copied_scoring, training = {}, {}
    for name, source in source_scoring.items():
        path = root / "scoring" / name
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, path)
        copied_scoring[name] = local_binding(path, root)
    for fold, (inventory, calibration, native, files) in prepared.items():
        directory = root / "training" / fold
        directory.mkdir(parents=True)
        for name in ("rows.parquet", "sources.npz"):
            shutil.copyfile(verify(files[name]), directory / name)
        io.write_json_once(directory / "native.json", dict(packets=native))
        io.write_json_once(directory / "calibration.json", calibration)
        packet = dict(fold=fold, rows=inventory["rows"], languages=inventory["languages"],
                      trajectories=inventory["configurations"], training_only_responses=True,
                      files=[local_binding(directory / n, directory) for n in ("rows.parquet", "sources.npz", "native.json")])
        io.write_json_once(directory / "packet.json", sealed(packet))
        training[fold] = dict(packet=local_binding(directory / "packet.json", root),
                              calibration=local_binding(directory / "calibration.json", root), inventory=inventory)
    provenance = dict(inputs=[*source_bindings, *(scoring[n] for n in names)],
                      protocols=dict(shared=search["protocol"], repetition=coverage["rules"]),
                      source_environments=dict(shared=search["environment"], repetition=coverage["environment"]))
    io.write_json_once(root / "provenance.json", provenance)
    manifest = dict(schema=SCHEMA, scope=protocol.VERSION if full else "main_table", cells=protocol.cells() if full else cells(), native_specs=specs, training=training,
                    memberships={f: search["memberships"][f] for f in folds}, scoring=copied_scoring,
                    provenance=local_binding(root / "provenance.json", root),
                    calibration="reuse fold-specific training-only calibration; do not recalibrate",
                    native_support="primary: original loss-blind domain; all_grid: training covariates only",
                    repetition_regularization="RN14 uses zero ridge; preserve calibration regularization separately",
                    evaluation="configuration-balanced primary RMSE; separate equal-language L26 RMSE",
                    evidence="fixed-law reproduction; no new selection, independent seed, or blind L26 forecast")
    io.write_json_once(root / "manifest.json", sealed(manifest))
    return verify_bundle(root / "manifest.json") | {"manifest": str(root / "manifest.json")}
