"""Split-first, immutable training packets and loss-blind exposure assembly."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import execution as io
from .. import fit
from . import model
from .. import m6_composition
from .. import selection
from ..atlas_exposure import exposure_components
from ..data import FitRow
from ..data import FIT_ROW_COLUMNS
from ..data import canonical_json_sha256
from .protocol import INNER
from .protocol import PRIMARY
from ..artifacts import binding
from ..artifacts import verify
from ..problems import problem

KEYS = ["trajectory_id", "readout_id", "language"]
RESPONSES = {"adjusted_loss", "raw_nll", "nll_sum", "token_count", "unigram_log_probability_sum"}


def subset(rows, membership, split, *, scored=True):
    selected = rows[rows.trajectory_id.isin(membership[f"{split}_trajectory_ids"])].copy()
    ids = membership.get(f"{split}_readout_ids")
    # Historical whole-trajectory folds encode unrestricted readouts as [].
    if ids:
        selected = selected[selected.readout_id.isin(ids)]
    if split == "test" and scored:
        selected = selected[selected.language.isin(membership["score_languages"])]
    return selected.reset_index(drop=True)


def membership(train, test):
    if set(map(tuple, train[KEYS].to_numpy())) & set(map(tuple, test[KEYS].to_numpy())):
        raise ValueError("training and test responses overlap")
    score = sorted(set(train.language) & set(test.language))
    return dict(train_trajectory_ids=sorted(train.trajectory_id.unique()),
                test_trajectory_ids=sorted(test.trajectory_id.unique()),
                train_readout_ids=sorted(train.readout_id.unique()),
                test_readout_ids=sorted(test.readout_id.unique()), score_languages=score)


def splits(rows, folds, input_sha):
    result = {f: folds[f]["membership"] for f in PRIMARY}
    safe_ids = set.intersection(*(set(result[f]["train_trajectory_ids"]) for f in PRIMARY))
    safe = rows[rows.trajectory_id.isin(safe_ids)]
    if safe.trajectory_id.nunique() != 27 or set(safe.E) != {1,4,16} or set(safe.K) != {1,12,50}:
        raise ValueError("unexpected primary-training intersection")
    terminal = safe.groupby("trajectory_id").D_total.transform("max")
    for name, mask in zip(INNER, (safe.ladder.eq("C"),safe.E.eq(16),safe.K.eq(50),safe.D_total.eq(terminal))):
        result[name] = membership(safe[~mask],safe[mask])
        for outer in PRIMARY:
            if set(result[name]["train_trajectory_ids"]+result[name]["test_trajectory_ids"]) & set(result[outer]["test_trajectory_ids"]):
                raise ValueError("inner response entered a primary test configuration")
    result["all_grid"] = membership(rows,rows.iloc[:0])
    fitrows = [FitRow(**r) for r in rows[list(FIT_ROW_COLUMNS)].to_dict("records")]
    old = selection.build_holdout_manifest(fitrows,input_sha256=input_sha).to_dict()
    for key in ("random_holdout","lowest_loss_extrapolation"):
        entry = old[key]
        result[entry["fold_id"]] = entry
    terminal = old["terminal_continuation"]
    result["terminal"] = dict(train_trajectory_ids=terminal["trajectory_ids"],
        test_trajectory_ids=terminal["trajectory_ids"], train_readout_ids=terminal["train_readout_ids"],
        test_readout_ids=terminal["test_readout_ids"],score_languages=sorted(rows.language.unique()))
    general = selection.build_generalization_holdout_manifest(fitrows,input_sha256=input_sha,
        source_holdouts={"path":"inline:holdouts", "file_sha256":canonical_json_sha256(old),
                         "manifest_sha256":old["manifest_sha256"]})
    result["high_compute_20"] = general[selection.LARGEST_COMPUTE_VIEW]
    return result, dict(intersection_trajectories=sorted(safe_ids), rows=len(safe),
                       languages=safe.language.nunique(), repeated_target_rows=int((safe.D_t>safe.U_t).sum()),
                       secondary_holdouts=old, generalization_holdout=general)


def source_arrays(rows, universe, inventories, metadata):
    """Accept source covariates only; inventories are indexed by both K and language."""
    if RESPONSES & set(universe.columns):
        raise ValueError("source assembly accepts covariates, not responses")
    languages = sorted(metadata)
    readouts = list(dict.fromkeys(rows.readout_id))
    pivot = universe.pivot(index="readout_id",columns="language",values="D_t").reindex(
        index=readouts,columns=languages).fillna(0.)
    D = pivot.to_numpy()
    settings = universe.drop_duplicates("readout_id").set_index("readout_id").loc[readouts]
    U = np.array([[inventories.get((l,int(k)),0.) if D[i,j]>0 else 0.
                   for j,l in enumerate(languages)] for i,k in enumerate(settings.K)])
    if np.any((D>0)&(U<=0)):
        raise ValueError("positive source exposure lacks its exact K-specific inventory")
    np.testing.assert_allclose(D.sum(axis=1),settings.D_total,rtol=1e-12,atol=1e-6)
    groups = m6_composition.composition_sets(metadata)
    family = np.array([[l in groups[t]["family"] for l in languages] for t in languages],float)
    script = np.array([[l in groups[t]["script"] for l in languages] for t in languages],float)
    rmap,tmap = {r:i for i,r in enumerate(readouts)},{l:i for i,l in enumerate(languages)}
    r,t = np.array([rmap[r] for r in rows.readout_id]),np.array([tmap[l] for l in rows.language])
    if len(rows):
        np.testing.assert_allclose(D[r,t],rows.D_t,rtol=0,atol=0)
        np.testing.assert_allclose((D@family.T)[r,t],rows.D_family,rtol=1e-12,atol=1e-6)
        np.testing.assert_allclose((D@script.T)[r,t],rows.D_script,rtol=1e-12,atol=1e-6)
    return model.Sources(D,U,settings.N_act.to_numpy(),settings.N_total.to_numpy(),r,t,family,script)


def donor_covariates(rows, universe, inventories, metadata, language, donors):
    """Five-source ATLAS adaptation; donor choices supplied from training only."""
    if RESPONSES & set(universe.columns):
        raise ValueError("donor assembly accepts covariates only")
    langs = sorted(metadata)
    source = universe.pivot(index="readout_id",columns="language",values="D_t").reindex(
        index=rows.readout_id,columns=langs).fillna(0.)
    other = [l for l in langs if l not in [language,*donors]]
    inventory = np.array([[inventories.get((l,int(k)),0.) if d>0 else 0.
                          for l,d in zip(langs,row)] for row,k in zip(source.to_numpy(),rows.K)])
    if len(rows) == 0:
        inventory = np.empty((0,len(langs)))
    indices = {l:i for i,l in enumerate(langs)}
    U = np.column_stack((inventory[:,indices[language]],
        inventory[:,[indices[l] for l in donors]], inventory[:,[indices[l] for l in other]].sum(axis=1)))
    D = np.column_stack((rows.D_t,source[list(donors)],source[other].sum(axis=1)))
    np.testing.assert_allclose(D.sum(axis=1),rows.D_total,rtol=1e-12,atol=1e-6)
    family = metadata[language]["family"]
    members = [l for l in langs if l == language or (family is not None and metadata[l]["family"] == family)]
    return dict(D=D.tolist(),U=U.tolist(),N=rows.N_act.tolist(),N_total=rows.N_total.tolist(),
        E=rows.E.tolist(),K=rows.K.tolist(),family_share=(source[members].sum(axis=1).to_numpy()/rows.D_total.to_numpy()).tolist(),
        trajectory_ids=rows.trajectory_id.tolist(),readout_ids=rows.readout_id.tolist())


def native_packets(train, universe, inventories, metadata):
    _, donors = exposure_components(train)
    weights = fit.trajectory_weights(train.trajectory_id.tolist())
    packets = []
    for lang in sorted(train.language.unique()):
        idx = np.flatnonzero(train.language.eq(lang))
        rows = train.iloc[idx]
        packet = donor_covariates(rows,universe,inventories,metadata,lang,donors[lang])
        packet.update(y=rows.raw_nll.tolist(),weights=weights[idx].tolist(),training_minimum=float(rows.raw_nll.min()))
        packets.append(dict(language=lang,indices=idx.tolist(),donors=list(donors[lang]),train=packet,
                            unigram_nll=float(rows.unigram_nll.iloc[0])))
    return packets


def write_training(root, fold, train, covariates, inventories, metadata):
    path = root/"training"/fold
    path.mkdir(parents=True)
    if train.empty or train.duplicated(KEYS).any() or train.D_total.max() > 33e9:
        raise ValueError("invalid old-grid training packet")
    train.to_parquet(path/"rows.parquet",index=False)
    source = source_arrays(train,covariates,inventories,metadata)
    np.savez_compressed(path/"sources.npz",**{f.name:getattr(source,f.name) for f in fields(source)})
    io.write_json_once(path/"native.json",dict(packets=native_packets(train,covariates,inventories,metadata)))
    manifest = dict(fold=fold,rows=len(train),trajectories=train.trajectory_id.nunique(),
        languages=train.language.nunique(), training_only_responses=True,
        files=[binding(path/n) for n in ("rows.parquet","sources.npz","native.json")])
    manifest["sha256"] = canonical_json_sha256(manifest)
    io.write_json_once(path/"packet.json",manifest)
    return binding(path/"packet.json")


def read_training(entry):
    packet_path = verify(entry)
    packet = io.read_json(packet_path)
    unsigned = dict(packet); digest = unsigned.pop("sha256")
    if canonical_json_sha256(unsigned) != digest or not packet["training_only_responses"]:
        raise ValueError("training packet contract changed")
    # Relative bindings allow a training packet to move with its data files.
    paths = {Path(e["path"]).name:verify(e | {"path": str(packet_path.parent / e["path"])})
             for e in packet["files"]}
    rows = pd.read_parquet(paths["rows.parquet"])
    with np.load(paths["sources.npz"],allow_pickle=False) as arrays:
        sources = model.Sources(**{f.name:arrays[f.name] for f in fields(model.Sources)})
    return rows,sources,io.read_json(paths["native.json"])["packets"]


def make_model(name, rows, sources, packets, regularization, tau=0., delta=None):
    prepared = []
    for item in packets:
        packet = dict(item["train"]); packet["delta"] = delta
        prepared.append((np.asarray(item["indices"],int),packet))
    return model.Model(name,problem(rows,regularization),rows.U_t.to_numpy(),sources,tau,prepared)
