"""Fit-problem construction and exposure inventory checks; no workflow imports."""
import numpy as np
from . import fit


def inventories(packets):
    values = {}
    for packet in packets:
        for k, u in zip(packet["support"]["K"], packet["support"]["U"]):
            key, value = (packet["language"], int(k)), float(u[0])
            if not np.isfinite(value) or value <= 0 or (key in values and values[key] != value):
                raise ValueError("missing or inconsistent unique target inventory")
            values[key] = value
    return values


def partition(rows, membership, split):
    train, test = set(membership["train_trajectory_ids"]), set(membership["test_trajectory_ids"])
    if train & test:
        raise ValueError("primary trajectory split overlaps")
    selected = rows[rows.trajectory_id.isin(train if split == "train" else test)].copy()
    if split == "test":
        selected = selected[selected.language.isin(membership["score_languages"])]
    if selected.empty:
        raise ValueError("empty training/test partition")
    if split == "train" and selected.D_total.max() > 33e9:
        raise ValueError("long-run endpoint entered training")
    return selected.reset_index(drop=True)


def problem(rows, regularization):
    return fit.ProjectedProblem(y=rows.adjusted_loss.to_numpy(),
        weights=fit.trajectory_weights(rows.trajectory_id.tolist()),
        N=rows.N_act.to_numpy(), D=rows.D_t.to_numpy(), E=rows.E.to_numpy(), K=rows.K.to_numpy(),
        language_ids=tuple(rows.language), physical_floors=-rows.unigram_nll.to_numpy(),
        D_family=rows.D_family.to_numpy(), D_script=rows.D_script.to_numpy(),
        D_other=rows.D_other.to_numpy(), regularization=fit.M5M6Regularization.from_mapping(regularization))


def counts(configs):
    result, evidence = {}, []
    for row in configs.itertuples():
        h, v, layers, e = int(row.hidden_size), int(row.vocab_size), int(row.num_layers), int(row.experts)
        total = 2*h*v+(4+9*e)*layers*h*h
        active = 2*h*v+13*layers*h*h
        if total != row.configured_design_total_parameters or active != row.declared_active_parameters:
            raise ValueError("configured parameter-count formula does not reproduce release")
        if e > 1 and (row.moe_router_topk != 1 or row.ffn_hidden_size != 3*h):
            raise ValueError("parameter sensitivity only covers fixed top-1, FFN=3H design")
        key = (str(row.ladder), e)
        if key in result:
            raise ValueError("duplicate architecture count key")
        result[key] = total
        evidence.append(dict(ladder=key[0], E=e, active=active, total=total,
                             scope="configured design; not runtime-exact norm/router-inclusive count"))
    return result, evidence
