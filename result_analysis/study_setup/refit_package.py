"""Portable observations from a sealed paper refit; no optimization or analysis outputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from multilingual_megatron.contracts.hashing import canonical_json_sha256
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_comparison_workflow import model as native
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow import inputs, protocol
from multilingual_megatron.experiments.scaling.stage1_fit.paper_refit_workflow.seal import GATES

TABLES = ("fit_predictions", "fit_parameters", "fit_status", "fit_membership")
CELL = ["model", "fold", "arm"]
KEYS = ["trajectory_id", "readout_id", "language"]
CAMPAIGN = "paper_refit"


def sealed(export, path, expected=None):
    value = export.json(path, expected)
    if value.get("sha256") != canonical_json_sha256({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"Invalid refit seal: {path}")
    return value


def history(frame):
    """Keep earlier experiments explicitly identified; never relabel their optimizer."""
    return frame.assign(**{k: value for k, value in
        {"fit_revision": "historical", "arm": "historical", "fit_campaign": "historical"}.items()
        if k not in frame})


def parameters(entry, packets, revision, source_hash):
    result = entry["result"]
    identity = {k: entry[k] for k in CELL}
    identity.update(fit_campaign=CAMPAIGN, fit_revision=revision, source=entry["receipt"],
                    source_sha256=source_hash)
    rows = []
    if "groups" in result:
        for group in result["groups"]:
            spec = native.Spec(**group["spec"])
            lo, hi = native.bounds(spec)
            context = dict(language=group["language"], scope="native_language",
                model_spec_json=json.dumps(group["spec"], sort_keys=True), donors=group["donors"],
                training_minimum=group["training_minimum"], unigram_nll=group["unigram_nll"],
                huber_delta=packets[group["language"]]["train"]["delta"])
            for i, (name, value) in enumerate(zip(native.names(spec), group["selected"]["coordinates"], strict=True)):
                rows.append(identity | context | dict(parameter=name, coordinate_index=i,
                    value=float(value), lower_bound=float(lo[i]), upper_bound=float(hi[i])))
    else:
        selected = result["selected"]
        for i, (value, bound) in enumerate(zip(selected["coordinates"], result["bounds"], strict=True)):
            rows.append(identity | dict(language=None, scope="shared", parameter=f"coordinate_{i}",
                coordinate_name=bound["name"], coordinate_index=i, value=float(value),
                lower_bound=bound["lower"], upper_bound=bound["upper"]))
        for local in selected["language_coefficients"]:
            context = dict(language=local["language"], scope="language")
            if entry["model"].startswith("Pooled-"):
                item = packets[local["language"]]
                context.update(donors=item["donors"], training_minimum=item["train"]["training_minimum"],
                               unigram_nll=item["unigram_nll"])
            for name, value in local.items():
                if name != "language":
                    rows.append(identity | context | dict(parameter=name, coordinate_index=None, value=float(value)))
    return rows


def export_refit(export, baseline):
    """Append the adopted campaign to historical fits after checking its complete lineage."""
    cfg = export.config["fit_update"]
    seal_path = Path(cfg["path"]).resolve()
    root = seal_path.parent
    release = sealed(export, seal_path, cfg["sha256"])
    bound = lambda entry: export.source(root / entry["path"], entry["sha256"])
    report = sealed(export, bound(release["report"]))
    plan = sealed(export, root / "plan.json")
    review = export.json(bound(release["review"]))
    if (report["plan_sha256"] != plan["sha256"] or release["plan_sha256"] != plan["sha256"]
            or release["report_sha256"] != report["sha256"] or review["report_sha256"] != report["sha256"]
            or not report["all_cells_certified"] or not report["all_scoring_complete"]
            or set(review["gates"]) != set(GATES)
            or not review["story_unchanged"] or not all(g["passed"] for g in review["gates"].values())):
        raise ValueError("Incomplete or mismatched paper adoption")
    for entry in release["evidence"] + report["outputs"]:
        bound(entry)
    # Verify the certified winners and the frozen inputs used for their coefficients.
    receipt_hashes = {e["path"]: e["sha256"] for e in release["receipts"]}
    input_path = bound(plan["input"])
    manifest = sealed(export, input_path)
    input_file = lambda e: export.source(input_path.parent / e["path"], e["sha256"])
    grid = export.read(input_file(manifest["scoring"]["grid.parquet"]))
    endpoint = export.read(input_file(manifest["scoring"]["endpoint.parquet"]))
    # This refit must target the exact measured observations already in the package.
    for name, measured in (("main_grid", grid), ("large_model", endpoint)):
        saved = pd.read_parquet(baseline / f"{name}.parquet")
        if name == "main_grid":
            saved = saved[saved.metric_role.eq("trained")]
        a, b = saved.set_index(KEYS).sort_index(), measured.set_index(KEYS).sort_index()
        pd.testing.assert_index_equal(a.index, b.index, exact=False)
        for c in ("adjusted_loss", "D_t", "D_total", "N_act", "N_total", "U_t"):
            np.testing.assert_allclose(a[c], b[c], rtol=0, atol=1e-12, err_msg=f"changed {name}/{c}")
    report_root = bound(release["report"]).parent
    fits = export.json(report_root / "coefficients.json")["fits"]
    predicted = export.read(report_root / "predictions.parquet")
    metrics = export.read(report_root / "metrics.csv")
    expected = {(t["model"], t["fold"], t["arm"]) for t in plan["tasks"] if t["stage"] == "certify"}
    actual = {tuple(f[k] for k in CELL) for f in fits}
    if actual != expected or len(fits) != len(expected) or len(fits) != report["fit_cells"]:
        raise ValueError("Refit cell inventory differs from plan")
    revision = cfg["revision"]
    predictions = predicted.assign(fit_campaign=CAMPAIGN, fit_revision=revision)
    if predictions.duplicated(CELL + ["case"] + KEYS).any() or not np.isfinite(predictions.prediction).all():
        raise ValueError("Duplicate or invalid refit predictions")
    if set(map(tuple, predicted[CELL].drop_duplicates().to_numpy())) != expected:
        raise ValueError("Missing prediction cells")
    param_rows, status_rows = [], []
    prediction_groups = {key: part for key, part in predicted.groupby(CELL + ["case"])}
    metric_groups = {key: part for key, part in metrics.groupby(CELL)}
    packets, calibrations = {}, {}
    for fold, entry in manifest["training"].items():
        packet_path = input_file(entry["packet"])
        packet = sealed(export, packet_path)
        for e in packet["files"]:
            export.source(packet_path.parent / e["path"], e["sha256"])
        _, _, items, calibration = inputs.read_training(input_path, fold)
        packets[fold] = {item["language"]: item for item in items}
        calibrations[fold] = calibration
    for entry in fits:
        result = sealed(export, root / entry["receipt"], receipt_hashes[entry["receipt"]])
        if result != entry["result"] or not result["certified"] or not result["complete"]:
            raise ValueError("Uncertified or changed coefficient source")
        if result["plan_sha256"] != plan["sha256"] or any(result["task"][k] != entry[k] for k in CELL):
            raise ValueError("Coefficients belong to a different refit cell")
        row = {k: entry[k] for k in CELL}
        row.update(fit_campaign=CAMPAIGN, fit_revision=revision, optimizer=protocol.contract(entry["arm"])["optimizer"],
            source=entry["receipt"], source_sha256=receipt_hashes[entry["receipt"]],
            certified=True, complete=True, role="adopted" if entry["arm"] == "lbfgsb_box" else "optimizer control",
            positive_effective_exponents=result.get("positive_effective_exponents"),
            huber_delta=result.get("delta"), regularization_json=json.dumps(result.get("regularization"), sort_keys=True),
            equalities_json=json.dumps(result.get("equalities", [])),
            numerical_policy_json=json.dumps(protocol.contract(entry["arm"]), sort_keys=True),
            calibration_json=json.dumps(calibrations[entry["fold"]], sort_keys=True))
        checks = result.get("groups", [result])
        audit_optimizers = {str(x["audit"].get("solver", x["audit"].get("optimizer", row["optimizer"]))) for x in checks}
        if result.get("audit_recovery"):
            audit_optimizers = {result["audit_recovery"][-1]["optimizer"]}
        row.update(selected_kkt=max(x["selected"]["kkt"] for x in checks),
            audit_kkt=max(x["audit"]["kkt"] for x in checks),
            audit_optimizer=",".join(sorted(audit_optimizers)))
        param_rows.extend(parameters(entry, packets[entry["fold"]], revision, row["source_sha256"]))
        cell = tuple(entry[k] for k in CELL)
        cases = metric_groups[cell]
        if cases.empty or not cases.complete.all() or not cases.certified.all():
            raise ValueError("Incomplete scoring status")
        for record in cases.to_dict("records"):
            part = prediction_groups[(*cell, record["case"])]
            if len(part) != record["rows"] or record["rows"] != record["expected_rows"]:
                raise ValueError("Scoring population differs from receipt")
            status_rows.append(row | {k: record[k] for k in ("case", "rows", "expected_rows")})
    members = pd.DataFrame([dict(fit_campaign=CAMPAIGN, fit_revision=revision, fold=fold,
        membership_kind=kind, member=str(member)) for fold, membership in manifest["memberships"].items()
        for kind, values in membership.items() for member in values])
    frames = dict(fit_predictions=predictions, fit_parameters=pd.DataFrame(param_rows),
                  fit_status=pd.DataFrame(status_rows), fit_membership=members)
    old = {name: history(pd.read_parquet(baseline / f"{name}.parquet")) for name in TABLES}
    old = {name: frame[~frame.fit_revision.eq(revision)].copy() for name, frame in old.items()}
    # Original primary membership remains the same; new readout/inner memberships are additive.
    membership_keys = ["fold", "membership_kind", "member"]
    joined = old["fit_membership"][membership_keys].merge(members[membership_keys], how="left", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Historical primary fold membership changed")
    for name, frame in frames.items():
        combined = pd.concat([old[name], frame], ignore_index=True, sort=False)
        export.write(name, combined, {
            "fit_predictions": "one revision x optimizer arm x fitted model x fold x scored observation",
            "fit_parameters": "one revision x optimizer arm x model x fold x shared or language parameter",
            "fit_status": "fit/scoring certification and numerical protocol by revision, model, fold, arm and case",
            "fit_membership": "exact training/test readout, trajectory and scoring-language membership by revision and fold",
        }[name])
    export.fit_revision = dict(revision=revision, fit_campaign=CAMPAIGN, default_arm="lbfgsb_box",
        release_seal=cfg, plan_sha256=plan["sha256"], report_sha256=report["sha256"],
        exporter_sha256=export.sources[str(export.source(Path(__file__)))],
        models=len(plan["models"]), certified_cells=len(fits), prediction_rows=len(predicted),
        historical_rows_retained={name: len(frame) for name, frame in old.items()},
        files={str(Path(p).relative_to(root)): h for p, h in export.sources.items() if Path(p).is_relative_to(root)},
        historical_selection="RN13 remains the historical inner-selected form; refitting does not rerun selection.")
