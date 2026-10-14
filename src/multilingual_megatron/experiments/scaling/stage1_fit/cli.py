"""Thin command line for the lean Stage 1 fitter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import (DEFAULT_ARCHIVED_MANIFEST_PATH, DEFAULT_RELEASE_ROOT, FitRow,
    build_release_contract, load_fit_rows_snapshot, prepare_fit_data, sha256_file,
    write_release_contract)
from .execution import (bind_membership, build_manifest, execute_task, load_manifest, read_json, reduce,
    render_slurm, run_serial, write_json_once, write_manifest, write_text_once)
from .campaign import (audit_grid, plan_candidate_campaign, plan_d16_refits,
    plan_generalization_comparison, plan_literature_comparison, plan_secondary_diagnostic, plan_selected_all_data)
from .amendment import plan_warm_audit_amendment, run_warm_audit_amendment
from .fit import OuterBound
from .report import (build_candidate_report, build_d16_report, build_literature_comparison_report,
    build_generalization_report, build_secondary_diagnostics_report, execution_summary, write_candidate_report,
    write_d16_report, write_generalization_report, write_literature_comparison_report,
    write_secondary_diagnostics_report)
from .selection import build_generalization_holdout_manifest, build_holdout_manifest, build_resampling_manifest

def _print(value: Any) -> None: print(json.dumps(value,sort_keys=True,allow_nan=False))
def _freeze_release(args: argparse.Namespace) -> dict[str, Any]:
    contract = build_release_contract(args.release_root, args.archive_manifest)
    result = {**contract.to_dict(), 'written': False}
    if not args.dry_run:
        path = write_release_contract(contract, args.output)
        result.update({'path': str(path), 'file_sha256': sha256_file(path), 'written': True})
    return result
def _prepare(args: argparse.Namespace) -> dict[str,Any]:
    prepared,receipt=prepare_fit_data(args.release_root,archive_manifest_path=args.archive_manifest,
        release_contract_path=getattr(args,'release_contract',None),output_root=args.output_root,dry_run=args.dry_run)
    result={"fit_rows":len(prepared.rows),"trajectories":prepared.trajectory_count,
            "readouts":prepared.readout_count,"fit_rows_content_sha256":prepared.content_sha256,"written":receipt is not None}
    if receipt is not None:
        holdouts=build_holdout_manifest(prepared.rows,input_sha256=prepared.content_sha256)
        trajectories=sorted({row.trajectory_id for row in prepared.rows})
        resampling=build_resampling_manifest(trajectories,input_sha256=prepared.content_sha256)
        holdout_path=Path(args.output_root)/"input"/"holdouts.json"
        resampling_path=Path(args.output_root)/"input"/"resampling.json"
        write_json_once(holdout_path,holdouts.to_dict()); write_json_once(resampling_path,resampling.to_dict())
        result.update({"holdouts_path":str(holdout_path),"resampling_path":str(resampling_path)})
    return result

def _prepare_generalization(args: argparse.Namespace) -> dict[str, Any]:
    receipt = read_json(args.input_receipt); frame = load_fit_rows_snapshot(receipt); bind_membership(args.holdouts, kind='all')
    holdouts = read_json(args.holdouts)
    if holdouts.get('input_sha256') != receipt['fit_rows_content_sha256']: raise ValueError('holdouts belong to different fit rows')
    source = {'path': str(args.holdouts.resolve()), 'file_sha256': sha256_file(args.holdouts), 'manifest_sha256': holdouts['manifest_sha256']}
    rows = tuple(FitRow(**record) for record in frame.to_dict('records'))
    manifest = build_generalization_holdout_manifest(rows, input_sha256=receipt['fit_rows_content_sha256'], source_holdouts=source)
    result = {'manifest_sha256': manifest['manifest_sha256'], **manifest['logical_compute'], 'written': False}
    if not args.dry_run:
        path = args.output_root / 'input' / 'generalization_holdouts.json'; write_json_once(path, manifest)
        result.update({'path': str(path), 'written': True})
    return result

def _profile(path: Path) -> dict[str,Any]:
    value=json.loads(path.read_text());
    if not isinstance(value,dict): raise ValueError("profile must be an object")
    return value

def _plan(args: argparse.Namespace) -> dict[str,Any]:
    if not args.review_only:
        raise ValueError("generic plan is review-only; use plan-candidate, plan-all-data, or plan-d16 for definitive work")
    receipt=read_json(args.input_receipt); profile=_profile(args.profile)
    objective=args.objective; delta=None
    if objective=="huber":
        if args.parent is None: raise ValueError("Huber planning requires its frozen M0-OLS parent result")
        parent=read_json(args.parent); scale=parent.get("adaptive_huber_scale")
        identity=parent.get("fit_identity",{})
        if parent.get("fit_rows_content_sha256")!=receipt["fit_rows_content_sha256"] or identity.get("fold_id")!=args.fold:
            raise ValueError("Huber parent belongs to different input rows or fold")
        if isinstance(scale,dict) and (identity.get("model"),identity.get("objective")) != ("M0","ols"):
            raise ValueError("adaptive Huber scale must come from an M0-OLS parent")
        delta=scale.get("delta") if isinstance(scale,dict) else parent.get("frozen_delta")
        if delta is None: objective="ols"
    bounds=None
    if args.bounds:
        raw=json.loads(args.bounds.read_text()); bounds=tuple(OuterBound(**item) for item in raw)
    manifest=build_manifest(output_root=Path(profile["output_root"]),fit_rows_path=Path(receipt["fit_rows_path"]),
        fit_rows_content_sha256=receipt["fit_rows_content_sha256"],phase=args.phase,model=args.model,
        fold_id=args.fold,objective=objective,delta=delta,chunk_size=profile["chunk_size"],bounds=bounds,
        bound_version=args.bound_version,replicate_id=args.replicate,parent_result=args.parent,start_total=args.start_count,
        review_only=True,
        blas_threads=profile["blas_threads"],partition=profile.get("partition"),account=profile.get("account"),
        reservation=profile.get("reservation"),memory=profile["memory"],time_limit=profile["time_limit"],
        concurrency=profile["array_concurrency"])
    if args.dry_run: return {"manifest_sha256":manifest["manifest_sha256"],"tasks":len(manifest["tasks"]),"written":False}
    path=write_manifest(manifest); return {"manifest_sha256":manifest["manifest_sha256"],"tasks":len(manifest["tasks"]),"path":str(path),"written":True}

def _completed(path: Path | None) -> dict[str,Path]:
    if path is None: return {}
    value=read_json(path)
    if set(value)!={"completed"} or not isinstance(value["completed"],dict):
        raise ValueError("completed index must contain exactly a completed object")
    return {str(key):Path(str(item)) for key,item in value["completed"].items()}

def _campaign(args: argparse.Namespace) -> dict[str,Any]:
    profile=_profile(args.profile); completed=_completed(args.completed)
    if args.command=="plan-candidate":
        planned=plan_candidate_campaign(input_receipt_path=args.input_receipt,holdouts_path=args.holdouts,profile=profile,completed=completed)
    elif args.command=="plan-all-data":
        planned=plan_selected_all_data(candidate_report_path=args.candidate_report,input_receipt_path=args.input_receipt,holdouts_path=args.holdouts,profile=profile,completed=completed)
    elif args.command=="plan-d16":
        planned=plan_d16_refits(candidate_report_path=args.candidate_report,input_receipt_path=args.input_receipt,resampling_path=args.resampling,profile=profile,completed=completed)
    elif args.command=="plan-generalization":
        planned=plan_generalization_comparison(candidate_report_path=args.candidate_report,input_receipt_path=args.input_receipt,
            holdouts_path=args.holdouts,generalization_holdouts_path=args.generalization_holdouts,profile=profile,completed=completed)
    elif args.command=="plan-literature":
        planned=plan_literature_comparison(candidate_report_path=args.candidate_report,
            input_receipt_path=args.input_receipt,holdouts_path=args.holdouts,profile=profile,completed=completed)
    else:
        planned=plan_secondary_diagnostic(diagnostic=args.diagnostic,candidate_report_path=args.candidate_report,input_receipt_path=args.input_receipt,holdouts_path=args.holdouts,profile=profile,completed=completed)
    return {"planned":{key:str(value) for key,value in planned.items()},"count":len(planned),"submitted":False}

def _warm_amendment(args: argparse.Namespace) -> Mapping[str,Any]:
    if args.command=="plan-warm-amendment":
        plan,path,script=plan_warm_audit_amendment(grid_manifest_path=args.grid_manifest,
            warm_manifest_path=args.warm_manifest,profile=_profile(args.profile))
        return {"plan_sha256":plan["plan_sha256"],"plan_path":str(path),"script_path":str(script),
            "submitted":False}
    return run_warm_audit_amendment(args.plan)

def _task(args: argparse.Namespace) -> dict[str,Any]:
    manifest=load_manifest(args.manifest); index=args.task_index
    if index is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ: raise ValueError("task index or SLURM_ARRAY_TASK_ID is required")
        index=int(os.environ["SLURM_ARRAY_TASK_ID"])
    return execute_task(manifest,index)

def _report(args: argparse.Namespace) -> dict[str,Any]:
    indices=[args.candidate_index,args.secondary_index,args.d16_index,args.literature_index,args.generalization_index]
    if sum(item is not None for item in indices)>1: raise ValueError("report indices are mutually exclusive")
    if args.candidate_index is not None:
        index=read_json(args.candidate_index)
        results={model:{fold:Path(path) for fold,path in folds.items()} for model,folds in index["results"].items()}
        report=build_candidate_report(input_receipt_path=Path(index["input_receipt_path"]),
            holdouts_path=Path(index["holdouts_path"]),results=results)
        paths=write_candidate_report(Path(index["output_root"]),report)
        return {"report_sha256":report["report_sha256"],"paths":[str(path) for path in paths]}
    if args.secondary_index is not None:
        index=read_json(args.secondary_index)
        report=build_secondary_diagnostics_report(candidate_report_path=Path(index["candidate_report_path"]),
            input_receipt_path=Path(index["input_receipt_path"]),holdouts_path=Path(index["holdouts_path"]),
            selected_results={key:Path(value) for key,value in index["selected_results"].items()},
            lowest_results={key:Path(value) for key,value in index["lowest_results"].items()})
        path=write_secondary_diagnostics_report(Path(index["output_root"]),report)
        return {"report_sha256":report["report_sha256"],"path":str(path)}
    if args.d16_index is not None:
        index=read_json(args.d16_index)
        replicas={int(key):Path(value) for key,value in index["replicate_results"].items()}
        report=build_d16_report(full_result=Path(index["full_result"]),
            resampling_path=Path(index["resampling_path"]),replicate_results=replicas,
            prediction_points=index["prediction_points"])
        path=write_d16_report(Path(index["output_root"]),report)
        return {"report_sha256":report["report_sha256"],"path":str(path)}
    if args.literature_index is not None:
        index=read_json(args.literature_index)
        results={view:{model:Path(path) for model,path in models.items()}
            for view,models in index["results"].items()}
        report=build_literature_comparison_report(
            candidate_report_path=Path(index["candidate_report_path"]),
            input_receipt_path=Path(index["input_receipt_path"]),
            holdouts_path=Path(index["holdouts_path"]),
            logical_compute_contract_path=Path(index["logical_compute_contract_path"]), results=results)
        path=write_literature_comparison_report(Path(index["output_root"]),report)
        return {"report_sha256":report["report_sha256"],"path":str(path)}
    if args.generalization_index is not None:
        index=read_json(args.generalization_index)
        results={view:{model:Path(path) for model,path in models.items()} for view,models in index["results"].items()}
        report=build_generalization_report(candidate_report_path=Path(index["candidate_report_path"]),
            input_receipt_path=Path(index["input_receipt_path"]),holdouts_path=Path(index["holdouts_path"]),
            generalization_holdouts_path=Path(index["generalization_holdouts_path"]),results=results)
        path=write_generalization_report(Path(index["output_root"]),report); return {"report_sha256":report["report_sha256"],"path":str(path)}
    if args.manifest is None: raise ValueError("report requires a manifest or report index")
    return execution_summary(args.manifest)

def parser() -> argparse.ArgumentParser:
    root=argparse.ArgumentParser(); commands=root.add_subparsers(dest="command",required=True)
    freeze=commands.add_parser("freeze-release"); freeze.add_argument("--release-root",type=Path,required=True); freeze.add_argument("--archive-manifest",type=Path,required=True); freeze.add_argument("--output",type=Path,required=True); freeze.add_argument("--dry-run",action="store_true")
    prepare=commands.add_parser("prepare-data"); prepare.add_argument("--release-root",type=Path,default=DEFAULT_RELEASE_ROOT); prepare.add_argument("--archive-manifest",type=Path,default=DEFAULT_ARCHIVED_MANIFEST_PATH); prepare.add_argument("--release-contract",type=Path); prepare.add_argument("--output-root",type=Path); prepare.add_argument("--dry-run",action="store_true")
    generalization=commands.add_parser("prepare-generalization"); generalization.add_argument("--input-receipt",type=Path,required=True); generalization.add_argument("--holdouts",type=Path,required=True); generalization.add_argument("--output-root",type=Path,required=True); generalization.add_argument("--dry-run",action="store_true")
    plan=commands.add_parser("plan"); plan.add_argument("--profile",type=Path,required=True); plan.add_argument("--input-receipt",type=Path,required=True); plan.add_argument("--phase",required=True); plan.add_argument("--model",choices=("C0","J0","JH","M0","M1","M2"),required=True); plan.add_argument("--fold",required=True); plan.add_argument("--objective",choices=("ols","huber"),required=True); plan.add_argument("--bounds",type=Path); plan.add_argument("--bound-version",type=int,default=0); plan.add_argument("--replicate",type=int); plan.add_argument("--parent",type=Path); plan.add_argument("--start-count",type=int); plan.add_argument("--review-only",action="store_true"); plan.add_argument("--dry-run",action="store_true")
    for name in ("plan-candidate","plan-all-data","plan-d16","plan-secondary","plan-literature","plan-generalization"):
        item=commands.add_parser(name); item.add_argument("--profile",type=Path,required=True); item.add_argument("--input-receipt",type=Path,required=True); item.add_argument("--completed",type=Path)
        if name in {"plan-candidate","plan-all-data","plan-secondary","plan-literature","plan-generalization"}: item.add_argument("--holdouts",type=Path,required=True)
        if name=="plan-d16": item.add_argument("--resampling",type=Path,required=True)
        if name!="plan-candidate": item.add_argument("--candidate-report",type=Path,required=True)
        if name=="plan-secondary": item.add_argument("--diagnostic",choices=("random","lowest","terminal"),required=True)
        if name=="plan-generalization": item.add_argument("--generalization-holdouts",type=Path,required=True)
    audit=commands.add_parser("audit"); audit.add_argument("--manifest",type=Path,required=True)
    amendment=commands.add_parser("plan-warm-amendment"); amendment.add_argument("--grid-manifest",type=Path,required=True); amendment.add_argument("--warm-manifest",type=Path,required=True); amendment.add_argument("--profile",type=Path,required=True)
    run_amendment=commands.add_parser("run-warm-amendment"); run_amendment.add_argument("--plan",type=Path,required=True)
    task=commands.add_parser("run-task"); task.add_argument("--manifest",type=Path,required=True); task.add_argument("--task-index",type=int)
    reduce_p=commands.add_parser("reduce"); reduce_p.add_argument("--manifest",type=Path,required=True)
    serial=commands.add_parser("run-serial"); serial.add_argument("--manifest",type=Path,required=True)
    report=commands.add_parser("report"); report.add_argument("--manifest",type=Path); report.add_argument("--candidate-index",type=Path); report.add_argument("--secondary-index",type=Path); report.add_argument("--d16-index",type=Path); report.add_argument("--literature-index",type=Path); report.add_argument("--generalization-index",type=Path)
    slurm=commands.add_parser("emit-slurm"); slurm.add_argument("--manifest",type=Path,required=True); slurm.add_argument("--output",type=Path)
    return root

def main(argv: Sequence[str]|None=None) -> int:
    args=parser().parse_args(argv)
    if args.command=="freeze-release": result=_freeze_release(args)
    elif args.command=="prepare-data": result=_prepare(args)
    elif args.command=="prepare-generalization": result=_prepare_generalization(args)
    elif args.command=="plan": result=_plan(args)
    elif args.command in {"plan-candidate","plan-all-data","plan-d16","plan-secondary","plan-literature","plan-generalization"}: result=_campaign(args)
    elif args.command in {"plan-warm-amendment","run-warm-amendment"}: result=_warm_amendment(args)
    elif args.command=="audit": result=audit_grid(args.manifest)
    elif args.command=="run-task": result=_task(args)
    elif args.command=="reduce": result=reduce(load_manifest(args.manifest))
    elif args.command=="run-serial": result=run_serial(load_manifest(args.manifest))
    elif args.command=="report": result=_report(args)
    else:
        manifest=load_manifest(args.manifest); script,launch=render_slurm(args.manifest,manifest)
        output=args.output or Path(manifest["execution"]["output_root"])/"manifests"/f"slurm-{manifest['manifest_sha256'][:20]}.sbatch"
        write_text_once(output,script); result={"path":str(output),"launch_command":launch,"submitted":False}
    _print(result); return 0

if __name__=="__main__": raise SystemExit(main())
