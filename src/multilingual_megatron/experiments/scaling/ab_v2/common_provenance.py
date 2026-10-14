"""Hash-bound source and Git provenance for A/B-v2 manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .common import (
    LANGUAGE_COUNTS,
    REPO_ROOT,
    WorkflowError,
    _abs,
    _file_sha256,
    _language_map,
    _rel_or_abs,
)


def _config_inheritance_chain(path: Path) -> tuple[Path, ...]:
    """Return a config and every parent whose contents affect its resolution."""

    chain: list[Path] = []
    current = path.resolve()
    while True:
        if current in chain:
            raise WorkflowError(f"JSON inheritance cycle at {current}")
        chain.append(current)
        try:
            payload = json.loads(current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot read inherited config {current}: {exc}") from exc
        inherited = payload.get("inherits") if isinstance(payload, Mapping) else None
        if inherited is None:
            return tuple(chain)
        if isinstance(inherited, Mapping):
            inherited = inherited.get("spec_path")
        if not isinstance(inherited, str) or not inherited.strip():
            raise WorkflowError(f"invalid JSON inheritance path in {current}")
        parent = Path(inherited)
        current = (
            parent.resolve()
            if parent.is_absolute()
            else (current.parent / parent).resolve()
        )


def _workflow_source_files() -> tuple[Path, ...]:
    """Return every module whose code can affect an A/B-v2 worker."""

    implementation_root = Path(__file__).resolve().parent
    return tuple(sorted(implementation_root.rglob("*.py")))


def _fineweb_bootstrap_source_files() -> tuple[Path, ...]:
    """Return the compatibility facade and complete bootstrap implementation."""

    tools_root = _abs(
        "src/multilingual_megatron/pipeline/stages/sampling/tools"
    )
    facade = tools_root / "bootstrap_fineweb_v2.py"
    implementation_root = tools_root / "fineweb_v2_bootstrap"
    return (facade, *sorted(implementation_root.rglob("*.py")))


def _contract_files(spec_path: Path, runtime_path: Path) -> tuple[Path, ...]:
    # Variants may own ladder-specific adapters outside this package (C/D).
    # Their source files must be frozen alongside the shared implementation.
    from .variant import variant_input_files

    return (
        *_config_inheritance_chain(spec_path),
        *_config_inheritance_chain(runtime_path),
        *_workflow_source_files(),
        *(path.resolve() for path in variant_input_files()),
        _abs("configs/data_sources/fineweb2_v2_1_1_source_lock.json"),
        *(_abs(_language_map(k)) for k in LANGUAGE_COUNTS),
        *_fineweb_bootstrap_source_files(),
        _abs("src/multilingual_megatron/pipeline/stages/sampling/fineweb_v2_policy.py"),
        _abs(
            "src/multilingual_megatron/pipeline/stages/sampling/tools/materialize_fineweb_v2_source_lock.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/sampling/tools/calibrate_fineweb_v2_tokens.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/sampling/tools/build_fineweb_v2_sampling_plan.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/sampling/tools/finalize_fineweb_v2_corpus.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/tokenizer_training/build_sentence_artifact.py"
        ),
        _abs("src/multilingual_megatron/pipeline/tokenizer_sentence_contract.py"),
        _abs(
            "src/multilingual_megatron/pipeline/stages/tokenizer_training/plan_sentence_acquisition.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/tokenizer_training/sentencepiece_v2.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/tokenizer_training/train_hf_tokenizer.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/tokenization/fineweb_v2_presplit.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/tokenization/tools/run_megatron_preprocess.py"
        ),
        *sorted(_abs("src/multilingual_megatron/experiments/scaling/k12_migration").glob("*.py")),
        _abs(
            "src/multilingual_megatron/pipeline/backends/megatron_bridge/scaling_registry.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/sampling/tools/multilingual_moe_scaling.py"
        ),
        _abs("src/multilingual_megatron/pipeline/executor.py"),
        _abs("src/multilingual_megatron/pipeline/stages/training/batch_policy.py"),
        _abs("src/multilingual_megatron/pipeline/stages/training/planner.py"),
        _abs("src/multilingual_megatron/pipeline/stages/training/segment_runner.py"),
        _abs("src/multilingual_megatron/pipeline/stages/training/checkpoint_state.py"),
        _abs("src/multilingual_megatron/pipeline/stages/training/metrics.py"),
        _abs(
            "src/multilingual_megatron/pipeline/stages/final_loss/per_language_checkpoint_eval.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/final_loss/validation_metrics.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/stages/final_loss/moe_validation_diagnostics.py"
        ),
        _abs("src/multilingual_megatron/pipeline/stages/audit/fineweb_v2_release.py"),
        *tuple(sorted(_abs("configs/pipeline/defaults").glob("*.json"))),
        _abs("configs/pipeline/schema.json"),
        _abs("configs/scaling/h100_ab_ladder_launch_readiness.yaml"),
        _abs(
            "src/multilingual_megatron/pipeline/backends/megatron_bridge/study_bridge_recipes.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/backends/megatron_bridge/run_megatron_bridge_training.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/backends/megatron_bridge/run_study_bridge_recipe.py"
        ),
        _abs(
            "src/multilingual_megatron/pipeline/backends/megatron_bridge/run_megatron_bridge_per_language_validation.py"
        ),
        _abs("external/Megatron-Bridge/src/megatron/bridge/training/eval.py"),
        _abs(
            "external/Megatron-Bridge/src/megatron/bridge/training/utils/moe_token_drop_metrics.py"
        ),
        _abs("external/Megatron-Bridge/src/megatron/bridge/training/setup.py"),
        _abs(
            "external/Megatron-Bridge/src/megatron/bridge/training/utils/train_utils.py"
        ),
        _abs("external/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py"),
    )


def _fixed_hashes(spec_path: Path, runtime_path: Path) -> dict[str, str]:
    return {
        _rel_or_abs(path): _file_sha256(path)
        for path in _contract_files(spec_path, runtime_path)
    }


def _git_snapshot(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            raise WorkflowError(
                f"git provenance command failed in {root}: {' '.join(arguments)}: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        return completed.stdout

    commit = run("rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = run("diff", "--binary", "HEAD", "--")
    return {
        "root": str(root.resolve()),
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _git_provenance() -> dict[str, Any]:
    return {
        "orchestration_repo": _git_snapshot(REPO_ROOT),
        "megatron_bridge_submodule": _git_snapshot(_abs("external/Megatron-Bridge")),
        "policy": (
            "Git identities are recorded for audit but do not gate execution; "
            "fixed_input_sha256 is the execution-integrity authority"
        ),
    }


def _tracked_git_provenance_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only execution-relevant Git state."""

    identity: dict[str, Any] = {}
    for name in ("orchestration_repo", "megatron_bridge_submodule"):
        raw = payload.get(name)
        if not isinstance(raw, Mapping):
            raise WorkflowError(f"git provenance is missing {name}")
        identity[name] = {
            "root": raw.get("root"),
            "commit": raw.get("commit"),
            "tracked_diff_sha256": raw.get("tracked_diff_sha256"),
        }
    return identity


def _git_diff_identity(root: Path, base_commit: str, head_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            raise WorkflowError(
                f"git recovery-diff command failed in {root}: {' '.join(arguments)}: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        return completed.stdout

    name_status = run(
        "diff", "--name-status", "--no-renames", base_commit, head_commit, "--"
    )
    binary_diff = run("diff", "--binary", base_commit, head_commit, "--")
    changed_paths = []
    for raw in name_status.decode("utf-8", errors="strict").splitlines():
        fields = raw.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise WorkflowError(f"malformed recovery git name-status row: {raw!r}")
        changed_paths.append({"status": fields[0], "path": fields[1]})
    return {
        "base_commit": base_commit,
        "head_commit": head_commit,
        "name_status_sha256": hashlib.sha256(name_status).hexdigest(),
        "binary_diff_sha256": hashlib.sha256(binary_diff).hexdigest(),
        "changed_paths": changed_paths,
    }
